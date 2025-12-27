# ---------------------------------------------------------------------
# Borg Organism - Full Source (PART 1/6)
# Imports, config, persistence, logger, SmartGameDetector (predictive)
# ---------------------------------------------------------------------

import os
import sys
import time
import json
import math
import threading
import traceback
import subprocess
from collections import defaultdict, Counter

# Optional runtime libs
try:
    import psutil
except Exception:
    psutil = None

# Optional Windows helpers
try:
    import win32gui
    import win32process
except Exception:
    win32gui = None
    win32process = None

# Optional NVIDIA NVML for GPU heuristics
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ---------------------------------------------------------------------
# Basic constants and file paths
# ---------------------------------------------------------------------
NODE_ID = os.environ.get("BORG_NODE_ID", "BORG-NODE-01")
CLUSTER_ID = os.environ.get("BORG_CLUSTER_ID", "BORG-CLUSTER-ALPHA")
CONFIG_FILE = os.environ.get("BORG_CONFIG_FILE", "borg_config.json")
CLUSTER_FILE = os.environ.get("BORG_CLUSTER_FILE", "borg_cluster.json")
GAME_PROFILES_FILE = os.environ.get("BORG_GAME_PROFILES_FILE", "borg_game_profiles.json")
BRAIN_LOG_FILE = os.environ.get("BORG_BRAIN_LOG_FILE", "borg_brain.log")

# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# Load config and state
config = load_json(CONFIG_FILE, {"loop_interval_seconds": 5})
cluster_state = load_json(CLUSTER_FILE, {})
game_profiles = load_json(GAME_PROFILES_FILE, {})

def persist_cluster_state():
    save_json(CLUSTER_FILE, cluster_state)

def persist_game_profiles():
    save_json(GAME_PROFILES_FILE, game_profiles)

# ---------------------------------------------------------------------
# Simple thread-safe logger for the brain
# ---------------------------------------------------------------------
class BrainLogger:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        try:
            with self.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass
        # Also print to stdout for immediate feedback
        try:
            print(line)
        except Exception:
            pass

brain_log = BrainLogger(BRAIN_LOG_FILE)

# ---------------------------------------------------------------------
# SmartGameDetector - Final Predictive Upgrade (drop-in replacement)
# ---------------------------------------------------------------------
class SmartGameDetector:
    """
    Final predictive SmartGameDetector:
    - Multi-heuristic detection (exe, cmdline, parent, foreground)
    - Fuzzy matching (Levenshtein similarity)
    - Bayesian posterior combining decayed history prior and time-of-day/minute-of-day/day-of-week likelihoods
    - Laplace smoothing for bins
    - Minute-of-day bins for fine-grained prediction
    - Confidence calibration and thresholds
    - FPS probing (hook, window title, NVML heuristic)
    - Persistence to borg_game_history.json
    - Diagnostics API
    """

    HISTORY_FILE = "borg_game_history.json"
    DEFAULT_EXES = ["back4blood.exe", "valorant.exe", "csgo.exe", "dota2.exe", "fortnite.exe"]

    def __init__(self, config, cluster_state, logger=None):
        self.config = config or {}
        self.cluster_state = cluster_state
        self.logger = logger or brain_log
        self.current = None
        self.last_telemetry = {}
        self.targets = self._load_targets()
        self.history = self._load_history()
        # decay half life seconds for history weighting
        self.decay_half_life = float(self.config.get("history_half_life_seconds", 24 * 3600))
        self.grace_seconds = int(self.config.get("game_detection_grace_seconds", 10))
        self.prediction_time_window = int(self.config.get("prediction_time_window_seconds", 7 * 24 * 3600))
        self.min_score_threshold = float(self.config.get("prediction_min_score", 0.02))
        self.fuzzy_threshold = float(self.config.get("fuzzy_similarity_threshold", 0.70))
        self.confidence_threshold = float(self.config.get("detection_confidence_threshold", 0.6))
        # weights
        self.weights = {
            "name_match": float(self.config.get("w_name", 5.0)),
            "path_match": float(self.config.get("w_path", 3.0)),
            "cmd_match": float(self.config.get("w_cmd", 2.5)),
            "parent_launcher": float(self.config.get("w_parent", 1.0)),
            "foreground": float(self.config.get("w_foreground", 2.0)),
            "recency": float(self.config.get("w_recency", 1.0)),
            "fuzzy_bonus": float(self.config.get("w_fuzzy", 1.2)),
            "gpu_fps_bonus": float(self.config.get("w_gpu_fps", 1.0)),
        }
        self.fps_hooks = self.config.get("fps_hooks", {})  # e.g., {"game.exe": "C:\\tools\\read_fps.exe --pid {pid}"}
        self._last_nvml_query = 0
        self._last_gpu_util = None

    # -------------------------
    # Persistence and helpers
    # -------------------------
    def _load_targets(self):
        raw = self.config.get("game_executables", None)
        if raw and isinstance(raw, list) and raw:
            return [self._norm(x) for x in raw]
        return [self._norm(x) for x in self.DEFAULT_EXES]

    def _now_ts(self):
        return time.time()

    def _now_utc_iso(self):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _norm(self, s):
        if not s:
            return ""
        return s.strip().lower()

    def _load_history(self):
        try:
            with open(self.HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # structure: sessions: {game_id: {count:int, starts:[ts,...]}}, hour_bins, dow_bins, minute_bins
            return {"sessions": {}, "hour_bins": {}, "dow_bins": {}, "minute_bins": {}, "last_updated": self._now_ts()}

    def _save_history(self):
        try:
            self.history["last_updated"] = self._now_ts()
            with open(self.HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def _decay_weight(self, ts):
        try:
            age = max(0.0, self._now_ts() - float(ts))
            if self.decay_half_life <= 0:
                return 1.0
            return math.pow(2.0, -age / self.decay_half_life)
        except Exception:
            return 0.0

    # -------------------------
    # Levenshtein similarity
    # -------------------------
    def _levenshtein(self, a, b):
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                add = prev[j] + 1
                delete = cur[j - 1] + 1
                change = prev[j - 1] + (0 if ca == cb else 1)
                cur[j] = min(add, delete, change)
            prev = cur
        return prev[-1]

    def _similarity(self, a, b):
        a = (a or "").lower()
        b = (b or "").lower()
        if not a or not b:
            return 0.0
        dist = self._levenshtein(a, b)
        maxlen = max(len(a), len(b))
        if maxlen == 0:
            return 1.0
        return 1.0 - (dist / maxlen)

    # -------------------------
    # Heuristics and matching
    # -------------------------
    def _match_exact_or_fuzzy(self, token, target):
        token = (token or "").lower()
        target = (target or "").lower()
        if not token or not target:
            return False, 0.0
        if target in token or token in target:
            return True, 1.0
        sim = self._similarity(token, target)
        if sim >= self.fuzzy_threshold:
            return True, sim
        return False, sim

    def _foreground_pid(self):
        try:
            if win32gui and win32process:
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    return int(pid)
        except Exception:
            pass
        return None

    # -------------------------
    # NVML GPU helper
    # -------------------------
    def _query_gpu_util(self):
        now = self._now_ts()
        if not NVML_AVAILABLE:
            return None
        if now - self._last_nvml_query < 1.0 and self._last_gpu_util is not None:
            return self._last_gpu_util
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            self._last_nvml_query = now
            self._last_gpu_util = gpu_util
            return gpu_util
        except Exception:
            return None

    # -------------------------
    # FPS probe strategies
    # -------------------------
    def _probe_fps_via_hook(self, game_exe, pid):
        try:
            cmd_template = self.fps_hooks.get(game_exe.lower())
            if not cmd_template:
                return None
            cmd = cmd_template.format(pid=pid)
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=1)
            s = out.decode("utf-8", errors="ignore").strip()
            return float(s)
        except Exception:
            return None

    def _probe_fps_from_window_title(self, pid):
        try:
            if not win32gui or not win32process:
                return None
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            _, fgpid = win32process.GetWindowThreadProcessId(hwnd)
            if fgpid != pid:
                return None
            title = win32gui.GetWindowText(hwnd) or ""
            import re
            m = re.search(r"(\d{2,3})\s*fps", title, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            m2 = re.search(r"fps[:\s]*([0-9]{1,3})", title, flags=re.IGNORECASE)
            if m2:
                return float(m2.group(1))
        except Exception:
            pass
        return None

    def _probe_fps_via_gpu(self, pid):
        try:
            gpu_util = self._query_gpu_util()
            if gpu_util is None:
                return None
            est = max(5.0, (gpu_util / 100.0) * 200.0)
            return est
        except Exception:
            return None

    def probe_fps(self, game_exe, pid):
        fps = None
        try:
            fps = self._probe_fps_via_hook(game_exe, pid)
            if fps:
                return fps
            fps = self._probe_fps_from_window_title(pid)
            if fps:
                return fps
            fps = self._probe_fps_via_gpu(pid)
            return fps
        except Exception:
            return None

    # -------------------------
    # Process scanning
    # -------------------------
    def _scan_processes(self):
        if psutil is None:
            return []

        candidates = []
        try:
            for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "ppid", "create_time"]):
                try:
                    info = proc.info
                    pid = info.get("pid")
                    name = (info.get("name") or "").strip()
                    exe = (info.get("exe") or "") or ""
                    cmd = " ".join(info.get("cmdline") or [])
                    ppid = info.get("ppid")
                    start = info.get("create_time") or 0.0

                    if pid in (0, 4):
                        continue

                    score = 0.0
                    best_name_sim = 0.0

                    for t in self.targets:
                        matched, sim = self._match_exact_or_fuzzy(name, t)
                        if matched:
                            score += self.weights["name_match"] * (1.0 if sim >= 0.99 else sim)
                            best_name_sim = max(best_name_sim, sim)
                        if t in (exe or "").lower():
                            score += self.weights["path_match"]
                        if t in (cmd or "").lower():
                            score += self.weights["cmd_match"]
                        if sim >= self.fuzzy_threshold and sim < 0.99:
                            score += self.weights["fuzzy_bonus"] * sim

                    parent_name = ""
                    try:
                        p = psutil.Process(ppid) if ppid else None
                        parent_name = (p.name() if p else "") or ""
                    except Exception:
                        parent_name = ""
                    parent_name = parent_name.lower()
                    if any(x in parent_name for x in ("steam", "epic", "riot", "launcher", "gameoverlayrenderer")):
                        score += self.weights["parent_launcher"]

                    fg = self._foreground_pid()
                    if fg and fg == pid:
                        score += self.weights["foreground"]

                    age = max(0.0, self._now_ts() - float(start))
                    if age < 60:
                        score += self.weights["recency"] * 1.5
                    elif age < 300:
                        score += self.weights["recency"] * 0.5

                    try:
                        gpu_est = self._probe_fps_via_gpu(pid)
                        if gpu_est:
                            score += (self.weights["gpu_fps_bonus"] * min(1.0, gpu_est / 120.0))
                    except Exception:
                        pass

                    if score > 0.0:
                        candidates.append({
                            "pid": pid,
                            "name": name,
                            "exe": exe,
                            "cmdline": cmd,
                            "ppid": ppid,
                            "start_time": start,
                            "score": score,
                            "best_name_sim": best_name_sim,
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            self.logger.log("[SmartGameDetector] process scan failed: " + traceback.format_exc())

        candidates.sort(key=lambda x: (x["score"], -x.get("start_time", 0)), reverse=True)
        return candidates

    # -------------------------
    # History and Bayesian predictor (minute bins + Laplace smoothing)
    # -------------------------
    def _record_session_start(self, game_id):
        now = self._now_ts()
        s = self.history.setdefault("sessions", {})
        entry = s.get(game_id, {"count": 0, "starts": []})
        entry["count"] = entry.get("count", 0) + 1
        entry["starts"] = entry.get("starts", []) + [now]
        if len(entry["starts"]) > 2000:
            entry["starts"] = entry["starts"][-2000:]
        s[game_id] = entry

        hour = time.localtime(now).tm_hour
        dow = time.localtime(now).tm_wday
        minute = time.localtime(now).tm_hour * 60 + time.localtime(now).tm_min

        hb = self.history.setdefault("hour_bins", {})
        db = self.history.setdefault("dow_bins", {})
        mb = self.history.setdefault("minute_bins", {})

        hb.setdefault(game_id, {}).setdefault(str(hour), 0)
        hb[game_id][str(hour)] = hb[game_id].get(str(hour), 0) + 1

        db.setdefault(game_id, {}).setdefault(str(dow), 0)
        db[game_id][str(dow)] = db[game_id].get(str(dow), 0) + 1

        mb.setdefault(game_id, {}).setdefault(str(minute), 0)
        mb[game_id][str(minute)] = mb[game_id].get(str(minute), 0) + 1

        self._save_history()

    def _compute_prior(self):
        now = self._now_ts()
        scores = {}
        for gid, info in self.history.get("sessions", {}).items():
            total = 0.0
            for ts in info.get("starts", []):
                if now - ts > self.prediction_time_window:
                    continue
                total += self._decay_weight(ts)
            if total > 0:
                scores[gid] = total
        total = sum(scores.values())
        if total <= 0:
            return {}
        return {gid: (v / total) for gid, v in scores.items()}

    def _time_likelihood(self, gid):
        # combine minute bin, hour bin, dow bin with Laplace smoothing
        now = self._now_ts()
        hour = time.localtime(now).tm_hour
        dow = time.localtime(now).tm_wday
        minute = time.localtime(now).tm_hour * 60 + time.localtime(now).tm_min

        mb = self.history.get("minute_bins", {}).get(gid, {})
        hb = self.history.get("hour_bins", {}).get(gid, {})
        db = self.history.get("dow_bins", {}).get(gid, {})

        # Laplace smoothing constants
        alpha = 1.0
        # minute likelihood
        minute_count = mb.get(str(minute), 0)
        minute_total = sum(mb.values()) if mb else 0
        minute_l = (minute_count + alpha) / (minute_total + alpha * 1440) if minute_total >= 0 else 1.0

        # hour likelihood
        hour_count = hb.get(str(hour), 0)
        hour_total = sum(hb.values()) if hb else 0
        hour_l = (hour_count + alpha) / (hour_total + alpha * 24) if hour_total >= 0 else 1.0

        # dow likelihood
        dow_count = db.get(str(dow), 0)
        dow_total = sum(db.values()) if db else 0
        dow_l = (dow_count + alpha) / (dow_total + alpha * 7) if dow_total >= 0 else 1.0

        # combine multiplicatively but clamp
        likelihood = minute_l * (0.6 * hour_l + 0.4 * dow_l)
        return likelihood

    def get_prediction(self, top_n=5):
        prior = self._compute_prior()
        if not prior:
            return []
        posterior = {}
        for gid, p in prior.items():
            likelihood = self._time_likelihood(gid)
            posterior[gid] = p * likelihood
        total = sum(posterior.values())
        if total <= 0:
            return []
        normalized = sorted([(g, s / total) for g, s in posterior.items()], key=lambda x: x[1], reverse=True)
        filtered = [(g, s) for g, s in normalized if s >= self.min_score_threshold]
        return filtered[:top_n]

    # -------------------------
    # Public API: update loop
    # -------------------------
    def update(self):
        candidates = self._scan_processes()
        now_iso = self._now_utc_iso()
        chosen = None
        chosen_score = 0.0

        if candidates:
            top = candidates[0]
            chosen = top
            chosen_score = top.get("score", 0.0)

            det_norm = 1.0 - math.exp(-chosen_score / 6.0)
            game_name = (top.get("name") or os.path.basename(top.get("exe") or "") or f"pid-{top.get('pid')}")
            prior_map = self._compute_prior()
            prior = prior_map.get(game_name, 0.0)
            time_like = self._time_likelihood(game_name)
            # combine into confidence: weighted detection + prior + time likelihood
            confidence = (0.55 * det_norm) + (0.30 * prior) + (0.15 * (time_like / (time_like + 1.0)))
            if top.get("best_name_sim", 0.0) < self.fuzzy_threshold:
                confidence *= 0.85
            confidence = max(0.0, min(1.0, confidence))
            is_high_confidence = confidence >= self.confidence_threshold

            pid = top["pid"]
            game_id = game_name

            if self.current and self.current.get("pid") == pid:
                self.current["last_seen_utc"] = now_iso
                self.current["confidence"] = confidence
                self.current["high_confidence"] = is_high_confidence
            else:
                self.current = {
                    "game_id": game_id,
                    "pid": pid,
                    "exe": top.get("exe") or "",
                    "start_time_utc": now_iso,
                    "last_seen_utc": now_iso,
                    "confidence": confidence,
                    "high_confidence": is_high_confidence,
                }
                self.last_telemetry = {}
                self.logger.log(f"[SmartGameDetector] Detected game start: {game_id} (pid={pid}) conf={confidence:.2f} high={is_high_confidence}")
                try:
                    self._record_session_start(game_id)
                except Exception:
                    pass

            # FPS probe
            try:
                fps = self.probe_fps(game_id, pid)
                if fps:
                    self.last_telemetry["fps_est"] = fps
            except Exception:
                pass

            # publish to cluster_state
            gs = self.cluster_state.setdefault("game_session", {})
            gs.update({
                "game_id": self.current.get("game_id"),
                "pid": self.current.get("pid"),
                "exe": self.current.get("exe"),
                "start_time": self.current.get("start_time_utc"),
                "last_seen_utc": self.current.get("last_seen_utc"),
                "last_telemetry": self.last_telemetry,
                "drive_strategy": gs.get("drive_strategy", {}),
                "detection_score": chosen_score,
                "detection_confidence": self.current.get("confidence"),
                "detection_high_confidence": self.current.get("high_confidence"),
                "prediction": self.get_prediction(5),
            })
        else:
            if self.current:
                try:
                    last_seen = self.current.get("last_seen_utc")
                    last_seen_ts = time.mktime(time.strptime(last_seen, "%Y-%m-%dT%H:%M:%SZ"))
                except Exception:
                    last_seen_ts = self._now_ts()
                if self._now_ts() - last_seen_ts > self.grace_seconds:
                    self.logger.log(f"[SmartGameDetector] Game stopped: {self.current.get('game_id')} (pid={self.current.get('pid')})")
                    self.current = None
                    if "game_session" in self.cluster_state:
                        self.cluster_state.pop("game_session", None)
            else:
                if "game_session" in self.cluster_state:
                    self.cluster_state.pop("game_session", None)

    def set_last_telemetry(self, telemetry):
        if not self.current:
            return
        self.last_telemetry = telemetry or {}
        gs = self.cluster_state.setdefault("game_session", {})
        gs["last_telemetry"] = self.last_telemetry
        gs["last_seen_utc"] = self._now_utc_iso()

    def get_active_game(self):
        return self.current

    def get_confidence(self):
        if not self.current:
            return 0.0
        return float(self.current.get("confidence", 0.0))

    def get_history_summary(self):
        out = {}
        for gid, info in self.history.get("sessions", {}).items():
            out[gid] = {"count": info.get("count", 0), "recent": len(info.get("starts", []))}
        return out

    def get_diagnostics(self):
        return {
            "targets": self.targets,
            "history_summary": self.get_history_summary(),
            "current": self.current,
            "nvml": NVML_AVAILABLE,
            "psutil": psutil is not None,
        }

# Instantiate detector (will be used by QueenService)
smart_game_detector = SmartGameDetector(config, cluster_state, brain_log)

# --- Core subsystem stubs (replace with real implementations as needed) ---

class DriveManager:
    """
    DriveManager stub:
    - scan_drives() should populate cluster_state['drives'] with drive metadata:
      { drive_id: {"path": "...", "free_gb": 123, "total_gb": 512, "status":"ok"} }
    """
    def __init__(self):
        self.last_scan = 0

    def scan_drives(self):
        # Best-effort local drives listing; replace with your production logic
        try:
            drives = {}
            # simple local root probe (cross-platform)
            roots = []
            if os.name == "nt":
                # Windows: check common drive letters
                for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                    path = f"{letter}:/"
                    if os.path.exists(path):
                        roots.append(path)
            else:
                roots = ["/", "/mnt", "/media"]
            idx = 0
            for r in roots:
                try:
                    if not os.path.exists(r):
                        continue
                    stat = os.statvfs(r)
                    total = (stat.f_blocks * stat.f_frsize) / (1024**3)
                    free = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    drives[f"local-{idx}"] = {"path": r, "free_gb": round(free, 2), "total_gb": round(total, 2), "status": "ok"}
                    idx += 1
                except Exception:
                    continue
            cluster_state["drives"] = drives
            self.last_scan = time.time()
        except Exception:
            brain_log.log("[DriveManager] scan_drives failed:\n" + traceback.format_exc())

drive_manager = DriveManager()

class ReasoningEngine:
    """
    ReasoningEngine stub:
    - decide_cache_drives_for_network(drives) returns a dict describing which drives to cache.
    """
    def decide_cache_drives_for_network(self, drives):
        try:
            # simple heuristic: pick the largest free drive
            best = None
            best_free = -1
            for did, info in (drives or {}).items():
                try:
                    free = float(info.get("free_gb", 0))
                    if free > best_free:
                        best_free = free
                        best = did
                except Exception:
                    continue
            return {"cache_drives": [best] if best else [], "reason": "largest_free"}
        except Exception:
            return {"cache_drives": [], "reason": "error"}

reasoning_engine = ReasoningEngine()

class GameTelemetryCollector:
    """
    GameTelemetryCollector stub:
    - sample() should collect per-process CPU/memory for active game and call smart_game_detector.set_last_telemetry()
    """
    def __init__(self):
        self.last_sample = 0

    def sample(self):
        try:
            gs = cluster_state.get("game_session")
            if not gs:
                return
            pid = gs.get("pid")
            if not pid or psutil is None:
                return
            try:
                p = psutil.Process(int(pid))
                cpu = p.cpu_percent(interval=0.1)
                mem = p.memory_info().rss / (1024**2)  # MB
                telemetry = {"cpu_percent": round(cpu, 2), "mem_mb": round(mem, 2)}
                # attach fps if detector probed it
                # let detector attach fps via set_last_telemetry if available
                try:
                    smart_game_detector.set_last_telemetry(telemetry)
                except Exception:
                    cluster_state.setdefault("game_session", {}).setdefault("last_telemetry", telemetry)
                self.last_sample = time.time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
        except Exception:
            brain_log.log("[GameTelemetryCollector] sample failed:\n" + traceback.format_exc())

game_telemetry = GameTelemetryCollector()

class SystemMonitor:
    """
    SystemMonitor stub:
    - sample() should update cluster_state['system_monitor'] with cpu and memory percentages
    """
    def __init__(self):
        self.last = {}

    def sample(self):
        try:
            if psutil is None:
                cluster_state["system_monitor"] = {"cpu_percent": 0.0, "mem_percent": 0.0}
                return
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            cluster_state["system_monitor"] = {"cpu_percent": round(cpu, 1), "mem_percent": round(mem, 1)}
            self.last = cluster_state["system_monitor"]
        except Exception:
            brain_log.log("[SystemMonitor] sample failed:\n" + traceback.format_exc())

system_monitor = SystemMonitor()

class UserActivityMonitor:
    """
    UserActivityMonitor stub:
    - sample() should update cluster_state['user_activity'] with last input timestamp or idle seconds
    """
    def __init__(self):
        self.last_input = time.time()

    def sample(self):
        try:
            # best-effort: approximate user active if CPU > small threshold or foreground window exists
            active = True
            if psutil:
                cpu = psutil.cpu_percent(interval=0.0)
                active = cpu > 1.0
            cluster_state["user_activity"] = {"active": active, "last_seen": time.time()}
        except Exception:
            brain_log.log("[UserActivityMonitor] sample failed:\n" + traceback.format_exc())

user_activity_monitor = UserActivityMonitor()

class PriorityPolicyManager:
    """
    PriorityPolicyManager stub:
    - update_posture() should set cluster_state['posture'] based on load and policies
    """
    def update_posture(self):
        try:
            sysmon = cluster_state.get("system_monitor", {})
            cpu = sysmon.get("cpu_percent", 0.0)
            mem = sysmon.get("mem_percent", 0.0)
            if cpu > 85 or mem > 90:
                posture = "HIGH_LOAD"
            elif cpu > 50 or mem > 70:
                posture = "ELEVATED"
            else:
                posture = "CALM"
            cluster_state["posture"] = posture
        except Exception:
            brain_log.log("[PriorityPolicyManager] update_posture failed:\n" + traceback.format_exc())

priority_policy = PriorityPolicyManager()

class CoPilotOptimizationCortex:
    """
    CoPilotOptimizationCortex stub:
    - maybe_run() can perform opportunistic tasks; keep lightweight
    """
    def maybe_run(self):
        try:
            # placeholder: occasionally log that cortex ran
            if int(time.time()) % 60 == 0:
                brain_log.log("[CoPilotOptimizationCortex] heartbeat")
        except Exception:
            brain_log.log("[CoPilotOptimizationCortex] maybe_run failed:\n" + traceback.format_exc())

copilot_cortex = CoPilotOptimizationCortex()

class TaskScheduler:
    """
    TaskScheduler stub:
    - tick() should run scheduled tasks; here we keep it minimal
    """
    def tick(self):
        try:
            # placeholder: no-op
            return
        except Exception:
            brain_log.log("[TaskScheduler] tick failed:\n" + traceback.format_exc())

task_scheduler = TaskScheduler()

class ClusterCoordinator:
    """
    ClusterCoordinator stub:
    - elect_queen() placeholder for cluster leadership logic
    """
    def elect_queen(self):
        try:
            # simple local leader: node with smallest NODE_ID string (single-node default)
            cluster_state.setdefault("cluster_meta", {})["queen"] = NODE_ID
        except Exception:
            brain_log.log("[ClusterCoordinator] elect_queen failed:\n" + traceback.format_exc())

cluster_coordinator = ClusterCoordinator()

# ---------------------------------------------------------------------
# QueenService - main background loop (heartbeat)
# ---------------------------------------------------------------------
class QueenService(threading.Thread):
    def __init__(self, config, cluster_state, logger):
        super().__init__(daemon=True)
        self.config = config
        self.cluster_state = cluster_state
        self.logger = logger
        self.running = False

    def run(self):
        self.running = True
        self.logger.log(f"[SYSTEM] Borg organism starting on node {NODE_ID}")
        interval = float(self.config.get("loop_interval_seconds", 5))
        last_cache_decision = 0

        while self.running:
            loop_start = time.time()
            try:
                # Core system updates
                drive_manager.scan_drives()
                system_monitor.sample()
                user_activity_monitor.sample()
                smart_game_detector.update()
                game_telemetry.sample()
                priority_policy.update_posture()
                cluster_coordinator.elect_queen()

                # Cache drive decision every 60 seconds
                now = time.time()
                if now - last_cache_decision > 60:
                    drives_info = self.cluster_state.get("drives", {})
                    cd = reasoning_engine.decide_cache_drives_for_network(drives_info)
                    self.cluster_state["cache_drives"] = cd
                    last_cache_decision = now

                # Copilot + tasks
                copilot_cortex.maybe_run()
                task_scheduler.tick()

                # Persist state
                persist_cluster_state()
                persist_game_profiles()

            except Exception:
                self.logger.log("[QUEEN] Exception:\n" + traceback.format_exc())

            elapsed = time.time() - loop_start
            wait = max(0.0, interval - elapsed)
            time.sleep(wait)

    def stop(self):
        self.running = False
        self.logger.log("[QUEEN] Stopping Borg Queen service.")

# Instantiate and prepare QueenService (do not start here; main() will start it)
queen_service = QueenService(config, cluster_state, brain_log)


import tkinter as tk
from tkinter import ttk, filedialog

class BorgGUI:
    def __init__(self, root, cluster_state):
        self.root = root
        self.cluster_state = cluster_state

        root.title(f"Borg Cluster Brain - {NODE_ID}")
        root.minsize(900, 550)
        root.configure(bg="#111111")

        self.style = ttk.Style()
        self._configure_style()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_game = ttk.Frame(self.notebook)
        self.tab_drives = ttk.Frame(self.notebook)
        self.tab_tasks = ttk.Frame(self.notebook)
        self.tab_logs = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text="Overview")
        self.notebook.add(self.tab_game, text="Game")
        self.notebook.add(self.tab_drives, text="Drives")
        self.notebook.add(self.tab_tasks, text="Tasks / Copilot")
        self.notebook.add(self.tab_logs, text="Logs")

        self._build_tab_overview()
        self._build_tab_game()
        self._build_tab_drives()
        self._build_tab_tasks()
        self._build_tab_logs()

        self.gui_log_max_lines = 400
        self._schedule_update()

    def _configure_style(self):
        bg = "#111111"
        fg = "#f0f0f0"
        amber = "#ffb300"
        panel_bg = "#1b1b1b"

        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=bg, foreground=fg, fieldbackground=panel_bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", background=bg, foreground=amber,
                             font=("Segoe UI", 10, "bold"))
        self.style.configure("Panel.TLabelframe", background=panel_bg, foreground=amber,
                             borderwidth=1, relief="solid")
        self.style.configure("Panel.TLabelframe.Label", background=panel_bg, foreground=amber,
                             font=("Segoe UI", 9, "bold"))
        self.style.configure("TButton", background="#222222", foreground=fg)
        self.style.map("TButton", background=[("active", "#333333")])
        self.style.configure("Treeview",
                             background=panel_bg,
                             foreground=fg,
                             fieldbackground=panel_bg,
                             rowheight=20)
        self.style.map("Treeview", background=[("selected", "#444444")])

    # ---------------------- OVERVIEW TAB ---------------------- #

    def _build_tab_overview(self):
        f = self.tab_overview
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)

        top = ttk.Labelframe(f, text="Node / Posture / Load", style="Panel.TLabelframe")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        for c in range(4):
            top.columnconfigure(c, weight=1)

        ttk.Label(top, text="Node:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.ov_node = ttk.Label(top, style="Header.TLabel", text=NODE_ID)
        self.ov_node.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="CPU:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        self.ov_cpu = ttk.Label(top, text="0%")
        self.ov_cpu.grid(row=0, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="Posture:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.ov_posture = ttk.Label(top, text="CALM")
        self.ov_posture.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="MEM:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        self.ov_mem = ttk.Label(top, text="0%")
        self.ov_mem.grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="Active Game:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.ov_game = ttk.Label(top, text="(none)")
        self.ov_game.grid(row=2, column=1, columnspan=3, sticky="w", padx=4, pady=2)

        bottom_left = ttk.Labelframe(f, text="User / Maintenance", style="Panel.TLabelframe")
        bottom_right = ttk.Labelframe(f, text="Queen / Time", style="Panel.TLabelframe")
        bottom_left.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        bottom_right.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        bottom_left.columnconfigure(0, weight=1)
        bottom_right.columnconfigure(0, weight=1)

        self.text_user = tk.Text(bottom_left, height=8, wrap="word",
                                 background="#1b1b1b", foreground="#f0f0f0",
                                 font=("Consolas", 9), bd=0, relief="flat")
        self.text_user.pack(fill="both", expand=True, padx=4, pady=4)

        self.text_node = tk.Text(bottom_right, height=8, wrap="word",
                                 background="#1b1b1b", foreground="#f0f0f0",
                                 font=("Consolas", 9), bd=0, relief="flat")
        self.text_node.pack(fill="both", expand=True, padx=4, pady=4)

    # ---------------------- GAME TAB ---------------------- #

    def _build_tab_game(self):
        f = self.tab_game
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)

        top = ttk.Labelframe(f, text="Active Game Session", style="Panel.TLabelframe")
        top.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        for c in range(6):
            top.columnconfigure(c, weight=1)

        ttk.Label(top, text="Game:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.gm_name = ttk.Label(top, text="(none)")
        self.gm_name.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="PID:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        self.gm_pid = ttk.Label(top, text="-")
        self.gm_pid.grid(row=0, column=3, sticky="w", padx=4, pady=2)

        # Confidence widget
        self.gm_conf = ttk.Label(top, text="Confidence: -")
        self.gm_conf.grid(row=0, column=4, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="CPU:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.gm_cpu = ttk.Label(top, text="-")
        self.gm_cpu.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="RAM:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        self.gm_mem = ttk.Label(top, text="-")
        self.gm_mem.grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="Drive Strategy:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.gm_strategy = ttk.Label(top, text="-")
        self.gm_strategy.grid(row=2, column=1, columnspan=3, sticky="w", padx=4, pady=2)

        # Predictions widget
        self.gm_prediction = ttk.Label(top, text="Predictions: -")
        self.gm_prediction.grid(row=2, column=4, sticky="w", padx=4, pady=2)

        bottom = ttk.Labelframe(f, text="Telemetry / Explanation", style="Panel.TLabelframe")
        bottom.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(0, weight=1)

        self.text_game = tk.Text(bottom, height=10, wrap="word",
                                 background="#1b1b1b", foreground="#f0f0f0",
                                 font=("Consolas", 9), bd=0, relief="flat")
        self.text_game.pack(fill="both", expand=True, padx=4, pady=4)

    # ---------------------- DRIVES TAB ---------------------- #

    def _build_tab_drives(self):
        f = self.tab_drives
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        f.rowconfigure(1, weight=0)
        f.rowconfigure(2, weight=0)

        self.show_offline_var = tk.BooleanVar(value=False)

        self.drives_tree = ttk.Treeview(
            f,
            columns=("id", "path", "type", "status", "free", "total", "perf", "roles"),
            show="headings",
            height=8,
        )
        for col, txt, w, anchor in [
            ("id", "ID", 80, "w"),
            ("path", "Path", 200, "w"),
            ("type", "Type", 60, "center"),
            ("status", "Status", 70, "center"),
            ("free", "Free (GB)", 80, "e"),
            ("total", "Total (GB)", 80, "e"),
            ("perf", "Perf", 60, "e"),
            ("roles", "Roles", 120, "w"),
        ]:
            self.drives_tree.heading(col, text=txt)
            self.drives_tree.column(col, width=w, anchor=anchor)

        self.drives_tree.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self.cache_label = ttk.Label(f, text="Cache: -")
        self.cache_label.grid(row=1, column=0, sticky="w", padx=4, pady=(0, 4))

        bottom = ttk.Frame(f)
        bottom.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=1)
        bottom.columnconfigure(3, weight=0)

        self.entry_drive_id = ttk.Entry(bottom, width=10)
        self.entry_path = ttk.Entry(bottom)
        self.entry_role = ttk.Entry(bottom, width=12)

        ttk.Label(bottom, text="ID").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Label(bottom, text="Path").grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(bottom, text="Role(s)").grid(row=0, column=2, sticky="w", padx=2)

        self.entry_drive_id.grid(row=1, column=0, sticky="ew", padx=2)
        self.entry_path.grid(row=1, column=1, sticky="ew", padx=2)
        self.entry_role.grid(row=1, column=2, sticky="ew", padx=2)

        self.button_add_drive = ttk.Button(
            bottom,
            text="Add (Browse)",
            command=self._browse_and_add_drive,
        )
        self.button_add_drive.grid(row=1, column=3, sticky="ew", padx=2)

        self.chk_offline = ttk.Checkbutton(
            bottom,
            text="Show offline drives",
            variable=self.show_offline_var,
            command=self._refresh_drives_view,
        )
        self.chk_offline.grid(row=2, column=0, columnspan=2, sticky="w", padx=2, pady=(4, 0))

    def _browse_and_add_drive(self):
        path = filedialog.askdirectory(title="Select network or local folder")
        if not path:
            return
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, path)

        did = self.entry_drive_id.get().strip()
        if not did:
            base = os.path.basename(path.rstrip("\\/")) or path
            self.entry_drive_id.insert(0, base)

        self._add_manual_drive()

    def _add_manual_drive(self):
        did = self.entry_drive_id.get().strip()
        path = self.entry_path.get().strip()
        role_text = self.entry_role.get().strip()

        if not did or not path:
            return

        roles = [r.strip().upper() for r in role_text.split(",") if r.strip()]
        nd = {"id": did, "path": path, "role": roles, "priority": 1.0}

        nds = config.get("network_drives", [])
        nds = [d for d in nds if d.get("id") != did]
        nds.append(nd)
        config["network_drives"] = nds
        save_json(CONFIG_FILE, config)
        brain_log.log(f"[GUI] Added manual network drive {did} -> {path} roles={roles}")

    def _refresh_drives_view(self):
        for i in self.drives_tree.get_children():
            self.drives_tree.delete(i)

        cs = self.cluster_state
        show_offline = self.show_offline_var.get()
        for d in cs.get("drives", {}).values():
            reachable = d.get("status", "ok") == "ok" or d.get("reachable", True)
            if not reachable and not show_offline:
                continue
            status = "ONLINE" if reachable else "OFFLINE"
            self.drives_tree.insert(
                "",
                "end",
                values=(
                    d.get("path") or d.get("id"),
                    d.get("path"),
                    d.get("type", "LOCAL"),
                    status,
                    d.get("free_gb"),
                    d.get("total_gb"),
                    d.get("perf_score", ""),
                    ",".join(d.get("role", [])) if isinstance(d.get("role", []), list) else d.get("role", ""),
                ),
            )

        cd = cs.get("cache_drives", {})
        if isinstance(cd, dict):
            self.cache_label.config(
                text=f"Cache: {cd.get('cache_drives')} - {cd.get('reason', cd.get('explanation','-'))}"
            )
        else:
            self.cache_label.config(text=f"Cache: {cd}")

    # ---------------------- TASKS / COPILOT TAB ---------------------- #

    def _build_tab_tasks(self):
        f = self.tab_tasks
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.text_tasks = tk.Text(
            f,
            height=12,
            wrap="word",
            background="#1b1b1b",
            foreground="#f0f0f0",
            font=("Consolas", 9),
            bd=0,
            relief="flat",
        )
        self.text_tasks.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    # ---------------------- LOGS TAB ---------------------- #

    def _build_tab_logs(self):
        f = self.tab_logs
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.text_logs = tk.Text(
            f,
            height=14,
            wrap="none",
            background="#000000",
            foreground="#00ff00",
            font=("Consolas", 9),
            bd=0,
            relief="flat",
        )
        self.text_logs.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    # ---------------------- UPDATE LOOP ---------------------- #

    def _schedule_update(self):
        self._update_view()
        self.root.after(1000, self._schedule_update)

    def _update_view(self):
        cs = self.cluster_state

        # System / posture summary
        sysmon = cs.get("system_monitor", {})
        posture = cs.get("posture", "CALM")
        cpu = sysmon.get("cpu_percent", 0.0)
        mem = sysmon.get("mem_percent", 0.0)
        game = cs.get("game_session")

        self.ov_cpu.config(text=f"{cpu:.1f}%")
        self.ov_mem.config(text=f"{mem:.1f}%")
        self.ov_posture.config(text=posture)
        self.ov_node.config(text=NODE_ID)
        if game:
            self.ov_game.config(text=str(game.get("game_id")))
        else:
            self.ov_game.config(text="(none)")

        ua = cs.get("user_activity", {})
        maint = cs.get("maintenance_state", {})
        time_ref = cs.get("time_reference", {})

        user_lines = [
            f"Foreground PID: {ua.get('foreground_pid')}",
            f"Foreground Name: {ua.get('foreground_name')}",
            f"Since: {ua.get('since_utc')}",
            "",
            f"Heavy Tasks Allowed: {maint.get('heavy_tasks_allowed')}",
            f"Reason: {maint.get('reason')}",
        ]
        self.text_user.delete("1.0", tk.END)
        self.text_user.insert(tk.END, "\n".join(user_lines))

        node_lines = [
            f"Cluster: {CLUSTER_ID}",
            f"Queen: {cs.get('active_queen', NODE_ID)}",
            f"Queen Score: {cs.get('queen_score', 0.0):.2f}",
            "",
            f"UTC:   {time_ref.get('last_update_utc')}",
            f"Local: {time_ref.get('local_time_str')}  {time_ref.get('day_of_week')} {time_ref.get('year')}",
        ]
        self.text_node.delete("1.0", tk.END)
        self.text_node.insert(tk.END, "\n".join(node_lines))

        # Game tab
        self.text_game.delete("1.0", tk.END)
        if game:
            lt = game.get("last_telemetry", {})
            ds = game.get("drive_strategy", {})
            self.gm_name.config(text=str(game.get("game_id")))
            self.gm_pid.config(text=str(game.get("pid")))
            self.gm_cpu.config(text=f"{lt.get('cpu_percent', 'n/a')}")
            self.gm_mem.config(text=f"{lt.get('mem_mb', 'n/a')} MB")
            self.gm_strategy.config(text=str(ds.get("strategy", "-")))

            g_lines = [
                f"Game: {game.get('game_id')} (pid={game.get('pid')})",
                f"Started: {game.get('start_time')}",
                f"CPU: {lt.get('cpu_percent', 'n/a')}%   MEM: {lt.get('mem_mb', 'n/a')} MB",
                "",
                f"Drive Strategy: {ds.get('strategy')}",
                f"  Primary: {ds.get('primary')}",
                f"  Secondary: {ds.get('secondary')}",
                f"  Explanation: {ds.get('explanation')}",
            ]
            self.text_game.insert(tk.END, "\n".join(g_lines))

            # Confidence, predictions, FPS
            conf = game.get("detection_confidence", None)
            pred = game.get("prediction", [])
            fps = lt.get("fps_est")

            if conf is not None:
                try:
                    self.gm_conf.config(text=f"Confidence: {conf:.2f}")
                except Exception:
                    pass
            else:
                self.gm_conf.config(text="Confidence: -")

            pred_text = ", ".join([f"{p[0]}:{p[1]:.2f}" for p in pred]) if pred else "(none)"
            self.gm_prediction.config(text=f"Predictions: {pred_text}")

            if fps:
                self.text_game.insert(tk.END, f"\nEstimated FPS: {fps:.1f}")

            # highlight low confidence
            try:
                if conf is not None and conf < 0.5:
                    self.notebook.tab(self.tab_game, foreground="#ffb300")
                else:
                    self.notebook.tab(self.tab_game, foreground=None)
            except Exception:
                pass

        else:
            self.gm_name.config(text="(none)")
            self.gm_pid.config(text="-")
            self.gm_cpu.config(text="-")
            self.gm_mem.config(text="-")
            self.gm_strategy.config(text="-")
            self.gm_conf.config(text="Confidence: -")
            self.gm_prediction.config(text="Predictions: -")
            self.text_game.insert(tk.END, "No active game session.")

        # Drives tab
        self._refresh_drives_view()

        # Tasks / Copilot
        self.text_tasks.delete("1.0", tk.END)
        tasks = cs.get("tasks", [])
        self.text_tasks.insert(tk.END, "Tasks:\n")
        for t in tasks:
            self.text_tasks.insert(
                tk.END,
                f"  [{t.get('type')}] {t.get('name')} (id={t.get('id')}), "
                f"interval={t.get('interval_seconds')}s, last_run={t.get('last_run_utc')}, "
                f"enabled={t.get('enabled')}\n",
            )
        self.text_tasks.insert(tk.END, "\nCo-Pilot:\n")
        cps = cs.get("copilot_optimization_state", {})
        self.text_tasks.insert(tk.END, f"  Last analysis: {cps.get('last_analysis_utc')}\n")
        self.text_tasks.insert(tk.END, f"  Summary: {cps.get('last_summary')}\n")
        self.text_tasks.insert(tk.END, f"  Recommendations: {cps.get('last_recommendations')}\n")

        # Logs tab
        try:
            if os.path.exists(BRAIN_LOG_FILE):
                with open(BRAIN_LOG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-self.gui_log_max_lines:]
            else:
                lines = []
        except Exception:
            lines = []
        self.text_logs.delete("1.0", tk.END)
        if lines:
            self.text_logs.insert(tk.END, "".join(lines))
        else:
            self.text_logs.insert(tk.END, "(no log data yet)")

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    # Start queen service
    try:
        queen_service.start()
    except Exception:
        brain_log.log("[MAIN] Queen service failed to start:\n" + traceback.format_exc())

    # Start GUI
    root = tk.Tk()
    gui = BorgGUI(root, cluster_state)
    try:
        root.mainloop()
    finally:
        try:
            queen_service.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()

import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# -------------------------
# Config reload helper
# -------------------------
def reload_config():
    """
    Reload config and apply to running components where possible.
    Call this from a signal handler or admin action.
    """
    global config
    try:
        new_conf = load_json(CONFIG_FILE, {})
        if not isinstance(new_conf, dict):
            brain_log.log("[CONFIG] reload failed: invalid format")
            return False
        config.update(new_conf)
        brain_log.log("[CONFIG] reloaded configuration")
        # propagate to subsystems that read config at runtime
        smart_game_detector.config = config
        smart_game_detector.targets = smart_game_detector._load_targets()
        # update queen_service interval if present
        try:
            queen_service.config = config
        except Exception:
            pass
        return True
    except Exception:
        brain_log.log("[CONFIG] reload exception:\n" + traceback.format_exc())
        return False

# -------------------------
# Simple HTTP status server
# -------------------------
class _StatusHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, code=200):
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # / -> brief status
        if self.path in ("/", "/status"):
            try:
                snapshot = {
                    "node_id": NODE_ID,
                    "cluster_id": CLUSTER_ID,
                    "posture": cluster_state.get("posture"),
                    "system_monitor": cluster_state.get("system_monitor"),
                    "game_session": cluster_state.get("game_session"),
                    "drives": cluster_state.get("drives"),
                    "queen": cluster_state.get("active_queen", NODE_ID),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self._send_json(snapshot)
            except Exception:
                self._send_json({"error": "failed to build status"}, code=500)
            return

        # /diagnostics -> more verbose
        if self.path.startswith("/diagnostics"):
            try:
                diag = {
                    "config": {k: config.get(k) for k in ("loop_interval_seconds", "game_executables")},
                    "detector": smart_game_detector.get_diagnostics() if hasattr(smart_game_detector, "get_diagnostics") else {},
                    "history_summary": smart_game_detector.get_history_summary() if hasattr(smart_game_detector, "get_history_summary") else {},
                    "cluster_state_snapshot": cluster_state,
                }
                self._send_json(diag)
            except Exception:
                self._send_json({"error": "failed to build diagnostics"}, code=500)
            return

        # unknown path
        self._send_json({"error": "not found"}, code=404)

    def log_message(self, format, *args):
        # suppress default logging to stderr; route to brain_log
        try:
            brain_log.log("[STATUS] " + (format % args))
        except Exception:
            pass

class StatusHTTPServer(threading.Thread):
    def __init__(self, host="127.0.0.1", port=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = int(port or config.get("status_http_port", 8765))
        self._httpd = None
        self._running = False

    def run(self):
        try:
            server_address = (self.host, self.port)
            self._httpd = HTTPServer(server_address, _StatusHandler)
            self._running = True
            brain_log.log(f"[STATUS] HTTP status server listening on {self.host}:{self.port}")
            self._httpd.serve_forever()
        except Exception:
            brain_log.log("[STATUS] HTTP server failed:\n" + traceback.format_exc())
        finally:
            self._running = False

    def stop(self):
        try:
            if self._httpd:
                self._httpd.shutdown()
                self._httpd.server_close()
                brain_log.log("[STATUS] HTTP status server stopped")
        except Exception:
            brain_log.log("[STATUS] HTTP server stop exception:\n" + traceback.format_exc())

# Instantiate status server (not started automatically)
status_server = StatusHTTPServer(host="127.0.0.1", port=config.get("status_http_port", 8765))

# -------------------------
# Graceful shutdown handling
# -------------------------
_shutdown_lock = threading.Lock()
_shutdown_in_progress = False

def _graceful_shutdown(signum=None, frame=None):
    global _shutdown_in_progress
    with _shutdown_lock:
        if _shutdown_in_progress:
            return
        _shutdown_in_progress = True
    brain_log.log(f"[SYSTEM] Received shutdown signal ({signum}). Shutting down gracefully...")
    try:
        # stop status server
        try:
            status_server.stop()
        except Exception:
            pass
        # stop queen service
        try:
            queen_service.stop()
        except Exception:
            pass
        # persist final state
        try:
            persist_cluster_state()
            persist_game_profiles()
        except Exception:
            pass
        # allow a short grace period for threads to exit
        time.sleep(0.5)
    except Exception:
        brain_log.log("[SYSTEM] Error during shutdown:\n" + traceback.format_exc())
    finally:
        brain_log.log("[SYSTEM] Shutdown complete.")
        # If running under main thread, exit process
        try:
            os._exit(0)
        except Exception:
            pass

# Register signals for graceful shutdown
try:
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
except Exception:
    # Some platforms (e.g., Windows in certain contexts) may behave differently
    pass

# -------------------------
# Admin helpers
# -------------------------
def start_status_server():
    try:
        if not status_server.is_alive():
            status_server.start()
    except Exception:
        brain_log.log("[STATUS] start failed:\n" + traceback.format_exc())

def stop_status_server():
    try:
        status_server.stop()
    except Exception:
        brain_log.log("[STATUS] stop failed:\n" + traceback.format_exc())

def dump_state_to_file(path="borg_cluster_snapshot.json"):
    try:
        save_json(path, cluster_state)
        brain_log.log(f"[DUMP] Cluster state written to {path}")
    except Exception:
        brain_log.log("[DUMP] failed:\n" + traceback.format_exc())

import socket
import platform
import shutil

# -------------------------
# Enhanced DriveManager
# -------------------------
class DriveManager:
    """
    Enhanced DriveManager:
    - Scans local mounts and configured network drives
    - Measures free/total space, basic latency check for network paths
    - Computes a simple perf_score based on type, free space, and latency
    - Populates cluster_state['drives'] with consistent metadata
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.lock = threading.Lock()
        self.last_scan = 0

    def _list_local_mounts(self):
        mounts = []
        try:
            if os.name == "nt":
                # Windows: check drive letters
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    path = f"{letter}:/"
                    if os.path.exists(path):
                        mounts.append({"id": f"local-{letter}", "path": path, "type": "LOCAL"})
            else:
                # Unix-like: parse /proc/mounts or use shutil.disk_usage roots
                # We'll probe common roots and /mnt /media
                roots = ["/", "/mnt", "/media"]
                for r in roots:
                    if os.path.exists(r):
                        mounts.append({"id": f"local-{r.strip('/').replace('/','_') or 'root'}", "path": r, "type": "LOCAL"})
        except Exception:
            brain_log.log("[DriveManager] _list_local_mounts failed:\n" + traceback.format_exc())
        return mounts

    def _list_configured_network(self):
        nds = []
        try:
            for nd in self.config.get("network_drives", []):
                did = nd.get("id") or nd.get("path")
                path = nd.get("path")
                nds.append({"id": did, "path": path, "type": nd.get("type", "NET"), "role": nd.get("role", [])})
        except Exception:
            brain_log.log("[DriveManager] _list_configured_network failed:\n" + traceback.format_exc())
        return nds

    def _probe_path(self, path):
        """
        Return (reachable, free_gb, total_gb, latency_ms)
        latency_ms is a best-effort: for network UNC paths we try a socket connect to host:445 or host resolved.
        """
        reachable = False
        free_gb = None
        total_gb = None
        latency_ms = None
        try:
            if os.path.exists(path):
                reachable = True
                try:
                    du = shutil.disk_usage(path)
                    total_gb = round(du.total / (1024 ** 3), 2)
                    free_gb = round(du.free / (1024 ** 3), 2)
                except Exception:
                    total_gb = None
                    free_gb = None
            # network latency probe for UNC or network-like paths
            if (not reachable) and (path and (path.startswith("\\\\") or path.startswith("//") or "://" in path)):
                # extract host
                host = None
                try:
                    p = path.replace("\\", "/")
                    if p.startswith("//"):
                        p = p[2:]
                    if "://" in p:
                        host = p.split("://", 1)[1].split("/", 1)[0]
                    else:
                        host = p.split("/", 1)[0]
                    # try resolve and TCP connect to common SMB port 445
                    start = time.time()
                    try:
                        ip = socket.gethostbyname(host)
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(0.8)
                        s.connect((ip, 445))
                        s.close()
                        latency_ms = int((time.time() - start) * 1000)
                        reachable = True
                    except Exception:
                        latency_ms = None
                except Exception:
                    latency_ms = None
        except Exception:
            brain_log.log("[DriveManager] _probe_path failed:\n" + traceback.format_exc())
        return reachable, free_gb, total_gb, latency_ms

    def _compute_perf_score(self, info):
        """
        Compute a simple perf score 0..1:
        - Local drives with >20GB free -> 0.9
        - Local drives with >5GB free -> 0.7
        - Network drives degrade by latency
        """
        try:
            typ = info.get("type", "LOCAL")
            free = info.get("free_gb") or 0
            latency = info.get("latency_ms")
            score = 0.0
            if typ == "LOCAL":
                if free >= 50:
                    score = 0.95
                elif free >= 20:
                    score = 0.85
                elif free >= 5:
                    score = 0.65
                else:
                    score = 0.4
            else:
                # network
                score = 0.6 if info.get("reachable") else 0.0
                if latency:
                    # penalize by latency up to 200ms
                    penalty = min(1.0, latency / 200.0)
                    score = max(0.05, score * (1.0 - 0.5 * penalty))
            return round(score, 2)
        except Exception:
            return 0.0

    def scan_drives(self):
        with self.lock:
            try:
                drives = {}
                # local mounts
                for m in self._list_local_mounts():
                    path = m["path"]
                    reachable, free_gb, total_gb, latency_ms = self._probe_path(path)
                    info = {
                        "id": m["id"],
                        "path": path,
                        "type": "LOCAL",
                        "reachable": reachable,
                        "free_gb": free_gb,
                        "total_gb": total_gb,
                        "latency_ms": latency_ms,
                        "role": [],
                    }
                    info["perf_score"] = self._compute_perf_score(info)
                    drives[m["id"]] = info

                # configured network drives
                for nd in self._list_configured_network():
                    path = nd.get("path")
                    did = nd.get("id")
                    reachable, free_gb, total_gb, latency_ms = self._probe_path(path)
                    info = {
                        "id": did,
                        "path": path,
                        "type": nd.get("type", "NET"),
                        "reachable": reachable,
                        "free_gb": free_gb,
                        "total_gb": total_gb,
                        "latency_ms": latency_ms,
                        "role": nd.get("role", []),
                    }
                    info["perf_score"] = self._compute_perf_score(info)
                    drives[did] = info

                cluster_state["drives"] = drives
                self.last_scan = time.time()
            except Exception:
                brain_log.log("[DriveManager] scan_drives failed:\n" + traceback.format_exc())

# Replace earlier drive_manager with enhanced one
drive_manager = DriveManager(config)

# -------------------------
# Improved ReasoningEngine
# -------------------------
class ReasoningEngine:
    """
    Improved ReasoningEngine:
    - Scores drives for caching and game placement using multiple signals:
      perf_score, free space, role preferences, and predicted game needs.
    - Returns a ranked list of candidate drives and an explanation.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def score_drive_for_game(self, drive_info, game_profile=None):
        """
        Compute a score 0..1 for how suitable this drive is for caching/placing game assets.
        """
        try:
            base = float(drive_info.get("perf_score", 0.0))
            free = float(drive_info.get("free_gb") or 0.0)
            typ = drive_info.get("type", "LOCAL")
            role = drive_info.get("role", [])
            score = base
            # free space bonus
            if free >= 50:
                score += 0.2
            elif free >= 20:
                score += 0.1
            elif free >= 5:
                score += 0.02
            # role preference: if drive role contains 'CACHE' or 'GAME' boost
            if isinstance(role, (list, tuple)):
                if any("CACHE" in r for r in role):
                    score += 0.15
                if any("GAME" in r for r in role):
                    score += 0.12
            # game_profile hint: prefer SSD-like drives if profile requests low latency
            if game_profile:
                if game_profile.get("prefer_ssd") and typ == "LOCAL":
                    score += 0.08
            # clamp
            score = max(0.0, min(1.0, score))
            return round(score, 3)
        except Exception:
            return 0.0

    def decide_cache_drives_for_network(self, drives, game_profile=None, top_n=2):
        """
        Return top N drives ranked for caching with explanation.
        """
        try:
            scored = []
            for did, info in (drives or {}).items():
                if not info.get("reachable"):
                    continue
                s = self.score_drive_for_game(info, game_profile)
                scored.append((did, s, info))
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [d for d, s, _ in scored[:top_n]]
            explanation = [{"id": d, "score": s} for d, s, _ in scored[:top_n]]
            return {"cache_drives": chosen, "explanation": explanation}
        except Exception:
            return {"cache_drives": [], "explanation": "error"}

reasoning_engine = ReasoningEngine(config)

# -------------------------
# Enhanced GameTelemetryCollector
# -------------------------
class GameTelemetryCollector:
    """
    Enhanced GameTelemetryCollector:
    - Samples per-process CPU and memory using psutil
    - Attempts to attach GPU usage via NVML if available
    - Calls smart_game_detector.set_last_telemetry() to attach telemetry to the active game
    """

    def __init__(self):
        self.last_sample = 0

    def _probe_gpu_for_pid(self, pid):
        """
        Best-effort: if NVML available, try to find GPU utilization for processes.
        NVML does not map processes to GPU usage reliably across drivers; this is heuristic.
        """
        if not NVML_AVAILABLE:
            return None
        try:
            # simple approach: return overall GPU util as proxy
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {"gpu_percent": util.gpu, "gpu_mem_percent": util.memory}
        except Exception:
            return None

    def sample(self):
        try:
            gs = cluster_state.get("game_session")
            if not gs:
                return
            pid = gs.get("pid")
            if not pid:
                return
            if psutil is None:
                return
            try:
                p = psutil.Process(int(pid))
            except Exception:
                # process may have exited
                return
            try:
                cpu = p.cpu_percent(interval=0.05)
            except Exception:
                cpu = 0.0
            try:
                mem = p.memory_info().rss / (1024 ** 2)
            except Exception:
                mem = 0.0
            telemetry = {"cpu_percent": round(cpu, 2), "mem_mb": int(mem)}
            # GPU probe
            try:
                gpu = self._probe_gpu_for_pid(pid)
                if gpu:
                    telemetry.update(gpu)
            except Exception:
                pass
            # attach FPS if detector provided an estimate earlier
            try:
                # detector may have probed FPS already; merge if present
                det = cluster_state.get("game_session", {}).get("last_telemetry", {})
                if det and det.get("fps_est"):
                    telemetry["fps_est"] = det.get("fps_est")
            except Exception:
                pass
            # publish telemetry via detector API if available
            try:
                smart_game_detector.set_last_telemetry(telemetry)
            except Exception:
                cluster_state.setdefault("game_session", {})["last_telemetry"] = telemetry
            self.last_sample = time.time()
        except Exception:
            brain_log.log("[GameTelemetryCollector] sample failed:\n" + traceback.format_exc())

# Replace earlier game_telemetry with enhanced one
game_telemetry = GameTelemetryCollector()

import argparse

# -------------------------
# Ensure default config file exists (safe to call at startup)
# -------------------------
def ensure_default_config(path=CONFIG_FILE):
    """
    Create a sensible default config file if none exists.
    This is safe to call at startup; it will not overwrite an existing config.
    """
    default = {
        "loop_interval_seconds": 5,
        "game_executables": ["back4blood.exe", "valorant.exe", "csgo.exe", "dota2.exe", "fortnite.exe"],
        "history_half_life_seconds": 86400,
        "game_detection_grace_seconds": 10,
        "prediction_time_window_seconds": 604800,
        "prediction_min_score": 0.02,
        "fuzzy_similarity_threshold": 0.70,
        "detection_confidence_threshold": 0.6,
        "fps_hooks": {
            # Example: "mygame.exe": "python fps_helper.py --pid {pid}"
        },
        "network_drives": [],
        "status_http_port": 8765,
        "w_name": 5.0,
        "w_path": 3.0,
        "w_cmd": 2.5,
        "w_parent": 1.0,
        "w_foreground": 2.0,
        "w_recency": 1.0,
        "w_fuzzy": 1.2,
        "w_gpu_fps": 1.0
    }
    try:
        if not os.path.exists(path):
            save_json(path, default)
            brain_log.log(f"[CONFIG] Wrote default config to {path}")
            return True
    except Exception:
        brain_log.log("[CONFIG] ensure_default_config failed:\n" + traceback.format_exc())
    return False

# -------------------------
# Small FPS helper script (can be used in fps_hooks)
# Usage: python fps_helper.py --pid 1234
# It prints a single numeric FPS estimate to stdout or exits non-zero on failure.
# -------------------------
def fps_helper_main():
    """
    Best-effort FPS helper:
    - If NVML available, uses GPU utilization as a proxy to estimate FPS.
    - If pywin32 available, tries to parse foreground window title for FPS patterns.
    - If neither yields a value, returns a small default or exits with non-zero.
    """
    parser = argparse.ArgumentParser(prog="fps_helper.py", description="Estimate FPS for a PID (best-effort).")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to probe")
    parser.add_argument("--method", choices=["nvml", "title", "both"], default="both", help="Preferred probe method")
    args = parser.parse_args()

    pid = args.pid
    method = args.method

    def probe_title(pid):
        try:
            if win32gui and win32process:
                hwnd = win32gui.GetForegroundWindow()
                if not hwnd:
                    return None
                _, fgpid = win32process.GetWindowThreadProcessId(hwnd)
                if fgpid != pid:
                    return None
                title = win32gui.GetWindowText(hwnd) or ""
                import re
                m = re.search(r"(\d{2,3})\s*fps", title, flags=re.IGNORECASE)
                if m:
                    return float(m.group(1))
                m2 = re.search(r"fps[:\s]*([0-9]{1,3})", title, flags=re.IGNORECASE)
                if m2:
                    return float(m2.group(1))
        except Exception:
            pass
        return None

    def probe_nvml():
        try:
            if not NVML_AVAILABLE:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            # heuristic mapping: 100% -> 200 FPS
            est = max(5.0, (gpu_util / 100.0) * 200.0)
            return float(est)
        except Exception:
            return None

    fps = None
    if method in ("both", "title"):
        fps = probe_title(pid)
    if fps is None and method in ("both", "nvml"):
        fps = probe_nvml()
    if fps is not None:
        # print only the numeric value for hooks
        print(f"{fps:.1f}")
        return 0
    # fallback: no reliable estimate
    return 2

# -------------------------
# Admin CLI for quick tasks
# -------------------------
def admin_cli():
    """
    Simple command-line admin helper:
    - init-config : create default config if missing
    - start-status : start the HTTP status server in the background
    - stop-status  : stop the HTTP status server
    - dump-state   : write a snapshot of cluster_state to disk
    - fps-helper   : run the fps helper (delegates to fps_helper_main)
    """
    parser = argparse.ArgumentParser(prog="borg_admin", description="Borg organism admin helper")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init-config", help="Create default config file if missing")
    sub.add_parser("start-status", help="Start HTTP status server")
    sub.add_parser("stop-status", help="Stop HTTP status server")
    sub.add_parser("dump-state", help="Dump cluster state to borg_cluster_snapshot.json")
    fps_p = sub.add_parser("fps-helper", help="Run FPS helper (pass --pid)")
    fps_p.add_argument("--pid", type=int, required=True, help="PID to probe")
    fps_p.add_argument("--method", choices=["nvml", "title", "both"], default="both", help="Probe method")

    args = parser.parse_args()
    if args.cmd == "init-config":
        ok = ensure_default_config()
        if ok:
            print(f"Default config written to {CONFIG_FILE}")
        else:
            print(f"Config already exists or failed to write: {CONFIG_FILE}")
        return

    if args.cmd == "start-status":
        start_status_server()
        print("Status server started (background).")
        return

    if args.cmd == "stop-status":
        stop_status_server()
        print("Status server stop requested.")
        return

    if args.cmd == "dump-state":
        dump_state_to_file()
        print("Cluster state dumped.")
        return

    if args.cmd == "fps-helper":
        # delegate to fps helper logic
        # emulate calling fps_helper_main with provided args
        sys.argv = ["fps_helper.py", "--pid", str(args.pid), "--method", args.method]
        try:
            rc = fps_helper_main()
            # fps_helper_main returns exit code; if numeric printed, it's already printed
            if isinstance(rc, int):
                sys.exit(rc)
        except SystemExit as e:
            raise
        except Exception:
            print("fps-helper failed", file=sys.stderr)
            sys.exit(3)

    parser.print_help()

# -------------------------
# If this file is executed directly, provide admin CLI
# -------------------------
if __name__ == "__main__":
    admin_cli()





