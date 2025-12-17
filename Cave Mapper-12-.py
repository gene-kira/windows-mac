#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, threading, queue, hashlib, math
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter, deque
from dataclasses import dataclass

# Optional libs
try: import psutil
except ImportError: psutil = None
try: import yaml
except ImportError: yaml = None
try: import networkx as nx
except ImportError: nx = None
# Optional GPU monitoring via NVML
try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
# Optional Windows window title via pywin32
try:
    import win32gui
    WIN32_AVAILABLE = True
except Exception:
    WIN32_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, messagebox

# =========================== requirements.txt auto-create ===========================
def ensure_requirements_file():
    reqs = ["psutil", "pyyaml", "networkx", "pynvml", "pywin32"]
    try:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(reqs))
        print("requirements.txt file created/updated.")
    except Exception:
        pass

# =============================== Windows path helpers ===============================
def expand_vars(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))

def guess_b4b_paths(user_name: Optional[str] = None) -> Dict[str, List[str]]:
    q_drive = "Q:\\"
    steam_common = os.path.join(q_drive, "SteamLibrary", "steamapps", "common", "Back 4 Blood")
    epic_common = os.path.join(q_drive, "EpicGames", "Back4Blood")
    q_users = os.path.join(q_drive, "Users")
    localapp_q = os.path.join(q_users, user_name, "AppData", "Local", "Back4Blood", "Saved") if user_name else ""
    appdata_q = os.path.join(q_users, user_name, "AppData", "Roaming", "Back4Blood") if user_name else ""
    localapp = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")

    items = {
        "shader_cache": [
            expand_vars(os.path.join(localapp, "Back4Blood", "Saved", "Shaders")),
            expand_vars(os.path.join(localapp, "Back4Blood", "Saved", "ShaderCache")),
        ],
        "profiles": [
            expand_vars(appdata_q) if appdata_q else "",
            expand_vars(localapp_q) if localapp_q else "",
            expand_vars(os.path.join(appdata, "Back4Blood")),
            expand_vars(os.path.join(localapp, "Back4Blood", "Saved")),
        ],
        "game_assets": [
            expand_vars(steam_common),
            expand_vars(epic_common),
        ],
        "steam_artifacts": [
            expand_vars(os.path.join(q_drive, "SteamLibrary", "steamapps", "appmanifest_1293860.acf")),
        ]
    }
    for k in list(items.keys()):
        items[k] = [p for p in items[k] if p and isinstance(p, str)]
    return items

# =============================== Disk watcher hooks ===============================
class DiskWatcher(threading.Thread):
    def __init__(self, log_q: queue.Queue, paths: Dict[str, List[str]], interval: float = 1.0):
        super().__init__(daemon=True)
        self.log_q = log_q
        self.paths_by_class = {cls: [p for p in lst if p and os.path.isdir(p)] for cls, lst in paths.items()}
        self.interval = interval
        self.running = True
        self.state: Dict[str, Dict[str, str]] = {cls: {} for cls in self.paths_by_class}

    def _log(self, msg, kind="thought"): self.log_q.put({"type": kind, "msg": msg})

    def _dir_signature(self, root: str, max_files: int = 300) -> str:
        h = hashlib.sha1(); count = 0
        try:
            for base, dirs, files in os.walk(root):
                dirs.sort(); files.sort()
                for fn in files:
                    fp = os.path.join(base, fn)
                    try:
                        st = os.stat(fp)
                        h.update(str(st.st_mtime_ns).encode()); h.update(str(st.st_size).encode())
                    except Exception:
                        continue
                    count += 1
                    if count >= max_files: break
                if count >= max_files: break
        except Exception:
            pass
        return h.hexdigest()

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        return {cls: {d: self._dir_signature(d) for d in self.paths_by_class.get(cls, [])}
                for cls in self.paths_by_class}

    def run(self):
        self.state = self.snapshot()
        total_dirs = sum(len(v) for v in self.paths_by_class.values())
        self._log(f"DiskWatcher: watching {total_dirs} directories in classes: {list(self.paths_by_class.keys())}.")
        while self.running:
            try:
                time.sleep(self.interval)
                snap = self.snapshot()
                for cls, dirsigs in snap.items():
                    for d, sig in dirsigs.items():
                        prev = self.state.get(cls, {}).get(d)
                        if prev and sig != prev:
                            self._log(f"Disk activity detected in {d} ({cls})")
                            self.log_q.put({"type": "disk_event", "msg": d, "class": cls})
                self.state = snap
            except Exception as e:
                self._log(f"DiskWatcher error: {e}", kind="anomaly")
                time.sleep(self.interval)

    def stop(self):
        self.running = False
        self._log("DiskWatcher stopping.")

# =================================== AutoLoader =====================================
class AutoLoader:
    def __init__(self, log_q: queue.Queue): self.log_q = log_q
    def _log(self, msg, kind="thought"): self.log_q.put({"type": kind, "msg": msg})
    def ensure(self, scopes: List[str]) -> bool:
        missing = []
        if "core" in scopes and psutil is None: missing.append("psutil")
        if "policy" in scopes and yaml is None: missing.append("pyyaml")
        if "graph" in scopes and nx is None: missing.append("networkx")
        if "gpu" in scopes and not NVML_AVAILABLE: missing.append("pynvml")
        if "win" in scopes and not WIN32_AVAILABLE: missing.append("pywin32")
        if not missing: return True
        self._log(f"Missing libraries: {missing}")
        self._log(f"Install with: pip install {' '.join(missing)}")
        return False

# ==================================== Inventory =====================================
@dataclass
class InventorySnapshot:
    hardware: Dict[str, Any]
    software: Dict[str, Any]
    profiles: Dict[str, Any]

class Inventory:
    def __init__(self, log_q: queue.Queue): self.log_q = log_q
    def _log(self, msg): self.log_q.put({"type": "thought", "msg": msg})
    def collect(self) -> InventorySnapshot:
        hw, sw = {}, {}
        hw["cpu"] = {"logical": psutil.cpu_count(True) if psutil else None,
                     "physical": psutil.cpu_count(False) if psutil else None}
        hw["memory"] = {"total": psutil.virtual_memory().total if psutil else None}
        hw["storage"] = []
        if psutil:
            for p in psutil.disk_partitions(False):
                hw["storage"].append({"device": p.device, "mount": p.mountpoint, "fs": p.fstype})
        hw["network"] = list(psutil.net_if_addrs().keys()) if psutil else []
        sw["os"] = {"platform": sys.platform, "python": sys.version.split()[0]}
        profiles = {"idle": {"cpu_pct": 5}, "work": {"cpu_pct": 35}, "game": {"cpu_pct": 60}}
        snap = InventorySnapshot(hw, sw, profiles)
        self._log("Inventory collected.")
        return snap

# =================================== Policy Engine ===================================
DEFAULT_POLICY = {
    "version": 27,
    "defaults": {
        "cpu_ceiling_pct": 15,
        "gpu_ceiling_pct": 60,
        "jump_ahead_seconds": 2,
        "preload_budget_mb_min": 160
    },
    "always_warm": {
        "apps": ["steam","steam.exe","epic","chrome","firefox","edge","code","powershell","terminal"],
        "games": ["back4blood","back4blood.exe","valorant","fortnite","elden ring","easyanticheat","overlay"]
    },
    "auto_target_thresholds": {"cpu_pct": 12, "gpu_pct": 25, "hit_count_min": 3, "evening_hours_bonus": 1},
    "routine_clustering": {"window_size": 20, "min_support": 3},
    "prediction": {
        "dirichlet_alpha": 0.8,
        "recency_decay": 0.997,
        "routine_bias": {"gaming": 1.38, "work": 1.0},
        "hour_priors_weight": 0.16,
        "weekday_priors_weight": 0.15,
        "month_priors_weight": 0.08,
        "motif_boost_weight": 0.2,
        "title_head_weight": 0.24,
        "markov_head_weight": 0.76,
        "baseline_weights": {"cpu": 0.1, "gpu": 0.08, "net": 0.12, "io": 0.12},
        "changepoint_sensitivity": {"cpu": 2.5, "gpu": 2.3, "net": 2.5, "io": 2.5},
        "cold_start_boosts": ["steam.exe", "overlay", "easyanticheat", "back4blood.exe"],
        "hysteresis": {"stable_frames": 3, "high_conf": 0.46, "medium_conf_with_temp": 0.40, "grace_period_s": 6.0, "temp_floor": 0.55},
        "explore_ratio": 0.18
    },
    "preloader": {
        "phase_budgets": {"auth": 0.1, "services": 0.15, "shader_hot": 0.22, "shader_cold": 0.18, "assets": 0.25, "secondary": 0.1},
        "temp_thresholds": {"shader_hot": 0.40, "shader_cold": 0.55, "assets": 0.60},
        "early_throttle": {"near_cpu_pct": 0.85, "near_gpu_pct": 0.85}
    },
    "persistence": {
        "path": "scanner_state.json",
        "interval_s": 120,
        "max_edges": 8000,
        "max_ctx": 6000
    },
    "soft_bans": {
        "ttl_medium_temp_s": 1200,
        "ttl_high_temp_s": 3600
    }
}

class PolicyEngine:
    def __init__(self, log_q: queue.Queue, path: Optional[str] = "policies.yaml"):
        self.log_q = log_q; self.path = path; self.policy = DEFAULT_POLICY
    def _log(self, msg): self.log_q.put({"type": "thought", "msg": msg})
    def load(self):
        try:
            if self.path and os.path.exists(self.path):
                if yaml:
                    with open(self.path, "r", encoding="utf-8") as f:
                        self.policy = yaml.safe_load(f)
                    self._log("Policy loaded (YAML).")
                else:
                    with open(self.path, "r", encoding="utf-8") as f:
                        self.policy = json.load(f)
                    self._log("Policy loaded (JSON fallback).")
            else:
                self._log("No policy file found; using defaults.")
        except Exception as e:
            self._log(f"Policy load failed: {e}; using defaults.")
    def save(self):
        try:
            if yaml:
                with open(self.path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self.policy, f)
                self._log("Policy saved (YAML).")
            else:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.policy, f, indent=2)
                self._log("Policy saved (JSON fallback).")
        except Exception as e:
            self._log(f"Policy save failed: {e}")
    def ceilings(self) -> Dict[str, Any]:
        d = self.policy.get("defaults", {})
        return {"cpu_pct": d.get("cpu_ceiling_pct", 15),
                "gpu_pct": d.get("gpu_ceiling_pct", 60),
                "preload_budget_mb_min": d.get("preload_budget_mb_min", 160)}
    def jump_ahead(self) -> int:
        return int(self.policy.get("defaults", {}).get("jump_ahead_seconds", 2))
    def always_warm(self) -> Dict[str, List[str]]:
        aw = self.policy.get("always_warm", {})
        return {"apps": [a.lower() for a in aw.get("apps", [])],
                "games": [g.lower() for g in aw.get("games", [])]}
    def auto_target_thresholds(self) -> Dict[str, Any]:
        return self.policy.get("auto_target_thresholds", {"cpu_pct": 12, "gpu_pct": 25, "hit_count_min": 3, "evening_hours_bonus": 1})
    def routine_params(self) -> Dict[str, Any]:
        return self.policy.get("routine_clustering", {"window_size": 20, "min_support": 3})
    def prediction_params(self) -> Dict[str, Any]:
        return self.policy.get("prediction", DEFAULT_POLICY["prediction"])
    def preloader_params(self) -> Dict[str, Any]:
        return self.policy.get("preloader", DEFAULT_POLICY["preloader"])
    def persistence_params(self) -> Dict[str, Any]:
        return self.policy.get("persistence", DEFAULT_POLICY["persistence"])
    def soft_ban_params(self) -> Dict[str, Any]:
        return self.policy.get("soft_bans", DEFAULT_POLICY["soft_bans"])

# ======================================== Agents =====================================
class SystemAgent:
    def sample(self) -> Dict[str, Any]:
        return {
            "cpu_pct": psutil.cpu_percent(None) if psutil else 0.0,
            "mem": dict(psutil.virtual_memory()._asdict()) if psutil else {},
            "proc_count": len(psutil.pids()) if psutil else 0,
            "disk_io": dict(psutil.disk_io_counters()._asdict()) if psutil else {}
        }

class GPUAgent:
    def __init__(self):
        self.available = NVML_AVAILABLE
        if self.available:
            try: pynvml.nvmlInit()
            except Exception: self.available = False
    def sample(self) -> Dict[str, Any]:
        if not self.available:
            return {"gpu_enabled": False, "util_pct": None, "mem_used_mb": None, "mem_total_mb": None}
        try:
            count = pynvml.nvmlDeviceGetCount()
            util = 0; mem_used = 0; mem_total = 0
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = max(util, int(u))
                mem_used = max(mem_used, int(m.used/1024/1024))
                mem_total = max(mem_total, int(m.total/1024/1024))
            return {"gpu_enabled": True, "util_pct": util, "mem_used_mb": mem_used, "mem_total_mb": mem_total}
        except Exception:
            return {"gpu_enabled": False, "util_pct": None, "mem_used_mb": None, "mem_total_mb": None}

# =============================== Network with geometry persistence ====================
class NetworkAgent:
    def __init__(self):
        self.window = deque(maxlen=180)  # last addresses + family
        self.timestamps = deque(maxlen=180)
        self.geom_persist = {"width": 0.0, "height": Counter()}  # decayed geometry
        self.decay = 0.85  # half-life-ish per tick
        self.min_frames = 2

    def _asn_family(self, addr: str) -> str:
        a = addr.lower()
        if any(k in a for k in ["steamcontent","valve","steam"]): return "steam"
        if any(k in a for k in ["cloudfront","amazon","aws"]): return "cloudfront"
        if "akamai" in a or "edgekey" in a or "akamaiedge" in a: return "akamai"
        if "epicgames" in a or "eac" in a or "easyanticheat" in a: return "epic/eac"
        return "other"

    def sample(self) -> Dict[str, Any]:
        conns = []
        now = time.time()
        if psutil:
            for c in psutil.net_connections("inet"):
                try:
                    r = str(c.raddr)
                    fam = self._asn_family(r)
                    conns.append({"laddr": str(c.laddr), "raddr": r, "status": c.status, "family": fam})
                    self.window.append((fam, now))
                    self.timestamps.append(now)
                except Exception:
                    pass

        recent = [(fam, ts) for (fam, _ts), ts in zip(self.window, self.timestamps) if now - ts <= 20]
        width = len(recent)
        height_by_fam = Counter([fam for fam, _ in recent])

        persistent = width >= self.min_frames
        self.geom_persist["width"] = self.decay * self.geom_persist["width"] + (1.0 - self.decay) * (width if persistent else 0.0)
        for fam in set(list(self.geom_persist["height"].keys()) + list(height_by_fam.keys())):
            prev = self.geom_persist["height"].get(fam, 0.0)
            val = height_by_fam.get(fam, 0.0) if persistent else 0.0
            self.geom_persist["height"][fam] = self.decay * prev + (1.0 - self.decay) * val
            if self.geom_persist["height"][fam] < 0.05:
                del self.geom_persist["height"][fam]

        geometry = {"width": int(width), "height": dict(height_by_fam), "persist": {"width": round(self.geom_persist["width"],2),
                                                                                    "height": {k: round(v,2) for k, v in self.geom_persist["height"].items()}}}
        cdn_burst = int(sum(height_by_fam[k] for k in ["steam","cloudfront","akamai","epic/eac"]))
        return {"connections": conns[:100], "geometry": geometry, "cdn_burst": cdn_burst, "height_by_fam": dict(height_by_fam)}

# =============================== Foreground/title stack ===============================
class ForegroundAgent:
    def __init__(self):
        self.last_focus: Optional[str] = None
        self.stability = deque(maxlen=5)
        self.last_title: Optional[str] = None
        self.title_stack = deque(maxlen=12)
    def _get_window_title(self) -> Optional[str]:
        if not WIN32_AVAILABLE: return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return None
    def sample_focus(self) -> Tuple[Optional[str], Optional[str], List[str]]:
        name = None
        if psutil:
            try:
                procs = sorted(psutil.process_iter(attrs=["name","cpu_percent","pid"]),
                               key=lambda p: p.info.get("cpu_percent") or 0, reverse=True)
                if procs:
                    name = (procs[0].info.get("name") or "").lower()
            except Exception:
                pass
        self.stability.append(name)
        if len(self.stability) >= 3 and len(set(self.stability)) == 1:
            self.last_focus = name
        title = self._get_window_title()
        if title and (not self.title_stack or self.title_stack[-1] != title):
            self.title_stack.append(title)
        if title: self.last_title = title
        return self.last_focus or name, self.last_title, list(self.title_stack)

# =============================== Process ancestry chains ===============================
class AncestryAgent:
    def __init__(self):
        self.edges = Counter()
        self.decay = 0.997
        self.last_tick = 0
    def record(self):
        if not psutil: return
        try:
            for p in psutil.process_iter(attrs=["pid","name"]):
                nm = (p.info.get("name") or "").lower()
                parent_nm = ""
                try:
                    parent = p.parent()
                    parent_nm = (parent.name() or "").lower() if parent else ""
                except Exception:
                    parent_nm = ""
                if parent_nm and nm:
                    self.edges[(parent_nm, nm)] += 1
        except Exception:
            pass
        tick = int(time.time()) // 30
        if tick != self.last_tick:
            for k in list(self.edges.keys()):
                self.edges[k] = int(self.edges[k] * self.decay)
            self.last_tick = tick
    def next_boosts(self, last_focus: Optional[str], candidates: List[str], max_strength: float = 0.25) -> Dict[str, float]:
        boost = {}
        if not last_focus: return boost
        for c in candidates:
            w = self.edges.get((last_focus.lower(), c.lower()), 0)
            if w > 0:
                boost[c] = min(max_strength, 0.03 * w)
        return boost
    def export(self, max_edges: int) -> Dict[str, int]:
        top = Counter(self.edges).most_common(max_edges)
        return {"::".join(k): v for k, v in top}
    def import_edges(self, data: Dict[str, int]):
        for k, v in data.items():
            try:
                a, b = k.split("::", 1)
                self.edges[(a, b)] = int(v)
            except Exception:
                continue

# =============================== Service telemetry agent ===============================
class ServiceAgent:
    def __init__(self, window_s: float = 12.0):
        self.window = deque(maxlen=40)
        self.window_s = window_s
    def sample(self):
        now = time.time()
        if psutil:
            try:
                for p in psutil.process_iter(attrs=["name"]):
                    nm = (p.info.get("name") or "").lower()
                    if any(k in nm for k in ["overlay","easyanticheat","eac"]):
                        self.window.append((nm, now))
            except Exception:
                pass
        self.window = deque([(nm, ts) for nm, ts in self.window if now - ts <= self.window_s], maxlen=40)
        cluster_4s = sum(1 for _, ts in self.window if now - ts <= 4.0)
        return {"wakes": len(self.window), "cluster": cluster_4s}

# =============================== Launch lattice (variants) =============================
class LaunchLattice:
    """
    Track variants of overlay -> eac -> game edges with timing distributions per edge.
    Rising-edge likelihood per target using variant-aware gaps.
    """
    def __init__(self, max_len: int = 8):
        self.events = deque(maxlen=max_len)  # (name, ts)
        # edges[target][variant] -> list of gap_deltas (overlay->eac, eac->game)
        self.edges = defaultdict(lambda: defaultdict(list))
        self.decay = 0.997
        self.last_tick = 0
        self.overlay_variants = ["overlay","steam overlay","steamwebhelper","epic overlay"]
        self.eac_variants = ["easyanticheat","eac","eac_launcher","eacservice"]

    def record_event(self, name: str):
        self.events.append((name.lower(), time.time()))
        tick = int(time.time()) // 30
        if tick != self.last_tick:
            for tgt in list(self.edges.keys()):
                for var in list(self.edges[tgt].keys()):
                    self.edges[tgt][var] = [d * self.decay for d in self.edges[tgt][var]]
            self.last_tick = tick

    def _match_variant(self, names: List[str]) -> Tuple[Optional[str], Optional[str]]:
        ov = next((v for v in self.overlay_variants if any(v in n for n in names)), None)
        ev = next((v for v in self.eac_variants if any(v in n for n in names)), None)
        return ov, ev

    def observe_curve(self, target: str):
        seq = list(self.events)
        names = [n for n, _ in seq]
        ov, ev = self._match_variant(names)
        if ov and ev and any(target in n for n in names):
            t_overlay = max((ts for n, ts in seq if ov in n), default=None)
            t_eac = max((ts for n, ts in seq if ev in n), default=None)
            t_game = max((ts for n, ts in seq if target in n), default=None)
            if t_overlay and t_eac and t_game and t_overlay < t_eac < t_game:
                gap = ((t_eac - t_overlay) + (t_game - t_eac)) / 2.0
                self.edges[target][f"{ov}|{ev}"].append(gap)

    def rising_edge(self, target: str) -> float:
        """
        Compare recent sequence to learned variant average gap; return [0,1] edge likelihood.
        """
        seq = list(self.events)
        names = [n for n, _ in seq]
        ov, ev = self._match_variant(names)
        if not ov or not ev: return 0.0
        curves = self.edges.get(target, {})
        if not curves or not curves.get(f"{ov}|{ev}"): return 0.0
        avg_gap = sum(curves[f"{ov}|{ev}"]) / len(curves[f"{ov}|{ev}"])
        t_overlay = max((ts for n, ts in seq if ov in n), default=None)
        t_eac = max((ts for n, ts in seq if ev in n), default=None)
        now = time.time()
        if t_overlay and t_eac and t_overlay < t_eac:
            elapsed = now - t_eac
            score = max(0.0, 1.0 - abs(elapsed - avg_gap) / (avg_gap + 1e-6))
            return min(1.0, score)
        return 0.0

# =============================== IO heatmap (classed) =================================
class IOHeatmap:
    def __init__(self, max_dirs: int = 50):
        self.heat_shader = Counter()
        self.heat_profiles = Counter()
        self.heat_assets = Counter()
        self.decay_shader = 0.995
        self.decay_profiles = 0.995
        self.decay_assets = 0.995
        self.last_tick = 0
        self.max_dirs = max_dirs

    def record_dir_touch(self, directory: str, cls: Optional[str] = None):
        key = (directory or "").lower()
        if cls == "shader_cache":
            self.heat_shader[key] += 4
        elif cls == "profiles":
            self.heat_profiles[key] += 3
        elif cls == "game_assets":
            self.heat_assets[key] += 3
        else:
            self.heat_assets[key] += 1  # default minor bump

    def sample(self) -> Dict[str, Dict[str, float]]:
        tick = int(time.time()) // 20
        if tick != self.last_tick:
            def decay_map(store: Counter, decay: float):
                for k in list(store.keys()):
                    store[k] = max(0.0, store[k] * decay)
                    if store[k] < 0.05: del store[k]
            decay_map(self.heat_shader, self.decay_shader)
            decay_map(self.heat_profiles, self.decay_profiles)
            decay_map(self.heat_assets, self.decay_assets)
            self.last_tick = tick
        def top_map(store: Counter) -> Dict[str, float]:
            return {k: round(v, 2) for k, v in store.most_common(self.max_dirs)}
        return {
            "shader": top_map(self.heat_shader),
            "profiles": top_map(self.heat_profiles),
            "assets": top_map(self.heat_assets),
        }
# =================================== Baseline + change-points (ensemble) ==============
class BaselineTracker:
    def __init__(self, window: int = 90, decay: float = 0.99):
        self.cpu = deque(maxlen=window)
        self.gpu = deque(maxlen=window)
        self.net = deque(maxlen=window)
        self.io = deque(maxlen=window)
        self.decay = decay
        self.avg = {"cpu": 0.0, "gpu": 0.0, "net": 0.0, "io": 0.0}
    def _update_avg(self, key: str, val: float):
        self.avg[key] = self.decay * self.avg[key] + (1 - self.decay) * val
    def record(self, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any]):
        cpu = float(sysd.get("cpu_pct") or 0.0)
        gpu = float(gpud.get("util_pct") or 0.0) if gpud.get("gpu_enabled") else 0.0
        net = float(len(netd.get("connections", [])))
        io = float(sysd.get("disk_io", {}).get("read_bytes", 0) + sysd.get("disk_io", {}).get("write_bytes", 0)) / 1_000_000.0
        self.cpu.append(cpu); self.gpu.append(gpu); self.net.append(net); self.io.append(io)
        self._update_avg("cpu", cpu); self._update_avg("gpu", gpu); self._update_avg("net", net); self._update_avg("io", io)
    def deltas(self) -> Dict[str, float]:
        def norm(val, avg):
            if avg <= 1e-6: return 0.0 if val == 0 else 1.0
            return max(0.0, (val - avg) / (avg + 1e-6))
        last_cpu = self.cpu[-1] if self.cpu else 0.0
        last_gpu = self.gpu[-1] if self.gpu else 0.0
        last_net = self.net[-1] if self.net else 0.0
        last_io  = self.io[-1] if self.io else 0.0
        return {"cpu": norm(last_cpu, self.avg["cpu"]),
                "gpu": norm(last_gpu, self.avg["gpu"]),
                "net": norm(last_net, self.avg["net"]),
                "io":  norm(last_io,  self.avg["io"])}

class ChangePointDetector:
    def __init__(self, k: float = 2.5, window: int = 80, cooldown_s: float = 6.0):
        self.k = k; self.window = deque(maxlen=window)
        self.last_hit_time = 0.0; self.cooldown = cooldown_s
    def record(self, val: float): self.window.append(val)
    def is_change(self) -> bool:
        if len(self.window) < 10: return False
        now = time.time()
        if now - self.last_hit_time < self.cooldown: return False
        mu = sum(self.window)/len(self.window)
        var = sum((x-mu)**2 for x in self.window)/len(self.window)
        sigma = (var ** 0.5) or 1e-6
        last = self.window[-1]
        hit = (last - mu) / sigma >= self.k
        if hit: self.last_hit_time = now
        return hit

class CPEnsemble:
    def __init__(self, window_s: float = 8.0):
        self.hits = deque(maxlen=20)  # (key, ts)
        self.window_s = window_s
    def record(self, key: str, hit: bool):
        if hit: self.hits.append((key, time.time()))
    def strong_intent(self) -> bool:
        now = time.time()
        recent = [k for k, ts in self.hits if now - ts <= self.window_s]
        return len(set(recent) & {"net", "io", "gpu"}) >= 2

# ================================= Routine clustering =================================
class RoutineClusterer:
    def __init__(self, window_size: int = 20, min_support: int = 3, max_n: int = 4):
        self.window = deque(maxlen=window_size)
        self.min_support = min_support
        self.max_n = max_n
        self.motifs = Counter()
        self.active_routine: Optional[str] = None
    def record(self, name: str):
        self.window.append(name.lower())
        seq = list(self.window)
        for n in range(2, min(self.max_n, len(seq))+1):
            motif = tuple(seq[-n:])
            self.motifs[motif] += 1
        self._update_active()
    def _update_active(self):
        if not self.motifs:
            self.active_routine = None; return
        motif, count = max(self.motifs.items(), key=lambda kv: kv[1])
        if count >= self.min_support:
            label = "gaming" if any(any(k in x for k in ["back4blood","steam","game","epic","anticheat","overlay","easyanticheat"]) for x in motif) else "work"
            self.active_routine = label
        else:
            self.active_routine = None
    def routine_label(self) -> Optional[str]: return self.active_routine

# ======================================= Sessionized motifs ===========================
class MotifBoosts:
    def __init__(self, max_len: int = 20, decay: float = 0.997, session_timeout_s: float = 30.0):
        self.window = deque(maxlen=max_len)
        self.edges = Counter()
        self.decay = decay
        self.last_tick = 0
        self.last_event_time = time.time()
        self.session_edges = Counter()
        self.session_timeout = session_timeout_s
    def record(self, name: str):
        now = time.time()
        self.window.append(name.lower())
        seq = list(self.window)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]
            self.edges[(a, b)] += 1
            if now - self.last_event_time <= self.session_timeout:
                self.session_edges[(a, b)] += 2
        self.last_event_time = now
        tick = int(time.time()) // 30
        if tick != self.last_tick:
            for k in list(self.edges.keys()):
                self.edges[k] = int(self.edges[k] * self.decay)
            for k in list(self.session_edges.keys()):
                self.session_edges[k] = int(self.session_edges[k] * self.decay)
            self.last_tick = tick
    def hits_recent(self) -> int:
        return min(5, len(self.window) - 1) if len(self.window) > 1 else 0
    def next_boosts(self, recent: List[str], candidates: List[str], max_strength: float = 0.25) -> Dict[str, float]:
        boost = {}
        last = recent[-1].lower() if recent else None
        if not last: return boost
        for c in candidates:
            w = self.edges.get((last, c.lower()), 0) + 2*self.session_edges.get((last, c.lower()), 0)
            if w > 0:
                boost[c] = min(max_strength, 0.03 * w)
        return boost

# ================================ Temporal cadence memory ==============================
class TemporalCadence:
    def __init__(self, decay: float = 0.997):
        self.hour_counts = Counter()
        self.week_counts = Counter()
        self.month_counts = Counter()
        self.decay = decay
        self.last_tick = 0
    def record(self, target: str):
        now = time.localtime()
        self.hour_counts[(now.tm_hour, target.lower())] += 1
        self.week_counts[(now.tm_wday, target.lower())] += 1
        self.month_counts[(now.tm_mon, target.lower())] += 1
        tick = int(time.time()) // 30
        if tick != self.last_tick:
            for store in (self.hour_counts, self.week_counts, self.month_counts):
                for k in list(store.keys()):
                    store[k] = int(store[k] * self.decay)
            self.last_tick = tick
    def hour_prior(self, target: str) -> float:
        now = time.localtime()
        return float(self.hour_counts.get((now.tm_hour, target.lower()), 0))
    def week_prior(self, target: str) -> float:
        now = time.localtime()
        return float(self.week_counts.get((now.tm_wday, target.lower()), 0))
    def month_prior(self, target: str) -> float:
        now = time.localtime()
        return float(self.month_counts.get((now.tm_mon, target.lower()), 0))

# ================================ Predictive Learner (dual-context) ====================
class PredictiveLearner:
    def __init__(self, log_q: queue.Queue, policy: PolicyEngine, window_size: int = 12, max_n: int = 5):
        self.log_q = log_q
        self.window = deque(maxlen=window_size)
        self.title_window = deque(maxlen=window_size)
        self.max_n = max_n
        self.ctx_counts = defaultdict(Counter)
        self.title_ctx_counts = defaultdict(Counter)
        self.total_counts = Counter()
        pred = policy.prediction_params()
        self.alpha = float(pred.get("dirichlet_alpha", 0.8))
        self.decay = float(pred.get("recency_decay", 0.997))
        self.routine_bias = pred.get("routine_bias", {"gaming": 1.38, "work": 1.0})
        self.hour_w = float(pred.get("hour_priors_weight", 0.16))
        self.week_w = float(pred.get("weekday_priors_weight", 0.15))
        self.month_w = float(pred.get("month_priors_weight", 0.08))
        self.motif_w = float(pred.get("motif_boost_weight", 0.2))
        self.baseline_w = pred.get("baseline_weights", {"cpu":0.1,"gpu":0.08,"net":0.12,"io":0.12})
        self.head_w = {"title": float(pred.get("title_head_weight", 0.24)), "markov": float(pred.get("markov_head_weight", 0.76))}
        self.last_tick = 0
        self.last_top_pred: Optional[str] = None
        self.last_why: Dict[str, Any] = {}
        self.intent_boosts = Counter()
        self.specific_boosts = Counter()
        self.temporal = TemporalCadence(decay=self.decay)
        self.last_baseline_deltas: Dict[str, float] = {"cpu":0,"gpu":0,"net":0,"io":0}

    def _bucket(self, value: Optional[float], step: int) -> int:
        try: return int((value or 0) // step)
        except Exception: return 0

    def _context(self, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str]) -> Tuple[int]:
        cpu_bucket = self._bucket(sysd.get("cpu_pct"), step=6)
        gpu_bucket = self._bucket(gpud.get("util_pct"), step=8)
        net_bucket = self._bucket(len(netd.get("connections", [])), step=15)
        io = sysd.get("disk_io", {})
        io_bucket = self._bucket(io.get("read_bytes", 0) + io.get("write_bytes", 0), step=250_000)
        now = time.localtime()
        routine_code = {"gaming": 1, "work": 2}.get(routine or "", 0)
        title_code = 1 if win_title and any(k in (win_title or "").lower() for k in ["steam","epic","back 4 blood","eac"]) else 0
        return (now.tm_hour, now.tm_wday, now.tm_mon, cpu_bucket, gpu_bucket, net_bucket, io_bucket, routine_code, title_code)

    def record(self, app: str, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str], baseline_deltas: Optional[Dict[str,float]] = None, title_stack: Optional[List[str]] = None):
        ctx = self._context(sysd, gpud, netd, routine, win_title)
        if baseline_deltas:
            self.last_baseline_deltas = baseline_deltas
        if self.window:
            seq = list(self.window)
            for n in range(1, min(self.max_n, len(seq))+1):
                prefix = tuple(seq[-n:])
                self.ctx_counts[(ctx,n)][(prefix,app)] += 1
        if win_title:
            self.title_window.append((win_title.lower(), app))
            seqt = [t for t, _ in list(self.title_window)]
            for n in range(1, min(self.max_n, len(seqt))+1):
                prefix_t = tuple(seqt[-n:])
                self.title_ctx_counts[(ctx,n)][(prefix_t,app)] += 1
        if title_stack:
            seqs = [t.lower() for t in title_stack]
            for n in range(2, min(self.max_n, len(seqs))+1):
                prefix_t = tuple(seqs[-n:])
                self.title_ctx_counts[(ctx,n)][(prefix_t,app)] += 1

        self.window.append(app)
        self.total_counts[app]+=1
        self.temporal.record(app)

        tick=int(time.time())//30
        if tick!=self.last_tick:
            def decay_map(m: Dict[Any, Counter]):
                for key in list(m.keys()):
                    for k in list(m[key].keys()):
                        m[key][k]=int(m[key][k]*self.decay)
            decay_map(self.ctx_counts); decay_map(self.title_ctx_counts)
            for k in list(self.total_counts.keys()):
                self.total_counts[k]=int(self.total_counts[k]*self.decay)
            for c in list(self.intent_boosts.keys()):
                self.intent_boosts[c] = max(0.0, self.intent_boosts[c] * 0.6)
                if self.intent_boosts[c] <= 0: del self.intent_boosts[c]
            for c in list(self.specific_boosts.keys()):
                self.specific_boosts[c] = max(0.0, self.specific_boosts[c] * 0.6)
                if self.specific_boosts[c] <= 0: del self.specific_boosts[c]
            self.last_tick=tick

    def add_intent_boost(self, candidates: List[str], strength: float = 0.1):
        for c in candidates: self.intent_boosts[c] += strength
    def add_specific_boosts(self, boost_map: Dict[str, float]):
        for c, s in boost_map.items(): self.specific_boosts[c] += float(s)

    def predict(self, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str], cands: List[str], top_k: int = 8) -> List[Tuple[str, float]]:
        if not cands: return []
        ctx=self._context(sysd, gpud, netd, routine, win_title); seq=list(self.window); seqt=[t for t,_ in list(self.title_window)]
        scores_markov=Counter(); scores_title=Counter()

        best_prefix=None; best_n=0
        for n in range(1,min(self.max_n,len(seq))+1):
            prefix=tuple(seq[-n:]); counts=self.ctx_counts.get((ctx,n),{})
            total_prefix = sum(counts.values()) + self.alpha * len(cands)
            for c in cands:
                count = counts.get((prefix,c),0)
                smoothed = (count + self.alpha) / total_prefix
                val=smoothed * n
                scores_markov[c]+=val
                if count>0 and n>=best_n:
                    best_prefix=prefix; best_n=n

        best_prefix_t=None; best_nt=0
        for n in range(1,min(self.max_n,len(seqt))+1):
            prefix_t=tuple(seqt[-n:]); counts_t=self.title_ctx_counts.get((ctx,n),{})
            total_prefix_t = sum(counts_t.values()) + self.alpha * len(cands)
            for c in cands:
                count_t = counts_t.get((prefix_t,c),0)
                smoothed_t = (count_t + self.alpha) / total_prefix_t
                val_t=smoothed_t * n
                scores_title[c]+=val_t
                if count_t>0 and n>=best_nt:
                    best_prefix_t=prefix_t; best_nt=n

        if (not scores_markov or sum(scores_markov.values())==0) and (not scores_title or sum(scores_title.values())==0):
            total_global = sum(self.total_counts.values()) + self.alpha * len(cands)
            for c in cands:
                count = self.total_counts.get(c,0)
                scores_markov[c]+= (count + self.alpha) / total_global

        scores=Counter()
        for c in cands:
            scores[c] = self.head_w["markov"] * scores_markov.get(c,0.0) + self.head_w["title"] * scores_title.get(c,0.0)

        for c in cands:
            h_prior = self.temporal.hour_prior(c); w_prior = self.temporal.week_prior(c); m_prior = self.temporal.month_prior(c)
            if h_prior > 0 or w_prior > 0 or m_prior > 0:
                scores[c] += self.hour_w * h_prior + self.week_w * w_prior + self.month_w * m_prior

        bias = self.routine_bias.get(routine or "", 1.0)
        for c in cands:
            if any(k in c for k in ["back4blood","game","steam","epic","valorant","fortnite","elden ring","anticheat","easyanticheat","overlay"]):
                scores[c] *= bias

        bd = self.last_baseline_deltas
        spike = (self.baseline_w["cpu"]*bd.get("cpu",0) + self.baseline_w["gpu"]*bd.get("gpu",0) +
                 self.baseline_w["net"]*bd.get("net",0) + self.baseline_w["io"]*bd.get("io",0))
        if spike > 0:
            for c in cands:
                if any(k in c for k in ["back4blood","steam","epic","anticheat","easyanticheat","overlay","game"]):
                    scores[c] += 0.05 * spike

        for c in cands:
            scores[c] += self.intent_boosts.get(c, 0.0)
            scores[c] += self.motif_w * self.specific_boosts.get(c, 0.0)

        tot=sum(scores.values()) or 1.0
        ranked=[(c,s/tot) for c,s in scores.items()]
        ranked.sort(key=lambda x:x[1],reverse=True)
        self.last_top_pred = ranked[0][0] if ranked else None
        self.last_why = {
            "prefix": list(best_prefix) if best_prefix else [],
            "prefix_title": list(best_prefix_t) if best_prefix_t else [],
            "context": {"hour":ctx[0],"weekday":ctx[1],"month":ctx[2],
                        "cpu_bucket":ctx[3],"gpu_bucket":ctx[4],"net_bucket":ctx[5],"io_bucket":ctx[6],
                        "routine_code":ctx[7],"title_code":ctx[8]},
            "n": best_n, "n_title": best_nt,
            "baseline_spike": round(spike, 3)
        }
        return ranked[:top_k]

    def feedback(self, predicted: Optional[str], actual: Optional[str], sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str]):
        if not predicted or not actual: return
        ctx=self._context(sysd, gpud, netd, routine, win_title); seq=list(self.window); seqt=[t for t,_ in list(self.title_window)]
        for n in range(1,min(self.max_n,len(seq))+1):
            prefix=tuple(seq[-n:]); key=(ctx,n)
            if predicted==actual:
                self.ctx_counts[key][(prefix,actual)]+=2
            else:
                wrong=(prefix,predicted)
                self.ctx_counts[key][wrong]=max(0,self.ctx_counts[key][wrong]-1)
        for n in range(1,min(self.max_n,len(seqt))+1):
            prefix_t=tuple(seqt[-n:]); keyt=(ctx,n)
            if predicted==actual:
                self.title_ctx_counts[keyt][(prefix_t,actual)]+=2
            else:
                wrong_t=(prefix_t,predicted)
                self.title_ctx_counts[keyt][wrong_t]=max(0,self.title_ctx_counts[keyt][wrong_t]-1)

    def export(self, max_ctx: int) -> Dict[str, Any]:
        def pack_counts(store: Dict[Any, Counter], limit: int) -> Dict[str, int]:
            flat = Counter()
            for k, c in store.items():
                for kk, vv in c.items():
                    flat[(k, kk)] += vv
            top = flat.most_common(limit)
            out = {}
            for (ctx, pair), v in top:
                try:
                    ctx_s = ",".join(str(x) for x in ctx)
                    pref_s = "|".join(pair[0]) if isinstance(pair[0], tuple) else str(pair[0])
                    out[f"{ctx_s}::{pref_s}::{pair[1]}"] = int(v)
                except Exception:
                    continue
            return out
        return {"ctx_counts": pack_counts(self.ctx_counts, max_ctx//2),
                "title_ctx_counts": pack_counts(self.title_ctx_counts, max_ctx//2),
                "total_counts": {k:int(v) for k,v in self.total_counts.items()}}

    def import_state(self, data: Dict[str, Any]):
        try:
            for k, v in data.get("total_counts", {}).items():
                self.total_counts[k] = int(v)
            def unpack_counts(flat: Dict[str, int], target: Dict[Any, Counter]):
                for ks, v in flat.items():
                    try:
                        ctx_s, pref_s, app = ks.split("::", 2)
                        ctx = tuple(int(x) for x in ctx_s.split(","))
                        pref = tuple(pref_s.split("|")) if "|" in pref_s else tuple([pref_s])
                        target[(ctx, len(pref))][(pref, app)] = int(v)
                    except Exception:
                        continue
            unpack_counts(data.get("ctx_counts", {}), self.ctx_counts)
            unpack_counts(data.get("title_ctx_counts", {}), self.title_ctx_counts)
        except Exception:
            pass

# ======================================= Preloader (micro-phased, elastic) ============
class Preloader:
    def __init__(self, log_q: queue.Queue, policy: PolicyEngine, user_name: Optional[str] = None, io_heatmap: Optional[IOHeatmap] = None):
        self.log_q = log_q; self.policy = policy; self.cancel = threading.Event()
        self.b4b_paths = guess_b4b_paths(user_name=user_name)
        self.io_heatmap = io_heatmap
    def _log(self, msg): self.log_q.put({"type": "thought", "msg": msg})

    def _recent_files(self, root: str, limit: int = 50) -> List[str]:
        items = []
        try:
            for base, _, files in os.walk(root):
                for fn in files:
                    fp = os.path.join(base, fn)
                    try:
                        st = os.stat(fp)
                        items.append((st.st_mtime_ns, st.st_size, fp))
                    except Exception:
                        continue
        except Exception:
            pass
        items.sort(reverse=True)
        return [fp for _,_,fp in items[:limit]]

    def targets_for(self, top: str) -> Dict[str, List[str]]:
        top_l = (top or "").lower()
        gaming = any(k in top_l for k in ["back4blood","back4blood.exe","valorant","fortnite","elden ring","steam","steam.exe","anticheat","easyanticheat","overlay"])
        if gaming:
            shader_dirs = self.b4b_paths.get("shader_cache", [])[:2]
            assets_dirs = self.b4b_paths.get("game_assets", [])[:1]
            prof_dirs = self.b4b_paths.get("profiles", [])[:2]
            return {
                "auth": ["auth_session"],
                "services": ["overlay","easyanticheat"],
                "shader_hot": shader_dirs,
                "shader_cold": shader_dirs,
                "assets": assets_dirs,
                "secondary": ["profiles"] + prof_dirs
            }
        return {"auth": ["browser_profile","dns_tls"], "services": ["lib_pages"], "shader_hot": [], "shader_cold": [], "assets": [], "secondary": []}

    def stage(self, phases: Dict[str, List[str]], sysd: Dict[str, Any], gpud: Dict[str, Any], temperature: float, stable: bool = False):
        ceilings = self.policy.ceilings()
        cpu = sysd.get("cpu_pct") or 0
        gpu = gpud.get("util_pct") or 0
        if cpu >= ceilings["cpu_pct"] or (gpud.get("gpu_enabled") and gpu >= ceilings["gpu_pct"]):
            self._log("Preload skipped (near CPU/GPU ceiling)."); return
        base_budget = ceilings.get("preload_budget_mb_min", 160)
        elastic = 1.0 + (0.35 if stable else 0.0)
        scale = elastic + min(0.75, 0.25 * max(0.0, temperature))
        budget = int(base_budget * scale)

        phase_f = self.policy.preloader_params().get("phase_budgets", {"auth":0.1,"services":0.15,"shader_hot":0.22,"shader_cold":0.18,"assets":0.25,"secondary":0.1})
        temps = self.policy.preloader_params().get("temp_thresholds", {"shader_hot":0.40,"shader_cold":0.55,"assets":0.60})

        def expand_items(items: List[str]) -> List[str]:
            order = []
            for t in items:
                if os.path.isdir(t):
                    order += self._recent_files(t, limit=40)
                else:
                    order.append(t)
            return order

        heat = self.io_heatmap.sample() if self.io_heatmap else {"shader": {}, "profiles": {}, "assets": {}}
        def heat_weight(paths: List[str], cls: str) -> float:
            store = heat.get(cls, {})
            return sum(store.get((p or "").lower(), 0.0) for p in paths)

        total_budget = float(budget)
        for phase, items in phases.items():
            if phase == "shader_hot" and temperature < temps.get("shader_hot", 0.40): 
                self._log("Gating shader_hot (temp below)."); continue
            if phase == "shader_cold" and temperature < temps.get("shader_cold", 0.55): 
                self._log("Gating shader_cold (temp below)."); continue
            if phase == "assets" and temperature < temps.get("assets", 0.60): 
                self._log("Gating assets (temp below)."); continue

            share = phase_f.get(phase, 0.1)
            hw = 0.0
            if phase.startswith("shader"):
                hw = heat_weight(items, "shader")
            elif phase == "assets":
                hw = heat_weight(items, "assets")
            elif phase == "secondary":
                hw = heat_weight(items, "profiles")
            if hw > 0:
                share = min(0.6, share + min(0.18, hw / 20.0))

            phase_budget = int(total_budget * share)
            order = expand_items(items)
            self._log(f"Preloading '{phase}' with budget {phase_budget} MB/min (temp={round(temperature,2)}, heat={round(hw,2)})")
            for item in order:
                if self.cancel.is_set(): break
                self._log(f"Preloading (symbolic): {item}")
                time.sleep(0.04)

# ===================================== Decision Engine =================================
class DecisionEngine:
    def __init__(self, log_q: queue.Queue, policy: PolicyEngine):
        self.log_q = log_q; self.policy = policy
    def _log(self, msg): self.log_q.put({"type": "thought", "msg": msg})
    def risk(self, sysd: Dict[str, Any], gpud: Dict[str, Any]) -> Dict[str, Any]:
        cpu = sysd.get("cpu_pct") or 0
        gpu = gpud.get("util_pct") or 0
        ceilings = self.policy.ceilings()
        low = cpu < ceilings["cpu_pct"] and (not gpud.get("gpu_enabled") or gpu < ceilings["gpu_pct"])
        return {"cpu": cpu, "gpu": gpu, "risk_low": low}

# ======================================= Graph Store ==================================
class GraphStore:
    def __init__(self): self.G = nx.Graph() if nx else None
    def update(self, sysd: Dict[str, Any], netd: Dict[str, Any], gpud: Dict[str, Any], routine: Optional[str],
               temperature: float, intent_details: Dict[str, float], baseline_deltas: Dict[str, float], focus: Optional[str], why_ext: Dict[str, Any], win_title: Optional[str]):
        if not self.G: return
        self.G.add_node("system", cpu=sysd.get("cpu_pct"), procs=sysd.get("proc_count"))
        self.G.add_node("network", conns=len(netd.get("connections", [])), cdn_burst=netd.get("cdn_burst"))
        self.G.add_node("gpu", util=gpud.get("util_pct"), mem_used=gpud.get("mem_used_mb"))
        self.G.add_node("routine", label=routine)
        self.G.add_node("temperature", value=round(temperature, 2))
        self.G.add_node("intent", **intent_details)
        self.G.add_node("baseline", **{k: round(v,2) for k, v in baseline_deltas.items()})
        self.G.add_node("focus", name=focus or "", title=win_title or "")
        self.G.add_node("why_ext", **why_ext)
        self.G.add_edge("system", "network"); self.G.add_edge("system", "gpu")
        self.G.add_edge("system", "routine"); self.G.add_edge("system", "temperature")
        self.G.add_edge("system", "intent"); self.G.add_edge("system", "baseline")
        self.G.add_edge("system", "focus"); self.G.add_edge("system", "why_ext")
    def snapshot(self) -> Dict[str, Any]:
        if not self.G: return {"graph_enabled": False}
        return {"nodes": list(self.G.nodes(data=True)), "edges": list(self.G.edges())}

# ========================================= AI Assist ==================================
class AIAssist:
    def __init__(self, log_q: queue.Queue, policy: PolicyEngine, learner: PredictiveLearner):
        self.log_q = log_q; self.policy = policy; self.learner = learner
        self.hits_games = Counter(); self.misses_games = Counter()
        self.hits_apps = Counter(); self.misses_apps = Counter()
        self.latency_obs = deque(maxlen=100)
        self.auto_candidates = Counter()
        self.last_intent_trigger: Optional[str] = None
        self.last_intent_details: Dict[str, float] = {}
        self.iso_bins = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
        self.iso_map  = [0.02, 0.06, 0.12, 0.22, 0.36, 0.52, 0.72, 0.86, 0.98]
        self.soft_bans_by_ctx = {}  # ctx_hash -> {target: expiry}
    def _log(self, msg, kind="thought"): self.log_q.put({"type": kind, "msg": msg})
    def classify(self, name: str) -> str:
        nm = (name or "").lower()
        if any(k in nm for k in ["back4blood","game","steam","steam.exe","epic","anticheat","easyanticheat","valorant","fortnite","elden ring","overlay"]):
            return "game"
        return "app"
    def calibrated_conf(self, target: str, raw_conf: float) -> float:
        pool = self.classify(target)
        hits = self.hits_games if pool=="game" else self.hits_apps
        misses = self.misses_games if pool=="game" else self.misses_apps
        a0, b0 = 2.0, 2.0
        a = a0 + float(hits.get(target, 0))
        b = b0 + float(misses.get(target, 0))
        posterior_mean = a / (a + b)
        return 0.5 * raw_conf + 0.5 * posterior_mean
    def isotonic(self, p: float) -> float:
        for i in range(1, len(self.iso_bins)):
            if p <= self.iso_bins[i]:
                x0, x1 = self.iso_bins[i-1], self.iso_bins[i]
                y0, y1 = self.iso_map[i-1], self.iso_map[i]
                t = (p - x0) / (x1 - x0 + 1e-9)
                return y0 + t * (y1 - y0)
        return self.iso_map[-1]
    def observe_prediction(self, top: Optional[str], conf: float, why: Dict[str, Any], routine: Optional[str], temperature: float, intent_details: Optional[Dict[str, float]] = None, why_ext: Optional[Dict[str, Any]] = None):
        if top:
            cal_conf = self.calibrated_conf(top, conf)
            cal_conf = 0.7 * cal_conf + 0.3 * self.isotonic(conf)
            msg = f"AI Assist: predicted {top} ({int(cal_conf*100)}%), routine={routine}, temp={round(temperature,2)}, ctx={why.get('context',{})}, baseline_spike={why.get('baseline_spike',0)}"
            if intent_details: msg += f", intent={intent_details}"
            if why_ext: msg += f", why_ext={why_ext}"
            self._log(msg)
    def record_outcome(self, predicted: Optional[str], actual: Optional[str], latency_ms: Optional[int] = None, ctx_hash: Optional[str] = None):
        if latency_ms is not None: self.latency_obs.append(latency_ms)
        if predicted and ctx_hash:
            pool = self.classify(predicted)
            if actual and predicted == actual:
                (self.hits_games if pool=="game" else self.hits_apps)[predicted] += 1
            else:
                (self.misses_games if pool=="game" else self.misses_apps)[predicted] += 1
                bans = self.soft_bans_by_ctx.setdefault(ctx_hash, {})
                # TTL buckets based on temp/routine would be supplied by orchestrator context if needed
                ttl_params = DEFAULT_POLICY["soft_bans"]
                expiry = time.time() + ttl_params.get("ttl_medium_temp_s", 1200)
                bans[predicted] = expiry
    def suggest_policy(self, routine: Optional[str]) -> Optional[str]:
        if not self.latency_obs: return None
        avg = sum(self.latency_obs)/len(self.latency_obs)
        ja = self.policy.jump_ahead()
        if routine == "gaming" and avg > 2200 and ja < 4:
            return f"Increase jump-ahead to {ja+1}s for gaming routine (avg launch {int(avg)}ms)."
        if routine == "work" and avg < 1400 and ja > 1:
            return f"Reduce jump-ahead to {ja-1}s for work routine (avg launch {int(avg)}ms)."
        return None
    def auto_target(self, sysd: Dict[str, Any], gpud: Dict[str, Any], focus_name: Optional[str]):
        if not focus_name: return
        thr = self.policy.auto_target_thresholds()
        cpu = sysd.get("cpu_pct") or 0; gpu = gpud.get("util_pct") or 0
        hour = time.localtime().tm_hour; is_evening = hour >= 18 or hour <= 1
        bonus = thr.get("evening_hours_bonus", 1) if is_evening else 0
        increment = 1
        if self.classify(focus_name) == "game":
            increment += bonus
            if gpud.get("gpu_enabled") and gpu >= thr.get("gpu_pct", 25): increment += 1
        name = focus_name.lower()
        self.auto_candidates[name] += increment
        hit_min = max(2, thr.get("hit_count_min", 3) - (1 if is_evening else 0))
        if (cpu >= thr.get("cpu_pct", 12) or (gpud.get("gpu_enabled") and gpu >= thr.get("gpu_pct", 25))) and self.auto_candidates[name] >= hit_min:
            aw = self.policy.always_warm()
            if name not in aw["apps"] and name not in aw["games"]:
                if self.classify(name) == "game": aw["games"].append(name)
                else: aw["apps"].append(name)
                self.policy.policy["always_warm"] = aw
                self._log(f"AI Assist: auto-targeted '{name}' into always_warm.")
                try: self.policy.save()
                except Exception: pass
    def intent_trigger(self, source: str, candidates: List[str], strength: float, details: Dict[str, float]):
        self.last_intent_trigger = source
        self.last_intent_details = details
        self._log(f"AI Assist: intent trigger from {source} with details {details}")
        self.learner.add_intent_boost(candidates, strength=strength)

# ======================================= Persistence store =============================
class PersistenceStore:
    def __init__(self, path: str, interval_s: int, max_edges: int, max_ctx: int, log_q: queue.Queue):
        self.path = path
        self.interval_s = interval_s
        self.max_edges = max_edges
        self.max_ctx = max_ctx
        self.log_q = log_q
        self.last_save = 0.0
    def _log(self, msg): self.log_q.put({"type": "thought", "msg": msg})
    def save(self, learner: PredictiveLearner, ancestry: AncestryAgent, lattice: LaunchLattice):
        now = time.time()
        if now - self.last_save < self.interval_s: return
        state = {
            "ancestry_edges": ancestry.export(self.max_edges),
            "learner": learner.export(self.max_ctx),
            "lattice": {tgt: {var: vals[-10:] for var, vals in v.items()} for tgt, v in lattice.edges.items()},
            "timestamp": int(now)
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(state, f)
            self.last_save = now
            self._log("State saved.")
        except Exception as e:
            self._log(f"State save failed: {e}")
    def load(self) -> Optional[Dict[str, Any]]:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log(f"State load failed: {e}")
        return None

# ======================================= Orchestrator =================================
class Orchestrator(threading.Thread):
    def __init__(self, log_q: queue.Queue, ui_q: queue.Queue, policy: PolicyEngine, user_name: Optional[str] = None):
        super().__init__(daemon=True)
        self.log_q = log_q; self.ui_q = ui_q; self.policy = policy
        self.system = SystemAgent(); self.network = NetworkAgent(); self.gpu = GPUAgent()
        self.foreground = ForegroundAgent()
        self.ancestry = AncestryAgent()
        self.services = ServiceAgent(window_s=12.0)
        self.lattice = LaunchLattice()
        self.io_heat = IOHeatmap()
        self.baselines = BaselineTracker(window=90, decay=0.99)
        pred = self.policy.prediction_params()
        self.cp_cpu = ChangePointDetector(k=pred.get("changepoint_sensitivity", {}).get("cpu", 2.5), window=80, cooldown_s=6.0)
        self.cp_io  = ChangePointDetector(k=pred.get("changepoint_sensitivity", {}).get("io", 2.5),  window=80, cooldown_s=6.0)
        self.cp_net = ChangePointDetector(k=pred.get("changepoint_sensitivity", {}).get("net", 2.5), window=80, cooldown_s=6.0)
        self.cp_gpu = ChangePointDetector(k=pred.get("changepoint_sensitivity", {}).get("gpu", 2.3), window=80, cooldown_s=6.0)
        self.cp_ensemble = CPEnsemble(window_s=8.0)

        self.learner = PredictiveLearner(log_q, policy, window_size=12, max_n=5)
        self.preloader = Preloader(log_q, policy, user_name=user_name, io_heatmap=self.io_heat)
        self.decider = DecisionEngine(log_q, policy)
        self.graph = GraphStore()
        self.assist = AIAssist(log_q, policy, self.learner)
        rp = self.policy.routine_params()
        self.clusterer = RoutineClusterer(window_size=rp.get("window_size", 20), min_support=rp.get("min_support", 3))
        self.motif = MotifBoosts(max_len=20, decay=self.policy.prediction_params().get("recency_decay", 0.997))
        self.cold_start = pred.get("cold_start_boosts", [])
        self.grace_until = 0.0

        # Persistence
        per = self.policy.persistence_params()
        self.persist = PersistenceStore(per.get("path","scanner_state.json"), per.get("interval_s",120), per.get("max_edges",8000), per.get("max_ctx",6000), log_q)
        state = self.persist.load()
        if state:
            self._log("Loading persisted state ...")
            try:
                self.ancestry.import_edges(state.get("ancestry_edges", {}))
                self.learner.import_state(state.get("learner", {}))
                for tgt, varmap in state.get("lattice", {}).items():
                    self.lattice.edges[tgt] = {var: list(vals) for var, vals in varmap.items()}
            except Exception as e:
                self._log(f"State import failed: {e}")

        # Disk watcher by class
        paths = {
            "game_assets": self.preloader.b4b_paths.get("game_assets", [])[:1],
            "shader_cache": self.preloader.b4b_paths.get("shader_cache", [])[:2],
            "profiles": self.preloader.b4b_paths.get("profiles", [])[:2]
        }
        self.disk_watcher = DiskWatcher(log_q, paths, interval=1.0)

        self.running = True
        self.top_history = deque(maxlen=5)
        self.timeline = deque(maxlen=60)  # last 60s intent signals

    def _log(self, msg, kind="thought"): self.log_q.put({"type": kind, "msg": msg})
    def stop(self):
        self.running = False
        self.disk_watcher.stop()
        self._log("Orchestrator stopping.")

    def _context_hash(self, routine: Optional[str], why_ctx: Dict[str, Any], net_geom: Dict[str, Any]) -> str:
        hour = why_ctx.get("hour"); wk = why_ctx.get("weekday"); title_code = why_ctx.get("title_code")
        width = net_geom.get("width", 0); height_key = sum(net_geom.get("height", {}).values()) if net_geom.get("height") else 0
        s = f"{hour}-{wk}-{title_code}-{routine}-{width}-{height_key}"
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def _intent_score(self, netd: Dict[str, Any], gpud: Dict[str, Any], disk_events_count: int, motif_hits: int, baseline_deltas: Dict[str, float], service_cluster: int) -> Tuple[float, Dict[str, float]]:
        geometry = netd.get("geometry", {"width":0, "height": {}, "persist": {"width":0, "height": {}}})
        width_persist = float(geometry.get("persist", {}).get("width", 0.0))
        height_persist = geometry.get("persist", {}).get("height", {})
        cdn_fams = ["steam","cloudfront","akamai","epic/eac"]
        height = float(sum(height_persist.get(k, 0.0) for k in cdn_fams))
        net_geom_score = min(1.0, 0.4 * (width_persist / 20.0) + 0.6 * (height / 12.0))
        gpu_uptick = float((gpud.get("util_pct") or 0) / 100.0)
        disk = float(disk_events_count)
        motif = float(motif_hits)
        spike = 0.35*baseline_deltas.get("cpu",0) + 0.3*baseline_deltas.get("io",0) + 0.25*baseline_deltas.get("net",0) + 0.2*baseline_deltas.get("gpu",0)
        score = min(4.0, 1.0*net_geom_score + 0.9*disk + 0.6*gpu_uptick + 0.6*motif + 0.45*service_cluster + 0.5*spike)
        details = {"net_width_persist": width_persist, "net_height_persist": height, "net_geom": round(net_geom_score,2), "disk": disk, "gpu": round(gpu_uptick, 2),
                   "motif": motif, "service_cluster": service_cluster, "spike": round(spike,2), "score": round(score,2)}
        return score, details

    def _temperature(self, intent_score: float, details: Dict[str, float], rising_edge: float) -> float:
        netg = details.get("net_geom", 0.0)
        svc = min(1.0, details.get("service_cluster", 0) / 4.0)
        spike = min(1.0, details.get("spike", 0.8) / 1.2)
        base = min(1.0, intent_score / 3.5)
        temp = 0.28*base + 0.24*netg + 0.18*svc + 0.16*spike + 0.14*rising_edge
        return max(0.0, min(1.0, temp))

    def run(self):
        self._log("Orchestrator started.")
        self.disk_watcher.start()
        while self.running:
            try:
                sysd = self.system.sample()
                gpud = self.gpu.sample()
                netd = self.network.sample()

                self.baselines.record(sysd, gpud, netd)
                baseline_deltas = self.baselines.deltas()

                self.cp_cpu.record(sysd.get("cpu_pct") or 0.0)
                self.cp_gpu.record((gpud.get("util_pct") or 0.0) if gpud.get("gpu_enabled") else 0.0)
                self.cp_net.record(len(netd.get("connections", [])))
                io_mb = (sysd.get("disk_io", {}).get("read_bytes", 0) + sysd.get("disk_io", {}).get("write_bytes", 0)) / 1_000_000.0
                self.cp_io.record(io_mb)
                hit_net = self.cp_net.is_change(); hit_io = self.cp_io.is_change(); hit_gpu = self.cp_gpu.is_change()
                self.cp_ensemble.record("net", hit_net); self.cp_ensemble.record("io", hit_io); self.cp_ensemble.record("gpu", hit_gpu)
                cp_strong = self.cp_ensemble.strong_intent()

                disk_events = 0
                while True:
                    try:
                        item = self.log_q.get_nowait()
                    except queue.Empty:
                        break
                    if item.get("type") == "disk_event":
                        d = item.get("msg", ""); cls = item.get("class", None)
                        disk_events += 1
                        if d: self.io_heat.record_dir_touch(d, cls=cls)
                    else:
                        self.ui_q.put({"log_passthrough": item})

                self.ancestry.record()
                service = self.services.sample()

                focus, win_title, title_stack = self.foreground.sample_focus()
                if focus:
                    self.clusterer.record(focus)
                    self.motif.record(focus)
                    self.lattice.record_event(focus)
                routine = self.clusterer.routine_label()

                if focus:
                    self.learner.record(focus, sysd, gpud, netd, routine, win_title, baseline_deltas=baseline_deltas, title_stack=title_stack)
                    self.assist.auto_target(sysd, gpud, focus)

                motif_hits = self.motif.hits_recent()
                rising_edge_b4b = self.lattice.rising_edge("back4blood")
                intent_score, intent_details = self._intent_score(netd, gpud, disk_events, motif_hits, baseline_deltas, service_cluster=service.get("cluster", 0))
                temperature = self._temperature(intent_score, intent_details, rising_edge=rising_edge_b4b)

                aw = self.policy.always_warm()
                all_cands = list(set(aw.get("apps", []) + aw.get("games", [])))
                game_cands = [c for c in all_cands if any(k in c for k in ["back4blood","steam","epic","anticheat","easyanticheat","overlay","game"])]

                if intent_score >= 1.0:
                    self.assist.intent_trigger("multi_signal_intent", game_cands, strength=min(0.25, 0.06 * intent_score), details=intent_details)

                motif_boost_map = self.motif.next_boosts(list(self.motif.window), game_cands, max_strength=0.25)
                if motif_boost_map: self.learner.add_specific_boosts(motif_boost_map)

                ancestry_boosts = self.ancestry.next_boosts(focus, game_cands, max_strength=0.25)
                if ancestry_boosts: self.learner.add_specific_boosts(ancestry_boosts)

                if routine == "gaming":
                    candidates = game_cands + ["overlay","easyanticheat"]
                else:
                    block = ["back4blood","back4blood.exe","valorant","fortnite","elden ring","overlay","anticheat","easyanticheat"]
                    candidates = [c for c in all_cands if c not in block]
                if len(candidates) < 3:
                    backfill = sorted(all_cands, key=lambda x: self.learner.total_counts.get(x,0), reverse=True)[:5]
                    for b in backfill:
                        if b not in candidates: candidates.append(b)

                if sum(self.learner.total_counts.values()) < 20 and self.cold_start:
                    for c in self.cold_start:
                        if c in all_cands:
                            self.learner.add_specific_boosts({c: 0.15})

                preds = self.learner.predict(sysd, gpud, netd, routine, win_title, candidates, top_k=8)
                ja = self.policy.jump_ahead()
                risk = self.decider.risk(sysd, gpud)

                ctx_hash = self._context_hash(routine, self.learner.last_why.get("context", {}), netd.get("geometry", {}))
                bans = self.assist.soft_bans_by_ctx.get(ctx_hash, {})
                now_time = time.time()
                candidates = [c for c in candidates if not (c in bans and bans[c] > now_time)]

                explore_ratio = self.policy.prediction_params().get("explore_ratio", 0.18)
                explore_ratio = min(0.4, explore_ratio * (1.0 + 1.2 * temperature))
                explore_pool = [c for c in all_cands if c not in candidates]
                explorer = None
                if explore_pool and risk["risk_low"] and (len(candidates) < int(8 * (1.0 + explore_ratio))):
                    explorer = max(explore_pool, key=lambda x: self.learner.total_counts.get(x, 0), default=None)
                    if explorer and explorer not in candidates:
                        candidates.append(explorer)
                        preds = self.learner.predict(sysd, gpud, netd, routine, win_title, candidates, top_k=8)

                hyst = self.policy.prediction_params().get("hysteresis", {"stable_frames":3,"high_conf":0.46,"medium_conf_with_temp":0.40,"grace_period_s":6.0,"temp_floor":0.55})
                proceed = False; stable = False
                if preds:
                    top, conf = preds[0]
                    self.top_history.append(top)
                    stable = len(self.top_history) >= hyst.get("stable_frames", 3) and len(set(self.top_history)) == 1
                    high_conf = conf >= hyst.get("high_conf", 0.46)
                    med_conf_temp = conf >= hyst.get("medium_conf_with_temp", 0.40) and temperature >= 0.6
                    in_grace = time.time() < self.grace_until and temperature >= hyst.get("temp_floor", 0.55)
                    strong_cp = cp_strong
                    proceed = stable or high_conf or med_conf_temp or in_grace or strong_cp
                    if proceed: self.grace_until = time.time() + hyst.get("grace_period_s", 6.0)

                self.timeline.append({
                    "ts": int(time.time()),
                    "cdn_burst": netd.get("cdn_burst", 0),
                    "net_width_persist": intent_details.get("net_width_persist", 0),
                    "service_cluster": intent_details.get("service_cluster", 0),
                    "cp_strong": int(cp_strong),
                    "disk_events": disk_events,
                    "temperature": round(temperature, 2)
                })

                why_ext = {
                    "explore_ratio": explore_ratio, "stable_top": stable,
                    "net_geometry": netd.get("geometry", {}),
                    "disk_events": disk_events,
                    "cp_strong": int(cp_strong),
                    "service_cluster": intent_details.get("service_cluster", 0),
                    "win_title": win_title,
                    "temperature": round(temperature, 2),
                    "ctx_hash": ctx_hash,
                    "rising_edge_b4b": round(rising_edge_b4b, 2)
                }
                self.graph.update(sysd, netd, gpud, routine, temperature, intent_details, baseline_deltas, focus, why_ext, win_title)

                if preds:
                    top, conf = preds[0]
                    if proceed and risk["risk_low"]:
                        phases = self.preloader.targets_for(top)
                        self._log(f"Prediction: {top} (conf {int(conf*100)}%), routine={routine}, focus={focus}, title={win_title}, temp={round(temperature,2)}, jump-ahead {ja}s")
                        self._log(f"Why: {json.dumps(self.learner.last_why)}; Why++: {json.dumps(why_ext)}")
                        self.assist.observe_prediction(top, conf, self.learner.last_why, routine, temperature, intent_details=intent_details, why_ext=why_ext)
                        self.preloader.stage(phases, sysd, gpud, temperature=temperature, stable=stable)
                        self.lattice.observe_curve(top)
                    else:
                        self._log("Holding preload (not stable or low confidence/temperature).")

                self.persist.save(self.learner, self.ancestry, self.lattice)

                suggestion = self.assist.suggest_policy(routine)
                if suggestion: self._log(f"AI Assist: {suggestion}")

                payload = {
                    "sys": sysd,
                    "gpu": gpud,
                    "net": {"count": len(netd.get("connections", [])), "cdn_burst": netd.get("cdn_burst", 0), "height_by_fam": netd.get("height_by_fam", {})},
                    "graph": self.graph.snapshot(),
                    "predictions": preds,
                    "risk": risk,
                    "ceilings": self.policy.ceilings(),
                    "jump_ahead": ja,
                    "why": self.learner.last_why,
                    "routine": routine,
                    "assist": {"suggestion": suggestion, "disk_trigger": self.assist.last_intent_trigger, "intent_details": intent_details, "why_ext": why_ext},
                    "candidates": candidates,
                    "focus": focus,
                    "win_title": win_title,
                    "temperature": temperature,
                    "timeline": list(self.timeline)
                }
                self.ui_q.put(payload)
                time.sleep(0.8)
            except Exception as e:
                self._log(f"Orchestrator error: {e}", kind="anomaly")
                time.sleep(1.0)

    def feedback_approve(self, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str]):
        ctx_hash = hashlib.sha1(json.dumps(self.learner.last_why.get("context", {})).encode()).hexdigest()[:12]
        self.learner.feedback(self.learner.last_top_pred, self.learner.last_top_pred, sysd, gpud, netd, routine, win_title)
        self.assist.record_outcome(self.learner.last_top_pred, self.learner.last_top_pred, latency_ms=None, ctx_hash=ctx_hash)
        self._log("Feedback: approved top prediction.")
    def feedback_deny(self, sysd: Dict[str, Any], gpud: Dict[str, Any], netd: Dict[str, Any], routine: Optional[str], win_title: Optional[str]):
        ctx_hash = hashlib.sha1(json.dumps(self.learner.last_why.get("context", {})).encode()).hexdigest()[:12]
        self.learner.feedback(self.learner.last_top_pred, None, sysd, gpud, netd, routine, win_title)
        self.assist.record_outcome(self.learner.last_top_pred, None, latency_ms=None, ctx_hash=ctx_hash)
        self._log("Feedback: denied top prediction (context-sliced soft-ban applied).")

# ============================================ GUI =====================================
class ScannerGUI:
    def __init__(self, root: tk.Tk, inventory: InventorySnapshot, policy: PolicyEngine,
                 orchestrator: Orchestrator, autoloader: AutoLoader):
        self.root = root; self.inventory = inventory; self.policy = policy
        self.orch = orchestrator; self.loader = autoloader
        self.log_q = self.orch.log_q; self.ui_q = self.orch.ui_q

        root.title("Predictive Scanner (Lattice + Geometry Persistence + Classed IO Heat + TTL Soft-bans + CP Quorum)")
        root.geometry("1480x1080")

        hdr = ttk.Frame(root); hdr.pack(fill="x", padx=8, pady=6)
        ttk.Button(hdr, text="Stop", command=self.stop).pack(side="left", padx=6)
        ttk.Button(hdr, text="Save policy", command=self.save_policy).pack(side="left", padx=6)

        ttk.Label(hdr, text="Jump-ahead (s)").pack(side="left", padx=8)
        self.var_jump = tk.IntVar(value=self.policy.jump_ahead())
        self.spin_jump = ttk.Spinbox(hdr, from_=0, to=10, textvariable=self.var_jump, width=5, command=self.update_jump)
        self.spin_jump.pack(side="left")

        ttk.Label(hdr, text="CPU ceiling (%)").pack(side="left", padx=8)
        self.var_cpu = tk.IntVar(value=self.policy.ceilings()["cpu_pct"])
        self.spin_cpu = ttk.Spinbox(hdr, from_=5, to=95, textvariable=self.var_cpu, width=5, command=self.update_cpu)
        self.spin_cpu.pack(side="left")

        ttk.Label(hdr, text="GPU ceiling (%)").pack(side="left", padx=8)
        self.var_gpu = tk.IntVar(value=self.policy.ceilings()["gpu_pct"])
        self.spin_gpu = ttk.Spinbox(hdr, from_=10, to=100, textvariable=self.var_gpu, width=5, command=self.update_gpu)
        self.spin_gpu.pack(side="left")

        nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_dash = ttk.Frame(nb); nb.add(self.tab_dash, text="Dashboard")
        self.tab_predict = ttk.Frame(nb); nb.add(self.tab_predict, text="Predictions")
        self.tab_policy = ttk.Frame(nb); nb.add(self.tab_policy, text="Policy")
        self.tab_candidates = ttk.Frame(nb); nb.add(self.tab_candidates, text="Candidates")
        self.tab_thoughts = ttk.Frame(nb); nb.add(self.tab_thoughts, text="Thoughts")
        self.tab_inventory = ttk.Frame(nb); nb.add(self.tab_inventory, text="Inventory")

        left = ttk.Frame(self.tab_dash); left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right = ttk.Frame(self.tab_dash); right.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        self.metrics = ttk.Treeview(left, columns=("metric","value"), show="headings")
        self.metrics.heading("metric", text="Metric"); self.metrics.heading("value", text="Value")
        self.metrics.pack(fill="both", expand=True)
        ttk.Label(right, text="Graph snapshot").pack(anchor="w")
        self.text_graph = tk.Text(right, height=12); self.text_graph.pack(fill="both", expand=True)
        ttk.Label(right, text="Intent timeline (60s)").pack(anchor="w")
        self.text_timeline = tk.Text(right, height=12); self.text_timeline.pack(fill="x")

        pred_top = ttk.Frame(self.tab_predict); pred_top.pack(fill="x", padx=6, pady=6)
        ttk.Label(pred_top, text="Top predicted targets with dual-context ‘why’ and temperature").pack(anchor="w")
        self.tree_preds = ttk.Treeview(self.tab_predict, columns=("target","confidence"), show="headings")
        self.tree_preds.heading("target", text="Target"); self.tree_preds.heading("confidence", text="Confidence")
        self.tree_preds.pack(fill="both", expand=True, padx=6, pady=6)

        self.text_why = tk.Text(self.tab_predict, height=12); self.text_why.pack(fill="x", padx=6, pady=6)
        self.text_assist = tk.Text(self.tab_predict, height=10); self.text_assist.pack(fill="x", padx=6, pady=6)

        pred_actions = ttk.Frame(self.tab_predict); pred_actions.pack(fill="x", padx=6, pady=6)
        ttk.Button(pred_actions, text="Approve top prediction", command=self.approve_top).pack(side="left")
        ttk.Button(pred_actions, text="Deny top prediction", command=self.deny_top).pack(side="left", padx=6)

        self.text_policy = tk.Text(self.tab_policy); self.text_policy.pack(fill="both", expand=True, padx=6, pady=6)
        self.load_policy_editor()

        cand_top = ttk.Frame(self.tab_candidates); cand_top.pack(fill="x", padx=6, pady=6)
        ttk.Label(cand_top, text="Warm candidates (apps + games)").pack(anchor="w")
        self.tree_cands = ttk.Treeview(self.tab_candidates, columns=("name","type"), show="headings")
        self.tree_cands.heading("name", text="Name"); self.tree_cands.heading("type", text="Type")
        self.tree_cands.pack(fill="both", expand=True, padx=6, pady=6)

        cand_controls = ttk.Frame(self.tab_candidates); cand_controls.pack(fill="x", padx=6, pady=6)
        ttk.Label(cand_controls, text="Add candidate").pack(side="left", padx=6)
        self.entry_cand = ttk.Entry(cand_controls, width=30); self.entry_cand.pack(side="left")
        self.var_cand_type = tk.StringVar(value="apps")
        ttk.Combobox(cand_controls, textvariable=self.var_cand_type, values=["apps","games"], width=8).pack(side="left", padx=6)
        ttk.Button(cand_controls, text="Add", command=self.add_candidate).pack(side="left", padx=6)
        ttk.Button(cand_controls, text="Remove selected", command=self.remove_candidate).pack(side="left", padx=6)
        ttk.Button(cand_controls, text="Save candidates to policy", command=self.save_candidates).pack(side="left", padx=6)

        self.text_thoughts = tk.Text(self.tab_thoughts); self.text_thoughts.pack(fill="both", expand=True, padx=6, pady=6)
        self.text_inventory = tk.Text(self.tab_inventory); self.text_inventory.pack(fill="both", expand=True, padx=6, pady=6)
        self.text_inventory.insert("end", json.dumps(self.inventory.__dict__, indent=2))

        root.after(400, self.poll_updates)

    def stop(self): self.orch.stop()
    def save_policy(self):
        content = self.text_policy.get("1.0","end").strip()
        try:
            if yaml: self.policy.policy = yaml.safe_load(content)
            else: self.policy.policy = json.loads(content)
            self.policy.save()
            messagebox.showinfo("Policy", "Saved.")
        except Exception as e:
            messagebox.showerror("Policy error", str(e))
    def load_policy_editor(self):
        try:
            if self.policy.path and os.path.exists(self.policy.path):
                with open(self.policy.path, "r", encoding="utf-8") as f:
                    self.text_policy.delete("1.0","end"); self.text_policy.insert("end", f.read()); return
        except Exception:
            pass
        dump_str = ""
        try:
            if yaml: dump_str = yaml.safe_dump(self.policy.policy)
            else: dump_str = json.dumps(self.policy.policy, indent=2)
        except Exception:
            dump_str = json.dumps(self.policy.policy, indent=2)
        self.text_policy.delete("1.0","end"); self.text_policy.insert("end", dump_str)

    def update_jump(self):
        try:
            val = int(self.var_jump.get())
            self.policy.policy.setdefault("defaults", {})["jump_ahead_seconds"] = val
            self.log_q.put({"type":"thought","msg":f"Jump-ahead set to {val}s"})
        except Exception as e:
            messagebox.showerror("Jump-ahead error", str(e))
    def update_cpu(self):
        try:
            val = int(self.var_cpu.get())
            self.policy.policy.setdefault("defaults", {})["cpu_ceiling_pct"] = val
            self.log_q.put({"type":"thought","msg":f"CPU ceiling set to {val}%"})
        except Exception as e:
            messagebox.showerror("Ceiling error", str(e))
    def update_gpu(self):
        try:
            val = int(self.var_gpu.get())
            self.policy.policy.setdefault("defaults", {})["gpu_ceiling_pct"] = val
            self.log_q.put({"type":"thought","msg":f"GPU ceiling set to {val}%"})
        except Exception as e:
            messagebox.showerror("Ceiling error", str(e))

    def approve_top(self):
        sysd = getattr(self, "_last_sys", {"cpu_pct": 0})
        gpud = getattr(self, "_last_gpu", {"gpu_enabled": False})
        netd = {"connections": [], "count": getattr(self, "_last_net_count", 0)}
        routine = getattr(self, "_last_routine", None)
        title = getattr(self, "_last_title", None)
        self.orch.feedback_approve(sysd, gpud, netd, routine, title)
    def deny_top(self):
        sysd = getattr(self, "_last_sys", {"cpu_pct": 0})
        gpud = getattr(self, "_last_gpu", {"gpu_enabled": False})
        netd = {"connections": [], "count": getattr(self, "_last_net_count", 0)}
        routine = getattr(self, "_last_routine", None)
        title = getattr(self, "_last_title", None)
        self.orch.feedback_deny(sysd, gpud, netd, routine, title)

    def add_candidate(self):
        name = (self.entry_cand.get() or "").strip().lower()
        t = self.var_cand_type.get()
        if not name: return
        aw = self.policy.always_warm()
        if name in aw[t]:
            messagebox.showinfo("Candidates", f"'{name}' already in {t}."); return
        aw[t].append(name)
        self.policy.policy["always_warm"] = aw
        self.refresh_candidates_table(aw)
        self.log_q.put({"type":"thought","msg":f"Candidate added: {name} -> {t}"})
    def remove_candidate(self):
        sel = self.tree_cands.selection()
        if not sel: return
        aw = self.policy.always_warm()
        for item in sel:
            name, typ = self.tree_cands.item(item, "values")
            if name in aw[typ]: aw[typ].remove(name)
        self.policy.policy["always_warm"] = aw
        self.refresh_candidates_table(aw)
        self.log_q.put({"type":"thought","msg":"Selected candidates removed"})
    def save_candidates(self):
        try:
            self.policy.save()
            messagebox.showinfo("Candidates", "Saved to policy.")
        except Exception as e:
            messagebox.showerror("Candidates", str(e))

    def refresh_candidates_table(self, aw: Dict[str, List[str]]):
        for i in self.tree_cands.get_children(): self.tree_cands.delete(i)
        for name in sorted(set(aw.get("apps", []) + aw.get("games", []))):
            typ = "games" if name in aw.get("games", []) else "apps"
            self.tree_cands.insert("", "end", values=(name, typ))

    def poll_updates(self):
        while True:
            try:
                item = self.log_q.get_nowait()
            except queue.Empty:
                break
            prefix = "[anomaly] " if item["type"] == "anomaly" else ""
            if item.get("type") != "disk_event":
                self.text_thoughts.insert("end", f"• {prefix}{item['msg']}\n"); self.text_thoughts.see("end")

        while True:
            try:
                payload = self.ui_q.get_nowait()
            except queue.Empty:
                break
            self._last_sys = payload.get("sys", {})
            self._last_gpu = payload.get("gpu", {})
            self._last_net_count = payload.get("net", {}).get("count", 0)
            self._last_routine = payload.get("routine", None)
            self._last_title = payload.get("win_title", None)
            self.update_dash(payload)
            self.update_preds(payload)
            self.update_why(payload)
            self.update_assist(payload)
            self.update_timeline(payload)
            aw = self.policy.always_warm()
            self.refresh_candidates_table(aw)

        self.root.after(400, self.poll_updates)

    def update_dash(self, payload: Dict[str, Any]):
        for i in self.metrics.get_children(): self.metrics.delete(i)
        sysd = payload.get("sys", {})
        gpud = payload.get("gpu", {})
        rows = [
            ("Routine", payload.get("routine", None)),
            ("Focus", payload.get("focus", "")),
            ("Window title", payload.get("win_title", "")),
            ("Temperature", round(payload.get("temperature", 0.0), 2)),
            ("CPU %", sysd.get("cpu_pct")),
            ("GPU %", gpud.get("util_pct")),
            ("GPU mem (MB)", gpud.get("mem_used_mb")),
            ("Proc count", sysd.get("proc_count")),
            ("Mem used %", sysd.get("mem", {}).get("percent")),
            ("Disk IO bytes", (sysd.get("disk_io", {}).get("read_bytes", 0) + sysd.get("disk_io", {}).get("write_bytes", 0))),
            ("Connections", payload.get("net", {}).get("count")),
            ("CDN burst (recent)", payload.get("net", {}).get("cdn_burst")),
            ("CPU ceiling %", payload.get("ceilings", {}).get("cpu_pct")),
            ("GPU ceiling %", payload.get("ceilings", {}).get("gpu_pct")),
            ("Risk low", payload.get("risk", {}).get("risk_low")),
            ("Jump-ahead (s)", payload.get("jump_ahead", None)),
        ]
        for m, v in rows: self.metrics.insert("", "end", values=(m, v))
        self.text_graph.delete("1.0","end"); self.text_graph.insert("end", json.dumps(payload.get("graph", {}), indent=2))

    def update_preds(self, payload: Dict[str, Any]):
        for i in self.tree_preds.get_children(): self.tree_preds.delete(i)
        preds = payload.get("predictions", [])
        for target, conf in preds:
            self.tree_preds.insert("", "end", values=(target, f"{int(conf*100)}%"))

    def update_why(self, payload: Dict[str, Any]):
        why = payload.get("why", {})
        self.text_why.delete("1.0","end")
        self.text_why.insert("end", json.dumps(why, indent=2))

    def update_assist(self, payload: Dict[str, Any]):
        assist = payload.get("assist", {})
        self.text_assist.delete("1.0","end")
        parts = []
        if assist.get("suggestion"): parts.append(f"Suggestion: {assist['suggestion']}")
        if assist.get("intent_details"):
            d = assist["intent_details"]
            parts.append(
                f"Intent: net_geom={d.get('net_geom',0)}, width_persist={d.get('net_width_persist',0)}, height_persist={d.get('net_height_persist',0)}, "
                f"disk={d.get('disk',0)}, gpu={d.get('gpu',0)}, motif={d.get('motif',0)}, service_cluster={d.get('service_cluster',0)}, spike={d.get('spike',0)} -> score={d.get('score',0)}"
            )
        if assist.get("why_ext"):
            we = assist["why_ext"]
            parts.append(f"Why++: explore_ratio={we.get('explore_ratio',0)}, stable_top={we.get('stable_top',False)}, cp_strong={we.get('cp_strong',0)}, service_cluster={we.get('service_cluster',0)}, rising_edge_b4b={we.get('rising_edge_b4b',0)}, ctx_hash={we.get('ctx_hash','')}, temperature={we.get('temperature',0.0)}")
        if parts: self.text_assist.insert("end", "\n".join(parts) + "\n")

    def update_timeline(self, payload: Dict[str, Any]):
        tl = payload.get("timeline", [])
        self.text_timeline.delete("1.0","end")
        for item in tl[-40:]:
            self.text_timeline.insert("end", f"{item['ts']}: width_persist={item.get('net_width_persist',0)}, cdn_burst={item.get('cdn_burst',0)}, svc={item.get('service_cluster',0)}, cp_strong={item.get('cp_strong',0)}, disk={item.get('disk_events',0)}, temp={item.get('temperature',0.0)}\n")
        self.text_timeline.see("end")

# ============================================ Main ====================================
def main():
    ensure_requirements_file()
    log_q = queue.Queue(); ui_q = queue.Queue()
    inventory = Inventory(log_q); snap = inventory.collect()
    policy = PolicyEngine(log_q); autoloader = AutoLoader(log_q)

    policy.load()
    user_name = None  # Optional: set your Q:\Users\<name> to include AppData paths
    orchestrator = Orchestrator(log_q, ui_q, policy, user_name=user_name)

    autoloader.ensure(["core","policy","graph","gpu","win"])
    orchestrator.start()

    root = tk.Tk()
    app = ScannerGUI(root, snap, policy, orchestrator, autoloader)
    root.protocol("WM_DELETE_WINDOW", lambda: (orchestrator.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()


