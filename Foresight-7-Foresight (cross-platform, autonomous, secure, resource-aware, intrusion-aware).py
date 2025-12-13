#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foresight (cross-platform, autonomous, secure, resource-aware, intrusion-aware, next-gen prediction with persistence):
- System-wide habit learner: apps, utilities, games, and website-like titles (via browser window title).
- Next-gen predictor:
  * Sequence modeling: first-order Markov chain with exponential decay for recent transitions.
  * Context features: time-of-day, day-of-week, and system load bucket (low/med/high).
  * Decay weighting: prioritizes recent transitions and events.
  * Clustering: maps raw habits into clusters (browser, IDE, office, terminal) to generalize.
  * Persistent state: transition tables and context bias are saved and restored across reboots.
- Encrypted local cache: device-fingerprint key (hostname, MAC, timezone, locale, screen).
- Daily auto-disable: unused habits parked; light probe for reactivation.
- Intrusion detection: flags unfamiliar, spiking processes at odd hours or with scripting parents.
- Compact GUI (~450x290): Suggestion banner, advisory/alert line, Top habits, Frequency chart, Learning toggle.

Author: You
"""

import sys
import os
import json
import time
import threading
import datetime
import importlib
import subprocess
import base64
import locale
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

# ----------------------------
# Auto-loader
# ----------------------------
def auto_loader(libraries: List[str]) -> None:
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception as e:
                print(f"[foresight] Failed to install {lib}: {e}")

auto_loader(["matplotlib", "psutil", "cryptography"])

# Tkinter baseline
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("[foresight] Tkinter not available.")
    print(e)
    sys.exit(1)

# Matplotlib (optional visual)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    plt = None
    FigureCanvasTkAgg = None

# System libraries
import psutil

# Cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# ----------------------------
# Cross-platform foreground window title
# ----------------------------
def get_foreground_window_title() -> Optional[str]:
    plat = sys.platform
    # Windows
    if plat.startswith("win"):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            hwnd = user32.GetForegroundWindow()
            if hwnd == 0:
                return None
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            return title if title else None
        except Exception:
            return None
    # macOS
    elif plat == "darwin":
        try:
            from AppKit import NSWorkspace
            active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            name = active_app.localizedName() if active_app else None
            return name.strip() if name else None
        except Exception:
            return None
    # Linux (X11, best-effort)
    elif plat.startswith("linux"):
        try:
            import subprocess as sp
            out = sp.check_output(["wmctrl", "-lx"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
            lines = [l for l in out.splitlines() if l.strip()]
            if lines:
                parts = lines[-1].split()
                title = parts[-1] if parts else ""
                return title.strip() or None
        except Exception:
            pass
        try:
            import subprocess as sp
            out = sp.check_output(["xprop", "-root", "_NET_ACTIVE_WINDOW"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
            parts = out.split()
            wid = parts[-1]
            if wid and wid != "0x0":
                out2 = sp.check_output(["xprop", "-id", wid, "WM_NAME"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
                if "=" in out2:
                    title = out2.split("=", 1)[1].strip().strip('"')
                    return title if title else None
        except Exception:
            return None
    return None

# ----------------------------
# SecurityGuardian: encryption + reputation + suspicion scoring
# ----------------------------
class SecurityGuardian:
    def __init__(self, salt_file: str = "foresight.salt"):
        self.salt_file = salt_file
        self.salt = self._load_or_create_salt()
        self.key = self._derive_key()
        self.fernet = Fernet(self.key)
        self.bad_markers = [
            "login", "verify", "password", "account locked", "free", "prize", "claim",
            "gift", "crypto", "investment", "urgent", "warning", "lottery", "sweepstakes"
        ]
        # Reputation
        self.whitelist: Set[str] = set()  # safe process habits, e.g., "proc:chrome.exe"
        self.watchlist: Set[str] = set()  # monitored but not blocked
        self.safe_daily: Dict[str, int] = defaultdict(int)  # website-like daily counts

    def _load_or_create_salt(self) -> bytes:
        if os.path.exists(self.salt_file):
            try:
                with open(self.salt_file, "rb") as f:
                    return f.read()
            except Exception:
                pass
        salt = os.urandom(16)
        try:
            with open(self.salt_file, "wb") as f:
                f.write(salt)
        except Exception:
            pass
        return salt

    def _screen_fingerprint(self) -> str:
        # Cross-platform screen size via Tk
        try:
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return f"{w}x{h}"
        except Exception:
            return "unknown_screen"

    def _derive_key(self) -> bytes:
        try:
            hostname = os.uname().nodename  # macOS/Linux
        except Exception:
            hostname = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown_host"
        mac = uuid.getnode()
        tz = time.tzname[0] if time.tzname else "unknown_tz"
        try:
            loc = locale.getdefaultlocale()[0] or "unknown_locale"
        except Exception:
            loc = "unknown_locale"
        screen = self._screen_fingerprint()
        fingerprint = f"{hostname}|{mac}|{tz}|{loc}|{screen}"
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=self.salt,
                         iterations=100_000, backend=default_backend())
        return base64.urlsafe_b64encode(kdf.derive(fingerprint.encode("utf-8")))

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data)

    def score_title_suspicion(self, title: str, hour: int, daily_count: int) -> float:
        t = (title or "").lower()
        score = 0.0
        for m in self.bad_markers:
            if m in t:
                score += 0.4
        if hour in (1, 2, 3, 4, 5):
            score += 0.2
        if daily_count == 0 and any(m in t for m in self.bad_markers):
            score += 0.3
        return min(score, 1.0)

    def score_process_suspicion(self, name: str, cpu: float, mem_mb: float,
                                hour: int, seen_before: bool, parent_name: Optional[str]) -> float:
        lname = (name or "").lower()
        score = 0.0
        if not seen_before and lname not in {x.lower() for x in self.whitelist}:
            score += 0.35
        if cpu >= 25.0 or mem_mb >= 800.0:
            score += 0.35
        if hour in (1, 2, 3, 4, 5):
            score += 0.15
        suspicious_parents = {"powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe", "bash", "zsh", "sh", "osascript"}
        if parent_name and parent_name.lower() in suspicious_parents:
            score += 0.25
        if lname.endswith(".scr") or lname.endswith(".js") or lname.endswith(".vbs"):
            score += 0.25
        return min(score, 1.0)

# ----------------------------
# Data structures
# ----------------------------
class HabitStats:
    __slots__ = ("count", "last_ts", "tod_hist")
    def __init__(self, count: int = 0, last_ts: float = 0.0, tod_hist: Optional[Dict[int, int]] = None):
        self.count = count
        self.last_ts = last_ts
        self.tod_hist = tod_hist if tod_hist is not None else {h: 0 for h in range(24)}
    def to_dict(self) -> Dict:
        return {"count": self.count, "last_ts": self.last_ts, "tod_hist": self.tod_hist}
    @staticmethod
    def from_dict(d: Dict) -> "HabitStats":
        return HabitStats(
            count=int(d.get("count", 0)),
            last_ts=float(d.get("last_ts", 0.0)),
            tod_hist={int(k): int(v) for k, v in d.get("tod_hist", {h: 0 for h in range(24)}).items()}
        )

# ----------------------------
# Next-gen predictor (sequence + context + decay + clustering) with persistence
# ----------------------------
class NextGenPredictor:
    """
    - First-order Markov chain over habit sequences with exponential decay for recent transitions.
    - Context features: hour-of-day (0..23), day-of-week (0..6), and system load bucket (low/med/high).
    - Simple clustering to generalize across similar habits.
    - Blends signals into final scores.
    - Persists its transition and context tables.
    """
    def __init__(self):
        self.transition_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.context_bias: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cluster_map = {
            "browser": ["chrome", "edge", "firefox", "brave", "opera", "safari"],
            "ide": ["code", "vscode", "pycharm", "idea", "sublime"],
            "office": ["word", "excel", "powerpoint", "outlook"],
            "terminal": ["powershell", "cmd", "bash", "zsh", "terminal"]
        }
        self.cluster_weight = 0.15
        self.decay = 0.98  # decay for old transitions
        self.lr = 0.05
        self.weights = {"sequence": 1.0, "context": 1.0, "global": 0.8}
        self._lock = threading.RLock()

    def _habit_cluster(self, habit_key: str) -> Optional[str]:
        h = (habit_key or "").lower()
        for cname, keywords in self.cluster_map.items():
            if any(k in h for k in keywords):
                return cname
        if habit_key and habit_key.startswith("web:"):
            return "browser"
        if habit_key and habit_key.startswith("work:"):
            return "office"
        return None

    def update_sequence(self, prev_habit: Optional[str], curr_habit: str) -> None:
        if not curr_habit:
            return
        with self._lock:
            # Decay all outgoing transitions from prev_habit slightly
            if prev_habit and prev_habit in self.transition_counts:
                for nxt in list(self.transition_counts[prev_habit].keys()):
                    self.transition_counts[prev_habit][nxt] *= self.decay
            # Increment transition
            if prev_habit:
                self.transition_counts[prev_habit][curr_habit] += 1.0
            # Also boost cluster-based transitions
            prev_cluster = self._habit_cluster(prev_habit) if prev_habit else None
            curr_cluster = self._habit_cluster(curr_habit)
            if prev_cluster and curr_cluster and prev_cluster == curr_cluster and prev_habit:
                self.transition_counts[prev_habit][curr_habit] += self.cluster_weight

    def update_context(self, hour: int, dow: int, habit_key: str) -> None:
        with self._lock:
            self.context_bias[(hour, dow)][habit_key] += 1.0

    def predict(self, current_habit: Optional[str], hour: int, dow: int,
                habits: Dict[str, HabitStats], event_log: List[Tuple[float, str]],
                system_load_bucket: int) -> Optional[str]:
        with self._lock:
            if not habits:
                return None
            # Sequence scores
            seq_scores = defaultdict(float)
            if current_habit and current_habit in self.transition_counts:
                for nxt, cnt in self.transition_counts[current_habit].items():
                    seq_scores[nxt] += cnt
            # Context scores
            ctx_scores = defaultdict(float)
            for h, cnt in self.context_bias[(hour, dow)].items():
                ctx_scores[h] += cnt
            # Global frequency
            glob_scores = {h: s.count for h, s in habits.items()}

            # Normalize helpers
            def norm(d: Dict[str, float]) -> Dict[str, float]:
                total = float(sum(d.values())) or 1.0
                return {k: (v / total) for k, v in d.items()}

            seq_n = norm(seq_scores)
            ctx_n = norm(ctx_scores)
            glob_n = norm(glob_scores)

            # Blend
            scores = defaultdict(float)
            for h in habits.keys():
                scores[h] = (
                    self.weights["sequence"] * seq_n.get(h, 0.0)
                    + self.weights["context"] * ctx_n.get(h, 0.0)
                    + self.weights["global"]  * glob_n.get(h, 0.0)
                )
                # Slight boost if cluster aligns with current habit
                if current_habit:
                    ch = self._habit_cluster(current_habit)
                    hh = self._habit_cluster(h)
                    if ch and hh and ch == hh:
                        scores[h] += self.cluster_weight * 0.5

            if not scores:
                return max(glob_n.items(), key=lambda x: x[1])[0] if glob_n else None

            return max(scores.items(), key=lambda x: x[1])[0]

    def reward(self, matched: bool) -> None:
        with self._lock:
            delta = self.lr if matched else -self.lr
            for k in self.weights:
                self.weights[k] = max(0.05, self.weights[k] + delta)

    # ----- Persistence of predictor state -----
    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "transition_counts": {src: dict(dst) for src, dst in self.transition_counts.items()},
                "context_bias": {f"{hour},{dow}": dict(hs) for (hour, dow), hs in self.context_bias.items()},
                "weights": dict(self.weights),
                "cluster_weight": self.cluster_weight,
                "decay": self.decay,
                "lr": self.lr,
            }

    def from_dict(self, d: Dict) -> None:
        with self._lock:
            tc = d.get("transition_counts", {})
            cb = d.get("context_bias", {})
            self.transition_counts = defaultdict(lambda: defaultdict(float))
            for src, dst_map in tc.items():
                self.transition_counts[src] = defaultdict(float, {k: float(v) for k, v in dst_map.items()})
            self.context_bias = defaultdict(lambda: defaultdict(float))
            for k, hs in cb.items():
                try:
                    hour_str, dow_str = k.split(",")
                    key = (int(hour_str), int(dow_str))
                except Exception:
                    continue
                self.context_bias[key] = defaultdict(float, {h: float(v) for h, v in hs.items()})
            w = d.get("weights", {"sequence": 1.0, "context": 1.0, "global": 0.8})
            self.weights = {k: float(v) for k, v in w.items()}
            self.cluster_weight = float(d.get("cluster_weight", self.cluster_weight))
            self.decay = float(d.get("decay", self.decay))
            self.lr = float(d.get("lr", self.lr))

# ----------------------------
# System monitor (cross-platform)
# ----------------------------
class SystemMonitor:
    def __init__(self):
        self.browser_markers = ["Chrome", "Edge", "Firefox", "Brave", "Opera", "Safari"]
        self.cpu_threshold = 2.0   # percent
        self.mem_threshold_mb = 100.0
        self.disabled_items: Set[str] = set()
        self._lock = threading.RLock()

    def normalize_habit(self, raw: str) -> str:
        raw = (raw or "").strip()
        return raw[:128] if raw else ""

    def derive_browser_title_habit(self, title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        for marker in self.browser_markers:
            sep = f" - {marker}"
            if sep in title:
                base = title.split(sep)[0].strip()
                if base:
                    return f"web:{base}"
        return f"app:{title}" if title else None

    def scan_foreground(self) -> Optional[str]:
        title = get_foreground_window_title()
        habit = self.derive_browser_title_habit(title)
        return self.normalize_habit(habit) if habit else None

    def scan_processes(self) -> List[Tuple[str, int, float, float, Optional[str]]]:
        """
        Returns list of tuples:
        (habit_key, pid, cpu_percent, mem_mb, parent_name)
        habit_key examples: "proc:chrome.exe", "game:Game.app", "work:code"
        """
        habits: List[Tuple[str, int, float, float, Optional[str]]] = []
        for p in psutil.process_iter(attrs=["pid", "name"]):
            try:
                name = p.info.get("name") or ""
                if not name:
                    continue
                pid = int(p.info.get("pid"))
                parent_name = None
                try:
                    par = p.parent()
                    if par:
                        parent_name = par.name()
                except Exception:
                    parent_name = None

                key = self.normalize_habit(f"proc:{name}")
                # Disabled light probe
                with self._lock:
                    disabled = key in self.disabled_items
                cpu = p.cpu_percent(interval=0.0) or 0.0
                mem_mb = (p.memory_info().rss / (1024.0 * 1024.0)) if not disabled else 0.0
                if disabled:
                    # Reactivate if CPU rises above low threshold
                    if cpu > self.cpu_threshold:
                        habits.append((key, pid, cpu, mem_mb, parent_name))
                    continue

                # Active monitoring
                if cpu >= self.cpu_threshold or mem_mb >= self.mem_threshold_mb:
                    habits.append((key, pid, cpu, mem_mb, parent_name))
                    lname = name.lower()
                    if any(tag in lname for tag in ["steam", "epic", "game"]):
                        habits.append((self.normalize_habit(f"game:{name}"), pid, cpu, mem_mb, parent_name))
                    if any(tag in lname for tag in ["powershell", "cmd", "terminal", "vscode", "pycharm", "excel", "word", "outlook", "bash", "zsh"]):
                        habits.append((self.normalize_habit(f"work:{name}"), pid, cpu, mem_mb, parent_name))
            except Exception:
                continue
        return habits

    def mark_disabled(self, items: List[str]) -> None:
        with self._lock:
            for it in items:
                if it:
                    self.disabled_items.add(it)

    def mark_enabled(self, items: List[str]) -> None:
        with self._lock:
            for it in items:
                if it and it in self.disabled_items:
                    self.disabled_items.remove(it)

# ----------------------------
# Intrusion alert container
# ----------------------------
class IntrusionAlert:
    def __init__(self, message: str, pid: Optional[int], habit_key: Optional[str], severity: float):
        self.message = message
        self.pid = pid
        self.habit_key = habit_key
        self.severity = severity  # 0..1
        self.timestamp = time.time()

# ----------------------------
# Foresight engine (encrypted cache + security + alerts + persistent next-gen prediction)
# ----------------------------
class Foresight:
    def __init__(self, cache_file: str = "foresight_cache.bin"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self.guard = SecurityGuardian()
        self.habits: Dict[str, HabitStats] = {}
        self.disabled_habits: Dict[str, HabitStats] = {}
        self.event_log: List[Tuple[float, str]] = []
        self.predictor = NextGenPredictor()
        self.learning_enabled = True
        self.activity_score = 0.0
        self.last_review_day = None
        self.monitor = SystemMonitor()
        self.alert_queue: deque[IntrusionAlert] = deque(maxlen=200)
        self._load_cache()

    # ----- Persistence (encrypted) -----
    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        with self._lock:
            try:
                with open(self.cache_file, "rb") as f:
                    enc = f.read()
                raw = self.guard.decrypt(enc)
                data = json.loads(raw.decode("utf-8"))
                self.habits = {h: HabitStats.from_dict(s) for h, s in data.get("habits", {}).items()}
                self.disabled_habits = {h: HabitStats.from_dict(s) for h, s in data.get("disabled", {}).items()}
                self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-50000:]
                self.last_review_day = data.get("last_review_day")
                wl = data.get("whitelist", [])
                self.guard.whitelist = set(wl)
                # Rehydrate monitor disabled set
                self.monitor.mark_disabled(list(self.disabled_habits.keys()))
                # Predictor state
                pred_state = data.get("predictor_state")
                if pred_state:
                    self.predictor.from_dict(pred_state)
            except Exception:
                self.habits = {}
                self.disabled_habits = {}
                self.event_log = []
                self.last_review_day = None
                self.guard.whitelist = set()

    def _save_cache(self) -> None:
        with self._lock:
            data = {
                "habits": {h: s.to_dict() for h, s in self.habits.items()},
                "disabled": {h: s.to_dict() for h, s in self.disabled_habits.items()},
                "event_log": self.event_log[-50000:],
                "last_review_day": self.last_review_day,
                "whitelist": list(self.guard.whitelist),
                "predictor_state": self.predictor.to_dict(),
            }
            raw = json.dumps(data).encode("utf-8")
            enc = self.guard.encrypt(raw)
            tmp = self.cache_file + ".tmp"
            with open(tmp, "wb") as f:
                f.write(enc)
            os.replace(tmp, self.cache_file)

    # ----- Learning & recording -----
    def record_habit(self, habit: str, ts: Optional[float] = None) -> None:
        if not habit or not habit.strip():
            return
        habit = habit.strip()
        now = ts if ts is not None else time.time()
        dt = datetime.datetime.fromtimestamp(now)
        hour = dt.hour
        dow = dt.weekday()
        with self._lock:
            # Reactivate if disabled
            if habit in self.disabled_habits:
                self.habits[habit] = self.disabled_habits.pop(habit)
                self.monitor.mark_enabled([habit])
            # Stats
            stats = self.habits.get(habit)
            if stats is None:
                stats = HabitStats()
                self.habits[habit] = stats
            stats.count += 1
            stats.last_ts = now
            stats.tod_hist[hour] = stats.tod_hist.get(hour, 0) + 1
            # Sequence update: previous event in log
            prev_habit = self.event_log[-1][1] if self.event_log else None
            if prev_habit and isinstance(prev_habit, str) and not prev_habit.startswith("advisory:"):
                self.predictor.update_sequence(prev_habit, habit)
            self.predictor.update_context(hour, dow, habit)
            # Log
            self.event_log.append((now, habit))
            if len(self.event_log) > 50000:
                self.event_log = self.event_log[-50000:]
            # Activity score
            self.activity_score = min(1.0, self.activity_score + 0.08)
            # Security daily count for websites
            if habit.startswith("web:"):
                today = dt.strftime("%Y-%m-%d")
                key = f"{today}|{habit}"
                self.guard.safe_daily[key] += 1
        self._save_cache()

    def get_top_habits(self, n: int = 12) -> List[Tuple[str, int]]:
        with self._lock:
            return sorted(((h, s.count) for h, s in self.habits.items()),
                          key=lambda x: x[1], reverse=True)[:n]

    def _system_load_bucket(self) -> int:
        try:
            load = psutil.cpu_percent(interval=0.0) or 0.0
            if load < 15.0:
                return 0
            elif load < 50.0:
                return 1
            else:
                return 2
        except Exception:
            return 1

    def predict_next(self, when_ts: Optional[float] = None) -> Optional[str]:
        ts = when_ts if when_ts is not None else time.time()
        dt = datetime.datetime.fromtimestamp(ts)
        hour = dt.hour
        dow = dt.weekday()
        with self._lock:
            current_habit = self.event_log[-1][1] if self.event_log else None
            bucket = self._system_load_bucket()
            return self.predictor.predict(current_habit, hour, dow, self.habits, self.event_log, bucket)

    def reinforce(self) -> None:
        with self._lock:
            if not self.event_log or not self.habits:
                return
            last_ts, last_habit = self.event_log[-1]
            dt = datetime.datetime.fromtimestamp(last_ts)
            bucket = self._system_load_bucket()
            predicted = self.predictor.predict(
                self.event_log[-2][1] if len(self.event_log) > 1 else None,
                dt.hour, dt.weekday(), self.habits, self.event_log[:-1], bucket
            )
            matched = (predicted == last_habit)
            self.predictor.reward(matched)
            self.activity_score = max(0.0, self.activity_score - 0.02)
        self._save_cache()

    # ----- Daily review & disabling -----
    def daily_review(self) -> int:
        with self._lock:
            now = time.time()
            today_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            if self.last_review_day == today_str:
                return 0
            cutoff = now - 24 * 3600
            disabled_count = 0
            to_disable = []
            for habit, stats in list(self.habits.items()):
                if stats.last_ts < cutoff:
                    self.disabled_habits[habit] = self.habits.pop(habit)
                    to_disable.append(habit)
                    disabled_count += 1
            self.last_review_day = today_str
        self.monitor.mark_disabled(to_disable)
        self._save_cache()
        return disabled_count

    # ----- Intrusion detection -----
    def assess_process(self, habit_key: str, pid: int, cpu: float, mem_mb: float, parent_name: Optional[str]) -> Optional[IntrusionAlert]:
        hour = datetime.datetime.fromtimestamp(time.time()).hour
        with self._lock:
            seen_before = habit_key in self.habits
        name = habit_key.replace("proc:", "") if habit_key.startswith("proc:") else habit_key
        suspicion = self.guard.score_process_suspicion(name=name, cpu=cpu, mem_mb=mem_mb,
                                                       hour=hour, seen_before=seen_before,
                                                       parent_name=parent_name)
        if suspicion >= 0.7:
            msg = f"Unauthorized access suspected: {name} (PID {pid})"
            alert = IntrusionAlert(message=msg, pid=pid, habit_key=habit_key, severity=suspicion)
            with self._lock:
                self.alert_queue.append(alert)
                self.event_log.append((time.time(), f"advisory:{msg}"))
                if len(self.event_log) > 50000:
                    self.event_log = self.event_log[-50000:]
            return alert
        return None

    # ----- Passive ingestion + security advisories -----
    def ingest_passive_events(self) -> Tuple[int, Optional[str]]:
        if not self.learning_enabled:
            return 0, None
        ingested = 0
        advisory = None

        # Foreground window
        fg = self.monitor.scan_foreground()
        if fg:
            self.record_habit(fg)
            ingested += 1
            if fg.startswith("web:"):
                title = fg[4:]
                dt = datetime.datetime.now()
                hour = dt.hour
                today = dt.strftime("%Y-%m-%d")
                daily_count = self.guard.safe_daily.get(f"{today}|{fg}", 0)
                susp = self.guard.score_title_suspicion(title, hour, daily_count)
                if susp >= 0.7:
                    advisory = f"Suspicious site behavior: '{title}'. Proceed with caution."

        # Processes
        for habit_key, pid, cpu, mem_mb, parent_name in self.monitor.scan_processes():
            self.record_habit(habit_key)
            ingested += 1
            self.assess_process(habit_key, pid, cpu, mem_mb, parent_name)

        if ingested > 0:
            self.reinforce()
        return ingested, advisory

    # ----- Intrusion actions -----
    def kill_process(self, pid: int) -> bool:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except psutil.TimeoutExpired:
                p.kill()
            return True
        except Exception:
            return False

    def allow_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.whitelist.add(habit_key)
        self._save_cache()

    def monitor_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.watchlist.add(habit_key)
        self._save_cache()

    # ----- Controls -----
    def set_learning(self, enabled: bool) -> None:
        with self._lock:
            self.learning_enabled = enabled

# ----------------------------
# Background scheduler
# ----------------------------
class Scheduler(threading.Thread):
    def __init__(self, foresight: Foresight, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.fs = foresight
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                disabled_ct = self.fs.daily_review()
                with self.fs._lock:
                    active = self.fs.activity_score
                    learning = self.fs.learning_enabled
                base_interval = 2.0 if learning else 6.0
                interval = max(0.8, base_interval * (1.5 - active))
                ingested, advisory = self.fs.ingest_passive_events()
                if ingested == 0 and learning and disabled_ct == 0:
                    self.fs.reinforce()
                if advisory:
                    with self.fs._lock:
                        self.fs.event_log.append((time.time(), f"advisory:{advisory}"))
                        if len(self.fs.event_log) > 50000:
                            self.fs.event_log = self.fs.event_log[-50000:]
                time.sleep(interval)
            except Exception:
                time.sleep(3.0)

# ----------------------------
# Minimal, compact GUI (~450x290) with intrusion prompt
# ----------------------------
class ForesightGUI:
    def __init__(self, foresight: Foresight, update_ms: int = 1000):
        self.fs = foresight
        self.update_ms = update_ms
        self.stop_event = threading.Event()
        self.root = tk.Tk()
        self.root.title("Foresight")
        try:
            self.root.call('tk', 'scaling', 1.0)
        except Exception:
            pass

        self.root.geometry("450x290")
        self.root.minsize(420, 260)

        self.figure = None
        self.ax = None
        self.canvas = None
        self._build_layout()

        self.scheduler = Scheduler(self.fs, self.stop_event)
        self.scheduler.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=6)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)

        # Top bar: suggestion banner + toggle
        top = ttk.Frame(container)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        top.columnconfigure(0, weight=1)
        self.suggestion_var = tk.StringVar(value="Suggestion: —")
        sugg_label = ttk.Label(top, textvariable=self.suggestion_var)
        sugg_label.grid(row=0, column=0, sticky="w")

        self.learning_var = tk.BooleanVar(value=True)
        learn_btn = ttk.Checkbutton(top, text="Learning", variable=self.learning_var, command=self._toggle_learning)
        learn_btn.grid(row=0, column=1, sticky="e")

        # Advisory/alert banner
        self.advisory_var = tk.StringVar(value="")
        advisory_label = ttk.Label(top, textvariable=self.advisory_var, foreground="#b22222")
        advisory_label.grid(row=1, column=0, columnspan=2, sticky="w")

        # Left: top habits list
        left = ttk.LabelFrame(container, text="Top habits", padding=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, height=10)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        # Right: chart
        right = ttk.LabelFrame(container, text="Frequency", padding=6)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        if plt is not None and FigureCanvasTkAgg is not None:
            self.figure = plt.Figure(figsize=(3.6, 2.2), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Frequency")
            self.ax.set_ylabel("Count")
            self.ax.tick_params(axis='x', rotation=35, labelsize=7)
            self.canvas = FigureCanvasTkAgg(self.figure, master=right)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", side="bottom")

    def _toggle_learning(self):
        self.fs.set_learning(self.learning_var.get())
        self.status_var.set("Learning is " + ("ON" if self.learning_var.get() else "OFF"))

    def _poll_advisory(self) -> Optional[str]:
        with self.fs._lock:
            for ts, h in reversed(self.fs.event_log[-50:]):
                if isinstance(h, str) and h.startswith("advisory:"):
                    return h[len("advisory:"):]
        return None

    def _poll_alert(self) -> Optional[IntrusionAlert]:
        with self.fs._lock:
            if self.fs.alert_queue:
                return self.fs.alert_queue.pop()
        return None

    def _show_intrusion_prompt(self, alert: IntrusionAlert):
        # Modal prompt: Kill / Allow / Monitor
        msg = f"{alert.message}\n\nChoose an action:\n- Kill: terminate the process\n- Allow: whitelist and ignore alerts for this item\n- Monitor: keep observing and alert if it escalates"
        resp = messagebox.askquestion("Intrusion detected", msg, icon="warning")
        if resp == "yes":  # interpret as Kill
            success = (alert.pid is not None) and self.fs.kill_process(alert.pid)
            self.status_var.set("Process terminated." if success else "Failed to terminate process.")
            return
        resp2 = messagebox.askyesno("Allow or Monitor?", "Allow this item (Yes) or Monitor it (No)?")
        if resp2:
            if alert.habit_key:
                self.fs.allow_habit(alert.habit_key)
                self.status_var.set("Item whitelisted.")
        else:
            if alert.habit_key:
                self.fs.monitor_habit(alert.habit_key)
                self.status_var.set("Item added to watchlist.")

    def _tick(self):
        try:
            # Update suggestion
            suggested = self.fs.predict_next()
            self.suggestion_var.set(f"Suggestion: {suggested if suggested else '—'}")

            # Advisory banner
            adv = self._poll_advisory()
            self.advisory_var.set(adv if adv else "")

            # Intrusion prompt (if any)
            alert = self._poll_alert()
            if alert:
                self._show_intrusion_prompt(alert)

            # Update habits list
            self.listbox.delete(0, tk.END)
            for h, c in self.fs.get_top_habits(20):
                self.listbox.insert(tk.END, f"{h}  —  {c}")

            # Update chart
            if self.canvas is not None and self.ax is not None:
                self.ax.clear()
                self.ax.set_title("Frequency")
                self.ax.set_ylabel("Count")
                data = self.fs.get_top_habits(10)
                if data:
                    habits = [h for h, _ in data]
                    counts = [c for _, c in data]
                    self.ax.bar(habits, counts, color="#66b3ff")
                    self.ax.tick_params(axis='x', rotation=35, labelsize=7)
                else:
                    self.ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
                self.canvas.draw()

            # Status
            with self.fs._lock:
                total = len(self.fs.habits)
                disabled_total = len(self.fs.disabled_habits)
                events = len(self.fs.event_log)
                act = self.fs.activity_score
                learn = self.fs.learning_enabled
            self.status_var.set(
                f"Active: {total} | Disabled: {disabled_total} | Events: {events} | Act: {act:.2f} | {'Learning ON' if learn else 'Learning OFF'}"
            )
        except Exception as e:
            self.status_var.set(f"Update error: {e}")

        self.root.after(self.update_ms, self._tick)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ----------------------------
# Entry point
# ----------------------------
def main():
    fs = Foresight(cache_file="foresight_cache.bin")
    gui = ForesightGUI(fs, update_ms=1000)
    gui.run()

if __name__ == "__main__":
    main()

