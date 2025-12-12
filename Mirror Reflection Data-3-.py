#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Telemetry + Foresight Dashboard
- System telemetry: CPU, memory, disk (per-drive selection), network, GPU, processes
- Foresight: habit learning, contextual prediction, encrypted cache, intrusion detection, compact GUI
- Single Tkinter application with two tabs (Telemetry, Foresight)
- Auto-loader for optional libraries: psutil, py-cpuinfo, pynvml, cryptography, matplotlib (optional)
"""

import sys
import os
import time
import csv
import json
import threading
import queue
import importlib
import subprocess
import platform
import base64
import locale
import uuid
import datetime
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# ---------------------------
# Auto-loader for libraries
# ---------------------------
def ensure(lib_name, package_name=None):
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        pkg = package_name or lib_name
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            importlib.invalidate_caches()
            importlib.import_module(lib_name)
            return True
        except Exception:
            return False

HAS_PSUTIL   = ensure("psutil")
HAS_CPUINFO  = ensure("cpuinfo", "py-cpuinfo")
HAS_PYNVML   = ensure("pynvml")
HAS_CRYPTO   = ensure("cryptography")
HAS_MPL      = ensure("matplotlib")

# Tkinter baseline
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except Exception as e:
    print("Tkinter not available:", e)
    sys.exit(1)

# Optional visuals
plt = None
FigureCanvasTkAgg = None
if HAS_MPL:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception:
        plt = None
        FigureCanvasTkAgg = None

# System libraries
if HAS_PSUTIL:
    import psutil

# Cryptography
if HAS_CRYPTO:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

import random  # fallback demo values

# ---------------------------
# Utility helpers
# ---------------------------
def safe_get(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def fmt_bytes(n):
    try:
        for unit in ["B","KB","MB","GB","TB","PB"]:
            if n < 1024.0:
                return f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} EB"
    except Exception:
        return "n/a"

def rolling_file(base_dir="telemetry_logs"):
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"session_{ts}.csv")

def is_network_fs(fstype, device, mountpoint):
    fstype = (fstype or "").lower()
    net_types = {"nfs", "smbfs", "cifs", "afp", "fuse.sshfs", "glusterfs"}
    if fstype in net_types:
        return True
    if platform.system() == "Windows" and device and device.startswith("\\\\"):
        return True
    return False

def basename_device(dev):
    if not dev:
        return ""
    base = os.path.basename(dev)
    if platform.system() == "Windows":
        if len(dev) >= 2 and dev[1] == ":":
            return dev[:2]
    return base

# ---------------------------
# Foreground window title (cross-platform)
# ---------------------------
def get_foreground_window_title() -> Optional[str]:
    plat = sys.platform
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
    elif plat == "darwin":
        try:
            from AppKit import NSWorkspace
            active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            name = active_app.localizedName() if active_app else None
            return name.strip() if name else None
        except Exception:
            return None
    elif plat.startswith("linux"):
        try:
            import subprocess as sp
            out = sp.check_output(["wmctrl", "-lx"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
            lines = [l for l in out.splitlines() if l.strip()]
            if lines:
                title = lines[-1].split()[-1]
                return title.strip()
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

# ---------------------------
# SecurityGuardian: encryption + reputation + suspicion scoring
# ---------------------------
class SecurityGuardian:
    def __init__(self, salt_file: str = "foresight.salt"):
        self.salt_file = salt_file
        self.salt = self._load_or_create_salt()
        self.key = self._derive_key()
        self.fernet = Fernet(self.key) if HAS_CRYPTO else None
        self.bad_markers = [
            "login", "verify", "password", "account locked", "free", "prize", "claim",
            "gift", "crypto", "investment", "urgent", "warning", "lottery", "sweepstakes"
        ]
        self.whitelist: Set[str] = set()
        self.watchlist: Set[str] = set()
        self.safe_daily: Dict[str, int] = defaultdict(int)

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
        try:
            root = tk.Tk(); root.withdraw()
            w = root.winfo_screenwidth(); h = root.winfo_screenheight()
            root.destroy()
            return f"{w}x{h}"
        except Exception:
            return "unknown_screen"

    def _derive_key(self) -> bytes:
        if not HAS_CRYPTO:
            return base64.urlsafe_b64encode(os.urandom(32))
        try:
            hostname = os.uname().nodename
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
        return self.fernet.encrypt(data) if self.fernet else data

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data) if self.fernet else data

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

# ---------------------------
# Foresight data structures and engine
# ---------------------------
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

class AdaptivePredictor:
    def __init__(self):
        self.weights = {"tod": 1.0, "recency": 1.0, "global": 1.0}
        self.lr = 0.05
        self.decay = 0.999
        self._lock = threading.RLock()
    def predict(self, hour: int, habits: Dict[str, HabitStats], event_log: List[Tuple[float, str]]) -> Optional[str]:
        if not habits:
            return None
        with self._lock:
            tod_scores = {h: s.tod_hist.get(hour, 0) for h, s in habits.items()}
            recent = event_log[-80:]
            recency = defaultdict(int)
            for _, h in recent:
                recency[h] += 1
            global_scores = {h: s.count for h, s in habits.items()}
            def norm(d: Dict[str, int]) -> Dict[str, float]:
                total = float(sum(d.values())) or 1.0
                return {k: (v / total) for k, v in d.items()}
            tod_n = norm(tod_scores); recency_n = norm(recency); global_n = norm(global_scores)
            scores = defaultdict(float)
            for h in habits.keys():
                scores[h] = (self.weights["tod"] * tod_n.get(h, 0.0)
                            + self.weights["recency"] * recency_n.get(h, 0.0)
                            + self.weights["global"] * global_n.get(h, 0.0))
            return max(scores.items(), key=lambda x: x[1])[0]
    def reward(self, matched: bool) -> None:
        with self._lock:
            delta = self.lr if matched else -self.lr
            for k in self.weights:
                self.weights[k] = max(0.05, self.weights[k] * self.decay + delta)

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
        habits: List[Tuple[str, int, float, float, Optional[str]]] = []
        if not HAS_PSUTIL:
            return habits
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
                with self._lock:
                    disabled = key in self.disabled_items
                cpu = p.cpu_percent(interval=0.0) or 0.0
                mem_mb = (p.memory_info().rss / (1024.0 * 1024.0)) if not disabled else 0.0
                if disabled:
                    if cpu > self.cpu_threshold:
                        habits.append((key, pid, cpu, mem_mb, parent_name))
                    continue

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

class IntrusionAlert:
    def __init__(self, message: str, pid: Optional[int], habit_key: Optional[str], severity: float):
        self.message = message
        self.pid = pid
        self.habit_key = habit_key
        self.severity = severity  # 0..1
        self.timestamp = time.time()

class Foresight:
    def __init__(self, cache_file: str = "foresight_cache.bin"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self.guard = SecurityGuardian()
        self.habits: Dict[str, HabitStats] = {}
        self.disabled_habits: Dict[str, HabitStats] = {}
        self.event_log: List[Tuple[float, str]] = []
        self.predictor = AdaptivePredictor()
        self.learning_enabled = True
        self.activity_score = 0.0
        self.last_review_day = None
        self.monitor = SystemMonitor()
        self.alert_queue: deque[IntrusionAlert] = deque(maxlen=200)
        self._load_cache()

    # Persistence
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
                self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-30000:]
                w = data.get("weights", {"tod": 1.0, "recency": 1.0, "global": 1.0})
                self.predictor.weights = {k: float(v) for k, v in w.items()}
                self.last_review_day = data.get("last_review_day")
                wl = data.get("whitelist", [])
                self.guard.whitelist = set(wl)
                self.monitor.mark_disabled(list(self.disabled_habits.keys()))
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
                "event_log": self.event_log[-30000:],
                "weights": self.predictor.weights,
                "last_review_day": self.last_review_day,
                "whitelist": list(self.guard.whitelist),
            }
            raw = json.dumps(data).encode("utf-8")
            enc = self.guard.encrypt(raw)
            tmp = self.cache_file + ".tmp"
            with open(tmp, "wb") as f:
                f.write(enc)
            os.replace(tmp, self.cache_file)

    # Learning
    def record_habit(self, habit: str, ts: Optional[float] = None) -> None:
        if not habit or not habit.strip():
            return
        habit = habit.strip()
        now = ts if ts is not None else time.time()
        hour = datetime.datetime.fromtimestamp(now).hour
        with self._lock:
            if habit in self.disabled_habits:
                self.habits[habit] = self.disabled_habits.pop(habit)
                self.monitor.mark_enabled([habit])
            stats = self.habits.get(habit)
            if stats is None:
                stats = HabitStats()
                self.habits[habit] = stats
            stats.count += 1
            stats.last_ts = now
            stats.tod_hist[hour] = stats.tod_hist.get(hour, 0) + 1
            self.event_log.append((now, habit))
            if len(self.event_log) > 30000:
                self.event_log = self.event_log[-30000:]
            self.activity_score = min(1.0, self.activity_score + 0.08)
            if habit.startswith("web:"):
                today = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
                key = f"{today}|{habit}"
                self.guard.safe_daily[key] += 1
        self._save_cache()

    def get_top_habits(self, n: int = 12) -> List[Tuple[str, int]]:
        with self._lock:
            return sorted(((h, s.count) for h, s in self.habits.items()),
                          key=lambda x: x[1], reverse=True)[:n]

    def predict_next(self, when_ts: Optional[float] = None) -> Optional[str]:
        ts = when_ts if when_ts is not None else time.time()
        hour = datetime.datetime.fromtimestamp(ts).hour
        with self._lock:
            return self.predictor.predict(hour, self.habits, self.event_log)

    def reinforce(self) -> None:
        with self._lock:
            if not self.event_log or not self.habits:
                return
            last_ts, last_habit = self.event_log[-1]
            predicted = self.predictor.predict(
                datetime.datetime.fromtimestamp(last_ts).hour, self.habits, self.event_log[:-1]
            )
            matched = (predicted == last_habit)
            self.predictor.reward(matched)
            self.activity_score = max(0.0, self.activity_score - 0.02)
        self._save_cache()

    # Daily review
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

    # Intrusion detection
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
                if len(self.event_log) > 30000:
                    self.event_log = self.event_log[-30000:]
            return alert
        return None

    # Passive ingestion + advisories
    def ingest_passive_events(self) -> Tuple[int, Optional[str]]:
        if not self.learning_enabled:
            return 0, None
        ingested = 0
        advisory = None

        fg = self.monitor.scan_foreground()
        if fg:
            self.record_habit(fg)
            ingested += 1
            if fg.startswith("web:"):
                title = fg[4:]
                hour = datetime.datetime.fromtimestamp(time.time()).hour
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                daily_count = self.guard.safe_daily.get(f"{today}|{fg}", 0)
                susp = self.guard.score_title_suspicion(title, hour, daily_count)
                if susp >= 0.7:
                    advisory = f"Suspicious site behavior: '{title}'. Proceed with caution."

        for habit_key, pid, cpu, mem_mb, parent_name in self.monitor.scan_processes():
            self.record_habit(habit_key)
            ingested += 1
            self.assess_process(habit_key, pid, cpu, mem_mb, parent_name)

        if ingested > 0:
            self.reinforce()
        return ingested, advisory

    # Intrusion actions and controls
    def kill_process(self, pid: int) -> bool:
        if not HAS_PSUTIL:
            return False
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except Exception:
                p.kill()
            return True
        except Exception:
            return False

    def allow_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.whitelist.add(habit_key)

    def monitor_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.watchlist.add(habit_key)

    def set_learning(self, enabled: bool) -> None:
        with self._lock:
            self.learning_enabled = enabled

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
                        if len(self.fs.event_log) > 30000:
                            self.fs.event_log = self.fs.event_log[-30000:]
                time.sleep(interval)
            except Exception:
                time.sleep(3.0)

# ---------------------------
# Telemetry collectors and engine
# ---------------------------
DEFAULT_THRESHOLDS = {
    "cpu_total_percent": 85.0,
    "mem_percent": 90.0,
    "disk_write_bps": 100 * 1024 * 1024,
    "disk_read_bps": 100 * 1024 * 1024,
    "net_send_bps": 50 * 1024 * 1024,
    "net_recv_bps": 50 * 1024 * 1024,
    "temp_c": 85.0,
}

class CollectorBase:
    name = "base"
    def collect(self):
        return {}

class CPUCollector(CollectorBase):
    name = "cpu"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            data["cpu_percent_per_core"] = psutil.cpu_percent(percpu=True)
            data["cpu_percent_total"] = psutil.cpu_percent()
            freq = safe_get(lambda: psutil.cpu_freq(), None)
            if freq:
                data["cpu_freq_current_mhz"] = freq.current
                data["cpu_freq_min_mhz"] = freq.min
                data["cpu_freq_max_mhz"] = freq.max
            if hasattr(os, "getloadavg"):
                try:
                    la1, la5, la15 = os.getloadavg()
                    data["load_avg_1m"] = la1
                    data["load_avg_5m"] = la5
                    data["load_avg_15m"] = la15
                except Exception:
                    pass
            temps = safe_get(lambda: psutil.sensors_temperatures(), {})
            if temps:
                flat = {}
                for k, arr in temps.items():
                    if arr:
                        t = arr[0]
                        flat[f"temp_{k}"] = getattr(t, "current", None)
                data["temps"] = flat
            batt = safe_get(lambda: psutil.sensors_battery(), None)
            if batt:
                data["battery_percent"] = batt.percent
                data["battery_plugged"] = bool(batt.power_plugged)
        else:
            data["cpu_percent_total"] = random.uniform(5, 50)
            data["cpu_percent_per_core"] = [random.uniform(5, 50) for _ in range(4)]
        if HAS_CPUINFO:
            from cpuinfo import get_cpu_info
            info = safe_get(get_cpu_info, {})
            if info:
                data["cpu_brand"] = info.get("brand_raw")
                data["arch"] = info.get("arch")
        else:
            data["cpu_brand"] = platform.processor()
            data["arch"] = platform.machine()
        return data

class MemoryCollector(CollectorBase):
    name = "memory"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            data["mem_total"] = vm.total
            data["mem_used"] = vm.used
            data["mem_available"] = vm.available
            data["mem_percent"] = vm.percent
            data["swap_total"] = sm.total
            data["swap_used"] = sm.used
            data["swap_percent"] = sm.percent
        return data

class DiskCollector(CollectorBase):
    name = "disk"
    def __init__(self, get_selection_fn):
        self.get_selection_fn = get_selection_fn
    def collect(self):
        data = {"perdisk": {}, "aggregate": {}}
        selected = set(self.get_selection_fn())
        if HAS_PSUTIL:
            perdisk = safe_get(lambda: psutil.disk_io_counters(perdisk=True), {}) or {}
            for key, stats in perdisk.items():
                base = basename_device(key)
                if not selected or key in selected or base in selected:
                    data["perdisk"][key] = {
                        "read_bytes": stats.read_bytes,
                        "write_bytes": stats.write_bytes,
                        "read_count": stats.read_count,
                        "write_count": stats.write_count,
                    }
            agg = safe_get(lambda: psutil.disk_io_counters(), None)
            if agg:
                data["aggregate"] = {
                    "disk_read_bytes": agg.read_bytes,
                    "disk_write_bytes": agg.write_bytes,
                    "disk_read_count": agg.read_count,
                    "disk_write_count": agg.write_count,
                }
        return data

class NetCollector(CollectorBase):
    name = "network"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            io = psutil.net_io_counters()
            if io:
                data["net_bytes_sent"] = io.bytes_sent
                data["net_bytes_recv"] = io.bytes_recv
                data["net_packets_sent"] = io.packets_sent
                data["net_packets_recv"] = io.packets_recv
        return data

class ProcessCollector(CollectorBase):
    name = "process"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            procs = []
            for p in psutil.process_iter(attrs=["pid","name","cpu_percent","memory_info"]):
                try:
                    info = p.info
                    procs.append({
                        "pid": info["pid"],
                        "name": info.get("name",""),
                        "cpu": info.get("cpu_percent", 0.0),
                        "rss": getattr(info.get("memory_info"), "rss", 0)
                    })
                except Exception:
                    continue
            procs.sort(key=lambda x: x["cpu"], reverse=True)
            data["top_processes"] = procs[:8]
        return data

class GPUCollector(CollectorBase):
    name = "gpu"
    def __init__(self):
        self.nv_ok = False
        if HAS_PYNVML:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.nv_ok = True
            except Exception:
                self.nv_ok = False
    def collect(self):
        data = {}
        if self.nv_ok:
            nv = self.pynvml
            try:
                count = nv.nvmlDeviceGetCount()
                gpus = []
                for i in range(count):
                    h = nv.nvmlDeviceGetHandleByIndex(i)
                    name = nv.nvmlDeviceGetName(h).decode()
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                    gpus.append({
                        "index": i,
                        "name": name,
                        "gpu_util": util.gpu,
                        "mem_util": (mem.used / mem.total * 100.0) if mem.total else 0.0,
                        "mem_total": mem.total,
                        "mem_used": mem.used,
                        "temp_c": temp
                    })
                data["gpus"] = gpus
            except Exception:
                pass
        return data

class TelemetryEngine:
    def __init__(self, interval=1.0, adaptive=True, enabled_plugins=None, thresholds=None):
        self.interval = interval
        self.adaptive = adaptive
        self.enabled_plugins = enabled_plugins or {
            "cpu": True,
            "memory": True,
            "disk": True,
            "network": True,
            "process": True,
            "gpu": True,
        }
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

        # Drive inventory and selection
        self.drive_inventory = []   # {device, mountpoint, fstype, is_network, key}
        self.selected_drives = set()
        self._enumerate_drives()

        self.collectors_map = {
            "cpu": CPUCollector(),
            "memory": MemoryCollector(),
            "disk": DiskCollector(self.get_selected_drives),
            "network": NetCollector(),
            "process": ProcessCollector(),
            "gpu": GPUCollector(),
        }

        self._stop = threading.Event()
        self._thread = None
        self.out_queue = queue.Queue(maxsize=64)
        self.last_cpu_util = deque(maxlen=5)
        self.logging_enabled = False
        self.log_file = None
        self.csv_writer = None

        self._prev_disk_per = {}
        self._prev_disk_agg = None
        self._prev_net = None
        self._prev_time = None

        self.alerts = []

    def _enumerate_drives(self):
        self.drive_inventory.clear()
        if HAS_PSUTIL:
            parts = safe_get(lambda: psutil.disk_partitions(all=True), []) or []
            for p in parts:
                dev = p.device
                mp = p.mountpoint
                fs = p.fstype
                net = is_network_fs(fs, dev, mp)
                key = basename_device(dev) or dev or mp
                self.drive_inventory.append({
                    "device": dev,
                    "mountpoint": mp,
                    "fstype": fs,
                    "is_network": net,
                    "key": key,
                })
            self.selected_drives = {d["key"] for d in self.drive_inventory}
        else:
            self.selected_drives = set()

    def get_selected_drives(self):
        return list(self.selected_drives)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_log()

    def set_interval(self, interval):
        self.interval = max(0.2, float(interval))

    def set_adaptive(self, adaptive):
        self.adaptive = bool(adaptive)

    def update_enabled_plugins(self, mapping):
        self.enabled_plugins.update(mapping)

    def update_thresholds(self, mapping):
        self.thresholds.update(mapping)

    def update_selected_drives(self, selection_keys):
        self.selected_drives = set(selection_keys)

    def enable_logging(self, enable=True):
        self.logging_enabled = bool(enable)
        if enable and not self.log_file:
            self._open_log()
        elif not enable:
            self._close_log()

    def _open_log(self):
        try:
            self.log_file_path = rolling_file()
            self.log_file = open(self.log_file_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.log_file)
        except Exception:
            self.log_file = None
            self.csv_writer = None

    def _close_log(self):
        try:
            if self.log_file:
                self.log_file.close()
        finally:
            self.log_file = None
            self.csv_writer = None

    def _flatten_row(self, snapshot):
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "cpu_total": snapshot.get("cpu", {}).get("cpu_percent_total"),
            "mem_percent": snapshot.get("memory", {}).get("mem_percent"),
            "disk_read_total": snapshot.get("disk", {}).get("aggregate", {}).get("disk_read_bytes"),
            "disk_write_total": snapshot.get("disk", {}).get("aggregate", {}).get("disk_write_bytes"),
            "net_sent": snapshot.get("network", {}).get("net_bytes_sent"),
            "net_recv": snapshot.get("network", {}).get("net_bytes_recv"),
            "load1": snapshot.get("cpu", {}).get("load_avg_1m"),
            "gpu0_util": None,
        }
        g = snapshot.get("gpu", {}).get("gpus", [])
        if g:
            row["gpu0_util"] = g[0].get("gpu_util")
        return row

    def _check_alerts(self, snapshot, deltas):
        alerts = []
        thr = self.thresholds
        cpu_total = snapshot.get("cpu", {}).get("cpu_percent_total", 0.0)
        if cpu_total is not None and cpu_total >= thr["cpu_total_percent"]:
            alerts.append(f"CPU total {cpu_total:.1f}% >= {thr['cpu_total_percent']:.1f}%")

        mem_percent = snapshot.get("memory", {}).get("mem_percent", 0.0)
        if mem_percent is not None and mem_percent >= thr["mem_percent"]:
            alerts.append(f"Memory {mem_percent:.1f}% >= {thr['mem_percent']:.1f}%")

        if deltas:
            dw = deltas.get("disk_write_bps", 0.0)
            dr = deltas.get("disk_read_bps", 0.0)
            ns = deltas.get("net_send_bps", 0.0)
            nr = deltas.get("net_recv_bps", 0.0)
            if dw >= thr["disk_write_bps"]:
                alerts.append(f"Disk write {fmt_bytes(dw)}/s >= {fmt_bytes(thr['disk_write_bps'])}/s")
            if dr >= thr["disk_read_bps"]:
                alerts.append(f"Disk read {fmt_bytes(dr)}/s >= {fmt_bytes(thr['disk_read_bps'])}/s")
            if ns >= thr["net_send_bps"]:
                alerts.append(f"Net send {fmt_bytes(ns)}/s >= {fmt_bytes(thr['net_send_bps'])}/s")
            if nr >= thr["net_recv_bps"]:
                alerts.append(f"Net recv {fmt_bytes(nr)}/s >= {fmt_bytes(thr['net_recv_bps'])}/s")

        temps = snapshot.get("cpu", {}).get("temps", {}) or {}
        for k, v in temps.items():
            if v is not None and v >= thr["temp_c"]:
                alerts.append(f"{k} {v:.1f}°C >= {thr['temp_c']:.1f}°C")

        gpus = snapshot.get("gpu", {}).get("gpus", []) or []
        for g in gpus:
            t = g.get("temp_c")
            if t is not None and t >= thr["temp_c"]:
                alerts.append(f"GPU{g.get('index')} {t:.1f}°C >= {thr['temp_c']:.1f}°C")

        self.alerts = alerts

    def _calc_deltas(self, snapshot, now):
        deltas = {}
        agg = snapshot.get("disk", {}).get("aggregate", {}) or None
        if self._prev_disk_agg and self._prev_time and agg:
            dt = max(0.001, now - self._prev_time)
            deltas["disk_write_bps"] = max(0.0, (agg.get("disk_write_bytes", 0) - self._prev_disk_agg.get("disk_write_bytes", 0)) / dt)
            deltas["disk_read_bps"] = max(0.0, (agg.get("disk_read_bytes", 0) - self._prev_disk_agg.get("disk_read_bytes", 0)) / dt)

        net = snapshot.get("network", {}) or None
        if self._prev_net and self._prev_time and net:
            dt = max(0.001, now - self._prev_time)
            deltas["net_send_bps"] = max(0.0, (net.get("net_bytes_sent", 0) - self._prev_net.get("net_bytes_sent", 0)) / dt)
            deltas["net_recv_bps"] = max(0.0, (net.get("net_bytes_recv", 0) - self._prev_net.get("net_bytes_recv", 0)) / dt)

        deltas["perdrive_bps"] = {}
        perdisk = snapshot.get("disk", {}).get("perdisk", {}) or {}
        if self._prev_disk_per and self._prev_time and perdisk:
            dt = max(0.001, now - self._prev_time)
            for k, curr in perdisk.items():
                prev = self._prev_disk_per.get(k)
                if prev:
                    deltas["perdrive_bps"][k] = {
                        "write_bps": max(0.0, (curr["write_bytes"] - prev.get("write_bytes", 0)) / dt),
                        "read_bps": max(0.0, (curr["read_bytes"] - prev.get("read_bytes", 0)) / dt),
                    }
        return deltas

    def _run(self):
        header_written = False
        while not self._stop.is_set():
            start = time.time()
            snapshot = {}
            for key, collector in self.collectors_map.items():
                if not self.enabled_plugins.get(key, True):
                    continue
                try:
                    snapshot[key] = collector.collect()
                except Exception:
                    snapshot[key] = {}

            cpu_total = snapshot.get("cpu", {}).get("cpu_percent_total", 0.0)
            self.last_cpu_util.append(cpu_total if cpu_total is not None else 0.0)
            if self.adaptive and len(self.last_cpu_util) >= 3:
                avg = sum(self.last_cpu_util) / len(self.last_cpu_util)
                if avg > 75:
                    target = max(0.5, self.interval * 1.5)
                elif avg < 20:
                    target = max(0.2, self.interval * 0.8)
                else:
                    target = self.interval
            else:
                target = self.interval

            now = time.time()
            deltas = self._calc_deltas(snapshot, now)
            self._check_alerts(snapshot, deltas)

            self._prev_time = now
            self._prev_disk_agg = snapshot.get("disk", {}).get("aggregate", None)
            self._prev_disk_per = snapshot.get("disk", {}).get("perdisk", {})
            self._prev_net = snapshot.get("network", None)

            snapshot["_deltas"] = deltas
            snapshot["_alerts"] = list(self.alerts)
            snapshot["_drives"] = list(self.drive_inventory)
            snapshot["_selected_drives"] = list(self.selected_drives)

            try:
                self.out_queue.put(snapshot, timeout=0.1)
            except queue.Full:
                pass

            if self.logging_enabled and self.csv_writer:
                row = self._flatten_row(snapshot)
                if not header_written:
                    self.csv_writer.writerow(list(row.keys()))
                    header_written = True
                self.csv_writer.writerow(list(row.values()))
                try:
                    self.log_file.flush()
                except Exception:
                    pass

            elapsed = time.time() - start
            remaining = max(0.0, target - elapsed)
            self._stop.wait(remaining)

# ---------------------------
# Settings dialog (Telemetry)
# ---------------------------
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, engine: TelemetryEngine, on_apply):
        super().__init__(master)
        self.title("Telemetry settings")
        self.resizable(True, True)
        self.engine = engine
        self.on_apply = on_apply

        container = ttk.Notebook(self)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        overlay_frame = ttk.Frame(container)
        container.add(overlay_frame, text="Overlay")
        self.overlay_var = tk.BooleanVar(value=bool(master.attributes("-topmost")))
        ttk.Checkbutton(overlay_frame, text="Always on top", variable=self.overlay_var).pack(anchor="w", padx=8, pady=8)

        plugins_frame = ttk.Frame(container)
        container.add(plugins_frame, text="Plugins")
        self.plugin_vars = {}
        for key in ["cpu","memory","disk","network","process","gpu"]:
            var = tk.BooleanVar(value=self.engine.enabled_plugins.get(key, True))
            ttk.Checkbutton(plugins_frame, text=key, variable=var).pack(anchor="w", padx=8, pady=4)
            self.plugin_vars[key] = var

        thr_frame = ttk.Frame(container)
        container.add(thr_frame, text="Thresholds")
        thr = self.engine.thresholds
        self.thr_vars = {
            "cpu_total_percent": tk.DoubleVar(value=thr["cpu_total_percent"]),
            "mem_percent": tk.DoubleVar(value=thr["mem_percent"]),
            "disk_write_bps": tk.DoubleVar(value=thr["disk_write_bps"]),
            "disk_read_bps": tk.DoubleVar(value=thr["disk_read_bps"]),
            "net_send_bps": tk.DoubleVar(value=thr["net_send_bps"]),
            "net_recv_bps": tk.DoubleVar(value=thr["net_recv_bps"]),
            "temp_c": tk.DoubleVar(value=thr["temp_c"]),
        }
        def add_thr_row(parent, label, var, hint=""):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=6)
            ttk.Label(row, text=label).pack(side="left")
            entry = ttk.Entry(row, width=20, textvariable=var)
            entry.pack(side="left", padx=8)
            if hint:
                ttk.Label(row, text=hint).pack(side="left")
        add_thr_row(thr_frame, "CPU total %", self.thr_vars["cpu_total_percent"])
        add_thr_row(thr_frame, "Memory %", self.thr_vars["mem_percent"])
        add_thr_row(thr_frame, "Disk write B/s", self.thr_vars["disk_write_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Disk read B/s", self.thr_vars["disk_read_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Net send B/s", self.thr_vars["net_send_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Net recv B/s", self.thr_vars["net_recv_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Temperature °C", self.thr_vars["temp_c"])

        drives_frame = ttk.Frame(container)
        container.add(drives_frame, text="Drives")
        self.drive_vars = {}
        ttk.Label(drives_frame, text="Select drives to monitor (local and network mounts):").pack(anchor="w", padx=8, pady=(8, 4))
        canvas = tk.Canvas(drives_frame, height=240)
        scroll_y = ttk.Scrollbar(drives_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scroll_y.pack(side="right", fill="y", pady=8)
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=canvas.winfo_width())
        inner.bind("<Configure>", on_configure)
        for d in self.engine.drive_inventory:
            label = f"{d['device']} ({d['mountpoint']}) [{d['fstype'] or 'fs'}]{' - network' if d['is_network'] else ''}"
            var = tk.BooleanVar(value=(d["key"] in self.engine.selected_drives))
            ttk.Checkbutton(inner, text=label, variable=var).pack(anchor="w", padx=8, pady=2)
            self.drive_vars[d["key"]] = var

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=8)
        ttk.Button(btns, text="Apply", command=self.apply).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def apply(self):
        plugin_mapping = {k: v.get() for k, v in self.plugin_vars.items()}
        thr_mapping = {k: v.get() for k, v in self.thr_vars.items()}
        overlay = self.overlay_var.get()
        selected_keys = [k for k, v in self.drive_vars.items() if v.get()]
        try:
            self.on_apply(plugin_mapping, thr_mapping, overlay, selected_keys)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}")
        self.destroy()

# ---------------------------
# Telemetry dashboard frame (for tab)
# ---------------------------
class TelemetryFrame(ttk.Frame):
    def __init__(self, master, engine: TelemetryEngine):
        super().__init__(master)
        self.engine = engine
        self._build_ui()

    def _build_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=8, pady=8)

        self.start_btn = ttk.Button(control_frame, text="Start", command=self.on_start)
        self.stop_btn  = ttk.Button(control_frame, text="Stop", command=self.on_stop)
        self.log_btn   = ttk.Button(control_frame, text="Enable logging", command=self.on_toggle_log)
        self.adapt_var = tk.BooleanVar(value=True)
        self.adapt_chk = ttk.Checkbutton(control_frame, text="Adaptive interval", variable=self.adapt_var, command=self.on_adaptive_toggle)
        self.settings_btn = ttk.Button(control_frame, text="Settings", command=self.on_settings)

        self.start_btn.pack(side="left", padx=4)
        self.stop_btn.pack(side="left", padx=4)
        self.log_btn.pack(side="left", padx=4)
        self.adapt_chk.pack(side="left", padx=12)
        self.settings_btn.pack(side="right", padx=4)

        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", padx=8)
        ttk.Label(slider_frame, text="Sampling interval (sec)").pack(side="left")
        self.interval_var = tk.DoubleVar(value=self.engine.interval)
        self.interval_slider = ttk.Scale(slider_frame, from_=0.2, to=5.0, variable=self.interval_var, command=self.on_interval_change)
        self.interval_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.interval_read = ttk.Label(slider_frame, text=f"{self.engine.interval:.2f}s")
        self.interval_read.pack(side="right", padx=4)

        self.status_var = tk.StringVar(value="Idle")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.pack(fill="x", padx=8, pady=4)

        self.alert_canvas = tk.Canvas(self, height=36, bg="#202020", highlightthickness=0)
        self.alert_canvas.pack(fill="x", padx=8)
        self.alert_banner_text_id = None

        charts_frame = ttk.Frame(self)
        charts_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.cpu_canvas = tk.Canvas(charts_frame, width=480, height=160, bg="#101820", highlightthickness=0)
        self.mem_canvas = tk.Canvas(charts_frame, width=480, height=160, bg="#101820", highlightthickness=0)
        self.cpu_canvas.pack(side="top", fill="x", expand=True)
        self.mem_canvas.pack(side="top", fill="x", expand=True, pady=(8,0))

        right_frame = ttk.Frame(self)
        right_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.metrics_tree = ttk.Treeview(right_frame, columns=("value",), show="tree headings")
        self.metrics_tree.heading("#0", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("#0", width=320)
        self.metrics_tree.column("value", width=320)
        self.metrics_tree.pack(fill="both", expand=True)

        self.log_path_var = tk.StringVar(value="Logging: disabled")
        ttk.Label(self, textvariable=self.log_path_var).pack(fill="x", padx=8, pady=(0,8))

        self.after(200, self.poll_engine)
        self.stop_btn.state(["disabled"])

    def on_start(self):
        self.engine.start()
        self.status_var.set("Collecting telemetry...")
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])

    def on_stop(self):
        self.engine.stop()
        self.status_var.set("Stopped")
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])

    def on_toggle_log(self):
        enabled = not self.engine.logging_enabled
        self.engine.enable_logging(enabled)
        if enabled and hasattr(self.engine, "log_file_path"):
            self.log_path_var.set(f"Logging: {self.engine.log_file_path}")
            self.log_btn.configure(text="Disable logging")
        else:
            self.log_path_var.set("Logging: disabled")
            self.log_btn.configure(text="Enable logging")

    def on_adaptive_toggle(self):
        self.engine.set_adaptive(self.adapt_var.get())

    def on_interval_change(self, _evt=None):
        val = float(self.interval_var.get())
        self.interval_read.configure(text=f"{val:.2f}s")
        self.engine.set_interval(val)

    def on_settings(self):
        def apply_settings(plugin_mapping, thr_mapping, overlay_mode, selected_drives):
            self.engine.update_enabled_plugins(plugin_mapping)
            self.engine.update_thresholds(thr_mapping)
            self.engine.update_selected_drives(selected_drives)
            try:
                self.winfo_toplevel().attributes("-topmost", overlay_mode)
            except Exception:
                pass
        SettingsDialog(self.winfo_toplevel(), self.engine, on_apply=apply_settings)

    def poll_engine(self):
        try:
            snapshot = self.engine.out_queue.get_nowait()
            self.render(snapshot)
        except queue.Empty:
            pass
        self.after(200, self.poll_engine)

    def render(self, s):
        self.draw_alerts(s.get("_alerts", []))
        self.draw_cpu_chart(s.get("cpu", {}))
        self.draw_mem_chart(s.get("memory", {}))
        self.update_metrics_tree(s)

    def draw_alerts(self, alerts):
        self.alert_canvas.delete("all")
        if alerts:
            self.alert_canvas.configure(bg="#3d0000")
            text = " | ".join(alerts[:4]) + (" ..." if len(alerts) > 4 else "")
            self.alert_canvas.create_text(12, 18, text=f"ALERT: {text}", anchor="w", fill="#ffcccb", font=("Arial", 12, "bold"))
            self.status_var.set("Alerts active")
        else:
            self.alert_canvas.configure(bg="#202020")
            self.alert_canvas.create_text(12, 18, text="No alerts", anchor="w", fill="#9aa0a6", font=("Arial", 11))
            if "Collecting" in self.status_var.get() or "Stopped" in self.status_var.get():
                pass
            else:
                self.status_var.set("Idle")

    def draw_cpu_chart(self, cpu):
        canvas = self.cpu_canvas
        canvas.delete("all")
        w = int(canvas["width"]); h = int(canvas["height"])
        canvas.create_text(8, 12, text="CPU per-core % and total", anchor="w", fill="#9aa0a6")
        total = cpu.get("cpu_percent_total", 0.0) or 0.0
        per_core = cpu.get("cpu_percent_per_core", []) or []
        n = max(1, len(per_core))
        bar_w = max(12, (w - 48) // n)
        for i, v in enumerate(per_core):
            x0 = 16 + i * bar_w
            y0 = h - 24
            bar_h = int((h - 56) * (float(v) / 100.0))
            canvas.create_rectangle(x0, y0 - bar_h, x0 + bar_w - 6, y0, fill="#00c853", outline="")
            canvas.create_text(x0 + bar_w//2 - 4, y0 + 12, text=str(i), fill="#9aa0a6")
        gauge_w = 320
        canvas.create_rectangle(16, 28, 16 + gauge_w, 44, outline="#37474f")
        fill_w = int(gauge_w * (total / 100.0))
        fill_color = "#ffab00" if total < 75 else "#ff6d00"
        canvas.create_rectangle(16, 28, 16 + fill_w, 44, fill=fill_color, outline="")
        canvas.create_text(16 + gauge_w + 8, 36, text=f"{total:.1f}%", anchor="w", fill="#9aa0a6")

    def draw_mem_chart(self, mem):
        canvas = self.mem_canvas
        canvas.delete("all")
        w = int(canvas["width"]); h = int(canvas["height"])
        canvas.create_text(8, 12, text="Memory and swap", anchor="w", fill="#9aa0a6")
        total = mem.get("mem_total", 0) or 1
        used = mem.get("mem_used", 0) or 0
        percent = mem.get("mem_percent", 0.0) or 0.0
        swap_total = mem.get("swap_total", 0) or 1
        swap_used = mem.get("swap_used", 0) or 0
        bar_w = w - 32
        canvas.create_rectangle(16, 28, 16 + bar_w, 44, outline="#37474f")
        canvas.create_rectangle(16, 28, 16 + int(bar_w * (used / total)), 44, fill="#2962ff", outline="")
        canvas.create_text(16 + bar_w + 8, 36, text=f"{percent:.1f}%", anchor="w", fill="#9aa0a6")
        canvas.create_rectangle(16, 64, 16 + bar_w, 80, outline="#37474f")
        canvas.create_rectangle(16, 64, 16 + int(bar_w * (swap_used / swap_total)), 80, fill="#d81b60", outline="")
        canvas.create_text(16, 100, anchor="w", fill="#9aa0a6",
                           text=f"Mem: {fmt_bytes(total)} total, {fmt_bytes(used)} used")
        canvas.create_text(16, 120, anchor="w", fill="#9aa0a6",
                           text=f"Swap: {fmt_bytes(swap_total)} total, {fmt_bytes(swap_used)} used")

    def update_metrics_tree(self, s):
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        cpu = s.get("cpu", {})
        self._add_metric("CPU total %", f"{float(cpu.get('cpu_percent_total', 0.0) or 0.0):.1f}")
        if "cpu_freq_current_mhz" in cpu:
            self._add_metric("CPU freq (MHz)", f"{cpu.get('cpu_freq_current_mhz', 0):.0f}")
        if "load_avg_1m" in cpu:
            self._add_metric("Load avg 1m", f"{cpu.get('load_avg_1m', 0):.2f}")
        temps = cpu.get("temps", {}) or {}
        for k, v in temps.items():
            if v is not None:
                self._add_metric(f"{k}", f"{v:.1f} °C")
        if "battery_percent" in cpu:
            bat = f"{cpu.get('battery_percent', 0):.0f}%"
            plug = "plugged" if cpu.get("battery_plugged") else "on battery"
            self._add_metric("Battery", f"{bat} ({plug})")

        mem = s.get("memory", {})
        self._add_metric("Mem used", fmt_bytes(mem.get("mem_used", 0)))
        self._add_metric("Mem avail", fmt_bytes(mem.get("mem_available", 0)))

        disk = s.get("disk", {}) or {}
        agg = disk.get("aggregate", {}) or {}
        deltas = s.get("_deltas", {}) or {}
        per_bps = deltas.get("perdrive_bps", {}) or {}
        self._add_metric("Disk read total", fmt_bytes(agg.get("disk_read_bytes", 0)))
        self._add_metric("Disk write total", fmt_bytes(agg.get("disk_write_bytes", 0)))
        if "disk_read_bps" in deltas:
            self._add_metric("Disk read rate", f"{fmt_bytes(deltas.get('disk_read_bps', 0))}/s")
        if "disk_write_bps" in deltas:
            self._add_metric("Disk write rate", f"{fmt_bytes(deltas.get('disk_write_bps', 0))}/s")

        perdisk = disk.get("perdisk", {}) or {}
        if perdisk:
            self._add_metric("--- Drives ---", "")
            selected = set(s.get("_selected_drives", []))
            for dev_key, stats in perdisk.items():
                base = basename_device(dev_key)
                if selected and (base not in selected and dev_key not in selected):
                    continue
                rb = stats.get("read_bytes", 0); wb = stats.get("write_bytes", 0)
                rbps = per_bps.get(dev_key, {}).get("read_bps", 0.0)
                wbps = per_bps.get(dev_key, {}).get("write_bps", 0.0)
                self._add_metric(f"{dev_key} read", f"{fmt_bytes(rb)} ({fmt_bytes(rbps)}/s)")
                self._add_metric(f"{dev_key} write", f"{fmt_bytes(wb)} ({fmt_bytes(wbps)}/s)")

        net = s.get("network", {}) or {}
        self._add_metric("Net sent", fmt_bytes(net.get("net_bytes_sent", 0)))
        self._add_metric("Net recv", fmt_bytes(net.get("net_bytes_recv", 0)))
        if "net_send_bps" in deltas:
            self._add_metric("Net send rate", f"{fmt_bytes(deltas.get('net_send_bps', 0))}/s")
        if "net_recv_bps" in deltas:
            self._add_metric("Net recv rate", f"{fmt_bytes(deltas.get('net_recv_bps', 0))}/s")

        gpu = s.get("gpu", {}).get("gpus", [])
        if gpu:
            g0 = gpu[0]
            self._add_metric(f"GPU0 {g0.get('name','')}", f"{g0.get('gpu_util',0)}% util, {g0.get('temp_c','n/a')} °C")
            self._add_metric("GPU0 mem", f"{fmt_bytes(g0.get('mem_used',0))}/{fmt_bytes(g0.get('mem_total',0))}")

        procs = s.get("process", {}).get("top_processes", [])
        for p in procs:
            name = p.get("name","proc")
            cpuv = p.get("cpu", 0.0)
            rss = p.get("rss", 0)
            self._add_metric(f"PID {p['pid']} {name}", f"{cpuv:.1f}% | {fmt_bytes(rss)}")

    def _add_metric(self, key, value):
        self.metrics_tree.insert("", "end", text=key, values=(value,))

# ---------------------------
# Foresight compact frame (for tab)
# ---------------------------
class ForesightFrame(ttk.Frame):
    def __init__(self, master, foresight: Foresight, update_ms: int = 1000):
        super().__init__(master)
        self.fs = foresight
        self.update_ms = update_ms
        self.stop_event = threading.Event()
        self._build_layout()
        self.scheduler = Scheduler(self.fs, self.stop_event)
        self.scheduler.start()
        self.after(self.update_ms, self._tick)

    def _build_layout(self):
        container = ttk.Frame(self, padding=6)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)

        top = ttk.Frame(container)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        top.columnconfigure(0, weight=1)
        self.suggestion_var = tk.StringVar(value="Suggestion: —")
        sugg_label = ttk.Label(top, textvariable=self.suggestion_var)
        sugg_label.grid(row=0, column=0, sticky="w")

        self.learning_var = tk.BooleanVar(value=True)
        learn_btn = ttk.Checkbutton(top, text="Learning", variable=self.learning_var, command=self._toggle_learning)
        learn_btn.grid(row=0, column=1, sticky="e")

        self.advisory_var = tk.StringVar(value="")
        advisory_label = ttk.Label(top, textvariable=self.advisory_var, foreground="#b22222")
        advisory_label.grid(row=1, column=0, columnspan=2, sticky="w")

        left = ttk.LabelFrame(container, text="Top habits", padding=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, height=10)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        right = ttk.LabelFrame(container, text="Frequency", padding=6)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.figure = None; self.ax = None; self.canvas = None
        if plt is not None and FigureCanvasTkAgg is not None:
            self.figure = plt.Figure(figsize=(3.6, 2.2), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Frequency")
            self.ax.set_ylabel("Count")
            self.ax.tick_params(axis='x', rotation=35, labelsize=7)
            self.canvas = FigureCanvasTkAgg(self.figure, master=right)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
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
        msg = f"{alert.message}\n\nChoose an action:\n- Kill: terminate the process\n- Allow: whitelist and ignore alerts for this item\n- Monitor: keep observing and alert if it escalates"
        resp = messagebox.askquestion("Intrusion detected", msg, icon="warning")
        if resp == "yes":
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
            suggested = self.fs.predict_next()
            self.suggestion_var.set(f"Suggestion: {suggested if suggested else '—'}")

            adv = self._poll_advisory()
            self.advisory_var.set(adv if adv else "")

            alert = self._poll_alert()
            if alert:
                self._show_intrusion_prompt(alert)

            self.listbox.delete(0, tk.END)
            for h, c in self.fs.get_top_habits(20):
                self.listbox.insert(tk.END, f"{h}  —  {c}")

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

        self.after(self.update_ms, self._tick)

    def stop(self):
        self.stop_event.set()

# ---------------------------
# Unified application
# ---------------------------
class UnifiedApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified Telemetry + Foresight Dashboard")
        try:
            style = ttk.Style(self.root)
            if platform.system() == "Windows":
                style.theme_use("vista")
            else:
                style.theme_use("clam")
        except Exception:
            pass
        self.root.geometry("1200x800")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)

        # Telemetry
        self.engine = TelemetryEngine(interval=1.0, adaptive=True)
        self.telemetry_tab = TelemetryFrame(self.nb, self.engine)
        self.nb.add(self.telemetry_tab, text="Telemetry")

        # Foresight
        self.foresight = Foresight(cache_file="foresight_cache.bin")
        self.foresight_tab = ForesightFrame(self.nb, self.foresight, update_ms=1000)
        self.nb.add(self.foresight_tab, text="Foresight")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        try:
            self.foresight_tab.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ---------------------------
# Entry point
# ---------------------------
def main():
    app = UnifiedApp()
    app.run()

if __name__ == "__main__":
    main()

