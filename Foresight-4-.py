#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foresight (compact, autonomous, secure, resource-aware, intrusion-aware):
- System-wide habit learner: apps, utilities, games, and website-like titles (via browser window title).
- Adaptive predictor (bandit-lite): blends time-of-day, recency, global frequency.
- Encrypted local cache: device-fingerprint key (hostname, MAC, timezone, locale, screen).
- Daily auto-disable: unused habits parked; light probe for reactivation.
- Intrusion detection:
  - Flags rogue/mysterious processes by deviation from habits, resource spikes, odd timing, unknown reputation.
  - GUI alerts with actions: Kill, Allow (whitelist), Monitor (watchlist).
- Compact GUI (~450x290): Suggestion banner, advisory/alert line, Top habits, Frequency, Learning toggle.

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

# Foreground window title (Windows)
try:
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.SetProcessDPIAware()
    def get_foreground_window_title() -> Optional[str]:
        hwnd = user32.GetForegroundWindow()
        if hwnd == 0:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        return title if title else None
except Exception:
    def get_foreground_window_title() -> Optional[str]:
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
        try:
            return f"{user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}"
        except Exception:
            return "unknown_screen"

    def _derive_key(self) -> bytes:
        hostname = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or \
                   (os.uname().nodename if hasattr(os, "uname") else "unknown_host")
        mac = uuid.getnode()
        tz = time.tzname[0] if time.tzname else "unknown_tz"
        loc = locale.getdefaultlocale()[0] or "unknown_locale"
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
        if parent_name and parent_name.lower() in ["powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe"]:
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
# Adaptive predictor (contextual bandit-lite)
# ----------------------------
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

# ----------------------------
# System monitor
# ----------------------------
class SystemMonitor:
    def __init__(self):
        self.browser_markers = ["Chrome", "Edge", "Firefox", "Brave", "Opera"]
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
        habit_key examples: "proc:chrome.exe", "game:Game.exe", "work:excel.exe"
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
                    # Only reactivate if CPU rises above low threshold
                    if cpu > self.cpu_threshold:
                        habits.append((key, pid, cpu, mem_mb, parent_name))
                    continue

                # Active monitoring
                if cpu >= self.cpu_threshold or mem_mb >= self.mem_threshold_mb:
                    habits.append((key, pid, cpu, mem_mb, parent_name))
                    lname = name.lower()
                    if any(tag in lname for tag in ["steam", "epic", "game"]):
                        habits.append((self.normalize_habit(f"game:{name}"), pid, cpu, mem_mb, parent_name))
                    if any(tag in lname for tag in ["powershell", "cmd", "terminal", "vscode", "pycharm", "excel", "word", "outlook"]):
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
# Foresight engine (encrypted cache + security + alerts)
# ----------------------------
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
                self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-30000:]
                w = data.get("weights", {"tod": 1.0, "recency": 1.0, "global": 1.0})
                self.predictor.weights = {k: float(v) for k, v in w.items()}
                self.last_review_day = data.get("last_review_day")
                wl = data.get("whitelist", [])
                self.guard.whitelist = set(wl)
                # Rehydrate monitor disabled set
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

    # ----- Learning & recording -----
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
            # Log advisory event for GUI banner
            with self._lock:
                self.event_log.append((time.time(), f"advisory:{msg}"))
                if len(self.event_log) > 30000:
                    self.event_log = self.event_log[-30000:]
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
                hour = datetime.datetime.fromtimestamp(time.time()).hour
                today = datetime.datetime.now().strftime("%Y-%m-%d")
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

    def monitor_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.watchlist.add(habit_key)

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
                        if len(self.fs.event_log) > 30000:
                            self.fs.event_log = self.fs.event_log[-30000:]
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
        # askquestion returns 'yes' or 'no'; we need 3 options, so use additional prompts
        # First prompt for Kill (Yes/No)
        if resp == "yes":  # interpret as Kill
            success = (alert.pid is not None) and self.fs.kill_process(alert.pid)
            self.status_var.set("Process terminated." if success else "Failed to terminate process.")
            return
        # If not Kill, ask Allow or Monitor
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

