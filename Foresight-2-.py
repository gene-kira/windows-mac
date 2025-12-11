#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foresight (autonomous, resource-aware):
- Learns user habits, caches them, and provides fast predictions.
- Adaptive predictor (bandit-lite) blends time-of-day, recency, and global frequency.
- Minimal GUI: suggestion banner, top habits list, frequency chart, single "Learning" toggle.
- Disabled habit list: habits not used daily are parked to save resources; auto-reactivated on use.
- Background scheduler: ingests passive events, reinforces predictions, performs daily review.
- Local-only JSON cache with atomic writes for resilience.

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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

# Ensure non-standard libs for charting
auto_loader(["matplotlib"])

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
    """
    Blends signals: time-of-day (ToD), recency, global frequency.
    Learns weights via simple bandit-style reinforcement.
    """
    def __init__(self):
        self.weights = {"tod": 1.0, "recency": 1.0, "global": 1.0}
        self.lr = 0.05     # learning rate
        self.decay = 0.999 # slow decay
        self._lock = threading.RLock()

    def predict(self, hour: int, habits: Dict[str, HabitStats], event_log: List[Tuple[float, str]]) -> Optional[str]:
        if not habits:
            return None
        with self._lock:
            tod_scores = {h: s.tod_hist.get(hour, 0) for h, s in habits.items()}
            recent = event_log[-50:]
            recency = defaultdict(int)
            for _, h in recent:
                recency[h] += 1
            global_scores = {h: s.count for h, s in habits.items()}

            def norm(d: Dict[str, int]) -> Dict[str, float]:
                total = float(sum(d.values())) or 1.0
                return {k: (v / total) for k, v in d.items()}

            tod_n = norm(tod_scores)
            recency_n = norm(recency)
            global_n = norm(global_scores)

            scores = defaultdict(float)
            for h in habits.keys():
                scores[h] = (
                    self.weights["tod"] * tod_n.get(h, 0.0)
                    + self.weights["recency"] * recency_n.get(h, 0.0)
                    + self.weights["global"] * global_n.get(h, 0.0)
                )
            choice = max(scores.items(), key=lambda x: x[1])[0]
            return choice

    def reward(self, matched: bool) -> None:
        with self._lock:
            delta = self.lr if matched else -self.lr
            for k in self.weights:
                self.weights[k] = max(0.05, self.weights[k] * self.decay + delta)

# ----------------------------
# Foresight engine
# ----------------------------
class Foresight:
    def __init__(self, cache_file: str = "foresight_cache.json"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self.habits: Dict[str, HabitStats] = {}           # active habits
        self.disabled_habits: Dict[str, HabitStats] = {}  # parked habits to save resources
        self.event_log: List[Tuple[float, str]] = []
        self.predictor = AdaptivePredictor()
        self.learning_enabled = True
        self.activity_score = 0.0  # used to auto-tune polling interval
        self.last_review_day = None
        self._load_cache()

    # ----- Persistence -----
    def _load_cache(self) -> None:
        with self._lock:
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.habits = {h: HabitStats.from_dict(s) for h, s in data.get("habits", {}).items()}
                    self.disabled_habits = {h: HabitStats.from_dict(s) for h, s in data.get("disabled", {}).items()}
                    self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-20000:]
                    w = data.get("weights", {"tod": 1.0, "recency": 1.0, "global": 1.0})
                    self.predictor.weights = {k: float(v) for k, v in w.items()}
                    self.last_review_day = data.get("last_review_day")
                except Exception:
                    self.habits = {}
                    self.disabled_habits = {}
                    self.event_log = []
                    self.last_review_day = None

    def _save_cache(self) -> None:
        with self._lock:
            data = {
                "habits": {h: s.to_dict() for h, s in self.habits.items()},
                "disabled": {h: s.to_dict() for h, s in self.disabled_habits.items()},
                "event_log": self.event_log[-20000:],
                "weights": self.predictor.weights,
                "last_review_day": self.last_review_day,
            }
            tmp = self.cache_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.cache_file)

    # ----- Learning & recording -----
    def record_habit(self, habit: str, ts: Optional[float] = None) -> None:
        if not habit or not habit.strip():
            return
        habit = habit.strip()
        now = ts if ts is not None else time.time()
        hour = datetime.datetime.fromtimestamp(now).hour
        with self._lock:
            # Reactivate if in disabled
            if habit in self.disabled_habits:
                self.habits[habit] = self.disabled_habits.pop(habit)
            stats = self.habits.get(habit)
            if stats is None:
                stats = HabitStats()
                self.habits[habit] = stats
            stats.count += 1
            stats.last_ts = now
            stats.tod_hist[hour] = stats.tod_hist.get(hour, 0) + 1
            self.event_log.append((now, habit))
            if len(self.event_log) > 20000:
                self.event_log = self.event_log[-20000:]
            self.activity_score = min(1.0, self.activity_score + 0.1)
        self._save_cache()

    def get_top_habits(self, n: int = 12) -> List[Tuple[str, int]]:
        with self._lock:
            return sorted(((h, s.count) for h, s in self.habits.items()), key=lambda x: x[1], reverse=True)[:n]

    def predict_next(self, when_ts: Optional[float] = None) -> Optional[str]:
        ts = when_ts if when_ts is not None else time.time()
        hour = datetime.datetime.fromtimestamp(ts).hour
        with self._lock:
            return self.predictor.predict(hour, self.habits, self.event_log)

    def reinforce(self) -> None:
        """
        Compare last suggestion to actual and reward the predictor.
        Approximation: check if predicted habit equals the most recent habit.
        """
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

    # ----- Daily review & resource-aware disabling -----
    def daily_review(self) -> int:
        """
        Run once per day:
        - Move habits with last_ts older than 24h to disabled list.
        - Keep active habits available for prediction.
        Returns the number of habits disabled.
        """
        with self._lock:
            now = time.time()
            today_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            if self.last_review_day == today_str:
                return 0  # already reviewed today

            cutoff = now - 24 * 3600
            disabled_count = 0
            for habit, stats in list(self.habits.items()):
                if stats.last_ts < cutoff:
                    self.disabled_habits[habit] = self.habits.pop(habit)
                    disabled_count += 1
            self.last_review_day = today_str
        self._save_cache()
        return disabled_count

    # ----- Passive ingestion stub -----
    def ingest_passive_events(self) -> int:
        """
        Passive ingestion stub for Windows signals.
        Replace with real hooks: app focus, process usage, file ops, etc.
        Current synthetic behavior simulates time-of-day habits.
        """
        if not self.learning_enabled:
            return 0
        now = time.time()
        hour = datetime.datetime.fromtimestamp(now).hour
        synthetic = None
        if hour in (8, 9):
            synthetic = "morning_check"
        elif hour in (12, 13):
            synthetic = "lunch_break"
        elif hour in (18, 19):
            synthetic = "evening_review"

        if synthetic:
            self.record_habit(synthetic, ts=now)
            self.reinforce()
            return 1
        return 0

    # ----- Controls -----
    def set_learning(self, enabled: bool) -> None:
        with self._lock:
            self.learning_enabled = enabled

# ----------------------------
# Background scheduler
# ----------------------------
class Scheduler(threading.Thread):
    """
    Autonomously:
    - Ingests passive events.
    - Reinforces learning.
    - Performs daily review to disable unused habits.
    Adapts interval based on activity score and learning toggle.
    """
    def __init__(self, foresight: Foresight, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.fs = foresight
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Ensure daily review runs at least once per calendar day
                self.fs.daily_review()

                # Base interval adaptive: active -> faster; idle -> slower
                with self.fs._lock:
                    active = self.fs.activity_score
                    learning = self.fs.learning_enabled
                base_interval = 2.0 if learning else 6.0
                interval = max(1.0, base_interval * (1.5 - active))

                # Ingest and reinforce
                ingested = self.fs.ingest_passive_events()
                if ingested == 0 and learning:
                    self.fs.reinforce()

                time.sleep(interval)
            except Exception:
                # Keep going; autonomy should not crash
                time.sleep(3.0)

# ----------------------------
# Minimal, effective GUI
# ----------------------------
class ForesightGUI:
    def __init__(self, foresight: Foresight, update_ms: int = 1000):
        self.fs = foresight
        self.update_ms = update_ms
        self.stop_event = threading.Event()
        self.root = tk.Tk()
        self.root.title("Foresight")
        try:
            self.root.call('tk', 'scaling', 1.2)
        except Exception:
            pass
        self.root.geometry("900x580")
        self.root.minsize(780, 480)

        self.figure = None
        self.ax = None
        self.canvas = None
        self._build_layout()

        self.scheduler = Scheduler(self.fs, self.stop_event)
        self.scheduler.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)

        # Top bar: suggestion banner + single toggle
        top = ttk.Frame(container)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=1)
        self.suggestion_var = tk.StringVar(value="Suggestion: —")
        sugg_label = ttk.Label(top, textvariable=self.suggestion_var, font=("Segoe UI", 11, "bold"))
        sugg_label.grid(row=0, column=0, sticky="w")

        self.learning_var = tk.BooleanVar(value=True)
        learn_btn = ttk.Checkbutton(top, text="Learning", variable=self.learning_var, command=self._toggle_learning)
        learn_btn.grid(row=0, column=1, sticky="e")

        # Left: top habits list
        left = ttk.LabelFrame(container, text="Top habits", padding=10)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        # Right: simple chart (optional)
        right = ttk.LabelFrame(container, text="Frequency", padding=10)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        if plt is not None and FigureCanvasTkAgg is not None:
            self.figure = plt.Figure(figsize=(5.2, 3.6), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Habit frequency")
            self.ax.set_ylabel("Count")
            self.ax.tick_params(axis='x', rotation=35)
            self.canvas = FigureCanvasTkAgg(self.figure, master=right)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", side="bottom")

    def _toggle_learning(self):
        self.fs.set_learning(self.learning_var.get())
        self.status_var.set("Learning is " + ("ON" if self.learning_var.get() else "OFF"))

    def _tick(self):
        try:
            # Update suggestion
            suggested = self.fs.predict_next()
            self.suggestion_var.set(f"Suggestion: {suggested if suggested else '—'}")

            # Update habits list
            self.listbox.delete(0, tk.END)
            for h, c in self.fs.get_top_habits(30):
                self.listbox.insert(tk.END, f"{h}  —  {c}")

            # Update chart
            if self.canvas is not None and self.ax is not None:
                self.ax.clear()
                self.ax.set_title("Habit frequency")
                self.ax.set_ylabel("Count")
                data = self.fs.get_top_habits(15)
                if data:
                    habits = [h for h, _ in data]
                    counts = [c for _, c in data]
                    self.ax.bar(habits, counts, color="#66b3ff")
                    self.ax.tick_params(axis='x', rotation=35, labelsize=8)
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
                f"Active habits: {total} | Disabled: {disabled_total} | Events: {events} | Activity: {act:.2f} | Learning: {'ON' if learn else 'OFF'}"
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
    fs = Foresight(cache_file="foresight_cache.json")
    gui = ForesightGUI(fs, update_ms=1000)
    gui.run()

if __name__ == "__main__":
    main()

