#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foresight: Learn user habits, cache them for fast access, and visualize with a GUI.

Features:
- Habit tracking engine: frequency, timestamps, time-of-day histograms.
- JSON cache for persistence and instant retrieval.
- Auto-loader for required libraries.
- DPI-aware, responsive Tkinter GUI with Matplotlib charts.
- Controls: record habit, predict next, clear cache, export CSV, import JSON.
- Live updating panels and safe, thread-aware operations.

Author: You
"""

import sys
import os
import json
import time
import threading
import queue
import csv
import datetime
import traceback
import importlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Attempt to import Tkinter (usually bundled with Python); handle name differences
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:
    print("[foresight] Tkinter not available or failed to import.")
    print(e)
    sys.exit(1)

# Auto-loader for pip-managed libraries (e.g., matplotlib)
def auto_loader(libraries: List[str]) -> None:
    """Ensure all required libraries are installed and importable."""
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[foresight] Installing missing library: {lib}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception as e:
                print(f"[foresight] Failed to install {lib}: {e}")
                # Continue; GUI may degrade gracefully

# Ensure non-standard libraries are present
auto_loader(["matplotlib"])

# Now import Matplotlib components
try:
    import matplotlib
    matplotlib.use("Agg")  # Prevent backend conflicts during figure creation
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception as e:
    print("[foresight] Matplotlib not available; charts will be disabled.")
    print(e)
    plt = None
    FigureCanvasTkAgg = None

# ----------------------------
# Habit tracking engine
# ----------------------------
class HabitStats:
    """Lightweight stats container for a habit."""
    __slots__ = ("count", "last_ts", "tod_hist")  # time-of-day histogram: hour -> count

    def __init__(self, count: int = 0, last_ts: float = 0.0, tod_hist: Optional[Dict[int, int]] = None):
        self.count = count
        self.last_ts = last_ts
        self.tod_hist = tod_hist if tod_hist is not None else {h: 0 for h in range(24)}

    def to_dict(self) -> Dict:
        return {"count": self.count, "last_ts": self.last_ts, "tod_hist": self.tod_hist}

    @staticmethod
    def from_dict(d: Dict) -> "HabitStats":
        hs = HabitStats(
            count=int(d.get("count", 0)),
            last_ts=float(d.get("last_ts", 0.0)),
            tod_hist={int(k): int(v) for k, v in d.get("tod_hist", {h: 0 for h in range(24)}).items()}
        )
        return hs


class Foresight:
    """Core engine for tracking and predicting user habits with caching."""
    def __init__(self, cache_file: str = "foresight_cache.json"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self.habits: Dict[str, HabitStats] = {}
        self.event_log: List[Tuple[float, str]] = []  # list of (timestamp, habit)
        self._load_cache()

    def _load_cache(self) -> None:
        with self._lock:
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    habits_data = data.get("habits", {})
                    self.habits = {h: HabitStats.from_dict(stats) for h, stats in habits_data.items()}
                    self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-10000:]
                except Exception:
                    # Corrupt cache should not break run; start fresh
                    self.habits = {}
                    self.event_log = []

    def _save_cache(self) -> None:
        with self._lock:
            data = {
                "habits": {h: stats.to_dict() for h, stats in self.habits.items()},
                "event_log": self.event_log[-10000:]
            }
            tmp_file = self.cache_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_file, self.cache_file)

    def record_habit(self, habit: str, ts: Optional[float] = None) -> None:
        """Record a habit occurrence; updates frequency, last_ts, and time-of-day histogram."""
        if not habit or not habit.strip():
            return
        habit = habit.strip()
        now = ts if ts is not None else time.time()
        hour = datetime.datetime.fromtimestamp(now).hour
        with self._lock:
            stats = self.habits.get(habit)
            if stats is None:
                stats = HabitStats()
                self.habits[habit] = stats
            stats.count += 1
            stats.last_ts = now
            stats.tod_hist[hour] = stats.tod_hist.get(hour, 0) + 1
            self.event_log.append((now, habit))
            if len(self.event_log) > 10000:
                self.event_log = self.event_log[-10000:]
        self._save_cache()

    def get_top_habits(self, n: int = 10) -> List[Tuple[str, int]]:
        """Fast retrieval of top habits by count."""
        with self._lock:
            return sorted(((h, s.count) for h, s in self.habits.items()), key=lambda x: x[1], reverse=True)[:n]

    def has_habit(self, habit: str) -> bool:
        with self._lock:
            return habit in self.habits

    def predict_next(self, when_ts: Optional[float] = None) -> Optional[str]:
        """
        Predict next likely habit based on:
        - Time-of-day histogram (dominant habit for current hour).
        - Recent recency bias (last N events).
        Returns the predicted habit or None.
        """
        with self._lock:
            if not self.habits:
                return None

            ts = when_ts if when_ts is not None else time.time()
            hour = datetime.datetime.fromtimestamp(ts).hour

            # Time-of-day dominant habit
            tod_candidates = sorted(
                ((h, s.tod_hist.get(hour, 0)) for h, s in self.habits.items()),
                key=lambda x: x[1], reverse=True
            )

            tod_choice = tod_candidates[0][0] if tod_candidates and tod_candidates[0][1] > 0 else None

            # Recency bias: prefer most recent repeated habit
            recent_window = self.event_log[-30:]
            if recent_window:
                freq_recent = defaultdict(int)
                for _, h in recent_window:
                    freq_recent[h] += 1
                recency_choice = max(freq_recent.items(), key=lambda x: x[1])[0]
            else:
                recency_choice = None

            # Blend strategies: prefer ToD if strong, else recency, else global top
            if tod_choice is not None and self.habits[tod_choice].tod_hist.get(hour, 0) >= 2:
                return tod_choice
            if recency_choice is not None:
                return recency_choice
            # fallback to global top
            tops = self.get_top_habits(1)
            return tops[0][0] if tops else None

    def export_csv(self, path: str) -> None:
        """Export habits and stats to CSV."""
        with self._lock:
            rows = [("habit", "count", "last_ts_iso", "hour", "hour_count")]
            for h, s in self.habits.items():
                last_iso = datetime.datetime.fromtimestamp(s.last_ts).isoformat() if s.last_ts else ""
                for hour in range(24):
                    rows.append((h, s.count, last_iso, hour, s.tod_hist.get(hour, 0)))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def import_json(self, path: str) -> None:
        """Import habits cache from a JSON file and merge."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}")

        with self._lock:
            habits_data = data.get("habits", {})
            for h, stats_dict in habits_data.items():
                incoming = HabitStats.from_dict(stats_dict)
                if h in self.habits:
                    # merge by max count, recent ts, and sum hist
                    current = self.habits[h]
                    current.count = max(current.count, incoming.count)
                    current.last_ts = max(current.last_ts, incoming.last_ts)
                    for hour in range(24):
                        current.tod_hist[hour] = current.tod_hist.get(hour, 0) + incoming.tod_hist.get(hour, 0)
                else:
                    self.habits[h] = incoming

            # Merge event logs conservatively
            incoming_events = [(float(ts), str(h)) for ts, h in data.get("event_log", [])]
            self.event_log.extend(incoming_events)
            self.event_log = self.event_log[-10000:]

        self._save_cache()

    def clear_cache(self) -> None:
        with self._lock:
            self.habits.clear()
            self.event_log.clear()
        self._save_cache()


# ----------------------------
# GUI
# ----------------------------
class ForesightGUI:
    """Tkinter + Matplotlib GUI for the Foresight engine."""
    def __init__(self, foresight: Foresight, update_interval_ms: int = 1000):
        self.foresight = foresight
        self.update_interval_ms = update_interval_ms
        self._queue = queue.Queue()
        self._stop_event = threading.Event()

        self.root = tk.Tk()
        self.root.title("Foresight Habit Tracker")
        # DPI-aware scaling
        try:
            self.root.call('tk', 'scaling', 1.25)  # Adjust scale if needed
        except Exception:
            pass

        # Window geometry and weights for responsiveness
        self.root.geometry("900x600")
        self.root.minsize(800, 500)

        self._build_layout()
        self._setup_menu()
        self._start_background_updater()

        # Clean exit
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Export CSV...", command=self._export_csv)
        filemenu.add_command(label="Import JSON...", command=self._import_json)
        filemenu.add_separator()
        filemenu.add_command(label="Clear Cache", command=self._clear_cache)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        actions = tk.Menu(menubar, tearoff=0)
        actions.add_command(label="Predict Next Habit", command=self._predict_next)
        menubar.add_cascade(label="Actions", menu=actions)

        self.root.config(menu=menubar)

    def _build_layout(self):
        # Main container
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        # Configure grid
        container.columnconfigure(0, weight=1, uniform="col")
        container.columnconfigure(1, weight=2, uniform="col")
        container.rowconfigure(0, weight=1)

        # Left panel: controls + habit list
        left = ttk.Frame(container, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        # Controls
        controls = ttk.LabelFrame(left, text="Controls", padding=10)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.habit_entry = ttk.Entry(controls)
        self.habit_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        record_btn = ttk.Button(controls, text="Record", command=self._record_habit)
        record_btn.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        predict_btn = ttk.Button(controls, text="Predict", command=self._predict_next)
        predict_btn.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        # Prediction label
        self.pred_label_var = tk.StringVar(value="Prediction: —")
        pred_label = ttk.Label(controls, textvariable=self.pred_label_var)
        pred_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # Habit list
        list_frame = ttk.LabelFrame(left, text="Top Habits", padding=10)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.habit_listbox = tk.Listbox(list_frame)
        self.habit_listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.habit_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.habit_listbox.config(yscrollcommand=list_scroll.set)

        # Right panel: chart
        right = ttk.Frame(container, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        chart_frame = ttk.LabelFrame(right, text="Habit Frequency", padding=10)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.chart_frame = chart_frame

        # Prepare Matplotlib figure
        self.figure = None
        self.canvas = None
        if plt is not None and FigureCanvasTkAgg is not None:
            self.figure = plt.Figure(figsize=(6, 4), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Habit Frequency")
            self.ax.set_ylabel("Count")
            self.ax.tick_params(axis='x', rotation=35)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
            widget = self.canvas.get_tk_widget()
            widget.grid(row=0, column=0, sticky="nsew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", side="bottom")

    def _start_background_updater(self):
        # Periodic UI updates
        self.root.after(self.update_interval_ms, self._update_ui_periodic)

    def _update_ui_periodic(self):
        try:
            self._refresh_habit_list()
            self._refresh_chart()
            # Update prediction display
            predicted = self.foresight.predict_next()
            self.pred_label_var.set(f"Prediction: {predicted if predicted else '—'}")
            self.status_var.set(f"Habits: {len(self.foresight.habits)} | Events: {len(self.foresight.event_log)}")
        except Exception as e:
            self.status_var.set(f"Update error: {e}")
        finally:
            if not self._stop_event.is_set():
                self.root.after(self.update_interval_ms, self._update_ui_periodic)

    def _refresh_habit_list(self):
        self.habit_listbox.delete(0, tk.END)
        for habit, freq in self.foresight.get_top_habits(50):
            self.habit_listbox.insert(tk.END, f"{habit}  —  {freq}")

    def _refresh_chart(self):
        if self.canvas is None or self.figure is None:
            return
        self.ax.clear()
        self.ax.set_title("Habit Frequency")
        self.ax.set_ylabel("Count")

        data = self.foresight.get_top_habits(15)
        if data:
            habits = [h for h, _ in data]
            counts = [c for _, c in data]
            self.ax.bar(habits, counts, color="#66b3ff")
            self.ax.tick_params(axis='x', rotation=35, labelsize=8)
        else:
            self.ax.text(0.5, 0.5, "No habits yet", ha="center", va="center")

        self.canvas.draw()

    # ---- Actions ----
    def _record_habit(self):
        habit = self.habit_entry.get().strip()
        if not habit:
            messagebox.showinfo("Record Habit", "Enter a habit name first.")
            return
        try:
            self.foresight.record_habit(habit)
            self.habit_entry.delete(0, tk.END)
            self.status_var.set(f"Recorded habit: {habit}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to record habit:\n{e}\n\n{traceback.format_exc()}")

    def _predict_next(self):
        try:
            predicted = self.foresight.predict_next()
            if predicted:
                self.pred_label_var.set(f"Prediction: {predicted}")
                self.status_var.set(f"Predicted next habit: {predicted}")
            else:
                self.pred_label_var.set("Prediction: —")
                self.status_var.set("No prediction available yet.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export Habits to CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.foresight.export_csv(path)
            self.status_var.set(f"Exported to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def _import_json(self):
        path = filedialog.askopenfilename(
            title="Import Habits JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.foresight.import_json(path)
            self.status_var.set(f"Imported from {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Import failed:\n{e}")

    def _clear_cache(self):
        if messagebox.askyesno("Clear Cache", "This will remove all cached habits and events. Continue?"):
            try:
                self.foresight.clear_cache()
                self.status_var.set("Cache cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Clear cache failed:\n{e}")

    def _on_close(self):
        self._stop_event.set()
        try:
            self.root.after_cancel(self._update_ui_periodic)
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ----------------------------
# Main
# ----------------------------
def main():
    foresight = Foresight(cache_file="foresight_cache.json")
    gui = ForesightGUI(foresight, update_interval_ms=1000)
    gui.run()


if __name__ == "__main__":
    main()

