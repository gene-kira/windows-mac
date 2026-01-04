#!/usr/bin/env python3
# MagicBox AI – Unified Aim & Vision Lab (Safe Build)
# - Tkinter main app (Back 4 Blood Old-Guy Helper + MagicBox)
# - OPTIONAL Camera Lab (OpenCV + Pygame tracking, launched via button)
# - OPTIONAL Enemy Training Sandbox (Pygame, launched via button)
#
# IMPORTANT:
# This program NEVER hooks or modifies any external game.
# All aiming, enemies, and overlays are confined to its own windows.

import sys
import subprocess
import importlib
import os
import json
import time
import threading
import math
import random
import logging

# ---------------------------------------------
# Auto-install required libraries
# ---------------------------------------------

def ensure(package, pip_name=None):
    """Import a package or install it via pip if missing."""
    if pip_name is None:
        pip_name = package
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(package)

psutil = ensure("psutil")
pyttsx3 = ensure("pyttsx3")
pynput = ensure("pynput")
cv2 = ensure("cv2", "opencv-python")
np = ensure("numpy")
pygame = ensure("pygame")

try:
    import winsound
except ImportError:
    winsound = None

import ctypes
from ctypes import wintypes
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pynput import mouse
from collections import deque

# ---------------------------------------------
# Pure Python Hungarian Algorithm (no SciPy)
# ---------------------------------------------

def linear_sum_assignment(cost_matrix):
    """
    Minimal Hungarian algorithm implementation for assignment.
    cost_matrix: 2D list or numpy array (n x m)
    Returns: (row_ind, col_ind) as numpy arrays
    """
    import numpy as _np
    cost = _np.array(cost_matrix, dtype=float)
    n, m = cost.shape
    u = _np.zeros(n + 1)
    v = _np.zeros(m + 1)
    p = _np.zeros(m + 1, dtype=int)
    way = _np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = _np.full(m + 1, float("inf"))
        used = _np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = _np.zeros(n, dtype=int)
    col_ind = _np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] > 0:
            row_ind[p[j] - 1] = p[j] - 1
            col_ind[p[j] - 1] = j - 1

    return row_ind, col_ind

# ---------------------------------------------
# File paths / logging
# ---------------------------------------------

PROFILE_FILE = "b4b_profiles.json"
MAGIC_MEMORY_FILE = "magic_profile.json"
LOG_FILE = "gamelog.txt"

logging.basicConfig(
    filename='magicbox_unified_lab.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ---------------------------------------------
# Windows helpers / game detection (for status only)
# ---------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = wintypes.HWND

GetWindowTextW = user32.GetWindowTextW
GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
GetWindowTextW.restype = ctypes.c_int

def get_foreground_window_title():
    hwnd = GetForegroundWindow()
    if not hwnd:
        return ""
    length = 512
    buffer = ctypes.create_unicode_buffer(length)
    GetWindowTextW(hwnd, buffer, length)
    return buffer.value or ""

GAME_PROCESS_NAME = "Back4Blood.exe"
GAME_TITLE_KEYWORD = "Back 4 Blood"

def is_game_running():
    for proc in psutil.process_iter(attrs=["name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() == GAME_PROCESS_NAME.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def is_game_foreground():
    title = get_foreground_window_title()
    return GAME_TITLE_KEYWORD.lower() in title.lower()

# ---------------------------------------------
# Shared CameraStats (for AI brain)
# ---------------------------------------------

class CameraStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.global_anomaly = False
        self.global_motion_z = 0.0
        self.avg_track_speed = 0.0
        self.num_tracks = 0
        self.num_anomaly_tracks = 0
        self.last_update_time = 0.0

    def update(self, global_anomaly, global_motion_z, avg_track_speed,
               num_tracks, num_anomaly_tracks):
        with self.lock:
            self.global_anomaly = global_anomaly
            self.global_motion_z = float(global_motion_z)
            self.avg_track_speed = float(avg_track_speed)
            self.num_tracks = int(num_tracks)
            self.num_anomaly_tracks = int(num_anomaly_tracks)
            self.last_update_time = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "global_anomaly": self.global_anomaly,
                "global_motion_z": self.global_motion_z,
                "avg_track_speed": self.avg_track_speed,
                "num_tracks": self.num_tracks,
                "num_anomaly_tracks": self.num_anomaly_tracks,
                "last_update_time": self.last_update_time,
            }

camera_stats = CameraStats()

# ---------------------------------------------
# MagicBox profile & voice (EchoDaemon)
# ---------------------------------------------

def load_magic_profile():
    if os.path.exists(MAGIC_MEMORY_FILE):
        try:
            with open(MAGIC_MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"name": "Adventurer", "class": "Wanderer", "last_spell": ""}

def save_magic_profile(profile):
    try:
        with open(MAGIC_MEMORY_FILE, "w") as f:
            json.dump(profile, f)
    except Exception:
        pass

class EchoDaemon:
    def __init__(self, engine, profile):
        self.engine = engine
        self.profile = profile
        self.configured = False
        self.lock = threading.Lock()

    def calibrate(self):
        cls = self.profile.get("class", "Wanderer")
        config = {
            "Warrior": {"rate": 160},
            "Mage": {"rate": 120},
            "Rogue": {"rate": 180},
            "Necromancer": {"rate": 100},
            "Summoner": {"rate": 140},
            "Wanderer": {"rate": 140}
        }
        rate = config.get(cls, config["Wanderer"])["rate"]
        self.engine.setProperty("rate", rate)
        self.configured = True

    def speak(self, text):
        with self.lock:
            if not self.configured:
                self.calibrate()
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.warning(f"Voice engine error: {e}")

voice_engine = pyttsx3.init()
magic_profile = load_magic_profile()
echo_daemon = EchoDaemon(voice_engine, magic_profile)

# ---------------------------------------------
# Mouse stats & drill stats
# ---------------------------------------------

class StatsTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.max_angles_kept = 1000
        self.reset()

    def reset(self):
        with self.lock:
            self.start_time = time.time()
            self.total_clicks = 0
            self.game_clicks = 0
            self.click_times = []
            self.last_pos = None
            self.angle_samples = []

    def register_click(self, in_game: bool):
        now = time.time()
        with self.lock:
            self.total_clicks += 1
            if in_game:
                self.game_clicks += 1
            self.click_times.append(now)

    def register_move(self, x: int, y: int):
        with self.lock:
            if self.last_pos is None:
                self.last_pos = (x, y)
                return
            x0, y0 = self.last_pos
            dx = x - x0
            dy = y - y0
            self.last_pos = (x, y)

            dist_sq = dx * dx + dy * dy
            if dist_sq < 2:
                return

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            self.angle_samples.append(angle_deg)
            if len(self.angle_samples) > self.max_angles_kept:
                self.angle_samples = self.angle_samples[-self.max_angles_kept:]

    def get_core_stats(self):
        with self.lock:
            now = time.time()
            elapsed = max(now - self.start_time, 1e-6)

            total_cpm = self.total_clicks / elapsed * 60.0
            game_cpm = self.game_clicks / elapsed * 60.0

            if len(self.click_times) >= 3:
                intervals = [
                    self.click_times[i + 1] - self.click_times[i]
                    for i in range(len(self.click_times) - 1)
                ]
                avg = sum(intervals) / len(intervals)
                var = sum((x - avg) ** 2 for x in intervals) / len(intervals)
                steadiness = max(0.0, 100.0 - min(var * 200.0, 100.0))
            else:
                steadiness = 0.0

            return {
                "elapsed": elapsed,
                "total_clicks": self.total_clicks,
                "game_clicks": self.game_clicks,
                "total_cpm": total_cpm,
                "game_cpm": game_cpm,
                "steadiness": steadiness,
            }

    def get_angle_info(self):
        with self.lock:
            if len(self.angle_samples) < 5:
                return {
                    "angle_stability": 0.0,
                    "drift_bias": 0.0,
                }

            angles = self.angle_samples
            mean_angle = sum(angles) / len(angles)
            var = sum((a - mean_angle) ** 2 for a in angles) / len(angles)
            stability = max(0.0, 100.0 - min(var * 0.5, 100.0))

            return {
                "angle_stability": stability,
                "drift_bias": mean_angle,
            }

    def to_dict(self):
        with self.lock:
            return {
                "start_time": self.start_time,
                "total_clicks": self.total_clicks,
                "game_clicks": self.game_clicks,
                "click_times": self.click_times,
                "angle_samples": self.angle_samples,
            }

    def from_dict(self, data):
        with self.lock:
            self.start_time = data.get("start_time", time.time())
            self.total_clicks = data.get("total_clicks", 0)
            self.game_clicks = data.get("game_clicks", 0)
            self.click_times = data.get("click_times", [])
            self.angle_samples = data.get("angle_samples", [])
            if len(self.angle_samples) > self.max_angles_kept:
                self.angle_samples = self.angle_samples[-self.max_angles_kept:]
            self.last_pos = None

class DrillStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.total_targets = 0
            self.hits = 0
            self.misses = 0
            self.reaction_times = []

    def register_target_spawn(self):
        with self.lock:
            self.total_targets += 1

    def register_hit(self, reaction_time: float):
        with self.lock:
            self.hits += 1
            self.reaction_times.append(reaction_time)

    def register_miss(self):
        with self.lock:
            self.misses += 1

    def get_stats(self):
        with self.lock:
            total = self.total_targets
            hits = self.hits
            misses = self.misses
            if total > 0:
                accuracy = hits / total * 100.0
            else:
                accuracy = 0.0
            if self.reaction_times:
                avg_react = sum(self.reaction_times) / len(self.reaction_times)
            else:
                avg_react = 0.0
            return {
                "total_targets": total,
                "hits": hits,
                "misses": misses,
                "accuracy": accuracy,
                "avg_reaction": avg_react,
            }

    def to_dict(self):
        with self.lock:
            return {
                "total_targets": self.total_targets,
                "hits": self.hits,
                "misses": self.misses,
                "reaction_times": self.reaction_times,
            }

    def from_dict(self, data):
        with self.lock:
            self.total_targets = data.get("total_targets", 0)
            self.hits = data.get("hits", 0)
            self.misses = data.get("misses", 0)
            self.reaction_times = data.get("reaction_times", [])

# ---------------------------------------------
# AI Advisor
# ---------------------------------------------

class AIAdvisor:
    def __init__(self):
        self.lock = threading.Lock()
        self.smoothing_level = 5.0
        self.recommended_sens = 1.0
        self.recommended_dpi_min = 800
        self.recommended_dpi_max = 1600
        self.last_update_time = 0.0
        self.last_voice_time = 0.0

    def update(self, core_stats, angle_info, drill_stats, camera_stats_snapshot):
        now = time.time()
        with self.lock:
            if now - self.last_update_time < 2.0:
                return
            self.last_update_time = now

            steadiness = core_stats.get("steadiness", 0.0)
            angle_stability = angle_info.get("angle_stability", 0.0)
            drift_bias = angle_info.get("drift_bias", 0.0)
            drill_accuracy = drill_stats.get("accuracy", 0.0)
            drill_react = drill_stats.get("avg_reaction", 0.0)

            cam_avg_speed = camera_stats_snapshot.get("avg_track_speed", 0.0)
            cam_anomaly_tracks = camera_stats_snapshot.get("num_anomaly_tracks", 0)
            cam_global_anomaly = camera_stats_snapshot.get("global_anomaly", False)
            cam_global_z = camera_stats_snapshot.get("global_motion_z", 0.0)

            combined_stability = (steadiness * 0.4) + (angle_stability * 0.6)

            cam_penalty = 0.0
            cam_penalty += min(cam_avg_speed * 5.0, 20.0)
            cam_penalty += min(cam_anomaly_tracks * 4.0, 30.0)
            if cam_global_anomaly and abs(cam_global_z) > 2.0:
                cam_penalty += 10.0

            combined_stability = max(0.0, combined_stability - cam_penalty * 0.4)

            if combined_stability > 70.0:
                target_smoothing = 3.0
            elif combined_stability > 40.0:
                target_smoothing = 8.0
            else:
                target_smoothing = 15.0

            react_factor = 0.7
            self.smoothing_level = (
                self.smoothing_level * (1.0 - react_factor)
                + target_smoothing * react_factor
            )
            self.smoothing_level = max(0.0, min(self.smoothing_level, 30.0))

            if combined_stability > 80.0:
                target_sens = 1.8
            elif combined_stability > 50.0:
                target_sens = 1.4
            else:
                target_sens = 1.0

            if drill_accuracy > 80.0 and drill_react < 0.300:
                target_sens += 0.2
            if drill_accuracy < 40.0 and drill_react > 0.500:
                target_sens -= 0.2

            if cam_global_anomaly or cam_anomaly_tracks >= 3:
                target_sens -= 0.1
            if cam_avg_speed > 2.5:
                target_sens -= 0.1

            target_sens = max(0.7, min(target_sens, 2.0))

            sens_react = 0.5
            self.recommended_sens = (
                self.recommended_sens * (1.0 - sens_react)
                + target_sens * sens_react
            )

            if self.recommended_sens >= 1.6:
                dpi_min, dpi_max = 1200, 2000
            elif self.recommended_sens >= 1.2:
                dpi_min, dpi_max = 1000, 1800
            else:
                dpi_min, dpi_max = 800, 1400

            if abs(drift_bias) > 60.0:
                dpi_min -= 100
                dpi_max -= 100

            if cam_global_anomaly:
                dpi_max -= 100

            self.recommended_dpi_min = max(400, min(dpi_min, 3200))
            self.recommended_dpi_max = max(600, min(dpi_max, 3600))

            if now - self.last_voice_time > 15.0:
                self.last_voice_time = now
                desc = "steady" if combined_stability > 60 else "shaky"
                echo_daemon.speak(
                    f"Sensitivity coach update. Your aim rhythm is {desc}. "
                    f"Suggested scale {self.recommended_sens:.2f} and DPI range "
                    f"{self.recommended_dpi_min} to {self.recommended_dpi_max}."
                )

    def get_recommendations(self):
        with self.lock:
            return {
                "smoothing_level": self.smoothing_level,
                "recommended_sens": self.recommended_sens,
                "recommended_dpi_min": self.recommended_dpi_min,
                "recommended_dpi_max": self.recommended_dpi_max,
            }

    def to_dict(self):
        with self.lock:
            return {
                "smoothing_level": self.smoothing_level,
                "recommended_sens": self.recommended_sens,
                "recommended_dpi_min": self.recommended_dpi_min,
                "recommended_dpi_max": self.recommended_dpi_max,
            }

    def from_dict(self, data):
        with self.lock:
            self.smoothing_level = data.get("smoothing_level", 5.0)
            self.recommended_sens = data.get("recommended_sens", 1.0)
            self.recommended_dpi_min = data.get("recommended_dpi_min", 800)
            self.recommended_dpi_max = data.get("recommended_dpi_max", 1600)

# ---------------------------------------------
# Player profile & persistence
# ---------------------------------------------

class PlayerProfile:
    def __init__(self, name: str):
        self.name = name
        self.mouse_stats = StatsTracker()
        self.drill_stats = DrillStats()
        self.ai = AIAdvisor()

    def reset_all(self):
        self.mouse_stats.reset()
        self.drill_stats.reset()

    def to_dict(self):
        return {
            "name": self.name,
            "mouse_stats": self.mouse_stats.to_dict(),
            "drill_stats": self.drill_stats.to_dict(),
            "ai": self.ai.to_dict(),
        }

    def from_dict(self, data):
        self.name = data.get("name", self.name)
        ms = data.get("mouse_stats", {})
        ds = data.get("drill_stats", {})
        ai = data.get("ai", {})
        self.mouse_stats.from_dict(ms)
        self.drill_stats.from_dict(ds)
        self.ai.from_dict(ai)

player_profiles = {
    "Player 1": PlayerProfile("Player 1"),
    "Player 2": PlayerProfile("Player 2"),
    "Player 3": PlayerProfile("Player 3"),
    "Player 4": PlayerProfile("Player 4"),
}
current_player_name = "Player 1"

def get_current_profile() -> PlayerProfile:
    return player_profiles[current_player_name]

def save_profiles_to_disk():
    data = {
        "current_player_name": current_player_name,
        "profiles": {},
    }
    for key, profile in player_profiles.items():
        data["profiles"][key] = profile.to_dict()
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save profiles: {e}")

def load_profiles_from_disk():
    global current_player_name, player_profiles
    if not os.path.exists(PROFILE_FILE):
        return
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load profiles: {e}")
        return

    profiles_data = data.get("profiles", {})
    new_profiles = {}

    for key, pdata in profiles_data.items():
        name = pdata.get("name", key)
        profile = PlayerProfile(name)
        profile.from_dict(pdata)
        new_profiles[key] = profile

    for default_key in ["Player 1", "Player 2", "Player 3", "Player 4"]:
        if default_key not in new_profiles:
            new_profiles[default_key] = PlayerProfile(default_key)

    player_profiles.clear()
    player_profiles.update(new_profiles)

    cp = data.get("current_player_name", "Player 1")
    if cp in player_profiles:
        current_player_name = cp
    else:
        current_player_name = "Player 1"

# ---------------------------------------------
# Manual SMB save
# ---------------------------------------------

def manual_save_to_network_drive():
    filepath = filedialog.asksaveasfilename(
        title="Save Profiles to Network Drive",
        defaultextension=".json",
        filetypes=[("JSON Files", "*.json")],
        initialfile="b4b_profiles.json"
    )

    if not filepath:
        return

    data = {
        "current_player_name": current_player_name,
        "profiles": {},
    }
    for key, profile in player_profiles.items():
        data["profiles"][key] = profile.to_dict()

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Profiles saved to: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save to network drive: {e}")

# ---------------------------------------------
# Mouse monitor
# ---------------------------------------------

class MouseMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.listener = None
        self.running = True

    def on_move(self, x, y):
        profile = get_current_profile()
        profile.mouse_stats.register_move(x, y)
        return True

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            in_game = is_game_foreground()
            profile = get_current_profile()
            profile.mouse_stats.register_click(in_game)
        return True

    def run(self):
        with mouse.Listener(on_move=self.on_move, on_click=self.on_click) as listener:
            self.listener = listener
            listener.join()

    def stop(self):
        self.running = False
        if self.listener is not None:
            self.listener.stop()

# ---------------------------------------------
# Drill window
# ---------------------------------------------

class DrillWindow:
    def __init__(self, parent, get_profile_func):
        self.parent = parent
        self.get_profile = get_profile_func

        self.window = tk.Toplevel(parent)
        self.window.title("Accuracy Drill")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.running = False
        self.target_id = None
        self.target_center = (0, 0)
        self.target_radius = 30
        self.target_spawn_time = None
        self.target_timeout_ms = 2000
        self.spawn_interval_ms = 1100

        self.big_font = ("Segoe UI", 14, "bold")
        self.med_font = ("Segoe UI", 12)

        top_frame = ttk.Frame(self.window, padding=5)
        top_frame.pack(fill="x")

        self.start_button = ttk.Button(
            top_frame,
            text="Start Drill",
            command=self.start_drill
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(
            top_frame,
            text="Stop Drill",
            command=self.stop_drill
        )
        self.stop_button.pack(side="left", padx=5)

        self.reset_button = ttk.Button(
            top_frame,
            text="Reset Drill Stats",
            command=self.reset_stats
        )
        self.reset_button.pack(side="left", padx=5)

        self.canvas = tk.Canvas(self.window, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        stats_frame = ttk.Frame(self.window, padding=5)
        stats_frame.pack(fill="x")

        self.targets_label = ttk.Label(stats_frame, text="Targets: 0", font=self.med_font)
        self.targets_label.pack(side="left", padx=5)

        self.hits_label = ttk.Label(stats_frame, text="Hits: 0", font=self.med_font)
        self.hits_label.pack(side="left", padx=5)

        self.misses_label = ttk.Label(stats_frame, text="Misses: 0", font=self.med_font)
        self.misses_label.pack(side="left", padx=5)

        self.accuracy_label = ttk.Label(stats_frame, text="Accuracy: 0.0 %", font=self.med_font)
        self.accuracy_label.pack(side="left", padx=5)

        self.reaction_label = ttk.Label(stats_frame, text="Avg Reaction: 0.000 s", font=self.med_font)
        self.reaction_label.pack(side="left", padx=5)

    def start_drill(self):
        if self.running:
            return
        self.running = True
        self.schedule_next_target()

    def stop_drill(self):
        self.running = False
        self.clear_target()

    def reset_stats(self):
        profile = self.get_profile()
        profile.drill_stats.reset()
        self.update_stats_gui()

    def close(self):
        self.running = False
        self.window.destroy()

    def clear_target(self):
        if self.target_id is not None:
            self.canvas.delete(self.target_id)
            self.target_id = None
        self.target_center = (0, 0)
        self.target_spawn_time = None

    def schedule_next_target(self):
        if not self.running:
            return
        self.spawn_target()
        self.window.after(self.spawn_interval_ms, self.schedule_next_target)

    def spawn_target(self):
        self.clear_target()

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 0 or h <= 0:
            w, h = 800, 600

        margin = 40
        x = random.randint(margin, w - margin)
        y = random.randint(margin, h - margin)
        r = self.target_radius

        self.target_center = (x, y)
        self.target_spawn_time = time.time()

        self.target_id = self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="yellow", width=3, fill="red"
        )

        profile = self.get_profile()
        profile.drill_stats.register_target_spawn()
        self.update_stats_gui()

        def timeout_check():
            if not self.running:
                return
            if self.target_id is not None and self.target_spawn_time is not None:
                now = time.time()
                if (now - self.target_spawn_time) * 1000.0 >= self.target_timeout_ms:
                    profile.drill_stats.register_miss()
                    self.clear_target()
                    self.update_stats_gui()

        self.window.after(self.target_timeout_ms + 50, timeout_check)

    def on_click(self, event):
        if self.target_id is None or self.target_spawn_time is None:
            return

        x, y = event.x, event.y
        tx, ty = self.target_center
        dx = x - tx
        dy = y - ty
        dist = math.hypot(dx, dy)

        profile = self.get_profile()
        if dist <= self.target_radius:
            reaction = time.time() - self.target_spawn_time
            profile.drill_stats.register_hit(reaction)
            echo_daemon.speak(f"Hit! Reaction time {reaction:.2f} seconds.")
            self.clear_target()
        else:
            profile.drill_stats.register_miss()
            echo_daemon.speak("Missed target.")

        self.update_stats_gui()

    def update_stats_gui(self):
        profile = self.get_profile()
        s = profile.drill_stats.get_stats()

        self.targets_label.config(text=f"Targets: {s['total_targets']}")
        self.hits_label.config(text=f"Hits: {s['hits']}")
        self.misses_label.config(text=f"Misses: {s['misses']}")
        self.accuracy_label.config(text=f"Accuracy: {s['accuracy']:.1f} %")
        self.reaction_label.config(text=f"Avg Reaction: {s['avg_reaction']:.3f} s")

# ---------------------------------------------
# Camera Lab Thread (OpenCV + pygame)
# ---------------------------------------------

class CameraLabThread(threading.Thread):
    def __init__(self, camera_stats: CameraStats):
        super().__init__(daemon=True)
        self.camera_stats = camera_stats
        self.running = False

        self.WIDTH, self.HEIGHT = 960, 540
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 215, 0)
        self.ORANGE = (255, 140, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)

        self.CAP_W, self.CAP_H = 640, 360

        self.VEL_Z_THRESH = 2.2
        self.HEADING_DELTA_THRESH = math.radians(75)
        self.AREA_JUMP_RATIO = 1.8
        self.LINGER_FRAMES = 90
        self.LINGER_RADIUS = 35
        self.GLOBAL_MOTION_Z = 2.5

    def open_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAP_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def motion_params_from_sensitivity(self, s):
        varThresh = int(np.interp(s, [0, 1], [52, 14]))
        min_area = int(np.interp(s, [0, 1], [1400, 240]))
        return varThresh, min_area

    def tracking_params_from_stability(self, stab):
        confirm_N = int(np.interp(stab, [0, 1], [1, 4]))
        max_age = int(np.interp(stab, [0, 1], [10, 6]))
        gate_mahal = np.interp(stab, [0, 1], [7.5, 4.2])
        Q_scale = np.interp(stab, [0, 1], [1.5, 0.7])
        R_scale = np.interp(stab, [0, 1], [1.3, 0.85])
        return confirm_N, max_age, gate_mahal, Q_scale, R_scale

    def detect_faces(self, gray, face_cascade):
        faces = face_cascade.detectMultiScale(gray, 1.08, 4, minSize=(24, 24))
        return [("face", (x, y, w, h)) for (x, y, w, h) in faces]

    def detect_motion(self, frame, sensitivity, backSub, kernel_size=3):
        varThresh, min_area = self.motion_params_from_sensitivity(sensitivity)
        backSub.setVarThreshold(varThresh)
        fgMask = backSub.apply(frame)
        _, fgMaskBin = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fgMaskClean = cv2.morphologyEx(fgMaskBin, cv2.MORPH_OPEN, kernel, iterations=1)
        fgMaskClean = cv2.morphologyEx(fgMaskClean, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(fgMaskClean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        border = 6
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if x < border or y < border or (x + w) > (self.CAP_W - border) or (y + h) > (self.CAP_H - border):
                continue
            detections.append(("motion", (x, y, w, h)))
        return detections, fgMaskClean

    class KalmanTrack:
        def __init__(self, tid, cx, cy, Q_scale, R_scale, label=None, bbox=None,
                     WIDTH=960, HEIGHT=540, CAP_W=640, CAP_H=360,
                     LINGER_FRAMES=90, LINGER_RADIUS=35,
                     VEL_Z_THRESH=2.2, HEADING_DELTA_THRESH=math.radians(75),
                     AREA_JUMP_RATIO=1.8):
            self.id = tid
            self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32) * 50.0
            self.Q_base = np.diag([1.0, 1.0, 3.0, 3.0]).astype(np.float32) * Q_scale
            self.R_base = np.diag([4.0, 4.0]).astype(np.float32) * R_scale
            self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            self.history = deque(maxlen=64)
            self.label = label
            self.last_bbox = bbox
            self.time_since_update = 0
            self.hits = 0
            self.confirmed = False

            self.vel_history = deque(maxlen=120)
            self.heading_prev = None
            self.area_prev = None
            self.linger_origin = None
            self.linger_start_frame = None
            self.is_anomaly = False
            self.anomaly_reason = ""

            self.WIDTH = WIDTH
            self.HEIGHT = HEIGHT
            self.CAP_W = CAP_W
            self.CAP_H = CAP_H
            self.LINGER_FRAMES = LINGER_FRAMES
            self.LINGER_RADIUS = LINGER_RADIUS
            self.VEL_Z_THRESH = VEL_Z_THRESH
            self.HEADING_DELTA_THRESH = HEADING_DELTA_THRESH
            self.AREA_JUMP_RATIO = AREA_JUMP_RATIO

        def F(self, dt):
            return np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

        def predict(self, dt=1 / 30.0):
            Fm = self.F(dt)
            self.x = Fm @ self.x
            self.P = Fm @ self.P @ Fm.T + self.Q_base
            self.time_since_update += 1
            return float(self.x[0, 0]), float(self.x[1, 0])

        def update(self, z, bbox=None, label=None, frame_idx=None):
            z = np.array(z, dtype=np.float32).reshape(2, 1)
            y = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R_base
            K = self.P @ self.H.T @ np.linalg.inv(S + np.eye(2) * 1e-6)
            self.x = self.x + K @ y
            I = np.eye(4, dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P
            self.time_since_update = 0
            self.hits += 1
            if bbox is not None:
                self.last_bbox = bbox
            if label is not None:
                self.label = label

            vx, vy = self.get_velocity()
            vmag = math.hypot(vx, vy)
            self.vel_history.append(vmag)

            heading = math.atan2(vy, vx) if vmag > 1e-3 else self.heading_prev
            heading_delta = 0.0
            if self.heading_prev is not None and heading is not None:
                diff = heading - self.heading_prev
                while diff > math.pi:
                    diff -= 2 * math.pi
                while diff < -math.pi:
                    diff += 2 * math.pi
                heading_delta = abs(diff)
            self.heading_prev = heading

            area_jump = 1.0
            if self.last_bbox is not None:
                x, y, w, h = self.last_bbox
                area = w * h
                if self.area_prev:
                    area_jump = (area / self.area_prev) if self.area_prev > 0 else 1.0
                self.area_prev = area

            cx, cy = self.get_position()
            cx_disp = int(cx * self.WIDTH / self.CAP_W)
            cy_disp = int(cy * self.HEIGHT / self.CAP_H)
            if self.linger_origin is None:
                self.linger_origin = (cx_disp, cy_disp)
                self.linger_start_frame = frame_idx
            else:
                dist = math.hypot(cx_disp - self.linger_origin[0],
                                   cy_disp - self.linger_origin[1])
                if dist > self.LINGER_RADIUS:
                    self.linger_origin = (cx_disp, cy_disp)
                    self.linger_start_frame = frame_idx

            linger_frames = (frame_idx - self.linger_start_frame) if (
                self.linger_start_frame is not None and frame_idx is not None
            ) else 0

            self.is_anomaly = False
            self.anomaly_reason = ""

            if len(self.vel_history) >= 30:
                vmean = float(np.mean(self.vel_history))
                vstd = float(np.std(self.vel_history)) + 1e-6
                z = (vmag - vmean) / vstd
                if z >= self.VEL_Z_THRESH:
                    self.is_anomaly = True
                    self.anomaly_reason = "velocity spike"
            if heading_delta >= self.HEADING_DELTA_THRESH:
                self.is_anomaly = True
                self.anomaly_reason = "direction change"
            if area_jump >= self.AREA_JUMP_RATIO:
                self.is_anomaly = True
                self.anomaly_reason = "size jump"
            if linger_frames >= self.LINGER_FRAMES:
                self.is_anomaly = True
                self.anomaly_reason = "lingering"

        def get_position(self):
            return float(self.x[0, 0]), float(self.x[1, 0])

        def get_velocity(self):
            return float(self.x[2, 0]), float(self.x[3, 0])

    class MultiTrackerHungarian:
        def __init__(self, max_age=8, confirm_N=3, gate_mahal=6.0,
                     Q_scale=1.0, R_scale=1.0, max_tracks=48,
                     WIDTH=960, HEIGHT=540, CAP_W=640, CAP_H=360,
                     LINGER_FRAMES=90, LINGER_RADIUS=35,
                     VEL_Z_THRESH=2.2, HEADING_DELTA_THRESH=math.radians(75),
                     AREA_JUMP_RATIO=1.8):
            self.tracks = []
            self.next_id = 0
            self.max_age = max_age
            self.confirm_N = confirm_N
            self.gate_mahal = gate_mahal
            self.Q_scale = Q_scale
            self.R_scale = R_scale
            self.max_tracks = max_tracks
            self.dt = 1 / 30.0
            self.frame_idx = 0
            self.WIDTH = WIDTH
            self.HEIGHT = HEIGHT
            self.CAP_W = CAP_W
            self.CAP_H = CAP_H
            self.LINGER_FRAMES = LINGER_FRAMES
            self.LINGER_RADIUS = LINGER_RADIUS
            self.VEL_Z_THRESH = VEL_Z_THRESH
            self.HEADING_DELTA_THRESH = HEADING_DELTA_THRESH
            self.AREA_JUMP_RATIO = AREA_JUMP_RATIO

        def step(self, detections):
            self.frame_idx += 1
            for t in self.tracks:
                t.predict(self.dt)

            det_centers = [((x + w / 2), (y + h / 2)) for _, (x, y, w, h) in detections]
            det_bboxes = [b for _, b in detections]
            det_labels = [label for label, _ in detections]
            T, D = len(self.tracks), len(det_centers)

            if T > 0 and D > 0:
                cost = np.zeros((T, D), dtype=np.float32)
                for i, tr in enumerate(self.tracks):
                    S = tr.H @ tr.P @ tr.H.T + tr.R_base
                    invS = np.linalg.inv(S + np.eye(2) * 1e-6)
                    hx = tr.H @ tr.x
                    for j, dc in enumerate(det_centers):
                        diff = np.array(dc, dtype=np.float32).reshape(2, 1) - hx
                        cost[i, j] = float(diff.T @ invS @ diff)
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_tracks = set()
                matched_dets = set()
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= self.gate_mahal:
                        tr = self.tracks[r]
                        tr.update(
                            det_centers[c],
                            bbox=det_bboxes[c],
                            label=det_labels[c],
                            frame_idx=self.frame_idx
                        )
                        tr.history.append((
                            int(tr.x[0, 0] * self.WIDTH / self.CAP_W),
                            int(tr.x[1, 0] * self.HEIGHT / self.CAP_H)
                        ))
                        matched_tracks.add(r)
                        matched_dets.add(c)
                survivors = []
                for i, tr in enumerate(self.tracks):
                    if i not in matched_tracks:
                        tr.time_since_update += 1
                    tr.confirmed = (tr.hits >= self.confirm_N)
                    if tr.time_since_update <= self.max_age:
                        survivors.append(tr)
                self.tracks = survivors
            else:
                for j in range(D):
                    if len(self.tracks) < self.max_tracks:
                        cx, cy = det_centers[j]
                        t = CameraLabThread.KalmanTrack(
                            self.next_id, cx, cy,
                            Q_scale=self.Q_scale, R_scale=self.R_scale,
                            label=det_labels[j], bbox=det_bboxes[j],
                            WIDTH=self.WIDTH, HEIGHT=self.HEIGHT,
                            CAP_W=self.CAP_W, CAP_H=self.CAP_H,
                            LINGER_FRAMES=self.LINGER_FRAMES,
                            LINGER_RADIUS=self.LINGER_RADIUS,
                            VEL_Z_THRESH=self.VEL_Z_THRESH,
                            HEADING_DELTA_THRESH=self.HEADING_DELTA_THRESH,
                            AREA_JUMP_RATIO=self.AREA_JUMP_RATIO
                        )
                        t.history.append((
                            int(cx * self.WIDTH / self.CAP_W),
                            int(cy * self.HEIGHT / self.CAP_H)
                        ))
                        self.tracks.append(t)
                        self.next_id += 1
                if D == 0:
                    self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self.tracks

    class Reticle:
        def __init__(self, WIDTH, HEIGHT, CAP_W, CAP_H, YELLOW):
            self.WIDTH = WIDTH
            self.HEIGHT = HEIGHT
            self.CAP_W = CAP_W
            self.CAP_H = CAP_H
            self.YELLOW = YELLOW
            self.pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
            self.alpha = 0.22
            self.snap_gain = 0.06

        def update(self, t):
            center = np.array([self.WIDTH // 2, self.HEIGHT // 2], dtype=np.float32)
            if t is None:
                self.pos = (1 - self.alpha) * self.pos + self.alpha * center
                return
            cx, cy = t.get_position()
            vx, vy = t.get_velocity()
            cx_disp = cx * self.WIDTH / self.CAP_W
            cy_disp = cy * self.HEIGHT / self.CAP_H
            tx = cx_disp + vx * (self.WIDTH / self.CAP_W) * 0.035
            ty = cy_disp + vy * (self.HEIGHT / self.CAP_H) * 0.035
            target = np.array([tx, ty], dtype=np.float32)
            self.pos = self.pos + self.snap_gain * (target - self.pos)
            self.pos = (1 - self.alpha) * self.pos + self.alpha * target

        def draw(self, surf):
            x, y = int(self.pos[0]), int(self.pos[1])
            pygame.draw.line(surf, self.YELLOW, (x - 12, y), (x + 12, y), 2)
            pygame.draw.line(surf, self.YELLOW, (x, y - 12), (x, y + 12), 2)
            pygame.draw.circle(surf, self.YELLOW, (x, y), 16, 1)

    class Slider:
        def __init__(self, x, y, w, label, init, font, WHITE, BLUE, CYAN):
            self.rect = pygame.Rect(x, y, w, 10)
            self.handle_x = x + int(init * w)
            self.label = label
            self.value = init
            self.dragging = False
            self.font = font
            self.WHITE = WHITE
            self.BLUE = BLUE
            self.CYAN = CYAN

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos) or abs(event.pos[0] - self.handle_x) < 10:
                    self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                x = np.clip(event.pos[0], self.rect.x, self.rect.x + self.rect.width)
                self.handle_x = int(x)
                self.value = (self.handle_x - self.rect.x) / self.rect.width

        def draw(self, surf):
            pygame.draw.rect(surf, self.WHITE, self.rect, 1)
            pygame.draw.rect(
                surf, self.BLUE,
                (self.rect.x, self.rect.y + 4, self.handle_x - self.rect.x, 2)
            )
            pygame.draw.circle(surf, self.CYAN, (self.handle_x, self.rect.y + 5), 6)
            text = self.font.render(f"{self.label}: {self.value:.2f}", True, self.WHITE)
            surf.blit(text, (self.rect.x, self.rect.y - 18))

    def target_score(self, track, center):
        face_bonus = 20.0 if (track.label == "face") else 0.0
        confirmed_bonus = 10.0 if track.confirmed else 0.0
        area = 0.0
        if track.last_bbox is not None:
            x, y, w, h = track.last_bbox
            area = w * h
        cx, cy = track.get_position()
        cx_disp = int(cx * self.WIDTH / self.CAP_W)
        cy_disp = int(cy * self.HEIGHT / self.CAP_H)
        dist_center = math.hypot(cx_disp - center[0], cy_disp - center[1])
        recent_penalty = track.time_since_update * 6.0
        age_bonus = min(track.hits * 2.0, 30.0)
        vx, vy = track.get_velocity()
        motion_mag = math.hypot(vx, vy)
        return (
            (area * 0.05)
            + face_bonus
            + confirmed_bonus
            + age_bonus
            + (motion_mag * 8.0)
            - (dist_center * 0.25)
            - recent_penalty
        )

    def pick_target(self, tracks, mouse_pos, locked_ids, current_lock_index):
        if locked_ids:
            for tid in locked_ids:
                t = next((tr for tr in tracks if tr.id == tid), None)
                if t:
                    vx, vy = t.get_velocity()
                    cx, cy = t.get_position()
                    cx_disp = int(cx * self.WIDTH / self.CAP_W)
                    cy_disp = int(cy * self.HEIGHT / self.CAP_H)
                    dx = (self.WIDTH // 2) - cx_disp
                    dy = (self.HEIGHT // 2) - cy_disp
                    if vx * dx + vy * dy > 0:
                        return t
            tid = locked_ids[current_lock_index % len(locked_ids)]
            return next((tr for tr in tracks if tr.id == tid), None)

        mx, my = mouse_pos
        best = None
        bests = -1e9
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        for t in tracks:
            cx, cy = t.get_position()
            cx_disp = int(cx * self.WIDTH / self.CAP_W)
            cy_disp = int(cy * self.HEIGHT / self.CAP_H)
            dist_mouse = math.hypot(mx - cx_disp, my - cy_disp)
            s = self.target_score(t, center) - dist_mouse * 0.25
            if s > bests:
                bests = s
                best = t
        return best

    def run(self):
        self.running = True

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Camera Lab — Auto X-Ray + Anomaly Detection")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 22)
        display_surface = pygame.Surface((self.WIDTH, self.HEIGHT))

        cap = self.open_camera()
        if not cap.isOpened():
            logging.error("Failed to open camera.")
            pygame.quit()
            return

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=600, varThreshold=26, detectShadows=True
        )

        sensitivity = 0.6
        stability = 0.6
        FACE_EVERY_N = 4
        MAX_TRACKS = 48

        confirm_N, max_age, gate_mahal, Q_scale, R_scale = self.tracking_params_from_stability(stability)
        tracker = self.MultiTrackerHungarian(
            max_age=max_age, confirm_N=confirm_N, gate_mahal=gate_mahal,
            Q_scale=Q_scale, R_scale=R_scale, max_tracks=MAX_TRACKS,
            WIDTH=self.WIDTH, HEIGHT=self.HEIGHT,
            CAP_W=self.CAP_W, CAP_H=self.CAP_H,
            LINGER_FRAMES=self.LINGER_FRAMES,
            LINGER_RADIUS=self.LINGER_RADIUS,
            VEL_Z_THRESH=self.VEL_Z_THRESH,
            HEADING_DELTA_THRESH=self.HEADING_DELTA_THRESH,
            AREA_JUMP_RATIO=self.AREA_JUMP_RATIO
        )

        reticle = self.Reticle(self.WIDTH, self.HEIGHT, self.CAP_W, self.CAP_H, self.YELLOW)
        locked_ids = []
        current_lock_index = 0
        mouse_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        slider_sens = self.Slider(
            10, self.HEIGHT - 40, 240, "Sensitivity",
            init=sensitivity, font=font,
            WHITE=self.WHITE, BLUE=self.BLUE, CYAN=self.CYAN
        )
        slider_stab = self.Slider(
            270, self.HEIGHT - 40, 240, "Stability",
            init=stability, font=font,
            WHITE=self.WHITE, BLUE=self.BLUE, CYAN=self.CYAN
        )
        frame_count = 0
        last_reopen_time = 0
        frames_no_dets = 0
        camera_hiccup = False
        global_motion_hist = deque(maxlen=180)
        last_voice_anom = 0.0

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEMOTION:
                        mouse_pos = event.pos
                    slider_sens.handle_event(event)
                    slider_stab.handle_event(event)

                sensitivity = slider_sens.value
                stability = slider_stab.value
                confirm_N, max_age, gate_mahal, Q_scale, R_scale = self.tracking_params_from_stability(stability)
                tracker.confirm_N = confirm_N
                tracker.max_age = max_age
                tracker.gate_mahal = gate_mahal
                tracker.Q_scale = Q_scale
                tracker.R_scale = R_scale

                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    camera_hiccup = True
                    logging.warning("Camera feed failed — possible conflict with another application.")
                    if time.time() - last_reopen_time > 1.0:
                        cap.release()
                        time.sleep(0.2)
                        cap = self.open_camera()
                        last_reopen_time = time.time()
                    clock.tick(30)
                    continue
                else:
                    camera_hiccup = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections_motion, motion_mask = self.detect_motion(frame, sensitivity, backSub)
                detections_faces = self.detect_faces(gray, face_cascade) if (frame_count % FACE_EVERY_N == 0) else []
                frame_count += 1
                detections = detections_faces + detections_motion

                tracks = tracker.step(detections)
                target_track = self.pick_target(tracks, mouse_pos, locked_ids, current_lock_index)
                reticle.update(target_track)

                motion_heat = float(np.count_nonzero(motion_mask)) / float(motion_mask.size)
                global_motion_hist.append(motion_heat)
                global_anomaly = False
                g_z = 0.0
                if len(global_motion_hist) >= 60:
                    g_mean = float(np.mean(global_motion_hist))
                    g_std = float(np.std(global_motion_hist)) + 1e-6
                    g_z = (motion_heat - g_mean) / g_std
                    global_anomaly = abs(g_z) >= self.GLOBAL_MOTION_Z

                fps = clock.get_fps()
                frames_no_dets = frames_no_dets + 1 if len(detections) == 0 else 0
                auto_xray = (camera_hiccup or fps < 18 or frames_no_dets >= 60 or global_anomaly)

                if auto_xray:
                    edges = cv2.Canny(gray, 100, 200)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    edges_resized = cv2.resize(edges_rgb, (self.WIDTH, self.HEIGHT))
                    motion_color = cv2.applyColorMap(
                        cv2.resize(motion_mask, (self.WIDTH, self.HEIGHT)),
                        cv2.COLORMAP_HOT
                    )
                    xray = cv2.addWeighted(edges_resized, 0.6, motion_color, 0.6, 0.0)
                    arr = np.transpose(xray, (1, 0, 2)).copy()
                    pygame.surfarray.blit_array(display_surface, arr)
                    screen.blit(display_surface, (0, 0))
                    screen.blit(font.render("X-RAY MODE (Auto)", True, self.CYAN),
                                (self.WIDTH - 200, 10))
                else:
                    vis = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
                    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    arr = np.transpose(rgb, (1, 0, 2)).copy()
                    pygame.surfarray.blit_array(display_surface, arr)
                    screen.blit(display_surface, (0, 0))

                total_speed = 0.0
                anomaly_count = 0
                for t in tracks:
                    cx, cy = t.get_position()
                    cx_disp = int(cx * self.WIDTH / self.CAP_W)
                    cy_disp = int(cy * self.HEIGHT / self.CAP_H)
                    color = self.BLUE if (t.label == "face") else self.ORANGE
                    pygame.draw.circle(screen, color, (cx_disp, cy_disp), 6)
                    lbl = f"{(t.label or 'ID').upper()} {t.id}{' ✓' if t.confirmed else ''}"
                    if t.is_anomaly:
                        pygame.draw.circle(screen, self.MAGENTA, (cx_disp, cy_disp), 16, 2)
                        lbl += f" ! {t.anomaly_reason}"
                        anomaly_count += 1
                    screen.blit(font.render(lbl, True, self.WHITE),
                                (cx_disp + 8, cy_disp - 18))
                    for i in range(1, len(t.history)):
                        pygame.draw.line(screen, color, t.history[i - 1], t.history[i], 2)

                    vx, vy = t.get_velocity()
                    total_speed += math.hypot(vx, vy)

                avg_speed = (total_speed / len(tracks)) if tracks else 0.0

                self.camera_stats.update(
                    global_anomaly=global_anomaly,
                    global_motion_z=g_z,
                    avg_track_speed=avg_speed,
                    num_tracks=len(tracks),
                    num_anomaly_tracks=anomaly_count
                )

                if global_anomaly and time.time() - last_voice_anom > 20.0:
                    last_voice_anom = time.time()
                    echo_daemon.speak("Camera Lab reports chaotic motion in your environment.")

                reticle.draw(screen)
                fps = clock.get_fps()
                status = f"FPS: {fps:.1f} | Tracks: {len(tracks)} | Dets: {len(detections)}"
                screen.blit(font.render(status, True, self.WHITE), (10, 10))
                if camera_hiccup:
                    screen.blit(font.render("CAMERA LOST — Reopening...", True, self.RED),
                                (10, 32))
                if global_anomaly:
                    screen.blit(font.render("GLOBAL ANOMALY DETECTED", True, self.MAGENTA),
                                (10, 54))

                slider_sens.draw(screen)
                slider_stab.draw(screen)

                pygame.display.flip()
                clock.tick(30)

        except Exception as e:
            logging.exception(f"CameraLabThread crashed: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            pygame.quit()
            self.running = False

    def stop(self):
        self.running = False

# ---------------------------------------------
# Enemy Training Sandbox (safe EnemyAI)
# ---------------------------------------------

class DummyEnemy:
    def __init__(self, x, y, max_health=100, weak_point=False):
        self.x = x
        self.y = y
        self.max_health = max_health
        self.health = max_health
        self.weak_point = weak_point
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)

    def update(self, width, height):
        self.x += self.vx
        self.y += self.vy
        if self.x < 20 or self.x > width - 20:
            self.vx *= -1
        if self.y < 20 or self.y > height - 20:
            self.vy *= -1

class TrainingGameState:
    def __init__(self):
        self.levels = []
        self.enemies = []

class EnemyAI:
    def __init__(self, game_state):
        self.screen = None
        self.obstacle_density = [1.0, 0.7, 0.5]
        self.current_level = 2
        self.target_locked = True
        self.game_state = game_state

    def adjust_difficulty(self):
        pass

    def enhance_visibility(self):
        for enemy in self.game_state.enemies:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(enemy.x), int(enemy.y)), 15, 2)
            pygame.draw.rect(
                self.screen, (255, 0, 0),
                pygame.Rect(int(enemy.x - 15), int(enemy.y - 15), 30, 30), 1
            )

    def add_target_lock_on(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.game_state.enemies:
            closest_enemy = min(
                self.game_state.enemies,
                key=lambda e: (e.x - mouse_pos[0])**2 + (e.y - mouse_pos[1])**2
            )
            if self.target_locked:
                pygame.draw.line(
                    self.screen, (0, 255, 0),
                    mouse_pos, (int(closest_enemy.x), int(closest_enemy.y)), 2
                )

    def display_health_bars(self):
        for enemy in self.game_state.enemies:
            health_ratio = enemy.health / enemy.max_health
            health_bar_rect = pygame.Rect(
                int(enemy.x - 10), int(enemy.y - 22), int(20 * health_ratio), 5
            )
            pygame.draw.rect(self.screen, (0, 255, 0), health_bar_rect)
            if enemy.weak_point:
                weak_point_rect = pygame.Rect(
                    int(enemy.x - 5), int(enemy.y - 15), 10, 5
                )
                pygame.draw.rect(self.screen, (255, 0, 0), weak_point_rect)

class EnemyTrainingThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False

    def run(self):
        self.running = True
        pygame.init()
        width, height = 800, 600
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Enemy Training Simulator (Sandbox)")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)

        gs = TrainingGameState()
        for _ in range(10):
            gs.enemies.append(
                DummyEnemy(
                    random.randint(50, width - 50),
                    random.randint(50, height - 50),
                    max_health=100,
                    weak_point=random.choice([True, False])
                )
            )
        enemy_ai = EnemyAI(gs)
        enemy_ai.screen = screen

        echo_daemon.speak("Enemy training sandbox started. Practice tracking and awareness.")

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False

                screen.fill((10, 10, 20))

                for enemy in gs.enemies:
                    enemy.update(width, height)

                enemy_ai.enhance_visibility()
                enemy_ai.display_health_bars()
                enemy_ai.add_target_lock_on()

                text = font.render(
                    "Move mouse to practice tracking. ESC to close.",
                    True, (200, 200, 200)
                )
                screen.blit(text, (20, 20))

                pygame.display.flip()
                clock.tick(60)
        finally:
            pygame.quit()
            self.running = False

    def stop(self):
        self.running = False

# ---------------------------------------------
# MagicBox UI (Toplevel)
# ---------------------------------------------

class MagicBoxUI:
    def __init__(self, parent):
        self.profile = magic_profile
        self.echo = echo_daemon

        self.window = tk.Toplevel(parent)
        self.window.title("MagicBox 🧙 – Voice & Spells")
        self.window.geometry("560x330")
        self.window.configure(bg="#1B1B1B")

        self.label = tk.Label(
            self.window,
            text=f"Welcome, {self.profile['name']} the {self.profile['class']}!",
            fg="#F0E6D2", bg="#1B1B1B", font=("Arial", 14)
        )
        self.label.pack(pady=10)

        self.entry = tk.Entry(self.window, width=40, font=("Arial", 13))
        self.entry.pack()
        self.entry.insert(0, self.profile.get("last_spell", ""))

        spells = {
            "Heal": "Restores health over time.",
            "Fireball": "Launches a fiery blast.",
            "Shadowbind": "Stealth and silence combo."
        }

        btn_frame = tk.Frame(self.window, bg="#1B1B1B")
        btn_frame.pack(pady=10)

        for spell, tip in spells.items():
            btn = tk.Button(
                btn_frame, text=spell, width=12,
                command=lambda s=spell: self.cast_spell(s),
                bg="#2E2E2E", fg="#F0E6D2", font=("Arial", 12)
            )
            btn.pack(side="left", padx=5)
            btn.bind("<Enter>", lambda e, t=tip: self.label.config(text=f"💡 {t}"))
            btn.bind(
                "<Leave>",
                lambda e: self.label.config(
                    text=f"Welcome, {self.profile['name']} the {self.profile['class']}!"
                )
            )

        cast_button = tk.Button(
            self.window, text="🪄 Cast Typed Spell", command=self.cast_typed,
            bg="#3A3A3A", fg="#F0E6D2", font=("Arial", 12)
        )
        cast_button.pack(pady=8)

        self.status = tk.Label(
            self.window, text="", fg="#B0FFC0",
            bg="#1B1B1B", font=("Arial", 11)
        )
        self.status.pack()

        threading.Thread(target=self.monitor_game_log, daemon=True).start()

    def cast_typed(self):
        spell = self.entry.get().strip()
        if spell:
            self.cast_spell(spell)

    def cast_spell(self, spell):
        spell_lc = spell.lower()
        if "heal" in spell_lc:
            response = "Healing spell activated. You feel refreshed."
        elif "fireball" in spell_lc:
            response = "Fireball launched!"
        elif "shadow" in spell_lc:
            response = "Shadowbind engaged. You vanish from sight."
        else:
            response = f"You cast: {spell}"

        self.echo.speak(response)
        self.label.config(text=response)
        self.status.config(text=f"Remembered spell: '{spell}'")

        self.profile["last_spell"] = spell
        save_magic_profile(self.profile)

        self.window.configure(bg="#2A2A2A")
        self.window.after(
            300, lambda: self.window.configure(bg="#1B1B1B")
        )

    def monitor_game_log(self):
        time.sleep(3)
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r") as f:
                    data = f.read().lower()
                    if "game started" in data or "loading world" in data:
                        if winsound:
                            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                        msg = f"Game started! Welcome back, {self.profile['name']}."
                        messagebox.showinfo("MagicBox", msg)
                        self.echo.speak(
                            f"Game started. Welcome, {self.profile['name']} the {self.profile['class']}."
                        )
                        self.status.config(
                            text="Game world detected. Ritual memory is active."
                        )
            except Exception:
                pass

# ---------------------------------------------
# Unified GUI – Tkinter main app
# ---------------------------------------------

class UnifiedGUI:
    def __init__(self, root):
        self.root = root
        root.title("MagicBox AI – Unified Aim & Vision Lab")
        root.geometry("900x780")

        self.big_font = ("Segoe UI", 16, "bold")
        self.med_font = ("Segoe UI", 12)
        self.small_font = ("Segoe UI", 10)

        self.drill_window = None
        self.camera_lab_thread = None
        self.magicbox_ui = None
        self.enemy_training_thread = None

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=5)

        ttk.Label(top_frame, text="Active Player:", font=self.med_font).pack(
            side="left", padx=(0, 5)
        )

        self.player_var = tk.StringVar(value=current_player_name)
        self.player_combo = ttk.Combobox(
            top_frame,
            textvariable=self.player_var,
            values=list(player_profiles.keys()),
            state="readonly",
            font=self.med_font,
            width=15
        )
        self.player_combo.pack(side="left", padx=5)
        self.player_combo.bind("<<ComboboxSelected>>", self.on_player_change)

        self.rename_button = ttk.Button(
            top_frame,
            text="Rename Player",
            command=self.rename_player
        )
        self.rename_button.pack(side="left", padx=5)

        self.reset_player_button = ttk.Button(
            top_frame,
            text="Reset Player Stats",
            command=self.reset_current_player_stats
        )
        self.reset_player_button.pack(side="left", padx=5)

        self.drill_button = ttk.Button(
            top_frame,
            text="Open Accuracy Drill",
            command=self.open_drill
        )
        self.drill_button.pack(side="right", padx=5)

        launcher_frame = ttk.Frame(main_frame)
        launcher_frame.pack(fill="x", pady=5)

        self.camera_lab_button = ttk.Button(
            launcher_frame,
            text="Launch Camera Lab",
            command=self.toggle_camera_lab
        )
        self.camera_lab_button.pack(side="left", padx=5)

        self.magicbox_button = ttk.Button(
            launcher_frame,
            text="Open MagicBox Voice & Spells",
            command=self.open_magicbox
        )
        self.magicbox_button.pack(side="left", padx=5)

        self.enemy_training_button = ttk.Button(
            launcher_frame,
            text="Open Enemy Training Sandbox",
            command=self.toggle_enemy_training
        )
        self.enemy_training_button.pack(side="left", padx=5)

        self.camera_status_label = ttk.Label(
            launcher_frame,
            text="Camera Lab: OFF",
            font=self.med_font
        )
        self.camera_status_label.pack(side="left", padx=10)

        self.game_status_label = ttk.Label(
            main_frame,
            text="Game Status: Checking...",
            font=self.big_font
        )
        self.game_status_label.pack(pady=10)

        stats_frame = ttk.LabelFrame(
            main_frame, text="Mouse & Rhythm Stats (Current Player)", padding=10
        )
        stats_frame.pack(fill="both", expand=True, pady=10)

        self.total_clicks_label = ttk.Label(
            stats_frame, text="Total Left Clicks: 0", font=self.med_font
        )
        self.total_clicks_label.pack(anchor="w", pady=2)

        self.game_clicks_label = ttk.Label(
            stats_frame, text="Clicks While In Game: 0", font=self.med_font
        )
        self.game_clicks_label.pack(anchor="w", pady=2)

        self.total_cpm_label = ttk.Label(
            stats_frame, text="Clicks/Minute (Overall): 0.0", font=self.med_font
        )
        self.total_cpm_label.pack(anchor="w", pady=2)

        self.game_cpm_label = ttk.Label(
            stats_frame, text="Clicks/Minute (In Game): 0.0", font=self.med_font
        )
        self.game_cpm_label.pack(anchor="w", pady=2)

        self.steadiness_label = ttk.Label(
            stats_frame, text="Click Rhythm Steadiness: 0 / 100", font=self.med_font
        )
        self.steadiness_label.pack(anchor="w", pady=2)

        angle_frame = ttk.LabelFrame(
            main_frame, text="Angle & Drift Analysis (Current Player)", padding=10
        )
        angle_frame.pack(fill="x", pady=5)

        self.angle_stability_label = ttk.Label(
            angle_frame, text="Angle Stability: 0 / 100", font=self.med_font
        )
        self.angle_stability_label.pack(anchor="w", pady=2)

        self.drift_bias_label = ttk.Label(
            angle_frame, text="Drift Bias: 0.0°", font=self.med_font
        )
        self.drift_bias_label.pack(anchor="w", pady=2)

        drill_stats_frame = ttk.LabelFrame(
            main_frame, text="Accuracy Drill Snapshot (Current Player)", padding=10
        )
        drill_stats_frame.pack(fill="x", pady=5)

        self.drill_targets_label = ttk.Label(
            drill_stats_frame, text="Targets: 0", font=self.med_font
        )
        self.drill_targets_label.pack(anchor="w")

        self.drill_hits_label = ttk.Label(
            drill_stats_frame, text="Hits: 0", font=self.med_font
        )
        self.drill_hits_label.pack(anchor="w")

        self.drill_accuracy_label = ttk.Label(
            drill_stats_frame, text="Accuracy: 0.0 %", font=self.med_font
        )
        self.drill_accuracy_label.pack(anchor="w")

        self.drill_reaction_label = ttk.Label(
            drill_stats_frame, text="Avg Reaction: 0.000 s", font=self.med_font
        )
        self.drill_reaction_label.pack(anchor="w")

        ai_frame = ttk.LabelFrame(
            main_frame, text="AI Comfort & Sensitivity Recommendations", padding=10
        )
        ai_frame.pack(fill="x", pady=5)

        self.ai_smoothing_label = ttk.Label(
            ai_frame, text="Suggested Smoothing Level: 0.0 (0-30)", font=self.med_font
        )
        self.ai_smoothing_label.pack(anchor="w", pady=2)

        self.ai_sens_label = ttk.Label(
            ai_frame, text="Suggested Sensitivity Scale: 1.00x", font=self.med_font
        )
        self.ai_sens_label.pack(anchor="w", pady=2)

        self.ai_dpi_label = ttk.Label(
            ai_frame, text="Suggested DPI Range: 800 - 1600", font=self.med_font
        )
        self.ai_dpi_label.pack(anchor="w", pady=2)

        camera_stats_frame = ttk.LabelFrame(
            main_frame, text="Camera Lab Motion & Anomaly Snapshot", padding=10
        )
        camera_stats_frame.pack(fill="x", pady=5)

        self.cam_tracks_label = ttk.Label(
            camera_stats_frame, text="Tracks: 0", font=self.med_font
        )
        self.cam_tracks_label.pack(anchor="w")

        self.cam_anomalies_label = ttk.Label(
            camera_stats_frame, text="Anomaly Tracks: 0", font=self.med_font
        )
        self.cam_anomalies_label.pack(anchor="w")

        self.cam_speed_label = ttk.Label(
            camera_stats_frame, text="Avg Track Speed: 0.00", font=self.med_font
        )
        self.cam_speed_label.pack(anchor="w")

        self.cam_global_label = ttk.Label(
            camera_stats_frame, text="Global Motion State: NORMAL", font=self.med_font
        )
        self.cam_global_label.pack(anchor="w")

        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill="x", pady=10)

        self.mode_label = ttk.Label(
            footer_frame,
            text=(
                "Mode: Analysis, Comfort, Coaching, Camera Lab, Sandbox & Voice ONLY "
                "(NO GAME HOOKS, NO CHEATS)"
            ),
            font=self.small_font
        )
        self.mode_label.pack(anchor="w", pady=3)

        self.save_network_button = ttk.Button(
            footer_frame,
            text="Save Profiles to Network Drive",
            command=manual_save_to_network_drive
        )
        self.save_network_button.pack(anchor="e", pady=3)

        self.update_interval_ms = 500
        self.update_gui()

    def on_player_change(self, event=None):
        global current_player_name
        name = self.player_var.get()
        if name in player_profiles:
            current_player_name = name

    def rename_player(self):
        old_name = self.player_var.get()
        if old_name not in player_profiles:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Rename Player")
        dialog.geometry("300x120")
        dialog.grab_set()

        ttk.Label(dialog, text="New player name:", font=self.med_font).pack(pady=5)
        name_var = tk.StringVar(value=old_name)
        entry = ttk.Entry(dialog, textvariable=name_var, font=self.med_font)
        entry.pack(pady=5)
        entry.focus_set()

        def do_rename():
            new_name = name_var.get().strip()
            if not new_name:
                dialog.destroy()
                return
            profile = player_profiles.pop(old_name)
            profile.name = new_name
            player_profiles[new_name] = profile

            self.player_combo["values"] = list(player_profiles.keys())
            self.player_var.set(new_name)

            global current_player_name
            current_player_name = new_name

            dialog.destroy()

        ttk.Button(dialog, text="OK", command=do_rename).pack(pady=5)

    def reset_current_player_stats(self):
        profile = get_current_profile()
        profile.reset_all()

    def open_drill(self):
        if self.drill_window is None or not tk.Toplevel.winfo_exists(self.drill_window.window):
            self.drill_window = DrillWindow(self.root, get_current_profile)
        else:
            self.drill_window.window.lift()

    def toggle_camera_lab(self):
        if self.camera_lab_thread is None or not self.camera_lab_thread.running:
            self.camera_lab_thread = CameraLabThread(camera_stats)
            self.camera_lab_thread.start()
            self.camera_status_label.config(text="Camera Lab: RUNNING")
            echo_daemon.speak("Camera Lab launched.")
        else:
            self.camera_lab_thread.stop()
            self.camera_status_label.config(text="Camera Lab: STOPPING...")
            echo_daemon.speak("Camera Lab stopping.")

    def open_magicbox(self):
        if self.magicbox_ui is None or not tk.Toplevel.winfo_exists(self.magicbox_ui.window):
            self.magicbox_ui = MagicBoxUI(self.root)
        else:
            self.magicbox_ui.window.lift()

    def toggle_enemy_training(self):
        if self.enemy_training_thread is None or not self.enemy_training_thread.running:
            self.enemy_training_thread = EnemyTrainingThread()
            self.enemy_training_thread.start()
        else:
            self.enemy_training_thread.stop()

    def update_gui(self):
        running = is_game_running()
        foreground = is_game_foreground()

        if running and foreground:
            status_text = "Game Status: ONLINE • IN FOCUS"
            status_color = "#008800"
        elif running and not foreground:
            status_text = "Game Status: RUNNING (Not focused)"
            status_color = "#CC8800"
        else:
            status_text = "Game Status: NOT RUNNING"
            status_color = "#880000"

        self.game_status_label.config(text=status_text, foreground=status_color)

        profile = get_current_profile()
        core = profile.mouse_stats.get_core_stats()
        angle_info = profile.mouse_stats.get_angle_info()
        drill_s = profile.drill_stats.get_stats()
        cam_s = camera_stats.snapshot()

        self.total_clicks_label.config(text=f"Total Left Clicks: {core['total_clicks']}")
        self.game_clicks_label.config(text=f"Clicks While In Game: {core['game_clicks']}")
        self.total_cpm_label.config(text=f"Clicks/Minute (Overall): {core['total_cpm']:.1f}")
        self.game_cpm_label.config(text=f"Clicks/Minute (In Game): {core['game_cpm']:.1f}")
        self.steadiness_label.config(
            text=f"Click Rhythm Steadiness: {core['steadiness']:.1f} / 100"
        )

        self.angle_stability_label.config(
            text=f"Angle Stability: {angle_info['angle_stability']:.1f} / 100"
        )
        self.drift_bias_label.config(
            text=f"Drift Bias: {angle_info['drift_bias']:.1f}°"
        )

        self.drill_targets_label.config(text=f"Targets: {drill_s['total_targets']}")
        self.drill_hits_label.config(text=f"Hits: {drill_s['hits']}")
        self.drill_accuracy_label.config(
            text=f"Accuracy: {drill_s['accuracy']:.1f} %"
        )
        self.drill_reaction_label.config(
            text=f"Avg Reaction: {drill_s['avg_reaction']:.3f} s"
        )

        profile.ai.update(core, angle_info, drill_s, cam_s)
        ai_rec = profile.ai.get_recommendations()

        self.ai_smoothing_label.config(
            text=f"Suggested Smoothing Level: {ai_rec['smoothing_level']:.1f} (0-30)"
        )
        self.ai_sens_label.config(
            text=f"Suggested Sensitivity Scale: {ai_rec['recommended_sens']:.2f}x"
        )
        self.ai_dpi_label.config(
            text=(
                f"Suggested DPI Range: "
                f"{ai_rec['recommended_dpi_min']} - {ai_rec['recommended_dpi_max']}"
            )
        )

        self.cam_tracks_label.config(
            text=f"Tracks: {cam_s['num_tracks']}"
        )
        self.cam_anomalies_label.config(
            text=f"Anomaly Tracks: {cam_s['num_anomaly_tracks']}"
        )
        self.cam_speed_label.config(
            text=f"Avg Track Speed: {cam_s['avg_track_speed']:.2f}"
        )
        if cam_s["global_anomaly"]:
            state_text = f"Global Motion State: CHAOTIC (z={cam_s['global_motion_z']:.2f})"
            state_color = "#CC00CC"
        else:
            state_text = f"Global Motion State: NORMAL (z={cam_s['global_motion_z']:.2f})"
            state_color = "#0088CC"
        self.cam_global_label.config(text=state_text, foreground=state_color)

        self.root.after(self.update_interval_ms, self.update_gui)

# ---------------------------------------------
# Main entry
# ---------------------------------------------

def main():
    load_profiles_from_disk()

    mouse_monitor = MouseMonitor()
    mouse_monitor.start()

    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = UnifiedGUI(root)

    def on_close():
        if app.camera_lab_thread is not None:
            app.camera_lab_thread.stop()
        if app.enemy_training_thread is not None:
            app.enemy_training_thread.stop()
        mouse_monitor.stop()
        save_profiles_to_disk()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    echo_daemon.speak(
        f"MagicBox unified lab online. Welcome, {magic_profile['name']} the {magic_profile['class']}."
    )
    root.mainloop()

if __name__ == "__main__":
    main()

