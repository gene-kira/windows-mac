import sys
import subprocess
import importlib
import traceback
import time
import random
import platform
from collections import deque
from datetime import datetime
import os
import shutil
import hashlib
import json

import tkinter as tk
from tkinter import filedialog

# ============================================================
#  GLOBAL CONFIG / CONSTANTS
# ============================================================

LOCAL_LOG_FILE = "borg_node_log.txt"
CONFIG_FILE = "borg_config.json"

MIN_FREE_SPACE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# How often (seconds) to rescan game files for changes
GAME_FILE_SCAN_INTERVAL = 60.0

# Max file size to hash (bytes) to avoid huge slowdowns (e.g. 200 MB)
GAME_FILE_HASH_MAX_BYTES = 200 * 1024 * 1024

# In-memory config (loaded at startup, updated at runtime)
CONFIG = {
    "primary_path": "",
    "backup_path": "",
    "game_folder": "",
}

# ============================================================
#  CONFIG PERSISTENCE
# ============================================================

def load_config():
    global CONFIG
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Only keep expected keys
            CONFIG["primary_path"] = data.get("primary_path", "")
            CONFIG["backup_path"] = data.get("backup_path", "")
            CONFIG["game_folder"] = data.get("game_folder", "")
    except Exception:
        # If anything goes wrong, start fresh; don't crash
        CONFIG = {
            "primary_path": "",
            "backup_path": "",
            "game_folder": "",
        }

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2)
    except Exception:
        # Config persistence failure should not kill the UI
        pass

# ============================================================
#  AUTOLOADER
# ============================================================

AUTOLOADER_LOG = []

def autoload(module_name: str, package_name: str = None):
    if package_name is None:
        package_name = module_name

    try:
        mod = importlib.import_module(module_name)
        AUTOLOADER_LOG.append(f"[AUTOLOADER] {module_name} already available.")
        return mod
    except ImportError:
        AUTOLOADER_LOG.append(f"[AUTOLOADER] {module_name} missing, installing '{package_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            mod = importlib.import_module(module_name)
            AUTOLOADER_LOG.append(f"[AUTOLOADER] Installed and loaded {module_name}.")
            return mod
        except Exception as e:
            AUTOLOADER_LOG.append(f"[AUTOLOADER] FAILED to install {module_name}: {e}")
            AUTOLOADER_LOG.append(traceback.format_exc())
            return None

psutil = autoload("psutil")
win32gui = autoload("win32gui", "pywin32")

# ============================================================
#  ALIEN GLYPH SYSTEM
# ============================================================

ALIEN_MAP = {
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               ["⟐","⟟","⟊","⟒","⟓","⟔","⟕","⟖","⟗","⟘","⟙","⟚","⟛","⟜","⟝","⟞",
                "⟠","⟡","⟢","⟣","⟤","⟥","⟦","⟧","⟨","⟩"])),
    **dict(zip("0123456789",
               ["⟪","⟫","⟬","⟭","⟮","⟯","⟰","⟱","⟲","⟳"])),
    " ": "·"
}

def to_alien(text: str) -> str:
    return "".join(ALIEN_MAP.get(ch.upper(), ch) for ch in text)

# ============================================================
#  LCARS-DARK COLOR PALETTE
# ============================================================

PALETTE = {
    "bg": "#0a0a12",
    "rail": "#1a1a2a",
    "rail_accent": "#3a3a5a",
    "pod_bg": "#11111a",
    "pod_border": "#2a2a3a",
    "text": "#c8c8ff",
    "glyph": "#7f7fff",
    "threat_calm": "#202c20",
    "threat_tense": "#2c2620",
    "threat_critical": "#3c2020",
}

POSTURE_THEMES = {
    "WATCHER": {
        "header_text": "WATCHER NODE · PASSIVE SCAN",
        "glyph_color": "#7f7fff",
    },
    "GUARDIAN": {
        "header_text": "GUARDIAN NODE · ACTIVE DEFENSE",
        "glyph_color": "#ffbf7f",
    },
    "CHAMELEON": {
        "header_text": "CHAMELEON NODE · LOW PROFILE",
        "glyph_color": "#7fff7f",
    },
}

# ============================================================
#  GAME / SITUATIONAL AWARENESS CONFIG
# ============================================================

KNOWN_GAMES = {
    "back4blood.exe": "BACK 4 BLOOD",
    "back4blood": "BACK 4 BLOOD",
    "steam.exe": "STEAM",
    "steam": "STEAM",
}

GAME_PROFILES = {
    "BACK 4 BLOOD": {
        "label": "BACK 4 BLOOD PROFILE",
        "description": "High-intensity co-op FPS; sustained CPU and GPU load expected.",
        "cpu_high_delta": 20,
        "mem_high_delta": 15,
        "cpu_idle_threshold": 10,
        "mem_critical": 85,
    },
    "GENERIC_GAME": {
        "label": "GENERIC GAME PROFILE",
        "description": "Unidentified game process; moderate sensitivity.",
        "cpu_high_delta": 25,
        "mem_high_delta": 20,
        "cpu_idle_threshold": 5,
        "mem_critical": 85,
    },
    "DESKTOP": {
        "label": "DESKTOP PROFILE",
        "description": "No primary game detected; spikes are suspicious.",
        "cpu_high_delta": 30,
        "mem_high_delta": 25,
        "cpu_idle_threshold": 2,
        "mem_critical": 90,
    },
}

def get_profile_for_game(game_name: str):
    if "BACK 4 BLOOD" in game_name:
        return GAME_PROFILES["BACK 4 BLOOD"]
    elif game_name != "None":
        return GAME_PROFILES["GENERIC_GAME"]
    else:
        return GAME_PROFILES["DESKTOP"]

# ============================================================
#  NETWORK DRIVE MONITOR
# ============================================================

class NetworkDriveStatus:
    def __init__(self, label: str, path: str):
        self.label = label
        self.path = path
        self.online = False
        self.free_bytes = None
        self.total_bytes = None
        self.last_error = None

    def probe(self):
        self.last_error = None
        self.online = False
        self.free_bytes = None
        self.total_bytes = None

        try:
            root = self.path
            if not root:
                self.last_error = "No path configured"
                return
            if not os.path.exists(root):
                self.last_error = "Path not reachable"
                return
            usage = shutil.disk_usage(root)
            self.free_bytes = usage.free
            self.total_bytes = usage.total
            self.online = True
        except Exception as e:
            self.last_error = str(e)
            self.online = False

    def human_free(self):
        if self.free_bytes is None:
            return "N/A"
        gb = self.free_bytes / (1024 * 1024 * 1024)
        return f"{gb:.2f} GB"

    def human_total(self):
        if self.total_bytes is None:
            return "N/A"
        gb = self.total_bytes / (1024 * 1024 * 1024)
        return f"{gb:.2f} GB"

# ============================================================
#  GAME FILE CACHE MONITOR
# ============================================================

class GameFileCacheMonitor:
    def __init__(self):
        self.path = ""
        self.enabled = False
        self.baseline = {}
        self.last_scan_time = 0.0
        self.first_scan_done = False
        self.last_status = "No folder selected."
        self.recent_events = deque(maxlen=30)

    def set_path(self, path: str):
        self.path = path
        self.enabled = bool(path and os.path.isdir(path))
        self.baseline = {}
        self.last_scan_time = 0.0
        self.first_scan_done = False
        if self.enabled:
            self.last_status = f"Path set: {path}. Baseline pending."
        else:
            self.last_status = "Invalid folder selected."

    def _hash_file(self, full_path):
        try:
            size = os.path.getsize(full_path)
            if size > GAME_FILE_HASH_MAX_BYTES:
                return None
            h = hashlib.sha1()
            with open(full_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _scan_once(self):
        if not self.enabled:
            self.last_status = "No folder selected."
            return []

        events = []
        current_index = {}

        try:
            for root, dirs, files in os.walk(self.path):
                for name in files:
                    full = os.path.join(root, name)
                    try:
                        st = os.stat(full)
                        size = st.st_size
                        mtime = st.st_mtime
                        rel = os.path.relpath(full, self.path)
                        file_hash = self._hash_file(full)
                        current_index[rel] = (size, mtime, file_hash)
                    except Exception:
                        continue
        except Exception as e:
            self.last_status = f"Scan failed: {e}"
            return []

        if not self.first_scan_done:
            self.baseline = current_index
            self.first_scan_done = True
            self.last_status = f"Baseline indexed: {len(self.baseline)} files."
            return []

        baseline_keys = set(self.baseline.keys())
        current_keys = set(current_index.keys())

        added = current_keys - baseline_keys
        missing = baseline_keys - current_keys
        common = baseline_keys & current_keys

        for rel in added:
            events.append(f"FILE_ADDED: {rel}")
        for rel in missing:
            events.append(f"FILE_MISSING: {rel}")
        for rel in common:
            old_size, old_mtime, old_hash = self.baseline[rel]
            new_size, new_mtime, new_hash = current_index[rel]
            if old_size != new_size or int(old_mtime) != int(new_mtime):
                if old_hash is not None and new_hash is not None and old_hash != new_hash:
                    events.append(f"FILE_CHANGED: {rel}")
                else:
                    events.append(f"FILE_CHANGED_META: {rel}")

        self.baseline = current_index

        if events:
            self.last_status = f"{len(events)} file cache changes detected."
        else:
            self.last_status = f"No file cache changes. Files indexed: {len(self.baseline)}."

        for ev in events:
            self.recent_events.append(ev)

        return events

    def tick(self):
        if not self.enabled:
            self.last_status = "No folder selected."
            return []

        now = time.time()
        if self.first_scan_done and (now - self.last_scan_time) < GAME_FILE_SCAN_INTERVAL:
            return []

        self.last_scan_time = now
        return self._scan_once()

# ============================================================
#  LOGGING WITH REDUNDANCY STATUS
# ============================================================

def safe_append(path: str, text: str):
    try:
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
        return True, None
    except Exception as e:
        return False, str(e)

def log_to_disk_and_network(node_id: str, snapshot: dict,
                            primary_drive: NetworkDriveStatus,
                            backup_drive: NetworkDriveStatus):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{ts}] NODE={node_id} "
        f"HEALTH={snapshot['health']:.1f} "
        f"POSTURE={snapshot['posture']} "
        f"GAME={snapshot['game']} "
        f"FG='{snapshot['foreground'][:40]}' "
        f"STATUS={snapshot['status']}\n"
    )

    local_ok, local_err = safe_append(LOCAL_LOG_FILE, line)

    primary_ok = False
    primary_err = None
    if primary_drive:
        primary_path = os.path.join(primary_drive.path, "borg_node_log.txt")
        primary_ok, primary_err = safe_append(primary_path, line)

    backup_ok = False
    backup_err = None
    if backup_drive:
        backup_path = os.path.join(backup_drive.path, "borg_node_log_backup.txt")
        backup_ok, backup_err = safe_append(backup_path, line)

    return {
        "local_ok": local_ok,
        "local_err": local_err,
        "primary_ok": primary_ok,
        "primary_err": primary_err,
        "backup_ok": backup_ok,
        "backup_err": backup_err,
    }

# ============================================================
#  BORG NODE BRAIN
# ============================================================

class BorgNodeBrain:
    def __init__(self, node_id="LOCAL", simulated=False,
                 primary_drive=None, backup_drive=None,
                 file_cache_monitor: GameFileCacheMonitor = None):
        self.node_id = node_id
        self.simulated = simulated
        self.primary_drive = primary_drive
        self.backup_drive = backup_drive
        self.file_cache_monitor = file_cache_monitor

        self.health_score = 100.0
        self.posture = "WATCHER"
        self.last_status = "Initializing node..."
        self.event_history = deque(maxlen=80)
        self.metric_history = deque(maxlen=200)
        self.game_telemetry_history = deque(maxlen=50)
        self.last_tick_time = time.time()
        self.current_game = "None"
        self.foreground_app = "Unknown"
        self.os_name = platform.system()

        self.game_proc = None
        self.last_redundancy_status = {
            "local_ok": True,
            "primary_ok": False,
            "backup_ok": False,
        }

    # ---------- Sensing ----------

    def _read_metrics_real(self):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        metrics = {"timestamp": now, "dt": dt}

        if psutil:
            try:
                metrics["cpu"] = psutil.cpu_percent(interval=None)
                metrics["mem"] = psutil.virtual_memory().percent
            except Exception:
                metrics["cpu"] = random.uniform(5, 95)
                metrics["mem"] = random.uniform(10, 90)
        else:
            metrics["cpu"] = random.uniform(5, 95)
            metrics["mem"] = random.uniform(10, 90)

        self.metric_history.append(metrics)
        return metrics

    def _read_metrics_sim(self):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        base = max(0.2, self.health_score / 100.0)
        cpu = random.uniform(10, 80) * base + random.uniform(0, 10)
        mem = random.uniform(20, 80)
        metrics = {
            "timestamp": now,
            "dt": dt,
            "cpu": max(0.0, min(100.0, cpu)),
            "mem": max(0.0, min(100.0, mem)),
        }
        self.metric_history.append(metrics)
        return metrics

    def _read_metrics(self):
        return self._read_metrics_sim() if self.simulated else self._read_metrics_real()

    def _find_game_process_real(self):
        if not psutil:
            return "None", None

        back4blood_proc = None
        generic_game_proc = None

        try:
            for proc in psutil.process_iter(attrs=["name"]):
                name = (proc.info.get("name") or "").lower()
                for exe_name, game_name in KNOWN_GAMES.items():
                    if exe_name in name:
                        if "back4blood" in exe_name:
                            back4blood_proc = proc
                        else:
                            generic_game_proc = proc
            if back4blood_proc:
                return "BACK 4 BLOOD", back4blood_proc
            if generic_game_proc:
                return "GENERIC GAME", generic_game_proc
        except Exception:
            pass

        return "None", None

    def _find_game_process_sim(self):
        r = random.random()
        if r < 0.1:
            return "BACK 4 BLOOD", None
        elif r < 0.3:
            return "GENERIC GAME", None
        return "None", None

    def _detect_game_and_proc(self):
        if self.simulated:
            return self._find_game_process_sim()
        return self._find_game_process_real()

    def _foreground_real(self):
        if self.os_name != "Windows" or not win32gui:
            return "Unknown"
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                return title or "No active window"
            return "No active window"
        except Exception:
            return "Error reading foreground"

    def _foreground_sim(self, game):
        if game == "BACK 4 BLOOD":
            return "Back 4 Blood - Mission"
        elif game != "None":
            return "Game Session"
        return "Desktop / Explorer"

    def _foreground(self, game):
        return self._foreground_sim(game) if self.simulated else self._foreground_real()

    def _read_game_telemetry(self, game_name, proc):
        tel = {
            "timestamp": time.time(),
            "game": game_name,
            "cpu": None,
            "rss_mb": None,
            "threads": None,
            "io_read_mb": None,
            "io_write_mb": None,
        }

        if game_name == "None" or proc is None or not psutil:
            self.game_telemetry_history.append(tel)
            return tel

        try:
            cpu = proc.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            threads = proc.num_threads()
            try:
                io = proc.io_counters()
                io_read_mb = io.read_bytes / (1024 * 1024)
                io_write_mb = io.write_bytes / (1024 * 1024)
            except Exception:
                io_read_mb = None
                io_write_mb = None

            tel.update({
                "cpu": cpu,
                "rss_mb": rss_mb,
                "threads": threads,
                "io_read_mb": io_read_mb,
                "io_write_mb": io_write_mb,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        self.game_telemetry_history.append(tel)
        return tel

    # ---------- Weirdness ----------

    def _detect_weirdness(self, metrics, game, fg, profile, game_tel, file_cache_events, redundancy_status):
        events = []

        if len(self.metric_history) >= 10:
            avg_cpu = sum(m["cpu"] for m in self.metric_history) / len(self.metric_history)
            avg_mem = sum(m["mem"] for m in self.metric_history) / len(self.metric_history)

            cpu = metrics["cpu"]
            mem = metrics["mem"]

            if cpu > avg_cpu + profile["cpu_high_delta"]:
                events.append("CPU_SPIKE")
            if mem > avg_mem + profile["mem_high_delta"]:
                events.append("MEM_PRESSURE")
            if cpu < max(0, avg_cpu - profile["cpu_high_delta"]) and cpu < 5:
                events.append("CPU_DROP")
            if mem < max(0, avg_mem - profile["mem_high_delta"]) and mem < 20:
                events.append("MEM_DROP")

            if game != "None" and cpu < profile["cpu_idle_threshold"]:
                events.append("GAME_IDLE_LOW_CPU")
            if game != "None" and mem > profile["mem_critical"]:
                events.append("GAME_HIGH_MEM")
            if "error" in fg.lower() or "crash" in fg.lower():
                events.append("FOREGROUND_ERROR_WINDOW")

        if game != "None" and game_tel["cpu"] is not None:
            if game_tel["cpu"] < 1.0:
                events.append("GAME_PROC_IDLE")
            if game_tel["cpu"] > 90.0:
                events.append("GAME_PROC_CPU_HEAVY")
        if game != "None" and game_tel["rss_mb"] is not None:
            if game_tel["rss_mb"] > 8000:
                events.append("GAME_PROC_HUGE_RAM")

        if self.primary_drive:
            if not self.primary_drive.online:
                events.append("PRIMARY_OFFLINE")
            elif self.primary_drive.free_bytes is not None and self.primary_drive.free_bytes < MIN_FREE_SPACE_BYTES:
                events.append("PRIMARY_LOW_SPACE")

        if self.backup_drive:
            if not self.backup_drive.online:
                events.append("BACKUP_OFFLINE")
            elif self.backup_drive.free_bytes is not None and self.backup_drive.free_bytes < MIN_FREE_SPACE_BYTES:
                events.append("BACKUP_LOW_SPACE")

        for ev in file_cache_events:
            events.append(ev)

        if not redundancy_status["primary_ok"]:
            events.append("REDUNDANCY_PRIMARY_FAIL")
        if not redundancy_status["backup_ok"]:
            events.append("REDUNDANCY_BACKUP_FAIL")

        if random.random() < 0.02:
            events.append("STRANGE_PATTERN")

        for ev in events:
            self.event_history.append((metrics["timestamp"], ev))
        return events

    # ---------- Reasoning ----------

    def _reason(self, events, metrics, game, fg, redundancy_status):
        health = self.health_score
        for ev in events:
            if ev in ("CPU_SPIKE", "MEM_PRESSURE", "GAME_HIGH_MEM",
                      "GAME_PROC_HUGE_RAM", "GAME_PROC_CPU_HEAVY"):
                health -= random.uniform(3, 8)
            elif ev in ("CPU_DROP", "MEM_DROP", "GAME_IDLE_LOW_CPU", "GAME_PROC_IDLE"):
                health -= random.uniform(1, 4)
            elif ev == "FOREGROUND_ERROR_WINDOW":
                health -= random.uniform(5, 12)
            elif ev == "STRANGE_PATTERN":
                health -= random.uniform(3, 8)
            elif ev in ("PRIMARY_OFFLINE", "BACKUP_OFFLINE"):
                health -= random.uniform(4, 10)
            elif ev in ("PRIMARY_LOW_SPACE", "BACKUP_LOW_SPACE"):
                health -= random.uniform(2, 6)
            elif ev.startswith("FILE_ADDED") or ev.startswith("FILE_MISSING"):
                health -= random.uniform(2, 6)
            elif ev.startswith("FILE_CHANGED"):
                health -= random.uniform(1, 4)
            elif ev == "REDUNDANCY_PRIMARY_FAIL":
                health -= random.uniform(4, 10)
            elif ev == "REDUNDANCY_BACKUP_FAIL":
                health -= random.uniform(4, 10)

        if not events:
            health += 1.2

        health = max(0.0, min(100.0, health))

        if health > 80:
            posture = "WATCHER"
            mood = "Calm, scanning."
        elif health > 50:
            posture = "GUARDIAN"
            mood = "Tense, actively defending."
        else:
            posture = "CHAMELEON"
            mood = "Critical, minimizing surface and adapting."

        if game != "None":
            context = f"In-game: {game}."
        else:
            context = "No primary game detected."

        drive_bits = []
        if self.primary_drive:
            if self.primary_drive.online:
                drive_bits.append(f"Primary OK ({self.primary_drive.human_free()} free).")
            else:
                drive_bits.append("Primary offline.")
        if self.backup_drive:
            if self.backup_drive.online:
                drive_bits.append(f"Backup OK ({self.backup_drive.human_free()} free).")
            else:
                drive_bits.append("Backup offline.")
        drive_str = " ".join(drive_bits)

        redundancy_bits = []
        if redundancy_status["primary_ok"]:
            redundancy_bits.append("Primary log OK.")
        else:
            redundancy_bits.append("Primary log FAILED.")
        if redundancy_status["backup_ok"]:
            redundancy_bits.append("Backup log OK.")
        else:
            redundancy_bits.append("Backup log FAILED.")
        redundancy_str = " ".join(redundancy_bits)

        if events:
            ev_summary = ", ".join(events)
            status = (
                f"{mood} {context} {drive_str} {redundancy_str} "
                f"Foreground: '{fg[:60]}'. "
                f"Anomalies: {ev_summary}. "
                f"CPU {metrics['cpu']:.1f}%, MEM {metrics['mem']:.1f}%."
            )
        else:
            status = (
                f"{mood} {context} {drive_str} {redundancy_str} "
                f"Foreground: '{fg[:60]}'. "
                f"No new anomalies. CPU {metrics['cpu']:.1f}%, MEM {metrics['mem']:.1f}%."
            )

        self.health_score = health
        self.posture = posture
        self.last_status = status
        self.current_game = game
        self.foreground_app = fg

    # ---------- Public Tick ----------

    def tick(self):
        if self.primary_drive:
            self.primary_drive.probe()
        if self.backup_drive:
            self.backup_drive.probe()

        file_cache_events = []
        file_cache_status = "No file cache monitor."
        file_cache_recent = []
        if self.file_cache_monitor:
            file_cache_events = self.file_cache_monitor.tick()
            file_cache_status = self.file_cache_monitor.last_status
            file_cache_recent = list(self.file_cache_monitor.recent_events)[-5:]

        metrics = self._read_metrics()
        game_name, game_proc = self._detect_game_and_proc()
        profile = get_profile_for_game(game_name)
        fg = self._foreground(game_name)
        game_tel = self._read_game_telemetry(game_name, game_proc)

        prelim_snapshot = {
            "node_id": self.node_id,
            "health": self.health_score,
            "posture": self.posture,
            "status": self.last_status,
            "events": list(self.event_history)[-5:],
            "game": game_name,
            "foreground": fg,
        }

        redundancy_status = log_to_disk_and_network(
            self.node_id, prelim_snapshot,
            self.primary_drive, self.backup_drive
        )

        self.last_redundancy_status = redundancy_status

        events = self._detect_weirdness(
            metrics, game_name, fg, profile, game_tel,
            file_cache_events, redundancy_status
        )
        self._reason(events, metrics, game_name, fg, redundancy_status)

        snapshot = {
            "node_id": self.node_id,
            "health": self.health_score,
            "posture": self.posture,
            "status": self.last_status,
            "events": list(self.event_history)[-5:],
            "game": self.current_game,
            "foreground": self.foreground_app,
            "autoloader_log": list(AUTOLOADER_LOG)[-5:],
            "profile_label": profile["label"],
            "profile_description": profile["description"],
            "game_telemetry": game_tel,
            "file_cache_status": file_cache_status,
            "file_cache_events": file_cache_recent,
            "redundancy_status": redundancy_status,
        }

        return snapshot

# ============================================================
#  DRIVE PICKER (VISIBLE, SIMPLE)
# ============================================================

class DrivePicker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Select Network Drives")
        self.configure(bg=PALETTE["bg"])
        self.geometry("500x300")
        self.resizable(False, False)

        self.primary_path = ""
        self.backup_path = ""

        self._build()

    def _build(self):
        title = tk.Label(
            self,
            text="Select Primary & Backup Drives",
            fg=PALETTE["glyph"],
            bg=PALETTE["bg"],
            font=("Consolas", 16, "bold")
        )
        title.pack(pady=10)

        btn_primary = tk.Button(
            self,
            text="Select PRIMARY Drive",
            command=self.select_primary,
            bg=PALETTE["rail_accent"],
            fg=PALETTE["text"],
            width=30
        )
        btn_primary.pack(pady=5)

        self.primary_label = tk.Label(
            self,
            text="Primary: (none)",
            fg=PALETTE["text"],
            bg=PALETTE["bg"]
        )
        self.primary_label.pack()

        btn_backup = tk.Button(
            self,
            text="Select BACKUP Drive",
            command=self.select_backup,
            bg=PALETTE["rail_accent"],
            fg=PALETTE["text"],
            width=30
        )
        btn_backup.pack(pady=10)

        self.backup_label = tk.Label(
            self,
            text="Backup: (none)",
            fg=PALETTE["text"],
            bg=PALETTE["bg"]
        )
        self.backup_label.pack()

        self.confirm_btn = tk.Button(
            self,
            text="CONFIRM & LAUNCH",
            command=self.confirm,
            state="disabled",
            bg=PALETTE["threat_calm"],
            fg=PALETTE["text"],
            width=30
        )
        self.confirm_btn.pack(pady=20)

    def select_primary(self):
        path = filedialog.askdirectory(title="Select Primary Drive")
        if path:
            self.primary_path = path
            self.primary_label.config(text=f"Primary: {path}")
            self._update_confirm()

    def select_backup(self):
        path = filedialog.askdirectory(title="Select Backup Drive")
        if path:
            self.backup_path = path
            self.backup_label.config(text=f"Backup: {path}")
            self._update_confirm()

    def _update_confirm(self):
        if self.primary_path and self.backup_path:
            self.confirm_btn.config(state="normal")

    def confirm(self):
        if not os.path.exists(self.primary_path):
            self.primary_label.config(text="Primary: INVALID PATH")
            return
        if not os.path.exists(self.backup_path):
            self.backup_label.config(text="Backup: INVALID PATH")
            return
        self.destroy()

# ============================================================
#  FULL LCARS UI
# ============================================================

class LCARSDefenseUI(tk.Tk):
    def __init__(self, primary_path, backup_path, game_folder=""):
        super().__init__()
        self.title("LCARS-DARK ASI Defense Interface · Borg Collective")
        self.configure(bg=PALETTE["bg"])
        self.geometry("1600x900")

        self.primary_drive = NetworkDriveStatus("PRIMARY", primary_path)
        self.backup_drive = NetworkDriveStatus("BACKUP", backup_path)

        self.file_cache_monitor = GameFileCacheMonitor()
        if game_folder and os.path.isdir(game_folder):
            self.file_cache_monitor.set_path(game_folder)

        self.nodes = [
            BorgNodeBrain("LOCAL", simulated=False,
                          primary_drive=self.primary_drive,
                          backup_drive=self.backup_drive,
                          file_cache_monitor=self.file_cache_monitor),
            BorgNodeBrain("REMOTE-1", simulated=True),
            BorgNodeBrain("REMOTE-2", simulated=True),
        ]

        self._build(game_folder)
        self._loop()

    def _build(self, game_folder):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = tk.Frame(self, bg=PALETTE["bg"])
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left rail
        left = tk.Frame(main, bg=PALETTE["rail"], width=340)
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)

        lh = tk.Label(
            left,
            text=to_alien("COLLECTIVE OVERVIEW"),
            fg=PALETTE["glyph"],
            bg=PALETTE["rail"],
            font=("Consolas", 16, "bold")
        )
        lh.pack(anchor="nw", padx=10, pady=(10, 5))

        self.collective_health = tk.Label(
            left,
            text="COLLECTIVE HEALTH: 100.0",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 12)
        )
        self.collective_health.pack(anchor="nw", padx=10, pady=5)

        self.node_count = tk.Label(
            left,
            text=f"NODES: {len(self.nodes)}",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 12)
        )
        self.node_count.pack(anchor="nw", padx=10, pady=5)

        # Node list
        node_frame = tk.LabelFrame(
            left,
            text="Nodes",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            bd=1
        )
        node_frame.pack(fill="x", padx=10, pady=(10, 5))
        self.node_list = tk.Listbox(
            node_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 10),
            height=5
        )
        self.node_list.pack(fill="both", padx=5, pady=5)

        # Network drives
        drive_frame = tk.LabelFrame(
            left,
            text="Network Drives",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            bd=1
        )
        drive_frame.pack(fill="x", padx=10, pady=(5, 5))

        self.primary_drive_label = tk.Label(
            drive_frame,
            text=f"PRIMARY: {self.primary_drive.path}",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=300
        )
        self.primary_drive_label.pack(anchor="w", padx=5, pady=(5, 2))

        self.primary_drive_status = tk.Label(
            drive_frame,
            text="Status: ...",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w"
        )
        self.primary_drive_status.pack(anchor="w", padx=5, pady=(0, 5))

        self.backup_drive_label = tk.Label(
            drive_frame,
            text=f"BACKUP: {self.backup_drive.path}",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=300
        )
        self.backup_drive_label.pack(anchor="w", padx=5, pady=(5, 2))

        self.backup_drive_status = tk.Label(
            drive_frame,
            text="Status: ...",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w"
        )
        self.backup_drive_status.pack(anchor="w", padx=5, pady=(0, 5))

        # Game file cache
        file_frame = tk.LabelFrame(
            left,
            text="Game File Cache",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            bd=1
        )
        file_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(3, weight=1)

        init_path_text = (
            f"Install path: {game_folder}" if game_folder else "Install path: (none)"
        )
        self.file_cache_path_label = tk.Label(
            file_frame,
            text=init_path_text,
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=300
        )
        self.file_cache_path_label.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 2))

        btn_game_folder = tk.Button(
            file_frame,
            text="Select Game Folder...",
            command=self._select_game_folder,
            bg=PALETTE["rail_accent"],
            fg=PALETTE["text"],
            activebackground=PALETTE["rail"],
            activeforeground=PALETTE["text"],
            relief="flat"
        )
        btn_game_folder.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

        initial_file_status = self.file_cache_monitor.last_status
        self.file_cache_status_label = tk.Label(
            file_frame,
            text=f"Status: {initial_file_status}",
            fg=PALETTE["text"],
            bg=PALETTE["rail"],
            font=("Consolas", 9),
            anchor="w",
            justify="left",
            wraplength=300
        )
        self.file_cache_status_label.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))

        self.file_cache_events_list = tk.Listbox(
            file_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 9)
        )
        self.file_cache_events_list.grid(row=3, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # Core panel
        core = tk.Frame(main, bg=PALETTE["pod_bg"], bd=2, relief="solid")
        core.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        core.columnconfigure(0, weight=1)
        core.rowconfigure(3, weight=1)
        core.rowconfigure(4, weight=1)

        self.header_label = tk.Label(
            core,
            text=to_alien("BORG NODE COLLECTIVE HEALTH"),
            fg=PALETTE["glyph"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 20, "bold")
        )
        self.header_label.grid(row=0, column=0, sticky="w", padx=20, pady=10)

        top = tk.Frame(core, bg=PALETTE["pod_bg"])
        top.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)

        self.health_label = tk.Label(
            top,
            text="Health: 100.0",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 14)
        )
        self.health_label.grid(row=0, column=0, sticky="w", padx=(0, 20))

        self.posture_label = tk.Label(
            top,
            text="Posture: WATCHER",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 14)
        )
        self.posture_label.grid(row=0, column=1, sticky="w")

        self.game_label = tk.Label(
            top,
            text="Game: None",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 14)
        )
        self.game_label.grid(row=0, column=2, sticky="w")

        mode_frame = tk.LabelFrame(
            core,
            text="Mode Card",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            bd=1
        )
        mode_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.mode_title_label = tk.Label(
            mode_frame,
            text="PROFILE: DESKTOP PROFILE",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 12, "bold"),
            anchor="w",
            justify="left"
        )
        self.mode_title_label.pack(anchor="w", padx=5, pady=(5, 2))

        self.mode_desc_label = tk.Label(
            mode_frame,
            text="No primary game detected; spikes are suspicious.",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 10),
            anchor="w",
            justify="left",
            wraplength=1100
        )
        self.mode_desc_label.pack(anchor="w", padx=5, pady=(0, 5))

        status_frame = tk.LabelFrame(
            core,
            text="Status / Reasoning",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            bd=1
        )
        status_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(
            status_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 11)
        )
        self.status_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.status_text.insert("end", "Node initializing...\n")
        self.status_text.config(state="disabled")

        bottom = tk.Frame(core, bg=PALETTE["pod_bg"])
        bottom.grid(row=4, column=0, sticky="nsew", padx=20, pady=(0, 20))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        events_frame = tk.LabelFrame(
            bottom,
            text="Recent anomalies",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            bd=1
        )
        events_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)

        self.events_list = tk.Listbox(
            events_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 10)
        )
        self.events_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        right_bottom = tk.Frame(bottom, bg=PALETTE["pod_bg"])
        right_bottom.grid(row=0, column=1, sticky="nsew")
        right_bottom.columnconfigure(0, weight=1)
        right_bottom.rowconfigure(0, weight=1)
        right_bottom.rowconfigure(1, weight=1)

        autoloader_frame = tk.LabelFrame(
            right_bottom,
            text="Autoloader",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            bd=1
        )
        autoloader_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=(0, 5))
        autoloader_frame.columnconfigure(0, weight=1)
        autoloader_frame.rowconfigure(0, weight=1)

        self.autoloader_list = tk.Listbox(
            autoloader_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 9)
        )
        self.autoloader_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        game_frame = tk.LabelFrame(
            right_bottom,
            text="Game Telemetry",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            bd=1
        )
        game_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(5, 0))
        game_frame.columnconfigure(0, weight=1)
        game_frame.rowconfigure(0, weight=1)

        self.game_tel_text = tk.Text(
            game_frame,
            fg=PALETTE["text"],
            bg=PALETTE["bg"],
            font=("Consolas", 9),
            height=6
        )
        self.game_tel_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.core_panel = core

    # ---------- GUI helpers ----------

    def _select_game_folder(self):
        path = filedialog.askdirectory(title="Select Game Install Folder (e.g., Back 4 Blood)")
        if path:
            self.file_cache_monitor.set_path(path)
            self.file_cache_path_label.config(text=f"Install path: {path}")
            CONFIG["game_folder"] = path
            save_config()

    def _apply_posture_theme(self, posture):
        theme = POSTURE_THEMES.get(posture, POSTURE_THEMES["WATCHER"])
        self.header_label.config(
            text=to_alien(theme["header_text"]),
            fg=theme["glyph_color"]
        )

    def _update_drive_status(self):
        if self.primary_drive.online:
            txt = f"Status: ONLINE ({self.primary_drive.human_free()} free / {self.primary_drive.human_total()} total)"
        else:
            err = self.primary_drive.last_error or "Offline"
            txt = f"Status: OFFLINE ({err})"
        self.primary_drive_status.config(text=txt)

        if self.backup_drive.online:
            txt = f"Status: ONLINE ({self.backup_drive.human_free()} free / {self.backup_drive.human_total()} total)"
        else:
            err = self.backup_drive.last_error or "Offline"
            txt = f"Status: OFFLINE ({err})"
        self.backup_drive_status.config(text=txt)

    def _update_game_telemetry_panel(self, game_tel):
        self.game_tel_text.config(state="normal")
        self.game_tel_text.delete("1.0", "end")

        if game_tel["game"] == "None":
            self.game_tel_text.insert("end", "No game process detected.\n")
        else:
            self.game_tel_text.insert("end", f"Game: {game_tel['game']}\n")
            if game_tel["cpu"] is not None:
                self.game_tel_text.insert("end", f"CPU: {game_tel['cpu']:.1f}%\n")
            else:
                self.game_tel_text.insert("end", "CPU: N/A\n")
            if game_tel["rss_mb"] is not None:
                self.game_tel_text.insert("end", f"Memory (RSS): {game_tel['rss_mb']:.1f} MB\n")
            else:
                self.game_tel_text.insert("end", "Memory (RSS): N/A\n")
            if game_tel["threads"] is not None:
                self.game_tel_text.insert("end", f"Threads: {game_tel['threads']}\n")
            else:
                self.game_tel_text.insert("end", "Threads: N/A\n")
            if game_tel["io_read_mb"] is not None:
                self.game_tel_text.insert("end", f"I/O Read: {game_tel['io_read_mb']:.1f} MB\n")
            else:
                self.game_tel_text.insert("end", "I/O Read: N/A\n")
            if game_tel["io_write_mb"] is not None:
                self.game_tel_text.insert("end", f"I/O Write: {game_tel['io_write_mb']:.1f} MB\n")
            else:
                self.game_tel_text.insert("end", "I/O Write: N/A\n")

        self.game_tel_text.config(state="disabled")

    def _update_file_cache_panel(self, status_text, events):
        self.file_cache_status_label.config(text=f"Status: {status_text}")
        self.file_cache_events_list.delete(0, "end")
        for ev in events:
            self.file_cache_events_list.insert("end", ev)

    def _update_view(self, snapshots):
        primary = snapshots[0]
        health = primary["health"]
        posture = primary["posture"]
        game = primary["game"]
        status_line = primary["status"]
        foreground = primary["foreground"]
        events = primary["events"]
        autolog = primary["autoloader_log"]
        profile_label = primary["profile_label"]
        profile_desc = primary["profile_description"]
        game_tel = primary["game_telemetry"]
        file_cache_status = primary["file_cache_status"]
        file_cache_events = primary["file_cache_events"]

        avg_health = sum(s["health"] for s in snapshots) / len(snapshots)
        self.collective_health.config(text=f"COLLECTIVE HEALTH: {avg_health:.1f}")
        self.node_count.config(text=f"NODES: {len(snapshots)}")

        self.node_list.delete(0, "end")
        for s in snapshots:
            self.node_list.insert(
                "end",
                f"{s['node_id']:>8}  {s['health']:6.1f}  {s['posture']:<9}  {s['game']}"
            )

        self.health_label.config(text=f"Health: {health:.1f}")
        self.posture_label.config(text=f"Posture: {posture}")
        self.game_label.config(text=f"Game: {game}")

        self.mode_title_label.config(text=f"PROFILE: {profile_label}")
        self.mode_desc_label.config(text=profile_desc)

        if health > 80:
            bg = PALETTE["threat_calm"]
        elif health > 50:
            bg = PALETTE["threat_tense"]
        else:
            bg = PALETTE["threat_critical"]

        self.core_panel.config(bg=bg)
        for child in self.core_panel.winfo_children():
            if isinstance(child, tk.Frame) or isinstance(child, tk.LabelFrame):
                child.config(bg=bg)
                for sub in child.winfo_children():
                    if isinstance(sub, tk.Label) or isinstance(sub, tk.Frame) or isinstance(sub, tk.LabelFrame):
                        sub.config(bg=bg)

        self._apply_posture_theme(posture)

        self.status_text.config(state="normal")
        self.status_text.insert("end", status_line + "\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")

        self.events_list.delete(0, "end")
        for ts, ev in events:
            timestr = time.strftime("%H:%M:%S", time.localtime(ts))
            self.events_list.insert("end", f"{timestr}  {ev}")

        self.autoloader_list.delete(0, "end")
        for line in autolog:
            self.autoloader_list.insert("end", line)

        self.title(f"LCARS-DARK Borg Collective · FG: {foreground[:50]}")

        self._update_drive_status()
        self._update_game_telemetry_panel(game_tel)
        self._update_file_cache_panel(file_cache_status, file_cache_events)

    def _loop(self):
        snapshots = [n.tick() for n in self.nodes]
        self._update_view(snapshots)
        self.after(1000, self._loop)

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    load_config()

    primary = CONFIG.get("primary_path", "")
    backup = CONFIG.get("backup_path", "")
    game_folder = CONFIG.get("game_folder", "")

    # If we already have valid paths, skip the picker
    if primary and backup and os.path.exists(primary) and os.path.exists(backup):
        app = LCARSDefenseUI(primary, backup, game_folder=game_folder)
        app.mainloop()
    else:
        picker = DrivePicker()
        picker.mainloop()

        if not picker.primary_path or not picker.backup_path:
            sys.exit(0)

        CONFIG["primary_path"] = picker.primary_path
        CONFIG["backup_path"] = picker.backup_path
        save_config()

        app = LCARSDefenseUI(picker.primary_path, picker.backup_path, game_folder=game_folder)
        app.mainloop()

