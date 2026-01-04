import sys
import os
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

import psutil
from PySide6 import QtWidgets, QtCore

# Optional GPU sensor
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# --- Windows-specific imports for foreground window and OS info ---
import ctypes
from ctypes import wintypes
import locale
import platform
import subprocess


# -------------------------------------------------------------------
# CONFIG & SETTINGS
# -------------------------------------------------------------------

APP_NAME = "Living Game Optimizer"

# Local persistent memory (per-user)
LOCAL_ROOT = Path(os.getenv("APPDATA", Path.home())) / "GameOptimizer"
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = LOCAL_ROOT / "settings.json"

GAME_DIR_HINTS = [
    r"\Steam\steamapps\common",
    r"\Epic Games\\",
    r"\GOG Galaxy\Games",
    r"\Riot Games\\",
    r"\Origin Games\\",
    r"\Battle.net\\",
]


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "network_root": "",  # SMB/UNC or mapped drive
    }


def save_settings(settings: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# -------------------------------------------------------------------
# Utility: OS info / language
# -------------------------------------------------------------------

def get_os_info():
    os_name = platform.system()
    os_version = platform.version()
    lang, enc = locale.getdefaultlocale()
    return {
        "os_name": os_name,
        "os_version": os_version,
        "language": lang or "unknown",
        "encoding": enc or "unknown",
    }


# -------------------------------------------------------------------
# Foreground window → process detection
# -------------------------------------------------------------------

user32 = ctypes.windll.user32

GetForegroundWindow = user32.GetForegroundWindow
GetWindowThreadProcessId = user32.GetWindowThreadProcessId

def get_foreground_pid():
    hwnd = GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value or None

def get_foreground_process():
    pid = get_foreground_pid()
    if not pid:
        return None
    try:
        return psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


# -------------------------------------------------------------------
# Game detection heuristics
# -------------------------------------------------------------------

def is_probable_game(proc: psutil.Process, gpu_usage_hint: float = 0.0) -> bool:
    try:
        exe = proc.exe()
        name = proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

    exe_lower = exe.lower()

    if any(hint.lower() in exe_lower for hint in GAME_DIR_HINTS):
        return True

    if exe_lower.endswith((".exe", ".bin")) and (
        "game" in exe_lower or "benchmark" in exe_lower or "demo" in exe_lower
    ):
        return True

    gamey_tokens = ["game", "demo", "benchmark", "playtest"]
    if any(tok in name for tok in gamey_tokens):
        return True

    if gpu_usage_hint > 40.0:
        return True

    return False


def profile_id_for_proc(proc: psutil.Process) -> str:
    try:
        exe = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        exe = "unknown"
    h = hashlib.sha1(exe.encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(exe).lower()}_{h}"


# -------------------------------------------------------------------
# Persistent profile storage (local + optional network)
# -------------------------------------------------------------------

class ProfileStore:
    def __init__(self, local_root: Path, network_root: str | None):
        self.local_root = local_root
        self.network_root = network_root if network_root else None

    def set_network_root(self, path: str | None):
        self.network_root = path if path else None

    def _profile_rel_path(self, profile_id: str) -> Path:
        return Path("profiles") / profile_id / "profile.json"

    def _local_path(self, profile_id: str) -> Path:
        return self.local_root / self._profile_rel_path(profile_id)

    def _network_path(self, profile_id: str) -> Path | None:
        if not self.network_root:
            return None
        return Path(self.network_root) / self._profile_rel_path(profile_id)

    def load_profile(self, profile_id: str, default_data: dict) -> dict:
        net_path = self._network_path(profile_id)
        if net_path and net_path.exists():
            try:
                data = json.loads(net_path.read_text(encoding="utf-8"))
                self._write_local(profile_id, data)
                return data
            except Exception:
                pass

        local_path = self._local_path(profile_id)
        if local_path.exists():
            try:
                return json.loads(local_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.save_profile(profile_id, default_data)
        return default_data

    def save_profile(self, profile_id: str, data: dict):
        self._write_local(profile_id, data)

        net_path = self._network_path(profile_id)
        if net_path:
            try:
                net_path.parent.mkdir(parents=True, exist_ok=True)
                net_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _write_local(self, profile_id: str, data: dict):
        path = self._local_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -------------------------------------------------------------------
# Profile data structures
# -------------------------------------------------------------------

@dataclass
class GameProfile:
    profile_id: str
    exe_path: str
    display_name: str
    sessions_observed: int = 0
    minutes_observed: float = 0.0
    scenario_diversity: float = 0.0
    pattern_noise: float = 0.0

    learning_percent: int = 0
    confidence_percent: int = 0

    successful_predictions: int = 0
    total_predictions: int = 0
    recent_anomalies: int = 0
    stability_score: float = 0.0

    last_update_ts: float = 0.0

    last_session_start: float = 0.0
    last_session_end: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict):
        return GameProfile(**data)


def default_profile(profile_id: str, exe_path: str) -> GameProfile:
    return GameProfile(
        profile_id=profile_id,
        exe_path=exe_path,
        display_name=os.path.basename(exe_path) if exe_path else profile_id,
    )


# -------------------------------------------------------------------
# Learning & confidence metrics
# -------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def update_learning_and_confidence(profile: GameProfile, seconds_delta: float):
    minutes = profile.minutes_observed
    sessions = profile.sessions_observed
    diversity = profile.scenario_diversity
    noise = profile.pattern_noise

    learning = (
        sessions * 10.0 +
        (minutes / 3.0) +
        (diversity * 15.0) -
        (noise * 20.0)
    )
    learning = int(clamp(learning, 0, 100))

    if profile.total_predictions > 0:
        success_rate = profile.successful_predictions / profile.total_predictions
    else:
        success_rate = 0.0

    confidence = (
        success_rate * 100.0 -
        (profile.recent_anomalies * 5.0) +
        (profile.stability_score * 0.5)
    )
    confidence = int(clamp(confidence, 0, 100))

    profile.learning_percent = learning
    profile.confidence_percent = confidence


# -------------------------------------------------------------------
# Inference engine abstraction (Movidius hook)
# -------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
        self.available = False

    def init(self):
        self.available = False

    def predict_risk(self, features: dict) -> float:
        return 0.0


class MovidiusInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.model_path = model_path
        self.device = None

    def init(self):
        try:
            # TODO: integrate real Movidius / OpenVINO setup here
            self.available = False
        except Exception:
            self.available = False

    def predict_risk(self, features: dict) -> float:
        if not self.available:
            return 0.0
        return 0.0


# -------------------------------------------------------------------
# Foresight engine (deep read-ahead / prediction, water-style)
# -------------------------------------------------------------------

class ForesightEngine:
    def __init__(self, window_size: int = 10000, inference_engine: InferenceEngine | None = None):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.long_term = {
            "cpu_avg": 0.0,
            "ram_avg": 0.0,
            "cpu_var": 0.0,
            "ram_var": 0.0,
            "samples": 0
        }
        self.smooth = {
            "cpu": 0.0,
            "ram": 0.0,
            "alpha": 0.2,
            "initialized": False,
        }
        self.inference_engine = inference_engine

    def add_sample(self, readings: dict):
        cpu = readings.get("cpu_total", 0.0)
        ram = readings.get("ram_percent", 0.0)

        self.history.append({
            "ts": time.time(),
            "cpu": cpu,
            "ram": ram,
        })

        lt = self.long_term
        n = lt["samples"] + 1
        lt["samples"] = n

        lt["cpu_avg"] += (cpu - lt["cpu_avg"]) / n
        lt["ram_avg"] += (ram - lt["ram_avg"]) / n

        lt["cpu_var"] += (cpu - lt["cpu_avg"]) ** 2
        lt["ram_var"] += (ram - lt["ram_avg"]) ** 2

        s = self.smooth
        if not s["initialized"]:
            s["cpu"] = cpu
            s["ram"] = ram
            s["initialized"] = True
        else:
            a = s["alpha"]
            s["cpu"] = a * cpu + (1 - a) * s["cpu"]
            s["ram"] = a * ram + (1 - a) * s["ram"]

    def compute_risk(self) -> float:
        if len(self.history) < 5:
            return 0.0

        cpu_now = self.history[-1]["cpu"]
        cpu_prev = self.history[0]["cpu"]

        ram_now = self.history[-1]["ram"]
        ram_prev = self.history[0]["ram"]

        cpu_trend = cpu_now - cpu_prev
        ram_trend = ram_now - ram_prev

        lt = self.long_term
        cpu_dev = abs(cpu_now - lt["cpu_avg"])
        ram_dev = abs(ram_now - lt["ram_avg"])

        s = self.smooth
        cpu_s = s["cpu"]
        ram_s = s["ram"]

        cpu_potential = max(cpu_s - lt["cpu_avg"], 0.0)
        ram_potential = max(ram_s - lt["ram_avg"], 0.0)

        risk = (
            max(cpu_trend, 0) * 0.6 +
            max(ram_trend, 0) * 0.4 +
            cpu_dev * 0.3 +
            ram_dev * 0.2 +
            cpu_potential * 0.3 +
            ram_potential * 0.2
        )

        base_risk = float(clamp(risk, 0.0, 100.0))

        if self.inference_engine and self.inference_engine.available:
            features = {
                "cpu_now": cpu_now,
                "ram_now": ram_now,
                "cpu_trend": cpu_trend,
                "ram_trend": ram_trend,
                "cpu_avg": lt["cpu_avg"],
                "ram_avg": lt["ram_avg"],
                "cpu_smooth": cpu_s,
                "ram_smooth": ram_s,
            }
            delta = self.inference_engine.predict_risk(features)
            base_risk = float(clamp(base_risk + delta, 0.0, 100.0))

        return base_risk

    def get_long_term_stats(self):
        lt = self.long_term
        if lt["samples"] > 1:
            cpu_var = lt["cpu_var"] / (lt["samples"] - 1)
            ram_var = lt["ram_var"] / (lt["samples"] - 1)
        else:
            cpu_var = 0.0
            ram_var = 0.0
        return {
            "cpu_avg": lt["cpu_avg"],
            "ram_avg": lt["ram_avg"],
            "cpu_var": cpu_var,
            "ram_var": ram_var,
            "samples": lt["samples"],
        }


# -------------------------------------------------------------------
# Power posture engine (CPU/GPU behavior hooks)
# -------------------------------------------------------------------

class Posture:
    CALM = "Calm"
    ENGAGED = "Game Engaged"
    REDLINE = "Redline"


class PowerPostureEngine:
    def __init__(self):
        self.plan_balanced = None
        self.plan_high_perf = None

    def _set_power_plan(self, guid: str | None):
        if not guid:
            return
        try:
            subprocess.run(
                ["powercfg", "/S", guid],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def apply_posture(self, posture: str, risk: float, game_proc: psutil.Process | None):
        if posture == Posture.CALM:
            self._set_power_plan(self.plan_balanced)
        elif posture == Posture.ENGAGED:
            if risk > 40.0:
                self._set_power_plan(self.plan_high_perf)
            else:
                self._set_power_plan(self.plan_balanced)
        elif posture == Posture.REDLINE:
            self._set_power_plan(self.plan_high_perf)

        self._adjust_priorities(posture, game_proc)

    def _adjust_priorities(self, posture: str, game_proc: psutil.Process | None):
        if game_proc is None:
            return

        try:
            if posture in (Posture.ENGAGED, Posture.REDLINE):
                game_proc.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                game_proc.nice(psutil.NORMAL_PRIORITY_CLASS)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        if posture == Posture.REDLINE:
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    if p.pid == game_proc.pid:
                        continue
                    name = (p.info["name"] or "").lower()
                    if any(tok in name for tok in ["updater", "update", "installer", "onedrive", "dropbox", "launcher"]):
                        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue


# -------------------------------------------------------------------
# Brain modes (altered states)
# -------------------------------------------------------------------

class BrainMode:
    BASELINE = "Baseline"
    HYPERVIGILANT = "Hypervigilant"
    RELAXED = "Relaxed"


# -------------------------------------------------------------------
# Posture decision engine
# -------------------------------------------------------------------

def decide_posture(readings: dict, profile: GameProfile | None, risk: float, brain_mode: str) -> str:
    cpu = readings.get("cpu_total", 0.0)
    ram = readings.get("ram_percent", 0.0)

    if not readings.get("is_game_active", False):
        return Posture.CALM

    risk_redline = 70.0
    cpu_redline = 80.0
    ram_redline = 85.0

    if brain_mode == BrainMode.HYPERVIGILANT:
        risk_redline -= 15.0
        cpu_redline -= 5.0
        ram_redline -= 5.0
    elif brain_mode == BrainMode.RELAXED:
        risk_redline += 10.0
        cpu_redline += 5.0
        ram_redline += 5.0

    if risk >= risk_redline:
        return Posture.REDLINE

    if cpu > cpu_redline or ram > ram_redline:
        return Posture.REDLINE

    return Posture.ENGAGED


# -------------------------------------------------------------------
# Sensors
# -------------------------------------------------------------------

def collect_gpu_usage() -> float:
    if not HAS_GPU:
        return 0.0
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0
        return max(gpu.load for gpu in gpus) * 100.0
    except Exception:
        return 0.0


def collect_system_metrics(active_game_proc: psutil.Process | None) -> dict:
    readings = {}
    readings["cpu_total"] = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    readings["ram_percent"] = ram.percent

    readings["gpu_total"] = collect_gpu_usage()

    if active_game_proc:
        try:
            readings["game_name"] = active_game_proc.name()
            readings["game_exe"] = active_game_proc.exe()
            p_cpu = active_game_proc.cpu_percent(interval=None)
            p_mem = active_game_proc.memory_percent()
            readings["game_cpu"] = p_cpu
            readings["game_mem"] = p_mem
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            readings["game_name"] = "Unknown"
            readings["game_exe"] = ""
            readings["game_cpu"] = 0.0
            readings["game_mem"] = 0.0
    else:
        readings["game_name"] = "None"
        readings["game_exe"] = ""
        readings["game_cpu"] = 0.0
        readings["game_mem"] = 0.0

    readings["fps"] = "--"
    readings["ping"] = "--"

    return readings


# -------------------------------------------------------------------
# Optimizer core (brain + loop)
# -------------------------------------------------------------------

class OptimizerCore(QtCore.QObject):
    state_updated = QtCore.Signal(dict)

    def __init__(self, profile_store: ProfileStore, poll_interval_ms: int = 500):
        super().__init__()
        self.store = profile_store
        self.poll_interval_ms = poll_interval_ms

        self.current_profile_id: str | None = None
        self.current_profile: GameProfile | None = None
        self.current_posture: str = Posture.CALM

        self.last_tick_ts = time.time()
        self.learning_enabled = True

        self.os_info = get_os_info()

        self.movidius_engine = MovidiusInferenceEngine(model_path=None)
        self.movidius_engine.init()

        self.foresight = ForesightEngine(window_size=10000, inference_engine=self.movidius_engine)
        self.power_engine = PowerPostureEngine()

        self.active_game_proc: psutil.Process | None = None

        self.brain_mode = BrainMode.BASELINE
        self.recent_stutter_score = 0.0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.poll_interval_ms)

    def tick(self):
        now = time.time()
        delta = now - self.last_tick_ts
        self.last_tick_ts = now

        fg_proc = get_foreground_process()
        gpu_hint = collect_gpu_usage() if HAS_GPU else 0.0
        is_game = is_probable_game(fg_proc, gpu_usage_hint=gpu_hint) if fg_proc else False

        self.active_game_proc = fg_proc if is_game else None

        readings = collect_system_metrics(self.active_game_proc)
        readings["is_game_active"] = is_game

        if is_game and fg_proc:
            pid = profile_id_for_proc(fg_proc)
            if pid != self.current_profile_id:
                self._load_profile_for_process(fg_proc, pid)
            if self.current_profile:
                self.current_profile.minutes_observed += delta / 60.0
                self.current_profile.scenario_diversity = min(
                    1.0, self.current_profile.scenario_diversity + delta / 600.0
                )
                self.current_profile.pattern_noise = max(
                    0.0, self.current_profile.pattern_noise - delta / 1200.0
                )
        else:
            if self.current_profile and self.current_profile.last_session_start > 0:
                self.current_profile.last_session_end = now
                self._save_current_profile()
            self.current_profile_id = None
            self.current_profile = None

        self.foresight.add_sample(readings)
        risk = self.foresight.compute_risk()
        readings["foresight_risk"] = risk

        if self.current_profile and self.learning_enabled:
            self.current_profile.total_predictions += 1
            if readings["cpu_total"] < 90.0:
                self.current_profile.successful_predictions += 1
                self.recent_stutter_score = max(0.0, self.recent_stutter_score - 0.5)
            else:
                self.current_profile.recent_anomalies += 1
                self.recent_stutter_score += 2.0

        self.recent_stutter_score = clamp(self.recent_stutter_score, 0.0, 100.0)

        if self.recent_stutter_score > 40.0:
            self.brain_mode = BrainMode.HYPERVIGILANT
        elif self.recent_stutter_score < 10.0 and self.current_profile and self.current_profile.learning_percent > 60:
            self.brain_mode = BrainMode.RELAXED
        else:
            self.brain_mode = BrainMode.BASELINE

        readings["brain_mode"] = self.brain_mode

        self.current_posture = decide_posture(readings, self.current_profile, risk, self.brain_mode)
        readings["posture"] = self.current_posture

        self.power_engine.apply_posture(self.current_posture, risk, self.active_game_proc)

        if self.current_profile and self.learning_enabled:
            self.current_profile.stability_score = max(
                0.0,
                100.0 - abs(readings["cpu_total"] - 50.0)
            )

            update_learning_and_confidence(self.current_profile, delta)

            if now - self.current_profile.last_update_ts > 5.0:
                self._save_current_profile()
                self.current_profile.last_update_ts = now

            readings["learning_percent"] = self.current_profile.learning_percent
            readings["confidence_percent"] = self.current_profile.confidence_percent
            readings["profile_name"] = self.current_profile.display_name
        else:
            readings["learning_percent"] = 0
            readings["confidence_percent"] = 0
            readings["profile_name"] = "None"

        lt_stats = self.foresight.get_long_term_stats()
        readings["lt_cpu_avg"] = lt_stats["cpu_avg"]
        readings["lt_ram_avg"] = lt_stats["ram_avg"]

        readings["os_info"] = self.os_info

        self.state_updated.emit(readings)

    def _load_profile_for_process(self, proc: psutil.Process, profile_id: str):
        now = time.time()
        try:
            exe = proc.exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            exe = ""
        default = default_profile(profile_id, exe).to_dict()
        data = self.store.load_profile(profile_id, default)
        self.current_profile = GameProfile.from_dict(data)
        self.current_profile_id = profile_id
        self.current_profile.sessions_observed += 1
        self.current_profile.last_update_ts = now
        self.current_profile.last_session_start = now

    def _save_current_profile(self):
        if self.current_profile is None:
            return
        self.store.save_profile(self.current_profile.profile_id, self.current_profile.to_dict())


# -------------------------------------------------------------------
# GUI - Main control panel
# -------------------------------------------------------------------

class OptimizerGUI(QtWidgets.QWidget):
    def __init__(self, core: OptimizerCore, store: ProfileStore, settings: dict):
        super().__init__()
        self.core = core
        self.store = store
        self.settings = settings

        self.core.state_updated.connect(self.update_stats)
        self.latest_stats = {}

        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self._build_layout()
        self.resize(420, 540)

    def _build_layout(self):
        layout = QtWidgets.QVBoxLayout()

        self.lbl_game = QtWidgets.QLabel("Game: None")
        self.lbl_profile = QtWidgets.QLabel("Profile: None")
        self.lbl_posture = QtWidgets.QLabel("Posture: Calm")
        self.lbl_mode = QtWidgets.QLabel("Brain Mode: Baseline")

        layout.addWidget(self.lbl_game)
        layout.addWidget(self.lbl_profile)
        layout.addWidget(self.lbl_posture)
        layout.addWidget(self.lbl_mode)

        self.lbl_learning = QtWidgets.QLabel("Learning: 0%")
        self.learning_bar = QtWidgets.QProgressBar()
        self.learning_bar.setRange(0, 100)
        self.learning_bar.setValue(0)

        self.lbl_confidence = QtWidgets.QLabel("Optimization Confidence: 0%")
        self.confidence_bar = QtWidgets.QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)

        layout.addWidget(self.lbl_learning)
        layout.addWidget(self.learning_bar)
        layout.addWidget(self.lbl_confidence)
        layout.addWidget(self.confidence_bar)

        self.lbl_risk = QtWidgets.QLabel("Foresight Risk: 0%")
        layout.addWidget(self.lbl_risk)

        self.lbl_cpu = QtWidgets.QLabel("CPU: -- %")
        self.lbl_ram = QtWidgets.QLabel("RAM: -- %")
        self.lbl_gpu = QtWidgets.QLabel("GPU: -- %")
        self.lbl_game_cpu = QtWidgets.QLabel("Game CPU: -- %")
        self.lbl_game_mem = QtWidgets.QLabel("Game RAM: -- %")
        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.lbl_ping = QtWidgets.QLabel("Ping: --")
        self.lbl_lt = QtWidgets.QLabel("Long-term CPU/RAM: -- / -- %")

        layout.addWidget(self.lbl_cpu)
        layout.addWidget(self.lbl_ram)
        layout.addWidget(self.lbl_gpu)
        layout.addWidget(self.lbl_game_cpu)
        layout.addWidget(self.lbl_game_mem)
        layout.addWidget(self.lbl_fps)
        layout.addWidget(self.lbl_ping)
        layout.addWidget(self.lbl_lt)

        self.chk_learning = QtWidgets.QCheckBox("Learning enabled")
        self.chk_learning.setChecked(True)
        self.chk_learning.stateChanged.connect(self._on_learning_toggled)
        layout.addWidget(self.chk_learning)

        self.btn_summary = QtWidgets.QPushButton("Show Session Summary")
        self.btn_summary.clicked.connect(self.show_summary)
        layout.addWidget(self.btn_summary)

        # Network memory selection
        self.lbl_net = QtWidgets.QLabel("Network memory: (none)")
        self._update_net_label()
        layout.addWidget(self.lbl_net)

        self.btn_net = QtWidgets.QPushButton("Select Network Memory Location…")
        self.btn_net.clicked.connect(self.select_network_memory)
        layout.addWidget(self.btn_net)

        self.lbl_os = QtWidgets.QLabel("OS: Unknown")
        layout.addWidget(self.lbl_os)

        self.setLayout(layout)

    def _update_net_label(self):
        root = self.settings.get("network_root", "") or "(none)"
        self.lbl_net.setText(f"Network memory: {root}")

    def _on_learning_toggled(self, state):
        self.core.learning_enabled = (state == QtCore.Qt.Checked)

    def select_network_memory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Network/Folder for Optimizer Memory",
            self.settings.get("network_root", "") or ""
        )
        if not path:
            return

        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / "._optimizer_test_write.tmp"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Network Path Error",
                f"Cannot write to selected path:\n{path}\n\n{e}"
            )
            return

        self.settings["network_root"] = path
        save_settings(self.settings)

        self.store.set_network_root(path)
        self._update_net_label()

        QtWidgets.QMessageBox.information(
            self,
            "Network Memory Set",
            f"Network memory location set to:\n{path}\n\n"
            "Profiles and learning data will be stored there for future reboots."
        )

    @QtCore.Slot(dict)
    def update_stats(self, stats: dict):
        self.latest_stats = stats

        game_name = stats.get("game_name", "None")
        profile_name = stats.get("profile_name", "None")
        posture = stats.get("posture", "Calm")
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        risk = stats.get("foresight_risk", 0.0)
        brain_mode = stats.get("brain_mode", BrainMode.BASELINE)

        self.lbl_game.setText(f"Game: {game_name}")
        self.lbl_profile.setText(f"Profile: {profile_name}")
        self.lbl_posture.setText(f"Posture: {posture}")
        self.lbl_mode.setText(f"Brain Mode: {brain_mode}")

        self.learning_bar.setValue(learn)
        self.lbl_learning.setText(f"Learning: {learn}%")
        self.confidence_bar.setValue(conf)
        self.lbl_confidence.setText(f"Optimization Confidence: {conf}%")

        self.lbl_risk.setText(f"Foresight Risk: {risk:.0f}%")

        self.lbl_cpu.setText(f"CPU: {stats.get('cpu_total', 0):.0f} %")
        self.lbl_ram.setText(f"RAM: {stats.get('ram_percent', 0):.0f} %")
        self.lbl_gpu.setText(f"GPU: {stats.get('gpu_total', 0):.0f} %")
        self.lbl_game_cpu.setText(f"Game CPU: {stats.get('game_cpu', 0):.0f} %")
        self.lbl_game_mem.setText(f"Game RAM: {stats.get('game_mem', 0):.0f} %")
        self.lbl_fps.setText(f"FPS: {stats.get('fps', '--')}")
        self.lbl_ping.setText(f"Ping: {stats.get('ping', '--')} ms")

        lt_cpu = stats.get("lt_cpu_avg", 0.0)
        lt_ram = stats.get("lt_ram_avg", 0.0)
        self.lbl_lt.setText(f"Long-term CPU/RAM: {lt_cpu:.0f} / {lt_ram:.0f} %")

        os_info = stats.get("os_info", {})
        os_str = f"{os_info.get('os_name', '')} {os_info.get('os_version', '')} / {os_info.get('language', '')}"
        self.lbl_os.setText(f"OS: {os_str}")

    def show_summary(self):
        stats = self.latest_stats or {}
        game = stats.get("game_name", "None")
        profile = stats.get("profile_name", "None")
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        risk = stats.get("foresight_risk", 0.0)
        cpu = stats.get("cpu_total", 0.0)
        ram = stats.get("ram_percent", 0.0)
        gpu = stats.get("gpu_total", 0.0)
        lt_cpu = stats.get("lt_cpu_avg", 0.0)
        lt_ram = stats.get("lt_ram_avg", 0.0)

        msg = (
            f"Game: {game}\n"
            f"Profile: {profile}\n"
            f"Learning: {learn}%\n"
            f"Optimization Confidence: {conf}%\n"
            f"Foresight Risk (last): {risk:.0f}%\n\n"
            f"Current CPU: {cpu:.0f}% | RAM: {ram:.0f}% | GPU: {gpu:.0f}%\n"
            f"Long-term CPU avg: {lt_cpu:.0f}% | RAM avg: {lt_ram:.0f}%\n"
        )
        QtWidgets.QMessageBox.information(self, "Session Summary", msg)


# -------------------------------------------------------------------
# Minimalist in-game overlay HUD
# -------------------------------------------------------------------

class OverlayHUD(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self._build_layout()
        self.resize(260, 60)
        self._position_top_center()

    def _build_layout(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_posture = QtWidgets.QLabel("Calm")
        self.lbl_posture.setStyleSheet("color: white; font-weight: bold;")

        self.risk_bar = QtWidgets.QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setFixedHeight(14)
        self.risk_bar.setTextVisible(False)
        self.risk_bar.setStyleSheet(
            "QProgressBar {"
            "  background-color: rgba(40,40,40,180);"
            "  border: 1px solid rgba(255,255,255,80);"
            "}"
            "QProgressBar::chunk {"
            "  background-color: rgba(200,80,80,220);"
            "}"
        )

        self.lbl_lc = QtWidgets.QLabel("L:0 C:0")
        self.lbl_lc.setStyleSheet("color: white;")

        layout.addWidget(self.lbl_posture)
        layout.addWidget(self.risk_bar, 1)
        layout.addWidget(self.lbl_lc)
        self.setLayout(layout)

    def _position_top_center(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w, h = self.width(), self.height()
        x = screen.x() + (screen.width() - w) // 2
        y = screen.y() + 10
        self.move(x, y)

    def update_overlay(self, posture: str, risk: float, learning: int, confidence: int, visible: bool):
        if not visible:
            self.hide()
            return
        self.show()
        self.lbl_posture.setText(posture)
        self.risk_bar.setValue(int(risk))
        self.lbl_lc.setText(f"L:{learning} C:{confidence}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    settings = load_settings()
    network_root = settings.get("network_root", "") or None
    if network_root:
        Path(network_root).mkdir(parents=True, exist_ok=True)

    store = ProfileStore(LOCAL_ROOT, network_root)

    app = QtWidgets.QApplication(sys.argv)
    core = OptimizerCore(store)
    gui = OptimizerGUI(core, store, settings)
    overlay = OverlayHUD()

    def on_state(stats: dict):
        posture = stats.get("posture", "Calm")
        risk = stats.get("foresight_risk", 0.0)
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        game_name = stats.get("game_name", "None")
        visible = (game_name != "None")
        overlay.update_overlay(posture, risk, learn, conf, visible)

    core.state_updated.connect(on_state)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

