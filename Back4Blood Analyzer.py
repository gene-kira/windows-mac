# ============================================================
#  LIVING HYBRID ORGANISM CONSOLE (NEON EDITION)
#  HybridBrain + Organs + Back4Blood Analyzer + GUI + NPU
# ============================================================

import sys
import os
import math
import time
import json
import random
import traceback
from collections import OrderedDict, deque
from typing import Dict, Any, List

# ------------------------------------------------------------
# Autoloader for core libraries
# ------------------------------------------------------------
import importlib
import subprocess


def autoload_libraries(required_libs: Dict[str, str]) -> Dict[str, Any]:
    loaded = {}
    for mod, pip_name in required_libs.items():
        try:
            loaded[mod] = importlib.import_module(mod)
        except ImportError:
            print(f"[AUTOLOADER] Missing: {mod} — installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                loaded[mod] = importlib.import_module(mod)
            except Exception as e:
                print(f"[AUTOLOADER] FAILED to install {pip_name}: {e}")
                loaded[mod] = None
    return loaded


LIBS = autoload_libraries({
    "psutil": "psutil",
    "PyQt5": "PyQt5",
    "numpy": "numpy",
    "pynvml": "nvidia-ml-py3",
})

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QFont, QLinearGradient, QBrush, QRadialGradient
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QTextEdit, QGridLayout, QFrame,
    QPushButton, QListWidget, QListWidgetItem, QSplitter,
    QLineEdit
)

# ============================================================
#  CONSTANTS / ROLES / GAME
# ============================================================

PRIMARY_ROLE = "guardian"
SECONDARY_ROLE = "oracle"

ROLE_LABELS = {
    "guardian": "System Guardian",
    "oracle": "Game Performance Oracle",
    "companion": "Predictive Companion",
    "combat": "Combat AI",
}

GAME_PROCESS_NAMES = ["Back4Blood.exe", "back4blood.exe", "b4b.exe"]

STATE_FILE = "hybrid_organism_state.json"

# Simple icons (glyphs) for organs
ORGAN_ICONS = {
    "DeepRAM": "🧠",
    "BackupEngine": "💾",
    "NetworkWatcher": "🌐",
    "GPUCache": "🎮",
    "Thermal": "🔥",
    "Disk": "💽",
    "VRAM": "🧊",
    "AICoach": "🎓",
    "SwarmNode": "🕸️",
    "Back4BloodAnalyzer": "🩸",
}

# ============================================================
#  TELEMETRY HELPERS
# ============================================================

def get_system_ram_usage():
    vm = psutil.virtual_memory()
    return vm.used / (1024 ** 3), vm.total / (1024 ** 3)


def get_cpu_usage():
    return psutil.cpu_percent(interval=None)


def get_gpu_vram_usage():
    if not NVML_AVAILABLE:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = mem.used / (1024 ** 3)
        total = mem.total / (1024 ** 3)
        return used, total
    except Exception:
        return None, None


def get_gpu_temp():
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return float(temp)
    except Exception:
        return None


def get_disk_io():
    try:
        io = psutil.disk_io_counters()
        return io.read_bytes, io.write_bytes
    except Exception:
        return 0, 0


def get_net_io():
    try:
        io = psutil.net_io_counters()
        return io.bytes_sent, io.bytes_recv
    except Exception:
        return 0, 0


def detect_game():
    for p in psutil.process_iter(attrs=["name"]):
        try:
            name = p.info["name"]
            if name in GAME_PROCESS_NAMES:
                return name
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def list_processes():
    procs = []
    for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info"]):
        try:
            info = p.info
            mem_mb = info["memory_info"].rss / (1024 ** 2)
            procs.append(
                f"{info['pid']:5d} | {info['name'][:25]:25s} | CPU={info['cpu_percent']:5.1f}% | MEM={mem_mb:7.1f} MB"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def list_services():
    services = []
    if hasattr(psutil, "win_service_iter"):
        try:
            for s in psutil.win_service_iter():
                try:
                    info = s.as_dict()
                    services.append(
                        f"{info['name'][:30]:30s} | status={info['status']}"
                    )
                except Exception:
                    continue
        except Exception:
            pass
    return services


def build_feature_vector(role: str, secondary: str, game_metrics: Dict[str, float]) -> List[float]:
    cpu = get_cpu_usage() / 100.0
    ram_used, ram_total = get_system_ram_usage()
    ram_ratio = ram_used / ram_total if ram_total > 0 else 0.0
    vram_used, vram_total = get_gpu_vram_usage()
    if vram_used is None:
        vram_ratio = 0.0
    else:
        vram_ratio = vram_used / vram_total if vram_total else 0.0

    game_active = 1.0 if game_metrics.get("active", 0.0) > 0 else 0.0
    fps_norm = game_metrics.get("fps_norm", 0.5)
    latency_norm = game_metrics.get("latency_norm", 0.5)

    features = [
        cpu,
        ram_ratio,
        vram_ratio,
        game_active,
        fps_norm,
        latency_norm,
    ]

    features.append(1.0 if role == "guardian" else 0.0)
    features.append(1.0 if role == "oracle" else 0.0)
    features.append(1.0 if role == "combat" else 0.0)
    features.append(1.0 if role == "companion" else 0.0)

    return features

# ============================================================
#  PREDICTION BUS
# ============================================================

class PredictionBus:
    def __init__(self):
        self.current_risk = 0.0
        self.turbulence = 0.0
        self.regime_shift_prob = 0.0

    def update(self, signal: float):
        self.current_risk = min(1.0, max(0.0, 0.6 * signal + 0.4 * self.current_risk))
        self.turbulence = min(1.0, max(0.0, abs(signal - self.current_risk)))
        self.regime_shift_prob = min(1.0, max(0.0, 0.5 * self.regime_shift_prob + 0.5 * self.turbulence))

# ============================================================
#  ORGANS
# ============================================================

class Organ:
    def __init__(self, name):
        self.name = name
        self.health = 1.0
        self.activity = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        self.activity = min(1.0, max(0.0, 0.7 * self.activity + 0.3 * drive))
        self.health = max(0.0, min(1.0, self.health + random.uniform(-0.003, 0.003)))


class DeepRamOrgan(Organ):
    def __init__(self):
        super().__init__("DeepRAM")
        self.memory_pressure = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.memory_pressure = telemetry.get("ram_ratio", 0.0)


class BackupEngineOrgan(Organ):
    def __init__(self):
        super().__init__("BackupEngine")
        self.last_backup_age = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.last_backup_age = min(1.0, self.last_backup_age + 0.01)


class NetworkWatcherOrgan(Organ):
    def __init__(self):
        super().__init__("NetworkWatcher")
        self.sent = 0
        self.recv = 0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.sent = telemetry.get("net_sent", 0.0)
        self.recv = telemetry.get("net_recv", 0.0)


class GPUCacheOrgan(Organ):
    def __init__(self):
        super().__init__("GPUCache")
        self.vram_ratio = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.vram_ratio = telemetry.get("vram_ratio", 0.0)


class ThermalOrgan(Organ):
    def __init__(self):
        super().__init__("Thermal")
        self.gpu_temp = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.gpu_temp = telemetry.get("gpu_temp_norm", 0.0)


class DiskOrgan(Organ):
    def __init__(self):
        super().__init__("Disk")
        self.read_bytes = 0
        self.write_bytes = 0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.read_bytes = telemetry.get("disk_read", 0.0)
        self.write_bytes = telemetry.get("disk_write", 0.0)


class VRAMOrgan(Organ):
    def __init__(self):
        super().__init__("VRAM")
        self.vram_ratio = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.vram_ratio = telemetry.get("vram_ratio", 0.0)


class AICoachOrgan(Organ):
    def __init__(self):
        super().__init__("AICoach")
        self.coaching_level = 0.5

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.coaching_level = min(1.0, max(0.0, 0.9 * self.coaching_level + 0.1 * drive))


class SwarmNodeOrgan(Organ):
    def __init__(self):
        super().__init__("SwarmNode")
        self.swarm_confidence = 0.5

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.swarm_confidence = min(1.0, max(0.0, 0.9 * self.swarm_confidence + 0.1 * drive))


class Back4BloodAnalyzer(Organ):
    def __init__(self):
        super().__init__("Back4BloodAnalyzer")
        self.game_active = 0.0
        self.fps_norm = 0.5
        self.latency_norm = 0.5
        self.risk = 0.0

    def update(self, drive: float, telemetry: Dict[str, float]):
        super().update(drive, telemetry)
        self.game_active = telemetry.get("game_active", 0.0)
        self.fps_norm = telemetry.get("fps_norm", 0.5)
        self.latency_norm = telemetry.get("latency_norm", 0.5)
        self.risk = min(1.0, max(0.0, 1.0 - self.fps_norm + self.latency_norm * 0.5))

# ============================================================
#  HYBRID BRAIN
# ============================================================

class HybridBrain:
    def __init__(self):
        self.primary_role = PRIMARY_ROLE
        self.secondary_role = SECONDARY_ROLE

        self.meta_state = "Idle"
        self.stance = "Balanced"
        self.model_integrity = 1.0

        self.meta_conf = 0.5
        self.reinforcement = 0.5

        self.mood_calm = 1.0
        self.mood_confident = 0.5
        self.mood_curious = 0.5

        self.fingerprint = {}
        self.mode_profiles = {
            "guardian": {"risk_bias": 0.6, "turb_sensitivity": 0.4},
            "oracle": {"risk_bias": 0.4, "turb_sensitivity": 0.6},
            "combat": {"risk_bias": 0.8, "turb_sensitivity": 0.7},
            "companion": {"risk_bias": 0.3, "turb_sensitivity": 0.3},
        }

        self.last_predictions = {
            "short": 0.5,
            "medium": 0.5,
            "long": 0.5,
            "baseline": 0.5,
            "best_guess": 0.5,
        }
        self.last_reasoning = deque(maxlen=80)
        self.last_heatmap = {
            "short": 0.5,
            "medium": 0.5,
            "long": 0.5,
            "turbulence": 0.2,
            "regime_shift": 0.1,
            "meta_conf": 0.5,
            "best_guess_contributors": {
                "short": 0.5,
                "medium": 0.3,
                "long": 0.2,
                "weights": {
                    "short": 0.5,
                    "medium": 0.3,
                    "long": 0.2,
                }
            }
        }

        self.persistent_memory = {
            "sessions": 0,
            "last_game_state": "Idle",
            "avg_risk": 0.0,
        }

    def update_mood(self, risk, turb, integrity):
        self.mood_calm = max(0.0, min(1.0, 1.0 - (risk * 0.5 + turb * 0.5)))
        self.mood_confident = max(0.0, min(1.0, integrity))
        self.mood_curious = max(0.0, min(1.0, 0.5 + turb * 0.5))

    def update_meta_state(self, risk, turb, game_active):
        if game_active < 0.5 and risk < 0.3:
            self.meta_state = "Idle"
        elif game_active >= 0.5 and risk < 0.4:
            self.meta_state = "Scanning"
        elif risk < 0.7:
            self.meta_state = "Engaged"
        else:
            self.meta_state = "Recovering"

    def update_stance(self, risk):
        if risk < 0.3:
            self.stance = "Conservative"
        elif risk < 0.7:
            self.stance = "Balanced"
        else:
            self.stance = "Beast"

    def update_integrity(self, organs, turb, regime):
        penalty = 0.0
        for o in organs:
            if isinstance(o, DeepRamOrgan):
                penalty += o.memory_pressure * 0.2
            if isinstance(o, ThermalOrgan):
                penalty += max(0.0, o.gpu_temp - 0.7) * 0.3
        penalty += turb * 0.2 + regime * 0.3
        self.model_integrity = max(0.0, min(1.0, 1.0 - penalty))

    def update_meta_conf(self, preds, game_metrics):
        fps = game_metrics.get("fps_norm", 0.5)
        latency = game_metrics.get("latency_norm", 0.5)
        spread = abs(preds["short"] - preds["long"])
        self.meta_conf = max(0.0, min(1.0, 0.7 * fps + 0.3 * (1.0 - spread) - 0.2 * latency))

    def update_reinforcement(self, preds, game_metrics):
        fps = game_metrics.get("fps_norm", 0.5)
        self.reinforcement = max(0.0, min(1.0, 0.9 * self.reinforcement + 0.1 * fps))

    def update_fingerprint(self, telemetry, game_metrics):
        self.fingerprint = {
            "cpu": telemetry.get("cpu", 0.0),
            "ram": telemetry.get("ram_ratio", 0.0),
            "vram": telemetry.get("vram_ratio", 0.0),
            "game_active": game_metrics.get("active", 0.0),
            "fps_norm": game_metrics.get("fps_norm", 0.5),
            "latency_norm": game_metrics.get("latency_norm", 0.5),
        }

    def update_persistent_memory(self, risk):
        self.persistent_memory["sessions"] += 1
        self.persistent_memory["avg_risk"] = (
            0.9 * self.persistent_memory["avg_risk"] + 0.1 * risk
        )

    def update(self, prediction_bus: PredictionBus, npu_preds: Dict[str, float],
               organs: List[Organ], telemetry: Dict[str, float],
               game_metrics: Dict[str, float]):

        risk = prediction_bus.current_risk
        turb = prediction_bus.turbulence
        regime = prediction_bus.regime_shift_prob
        game_active = game_metrics.get("active", 0.0)

        self.last_predictions = npu_preds

        self.update_meta_state(risk, turb, game_active)
        self.update_stance(risk)
        self.update_integrity(organs, turb, regime)
        self.update_meta_conf(npu_preds, game_metrics)
        self.update_reinforcement(npu_preds, game_metrics)
        self.update_mood(risk, turb, self.model_integrity)
        self.update_fingerprint(telemetry, game_metrics)
        self.update_persistent_memory(risk)

        self.last_heatmap["short"] = npu_preds["short"]
        self.last_heatmap["medium"] = npu_preds["medium"]
        self.last_heatmap["long"] = npu_preds["long"]
        self.last_heatmap["turbulence"] = turb
        self.last_heatmap["regime_shift"] = regime
        self.last_heatmap["meta_conf"] = self.meta_conf

        contrib = {
            "short": 0.5,
            "medium": 0.3,
            "long": 0.2,
            "weights": {
                "short": 0.5,
                "medium": 0.3,
                "long": 0.2,
            }
        }
        self.last_heatmap["best_guess_contributors"] = contrib

        mood_str = f"calm={self.mood_calm:.2f}, conf={self.mood_confident:.2f}, curious={self.mood_curious:.2f}"
        self.last_reasoning.appendleft(
            f"[{ROLE_LABELS.get(self.primary_role, self.primary_role)}] "
            f"Meta={self.meta_state}, Stance={self.stance}, "
            f"Risk={risk:.2f}, Turb={turb:.2f}, Regime={regime:.2f}, "
            f"Integrity={self.model_integrity:.2f}, MetaConf={self.meta_conf:.2f}, "
            f"Reinf={self.reinforcement:.2f}, Mood({mood_str})"
        )

# ============================================================
#  NPU WITH INFER INTERFACE + CACHE
# ============================================================

class ReplicaNPU_Extended:
    def __init__(self, cores=16, frequency_ghz=1.5,
                 l1_size=128, l2_size=512, l3_size=2048):

        self.cores = cores
        self.frequency_ghz = frequency_ghz

        self.cycles = 0
        self.energy = 0.0

        self.l1 = OrderedDict()
        self.l2 = OrderedDict()
        self.l3 = OrderedDict()
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size

        self.cache_hits = {"L1": 0, "L2": 0, "L3": 0}
        self.cache_misses = {"L1": 0, "L2": 0, "L3": 0}

        self.short = 0.5
        self.medium = 0.5
        self.long = 0.5
        self.baseline = 0.5
        self.best_guess = 0.5

    def _cache_get(self, key):
        if key in self.l1:
            self.cache_hits["L1"] += 1
            val = self.l1.pop(key)
            self.l1[key] = val
            return val
        self.cache_misses["L1"] += 1

        if key in self.l2:
            self.cache_hits["L2"] += 1
            val = self.l2.pop(key)
            self._cache_put(self.l1, self.l1_size, key, val)
            return val
        self.cache_misses["L2"] += 1

        if key in self.l3:
            self.cache_hits["L3"] += 1
            val = self.l3.pop(key)
            self._cache_put(self.l2, self.l2_size, key, val)
            return val
        self.cache_misses["L3"] += 1

        return None

    def _cache_put(self, cache, max_size, key, value):
        if key in cache:
            cache.pop(key)
        cache[key] = value
        if len(cache) > max_size:
            cache.popitem(last=False)

    def cache_store(self, key, value):
        self._cache_put(self.l3, self.l3_size, key, value)

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        chunk = math.ceil(len(v1) / self.cores)
        total = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                key = ("mac", v1[j], v2[j])
                cached = self._cache_get(key)
                if cached is not None:
                    partial += cached
                else:
                    val = self.mac(v1[j], v2[j])
                    self.cache_store(key, val)
                    partial += val
            total += partial
        return total

    def infer(self, features: List[float]) -> Dict[str, float]:
        signal = sum(features) / max(1, len(features))
        noise = random.uniform(-0.05, 0.05)
        signal = max(0.0, min(1.0, signal + noise))

        self.short = 0.6 * signal + 0.4 * self.short
        self.medium = 0.3 * signal + 0.7 * self.medium
        self.long = 0.1 * signal + 0.9 * self.long
        self.baseline = 0.5 * self.baseline + 0.25
        best = (
            0.5 * self.short +
            0.3 * self.medium +
            0.2 * self.long +
            random.uniform(-0.02, 0.02)
        )
        self.best_guess = max(0, min(1, best))
        return {
            "short": self.short,
            "medium": self.medium,
            "long": self.long,
            "baseline": self.baseline,
            "best_guess": self.best_guess
        }

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        hits = sum(self.cache_hits.values())
        misses = sum(self.cache_misses.values())
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "energy": self.energy,
            "time": time_sec,
            "hit_rate": hit_rate,
            "L1": (self.cache_hits["L1"], self.cache_misses["L1"]),
            "L2": (self.cache_hits["L2"], self.cache_misses["L2"]),
            "L3": (self.cache_hits["L3"], self.cache_misses["L3"]),
        }

# ============================================================
#  PANELS / WIDGETS (NEON VISUALS)
# ============================================================

class PredictionChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pred = {
            "short": 0.5,
            "medium": 0.5,
            "long": 0.5,
            "baseline": 0.5,
            "best_guess": 0.5
        }

    def update_pred(self, p):
        self.pred = p
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        # Gradient background
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor("#050510"))
        grad.setColorAt(0.5, QColor("#101020"))
        grad.setColorAt(1.0, QColor("#050510"))
        p.fillRect(self.rect(), grad)

        def y(v): return h - int(v * (h - 20)) - 10

        xs = int(w * 0.2)
        xm = int(w * 0.5)
        xl = int(w * 0.8)

        ys = y(self.pred["short"])
        ym = y(self.pred["medium"])
        yl = y(self.pred["long"])
        yb = y(self.pred["baseline"])
        yg = y(self.pred["best_guess"])

        # Baseline (glow)
        pen = QPen(QColor("#555555"))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, yb, w, yb)

        # Cyan prediction path with glow
        glow_pen = QPen(QColor(0, 204, 255, 80))
        glow_pen.setWidth(8)
        p.setPen(glow_pen)
        p.drawLine(xs, ys, xm, ym)
        p.drawLine(xm, ym, xl, yl)

        pen = QPen(QColor("#00ccff"))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(xs, ys, xm, ym)
        p.drawLine(xm, ym, xl, yl)

        # Best guess magenta line with glow
        glow_pen = QPen(QColor(255, 0, 255, 80))
        glow_pen.setWidth(8)
        p.setPen(glow_pen)
        p.drawLine(0, yg, w, yg)

        pen = QPen(QColor("#ff00ff"))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(0, yg, w, yg)

        # Text
        p.setPen(QColor("#aaaaaa"))
        p.setFont(QFont("Consolas", 9))
        p.drawText(
            QRectF(8, 6, w - 16, 40),
            Qt.AlignLeft | Qt.AlignTop,
            "Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)",
        )


class HeatmapWidget(QWidget):
    def __init__(self, brain: HybridBrain, prediction_bus: PredictionBus, parent=None):
        super().__init__(parent)
        self.brain = brain
        self.prediction_bus = prediction_bus

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        # Gradient background
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#050510"))
        grad.setColorAt(1.0, QColor("#101020"))
        p.fillRect(self.rect(), grad)

        items = [
            ("short", self.brain.last_heatmap.get("short", 0.0), QColor("#00ccff")),
            ("medium", self.brain.last_heatmap.get("medium", 0.0), QColor("#00aa88")),
            ("long", self.brain.last_heatmap.get("long", 0.0), QColor("#ffaa00")),
            ("turb", self.prediction_bus.turbulence, QColor("#ff6666")),
            ("regime", self.prediction_bus.regime_shift_prob, QColor("#ff00ff")),
            ("meta_conf", self.brain.meta_conf, QColor("#66ff66")),
        ]

        bar_height = h // (len(items) + 1)
        margin = 12
        max_width = w - 2 * margin

        p.setFont(QFont("Consolas", 9))

        for i, (name, val, color) in enumerate(items):
            y = margin + i * bar_height
            v = max(0.0, min(1.0, val))
            width = int(max_width * v)

            # 3D bar: shadow
            shadow_color = QColor(0, 0, 0, 120)
            p.setPen(Qt.NoPen)
            p.setBrush(shadow_color)
            p.drawRoundedRect(margin + 2, y + 4, width, bar_height - 4, 4, 4)

            # 3D bar: main
            grad = QLinearGradient(margin, y, margin, y + bar_height)
            grad.setColorAt(0.0, color.lighter(140))
            grad.setColorAt(0.5, color)
            grad.setColorAt(1.0, color.darker(160))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(margin, y, width, bar_height - 6, 4, 4)

            # Label
            p.setPen(QColor("#ffffff"))
            p.drawText(margin + 6, y + bar_height // 2 + 4, f"{name}: {v:.2f}")


class BrainCortexPanel(QWidget):
    def __init__(self, brain: HybridBrain, prediction_bus: PredictionBus,
                 organs: List[Organ], npu: ReplicaNPU_Extended, parent=None):
        super().__init__(parent)
        self.brain = brain
        self.prediction_bus = prediction_bus
        self.organs = organs
        self.npu = npu

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top info grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        layout.addLayout(grid)

        self.lbl_role = QLabel()
        self.lbl_meta = QLabel()
        self.lbl_stance = QLabel()
        self.lbl_meta_conf = QLabel()
        self.lbl_integrity = QLabel()
        self.lbl_risk = QLabel()

        for lbl in [self.lbl_role, self.lbl_meta, self.lbl_stance,
                    self.lbl_meta_conf, self.lbl_integrity, self.lbl_risk]:
            lbl.setStyleSheet("color: #e0e0ff; font-weight: bold;")

        row = 0
        grid.addWidget(QLabel("Role:"), row, 0)
        grid.addWidget(self.lbl_role, row, 1)
        grid.addWidget(QLabel("Meta-State:"), row, 2)
        grid.addWidget(self.lbl_meta, row, 3)

        row += 1
        grid.addWidget(QLabel("Stance:"), row, 0)
        grid.addWidget(self.lbl_stance, row, 1)
        grid.addWidget(QLabel("Meta-Confidence:"), row, 2)
        grid.addWidget(self.lbl_meta_conf, row, 3)

        row += 1
        grid.addWidget(QLabel("Model Integrity:"), row, 0)
        grid.addWidget(self.lbl_integrity, row, 1)
        grid.addWidget(QLabel("Current Risk:"), row, 2)
        grid.addWidget(self.lbl_risk, row, 3)

        # Organs summary
        self.lbl_deep_ram = QLabel()
        self.lbl_backup = QLabel()
        self.lbl_net = QLabel()
        self.lbl_gpu = QLabel()
        self.lbl_thermal = QLabel()
        self.lbl_disk = QLabel()
        self.lbl_vram = QLabel()
        self.lbl_swarm = QLabel()
        self.lbl_coach = QLabel()

        organ_labels = [
            self.lbl_deep_ram, self.lbl_backup, self.lbl_net, self.lbl_gpu,
            self.lbl_thermal, self.lbl_disk, self.lbl_vram, self.lbl_swarm,
            self.lbl_coach
        ]
        for lbl in organ_labels:
            lbl.setStyleSheet("color: #c0ffc0;")

        row += 1
        grid.addWidget(QLabel("Deep RAM:"), row, 0)
        grid.addWidget(self.lbl_deep_ram, row, 1)
        grid.addWidget(QLabel("Backup:"), row, 2)
        grid.addWidget(self.lbl_backup, row, 3)

        row += 1
        grid.addWidget(QLabel("Network:"), row, 0)
        grid.addWidget(self.lbl_net, row, 1)
        grid.addWidget(QLabel("GPU Cache:"), row, 2)
        grid.addWidget(self.lbl_gpu, row, 3)

        row += 1
        grid.addWidget(QLabel("Thermal:"), row, 0)
        grid.addWidget(self.lbl_thermal, row, 1)
        grid.addWidget(QLabel("Disk:"), row, 2)
        grid.addWidget(self.lbl_disk, row, 3)

        row += 1
        grid.addWidget(QLabel("VRAM:"), row, 0)
        grid.addWidget(self.lbl_vram, row, 1)
        grid.addWidget(QLabel("Swarm:"), row, 2)
        grid.addWidget(self.lbl_swarm, row, 3)

        row += 1
        grid.addWidget(QLabel("AI Coach:"), row, 0)
        grid.addWidget(self.lbl_coach, row, 1)

        # Prediction chart
        self.chart = PredictionChart()
        self.chart.setMinimumHeight(180)
        self.chart.setStyleSheet("border: 1px solid #333333;")
        layout.addWidget(self.chart)

        # Reasoning tail
        self.txt_reasoning = QTextEdit()
        self.txt_reasoning.setReadOnly(True)
        self.txt_reasoning.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 10pt;"
        )
        layout.addWidget(self.txt_reasoning, 2)

        # Save button
        self.btn_save_now = QPushButton("Save Memory Now")
        self.btn_save_now.setStyleSheet(
            "QPushButton { background-color: #303060; color: #ffffff; border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #404080; }"
        )
        layout.addWidget(self.btn_save_now)

        self.btn_save_now.clicked.connect(self.save_now)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)

    def save_now(self):
        # Hook for external state save if needed
        pass

    def refresh(self):
        self.lbl_role.setText(
            f"{ROLE_LABELS.get(self.brain.primary_role, self.brain.primary_role)}"
        )
        self.lbl_meta.setText(self.brain.meta_state)
        self.lbl_stance.setText(self.brain.stance)
        self.lbl_meta_conf.setText(f"{self.brain.meta_conf:.2f}")
        self.lbl_integrity.setText(f"{self.brain.model_integrity:.2f}")
        self.lbl_risk.setText(f"{self.prediction_bus.current_risk:.2f}")

        deep = next((o for o in self.organs if isinstance(o, DeepRamOrgan)), None)
        backup = next((o for o in self.organs if isinstance(o, BackupEngineOrgan)), None)
        net = next((o for o in self.organs if isinstance(o, NetworkWatcherOrgan)), None)
        gpu = next((o for o in self.organs if isinstance(o, GPUCacheOrgan)), None)
        therm = next((o for o in self.organs if isinstance(o, ThermalOrgan)), None)
        disk = next((o for o in self.organs if isinstance(o, DiskOrgan)), None)
        vram = next((o for o in self.organs if isinstance(o, VRAMOrgan)), None)
        swarm = next((o for o in self.organs if isinstance(o, SwarmNodeOrgan)), None)
        coach = next((o for o in self.organs if isinstance(o, AICoachOrgan)), None)

        self.lbl_deep_ram.setText(
            f"health={deep.health:.2f}, memP={deep.memory_pressure:.2f}" if deep else "N/A"
        )
        self.lbl_backup.setText(
            f"health={backup.health:.2f}, age={backup.last_backup_age:.2f}" if backup else "N/A"
        )
        self.lbl_net.setText(
            f"health={net.health:.2f}, sent={net.sent:.0f}, recv={net.recv:.0f}" if net else "N/A"
        )
        self.lbl_gpu.setText(
            f"health={gpu.health:.2f}, vram={gpu.vram_ratio:.2f}" if gpu else "N/A"
        )
        self.lbl_thermal.setText(
            f"health={therm.health:.2f}, tempN={therm.gpu_temp:.2f}" if therm else "N/A"
        )
        self.lbl_disk.setText(
            f"health={disk.health:.2f}, R={disk.read_bytes}, W={disk.write_bytes}" if disk else "N/A"
        )
        self.lbl_vram.setText(
            f"health={vram.health:.2f}, vram={vram.vram_ratio:.2f}" if vram else "N/A"
        )
        self.lbl_swarm.setText(
            f"health={swarm.health:.2f}, conf={swarm.swarm_confidence:.2f}" if swarm else "N/A"
        )
        self.lbl_coach.setText(
            f"health={coach.health:.2f}, coach={coach.coaching_level:.2f}" if coach else "N/A"
        )

        self.chart.update_pred(self.brain.last_predictions)

        self.txt_reasoning.setPlainText(
            "Reasoning Tail:\n" + "\n".join(f"  - {r}" for r in self.brain.last_reasoning)
        )

# ============================================================
#  TIMELINE
# ============================================================

class TimelineWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 9pt;"
        )

    def add_event(self, text):
        self.append(text)

# ============================================================
#  PERSISTENCE
# ============================================================

def load_state(brain: HybridBrain, organs: List[Organ]):
    if not os.path.isfile(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        brain.meta_state = data.get("meta_state", brain.meta_state)
        brain.stance = data.get("stance", brain.stance)
        brain.model_integrity = data.get("model_integrity", brain.model_integrity)
        brain.primary_role = data.get("primary_role", brain.primary_role)
        brain.secondary_role = data.get("secondary_role", brain.secondary_role)
        brain.persistent_memory = data.get("persistent_memory", brain.persistent_memory)
        organ_states = data.get("organs", {})
        for o in organs:
            st = organ_states.get(o.name, {})
            o.health = st.get("health", o.health)
    except Exception as e:
        print(f"[STATE] Failed to load state: {e}")


def save_state(brain: HybridBrain, organs: List[Organ]):
    data = {
        "meta_state": brain.meta_state,
        "stance": brain.stance,
        "model_integrity": brain.model_integrity,
        "primary_role": brain.primary_role,
        "secondary_role": brain.secondary_role,
        "persistent_memory": brain.persistent_memory,
        "organs": {},
    }
    for o in organs:
        st = {"health": o.health}
        data["organs"][o.name] = st
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[STATE] Failed to save state: {e}")

# ============================================================
#  MAIN GUI / NERVE CENTER
# ============================================================

class NeuralGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HybridBrain Back4Blood Analyzer — Neon Cortex")
        self.setMinimumSize(1700, 950)

        self.npu = ReplicaNPU_Extended()
        self.prediction_bus = PredictionBus()
        self.brain = HybridBrain()
        self.organs: List[Organ] = [
            DeepRamOrgan(),
            BackupEngineOrgan(),
            NetworkWatcherOrgan(),
            GPUCacheOrgan(),
            ThermalOrgan(),
            DiskOrgan(),
            VRAMOrgan(),
            AICoachOrgan(),
            SwarmNodeOrgan(),
            Back4BloodAnalyzer(),
        ]

        load_state(self.brain, self.organs)

        self.timeline_events = 0

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT: Brain Cortex Panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        self.cortex_panel = BrainCortexPanel(self.brain, self.prediction_bus, self.organs, self.npu)
        self.cortex_panel.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            "stop:0 #050510, stop:0.5 #101020, stop:1 #050510); "
            "border: 1px solid #333333; border-radius: 6px;"
        )
        left_layout.addWidget(self.cortex_panel, 3)

        self.lst_organs = QListWidget()
        self.lst_organs.setStyleSheet(
            "QListWidget { background-color: #050510; color: #e0e0ff; font-family: Consolas; }"
            "QListWidget::item { padding: 2px 4px; }"
            "QListWidget::item:selected { background-color: #303060; }"
        )
        left_layout.addWidget(self.lst_organs, 2)

        splitter.addWidget(left_widget)

        # RIGHT: Tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333333; }"
            "QTabBar::tab { background: #101020; color: #e0e0ff; padding: 4px 10px; }"
            "QTabBar::tab:selected { background: #303060; }"
        )
        right_layout.addWidget(tabs)

        # Heatmap
        heatmap_container = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_container)
        self.heatmap_widget = HeatmapWidget(self.brain, self.prediction_bus)
        self.heatmap_widget.setMinimumHeight(220)
        self.heatmap_widget.setStyleSheet("border: 1px solid #333333; border-radius: 6px;")
        heatmap_layout.addWidget(self.heatmap_widget)
        tabs.addTab(heatmap_container, "Heatmap")

        # Logs
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 9pt;"
        )
        tabs.addTab(self.txt_logs, "Logs")

        # Game State
        game_container = QWidget()
        game_layout = QVBoxLayout(game_container)
        self.lbl_game = QLabel("Game: Not detected")
        self.lbl_game_state = QLabel("State: Idle")
        self.lbl_game_risk = QLabel("Game Risk: 0.00")
        for w in (self.lbl_game, self.lbl_game_state, self.lbl_game_risk):
            w.setStyleSheet("font-size: 14px; color: #ccccff;")
            game_layout.addWidget(w)
        tabs.addTab(game_container, "Back4Blood")

        # System Control
        sys_container = QWidget()
        sys_layout = QVBoxLayout(sys_container)
        lbl_sys = QLabel("System Control (read-only views):")
        lbl_sys.setStyleSheet("color: #e0e0ff;")
        sys_layout.addWidget(lbl_sys)

        self.btn_refresh_processes = QPushButton("Refresh Processes")
        self.btn_refresh_services = QPushButton("Refresh Services")
        for b in (self.btn_refresh_processes, self.btn_refresh_services):
            b.setStyleSheet(
                "QPushButton { background-color: #303060; color: #ffffff; border-radius: 4px; padding: 4px 10px; }"
                "QPushButton:hover { background-color: #404080; }"
            )

        self.txt_processes = QTextEdit()
        self.txt_processes.setReadOnly(True)
        self.txt_processes.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 8pt;"
        )
        self.txt_services = QTextEdit()
        self.txt_services.setReadOnly(True)
        self.txt_services.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 8pt;"
        )

        sys_layout.addWidget(self.btn_refresh_processes)
        sys_layout.addWidget(self.txt_processes, 2)
        sys_layout.addWidget(self.btn_refresh_services)
        sys_layout.addWidget(self.txt_services, 1)

        self.btn_refresh_processes.clicked.connect(self.refresh_processes)
        self.btn_refresh_services.clicked.connect(self.refresh_services)

        tabs.addTab(sys_container, "System")

        # Command Console
        cmd_container = QWidget()
        cmd_layout = QVBoxLayout(cmd_container)
        self.txt_cmd_output = QTextEdit()
        self.txt_cmd_output.setReadOnly(True)
        self.txt_cmd_output.setStyleSheet(
            "background-color: #050510; color: #e0e0ff; font-family: Consolas; font-size: 9pt;"
        )
        self.txt_cmd_input = QLineEdit()
        self.txt_cmd_input.setPlaceholderText(
            "Commands: set role guardian/oracle/combat/companion | set stance Conservative/Balanced/Beast | dump state | dump organs"
        )
        self.txt_cmd_input.setStyleSheet(
            "background-color: #101020; color: #e0e0ff; border: 1px solid #333333; padding: 3px;"
        )
        self.btn_cmd_exec = QPushButton("Execute")
        self.btn_cmd_exec.setStyleSheet(
            "QPushButton { background-color: #303060; color: #ffffff; border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #404080; }"
        )
        cmd_layout.addWidget(self.txt_cmd_output, 3)
        cmd_layout.addWidget(self.txt_cmd_input, 0)
        cmd_layout.addWidget(self.btn_cmd_exec, 0)
        self.btn_cmd_exec.clicked.connect(self.execute_command)
        self.txt_cmd_input.returnPressed.connect(self.execute_command)
        tabs.addTab(cmd_container, "Command")

        # Timeline
        self.timeline = TimelineWidget()
        tabs.addTab(self.timeline, "Timeline")

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Timers
        self.timer_update = QTimer(self)
        self.timer_update.timeout.connect(self.nerve_center_tick)
        self.timer_update.start(800)

        self.refresh_organs_list()
        self.refresh_processes()
        self.refresh_services()

    # --------------------------------------------------------
    # System control
    # --------------------------------------------------------
    def refresh_processes(self):
        procs = list_processes()
        self.txt_processes.setPlainText("\n".join(procs))

    def refresh_services(self):
        services = list_services()
        if not services:
            self.txt_services.setPlainText("No service info available on this platform.")
        else:
            self.txt_services.setPlainText("\n".join(services))

    # --------------------------------------------------------
    # Command console
    # --------------------------------------------------------
    def execute_command(self):
        cmd = self.txt_cmd_input.text().strip()
        if not cmd:
            return
        self.txt_cmd_input.clear()
        self.txt_cmd_output.append(f"> {cmd}")

        parts = cmd.split()
        if parts[0] == "set" and len(parts) >= 3:
            if parts[1] == "role":
                role = parts[2].lower()
                if role in ROLE_LABELS:
                    self.brain.primary_role = role
                    self.txt_cmd_output.append(f"  primary_role set to {role}")
                else:
                    self.txt_cmd_output.append("  unknown role")
            elif parts[1] == "stance":
                stance = parts[2].capitalize()
                if stance in ["Conservative", "Balanced", "Beast"]:
                    self.brain.stance = stance
                    self.txt_cmd_output.append(f"  stance set to {stance}")
                else:
                    self.txt_cmd_output.append("  unknown stance")
            else:
                self.txt_cmd_output.append("  unknown set target")
        elif cmd == "dump state":
            self.txt_cmd_output.append(
                f"  meta_state={self.brain.meta_state}, stance={self.brain.stance}, "
                f"integrity={self.brain.model_integrity:.2f}, role={self.brain.primary_role}, "
                f"meta_conf={self.brain.meta_conf:.2f}, reinforcement={self.brain.reinforcement:.2f}"
            )
        elif cmd == "dump organs":
            for o in self.organs:
                self.txt_cmd_output.append(f"  {o.name}: health={o.health:.2f}, activity={o.activity:.2f}")
        else:
            self.txt_cmd_output.append("  unknown command")

    # --------------------------------------------------------
    # Organs list
    # --------------------------------------------------------
    def refresh_organs_list(self):
        self.lst_organs.clear()
        for o in self.organs:
            icon = ORGAN_ICONS.get(o.name, "⚙️")
            extra = ""
            if isinstance(o, DeepRamOrgan):
                extra = f" | memP={o.memory_pressure:.2f}"
            if isinstance(o, BackupEngineOrgan):
                extra = f" | age={o.last_backup_age:.2f}"
            if isinstance(o, NetworkWatcherOrgan):
                extra = f" | sent={o.sent:.0f}, recv={o.recv:.0f}"
            if isinstance(o, GPUCacheOrgan):
                extra = f" | vram={o.vram_ratio:.2f}"
            if isinstance(o, ThermalOrgan):
                extra = f" | tempN={o.gpu_temp:.2f}"
            if isinstance(o, DiskOrgan):
                extra = f" | R={o.read_bytes}, W={o.write_bytes}"
            if isinstance(o, VRAMOrgan):
                extra = f" | vram={o.vram_ratio:.2f}"
            if isinstance(o, AICoachOrgan):
                extra = f" | coach={o.coaching_level:.2f}"
            if isinstance(o, SwarmNodeOrgan):
                extra = f" | conf={o.swarm_confidence:.2f}"
            if isinstance(o, Back4BloodAnalyzer):
                extra = f" | active={o.game_active:.2f}, fps={o.fps_norm:.2f}, lat={o.latency_norm:.2f}, risk={o.risk:.2f}"
            item = QListWidgetItem(f"{icon} {o.name} | health={o.health:.2f} | activity={o.activity:.2f}{extra}")
            self.lst_organs.addItem(item)

    # --------------------------------------------------------
    # Nerve center main loop
    # --------------------------------------------------------
    def nerve_center_tick(self):
        # Telemetry snapshot
        cpu = get_cpu_usage() / 100.0
        ram_used, ram_total = get_system_ram_usage()
        ram_ratio = ram_used / ram_total if ram_total > 0 else 0.0
        vram_used, vram_total = get_gpu_vram_usage()
        if vram_used is None:
            vram_ratio = 0.0
        else:
            vram_ratio = vram_used / vram_total if vram_total else 0.0
        gpu_temp = get_gpu_temp()
        if gpu_temp is None:
            gpu_temp_norm = 0.5
        else:
            gpu_temp_norm = max(0.0, min(1.0, (gpu_temp - 30.0) / 60.0))

        disk_read, disk_write = get_disk_io()
        net_sent, net_recv = get_net_io()

        game_name = detect_game()
        game_active = 1.0 if game_name else 0.0

        # Approximate FPS/latency from load
        fps_norm = max(0.0, min(1.0, 1.0 - (cpu * 0.5 + vram_ratio * 0.5)))
        latency_norm = max(0.0, min(1.0, (cpu * 0.3 + vram_ratio * 0.7)))

        telemetry = {
            "cpu": cpu,
            "ram_ratio": ram_ratio,
            "vram_ratio": vram_ratio,
            "gpu_temp_norm": gpu_temp_norm,
            "disk_read": disk_read,
            "disk_write": disk_write,
            "net_sent": net_sent,
            "net_recv": net_recv,
        }

        game_metrics = {
            "active": game_active,
            "fps_norm": fps_norm,
            "latency_norm": latency_norm,
        }

        # NPU inference
        features = build_feature_vector(self.brain.primary_role, self.brain.secondary_role, game_metrics)
        preds = self.npu.infer(features)

        # Prediction bus
        self.prediction_bus.update(preds["best_guess"])

        # Organs update
        for o in self.organs:
            o.update(preds["best_guess"], {**telemetry, **game_metrics})

        # Brain update
        self.brain.update(self.prediction_bus, preds, self.organs, telemetry, game_metrics)

        # GUI updates
        self.cortex_panel.chart.update_pred(preds)
        self.heatmap_widget.update()
        self.refresh_organs_list()

        self.txt_logs.setPlainText(
            "Logs:\n"
            f"  CPU={cpu*100:.1f}%, RAM={ram_ratio:.2f}, VRAM={vram_ratio:.2f}, "
            f"GPUtempN={gpu_temp_norm:.2f}\n"
            f"  GameActive={game_active:.1f}, FPSn={fps_norm:.2f}, Latn={latency_norm:.2f}\n"
            f"  Risk={self.prediction_bus.current_risk:.2f}, Turb={self.prediction_bus.turbulence:.2f}, "
            f"Regime={self.prediction_bus.regime_shift_prob:.2f}\n"
        )

        if game_name:
            self.lbl_game.setText(f"Game: {game_name}")
            self.lbl_game_state.setText(self.brain.meta_state)
            b4b = next((o for o in self.organs if isinstance(o, Back4BloodAnalyzer)), None)
            if b4b:
                self.lbl_game_risk.setText(f"Game Risk: {b4b.risk:.2f}")
            else:
                self.lbl_game_risk.setText(f"Game Risk: {self.prediction_bus.current_risk:.2f}")
        else:
            self.lbl_game.setText("Game: Not detected")
            self.lbl_game_state.setText("State: Idle")
            self.lbl_game_risk.setText("Game Risk: 0.00")

        self.timeline_events += 1
        if self.timeline_events % 10 == 0:
            self.timeline.add_event(
                f"[{time.strftime('%H:%M:%S')}] Meta={self.brain.meta_state}, "
                f"Stance={self.brain.stance}, Risk={self.prediction_bus.current_risk:.2f}, "
                f"Integrity={self.brain.model_integrity:.2f}, MetaConf={self.brain.meta_conf:.2f}"
            )

    # --------------------------------------------------------
    # Close event
    # --------------------------------------------------------
    def closeEvent(self, event):
        save_state(self.brain, self.organs)
        super().closeEvent(event)

# ============================================================
#  Entry point
# ============================================================

def main():
    app = QApplication(sys.argv)

    # Global dark / neon theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #050510;
        }
        QLabel {
            color: #e0e0ff;
        }
        QSplitter::handle {
            background-color: #202040;
        }
        QScrollBar:vertical {
            background: #050510;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #303060;
            min-height: 20px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)

    win = NeuralGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

