#!/usr/bin/env python3
"""
Hybrid Brain MagicBox Cockpit with Tabs & Persistence (Single File)
Layout: Option C — Four Tabs (Full Control Room)

Tabs:
1) Status
   - Status grid
   - Core metrics, stance, risk, bottlenecks, Movidius status
2) MagicBox
   - Tri-Stance descriptive panel
   - Deep RAM target display
   - Reinforcement summary (wins/losses)
3) Visuals
   - Health timeline graph
   - Predicted CPU/MEM/DISK load bars
   - Memory tier map (conceptual)
   - Data flow (water physics) visualization
4) Memory Backup
   - Local backup folder picker
   - SMB UNC path entry (anonymous)
   - Test SMB connection
   - Save Memory Now
   - Load Memory
   - Status + last save/load timestamps

Engine:
- HybridBrain
- TriStanceDecisionEngine (Conservative / Balanced / Beast)
- PredictionBus, DecisionLog, risk scoring, bottleneck detection
- Deep RAM target logic + per-source scaling hints
- SensorBus, CollectiveHealthScore
- PredictiveCortex (prediction + confidence + best-guess action)
- SituationalAwarenessCortex (altered states)
- DataPhysicsEngine (pressure + turbulence)
- MovidiusInferenceEngine stub

Persistence:
- settings.json
- brain_state.json
"""

import importlib
import sys
import time
import math
import threading
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

# =========================
# 1. AUTOLOADER
# =========================

REQUIRED_LIBS = [
    "psutil",
    "numpy",
    "scipy",
    "sklearn",
]

def autoload_libraries(required_libs: List[str]) -> Dict[str, Any]:
    modules = {}
    for name in required_libs:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            modules[name] = None
            print(f"[AUTOLOADER] Missing library: {name} (some features degraded)", file=sys.stderr)
    return modules

LIBS = autoload_libraries(REQUIRED_LIBS)
np = LIBS.get("numpy", None)
psutil = LIBS.get("psutil", None)
scipy = LIBS.get("scipy", None)
sklearn = LIBS.get("sklearn", None)

# =========================
# 2. ENUMS / CONSTANTS
# =========================

class AlteredState:
    NORMAL = "NORMAL"
    FLOW = "FLOW"
    ALERT = "ALERT"
    RECOVERY = "RECOVERY"
    DREAMING = "DREAMING"
    DEGRADED = "DEGRADED"


class TriStance:
    CONSERVATIVE = "A_CONSERVATIVE"
    BALANCED = "B_BALANCED"
    BEAST = "C_BEAST"


TRI_STANCE_DESCRIPTIONS = {
    TriStance.CONSERVATIVE: {
        "name": "A — Conservative",
        "lines": [
            "Protects system stability",
            "Shrinks Deep RAM early",
            "Avoids aggressive mode switching",
            "Prioritizes low-risk behavior",
        ],
        "color": "#fdf3f2",
        "border": "#d13438",
    },
    TriStance.BALANCED: {
        "name": "B — Balanced",
        "lines": [
            "Normal operating stance",
            "Predictive but not reckless",
            "Smooth ingestion scaling",
            "Moderate Deep RAM appetite",
        ],
        "color": "#f2f6fc",
        "border": "#0078d4",
    },
    TriStance.BEAST: {
        "name": "C — Beast",
        "lines": [
            "Maximum performance",
            "Aggressive VRAM/RAM usage",
            "High Deep RAM appetite",
            "Fast mode switching",
        ],
        "color": "#fff8e1",
        "border": "#ff8c00",
    },
}

SETTINGS_FILENAME = "settings.json"
BRAIN_STATE_FILENAME = "brain_state.json"

# =========================
# 3. SENSOR BUS + HEALTH
# =========================

class SystemSnapshot:
    def __init__(self,
                 timestamp: float,
                 cpu_usage: float,
                 mem_usage: float,
                 disk_usage: float,
                 net_activity: float,
                 gpu_usage: Optional[float],
                 temperature_score: float,
                 errors_recent: int,
                 altered_state: str):
        self.timestamp = timestamp
        self.cpu_usage = cpu_usage
        self.mem_usage = mem_usage
        self.disk_usage = disk_usage
        self.net_activity = net_activity
        self.gpu_usage = gpu_usage
        self.temperature_score = temperature_score
        self.errors_recent = errors_recent
        self.altered_state = altered_state


class SensorBus:
    def __init__(self):
        self._lock = threading.Lock()
        self.history: List[SystemSnapshot] = []

    def sample_system(self, altered_state: str) -> SystemSnapshot:
        now = time.time()

        if psutil is None:
            cpu = 0.2
            mem = 0.3
            disk = 0.1
            net = 0.05
        else:
            cpu = psutil.cpu_percent(interval=None) / 100.0
            mem = psutil.virtual_memory().percent / 100.0
            disk = psutil.disk_usage("/").percent / 100.0
            net_io = psutil.net_io_counters()
            net = min(1.0, (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024 * 1024))

        gpu_usage = None
        temperature_score = 0.5
        errors_recent = 0

        snap = SystemSnapshot(
            timestamp=now,
            cpu_usage=cpu,
            mem_usage=mem,
            disk_usage=disk,
            net_activity=net,
            gpu_usage=gpu_usage,
            temperature_score=temperature_score,
            errors_recent=errors_recent,
            altered_state=altered_state,
        )

        with self._lock:
            self.history.append(snap)
            if len(self.history) > 10000:
                self.history = self.history[-5000:]

        return snap


class CollectiveHealthScore:
    def __init__(self):
        self.last_score: float = 1.0

    def compute(self, snapshots: List[SystemSnapshot]) -> float:
        if not snapshots:
            self.last_score = 0.5
            return self.last_score

        recent = snapshots[-1]
        cpu = recent.cpu_usage
        mem = recent.mem_usage
        disk = recent.disk_usage
        temp = recent.temperature_score
        errors = recent.errors_recent

        stress = (cpu + mem + disk) / 3.0
        thermal_penalty = abs(temp - 0.5) * 2.0
        error_penalty = min(1.0, errors / 10.0)

        raw = 1.0 - (0.4 * stress + 0.3 * thermal_penalty + 0.3 * error_penalty)
        score = max(0.0, min(1.0, raw))
        self.last_score = score
        return score

# =========================
# 4. PREDICTIVE CORTEX
# =========================

class PredictiveCortex:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.last_predictions: Dict[str, Any] = {}

    def train_if_possible(self, history: List[SystemSnapshot]) -> None:
        if np is None or sklearn is None:
            return

    def predict_load(self, history: List[SystemSnapshot]) -> Dict[str, float]:
        with self._lock:
            if not history:
                pred = {"cpu": 0.3, "mem": 0.3, "disk": 0.3}
                self.last_predictions = pred
                return pred

            if len(history) < 3 or np is None:
                last = history[-1]
                pred = {
                    "cpu": last.cpu_usage,
                    "mem": last.mem_usage,
                    "disk": last.disk_usage,
                }
                self.last_predictions = pred
                return pred

            def linear_predict(vals: List[float]) -> float:
                n = len(vals)
                if n < 2:
                    return vals[-1]
                slope = (vals[-1] - vals[0]) / (n - 1)
                return max(0.0, min(1.0, vals[-1] + slope))

            recent = history[-10:]
            cpu_series = [s.cpu_usage for s in recent]
            mem_series = [s.mem_usage for s in recent]
            disk_series = [s.disk_usage for s in recent]

            pred = {
                "cpu": linear_predict(cpu_series),
                "mem": linear_predict(mem_series),
                "disk": linear_predict(disk_series),
            }
            self.last_predictions = pred
            return pred

    def best_guess_action(self,
                          health_score: float,
                          load_pred: Dict[str, float]) -> str:
        cpu = load_pred.get("cpu", 0.3)
        mem = load_pred.get("mem", 0.3)
        disk = load_pred.get("disk", 0.3)
        avg_load = (cpu + mem + disk) / 3.0

        if health_score < 0.3:
            if avg_load > 0.5:
                return "ENTER_RECOVERY_MODE"
            else:
                return "DIAGNOSE_AND_REPAIR"

        if health_score > 0.7 and avg_load < 0.4:
            return "PREFETCH_AND_PREPARE"

        if avg_load > 0.8:
            return "SHED_NONCRITICAL_LOAD"

        return "CONTINUE_AND_MONITOR"

    def confidence_score(self, history: List[SystemSnapshot]) -> float:
        if len(history) < 5 or np is None:
            return 0.5

        recent = history[-30:]
        cpu_vals = np.array([s.cpu_usage for s in recent])
        mem_vals = np.array([s.mem_usage for s in recent])

        cpu_var = float(np.var(cpu_vals))
        mem_var = float(np.var(mem_vals))

        stability = math.exp(- (cpu_var + mem_var))
        return max(0.0, min(1.0, stability))

# =========================
# 5. SITUATIONAL AWARENESS
# =========================

class SituationalAwarenessCortex:
    def __init__(self):
        self.current_state: str = AlteredState.NORMAL
        self.last_change_ts: float = time.time()

    def update_state(self,
                     health_score: float,
                     pred_load: Dict[str, float],
                     errors_recent: int) -> str:
        now = time.time()
        cpu = pred_load.get("cpu", 0.3)
        avg_load = (cpu + pred_load.get("mem", 0.3) + pred_load.get("disk", 0.3)) / 3.0

        new_state = self.current_state

        if health_score < 0.3:
            new_state = AlteredState.DEGRADED
        elif errors_recent > 0 and health_score < 0.6:
            new_state = AlteredState.ALERT
        elif avg_load > 0.8 and health_score > 0.5:
            new_state = AlteredState.FLOW
        elif avg_load < 0.2 and health_score > 0.7:
            new_state = AlteredState.DREAMING
        else:
            new_state = AlteredState.NORMAL

        if new_state != self.current_state:
            self.current_state = new_state
            self.last_change_ts = now

        return self.current_state

# =========================
# 6. DATA PHYSICS ENGINE
# =========================

class DataPhysicsEngine:
    def __init__(self):
        self.pressure_level: float = 0.0
        self.turbulence_level: float = 0.0

    def update_from_snapshot(self, snap: SystemSnapshot) -> None:
        avg_load = (snap.cpu_usage + snap.mem_usage + snap.disk_usage) / 3.0
        self.pressure_level = 0.7 * self.pressure_level + 0.3 * avg_load

        turbulence = (snap.disk_usage + snap.net_activity) / 2.0
        self.turbulence_level = 0.6 * self.turbulence_level + 0.4 * turbulence

    def qualitative_state(self) -> str:
        if self.pressure_level > 0.8 and self.turbulence_level > 0.5:
            return "RAPIDS"
        if self.pressure_level > 0.6:
            return "FAST_FLOW"
        if self.pressure_level < 0.3 and self.turbulence_level < 0.3:
            return "CALM"
        return "STEADY_FLOW"

# =========================
# 7. MOVIDIUS HOOK
# =========================

class MovidiusInferenceEngine:
    def __init__(self):
        self.available = False
        self.device = None
        self._detect_device()

    def _detect_device(self) -> None:
        self.available = False
        self.device = None

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return {"prediction": None, "confidence": 0.1, "note": "Movidius not available"}
        return {"prediction": None, "confidence": 0.9, "note": "Movidius placeholder"}

# =========================
# 8. TRI-STANCE ENGINE + SUPPORT
# =========================

class PredictionBus:
    def __init__(self,
                 snapshot: SystemSnapshot,
                 health_score: float,
                 predicted_load: Dict[str, float],
                 prediction_confidence: float,
                 flow_state: str,
                 pressure: float,
                 turbulence: float):
        self.snapshot = snapshot
        self.health_score = health_score
        self.predicted_load = predicted_load
        self.prediction_confidence = prediction_confidence
        self.flow_state = flow_state
        self.pressure = pressure
        self.turbulence = turbulence


class DecisionLogEntry:
    def __init__(self,
                 timestamp: float,
                 stance: str,
                 risk_score: float,
                 bottlenecks: List[str],
                 health_before: float):
        self.timestamp = timestamp
        self.stance = stance
        self.risk_score = risk_score
        self.bottlenecks = bottlenecks
        self.health_before = health_before
        self.health_after: Optional[float] = None


class TriStanceDecisionEngine:
    def __init__(self):
        self.current_stance: str = TriStance.BALANCED
        self.decision_log: List[DecisionLogEntry] = []
        self._lock = threading.Lock()

        self.reinforcement = {
            TriStance.CONSERVATIVE: {"wins": 0, "losses": 0},
            TriStance.BALANCED: {"wins": 0, "losses": 0},
            TriStance.BEAST: {"wins": 0, "losses": 0},
        }

        self.current_deep_ram_ratio: float = 0.5
        self.target_deep_ram_ratio: float = 0.5

    def risk_score(self, bus: PredictionBus) -> float:
        cpu = bus.predicted_load.get("cpu", 0.3)
        mem = bus.predicted_load.get("mem", 0.3)
        disk = bus.predicted_load.get("disk", 0.3)
        avg_load = (cpu + mem + disk) / 3.0

        health_term = 1.0 - bus.health_score
        load_term = avg_load
        turbulence_term = bus.turbulence
        conf_term = 1.0 - bus.prediction_confidence

        raw = 0.35 * health_term + 0.35 * load_term + 0.2 * turbulence_term + 0.1 * conf_term
        return max(0.0, min(1.0, raw))

    def detect_bottlenecks(self, bus: PredictionBus) -> List[str]:
        bottlenecks = []
        if bus.predicted_load.get("cpu", 0.0) > 0.9:
            bottlenecks.append("CPU")
        if bus.predicted_load.get("mem", 0.0) > 0.9:
            bottlenecks.append("MEMORY")
        if bus.predicted_load.get("disk", 0.0) > 0.9:
            bottlenecks.append("DISK")
        if bus.pressure > 0.9:
            bottlenecks.append("PRESSURE")
        if bus.turbulence > 0.8:
            bottlenecks.append("TURBULENCE")
        return bottlenecks

    def deep_ram_target_for_stance(self, stance: str) -> float:
        if stance == TriStance.CONSERVATIVE:
            return 0.2
        if stance == TriStance.BEAST:
            return 0.8
        return 0.5

    def per_source_scaling_hints(self, stance: str) -> Dict[str, str]:
        hints = {}
        if stance == TriStance.CONSERVATIVE:
            hints["ingestion"] = "SLOW"
            hints["threads"] = "LIMITED"
            hints["cache"] = "SHRINK_EARLY"
        elif stance == TriStance.BEAST:
            hints["ingestion"] = "AGGRESSIVE"
            hints["threads"] = "MAX"
            hints["cache"] = "EXPAND"
        else:
            hints["ingestion"] = "MODERATE"
            hints["threads"] = "NORMAL"
            hints["cache"] = "BALANCED"
        return hints

    def _reinforce_last_decision(self, new_health: float) -> None:
        if not self.decision_log:
            return

        last_entry = self.decision_log[-1]
        if last_entry.health_after is not None:
            return

        last_entry.health_after = new_health
        delta = new_health - last_entry.health_before
        record = self.reinforcement[last_entry.stance]
        if delta >= 0.0:
            record["wins"] += 1
        else:
            record["losses"] += 1

    def decide_stance(self, bus: PredictionBus) -> Tuple[str, float, List[str], Dict[str, str]]:
        with self._lock:
            risk = self.risk_score(bus)
            bottlenecks = self.detect_bottlenecks(bus)

            mem_pred = bus.predicted_load.get("mem", 0.3)
            cpu_pred = bus.predicted_load.get("cpu", 0.3)
            temp_score = bus.snapshot.temperature_score

            mem_high = mem_pred > 0.9
            temp_high = temp_score > 0.75
            temp_low = temp_score < 0.35
            headroom_good = (bus.health_score > 0.7 and mem_pred < 0.8 and cpu_pred < 0.8)

            beast_rec = self.reinforcement[TriStance.BEAST]
            beast_helpful = beast_rec["wins"] >= beast_rec["losses"]

            new_stance = self.current_stance

            if self.current_stance == TriStance.BALANCED:
                if mem_high or temp_high or risk > 0.7 or bus.health_score < 0.4:
                    new_stance = TriStance.CONSERVATIVE
                elif headroom_good and temp_low and risk < 0.4 and beast_helpful and bus.pressure > 0.4:
                    new_stance = TriStance.BEAST

            elif self.current_stance == TriStance.CONSERVATIVE:
                if bus.health_score > 0.6 and risk < 0.5 and not mem_high and not temp_high:
                    new_stance = TriStance.BALANCED

            elif self.current_stance == TriStance.BEAST:
                if risk > 0.6 or bus.health_score < 0.5 or mem_high or temp_high:
                    new_stance = TriStance.BALANCED

            self.target_deep_ram_ratio = self.deep_ram_target_for_stance(new_stance)

            entry = DecisionLogEntry(
                timestamp=time.time(),
                stance=new_stance,
                risk_score=risk,
                bottlenecks=bottlenecks,
                health_before=bus.health_score,
            )
            self.decision_log.append(entry)
            self.current_stance = new_stance

            hints = self.per_source_scaling_hints(new_stance)
            return new_stance, risk, bottlenecks, hints

    def reinforce_with_new_health(self, new_health: float) -> None:
        with self._lock:
            self._reinforce_last_decision(new_health)

# =========================
# 9. HYBRID BRAIN WRAPPER
# =========================

class HybridBrain:
    def __init__(self):
        self.sensor_bus = SensorBus()
        self.health = CollectiveHealthScore()
        self.predictive = PredictiveCortex()
        self.situational = SituationalAwarenessCortex()
        self.physics = DataPhysicsEngine()
        self.movidius = MovidiusInferenceEngine()
        self.tristance = TriStanceDecisionEngine()

        self.current_altered_state: str = AlteredState.NORMAL
        self.health_history: List[Tuple[float, float]] = []

    def step(self) -> Dict[str, Any]:
        snap = self.sensor_bus.sample_system(self.current_altered_state)
        self.physics.update_from_snapshot(snap)

        health_score = self.health.compute(self.sensor_bus.history)
        self.health_history.append((snap.timestamp, health_score))
        if len(self.health_history) > 300:
            self.health_history = self.health_history[-300:]

        self.tristance.reinforce_with_new_health(health_score)

        pred_load = self.predictive.predict_load(self.sensor_bus.history)
        confidence = self.predictive.confidence_score(self.sensor_bus.history)

        flow_state = self.physics.qualitative_state()

        new_state = self.situational.update_state(
            health_score=health_score,
            pred_load=pred_load,
            errors_recent=snap.errors_recent,
        )
        self.current_altered_state = new_state

        bus = PredictionBus(
            snapshot=snap,
            health_score=health_score,
            predicted_load=pred_load,
            prediction_confidence=confidence,
            flow_state=flow_state,
            pressure=self.physics.pressure_level,
            turbulence=self.physics.turbulence_level,
        )

        stance, risk, bottlenecks, scaling_hints = self.tristance.decide_stance(bus)
        action = self.predictive.best_guess_action(health_score, pred_load)

        movidius_result = self.movidius.infer({
            "health": health_score,
            "pred_load": pred_load,
            "state": new_state,
            "pressure": self.physics.pressure_level,
            "turbulence": self.physics.turbulence_level,
            "stance": stance,
            "risk": risk,
        })

        return {
            "timestamp": snap.timestamp,
            "health_score": health_score,
            "predicted_load": pred_load,
            "prediction_confidence": confidence,
            "altered_state": new_state,
            "flow_state": flow_state,
            "action_best_guess": action,
            "movidius": movidius_result,
            "stance": stance,
            "risk_score": risk,
            "bottlenecks": bottlenecks,
            "deep_ram_target": self.tristance.target_deep_ram_ratio,
            "scaling_hints": scaling_hints,
            "pressure": self.physics.pressure_level,
            "turbulence": self.physics.turbulence_level,
        }

    # ===== Persistence =====

    def export_state(self) -> Dict[str, Any]:
        return {
            "health_history": self.health_history,
            "tristance_reinforcement": self.tristance.reinforcement,
            "tristance_current_stance": self.tristance.current_stance,
            "tristance_decision_log": [
                {
                    "timestamp": e.timestamp,
                    "stance": e.stance,
                    "risk_score": e.risk_score,
                    "bottlenecks": e.bottlenecks,
                    "health_before": e.health_before,
                    "health_after": e.health_after,
                }
                for e in self.tristance.decision_log[-200:]
            ],
            "current_altered_state": self.current_altered_state,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        try:
            self.health_history = [
                (float(ts), float(score)) for ts, score in state.get("health_history", [])
            ]
            self.tristance.reinforcement = state.get(
                "tristance_reinforcement",
                self.tristance.reinforcement,
            )
            self.tristance.current_stance = state.get(
                "tristance_current_stance",
                self.tristance.current_stance,
            )
            raw_log = state.get("tristance_decision_log", [])
            self.tristance.decision_log = []
            for item in raw_log:
                e = DecisionLogEntry(
                    timestamp=float(item["timestamp"]),
                    stance=item["stance"],
                    risk_score=float(item["risk_score"]),
                    bottlenecks=list(item.get("bottlenecks", [])),
                    health_before=float(item["health_before"]),
                )
                ha = item.get("health_after", None)
                e.health_after = float(ha) if ha is not None else None
                self.tristance.decision_log.append(e)

            self.current_altered_state = state.get(
                "current_altered_state",
                self.current_altered_state,
            )
        except Exception as e:
            print(f"[BRAIN IMPORT] Failed to import state: {e}", file=sys.stderr)

# =========================
# 10. SETTINGS & BACKUP HELPERS
# =========================

def load_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_FILENAME):
        return {
            "local_backup_path": "",
            "smb_backup_path": "",
            "auto_save_interval_sec": 300,
        }
    try:
        with open(SETTINGS_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("local_backup_path", "")
        data.setdefault("smb_backup_path", "")
        data.setdefault("auto_save_interval_sec", 300)
        return data
    except Exception as e:
        print(f"[SETTINGS] Failed to load settings: {e}", file=sys.stderr)
        return {
            "local_backup_path": "",
            "smb_backup_path": "",
            "auto_save_interval_sec": 300,
        }

def save_settings(settings: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[SETTINGS] Failed to save settings: {e}", file=sys.stderr)

def write_json(path: str, filename: str, data: Dict[str, Any]) -> bool:
    try:
        if not path:
            return False
        os.makedirs(path, exist_ok=True)
        full = os.path.join(path, filename)
        with open(full, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[WRITE_JSON] Failed to write {filename} to {path}: {e}", file=sys.stderr)
        return False

def read_json(path: str, filename: str) -> Optional[Dict[str, Any]]:
    try:
        full = os.path.join(path, filename)
        if not os.path.exists(full):
            return None
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[READ_JSON] Failed to read {filename} from {path}: {e}", file=sys.stderr)
        return None

def test_smb_path(path: str) -> bool:
    if not path:
        return False
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        _ = os.listdir(path)
        return True
    except Exception as e:
        print(f"[SMB TEST] SMB path test failed for {path}: {e}", file=sys.stderr)
        return False

# =========================
# 11. TKINTER MAGICBOX COCKPIT GUI WITH TABS
# =========================

class BrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Brain MagicBox Cockpit — Full Control Room")
        self.root.geometry("1180x680")
        self.root.resizable(False, False)

        self.brain = HybridBrain()

        self.settings = load_settings()
        self.last_save_time_local: Optional[float] = None
        self.last_save_time_smb: Optional[float] = None
        self.last_load_time: Optional[float] = None
        self.last_auto_save_ts: float = time.time()

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(side="left", fill="both", expand=False, padx=(0, 10))

        # Right visuals (shared, independent of tabs)
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Tabs
        self.tab_status = ttk.Frame(self.notebook)
        self.tab_magic = ttk.Frame(self.notebook)
        self.tab_visuals = ttk.Frame(self.notebook)
        self.tab_backup = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_status, text="Status")
        self.notebook.add(self.tab_magic, text="MagicBox")
        self.notebook.add(self.tab_visuals, text="Visuals")
        self.notebook.add(self.tab_backup, text="Memory Backup")

        # ---- STATUS TAB ----
        title_label = ttk.Label(
            self.tab_status,
            text="Hybrid Brain Status",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        self.status_grid = ttk.Frame(self.tab_status)
        self.status_grid.pack(fill="x", pady=(0, 10))

        status_fields = [
            "Health Score",
            "Prediction Confidence",
            "Predicted CPU Load",
            "Predicted MEM Load",
            "Predicted DISK Load",
            "Altered State",
            "Tri-Stance",
            "Risk Score",
            "Flow State",
            "Pressure",
            "Turbulence",
            "Deep RAM Target",
            "Best Guess Action",
            "Bottlenecks",
            "Movidius Status",
        ]

        self.labels = {}
        for i, field in enumerate(status_fields):
            lbl = ttk.Label(self.status_grid, text=f"{field}:", font=("Segoe UI", 10))
            lbl.grid(row=i, column=0, sticky="w", pady=2)
            val = ttk.Label(self.status_grid, text="...", font=("Segoe UI", 10, "bold"))
            val.grid(row=i, column=1, sticky="w", pady=2)
            self.labels[field] = val

        # ---- MAGICBOX TAB ----
        self.magicbox_frame = ttk.LabelFrame(
            self.tab_magic,
            text="MagicBox — Tri-Stance Decision Engine",
            padding=8
        )
        self.magicbox_frame.pack(fill="x", pady=(10, 10), padx=5)

        self.magicbox_canvas = tk.Canvas(
            self.magicbox_frame,
            width=380,
            height=180,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.magicbox_canvas.pack()

        # Deep RAM display in MagicBox tab
        self.magicbox_ram_label = ttk.Label(
            self.tab_magic,
            text="Deep RAM Target: (pending)",
            font=("Segoe UI", 10)
        )
        self.magicbox_ram_label.pack(anchor="w", padx=5, pady=(2, 2))

        # Reinforcement summary (MagicBox tab)
        self.magic_reinforcement_label = ttk.Label(
            self.tab_magic,
            text="Tri-Stance Reinforcement (wins/losses):",
            font=("Segoe UI", 10, "bold")
        )
        self.magic_reinforcement_label.pack(anchor="w", padx=5)

        self.magic_reinforcement_text = tk.Text(
            self.tab_magic,
            height=6,
            width=50,
            state="disabled",
            font=("Segoe UI", 9),
            bg="#f9f9f9"
        )
        self.magic_reinforcement_text.pack(fill="x", padx=5, pady=(2, 5))

        # ---- VISUALS TAB (LEFT-SIDE CONTENT) ----
        visuals_info = ttk.Label(
            self.tab_visuals,
            text="Visuals live on the right panel: Health, Load, Tiers, Water Flow.",
            font=("Segoe UI", 10)
        )
        visuals_info.pack(anchor="w", padx=5, pady=5)

        # ---- BACKUP TAB ----
        self.backup_frame = ttk.LabelFrame(
            self.tab_backup,
            text="Memory Backup (Local + SMB)",
            padding=8
        )
        self.backup_frame.pack(fill="x", pady=(10, 10), padx=5)

        self.local_path_label = ttk.Label(
            self.backup_frame,
            text=f"Local path: {self.settings.get('local_backup_path','') or '(none)'}",
            font=("Segoe UI", 9)
        )
        self.local_path_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        self.btn_choose_local = ttk.Button(
            self.backup_frame,
            text="Choose Local Backup Folder",
            command=self.choose_local_folder
        )
        self.btn_choose_local.grid(row=1, column=0, sticky="w", pady=2)

        self.smb_path_label = ttk.Label(
            self.backup_frame,
            text=f"SMB path: {self.settings.get('smb_backup_path','') or '(none)'}",
            font=("Segoe UI", 9)
        )
        self.smb_path_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        self.btn_set_smb = ttk.Button(
            self.backup_frame,
            text="Set SMB UNC Path",
            command=self.set_smb_path
        )
        self.btn_set_smb.grid(row=3, column=0, sticky="w", pady=2)

        self.btn_test_smb = ttk.Button(
            self.backup_frame,
            text="Test SMB Connection",
            command=self.test_smb_connection
        )
        self.btn_test_smb.grid(row=3, column=1, sticky="w", pady=2)

        self.btn_save_now = ttk.Button(
            self.backup_frame,
            text="Save Memory Now",
            command=self.save_memory_now
        )
        self.btn_save_now.grid(row=4, column=0, sticky="w", pady=4)

        self.btn_load_now = ttk.Button(
            self.backup_frame,
            text="Load Memory",
            command=self.load_memory_now
        )
        self.btn_load_now.grid(row=4, column=1, sticky="w", pady=4)

        self.backup_status_label = ttk.Label(
            self.backup_frame,
            text="Status: Idle",
            font=("Segoe UI", 9)
        )
        self.backup_status_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4,0))

        self.last_save_label = ttk.Label(
            self.backup_frame,
            text="Last Save: (none)",
            font=("Segoe UI", 8)
        )
        self.last_save_label.grid(row=6, column=0, columnspan=2, sticky="w")

        self.last_load_label = ttk.Label(
            self.backup_frame,
            text="Last Load: (none)",
            font=("Segoe UI", 8)
        )
        self.last_load_label.grid(row=7, column=0, columnspan=2, sticky="w")

        # ---- RIGHT VISUALS (shared) ----
        self.top_visuals = ttk.Frame(self.right_frame)
        self.top_visuals.pack(fill="x", pady=(0, 10))

        self.health_canvas = tk.Canvas(
            self.top_visuals,
            width=420,
            height=80,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.health_canvas.pack(side="left", padx=(0, 10))
        self.health_canvas.create_text(
            5, 5, anchor="nw",
            text="Health Timeline",
            font=("Segoe UI", 9, "bold")
        )

        self.load_canvas = tk.Canvas(
            self.top_visuals,
            width=280,
            height=80,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.load_canvas.pack(side="right")
        self.load_canvas.create_text(
            5, 5, anchor="nw",
            text="Predicted Load",
            font=("Segoe UI", 9, "bold")
        )

        self.middle_visuals = ttk.Frame(self.right_frame)
        self.middle_visuals.pack(fill="x", pady=(0, 10))

        self.tier_canvas = tk.Canvas(
            self.middle_visuals,
            width=420,
            height=160,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.tier_canvas.pack(side="left", padx=(0, 10))
        self.tier_canvas.create_text(
            5, 5, anchor="nw",
            text="Memory Tiers (conceptual)",
            font=("Segoe UI", 9, "bold")
        )

        self.flow_canvas = tk.Canvas(
            self.middle_visuals,
            width=280,
            height=160,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.flow_canvas.pack(side="right")
        self.flow_canvas.create_text(
            5, 5, anchor="nw",
            text="Data Flow (Water Model)",
            font=("Segoe UI", 9, "bold")
        )

        self.bottom_visuals = ttk.Frame(self.right_frame)
        self.bottom_visuals.pack(fill="x")

        self.reinforcement_label = ttk.Label(
            self.bottom_visuals,
            text="Reinforcement Summary (Tri-Stance wins/losses):",
            font=("Segoe UI", 10, "bold")
        )
        self.reinforcement_label.pack(anchor="w")

        self.reinforcement_text = tk.Text(
            self.bottom_visuals,
            height=4,
            width=90,
            state="disabled",
            font=("Segoe UI", 9),
            bg="#f9f9f9"
        )
        self.reinforcement_text.pack(fill="x", pady=(2, 0))

        # Auto-load previous brain state
        self.try_auto_load()

        self.update_gui()

    # ===== Backup handlers =====

    def choose_local_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.settings["local_backup_path"] = path
            save_settings(self.settings)
            self.local_path_label.config(text=f"Local path: {path}")

    def set_smb_path(self):
        current = self.settings.get("smb_backup_path", "")
        path = simpledialog.askstring(
            "SMB Path",
            "Enter SMB UNC path (anonymous):\nExample: \\\\SERVER\\Share\\Folder",
            initialvalue=current or ""
        )
        if path:
            self.settings["smb_backup_path"] = path
            save_settings(self.settings)
            self.smb_path_label.config(text=f"SMB path: {path}")

    def test_smb_connection(self):
        path = self.settings.get("smb_backup_path", "")
        ok = test_smb_path(path)
        if ok:
            messagebox.showinfo("SMB Test", f"Success: SMB path is reachable.\n{path}")
        else:
            messagebox.showwarning("SMB Test", f"Failed: SMB path is not reachable or cannot be created.\n{path}")

    def save_memory_now(self):
        self.backup_status_label.config(text="Status: Saving...", foreground="#0078d4")
        self.root.update_idletasks()

        brain_state = self.brain.export_state()
        settings_copy = dict(self.settings)

        local_ok = write_json(self.settings.get("local_backup_path", ""),
                              BRAIN_STATE_FILENAME, brain_state)
        local_ok &= write_json(self.settings.get("local_backup_path", ""),
                               SETTINGS_FILENAME, settings_copy)

        smb_ok = write_json(self.settings.get("smb_backup_path", ""),
                            BRAIN_STATE_FILENAME, brain_state)
        smb_ok &= write_json(self.settings.get("smb_backup_path", ""),
                             SETTINGS_FILENAME, settings_copy)

        now = time.time()
        if local_ok:
            self.last_save_time_local = now
        if smb_ok:
            self.last_save_time_smb = now

        status_parts = []
        if local_ok:
            status_parts.append("Local: OK")
        elif self.settings.get("local_backup_path"):
            status_parts.append("Local: FAIL")

        if smb_ok:
            status_parts.append("SMB: OK")
        elif self.settings.get("smb_backup_path"):
            status_parts.append("SMB: FAIL")

        status_text = "Status: " + (", ".join(status_parts) if status_parts else "No paths configured")
        self.backup_status_label.config(
            text=status_text,
            foreground="#107c10" if (local_ok or smb_ok) else "#d13438"
        )

        parts = []
        if self.last_save_time_local:
            parts.append("Local " + time.strftime("%H:%M:%S", time.localtime(self.last_save_time_local)))
        if self.last_save_time_smb:
            parts.append("SMB " + time.strftime("%H:%M:%S", time.localtime(self.last_save_time_smb)))
        self.last_save_label.config(
            text="Last Save: " + (", ".join(parts) if parts else "(none)")
        )

    def load_memory_from_path(self, base_path: str) -> bool:
        if not base_path:
            return False
        state = read_json(base_path, BRAIN_STATE_FILENAME)
        if not state:
            return False
        self.brain.import_state(state)
        settings_from_path = read_json(base_path, SETTINGS_FILENAME)
        if settings_from_path:
            self.settings.update(settings_from_path)
            save_settings(self.settings)
            self.local_path_label.config(
                text=f"Local path: {self.settings.get('local_backup_path','') or '(none)'}"
            )
            self.smb_path_label.config(
                text=f"SMB path: {self.settings.get('smb_backup_path','') or '(none)'}"
            )
        return True

    def load_memory_now(self):
        local_ok = self.load_memory_from_path(self.settings.get("local_backup_path", ""))
        smb_ok = False
        if not local_ok:
            smb_ok = self.load_memory_from_path(self.settings.get("smb_backup_path", ""))

        if local_ok or smb_ok:
            self.last_load_time = time.time()
            self.last_load_label.config(
                text="Last Load: " + time.strftime("%H:%M:%S", time.localtime(self.last_load_time))
            )
            messagebox.showinfo("Load Memory", "Brain state loaded successfully.")
        else:
            messagebox.showwarning("Load Memory", "No valid brain_state.json found in local or SMB paths.")

    def try_auto_load(self):
        local_ok = self.load_memory_from_path(self.settings.get("local_backup_path", ""))
        if not local_ok:
            smb_ok = self.load_memory_from_path(self.settings.get("smb_backup_path", ""))
            if smb_ok:
                self.last_load_time = time.time()
                self.last_load_label.config(
                    text="Last Load: " + time.strftime("%H:%M:%S", time.localtime(self.last_load_time))
                )
        else:
            self.last_load_time = time.time()
            self.last_load_label.config(
                text="Last Load: " + time.strftime("%H:%M:%S", time.localtime(self.last_load_time))
            )

    # ===== Visual helpers =====

    def draw_health_timeline(self, history: List[Tuple[float, float]]) -> None:
        c = self.health_canvas
        c.delete("data")

        w = int(c["width"])
        h = int(c["height"]) - 15

        if not history:
            return

        recent = history[-100:]
        n = len(recent)
        if n < 2:
            return

        dx = (w - 10) / (n - 1)
        last_x, last_y = None, None
        for i, (_, score) in enumerate(recent):
            x = 5 + i * dx
            y = 10 + (1.0 - score) * (h - 10)
            c.create_oval(x - 1, y - 1, x + 1, y + 1, fill="#0078d4", outline="", tags="data")
            if last_x is not None:
                c.create_line(last_x, last_y, x, y, fill="#0078d4", width=1, tags="data")
            last_x, last_y = x, y

    def draw_load_bars(self, predicted_load: dict) -> None:
        c = self.load_canvas
        c.delete("data")

        w = int(c["width"])
        labels = ["cpu", "mem", "disk"]
        colors = {"cpu": "#d13438", "mem": "#107c10", "disk": "#0078d4"}
        bar_width = (w - 40) / len(labels)

        for i, key in enumerate(labels):
            val = float(predicted_load.get(key, 0.0))
            val = max(0.0, min(1.0, val))
            x0 = 10 + i * bar_width
            x1 = x0 + bar_width - 10
            y1 = 70
            y0 = y1 - val * 40.0
            c.create_rectangle(x0, y0, x1, y1, fill=colors[key], outline="", tags="data")
            c.create_text(
                (x0 + x1) / 2, y1 + 5,
                text=key.upper(),
                font=("Segoe UI", 8),
                tags="data",
                anchor="n"
            )

    def draw_tier_map(self, deep_target: float, stance: str) -> None:
        c = self.tier_canvas
        c.delete("data")

        w = int(c["width"])
        h = int(c["height"]) - 20

        tiers = [("TIER 0 - HOT (RAM/VRAM)", "#d13438"),
                 ("TIER 1 - WARM (NVMe/SSD)", "#ffaa44"),
                 ("TIER 2 - COLD (HDD/SMB)", "#0078d4")]
        tier_height = (h - 20) / len(tiers)

        for i, (label, color) in enumerate(tiers):
            y0 = 20 + i * tier_height
            y1 = y0 + tier_height - 5
            c.create_rectangle(10, y0, w - 10, y1, outline=color, width=2, tags="data")
            c.create_text(
                15, y0 + 5, anchor="nw",
                text=label,
                font=("Segoe UI", 9),
                tags="data"
            )

        c.create_text(
            10, 10, anchor="nw",
            text=f"Deep RAM Target: {deep_target:.2f}   (Stance: {stance})",
            font=("Segoe UI", 9, "bold"),
            tags="data"
        )

    def draw_flow_state(self, flow_state: str, pressure: float, turbulence: float) -> None:
        c = self.flow_canvas
        c.delete("data")

        w = int(c["width"])
        h = int(c["height"]) - 20

        bg_color = "#e5f1fb"
        if flow_state == "CALM":
            bg_color = "#e9f7ef"
        elif flow_state == "FAST_FLOW":
            bg_color = "#fff4ce"
        elif flow_state == "RAPIDS":
            bg_color = "#fde7e9"
        elif flow_state == "STEADY_FLOW":
            bg_color = "#e5f1fb"

        c.create_rectangle(0, 0, w, h + 20, fill=bg_color, outline="", tags="data")

        c.create_text(
            5, 5, anchor="nw",
            text=f"Flow: {flow_state}",
            font=("Segoe UI", 10, "bold"),
            tags="data"
        )
        c.create_text(
            5, 25, anchor="nw",
            text=f"Pressure: {pressure:.2f}",
            font=("Segoe UI", 9),
            tags="data"
        )
        c.create_text(
            5, 40, anchor="nw",
            text=f"Turbulence: {turbulence:.2f}",
            font=("Segoe UI", 9),
            tags="data"
        )

        base_y = h / 2 + 10
        thickness = 10 + pressure * 20

        c.create_rectangle(10, base_y - thickness / 2, w - 10, base_y + thickness / 2,
                           fill="#0078d4", outline="", tags="data")

        bubbles = int(5 + turbulence * 20)
        for i in range(bubbles):
            x = 15 + i * (w - 30) / max(1, bubbles - 1)
            y = base_y - thickness / 2 - 5 - (turbulence * 10)
            c.create_oval(x - 3, y - 3, x + 3, y + 3,
                          fill="#004578", outline="", tags="data")

    def draw_magicbox(self, stance: str, risk: float) -> None:
        c = self.magicbox_canvas
        c.delete("all")

        info = TRI_STANCE_DESCRIPTIONS.get(stance, TRI_STANCE_DESCRIPTIONS[TriStance.BALANCED])
        name = info["name"]
        lines = info["lines"]
        bg = info["color"]
        border = info["border"]

        w = int(c["width"])
        h = int(c["height"])

        c.create_rectangle(0, 0, w, h, fill=bg, outline=border, width=2)

        c.create_text(
            8, 8, anchor="nw",
            text=name,
            font=("Segoe UI", 11, "bold"),
            fill="#000000"
        )

        y = 30
        for line in lines:
            c.create_text(
                20, y, anchor="nw",
                text=f"• {line}",
                font=("Segoe UI", 9),
                fill="#333333"
            )
            y += 16

        risk_color = "#0078d4"
        if risk > 0.7:
            risk_color = "#d13438"
        elif risk > 0.4:
            risk_color = "#ff8c00"

        c.create_rectangle(0, h - 20, w, h, fill=risk_color, outline="")
        c.create_text(
            8, h - 10, anchor="w",
            text=f"Risk: {risk:.2f}",
            font=("Segoe UI", 9, "bold"),
            fill="#ffffff"
        )
        c.create_text(
            w - 8, h - 10, anchor="e",
            text="Tri-Stance running autonomously",
            font=("Segoe UI", 8),
            fill="#ffffff"
        )

    def update_reinforcement_text(self) -> None:
        r = self.brain.tristance.reinforcement
        text_lines = []
        for stance, rec in r.items():
            text_lines.append(f"{stance}: wins={rec['wins']} losses={rec['losses']}")
        joined = "\n".join(text_lines)

        self.reinforcement_text.configure(state="normal")
        self.reinforcement_text.delete("1.0", tk.END)
        self.reinforcement_text.insert("1.0", joined)
        self.reinforcement_text.configure(state="disabled")

        self.magic_reinforcement_text.configure(state="normal")
        self.magic_reinforcement_text.delete("1.0", tk.END)
        self.magic_reinforcement_text.insert("1.0", joined)
        self.magic_reinforcement_text.configure(state="disabled")

    # ===== main loop =====

    def update_gui(self):
        state = self.brain.step()

        # Status tab labels
        self.labels["Health Score"].config(text=f"{state['health_score']:.2f}")
        self.labels["Prediction Confidence"].config(text=f"{state['prediction_confidence']:.2f}")
        self.labels["Predicted CPU Load"].config(text=f"{state['predicted_load']['cpu']:.2f}")
        self.labels["Predicted MEM Load"].config(text=f"{state['predicted_load']['mem']:.2f}")
        self.labels["Predicted DISK Load"].config(text=f"{state['predicted_load']['disk']:.2f}")
        self.labels["Altered State"].config(text=state["altered_state"])
        self.labels["Tri-Stance"].config(text=state["stance"])
        self.labels["Risk Score"].config(text=f"{state['risk_score']:.2f}")
        self.labels["Flow State"].config(text=state["flow_state"])
        self.labels["Pressure"].config(text=f"{state['pressure']:.2f}")
        self.labels["Turbulence"].config(text=f"{state['turbulence']:.2f}")
        self.labels["Deep RAM Target"].config(text=f"{state['deep_ram_target']:.2f}")
        self.labels["Best Guess Action"].config(text=state["action_best_guess"])

        bottlenecks = state["bottlenecks"]
        self.labels["Bottlenecks"].config(
            text=", ".join(bottlenecks) if bottlenecks else "None"
        )

        mov = state["movidius"]
        if mov["prediction"] is None:
            self.labels["Movidius Status"].config(text="Unavailable")
        else:
            self.labels["Movidius Status"].config(
                text=f"{mov['prediction']} ({mov['confidence']:.2f})"
            )

        # Visuals
        self.draw_health_timeline(self.brain.health_history)
        self.draw_load_bars(state["predicted_load"])
        self.draw_tier_map(state["deep_ram_target"], state["stance"])
        self.draw_flow_state(state["flow_state"],
                             state["pressure"],
                             state["turbulence"])
        self.draw_magicbox(state["stance"], state["risk_score"])
        self.magicbox_ram_label.config(
            text=f"Deep RAM Target: {state['deep_ram_target']:.2f} (Stance: {state['stance']})"
        )
        self.update_reinforcement_text()

        # Auto-save
        interval = self.settings.get("auto_save_interval_sec", 300)
        now = time.time()
        if interval > 0 and (now - self.last_auto_save_ts) >= interval:
            self.save_memory_now()
            self.last_auto_save_ts = now

        self.root.after(1000, self.update_gui)

# =========================
# 12. ENTRY POINT
# =========================

def main():
    root = tk.Tk()
    gui = BrainGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

