from __future__ import annotations

# ============================================================
# Imports
# ============================================================

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
import importlib
import traceback
import threading
import queue
import numpy as np
import time
import os
import sys
import random

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QComboBox,
    QMainWindow,
)
from PySide6.QtGui import QFont

# ============================================================
# 0. OpenVINO core detection
# ============================================================

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except Exception as e:
    print(f"[AUTOLOADER] OpenVINO import failed: {e}")
    Core = None
    OPENVINO_AVAILABLE = False


# ============================================================
# 1. Core enums and small data structures
# ============================================================

class Mission(Enum):
    PROTECT = auto()
    STABILITY = auto()
    LEARN = auto()
    OPTIMIZE = auto()
    AUTO = auto()


class Environment(Enum):
    CALM = auto()
    TENSE = auto()
    DANGER = auto()
    UNKNOWN = auto()


class PredictionHorizon(Enum):
    SHORT = auto()
    MEDIUM = auto()
    LONG = auto()


class HiveSyncMode(Enum):
    AGGRESSIVE = auto()
    CONSERVATIVE = auto()
    LOCAL_ONLY = auto()


class HealthTrend(Enum):
    IMPROVING = auto()
    STABLE = auto()
    DECLINING = auto()
    UNKNOWN = auto()


class CommandType(Enum):
    STABILIZE_SYSTEM = auto()
    HIGH_ALERT_MODE = auto()
    BEGIN_LEARNING_CYCLE = auto()
    OPTIMIZE_PERFORMANCE = auto()
    PURGE_ANOMALY_MEMORY = auto()
    REBUILD_PREDICTIVE_MODEL = auto()
    RESET_SITUATIONAL_CORTEX = auto()
    SNAPSHOT_BRAIN_STATE = auto()
    ROLLBACK_PREVIOUS_STATE = auto()


@dataclass
class Command:
    type: CommandType
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueAnswer:
    intent: str
    confidence: float
    alternatives: List[str]
    risks: List[str]
    expected_outcomes: List[str]


# ============================================================
# 2. Movidius/OpenVINO-backed engine + controller
# ============================================================

class MovidiusEngine:
    """
    OpenVINO-backed engine that prefers MYRIAD, falls back to CPU/GPU, else NONE.
    """

    def __init__(self):
        self.core = None
        self.device_connected = False
        self.backend_device = None  # "MYRIAD", "CPU", "GPU", or "NONE"
        self.available_devices = []

        self.loaded_graphs = {}
        self.total_inferences = 0
        self.latencies_ms = []
        self.queue_depth = 0
        self.lock = threading.Lock()
        self.running = True

        self.init_core()

    def init_core(self):
        if not OPENVINO_AVAILABLE:
            print("[Engine] OpenVINO not available. Running in 'NONE' mode.")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False
            return

        try:
            self.core = Core()
            self.available_devices = self.core.available_devices
            print(f"[Engine] OpenVINO available devices: {self.available_devices}")
        except Exception as e:
            print(f"[Engine] Failed to initialize OpenVINO Core: {e}")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False

    def connect_device(self):
        with self.lock:
            if self.core is None:
                self.device_connected = False
                self.backend_device = "NONE"
                return False

            devices = self.available_devices
            chosen = None
            if "MYRIAD" in devices:
                chosen = "MYRIAD"
            elif "GPU" in devices:
                chosen = "GPU"
            elif "CPU" in devices:
                chosen = "CPU"

            if chosen is None:
                self.device_connected = False
                self.backend_device = "NONE"
                return False

            self.backend_device = chosen
            self.device_connected = True
            return True

    def disconnect_device(self):
        with self.lock:
            self.device_connected = False
            self.backend_device = "NONE"
            self.loaded_graphs.clear()

    def load_graph(self, graph_id: str, graph_path: str):
        with self.lock:
            if not self.device_connected or self.core is None or self.backend_device == "NONE":
                raise RuntimeError("No acceleration device connected")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Model XML not found: {graph_path}")

        model = self.core.read_model(graph_path)
        compiled_model = self.core.compile_model(model=model, device_name=self.backend_device)

        input_layer = compiled_model.inputs[0]
        output_layer = compiled_model.outputs[0]
        input_name = input_layer.get_any_name()
        output_name = output_layer.get_any_name()

        with self.lock:
            self.loaded_graphs[graph_id] = {
                "model_path": graph_path,
                "compiled_model": compiled_model,
                "input_name": input_name,
                "output_name": output_name,
            }

    def unload_graph(self, graph_id: str):
        with self.lock:
            if graph_id in self.loaded_graphs:
                del self.loaded_graphs[graph_id]

    def run_inference(self, graph_id: str, input_array: np.ndarray):
        with self.lock:
            if not self.device_connected or self.core is None or self.backend_device == "NONE":
                raise RuntimeError("No acceleration device connected")
            if graph_id not in self.loaded_graphs:
                raise RuntimeError(f"Graph not loaded: {graph_id}")
            entry = self.loaded_graphs[graph_id]
            compiled_model = entry["compiled_model"]
            input_name = entry["input_name"]
            self.queue_depth += 1

        start = time.time()
        infer_request = compiled_model.create_infer_request()
        infer_request.infer({input_name: input_array})
        output = infer_request.get_output_tensor().data
        latency_ms = (time.time() - start) * 1000.0

        with self.lock:
            self.queue_depth = max(0, self.queue_depth - 1)
            self.total_inferences += 1
            self.latencies_ms.append(latency_ms)
            self.latencies_ms = self.latencies_ms[-200:]

        output = np.array(output)
        return output, latency_ms

    def get_metrics(self):
        with self.lock:
            if self.latencies_ms:
                avg_latency = sum(self.latencies_ms) / len(self.latencies_ms)
                fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
            else:
                avg_latency = 0.0
                fps = 0.0
            metrics = {
                "device_connected": self.device_connected,
                "backend_device": self.backend_device,
                "available_devices": list(self.available_devices),
                "loaded_graphs": {k: v["model_path"] for k, v in self.loaded_graphs.items()},
                "avg_latency_ms": avg_latency,
                "fps": fps,
                "queue_depth": self.queue_depth,
                "total_inferences": self.total_inferences,
            }
        return metrics

    def stop(self):
        self.running = False


class EngineController:
    """
    Threaded controller that serializes operations to the engine.
    """

    def __init__(self, engine: MovidiusEngine, log_queue: queue.Queue, event_queue: queue.Queue):
        self.engine = engine
        self.cmd_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.log_queue = log_queue
        self.event_queue = event_queue
        self.thread.start()

    def _log(self, msg: str):
        try:
            self.log_queue.put(msg, block=False)
        except queue.Full:
            pass

    def _event(self, event_type: str, payload: dict):
        try:
            self.event_queue.put((event_type, payload), block=False)
        except queue.Full:
            pass

    def _worker_loop(self):
        self._log("[EngineController] Worker thread started.")
        while self.engine.running:
            try:
                cmd, args, res_queue = self.cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if cmd == "connect_device":
                    ok = self.engine.connect_device()
                    status = "connected" if ok else "failed"
                    self._event("device", {"status": status})
                    res_queue.put((ok, ok))

                elif cmd == "disconnect_device":
                    self.engine.disconnect_device()
                    self._event("device", {"status": "disconnected"})
                    res_queue.put((True, True))

                elif cmd == "load_graph":
                    graph_id, graph_path = args
                    self.engine.load_graph(graph_id, graph_path)
                    self._event("graph", {"graph_id": graph_id, "status": "loaded"})
                    res_queue.put((True, True))

                elif cmd == "unload_graph":
                    graph_id = args[0]
                    self.engine.unload_graph(graph_id)
                    self._event("graph", {"graph_id": graph_id, "status": "unloaded"})
                    res_queue.put((True, True))

                elif cmd == "run_inference":
                    graph_id, input_arr = args
                    output, latency = self.engine.run_inference(graph_id, input_arr)
                    res_queue.put((True, (output, latency)))

                elif cmd == "get_metrics":
                    metrics = self.engine.get_metrics()
                    res_queue.put((True, metrics))

                elif cmd == "stop":
                    self.engine.stop()
                    res_queue.put((True, True))
                    break

                else:
                    raise ValueError(f"Unknown command: {cmd}")

            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"[EngineController] ERROR in cmd '{cmd}': {e}\n{tb}")
                res_queue.put((False, e))


# ============================================================
# 3. MovidiusInferenceEngine organ wrapper
# ============================================================

class MovidiusInferenceEngine:
    """
    High-level organ used by the brain runtime.
    Provides accelerate(state, forecast) with safe fallback.
    """

    def __init__(self,
                 model_graph_id: str = "default_graph",
                 model_xml_path: str | None = None):
        self.log_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=1000)

        self.engine = MovidiusEngine()
        self.controller = EngineController(self.engine, self.log_queue, self.event_queue)

        self.model_graph_id = model_graph_id
        self.model_xml_path = model_xml_path  # set to real model if you have one

        self.device_ready = False
        self.graph_loaded = False

        self._init_thread = threading.Thread(target=self._bootstrap, daemon=True)
        self._init_thread.start()

    def _call(self, cmd: str, *args, timeout: float = 3.0):
        res_queue = queue.Queue(maxsize=1)
        self.controller.cmd_queue.put((cmd, args, res_queue))
        ok, payload = res_queue.get(timeout=timeout)
        if not ok:
            raise RuntimeError(payload)
        return payload

    def _bootstrap(self):
        try:
            self.device_ready = self._call("connect_device")
        except Exception as e:
            print(f"[MovidiusInferenceEngine] Device connect failed: {e}")
            self.device_ready = False

        if self.device_ready and self.model_xml_path:
            try:
                self._call("load_graph", self.model_graph_id, self.model_xml_path)
                self.graph_loaded = True
            except Exception as e:
                print(f"[MovidiusInferenceEngine] Graph load failed: {e}")
                self.graph_loaded = False
        else:
            self.graph_loaded = False

    def get_status(self) -> dict:
        try:
            metrics = self._call("get_metrics")
        except Exception as e:
            metrics = {"error": str(e)}
        return {
            "device_ready": self.device_ready,
            "graph_loaded": self.graph_loaded,
            "metrics": metrics,
        }

    def accelerate(self, state: dict, forecast: dict) -> dict:
        if not self.device_ready:
            try:
                self.device_ready = self._call("connect_device")
            except Exception:
                self.device_ready = False

        if self.device_ready and not self.graph_loaded and self.model_xml_path:
            try:
                self._call("load_graph", self.model_graph_id, self.model_xml_path)
                self.graph_loaded = True
            except Exception:
                self.graph_loaded = False

        if (not self.device_ready) or (not self.graph_loaded):
            status = self.get_status()
            metrics = status.get("metrics", {})
            backend = metrics.get("backend_device", "NONE")
            return {
                "used": False,
                "backend": backend,
                "anomaly_score": None,
                "risk_adjustment": 0.0,
                "metrics": metrics,
            }

        risk = float(state.get("risk_score", 0.0))
        opportunity = float(state.get("opportunity_score", 0.0))
        anomaly_risk = float(forecast.get("anomaly_risk", risk))
        drive_risk = float(forecast.get("drive_risk", risk))
        hive_risk = float(forecast.get("hive_risk", risk))

        features = np.array([[risk, opportunity, anomaly_risk, drive_risk, hive_risk]], dtype=np.float32)

        try:
            ok, result = self._safe_run_inference(features)
            if not ok:
                raise RuntimeError(result)

            output, latency_ms = result

            if output.ndim == 0:
                anomaly_score = float(output)
                risk_delta = anomaly_score - anomaly_risk
            else:
                flat = output.flatten()
                anomaly_score = float(flat[0])
                risk_delta = float(flat[1]) if flat.size > 1 else anomaly_score - anomaly_risk

            metrics = self._call("get_metrics")
            backend = metrics.get("backend_device", "UNKNOWN")

            return {
                "used": True,
                "backend": backend,
                "anomaly_score": anomaly_score,
                "risk_adjustment": risk_delta,
                "metrics": {
                    "latency_ms": latency_ms,
                    **metrics,
                },
            }

        except Exception as e:
            print(f"[MovidiusInferenceEngine] accelerate() error: {e}")
            status = self.get_status()
            metrics = status.get("metrics", {})
            backend = metrics.get("backend_device", "NONE")
            return {
                "used": False,
                "backend": backend,
                "anomaly_score": None,
                "risk_adjustment": 0.0,
                "metrics": metrics,
            }

    def _safe_run_inference(self, features: np.ndarray, timeout: float = 3.0):
        res_queue = queue.Queue(maxsize=1)
        self.controller.cmd_queue.put(("run_inference", (self.model_graph_id, features), res_queue))
        ok, payload = res_queue.get(timeout=timeout)
        return ok, payload

    def stop(self):
        try:
            self._call("stop")
        except Exception:
            pass


# ============================================================
# 4. Panels / organs as stateful components
# ============================================================

@dataclass
class SituationalState:
    mission: Mission = Mission.AUTO
    effective_mission: Mission = Mission.PROTECT
    environment: Environment = Environment.UNKNOWN
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    anticipation: str = "Unknown"
    mission_override_active: bool = False
    risk_tolerance: float = 0.5
    opportunity_bias: float = 0.5
    learning_window_bias: float = 0.5


class SituationalAwarenessCortex:
    def __init__(self):
        self.state = SituationalState()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "mission": self.state.mission.name,
            "effective_mission": self.state.effective_mission.name,
            "environment": self.state.environment.name,
            "opportunity_score": self.state.opportunity_score,
            "risk_score": self.state.risk_score,
            "anticipation": self.state.anticipation,
            "risk_tolerance": self.state.risk_tolerance,
            "opportunity_bias": self.state.opportunity_bias,
            "learning_window_bias": self.state.learning_window_bias,
            "mission_override_active": self.state.mission_override_active,
        }

    def set_mission_override(self, mission: Mission):
        self.state.mission_override_active = True
        self.state.mission = mission

    def force_protect(self): self.set_mission_override(Mission.PROTECT)
    def force_learn(self): self.set_mission_override(Mission.LEARN)
    def force_optimize(self): self.set_mission_override(Mission.OPTIMIZE)

    def return_to_auto(self):
        self.state.mission_override_active = False
        self.state.mission = Mission.AUTO

    def set_risk_tolerance(self, value: float):
        self.state.risk_tolerance = min(max(value, 0.0), 1.0)

    def accept_more_risk(self, step: float = 0.05):
        self.set_risk_tolerance(self.state.risk_tolerance + step)

    def be_more_conservative(self, step: float = 0.05):
        self.set_risk_tolerance(self.state.risk_tolerance - step)

    def set_opportunity_bias(self, value: float):
        self.state.opportunity_bias = min(max(value, 0.0), 1.0)

    def exploit_opportunities_more(self, step: float = 0.05):
        self.set_opportunity_bias(self.state.opportunity_bias + step)

    def exploit_opportunities_less(self, step: float = 0.05):
        self.set_opportunity_bias(self.state.opportunity_bias - step)

    def prioritize_learning_windows(self, step: float = 0.05):
        self.state.learning_window_bias = min(self.state.learning_window_bias + step, 1.0)

    def ignore_learning_windows(self, step: float = 0.05):
        self.state.learning_window_bias = max(self.state.learning_window_bias - step, 0.0)

    def update_from_brain(self, env_level: Environment,
                          opportunity_score: float,
                          risk_score: float,
                          anticipation: str,
                          auto_mission: Mission):
        self.state.environment = env_level
        self.state.opportunity_score = float(opportunity_score)
        self.state.risk_score = float(risk_score)
        self.state.anticipation = anticipation
        if self.state.mission_override_active:
            self.state.effective_mission = self.state.mission
        else:
            self.state.effective_mission = auto_mission


@dataclass
class PredictiveState:
    anomaly_risk: float = 0.0
    drive_risk: float = 0.0
    hive_risk: float = 0.0
    collective_health_score: float = 0.5
    health_trend: HealthTrend = HealthTrend.UNKNOWN
    forecast_summary: str = "No forecast"
    horizon: PredictionHorizon = PredictionHorizon.SHORT
    anomaly_sensitivity: float = 0.5
    hive_weight: float = 0.5


class PredictiveIntelligencePanel:
    def __init__(self):
        self.state = PredictiveState()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "anomaly_risk": self.state.anomaly_risk,
            "drive_risk": self.state.drive_risk,
            "hive_risk": self.state.hive_risk,
            "collective_health_score": self.state.collective_health_score,
            "health_trend": self.state.health_trend.name,
            "forecast_summary": self.state.forecast_summary,
            "horizon": self.state.horizon.name,
            "anomaly_sensitivity": self.state.anomaly_sensitivity,
            "hive_weight": self.state.hive_weight,
        }

    def set_horizon(self, horizon: PredictionHorizon):
        self.state.horizon = horizon

    def set_short_horizon(self): self.set_horizon(PredictionHorizon.SHORT)
    def set_medium_horizon(self): self.set_horizon(PredictionHorizon.MEDIUM)
    def set_long_horizon(self): self.set_horizon(PredictionHorizon.LONG)

    def set_anomaly_sensitivity(self, value: float):
        self.state.anomaly_sensitivity = min(max(value, 0.0), 1.0)

    def increase_anomaly_sensitivity(self, step: float = 0.05):
        self.set_anomaly_sensitivity(self.state.anomaly_sensitivity + step)

    def decrease_anomaly_sensitivity(self, step: float = 0.05):
        self.set_anomaly_sensitivity(self.state.anomaly_sensitivity - step)

    def set_hive_weight(self, value: float):
        self.state.hive_weight = min(max(value, 0.0), 1.0)

    def prioritize_hive_signals(self, step: float = 0.05):
        self.set_hive_weight(self.state.hive_weight + step)

    def prioritize_local_signals(self, step: float = 0.05):
        self.set_hive_weight(self.state.hive_weight - step)

    def update_from_forecast(self,
                             anomaly_risk: float,
                             drive_risk: float,
                             hive_risk: float,
                             collective_health_score: float,
                             health_trend: HealthTrend,
                             summary: str):
        self.state.anomaly_risk = float(anomaly_risk)
        self.state.drive_risk = float(drive_risk)
        self.state.hive_risk = float(hive_risk)
        self.state.collective_health_score = float(collective_health_score)
        self.state.health_trend = health_trend
        self.state.forecast_summary = summary


@dataclass
class HiveState:
    collective_risk_score: float = 0.0
    hive_density: float = 0.0
    node_agreement: float = 0.5
    divergence_patterns: List[str] = field(default_factory=list)
    hive_sync_mode: HiveSyncMode = HiveSyncMode.CONSERVATIVE
    consensus_weight: float = 0.5
    propagate_settings: bool = False
    isolated: bool = False


class CollectiveHealthHiveInfluence:
    def __init__(self):
        self.state = HiveState()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "collective_risk_score": self.state.collective_risk_score,
            "hive_density": self.state.hive_density,
            "node_agreement": self.state.node_agreement,
            "divergence_patterns": list(self.state.divergence_patterns),
            "hive_sync_mode": self.state.hive_sync_mode.name,
            "consensus_weight": self.state.consensus_weight,
            "propagate_settings": self.state.propagate_settings,
            "isolated": self.state.isolated,
        }

    def set_hive_sync_mode(self, mode: HiveSyncMode):
        self.state.hive_sync_mode = mode

    def aggressive_sync(self): self.set_hive_sync_mode(HiveSyncMode.AGGRESSIVE)
    def conservative_sync(self): self.set_hive_sync_mode(HiveSyncMode.CONSERVATIVE)
    def local_only(self): self.set_hive_sync_mode(HiveSyncMode.LOCAL_ONLY)

    def set_consensus_weight(self, value: float):
        self.state.consensus_weight = min(max(value, 0.0), 1.0)

    def trust_hive_more(self, step: float = 0.05):
        self.set_consensus_weight(self.state.consensus_weight + step)

    def trust_hive_less(self, step: float = 0.05):
        self.set_consensus_weight(self.state.consensus_weight - step)

    def propagate_my_settings(self):
        self.state.propagate_settings = True
        self.state.isolated = False

    def isolate_this_node(self):
        self.state.isolated = True
        self.state.propagate_settings = False

    def update_from_hive(self,
                         collective_risk: float,
                         hive_density: float,
                         node_agreement: float,
                         divergence_patterns: List[str]):
        self.state.collective_risk_score = float(collective_risk)
        self.state.hive_density = float(hive_density)
        self.state.node_agreement = float(node_agreement)
        self.state.divergence_patterns = list(divergence_patterns)


class CommandBar:
    def __init__(self):
        self.pending_commands: List[Command] = []

    def _queue(self, ctype: CommandType, **kwargs):
        self.pending_commands.append(Command(ctype, kwargs))

    def stabilize_system(self): self._queue(CommandType.STABILIZE_SYSTEM)
    def enter_high_alert_mode(self): self._queue(CommandType.HIGH_ALERT_MODE)
    def begin_learning_cycle(self): self._queue(CommandType.BEGIN_LEARNING_CYCLE)
    def optimize_performance(self): self._queue(CommandType.OPTIMIZE_PERFORMANCE)
    def purge_anomaly_memory(self): self._queue(CommandType.PURGE_ANOMALY_MEMORY)
    def rebuild_predictive_model(self): self._queue(CommandType.REBUILD_PREDICTIVE_MODEL)
    def reset_situational_cortex(self): self._queue(CommandType.RESET_SITUATIONAL_CORTEX)
    def snapshot_brain_state(self, label: Optional[str] = None):
        self._queue(CommandType.SNAPSHOT_BRAIN_STATE, label=label)
    def rollback_previous_state(self): self._queue(CommandType.ROLLBACK_PREVIOUS_STATE)

    def fetch_commands(self) -> List[Command]:
        cmds = self.pending_commands
        self.pending_commands = []
        return cmds


class ASIDialogueInterface:
    def __init__(self, brain_runtime_ref: Any):
        self.brain = brain_runtime_ref

    def why_did_you_choose_this_mission(self) -> DialogueAnswer:
        mission = getattr(self.brain, "current_mission", "UNKNOWN")
        return DialogueAnswer(
            intent=f"I chose mission {mission} based on risk, opportunity, and your overrides.",
            confidence=0.78,
            alternatives=[
                "Switch to PROTECT under higher risk.",
                "Switch to OPTIMIZE if risk remains low."
            ],
            risks=[
                "Current mission may underutilize learning windows.",
                "Aggressive mission change could destabilize resources."
            ],
            expected_outcomes=[
                "Maintain alignment with your configured mission priorities.",
                "Adapt quickly if telemetry crosses thresholds.",
            ],
        )

    def what_are_you_predicting_next(self) -> DialogueAnswer:
        forecast = getattr(self.brain, "last_forecast", {})
        summary = forecast.get("summary", "No forecast available.")
        return DialogueAnswer(
            intent=f"Next, I expect: {summary}",
            confidence=float(forecast.get("confidence", 0.6)),
            alternatives=forecast.get("alternatives", []),
            risks=forecast.get("risks", []),
            expected_outcomes=forecast.get("outcomes", []),
        )

    def what_are_you_uncertain_about(self) -> DialogueAnswer:
        uncertainties = getattr(self.brain, "uncertainties", ["Insufficient telemetry granularity."])
        return DialogueAnswer(
            intent="I am uncertain about several factors in the current environment.",
            confidence=0.5,
            alternatives=[],
            risks=["Hidden anomalies may bypass current sensitivity settings."],
            expected_outcomes=uncertainties,
        )

    def what_do_you_need_from_me(self) -> DialogueAnswer:
        return DialogueAnswer(
            intent="I need clearer mission priorities and updated risk tolerance from you.",
            confidence=0.9,
            alternatives=["You can also tune learning window bias and hive trust."],
            risks=["Without updates, my decisions may diverge from your intent under stress."],
            expected_outcomes=[
                "Tighter alignment between your priorities and my actions.",
                "More accurate mission selection under changing conditions.",
            ],
        )

    def explain_your_reasoning(self) -> DialogueAnswer:
        explanation = getattr(self.brain, "last_explanation", "No explanation recorded.")
        return DialogueAnswer(
            intent=explanation,
            confidence=0.8,
            alternatives=[],
            risks=["Reasoning is based on current models; model drift could degrade validity."],
            expected_outcomes=["You can audit and correct my priorities based on this trace."],
        )


# ============================================================
# 5. Stub organs (so this file runs standalone)
# ============================================================

class JudgmentCortex:
    """Very simple stub: classifies risk level and verdict."""

    def evaluate(self, state: dict, forecast: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        if risk < 0.3:
            risk_level = "low"
        elif risk < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        verdict = "normal" if risk_level != "high" else "risky"
        return {"verdict": verdict, "risk_level": risk_level}


class ConfidenceEngine:
    """Combines risk and anomaly to give a 0..1 confidence and strategy."""

    def assess(self, state: dict, forecast: dict, judgment: dict, mv_result: dict) -> dict:
        base_risk = float(state.get("risk_score", 0.0))
        anomaly_risk = float(forecast.get("anomaly_risk", base_risk))
        mv_anomaly = mv_result.get("anomaly_score")
        if mv_anomaly is not None:
            anomaly_risk = 0.7 * anomaly_risk + 0.3 * float(mv_anomaly)

        confidence = 1.0 - anomaly_risk
        confidence = max(0.0, min(1.0, confidence))

        if confidence > 0.75:
            strategy = "trust"
        elif confidence > 0.4:
            strategy = "verify"
        else:
            strategy = "cautious"

        return {"score": confidence, "strategy": strategy}


class DataPhysicsEngine:
    """Very simple 'data physics' stub based on risk/opportunity."""

    def analyze_flow(self, state: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        opp = float(state.get("opportunity_score", 0.0))
        pressure = risk * 0.7 + opp * 0.3
        bottlenecks = []
        if pressure > 0.6:
            bottlenecks.append("IO")
        if risk > 0.8:
            bottlenecks.append("CPU")
        return {"bottlenecks": bottlenecks, "pressure": pressure}


class CollectiveHealthScore:
    """Aggregates organ readiness + forecast into a health score."""

    def compute(self, organ_health: dict, forecast: dict, confidence: dict) -> dict:
        ready_count = sum(1 for v in organ_health.values() if v["ready"])
        total = len(organ_health) if organ_health else 1
        readiness = ready_count / total

        conf = float(confidence.get("score", 0.5))
        anomaly_risk = float(forecast.get("anomaly_risk", 0.5))

        score = 0.5 * readiness + 0.4 * conf + 0.1 * (1.0 - anomaly_risk)
        score = max(0.0, min(1.0, score))

        status = "good"
        if score < 0.3:
            status = "critical"
        elif score < 0.6:
            status = "degraded"

        return {"score": score, "status": status}


class AlteredStatesManager:
    """Selects 'mode' and suggests mission based on health/risk."""

    def select_mode(self, state: dict, health: dict, forecast: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        health_score = float(health.get("score", 0.5))

        if risk > 0.7:
            mode = "high_alert"
            mission = "PROTECT"
        elif health_score < 0.4:
            mode = "self_repair"
            mission = "STABILITY"
        elif forecast.get("anomaly_risk", 0.0) < 0.3:
            mode = "optimize"
            mission = "OPTIMIZE"
        else:
            mode = "baseline"
            mission = "PROTECT"

        return {"mode": mode, "mission": mission}


class PredictiveIntelEngine:
    """Simple forecasting stub used as predictive_intel organ."""

    def forecast(self, state: dict, flow: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        pressure = float(flow.get("pressure", 0.0))

        anomaly_risk = min(1.0, (risk + pressure) / 2.0)
        drive_risk = risk * 0.8
        hive_risk = risk * 0.7
        health_trend = "STABLE"

        summary = f"Risk={risk:.2f}, pressure={pressure:.2f}, anomaly≈{anomaly_risk:.2f}"

        return {
            "anomaly_risk": anomaly_risk,
            "drive_risk": drive_risk,
            "hive_risk": hive_risk,
            "collective_health_score": 0.7,
            "health_trend": health_trend,
            "summary": summary,
            "branches": [],
            "risks": [],
            "expected_outcomes": [],
        }


# ============================================================
# 6. Autoloader infrastructure for organs
# ============================================================

class OrganWrapper:
    def __init__(self, name: str, instance: Any, failed: bool, error: str | None):
        self.name = name
        self.instance = instance
        self.failed = failed
        self.error = error

    def is_ready(self) -> bool:
        return (not self.failed) and self.instance is not None


def safe_call(fun: Callable, default: Any, label: str) -> Any:
    try:
        return fun()
    except Exception as e:
        print(f"[{label}] ERROR: {e}")
        traceback.print_exc()
        return default


ORGAN_SPECS = {
    "judgment_cortex": {
        "module": __name__,
        "class": "JudgmentCortex",
    },
    "confidence_engine": {
        "module": __name__,
        "class": "ConfidenceEngine",
    },
    "data_physics": {
        "module": __name__,
        "class": "DataPhysicsEngine",
    },
    "collective_health": {
        "module": __name__,
        "class": "CollectiveHealthScore",
    },
    "altered_states": {
        "module": __name__,
        "class": "AlteredStatesManager",
    },
    "movidius_inference": {
        "module": __name__,
        "class": "MovidiusInferenceEngine",
    },
    "predictive_intel": {
        "module": __name__,
        "class": "PredictiveIntelEngine",
    },
}


class BrainAutoloader:
    def __init__(self):
        self.organs: Dict[str, OrganWrapper] = {}

    def load_organs(self):
        for name, spec in ORGAN_SPECS.items():
            module_name = spec["module"]
            class_name = spec["class"]
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                instance = cls()
                self.organs[name] = OrganWrapper(name, instance, False, None)
                print(f"[AUTOLOADER] Loaded organ: {name}")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self.organs[name] = OrganWrapper(name, None, True, err)
                print(f"[AUTOLOADER] FAILED organ: {name} -> {err}")

    def get(self, name: str) -> Any | None:
        ow = self.organs.get(name)
        if ow and ow.is_ready():
            return ow.instance
        return None

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            name: {
                "ready": ow.is_ready(),
                "failed": ow.failed,
                "error": ow.error,
            }
            for name, ow in self.organs.items()
        }


# ============================================================
# 7. HybridBrainCore – using panels + organs
# ============================================================

class HybridBrainCore:
    def __init__(self,
                 situational: SituationalAwarenessCortex,
                 predictive_panel: PredictiveIntelligencePanel,
                 hive_panel: CollectiveHealthHiveInfluence,
                 command_bar: CommandBar):
        self.situational = situational
        self.predictive_panel = predictive_panel
        self.hive_panel = hive_panel
        self.command_bar = command_bar

        self.current_mission: str = "PROTECT"
        self.last_forecast: Dict[str, Any] = {}
        self.last_explanation: str = "No decisions yet."
        self.uncertainties: List[str] = []
        self.brain_state_history: List[Dict[str, Any]] = []

    def _apply_commands(self):
        cmds = self.command_bar.fetch_commands()
        for cmd in cmds:
            if cmd.type == CommandType.STABILIZE_SYSTEM:
                self.situational.force_protect()
                self.situational.be_more_conservative(0.2)
            elif cmd.type == CommandType.HIGH_ALERT_MODE:
                self.situational.force_protect()
                self.predictive_panel.increase_anomaly_sensitivity(0.2)
            elif cmd.type == CommandType.BEGIN_LEARNING_CYCLE:
                self.situational.force_learn()
                self.situational.prioritize_learning_windows(0.3)
            elif cmd.type == CommandType.OPTIMIZE_PERFORMANCE:
                self.situational.force_optimize()
            elif cmd.type == CommandType.PURGE_ANOMALY_MEMORY:
                pass
            elif cmd.type == CommandType.REBUILD_PREDICTIVE_MODEL:
                pass
            elif cmd.type == CommandType.RESET_SITUATIONAL_CORTEX:
                self.situational = SituationalAwarenessCortex()
            elif cmd.type == CommandType.SNAPSHOT_BRAIN_STATE:
                label = cmd.args.get("label", "snapshot")
                self.brain_state_history.append({"label": label})
            elif cmd.type == CommandType.ROLLBACK_PREVIOUS_STATE:
                if self.brain_state_history:
                    _ = self.brain_state_history.pop()

    def decide(self,
               state: Dict[str, Any],
               flow: Dict[str, Any],
               forecast: Dict[str, Any],
               judgment: Dict[str, Any],
               confidence: Dict[str, Any],
               health: Dict[str, Any],
               mode: Dict[str, Any],
               mv_result: Dict[str, Any]) -> Dict[str, Any]:

        self.current_mission = mode.get("mission", "PROTECT")
        self.last_forecast = {
            "summary": forecast.get("summary", "Unknown"),
            "confidence": confidence.get("score", 0.6),
            "alternatives": forecast.get("branches", []),
            "risks": forecast.get("risks", []),
            "outcomes": forecast.get("expected_outcomes", []),
        }
        self.last_explanation = (
            f"Acted under mission {self.current_mission} "
            f"with risk={state.get('risk_score', 'n/a')} "
            f"and opportunity={state.get('opportunity_score', 'n/a')}."
        )

        self._apply_commands()

        effective_mission = self.situational.state.effective_mission
        risk = state.get("risk_score", 0.0)
        risk_tol = self.situational.state.risk_tolerance

        action = "observe"

        if effective_mission == Mission.PROTECT:
            if risk > risk_tol:
                action = "defend"
            else:
                action = "harden_passively"
        elif effective_mission == Mission.LEARN:
            if risk < risk_tol:
                action = "explore"
            else:
                action = "cautious_learn"
        elif effective_mission == Mission.OPTIMIZE:
            if risk < risk_tol:
                action = "optimize_aggressively"
            else:
                action = "optimize_conservatively"
        elif effective_mission == Mission.STABILITY:
            action = "stabilize_resources"

        return {
            "action": action,
            "mission": effective_mission.name,
            "reason": self.last_explanation,
        }


# ============================================================
# 8. BrainRuntime – full loop
# ============================================================

class BrainRuntime:
    def __init__(self, loader: BrainAutoloader):
        self.loader = loader

        self.situational = SituationalAwarenessCortex()
        self.predictive_panel = PredictiveIntelligencePanel()
        self.hive_panel = CollectiveHealthHiveInfluence()
        self.command_bar = CommandBar()

        self.hybrid_brain = HybridBrainCore(
            self.situational,
            self.predictive_panel,
            self.hive_panel,
            self.command_bar,
        )

        self.dialogue = ASIDialogueInterface(self.hybrid_brain)

    def tick(self, raw_telemetry: dict) -> dict:
        env = Environment.CALM
        opportunity_score = float(raw_telemetry.get("opportunity", 0.3))
        risk_score = float(raw_telemetry.get("risk", 0.2))
        anticipation = "Normal operation expected."
        auto_mission = Mission.PROTECT

        self.situational.update_from_brain(
            env_level=env,
            opportunity_score=opportunity_score,
            risk_score=risk_score,
            anticipation=anticipation,
            auto_mission=auto_mission,
        )

        state_for_brain = {
            "environment": env.name,
            "opportunity_score": opportunity_score,
            "risk_score": risk_score,
            "anticipation": anticipation,
            "risk_tolerance": self.situational.state.risk_tolerance,
        }

        dp = self.loader.get("data_physics")
        flow = safe_call(
            lambda: dp.analyze_flow(state_for_brain),
            default={"bottlenecks": [], "pressure": 0.0},
            label="data_physics",
        ) if dp else {"bottlenecks": [], "pressure": 0.0}

        pe = self.loader.get("predictive_intel")
        if pe:
            raw_forecast = safe_call(
                lambda: pe.forecast(state_for_brain, flow),
                default={"anomaly_risk": risk_score, "drive_risk": risk_score * 0.8,
                         "hive_risk": risk_score * 0.7, "collective_health_score": 0.7,
                         "summary": "No forecast"},
                label="predictive_intel",
            )
        else:
            raw_forecast = {"anomaly_risk": risk_score, "drive_risk": risk_score * 0.8,
                            "hive_risk": risk_score * 0.7, "collective_health_score": 0.7,
                            "summary": "Static forecast from risk score"}

        anomaly_risk = float(raw_forecast.get("anomaly_risk", risk_score))
        drive_risk = float(raw_forecast.get("drive_risk", risk_score * 0.8))
        hive_risk = float(raw_forecast.get("hive_risk", risk_score * 0.7))
        collective_health = float(raw_forecast.get("collective_health_score", 0.7))
        health_trend = HealthTrend.STABLE

        self.predictive_panel.update_from_forecast(
            anomaly_risk=anomaly_risk,
            drive_risk=drive_risk,
            hive_risk=hive_risk,
            collective_health_score=collective_health,
            health_trend=health_trend,
            summary=raw_forecast.get("summary", "No forecast"),
        )

        forecast_for_brain = {
            "anomaly_risk": anomaly_risk,
            "drive_risk": drive_risk,
            "hive_risk": hive_risk,
            "summary": raw_forecast.get("summary", ""),
            "branches": raw_forecast.get("branches", []),
            "risks": raw_forecast.get("risks", []),
            "expected_outcomes": raw_forecast.get("expected_outcomes", []),
        }

        jc = self.loader.get("judgment_cortex")
        judgment = safe_call(
            lambda: jc.evaluate(state_for_brain, forecast_for_brain),
            default={"verdict": "unknown", "risk_level": "unknown"},
            label="judgment_cortex",
        ) if jc else {"verdict": "unknown", "risk_level": "unknown"}

        mv = self.loader.get("movidius_inference")
        if mv:
            mv_result = safe_call(
                lambda: mv.accelerate(state_for_brain, forecast_for_brain),
                default={"used": False, "backend": "NONE", "anomaly_score": None,
                         "risk_adjustment": 0.0, "metrics": {}},
                label="movidius_inference",
            )
        else:
            mv_result = {"used": False, "backend": "NONE", "anomaly_score": None,
                         "risk_adjustment": 0.0, "metrics": {}}

        ce = self.loader.get("confidence_engine")
        confidence = safe_call(
            lambda: ce.assess(state_for_brain, forecast_for_brain, judgment, mv_result),
            default={"score": 0.5, "strategy": "verify"},
            label="confidence_engine",
        ) if ce else {"score": 0.5, "strategy": "verify"}

        ch = self.loader.get("collective_health")
        health = safe_call(
            lambda: ch.compute(self.loader.health_snapshot(), forecast_for_brain, confidence),
            default={"score": collective_health, "status": "degraded"},
            label="collective_health",
        ) if ch else {"score": collective_health, "status": "degraded"}

        self.hive_panel.update_from_hive(
            collective_risk=health["score"],
            hive_density=raw_forecast.get("hive_density", 0.5),
            node_agreement=raw_forecast.get("node_agreement", 0.6),
            divergence_patterns=raw_forecast.get("divergence_patterns", []),
        )

        as_mod = self.loader.get("altered_states")
        mode = safe_call(
            lambda: as_mod.select_mode(state_for_brain, health, forecast_for_brain),
            default={"mode": "baseline", "mission": self.situational.state.effective_mission.name},
            label="altered_states",
        ) if as_mod else {"mode": "baseline", "mission": self.situational.state.effective_mission.name}

        decision = self.hybrid_brain.decide(
            state_for_brain,
            flow,
            forecast_for_brain,
            judgment,
            confidence,
            health,
            mode,
            mv_result,
        )

        return {
            "situational": self.situational.get_snapshot(),
            "predictive": self.predictive_panel.get_snapshot(),
            "hive": self.hive_panel.get_snapshot(),
            "judgment": judgment,
            "movidius": mv_result,
            "confidence": confidence,
            "health": health,
            "mode": mode,
            "decision": decision,
        }


# ============================================================
# 9. Glass Protocol UI (PySide6)
# ============================================================

def make_kv_row(key: str, value_widget: QWidget) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(key)
    label.setStyleSheet("color: #CCCCCC; font-weight: bold;")
    layout.addWidget(label)
    layout.addWidget(value_widget, 1)
    return row


class SituationalPanel(QGroupBox):
    def __init__(self, brain: BrainRuntime):
        super().__init__("Situational Awareness Cortex")
        self.brain = brain
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.mission_label = QLabel("-")
        self.env_label = QLabel("-")
        self.opp_label = QLabel("0.0")
        self.risk_label = QLabel("0.0")
        self.anticipation_label = QLabel("-")

        for lbl in (self.mission_label, self.env_label,
                    self.opp_label, self.risk_label, self.anticipation_label):
            lbl.setStyleSheet("color: #EEEEEE;")

        layout.addWidget(make_kv_row("Effective mission:", self.mission_label))
        layout.addWidget(make_kv_row("Environment:", self.env_label))
        layout.addWidget(make_kv_row("Opportunity:", self.opp_label))
        layout.addWidget(make_kv_row("Risk:", self.risk_label))
        layout.addWidget(make_kv_row("Anticipation:", self.anticipation_label))

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setSpacing(4)
        for_m = QPushButton("Force PROTECT")
        for_l = QPushButton("Force LEARN")
        for_o = QPushButton("Force OPTIMIZE")
        auto_b = QPushButton("AUTO")

        for b in (for_m, for_l, for_o, auto_b):
            b.setStyleSheet("QPushButton { background-color: #333333; color: #EEEEEE; border-radius: 6px; padding: 4px 8px; }"
                            "QPushButton:hover { background-color: #555555; }")
            b.setCursor(Qt.PointingHandCursor)

        for_m.clicked.connect(self.force_protect)
        for_l.clicked.connect(self.force_learn)
        for_o.clicked.connect(self.force_optimize)
        auto_b.clicked.connect(self.return_auto)

        btn_layout.addWidget(for_m)
        btn_layout.addWidget(for_l)
        btn_layout.addWidget(for_o)
        btn_layout.addWidget(auto_b)
        layout.addWidget(btn_row)

        rt_label = QLabel("Risk tolerance")
        rt_label.setStyleSheet("color: #CCCCCC;")
        self.rt_slider = QSlider(Qt.Horizontal)
        self.rt_slider.setRange(0, 100)
        self.rt_slider.setValue(50)
        self.rt_slider.valueChanged.connect(self.on_rt_changed)

        layout.addWidget(rt_label)
        layout.addWidget(self.rt_slider)

    def force_protect(self):
        self.brain.situational.force_protect()

    def force_learn(self):
        self.brain.situational.force_learn()

    def force_optimize(self):
        self.brain.situational.force_optimize()

    def return_auto(self):
        self.brain.situational.return_to_auto()

    def on_rt_changed(self, value: int):
        self.brain.situational.set_risk_tolerance(value / 100.0)

    def update_from_snapshot(self, snap: dict):
        self.mission_label.setText(snap.get("effective_mission", "-"))
        self.env_label.setText(snap.get("environment", "-"))
        self.opp_label.setText(f"{snap.get('opportunity_score', 0.0):.2f}")
        self.risk_label.setText(f"{snap.get('risk_score', 0.0):.2f}")
        self.anticipation_label.setText(snap.get("anticipation", "-"))

        rt = snap.get("risk_tolerance", 0.5)
        self.rt_slider.blockSignals(True)
        self.rt_slider.setValue(int(rt * 100))
        self.rt_slider.blockSignals(False)


class PredictivePanel(QGroupBox):
    def __init__(self, brain: BrainRuntime):
        super().__init__("Predictive Intelligence")
        self.brain = brain
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.anomaly_label = QLabel("0.0")
        self.drive_label = QLabel("0.0")
        self.hive_label = QLabel("0.0")
        self.health_label = QLabel("0.0")
        self.trend_label = QLabel("-")
        self.summary_label = QLabel("-")
        self.summary_label.setWordWrap(True)

        for lbl in (self.anomaly_label, self.drive_label, self.hive_label,
                    self.health_label, self.trend_label, self.summary_label):
            lbl.setStyleSheet("color: #EEEEEE;")

        layout.addWidget(make_kv_row("Anomaly risk:", self.anomaly_label))
        layout.addWidget(make_kv_row("Drive risk:", self.drive_label))
        layout.addWidget(make_kv_row("Hive risk:", self.hive_label))
        layout.addWidget(make_kv_row("Collective health:", self.health_label))
        layout.addWidget(make_kv_row("Health trend:", self.trend_label))
        layout.addWidget(make_kv_row("Forecast:", self.summary_label))

        hbox = QWidget()
        h_layout = QHBoxLayout(hbox)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(6)
        h_label = QLabel("Horizon:")
        h_label.setStyleSheet("color: #CCCCCC;")
        self.h_combo = QComboBox()
        self.h_combo.addItems(["Short", "Medium", "Long"])
        self.h_combo.currentIndexChanged.connect(self.on_horizon_changed)
        h_layout.addWidget(h_label)
        h_layout.addWidget(self.h_combo, 1)
        layout.addWidget(hbox)

        s_label = QLabel("Anomaly sensitivity")
        s_label.setStyleSheet("color: #CCCCCC;")
        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setRange(0, 100)
        self.s_slider.setValue(50)
        self.s_slider.valueChanged.connect(self.on_sensitivity_changed)

        layout.addWidget(s_label)
        layout.addWidget(self.s_slider)

        w_label = QLabel("Hive vs local weighting")
        w_label.setStyleSheet("color: #CCCCCC;")
        self.w_slider = QSlider(Qt.Horizontal)
        self.w_slider.setRange(0, 100)
        self.w_slider.setValue(50)
        self.w_slider.valueChanged.connect(self.on_weighting_changed)

        layout.addWidget(w_label)
        layout.addWidget(self.w_slider)

    def on_horizon_changed(self, idx: int):
        if idx == 0:
            self.brain.predictive_panel.set_short_horizon()
        elif idx == 1:
            self.brain.predictive_panel.set_medium_horizon()
        else:
            self.brain.predictive_panel.set_long_horizon()

    def on_sensitivity_changed(self, value: int):
        self.brain.predictive_panel.set_anomaly_sensitivity(value / 100.0)

    def on_weighting_changed(self, value: int):
        self.brain.predictive_panel.set_hive_weight(value / 100.0)

    def update_from_snapshot(self, snap: dict):
        self.anomaly_label.setText(f"{snap.get('anomaly_risk', 0.0):.2f}")
        self.drive_label.setText(f"{snap.get('drive_risk', 0.0):.2f}")
        self.hive_label.setText(f"{snap.get('hive_risk', 0.0):.2f}")
        self.health_label.setText(f"{snap.get('collective_health_score', 0.0):.2f}")
        self.trend_label.setText(snap.get("health_trend", "-"))
        self.summary_label.setText(snap.get("forecast_summary", "-"))

        sens = snap.get("anomaly_sensitivity", 0.5)
        hive_w = snap.get("hive_weight", 0.5)
        self.s_slider.blockSignals(True)
        self.s_slider.setValue(int(sens * 100))
        self.s_slider.blockSignals(False)
        self.w_slider.blockSignals(True)
        self.w_slider.setValue(int(hive_w * 100))
        self.w_slider.blockSignals(False)

        horizon = snap.get("horizon", "SHORT")
        idx = {"SHORT": 0, "MEDIUM": 1, "LONG": 2}.get(horizon.upper(), 0)
        self.h_combo.blockSignals(True)
        self.h_combo.setCurrentIndex(idx)
        self.h_combo.blockSignals(False)


class HivePanel(QGroupBox):
    def __init__(self, brain: BrainRuntime):
        super().__init__("Collective Health & Hive Influence")
        self.brain = brain
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.collective_label = QLabel("0.0")
        self.density_label = QLabel("0.0")
        self.agreement_label = QLabel("0.0")
        self.divergence_label = QLabel("-")
        self.mode_label = QLabel("-")

        for lbl in (self.collective_label, self.density_label,
                    self.agreement_label, self.divergence_label, self.mode_label):
            lbl.setStyleSheet("color: #EEEEEE;")

        layout.addWidget(make_kv_row("Collective risk:", self.collective_label))
        layout.addWidget(make_kv_row("Hive density:", self.density_label))
        layout.addWidget(make_kv_row("Node agreement:", self.agreement_label))
        layout.addWidget(make_kv_row("Divergence:", self.divergence_label))
        layout.addWidget(make_kv_row("Sync mode:", self.mode_label))

        sync_row = QWidget()
        sync_layout = QHBoxLayout(sync_row)
        sync_layout.setContentsMargins(0, 0, 0, 0)
        sync_layout.setSpacing(6)
        sync_label = QLabel("Sync mode:")
        sync_label.setStyleSheet("color: #CCCCCC;")
        self.sync_combo = QComboBox()
        self.sync_combo.addItems(["Aggressive", "Conservative", "Local-only"])
        self.sync_combo.currentIndexChanged.connect(self.on_sync_changed)
        sync_layout.addWidget(sync_label)
        sync_layout.addWidget(self.sync_combo, 1)
        layout.addWidget(sync_row)

        cons_label = QLabel("Consensus weight (trust hive)")
        cons_label.setStyleSheet("color: #CCCCCC;")
        self.cons_slider = QSlider(Qt.Horizontal)
        self.cons_slider.setRange(0, 100)
        self.cons_slider.setValue(50)
        self.cons_slider.valueChanged.connect(self.on_consensus_changed)
        layout.addWidget(cons_label)
        layout.addWidget(self.cons_slider)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setSpacing(4)
        prop_btn = QPushButton("Propagate settings")
        iso_btn = QPushButton("Isolate node")

        for b in (prop_btn, iso_btn):
            b.setStyleSheet("QPushButton { background-color: #333333; color: #EEEEEE; border-radius: 6px; padding: 4px 8px; }"
                            "QPushButton:hover { background-color: #555555; }")
            b.setCursor(Qt.PointingHandCursor)

        prop_btn.clicked.connect(self.propagate)
        iso_btn.clicked.connect(self.isolate)

        btn_layout.addWidget(prop_btn)
        btn_layout.addWidget(iso_btn)
        layout.addWidget(btn_row)

    def on_sync_changed(self, idx: int):
        if idx == 0:
            self.brain.hive_panel.aggressive_sync()
        elif idx == 1:
            self.brain.hive_panel.conservative_sync()
        else:
            self.brain.hive_panel.local_only()

    def on_consensus_changed(self, value: int):
        self.brain.hive_panel.set_consensus_weight(value / 100.0)

    def propagate(self):
        self.brain.hive_panel.propagate_my_settings()

    def isolate(self):
        self.brain.hive_panel.isolate_this_node()

    def update_from_snapshot(self, snap: dict):
        self.collective_label.setText(f"{snap.get('collective_risk_score', 0.0):.2f}")
        self.density_label.setText(f"{snap.get('hive_density', 0.0):.2f}")
        self.agreement_label.setText(f"{snap.get('node_agreement', 0.0):.2f}")
        self.divergence_label.setText(", ".join(snap.get("divergence_patterns", [])) or "-")
        self.mode_label.setText(snap.get("hive_sync_mode", "-"))

        cw = snap.get("consensus_weight", 0.5)
        self.cons_slider.blockSignals(True)
        self.cons_slider.setValue(int(cw * 100))
        self.cons_slider.blockSignals(False)

        mode = snap.get("hive_sync_mode", "CONSERVATIVE").upper()
        idx = {"AGGRESSIVE": 0, "CONSERVATIVE": 1, "LOCAL_ONLY": 2}.get(mode, 1)
        self.sync_combo.blockSignals(True)
        self.sync_combo.setCurrentIndex(idx)
        self.sync_combo.blockSignals(False)


class MovidiusPanel(QGroupBox):
    def __init__(self):
        super().__init__("Movidius / OpenVINO Engine")
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.backend_label = QLabel("-")
        self.used_label = QLabel("-")
        self.latency_label = QLabel("0.0 ms")
        self.fps_label = QLabel("0.0")
        self.queue_label = QLabel("0")
        self.total_label = QLabel("0")

        for lbl in (self.backend_label, self.used_label, self.latency_label,
                    self.fps_label, self.queue_label, self.total_label):
            lbl.setStyleSheet("color: #EEEEEE;")

        layout.addWidget(make_kv_row("Backend:", self.backend_label))
        layout.addWidget(make_kv_row("Used this tick:", self.used_label))
        layout.addWidget(make_kv_row("Avg latency:", self.latency_label))
        layout.addWidget(make_kv_row("FPS:", self.fps_label))
        layout.addWidget(make_kv_row("Queue depth:", self.queue_label))
        layout.addWidget(make_kv_row("Total inferences:", self.total_label))

    def update_from_mv(self, mv: dict):
        self.backend_label.setText(mv.get("backend", "NONE"))
        self.used_label.setText("Yes" if mv.get("used", False) else "No")
        metrics = mv.get("metrics", {}) or {}
        self.latency_label.setText(f"{metrics.get("avg_latency_ms", 0.0):.2f} ms")
        self.fps_label.setText(f"{metrics.get("fps", 0.0):.2f}")
        self.queue_label.setText(str(metrics.get("queue_depth", 0)))
        self.total_label.setText(str(metrics.get("total_inferences", 0)))


class CommandPanel(QGroupBox):
    def __init__(self, cmd: CommandBar):
        super().__init__("Command Bar")
        self.cmd = cmd
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QGridLayout(self)
        layout.setSpacing(6)

        def make_btn(text, cb):
            b = QPushButton(text)
            b.setStyleSheet("QPushButton { background-color: #2F3A4A; color: #FFFFFF; "
                            "border-radius: 8px; padding: 6px 10px; }"
                            "QPushButton:hover { background-color: #40536A; }")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            return b

        buttons = [
            ("Stabilize system", self.cmd.stabilize_system),
            ("High-alert mode", self.cmd.enter_high_alert_mode),
            ("Begin learning cycle", self.cmd.begin_learning_cycle),
            ("Optimize performance", self.cmd.optimize_performance),
            ("Purge anomaly memory", self.cmd.purge_anomaly_memory),
            ("Rebuild predictive model", self.cmd.rebuild_predictive_model),
            ("Reset situational cortex", self.cmd.reset_situational_cortex),
            ("Snapshot brain state", lambda: self.cmd.snapshot_brain_state("manual")),
            ("Rollback previous state", self.cmd.rollback_previous_state),
        ]

        row, col = 0, 0
        for text, cb in buttons:
            layout.addWidget(make_btn(text, cb), row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1


class DialoguePanel(QGroupBox):
    def __init__(self, brain: BrainRuntime):
        super().__init__("ASI Dialogue Window")
        self.brain = brain
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("QTextEdit { background: #111111; color: #EEEEEE; border-radius: 8px; }")
        layout.addWidget(self.text, 1)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setSpacing(4)

        def make_btn(text, cb):
            b = QPushButton(text)
            b.setStyleSheet("QPushButton { background-color: #333333; color: #EEEEEE; border-radius: 6px; padding: 4px 8px; }"
                            "QPushButton:hover { background-color: #555555; }")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            return b

        btn1 = make_btn("Why this mission?", self.ask_why_mission)
        btn2 = make_btn("Predict next", self.ask_predict_next)
        btn3 = make_btn("What uncertain?", self.ask_uncertain)
        btn4 = make_btn("What do you need?", self.ask_need)
        btn5 = make_btn("Explain reasoning", self.ask_explain)

        for b in (btn1, btn2, btn3, btn4, btn5):
            btn_layout.addWidget(b)

        layout.addWidget(btn_row)

    def _append_answer(self, title: str, ans):
        self.text.append(f"=== {title} ===")
        self.text.append(f"Intent: {ans.intent}")
        self.text.append(f"Confidence: {ans.confidence:.2f}")
        if ans.alternatives:
            self.text.append("Alternatives:")
            for alt in ans.alternatives:
                self.text.append(f"  - {alt}")
        if ans.risks:
            self.text.append("Risks:")
            for r in ans.risks:
                self.text.append(f"  - {r}")
        if ans.expected_outcomes:
            self.text.append("Expected outcomes:")
            for o in ans.expected_outcomes:
                self.text.append(f"  - {o}")
        self.text.append("")

    def ask_why_mission(self):
        ans = self.brain.dialogue.why_did_you_choose_this_mission()
        self._append_answer("Why did you choose this mission", ans)

    def ask_predict_next(self):
        ans = self.brain.dialogue.what_are_you_predicting_next()
        self._append_answer("What are you predicting next", ans)

    def ask_uncertain(self):
        ans = self.brain.dialogue.what_are_you_uncertain_about()
        self._append_answer("What are you uncertain about", ans)

    def ask_need(self):
        ans = self.brain.dialogue.what_do_you_need_from_me()
        self._append_answer("What do you need from me", ans)

    def ask_explain(self):
        ans = self.brain.dialogue.explain_your_reasoning()
        self._append_answer("Explain your reasoning", ans)


# ============================================================
# 10. Main window (Glass Protocol layout)
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASI Glass Protocol Cockpit")
        self.resize(1400, 800)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #050911;
            }
            QGroupBox {
                background-color: rgba(20, 30, 50, 180);
                color: #FFFFFF;
                font-weight: bold;
                margin-top: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
            }
            QLabel {
                color: #DDDDDD;
            }
        """)

        loader = BrainAutoloader()
        loader.load_organs()
        self.brain = BrainRuntime(loader)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QGridLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.situ_panel = SituationalPanel(self.brain)
        self.pred_panel = PredictivePanel(self.brain)
        self.hive_panel = HivePanel(self.brain)
        self.mv_panel = MovidiusPanel()
        self.cmd_panel = CommandPanel(self.brain.command_bar)
        self.dialog_panel = DialoguePanel(self.brain)

        main_layout.addWidget(self.situ_panel, 0, 0)
        main_layout.addWidget(self.pred_panel, 0, 1)
        main_layout.addWidget(self.hive_panel, 1, 0)
        main_layout.addWidget(self.mv_panel, 1, 1)
        main_layout.addWidget(self.cmd_panel, 2, 0, 1, 2)
        main_layout.addWidget(self.dialog_panel, 3, 0, 1, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(500)  # ms

    def on_tick(self):
        telemetry = {
            "risk": random.uniform(0.0, 1.0),
            "opportunity": random.uniform(0.0, 1.0),
        }

        snapshot = self.brain.tick(telemetry)

        self.situ_panel.update_from_snapshot(snapshot["situational"])
        self.pred_panel.update_from_snapshot(snapshot["predictive"])
        self.hive_panel.update_from_snapshot(snapshot["hive"])
        self.mv_panel.update_from_mv(snapshot["movidius"])
        # Dialogue is user-driven


# ============================================================
# 11. Entry point
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

