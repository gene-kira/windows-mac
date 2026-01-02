from __future__ import annotations

# ============================================================
# ASI USB Movidius Turbo Brain – All Organs + GUI
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
import json
import hashlib
import socket
import socketserver

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
    QTabWidget,
)
from PySide6.QtGui import QFont

# ============================================================
# 0. OpenVINO / Movidius detection + USB metrics
# ============================================================

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except Exception as e:
    print(f"[AUTOLOADER] OpenVINO import failed: {e}")
    Core = None
    OPENVINO_AVAILABLE = False

try:
    import usb.core
    import usb.util
    PYUSB_AVAILABLE = True
except Exception as e:
    print(f"[USB] pyusb import failed: {e}")
    usb = None
    PYUSB_AVAILABLE = False

MOVIDIUS_USB_IDS = [
    (0x03e7, 0x2485),  # NCS2
    (0x03e7, 0x2150),  # NCS
]


def find_movidius_usb() -> Dict[str, Any]:
    if not PYUSB_AVAILABLE:
        return {
            "present": False,
            "vendor_id": None,
            "product_id": None,
            "bus": None,
            "address": None,
            "latency_estimate_ms": None,
            "error": "pyusb_not_available",
        }

    try:
        devs = list(usb.core.find(find_all=True))
    except Exception as e:
        return {
            "present": False,
            "vendor_id": None,
            "product_id": None,
            "bus": None,
            "address": None,
            "latency_estimate_ms": None,
            "error": f"usb_scan_failed:{e}",
        }

    for dev in devs:
        vid = dev.idVendor
        pid = dev.idProduct
        if (vid, pid) in MOVIDIUS_USB_IDS:
            try:
                bus = getattr(dev, "bus", None)
                address = getattr(dev, "address", None)
            except Exception:
                bus = None
                address = None

            latency_estimate = 2.0
            try:
                if hasattr(dev, "speed"):
                    speed = dev.speed
                    if speed >= 5000:
                        latency_estimate = 1.0
                    elif speed >= 480:
                        latency_estimate = 2.0
                    else:
                        latency_estimate = 5.0
            except Exception:
                pass

            return {
                "present": True,
                "vendor_id": vid,
                "product_id": pid,
                "bus": bus,
                "address": address,
                "latency_estimate_ms": latency_estimate,
                "error": None,
            }

    return {
        "present": False,
        "vendor_id": None,
        "product_id": None,
        "bus": None,
        "address": None,
        "latency_estimate_ms": None,
        "error": "movidius_not_found",
    }


# ============================================================
# 1. Enums & core structs
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
# 2. Movidius/OpenVINO engine + controller
# ============================================================

class MovidiusEngine:
    """
    OpenVINO backend preferring USB MYRIAD. Injects USB-aware timing noise.
    """

    def __init__(self):
        self.core = None
        self.device_connected = False
        self.backend_device = "NONE"
        self.available_devices: List[str] = []

        self.loaded_graphs: Dict[str, Dict[str, Any]] = {}
        self.total_inferences = 0
        self.latencies_ms: List[float] = []
        self.queue_depth = 0
        self.lock = threading.Lock()
        self.running = True

        self.usb_info = find_movidius_usb()
        self.usb_error_count = 0

        self.init_core()

    def refresh_usb_info(self):
        info = find_movidius_usb()
        if not info["present"]:
            self.usb_error_count += 1
        self.usb_info = info

    def init_core(self):
        if not OPENVINO_AVAILABLE:
            print("[MovidiusEngine] OpenVINO not available. NONE backend.")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False
            return

        try:
            self.core = Core()
            self.available_devices = self.core.available_devices
            print(f"[MovidiusEngine] Devices: {self.available_devices}")
        except Exception as e:
            print(f"[MovidiusEngine] Core init failed: {e}")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False

    def connect_device(self) -> bool:
        with self.lock:
            if self.core is None:
                self.device_connected = False
                self.backend_device = "NONE"
                return False

            self.refresh_usb_info()
            devices = self.available_devices
            chosen = None

            if self.usb_info.get("present") and "MYRIAD" in devices:
                chosen = "MYRIAD"
                print("[MovidiusEngine] USB Movidius → MYRIAD backend.")
            elif "MYRIAD" in devices:
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
                raise RuntimeError("No device connected")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Model XML not found: {graph_path}")

        model = self.core.read_model(graph_path)
        compiled_model = self.core.compile_model(model=model, device_name=self.backend_device)
        input_layer = compiled_model.inputs[0]
        input_name = input_layer.get_any_name()

        with self.lock:
            self.loaded_graphs[graph_id] = {
                "model_path": graph_path,
                "compiled_model": compiled_model,
                "input_name": input_name,
            }

    def unload_graph(self, graph_id: str):
        with self.lock:
            if graph_id in self.loaded_graphs:
                del self.loaded_graphs[graph_id]

    def run_inference(self, graph_id: str, input_array: np.ndarray):
        with self.lock:
            if not self.device_connected or self.core is None or self.backend_device == "NONE":
                raise RuntimeError("No device connected")
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

        self.refresh_usb_info()
        usb_lat = self.usb_info.get("latency_estimate_ms") or 2.0
        jitter = random.uniform(-0.2, 0.2) * usb_lat
        latency_ms = max(0.0, latency_ms + jitter)

        with self.lock:
            self.queue_depth = max(0, self.queue_depth - 1)
            self.total_inferences += 1
            self.latencies_ms.append(latency_ms)
            self.latencies_ms = self.latencies_ms[-200:]

        return np.array(output), latency_ms

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            if self.latencies_ms:
                avg_latency = sum(self.latencies_ms) / len(self.latencies_ms)
                fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
            else:
                avg_latency = 0.0
                fps = 0.0

            usb_info_now = self.usb_info
            return {
                "device_connected": self.device_connected,
                "backend_device": self.backend_device,
                "available_devices": list(self.available_devices),
                "loaded_graphs": {k: v["model_path"] for k, v in self.loaded_graphs.items()},
                "avg_latency_ms": avg_latency,
                "fps": fps,
                "queue_depth": self.queue_depth,
                "total_inferences": self.total_inferences,
                "usb_present": usb_info_now.get("present", False),
                "usb_vendor_id": usb_info_now.get("vendor_id"),
                "usb_product_id": usb_info_now.get("product_id"),
                "usb_bus": usb_info_now.get("bus"),
                "usb_address": usb_info_now.get("address"),
                "usb_latency_estimate_ms": usb_info_now.get("latency_estimate_ms"),
                "usb_error": usb_info_now.get("error"),
                "usb_error_count": self.usb_error_count,
            }

    def stop(self):
        self.running = False


class EngineController:
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
        self._log("[EngineController] Worker started.")
        while self.engine.running:
            try:
                cmd, args, res_queue = self.cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if cmd == "connect_device":
                    ok = self.engine.connect_device()
                    self._event("device", {"status": "connected" if ok else "failed"})
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
                self._log(f"[EngineController] ERROR in {cmd}: {e}\n{tb}")
                res_queue.put((False, e))


# ============================================================
# 3. MovidiusInferenceEngine + Turbocharger
# ============================================================

class MovidiusInferenceEngine:
    def __init__(self, model_graph_id: str = "default_graph", model_xml_path: str | None = None):
        self.log_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=1000)

        self.engine = MovidiusEngine()
        self.controller = EngineController(self.engine, self.log_queue, self.event_queue)

        self.model_graph_id = model_graph_id
        self.model_xml_path = model_xml_path

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
            print(f"[MovidiusInferenceEngine] Connect failed: {e}")
            self.device_ready = False

        if self.device_ready and self.model_xml_path:
            try:
                self._call("load_graph", self.model_graph_id, self.model_xml_path)
                self.graph_loaded = True
            except Exception as e:
                print(f"[MovidiusInferenceEngine] Load graph failed: {e}")
                self.graph_loaded = False
        else:
            self.graph_loaded = False

    def get_status(self) -> dict:
        try:
            metrics = self._call("get_metrics")
        except Exception as e:
            metrics = {"error": str(e)}
        return {"device_ready": self.device_ready, "graph_loaded": self.graph_loaded, "metrics": metrics}

    def usb_health_check(self):
        status = self.get_status()
        metrics = status.get("metrics", {})
        if not status["device_ready"] or not metrics.get("usb_present", False):
            print("[MovidiusInferenceEngine] USB issue. Reconnecting...")
            try:
                self.device_ready = self._call("connect_device")
            except Exception as e:
                print(f"[MovidiusInferenceEngine] Reconnect failed: {e}")
                self.device_ready = False

    def accelerate(self, state: dict, forecast: dict) -> dict:
        self.usb_health_check()

        if self.device_ready and not self.graph_loaded and self.model_xml_path:
            try:
                self._call("load_graph", self.model_graph_id, self.model_xml_path)
                self.graph_loaded = True
            except Exception:
                self.graph_loaded = False

        if not self.device_ready or not self.graph_loaded:
            status = self.get_status()
            metrics = status.get("metrics", {})
            backend = metrics.get("backend_device", "NONE")
            return {
                "used": False,
                "backend": backend,
                "anomaly_score": None,
                "risk_adjustment": 0.0,
                "intent_state": "neutral",
                "flow_state": "normal",
                "metrics": metrics,
            }

        risk = float(state.get("risk_score", 0.0))
        opportunity = float(state.get("opportunity_score", 0.0))
        anomaly_risk = float(forecast.get("anomaly_risk", risk))
        drive_risk = float(forecast.get("drive_risk", risk))
        hive_risk = float(forecast.get("hive_risk", risk))

        ip = state.get("input_profile", {}) or {}
        keys_ps = float(ip.get("keys_per_sec", 0.0))
        mouse_ps = float(ip.get("mouse_moves_per_sec", 0.0))
        clicks_ps = float(ip.get("clicks_per_sec", 0.0))
        burst = float(ip.get("burst_factor", 0.0))

        features = np.array([[
            risk,
            opportunity,
            anomaly_risk,
            drive_risk,
            hive_risk,
            keys_ps,
            mouse_ps,
            clicks_ps,
            burst,
        ]], dtype=np.float32)

        try:
            ok, result = self._safe_run_inference(features)
            if not ok:
                raise RuntimeError(result)
            output, latency_ms = result
            flat = np.array(output).flatten()

            anomaly_score = float(flat[0]) if flat.size > 0 else anomaly_risk
            risk_delta = float(flat[1]) if flat.size > 1 else (anomaly_score - anomaly_risk)
            intent_arousal = float(flat[2]) if flat.size > 2 else burst
            flow_arousal = float(flat[3]) if flat.size > 3 else keys_ps

            if intent_arousal < 0.25:
                intent_state = "low_engagement"
            elif intent_arousal < 0.6:
                intent_state = "focused"
            else:
                intent_state = "hyperfocused"

            if flow_arousal < 0.25:
                flow_state = "still"
            elif flow_arousal < 0.6:
                flow_state = "normal"
            else:
                flow_state = "rapid"

            metrics = self._call("get_metrics")
            backend = metrics.get("backend_device", "UNKNOWN")

            usb_present = metrics.get("usb_present", False)
            usb_lat = metrics.get("usb_latency_estimate_ms")
            if usb_present and usb_lat is not None:
                jitter_factor = usb_lat / 10.0
                risk_delta += random.uniform(-0.05, 0.05) * jitter_factor

            return {
                "used": True,
                "backend": backend,
                "anomaly_score": anomaly_score,
                "risk_adjustment": risk_delta,
                "intent_state": intent_state,
                "flow_state": flow_state,
                "metrics": {"latency_ms": latency_ms, **metrics},
            }
        except Exception as e:
            print(f"[MovidiusInferenceEngine] accelerate error: {e}")
            status = self.get_status()
            metrics = status.get("metrics", {})
            backend = metrics.get("backend_device", "NONE")
            return {
                "used": False,
                "backend": backend,
                "anomaly_score": None,
                "risk_adjustment": 0.0,
                "intent_state": "neutral",
                "flow_state": "normal",
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


class Turbocharger:
    """
    System turbocharger on top of MovidiusInferenceEngine.

    - Cache
    - CPU fallback
    - System-wide API (via BrainRuntime TCP server)
    """

    def __init__(self, mv_engine: MovidiusInferenceEngine | None):
        self.mv_engine = mv_engine
        self.cache: Dict[str, Any] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.cpu_fallback_count = 0
        self.lock = threading.Lock()

    def _make_key(self, op_type: str, payload: dict) -> str:
        try:
            blob = json.dumps({"op": op_type, "payload": payload}, sort_keys=True)
        except TypeError:
            blob = str(payload)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _should_use_movidius(self, cost_estimate: float, mv_status: dict) -> bool:
        if not mv_status.get("device_ready", False):
            return False
        backend = mv_status.get("metrics", {}).get("backend_device", "NONE")
        if backend != "MYRIAD":
            return False
        if cost_estimate < 1e5:
            return False
        return True

    def accelerated_infer(self,
                          state: dict,
                          forecast: dict,
                          op_type: str = "generic_infer",
                          cost_estimate: float = 1e6) -> dict:
        if not self.mv_engine:
            return {
                "backend": "CPU",
                "anomaly_score": float(forecast.get("anomaly_risk", state.get("risk_score", 0.0))),
                "risk_adjustment": 0.0,
                "intent_state": "neutral",
                "flow_state": "normal",
                "metrics": {"reason": "no_movidius_engine"},
                "from_cache": False,
                "used_movidius": False,
            }

        key_payload = {
            "op_type": op_type,
            "state": {
                "risk_score": state.get("risk_score"),
                "opportunity_score": state.get("opportunity_score"),
                "input_profile": state.get("input_profile"),
            },
            "forecast": {
                "anomaly_risk": forecast.get("anomaly_risk"),
                "drive_risk": forecast.get("drive_risk"),
                "hive_risk": forecast.get("hive_risk"),
            },
        }
        cache_key = self._make_key(op_type, key_payload)

        with self.lock:
            if cache_key in self.cache:
                self.hit_count += 1
                cached = dict(self.cache[cache_key])
                cached["from_cache"] = True
                return cached
            self.miss_count += 1

        status = self.mv_engine.get_status()
        use_mv = self._should_use_movidius(cost_estimate, status)

        if use_mv:
            mv_res = self.mv_engine.accelerate(state, forecast)
            result = {
                "backend": mv_res.get("backend", "MYRIAD"),
                "anomaly_score": mv_res.get("anomaly_score"),
                "risk_adjustment": mv_res.get("risk_adjustment"),
                "intent_state": mv_res.get("intent_state"),
                "flow_state": mv_res.get("flow_state"),
                "metrics": mv_res.get("metrics", {}),
                "from_cache": False,
                "used_movidius": mv_res.get("used", False),
            }
        else:
            self.cpu_fallback_count += 1
            anomaly_score = float(forecast.get("anomaly_risk", state.get("risk_score", 0.0)))
            risk_adjustment = 0.0
            result = {
                "backend": "CPU",
                "anomaly_score": anomaly_score,
                "risk_adjustment": risk_adjustment,
                "intent_state": "neutral",
                "flow_state": "normal",
                "metrics": {"reason": "cpu_fallback_or_small_cost"},
                "from_cache": False,
                "used_movidius": False,
            }

        with self.lock:
            self.cache[cache_key] = dict(result)
        return result

    def stats(self) -> dict:
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "hits": self.hit_count,
                "misses": self.miss_count,
                "cpu_fallbacks": self.cpu_fallback_count,
            }


# ============================================================
# 4. Brain organs (situational/predictive/hive/etc.)
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
            intent=f"I chose mission {mission} based on risk, opportunity, device health, and your overrides.",
            confidence=0.78,
            alternatives=["Switch to PROTECT under higher risk.", "Switch to OPTIMIZE if risk remains low."],
            risks=["May underutilize learning windows.", "Aggressive change could destabilize resources."],
            expected_outcomes=[
                "Maintain alignment with your mission priorities.",
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
            risks=["Hidden anomalies or device instability may bypass current sensitivity settings."],
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
            risks=["Reasoning is based on current models and device state; drift could degrade validity."],
            expected_outcomes=["You can audit and correct my priorities from this trace."],
        )


class JudgmentCortex:
    def evaluate(self, state: dict, forecast: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        anomaly_risk = float(forecast.get("anomaly_risk", risk))
        combined = 0.6 * risk + 0.4 * anomaly_risk
        if combined < 0.3:
            risk_level = "low"
        elif combined < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        verdict = "normal" if risk_level != "high" else "risky"
        return {"verdict": verdict, "risk_level": risk_level, "combined_risk": combined}


class ConfidenceEngine:
    def assess(self, state: dict, forecast: dict, judgment: dict, mv_result: dict) -> dict:
        base_risk = float(judgment.get("combined_risk", state.get("risk_score", 0.0)))
        mv_anomaly = mv_result.get("anomaly_score")
        if mv_anomaly is not None:
            base_risk = 0.7 * base_risk + 0.3 * float(mv_anomaly)

        confidence = 1.0 - base_risk
        confidence = max(0.0, min(1.0, confidence))

        metrics = mv_result.get("metrics", {})
        if metrics.get("usb_present") and metrics.get("usb_error_count", 0) > 0:
            confidence *= 0.95

        if confidence > 0.8:
            strategy = "trust"
        elif confidence > 0.5:
            strategy = "verify"
        else:
            strategy = "cautious"

        return {"score": confidence, "strategy": strategy}


class DataPhysicsEngine:
    def analyze_flow(self, state: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        opp = float(state.get("opportunity_score", 0.0))
        tol = float(state.get("risk_tolerance", 0.5))

        ip = state.get("input_profile", {}) or {}
        keys_ps = float(ip.get("keys_per_sec", 0.0))
        mouse_ps = float(ip.get("mouse_moves_per_sec", 0.0))
        clicks_ps = float(ip.get("clicks_per_sec", 0.0))
        burst = float(ip.get("burst_factor", 0.0))

        pressure = risk * 0.7 + opp * 0.3
        flow_rate = (keys_ps + mouse_ps + clicks_ps) / 3.0
        turbulence = burst * 0.6 + abs(risk - opp) * 0.4

        if risk > opp:
            flow_direction = "toward_risk"
        elif opp > risk:
            flow_direction = "toward_opportunity"
        else:
            flow_direction = "balanced"

        if pressure < 0.3 and turbulence < 0.3:
            zone = "laminar"
        elif pressure < 0.7 and turbulence < 0.7:
            zone = "transitional"
        else:
            zone = "turbulent"

        bottlenecks = []
        if pressure > 0.6:
            bottlenecks.append("IO")
        if risk > tol + 0.3:
            bottlenecks.append("CPU")
        if flow_rate > 0.7 or burst > 0.7:
            bottlenecks.append("INPUT_FOCUS")

        return {
            "bottlenecks": bottlenecks,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "turbulence": turbulence,
            "flow_direction": flow_direction,
            "zone": zone,
        }


class CollectiveHealthScore:
    def compute(self, organ_health: dict, forecast: dict, confidence: dict) -> dict:
        ready_count = sum(1 for v in organ_health.values() if v["ready"])
        total = len(organ_health) if organ_health else 1
        readiness = ready_count / total

        conf = float(confidence.get("score", 0.5))
        anomaly_risk = float(forecast.get("anomaly_risk", 0.5))

        score = 0.5 * readiness + 0.4 * conf + 0.1 * (1.0 - anomaly_risk)
        score = max(0.0, min(1.0, score))

        if score < 0.3:
            status = "critical"
        elif score < 0.6:
            status = "degraded"
        else:
            status = "good"

        return {"score": score, "status": status}


class AlteredStatesManager:
    def select_mode(self, state: dict, health: dict, forecast: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        health_score = float(health.get("score", 0.5))
        anomaly_risk = float(forecast.get("anomaly_risk", risk))

        physics = state.get("data_physics", {}) or {}
        zone = physics.get("zone", "laminar")
        pressure = float(physics.get("pressure", 0.0))

        mv_intent = state.get("mv_intent_state", "neutral")
        mv_flow = state.get("mv_flow_state", "normal")

        if risk > 0.75 or anomaly_risk > 0.8 or health_score < 0.3:
            if zone == "turbulent" or mv_flow == "rapid":
                mode = "hypervigilant"
            else:
                mode = "high_alert"
            mission = "PROTECT"
        elif health_score < 0.45 and risk < 0.6:
            mode = "self_repair"
            mission = "STABILITY"
        elif mv_intent in ("focused", "hyperfocused") and risk < 0.6 and anomaly_risk < 0.5:
            if zone == "laminar" and pressure < 0.5:
                mode = "deep_focus"
            else:
                mode = "focused_vigilance"
            mission = "LEARN"
        elif risk < 0.3 and anomaly_risk < 0.3 and health_score > 0.7 and mv_flow == "still":
            mode = "simulation_dream"
            mission = "LEARN"
        elif risk < 0.4 and anomaly_risk < 0.4 and health_score > 0.6:
            mode = "optimize"
            mission = "OPTIMIZE"
        else:
            mode = "baseline"
            mission = "PROTECT"

        return {"mode": mode, "mission": mission}


class PredictiveIntelEngine:
    def forecast(self, state: dict, flow: dict) -> dict:
        risk = float(state.get("risk_score", 0.0))
        pressure = float(flow.get("pressure", 0.0))
        turbulence = float(flow.get("turbulence", 0.0))

        anomaly_risk = min(1.0, (risk + pressure + 0.5 * turbulence) / 2.0)
        drive_risk = min(1.0, risk * 0.8 + 0.2 * pressure)
        hive_risk = min(1.0, risk * 0.7 + 0.3 * turbulence)
        summary = f"risk={risk:.2f}, pressure={pressure:.2f}, anomaly≈{anomaly_risk:.2f}"

        return {
            "anomaly_risk": anomaly_risk,
            "drive_risk": drive_risk,
            "hive_risk": hive_risk,
            "collective_health_score": 0.7,
            "health_trend": "STABLE",
            "summary": summary,
            "branches": [],
            "risks": [],
            "expected_outcomes": [],
        }


# ------------------------------------------------------------
# Extra “organs” – sensory, firewall, watchdog, etc.
# (Right now they are logical views over existing signals)
# ------------------------------------------------------------

class SensoryOrgan:
    """
    High-level sensory organ using:
    - input_profile
    - data physics
    - Movidius intent/flow
    """

    def derive_state(self, state: dict, flow: dict, mv_result: dict) -> dict:
        keys = state.get("input_profile", {}).get("keys_per_sec", 0.0)
        mouse = state.get("input_profile", {}).get("mouse_moves_per_sec", 0.0)
        burst = state.get("input_profile", {}).get("burst_factor", 0.0)
        intent = mv_result.get("intent_state", "neutral")
        flow_state = mv_result.get("flow_state", "normal")
        zone = flow.get("zone", "laminar")
        return {
            "keys_ps": keys,
            "mouse_ps": mouse,
            "burst": burst,
            "intent_state": intent,
            "flow_state": flow_state,
            "zone": zone,
        }


class AnomalyFirewall:
    """
    Very simple behavioral firewall: uses risk + anomaly + USB errors.
    """

    def assess(self, risk: float, anomaly: float, usb_errors: int) -> dict:
        firewall_state = "allow"
        if risk > 0.8 or anomaly > 0.8:
            firewall_state = "block_suspicious"
        if usb_errors > 5:
            firewall_state = "degraded_observe"
        return {
            "state": firewall_state,
            "risk": risk,
            "anomaly": anomaly,
            "usb_errors": usb_errors,
        }


class WatchdogDaemon:
    """
    Watchdog uses health score + USB metrics to flag danger.
    """

    def tick(self, health: dict, mv_metrics: dict) -> dict:
        score = float(health.get("score", 0.5))
        usb_present = mv_metrics.get("usb_present", False)
        usb_errors = mv_metrics.get("usb_error_count", 0)
        status = "ok"

        if score < 0.3 or usb_errors > 10 or not usb_present:
            status = "alert"
        elif score < 0.5 or usb_errors > 3:
            status = "warning"

        return {
            "status": status,
            "health_score": score,
            "usb_present": usb_present,
            "usb_errors": usb_errors,
        }


# ============================================================
# 5. Autoloader
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
    "judgment_cortex": {"module": __name__, "class": "JudgmentCortex"},
    "confidence_engine": {"module": __name__, "class": "ConfidenceEngine"},
    "data_physics": {"module": __name__, "class": "DataPhysicsEngine"},
    "collective_health": {"module": __name__, "class": "CollectiveHealthScore"},
    "altered_states": {"module": __name__, "class": "AlteredStatesManager"},
    "movidius_inference": {"module": __name__, "class": "MovidiusInferenceEngine"},
    "predictive_intel": {"module": __name__, "class": "PredictiveIntelEngine"},
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
# 6. HybridBrainCore
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
                self.situational.be_more_conservative(0.1)
            elif cmd.type == CommandType.BEGIN_LEARNING_CYCLE:
                self.situational.force_learn()
                self.situational.prioritize_learning_windows(0.3)
            elif cmd.type == CommandType.OPTIMIZE_PERFORMANCE:
                self.situational.force_optimize()
            elif cmd.type == CommandType.RESET_SITUATIONAL_CORTEX:
                self.situational = SituationalAwarenessCortex()

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
            f"Acted under mission {self.current_mission} | "
            f"risk={state.get('risk_score', 0.0):.3f} | "
            f"opp={state.get('opportunity_score', 0.0):.3f} | "
            f"mode={mode.get('mode', 'baseline')} | "
            f"intent={state.get('mv_intent_state', 'neutral')} | "
            f"flow={state.get('mv_flow_state', 'normal')}"
        )

        self._apply_commands()

        effective_mission = self.situational.state.effective_mission
        risk = state.get("risk_score", 0.0)
        risk_tol = self.situational.state.risk_tolerance
        anomaly_risk = forecast.get("anomaly_risk", risk)
        conf_score = confidence.get("score", 0.5)

        action = "observe"

        if effective_mission == Mission.PROTECT:
            if risk > risk_tol or anomaly_risk > 0.6:
                action = "defend"
            else:
                action = "harden_passively"
        elif effective_mission == Mission.LEARN:
            if conf_score > 0.6 and risk < risk_tol:
                action = "explore"
            else:
                action = "cautious_learn"
        elif effective_mission == Mission.OPTIMIZE:
            if conf_score > 0.7 and risk < risk_tol:
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
# 7. BrainRuntime + Turbocharger + TCP API + Extra Organs
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

        mv = self.loader.get("movidius_inference")
        self.turbo = Turbocharger(mv)

        self.sensory = SensoryOrgan()
        self.firewall = AnomalyFirewall()
        self.watchdog = WatchdogDaemon()

        self._start_turbo_server()

    def _start_turbo_server(self, host: str = "127.0.0.1", port: int = 7777):
        runtime_ref = self

        class TurboHandler(socketserver.StreamRequestHandler):
            def handle(self_inner):
                try:
                    raw = self_inner.rfile.readline().decode("utf-8").strip()
                    if not raw:
                        return
                    req = json.loads(raw)
                    op_type = req.get("op_type", "remote_op")
                    state = req.get("state", {})
                    forecast = req.get("forecast", {})
                    cost_estimate = float(req.get("cost_estimate", 1e6))
                    res = runtime_ref.turbo.accelerated_infer(state, forecast, op_type, cost_estimate)
                    self_inner.wfile.write((json.dumps(res) + "\n").encode("utf-8"))
                except Exception as e:
                    self_inner.wfile.write((json.dumps({"error": str(e)}) + "\n").encode("utf-8"))

        def server_thread():
            try:
                with socketserver.TCPServer((host, port), TurboHandler) as srv:
                    print(f"[TurboServer] Listening on {host}:{port}")
                    srv.serve_forever()
            except Exception as e:
                print(f"[TurboServer] Failed: {e}")

        t = threading.Thread(target=server_thread, daemon=True)
        t.start()

    def _synthetic_input_profile(self) -> dict:
        return {
            "keys_per_sec": random.uniform(0.0, 1.0),
            "mouse_moves_per_sec": random.uniform(0.0, 1.0),
            "clicks_per_sec": random.uniform(0.0, 0.8),
            "burst_factor": random.uniform(0.0, 1.0),
        }

    def tick(self, raw_telemetry: dict) -> dict:
        env = Environment.CALM
        opportunity_score = float(raw_telemetry.get("opportunity", 0.3))
        risk_score = float(raw_telemetry.get("risk", 0.2))
        anticipation = "Normal operation expected."
        auto_mission = Mission.PROTECT

        input_profile = self._synthetic_input_profile()

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
            "input_profile": input_profile,
        }

        dp = self.loader.get("data_physics")
        flow = safe_call(
            lambda: dp.analyze_flow(state_for_brain),
            default={"bottlenecks": [], "pressure": 0.0, "flow_rate": 0.0,
                     "turbulence": 0.0, "flow_direction": "neutral", "zone": "laminar"},
            label="data_physics",
        ) if dp else {"bottlenecks": [], "pressure": 0.0, "flow_rate": 0.0,
                      "turbulence": 0.0, "flow_direction": "neutral", "zone": "laminar"}

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
            default={"verdict": "unknown", "risk_level": "unknown", "combined_risk": risk_score},
            label="judgment_cortex",
        ) if jc else {"verdict": "unknown", "risk_level": "unknown", "combined_risk": risk_score}

        turbo_res = self.turbo.accelerated_infer(
            state_for_brain,
            forecast_for_brain,
            op_type="brain_tick",
            cost_estimate=1e6,
        )
        mv_result = {
            "used": turbo_res["used_movidius"],
            "backend": turbo_res["backend"],
            "anomaly_score": turbo_res["anomaly_score"],
            "risk_adjustment": turbo_res["risk_adjustment"],
            "intent_state": turbo_res["intent_state"],
            "flow_state": turbo_res["flow_state"],
            "metrics": turbo_res["metrics"],
        }

        state_for_brain["data_physics"] = flow
        state_for_brain["mv_intent_state"] = mv_result.get("intent_state", "neutral")
        state_for_brain["mv_flow_state"] = mv_result.get("flow_state", "normal")

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

        sensory_state = self.sensory.derive_state(state_for_brain, flow, mv_result)
        mv_metrics = mv_result.get("metrics", {})
        firewall_state = self.firewall.assess(
            risk=risk_score,
            anomaly=anomaly_risk,
            usb_errors=mv_metrics.get("usb_error_count", 0),
        )
        watchdog_state = self.watchdog.tick(health, mv_metrics)

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
            "sensory": sensory_state,
            "firewall": firewall_state,
            "watchdog": watchdog_state,
        }


# ============================================================
# 8. GUI helpers & panels
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
            b.setStyleSheet(
                "QPushButton { background-color: #333333; color: #EEEEEE; border-radius: 6px; padding: 4px 8px; }"
                "QPushButton:hover { background-color: #555555; }"
            )
            b.setCursor(Qt.PointingHandCursor)

        for_m.clicked.connect(self.brain.situational.force_protect)
        for_l.clicked.connect(self.brain.situational.force_learn)
        for_o.clicked.connect(self.brain.situational.force_optimize)
        auto_b.clicked.connect(self.brain.situational.return_to_auto)

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


class PredictivePanelGUI(QGroupBox):
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

    def update_from_snapshot(self, snap: dict):
        self.anomaly_label.setText(f"{snap.get('anomaly_risk', 0.0):.2f}")
        self.drive_label.setText(f"{snap.get('drive_risk', 0.0):.2f}")
        self.hive_label.setText(f"{snap.get('hive_risk', 0.0):.2f}")
        self.health_label.setText(f"{snap.get('collective_health_score', 0.0):.2f}")
        self.trend_label.setText(snap.get("health_trend", "-"))
        self.summary_label.setText(snap.get("forecast_summary", "-"))


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

    def update_from_snapshot(self, snap: dict):
        self.collective_label.setText(f"{snap.get('collective_risk_score', 0.0):.2f}")
        self.density_label.setText(f"{snap.get('hive_density', 0.0):.2f}")
        self.agreement_label.setText(f"{snap.get('node_agreement', 0.0):.2f}")
        self.divergence_label.setText(", ".join(snap.get("divergence_patterns", [])) or "-")
        self.mode_label.setText(snap.get("hive_sync_mode", "-"))


class MovidiusPanel(QGroupBox):
    def __init__(self, brain: BrainRuntime):
        super().__init__("Movidius / USB Turbocharger")
        self.brain = brain
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.backend_label = QLabel("-")
        self.used_label = QLabel("-")
        self.latency_label = QLabel("0.0 ms")
        self.fps_label = QLabel("0.0")
        self.queue_label = QLabel("0")
        self.total_label = QLabel("0")
        self.usb_mode_label = QLabel("-")
        self.usb_lat_label = QLabel("-")
        self.usb_err_label = QLabel("-")
        self.cache_size_label = QLabel("-")
        self.cache_hit_label = QLabel("-")
        self.cpu_fallback_label = QLabel("-")

        for lbl in (self.backend_label, self.used_label, self.latency_label,
                    self.fps_label, self.queue_label, self.total_label,
                    self.usb_mode_label, self.usb_lat_label, self.usb_err_label,
                    self.cache_size_label, self.cache_hit_label, self.cpu_fallback_label):
            lbl.setStyleSheet("color: #EEEEEE;")

        layout.addWidget(make_kv_row("Backend:", self.backend_label))
        layout.addWidget(make_kv_row("Used this tick:", self.used_label))
        layout.addWidget(make_kv_row("Avg latency:", self.latency_label))
        layout.addWidget(make_kv_row("FPS:", self.fps_label))
        layout.addWidget(make_kv_row("Queue depth:", self.queue_label))
        layout.addWidget(make_kv_row("Total inferences:", self.total_label))
        layout.addWidget(make_kv_row("USB mode:", self.usb_mode_label))
        layout.addWidget(make_kv_row("USB latency est:", self.usb_lat_label))
        layout.addWidget(make_kv_row("USB errors:", self.usb_err_label))
        layout.addWidget(make_kv_row("Turbo cache size:", self.cache_size_label))
        layout.addWidget(make_kv_row("Turbo cache hits:", self.cache_hit_label))
        layout.addWidget(make_kv_row("CPU fallbacks:", self.cpu_fallback_label))

    def update_from_mv(self, mv: dict, turbo_stats: dict):
        self.backend_label.setText(mv.get("backend", "NONE"))
        self.used_label.setText("Yes" if mv.get("used", False) else "No")
        metrics = mv.get("metrics", {}) or {}
        self.latency_label.setText(f"{metrics.get('avg_latency_ms', 0.0):.2f} ms")
        self.fps_label.setText(f"{metrics.get('fps', 0.0):.2f}")
        self.queue_label.setText(str(metrics.get("queue_depth", 0)))
        self.total_label.setText(str(metrics.get("total_inferences", 0)))

        usb_present = metrics.get("usb_present", False)
        backend = metrics.get("backend_device", "NONE")
        self.usb_mode_label.setText("USB Movidius" if usb_present and backend == "MYRIAD" else "Inactive")
        usb_lat = metrics.get("usb_latency_estimate_ms")
        self.usb_lat_label.setText(f"{usb_lat:.2f} ms" if usb_lat is not None else "-")
        self.usb_err_label.setText(str(metrics.get("usb_error_count", 0)))

        self.cache_size_label.setText(str(turbo_stats.get("cache_size", 0)))
        self.cache_hit_label.setText(str(turbo_stats.get("hits", 0)))
        self.cpu_fallback_label.setText(str(turbo_stats.get("cpu_fallbacks", 0)))


class CommandPanel(QGroupBox):
    def __init__(self, cmd: CommandBar):
        super().__init__("Command Bar")
        self.cmd = cmd
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QGridLayout(self)
        layout.setSpacing(6)

        def make_btn(text, cb):
            b = QPushButton(text)
            b.setStyleSheet(
                "QPushButton { background-color: #2F3A4A; color: #FFFFFF; "
                "border-radius: 8px; padding: 6px 10px; }"
                "QPushButton:hover { background-color: #40536A; }"
            )
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
            b.setStyleSheet(
                "QPushButton { background-color: #333333; color: #EEEEEE; border-radius: 6px; padding: 4px 8px; }"
                "QPushButton:hover { background-color: #555555; }"
            )
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


class SensoryPanel(QGroupBox):
    def __init__(self):
        super().__init__("Sensory Organ (Input & Flow)")
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.keys_label = QLabel("0.0")
        self.mouse_label = QLabel("0.0")
        self.burst_label = QLabel("0.0")
        self.intent_label = QLabel("-")
        self.flow_label = QLabel("-")
        self.zone_label = QLabel("-")
        for lbl in (self.keys_label, self.mouse_label, self.burst_label,
                    self.intent_label, self.flow_label, self.zone_label):
            lbl.setStyleSheet("color: #EEEEEE;")
        layout.addWidget(make_kv_row("Keys/sec:", self.keys_label))
        layout.addWidget(make_kv_row("Mouse/sec:", self.mouse_label))
        layout.addWidget(make_kv_row("Burst factor:", self.burst_label))
        layout.addWidget(make_kv_row("Intent state:", self.intent_label))
        layout.addWidget(make_kv_row("Flow state:", self.flow_label))
        layout.addWidget(make_kv_row("Flow zone:", self.zone_label))

    def update_from_snapshot(self, snap: dict):
        self.keys_label.setText(f"{snap.get('keys_ps', 0.0):.2f}")
        self.mouse_label.setText(f"{snap.get('mouse_ps', 0.0):.2f}")
        self.burst_label.setText(f"{snap.get('burst', 0.0):.2f}")
        self.intent_label.setText(snap.get("intent_state", "-"))
        self.flow_label.setText(snap.get("flow_state", "-"))
        self.zone_label.setText(snap.get("zone", "-"))


class FirewallPanel(QGroupBox):
    def __init__(self):
        super().__init__("Anomaly Firewall")
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.state_label = QLabel("-")
        self.risk_label = QLabel("0.0")
        self.anom_label = QLabel("0.0")
        self.usb_err_label = QLabel("0")
        for lbl in (self.state_label, self.risk_label, self.anom_label, self.usb_err_label):
            lbl.setStyleSheet("color: #EEEEEE;")
        layout.addWidget(make_kv_row("Firewall state:", self.state_label))
        layout.addWidget(make_kv_row("Risk:", self.risk_label))
        layout.addWidget(make_kv_row("Anomaly:", self.anom_label))
        layout.addWidget(make_kv_row("USB errors:", self.usb_err_label))

    def update_from_snapshot(self, snap: dict):
        self.state_label.setText(snap.get("state", "-"))
        self.risk_label.setText(f"{snap.get('risk', 0.0):.2f}")
        self.anom_label.setText(f"{snap.get('anomaly', 0.0):.2f}")
        self.usb_err_label.setText(str(snap.get("usb_errors", 0)))


class WatchdogPanel(QGroupBox):
    def __init__(self):
        super().__init__("Watchdog Daemon")
        self.setStyleSheet("QGroupBox { border: 1px solid #444444; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.status_label = QLabel("-")
        self.health_label = QLabel("0.0")
        self.usb_present_label = QLabel("-")
        self.usb_err_label = QLabel("0")
        for lbl in (self.status_label, self.health_label, self.usb_present_label, self.usb_err_label):
            lbl.setStyleSheet("color: #EEEEEE;")
        layout.addWidget(make_kv_row("Status:", self.status_label))
        layout.addWidget(make_kv_row("Health score:", self.health_label))
        layout.addWidget(make_kv_row("USB present:", self.usb_present_label))
        layout.addWidget(make_kv_row("USB errors:", self.usb_err_label))

    def update_from_snapshot(self, snap: dict):
        self.status_label.setText(snap.get("status", "-"))
        self.health_label.setText(f"{snap.get('health_score', 0.0):.2f}")
        self.usb_present_label.setText("Yes" if snap.get("usb_present", False) else "No")
        self.usb_err_label.setText(str(snap.get("usb_errors", 0)))


# ============================================================
# 9. Main Window (Tabbed, nice GUI)
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASI USB Movidius Turbo Cockpit")
        self.resize(1600, 900)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #050911;
            }
            QGroupBox {
                background-color: rgba(20, 30, 50, 200);
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
            QTabWidget::pane {
                border: 1px solid #333333;
            }
            QTabBar::tab {
                background: #111724;
                color: #DDDDDD;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background: #1F2A3D;
            }
        """)

        loader = BrainAutoloader()
        loader.load_organs()
        self.brain = BrainRuntime(loader)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Tab 1: Core mission + prediction
        core_tab = QWidget()
        core_layout = QGridLayout(core_tab)
        core_layout.setSpacing(10)
        core_layout.setContentsMargins(10, 10, 10, 10)

        self.situ_panel = SituationalPanel(self.brain)
        self.pred_panel = PredictivePanelGUI(self.brain)
        self.hive_panel = HivePanel(self.brain)
        self.cmd_panel = CommandPanel(self.brain.command_bar)

        core_layout.addWidget(self.situ_panel, 0, 0)
        core_layout.addWidget(self.pred_panel, 0, 1)
        core_layout.addWidget(self.hive_panel, 1, 0)
        core_layout.addWidget(self.cmd_panel, 1, 1)
        tabs.addTab(core_tab, "Core Brain")

        # Tab 2: Movidius + turbo + USB
        mv_tab = QWidget()
        mv_layout = QGridLayout(mv_tab)
        mv_layout.setSpacing(10)
        mv_layout.setContentsMargins(10, 10, 10, 10)

        self.mv_panel = MovidiusPanel(self.brain)
        self.sensory_panel = SensoryPanel()
        self.firewall_panel = FirewallPanel()
        self.watchdog_panel = WatchdogPanel()

        mv_layout.addWidget(self.mv_panel, 0, 0, 2, 1)
        mv_layout.addWidget(self.sensory_panel, 0, 1)
        mv_layout.addWidget(self.firewall_panel, 1, 1)
        mv_layout.addWidget(self.watchdog_panel, 2, 0, 1, 2)

        tabs.addTab(mv_tab, "Turbo / USB / Safety")

        # Tab 3: Dialogue / ASI
        dlg_tab = QWidget()
        dlg_layout = QVBoxLayout(dlg_tab)
        dlg_layout.setSpacing(10)
        dlg_layout.setContentsMargins(10, 10, 10, 10)
        self.dialog_panel = DialoguePanel(self.brain)
        dlg_layout.addWidget(self.dialog_panel)
        tabs.addTab(dlg_tab, "ASI Dialogue")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(700)

    def on_tick(self):
        telemetry = {
            "risk": random.uniform(0.0, 1.0),
            "opportunity": random.uniform(0.0, 1.0),
        }

        snapshot = self.brain.tick(telemetry)

        self.situ_panel.update_from_snapshot(snapshot["situational"])
        self.pred_panel.update_from_snapshot(snapshot["predictive"])
        self.hive_panel.update_from_snapshot(snapshot["hive"])

        mv_res = snapshot["movidius"]
        turbo_stats = self.brain.turbo.stats()
        self.mv_panel.update_from_mv(mv_res, turbo_stats)

        self.sensory_panel.update_from_snapshot(snapshot["sensory"])
        self.firewall_panel.update_from_snapshot(snapshot["firewall"])
        self.watchdog_panel.update_from_snapshot(snapshot["watchdog"])


# ============================================================
# 10. Entry point
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

