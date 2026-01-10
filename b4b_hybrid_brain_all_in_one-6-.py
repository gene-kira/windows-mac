"""
nerve_center_full_cluster.py

System nerve center organism with full organ cluster:

ORGANS:
- DeepRamOrgan: deep RAM cache organ (virtual)
- BackupEngineOrgan: backup pressure and fallback prep
- NetworkWatcherOrgan: real network IO and connection pressure
- GPUCacheOrgan: GPU cache/VRAM-like virtual pressure
- ThermalOrgan: real CPU temperature-based thermal pressure (when available)
- DiskOrgan: real disk IO pressure
- VRAMOrgan: real GPU VRAM pressure (via pynvml if available)
- AICoachOrgan: generates hints from brain output + context
- SwarmNodeOrgan: simple file-based swarm sync across nodes

BRAIN:
- HybridBrain:
    - Multi-horizon forecasting (1s, 5s, 30s, 120s)
    - Multi-engine prediction (EWMA + trend + baseline)
    - Meta-confidence fusion
    - Behavioral fingerprinting
    - Dynamic stance thresholds
    - Predictive risk dampening
    - Stability-first reinforcement
    - Meta-states: Hyper-Flow, Deep-Dream, Sentinel, Recovery-Flow
    - Model integrity self-check
    - Mode-aware profiles (Back4Blood / idle / work)
    - Persistent state to local + optional SMB path

GUI:
- BrainCortexPanel (Tkinter):
    - Health / risk / meta-confidence / model integrity
    - Meta-state
    - Stance thresholds
    - Deep RAM / Backup / Network / GPU / Thermal / Disk / VRAM / Swarm / Coach status
    - Actions
    - Internal reasoning tail
    - Health/Risk history graph
    - Stance override
    - Meta-state override
    - Memory paths, Save Memory Now, SMB path

INPUT:
- Back4BloodAnalyzer:
    - System snapshot (CPU, RAM, IO, threads)
    - Organ metrics
    - Game metrics (FPS via PresentMon, ping via ping, input latency via WH_MOUSE_LL)
    - Mode-aware stability score
"""

import importlib
import math
import time
import traceback
import subprocess
import threading
import re
import ctypes
import json
import os
from ctypes import wintypes
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List, Tuple

# ------------------------------------------------------------
# Basic deps
# ------------------------------------------------------------

try:
    import psutil
except ImportError:
    psutil = None

try:
    import tkinter as tk
    from tkinter import ttk, simpledialog
    TK_AVAILABLE = True
except Exception:
    tk = None
    ttk = None
    simpledialog = None
    TK_AVAILABLE = False


def get_default_memory_path() -> str:
    base = os.path.join(os.path.expanduser("~"), "HybridBrainMemory")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "brain_state.json")


LOCAL_MEMORY_PATH = get_default_memory_path()


# ============================================================
#  AUTOLOADER
# ============================================================

class AutoLoader:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self.available: Dict[str, bool] = {}

    def load(self, name: str, alias: Optional[str] = None):
        key = alias or name
        if key in self._cache:
            return self._cache[key]
        try:
            module = importlib.import_module(name)
            self._cache[key] = module
            self.available[key] = True
            return module
        except Exception:
            self.available[key] = False
            self._cache[key] = None
            return None

    def summary(self) -> Dict[str, bool]:
        return dict(self.available)


AUTO = AutoLoader()
np = AUTO.load("numpy", alias="np")
ov = AUTO.load("openvino.runtime", alias="openvino")  # optional Movidius/OpenVINO


# ============================================================
#  META-STATES
# ============================================================

class MetaState:
    HYPER_FLOW = "Hyper-Flow"
    DEEP_DREAM = "Deep-Dream"
    SENTINEL = "Sentinel"
    RECOVERY_FLOW = "Recovery-Flow"


META_STATE_CONFIG = {
    MetaState.HYPER_FLOW: {
        "prediction_weight": 1.3,
        "stance_aggressiveness": 1.3,
        "deep_ram_appetite": 1.4,
        "thread_expansion": 1.4,
        "cache_push": 1.3,
    },
    MetaState.DEEP_DREAM: {
        "prediction_weight": 0.9,
        "stance_aggressiveness": 0.6,
        "deep_ram_appetite": 1.8,
        "thread_expansion": 0.7,
        "cache_push": 1.1,
    },
    MetaState.SENTINEL: {
        "prediction_weight": 1.5,
        "stance_aggressiveness": 0.8,
        "deep_ram_appetite": 0.7,
        "thread_expansion": 0.6,
        "cache_push": 0.9,
    },
    MetaState.RECOVERY_FLOW: {
        "prediction_weight": 1.1,
        "stance_aggressiveness": 0.5,
        "deep_ram_appetite": 0.6,
        "thread_expansion": 0.4,
        "cache_push": 0.8,
    },
}


# ============================================================
#  ORGANS
# ============================================================

class DeepRamOrgan:
    def __init__(self):
        self.current_fill = 0.4
        self.target_fill = 0.5
        self.mode = "Normal"
        self.last_update = time.time()

    def _tick_internal_dynamics(self):
        now = time.time()
        dt = now - self.last_update
        if dt <= 0:
            return
        self.last_update = now

        if self.mode == "Normal":
            decay_rate = 0.02
        elif self.mode == "Tight":
            decay_rate = 0.05
        elif self.mode == "Aggressive":
            decay_rate = 0.08
        elif self.mode == "Expansion":
            decay_rate = 0.01
        else:
            decay_rate = 0.02

        self.current_fill -= decay_rate * dt
        self.current_fill += 0.02 * dt * self.target_fill
        self.current_fill = max(0.0, min(1.0, self.current_fill))

    def apply_action(self, action: str):
        if action == "SHRINK_DEEP_RAM":
            self.mode = "Aggressive"
            self.target_fill = max(0.1, self.target_fill - 0.2)
        elif action == "TRIM_DEEP_RAM":
            self.mode = "Tight"
            self.target_fill = max(0.2, self.target_fill - 0.1)
        elif action == "INCREASE_DEEP_RAM_CACHE":
            self.mode = "Expansion"
            self.target_fill = min(0.9, self.target_fill + 0.2)
        elif action == "ALLOW_THREAD_EXPANSION":
            if self.mode not in ("Aggressive", "Tight"):
                self.mode = "Normal"

    def metrics(self) -> Dict[str, float]:
        self._tick_internal_dynamics()
        deep_ram_usage = self.current_fill
        if deep_ram_usage < 0.5:
            cache_pressure = deep_ram_usage * 0.8
        else:
            cache_pressure = 0.8 + (deep_ram_usage - 0.5) * 0.4
        cache_pressure = max(0.0, min(1.0, cache_pressure))
        return {
            "deep_ram_usage": deep_ram_usage,
            "cache_pressure": cache_pressure,
        }

    def describe(self) -> str:
        return f"{self.mode} | fill={self.current_fill:.2f} target={self.target_fill:.2f}"


class BackupEngineOrgan:
    def __init__(self):
        self.queue_depth = 0
        self.last_backup_time = time.time()
        self.mode = "Normal"

    def simulate_activity(self):
        now = time.time()
        if now - self.last_backup_time > 300:
            self.queue_depth = min(10, self.queue_depth + 1)

    def apply_action(self, action: str):
        if action == "PREPARE_FALLBACK_PATHS":
            self.mode = "Prepping"
            self.queue_depth = min(10, self.queue_depth + 2)

    def backup_completed(self):
        self.queue_depth = max(0, self.queue_depth - 1)
        self.last_backup_time = time.time()
        self.mode = "Normal"

    def metrics(self) -> Dict[str, float]:
        self.simulate_activity()
        now = time.time()
        age = now - self.last_backup_time
        age_norm = min(1.0, age / 900.0)
        pressure = min(1.0, (self.queue_depth / 10.0) + 0.3 * age_norm)
        return {
            "backup_pressure": pressure,
            "last_backup_age_norm": age_norm,
        }

    def describe(self) -> str:
        age = time.time() - self.last_backup_time
        return f"{self.mode} | q={self.queue_depth} age={age/60:.1f}m"


class NetworkWatcherOrgan:
    """
    Real-data network watcher.
    - Uses psutil.net_io_counters for bandwidth.
    - Uses psutil.net_connections for connection count.
    """

    def __init__(self):
        self.mode = "Normal"
        self.alert_rate = 0.1
        self.bandwidth_norm = 0.3

        self._last_bytes_sent = None
        self._last_bytes_recv = None
        self._last_time = None

    def _update_bandwidth(self):
        if psutil is None:
            return
        now = time.time()
        net = psutil.net_io_counters()
        if self._last_time is None:
            self._last_time = now
            self._last_bytes_sent = net.bytes_sent
            self._last_bytes_recv = net.bytes_recv
            return

        dt = now - self._last_time
        if dt <= 0:
            return

        d_sent = net.bytes_sent - self._last_bytes_sent
        d_recv = net.bytes_recv - self._last_bytes_recv
        total = max(0, d_sent + d_recv)

        bw = total / dt
        self.bandwidth_norm = min(1.0, bw / (100 * 1024 * 1024))  # assume 100 MB/s max

        self._last_time = now
        self._last_bytes_sent = net.bytes_sent
        self._last_bytes_recv = net.bytes_recv

    def _update_alert_rate(self):
        if psutil is None:
            return
        try:
            conns = psutil.net_connections(kind="inet")
            active = sum(1 for c in conns if c.status == "ESTABLISHED")
        except Exception:
            active = 0

        self.alert_rate = min(1.0, active / 200.0)
        if self.mode == "Throttled":
            self.alert_rate *= 0.7

    def apply_action(self, action: str):
        if action == "REDUCE_INGESTION":
            self.mode = "Throttled"
        elif action == "LIMIT_INGESTION":
            self.mode = "Throttled"

    def metrics(self) -> Dict[str, float]:
        self._update_bandwidth()
        self._update_alert_rate()
        return {
            "net_alert_rate": self.alert_rate,
            "net_bandwidth_norm": self.bandwidth_norm,
        }

    def describe(self) -> str:
        return f"{self.mode} | alerts={self.alert_rate:.2f} bw={self.bandwidth_norm:.2f}"


class GPUCacheOrgan:
    def __init__(self):
        self.mode = "Normal"
        self.cache_fill = 0.4
        self.gpu_mem_usage = 0.5
        self.last_update = time.time()

    def _tick_internal_dynamics(self):
        now = time.time()
        dt = now - self.last_update
        if dt <= 0:
            return
        self.last_update = now

        if self.mode == "Normal":
            drift = 0.02
        elif self.mode == "Tight":
            drift = -0.04
        elif self.mode == "Expansion":
            drift = 0.05
        else:
            drift = 0.02

        self.cache_fill += drift * dt * 0.1
        self.cache_fill = max(0.0, min(1.0, self.cache_fill))
        self.gpu_mem_usage = max(0.0, min(1.0, self.cache_fill + 0.2))

    def apply_action(self, action: str):
        if action == "INCREASE_DEEP_RAM_CACHE":
            self.mode = "Expansion"
        elif action in ("SHRINK_DEEP_RAM", "TRIM_DEEP_RAM"):
            self.mode = "Tight"

    def metrics(self) -> Dict[str, float]:
        self._tick_internal_dynamics()
        return {
            "gpu_cache_fill": self.cache_fill,
            "gpu_mem_usage": self.gpu_mem_usage,
        }

    def describe(self) -> str:
        return f"{self.mode} | cache={self.cache_fill:.2f} mem={self.gpu_mem_usage:.2f}"


class ThermalOrgan:
    def __init__(self):
        self.temp_norm = 0.2
        self.pressure = 0.2

    def _read_temp_c(self) -> Optional[float]:
        if psutil is None or not hasattr(psutil, "sensors_temperatures"):
            return None
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for key in ("coretemp", "cpu-thermal", "k10temp"):
                if key in temps:
                    entries = temps[key]
                    if entries:
                        return float(entries[0].current)
            for entries in temps.values():
                if entries:
                    return float(entries[0].current)
        except Exception:
            return None
        return None

    def metrics(self) -> Dict[str, float]:
        temp_c = self._read_temp_c()
        if temp_c is None:
            self.temp_norm = 0.3
        else:
            self.temp_norm = min(1.0, max(0.0, (temp_c - 30.0) / 60.0))
        self.pressure = 0.7 * self.pressure + 0.3 * self.temp_norm
        return {
            "thermal_temp_norm": self.temp_norm,
            "thermal_pressure": self.pressure,
        }

    def describe(self) -> str:
        return f"temp_norm={self.temp_norm:.2f} pressure={self.pressure:.2f}"


class DiskOrgan:
    def __init__(self):
        self._last_read = None
        self._last_write = None
        self._last_time = None
        self.disk_io_norm = 0.2
        self.queue_pressure = 0.2

    def _update_io(self):
        if psutil is None:
            return
        now = time.time()
        io = psutil.disk_io_counters()
        if self._last_time is None:
            self._last_time = now
            self._last_read = io.read_bytes
            self._last_write = io.write_bytes
            return
        dt = now - self._last_time
        if dt <= 0:
            return
        d_read = io.read_bytes - self._last_read
        d_write = io.write_bytes - self._last_write
        total = max(0, d_read + d_write)
        rate = total / dt
        self.disk_io_norm = min(1.0, rate / (500 * 1024 * 1024))  # 500 MB/s max
        self.queue_pressure = 0.7 * self.queue_pressure + 0.3 * self.disk_io_norm
        self._last_time = now
        self._last_read = io.read_bytes
        self._last_write = io.write_bytes

    def metrics(self) -> Dict[str, float]:
        self._update_io()
        return {
            "disk_io_norm": self.disk_io_norm,
            "disk_queue_pressure": self.queue_pressure,
        }

    def describe(self) -> str:
        return f"io={self.disk_io_norm:.2f} q={self.queue_pressure:.2f}"


class VRAMOrgan:
    def __init__(self):
        self.vram_usage_norm = 0.3
        self.vram_pressure = 0.3
        self._nvml = AUTO.load("pynvml", alias="pynvml")
        if self._nvml:
            try:
                self._nvml.nvmlInit()
            except Exception:
                self._nvml = None

    def _read_vram_usage(self) -> Optional[float]:
        if not self._nvml:
            return None
        try:
            nvml = self._nvml
            count = nvml.nvmlDeviceGetCount()
            if count == 0:
                return None
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem.used
            total = mem.total or 1
            return used / total
        except Exception:
            return None

    def metrics(self) -> Dict[str, float]:
        usage = self._read_vram_usage()
        if usage is None:
            usage = 0.3
        self.vram_usage_norm = float(max(0.0, min(1.0, usage)))
        self.vram_pressure = 0.7 * self.vram_pressure + 0.3 * self.vram_usage_norm
        return {
            "vram_usage_norm": self.vram_usage_norm,
            "vram_pressure": self.vram_pressure,
        }

    def describe(self) -> str:
        return f"vram={self.vram_usage_norm:.2f} pressure={self.vram_pressure:.2f}"


class AICoachOrgan:
    def __init__(self):
        self.last_hint = "Coach idle."

    def update(self, brain_out: Dict[str, Any], ctx: Dict[str, Any]):
        risk = brain_out.get("risk_score", 0.0)
        health = brain_out.get("health_score", 0.0)
        fps = ctx.get("fps", None)
        ping = ctx.get("ping", None)
        stance = ctx.get("stance", "Balanced")

        if risk > 0.8:
            self.last_hint = "Coach: System approaching overload. Calm stance / closing background tasks recommended."
        elif health > 0.85 and risk < 0.4 and stance != "Beast":
            self.last_hint = "Coach: Plenty of headroom. Beast stance is safe if you need more performance."
        elif fps is not None and fps < 60:
            self.last_hint = "Coach: FPS is low. Deep RAM + network trimming may help if risk is moderate."
        elif ping is not None and ping > 120:
            self.last_hint = "Coach: Network latency high. Non-game network ingestion should be minimized."
        else:
            self.last_hint = "Coach: System stable. Current stance and meta-state look reasonable."

    def describe(self) -> str:
        return self.last_hint


class SwarmNodeOrgan:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.swarm_dir = os.path.join(os.path.expanduser("~"), "HybridBrainSwarm")
        os.makedirs(self.swarm_dir, exist_ok=True)
        self.swarm_node_count = 1
        self.swarm_avg_health = 1.0

    def _heartbeat(self, health: float, risk: float):
        payload = {
            "node_id": self.node_id,
            "ts": time.time(),
            "health": float(health),
            "risk": float(risk),
        }
        path = os.path.join(self.swarm_dir, f"{self.node_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _read_swarm(self):
        total_health = 0.0
        count = 0
        now = time.time()
        for name in os.listdir(self.swarm_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.swarm_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if now - data.get("ts", 0) > 60:
                    continue
                total_health += float(data.get("health", 0.0))
                count += 1
            except Exception:
                continue
        if count == 0:
            self.swarm_node_count = 1
            self.swarm_avg_health = 1.0
        else:
            self.swarm_node_count = count
            self.swarm_avg_health = total_health / count

    def update(self, health: float, risk: float) -> Dict[str, float]:
        self._heartbeat(health, risk)
        self._read_swarm()
        return {
            "swarm_node_count": float(self.swarm_node_count),
            "swarm_avg_health": float(self.swarm_avg_health),
        }

    def describe(self) -> str:
        return f"nodes={self.swarm_node_count} avg_health={self.swarm_avg_health:.2f}"


# ============================================================
#  MOVIDIUS (optional)
# ============================================================

class MovidiusInferenceEngine:
    def __init__(self):
        self.ready = ov is not None
        self._reason_trace: List[str] = []

    def clear_trace(self):
        self._reason_trace.clear()

    def log(self, msg: str):
        self._reason_trace.append(msg)

    def get_trace(self) -> List[str]:
        return list(self._reason_trace)

    def infer(self, feature_vector: List[float]) -> Dict[str, float]:
        self.clear_trace()
        if not self.ready:
            self.log("Movidius not available: running in CPU-only prediction mode.")
            return {}
        self.log(f"Movidius would process feature vector of length {len(feature_vector)}.")
        self.log("Replace this stub with actual OpenVINO/Movidius inference graph.")
        return {
            "risk_delta": 0.0,
            "confidence_delta": 0.0,
        }


# ============================================================
#  HYBRID BRAIN
# ============================================================

class HybridBrain:
    def __init__(
        self,
        history_seconds: float = 300.0,
        max_snapshots: int = 600,
        initial_meta_state: str = MetaState.SENTINEL,
        reasoning_log_size: int = 200,
    ):
        self.history_seconds = history_seconds
        self.max_snapshots = max_snapshots
        self.history = deque(maxlen=max_snapshots)

        self.reinforcement = defaultdict(lambda: {
            "success": 0,
            "failure": 0,
            "stability_score_sum": 0.0,
            "count": 0,
        })

        self.stance_thresholds = {
            "Calm":  {"risk_low": 0.2, "risk_high": 0.4},
            "Balanced": {"risk_low": 0.3, "risk_high": 0.6},
            "Beast": {"risk_low": 0.4, "risk_high": 0.8},
        }

        self.fingerprints: List[Dict[str, Any]] = []
        self.meta_state = initial_meta_state
        self.movidius = MovidiusInferenceEngine()

        self.last_confidence = 0.5
        self.last_health_score = 0.8
        self.last_risk_score = 0.2
        self._fp_id_counter = 0

        self._reason_log = deque(maxlen=reasoning_log_size)
        self.model_integrity = 1.0

        self.mode_profiles = {
            "Back4Blood": {
                "target_cpu": 0.7,
                "target_mem": 0.7,
                "max_latency_ms": 40,
                "weight_fps": 0.4,
            },
            "idle": {
                "target_cpu": 0.2,
                "target_mem": 0.4,
                "max_latency_ms": 100,
                "weight_fps": 0.0,
            },
            "work": {
                "target_cpu": 0.5,
                "target_mem": 0.6,
                "max_latency_ms": 60,
                "weight_fps": 0.1,
            },
        }

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._reason_log.append(f"[{ts}] {msg}")

    def get_reasoning_log(self, last_n: int = 20) -> List[str]:
        items = list(self._reason_log)
        if last_n >= len(items):
            return items
        return items[-last_n:]

    def update(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = timestamp or time.time()
        context = context or {}

        self._log("Update cycle started.")
        self._log(f"Context stance={context.get('stance', 'Balanced')} meta_state={self.meta_state}")

        self._push_history(timestamp, metrics)
        predictions = self._predict_multi_horizon()
        self._log("Multi-horizon predictions (multi-engine) computed.")

        self.model_integrity = self._self_check(metrics, predictions)
        self._log(f"Model integrity={self.model_integrity:.2f}")

        meta_conf = self._compute_meta_confidence(predictions, metrics)
        self._log(f"Meta-confidence fused -> {meta_conf:.3f}")

        health = self._compute_health_score(metrics, predictions, meta_conf)
        self._log(f"Health score computed -> {health:.3f}")

        self._update_reinforcement(context, health)
        self._update_dynamic_stance_thresholds()
        self._log("Reinforcement and stance thresholds updated.")

        risk_score, risk_band, actions = self._compute_risk_and_actions(
            predictions, meta_conf, metrics, context
        )
        self._log(f"Risk score={risk_score:.3f}, band={risk_band}, actions={actions}")

        self._update_meta_state(risk_score, health, context)
        self._log(f"Meta-state now -> {self.meta_state}")

        fingerprint_id = self._update_behavioral_fingerprints(
            metrics, predictions, context, risk_score, health
        )
        self._log(f"Behavioral fingerprint updated, id={fingerprint_id}")

        self.last_confidence = meta_conf
        self.last_health_score = health
        self.last_risk_score = risk_score

        return {
            "timestamp": timestamp,
            "predictions": predictions,
            "meta_confidence": meta_conf,
            "health_score": health,
            "risk_score": risk_score,
            "risk_band": risk_band,
            "actions": actions,
            "meta_state": self.meta_state,
            "stance_thresholds": self.stance_thresholds,
            "fingerprint_id": fingerprint_id,
            "autoloader": AUTO.summary(),
            "reasoning_log_tail": self.get_reasoning_log(10),
            "model_integrity": self.model_integrity,
        }

    # persistent state

    def to_state_dict(self) -> dict:
        hist = list(self.history)
        reinf = {}
        for (stance, meta, hour), stats in self.reinforcement.items():
            key = f"{stance}|{meta}|{hour}"
            reinf[key] = stats
        state = {
            "history": hist,
            "reinforcement": reinf,
            "stance_thresholds": self.stance_thresholds,
            "fingerprints": self.fingerprints,
            "meta_state": self.meta_state,
            "last_confidence": self.last_confidence,
            "last_health_score": self.last_health_score,
            "last_risk_score": self.last_risk_score,
            "_fp_id_counter": self._fp_id_counter,
            "mode_profiles": self.mode_profiles,
        }
        return state

    def from_state_dict(self, state: dict):
        try:
            hist = state.get("history", [])
            self.history.clear()
            for ts, metrics in hist:
                self.history.append((float(ts), dict(metrics)))

            self.reinforcement.clear()
            reinf = state.get("reinforcement", {})
            for key, stats in reinf.items():
                try:
                    stance, meta, hour = key.split("|", 2)
                    hour = int(hour)
                    self.reinforcement[(stance, meta, hour)] = dict(stats)
                except Exception:
                    continue

            st_thresh = state.get("stance_thresholds")
            if isinstance(st_thresh, dict):
                self.stance_thresholds = st_thresh

            fps = state.get("fingerprints", [])
            if isinstance(fps, list):
                self.fingerprints = fps

            self.meta_state = state.get("meta_state", self.meta_state)
            self.last_confidence = float(state.get("last_confidence", self.last_confidence))
            self.last_health_score = float(state.get("last_health_score", self.last_health_score))
            self.last_risk_score = float(state.get("last_risk_score", self.last_risk_score))
            self._fp_id_counter = int(state.get("_fp_id_counter", self._fp_id_counter))

            mp = state.get("mode_profiles")
            if isinstance(mp, dict):
                self.mode_profiles.update(mp)

            self._log("State restored from disk.")
        except Exception as e:
            self._log(f"Failed to restore state: {e}")

    def save_state(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_state_dict(), f)
            self._log(f"State saved to {path}")
        except Exception as e:
            self._log(f"Failed to save state to {path}: {e}")

    def load_state(self, path: str):
        if not os.path.exists(path):
            self._log(f"No existing state at {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.from_state_dict(state)
            self._log(f"Loaded state from {path}")
        except Exception as e:
            self._log(f"Failed to load state from {path}: {e}")

    # history

    def _push_history(self, ts: float, metrics: Dict[str, float]):
        self.history.append((ts, dict(metrics)))
        cutoff = ts - self.history_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def _extract_series(self, key: str) -> List[Tuple[float, float]]:
        return [(ts, m.get(key, 0.0)) for ts, m in self.history]

    # prediction

    def _predict_series_multi_engine(self, series: List[Tuple[float, float]], horizon: float) -> float:
        if len(series) < 3:
            return series[-1][1]

        t0 = series[0][0]
        xs = [ts - t0 for ts, _ in series]
        ys = [v for _, v in series]
        n = len(xs)

        alpha = 0.3
        ewma = ys[0]
        for v in ys[1:]:
            ewma = alpha * v + (1 - alpha) * ewma

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n)) or 1e-9
        slope = num / den
        trend_pred = ys[-1] + slope * horizon

        baseline = mean_y

        h_norm = min(1.0, horizon / 60.0)
        w_ewma = 1.0 - 0.5 * h_norm
        w_trend = 0.3 * h_norm
        w_base = 1.0 - w_ewma - w_trend

        pred = w_ewma * ewma + w_trend * trend_pred + w_base * baseline
        return max(0.0, min(1.0, pred))

    def _predict_multi_horizon(self) -> Dict[str, Dict[str, float]]:
        if not self.history:
            self._log("No history yet, predictions are empty.")
            return {"1s": {}, "5s": {}, "30s": {}, "120s": {}}

        horizons = {"1s": 1.0, "5s": 5.0, "30s": 30.0, "120s": 120.0}
        latest_ts, latest_metrics = self.history[-1]
        predictions = {h: {} for h in horizons.keys()}

        for key, current_value in latest_metrics.items():
            series = self._extract_series(key)
            if len(series) < 3:
                for h in horizons.keys():
                    predictions[h][key] = current_value
                continue
            for h_name, dt in horizons.items():
                pred = self._predict_series_multi_engine(series[-60:], dt)
                predictions[h_name][key] = pred

        try:
            feats = self._build_feature_vector_for_inference()
            hw_out = self.movidius.infer(feats)
            for msg in self.movidius.get_trace():
                self._log(f"[Movidius] {msg}")
            if hw_out:
                self._log(f"Hardware inference corrections: {hw_out}")
        except Exception:
            self._log("Hardware inference failed; continuing with CPU-only predictions.")
            traceback.print_exc()

        return predictions

    def _build_feature_vector_for_inference(self) -> List[float]:
        recent = list(self.history)[-10:]
        keys = ["cpu_load", "mem_load", "disk_io", "net_io"]
        vector = []
        for _, m in recent:
            for k in keys:
                vector.append(float(m.get(k, 0.0)))
        return vector

    # integrity

    def _self_check(self, metrics: Dict[str, float], predictions: Dict[str, Dict[str, float]]) -> float:
        score = 1.0

        if len(self.history) < 10:
            score *= 0.6

        required = ["cpu_load", "mem_load"]
        missing = [k for k in required if k not in metrics]
        if missing:
            score *= 0.7

        for _, pred_map in predictions.items():
            for _, v in pred_map.items():
                if not (0.0 <= v <= 1.0):
                    score *= 0.5
                    break

        return max(0.0, min(1.0, score))

    # meta-confidence

    def _compute_meta_confidence(
        self,
        predictions: Dict[str, Dict[str, float]],
        current_metrics: Dict[str, float],
    ) -> float:
        if not self.history:
            return 0.5

        horizon_keys = list(predictions.keys())
        per_metric_conf = []
        for key in current_metrics.keys():
            vals = []
            for h in horizon_keys:
                v = predictions.get(h, {}).get(key)
                if v is not None:
                    vals.append(v)
            if len(vals) > 1:
                mean_v = sum(vals) / len(vals)
                var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                c = math.exp(-var * 10.0)
                per_metric_conf.append(c)
        horizon_conf = sum(per_metric_conf) / len(per_metric_conf) if per_metric_conf else 0.5

        trend_conf = self._trend_stability_confidence()
        sensor_noise = current_metrics.get("sensor_noise", 0.1)
        sensor_conf = math.exp(-sensor_noise * 3.0)
        reinforcement_conf = self._reinforcement_confidence()
        turbulence_conf = self._turbulence_confidence()

        self._log(
            f"Meta-conf components: horizon={horizon_conf:.3f}, "
            f"trend={trend_conf:.3f}, sensor={sensor_conf:.3f}, "
            f"reinforcement={reinforcement_conf:.3f}, turbulence={turbulence_conf:.3f}"
        )

        weights = {
            "horizon": 0.25,
            "trend": 0.25,
            "sensor": 0.15,
            "reinforcement": 0.2,
            "turbulence": 0.15,
        }
        raw = (
            horizon_conf * weights["horizon"]
            + trend_conf * weights["trend"]
            + sensor_conf * weights["sensor"]
            + reinforcement_conf * weights["reinforcement"]
            + turbulence_conf * weights["turbulence"]
        )

        meta_conf = 0.7 * self.last_confidence + 0.3 * raw
        meta_conf *= (0.5 + 0.5 * self.model_integrity)
        return max(0.0, min(1.0, meta_conf))

    def _trend_stability_confidence(self) -> float:
        keys = ["cpu_load", "mem_load", "disk_io", "net_io"]
        window = list(self.history)[-30:]
        if len(window) < 5:
            return 0.5
        per_metric = []
        for key in keys:
            vals = [m.get(key, 0.0) for _, m in window]
            mean_v = sum(vals) / len(vals)
            var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            per_metric.append(math.exp(-var * 10.0))
        return sum(per_metric) / len(per_metric) if per_metric else 0.5

    def _reinforcement_confidence(self) -> float:
        if not self.reinforcement:
            return 0.5
        total_s = 0
        total_f = 0
        total_stability = 0.0
        total_count = 0
        for stats in self.reinforcement.values():
            total_s += stats["success"]
            total_f += stats["failure"]
            total_stability += stats["stability_score_sum"]
            total_count += stats["count"]
        if total_count == 0:
            return 0.5
        success_rate = total_s / max(1, (total_s + total_f))
        avg_stability = total_stability / total_count
        conf = 0.7 * success_rate + 0.3 * avg_stability
        return max(0.0, min(1.0, conf))

    def _turbulence_confidence(self) -> float:
        window = list(self.history)[-10:]
        if len(window) < 3:
            return 0.5
        keys = ["cpu_load", "mem_load"]
        total_deriv = 0.0
        count = 0
        for key in keys:
            prev = None
            for _, m in window:
                v = m.get(key, 0.0)
                if prev is not None:
                    total_deriv += abs(v - prev)
                    count += 1
                prev = v
        if count == 0:
            return 0.5
        avg_deriv = total_deriv / count
        return math.exp(-avg_deriv * 5.0)

    # health

    def _compute_health_score(
        self,
        metrics: Dict[str, float],
        predictions: Dict[str, Dict[str, float]],
        meta_conf: float,
    ) -> float:
        cpu = metrics.get("cpu_load", 0.0)
        mem = metrics.get("mem_load", 0.0)
        disk = metrics.get("disk_io_norm", metrics.get("disk_io", 0.0))
        net = metrics.get("net_io", metrics.get("net_bandwidth_norm", 0.0))
        deep_ram = metrics.get("deep_ram_usage", 0.0)
        cache_pressure = metrics.get("cache_pressure", 0.0)

        resource_load = (cpu + mem + 0.5 * disk + 0.5 * net + deep_ram + cache_pressure) / 4.0
        resource_score = math.exp(-resource_load * 2.5)

        fut_cpu = 0.5 * (
            predictions.get("5s", {}).get("cpu_load", cpu)
            + predictions.get("30s", {}).get("cpu_load", cpu)
        )
        fut_mem = 0.5 * (
            predictions.get("5s", {}).get("mem_load", mem)
            + predictions.get("30s", {}).get("mem_load", mem)
        )
        future_load = (fut_cpu + fut_mem) / 2.0
        future_score = math.exp(-future_load * 2.0)

        turbulence = 1.0 - self._turbulence_confidence()

        raw = (
            0.4 * resource_score
            + 0.3 * future_score
            + 0.2 * meta_conf
            + 0.1 * (1.0 - turbulence)
        )

        health = 0.6 * self.last_health_score + 0.4 * raw
        return max(0.0, min(1.0, health))

    # reinforcement

    def _update_reinforcement(self, context: Dict[str, Any], health: float):
        outcome = context.get("outcome")
        if not outcome:
            return
        stance = context.get("stance", "Balanced")
        hour = time.localtime().tm_hour
        key = (stance, self.meta_state, hour)
        stats = self.reinforcement[key]

        stability_score = context.get("stability_score", health)
        stats["stability_score_sum"] += float(stability_score)
        stats["count"] += 1

        if outcome in ("win", "stable"):
            stats["success"] += 1
        elif outcome in ("loss", "overload"):
            stats["failure"] += 1

    def _update_dynamic_stance_thresholds(self):
        lr = 0.02
        stance_stats = defaultdict(lambda: {"success": 0, "failure": 0, "count": 0})
        for (stance, meta, hour), stats in self.reinforcement.items():
            agg = stance_stats[stance]
            agg["success"] += stats["success"]
            agg["failure"] += stats["failure"]
            agg["count"] += stats["count"]

        for stance, agg in stance_stats.items():
            s = agg["success"]
            f = agg["failure"]
            total = s + f
            if total == 0:
                continue
            success_rate = s / total
            base = self.stance_thresholds.get(stance, {"risk_low": 0.3, "risk_high": 0.6})
            target_high = 0.8 - 0.4 * success_rate
            target_high = max(0.3, min(0.9, target_high))
            target_low = max(0.1, target_high - 0.3)

            new_low = (1 - lr) * base["risk_low"] + lr * target_low
            new_high = (1 - lr) * base["risk_high"] + lr * target_high
            self.stance_thresholds[stance] = {
                "risk_low": max(0.05, min(0.7, new_low)),
                "risk_high": max(0.2, min(0.95, new_high)),
            }

    # risk / actions

    def _compute_risk_and_actions(
        self,
        predictions: Dict[str, Dict[str, float]],
        meta_conf: float,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> Tuple[float, str, List[str]]:
        stance = context.get("stance", "Balanced")
        thresholds = self.stance_thresholds.get(stance, {"risk_low": 0.3, "risk_high": 0.6})

        cpu_120 = predictions.get("120s", {}).get("cpu_load", metrics.get("cpu_load", 0.0))
        mem_120 = predictions.get("120s", {}).get("mem_load", metrics.get("mem_load", 0.0))
        cpu_30 = predictions.get("30s", {}).get("cpu_load", metrics.get("cpu_load", 0.0))
        mem_30 = predictions.get("30s", {}).get("mem_load", metrics.get("mem_load", 0.0))

        long_term = (cpu_120 + mem_120) / 2.0
        mid_term = (cpu_30 + mem_30) / 2.0

        if self.meta_state in (MetaState.SENTINEL, MetaState.RECOVERY_FLOW):
            risk_raw = 0.6 * long_term + 0.4 * mid_term
        else:
            risk_raw = 0.4 * long_term + 0.6 * mid_term

        risk = risk_raw * (0.7 + 0.3 * meta_conf)
        risk = max(0.0, min(1.0, risk))
        risk *= (0.6 + 0.4 * self.model_integrity)
        risk = max(0.0, min(1.0, risk))

        actions = []
        if risk > thresholds["risk_high"]:
            actions.append("SHRINK_DEEP_RAM")
            actions.append("REDUCE_INGESTION")
            actions.append("SLOW_THREAD_EXPANSION")
            actions.append("PREPARE_FALLBACK_PATHS")
        elif risk > thresholds["risk_low"]:
            actions.append("TRIM_DEEP_RAM")
            actions.append("LIMIT_INGESTION")
            actions.append("CAP_THREAD_GROWTH")

        if risk < thresholds["risk_low"] * 0.7:
            if stance == "Beast" and self.meta_state == MetaState.HYPER_FLOW:
                actions.append("ALLOW_THREAD_EXPANSION")
                actions.append("INCREASE_DEEP_RAM_CACHE")

        if risk < thresholds["risk_low"]:
            band = "low"
        elif risk < thresholds["risk_high"]:
            band = "medium"
        else:
            band = "high"

        return risk, band, actions

    # meta-state

    def _update_meta_state(self, risk_score: float, health: float, context: Dict[str, Any]):
        requested = context.get("requested_meta_state")
        if requested in META_STATE_CONFIG:
            self._log(f"Meta-state manually forced to {requested}.")
            self.meta_state = requested
            return

        current = self.meta_state

        if current == MetaState.SENTINEL:
            if health > 0.85 and risk_score < 0.3:
                self.meta_state = MetaState.HYPER_FLOW
            elif health < 0.5 and risk_score > 0.7:
                self.meta_state = MetaState.RECOVERY_FLOW

        elif current == MetaState.HYPER_FLOW:
            if risk_score > 0.8 or health < 0.6:
                self.meta_state = MetaState.SENTINEL

        elif current == MetaState.RECOVERY_FLOW:
            if health > 0.7 and risk_score < 0.4:
                self.meta_state = MetaState.SENTINEL

        elif current == MetaState.DEEP_DREAM:
            if risk_score > 0.4:
                self.meta_state = MetaState.SENTINEL

    # fingerprints

    def _update_behavioral_fingerprints(
        self,
        metrics: Dict[str, float],
        predictions: Dict[str, Dict[str, float]],
        context: Dict[str, Any],
        risk_score: float,
        health_score: float,
    ) -> int:
        fp_vec = self._build_fingerprint_vector(metrics, predictions, risk_score, health_score)
        label = self._infer_fingerprint_label(context, risk_score, health_score)
        stance = context.get("stance", "Balanced")
        mode = context.get("mode", "default")

        best_id = None
        best_sim = 0.0
        for fp in self.fingerprints:
            sim = self._cosine_similarity(fp["vector"], fp_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = fp["id"]

        if best_sim > 0.95 and best_id is not None:
            for fp in self.fingerprints:
                if fp["id"] == best_id:
                    fp["count"] += 1
                    fp["last_label"] = label
                    fp["labels"][label] = fp["labels"].get(label, 0) + 1
                    fp["last_updated"] = time.time()
                    break
            return best_id
        else:
            self._fp_id_counter += 1
            fp_id = self._fp_id_counter
            self.fingerprints.append({
                "id": fp_id,
                "vector": fp_vec,
                "primary_mode": mode,
                "primary_stance": stance,
                "labels": {label: 1},
                "last_label": label,
                "count": 1,
                "created": time.time(),
                "last_updated": time.time(),
            })
            return fp_id

    def _build_fingerprint_vector(
        self,
        metrics: Dict[str, float],
        predictions: Dict[str, Dict[str, float]],
        risk_score: float,
        health_score: float,
    ) -> List[float]:
        keys = [
            "cpu_load", "mem_load", "disk_io_norm", "net_io",
            "deep_ram_usage", "thread_count_norm", "cache_pressure",
            "backup_pressure", "net_alert_rate", "gpu_cache_fill",
            "thermal_pressure", "disk_queue_pressure", "vram_pressure",
        ]
        v = []
        for k in keys:
            v.append(float(metrics.get(k, 0.0)))
            v.append(float(predictions.get("5s", {}).get(k, metrics.get(k, 0.0))))
            v.append(float(predictions.get("30s", {}).get(k, metrics.get(k, 0.0))))
        v.append(risk_score)
        v.append(health_score)
        return v

    def _infer_fingerprint_label(
        self,
        context: Dict[str, Any],
        risk_score: float,
        health_score: float,
    ) -> str:
        outcome = context.get("outcome")
        if outcome == "overload":
            return "overload"
        if outcome == "win":
            return "beast_win"
        if outcome == "stable":
            return "stable"
        if outcome == "loss":
            return "loss"

        if risk_score > 0.8:
            return "overload"
        if health_score > 0.8 and risk_score < 0.4:
            return "stable"
        if health_score > 0.7 and risk_score > 0.5 and context.get("stance") == "Beast":
            return "beast_win"
        return "neutral"

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ============================================================
#  GUI PANEL
# ============================================================

class BrainCortexPanel:
    def __init__(
        self,
        parent,
        brain: HybridBrain,
        local_path: str,
        deep_ram: DeepRamOrgan,
        backup: BackupEngineOrgan,
        network: NetworkWatcherOrgan,
        gpu: GPUCacheOrgan,
        thermal: ThermalOrgan,
        disk: DiskOrgan,
        vram: VRAMOrgan,
        swarm: SwarmNodeOrgan,
        coach: AICoachOrgan,
    ):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter is not available on this system.")

        self.brain = brain
        self.local_path = local_path
        self.network_path: Optional[str] = None
        self.deep_ram = deep_ram
        self.backup = backup
        self.network = network
        self.gpu = gpu
        self.thermal = thermal
        self.disk = disk
        self.vram = vram
        self.swarm = swarm
        self.coach = coach

        self.manual_stance: Optional[str] = None
        self.manual_meta_state: Optional[str] = None

        self.frame = ttk.LabelFrame(parent, text="Hybrid Brain Cortex – Nerve Center")

        self.lbl_health = ttk.Label(self.frame, text="Health: --")
        self.lbl_risk = ttk.Label(self.frame, text="Risk: -- (--)")
        self.lbl_conf = ttk.Label(self.frame, text="Meta-Conf: --")
        self.lbl_integrity = ttk.Label(self.frame, text="Model Integrity: --")

        self.progress_health = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate", length=140)
        self.progress_risk = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate", length=140)
        self.progress_conf = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate", length=140)
        self.progress_integrity = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate", length=140)

        self.lbl_meta_state = ttk.Label(self.frame, text="Meta-State: --")
        self.lbl_stance_thresholds = ttk.Label(self.frame, text="Stance thresholds: --")

        self.lbl_deepram = ttk.Label(self.frame, text="Deep RAM: --")
        self.lbl_backup = ttk.Label(self.frame, text="Backup: --")
        self.lbl_network = ttk.Label(self.frame, text="Network: --")
        self.lbl_gpu = ttk.Label(self.frame, text="GPU Cache: --")
        self.lbl_thermal = ttk.Label(self.frame, text="Thermal: --")
        self.lbl_disk = ttk.Label(self.frame, text="Disk: --")
        self.lbl_vram = ttk.Label(self.frame, text="VRAM: --")
        self.lbl_swarm = ttk.Label(self.frame, text="Swarm: --")
        self.lbl_coach = ttk.Label(self.frame, text="Coach: --", wraplength=420, justify="left")

        self.lbl_actions = ttk.Label(self.frame, text="Actions: --", wraplength=420, justify="left")

        self.lbl_local_mem = ttk.Label(self.frame, text=f"Local memory: {self.local_path}")
        self.lbl_network_mem = ttk.Label(self.frame, text="Network memory: (not set)")

        self.btn_save = ttk.Button(self.frame, text="Save Memory Now", command=self._save_memory_now)
        self.btn_set_network = ttk.Button(self.frame, text="Set Network Drive (SMB)", command=self._set_network_path)

        self.lbl_stance_ui = ttk.Label(self.frame, text="Stance override:")
        self.cmb_stance = ttk.Combobox(self.frame, values=["", "Calm", "Balanced", "Beast"], state="readonly")
        self.cmb_stance.bind("<<ComboboxSelected>>", self._on_stance_change)
        self.lbl_meta_ui = ttk.Label(self.frame, text="Meta-state override:")
        self.cmb_meta = ttk.Combobox(
            self.frame,
            values=["", MetaState.SENTINEL, MetaState.HYPER_FLOW, MetaState.RECOVERY_FLOW, MetaState.DEEP_DREAM],
            state="readonly",
        )
        self.cmb_meta.bind("<<ComboboxSelected>>", self._on_meta_change)

        self.txt_reason = tk.Text(self.frame, height=7, width=80, state="disabled", wrap="word")
        self.txt_reason.configure(font=("Consolas", 8))

        self._hist_health: List[float] = []
        self._hist_risk: List[float] = []
        self.graph_canvas = tk.Canvas(self.frame, height=120, bg="#101010")

        row = 0
        self.lbl_health.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.progress_health.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1
        self.lbl_risk.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.progress_risk.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1
        self.lbl_conf.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.progress_conf.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1
        self.lbl_integrity.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.progress_integrity.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1

        self.lbl_meta_state.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_stance_thresholds.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1

        self.lbl_deepram.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_backup.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_network.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_gpu.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_thermal.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_disk.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_vram.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_swarm.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_coach.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1

        self.lbl_actions.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1

        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=4); row += 1

        self.lbl_local_mem.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1
        self.lbl_network_mem.grid(row=row, column=0, columnspan=2, sticky="w", padx=3, pady=2); row += 1

        self.btn_save.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.btn_set_network.grid(row=row, column=1, sticky="e", padx=3, pady=2); row += 1

        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=4); row += 1

        self.lbl_stance_ui.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.cmb_stance.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1
        self.lbl_meta_ui.grid(row=row, column=0, sticky="w", padx=3, pady=2)
        self.cmb_meta.grid(row=row, column=1, sticky="ew", padx=3, pady=2); row += 1

        ttk.Label(self.frame, text="Internal reasoning (tail):").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=3, pady=(5, 0)
        ); row += 1
        self.txt_reason.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=3, pady=(0, 3)); row += 1

        ttk.Label(self.frame, text="Health / Risk History:").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=3, pady=(5, 0)
        ); row += 1
        self.graph_canvas.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=3, pady=(0, 5))

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(row, weight=1)

    def get_frame(self):
        return self.frame

    def _on_stance_change(self, event=None):
        val = self.cmb_stance.get().strip()
        self.manual_stance = val or None

    def _on_meta_change(self, event=None):
        val = self.cmb_meta.get().strip()
        self.manual_meta_state = val or None

    def _save_memory_now(self):
        self.brain.save_state(self.local_path)
        if self.network_path:
            self.brain.save_state(self.network_path)

    def _set_network_path(self):
        if simpledialog is None:
            return
        path = simpledialog.askstring(
            "Network Drive Path",
            "Enter SMB/UNC path for network memory\n(e.g. \\\\SERVER\\Share\\HybridBrain\\brain_state.json):",
            initialvalue=self.network_path or ""
        )
        if not path:
            return
        try:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self.network_path = path
            self.lbl_network_mem.config(text=f"Network memory: {self.network_path}")
            self.brain.save_state(self.network_path)
        except Exception as e:
            self.brain._log(f"Failed to set network path {path}: {e}")

    def _draw_graph(self):
        w = int(self.graph_canvas.winfo_width() or 400)
        h = int(self.graph_canvas.winfo_height() or 120)
        self.graph_canvas.delete("all")
        if not self._hist_health:
            return
        n = len(self._hist_health)
        if n < 2:
            return

        def scale(idx, val):
            x = (idx / (n - 1)) * (w - 10) + 5
            y = h - (val * (h - 10) + 5)
            return x, y

        self.graph_canvas.create_line(5, 5, 5, h - 5, fill="#444")
        self.graph_canvas.create_line(5, h - 5, w - 5, h - 5, fill="#444")

        coords_h = []
        for i, val in enumerate(self._hist_health):
            coords_h.extend(scale(i, val))
        self.graph_canvas.create_line(*coords_h, fill="#00ff00", width=2)

        coords_r = []
        for i, val in enumerate(self._hist_risk):
            coords_r.extend(scale(i, val))
        self.graph_canvas.create_line(*coords_r, fill="#ff4040", width=2)

    def update_view(self, brain_output: Dict[str, Any]):
        health = float(brain_output.get("health_score", 0.0))
        risk = float(brain_output.get("risk_score", 0.0))
        conf = float(brain_output.get("meta_confidence", 0.0))
        band = brain_output.get("risk_band", "--")
        meta_state = brain_output.get("meta_state", "--")
        thresholds = brain_output.get("stance_thresholds", {})
        actions = brain_output.get("actions", [])
        reason_tail = brain_output.get("reasoning_log_tail", [])
        integrity = float(brain_output.get("model_integrity", 1.0))

        self.lbl_health.config(text=f"Health: {health:.2f}")
        self.lbl_risk.config(text=f"Risk: {risk:.2f} ({band})")
        self.lbl_conf.config(text=f"Meta-Conf: {conf:.2f}")
        self.lbl_integrity.config(text=f"Model Integrity: {integrity:.2f}")
        self.lbl_meta_state.config(text=f"Meta-State: {meta_state}")

        self.progress_health["value"] = max(0, min(100, int(health * 100)))
        self.progress_risk["value"] = max(0, min(100, int(risk * 100)))
        self.progress_conf["value"] = max(0, min(100, int(conf * 100)))
        self.progress_integrity["value"] = max(0, min(100, int(integrity * 100)))

        if isinstance(thresholds, dict):
            st = thresholds.get("Balanced", thresholds.get("Calm", {}))
            low = st.get("risk_low", None)
            high = st.get("risk_high", None)
            if low is not None and high is not None:
                self.lbl_stance_thresholds.config(
                    text=f"Stance thresholds (Balanced): low={low:.2f}, high={high:.2f}"
                )
            else:
                self.lbl_stance_thresholds.config(text="Stance thresholds: n/a")
        else:
            self.lbl_stance_thresholds.config(text="Stance thresholds: n/a")

        if actions:
            self.lbl_actions.config(text=f"Actions: {', '.join(actions)}")
        else:
            self.lbl_actions.config(text="Actions: (none)")

        self.lbl_deepram.config(text=f"Deep RAM: {self.deep_ram.describe()}")
        self.lbl_backup.config(text=f"Backup: {self.backup.describe()}")
        self.lbl_network.config(text=f"Network: {self.network.describe()}")
        self.lbl_gpu.config(text=f"GPU Cache: {self.gpu.describe()}")
        self.lbl_thermal.config(text=f"Thermal: {self.thermal.describe()}")
        self.lbl_disk.config(text=f"Disk: {self.disk.describe()}")
        self.lbl_vram.config(text=f"VRAM: {self.vram.describe()}")
        self.lbl_swarm.config(text=f"Swarm: {self.swarm.describe()}")
        self.lbl_coach.config(text=f"Coach: {self.coach.describe()}")

        self.txt_reason.configure(state="normal")
        self.txt_reason.delete("1.0", "end")
        for line in reason_tail:
            self.txt_reason.insert("end", line + "\n")
        self.txt_reason.configure(state="disabled")

        self._hist_health.append(health)
        self._hist_risk.append(risk)
        max_len = 120
        if len(self._hist_health) > max_len:
            self._hist_health = self._hist_health[-max_len:]
            self._hist_risk = self._hist_risk[-max_len:]

        self._draw_graph()


# ============================================================
#  WINDOWS HOOKS: FPS, PING, INPUT LATENCY
# ============================================================

GAME_PROCESS_NAMES = ["Back4Blood.exe", "back4blood.exe"]


class PresentMonFPS:
    def __init__(self):
        self.fps = 60.0
        self._running = False
        self.proc = None

    def start(self):
        if self._running:
            return
        self._running = True

        cmd = [
            "PresentMon.exe",
            "--no_csv",
            "--output_stdout",
            "--process_name", "Back4Blood.exe",
            "--simple"
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except Exception as e:
            print("Failed to start PresentMon:", e)
            self._running = False
            return

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fps_pattern = re.compile(r"(\d+\.\d+)\s+FPS")
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            m = fps_pattern.search(line)
            if m:
                try:
                    self.fps = float(m.group(1))
                except Exception:
                    pass

    def get_fps(self) -> float:
        return self.fps


def get_b4b_server_ip() -> Optional[str]:
    if psutil is None:
        return None
    for proc in psutil.process_iter(attrs=["name", "connections"]):
        try:
            if proc.info["name"] in GAME_PROCESS_NAMES:
                for conn in proc.connections(kind="inet"):
                    if conn.raddr:
                        return conn.raddr.ip
        except Exception:
            continue
    return None


def ping_server(ip: Optional[str]) -> Optional[float]:
    if not ip:
        return None
    try:
        output = subprocess.check_output(
            ["ping", "-n", "1", "-w", "200", ip],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"Average = (\d+)ms", output)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


WH_MOUSE_LL = 14
WM_MOUSEMOVE = 0x0200

LowLevelMouseProc = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class InputLatencyTracker:
    def __init__(self):
        self.last_event_time = time.perf_counter()
        self.last_latency_ms = 20.0
        self.hook_id = None
        self._start_hook()

    def _mouse_proc(self, nCode, wParam, lParam):
        if nCode == 0 and wParam == WM_MOUSEMOVE:
            now = time.perf_counter()
            latency = (now - self.last_event_time) * 1000.0
            self.last_latency_ms = latency
            self.last_event_time = now
        return user32.CallNextHookEx(self.hook_id, nCode, wParam, lParam)

    def _start_hook(self):
        CMPFUNC = LowLevelMouseProc(self._mouse_proc)
        self._callback = CMPFUNC
        self.hook_id = user32.SetWindowsHookExW(
            WH_MOUSE_LL,
            self._callback,
            kernel32.GetModuleHandleW(None),
            0
        )

        def msg_loop():
            msg = wintypes.MSG()
            while True:
                bRet = user32.GetMessageW(ctypes.byref(msg), 0, 0, 0)
                if bRet == 0 or bRet == -1:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

        threading.Thread(target=msg_loop, daemon=True).start()

    def get_latency(self) -> float:
        return self.last_latency_ms


# ============================================================
#  BACK 4 BLOOD ANALYZER
# ============================================================

class Back4BloodAnalyzer:
    def __init__(
        self,
        brain: HybridBrain,
        deep_ram: DeepRamOrgan,
        backup: BackupEngineOrgan,
        network: NetworkWatcherOrgan,
        gpu: GPUCacheOrgan,
        thermal: ThermalOrgan,
        disk: DiskOrgan,
        vram: VRAMOrgan,
        swarm: SwarmNodeOrgan,
        coach: AICoachOrgan,
        panel: BrainCortexPanel,
    ):
        self.brain = brain
        self.deep_ram = deep_ram
        self.backup = backup
        self.network = network
        self.gpu = gpu
        self.thermal = thermal
        self.disk = disk
        self.vram = vram
        self.swarm = swarm
        self.coach = coach
        self.panel = panel

        self.game_running = False
        self.current_stance = "Balanced"
        self.session_outcome = None
        self.last_stability_score = 0.8

        self.fps_reader = PresentMonFPS()
        self.fps_reader.start()

        self.input_latency = InputLatencyTracker()
        self.server_ip = None
        self.last_ping = 40.0
        self.last_fps = 60.0

        self.last_context: Dict[str, Any] = {}

    def is_game_running(self) -> bool:
        if psutil is None:
            return False
        for proc in psutil.process_iter(attrs=["name"]):
            try:
                if proc.info["name"] in GAME_PROCESS_NAMES:
                    return True
            except Exception:
                continue
        return False

    def get_game_fps(self) -> float:
        self.last_fps = self.fps_reader.get_fps()
        return self.last_fps

    def get_game_ping(self) -> float:
        if self.server_ip is None:
            self.server_ip = get_b4b_server_ip()
        ping_val = ping_server(self.server_ip)
        if ping_val is not None:
            self.last_ping = ping_val
        return self.last_ping

    def get_input_latency_ms(self) -> float:
        return self.input_latency.get_latency()

    def get_match_phase(self) -> str:
        sec = int(time.time()) % 120
        if sec < 20:
            return "lobby"
        elif sec < 50:
            return "early_wave"
        elif sec < 90:
            return "mid_wave"
        elif sec < 110:
            return "boss"
        else:
            return "clutch"

    def build_system_snapshot(self) -> dict:
        if psutil is None:
            base = {
                "cpu_load": 0.2,
                "mem_load": 0.3,
                "disk_io_norm": 0.2,
                "net_io": 0.2,
                "thread_count_norm": 0.3,
                "sensor_noise": 0.05,
            }
        else:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_load = min(1.0, cpu_percent / 100.0)

            vm = psutil.virtual_memory()
            mem_load = min(1.0, vm.percent / 100.0)

            disk_io_norm = 0.3
            net_io_norm = 0.2

            thread_count = 0
            try:
                for p in psutil.process_iter():
                    thread_count += p.num_threads()
            except Exception:
                pass
            thread_count_norm = min(1.0, thread_count / 2000.0)

            sensor_noise = 0.05

            base = {
                "cpu_load": cpu_load,
                "mem_load": mem_load,
                "disk_io_norm": disk_io_norm,
                "net_io": net_io_norm,
                "thread_count_norm": thread_count_norm,
                "sensor_noise": sensor_noise,
            }

        base.update(self.deep_ram.metrics())
        base.update(self.backup.metrics())
        base.update(self.network.metrics())
        base.update(self.gpu.metrics())
        base.update(self.thermal.metrics())
        base.update(self.disk.metrics())
        base.update(self.vram.metrics())
        return base

    def build_context(self) -> dict:
        phase = self.get_match_phase()
        fps = self.get_game_fps()
        ping = self.get_game_ping()
        latency = self.get_input_latency_ms()

        if self.panel.manual_stance:
            self.current_stance = self.panel.manual_stance
        else:
            if phase in ("boss", "clutch") and fps > 80 and ping < 70:
                self.current_stance = "Beast"
            elif mem_overshoot_risk(latency, ping):
                self.current_stance = "Calm"
            else:
                self.current_stance = "Balanced"

        ctx = {
            "stance": self.current_stance,
            "mode": "Back4Blood",
            "phase": phase,
            "fps": fps,
            "ping": ping,
            "input_latency_ms": latency,
        }

        if self.panel.manual_meta_state:
            ctx["requested_meta_state"] = self.panel.manual_meta_state

        if self.session_outcome is not None:
            ctx["outcome"] = self.session_outcome
            ctx["stability_score"] = self.last_stability_score

        self.last_context = ctx
        return ctx

    def tick(self) -> dict:
        self.game_running = self.is_game_running()

        snapshot = self.build_system_snapshot()
        if not self.game_running:
            ctx = {
                "stance": self.panel.manual_stance or "Calm",
                "mode": "idle",
            }
            if self.panel.manual_meta_state:
                ctx["requested_meta_state"] = self.panel.manual_meta_state
            self.last_context = ctx
            out = self.brain.update(snapshot, context=ctx)
            self.last_stability_score = self._compute_stability_score(out.get("health_score", 0.0), ctx)
            return out

        ctx = self.build_context()
        out = self.brain.update(snapshot, context=ctx)
        health = out.get("health_score", 0.0)
        self.last_stability_score = self._compute_stability_score(health, ctx)
        return out

    def _compute_stability_score(self, health: float, ctx: dict) -> float:
        mode = ctx.get("mode", "Back4Blood")
        profile = self.brain.mode_profiles.get(mode, self.brain.mode_profiles.get("Back4Blood"))

        fps = ctx.get("fps", self.last_fps)
        ping = ctx.get("ping", self.last_ping)
        latency = ctx.get("input_latency_ms", self.input_latency.get_latency())

        t_fps = 90.0
        if fps >= t_fps:
            fps_score = 1.0
        elif fps <= 40:
            fps_score = 0.0
        else:
            fps_score = (fps - 40) / (t_fps - 40)

        if ping <= 50:
            ping_score = 1.0
        elif ping >= 180:
            ping_score = 0.0
        else:
            ping_score = 1.0 - (ping - 50) / 130.0

        if latency <= profile.get("max_latency_ms", 40):
            lat_score = 1.0
        elif latency >= 120:
            lat_score = 0.0
        else:
            lat_score = 1.0 - (latency - profile.get("max_latency_ms", 40)) / (120 - profile.get("max_latency_ms", 40))

        w_fps = profile.get("weight_fps", 0.4)
        stability = (
            0.4 * health +
            w_fps * fps_score +
            0.2 * ping_score +
            0.15 * lat_score
        )
        return max(0.0, min(1.0, stability))


def mem_overshoot_risk(latency_ms: float, ping_ms: float) -> bool:
    return latency_ms > 60.0 or ping_ms > 140.0


# ============================================================
#  MAIN NERVE CENTER
# ============================================================

def main():
    if not TK_AVAILABLE:
        print("Tkinter not available. Install it to see the cortex GUI.")
        return
    if psutil is None:
        print("psutil not available. Install it with 'pip install psutil'.")
        return

    root = tk.Tk()
    root.title("System Nerve Center – Full Organs & Swarm")

    brain = HybridBrain()
    brain.load_state(LOCAL_MEMORY_PATH)

    deep_ram = DeepRamOrgan()
    backup = BackupEngineOrgan()
    network = NetworkWatcherOrgan()
    gpu = GPUCacheOrgan()
    thermal = ThermalOrgan()
    disk = DiskOrgan()
    vram = VRAMOrgan()
    swarm = SwarmNodeOrgan(node_id="node1")
    coach = AICoachOrgan()

    panel = BrainCortexPanel(
        root, brain, LOCAL_MEMORY_PATH,
        deep_ram, backup, network, gpu,
        thermal, disk, vram, swarm, coach
    )
    panel.get_frame().pack(fill="both", expand=True, padx=5, pady=5)

    analyzer = Back4BloodAnalyzer(
        brain, deep_ram, backup, network, gpu,
        thermal, disk, vram, swarm, coach, panel
    )

    last_save_time = time.time()

    def tick():
        nonlocal last_save_time
        out = analyzer.tick()

        for action in out.get("actions", []):
            deep_ram.apply_action(action)
            backup.apply_action(action)
            network.apply_action(action)
            gpu.apply_action(action)
            # thermal/disk/vram have no actions yet

        swarm_metrics = swarm.update(out.get("health_score", 0.0), out.get("risk_score", 0.0))
        # Optionally, could feed swarm metrics back into next snapshot

        coach.update(out, analyzer.last_context)

        panel.update_view(out)

        now = time.time()
        if now - last_save_time > 30.0:
            brain.save_state(LOCAL_MEMORY_PATH)
            if panel.network_path:
                brain.save_state(panel.network_path)
            last_save_time = now

        root.after(500, tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()

