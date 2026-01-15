#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Living Nerve Center – HybridBrain + Tri-Stance DecisionEngine + Organs + TransportCache + GUI + Reboot Memory
(Part 1/3 – Imports, NPU, Movidius, utilities, organs)
"""

# -------------------------
# Autoloader for libraries
# -------------------------
import os
import sys
import math
import time
import json
import random
import threading
import queue
from datetime import datetime, timedelta
from collections import deque

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    raise RuntimeError("Tkinter is required for the Nerve Center GUI")

# -------------------------
# Replica NPU (software NPU)
# -------------------------
class ReplicaNPU:
    """Software-simulated Neural Processing Unit (NPU)"""

    def __init__(self, cores=8, frequency_ghz=1.2):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.cycles = 0
        self.energy = 0.0  # arbitrary units

    def mac(self, a, b):
        """Multiply-Accumulate"""
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        """Parallel MAC operation"""
        assert len(v1) == len(v2)
        chunk = math.ceil(len(v1) / self.cores)
        result = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            result += partial
        return result

    def matmul(self, A, B):
        """Matrix multiplication using NPU parallelism"""
        result = [[0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                col = [B[k][j] for k in range(len(B))]
                result[i][j] = self.vector_mac(A[i], col)
        return result

    def relu(self, x):
        self.cycles += 1
        return max(0.0, x)

    def sigmoid(self, x):
        self.cycles += 2
        return 1 / (1 + math.exp(-x))

    def activate(self, tensor, mode="relu"):
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                if mode == "relu":
                    tensor[i][j] = self.relu(tensor[i][j])
                elif mode == "sigmoid":
                    tensor[i][j] = self.sigmoid(tensor[i][j])
        return tensor

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
        }

# -------------------------
# Movidius Inference Engine (stub + training hook)
# -------------------------
class MovidiusInferenceEngine:
    """
    Stub for ONNX/Movidius model.
    In real deployment, load ONNX and run inference on telemetry vector.
    """

    def __init__(self):
        self.model_loaded = False
        self.confidence_scale = 1.0
        self.model_path = None

    def load_model(self, path: str):
        # TODO: integrate real ONNX runtime
        self.model_loaded = True
        self.model_path = path

    def train_model(self, dataset_path: str, output_model_path: str = None):
        """
        Training hook: in real implementation, you would:
        - Load dataset from dataset_path
        - Train ONNX model
        - Save to output_model_path
        Here we just simulate success and bump confidence_scale.
        """
        time.sleep(0.5)  # pretend to train
        self.confidence_scale = min(1.5, self.confidence_scale + 0.1)
        if output_model_path:
            self.model_path = output_model_path
        self.model_loaded = True

    def infer_risk(self, features):
        if not self.model_loaded:
            avg = sum(features) / max(len(features), 1)
            risk = min(max(avg, 0.0), 1.0)
            return risk, 0.3
        avg = sum(features) / max(len(features), 1)
        risk = min(max(avg, 0.0), 1.0)
        return risk, 0.8 * self.confidence_scale

# -------------------------
# Utility: EWMA, variance, trend, turbulence
# -------------------------
def ewma(values, alpha=0.3):
    if not values:
        return 0.0
    s = values[0]
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s

def linear_trend(values):
    n = len(values)
    if n < 2:
        return 0.0
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(values) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, values))
    den = sum((xi - mean_x) ** 2 for xi in x) or 1.0
    slope = num / den
    return slope

def variance(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)

def turbulence(values):
    if len(values) < 3:
        return 0.0
    diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return ewma(diffs, alpha=0.5)

# -------------------------
# Base Organ
# -------------------------
class OrganBase:
    def __init__(self, name):
        self.name = name
        self.health = 1.0
        self.risk = 0.0
        # Appetite is what the brain modulates; appetite_scale is user-level scaling.
        self.appetite = 1.0
        self.appetite_scale = 1.0
        # Micro-recovery aggressiveness (1.0 = normal, >1 = more aggressive)
        self.micro_recovery_factor = 1.0
        self.last_metrics = {}
        self.micro_recovery_enabled = True

    def update(self):
        pass

    def micro_recovery(self):
        if not self.micro_recovery_enabled:
            return
        if self.risk > 0.7:
            # Appetite shrinks faster if micro_recovery_factor is high
            self.appetite = max(0.1, self.appetite - 0.05 * self.micro_recovery_factor)

    def snapshot(self):
        return {
            "name": self.name,
            "health": self.health,
            "risk": self.risk,
            "appetite": self.appetite,
            "appetite_scale": self.appetite_scale,
            "micro_recovery_factor": self.micro_recovery_factor,
            "last_metrics": self.last_metrics,
        }

    def restore(self, data):
        self.health = data.get("health", 1.0)
        self.risk = data.get("risk", 0.0)
        self.appetite = data.get("appetite", 1.0)
        self.appetite_scale = data.get("appetite_scale", 1.0)
        self.micro_recovery_factor = data.get("micro_recovery_factor", 1.0)
        self.last_metrics = data.get("last_metrics", {})

# -------------------------
# Specific Organs
# -------------------------
class DeepRamOrgan(OrganBase):
    def __init__(self):
        super().__init__("DeepRam")
        self.target_ratio = 0.7

    def update(self):
        if psutil is None:
            self.health = 1.0
            self.risk = 0.2
            self.last_metrics = {"note": "psutil not available"}
            return
        mem = psutil.virtual_memory()
        used_ratio = mem.percent / 100.0
        self.risk = used_ratio
        self.health = 1.0 - used_ratio
        self.last_metrics = {
            "total": mem.total,
            "used": mem.used,
            "percent": mem.percent,
            "target_ratio": self.target_ratio,
            "appetite": self.appetite,
        }
        if self.risk > self.target_ratio:
            self.appetite = max(0.1, self.appetite - 0.05 * self.micro_recovery_factor)

class BackupEngineOrgan(OrganBase):
    def __init__(self):
        super().__init__("BackupEngine")
        self.integrity_score = 1.0

    def update(self):
        self.integrity_score = 1.0
        self.health = self.integrity_score
        self.risk = 1.0 - self.integrity_score
        self.last_metrics = {"integrity_score": self.integrity_score}

class NetworkWatcherOrgan(OrganBase):
    def __init__(self):
        super().__init__("NetworkWatcher")
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.last_time = time.time()

    def update(self):
        if psutil is None:
            self.health = 1.0
            self.risk = 0.2
            self.last_metrics = {"note": "psutil not available"}
            return
        now = time.time()
        net = psutil.net_io_counters()
        dt = max(now - self.last_time, 1e-3)
        sent_rate = (net.bytes_sent - self.last_bytes_sent) / dt
        recv_rate = (net.bytes_recv - self.last_bytes_recv) / dt
        self.last_bytes_sent = net.bytes_sent
        self.last_bytes_recv = net.bytes_recv
        self.last_time = now
        load = min((sent_rate + recv_rate) / (10 * 1024 * 1024), 1.0)
        self.risk = load
        self.health = 1.0 - load
        self.last_metrics = {
            "sent_rate": sent_rate,
            "recv_rate": recv_rate,
            "load": load,
            "appetite": self.appetite,
        }

class GPUCacheOrgan(OrganBase):
    def __init__(self):
        super().__init__("GPUCache")

    def update(self):
        if pynvml is None:
            self.health = 1.0
            self.risk = 0.2
            self.last_metrics = {"note": "pynvml not available"}
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_ratio = mem.used / mem.total
            self.risk = used_ratio
            self.health = 1.0 - used_ratio
            self.last_metrics = {
                "total": mem.total,
                "used": mem.used,
                "used_ratio": used_ratio,
                "appetite": self.appetite,
            }
        except Exception as e:
            self.health = 1.0
            self.risk = 0.3
            self.last_metrics = {"error": str(e)}

class ThermalOrgan(OrganBase):
    def __init__(self):
        super().__init__("Thermal")
        self.cpu_temp = 50.0
        self.gpu_temp = 50.0

    def update(self):
        temp = 50.0
        if psutil and hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                first = next(iter(temps.values()))
                if first:
                    temp = first[0].current
        self.cpu_temp = temp
        if pynvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                t = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self.gpu_temp = t
            except Exception:
                pass
        max_temp = max(self.cpu_temp, self.gpu_temp)
        risk = min(max((max_temp - 40) / 40.0, 0.0), 1.0)
        self.risk = risk
        self.health = 1.0 - risk
        self.last_metrics = {
            "cpu_temp": self.cpu_temp,
            "gpu_temp": self.gpu_temp,
            "appetite": self.appetite,
        }
        if self.risk > 0.7:
            self.appetite = max(0.1, self.appetite - 0.1 * self.micro_recovery_factor)

class DiskOrgan(OrganBase):
    def __init__(self):
        super().__init__("Disk")
        self.last_read = 0
        self.last_write = 0
        self.last_time = time.time()

    def update(self):
        if psutil is None:
            self.health = 1.0
            self.risk = 0.2
            self.last_metrics = {"note": "psutil not available"}
            return
        now = time.time()
        io = psutil.disk_io_counters()
        dt = max(now - self.last_time, 1e-3)
        read_rate = (io.read_bytes - self.last_read) / dt
        write_rate = (io.write_bytes - self.last_write) / dt
        self.last_read = io.read_bytes
        self.last_write = io.write_bytes
        self.last_time = now
        load = min((read_rate + write_rate) / (50 * 1024 * 1024), 1.0)
        self.risk = load
        self.health = 1.0 - load
        self.last_metrics = {
            "read_rate": read_rate,
            "write_rate": write_rate,
            "load": load,
            "appetite": self.appetite,
        }

class VRAMOrgan(OrganBase):
    def __init__(self):
        super().__init__("VRAM")

    def update(self):
        if pynvml is None:
            self.health = 1.0
            self.risk = 0.2
            self.last_metrics = {"note": "pynvml not available"}
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_ratio = mem.used / mem.total
            self.risk = used_ratio
            self.health = 1.0 - used_ratio
            self.last_metrics = {
                "total": mem.total,
                "used": mem.used,
                "used_ratio": used_ratio,
                "appetite": self.appetite,
            }
        except Exception as e:
            self.health = 1.0
            self.risk = 0.3
            self.last_metrics = {"error": str(e)}

class AICoachOrgan(OrganBase):
    def __init__(self):
        super().__init__("AICoach")
        self.coaching_score = 0.5

    def update(self):
        self.coaching_score = random.uniform(0.4, 0.9)
        self.health = self.coaching_score
        self.risk = 1.0 - self.coaching_score
        self.last_metrics = {
            "coaching_score": self.coaching_score,
            "appetite": self.appetite,
        }

class SwarmNodeOrgan(OrganBase):
    def __init__(self):
        super().__init__("SwarmNode")
        self.collective_risk = 0.3
        self.node_agreement = 0.8
        self.hive_density = 0.5

    def update(self):
        self.collective_risk = min(max(self.collective_risk + random.uniform(-0.05, 0.05), 0.0), 1.0)
        self.node_agreement = min(max(self.node_agreement + random.uniform(-0.05, 0.05), 0.0), 1.0)
        self.hive_density = min(max(self.hive_density + random.uniform(-0.05, 0.05), 0.0), 1.0)
        self.risk = self.collective_risk
        self.health = 1.0 - self.collective_risk
        self.last_metrics = {
            "collective_risk": self.collective_risk,
            "node_agreement": self.node_agreement,
            "hive_density": self.hive_density,
            "appetite": self.appetite,
        }

class Back4BloodAnalyzer(OrganBase):
    def __init__(self):
        super().__init__("Back4BloodAnalyzer")
        self.game_metrics = {}

    def update_with_game_data(self, metrics: dict):
        self.game_metrics = metrics
        chaos = metrics.get("chaos", 0.5)
        self.risk = min(max(chaos, 0.0), 1.0)
        self.health = 1.0 - self.risk
        self.last_metrics = metrics | {"appetite": self.appetite}

    def update(self):
        if not self.game_metrics:
            self.risk = 0.3
            self.health = 0.7
            self.last_metrics = {"note": "no live game data", "appetite": self.appetite}

class SelfIntegrityOrgan(OrganBase):
    def __init__(self):
        super().__init__("SelfIntegrity")
        self.integrity = 1.0
        self.issues = []

    def check_integrity(self, brain, organs):
        self.issues = []
        if not organs:
            self.issues.append("No organs registered")
        for o in organs:
            if o.health < 0.2:
                self.issues.append(f"Low health organ: {o.name}")
        if brain.prediction_drift > 0.5:
            self.issues.append("High prediction drift")
        if brain.model_integrity < 0.5:
            self.issues.append("Model integrity low")
        self.integrity = max(0.0, 1.0 - 0.2 * len(self.issues))
        self.health = self.integrity
        self.risk = 1.0 - self.integrity
        self.last_metrics = {"issues": self.issues, "integrity": self.integrity}

# -------------------------
# TransportCacheOrgan – “transport truck / front-door delivery”
# -------------------------
class TransportCacheOrgan(OrganBase):
    def __init__(self, max_items=128):
        super().__init__("TransportCache")
        self.max_items = max_items
        self.cache = {}          # key -> {"value": ..., "ts": ..., "hits": ...}
        self.freq = {}           # key -> count
        self.hit_rate = 0.0
        self.miss_rate = 0.0
        self._hits = 0
        self._misses = 0

    def make_key(self, request_fingerprint: dict) -> str:
        return json.dumps(request_fingerprint, sort_keys=True)

    def get(self, request_fingerprint: dict):
        key = self.make_key(request_fingerprint)
        entry = self.cache.get(key)
        if entry:
            self._hits += 1
            entry["hits"] += 1
            self.freq[key] = self.freq.get(key, 0) + 1
            return entry["value"]
        else:
            self._misses += 1
            return None

    def put(self, request_fingerprint: dict, value):
        key = self.make_key(request_fingerprint)
        if len(self.cache) >= self.max_items and key not in self.cache:
            if self.freq:
                victim = min(self.freq.items(), key=lambda kv: kv[1])[0]
                self.cache.pop(victim, None)
                self.freq.pop(victim, None)
        self.cache[key] = {
            "value": value,
            "ts": time.time(),
            "hits": self.cache.get(key, {}).get("hits", 0),
        }
        self.freq[key] = self.freq.get(key, 0) + 1

    def update(self):
        total = self._hits + self._misses
        if total > 0:
            self.hit_rate = self._hits / total
            self.miss_rate = self._misses / total
        self.health = self.hit_rate
        self.risk = 1.0 - self.hit_rate
        top_keys = sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
        self.last_metrics = {
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "items": len(self.cache),
            "top_keys": top_keys,
        }

# ============================================================
# PART 2 / 3 — PredictionBus, DecisionEngine, HybridBrain, State
# ============================================================

# -------------------------
# PredictionBus
# -------------------------
class PredictionBus:
    def __init__(self):
        self.current_risk = 0.3
        self.short = 0.3
        self.medium = 0.3
        self.long = 0.3
        self.best_guess = 0.3
        self.baseline = 0.3
        self.meta_conf = 0.5
        self.bottleneck = None
        self.timestamp = time.time()

    def update_from_brain(self, brain):
        p = brain.last_predictions
        self.short = p["short"]
        self.medium = p["medium"]
        self.long = p["long"]
        self.best_guess = p["best_guess"]
        self.baseline = p["baseline"]
        self.meta_conf = p["meta_conf"]
        self.timestamp = time.time()

    def update_current_risk(self, organs):
        if organs:
            self.current_risk = sum(o.risk for o in organs) / len(organs)
        else:
            self.current_risk = 0.3


# -------------------------
# Tri-Stance DecisionEngine
# -------------------------
class DecisionEngine:
    """
    Tri-Stance decision engine:
    - Conservative (A)
    - Balanced (B)
    - Beast (C)
    """

    def __init__(self):
        self.stance = "Balanced"
        self.decision_log = deque(maxlen=200)
        self.reinforcement_score = 0.0

    def decide(self, brain, organs, prediction_bus):
        deep_ram = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        thermal = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        vram = next((o for o in organs if isinstance(o, VRAMOrgan)), None)

        mem_ratio = deep_ram.risk if deep_ram else 0.3
        temp_high = max(thermal.cpu_temp if thermal else 50.0,
                        thermal.gpu_temp if thermal else 50.0)
        temp_risk = min(max((temp_high - 40) / 40.0, 0.0), 1.0)
        vram_ratio = vram.risk if vram else 0.3

        pred_risk = prediction_bus.best_guess
        headroom = max(0.0, 1.0 - pred_risk)

        bottleneck = None
        if organs:
            bottleneck = max(organs, key=lambda o: o.risk).name
        prediction_bus.bottleneck = bottleneck

        prev_stance = self.stance

        # Conservative triggers
        if (mem_ratio > 0.9 or temp_risk > 0.7 or pred_risk > 0.7):
            self.stance = "Conservative"

        # Beast triggers
        elif (headroom > 0.4 and temp_risk < 0.4 and
              self.reinforcement_score > -0.5 and pred_risk < 0.6):
            self.stance = "Beast"

        # Balanced fallback
        else:
            if deep_ram:
                if abs(deep_ram.risk - deep_ram.target_ratio) < 0.1 and pred_risk < 0.6:
                    self.stance = "Balanced"

        # Log stance changes
        if self.stance != prev_stance:
            self.decision_log.appendleft({
                "time": datetime.now().isoformat(timespec="seconds"),
                "from": prev_stance,
                "to": self.stance,
                "mem_ratio": mem_ratio,
                "temp_risk": temp_risk,
                "pred_risk": pred_risk,
                "vram_ratio": vram_ratio,
                "bottleneck": bottleneck,
            })

        # Adjust Deep RAM target
        if deep_ram:
            if self.stance == "Conservative":
                deep_ram.target_ratio = 0.6
            elif self.stance == "Balanced":
                deep_ram.target_ratio = 0.75
            elif self.stance == "Beast":
                deep_ram.target_ratio = 0.9

        return self.stance

    def reinforce_outcome(self, good: bool):
        delta = 0.1 if good else -0.1
        if self.stance == "Beast":
            self.reinforcement_score += delta
        elif self.stance == "Conservative":
            self.reinforcement_score -= delta
        self.reinforcement_score = max(-2.0, min(2.0, self.reinforcement_score))


# -------------------------
# HybridBrain
# -------------------------
class HybridBrain:
    META_STATES = ["Hyper-Flow", "Deep-Dream", "Sentinel", "Recovery-Flow"]
    STANCES = ["Conservative", "Balanced", "Beast"]

    def __init__(self):
        self.history = deque(maxlen=300)
        self.meta_state = "Sentinel"
        self.stance = "Balanced"
        self.meta_state_momentum = 0.0

        self.pattern_memory = []
        self.reinforcement_memory = []
        self.baseline_risk = 0.3
        self.model_integrity = 1.0
        self.prediction_drift = 0.0

        self.movidius = MovidiusInferenceEngine()
        self.npu = ReplicaNPU(cores=16, frequency_ghz=1.5)

        self.last_predictions = {
            "short": 0.3,
            "medium": 0.3,
            "long": 0.3,
            "best_guess": 0.3,
            "baseline": self.baseline_risk,
            "meta_conf": 0.5,
        }
        self.last_reasoning = []
        self.last_heatmap = {}

        self.last_auto_calibration = datetime.now()

        self.meta_state_profiles = {
            "Hyper-Flow": {"appetite": 1.2, "horizon_bias": 1.2, "dampening": 0.8},
            "Deep-Dream": {"appetite": 0.8, "horizon_bias": 1.5, "dampening": 0.9},
            "Sentinel": {"appetite": 1.0, "horizon_bias": 1.0, "dampening": 1.0},
            "Recovery-Flow": {"appetite": 0.7, "horizon_bias": 0.8, "dampening": 1.2},
        }

    # ---------------------------------------------------------
    # Main update (runs in background thread)
    # ---------------------------------------------------------
    def update(self, organs, decision_engine: DecisionEngine, prediction_bus: PredictionBus, transport_cache=None):
        profile = self.meta_state_profiles.get(self.meta_state, {"appetite": 1.0, "dampening": 1.0})
        appetite_factor = profile.get("appetite", 1.0)
        dampening = profile.get("dampening", 1.0)
        micro_factor = max(0.5, min(2.0, dampening))

        # Apply meta-state appetite + micro-recovery scaling
        for o in organs:
            o.appetite = o.appetite_scale * appetite_factor
            o.micro_recovery_factor = micro_factor

        # Compute current risk
        organ_risks = [o.risk for o in organs]
        current_risk = sum(organ_risks) / len(organ_risks) if organ_risks else 0.3
        self.history.append(current_risk)

        # Build fingerprint for TransportCache
        fingerprint = {
            "meta_state": self.meta_state,
            "stance": self.stance,
            "recent_risk": list(self.history)[-5:],
        }

        # Try cache first
        cached = transport_cache.get(fingerprint) if transport_cache else None

        if cached is not None:
            self.last_predictions = cached["predictions"]
            self.last_reasoning = cached["reasoning"]
            self.last_heatmap = cached["heatmap"]
            prediction_bus.update_from_brain(self)
            prediction_bus.update_current_risk(organs)
            self.stance = decision_engine.decide(self, organs, prediction_bus)
            return

        # Compute predictions
        short_pred, med_pred, long_pred, regime, engines, heatmap = self._predict_multi_horizon()
        best_guess, vote_detail = self._best_guess(engines)
        meta_conf = self._meta_confidence(engines)

        # Update memories
        self._update_pattern_memory(current_risk, best_guess)
        self._update_meta_state(current_risk, best_guess)
        self._auto_tune(current_risk, best_guess)
        self._auto_calibrate_if_needed()

        # Integrity
        self.prediction_drift = abs(best_guess - current_risk)
        self.model_integrity = max(0.0, 1.0 - self.prediction_drift)

        # Store predictions
        self.last_predictions = {
            "short": short_pred,
            "medium": med_pred,
            "long": long_pred,
            "best_guess": best_guess,
            "baseline": self.baseline_risk,
            "meta_conf": meta_conf,
        }

        self.last_reasoning = [
            f"Regime: {regime}",
            f"Short={short_pred:.2f}, Med={med_pred:.2f}, Long={long_pred:.2f}",
            f"BestGuess={best_guess:.2f}, MetaConf={meta_conf:.2f}",
            f"MetaState={self.meta_state}, Stance={self.stance}",
        ]
        self.last_heatmap = heatmap
        self.last_heatmap["best_guess_contributors"] = vote_detail

        # Push to prediction bus
        prediction_bus.update_from_brain(self)
        prediction_bus.update_current_risk(organs)

        # Decide stance
        self.stance = decision_engine.decide(self, organs, prediction_bus)

        # Store in cache
        if transport_cache:
            transport_cache.put(fingerprint, {
                "predictions": self.last_predictions,
                "reasoning": self.last_reasoning,
                "heatmap": self.last_heatmap,
            })

    # ---------------------------------------------------------
    # Prediction engines
    # ---------------------------------------------------------
    def _predict_multi_horizon(self):
        values = list(self.history) or [self.baseline_risk]

        ew = ewma(values, alpha=0.3)
        tr = linear_trend(values)
        var = variance(values)
        turb = turbulence(values)

        regime = self._detect_regime(var, turb, tr)

        features = [ew, tr, var, turb, self.baseline_risk]
        mov_risk, mov_conf = self.movidius.infer_risk(features)

        # NPU fusion
        A = [[ew, tr, var], [turb, self.baseline_risk, mov_risk]]
        B = [[0.4, -0.2], [0.1, 0.3], [-0.3, 0.5]]
        out = self.npu.matmul(A, B)
        out = self.npu.activate(out, mode="relu")
        npu_signal = min(max(sum(sum(row) for row in out) / 10.0, 0.0), 1.0)

        # Meta-state horizon bias
        profile = self.meta_state_profiles.get(self.meta_state, {"horizon_bias": 1.0})
        hb = profile.get("horizon_bias", 1.0)

        horizon_factor_short = 1.0 * hb
        horizon_factor_med = 1.0 * hb
        horizon_factor_long = 1.0 * hb

        if regime == "High-variance chaotic":
            horizon_factor_long *= 0.6
        elif regime == "Rising-load":
            horizon_factor_short *= 1.1
            horizon_factor_med *= 1.1
        elif regime == "Cooling-down":
            horizon_factor_long *= 0.9

        short_pred = min(max(ew + tr * 2 * horizon_factor_short, 0.0), 1.0)
        med_pred = min(max(ew + tr * 10 * horizon_factor_med, 0.0), 1.0)
        long_pred = min(max(ew + tr * 60 * horizon_factor_long, 0.0), 1.0)

        engines = {
            "ewma": ew,
            "trend": tr,
            "variance": var,
            "turbulence": turb,
            "movidius_risk": mov_risk,
            "movidius_conf": mov_conf,
            "npu_signal": npu_signal,
        }

        heatmap = {
            "ewma_weight": 0.2,
            "trend_weight": 0.15,
            "variance_weight": 0.15,
            "turbulence_weight": 0.15,
            "movidius_weight": 0.2,
            "npu_weight": 0.15,
        }

        return short_pred, med_pred, long_pred, regime, engines, heatmap

    def _detect_regime(self, var, turb, trend_val):
        if var < 0.01 and turb < 0.01:
            return "Low-variance stable"
        if var > 0.05 and turb > 0.05:
            return "High-variance chaotic"
        if trend_val > 0.0:
            return "Rising-load"
        if trend_val < 0.0:
            return "Cooling-down"
        return "Neutral"

    def _meta_confidence(self, engines):
        var = engines["variance"]
        turb = engines["turbulence"]
        mov_conf = engines["movidius_conf"]
        var_term = max(0.0, 1.0 - min(var * 10.0, 1.0))
        turb_term = max(0.0, 1.0 - min(turb * 10.0, 1.0))
        base_conf = 0.4 * var_term + 0.4 * turb_term + 0.2 * mov_conf
        return max(0.0, min(base_conf, 1.0))

    def _best_guess(self, engines):
        ew = engines["ewma"]
        tr = engines["trend"]
        var = engines["variance"]
        turb = engines["turbulence"]
        mov_risk = engines["movidius_risk"]
        mov_conf = engines["movidius_conf"]
        npu_signal = engines["npu_signal"]

        weights = {
            "ewma": 0.2,
            "trend": 0.15,
            "variance": 0.1,
            "turbulence": 0.1,
            "movidius": 0.25 * mov_conf,
            "npu": 0.2,
        }
        total_w = sum(weights.values()) or 1.0

        ew_norm = ew
        trend_norm = min(max(0.5 + tr * 10.0, 0.0), 1.0)
        var_norm = min(max(var * 5.0, 0.0), 1.0)
        turb_norm = min(max(turb * 5.0, 0.0), 1.0)
        mov_norm = mov_risk
        npu_norm = npu_signal

        best_guess = (
            ew_norm * weights["ewma"] +
            trend_norm * weights["trend"] +
            var_norm * weights["variance"] +
            turb_norm * weights["turbulence"] +
            mov_norm * weights["movidius"] +
            npu_norm * weights["npu"]
        ) / total_w

        vote_detail = {
            "ewma": ew_norm,
            "trend": trend_norm,
            "variance": var_norm,
            "turbulence": turb_norm,
            "movidius": mov_norm,
            "npu": npu_norm,
            "weights": weights,
        }
        return min(max(best_guess, 0.0), 1.0), vote_detail

    # ---------------------------------------------------------
    # Memory + Meta-state evolution
    # ---------------------------------------------------------
    def _update_pattern_memory(self, current_risk, best_guess):
        fingerprint = {
            "risk": current_risk,
            "best_guess": best_guess,
            "meta_state": self.meta_state,
            "stance": self.stance,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.pattern_memory.append(fingerprint)
        if len(self.pattern_memory) > 1000:
            self.pattern_memory.pop(0)

    def _update_meta_state(self, current_risk, best_guess):
        if current_risk < 0.3 and best_guess < 0.4:
            self.meta_state_momentum += 0.05
        elif current_risk > 0.7 or best_guess > 0.7:
            self.meta_state_momentum -= 0.05
        else:
            self.meta_state_momentum *= 0.95

        self.meta_state_momentum = max(-1.0, min(1.0, self.meta_state_momentum))

        prev_state = self.meta_state
        if self.meta_state == "Hyper-Flow":
            if self.meta_state_momentum < -0.3:
                self.meta_state = "Sentinel"
        elif self.meta_state == "Sentinel":
            if self.meta_state_momentum > 0.4:
                self.meta_state = "Hyper-Flow"
            elif self.meta_state_momentum < -0.4:
                self.meta_state = "Recovery-Flow"
        elif self.meta_state == "Recovery-Flow":
            if self.meta_state_momentum > 0.2 and current_risk < 0.5:
                self.meta_state = "Sentinel"
        elif self.meta_state == "Deep-Dream":
            if self.meta_state_momentum > 0.3:
                self.meta_state = "Sentinel"

        if prev_state != self.meta_state:
            self.reinforcement_memory.append({
                "from": prev_state,
                "to": self.meta_state,
                "momentum": self.meta_state_momentum,
                "time": datetime.now().isoformat(timespec="seconds"),
            })

    def _auto_tune(self, current_risk, best_guess):
        error = best_guess - self.baseline_risk
        profile = self.meta_state_profiles.get(self.meta_state, {"dampening": 1.0})
        damp = profile.get("dampening", 1.0)

        # Auto‑tune baseline risk
        self.baseline_risk += 0.01 * (error) / max(damp, 0.1)
        self.baseline_risk = max(0.0, min(1.0, self.baseline_risk))

    def _auto_calibrate_if_needed(self):
        if datetime.now() - self.last_auto_calibration > timedelta(hours=24):
            self._auto_calibrate()
            self.last_auto_calibration = datetime.now()

    def _auto_calibrate(self):
        if not self.history:
            return
        # Recompute baseline from long‑term history
        self.baseline_risk = sum(self.history) / len(self.history)

    # ---------------------------------------------------------
    # Snapshot / Restore
    # ---------------------------------------------------------
    def snapshot(self):
        return {
            "history": list(self.history),
            "meta_state": self.meta_state,
            "stance": self.stance,
            "meta_state_momentum": self.meta_state_momentum,
            "pattern_memory": self.pattern_memory,
            "reinforcement_memory": self.reinforcement_memory,
            "baseline_risk": self.baseline_risk,
            "model_integrity": self.model_integrity,
            "prediction_drift": self.prediction_drift,
            "meta_state_profiles": self.meta_state_profiles,
        }

    def restore(self, data):
        self.history = deque(data.get("history", []), maxlen=300)
        self.meta_state = data.get("meta_state", "Sentinel")
        self.stance = data.get("stance", "Balanced")
        self.meta_state_momentum = data.get("meta_state_momentum", 0.0)
        self.pattern_memory = data.get("pattern_memory", [])
        self.reinforcement_memory = data.get("reinforcement_memory", [])
        self.baseline_risk = data.get("baseline_risk", 0.3)
        self.model_integrity = data.get("model_integrity", 1.0)
        self.prediction_drift = data.get("prediction_drift", 0.0)
        self.meta_state_profiles = data.get("meta_state_profiles", self.meta_state_profiles)


# ============================================================
# OrganismState — Full System Snapshot + Restore
# ============================================================
class OrganismState:
    @staticmethod
    def snapshot(brain, organs, integrity_organ, decision_engine, prediction_bus):
        return {
            "brain": brain.snapshot(),
            "organs": {o.name: o.snapshot() for o in organs},
            "integrity": integrity_organ.snapshot(),
            "decision_engine": {
                "stance": decision_engine.stance,
                "reinforcement_score": decision_engine.reinforcement_score,
                "decision_log": list(decision_engine.decision_log),
            },
            "prediction_bus": {
                "current_risk": prediction_bus.current_risk,
                "short": prediction_bus.short,
                "medium": prediction_bus.medium,
                "long": prediction_bus.long,
                "best_guess": prediction_bus.best_guess,
                "baseline": prediction_bus.baseline,
                "meta_conf": prediction_bus.meta_conf,
                "bottleneck": prediction_bus.bottleneck,
            },
        }

    @staticmethod
    def restore(data, brain, organs, integrity_organ, decision_engine, prediction_bus):
        if "brain" in data:
            brain.restore(data["brain"])

        if "organs" in data:
            by_name = {o.name: o for o in organs}
            for name, od in data["organs"].items():
                if name in by_name:
                    by_name[name].restore(od)

        if "integrity" in data:
            integrity_organ.restore(data["integrity"])

        if "decision_engine" in data:
            de = data["decision_engine"]
            decision_engine.stance = de.get("stance", "Balanced")
            decision_engine.reinforcement_score = de.get("reinforcement_score", 0.0)
            decision_engine.decision_log = deque(de.get("decision_log", []), maxlen=200)

        if "prediction_bus" in data:
            pb = data["prediction_bus"]
            prediction_bus.current_risk = pb.get("current_risk", 0.3)
            prediction_bus.short = pb.get("short", 0.3)
            prediction_bus.medium = pb.get("medium", 0.3)
            prediction_bus.long = pb.get("long", 0.3)
            prediction_bus.best_guess = pb.get("best_guess", 0.3)
            prediction_bus.baseline = pb.get("baseline", 0.3)
            prediction_bus.meta_conf = pb.get("meta_conf", 0.5)
            prediction_bus.bottleneck = pb.get("bottleneck", None)

# ============================================================
# PART 3A — Full GUI (Nerve Center, Panels, Charts, Controls)
# ============================================================

class NerveCenterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HybridBrain Nerve Center — Full Organism Mode")
        self.root.geometry("1500x900")

        # Core system components
        self.brain = HybridBrain()
        self.prediction_bus = PredictionBus()
        self.decision_engine = DecisionEngine()
        self.integrity_organ = SelfIntegrityOrgan()

        # Organs
        self.organs = [
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
            TransportCacheOrgan(max_items=128),
        ]

        # Background thread queue
        self.brain_queue = queue.Queue()

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Tabs
        self._build_tab_main()
        self._build_tab_altered_states()
        self._build_tab_transport()
        self._build_tab_back4blood()
        self._build_tab_reboot_memory()

        # Start background worker
        threading.Thread(target=self._background_loop, daemon=True).start()

        # Start UI update loop
        self.root.after(50, self._tick)

    # ============================================================
    # TAB 1 — MAIN NERVE CENTER
    # ============================================================
    def _build_tab_main(self):
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Nerve Center")

        # Left column — Brain + Predictions
        left = ttk.Frame(self.tab_main)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # Brain panel
        frame_brain = ttk.LabelFrame(left, text="HybridBrain State")
        frame_brain.pack(fill="x", pady=5)

        self.lbl_meta_state = ttk.Label(frame_brain, text="Meta-State: Sentinel")
        self.lbl_meta_state.pack(anchor="w")

        self.lbl_stance = ttk.Label(frame_brain, text="Stance: Balanced")
        self.lbl_stance.pack(anchor="w")

        self.lbl_meta_conf = ttk.Label(frame_brain, text="Meta-Confidence: 0.50")
        self.lbl_meta_conf.pack(anchor="w")

        self.lbl_model_integrity = ttk.Label(frame_brain, text="Model Integrity: 1.00")
        self.lbl_model_integrity.pack(anchor="w")

        self.lbl_current_risk = ttk.Label(frame_brain, text="Current Risk: 0.30")
        self.lbl_current_risk.pack(anchor="w")

        # Reinforcement buttons
        frame_reinforce = ttk.LabelFrame(left, text="Reinforcement")
        frame_reinforce.pack(fill="x", pady=5)

        ttk.Button(frame_reinforce, text="Reinforce GOOD", command=lambda: self._reinforce(True)).pack(fill="x", pady=2)
        ttk.Button(frame_reinforce, text="Reinforce BAD", command=lambda: self._reinforce(False)).pack(fill="x", pady=2)

        # Prediction panel
        frame_pred = ttk.LabelFrame(left, text="Predictions")
        frame_pred.pack(fill="x", pady=5)

        self.lbl_pred_short = ttk.Label(frame_pred, text="Short: 0.30")
        self.lbl_pred_short.pack(anchor="w")

        self.lbl_pred_med = ttk.Label(frame_pred, text="Medium: 0.30")
        self.lbl_pred_med.pack(anchor="w")

        self.lbl_pred_long = ttk.Label(frame_pred, text="Long: 0.30")
        self.lbl_pred_long.pack(anchor="w")

        self.lbl_pred_best = ttk.Label(frame_pred, text="Best Guess: 0.30")
        self.lbl_pred_best.pack(anchor="w")

        # Reasoning panel
        frame_reason = ttk.LabelFrame(left, text="Reasoning Tail")
        frame_reason.pack(fill="both", expand=True, pady=5)

        self.txt_reason = tk.Text(frame_reason, height=15, width=40)
        self.txt_reason.pack(fill="both", expand=True)

        # Right column — Organs + Chart
        right = ttk.Frame(self.tab_main)
        right.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Organs panel
        frame_organs = ttk.LabelFrame(right, text="Organs")
        frame_organs.pack(fill="x")

        self.org_labels = {}
        for o in self.organs:
            lbl = ttk.Label(frame_organs, text=f"{o.name}: H=1.00 R=0.00")
            lbl.pack(anchor="w")
            self.org_labels[o.name] = lbl

        # Chart panel
        frame_chart = ttk.LabelFrame(right, text="Prediction Graph")
        frame_chart.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(frame_chart, bg="black", height=300)
        self.canvas.pack(fill="both", expand=True)

    # ============================================================
    # TAB 2 — ALTERED STATES CORTEX
    # ============================================================
    def _build_tab_altered_states(self):
        self.tab_states = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_states, text="Altered States Cortex")

        frame = ttk.LabelFrame(self.tab_states, text="Meta-State Controls")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.state_sliders = {}

        for state in HybridBrain.META_STATES:
            sub = ttk.LabelFrame(frame, text=state)
            sub.pack(fill="x", pady=5)

            appetite = tk.DoubleVar(value=self.brain.meta_state_profiles[state]["appetite"])
            horizon = tk.DoubleVar(value=self.brain.meta_state_profiles[state]["horizon_bias"])
            damp = tk.DoubleVar(value=self.brain.meta_state_profiles[state]["dampening"])

            ttk.Label(sub, text="Appetite").pack(anchor="w")
            tk.Scale(sub, from_=0.5, to=2.0, resolution=0.05,
                     orient="horizontal", variable=appetite,
                     command=lambda v, s=state: self._update_state_profile(s)).pack(fill="x")

            ttk.Label(sub, text="Horizon Bias").pack(anchor="w")
            tk.Scale(sub, from_=0.5, to=2.0, resolution=0.05,
                     orient="horizontal", variable=horizon,
                     command=lambda v, s=state: self._update_state_profile(s)).pack(fill="x")

            ttk.Label(sub, text="Dampening").pack(anchor="w")
            tk.Scale(sub, from_=0.5, to=2.0, resolution=0.05,
                     orient="horizontal", variable=damp,
                     command=lambda v, s=state: self._update_state_profile(s)).pack(fill="x")

            self.state_sliders[state] = {
                "appetite": appetite,
                "horizon": horizon,
                "damp": damp,
            }

    # ============================================================
    # TAB 3 — TRANSPORT CACHE PANEL
    # ============================================================
    def _build_tab_transport(self):
        self.tab_transport = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_transport, text="Transport Cache")

        frame = ttk.LabelFrame(self.tab_transport, text="TransportCache Status")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.lbl_cache_hits = ttk.Label(frame, text="Hit Rate: 0.00")
        self.lbl_cache_hits.pack(anchor="w")

        self.lbl_cache_miss = ttk.Label(frame, text="Miss Rate: 0.00")
        self.lbl_cache_miss.pack(anchor="w")

        self.txt_cache_top = tk.Text(frame, height=10)
        self.txt_cache_top.pack(fill="both", expand=True)

    # ============================================================
    # TAB 4 — BACK 4 BLOOD ANALYZER
    # ============================================================
    def _build_tab_back4blood(self):
        self.tab_b4b = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_b4b, text="Back 4 Blood Analyzer")

        frame = ttk.LabelFrame(self.tab_b4b, text="Game Telemetry")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.lbl_b4b_risk = ttk.Label(frame, text="Chaos Risk: 0.30")
        self.lbl_b4b_risk.pack(anchor="w")

        ttk.Button(frame, text="Inject Test Chaos",
                   command=lambda: self._inject_b4b({"chaos": random.uniform(0, 1)})).pack(pady=5)

    # ============================================================
    # TAB 5 — REBOOT MEMORY
    # ============================================================
    def _build_tab_reboot_memory(self):
        self.tab_reboot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_reboot, text="Reboot Memory")

        frame = ttk.LabelFrame(self.tab_reboot, text="Reboot Memory Persistence")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="SMB / UNC Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        ttk.Button(frame, text="Pick SMB Path", command=self._pick_reboot_path).pack(anchor="w", pady=3)
        ttk.Button(frame, text="Test SMB Path", command=self._test_reboot_path).pack(anchor="w", pady=3)
        ttk.Button(frame, text="Save Memory for Reboot", command=self._save_reboot_memory).pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Load memory from SMB on startup",
                        variable=self.var_reboot_autoload).pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame, text="Status: Ready", anchor="w", fg="#00cc66")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

# ============================================================
# PART 3B — Background Thread, UI Updates, Callbacks, Main Loop
# ============================================================

    # ============================================================
    # BACKGROUND WORKER THREAD
    # ============================================================
    def _background_loop(self):
        while True:
            try:
                self._background_tick()
            except Exception as e:
                print("Background error:", e)
            time.sleep(1)  # 1 Hz brain update

    def _background_tick(self):
        # Update Back4Blood analyzer (if any live data)
        b4b = next((o for o in self.organs if isinstance(o, Back4BloodAnalyzer)), None)
        if b4b and b4b.game_metrics:
            b4b.update_with_game_data(b4b.game_metrics)

        # Update all organs
        for o in self.organs:
            o.update()
            o.micro_recovery()

        # Integrity check
        self.integrity_organ.check_integrity(self.brain, self.organs)

        # Transport cache
        transport_cache = next((o for o in self.organs if isinstance(o, TransportCacheOrgan)), None)

        # Brain update
        self.brain.update(
            self.organs,
            self.decision_engine,
            self.prediction_bus,
            transport_cache=transport_cache
        )

        # Push results to UI thread
        self.brain_queue.put({
            "pred": self.brain.last_predictions,
            "meta_state": self.brain.meta_state,
            "stance": self.brain.stance,
            "meta_conf": self.brain.last_predictions["meta_conf"],
            "model_integrity": self.brain.model_integrity,
            "current_risk": self.prediction_bus.current_risk,
            "reasoning": self.brain.last_reasoning,
            "heatmap": self.brain.last_heatmap,
            "organs": {o.name: (o.health, o.risk, o.last_metrics) for o in self.organs},
            "cache": next((o for o in self.organs if isinstance(o, TransportCacheOrgan)), None),
            "b4b": next((o for o in self.organs if isinstance(o, Back4BloodAnalyzer)), None),
        })

    # ============================================================
    # UI UPDATE LOOP
    # ============================================================
    def _tick(self):
        try:
            data = self.brain_queue.get_nowait()
            self._apply_brain_update(data)
        except queue.Empty:
            pass

        self.root.after(50, self._tick)

    # ============================================================
    # APPLY BACKGROUND RESULTS TO UI
    # ============================================================
    def _apply_brain_update(self, data):
        pred = data["pred"]

        # Brain labels
        self.lbl_meta_state.config(text=f"Meta-State: {data['meta_state']}")
        self.lbl_stance.config(text=f"Stance: {data['stance']}")
        self.lbl_meta_conf.config(text=f"Meta-Confidence: {data['meta_conf']:.2f}")
        self.lbl_model_integrity.config(text=f"Model Integrity: {data['model_integrity']:.2f}")
        self.lbl_current_risk.config(text=f"Current Risk: {data['current_risk']:.2f}")

        # Predictions
        self.lbl_pred_short.config(text=f"Short: {pred['short']:.2f}")
        self.lbl_pred_med.config(text=f"Medium: {pred['medium']:.2f}")
        self.lbl_pred_long.config(text=f"Long: {pred['long']:.2f}")
        self.lbl_pred_best.config(text=f"Best Guess: {pred['best_guess']:.2f}")

        # Reasoning
        self.txt_reason.delete("1.0", "end")
        for line in data["reasoning"]:
            self.txt_reason.insert("end", line + "\n")

        # Organs
        for name, (h, r, metrics) in data["organs"].items():
            if name in self.org_labels:
                self.org_labels[name].config(text=f"{name}: H={h:.2f} R={r:.2f}")

        # Transport cache
        cache = data["cache"]
        if cache:
            self.lbl_cache_hits.config(text=f"Hit Rate: {cache.hit_rate:.2f}")
            self.lbl_cache_miss.config(text=f"Miss Rate: {cache.miss_rate:.2f}")
            self.txt_cache_top.delete("1.0", "end")
            for k, v in cache.last_metrics.get("top_keys", []):
                self.txt_cache_top.insert("end", f"{k} → {v} hits\n")

        # Back4Blood
        b4b = data["b4b"]
        if b4b:
            self.lbl_b4b_risk.config(text=f"Chaos Risk: {b4b.risk:.2f}")

        # Chart
        self._draw_chart()

    # ============================================================
    # DRAW MICRO-CHART
    # ============================================================
    def _draw_chart(self):
        self.canvas.delete("all")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w < 10 or h < 10:
            return

        hist = list(self.brain.history)
        if len(hist) < 2:
            return

        # Normalize
        max_r = max(hist)
        min_r = min(hist)
        rng = max(0.01, max_r - min_r)

        def y(v):
            return h - ((v - min_r) / rng) * h

        # Draw history line
        for i in range(1, len(hist)):
            x1 = (i - 1) / len(hist) * w
            x2 = i / len(hist) * w
            self.canvas.create_line(x1, y(hist[i - 1]), x2, y(hist[i]), fill="#00ff88")

        # Predictions
        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        long = p["long"]
        best = p["best_guess"]
        base = p["baseline"]

        # Draw prediction markers
        self.canvas.create_line(0, y(short), w, y(short), fill="#ffaa00")
        self.canvas.create_line(0, y(med), w, y(med), fill="#ff6600")
        self.canvas.create_line(0, y(long), w, y(long), fill="#ff0000")
        self.canvas.create_line(0, y(best), w, y(best), fill="#00aaff", width=2)
        self.canvas.create_line(0, y(base), w, y(base), fill="#888888", dash=(4, 4))

    # ============================================================
    # CALLBACKS
    # ============================================================
    def _reinforce(self, good):
        self.decision_engine.reinforce_outcome(good)

    def _inject_b4b(self, metrics):
        b4b = next((o for o in self.organs if isinstance(o, Back4BloodAnalyzer)), None)
        if b4b:
            b4b.game_metrics = metrics

    def _update_state_profile(self, state):
        sliders = self.state_sliders[state]
        self.brain.meta_state_profiles[state]["appetite"] = sliders["appetite"].get()
        self.brain.meta_state_profiles[state]["horizon_bias"] = sliders["horizon"].get()
        self.brain.meta_state_profiles[state]["dampening"] = sliders["damp"].get()

    # ============================================================
    # SMB SAVE / LOAD
    # ============================================================
    def _pick_reboot_path(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_reboot_path.delete(0, "end")
            self.entry_reboot_path.insert(0, path)

    def _test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self.lbl_reboot_status.config(text="Status: No path", fg="red")
            return
        if os.path.isdir(path):
            self.lbl_reboot_status.config(text="Status: OK", fg="#00cc66")
        else:
            self.lbl_reboot_status.config(text="Status: Invalid", fg="red")

    def _save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path or not os.path.isdir(path):
            self.lbl_reboot_status.config(text="Status: Invalid path", fg="red")
            return

        state = OrganismState.snapshot(
            self.brain,
            self.organs,
            self.integrity_organ,
            self.decision_engine,
            self.prediction_bus
        )

        try:
            with open(os.path.join(path, "organism_state.json"), "w") as f:
                json.dump(state, f, indent=2)
            self.lbl_reboot_status.config(text="Status: Saved", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Error: {e}", fg="red")

    # ============================================================
    # MAIN LOOP
    # ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = NerveCenterGUI(root)
    root.mainloop()



