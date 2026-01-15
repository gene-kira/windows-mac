#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Living Nerve Center – HybridBrain + Tri-Stance DecisionEngine + Organs + GUI + Reboot Memory

Upgrades:
- Back4BloodAnalyzer wired to real-ish telemetry via psutil (process "Back4Blood")
- Reinforce Good / Bad Outcome buttons in GUI
- Altered States tab as Meta-State Cortex Viewer with per-state appetite/horizon/dampening sliders
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
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
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
# Movidius Inference Engine (stub)
# -------------------------
class MovidiusInferenceEngine:
    """
    Stub for ONNX/Movidius model.
    In real deployment, load ONNX and run inference on telemetry vector.
    """

    def __init__(self):
        self.model_loaded = False
        self.confidence_scale = 1.0

    def load_model(self, path: str):
        # TODO: integrate real ONNX runtime
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
        self.appetite = 1.0
        self.last_metrics = {}
        self.micro_recovery_enabled = True

    def update(self):
        pass

    def micro_recovery(self):
        if not self.micro_recovery_enabled:
            return
        if self.risk > 0.7:
            self.appetite = max(0.1, self.appetite - 0.05)

    def snapshot(self):
        return {
            "name": self.name,
            "health": self.health,
            "risk": self.risk,
            "appetite": self.appetite,
            "last_metrics": self.last_metrics,
        }

    def restore(self, data):
        self.health = data.get("health", 1.0)
        self.risk = data.get("risk", 0.0)
        self.appetite = data.get("appetite", 1.0)
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
        }
        if self.risk > self.target_ratio:
            self.appetite = max(0.1, self.appetite - 0.05)

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
        self.last_metrics = {"cpu_temp": self.cpu_temp, "gpu_temp": self.gpu_temp}
        if self.risk > 0.7:
            self.appetite = max(0.1, self.appetite - 0.1)

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
        self.last_metrics = {"coaching_score": self.coaching_score}

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
        self.last_metrics = metrics

    def update(self):
        if not self.game_metrics:
            self.risk = 0.3
            self.health = 0.7
            self.last_metrics = {"note": "no live game data"}

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

        if (mem_ratio > 0.9 or temp_risk > 0.7 or pred_risk > 0.7):
            self.stance = "Conservative"
        elif (headroom > 0.4 and temp_risk < 0.4 and
              self.reinforcement_score > -0.5 and pred_risk < 0.6):
            self.stance = "Beast"
        else:
            if deep_ram:
                if abs(deep_ram.risk - deep_ram.target_ratio) < 0.1 and pred_risk < 0.6:
                    self.stance = "Balanced"

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

        # Meta-state profiles: appetite, horizon_bias, dampening
        self.meta_state_profiles = {
            "Hyper-Flow": {"appetite": 1.2, "horizon_bias": 1.2, "dampening": 0.8},
            "Deep-Dream": {"appetite": 0.8, "horizon_bias": 1.5, "dampening": 0.9},
            "Sentinel": {"appetite": 1.0, "horizon_bias": 1.0, "dampening": 1.0},
            "Recovery-Flow": {"appetite": 0.7, "horizon_bias": 0.8, "dampening": 1.2},
        }

    def update(self, organs, decision_engine: DecisionEngine, prediction_bus: PredictionBus):
        organ_risks = [o.risk for o in organs]
        current_risk = sum(organ_risks) / len(organ_risks) if organ_risks else 0.3
        self.history.append(current_risk)

        short_pred, med_pred, long_pred, regime, engines, heatmap = self._predict_multi_horizon()
        best_guess, vote_detail = self._best_guess(engines)
        meta_conf = self._meta_confidence(engines)

        self._update_pattern_memory(current_risk, best_guess)
        self._update_meta_state(current_risk, best_guess)
        self._auto_tune(current_risk, best_guess)
        self._auto_calibrate_if_needed()

        self.prediction_drift = abs(best_guess - current_risk)
        self.model_integrity = max(0.0, 1.0 - self.prediction_drift)

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

        prediction_bus.update_from_brain(self)
        prediction_bus.update_current_risk(organs)

        self.stance = decision_engine.decide(self, organs, prediction_bus)

    def _predict_multi_horizon(self):
        values = list(self.history) or [self.baseline_risk]

        ew = ewma(values, alpha=0.3)
        tr = linear_trend(values)
        var = variance(values)
        turb = turbulence(values)

        regime = self._detect_regime(var, turb, tr)

        features = [ew, tr, var, turb, self.baseline_risk]
        mov_risk, mov_conf = self.movidius.infer_risk(features)

        A = [[ew, tr, var], [turb, self.baseline_risk, mov_risk]]
        B = [[0.4, -0.2], [0.1, 0.3], [-0.3, 0.5]]
        out = self.npu.matmul(A, B)
        out = self.npu.activate(out, mode="relu")
        npu_signal = min(max(sum(sum(row) for row in out) / 10.0, 0.0), 1.0)

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
        self.baseline_risk += 0.01 * error / max(damp, 0.1)
        self.baseline_risk = max(0.0, min(self.baseline_risk, 1.0))

    def _auto_calibrate_if_needed(self):
        if datetime.now() - self.last_auto_calibration > timedelta(hours=24):
            self._auto_calibrate()
            self.last_auto_calibration = datetime.now()

    def _auto_calibrate(self):
        if not self.history:
            return
        self.baseline_risk = sum(self.history) / len(self.history)

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

# -------------------------
# State serialization
# -------------------------
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

# -------------------------
# Nerve Center GUI
# -------------------------
class NerveCenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Living Nerve Center – HybridBrain")

        self.brain = HybridBrain()
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
        ]
        self.integrity_organ = SelfIntegrityOrgan()
        self.decision_engine = DecisionEngine()
        self.prediction_bus = PredictionBus()

        self.reboot_autoload = False
        self.reboot_path = ""

        self.meta_state_sliders = {}

        self._build_gui()
        self._start_update_loop()

    def _build_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # TAB 1: Nerve Center
        tab_main = ttk.Frame(notebook)
        notebook.add(tab_main, text="Nerve Center")

        frame_brain = ttk.LabelFrame(tab_main, text="Brain Cortex Panel")
        frame_brain.pack(side="left", fill="both", expand=True, padx=10, pady=10)

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

        self.canvas_chart = tk.Canvas(frame_brain, width=320, height=120, bg="#111111")
        self.canvas_chart.pack(pady=5)

        frame_buttons = ttk.Frame(frame_brain)
        frame_buttons.pack(anchor="w", pady=5)
        self.btn_reinforce_good = ttk.Button(frame_buttons, text="Reinforce GOOD outcome",
                                             command=lambda: self._reinforce(True))
        self.btn_reinforce_good.pack(side="left", padx=2)
        self.btn_reinforce_bad = ttk.Button(frame_buttons, text="Reinforce BAD outcome",
                                            command=lambda: self._reinforce(False))
        self.btn_reinforce_bad.pack(side="left", padx=2)

        frame_reason = ttk.LabelFrame(tab_main, text="Reasoning Tail / Heatmap")
        frame_reason.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.txt_reason = tk.Text(frame_reason, height=10, width=60)
        self.txt_reason.pack(fill="both", expand=True)

        # TAB 2: Altered States – Meta-State Cortex Viewer
        tab_states = ttk.Frame(notebook)
        notebook.add(tab_states, text="Altered States")

        ttk.Label(tab_states, text="Meta-State Cortex Viewer").pack(anchor="w", padx=10, pady=5)

        for ms in HybridBrain.META_STATES:
            frame_ms = ttk.LabelFrame(tab_states, text=ms)
            frame_ms.pack(fill="x", padx=10, pady=5)

            appetite_var = tk.DoubleVar(value=self.brain.meta_state_profiles[ms]["appetite"])
            horizon_var = tk.DoubleVar(value=self.brain.meta_state_profiles[ms]["horizon_bias"])
            damp_var = tk.DoubleVar(value=self.brain.meta_state_profiles[ms]["dampening"])

            ttk.Label(frame_ms, text="Appetite").grid(row=0, column=0, sticky="w")
            s_app = ttk.Scale(frame_ms, from_=0.5, to=1.5, orient="horizontal",
                              variable=appetite_var,
                              command=lambda v, state=ms: self._update_meta_profile(state))
            s_app.grid(row=0, column=1, sticky="ew")

            ttk.Label(frame_ms, text="Horizon Bias").grid(row=1, column=0, sticky="w")
            s_hor = ttk.Scale(frame_ms, from_=0.5, to=1.8, orient="horizontal",
                              variable=horizon_var,
                              command=lambda v, state=ms: self._update_meta_profile(state))
            s_hor.grid(row=1, column=1, sticky="ew")

            ttk.Label(frame_ms, text="Dampening").grid(row=2, column=0, sticky="w")
            s_damp = ttk.Scale(frame_ms, from_=0.5, to=1.8, orient="horizontal",
                               variable=damp_var,
                               command=lambda v, state=ms: self._update_meta_profile(state))
            s_damp.grid(row=2, column=1, sticky="ew")

            frame_ms.columnconfigure(1, weight=1)

            self.meta_state_sliders[ms] = {
                "appetite": appetite_var,
                "horizon_bias": horizon_var,
                "dampening": damp_var,
            }

        # TAB 3: Reboot Memory
        tab_reboot = ttk.Frame(notebook)
        notebook.add(tab_reboot, text="Reboot Memory")

        frame_reboot = ttk.LabelFrame(tab_reboot, text="Reboot Memory Persistence")
        frame_reboot.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame_reboot, text="SMB / UNC or Local Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame_reboot, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame_reboot, text="Pick Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame_reboot, text="Test Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame_reboot, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame_reboot,
            text="Load memory from path on startup",
            variable=self.var_reboot_autoload,
            command=self._update_autoload_flag
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame_reboot, text="Status: Ready", anchor="w", fg="#00cc66")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    # -------------------------
    # GUI commands
    # -------------------------
    def _reinforce(self, good: bool):
        self.decision_engine.reinforce_outcome(good)

    def _update_meta_profile(self, state_name: str):
        sliders = self.meta_state_sliders.get(state_name)
        if not sliders:
            return
        self.brain.meta_state_profiles[state_name] = {
            "appetite": sliders["appetite"].get(),
            "horizon_bias": sliders["horizon_bias"].get(),
            "dampening": sliders["dampening"].get(),
        }

    def cmd_pick_reboot_path(self):
        path = filedialog.asksaveasfilename(
            title="Select memory file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            messagebox.showwarning("Test Path", "No path specified.")
            return
        directory = os.path.dirname(path) or "."
        if os.path.isdir(directory):
            self.lbl_reboot_status.config(text="Status: Path OK", fg="#00cc66")
        else:
            self.lbl_reboot_status.config(text="Status: Path invalid", fg="#ff4444")

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            messagebox.showwarning("Save Memory", "No path specified.")
            return
        try:
            state = OrganismState.snapshot(
                self.brain, self.organs, self.integrity_organ,
                self.decision_engine, self.prediction_bus
            )
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
            self.lbl_reboot_status.config(text="Status: Memory saved", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Save failed: {e}", fg="#ff4444")

    def _update_autoload_flag(self):
        self.reboot_autoload = self.var_reboot_autoload.get()

    # -------------------------
    # Back 4 Blood telemetry wiring
    # -------------------------
    def _update_back4blood(self):
        analyzer = next((o for o in self.organs if isinstance(o, Back4BloodAnalyzer)), None)
        if analyzer is None or psutil is None:
            return
        chaos = 0.3
        fps_est = 60.0
        enemies = 10
        damage_in = 0.1
        try:
            for p in psutil.process_iter(["name", "cpu_percent", "num_threads"]):
                if p.info["name"] and "back4blood" in p.info["name"].lower():
                    cpu = p.info["cpu_percent"] / 100.0
                    threads = p.info["num_threads"] or 1
                    chaos = min(max(cpu + (threads / 200.0), 0.0), 1.0)
                    fps_est = max(10.0, 120.0 * (1.0 - cpu))
                    enemies = int(5 + 50 * cpu)
                    damage_in = min(max(cpu * 0.5, 0.0), 1.0)
                    break
        except Exception:
            pass

        metrics = {
            "chaos": chaos,
            "fps_est": fps_est,
            "enemies": enemies,
            "damage_in": damage_in,
        }
        analyzer.update_with_game_data(metrics)

    # -------------------------
    # Update loop
    # -------------------------
    def _start_update_loop(self):
        self._tick()
        self.root.after(1000, self._start_update_loop)

    def _tick(self):
        self._update_back4blood()

        for o in self.organs:
            o.update()
            o.micro_recovery()

        self.integrity_organ.check_integrity(self.brain, self.organs)
        self.brain.update(self.organs, self.decision_engine, self.prediction_bus)

        self._update_gui()

    def _update_gui(self):
        p = self.brain.last_predictions
        self.lbl_meta_state.config(text=f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.config(text=f"Stance: {self.brain.stance}")
        self.lbl_meta_conf.config(text=f"Meta-Confidence: {p['meta_conf']:.2f}")
        self.lbl_model_integrity.config(text=f"Model Integrity: {self.brain.model_integrity:.2f}")
        self.lbl_current_risk.config(text=f"Current Risk: {self.prediction_bus.current_risk:.2f}")

        self._draw_chart()
        self._update_reasoning()

    def _draw_chart(self):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])

        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#111111", outline="")

        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        long = p["long"]
        baseline = p["baseline"]
        best_guess = p["best_guess"]

        def y_from_val(v):
            return h - int(v * (h - 10)) - 5

        x_short = w * 0.2
        x_med = w * 0.5
        x_long = w * 0.8

        y_short = y_from_val(short)
        y_med = y_from_val(med)
        y_long = y_from_val(long)
        y_base = y_from_val(baseline)
        y_best = y_from_val(best_guess)

        self.canvas_chart.create_line(0, y_base, w, y_base, fill="#555555", dash=(2, 2))

        self.canvas_chart.create_line(x_short, y_short, x_med, y_med, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_med, y_med, x_long, y_long, fill="#00ccff", width=2)

        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.brain.stance, "#ffffff")
        self.canvas_chart.create_line(x_short, y_med, x_long, y_med, fill=stance_color, width=1)

        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)

        self.canvas_chart.create_text(5, 5, anchor="nw", fill="#aaaaaa",
                                      text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)")

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")

        self.txt_reason.insert(tk.END, "\nReasoning Heatmap:\n")
        for k, v in self.brain.last_heatmap.items():
            if k == "best_guess_contributors":
                continue
            self.txt_reason.insert(tk.END, f"  {k}: {v}\n")

        self.txt_reason.insert(tk.END, "\nBest-Guess Contributors:\n")
        contrib = self.brain.last_heatmap.get("best_guess_contributors", {})
        for k, v in contrib.items():
            if k == "weights":
                continue
            self.txt_reason.insert(tk.END, f"  {k}: {v:.3f}\n")

        weights = contrib.get("weights", {})
        if weights:
            self.txt_reason.insert(tk.END, "\nWeights:\n")
            for k, w in weights.items():
                self.txt_reason.insert(tk.END, f"  {k}: {w:.3f}\n")

# -------------------------
# Main
# -------------------------
def main():
    root = tk.Tk()
    app = NerveCenterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

