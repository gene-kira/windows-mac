#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Cipher Engine – Nerve Center (Augmented + Micro-Chart + Reboot Memory)

PART 1 of 2
- Imports
- Dependency loader
- Telemetry
- Prediction engines
- Pattern memory
- Judgment engine
- Situational cortex
- Collective health
- Organs
- Self-Integrity organ
- Augmentation engine
- HybridBrain (with Best-Guess)

Paste PART 1 and PART 2 into a single .py file, with PART 2 immediately following PART 1.
"""

import importlib
import math
import random
import sys
import time
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ------------------------------------------------------------
# 0. Auto-loader
# ------------------------------------------------------------

class DependencyLoader:
    def __init__(self):
        self.modules: Dict[str, Optional[Any]] = {}
        self.detect()

    def try_import(self, name: str):
        try:
            module = importlib.import_module(name)
            self.modules[name] = module
            return module
        except Exception:
            self.modules[name] = None
            return None

    def detect(self):
        self.try_import("numpy")
        self.try_import("onnxruntime")
        self.try_import("psutil")
        self.try_import("pynvml")

    def has(self, name: str) -> bool:
        return self.modules.get(name) is not None

    def get(self, name: str):
        return self.modules.get(name)


DEPS = DependencyLoader()

# ------------------------------------------------------------
# 1. Core telemetry & helpers
# ------------------------------------------------------------

@dataclass
class TelemetrySnapshot:
    timestamp: float
    cpu_load: float
    mem_load: float
    disk_load: float
    net_load: float
    temp: float
    vram_load: float
    anomaly_score: float
    turbulence: float
    label: Optional[str] = None


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# ------------------------------------------------------------
# 2. Prediction engines
# ------------------------------------------------------------

class EWMAPredictor:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class TrendPredictor:
    def __init__(self, window: int = 30):
        self.window = window
        self.samples: List[Tuple[float, float]] = []

    def update(self, t: float, x: float) -> float:
        self.samples.append((t, x))
        if len(self.samples) > self.window:
            self.samples.pop(0)
        if len(self.samples) < 2:
            return x
        t0, x0 = self.samples[0]
        t1, x1 = self.samples[-1]
        dt = max(t1 - t0, 1e-6)
        slope = (x1 - x0) / dt
        horizon = 5.0
        return clamp(x1 + slope * horizon, 0.0, 1.0)

    def trend_stability(self) -> float:
        if len(self.samples) < 3:
            return 0.5
        t0, x0 = self.samples[0]
        t1, x1 = self.samples[-1]
        dt = max(t1 - t0, 1e-6)
        slope = abs((x1 - x0) / dt)
        return clamp(1.0 - min(1.0, slope * 10.0))


class VarianceTracker:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> float:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.n < 2:
            return 0.0
        return clamp(self.M2 / (self.n - 1), 0.0, 1.0)


class TurbulenceTracker:
    def __init__(self):
        self.last: Optional[float] = None
        self.avg_delta = 0.0
        self.alpha = 0.3

    def update(self, x: float) -> float:
        if self.last is None:
            self.last = x
            return 0.0
        delta = abs(x - self.last)
        self.last = x
        self.avg_delta = self.alpha * delta + (1 - self.alpha) * self.avg_delta
        return clamp(self.avg_delta, 0.0, 1.0)


class MovidiusInferenceEngine:
    def __init__(self):
        self.enabled = DEPS.has("onnxruntime")
        self.model_conf = 0.6
        self.training_stats = {"samples": 0, "last_train_ts": None}

    def predict(self, snapshot: TelemetrySnapshot, horizon: float) -> Tuple[float, float]:
        if not self.enabled:
            load = (snapshot.cpu_load + snapshot.mem_load + snapshot.disk_load + snapshot.net_load) / 4.0
            risk = clamp(0.4 * load + 0.4 * snapshot.anomaly_score + 0.2 * snapshot.turbulence)
            self.model_conf = clamp(0.5 + 0.3 * (1.0 - snapshot.turbulence))
            return risk, self.model_conf
        else:
            load = (snapshot.cpu_load + snapshot.mem_load + snapshot.disk_load + snapshot.net_load) / 4.0
            risk = clamp(0.5 * load + 0.5 * snapshot.anomaly_score)
            self.model_conf = clamp(0.6 + 0.4 * (1.0 - snapshot.turbulence))
            return risk, self.model_conf

    def train(self, historical: List[Tuple[TelemetrySnapshot, float]]):
        if not historical:
            return
        self.training_stats["samples"] += len(historical)
        self.training_stats["last_train_ts"] = time.time()

# ------------------------------------------------------------
# 3. Pattern memory
# ------------------------------------------------------------

@dataclass
class FingerprintRecord:
    fingerprint: Tuple[int, int, int, int]
    label: str
    count: int = 0


class PatternMemory:
    def __init__(self):
        self.records: Dict[Tuple[int, int, int, int, str], FingerprintRecord] = {}

    def _fingerprint(self, snap: TelemetrySnapshot) -> Tuple[int, int, int, int]:
        def bucket(x: float) -> int:
            return min(3, int(x * 4.0))
        return (
            bucket(snap.cpu_load),
            bucket(snap.mem_load),
            bucket(snap.disk_load),
            bucket(snap.turbulence),
        )

    def record(self, snap: TelemetrySnapshot, label: str):
        fp = self._fingerprint(snap)
        key = (*fp, label)
        rec = self.records.get(key)
        if rec is None:
            rec = FingerprintRecord(fingerprint=fp, label=label, count=0)
            self.records[key] = rec
        rec.count += 1

    def summarize(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for rec in self.records.values():
            summary[rec.label] = summary.get(rec.label, 0) + rec.count
        return summary

# ------------------------------------------------------------
# 4. Judgment engine
# ------------------------------------------------------------

@dataclass
class ReinforcementStats:
    good: int = 0
    bad: int = 0

    @property
    def total(self) -> int:
        return self.good + self.bad


class JudgmentEngine:
    def __init__(self):
        self.stats = ReinforcementStats()
        self.bias_drift = 0.0
        self.frozen = False
        self.learning_rate = 1.0

    def reinforce_good(self):
        if self.frozen:
            return
        self.stats.good += max(1, int(1 * self.learning_rate))

    def reinforce_bad(self):
        if self.frozen:
            return
        self.stats.bad += max(1, int(1 * self.learning_rate))

    def judgment_confidence(self) -> float:
        t = self.stats.total
        c = 1.0 - math.exp(-t / 50.0)
        return clamp(c)

# ------------------------------------------------------------
# 5. Situational cortex / collective
# ------------------------------------------------------------

@dataclass
class SituationalCortexState:
    mission: str = "STABILITY"
    env: str = "CALM"
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    anticipation: str = "Unknown"
    mission_override: Optional[str] = None
    risk_tolerance: float = 0.5
    prioritize_learning_windows: bool = True

    def effective_mission(self) -> str:
        return self.mission_override or self.mission


class SituationalCortex:
    def __init__(self):
        self.state = SituationalCortexState()

    def update_from_snapshot(self, snap: TelemetrySnapshot):
        load = (snap.cpu_load + snap.mem_load + snap.disk_load + snap.net_load) / 4.0
        risk = clamp(0.4 * load + 0.6 * snap.anomaly_score)
        self.state.risk_score = risk
        self.state.opportunity_score = clamp(1.0 - risk)
        if risk < 0.3:
            self.state.env = "CALM"
        elif risk < 0.7:
            self.state.env = "TENSE"
        else:
            self.state.env = "DANGER"
        if self.state.env == "DANGER":
            self.state.anticipation = "Incoming instability"
        elif self.state.env == "TENSE":
            self.state.anticipation = "Volatile but controllable"
        else:
            self.state.anticipation = "Stable window"


@dataclass
class CollectiveState:
    collective_risk_score: float = 0.0
    hive_density: float = 0.1
    node_agreement: float = 1.0
    divergence_patterns: str = "None"
    hive_sync_mode: str = "Conservative"
    consensus_weighting: float = 0.5
    propagate_settings: bool = False
    isolated: bool = False


class CollectiveHealth:
    def __init__(self):
        self.state = CollectiveState()

    def update_from_local(self, local_risk: float):
        self.state.collective_risk_score = lerp(self.state.collective_risk_score, local_risk, 0.2)
        if self.state.collective_risk_score > 0.7:
            self.state.divergence_patterns = "Multiple nodes under stress"
        elif self.state.collective_risk_score > 0.4:
            self.state.divergence_patterns = "Some hotspots detected"
        else:
            self.state.divergence_patterns = "Nominal variance"

# ------------------------------------------------------------
# 6. Organs
# ------------------------------------------------------------

@dataclass
class OrganState:
    health: float = 1.0
    risk: float = 0.0
    appetite: float = 0.5
    ingestion_rate: float = 0.5
    notes: str = ""


class BaseOrgan:
    def __init__(self, name: str):
        self.name = name
        self.state = OrganState()

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        raise NotImplementedError

    def micro_recovery(self, snap: TelemetrySnapshot, risk_forecast: float):
        if risk_forecast > 0.7:
            self.state.appetite = clamp(self.state.appetite - 0.05)
            self.state.ingestion_rate = clamp(self.state.ingestion_rate - 0.05)
            if not self.state.notes:
                self.state.notes = "Micro-recovery"


class ThermalOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("ThermalOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.temp)
        self.state.health = clamp(1.0 - snap.temp)
        if snap.temp > 0.8:
            self.state.notes = "High thermal – cooling"
            self.state.appetite = clamp(self.state.appetite - 0.2)
        else:
            self.state.notes = "Thermal nominal"
        self.micro_recovery(snap, risk_forecast)


class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DiskOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.disk_load)
        self.state.health = clamp(1.0 - 0.7 * snap.disk_load)
        if snap.disk_load > 0.8:
            self.state.notes = "Stagger IO"
            self.state.ingestion_rate = clamp(self.state.ingestion_rate - 0.2)
        else:
            self.state.notes = "Disk normal"
        self.micro_recovery(snap, risk_forecast)


class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("NetworkWatcherOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.net_load)
        self.state.health = clamp(1.0 - 0.5 * snap.net_load)
        self.state.notes = "Network normal" if snap.net_load < 0.8 else "Pulse-limit bursts"
        self.micro_recovery(snap, risk_forecast)


class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRamOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.mem_load)
        self.state.health = clamp(1.0 - snap.mem_load)
        if risk_forecast > 0.7 or snap.mem_load > 0.85:
            self.state.notes = "Shrink Deep RAM"
            self.state.appetite = clamp(self.state.appetite - 0.2)
        else:
            self.state.notes = "Deep RAM normal"
        self.micro_recovery(snap, risk_forecast)


class VRAMOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("VRAMOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.vram_load)
        self.state.health = clamp(1.0 - snap.vram_load)
        self.state.notes = "VRAM cool-down" if snap.vram_load > 0.8 else "VRAM stable"
        self.micro_recovery(snap, risk_forecast)


class BackupEngineOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("BackupEngineOrgan")
        self.integrity_score = 1.0

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(snap.anomaly_score * 0.7 + risk_forecast * 0.3)
        self.state.health = clamp(1.0 - self.state.risk)
        self.integrity_score = clamp(self.integrity_score - 0.01 * self.state.risk)
        self.state.notes = f"Integrity={self.integrity_score:.2f}"
        self.micro_recovery(snap, risk_forecast)


class GPUCacheOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("GPUCacheOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(0.5 * snap.cpu_load + 0.5 * snap.vram_load)
        self.state.health = clamp(1.0 - self.state.risk)
        self.state.notes = "GPU cache adaptive"
        self.micro_recovery(snap, risk_forecast)


class AICoachOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("AICoachOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        opportunity = clamp(1.0 - snap.anomaly_score)
        self.state.risk = clamp(0.3 * snap.anomaly_score + 0.2 * snap.turbulence)
        self.state.health = opportunity
        self.state.notes = "Coaching windows" if opportunity > 0.5 else "Hold coaching"
        self.micro_recovery(snap, risk_forecast)


class SwarmNodeOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("SwarmNodeOrgan")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(risk_forecast)
        self.state.health = clamp(1.0 - self.state.risk)
        self.state.notes = "Swarm link nominal"
        self.micro_recovery(snap, risk_forecast)


class Back4BloodAnalyzer(BaseOrgan):
    def __init__(self):
        super().__init__("Back4BloodAnalyzer")

    def update(self, snap: TelemetrySnapshot, risk_forecast: float):
        self.state.risk = clamp(0.3 * snap.cpu_load + 0.3 * snap.vram_load + 0.4 * snap.anomaly_score)
        self.state.health = clamp(1.0 - self.state.risk)
        self.state.notes = "Game analysis active"
        self.micro_recovery(snap, risk_forecast)

# ------------------------------------------------------------
# 7. Self-Integrity Organ
# ------------------------------------------------------------

@dataclass
class SelfIntegrityState:
    integrity_score: float = 1.0
    stale_sensors: bool = False
    inconsistent_metrics: bool = False
    missing_organs: bool = False
    prediction_drift: float = 0.0
    model_degradation: float = 0.0
    safe_mode: bool = False


class SelfIntegrityOrgan:
    def __init__(self, organ_registry: Dict[str, BaseOrgan]):
        self.state = SelfIntegrityState()
        self.organ_registry = organ_registry
        self.last_snapshot: Optional[TelemetrySnapshot] = None
        self.last_prediction: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def update(self, snap: TelemetrySnapshot, predicted_risk: float, brain_meta_conf: float):
        self.state.missing_organs = len(self.organ_registry) == 0

        if self.last_snapshot is not None and self.last_timestamp is not None:
            dt = snap.timestamp - self.last_timestamp
            same = (
                abs(snap.cpu_load - self.last_snapshot.cpu_load) < 1e-3 and
                abs(snap.mem_load - self.last_snapshot.mem_load) < 1e-3 and
                abs(snap.disk_load - self.last_snapshot.disk_load) < 1e-3 and
                abs(snap.net_load - self.last_snapshot.net_load) < 1e-3
            )
            self.state.stale_sensors = same and dt > 10.0
        else:
            self.state.stale_sensors = False

        self.state.inconsistent_metrics = any([
            snap.cpu_load < 0.0 or snap.cpu_load > 1.0,
            snap.mem_load < 0.0 or snap.mem_load > 1.0,
            snap.disk_load < 0.0 or snap.disk_load > 1.0,
            snap.net_load < 0.0 or snap.net_load > 1.0,
        ])

        if self.last_prediction is not None:
            drift = abs(predicted_risk - self.last_prediction)
            self.state.prediction_drift = clamp(lerp(self.state.prediction_drift, drift, 0.3))
        else:
            self.state.prediction_drift = 0.0

        self.state.model_degradation = clamp(1.0 - brain_meta_conf)

        penalties = 0.0
        if self.state.stale_sensors:
            penalties += 0.2
        if self.state.inconsistent_metrics:
            penalties += 0.3
        if self.state.missing_organs:
            penalties += 0.3
        penalties += 0.2 * self.state.prediction_drift
        penalties += 0.2 * self.state.model_degradation
        self.state.integrity_score = clamp(1.0 - penalties)
        self.state.safe_mode = self.state.integrity_score < 0.6

        self.last_snapshot = snap
        self.last_prediction = predicted_risk
        self.last_timestamp = snap.timestamp

# ------------------------------------------------------------
# 8. Augmentation Engine
# ------------------------------------------------------------

@dataclass
class Augmentation:
    name: str
    type: str  # "behavior", "predictive", "meta", "organ", "gui"
    enabled: bool = True
    impact_score: float = 0.0
    last_applied: Optional[float] = None


class AugmentationEngine:
    def __init__(self):
        self.augmentations: Dict[str, Augmentation] = {}
        self.register("stability_bias", "behavior")
        self.register("turbulence_shortener", "predictive")
        self.register("meta_state_bias", "meta")
        self.last_mutation: Optional[str] = None

    def register(self, name: str, aug_type: str):
        self.augmentations[name] = Augmentation(name=name, type=aug_type)

    def list_active(self) -> List[Augmentation]:
        return [a for a in self.augmentations.values() if a.enabled]

    def set_enabled(self, name: str, enabled: bool):
        if name in self.augmentations:
            self.augmentations[name].enabled = enabled

    def before_prediction(self, brain: "HybridBrain", snap: TelemetrySnapshot):
        pass

    def after_prediction(self, brain: "HybridBrain", snap: TelemetrySnapshot):
        aug = self.augmentations.get("meta_state_bias")
        if not aug or not aug.enabled:
            return
        total = brain.long_term_success + brain.long_term_fail
        if total < 50:
            return
        ratio = brain.long_term_success / total
        if ratio > 0.7 and brain.current_meta_state_name != "Hyper-Flow":
            brain.current_meta_state_name = "Hyper-Flow"
            aug.last_applied = time.time()
            aug.impact_score = lerp(aug.impact_score, 0.8, 0.3)
            self.last_mutation = "meta_state_bias→Hyper-Flow"

    def before_stance(self, brain: "HybridBrain", snap: TelemetrySnapshot):
        aug = self.augmentations.get("turbulence_shortener")
        if not aug or not aug.enabled:
            return
        variance = brain.var_tracker.mean
        if variance > 0.15:
            m = brain.get_meta_state()
            m.prediction_horizon_short = max(1.0, m.prediction_horizon_short * 0.8)
            m.prediction_horizon_medium = max(3.0, m.prediction_horizon_medium * 0.9)
            aug.last_applied = time.time()
            aug.impact_score = lerp(aug.impact_score, 0.6, 0.2)
            self.last_mutation = "turbulence_shortener"

    def after_stance(self, brain: "HybridBrain", snap: TelemetrySnapshot):
        aug = self.augmentations.get("stability_bias")
        if not aug or not aug.enabled:
            return
        if brain.tri_stance == "Conservative" and brain.pred_medium < 0.4:
            brain.tri_stance = "Balanced"
            aug.last_applied = time.time()
            aug.impact_score = lerp(aug.impact_score, 0.5, 0.2)
            self.last_mutation = "stability_bias→Balanced"

    def before_organs(self, brain: "HybridBrain", snap: TelemetrySnapshot, organs: Dict[str, BaseOrgan]):
        pass

    def after_organs(self, brain: "HybridBrain", snap: TelemetrySnapshot, organs: Dict[str, BaseOrgan]):
        pass

# ------------------------------------------------------------
# 9. HybridBrain + Tri-Stance + Best-Guess
# ------------------------------------------------------------

@dataclass
class MetaState:
    name: str
    prediction_horizon_short: float
    prediction_horizon_medium: float
    prediction_horizon_long: float
    stance_aggressiveness: float
    deep_ram_appetite_bias: float
    thread_expansion_bias: float
    cache_behavior_bias: float


@dataclass
class ReasoningHeatmapTick:
    engine_contrib: Dict[str, float]
    organ_risk: Dict[str, float]
    fingerprints_triggered: List[str]
    meta_rules_fired: List[str]
    tri_stance: str


class HybridBrain:
    def __init__(self, judgment: JudgmentEngine, cortex: SituationalCortex,
                 collective: CollectiveHealth, pattern_memory: PatternMemory,
                 augmentation_engine: AugmentationEngine):
        self.judgment = judgment
        self.cortex = cortex
        self.collective = collective
        self.pattern_memory = pattern_memory
        self.aug = augmentation_engine

        self.ewma = EWMAPredictor(alpha=0.3)
        self.trend = TrendPredictor()
        self.var_tracker = VarianceTracker()
        self.turbulence_tracker = TurbulenceTracker()
        self.movidius = MovidiusInferenceEngine()

        self.mode = "stability"
        self.volatility = 0.0
        self.trust = 0.5
        self.cognitive_load = 0.0

        self.stability_threshold = 0.4
        self.reflex_threshold = 0.7

        self.meta_states: Dict[str, MetaState] = {
            "Hyper-Flow": MetaState("Hyper-Flow", 1.0, 5.0, 30.0, 0.9, 0.7, 0.9, 0.8),
            "Deep-Dream": MetaState("Deep-Dream", 5.0, 30.0, 120.0, 0.4, 0.9, 0.3, 0.9),
            "Sentinel": MetaState("Sentinel", 1.0, 10.0, 60.0, 0.3, 0.4, 0.3, 0.5),
            "Recovery-Flow": MetaState("Recovery-Flow", 1.0, 5.0, 60.0, 0.2, 0.3, 0.2, 0.4),
        }
        self.current_meta_state_name = "Sentinel"
        self.meta_state_last_change = time.time()

        self.baseline_risk = 0.2
        self.regime = "stable"

        self.pred_short = 0.0
        self.pred_medium = 0.0
        self.pred_long = 0.0
        self.best_guess = 0.0
        self.meta_confidence = 0.5

        self.long_term_success = 0
        self.long_term_fail = 0
        self.last_calibration_ts = time.time()

        self.tri_stance = "Balanced"
        self.last_heatmap_tick: Optional[ReasoningHeatmapTick] = None

    def get_meta_state(self) -> MetaState:
        return self.meta_states[self.current_meta_state_name]

    def decide_tri_stance(self, snap: TelemetrySnapshot, pred_short: float,
                          pred_med: float, pred_long: float,
                          organs: Dict[str, BaseOrgan]) -> str:
        mem = snap.mem_load
        temp = snap.temp
        deep_ram = organs.get("DeepRamOrgan", None)
        deep_ratio = deep_ram.state.risk if deep_ram else mem
        risk_rising = pred_med > self.baseline_risk and pred_med > self.stability_threshold
        safe_headroom = pred_med < 0.4 and pred_long < 0.5
        reinforcement_conf = self.judgment.judgment_confidence()
        beast_helped = reinforcement_conf > 0.6 and self.long_term_success > self.long_term_fail
        pressured = any(o.state.risk > 0.8 for o in organs.values())

        current = self.tri_stance
        new_stance = current

        if (mem > 0.9 or temp > 0.85 or risk_rising or pressured or deep_ratio > 0.85):
            new_stance = "Conservative"
        elif (safe_headroom and temp < 0.6 and deep_ratio < 0.7 and beast_helped):
            new_stance = "Beast"
        elif (pred_med < 0.5 and not pressured and abs(deep_ratio - 0.75) < 0.1):
            new_stance = "Balanced"

        self.tri_stance = new_stance
        return new_stance

    def apply_tri_stance_scaling(self, stance: str, organs: Dict[str, BaseOrgan]):
        for organ in organs.values():
            if stance == "Conservative":
                organ.state.appetite = clamp(organ.state.appetite - 0.05)
                organ.state.ingestion_rate = clamp(organ.state.ingestion_rate - 0.05)
            elif stance == "Balanced":
                organ.state.appetite = clamp(lerp(organ.state.appetite, 0.5, 0.1))
                organ.state.ingestion_rate = clamp(lerp(organ.state.ingestion_rate, 0.5, 0.1))
            elif stance == "Beast":
                organ.state.appetite = clamp(organ.state.appetite + 0.05)
                organ.state.ingestion_rate = clamp(organ.state.ingestion_rate + 0.05)

    def _detect_regime(self, risk_now: float, risk_ewma: float, variance: float, turbulence: float) -> str:
        delta = risk_now - risk_ewma
        if variance < 0.05 and turbulence < 0.05:
            return "stable"
        if variance > 0.2 or turbulence > 0.2:
            if delta > 0.05:
                return "rising"
            elif delta < -0.05:
                return "cooling"
            else:
                return "chaotic"
        if delta > 0.05:
            return "rising"
        if delta < -0.05:
            return "cooling"
        return "stable"

    def _regime_formula(self, regime: str, base_risk: float, mov_risk: float,
                        variance: float, turbulence: float) -> float:
        if regime == "stable":
            w_model, w_ewma, w_var = 0.5, 0.4, 0.1
        elif regime == "chaotic":
            w_model, w_ewma, w_var = 0.3, 0.3, 0.4
        elif regime == "rising":
            w_model, w_ewma, w_var = 0.6, 0.2, 0.2
        else:
            w_model, w_ewma, w_var = 0.4, 0.4, 0.2
        risk = clamp(w_model * mov_risk + w_ewma * base_risk + w_var * (variance + turbulence) / 2.0)
        return risk

    def _meta_conf_fusion(self, variance: float, trend_stability: float,
                          sensor_noise: float, turbulence: float,
                          reinforcement_conf: float, model_conf: float) -> float:
        stability_score = clamp(
            (trend_stability + (1.0 - variance) + (1.0 - turbulence) + (1.0 - sensor_noise)) / 4.0
        )
        return clamp(0.4 * stability_score + 0.3 * reinforcement_conf + 0.3 * model_conf)

    def _multi_engine_vote(self, engines: Dict[str, Tuple[float, float]]) -> float:
        num = 0.0
        den = 0.0
        for v, w in engines.values():
            num += v * w
            den += w
        if den <= 0:
            return 0.0
        return clamp(num / den)

    def _auto_tune(self):
        total = self.long_term_success + self.long_term_fail
        if total < 50:
            return
        ratio = self.long_term_success / total
        if ratio > 0.7:
            self.stability_threshold = clamp(self.stability_threshold + 0.01, 0.1, 0.9)
            self.reflex_threshold = clamp(self.reflex_threshold + 0.005, 0.2, 0.99)
            self.baseline_risk = clamp(self.baseline_risk - 0.01, 0.05, 0.5)
        elif ratio < 0.4:
            self.stability_threshold = clamp(self.stability_threshold - 0.01, 0.1, 0.9)
            self.reflex_threshold = clamp(self.reflex_threshold - 0.005, 0.2, 0.99)
            self.baseline_risk = clamp(self.baseline_risk + 0.01, 0.05, 0.5)

    def _auto_calibrate_if_due(self):
        now = time.time()
        if now - self.last_calibration_ts < 60.0:
            return
        self.last_calibration_ts = now
        total = self.long_term_success + self.long_term_fail
        if total == 0:
            return
        ratio = self.long_term_success / total
        self.baseline_risk = clamp(0.2 + 0.2 * (0.5 - ratio), 0.05, 0.5)
        self.stability_threshold = clamp(0.3 + 0.2 * (0.5 - ratio), 0.1, 0.8)
        self.reflex_threshold = clamp(0.6 + 0.2 * (0.5 - ratio), 0.4, 0.95)

    def update(self, snap: TelemetrySnapshot, organs: Dict[str, BaseOrgan]) -> Tuple[float, float, float, float]:
        self.aug.before_prediction(self, snap)

        meta = self.get_meta_state()
        load_now = (snap.cpu_load + snap.mem_load + snap.disk_load + snap.net_load) / 4.0

        deep_ram = organs.get("DeepRamOrgan")
        mem_component = deep_ram.state.risk if deep_ram else snap.mem_load
        risk_now = clamp(0.4 * load_now + 0.3 * mem_component + 0.3 * snap.anomaly_score)

        risk_ewma = self.ewma.update(risk_now)
        variance = self.var_tracker.update(risk_now)
        turbulence = self.turbulence_tracker.update(risk_now)
        trend_pred = self.trend.update(snap.timestamp, risk_now)
        trend_stability = self.trend.trend_stability()
        sensor_noise = clamp(variance + 0.5 * turbulence)

        mov_short, mc_short = self.movidius.predict(snap, meta.prediction_horizon_short)
        mov_med, mc_med = self.movidius.predict(snap, meta.prediction_horizon_medium)
        mov_long, mc_long = self.movidius.predict(snap, meta.prediction_horizon_long)
        model_conf = (mc_short + mc_med + mc_long) / 3.0

        self.regime = self._detect_regime(risk_now, risk_ewma, variance, turbulence)
        reinforcement_conf = self.judgment.judgment_confidence()
        meta_conf = self._meta_conf_fusion(
            variance, trend_stability, sensor_noise, turbulence, reinforcement_conf, model_conf
        )

        engines_short = {
            "ewma": (risk_ewma, 0.3),
            "inst": (risk_now, 0.2),
            "trend": (trend_pred, 0.2),
            "mov": (mov_short, 0.2),
            "baseline": (self.baseline_risk, 0.1),
        }
        engines_med = {
            "ewma": (risk_ewma, 0.25),
            "inst": (risk_now, 0.1),
            "trend": (trend_pred, 0.2),
            "mov": (mov_med, 0.3),
            "baseline": (self.baseline_risk, 0.15),
        }
        engines_long = {
            "ewma": (risk_ewma, 0.2),
            "trend": (trend_pred, 0.2),
            "mov": (mov_long, 0.4),
            "baseline": (self.baseline_risk, 0.2),
        }

        pred_short_base = self._multi_engine_vote(engines_short)
        pred_med_base = self._multi_engine_vote(engines_med)
        pred_long_base = self._multi_engine_vote(engines_long)

        pred_short = self._regime_formula(self.regime, pred_short_base, mov_short, variance, turbulence)
        pred_med = self._regime_formula(self.regime, pred_med_base, mov_med, variance, turbulence)
        pred_long = self._regime_formula(self.regime, pred_long_base, mov_long, variance, turbulence)

        # Best-guess: fused medium-horizon vote
        self.best_guess = pred_med_base

        self.aug.after_prediction(self, snap)

        if pred_short > self.reflex_threshold:
            self.baseline_risk = lerp(self.baseline_risk, pred_short, 0.1)
        if pred_long > 0.6:
            self.stability_threshold = clamp(self.stability_threshold - 0.01, 0.1, 0.9)
        else:
            self.stability_threshold = clamp(self.stability_threshold + 0.005, 0.1, 0.9)

        risk_blend = pred_med
        if risk_blend < self.stability_threshold:
            self.mode = "stability"
        elif risk_blend < self.reflex_threshold:
            self.mode = "exploration"
        else:
            self.mode = "reflex"

        self.aug.before_stance(self, snap)

        fingerprints_triggered: List[str] = []
        if risk_now > 0.85:
            self.pattern_memory.record(snap, "overload")
            fingerprints_triggered.append("overload")
            self.long_term_fail += 1
        elif risk_now < 0.3:
            self.pattern_memory.record(snap, "stability")
            fingerprints_triggered.append("stability")
            self.long_term_success += 1
        elif risk_now > 0.6 and snap.vram_load > 0.6 and snap.cpu_load > 0.6:
            self.pattern_memory.record(snap, "beast_win")
            fingerprints_triggered.append("beast_win")
            self.long_term_success += 1

        summary = self.pattern_memory.summarize()
        overloads = summary.get("overload", 0)
        stables = summary.get("stability", 0)
        meta_rules_fired: List[str] = []
        if stables > overloads + 5:
            self.meta_states[self.current_meta_state_name].stance_aggressiveness = clamp(
                self.meta_states[self.current_meta_state_name].stance_aggressiveness + 0.01
            )
            meta_rules_fired.append("agg++")
        elif overloads > stables + 5:
            self.meta_states[self.current_meta_state_name].stance_aggressiveness = clamp(
                self.meta_states[self.current_meta_state_name].stance_aggressiveness - 0.01
            )
            meta_rules_fired.append("agg--")

        tri = self.decide_tri_stance(snap, pred_short, pred_med, pred_long, organs)
        self.apply_tri_stance_scaling(tri, organs)

        self.aug.after_stance(self, snap)

        self._auto_tune()
        self._auto_calibrate_if_due()

        self.cognitive_load = clamp(0.3 * variance + 0.4 * turbulence + 0.3 * (1.0 - meta_conf))
        self.volatility = clamp(0.5 * variance + 0.5 * turbulence)
        self.trust = clamp(lerp(self.trust, meta_conf, 0.2))
        self.meta_confidence = meta_conf
        self.pred_short, self.pred_medium, self.pred_long = pred_short, pred_med, pred_long

        organ_risk = {name: organ.state.risk for name, organ in organs.items()}
        engine_contrib = {
            "ewma": engines_med["ewma"][1],
            "inst": engines_med["inst"][1],
            "trend": engines_med["trend"][1],
            "mov": engines_med["mov"][1],
            "baseline": engines_med["baseline"][1],
        }
        self.last_heatmap_tick = ReasoningHeatmapTick(
            engine_contrib=engine_contrib,
            organ_risk=organ_risk,
            fingerprints_triggered=fingerprints_triggered,
            meta_rules_fired=meta_rules_fired,
            tri_stance=tri,
        )

        return pred_short, pred_med, pred_long, meta_conf

# ============================
# PART 2 of 2
# GUI + Nerve Center + Altered States + Reboot Memory + main()
# ============================

# (Continue in the same file, directly after PART 1)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ------------------------------------------------------------
# 10. Telemetry acquisition (already used by GUI)
# ------------------------------------------------------------

def get_real_snapshot(t: float, last_net: Dict[str, float], last_disk: Dict[str, float]) -> TelemetrySnapshot:
    psutil_mod = DEPS.get("psutil")
    pynvml_mod = DEPS.get("pynvml")
    if psutil_mod is None:
        return generate_fake_snapshot(t)
    import psutil as ps

    cpu = clamp(ps.cpu_percent(interval=None) / 100.0)
    mem = ps.virtual_memory()
    mem_load = clamp(mem.percent / 100.0)

    disk_io = ps.disk_io_counters()
    net_io = ps.net_io_counters()

    if not last_disk:
        last_disk["read"] = disk_io.read_bytes
        last_disk["write"] = disk_io.write_bytes
    if not last_net:
        last_net["bytes_sent"] = net_io.bytes_sent
        last_net["bytes_recv"] = net_io.bytes_recv

    disk_bytes = (disk_io.read_bytes - last_disk["read"]) + (disk_io.write_bytes - last_disk["write"])
    net_bytes = (net_io.bytes_sent - last_net["bytes_sent"]) + (net_io.bytes_recv - last_net["bytes_recv"])

    last_disk["read"] = disk_io.read_bytes
    last_disk["write"] = disk_io.write_bytes
    last_net["bytes_sent"] = net_io.bytes_sent
    last_net["bytes_recv"] = net_io.bytes_recv

    disk_load = clamp(disk_bytes / (1024 * 1024 * 50.0))
    net_load = clamp(net_bytes / (1024 * 1024 * 10.0))

    temp_norm = 0.4
    try:
        temps = ps.sensors_temperatures()
        if temps:
            all_t = []
            for arr in temps.values():
                for e in arr:
                    if e.current is not None:
                        all_t.append(e.current)
            if all_t:
                t_avg = sum(all_t) / len(all_t)
                temp_norm = clamp((t_avg - 40.0) / 40.0)
    except Exception:
        pass

    vram_load = 0.4
    if pynvml_mod is not None:
        try:
            import pynvml as nv
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            mem_info = nv.nvmlDeviceGetMemoryInfo(h)
            vram_load = clamp(mem_info.used / float(mem_info.total))
            nv.nvmlShutdown()
        except Exception:
            pass

    anomaly = clamp(0.1 * random.random() + 0.4 * max(0.0, cpu - 0.8))
    turbulence = clamp(abs(random.gauss(0.2, 0.05)))

    return TelemetrySnapshot(
        timestamp=t,
        cpu_load=cpu,
        mem_load=mem_load,
        disk_load=disk_load,
        net_load=net_load,
        temp=temp_norm,
        vram_load=vram_load,
        anomaly_score=anomaly,
        turbulence=turbulence,
    )


def generate_fake_snapshot(t: float) -> TelemetrySnapshot:
    base = 0.3 + 0.2 * math.sin(t / 10.0)
    spike = 0.4 if random.random() < 0.05 else 0.0
    cpu = clamp(base + spike + random.uniform(-0.05, 0.05))
    mem = clamp(0.4 + 0.2 * math.sin(t / 15.0) + random.uniform(-0.05, 0.05))
    disk = clamp(0.2 + 0.3 * random.random())
    net = clamp(0.2 + 0.2 * math.sin(t / 5.0) + random.uniform(-0.05, 0.05))
    temp = clamp(0.3 + 0.4 * cpu + 0.1 * random.random())
    vram = clamp(0.3 + 0.4 * random.random())
    anomaly = clamp(0.1 + 0.5 * random.random() if spike > 0.0 else 0.1 * random.random())
    turbulence = clamp(abs(random.gauss(0.2, 0.1)))
    return TelemetrySnapshot(
        timestamp=t,
        cpu_load=cpu,
        mem_load=mem,
        disk_load=disk,
        net_load=net,
        temp=temp,
        vram_load=vram,
        anomaly_score=anomaly,
        turbulence=turbulence,
    )

# ------------------------------------------------------------
# 11. GUI – System Nerve Center + Altered States + Reboot Memory
# ------------------------------------------------------------

class NerveCenterGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Autonomous Cipher Engine – Nerve Center (Augmented + Reboot Memory)")

        self.judgment = JudgmentEngine()
        self.cortex = SituationalCortex()
        self.collective = CollectiveHealth()
        self.pattern_memory = PatternMemory()
        self.augmentation_engine = AugmentationEngine()
        self.brain = HybridBrain(self.judgment, self.cortex, self.collective,
                                 self.pattern_memory, self.augmentation_engine)

        self.organs: Dict[str, BaseOrgan] = {
            "ThermalOrgan": ThermalOrgan(),
            "DiskOrgan": DiskOrgan(),
            "NetworkWatcherOrgan": NetworkWatcherOrgan(),
            "DeepRamOrgan": DeepRamOrgan(),
            "VRAMOrgan": VRAMOrgan(),
            "BackupEngineOrgan": BackupEngineOrgan(),
            "GPUCacheOrgan": GPUCacheOrgan(),
            "AICoachOrgan": AICoachOrgan(),
            "SwarmNodeOrgan": SwarmNodeOrgan(),
            "Back4BloodAnalyzer": Back4BloodAnalyzer(),
        }

        self.self_integrity = SelfIntegrityOrgan(self.organs)
        self.reasoning_tail: List[str] = []
        self.last_snapshot: Optional[TelemetrySnapshot] = None
        self.t0 = time.time()
        self.tick = 0
        self.last_net: Dict[str, float] = {}
        self.last_disk: Dict[str, float] = {}

        self.historical_for_training: List[Tuple[TelemetrySnapshot, float]] = []

        # For micro-chart: store last N predictions (short, med, long, best)
        self.pred_history: List[Tuple[float, float, float, float]] = []
        self.chart_max_points = 80

        self._build_layout()

        # Auto-load reboot memory if enabled
        self.root.after(200, self._autoload_reboot_memory)

        self._schedule_update()

    # ---------------- Layout ----------------

    def _build_layout(self):
        root = self.root
        root.geometry("1400x850")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Nerve Center
        tab_main = ttk.Frame(notebook)
        notebook.add(tab_main, text="Nerve Center")

        left = ttk.Frame(tab_main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(tab_main, width=380)
        right.pack(side="right", fill="y")

        # HybridBrain + Tri-Stance + Micro-Chart
        self.frame_brain = ttk.LabelFrame(left, text="Hybrid Brain & Tri-Stance")
        self.frame_brain.pack(fill="x", padx=5, pady=5)

        self.lbl_brain = tk.Label(self.frame_brain, justify="left", anchor="w", font=("Consolas", 9))
        self.lbl_brain.pack(fill="x")

        self.canvas_chart = tk.Canvas(self.frame_brain, height=90, bg="#111111",
                                      highlightthickness=1, highlightbackground="#333333")
        self.canvas_chart.pack(fill="x", padx=2, pady=(4, 2))

        # Cortex / Collective / Integrity
        self.frame_cortex = ttk.LabelFrame(left, text="Cortex / Collective / Integrity")
        self.frame_cortex.pack(fill="x", padx=5, pady=5)

        self.lbl_cortex = tk.Label(self.frame_cortex, justify="left", anchor="w", font=("Consolas", 9))
        self.lbl_cortex.pack(fill="x")

        # Organs table
        self.frame_organs = ttk.LabelFrame(left, text="Organs")
        self.frame_organs.pack(fill="both", expand=True, padx=5, pady=5)

        columns = ("health", "risk", "appetite", "ingestion", "notes")
        self.tree_organs = ttk.Treeview(self.frame_organs, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree_organs.heading(col, text=col.capitalize())
            self.tree_organs.column(col, width=115, anchor="center")
        self.tree_organs.pack(fill="both", expand=True)
        for name in self.organs.keys():
            self.tree_organs.insert("", "end", iid=name, values=("", "", "", "", ""))

        # Right side: Reasoning tail + Heatmap + Augmentations + Commands
        self.frame_tail = ttk.LabelFrame(right, text="Reasoning Tail")
        self.frame_tail.pack(fill="both", expand=True, padx=5, pady=5)

        self.txt_tail = tk.Text(self.frame_tail, height=15, width=45, font=("Consolas", 8))
        self.txt_tail.pack(fill="both", expand=True)

        self.frame_heatmap = ttk.LabelFrame(right, text="Reasoning Heatmap")
        self.frame_heatmap.pack(fill="both", expand=True, padx=5, pady=5)

        self.txt_heatmap = tk.Text(self.frame_heatmap, height=12, width=45, font=("Consolas", 8))
        self.txt_heatmap.pack(fill="both", expand=True)

        self.frame_aug = ttk.LabelFrame(right, text="Augmentations")
        self.frame_aug.pack(fill="both", expand=True, padx=5, pady=5)

        self.txt_aug = tk.Text(self.frame_aug, height=8, width=45, font=("Consolas", 8))
        self.txt_aug.pack(fill="both", expand=True)

        self.frame_cmd = ttk.LabelFrame(right, text="Commands")
        self.frame_cmd.pack(fill="x", padx=5, pady=5)

        btns = [
            ("Stabilize", self.cmd_stabilize),
            ("High-Alert", self.cmd_high_alert),
            ("Learn", self.cmd_learn),
            ("Optimize", self.cmd_optimize),
            ("Purge Mem", self.cmd_purge),
            ("Rebuild Model", self.cmd_rebuild),
        ]
        for text, cmd in btns:
            b = ttk.Button(self.frame_cmd, text=text, command=cmd)
            b.pack(side="left", padx=2, pady=2)

        btns2 = [
            ("Reset Cortex", self.cmd_reset_cortex),
            ("Snapshot", self.cmd_snapshot),
            ("Meta-State", self.cmd_meta_cycle),
            ("Train ONNX", self.cmd_train_onnx),
        ]
        for text, cmd in btns2:
            b = ttk.Button(self.frame_cmd, text=text, command=cmd)
            b.pack(side="left", padx=2, pady=2)

        btns3 = [
            ("Save Memory (Local)", self.cmd_save_memory_local),
            ("Save Memory (SMB)", self.cmd_save_memory_smb),
        ]
        for text, cmd in btns3:
            b = ttk.Button(self.frame_cmd, text=text, command=cmd)
            b.pack(side="left", padx=2, pady=2)

        # Tab 2: Altered States
        tab_states = ttk.Frame(notebook)
        notebook.add(tab_states, text="Altered States")

        self.lbl_states = tk.Label(tab_states, justify="left", anchor="nw", font=("Consolas", 9))
        self.lbl_states.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 3: Reboot Memory
        tab_reboot = ttk.Frame(notebook)
        notebook.add(tab_reboot, text="Reboot Memory")

        frame_reboot = ttk.LabelFrame(tab_reboot, text="Reboot Memory Persistence")
        frame_reboot.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame_reboot, text="SMB / UNC Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame_reboot, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame_reboot, text="Pick SMB Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame_reboot, text="Test SMB Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame_reboot, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame_reboot,
            text="Load memory from SMB on startup",
            variable=self.var_reboot_autoload
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame_reboot, text="Status: Ready", anchor="w", fg="#00cc66")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    # ---------------- Utility ----------------

    def log_reason(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.reasoning_tail.append(line)
        if len(self.reasoning_tail) > 80:
            self.reasoning_tail.pop(0)

    # ---------------- Commands ----------------

    def cmd_stabilize(self):
        self.cortex.state.mission = "STABILITY"
        self.cortex.state.mission_override = "STABILITY"
        self.brain.mode = "stability"
        self.log_reason("Command: Stabilize system.")

    def cmd_high_alert(self):
        self.cortex.state.mission = "PROTECT"
        self.cortex.state.mission_override = "PROTECT"
        self.brain.mode = "reflex"
        self.log_reason("Command: High-alert mode.")

    def cmd_learn(self):
        self.cortex.state.mission = "LEARN"
        self.cortex.state.prioritize_learning_windows = True
        self.log_reason("Command: Begin learning cycle.")

    def cmd_optimize(self):
        self.cortex.state.mission = "OPTIMIZE"
        self.log_reason("Command: Optimize performance.")

    def cmd_purge(self):
        self.pattern_memory.records.clear()
        self.log_reason("Command: Purged pattern memory.")

    def cmd_rebuild(self):
        self.brain.ewma = EWMAPredictor()
        self.brain.var_tracker = VarianceTracker()
        self.brain.turbulence_tracker = TurbulenceTracker()
        self.brain.baseline_risk = 0.2
        self.log_reason("Command: Rebuilt predictive core.")

    def cmd_reset_cortex(self):
        self.cortex.state = SituationalCortexState()
        self.log_reason("Command: Reset cortex.")

    def cmd_snapshot(self):
        self.log_reason("Command: Snapshot brain (placeholder).")

    def cmd_meta_cycle(self):
        names = list(self.brain.meta_states.keys())
        idx = names.index(self.brain.current_meta_state_name)
        new_name = names[(idx + 1) % len(names)]
        self.brain.current_meta_state_name = new_name
        self.log_reason(f"Command: Switched meta-state to {new_name}.")

    def cmd_train_onnx(self):
        if not self.historical_for_training:
            messagebox.showinfo("Train ONNX", "No historical samples collected yet.")
            return
        self.brain.movidius.train(self.historical_for_training)
        n = self.brain.movidius.training_stats["samples"]
        self.log_reason(f"Command: Train ONNX stub called. Samples total={n}.")
        messagebox.showinfo("Train ONNX", f"Training stub updated. Total samples: {n}")

    # --- Memory save (local / generic SMB path picker) ---

    def cmd_save_memory_local(self):
        path = filedialog.asksaveasfilename(
            title="Save Memory Snapshot (JSON+BIN)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        base = path.rsplit(".", 1)[0]
        json_path = base + ".json"
        bin_path = base + ".bin"
        self._save_memory_to_paths(json_path, bin_path)
        self.log_reason(f"Command: Saved memory snapshot to {json_path} and {bin_path}.")
        messagebox.showinfo("Save Memory", f"Saved:\n{json_path}\n{bin_path}")

    def cmd_save_memory_smb(self):
        path = filedialog.asksaveasfilename(
            title="Save Memory Snapshot to SMB/UNC (JSON+BIN)",
            defaultextension=".json",
            initialfile="brain_snapshot.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        base = path.rsplit(".", 1)[0]
        json_path = base + ".json"
        bin_path = base + ".bin"
        self._save_memory_to_paths(json_path, bin_path)
        self.log_reason(f"Command: Saved memory snapshot (SMB/local) to {json_path} and {bin_path}.")
        messagebox.showinfo("Save Memory (SMB)", f"Saved:\n{json_path}\n{bin_path}")

    # --- Reboot Memory tab commands ---

    def cmd_pick_reboot_path(self):
        path = filedialog.asksaveasfilename(
            title="Select SMB/UNC Path for Reboot Memory",
            defaultextension=".json",
            initialfile="brain_reboot.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)
            self.lbl_reboot_status.config(text="Status: Path selected", fg="#00cc66")

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self.lbl_reboot_status.config(text="Status: No path entered", fg="#cc0000")
            return
        test_path = path.rsplit(".", 1)[0] + "_test.txt"
        try:
            with open(test_path, "w") as f:
                f.write("test")
            self.lbl_reboot_status.config(text="Status: SMB path OK", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Error writing ({e})", fg="#cc0000")

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self.lbl_reboot_status.config(text="Status: No path entered", fg="#cc0000")
            return
        base = path.rsplit(".", 1)[0]
        json_path = base + ".json"
        bin_path = base + ".bin"
        try:
            self._save_memory_to_paths(json_path, bin_path)
            self.lbl_reboot_status.config(
                text=f"Status: Saved reboot memory\n{json_path}\n{bin_path}",
                fg="#00cc66"
            )
            self.log_reason(f"Reboot memory saved to {json_path} and {bin_path}.")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Save failed ({e})", fg="#cc0000")

    # --- Memory save implementation (JSON + BIN snapshot) ---

    def _save_memory_to_paths(self, json_path: str, bin_path: str):
        brain = self.brain
        cortex = self.cortex.state
        coll = self.collective.state
        si = self.self_integrity.state

        snapshot = {
            "timestamp": time.time(),
            "hybrid_brain": {
                "mode": brain.mode,
                "volatility": brain.volatility,
                "trust": brain.trust,
                "cognitive_load": brain.cognitive_load,
                "pred_short": brain.pred_short,
                "pred_medium": brain.pred_medium,
                "pred_long": brain.pred_long,
                "best_guess": brain.best_guess,
                "meta_confidence": brain.meta_confidence,
                "stability_threshold": brain.stability_threshold,
                "reflex_threshold": brain.reflex_threshold,
                "baseline_risk": brain.baseline_risk,
                "tri_stance": brain.tri_stance,
                "regime": brain.regime,
                "current_meta_state_name": brain.current_meta_state_name,
                "meta_states": {
                    name: {
                        "prediction_horizon_short": m.prediction_horizon_short,
                        "prediction_horizon_medium": m.prediction_horizon_medium,
                        "prediction_horizon_long": m.prediction_horizon_long,
                        "stance_aggressiveness": m.stance_aggressiveness,
                        "deep_ram_appetite_bias": m.deep_ram_appetite_bias,
                        "thread_expansion_bias": m.thread_expansion_bias,
                        "cache_behavior_bias": m.cache_behavior_bias,
                    }
                    for name, m in brain.meta_states.items()
                },
                "reinforcement": {
                    "good": brain.judgment.stats.good,
                    "bad": brain.judgment.stats.bad,
                    "learning_rate": brain.judgment.learning_rate,
                },
                "long_term": {
                    "success": brain.long_term_success,
                    "fail": brain.long_term_fail,
                },
            },
            "cortex": cortex.__dict__.copy(),
            "collective": coll.__dict__.copy(),
            "self_integrity": si.__dict__.copy(),
            "organs": {
                name: {
                    "health": organ.state.health,
                    "risk": organ.state.risk,
                    "appetite": organ.state.appetite,
                    "ingestion_rate": organ.state.ingestion_rate,
                    "notes": organ.state.notes,
                }
                for name, organ in self.organs.items()
            },
            "pattern_memory_summary": self.pattern_memory.summarize(),
            "augmentations": {
                name: {
                    "type": aug.type,
                    "enabled": aug.enabled,
                    "impact_score": aug.impact_score,
                    "last_applied": aug.last_applied,
                }
                for name, aug in self.augmentation_engine.augmentations.items()
            },
            "last_mutation": self.augmentation_engine.last_mutation,
        }

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        # Binary pickle
        with open(bin_path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Reboot Memory autoload + restore ---

    def _autoload_reboot_memory(self):
        if not hasattr(self, "var_reboot_autoload") or not self.var_reboot_autoload.get():
            return

        path = self.entry_reboot_path.get().strip()
        if not path:
            return

        json_path = path.rsplit(".", 1)[0] + ".json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._restore_memory_from_snapshot(data)
            self.lbl_reboot_status.config(text="Status: Loaded reboot memory", fg="#00cc66")
            self.log_reason("Auto-loaded reboot memory from SMB/local.")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Auto-load failed ({e})", fg="#cc0000")

    def _restore_memory_from_snapshot(self, snap):
        brain = snap["hybrid_brain"]

        self.brain.mode = brain["mode"]
        self.brain.volatility = brain["volatility"]
        self.brain.trust = brain["trust"]
        self.brain.cognitive_load = brain["cognitive_load"]
        self.brain.pred_short = brain["pred_short"]
        self.brain.pred_medium = brain["pred_medium"]
        self.brain.pred_long = brain["pred_long"]
        self.brain.best_guess = brain["best_guess"]
        self.brain.meta_confidence = brain["meta_confidence"]
        self.brain.stability_threshold = brain["stability_threshold"]
        self.brain.reflex_threshold = brain["reflex_threshold"]
        self.brain.baseline_risk = brain["baseline_risk"]
        self.brain.tri_stance = brain["tri_stance"]
        self.brain.regime = brain["regime"]
        self.brain.current_meta_state_name = brain["current_meta_state_name"]

        self.judgment.stats.good = brain["reinforcement"]["good"]
        self.judgment.stats.bad = brain["reinforcement"]["bad"]
        self.judgment.learning_rate = brain["reinforcement"]["learning_rate"]

        self.brain.long_term_success = brain["long_term"]["success"]
        self.brain.long_term_fail = brain["long_term"]["fail"]

        for name, state in snap["organs"].items():
            if name in self.organs:
                o = self.organs[name]
                o.state.health = state["health"]
                o.state.risk = state["risk"]
                o.state.appetite = state["appetite"]
                o.state.ingestion_rate = state["ingestion_rate"]
                o.state.notes = state["notes"]

        self.cortex.state.__dict__.update(snap["cortex"])
        self.collective.state.__dict__.update(snap["collective"])
        self.self_integrity.state.__dict__.update(snap["self_integrity"])

        self.log_reason("Reboot memory restored from snapshot.")

    # ---------------- Main loop ----------------

    def _schedule_update(self):
        self.root.after(500, self._update_loop)

    def _update_loop(self):
        now = time.time() - self.t0
        if DEPS.has("psutil"):
            snap = get_real_snapshot(now, self.last_net, self.last_disk)
        else:
            snap = generate_fake_snapshot(now)

        self.last_snapshot = snap

        pred_s, pred_m, pred_l, meta_conf = self.brain.update(snap, self.organs)
        self.cortex.update_from_snapshot(snap)
        self.collective.update_from_local(pred_m)

        for organ in self.organs.values():
            self.augmentation_engine.before_organs(self.brain, snap, self.organs)
            organ.update(snap, pred_m)
            self.augmentation_engine.after_organs(self.brain, snap, self.organs)

        self.self_integrity.update(snap, pred_m, meta_conf)

        if pred_m < 0.3:
            self.judgment.reinforce_good()
        elif pred_m > 0.8:
            self.judgment.reinforce_bad()

        self.historical_for_training.append((snap, pred_m))
        if len(self.historical_for_training) > 2000:
            self.historical_for_training.pop(0)

        self.pred_history.append((pred_s, pred_m, pred_l, self.brain.best_guess))
        if len(self.pred_history) > self.chart_max_points:
            self.pred_history.pop(0)

        self.log_reason(
            f"Tick {self.tick} | stance={self.brain.tri_stance} mode={self.brain.mode} "
            f"med={pred_m:.2f} best={self.brain.best_guess:.2f} meta={meta_conf:.2f}"
        )

        self._refresh_gui()
        self.tick += 1
        self._schedule_update()

    # ---------------- GUI refresh ----------------

    def _refresh_gui(self):
        snap = self.last_snapshot
        ctx = self.cortex.state
        coll = self.collective.state
        si = self.self_integrity.state
        meta = self.brain.get_meta_state()

        # Brain
        text_brain = []
        text_brain.append(f"Tri-Stance: {self.brain.tri_stance}")
        text_brain.append(f"Meta-State: {self.brain.current_meta_state_name}  Regime: {self.brain.regime}")
        text_brain.append(f"Mode: {self.brain.mode}")
        text_brain.append(
            f"Volatility={self.brain.volatility:.2f}  Trust={self.brain.trust:.2f}  CogLoad={self.brain.cognitive_load:.2f}"
        )
        text_brain.append(
            f"Pred S/M/L: {self.brain.pred_short:.2f}/{self.brain.pred_medium:.2f}/"
            f"{self.brain.pred_long:.2f}  Best={self.brain.best_guess:.2f}  MetaConf={self.brain.meta_confidence:.2f}"
        )
        text_brain.append(
            f"Stance thresholds: stability<{self.brain.stability_threshold:.2f}, "
            f"reflex>{self.brain.reflex_threshold:.2f}"
        )
        text_brain.append(
            f"Meta biases: H {meta.prediction_horizon_short}/{meta.prediction_horizon_medium}/"
            f"{meta.prediction_horizon_long} Agg {meta.stance_aggressiveness:.2f} "
            f"RAM {meta.deep_ram_appetite_bias:.2f} Thr {meta.thread_expansion_bias:.2f} "
            f"Cache {meta.cache_behavior_bias:.2f}"
        )
        self.lbl_brain.config(text="\n".join(text_brain))

        # Micro-chart
        self._draw_micro_chart()

        # Cortex / collective / integrity
        text_ctx = []
        text_ctx.append(f"Mission: {ctx.effective_mission()} (base={ctx.mission}, override={ctx.mission_override})")
        text_ctx.append(f"Env={ctx.env}  Opp={ctx.opportunity_score:.2f}  Risk={ctx.risk_score:.2f}")
        text_ctx.append(f"Anticipation: {ctx.anticipation}")
        text_ctx.append(
            f"Collective risk={coll.collective_risk_score:.2f}  HiveMode={coll.hive_sync_mode} "
            f"Consensus={coll.consensus_weighting:.2f}"
        )
        text_ctx.append(
            f"Integrity={si.integrity_score:.2f}  SafeMode={si.safe_mode}  "
            f"Drift={si.prediction_drift:.2f}  ModelDegr={si.model_degradation:.2f}"
        )
        self.lbl_cortex.config(text="\n".join(text_ctx))

        # Organs
        for name, organ in self.organs.items():
            vals = (
                f"{organ.state.health:.2f}",
                f"{organ.state.risk:.2f}",
                f"{organ.state.appetite:.2f}",
                f"{organ.state.ingestion_rate:.2f}",
                organ.state.notes,
            )
            if self.tree_organs.exists(name):
                self.tree_organs.item(name, values=vals)

        # Reasoning tail
        self.txt_tail.delete("1.0", tk.END)
        for line in self.reasoning_tail[-25:]:
            self.txt_tail.insert(tk.END, line + "\n")

        # Heatmap
        self.txt_heatmap.delete("1.0", tk.END)
        ht = self.brain.last_heatmap_tick
        if ht is None:
            self.txt_heatmap.insert(tk.END, "No heatmap data yet.\n")
        else:
            self.txt_heatmap.insert(tk.END, f"Tri-Stance: {ht.tri_stance}\n")
            self.txt_heatmap.insert(tk.END, "Engines:\n")
            total_w = sum(ht.engine_contrib.values()) or 1.0
            for name, w in ht.engine_contrib.items():
                pct = 100.0 * w / total_w
                bar = "#" * int(pct / 5)
                self.txt_heatmap.insert(tk.END, f"  {name:8s}: {pct:5.1f}% {bar}\n")
            self.txt_heatmap.insert(tk.END, "\nTop organs by risk:\n")
            for name, r in sorted(ht.organ_risk.items(), key=lambda x: x[1], reverse=True)[:5]:
                self.txt_heatmap.insert(tk.END, f"  {name:20s}: {r:.2f}\n")
            self.txt_heatmap.insert(
                tk.END,
                "\nFingerprints: " + (", ".join(ht.fingerprints_triggered) or "none") + "\n"
            )
            self.txt_heatmap.insert(
                tk.END,
                "Meta rules: " + (", ".join(ht.meta_rules_fired) or "none") + "\n"
            )

        # Augmentations panel
        self.txt_aug.delete("1.0", tk.END)
        self.txt_aug.insert(tk.END, "Active augmentations:\n")
        for aug in self.augmentation_engine.list_active():
            last = time.strftime("%H:%M:%S", time.localtime(aug.last_applied)) if aug.last_applied else "never"
            self.txt_aug.insert(
                tk.END,
                f"  {aug.name:20s} [{aug.type}] impact={aug.impact_score:.2f} last={last}\n"
            )
        self.txt_aug.insert(
            tk.END,
            f"\nLast mutation: {self.augmentation_engine.last_mutation or 'none'}\n"
        )

        # Altered States tab
        text_states = []
        text_states.append("Altered States Cortex")
        text_states.append("---------------------")
        text_states.append(f"Current Meta-State: {self.brain.current_meta_state_name}")
        text_states.append(f"Current Tri-Stance: {self.brain.tri_stance}")
        text_states.append("")
        for name, m in self.brain.meta_states.items():
            marker = ">>" if name == self.brain.current_meta_state_name else "  "
            text_states.append(f"{marker} {name}:")
            text_states.append(
                f"    Horizons: short={m.prediction_horizon_short}s, "
                f"med={m.prediction_horizon_medium}s, long={m.prediction_horizon_long}s"
            )
            text_states.append(
                f"    Agg={m.stance_aggressiveness:.2f} DeepRAM={m.deep_ram_appetite_bias:.2f} "
                f"Threads={m.thread_expansion_bias:.2f} Cache={m.cache_behavior_bias:.2f}"
            )
            text_states.append("")
        text_states.append("Tri-Stance Effects:")
        text_states.append("  Conservative: shrink DeepRAM, slow ingestion, high dampening.")
        text_states.append("  Balanced: smooth scaling, moderate appetite.")
        text_states.append("  Beast: aggressive VRAM/RAM, fast switching, low dampening.")
        self.lbl_states.config(text="\n".join(text_states))

    def _draw_micro_chart(self):
        c = self.canvas_chart
        c.delete("all")
        w = int(c.winfo_width() or 200)
        h = int(c.winfo_height() or 90)
        if not self.pred_history:
            return

        meta_color = {
            "Hyper-Flow": "#222233",
            "Deep-Dream": "#332244",
            "Sentinel": "#233322",
            "Recovery-Flow": "#332222",
        }.get(self.brain.current_meta_state_name, "#111111")
        c.create_rectangle(0, 0, w, h, fill=meta_color, outline="")

        baseline_y = h - int(self.brain.baseline_risk * (h - 10)) - 5
        c.create_line(0, baseline_y, w, baseline_y, fill="#555555", dash=(2, 2))

        stance_color = {
            "Conservative": "#66b2ff",
            "Balanced": "#88ff88",
            "Beast": "#ff6666",
        }.get(self.brain.tri_stance, "#ffffff")

        short_col = "#ffaa00"
        med_col = stance_color
        long_col = "#8888ff"
        best_col = "#ffffff"

        n = len(self.pred_history)
        if n < 2:
            return
        step_x = max(1, w // self.chart_max_points)

        def to_xy(idx: int, val: float) -> Tuple[int, int]:
            x = idx * step_x
            y = h - int(val * (h - 10)) - 5
            return x, y

        for i in range(n - 1):
            s1, m1, l1, b1 = self.pred_history[i]
            s2, m2, l2, b2 = self.pred_history[i + 1]
            x1, ys1 = to_xy(i, s1)
            x2, ys2 = to_xy(i + 1, s2)
            _, ym1 = to_xy(i, m1)
            _, ym2 = to_xy(i + 1, m2)
            _, yl1 = to_xy(i, l1)
            _, yl2 = to_xy(i + 1, l2)
            _, yb1 = to_xy(i, b1)
            _, yb2 = to_xy(i + 1, b2)

            c.create_line(x1, yl1, x2, yl2, fill=long_col, width=1)
            c.create_line(x1, ys1, x2, ys2, fill=short_col, width=1)
            c.create_line(x1, ym1, x2, ym2, fill=med_col, width=2)
            c.create_line(x1, yb1, x2, yb2, fill=best_col, width=2)

        x_now = (n - 1) * step_x
        c.create_line(x_now, 0, x_now, h, fill="#ffffff", dash=(2, 4))

        c.create_text(
            5, 5, anchor="nw",
            text="S/M/L/Best (short/med/long/best); baseline; stance color on M",
            fill="#bbbbbb",
            font=("Consolas", 7),
        )

# ------------------------------------------------------------
# 12. main()
# ------------------------------------------------------------

def main():
    root = tk.Tk()
    app = NerveCenterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

