#!/usr/bin/env python3
"""
Fused Organism:
- Guardian Hybrid Brain with Foresight and Baselines
- Capsule Cortex Organ with Symbolic Capsules, Daemon, Swarm, Node
- External Flask API (port 6666)
- Tkinter GUI (Guardian console + Capsule Cortex tile + details)
- Encrypted persistent memory (local + SMB mirroring)
"""

import os
import sys
import time
import json
import math
import random
import socket
import hashlib
import threading
import platform
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from flask import Flask, request, jsonify
from cryptography.fernet import Fernet


# ---------------------------------------------------------------------------
# Shared constants / keys
# ---------------------------------------------------------------------------

COLOR_MAP = {
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "red": "#e74c3c",
    "grey": "#7f8c8d",
}

CONSCIOUSNESS_STATES = ["CALM", "FOCUSED", "ALERT", "PARANOID"]

SENSITIVITY_FACTORS = {
    "CALM": 0.7,
    "FOCUSED": 1.0,
    "ALERT": 1.3,
    "PARANOID": 1.6,
}

# Network drive keys
PRIMARY_DRIVE_KEY = "network_drive_primary"
FALLBACK_DRIVE_KEY = "network_drive_fallback"
DRIVE_STATUS_KEY = "network_drive_status"
STORAGE_LOCATION_KEY = "storage_location"
STORAGE_SMB_PATH_KEY = "storage_smb_path"

EVENT_HISTORY_PREFIX = "events:"

# Capsule-specific keys
CAPSULE_EVENTS_KEY_PREFIX = "capsule_events:"
CAPSULE_METRICS_KEY = "capsule_metrics"


# ---------------------------------------------------------------------------
# Brain state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BrainState:
    level: str = "grey"
    judgment: float = 0.0
    confidence: float = 0.0
    situational_awareness: float = 0.0
    predictive_intelligence: float = 0.0
    collective_health_score: float = 0.0
    foresight_score: float = 0.0
    best_guess: str = "Idle"


@dataclass
class ModeStatus:
    brain: BrainState = field(default_factory=BrainState)
    raw_details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Encrypted storage
# ---------------------------------------------------------------------------

class EncryptedStorage:
    def __init__(self, base_dir: str, filename: str = "guardian_state.bin"):
        self.base_dir = base_dir
        self.filename = filename
        self.path = os.path.join(self.base_dir, self.filename)
        os.makedirs(self.base_dir, exist_ok=True)
        self.key_path = os.path.join(self.base_dir, "guardian_key.key")
        self.fernet = self._load_or_create_key()

    def _load_or_create_key(self) -> Fernet:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                key = f.read().strip()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as f:
                f.write(key)
        return Fernet(key)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "rb") as f:
                data = f.read()
            decrypted = self.fernet.decrypt(data)
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        try:
            raw = json.dumps(data).encode("utf-8")
            encrypted = self.fernet.encrypt(raw)
            with open(self.path, "wb") as f:
                f.write(encrypted)
        except Exception as e:
            print(f"[WARN] Failed to save encrypted storage: {e}")


# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------

class MemoryBackend:
    def __init__(self, persistent_storage: Optional[EncryptedStorage] = None):
        self._store: Dict[str, Any] = {}
        self.persistent_storage = persistent_storage
        if self.persistent_storage:
            self._load_persistent()

    def _load_persistent(self):
        data = self.persistent_storage.load()
        if isinstance(data, dict):
            self._store.update(data)

    def _flush_persistent(self):
        if self.persistent_storage:
            self.persistent_storage.save(self._store)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._flush_persistent()

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def replace_storage(self, new_storage: EncryptedStorage):
        self.persistent_storage = new_storage
        self._flush_persistent()


# ---------------------------------------------------------------------------
# Baseline, history, profiles, events
# ---------------------------------------------------------------------------

class BaselineManager:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory

    def set_baseline(self, name: str, data: Any) -> None:
        self.memory.set(f"baseline:{name}", data)

    def get_baseline(self, name: str) -> Optional[Any]:
        return self.memory.get(f"baseline:{name}")

    def add_history(self, name: str, snapshot: Dict[str, Any]) -> None:
        key = f"history:{name}"
        history: List[Tuple[float, Dict[str, Any]]] = self.memory.get(key, [])
        history.append((time.time(), snapshot))
        if len(history) > 500:
            history = history[-500:]
        self.memory.set(key, history)

    def get_history(self, name: str) -> List[Tuple[float, Dict[str, Any]]]:
        return self.memory.get(f"history:{name}", [])

    def add_profile_point(self, mode: str, snapshot: Dict[str, Any]):
        key = f"profile:{mode}"
        profiles = self.memory.get(key, {})
        t = time.localtime()
        bucket = f"{t.tm_wday}-{t.tm_hour}"
        entry = profiles.get(bucket, {"count": 0, "process_sum": 0, "conn_sum": 0})
        entry["count"] += 1
        entry["process_sum"] += snapshot.get("process_count", 0)
        entry["conn_sum"] += snapshot.get("connection_count", 0)
        profiles[bucket] = entry
        self.memory.set(key, profiles)

    def get_profile_expectation(self, mode: str) -> Dict[str, float]:
        key = f"profile:{mode}"
        profiles = self.memory.get(key, {})
        t = time.localtime()
        bucket = f"{t.tm_wday}-{t.tm_hour}"
        entry = profiles.get(bucket)
        if not entry or entry["count"] == 0:
            return {}
        return {
            "process_expected": entry["process_sum"] / entry["count"],
            "conn_expected": entry["conn_sum"] / entry["count"],
        }

    def add_event(self, mode: str, event: Dict[str, Any], max_events: int = 50):
        key = f"{EVENT_HISTORY_PREFIX}{mode}"
        events: List[Dict[str, Any]] = self.memory.get(key, [])
        events.append(event)
        if len(events) > max_events:
            events = events[-max_events:]
        self.memory.set(key, events)

    def get_events(self, mode: str) -> List[Dict[str, Any]]:
        return self.memory.get(f"{EVENT_HISTORY_PREFIX}{mode}", [])


# ---------------------------------------------------------------------------
# Inference + Risk
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory
        self.accelerator_available = False

    def infer(self, mode_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        deviation = float(features.get("deviation_score", 0.0))
        anomaly_count = int(features.get("anomaly_count", 0))
        baseline_age = float(features.get("baseline_age", 0.0))

        suggestions: Dict[str, Any] = {
            "raise_level": None,
            "adjust_confidence": 0.0,
            "adjust_judgment": 0.0,
        }

        if deviation > 0.7 or anomaly_count > 5:
            suggestions["raise_level"] = "red"
            suggestions["adjust_judgment"] += 15.0
        elif deviation > 0.3 or anomaly_count > 0:
            suggestions["raise_level"] = "yellow"
            suggestions["adjust_judgment"] += 5.0

        if baseline_age > 24 * 3600:
            suggestions["adjust_confidence"] -= 10.0

        return suggestions


class RiskModel:
    def score(self, features: Dict[str, Any]) -> float:
        dev = features.get("deviation", 0.0)
        anomalies = features.get("anomaly_count", 0)
        baseline_age = features.get("baseline_age", 0.0)
        pi = features.get("predictive_intelligence", 50.0)
        sa = features.get("situational_awareness", 50.0)
        drive_status = features.get("drive_status", "green")
        capsule_risk = features.get("capsule_risk", 0.0)

        risk = 0.0
        risk += dev * 60.0
        risk += anomalies * 4.0

        if baseline_age > 48 * 3600:
            risk += 10.0
        if pi < 40.0:
            risk += 10.0
        if sa < 40.0:
            risk += 10.0
        if drive_status == "yellow":
            risk += 5.0
        elif drive_status == "red":
            risk += 15.0

        risk += capsule_risk

        return max(0.0, min(100.0, risk))


# ---------------------------------------------------------------------------
# Data physics + foresight
# ---------------------------------------------------------------------------

class DataPhysicsEngine:
    def __init__(self, baseline_manager: BaselineManager, inference_engine: InferenceEngine, risk_model: RiskModel):
        self.baseline_manager = baseline_manager
        self.inference_engine = inference_engine
        self.risk_model = risk_model

    def compute_brain_state(
        self,
        mode_name: str,
        snapshot: Dict[str, Any],
        consciousness_state: str,
    ) -> ModeStatus:

        baseline = self.baseline_manager.get_baseline(mode_name)
        history = self.baseline_manager.get_history(mode_name)
        now = time.time()

        deviation = self._compute_deviation_score(mode_name, baseline, snapshot)
        anomaly_count = int(snapshot.get("anomaly_count", 0))

        baseline_time = snapshot.get("baseline_time", now)
        baseline_age = float(now - baseline_time)

        judgment = min(100.0, deviation * 120 + anomaly_count * 5)
        confidence = 80.0 - deviation * 30.0
        situational_awareness = self._compute_coverage(snapshot, history)
        predictive_intelligence = self._compute_predictive_power(history, snapshot)

        sensitivity_factor = SENSITIVITY_FACTORS.get(consciousness_state, 1.0)
        judgment *= sensitivity_factor
        deviation *= sensitivity_factor

        features = {
            "deviation_score": deviation,
            "anomaly_count": anomaly_count,
            "baseline_age": baseline_age,
            "history_len": len(history),
        }
        suggestions = self.inference_engine.infer(mode_name, features)

        judgment += suggestions.get("adjust_judgment", 0.0)
        confidence += suggestions.get("adjust_confidence", 0.0)

        judgment = max(0.0, min(100.0, judgment))
        confidence = max(0.0, min(100.0, confidence))
        situational_awareness = max(0.0, min(100.0, situational_awareness))
        predictive_intelligence = max(0.0, min(100.0, predictive_intelligence))

        health = (
            0.35 * confidence
            + 0.25 * predictive_intelligence
            + 0.2 * situational_awareness
            - 30.0 * deviation
            - 2.0 * anomaly_count
        )
        health = max(0.0, min(100.0, health))

        drive_status = snapshot.get("drive_status", self.baseline_manager.memory.get(DRIVE_STATUS_KEY, "green"))

        capsule_risk = snapshot.get("capsule_risk", 0.0)
        risk_features = {
            "deviation": deviation,
            "anomaly_count": anomaly_count,
            "baseline_age": baseline_age,
            "predictive_intelligence": predictive_intelligence,
            "situational_awareness": situational_awareness,
            "drive_status": drive_status,
            "capsule_risk": capsule_risk,
        }
        risk_score = self.risk_model.score(risk_features)
        health -= risk_score * 0.3
        health = max(0.0, min(100.0, health))

        foresight_score = self._compute_foresight(
            mode_name, history, health, risk_score, predictive_intelligence
        )

        if health >= 70.0:
            level = "green"
        elif health >= 40.0:
            level = "yellow"
        else:
            level = "red"

        raise_level = suggestions.get("raise_level")
        if raise_level in ("yellow", "red"):
            level = raise_level

        if level == "green" and foresight_score < 50.0:
            level = "yellow"
        if level == "yellow" and foresight_score < 30.0:
            level = "red"

        best_guess = self._compose_best_guess(
            level, deviation, anomaly_count, consciousness_state, foresight_score
        )

        brain = BrainState(
            level=level,
            judgment=judgment,
            confidence=confidence,
            situational_awareness=situational_awareness,
            predictive_intelligence=predictive_intelligence,
            collective_health_score=health,
            foresight_score=foresight_score,
            best_guess=best_guess,
        )

        return ModeStatus(
            brain=brain,
            raw_details={
                "deviation": deviation,
                "anomaly_count": anomaly_count,
                "risk_score": risk_score,
                "baseline_age": baseline_age,
                "history_len": len(history),
            },
        )

    def _compute_deviation_score(self, mode_name: str, baseline: Any, snapshot: Dict[str, Any]) -> float:
        score = 0.0
        if baseline is not None:
            if "process_count" in snapshot and "process_count" in baseline:
                b = baseline["process_count"]
                c = snapshot["process_count"]
                if b > 0:
                    score += abs(c - b) / max(b, 1)
            if "connection_count" in snapshot and "connection_count" in baseline:
                b = baseline["connection_count"]
                c = snapshot["connection_count"]
                if b > 0:
                    score += 0.5 * abs(c - b) / max(b, 1)

        profile = self.baseline_manager.get_profile_expectation(mode_name)
        if profile:
            pe = profile.get("process_expected", 0)
            ce = profile.get("conn_expected", 0)
            if pe > 0:
                score += 0.3 * abs(snapshot.get("process_count", 0) - pe) / max(pe, 1)
            if ce > 0:
                score += 0.3 * abs(snapshot.get("connection_count", 0) - ce) / max(ce, 1)

        if baseline is None and not profile:
            score = 0.3

        return max(0.0, min(1.0, score))

    def _compute_coverage(self, snapshot: Dict[str, Any], history: List[Any]) -> float:
        metrics_present = 0
        for k in ("process_count", "connection_count", "personal_files_protected", "worker_alerts", "capsule_count"):
            if k in snapshot:
                metrics_present += 1
        coverage = metrics_present * 20.0 + len(history) * 0.5
        return coverage

    def _compute_predictive_power(self, history: List[Any], snapshot: Dict[str, Any]) -> float:
        if len(history) < 5:
            return 40.0

        proc = snapshot.get("process_count", 0)
        conn = snapshot.get("connection_count", 0)

        proc_vals = [h.get("process_count", 0) for _, h in history]
        conn_vals = [h.get("connection_count", 0) for _, h in history]

        def trend_score(values):
            if len(values) < 3:
                return 0.0
            v1 = values[-1] - values[-2]
            v2 = values[-2] - values[-3]
            accel = v1 - v2
            return max(0.0, 40.0 - min(40.0, abs(accel)))

        proc_trend = trend_score(proc_vals)
        conn_trend = trend_score(conn_vals)

        score = 40.0 + proc_trend * 0.3 + conn_trend * 0.3

        def near_trend(values, current):
            if len(values) < 3:
                return True
            v1 = values[-1] - values[-2]
            predicted = values[-1] + v1
            return abs(current - predicted) <= max(5, 0.2 * max(values))

        if near_trend(proc_vals, proc):
            score += 10.0
        if near_trend(conn_vals, conn):
            score += 10.0

        return max(0.0, min(100.0, score))

    def _compute_foresight(
        self,
        mode_name: str,
        history: List[Tuple[float, Dict[str, Any]]],
        current_health: float,
        risk_score: float,
        predictive_intelligence: float,
    ) -> float:
        if len(history) < 5:
            base = current_health - risk_score * 0.2
            return max(0.0, min(100.0, base))

        health_vals = []
        for _, snap in history:
            h = snap.get("__health_estimate__", None)
            if h is not None:
                health_vals.append(h)

        if not health_vals:
            base = current_health - risk_score * 0.2
        else:
            recent = health_vals[-5:]
            if len(recent) >= 2:
                v = recent[-1] - recent[-2]
            else:
                v = 0.0
            predicted_future = current_health + v * 2.0
            base = predicted_future - risk_score * 0.25

        if predictive_intelligence < 40.0:
            base -= 10.0

        return max(0.0, min(100.0, base))

    def _compose_best_guess(
        self,
        level: str,
        deviation: float,
        anomalies: int,
        consciousness_state: str,
        foresight_score: float,
    ) -> str:
        future_msg = ""
        if foresight_score >= 70.0:
            future_msg = " trajectory stable"
        elif foresight_score >= 40.0:
            future_msg = " trajectory uncertain"
        else:
            future_msg = " trajectory heading toward trouble"

        if level == "green":
            if anomalies == 0 and deviation < 0.2:
                return f"{consciousness_state}: Stable, behavior matches expectations," + future_msg
            return f"{consciousness_state}: Small deviations," + future_msg
        elif level == "yellow":
            return f"{consciousness_state}: Pattern shift detected," + future_msg
        else:
            if anomalies > 0:
                return f"{consciousness_state}: High-risk anomaly cluster," + future_msg
            return f"{consciousness_state}: Severe deviation," + future_msg


# ---------------------------------------------------------------------------
# Guardian base modes
# ---------------------------------------------------------------------------

class BaseMode:
    def __init__(
        self,
        name: str,
        baseline_manager: BaselineManager,
        memory: MemoryBackend,
        physics_engine: DataPhysicsEngine,
    ):
        self.name = name
        self.baseline_manager = baseline_manager
        self.memory = memory
        self.physics_engine = physics_engine
        self.status = ModeStatus()

    def init_baseline(self):
        snapshot = self._capture_snapshot()
        snapshot["baseline_time"] = time.time()
        self.baseline_manager.set_baseline(self.name, snapshot)
        self.baseline_manager.add_history(self.name, snapshot)
        self.status = self.physics_engine.compute_brain_state(self.name, snapshot, "FOCUSED")

    def update_state(self, consciousness_state: str):
        snapshot = self._capture_snapshot()

        stale_flag = self.memory.get(f"baseline_stale:{self.name}", False)
        if stale_flag:
            history = self.baseline_manager.get_history(self.name)
            if len(history) >= 10:
                median_snapshot = history[-1][1]
                median_snapshot["baseline_time"] = time.time()
                self.baseline_manager.set_baseline(self.name, median_snapshot)
                self.memory.set(f"baseline_stale:{self.name}", False)

        self.baseline_manager.add_history(self.name, snapshot)
        self.baseline_manager.add_profile_point(self.name, snapshot)
        self.status = self.physics_engine.compute_brain_state(self.name, snapshot, consciousness_state)

        brain = self.status.brain
        snapshot["__health_estimate__"] = brain.collective_health_score
        self.baseline_manager.add_history(self.name, snapshot)

        deviation = self.status.raw_details.get("deviation", 0.0)
        anomaly_count = self.status.raw_details.get("anomaly_count", 0)
        if deviation > 0.9 and anomaly_count == 0:
            self.memory.set(f"baseline_stale:{self.name}", True)

        self._maybe_record_event(brain, self.status.raw_details)

    def get_status(self) -> ModeStatus:
        return self.status

    def _capture_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _maybe_record_event(self, brain: BrainState, details: Dict[str, Any]):
        if brain.level not in ("yellow", "red"):
            return
        event = {
            "timestamp": time.time(),
            "level": brain.level,
            "health": brain.collective_health_score,
            "foresight": brain.foresight_score,
            "judgment": brain.judgment,
            "confidence": brain.confidence,
            "situational_awareness": brain.situational_awareness,
            "predictive_intelligence": brain.predictive_intelligence,
            "best_guess": brain.best_guess,
            "deviation": details.get("deviation"),
            "anomaly_count": details.get("anomaly_count"),
            "risk_score": details.get("risk_score"),
        }
        self.baseline_manager.add_event(self.name, event)


class PresidentialMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        procs = list(psutil.process_iter(["pid"]))
        process_count = len(procs)
        conn_count = 0
        try:
            conns = psutil.net_connections(kind="inet")
            conn_count = len(conns)
        except Exception:
            pass
        anomaly_count = max(0, process_count - 200)
        return {
            "mode_name": "Presidential",
            "process_count": process_count,
            "connection_count": conn_count,
            "anomaly_count": anomaly_count,
        }


class BloodhoundMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        suspicious = 0
        total = 0
        try:
            for c in psutil.net_connections(kind="inet"):
                if not c.raddr:
                    continue
                total += 1
                ip = c.raddr.ip
                if ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172.16."):
                    continue
                suspicious += 1
        except Exception:
            pass
        return {
            "mode_name": "Bloodhound",
            "connection_count": total,
            "anomaly_count": suspicious,
        }


class GodMode(BaseMode):
    CRITICAL_PATHS = [
        r"C:\Windows\System32\kernel32.dll",
        r"C:\Windows\System32\ntdll.dll",
    ]

    def _capture_snapshot(self) -> Dict[str, Any]:
        primary = self.memory.get(PRIMARY_DRIVE_KEY)
        fallback = self.memory.get(FALLBACK_DRIVE_KEY)

        primary_ok = self._test_path(primary) if primary else False
        fallback_ok = self._test_path(fallback) if fallback else False

        integrity_ok, changed_files = self._check_integrity()
        backups_ok = primary_ok or fallback_ok

        anomaly_count = 0
        if not integrity_ok:
            anomaly_count += 3
        if not backups_ok:
            anomaly_count += 2

        drive_status_level = "red"
        if backups_ok:
            if primary_ok and fallback_ok:
                drive_status_level = "green"
            else:
                drive_status_level = "yellow"

        self.memory.set(DRIVE_STATUS_KEY, drive_status_level)

        return {
            "mode_name": "God",
            "integrity_ok": integrity_ok,
            "backups_ok": backups_ok,
            "primary_ok": primary_ok,
            "fallback_ok": fallback_ok,
            "changed_files": changed_files,
            "anomaly_count": anomaly_count,
            "drive_status": drive_status_level,
        }

    def _test_path(self, path: Optional[str]) -> bool:
        if not path:
            return False
        try:
            return os.path.exists(path)
        except Exception:
            return False

    def _hash_file(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _check_integrity(self) -> Tuple[bool, List[str]]:
        changed = []
        baseline_hashes = self.memory.get("god_integrity_baseline", {})
        current_hashes = {}
        for path in self.CRITICAL_PATHS:
            hash_val = self._hash_file(path)
            current_hashes[path] = hash_val
            baseline_hash = baseline_hashes.get(path)
            if baseline_hash is None:
                continue
            if baseline_hash != hash_val:
                changed.append(path)

        if not baseline_hashes:
            self.memory.set("god_integrity_baseline", current_hashes)
            return True, []

        integrity_ok = (len(changed) == 0)
        return integrity_ok, changed


class SoftwareInterrogatorMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        proc_count = 0
        try:
            for _ in psutil.process_iter(["pid"]):
                proc_count += 1
        except Exception:
            pass
        anomaly_count = max(0, proc_count - 200)
        return {
            "mode_name": "SoftwareInterrogation",
            "process_count": proc_count,
            "anomaly_count": anomaly_count,
        }


class SnowWhiteMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        personal_files_protected = True
        anomaly_count = 0 if personal_files_protected else 5
        return {
            "mode_name": "SnowWhite",
            "personal_files_protected": personal_files_protected,
            "anomaly_count": anomaly_count,
        }


class BorgHiveMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        worker_alerts = 0
        return {
            "mode_name": "BorgHive",
            "worker_alerts": worker_alerts,
            "anomaly_count": worker_alerts,
        }


# ---------------------------------------------------------------------------
# Capsule Engine / Cortex
# ---------------------------------------------------------------------------

class BitType(Enum):
    FUSION = '⨂'
    XOR = '⊕'
    TENSOR = '⊗'
    GRADIENT = '∇'
    PRIMAL = '●'
    VOID = '∅'


class MutationState(Enum):
    PRISTINE = 'pristine'
    FUSED = 'fused'
    DECAYED = 'decayed'
    RESONANT = 'resonant'
    CHAOTIC = 'chaotic'


αB = {
    b: {
        m: v
        for m, v in zip(
            MutationState,
            [1280, 2048, 640, 4096, 8192] if b == BitType.FUSION else
            [320, 512, 160, 1024, 2048] if b == BitType.XOR else
            [512, 1024, 256, 2048, 4096] if b == BitType.TENSOR else
            [256, 512, 128, 1024, 2048] if b == BitType.GRADIENT else
            [64, 128, 32, 256, 512] if b == BitType.PRIMAL else [0]*5
        )
    }
    for b in BitType
}


def flops_for_capsule(b: BitType, m: MutationState, e: float, t: float) -> float:
    return αB[b][m] * 1e12 * e * t * (psutil.cpu_percent() / 100) * (psutil.virtual_memory().percent / 100)


def encode_symbolic(d: str, b: BitType, m: MutationState) -> str:
    return f"{b.value}[{m.value}]::{d}::{m.value}[{b.value}]"


def reverse_string(d: str) -> str:
    return ''.join(chr(~ord(c) & 0xFF) for c in d)


class Capsule:
    def __init__(self, raw: str, flop: bool = False):
        self.b, self.m = (BitType.VOID, MutationState.CHAOTIC) if flop else (
            random.choice(list(BitType)), random.choice(list(MutationState))
        )
        self.e = round((psutil.cpu_percent() + psutil.virtual_memory().percent + 50) / 300, 2)
        loadavg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0
        self.t = round(1 + math.sin(loadavg / 10), 2)
        self.f = flops_for_capsule(self.b, self.m, self.e, self.t)
        payload = ''.join(random.choice("X#@$%&*!") for _ in range(32)) if flop else raw
        symbolic = encode_symbolic(payload, self.b, self.m)
        if "backdoor" in raw.lower():
            symbolic = f"☠ {symbolic} ☠"
        self.data = reverse_string(symbolic)
        threading.Timer(86400 if not flop else 300, self.destroy).start()

    def destroy(self):
        self.data = None


class CapsuleState:
    def __init__(self):
        self.state = {
            "active": False,
            "self_destruct": False,
            "visibility": False,
            "key": Fernet.generate_key(),
        }
        self.fernet = Fernet(self.state["key"])
        self.capsule_store: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def set_active(self, active: bool):
        self.state["active"] = active

    def toggle_self_destruct(self) -> bool:
        self.state["self_destruct"] ^= 1
        return self.state["self_destruct"]

    def toggle_visibility(self) -> bool:
        self.state["visibility"] ^= 1
        return self.state["visibility"]

    def create_capsule(self, data: str, duration: int, flop: bool = False) -> Dict[str, Any]:
        with self.lock:
            c = Capsule(data, flop=flop)
            enc = self.fernet.encrypt(c.data.encode()).decode()
            cid = f"capsule_{int(time.time()*1000)}"
            self.capsule_store[cid] = {
                "capsule": enc,
                "timestamp": time.time(),
                "flops": c.f,
                "bit": c.b.value,
                "mutation": c.m.value,
            }
            threading.Timer(duration, lambda: self.capsule_store.pop(cid, None)).start()
            return {
                "id": cid,
                "capsule": enc,
                "timestamp": self.capsule_store[cid]["timestamp"],
                "flops": c.f,
                "bit": c.b.value,
                "mutation": c.m.value,
            }

    def get_all_capsules(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.capsule_store)

    def decrypt_capsule(self, enc: str) -> str:
        return self.fernet.decrypt(enc.encode()).decode()


class CapsuleMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            "total_capsules": 0,
            "unauthorized_attempts": 0,
            "flop_sum": 0.0,
            "last_bit": None,
            "last_mutation": None,
            "entropy": 0.0,
            "api_calls": 0,
            "self_destruct_toggles": 0,
            "visibility_toggles": 0,
        }

    def record_capsule(self, bit: str, mutation: str, flops: float):
        with self.lock:
            self.metrics["total_capsules"] += 1
            self.metrics["flop_sum"] += flops
            self.metrics["last_bit"] = bit
            self.metrics["last_mutation"] = mutation

    def record_unauthorized(self):
        with self.lock:
            self.metrics["unauthorized_attempts"] += 1

    def record_api_call(self):
        with self.lock:
            self.metrics["api_calls"] += 1

    def record_self_destruct_toggle(self):
        with self.lock:
            self.metrics["self_destruct_toggles"] += 1

    def record_visibility_toggle(self):
        with self.lock:
            self.metrics["visibility_toggles"] += 1

    def set_entropy(self, value: float):
        with self.lock:
            self.metrics["entropy"] = value

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.metrics)


class Swarm:
    def sync(self):
        host = socket.gethostbyname(socket.gethostname())
        sysname = platform.system()
        print(f"[Swarm] {host} → {sysname}")


class Node:
    def __init__(self, id_: str):
        self.id = id_
        self.entropy = random.random()

    def mutate(self):
        self.entropy = random.random()

    def replicate(self) -> 'Node':
        return Node(f"{self.id}_clone_{int(time.time())}")


class CapsuleDaemon:
    def __init__(self, capsule_state: CapsuleState, metrics: CapsuleMetrics):
        self.capsule_state = capsule_state
        self.metrics = metrics
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            time.sleep(5)
            if self.capsule_state.state["active"]:
                data = f"daemon_capsule_{random.randint(1000,9999)}"
                record = self.capsule_state.create_capsule(data, duration=300, flop=(random.random() < 0.3))
                self.metrics.record_capsule(record["bit"], record["mutation"], record["flops"])


# ---------------------------------------------------------------------------
# Capsule Cortex Mode (Guardian organ)
# ---------------------------------------------------------------------------

class CapsuleCortexMode(BaseMode):
    def __init__(
        self,
        name: str,
        baseline_manager: BaselineManager,
        memory: MemoryBackend,
        physics_engine: DataPhysicsEngine,
        capsule_state: CapsuleState,
        capsule_metrics: CapsuleMetrics,
    ):
        super().__init__(name, baseline_manager, memory, physics_engine)
        self.capsule_state = capsule_state
        self.capsule_metrics = capsule_metrics

    def _capture_snapshot(self) -> Dict[str, Any]:
        m = self.capsule_metrics.snapshot()
        total_capsules = m.get("total_capsules", 0)
        unauthorized = m.get("unauthorized_attempts", 0)
        flop_sum = m.get("flop_sum", 0.0)
        entropy = m.get("entropy", 0.0)
        api_calls = m.get("api_calls", 0)
        sd_toggles = m.get("self_destruct_toggles", 0)
        vis_toggles = m.get("visibility_toggles", 0)

        anomaly_count = unauthorized + sd_toggles + vis_toggles
        capsule_risk = (
            unauthorized * 20.0
            + (flop_sum / 1e12 if flop_sum > 0 else 0) * 5.0
            + entropy * 10.0
            + api_calls * 0.2
            + sd_toggles * 5.0
            + vis_toggles * 3.0
        )
        capsule_risk = min(capsule_risk, 100.0)

        self.memory.set(CAPSULE_METRICS_KEY, m)

        return {
            "mode_name": "CapsuleCortex",
            "capsule_count": total_capsules,
            "unauthorized": unauthorized,
            "flop_sum": flop_sum,
            "entropy": entropy,
            "api_calls": api_calls,
            "self_destruct_toggles": sd_toggles,
            "visibility_toggles": vis_toggles,
            "process_count": total_capsules,
            "connection_count": api_calls,
            "anomaly_count": anomaly_count,
            "capsule_risk": capsule_risk,
        }

    def _maybe_record_event(self, brain: BrainState, details: Dict[str, Any]):
        if brain.level not in ("yellow", "red"):
            return
        m = self.capsule_metrics.snapshot()
        event = {
            "timestamp": time.time(),
            "level": brain.level,
            "health": brain.collective_health_score,
            "foresight": brain.foresight_score,
            "judgment": brain.judgment,
            "confidence": brain.confidence,
            "situational_awareness": brain.situational_awareness,
            "predictive_intelligence": brain.predictive_intelligence,
            "best_guess": brain.best_guess,
            "deviation": details.get("deviation"),
            "anomaly_count": details.get("anomaly_count"),
            "risk_score": details.get("risk_score"),
            "capsule_metrics": m,
        }
        self.baseline_manager.add_event(self.name, event)


# ---------------------------------------------------------------------------
# Guardian Engine (with Capsule Cortex)
# ---------------------------------------------------------------------------

class GuardianEngine:
    def __init__(self):
        self.local_base = os.path.join(os.path.expanduser("~"), "GuardianLocal")
        self.storage = EncryptedStorage(self.local_base)
        self.memory = MemoryBackend(self.storage)

        self.baseline_manager = BaselineManager(self.memory)
        self.inference_engine = InferenceEngine(self.memory)
        self.risk_model = RiskModel()
        self.physics_engine = DataPhysicsEngine(self.baseline_manager, self.inference_engine, self.risk_model)

        self.consciousness_state = "FOCUSED"
        self._stop = False

        self.capsule_state = CapsuleState()
        self.capsule_metrics = CapsuleMetrics()

        self.node = Node("node_001")
        self.swarm = Swarm()
        self.daemon = CapsuleDaemon(self.capsule_state, self.capsule_metrics)

        self.swarm.sync()
        self.node.mutate()
        self.capsule_metrics.set_entropy(self.node.entropy)

        self.modes: Dict[str, BaseMode] = {
            "Presidential": PresidentialMode("Presidential", self.baseline_manager, self.memory, self.physics_engine),
            "Bloodhound": BloodhoundMode("Bloodhound", self.baseline_manager, self.memory, self.physics_engine),
            "God": GodMode("God", self.baseline_manager, self.memory, self.physics_engine),
            "Software Interrogation": SoftwareInterrogatorMode("SoftwareInterrogation", self.baseline_manager, self.memory, self.physics_engine),
            "Snow White": SnowWhiteMode("SnowWhite", self.baseline_manager, self.memory, self.physics_engine),
            "Borg Hive": BorgHiveMode("BorgHive", self.baseline_manager, self.memory, self.physics_engine),
            "Capsule Cortex": CapsuleCortexMode("CapsuleCortex", self.baseline_manager, self.memory, self.physics_engine, self.capsule_state, self.capsule_metrics),
        }

        self._start_drive_retry_loop(interval_sec=60)
        threading.Thread(target=self._background_ritual, daemon=True).start()

    def init_baselines(self):
        for m in self.modes.values():
            m.init_baseline()

    def start_loop(self, interval_sec: int = 10):
        def loop():
            while not self._stop:
                self.node.mutate()
                self.capsule_metrics.set_entropy(self.node.entropy)
                for m in self.modes.values():
                    m.update_state(self.consciousness_state)
                time.sleep(interval_sec)
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def get_mode_status(self, name: str) -> ModeStatus:
        return self.modes[name].get_status()

    def get_mode_events(self, name: str) -> List[Dict[str, Any]]:
        return self.baseline_manager.get_events(name)

    def set_consciousness_state(self, state: str):
        if state in CONSCIOUSNESS_STATES:
            self.consciousness_state = state

    def get_primary_drive(self) -> Optional[str]:
        return self.memory.get(PRIMARY_DRIVE_KEY)

    def get_fallback_drive(self) -> Optional[str]:
        return self.memory.get(FALLBACK_DRIVE_KEY)

    def set_primary_drive(self, path: str):
        self.memory.set(PRIMARY_DRIVE_KEY, path)

    def set_fallback_drive(self, path: str):
        self.memory.set(FALLBACK_DRIVE_KEY, path)

    def test_drive(self, path: Optional[str]) -> bool:
        if not path:
            return False
        try:
            return os.path.exists(path)
        except Exception:
            return False

    def _start_drive_retry_loop(self, interval_sec: int = 60):
        def loop():
            while not self._stop:
                primary = self.get_primary_drive()
                fallback = self.get_fallback_drive()

                primary_ok = self.test_drive(primary)
                fallback_ok = self.test_drive(fallback)

                if primary_ok or fallback_ok:
                    drive_status_level = "green" if (primary_ok and fallback_ok) else "yellow"
                else:
                    drive_status_level = "red"

                self.memory.set(DRIVE_STATUS_KEY, drive_status_level)

                if drive_status_level in ("green", "yellow"):
                    self._ensure_smb_storage(primary if primary_ok else fallback)

                time.sleep(interval_sec)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _ensure_smb_storage(self, active_path: str):
        try:
            smb_dir = os.path.join(active_path, "GuardianState")
            os.makedirs(smb_dir, exist_ok=True)

            current_location = self.memory.get(STORAGE_LOCATION_KEY, "local")
            current_smb_path = self.memory.get(STORAGE_SMB_PATH_KEY, None)

            if current_location == "smb" and current_smb_path == smb_dir:
                return

            local_data = self.storage.load()
            smb_storage = EncryptedStorage(smb_dir)
            smb_storage.save(local_data)

            self.storage = smb_storage
            self.memory.replace_storage(smb_storage)

            self.memory.set(STORAGE_LOCATION_KEY, "smb")
            self.memory.set(STORAGE_SMB_PATH_KEY, smb_dir)
            print(f"[INFO] Switched persistent storage to SMB at {smb_dir}")
        except Exception as e:
            print(f"[WARN] Failed to mirror to SMB: {e}")

    def _background_ritual(self):
        while not self._stop:
            print("🌀 Background ritual running...")
            time.sleep(10)


# ---------------------------------------------------------------------------
# Flask API (external)
# ---------------------------------------------------------------------------

def create_flask_app(engine: GuardianEngine) -> Flask:
    app = Flask(__name__)

    @app.route("/activate", methods=["POST"])
    def activate():
        engine.capsule_state.set_active(True)
        return jsonify({"status": "activated"})

    @app.route("/deactivate", methods=["POST"])
    def deactivate():
        engine.capsule_state.set_active(False)
        return jsonify({"status": "deactivated"})

    @app.route("/toggle/self_destruct", methods=["POST"])
    def toggle_sd():
        val = engine.capsule_state.toggle_self_destruct()
        engine.capsule_metrics.record_self_destruct_toggle()
        return jsonify({"self_destruct": val})

    @app.route("/toggle/visibility", methods=["POST"])
    def toggle_vis():
        val = engine.capsule_state.toggle_visibility()
        engine.capsule_metrics.record_visibility_toggle()
        return jsonify({"visibility": val})

    @app.route("/protect", methods=["POST"])
    def protect():
        engine.capsule_metrics.record_api_call()
        if not engine.capsule_state.state["active"]:
            return jsonify({"error": "Engine not active"}), 403

        data = request.json.get("data", "")
        duration = int(request.json.get("duration", 300))
        if not data:
            return jsonify({"error": "No data"}), 400

        if engine.capsule_state.state["self_destruct"] and "unauthorized" in data.lower():
            engine.capsule_metrics.record_unauthorized()
            event = {
                "timestamp": time.time(),
                "type": "unauthorized_payload",
                "data_preview": data[:80],
            }
            engine.baseline_manager.add_event("CapsuleCortex", event)
            return jsonify({"status": "self-destruct triggered", "capsule": None})

        record = engine.capsule_state.create_capsule(data, duration=duration, flop=False)
        engine.capsule_metrics.record_capsule(record["bit"], record["mutation"], record["flops"])
        event = {
            "timestamp": time.time(),
            "type": "capsule_created",
            "id": record["id"],
            "flops": record["flops"],
            "bit": record["bit"],
            "mutation": record["mutation"],
        }
        engine.baseline_manager.add_event("CapsuleCortex", event)

        engine.baseline_manager.add_history("CapsuleCortex", {"capsule_event": record})

        return jsonify({
            "id": record["id"],
            "capsule": record["capsule"],
            "visibility": "visible" if engine.capsule_state.state["visibility"] else "cloaked",
            "timestamp": record["timestamp"],
            "flops": record["flops"],
            "bit": record["bit"],
            "mutation": record["mutation"],
        })

    @app.route("/decrypt", methods=["POST"])
    def decrypt():
        engine.capsule_metrics.record_api_call()
        try:
            capsule = request.json.get("capsule", "")
            data = engine.capsule_state.decrypt_capsule(capsule)
            event = {
                "timestamp": time.time(),
                "type": "capsule_decrypted",
            }
            engine.baseline_manager.add_event("CapsuleCortex", event)
            return jsonify({"data": data})
        except Exception as e:
            event = {
                "timestamp": time.time(),
                "type": "decrypt_error",
                "error": str(e),
            }
            engine.baseline_manager.add_event("CapsuleCortex", event)
            return jsonify({"error": str(e)}), 400

    @app.route("/capsules", methods=["GET"])
    def get_capsules():
        engine.capsule_metrics.record_api_call()
        return jsonify(engine.capsule_state.get_all_capsules())

    return app


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class ModeDetailsWindow(tk.Toplevel):
    def __init__(self, parent, engine: GuardianEngine, mode_name: str):
        super().__init__(parent)
        self.engine = engine
        self.mode_name = mode_name

        self.title(f"{mode_name} – Details")
        self.geometry("750x420")

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text=f"{mode_name} – Recent Events", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 5)
        )

        cols = ("time", "level", "health", "foresight", "dev", "anom", "risk", "summary")
        self.tree = ttk.Treeview(
            main_frame,
            columns=cols,
            show="headings",
            height=14,
        )
        for col in cols:
            self.tree.heading(col, text=col.capitalize())

        self.tree.column("time", width=130)
        self.tree.column("level", width=60)
        self.tree.column("health", width=70)
        self.tree.column("foresight", width=80)
        self.tree.column("dev", width=70)
        self.tree.column("anom", width=50)
        self.tree.column("risk", width=60)
        self.tree.column("summary", width=230)

        self.tree.pack(fill="both", expand=True)

        self._populate()

    def _populate(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        events = self.engine.get_mode_events(self.mode_name)
        events = sorted(events, key=lambda e: e.get("timestamp", 0), reverse=True)

        for ev in events:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.get("timestamp", time.time())))
            level = ev.get("level", ev.get("type", ""))
            health = f"{ev.get('health', 0):.0f}" if ev.get("health") is not None else ""
            foresight = f"{ev.get('foresight', 0):.0f}" if ev.get("foresight") is not None else ""
            dev = f"{ev.get('deviation', 0.0):.2f}" if ev.get("deviation") is not None else ""
            anom = str(ev.get("anomaly_count", ""))
            risk = f"{ev.get('risk_score', 0.0):.0f}" if ev.get("risk_score") is not None else ""
            summary = ev.get("best_guess", ev.get("type", ""))[:80]
            self.tree.insert(
                "",
                "end",
                values=(ts_str, level, health, foresight, dev, anom, risk, summary),
            )


class ModeFrame(ttk.Frame):
    def __init__(self, parent, mode_name: str, engine: GuardianEngine):
        super().__init__(parent, padding=8)
        self.mode_name = mode_name
        self.engine = engine

        self["borderwidth"] = 2
        self["relief"] = "groove"

        self.title_label = ttk.Label(self, text=mode_name, anchor="center")
        self.summary_label = ttk.Label(self, text="Initializing...", wraplength=260, anchor="w", justify="left")
        self.health_label = ttk.Label(self, text="Health: --", anchor="w")
        self.foresight_label = ttk.Label(self, text="Foresight: --", anchor="w")
        self.details_button = ttk.Button(self, text="Details", command=self.show_details)

        self.title_label.grid(row=0, column=0, sticky="ew")
        self.summary_label.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        self.health_label.grid(row=2, column=0, sticky="ew")
        self.foresight_label.grid(row=3, column=0, sticky="ew")
        self.details_button.grid(row=4, column=0, sticky="e", pady=(4, 0))

        self.columnconfigure(0, weight=1)

        self._current_color = COLOR_MAP["grey"]
        self._set_bg(self._current_color)

    def _set_bg(self, color: str):
        self._current_color = color
        style_name = f"{self.mode_name}.TFrame"
        label_style = f"{self.mode_name}.TLabel"
        style = ttk.Style()
        style.configure(style_name, background=color)
        style.configure(label_style, background=color)

        self.configure(style=style_name)
        for lbl in (self.title_label, self.summary_label, self.health_label, self.foresight_label):
            lbl.configure(style=label_style)

    def refresh(self):
        status = self.engine.get_mode_status(self.mode_name)
        brain = status.brain

        color = COLOR_MAP.get(brain.level, COLOR_MAP["grey"])
        self._set_bg(color)

        self.summary_label.configure(text=brain.best_guess)
        self.health_label.configure(
            text=(
                f"Health: {brain.collective_health_score:.0f} | "
                f"J:{brain.judgment:.0f} C:{brain.confidence:.0f} "
                f"SA:{brain.situational_awareness:.0f} PI:{brain.predictive_intelligence:.0f}"
            )
        )
        self.foresight_label.configure(text=f"Foresight: {brain.foresight_score:.0f}")

    def show_details(self):
        ModeDetailsWindow(self, self.engine, self.mode_name)


class DriveStatusFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, padding=8)
        self.engine = engine

        self["borderwidth"] = 2
        self["relief"] = "groove"

        self.title_label = ttk.Label(self, text="Network Drives", anchor="center")
        self.primary_var = tk.StringVar(value=engine.get_primary_drive() or "Primary: Not selected")
        self.fallback_var = tk.StringVar(value=engine.get_fallback_drive() or "Fallback: Not selected")
        self.status_var = tk.StringVar(value="Status: Unknown")

        self.primary_label = ttk.Label(self, textvariable=self.primary_var)
        self.fallback_label = ttk.Label(self, textvariable=self.fallback_var)
        self.status_label = ttk.Label(self, textvariable=self.status_var)

        self.btn_primary = ttk.Button(self, text="Select Primary", command=self._select_primary)
        self.btn_fallback = ttk.Button(self, text="Select Fallback", command=self._select_fallback)
        self.btn_test = ttk.Button(self, text="Test Connection", command=self._test_drives)

        self.title_label.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.primary_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.btn_primary.grid(row=1, column=2, sticky="e", padx=4)

        self.fallback_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.btn_fallback.grid(row=2, column=2, sticky="e", padx=4)

        self.status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.btn_test.grid(row=3, column=2, sticky="e", padx=4)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)

        self._current_color = COLOR_MAP["grey"]
        self._apply_color()

    def _apply_color(self):
        level = self.engine.memory.get(DRIVE_STATUS_KEY, "grey")
        color = COLOR_MAP.get(level, COLOR_MAP["grey"])
        self._current_color = color

        style_name = "DriveStatus.TFrame"
        label_style = "DriveStatus.TLabel"
        style = ttk.Style()
        style.configure(style_name, background=color)
        style.configure(label_style, background=color)

        self.configure(style=style_name)
        for lbl in (self.title_label, self.primary_label, self.fallback_label, self.status_label):
            lbl.configure(style=label_style)

    def _select_primary(self):
        path = filedialog.askdirectory(title="Select Primary SMB Network Drive", mustexist=True)
        if path:
            self.engine.set_primary_drive(path)
            self.primary_var.set(f"Primary: {path}")

    def _select_fallback(self):
        path = filedialog.askdirectory(title="Select Fallback SMB Network Drive", mustexist=True)
        if path:
            self.engine.set_fallback_drive(path)
            self.fallback_var.set(f"Fallback: {path}")

    def _test_drives(self):
        primary = self.engine.get_primary_drive()
        fallback = self.engine.get_fallback_drive()

        primary_ok = self.engine.test_drive(primary)
        fallback_ok = self.engine.test_drive(fallback)

        if primary_ok or fallback_ok:
            level = "green" if (primary_ok and fallback_ok) else "yellow"
            msg = "Primary and fallback online" if level == "green" else "One drive online, one offline"
        else:
            level = "red"
            msg = "Both drives offline or not set"

        self.engine.memory.set(DRIVE_STATUS_KEY, level)
        self.status_var.set(f"Status: {msg}")
        self._apply_color()
        messagebox.showinfo("Drive Test", msg)

    def refresh(self):
        level = self.engine.memory.get(DRIVE_STATUS_KEY, "grey")
        if level == "green":
            txt = "Status: Both drives healthy or at least one online"
        elif level == "yellow":
            txt = "Status: Partial availability – using available drive"
        elif level == "red":
            txt = "Status: No online drive – using local fallback only"
        else:
            txt = "Status: Unknown"
        self.status_var.set(txt)
        self._apply_color()


class AlteredStatesFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, padding=8)
        self.engine = engine
        self["borderwidth"] = 2
        self["relief"] = "groove"

        ttk.Label(self, text="Altered States (Sensitivity)", anchor="center").grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4)
        )

        row = 1
        for state in CONSCIOUSNESS_STATES:
            factor = SENSITIVITY_FACTORS[state]
            ttk.Label(self, text=state + ":").grid(row=row, column=0, sticky="w")
            ttk.Label(
                self,
                text=f"Sensitivity x{factor:.1f}"
            ).grid(row=row, column=1, sticky="w")
            row += 1

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)


class GuardianApp(tk.Tk):
    def __init__(self, engine: GuardianEngine):
        super().__init__()
        self.engine = engine

        self.title("Fused Guardian – Hybrid Brain with Capsule Cortex")
        self.geometry("1450x820")

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        top_frame = ttk.Frame(container)
        top_frame.pack(fill="x", pady=(0, 10))

        self.global_label = ttk.Label(top_frame, text="Global: Initializing...", font=("Segoe UI", 12, "bold"))
        self.global_label.pack(side="left")

        self.state_var = tk.StringVar(value=self.engine.consciousness_state)
        state_menu = ttk.OptionMenu(
            top_frame,
            self.state_var,
            self.engine.consciousness_state,
            *CONSCIOUSNESS_STATES,
            command=self._on_state_change,
        )
        ttk.Label(top_frame, text="State:").pack(side="right")
        state_menu.pack(side="right")

        middle_frame = ttk.Frame(container)
        middle_frame.pack(fill="both", expand=True)

        modes_frame = ttk.Frame(middle_frame)
        modes_frame.pack(side="left", fill="both", expand=True)

        right_column = ttk.Frame(middle_frame)
        right_column.pack(side="right", fill="y", padx=(10, 0))

        self.drive_frame = DriveStatusFrame(right_column, self.engine)
        self.drive_frame.pack(fill="x", pady=(0, 10))

        self.states_frame = AlteredStatesFrame(right_column, self.engine)
        self.states_frame.pack(fill="x")

        self.engine_state_label = ttk.Label(
            right_column,
            text="Capsule Engine: INACTIVE",
            font=("Segoe UI", 10, "bold")
        )
        self.engine_state_label.pack(fill="x", pady=(10, 5))

        btn_frame = ttk.Frame(right_column)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Activate Capsule Engine", command=self._activate_capsule_engine).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Deactivate Capsule Engine", command=self._deactivate_capsule_engine).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Toggle Self-Destruct", command=self._toggle_sd).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Toggle Visibility", command=self._toggle_vis).pack(fill="x", pady=2)

        self.mode_frames: Dict[str, ModeFrame] = {}
        mode_names = list(self.engine.modes.keys())

        rows = 3
        cols = 3
        idx = 0
        for r in range(rows):
            modes_frame.rowconfigure(r, weight=1)
            for c in range(cols):
                modes_frame.columnconfigure(c, weight=1)
                if idx < len(mode_names):
                    name = mode_names[idx]
                    frame = ModeFrame(modes_frame, name, self.engine)
                    frame.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
                    self.mode_frames[name] = frame
                    idx += 1

        self.after(1000, self.refresh)

    def _on_state_change(self, value):
        self.engine.set_consciousness_state(value)

    def _activate_capsule_engine(self):
        self.engine.capsule_state.set_active(True)
        self.engine_state_label.configure(text="Capsule Engine: ACTIVE")

    def _deactivate_capsule_engine(self):
        self.engine.capsule_state.set_active(False)
        self.engine_state_label.configure(text="Capsule Engine: INACTIVE")

    def _toggle_sd(self):
        val = self.engine.capsule_state.toggle_self_destruct()
        messagebox.showinfo("Self-Destruct", f"Self-destruct is now {'ON' if val else 'OFF'}")

    def _toggle_vis(self):
        val = self.engine.capsule_state.toggle_visibility()
        messagebox.showinfo("Visibility", f"Visibility is now {'ON' if val else 'OFF'}")

    def refresh(self):
        for frame in self.mode_frames.values():
            frame.refresh()

        self.drive_frame.refresh()

        worst_level = "green"
        worst_health = 100.0
        level_order = {"green": 3, "yellow": 2, "red": 1}
        for name in self.mode_frames:
            status = self.engine.get_mode_status(name)
            level = status.brain.level
            health = status.brain.collective_health_score
            if level_order[level] < level_order.get(worst_level, 3):
                worst_level = level
                worst_health = health
            elif level == worst_level and health < worst_health:
                worst_health = health

        self.global_label.configure(
            text=f"Global: {worst_level.upper()} | Health: {worst_health:.0f} | State: {self.engine.consciousness_state}"
        )

        self.after(2000, self.refresh)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    engine = GuardianEngine()
    engine.init_baselines()
    engine.start_loop(interval_sec=15)

    app_flask = create_flask_app(engine)
    flask_thread = threading.Thread(target=lambda: app_flask.run(port=6666, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()

    gui = GuardianApp(engine)
    gui.mainloop()
    engine.stop()


if __name__ == "__main__":
    main()

