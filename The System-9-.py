#!/usr/bin/env python3
"""
Fused Guardian – Evolution Tier + Game Cortex + Assistant Brain + Storage Tab + HUD

Single unified file including:
- Multi-horizon baselines (fast/short/mid/day/month)
- Meta-cognition (SelfEvalEngine)
- WorldModel (daily/monthly drift)
- Predictive physics (DataPhysicsEngine)
- Capsule Engine (state + metrics + daemon)
- Game Detection Engine (Steam/Epic)
- Game Baseline & Foresight Engine
- GAME_MODE consciousness
- Tkinter GUI (Dashboard + Game Cortex Tactical HUD + Security Lists + Storage)
- Flask API with /status and /prometheus_map
- EXE-based Whitelist & Blacklist (no auto-kill except explicit blacklist)
- Manual tab switching (no auto-switch to Game Cortex)
- AssistantBrain (system-wide reasoning, briefings, suggestions)
- DiagnosticOverlay (floating HUD)
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
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from flask import Flask, request, jsonify
from cryptography.fernet import Fernet

try:
    import winreg
except ImportError:
    winreg = None

# ---------------------------------------------------------------------------
# Constants, colors, modes
# ---------------------------------------------------------------------------

COLOR_MAP = {
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "red": "#e74c3c",
    "grey": "#bdc3c7",
}

CONSCIOUSNESS_STATES = ["CALM", "FOCUSED", "ALERT", "PARANOID"]
SENSITIVITY_FACTORS: Dict[str, float] = {
    "CALM": 0.7,
    "FOCUSED": 1.0,
    "ALERT": 1.3,
    "PARANOID": 1.6,
}

GAME_MODE_STATE = "GAME_MODE"
if GAME_MODE_STATE not in CONSCIOUSNESS_STATES:
    CONSCIOUSNESS_STATES.append(GAME_MODE_STATE)
SENSITIVITY_FACTORS.setdefault(GAME_MODE_STATE, 0.6)

FAST_BASELINE_WINDOW = 5 * 60
SHORT_BASELINE_WINDOW = 60 * 60
MID_BASELINE_WINDOW = 4 * 60 * 60
DAY_BASELINE_WINDOW = 24 * 60 * 60
MONTH_BASELINE_WINDOW = 30 * 24 * 60 * 60

PRIMARY_DRIVE_KEY = "network_drive_primary"
FALLBACK_DRIVE_KEY = "network_drive_fallback"
DRIVE_STATUS_KEY = "network_drive_status"
STORAGE_LOCATION_KEY = "storage_location"
STORAGE_SMB_PATH_KEY = "storage_smb_path"

EVENT_HISTORY_PREFIX = "events:"
CAPSULE_STATE_KEY = "capsule_state"
CAPSULE_METRICS_KEY = "capsule_metrics"
CONSCIOUSNESS_STATE_KEY = "consciousness_state"
NODE_ID_KEY = "node_id"
NODE_ENTROPY_KEY = "node_entropy"
BORG_BASELINE_KEY = "borg_baseline_addresses"
SOFTWARE_INSTALLED_NAMES_KEY = "software_installed_names"
SOFTWARE_UNREGISTERED_PROCESSES_KEY = "software_unregistered_processes"

BASELINE_MULTI_PREFIX = "baseline_multi:"
SELF_EVAL_KEY = "self_eval"
WORLD_MODEL_KEY = "world_model"

# Whitelist / Blacklist keys (EXE-based)
WHITELIST_KEY = "exe_whitelist"
BLACKLIST_KEY = "exe_blacklist"

GAME_PARENT_WHITELIST = [
    "steam.exe",
    "steamservice.exe",
    "epicgameslauncher.exe",
    "epicwebhelper.exe",
]

# Default implicit safe roots (Windows + typical game locations)
SAFE_PATH_ROOTS = [
    r"c:\windows",
    r"c:\program files",
    r"c:\program files (x86)",
    r"c:\programdata",
]

# ---------------------------------------------------------------------------
# Dataclasses
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
    uncertainty: float = 1.0
    collapse_probability: float = 0.0
    best_guess: str = "Unknown – warming up"


@dataclass
class ModeStatus:
    brain: BrainState = field(default_factory=BrainState)
    raw_details: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Encrypted storage + memory backend
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


class MemoryBackend:
    def __init__(self, persistent_storage: Optional[EncryptedStorage] = None):
        self._store: Dict[str, Any] = {}
        self.persistent_storage = persistent_storage
        if self.persistent_storage:
            self._load_persistent()
        self._ensure_default_lists()

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

    def _ensure_default_lists(self):
        wl = self._store.get(WHITELIST_KEY)
        bl = self._store.get(BLACKLIST_KEY)
        if wl is None:
            self._store[WHITELIST_KEY] = []
        if bl is None:
            self._store[BLACKLIST_KEY] = []
        self._flush_persistent()

    # Convenience helpers for whitelist / blacklist
    def get_whitelist(self) -> List[str]:
        return list(self._store.get(WHITELIST_KEY, []))

    def get_blacklist(self) -> List[str]:
        return list(self._store.get(BLACKLIST_KEY, []))

    def add_to_whitelist(self, exe_path: str):
        exe_path = exe_path.lower()
        wl = self.get_whitelist()
        if exe_path not in wl:
            wl.append(exe_path)
            self._store[WHITELIST_KEY] = wl
            self._flush_persistent()

    def remove_from_whitelist(self, exe_path: str):
        exe_path = exe_path.lower()
        wl = self.get_whitelist()
        if exe_path in wl:
            wl.remove(exe_path)
            self._store[WHITELIST_KEY] = wl
            self._flush_persistent()

    def add_to_blacklist(self, exe_path: str):
        exe_path = exe_path.lower()
        bl = self.get_blacklist()
        if exe_path not in bl:
            bl.append(exe_path)
            self._store[BLACKLIST_KEY] = bl
            self._flush_persistent()

    def remove_from_blacklist(self, exe_path: str):
        exe_path = exe_path.lower()
        bl = self.get_blacklist()
        if exe_path in bl:
            bl.remove(exe_path)
            self._store[BLACKLIST_KEY] = bl
            self._flush_persistent()

# ---------------------------------------------------------------------------
# Baselines, history, profiles, events
# ---------------------------------------------------------------------------

class BaselineManager:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory

    def set_baseline(self, name: str, data: Any) -> None:
        self.memory.set(f"baseline:{name}", data)

    def get_baseline(self, name: str) -> Optional[Any]:
        return self.memory.get(f"baseline:{name}")

    def _multi_key(self, mode: str) -> str:
        return f"{BASELINE_MULTI_PREFIX}{mode}"

    def get_multi_baselines(self, mode: str) -> Dict[str, Dict[str, Any]]:
        return self.memory.get(self._multi_key(mode), {})

    def set_multi_baseline(self, mode: str, horizon: str, snapshot: Dict[str, Any]) -> None:
        data = self.get_multi_baselines(mode)
        data[horizon] = {"snapshot": snapshot, "time": time.time()}
        self.memory.set(self._multi_key(mode), data)

    def maybe_refresh_multi_baselines(self, mode: str, snapshot: Dict[str, Any]) -> None:
        now = time.time()
        data = self.get_multi_baselines(mode)
        horizons = {
            "fast_5min": FAST_BASELINE_WINDOW,
            "short_1h": SHORT_BASELINE_WINDOW,
            "mid_4h": MID_BASELINE_WINDOW,
            "day": DAY_BASELINE_WINDOW,
            "month": MONTH_BASELINE_WINDOW,
        }
        changed = False
        for h_name, window in horizons.items():
            entry = data.get(h_name)
            if entry is None or now - entry["time"] >= window:
                data[h_name] = {"snapshot": snapshot, "time": now}
                changed = True
        if changed:
            self.memory.set(self._multi_key(mode), data)

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
# Self-eval + WorldModel
# ---------------------------------------------------------------------------

class SelfEvalEngine:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory

    def _get_state(self) -> Dict[str, Any]:
        return self.memory.get(SELF_EVAL_KEY, {})

    def _save_state(self, data: Dict[str, Any]):
        self.memory.set(SELF_EVAL_KEY, data)

    def update(
        self,
        mode: str,
        previous_health: float,
        new_health: float,
        previous_collapse_prob: float,
        new_anomaly_count: int,
    ):
        data = self._get_state()
        m = data.get(mode, {"over_reacts": 0, "under_reacts": 0, "samples": 0, "aggressiveness": 1.0})

        predicted_risk = previous_collapse_prob
        actual_worse = new_health < previous_health - 10 or new_anomaly_count > 0

        if predicted_risk > 0.7 and not actual_worse:
            m["over_reacts"] += 1
        if predicted_risk < 0.3 and actual_worse:
            m["under_reacts"] += 1

        m["samples"] += 1

        if m["samples"] >= 10:
            over = m["over_reacts"]
            under = m["under_reacts"]
            if over > under * 2:
                m["aggressiveness"] = max(0.5, m["aggressiveness"] - 0.05)
            elif under > over * 2:
                m["aggressiveness"] = min(1.5, m["aggressiveness"] + 0.05)
            m["samples"] = 0
            m["over_reacts"] = 0
            m["under_reacts"] = 0

        data[mode] = m
        self._save_state(data)

    def get_aggressiveness(self, mode: str) -> float:
        data = self._get_state()
        return float(data.get(mode, {}).get("aggressiveness", 1.0))


class WorldModel:
    def __init__(self, memory: MemoryBackend, baseline_manager: BaselineManager):
        self.memory = memory
        self.baseline_manager = baseline_manager

    def _get_state(self) -> Dict[str, Any]:
        return self.memory.get(WORLD_MODEL_KEY, {})

    def _save_state(self, data: Dict[str, Any]):
        self.memory.set(WORLD_MODEL_KEY, data)

    def update_for_mode(self, mode: str, snapshot: Dict[str, Any]):
        state = self._get_state()
        mode_drifts = state.get("mode_drifts", {})

        multi = self.baseline_manager.get_multi_baselines(mode)
        day_entry = multi.get("day", {})
        month_entry = multi.get("month", {})
        day_snap = day_entry.get("snapshot")
        month_snap = month_entry.get("snapshot")

        def metric_diff(base, cur):
            if not base:
                return 0.0
            d = 0.0
            if "process_count" in cur and "process_count" in base:
                b = base["process_count"]
                c = cur["process_count"]
                if b > 0:
                    d += abs(c - b) / b
            if "connection_count" in cur and "connection_count" in base:
                b = base["connection_count"]
                c = cur["connection_count"]
                if b > 0:
                    d += abs(c - b) / b
            return d

        day_drift = metric_diff(day_snap, snapshot)
        month_drift = metric_diff(month_snap, snapshot)

        mode_drifts[mode] = {
            "day_drift": float(day_drift),
            "month_drift": float(month_drift),
        }

        state["mode_drifts"] = mode_drifts
        state["last_update"] = time.time()
        self._save_state(state)

    def get_mode_drift(self, mode: str) -> Dict[str, float]:
        state = self._get_state()
        return state.get("mode_drifts", {}).get(mode, {"day_drift": 0.0, "month_drift": 0.0})

# ---------------------------------------------------------------------------
# Inference + RiskModel
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory

    def infer(self, mode_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        deviation = float(features.get("deviation", 0.0))
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
# Game detection + baselines + foresight + fusion helpers
# ---------------------------------------------------------------------------

class GameDetectionEngine:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory
        self.current_game = None
        self.last_detection_time = 0

    def detect_game(self) -> Optional[Dict[str, Any]]:
        try:
            for p in psutil.process_iter(["pid", "name", "exe", "ppid"]):
                name = (p.info.get("name") or "").lower()
                pid = p.info.get("pid")
                ppid = p.info.get("ppid")

                try:
                    parent = psutil.Process(ppid)
                    parent_name = (parent.name() or "").lower()
                except Exception:
                    parent_name = ""

                if parent_name in GAME_PARENT_WHITELIST:
                    game_info = {
                        "is_game": True,
                        "exe": name,
                        "title": self._infer_title_from_exe(name),
                        "platform": "Steam" if "steam" in parent_name else "Epic",
                        "pid": pid,
                        "parent": parent_name,
                        "anti_cheat": self._detect_anti_cheat(pid),
                    }
                    self.current_game = game_info
                    self.last_detection_time = time.time()
                    self.memory.set("current_game_info", game_info)
                    return game_info

        except Exception:
            pass

        self.current_game = None
        self.memory.set("current_game_info", None)
        return None

    def _infer_title_from_exe(self, exe: str) -> str:
        exe = exe.replace(".exe", "")
        exe = exe.replace("_", " ").replace("-", " ")
        return exe.title()

    def _detect_anti_cheat(self, pid: int) -> str:
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                cname = (child.name() or "").lower()
                if "cheat" in cname or "eac" in cname or "battleye" in cname:
                    return "Active"
        except Exception:
            pass
        return "Normal"

    def get_current_game(self):
        return self.memory.get("current_game_info", None)


class GameBaselineManager:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory

    def _key(self, title: str) -> str:
        return f"game_baseline:{title}"

    def update_baseline(self, game_title: str, snapshot: Dict[str, Any]):
        key = self._key(game_title)
        data = self.memory.get(key, {})
        now = time.time()
        horizons = {
            "fast_5min": FAST_BASELINE_WINDOW,
            "short_1h": SHORT_BASELINE_WINDOW,
            "mid_4h": MID_BASELINE_WINDOW,
            "day": DAY_BASELINE_WINDOW,
            "month": MONTH_BASELINE_WINDOW,
        }
        changed = False
        for h_name, window in horizons.items():
            entry = data.get(h_name)
            if entry is None or now - entry["time"] >= window:
                data[h_name] = {"snapshot": snapshot, "time": now}
                changed = True
        if changed:
            self.memory.set(key, data)

    def get_baselines(self, game_title: str) -> Dict[str, Any]:
        return self.memory.get(self._key(game_title), {})


class GameForesightEngine:
    def __init__(self, memory: MemoryBackend, baseline_manager: GameBaselineManager):
        self.memory = memory
        self.baseline_manager = baseline_manager

    def predict(self, game_title: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        baselines = self.baseline_manager.get_baselines(game_title)
        cpu = snapshot.get("cpu", 0)
        ram = snapshot.get("ram", 0)
        conns = snapshot.get("connections", 0)

        drift = 0.0
        for entry in baselines.values():
            base = entry.get("snapshot", {})
            if "cpu" in base and base["cpu"] > 0:
                drift += abs(cpu - base["cpu"]) / base["cpu"]
            if "ram" in base and base["ram"] > 0:
                drift += abs(ram - base["ram"]) / base["ram"]
            if "connections" in base and base["connections"] > 0:
                drift += abs(conns - base["connections"]) / base["connections"]

        if baselines:
            drift /= len(baselines)

        if drift < 0.2:
            foresight = "Stable performance expected"
        elif drift < 0.5:
            foresight = "Minor performance fluctuations likely"
        else:
            foresight = "Performance instability predicted"

        return {"drift": drift, "foresight": foresight}


def fuse_game_into_snapshot(snapshot: Dict[str, Any], game_info: Optional[Dict[str, Any]]):
    if not game_info:
        snapshot["is_game_process"] = False
        return snapshot
    exe = (snapshot.get("exe") or "").lower()
    if exe == (game_info.get("exe") or "").lower():
        snapshot["is_game_process"] = True
        snapshot["game_title"] = game_info.get("title")
        snapshot["game_platform"] = game_info.get("platform")
        snapshot["game_pid"] = game_info.get("pid")
        snapshot["game_parent"] = game_info.get("parent")
        snapshot["anti_cheat"] = game_info.get("anti_cheat")
    else:
        snapshot["is_game_process"] = False
    return snapshot


def normalize_risk_for_game(snapshot: Dict[str, Any], risk: float) -> float:
    if snapshot.get("is_game_process"):
        return risk * 0.35
    return risk


def normalize_anomalies_for_game(snapshot: Dict[str, Any], anomalies: int) -> int:
    if snapshot.get("is_game_process"):
        return max(0, anomalies - 5)
    return anomalies


def fuse_game_foresight_into_snapshot(snapshot: Dict[str, Any],
                                      game_foresight: Optional[Dict[str, Any]]):
    if snapshot.get("is_game_process") and game_foresight:
        snapshot["game_drift"] = game_foresight.get("drift", 0.0)
        snapshot["game_foresight"] = game_foresight.get("foresight", "Unknown")
    return snapshot

# ---------------------------------------------------------------------------
# DataPhysicsEngine
# ---------------------------------------------------------------------------

class DataPhysicsEngine:
    def __init__(
        self,
        baseline_manager: BaselineManager,
        inference_engine: InferenceEngine,
        risk_model: RiskModel,
        self_eval: SelfEvalEngine,
        world_model: WorldModel,
    ):
        self.baseline_manager = baseline_manager
        self.inference_engine = inference_engine
        self.risk_model = risk_model
        self.self_eval = self_eval
        self.world_model = world_model
        self.memory = baseline_manager.memory

    def compute_brain_state(
        self,
        mode_name: str,
        snapshot: Dict[str, Any],
        consciousness_state: str,
        uptime: float,
        cross_mode_levels: Dict[str, str],
        previous_brain: Optional[BrainState],
    ) -> ModeStatus:

        self.baseline_manager.maybe_refresh_multi_baselines(mode_name, snapshot)
        multi = self.baseline_manager.get_multi_baselines(mode_name)

        deviation = 0.0
        count = 0
        for entry in multi.values():
            base = entry.get("snapshot", {})
            if "process_count" in base and "process_count" in snapshot:
                b = base["process_count"]
                c = snapshot["process_count"]
                if b > 0:
                    deviation += abs(c - b) / b
                    count += 1
            if "connection_count" in base and "connection_count" in snapshot:
                b = base["connection_count"]
                c = snapshot["connection_count"]
                if b > 0:
                    deviation += abs(c - b) / b
                    count += 1
        if count > 0:
            deviation /= count

        anomaly_count = int(snapshot.get("anomaly_count", 0))

        baseline_age = 0.0
        if "fast_5min" in multi:
            baseline_age = time.time() - multi["fast_5min"]["time"]

        self.world_model.update_for_mode(mode_name, snapshot)
        drift_info = self.world_model.get_mode_drift(mode_name)
        day_drift = drift_info.get("day_drift", 0.0)
        month_drift = drift_info.get("month_drift", 0.0)

        predictive_intelligence = max(0.0, 100.0 - (deviation * 50.0 + anomaly_count * 2.0))
        situational_awareness = max(0.0, 100.0 - (day_drift * 40.0 + month_drift * 20.0))

        features = {
            "deviation": deviation,
            "anomaly_count": anomaly_count,
            "baseline_age": baseline_age,
            "predictive_intelligence": predictive_intelligence,
            "situational_awareness": situational_awareness,
            "capsule_risk": snapshot.get("capsule_risk", 0.0),
        }
        risk_score = self.risk_model.score(features)

        collapse_probability = min(1.0, risk_score / 100.0)
        judgment = max(0.0, 100.0 - risk_score)
        confidence = max(0.0, 100.0 - deviation * 50.0)

        if risk_score < 30:
            level = "green"
        elif risk_score < 60:
            level = "yellow"
        else:
            level = "red"

        sensitivity = SENSITIVITY_FACTORS.get(consciousness_state, 1.0)
        deviation *= sensitivity
        anomaly_count = int(anomaly_count * sensitivity)

        game_info = self.memory.get("current_game_info", None)
        if game_info and snapshot.get("is_game_process"):
            level = "green"
            judgment = min(100.0, judgment + 10.0)
            confidence = min(100.0, confidence + 10.0)
            predictive_intelligence = max(predictive_intelligence, 70.0)
            collapse_probability = min(collapse_probability, 0.2)
            best_guess = f"Game '{snapshot.get('game_title', 'Unknown')}' running – normalized"
        else:
            best_guess = f"Mode '{mode_name}' stable" if level == "green" else f"Mode '{mode_name}' under load"

        brain = BrainState(
            level=level,
            judgment=judgment,
            confidence=confidence,
            situational_awareness=situational_awareness,
            predictive_intelligence=predictive_intelligence,
            collective_health_score=max(0.0, 100.0 - risk_score),
            foresight_score=max(0.0, 100.0 - collapse_probability * 100.0),
            uncertainty=max(0.0, min(1.0, deviation)),
            collapse_probability=collapse_probability,
            best_guess=best_guess,
        )

        if previous_brain is not None:
            self.self_eval.update(
                mode=mode_name,
                previous_health=previous_brain.collective_health_score,
                new_health=brain.collective_health_score,
                previous_collapse_prob=previous_brain.collapse_probability,
                new_anomaly_count=anomaly_count,
            )

        return ModeStatus(
            brain=brain,
            raw_details={
                "deviation": deviation,
                "anomaly_count": anomaly_count,
                "risk_score": risk_score,
                "baseline_age": baseline_age,
                "day_drift": day_drift,
                "month_drift": month_drift,
            },
        )

# ---------------------------------------------------------------------------
# Capsule / Node / Swarm minimal implementations + all modes
# ---------------------------------------------------------------------------

class CapsuleState:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory
        state = self.memory.get(CAPSULE_STATE_KEY, None)
        if not state:
            state = {
                "active": True,
                "self_destruct": False,
                "visible": True,
            }
            self.memory.set(CAPSULE_STATE_KEY, state)
        self.state = state

    def set_active(self, active: bool):
        self.state["active"] = active
        self.memory.set(CAPSULE_STATE_KEY, self.state)

    def toggle_self_destruct(self) -> bool:
        self.state["self_destruct"] = not self.state["self_destruct"]
        self.memory.set(CAPSULE_STATE_KEY, self.state)
        return self.state["self_destruct"]

    def toggle_visibility(self) -> bool:
        self.state["visible"] = not self.state["visible"]
        self.memory.set(CAPSULE_STATE_KEY, self.state)
        return self.state["visible"]


class CapsuleMetrics:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory
        m = self.memory.get(CAPSULE_METRICS_KEY, None)
        if not m:
            m = {
                "total_capsules": 0,
                "unauthorized_attempts": 0,
                "flop_sum": 0.0,
                "entropy": 0.0,
                "api_calls": 0,
                "self_destruct_toggles": 0,
                "visibility_toggles": 0,
            }
            self.memory.set(CAPSULE_METRICS_KEY, m)
        self.metrics = m

    def snapshot(self) -> Dict[str, Any]:
        return self.memory.get(CAPSULE_METRICS_KEY, self.metrics)

    def set_entropy(self, value: float):
        m = self.snapshot()
        m["entropy"] = value
        self.memory.set(CAPSULE_METRICS_KEY, m)


class Node:
    def __init__(self, memory: MemoryBackend):
        self.memory = memory
        node_id = self.memory.get(NODE_ID_KEY, None)
        if not node_id:
            node_id = hashlib.sha256(os.urandom(16)).hexdigest()[:12]
            self.memory.set(NODE_ID_KEY, node_id)
        self.id = node_id
        self.entropy = float(self.memory.get(NODE_ENTROPY_KEY, 0.5))

    def mutate(self):
        self.entropy = max(0.0, min(1.0, self.entropy + random.uniform(-0.02, 0.02)))
        self.memory.set(NODE_ENTROPY_KEY, self.entropy)


class Swarm:
    def sync(self):
        pass


class CapsuleDaemon:
    def __init__(self, capsule_state: CapsuleState, capsule_metrics: CapsuleMetrics):
        self.capsule_state = capsule_state
        self.capsule_metrics = capsule_metrics

    def run_cycle(self):
        if not self.capsule_state.state["active"]:
            return
        m = self.capsule_metrics.snapshot()
        m["total_capsules"] = m.get("total_capsules", 0) + 1
        m["api_calls"] = m.get("api_calls", 0) + random.randint(0, 3)
        self.capsule_metrics.memory.set(CAPSULE_METRICS_KEY, m)

# ---------------------------------------------------------------------------
# BaseMode + whitelist/blacklist helpers
# ---------------------------------------------------------------------------

class BaseMode:
    def __init__(
        self,
        name: str,
        baseline_manager: BaselineManager,
        memory: MemoryBackend,
        physics_engine: DataPhysicsEngine,
        uptime_provider,
    ):
        self.name = name
        self.baseline_manager = baseline_manager
        self.memory = memory
        self.physics_engine = physics_engine
        self.uptime_provider = uptime_provider
        self._last_brain: Optional[BrainState] = None

    def init_baseline(self):
        snap = self._capture_snapshot()
        self.baseline_manager.set_baseline(self.name, snap)
        self.baseline_manager.maybe_refresh_multi_baselines(self.name, snap)

    def update_state(self, consciousness_state: str, cross_mode_levels: Dict[str, str]):
        snap = self._capture_snapshot()

        game_info = self.memory.get("current_game_info", None)
        snap = fuse_game_into_snapshot(snap, game_info)
        snap["anomaly_count"] = normalize_anomalies_for_game(snap, snap.get("anomaly_count", 0))

        self.baseline_manager.add_history(self.name, snap)

        uptime = self.uptime_provider()
        status = self.physics_engine.compute_brain_state(
            self.name,
            snap,
            consciousness_state,
            uptime,
            cross_mode_levels,
            self._last_brain,
        )
        self._last_brain = status.brain
        return status

    def get_status(self) -> ModeStatus:
        if self._last_brain is None:
            return ModeStatus()
        return ModeStatus(self._last_brain, {})

    def _capture_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError


def is_exe_whitelisted(memory: MemoryBackend, exe_path: str) -> bool:
    if not exe_path:
        return False
    exe_path = exe_path.lower()
    if exe_path in memory.get_whitelist():
        return True
    for root in SAFE_PATH_ROOTS:
        if exe_path.startswith(root):
            return True
    return False


def is_exe_blacklisted(memory: MemoryBackend, exe_path: str) -> bool:
    if not exe_path:
        return False
    exe_path = exe_path.lower()
    return exe_path in memory.get_blacklist()

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

class PresidentialMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        procs = list(psutil.process_iter(["pid"]))
        process_count = len(procs)

        try:
            conns = psutil.net_connections(kind="inet")
            conn_count = len(conns)
        except Exception:
            conn_count = 0

        anomaly_count = max(0, process_count - 200)

        return {
            "mode_name": "Presidential",
            "process_count": process_count,
            "connection_count": conn_count,
            "anomaly_count": anomaly_count,
            "exe": "",
        }


class BloodhoundMode(BaseMode):
    @staticmethod
    def _is_private_ip(ip: str) -> bool:
        return (
            ip.startswith("10.") or
            ip.startswith("192.168.") or
            (ip.startswith("172.") and any(ip.startswith(f"172.{i}.") for i in range(16, 32))) or
            ip.startswith("127.")
        )

    def _capture_snapshot(self) -> Dict[str, Any]:
        suspicious = 0
        total = 0

        unreg_list = self.memory.get(SOFTWARE_UNREGISTERED_PROCESSES_KEY, [])
        unreg_pids = {item.get("pid") for item in unreg_list if "pid" in item}

        pid_to_proc = {}
        try:
            for p in psutil.process_iter(["pid", "name", "exe"]):
                pid_to_proc[p.info["pid"]] = p
        except Exception:
            pass

        game_info = self.memory.get("current_game_info", None)
        game_pid = game_info.get("pid") if game_info else None

        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            conns = []

        for c in conns:
            if not c.raddr:
                continue
            ip = c.raddr.ip
            total += 1
            pid = c.pid

            if game_pid is not None and pid == game_pid:
                continue

            proc = pid_to_proc.get(pid)
            exe_path = (proc.info.get("exe") or "").lower() if proc else ""

            if is_exe_whitelisted(self.memory, exe_path):
                continue

            if is_exe_blacklisted(self.memory, exe_path):
                suspicious += 5
                continue

            if not self._is_private_ip(ip):
                suspicious += 1

        anomaly_count = suspicious

        return {
            "mode_name": "Bloodhound",
            "connection_count": total,
            "anomaly_count": anomaly_count,
            "high_risk_terminated": 0,
            "exe": "",
        }


class GodMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        try:
            procs = list(psutil.process_iter(["pid"]))
            process_count = len(procs)
        except Exception:
            process_count = 0

        try:
            conns = psutil.net_connections(kind="inet")
            conn_count = len(conns)
        except Exception:
            conn_count = 0

        anomaly_count = max(0, process_count - 300)

        return {
            "mode_name": "God",
            "process_count": process_count,
            "connection_count": conn_count,
            "anomaly_count": anomaly_count,
            "exe": "",
        }


class SoftwareInterrogatorMode(BaseMode):
    @staticmethod
    def _get_installed_software_names() -> List[str]:
        names = []
        if winreg is None:
            return names
        paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for path in paths:
                try:
                    with winreg.OpenKey(root, path) as key:
                        for i in range(0, winreg.QueryInfoKey(key)[0]):
                            try:
                                sub = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, sub) as sk:
                                    name, _ = winreg.QueryValueEx(sk, "DisplayName")
                                    if name:
                                        names.append(name.lower())
                            except Exception:
                                pass
                except Exception:
                    pass
        return names

    @staticmethod
    def _is_under_windows_dirs(path: Optional[str]) -> bool:
        if not path:
            return False
        p = path.lower()
        return (
            p.startswith("c:\\windows") or
            p.startswith("c:\\program files") or
            p.startswith("c:\\program files (x86)")
        )

    def _capture_snapshot(self) -> Dict[str, Any]:
        installed_names = self.memory.get(SOFTWARE_INSTALLED_NAMES_KEY, None)
        if installed_names is None:
            installed_names = self._get_installed_software_names()
            self.memory.set(SOFTWARE_INSTALLED_NAMES_KEY, installed_names)

        installed_name_set = {n for n in installed_names}

        total_procs = 0
        installed_count = 0
        unregistered = []

        game_info = self.memory.get("current_game_info", None)
        game_exe = (game_info.get("exe") or "").lower() if game_info else None

        try:
            for p in psutil.process_iter(["pid", "name", "exe"]):
                total_procs += 1
                name = (p.info.get("name") or "").lower()
                exe = (p.info.get("exe") or "").lower()

                if game_exe and name == game_exe:
                    installed_count += 1
                    continue

                if is_exe_whitelisted(self.memory, exe):
                    installed_count += 1
                    continue

                if is_exe_blacklisted(self.memory, exe):
                    unregistered.append({
                        "pid": p.info["pid"],
                        "name": name,
                        "exe": exe,
                        "reason": "blacklisted",
                    })
                    continue

                if self._is_under_windows_dirs(exe):
                    installed_count += 1
                    continue

                known = False
                for inst in installed_name_set:
                    if inst and inst in name:
                        known = True
                        break

                if known:
                    installed_count += 1
                else:
                    unregistered.append({
                        "pid": p.info["pid"],
                        "name": name,
                        "exe": exe,
                        "reason": "unknown",
                    })
        except Exception:
            pass

        if len(unregistered) > 200:
            unregistered = unregistered[:200]
        self.memory.set(SOFTWARE_UNREGISTERED_PROCESSES_KEY, unregistered)

        anomaly_count = max(0, len(unregistered) // 10)

        return {
            "mode_name": "SoftwareInterrogation",
            "process_count": total_procs,
            "installed_process_count": installed_count,
            "unregistered_process_count": len(unregistered),
            "anomaly_count": anomaly_count,
            "sample_unregistered": unregistered[:10],
            "exe": "",
        }


class SnowWhiteMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        try:
            procs = list(psutil.process_iter(["pid"]))
            process_count = len(procs)
        except Exception:
            process_count = 0

        try:
            conns = psutil.net_connections(kind="inet")
            conn_count = len(conns)
        except Exception:
            conn_count = 0

        anomaly_count = max(0, conn_count - 500)

        return {
            "mode_name": "SnowWhite",
            "process_count": process_count,
            "connection_count": conn_count,
            "anomaly_count": anomaly_count,
            "exe": "",
        }


class BorgHiveMode(BaseMode):
    def _capture_snapshot(self) -> Dict[str, Any]:
        endpoints = []
        try:
            conns = psutil.net_connections(kind="inet")
            for c in conns:
                laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "?:?"
                raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "?:?"
                status = c.status
                endpoints.append((laddr, raddr, status))
        except Exception:
            pass

        endpoint_keys = [f"{l}|{r}|{s}" for (l, r, s) in endpoints]
        baseline = self.memory.get(BORG_BASELINE_KEY, None)

        if baseline is None:
            self.memory.set(BORG_BASELINE_KEY, endpoint_keys)
            rogue_endpoints = []
            missing_endpoints = []
            changed_endpoints = []
            rogue_count = 0
            anomaly_count = 0
        else:
            baseline_set = set(baseline)
            current_set = set(endpoint_keys)

            rogue_set = current_set - baseline_set
            missing_set = baseline_set - current_set
            changed_set = rogue_set.union(missing_set)

            rogue_endpoints = list(rogue_set)
            missing_endpoints = list(missing_set)
            changed_endpoints = list(changed_set)

            rogue_count = len(rogue_endpoints)
            anomaly_count = rogue_count + len(missing_endpoints)

        return {
            "mode_name": "BorgHive",
            "process_count": len(endpoints),
            "connection_count": len(endpoints),
            "rogue_count": rogue_count,
            "anomaly_count": anomaly_count,
            "rogue_endpoints": rogue_endpoints[:50],
            "missing_endpoints": missing_endpoints[:50],
            "changed_endpoints": changed_endpoints[:50],
            "exe": "",
        }


class CapsuleCortexMode(BaseMode):
    def __init__(
        self,
        name: str,
        baseline_manager: BaselineManager,
        memory: MemoryBackend,
        physics_engine: DataPhysicsEngine,
        uptime_provider,
        capsule_state: CapsuleState,
        capsule_metrics: CapsuleMetrics,
    ):
        super().__init__(name, baseline_manager, memory, physics_engine, uptime_provider)
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
            "exe": "",
        }

# ---------------------------------------------------------------------------
# GuardianEngine
# ---------------------------------------------------------------------------

class GuardianEngine:
    def __init__(self):
        self.local_base = os.path.join(os.path.expanduser("~"), "GuardianLocal")
        self.storage = EncryptedStorage(self.local_base)
        self.memory = MemoryBackend(self.storage)

        self.baseline_manager = BaselineManager(self.memory)
        self.inference_engine = InferenceEngine(self.memory)
        self.risk_model = RiskModel()
        self.self_eval = SelfEvalEngine(self.memory)
        self.world_model = WorldModel(self.memory, self.baseline_manager)

        self.physics_engine = DataPhysicsEngine(
            self.baseline_manager,
            self.inference_engine,
            self.risk_model,
            self.self_eval,
            self.world_model,
        )

        stored_state = self.memory.get(CONSCIOUSNESS_STATE_KEY, "FOCUSED")
        self.consciousness_state = (
            stored_state if stored_state in CONSCIOUSNESS_STATES else "FOCUSED"
        )

        self.capsule_state = CapsuleState(self.memory)
        self.capsule_metrics = CapsuleMetrics(self.memory)

        self.node = Node(self.memory)
        self.swarm = Swarm()
        self.daemon = CapsuleDaemon(self.capsule_state, self.capsule_metrics)

        self.swarm.sync()
        self.node.mutate()
        self.capsule_metrics.set_entropy(self.node.entropy)

        self.game_detector = GameDetectionEngine(self.memory)
        self.game_baselines = GameBaselineManager(self.memory)
        self.game_foresight = GameForesightEngine(self.memory, self.game_baselines)

        def uptime_provider():
            return time.time() - self._start_time

        self._start_time = time.time()
        self._stop = False

        self.modes: Dict[str, BaseMode] = {
            "Presidential": PresidentialMode(
                "Presidential",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "Bloodhound": BloodhoundMode(
                "Bloodhound",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "God": GodMode(
                "God",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "Software Interrogation": SoftwareInterrogatorMode(
                "SoftwareInterrogation",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "Snow White": SnowWhiteMode(
                "SnowWhite",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "Borg Hive": BorgHiveMode(
                "BorgHive",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
            ),
            "Capsule Cortex": CapsuleCortexMode(
                "CapsuleCortex",
                self.baseline_manager,
                self.memory,
                self.physics_engine,
                uptime_provider,
                self.capsule_state,
                self.capsule_metrics,
            ),
        }

        self._start_drive_retry_loop(interval_sec=60)
        threading.Thread(target=self._background_ritual, daemon=True).start()

    def set_consciousness_state(self, state: str):
        if state in CONSCIOUSNESS_STATES:
            self.consciousness_state = state
            self.memory.set(CONSCIOUSNESS_STATE_KEY, state)

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def init_baselines(self):
        for m in self.modes.values():
            m.init_baseline()

    def start_loop(self, interval_sec: int = 10):
        def loop():
            while not self._stop:
                self.node.mutate()
                self.capsule_metrics.set_entropy(self.node.entropy)

                game_info = self.game_detector.detect_game()
                if game_info:
                    if self.consciousness_state != GAME_MODE_STATE:
                        self.set_consciousness_state(GAME_MODE_STATE)
                else:
                    if self.consciousness_state == GAME_MODE_STATE:
                        self.set_consciousness_state("FOCUSED")

                cross_levels = {
                    name: mode.get_status().brain.level
                    for name, mode in self.modes.items()
                }

                for mode in self.modes.values():
                    mode.update_state(self.consciousness_state, cross_levels)

                time.sleep(interval_sec)

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._stop = True

    def _start_drive_retry_loop(self, interval_sec: int = 60):
        def drive_loop():
            while not self._stop:
                self._retry_drive_mount()
                time.sleep(interval_sec)
        threading.Thread(target=drive_loop, daemon=True).start()

    def _retry_drive_mount(self):
        status = self.memory.get(DRIVE_STATUS_KEY, "Unknown")
        if status == "Unknown":
            self.memory.set(DRIVE_STATUS_KEY, "green")

    def _background_ritual(self):
        while not self._stop:
            try:
                self.daemon.run_cycle()
            except Exception:
                pass
            time.sleep(5)

    def get_mode_status(self, name: str) -> ModeStatus:
        mode = self.modes.get(name)
        if not mode:
            return ModeStatus()
        return mode.get_status()

# ---------------------------------------------------------------------------
# GUI – Tactical HUD styling and dashboard
# ---------------------------------------------------------------------------

TACTICAL_BG = "#050b16"
TACTICAL_GRID = "#101724"
TACTICAL_BLUE = "#1f8cff"
TACTICAL_ORANGE = "#ff8c32"
TACTICAL_CYAN = "#7fd9ff"
TACTICAL_RED = "#ff4d4d"

def configure_tactical_styles(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("default")

    style.configure("TFrame", background=TACTICAL_BG)
    style.configure("TLabel", background=TACTICAL_BG, foreground=TACTICAL_CYAN)
    style.configure("TButton", background=TACTICAL_GRID, foreground=TACTICAL_CYAN)
    style.map("TButton", background=[("active", TACTICAL_BLUE)])

    style.configure("TNotebook", background=TACTICAL_BG, borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=TACTICAL_GRID,
                    foreground=TACTICAL_CYAN,
                    padding=(10, 5))
    style.map("TNotebook.Tab",
              background=[("selected", TACTICAL_BLUE)],
              foreground=[("selected", "#ffffff")])

    style.configure("ModeTile.TFrame", background=TACTICAL_GRID, relief="ridge", borderwidth=2)
    style.configure("ModeTile.TLabel", background=TACTICAL_GRID, foreground=TACTICAL_CYAN)

    style.configure("SidePanel.TFrame", background=TACTICAL_GRID, relief="groove", borderwidth=2)
    style.configure("SidePanel.TLabel", background=TACTICAL_GRID, foreground=TACTICAL_CYAN)

# ---------------------------------------------------------------------------
# Mode Tile
# ---------------------------------------------------------------------------

class ModeFrame(ttk.Frame):
    def __init__(self, parent, mode_name: str, engine: GuardianEngine):
        super().__init__(parent, style="ModeTile.TFrame", padding=6)
        self.engine = engine
        self.mode_name = mode_name

        self.label_title = ttk.Label(self, text=mode_name, style="ModeTile.TLabel",
                                     font=("Consolas", 11, "bold"))
        self.label_title.pack(anchor="w")

        self.label_status = ttk.Label(self, text="Status: ---", style="ModeTile.TLabel")
        self.label_status.pack(anchor="w")

        self.label_health = ttk.Label(self, text="Health: ---", style="ModeTile.TLabel")
        self.label_health.pack(anchor="w")

        self.label_guess = ttk.Label(self, text="Guess: ---", style="ModeTile.TLabel", wraplength=180)
        self.label_guess.pack(anchor="w")

    def refresh(self):
        status = self.engine.get_mode_status(self.mode_name)
        brain = status.brain

        self.label_status.config(text=f"Status: {brain.level.upper()}")
        self.label_health.config(text=f"Health: {brain.collective_health_score:.0f}")
        self.label_guess.config(text=f"Guess: {brain.best_guess}")

# ---------------------------------------------------------------------------
# Drive Status Panel
# ---------------------------------------------------------------------------

class DriveStatusFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, style="SidePanel.TFrame", padding=6)
        self.engine = engine

        ttk.Label(self, text="DRIVE STATUS", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).pack(anchor="w")

        self.lbl_primary = ttk.Label(self, text="Primary: ---", style="SidePanel.TLabel")
        self.lbl_fallback = ttk.Label(self, text="Fallback: ---", style="SidePanel.TLabel")
        self.lbl_status = ttk.Label(self, text="Status: ---", style="SidePanel.TLabel")

        self.lbl_primary.pack(anchor="w")
        self.lbl_fallback.pack(anchor="w")
        self.lbl_status.pack(anchor="w")

    def refresh(self):
        primary = self.engine.memory.get(PRIMARY_DRIVE_KEY, "None")
        fallback = self.engine.memory.get(FALLBACK_DRIVE_KEY, "None")
        status = self.engine.memory.get(DRIVE_STATUS_KEY, "Unknown")

        self.lbl_primary.config(text=f"Primary: {primary}")
        self.lbl_fallback.config(text=f"Fallback: {fallback}")
        self.lbl_status.config(text=f"Status: {status}")

# ---------------------------------------------------------------------------
# Altered States Panel
# ---------------------------------------------------------------------------

class AlteredStatesFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, style="SidePanel.TFrame", padding=6)
        self.engine = engine

        ttk.Label(self, text="ALTERED STATES", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).pack(anchor="w")

        self.lbl_state = ttk.Label(self, text="State: ---", style="SidePanel.TLabel")
        self.lbl_state.pack(anchor="w")

    def refresh(self):
        state = self.engine.consciousness_state
        self.lbl_state.config(text=f"State: {state}")

# ---------------------------------------------------------------------------
# Security Lists Panel
# ---------------------------------------------------------------------------

class SecurityListsFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, style="SidePanel.TFrame", padding=6)
        self.engine = engine

        ttk.Label(self, text="SECURITY LISTS", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).pack(anchor="w")

        ttk.Label(self, text="Whitelist:", style="SidePanel.TLabel").pack(anchor="w", pady=(5, 0))
        self.whitelist_box = tk.Listbox(self, height=6, bg=TACTICAL_BG, fg=TACTICAL_CYAN)
        self.whitelist_box.pack(fill="x", pady=(0, 5))

        btn_wl_add = ttk.Button(self, text="Add to Whitelist", command=self._add_whitelist)
        btn_wl_remove = ttk.Button(self, text="Remove Selected", command=self._remove_whitelist)
        btn_wl_add.pack(fill="x")
        btn_wl_remove.pack(fill="x", pady=(0, 10))

        ttk.Label(self, text="Blacklist:", style="SidePanel.TLabel").pack(anchor="w")
        self.blacklist_box = tk.Listbox(self, height=6, bg=TACTICAL_BG, fg=TACTICAL_ORANGE)
        self.blacklist_box.pack(fill="x", pady=(0, 5))

        btn_bl_add = ttk.Button(self, text="Add to Blacklist", command=self._add_blacklist)
        btn_bl_remove = ttk.Button(self, text="Remove Selected", command=self._remove_blacklist)
        btn_bl_add.pack(fill="x")
        btn_bl_remove.pack(fill="x")

        self.refresh()

    def refresh(self):
        self.whitelist_box.delete(0, tk.END)
        for item in self.engine.memory.get_whitelist():
            self.whitelist_box.insert(tk.END, item)

        self.blacklist_box.delete(0, tk.END)
        for item in self.engine.memory.get_blacklist():
            self.blacklist_box.insert(tk.END, item)

    def _add_whitelist(self):
        path = filedialog.askopenfilename(title="Select EXE to Whitelist")
        if path:
            self.engine.memory.add_to_whitelist(path)
            self.refresh()

    def _remove_whitelist(self):
        sel = self.whitelist_box.curselection()
        if sel:
            exe = self.whitelist_box.get(sel[0])
            self.engine.memory.remove_from_whitelist(exe)
            self.refresh()

    def _add_blacklist(self):
        path = filedialog.askopenfilename(title="Select EXE to Blacklist")
        if path:
            self.engine.memory.add_to_blacklist(path)
            self.refresh()

    def _remove_blacklist(self):
        sel = self.blacklist_box.curselection()
        if sel:
            exe = self.blacklist_box.get(sel[0])
            self.engine.memory.remove_from_blacklist(exe)
            self.refresh()

# ---------------------------------------------------------------------------
# Storage Tab – Manual Storage Controls + Health
# ---------------------------------------------------------------------------

class StorageFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, padding=10)
        self.engine = engine

        ttk.Label(self, text="Storage Configuration", font=("Consolas", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(self, text="Storage Health:", font=("Consolas", 10, "bold")).pack(anchor="w")
        self.lbl_health = ttk.Label(self, text="---", font=("Consolas", 10))
        self.lbl_health.pack(anchor="w", pady=(0, 10))

        ttk.Label(self, text="Primary Storage Location:", font=("Consolas", 10, "bold")).pack(anchor="w")
        self.lbl_primary = ttk.Label(self, text="---", font=("Consolas", 10))
        self.lbl_primary.pack(anchor="w", pady=(0, 5))

        ttk.Button(self, text="Select Primary Storage Folder", command=self._select_primary).pack(anchor="w", pady=(0, 10))

        ttk.Label(self, text="Fallback Storage Location:", font=("Consolas", 10, "bold")).pack(anchor="w")
        self.lbl_fallback = ttk.Label(self, text="---", font=("Consolas", 10))
        self.lbl_fallback.pack(anchor="w", pady=(0, 5))

        ttk.Button(self, text="Select Fallback Storage Folder", command=self._select_fallback).pack(anchor="w", pady=(0, 10))

        ttk.Label(self, text="SMB Network Path:", font=("Consolas", 10, "bold")).pack(anchor="w")
        self.lbl_smb = ttk.Label(self, text="---", font=("Consolas", 10))
        self.lbl_smb.pack(anchor="w", pady=(0, 10))

        ttk.Button(self, text="Set SMB Network Path", command=self._set_smb).pack(anchor="w", pady=5)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(self, text="Retry Mount", command=self._retry_mount).pack(anchor="w", pady=5)
        ttk.Button(self, text="Force Save Now", command=self._force_save).pack(anchor="w", pady=5)
        ttk.Button(self, text="Run Storage Test", command=self._storage_test).pack(anchor="w", pady=5)

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(self, text="Auto Failover Indicator:", font=("Consolas", 10, "bold")).pack(anchor="w")
        self.lbl_failover = ttk.Label(self, text="---", font=("Consolas", 10))
        self.lbl_failover.pack(anchor="w", pady=(0, 10))

        self.refresh()

    def refresh(self):
        primary = self.engine.memory.get(PRIMARY_DRIVE_KEY, "None")
        fallback = self.engine.memory.get(FALLBACK_DRIVE_KEY, "None")
        smb = self.engine.memory.get(STORAGE_SMB_PATH_KEY, "None")
        status = self.engine.memory.get(DRIVE_STATUS_KEY, "Unknown")

        self.lbl_primary.config(text=primary)
        self.lbl_fallback.config(text=fallback)
        self.lbl_smb.config(text=smb)

        if status == "green":
            health = "Healthy (primary reachable)"
        elif status == "yellow":
            health = "Degraded (using fallback or partial)"
        elif status == "red":
            health = "Failed (storage issues)"
        else:
            health = "Unknown – pending tests"

        self.lbl_health.config(text=health)

        if primary != "None" and fallback != "None":
            self.lbl_failover.config(text="Failover: Configured (primary → fallback)")
        elif primary != "None":
            self.lbl_failover.config(text="Failover: Primary only")
        elif fallback != "None":
            self.lbl_failover.config(text="Failover: Fallback only")
        else:
            self.lbl_failover.config(text="Failover: Not configured")

    def _select_primary(self):
        path = filedialog.askdirectory(title="Select Primary Storage Folder")
        if path:
            self.engine.memory.set(PRIMARY_DRIVE_KEY, path)
            self.engine.memory.set(DRIVE_STATUS_KEY, "green")
            self.refresh()

    def _select_fallback(self):
        path = filedialog.askdirectory(title="Select Fallback Storage Folder")
        if path:
            self.engine.memory.set(FALLBACK_DRIVE_KEY, path)
            self.refresh()

    def _set_smb(self):
        path = tk.simpledialog.askstring("SMB Path", "Enter SMB network path (e.g. \\\\server\\share):")
        if path:
            self.engine.memory.set(STORAGE_SMB_PATH_KEY, path)
            self.refresh()

    def _retry_mount(self):
        self.engine._retry_drive_mount()
        self.refresh()
        messagebox.showinfo("Storage", "Drive status re-evaluated.")

    def _force_save(self):
        if self.engine.memory.persistent_storage:
            self.engine.memory.persistent_storage.save(self.engine.memory._store)
        messagebox.showinfo("Storage", "Memory saved successfully.")

    def _storage_test(self):
        primary = self.engine.memory.get(PRIMARY_DRIVE_KEY, None)
        path = primary or self.engine.local_base
        test_file = os.path.join(path, "guardian_storage_test.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("guardian_storage_test")
            with open(test_file, "r") as f:
                data = f.read().strip()
            os.remove(test_file)
            if data == "guardian_storage_test":
                self.engine.memory.set(DRIVE_STATUS_KEY, "green")
                messagebox.showinfo("Storage Test", f"Storage test OK at: {path}")
            else:
                self.engine.memory.set(DRIVE_STATUS_KEY, "yellow")
                messagebox.showwarning("Storage Test", f"Read/write mismatch at: {path}")
        except Exception as e:
            self.engine.memory.set(DRIVE_STATUS_KEY, "red")
            messagebox.showerror("Storage Test", f"Storage test failed at: {path}\n{e}")
        self.refresh()

# ---------------------------------------------------------------------------
# GuardianApp (Dashboard + Security + Storage)
# ---------------------------------------------------------------------------

class GuardianApp(tk.Tk):
    def __init__(self, engine: GuardianEngine):
        super().__init__()
        self.engine = engine

        self.title("Fused Guardian – Evolution Tier + Game Cortex")
        self.geometry("1600x900")

        configure_tactical_styles(self)

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="Dashboard")

        self.game_cortex_frame = None

        self.security_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.security_frame, text="Security Lists")

        self.storage_frame = StorageFrame(self.notebook, self.engine)
        self.notebook.add(self.storage_frame, text="Storage")

        self._build_dashboard_ui(self.dashboard_frame)
        self._build_security_ui(self.security_frame)

        self.after(1000, self.refresh)

    def _build_dashboard_ui(self, parent):
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill="x", pady=(0, 10))

        self.global_label = ttk.Label(
            top_frame,
            text="Global: Initializing...",
            font=("Consolas", 12, "bold")
        )
        self.global_label.pack(side="left")

        # HUD button will be injected later by run_guardian
        self.state_var = tk.StringVar(value=self.engine.consciousness_state)
        state_menu = ttk.OptionMenu(
            top_frame,
            self.state_var,
            self.engine.consciousness_state,
            *CONSCIOUSNESS_STATES,
            command=self._on_state_change,
        )
        ttk.Label(top_frame, text="State:", font=("Consolas", 10)).pack(side="right")
        state_menu.pack(side="right")

        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill="both", expand=True)

        modes_frame = ttk.Frame(middle_frame)
        modes_frame.pack(side="left", fill="both", expand=True)

        right_column = ttk.Frame(middle_frame)
        right_column.pack(side="right", fill="y", padx=(10, 0))

        self.drive_frame = DriveStatusFrame(right_column, self.engine)
        self.drive_frame.pack(fill="x", pady=(0, 10))

        self.states_frame = AlteredStatesFrame(right_column, self.engine)
        self.states_frame.pack(fill="x", pady=(0, 10))

        self.security_lists_frame = SecurityListsFrame(right_column, self.engine)
        self.security_lists_frame.pack(fill="x", pady=(0, 10))

        initial_engine_state = "ACTIVE" if self.engine.capsule_state.state["active"] else "INACTIVE"
        self.engine_state_label = ttk.Label(
            right_column,
            text=f"Capsule Engine: {initial_engine_state}",
            font=("Consolas", 10, "bold")
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

    def _build_security_ui(self, parent):
        ttk.Label(parent, text="Security Lists", font=("Consolas", 14, "bold")).pack(anchor="w", pady=10)
        ttk.Label(parent, text="Manage whitelist and blacklist entries.", font=("Consolas", 10)).pack(anchor="w")

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
        self.states_frame.refresh()
        self.security_lists_frame.refresh()
        self.storage_frame.refresh()

        worst_level = "green"
        worst_health = 100.0
        level_order = {"green": 3, "yellow": 2, "red": 1, "grey": 4}

        for name in self.mode_frames:
            status = self.engine.get_mode_status(name)
            level = status.brain.level
            health = status.brain.collective_health_score
            if level_order[level] < level_order.get(worst_level, 3):
                worst_level = level
                worst_health = health
            elif level == worst_level and health < worst_health:
                worst_health = health

        uptime = self.engine.get_uptime()
        if uptime < FAST_BASELINE_WINDOW:
            warmup_msg = f"WARM-UP ({int(FAST_BASELINE_WINDOW - uptime)}s remaining)"
        else:
            warmup_msg = "STABLE BASELINE ESTABLISHED"

        self.global_label.configure(
            text=f"Global: {worst_level.upper()} | Health: {worst_health:.0f} | State: {self.engine.consciousness_state} | {warmup_msg}"
        )

        self.after(2000, self.refresh)

# ---------------------------------------------------------------------------
# Game Cortex Tab
# ---------------------------------------------------------------------------

class GameCortexFrame(ttk.Frame):
    def __init__(self, parent, engine: GuardianEngine):
        super().__init__(parent, style="TFrame", padding=8)
        self.engine = engine
        self.last_seen_game_pid = None
        self.game_active = False

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        for r in range(3):
            self.rowconfigure(r, weight=1)

        self.game_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.game_panel.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.game_panel, text="GAME IDENTIFICATION", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.lbl_title = ttk.Label(self.game_panel, text="Title: ---", style="SidePanel.TLabel")
        self.lbl_exe = ttk.Label(self.game_panel, text="EXE: ---", style="SidePanel.TLabel")
        self.lbl_platform = ttk.Label(self.game_panel, text="Platform: ---", style="SidePanel.TLabel")
        self.lbl_parent = ttk.Label(self.game_panel, text="Parent: ---", style="SidePanel.TLabel")
        self.lbl_ac = ttk.Label(self.game_panel, text="Anti-Cheat: ---", style="SidePanel.TLabel")
        self.lbl_mode = ttk.Label(self.game_panel, text="GAME_MODE: INACTIVE", style="SidePanel.TLabel")

        self.lbl_title.grid(row=1, column=0, sticky="w")
        self.lbl_exe.grid(row=2, column=0, sticky="w")
        self.lbl_platform.grid(row=3, column=0, sticky="w")
        self.lbl_parent.grid(row=4, column=0, sticky="w")
        self.lbl_ac.grid(row=5, column=0, sticky="w")
        self.lbl_mode.grid(row=6, column=0, sticky="w", pady=(4, 0))

        self.perf_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.perf_panel.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.perf_panel, text="PERFORMANCE", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.lbl_cpu = ttk.Label(self.perf_panel, text="CPU: ---", style="SidePanel.TLabel")
        self.lbl_ram = ttk.Label(self.perf_panel, text="RAM: ---", style="SidePanel.TLabel")
        self.lbl_gpu = ttk.Label(self.perf_panel, text="GPU: --- (placeholder)", style="SidePanel.TLabel")
        self.lbl_fps = ttk.Label(self.perf_panel, text="FPS: --- (optional)", style="SidePanel.TLabel")
        self.lbl_perf_foresight = ttk.Label(self.perf_panel, text="Foresight: ---", style="SidePanel.TLabel")

        self.lbl_cpu.grid(row=1, column=0, sticky="w")
        self.lbl_ram.grid(row=2, column=0, sticky="w")
        self.lbl_gpu.grid(row=3, column=0, sticky="w")
        self.lbl_fps.grid(row=4, column=0, sticky="w")
        self.lbl_perf_foresight.grid(row=5, column=0, sticky="w", pady=(4, 0))

        self.net_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.net_panel.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.net_panel, text="NETWORK", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.lbl_conns = ttk.Label(self.net_panel, text="Connections: ---", style="SidePanel.TLabel")
        self.lbl_latency = ttk.Label(self.net_panel, text="Latency: ---", style="SidePanel.TLabel")
        self.lbl_net_drift = ttk.Label(self.net_panel, text="Drift: ---", style="SidePanel.TLabel")

        self.lbl_conns.grid(row=1, column=0, sticky="w")
        self.lbl_latency.grid(row=2, column=0, sticky="w")
        self.lbl_net_drift.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.base_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.base_panel.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.base_panel, text="BASELINES", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.lbl_dev = ttk.Label(self.base_panel, text="Deviation: ---", style="SidePanel.TLabel")
        self.lbl_stability = ttk.Label(self.base_panel, text="Stability: ---", style="SidePanel.TLabel")
        self.lbl_horizons = ttk.Label(self.base_panel, text="Horizons: 5m / 1h / 4h / 1d / 30d", style="SidePanel.TLabel")

        self.lbl_dev.grid(row=1, column=0, sticky="w")
        self.lbl_stability.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.lbl_horizons.grid(row=3, column=0, sticky="w")

        self.capsule_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.capsule_panel.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.capsule_panel, text="CAPSULES", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.lbl_game_entropy = ttk.Label(self.capsule_panel, text="Game Entropy: ---", style="SidePanel.TLabel")
        self.lbl_game_capsules = ttk.Label(self.capsule_panel, text="Game Capsules: ---", style="SidePanel.TLabel")
        self.lbl_game_capsule_risk = ttk.Label(self.capsule_panel, text="Capsule Drift: ---", style="SidePanel.TLabel")

        self.lbl_game_entropy.grid(row=1, column=0, sticky="w")
        self.lbl_game_capsules.grid(row=2, column=0, sticky="w")
        self.lbl_game_capsule_risk.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.map_panel = ttk.Frame(self, style="SidePanel.TFrame", padding=6)
        self.map_panel.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)

        ttk.Label(self.map_panel, text="PROMETHEUS MAP (PREVIEW)", style="SidePanel.TLabel",
                  font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")

        self.map_list = tk.Listbox(
            self.map_panel,
            height=6,
            bg=TACTICAL_BG,
            fg=TACTICAL_CYAN,
            highlightbackground=TACTICAL_BLUE,
            selectbackground=TACTICAL_ORANGE
        )
        self.map_list.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.map_panel.rowconfigure(1, weight=1)
        self.map_panel.columnconfigure(0, weight=1)

        self.after(1000, self.refresh)

    def refresh(self):
        game_info = self.engine.memory.get("current_game_info", None)

        if game_info and game_info.get("is_game"):
            self.game_active = True
            pid = game_info.get("pid")
            self.last_seen_game_pid = pid

            self.lbl_title.config(text=f"Title: {game_info.get('title', 'Unknown')}")
            self.lbl_exe.config(text=f"EXE: {game_info.get('exe', 'Unknown')}")
            self.lbl_platform.config(text=f"Platform: {game_info.get('platform', 'Unknown')}")
            self.lbl_parent.config(text=f"Parent: {game_info.get('parent', 'Unknown')}")
            self.lbl_ac.config(text=f"Anti-Cheat: {game_info.get('anti_cheat', 'Normal')}")
            self.lbl_mode.config(text="GAME_MODE: ACTIVE", foreground=TACTICAL_ORANGE)

            try:
                p = psutil.Process(pid)
                cpu = p.cpu_percent(interval=0.1)
                mem = p.memory_info().rss / (1024 * 1024 * 1024)
            except Exception:
                cpu = 0.0
                mem = 0.0

            perf_snapshot = {
                "cpu": cpu,
                "ram": mem,
                "connections": self._count_game_connections(pid),
            }

            self.engine.game_baselines.update_baseline(game_info["title"], perf_snapshot)
            foresight = self.engine.game_foresight.predict(game_info["title"], perf_snapshot)

            self.lbl_cpu.config(text=f"CPU: {cpu:.1f}%")
            self.lbl_ram.config(text=f"RAM: {mem:.2f} GB")
            self.lbl_gpu.config(text="GPU: (not wired yet)")
            self.lbl_fps.config(text="FPS: (overlay optional)")
            self.lbl_perf_foresight.config(
                text=f"Foresight: {foresight.get('foresight', '---')}",
                foreground=TACTICAL_ORANGE if foresight.get("drift", 0) > 0.3 else TACTICAL_CYAN
            )

            conns = perf_snapshot["connections"]
            latency_est = self._estimate_latency()
            drift = foresight.get("drift", 0.0)
            self.lbl_conns.config(text=f"Connections: {conns}")
            self.lbl_latency.config(text=f"Latency: ~{latency_est} ms")
            self.lbl_net_drift.config(text=f"Drift: {drift:.2f}",
                                      foreground=TACTICAL_ORANGE if drift > 0.3 else TACTICAL_CYAN)

            stability = "Stable" if drift < 0.2 else "Shifting" if drift < 0.5 else "Unstable"
            self.lbl_dev.config(text=f"Deviation: {drift:.2f}")
            self.lbl_stability.config(text=f"Stability: {stability}",
                                      foreground=TACTICAL_ORANGE if drift > 0.3 else TACTICAL_CYAN)

            metrics = self.engine.capsule_metrics.snapshot()
            total_caps = metrics.get("total_capsules", 0)
            entropy = metrics.get("entropy", 0.0)
            unauth = metrics.get("unauthorized_attempts", 0)

            self.lbl_game_entropy.config(text=f"Game Entropy: {entropy:.2f}")
            self.lbl_game_capsules.config(text=f"Game Capsules: {total_caps}")
            self.lbl_game_capsule_risk.config(
                text=f"Capsule Drift: Unaudited={unauth}",
                foreground=TACTICAL_ORANGE if unauth > 0 else TACTICAL_CYAN
            )

            self._update_prometheus_preview(pid)

        else:
            if self.game_active:
                self.lbl_mode.config(text="GAME_MODE: INACTIVE", foreground=TACTICAL_CYAN)
            self.game_active = False
            self.lbl_title.config(text="Title: ---")
            self.lbl_exe.config(text="EXE: ---")
            self.lbl_platform.config(text="Platform: ---")
            self.lbl_parent.config(text="Parent: ---")
            self.lbl_ac.config(text="Anti-Cheat: ---")

        self.after(1000, self.refresh)

    def _count_game_connections(self, pid: int) -> int:
        total = 0
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.pid == pid:
                    total += 1
        except Exception:
            pass
        return total

    def _estimate_latency(self) -> int:
        return random.randint(20, 80)

    def _update_prometheus_preview(self, pid: int):
        self.map_list.delete(0, tk.END)
        self.map_list.insert(tk.END, f"Game Node: PID {pid}")

        try:
            p = psutil.Process(pid)
            for child in p.children(recursive=True):
                cname = (child.name() or "").lower()
                self.map_list.insert(tk.END, f"Child: {cname} (PID {child.pid})")
        except Exception:
            pass

        try:
            seen = set()
            for c in psutil.net_connections(kind="inet"):
                if c.pid == pid and c.raddr:
                    endpoint = f"{c.raddr.ip}:{c.raddr.port}"
                    if endpoint not in seen:
                        seen.add(endpoint)
                        self.map_list.insert(tk.END, f"Endpoint: {endpoint}")
        except Exception:
            pass


def add_game_cortex_tab(app: GuardianApp):
    app.game_cortex_frame = GameCortexFrame(app.notebook, app.engine)
    app.notebook.add(app.game_cortex_frame, text="Game Cortex")

# ---------------------------------------------------------------------------
# AssistantBrain – unified reasoning over all modes
# ---------------------------------------------------------------------------

class AssistantBrain:
    def __init__(self, engine: GuardianEngine):
        self.engine = engine

    def get_system_summary(self):
        modes_summary = {}
        worst_level = "green"
        worst_health = 100.0
        collapse_probability = 0.0
        level_order = {"green": 3, "yellow": 2, "red": 1, "grey": 4}

        for name, mode in self.engine.modes.items():
            st = mode.get_status().brain
            modes_summary[name] = {
                "level": st.level,
                "health": st.collective_health_score,
                "judgment": st.judgment,
                "confidence": st.confidence,
                "foresight": st.foresight_score,
                "collapse_probability": st.collapse_probability,
                "best_guess": st.best_guess,
            }

            if level_order.get(st.level, 4) < level_order.get(worst_level, 4):
                worst_level = st.level
                worst_health = st.collective_health_score
            elif st.level == worst_level and st.collective_health_score < worst_health:
                worst_health = st.collective_health_score

            collapse_probability = max(collapse_probability, st.collapse_probability)

        game_info = self.engine.memory.get("current_game_info", None)
        briefing = self.get_operator_briefing(
            worst_level, worst_health, collapse_probability, modes_summary, game_info
        )

        return {
            "global_level": worst_level,
            "global_health": worst_health,
            "collapse_probability": collapse_probability,
            "modes": modes_summary,
            "game_info": game_info,
            "briefing": briefing,
        }

    def suggest_actions(self):
        summary = self.get_system_summary()
        suggestions = []

        level = summary["global_level"]
        health = summary["global_health"]
        collapse = summary["collapse_probability"]
        game = summary["game_info"]

        if level == "green" and health > 90 and collapse < 0.1:
            suggestions.append("System posture is strong – no immediate actions required.")
        elif level == "yellow":
            suggestions.append("Elevated load detected – consider checking network or background apps.")
        elif level == "red":
            suggestions.append("High risk detected – review processes and connections.")

        if game and game.get("is_game"):
            suggestions.append(f"Game '{game.get('title', 'Unknown')}' is active – monitoring drift.")

        wl = self.engine.memory.get_whitelist()
        bl = self.engine.memory.get_blacklist()

        if len(bl) > 0:
            suggestions.append(f"{len(bl)} executable(s) are blacklisted – review Security Lists.")
        if len(wl) == 0:
            suggestions.append("Whitelist is empty – consider whitelisting trusted apps.")

        return suggestions

    def get_operator_briefing(self, worst_level, worst_health, collapse, modes_summary, game_info):
        base = f"{worst_level.upper()} | Health {worst_health:.0f} | Collapse {collapse:.2f}"

        hot_modes = [name for name, m in modes_summary.items() if m["level"] in ("yellow", "red")]
        if hot_modes:
            base += f" | Watch: {', '.join(hot_modes)}"

        if game_info and game_info.get("is_game"):
            base += f" | Game: {game_info.get('title', 'Unknown')}"

        return base

# ---------------------------------------------------------------------------
# Diagnostic Overlay – floating HUD
# ---------------------------------------------------------------------------

class DiagnosticOverlay(tk.Toplevel):
    def __init__(self, parent, engine: GuardianEngine, brain: AssistantBrain):
        super().__init__(parent)
        self.engine = engine
        self.brain = brain

        self.title("Guardian HUD")
        self.attributes("-topmost", True)
        try:
            self.attributes("-alpha", 0.9)
        except Exception:
            pass

        self.configure(bg="#050b16")
        self._closing = False

        self._build_ui()
        self.after(500, self._refresh_loop)

    def _build_ui(self):
        self.main_frame = ttk.Frame(self, padding=6)
        self.main_frame.pack(fill="both", expand=True)

        ttk.Label(
            self.main_frame,
            text="Guardian Diagnostic Overlay",
            font=("Consolas", 11, "bold")
        ).pack(anchor="w", pady=(0, 4))

        self.lbl_global = ttk.Label(self.main_frame, text="Global: ---", font=("Consolas", 10))
        self.lbl_global.pack(anchor="w")

        self.modes_text = tk.Text(
            self.main_frame, height=10, width=60,
            bg="#050b16", fg="#7fd9ff", relief="flat"
        )
        self.modes_text.pack(fill="both", expand=True, pady=(4, 4))
        self.modes_text.config(state="disabled")

        self.lbl_briefing = ttk.Label(self.main_frame, text="Briefing: ---", font=("Consolas", 10))
        self.lbl_briefing.pack(anchor="w")

        self.suggestions_text = tk.Text(
            self.main_frame, height=5, width=60,
            bg="#050b16", fg="#ff8c32", relief="flat"
        )
        self.suggestions_text.pack(fill="x", pady=(4, 0))
        self.suggestions_text.config(state="disabled")

    def _refresh_loop(self):
        if not self._closing:
            self._refresh_contents()
            self.after(1500, self._refresh_loop)

    def _refresh_contents(self):
        summary = self.brain.get_system_summary()
        suggestions = self.brain.suggest_actions()

        self.lbl_global.config(
            text=f"Global: {summary['global_level'].upper()} | "
                 f"Health {summary['global_health']:.0f} | "
                 f"Collapse {summary['collapse_probability']:.2f}"
        )

        self.modes_text.config(state="normal")
        self.modes_text.delete("1.0", tk.END)
        for name, info in summary["modes"].items():
            self.modes_text.insert(
                tk.END,
                f"{name}: {info['level'].upper()} "
                f"(Health {info['health']:.0f}, "
                f"Judg {info['judgment']:.0f}, "
                f"Conf {info['confidence']:.0f}, "
                f"Coll {info['collapse_probability']:.2f})\n"
                f"    {info['best_guess']}\n"
            )
        self.modes_text.config(state="disabled")

        self.lbl_briefing.config(text=f"Briefing: {summary['briefing']}")

        self.suggestions_text.config(state="normal")
        self.suggestions_text.delete("1.0", tk.END)
        for s in suggestions:
            self.suggestions_text.insert(tk.END, f"- {s}\n")
        self.suggestions_text.config(state="disabled")

    def destroy(self):
        self._closing = True
        super().destroy()

# ---------------------------------------------------------------------------
# Flask API + Prometheus Map
# ---------------------------------------------------------------------------

def create_flask_app(engine: GuardianEngine) -> Flask:
    app = Flask(__name__)

    @app.route("/status", methods=["GET"])
    def status():
        data = {}
        for name, mode in engine.modes.items():
            st = mode.get_status().brain
            data[name] = {
                "level": st.level,
                "health": st.collective_health_score,
                "judgment": st.judgment,
                "confidence": st.confidence,
                "foresight": st.foresight_score,
                "best_guess": st.best_guess,
            }
        data["consciousness_state"] = engine.consciousness_state
        data["uptime"] = engine.get_uptime()
        data["game_info"] = engine.memory.get("current_game_info", None)
        return jsonify(data)

    @app.route("/prometheus_map", methods=["GET"])
    def prometheus_map():
        nodes = []
        edges = []

        for name, mode in engine.modes.items():
            st = mode.get_status().brain
            nodes.append({
                "id": f"mode:{name}",
                "type": "mode",
                "label": name,
                "risk": 100 - st.collective_health_score,
                "level": st.level,
            })

        try:
            for p in psutil.process_iter(["pid", "name"]):
                pid = p.info["pid"]
                label = p.info.get("name") or str(pid)
                nodes.append({
                    "id": f"proc:{pid}",
                    "type": "process",
                    "label": label,
                    "risk": 0.0,
                })
        except Exception:
            pass

        try:
            conns = psutil.net_connections(kind="inet")
            for c in conns:
                if not c.raddr:
                    continue
                pid = c.pid
                src_id = f"proc:{pid}" if pid is not None else "proc:unknown"
                dst_id = f"endpoint:{c.raddr.ip}:{c.raddr.port}"

                nodes.append({
                    "id": dst_id,
                    "type": "endpoint",
                    "label": dst_id,
                    "risk": 0.0,
                })
                edges.append({
                    "source": src_id,
                    "target": dst_id,
                    "type": "connection",
                    "weight": 1.0,
                })
        except Exception:
            pass

        for name, mode in engine.modes.items():
            st = mode.get_status().brain
            if st.level in ("yellow", "red"):
                for other in engine.modes.keys():
                    if other == name:
                        continue
                    edges.append({
                        "source": f"mode:{name}",
                        "target": f"mode:{other}",
                        "type": "influence",
                        "weight": 1.0 if st.level == "red" else 0.5,
                    })

        game_info = engine.memory.get("current_game_info", None)
        if game_info and game_info.get("is_game"):
            game_node_id = f"game:{game_info['pid']}"
            nodes.append({
                "id": game_node_id,
                "type": "game",
                "label": game_info["title"],
                "risk": 0.0,
                "level": "game",
            })

            proc_node_id = f"proc:{game_info['pid']}"
            edges.append({
                "source": proc_node_id,
                "target": game_node_id,
                "type": "association",
                "weight": 1.0,
            })

            if game_info.get("anti_cheat") == "Active":
                ac_id = f"anticheat:{game_info['pid']}"
                nodes.append({
                    "id": ac_id,
                    "type": "anti_cheat",
                    "label": "Anti-Cheat",
                    "risk": 0.0,
                })
                edges.append({
                    "source": game_node_id,
                    "target": ac_id,
                    "type": "security",
                    "weight": 1.0,
                })

        return jsonify({"nodes": nodes, "edges": edges})

    return app

# ---------------------------------------------------------------------------
# Entry point – Guardian + Assistant + HUD
# ---------------------------------------------------------------------------

def run_guardian():
    engine = GuardianEngine()
    engine.init_baselines()
    engine.start_loop(interval_sec=10)

    brain = AssistantBrain(engine)

    def run_api():
        app = create_flask_app(engine)
        app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False)

    threading.Thread(target=run_api, daemon=True).start()

    gui = GuardianApp(engine)
    add_game_cortex_tab(gui)

    overlay_holder = {"instance": None}

    def toggle_overlay():
        if overlay_holder["instance"] is None or not overlay_holder["instance"].winfo_exists():
            overlay_holder["instance"] = DiagnosticOverlay(gui, engine, brain)
        else:
            overlay_holder["instance"].destroy()
            overlay_holder["instance"] = None

    gui.toggle_overlay = toggle_overlay

    def inject_button():
        try:
            top_frame = gui.dashboard_frame.winfo_children()[0]
            ttk.Button(
                top_frame,
                text="Diagnostic HUD",
                command=gui.toggle_overlay
            ).pack(side="right", padx=10)
        except Exception:
            pass

    gui.after(500, inject_button)

    gui.mainloop()


if __name__ == "__main__":
    run_guardian()

