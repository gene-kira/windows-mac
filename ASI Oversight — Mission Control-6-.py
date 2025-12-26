"""
MagicBox + LCARS-DARK ASI Defense Interface
Single-file fusion: background organism + alien LCARS-style defense console.
"""

# =========================
# Standard library imports
# =========================
import importlib
import subprocess
import sys
import threading
import queue
import random
import time
import datetime
import json
import os
import base64
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable

# =========================
# Dependency Autoloader
# =========================


class DependencyAutoloader:
    def __init__(
        self,
        required_packages: List[str],
        on_status: Optional[Callable[[str, str], None]] = None,
    ):
        self.required_packages = required_packages
        self.on_status = on_status
        self.status: Dict[str, str] = {}

    def _emit(self, pkg: str, message: str) -> None:
        self.status[pkg] = message
        if self.on_status:
            self.on_status(pkg, message)

    def _is_installed(self, pkg: str) -> bool:
        try:
            importlib.import_module(pkg)
            return True
        except ImportError:
            return False

    def _install_package(self, pkg: str) -> bool:
        self._emit(pkg, "installing")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self._emit(pkg, "ok")
                return True
            else:
                self._emit(pkg, f"failed: {result.stderr.strip()[:300]}")
                return False
        except Exception as e:
            self._emit(pkg, f"failed: {e}")
            return False

    def ensure_all(self, async_mode: bool = True) -> None:
        if async_mode:
            threading.Thread(target=self._ensure_all_sync, daemon=True).start()
        else:
            self._ensure_all_sync()

    def _ensure_all_sync(self) -> None:
        for pkg in self.required_packages:
            if self._is_installed(pkg):
                self._emit(pkg, "ok")
            else:
                self._install_package(pkg)


def _dep_status(pkg: str, status: str) -> None:
    print(f"[AUTOLOADER] {pkg}: {status}")


REQUIRED_PACKAGES = [
    "numpy",
    "psutil",
    "requests",
]

autoloader = DependencyAutoloader(REQUIRED_PACKAGES, on_status=_dep_status)
autoloader.ensure_all(async_mode=True)

import numpy as np
import psutil

STATE_SNAPSHOT_FILE = "magicbox_state.json"

# =========================
# Simple encryption helpers
# =========================


def _derive_key_from_passphrase(passphrase: str) -> bytes:
    return hashlib.sha256(passphrase.encode("utf-8")).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    out = bytearray()
    kl = len(key)
    for i, b in enumerate(data):
        out.append(b ^ key[i % kl])
    return bytes(out)


# =========================
# Core Enums and State Models
# =========================


class BrainMode(Enum):
    STABILITY = "Stability"
    REFLEX = "Reflex"
    EXPLORATION = "Exploration"
    AUTO = "Auto"


class EnvironmentState(Enum):
    CALM = "Calm"
    TENSE = "Tense"
    DANGER = "Danger"


class MissionType(Enum):
    PROTECT = "PROTECT"
    STABILITY = "STABILITY"
    LEARN = "LEARN"
    OPTIMIZE = "OPTIMIZE"
    AUTO = "AUTO"


class PredictionHorizon(Enum):
    SHORT = "Short"
    MEDIUM = "Medium"
    LONG = "Long"


class Posture(Enum):
    WATCHER = "Watcher"
    GUARDIAN = "Guardian"
    CHAMELEON = "Chameleon"


@dataclass
class HybridBrainCore:
    mode: BrainMode = BrainMode.AUTO
    volatility: float = 0.0
    trust: float = 0.5
    cognitive_load: float = 0.0
    sensitivity: float = 0.5

    def set_mode(self, mode: BrainMode) -> None:
        self.mode = mode

    def adjust_sensitivity(self, delta: float) -> None:
        self.sensitivity = min(1.0, max(0.0, self.sensitivity + delta))

    def calibrate_trust(self, delta: float) -> None:
        self.trust = min(1.0, max(0.0, self.trust + delta))


@dataclass
class JudgmentEngine:
    judgment_confidence: float = 0.5
    good_outcomes: int = 0
    bad_outcomes: int = 0
    bias_drift: float = 0.0
    learning_frozen: bool = False
    last_outcome_good: Optional[bool] = None

    def record_outcome(self, good: bool) -> None:
        if self.learning_frozen:
            return
        self.last_outcome_good = good
        if good:
            self.good_outcomes += 1
        else:
            self.bad_outcomes += 1
        total = max(1, self.good_outcomes + self.bad_outcomes)
        self.judgment_confidence = self.good_outcomes / total
        self.bias_drift = np.clip(self.bias_drift + (0.05 if good else -0.05), -1.0, 1.0)

    def reset_bias(self) -> None:
        self.bias_drift = 0.0

    def freeze_learning(self, frozen: bool) -> None:
        self.learning_frozen = frozen

    def accelerate_learning(self, factor: float = 1.2) -> None:
        self.bias_drift = max(-1.0, min(1.0, self.bias_drift * factor))


@dataclass
class SituationalCortex:
    mission: MissionType = MissionType.AUTO
    environment: EnvironmentState = EnvironmentState.CALM
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    anticipation: str = "Neutral"
    risk_tolerance: float = 0.5
    opportunity_focus: float = 0.5

    def set_mission(self, mission: MissionType) -> None:
        self.mission = mission

    def adjust_risk_tolerance(self, delta: float) -> None:
        self.risk_tolerance = min(1.0, max(0.0, self.risk_tolerance + delta))

    def adjust_opportunity_focus(self, delta: float) -> None:
        self.opportunity_focus = min(1.0, max(0.0, self.opportunity_focus + delta))


@dataclass
class PredictiveIntelligence:
    anomaly_risk: float = 0.0
    drive_system_risk: float = 0.0
    hive_risk: float = 0.0
    collective_health_score: float = 1.0
    health_trend: float = 0.0
    forecast_summary: str = "CALM"

    horizon: PredictionHorizon = PredictionHorizon.SHORT
    anomaly_sensitivity: float = 0.5
    hive_weight: float = 0.5

    def set_horizon(self, horizon: PredictionHorizon) -> None:
        self.horizon = horizon

    def adjust_anomaly_sensitivity(self, delta: float) -> None:
        self.anomaly_sensitivity = min(1.0, max(0.0, self.anomaly_sensitivity + delta))

    def adjust_hive_weight(self, delta: float) -> None:
        self.hive_weight = min(1.0, max(0.0, self.hive_weight + delta))


@dataclass
class CollectiveHealth:
    collective_risk_score: float = 0.0
    hive_density: float = 0.0
    node_agreement: float = 1.0
    divergence_patterns: List[str] = field(default_factory=list)
    hive_mode: str = "Conservative"
    hive_trust_bias: float = 0.5

    def set_hive_mode(self, mode: str) -> None:
        self.hive_mode = mode

    def adjust_hive_trust(self, delta: float) -> None:
        self.hive_trust_bias = min(1.0, max(0.0, self.hive_trust_bias + delta))


@dataclass
class AutonomousCipherEngine:
    mode: str = "Normal"
    key_entropy: float = 0.5
    rotation_interval: int = 300
    ghost_sync_detected: bool = False
    defense_level: int = 1
    last_rotation_reason: str = "Initial"
    telemetry_retention_seconds: int = 3600
    phantom_node_enabled: bool = False

    def escalate_defense(self, reason: str = "Manual escalation") -> None:
        self.defense_level = min(5, self.defense_level + 1)
        if self.defense_level >= 3:
            self.mode = "Hardened"
        if self.defense_level >= 5:
            self.mode = "Blackout"
        self.last_rotation_reason = reason

    def rotate_keys(self, reason: str = "Scheduled rotation") -> None:
        self.key_entropy = min(1.0, self.key_entropy + 0.1)
        self.last_rotation_reason = reason

    def handle_ghost_sync(self) -> None:
        self.ghost_sync_detected = True
        self.telemetry_retention_seconds = max(60, self.telemetry_retention_seconds // 4)
        self.phantom_node_enabled = True
        self.escalate_defense("Ghost Sync Detected")
        self.rotate_keys("Ghost Sync Key Rotation")


@dataclass
class CodexMutationEngine:
    agent_weights: np.ndarray = field(default_factory=lambda: np.array([0.6, -0.8, -0.3], dtype=float))
    mutation_log: List[Dict] = field(default_factory=list)
    adaptive_enabled: bool = True
    drift_magnitude: float = 0.0
    last_mutation_reason: str = "Initial"

    def mutate_based_on_frame(self, frame: Dict) -> None:
        if not self.adaptive_enabled:
            return

        anomaly = frame.get("anomaly_risk", 0.0)
        divergence = frame.get("hive_divergence", 0.0)
        ghost_sync = frame.get("ghost_sync", False)
        health = frame.get("collective_health", 1.0)

        delta = np.array([
            anomaly * 0.05,
            divergence * 0.05,
            (0.2 if ghost_sync else -0.02 * health),
        ], dtype=float)

        old_weights = self.agent_weights.copy()
        self.agent_weights = self.agent_weights + delta
        self.drift_magnitude = float(np.linalg.norm(self.agent_weights - old_weights))

        reason = "ghost_sync" if ghost_sync else "risk_frame"
        self.last_mutation_reason = reason

        record = {
            "reason": reason,
            "delta": [float(x) for x in delta.tolist()],
            "new_weights": [float(x) for x in self.agent_weights.tolist()],
            "frame": frame,
        }
        self.mutation_log.append(record)

    def reset_drift(self) -> None:
        self.drift_magnitude = 0.0
        self.last_mutation_reason = "Drift reset"

    def toggle_adaptive(self, enabled: bool) -> None:
        self.adaptive_enabled = enabled


@dataclass
class MissionControlState:
    brain: HybridBrainCore = field(default_factory=HybridBrainCore)
    judgment: JudgmentEngine = field(default_factory=JudgmentEngine)
    cortex: SituationalCortex = field(default_factory=SituationalCortex)
    prediction: PredictiveIntelligence = field(default_factory=PredictiveIntelligence)
    collective: CollectiveHealth = field(default_factory=CollectiveHealth)
    cipher: AutonomousCipherEngine = field(default_factory=AutonomousCipherEngine)
    codex: CodexMutationEngine = field(default_factory=CodexMutationEngine)

    last_snapshot: Optional["MissionControlState"] = None
    posture: Posture = Posture.WATCHER

    def snapshot(self) -> None:
        import copy
        self.last_snapshot = copy.deepcopy(self)

    def rollback(self) -> None:
        if self.last_snapshot is not None:
            restored = self.last_snapshot
            self.brain = restored.brain
            self.judgment = restored.judgment
            self.cortex = restored.cortex
            self.prediction = restored.prediction
            self.collective = restored.collective
            self.cipher = restored.cipher
            self.codex = restored.codex


# =========================
# State Persistence Manager
# =========================


class StatePersistenceManager:
    def __init__(self, path: str = STATE_SNAPSHOT_FILE, passphrase: Optional[str] = None):
        self.path = path
        self.passphrase = passphrase

    def save(self, state: MissionControlState) -> None:
        try:
            data = {
                "brain": {
                    "mode": state.brain.mode.value,
                    "volatility": state.brain.volatility,
                    "trust": state.brain.trust,
                    "cognitive_load": state.brain.cognitive_load,
                    "sensitivity": state.brain.sensitivity,
                },
                "judgment": {
                    "judgment_confidence": state.judgment.judgment_confidence,
                    "good_outcomes": state.judgment.good_outcomes,
                    "bad_outcomes": state.judgment.bad_outcomes,
                    "bias_drift": state.judgment.bias_drift,
                },
                "cortex": {
                    "mission": state.cortex.mission.value,
                    "environment": state.cortex.environment.value,
                    "opportunity_score": state.cortex.opportunity_score,
                    "risk_score": state.cortex.risk_score,
                    "anticipation": state.cortex.anticipation,
                    "risk_tolerance": state.cortex.risk_tolerance,
                    "opportunity_focus": state.cortex.opportunity_focus,
                },
                "prediction": {
                    "anomaly_risk": state.prediction.anomaly_risk,
                    "drive_system_risk": state.prediction.drive_system_risk,
                    "hive_risk": state.prediction.hive_risk,
                    "collective_health_score": state.prediction.collective_health_score,
                    "health_trend": state.prediction.health_trend,
                    "horizon": state.prediction.horizon.value,
                    "anomaly_sensitivity": state.prediction.anomaly_sensitivity,
                    "hive_weight": state.prediction.hive_weight,
                    "forecast_summary": state.prediction.forecast_summary,
                },
                "collective": {
                    "collective_risk_score": state.collective.collective_risk_score,
                    "hive_density": state.collective.hive_density,
                    "node_agreement": state.collective.node_agreement,
                    "divergence_patterns": state.collective.divergence_patterns,
                    "hive_mode": state.collective.hive_mode,
                    "hive_trust_bias": state.collective.hive_trust_bias,
                },
                "cipher": {
                    "mode": state.cipher.mode,
                    "key_entropy": state.cipher.key_entropy,
                    "rotation_interval": state.cipher.rotation_interval,
                    "ghost_sync_detected": state.cipher.ghost_sync_detected,
                    "defense_level": state.cipher.defense_level,
                    "last_rotation_reason": state.cipher.last_rotation_reason,
                    "telemetry_retention_seconds": state.cipher.telemetry_retention_seconds,
                    "phantom_node_enabled": state.cipher.phantom_node_enabled,
                },
                "codex": {
                    "agent_weights": [float(x) for x in state.codex.agent_weights.tolist()],
                    "adaptive_enabled": state.codex.adaptive_enabled,
                    "drift_magnitude": state.codex.drift_magnitude,
                    "last_mutation_reason": state.codex.last_mutation_reason,
                },
                "posture": state.posture.value,
            }
            raw = json.dumps(data, indent=2).encode("utf-8")
            if self.passphrase:
                key = _derive_key_from_passphrase(self.passphrase)
                cipher = _xor_bytes(raw, key)
                payload = {
                    "encrypted": True,
                    "data": base64.b64encode(cipher).decode("ascii"),
                }
            else:
                payload = {
                    "encrypted": False,
                    "data": raw.decode("utf-8"),
                }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[STATE] Saved organism state to {self.path} (encrypted={bool(self.passphrase)})")
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

    def load_into(self, state: MissionControlState) -> None:
        if not os.path.exists(self.path):
            print("[STATE] No previous state snapshot found.")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if payload.get("encrypted"):
                if not self.passphrase:
                    print("[STATE] Snapshot is encrypted but no passphrase provided. Skipping restore.")
                    return
                key = _derive_key_from_passphrase(self.passphrase)
                cipher = base64.b64decode(payload["data"].encode("ascii"))
                raw = _xor_bytes(cipher, key).decode("utf-8")
                data = json.loads(raw)
            else:
                data = json.loads(payload["data"]) if isinstance(payload.get("data"), str) else payload

            # Brain
            state.brain.mode = BrainMode(data["brain"]["mode"])
            state.brain.volatility = data["brain"]["volatility"]
            state.brain.trust = data["brain"]["trust"]
            state.brain.cognitive_load = data["brain"]["cognitive_load"]
            state.brain.sensitivity = data["brain"]["sensitivity"]

            # Judgment
            state.judgment.judgment_confidence = data["judgment"]["judgment_confidence"]
            state.judgment.good_outcomes = data["judgment"]["good_outcomes"]
            state.judgment.bad_outcomes = data["judgment"]["bad_outcomes"]
            state.judgment.bias_drift = data["judgment"]["bias_drift"]

            # Cortex
            cx = data["cortex"]
            state.cortex.mission = MissionType(cx["mission"])
            state.cortex.environment = EnvironmentState(cx["environment"])
            state.cortex.opportunity_score = cx["opportunity_score"]
            state.cortex.risk_score = cx["risk_score"]
            state.cortex.anticipation = cx["anticipation"]
            state.cortex.risk_tolerance = cx["risk_tolerance"]
            state.cortex.opportunity_focus = cx["opportunity_focus"]

            # Prediction
            pr = data["prediction"]
            state.prediction.anomaly_risk = pr["anomaly_risk"]
            state.prediction.drive_system_risk = pr["drive_system_risk"]
            state.prediction.hive_risk = pr["hive_risk"]
            state.prediction.collective_health_score = pr["collective_health_score"]
            state.prediction.health_trend = pr["health_trend"]
            state.prediction.horizon = PredictionHorizon(pr["horizon"])
            state.prediction.anomaly_sensitivity = pr["anomaly_sensitivity"]
            state.prediction.hive_weight = pr["hive_weight"]
            state.prediction.forecast_summary = pr["forecast_summary"]

            # Collective
            co = data["collective"]
            state.collective.collective_risk_score = co["collective_risk_score"]
            state.collective.hive_density = co["hive_density"]
            state.collective.node_agreement = co["node_agreement"]
            state.collective.divergence_patterns = co["divergence_patterns"]
            state.collective.hive_mode = co["hive_mode"]
            state.collective.hive_trust_bias = co["hive_trust_bias"]

            # Cipher
            ci = data["cipher"]
            state.cipher.mode = ci["mode"]
            state.cipher.key_entropy = ci["key_entropy"]
            state.cipher.rotation_interval = ci["rotation_interval"]
            state.cipher.ghost_sync_detected = ci["ghost_sync_detected"]
            state.cipher.defense_level = ci["defense_level"]
            state.cipher.last_rotation_reason = ci["last_rotation_reason"]
            state.cipher.telemetry_retention_seconds = ci["telemetry_retention_seconds"]
            state.cipher.phantom_node_enabled = ci["phantom_node_enabled"]

            # Codex
            cd = data["codex"]
            state.codex.agent_weights = np.array(cd["agent_weights"], dtype=float)
            state.codex.adaptive_enabled = cd["adaptive_enabled"]
            state.codex.drift_magnitude = cd["drift_magnitude"]
            state.codex.last_mutation_reason = cd["last_mutation_reason"]

            # Posture
            if "posture" in data:
                try:
                    state.posture = Posture(data["posture"])
                except Exception:
                    state.posture = Posture.WATCHER

            print(f"[STATE] Restored organism state from {self.path}")
        except Exception as e:
            print(f"[STATE] Failed to load state: {e}")


# =========================
# Borg Mesh Support Stubs
# =========================

BORG_MESH_CONFIG = {
    "max_corridors": 10000,
    "unknown_bias": 0.6,
}


class MemoryManager:
    def record_mesh_event(self, evt: Dict) -> None:
        print(f"[MEMORY] Mesh event: {evt}")


class BorgCommsRouter:
    def send_secure(self, topic: str, message: str, channel: str) -> None:
        print(f"[COMMS] [{topic}] ({channel}) {message}")


class SecurityGuardian:
    def disassemble(self, snippet: str) -> Dict:
        entropy = min(1.0, (len(set(snippet)) / max(1, len(snippet))) if snippet else 0.0)
        flags = []
        if "login" in snippet or "password" in snippet:
            flags.append("CRED")
        if "token" in snippet:
            flags.append("TOKEN")
        return {"entropy": entropy, "pattern_flags": flags}

    def _pii_count(self, snippet: str) -> int:
        return snippet.count("@") + snippet.count("SSN")

    def reassemble(self, url: str, snippet: str, raw_pii_hits: int) -> Dict:
        if raw_pii_hits > 0 or "CRED" in snippet or "password" in snippet:
            status = "HOSTILE"
        else:
            status = "SAFE_FOR_TRAVEL"
        return {"url": url, "status": status}


def privacy_filter(snippet: str):
    return snippet, False


class BorgMesh:
    def __init__(self, memory: MemoryManager, comms: BorgCommsRouter, guardian: SecurityGuardian):
        self.nodes = {}
        self.edges = set()
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: list):
        risk = self._risk(snippet)
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            if len(self.edges) < self.max_corridors:
                self.edges.add((url, l))
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "discover",
            "url": url,
            "risk": risk,
            "links": len(links),
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:discover", f"{url} risk={risk} links={len(links)}", "Default")

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "build",
            "url": url,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        return True

    def enforce(self, url: str, snippet: str):
        if url not in self.nodes:
            return False
        verdict = self.guardian.reassemble(
            url,
            privacy_filter(snippet or "")[0],
            raw_pii_hits=self.guardian._pii_count(snippet or ""),
        )
        status = verdict.get("status", "HOSTILE")
        self.nodes[url]["state"] = "enforced"
        self.nodes[url]["risk"] = 0 if status == "SAFE_FOR_TRAVEL" else max(50, self.nodes[url]["risk"])
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "enforce",
            "url": url,
            "status": status,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:enforce", f"{url} status={status}", "Default")
        return True

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        corridors = len(self.edges)
        avg_risk = 0.0
        if total > 0:
            avg_risk = sum(n["risk"] for n in self.nodes.values()) / total
        return {
            "total": total,
            "discovered": discovered,
            "built": built,
            "enforced": enforced,
            "corridors": corridors,
            "avg_risk": avg_risk,
        }


@dataclass
class BorgScanEvent:
    url: str
    snippet: str
    links: List[str]


class BorgScanner(threading.Thread):
    def __init__(self, mesh: BorgMesh, in_events: queue.Queue, out_ops: queue.Queue, label="SCANNER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print(f"[BORG-{self.label}] Scanner started.")
        while self.running:
            try:
                ev: BorgScanEvent = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            unseen_links = [
                l for l in ev.links
                if l not in self.mesh.nodes and random.random() < BORG_MESH_CONFIG["unknown_bias"]
            ]
            self.mesh.discover(ev.url, ev.snippet, unseen_links or ev.links)
            self.out_ops.put(("build", ev.url))
            time.sleep(random.uniform(0.2, 0.6))


class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print(f"[BORG-{self.label}] Worker started.")
        while self.running:
            try:
                op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if op == "build":
                if self.mesh.build(url):
                    self.ops_q.put(("enforce", url))
            elif op == "enforce":
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))


class BorgEnforcer(threading.Thread):
    def __init__(self, mesh: BorgMesh, guardian: SecurityGuardian, label="ENFORCER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.guardian = guardian
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print(f"[BORG-{self.label}] Enforcer started.")
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    self.mesh.enforce(url, snippet="")
            time.sleep(1.2)


# =========================
# Real Telemetry Collector
# =========================


@dataclass
class RealTelemetryCollector:
    last_input_time: float = field(default_factory=time.time)

    def _cpu_load(self) -> float:
        return psutil.cpu_percent(interval=0.0) / 100.0

    def _memory_load(self) -> float:
        mem = psutil.virtual_memory()
        return mem.percent / 100.0

    def _disk_health(self) -> float:
        try:
            parts = psutil.disk_partitions(all=False)
            if not parts:
                return 0.0
        except Exception:
            return 0.0
        usage_vals = []
        for p in parts:
            try:
                u = psutil.disk_usage(p.mountpoint)
                usage_vals.append(u.percent / 100.0)
            except PermissionError:
                continue
        if not usage_vals:
            return 0.0
        return sum(usage_vals) / len(usage_vals)

    def _net_activity(self) -> float:
        try:
            c1 = psutil.net_io_counters()
            time.sleep(0.05)
            c2 = psutil.net_io_counters()
            delta = (c2.bytes_sent + c2.bytes_recv) - (c1.bytes_sent + c1.bytes_recv)
            return min(1.0, delta / (1024 * 50))
        except Exception:
            return 0.0

    def _foreground_app(self) -> str:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "Unknown"
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()
            return title or "Unknown"
        except Exception:
            return "Unknown"

    def _input_activity_level(self) -> float:
        idle = time.time() - self.last_input_time
        if idle <= 1.0:
            return 1.0
        if idle >= 60.0:
            return 0.0
        return 1.0 - idle / 60.0

    def generate_telemetry_frame(self) -> Dict:
        cpu = self._cpu_load()
        mem = self._memory_load()
        disk = self._disk_health()
        net = self._net_activity()
        activity = self._input_activity_level()
        fg_app = self._foreground_app()

        anomaly_risk = max(cpu, mem, disk)
        hive_divergence = net * 0.2
        ghost_sync = False
        collective_health = max(0.0, 1.0 - anomaly_risk * 0.7)

        frame = {
            "anomaly_risk": float(anomaly_risk),
            "hive_divergence": float(hive_divergence),
            "ghost_sync": bool(ghost_sync),
            "collective_health": float(collective_health),
            "cpu_load": float(cpu),
            "mem_load": float(mem),
            "disk_load": float(disk),
            "net_activity": float(net),
            "activity_level": float(activity),
            "foreground_app": fg_app,
        }
        return frame


# =========================
# MagicBox Organism Model
# =========================


@dataclass
class MagicBoxBrain:
    hybrid_core: HybridBrainCore
    judgment_engine: JudgmentEngine
    situational_cortex: SituationalCortex
    predictive_engine: PredictiveIntelligence

    def ingest_bot_report(self, anomaly_score: float, load: float, volatility: float) -> None:
        self.hybrid_core.cognitive_load = load
        self.hybrid_core.volatility = volatility
        self.predictive_engine.anomaly_risk = anomaly_score


@dataclass
class MagicBoxQueen:
    collective: CollectiveHealth

    def process_hive_event(self, risk: float, density: float, agreement: float, divergence_tags: List[str]) -> None:
        self.collective.collective_risk_score = risk
        self.collective.hive_density = density
        self.collective.node_agreement = agreement
        self.collective.divergence_patterns = divergence_tags


@dataclass
class MagicBoxCodex:
    mutation_engine: CodexMutationEngine

    def rewrite_purge_logic(self, frame: Dict, cipher_engine: AutonomousCipherEngine) -> None:
        w = self.mutation_engine.agent_weights
        anomaly = frame.get("anomaly_risk", 0.0)
        divergence = frame.get("hive_divergence", 0.0)
        ghost_sync = frame.get("ghost_sync", False)
        health = frame.get("collective_health", 1.0)

        score = w[0] * anomaly + w[1] * divergence + w[2] * (1.0 - health)
        base_retention = cipher_engine.telemetry_retention_seconds
        if ghost_sync or score > 0.2:
            new_retention = max(60, int(base_retention * 0.25))
        elif score > 0.0:
            new_retention = max(120, int(base_retention * 0.5))
        else:
            new_retention = min(7200, int(base_retention * 1.1))

        cipher_engine.telemetry_retention_seconds = new_retention

        print(
            f"[CODEX] Purge logic rewritten: score={score:.3f}, "
            f"ghost_sync={ghost_sync}, new_retention={new_retention}s"
        )

    def apply_mutation(self, frame: Dict) -> None:
        self.mutation_engine.mutate_based_on_frame(frame)


@dataclass
class MagicBoxCipher:
    engine: AutonomousCipherEngine

    def process_security_signals(self, frame: Dict) -> None:
        if frame.get("ghost_sync", False):
            print("[ACE] Ghost sync detected. Escalating defense and mutating retention.")
            self.engine.handle_ghost_sync()
        else:
            if np.random.rand() < 0.1:
                self.engine.rotate_keys("Background entropy maintenance")


@dataclass
class MagicBoxMeshOrgans:
    mesh: BorgMesh
    scanner: BorgScanner
    worker: BorgWorker
    enforcer: BorgEnforcer
    scan_queue: queue.Queue
    ops_queue: queue.Queue


@dataclass
class MagicBox:
    state: MissionControlState
    brain: MagicBoxBrain
    queen: MagicBoxQueen
    bots: RealTelemetryCollector
    cipher: MagicBoxCipher
    codex: MagicBoxCodex
    mesh_organs: MagicBoxMeshOrgans
    _persistence: Optional[StatePersistenceManager] = None

    def _purge_daemon(self, threat_score: float) -> None:
        if threat_score < 0.2:
            return
        if threat_score < 0.5:
            self.state.cipher.telemetry_retention_seconds = max(
                300, int(self.state.cipher.telemetry_retention_seconds * 0.9)
            )
            return
        self.state.cipher.telemetry_retention_seconds = max(
            120, int(self.state.cipher.telemetry_retention_seconds * 0.5)
        )
        self.state.cipher.escalate_defense("Threat score critical")
        self.state.cortex.environment = EnvironmentState.DANGER

    def tick(self) -> None:
        frame = self.bots.generate_telemetry_frame()

        # Real signals into brain/cortex
        self.state.brain.cognitive_load = frame["cpu_load"]
        self.state.brain.volatility = frame["net_activity"]
        self.state.cortex.opportunity_score = max(0.0, 1.0 - frame["mem_load"])

        fg = frame["foreground_app"].lower()
        if ("game" in fg or fg.endswith(".exe")) and self.state.cortex.mission == MissionType.AUTO:
            self.state.cortex.mission = MissionType.LEARN
        if ("game" in fg or fg.endswith(".exe")) and self.state.brain.mode == BrainMode.AUTO:
            self.state.brain.mode = BrainMode.REFLEX

        # Posture-aware tuning
        if self.state.posture == Posture.GUARDIAN:
            self.state.prediction.adjust_anomaly_sensitivity(0.05)
            self.state.cortex.adjust_risk_tolerance(-0.05)
            if self.state.cortex.risk_score > 0.4 and self.state.cortex.mission == MissionType.AUTO:
                self.state.cortex.mission = MissionType.PROTECT
        elif self.state.posture == Posture.WATCHER:
            if self.state.prediction.anomaly_risk < 0.2 and self.state.cortex.mission == MissionType.AUTO:
                self.state.cortex.mission = MissionType.LEARN
            self.state.cortex.adjust_risk_tolerance(0.01)
        elif self.state.posture == Posture.CHAMELEON:
            self.state.prediction.adjust_anomaly_sensitivity(0.03)
            self.state.cortex.adjust_risk_tolerance(-0.03)
            if self.state.cortex.risk_score > 0.4 and self.state.cortex.mission == MissionType.AUTO:
                self.state.cortex.mission = MissionType.PROTECT

        self.brain.ingest_bot_report(
            anomaly_score=frame["anomaly_risk"],
            load=frame["cpu_load"],
            volatility=frame["net_activity"],
        )

        self.codex.apply_mutation(frame)
        self.codex.rewrite_purge_logic(frame, self.cipher.engine)
        self.cipher.process_security_signals(frame)

        hive_risk = frame["anomaly_risk"] * 0.7
        hive_density = 0.5 + float(np.random.rand()) * 0.5
        hive_agreement = 0.8 - frame["hive_divergence"]
        divergence_tags = []
        if frame["ghost_sync"]:
            divergence_tags.append("phantom_node")
        self.queen.process_hive_event(hive_risk, hive_density, hive_agreement, divergence_tags)

        self.state.prediction.hive_risk = hive_risk
        self.state.prediction.collective_health_score = frame["collective_health"]
        self.state.prediction.health_trend = frame["collective_health"] - 0.8

        if frame["ghost_sync"]:
            self.state.cortex.environment = EnvironmentState.DANGER
            self.state.cortex.risk_score = min(1.0, self.state.cortex.risk_score + 0.3)
        else:
            self.state.cortex.risk_score = max(0.0, self.state.cortex.risk_score - 0.02)

        url = f"https://node.example/{int(time.time())}"
        links = [f"https://node.example/{int(time.time())+i}" for i in range(1, 4)]
        snippet = f"cpu={frame['cpu_load']:.2f} mem={frame['mem_load']:.2f} app={frame['foreground_app']}"
        scan_event = BorgScanEvent(url=url, snippet=snippet, links=links)
        try:
            self.mesh_organs.scan_queue.put_nowait(scan_event)
        except queue.Full:
            pass

        mesh_stats = self.mesh_organs.mesh.stats()
        avg_risk = mesh_stats["avg_risk"]
        self.state.prediction.drive_system_risk = avg_risk / 100.0
        self.state.collective.collective_risk_score = max(
            self.state.collective.collective_risk_score,
            avg_risk / 100.0,
        )

        # Combined threat score
        threat_score = (
            0.3 * self.state.prediction.anomaly_risk +
            0.2 * self.state.prediction.drive_system_risk +
            0.2 * self.state.prediction.hive_risk +
            0.2 * self.state.collective.collective_risk_score +
            0.1 * (self.state.cipher.defense_level / 5.0)
        )

        if threat_score < 0.2:
            self.state.prediction.forecast_summary = "CALM"
        elif threat_score < 0.5:
            self.state.prediction.forecast_summary = "TENSE"
        else:
            self.state.prediction.forecast_summary = "CRITICAL"

        self._purge_daemon(threat_score)


# =========================
# Background Tick & Health Monitor
# =========================


class BackgroundTickEngine(threading.Thread):
    def __init__(self, magicbox: MagicBox, interval: float = 1.0, label: str = "BACKGROUND_TICK"):
        super().__init__(daemon=True)
        self.magicbox = magicbox
        self.interval = interval
        self.label = label
        self.running = True
        self._last_save = time.time()
        self._save_interval = 30.0

    def stop(self):
        self.running = False

    def run(self):
        print(f"[{self.label}] Tick engine started (interval={self.interval}s)")
        while self.running:
            try:
                self.magicbox.tick()
                now = time.time()
                if now - self._last_save >= self._save_interval:
                    if self.magicbox._persistence is not None:
                        self.magicbox._persistence.save(self.magicbox.state)
                    self._last_save = now
            except Exception as e:
                print(f"[{self.label}] Error during tick: {e}")
            time.sleep(self.interval)


class ThreadHealthMonitor(threading.Thread):
    def __init__(self, magicbox: MagicBox, tick_engine: BackgroundTickEngine, interval: float = 5.0):
        super().__init__(daemon=True)
        self.magicbox = magicbox
        self.tick_engine = tick_engine
        self.interval = interval
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print("[HEALTH] Thread health monitor started.")
        while self.running:
            mesh_org = self.magicbox.mesh_organs
            for name, t in [
                ("SCANNER", mesh_org.scanner),
                ("WORKER", mesh_org.worker),
                ("ENFORCER", mesh_org.enforcer),
                ("TICK", self.tick_engine),
            ]:
                if not t.is_alive():
                    print(f"[HEALTH] WARNING: {name} thread not alive. Attempting restart...")
                    if name == "SCANNER":
                        mesh_org.scanner = BorgScanner(
                            mesh=mesh_org.mesh,
                            in_events=mesh_org.scan_queue,
                            out_ops=mesh_org.ops_queue,
                        )
                        mesh_org.scanner.start()
                    elif name == "WORKER":
                        mesh_org.worker = BorgWorker(
                            mesh=mesh_org.mesh,
                            ops_q=mesh_org.ops_queue,
                        )
                        mesh_org.worker.start()
                    elif name == "ENFORCER":
                        mesh_org.enforcer = BorgEnforcer(
                            mesh=mesh_org.mesh,
                            guardian=SecurityGuardian(),
                        )
                        mesh_org.enforcer.start()
                    elif name == "TICK":
                        self.tick_engine = BackgroundTickEngine(self.magicbox, self.tick_engine.interval)
                        self.tick_engine.start()
            time.sleep(self.interval)


# =========================
# Alien Glyph System for LCARS-DARK
# =========================

ALIEN_MAP = {
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               ["⟐","⟟","⟊","⟒","⟓","⟔","⟕","⟖","⟗","⟘","⟙","⟚","⟛","⟜","⟝","⟞",
                "⟠","⟡","⟢","⟣","⟤","⟥","⟦","⟧","⟨","⟩"])),
    **dict(zip("0123456789", ["⟪","⟫","⟬","⟭","⟮","⟯","⟰","⟱","⟲","⟳"])),
    " ": "·"
}

REVERSE_ALIEN_MAP = {v: k for k, v in ALIEN_MAP.items()}


def to_alien(text: str) -> str:
    return "".join(ALIEN_MAP.get(ch.upper(), ch) for ch in text)


# =========================
# LCARS-DARK Color Palette
# =========================

PALETTE = {
    "bg": "#0a0a12",
    "rail": "#1a1a2a",
    "rail_accent": "#3a3a5a",
    "pod_bg": "#11111a",
    "pod_border": "#2a2a3a",
    "text": "#c8c8ff",
    "glyph": "#7f7fff",
    "threat_calm": "#202c20",
    "threat_tense": "#2c2620",
    "threat_critical": "#3c2020",
}


# =========================
# LCARS-DARK Defense UI
# =========================

import tkinter as tk
from tkinter import ttk


class LCARSDefenseUI(tk.Tk):
    def __init__(self, magicbox: MagicBox, state: MissionControlState,
                 tick_engine: Optional[BackgroundTickEngine] = None,
                 health_monitor: Optional[ThreadHealthMonitor] = None):
        super().__init__()
        self.magicbox = magicbox
        self.state = state
        self.tick_engine = tick_engine
        self.health_monitor = health_monitor

        self.title("LCARS-DARK ASI Defense Interface")
        self.configure(bg=PALETTE["bg"])
        self.geometry("1600x900")

        self._build_layout()
        self._animate()

    # Layout
    def _build_layout(self):
        # TOP RAIL
        self.top_rail = tk.Frame(self, bg=PALETTE["rail"], height=100)
        self.top_rail.place(relx=0, rely=0, relwidth=1, relheight=0.12)

        # LEFT RAIL
        self.left_rail = tk.Frame(self, bg=PALETTE["rail"], width=200)
        self.left_rail.place(relx=0, rely=0.12, relwidth=0.14, relheight=0.80)

        # BOTTOM RAIL
        self.bottom_rail = tk.Frame(self, bg=PALETTE["rail"], height=80)
        self.bottom_rail.place(relx=0, rely=0.92, relwidth=1, relheight=0.08)

        # PODS (main 2x2 grid)
        self.pod_brain = self._make_pod(0.14, 0.12, 0.42, 0.40, "HYBRID BRAIN / CORTEX")
        self.pod_ace = self._make_pod(0.56, 0.12, 0.44, 0.40, "ACE / CODEX DEFENSE CORE")
        self.pod_mesh = self._make_pod(0.14, 0.54, 0.42, 0.38, "BORG MESH / HIVE")
        self.pod_snapshot = self._make_pod(0.56, 0.54, 0.44, 0.38, "SYSTEM SNAPSHOT")

        self._populate_top_rail()
        self._populate_left_rail()
        self._populate_pods()
        self._populate_bottom_rail()

    def _make_pod(self, x, y, w, h, title):
        pod = tk.Frame(self, bg=PALETTE["pod_bg"],
                       highlightbackground=PALETTE["pod_border"],
                       highlightthickness=2)
        pod.place(relx=x, rely=y, relwidth=w, relheight=h)

        lbl = tk.Label(pod, text=title, fg=PALETTE["text"], bg=PALETTE["pod_bg"],
                       font=("Consolas", 14, "bold"))
        lbl.pack(anchor="nw", padx=10, pady=5)

        glyph = tk.Label(pod, text=to_alien(title), fg=PALETTE["glyph"],
                         bg=PALETTE["pod_bg"], font=("Consolas", 11))
        glyph.pack(anchor="nw", padx=10)

        return pod

    def _populate_top_rail(self):
        self.threat_label = tk.Label(self.top_rail, text="THREAT: CALM",
                                     fg=PALETTE["text"], bg=PALETTE["rail"],
                                     font=("Consolas", 16, "bold"))
        self.threat_label.place(relx=0.02, rely=0.2)

        self.sparkline = tk.Label(self.top_rail, text="░░░░░░░░░░░░░░░░░░",
                                  fg=PALETTE["glyph"], bg=PALETTE["rail"],
                                  font=("Consolas", 14))
        self.sparkline.place(relx=0.30, rely=0.2)

        self.clock = tk.Label(self.top_rail, text="", fg=PALETTE["text"],
                              bg=PALETTE["rail"], font=("Consolas", 16))
        self.clock.place(relx=0.85, rely=0.2)

    def _populate_left_rail(self):
        tk.Label(self.left_rail, text="NODE: AURELION",
                 fg=PALETTE["text"], bg=PALETTE["rail"],
                 font=("Consolas", 16, "bold")).pack(pady=10)

        tk.Label(self.left_rail, text=to_alien("AURELION"),
                 fg=PALETTE["glyph"], bg=PALETTE["rail"],
                 font=("Consolas", 14)).pack()

        self.posture_label = tk.Label(self.left_rail, text="POSTURE: WATCHER",
                                      fg=PALETTE["text"], bg=PALETTE["rail"],
                                      font=("Consolas", 14))
        self.posture_label.pack(pady=20)

    def _populate_bottom_rail(self):
        self.event_label = tk.Label(self.bottom_rail, text="Events: (defense actions will appear here)",
                                    fg=PALETTE["text"], bg=PALETTE["rail"],
                                    font=("Consolas", 11), anchor="w", justify="left")
        self.event_label.place(relx=0.02, rely=0.2, relwidth=0.96)

    def _populate_pods(self):
        # Brain pod
        self.lbl_brain = tk.Label(
            self.pod_brain,
            text="",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 11),
            justify="left",
        )
        self.lbl_brain.pack(anchor="nw", padx=20, pady=10)

        # ACE/Codex pod
        self.lbl_ace = tk.Label(
            self.pod_ace,
            text="",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 11),
            justify="left",
        )
        self.lbl_ace.pack(anchor="nw", padx=20, pady=10)

        # Mesh pod
        self.lbl_mesh = tk.Label(
            self.pod_mesh,
            text="",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 11),
            justify="left",
        )
        self.lbl_mesh.pack(anchor="nw", padx=20, pady=10)

        # Snapshot pod
        self.lbl_snapshot = tk.Label(
            self.pod_snapshot,
            text="",
            fg=PALETTE["text"],
            bg=PALETTE["pod_bg"],
            font=("Consolas", 11),
            justify="left",
        )
        self.lbl_snapshot.pack(anchor="nw", padx=20, pady=10)

    # Animation / state refresh
    def _animate(self):
        self.clock.config(text=time.strftime("%H:%M:%S"))

        s = self.state

        forecast = getattr(s.prediction, "forecast_summary", "CALM")
        self.threat_label.config(text=f"THREAT: {forecast}")

        bars = {"CALM": "░", "TENSE": "▒", "CRITICAL": "▓"}
        glyph = bars.get(forecast, "░")
        self.sparkline.config(text=glyph * 20)

        self.posture_label.config(text=f"POSTURE: {s.posture.value.upper()}")

        brain = s.brain
        cortex = s.cortex
        brain_text = (
            f"Mode: {brain.mode.value}\n"
            f"Trust: {brain.trust:.2f}\n"
            f"Volatility: {brain.volatility:.2f}\n"
            f"Cognitive Load: {brain.cognitive_load:.2f}\n"
            f"Sensitivity: {brain.sensitivity:.2f}\n\n"
            f"Mission: {cortex.mission.value}\n"
            f"Environment: {cortex.environment.value}\n"
            f"Risk Score: {cortex.risk_score:.2f}\n"
            f"Opportunity: {cortex.opportunity_score:.2f}\n"
            f"Risk Tolerance: {cortex.risk_tolerance:.2f}\n"
        )
        self.lbl_brain.config(text=brain_text)

        ace = s.cipher
        codex = s.codex
        ace_text = (
            f"ACE Mode: {ace.mode}\n"
            f"Defense Level: {ace.defense_level}\n"
            f"Key Entropy: {ace.key_entropy:.2f}\n"
            f"Retention: {ace.telemetry_retention_seconds}s\n"
            f"Ghost Sync: {ace.ghost_sync_detected}\n"
            f"Phantom Node: {ace.phantom_node_enabled}\n"
            f"Last Rotation: {ace.last_rotation_reason}\n\n"
            f"Codex Drift: {codex.drift_magnitude:.4f}\n"
            f"Last Mutation: {codex.last_mutation_reason}\n"
            f"Adaptive: {codex.adaptive_enabled}\n"
        )
        self.lbl_ace.config(text=ace_text)

        mesh_stats = self.magicbox.mesh_organs.mesh.stats()
        coll = s.collective
        mesh_text = (
            f"Nodes: {mesh_stats['total']} "
            f"(D:{mesh_stats['discovered']} B:{mesh_stats['built']} E:{mesh_stats['enforced']})\n"
            f"Corridors: {mesh_stats['corridors']}\n"
            f"Avg Mesh Risk: {mesh_stats['avg_risk']:.1f}\n\n"
            f"Hive Density: {coll.hive_density:.2f}\n"
            f"Node Agreement: {coll.node_agreement:.2f}\n"
            f"Collective Risk: {coll.collective_risk_score:.2f}\n"
            f"Hive Mode: {coll.hive_mode}\n"
        )
        self.lbl_mesh.config(text=mesh_text)

        frame = self.magicbox.bots.generate_telemetry_frame()
        cpu = frame["cpu_load"]
        mem = frame["mem_load"]
        disk = frame["disk_load"]
        net = frame["net_activity"]
        app = frame["foreground_app"]
        activity = frame["activity_level"]

        def spark(v, length=12):
            v = max(0.0, min(1.0, v))
            high = int(round(v * length))
            med = max(0, high // 2)
            low = length - high
            return "░" * low + "▒" * med + "▓" * (high - med)

        pulse = max(cpu, mem, net)
        pulse_glyph = "·" if pulse < 0.3 else "●" if pulse < 0.7 else "⬤"
        pulse_bar = pulse_glyph * 10

        snapshot_text = (
            f"CPU:  {cpu:.2f}  {spark(cpu)}\n"
            f"RAM:  {mem:.2f}  {spark(mem)}\n"
            f"DISK: {disk:.2f} {spark(disk)}\n"
            f"NET:  {net:.2f}  {spark(net)}\n"
            f"APP:  {app!r}\n"
            f"ACT:  {activity:.2f}\n"
            f"Pulse: {pulse_bar}\n"
        )
        self.lbl_snapshot.config(text=snapshot_text)

        if forecast == "CALM":
            color = PALETTE["threat_calm"]
        elif forecast == "TENSE":
            color = PALETTE["threat_tense"]
        else:
            color = PALETTE["threat_critical"]
        self.top_rail.configure(bg=color)
        self.clock.configure(bg=color)
        self.threat_label.configure(bg=color)
        self.sparkline.configure(bg=color)

        self.after(500, self._animate)


# =========================
# Bootstrap: build MagicBox and launch LCARS-DARK
# =========================


def build_magicbox():
    PASSPHRASE = "change_this_to_your_secret"
    state = MissionControlState()

    persistence = StatePersistenceManager(passphrase=PASSPHRASE)
    persistence.load_into(state)

    brain = MagicBoxBrain(
        hybrid_core=state.brain,
        judgment_engine=state.judgment,
        situational_cortex=state.cortex,
        predictive_engine=state.prediction,
    )

    queen = MagicBoxQueen(collective=state.collective)
    bots = RealTelemetryCollector()
    cipher = MagicBoxCipher(engine=state.cipher)
    codex = MagicBoxCodex(mutation_engine=state.codex)

    memory_mgr = MemoryManager()
    comms = BorgCommsRouter()
    guardian = SecurityGuardian()
    mesh = BorgMesh(memory=memory_mgr, comms=comms, guardian=guardian)
    scan_q = queue.Queue(maxsize=1000)
    ops_q = queue.Queue(maxsize=1000)

    scanner = BorgScanner(mesh=mesh, in_events=scan_q, out_ops=ops_q)
    worker = BorgWorker(mesh=mesh, ops_q=ops_q)
    enforcer = BorgEnforcer(mesh=mesh, guardian=guardian)

    scanner.start()
    worker.start()
    enforcer.start()

    mesh_organs = MagicBoxMeshOrgans(
        mesh=mesh,
        scanner=scanner,
        worker=worker,
        enforcer=enforcer,
        scan_queue=scan_q,
        ops_queue=ops_q,
    )

    magicbox = MagicBox(
        state=state,
        brain=brain,
        queen=queen,
        bots=bots,
        cipher=cipher,
        codex=codex,
        mesh_organs=mesh_organs,
        _persistence=persistence,
    )

    return magicbox, state


if __name__ == "__main__":
    magicbox, state = build_magicbox()

    tick_engine = BackgroundTickEngine(magicbox, interval=1.0)
    tick_engine.start()

    health_monitor = ThreadHealthMonitor(magicbox, tick_engine, interval=5.0)
    health_monitor.start()

    ui = LCARSDefenseUI(magicbox, state, tick_engine, health_monitor)

    def on_close():
        print("[MAIN] Shutting down. Saving state...")
        if magicbox._persistence is not None:
            magicbox._persistence.save(magicbox.state)
        tick_engine.stop()
        health_monitor.stop()
        ui.destroy()

    ui.protocol("WM_DELETE_WINDOW", on_close)
    ui.mainloop()

