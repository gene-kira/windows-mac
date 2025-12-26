"""
MagicBox ASI Oversight — Mission Control + ACE + Codex Mutation + Borg Mesh
Persistent, self-monitoring organism with GUI + background engine + real telemetry
and ASI Memory View with System Snapshot.
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
    WATCHER = "Watcher"      # calm observer
    GUARDIAN = "Guardian"    # defensive, cautious
    CHAMELEON = "Chameleon"  # redacted/stealthy presentation


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
    forecast_summary: str = "Stable"

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

        # Use real signals
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
# GUI (Tkinter)
# =========================

import tkinter as tk
from tkinter import ttk

style = ttk.Style()


class MissionControlPane(ttk.Frame):
    def __init__(self, master, state: MissionControlState, magicbox: MagicBox, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.magicbox = magicbox
        self._build_layout()
        self._start_tick_loop()

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.brain_frame = self._build_brain_core(self)
        self.brain_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.judgment_frame = self._build_judgment_engine(self)
        self.judgment_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.cortex_frame = self._build_cortex(self)
        self.cortex_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.predict_frame = self._build_predictive(self)
        self.predict_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.collective_frame = self._build_collective(self)
        self.collective_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.ace_frame = self._build_cipher_engine(self)
        self.ace_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        self.mesh_frame = self._build_mesh_panel(self)
        self.mesh_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        self.codex_frame = self._build_codex_panel(self)
        self.codex_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)

        self.command_frame = self._build_command_bar(self)
        self.command_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        self.dialog_frame = self._build_dialogue(self)
        self.dialog_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        for r in range(6):
            self.rowconfigure(r, weight=1)

    # ------- Brain Core -------

    def _build_brain_core(self, parent):
        frame = ttk.LabelFrame(parent, text="🧠 Hybrid Brain Core")
        s = self.state.brain

        self.lbl_mode = ttk.Label(frame, text=f"Mode: {s.mode.value}")
        self.lbl_mode.grid(row=0, column=0, sticky="w")
        self.lbl_volatility = ttk.Label(frame, text=f"Volatility: {s.volatility:.2f}")
        self.lbl_volatility.grid(row=1, column=0, sticky="w")
        self.lbl_trust = ttk.Label(frame, text=f"Trust: {s.trust:.2f}")
        self.lbl_trust.grid(row=2, column=0, sticky="w")
        self.lbl_load = ttk.Label(frame, text=f"Cognitive load: {s.cognitive_load:.2f}")
        self.lbl_load.grid(row=3, column=0, sticky="w")
        self.lbl_sens = ttk.Label(frame, text=f"Sensitivity: {s.sensitivity:.2f}")
        self.lbl_sens.grid(row=4, column=0, sticky="w")

        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Label(mode_frame, text="Mode override:").grid(row=0, column=0, columnspan=4, sticky="w")

        def set_mode(mode):
            self.state.brain.set_mode(mode)
            self.lbl_mode.config(text=f"Mode: {self.state.brain.mode.value}")

        ttk.Button(mode_frame, text="Stability", command=lambda: set_mode(BrainMode.STABILITY)).grid(row=1, column=0)
        ttk.Button(mode_frame, text="Reflex", command=lambda: set_mode(BrainMode.REFLEX)).grid(row=1, column=1)
        ttk.Button(mode_frame, text="Exploration", command=lambda: set_mode(BrainMode.EXPLORATION)).grid(row=1, column=2)
        ttk.Button(mode_frame, text="Auto", command=lambda: set_mode(BrainMode.AUTO)).grid(row=1, column=3)

        ctl_frame = ttk.Frame(frame)
        ctl_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Button(ctl_frame, text="Reduce sensitivity", command=lambda: self._adjust_sensitivity(-0.1)).grid(row=0, column=0)
        ttk.Button(ctl_frame, text="Increase sensitivity", command=lambda: self._adjust_sensitivity(0.1)).grid(row=0, column=1)
        ttk.Button(ctl_frame, text="Be more cautious", command=lambda: self._adjust_trust(-0.1)).grid(row=1, column=0)
        ttk.Button(ctl_frame, text="Be more assertive", command=lambda: self._adjust_trust(0.1)).grid(row=1, column=1)

        return frame

    def _adjust_sensitivity(self, delta: float):
        self.state.brain.adjust_sensitivity(delta)
        self.lbl_sens.config(text=f"Sensitivity: {self.state.brain.sensitivity:.2f}")

    def _adjust_trust(self, delta: float):
        self.state.brain.calibrate_trust(delta)
        self.lbl_trust.config(text=f"Trust: {self.state.brain.trust:.2f}")

    def _refresh_brain(self):
        s = self.state.brain
        self.lbl_mode.config(text=f"Mode: {s.mode.value}")
        self.lbl_volatility.config(text=f"Volatility: {s.volatility:.2f}")
        self.lbl_trust.config(text=f"Trust: {s.trust:.2f}")
        self.lbl_load.config(text=f"Cognitive load: {s.cognitive_load:.2f}")
        self.lbl_sens.config(text=f"Sensitivity: {s.sensitivity:.2f}")

    # ------- Judgment -------

    def _build_judgment_engine(self, parent):
        frame = ttk.LabelFrame(parent, text="⚖️ Judgment Engine")

        self.lbl_jconf = ttk.Label(frame, text=f"Judgment confidence: {self.state.judgment.judgment_confidence:.2f}")
        self.lbl_jconf.grid(row=0, column=0, sticky="w")
        self.lbl_ratio = ttk.Label(
            frame,
            text=f"Good / Bad: {self.state.judgment.good_outcomes} / {self.state.judgment.bad_outcomes}",
        )
        self.lbl_ratio.grid(row=1, column=0, sticky="w")
        self.lbl_bias = ttk.Label(frame, text=f"Bias drift: {self.state.judgment.bias_drift:.2f}")
        self.lbl_bias.grid(row=2, column=0, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, sticky="w", pady=(5, 0))

        ttk.Button(btn_frame, text="Reinforce (good)", command=lambda: self._record_outcome(True)).grid(row=0, column=0)
        ttk.Button(btn_frame, text="Correct (bad)", command=lambda: self._record_outcome(False)).grid(row=0, column=1)
        ttk.Button(btn_frame, text="Reset bias", command=self._reset_bias).grid(row=1, column=0)
        ttk.Button(btn_frame, text="Freeze learning", command=lambda: self._set_freeze(True)).grid(row=1, column=1)
        ttk.Button(btn_frame, text="Unfreeze", command=lambda: self._set_freeze(False)).grid(row=1, column=2)
        ttk.Button(btn_frame, text="Accelerate learning", command=self._accelerate_learning).grid(row=2, column=0)

        return frame

    def _record_outcome(self, good: bool):
        self.state.judgment.record_outcome(good)
        self._refresh_judgment()

    def _reset_bias(self):
        self.state.judgment.reset_bias()
        self._refresh_judgment()

    def _set_freeze(self, frozen: bool):
        self.state.judgment.freeze_learning(frozen)
        self._refresh_judgment()

    def _accelerate_learning(self):
        self.state.judgment.accelerate_learning()
        self._refresh_judgment()

    def _refresh_judgment(self):
        j = self.state.judgment
        self.lbl_jconf.config(text=f"Judgment confidence: {j.judgment_confidence:.2f}")
        self.lbl_ratio.config(text=f"Good / Bad: {j.good_outcomes} / {j.bad_outcomes}")
        self.lbl_bias.config(text=f"Bias drift: {j.bias_drift:.2f}")

    # ------- Cortex -------

    def _build_cortex(self, parent):
        frame = ttk.LabelFrame(parent, text="🕸️ Situational Awareness Cortex")

        c = self.state.cortex
        self.lbl_mission = ttk.Label(frame, text=f"Mission: {c.mission.value}")
        self.lbl_mission.grid(row=0, column=0, sticky="w")
        self.lbl_env = ttk.Label(frame, text=f"Environment: {c.environment.value}")
        self.lbl_env.grid(row=1, column=0, sticky="w")
        self.lbl_opp = ttk.Label(frame, text=f"Opportunity: {c.opportunity_score:.2f}")
        self.lbl_opp.grid(row=2, column=0, sticky="w")
        self.lbl_risk = ttk.Label(frame, text=f"Risk: {c.risk_score:.2f}")
        self.lbl_risk.grid(row=3, column=0, sticky="w")
        self.lbl_anticipation = ttk.Label(frame, text=f"Anticipation: {c.anticipation}")
        self.lbl_anticipation.grid(row=4, column=0, sticky="w")
        self.lbl_posture = ttk.Label(frame, text=f"Posture: {self.state.posture.value}")
        self.lbl_posture.grid(row=5, column=0, sticky="w")

        mission_frame = ttk.Frame(frame)
        mission_frame.grid(row=6, column=0, sticky="w", pady=(5, 0))
        ttk.Label(mission_frame, text="Mission override:").grid(row=0, column=0, columnspan=4, sticky="w")

        def set_mission(m):
            self.state.cortex.set_mission(m)
            self._refresh_cortex()

        ttk.Button(mission_frame, text="Force PROTECT", command=lambda: set_mission(MissionType.PROTECT)).grid(row=1, column=0)
        ttk.Button(mission_frame, text="Force LEARN", command=lambda: set_mission(MissionType.LEARN)).grid(row=1, column=1)
        ttk.Button(mission_frame, text="Force OPTIMIZE", command=lambda: set_mission(MissionType.OPTIMIZE)).grid(row=1, column=2)
        ttk.Button(mission_frame, text="AUTO", command=lambda: set_mission(MissionType.AUTO)).grid(row=1, column=3)

        tune_frame = ttk.Frame(frame)
        tune_frame.grid(row=7, column=0, sticky="w", pady=(5, 0))
        ttk.Button(tune_frame, text="Accept more risk", command=lambda: self._adjust_risk_tolerance(0.1)).grid(row=0, column=0)
        ttk.Button(tune_frame, text="Be more conservative", command=lambda: self._adjust_risk_tolerance(-0.1)).grid(row=0, column=1)
        ttk.Button(tune_frame, text="Prioritize learning windows", command=lambda: self._adjust_opp_focus(0.1)).grid(row=1, column=0)
        ttk.Button(tune_frame, text="Ignore learning windows", command=lambda: self._adjust_opp_focus(-0.1)).grid(row=1, column=1)

        return frame

    def _adjust_risk_tolerance(self, delta: float):
        self.state.cortex.adjust_risk_tolerance(delta)
        self._refresh_cortex()

    def _adjust_opp_focus(self, delta: float):
        self.state.cortex.adjust_opportunity_focus(delta)
        self._refresh_cortex()

    def _refresh_cortex(self):
        c = self.state.cortex
        self.lbl_mission.config(text=f"Mission: {c.mission.value}")
        self.lbl_env.config(text=f"Environment: {c.environment.value}")
        self.lbl_opp.config(text=f"Opportunity: {c.opportunity_score:.2f}")
        self.lbl_risk.config(text=f"Risk: {c.risk_score:.2f}")
        self.lbl_anticipation.config(text=f"Anticipation: {c.anticipation}")
        self.lbl_posture.config(text=f"Posture: {self.state.posture.value}")

    # ------- Predictive -------

    def _build_predictive(self, parent):
        frame = ttk.LabelFrame(parent, text="🔮 Predictive Intelligence")

        p = self.state.prediction
        self.lbl_anomaly = ttk.Label(frame, text=f"Anomaly risk: {p.anomaly_risk:.2f}")
        self.lbl_anomaly.grid(row=0, column=0, sticky="w")
        self.lbl_drive = ttk.Label(frame, text=f"Drive/System risk: {p.drive_system_risk:.2f}")
        self.lbl_drive.grid(row=1, column=0, sticky="w")
        self.lbl_hive_risk = ttk.Label(frame, text=f"Hive risk: {p.hive_risk:.2f}")
        self.lbl_hive_risk.grid(row=2, column=0, sticky="w")
        self.lbl_health = ttk.Label(frame, text=f"Collective health: {p.collective_health_score:.2f}")
        self.lbl_health.grid(row=3, column=0, sticky="w")
        self.lbl_trend = ttk.Label(frame, text=f"Health trend: {p.health_trend:.2f}")
        self.lbl_trend.grid(row=4, column=0, sticky="w")
        self.lbl_forecast = ttk.Label(frame, text=f"Forecast: {p.forecast_summary}")
        self.lbl_forecast.grid(row=5, column=0, sticky="w")

        horizon_frame = ttk.Frame(frame)
        horizon_frame.grid(row=6, column=0, sticky="w", pady=(5, 0))
        ttk.Label(horizon_frame, text="Prediction horizon:").grid(row=0, column=0, columnspan=3, sticky="w")

        def set_horizon(h):
            self.state.prediction.set_horizon(h)

        ttk.Button(horizon_frame, text="Short", command=lambda: set_horizon(PredictionHorizon.SHORT)).grid(row=1, column=0)
        ttk.Button(horizon_frame, text="Medium", command=lambda: set_horizon(PredictionHorizon.MEDIUM)).grid(row=1, column=1)
        ttk.Button(horizon_frame, text="Long", command=lambda: set_horizon(PredictionHorizon.LONG)).grid(row=1, column=2)

        tune_frame = ttk.Frame(frame)
        tune_frame.grid(row=7, column=0, sticky="w", pady=(5, 0))
        ttk.Button(tune_frame, text="Increase anomaly sensitivity", command=lambda: self._adjust_anomaly_sens(0.1)).grid(row=0, column=0)
        ttk.Button(tune_frame, text="Decrease anomaly sensitivity", command=lambda: self._adjust_anomaly_sens(-0.1)).grid(row=0, column=1)
        ttk.Button(tune_frame, text="Prioritize hive signals", command=lambda: self._adjust_hive_weight(0.1)).grid(row=1, column=0)
        ttk.Button(tune_frame, text="Prioritize local signals", command=lambda: self._adjust_hive_weight(-0.1)).grid(row=1, column=1)

        return frame

    def _adjust_anomaly_sens(self, delta: float):
        self.state.prediction.adjust_anomaly_sensitivity(delta)

    def _adjust_hive_weight(self, delta: float):
        self.state.prediction.adjust_hive_weight(delta)

    def _refresh_predictive(self):
        p = self.state.prediction
        self.lbl_anomaly.config(text=f"Anomaly risk: {p.anomaly_risk:.2f}")
        self.lbl_drive.config(text=f"Drive/System risk: {p.drive_system_risk:.2f}")
        self.lbl_hive_risk.config(text=f"Hive risk: {p.hive_risk:.2f}")
        self.lbl_health.config(text=f"Collective health: {p.collective_health_score:.2f}")
        self.lbl_trend.config(text=f"Health trend: {p.health_trend:.2f}")
        self.lbl_forecast.config(text=f"Forecast: {p.forecast_summary}")

        # Cinematic tint based on forecast
        summary = p.forecast_summary
        color = "#222222"
        if summary == "CALM":
            color = "#202c20"
        elif summary == "TENSE":
            color = "#2c2620"
        elif summary == "CRITICAL":
            color = "#3c2020"

        style.configure("Predict.TLabelframe", background=color)
        self.predict_frame.configure(style="Predict.TLabelframe")

    # ------- Collective -------

    def _build_collective(self, parent):
        frame = ttk.LabelFrame(parent, text="🌐 Collective Health & Hive Influence")

        c = self.state.collective
        self.lbl_col_risk = ttk.Label(frame, text=f"Collective risk: {c.collective_risk_score:.2f}")
        self.lbl_col_risk.grid(row=0, column=0, sticky="w")
        self.lbl_density = ttk.Label(frame, text=f"Hive density: {c.hive_density:.2f}")
        self.lbl_density.grid(row=1, column=0, sticky="w")
        self.lbl_agree = ttk.Label(frame, text=f"Node agreement: {c.node_agreement:.2f}")
        self.lbl_agree.grid(row=2, column=0, sticky="w")
        self.lbl_divergence = ttk.Label(frame, text=f"Divergence: {', '.join(c.divergence_patterns) or 'None'}")
        self.lbl_divergence.grid(row=3, column=0, sticky="w")
        self.lbl_hmode = ttk.Label(frame, text=f"Hive mode: {c.hive_mode}")
        self.lbl_hmode.grid(row=4, column=0, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, sticky="w", pady=(5, 0))

        ttk.Button(btn_frame, text="Aggressive sync", command=lambda: self._set_hive_mode("Aggressive")).grid(row=0, column=0)
        ttk.Button(btn_frame, text="Conservative sync", command=lambda: self._set_hive_mode("Conservative")).grid(row=0, column=1)
        ttk.Button(btn_frame, text="Local-only", command=lambda: self._set_hive_mode("Local-only")).grid(row=0, column=2)

        ttk.Button(btn_frame, text="Trust hive more", command=lambda: self._adjust_hive_trust(0.1)).grid(row=1, column=0)
        ttk.Button(btn_frame, text="Trust hive less", command=lambda: self._adjust_hive_trust(-0.1)).grid(row=1, column=1)

        ttk.Button(btn_frame, text="Propagate my settings", command=self._propagate_settings).grid(row=2, column=0)
        ttk.Button(btn_frame, text="Isolate this node", command=self._isolate_node).grid(row=2, column=1)

        return frame

    def _set_hive_mode(self, mode: str):
        self.state.collective.set_hive_mode(mode)
        self.lbl_hmode.config(text=f"Hive mode: {self.state.collective.hive_mode}")

    def _adjust_hive_trust(self, delta: float):
        self.state.collective.adjust_hive_trust(delta)

    def _propagate_settings(self):
        print("[HIVE] Propagating mission control settings to hive...")

    def _isolate_node(self):
        print("[HIVE] Isolating this node from hive...")

    def _refresh_collective(self):
        c = self.state.collective
        self.lbl_col_risk.config(text=f"Collective risk: {c.collective_risk_score:.2f}")
        self.lbl_density.config(text=f"Hive density: {c.hive_density:.2f}")
        self.lbl_agree.config(text=f"Node agreement: {c.node_agreement:.2f}")
        self.lbl_divergence.config(text=f"Divergence: {', '.join(c.divergence_patterns) or 'None'}")
        self.lbl_hmode.config(text=f"Hive mode: {c.hive_mode}")

    # ------- ACE -------

    def _build_cipher_engine(self, parent):
        frame = ttk.LabelFrame(parent, text="🜏🔒 Autonomous Cipher Engine")

        ace = self.state.cipher
        self.lbl_ace_mode = ttk.Label(frame, text=f"Mode: {ace.mode}")
        self.lbl_ace_mode.grid(row=0, column=0, sticky="w")
        self.lbl_ace_def = ttk.Label(frame, text=f"Defense level: {ace.defense_level}")
        self.lbl_ace_def.grid(row=1, column=0, sticky="w")
        self.lbl_ace_entropy = ttk.Label(frame, text=f"Key entropy: {ace.key_entropy:.2f}")
        self.lbl_ace_entropy.grid(row=2, column=0, sticky="w")
        self.lbl_ace_ret = ttk.Label(frame, text=f"Telemetry retention: {ace.telemetry_retention_seconds}s")
        self.lbl_ace_ret.grid(row=3, column=0, sticky="w")
        self.lbl_ace_ghost = ttk.Label(frame, text=f"Ghost sync: {ace.ghost_sync_detected}")
        self.lbl_ace_ghost.grid(row=4, column=0, sticky="w")
        self.lbl_ace_phantom = ttk.Label(frame, text=f"Phantom node: {ace.phantom_node_enabled}")
        self.lbl_ace_phantom.grid(row=5, column=0, sticky="w")
        self.lbl_ace_reason = ttk.Label(frame, text=f"Last rotation: {ace.last_rotation_reason}")
        self.lbl_ace_reason.grid(row=6, column=0, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=7, column=0, sticky="w", pady=(5, 0))
        ttk.Button(btn_frame, text="🜏 Escalate defense", command=self._ace_escalate).grid(row=0, column=0)
        ttk.Button(btn_frame, text="Rotate keys now", command=self._ace_rotate).grid(row=0, column=1)
        ttk.Button(btn_frame, text="Enter Blackout", command=self._ace_blackout).grid(row=1, column=0)
        ttk.Button(btn_frame, text="Clear ghost flag", command=self._ace_clear_ghost).grid(row=1, column=1)

        return frame

    def _ace_escalate(self):
        self.state.cipher.escalate_defense("Manual escalation")
        self._refresh_ace()

    def _ace_rotate(self):
        self.state.cipher.rotate_keys("Manual rotation")
        self._refresh_ace()

    def _ace_blackout(self):
        self.state.cipher.mode = "Blackout"
        self.state.cipher.defense_level = 5
        self._refresh_ace()

    def _ace_clear_ghost(self):
        self.state.cipher.ghost_sync_detected = False
        self._refresh_ace()

    def _refresh_ace(self):
        ace = self.state.cipher
        self.lbl_ace_mode.config(text=f"Mode: {ace.mode}")
        self.lbl_ace_def.config(text=f"Defense level: {ace.defense_level}")
        self.lbl_ace_entropy.config(text=f"Key entropy: {ace.key_entropy:.2f}")
        self.lbl_ace_ret.config(text=f"Telemetry retention: {ace.telemetry_retention_seconds}s")
        self.lbl_ace_ghost.config(text=f"Ghost sync: {ace.ghost_sync_detected}")
        self.lbl_ace_phantom.config(text=f"Phantom node: {ace.phantom_node_enabled}")
        self.lbl_ace_reason.config(text=f"Last rotation: {ace.last_rotation_reason}")

    # ------- Mesh -------

    def _build_mesh_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🕸️ Borg Mesh Overlay")

        stats = self.magicbox.mesh_organs.mesh.stats()
        self.lbl_mesh_total = ttk.Label(frame, text=f"Nodes: {stats['total']}")
        self.lbl_mesh_total.grid(row=0, column=0, sticky="w")
        self.lbl_mesh_disc = ttk.Label(frame, text=f"Discovered: {stats['discovered']}")
        self.lbl_mesh_disc.grid(row=1, column=0, sticky="w")
        self.lbl_mesh_built = ttk.Label(frame, text=f"Built: {stats['built']}")
        self.lbl_mesh_built.grid(row=2, column=0, sticky="w")
        self.lbl_mesh_enf = ttk.Label(frame, text=f"Enforced: {stats['enforced']}")
        self.lbl_mesh_enf.grid(row=3, column=0, sticky="w")
        self.lbl_mesh_corr = ttk.Label(frame, text=f"Corridors: {stats['corridors']}")
        self.lbl_mesh_corr.grid(row=4, column=0, sticky="w")
        self.lbl_mesh_avg = ttk.Label(frame, text=f"Avg risk: {stats['avg_risk']:.1f}")
        self.lbl_mesh_avg.grid(row=5, column=0, sticky="w")

        ttk.Button(frame, text="Inject synthetic node", command=self._mesh_inject_node).grid(row=6, column=0, sticky="w")

        return frame

    def _mesh_inject_node(self):
        url = f"https://manual.node/{int(time.time())}"
        links = [f"https://manual.node/{int(time.time())+i}" for i in range(1, 4)]
        snippet = "manual_inject=true"
        scan_event = BorgScanEvent(url=url, snippet=snippet, links=links)
        try:
            self.magicbox.mesh_organs.scan_queue.put_nowait(scan_event)
        except queue.Full:
            pass
        self._append_response(f"Mesh: injected node {url}")

    def _refresh_mesh(self):
        stats = self.magicbox.mesh_organs.mesh.stats()
        self.lbl_mesh_total.config(text=f"Nodes: {stats['total']}")
        self.lbl_mesh_disc.config(text=f"Discovered: {stats['discovered']}")
        self.lbl_mesh_built.config(text=f"Built: {stats['built']}")
        self.lbl_mesh_enf.config(text=f"Enforced: {stats['enforced']}")
        self.lbl_mesh_corr.config(text=f"Corridors: {stats['corridors']}")
        self.lbl_mesh_avg.config(text=f"Avg risk: {stats['avg_risk']:.1f}")

    # ------- Codex -------

    def _build_codex_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🧬 Adaptive Codex Mutation")

        codex = self.state.codex
        self.lbl_codex_drift = ttk.Label(frame, text=f"Drift magnitude: {codex.drift_magnitude:.3f}")
        self.lbl_codex_drift.grid(row=0, column=0, sticky="w")
        self.lbl_codex_reason = ttk.Label(frame, text=f"Last mutation: {codex.last_mutation_reason}")
        self.lbl_codex_reason.grid(row=1, column=0, sticky="w")
        self.lbl_codex_enabled = ttk.Label(frame, text=f"Adaptive: {codex.adaptive_enabled}")
        self.lbl_codex_enabled.grid(row=2, column=0, sticky="w")

        weights_str = ", ".join(f"{float(x):.2f}" for x in codex.agent_weights.tolist())
        self.lbl_codex_weights = ttk.Label(frame, text=f"Weights: [{weights_str}]")
        self.lbl_codex_weights.grid(row=3, column=0, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, sticky="w", pady=(5, 0))
        ttk.Button(btn_frame, text="Mutate Codex Now", command=self._codex_mutate_now).grid(row=0, column=0)
        ttk.Button(btn_frame, text="Reset Codex Drift", command=self._codex_reset_drift).grid(row=0, column=1)
        ttk.Button(btn_frame, text="Enable Adaptive", command=lambda: self._codex_set_adaptive(True)).grid(row=1, column=0)
        ttk.Button(btn_frame, text="Disable Adaptive", command=lambda: self._codex_set_adaptive(False)).grid(row=1, column=1)
        ttk.Button(btn_frame, text="View Mutation Log", command=self._codex_show_log).grid(row=2, column=0)

        return frame

    def _codex_mutate_now(self):
        frame = self.magicbox.bots.generate_telemetry_frame()
        self.state.codex.mutate_based_on_frame(frame)
        self._refresh_codex()

    def _codex_reset_drift(self):
        self.state.codex.reset_drift()
        self._refresh_codex()

    def _codex_set_adaptive(self, enabled: bool):
        self.state.codex.toggle_adaptive(enabled)
        self._refresh_codex()

    def _codex_show_log(self):
        if not self.state.codex.mutation_log:
            self._append_response("Codex log: no mutations yet.")
            return
        last = self.state.codex.mutation_log[-1]
        self._append_response(
            f"Last mutation: reason={last['reason']}, "
            f"delta={last['delta']}, weights={last['new_weights']}"
        )

    def _refresh_codex(self):
        codex = self.state.codex
        self.lbl_codex_drift.config(text=f"Drift magnitude: {codex.drift_magnitude:.3f}")
        self.lbl_codex_reason.config(text=f"Last mutation: {codex.last_mutation_reason}")
        self.lbl_codex_enabled.config(text=f"Adaptive: {codex.adaptive_enabled}")
        weights_str = ", ".join(f"{float(x):.2f}" for x in codex.agent_weights.tolist())
        self.lbl_codex_weights.config(text=f"Weights: [{weights_str}]")

    # ------- Command Bar -------

    def _build_command_bar(self, parent):
        frame = ttk.LabelFrame(parent, text="🛡️ Mission Command Bar")

        commands = [
            ("Stabilize system", self._cmd_stabilize),
            ("Enter high-alert mode", self._cmd_high_alert),
            ("Begin learning cycle", self._cmd_learning_cycle),
            ("Optimize performance", self._cmd_optimize),
            ("Purge anomaly memory", self._cmd_purge_anomalies),
            ("Rebuild predictive model", self._cmd_rebuild_model),
            ("Reset situational cortex", self._cmd_reset_cortex),
            ("Snapshot brain state", self._cmd_snapshot),
            ("Rollback to previous state", self._cmd_rollback),
            ("Posture: Watcher", self._cmd_posture_watcher),
            ("Posture: Guardian", self._cmd_posture_guardian),
            ("Posture: Chameleon", self._cmd_posture_chameleon),
        ]

        for i, (label, func) in enumerate(commands):
            ttk.Button(frame, text=label, command=func).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)

        return frame

    def _cmd_stabilize(self):
        self.state.brain.set_mode(BrainMode.STABILITY)
        self.state.cortex.set_mission(MissionType.STABILITY)
        self._refresh_brain()
        self._refresh_cortex()
        self._append_response("Command: Stabilize system executed.")

    def _cmd_high_alert(self):
        self.state.brain.set_mode(BrainMode.REFLEX)
        self.state.cortex.environment = EnvironmentState.DANGER
        self._refresh_brain()
        self._refresh_cortex()
        self._append_response("Command: Entered high-alert mode.")

    def _cmd_learning_cycle(self):
        self.state.cortex.set_mission(MissionType.LEARN)
        self._refresh_cortex()
        self._append_response("Command: Begin learning cycle.")

    def _cmd_optimize(self):
        self.state.cortex.set_mission(MissionType.OPTIMIZE)
        self._refresh_cortex()
        self._append_response("Command: Optimize performance.")

    def _cmd_purge_anomalies(self):
        print("[PREDICT] Purging anomaly memory...")
        self._append_response("Command: Anomaly memory purge requested (hook to real store).")

    def _cmd_rebuild_model(self):
        print("[PREDICT] Rebuilding predictive model...")
        self._append_response("Command: Predictive model rebuild requested (hook to real engine).")

    def _cmd_reset_cortex(self):
        self.state.cortex = SituationalCortex()
        self._refresh_cortex()
        self._append_response("Command: Situational cortex reset.")

    def _cmd_snapshot(self):
        self.state.snapshot()
        self._append_response("Command: Brain snapshot saved.")

    def _cmd_rollback(self):
        self.state.rollback()
        self._refresh_brain()
        self._refresh_judgment()
        self._refresh_cortex()
        self._refresh_predictive()
        self._refresh_collective()
        self._refresh_ace()
        self._refresh_mesh()
        self._refresh_codex()
        self._append_response("Command: Rolled back to previous state.")

    def _cmd_posture_watcher(self):
        self.state.posture = Posture.WATCHER
        self._append_response("Posture set to WATCHER (calm observer).")

    def _cmd_posture_guardian(self):
        self.state.posture = Posture.GUARDIAN
        self._append_response("Posture set to GUARDIAN (defensive, cautious).")

    def _cmd_posture_chameleon(self):
        self.state.posture = Posture.CHAMELEON
        self._append_response("Posture set to CHAMELEON (redacted view).")

    # ------- Dialogue -------

    def _build_dialogue(self, parent):
        frame = ttk.LabelFrame(parent, text="🗨️ ASI Dialogue Window")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=0, sticky="w")

        questions = [
            ("Why did you choose this mission?", self._ask_why_mission),
            ("What are you predicting next?", self._ask_forecast),
            ("What are you uncertain about?", self._ask_uncertainty),
            ("What do you need from me?", self._ask_needs),
            ("Explain your reasoning", self._ask_reasoning),
        ]

        for i, (label, func) in enumerate(questions):
            ttk.Button(btn_frame, text=label, command=func).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)

        self.txt_response = tk.Text(frame, height=6, wrap="word")
        self.txt_response.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        return frame

    def _append_response(self, text: str):
        self.txt_response.insert("end", text + "\n")
        self.txt_response.see("end")

    def _ask_why_mission(self):
        c = self.state.cortex
        self._append_response(
            f"Intent: Mission is {c.mission.value} based on environment {c.environment.value}, "
            f"risk={c.risk_score:.2f}, opportunity={c.opportunity_score:.2f}."
        )

    def _ask_forecast(self):
        p = self.state.prediction
        self._append_response(
            f"Forecast: {p.forecast_summary} (anomaly={p.anomaly_risk:.2f}, drive={p.drive_system_risk:.2f}, "
            f"hive={p.hive_risk:.2f}, health={p.collective_health_score:.2f})."
        )

    def _ask_uncertainty(self):
        p = self.state.prediction
        self._append_response(
            f"Uncertainties: trend={p.health_trend:.2f}, hive_weight={p.hive_weight:.2f}, "
            f"anomaly_sensitivity={p.anomaly_sensitivity:.2f}."
        )

    def _ask_needs(self):
        self._append_response(
            "Needs: clarify mission priorities, adjust risk tolerance, confirm hive sync mode, "
            "and validate mesh corridors for upcoming forecasts."
        )

    def _ask_reasoning(self):
        b = self.state.brain
        c = self.state.cortex
        mesh_stats = self.magicbox.mesh_organs.mesh.stats()
        self._append_response(
            f"Reasoning: operating in {b.mode.value} with trust={b.trust:.2f} and sensitivity={b.sensitivity:.2f}, "
            f"aligning mission {c.mission.value} to environment {c.environment.value} and risk tolerance "
            f"{c.risk_tolerance:.2f}. Mesh nodes={mesh_stats['total']} avg_risk={mesh_stats['avg_risk']:.1f}."
        )

    # ------- GUI Tick -------

    def _tick(self):
        self._refresh_brain()
        self._refresh_judgment()
        self._refresh_cortex()
        self._refresh_predictive()
        self._refresh_collective()
        self._refresh_ace()
        self._refresh_mesh()
        self._refresh_codex()
        self.after(1000, self._tick)

    def _start_tick_loop(self):
        self.after(1000, self._tick)


# =========================
# ASI Memory View with System Snapshot
# =========================


class ASIMemoryViewPane(ttk.Frame):
    def __init__(self, master, state: MissionControlState, magicbox: MagicBox, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.magicbox = magicbox
        self.camouflage = tk.BooleanVar(value=False)
        self.show_snapshot = tk.BooleanVar(value=True)
        self.snapshot_expanded = tk.BooleanVar(value=True)  # expanded by default
        self.last_frame: Optional[Dict] = None
        self._build_layout()
        self._start_refresh_loop()

    def _build_layout(self):
        self.columnconfigure(0, weight=1)

        top_bar = ttk.Frame(self)
        top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        ttk.Label(top_bar, text="View mode:").pack(side="left")
        ttk.Checkbutton(
            top_bar,
            text="Camouflage (redacted)",
            variable=self.camouflage,
        ).pack(side="left", padx=(5, 0))

        ttk.Checkbutton(
            top_bar,
            text="Show System Snapshot",
            variable=self.show_snapshot,
        ).pack(side="left", padx=(10, 0))

        self.txt = tk.Text(self, wrap="word")
        self.txt.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.rowconfigure(1, weight=1)

    def _sparkline(self, value: float, length: int = 12) -> str:
        value = max(0.0, min(1.0, value))
        high_count = int(round(value * length))
        med_count = max(0, high_count // 2)
        low_count = length - high_count
        bar = "░" * low_count + "▒" * med_count + "▓" * (high_count - med_count)
        return bar[:length]

    def _pulse_glyph(self, load: float) -> str:
        if load < 0.3:
            return "·"
        elif load < 0.7:
            return "●"
        else:
            return "⬤"

    def _snapshot_tint(self, load: float) -> str:
        if load < 0.3:
            return "#202c20"
        elif load < 0.7:
            return "#2c2c20"
        else:
            return "#3c2020"

    def _refresh_view(self):
        s = self.state
        mesh_stats = self.magicbox.mesh_organs.mesh.stats()
        codex = s.codex

        # posture-driven camouflage
        if s.posture == Posture.CHAMELEON:
            self.camouflage.set(True)

        camo = self.camouflage.get()

        # grab a fresh frame to base snapshot on
        frame = self.magicbox.bots.generate_telemetry_frame()
        self.last_frame = frame

        cpu = frame["cpu_load"]
        mem = frame["mem_load"]
        disk = frame["disk_load"]
        net = frame["net_activity"]
        activity = frame["activity_level"]
        fg_app = frame["foreground_app"]

        lines = []

        lines.append("🧠 Hybrid Brain Core")
        lines.append(f"  Mode: {s.brain.mode.value}")
        lines.append(f"  Trust: {s.brain.trust:.3f}")
        lines.append(f"  Volatility: {s.brain.volatility:.3f}")
        lines.append(f"  Cognitive Load: {s.brain.cognitive_load:.3f}")
        lines.append(f"  Sensitivity: {s.brain.sensitivity:.3f}")
        lines.append("")

        lines.append("🕸️ Situational Cortex")
        lines.append(f"  Mission: {s.cortex.mission.value}")
        lines.append(f"  Environment: {s.cortex.environment.value}")
        lines.append(f"  Risk Score: {s.cortex.risk_score:.3f}")
        lines.append(f"  Opportunity Score: {s.cortex.opportunity_score:.3f}")
        lines.append(f"  Risk Tolerance: {s.cortex.risk_tolerance:.3f}")
        lines.append(f"  Posture: {s.posture.value}")
        lines.append("")

        lines.append("🔮 Predictive Intelligence")
        lines.append(f"  Anomaly Risk: {s.prediction.anomaly_risk:.3f}")
        lines.append(f"  Drive/System Risk: {s.prediction.drive_system_risk:.3f}")
        lines.append(f"  Hive Risk: {s.prediction.hive_risk:.3f}")
        lines.append(f"  Collective Health: {s.prediction.collective_health_score:.3f}")
        lines.append(f"  Health Trend: {s.prediction.health_trend:.3f}")
        lines.append(f"  Horizon: {s.prediction.horizon.value}")
        lines.append(f"  Forecast: {s.prediction.forecast_summary}")
        lines.append("")

        # System Snapshot (collapsible)
        if self.show_snapshot.get():
            # header with expand/collapse glyph
            header_glyph = "▼" if self.snapshot_expanded.get() else "▶"
            lines.append(f"{header_glyph} 🖥️ System Snapshot")
            if self.snapshot_expanded.get() and self.last_frame is not None:
                cpu_bar = self._sparkline(cpu)
                mem_bar = self._sparkline(mem)
                disk_bar = self._sparkline(disk)
                net_bar = self._sparkline(net)

                pulse = self._pulse_glyph(max(cpu, mem, net))
                pulse_bar = pulse * 10

                lines.append(f"    CPU Load: {cpu:.2f}    {cpu_bar}")
                lines.append(f"    RAM Load: {mem:.2f}    {mem_bar}")
                lines.append(f"    Disk Load: {disk:.2f}   {disk_bar}")
                lines.append(f"    Net Activity: {net:.2f} {net_bar}")
                if camo and s.posture == Posture.CHAMELEON:
                    lines.append(f"    Foreground App: [REDACTED]")
                else:
                    lines.append(f"    Foreground App: {fg_app!r}")
                lines.append(f"    Activity Level: {activity:.2f}")
                lines.append(f"    Pulse: {pulse_bar}")
            lines.append("")

        lines.append("🜏 Autonomous Cipher Engine")
        lines.append(f"  Mode: {s.cipher.mode}")
        lines.append(f"  Defense Level: {s.cipher.defense_level}")
        lines.append(f"  Telemetry Retention: {s.cipher.telemetry_retention_seconds}s")
        lines.append(f"  Ghost Sync: {s.cipher.ghost_sync_detected}")
        lines.append(f"  Phantom Node: {s.cipher.phantom_node_enabled}")
        lines.append(f"  Last Rotation Reason: {'[REDACTED]' if camo else s.cipher.last_rotation_reason}")
        lines.append("")

        lines.append("🧬 Codex Mutation Engine")
        if camo:
            lines.append("  Weights: [REDACTED]")
            lines.append("  Last Mutation Reason: [REDACTED]")
        else:
            weights_str = ", ".join(f"{float(x):.3f}" for x in codex.agent_weights.tolist())
            lines.append(f"  Weights: [{weights_str}]")
            lines.append(f"  Last Mutation Reason: {codex.last_mutation_reason}")
        lines.append(f"  Drift Magnitude: {codex.drift_magnitude:.6f}")
        lines.append(f"  Adaptive Enabled: {codex.adaptive_enabled}")
        lines.append(f"  Mutation Count: {len(codex.mutation_log)}")
        lines.append("")

        lines.append("🕸️ Borg Mesh")
        lines.append(f"  Nodes: {mesh_stats['total']}")
        lines.append(f"  Discovered: {mesh_stats['discovered']}")
        lines.append(f"  Built: {mesh_stats['built']}")
        lines.append(f"  Enforced: {mesh_stats['enforced']}")
        lines.append(f"  Corridors: {mesh_stats['corridors']}")
        lines.append(f"  Average Risk: {mesh_stats['avg_risk']:.2f}")
        lines.append("")

        lines.append("Threat Glyphs:")
        glyph = {
            "CALM": "░",
            "TENSE": "▒",
            "CRITICAL": "▓",
        }.get(s.prediction.forecast_summary, "░")
        lines.append(f"  [{glyph * 20}]")
        lines.append("")

        text = "\n".join(lines)

        # apply snapshot tint to text widget background (subtle)
        tint = self._snapshot_tint(max(cpu, mem, net))
        try:
            self.txt.configure(bg=tint)
        except Exception:
            pass

        self.txt.delete("1.0", "end")
        self.txt.insert("end", text)
        self.txt.see("end")

    def _refresh_tick(self):
        self._refresh_view()
        self.after(2000, self._refresh_tick)

    def _start_refresh_loop(self):
        self.after(2000, self._refresh_tick)


# =========================
# Bootstrap
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

    root = tk.Tk()
    root.title("MagicBox ASI Oversight — Mission Control + ACE + Codex + Borg Mesh + Memory View")

    notebook = ttk.Notebook(root)
    mission_tab = MissionControlPane(notebook, state, magicbox)
    memory_tab = ASIMemoryViewPane(notebook, state, magicbox)

    notebook.add(mission_tab, text="Mission Control")
    notebook.add(memory_tab, text="ASI Memory View")
    notebook.pack(fill="both", expand=True)

    def on_close():
        print("[MAIN] Shutting down. Saving state...")
        if magicbox._persistence is not None:
            magicbox._persistence.save(magicbox.state)
        tick_engine.stop()
        health_monitor.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

