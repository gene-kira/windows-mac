from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import hashlib
import json
import os
import zlib
import sys

from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QTabWidget,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
)


# ============================================================
# Environment snapshot
# ============================================================

@dataclass
class EnvironmentSnapshot:
    host_id: str
    geo: str
    process_id: str
    process_hash: str
    swarm_id: str
    roles: List[str]
    risk_score: float
    mac_address: str          # machine MAC or hardware fingerprint
    timestamp: float = field(default_factory=lambda: time.time())


# ============================================================
# Sanitization for logs / beacons (no raw secrets or hardware IDs)
# ============================================================

def sanitize_env_for_log(env: EnvironmentSnapshot) -> Dict[str, Any]:
    return {
        "host_fingerprint": hashlib.sha256(env.host_id.encode()).hexdigest(),
        "geo": env.geo,
        "process_fingerprint": hashlib.sha256(env.process_hash.encode()).hexdigest(),
        "swarm_id": env.swarm_id,
        "roles": env.roles,
        "risk_score": env.risk_score,
        "timestamp": env.timestamp,
    }


# ============================================================
# Audit organ
# ============================================================

class AuditOrgan:
    def __init__(self):
        self._events = []  # in reality: append-only, tamper-evident

    def log(self, capsule_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts": time.time(),
            "capsule_id": capsule_id,
            "event_type": event_type,
            "payload": payload,
        }
        self._events.append(event)

    def export_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


# ============================================================
# Beacon organ (call home on theft)
# ============================================================

class BeaconOrgan:
    """
    Called when a capsule believes it has been stolen or opened
    in an unauthorized environment.
    """

    def __init__(self):
        self._beacons = []  # in reality: send to central swarm / SIEM / SOC

    def emit(self,
             capsule_id: str,
             reason: str,
             env_snapshot: Dict[str, Any]) -> None:
        beacon = {
            "ts": time.time(),
            "capsule_id": capsule_id,
            "reason": reason,
            "env": env_snapshot,
        }
        self._beacons.append(beacon)

    def get_beacons(self) -> List[Dict[str, Any]]:
        return list(self._beacons)


# ============================================================
# Policy organ (self-aware, history-aware)
# ============================================================

class PolicyOrgan:
    def __init__(self, audit_organ: AuditOrgan):
        self.audit_organ = audit_organ

    def evaluate(self,
                 capsule: "DataCapsule",
                 env: EnvironmentSnapshot,
                 agent_id: str) -> Dict[str, Any]:
        recent_events = [
            e for e in self.audit_organ.export_events()
            if e["capsule_id"] == capsule.capsule_id
        ]

        violations_from_host = [
            e for e in recent_events
            if e["event_type"] == "violation"
            and e["payload"].get("env", {}).get("host_fingerprint") ==
               hashlib.sha256(env.host_id.encode()).hexdigest()
        ]

        if len(violations_from_host) > 3:
            return {
                "allowed": False,
                "reason": "too_many_violations_from_host",
                "degrade": True
            }

        if capsule.sensitivity_level in ("critical", "biometric") and env.risk_score > 0.0:
            return {
                "allowed": False,
                "reason": "high_risk_for_sensitive_data",
                "degrade": True
            }

        return {
            "allowed": True,
            "reason": "policy_ok",
            "degrade": False
        }


# ============================================================
# Key organ (env- and MAC-bound key derivation)
# ============================================================

class KeyOrgan:
    """
    Env-bound, policy-bound key derivation.
    Stubbed with SHA-256; replace with HSM/TEE + quorum in real use.
    """

    def derive_key(self,
                   key_ref: str,
                   env: EnvironmentSnapshot,
                   key_policy: Dict[str, Any],
                   chameleon_salt: Optional[bytes] = None) -> bytes:
        material = {
            "key_ref": key_ref,
            "host_id": env.host_id if key_policy.get("bind_to_host") else "",
            "swarm_id": env.swarm_id if key_policy.get("bind_to_swarm") else "",
            "geo": env.geo if key_policy.get("bind_to_geo") else "",
            "mac": env.mac_address if key_policy.get("bind_to_mac") else "",
            "salt": chameleon_salt.hex() if chameleon_salt else "",
        }
        blob = json.dumps(material, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).digest()


# ============================================================
# Toy crypto helpers (replace with real AES-GCM)
# ============================================================

def _xor_bytes(data: bytes, key: bytes) -> bytes:
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return bytes(out)


def encrypt_payload(plaintext_obj: Any, key: bytes) -> bytes:
    raw = json.dumps(plaintext_obj, sort_keys=True).encode("utf-8")
    compressed = zlib.compress(raw)
    return _xor_bytes(compressed, key)


def decrypt_payload(ciphertext: bytes, key: bytes) -> Any:
    compressed = _xor_bytes(ciphertext, key)
    raw = zlib.decompress(compressed)
    return json.loads(raw.decode("utf-8"))


# ============================================================
# Data capsule
# ============================================================

@dataclass
class DataCapsule:
    capsule_id: str
    owner_id: str
    sensitivity_level: str
    schema_version: str
    lineage: Dict[str, Any]

    allowed_hosts: List[str]
    allowed_geos: List[str]
    allowed_processes: List[str]
    allowed_swarms: List[str]
    time_not_before: Optional[float] = None
    time_not_after: Optional[float] = None

    allowed_roles: List[str] = field(default_factory=list)
    allowed_agents: List[str] = field(default_factory=list)
    max_risk_score: float = 0.0
    export_policies: Dict[str, Any] = field(default_factory=dict)

    on_violation: str = "self_destruct"
    degradation_mode: str = "redact"
    audit_level: str = "full"
    beacon_on_violation: bool = True

    ciphertext: bytes = b""
    encryption_scheme: str = "TOY-XOR+ZLIB"
    key_ref: str = ""
    key_policy: Dict[str, Any] = field(default_factory=dict)
    chameleon_salt: bytes = field(default_factory=lambda: os.urandom(16))

    destroyed: bool = False

    @classmethod
    def from_plaintext(cls,
                       capsule_id: str,
                       owner_id: str,
                       sensitivity_level: str,
                       schema_version: str,
                       lineage: Dict[str, Any],
                       plaintext_obj: Any,
                       env: EnvironmentSnapshot,
                       allowed_hosts: List[str],
                       allowed_geos: List[str],
                       allowed_processes: List[str],
                       allowed_swarms: List[str],
                       key_organ: KeyOrgan,
                       key_ref: Optional[str] = None,
                       key_policy: Optional[Dict[str, Any]] = None,
                       **kwargs) -> "DataCapsule":
        if key_policy is None:
            key_policy = {
                "bind_to_host": True,
                "bind_to_swarm": True,
                "bind_to_geo": True,
                "bind_to_mac": True
            }

        if not key_policy.get("bind_to_mac"):
            raise ValueError("All personal/biometric data must be bound to MAC address")

        if key_ref is None:
            key_ref = f"{capsule_id}:{os.urandom(8).hex()}"

        chameleon_salt = os.urandom(16)
        key = key_organ.derive_key(key_ref, env, key_policy, chameleon_salt=chameleon_salt)
        ciphertext = encrypt_payload(plaintext_obj, key)

        return cls(
            capsule_id=capsule_id,
            owner_id=owner_id,
            sensitivity_level=sensitivity_level,
            schema_version=schema_version,
            lineage=lineage,
            allowed_hosts=allowed_hosts,
            allowed_geos=allowed_geos,
            allowed_processes=allowed_processes,
            allowed_swarms=allowed_swarms,
            ciphertext=ciphertext,
            key_ref=key_ref,
            key_policy=key_policy,
            chameleon_salt=chameleon_salt,
            **kwargs
        )

    def access(self,
               env: EnvironmentSnapshot,
               agent_id: str,
               key_organ: KeyOrgan,
               policy_organ: PolicyOrgan,
               audit_organ: AuditOrgan,
               beacon_organ: BeaconOrgan) -> Optional[Any]:
        if self.destroyed:
            audit_organ.log(self.capsule_id, "access_attempt_on_destroyed", {
                "env": sanitize_env_for_log(env),
                "agent_id": agent_id
            })
            return None

        decision = self._evaluate_environment(env, agent_id, policy_organ)

        if not decision["allowed"]:
            self._handle_violation(decision, env, audit_organ, beacon_organ)
            return decision.get("degraded_payload")

        key = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        plaintext = self._decrypt_payload(key)

        if self.audit_level == "full":
            audit_organ.log(self.capsule_id, "access_granted", {
                "env": sanitize_env_for_log(env),
                "agent_id": agent_id,
                "reason": decision["reason"]
            })

        return plaintext

    def morph_ciphertext(self,
                         env: EnvironmentSnapshot,
                         key_organ: KeyOrgan) -> None:
        if self.destroyed or not self.ciphertext:
            return

        old_key = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        plaintext = decrypt_payload(self.ciphertext, old_key)

        self.chameleon_salt = os.urandom(16)
        new_key = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        self.ciphertext = encrypt_payload(plaintext, new_key)

    def _evaluate_environment(self,
                              env: EnvironmentSnapshot,
                              agent_id: str,
                              policy_organ: PolicyOrgan) -> Dict[str, Any]:
        if self.allowed_hosts and env.host_id not in self.allowed_hosts:
            return {"allowed": False, "reason": "host_not_allowed"}

        if self.allowed_geos and env.geo not in self.allowed_geos:
            return {"allowed": False, "reason": "geo_not_allowed"}

        if self.allowed_swarms and env.swarm_id not in self.allowed_swarms:
            return {"allowed": False, "reason": "swarm_not_allowed"}

        if self.allowed_processes and env.process_hash not in self.allowed_processes:
            return {"allowed": False, "reason": "process_not_allowed"}

        if self.time_not_before and env.timestamp < self.time_not_before:
            return {"allowed": False, "reason": "too_early"}

        if self.time_not_after and env.timestamp > self.time_not_after:
            return {"allowed": False, "reason": "expired"}

        if env.risk_score > self.max_risk_score:
            return {"allowed": False, "reason": "risk_too_high"}

        if self.allowed_roles and not any(r in env.roles for r in self.allowed_roles):
            return {"allowed": False, "reason": "role_not_allowed"}

        if self.allowed_agents and agent_id not in self.allowed_agents:
            return {"allowed": False, "reason": "agent_not_allowed"}

        policy_decision = policy_organ.evaluate(self, env, agent_id)

        if not policy_decision["allowed"]:
            if policy_decision.get("degrade"):
                degraded = self._produce_degraded_payload(policy_decision)
                return {
                    "allowed": False,
                    "reason": policy_decision["reason"],
                    "degraded_payload": degraded
                }
        return {"allowed": True, "reason": policy_decision["reason"]}

    def _handle_violation(self,
                          decision: Dict[str, Any],
                          env: EnvironmentSnapshot,
                          audit_organ: AuditOrgan,
                          beacon_organ: BeaconOrgan) -> None:
        audit_organ.log(self.capsule_id, "violation", {
            "reason": decision["reason"],
            "env": sanitize_env_for_log(env),
            "mode": self.on_violation
        })

        if self.beacon_on_violation:
            beacon_organ.emit(
                capsule_id=self.capsule_id,
                reason=decision["reason"],
                env_snapshot=sanitize_env_for_log(env)
            )

        if self.on_violation == "self_destruct":
            self._self_destruct()

    def _self_destruct(self) -> None:
        self.ciphertext = b""
        self.destroyed = True

    def _decrypt_payload(self, key: bytes) -> Any:
        if not self.ciphertext:
            return None
        return decrypt_payload(self.ciphertext, key)

    def _produce_degraded_payload(self, policy_decision: Dict[str, Any]) -> Any:
        mode = self.degradation_mode
        redact_fields = self.export_policies.get("redact_fields", [])
        partial_fields = self.export_policies.get("partial_fields", [])

        precomputed = self.export_policies.get("precomputed_degraded")
        if precomputed is not None:
            return precomputed

        return {
            "status": "degraded",
            "mode": mode,
            "reason": policy_decision.get("reason", ""),
            "redact_fields": redact_fields,
            "partial_fields": partial_fields,
        }


# ============================================================
# Capsule runtime
# ============================================================

class CapsuleRuntime:
    def __init__(self):
        self.audit = AuditOrgan()
        self.policy = PolicyOrgan(self.audit)
        self.keys = KeyOrgan()
        self.beacon = BeaconOrgan()
        self.capsule_states: Dict[str, Dict[str, Any]] = {}

    def access_capsule(self,
                       capsule: DataCapsule,
                       env: EnvironmentSnapshot,
                       agent_id: str) -> Optional[Any]:
        result = capsule.access(
            env=env,
            agent_id=agent_id,
            key_organ=self.keys,
            policy_organ=self.policy,
            audit_organ=self.audit,
            beacon_organ=self.beacon
        )

        state = "destroyed" if capsule.destroyed else ("degraded" if isinstance(result, dict) and result.get("status") == "degraded" else "alive")
        self.capsule_states[capsule.capsule_id] = {
            "state": state,
            "last_access": time.time(),
        }
        return result


# ============================================================
# Storage backends
# ============================================================

class BrowserStorageBackend:
    def __init__(self):
        self._kv = {}

    def write(self, key: str, record: Dict[str, Any]) -> None:
        self._kv[key] = json.dumps(record)

    def read(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._kv.get(key)
        return json.loads(raw) if raw else None


class FileStorageBackend:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_")
        return os.path.join(self.base_dir, f"{safe}.json")

    def write(self, key: str, record: Dict[str, Any]) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f)

    def read(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ============================================================
# Data protection gateway
# ============================================================

class DataProtectionGateway:
    def __init__(self, runtime: CapsuleRuntime, storage_backend):
        self.runtime = runtime
        self.storage = storage_backend

    def _make_env(self, context: Dict[str, Any]) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            host_id=context["host_id"],
            geo=context.get("geo", "UNKNOWN"),
            process_id=context.get("process_id", "app"),
            process_hash=context.get("process_hash", ""),
            swarm_id=context.get("swarm_id", "default-swarm"),
            roles=context.get("roles", []),
            risk_score=context.get("risk_score", 0.0),
            mac_address=context["mac_address"],
        )

    def store_personal(self,
                       key: str,
                       payload: Dict[str, Any],
                       context: Dict[str, Any],
                       sensitivity: str = "critical") -> str:
        env = self._make_env(context)
        capsule_id = f"{key}:{int(time.time())}"

        capsule = DataCapsule.from_plaintext(
            capsule_id=capsule_id,
            owner_id=context.get("owner_id", "unknown-owner"),
            sensitivity_level=sensitivity,
            schema_version="1.0",
            lineage={"source": context.get("source", "app")},
            plaintext_obj=payload,
            env=env,
            allowed_hosts=[env.host_id],
            allowed_geos=[env.geo],
            allowed_processes=[env.process_hash] if env.process_hash else [],
            allowed_swarms=[env.swarm_id],
            key_organ=self.runtime.keys,
            export_policies=context.get("export_policies", {}),
            on_violation="self_destruct",
            degradation_mode="redact",
            max_risk_score=context.get("max_risk_score", 0.0),
        )

        record = {
            "capsule_id": capsule.capsule_id,
            "owner_id": capsule.owner_id,
            "sensitivity_level": capsule.sensitivity_level,
            "schema_version": capsule.schema_version,
            "lineage": capsule.lineage,
            "allowed_hosts": capsule.allowed_hosts,
            "allowed_geos": capsule.allowed_geos,
            "allowed_processes": capsule.allowed_processes,
            "allowed_swarms": capsule.allowed_swarms,
            "time_not_before": capsule.time_not_before,
            "time_not_after": capsule.time_not_after,
            "allowed_roles": capsule.allowed_roles,
            "allowed_agents": capsule.allowed_agents,
            "max_risk_score": capsule.max_risk_score,
            "export_policies": capsule.export_policies,
            "on_violation": capsule.on_violation,
            "degradation_mode": capsule.degradation_mode,
            "audit_level": capsule.audit_level,
            "beacon_on_violation": capsule.beacon_on_violation,
            "encryption_scheme": capsule.encryption_scheme,
            "key_ref": capsule.key_ref,
            "key_policy": capsule.key_policy,
            "chameleon_salt": capsule.chameleon_salt.hex(),
            "ciphertext": capsule.ciphertext.hex(),
        }

        self.storage.write(key, record)
        return capsule_id

    def load_personal(self,
                      key: str,
                      context: Dict[str, Any],
                      agent_id: str) -> Optional[Dict[str, Any]]:
        record = self.storage.read(key)
        if not record:
            return None

        env = self._make_env(context)

        capsule = DataCapsule(
            capsule_id=record["capsule_id"],
            owner_id=record["owner_id"],
            sensitivity_level=record["sensitivity_level"],
            schema_version=record["schema_version"],
            lineage=record["lineage"],
            allowed_hosts=record["allowed_hosts"],
            allowed_geos=record["allowed_geos"],
            allowed_processes=record["allowed_processes"],
            allowed_swarms=record["allowed_swarms"],
            time_not_before=record["time_not_before"],
            time_not_after=record["time_not_after"],
            allowed_roles=record["allowed_roles"],
            allowed_agents=record["allowed_agents"],
            max_risk_score=record["max_risk_score"],
            export_policies=record["export_policies"],
            on_violation=record["on_violation"],
            degradation_mode=record["degradation_mode"],
            audit_level=record["audit_level"],
            beacon_on_violation=record["beacon_on_violation"],
            ciphertext=bytes.fromhex(record["ciphertext"]),
            encryption_scheme=record["encryption_scheme"],
            key_ref=record["key_ref"],
            key_policy=record["key_policy"],
            chameleon_salt=bytes.fromhex(record["chameleon_salt"]),
        )

        result = self.runtime.access_capsule(capsule, env, agent_id=agent_id)
        return result


# ============================================================
# PyQt6 GUI — signal bridge + listener thread
# ============================================================

class BeaconSignalBridge(QObject):
    beacon_received = pyqtSignal(dict)


class BeaconListenerThread(QThread):
    def __init__(self, runtime, signal_bridge, poll_interval=1.0):
        super().__init__()
        self.runtime = runtime
        self.signal_bridge = signal_bridge
        self.poll_interval = poll_interval
        self._running = True
        self._last_seen = 0

    def run(self):
        while self._running:
            beacons = self.runtime.beacon.get_beacons()
            if len(beacons) > self._last_seen:
                new_items = beacons[self._last_seen:]
                for beacon in new_items:
                    self.signal_bridge.beacon_received.emit(beacon)
                self._last_seen = len(beacons)
            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False
        self.wait()


# ============================================================
# Threat Dashboard Tab
# ============================================================

class ThreatDashboardTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime, signal_bridge: BeaconSignalBridge):
        super().__init__()
        self.runtime = runtime
        self.signal_bridge = signal_bridge

        main_layout = QVBoxLayout()

        # Top row: severity + risk
        top_row = QHBoxLayout()

        self.severity_indicator = QLabel("Severity: NORMAL")
        self.severity_indicator.setObjectName("severityLabel")
        self.severity_indicator.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.risk_label = QLabel("Risk: 0.0")
        self.risk_label.setObjectName("riskLabel")
        self.risk_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        top_row.addWidget(self.severity_indicator)
        top_row.addWidget(self.risk_label)

        main_layout.addLayout(top_row)

        # Environment mismatch panel
        mismatch_frame = QFrame()
        mismatch_frame.setObjectName("mismatchFrame")
        mismatch_layout = QVBoxLayout()

        self.mismatch_title = QLabel("Environment Mismatch Monitor")
        self.mismatch_title.setObjectName("sectionTitle")

        self.mismatch_text = QTextEdit()
        self.mismatch_text.setReadOnly(True)
        self.mismatch_text.setPlaceholderText("No mismatches detected yet...")

        mismatch_layout.addWidget(self.mismatch_title)
        mismatch_layout.addWidget(self.mismatch_text)
        mismatch_frame.setLayout(mismatch_layout)

        main_layout.addWidget(mismatch_frame)

        # Threat feed
        self.threat_feed = QTextEdit()
        self.threat_feed.setReadOnly(True)
        self.threat_feed.setPlaceholderText("Threat events will appear here...")
        main_layout.addWidget(self.threat_feed)

        # Capsule state table
        self.capsule_table = QTableWidget(0, 4)
        self.capsule_table.setHorizontalHeaderLabels(["Capsule ID", "State", "Last Access", "Last Violation"])
        header = self.capsule_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(self.capsule_table)

        self.setLayout(main_layout)

        signal_bridge.beacon_received.connect(self.on_beacon_received)

    def on_beacon_received(self, beacon_event: dict):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(beacon_event["ts"]))
        capsule_id = beacon_event["capsule_id"]
        reason = beacon_event["reason"]
        env = beacon_event["env"]

        entry = (
            f"[{ts}] STOLEN / VIOLATION\n"
            f"  Capsule: {capsule_id}\n"
            f"  Reason: {reason}\n"
            f"  Env: {env}\n"
            "----------------------------------------\n"
        )
        self.threat_feed.append(entry)

        self.severity_indicator.setText("Severity: CRITICAL")
        self.severity_indicator.setProperty("severity", "critical")
        self.severity_indicator.style().unpolish(self.severity_indicator)
        self.severity_indicator.style().polish(self.severity_indicator)

        self._update_mismatch_view(reason, env)
        self._update_capsule_table()

    def _update_mismatch_view(self, reason: str, env: Dict[str, Any]):
        lines = [f"Reason: {reason}", f"Env: {env}"]
        self.mismatch_text.append("\n".join(lines) + "\n----------------------------------------")

    def _update_capsule_table(self):
        states = self.runtime.capsule_states
        self.capsule_table.setRowCount(len(states))
        for row, (capsule_id, info) in enumerate(states.items()):
            state = info.get("state", "unknown")
            last_access_ts = info.get("last_access", 0)
            last_access_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_access_ts)) if last_access_ts else "-"

            self.capsule_table.setItem(row, 0, QTableWidgetItem(capsule_id))
            self.capsule_table.setItem(row, 1, QTableWidgetItem(state))
            self.capsule_table.setItem(row, 2, QTableWidgetItem(last_access_str))
            self.capsule_table.setItem(row, 3, QTableWidgetItem("see threat feed"))


# ============================================================
# System Status Tab (original simple GUI)
# ============================================================

class SystemStatusTab(QWidget):
    def __init__(self, runtime, signal_bridge):
        super().__init__()
        self.runtime = runtime
        self.signal_bridge = signal_bridge

        layout = QVBoxLayout()

        self.status_label = QLabel("System Running")
        self.engine_label = QLabel("Capsule Engine Active")
        self.beacon_label = QLabel("Beacon Listener Online")

        layout.addWidget(self.status_label)
        layout.addWidget(self.engine_label)
        layout.addWidget(self.beacon_label)

        self.beacon_feed = QTextEdit()
        self.beacon_feed.setReadOnly(True)
        self.beacon_feed.setPlaceholderText("Waiting for beacon events...")
        layout.addWidget(self.beacon_feed)

        self.setLayout(layout)

        self.signal_bridge.beacon_received.connect(self.on_beacon_received)

    def on_beacon_received(self, beacon_event: dict):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(beacon_event["ts"]))
        capsule_id = beacon_event["capsule_id"]
        reason = beacon_event["reason"]
        env = beacon_event["env"]

        entry = (
            f"[{ts}] BEACON RECEIVED\n"
            f"  Capsule: {capsule_id}\n"
            f"  Reason: {reason}\n"
            f"  Env: {env}\n"
            "----------------------------------------\n"
        )
        self.beacon_feed.append(entry)


# ============================================================
# Main Window with Tabs
# ============================================================

class CapsuleMainWindow(QWidget):
    def __init__(self, runtime: CapsuleRuntime, signal_bridge: BeaconSignalBridge):
        super().__init__()
        self.runtime = runtime
        self.signal_bridge = signal_bridge

        self.setWindowTitle("Capsule Protection Console")
        self.setMinimumSize(800, 600)

        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")

        self.status_tab = SystemStatusTab(runtime, signal_bridge)
        self.threat_tab = ThreatDashboardTab(runtime, signal_bridge)

        self.tabs.addTab(self.status_tab, "System Status")
        self.tabs.addTab(self.threat_tab, "Threat Dashboard")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)


# ============================================================
# Dark Security Console Style
# ============================================================

DARK_STYLE = """
QWidget {
    background-color: #111111;
    color: #E0E0E0;
    font-family: Consolas, "Fira Code", monospace;
    font-size: 11pt;
}

QTabWidget::pane {
    border: 1px solid #333333;
    background: #111111;
}

QTabBar::tab {
    background: #222222;
    color: #CCCCCC;
    padding: 6px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #007ACC;
    color: #FFFFFF;
}

QTextEdit {
    background-color: #000000;
    color: #00FF9C;
    border: 1px solid #333333;
}

QLabel#severityLabel[severity="critical"] {
    color: #FF5555;
    font-weight: bold;
}

QLabel#severityLabel {
    color: #00FF9C;
    font-weight: bold;
}

QLabel#riskLabel {
    color: #FFA500;
}

QLabel#sectionTitle {
    color: #00BFFF;
    font-weight: bold;
}

QFrame#mismatchFrame {
    border: 1px solid #333333;
    border-radius: 4px;
    background-color: #181818;
}

QTableWidget {
    background-color: #000000;
    color: #E0E0E0;
    gridline-color: #333333;
}

QHeaderView::section {
    background-color: #222222;
    color: #CCCCCC;
    padding: 4px;
    border: 1px solid #333333;
}
"""


# ============================================================
# GUI bootstrap
# ============================================================

def start_gui(runtime: CapsuleRuntime):
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    signal_bridge = BeaconSignalBridge()
    main_window = CapsuleMainWindow(runtime, signal_bridge)
    main_window.show()

    listener_thread = BeaconListenerThread(runtime, signal_bridge)
    listener_thread.start()

    exit_code = app.exec()
    listener_thread.stop()
    return exit_code


# ============================================================
# Example usage + GUI launch
# ============================================================

if __name__ == "__main__":
    runtime = CapsuleRuntime()
    storage = FileStorageBackend(base_dir="./capsule_store")
    gateway = DataProtectionGateway(runtime, storage)

    context = {
        "host_id": "host-123",
        "mac_address": "AA:BB:CC:DD:EE:FF",
        "process_id": "my_app",
        "process_hash": "proc-hash-xyz",
        "swarm_id": "prod-swarm",
        "roles": ["secrets_manager"],
        "risk_score": 0.0,
        "owner_id": "operator-1",
        "source": "app",
        "max_risk_score": 0.0,
        "export_policies": {"redact_fields": ["api_key", "biometric_blob"]},
    }

    gateway.store_personal(
        key="serviceX_api_key",
        payload={"api_key": "SUPER-SECRET-KEY-123"},
        context=context,
        sensitivity="critical",
    )

    gateway.store_personal(
        key="user123_biometrics",
        payload={"biometric_blob": "BASE64-OR-BINARY-HERE"},
        context=context,
        sensitivity="biometric",
    )

    bad_context = dict(context)
    bad_context["host_id"] = "stolen-host"
    bad_context["mac_address"] = "11:22:33:44:55:66"

    _ = gateway.load_personal(
        key="serviceX_api_key",
        context=bad_context,
        agent_id="unknown_agent",
    )

    start_gui(runtime)

