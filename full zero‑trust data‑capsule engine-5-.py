import os
import sys
import time
import json
import zlib
import ssl
import hashlib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

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
    QPushButton,
    QLineEdit,
    QFormLayout,
    QComboBox,
)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import requests


# ============================================================
# Optional secure memory C-extension hook
# ============================================================

try:
    import securemem  # hypothetical C extension
    HAVE_SECUREMEM = True
except ImportError:
    HAVE_SECUREMEM = False


def secure_wipe_bytes(b: bytearray):
    if HAVE_SECUREMEM:
        try:
            securemem.secure_zero(b)
            return
        except Exception:
            pass
    for i in range(len(b)):
        b[i] = 0


def secure_wipe_str(s: str):
    if HAVE_SECUREMEM:
        try:
            securemem.secure_zero_str(s)
        except Exception:
            pass


# ============================================================
# TPM / Root of Trust abstraction
# ============================================================

class TPMBackend:
    """
    Abstract TPM-like backend.
    In real deployment, this would call OS/TPM APIs.
    Here we simulate with a file-based secret but keep the API clean.
    """

    def __init__(self, secret_path: Optional[str] = None):
        if secret_path is None:
            secret_path = os.path.join(os.path.dirname(__file__), ".device_secret")
        self.secret_path = secret_path
        self._cached_secret: Optional[bytes] = None

    def get_device_secret(self) -> bytes:
        if self._cached_secret is not None:
            return self._cached_secret
        if os.path.exists(self.secret_path):
            with open(self.secret_path, "rb") as f:
                self._cached_secret = f.read()
        else:
            self._cached_secret = os.urandom(32)
            with open(self.secret_path, "wb") as f:
                f.write(self._cached_secret)
        return self._cached_secret


class RootOfTrustOrgan:
    """
    Root of trust organ that can be backed by TPMBackend or real TPM.
    """

    def __init__(self, backend: Optional[TPMBackend] = None):
        self.backend = backend or TPMBackend()

    def get_device_secret(self) -> bytes:
        return self.backend.get_device_secret()


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
    mac_address: str
    timestamp: float = field(default_factory=lambda: time.time())


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
# Audit organ + Forensics organ
# ============================================================

class AuditOrgan:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

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


class ForensicsOrgan:
    """
    Builds per-capsule and per-host forensic views + reputation.
    """

    def __init__(self, audit_organ: AuditOrgan):
        self.audit = audit_organ
        self.capsule_reputation: Dict[str, float] = {}
        self.host_reputation: Dict[str, float] = {}

    def recompute(self):
        self.capsule_reputation.clear()
        self.host_reputation.clear()
        for e in self.audit.export_events():
            cid = e["capsule_id"]
            env = e["payload"].get("env", {})
            host_fp = env.get("host_fingerprint", "unknown")
            etype = e["event_type"]

            delta_capsule = 0.0
            delta_host = 0.0

            if etype == "access_granted":
                delta_capsule += 0.1
                delta_host += 0.1
            elif etype == "violation":
                delta_capsule -= 1.0
                delta_host -= 1.0
            elif etype == "rate_limited":
                delta_capsule -= 0.3
                delta_host -= 0.3
            elif etype == "access_attempt_on_destroyed":
                delta_capsule -= 0.5
                delta_host -= 0.5

            self.capsule_reputation[cid] = self.capsule_reputation.get(cid, 0.0) + delta_capsule
            self.host_reputation[host_fp] = self.host_reputation.get(host_fp, 0.0) + delta_host

    def get_capsule_reputation(self, capsule_id: str) -> float:
        return self.capsule_reputation.get(capsule_id, 0.0)

    def get_host_reputation(self, host_fingerprint: str) -> float:
        return self.host_reputation.get(host_fingerprint, 0.0)

    def get_capsule_timeline(self, capsule_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.audit.export_events() if e["capsule_id"] == capsule_id]


# ============================================================
# Beacon organ
# ============================================================

class BeaconOrgan:
    def __init__(self):
        self._beacons: List[Dict[str, Any]] = []

    def emit(self, capsule_id: str, reason: str, env_snapshot: Dict[str, Any]) -> None:
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
# Swarm coordinator state + Raft-style log
# ============================================================

class SwarmCoordinatorState:
    def __init__(self):
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.policy_version: int = 0
        self.policy_blob: Dict[str, Any] = {}
        # Raft-style log (simplified)
        self.raft_log: List[Dict[str, Any]] = []
        self.current_term: int = 0
        self.leader_id: Optional[str] = None


COORD_STATE = SwarmCoordinatorState()


# ============================================================
# Swarm coordinator HTTP handler (encrypted channels via HTTPS + shared secret)
# ============================================================

SWARM_SHARED_SECRET = os.getenv("SWARM_SHARED_SECRET", "change-me-shared-secret")


class SwarmCoordinatorHandler(BaseHTTPRequestHandler):
    def _auth_ok(self) -> bool:
        # Simple HMAC-like shared secret header
        token = self.headers.get("X-SWARM-TOKEN", "")
        expected = hashlib.sha256(SWARM_SHARED_SECRET.encode()).hexdigest()
        return token == expected

    def _send_json(self, code: int, payload: Dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        if not self._auth_ok():
            self._send_json(403, {"error": "unauthorized"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid_json"})
            return

        if self.path == "/register":
            swarm_id = data.get("swarm_id")
            node_id = data.get("node_id")
            info = data.get("info", {})
            if not swarm_id or not node_id:
                self._send_json(400, {"error": "missing_fields"})
                return
            COORD_STATE.peers[node_id] = {"swarm_id": swarm_id, "info": info}
            self._send_json(200, {"status": "ok", "peers": COORD_STATE.peers})
            return

        if self.path == "/policy":
            swarm_id = data.get("swarm_id")
            node_id = data.get("node_id")
            policy = data.get("policy", {})
            if not swarm_id or not node_id:
                self._send_json(400, {"error": "missing_fields"})
                return

            # Raft-style append (simplified: leader only)
            if COORD_STATE.leader_id is None:
                COORD_STATE.leader_id = node_id

            if node_id != COORD_STATE.leader_id:
                self._send_json(403, {"error": "not_leader"})
                return

            COORD_STATE.current_term += 1
            entry = {
                "term": COORD_STATE.current_term,
                "policy": policy,
                "ts": time.time(),
            }
            COORD_STATE.raft_log.append(entry)
            COORD_STATE.policy_version += 1
            COORD_STATE.policy_blob = policy

            self._send_json(200, {
                "status": "ok",
                "version": COORD_STATE.policy_version,
                "policy": COORD_STATE.policy_blob,
                "term": COORD_STATE.current_term,
            })
            return

        self._send_json(404, {"error": "not_found"})

    def do_GET(self):
        if not self._auth_ok():
            self._send_json(403, {"error": "unauthorized"})
            return

        if self.path.startswith("/policy"):
            self._send_json(200, {
                "version": COORD_STATE.policy_version,
                "policy": COORD_STATE.policy_blob,
                "term": COORD_STATE.current_term,
            })
            return
        if self.path.startswith("/peers"):
            self._send_json(200, {"peers": COORD_STATE.peers})
            return
        if self.path.startswith("/raft_log"):
            self._send_json(200, {"log": COORD_STATE.raft_log})
            return
        self._send_json(404, {"error": "not_found"})


def start_swarm_coordinator_server(port: int = 9443, certfile: Optional[str] = None, keyfile: Optional[str] = None):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, SwarmCoordinatorHandler)
    if certfile and keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# ============================================================
# Swarm sync organ (networked, HTTPS + shared secret)
# ============================================================

class SwarmSyncOrgan:
    """
    Networked swarm sync:
      - pulls policy from coordinator
      - pushes updates (leader only)
      - tracks peers + versions
      - uses HTTPS + shared secret header
    """

    def __init__(self, swarm_id: str, node_id: str, coord_url: str, shared_secret: str):
        if not coord_url.lower().startswith("https://"):
            raise ValueError("Swarm coordinator URL must use HTTPS")
        self.swarm_id = swarm_id
        self.node_id = node_id
        self.coord_url = coord_url.rstrip("/")
        self.shared_secret = shared_secret
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.policy_version: int = 0
        self.policy_blob: Dict[str, Any] = {}
        self.term: int = 0

    def _headers(self) -> Dict[str, str]:
        token = hashlib.sha256(self.shared_secret.encode()).hexdigest()
        return {"X-SWARM-TOKEN": token}

    def register_peer(self, info: Dict[str, Any]):
        payload = {
            "swarm_id": self.swarm_id,
            "node_id": self.node_id,
            "info": info,
        }
        try:
            r = requests.post(
                f"{self.coord_url}/register",
                json=payload,
                timeout=3,
                verify=False,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            self.peers = data.get("peers", {})
        except Exception as e:
            print(f"[SwarmSync] register_peer failed: {e}")

    def pull_policy(self):
        try:
            r = requests.get(
                f"{self.coord_url}/policy",
                params={"swarm_id": self.swarm_id},
                timeout=3,
                verify=False,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            version = data.get("version", 0)
            blob = data.get("policy", {})
            term = data.get("term", 0)
            if version > self.policy_version or term > self.term:
                self.receive_policy(blob, version, term)
        except Exception as e:
            print(f"[SwarmSync] pull_policy failed: {e}")

    def push_policy(self, policy_blob: Dict[str, Any]):
        try:
            payload = {
                "swarm_id": self.swarm_id,
                "node_id": self.node_id,
                "policy": policy_blob,
            }
            r = requests.post(
                f"{self.coord_url}/policy",
                json=payload,
                timeout=3,
                verify=False,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            version = data.get("version", 0)
            blob = data.get("policy", {})
            term = data.get("term", 0)
            self.receive_policy(blob, version, term)
        except Exception as e:
            print(f"[SwarmSync] push_policy failed: {e}")

    def receive_policy(self, policy_blob: Dict[str, Any], version: int, term: int):
        if term > self.term or version > self.policy_version:
            self.policy_version = version
            self.policy_blob = policy_blob
            self.term = term

    def get_peers(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.peers)

    def get_policy(self) -> Dict[str, Any]:
        return {
            "version": self.policy_version,
            "policy": self.policy_blob,
            "term": self.term,
        }


# ============================================================
# Raft-style consensus organ
# ============================================================

class RaftConsensusOrgan:
    """
    Simplified Raft-style consensus wrapper:
      - if is_leader: can push policy
      - if follower: only pulls
    """

    def __init__(self, swarm_sync: SwarmSyncOrgan, is_leader: bool):
        self.swarm_sync = swarm_sync
        self.is_leader = is_leader

    def update_policy(self, policy_blob: Dict[str, Any]):
        if not self.is_leader:
            print("[RaftConsensus] Not leader; refusing to update policy")
            return
        self.swarm_sync.push_policy(policy_blob)

    def refresh_policy(self):
        self.swarm_sync.pull_policy()

    def get_policy(self) -> Dict[str, Any]:
        return self.swarm_sync.get_policy()


# ============================================================
# Policy organ
# ============================================================

class PolicyOrgan:
    def __init__(self, audit_organ: AuditOrgan, consensus: RaftConsensusOrgan, forensics: ForensicsOrgan):
        self.audit_organ = audit_organ
        self.consensus = consensus
        self.forensics = forensics

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

        swarm_policy = self.consensus.get_policy().get("policy", {})
        max_global_risk = swarm_policy.get("max_global_risk", None)
        if max_global_risk is not None and env.risk_score > max_global_risk:
            return {
                "allowed": False,
                "reason": "exceeds_global_risk_policy",
                "degrade": True
            }

        # Reputation-based tweak
        host_fp = hashlib.sha256(env.host_id.encode()).hexdigest()
        host_rep = self.forensics.get_host_reputation(host_fp)
        if host_rep < -5.0:
            return {
                "allowed": False,
                "reason": "host_reputation_too_low",
                "degrade": True
            }

        return {
            "allowed": True,
            "reason": "policy_ok",
            "degrade": False
        }


# ============================================================
# Key organ (AES-GCM, env-bound + root of trust)
# ============================================================

class KeyOrgan:
    def __init__(self, root_of_trust: RootOfTrustOrgan):
        self.root_of_trust = root_of_trust

    def derive_key(self,
                   key_ref: str,
                   env: EnvironmentSnapshot,
                   key_policy: Dict[str, Any],
                   chameleon_salt: Optional[bytes] = None) -> bytes:
        device_secret = self.root_of_trust.get_device_secret()
        material = {
            "key_ref": key_ref,
            "host_id": env.host_id if key_policy.get("bind_to_host") else "",
            "swarm_id": env.swarm_id if key_policy.get("bind_to_swarm") else "",
            "geo": env.geo if key_policy.get("bind_to_geo") else "",
            "mac": env.mac_address if key_policy.get("bind_to_mac") else "",
            "salt": chameleon_salt.hex() if chameleon_salt else "",
            "device_secret": hashlib.sha256(device_secret).hexdigest(),
        }
        blob = json.dumps(material, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).digest()

    def encrypt(self, plaintext_obj: Any, key: bytes) -> bytes:
        raw = json.dumps(plaintext_obj, sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, compressed, None)
        return nonce + ct

    def decrypt(self, ciphertext: bytes, key: bytes) -> Any:
        nonce = ciphertext[:12]
        ct = ciphertext[12:]
        aesgcm = AESGCM(key)
        compressed = aesgcm.decrypt(nonce, ct, None)
        raw = zlib.decompress(compressed)
        return json.loads(raw.decode("utf-8"))


# ============================================================
# Rate limiting / backoff organ
# ============================================================

class RateLimiterOrgan:
    def __init__(self,
                 base_backoff: float = 1.0,
                 max_backoff: float = 60.0,
                 max_attempts: int = 10):
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.max_attempts = max_attempts
        self.state: Dict[str, Dict[str, Any]] = {}

    def _key(self, capsule_id: str, env: EnvironmentSnapshot) -> str:
        return f"{capsule_id}:{hashlib.sha256(env.host_id.encode()).hexdigest()}"

    def check(self, capsule_id: str, env: EnvironmentSnapshot) -> Dict[str, Any]:
        k = self._key(capsule_id, env)
        now = time.time()
        info = self.state.get(k, {"attempts": 0, "next_allowed_ts": 0.0, "locked": False})

        if info["locked"]:
            return {"allowed": False, "reason": "locked_out"}

        if now < info["next_allowed_ts"]:
            return {"allowed": False, "reason": "backoff_active"}

        return {"allowed": True, "reason": "ok"}

    def record_failure(self, capsule_id: str, env: EnvironmentSnapshot):
        k = self._key(capsule_id, env)
        now = time.time()
        info = self.state.get(k, {"attempts": 0, "next_allowed_ts": 0.0, "locked": False})

        info["attempts"] += 1
        backoff = min(self.base_backoff * (2 ** (info["attempts"] - 1)), self.max_backoff)
        info["next_allowed_ts"] = now + backoff

        if info["attempts"] >= self.max_attempts:
            info["locked"] = True

        self.state[k] = info

    def record_success(self, capsule_id: str, env: EnvironmentSnapshot):
        k = self._key(capsule_id, env)
        self.state[k] = {"attempts": 0, "next_allowed_ts": 0.0, "locked": False}


# ============================================================
# Migration engine (versioned)
# ============================================================

CURRENT_SCHEMA_VERSION = "1.2"


class MigrationEngine:
    def __init__(self):
        self.decryptors: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.migrations: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register_decryptor(self, version: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.decryptors[version] = fn

    def register_migration(self, from_version: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.migrations[from_version] = fn

    def normalize(self, version: str) -> str:
        return str(version)

    def decrypt_and_migrate(self, capsule: "DataCapsule", plaintext: Dict[str, Any]) -> Dict[str, Any]:
        version = self.normalize(capsule.schema_version)

        if version in self.decryptors:
            plaintext = self.decryptors[version](plaintext)

        while version in self.migrations and version != CURRENT_SCHEMA_VERSION:
            migrate_fn = self.migrations[version]
            plaintext = migrate_fn(plaintext)
            major, minor = version.split(".")
            version = f"{major}.{int(minor) + 1}"

        return plaintext


def decrypt_1_0(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload


def migrate_1_0_to_1_1(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "created_at" not in payload:
        payload["created_at"] = time.time()
    return payload


def migrate_1_1_to_1_2(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "apiKey" in payload and "api_key" not in payload:
        payload["api_key"] = payload.pop("apiKey")
    return payload


# ============================================================
# Data capsule + cloning / replication
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
    encryption_scheme: str = "AES-GCM+ZLIB"
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
        ciphertext = key_organ.encrypt(plaintext_obj, key)

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

    def clone(self, new_capsule_id: str, new_owner_id: Optional[str] = None) -> "DataCapsule":
        """
        Clone capsule metadata + ciphertext, update lineage.
        """
        lineage = dict(self.lineage)
        lineage["parent_id"] = self.capsule_id
        return DataCapsule(
            capsule_id=new_capsule_id,
            owner_id=new_owner_id or self.owner_id,
            sensitivity_level=self.sensitivity_level,
            schema_version=self.schema_version,
            lineage=lineage,
            allowed_hosts=list(self.allowed_hosts),
            allowed_geos=list(self.allowed_geos),
            allowed_processes=list(self.allowed_processes),
            allowed_swarms=list(self.allowed_swarms),
            time_not_before=self.time_not_before,
            time_not_after=self.time_not_after,
            allowed_roles=list(self.allowed_roles),
            allowed_agents=list(self.allowed_agents),
            max_risk_score=self.max_risk_score,
            export_policies=dict(self.export_policies),
            on_violation=self.on_violation,
            degradation_mode=self.degradation_mode,
            audit_level=self.audit_level,
            beacon_on_violation=self.beacon_on_violation,
            ciphertext=self.ciphertext,
            encryption_scheme=self.encryption_scheme,
            key_ref=self.key_ref,
            key_policy=dict(self.key_policy),
            chameleon_salt=self.chameleon_salt,
            destroyed=self.destroyed,
        )

    def access(self,
               env: EnvironmentSnapshot,
               agent_id: str,
               key_organ: KeyOrgan,
               policy_organ: PolicyOrgan,
               audit_organ: AuditOrgan,
               beacon_organ: BeaconOrgan,
               rate_limiter: RateLimiterOrgan,
               migration_engine: MigrationEngine) -> Optional[Any]:
        if self.destroyed:
            audit_organ.log(self.capsule_id, "access_attempt_on_destroyed", {
                "env": sanitize_env_for_log(env),
                "agent_id": agent_id
            })
            return None

        rl_check = rate_limiter.check(self.capsule_id, env)
        if not rl_check["allowed"]:
            audit_organ.log(self.capsule_id, "rate_limited", {
                "env": sanitize_env_for_log(env),
                "agent_id": agent_id,
                "reason": rl_check["reason"],
            })
            return None

        decision = self._evaluate_environment(env, agent_id, policy_organ)

        if not decision["allowed"]:
            rate_limiter.record_failure(self.capsule_id, env)
            self._handle_violation(decision, env, audit_organ, beacon_organ)
            return decision.get("degraded_payload")

        key = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        plaintext = key_organ.decrypt(self.ciphertext, key)

        plaintext = migration_engine.decrypt_and_migrate(self, plaintext)

        rate_limiter.record_success(self.capsule_id, env)

        if self.audit_level == "full":
            audit_organ.log(self.capsule_id, "access_granted", {
                "env": sanitize_env_for_log(env),
                "agent_id": agent_id,
                "reason": decision["reason"]
            })

        return plaintext

    def morph_ciphertext(self, env: EnvironmentSnapshot, key_organ: KeyOrgan):
        if self.destroyed or not self.ciphertext:
            return

        key_old = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        plaintext = key_organ.decrypt(self.ciphertext, key_old)

        self.chameleon_salt = os.urandom(16)
        key_new = key_organ.derive_key(self.key_ref, env, self.key_policy, chameleon_salt=self.chameleon_salt)
        self.ciphertext = key_organ.encrypt(plaintext, key_new)

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
        if self.ciphertext:
            buf = bytearray(self.ciphertext)
            secure_wipe_bytes(buf)
        self.ciphertext = b""
        self.destroyed = True

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
    def __init__(self, swarm_id: str = "default-swarm", node_id: str = "node-1", is_leader: bool = False):
        self.root_of_trust = RootOfTrustOrgan()
        self.audit = AuditOrgan()
        self.forensics = ForensicsOrgan(self.audit)
        self.beacon = BeaconOrgan()
        self.rate_limiter = RateLimiterOrgan()
        self.migration_engine = MigrationEngine()
        self.migration_engine.register_decryptor("1.0", decrypt_1_0)
        self.migration_engine.register_migration("1.0", migrate_1_0_to_1_1)
        self.migration_engine.register_migration("1.1", migrate_1_1_to_1_2)

        self.keys = KeyOrgan(self.root_of_trust)

        coord_url = os.getenv("SWARM_COORD_URL", "https://localhost:9443")
        shared_secret = os.getenv("SWARM_SHARED_SECRET", "change-me-shared-secret")

        self.swarm_sync = SwarmSyncOrgan(
            swarm_id=swarm_id,
            node_id=node_id,
            coord_url=coord_url,
            shared_secret=shared_secret,
        )
        self.consensus = RaftConsensusOrgan(self.swarm_sync, is_leader=is_leader)
        self.policy = PolicyOrgan(self.audit, self.consensus, self.forensics)

        self.swarm_sync.pull_policy()

        self.capsule_states: Dict[str, Dict[str, Any]] = {}
        self.lineage_graph: Dict[str, List[str]] = {}
        self.threat_heatmap: Dict[str, int] = {}
        self.network_edges: List[Tuple[str, str, str]] = []  # (from_host, to_host, reason)

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
            beacon_organ=self.beacon,
            rate_limiter=self.rate_limiter,
            migration_engine=self.migration_engine,
        )

        state = "destroyed" if capsule.destroyed else (
            "degraded" if isinstance(result, dict) and result.get("status") == "degraded" else "alive"
        )
        self.capsule_states[capsule.capsule_id] = {
            "state": state,
            "last_access": time.time(),
        }

        parent = capsule.lineage.get("parent_id")
        if parent:
            self.lineage_graph.setdefault(parent, []).append(capsule.capsule_id)

        if state != "alive":
            self.threat_heatmap[capsule.capsule_id] = self.threat_heatmap.get(capsule.capsule_id, 0) + 1

        # Update forensics + network edges
        self.forensics.recompute()
        host_fp = hashlib.sha256(env.host_id.encode()).hexdigest()
        if state != "alive":
            self.network_edges.append((host_fp, "capsule:"+capsule.capsule_id, state))

        return result

    def clone_capsule(self, capsule: DataCapsule, new_capsule_id: str, new_owner_id: Optional[str] = None) -> DataCapsule:
        clone = capsule.clone(new_capsule_id=new_capsule_id, new_owner_id=new_owner_id)
        parent = capsule.capsule_id
        self.lineage_graph.setdefault(parent, []).append(clone.capsule_id)
        return clone

# ============================================================
# Storage backend
# ============================================================

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

        if result is not None and capsule.schema_version != CURRENT_SCHEMA_VERSION:
            capsule.schema_version = CURRENT_SCHEMA_VERSION
            key = self.runtime.keys.derive_key(
                capsule.key_ref,
                env,
                capsule.key_policy,
                chameleon_salt=capsule.chameleon_salt,
            )
            capsule.ciphertext = self.runtime.keys.encrypt(result, key)
            record["schema_version"] = capsule.schema_version
            record["ciphertext"] = capsule.ciphertext.hex()
            self.storage.write(key, record)

        return result


# ============================================================
# GUI: signal bridge + listener thread
# ============================================================

class BeaconSignalBridge(QObject):
    beacon_received = pyqtSignal(dict)


class BeaconListenerThread(QThread):
    def __init__(self, runtime: CapsuleRuntime, signal_bridge: BeaconSignalBridge, poll_interval: float = 1.0):
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

        self.threat_feed = QTextEdit()
        self.threat_feed.setReadOnly(True)
        self.threat_feed.setPlaceholderText("Threat events will appear here...")
        main_layout.addWidget(self.threat_feed)

        self.capsule_table = QTableWidget(0, 5)
        self.capsule_table.setHorizontalHeaderLabels(["Capsule ID", "State", "Last Access", "Threat Count", "Reputation"])
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
        heatmap = self.runtime.threat_heatmap
        self.runtime.forensics.recompute()
        self.capsule_table.setRowCount(len(states))
        for row, (capsule_id, info) in enumerate(states.items()):
            state = info.get("state", "unknown")
            last_access_ts = info.get("last_access", 0)
            last_access_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_access_ts)) if last_access_ts else "-"
            threat_count = heatmap.get(capsule_id, 0)
            rep = self.runtime.forensics.get_capsule_reputation(capsule_id)

            self.capsule_table.setItem(row, 0, QTableWidgetItem(capsule_id))
            self.capsule_table.setItem(row, 1, QTableWidgetItem(state))
            self.capsule_table.setItem(row, 2, QTableWidgetItem(last_access_str))
            self.capsule_table.setItem(row, 3, QTableWidgetItem(str(threat_count)))
            self.capsule_table.setItem(row, 4, QTableWidgetItem(f"{rep:.2f}"))


# ============================================================
# System Status Tab
# ============================================================

class SystemStatusTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime, signal_bridge: BeaconSignalBridge):
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
# Capsule Inspector Tab (with basic forensics)
# ============================================================

class CapsuleInspectorTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime):
        super().__init__()
        self.runtime = runtime

        layout = QVBoxLayout()

        self.info_label = QLabel("Capsule Inspector")
        self.info_label.setObjectName("sectionTitle")
        layout.addWidget(self.info_label)

        self.inspect_feed = QTextEdit()
        self.inspect_feed.setReadOnly(True)
        self.inspect_feed.setPlaceholderText("Capsule details and forensics will appear here...")
        layout.addWidget(self.inspect_feed)

        self.setLayout(layout)

    def refresh(self):
        self.inspect_feed.clear()
        self.runtime.forensics.recompute()
        for capsule_id, state in self.runtime.capsule_states.items():
            last_access_ts = state.get("last_access", 0)
            last_access_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_access_ts)) if last_access_ts else "-"
            rep = self.runtime.forensics.get_capsule_reputation(capsule_id)
            line = (
                f"Capsule: {capsule_id}\n"
                f"  State: {state.get('state')}\n"
                f"  Last Access: {last_access_str}\n"
                f"  Reputation: {rep:.2f}\n"
                "  Timeline:\n"
            )
            self.inspect_feed.append(line)
            for ev in self.runtime.forensics.get_capsule_timeline(capsule_id):
                ets = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev["ts"]))
                self.inspect_feed.append(f"    [{ets}] {ev['event_type']} {ev['payload']}")
            self.inspect_feed.append("----------------------------------------")


# ============================================================
# Lineage Graph Tab
# ============================================================

class LineageGraphTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime):
        super().__init__()
        self.runtime = runtime

        layout = QVBoxLayout()

        self.title_label = QLabel("Lineage Graph")
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)

        self.graph_view = QTextEdit()
        self.graph_view.setReadOnly(True)
        self.graph_view.setPlaceholderText("Lineage relationships will appear here...")
        layout.addWidget(self.graph_view)

        self.setLayout(layout)

    def refresh(self):
        self.graph_view.clear()
        for parent, children in self.runtime.lineage_graph.items():
            self.graph_view.append(f"Parent: {parent}")
            for child in children:
                self.graph_view.append(f"  -> Child: {child}")
            self.graph_view.append("----------------------------------------")


# ============================================================
# Network Threat Map Tab
# ============================================================

class NetworkThreatMapTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime):
        super().__init__()
        self.runtime = runtime

        layout = QVBoxLayout()

        self.title_label = QLabel("Network Threat Map")
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)

        self.map_view = QTextEdit()
        self.map_view.setReadOnly(True)
        self.map_view.setPlaceholderText("Threat edges will appear here...")
        layout.addWidget(self.map_view)

        self.setLayout(layout)

    def refresh(self):
        self.map_view.clear()
        for (src, dst, reason) in self.runtime.network_edges:
            self.map_view.append(f"{src}  ==>  {dst}   ({reason})")
        if not self.runtime.network_edges:
            self.map_view.append("No threat edges recorded yet.")


# ============================================================
# Policy Editor Tab
# ============================================================

class PolicyEditorTab(QWidget):
    def __init__(self, runtime: CapsuleRuntime):
        super().__init__()
        self.runtime = runtime

        layout = QVBoxLayout()

        self.title_label = QLabel("Swarm Policy Editor")
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)

        form = QFormLayout()

        self.max_risk_input = QLineEdit()
        self.max_risk_input.setPlaceholderText("e.g. 0.5")
        form.addRow("Max Global Risk:", self.max_risk_input)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["strict", "balanced", "relaxed"])
        form.addRow("Policy Mode:", self.mode_combo)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Current Policy")
        self.save_btn = QPushButton("Save Policy (Leader Only)")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.load_btn.clicked.connect(self.on_load_policy)
        self.save_btn.clicked.connect(self.on_save_policy)

    def on_load_policy(self):
        pol = self.runtime.consensus.get_policy().get("policy", {})
        max_risk = pol.get("max_global_risk", "")
        mode = pol.get("mode", "balanced")
        self.max_risk_input.setText(str(max_risk))
        idx = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.status_label.setText("Policy loaded from swarm.")

    def on_save_policy(self):
        try:
            max_risk = float(self.max_risk_input.text()) if self.max_risk_input.text() else None
        except ValueError:
            self.status_label.setText("Invalid max risk value.")
            return

        policy = {
            "max_global_risk": max_risk,
            "mode": self.mode_combo.currentText(),
        }
        self.runtime.consensus.update_policy(policy)
        self.status_label.setText("Policy update requested (leader only).")


# ============================================================
# Main Window with Tabs
# ============================================================

class CapsuleMainWindow(QWidget):
    def __init__(self, runtime: CapsuleRuntime, signal_bridge: BeaconSignalBridge):
        super().__init__()
        self.runtime = runtime
        self.signal_bridge = signal_bridge

        self.setWindowTitle("Capsule Protection Console")
        self.setMinimumSize(1000, 700)

        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")

        self.status_tab = SystemStatusTab(runtime, signal_bridge)
        self.threat_tab = ThreatDashboardTab(runtime, signal_bridge)
        self.inspector_tab = CapsuleInspectorTab(runtime)
        self.lineage_tab = LineageGraphTab(runtime)
        self.network_tab = NetworkThreatMapTab(runtime)
        self.policy_tab = PolicyEditorTab(runtime)

        self.tabs.addTab(self.status_tab, "System Status")
        self.tabs.addTab(self.threat_tab, "Threat Dashboard")
        self.tabs.addTab(self.inspector_tab, "Capsule Inspector")
        self.tabs.addTab(self.lineage_tab, "Lineage Graph")
        self.tabs.addTab(self.network_tab, "Network Threat Map")
        self.tabs.addTab(self.policy_tab, "Policy Editor")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def refresh_inspector(self):
        self.inspector_tab.refresh()

    def refresh_lineage(self):
        self.lineage_tab.refresh()

    def refresh_network(self):
        self.network_tab.refresh()


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

QLineEdit {
    background-color: #000000;
    color: #E0E0E0;
    border: 1px solid #333333;
}

QComboBox {
    background-color: #000000;
    color: #E0E0E0;
    border: 1px solid #333333;
}

QPushButton {
    background-color: #222222;
    color: #E0E0E0;
    border: 1px solid #444444;
    padding: 4px 8px;
}

QPushButton:hover {
    background-color: #333333;
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

    listener_thread = BeaconListenerThread(runtime, signal_bridge)
    listener_thread.start()

    main_window.show()
    main_window.refresh_inspector()
    main_window.refresh_lineage()
    main_window.refresh_network()

    exit_code = app.exec()
    listener_thread.stop()
    return exit_code


# ============================================================
# Example usage + GUI launch
# ============================================================

if __name__ == "__main__":
    # Start local HTTPS swarm coordinator (self-signed or provided certs)
    certfile = None
    keyfile = None
    start_swarm_coordinator_server(port=9443, certfile=certfile, keyfile=keyfile)

    os.environ.setdefault("SWARM_COORD_URL", "https://localhost:9443")
    os.environ.setdefault("SWARM_SHARED_SECRET", "change-me-shared-secret")

    runtime = CapsuleRuntime(swarm_id="prod-swarm", node_id="leader", is_leader=True)
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

