import os
import json
import time
import math
import random
import uuid
import socket
import threading
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
#  OPTIONAL DEPENDENCIES (DearPyGUI, psutil, cryptography)
# ============================================================

HAS_DPG = False
HAS_PSUTIL = False
HAS_CRYPTO = False

try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except Exception:
    HAS_DPG = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


# ============================================================
#  BASIC UTILITIES / STUBS
# ============================================================

GUI_LISTENERS: List[Any] = []


def register_gui_listener(fn):
    GUI_LISTENERS.append(fn)


def send_beacon(event: Dict[str, Any]) -> None:
    for fn in GUI_LISTENERS:
        try:
            fn(event)
        except Exception:
            pass


def get_environment_fingerprint() -> Dict[str, Any]:
    try:
        import getpass

        user = getpass.getuser()
    except Exception:
        user = "unknown"

    return {
        "hostname": socket.gethostname(),
        "user": user,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "geo": "unknown",
    }


# ============================================================
#  CRYPTO ORGAN
# ============================================================

class CryptoOrgan:
    def __init__(self, key_label: str = "species_master_key") -> None:
        self.key_label = key_label
        self._fernet = self._load_or_create_key()

    def _key_path(self) -> Path:
        return Path(f"{self.key_label}.key")

    def _load_or_create_key(self) -> Fernet:
        if not HAS_CRYPTO:
            raise RuntimeError("cryptography is required for CryptoOrgan")
        path = self._key_path()
        if path.exists():
            key = path.read_bytes()
        else:
            key = Fernet.generate_key()
            path.write_bytes(key)
        return Fernet(key)

    def encrypt(self, raw: bytes) -> bytes:
        return self._fernet.encrypt(raw)

    def decrypt(self, encrypted: bytes) -> bytes:
        try:
            return self._fernet.decrypt(encrypted)
        except InvalidToken:
            return b""


# ============================================================
#  SPECIES REGISTRY
# ============================================================

class SpeciesProfile:
    def __init__(
        self,
        species_id: str,
        protocol_version: str,
        default_policy: Dict[str, Any],
        risk_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.species_id = species_id
        self.protocol_version = protocol_version
        self.default_policy = default_policy
        self.risk_thresholds = risk_thresholds or {
            "deny_above": 0.7,
        }


class SpeciesRegistry:
    _profiles: Dict[str, SpeciesProfile] = {}

    @classmethod
    def register(cls, profile: SpeciesProfile) -> None:
        cls._profiles[profile.species_id] = profile

    @classmethod
    def get(cls, species_id: str) -> SpeciesProfile:
        if species_id not in cls._profiles:
            cls._profiles[species_id] = SpeciesProfile(
                species_id=species_id,
                protocol_version="1.0",
                default_policy={
                    "allowed_hosts": None,
                    "allowed_geos": None,
                    "allow_export": False,
                },
            )
        return cls._profiles[species_id]


SpeciesRegistry.register(
    SpeciesProfile(
        species_id="species-oxygen",
        protocol_version="1.0",
        default_policy={
            "allowed_hosts": None,
            "allowed_geos": None,
            "allow_export": False,
        },
        risk_thresholds={"deny_above": 0.7},
    )
)
SpeciesRegistry.register(
    SpeciesProfile(
        species_id="species-silver",
        protocol_version="1.0",
        default_policy={
            "allowed_hosts": None,
            "allowed_geos": None,
            "allow_export": False,
        },
        risk_thresholds={"deny_above": 0.8},
    )
)


# ============================================================
#  UI AUTOMATION ORGAN (STUB)
# ============================================================

class UIAutomationOrgan:
    def __init__(self) -> None:
        pass

    def get_active_context(self) -> Dict[str, Any]:
        return {
            "suspicious": False,
            "window_title": "",
            "app": "",
        }


# ============================================================
#  SYNC ORGAN (HIVE STUB)
# ============================================================

class SyncOrgan:
    def __init__(self, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def sync_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def sync_capsule_metadata(self, meta: Dict[str, Any]) -> None:
        pass

    def fetch_species_config(self, species_id: str) -> Optional[Dict[str, Any]]:
        return None

    def fetch_capsule_policy(self, capsule_id: str) -> Optional[Dict[str, Any]]:
        return None


# ============================================================
#  AUTOLOADER (STUB)
# ============================================================

class OrganAutoloader:
    def __init__(self) -> None:
        self.log_path = Path("autoloader_log.txt")

    def log(self, msg: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{time.time():.0f} {msg}\n")
        except Exception:
            pass


# ============================================================
#  BEHAVIOR HISTORY ORGAN
# ============================================================

class BehaviorHistoryOrgan:
    def __init__(self, path: Path = Path("behavior_history.json")) -> None:
        self.path = path
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception:
            pass

    def log_event(self, user: str, hostname: str, capsule_id: str, intent: str, allowed: bool) -> None:
        key = f"{user}@{hostname}"
        entry = {
            "ts": time.time(),
            "capsule_id": capsule_id,
            "intent": intent,
            "allowed": allowed,
        }
        self._data.setdefault(key, []).append(entry)
        self._data[key] = [e for e in self._data[key] if time.time() - e["ts"] < 7 * 24 * 3600]
        self._save()

    def anomaly_score(self, user: str, hostname: str) -> float:
        key = f"{user}@{hostname}"
        events = self._data.get(key, [])
        if not events:
            return 0.0
        now = time.time()
        recent = [e for e in events if now - e["ts"] < 3600]
        spike_factor = min(1.0, len(recent) / 20.0)
        night_events = [
            e for e in events
            if 0 <= time.localtime(e["ts"]).tm_hour <= 5
        ]
        night_factor = min(1.0, len(night_events) / 10.0)
        return min(1.0, 0.6 * spike_factor + 0.4 * night_factor)


# ============================================================
#  LOCAL REPUTATION ORGAN
# ============================================================

class LocalReputationOrgan:
    def __init__(self, path: Path = Path("local_reputation.json"), half_life_hours: float = 24.0) -> None:
        self.path = path
        self.half_life = half_life_hours * 3600.0
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception:
            pass

    def _decay(self, key: str) -> None:
        entry = self._data.get(key)
        if not entry:
            return
        now = time.time()
        last = entry.get("last_update", now)
        dt = now - last
        if dt <= 0:
            return
        score = entry.get("score", 0.0)
        if self.half_life > 0:
            decay_factor = 0.5 ** (dt / self.half_life)
            score *= decay_factor
        entry["score"] = max(0.0, min(score, 1.0))
        entry["last_update"] = now

    def _key(self, kind: str, identifier: str) -> str:
        return f"{kind}:{identifier}"

    def adjust(self, kind: str, identifier: str, delta: float) -> None:
        key = self._key(kind, identifier)
        if key not in self._data:
            self._data[key] = {"score": 0.0, "last_update": time.time()}
        self._decay(key)
        self._data[key]["score"] = max(0.0, min(self._data[key]["score"] + delta, 1.0))
        self._data[key]["last_update"] = time.time()
        self._save()

    def get_score(self, kind: str, identifier: str) -> float:
        key = self._key(kind, identifier)
        if key not in self._data:
            return 0.0
        self._decay(key)
        return self._data[key]["score"]


# ============================================================
#  SYSTEM METRICS + SYSTEM MONITOR ORGAN
# ============================================================

class SystemMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.capsule_bytes_in = 0
        self.capsule_bytes_out = 0
        self.disk_read_bytes = 0
        self.disk_write_bytes = 0
        self.net_sent_bytes = 0
        self.net_recv_bytes = 0
        self.top_processes: List[Dict[str, Any]] = []

    def update_capsule_in(self, n: int):
        with self.lock:
            self.capsule_bytes_in += max(0, n)

    def update_capsule_out(self, n: int):
        with self.lock:
            self.capsule_bytes_out += max(0, n)

    def update_system_io(self, disk_read: int, disk_write: int, net_sent: int, net_recv: int):
        with self.lock:
            self.disk_read_bytes = disk_read
            self.disk_write_bytes = disk_write
            self.net_sent_bytes = net_sent
            self.net_recv_bytes = net_recv

    def update_top_processes(self, procs: List[Dict[str, Any]]):
        with self.lock:
            self.top_processes = procs[:20]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "capsule_bytes_in": self.capsule_bytes_in,
                "capsule_bytes_out": self.capsule_bytes_out,
                "disk_read_bytes": self.disk_read_bytes,
                "disk_write_bytes": self.disk_write_bytes,
                "net_sent_bytes": self.net_sent_bytes,
                "net_recv_bytes": self.net_recv_bytes,
                "top_processes": list(self.top_processes),
            }


SYSTEM_METRICS = SystemMetrics()


class SystemMonitorOrgan:
    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.enabled = HAS_PSUTIL
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled:
            print("[SystemMonitorOrgan] psutil not available; organ disabled.")
            return

        def loop():
            while not self._stop:
                try:
                    disk = psutil.disk_io_counters()
                    net = psutil.net_io_counters()
                    disk_read = disk.read_bytes
                    disk_write = disk.write_bytes
                    net_sent = net.bytes_sent
                    net_recv = net.bytes_recv

                    SYSTEM_METRICS.update_system_io(
                        disk_read=disk_read,
                        disk_write=disk_write,
                        net_sent=net_sent,
                        net_recv=net_recv,
                    )
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True


# ============================================================
#  POLICY AI ORGAN
# ============================================================

class PolicyAIOrgan:
    def __init__(
        self,
        sync_organ: SyncOrgan,
        behavior_history: BehaviorHistoryOrgan,
        local_rep: LocalReputationOrgan,
    ) -> None:
        self.sync_organ = sync_organ
        self.behavior_history = behavior_history
        self.local_rep = local_rep

    def _base_decision(
        self,
        species_profile: SpeciesProfile,
        capsule_metadata: Dict[str, Any],
        env: Dict[str, Any],
        ui_context: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        policy = capsule_metadata.get("policy", {})
        allowed_hosts = policy.get("allowed_hosts")
        allowed_geos = policy.get("allowed_geos")

        if allowed_hosts is not None and env["hostname"] not in allowed_hosts:
            return {
                "allow": False,
                "risk_score": 0.9,
                "reason": "hostname_not_allowed",
            }

        if allowed_geos is not None and env["geo"] not in allowed_geos:
            return {
                "allow": False,
                "risk_score": 0.9,
                "reason": "geo_not_allowed",
            }

        if ui_context.get("suspicious", False):
            return {
                "allow": False,
                "risk_score": 0.95,
                "reason": "suspicious_viewer",
            }

        risk = 0.1
        if env.get("geo", "unknown") == "unknown":
            risk += 0.1

        return {
            "allow": True,
            "risk_score": risk,
            "reason": "baseline_allow",
        }

    def assess_access(
        self,
        species_id: str,
        capsule_metadata: Dict[str, Any],
        env: Dict[str, Any],
        ui_context: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        species_profile = SpeciesRegistry.get(species_id)
        decision = self._base_decision(species_profile, capsule_metadata, env, ui_context, intent)

        user = env.get("user", "unknown")
        hostname = env.get("hostname", "unknown")
        capsule_id = capsule_metadata["capsule_id"]

        anomaly = self.behavior_history.anomaly_score(user, hostname)
        host_score = self.local_rep.get_score("host", hostname)
        user_score = self.local_rep.get_score("user", user)
        cap_score = self.local_rep.get_score("capsule", capsule_id)

        combined_local_risk = min(1.0, (anomaly + host_score + user_score + cap_score) / 4.0)
        decision["risk_score"] = min(1.0, decision["risk_score"] + 0.5 * combined_local_risk)

        species_cfg = self.sync_organ.fetch_species_config(species_id) or {}
        thresholds = species_cfg.get("risk_thresholds", species_profile.risk_thresholds)

        hive_policy = self.sync_organ.fetch_capsule_policy(capsule_id)
        if hive_policy:
            if hive_policy.get("force_revoke"):
                return {
                    "allow": False,
                    "risk_score": 1.0,
                    "reason": "hive_force_revoke",
                }
            updated_policy = hive_policy.get("updated_policy")
            if updated_policy:
                capsule_metadata["policy"] = updated_policy
                decision = self._base_decision(species_profile, capsule_metadata, env, ui_context, intent)

        if decision["risk_score"] >= thresholds.get("deny_above", 0.7):
            decision["allow"] = False
            decision["reason"] = f"risk_above_deny_threshold:{decision['reason']}"

        return decision

    def record_outcome(
        self,
        env: Dict[str, Any],
        capsule_id: str,
        intent: str,
        allowed: bool,
        reason: str,
    ) -> None:
        user = env.get("user", "unknown")
        hostname = env.get("hostname", "unknown")
        self.behavior_history.log_event(user, hostname, capsule_id, intent, allowed)

        delta = 0.0
        if not allowed:
            delta = 0.2
        elif "exported_encrypted_capsule" in reason:
            delta = 0.05

        self.local_rep.adjust("host", hostname, delta)
        self.local_rep.adjust("user", user, delta)
        self.local_rep.adjust("capsule", capsule_id, delta)


# ============================================================
#  PREDICTION ORGAN
# ============================================================

class PredictionOrgan:
    def __init__(self, behavior_history: BehaviorHistoryOrgan, local_rep: LocalReputationOrgan) -> None:
        self.behavior_history = behavior_history
        self.local_rep = local_rep

    def _risk_from_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        series = []
        for e in sorted(events, key=lambda x: x.get("timestamp", 0)):
            ts = e.get("timestamp", 0)
            reason = e.get("reason") or ""
            w = 0.1
            if "read_denied" in reason or "forbidden" in reason:
                w = 1.0
            elif "destruct" in reason:
                w = 0.9
            elif "export" in reason:
                w = 0.5
            elif "read" in reason:
                w = 0.3
            series.append({"t": ts, "r": w})
        return series

    def _slope_and_volatility(self, series: List[Dict[str, float]]) -> (float, float):
        if len(series) < 2:
            return 0.0, 0.0
        t0 = series[0]["t"]
        t1 = series[-1]["t"]
        r0 = series[0]["r"]
        r1 = series[-1]["r"]
        dt = max(1.0, t1 - t0)
        slope = (r1 - r0) / dt

        mean = sum(p["r"] for p in series) / len(series)
        var = sum((p["r"] - mean) ** 2 for p in series) / len(series)
        vol = math.sqrt(var)
        return slope, vol

    def _predict_next_event_label(self, last_risk: float, slope: float) -> str:
        if last_risk >= 0.9:
            return "Destruct / hard deny likely"
        if last_risk >= 0.7 and slope > 0:
            return "Deny event likely soon"
        if slope > 0.001:
            return "Risk trending upward"
        if slope < -0.001:
            return "Risk trending downward"
        return "Stable / low movement"

    def predict_for_capsule(
        self,
        capsule_id: str,
        events: List[Dict[str, Any]],
        env: Dict[str, Any],
    ) -> Dict[str, Any]:
        series = self._risk_from_events(events)
        if not series:
            return {
                "capsule_id": capsule_id,
                "predicted_risk": 0.0,
                "slope": 0.0,
                "volatility": 0.0,
                "next_event_label": "No history yet",
                "trajectory": [],
            }

        slope, vol = self._slope_and_volatility(series)
        last_risk = series[-1]["r"]

        horizon = 600.0
        predicted_risk = last_risk + slope * horizon
        predicted_risk = max(0.0, min(1.0, predicted_risk))

        next_label = self._predict_next_event_label(last_risk, slope)

        trajectory = series[-50:]

        return {
            "capsule_id": capsule_id,
            "predicted_risk": predicted_risk,
            "slope": slope,
            "volatility": vol,
            "next_event_label": next_label,
            "trajectory": trajectory,
        }


# ============================================================
#  DATA CAPSULE
# ============================================================

class DataCapsule:
    def __init__(
        self,
        species_id: str,
        encrypted_payload: bytes,
        policy: Dict[str, Any],
        crypto: CryptoOrgan,
        ui_organ: UIAutomationOrgan,
        policy_ai: PolicyAIOrgan,
        sync_organ: SyncOrgan,
        safe_mode: bool = False,
        capsule_id: Optional[str] = None,
        demo: bool = False,
    ) -> None:
        raw_id = capsule_id or str(uuid.uuid4())
        self.demo = demo
        if demo:
            self.capsule_id = f"demo_capsule_{raw_id}"
        else:
            self.capsule_id = raw_id

        self.species_id = species_id
        profile = SpeciesRegistry.get(species_id)
        self.species = {
            "species_id": profile.species_id,
            "protocol_version": profile.protocol_version,
        }
        merged_policy = dict(profile.default_policy)
        merged_policy.update(policy or {})
        self.policy = merged_policy
        self._encrypted_payload = encrypted_payload
        self._destroyed = False
        self._crypto = crypto
        self._ui_organ = ui_organ
        self._policy_ai = policy_ai
        self._sync_organ = sync_organ
        self._created_at = time.time()
        self._safe_mode = safe_mode

        SYSTEM_METRICS.update_capsule_in(len(encrypted_payload))

    def _env(self) -> Dict[str, Any]:
        return get_environment_fingerprint()

    def _ui_context(self) -> Dict[str, Any]:
        return self._ui_organ.get_active_context()

    def _beacon(self, reason: str, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        event = {
            "capsule_id": self.capsule_id,
            "species": self.species,
            "reason": reason,
            "env": self._env(),
            "policy": self.policy,
            "destroyed": self._destroyed,
            "timestamp": time.time(),
            "demo": self.demo,
        }
        if extra:
            event.update(extra)
        send_beacon(event)
        return self._sync_organ.sync_event(event)

    def _self_destruct(self, reason: str) -> None:
        if self._safe_mode:
            self._beacon(f"safe_mode_block_instead_of_destruct:{reason}")
            return
        self._encrypted_payload = b""
        self._destroyed = True
        self._beacon(reason)

    def is_destroyed(self) -> bool:
        return self._destroyed

    def to_metadata(self) -> Dict[str, Any]:
        meta = {
            "capsule_id": self.capsule_id,
            "species": self.species,
            "policy": self.policy,
            "destroyed": self._destroyed,
            "created_at": self._created_at,
            "host_env": self._env(),
            "demo": self.demo,
        }
        self._sync_organ.sync_capsule_metadata(meta)
        return meta

    def read(self) -> Optional[bytes]:
        if self._destroyed:
            return None

        env = self._env()
        ui_ctx = self._ui_context()
        meta = self.to_metadata()

        decision = self._policy_ai.assess_access(
            species_id=self.species_id,
            capsule_metadata=meta,
            env=env,
            ui_context=ui_ctx,
            intent="read",
        )

        allowed = decision.get("allow", False)
        reason = decision.get("reason", "unknown")
        self._policy_ai.record_outcome(env, self.capsule_id, "read", allowed, reason)

        if not allowed:
            self._self_destruct(f"read_denied:{reason}")
            return None

        decrypted = self._crypto.decrypt(self._encrypted_payload)
        if not decrypted:
            self._self_destruct("decryption_failure")
            return None

        self._beacon("read")
        return decrypted

    def export(self, path: Path) -> bool:
        if self._destroyed:
            return False

        env = self._env()
        ui_ctx = self._ui_context()
        meta = self.to_metadata()

        decision = self._policy_ai.assess_access(
            species_id=self.species_id,
            capsule_metadata=meta,
            env=env,
            ui_context=ui_ctx,
            intent="export",
        )

        allowed = decision.get("allow", False)
        reason = decision.get("reason", "unknown")
        self._policy_ai.record_outcome(env, self.capsule_id, "export", allowed, reason)

        if not allowed or not self.policy.get("allow_export", False):
            self._self_destruct(f"forbidden_export_attempt:{reason}")
            return False

        path.write_bytes(self._encrypted_payload)
        SYSTEM_METRICS.update_capsule_out(len(self._encrypted_payload))
        self._beacon("exported_encrypted_capsule", {"export_path": str(path)})
        return True


# ============================================================
#  SPECIES COORDINATOR
# ============================================================

class SpeciesCoordinator:
    def __init__(
        self,
        autoloader: OrganAutoloader,
        crypto: CryptoOrgan,
        ui_organ: UIAutomationOrgan,
        policy_ai: PolicyAIOrgan,
        sync_organ: SyncOrgan,
        safe_mode: bool = False,
    ) -> None:
        self.autoloader = autoloader
        self.crypto = crypto
        self.ui_organ = ui_organ
        self.policy_ai = policy_ai
        self.sync_organ = sync_organ
        self.safe_mode = safe_mode
        self._capsules: Dict[str, DataCapsule] = {}

    def create_capsule(self, species_id: str, raw_payload: bytes, policy: Dict[str, Any], demo: bool = False) -> DataCapsule:
        encrypted = self.crypto.encrypt(raw_payload)
        capsule = DataCapsule(
            species_id=species_id,
            encrypted_payload=encrypted,
            policy=policy,
            crypto=self.crypto,
            ui_organ=self.ui_organ,
            policy_ai=self.policy_ai,
            sync_organ=self.sync_organ,
            safe_mode=self.safe_mode,
            demo=demo,
        )
        self._capsules[capsule.capsule_id] = capsule
        return capsule

    def register_capsule(self, capsule: DataCapsule) -> None:
        self._capsules[capsule.capsule_id] = capsule

    def list_capsules(self) -> Dict[str, Dict[str, Any]]:
        return {
            cid: cap.to_metadata()
            for cid, cap in self._capsules.items()
        }

    def get_capsule(self, capsule_id: str) -> Optional[DataCapsule]:
        return self._capsules.get(capsule_id)


# ============================================================
#  ACCESS-DRIVEN CAPSULE CREATION ORGAN
# ============================================================

class AccessDrivenCapsuleOrgan:
    def __init__(self, coordinator: SpeciesCoordinator, interval: float = 5.0, max_capsules: int = 500) -> None:
        self.coordinator = coordinator
        self.interval = interval
        self.max_capsules = max_capsules
        self.enabled = HAS_PSUTIL
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._seen_files: set[str] = set()

    def _classify_species_for_path(self, path: str) -> str:
        lower = path.lower()
        if "users" in lower and any(x in lower for x in ["documents", "desktop", "downloads", "pictures"]):
            return "species-oxygen"
        return "species-silver"

    def _is_local_data_path(self, path: str) -> bool:
        if platform.system() == "Windows":
            if len(path) >= 2 and path[1] == ":":
                drive = path[0].upper()
                return "C" <= drive <= "Z"
            return False
        else:
            return path.startswith("/")

    def _wrap_file(self, path: str, pid: int, pname: str):
        if len(self._seen_files) >= self.max_capsules:
            return
        if path in self._seen_files:
            return
        if not self._is_local_data_path(path):
            return

        self._seen_files.add(path)
        self._create_capsule_for_path(path, pid, pname)

    def _create_capsule_for_path(self, path: str, pid: int, pname: str):
        species_id = self._classify_species_for_path(path)
        env = get_environment_fingerprint()
        payload = {
            "path": path,
            "pid": pid,
            "process_name": pname,
            "hostname": env["hostname"],
            "user": env["user"],
            "created_ts": time.time(),
        }
        raw_payload = json.dumps(payload).encode("utf-8")

        policy = {
            "allowed_hosts": [env["hostname"]],
            "allowed_geos": None,
            "allow_export": False,
        }

        capsule = self.coordinator.create_capsule(species_id, raw_payload=raw_payload, policy=policy)
        send_beacon(
            {
                "capsule_id": capsule.capsule_id,
                "species": capsule.species,
                "reason": "auto_capsule_created",
                "env": env,
                "policy": capsule.policy,
                "destroyed": capsule.is_destroyed(),
                "timestamp": time.time(),
                "file_path": path,
                "pid": pid,
                "process_name": pname,
            }
        )

    def start(self):
        if not self.enabled:
            print("[AccessDrivenCapsuleOrgan] psutil not available; organ disabled.")
            return

        def loop():
            while not self._stop:
                try:
                    for p in psutil.process_iter(["pid", "name"]):
                        pid = p.info.get("pid")
                        pname = p.info.get("name") or "unknown"
                        try:
                            files = p.open_files()
                        except Exception:
                            continue
                        for f in files:
                            path = f.path
                            if not path:
                                continue
                            self._wrap_file(path, pid, pname)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True


# ============================================================
#  GUI STATE + THREAT GLYPHS + OVERLAYS
# ============================================================

GUI_STATE = {
    "coordinator": None,
    "capsules": [],
    "events": [],
    "capsule_events": {},
    "selected_capsule_id": None,
    "species_counts": {},
    "destroyed_counts": {},
    "safe_mode": False,
    "risk_cache": {},
    "capsule_row_tags": [],
    "overlay_phase": 0.0,
    "policy_edits": {},
    "prediction_organ": None,
    "prediction_cache": {},
}


def gui_event_listener(event: Dict[str, Any]) -> None:
    GUI_STATE["events"].append(event)
    GUI_STATE["events"] = GUI_STATE["events"][-400:]

    cid = event.get("capsule_id")
    if cid:
        GUI_STATE["capsule_events"].setdefault(cid, []).append(event)
        GUI_STATE["capsule_events"][cid] = GUI_STATE["capsule_events"][cid][-300:]


def estimate_risk_from_events(capsule_id: str) -> float:
    events = GUI_STATE["capsule_events"].get(capsule_id, [])
    now = time.time()
    recent = [e for e in events if now - e.get("timestamp", 0) < 3600]
    bad = [
        e for e in recent
        if "denied" in (e.get("reason") or "")
        or "forbidden" in (e.get("reason") or "")
        or "destruct" in (e.get("reason") or "")
    ]
    if not recent:
        return 0.0
    ratio = len(bad) / max(1, len(recent))
    return min(1.0, ratio)


def threat_glyph_for_risk(risk: float) -> str:
    if risk >= 0.9:
        return "💀"
    if risk >= 0.7:
        return "⚠️"
    if risk >= 0.4:
        return "🛡️"
    if risk > 0.0:
        return "🕊️"
    return "·"


def overlay_pulse_value() -> float:
    phase = GUI_STATE.get("overlay_phase", 0.0)
    phase = (phase + 0.08) % (2 * math.pi)
    GUI_STATE["overlay_phase"] = phase
    return 0.5 + 0.5 * math.sin(phase)


# ============================================================
#  GUI REFRESH, TIMELINE, PREDICTION, CALLBACKS
# ============================================================

def refresh_species_state(sender=None, app_data=None, user_data=None):
    coord: SpeciesCoordinator = GUI_STATE["coordinator"]
    if coord is None or not HAS_DPG:
        if HAS_DPG:
            dpg.set_frame_callback(dpg.get_frame_count() + 30, refresh_species_state)
        return

    meta_map = coord.list_capsules()
    rows = list(meta_map.values())
    GUI_STATE["capsules"] = rows

    species_counts: Dict[str, int] = {}
    destroyed_counts: Dict[str, int] = {}
    for r in rows:
        sid = r.get("species", {}).get("species_id", "unknown")
        species_counts[sid] = species_counts.get(sid, 0) + 1
        if r.get("destroyed"):
            destroyed_counts[sid] = destroyed_counts.get(sid, 0) + 1

    GUI_STATE["species_counts"] = species_counts
    GUI_STATE["destroyed_counts"] = destroyed_counts

    total = len(rows)
    destroyed = sum(1 for r in rows if r.get("destroyed"))
    species_str = ", ".join(f"{k}: {v}" for k, v in species_counts.items())
    if dpg.does_item_exist("overview_text"):
        dpg.set_value("overview_text", f"Total: {total} | Destroyed: {destroyed} | {species_str}")

    for tag in GUI_STATE["capsule_row_tags"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    GUI_STATE["capsule_row_tags"].clear()

    overlay_pulse_value()

    for r in rows:
        cid = r.get("capsule_id")
        sid = r.get("species", {}).get("species_id", "")
        destroyed_flag = "Yes" if r.get("destroyed") else "No"
        host = r.get("host_env", {}).get("hostname", "")
        user = r.get("host_env", {}).get("user", "")
        created = f"{r.get('created_at', 0):.0f}"

        risk = estimate_risk_from_events(cid)
        GUI_STATE["risk_cache"][cid] = risk
        glyph = threat_glyph_for_risk(risk)

        label = f"{glyph} {cid}"
        row_tag = f"capsule_row_{cid}"
        if dpg.does_item_exist("capsule_table"):
            with dpg.table_row(parent="capsule_table", tag=row_tag):
                dpg.add_selectable(label=label, user_data=cid, callback=capsule_row_clicked)
                dpg.add_text(sid)
                dpg.add_text(destroyed_flag)
                dpg.add_text(host)
                dpg.add_text(user)
                dpg.add_text(created)
            GUI_STATE["capsule_row_tags"].append(row_tag)

    stats_lines = []
    for sid, count in species_counts.items():
        d = destroyed_counts.get(sid, 0)
        stats_lines.append(f"{sid}: total={count}, destroyed={d}")
    stats_text = "\n".join(stats_lines)
    if dpg.does_item_exist("species_stats_text"):
        dpg.set_value("species_stats_text", stats_text)
    if dpg.does_item_exist("species_stats_text2"):
        dpg.set_value("species_stats_text2", stats_text)

    lines = []
    for e in reversed(GUI_STATE["events"]):
        line = (
            f"[{e.get('timestamp', 0):.0f}] {e.get('reason')} | "
            f"capsule={e.get('capsule_id')} | "
            f"host={e.get('env', {}).get('hostname')} | "
            f"user={e.get('env', {}).get('user')}"
        )
        lines.append(line)
    if dpg.does_item_exist("event_feed"):
        dpg.set_value("event_feed", "\n".join(lines))

    cid = GUI_STATE["selected_capsule_id"]
    if cid:
        meta = next((m for m in rows if m.get("capsule_id") == cid), None)
        if meta and dpg.does_item_exist("capsule_details"):
            dpg.set_value("capsule_details", json.dumps(meta, indent=2))
            risk = estimate_risk_from_events(cid)
            if dpg.does_item_exist("risk_value"):
                dpg.set_value("risk_value", f"{risk:.2f}")
            if dpg.does_item_exist("risk_bar"):
                dpg.set_value("risk_bar", risk)
            GUI_STATE["risk_cache"][cid] = risk
            update_timeline_plot(cid)
            update_prediction_for_capsule(cid)

    metrics = SYSTEM_METRICS.snapshot()
    if dpg.does_item_exist("data_flow_text"):
        text = (
            f"Capsule In: {metrics['capsule_bytes_in']} bytes\n"
            f"Capsule Out: {metrics['capsule_bytes_out']} bytes\n"
            f"Disk Read: {metrics['disk_read_bytes']} bytes\n"
            f"Disk Write: {metrics['disk_write_bytes']} bytes\n"
            f"Net Sent: {metrics['net_sent_bytes']} bytes\n"
            f"Net Recv: {metrics['net_recv_bytes']} bytes\n"
        )
        dpg.set_value("data_flow_text", text)

    if HAS_DPG:
        dpg.set_frame_callback(dpg.get_frame_count() + 30, refresh_species_state)


def capsule_row_clicked(sender, app_data, user_data):
    cid = user_data
    GUI_STATE["selected_capsule_id"] = cid
    rows = GUI_STATE["capsules"]
    meta = next((m for m in rows if m.get("capsule_id") == cid), None)
    if meta and HAS_DPG:
        if dpg.does_item_exist("capsule_details"):
            dpg.set_value("capsule_details", json.dumps(meta, indent=2))
        risk = estimate_risk_from_events(cid)
        if dpg.does_item_exist("risk_value"):
            dpg.set_value("risk_value", f"{risk:.2f}")
        if dpg.does_item_exist("risk_bar"):
            dpg.set_value("risk_bar", risk)
        GUI_STATE["risk_cache"][cid] = risk
        update_timeline_plot(cid)
        update_prediction_for_capsule(cid)


def update_timeline_plot(capsule_id: str):
    if not HAS_DPG:
        return

    events = GUI_STATE["capsule_events"].get(capsule_id, [])

    if dpg.does_item_exist("timeline_series"):
        dpg.delete_item("timeline_series")

    if not events:
        if dpg.does_item_exist("timeline_x"):
            dpg.set_axis_limits("timeline_x", 0, 10)
        if dpg.does_item_exist("timeline_y"):
            dpg.set_axis_limits("timeline_y", 0, 1)
        return

    xs = []
    ys = []

    for e in sorted(events, key=lambda x: x.get("timestamp", 0)):
        ts = e.get("timestamp", 0)
        reason = e.get("reason") or ""

        weight = 0.1
        if "read_denied" in reason or "forbidden" in reason:
            weight = 1.0
        elif "destruct" in reason:
            weight = 0.9
        elif "export" in reason:
            weight = 0.5
        elif "read" in reason:
            weight = 0.3

        xs.append(ts)
        ys.append(weight)

    dpg.add_line_series(xs, ys, parent="timeline_y", tag="timeline_series")
    dpg.set_axis_limits("timeline_x", min(xs), max(xs))
    dpg.set_axis_limits("timeline_y", 0, max(1.0, max(ys)))


def update_prediction_for_capsule(capsule_id: str):
    if not HAS_DPG:
        return
    pred_organ: PredictionOrgan = GUI_STATE.get("prediction_organ")
    if pred_organ is None:
        return

    events = GUI_STATE["capsule_events"].get(capsule_id, [])
    env = get_environment_fingerprint()
    prediction = pred_organ.predict_for_capsule(capsule_id, events, env)
    GUI_STATE["prediction_cache"][capsule_id] = prediction

    if dpg.does_item_exist("prediction_text"):
        txt = (
            f"Capsule: {capsule_id}\n"
            f"Predicted Risk (10 min): {prediction['predicted_risk']:.2f}\n"
            f"Slope: {prediction['slope']:.6f}\n"
            f"Volatility: {prediction['volatility']:.3f}\n"
            f"Next Event: {prediction['next_event_label']}"
        )
        dpg.set_value("prediction_text", txt)

    traj = prediction.get("trajectory", [])
    if dpg.does_item_exist("prediction_series"):
        dpg.delete_item("prediction_series")

    if not traj:
        if dpg.does_item_exist("prediction_x"):
            dpg.set_axis_limits("prediction_x", 0, 10)
        if dpg.does_item_exist("prediction_y"):
            dpg.set_axis_limits("prediction_y", 0, 1)
        return

    xs = [p["t"] for p in traj]
    ys = [p["r"] for p in traj]

    dpg.add_line_series(xs, ys, parent="prediction_y", tag="prediction_series")
    dpg.set_axis_limits("prediction_x", min(xs), max(xs))
    dpg.set_axis_limits("prediction_y", 0, max(1.0, max(ys)))


def safe_mode_toggled(sender, app_data, user_data):
    if not HAS_DPG:
        return
    enabled = dpg.get_value("safe_mode_checkbox")
    GUI_STATE["safe_mode"] = enabled
    coord: SpeciesCoordinator = GUI_STATE["coordinator"]
    if coord is None:
        return
    coord.safe_mode = enabled
    for cap in coord._capsules.values():
        cap._safe_mode = enabled


def force_revoke_clicked(sender, app_data, user_data):
    cid = GUI_STATE["selected_capsule_id"]
    if not cid:
        print("[GUI] No capsule selected for force revoke.")
        return
    event = {
        "capsule_id": cid,
        "species": {},
        "reason": "gui_force_revoke_request",
        "env": get_environment_fingerprint(),
        "policy": {},
        "destroyed": False,
        "timestamp": time.time(),
    }
    send_beacon(event)
    print(f"[GUI] Force revoke requested for capsule {cid}")


def update_policy_editor_text():
    if not HAS_DPG:
        return
    coord: SpeciesCoordinator = GUI_STATE["coordinator"]
    if coord is None:
        return
    rows = coord.list_capsules().values()
    lines = []
    for r in rows:
        cid = r.get("capsule_id")
        pol = r.get("policy", {})
        lines.append(f"{cid}: {json.dumps(pol)}")
    if dpg.does_item_exist("policy_editor_text"):
        dpg.set_value("policy_editor_text", "\n".join(lines))


def update_lineage_text():
    if not HAS_DPG:
        return
    lines = []
    for cid, evs in GUI_STATE["capsule_events"].items():
        lines.append(f"Capsule {cid}:")
        for e in evs[-5:]:
            lines.append(f"  - {e.get('reason')} @ {e.get('timestamp', 0):.0f}")
    if dpg.does_item_exist("lineage_text"):
        dpg.set_value("lineage_text", "\n".join(lines))


# ============================================================
#  DARK MODE THEME (OFFICE-LIKE)
# ============================================================

def apply_office_dark_theme():
    if not HAS_DPG:
        return

    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 32, 36, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (40, 40, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (40, 40, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 70, 80, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 235, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Header, (60, 90, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (80, 110, 170, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (90, 120, 180, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 90, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 110, 170, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 120, 180, 255))

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (60, 60, 70, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (70, 70, 80, 255))

            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (25, 25, 30, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 40, 50, 255))

            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (90, 120, 180, 255))

    dpg.bind_theme(theme_id)


# ============================================================
#  GUI BUILD (OFFICE-LIKE THREE-COLUMN LAYOUT + PREDICTIVE TAB)
# ============================================================

def build_gui():
    if not HAS_DPG:
        return

    apply_office_dark_theme()

    with dpg.window(label="Dark Nebula Command Deck — Access-Driven", tag="main_window", width=1800, height=950):
        dpg.add_text("Dark Nebula — Global Data Species (Access-Driven)")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=800, height=800):
                dpg.add_text("Overview")
                dpg.add_text("", tag="overview_text")
                dpg.add_checkbox(label="Safe Mode (no self-destruct)", tag="safe_mode_checkbox", callback=safe_mode_toggled)
                dpg.add_separator()

                dpg.add_text("Capsules")
                with dpg.table(
                    tag="capsule_table",
                    header_row=True,
                    resizable=True,
                    borders_innerH=True,
                    borders_innerV=True,
                    borders_outerH=True,
                    borders_outerV=True,
                    policy=dpg.mvTable_SizingStretchProp,
                    scrollY=True,
                    height=250,
                ):
                    dpg.add_table_column(label="Capsule")
                    dpg.add_table_column(label="Species")
                    dpg.add_table_column(label="Destroyed")
                    dpg.add_table_column(label="Host")
                    dpg.add_table_column(label="User")
                    dpg.add_table_column(label="Created")

                dpg.add_separator()
                dpg.add_text("Selected Capsule Details")
                dpg.add_input_text(
                    multiline=True,
                    readonly=True,
                    tag="capsule_details",
                    width=-1,
                    height=200,
                    default_value="",
                )

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=250, height=180):
                        dpg.add_text("Risk Gauge")
                        dpg.add_text("0.00", tag="risk_value")
                        dpg.add_progress_bar(tag="risk_bar", default_value=0.0, overlay="risk", width=-1)
                        dpg.add_spacer(height=10)
                        dpg.add_button(label="Force Revoke (Local)", callback=force_revoke_clicked)

                    with dpg.child_window(width=500, height=180):
                        dpg.add_text("Timeline (risk-ish over time)")
                        with dpg.plot(label="", height=150, width=-1, tag="timeline_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="time (unix seconds)", tag="timeline_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="risk-ish", tag="timeline_y"):
                                dpg.add_line_series([], [], parent="timeline_y", tag="timeline_series")

                dpg.add_separator()
                dpg.add_text("Global Event Feed")
                dpg.add_input_text(
                    multiline=True,
                    readonly=True,
                    tag="event_feed",
                    width=-1,
                    height=150,
                    default_value="",
                )

            with dpg.child_window(width=500, height=800):
                with dpg.tab_bar():
                    with dpg.tab(label="Species"):
                        dpg.add_text("species-oxygen — high value data")
                        dpg.add_input_text(
                            multiline=True,
                            readonly=True,
                            tag="species_stats_text",
                            width=-1,
                            height=150,
                            default_value="",
                        )
                        dpg.add_spacer(height=10)
                        dpg.add_text("species-silver — medium value data")
                        dpg.add_input_text(
                            multiline=True,
                            readonly=True,
                            tag="species_stats_text2",
                            width=-1,
                            height=150,
                            default_value="",
                        )

                    with dpg.tab(label="Policy Snapshot"):
                        dpg.add_text("Per-capsule policy snapshot (read-only)")
                        dpg.add_input_text(
                            multiline=True,
                            readonly=True,
                            tag="policy_editor_text",
                            width=-1,
                            height=350,
                            default_value="",
                        )

                    with dpg.tab(label="Lineage"):
                        dpg.add_text("Capsule Lineage (recent events per capsule)")
                        dpg.add_input_text(
                            multiline=True,
                            readonly=True,
                            tag="lineage_text",
                            width=-1,
                            height=350,
                            default_value="",
                        )

                    with dpg.tab(label="Predictive Intelligence"):
                        dpg.add_text("Capsule Risk Trajectory & Forecast")
                        dpg.add_input_text(
                            multiline=True,
                            readonly=True,
                            tag="prediction_text",
                            width=-1,
                            height=120,
                            default_value="Select a capsule to see predictions.",
                        )
                        dpg.add_spacer(height=5)
                        dpg.add_text("Risk Trajectory (recent)")
                        with dpg.plot(label="", height=200, width=-1, tag="prediction_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="time (unix seconds)", tag="prediction_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="risk", tag="prediction_y"):
                                dpg.add_line_series([], [], parent="prediction_y", tag="prediction_series")

            with dpg.child_window(width=450, height=800):
                dpg.add_text("Data Flow & Notes")
                dpg.add_separator()
                dpg.add_text("Data Flow (bytes)")
                dpg.add_input_text(
                    multiline=True,
                    readonly=True,
                    tag="data_flow_text",
                    width=-1,
                    height=140,
                    default_value="",
                )
                dpg.add_separator()
                dpg.add_text("Operator Notes (local only)")
                dpg.add_input_text(
                    multiline=True,
                    readonly=False,
                    tag="operator_notes",
                    width=-1,
                    height=500,
                    default_value="",
                )


# ============================================================
#  DEMO ORGANISM ACTIVITY (DEMO CAPSULE)
# ============================================================

def start_organism_demo(coordinator: SpeciesCoordinator) -> None:
    def run():
        species_id = "species-oxygen"
        policy = {
            "allowed_hosts": [socket.gethostname()],
            "allowed_geos": None,
            "allow_export": False,
        }
        raw_data = b"DEMO CAPSULE PAYLOAD -- This is a demo capsule."

        capsule = coordinator.create_capsule(
            species_id,
            raw_payload=raw_data,
            policy=policy,
            demo=True,
        )

        time.sleep(1)
        capsule._beacon("demo_capsule_created")
        time.sleep(1)
        capsule.read()
        time.sleep(1)
        capsule._beacon("demo_capsule_activity")
        time.sleep(1)
        capsule._beacon("demo_capsule_completed")

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ============================================================
#  COORDINATOR BUILDER
# ============================================================

def build_coordinator() -> SpeciesCoordinator:
    autoloader = OrganAutoloader()
    crypto = CryptoOrgan(key_label="species_master_key")
    ui_organ = UIAutomationOrgan()

    hive_url = os.getenv("DATA_SPECIES_HIVE_URL")
    hive_api_key = os.getenv("DATA_SPECIES_HIVE_API_KEY")
    sync_organ = SyncOrgan(base_url=hive_url, api_key=hive_api_key)

    behavior_history = BehaviorHistoryOrgan()
    local_rep = LocalReputationOrgan()

    policy_ai = PolicyAIOrgan(
        sync_organ=sync_organ,
        behavior_history=behavior_history,
        local_rep=local_rep,
    )

    prediction_organ = PredictionOrgan(
        behavior_history=behavior_history,
        local_rep=local_rep,
    )
    GUI_STATE["prediction_organ"] = prediction_organ

    coordinator = SpeciesCoordinator(
        autoloader=autoloader,
        crypto=crypto,
        ui_organ=ui_organ,
        policy_ai=policy_ai,
        sync_organ=sync_organ,
        safe_mode=False,
    )
    return coordinator


# ============================================================
#  MAIN
# ============================================================

def main():
    if not HAS_DPG:
        print("DearPyGUI is not available and could not be auto-installed. Check autoloader_log.txt.")
        return

    dpg.create_context()
    dpg.create_viewport(title="Dark Nebula Command Deck — Access-Driven", width=1800, height=950)

    coord = build_coordinator()
    GUI_STATE["coordinator"] = coord

    register_gui_listener(gui_event_listener)

    start_organism_demo(coord)

    monitor = SystemMonitorOrgan(interval=2.0)
    monitor.start()

    access_organ = AccessDrivenCapsuleOrgan(coordinator=coord, interval=5.0, max_capsules=500)
    access_organ.start()

    build_gui()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    dpg.set_frame_callback(dpg.get_frame_count() + 10, refresh_species_state)

    def periodic_updates(sender, app_data):
        update_policy_editor_text()
        update_lineage_text()
        cid = GUI_STATE.get("selected_capsule_id")
        if cid:
            update_prediction_for_capsule(cid)
        dpg.set_frame_callback(dpg.get_frame_count() + 60, periodic_updates)

    dpg.set_frame_callback(dpg.get_frame_count() + 30, periodic_updates)

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

