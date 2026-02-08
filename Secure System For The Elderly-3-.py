import asyncio
import threading
import time
import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Iterable, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, ttk

# ================== Global Settings =======================

ALWAYS_RADIOACTIVE = True          # High security: all programs treated as radioactive
DEFAULT_DATA_DIR = "guardian_data"

PERSIST_MIN_INTERVAL = 30.0        # seconds between disk writes (coalesced)
TELEMETRY_MAX_AGE = 300.0          # seconds to keep telemetry events for clustering

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def is_writable_dir(path: str) -> bool:
    try:
        ensure_dir(path)
        test_file = os.path.join(path, ".guardian_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False


def auto_select_storage_path() -> str:
    """
    Priority:
    1. Any writable local drive D:–Z: -> GuardianShell
    2. C:\GuardianShell
    3. Program directory / GuardianShell
    """
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:"
        if os.path.exists(root):
            candidate = os.path.join(root, "GuardianShell")
            if is_writable_dir(candidate):
                return candidate

    c_root = "C:"
    if os.path.exists(c_root):
        candidate = os.path.join(c_root, "GuardianShell")
        if is_writable_dir(candidate):
            return candidate

    prog_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(prog_dir, "GuardianShell")
    ensure_dir(candidate)
    return candidate


# ================== Autoloader ============================

import importlib
import subprocess
import sys


class AutoLoader:
    def __init__(self, log_callback):
        self.log = log_callback
        self.required_libs = [
            "psutil"
        ]

    def ensure_libraries(self):
        for lib in self.required_libs:
            if not self._is_installed(lib):
                self.log(f"Installing missing library: {lib}")
                self._install(lib)
            else:
                self.log(f"Library OK: {lib}")

    def _is_installed(self, lib):
        try:
            importlib.import_module(lib)
            return True
        except ImportError:
            return False

    def _install(self, lib):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except Exception as e:
            self.log(f"Failed to install {lib}: {e}")


# ================== Core Models & Enums ===================

class LedState(Enum):
    OFF = auto()
    GREEN = auto()
    YELLOW = auto()
    RED = auto()


class RiskLevel(Enum):
    COLD = auto()
    WARM = auto()
    HOT = auto()


@dataclass
class FileMarker:
    path: str
    first_seen: float
    last_scan: float = 0.0
    last_hash: str = ""
    last_size: int = 0
    last_modified: float = 0.0
    risk: RiskLevel = RiskLevel.COLD
    ever_scanned: bool = False


@dataclass
class NetworkConnectionInfo:
    process_name: str
    destination_ip: str
    destination_country: str
    destination_port: int
    is_encrypted: bool
    is_outbound: bool
    is_blocked: bool = False
    is_known_bad: bool = False
    is_outside_us: bool = False
    anomaly_score: float = 0.0


@dataclass
class CameraMicEvent:
    time: float
    process_name: str
    is_camera: bool
    is_microphone: bool
    driver_activated: bool
    software_requested: bool
    blocked: bool = False


@dataclass
class IdentityField:
    name: str
    value: str


@dataclass
class OutboundRequest:
    process_name: str
    url: str
    method: str = "GET"
    fields: List[IdentityField] = field(default_factory=list)
    is_game: bool = False
    is_browser: bool = False


@dataclass
class OutboundRequestResult:
    original: OutboundRequest
    rewritten: OutboundRequest
    blocked: bool = False
    reason: str = ""


# ================== Persistence Helpers ===================

def save_json(path: str, data):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


# ================== System Load Monitor ===================

class SystemLoadMonitor:
    def __init__(self):
        try:
            import psutil  # noqa
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False

    @property
    def cpu_usage(self) -> float:
        if self._psutil_available:
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        return 0.1

    @property
    def is_user_active(self) -> bool:
        # Stub: could be wired to input/idle detection
        return True


async def smart_sleep(base_seconds: float, load_monitor: SystemLoadMonitor):
    cpu = load_monitor.cpu_usage
    if cpu > 0.8:
        await asyncio.sleep(base_seconds * 2.0)
    elif cpu > 0.5:
        await asyncio.sleep(base_seconds * 1.5)
    else:
        await asyncio.sleep(base_seconds)


# ================== Shared Telemetry Bus ==================

@dataclass
class TelemetryEvent:
    time: float
    kind: str
    data: Dict


class TelemetryBus:
    """
    Simple in-process telemetry bus.
    In a future wave this can be replaced with real shared memory.
    """
    def __init__(self):
        self._events: List[TelemetryEvent] = []

    def publish(self, kind: str, data: Dict):
        now = time.time()
        self._events.append(TelemetryEvent(time=now, kind=kind, data=data))
        self._prune(now)

    def _prune(self, now: float):
        cutoff = now - TELEMETRY_MAX_AGE
        self._events = [e for e in self._events if e.time >= cutoff]

    def snapshot(self) -> List[TelemetryEvent]:
        return list(self._events)


# ================== Predictive Learning Engine ============

class LearningEngine:
    """
    Holds:
    - process profiles
    - risk scores
    - file risk
    - allow/block lists (nuclear mode)
    - momentum, sparklines, IP anomaly history
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        ensure_dir(self.base_dir)
        self.learning_file = os.path.join(self.base_dir, "learning.json")
        self.data = load_json(self.learning_file, {
            "safe_sites": [],
            "bad_sites": [],
            "process_profiles": {},
            "storage_path": self.base_dir,
            "process_risk": {},
            "file_risk": {},
            "allow_list": [],
            "block_list": [],
            "process_momentum": {},
            "process_sparkline": {},
            "ip_anomaly_history": {},
        })
        self._dirty = False
        self._last_persist = 0.0

    @property
    def storage_path(self) -> str:
        return self.data.get("storage_path", self.base_dir)

    @storage_path.setter
    def storage_path(self, path: str):
        self.data["storage_path"] = path
        self._dirty = True

    # ---------- Allow / Block list management ----------

    def add_allow(self, item_type: str, value: str, notes: str = ""):
        entry = {"type": item_type, "value": value, "notes": notes, "time": time.time()}
        if entry not in self.data["allow_list"]:
            self.data["allow_list"].append(entry)
            self._dirty = True

    def add_block(self, item_type: str, value: str, notes: str = ""):
        entry = {"type": item_type, "value": value, "notes": notes, "time": time.time()}
        if entry not in self.data["block_list"]:
            self.data["block_list"].append(entry)
            self._dirty = True

    def get_allow_list(self) -> List[Dict]:
        return list(self.data.get("allow_list", []))

    def get_block_list(self) -> List[Dict]:
        return list(self.data.get("block_list", []))

    def _match_block_entry(self, item_type: str, value: str) -> bool:
        for e in self.data.get("block_list", []):
            if e.get("type") != item_type:
                continue
            rule_val = e.get("value", "")
            if rule_val == value:
                return True
            if rule_val.endswith("*"):
                prefix = rule_val[:-1]
                if value.startswith(prefix):
                    return True
        return False

    def is_blocked(self, item_type: str, value: str) -> bool:
        return self._match_block_entry(item_type, value)

    def is_allowed(self, item_type: str, value: str) -> bool:
        for e in self.data.get("allow_list", []):
            if e.get("type") == item_type and e.get("value") == value:
                return True
        return False

    # ---------- Profiles & risk ----------

    def record_safe_site(self, host: str):
        if host not in self.data["safe_sites"]:
            self.data["safe_sites"].append(host)
            self._dirty = True

    def record_malicious_site(self, host: str):
        if host not in self.data["bad_sites"]:
            self.data["bad_sites"].append(host)
            self._dirty = True

    def record_process_network(self, process_name: str, ip: str):
        profiles = self.data.setdefault("process_profiles", {})
        prof = profiles.setdefault(process_name, {"dest_ips": {}, "last_seen": 0})
        dest_ips = prof["dest_ips"]
        dest_ips[ip] = dest_ips.get(ip, 0) + 1
        prof["last_seen"] = time.time()
        self._dirty = True

    def get_process_profile(self, process_name: str):
        return self.data.get("process_profiles", {}).get(process_name, None)

    def compute_network_anomaly(self, process_name: str, ip: str) -> float:
        prof = self.get_process_profile(process_name)
        if prof is None:
            return 0.5
        dest_ips = prof.get("dest_ips", {})
        if ip in dest_ips:
            return 0.0
        return 0.8

    def set_process_risk(self, process_name: str, risk: float):
        self.data.setdefault("process_risk", {})[process_name] = risk
        self._update_momentum(process_name, risk)
        self._dirty = True

    def set_file_risk(self, path: str, risk: float):
        self.data.setdefault("file_risk", {})[path] = risk
        self._dirty = True

    def get_process_risk_snapshot(self):
        return self.data.get("process_risk", {}).copy()

    def get_file_risk_snapshot(self):
        return self.data.get("file_risk", {}).copy()

    # ---------- Momentum & Sparklines ----------

    def _update_momentum(self, process_name: str, risk: float):
        momentum = self.data.setdefault("process_momentum", {})
        spark = self.data.setdefault("process_sparkline", {})
        history = spark.setdefault(process_name, [])
        now = time.time()
        history.append((now, risk))
        history[:] = history[-50:]
        if len(history) >= 2:
            (t1, r1), (t2, r2) = history[-2], history[-1]
            dt = max(t2 - t1, 1e-3)
            dr = r2 - r1
            momentum[process_name] = dr / dt
        else:
            momentum[process_name] = 0.0

    def get_process_momentum_snapshot(self) -> Dict[str, float]:
        return self.data.get("process_momentum", {}).copy()

    def get_process_sparkline_snapshot(self) -> Dict[str, List[Tuple[float, float]]]:
        return self.data.get("process_sparkline", {}).copy()

    # ---------- IP anomaly history ----------

    def record_ip_anomaly(self, ip: str, score: float):
        hist = self.data.setdefault("ip_anomaly_history", {})
        arr = hist.setdefault(ip, [])
        now = time.time()
        arr.append((now, score))
        arr[:] = arr[-100:]
        self._dirty = True

    def get_ip_anomaly_history_snapshot(self) -> Dict[str, List[Tuple[float, float]]]:
        return self.data.get("ip_anomaly_history", {}).copy()

    # ---------- Persistence ----------

    def persist_state(self, force: bool = False):
        now = time.time()
        if not self._dirty and not force:
            return
        if not force and (now - self._last_persist) < PERSIST_MIN_INTERVAL:
            return
        save_json(self.learning_file, self.data)
        self._dirty = False
        self._last_persist = now

    def load_state(self):
        self.data = load_json(self.learning_file, self.data)


# ================== File Marker Store =====================

class FileMarkerStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        ensure_dir(self.base_dir)
        self.markers_file = os.path.join(self.base_dir, "markers.json")
        self._markers: Dict[str, FileMarker] = {}
        self._load()

    def _load(self):
        data = load_json(self.markers_file, {})
        for path, m in data.items():
            self._markers[path] = FileMarker(
                path=path,
                first_seen=m.get("first_seen", time.time()),
                last_scan=m.get("last_scan", 0.0),
                last_hash=m.get("last_hash", ""),
                last_size=m.get("last_size", 0),
                last_modified=m.get("last_modified", 0.0),
                risk=RiskLevel[m.get("risk", "COLD")],
                ever_scanned=m.get("ever_scanned", False),
            )

    def save(self):
        data = {}
        for path, m in self._markers.items():
            d = asdict(m)
            d["risk"] = m.risk.name
            data[path] = d
        save_json(self.markers_file, data)

    def get_or_create_marker(self, path: str) -> FileMarker:
        marker = self._markers.get(path)
        if marker is None:
            now = time.time()
            marker = FileMarker(
                path=path,
                first_seen=now,
                last_modified=now,
            )
            self._markers[path] = marker
        return marker

    def update_marker(self, marker: FileMarker) -> None:
        self._markers[marker.path] = marker

    def get_hot_markers(self) -> Iterable[FileMarker]:
        return [m for m in self._markers.values()
                if m.risk in (RiskLevel.HOT, RiskLevel.WARM)]


# ================== File Scanner (Predictive) =============

class FileScanner:
    def __init__(self, store: FileMarkerStore, load_monitor: SystemLoadMonitor, gui, learning: LearningEngine, telemetry: TelemetryBus):
        self._store = store
        self._load_monitor = load_monitor
        self._gui = gui
        self._learning = learning
        self._telemetry = telemetry

    async def initial_marker_walk(self):
        await asyncio.sleep(0)

    async def scan_on_use(self, path: str):
        marker = self._store.get_or_create_marker(path)
        if not marker.ever_scanned or self._has_changed(marker):
            await self._deep_scan(path, marker)
            marker.last_scan = time.time()
            marker.ever_scanned = True
            marker.risk = RiskLevel.HOT
            self._store.update_marker(marker)
            self._store.save()
            self._learning.set_file_risk(path, 0.7)
            self._telemetry.publish("file_scan", {"path": path, "risk": 0.7})

    async def scan_hot_set(self):
        for marker in self._store.get_hot_markers():
            if self._should_throttle():
                await asyncio.sleep(0.5)
            if self._has_changed(marker) or (time.time() - marker.last_scan) >= 24 * 3600:
                await self._deep_scan(marker.path, marker)
                marker.last_scan = time.time()
                self._store.update_marker(marker)
                self._learning.set_file_risk(marker.path, 0.5)
                self._telemetry.publish("file_scan", {"path": marker.path, "risk": 0.5})
            await asyncio.sleep(0)
        self._store.save()

    def _has_changed(self, marker: FileMarker) -> bool:
        try:
            st = os.stat(marker.path)
        except FileNotFoundError:
            return False
        if st.st_size != marker.last_size or st.st_mtime != marker.last_modified:
            marker.risk = RiskLevel.HOT
            self._learning.set_file_risk(marker.path, 0.9)
            self._telemetry.publish("file_changed", {"path": marker.path, "risk": 0.9})
            return True
        return False

    def _should_throttle(self) -> bool:
        return self._load_monitor.is_user_active or self._load_monitor.cpu_usage > 0.8

    async def _deep_scan(self, path: str, marker: FileMarker):
        try:
            st = os.stat(path)
            marker.last_size = st.st_size
            marker.last_modified = st.st_mtime
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            marker.last_hash = h.hexdigest()
            self._gui.show_notification("Scan", f"Scanned {path}")
        except Exception as e:
            self._gui.show_notification("Scan Error", f"{path}: {e}")
        await asyncio.sleep(0.01)


# ================== Identity Firewall =====================

class IdentityFirewall:
    def __init__(self):
        self._never_allow_fields = {
            "ssn", "social_security", "biometric", "fingerprint", "face", "voice",
            "iris", "passport", "driver_license", "tax_id", "machine_id", "device_id"
        }
        self._trusted_sites: Dict[str, set] = {}

    def add_trusted_site(self, host: str, allowed_fields: Iterable[str]):
        host = host.lower()
        if host not in self._trusted_sites:
            self._trusted_sites[host] = set()
        for f in allowed_fields:
            self._trusted_sites[host].add(f.lower())

    def filter_outbound(self, request: OutboundRequest) -> OutboundRequestResult:
        rewritten = OutboundRequest(
            process_name=request.process_name,
            url=request.url,
            method=request.method,
            is_game=request.is_game,
            is_browser=request.is_browser,
            fields=[]
        )

        host = self._extract_host(request.url)
        allowed_fields = self._trusted_sites.get(host, set())
        reason = ""

        for field in request.fields:
            name = field.name.lower()

            if name in self._never_allow_fields:
                rewritten.fields.append(
                    IdentityField(name=field.name, value=self._generate_garbage_value(name))
                )
                reason = "Critical personal/biometric field replaced with garbage (radioactive mode)."
                continue

            if ALWAYS_RADIOACTIVE:
                rewritten.fields.append(
                    IdentityField(name=field.name, value=self._generate_safe_value(name))
                )
                reason = "All personal data sanitized (radioactive mode)."
                continue

            if name in allowed_fields:
                rewritten.fields.append(field)
            else:
                rewritten.fields.append(
                    IdentityField(name=field.name, value=self._generate_safe_value(name))
                )
                reason = "Personal field sanitized."

        return OutboundRequestResult(
            original=request,
            rewritten=rewritten,
            blocked=False,
            reason=reason
        )

    def _extract_host(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).hostname or ""
        except Exception:
            return ""

    def _generate_garbage_value(self, field_name: str) -> str:
        return "XXX-REDACTED-XXX"

    def _generate_safe_value(self, field_name: str) -> str:
        return "N/A"


# ================== Network Guardian (Predictive) =========

class NetworkGuardian:
    """
    Safe version:
    - Computes anomaly and block flags
    - Does NOT actually kill processes or block sockets
    """

    def __init__(self, gui, learning_engine: LearningEngine, load_monitor: SystemLoadMonitor, telemetry: TelemetryBus):
        self._gui = gui
        self._learning = learning_engine
        self._load_monitor = load_monitor
        self._block_outside_us = True
        self._recent: List[NetworkConnectionInfo] = []
        self._telemetry = telemetry
        try:
            import psutil  # noqa
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False

    def get_recent_connections(self) -> List[NetworkConnectionInfo]:
        return list(self._recent)

    async def monitor_loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                self._gui.show_notification("Error", f"NetworkGuardian: {e}")
                await asyncio.sleep(1.0)

    async def _tick(self):
        self._recent.clear()
        if self._psutil_available:
            import psutil
            for conn in psutil.net_connections(kind="inet"):
                if not conn.raddr:
                    continue
                ip = conn.raddr.ip
                port = conn.raddr.port
                pid = conn.pid or 0
                try:
                    p = psutil.Process(pid)
                    pname = p.name()
                    exe_path = p.exe()
                except Exception:
                    pname = "unknown"
                    exe_path = ""

                is_outside_us = not ip.startswith("192.168.") and not ip.startswith("10.") and not ip.startswith("127.")
                anomaly = self._learning.compute_network_anomaly(pname, ip)
                self._learning.record_process_network(pname, ip)
                self._learning.record_ip_anomaly(ip, anomaly)

                info = NetworkConnectionInfo(
                    process_name=pname,
                    destination_ip=ip,
                    destination_country="Unknown" if is_outside_us else "Local",
                    destination_port=port,
                    is_encrypted=False,
                    is_outbound=True,
                    is_outside_us=is_outside_us,
                    anomaly_score=anomaly
                )

                blocked = False

                if self._learning.is_blocked("process", pname):
                    blocked = True
                family_prefix = pname.split('.')[0]
                if self._learning.is_blocked("process_family", f"{family_prefix}*"):
                    blocked = True
                if self._learning.is_blocked("ip", ip):
                    blocked = True
                ip_parts = ip.split(".")
                if len(ip_parts) == 4:
                    ip_prefix = ".".join(ip_parts[:3]) + ".*"
                    if self._learning.is_blocked("ip_range", ip_prefix):
                        blocked = True

                if is_outside_us and self._block_outside_us and not self._learning.is_allowed("ip", ip):
                    blocked = True

                info.is_blocked = blocked
                self._recent.append(info)

                base_risk = anomaly
                if is_outside_us:
                    base_risk = max(base_risk, 0.7)
                if blocked:
                    base_risk = 1.0
                self._learning.set_process_risk(pname, base_risk)

                self._telemetry.publish("net_conn", {
                    "process_name": pname,
                    "ip": ip,
                    "port": port,
                    "blocked": blocked,
                    "risk": base_risk,
                    "exe": exe_path,
                })

        outside_us = [c for c in self._recent if c.is_outside_us]
        risky = [c for c in self._recent if c.anomaly_score >= 0.8 or c.is_blocked]

        if risky:
            self._gui.set_network_led(LedState.RED, "Suspicious / blocked network activity")
        elif outside_us:
            self._gui.set_network_led(LedState.YELLOW, "Outside-US connections")
        else:
            self._gui.set_network_led(LedState.GREEN, "Network normal")

        self._gui.update_outside_us_connections(risky or outside_us)
        self._learning.persist_state()
        self._gui.update_dashboard(
            self._learning.get_process_risk_snapshot(),
            self._learning.get_file_risk_snapshot(),
            self._recent,
            self._learning.get_allow_list(),
            self._learning.get_block_list(),
            self._learning.get_process_momentum_snapshot(),
            self._learning.get_process_sparkline_snapshot(),
            self._learning.get_ip_anomaly_history_snapshot()
        )
        await smart_sleep(5.0, self._load_monitor)


# ================== Camera/Mic Guardian (stub) ============

class CameraMicGuardian:
    def __init__(self, gui, learning: LearningEngine):
        self._gui = gui
        self._learning = learning
        self._events: List[CameraMicEvent] = []

    async def monitor_loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                self._gui.show_notification("Error", f"CameraMicGuardian: {e}")
                await asyncio.sleep(1.0)

    async def _tick(self):
        await asyncio.sleep(1.0)
        cam_on = any(e.driver_activated and e.is_camera for e in self._events)
        mic_on = any(e.driver_activated and e.is_microphone for e in self._events)
        suspicious = any(e.driver_activated and not e.software_requested for e in self._events)

        if any(self._learning.is_blocked("camera_app", e.process_name) for e in self._events):
            suspicious = True

        self._gui.set_camera_led(
            LedState.RED if suspicious or cam_on else LedState.GREEN,
            "Camera active" if cam_on else "Camera off"
        )
        self._gui.set_mic_led(
            LedState.RED if suspicious or mic_on else LedState.GREEN,
            "Microphone active" if mic_on else "Microphone off"
        )


# ================== IoT & Integrity (stubs) ===============

class IoTGuardian:
    def __init__(self, gui):
        self._gui = gui

    async def scan_network(self):
        await asyncio.sleep(1.0)


class SelfIntegrityMonitor:
    def __init__(self):
        self.is_healthy: bool = True
        self.last_issue: str = ""

    async def verify_self(self):
        await asyncio.sleep(0.1)
        self.is_healthy = True
        self.last_issue = ""


# ================== Process Guardian (lineage) ============

@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    name: str
    exe: str
    create_time: float


class ProcessGuardian:
    def __init__(self, gui, telemetry: TelemetryBus):
        self._gui = gui
        self._telemetry = telemetry
        self._psutil_available = False
        try:
            import psutil  # noqa
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
        self._processes: Dict[int, ProcessInfo] = {}

    def get_snapshot(self) -> Dict[int, ProcessInfo]:
        return dict(self._processes)

    async def monitor_loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                self._gui.show_notification("Error", f"ProcessGuardian: {e}")
                await asyncio.sleep(1.0)

    async def _tick(self):
        if not self._psutil_available:
            await asyncio.sleep(5.0)
            return

        import psutil
        new_procs: Dict[int, ProcessInfo] = {}
        now = time.time()

        for p in psutil.process_iter(attrs=["pid", "ppid", "name", "exe", "create_time"]):
            try:
                info = p.info
                pi = ProcessInfo(
                    pid=info["pid"],
                    ppid=info["ppid"],
                    name=info.get("name") or "",
                    exe=info.get("exe") or "",
                    create_time=info.get("create_time") or now,
                )
                new_procs[pi.pid] = pi
                if pi.pid not in self._processes:
                    self._telemetry.publish("proc_spawn", {
                        "pid": pi.pid,
                        "ppid": pi.ppid,
                        "name": pi.name,
                        "exe": pi.exe,
                        "create_time": pi.create_time,
                    })
            except Exception:
                continue

        for pid, old in self._processes.items():
            if pid not in new_procs:
                self._telemetry.publish("proc_exit", {
                    "pid": old.pid,
                    "ppid": old.ppid,
                    "name": old.name,
                    "exe": old.exe,
                })

        self._processes = new_procs
        await asyncio.sleep(5.0)


# ================== Anomaly Cluster Engine ================

class AnomalyClusterEngine:
    """
    Simple real-time anomaly clustering over telemetry.
    Not heavy ML: uses grouping by IP/process and recent risk spikes.
    """
    def __init__(self, gui, telemetry: TelemetryBus):
        self._gui = gui
        self._telemetry = telemetry

    async def monitor_loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                self._gui.show_notification("Error", f"AnomalyClusterEngine: {e}")
                await asyncio.sleep(1.0)

    async def _tick(self):
        events = self._telemetry.snapshot()
        clusters = self._simple_cluster(events)
        if clusters:
            self._gui.show_notification("Cluster", f"{len(clusters)} anomaly clusters detected (lightweight).")
        await asyncio.sleep(10.0)

    def _simple_cluster(self, events: List[TelemetryEvent]) -> List[Dict]:
        net_events = [e for e in events if e.kind == "net_conn"]
        by_ip: Dict[str, List[TelemetryEvent]] = {}
        for e in net_events:
            ip = e.data.get("ip", "")
            by_ip.setdefault(ip, []).append(e)

        clusters = []
        now = time.time()
        for ip, evs in by_ip.items():
            recent = [e for e in evs if now - e.time < 60.0]
            if len(recent) >= 3:
                avg_risk = sum(e.data.get("risk", 0.0) for e in recent) / len(recent)
                if avg_risk >= 0.6:
                    clusters.append({
                        "ip": ip,
                        "count": len(recent),
                        "avg_risk": avg_risk,
                    })
        return clusters


# ================== Tkinter GUI Adapter ===================

class TkGuiAdapter:
    def __init__(self, root: tk.Tk, learning: LearningEngine):
        self.root = root
        self.learning = learning

        root.title("GuardianShell Nuclear Mode (Safe Engine)")
        root.geometry("1350x800")

        top_frame = tk.Frame(root)
        top_frame.pack(side="top", fill="x")

        self.system_led = self._create_led(top_frame, "System")
        self.personal_led = self._create_led(top_frame, "Personal Data")
        self.network_led = self._create_led(top_frame, "Network")
        self.camera_led = self._create_led(top_frame, "Camera")
        self.mic_led = self._create_led(top_frame, "Microphone")

        self.log_box = tk.Text(root, height=6, state="disabled")
        self.log_box.pack(fill="x", padx=10, pady=5)

        self.outside_us_label = tk.Label(root, text="Outside-US connections: none")
        self.outside_us_label.pack(pady=5)

        self.save_location_label = tk.Label(root, text=f"Save location: {self.learning.storage_path}")
        self.save_location_label.pack(pady=5)

        self.save_button = tk.Button(root, text="Advanced Storage Settings", command=self._choose_save_location)
        self.save_button.pack(pady=5)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Dashboard tab
        dash_tab = tk.Frame(notebook)
        notebook.add(dash_tab, text="Dashboard")

        dash_frame = tk.Frame(dash_tab)
        dash_frame.pack(fill="both", expand=True)

        proc_frame = tk.LabelFrame(dash_frame, text="Process Risk")
        proc_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.proc_tree = ttk.Treeview(proc_frame, columns=("risk", "momentum"), show="headings", height=10)
        self.proc_tree.heading("risk", text="Process (Risk)")
        self.proc_tree.heading("momentum", text="Momentum")
        self.proc_tree.column("risk", width=200)
        self.proc_tree.column("momentum", width=100)
        self.proc_tree.pack(fill="both", expand=True)

        file_frame = tk.LabelFrame(dash_frame, text="File Risk")
        file_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.file_tree = ttk.Treeview(file_frame, columns=("risk",), show="headings", height=10)
        self.file_tree.heading("risk", text="Risk")
        self.file_tree.column("risk", width=80)
        self.file_tree.pack(fill="both", expand=True)

        net_frame = tk.LabelFrame(dash_frame, text="Network Connections (All)")
        net_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.net_tree = ttk.Treeview(net_frame, columns=("proc", "ip", "port", "risk"), show="headings", height=10)
        self.net_tree.heading("proc", text="Process")
        self.net_tree.heading("ip", text="IP")
        self.net_tree.heading("port", text="Port")
        self.net_tree.heading("risk", text="Risk")
        self.net_tree.column("proc", width=120)
        self.net_tree.column("ip", width=120)
        self.net_tree.column("port", width=60)
        self.net_tree.column("risk", width=60)
        self.net_tree.pack(fill="both", expand=True)

        # Network Threat Panel tab
        threat_tab = tk.Frame(notebook)
        notebook.add(threat_tab, text="Network Threat Panel")

        threat_notebook = ttk.Notebook(threat_tab)
        threat_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        suspicious_frame = tk.Frame(threat_notebook)
        all_frame = tk.Frame(threat_notebook)
        threat_notebook.add(suspicious_frame, text="Suspicious")
        threat_notebook.add(all_frame, text="All")

        self.suspicious_tree = ttk.Treeview(suspicious_frame, columns=("proc", "ip", "port", "risk"), show="headings", height=15)
        for col, text, width in [("proc", "Process", 150), ("ip", "IP", 150), ("port", "Port", 60), ("risk", "Risk", 60)]:
            self.suspicious_tree.heading(col, text=text)
            self.suspicious_tree.column(col, width=width)
        self.suspicious_tree.pack(fill="both", expand=True, padx=5, pady=5)

        self.all_tree = ttk.Treeview(all_frame, columns=("proc", "ip", "port", "risk"), show="headings", height=15)
        for col, text, width in [("proc", "Process", 150), ("ip", "IP", 150), ("port", "Port", 60), ("risk", "Risk", 60)]:
            self.all_tree.heading(col, text=text)
            self.all_tree.column(col, width=width)
        self.all_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Process Timeline tab
        timeline_tab = tk.Frame(notebook)
        notebook.add(timeline_tab, text="Process Timeline")

        self.timeline_tree = ttk.Treeview(timeline_tab, columns=("sparkline", "momentum"), show="headings", height=15)
        self.timeline_tree.heading("sparkline", text="Process (Sparkline)")
        self.timeline_tree.heading("momentum", text="Momentum")
        self.timeline_tree.column("sparkline", width=300)
        self.timeline_tree.column("momentum", width=100)
        self.timeline_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # IP Anomaly History tab
        ip_tab = tk.Frame(notebook)
        notebook.add(ip_tab, text="IP Anomaly History")

        self.ip_tree = ttk.Treeview(ip_tab, columns=("ip", "history"), show="headings", height=15)
        self.ip_tree.heading("ip", text="IP")
        self.ip_tree.heading("history", text="Recent Anomaly Scores")
        self.ip_tree.column("ip", width=200)
        self.ip_tree.column("history", width=500)
        self.ip_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Allow List tab
        allow_tab = tk.Frame(notebook)
        notebook.add(allow_tab, text="Allow List (Green)")

        self.allow_tree = ttk.Treeview(allow_tab, columns=("type", "value", "notes"), show="headings", height=15)
        self.allow_tree.heading("type", text="Type")
        self.allow_tree.heading("value", text="Value")
        self.allow_tree.heading("notes", text="Notes")
        self.allow_tree.column("type", width=100)
        self.allow_tree.column("value", width=300)
        self.allow_tree.column("notes", width=300)
        self.allow_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Block List tab
        block_tab = tk.Frame(notebook)
        notebook.add(block_tab, text="Block List (Red)")

        self.block_tree = ttk.Treeview(block_tab, columns=("type", "value", "notes"), show="headings", height=15)
        self.block_tree.heading("type", text="Type")
        self.block_tree.heading("value", text="Value")
        self.block_tree.heading("notes", text="Notes")
        self.block_tree.column("type", width=100)
        self.block_tree.column("value", width=300)
        self.block_tree.column("notes", width=300)
        self.block_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Context menu and bindings
        self._build_context_menu()
        self.proc_tree.bind("<Button-3>", self._on_proc_right_click)
        self.net_tree.bind("<Button-3>", self._on_net_right_click)
        self.file_tree.bind("<Button-3>", self._on_file_right_click)
        self.suspicious_tree.bind("<Button-3>", self._on_net_right_click)
        self.all_tree.bind("<Button-3>", self._on_net_right_click)

        self.net_tree.bind("<Double-1>", self._on_net_double_click)
        self.suspicious_tree.bind("<Double-1>", self._on_net_double_click)
        self.all_tree.bind("<Double-1>", self._on_net_double_click)

        self.context_target: Optional[tuple] = None

    # ---------- Context menu ----------

    def _build_context_menu(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Allow", command=self._allow_selected)
        self.context_menu.add_command(label="Block Forever", command=self._block_selected)
        self.context_menu.add_command(label="Block Process + All Children", command=self._block_with_children)
        self.context_menu.add_command(label="Block Entire Process Family", command=self._block_family)
        self.context_menu.add_command(label="Block + Registry Persistence", command=self._block_persistence)
        self.context_menu.add_command(label="Block + Domain + IP Range", command=self._block_net_scope)
        self.context_menu.add_command(label="Block + File Path + Hash", command=self._block_file_scope)
        self.context_menu.add_command(label="Block + Auto-Quarantine Executable", command=self._block_quarantine)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Radioactive (default)", state="disabled")

    def _on_proc_right_click(self, event):
        row = self.proc_tree.identify_row(event.y)
        if row:
            self.proc_tree.selection_set(row)
            raw = self.proc_tree.item(row)["values"][0]
            pname = raw.split(" ")[0]
            self.context_target = ("process", pname)
            self.context_menu.post(event.x_root, event.y_root)

    def _on_net_right_click(self, event):
        tree = event.widget
        row = tree.identify_row(event.y)
        if row:
            tree.selection_set(row)
            vals = tree.item(row)["values"]
            proc, ip = vals[0], vals[1]
            self.context_target = ("ip", ip)
            self.context_menu.post(event.x_root, event.y_root)

    def _on_file_right_click(self, event):
        row = self.file_tree.identify_row(event.y)
        if row:
            self.file_tree.selection_set(row)
            path = self.file_tree.item(row)["text"]
            self.context_target = ("file", path)
            self.context_menu.post(event.x_root, event.y_root)

    def _allow_selected(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        self.learning.add_allow(item_type, value, notes="Operator allow")
        self.learning.persist_state(force=True)
        self.show_notification("Allow", f"Allowed {item_type}: {value}")
        self._highlight_block_state(item_type, value, allowed=True)

    def _block_selected(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        self.learning.add_block(item_type, value, notes="Operator block")
        self.learning.persist_state(force=True)
        self.show_notification("Block", f"Blocked {item_type}: {value}")
        self._highlight_block_state(item_type, value, allowed=False)

    def _block_with_children(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type != "process":
            self.show_notification("Error", "This option only applies to processes.")
            return
        self.learning.add_block("process", value, notes="Operator nuclear block")
        wildcard = f"{value}*"
        self.learning.add_block("process", wildcard, notes="Block all children")
        self.learning.persist_state(force=True)
        self.show_notification("Nuclear Block", f"Blocked {value} and all child processes (rules only).")
        self._highlight_block_state("process", value, allowed=False)

    def _block_family(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type != "process":
            self.show_notification("Error", "Family block only applies to processes.")
            return
        family_prefix = value.split('.')[0]
        wildcard = f"{family_prefix}*"
        self.learning.add_block("process_family", wildcard, notes="Block entire family")
        self.learning.persist_state(force=True)
        self.show_notification("Family Block", f"Blocked family: {wildcard} (rules only).")
        self._highlight_block_state("process_family", wildcard, allowed=False)

    def _block_persistence(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type != "process":
            self.show_notification("Error", "Persistence block only applies to processes.")
            return
        pname = value
        self.learning.add_block("process", pname, notes="Block process")
        self.learning.add_block("startup", pname, notes="Block startup persistence")
        self.learning.add_block("scheduled_task", pname, notes="Block scheduled tasks")
        self.learning.add_block("registry_key", f"HKCU\\Software\\{pname}", notes="Block registry persistence")
        self.learning.add_block("registry_key", f"HKLM\\Software\\{pname}", notes="Block registry persistence")
        self.learning.persist_state(force=True)
        self.show_notification("Persistence Block", f"Blocked process and persistence (rules only): {pname}")
        self._highlight_block_state("process", pname, allowed=False)

    def _block_net_scope(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type == "ip":
            ip = value
            ip_parts = ip.split(".")
            if len(ip_parts) == 4:
                ip_prefix = ".".join(ip_parts[:3]) + ".*"
                self.learning.add_block("ip", ip, notes="Block IP")
                self.learning.add_block("ip_range", ip_prefix, notes="Block IP range")
                self.learning.persist_state(force=True)
                self.show_notification("Network Scope Block", f"Blocked IP and range (rules only): {ip_prefix}")
                self._highlight_block_state("ip", ip, allowed=False)
            else:
                self.show_notification("Error", "Invalid IP format.")
        elif item_type == "process":
            pname = value
            self.learning.add_block("process", pname, notes="Block process (net scope)")
            self.learning.persist_state(force=True)
            self.show_notification("Network Scope Block", f"Blocked process (rules only): {pname}")
            self._highlight_block_state("process", pname, allowed=False)
        else:
            self.show_notification("Error", "Network scope block applies to process or IP.")

    def _block_file_scope(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type != "process":
            self.show_notification("Error", "File scope block only applies to processes.")
            return
        pname = value
        exe_path = f"<path-of-{pname}>"
        sha256_hash = "<sha256-of-exe>"
        self.learning.add_block("process", pname, notes="Block process")
        self.learning.add_block("file_path", exe_path, notes="Block file path")
        self.learning.add_block("file_hash", sha256_hash, notes="Block file hash")
        self.learning.persist_state(force=True)
        self.show_notification("File Scope Block", f"Blocked process, path, and hash (rules only): {pname}")
        self._highlight_block_state("process", pname, allowed=False)

    def _block_quarantine(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        if item_type != "process":
            self.show_notification("Error", "Quarantine block only applies to processes.")
            return
        pname = value
        exe_path = f"<path-of-{pname}>"
        sha256_hash = "<sha256-of-exe>"
        self.learning.add_block("process", pname, notes="Block process")
        self.learning.add_block("file_hash", sha256_hash, notes="Quarantine hash")
        self.learning.add_block("file_path", exe_path, notes="Quarantine path")
        self.learning.add_block("quarantine", exe_path, notes="Quarantine executable")
        self.learning.persist_state(force=True)
        self.show_notification("Quarantine", f"Quarantine rule added (rules only) for: {pname}")
        self._highlight_block_state("process", pname, allowed=False)

    def _highlight_block_state(self, item_type: str, value: str, allowed: bool):
        color = "green" if allowed else "red"

        def apply_color(tree: ttk.Treeview):
            for row in tree.get_children():
                vals = tree.item(row)["values"]
                if not vals:
                    continue
                if item_type in ("process", "process_family"):
                    pname = vals[0]
                    if pname.startswith(value.split("*")[0]):
                        tree.item(row, tags=("highlight",))
                elif item_type == "ip":
                    ip = vals[1]
                    if ip == value:
                        tree.item(row, tags=("highlight",))

            tree.tag_configure("highlight", background=color, foreground="white")

        self.root.after(0, lambda: [apply_color(self.net_tree),
                                    apply_color(self.suspicious_tree),
                                    apply_color(self.all_tree)])

    # ---------- LEDs & UI ----------

    def _create_led(self, root, label_text: str):
        frame = tk.Frame(root)
        frame.pack(side="left", padx=5, pady=5)

        label = tk.Label(frame, text=label_text, font=("Arial", 12, "bold"))
        label.pack(side="top")

        canvas = tk.Canvas(frame, width=40, height=40)
        canvas.pack(side="top", pady=2)
        circle = canvas.create_oval(5, 5, 35, 35, fill="grey")

        text_label = tk.Label(frame, text="", font=("Arial", 10))
        text_label.pack(side="top")

        return {"canvas": canvas, "circle": circle, "text": text_label}

    def _set_led(self, led, state: LedState, message: str):
        color = {
            LedState.OFF: "grey",
            LedState.GREEN: "green",
            LedState.YELLOW: "yellow",
            LedState.RED: "red",
        }.get(state, "grey")

        def update():
            led["canvas"].itemconfig(led["circle"], fill=color)
            led["text"].config(text=message)

        self.root.after(0, update)

    def set_system_led(self, state: LedState, message: str):
        self._set_led(self.system_led, state, message)

    def set_personal_data_led(self, state: LedState, message: str):
        self._set_led(self.personal_led, state, message)

    def set_network_led(self, state: LedState, message: str):
        self._set_led(self.network_led, state, message)

    def set_camera_led(self, state: LedState, message: str):
        self._set_led(self.camera_led, state, message)

    def set_mic_led(self, state: LedState, message: str):
        self._set_led(self.mic_led, state, message)

    def show_notification(self, title: str, message: str):
        def append():
            self.log_box.config(state="normal")
            self.log_box.insert("end", f"[{title}] {message}\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")

        self.root.after(0, append)

    def update_outside_us_connections(self, connections: Iterable[NetworkConnectionInfo]):
        text = "Outside-US connections: "
        items = [f"{c.process_name}->{c.destination_ip}" for c in connections]
        if not items:
            text += "none"
        else:
            text += ", ".join(items)

        def update():
            self.outside_us_label.config(text=text)

        self.root.after(0, update)

    def _choose_save_location(self):
        path = filedialog.askdirectory(title="Select Save Location (Local or SMB)")
        if path:
            self.learning.storage_path = path
            self.save_location_label.config(text=f"Save location: {path}")
            self.show_notification("Save Location", f"State will be saved to: {path}")
            self.learning.persist_state(force=True)

    def _risk_color(self, risk: float, blocked: bool = False, outside_us: bool = False) -> str:
        if blocked:
            return "red"
        if risk >= 0.8:
            return "red"
        if risk >= 0.5:
            return "yellow"
        if outside_us:
            return "purple"
        if risk <= 0.1:
            return "green"
        return "white"

    def _on_net_double_click(self, event):
        tree = event.widget
        row = tree.identify_row(event.y)
        if not row:
            return
        vals = tree.item(row)["values"]
        if len(vals) < 4:
            return
        proc, ip, port, risk = vals
        popup = tk.Toplevel(self.root)
        popup.title("Connection Details")
        popup.geometry("400x200")
        details = f"Process: {proc}\nIP: {ip}\nPort: {port}\nRisk: {risk}"
        label = tk.Label(popup, text=details, justify="left", font=("Consolas", 10))
        label.pack(fill="both", expand=True, padx=10, pady=10)

    def update_dashboard(self,
                         process_risk: Dict[str, float],
                         file_risk: Dict[str, float],
                         connections: List[NetworkConnectionInfo],
                         allow_list: List[Dict],
                         block_list: List[Dict],
                         process_momentum: Dict[str, float],
                         process_sparkline: Dict[str, List[Tuple[float, float]]],
                         ip_anomaly_history: Dict[str, List[Tuple[float, float]]]):
        def refresh():
            # Process risk
            for i in self.proc_tree.get_children():
                self.proc_tree.delete(i)
            for pname, risk in sorted(process_risk.items(), key=lambda x: -x[1])[:80]:
                mom = process_momentum.get(pname, 0.0)
                row = self.proc_tree.insert("", "end", values=(f"{pname} ({risk:.2f})", f"{mom:+.3f}"))
                color = self._risk_color(risk)
                self.proc_tree.item(row, tags=(color,))
            for color in ["green", "white", "yellow", "red", "purple"]:
                fg = "black" if color in ("white", "yellow") else "white"
                self.proc_tree.tag_configure(color, background=color, foreground=fg)

            # File risk
            for i in self.file_tree.get_children():
                self.file_tree.delete(i)
            for path, risk in sorted(file_risk.items(), key=lambda x: -x[1])[:80]:
                row = self.file_tree.insert("", "end", values=(f"{risk:.2f}",), text=path)
                color = self._risk_color(risk)
                self.file_tree.item(row, tags=(color,))
            for color in ["green", "white", "yellow", "red", "purple"]:
                fg = "black" if color in ("white", "yellow") else "white"
                self.file_tree.tag_configure(color, background=color, foreground=fg)

            # Network (Dashboard All)
            for i in self.net_tree.get_children():
                self.net_tree.delete(i)
            for c in connections[:80]:
                risk = max(c.anomaly_score, 1.0 if c.is_blocked else 0.0)
                row = self.net_tree.insert("", "end",
                                           values=(c.process_name, c.destination_ip, c.destination_port, f"{risk:.2f}"))
                color = self._risk_color(risk, blocked=c.is_blocked, outside_us=c.is_outside_us)
                self.net_tree.item(row, tags=(color,))
            for color in ["green", "white", "yellow", "red", "purple"]:
                fg = "black" if color in ("white", "yellow") else "white"
                self.net_tree.tag_configure(color, background=color, foreground=fg)

            # Network Threat Panel: Suspicious vs All
            for tree in (self.suspicious_tree, self.all_tree):
                for i in tree.get_children():
                    tree.delete(i)

            for c in connections[:200]:
                risk = max(c.anomaly_score, 1.0 if c.is_blocked else 0.0)
                vals = (c.process_name, c.destination_ip, c.destination_port, f"{risk:.2f}")
                color = self._risk_color(risk, blocked=c.is_blocked, outside_us=c.is_outside_us)
                row_all = self.all_tree.insert("", "end", values=vals)
                self.all_tree.item(row_all, tags=(color,))
                if c.is_blocked or risk >= 0.8:
                    row_susp = self.suspicious_tree.insert("", "end", values=vals)
                    self.suspicious_tree.item(row_susp, tags=(color,))

            for tree in (self.suspicious_tree, self.all_tree):
                for color in ["green", "white", "yellow", "red", "purple"]:
                    fg = "black" if color in ("white", "yellow") else "white"
                    tree.tag_configure(color, background=color, foreground=fg)

            # Process Timeline (sparkline + momentum)
            for i in self.timeline_tree.get_children():
                self.timeline_tree.delete(i)
            for pname, history in process_sparkline.items():
                if not history:
                    continue
                risks = [f"{r:.2f}" for (_, r) in history[-10:]]
                spark = f"{pname}: " + " ".join(risks)
                mom = process_momentum.get(pname, 0.0)
                row = self.timeline_tree.insert("", "end", values=(spark, f"{mom:+.3f}"))
                last_risk = history[-1][1]
                color = self._risk_color(last_risk)
                self.timeline_tree.item(row, tags=(color,))
            for color in ["green", "white", "yellow", "red", "purple"]:
                fg = "black" if color in ("white", "yellow") else "white"
                self.timeline_tree.tag_configure(color, background=color, foreground=fg)

            # IP anomaly history
            for i in self.ip_tree.get_children():
                self.ip_tree.delete(i)
            for ip, hist in ip_anomaly_history.items():
                scores = [f"{s:.2f}" for (_, s) in hist[-10:]]
                row = self.ip_tree.insert("", "end", values=(ip, " ".join(scores)))
                last = hist[-1][1] if hist else 0.0
                color = self._risk_color(last)
                self.ip_tree.item(row, tags=(color,))
            for color in ["green", "white", "yellow", "red", "purple"]:
                fg = "black" if color in ("white", "yellow") else "white"
                self.ip_tree.tag_configure(color, background=color, foreground=fg)

            # Allow list
            for i in self.allow_tree.get_children():
                self.allow_tree.delete(i)
            for e in allow_list:
                self.allow_tree.insert("", "end",
                                       values=(e.get("type", ""), e.get("value", ""), e.get("notes", "")))

            # Block list
            for i in self.block_tree.get_children():
                self.block_tree.delete(i)
            for e in block_list:
                self.block_tree.insert("", "end",
                                       values=(e.get("type", ""), e.get("value", ""), e.get("notes", "")))

        self.root.after(0, refresh)


# ================== Guardian Engine =======================

class GuardianEngine:
    def __init__(
        self,
        file_scanner: FileScanner,
        network_guardian: NetworkGuardian,
        identity_firewall: IdentityFirewall,
        camera_mic_guardian: CameraMicGuardian,
        iot_guardian: IoTGuardian,
        self_integrity: SelfIntegrityMonitor,
        learning: LearningEngine,
        gui: TkGuiAdapter,
        marker_store: FileMarkerStore,
        load_monitor: SystemLoadMonitor,
        process_guardian: ProcessGuardian,
        anomaly_cluster: AnomalyClusterEngine,
    ):
        self._file_scanner = file_scanner
        self._network_guardian = network_guardian
        self._identity_firewall = identity_firewall
        self._camera_mic_guardian = camera_mic_guardian
        self._iot_guardian = iot_guardian
        self._self_integrity = self_integrity
        self._learning = learning
        self._gui = gui
        self._marker_store = marker_store
        self._load_monitor = load_monitor
        self._process_guardian = process_guardian
        self._anomaly_cluster = anomaly_cluster
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        self._learning.load_state()
        await self._file_scanner.initial_marker_walk()

        self._tasks.append(asyncio.create_task(self._network_guardian.monitor_loop()))
        self._tasks.append(asyncio.create_task(self._camera_mic_guardian.monitor_loop()))
        self._tasks.append(asyncio.create_task(self._daily_hot_scan_loop()))
        self._tasks.append(asyncio.create_task(self._daily_integrity_check_loop()))
        self._tasks.append(asyncio.create_task(self._daily_iot_scan_loop()))
        self._tasks.append(asyncio.create_task(self._process_guardian.monitor_loop()))
        self._tasks.append(asyncio.create_task(self._anomaly_cluster.monitor_loop()))

        self._gui.set_system_led(LedState.GREEN, "Guardian running (Nuclear Mode, safe engine).")
        self._gui.set_personal_data_led(
            LedState.YELLOW,
            "Radioactive: all programs sanitized"
        )
        self._gui.set_network_led(LedState.GREEN, "Network monitored.")

    async def _daily_hot_scan_loop(self):
        while True:
            try:
                await self._file_scanner.scan_hot_set()
            except Exception as e:
                self._gui.show_notification("Error", f"HotScan: {e}")
            await smart_sleep(24 * 3600, self._load_monitor)

    async def _daily_integrity_check_loop(self):
        while True:
            try:
                await self._self_integrity.verify_self()
                if not self._self_integrity.is_healthy:
                    self._gui.set_system_led(LedState.RED, "Guardian integrity issue.")
                    self._gui.show_notification("Guardian damaged", self._self_integrity.last_issue)
            except Exception as e:
                self._gui.show_notification("Error", f"IntegrityCheck: {e}")
            await smart_sleep(24 * 3600, self._load_monitor)

    async def _daily_iot_scan_loop(self):
        while True:
            try:
                await self._iot_guardian.scan_network()
            except Exception as e:
                self._gui.show_notification("Error", f"IoTScan: {e}")
            await smart_sleep(24 * 3600, self._load_monitor)

    def on_outbound_request(self, request: OutboundRequest) -> OutboundRequestResult:
        pname = request.process_name
        family_prefix = pname.split('.')[0]

        if self._learning.is_blocked("process", pname) or \
           self._learning.is_blocked("process_family", f"{family_prefix}*"):
            rewritten = OutboundRequest(
                process_name=request.process_name,
                url=request.url,
                method=request.method,
                is_game=request.is_game,
                is_browser=request.is_browser,
                fields=[]
            )
            result = OutboundRequestResult(
                original=request,
                rewritten=rewritten,
                blocked=True,
                reason="Process is blocked by rules (safe engine)."
            )
            self._gui.show_notification("Blocked", result.reason)
            self._gui.set_personal_data_led(LedState.RED, "Blocked dangerous process (rules only).")
            return result

        result = self._identity_firewall.filter_outbound(request)
        if result.reason:
            self._gui.set_personal_data_led(
                LedState.YELLOW,
                "Radioactive: all data sanitized"
            )
            self._gui.show_notification("Sanitized", result.reason)
        return result


# ================== Async + Tkinter Bridge ================

def start_asyncio_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def async_main(gui: TkGuiAdapter, learning: LearningEngine):
    autoloader = AutoLoader(lambda msg: gui.show_notification("Autoloader", msg))
    autoloader.ensure_libraries()

    load_monitor = SystemLoadMonitor()

    if not learning.storage_path or not os.path.exists(learning.storage_path):
        auto_path = auto_select_storage_path()
        learning.storage_path = auto_path
        learning.persist_state(force=True)
        gui.show_notification("Storage", f"Auto-selected storage: {auto_path}")

    base_dir = learning.storage_path
    ensure_dir(base_dir)

    telemetry = TelemetryBus()
    marker_store = FileMarkerStore(base_dir)
    file_scanner = FileScanner(marker_store, load_monitor, gui, learning, telemetry)
    identity_firewall = IdentityFirewall()
    self_integrity = SelfIntegrityMonitor()
    iot_guardian = IoTGuardian(gui)
    network_guardian = NetworkGuardian(gui, learning, load_monitor, telemetry)
    camera_mic_guardian = CameraMicGuardian(gui, learning)
    process_guardian = ProcessGuardian(gui, telemetry)
    anomaly_cluster = AnomalyClusterEngine(gui, telemetry)

    engine = GuardianEngine(
        file_scanner=file_scanner,
        network_guardian=network_guardian,
        identity_firewall=identity_firewall,
        camera_mic_guardian=camera_mic_guardian,
        iot_guardian=iot_guardian,
        self_integrity=self_integrity,
        learning=learning,
        gui=gui,
        marker_store=marker_store,
        load_monitor=load_monitor,
        process_guardian=process_guardian,
        anomaly_cluster=anomaly_cluster,
    )

    await engine.start()

    await asyncio.sleep(2)
    req = OutboundRequest(
        process_name="chrome.exe",
        url="https://example.com/login",
        is_browser=True,
        fields=[
            IdentityField(name="name", value="John Doe"),
            IdentityField(name="ssn", value="123-45-6789"),
            IdentityField(name="email", value="john@example.com"),
        ],
    )
    engine.on_outbound_request(req)


def main():
    auto_base = auto_select_storage_path()
    learning = LearningEngine(auto_base)

    root = tk.Tk()
    gui = TkGuiAdapter(root, learning)

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_asyncio_loop, args=(loop,), daemon=True)
    t.start()

    asyncio.run_coroutine_threadsafe(async_main(gui, learning), loop)

    root.mainloop()
    loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()

