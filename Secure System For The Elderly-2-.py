import asyncio
import threading
import time
import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Iterable, Optional
from collections import deque

import tkinter as tk
from tkinter import filedialog, ttk

# ================== Global Settings =======================

ALWAYS_RADIOACTIVE = True          # High security: all programs treated as radioactive
DEFAULT_DATA_DIR = "guardian_data"


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
    threat_momentum: float = 0.0


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


# ================== Predictive Learning Engine ============

class LearningEngine:
    """
    Holds:
    - process profiles
    - risk scores
    - file risk
    - allow/block lists (nuclear mode)
    - per-process sparkline
    - per-IP anomaly history
    - threat momentum
    """

    DECAY_HALF_LIFE = 3600.0  # 1 hour

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
            "process_risk_time": {},
            "file_risk_time": {},
            "process_momentum": {},
            "ip_history": {},          # ip -> {"anomalies": [(t,score),...], "momentum": float}
        })
        self.spark: Dict[str, deque] = {}  # process_name -> deque(maxlen=20)

    @property
    def storage_path(self) -> str:
        return self.data.get("storage_path", self.base_dir)

    @storage_path.setter
    def storage_path(self, path: str):
        self.data["storage_path"] = path

    # ---------- Allow / Block list management ----------

    def add_allow(self, item_type: str, value: str, notes: str = ""):
        entry = {"type": item_type, "value": value, "notes": notes, "time": time.time()}
        if entry not in self.data["allow_list"]:
            self.data["allow_list"].append(entry)

    def add_block(self, item_type: str, value: str, notes: str = ""):
        entry = {"type": item_type, "value": value, "notes": notes, "time": time.time()}
        if entry not in self.data["block_list"]:
            self.data["block_list"].append(entry)

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

    def record_malicious_site(self, host: str):
        if host not in self.data["bad_sites"]:
            self.data["bad_sites"].append(host)

    def record_process_network(self, process_name: str, ip: str):
        profiles = self.data.setdefault("process_profiles", {})
        prof = profiles.setdefault(process_name, {"dest_ips": {}, "last_seen": 0})
        dest_ips = prof["dest_ips"]
        dest_ips[ip] = dest_ips.get(ip, 0) + 1
        prof["last_seen"] = time.time()

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

    def _decay_factor(self, dt: float) -> float:
        import math
        if dt <= 0:
            return 1.0
        return math.exp(-dt * math.log(2) / self.DECAY_HALF_LIFE)

    def bump_process_event(self, process_name: str, weight: float = 0.1):
        now = time.time()
        pr = self.data.setdefault("process_risk", {})
        last = pr.get(process_name, 0.0)
        prt = self.data.setdefault("process_risk_time", {})
        last_time = prt.get(process_name, now)
        decay = self._decay_factor(now - last_time)
        new_val = last * decay + weight
        new_val = max(0.0, min(1.0, new_val))
        pr[process_name] = new_val
        prt[process_name] = now

        # threat momentum (rate of change)
        pm = self.data.setdefault("process_momentum", {})
        pm[process_name] = new_val - last

        self.record_spark(process_name, new_val)

    def bump_file_event(self, path: str, weight: float = 0.1):
        now = time.time()
        fr = self.data.setdefault("file_risk", {})
        last = fr.get(path, 0.0)
        frt = self.data.setdefault("file_risk_time", {})
        last_time = frt.get(path, now)
        decay = self._decay_factor(now - last_time)
        new_val = last * decay + weight
        new_val = max(0.0, min(1.0, new_val))
        fr[path] = new_val
        frt[path] = now

    def set_process_risk(self, process_name: str, risk: float):
        self.data.setdefault("process_risk", {})[process_name] = risk

    def set_file_risk(self, path: str, risk: float):
        self.data.setdefault("file_risk", {})[path] = risk

    def get_process_risk_snapshot(self):
        return self.data.get("process_risk", {}).copy()

    def get_file_risk_snapshot(self):
        return self.data.get("file_risk", {}).copy()

    def get_process_momentum_snapshot(self):
        return self.data.get("process_momentum", {}).copy()

    def record_spark(self, process_name: str, risk: float):
        dq = self.spark.setdefault(process_name, deque(maxlen=20))
        dq.append(risk)

    def get_sparkline(self, process_name: str) -> str:
        if process_name not in self.spark:
            return ""
        vals = list(self.spark[process_name])
        if not vals:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        out = ""
        for v in vals:
            idx = min(7, max(0, int(v * 7)))
            out += blocks[idx]
        return out

    def record_ip_anomaly(self, ip: str, score: float):
        now = time.time()
        hist = self.data.setdefault("ip_history", {})
        entry = hist.setdefault(ip, {"anomalies": [], "momentum": 0.0})
        anomalies = entry["anomalies"]
        anomalies.append((now, score))
        if len(anomalies) > 50:
            anomalies.pop(0)
        last_score = anomalies[-2][1] if len(anomalies) > 1 else 0.0
        entry["momentum"] = score - last_score

    def get_ip_history_snapshot(self):
        return self.data.get("ip_history", {}).copy()

    def compute_global_risk(self) -> float:
        pr = self.get_process_risk_snapshot()
        fr = self.get_file_risk_snapshot()
        vals = list(pr.values()) + list(fr.values())
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def persist_state(self):
        save_json(self.learning_file, self.data)

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
    def __init__(self, store: FileMarkerStore, load_monitor: SystemLoadMonitor, gui, learning: LearningEngine):
        self._store = store
        self._load_monitor = load_monitor
        self._gui = gui
        self._learning = learning

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
            self._learning.bump_file_event(path, weight=0.3)

    async def scan_hot_set(self):
        for marker in self._store.get_hot_markers():
            if self._should_throttle():
                await asyncio.sleep(0.5)
            if self._has_changed(marker) or (time.time() - marker.last_scan) >= 24 * 3600:
                await self._deep_scan(marker.path, marker)
                marker.last_scan = time.time()
                self._store.update_marker(marker)
                self._learning.bump_file_event(marker.path, weight=0.2)
            await asyncio.sleep(0)
        self._store.save()

    def _has_changed(self, marker: FileMarker) -> bool:
        try:
            st = os.stat(marker.path)
        except FileNotFoundError:
            return False
        if st.st_size != marker.last_size or st.st_mtime != marker.last_modified:
            marker.risk = RiskLevel.HOT
            self._learning.bump_file_event(marker.path, weight=0.4)
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

    def __init__(self, gui, learning_engine: LearningEngine, load_monitor: SystemLoadMonitor):
        self._gui = gui
        self._learning = learning_engine
        self._load_monitor = load_monitor
        self._block_outside_us = True
        self._recent: List[NetworkConnectionInfo] = []
        try:
            import psutil  # noqa
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False

    def get_recent_connections(self) -> List[NetworkConnectionInfo]:
        return list(self._recent)

    async def monitor_loop(self):
        while True:
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

                    base_risk = anomaly
                    if is_outside_us:
                        base_risk = max(base_risk, 0.7)
                    if blocked:
                        base_risk = 1.0

                    self._learning.bump_process_event(pname, weight=base_risk * 0.3)

                    # threat momentum from learning
                    momentum_snapshot = self._learning.get_process_momentum_snapshot()
                    info.threat_momentum = momentum_snapshot.get(pname, 0.0)

                    self._recent.append(info)

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
            self._gui.update_dashboard(self._learning.get_process_risk_snapshot(),
                                       self._learning.get_file_risk_snapshot(),
                                       self._recent,
                                       self._learning.get_allow_list(),
                                       self._learning.get_block_list(),
                                       self._learning.get_process_momentum_snapshot(),
                                       self._learning.get_ip_history_snapshot())
            await smart_sleep(5.0, self._load_monitor)


# ================== Camera/Mic Guardian (stub) ============

class CameraMicGuardian:
    def __init__(self, gui, learning: LearningEngine):
        self._gui = gui
        self._learning = learning
        self._events: List[CameraMicEvent] = []

    async def monitor_loop(self):
        while True:
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


# ================== Tkinter GUI Adapter ===================

class TkGuiAdapter:
    def __init__(self, root: tk.Tk, learning: LearningEngine):
        self.root = root
        self.learning = learning
        self._last_connections: List[NetworkConnectionInfo] = []

        root.title("GuardianShell Nuclear Mode (Safe Engine)")
        root.geometry("1300x800")

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

        self.proc_tree = ttk.Treeview(proc_frame, columns=("risk",), show="headings", height=10)
        self.proc_tree.heading("risk", text="Process (Risk)")
        self.proc_tree.column("risk", width=260)
        self.proc_tree.pack(fill="both", expand=True)

        file_frame = tk.LabelFrame(dash_frame, text="File Risk")
        file_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.file_tree = ttk.Treeview(file_frame, columns=("risk",), show="headings", height=10)
        self.file_tree.heading("risk", text="Risk")
        self.file_tree.column("risk", width=80)
        self.file_tree.pack(fill="both", expand=True)

        net_frame = tk.LabelFrame(dash_frame, text="Network Connections")
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

        # Network Threat Panel
        net_panel = ttk.Notebook(notebook)
        notebook.add(net_panel, text="Network Threat Panel")

        suspicious_tab = tk.Frame(net_panel)
        net_panel.add(suspicious_tab, text="Suspicious")

        self.suspicious_tree = ttk.Treeview(
            suspicious_tab,
            columns=("proc", "ip", "port", "risk"),
            show="headings",
            height=15
        )
        for col in ("proc", "ip", "port", "risk"):
            self.suspicious_tree.heading(col, text=col.capitalize())
        self.suspicious_tree.pack(fill="both", expand=True)

        all_tab = tk.Frame(net_panel)
        net_panel.add(all_tab, text="All Connections")

        self.all_tree = ttk.Treeview(
            all_tab,
            columns=("proc", "ip", "port", "risk"),
            show="headings",
            height=15
        )
        for col in ("proc", "ip", "port", "risk"):
            self.all_tree.heading(col, text=col.capitalize())
        self.all_tree.pack(fill="both", expand=True)

        # Process Timeline Panel
        timeline_tab = tk.Frame(notebook)
        notebook.add(timeline_tab, text="Process Timeline")

        self.timeline_tree = ttk.Treeview(
            timeline_tab,
            columns=("proc", "spark", "momentum"),
            show="headings",
            height=15
        )
        self.timeline_tree.heading("proc", text="Process")
        self.timeline_tree.heading("spark", text="Risk Sparkline")
        self.timeline_tree.heading("momentum", text="Momentum")
        self.timeline_tree.column("proc", width=160)
        self.timeline_tree.column("spark", width=260)
        self.timeline_tree.column("momentum", width=80)
        self.timeline_tree.pack(fill="both", expand=True)

        # IP Anomaly History Panel
        ip_tab = tk.Frame(notebook)
        notebook.add(ip_tab, text="IP Anomaly History")

        self.ip_tree = ttk.Treeview(
            ip_tab,
            columns=("ip", "count", "last", "momentum"),
            show="headings",
            height=15
        )
        self.ip_tree.heading("ip", text="IP")
        self.ip_tree.heading("count", text="Events")
        self.ip_tree.heading("last", text="Last Anomaly")
        self.ip_tree.heading("momentum", text="Momentum")
        self.ip_tree.column("ip", width=160)
        self.ip_tree.column("count", width=80)
        self.ip_tree.column("last", width=100)
        self.ip_tree.column("momentum", width=100)
        self.ip_tree.pack(fill="both", expand=True)

        # Build context menu and bind right-clicks
        self._build_context_menu()
        self.proc_tree.bind("<Button-3>", self._on_proc_right_click)
        self.net_tree.bind("<Button-3>", self._on_net_right_click)
        self.file_tree.bind("<Button-3>", self._on_file_right_click)

        # Double-click details on network trees
        self.net_tree.bind("<Double-1>", self._on_net_double_click)
        self.suspicious_tree.bind("<Double-1>", self._on_net_double_click)
        self.all_tree.bind("<Double-1>", self._on_net_double_click)

        # Color tags
        self._init_color_tags(self.proc_tree)
        self._init_color_tags(self.file_tree)
        self._init_color_tags(self.net_tree)
        self._init_color_tags(self.suspicious_tree)
        self._init_color_tags(self.all_tree)
        self._init_color_tags(self.timeline_tree)
        self._init_color_tags(self.ip_tree)

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
        row = self.net_tree.identify_row(event.y)
        if row:
            self.net_tree.selection_set(row)
            vals = self.net_tree.item(row)["values"]
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
        self.learning.persist_state()
        self.show_notification("Allow", f"Allowed {item_type}: {value}")
        self._flash_selected_row(color="green")

    def _block_selected(self):
        if not self.context_target:
            return
        item_type, value = self.context_target
        self.learning.add_block(item_type, value, notes="Operator block")
        self.learning.persist_state()
        self.show_notification("Block", f"Blocked {item_type}: {value}")
        self._flash_selected_row(color="red")

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
        self.learning.persist_state()
        self.show_notification("Nuclear Block", f"Blocked {value} and all child processes (rules only).")
        self._flash_selected_row(color="red")

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
        self.learning.persist_state()
        self.show_notification("Family Block", f"Blocked family: {wildcard} (rules only).")
        self._flash_selected_row(color="red")

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
        self.learning.persist_state()
        self.show_notification("Persistence Block", f"Blocked process and persistence (rules only): {pname}")
        self._flash_selected_row(color="red")

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
                self.learning.persist_state()
                self.show_notification("Network Scope Block", f"Blocked IP and range (rules only): {ip_prefix}")
                self._flash_selected_row(color="red")
            else:
                self.show_notification("Error", "Invalid IP format.")
        elif item_type == "process":
            pname = value
            self.learning.add_block("process", pname, notes="Block process (net scope)")
            self.learning.persist_state()
            self.show_notification("Network Scope Block", f"Blocked process (rules only): {pname}")
            self._flash_selected_row(color="red")
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
        self.learning.persist_state()
        self.show_notification("File Scope Block", f"Blocked process, path, and hash (rules only): {pname}")
        self._flash_selected_row(color="red")

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
        self.learning.persist_state()
        self.show_notification("Quarantine", f"Quarantine rule added (rules only) for: {pname}")
        self._flash_selected_row(color="red")

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
            self.learning.persist_state()

    def _init_color_tags(self, tree: ttk.Treeview):
        tree.tag_configure("green", background="#d4f8d4")
        tree.tag_configure("white", background="#ffffff")
        tree.tag_configure("yellow", background="#fff7c2")
        tree.tag_configure("red", background="#ffd6d6")
        tree.tag_configure("purple", background="#e6d6ff")
        tree.tag_configure("flash", background="#ffaaaa")

    def _risk_to_tag(self, risk: float, blocked: bool = False, outside_us: bool = False) -> str:
        if blocked:
            return "purple"
        if risk >= 0.8:
            return "red"
        if risk >= 0.4:
            return "yellow"
        if risk <= 0.2:
            return "green"
        return "white"

    def flash_row(self, tree, row_id, color="red", duration=800):
        tree.tag_configure("flash", background=color)
        orig_tags = tree.item(row_id, "tags")

        tree.item(row_id, tags=("flash",))

        def restore():
            tree.item(row_id, tags=orig_tags)

        self.root.after(duration, restore)

    def _flash_selected_row(self, color):
        for tree in (self.proc_tree, self.net_tree, self.file_tree,
                     self.suspicious_tree, self.all_tree,
                     self.timeline_tree, self.ip_tree):
            sel = tree.selection()
            if sel:
                self.flash_row(tree, sel[0], color=color)

    def _on_net_double_click(self, event):
        tree = event.widget
        row = tree.identify_row(event.y)
        if not row:
            return
        vals = tree.item(row)["values"]
        if len(vals) < 4:
            return
        proc, ip, port, risk_str = vals
        port = int(port)
        info = None
        for c in self._last_connections:
            if c.process_name == proc and c.destination_ip == ip and c.destination_port == port:
                info = c
                break
        if not info:
            return

        win = tk.Toplevel(self.root)
        win.title("Connection Details")
        win.geometry("400x260")

        lines = [
            f"Process: {info.process_name}",
            f"IP: {info.destination_ip}",
            f"Country: {info.destination_country}",
            f"Port: {info.destination_port}",
            f"Encrypted: {info.is_encrypted}",
            f"Outbound: {info.is_outbound}",
            f"Outside US: {info.is_outside_us}",
            f"Blocked (rules only): {info.is_blocked}",
            f"Anomaly Score: {info.anomaly_score:.2f}",
            f"Threat Momentum: {info.threat_momentum:.3f}",
        ]
        text = tk.Text(win, state="normal")
        text.pack(fill="both", expand=True)
        for line in lines:
            text.insert("end", line + "\n")
        text.config(state="disabled")

    def update_dashboard(self,
                         process_risk: Dict[str, float],
                         file_risk: Dict[str, float],
                         connections: List[NetworkConnectionInfo],
                         allow_list: List[Dict],
                         block_list: List[Dict],
                         process_momentum: Dict[str, float],
                         ip_history: Dict[str, Dict]):
        self._last_connections = list(connections)

        def refresh():
            # Process risk
            for i in self.proc_tree.get_children():
                self.proc_tree.delete(i)
            for pname, risk in sorted(process_risk.items(), key=lambda x: -x[1])[:80]:
                spark = self.learning.get_sparkline(pname)
                val = f"{pname} ({risk:.2f})  {spark}"
                tag = self._risk_to_tag(risk)
                self.proc_tree.insert("", "end", values=(val,), tags=(tag,))

            # File risk
            for i in self.file_tree.get_children():
                self.file_tree.delete(i)
            for path, risk in sorted(file_risk.items(), key=lambda x: -x[1])[:80]:
                tag = self._risk_to_tag(risk)
                self.file_tree.insert("", "end", values=(f"{risk:.2f}",), text=path, tags=(tag,))

            # Network (main)
            for i in self.net_tree.get_children():
                self.net_tree.delete(i)
            for c in connections[:80]:
                risk = max(c.anomaly_score, 1.0 if c.is_blocked else 0.0)
                tag = self._risk_to_tag(risk, blocked=c.is_blocked, outside_us=c.is_outside_us)
                self.net_tree.insert(
                    "",
                    "end",
                    values=(c.process_name, c.destination_ip, c.destination_port, f"{risk:.2f}"),
                    tags=(tag,)
                )

            # Network Threat Panel: Suspicious + All
            for i in self.suspicious_tree.get_children():
                self.suspicious_tree.delete(i)
            for i in self.all_tree.get_children():
                self.all_tree.delete(i)

            for c in connections:
                risk = max(c.anomaly_score, 1.0 if c.is_blocked else 0.0)
                row = (c.process_name, c.destination_ip, c.destination_port, f"{risk:.2f}")
                tag = self._risk_to_tag(risk, blocked=c.is_blocked, outside_us=c.is_outside_us)

                self.all_tree.insert("", "end", values=row, tags=(tag,))

                if c.is_blocked or c.anomaly_score >= 0.8 or c.is_outside_us:
                    self.suspicious_tree.insert("", "end", values=row, tags=(tag,))

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

            # Process Timeline Panel
            for i in self.timeline_tree.get_children():
                self.timeline_tree.delete(i)
            for pname, risk in sorted(process_risk.items(), key=lambda x: -x[1])[:120]:
                spark = self.learning.get_sparkline(pname)
                mom = process_momentum.get(pname, 0.0)
                tag = self._risk_to_tag(risk)
                self.timeline_tree.insert(
                    "",
                    "end",
                    values=(pname, spark, f"{mom:+.3f}"),
                    tags=(tag,)
                )

            # IP Anomaly History Panel
            for i in self.ip_tree.get_children():
                self.ip_tree.delete(i)
            for ip, info in ip_history.items():
                anomalies = info.get("anomalies", [])
                count = len(anomalies)
                last = anomalies[-1][1] if anomalies else 0.0
                mom = info.get("momentum", 0.0)
                tag = self._risk_to_tag(last)
                self.ip_tree.insert(
                    "",
                    "end",
                    values=(ip, count, f"{last:.2f}", f"{mom:+.3f}"),
                    tags=(tag,)
                )

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
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        self._learning.load_state()
        await self._file_scanner.initial_marker_walk()

        self._tasks.append(asyncio.create_task(self._network_guardian.monitor_loop()))
        self._tasks.append(asyncio.create_task(self._camera_mic_guardian.monitor_loop()))
        self._tasks.append(asyncio.create_task(self._daily_hot_scan_loop()))
        self._tasks.append(asyncio.create_task(self._daily_integrity_check_loop()))
        self._tasks.append(asyncio.create_task(self._daily_iot_scan_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

        self._gui.set_system_led(LedState.GREEN, "Guardian running (Nuclear Mode, safe engine).")
        self._gui.set_personal_data_led(
            LedState.YELLOW,
            "Radioactive: all programs sanitized"
        )
        self._gui.set_network_led(LedState.GREEN, "Network monitored.")

    async def _daily_hot_scan_loop(self):
        last_run = 0.0
        while True:
            now = time.time()
            if now - last_run >= 24 * 3600:
                await self._file_scanner.scan_hot_set()
                last_run = now
            await smart_sleep(60.0, self._load_monitor)

    async def _daily_integrity_check_loop(self):
        last_run = 0.0
        while True:
            now = time.time()
            if now - last_run >= 24 * 3600:
                await self._self_integrity.verify_self()
                if not self._self_integrity.is_healthy:
                    self._gui.set_system_led(LedState.RED, "Guardian integrity issue.")
                    self._gui.show_notification("Guardian damaged", self._self_integrity.last_issue)
                last_run = now
            await smart_sleep(60.0, self._load_monitor)

    async def _daily_iot_scan_loop(self):
        last_run = 0.0
        while True:
            now = time.time()
            if now - last_run >= 24 * 3600:
                await self._iot_guardian.scan_network()
                last_run = now
            await smart_sleep(60.0, self._load_monitor)

    async def _heartbeat_loop(self):
        while True:
            global_risk = self._learning.compute_global_risk()
            if global_risk >= 0.8:
                self._gui.set_system_led(LedState.RED, "High global risk (rules only).")
            elif global_risk >= 0.4:
                self._gui.set_system_led(LedState.YELLOW, "Elevated global risk.")
            else:
                self._gui.set_system_led(LedState.GREEN, "Guardian running (low risk).")
            await smart_sleep(5.0, self._load_monitor)

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._learning.persist_state()

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
        learning.persist_state()
        gui.show_notification("Storage", f"Auto-selected storage: {auto_path}")

    base_dir = learning.storage_path
    ensure_dir(base_dir)

    marker_store = FileMarkerStore(base_dir)
    file_scanner = FileScanner(marker_store, load_monitor, gui, learning)
    identity_firewall = IdentityFirewall()
    self_integrity = SelfIntegrityMonitor()
    iot_guardian = IoTGuardian(gui)
    network_guardian = NetworkGuardian(gui, learning, load_monitor)
    camera_mic_guardian = CameraMicGuardian(gui, learning)

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
    )

    gui.engine = engine  # optional handle if you want later

    await engine.start()

    # Example outbound request to exercise identity firewall
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

