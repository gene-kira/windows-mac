import sys
import time
import threading
import queue
import re
import os
import json
import string
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, List, Set, Optional

import psutil  # pip install psutil
from PyQt5 import QtWidgets, QtCore, QtGui


# =========================
# Reboot memory manager
# =========================

class RebootMemoryManager:
    """
    Persistent reboot memory with drive priority:
    1. Any drive D: through Z:
    2. If none available, fallback to C:
    """

    def __init__(self, filename="reboot_memory.json"):
        self.filename = filename
        self.memory_path = self._select_memory_path()
        self.full_path = self.memory_path / self.filename
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def _select_memory_path(self) -> Path:
        for letter in string.ascii_uppercase[3:]:  # D is index 3
            drive = Path(f"{letter}:/")
            if drive.exists():
                return drive / "QueenGuardMemory"
        return Path("C:/QueenGuardMemory")

    def save(self, state: dict):
        try:
            with open(self.full_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"[RebootMemory] ERROR saving memory: {e}")

    def load(self) -> dict:
        if not self.full_path.exists():
            return {}
        try:
            with open(self.full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_storage_location(self) -> str:
        return str(self.full_path)


# =========================
# Data models and utilities
# =========================

@dataclass
class SensitiveTokenConfig:
    personal_tokens: List[str] = field(default_factory=list)
    system_tokens: List[str] = field(default_factory=list)


@dataclass
class ConnectionEvent:
    timestamp: float
    direction: str          # "OUTBOUND" or "INBOUND"
    process_name: str
    pid: int
    origin: str             # designator (Windows, Steam, Game, Browser, etc.)
    local_addr: str
    remote_addr: str
    remote_port: int
    country: str
    region: str
    city: str
    resolution: int         # 0-3
    confidence: str         # Low/Medium/High
    overseas: bool
    pii_detected: bool
    pii_types: List[str]
    safe_startup: bool      # .py in Startup
    heavy_watch: bool = False   # radioactive / heavy monitoring
    risk_score: int = 0         # filled by queen
    pending_approval: bool = False  # yellow state


@dataclass
class PolicyDecision:
    allow_always: bool
    block_always: bool
    radioactive: bool = False   # radioactive policy


# =========================
# PII detection
# =========================

class PIIDetector:
    def __init__(self, config: SensitiveTokenConfig):
        self.config = config
        self.ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        self.phone_pattern = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
        self.email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def scan_payload(self, payload: str) -> Tuple[bool, List[str]]:
        types = []

        if self.ssn_pattern.search(payload):
            types.append("SSN")
        if self.phone_pattern.search(payload):
            types.append("PHONE")
        if self.email_pattern.search(payload):
            types.append("EMAIL")

        for token in self.config.personal_tokens + self.config.system_tokens:
            if token and token in payload:
                types.append("TOKEN")

        return (len(types) > 0, list(set(types)))


# =========================
# Geo / Prometheus-style scanner
# =========================

class GeoResolver:
    """
    Prometheus-style scanner:
    - Country
    - Region (state/province)
    - City
    - Resolution score (0-3)
    - Confidence level
    """

    def __init__(self, home_country: str = "US", db_path: str = "GeoLite2-City.mmdb"):
        self.home_country = home_country
        self.db_path = db_path
        self._geo_db_loaded = False
        self._geo_reader = None

    def _lazy_load_geo_db(self):
        if self._geo_db_loaded:
            return
        self._geo_db_loaded = True
        try:
            import geoip2.database  # pip install geoip2
            self._geo_reader = geoip2.database.Reader(self.db_path)
        except Exception:
            self._geo_reader = None

    def scan_ip(self, ip: str) -> Tuple[str, str, str, int, str, bool]:
        """
        Returns:
        country, region, city, resolution(0-3), confidence("Low/Medium/High"), overseas(bool)
        """
        if ip == "127.0.0.1" or ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172."):
            country = self.home_country
            region = "Local"
            city = "Local"
            resolution = 3
            confidence = "High"
            overseas = False
            return country, region, city, resolution, confidence, overseas

        self._lazy_load_geo_db()
        if not self._geo_reader:
            return "??", "Unknown", "Unknown", 0, "Low", True

        try:
            resp = self._geo_reader.city(ip)
            country = resp.country.iso_code or "??"
            region = (resp.subdivisions.most_specific.name or "Unknown") if resp.subdivisions else "Unknown"
            city = resp.city.name or "Unknown"

            resolution = 0
            if country != "??":
                resolution = 1
            if region != "Unknown":
                resolution = 2
            if city != "Unknown":
                resolution = 3

            if resolution == 3:
                confidence = "High"
            elif resolution == 2:
                confidence = "Medium"
            elif resolution == 1:
                confidence = "Low"
            else:
                confidence = "Low"

            overseas = (country != self.home_country)
            return country, region, city, resolution, confidence, overseas
        except Exception:
            return "??", "Unknown", "Unknown", 0, "Low", True


# =========================
# Startup .py safe detection
# =========================

def get_startup_dirs() -> List[Path]:
    dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / r"Microsoft\Windows\Start Menu\Programs\Startup")
    programdata = os.environ.get("PROGRAMDATA")
    if programdata:
        dirs.append(Path(programdata) / r"Microsoft\Windows\Start Menu\Programs\Startup")
    return dirs


def is_safe_startup_py(proc: psutil.Process, startup_dirs: List[Path]) -> bool:
    try:
        cmdline = proc.cmdline()
    except Exception:
        cmdline = []

    for arg in cmdline[1:]:
        p = Path(arg)
        if p.suffix.lower() == ".py":
            for sdir in startup_dirs:
                try:
                    if sdir.exists() and sdir in p.parents:
                        return True
                except Exception:
                    continue
    return False


# =========================
# Origin / designator classification
# =========================

def classify_origin(proc: psutil.Process) -> str:
    try:
        name = proc.name().lower()
    except Exception:
        name = "unknown"

    try:
        exe = proc.exe().lower()
    except Exception:
        exe = ""

    # Games (extend as needed)
    if "back4blood" in exe or "back4blood" in name:
        return "Game.Back4Blood"
    if "steam.exe" in exe or "steam" in name:
        return "Steam.Client"
    if "epicgameslauncher" in exe or "epic" in name:
        return "Epic.Launcher"

    # Windows / system
    if "svchost.exe" in exe or "svchost" in name:
        return "Windows.ServiceHost"
    if "explorer.exe" in exe or "explorer" in name:
        return "Windows.Shell"
    if "system" in name:
        return "Windows.Core"

    # GPU / telemetry
    if "nvidia" in exe or "nvcontainer" in name:
        return "NVIDIA.Telemetry"

    # Browsers
    if "chrome.exe" in exe or "chrome" in name:
        return "Browser.Chrome"
    if "msedge.exe" in exe or "edge" in name:
        return "Browser.Edge"
    if "firefox.exe" in exe or "firefox" in name:
        return "Browser.Firefox"

    # Python
    if exe.endswith(".exe") and "python" in exe:
        return "Python.Script"
    if "python" in name:
        return "Python.Script"

    if name == "unknown":
        return "Unknown.Process"

    return f"App.{name}"


def is_game_origin(origin: str) -> bool:
    return origin.startswith("Game.") or origin.startswith("Steam.") or origin.startswith("Epic.")


def is_browser_origin(origin: str) -> bool:
    return origin.startswith("Browser.")


# =========================
# Risk engine (predictive + anomaly + ML hook)
# =========================

class RiskEngine:
    """
    Risk scoring based on:
    - process trust
    - destination novelty
    - overseas
    - PII
    - heavy-watch (radioactive)
    - browser sensitivity
    - fan-out (many destinations)
    - drift (new countries)
    - anomaly detection
    - ML scoring hook
    """

    def __init__(self, history: Dict[Tuple[str, str], dict]):
        self.history = history

    def compute_risk(self, event: ConnectionEvent) -> int:
        base = self._compute_base_risk(event)
        if self.is_anomalous(event):
            base += 20
        ml = self.ml_score(event)
        risk = max(0, min(100, base + ml))
        return risk

    def _compute_base_risk(self, event: ConnectionEvent) -> int:
        key = (event.process_name, event.remote_addr)
        h = self.history.get(key, {})

        risk = 0

        # Process trust
        if event.safe_startup:
            risk += 0
        else:
            risk += 10

        # Overseas
        if event.overseas:
            risk += 25

        # Novelty (per (proc, remote))
        if not h:
            risk += 20
        else:
            count = h.get("count", 1)
            if count < 3:
                risk += 10

        # Browser sensitivity
        if is_browser_origin(event.origin):
            risk += 15

        # PII (we still score it, but we won't block)
        if event.pii_detected:
            risk += 30

        # Heavy watch (radioactive)
        if event.heavy_watch:
            risk += 20

        # Fan-out: how many distinct remotes this process has
        proc_key = (event.process_name, "__ALL__")
        proc_hist = self.history.get(proc_key, {})
        distinct_remotes = proc_hist.get("distinct_remotes", 1)
        if distinct_remotes > 10:
            risk += 15
        elif distinct_remotes > 5:
            risk += 8

        # Drift: new countries for this process
        countries_seen: Set[str] = set(proc_hist.get("countries", []))
        if event.country not in countries_seen and event.country not in ("??", ""):
            risk += 10

        return risk

    def is_anomalous(self, event: ConnectionEvent) -> bool:
        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        countries = set(ph.get("countries", []))
        remotes = set(ph.get("remotes", []))

        new_country = event.country not in ("", "??") and event.country not in countries
        new_remote = event.remote_addr not in remotes

        return new_country or new_remote

    def ml_score(self, event: ConnectionEvent) -> int:
        # Placeholder for future ML model
        # You can later plug in a real model here.
        return 0


# =========================
# Board guard monitor (event bus producer)
# =========================

class BoardGuardMonitor(threading.Thread):
    """
    Background watcher: scans active connections, tags geo + PII + origin,
    and reports to the queen via event_queue as bus messages.
    """
    def __init__(self,
                 event_queue: queue.Queue,
                 pii_detector: PIIDetector,
                 geo_resolver: GeoResolver,
                 poll_interval: float = 2.0):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.pii_detector = pii_detector
        self.geo_resolver = geo_resolver
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._seen_connections = set()
        self.startup_dirs = get_startup_dirs()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.scan_connections()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def scan_connections(self):
        for conn in psutil.net_connections(kind='inet'):
            if not conn.raddr:
                continue

            pid = conn.pid or -1
            laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            raddr = conn.raddr.ip
            rport = conn.raddr.port

            key = (pid, laddr, raddr, rport)
            if key in self._seen_connections:
                continue
            self._seen_connections.add(key)

            try:
                proc = psutil.Process(pid) if pid > 0 else None
                pname = proc.name() if proc else "UNKNOWN"
            except Exception:
                proc = None
                pname = "UNKNOWN"

            safe_startup = False
            origin = "Unknown.Process"
            if proc is not None:
                safe_startup = is_safe_startup_py(proc, self.startup_dirs)
                origin = classify_origin(proc)
            else:
                origin = "Unknown.Process"

            country, region, city, resolution, confidence, overseas = self.geo_resolver.scan_ip(raddr)

            # Simulated payload for PII detection (we allow PII, just score it)
            if raddr == "127.0.0.1":
                pii_detected = False
                pii_types = []
            else:
                if safe_startup:
                    pii_detected = False
                    pii_types = []
                else:
                    simulated_payload = f"{pname} talking to {raddr}"
                    pii_detected, pii_types = self.pii_detector.scan_payload(simulated_payload)

            heavy_watch = True

            event = ConnectionEvent(
                timestamp=time.time(),
                direction="OUTBOUND",
                process_name=pname,
                pid=pid,
                origin=origin,
                local_addr=laddr,
                remote_addr=raddr,
                remote_port=rport,
                country=country,
                region=region,
                city=city,
                resolution=resolution,
                confidence=confidence,
                overseas=overseas,
                pii_detected=pii_detected,
                pii_types=pii_types,
                safe_startup=safe_startup,
                heavy_watch=heavy_watch
            )
            msg = {"type": "connection", "payload": event}
            self.event_queue.put(msg)


# =========================
# Queen GUI (PyQt5)
# =========================

class QueenWindow(QtWidgets.QMainWindow):
    """
    Fully automated queen:
    - Everything starts radioactive (heavy-watch)
    - System can identify "would-be trusted" flows, but:
      * They go to yellow "Pending Admin Approval"
      * They remain radioactive internally until admin approves
    - PII is silently allowed (but scored and shown)
    - Auto-block rules for extreme risk
    - Anomaly detection + ML hook
    - Right-click inline controls for manual override
    - Left-side manual buttons for override + Approve Trust
    - Legend bar for colors
    - Threat timeline
    - Quarantine mode
    - JSON export
    - Dark mode (set in main)
    """

    MODE_AUTOMATED = "Automated Sentinel"

    def __init__(self, event_queue: queue.Queue, memory: RebootMemoryManager):
        super().__init__()
        self.event_queue = event_queue
        self.memory = memory
        self.policy: Dict[Tuple[str, str], PolicyDecision] = {}
        self.history: Dict[Tuple[str, str], dict] = {}
        self.mode = self.MODE_AUTOMATED
        self.quarantined_procs: Set[str] = set()
        self.event_log: List[dict] = []
        self.timeline: List[dict] = []

        saved = self.memory.load()
        if "policies" in saved:
            for key_str, val in saved["policies"].items():
                try:
                    proc_name, remote_addr = key_str.split("||", 1)
                except ValueError:
                    continue
                self.policy[(proc_name, remote_addr)] = PolicyDecision(
                    allow_always=val.get("allow_always", False),
                    block_always=val.get("block_always", False),
                    radioactive=val.get("radioactive", False)
                )
        if "history" in saved:
            hist_raw = saved["history"]
            for key_str, h in hist_raw.items():
                try:
                    proc_name, remote_addr = key_str.split("||", 1)
                except ValueError:
                    continue
                self.history[(proc_name, remote_addr)] = h

        self.risk_engine = RiskEngine(self.history)

        self.setWindowTitle("Queen Guard - Personal Data Sentinel (Automated)")
        self.resize(1500, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top status
        top_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Automated Monitoring (No Popups, Admin-Gated Trust)")
        self.status_label.setStyleSheet("color: #22c55e; font-weight: bold;")
        top_layout.addWidget(self.status_label)

        self.mem_label = QtWidgets.QLabel(f"Memory: {self.memory.get_storage_location()}")
        self.mem_label.setStyleSheet("color: gray; font-size: 10px;")
        top_layout.addWidget(self.mem_label)

        main_layout.addLayout(top_layout)

        # Legend bar
        legend_layout = QtWidgets.QHBoxLayout()
        legend_label = QtWidgets.QLabel(
            "Legend:  "
            "<span style='background-color:#14532d;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Always Allow   "
            "<span style='background-color:#7f1d1d;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Always Block   "
            "<span style='background-color:#4c1d95;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Radioactive   "
            "<span style='background-color:#854d0e;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Pending Admin Approval"
        )
        legend_label.setStyleSheet("font-size: 12px;")
        legend_layout.addWidget(legend_label)
        main_layout.addLayout(legend_layout)

        # Main horizontal layout: left controls + center table + right panel
        body_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(body_layout)

        # Left-side manual override buttons
        left_panel = QtWidgets.QVBoxLayout()

        self.btn_allow = QtWidgets.QPushButton("Always Allow")
        self.btn_allow.clicked.connect(self.manual_always_allow)
        left_panel.addWidget(self.btn_allow)

        self.btn_block = QtWidgets.QPushButton("Always Block")
        self.btn_block.clicked.connect(self.manual_always_block)
        left_panel.addWidget(self.btn_block)

        self.btn_radio = QtWidgets.QPushButton("Keep Radioactive")
        self.btn_radio.clicked.connect(self.manual_keep_radioactive)
        left_panel.addWidget(self.btn_radio)

        self.btn_approve = QtWidgets.QPushButton("Approve Trust")
        self.btn_approve.clicked.connect(self.manual_approve_trust)
        left_panel.addWidget(self.btn_approve)

        self.btn_quarantine = QtWidgets.QPushButton("Quarantine Process")
        self.btn_quarantine.clicked.connect(self.quarantine_selected_process)
        left_panel.addWidget(self.btn_quarantine)

        self.btn_timeline = QtWidgets.QPushButton("Threat Timeline")
        self.btn_timeline.clicked.connect(self.show_timeline)
        left_panel.addWidget(self.btn_timeline)

        self.btn_proc_dash = QtWidgets.QPushButton("Process Dashboard")
        self.btn_proc_dash.clicked.connect(self.show_process_dashboard)
        left_panel.addWidget(self.btn_proc_dash)

        self.btn_export = QtWidgets.QPushButton("Export JSON")
        self.btn_export.clicked.connect(self.export_json)
        left_panel.addWidget(self.btn_export)

        left_panel.addStretch()
        body_layout.addLayout(left_panel, 1)

        # Center: table
        center_layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 18)
        self.table.setHorizontalHeaderLabels([
            "Time", "Dir", "Process", "PID", "Origin",
            "Local", "Remote",
            "Country", "Region", "City", "Res", "Conf",
            "Overseas", "PII", "Risk", "Watch", "Radioactive", "Pending"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        center_layout.addWidget(self.table)
        body_layout.addLayout(center_layout, 4)

        # Right panel: simple info / future expansion
        right_panel = QtWidgets.QVBoxLayout()

        self.save_local_btn = QtWidgets.QPushButton("Save Memory (Local)")
        self.save_local_btn.clicked.connect(self.manual_save_local)
        right_panel.addWidget(self.save_local_btn)

        self.save_smb_btn = QtWidgets.QPushButton("Save Memory (SMB)")
        self.save_smb_btn.clicked.connect(self.manual_save_smb)
        right_panel.addWidget(self.save_smb_btn)

        self.kill_btn = QtWidgets.QPushButton("Kill Selected Process")
        self.kill_btn.clicked.connect(self.kill_selected_process)
        right_panel.addWidget(self.kill_btn)

        right_panel.addStretch()
        body_layout.addLayout(right_panel, 1)

        # Right-click context menu on table
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_table_context_menu)

        # Timers
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.drain_events)
        self.timer.start(500)

        self.save_timer = QtCore.QTimer(self)
        self.save_timer.timeout.connect(self.save_state)
        self.save_timer.start(5000)

    # ---------- Event bus drain ----------

    def drain_events(self):
        while True:
            try:
                msg = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if not isinstance(msg, dict) or "type" not in msg:
                continue

            if msg["type"] == "connection":
                event: ConnectionEvent = msg["payload"]
                self.handle_event(event)
            elif msg["type"] == "alert":
                # placeholder for future alert types
                pass

    # ---------- History / event log ----------

    def update_history(self, event: ConnectionEvent):
        key = (event.process_name, event.remote_addr)
        h = self.history.get(key, {})
        h["count"] = h.get("count", 0) + 1
        h["first_seen"] = h.get("first_seen", event.timestamp)
        h["last_seen"] = event.timestamp
        h["overseas"] = event.overseas
        h["country"] = event.country
        h["region"] = event.region
        h["port"] = event.remote_port
        h["origin"] = event.origin
        self.history[key] = h

        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        remotes: Set[str] = set(ph.get("remotes", []))
        remotes.add(event.remote_addr)
        ph["remotes"] = list(remotes)
        ph["distinct_remotes"] = len(remotes)

        countries: Set[str] = set(ph.get("countries", []))
        if event.country not in ("", "??"):
            countries.add(event.country)
        ph["countries"] = list(countries)

        self.history[proc_key] = ph

    def log_event(self, event: ConnectionEvent, action: str):
        d = asdict(event)
        d["action"] = action
        self.event_log.append(d)

        if event.risk_score >= 60 or action in ("auto_block", "quarantine_block"):
            self.timeline.append({
                "timestamp": event.timestamp,
                "process": event.process_name,
                "remote": event.remote_addr,
                "risk": event.risk_score,
                "action": action,
            })

    # ---------- Trust / auto logic ----------

    def _auto_trusted(self, event: ConnectionEvent) -> bool:
        if event.process_name == "UNKNOWN":
            return False
        if event.origin.startswith("Unknown"):
            return False
        if event.overseas:
            return False
        if event.risk_score >= 30:
            return False
        if event.resolution >= 2 and event.confidence in ("Medium", "High"):
            return True
        return False

    def handle_event(self, event: ConnectionEvent):
        event.heavy_watch = True
        event.pending_approval = False

        self.update_history(event)
        self.risk_engine.history = self.history
        event.risk_score = self.risk_engine.compute_risk(event)

        key = (event.process_name, event.remote_addr)
        decision = self.policy.get(key)

        # Quarantine mode
        if event.process_name in self.quarantined_procs:
            auto_dec = PolicyDecision(False, True, False)
            self.policy[key] = auto_dec
            row = self.add_event_row(event, forced_block=True)
            self._recolor_row_policy(row, "block")
            self.log_event(event, "quarantine_block")
            return

        # Auto-block rules (extreme risk)
        if event.risk_score >= 80 or (event.overseas and event.pii_detected):
            auto_dec = PolicyDecision(False, True, False)
            self.policy[key] = auto_dec
            row = self.add_event_row(event, forced_block=True)
            self._recolor_row_policy(row, "block")
            self.log_event(event, "auto_block")
            return

        # Manual policy first
        if decision:
            if decision.block_always:
                row = self.add_event_row(event, forced_block=True)
                self._recolor_row_policy(row, "block")
                self.log_event(event, "manual_block")
                return

            if decision.allow_always:
                event.heavy_watch = False
                row = self.add_event_row(event, forced_allow=True)
                self._recolor_row_policy(row, "allow")
                self.log_event(event, "manual_allow")
                return

            if decision.radioactive:
                event.heavy_watch = True
                row = self.add_event_row(event, forced_allow=True)
                self._recolor_row_policy(row, "radioactive")
                self.log_event(event, "manual_radioactive")
                return

        # Automated classification (no auto-green, only pending)
        if self._auto_trusted(event):
            event.pending_approval = True
            event.heavy_watch = True
            row = self.add_event_row(event, forced_allow=True, pending=True)
            self._recolor_row_policy(row, "pending")
            self.log_event(event, "pending_trust")
        else:
            event.heavy_watch = True
            auto_dec = self.policy.get(key, PolicyDecision(False, False, True))
            auto_dec.radioactive = True
            self.policy[key] = auto_dec
            row = self.add_event_row(event, forced_allow=True)
            self._recolor_row_policy(row, "radioactive")
            self.log_event(event, "auto_radioactive")

    # ---------- Table row ----------

    def add_event_row(self, event: ConnectionEvent,
                      forced_allow: bool = False,
                      forced_block: bool = False,
                      pending: bool = False) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)

        t_str = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        pii_str = "YES" if event.pii_detected else "NO"
        overseas_str = "YES" if event.overseas else "NO"
        risk_str = str(event.risk_score)
        watch_str = "HEAVY" if event.heavy_watch else ""
        radioactive_str = "YES" if event.heavy_watch else "NO"
        pending_str = "YES" if pending else "NO"
        res_str = f"{event.resolution}/3"

        values = [
            t_str,
            event.direction,
            event.process_name,
            str(event.pid),
            event.origin,
            event.local_addr,
            f"{event.remote_addr}:{event.remote_port}",
            event.country,
            event.region,
            event.city,
            res_str,
            event.confidence,
            overseas_str,
            pii_str,
            risk_str,
            watch_str,
            radioactive_str,
            pending_str
        ]

        for col, val in enumerate(values):
            item = QtWidgets.QTableWidgetItem(val)
            if forced_block:
                item.setBackground(QtGui.QColor("#7f1d1d"))
            elif pending:
                item.setBackground(QtGui.QColor("#854d0e"))
            elif event.heavy_watch:
                item.setBackground(QtGui.QColor("#4c1d95"))
            else:
                if event.risk_score >= 70:
                    item.setBackground(QtGui.QColor("#7f1d1d"))
                elif event.risk_score >= 40:
                    item.setBackground(QtGui.QColor("#854d0e"))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        return row

    # ---------- Context menu ----------

    def show_table_context_menu(self, pos: QtCore.QPoint):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return

        menu = QtWidgets.QMenu(self)

        always_allow_action = menu.addAction("Always Allow")
        always_block_action = menu.addAction("Always Block")
        radioactive_action = menu.addAction("Keep Radioactive")
        approve_action = menu.addAction("Approve Trust")

        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if not action:
            return

        key = self._get_selected_key_from_row(row)
        if not key:
            return
        process_name, remote_addr = key

        decision = self.policy.get((process_name, remote_addr), PolicyDecision(False, False, False))

        if action == always_allow_action:
            decision.allow_always = True
            decision.block_always = False
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "allow")
        elif action == always_block_action:
            decision.allow_always = False
            decision.block_always = True
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "block")
        elif action == radioactive_action:
            decision.allow_always = False
            decision.block_always = False
            decision.radioactive = True
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "radioactive")
        elif action == approve_action:
            decision.allow_always = True
            decision.block_always = False
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "allow")
            pending_item = self.table.item(row, 17)
            if pending_item:
                pending_item.setText("NO")

    # ---------- Manual buttons helpers ----------

    def _get_selected_row(self) -> int:
        return self.table.currentRow()

    def _get_selected_key_from_row(self, row: int) -> Optional[Tuple[str, str]]:
        proc_item = self.table.item(row, 2)
        remote_item = self.table.item(row, 6)
        if not proc_item or not remote_item:
            return None
        process_name = proc_item.text()
        remote_addr = remote_item.text().split(":", 1)[0]
        return process_name, remote_addr

    def _get_selected_key(self):
        row = self._get_selected_row()
        if row < 0:
            return None
        key = self._get_selected_key_from_row(row)
        if not key:
            return None
        process_name, remote_addr = key
        return process_name, remote_addr, row

    def manual_always_allow(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(True, False, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "allow")
        pending_item = self.table.item(row, 17)
        if pending_item:
            pending_item.setText("NO")

    def manual_always_block(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(False, True, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "block")

    def manual_keep_radioactive(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(False, False, True)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "radioactive")

    def manual_approve_trust(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(True, False, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "allow")
        pending_item = self.table.item(row, 17)
        if pending_item:
            pending_item.setText("NO")

    def _recolor_row_policy(self, row: int, mode: str):
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if not item:
                continue
            if mode == "allow":
                item.setBackground(QtGui.QColor("#14532d"))
            elif mode == "block":
                item.setBackground(QtGui.QColor("#7f1d1d"))
            elif mode == "radioactive":
                item.setBackground(QtGui.QColor("#4c1d95"))
            elif mode == "pending":
                item.setBackground(QtGui.QColor("#854d0e"))

    # ---------- Quarantine ----------

    def quarantine_selected_process(self):
        row = self._get_selected_row()
        if row < 0:
            return
        proc_item = self.table.item(row, 2)
        if not proc_item:
            return
        process_name = proc_item.text()
        self.quarantined_procs.add(process_name)

    # ---------- Threat timeline ----------

    def show_timeline(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threat Timeline")
        dlg.resize(800, 400)
        layout = QtWidgets.QVBoxLayout(dlg)

        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Time", "Process", "Remote", "Action/Risk"])
        layout.addWidget(table)

        for entry in sorted(self.timeline, key=lambda x: x["timestamp"]):
            row = table.rowCount()
            table.insertRow(row)
            t_str = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(t_str))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(entry["process"]))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(entry["remote"]))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{entry['action']} (risk={entry['risk']})"))

        dlg.exec_()

    # ---------- Process dashboard ----------

    def show_process_dashboard(self):
        row = self._get_selected_row()
        if row < 0:
            return
        proc_item = self.table.item(row, 2)
        if not proc_item:
            return
        process_name = proc_item.text()

        per_remote = {
            remote: h for (p, remote), h in self.history.items()
            if p == process_name and remote != "__ALL__"
        }
        proc_key = (process_name, "__ALL__")
        aggregate = self.history.get(proc_key, {})

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Process Dashboard - {process_name}")
        dlg.resize(900, 500)
        layout = QtWidgets.QVBoxLayout(dlg)

        agg_label = QtWidgets.QLabel(
            f"Distinct remotes: {aggregate.get('distinct_remotes', 0)} | "
            f"Countries: {', '.join(aggregate.get('countries', []))}"
        )
        layout.addWidget(agg_label)

        table = QtWidgets.QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["Remote", "Count", "Country", "Region", "Port", "Last Seen"])
        layout.addWidget(table)

        for remote, h in per_remote.items():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(remote))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(h.get("count", 0))))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(h.get("country", "")))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(h.get("region", "")))
            table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(h.get("port", ""))))
            ts = h.get("last_seen", 0)
            t_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
            table.setItem(row, 5, QtWidgets.QTableWidgetItem(t_str))

        dlg.exec_()

    # ---------- JSON export ----------

    def export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        state = self._serialize_state()
        data = {
            "events": self.event_log,
            "policies": state["policies"],
            "history": state["history"],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export JSON:\n{e}")

    # ---------- State serialization ----------

    def _serialize_state(self) -> dict:
        policies_serialized = {}
        for (proc_name, remote_addr), dec in self.policy.items():
            key_str = f"{proc_name}||{remote_addr}"
            policies_serialized[key_str] = {
                "allow_always": dec.allow_always,
                "block_always": dec.block_always,
                "radioactive": dec.radioactive,
            }

        history_serialized = {}
        for (proc_name, remote_addr), h in self.history.items():
            key_str = f"{proc_name}||{remote_addr}"
            history_serialized[key_str] = h

        state = {
            "policies": policies_serialized,
            "history": history_serialized,
        }
        return state

    def save_state(self):
        state = self._serialize_state()
        self.memory.save(state)

    def _save_to_path(self, path: Path):
        state = self._serialize_state()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save memory:\n{e}")

    # ---------- Manual save buttons ----------

    def manual_save_local(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Local Folder")
        if not folder:
            return
        path = Path(folder) / "queen_memory.json"
        self._save_to_path(path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Memory saved to:\n{path}")

    def manual_save_smb(self):
        smb_path, ok = QtWidgets.QInputDialog.getText(
            self,
            "SMB Path",
            "Enter SMB path (e.g. \\\\SERVER\\Share):"
        )
        if not ok or not smb_path:
            return
        path = Path(smb_path) / "queen_memory.json"
        self._save_to_path(path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Memory saved to:\n{path}")

    # ---------- Kill process ----------

    def kill_selected_process(self):
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Select a row first.")
            return

        pid_item = self.table.item(row, 3)
        if not pid_item:
            QtWidgets.QMessageBox.warning(self, "No PID", "No PID found for selected row.")
            return

        try:
            pid = int(pid_item.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid PID", "PID is not a valid number.")
            return

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Kill Process",
            f"Are you sure you want to kill PID {pid}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        try:
            p = psutil.Process(pid)
            p.terminate()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to kill process:\n{e}")


# =========================
# Main entry point
# =========================

def main():
    sensitive_config = SensitiveTokenConfig(
        personal_tokens=[
            # e.g. "555-123-4567", "myemail@example.com"
        ],
        system_tokens=[
            # e.g. hostname, serials, etc.
        ]
    )

    pii_detector = PIIDetector(sensitive_config)
    geo_resolver = GeoResolver(home_country="US", db_path="GeoLite2-City.mmdb")

    memory = RebootMemoryManager()
    event_q = queue.Queue()

    monitor = BoardGuardMonitor(event_q, pii_detector, geo_resolver, poll_interval=3.0)
    monitor.start()

    app = QtWidgets.QApplication(sys.argv)

    # Dark mode stylesheet
    dark_qss = """
    QWidget {
        background-color: #0b1120;
        color: #e5e7eb;
    }
    QTableWidget {
        gridline-color: #1f2937;
        background-color: #020617;
    }
    QHeaderView::section {
        background-color: #111827;
        color: #e5e7eb;
    }
    QPushButton {
        background-color: #1f2933;
        color: #e5e7eb;
        border: 1px solid #374151;
        padding: 4px 8px;
    }
    QPushButton:hover {
        background-color: #111827;
    }
    QLineEdit, QPlainTextEdit, QTextEdit {
        background-color: #020617;
        color: #e5e7eb;
        border: 1px solid #374151;
    }
    QDialog {
        background-color: #020617;
    }
    """
    app.setStyleSheet(dark_qss)

    win = QueenWindow(event_q, memory)
    win.show()
    ret = app.exec_()

    monitor.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()

