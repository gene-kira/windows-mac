"""
Show Me Your Papers - Connection Scanner Cockpit (No WinDivert, Single File, Auto-Elevated)

User-mode only:
- Scans active TCP/UDP connections using psutil.net_connections.
- Maps connections to process name + PID.
- Per-domain trust + per-process policies.
- Behavior fingerprints + simple ML-style anomaly scoring.
- Threat lineage (per process + per domain).
- Real-time alerts for suspicious activity.
- Process Reputation panel.
- Domain Reputation panel.
- Threat Heatmap panel.
- Process Behavior Graph panel.
- PyQt5 GUI with color-coded rows.

Does NOT:
- Block or encrypt packets on the wire.
- Hook kernel or modify live traffic.
"""

import json
import os
import re
import uuid
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import ctypes
import psutil
from cryptography.fernet import Fernet

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QLabel, QComboBox, QTextEdit, QFormLayout, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

import sys
import platform


# === AUTO-ELEVATION CHECK (Windows only) ===
def ensure_admin():
    try:
        if platform.system().lower() == "windows":
            try:
                if ctypes.windll.shell32.IsUserAnAdmin():
                    return
            except Exception:
                pass

            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            if ret <= 32:
                try:
                    ctypes.windll.user32.MessageBoxW(
                        None,
                        "[Show Me Your Papers]\n\nElevation failed.\nPlease run this program as Administrator.",
                        "Elevation Error",
                        0x00000010
                    )
                except Exception:
                    print("[Elevation] Failed to elevate and failed to show MessageBox.")
            sys.exit()
    except Exception as e:
        print(f"[Elevation] Failed to elevate: {e}")
        try:
            ctypes.windll.user32.MessageBoxW(
                None,
                "[Show Me Your Papers]\n\nElevation failed.\nPlease run this program as Administrator.",
                "Elevation Error",
                0x00000010
            )
        except Exception:
            pass
        sys.exit()


ensure_admin()


# ---------------------------
# IdentityVault (keys + MAC binding)
# ---------------------------

class IdentityVault:
    def __init__(self, machine_uuid: str, mac: str, salt: bytes):
        self.machine_uuid = machine_uuid
        self.mac = mac
        self.salt = salt
        self._fernet = self._build_fernet()

    def _build_fernet(self) -> Fernet:
        import hashlib, base64
        material = f"{self.machine_uuid}|{self.mac}".encode("utf-8") + self.salt
        key_bytes = hashlib.sha256(material).digest()
        key_b64 = base64.urlsafe_b64encode(key_bytes)
        return Fernet(key_b64)

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)


# ---------------------------
# PersistenceEngine
# ---------------------------

class PersistenceEngine:
    def __init__(self, bootstrap_path: str, vault: IdentityVault):
        self.bootstrap_path = bootstrap_path
        self.vault = vault
        self.state_path: Optional[str] = None
        self.state: Dict[str, Any] = {
            "trusted_sources": {},        # ip/domain -> trust_level
            "process_policies": {},       # proc_name -> {role, policy}
            "blocklist": [],              # list of IPs/domains
            "history": [],                # connection events
            "settings": {},               # misc settings
            "behavior_fingerprints": {},  # proc_name -> baseline stats
            "lineage": {                  # lineage trees
                "process": {},            # proc_name -> list of events
                "domain": {}              # domain/ip -> list of events
            }
        }
        self._dirty_count = 0
        self._flush_threshold = 10
        self._load_bootstrap()
        self.load_state()

    def _load_bootstrap(self):
        if os.path.exists(self.bootstrap_path):
            with open(self.bootstrap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.state_path = data.get("state_path")
        else:
            self.state_path = None

    def set_state_path(self, path: str):
        existing = {}
        if os.path.exists(self.bootstrap_path):
            try:
                with open(self.bootstrap_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing["state_path"] = path
        with open(self.bootstrap_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        self.state_path = path
        self.save_state(force=True)

    def load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "rb") as f:
                enc = f.read()
            raw = self.vault.decrypt(enc)
            self.state = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[Persistence] Failed to load state, starting fresh: {e}")
            self.state = {
                "trusted_sources": {},
                "process_policies": {},
                "blocklist": [],
                "history": [],
                "settings": {},
                "behavior_fingerprints": {},
                "lineage": {"process": {}, "domain": {}}
            }

    def save_state(self, force: bool = False):
        if not self.state_path:
            return
        if not force and self._dirty_count < self._flush_threshold:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            raw = json.dumps(self.state, indent=2).encode("utf-8")
            enc = self.vault.encrypt(raw)
            with open(self.state_path, "wb") as f:
                f.write(enc)
            self._dirty_count = 0
        except Exception as e:
            print(f"[Persistence] Failed to save state: {e}")

    def mark_dirty(self):
        self._dirty_count += 1
        self.save_state(force=False)

    @property
    def trusted_sources(self) -> Dict[str, Any]:
        return self.state.setdefault("trusted_sources", {})

    @property
    def process_policies(self) -> Dict[str, Any]:
        return self.state.setdefault("process_policies", {})

    @property
    def blocklist(self) -> List[str]:
        return self.state.setdefault("blocklist", [])

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self.state.setdefault("history", [])

    @property
    def settings(self) -> Dict[str, Any]:
        return self.state.setdefault("settings", {
            "suspicious_ip_patterns": ["::ffff:", "10.", "192.0.2."],
        })

    @property
    def behavior_fingerprints(self) -> Dict[str, Any]:
        return self.state.setdefault("behavior_fingerprints", {})

    @property
    def lineage(self) -> Dict[str, Any]:
        return self.state.setdefault("lineage", {
            "process": {},
            "domain": {}
        })

    def get_process_policy(self, proc_name: str) -> Dict[str, Any]:
        return self.process_policies.get(proc_name.lower(), {
            "role": "unknown",
            "policy": "inspect"
        })

    def set_process_policy(self, proc_name: str, role: str, policy: str):
        self.process_policies[proc_name.lower()] = {
            "role": role,
            "policy": policy
        }
        self.mark_dirty()

    def get_trust_for_domain(self, domain: str) -> str:
        # trust_level: "trusted", "neutral", "untrusted"
        return self.trusted_sources.get(domain, "neutral")

    def set_trust_for_domain(self, domain: str, level: str):
        self.trusted_sources[domain] = level
        self.mark_dirty()


# ---------------------------
# AnomalyEngine (simple)
# ---------------------------

class AnomalyEngine:
    def __init__(self, window: int = 200):
        self.window = window
        self.ip_events: Dict[str, List[bool]] = {}
        self.proc_events: Dict[str, List[bool]] = {}

    def record_event(self, ip: str, proc: str, violated: bool):
        self._record(self.ip_events, ip, violated)
        self._record(self.proc_events, proc, violated)

    def _record(self, store: Dict[str, List[bool]], key: str, violated: bool):
        lst = store.setdefault(key, [])
        lst.append(violated)
        if len(lst) > self.window:
            del lst[0]

    def score_ip(self, ip: str) -> float:
        return self._score(self.ip_events.get(ip))

    def score_proc(self, proc: str) -> float:
        return self._score(self.proc_events.get(proc))

    @staticmethod
    def _score(events: Optional[List[bool]]) -> float:
        if not events:
            return 0.0
        return sum(1 for v in events if v) / len(events)


# ---------------------------
# ML-style scoring (simple)
# ---------------------------

class MLScorer:
    """
    Simple ML-like scoring:
    - Combines anomaly scores for process and domain.
    - Adds penalty for untrusted domain and suspicious classification.
    - Returns risk score in [0, 1].
    """

    def __init__(self, anomaly: AnomalyEngine, persistence: PersistenceEngine):
        self.anomaly = anomaly
        self.persistence = persistence

    def score(self, proc_name: str, domain: str, classification: str) -> float:
        proc_score = self.anomaly.score_proc(proc_name)
        ip_score = self.anomaly.score_ip(domain)
        trust = self.persistence.get_trust_for_domain(domain)
        trust_penalty = 0.0
        if trust == "trusted":
            trust_penalty = -0.2
        elif trust == "untrusted":
            trust_penalty = 0.2

        class_penalty = 0.0
        if classification == "suspicious":
            class_penalty = 0.4
        elif classification == "pii":
            class_penalty = 0.2

        raw = 0.5 * proc_score + 0.5 * ip_score + trust_penalty + class_penalty
        return max(0.0, min(1.0, raw))


# ---------------------------
# DataShield (PII classification stub)
# ---------------------------

class DataShield:
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def __init__(self, vault: IdentityVault):
        self.vault = vault

    def classify_text(self, text: str) -> str:
        if self.SSN_PATTERN.search(text):
            return "critical"
        if self.EMAIL_PATTERN.search(text):
            return "sensitive"
        if any(token in text.lower() for token in ["fingerprint", "biometric", "social security"]):
            return "critical"
        return "none"

    def transform(self, data: bytes, classification: str, destination_trusted: bool) -> Optional[bytes]:
        if classification == "critical":
            if not destination_trusted:
                return None
            return self.vault.encrypt(data)
        elif classification == "sensitive":
            if not destination_trusted:
                return self.vault.encrypt(data)
            return data
        else:
            return data


# ---------------------------
# Connection model
# ---------------------------

@dataclass
class ConnectionInfo:
    direction: str  # "inbound" or "outbound" or "unknown"
    process_name: str
    pid: int
    local_ip: str
    local_port: int
    remote_ip: str
    remote_port: int


# ---------------------------
# Behavior fingerprints
# ---------------------------

class BehaviorFingerprintEngine:
    """
    Tracks simple per-process behavior baselines:
    - avg connections per scan
    - unique remote count
    - outbound ratio
    """

    def __init__(self, persistence: PersistenceEngine):
        self.persistence = persistence
        self.current_counts: Dict[str, int] = {}
        self.current_remotes: Dict[str, set] = {}
        self.current_outbound: Dict[str, int] = {}
        self.current_total: Dict[str, int] = {}

    def record_connection(self, conn: ConnectionInfo):
        name = conn.process_name
        self.current_counts[name] = self.current_counts.get(name, 0) + 1
        self.current_remotes.setdefault(name, set()).add(conn.remote_ip)
        self.current_total[name] = self.current_total.get(name, 0) + 1
        if conn.direction == "outbound":
            self.current_outbound[name] = self.current_outbound.get(name, 0) + 1

    def end_scan_cycle(self):
        for name, count in self.current_counts.items():
            remotes = len(self.current_remotes.get(name, set()))
            total = self.current_total.get(name, 0)
            outbound = self.current_outbound.get(name, 0)
            outbound_ratio = outbound / total if total > 0 else 0.0

            fp = self.persistence.behavior_fingerprints.get(name, {
                "avg_connections": 0.0,
                "avg_unique_remotes": 0.0,
                "avg_outbound_ratio": 0.0,
                "samples": 0
            })
            n = fp.get("samples", 0)
            fp["avg_connections"] = (fp["avg_connections"] * n + count) / (n + 1)
            fp["avg_unique_remotes"] = (fp["avg_unique_remotes"] * n + remotes) / (n + 1)
            fp["avg_outbound_ratio"] = (fp["avg_outbound_ratio"] * n + outbound_ratio) / (n + 1)
            fp["samples"] = n + 1
            self.persistence.behavior_fingerprints[name] = fp

        self.current_counts.clear()
        self.current_remotes.clear()
        self.current_outbound.clear()
        self.current_total.clear()
        self.persistence.mark_dirty()

    def deviation_score(self, proc_name: str, count: int, unique_remotes: int, outbound_ratio: float) -> float:
        fp = self.persistence.behavior_fingerprints.get(proc_name)
        if not fp or fp.get("samples", 0) < 5:
            return 0.0
        def dev(a, b):
            if b == 0:
                return 0.0
            return abs(a - b) / (b + 1e-6)
        d1 = dev(count, fp["avg_connections"])
        d2 = dev(unique_remotes, fp["avg_unique_remotes"])
        d3 = dev(outbound_ratio, fp["avg_outbound_ratio"])
        raw = (d1 + d2 + d3) / 3.0
        return max(0.0, min(1.0, raw))


# ---------------------------
# GuardCore (policy + classification + lineage + ML)
# ---------------------------

class GuardCore:
    def __init__(self, persistence: PersistenceEngine, data_shield: DataShield,
                 anomaly: AnomalyEngine, ml_scorer: MLScorer, behavior_engine: BehaviorFingerprintEngine):
        self.persistence = persistence
        self.data_shield = data_shield
        self.anomaly = anomaly
        self.ml_scorer = ml_scorer
        self.behavior_engine = behavior_engine
        self.mode = "Enforce"

    def is_trusted_source(self, domain: str) -> bool:
        return self.persistence.get_trust_for_domain(domain) == "trusted"

    def is_blocked(self, domain: str) -> bool:
        return domain in self.persistence.blocklist

    def is_suspicious_ip(self, ip: str) -> bool:
        if ip in self.persistence.blocklist:
            return True
        for pat in self.persistence.settings.get("suspicious_ip_patterns", []):
            if ip.startswith(pat):
                return True
        return False

    def classify_connection(self, conn: ConnectionInfo) -> Dict[str, Any]:
        domain = conn.remote_ip  # user-mode: treat IP as domain key
        proc_policy = self.persistence.get_process_policy(conn.process_name)
        role = proc_policy.get("role", "unknown")
        policy = proc_policy.get("policy", "inspect")

        violated = False
        classification = "clean"
        color = "green"
        reason = "trusted"

        if self.is_blocked(domain) or policy == "block":
            violated = True
            classification = "suspicious"
            color = "red"
            reason = "blocked domain or process policy"
        elif self.is_suspicious_ip(domain):
            violated = True
            classification = "suspicious"
            color = "red"
            reason = "suspicious IP pattern"
        elif policy == "encrypt_all" or role in ["browser", "pii", "identity"]:
            classification = "pii"
            color = "yellow"
            reason = "PII-heavy process or encrypt_all policy"
        else:
            classification = "clean"
            color = "green"
            reason = "no known risk"

        self.anomaly.record_event(domain, conn.process_name, violated)

        risk_score = self.ml_scorer.score(conn.process_name, domain, classification)

        event = {
            "classification": classification,
            "color": color,
            "reason": reason,
            "role": role,
            "policy": policy,
            "domain": domain,
            "risk_score": risk_score
        }
        return event

    def log_event(self, conn: ConnectionInfo, info: Dict[str, Any]):
        entry = {
            "direction": conn.direction,
            "process": conn.process_name,
            "pid": conn.pid,
            "local_ip": conn.local_ip,
            "local_port": conn.local_port,
            "remote_ip": conn.remote_ip,
            "remote_port": conn.remote_port,
            "classification": info["classification"],
            "color": info["color"],
            "reason": info["reason"],
            "role": info["role"],
            "policy": info["policy"],
            "domain": info["domain"],
            "risk_score": info["risk_score"],
            "timestamp": time.time()
        }
        self.persistence.history.append(entry)
        if len(self.persistence.history) > 2000:
            del self.persistence.history[0]

        # lineage: process
        proc_line = self.persistence.lineage.setdefault("process", {})
        proc_events = proc_line.setdefault(conn.process_name, [])
        proc_events.append(entry)
        if len(proc_events) > 200:
            del proc_events[0]

        # lineage: domain
        dom_line = self.persistence.lineage.setdefault("domain", {})
        dom_events = dom_line.setdefault(info["domain"], [])
        dom_events.append(entry)
        if len(dom_events) > 200:
            del dom_events[0]

        self.persistence.mark_dirty()


# ---------------------------
# ConnectionScanner (psutil-based)
# ---------------------------

class ConnectionScanner:
    def __init__(self, guard: GuardCore, interval: float = 2.0, behavior_engine: BehaviorFingerprintEngine = None):
        self.guard = guard
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.seen: Dict[Tuple[str, int, str, int, str], float] = {}
        self.behavior_engine = behavior_engine
        self._cycle_start = time.time()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                self._scan_once()
            except Exception as e:
                print(f"[Scanner] Error: {e}")
            time.sleep(self.interval)
            if self.behavior_engine and time.time() - self._cycle_start >= self.interval:
                self.behavior_engine.end_scan_cycle()
                self._cycle_start = time.time()

    def _scan_once(self):
        conns = psutil.net_connections(kind="inet")
        now = time.time()

        for c in conns:
            if not c.laddr or not c.raddr:
                continue
            local_ip = getattr(c.laddr, "ip", None) or c.laddr[0]
            local_port = getattr(c.laddr, "port", None) or c.laddr[1]
            remote_ip = getattr(c.raddr, "ip", None) or c.raddr[0]
            remote_port = getattr(c.raddr, "port", None) or c.raddr[1]

            direction = "outbound" if local_port > 1024 else "inbound"

            key = (local_ip, local_port, remote_ip, remote_port, direction)
            if key in self.seen and now - self.seen[key] < self.interval:
                continue
            self.seen[key] = now

            pid = c.pid or -1
            try:
                p = psutil.Process(pid) if pid > 0 else None
                proc_name = p.name() if p else "unknown"
            except Exception:
                proc_name = "unknown"

            conn = ConnectionInfo(
                direction=direction,
                process_name=proc_name,
                pid=pid,
                local_ip=local_ip,
                local_port=local_port,
                remote_ip=remote_ip,
                remote_port=remote_port
            )

            if self.behavior_engine:
                self.behavior_engine.record_connection(conn)

            info = self.guard.classify_connection(conn)
            self.guard.log_event(conn, info)


# ---------------------------
# Admin helper
# ---------------------------

def is_admin_now() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------
# PyQt5 Cockpit
# ---------------------------

class MainWindow(QMainWindow):
    def __init__(self, guard: GuardCore, persistence: PersistenceEngine, scanner: ConnectionScanner,
                 anomaly: AnomalyEngine, behavior_engine: BehaviorFingerprintEngine):
        super().__init__()
        self.guard = guard
        self.persistence = persistence
        self.scanner = scanner
        self.anomaly = anomaly
        self.behavior_engine = behavior_engine

        self.setWindowTitle("Show Me Your Papers - Connection Cockpit")
        self.resize(1400, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_controls_tab()
        self._init_live_tab()
        self._init_events_tab()
        self._init_processes_tab()
        self._init_settings_tab()
        self._init_process_reputation_tab()
        self._init_domain_reputation_tab()
        self._init_heatmap_tab()
        self._init_behavior_graph_tab()
        self._init_lineage_tab()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_all_views)
        self.timer.start(2000)

    # ---------- Tabs ----------

    def _init_controls_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Learning", "Enforce", "Nuclear"])
        self.mode_combo.setCurrentText(self.guard.mode)
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        admin_layout = QHBoxLayout()
        admin_label_title = QLabel("Running as Administrator:")
        self.admin_label = QLabel("Yes" if is_admin_now() else "No")
        admin_layout.addWidget(admin_label_title)
        admin_layout.addWidget(self.admin_label)
        layout.addLayout(admin_layout)

        path_btn = QPushButton("Set State Storage Path")
        path_btn.clicked.connect(self._choose_state_path)
        layout.addWidget(path_btn)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Controls")

    def _init_live_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.live_table = QTableWidget(0, 9)
        self.live_table.setHorizontalHeaderLabels([
            "Direction", "Process", "PID",
            "Local", "Remote",
            "Classification", "Role", "Policy", "Reason"
        ])
        layout.addWidget(self.live_table)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Live Connections")

    def _init_events_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.events_table = QTableWidget(0, 9)
        self.events_table.setHorizontalHeaderLabels([
            "Direction", "Process", "PID",
            "Local", "Remote",
            "Classification", "Role", "Policy", "Reason"
        ])
        layout.addWidget(self.events_table)

        self.alerts_box = QTextEdit()
        self.alerts_box.setReadOnly(True)
        layout.addWidget(QLabel("Real-time Alerts"))
        layout.addWidget(self.alerts_box)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Events & Alerts")

    def _init_processes_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.proc_table = QTableWidget(0, 4)
        self.proc_table.setHorizontalHeaderLabels([
            "Process", "Role", "Policy", "Save"
        ])
        layout.addWidget(self.proc_table)

        refresh_btn = QPushButton("Refresh Processes")
        refresh_btn.clicked.connect(self._refresh_process_table)
        layout.addWidget(refresh_btn)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Processes")

        self._refresh_process_table()

    def _init_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        form = QFormLayout()

        self.suspicious_edit = QTextEdit()
        self.suspicious_edit.setPlainText(
            json.dumps(self.persistence.settings.get("suspicious_ip_patterns", []), indent=2)
        )
        form.addRow("Suspicious IP patterns (JSON list, prefix match):", self.suspicious_edit)

        layout.addLayout(form)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Settings")

    def _init_process_reputation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.proc_rep_table = QTableWidget(0, 5)
        self.proc_rep_table.setHorizontalHeaderLabels([
            "Process", "Risk Score", "Anomaly Score", "Samples", "Role"
        ])
        layout.addWidget(self.proc_rep_table)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Process Reputation")

    def _init_domain_reputation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.domain_rep_table = QTableWidget(0, 5)
        self.domain_rep_table.setHorizontalHeaderLabels([
            "Domain/IP", "Risk Score", "Anomaly Score", "Trust", "Events"
        ])
        layout.addWidget(self.domain_rep_table)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Domain Reputation")

    def _init_heatmap_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.heatmap_table = QTableWidget(0, 4)
        self.heatmap_table.setHorizontalHeaderLabels([
            "Process", "Domain/IP", "Events", "Risk"
        ])
        layout.addWidget(self.heatmap_table)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Threat Heatmap")

    def _init_behavior_graph_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.behavior_text = QTextEdit()
        self.behavior_text.setReadOnly(True)
        layout.addWidget(QLabel("Process Behavior Graph (textual summary)"))
        layout.addWidget(self.behavior_text)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Behavior Graph")

    def _init_lineage_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.lineage_text = QTextEdit()
        self.lineage_text.setReadOnly(True)
        layout.addWidget(QLabel("Threat Lineage (per process & domain)"))
        layout.addWidget(self.lineage_text)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Lineage")

    # ---------- Controls ----------

    def _mode_changed(self, mode: str):
        self.guard.mode = mode

    def _choose_state_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Choose State File", "", "Encrypted State (*.bin)")
        if path:
            self.persistence.set_state_path(path)

    def _save_settings(self):
        try:
            patterns = json.loads(self.suspicious_edit.toPlainText())
            if isinstance(patterns, list):
                self.persistence.settings["suspicious_ip_patterns"] = patterns
            else:
                print("[Settings] suspicious_ip_patterns must be a list.")
        except Exception as e:
            print(f"[Settings] Failed to parse suspicious_ip_patterns JSON: {e}")
        self.persistence.mark_dirty()
        print("[Settings] Saved.")

    # ---------- Processes tab ----------

    def _refresh_process_table(self):
        procs = set(e["process"] for e in self.persistence.history if e.get("process"))
        procs = sorted(procs)

        self.proc_table.setRowCount(len(procs))

        for row, name in enumerate(procs):
            policy = self.persistence.get_process_policy(name)
            role = policy.get("role", "unknown")
            pol = policy.get("policy", "inspect")

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.proc_table.setItem(row, 0, name_item)

            role_edit = QLineEdit(role)
            self.proc_table.setCellWidget(row, 1, role_edit)

            policy_combo = QComboBox()
            policy_combo.addItems(["allow", "inspect", "block", "encrypt_all"])
            idx = policy_combo.findText(pol)
            if idx >= 0:
                policy_combo.setCurrentIndex(idx)
            self.proc_table.setCellWidget(row, 2, policy_combo)

            save_btn = QPushButton("Save")
            save_btn.clicked.connect(lambda _, r=row: self._save_process_row(r))
            self.proc_table.setCellWidget(row, 3, save_btn)

    def _save_process_row(self, row: int):
        name_item = self.proc_table.item(row, 0)
        if not name_item:
            return
        proc_name = name_item.text()

        role_widget = self.proc_table.cellWidget(row, 1)
        policy_widget = self.proc_table.cellWidget(row, 2)

        role = role_widget.text().strip() if isinstance(role_widget, QLineEdit) else "unknown"
        policy = policy_widget.currentText() if isinstance(policy_widget, QComboBox) else "inspect"

        self.persistence.set_process_policy(proc_name, role, policy)
        print(f"[Processes] Saved policy for {proc_name}: role={role}, policy={policy}")

    # ---------- Tables & views ----------

    def refresh_all_views(self):
        self.refresh_tables()
        self.refresh_process_reputation()
        self.refresh_domain_reputation()
        self.refresh_heatmap()
        self.refresh_behavior_graph()
        self.refresh_lineage()
        self.refresh_alerts()

    def refresh_tables(self):
        history = self.persistence.history[-300:]

        self.live_table.setRowCount(len(history))
        self.events_table.setRowCount(len(history))

        for row, entry in enumerate(history):
            self._set_row(self.live_table, row, entry)
            self._set_row(self.events_table, row, entry)

    def _set_row(self, table: QTableWidget, row: int, entry: Dict[str, Any]):
        table.setItem(row, 0, QTableWidgetItem(entry["direction"]))
        table.setItem(row, 1, QTableWidgetItem(entry["process"]))
        table.setItem(row, 2, QTableWidgetItem(str(entry["pid"])))
        table.setItem(row, 3, QTableWidgetItem(f'{entry["local_ip"]}:{entry["local_port"]}'))
        table.setItem(row, 4, QTableWidgetItem(f'{entry["remote_ip"]}:{entry["remote_port"]}'))
        table.setItem(row, 5, QTableWidgetItem(entry["classification"]))
        table.setItem(row, 6, QTableWidgetItem(entry.get("role", "")))
        table.setItem(row, 7, QTableWidgetItem(entry.get("policy", "")))
        table.setItem(row, 8, QTableWidgetItem(entry.get("reason", "")))

        color = entry.get("color", "green")
        if color == "red":
            bg = QColor(255, 100, 100)
        elif color == "yellow":
            bg = QColor(255, 255, 150)
        elif color == "green":
            bg = QColor(180, 255, 180)
        else:
            bg = QColor(230, 230, 230)

        for col in range(0, 9):
            item = table.item(row, col)
            if item:
                item.setBackground(bg)

    # ---------- Alerts ----------

    def refresh_alerts(self):
        history = self.persistence.history[-100:]
        lines = []
        for e in history:
            if e.get("classification") == "suspicious" or e.get("risk_score", 0) >= 0.7:
                lines.append(
                    f"[ALERT] {e['process']} ({e['pid']}) -> {e['remote_ip']} "
                    f"class={e['classification']} risk={e.get('risk_score', 0):.2f} reason={e.get('reason', '')}"
                )
        self.alerts_box.setPlainText("\n".join(lines))

    # ---------- Process Reputation ----------

    def refresh_process_reputation(self):
        # aggregate per process
        stats: Dict[str, Dict[str, Any]] = {}
        for e in self.persistence.history:
            name = e["process"]
            s = stats.setdefault(name, {"risk_sum": 0.0, "count": 0, "role": e.get("role", "unknown")})
            s["risk_sum"] += e.get("risk_score", 0.0)
            s["count"] += 1

        procs = sorted(stats.items(), key=lambda kv: (kv[1]["risk_sum"] / max(1, kv[1]["count"])), reverse=True)
        self.proc_rep_table.setRowCount(len(procs))

        for row, (name, s) in enumerate(procs):
            avg_risk = s["risk_sum"] / max(1, s["count"])
            anomaly_score = self.anomaly.score_proc(name)
            fp = self.persistence.behavior_fingerprints.get(name, {})
            samples = fp.get("samples", 0)

            self.proc_rep_table.setItem(row, 0, QTableWidgetItem(name))
            self.proc_rep_table.setItem(row, 1, QTableWidgetItem(f"{avg_risk:.2f}"))
            self.proc_rep_table.setItem(row, 2, QTableWidgetItem(f"{anomaly_score:.2f}"))
            self.proc_rep_table.setItem(row, 3, QTableWidgetItem(str(samples)))
            self.proc_rep_table.setItem(row, 4, QTableWidgetItem(s.get("role", "unknown")))

            bg = self._risk_color(avg_risk)
            for col in range(5):
                item = self.proc_rep_table.item(row, col)
                if item:
                    item.setBackground(bg)

    # ---------- Domain Reputation ----------

    def refresh_domain_reputation(self):
        stats: Dict[str, Dict[str, Any]] = {}
        for e in self.persistence.history:
            dom = e.get("domain", e["remote_ip"])
            s = stats.setdefault(dom, {"risk_sum": 0.0, "count": 0})
            s["risk_sum"] += e.get("risk_score", 0.0)
            s["count"] += 1

        domains = sorted(stats.items(), key=lambda kv: (kv[1]["risk_sum"] / max(1, kv[1]["count"])), reverse=True)
        self.domain_rep_table.setRowCount(len(domains))

        for row, (dom, s) in enumerate(domains):
            avg_risk = s["risk_sum"] / max(1, s["count"])
            anomaly_score = self.anomaly.score_ip(dom)
            trust = self.persistence.get_trust_for_domain(dom)

            self.domain_rep_table.setItem(row, 0, QTableWidgetItem(dom))
            self.domain_rep_table.setItem(row, 1, QTableWidgetItem(f"{avg_risk:.2f}"))
            self.domain_rep_table.setItem(row, 2, QTableWidgetItem(f"{anomaly_score:.2f}"))
            self.domain_rep_table.setItem(row, 3, QTableWidgetItem(trust))
            self.domain_rep_table.setItem(row, 4, QTableWidgetItem(str(s["count"])))

            bg = self._risk_color(avg_risk)
            for col in range(5):
                item = self.domain_rep_table.item(row, col)
                if item:
                    item.setBackground(bg)

    # ---------- Heatmap ----------

    def refresh_heatmap(self):
        # process-domain pairs
        stats: List[Tuple[str, str, int, float]] = []
        pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for e in self.persistence.history:
            proc = e["process"]
            dom = e.get("domain", e["remote_ip"])
            key = (proc, dom)
            s = pair_map.setdefault(key, {"count": 0, "risk_sum": 0.0})
            s["count"] += 1
            s["risk_sum"] += e.get("risk_score", 0.0)

        for (proc, dom), s in pair_map.items():
            avg_risk = s["risk_sum"] / max(1, s["count"])
            stats.append((proc, dom, s["count"], avg_risk))

        stats.sort(key=lambda x: x[3], reverse=True)
        self.heatmap_table.setRowCount(len(stats))

        for row, (proc, dom, count, risk) in enumerate(stats):
            self.heatmap_table.setItem(row, 0, QTableWidgetItem(proc))
            self.heatmap_table.setItem(row, 1, QTableWidgetItem(dom))
            self.heatmap_table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.heatmap_table.setItem(row, 3, QTableWidgetItem(f"{risk:.2f}"))

            bg = self._risk_color(risk)
            for col in range(4):
                item = self.heatmap_table.item(row, col)
                if item:
                    item.setBackground(bg)

    # ---------- Behavior Graph ----------

    def refresh_behavior_graph(self):
        lines = []
        for name, fp in self.persistence.behavior_fingerprints.items():
            lines.append(
                f"{name}: avg_conn={fp.get('avg_connections', 0):.1f}, "
                f"avg_unique={fp.get('avg_unique_remotes', 0):.1f}, "
                f"avg_outbound_ratio={fp.get('avg_outbound_ratio', 0):.2f}, "
                f"samples={fp.get('samples', 0)}"
            )
        self.behavior_text.setPlainText("\n".join(lines))

    # ---------- Lineage ----------

    def refresh_lineage(self):
        lines = []
        proc_line = self.persistence.lineage.get("process", {})
        dom_line = self.persistence.lineage.get("domain", {})

        lines.append("=== Process Lineage ===")
        for proc, events in proc_line.items():
            lines.append(f"{proc}:")
            for e in events[-5:]:
                lines.append(
                    f"  -> {e['remote_ip']} class={e['classification']} "
                    f"risk={e.get('risk_score', 0):.2f} reason={e.get('reason', '')}"
                )

        lines.append("\n=== Domain Lineage ===")
        for dom, events in dom_line.items():
            lines.append(f"{dom}:")
            for e in events[-5:]:
                lines.append(
                    f"  <- {e['process']}({e['pid']}) class={e['classification']} "
                    f"risk={e.get('risk_score', 0):.2f}"
                )

        self.lineage_text.setPlainText("\n".join(lines))

    # ---------- Helpers ----------

    def _risk_color(self, risk: float) -> QColor:
        if risk >= 0.8:
            return QColor(255, 80, 80)
        elif risk >= 0.5:
            return QColor(255, 180, 80)
        elif risk >= 0.2:
            return QColor(220, 255, 180)
        else:
            return QColor(200, 255, 200)


# ---------------------------
# Bootstrap helpers
# ---------------------------

def get_mac() -> str:
    mac = uuid.getnode()
    return ":".join(f"{(mac >> ele) & 0xff:02x}" for ele in range(40, -8, -8))


def load_or_create_machine_uuid(bootstrap_path: str) -> str:
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "machine_uuid" in data:
                    return data["machine_uuid"]
        except Exception:
            pass
    mu = str(uuid.uuid4())
    existing = {}
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing["machine_uuid"] = mu
    with open(bootstrap_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return mu


def load_or_create_salt(bootstrap_path: str) -> bytes:
    import base64, os as _os
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "salt" in data:
                    return base64.b64decode(data["salt"])
        except Exception:
            pass
    salt = _os.urandom(16)
    existing = {}
    if os.path.exists(bootstrap_path):
        try:
            with open(bootstrap_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    import base64 as _b64
    existing["salt"] = _b64.b64encode(salt).decode("ascii")
    with open(bootstrap_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return salt


# ---------------------------
# Main
# ---------------------------

def main():
    bootstrap_path = os.path.expanduser("~/.smy_papers_bootstrap_scan.json")

    machine_uuid = load_or_create_machine_uuid(bootstrap_path)
    mac = get_mac()
    salt = load_or_create_salt(bootstrap_path)

    vault = IdentityVault(machine_uuid, mac, salt)
    persistence = PersistenceEngine(bootstrap_path, vault)
    anomaly = AnomalyEngine()
    data_shield = DataShield(vault)
    behavior_engine = BehaviorFingerprintEngine(persistence)
    ml_scorer = MLScorer(anomaly, persistence)
    guard = GuardCore(persistence, data_shield, anomaly, ml_scorer, behavior_engine)

    scanner = ConnectionScanner(guard, interval=2.0, behavior_engine=behavior_engine)
    scanner.start()

    app = QApplication([])
    window = MainWindow(guard, persistence, scanner, anomaly, behavior_engine)
    window.show()
    app.exec_()

    scanner.stop()
    persistence.save_state(force=True)


if __name__ == "__main__":
    main()

