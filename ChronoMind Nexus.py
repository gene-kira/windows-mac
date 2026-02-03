import sys
import os
import time
import json
import threading
import socket
import traceback

import yaml
import win32gui
import win32process
import psutil
from pynput import mouse, keyboard
import requests

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QPushButton, QLineEdit, QComboBox,
    QTabWidget, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QColor, QFont

# =========================
# GLOBALS / CONFIG
# =========================

TCP_HOST = "127.0.0.1"
TCP_PORT = 9009

CONFIG_PATH = "config.yaml"
THEMES_PATH = "themes.yaml"
MAX_TIMELINE_ROWS = 600

TIME_CHECK_INTERVAL_SEC = 60          # how often to check reference time
DRIFT_ALERT_THRESHOLD_SEC = 5         # drift above this triggers an event
DRIFT_SUSPICIOUS_THRESHOLD_SEC = 60   # big jumps get higher risk

DEFAULT_CONFIG = {
    "mode": "strict",
    "theme": "dark",
    "firewall": {
        "block_patterns": [
            "ignore\\s+all\\s+previous",
            "system\\s*:",
            "assistant\\s*:",
            "<script",
            "data:",
            "base64",
        ],
    },
}

DEFAULT_THEMES = {
    "dark": {
        "background": "#111111",
        "text": "#e0e0e0",
        "accent": "#00aaff",
        "border": "#333333",
    },
    "neon": {
        "background": "#000000",
        "text": "#ffffff",
        "accent": "#00ffcc",
        "border": "#00ffcc",
    },
}

# =========================
# CONFIG + THEMES
# =========================

def deep_merge(defaults, data):
    if not isinstance(defaults, dict):
        return data if data is not None else defaults
    result = dict(defaults)
    if isinstance(data, dict):
        for k, v in data.items():
            if k in result and isinstance(result[k], dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
    return result

class Config:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.data = {}
        self.last_mtime = 0
        self.load_or_init()

    def load_or_init(self):
        if not os.path.exists(self.path):
            self.data = DEFAULT_CONFIG.copy()
            self.save()
        else:
            self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self.data = deep_merge(DEFAULT_CONFIG, raw)
            self.last_mtime = os.path.getmtime(self.path)
        except Exception:
            traceback.print_exc()
            self.data = DEFAULT_CONFIG.copy()

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.dump(self.data, f)
            self.last_mtime = os.path.getmtime(self.path)
        except Exception:
            traceback.print_exc()

    def get(self, key, default=None):
        parts = key.split(".")
        node = self.data
        for p in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(p, None)
            if node is None:
                return default
        return node

config = Config()

class ThemeManager:
    def __init__(self, config, path=THEMES_PATH):
        self.config = config
        self.path = path
        self.themes = {}
        self.load_or_init()

    def load_or_init(self):
        if not os.path.exists(self.path):
            self.themes = DEFAULT_THEMES
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.dump({"themes": self.themes}, f)
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.themes = data.get("themes", DEFAULT_THEMES)
            except Exception:
                traceback.print_exc()
                self.themes = DEFAULT_THEMES

    def apply(self, app):
        theme_name = self.config.get("theme", "dark")
        theme = self.themes.get(theme_name, self.themes["dark"])
        app.setStyleSheet(f"""
            QWidget {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-size: 11px;
            }}
            QGroupBox {{
                border: 1px solid {theme['border']};
                margin-top: 4px;
                padding: 4px;
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: {theme['background']};
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
            }}
            QTableWidget {{
                gridline-color: {theme['border']};
                font-size: 10px;
            }}
            QHeaderView::section {{
                background-color: {theme['border']};
                color: {theme['text']};
                padding: 2px;
                font-size: 10px;
            }}
            QTabBar::tab {{
                padding: 3px 6px;
                font-size: 10px;
            }}
        """)

class ThemeSwitcher(QWidget):
    def __init__(self, config, theme_manager, app):
        super().__init__()
        self.config = config
        self.theme_manager = theme_manager
        self.app = app
        self.combo = QComboBox()
        self.combo.addItems(sorted(self.theme_manager.themes.keys()))
        self.combo.setCurrentText(config.get("theme", "dark"))
        self.combo.currentTextChanged.connect(self.change_theme)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Theme:"))
        layout.addWidget(self.combo)
        self.setLayout(layout)

    def change_theme(self, theme):
        self.config.data["theme"] = theme
        self.config.save()
        self.theme_manager.apply(self.app)

# =========================
# TCP EVENT BUS (SERVER)
# =========================

clients_lock = threading.Lock()
clients = []  # list of (socket, file-like)

def broadcast_event(event):
    data = json.dumps(event, ensure_ascii=False) + "\n"
    dead = []
    with clients_lock:
        for s, f in clients:
            try:
                f.write(data)
                f.flush()
            except Exception:
                dead.append((s, f))
        for d in dead:
            try:
                d[0].close()
            except Exception:
                pass
            if d in clients:
                clients.remove(d)

def handle_bus_client(sock, addr):
    print(f"[BUS] Client connected from {addr}")
    f_in = sock.makefile("r", encoding="utf-8")
    f_out = sock.makefile("w", encoding="utf-8")
    with clients_lock:
        clients.append((sock, f_out))
    try:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                traceback.print_exc()
                continue
            broadcast_event(event)
    except Exception:
        traceback.print_exc()
    finally:
        print(f"[BUS] Client disconnected {addr}")
        with clients_lock:
            for c in list(clients):
                if c[0] is sock:
                    clients.remove(c)
        try:
            sock.close()
        except Exception:
            pass

def bus_server_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((TCP_HOST, TCP_PORT))
    srv.listen(10)
    print(f"[BUS] Listening on {TCP_HOST}:{TCP_PORT}")
    while True:
        try:
            sock, addr = srv.accept()
            t = threading.Thread(target=handle_bus_client, args=(sock, addr), daemon=True)
            t.start()
        except Exception:
            traceback.print_exc()
            time.sleep(1)

# =========================
# GUI EVENT BUS (CLIENT SIDE)
# =========================

class EventBus(QObject):
    event_received = Signal(dict)

bus = EventBus()

def tcp_client_loop():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TCP_HOST, TCP_PORT))
            f = sock.makefile("r", encoding="utf-8")
            print("[CLIENT] Connected to bus")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    bus.event_received.emit(event)
                except Exception:
                    traceback.print_exc()
            f.close()
            sock.close()
        except Exception:
            print("[CLIENT] Connection failed, retrying in 2s...")
            time.sleep(2)

# =========================
# GUI WIDGETS
# =========================

class ModeSwitcher(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.combo = QComboBox()
        self.combo.addItems(["strict", "normal", "learning"])
        self.combo.setCurrentText(config.get("mode", "strict"))
        self.combo.currentTextChanged.connect(self.change_mode)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.combo)
        self.setLayout(layout)

    def change_mode(self, mode):
        self.config.data["mode"] = mode
        self.config.save()

class RuleEditor(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patterns = list(config.get("firewall.block_patterns", []))
        self.table = QTableWidget(len(self.patterns), 3)
        self.table.setHorizontalHeaderLabels(["Pattern", "Valid", "Remove"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(18)
        for i, p in enumerate(self.patterns):
            self._set_row(i, p)
        self.add_input = QLineEdit()
        self.add_btn = QPushButton("Add Rule")
        self.add_btn.clicked.connect(self.add_rule)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        layout.addWidget(self.table)
        layout.addWidget(self.add_input)
        layout.addWidget(self.add_btn)
        self.setLayout(layout)

    def _validate_pattern(self, pattern):
        import re
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def _set_row(self, row, pattern):
        self.table.setItem(row, 0, QTableWidgetItem(pattern))
        valid = self._validate_pattern(pattern)
        valid_item = QTableWidgetItem("✔" if valid else "✖")
        valid_item.setForeground(QColor("#00ff88" if valid else "#ff4444"))
        self.table.setItem(row, 1, valid_item)
        btn = QPushButton("X")
        btn.setMaximumWidth(30)
        btn.clicked.connect(lambda _, r=row: self.remove_rule(r))
        self.table.setCellWidget(row, 2, btn)

    def add_rule(self):
        rule = self.add_input.text().strip()
        if not rule:
            return
        if not self._validate_pattern(rule):
            QMessageBox.warning(self, "Invalid Regex", "The pattern is not a valid regular expression.")
            return
        self.patterns.append(rule)
        self.save()
        self.refresh()
        self.add_input.clear()

    def remove_rule(self, row):
        if 0 <= row < len(self.patterns):
            del self.patterns[row]
            self.save()
            self.refresh()

    def save(self):
        self.config.data.setdefault("firewall", {})
        self.config.data["firewall"]["block_patterns"] = self.patterns
        self.config.save()

    def refresh(self):
        self.table.setRowCount(len(self.patterns))
        for i, p in enumerate(self.patterns):
            self._set_row(i, p)

class DomainHeatmap(QWidget):
    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Domain", "Total", "Blocked", "Risk"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(18)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.domains = {}

    def update_with_event(self, event):
        domain = event.get("domain", "")
        if not domain:
            return
        d = self.domains.setdefault(domain, {"total": 0, "blocked": 0, "risk": []})
        d["total"] += 1
        if event.get("decision") == "block":
            d["blocked"] += 1
        d["risk"].append(event.get("risk", 0))
        self.refresh()

    def refresh(self):
        self.table.setRowCount(0)
        for domain, stats in self.domains.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            avg_risk = sum(stats["risk"]) / len(stats["risk"]) if stats["risk"] else 0
            color = "#00aa55"
            if avg_risk >= 5:
                color = "#ff4444"
            elif avg_risk >= 2:
                color = "#ffaa00"
            item = QTableWidgetItem(domain)
            item.setBackground(QColor(color))
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem(str(stats["total"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(stats["blocked"])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{avg_risk:.2f}"))

class TimeHealthPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.label_status = QLabel("No time data yet")
        self.label_drift = QLabel("Drift: n/a")
        self.label_ref = QLabel("Reference: n/a")
        self.label_sys = QLabel("System: n/a")

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        layout.addWidget(self.label_status)
        layout.addWidget(self.label_drift)
        layout.addWidget(self.label_ref)
        layout.addWidget(self.label_sys)
        self.setLayout(layout)

    def update_from_event(self, event: dict):
        meta = event.get("meta", {})
        drift = meta.get("drift_seconds", 0.0)
        ref_ts = meta.get("reference_timestamp", None)
        sys_ts = meta.get("system_timestamp", None)
        risk = event.get("risk", 0)

        status = "OK"
        if risk >= 7:
            status = "HIGH RISK time manipulation"
        elif risk >= 3:
            status = "Suspicious drift"

        self.label_status.setText(f"Status: {status} (risk {risk})")
        self.label_drift.setText(f"Drift: {drift:+.2f} seconds")

        if ref_ts is not None:
            self.label_ref.setText(f"Reference: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ref_ts))}")
        else:
            self.label_ref.setText("Reference: unavailable")

        if sys_ts is not None:
            self.label_sys.setText(f"System:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sys_ts))}")
        else:
            self.label_sys.setText("System:    n/a")

    def show_reference_unreachable(self):
        self.label_status.setText("Status: Reference unreachable")
        self.label_drift.setText("Drift: n/a")
        self.label_ref.setText("Reference: unreachable")
        self.label_sys.setText(f"System:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

class Cockpit(QWidget):
    def __init__(self, app, config, theme_manager):
        super().__init__()
        self.app = app
        self.config = config
        self.theme_manager = theme_manager
        self.setWindowTitle("Click Intelligence Cockpit (All-in-One + Time Sentinel)")
        self.resize(1100, 650)
        self.setContentsMargins(4, 4, 4, 4)

        self.mode_label = QLabel("")
        self.last_event_label = QLabel("No events yet")
        self.last_event_label.setWordWrap(True)

        self.overlay_raw = QLabel("")
        self.overlay_clean = QLabel("")
        self.overlay_flags = QLabel("")

        small_font = QFont()
        small_font.setPointSize(9)
        self.overlay_clean.setFont(small_font)
        self.overlay_raw.setFont(small_font)
        self.overlay_flags.setFont(small_font)

        self.timeline = QTableWidget(0, 6)
        self.timeline.setHorizontalHeaderLabels(
            ["Time", "Source", "Domain", "Text", "Decision", "Risk"]
        )
        self.timeline.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.timeline.verticalHeader().setDefaultSectionSize(18)

        self.domain_heatmap = DomainHeatmap()
        self.rule_editor = RuleEditor(config)
        self.mode_switcher = ModeSwitcher(config)
        self.theme_switcher = ThemeSwitcher(config, theme_manager, app)
        self.time_panel = TimeHealthPanel()

        self._build_layout()
        self.apply_config(config.data)

        bus.event_received.connect(self.handle_event)

    def _build_layout(self):
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(6)
        top_bar.addWidget(self.mode_label)
        top_bar.addWidget(self.last_event_label, stretch=1)
        top_bar.addWidget(self.mode_switcher)
        top_bar.addWidget(self.theme_switcher)
        root.addLayout(top_bar)

        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(6)

        left_box = QGroupBox("Real-time Overlay")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        left_layout.addWidget(QLabel("Sanitized:"))
        left_layout.addWidget(self.overlay_clean)
        left_layout.addWidget(QLabel("Raw:"))
        left_layout.addWidget(self.overlay_raw)
        left_layout.addWidget(QLabel("Flags:"))
        left_layout.addWidget(self.overlay_flags)
        left_box.setLayout(left_layout)

        center_box = QGroupBox("Timeline")
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(4, 4, 4, 4)
        center_layout.setSpacing(4)
        center_layout.addWidget(self.timeline)
        center_box.setLayout(center_layout)

        right_tabs = QTabWidget()
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout()
        heatmap_layout.setContentsMargins(2, 2, 2, 2)
        heatmap_layout.setSpacing(4)
        heatmap_layout.addWidget(self.domain_heatmap)
        heatmap_tab.setLayout(heatmap_layout)

        rules_tab = QWidget()
        rules_layout = QVBoxLayout()
        rules_layout.setContentsMargins(2, 2, 2, 2)
        rules_layout.setSpacing(4)
        rules_layout.addWidget(self.rule_editor)
        rules_tab.setLayout(rules_layout)

        time_tab = QWidget()
        time_layout = QVBoxLayout()
        time_layout.setContentsMargins(2, 2, 2, 2)
        time_layout.setSpacing(4)
        time_layout.addWidget(self.time_panel)
        time_tab.setLayout(time_layout)

        right_tabs.addTab(heatmap_tab, "Domains")
        right_tabs.addTab(rules_tab, "Rules")
        right_tabs.addTab(time_tab, "Time")

        main.addWidget(left_box, 2)
        main.addWidget(center_box, 4)
        main.addWidget(right_tabs, 3)

        root.addLayout(main)
        self.setLayout(root)

    def apply_config(self, data):
        mode = self.config.get("mode", "strict")
        self.mode_label.setText(f"Mode: {mode.capitalize()}")

    def _prune_timeline(self):
        rows = self.timeline.rowCount()
        if rows > MAX_TIMELINE_ROWS:
            excess = rows - MAX_TIMELINE_ROWS
            for _ in range(excess):
                self.timeline.removeRow(0)

    def handle_event(self, event):
        try:
            clean = event.get("clean", "")
            raw = event.get("raw", "")
            decision = event.get("decision", "allow")
            risk = event.get("risk", 0)
            domain = event.get("domain", "")
            source_name = event.get("source", {}).get("name", "")
            ts = event.get("timestamp", time.time())
            meta = event.get("meta", {})
            event_type = meta.get("type", "")

            if event_type == "time_change":
                self.time_panel.update_from_event(event)

            if event_type == "time_reference_unreachable":
                self.time_panel.show_reference_unreachable()

            flags = []
            if decision == "block":
                flags.append("BLOCKED")
            elif decision == "flag":
                flags.append("FLAGGED")
            else:
                flags.append("ALLOWED")

            self.overlay_clean.setText(clean)
            self.overlay_raw.setText(raw)
            self.overlay_flags.setText(", ".join(flags))
            self.last_event_label.setText(f"Last: {flags[-1]} | {clean[:70]}")

            row = self.timeline.rowCount()
            self.timeline.insertRow(row)
            self.timeline.setItem(row, 0, QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))))
            self.timeline.setItem(row, 1, QTableWidgetItem(source_name))
            self.timeline.setItem(row, 2, QTableWidgetItem(domain))
            self.timeline.setItem(row, 3, QTableWidgetItem(clean[:120]))
            self.timeline.setItem(row, 4, QTableWidgetItem(decision.upper()))
            self.timeline.setItem(row, 5, QTableWidgetItem(str(risk)))

            self._prune_timeline()
            self.domain_heatmap.update_with_event(event)
        except Exception:
            traceback.print_exc()

# =========================
# PRODUCER: WIN32 HOOKS + TIME SENTINEL
# =========================

class EventProducer:
    def __init__(self, host=TCP_HOST, port=TCP_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.file = None

    def connect(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.file = self.sock.makefile("w", encoding="utf-8")
                print(f"[PRODUCER] Connected to {self.host}:{self.port}")
                return
            except Exception:
                print("[PRODUCER] Connect failed, retrying in 2s...")
                time.sleep(2)

    def send_event(self, event: dict):
        if self.file is None:
            self.connect()
        try:
            line = json.dumps(event, ensure_ascii=False) + "\n"
            self.file.write(line)
            self.file.flush()
        except Exception:
            traceback.print_exc()
            try:
                if self.file:
                    self.file.close()
                if self.sock:
                    self.sock.close()
            except:
                pass
            self.file = None
            self.sock = None
            self.connect()

producer = EventProducer()

def get_active_window_info():
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        pname = process.name()
        return title, pname
    except Exception:
        return "", ""

def build_event(event_type, raw_text, source_name, extra=None):
    ts = time.time()
    event = {
        "timestamp": ts,
        "raw": raw_text,
        "clean": raw_text,
        "hidden": False,
        "decision": "allow",
        "risk": 0,
        "domain": "",
        "source": {"name": source_name},
        "meta": {"type": event_type},
    }
    if extra:
        event["meta"].update(extra)
    return event

def get_reference_time():
    """
    Multi-source reference time:
    1) timeapi.io (UTC)
    2) Google Date header
    Returns Unix timestamp or None.
    """
    # Option 1: timeapi.io
    try:
        resp = requests.get("https://timeapi.io/api/Time/current/zone?timeZone=UTC", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            dt = data.get("dateTime")
            if dt:
                import datetime
                ts = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp()
                return ts
    except Exception:
        traceback.print_exc()

    # Option 2: Google Date header
    try:
        resp = requests.get("https://www.google.com", timeout=2)
        date_header = resp.headers.get("Date")
        if date_header:
            import email.utils
            dt = email.utils.parsedate_to_datetime(date_header)
            return dt.timestamp()
    except Exception:
        traceback.print_exc()

    return None

def compute_time_risk(drift_sec: float) -> int:
    drift = abs(drift_sec)
    if drift < DRIFT_ALERT_THRESHOLD_SEC:
        return 0
    if drift < DRIFT_SUSPICIOUS_THRESHOLD_SEC:
        return 3
    return 7

def time_integrity_sentinel():
    """
    Periodically compares system time to a trusted reference.
    Emits:
      - 'time_change' when drift exceeds threshold
      - 'time_reference_unreachable' when no reference is available
    """
    while True:
        try:
            ref_ts = get_reference_time()
            sys_ts = time.time()

            if ref_ts is None:
                raw = "Time reference unreachable"
                event = {
                    "timestamp": sys_ts,
                    "raw": raw,
                    "clean": raw,
                    "hidden": False,
                    "decision": "flag",
                    "risk": 2,
                    "domain": "",
                    "source": {"name": "TimeSentinel"},
                    "meta": {
                        "type": "time_reference_unreachable",
                        "system_timestamp": sys_ts,
                    },
                }
                producer.send_event(event)
            else:
                drift = sys_ts - ref_ts
                risk = compute_time_risk(drift)

                if abs(drift) >= DRIFT_ALERT_THRESHOLD_SEC:
                    raw = f"Time drift detected: {drift:+.2f}s (system vs reference)"
                    event = {
                        "timestamp": sys_ts,
                        "raw": raw,
                        "clean": raw,
                        "hidden": False,
                        "decision": "allow" if risk < 5 else "flag",
                        "risk": risk,
                        "domain": "",
                        "source": {"name": "TimeSentinel"},
                        "meta": {
                            "type": "time_change",
                            "drift_seconds": drift,
                            "reference_timestamp": ref_ts,
                            "system_timestamp": sys_ts,
                        },
                    }
                    producer.send_event(event)

        except Exception:
            traceback.print_exc()

        time.sleep(TIME_CHECK_INTERVAL_SEC)

def monitor_window_focus():
    producer.connect()
    last_title = ""
    last_proc = ""
    while True:
        try:
            title, proc = get_active_window_info()
            if title != last_title or proc != last_proc:
                last_title = title
                last_proc = proc
                raw = f"WindowFocus: {proc} | {title}"
                event = build_event(
                    "window_focus",
                    raw,
                    proc,
                    extra={"window_title": title},
                )
                producer.send_event(event)
        except Exception:
            traceback.print_exc()
        time.sleep(0.1)

def on_click(x, y, button, pressed):
    if not pressed:
        return
    try:
        title, proc = get_active_window_info()
        raw = f"Click: {proc} | {title} | {button} @ ({x},{y})"
        event = build_event(
            "mouse_click",
            raw,
            proc,
            extra={"x": x, "y": y, "button": str(button), "window_title": title},
        )
        producer.send_event(event)
    except Exception:
        traceback.print_exc()

def on_press(key):
    try:
        title, proc = get_active_window_info()
        try:
            key_str = key.char if hasattr(key, "char") else str(key)
        except:
            key_str = str(key)
        raw = f"Keypress: {proc} | {title} | {key_str}"
        event = build_event(
            "key_press",
            raw,
            proc,
            extra={"key": key_str, "window_title": title},
        )
        producer.send_event(event)
    except Exception:
        traceback.print_exc()

def start_producer_hooks():
    threading.Thread(target=monitor_window_focus, daemon=True).start()
    threading.Thread(target=time_integrity_sentinel, daemon=True).start()
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    print("[PRODUCER] Win32 hook producer + Time Sentinel running...")

# =========================
# MAIN
# =========================

def main():
    threading.Thread(target=bus_server_loop, daemon=True).start()
    start_producer_hooks()

    app = QApplication(sys.argv)
    theme_manager = ThemeManager(config)
    theme_manager.apply(app)
    cockpit = Cockpit(app, config, theme_manager)
    cockpit.show()

    threading.Thread(target=tcp_client_loop, daemon=True).start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

