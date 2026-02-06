#!/usr/bin/env python3
# guardian_hybrid_edr_ai_encoding.py

import sys
import subprocess
import importlib
import threading
import time
import os
import json
import socket
import ctypes
from typing import Dict, Any, Optional, List

# -----------------------------
# Autoloader
# -----------------------------

REQUIRED_PACKAGES = [
    "PyQt5",
    "psutil",
    "pywin32",
    "cryptography",
]

def ensure_package(pkg_name, log_func=None):
    try:
        importlib.import_module(pkg_name)
        if log_func:
            log_func(f"[AUTOLOADER] {pkg_name} already installed.")
        return True
    except ImportError:
        if log_func:
            log_func(f"[AUTOLOADER] {pkg_name} missing. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.invalidate_caches()
            importlib.import_module(pkg_name)
            if log_func:
                log_func(f"[AUTOLOADER] {pkg_name} installed successfully.")
            return True
        except Exception as e:
            if log_func:
                log_func(f"[AUTOLOADER] Failed to install {pkg_name}: {e}")
            return False

def run_autoloader(log_func=None):
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        ok = ensure_package(pkg, log_func=log_func)
        all_ok = all_ok and ok
    return all_ok

# -----------------------------
# Imports after autoloader
# -----------------------------

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTabWidget, QTextEdit, QCheckBox, QFrame,
        QSpacerItem, QSizePolicy, QFileDialog, QLineEdit
    )
    from PyQt5.QtGui import QColor, QPalette, QFont
except ImportError:
    print("PyQt5 is required but could not be imported. Exiting.")
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None

from cryptography.fernet import Fernet

# pywin32 (for service)
try:
    import win32service
    import win32serviceutil
    import win32event
    import servicemanager
except ImportError:
    win32service = None
    win32serviceutil = None
    win32event = None
    servicemanager = None

# -----------------------------
# Common helpers / paths
# -----------------------------

def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False

HOST = "127.0.0.1"
PORT = 9797
SERVICE_NAME = "GuardianSystemService"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILENAME = "guardian_config.json"
STATE_FILENAME = "guardian_state.json"

def load_config() -> Dict[str, Any]:
    path = os.path.join(SCRIPT_DIR, CONFIG_FILENAME)
    if not os.path.exists(path):
        return {"state_path": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "state_path" not in cfg:
            cfg["state_path"] = None
        return cfg
    except Exception:
        return {"state_path": None}

def save_config(config: Dict[str, Any]):
    path = os.path.join(SCRIPT_DIR, CONFIG_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

def resolve_state_dir(config: Dict[str, Any]) -> str:
    custom = config.get("state_path")
    if custom and os.path.isdir(custom):
        return custom
    return SCRIPT_DIR

def load_state(config: Dict[str, Any]) -> Dict[str, Any]:
    state_dir = resolve_state_dir(config)
    path = os.path.join(state_dir, STATE_FILENAME)
    if not os.path.exists(path):
        return {"ai_enabled": False, "encoding_enabled": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if "ai_enabled" not in st:
            st["ai_enabled"] = False
        if "encoding_enabled" not in st:
            st["encoding_enabled"] = False
        return st
    except Exception:
        return {"ai_enabled": False, "encoding_enabled": False}

def save_state(config: Dict[str, Any], state: Dict[str, Any]):
    state_dir = resolve_state_dir(config)
    try:
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, STATE_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

# -----------------------------
# Service management helpers
# -----------------------------

def service_is_installed() -> bool:
    if win32serviceutil is None:
        return False
    try:
        win32serviceutil.QueryServiceStatus(SERVICE_NAME)
        return True
    except Exception:
        return False

def service_is_running() -> bool:
    if win32serviceutil is None:
        return False
    try:
        status = win32serviceutil.QueryServiceStatus(SERVICE_NAME)
        return status[1] == 4  # SERVICE_RUNNING
    except Exception:
        return False

def service_is_available() -> bool:
    return service_is_running()

def try_install_and_start_service(log_func=None) -> bool:
    if win32serviceutil is None:
        if log_func:
            log_func("[SYSTEM] pywin32 not available; cannot manage service.")
        return False

    log = log_func or (lambda x: None)

    try:
        log("[SYSTEM] Checking if service is installed...")
        installed = service_is_installed()

        if not installed:
            log("[SYSTEM] Service not installed. Installing...")
            win32serviceutil.InstallService(
                pythonClassString=f"{__name__}.GuardianSystemService",
                serviceName=SERVICE_NAME,
                displayName="Guardian SYSTEM Service (Hybrid EDR)",
                description="Provides SYSTEM-level defensive operations for Guardian Hybrid EDR."
            )
            log("[SYSTEM] Service installed.")

        log("[SYSTEM] Starting service...")
        try:
            win32serviceutil.StartService(SERVICE_NAME)
        except Exception as e:
            log(f"[SYSTEM] StartService raised: {e}")
        time.sleep(2.0)

        if service_is_available():
            log("[SYSTEM] Service is now RUNNING.")
            return True
        else:
            log("[SYSTEM] Service did not reach RUNNING state.")
            return False

    except Exception as e:
        log(f"[SYSTEM] Failed to install/start service: {e}")
        return False

def force_reinstall_service(log_func=None) -> bool:
    if win32serviceutil is None:
        if log_func:
            log_func("[SYSTEM] pywin32 not available; cannot force reinstall.")
        return False

    log = log_func or (lambda x: None)

    try:
        if service_is_installed():
            log("[SYSTEM] Stopping service for reinstall...")
            try:
                win32serviceutil.StopService(SERVICE_NAME)
                time.sleep(1.0)
            except Exception as e:
                log(f"[SYSTEM] StopService during reinstall: {e}")

            log("[SYSTEM] Removing service...")
            try:
                win32serviceutil.RemoveService(SERVICE_NAME)
            except Exception as e:
                log(f"[SYSTEM] RemoveService during reinstall: {e}")
        else:
            log("[SYSTEM] Service not installed; nothing to remove.")

        log("[SYSTEM] Installing fresh service...")
        win32serviceutil.InstallService(
            pythonClassString=f"{__name__}.GuardianSystemService",
            serviceName=SERVICE_NAME,
            displayName="Guardian SYSTEM Service (Hybrid EDR)",
            description="Provides SYSTEM-level defensive operations for Guardian Hybrid EDR."
        )
        log("[SYSTEM] Fresh service installed.")

        log("[SYSTEM] Starting fresh service...")
        try:
            win32serviceutil.StartService(SERVICE_NAME)
        except Exception as e:
            log(f"[SYSTEM] StartService after reinstall: {e}")
        time.sleep(2.0)

        if service_is_available():
            log("[SYSTEM] Force reinstall successful; service RUNNING.")
            return True
        else:
            log("[SYSTEM] Force reinstall attempted, but service not RUNNING.")
            return False

    except Exception as e:
        log(f"[SYSTEM] Force reinstall failed: {e}")
        return False

def test_port_status(host: str, port: int) -> Dict[str, Any]:
    info = {"can_connect": False, "in_use": False, "pid": None, "error": None}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            info["can_connect"] = True
            info["in_use"] = True
        except Exception as e:
            info["error"] = str(e)
        finally:
            s.close()
    except Exception as e:
        info["error"] = str(e)

    if psutil:
        try:
            for c in psutil.net_connections(kind="tcp"):
                if c.laddr and c.laddr.port == port and c.laddr.ip in ("127.0.0.1", "0.0.0.0"):
                    info["in_use"] = True
                    info["pid"] = c.pid
                    break
        except Exception as e:
            info["error"] = f"{info.get('error') or ''}; psutil: {e}"

    return info

# -----------------------------
# SYSTEM Service (LocalSystem)
# -----------------------------

if win32service is not None:

    class GuardianSystemService(win32serviceutil.ServiceFramework):
        _svc_name_ = SERVICE_NAME
        _svc_display_name_ = "Guardian SYSTEM Service (Hybrid EDR)"
        _svc_description_ = "Provides SYSTEM-level defensive operations for Guardian Hybrid EDR."

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.running = True

        def SvcStop(self):
            self.running = False
            win32event.SetEvent(self.stop_event)

        def SvcDoRun(self):
            servicemanager.LogInfoMsg("Guardian SYSTEM Service (Hybrid EDR) started.")
            self.run_service()

        def run_service(self):
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((HOST, PORT))
            except Exception as e:
                if servicemanager:
                    servicemanager.LogErrorMsg(f"Guardian SYSTEM Service bind error: {e}")
                return

            server.listen(5)

            while self.running:
                try:
                    server.settimeout(1.0)
                    try:
                        client, addr = server.accept()
                    except socket.timeout:
                        continue

                    data = client.recv(4096).decode("utf-8")
                    try:
                        req = json.loads(data)
                        cmd = req.get("cmd")
                        if cmd == "ping":
                            result = {"status": "ok", "role": "SYSTEM_SERVICE"}
                        elif cmd == "deep_scrub":
                            result = self.deep_scrub()
                        elif cmd == "kill_process":
                            target = req.get("target")
                            result = self.kill_process(target)
                        elif cmd == "move_file":
                            src = req.get("src")
                            dst = req.get("dst")
                            result = self.move_file(src, dst)
                        elif cmd == "get_status":
                            result = self.get_status()
                        else:
                            result = {"status": "error", "message": "Unknown command"}
                    except Exception as e:
                        result = {"status": "error", "message": str(e)}

                    client.send(json.dumps(result).encode("utf-8"))
                    client.close()
                except Exception as e:
                    if servicemanager:
                        servicemanager.LogErrorMsg(f"Guardian SYSTEM Service error: {e}")
                    time.sleep(1.0)

        def deep_scrub(self):
            try:
                os.system("reg add HKLM\\Software\\Policies\\Microsoft\\Windows\\DataCollection /v AllowTelemetry /t REG_DWORD /d 0 /f")
                os.system("schtasks /Change /TN \"Microsoft\\Windows\\Customer Experience Improvement Program\\Consolidator\" /Disable")
                os.system("schtasks /Change /TN \"Microsoft\\Windows\\Customer Experience Improvement Program\\UsbCeip\" /Disable")
                os.system("sc stop DiagTrack")
                os.system("sc config DiagTrack start= disabled")
                return {"status": "ok", "message": "Deep scrub completed."}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        def kill_process(self, target):
            if not target:
                return {"status": "error", "message": "No target specified"}
            try:
                import psutil as _ps
                killed = []
                if target.isdigit():
                    pid = int(target)
                    try:
                        p = _ps.Process(pid)
                        p.terminate()
                        killed.append(pid)
                    except Exception:
                        pass
                else:
                    for p in _ps.process_iter(["pid", "name"]):
                        if p.info["name"] and p.info["name"].lower() == target.lower():
                            try:
                                _ps.Process(p.info["pid"]).terminate()
                                killed.append(p.info["pid"])
                            except Exception:
                                pass
                if killed:
                    return {"status": "ok", "killed": killed}
                else:
                    return {"status": "error", "message": "No matching process killed"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        def move_file(self, src: str, dst: str):
            try:
                if not src or not dst:
                    return {"status": "error", "message": "src/dst required"}

                try:
                    os.system(f'takeown /f "{src}" /a /r /d y')
                    os.system(f'icacls "{src}" /grant administrators:F /t /c')
                except Exception:
                    pass

                dst_dir = os.path.dirname(dst)
                if dst_dir and not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)

                try:
                    import shutil
                    shutil.move(src, dst)
                    return {"status": "ok", "message": "moved", "src": src, "dst": dst}
                except Exception as e:
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                        os.remove(src)
                        return {"status": "ok", "message": "copied+deleted", "src": src, "dst": dst}
                    except Exception as e2:
                        return {"status": "error", "message": f"move failed: {e}; copy+delete failed: {e2}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        def get_status(self):
            try:
                return {
                    "status": "ok",
                    "role": "SYSTEM_SERVICE",
                    "time": int(time.time())
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

# -----------------------------
# Monitoring thread
# -----------------------------

class MonitoringThread(QThread):
    stats_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self):
        self.log_signal.emit("[MONITOR] Monitoring thread started.")
        while self._running:
            stats = {}
            try:
                if psutil:
                    stats["cpu"] = psutil.cpu_percent(interval=0.5)
                    stats["mem"] = psutil.virtual_memory().percent
                    net = psutil.net_io_counters()
                    stats["net"] = net.bytes_sent + net.bytes_recv
                else:
                    stats["cpu"] = 0
                    stats["mem"] = 0
                    stats["net"] = 0
            except Exception as e:
                self.log_signal.emit(f"[MONITOR] Error collecting stats: {e}")
                stats = {"cpu": 0, "mem": 0, "net": 0}

            self.stats_signal.emit(stats)
            time.sleep(1.0)

    def stop(self):
        self._running = False

# -----------------------------
# File scanner thread (real detector)
# -----------------------------

class FileScannerThread(QThread):
    found_suspicious = pyqtSignal(dict)

    def __init__(self, watch_paths: List[str], parent=None):
        super().__init__(parent)
        self.watch_paths = watch_paths
        self.running = True
        self.seen_files = set()

    def run(self):
        while self.running:
            for folder in self.watch_paths:
                if not folder or not os.path.exists(folder):
                    continue
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        full = os.path.join(root, f)
                        if full in self.seen_files:
                            continue
                        self.seen_files.add(full)
                        if self.is_suspicious(full):
                            self.found_suspicious.emit({"path": full})
            time.sleep(5)

    def stop(self):
        self.running = False

    def is_suspicious(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".exe", ".dll", ".js", ".vbs", ".ps1", ".bat"]:
            return True
        try:
            with open(path, "rb") as fp:
                data = fp.read(4096)
                low = data.lower()
                if b"powershell" in low:
                    return True
                if b"base64" in low and b"invoke" in low:
                    return True
        except Exception:
            pass
        return False

# -----------------------------
# SYSTEM client
# -----------------------------

class DeepScrubClient:
    def __init__(self, log_func=None):
        self.log = log_func or (lambda x: None)
        self.host = HOST
        self.port = PORT

    def _send_cmd(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            s.send(json.dumps(payload).encode("utf-8"))
            resp = json.loads(s.recv(4096).decode("utf-8"))
            s.close()
            return resp
        except Exception as e:
            self.log(f"[SYSTEM] Error contacting SYSTEM service: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        return self._send_cmd({"cmd": "get_status"})

    def request_deep_scrub(self) -> Dict[str, Any]:
        self.log("[SYSTEM] Requesting deep scrub from SYSTEM service...")
        return self._send_cmd({"cmd": "deep_scrub"})

    def kill_protected_process(self, target: str) -> Dict[str, Any]:
        self.log(f"[SYSTEM] Requesting kill of protected process: {target}")
        return self._send_cmd({"cmd": "kill_process", "target": target})

    def move_file(self, src: str, dst: str) -> Dict[str, Any]:
        self.log(f"[SYSTEM] Requesting move_file: {src} -> {dst}")
        return self._send_cmd({"cmd": "move_file", "src": src, "dst": dst})

# -----------------------------
# AI + Encoding (lightweight)
# -----------------------------

class SimpleAIEngine:
    def analyze_file(self, path: str) -> Dict[str, Any]:
        score = 0.0
        ext = os.path.splitext(path)[1].lower()
        if ext in [".exe", ".dll", ".js", ".vbs", ".ps1", ".bat"]:
            score += 0.4
        try:
            with open(path, "rb") as fp:
                data = fp.read(4096)
                low = data.lower()
                if b"powershell" in low:
                    score += 0.3
                if b"base64" in low:
                    score += 0.2
        except Exception:
            pass
        return {"risk_score": min(score, 1.0)}

class EncodingEngine:
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or Fernet.generate_key()
        self.fernet = Fernet(self.key)

    def encode(self, data: str) -> str:
        token = self.fernet.encrypt(data.encode("utf-8")).decode("utf-8")
        return f"[ENC]{token}[/ENC]"

# -----------------------------
# GUI helpers
# -----------------------------

def apply_dark_hybrid_theme(app):
    palette = QPalette()
    base_color = QColor(15, 15, 20)
    panel_color = QColor(25, 25, 32)
    accent_color = QColor(0, 180, 255)
    text_color = QColor(230, 230, 240)

    palette.setColor(QPalette.Window, base_color)
    palette.setColor(QPalette.WindowText, text_color)
    palette.setColor(QPalette.Base, panel_color)
    palette.setColor(QPalette.AlternateBase, base_color)
    palette.setColor(QPalette.ToolTipBase, panel_color)
    palette.setColor(QPalette.ToolTipText, text_color)
    palette.setColor(QPalette.Text, text_color)
    palette.setColor(QPalette.Button, panel_color)
    palette.setColor(QPalette.ButtonText, text_color)
    palette.setColor(QPalette.Highlight, accent_color)
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(palette)
    app.setStyle("Fusion")

def make_section_frame(title: str):
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setFrameShadow(QFrame.Raised)
    layout = QVBoxLayout(frame)
    label = QLabel(title)
    label.setFont(QFont("Segoe UI", 10, QFont.Bold))
    layout.addWidget(label)
    return frame, layout

# -----------------------------
# GUI main app
# -----------------------------

class GuardianApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian Hybrid EDR - SYSTEM + AI + Encoding")
        self.resize(1250, 780)

        # Load config + state (persistent)
        self.config = load_config()
        self.state = load_state(self.config)
        self.state_dir = resolve_state_dir(self.config)

        self.deep_client = DeepScrubClient(log_func=self.append_log)
        self.freeze_active = False
        self.outbound_shield_active = True
        self.selected_path = ""

        self.service_watchdog_enabled = True
        self.last_service_error = "None"
        self.last_service_action = "None"

        self.file_move_suggestions: List[Dict[str, str]] = []

        self.ai_engine = SimpleAIEngine()
        self.encoding_engine = EncodingEngine()
        self.ai_enabled = bool(self.state.get("ai_enabled", False))
        self.encoding_enabled = bool(self.state.get("encoding_enabled", False))

        self.init_ui()
        self.start_background_threads()
        self.init_scanner()
        self.check_system_service_status()
        self.init_service_watchdog()
        self.apply_persistent_states_to_ui()

    # ---------- UI ----------

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard_tab = QWidget()
        self.controls_tab = QWidget()
        self.telemetry_tab = QWidget()
        self.logs_tab = QWidget()
        self.settings_tab = QWidget()
        self.diagnostics_tab = QWidget()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.controls_tab, "Controls")
        self.tabs.addTab(self.telemetry_tab, "SYSTEM / Telemetry")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")

        self.init_dashboard_tab()
        self.init_controls_tab()
        self.init_telemetry_tab()
        self.init_logs_tab()
        self.init_settings_tab()
        self.init_diagnostics_tab()

    def init_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)

        # SYSTEM banner
        self.system_banner = QFrame()
        self.system_banner.setFrameShape(QFrame.StyledPanel)
        self.system_banner.setStyleSheet("""
            QFrame {
                background-color: #224466;
                border: 1px solid #4488cc;
                border-radius: 4px;
            }
            QLabel {
                color: white;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0088ff;
            }
        """)
        banner_layout = QHBoxLayout(self.system_banner)
        self.system_banner_label = QLabel("SYSTEM service status: UNKNOWN")
        self.system_banner_button = QPushButton("Refresh SYSTEM Status")
        self.system_banner_button.clicked.connect(self.check_system_service_status)
        banner_layout.addWidget(self.system_banner_label)
        banner_layout.addStretch()
        banner_layout.addWidget(self.system_banner_button)
        layout.addWidget(self.system_banner)

        # Top row: Status, Metrics, AI, Encoding, Storage Path
        top_row = QHBoxLayout()

        status_frame, status_layout = make_section_frame("Status")
        self.status_label = QLabel("Guardian Hybrid EDR: Online")
        self.status_label.setFont(QFont("Segoe UI", 11))
        status_layout.addWidget(self.status_label)
        top_row.addWidget(status_frame)

        stats_frame, stats_layout = make_section_frame("Live Metrics")
        self.cpu_label = QLabel("CPU: -- %")
        self.mem_label = QLabel("Memory: -- %")
        self.net_label = QLabel("Net I/O: -- bytes")
        for lbl in (self.cpu_label, self.mem_label, self.net_label):
            lbl.setFont(QFont("Consolas", 10))
            stats_layout.addWidget(lbl)
        top_row.addWidget(stats_frame)

        # AI button (small tactical, red OFF / green ON)
        self.ai_button = QPushButton("AI: OFF")
        self.ai_button.setCheckable(True)
        self.ai_button.setMinimumHeight(32)
        self.ai_button.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                font-size: 12px;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:checked {
                background-color: #00cc44;
                color: black;
            }
        """)
        self.ai_button.setToolTip("Toggle AI risk analysis engine (persistent across reboots)")
        self.ai_button.clicked.connect(self.toggle_ai_button)
        top_row.addWidget(self.ai_button)

        # Encoding button (small tactical, red OFF / green ON)
        self.encoding_button = QPushButton("Encoding: OFF")
        self.encoding_button.setCheckable(True)
        self.encoding_button.setMinimumHeight(32)
        self.encoding_button.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                font-size: 12px;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:checked {
                background-color: #00cc44;
                color: black;
            }
        """)
        self.encoding_button.setToolTip("Toggle encoding engine for sample extraction and preview (persistent across reboots)")
        self.encoding_button.clicked.connect(self.toggle_encoding_button)
        top_row.addWidget(self.encoding_button)

        # Storage path button (small tactical, red default / blue custom)
        self.storage_path_button = QPushButton("AI/Encoding Storage Path")
        self.storage_path_button.setCheckable(True)
        self.storage_path_button.setMinimumHeight(32)
        self.storage_path_button.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                font-size: 12px;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:checked {
                background-color: #0066cc;
                color: white;
            }
        """)
        self.storage_path_button.setToolTip("Select local or SMB folder for persistent AI/Encoding state storage")
        self.storage_path_button.clicked.connect(self.on_storage_path_button_clicked)
        top_row.addWidget(self.storage_path_button)

        layout.addLayout(top_row)

        # Events
        events_frame, events_layout = make_section_frame("Recent Events")
        self.events_log = QTextEdit()
        self.events_log.setReadOnly(True)
        events_layout.addWidget(self.events_log)
        layout.addWidget(events_frame)

    def init_controls_tab(self):
        layout = QVBoxLayout(self.controls_tab)

        shield_frame, shield_layout = make_section_frame("Outbound Shield")
        self.shield_toggle = QCheckBox("Enable Outbound Shield")
        self.shield_toggle.setChecked(True)
        self.shield_toggle.stateChanged.connect(self.toggle_outbound_shield)
        shield_layout.addWidget(self.shield_toggle)
        layout.addWidget(shield_frame)

        freeze_frame, freeze_layout = make_section_frame("Freeze Switch")
        self.freeze_button = QPushButton("ENGAGE FREEZE")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setMinimumHeight(60)
        self.freeze_button.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                font-size: 16px;
                border-radius: 6px;
            }
            QPushButton:checked {
                background-color: #ff3333;
            }
        """)
        self.freeze_button.clicked.connect(self.toggle_freeze)
        freeze_layout.addWidget(self.freeze_button)
        layout.addWidget(freeze_frame)

        deep_frame, deep_layout = make_section_frame("SYSTEM Operations")
        self.deep_button = QPushButton("Execute Deep Scrub (SYSTEM)")
        self.deep_button.setMinimumHeight(40)
        self.deep_button.clicked.connect(self.on_deep_scrub)
        deep_layout.addWidget(self.deep_button)

        self.kill_proc_input = QLineEdit()
        self.kill_proc_input.setPlaceholderText("Process name or PID to kill via SYSTEM")
        self.kill_proc_button = QPushButton("Kill Protected Process (SYSTEM)")
        self.kill_proc_button.clicked.connect(self.on_kill_protected_process)
        deep_layout.addWidget(self.kill_proc_input)
        deep_layout.addWidget(self.kill_proc_button)

        layout.addWidget(deep_frame)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def init_telemetry_tab(self):
        layout = QVBoxLayout(self.telemetry_tab)

        status_frame, status_layout = make_section_frame("SYSTEM Status")
        self.telemetry_status_label = QLabel("SYSTEM service: unknown")
        self.telemetry_last_result_label = QLabel("Last SYSTEM action: --")
        status_layout.addWidget(self.telemetry_status_label)
        status_layout.addWidget(self.telemetry_last_result_label)
        layout.addWidget(status_frame)

        btn_frame, btn_layout = make_section_frame("Actions")
        self.telemetry_run_button = QPushButton("Run Deep Scrub (SYSTEM)")
        self.telemetry_run_button.clicked.connect(self.on_deep_scrub)
        btn_layout.addWidget(self.telemetry_run_button)
        layout.addWidget(btn_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def init_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

    def init_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)

        autoloader_frame, autoloader_layout = make_section_frame("Autoloader Status")
        self.autoloader_status_label = QLabel("Autoloader: Completed at startup.")
        autoloader_layout.addWidget(self.autoloader_status_label)
        layout.addWidget(autoloader_frame)

        path_frame, path_layout = make_section_frame("Local / SMB Path (General)")
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_button = QPushButton("Select Local/SMB Folder")
        self.path_button.clicked.connect(self.select_path)
        path_layout.addWidget(QLabel("Selected path:"))
        path_layout.addWidget(self.path_display)
        path_layout.addWidget(self.path_button)
        layout.addWidget(path_frame)

        theme_frame, theme_layout = make_section_frame("Theme")
        theme_layout.addWidget(QLabel("Current theme: Dark Hybrid EDR"))
        layout.addWidget(theme_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def init_diagnostics_tab(self):
        layout = QVBoxLayout(self.diagnostics_tab)

        svc_frame, svc_layout = make_section_frame("SYSTEM Service State")
        self.diag_installed_label = QLabel("Installed: unknown")
        self.diag_running_label = QLabel("Running: unknown")
        self.diag_last_action_label = QLabel("Last action: None")
        self.diag_last_error_label = QLabel("Last error: None")
        for lbl in (self.diag_installed_label, self.diag_running_label,
                    self.diag_last_action_label, self.diag_last_error_label):
            lbl.setFont(QFont("Consolas", 9))
            svc_layout.addWidget(lbl)
        layout.addWidget(svc_frame)

        port_frame, port_layout = make_section_frame("Port / IPC Diagnostics")
        self.diag_port_status_label = QLabel("Port 9797: unknown")
        self.diag_port_pid_label = QLabel("PID using port: None")
        port_layout.addWidget(self.diag_port_status_label)
        port_layout.addWidget(self.diag_port_pid_label)
        layout.addWidget(port_frame)

        btn_frame, btn_layout = make_section_frame("Controls")
        self.diag_refresh_button = QPushButton("Refresh Diagnostics")
        self.diag_refresh_button.clicked.connect(self.refresh_diagnostics)

        self.diag_force_reinstall_button = QPushButton("Force Reinstall SYSTEM Service")
        self.diag_force_reinstall_button.clicked.connect(self.on_force_reinstall_service)

        self.diag_start_button = QPushButton("Start Service")
        self.diag_start_button.clicked.connect(self.on_start_service)

        self.diag_stop_button = QPushButton("Stop Service")
        self.diag_stop_button.clicked.connect(self.on_stop_service)

        self.diag_watchdog_toggle = QCheckBox("Enable SYSTEM Service Watchdog (auto-repair)")
        self.diag_watchdog_toggle.setChecked(True)
        self.diag_watchdog_toggle.stateChanged.connect(self.toggle_watchdog)

        btn_layout.addWidget(self.diag_refresh_button)
        btn_layout.addWidget(self.diag_force_reinstall_button)
        btn_layout.addWidget(self.diag_start_button)
        btn_layout.addWidget(self.diag_stop_button)
        btn_layout.addWidget(self.diag_watchdog_toggle)
        layout.addWidget(btn_frame)

        suggest_frame, suggest_layout = make_section_frame("File Move Suggestions (Hybrid)")
        self.suggested_file_list = QTextEdit()
        self.suggested_file_list.setReadOnly(True)

        self.manual_file_input = QLineEdit()
        self.manual_file_input.setPlaceholderText("Path to file to suggest move for")

        self.manual_quarantine_input = QLineEdit()
        self.manual_quarantine_input.setPlaceholderText("Quarantine destination (e.g. C:\\Quarantine)")

        self.manual_suggest_button = QPushButton("Suggest Move for File")
        self.manual_suggest_button.clicked.connect(self.on_manual_suggest_move)

        self.apply_selected_move_button = QPushButton("Execute Suggested Move via SYSTEM")
        self.apply_selected_move_button.clicked.connect(self.on_apply_selected_move)

        suggest_layout.addWidget(QLabel("Pending suggestions:"))
        suggest_layout.addWidget(self.suggested_file_list)
        suggest_layout.addWidget(self.manual_file_input)
        suggest_layout.addWidget(self.manual_quarantine_input)
        suggest_layout.addWidget(self.manual_suggest_button)
        suggest_layout.addWidget(self.apply_selected_move_button)

        layout.addWidget(suggest_frame)

        log_frame, log_layout = make_section_frame("Diagnostics Log")
        self.diag_log = QTextEdit()
        self.diag_log.setReadOnly(True)
        log_layout.addWidget(self.diag_log)
        layout.addWidget(log_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.refresh_diagnostics()
        self.refresh_file_move_suggestions()

    # ---------- Threads / watchdog / scanner ----------

    def start_background_threads(self):
        self.monitor_thread = MonitoringThread()
        self.monitor_thread.stats_signal.connect(self.update_dashboard_stats)
        self.monitor_thread.log_signal.connect(self.append_log)
        self.monitor_thread.start()

    def init_scanner(self):
        watch_paths = []
        public = r"C:\Users\Public"
        downloads = os.path.expandvars(r"%USERPROFILE%\Downloads")
        for p in [public, downloads]:
            if p and os.path.exists(p):
                watch_paths.append(p)
        self.file_scanner = FileScannerThread(watch_paths)
        self.file_scanner.found_suspicious.connect(self.on_scanner_detect)
        self.file_scanner.start()
        self.diag_log.append(f"[SCANNER] Watching paths: {watch_paths}")

    def init_service_watchdog(self):
        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.setInterval(5000)
        self.watchdog_timer.timeout.connect(self.service_watchdog_tick)
        self.watchdog_timer.start()

    def service_watchdog_tick(self):
        if not self.service_watchdog_enabled:
            return
        result = self.deep_client.get_status()
        if result.get("status") == "ok":
            return
        self.append_log("[WATCHDOG] SYSTEM service appears offline. Attempting auto-repair...")
        self.diag_log.append("[WATCHDOG] SYSTEM service offline. Auto-repair in progress...")
        ok = try_install_and_start_service(log_func=self.append_log)
        if ok:
            self.append_log("[WATCHDOG] Auto-repair successful.")
            self.diag_log.append("[WATCHDOG] Auto-repair successful.")
        else:
            self.append_log("[WATCHDOG] Auto-repair failed.")
            self.diag_log.append("[WATCHDOG] Auto-repair failed.")
        self.refresh_diagnostics()
        self.check_system_service_status(auto=True)

    # ---------- SYSTEM status / diagnostics ----------

    def check_system_service_status(self, auto: bool = False):
        result = self.deep_client.get_status()
        if result.get("status") == "ok":
            self.system_banner_label.setText("SYSTEM service status: ONLINE (LocalSystem)")
            self.telemetry_status_label.setText("SYSTEM service: ONLINE")
            if not auto:
                self.append_log("[SYSTEM] SYSTEM service is ONLINE.")
            return

        if not auto:
            self.append_log("[SYSTEM] SYSTEM service appears OFFLINE. Attempting self-heal (install/start)...")
        self.system_banner_label.setText("SYSTEM service status: RECOVERING (attempting self-heal)")
        self.telemetry_status_label.setText("SYSTEM service: RECOVERING")

        ok = try_install_and_start_service(log_func=self.append_log)

        if ok:
            result = self.deep_client.get_status()
            if result.get("status") == "ok":
                self.system_banner_label.setText("SYSTEM service status: ONLINE (LocalSystem)")
                self.telemetry_status_label.setText("SYSTEM service: ONLINE")
                if not auto:
                    self.append_log("[SYSTEM] Self-heal successful. SYSTEM service is ONLINE.")
            else:
                self.system_banner_label.setText("SYSTEM service status: OFFLINE (post self-heal)")
                self.telemetry_status_label.setText("SYSTEM service: OFFLINE")
                if not auto:
                    self.append_log("[SYSTEM] Self-heal attempted, but service still unreachable.")
        else:
            self.system_banner_label.setText("SYSTEM service status: OFFLINE (self-heal failed)")
            self.telemetry_status_label.setText("SYSTEM service: OFFLINE")
            if not auto:
                self.append_log("[SYSTEM] Self-heal failed. Service is still OFFLINE.")

    def refresh_diagnostics(self):
        installed = service_is_installed()
        running = service_is_running()
        self.diag_installed_label.setText(f"Installed: {installed}")
        self.diag_running_label.setText(f"Running: {running}")
        self.diag_last_action_label.setText(f"Last action: {self.last_service_action}")
        self.diag_last_error_label.setText(f"Last error: {self.last_service_error}")

        port_info = test_port_status(HOST, PORT)
        if port_info["can_connect"]:
            self.diag_port_status_label.setText("Port 9797: reachable (something is listening)")
        elif port_info["in_use"]:
            self.diag_port_status_label.setText("Port 9797: in use (cannot connect)")
        else:
            self.diag_port_status_label.setText(f"Port 9797: not reachable ({port_info.get('error')})")

        self.diag_port_pid_label.setText(f"PID using port: {port_info.get('pid')}")
        self.diag_log.append("[DIAG] Diagnostics refreshed.")

    def on_force_reinstall_service(self):
        self.diag_log.append("[DIAG] Force reinstall requested.")
        self.last_service_action = "Force reinstall"
        ok = force_reinstall_service(log_func=self.append_log)
        if ok:
            self.last_service_error = "None"
            self.diag_log.append("[DIAG] Force reinstall successful.")
        else:
            self.last_service_error = "Force reinstall failed"
            self.diag_log.append("[DIAG] Force reinstall failed.")
        self.refresh_diagnostics()
        self.check_system_service_status(auto=True)

    def on_start_service(self):
        self.diag_log.append("[DIAG] Start service requested.")
        self.last_service_action = "Start service"
        ok = try_install_and_start_service(log_func=self.append_log)
        if ok:
            self.last_service_error = "None"
            self.diag_log.append("[DIAG] Service started successfully.")
        else:
            self.last_service_error = "Start service failed"
            self.diag_log.append("[DIAG] Service start failed.")
        self.refresh_diagnostics()
        self.check_system_service_status(auto=True)

    def on_stop_service(self):
        self.diag_log.append("[DIAG] Stop service requested.")
        self.last_service_action = "Stop service"
        if win32serviceutil is None:
            self.last_service_error = "pywin32 missing"
            self.diag_log.append("[DIAG] Cannot stop service; pywin32 missing.")
            return
        try:
            if service_is_running():
                win32serviceutil.StopService(SERVICE_NAME)
                time.sleep(1.0)
                self.diag_log.append("[DIAG] Service stop requested.")
                self.last_service_error = "None"
            else:
                self.diag_log.append("[DIAG] Service already stopped.")
        except Exception as e:
            self.last_service_error = str(e)
            self.diag_log.append(f"[DIAG] Stop service failed: {e}")
        self.refresh_diagnostics()
        self.check_system_service_status(auto=True)

    def toggle_watchdog(self, state):
        self.service_watchdog_enabled = (state == Qt.Checked)
        status = "ENABLED" if self.service_watchdog_enabled else "DISABLED"
        self.diag_log.append(f"[WATCHDOG] SYSTEM service watchdog {status}.")
        self.append_log(f"[WATCHDOG] SYSTEM service watchdog {status}.")

    # ---------- Controls / AI / Encoding / Storage path ----------

    def apply_persistent_states_to_ui(self):
        # AI button
        self.ai_button.blockSignals(True)
        self.ai_button.setChecked(self.ai_enabled)
        self.ai_button.setText("AI: ON" if self.ai_enabled else "AI: OFF")
        self.ai_button.blockSignals(False)

        # Encoding button
        self.encoding_button.blockSignals(True)
        self.encoding_button.setChecked(self.encoding_enabled)
        self.encoding_button.setText("Encoding: ON" if self.encoding_enabled else "Encoding: OFF")
        self.encoding_button.blockSignals(False)

        # Storage path button
        custom_path = self.config.get("state_path")
        self.storage_path_button.blockSignals(True)
        if custom_path and os.path.isdir(custom_path):
            self.storage_path_button.setChecked(True)
            self.storage_path_button.setText("AI/Encoding Storage Path (Custom)")
            self.storage_path_button.setToolTip(
                f"Select local or SMB folder for persistent AI/Encoding state storage\nCurrent: {custom_path}"
            )
        else:
            self.storage_path_button.setChecked(False)
            self.storage_path_button.setText("AI/Encoding Storage Path")
            self.storage_path_button.setToolTip(
                "Select local or SMB folder for persistent AI/Encoding state storage\nCurrent: Script folder"
            )
        self.storage_path_button.blockSignals(False)

    def update_dashboard_stats(self, stats: dict):
        cpu = stats.get("cpu", 0)
        mem = stats.get("mem", 0)
        net = stats.get("net", 0)

        self.cpu_label.setText(f"CPU: {cpu:.1f} %")
        self.mem_label.setText(f"Memory: {mem:.1f} %")
        self.net_label.setText(f"Net I/O: {net} bytes")
        self.events_log.append(f"[STATS] CPU={cpu:.1f}% MEM={mem:.1f}% NET={net} bytes")

    def append_log(self, message: str):
        self.log_view.append(message)

    def toggle_freeze(self):
        self.freeze_active = self.freeze_button.isChecked()
        if self.freeze_active:
            self.status_label.setText("Guardian Hybrid EDR: FREEZE ENGAGED")
            self.append_log("[CONTROL] Freeze engaged.")
            self.events_log.append("[CONTROL] Freeze engaged.")
        else:
            self.status_label.setText("Guardian Hybrid EDR: Online")
            self.append_log("[CONTROL] Freeze disengaged.")
            self.events_log.append("[CONTROL] Freeze disengaged.")

    def toggle_outbound_shield(self, state):
        self.outbound_shield_active = (state == Qt.Checked)
        status = "ENABLED" if self.outbound_shield_active else "DISABLED"
        self.append_log(f"[CONTROL] Outbound Shield {status}.")
        self.events_log.append(f"[CONTROL] Outbound Shield {status}.")

    def toggle_ai_button(self):
        self.ai_enabled = self.ai_button.isChecked()
        self.ai_button.setText("AI: ON" if self.ai_enabled else "AI: OFF")
        self.state["ai_enabled"] = self.ai_enabled
        save_state(self.config, self.state)
        self.append_log(f"[CONTROL] AI Engine {'ENABLED' if self.ai_enabled else 'DISABLED'}.")
        self.events_log.append(f"[CONTROL] AI Engine {'ENABLED' if self.ai_enabled else 'DISABLED'}.")

    def toggle_encoding_button(self):
        self.encoding_enabled = self.encoding_button.isChecked()
        self.encoding_button.setText("Encoding: ON" if self.encoding_enabled else "Encoding: OFF")
        self.state["encoding_enabled"] = self.encoding_enabled
        save_state(self.config, self.state)
        self.append_log(f"[CONTROL] Encoding Engine {'ENABLED' if self.encoding_enabled else 'DISABLED'}.")
        self.events_log.append(f"[CONTROL] Encoding Engine {'ENABLED' if self.encoding_enabled else 'DISABLED'}.")

    def on_storage_path_button_clicked(self):
        # Always open picker; button check state just indicates custom vs default
        new_dir = QFileDialog.getExistingDirectory(self, "Select AI/Encoding State Folder", self.state_dir)
        if not new_dir:
            # User cancelled; restore visual state
            self.apply_persistent_states_to_ui()
            return

        new_dir = os.path.normpath(new_dir)
        old_dir = resolve_state_dir(self.config)
        old_path = os.path.join(old_dir, STATE_FILENAME)
        new_path = os.path.join(new_dir, STATE_FILENAME)

        try:
            os.makedirs(new_dir, exist_ok=True)
            if os.path.exists(old_path):
                import shutil
                shutil.move(old_path, new_path)
        except Exception:
            # If move fails, we still switch path; state will be recreated on next save
            pass

        # Update config
        if new_dir == SCRIPT_DIR:
            self.config["state_path"] = None
        else:
            self.config["state_path"] = new_dir
        save_config(self.config)

        # Update internal state_dir
        self.state_dir = resolve_state_dir(self.config)

        # Update tooltip + visual
        self.apply_persistent_states_to_ui()

    def select_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Local/SMB Folder", "")
        if path:
            self.selected_path = path
            self.path_display.setText(path)
            self.append_log(f"[SETTINGS] Path selected: {path}")

    def on_deep_scrub(self):
        self.events_log.append("[SYSTEM] Deep scrub requested.")
        result = self.deep_client.request_deep_scrub()
        self.events_log.append(f"[SYSTEM] Deep scrub result: {result}")
        self.telemetry_last_result_label.setText(f"Last SYSTEM action: {result.get('status', 'unknown')}")

    def on_kill_protected_process(self):
        target = self.kill_proc_input.text().strip()
        if not target:
            self.events_log.append("[SYSTEM] Kill request ignored (no target).")
            return
        self.events_log.append(f"[SYSTEM] Kill protected process requested: {target}")
        result = self.deep_client.kill_protected_process(target)
        self.events_log.append(f"[SYSTEM] Kill result: {result}")

    # ---------- Scanner → auto-suggest + AI + Encoding ----------

    def on_scanner_detect(self, info: Dict[str, Any]):
        path = info["path"]
        self.events_log.append(f"[SCANNER] Suspicious file detected: {path}")

        if self.ai_enabled:
            ai_result = self.ai_engine.analyze_file(path)
            risk = ai_result.get("risk_score", 0.0)
            self.events_log.append(f"[AI] Risk score for {path}: {risk:.2f}")
        else:
            risk = 0.5

        quarantine_root = "C:\\Quarantine"
        dst = os.path.join(quarantine_root, os.path.basename(path))

        reason = f"Scanner detection (risk={risk:.2f})"
        self.add_file_move_suggestion(path, dst, reason)

        if self.encoding_enabled:
            try:
                with open(path, "rb") as fp:
                    data = fp.read(1024).decode("utf-8", errors="ignore")
                encoded = self.encoding_engine.encode(data)
                self.events_log.append(f"[ENC] Sample encoded preview for {path}: {encoded[:120]}...")
            except Exception as e:
                self.events_log.append(f"[ENC] Failed to encode sample from {path}: {e}")

    # ---------- File move suggestions (Hybrid) ----------

    def add_file_move_suggestion(self, src: str, dst: str, reason: str):
        suggestion = {"src": src, "dst": dst, "reason": reason}
        self.file_move_suggestions.append(suggestion)
        self.refresh_file_move_suggestions()
        self.diag_log.append(f"[FILE] Suggested move: {src} -> {dst} ({reason})")

    def refresh_file_move_suggestions(self):
        self.suggested_file_list.clear()
        if not self.file_move_suggestions:
            self.suggested_file_list.append("No pending suggestions.")
            return
        for idx, s in enumerate(self.file_move_suggestions, start=1):
            self.suggested_file_list.append(
                f"{idx}. {s['src']} -> {s['dst']}  [Reason: {s['reason']}]"
            )

    def on_manual_suggest_move(self):
        src = self.manual_file_input.text().strip()
        dst_root = self.manual_quarantine_input.text().strip() or "C:\\Quarantine"
        if not src:
            self.diag_log.append("[FILE] Manual suggest ignored (no source path).")
            return
        dst = os.path.join(dst_root, os.path.basename(src))
        self.add_file_move_suggestion(src=src, dst=dst, reason="Manual operator suggestion")

    def on_apply_selected_move(self):
        if not self.file_move_suggestions:
            self.diag_log.append("[FILE] No suggestions to apply.")
            return

        suggestion = self.file_move_suggestions.pop(0)
        src = suggestion["src"]
        dst = suggestion["dst"]
        reason = suggestion["reason"]

        self.diag_log.append(f"[FILE] Executing move via SYSTEM: {src} -> {dst} ({reason})")
        result = self.deep_client.move_file(src, dst)
        self.diag_log.append(f"[FILE] Move result: {result}")

        if result.get("status") == "ok":
            self.events_log.append(f"[SYSTEM] File moved: {src} -> {dst}")
        else:
            self.events_log.append(f"[SYSTEM] File move FAILED for {src}: {result}")

        self.refresh_file_move_suggestions()

    # ---------- Close ----------

    def closeEvent(self, event):
        try:
            if hasattr(self, "monitor_thread") and self.monitor_thread.isRunning():
                self.monitor_thread.stop()
                self.monitor_thread.wait(2000)
        except Exception:
            pass
        try:
            if hasattr(self, "file_scanner") and self.file_scanner.isRunning():
                self.file_scanner.stop()
                self.file_scanner.wait(2000)
        except Exception:
            pass
        event.accept()

# -----------------------------
# Entry point
# -----------------------------

def main():
    def console_log(msg):
        print(msg)

    run_autoloader(log_func=console_log)

    if win32serviceutil is not None and len(sys.argv) > 1 and sys.argv[1].lower() in (
        "install", "remove", "start", "stop", "restart"
    ):
        win32serviceutil.HandleCommandLine(GuardianSystemService)
        return

    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            f'"{os.path.abspath(sys.argv[0])}"',
            None,
            1
        )
        os._exit(0)

    app = QApplication(sys.argv)
    apply_dark_hybrid_theme(app)

    window = GuardianApp()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

