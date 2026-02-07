#!/usr/bin/env python3
# guardian_hybrid_full_platform_swarm_llm_gui_upgraded_patched.py

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

# ---------------------------------------------------------
# NORMAL AUTOLOADER
# ---------------------------------------------------------

REQUIRED_PACKAGES = [
    "PyQt5",
    "psutil",
    "pywin32",
    "cryptography",
    "pyserial",
    "requests",
]

def run_cmd(cmd, log_func=None):
    try:
        if log_func:
            log_func(f"[AUTOLOADER] Running: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        if log_func:
            if result.stdout.strip():
                log_func(f"[AUTOLOADER] stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                log_func(f"[AUTOLOADER] stderr: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        if log_func:
            log_func(f"[AUTOLOADER] Command failed: {e}")
        return False

def install_package(pkg, log_func=None):
    python = sys.executable
    if run_cmd(f'"{python}" -m pip install {pkg}', log_func):
        return True
    if run_cmd(f'"{python}" -m pip install --user {pkg}', log_func):
        return True
    return False

def ensure_package(pkg, log_func=None):
    try:
        importlib.import_module(pkg)
        if log_func:
            log_func(f"[AUTOLOADER] {pkg} already installed.")
        return True
    except ImportError:
        if log_func:
            log_func(f"[AUTOLOADER] {pkg} missing. Attempting normal install...")
        if install_package(pkg, log_func):
            try:
                importlib.invalidate_caches()
                importlib.import_module(pkg)
                if log_func:
                    log_func(f"[AUTOLOADER] {pkg} installed successfully.")
                return True
            except Exception as e:
                if log_func:
                    log_func(f"[AUTOLOADER] {pkg} installed but import failed: {e}")
                return False
        if log_func:
            log_func(f"[AUTOLOADER] FAILED to install {pkg}.")
        return False

def run_autoloader(log_func=None):
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        ok = ensure_package(pkg, log_func)
        if not ok:
            all_ok = False
    return all_ok

def _console_log(msg: str):
    print(msg)

run_autoloader(log_func=_console_log)

# ---------------------------------------------------------
# Imports after autoloader
# ---------------------------------------------------------

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTabWidget, QTextEdit, QCheckBox, QFrame,
        QSpacerItem, QSizePolicy, QFileDialog, QLineEdit, QDoubleSpinBox,
        QSpinBox, QComboBox
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

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    import requests
except ImportError:
    requests = None

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
        return {
            "state_path": None,
            "swarm_endpoint": None,
            "update_url": None,
            "llm_active": None,
            "swarm_node_id": None,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("state_path", None)
        cfg.setdefault("swarm_endpoint", None)
        cfg.setdefault("update_url", None)
        cfg.setdefault("llm_active", None)
        cfg.setdefault("swarm_node_id", None)
        return cfg
    except Exception:
        return {
            "state_path": None,
            "swarm_endpoint": None,
            "update_url": None,
            "llm_active": None,
            "swarm_node_id": None,
        }

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
        return {
            "ai_enabled": False,
            "encoding_enabled": False,
            "auto_move_enabled": False,
            "auto_move_threshold": 0.7,
            "auto_move_cooldown_sec": 30,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        st.setdefault("ai_enabled", False)
        st.setdefault("encoding_enabled", False)
        st.setdefault("auto_move_enabled", False)
        st.setdefault("auto_move_threshold", 0.7)
        st.setdefault("auto_move_cooldown_sec", 30)
        return st
    except Exception:
        return {
            "ai_enabled": False,
            "encoding_enabled": False,
            "auto_move_enabled": False,
            "auto_move_threshold": 0.7,
            "auto_move_cooldown_sec": 30,
        }

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
# Service helpers
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
        return status[1] == 4
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
# SYSTEM Service
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
# File scanner thread
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
# Thermal control thread
# -----------------------------

class ThermalControlThread(QThread):
    temp_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.running = True
        self.ser = None

    def run(self):
        if serial is None:
            self.log_signal.emit("[THERMAL] pyserial not available; cannot open port.")
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.log_signal.emit(f"[THERMAL] Connected to {self.port} @ {self.baud}")
        except Exception as e:
            self.log_signal.emit(f"[THERMAL] Failed to open {self.port}: {e}")
            return

        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("TEMP:"):
                    try:
                        t = float(line.split(":", 1)[1])
                        self.temp_signal.emit(t)
                    except ValueError:
                        pass
            except Exception as e:
                self.log_signal.emit(f"[THERMAL] Error reading: {e}")
                break

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def stop(self):
        self.running = False

    def send_command(self, cmd: str):
        try:
            if self.ser and self.ser.is_open:
                self.ser.write((cmd.strip() + "\n").encode("utf-8"))
        except Exception as e:
            self.log_signal.emit(f"[THERMAL] Failed to send '{cmd}': {e}")

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
# AI + Encoding
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
# Data‑driven Peltier organ
# -----------------------------

class DataPeltier:
    def __init__(
        self,
        ambient=25.0,
        cold_start=25.0,
        hot_start=25.0,
        max_transfer=2.0,
        leak=0.05,
        data_gain=0.1,
    ):
        self.ambient = ambient
        self.cold = cold_start
        self.hot = hot_start
        self.max_transfer = max_transfer
        self.leak = leak
        self.data_gain = data_gain

        self.power = 0.0
        self.direction = 1
        self.heat_load = 0.0

    def set_power(self, pct):
        self.power = max(0.0, min(1.0, pct / 100.0))

    def set_direction(self, cool=True):
        self.direction = 1 if cool else -1

    def feed_data(self, value):
        self.heat_load += value * self.data_gain

    def step(self, dt):
        self.cold += self.heat_load * dt

        transfer = self.max_transfer * self.power * dt
        self.cold -= self.direction * transfer
        self.hot += self.direction * transfer

        self.cold += (self.ambient - self.cold) * self.leak * dt
        self.hot += (self.ambient - self.hot) * self.leak * dt

        self.heat_load *= 0.9

    def snapshot(self):
        return {
            "cold": self.cold,
            "hot": self.hot,
            "ambient": self.ambient,
            "power": self.power,
            "direction": self.direction,
            "heat_load": self.heat_load,
        }

# -----------------------------
# Swarm sync organ (real HTTP)
# -----------------------------

class SwarmSyncOrgan:
    def __init__(self, node_id: str, swarm_endpoint: Optional[str] = None, log_func=None):
        self.node_id = node_id
        self.swarm_endpoint = swarm_endpoint.rstrip("/") if swarm_endpoint else None
        self.log = log_func or (lambda x: None)
        self.incoming_directives: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def is_enabled(self) -> bool:
        return bool(self.swarm_endpoint and requests is not None)

    def publish_event(self, event_type: str, payload: Dict[str, Any]):
        if not self.is_enabled():
            return
        url = f"{self.swarm_endpoint}/events"
        data = {
            "node_id": self.node_id,
            "type": event_type,
            "payload": payload,
            "ts": int(time.time()),
        }
        try:
            resp = requests.post(url, json=data, timeout=3)
            if resp.status_code != 200:
                self.log(f"[SWARM] POST {url} failed: {resp.status_code} {resp.text}")
            else:
                self.log(f"[SWARM] Event published: {event_type}")
        except Exception as e:
            self.log(f"[SWARM] Error publishing event: {e}")

    def fetch_directives(self) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []
        url = f"{self.swarm_endpoint}/directives"
        try:
            resp = requests.get(url, params={"node_id": self.node_id}, timeout=3)
            if resp.status_code != 200:
                self.log(f"[SWARM] GET {url} failed: {resp.status_code} {resp.text}")
                return []
            directives = resp.json()
            if not isinstance(directives, list):
                self.log("[SWARM] Unexpected directives format (expected list).")
                return []
            return directives
        except Exception as e:
            self.log(f"[SWARM] Error fetching directives: {e}")
            return []

    def simulate_incoming_directive(self, directive: Dict[str, Any]):
        with self.lock:
            self.incoming_directives.append(directive)

# -----------------------------
# Threat matrix organ
# -----------------------------

class ThreatMatrixOrgan:
    def __init__(self, log_func=None):
        self.log = log_func or (lambda x: None)
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def record_detection(self, path: str, risk: float, source: str):
        evt = {
            "type": "detection",
            "path": path,
            "risk": risk,
            "source": source,
            "ts": int(time.time()),
        }
        with self.lock:
            self.events.append(evt)
        self.log(f"[THREAT] Detection: {path} risk={risk:.2f} source={source}")

    def record_action(self, action: str, details: Dict[str, Any]):
        evt = {
            "type": "action",
            "action": action,
            "details": details,
            "ts": int(time.time()),
        }
        with self.lock:
            self.events.append(evt)
        self.log(f"[THREAT] Action: {action} {details}")

    def current_score(self) -> float:
        now = time.time()
        score = 0.0
        with self.lock:
            for e in self.events[-200:]:
                if e["type"] == "detection":
                    age = max(1.0, now - e["ts"])
                    weight = 1.0 / age
                    score += e["risk"] * weight
        return min(score, 10.0)

# -----------------------------
# Update organ
# -----------------------------

class UpdateOrgan:
    def __init__(self, current_version: str, update_url: Optional[str] = None, log_func=None):
        self.current_version = current_version
        self.update_url = update_url
        self.log = log_func or (lambda x: None)

    def check_for_updates(self) -> Dict[str, Any]:
        if not self.update_url or requests is None:
            self.log("[UPDATE] No update URL configured or requests missing.")
            return {"status": "disabled"}
        try:
            resp = requests.get(self.update_url, timeout=3)
            if resp.status_code != 200:
                self.log(f"[UPDATE] GET failed: {resp.status_code} {resp.text}")
                return {"status": "error", "message": "http_error"}
            data = resp.json()
            latest = data.get("latest_version", self.current_version)
            update_available = latest != self.current_version
            self.log(f"[UPDATE] Current={self.current_version}, Latest={latest}, Available={update_available}")
            return {
                "status": "ok",
                "latest_version": latest,
                "update_available": update_available,
            }
        except Exception as e:
            self.log(f"[UPDATE] Error checking updates: {e}")
            return {"status": "error", "message": str(e)}

    def download_and_stage(self, version: str) -> Dict[str, Any]:
        self.log(f"[UPDATE] Would download and stage version {version}.")
        return {"status": "ok", "staged_version": version}

# -----------------------------
# LLM registry
# -----------------------------

class LLMBackend:
    def __init__(self, name: str):
        self.name = name

    def analyze_text(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

class DummyLLMBackend(LLMBackend):
    def __init__(self):
        super().__init__("dummy")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        return {"summary": text[:100], "tokens": len(text.split())}

class HeuristicThreatLLMBackend(LLMBackend):
    def __init__(self):
        super().__init__("heuristic")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        tokens = text.lower().split()
        score = 0
        for t in tokens:
            if "suspicious" in t or "risk" in t or "quarantine" in t:
                score += 1
            if "error" in t or "failed" in t:
                score += 0.5
        threat_level = min(10, score)
        comment = "Low activity" if threat_level < 3 else "Elevated" if threat_level < 7 else "High"
        return {
            "tokens": len(tokens),
            "threat_level": threat_level,
            "comment": comment,
        }

class LLMRegistry:
    def __init__(self, log_func=None):
        self.log = log_func or (lambda x: None)
        self.backends: Dict[str, LLMBackend] = {}
        self.active_name: Optional[str] = None
        self.lock = threading.Lock()

    def register(self, backend: LLMBackend):
        with self.lock:
            self.backends[backend.name] = backend
            if self.active_name is None:
                self.active_name = backend.name
        self.log(f"[LLM] Registered backend: {backend.name}")

    def set_active(self, name: str):
        with self.lock:
            if name in self.backends:
                self.active_name = name
                self.log(f"[LLM] Active backend set to: {name}")
            else:
                self.log(f"[LLM] Unknown backend: {name}")

    def active(self) -> Optional[LLMBackend]:
        with self.lock:
            if self.active_name is None:
                return None
            return self.backends.get(self.active_name)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        backend = self.active()
        if not backend:
            self.log("[LLM] No active backend.")
            return {"error": "no_active_backend"}
        return backend.analyze_text(text)

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
# Serial auto-detect
# -----------------------------

def score_serial_port(port) -> int:
    score = 0
    desc = (port.description or "").lower()
    manu = (port.manufacturer or "").lower()
    hwid = (port.hwid or "").lower()
    vidpid = hwid

    known_patterns = [
        "2341:0043",
        "2341:0001",
        "1a86:7523",
        "10c4:ea60",
        "2e8a:000a",
        "0403:6001",
    ]
    for pat in known_patterns:
        if pat in vidpid:
            score += 3
            break

    hints = ["arduino", "esp32", "esp8266", "pico", "ch340", "cp210", "usb-serial", "ftdi"]
    if any(h in desc for h in hints):
        score += 2
    if any(h in manu for h in hints):
        score += 2

    return score

def auto_detect_thermal_port(log_func=None) -> Optional[str]:
    log = log_func or (lambda x: None)
    if serial is None:
        log("[THERMAL] pyserial not available; cannot scan ports.")
        return None

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        log("[THERMAL] No serial ports found.")
        return None

    best_port = None
    best_score = -1

    for p in ports:
        s = score_serial_port(p)
        log(f"[THERMAL] Port {p.device} score={s} desc='{p.description}' manu='{p.manufacturer}' hwid='{p.hwid}'")
        if s > best_score:
            best_score = s
            best_port = p

    if best_port is None:
        log("[THERMAL] No suitable port found.")
        return None

    log(f"[THERMAL] Selected port {best_port.device} with score={best_score}")
    return best_port.device

# -----------------------------
# GUI main app (start)
# -----------------------------

class GuardianApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian Hybrid EDR - SYSTEM + AI + Encoding + Thermal + Swarm + LLM")
        self.resize(1400, 880)

        # Early log buffer (for safe logging before log_view exists)
        self._early_logs: List[str] = []

        self.config = load_config()
        self.state = load_state(self.config)
        self.state_dir = resolve_state_dir(self.config)

        # DO NOT create DeepScrubClient yet (it uses append_log).
        # Wait until after UI is initialized so log_view exists.

        self.freeze_active = False
        self.outbound_shield_active = True
        self.selected_path = ""

        self.service_watchdog_enabled = True
        self.last_service_error = "None"
        self.last_service_action = "None"

        self.file_move_suggestions: List[Dict[str, Any]] = []
        self.rollback_entries: List[Dict[str, Any]] = []
        self.auto_move_last_times: List[float] = []

        self.file_move_lock = threading.Lock()
        self.rollback_lock = threading.Lock()

        self.ai_engine = SimpleAIEngine()
        self.encoding_engine = EncodingEngine()
        self.ai_enabled = bool(self.state.get("ai_enabled", False))
        self.encoding_enabled = bool(self.state.get("encoding_enabled", False))

        self.auto_move_enabled = bool(self.state.get("auto_move_enabled", False))
        self.auto_move_threshold = float(self.state.get("auto_move_threshold", 0.7))
        self.auto_move_cooldown_sec = int(self.state.get("auto_move_cooldown_sec", 30))

        self.thermal_thread: Optional[ThermalControlThread] = None
        self.current_temp: Optional[float] = None
        self.thermal_port: Optional[str] = None

        self.data_peltier = DataPeltier(
            ambient=25.0,
            cold_start=25.0,
            hot_start=25.0,
            max_transfer=1.5,
            leak=0.02,
            data_gain=0.02,
        )
        self.data_peltier.set_power(70)
        self.data_peltier.set_direction(cool=True)

        self.node_id = self.config.get("swarm_node_id") or socket.gethostname()
        self.swarm = SwarmSyncOrgan(
            node_id=self.node_id,
            swarm_endpoint=self.config.get("swarm_endpoint"),
            log_func=self.append_log
        )

        self.threat_matrix = ThreatMatrixOrgan(log_func=self.append_log)
        self.current_version = "1.0.0"
        self.updates = UpdateOrgan(
            current_version=self.current_version,
            update_url=self.config.get("update_url"),
            log_func=self.append_log
        )

        self.llms = LLMRegistry(log_func=self.append_log)
        self.llms.register(DummyLLMBackend())
        self.llms.register(HeuristicThreatLLMBackend())
        if self.config.get("llm_active"):
            self.llms.set_active(self.config["llm_active"])

        # Build UI first so log_view exists
        self.init_ui()

        # Now it's safe to create DeepScrubClient using append_log
        self.deep_client = DeepScrubClient(log_func=self.append_log)

        self.start_background_threads()
        self.init_scanner()
        self.check_system_service_status()
        self.init_service_watchdog()
        self.init_quarantine_analyzer()
        self.init_swarm_timer()
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
        self.swarm_tab = QWidget()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.controls_tab, "Controls")
        self.tabs.addTab(self.telemetry_tab, "SYSTEM / Telemetry")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")
        self.tabs.addTab(self.swarm_tab, "Swarm")

        self.init_dashboard_tab()
        self.init_controls_tab()
        self.init_telemetry_tab()
        self.init_logs_tab()
        self.init_settings_tab()
        self.init_diagnostics_tab()
        self.init_swarm_tab()

    def init_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)

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

        top_row = QHBoxLayout()

        status_frame, status_layout = make_section_frame("Status")
        self.status_label = QLabel("Guardian Hybrid EDR: Online")
        self.status_label.setFont(QFont("Segoe UI", 11))
        status_layout.addWidget(self.status_label)

        self.threat_level_label = QLabel("Threat Level: 0.00 / 10.00")
        self.threat_level_label.setFont(QFont("Consolas", 10))
        status_layout.addWidget(self.threat_level_label)

        top_row.addWidget(status_frame)

        stats_frame, stats_layout = make_section_frame("Live Metrics")
        self.cpu_label = QLabel("CPU: -- %")
        self.mem_label = QLabel("Memory: -- %")
        self.net_label = QLabel("Net I/O: -- bytes")
        for lbl in (self.cpu_label, self.mem_label, self.net_label):
            lbl.setFont(QFont("Consolas", 10))
            stats_layout.addWidget(lbl)
        top_row.addWidget(stats_frame)

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
        self.storage_path_button.setToolTip("Select local or SMB folder for persistent AI/Encoding state storage\nCurrent: Script folder")
        self.storage_path_button.clicked.connect(self.on_storage_path_button_clicked)
        top_row.addWidget(self.storage_path_button)

        layout.addLayout(top_row)

        events_frame, events_layout = make_section_frame("Recent Events")
        self.events_log = QTextEdit()
        self.events_log.setReadOnly(True)
        events_layout.addWidget(self.events_log)
        layout.addWidget(events_frame)

        thermal_frame, thermal_layout = make_section_frame("Thermal Control (Peltier)")
        self.thermal_status_label = QLabel("Thermal: DISCONNECTED")
        self.thermal_temp_label = QLabel("Temp: -- °C")

        self.thermal_connect_button = QPushButton("Auto-Connect Thermal Device")
        self.thermal_connect_button.clicked.connect(self.on_thermal_connect)

        self.thermal_on_button = QPushButton("Peltier ON")
        self.thermal_on_button.clicked.connect(lambda: self.send_thermal_command("ON"))

        self.thermal_off_button = QPushButton("Peltier OFF")
        self.thermal_off_button.clicked.connect(lambda: self.send_thermal_command("OFF"))

        self.thermal_power_button = QPushButton("Set Power 50%")
        self.thermal_power_button.clicked.connect(lambda: self.send_thermal_command("POWER 50"))

        self.virtual_peltier_label = QLabel("Virtual Peltier: cold=-- °C, hot=-- °C")

        thermal_layout.addWidget(self.thermal_status_label)
        thermal_layout.addWidget(self.thermal_temp_label)
        thermal_layout.addWidget(self.thermal_connect_button)
        thermal_layout.addWidget(self.thermal_on_button)
        thermal_layout.addWidget(self.thermal_off_button)
        thermal_layout.addWidget(self.thermal_power_button)
        thermal_layout.addWidget(self.virtual_peltier_label)

        layout.addWidget(thermal_frame)

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

        # Flush any early logs that were buffered before log_view existed
        if hasattr(self, "_early_logs") and self._early_logs:
            for msg in self._early_logs:
                self.log_view.append(msg)
            self._early_logs = []

    def init_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)

        autoloader_frame, autoloader_layout = make_section_frame("Autoloader Status")
        self.autoloader_status_label = QLabel("Autoloader: Normal mode (no auto-restart).")
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

        update_frame, update_layout = make_section_frame("Updates")
        self.update_check_button = QPushButton("Check for Updates")
        self.update_check_button.clicked.connect(self.on_check_updates)
        update_layout.addWidget(self.update_check_button)
        layout.addWidget(update_frame)

        llm_frame, llm_layout = make_section_frame("LLM Backend")
        self.llm_selector = QComboBox()
        for name in self.llms.backends.keys():
            self.llm_selector.addItem(name)
        self.llm_selector.currentTextChanged.connect(self.on_llm_changed)
        llm_layout.addWidget(QLabel("Active LLM:"))
        llm_layout.addWidget(self.llm_selector)

        self.llm_input = QTextEdit()
        self.llm_input.setPlaceholderText("Enter text to analyze with selected LLM...")
        self.llm_analyze_button = QPushButton("Analyze Text with LLM")
        self.llm_analyze_button.clicked.connect(self.on_llm_analyze)
        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)

        llm_layout.addWidget(self.llm_input)
        llm_layout.addWidget(self.llm_analyze_button)
        llm_layout.addWidget(QLabel("LLM Output:"))
        llm_layout.addWidget(self.llm_output)

        layout.addWidget(llm_frame)

        swarm_cfg_frame, swarm_cfg_layout = make_section_frame("Swarm Configuration")
        self.swarm_endpoint_input = QLineEdit()
        self.swarm_endpoint_input.setPlaceholderText("Swarm endpoint base URL (e.g. https://swarm/api)")
        if self.config.get("swarm_endpoint"):
            self.swarm_endpoint_input.setText(self.config["swarm_endpoint"])

        self.swarm_node_id_input = QLineEdit()
        self.swarm_node_id_input.setPlaceholderText("Node ID override (optional)")
        if self.config.get("swarm_node_id"):
            self.swarm_node_id_input.setText(self.config["swarm_node_id"])
        else:
            self.swarm_node_id_input.setText(self.node_id)

        self.swarm_save_button = QPushButton("Save Swarm Settings")
        self.swarm_save_button.clicked.connect(self.on_swarm_save_settings)

        self.swarm_heartbeat_button = QPushButton("Send Swarm Heartbeat")
        self.swarm_heartbeat_button.clicked.connect(self.on_swarm_heartbeat)

        self.swarm_test_event_button = QPushButton("Publish Test Event")
        self.swarm_test_event_button.clicked.connect(self.on_swarm_test_event)

        swarm_cfg_layout.addWidget(QLabel("Swarm endpoint:"))
        swarm_cfg_layout.addWidget(self.swarm_endpoint_input)
        swarm_cfg_layout.addWidget(QLabel("Node ID:"))
        swarm_cfg_layout.addWidget(self.swarm_node_id_input)
        swarm_cfg_layout.addWidget(self.swarm_save_button)
        swarm_cfg_layout.addWidget(self.swarm_heartbeat_button)
        swarm_cfg_layout.addWidget(self.swarm_test_event_button)

        layout.addWidget(swarm_cfg_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def init_diagnostics_tab(self):
        layout = QVBoxLayout(self.diagnostics_tab)

        svc_frame, svc_layout = make_section_frame("SYSTEM Service State")
        self.diag_installed_label = QLabel("Installed: unknown")
        self.diag_running_label = QLabel("Running: unknown")
        self.diag_last_action_label = QLabel("Last action: None")
        self.diag_last_error_label = QLabel("Last error: None")
        self.diag_service_pid_label = QLabel("Service PID: unknown")
        for lbl in (self.diag_installed_label, self.diag_running_label,
                    self.diag_last_action_label, self.diag_last_error_label,
                    self.diag_service_pid_label):
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

        self.diag_restart_button = QPushButton("Restart Service")
        self.diag_restart_button.clicked.connect(self.on_restart_service)

        self.diag_watchdog_toggle = QCheckBox("Enable SYSTEM Service Watchdog (auto-repair)")
        self.diag_watchdog_toggle.setChecked(True)
        self.diag_watchdog_toggle.stateChanged.connect(self.toggle_watchdog)

        self.auto_move_toggle = QCheckBox("Enable Automatic File Move Mode")
        self.auto_move_toggle.setChecked(self.auto_move_enabled)
        self.auto_move_toggle.stateChanged.connect(self.toggle_auto_move_mode)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Auto-move risk threshold:"))
        self.auto_move_threshold_spin = QDoubleSpinBox()
        self.auto_move_threshold_spin.setRange(0.0, 1.0)
        self.auto_move_threshold_spin.setSingleStep(0.05)
        self.auto_move_threshold_spin.setValue(self.auto_move_threshold)
        self.auto_move_threshold_spin.valueChanged.connect(self.on_auto_move_threshold_changed)
        threshold_row.addWidget(self.auto_move_threshold_spin)

        cooldown_row = QHBoxLayout()
        cooldown_row.addWidget(QLabel("Auto-move cooldown (seconds):"))
        self.auto_move_cooldown_spin = QSpinBox()
        self.auto_move_cooldown_spin.setRange(0, 600)
        self.auto_move_cooldown_spin.setValue(self.auto_move_cooldown_sec)
        self.auto_move_cooldown_spin.valueChanged.connect(self.on_auto_move_cooldown_changed)
        cooldown_row.addWidget(self.auto_move_cooldown_spin)

        self.rollback_button = QPushButton("Rollback Last Move")
        self.rollback_button.clicked.connect(self.on_rollback_last_move)

        self.show_quarantine_button = QPushButton("Open Quarantine Folder")
        self.show_quarantine_button.clicked.connect(self.on_open_quarantine_folder)

        btn_layout.addWidget(self.diag_refresh_button)
        btn_layout.addWidget(self.diag_force_reinstall_button)
        btn_layout.addWidget(self.diag_start_button)
        btn_layout.addWidget(self.diag_stop_button)
        btn_layout.addWidget(self.diag_restart_button)
        btn_layout.addWidget(self.diag_watchdog_toggle)
        btn_layout.addWidget(self.auto_move_toggle)
        btn_layout.addLayout(threshold_row)
        btn_layout.addLayout(cooldown_row)
        btn_layout.addWidget(self.rollback_button)
        btn_layout.addWidget(self.show_quarantine_button)
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

        log_frame, log_layout = make_section_frame("Diagnostics Log (SYSTEM / Watchdog)")
        self.diag_log = QTextEdit()
        self.diag_log.setReadOnly(True)
        log_layout.addWidget(self.diag_log)
        layout.addWidget(log_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.refresh_diagnostics()
        self.refresh_file_move_suggestions()

    def init_swarm_tab(self):
        layout = QVBoxLayout(self.swarm_tab)

        swarm_frame, swarm_layout = make_section_frame("Swarm Directives")
        self.swarm_log = QTextEdit()
        self.swarm_log.setReadOnly(True)
        self.swarm_fetch_button = QPushButton("Fetch Swarm Directives Now")
        self.swarm_fetch_button.clicked.connect(self.on_swarm_fetch)
        swarm_layout.addWidget(self.swarm_log)
        swarm_layout.addWidget(self.swarm_fetch_button)
        layout.addWidget(swarm_frame)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    # ---------- Threads / timers ----------

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

    def init_quarantine_analyzer(self):
        self.quarantine_timer = QTimer(self)
        self.quarantine_timer.setInterval(60000)
        self.quarantine_timer.timeout.connect(self.quarantine_tick)
        self.quarantine_timer.start()

    def init_swarm_timer(self):
        self.swarm_timer = QTimer(self)
        self.swarm_timer.setInterval(30000)
        self.swarm_timer.timeout.connect(self.swarm_tick)
        self.swarm_timer.start()

    # ---------- Periodic tasks ----------

    def quarantine_tick(self):
        quarantine_root = "C:\\Quarantine"
        if not os.path.exists(quarantine_root):
            return
        count = 0
        for root, dirs, files in os.walk(quarantine_root):
            count += len(files)
        self.diag_log.append(f"[QUARANTINE] Files currently in quarantine: {count}")

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

    def swarm_tick(self):
        if not self.swarm.is_enabled():
            return
        directives = self.swarm.fetch_directives()
        if not directives:
            return
        for d in directives:
            self.swarm_log.append(f"[SWARM] Directive: {d}")
            self.events_log.append(f"[SWARM] Directive received: {d}")

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
        self.diag_service_pid_label.setText(f"Service PID: {port_info.get('pid')}")
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

    def on_restart_service(self):
        self.diag_log.append("[DIAG] Restart service requested.")
        self.last_service_action = "Restart service"
        if win32serviceutil is None:
            self.last_service_error = "pywin32 missing"
            self.diag_log.append("[DIAG] Cannot restart service; pywin32 missing.")
            return
        try:
            if service_is_running():
                win32serviceutil.StopService(SERVICE_NAME)
                time.sleep(1.0)
            win32serviceutil.StartService(SERVICE_NAME)
            time.sleep(2.0)
            self.last_service_error = "None"
            self.diag_log.append("[DIAG] Service restart requested.")
        except Exception as e:
            self.last_service_error = str(e)
            self.diag_log.append(f"[DIAG] Restart service failed: {e}")
        self.refresh_diagnostics()
        self.check_system_service_status(auto=True)

    def toggle_watchdog(self, state):
        self.service_watchdog_enabled = (state == Qt.Checked)
        status = "ENABLED" if self.service_watchdog_enabled else "DISABLED"
        self.diag_log.append(f"[WATCHDOG] SYSTEM service watchdog {status}.")
        self.append_log(f"[WATCHDOG] SYSTEM service watchdog {status}.")

    # ---------- Controls / AI / Encoding / Storage / Auto-move ----------

    def apply_persistent_states_to_ui(self):
        self.ai_button.blockSignals(True)
        self.ai_button.setChecked(self.ai_enabled)
        self.ai_button.setText("AI: ON" if self.ai_enabled else "AI: OFF")
        self.ai_button.blockSignals(False)

        self.encoding_button.blockSignals(True)
        self.encoding_button.setChecked(self.encoding_enabled)
        self.encoding_button.setText("Encoding: ON" if self.encoding_enabled else "Encoding: OFF")
        self.encoding_button.blockSignals(False)

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

        if hasattr(self, "auto_move_toggle"):
            self.auto_move_toggle.blockSignals(True)
            self.auto_move_toggle.setChecked(self.auto_move_enabled)
            self.auto_move_toggle.blockSignals(False)

        if hasattr(self, "auto_move_threshold_spin"):
            self.auto_move_threshold_spin.blockSignals(True)
            self.auto_move_threshold_spin.setValue(self.auto_move_threshold)
            self.auto_move_threshold_spin.blockSignals(False)

        if hasattr(self, "auto_move_cooldown_spin"):
            self.auto_move_cooldown_spin.blockSignals(True)
            self.auto_move_cooldown_spin.setValue(self.auto_move_cooldown_sec)
            self.auto_move_cooldown_spin.blockSignals(False)

        active_llm = self.config.get("llm_active") or self.llms.active().name
        idx = self.llm_selector.findText(active_llm)
        if idx >= 0:
            self.llm_selector.setCurrentIndex(idx)

    def update_dashboard_stats(self, stats: dict):
        cpu = stats.get("cpu", 0)
        mem = stats.get("mem", 0)
        net = stats.get("net", 0)

        self.cpu_label.setText(f"CPU: {cpu:.1f} %")
        self.mem_label.setText(f"Memory: {mem:.1f} %")
        self.net_label.setText(f"Net I/O: {net} bytes")
        self.events_log.append(f"[STATS] CPU={cpu:.1f}% MEM={mem:.1f}% NET={net} bytes")

        heat_input = cpu + mem + (net / 1_000_000.0)
        self.data_peltier.feed_data(heat_input)
        self.data_peltier.step(1.0)

        snap = self.data_peltier.snapshot()
        self.virtual_peltier_label.setText(
            f"Virtual Peltier: cold={snap['cold']:.2f} °C, hot={snap['hot']:.2f} °C"
        )

        threat_score = self.threat_matrix.current_score()
        self.threat_level_label.setText(f"Threat Level: {threat_score:.2f} / 10.00")

    # ---------- PATCHED SAFE LOGGER ----------

    def append_log(self, message: str):
        """
        Safe logging:
        - Before log_view exists: buffer messages in _early_logs
        - After log_view exists: append normally and flush buffer
        """
        if not hasattr(self, "log_view") or self.log_view is None:
            if not hasattr(self, "_early_logs"):
                self._early_logs = []
            self._early_logs.append(message)
            return

        self.log_view.append(message)

        if hasattr(self, "_early_logs") and self._early_logs:
            for msg in self._early_logs:
                self.log_view.append(msg)
            self._early_logs = []

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

    def toggle_auto_move_mode(self, state):
        self.auto_move_enabled = (state == Qt.Checked)
        self.state["auto_move_enabled"] = self.auto_move_enabled
        save_state(self.config, self.state)
        mode = "ENABLED" if self.auto_move_enabled else "DISABLED"
        self.diag_log.append(f"[AUTO-MOVE] Automatic file move mode {mode}.")
        self.events_log.append(f"[AUTO-MOVE] Automatic file move mode {mode}.")

    def on_auto_move_threshold_changed(self, value: float):
        self.auto_move_threshold = float(value)
        self.state["auto_move_threshold"] = self.auto_move_threshold
        save_state(self.config, self.state)
        self.diag_log.append(f"[AUTO-MOVE] Threshold set to {self.auto_move_threshold:.2f}")

    def on_auto_move_cooldown_changed(self, value: int):
        self.auto_move_cooldown_sec = int(value)
        self.state["auto_move_cooldown_sec"] = self.auto_move_cooldown_sec
        save_state(self.config, self.state)
        self.diag_log.append(f"[AUTO-MOVE] Cooldown set to {self.auto_move_cooldown_sec} seconds")

    def select_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Local/SMB Folder", "")
        if path:
            self.selected_path = path
            self.path_display.setText(path)
            self.append_log(f"[SETTINGS] Path selected: {path}")

    def on_storage_path_button_clicked(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select AI/Encoding State Folder", self.state_dir)
        if not new_dir:
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
            pass

        if new_dir == SCRIPT_DIR:
            self.config["state_path"] = None
        else:
            self.config["state_path"] = new_dir
        save_config(self.config)

        self.state_dir = resolve_state_dir(self.config)
        self.apply_persistent_states_to_ui()

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

    def on_check_updates(self):
        result = self.updates.check_for_updates()
        self.events_log.append(f"[UPDATE] Check result: {result}")

    def on_llm_changed(self, name: str):
        self.llms.set_active(name)
        self.config["llm_active"] = name
        save_config(self.config)
        self.events_log.append(f"[LLM] Active backend changed to: {name}")

    def on_llm_analyze(self):
        text = self.llm_input.toPlainText().strip()
        if not text:
            self.llm_output.setPlainText("No text provided.")
            return
        result = self.llms.analyze_text(text)
        self.llm_output.setPlainText(json.dumps(result, indent=2))
        self.events_log.append(f"[LLM] Analyzed text with backend '{self.llms.active().name}'.")

    # ---------- Swarm config actions ----------

    def on_swarm_save_settings(self):
        endpoint = self.swarm_endpoint_input.text().strip()
        node_id = self.swarm_node_id_input.text().strip() or socket.gethostname()

        self.config["swarm_endpoint"] = endpoint if endpoint else None
        self.config["swarm_node_id"] = node_id
        save_config(self.config)

        self.node_id = node_id
        self.swarm = SwarmSyncOrgan(
            node_id=self.node_id,
            swarm_endpoint=self.config.get("swarm_endpoint"),
            log_func=self.append_log
        )
        self.events_log.append(f"[SWARM] Settings saved. Endpoint={endpoint}, NodeID={node_id}")

    def on_swarm_heartbeat(self):
        if not self.swarm.is_enabled():
            self.events_log.append("[SWARM] Heartbeat skipped (swarm disabled or no endpoint).")
            return
        self.swarm.publish_event("heartbeat", {"status": "online"})
        self.events_log.append("[SWARM] Heartbeat event published.")

    def on_swarm_test_event(self):
        if not self.swarm.is_enabled():
            self.events_log.append("[SWARM] Test event skipped (swarm disabled or no endpoint).")
            return
        self.swarm.publish_event("test_event", {"message": "This is a test event from Guardian."})
        self.events_log.append("[SWARM] Test event published.")

    # ---------- Scanner → AI + Swarm + Threat + Auto-move ----------

    def on_scanner_detect(self, info: Dict[str, Any]):
        path = info["path"]
        self.events_log.append(f"[SCANNER] Suspicious file detected: {path}")

        if self.ai_enabled:
            ai_result = self.ai_engine.analyze_file(path)
            risk = ai_result.get("risk_score", 0.0)
            self.events_log.append(f"[AI] Risk score for {path}: {risk:.2f}")
        else:
            risk = 0.5

        self.threat_matrix.record_detection(path=path, risk=risk, source="scanner")
        self.swarm.publish_event("suspicious_file", {"path": path, "risk": risk})

        quarantine_root = "C:\\Quarantine"
        dst = os.path.join(quarantine_root, os.path.basename(path))
        reason = f"Scanner detection (risk={risk:.2f})"
        self.add_file_move_suggestion(path, dst, reason, risk=risk)

        if self.encoding_enabled:
            try:
                with open(path, "rb") as fp:
                    data = fp.read(1024).decode("utf-8", errors="ignore")
                encoded = self.encoding_engine.encode(data)
                self.events_log.append(f"[ENC] Sample encoded preview for {path}: {encoded[:120]}...")
            except Exception as e:
                self.events_log.append(f"[ENC] Failed to encode sample from {path}: {e}")

    # ---------- File move / auto-move / rollback (thread-safe) ----------

    def _auto_move_cooldown_ok(self) -> bool:
        if self.auto_move_cooldown_sec <= 0:
            return True
        now = time.time()
        with self.file_move_lock:
            self.auto_move_last_times = [t for t in self.auto_move_last_times if now - t <= self.auto_move_cooldown_sec]
            if self.auto_move_last_times and (now - self.auto_move_last_times[-1]) < self.auto_move_cooldown_sec:
                return False
        return True

    def _record_auto_move_time(self):
        with self.file_move_lock:
            self.auto_move_last_times.append(time.time())

    def add_file_move_suggestion(self, src: str, dst: str, reason: str, risk: Optional[float] = None):
        suggestion = {"src": src, "dst": dst, "reason": reason, "risk": risk}

        if self.auto_move_enabled and risk is not None and risk >= self.auto_move_threshold and self._auto_move_cooldown_ok():
            self.diag_log.append(f"[AUTO-MOVE] Auto-executing: {src} -> {dst} ({reason}, risk={risk:.2f})")
            result = self.deep_client.move_file(src, dst)
            self.diag_log.append(f"[AUTO-MOVE] Result: {result}")

            if result.get("status") == "ok":
                self.events_log.append(f"[AUTO-MOVE] File moved automatically: {src} -> {dst}")
                self._record_auto_move_time()
                with self.rollback_lock:
                    self.rollback_entries.append({
                        "src": src,
                        "dst": dst,
                        "time": time.time(),
                        "reason": reason,
                        "risk": risk,
                    })
                self.threat_matrix.record_action("auto_quarantine", {
                    "src": src,
                    "dst": dst,
                    "risk": risk,
                })
            else:
                self.events_log.append(f"[AUTO-MOVE] FAILED to auto-move {src}: {result}")
            return

        if self.auto_move_enabled and risk is not None and risk < self.auto_move_threshold:
            self.diag_log.append(f"[AUTO-MOVE] Risk {risk:.2f} below threshold {self.auto_move_threshold:.2f}; queuing manual suggestion.")

        if self.auto_move_enabled and not self._auto_move_cooldown_ok():
            self.diag_log.append(f"[AUTO-MOVE] Cooldown active ({self.auto_move_cooldown_sec}s); queuing manual suggestion.")

        with self.file_move_lock:
            self.file_move_suggestions.append(suggestion)
        self.refresh_file_move_suggestions()
        self.diag_log.append(f"[FILE] Suggested move: {src} -> {dst} ({reason})")

    def refresh_file_move_suggestions(self):
        self.suggested_file_list.clear()
        with self.file_move_lock:
            suggestions = list(self.file_move_suggestions)
        if not suggestions:
            self.suggested_file_list.append("No pending suggestions.")
            return
        for idx, s in enumerate(suggestions, start=1):
            risk_str = f", risk={s['risk']:.2f}" if s.get("risk") is not None else ""
            self.suggested_file_list.append(
                f"{idx}. {s['src']} -> {s['dst']}  [Reason: {s['reason']}{risk_str}]"
            )

    def on_manual_suggest_move(self):
        src = self.manual_file_input.text().strip()
        dst_root = self.manual_quarantine_input.text().strip() or "C:\\Quarantine"
        if not src:
            self.diag_log.append("[FILE] Manual suggest ignored (no source path).")
            return
        dst = os.path.join(dst_root, os.path.basename(src))
        self.add_file_move_suggestion(src=src, dst=dst, reason="Manual operator suggestion", risk=None)

    def on_apply_selected_move(self):
        with self.file_move_lock:
            if not self.file_move_suggestions:
                self.diag_log.append("[FILE] No suggestions to apply.")
                return
            suggestion = self.file_move_suggestions.pop(0)
        src = suggestion["src"]
        dst = suggestion["dst"]
        reason = suggestion["reason"]
        risk = suggestion.get("risk")

        self.diag_log.append(f"[FILE] Executing move via SYSTEM: {src} -> {dst} ({reason})")
        result = self.deep_client.move_file(src, dst)
        self.diag_log.append(f"[FILE] Move result: {result}")

        if result.get("status") == "ok":
            self.events_log.append(f"[SYSTEM] File moved: {src} -> {dst}")
            with self.rollback_lock:
                self.rollback_entries.append({
                    "src": src,
                    "dst": dst,
                    "time": time.time(),
                    "reason": reason,
                    "risk": risk,
                })
            self.threat_matrix.record_action("manual_quarantine", {
                "src": src,
                "dst": dst,
                "risk": risk,
            })
        else:
            self.events_log.append(f"[SYSTEM] File move FAILED for {src}: {result}")

        self.refresh_file_move_suggestions()

    def on_rollback_last_move(self):
        with self.rollback_lock:
            if not self.rollback_entries:
                self.diag_log.append("[ROLLBACK] No moves to rollback.")
                return
            entry = self.rollback_entries.pop()
        src = entry["src"]
        dst = entry["dst"]
        self.diag_log.append(f"[ROLLBACK] Attempting rollback: {dst} -> {src}")
        result = self.deep_client.move_file(dst, src)
        self.diag_log.append(f"[ROLLBACK] Result: {result}")
        if result.get("status") == "ok":
            self.events_log.append(f"[ROLLBACK] Restored file from {dst} to {src}")
            self.threat_matrix.record_action("rollback", {"src": src, "dst": dst})
        else:
            self.events_log.append(f"[ROLLBACK] FAILED to restore {dst} to {src}: {result}")

    def on_open_quarantine_folder(self):
        quarantine_root = "C:\\Quarantine"
        if not os.path.exists(quarantine_root):
            try:
                os.makedirs(quarantine_root, exist_ok=True)
            except Exception as e:
                self.diag_log.append(f"[QUARANTINE] Failed to create folder: {e}")
                return
        try:
            os.startfile(quarantine_root)
        except Exception as e:
            self.diag_log.append(f"[QUARANTINE] Failed to open folder: {e}")

    # ---------- Thermal organ ----------

    def on_thermal_connect(self):
        self.events_log.append("[THERMAL] Auto-detecting thermal device...")
        port = auto_detect_thermal_port(log_func=self.append_log)
        if not port:
            self.events_log.append("[THERMAL] No suitable thermal device found.")
            self.thermal_status_label.setText("Thermal: DISCONNECTED (no device)")
            return

        self.thermal_port = port
        self.events_log.append(f"[THERMAL] Using port {self.thermal_port}")

        try:
            if self.thermal_thread and self.thermal_thread.isRunning():
                self.thermal_thread.stop()
                self.thermal_thread.wait(2000)
        except Exception:
            pass

        self.thermal_thread = ThermalControlThread(self.thermal_port)
        self.thermal_thread.temp_signal.connect(self.on_thermal_temp_update)
        self.thermal_thread.log_signal.connect(self.append_log)
        self.thermal_thread.start()

        self.thermal_status_label.setText(f"Thermal: CONNECTED ({self.thermal_port})")

    def on_thermal_temp_update(self, temp: float):
        self.current_temp = temp
        self.thermal_temp_label.setText(f"Temp: {temp:.1f} °C")
        self.events_log.append(f"[THERMAL] Temp update: {temp:.1f} °C")

    def send_thermal_command(self, cmd: str):
        if not self.thermal_thread or not self.thermal_thread.isRunning():
            self.events_log.append("[THERMAL] Cannot send command; not connected.")
            return
        self.events_log.append(f"[THERMAL] Sending command: {cmd}")
        self.thermal_thread.send_command(cmd)

    # ---------- Swarm tab actions ----------

    def on_swarm_fetch(self):
        directives = self.swarm.fetch_directives()
        if not directives:
            self.swarm_log.append("[SWARM] No directives received.")
            return
        for d in directives:
            self.swarm_log.append(f"[SWARM] Directive: {d}")
            self.events_log.append(f"[SWARM] Directive received: {d}")

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
        try:
            if hasattr(self, "thermal_thread") and self.thermal_thread and self.thermal_thread.isRunning():
                self.thermal_thread.stop()
                self.thermal_thread.wait(2000)
        except Exception:
            pass
        event.accept()

# -----------------------------
# Entry point
# -----------------------------

def main():
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

