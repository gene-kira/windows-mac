import os
import json
import pickle
import threading
import time
import subprocess
import socket
import random
from typing import Any, Dict, Optional, Tuple, List

import psutil
import requests
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox, QListWidgetItem

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

# ============================================================
# GLOBAL CONFIG
# ============================================================

CONFIG_FILE = "backplane_config.json"

LOCAL_MEMORY_FILE = "copilot_memory.dat"
SMB_MEMORY_FILE = None
VRAM_ALLOCATION = 8 * 1024**3   # 8 GB
RAM_ALLOCATION = 20 * 1024**3   # 20 GB
AUTO_BACKUP_MINUTES = 10        # default auto-backup interval

MANAGER_HOST = "127.0.0.1"
MANAGER_PORT = 443  # will be overridden by find_open_port + config
MANAGER_BASE_URL = None  # set after port is known

TELEMETRY_MAX_EVENTS = 200

# Prediction horizon / windows
PRED_WINDOW = 30       # how many samples to keep
EWMA_ALPHA = 0.3       # smoothing factor

# Meta-states
META_CALM = "CALM"
META_FOCUSED = "FOCUSED"
META_BURST = "BURST"
META_OVERLOAD = "OVERLOAD"
META_SAFE = "SAFE_MODE"


# ============================================================
# PORT SELECTION
# ============================================================

def find_open_port(preferred_port=443):
    # Try preferred port first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((MANAGER_HOST, preferred_port))
            return preferred_port
        except OSError:
            pass

    # If preferred port is taken, pick a random port
    while True:
        port = random.randint(20000, 40000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((MANAGER_HOST, port))
                return port
            except OSError:
                continue


# ============================================================
# CONFIG LOAD/SAVE
# ============================================================

def load_config():
    global LOCAL_MEMORY_FILE, SMB_MEMORY_FILE, AUTO_BACKUP_MINUTES
    global MANAGER_PORT, MANAGER_BASE_URL

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            LOCAL_MEMORY_FILE = cfg.get("local_backup_path", LOCAL_MEMORY_FILE)
            SMB_MEMORY_FILE = cfg.get("smb_backup_path", SMB_MEMORY_FILE)
            AUTO_BACKUP_MINUTES = cfg.get("auto_backup_minutes", AUTO_BACKUP_MINUTES)
            MANAGER_PORT = cfg.get("manager_port", find_open_port(443))
            print(f"[MANAGER] Config loaded from {CONFIG_FILE}")
        except Exception as e:
            print("[MANAGER] Failed to load config, using defaults:", e)
            MANAGER_PORT = find_open_port(443)
    else:
        MANAGER_PORT = find_open_port(443)
        save_config()
        print("[MANAGER] No config found, created default config")

    MANAGER_BASE_URL = f"http://{MANAGER_HOST}:{MANAGER_PORT}"


def save_config():
    cfg = {
        "local_backup_path": LOCAL_MEMORY_FILE,
        "smb_backup_path": SMB_MEMORY_FILE,
        "auto_backup_minutes": AUTO_BACKUP_MINUTES,
        "manager_port": MANAGER_PORT,
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[MANAGER] Config saved to {CONFIG_FILE}")
    except Exception as e:
        print("[MANAGER] Failed to save config:", e)


# ============================================================
# MEMORY STATE HELPERS
# ============================================================

def load_memory_state(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                print(f"[MANAGER] Loaded memory state from {path}")
                return pickle.load(f)
        except Exception:
            print("[MANAGER] Corrupt memory file → starting fresh")
    else:
        print(f"[MANAGER] No existing memory file at {path} → starting fresh")
    return {}


def save_memory_state(path: str, state: Dict[str, Any]):
    if not path:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[MANAGER] Saved memory state to {path}")
    except Exception as e:
        print(f"[MANAGER] Save failed for {path}:", e)
        raise


# ============================================================
# SYSTEM / GPU HELPERS
# ============================================================

def autoload_libraries():
    try:
        import torch  # noqa: F401
        import onnxruntime  # noqa: F401
        print("[MANAGER] Core libraries loaded")
        return True
    except Exception as e:
        print("[MANAGER] Autoload failed (non-fatal):", e)
        return False


def init_memory():
    """
    Try to allocate VRAM; if CUDA not available, fall back to system RAM.
    Returns: (mem_block, ctx, mode_str)
    """
    try:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.make_context()
        gpu_mem = cuda.mem_alloc(VRAM_ALLOCATION)
        print("[MANAGER] Allocated 8GB VRAM for agent")
        return gpu_mem, ctx, "GPU"
    except Exception as e:
        print("[MANAGER] CUDA not available or failed → using RAM only:", e)
        ram_mem = bytearray(RAM_ALLOCATION)
        return ram_mem, None, "RAM"


def get_vram_usage():
    """
    Returns (used_vram_mb, total_vram_mb).
    Falls back to (0, 0) if nvidia-smi not available or no NVIDIA GPU.
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ]
        ).decode().strip()
        line = result.splitlines()[0]
        used, total = map(int, line.split(","))
        return used, total
    except Exception:
        return 0, 0


def get_ram_usage():
    """
    Returns (used_ram_mb, total_ram_mb).
    """
    mem = psutil.virtual_memory()
    return mem.used // (1024**2), mem.total // (1024**2)


def detect_game_process():
    """
    Simple heuristic:
    - If any process name contains 'steam' or 'epic', we assume a game context.
    """
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            name = (proc.info["name"] or "").lower()
            if "steam" in name or "epic" in name:
                return True
    except Exception:
        pass
    return False


# ============================================================
# TELEMETRY LOG
# ============================================================

class TelemetryLog:
    def __init__(self, max_events: int = TELEMETRY_MAX_EVENTS):
        self.max_events = max_events
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add(self, event: Dict[str, Any]):
        with self.lock:
            event["time"] = time.time()
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]

    def snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.events)


TELEMETRY = TelemetryLog()


# ============================================================
# PREDICTION ENGINE
# ============================================================

def ewma(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    avg = values[0]
    for v in values[1:]:
        avg = alpha * v + (1 - alpha) * avg
    return avg


class PredictionEngine:
    def __init__(self, window: int = PRED_WINDOW, alpha: float = EWMA_ALPHA):
        self.window = window
        self.alpha = alpha
        self.vram_hist: List[float] = []
        self.ram_hist: List[float] = []
        self.game_hist: List[int] = []
        self.mem_size_hist: List[int] = []
        self.pre_reason_times: List[float] = []  # timestamps of pre_reason calls
        self.lock = threading.Lock()
        self.meta_state = META_CALM

    def add_sample(
        self,
        used_vram: float,
        total_vram: float,
        used_ram: float,
        total_ram: float,
        game_mode: bool,
        memory_size: int,
    ):
        with self.lock:
            vram_pct = (used_vram / total_vram * 100) if total_vram else 0
            ram_pct = (used_ram / total_ram * 100) if total_ram else 0
            self.vram_hist.append(vram_pct)
            self.ram_hist.append(ram_pct)
            self.game_hist.append(1 if game_mode else 0)
            self.mem_size_hist.append(memory_size)

            if len(self.vram_hist) > self.window:
                self.vram_hist = self.vram_hist[-self.window :]
            if len(self.ram_hist) > self.window:
                self.ram_hist = self.ram_hist[-self.window :]
            if len(self.game_hist) > self.window:
                self.game_hist = self.game_hist[-self.window :]
            if len(self.mem_size_hist) > self.window:
                self.mem_size_hist = self.mem_size_hist[-self.window :]

    def record_pre_reason(self):
        with self.lock:
            self.pre_reason_times.append(time.time())
            if len(self.pre_reason_times) > self.window:
                self.pre_reason_times = self.pre_reason_times[-self.window :]

    def _compute_load_meta_state(
        self,
        vram_pred: float,
        ram_pred: float,
        game_prob: float,
        agent_rate: float,
    ) -> str:
        # Simple heuristic meta-state logic
        high_vram = vram_pred > 80
        high_ram = ram_pred > 80
        high_load = agent_rate > 0.5  # calls per second
        game_likely = game_prob > 0.5

        if high_vram or high_ram:
            return META_OVERLOAD
        if high_load or game_likely:
            return META_BURST
        if vram_pred > 40 or ram_pred > 40:
            return META_FOCUSED
        return META_CALM

    def predict(self) -> Dict[str, Any]:
        with self.lock:
            vram_pred = ewma(self.vram_hist, self.alpha)
            ram_pred = ewma(self.ram_hist, self.alpha)
            if self.game_hist:
                game_prob = sum(self.game_hist) / len(self.game_hist)
            else:
                game_prob = 0.0

            # Memory growth: simple slope between first and last
            if len(self.mem_size_hist) >= 2:
                mem_growth = self.mem_size_hist[-1] - self.mem_size_hist[0]
            else:
                mem_growth = 0

            # Agent call rate: pre_reason calls per second (approx)
            if len(self.pre_reason_times) >= 2:
                dt = self.pre_reason_times[-1] - self.pre_reason_times[0]
                agent_rate = len(self.pre_reason_times) / dt if dt > 0 else 0.0
            else:
                agent_rate = 0.0

            meta = self._compute_load_meta_state(vram_pred, ram_pred, game_prob, agent_rate)
            self.meta_state = meta

            return {
                "predicted_vram_pct": vram_pred,
                "predicted_ram_pct": ram_pred,
                "game_mode_probability": game_prob,
                "memory_growth": mem_growth,
                "agent_rate": agent_rate,
                "meta_state": meta,
            }


PREDICTOR = PredictionEngine()


# ============================================================
# MANAGER CORE
# ============================================================

class BackplaneManager:
    def __init__(self):
        print("[MANAGER] Boot → load memory")
        self.state = load_memory_state(LOCAL_MEMORY_FILE)
        self.is_game_mode = False
        self.mem_block, self.ctx, self.mode = init_memory()
        self.memory_path = LOCAL_MEMORY_FILE
        self.lock = threading.Lock()

    def run_cycle(self):
        self.is_game_mode = detect_game_process()
        used_ram, total_ram = get_ram_usage()
        used_vram, total_vram = get_vram_usage()
        memory_size = len(pickle.dumps(self.state)) if self.state else 0
        PREDICTOR.add_sample(
            used_vram, total_vram, used_ram, total_ram, self.is_game_mode, memory_size
        )

    def save_state_all(self):
        try:
            if LOCAL_MEMORY_FILE:
                save_memory_state(LOCAL_MEMORY_FILE, self.state)
            if SMB_MEMORY_FILE:
                save_memory_state(SMB_MEMORY_FILE, self.state)
        except Exception as e:
            print("[MANAGER] save_state_all error:", e)

    # HTTP API helpers

    def get_pre_reason_info(self, client_id: str) -> Dict[str, Any]:
        self.is_game_mode = detect_game_process()
        PREDICTOR.record_pre_reason()
        TELEMETRY.add(
            {
                "type": "pre_reason",
                "client": client_id,
                "mode": self.mode,
                "game_mode": self.is_game_mode,
            }
        )
        return {
            "path": self.memory_path,
            "mode": self.mode,
            "game_mode": self.is_game_mode,
        }

    def handle_post_reason_notification(self, client_id: str):
        TELEMETRY.add(
            {
                "type": "post_reason",
                "client": client_id,
            }
        )
        print(f"[MANAGER] post_reason from {client_id}")

    def heartbeat_info(self, client_id: str) -> Dict[str, Any]:
        self.is_game_mode = detect_game_process()
        used_ram, total_ram = get_ram_usage()
        used_vram, total_vram = get_vram_usage()
        TELEMETRY.add(
            {
                "type": "heartbeat",
                "client": client_id,
                "mode": self.mode,
                "game_mode": self.is_game_mode,
            }
        )
        return {
            "alive": True,
            "mode": self.mode,
            "game_mode": self.is_game_mode,
            "telemetry": {
                "ram_used_mb": used_ram,
                "ram_total_mb": total_ram,
                "vram_used_mb": used_vram,
                "vram_total_mb": total_vram,
            },
            "prediction": PREDICTOR.predict(),
        }


MANAGER: Optional[BackplaneManager] = None  # set in main()


# ============================================================
# HTTP API SERVER
# ============================================================

class ManagerRequestHandler(BaseHTTPRequestHandler):
    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        global MANAGER
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        client_id = query.get("client", ["unknown"])[0]

        if MANAGER is None:
            self._set_json_headers(500)
            self.wfile.write(json.dumps({"error": "manager not ready"}).encode("utf-8"))
            return

        if path == "/pre_reason":
            info = MANAGER.get_pre_reason_info(client_id)
            self._set_json_headers(200)
            self.wfile.write(json.dumps(info).encode("utf-8"))
        elif path == "/heartbeat":
            info = MANAGER.heartbeat_info(client_id)
            self._set_json_headers(200)
            self.wfile.write(json.dumps(info).encode("utf-8"))
        elif path == "/telemetry":
            events = TELEMETRY.snapshot()
            self._set_json_headers(200)
            self.wfile.write(json.dumps(events).encode("utf-8"))
        else:
            self._set_json_headers(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode("utf-8"))

    def do_POST(self):
        global MANAGER
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        client_id = query.get("client", ["unknown"])[0]

        data = self._read_json_body()

        if MANAGER is None:
            self._set_json_headers(500)
            self.wfile.write(json.dumps({"error": "manager not ready"}).encode("utf-8"))
            return

        if path == "/post_reason":
            MANAGER.handle_post_reason_notification(client_id)
            self._set_json_headers(200)
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
        else:
            self._set_json_headers(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode("utf-8"))

    def log_message(self, fmt, *args):
        return


def run_manager_server():
    server = HTTPServer((MANAGER_HOST, MANAGER_PORT), ManagerRequestHandler)
    print(f"[MANAGER] HTTP API running on {MANAGER_HOST}:{MANAGER_PORT}")
    server.serve_forever()


# ============================================================
# GUI
# ============================================================

class BackplaneGUI(QtWidgets.QWidget):
    def __init__(self, manager: BackplaneManager):
        super().__init__()
        self.manager = manager
        self.starting_out = True
        self.initUI()
        self.init_auto_backup_timer()

    def initUI(self):
        self.setWindowTitle("Backplane Memory Manager (Predictive)")
        layout = QtWidgets.QVBoxLayout()

        self.status_label = QtWidgets.QLabel("Status: Agent starting out")
        layout.addWidget(self.status_label)

        self.mode_label = QtWidgets.QLabel(f"Memory Mode: {self.manager.mode}")
        layout.addWidget(self.mode_label)

        self.port_label = QtWidgets.QLabel(f"Backplane Port: {MANAGER_PORT}")
        layout.addWidget(self.port_label)

        self.vram_label = QtWidgets.QLabel("VRAM Usage:")
        layout.addWidget(self.vram_label)
        self.vram_bar = QtWidgets.QProgressBar()
        self.vram_bar.setMaximum(100)
        layout.addWidget(self.vram_bar)

        self.ram_label = QtWidgets.QLabel("System RAM Usage:")
        layout.addWidget(self.ram_label)
        self.ram_bar = QtWidgets.QProgressBar()
        self.ram_bar.setMaximum(100)
        layout.addWidget(self.ram_bar)

        # Predictive Intelligence Panel
        self.meta_label = QtWidgets.QLabel("Meta-State: CALM")
        layout.addWidget(self.meta_label)

        self.pred_vram_label = QtWidgets.QLabel("Predicted VRAM: 0%")
        layout.addWidget(self.pred_vram_label)

        self.pred_ram_label = QtWidgets.QLabel("Predicted RAM: 0%")
        layout.addWidget(self.pred_ram_label)

        self.pred_game_label = QtWidgets.QLabel("Game Mode Probability: 0.00")
        layout.addWidget(self.pred_game_label)

        self.mem_growth_label = QtWidgets.QLabel("Memory Growth: 0 bytes")
        layout.addWidget(self.mem_growth_label)

        self.agent_rate_label = QtWidgets.QLabel("Agent Call Rate: 0.00 calls/s")
        layout.addWidget(self.agent_rate_label)

        # Telemetry list
        self.telemetry_list = QtWidgets.QListWidget()
        layout.addWidget(self.telemetry_list)

        self.local_picker_btn = QtWidgets.QPushButton("Select Local Backup Location")
        self.local_picker_btn.clicked.connect(self.select_local)
        layout.addWidget(self.local_picker_btn)

        self.local_path_label = QtWidgets.QLabel(f"Local Backup: {LOCAL_MEMORY_FILE}")
        layout.addWidget(self.local_path_label)

        self.local_btn = QtWidgets.QPushButton("Save Local Backup Now")
        self.local_btn.clicked.connect(self.save_local_with_popup)
        layout.addWidget(self.local_btn)

        self.smb_btn = QtWidgets.QPushButton("Select SMB Backup Location")
        self.smb_btn.clicked.connect(self.select_smb)
        layout.addWidget(self.smb_btn)

        smb_text = f"SMB Backup: {SMB_MEMORY_FILE}" if SMB_MEMORY_FILE else "SMB Backup: Not Set"
        self.smb_path_label = QtWidgets.QLabel(smb_text)
        layout.addWidget(self.smb_path_label)

        self.auto_label = QtWidgets.QLabel(f"Auto-backup every {AUTO_BACKUP_MINUTES} minutes")
        layout.addWidget(self.auto_label)

        self.setLayout(layout)

        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def init_auto_backup_timer(self):
        self.auto_backup_timer = QtCore.QTimer()
        self.auto_backup_timer.timeout.connect(self.auto_backup)
        self.auto_backup_timer.start(AUTO_BACKUP_MINUTES * 60 * 1000)

    def select_local(self):
        global LOCAL_MEMORY_FILE
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Local Backup File"
        )
        if path:
            LOCAL_MEMORY_FILE = path
            self.manager.memory_path = path
            self.local_path_label.setText(f"Local Backup: {LOCAL_MEMORY_FILE}")
            save_config()
            print(f"[MANAGER] Local backup path set to: {LOCAL_MEMORY_FILE}")

    def save_local_with_popup(self):
        if not LOCAL_MEMORY_FILE:
            QMessageBox.critical(self, "Backup Error", "Local backup path is not set.")
            return
        directory = os.path.dirname(LOCAL_MEMORY_FILE)
        if directory and not os.path.exists(directory):
            QMessageBox.critical(
                self,
                "Backup Error",
                f"Local drive/folder missing:\n{directory}",
            )
            return
        try:
            save_memory_state(LOCAL_MEMORY_FILE, self.manager.state)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Backup Error",
                f"Failed to save local backup:\n{e}",
            )

    def select_smb(self):
        global SMB_MEMORY_FILE
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select SMB Backup Location"
        )
        if path:
            SMB_MEMORY_FILE = path
            self.smb_path_label.setText(f"SMB Backup: {SMB_MEMORY_FILE}")
            save_config()
            print(f"[MANAGER] SMB backup path set to: {SMB_MEMORY_FILE}")

    def auto_backup(self):
        if LOCAL_MEMORY_FILE:
            directory = os.path.dirname(LOCAL_MEMORY_FILE)
            if directory and not os.path.exists(directory):
                QMessageBox.warning(
                    self,
                    "Auto-backup Warning",
                    f"Local backup directory missing:\n{directory}",
                )
            else:
                try:
                    save_memory_state(LOCAL_MEMORY_FILE, self.manager.state)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Auto-backup Warning",
                        f"Failed to auto-backup local:\n{e}",
                    )
        if SMB_MEMORY_FILE:
            directory = os.path.dirname(SMB_MEMORY_FILE)
            if directory and not os.path.exists(directory):
                QMessageBox.warning(
                    self,
                    "Auto-backup Warning",
                    f"SMB backup directory missing or unreachable:\n{directory}",
                )
            else:
                try:
                    save_memory_state(SMB_MEMORY_FILE, self.manager.state)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Auto-backup Warning",
                        f"Failed to auto-backup SMB:\n{e}",
                    )

    def update_status(self):
        self.manager.run_cycle()

        if self.starting_out:
            self.status_label.setText("Status: Agent starting out")
            self.starting_out = False
        else:
            if self.manager.is_game_mode:
                self.status_label.setText("Status: Game Mode (Memory Released)")
            else:
                self.status_label.setText("Status: Normal Mode (Memory Reserved)")

        self.mode_label.setText(f"Memory Mode: {self.manager.mode}")
        self.port_label.setText(f"Backplane Port: {MANAGER_PORT}")

        used_vram, total_vram = get_vram_usage()
        percent_vram = int((used_vram / total_vram) * 100) if total_vram else 0
        self.vram_bar.setValue(percent_vram)
        self.vram_label.setText(f"VRAM Usage: {used_vram} MB / {total_vram} MB")

        used_ram, total_ram = get_ram_usage()
        percent_ram = int((used_ram / total_ram) * 100) if total_ram else 0
        self.ram_bar.setValue(percent_ram)
        self.ram_label.setText(f"System RAM Usage: {used_ram} MB / {total_ram} MB")

        pred = PREDICTOR.predict()
        self.meta_label.setText(f"Meta-State: {pred['meta_state']}")
        self.pred_vram_label.setText(f"Predicted VRAM: {pred['predicted_vram_pct']:.1f}%")
        self.pred_ram_label.setText(f"Predicted RAM: {pred['predicted_ram_pct']:.1f}%")
        self.pred_game_label.setText(f"Game Mode Probability: {pred['game_mode_probability']:.2f}")
        self.mem_growth_label.setText(f"Memory Growth: {pred['memory_growth']} bytes")
        self.agent_rate_label.setText(f"Agent Call Rate: {pred['agent_rate']:.2f} calls/s")

        self.telemetry_list.clear()
        events = TELEMETRY.snapshot()
        for ev in reversed(events[-50:]):
            t = time.strftime("%H:%M:%S", time.localtime(ev["time"]))
            line = f"[{t}] {ev.get('type','?')} from {ev.get('client','?')} | mode={ev.get('mode','?')} game={ev.get('game_mode','?')}"
            item = QListWidgetItem(line)
            self.telemetry_list.addItem(item)


# ============================================================
# BACKPLANE CLIENT (for agents)
# ============================================================

class BackplaneClient:
    def __init__(self, client_id: str, base_url: Optional[str] = None):
        self.client_id = client_id

        if base_url is None:
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
                    port = cfg.get("manager_port", MANAGER_PORT)
                    self.base_url = f"http://127.0.0.1:{port}"
            except Exception:
                self.base_url = f"http://127.0.0.1:{MANAGER_PORT}"
        else:
            self.base_url = base_url

        self.safe_mode = False
        self.last_mode = "RAM"
        self.last_game_mode = False
        self.last_memory_path: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False
        self.heartbeat_interval_sec = 5

    def _get(self, path: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}?client={self.client_id}"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def _post(self, path: str, data: Dict[str, Any], timeout: float = 2.0) -> bool:
        url = f"{self.base_url}{path}?client={self.client_id}"
        try:
            resp = requests.post(url, json=data, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def pre_reason(self) -> Tuple[Dict[str, Any], str, bool]:
        if self.safe_mode:
            return {}, "RAM", False

        info = self._get("/pre_reason")
        if not info:
            self.safe_mode = True
            self.last_mode = "RAM"
            self.last_game_mode = False
            self.last_memory_path = None
            return {}, "RAM", False

        path = info.get("path")
        mode = info.get("mode", "RAM")
        game_mode = bool(info.get("game_mode", False))

        self.last_mode = mode
        self.last_game_mode = game_mode
        self.last_memory_path = path

        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
            except Exception:
                state = {}
        else:
            state = {}

        return state, mode, game_mode

    def post_reason(self, state: Dict[str, Any]) -> None:
        if self.safe_mode:
            if self.last_memory_path:
                try:
                    directory = os.path.dirname(self.last_memory_path)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    with open(self.last_memory_path, "wb") as f:
                        pickle.dump(state, f)
                except Exception:
                    pass
            return

        if not self.last_memory_path:
            return

        try:
            directory = os.path.dirname(self.last_memory_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.last_memory_path, "wb") as f:
                pickle.dump(state, f)
        except Exception:
            pass

        ok = self._post("/post_reason", {"updated": True})
        if not ok:
            self.safe_mode = True

    def heartbeat_once(self):
        info = self._get("/heartbeat", timeout=1.5)
        if not info:
            self.safe_mode = True
            self.last_mode = "RAM"
            self.last_game_mode = False
            return
        self.safe_mode = False
        self.last_mode = info.get("mode", self.last_mode)
        self.last_game_mode = bool(info.get("game_mode", self.last_game_mode))

    def _heartbeat_loop(self):
        while self._heartbeat_running:
            self.heartbeat_once()
            time.sleep(self.heartbeat_interval_sec)

    def start_heartbeat(self):
        if self._heartbeat_running:
            return
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._heartbeat_running = False


# ============================================================
# DUMMY AGENT USING BACKPLANE
# ============================================================

def dummy_agent_full_reason(state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    counter = state.get("full_counter", 0) + 1
    state["full_counter"] = counter
    state["last_mode"] = mode
    print(f"[AGENT] FULL reasoning. full_counter={counter}, mode={mode}")
    return state


def dummy_agent_light_reason(state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    counter = state.get("light_counter", 0) + 1
    state["light_counter"] = counter
    state["last_mode"] = f"LIGHT-{mode}"
    print(f"[AGENT] LIGHT reasoning. light_counter={counter}, mode={mode}")
    return state


def dummy_agent_safe_reason(state: Dict[str, Any]) -> Dict[str, Any]:
    counter = state.get("safe_counter", 0) + 1
    state["safe_counter"] = counter
    print(f"[AGENT] SAFE_MODE reasoning. safe_counter={counter}")
    return state


def dummy_agent_loop():
    client = BackplaneClient(client_id="dummy_agent")
    client.start_heartbeat()

    try:
        while True:
            state, mode, game_mode = client.pre_reason()

            if client.safe_mode:
                new_state = dummy_agent_safe_reason(state)
            else:
                if game_mode:
                    new_state = dummy_agent_light_reason(state, mode)
                else:
                    new_state = dummy_agent_full_reason(state, mode)

            client.post_reason(new_state)
            time.sleep(3)
    except KeyboardInterrupt:
        print("[AGENT] Loop interrupted")
    finally:
        client.stop_heartbeat()


# ============================================================
# MAIN
# ============================================================

def main():
    global MANAGER

    load_config()
    autoload_libraries()
    MANAGER = BackplaneManager()

    threading.Thread(target=run_manager_server, daemon=True).start()
    threading.Thread(target=dummy_agent_loop, daemon=True).start()

    app = QtWidgets.QApplication([])
    gui = BackplaneGUI(MANAGER)
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()

