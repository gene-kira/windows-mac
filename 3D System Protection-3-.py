import sys
import os
import time
import socket
import uuid
import platform
import subprocess
import importlib
import logging
from hashlib import sha256
import re
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

REQUIRED_LIBS = [
    "psutil",
    "PyQt5",
    "cryptography"
]

OPTIONAL_LIBS = [
    "geoip2"
]

def ensure_lib(lib_name, pip_name=None, required=True):
    pip_name = pip_name or lib_name
    try:
        importlib.import_module(lib_name)
        logging.info(f"Library loaded: {lib_name}")
        return True
    except ImportError:
        logging.warning(f"Library missing: {lib_name}, attempting install: {pip_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            importlib.import_module(lib_name)
            logging.info(f"Library installed and loaded: {lib_name}")
            return True
        except Exception as e:
            msg = f"Failed to install {lib_name}: {e}"
            if required:
                logging.error(msg)
            else:
                logging.warning(msg)
            return False

def autoload_all():
    ok = True
    for lib in REQUIRED_LIBS:
        if not ensure_lib(lib, required=True):
            ok = False
    for lib in OPTIONAL_LIBS:
        ensure_lib(lib, required=False)
    return ok

if not autoload_all():
    logging.error("Critical libraries missing. Exiting.")
    sys.exit(1)

import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTextEdit, QLabel, QSplitter,
    QPushButton, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

try:
    import geoip2.database
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

# ---------- Borg-style machine-bound crypto ----------

def get_hw_fingerprint() -> bytes:
    node = platform.node()
    mac = uuid.getnode()
    data = f"{node}-{mac}".encode("utf-8")
    return sha256(data).digest()

def get_os_id() -> bytes:
    info = f"{platform.system()}-{platform.release()}-{platform.version()}"
    return sha256(info.encode("utf-8")).digest()

MACHINE_SALT = b"YOUR_STATIC_SALT_HERE_CHANGE_ME"
MACHINE_ROOT = sha256(get_hw_fingerprint() + get_os_id() + MACHINE_SALT).digest()

def derive_stream_key(process_id: int, dest_ip: str, dest_port: int, nonce: bytes) -> bytes:
    material = (
        MACHINE_ROOT +
        process_id.to_bytes(8, "big", signed=False) +
        dest_ip.encode("utf-8") +
        dest_port.to_bytes(2, "big", signed=False) +
        nonce
    )
    return sha256(material).digest()

def borg_encrypt(process_id: int, dest_ip: str, dest_port: int, data: bytes) -> bytes:
    nonce = os.urandom(12)
    key = derive_stream_key(process_id, dest_ip, dest_port, nonce)
    aead = ChaCha20Poly1305(key)
    ct = aead.encrypt(nonce, data, b"")
    return nonce + ct

# ---------- Personal data detection ----------

SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
DOB_REGEX = re.compile(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
MAC_REGEX = re.compile(r"\b[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}\b")

def detect_personal_data(data: bytes) -> bool:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return False
    if SSN_REGEX.search(text):
        return True
    if PHONE_REGEX.search(text):
        return True
    if DOB_REGEX.search(text):
        return True
    if MAC_REGEX.search(text):
        return True
    return False

# ---------- Memory manager (.borgmind, encrypted, skip E:) ----------

class MemoryManager:
    def __init__(self):
        self.data = {
            "programs": {},        # exe -> {class, status, override}
            "connections": {},     # ip -> {region, confidence}
            "region_history": {},  # exe -> region -> {first_seen, last_seen, count}
            "behavior_history": {},# exe -> behavior profile
            "integrity_profile": {}# exe -> integrity profile
        }
        self.path = None

    def _candidate_roots(self):
        roots = []
        try:
            for part in psutil.disk_partitions(all=False):
                if "network" in (part.fstype or "").lower() or "cifs" in (part.fstype or "").lower():
                    roots.append(part.mountpoint)
        except Exception:
            pass
        for letter in "DFGHIJKLMNOPQRSTUVWXYZ":  # skip E:
            root = f"{letter}:\\"
            if os.path.exists(root):
                roots.append(root)
        if os.path.exists("C:\\"):
            roots.append("C:\\")
        if not roots:
            roots.append(os.path.expanduser("~"))
        return roots

    def find_storage_path(self):
        if self.path and os.path.isdir(os.path.dirname(self.path)):
            return self.path
        for root in self._candidate_roots():
            try:
                if not os.path.isdir(root):
                    continue
                test_path = os.path.join(root, ".borgmind")
                try:
                    with open(test_path + ".test", "wb") as f:
                        f.write(b"test")
                    os.remove(test_path + ".test")
                    self.path = test_path
                    logging.info(f"Memory storage path set to: {self.path}")
                    return self.path
                except Exception:
                    continue
            except Exception:
                continue
        self.path = os.path.join(os.getcwd(), ".borgmind")
        logging.info(f"Memory storage fallback path: {self.path}")
        return self.path

    def load_memory(self):
        path = self.find_storage_path()
        if not os.path.exists(path):
            logging.info("No existing .borgmind file found.")
            return
        try:
            with open(path, "rb") as f:
                blob = f.read()
            if len(blob) < 12:
                logging.warning("Invalid .borgmind file.")
                return
            nonce = blob[:12]
            ct = blob[12:]
            key = sha256(MACHINE_ROOT + nonce).digest()
            aead = ChaCha20Poly1305(key)
            plaintext = aead.decrypt(nonce, ct, b"")
            self.data = json.loads(plaintext.decode("utf-8"))
            logging.info("Loaded memory from .borgmind.")
        except Exception as e:
            logging.warning(f"Failed to load .borgmind: {e}")

    def save_memory(self):
        path = self.find_storage_path()
        try:
            plaintext = json.dumps(self.data).encode("utf-8")
            nonce = os.urandom(12)
            key = sha256(MACHINE_ROOT + nonce).digest()
            aead = ChaCha20Poly1305(key)
            ct = aead.encrypt(nonce, plaintext, b"")
            blob = nonce + ct
            with open(path, "wb") as f:
                f.write(blob)
            logging.info(f"Saved memory to {path}")
        except Exception as e:
            logging.warning(f"Failed to save .borgmind: {e}")

    def apply_prior_knowledge(self, procs_info):
        for pid, info in procs_info.items():
            exe = info.get("exe") or ""
            if not exe:
                continue
            mem = self.data["programs"].get(exe)
            if not mem:
                continue
            override = mem.get("override")

            if override == "force_friendly":
                info["class"] = 2.0
                info["status"] = "friendly_forced"
                info["override"] = override
                continue
            if override == "force_block":
                info["class"] = -1.0
                info["status"] = "blocked_forced"
                info["override"] = override
                continue
            if override == "force_disable":
                info["class"] = 0.0
                info["status"] = "disabled_forced"
                info["override"] = override
                continue
            if override == "force_kill":
                info["class"] = -3.0
                info["status"] = "kill_forced"
                info["override"] = override
                continue
            if override == "force_radioactive":
                info["class"] = 1.0
                info["status"] = "radioactive"
                info["override"] = override
                continue

            info["class"] = mem.get("class", info["class"])
            info["status"] = mem.get("status", info["status"])
            info["override"] = mem.get("override", info.get("override"))

    def update_from_live(self, procs_info, conns_info):
        for pid, info in procs_info.items():
            exe = info.get("exe") or ""
            if not exe:
                continue
            existing = self.data["programs"].get(exe, {})
            override = existing.get("override")
            self.data["programs"][exe] = {
                "class": info["class"],
                "status": info["status"],
                "override": override
            }
            if override:
                info["override"] = override

        for c in conns_info:
            ip = c.get("raddr_ip") or ""
            if not ip:
                continue
            self.data["connections"][ip] = {
                "region": c.get("region", "Unknown"),
                "confidence": c.get("confidence", 0.0)
            }

    def apply_override(self, exe, override_type):
        prog = self.data["programs"].get(exe, {})
        if override_type == "force_friendly":
            prog["override"] = "force_friendly"
            prog["class"] = 2.0
            prog["status"] = "friendly_forced"
        elif override_type == "force_block":
            prog["override"] = "force_block"
            prog["class"] = -1.0
            prog["status"] = "blocked_forced"
        elif override_type == "force_disable":
            prog["override"] = "force_disable"
            prog["class"] = 0.0
            prog["status"] = "disabled_forced"
        elif override_type == "force_kill":
            prog["override"] = "force_kill"
            prog["class"] = -3.0
            prog["status"] = "kill_forced"
        elif override_type == "force_radioactive":
            prog["override"] = "force_radioactive"
            prog["class"] = 1.0
            prog["status"] = "radioactive"
        self.data["programs"][exe] = prog
        self.save_memory()

    def update_region_history(self, exe, region):
        if not exe or not region:
            return
        rh = self.data.setdefault("region_history", {})
        per_exe = rh.setdefault(exe, {})
        entry = per_exe.get(region, {
            "first_seen": time.time(),
            "last_seen": time.time(),
            "count": 0
        })
        entry["last_seen"] = time.time()
        entry["count"] += 1
        per_exe[region] = entry
        rh[exe] = per_exe

    def update_behavior_history(self, exe, ports, cmdline):
        if not exe:
            return
        bh = self.data.setdefault("behavior_history", {})
        entry = bh.get(exe, {
            "avg_conn_rate": 0.0,
            "last_conn_time": 0.0,
            "ports": [],
            "cmdline_hash": ""
        })
        now = time.time()
        last = entry.get("last_conn_time", 0.0)
        if last > 0:
            interval = max(now - last, 1.0)
            rate = 1.0 / interval
            entry["avg_conn_rate"] = (entry["avg_conn_rate"] * 0.8) + (rate * 0.2)
        entry["last_conn_time"] = now
        existing_ports = set(entry.get("ports", []))
        for p in ports:
            existing_ports.add(p)
        entry["ports"] = sorted(existing_ports)
        cmdline_str = " ".join(cmdline or [])
        entry["cmdline_hash"] = sha256(cmdline_str.encode("utf-8")).hexdigest()
        bh[exe] = entry

    def get_behavior_entry(self, exe):
        return self.data.get("behavior_history", {}).get(exe, None)

    # ---------- Integrity profile helpers ----------

    def ensure_integrity_entry(self, exe: str):
        ip = self.data.setdefault("integrity_profile", {})
        if exe not in ip:
            ip[exe] = {
                "state": "ok",
                "image_hash": "",
                "signature_status": "unknown",
                "dll_set_hash": "",
                "last_dlls": [],
                "injection_flags": {
                    "remote_thread": False,
                    "suspicious_alloc": False,
                    "cross_process_write": False
                },
                "persistence_flags": {
                    "run_key": False,
                    "services": False,
                    "scheduled_task": False
                },
                "api_flags": {
                    "injection_related": 0,
                    "keylogging_related": 0,
                    "credential_access": 0,
                    "crypto_abuse": 0
                },
                "file_flags": {
                    "system_dir_write": 0,
                    "sensitive_file_access": 0
                },
                "registry_flags": {
                    "sensitive_key_access": 0
                },
                "last_update": 0.0,
                "violation_reasons": []
            }
        return ip[exe]

    def get_integrity_entry(self, exe: str):
        return self.data.get("integrity_profile", {}).get(exe)

    def get_integrity_state(self, exe: str) -> str:
        entry = self.get_integrity_entry(exe)
        if not entry:
            return "ok"
        return entry.get("state", "ok")

    def get_integrity_reasons(self, exe: str):
        entry = self.get_integrity_entry(exe)
        if not entry:
            return []
        return entry.get("violation_reasons", [])

    def update_integrity_from_event(self, exe: str, event: dict):
        """
        Placeholder for future integrity events.
        In the future, feed PROCESS_EVENT / IMAGE_LOAD_EVENT / MEMORY_EVENT here.
        """
        if not exe:
            return
        entry = self.ensure_integrity_entry(exe)
        entry["last_update"] = time.time()

        etype = event.get("type")
        # Example: remote thread injection
        if etype == "MEMORY_EVENT" and "remote_thread" in event.get("flags", []):
            entry["injection_flags"]["remote_thread"] = True
            if entry["state"] != "violation":
                entry["state"] = "violation"
                reasons = entry.setdefault("violation_reasons", [])
                if "remote_thread_injection" not in reasons:
                    reasons.append("remote_thread_injection")

MEMORY = MemoryManager()
MEMORY.load_memory()

# ---------- Classifier + exceptions ----------

SYSTEM_DIR_HINTS = [
    os.path.normpath(p) for p in [
        "/bin", "/usr/bin", "/usr/sbin", "/sbin",
        "C:\\Windows\\System32", "C:\\Windows"
    ]
]

def is_system_path(path: str) -> bool:
    if not path:
        return False
    path = os.path.normpath(path)
    for hint in SYSTEM_DIR_HINTS:
        if path.startswith(hint):
            return True
    return False

def is_startup_python(proc_info) -> bool:
    exe = (proc_info.get("exe") or "").lower()
    cmdline_list = proc_info.get("cmdline") or []
    cmdline = " ".join(cmdline_list).lower()
    if "python" not in exe and "python" not in cmdline:
        return False
    startup_keywords = [
        "\\start menu\\programs\\startup",
        "\\appdata\\roaming\\microsoft\\windows\\start menu\\programs\\startup",
        "\\programdata\\microsoft\\windows\\start menu\\programs\\startup"
    ]
    combined = (exe + " " + cmdline).replace("/", "\\")
    for kw in startup_keywords:
        if kw in combined:
            return True
    return False

def classify_program(proc: psutil.Process):
    try:
        info = proc.as_dict(attrs=["pid", "name", "exe", "cmdline"])
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0.0, "unknown", {}
    name = info.get("name") or ""
    exe = info.get("exe") or ""
    if is_startup_python(info):
        return 2.0, "startup_python_friendly", info
    if is_system_path(exe):
        return 1.0, "os", info
    if name and exe:
        return 2.0, "user", info
    return 0.0, "unknown", info

def escalate_hostile(class_number: float, has_remote: bool) -> float:
    if class_number > 0:
        return class_number
    if has_remote:
        return -1.0
    return class_number

# ---------- Prometheus-style region guess ----------

GEOIP_READER = None
if GEOIP_AVAILABLE:
    default_db_path = os.path.expanduser("~/GeoLite2-City.mmdb")
    if os.path.exists(default_db_path):
        try:
            GEOIP_READER = geoip2.database.Reader(default_db_path)
            logging.info("GeoIP database loaded.")
        except Exception as e:
            logging.warning(f"Failed to load GeoIP DB: {e}")
            GEOIP_READER = None

def guess_region(ip: str):
    try:
        if ip.startswith("127."):
            return "Local-Trusted", 1.0
        if ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172.16."):
            return "Local", 1.0
        if GEOIP_READER:
            resp = GEOIP_READER.city(ip)
            country = resp.country.iso_code
            if country == "US":
                return "US", 0.9
            else:
                return "Overseas", 0.9
        else:
            return "Unknown", 0.3
    except Exception:
        return "Unknown", 0.0

# ---------- Data Guardian (policy logic) ----------

def guard_outbound(process_id: int, class_number: float, override: str, dest_ip: str, dest_port: int, data: bytes):
    if dest_ip == "127.0.0.1":
        return "ALLOW_LOCALHOST", data
    has_personal = detect_personal_data(data)

    if override == "force_radioactive":
        if has_personal:
            return "ALLOW_RADIOACTIVE_PERSONAL", data
        return "ALLOW_RADIOACTIVE", data

    if override in ("force_block", "force_kill", "force_disable"):
        return "BLOCK_OVERRIDE", None

    if class_number <= 0:
        if has_personal:
            return "BLOCK", None
        else:
            ct = borg_encrypt(process_id, dest_ip, dest_port, data)
            return "BORG_ENCRYPT", ct

    if has_personal:
        return "ALLOW_WITH_PERSONAL", data
    return "ALLOW", data

# ---------- Live data collection ----------

def collect_processes_and_connections():
    procs_info = {}
    conns_info = []

    for proc in psutil.process_iter(attrs=["pid", "name", "exe", "cmdline"]):
        pid = proc.info["pid"]
        try:
            class_num, status, info = classify_program(proc)
        except psutil.NoSuchProcess:
            continue
        procs_info[pid] = {
            "pid": pid,
            "name": info.get("name") or "",
            "exe": info.get("exe") or "",
            "cmdline": info.get("cmdline") or [],
            "class": class_num,
            "status": status,
            "has_remote": False,
            "override": None
        }

    MEMORY.apply_prior_knowledge(procs_info)

    for conn in psutil.net_connections(kind="inet"):
        pid = conn.pid
        if pid is None or pid not in procs_info:
            continue
        laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
        raddr_ip = conn.raddr.ip if conn.raddr else ""
        raddr_port = conn.raddr.port if conn.raddr else 0
        status = conn.status

        if raddr_ip:
            region, conf = guess_region(raddr_ip)
            has_remote = (raddr_ip != "127.0.0.1")
        else:
            region, conf = ("Local", 1.0)
            has_remote = False

        if has_remote:
            procs_info[pid]["has_remote"] = True

        exe = procs_info[pid]["exe"]
        MEMORY.update_region_history(exe, region)

        conns_info.append({
            "pid": pid,
            "laddr": laddr,
            "raddr_ip": raddr_ip,
            "raddr_port": raddr_port,
            "status": status,
            "region": region,
            "confidence": conf
        })

        ports = [raddr_port] if raddr_port else []
        MEMORY.update_behavior_history(exe, ports, procs_info[pid].get("cmdline"))

    for pid, info in procs_info.items():
        info["class"] = escalate_hostile(info["class"], info["has_remote"])
        if info["class"] < 0 and not str(info["status"]).startswith("hostile_forced"):
            if not str(info["status"]).startswith("kill_forced"):
                info["status"] = "hostile"

    MEMORY.update_from_live(procs_info, conns_info)
    return procs_info, conns_info

# ---------- GUI Control Tower ----------

class ControlTower(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Control Tower – Live Borgmind")
        self.resize(1500, 900)

        main_layout = QVBoxLayout(self)

        header = QLabel("Live Trust Axis + Prometheus + Borgmind + Admin Overrides + Radioactive + Integrity")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)

        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(7)
        self.proc_table.setHorizontalHeaderLabels(
            ["PID", "Name", "Class", "Status", "Exe", "Cmdline", "Integrity"]
        )
        upper_layout.addWidget(self.proc_table)

        self.conn_table = QTableWidget()
        self.conn_table.setColumnCount(7)
        self.conn_table.setHorizontalHeaderLabels(
            ["PID", "Local", "Remote IP", "Remote Port", "Region", "Conf", "Status"]
        )
        upper_layout.addWidget(self.conn_table)

        splitter.addWidget(upper_widget)

        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)

        self.blocked_table = QTableWidget()
        self.blocked_table.setColumnCount(3)
        self.blocked_table.setHorizontalHeaderLabels(["Exe", "Class", "Status"])
        lower_layout.addWidget(QLabel("Blocked / Forced-Hostile / Disabled / Kill-Label Programs"))
        lower_layout.addWidget(self.blocked_table)

        hp_group = QGroupBox("High Probability Action Panel")
        hp_layout = QVBoxLayout(hp_group)

        self.hp_selected_label = QLabel("Selected Process: <none>")
        hp_layout.addWidget(self.hp_selected_label)

        hp_button_row = QHBoxLayout()
        self.btn_hp_allow = QPushButton("ALLOW")
        self.btn_hp_block = QPushButton("BLOCK")
        self.btn_hp_disable = QPushButton("DISABLE")
        self.btn_hp_kill = QPushButton("KILL (ADMIN LABEL)")
        self.btn_hp_radioactive = QPushButton("RADIOACTIVE")

        hp_button_row.addWidget(self.btn_hp_allow)
        hp_button_row.addWidget(self.btn_hp_block)
        hp_button_row.addWidget(self.btn_hp_disable)
        hp_button_row.addWidget(self.btn_hp_kill)
        hp_button_row.addWidget(self.btn_hp_radioactive)

        hp_layout.addLayout(hp_button_row)
        lower_layout.addWidget(hp_group)

        button_row = QHBoxLayout()
        self.btn_force_block = QPushButton("Force Block Selected Process")
        self.btn_force_allow = QPushButton("Force Allow Selected Process")
        button_row.addWidget(self.btn_force_block)
        button_row.addWidget(self.btn_force_allow)
        lower_layout.addLayout(button_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lower_layout.addWidget(self.log)

        splitter.addWidget(lower_widget)

        self.proc_table.currentCellChanged.connect(self.update_hp_selected_label)

        self.btn_hp_allow.clicked.connect(self.hp_allow_clicked)
        self.btn_hp_block.clicked.connect(self.hp_block_clicked)
        self.btn_hp_disable.clicked.connect(self.hp_disable_clicked)
        self.btn_hp_kill.clicked.connect(self.hp_kill_clicked)
        self.btn_hp_radioactive.clicked.connect(self.hp_radioactive_clicked)

        self.btn_force_block.clicked.connect(self.force_block_selected)
        self.btn_force_allow.clicked.connect(self.force_allow_selected)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_view)
        self.timer.start(3000)

        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.save_memory_tick)
        self.memory_timer.start(60000)

        self.refresh_view(initial=True)

    def color_for_class(self, class_num: float):
        if class_num >= 2:
            return Qt.blue
        if class_num == 1:
            return Qt.red  # radioactive / watched
        if class_num == 0:
            return Qt.yellow
        if class_num < 0:
            return Qt.darkRed
        return Qt.white

    def _get_selected_pid(self):
        row = self.proc_table.currentRow()
        if row < 0:
            return None
        item = self.proc_table.item(row, 0)
        if not item:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None

    def _get_selected_exe(self):
        row = self.proc_table.currentRow()
        if row < 0:
            return None
        exe_item = self.proc_table.item(row, 4)
        if not exe_item:
            return None
        exe = exe_item.text().strip()
        return exe or None

    def update_hp_selected_label(self, currentRow, currentColumn, prevRow, prevColumn):
        exe = self._get_selected_exe()
        if exe:
            self.hp_selected_label.setText(f"Selected Process: {exe}")
        else:
            self.hp_selected_label.setText("Selected Process: <none>")

    def hp_allow_clicked(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[HP] No process selected for ALLOW.")
            return
        MEMORY.apply_override(exe, "force_friendly")
        self.log.append(f"[HP] ALLOW applied to {exe}")
        self.refresh_view()

    def hp_block_clicked(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[HP] No process selected for BLOCK.")
            return
        MEMORY.apply_override(exe, "force_block")
        self.log.append(f"[HP] BLOCK applied to {exe}")
        self.refresh_view()

    def hp_disable_clicked(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[HP] No process selected for DISABLE.")
            return
        MEMORY.apply_override(exe, "force_disable")
        self.log.append(f"[HP] DISABLE applied to {exe}")
        self.refresh_view()

    def hp_kill_clicked(self):
        exe = self._get_selected_exe()
        pid = self._get_selected_pid()
        if not exe or pid is None:
            self.log.append("[HP] No process selected for KILL label.")
            return
        MEMORY.apply_override(exe, "force_kill")
        self.log.append(f"[HP] KILL (ADMIN LABEL) applied to {exe} (PID {pid})")
        self.refresh_view()

    def hp_radioactive_clicked(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[HP] No process selected for RADIOACTIVE.")
            return
        MEMORY.apply_override(exe, "force_radioactive")
        self.log.append(f"[HP] RADIOACTIVE applied to {exe}")
        self.refresh_view()

    def force_block_selected(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[!] No process selected for force block.")
            return
        MEMORY.apply_override(exe, "force_block")
        self.log.append(f"[!] Admin override: FORCE BLOCK {exe}")
        self.refresh_view()

    def force_allow_selected(self):
        exe = self._get_selected_exe()
        if not exe:
            self.log.append("[!] No process selected for force allow.")
            return
        MEMORY.apply_override(exe, "force_friendly")
        self.log.append(f"[!] Admin override: FORCE ALLOW {exe}")
        self.refresh_view()

    def refresh_blocked_table(self):
        blocked = []
        for exe, info in MEMORY.data["programs"].items():
            override = info.get("override")
            cls = info.get("class", 0.0)
            status = info.get("status", "")
            if override in ("force_block", "force_disable", "force_kill") or cls < 0:
                blocked.append((exe, cls, status))
        self.blocked_table.setRowCount(len(blocked))
        for row, (exe, cls, status) in enumerate(blocked):
            exe_item = QTableWidgetItem(exe)
            class_item = QTableWidgetItem(str(cls))
            status_item = QTableWidgetItem(status)
            color = self.color_for_class(cls)
            class_item.setForeground(color)
            status_item.setForeground(color)
            self.blocked_table.setItem(row, 0, exe_item)
            self.blocked_table.setItem(row, 1, class_item)
            self.blocked_table.setItem(row, 2, status_item)

    def refresh_view(self, initial=False):
        procs_info, conns_info = collect_processes_and_connections()

        self.proc_table.setRowCount(len(procs_info))
        for row, (pid, info) in enumerate(sorted(procs_info.items())):
            pid_item = QTableWidgetItem(str(pid))
            name_item = QTableWidgetItem(info["name"])
            class_item = QTableWidgetItem(str(info["class"]))
            status_item = QTableWidgetItem(info["status"])
            exe_item = QTableWidgetItem(info["exe"])
            cmdline_item = QTableWidgetItem(" ".join(info.get("cmdline") or []))

            exe_path = info["exe"]
            integrity_state = MEMORY.get_integrity_state(exe_path)
            integrity_item = QTableWidgetItem(integrity_state.upper())

            color = self.color_for_class(info["class"])
            class_item.setForeground(color)
            status_item.setForeground(color)

            if integrity_state == "ok":
                pass
            elif integrity_state == "warn":
                integrity_item.setForeground(Qt.yellow)
            elif integrity_state == "violation":
                integrity_item.setForeground(Qt.red)
                pid_item.setBackground(Qt.red)

            reasons = MEMORY.get_integrity_reasons(exe_path)
            if reasons:
                integrity_item.setToolTip("Integrity: " + ", ".join(reasons))

            if info.get("override") == "force_radioactive":
                pid_item.setBackground(Qt.red)
                status_item.setForeground(Qt.red)

            self.proc_table.setItem(row, 0, pid_item)
            self.proc_table.setItem(row, 1, name_item)
            self.proc_table.setItem(row, 2, class_item)
            self.proc_table.setItem(row, 3, status_item)
            self.proc_table.setItem(row, 4, exe_item)
            self.proc_table.setItem(row, 5, cmdline_item)
            self.proc_table.setItem(row, 6, integrity_item)

        self.conn_table.setRowCount(len(conns_info))
        for row, c in enumerate(conns_info):
            pid_item = QTableWidgetItem(str(c["pid"]))
            laddr_item = QTableWidgetItem(c["laddr"])
            raddr_ip_item = QTableWidgetItem(c["raddr_ip"])
            raddr_port_item = QTableWidgetItem(str(c["raddr_port"]))
            region_item = QTableWidgetItem(c["region"])
            conf_item = QTableWidgetItem(f"{c['confidence']:.2f}")
            status_item = QTableWidgetItem(c["status"])

            class_num = procs_info[c["pid"]]["class"]
            color = self.color_for_class(class_num)
            region_item.setForeground(color)
            status_item.setForeground(color)

            if procs_info[c["pid"]].get("override") == "force_radioactive":
                pid_item.setBackground(Qt.red)
                status_item.setForeground(Qt.red)
                region_item.setForeground(Qt.red)

            self.conn_table.setItem(row, 0, pid_item)
            self.conn_table.setItem(row, 1, laddr_item)
            self.conn_table.setItem(row, 2, raddr_ip_item)
            self.conn_table.setItem(row, 3, raddr_port_item)
            self.conn_table.setItem(row, 4, region_item)
            self.conn_table.setItem(row, 5, conf_item)
            self.conn_table.setItem(row, 6, status_item)

        self.refresh_blocked_table()

        now = time.strftime("%H:%M:%S")
        if initial:
            self.log.append(f"[{now}] Initial live scan complete. Memory path: {MEMORY.find_storage_path()}")
        else:
            self.log.append(f"[{now}] Refreshed live processes and connections.")

    def save_memory_tick(self):
        MEMORY.save_memory()
        now = time.strftime("%H:%M:%S")
        self.log.append(f"[{now}] .borgmind autosave completed.")

def main():
    app = QApplication(sys.argv)
    tower = ControlTower()
    tower.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

