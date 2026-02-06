# ===== MR FREEZE v4.9 – SINGLE FILE =====
# Adaptive Thread-Pool Scanner + Persistent Encrypted Logs
# Admin Mode God View + Privacy Firewall + Outbound Shield (BLOCK/ALLOW)

import importlib
import subprocess
import sys
import threading
import time
import random
import platform
import json
import os
import socket
from datetime import datetime, timedelta
import statistics
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- AUTOLOADER ----------
def autoload(pkg, import_name=None):
    name = import_name or pkg
    try:
        return importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(name)

tk = autoload("tkinter")
filedialog = autoload("tkinter.filedialog")

# ---------- GLOBAL MEMORY CONFIG ----------
DEFAULT_MEMORY_DIR = os.path.abspath(".")
MEMORY_FILENAME = "mr_freeze_memory_v4_8.json.enc"  # kept same for continuity
LOG_FILENAME = "mr_freeze_log_v4_8.log.enc"         # kept same for continuity
LOG_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BATCH_LOAD = 200
LOG_INITIAL_LINES = 500
LOG_BACKGROUND_SLEEP = 0.05  # seconds between background batches

def get_machine_fingerprint():
    data = platform.node() + "|" + platform.system() + "|" + platform.machine()
    return hashlib.sha256(data.encode("utf-8")).digest()

def encrypt_bytes(data: bytes) -> bytes:
    key = get_machine_fingerprint()
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return bytes(out)

def decrypt_bytes(data: bytes) -> bytes:
    return encrypt_bytes(data)

# ---------- PRIVACY GUARDIAN ORGAN ----------
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MAC_PATTERN = re.compile(r"\b[0-9A-Fa-f]{2}([-:])[0-9A-Fa-f]{2}(?:\1[0-9A-Fa-f]{2}){4}\b")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HOST_PATTERN = re.compile(r"\b[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b")
BIOMETRIC_WORDS = ["fingerprint", "faceid", "iris", "retina", "palm", "voiceprint"]

def looks_sensitive(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(w in t for w in BIOMETRIC_WORDS):
        return True
    if SSN_PATTERN.search(text):
        return True
    if EMAIL_PATTERN.search(text):
        return True
    if PHONE_PATTERN.search(text):
        return True
    if MAC_PATTERN.search(text):
        return True
    if HOST_PATTERN.search(text):
        return True
    if IP_PATTERN.search(text):
        return True
    return False

def mask_sensitive(text: str, admin: bool = False) -> str:
    if admin:
        return text
    if not text:
        return text
    text = SSN_PATTERN.sub("[SSN]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = MAC_PATTERN.sub("[MAC]", text)
    text = HOST_PATTERN.sub("[HOST]", text)
    text = IP_PATTERN.sub("[IP]", text)
    for w in BIOMETRIC_WORDS:
        text = re.sub(w, "[BIOMETRIC]", text, flags=re.IGNORECASE)
    return text

def display_address(addr: str, admin: bool = False) -> str:
    if admin:
        return addr
    if not addr:
        return addr
    m = IP_PATTERN.search(addr)
    if not m:
        return mask_sensitive(addr, admin=False)
    ip = m.group(0)
    parts = ip.split(".")
    if len(parts) == 4:
        masked_ip = ".".join(parts[:3] + ["x"])
        return addr.replace(ip, masked_ip)
    return mask_sensitive(addr, admin=False)

# ---------- OUI → Vendor Map ----------
OUI_VENDOR_MAP = {
    "00:1A:2B": "Cisco",
    "00:1B:63": "Apple",
    "00:1C:BF": "Dell",
    "00:1D:D8": "HP",
    "00:1E:68": "Huawei",
    "F4:5C:89": "Ubiquiti",
}

def lookup_vendor(mac):
    if not mac:
        return "Unknown"
    mac = mac.upper()
    prefix = mac[:8]
    return OUI_VENDOR_MAP.get(prefix, "Unknown")

# ---------- ARP / DNS / PORTS ----------
def get_mac_from_arp(ip):
    try:
        if platform.system().lower() == "windows":
            cmd = ["arp", "-a", ip]
        else:
            cmd = ["arp", "-n", ip]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if ip in line:
                parts = line.split()
                for p in parts:
                    if "-" in p and len(p) >= 17:
                        return p.replace("-", ":").upper()
                    if ":" in p and len(p) >= 17:
                        return p.upper()
    except Exception:
        pass
    return None

def reverse_dns_name(ip):
    try:
        name, _, _ = socket.gethostbyaddr(ip)
        return name
    except Exception:
        return None

def scan_ports(ip, ports, timeout=0.3):
    open_ports = []
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((ip, port)) == 0:
                    open_ports.append(port)
        except Exception:
            pass
    return open_ports

def detect_primary_lan_range():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        return None
    parts = local_ip.split(".")
    if len(parts) != 4:
        return None
    base = ".".join(parts[:3])
    return f"{base}.1-254"

# ---------- REAL MODEL WRAPPER ----------
class LocalAIPredictor:
    def __init__(self, use_gpu=True):
        self.use_gpu_requested = use_gpu
        self.backend = "heuristic"
        self.device = "cpu"
        self.model = None

        try:
            self.torch = importlib.import_module("torch")
        except ImportError:
            self.torch = None

        if self.torch is None:
            self.backend = "heuristic"
            return

        if use_gpu and self.torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            # self.model = self.torch.jit.load("mr_freeze_risk_model.pt", map_location=self.device)
            self.model = None
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                self.backend = "torch"
            else:
                self.backend = "heuristic"
        except Exception:
            self.backend = "heuristic"
            self.model = None
            self.device = "cpu"

    def _build_feature_vector(self, machine):
        lat = machine.latency if machine.latency is not None else 0.0
        avg_lat = statistics.mean(machine.latency_history) if machine.latency_history else 0.0
        loss_rate = (sum(machine.loss_history) / len(machine.loss_history)) if machine.loss_history else 0.0
        jitter_avg = statistics.mean(machine.jitter_history) if machine.jitter_history else 0.0
        fused = machine.fused_anomaly
        threat = machine.threat_score
        flaps = machine.flap_count
        offline = machine.offline_count

        profile_map = {
            "UNKNOWN": 0,
            "STABLE": 1,
            "SPIKY": 2,
            "UNSTABLE": 3,
            "FLAPPY": 4,
            "DEAD-PRONE": 5,
        }
        profile_code = profile_map.get(machine.behavior_profile, 0)

        return [
            float(lat),
            float(avg_lat),
            float(loss_rate),
            float(jitter_avg),
            float(fused),
            float(threat),
            float(flaps),
            float(offline),
            float(profile_code),
        ]

    def _predict_torch(self, machine):
        if self.model is None:
            return self._predict_heuristic(machine)

        torch = self.torch
        feats = self._build_feature_vector(machine)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        risk = out.squeeze().item()
        risk = max(0.0, min(1.0, float(risk)))

        if risk >= 0.8:
            label = "OFFLINE_RISK"
            conf = risk
        elif risk >= 0.6:
            label = "DEGRADE_SOON"
            conf = risk
        elif risk >= 0.4:
            label = "SPIKE_RISK"
            conf = risk
        elif risk >= 0.2:
            label = "WATCH"
            conf = 0.5 + risk / 2.0
        else:
            label = "STABLE"
            conf = 0.6 + (0.2 - risk)

        conf = max(0.0, min(1.0, conf))
        return label, conf

    def _predict_heuristic(self, machine):
        if not machine.latency_history or len(machine.latency_history) < 5:
            return "UNKNOWN", 0.0

        recent = machine.latency_history[-10:] if len(machine.latency_history) >= 10 else machine.latency_history
        t = list(range(len(recent)))
        avg = statistics.mean(recent)
        stdev = statistics.pstdev(recent) if len(recent) > 1 else 0.0

        if len(recent) > 1:
            n = len(recent)
            mean_t = statistics.mean(t)
            mean_y = avg
            num = sum((ti - mean_t) * (yi - mean_y) for ti, yi in zip(t, recent))
            den = sum((ti - mean_t) ** 2 for ti in t) or 1.0
            slope = num / den
        else:
            slope = 0.0

        loss_window = machine.loss_history[-50:] if machine.loss_history else []
        loss_rate = sum(loss_window) / max(1, len(loss_window)) if loss_window else 0.0

        fused = machine.fused_anomaly
        threat = machine.threat_score

        label = "STABLE"
        conf = 0.3

        if machine.status == "OFFLINE":
            label = "OFFLINE"
            conf = 1.0
        elif loss_rate > 0.3 or threat > 70 or fused > 0.8:
            label = "OFFLINE_RISK"
            conf = min(1.0, 0.5 + fused)
        elif slope > 5 and avg > 150:
            label = "DEGRADE_SOON"
            conf = min(1.0, 0.4 + slope / 50.0)
        elif stdev > 50 and fused > 0.5:
            label = "SPIKE_RISK"
            conf = min(1.0, 0.4 + fused)
        elif fused < 0.3 and threat < 30 and loss_rate < 0.1:
            label = "STABLE"
            conf = 0.7

        return label, max(0.0, min(1.0, conf))

    def predict(self, machine):
        if self.backend == "torch":
            try:
                return self._predict_torch(machine)
            except Exception:
                return self._predict_heuristic(machine)
        else:
            return self._predict_heuristic(machine)

# ---------- MACHINE MODEL ----------
class Machine:
    def __init__(self, address, status="RUNNING", tags=None, frozen=False,
                 name=None, mac=None, vendor=None, open_ports=None,
                 long_term_stats=None, prediction_label="UNKNOWN", prediction_confidence=0.0):
        self.address = address
        self.status = status
        self.latency = None
        self.tags = tags if tags is not None else ["TEST"]

        self.name = name or address
        self.mac = mac
        self.vendor = vendor or "Unknown"
        self.open_ports = open_ports if open_ports is not None else []

        self.last_status_change = datetime.utcnow()
        self.last_latency = None
        self.latency_history = []
        self.jitter_history = []
        self.loss_history = []
        self.offline_count = 0
        self.flap_count = 0
        self.auto_frozen = frozen
        self.manual_frozen = False
        self.sensitive_frozen = False
        self.last_seen = None

        self.behavior_profile = "UNKNOWN"
        self.threat_score = 0
        self.anomaly_score = 0.0
        self.fused_anomaly = 0.0
        self.last_root_cause = "None"

        self.map_x = random.uniform(100, 400)
        self.map_y = random.uniform(100, 250)
        self.vx = 0.0
        self.vy = 0.0

        self.long_term_stats = long_term_stats or {
            "total_pings": 0,
            "total_offline_events": 0,
            "total_freezes": 0,
            "avg_latency": None,
            "avg_loss": None,
            "avg_jitter": None,
            "last_update": None,
        }

        self.prediction_label = prediction_label
        self.prediction_confidence = prediction_confidence

        self._stop_event = threading.Event()
        self._thread = None

    def real_ping(self):
        ip = self.address.split(":")[0]
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", "1", "-w", "1000", ip]
        else:
            cmd = ["ping", "-c", "1", "-W", "1", ip]

        start = time.time()
        success = False
        try:
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            success = True
        except subprocess.CalledProcessError:
            success = False
        end = time.time()

        if success:
            self.latency = int((end - start) * 1000)
            self.last_latency = self.latency
            self.latency_history.append(self.latency)
            if len(self.latency_history) > 200:
                self.latency_history.pop(0)

            if len(self.latency_history) >= 2:
                j = abs(self.latency_history[-1] - self.latency_history[-2])
                self.jitter_history.append(j)
                if len(self.jitter_history) > 200:
                    self.jitter_history.pop(0)

            self.loss_history.append(0)
            if len(self.loss_history) > 200:
                self.loss_history.pop(0)

            self.last_seen = datetime.utcnow()
            if self.status == "OFFLINE":
                self.flap_count += 1
            if not self.is_frozen():
                self.status = "RUNNING"
        else:
            self.latency = None
            self.offline_count += 1
            self.loss_history.append(1)
            if len(self.loss_history) > 200:
                self.loss_history.pop(0)
            self.status = "OFFLINE"

        self.update_behavior_profile()
        self.update_threat_score()
        self.update_anomaly_score()
        self.update_fused_anomaly()
        self.update_long_term_stats()
        self.update_root_cause()

    def is_frozen(self):
        return self.manual_frozen or self.auto_frozen or self.sensitive_frozen

    def freeze(self, manual=True, sensitive=False):
        if sensitive:
            self.sensitive_frozen = True
        elif manual:
            self.manual_frozen = True
        else:
            self.auto_frozen = True
        if self.status != "OFFLINE":
            self.status = "FROZEN"
        self.last_status_change = datetime.utcnow()
        self.long_term_stats["total_freezes"] += 1

    def unfreeze(self):
        self.manual_frozen = False
        self.auto_frozen = False
        self.sensitive_frozen = False
        if self.status != "OFFLINE":
            self.status = "RUNNING"
        self.last_status_change = datetime.utcnow()

    def enrich_identity(self, common_ports=None):
        ip = self.address.split(":")[0]

        rdns = reverse_dns_name(ip)
        if rdns:
            if looks_sensitive(rdns):
                self.sensitive_frozen = True
            self.name = rdns

        mac = get_mac_from_arp(ip)
        if mac:
            if looks_sensitive(mac):
                self.sensitive_frozen = True
            self.mac = mac
            self.vendor = lookup_vendor(mac)

        if common_ports is None:
            common_ports = [22, 80, 443, 3389, 502, 8080]
        ports = scan_ports(ip, common_ports)
        self.open_ports = ports

    def update_behavior_profile(self):
        if not self.latency_history:
            self.behavior_profile = "UNKNOWN"
            return

        if self.status == "OFFLINE":
            if self.offline_count > 10:
                self.behavior_profile = "DEAD-PRONE"
            else:
                self.behavior_profile = "UNSTABLE"
            return

        lat = self.latency_history[-50:] if len(self.latency_history) > 50 else self.latency_history
        if len(lat) < 3:
            self.behavior_profile = "UNKNOWN"
            return

        avg = statistics.mean(lat)
        stdev = statistics.pstdev(lat)
        loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))

        if loss_rate > 0.3:
            self.behavior_profile = "UNSTABLE"
        elif stdev < 10 and avg < 80:
            self.behavior_profile = "STABLE"
        elif stdev > 50 and avg < 200:
            self.behavior_profile = "SPIKY"
        elif self.flap_count >= 3:
            self.behavior_profile = "FLAPPY"
        else:
            self.behavior_profile = "UNSTABLE"

    def update_threat_score(self):
        score = 0
        if self.status == "OFFLINE":
            score += 60
            self.long_term_stats["total_offline_events"] += 1
        elif self.status == "FROZEN":
            score += 30

        if self.latency is not None:
            if self.latency > 300:
                score += 30
            elif self.latency > 200:
                score += 20
            elif self.latency > 100:
                score += 10

        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            if javg > 50:
                score += 20
            elif javg > 20:
                score += 10

        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            if loss_rate > 0.3:
                score += 30
            elif loss_rate > 0.1:
                score += 15

        if self.flap_count >= 3:
            score += 20

        if self.behavior_profile in ("UNSTABLE", "DEAD-PRONE", "FLAPPY"):
            score += 10

        if self.sensitive_frozen:
            score += 10

        self.threat_score = max(0, min(100, score))

    def update_anomaly_score(self):
        if len(self.latency_history) < 10 or self.latency is None:
            self.anomaly_score = 0.0
            return
        window = self.latency_history[-50:]
        avg = statistics.mean(window)
        stdev = statistics.pstdev(window) or 1.0
        z = abs(self.latency - avg) / stdev
        self.anomaly_score = z

    def update_fused_anomaly(self):
        lat_component = min(self.anomaly_score / 5.0, 1.0)
        jitter_component = 0.0
        loss_component = 0.0
        flap_component = min(self.flap_count / 5.0, 1.0)
        threat_component = self.threat_score / 100.0

        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            jitter_component = min(javg / 100.0, 1.0)

        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            loss_component = min(loss_rate * 3.0, 1.0)

        self.fused_anomaly = max(0.0, min(1.0, 0.3 * lat_component +
                                          0.2 * jitter_component +
                                          0.2 * loss_component +
                                          0.1 * flap_component +
                                          0.2 * threat_component))

    def update_long_term_stats(self):
        self.long_term_stats["total_pings"] += 1
        self.long_term_stats["last_update"] = datetime.utcnow().isoformat()

        if self.latency_history:
            self.long_term_stats["avg_latency"] = statistics.mean(self.latency_history)
        if self.jitter_history:
            self.long_term_stats["avg_jitter"] = statistics.mean(self.jitter_history)
        if self.loss_history:
            self.long_term_stats["avg_loss"] = sum(self.loss_history) / len(self.loss_history)

    def update_root_cause(self):
        reasons = []
        if self.status == "OFFLINE":
            reasons.append("offline")
        if self.latency is not None and self.latency > 250:
            reasons.append("high latency")
        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            if javg > 40:
                reasons.append("high jitter")
        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            if loss_rate > 0.2:
                reasons.append("packet loss")
        if self.flap_count >= 3:
            reasons.append("flapping")
        if self.sensitive_frozen:
            reasons.append("privacy freeze")

        if not reasons:
            self.last_root_cause = "Stable"
        else:
            self.last_root_cause = ", ".join(reasons)

    def start_ping_loop(self, callback, interval=1.0):
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while not self._stop_event.is_set():
                self.real_ping()
                callback(self)
                time.sleep(interval)

        self._stop_event.clear()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_ping_loop(self):
        self._stop_event.set()

    def to_dict(self):
        return {
            "address": self.address,
            "status": self.status,
            "tags": self.tags,
            "auto_frozen": self.auto_frozen,
            "manual_frozen": self.manual_frozen,
            "sensitive_frozen": self.sensitive_frozen,
            "name": self.name,
            "mac": self.mac,
            "vendor": self.vendor,
            "open_ports": self.open_ports,
            "behavior_profile": self.behavior_profile,
            "threat_score": self.threat_score,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "long_term_stats": self.long_term_stats,
            "prediction_label": self.prediction_label,
            "prediction_confidence": self.prediction_confidence,
        }

    @staticmethod
    def from_dict(data):
        m = Machine(
            data["address"],
            status=data.get("status", "RUNNING"),
            tags=data.get("tags", ["TEST"]),
            frozen=data.get("auto_frozen", False),
            name=data.get("name"),
            mac=data.get("mac"),
            vendor=data.get("vendor"),
            open_ports=data.get("open_ports", []),
            long_term_stats=data.get("long_term_stats", None),
            prediction_label=data.get("prediction_label", "UNKNOWN"),
            prediction_confidence=data.get("prediction_confidence", 0.0),
        )
        m.manual_frozen = data.get("manual_frozen", False)
        m.sensitive_frozen = data.get("sensitive_frozen", False)
        m.behavior_profile = data.get("behavior_profile", "UNKNOWN")
        m.threat_score = data.get("threat_score", 0)
        m.map_x = data.get("map_x", random.uniform(100, 400))
        m.map_y = data.get("map_y", random.uniform(100, 250))
        return m

# ---------- PERSISTENT LOG ORGAN ----------
class PersistentLog:
    def __init__(self, memory_dir_getter, admin_mode_getter):
        self.memory_dir_getter = memory_dir_getter
        self.admin_mode_getter = admin_mode_getter
        self.log_buffer = []  # in-memory lines
        self.log_lock = threading.Lock()
        self.background_loaded = False
        self.background_thread = None

    def _log_path(self):
        return os.path.join(self.memory_dir_getter(), LOG_FILENAME)

    def _rotate_if_needed(self):
        path = self._log_path()
        if not os.path.exists(path):
            return
        size = os.path.getsize(path)
        if size <= LOG_MAX_SIZE_BYTES:
            return
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive = os.path.join(self.memory_dir_getter(), f"mr_freeze_log_v4_8_{ts}.log.enc")
        try:
            os.rename(path, archive)
        except Exception:
            pass

    def append(self, raw_text: str):
        admin = self.admin_mode_getter()
        safe_text = mask_sensitive(raw_text, admin=admin)
        line = f"{datetime.utcnow().isoformat()} | {safe_text}"
        with self.log_lock:
            self.log_buffer.append(line)
        try:
            self._rotate_if_needed()
            path = self._log_path()
            enc_line = encrypt_bytes((line + "\n").encode("utf-8"))
            with open(path, "ab") as f:
                f.write(enc_line)
        except Exception:
            pass

    def load_initial(self):
        path = self._log_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "rb") as f:
                enc = f.read()
            if not enc:
                return []
            raw = decrypt_bytes(enc).decode("utf-8", errors="ignore")
            lines = [l for l in raw.splitlines() if l.strip()]
            if len(lines) <= LOG_INITIAL_LINES:
                with self.log_lock:
                    self.log_buffer = lines[:]
                return lines
            initial = lines[-LOG_INITIAL_LINES:]
            with self.log_lock:
                self.log_buffer = initial[:]
            # start background loader for older lines
            self.background_thread = threading.Thread(
                target=self._background_load_older,
                args=(lines[:-LOG_INITIAL_LINES],),
                daemon=True
            )
            self.background_thread.start()
            return initial
        except Exception:
            return []

    def _background_load_older(self, older_lines):
        idx = len(older_lines)
        while idx > 0:
            batch_start = max(0, idx - LOG_BATCH_LOAD)
            batch = older_lines[batch_start:idx]
            with self.log_lock:
                self.log_buffer = batch + self.log_buffer
            idx = batch_start
            time.sleep(LOG_BACKGROUND_SLEEP)
        self.background_loaded = True

    def get_buffer_snapshot(self):
        with self.log_lock:
            return list(self.log_buffer)

# ---------- COCKPIT ----------
class MrFreezeCockpit:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mr Freeze v4.9 – Predictive Swarm Cockpit")
        self.root.configure(bg="#050814")

        self.machines = []
        self.rows = []
        self.selected_index = None

        self.avg_latency = None
        self.running_count = 0
        self.frozen_count = 0
        self.offline_count = 0
        self.degraded_count = 0
        self.last_scan_summary = "Never"
        self.auto_scan_subnet = None
        self.auto_scan_interval = 60

        self.scan_stop_event = threading.Event()
        self.scan_lock = threading.Lock()

        self.last_ui_refresh = datetime.utcnow()
        self.last_auto_scan = None

        self.memory_dir = DEFAULT_MEMORY_DIR
        self.state_dirty = False

        self.ai_predictor = LocalAIPredictor(use_gpu=True)

        self.admin_mode = False
        self.outbound_allowed = False  # Outbound Shield: False = BLOCK, True = ALLOW

        # Persistent log organ
        self.log_organ = PersistentLog(
            memory_dir_getter=lambda: self.memory_dir,
            admin_mode_getter=lambda: self.admin_mode
        )

        self.build_ui()
        self.load_state()

        # Load logs (initial fast batch)
        initial_logs = self.log_organ.load_initial()
        for line in initial_logs:
            self.log.insert("end", line + "\n")
        self.log.see("end")

        self.log_event("[MR FREEZE v4.9] Cockpit initialized.")

        threading.Thread(target=self.startup_scan, daemon=True).start()

        self.start_background_organs()
        self.root.mainloop()

    # ---------- UI BUILD ----------
    def build_ui(self):
        title = tk.Label(
            self.root,
            text="MR FREEZE v4.9",
            font=("Consolas", 24, "bold"),
            fg="#00ff9d",
            bg="#050814"
        )
        title.pack(pady=5)

        main_frame = tk.Frame(self.root, bg="#050814")
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame, bg="#050814")
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame, bg="#050814")
        right_frame.pack(side="right", fill="y")

        self.map_canvas = tk.Canvas(
            left_frame,
            width=500,
            height=300,
            bg="#050814",
            highlightthickness=1,
            highlightbackground="#111833"
        )
        self.map_canvas.pack(padx=10, pady=10, fill="x")

        table_container = tk.Frame(left_frame, bg="#050814")
        table_container.pack(padx=10, pady=10, fill="both", expand=True)

        ROW_HEIGHT = 26
        VISIBLE_ROWS = 10

        self.table_canvas = tk.Canvas(
            table_container,
            bg="#0b1020",
            highlightthickness=0,
            height=ROW_HEIGHT * VISIBLE_ROWS
        )
        self.table_canvas.pack(side="left", fill="both", expand=False)

        scrollbar = tk.Scrollbar(
            table_container,
            orient="vertical",
            command=self.table_canvas.yview
        )
        scrollbar.pack(side="right", fill="y")

        self.table_canvas.configure(yscrollcommand=scrollbar.set)

        self.table = tk.Frame(self.table_canvas, bg="#0b1020")
        self.table_canvas.create_window((0, 0), window=self.table, anchor="nw")

        def update_scroll_region(event):
            self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))

        self.table.bind("<Configure>", update_scroll_region)

        self.table_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.table_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        )

        headers = [
            "Machine", "Status", "Latency", "Jitter", "Loss",
            "Threat", "Profile", "Anomaly", "Fused", "Prediction",
            "Root Cause", "MAC", "Vendor", "Ports"
        ]
        for i, h in enumerate(headers):
            tk.Label(
                self.table,
                text=h,
                fg="#00c8ff",
                bg="#0b1020",
                font=("Consolas", 11, "bold")
            ).grid(row=0, column=i, padx=6, pady=5, sticky="w")

        self.controls = tk.Frame(left_frame, bg="#050814")
        self.controls.pack(pady=10)

        def add_btn(text, cmd, bg, fg="black", col=0):
            b = tk.Button(
                self.controls,
                text=text,
                command=cmd,
                bg=bg,
                fg=fg,
                width=13
            )
            b.grid(row=0, column=col, padx=3)
            return b

        add_btn("Add Machine", self.add_machine_popup, "#00c8ff", col=0)
        add_btn("Freeze Selected", self.freeze_selected, "#ff0055", "white", col=1)
        add_btn("Unfreeze Selected", self.unfreeze_selected, "#00ff9d", col=2)
        add_btn("Edit Tags", self.edit_tags_popup, "#b400ff", "white", col=3)
        add_btn("Scan Network", self.scan_network_popup, "#ffaa00", col=4)
        add_btn("Set Auto Scan", self.auto_scan_popup, "#ff8800", col=5)
        add_btn("Stop Scan", self.stop_scan, "#ff4444", "white", col=6)
        add_btn("Remove Selected", self.remove_selected, "#4444ff", "white", col=7)
        add_btn("Memory Drive", self.open_memory_drive_picker, "#00c8ff", "black", col=8)

        self.admin_button = tk.Button(
            self.controls,
            text="Admin Mode: OFF",
            command=self.toggle_admin_mode,
            bg="#333333",
            fg="white",
            width=16
        )
        self.admin_button.grid(row=0, column=9, padx=3)

        # Outbound Shield button (new)
        self.shield_button = tk.Button(
            self.controls,
            text="Outbound Shield: BLOCK",
            command=self.toggle_outbound_shield,
            bg="#aa0000",   # red
            fg="white",
            width=18
        )
        self.shield_button.grid(row=0, column=10, padx=3)

        self.command_frame = tk.Frame(left_frame, bg="#050814")
        self.command_frame.pack(pady=5)

        tk.Button(
            self.command_frame,
            text="Freeze Unstable",
            command=self.cmd_freeze_unstable,
            bg="#ff5500",
            fg="white",
            width=14
        ).grid(row=0, column=0, padx=3)

        tk.Button(
            self.command_frame,
            text="Unfreeze Stable",
            command=self.cmd_unfreeze_stable,
            bg="#00aa88",
            fg="white",
            width=14
        ).grid(row=0, column=1, padx=3)

        tk.Button(
            self.command_frame,
            text="Purge Offline",
            command=self.cmd_purge_offline,
            bg="#aa00aa",
            fg="white",
            width=14
        ).grid(row=0, column=2, padx=3)

        tk.Button(
            self.command_frame,
            text="Highlight Anomalies",
            command=self.cmd_highlight_anomalies,
            bg="#ffaa00",
            fg="black",
            width=18
        ).grid(row=0, column=3, padx=3)

        self.log = tk.Text(
            left_frame,
            height=8,
            bg="#0b1020",
            fg="#00c8ff",
            insertbackground="#00ff9d"
        )
        self.log.pack(padx=10, pady=10, fill="both", expand=True)

        sidebar_title = tk.Label(
            right_frame,
            text="TACTICAL SIDEBAR",
            font=("Consolas", 14, "bold"),
            fg="#00c8ff",
            bg="#050814"
        )
        sidebar_title.pack(pady=5)

        self.sidebar_frame = tk.Frame(right_frame, bg="#050814")
        self.sidebar_frame.pack(padx=10, pady=5, fill="y")

        def make_metric(label_text):
            frame = tk.Frame(self.sidebar_frame, bg="#050814")
            frame.pack(anchor="w", pady=2, fill="x")
            lbl = tk.Label(frame, text=label_text, fg="#00c8ff", bg="#050814", font=("Consolas", 10))
            lbl.pack(side="left")
            val = tk.Label(frame, text="—", fg="#00ff9d", bg="#050814", font=("Consolas", 10, "bold"))
            val.pack(side="right")
            return val

        self.metric_nodes = make_metric("Nodes:")
        self.metric_running = make_metric("Running:")
        self.metric_frozen = make_metric("Frozen:")
        self.metric_offline = make_metric("Offline:")
        self.metric_degraded = make_metric("Degraded:")
        self.metric_avg_latency = make_metric("Avg Latency:")
        self.metric_avg_jitter = make_metric("Avg Jitter:")
        self.metric_avg_loss = make_metric("Avg Loss:")
        self.metric_best = make_metric("Best Node:")
        self.metric_worst = make_metric("Worst Node:")
        self.metric_last_scan = make_metric("Last Scan:")
        self.metric_watchdog = make_metric("Watchdog:")
        self.metric_anomaly = make_metric("Global Anomaly:")
        self.metric_swarm_health = make_metric("Swarm Health:")
        self.metric_predicted_risk = make_metric("Predicted Risk:")
        self.metric_memory_path = make_metric("Memory Path:")

    # ---------- ADMIN MODE TOGGLE ----------
    def toggle_admin_mode(self):
        self.admin_mode = not self.admin_mode
        if self.admin_mode:
            self.admin_button.configure(text="Admin Mode: ON", bg="#ff00aa")
            self.log_event("[ADMIN MODE] ENABLED – full visibility, no autonomous actions, privacy firewall still active.")
        else:
            self.admin_button.configure(text="Admin Mode: OFF", bg="#333333")
            self.log_event("[ADMIN MODE] DISABLED – autonomous actions and privacy masking restored.")
        self.refresh_table()
        self.draw_node_map()

    # ---------- OUTBOUND SHIELD TOGGLE ----------
    def toggle_outbound_shield(self):
        self.outbound_allowed = not self.outbound_allowed
        if self.outbound_allowed:
            self.shield_button.configure(text="Outbound Shield: ALLOW", bg="#00aa00", fg="black")
            self.log_event("[SHIELD] Outbound summaries allowed (still masked, no personal identifiers).")
        else:
            self.shield_button.configure(text="Outbound Shield: BLOCK", bg="#aa0000", fg="white")
            self.log_event("[SHIELD] Outbound summaries blocked. Local-only operation.")

    # ---------- SWARM MAP ----------
    def update_swarm_physics(self):
        if not self.machines:
            self.map_canvas.after(80, self.update_swarm_physics)
            return

        width = int(self.map_canvas.winfo_width() or 500)
        height = int(self.map_canvas.winfo_height() or 300)
        cx, cy = width / 2, height / 2

        k_repulsion = 2000.0
        k_center = 0.02
        damping = 0.85

        for i, m1 in enumerate(self.machines):
            fx, fy = 0.0, 0.0

            for j, m2 in enumerate(self.machines):
                if i == j:
                    continue
                dx = m1.map_x - m2.map_x
                dy = m1.map_y - m2.map_y
                dist2 = dx * dx + dy * dy + 0.01
                force = k_repulsion / dist2
                fx += (dx / (dist2 ** 0.5)) * force
                fy += (dy / (dist2 ** 0.5)) * force

            dx_c = cx - m1.map_x
            dy_c = cy - m1.map_y
            fx += dx_c * k_center
            fy += dy_c * k_center

            threat_factor = (m1.threat_score / 100.0)
            fx += dx_c * k_center * threat_factor
            fy += dy_c * k_center * threat_factor

            m1.vx = (m1.vx + fx * 0.02) * damping
            m1.vy = (m1.vy + fy * 0.02) * damping

        for m in self.machines:
            m.map_x += m.vx
            m.map_y += m.vy
            m.map_x = max(30, min(width - 30, m.map_x))
            m.map_y = max(30, min(height - 30, m.map_y))

        self.draw_node_map()
        self.map_canvas.after(80, self.update_swarm_physics)

    def draw_node_map(self):
        self.map_canvas.delete("all")
        if len(self.machines) > 1:
            for m in self.machines:
                nearest = None
                nearest_d2 = None
                for n in self.machines:
                    if n is m:
                        continue
                    dx = m.map_x - n.map_x
                    dy = m.map_y - n.map_y
                    d2 = dx * dx + dy * dy
                    if nearest is None or d2 < nearest_d2:
                        nearest = n
                        nearest_d2 = d2
                if nearest:
                    self.map_canvas.create_line(
                        m.map_x, m.map_y, nearest.map_x, nearest.map_y,
                        fill="#111833"
                    )

        for m in self.machines:
            r = 8
            color = "#00ff9d"
            if m.status == "OFFLINE":
                color = "#b400ff"
            elif m.is_frozen():
                color = "#ff0055"
            elif m.threat_score >= 60:
                color = "#ff5500"
            elif m.threat_score >= 40:
                color = "#ffaa00"

            outline = "#050814"
            if m.sensitive_frozen and not self.admin_mode:
                color = "#222233"
                outline = "#111122"

            x, y = m.map_x, m.map_y
            self.map_canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color,
                outline=outline
            )
            label = m.name
            if not self.admin_mode and looks_sensitive(label):
                label = "[MASKED]"
            self.map_canvas.create_text(
                x, y - 12,
                text=label,
                fill="#00c8ff",
                font=("Consolas", 8)
            )

    # ---------- STARTUP SCAN ----------
    def startup_scan(self):
        subnet = detect_primary_lan_range()
        if not subnet:
            self.log_event("[STARTUP] Could not detect primary LAN.")
            return
        self.log_event(f"[STARTUP] Detected primary LAN: {display_address(subnet, admin=False)}. Scanning once...")
        self.scan_network(subnet, label_prefix="[STARTUP] ")

    # ---------- BASIC CONTROLS ----------
    def add_machine_popup(self):
        win = tk.Toplevel(self.root)
        win.title("Add Machine")
        win.configure(bg="#050814")

        tk.Label(
            win,
            text="IP:PORT or IP",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        entry = tk.Entry(win)
        entry.pack(pady=5)

        def add():
            addr = entry.get().strip()
            if addr:
                self.add_machine(addr)
            win.destroy()

        tk.Button(
            win,
            text="Add",
            command=add,
            bg="#00ff9d",
            fg="black"
        ).pack(pady=5)

    def add_machine(self, address):
        for m in self.machines:
            if m.address == address:
                self.log_event(f"[ADD] {display_address(address, admin=self.admin_mode)} already present.")
                return

        m = Machine(address)
        self.machines.append(m)
        m.start_ping_loop(self.on_machine_update)
        self.log_event(f"[ADD] {display_address(address, admin=self.admin_mode)}")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    def remove_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.machines):
            self.log_event("[INFO] No machine selected to remove.")
            return
        m = self.machines[self.selected_index]
        m.stop_ping_loop()
        addr = m.address
        del self.machines[self.selected_index]
        self.selected_index = None
        self.log_event(f"[REMOVE] {display_address(addr, admin=self.admin_mode)}")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    # ---------- AUTO SCAN ----------
    def auto_scan_popup(self):
        win = tk.Toplevel(self.root)
        win.title("Auto Scan Configuration")
        win.configure(bg="#050814")

        tk.Label(
            win,
            text="Subnet/IP range (e.g., 192.168.1.1-50 or 192.168.1.0/24):",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        entry = tk.Entry(win, width=30)
        if self.auto_scan_subnet:
            entry.insert(0, self.auto_scan_subnet)
        else:
            primary = detect_primary_lan_range()
            if primary:
                entry.insert(0, primary)
        entry.pack(pady=5)

        tk.Label(
            win,
            text="Interval (seconds):",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        interval_entry = tk.Entry(win, width=10)
        interval_entry.insert(0, str(self.auto_scan_interval))
        interval_entry.pack(pady=5)

        def save():
            subnet = entry.get().strip()
            try:
                interval = int(interval_entry.get().strip())
                if interval < 10:
                    interval = 10
            except ValueError:
                interval = 60
            self.auto_scan_subnet = subnet if subnet else None
            self.auto_scan_interval = interval
            if self.auto_scan_subnet:
                self.log_event(f"[AUTO-SCAN] Enabled on {display_address(self.auto_scan_subnet, admin=False)} every {self.auto_scan_interval}s")
            else:
                self.log_event("[AUTO-SCAN] Disabled")
            self.state_dirty = True
            self.save_state_if_dirty()
            win.destroy()

        tk.Button(
            win,
            text="Save",
            command=save,
            bg="#00ff9d",
            fg="black"
        ).pack(pady=5)

    def scan_network_popup(self):
        win = tk.Toplevel(self.root)
        win.title("Network Scan")
        win.configure(bg="#050814")

        tk.Label(
            win,
            text="Enter subnet or IP range (e.g., 192.168.1.1-50 or 192.168.1.0/24):",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        entry = tk.Entry(win, width=30)
        primary = detect_primary_lan_range()
        if primary:
            entry.insert(0, primary)
        entry.pack(pady=5)

        def scan():
            subnet = entry.get().strip()
            if subnet:
                threading.Thread(target=self.scan_network, args=(subnet,), daemon=True).start()
            win.destroy()

        tk.Button(
            win,
            text="Start Scan",
            command=scan,
            bg="#00ff9d",
            fg="black"
        ).pack(pady=5)

    def stop_scan(self):
        self.scan_stop_event.set()
        self.log_event("[SCAN] Stop requested.")

    # ---------- ADAPTIVE THREAD-POOL SCANNER ----------
    def _estimate_thread_count(self, num_ips):
        min_threads = 20
        max_threads = 200
        try:
            cores = os.cpu_count() or 4
        except Exception:
            cores = 4
        base = max(min_threads, min(max_threads, cores * 10))
        if num_ips < base:
            base = max(min_threads, num_ips)
        return base

    def scan_network(self, subnet, label_prefix="[SCAN] "):
        with self.scan_lock:
            self.scan_stop_event.clear()
            self.log_event(f"{label_prefix}Starting network scan: {display_address(subnet, admin=False)}")

            ips = []
            try:
                if "-" in subnet:
                    base, rng = subnet.rsplit(".", 1)
                    start, end = map(int, rng.split("-"))
                    ips = [f"{base}.{i}" for i in range(start, end + 1)]
                elif "/" in subnet:
                    import ipaddress
                    net = ipaddress.ip_network(subnet, strict=False)
                    ips = [str(ip) for ip in net.hosts()]
                else:
                    ips = [subnet]
            except Exception as e:
                self.log_event(f"{label_prefix}[ERROR] Invalid subnet format: {e}")
                return

            if not ips:
                self.log_event(f"{label_prefix}[INFO] No IPs to scan.")
                return

            num_ips = len(ips)
            threads = self._estimate_thread_count(num_ips)
            self.log_event(f"{label_prefix}Adaptive scanner using {threads} threads for {num_ips} IPs.")

            live_hosts = []
            common_ports = [22, 80, 443, 3389, 502, 8080]

            def worker(ip):
                if self.scan_stop_event.is_set():
                    return None
                m = Machine(ip)
                m.real_ping()
                if m.status == "RUNNING":
                    m.enrich_identity(common_ports=common_ports)
                    return m
                return None

            try:
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    future_map = {executor.submit(worker, ip): ip for ip in ips}
                    for future in as_completed(future_map):
                        if self.scan_stop_event.is_set():
                            break
                        machine = future.result()
                        if machine is not None:
                            live_hosts.append(machine.address)

                            def add_scanned_machine(machine=machine):
                                for existing in self.machines:
                                    if existing.address == machine.address:
                                        return
                                self.machines.append(machine)
                                machine.start_ping_loop(self.on_machine_update)
                                self.state_dirty = True
                                self.refresh_table()
                                self.save_state_if_dirty()

                            self.root.after(0, add_scanned_machine)
            except Exception as e:
                self.log_event(f"{label_prefix}[ERROR] Scan failed: {e}")
                return

            if self.scan_stop_event.is_set():
                self.log_event(f"{label_prefix}Scan stopped by user.")
            summary = f"Live hosts: {', '.join(display_address(h, admin=False) for h in live_hosts) if live_hosts else 'None'}"
            self.last_scan_summary = summary
            self.log_event(f"{label_prefix}Network scan completed. {summary}")
            self.root.after(0, self.refresh_metrics)

    # ---------- TABLE / PREDICTION ----------
    def threat_label_and_color(self, m: Machine):
        score = m.threat_score
        if score >= 80:
            return "CRITICAL", "#ff0000"
        elif score >= 60:
            return "HIGH", "#ff5500"
        elif score >= 40:
            return "ELEVATED", "#ffaa00"
        elif score >= 20:
            return "WATCH", "#00c8ff"
        else:
            return "OK", "#00ff9d"

    def prediction_color(self, m: Machine):
        label = m.prediction_label
        conf = m.prediction_confidence
        if label in ("OFFLINE_RISK", "DEGRADE_SOON") and conf > 0.6:
            return "#ff0000"
        if label in ("SPIKE_RISK",) and conf > 0.5:
            return "#ffaa00"
        if label == "STABLE" and conf > 0.6:
            return "#00ff9d"
        return "#00c8ff"

    def refresh_table(self):
        for r in self.rows:
            for w in r:
                w.destroy()
        self.rows.clear()

        for i, m in enumerate(self.machines, start=1):
            row = []

            def make_select_callback(index):
                def on_click(event):
                    self.selected_index = index
                    self.highlight_selected()
                return on_click

            threat_label, threat_color = self.threat_label_and_color(m)

            if m.status == "OFFLINE":
                base_color = "#b400ff"
            elif m.is_frozen():
                base_color = "#ff0055"
            else:
                base_color = threat_color

            machine_label = f"{m.name} ({display_address(m.address, admin=self.admin_mode)})"
            if not self.admin_mode and looks_sensitive(machine_label):
                machine_label = mask_sensitive(machine_label, admin=False)

            jitter_text = "—"
            if m.jitter_history:
                jitter_text = f"{int(statistics.mean(m.jitter_history[-10:]))} ms"

            loss_text = "—"
            if m.loss_history:
                loss_rate = sum(m.loss_history[-50:]) / max(1, len(m.loss_history[-50:]))
                loss_text = f"{int(loss_rate * 100)}%"

            fused_color = "#00ff9d"
            if m.fused_anomaly > 0.7:
                fused_color = "#ff0000"
            elif m.fused_anomaly > 0.4:
                fused_color = "#ffaa00"

            label, conf = self.ai_predictor.predict(m)
            m.prediction_label = label
            m.prediction_confidence = conf
            pred_text = f"{label} ({int(conf * 100)}%)"
            pred_color = self.prediction_color(m)

            mac_display = m.mac if self.admin_mode else mask_sensitive(m.mac or "", admin=False) if m.mac else "—"
            vendor_display = m.vendor if self.admin_mode else mask_sensitive(m.vendor or "", admin=False) if m.vendor else "Unknown"
            ports_display = ", ".join(str(p) for p in m.open_ports) if m.open_ports else "—"

            fields = [
                (machine_label, base_color),
                (m.status, base_color),
                (f"{m.latency} ms" if m.latency is not None else "—", "#00c8ff"),
                (jitter_text, "#00c8ff"),
                (loss_text, "#00c8ff"),
                (f"{threat_label} ({m.threat_score})", threat_color),
                (m.behavior_profile, "#00c8ff"),
                (f"{m.anomaly_score:.2f}", "#ffaa00" if m.anomaly_score > 2.5 else "#00ff9d"),
                (f"{m.fused_anomaly:.2f}", fused_color),
                (pred_text, pred_color),
                (m.last_root_cause, "#ff8800" if m.last_root_cause != "Stable" else "#00ff9d"),
                (mac_display, "#00c8ff"),
                (vendor_display, "#00c8ff"),
                (ports_display, "#00c8ff"),
            ]

            for col, (text, color) in enumerate(fields):
                e = tk.Label(self.table, text=text, fg=color, bg="#0b1020", font=("Consolas", 10))
                e.grid(row=i, column=col, padx=4, pady=3, sticky="w")
                e.bind("<Button-1>", make_select_callback(i - 1))
                row.append(e)

            self.rows.append(row)

        self.highlight_selected()
        self.refresh_metrics()
        self.draw_node_map()

    def highlight_selected(self):
        for idx, row in enumerate(self.rows):
            for w in row:
                w.configure(bg="#111833" if idx == self.selected_index else "#0b1020")

    # ---------- TAGS / FREEZE ----------
    def edit_tags_popup(self):
        if self.selected_index is None or self.selected_index >= len(self.machines):
            self.log_event("[INFO] No machine selected for tag edit.")
            return

        m = self.machines[self.selected_index]

        win = tk.Toplevel(self.root)
        win.title(f"Edit Tags – {display_address(m.address, admin=self.admin_mode)}")
        win.configure(bg="#050814")

        tk.Label(
            win,
            text="Tags (comma-separated):",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        entry = tk.Entry(win, width=40)
        entry.insert(0, ", ".join(m.tags))
        entry.pack(pady=5)

        def save():
            raw = entry.get().strip()
            if raw:
                m.tags = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                m.tags = []
            self.log_event(f"[TAGS] {display_address(m.address, admin=self.admin_mode)} -> {', '.join(m.tags)}")
            self.state_dirty = True
            self.refresh_table()
            self.save_state_if_dirty()
            win.destroy()

        tk.Button(
            win,
            text="Save",
            command=save,
            bg="#00ff9d",
            fg="black"
        ).pack(pady=5)

    def freeze_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.machines):
            self.log_event("[INFO] No machine selected to freeze.")
            return
        m = self.machines[self.selected_index]
        m.freeze(manual=True)
        self.log_event(f"[FREEZE] {display_address(m.address, admin=self.admin_mode)} (manual)")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    def unfreeze_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.machines):
            self.log_event("[INFO] No machine selected to unfreeze.")
            return
        m = self.machines[self.selected_index]
        m.unfreeze()
        self.log_event(f"[UNFREEZE] {display_address(m.address, admin=self.admin_mode)}")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    # ---------- COMMAND ORGAN ----------
    def cmd_freeze_unstable(self):
        count = 0
        for m in self.machines:
            if m.behavior_profile in ("UNSTABLE", "DEAD-PRONE", "FLAPPY") or m.fused_anomaly > 0.6:
                if not m.is_frozen():
                    m.freeze(manual=True)
                    count += 1
        self.log_event(f"[CMD] Freeze Unstable -> {count} nodes frozen.")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    def cmd_unfreeze_stable(self):
        count = 0
        for m in self.machines:
            if m.behavior_profile == "STABLE" and m.fused_anomaly < 0.3:
                if m.is_frozen():
                    m.unfreeze()
                    count += 1
        self.log_event(f"[CMD] Unfreeze Stable -> {count} nodes unfrozen.")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    def cmd_purge_offline(self):
        before = len(self.machines)
        survivors = []
        for m in self.machines:
            if m.status == "OFFLINE" and m.offline_count > 10:
                m.stop_ping_loop()
            else:
                survivors.append(m)
        removed = before - len(survivors)
        self.machines = survivors
        self.selected_index = None
        self.log_event(f"[CMD] Purge Offline -> {removed} nodes removed.")
        self.state_dirty = True
        self.refresh_table()
        self.save_state_if_dirty()

    def cmd_highlight_anomalies(self):
        high = [m for m in self.machines if m.fused_anomaly > 0.6 or m.prediction_label in ("OFFLINE_RISK", "DEGRADE_SOON")]
        if not high:
            self.log_event("[CMD] Highlight Anomalies -> none above threshold.")
        else:
            addrs = ", ".join(display_address(m.address, admin=self.admin_mode) for m in high)
            self.log_event(f"[CMD] Highlight Anomalies -> {addrs}")
        self.refresh_table()

    # ---------- LOG ----------
    def log_event(self, msg):
        self.log_organ.append(msg)
        self.log.insert("end", f"{datetime.utcnow().isoformat()} | {mask_sensitive(msg, admin=self.admin_mode)}\n")
        self.log.see("end")

    # ---------- MEMORY DRIVE PICKER ----------
    def open_memory_drive_picker(self):
        win = tk.Toplevel(self.root)
        win.title("Memory Drive Selection")
        win.configure(bg="#050814")

        tk.Label(
            win,
            text="Current memory directory:",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        current_label = tk.Label(
            win,
            text=self.memory_dir,
            fg="#00ff9d",
            bg="#050814",
            font=("Consolas", 9)
        )
        current_label.pack(pady=5)

        tk.Label(
            win,
            text="Select new directory (local or SMB-mounted):",
            fg="#00c8ff",
            bg="#050814"
        ).pack(pady=5)

        entry = tk.Entry(win, width=50)
        entry.insert(0, self.memory_dir)
        entry.pack(pady=5)

        def browse():
            path = filedialog.askdirectory()
            if path:
                entry.delete(0, "end")
                entry.insert(0, path)

        tk.Button(
            win,
            text="Browse",
            command=browse,
            bg="#00c8ff",
            fg="black"
        ).pack(pady=5)

        def save():
            path = entry.get().strip()
            if not path:
                win.destroy()
                return
            if not os.path.isdir(path):
                self.log_event(f"[MEMORY] Invalid directory: {path}")
                win.destroy()
                return
            self.memory_dir = os.path.abspath(path)
            self.log_event(f"[MEMORY] Memory directory set to: {self.memory_dir}")
            self.state_dirty = True
            self.save_state_if_dirty()
            self.refresh_metrics()
            win.destroy()

        tk.Button(
            win,
            text="Save",
            command=save,
            bg="#00ff9d",
            fg="black"
        ).pack(pady=5)

    # ---------- METRICS / SWARM SINK ----------
    def refresh_metrics(self):
        total = len(self.machines)
        running = sum(1 for m in self.machines if m.status == "RUNNING")
        frozen = sum(1 for m in self.machines if m.is_frozen())
        offline = sum(1 for m in self.machines if m.status == "OFFLINE")
        degraded = sum(1 for m in self.machines if m.threat_score >= 40)

        self.running_count = running
        self.frozen_count = frozen
        self.offline_count = offline
        self.degraded_count = degraded

        latencies = [m.latency for m in self.machines if m.latency is not None]
        self.avg_latency = sum(latencies) / len(latencies) if latencies else None

        jitters = []
        losses = []
        anomalies = []
        fused = []
        pred_risks = []
        for m in self.machines:
            if m.jitter_history:
                jitters.extend(m.jitter_history[-10:])
            if m.loss_history:
                losses.extend(m.loss_history[-50:])
            anomalies.append(m.anomaly_score)
            fused.append(m.fused_anomaly)
            if m.prediction_label in ("OFFLINE_RISK", "DEGRADE_SOON", "SPIKE_RISK"):
                pred_risks.append(m.prediction_confidence)

        avg_jitter = statistics.mean(jitters) if jitters else None
        avg_loss = (sum(losses) / len(losses)) if losses else None
        avg_anomaly = statistics.mean(anomalies) if anomalies else 0.0
        avg_fused = statistics.mean(fused) if fused else 0.0
        avg_pred_risk = statistics.mean(pred_risks) if pred_risks else 0.0

        best_node = None
        worst_node = None
        if latencies:
            best = min(self.machines, key=lambda m: m.latency if m.latency is not None else 10**9)
            worst = max(self.machines, key=lambda m: m.latency if m.latency is not None else -1)
            best_node = f"{display_address(best.address, admin=self.admin_mode)} ({best.latency} ms)" if best.latency is not None else None
            worst_node = f"{display_address(worst.address, admin=self.admin_mode)} ({worst.latency} ms)" if worst.latency is not None else None

        swarm_health = max(0.0, min(1.0, 1.0 - avg_fused))

        self.metric_nodes.config(text=str(total))
        self.metric_running.config(text=str(running))
        self.metric_frozen.config(text=str(frozen))
        self.metric_offline.config(text=str(offline))
        self.metric_degraded.config(text=str(degraded))
        self.metric_avg_latency.config(text=f"{int(self.avg_latency)} ms" if self.avg_latency is not None else "—")
        self.metric_avg_jitter.config(text=f"{int(avg_jitter)} ms" if avg_jitter is not None else "—")
        self.metric_avg_loss.config(text=f"{int(avg_loss * 100)}%" if avg_loss is not None else "—")
        self.metric_best.config(text=best_node or "—")
        self.metric_worst.config(text=worst_node or "—")
        self.metric_last_scan.config(text=self.last_scan_summary)
        self.metric_anomaly.config(text=f"{avg_anomaly:.2f}")
        self.metric_swarm_health.config(text=f"{int(swarm_health * 100)}%")
        self.metric_predicted_risk.config(text=f"{int(avg_pred_risk * 100)}%")
        self.metric_memory_path.config(text=self.memory_dir)

        now = datetime.utcnow()
        ui_delta = (now - self.last_ui_refresh).total_seconds()
        self.metric_watchdog.config(text="OK" if ui_delta < 5 else f"LAG {int(ui_delta)}s")

        self.swarm_sink_export(avg_fused, avg_pred_risk)

    def swarm_sink_export(self, avg_fused, avg_pred_risk):
        # Respect Outbound Shield: only export if allowed
        if not self.outbound_allowed:
            return
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "node_count": len(self.machines),
                "avg_fused_anomaly": avg_fused,
                "avg_predicted_risk": avg_pred_risk,
            }
            path = os.path.join(self.memory_dir, "mr_freeze_swarm_summary.json")
            with open(path, "w") as f:
                json.dump(summary, f)
        except Exception:
            pass

    # ---------- BACKGROUND ORGANS ----------
    def start_background_organs(self):
        self.root.after(500, self.ui_refresh_loop)
        self.map_canvas.after(500, self.update_swarm_physics)
        threading.Thread(target=self.auto_scan_loop, daemon=True).start()
        threading.Thread(target=self.watchdog_loop, daemon=True).start()

        for m in self.machines:
            m.start_ping_loop(self.on_machine_update)

    def ui_refresh_loop(self):
        self.refresh_table()
        self.last_ui_refresh = datetime.utcnow()
        self.root.after(1000, self.ui_refresh_loop)

    def auto_scan_loop(self):
        while True:
            if self.auto_scan_subnet:
                try:
                    self.last_auto_scan = datetime.utcnow()
                    self.scan_network(self.auto_scan_subnet, label_prefix="[AUTO-SCAN] ")
                except Exception as e:
                    self.log_event(f"[AUTO-SCAN] ERROR: {e}")
            time.sleep(self.auto_scan_interval if self.auto_scan_subnet else 5)

    def watchdog_loop(self):
        while True:
            now = datetime.utcnow()
            ui_delta = (now - self.last_ui_refresh).total_seconds()
            if ui_delta > 10:
                self.log_event(f"[WATCHDOG] UI refresh delayed: {int(ui_delta)}s")
            time.sleep(5)

    # ---------- MACHINE UPDATE ----------
    def on_machine_update(self, machine: Machine):
        if machine.sensitive_frozen and not self.admin_mode:
            self.root.after(0, lambda addr=machine.address:
                            self.log_event(f"[PRIVACY] {display_address(addr, admin=False)} flagged as sensitive; data masked."))

        if not self.admin_mode:
            if not machine.is_frozen():
                if machine.latency is not None and machine.latency > 350:
                    machine.freeze(manual=False)
                    self.root.after(0, lambda addr=machine.address, lat=machine.latency:
                                    self.log_event(f"[AUTO-FREEZE] {display_address(addr, admin=False)} high latency {lat} ms"))
                    self.state_dirty = True
                elif machine.fused_anomaly > 0.8 or machine.prediction_label == "OFFLINE_RISK":
                    machine.freeze(manual=False)
                    self.root.after(0, lambda addr=machine.address, fa=machine.fused_anomaly:
                                    self.log_event(f"[AUTO-FREEZE] {display_address(addr, admin=False)} predictive risk / fused {fa:.2f}"))
                    self.state_dirty = True

            if machine.status == "RUNNING" and machine.auto_frozen:
                if (datetime.utcnow() - machine.last_status_change) > timedelta(seconds=10):
                    machine.unfreeze()
                    self.root.after(0, lambda addr=machine.address:
                                    self.log_event(f"[AUTO-UNFREEZE] {display_address(addr, admin=False)} recovered"))
                    self.state_dirty = True

        if machine.status == "OFFLINE" and machine.offline_count in (3, 10):
            self.root.after(0, lambda addr=machine.address, c=machine.offline_count:
                            self.log_event(f"[ALERT] {display_address(addr, admin=False)} offline {c} times."))

        self.root.after(0, self.refresh_metrics)
        self.save_state_if_dirty()

    # ---------- STATE PERSISTENCE ----------
    def get_state_path(self):
        return os.path.join(self.memory_dir, MEMORY_FILENAME)

    def save_state_if_dirty(self):
        if not self.state_dirty:
            return
        self.save_state()
        self.state_dirty = False

    def save_state(self):
        try:
            data = {
                "memory_dir": self.memory_dir,
                "machines": [m.to_dict() for m in self.machines],
                "auto_scan_subnet": self.auto_scan_subnet,
                "auto_scan_interval": self.auto_scan_interval,
                "outbound_allowed": self.outbound_allowed,
            }
            raw = json.dumps(data, indent=2).encode("utf-8")
            enc = encrypt_bytes(raw)
            path = self.get_state_path()
            with open(path, "wb") as f:
                f.write(enc)
        except Exception as e:
            self.log_event(f"[STATE] Save failed: {e}")

    def load_state(self):
        path = self.get_state_path()
        if not os.path.exists(path):
            legacy = os.path.join(DEFAULT_MEMORY_DIR, "mr_freeze_state_v4_4.json")
            if not os.path.exists(legacy):
                return
            try:
                with open(legacy, "r") as f:
                    data = json.load(f)
                self.auto_scan_subnet = data.get("auto_scan_subnet")
                self.auto_scan_interval = data.get("auto_scan_interval", 60)
                for mdata in data.get("machines", []):
                    m = Machine.from_dict(mdata)
                    self.machines.append(m)
                self.log_event(f"[STATE] Restored {len(self.machines)} machines from legacy state.")
            except Exception as e:
                self.log_event(f"[STATE] Legacy load failed: {e}")
            return

        try:
            with open(path, "rb") as f:
                enc = f.read()
            raw = decrypt_bytes(enc)
            data = json.loads(raw.decode("utf-8"))

            self.memory_dir = data.get("memory_dir", DEFAULT_MEMORY_DIR)
            self.auto_scan_subnet = data.get("auto_scan_subnet")
            self.auto_scan_interval = data.get("auto_scan_interval", 60)
            self.outbound_allowed = data.get("outbound_allowed", False)

            for mdata in data.get("machines", []):
                m = Machine.from_dict(mdata)
                self.machines.append(m)
            self.log_event(f"[STATE] Restored {len(self.machines)} machines from encrypted state.")
        except Exception as e:
            self.log_event(f"[STATE] Load failed: {e}")

# ---------- RUN ----------
if __name__ == "__main__":
    MrFreezeCockpit()

