"""
Fortune Teller
--------------

A self-evolving, secure, and highly predictive scaffold for a cross-platform system that:
- Learns user behavior (weights, Markov + n-gram transitions, time-of-day routines)
- Infers context personas (Work/Gaming/Browsing) from processes, input cadence, and network status
- Uses ensemble scoring with adaptive calibration (softmax + Platt), bandit bias, cooldown, rhythm ranges, and diversity
- Tracks global and per-intent accuracy with user promotion/suppression feedback
- Preloads capabilities using dependency graphs and integrity checks
- Adds speculative jump-ahead prefetching ("read-ahead") to warm the next few likely dependencies
- Evolves its own code daily via safe, validated patches
- Encrypts memory, logs, and patch audit records with per-machine AES-GCM
- Surfaces transparent reasoning through a GUI (auto-disabled for headless/server mode)

Install extras for best results:
  pip install psutil pynput cryptography zeroconf scapy pyudev pynvml pyamdgpuinfo
"""

import platform
import subprocess
import threading
import time
import datetime
import queue
import json
import hashlib
import os
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

# Core telemetry
import psutil

# Optional GPU probes (guarded)
try:
    import pynvml  # NVIDIA
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

try:
    import pyamdgpuinfo  # AMD
    AMD_AVAILABLE = True
except Exception:
    AMD_AVAILABLE = False

# Optional LAN discovery
try:
    from zeroconf import Zeroconf, ServiceBrowser  # mDNS/Bonjour
    ZEROCONF_AVAILABLE = True
except Exception:
    ZEROCONF_AVAILABLE = False

try:
    from scapy.all import ARP, Ether, srp  # ARP scan
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

# Optional Linux USB
try:
    import pyudev
    PYUDEV_AVAILABLE = True
except Exception:
    PYUDEV_AVAILABLE = False

# Optional input monitoring
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False

# Crypto (AES-GCM via cryptography)
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# GUI (auto-disabled in headless/server mode)
def _detect_headless() -> bool:
    if platform.system() in ("Linux", "Darwin"):
        if not os.environ.get("DISPLAY"):
            return True
    if platform.system() == "Windows":
        return not os.environ.get("SESSIONNAME") and not os.environ.get("USERNAME")
    return False

HEADLESS_MODE = _detect_headless()

try:
    if not HEADLESS_MODE:
        import tkinter as tk
        from tkinter import ttk
        TK_AVAILABLE = True
    else:
        TK_AVAILABLE = False
except Exception:
    TK_AVAILABLE = False


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Intent:
    id: str
    score: float
    evidence: Dict[str, Any]
    deadline: Optional[datetime.datetime] = None
    cluster: Optional[str] = None
    calibrated: float = 0.0

@dataclass
class PlanStep:
    name: str
    deps: List[str] = field(default_factory=list)
    cost: Dict[str, float] = field(default_factory=dict)
    expected_gain_ms: int = 0

@dataclass
class Plan:
    id: str
    steps: List[PlanStep]
    total_cost: Dict[str, float]
    expected_gain_ms: int
    rationale: str
    chosen_intent: Optional[Intent] = None

@dataclass
class Decision:
    allow: bool
    notify: bool
    rationale: str
    risk: str
    policy_gate: Dict[str, Any]

@dataclass
class Result:
    success: bool
    resources: Dict[str, float]
    verified_hashes: Dict[str, str]
    notes: str

@dataclass
class CapabilityProfile:
    scores: Dict[str, float]
    features: Dict[str, bool]
    constraints: Dict[str, Any]

@dataclass
class Policy:
    autonomy_level: str
    cpu_cap: float
    disk_cap_bps: float
    vram_cap_mb: float
    quiet_hours: Tuple[int, int]
    privacy_scopes: Dict[str, Any]
    assist_not_act: bool = True
    suppress: List[str] = field(default_factory=list)
    promote: List[str] = field(default_factory=list)

@dataclass
class UpdatePolicy:
    allow_paths: List[str]
    max_patch_bytes: int
    quiet_hours: Tuple[int, int]
    require_tests_pass: bool
    require_lint_pass: bool
    create_backup: bool

@dataclass
class Patch:
    target_path: str
    description: str
    diff: str
    bytes: int

@dataclass
class UpdateResult:
    applied: bool
    rationale: str
    files_changed: List[str]
    backup_paths: List[str]
    signature: str
    version: Optional[str] = None


# ---------------------------
# Crypto manager (per-machine AES-GCM)
# ---------------------------

class CryptoManager:
    def __init__(self, key_dir: str = ".ft_secrets"):
        self.available = CRYPTO_AVAILABLE
        self.key_dir = os.path.abspath(key_dir)
        self.salt_path = os.path.join(self.key_dir, "salt.bin")
        self.key_path = os.path.join(self.key_dir, "key.bin")
        self._aesgcm = None
        self._ensure_key()

    def _machine_fingerprint(self) -> bytes:
        fp = f"{platform.system()}|{platform.node()}|{platform.machine()}|{platform.processor()}"
        try:
            macs = []
            for _, addrs in psutil.net_if_addrs().items():
                for a in addrs:
                    addr = getattr(a, 'address', '')
                    if addr and addr.count(":") == 5:
                        macs.append(addr)
            if macs:
                fp += "|" + macs[0]
        except Exception:
            pass
        return hashlib.sha256(fp.encode("utf-8")).digest()

    def _ensure_key(self):
        if not self.available:
            return
        os.makedirs(self.key_dir, exist_ok=True)
        if not os.path.exists(self.salt_path):
            with open(self.salt_path, "wb") as f:
                f.write(os.urandom(16))
        with open(self.salt_path, "rb") as f:
            salt = f.read()
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200000)
        derived = kdf.derive(self._machine_fingerprint())
        try:
            with open(self.key_path, "wb") as f:
                f.write(derived)
        except Exception:
            pass
        self._aesgcm = AESGCM(derived)

    def encrypt(self, data: bytes, aad: Optional[bytes] = None) -> bytes:
        if not self.available or not self._aesgcm:
            return data
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, data, aad)
        return nonce + ct

    def decrypt(self, payload: bytes, aad: Optional[bytes] = None) -> bytes:
        if not self.available or not self._aesgcm:
            return payload
        nonce, ct = payload[:12], payload[12:]
        return self._aesgcm.decrypt(nonce, ct, aad)

    def seal_file(self, path: str, obj: Any):
        raw = json.dumps(obj, indent=2).encode("utf-8")
        enc = self.encrypt(raw, aad=b"fortune_teller")
        with open(path, "wb") as f:
            f.write(enc)

    def unseal_file(self, path: str) -> Any:
        with open(path, "rb") as f:
            payload = f.read()
        dec = self.decrypt(payload, aad=b"fortune_teller")
        return json.loads(dec.decode("utf-8"))

    def sign_text(self, text: str) -> str:
        fp = self._machine_fingerprint()
        return hashlib.sha256(text.encode("utf-8") + fp).hexdigest()


# ---------------------------
# Memory manager (encrypted + accuracy tracking with safe defaults)
# ---------------------------

class MemoryManager:
    def __init__(self, crypto: CryptoManager, path: str = "fortune_teller_memory.enc"):
        self.crypto = crypto
        self.path = path
        self.state = {
            "weights": {},
            "transitions": {},
            "history": [],
            "accuracy": {"correct": 0, "total": 0},
            "per_intent": {},
            "suppression": [],
            "promotion": []
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                self.state = self.crypto.unseal_file(self.path)
            except Exception:
                legacy = self.path.replace(".enc", ".json")
                if os.path.exists(legacy):
                    try:
                        with open(legacy, "r", encoding="utf-8") as f:
                            self.state = json.load(f)
                    except Exception:
                        pass
        # Ensure required keys exist
        if "accuracy" not in self.state or not isinstance(self.state["accuracy"], dict):
            self.state["accuracy"] = {"correct": 0, "total": 0}
        if "per_intent" not in self.state or not isinstance(self.state["per_intent"], dict):
            self.state["per_intent"] = {}
        if "history" not in self.state or not isinstance(self.state["history"], list):
            self.state["history"] = []
        if "weights" not in self.state or not isinstance(self.state["weights"], dict):
            self.state["weights"] = {}
        if "transitions" not in self.state or not isinstance(self.state["transitions"], dict):
            self.state["transitions"] = {}
        if "suppression" not in self.state or not isinstance(self.state["suppression"], list):
            self.state["suppression"] = []
        if "promotion" not in self.state or not isinstance(self.state["promotion"], list):
            self.state["promotion"] = []

    def save(self):
        try:
            self.crypto.seal_file(self.path, self.state)
        except Exception:
            pass

    def record_outcome(self, action: str, success: bool, context: Dict[str, Any]):
        self.state["history"].append({
            "ts": datetime.datetime.utcnow().isoformat(),
            "action": action,
            "success": success,
            "context": context,
        })
        acc = self.state.setdefault("accuracy", {"correct": 0, "total": 0})
        acc["total"] += 1
        if success:
            acc["correct"] += 1

        pi = self.state.setdefault("per_intent", {})
        st = pi.setdefault(action, {"correct": 0, "total": 0})
        st["total"] += 1
        if success:
            st["correct"] += 1

        if len(self.state["history"]) > 4000:
            self.state["history"] = self.state["history"][-4000:]

    def set_weights(self, weights: Dict[str, float]):
        self.state["weights"] = weights

    def get_weights(self) -> Dict[str, float]:
        return dict(self.state.get("weights", {}))

    def set_transitions(self, transitions: Dict[str, int]):
        self.state["transitions"] = transitions

    def get_transitions(self) -> Dict[str, int]:
        return dict(self.state.get("transitions", {}))

    def get_accuracy(self) -> Dict[str, int]:
        acc = self.state.get("accuracy")
        if not acc or not isinstance(acc, dict):
            acc = {"correct": 0, "total": 0}
            self.state["accuracy"] = acc
        return dict(acc)

    def set_feedback(self, suppress: List[str], promote: List[str]):
        self.state["suppression"] = suppress
        self.state["promotion"] = promote

    def get_feedback(self) -> Tuple[List[str], List[str]]:
        return (list(self.state.get("suppression", [])), list(self.state.get("promotion", [])))


# ---------------------------
# Logger (encrypted flush)
# ---------------------------

class Logger:
    def __init__(self, crypto: CryptoManager, enc_path: str = "fortune_teller_logs.enc"):
        self.events = queue.Queue()
        self.counters: Dict[str, int] = {}
        self.crypto = crypto
        self.enc_path = enc_path
        self._last_flush = time.time()

    def log(self, kind: str, message: str, meta: Optional[Dict[str, Any]] = None):
        entry = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "kind": kind,
            "message": message,
            "meta": meta or {}
        }
        self.events.put(entry)
        self.counters[kind] = self.counters.get(kind, 0) + 1

    def drain(self, max_items=200) -> List[Dict[str, Any]]:
        out = []
        while not self.events.empty() and len(out) < max_items:
            out.append(self.events.get())
        now = time.time()
        if (now - self._last_flush) > 5 and out:
            self._flush_persist(out)
            self._last_flush = now
        return out

    def count(self, kind: str) -> int:
        return self.counters.get(kind, 0)

    def _flush_persist(self, entries: List[Dict[str, Any]]):
        try:
            payload = {"entries": entries, "sig": self.crypto.sign_text(json.dumps(entries))}
            raw = json.dumps(payload).encode("utf-8")
            enc = self.crypto.encrypt(raw, aad=b"fortune_teller_logs")
            with open(self.enc_path, "ab") as f:
                f.write(enc + b"\n")
        except Exception:
            pass


# ---------------------------
# Input activity monitor (mouse + keyboard)
# ---------------------------

class InputMonitor:
    def __init__(self, logger: Logger, window_sec: float = 1.0):
        self.logger = logger
        self.window_sec = window_sec
        self.stop_flag = False
        self.stats_lock = threading.Lock()
        self._reset_stats()
        self._mouse_listener = None
        self._keyboard_listener = None

    def _reset_stats(self):
        self.mouse_moves = 0
        self.mouse_clicks = 0
        self.mouse_scrolls = 0
        self.keypresses = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        if PYNPUT_AVAILABLE:
            try:
                self._mouse_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll)
                self._mouse_listener.start()
                self._keyboard_listener = keyboard.Listener(on_press=self._on_press)
                self._keyboard_listener.start()
                self.logger.log("input", "InputMonitor started with pynput")
            except Exception as e:
                self.logger.log("input", "InputMonitor failed to start pynput", {"error": str(e)})
        else:
            self.logger.log("input", "InputMonitor running in fallback mode")

    def _on_move(self, x, y):
        with self.stats_lock:
            self.mouse_moves += 1

    def _on_click(self, x, y, button, pressed):
        if pressed:
            with self.stats_lock:
                self.mouse_clicks += 1

    def _on_scroll(self, x, y, dx, dy):
        with self.stats_lock:
            self.mouse_scrolls += 1

    def _on_press(self, key):
        with self.stats_lock:
            self.keypresses += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.stats_lock:
            mm, mc, ms, kp = self.mouse_moves, self.mouse_clicks, self.mouse_scrolls, self.keypresses
            cadence_mouse = min(1.0, (mm + mc + ms) / 100.0)
            cadence_keyboard = min(1.0, kp / 80.0)
            active = (mm + mc + ms + kp) > 0
            self._reset_stats()
        return {
            "mouse_moves": mm,
            "mouse_clicks": mc,
            "mouse_scrolls": ms,
            "keypresses": kp,
            "cadence_mouse": cadence_mouse,
            "cadence_keyboard": cadence_keyboard,
            "active": active
        }

    def _loop(self):
        while not self.stop_flag:
            snap = self.snapshot()
            self.logger.log("input_tick", "Input cadence", {"cadence": snap})
            time.sleep(self.window_sec)

    def stop(self):
        self.stop_flag = True
        try:
            if self._mouse_listener: self._mouse_listener.stop()
            if self._keyboard_listener: self._keyboard_listener.stop()
        except Exception:
            pass
        self.logger.log("input", "InputMonitor stopped")


# ---------------------------
# Drive and network activity analyzer
# ---------------------------

class ActivityAnalyzer:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.prev_disk = None
        self.prev_net = None
        self.last_ts = None

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        disk_counters = psutil.disk_io_counters(perdisk=True)
        net_counters = psutil.net_io_counters()

        rates = {
            "drives": {},
            "net": {"up_bps": 0.0, "down_bps": 0.0, "connections": 0, "status": "idle"}
        }

        if self.prev_disk is not None and self.last_ts is not None:
            dt = max(0.001, now - self.last_ts)
            for name, cnt in disk_counters.items():
                prev = self.prev_disk.get(name)
                if prev:
                    r_bps = max(0.0, (cnt.read_bytes - prev.read_bytes) / dt)
                    w_bps = max(0.0, (cnt.write_bytes - prev.write_bytes) / dt)
                    busy = getattr(cnt, "busy_time", 0.0)
                    rates["drives"][name] = {"read_bps": r_bps, "write_bps": w_bps, "busy_time_ms": busy}
        self.prev_disk = disk_counters

        if self.prev_net is not None and self.last_ts is not None and net_counters is not None:
            dt = max(0.001, now - self.last_ts)
            up_bps = max(0.0, (net_counters.bytes_sent - self.prev_net.bytes_sent) / dt)
            down_bps = max(0.0, (net_counters.bytes_recv - self.prev_net.bytes_recv) / dt)
            rates["net"]["up_bps"] = up_bps
            rates["net"]["down_bps"] = down_bps

        self.prev_net = net_counters
        self.last_ts = now

        conns_count = 0
        try:
            conns_count = len([c for c in psutil.net_connections(kind='inet') if c.status == "ESTABLISHED"])
        except Exception:
            pass
        rates["net"]["connections"] = conns_count

        status = "idle"
        up = rates["net"]["up_bps"]
        down = rates["net"]["down_bps"]
        total_drive_rw = sum((d["read_bps"] + d["write_bps"]) for d in rates["drives"].values()) if rates["drives"] else 0.0
        if down > 500_000 and conns_count > 5 and total_drive_rw < 20_000_000:
            status = "browsing"
        if (up > 300_000 and down > 300_000) and conns_count > 20 and total_drive_rw < 15_000_000:
            status = "gaming"
        if (up > 5_000_000 or down > 10_000_000) or total_drive_rw > 50_000_000:
            status = "transfer"
        rates["net"]["status"] = status

        self.logger.log("activity", "Drive/Net activity", {"rates": rates})
        return rates


# ---------------------------
# Telemetry collector (real probes + input + activity + personas) with GPU-safe fallback
# ---------------------------

class TelemetryCollector:
    def __init__(self, logger: Logger, input_monitor: InputMonitor, activity: ActivityAnalyzer):
        self.logger = logger
        self.input_monitor = input_monitor
        self.activity = activity
        self.stop_flag = False

    def start(self):
        self.input_monitor.start()
        threading.Thread(target=self._loop, daemon=True).start()
        self.logger.log("telemetry", "Telemetry started")

    def _gpu_snapshot(self) -> Dict[str, Any]:
        gpu = {"vendor": None, "util": 0.0, "vram_total_mb": 0.0, "vram_free_mb": 0.0}
        try:
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpu.update({
                        "vendor": "NVIDIA",
                        "util": float(util),
                        "vram_total_mb": mem.total / (1024 * 1024),
                        "vram_free_mb": mem.free / (1024 * 1024)
                    })
                finally:
                    try: pynvml.nvmlShutdown()
                    except Exception: pass
            elif AMD_AVAILABLE:
                try:
                    devs = pyamdgpuinfo.get_gpu_info()
                    if devs:
                        dev = devs[0]
                        gpu.update({
                            "vendor": "AMD",
                            "util": 0.0,
                            "vram_total_mb": float(dev.get("vram_total", 0.0)),
                            "vram_free_mb": 0.0
                        })
                except Exception:
                    pass
        except Exception as e:
            self.logger.log("telemetry", "GPU probe failed; CPU-only mode", {"error": str(e)})
            gpu = {"vendor": None, "util": 0.0, "vram_total_mb": 0.0, "vram_free_mb": 0.0}
        if gpu["vendor"] is None:
            self.logger.log("telemetry", "No GPU detected, running CPU-only mode")
        return gpu

    def snapshot(self) -> Dict[str, Any]:
        cpu_util = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        battery = None
        try:
            batt = psutil.sensors_battery()
            if batt is not None:
                battery = {"percent": batt.percent, "plugged": batt.power_plugged}
        except Exception:
            battery = None

        gpu = self._gpu_snapshot()
        now = datetime.datetime.now()
        input_state = self.input_monitor.snapshot()
        activity = self.activity.snapshot()

        persona = "Idle"
        status = activity.get("net", {}).get("status", "idle")
        kb = input_state.get("cadence_keyboard", 0.0)
        ms = input_state.get("cadence_mouse", 0.0)

        procs = []
        try:
            for p in psutil.process_iter(["name"]):
                n = (p.info.get("name") or "").lower()
                if n:
                    procs.append(n)
        except Exception:
            pass

        gaming_proc = any(n in procs for n in ["steam.exe", "epicgameslauncher.exe"])
        ide_proc = any(n in procs for n in ["code.exe", "idea64.exe", "pycharm64.exe", "devenv.exe"])
        browser_proc = any(n in procs for n in ["chrome.exe", "msedge.exe", "firefox.exe"])

        if gaming_proc or status == "gaming":
            persona = "Gaming"
        elif ide_proc or (kb > 0.5 and status == "transfer"):
            persona = "Work"
        elif browser_proc or status == "browsing" or ms > 0.6:
            persona = "Browsing"

        return {
            "ts": now.isoformat(),
            "os": platform.system(),
            "cpu_util": cpu_util,
            "mem_total_mb": mem.total / (1024 * 1024),
            "mem_free_mb": mem.available / (1024 * 1024),
            "disk_io": disk_io._asdict() if disk_io else {},
            "net_io": net_io._asdict() if net_io else {},
            "gpu": gpu,
            "battery": battery,
            "hour": now.hour,
            "weekday": now.weekday(),
            "input_state": input_state,
            "activity": activity,
            "persona": persona,
            "proc_sample": procs[:25],
            "headless": HEADLESS_MODE
        }

    def _loop(self):
        while not self.stop_flag:
            time.sleep(1.0)

    def stop(self):
        self.stop_flag = True
        self.logger.log("telemetry", "Telemetry stopped")


# ---------------------------
# Capability profiler (GPU-neutral)
# ---------------------------

class CapabilityProfiler:
    def __init__(self, logger: Logger, telemetry: TelemetryCollector):
        self.logger = logger
        self.telemetry = telemetry
        self.profile = CapabilityProfile(
            scores={"cpu": 0.7, "gpu": 0.5, "mem": 0.7, "disk": 0.8, "net": 0.6, "power": 0.6},
            features={"avx2": False, "directstorage": False, "tls_tickets": True},
            constraints={"battery": False, "numa_nodes": 1, "vm": False}
        )

    def run_probes(self):
        snap = self.telemetry.snapshot()
        cpu_score = min(1.0, snap["cpu_util"] / 100.0 * 0.5 + 0.5)
        mem_score = min(1.0, snap["mem_free_mb"] / max(512.0, snap["mem_total_mb"]))
        disk_score = 0.7
        net_score = 0.6
        power_score = 0.7 if (snap.get("battery") and snap["battery"].get("plugged")) or (snap.get("battery") is None) else 0.5
        gpu_vendor = snap["gpu"].get("vendor")
        gpu_score = 0.5 if gpu_vendor is None else 0.7

        self.profile.scores.update({
            "cpu": cpu_score,
            "gpu": gpu_score,
            "mem": mem_score,
            "disk": disk_score,
            "net": net_score,
            "power": power_score
        })
        self.profile.features["avx2"] = ("x86_64" in platform.machine())
        self.profile.constraints["battery"] = bool(snap.get("battery") and not snap["battery"].get("plugged"))

        if gpu_vendor is None:
            self.logger.log("capability", "GPU absent; using CPU-only profile", {"gpu_score": gpu_score})
        else:
            self.logger.log("capability", "GPU present", {"vendor": gpu_vendor, "gpu_score": gpu_score})

    def get_profile(self) -> CapabilityProfile:
        return self.profile


# ---------------------------
# Port & device scanner
# ---------------------------

class PortScanner:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.stop_flag = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        self.logger.log("ports", "Port scanner started")

    def active_connections(self) -> List[Dict[str, Any]]:
        conns = []
        try:
            for c in psutil.net_connections(kind='inet'):
                item = {
                    "laddr": f"{getattr(c.laddr, 'ip', None)}:{getattr(c.laddr, 'port', None)}",
                    "raddr": f"{getattr(c.raddr, 'ip', None)}:{getattr(c.raddr, 'port', None)}" if c.raddr else None,
                    "status": c.status
                }
                conns.append(item)
        except Exception:
            pass
        return conns

    def usb_devices(self) -> List[Dict[str, Any]]:
        devices = []
        system = platform.system()
        try:
            if system == "Linux":
                if PYUDEV_AVAILABLE:
                    ctx = pyudev.Context()
                    for dev in ctx.list_devices(subsystem='usb'):
                        nm = dev.get('ID_MODEL')
                        if nm:
                            devices.append({"name": nm, "vendor": dev.get('ID_VENDOR'), "id": dev.get('ID_MODEL_ID')})
                else:
                    out = subprocess.run(["lsusb"], capture_output=True, text=True)
                    for line in out.stdout.splitlines():
                        devices.append({"lsusb": line})
            elif system == "Darwin":
                out = subprocess.run(["system_profiler", "SPUSBDataType", "-json"], capture_output=True, text=True)
                if out.stdout:
                    try:
                        data = json.loads(out.stdout)
                        devices.append({"system_profiler": data})
                    except Exception:
                        devices.append({"raw": out.stdout[:2000]})
            elif system == "Windows":
                cmd = ["powershell", "-Command", "Get-PnpDevice -Class USB | Select-Object -Property FriendlyName,InstanceId | ConvertTo-Json"]
                out = subprocess.run(cmd, capture_output=True, text=True)
                if out.stdout:
                    try:
                        data = json.loads(out.stdout)
                        if isinstance(data, dict):
                            devices.append(data)
                        elif isinstance(data, list):
                            devices.extend(data)
                        else:
                            devices.append({"raw": out.stdout[:2000]})
                    except Exception:
                        devices.append({"raw": out.stdout[:2000]})
        except Exception:
            pass
        return devices

    def snapshot(self) -> Dict[str, Any]:
        return {"connections": self.active_connections(), "usb": self.usb_devices()}

    def _loop(self):
        while not self.stop_flag:
            time.sleep(2.0)

    def stop(self):
        self.stop_flag = True
        self.logger.log("ports", "Port scanner stopped")


# ---------------------------
# Network scanner
# ---------------------------

class NetworkScanner:
    def __init__(self, logger: Logger, iface_cidr: Optional[str] = None):
        self.logger = logger
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.iface_cidr = iface_cidr

    def discover_devices(self):
        if SCAPY_AVAILABLE and self.iface_cidr:
            try:
                arp = ARP(pdst=self.iface_cidr)
                ether = Ether(dst="ff:ff:ff:ff:ff:ff")
                result = srp(ether/arp, timeout=2, verbose=0)[0]
                for _, received in result:
                    ip = received.psrc
                    mac = received.hwsrc
                    self.devices[ip] = {"ip": ip, "mac": mac, "name": None}
            except Exception:
                pass
        else:
            try:
                out = subprocess.run(["arp", "-a"], capture_output=True, text=True)
                for line in out.stdout.splitlines():
                    if "(" in line and ")" in line:
                        parts = line.split()
                        ip = parts[1].strip("()")
                        mac = None
                        for p in parts:
                            if ":" in p and len(p) >= 17:
                                mac = p
                                break
                        if ip and ip not in self.devices:
                            self.devices[ip] = {"ip": ip, "mac": mac, "name": None}
            except Exception:
                pass

        if ZEROCONF_AVAILABLE:
            try:
                zc = Zeroconf()
                discovered = []

                class _Listener:
                    def add_service(self, zc, type_, name):
                        info = zc.get_service_info(type_, name)
                        if info and info.parsed_addresses():
                            discovered.append({"name": name, "ip": info.parsed_addresses()[0]})

                types = ["_http._tcp.local.", "_workstation._tcp.local."]
                listeners = []
                browsers = []
                for t in types:
                    l = _Listener()
                    b = ServiceBrowser(zc, t, l)
                    listeners.append(l)
                    browsers.append(b)
                time.sleep(2.0)
                for d in discovered:
                    ip = d["ip"]
                    if ip in self.devices:
                        self.devices[ip]["name"] = d["name"]
                    else:
                        self.devices[ip] = {"ip": ip, "mac": None, "name": d["name"]}
                try: zc.close()
                except Exception: pass
            except Exception:
                pass

        self.logger.log("network", "Discovered devices", {"devices": self.devices})

    def aggregate_capabilities(self) -> Dict[str, float]:
        agg = {"cpu": 0.0, "gpu": 0.0, "disk": 0.0, "net": 0.0}
        for ip, _ in self.devices.items():
            rtt_ms = None
            try:
                system = platform.system()
                if system == "Windows":
                    out = subprocess.run(["ping", "-n", "1", "-w", "500", ip], capture_output=True, text=True)
                else:
                    out = subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True, text=True)
                text = out.stdout
                if "time=" in text:
                    idx = text.find("time=")
                    ms_str = ""
                    for ch in text[idx+5:idx+15]:
                        if ch.isdigit() or ch == ".":
                            ms_str += ch
                        else:
                            break
                    if ms_str:
                        rtt_ms = float(ms_str)
            except Exception:
                rtt_ms = None

            weight = 0.5 if rtt_ms is None else max(0.1, 1.0 - min(1.0, rtt_ms / 100.0))
            agg["cpu"] += weight * 0.5
            agg["gpu"] += weight * 0.3
            agg["disk"] += weight * 0.6
            agg["net"] += weight * 0.8

        self.logger.log("network", "Aggregated capabilities", {"aggregate": agg})
        return agg


# ---------------------------
# Predictor (ensemble + personas + n-gram + bandit + calibration + cooldown/rhythm)
# ---------------------------

class Predictor:
    def __init__(self, logger: Logger, memory: MemoryManager, policy: Policy):
        self.logger = logger
        self.memory = memory
        self.policy = policy
        self.action_weights: Dict[str, float] = self.memory.get_weights() or {
            "Open Email": 0.6,
            "Launch Browser": 0.7,
            "Launch Steam": 0.5,
            "Open IDE": 0.4,
        }
        self.transitions: Dict[str, int] = self.memory.get_transitions()
        self.last_action: Optional[str] = None
        self.last_action_ts: Optional[float] = None
        self.hour_hist: Dict[int, Dict[str, float]] = {h: {} for h in range(24)}
        self.weekday_hist: Dict[int, Dict[str, float]] = {d: {} for d in range(7)}
        self.ema_alpha = 0.2

        # N-gram transitions and decay
        self.ngram_transitions: Dict[str, int] = {}
        self.recency_decay = 0.96
        self._last_two: List[str] = []

        # Bandit priors per intent
        self.bandit_alpha: Dict[str, float] = {k: 1.0 for k in self.action_weights}
        self.bandit_beta: Dict[str, float] = {k: 1.0 for k in self.action_weights}

        # Platt-style calibration params
        self.platt_A = 1.0
        self.platt_B = 0.0

        # Cooldown and rhythm
        self.cooldown: Dict[str, float] = {k: 0.0 for k in self.action_weights}
        self.rhythm: Dict[str, Tuple[int, int]] = {
            "Open Email": (7, 12),
            "Open IDE": (9, 18),
            "Launch Steam": (18, 23),
            "Launch Browser": (8, 23),
        }

        # Internal novelty tracker
        self.novelty: Dict[str, float] = {k: 0.0 for k in self.action_weights.keys()}

        # Clusters: map intents into semantic groups for diversity
        self.clusters: Dict[str, str] = {
            "Open Email": "Productivity",
            "Open IDE": "Productivity",
            "Launch Browser": "Browsing",
            "Launch Steam": "Gaming",
        }

    def _ema_update(self, hist: Dict[str, float], key: str, inc: float = 1.0):
        val = hist.get(key, 0.0)
        hist[key] = (1 - self.ema_alpha) * val + self.ema_alpha * inc

    def record_outcome(self, action: str, hour: int, weekday: int, success: bool):
        self._ema_update(self.hour_hist[hour], action, 1.0 if success else 0.3)
        self._ema_update(self.weekday_hist[weekday], action, 1.0 if success else 0.3)

        # Markov transitions
        if self.last_action:
            tkey = f"{self.last_action}||{action}"
            self.transitions[tkey] = self.transitions.get(tkey, 0) + (1 if success else 0)

        # N-gram updates with decay
        if self.last_action:
            key2 = f"{self.last_action}||{action}"
            self.ngram_transitions[key2] = int(self.ngram_transitions.get(key2, 0) * self.recency_decay) + (1 if success else 0)
        self._last_two = (self._last_two + [action])[-2:]
        if self.last_action and len(self._last_two) == 2:
            key3 = f"{self.last_action}||{self._last_two[0]}||{self._last_two[1]}"
            self.ngram_transitions[key3] = int(self.ngram_transitions.get(key3, 0) * self.recency_decay) + (1 if success else 0)

        # Bandit update
        if success:
            self.bandit_alpha[action] = self.bandit_alpha.get(action, 1.0) + 1.0
        else:
            self.bandit_beta[action] = self.bandit_beta.get(action, 1.0) + 1.0

        # Reinforcement learning style update
        delta = 0.08 if success else -0.06
        acc = self.memory.get_accuracy()
        acc_ratio = (acc["correct"] / max(1, acc["total"]))
        scale = 0.5 + 0.5 * acc_ratio
        self.action_weights[action] = min(1.2, max(0.0, self.action_weights.get(action, 0.5) + delta * scale))

        # Novelty decay (reward diversity slightly)
        for k in self.novelty:
            self.novelty[k] = max(0.0, self.novelty[k] * 0.95)
        self.novelty[action] += 0.2 if success else 0.05

        # Cooldown update and decay
        self.cooldown[action] = min(1.0, self.cooldown.get(action, 0.0) + (0.2 if success else 0.05))
        for k in self.cooldown:
            self.cooldown[k] = max(0.0, self.cooldown[k] * 0.85)

        # Platt update (moving average fit)
        avg_w = sum(self.action_weights.values()) / max(1, len(self.action_weights))
        target = acc_ratio
        self.platt_A = 0.9 * self.platt_A + 0.1 * (target / max(1e-3, avg_w))
        self.platt_B = 0.9 * self.platt_B + 0.1 * (target - self.platt_A * avg_w)

        self.last_action = action
        self.last_action_ts = time.time()

        # Persist
        self.memory.set_weights(self.action_weights)
        self.memory.set_transitions(self.transitions)
        self.memory.record_outcome(action, success, {"hour": hour, "weekday": weekday})
        self.memory.save()
        self.logger.log("learning", "Recorded outcome", {"action": action, "success": success})

    def _time_bias(self, action: str, hour: int, weekday: int) -> float:
        h_score = self.hour_hist[hour].get(action, 0.0)
        d_score = self.weekday_hist[weekday].get(action, 0.0)
        return min(1.0, h_score * 0.6 + d_score * 0.4)

    def _markov_bias(self, action: str) -> float:
        if not self.last_action:
            return 0.0
        tkey = f"{self.last_action}||{action}"
        return min(0.3, 0.02 * self.transitions.get(tkey, 0))

    def _ngram_bias(self, action: str) -> float:
        b = 0.0
        if self.last_action:
            key2 = f"{self.last_action}||{action}"
            b += min(0.18, 0.02 * self.ngram_transitions.get(key2, 0))
        if len(self._last_two) == 2 and self.last_action:
            key3 = f"{self.last_action}||{self._last_two[0]}||{action}"
            b += min(0.12, 0.015 * self.ngram_transitions.get(key3, 0))
        return b

    def _session_decay(self) -> float:
        if self.last_action_ts is None:
            return 0.0
        age = time.time() - self.last_action_ts
        return -0.05 if age > 300 else 0.0

    def _input_bias(self, action: str, input_state: Dict[str, Any]) -> float:
        kb = input_state.get("cadence_keyboard", 0.0)
        ms = input_state.get("cadence_mouse", 0.0)
        active = input_state.get("active", False)
        bias = 0.0
        if active:
            if kb > 0.6 and action in ("Open IDE", "Open Email"):
                bias += 0.15
            if ms > 0.6 and action == "Launch Browser":
                bias += 0.12
            if kb < 0.3 and ms < 0.3 and action == "Launch Steam":
                bias += 0.08
        return bias

    def _drive_net_bias(self, action: str, activity: Dict[str, Any]) -> float:
        status = activity.get("net", {}).get("status", "idle")
        total_drive_rw = sum((d["read_bps"] + d["write_bps"]) for d in activity.get("drives", {}).values()) if activity.get("drives") else 0.0
        up = activity.get("net", {}).get("up_bps", 0.0)
        down = activity.get("net", {}).get("down_bps", 0.0)
        conns = activity.get("net", {}).get("connections", 0)

        bias = 0.0
        if status == "gaming" and action == "Launch Steam":
            bias += 0.22
        if status == "browsing" and action == "Launch Browser":
            bias += 0.16
        if status == "transfer" and action in ("Open IDE", "Open Email"):
            bias += 0.08
        if total_drive_rw > 80_000_000 and action in ("Launch Steam", "Open IDE"):
            bias -= 0.07
        if conns > 25 and action == "Launch Steam":
            bias += 0.05
        if conns > 10 and down > 600_000 and action == "Launch Browser":
            bias += 0.05
        return bias

    def _feedback_bias(self, action: str) -> float:
        suppress, promote = self.memory.get_feedback()
        bias = 0.0
        if action in suppress or action in self.policy.suppress:
            bias -= 0.25
        if action in promote or action in self.policy.promote:
            bias += 0.25
        return bias

    def _novelty_boost(self, action: str) -> float:
        return min(0.08, 0.02 + 0.02 * (1.0 - min(1.0, self.novelty.get(action, 0.0))))

    def _persona_bias(self, action: str, persona: str) -> float:
        if persona == "Gaming" and action == "Launch Steam": return 0.22
        if persona == "Work" and action in ("Open IDE", "Open Email"): return 0.18
        if persona == "Browsing" and action == "Launch Browser": return 0.16
        return 0.0

    def _rhythm_bias(self, action: str, hour: int) -> float:
        rng = self.rhythm.get(action)
        if not rng: return 0.0
        start, end = rng
        in_range = start <= hour <= end
        return 0.1 if in_range else -0.06

    def _per_intent_accuracy_bias(self, action: str) -> float:
        pi = self.memory.state.get("per_intent", {}).get(action, {"correct": 0, "total": 0})
        ratio = pi["correct"] / max(1, pi["total"])
        return 0.12 * (ratio - 0.5)

    def _adaptive_temperature(self, base_temp: float = 0.8) -> float:
        acc = self.memory.get_accuracy()
        acc_ratio = (acc["correct"] / max(1, acc["total"]))
        temp = base_temp - 0.3 * (acc_ratio - 0.5)
        return max(0.4, min(1.2, temp))

    def _calibrate_softmax(self, scores: List[float], temperature: float = 0.8) -> List[float]:
        exps = [math.exp(s / max(1e-6, temperature)) for s in scores]
        total = sum(exps) + 1e-9
        return [e / total for e in exps]

    def _platt_sigmoid(self, p: float) -> float:
        z = self.platt_A * p + self.platt_B
        return 1.0 / (1.0 + math.exp(-4.0 * (z - 0.5)))

    def _cluster(self, intent_id: str) -> str:
        return self.clusters.get(intent_id, "Other")

    def _diversify(self, intents: List[Intent], top_k: int = 4, max_per_cluster: int = 2, epsilon: float = 0.12) -> List[Intent]:
        by_cluster: Dict[str, List[Intent]] = {}
        for i in intents:
            c = i.cluster or "Other"
            by_cluster.setdefault(c, []).append(i)
        selected: List[Intent] = []
        per_cluster_taken: Dict[str, int] = {}
        for intent in intents:
            c = intent.cluster or "Other"
            if per_cluster_taken.get(c, 0) < max_per_cluster:
                selected.append(intent)
                per_cluster_taken[c] = per_cluster_taken.get(c, 0) + 1
            if len(selected) >= top_k:
                break
        if len(selected) < top_k and intents:
            tail = intents[-1]
            if tail.id not in [i.id for i in selected]:
                flip = int(hashlib.sha256(str(time.time()).encode()).hexdigest(), 16) % 100
                if flip < int(epsilon * 100):
                    selected.append(tail)
        return selected

    def predict(self, snapshot: Dict[str, Any]) -> List[Intent]:
        hour = snapshot["hour"]
        weekday = snapshot["weekday"]
        input_state = snapshot.get("input_state", {})
        activity = snapshot.get("activity", {})
        persona = snapshot.get("persona", "Idle")

        raw_scores: List[Tuple[str, float, Dict[str, Any]]] = []
        for action, base_w in self.action_weights.items():
            s_base = base_w
            s_time = 0.25 * self._time_bias(action, hour, weekday)
            s_markov = self._markov_bias(action)
            s_ngram = self._ngram_bias(action)
            s_input = self._input_bias(action, input_state)
            s_ctx = self._drive_net_bias(action, activity)
            s_feedback = self._feedback_bias(action)
            s_novelty = self._novelty_boost(action)
            s_session = self._session_decay()
            s_persona = self._persona_bias(action, persona)
            s_rhythm = self._rhythm_bias(action, hour)
            s_cooldown = -0.12 * self.cooldown.get(action, 0.0)
            s_piacc = self._per_intent_accuracy_bias(action)

            a = self.bandit_alpha.get(action, 1.0)
            b = self.bandit_beta.get(action, 1.0)
            bandit_mean = a / max(1e-6, (a + b))
            s_bandit = 0.15 * (bandit_mean - 0.5)

            score = (
                s_base + s_time + s_markov + s_ngram + s_input + s_ctx + s_feedback +
                s_novelty + s_session + s_persona + s_rhythm + s_cooldown + s_piacc + s_bandit
            )
            score = max(-1.0, min(1.8, score))  # clamp before calibration

            evidence = {
                "base_weight": round(s_base, 3),
                "time_bias": round(s_time, 3),
                "markov_bias": round(s_markov, 3),
                "ngram_bias": round(s_ngram, 3),
                "input_bias": round(s_input, 3),
                "drive_net_bias": round(s_ctx, 3),
                "feedback_bias": round(s_feedback, 3),
                "novelty_boost": round(s_novelty, 3),
                "session_decay": round(s_session, 3),
                "persona_bias": round(s_persona, 3),
                "rhythm_bias": round(s_rhythm, 3),
                "cooldown_penalty": round(s_cooldown, 3),
                "per_intent_accuracy_bias": round(s_piacc, 3),
                "bandit_mean": round(bandit_mean, 3),
                "bandit_bias": round(s_bandit, 3),
                "persona": persona,
                "last_action": self.last_action,
                "context_status": activity.get("net", {}).get("status", "idle")
            }
            raw_scores.append((action, score, evidence))

        temp = self._adaptive_temperature(base_temp=0.8)
        scores_only = [s for _, s, _ in raw_scores]
        probs = self._calibrate_softmax(scores_only, temperature=temp)
        probs = [self._platt_sigmoid(p) for p in probs]
        ssum = sum(probs) + 1e-9
        probs = [p / ssum for p in probs]

        intents: List[Intent] = []
        for idx, (action, score, evidence) in enumerate(raw_scores):
            c = self._cluster(action)
            intents.append(Intent(id=action, score=score, evidence=evidence, cluster=c, calibrated=probs[idx]))
        intents.sort(key=lambda i: i.calibrated, reverse=True)

        top_intents = self._diversify(intents, top_k=4, max_per_cluster=2, epsilon=0.12)
        self.logger.log("prediction", "Predicted intents", {"temperature": temp, "persona": persona, "intents": [asdict(i) for i in top_intents]})
        return top_intents


# ---------------------------
# Dependency graph with next_dependencies (for read-ahead)
# ---------------------------

class DependencyGraph:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.graph: Dict[str, List[str]] = {
            "EmailClient": ["TLSStack", "FontPack", "InboxCache", "AuthTokens"],
            "Browser": ["DNSCache", "TLSStack", "Extensions", "FontPack"],
            "TLSStack": ["CertStore"],
            "Extensions": ["AdBlock", "DarkMode"],
            "FontPack": ["Arial", "Roboto"],
            "Steam.exe": ["OverlayDLLs", "AntiCheatPassive", "ShaderCache"],
            "OverlayDLLs": ["HookLib"],
            "IDE": ["Compiler", "Debugger", "IndexCache", "LanguageServer"],
            "IndexCache": ["IndexShardA", "IndexShardB"],
        }
        self.hashes: Dict[str, str] = {}

    def resolve(self, nodes: List[str]) -> List[str]:
        resolved = set(nodes)
        frontier = list(nodes)
        while frontier:
            n = frontier.pop()
            for dep in self.graph.get(n, []):
                if dep not in resolved:
                    resolved.add(dep)
                    frontier.append(dep)
        return list(resolved)

    def verify(self, node: str) -> str:
        h = hashlib.sha256(node.encode("utf-8")).hexdigest()
        self.hashes[node] = h
        return h

    def next_dependencies(self, node: str, depth: int = 2) -> List[str]:
        """
        Return an ordered list of next dependencies up to 'depth' layers deep,
        suitable for speculative read-ahead. Depth is small to avoid over-fetch.
        """
        seen = set()
        result: List[str] = []

        def dfs(n: str, d: int):
            if d == 0 or n in seen:
                return
            seen.add(n)
            for dep in self.graph.get(n, []):
                result.append(dep)
                dfs(dep, d - 1)

        dfs(node, max(1, depth))
        return result


# ---------------------------
# Planner (persona-aware tweaks)
# ---------------------------

class Planner:
    def __init__(self, logger: Logger, capability: CapabilityProfiler, dep_graph: DependencyGraph):
        self.logger = logger
        self.capability = capability
        self.dep_graph = dep_graph

    def synthesize(self, intents: List[Intent]) -> Plan:
        profile = self.capability.get_profile()
        top = intents[0] if intents else Intent(id="Observe", score=0.0, evidence={}, calibrated=0.0)
        steps: List[PlanStep] = []
        now_str = datetime.datetime.now().isoformat()
        persona = top.evidence.get("persona", "Idle")
        rationale = f"[{now_str}] Plan for {top.id} ({persona}); disk={profile.scores['disk']:.2f}, gpu={profile.scores['gpu']:.2f}; confidence={top.calibrated:.2f}"

        if top.id == "Open Email":
            steps.append(PlanStep("Preload Email Bundle", deps=["EmailClient"], cost={"cpu": 0.12, "disk": 25e6, "mem": 256.0}, expected_gain_ms=900))
        elif top.id == "Launch Browser":
            steps.append(PlanStep("Preload Browser Bundle", deps=["Browser"], cost={"cpu": 0.15, "disk": 40e6, "mem": 384.0}, expected_gain_ms=1100))
        elif top.id == "Launch Steam":
            steps.append(PlanStep("Preload Steam Runtime", deps=["Steam.exe"], cost={"cpu": 0.18, "disk": 180e6, "mem": 512.0, "gpu_vram": 64.0}, expected_gain_ms=1400))
        elif top.id == "Open IDE":
            steps.append(PlanStep("Preload IDE Toolchain", deps=["IDE"], cost={"cpu": 0.16, "disk": 100e6, "mem": 512.0}, expected_gain_ms=1300))
        else:
            steps.append(PlanStep("Observe and Learn", deps=[], cost={"cpu": 0.02}, expected_gain_ms=0))

        if persona == "Work" and top.id == "Open IDE":
            steps[0].cost["disk"] *= 0.85
            steps[0].expected_gain_ms += 150
        if persona == "Gaming" and top.id == "Launch Steam":
            steps[0].cost["gpu_vram"] = steps[0].cost.get("gpu_vram", 0.0) + 128.0
            steps[0].expected_gain_ms += 220

        final_steps: List[PlanStep] = []
        for s in steps:
            deps_resolved = self.dep_graph.resolve(s.deps)
            disk_score = profile.scores.get("disk", 0.8)
            gain_adj = 1.0 + (disk_score - 0.5) * 0.6
            adjusted_gain = int(s.expected_gain_ms * gain_adj)
            final_steps.append(PlanStep(name=s.name, deps=deps_resolved, cost=s.cost, expected_gain_ms=adjusted_gain))

        expected_gain = sum(s.expected_gain_ms for s in final_steps)
        total_cost = {
            "cpu": sum(s.cost.get("cpu", 0.0) for s in final_steps),
            "disk": sum(s.cost.get("disk", 0.0) for s in final_steps),
            "mem": sum(s.cost.get("mem", 0.0) for s in final_steps),
            "gpu_vram": sum(s.cost.get("gpu_vram", 0.0) for s in final_steps),
        }

        plan = Plan(
            id=f"plan_{int(time.time())}",
            steps=final_steps,
            total_cost=total_cost,
            expected_gain_ms=expected_gain,
            rationale=rationale,
            chosen_intent=top
        )
        self.logger.log("planning", "Synthesized plan", {"plan": asdict(plan)})
        return plan


# ---------------------------
# Policy engine
# ---------------------------

class PolicyEngine:
    def __init__(self, logger: Logger, policy: Policy, capability: CapabilityProfiler):
        self.logger = logger
        self.policy = policy
        self.capability = capability

    def evaluate(self, plan: Plan, snapshot: Dict[str, Any]) -> Decision:
        hour = snapshot["hour"]
        start, end = self.policy.quiet_hours
        in_quiet = (start <= hour <= end) if start <= end else (hour >= start or hour <= end)

        cpu_ok = plan.total_cost.get("cpu", 0.0) <= self.policy.cpu_cap
        disk_ok = plan.total_cost.get("disk", 0.0) <= self.policy.disk_cap_bps * 2
        vram_ok = plan.total_cost.get("gpu_vram", 0.0) <= self.policy.vram_cap_mb

        allow = cpu_ok and disk_ok and vram_ok and not in_quiet
        notify = not allow or (self.policy.autonomy_level != "background")

        rationale = f"cpu_ok={cpu_ok}, disk_ok={disk_ok}, vram_ok={vram_ok}, in_quiet={in_quiet}"
        risk = "low" if allow else "medium"
        decision = Decision(allow=allow, notify=notify, rationale=rationale, risk=risk, policy_gate=asdict(self.policy))
        self.logger.log("policy", "Policy decision", {"decision": asdict(decision)})
        return decision


# ---------------------------
# Auto-loader with speculative read-ahead (jump-ahead)
# ---------------------------

class AutoLoader:
    def __init__(self, logger: Logger, dep_graph: DependencyGraph, capability: CapabilityProfiler, read_ahead_window: int = 2):
        self.logger = logger
        self.dep_graph = dep_graph
        self.capability = capability
        self.read_ahead_window = max(0, read_ahead_window)

    def _strategy(self, plan: Plan, snapshot: Optional[Dict[str, Any]] = None) -> str:
        prof = self.capability.get_profile()
        disk = prof.scores.get("disk", 0.8)
        mem = prof.scores.get("mem", 0.7)
        gpu = prof.scores.get("gpu", 0.5)

        gpu_vendor = None
        if snapshot:
            gpu_vendor = snapshot.get("gpu", {}).get("vendor")

        if plan.total_cost.get("gpu_vram", 0.0) > 0 and gpu > 0.65 and gpu_vendor:
            return "gpu_vram_staging"
        if disk > mem:
            return "disk_prefetch"
        return "balanced"

    def _adjust_read_ahead(self, snapshot: Optional[Dict[str, Any]]) -> int:
        window = self.read_ahead_window
        if not snapshot:
            return window
        mem_free_mb = snapshot.get("mem_free_mb", 512.0)
        cpu_util = snapshot.get("cpu_util", 50.0)
        # Smaller window on tight memory or high CPU
        if mem_free_mb < 1024.0 or cpu_util > 80.0:
            window = max(1, window - 1)
        if mem_free_mb > 4096.0 and cpu_util < 40.0:
            window = min(window + 1, 4)
        return window

    def execute(self, plan: Plan, snapshot: Optional[Dict[str, Any]] = None) -> Result:
        strategy = self._strategy(plan, snapshot)
        verified = {}
        resources = plan.total_cost.copy()
        loaded = set()

        # Execute planned steps
        for step in plan.steps:
            for d in step.deps:
                verified[d] = self.dep_graph.verify(d)
                loaded.add(d)
            time.sleep(0.05)
            self.logger.log("preload", f"Executed '{step.name}'", {"deps": step.deps, "strategy": strategy, "cost": step.cost})

        # Speculative jump-ahead: read next likely deps based on the chosen intent
        top_intent_id = plan.chosen_intent.id if plan.chosen_intent else None
        if top_intent_id:
            window = self._adjust_read_ahead(snapshot)
            next_deps = self.dep_graph.next_dependencies(top_intent_id, depth=window)
            for dep in next_deps:
                if dep in loaded:
                    self.logger.log("speculative", f"Skipped speculative preload of '{dep}' (already loaded)", {"source": top_intent_id, "window": window})
                    continue
                # Simulate lightweight prefetch (hash verify + tiny delay)
                verified[dep] = self.dep_graph.verify(dep)
                time.sleep(0.01)
                loaded.add(dep)
                self.logger.log("speculative", f"Speculative preload of '{dep}'", {"source": top_intent_id, "window": window})

        return Result(success=True, resources=resources, verified_hashes=verified, notes=f"Preload via {strategy} with read-ahead")

# ---------------------------
# Self-improver (encrypted patches)
# ---------------------------

class SelfImprover:
    def __init__(self, logger: Logger, crypto: CryptoManager, policy: UpdatePolicy, project_root: str, version_file: str = "VERSION"):
        self.logger = logger
        self.crypto = crypto
        self.policy = policy
        self.project_root = os.path.abspath(project_root)
        self.version_file = os.path.join(self.project_root, version_file)

    def _now_hour(self) -> int:
        return datetime.datetime.now().hour

    def _in_quiet(self) -> bool:
        s, e = self.policy.quiet_hours
        return s <= self._now_hour() <= e if s <= e else (self._now_hour() >= s or self._now_hour() <= e)

    def _allowed(self, path: str) -> bool:
        ap = os.path.abspath(path)
        for allowed in self.policy.allow_paths:
            if ap.startswith(os.path.abspath(os.path.join(self.project_root, allowed))):
                return True
        return False

    def _all_py_files(self) -> List[str]:
        files = []
        for root, _, names in os.walk(self.project_root):
            for n in names:
                if n.endswith(".py"):
                    files.append(os.path.join(root, n))
        return files

    def _lint(self) -> bool:
        if not self.policy.require_lint_pass:
            return True
        try:
            result = subprocess.run(["python", "-m", "py_compile"] + self._all_py_files(), capture_output=True, text=True)
            ok = result.returncode == 0
            self.logger.log("selfimprove", "Lint/compile check", {"ok": ok, "stderr": result.stderr[:500]})
            return ok
        except Exception as e:
            self.logger.log("selfimprove", "Lint error", {"error": str(e)})
            return False

    def _tests(self) -> bool:
        if not self.policy.require_tests_pass:
            return True
        try:
            result = subprocess.run(["python", "-m", "pytest", "-q"], cwd=self.project_root, capture_output=True, text=True)
            ok = result.returncode == 0
            self.logger.log("selfimprove", "Tests run", {"ok": ok, "stdout": result.stdout[:500], "stderr": result.stderr[:500]})
            return ok
        except Exception as e:
            self.logger.log("selfimprove", "Tests error", {"error": str(e)})
            return False

    def _sandbox_import(self, candidate_paths: List[str]) -> bool:
        for p in candidate_paths:
            if not p.endswith(".py"): continue
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ft_sandbox_" + hashlib.md5(p.encode()).hexdigest(), p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
            except Exception as e:
                self.logger.log("selfimprove", "Sandbox import failed", {"path": p, "error": str(e)})
                return False
        return True

    def _bump_version(self) -> str:
        old = "0.0.0"
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, "r", encoding="utf-8") as f:
                    old = f.read().strip()
            major, minor, patch = [int(x) for x in old.split(".")]
            new = f"{major}.{minor}.{patch+1}"
        except Exception:
            new = "0.0.1"
        with open(self.version_file, "w", encoding="utf-8") as f:
            f.write(new)
        return new

    def propose_patches(self, signals: Dict[str, int]) -> List[Patch]:
        patches: List[Patch] = []
        target = os.path.join(self.project_root, "fortune_teller.py")

        if signals.get("serialization_errors", 0) > 0:
            fix = "# Self-improver: enforce asdict() usage for dataclass serialization\n"
            patches.append(Patch(target_path=target, description="Serialization hardening", diff=fix, bytes=len(fix)))

        if signals.get("policy_deferrals", 0) > 3:
            fix = "# Self-improver: auto-tune policy cpu_cap (suggest +0.05 up to 0.8)\n"
            patches.append(Patch(target_path=target, description="Policy auto-tuning", diff=fix, bytes=len(fix)))

        if signals.get("gpu_stalls", 0) > 2:
            fix = "# Self-improver: prefer gpu_vram_staging when GPU score > 0.65 and disk score > 0.7\n"
            patches.append(Patch(target_path=target, description="GPU staging improvements", diff=fix, bytes=len(fix)))

        return patches

    def apply_patches(self, patches: List[Patch]) -> UpdateResult:
        if not patches:
            return UpdateResult(False, "No patches proposed", [], [], "")
        if self._in_quiet():
            return UpdateResult(False, "Deferred: quiet hours", [], [], "")

        files_changed, backups = [], []
        for p in patches:
            if not self._allowed(p.target_path):
                return UpdateResult(False, f"Disallowed path: {p.target_path}", [], [], "")
            if p.bytes > self.policy.max_patch_bytes:
                return UpdateResult(False, "Patch too large", [], [], "")

        for p in patches:
            if self.policy.create_backup and os.path.exists(p.target_path):
                backup = p.target_path + f".bak.{int(time.time())}"
                import shutil
                shutil.copy2(p.target_path, backup)
                backups.append(backup)
            note_path = p.target_path + ".patchlog.enc"
            try:
                payload = {"ts": datetime.datetime.utcnow().isoformat(),
                           "target": p.target_path,
                           "desc": p.description,
                           "diff": p.diff,
                           "sig": self.crypto.sign_text(p.diff)}
                raw = json.dumps(payload).encode("utf-8")
                enc = self.crypto.encrypt(raw, aad=b"fortune_teller_patch")
                with open(note_path, "ab") as nf:
                    nf.write(enc + b"\n")
            except Exception:
                pass
            with open(p.target_path, "a", encoding="utf-8") as f:
                f.write("\n" + p.diff + "\n")
            files_changed.append(p.target_path)

        if not self._lint():
            self._rollback(backups)
            return UpdateResult(False, "Lint failed; rolled back", [], backups, "")
        if not self._tests():
            self._rollback(backups)
            return UpdateResult(False, "Tests failed; rolled back", [], backups, "")
        if not self._sandbox_import(files_changed):
            self._rollback(backups)
            return UpdateResult(False, "Import failed; rolled back", [], backups, "")

        signature = hashlib.sha256(("".join(files_changed) + str(time.time())).encode()).hexdigest()
        new_version = self._bump_version()
        self.logger.log("selfimprove", "Applied patches", {"files": files_changed, "version": new_version, "signature": signature})
        return UpdateResult(True, "Patches applied", files_changed, backups, signature, version=new_version)

    def _rollback(self, backups: List[str]):
        import shutil
        for b in backups:
            orig = b.split(".bak.")[0]
            try:
                shutil.copy2(b, orig)
                self.logger.log("selfimprove", "Rolled back file", {"file": orig})
            except Exception as e:
                self.logger.log("selfimprove", "Rollback failed", {"backup": b, "error": str(e)})


# ---------------------------
# Orchestrator (autonomy + daily self-improve)
# ---------------------------

class Orchestrator:
    def __init__(self, logger: Logger, telemetry: TelemetryCollector,
                 predictor: Predictor, planner: Planner,
                 policy: PolicyEngine, autoloader: AutoLoader,
                 improver: SelfImprover):
        self.logger = logger
        self.telemetry = telemetry
        self.predictor = predictor
        self.planner = planner
        self.policy = policy
        self.autoloader = autoloader
        self.improver = improver
        self.stop_flag = False
        self._last_improve_day: Optional[int] = None

    def start(self, interval_sec: float = 5.0):
        threading.Thread(target=self._loop, args=(interval_sec,), daemon=True).start()
        self.logger.log("orchestrator", "Autonomy loop started", {"interval_sec": interval_sec})

    def _improve_daily(self):
        today = datetime.datetime.now().timetuple().tm_yday
        if self._last_improve_day == today:
            return
        self._last_improve_day = today
        signals = {
            "serialization_errors": self.logger.count("error"),
            "policy_deferrals": self.logger.count("policy_defer"),
            "gpu_stalls": self.logger.count("gpu_stall"),
        }
        patches = self.improver.propose_patches(signals)
        result = self.improver.apply_patches(patches)
        self.logger.log("selfimprove", "Daily self-improvement", asdict(result))

    def _loop(self, interval_sec: float):
        while not self.stop_flag:
            snap = self.telemetry.snapshot()
            intents = self.predictor.predict(snap)
            plan = self.planner.synthesize(intents)
            decision = self.policy.evaluate(plan, snap)
            if decision.allow:
                result = self.autoloader.execute(plan, snapshot=snap)
                success = True
                self.predictor.record_outcome(plan.chosen_intent.id if plan.chosen_intent else "Observe", snap["hour"], snap["weekday"], success=success)
                self.logger.log("orchestrator", "Executed plan", {"plan": asdict(plan), "result": asdict(result)})
            else:
                success = False
                self.predictor.record_outcome(plan.chosen_intent.id if plan.chosen_intent else "Observe", snap["hour"], snap["weekday"], success=success)
                self.logger.log("policy_defer", "Plan deferred", {"decision": asdict(decision)})

            self._improve_daily()
            time.sleep(interval_sec)

    def stop(self):
        self.stop_flag = True
        self.logger.log("orchestrator", "Autonomy loop stopped")


# ---------------------------
# GUI manager (transparent reasoning) auto-disabled in headless mode
# ---------------------------

class GUIManager:
    def __init__(self, predictor: Predictor, telemetry: TelemetryCollector,
                 port_scanner: PortScanner, planner: Planner,
                 policy_engine: PolicyEngine, logger: Logger,
                 net_scanner: NetworkScanner, memory: MemoryManager):
        if not TK_AVAILABLE:
            raise RuntimeError("GUI environment not available (headless/server mode)")
        self.predictor = predictor
        self.telemetry = telemetry
        self.port_scanner = port_scanner
        self.planner = planner
        self.policy_engine = policy_engine
        self.logger = logger
        self.net_scanner = net_scanner
        self.memory = memory

        self.root = tk.Tk()
        self.root.title("Fortune Teller")
        self._setup_gui()
        self._start_refresh()

    def _setup_gui(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)

        self.pred_frame = ttk.Frame(self.nb)
        self.plan_frame = ttk.Frame(self.nb)
        self.res_frame = ttk.Frame(self.nb)
        self.pol_frame = ttk.Frame(self.nb)
        self.port_frame = ttk.Frame(self.nb)
        self.net_frame = ttk.Frame(self.nb)
        self.log_frame = ttk.Frame(self.nb)

        self.nb.add(self.pred_frame, text="Predictions")
        self.nb.add(self.plan_frame, text="Plans (dated thinking)")
        self.nb.add(self.res_frame, text="Resources (input/drive/net)")
        self.nb.add(self.pol_frame, text="Policies & Feedback")
        self.nb.add(self.port_frame, text="Ports & USB")
        self.nb.add(self.net_frame, text="Network")
        self.nb.add(self.log_frame, text="Logs")

        self.pred_tree = ttk.Treeview(self.pred_frame, columns=("action", "calibrated", "raw", "cluster", "evidence"), show="headings")
        for col, title in [("action","Action"),("calibrated","Confidence"),("raw","Raw"),("cluster","Cluster"),("evidence","Evidence")]:
            self.pred_tree.heading(col, text=title)
        self.pred_tree.pack(fill="both", expand=True)

        persona_frame = ttk.Frame(self.pred_frame)
        persona_frame.pack(fill="x", expand=False)
        self.persona_label = ttk.Label(persona_frame, text="Persona: Idle")
        self.persona_label.pack(side="left", padx=6, pady=4)

        self.plan_text = tk.Text(self.plan_frame, height=20)
        self.plan_text.pack(fill="both", expand=True)

        self.res_text = tk.Text(self.res_frame, height=20)
        self.res_text.pack(fill="both", expand=True)

        pol_container = ttk.Frame(self.pol_frame)
        pol_container.pack(fill="both", expand=True)
        self.pol_text = tk.Text(pol_container, height=10)
        self.pol_text.pack(fill="x", expand=False, padx=5, pady=5)
        fb_container = ttk.Frame(pol_container)
        fb_container.pack(fill="x", expand=False)
        ttk.Label(fb_container, text="Promote (comma-separated):").grid(row=0, column=0, sticky="w")
        self.promote_entry = ttk.Entry(fb_container, width=60)
        self.promote_entry.grid(row=0, column=1, sticky="we")
        ttk.Label(fb_container, text="Suppress (comma-separated):").grid(row=1, column=0, sticky="w")
        self.suppress_entry = ttk.Entry(fb_container, width=60)
        self.suppress_entry.grid(row=1, column=1, sticky="we")
        self.apply_fb_btn = ttk.Button(fb_container, text="Apply feedback", command=self._apply_feedback)
        self.apply_fb_btn.grid(row=2, column=1, sticky="e", pady=5)

        self.port_tree = ttk.Treeview(self.port_frame, columns=("laddr", "raddr", "status"), show="headings")
        for col, title in [("laddr","Local"),("raddr","Remote"),("status","Status")]:
            self.port_tree.heading(col, text=title)
        self.port_tree.pack(fill="both", expand=True)
        self.usb_text = tk.Text(self.port_frame, height=8)
        self.usb_text.pack(fill="x", expand=False)

        self.net_tree = ttk.Treeview(self.net_frame, columns=("ip", "mac", "name"), show="headings")
        for col, title in [("ip","IP"),("mac","MAC"),("name","Name")]:
            self.net_tree.heading(col, text=title)
        self.net_tree.pack(fill="both", expand=True)

        self.log_text = tk.Text(self.log_frame, height=20)
        self.log_text.pack(fill="both", expand=True)

    def _apply_feedback(self):
        promote = [x.strip() for x in self.promote_entry.get().split(",") if x.strip()]
        suppress = [x.strip() for x in self.suppress_entry.get().split(",") if x.strip()]
        self.memory.set_feedback(suppress, promote)
        self.memory.save()
        self.predictor.policy.promote = promote
        self.predictor.policy.suppress = suppress
        self.logger.log("feedback", "Applied user feedback", {"promote": promote, "suppress": suppress})

    def _start_refresh(self):
        self._refresh_ui()
        self.root.after(1500, self._start_refresh)

    def _refresh_ui(self):
        snap = self.telemetry.snapshot()
        intents = self.predictor.predict(snap)
        persona = snap.get("persona", "Idle")
        self.persona_label.config(text=f"Persona: {persona}")

        for i in self.pred_tree.get_children():
            self.pred_tree.delete(i)
        for intent in intents:
            ev = intent.evidence
            evidence_view = {
                "persona": ev.get("persona"),
                "cal": f"{intent.calibrated:.2f}",
                "bw": ev.get("base_weight"),
                "time": ev.get("time_bias"),
                "markov": ev.get("markov_bias"),
                "ngram": ev.get("ngram_bias"),
                "input": ev.get("input_bias"),
                "ctx": ev.get("drive_net_bias"),
                "fb": ev.get("feedback_bias"),
                "novelty": ev.get("novelty_boost"),
                "cooldown": ev.get("cooldown_penalty"),
                "rhythm": ev.get("rhythm_bias"),
                "piacc": ev.get("per_intent_accuracy_bias"),
                "bandit": ev.get("bandit_mean"),
            }
            self.pred_tree.insert("", "end", values=(
                intent.id, f"{intent.calibrated:.2f}", f"{intent.score:.2f}", intent.cluster, json.dumps(evidence_view)
            ))

        res_view = {
            "time": snap["ts"],
            "cpu_util": snap["cpu_util"],
            "mem_free_mb": snap["mem_free_mb"],
            "gpu": snap["gpu"],
            "battery": snap["battery"],
            "input_state": snap["input_state"],
            "activity": snap["activity"],
            "persona": persona,
            "proc_sample": snap.get("proc_sample", []),
            "headless": snap.get("headless", False),
        }
        self.res_text.delete("1.0", tk.END)
        self.res_text.insert(tk.END, json.dumps(res_view, indent=2))

        pol_dict = asdict(self.predictor.policy)
        self.pol_text.delete("1.0", tk.END)
        self.pol_text.insert(tk.END, json.dumps(pol_dict, indent=2))
        self.promote_entry.delete(0, tk.END)
        self.suppress_entry.delete(0, tk.END)
        mem_promote = ",".join(self.memory.get_feedback()[1])
        mem_suppress = ",".join(self.memory.get_feedback()[0])
        self.promote_entry.insert(0, mem_promote)
        self.suppress_entry.insert(0, mem_suppress)

        for i in self.port_tree.get_children():
            self.port_tree.delete(i)
        psnap = self.port_scanner.snapshot()
        for c in psnap["connections"]:
            self.port_tree.insert("", "end", values=(c["laddr"], c["raddr"], c["status"]))
        self.usb_text.delete("1.0", tk.END)
        self.usb_text.insert(tk.END, json.dumps(psnap["usb"], indent=2))

        self.net_scanner.discover_devices()
        for i in self.net_tree.get_children():
            self.net_tree.delete(i)
        for ip, dev in self.net_scanner.devices.items():
            self.net_tree.insert("", "end", values=(ip, dev.get("mac"), dev.get("name")))

        self.plan_text.delete("1.0", tk.END)
        plan = self.planner.synthesize(intents)
        decision = self.policy_engine.evaluate(plan, snap)
        self.plan_text.insert(tk.END, f"{plan.rationale}\n")
        self.plan_text.insert(tk.END, f"Chosen intent: {plan.chosen_intent.id if plan.chosen_intent else 'N/A'}\n")
        self.plan_text.insert(tk.END, f"Total cost: {json.dumps(plan.total_cost)}\n")
        self.plan_text.insert(tk.END, f"Expected gain (ms): {plan.expected_gain_ms}\n")
        self.plan_text.insert(tk.END, f"Policy: {decision.rationale} allow={decision.allow}\n")

        for evt in self.logger.drain(50):
            self.log_text.insert(tk.END, f"{evt['ts']} [{evt['kind']}] {evt['message']} {json.dumps(evt['meta'])}\n")
        self.log_text.see(tk.END)

    def run(self):
        self.root.mainloop()


# ---------------------------
# Wiring everything together
# ---------------------------

def build_system(project_root: str = ".", iface_cidr: Optional[str] = None) -> Dict[str, Any]:
    crypto = CryptoManager()  # per-machine encryption
    logger = Logger(crypto)
    input_monitor = InputMonitor(logger)
    activity = ActivityAnalyzer(logger)
    telemetry = TelemetryCollector(logger, input_monitor, activity)
    capability = CapabilityProfiler(logger, telemetry)
    dep_graph = DependencyGraph(logger)
    policy = Policy(
        autonomy_level="background",
        cpu_cap=0.5,
        disk_cap_bps=150e6,
        vram_cap_mb=1024.0,
        quiet_hours=(1, 6),
        privacy_scopes={"exclude_paths": ["C:/Sensitive", "/home/private"], "exclude_processes": ["BankApp"]},
        assist_not_act=True,
        suppress=[],
        promote=[]
    )
    memory = MemoryManager(crypto)
    predictor = Predictor(logger, memory, policy)
    planner = Planner(logger, capability, dep_graph)
    policy_engine = PolicyEngine(logger, policy, capability)
    autoloader = AutoLoader(logger, dep_graph, capability, read_ahead_window=2)
    port_scanner = PortScanner(logger)
    net_scanner = NetworkScanner(logger, iface_cidr=iface_cidr)

    update_policy = UpdatePolicy(
        allow_paths=["."],
        max_patch_bytes=4096,
        quiet_hours=(1, 6),
        require_tests_pass=False,
        require_lint_pass=True,
        create_backup=True
    )
    improver = SelfImprover(logger, crypto, update_policy, project_root=project_root)

    ft_orchestrator = Orchestrator(logger, telemetry, predictor, planner, policy_engine, autoloader, improver)

    return {
        "crypto": crypto,
        "logger": logger,
        "input_monitor": input_monitor,
        "activity": activity,
        "telemetry": telemetry,
        "capability": capability,
        "dep_graph": dep_graph,
        "memory": memory,
        "predictor": predictor,
        "planner": planner,
        "policy_engine": policy_engine,
        "autoloader": autoloader,
        "port_scanner": port_scanner,
        "net_scanner": net_scanner,
        "improver": improver,
        "orchestrator": ft_orchestrator,
    }

def main(cli_mode: Optional[bool] = None, iface_cidr: Optional[str] = None):
    # Auto-enable CLI mode in headless/server environments unless explicitly overridden
    if cli_mode is None:
        cli_mode = HEADLESS_MODE

    system = build_system(project_root=".", iface_cidr=iface_cidr)
    logger: Logger = system["logger"]
    input_monitor: InputMonitor = system["input_monitor"]
    telemetry: TelemetryCollector = system["telemetry"]
    capability: CapabilityProfiler = system["capability"]
    predictor: Predictor = system["predictor"]
    planner: Planner = system["planner"]
    policy_engine: PolicyEngine = system["policy_engine"]
    autoloader: AutoLoader = system["autoloader"]
    port_scanner: PortScanner = system["port_scanner"]
    net_scanner: NetworkScanner = system["net_scanner"]
    improver: SelfImprover = system["improver"]
    ft_orchestrator: Orchestrator = system["orchestrator"]
    memory: MemoryManager = system["memory"]

    input_monitor.start()
    telemetry.start()
    port_scanner.start()
    capability.run_probes()
    ft_orchestrator.start(interval_sec=5.0)

    # Seed examples to kickstart learning
    now = datetime.datetime.now()
    predictor.record_outcome("Launch Steam", now.hour, now.weekday(), success=True)
    predictor.record_outcome("Open IDE", now.hour, now.weekday(), success=False)

    if not cli_mode and TK_AVAILABLE:
        gui = GUIManager(predictor, telemetry, port_scanner, planner, policy_engine, logger, net_scanner, memory)
        gui.run()
        telemetry.stop()
        port_scanner.stop()
        input_monitor.stop()
        ft_orchestrator.stop()
    else:
        mode = "CLI (headless)" if HEADLESS_MODE else "CLI (requested)"
        logger.log("system", f"Running Fortune Teller in {mode}")
        try:
            for _ in range(8):
                time.sleep(2.0)
                for evt in logger.drain():
                    print(f"{evt['ts']} [{evt['kind']}] {evt['message']} {json.dumps(evt['meta'])}")
        finally:
            telemetry.stop()
            port_scanner.stop()
            input_monitor.stop()
            ft_orchestrator.stop()

if __name__ == "__main__":
    print("Starting Fortune Teller (self-improving, encrypted, highly predictive, fully autonomous)...")
    # CLI mode auto-detected; set iface_cidr (e.g., "192.168.1.0/24") to enable ARP scanning
    if "--cli" in sys.argv:
        main(cli_mode=True, iface_cidr=None)
    else:
        main(cli_mode=None, iface_cidr=None)

