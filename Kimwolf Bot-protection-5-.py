#!/usr/bin/env python3
"""
BorgShield Queen v1.2 (Unified backbone)

Features:
- Cross-platform (Windows, macOS, Linux)
- Autonomous: sensors + engine start on launch
- Baseline scan on first run:
  - OS, hardware, basic software directories
- Persistent memory backbone:
  - Saves stats + baseline to configurable directory (local or SMB/mounted share)
  - Configurable via GUI "Select Memory Location" button
- Local sensors:
  - CPU, memory, processes
  - Network connections (ports)
  - File system (watch directory)
- Hybrid brain:
  - Rules + heuristics + baseline deviation + predictive context
- Situational awareness cortex:
  - Threat level, mode, health score, predicted next risk
- Baseline manager:
  - Per-source profiling (router/net/chat/api/etc.)
- Router support:
  - Rule types for router logs/flows (placeholders)
  - Router Feed panel in GUI (wired to show router events)
- External ingestion:
  - ingest_external_event(source, content, metadata) for router/chat/api/socket feeds
"""

import sys
import subprocess
import importlib
import threading
import time
import json
import queue
import platform
import shutil
import os
from pathlib import Path
from collections import deque

# =========================
# 1. AUTOLOADER
# =========================

REQUIRED_PACKAGES = [
    "requests",   # reserved for future Borg sync / remote hive
    "psutil",     # CPU, memory, processes, connections
    "watchdog",   # filesystem events
]

def ensure_package(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
        return True, False
    except ImportError:
        print(f"[AUTOLOADER] Installing '{pkg_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.invalidate_caches()
            importlib.import_module(pkg_name)
            return True, True
        except Exception as e:
            print(f"[AUTOLOADER] Failed to install '{pkg_name}': {e}")
            return False, False

def autoload_dependencies():
    status = {}
    for pkg in REQUIRED_PACKAGES:
        ok, installed_now = ensure_package(pkg)
        status[pkg] = {"ok": ok, "installed_now": installed_now}
    return status

AUTOLOADER_STATUS = autoload_dependencies()

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    tk = None
    ttk = None
    filedialog = None

# =========================
# 2. DATA DIR, CONFIG, BASELINE, MEMORY
# =========================

DATA_DIR = Path.home() / ".borgshield"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SIGNATURE_FILE = DATA_DIR / "signatures.json"
BASELINE_FILE = DATA_DIR / "baseline.json"
CONFIG_FILE = DATA_DIR / "config.json"

DEFAULT_CONFIG = {
    "memory_dir": None  # if None, use DATA_DIR
}

DEFAULT_SIGNATURES = {
    "version": 2,
    "rules": [
        # Generic text/keyword example
        {
            "id": "kw_kimwolf_example",
            "type": "keyword",
            "pattern": "kimwolf",
            "action": "flag"
        },
        # Router log keyword signatures (placeholders)
        {
            "id": "router_port_scan",
            "type": "router_keyword",
            "pattern": "PORT SCAN",
            "action": "block"
        },
        {
            "id": "router_dos_detected",
            "type": "router_keyword",
            "pattern": "DOS",
            "action": "block"
        },
        {
            "id": "router_blocked_suspicious",
            "type": "router_keyword",
            "pattern": "BLOCKED",
            "action": "flag"
        },
        # Example subnet placeholder; replace with real kimwolf IP range later
        {
            "id": "router_kimwolf_ip",
            "type": "router_ip",
            "pattern": "203.0.113.",  # RFC 5737 test-net as example
            "action": "block"
        }
    ]
}

def load_signatures():
    if not SIGNATURE_FILE.exists():
        save_signatures(DEFAULT_SIGNATURES)
        return DEFAULT_SIGNATURES
    try:
        with SIGNATURE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[BORG] Load signatures failed, defaults used: {e}")
        return DEFAULT_SIGNATURES

def save_signatures(data):
    try:
        with SIGNATURE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[BORG] Save signatures failed: {e}")
        return False

def load_config():
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception as e:
        print(f"[CONFIG] Load failed, defaults used: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"[CONFIG] Save failed: {e}")
        return False

def scan_system_baseline():
    print("[BASELINE] Scanning system baseline...")
    info = {}
    info["os_system"] = platform.system()
    info["os_release"] = platform.release()
    info["os_version"] = platform.version()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor()

    try:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        info["cpu_count_logical"] = None
        info["cpu_count_physical"] = None

    try:
        mem = psutil.virtual_memory()
        info["memory_total_bytes"] = mem.total
    except Exception:
        info["memory_total_bytes"] = None

    try:
        disk = shutil.disk_usage(str(Path.home()))
        info["disk_total_bytes"] = disk.total
    except Exception:
        info["disk_total_bytes"] = None

    software_dirs = []
    system = info["os_system"].lower()
    if system == "windows":
        pf = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        pf86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        software_dirs.extend([str(pf), str(pf86)])
    elif system == "darwin":
        software_dirs.append("/Applications")
    else:
        software_dirs.extend(["/usr/bin", "/usr/local/bin"])

    info["software_dirs"] = software_dirs
    info["timestamp"] = time.time()
    return info

def load_or_create_baseline():
    if BASELINE_FILE.exists():
        try:
            with BASELINE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BASELINE] Load failed, rescanning: {e}")

    baseline = scan_system_baseline()
    try:
        with BASELINE_FILE.open("w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
    except Exception as e:
        print(f"[BASELINE] Save failed: {e}")
    return baseline

class MemoryManager:
    """
    Manages where memory is stored.
    Memory is an aggregated snapshot (not full event history).
    """
    def __init__(self, config, baseline):
        self.config = config
        self.baseline = baseline
        self._memory_dir_path = self._resolve_memory_dir()

    def _resolve_memory_dir(self):
        mem_dir = self.config.get("memory_dir")
        if mem_dir:
            p = Path(mem_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            return DATA_DIR

    def set_memory_dir(self, path_str):
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        self._memory_dir_path = p
        self.config["memory_dir"] = str(p)
        save_config(self.config)

    @property
    def memory_dir(self):
        return self._memory_dir_path

    @property
    def memory_file(self):
        return self._memory_dir_path / "borg_memory.json"

    def save_memory_snapshot(self, context_stats, signatures_version):
        snapshot = {
            "timestamp": time.time(),
            "baseline": self.baseline,
            "context_stats": context_stats,
            "signatures_version": signatures_version
        }
        try:
            with self.memory_file.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"[MEMORY] Save failed: {e}")

    def load_memory_snapshot(self):
        if not self.memory_file.exists():
            return None
        try:
            with self.memory_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[MEMORY] Load failed: {e}")
            return None

# =========================
# 3. SIGNATURES + SYNC
# =========================

BORG_SIGNATURES = load_signatures()

def borg_sync_stub():
    global BORG_SIGNATURES
    BORG_SIGNATURES = load_signatures()
    return {"status": "local_refresh", "version": BORG_SIGNATURES.get("version", 0)}

# =========================
# 4. HYBRID BRAIN / CONTEXT
# =========================

class Event:
    def __init__(self, source, content, metadata=None):
        self.source = source
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()

class Decision:
    def __init__(self, event, verdict, reason, rules_triggered=None,
                 base_risk=0.0, predicted_risk=0.0, confidence=0.5,
                 threat_level="CALM", mode="CALM"):
        self.event = event
        self.verdict = verdict
        self.reason = reason
        self.rules_triggered = rules_triggered or []
        self.timestamp = time.time()
        self.base_risk = base_risk
        self.predicted_risk = predicted_risk
        self.confidence = confidence
        self.threat_level = threat_level
        self.mode = mode

class SituationalContext:
    def __init__(self, window_size=500):
        self.window = deque(maxlen=window_size)

    def record(self, decision: Decision):
        self.window.append(decision)

    def stats(self):
        if not self.window:
            return {
                "count": 0,
                "hostile_ratio": 0.0,
                "avg_risk": 0.0,
                "events_per_min": 0.0
            }
        now = time.time()
        count = len(self.window)
        hostile = sum(1 for d in self.window if d.verdict in ("flag", "block"))
        hostile_ratio = hostile / count
        avg_risk = sum(d.base_risk for d in self.window) / count
        times = [d.timestamp for d in self.window]
        span = max(1.0, max(times) - min(times))
        events_per_min = count / (span / 60.0)
        return {
            "count": count,
            "hostile_ratio": hostile_ratio,
            "avg_risk": avg_risk,
            "events_per_min": events_per_min
        }

    def threat_level(self):
        s = self.stats()
        combined = 0.6 * s["hostile_ratio"] + 0.4 * s["avg_risk"]
        if combined < 0.2:
            return "CALM"
        elif combined < 0.5:
            return "FOCUSED"
        else:
            return "ALTERED"

    def mode(self):
        tl = self.threat_level()
        if tl == "CALM":
            return "CALM"
        elif tl == "FOCUSED":
            return "FOCUSED"
        else:
            return "ALTERED"

    def predicted_next_risk(self):
        s = self.stats()
        base = 0.5 * s["hostile_ratio"] + 0.5 * s["avg_risk"]
        rate_boost = min(0.3, s["events_per_min"] / 100.0)
        return max(0.0, min(1.0, base + rate_boost))

    def health_score(self):
        s = self.stats()
        score = 100.0
        score -= s["hostile_ratio"] * 50.0
        score -= s["avg_risk"] * 30.0
        if s["events_per_min"] > 200:
            score -= 10.0
        return max(0.0, min(100.0, score))

# =========================
# 5. BASELINE MANAGER
# =========================

class BaselineProfile:
    def __init__(self):
        self.event_count = 0
        self.last_update = time.time()
        self.ports_seen = {}
        self.ip_seen = {}
        self.router_flags_seen = {}

    def update_from_event(self, event: Event):
        self.event_count += 1
        self.last_update = time.time()

        src = event.source
        meta = event.metadata or {}

        if src in ("router", "net"):
            conn_list = meta.get("connections", [])
            for c in conn_list:
                for key in ("laddr", "raddr"):
                    val = c.get(key, "")
                    if ":" in val:
                        _, port = val.rsplit(":", 1)
                        if port.isdigit():
                            p = int(port)
                            self.ports_seen[p] = self.ports_seen.get(p, 0) + 1

            flag = meta.get("router_flag")
            if flag:
                self.router_flags_seen[flag] = self.router_flags_seen.get(flag, 0) + 1

            ip = meta.get("src_ip")
            if ip:
                self.ip_seen[ip] = self.ip_seen.get(ip, 0) + 1

    def deviation_score(self, event: Event):
        if self.event_count < 50:
            return 0.2

        score = 0.0
        meta = event.metadata or {}
        src = event.source

        if src in ("router", "net"):
            conn_list = meta.get("connections", [])
            unseen_ports = 0
            total_ports = 0
            for c in conn_list:
                for key in ("laddr", "raddr"):
                    val = c.get(key, "")
                    if ":" in val:
                        _, port = val.rsplit(":", 1)
                        if port.isdigit():
                            p = int(port)
                            total_ports += 1
                            if p not in self.ports_seen:
                                unseen_ports += 1
            if total_ports > 0:
                ratio_new = unseen_ports / total_ports
                score += ratio_new * 0.5

            flag = meta.get("router_flag")
            if flag and flag not in self.router_flags_seen:
                score += 0.3

            ip = meta.get("src_ip")
            if ip and ip not in self.ip_seen:
                score += 0.4

        return max(0.0, min(1.0, score))

class BaselineManager:
    def __init__(self):
        self.profiles = {}

    def get_profile(self, source: str) -> BaselineProfile:
        if source not in self.profiles:
            self.profiles[source] = BaselineProfile()
        return self.profiles[source]

    def update(self, event: Event):
        profile = self.get_profile(event.source)
        profile.update_from_event(event)

    def deviation_score(self, event: Event):
        profile = self.get_profile(event.source)
        return profile.deviation_score(event)

# =========================
# 6. RULES + PREDICTIVE JUDGMENT
# =========================

def apply_rules(event, signatures):
    triggered = []
    base_risk = 0.0
    meta = event.metadata or {}
    src = event.source
    text_content = str(event.content).lower()

    for rule in signatures.get("rules", []):
        rtype = rule.get("type")
        pattern = rule.get("pattern", "")

        if rtype == "keyword":
            if pattern and pattern.lower() in text_content:
                triggered.append(rule)

        elif rtype == "router_keyword" and src == "router":
            patt = pattern.lower()
            router_flag = str(meta.get("router_flag", "")).lower()
            if patt and (patt in text_content or patt in router_flag):
                triggered.append(rule)

        elif rtype == "router_ip" and src == "router":
            src_ip = str(meta.get("src_ip", ""))
            if pattern and src_ip.startswith(pattern):
                triggered.append(rule)

    if src in ("net", "socket", "chat", "api", "router"):
        base_risk += 0.3
    elif src in ("file", "process"):
        base_risk += 0.2
    else:
        base_risk += 0.1

    if not triggered:
        return Decision(
            event=event,
            verdict="allow",
            reason="no_rules_triggered",
            rules_triggered=[],
            base_risk=min(1.0, base_risk)
        )

    base_risk += 0.5 + 0.1 * (len(triggered) - 1)

    strongest_action = "flag"
    for r in triggered:
        if r.get("action") == "block":
            strongest_action = "block"
            break

    return Decision(
        event=event,
        verdict=strongest_action,
        reason=f"{len(triggered)} rule(s) triggered",
        rules_triggered=[r.get("id") for r in triggered],
        base_risk=min(1.0, base_risk)
    )

def integrate_prediction(decision: Decision, context: SituationalContext):
    predicted_risk = context.predicted_next_risk()
    agreement = 1.0 - abs(decision.base_risk - predicted_risk)
    confidence = 0.4 + 0.6 * agreement
    tl = context.threat_level()
    mode = context.mode()

    decision.predicted_risk = predicted_risk
    decision.confidence = max(0.0, min(1.0, confidence))
    decision.threat_level = tl
    decision.mode = mode
    return decision

# =========================
# 7. SENSORS (LOCAL)
# =========================

class FileSensorHandler(FileSystemEventHandler):
    def __init__(self, event_queue, watch_label="file"):
        super().__init__()
        self.event_queue = event_queue
        self.watch_label = watch_label

    def on_any_event(self, event):
        meta = {
            "event_type": event.event_type,
            "is_directory": event.is_directory,
            "src_path": event.src_path
        }
        e = Event(
            source=self.watch_label,
            content=f"File event: {event.event_type} {event.src_path}",
            metadata=meta
        )
        self.event_queue.put(e)

class SensorManager(threading.Thread):
    def __init__(self, raw_event_queue, watch_dir=None, sample_interval=3.0):
        super().__init__(daemon=True)
        self.raw_event_queue = raw_event_queue
        self.sample_interval = sample_interval
        self.running = False

        self.watch_dir = watch_dir or (Path.home() / "borgshield_watch")
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.observer = Observer()
        handler = FileSensorHandler(self.raw_event_queue, watch_label="file")
        self.observer.schedule(handler, str(self.watch_dir), recursive=True)

    def run(self):
        self.running = True
        self.observer.start()
        print("[SENSORS] Started, watching:", self.watch_dir)
        try:
            while self.running:
                self._sample_system()
                time.sleep(self.sample_interval)
        finally:
            self.observer.stop()
            self.observer.join()
            self.running = False
            print("[SENSORS] Stopped")

    def stop(self):
        self.running = False

    def _sample_system(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        e = Event("system", f"CPU={cpu:.1f}%, MEM={mem:.1f}%", {"cpu": cpu, "mem": mem})
        self.raw_event_queue.put(e)

        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent"]):
            try:
                info = p.info
                procs.append(info)
            except psutil.NoSuchProcess:
                continue
        procs = sorted(procs, key=lambda x: x.get("cpu_percent", 0), reverse=True)[:5]
        e2 = Event("process", "Top processes snapshot", {"top": procs})
        self.raw_event_queue.put(e2)

        try:
            conns = psutil.net_connections(kind="inet")
            open_ports = []
            for c in conns:
                laddr = getattr(c, "laddr", None)
                raddr = getattr(c, "raddr", None)
                if laddr:
                    open_ports.append({
                        "laddr": f"{laddr.ip}:{laddr.port}",
                        "raddr": f"{getattr(raddr, 'ip', '')}:{getattr(raddr, 'port', '')}" if raddr else "",
                        "status": c.status
                    })
            e3 = Event("net", f"Net snapshot ({len(open_ports)} entries)", {"connections": open_ports[:50]})
            self.raw_event_queue.put(e3)
        except Exception as ex:
            e3 = Event("net", f"Net snapshot error: {ex}", {})
            self.raw_event_queue.put(e3)

# =========================
# 8. EXTERNAL INGESTION HOOK
# =========================

EXTERNAL_EVENT_QUEUE = None

def ingest_external_event(source: str, content: str, metadata=None):
    global EXTERNAL_EVENT_QUEUE
    if EXTERNAL_EVENT_QUEUE is None:
        return
    e = Event(source, content, metadata or {})
    EXTERNAL_EVENT_QUEUE.put(e)

# =========================
# 9. DEFENSE ENGINE
# =========================

class DefenseEngine(threading.Thread):
    def __init__(self, raw_event_queue, decision_queue, control_event,
                 context: SituationalContext, baseline_manager: BaselineManager,
                 memory_manager: MemoryManager):
        super().__init__(daemon=True)
        self.raw_event_queue = raw_event_queue
        self.decision_queue = decision_queue
        self.control_event = control_event
        self.context = context
        self.baseline_manager = baseline_manager
        self.memory_manager = memory_manager
        self.running = False
        self._last_memory_save = 0.0
        self._memory_save_interval = 10.0

    def run(self):
        self.running = True
        print("[ENGINE] Started")
        while not self.control_event.is_set():
            try:
                event = self.raw_event_queue.get(timeout=1.0)
            except queue.Empty:
                self._maybe_save_memory()
                continue

            self.baseline_manager.update(event)
            deviation = self.baseline_manager.deviation_score(event)

            decision = apply_rules(event, BORG_SIGNATURES)
            decision.base_risk = max(
                0.0,
                min(1.0, decision.base_risk + deviation * 0.5)
            )

            decision = integrate_prediction(decision, self.context)
            self.context.record(decision)
            self.decision_queue.put(decision)
            self._maybe_save_memory()

        self.running = False
        print("[ENGINE] Stopped")

    def _maybe_save_memory(self):
        now = time.time()
        if now - self._last_memory_save >= self._memory_save_interval:
            stats = self.context.stats()
            signatures_version = BORG_SIGNATURES.get("version", 0)
            self.memory_manager.save_memory_snapshot(stats, signatures_version)
            self._last_memory_save = now

# =========================
# 10. GUI
# =========================

class BorgShieldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BorgShield Queen v1.2")

        self.config = load_config()
        self.baseline = load_or_create_baseline()
        self.memory_manager = MemoryManager(self.config, self.baseline)

        self.raw_event_queue = queue.Queue()
        self.decision_queue = queue.Queue()
        self.engine_control_event = threading.Event()
        self.context = SituationalContext()
        self.baseline_manager = BaselineManager()

        global EXTERNAL_EVENT_QUEUE
        EXTERNAL_EVENT_QUEUE = self.raw_event_queue

        self.sensor_manager = SensorManager(self.raw_event_queue)
        self.engine_thread = DefenseEngine(
            self.raw_event_queue,
            self.decision_queue,
            self.engine_control_event,
            self.context,
            self.baseline_manager,
            self.memory_manager
        )

        self.status_var = tk.StringVar(value="STARTING")
        self.signatures_var = tk.StringVar(value=f"v{BORG_SIGNATURES.get('version', 0)}")
        self.threat_var = tk.StringVar(value="CALM")
        self.mode_var = tk.StringVar(value="CALM")
        self.health_var = tk.StringVar(value="100.0")
        self.predicted_risk_var = tk.StringVar(value="0.00")
        self.memory_path_var = tk.StringVar(value=str(self.memory_manager.memory_dir))
        self.baseline_os_var = tk.StringVar(
            value=f"{self.baseline.get('os_system','?')} {self.baseline.get('os_release','?')}"
        )
        self.router_status_var = tk.StringVar(value="Waiting for router data...")

        self._build_layout()
        self._start_autonomous()
        self._poll_decisions()
        self._update_context_labels()

    def _build_layout(self):
        status_frame = ttk.LabelFrame(self.root, text="Node Status")
        status_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

        ttk.Label(status_frame, text="Signatures:").grid(row=1, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.signatures_var).grid(row=1, column=1, sticky="w")

        ttk.Label(status_frame, text="Threat Level:").grid(row=2, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.threat_var).grid(row=2, column=1, sticky="w")

        ttk.Label(status_frame, text="Mode:").grid(row=3, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.mode_var).grid(row=3, column=1, sticky="w")

        ttk.Label(status_frame, text="Health Score:").grid(row=4, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.health_var).grid(row=4, column=1, sticky="w")

        ttk.Label(status_frame, text="Predicted Risk (Next):").grid(row=5, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.predicted_risk_var).grid(row=5, column=1, sticky="w")

        ttk.Label(status_frame, text="Baseline OS:").grid(row=6, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.baseline_os_var).grid(row=6, column=1, sticky="w")

        mem_frame = ttk.LabelFrame(self.root, text="Memory Backbone")
        mem_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(mem_frame, text="Memory Path:").grid(row=0, column=0, sticky="w")
        ttk.Label(mem_frame, textvariable=self.memory_path_var, wraplength=420).grid(row=0, column=1, sticky="w")

        self.mem_button = ttk.Button(mem_frame, text="Select Memory Location", command=self.select_memory_location)
        self.mem_button.grid(row=1, column=0, columnspan=2, pady=4, sticky="w")

        events_frame = ttk.LabelFrame(self.root, text="Recent Decisions")
        events_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.events_list = tk.Listbox(events_frame, height=12)
        self.events_list.pack(fill="both", expand=True, padx=4, pady=4)

        router_frame = ttk.LabelFrame(self.root, text="Router Feed (Reserved)")
        router_frame.pack(fill="both", expand=False, padx=8, pady=4)

        ttk.Label(router_frame, textvariable=self.router_status_var).pack(anchor="w", padx=4, pady=2)

        self.router_list = tk.Listbox(router_frame, height=6)
        self.router_list.pack(fill="both", expand=True, padx=4, pady=4)

        self.router_config_button = ttk.Button(
            router_frame,
            text="Configure Router Input",
            command=lambda: None  # placeholder for future config
        )
        self.router_config_button.pack(anchor="e", padx=4, pady=4)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=4)

        self.start_button = ttk.Button(controls, text="Start", command=self.start_engine)
        self.start_button.pack(side="left", padx=4)

        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_engine)
        self.stop_button.pack(side="left", padx=4)

        self.sync_button = ttk.Button(controls, text="Borg Sync", command=self.do_borg_sync)
        self.sync_button.pack(side="right", padx=4)

    def _start_autonomous(self):
        self.sensor_manager.start()
        self.engine_control_event.clear()
        self.engine_thread.start()
        self.status_var.set("RUNNING")
        self.start_button.config(state="disabled")

    def start_engine(self):
        if self.engine_thread.running:
            return
        self.engine_control_event.clear()
        self.engine_thread = DefenseEngine(
            self.raw_event_queue,
            self.decision_queue,
            self.engine_control_event,
            self.context,
            self.baseline_manager,
            self.memory_manager
        )
        self.engine_thread.start()
        self.status_var.set("RUNNING")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

    def stop_engine(self):
        self.engine_control_event.set()
        self.sensor_manager.stop()
        self.status_var.set("STOPPING...")
        self.stop_button.config(state="disabled")
        self.root.after(500, self._check_stopped)

    def _check_stopped(self):
        if not self.engine_thread.running and not self.sensor_manager.running:
            self.status_var.set("IDLE")
            self.start_button.config(state="normal")
        else:
            self.root.after(500, self._check_stopped)

    def do_borg_sync(self):
        result = borg_sync_stub()
        self.signatures_var.set(f"v{result['version']}")

    def select_memory_location(self):
        if filedialog is None:
            print("[MEMORY] filedialog not available.")
            return
        directory = filedialog.askdirectory(title="Select Memory Storage Directory")
        if directory:
            self.memory_manager.set_memory_dir(directory)
            self.memory_path_var.set(str(self.memory_manager.memory_dir))
            stats = self.context.stats()
            signatures_version = BORG_SIGNATURES.get("version", 0)
            self.memory_manager.save_memory_snapshot(stats, signatures_version)

    def _poll_decisions(self):
        try:
            while True:
                d = self.decision_queue.get_nowait()
                ts = time.strftime("%H:%M:%S", time.localtime(d.timestamp))
                src = d.event.source
                line = (
                    f"[{ts}] {d.verdict.upper()} src={src} "
                    f"mode={d.mode} "
                    f"risk={d.base_risk:.2f}|{d.predicted_risk:.2f} "
                    f"conf={d.confidence:.2f} "
                    f"reason={d.reason}"
                )
                self.events_list.insert(0, line)
                if self.events_list.size() > 400:
                    self.events_list.delete(400, tk.END)

                if src == "router":
                    self.router_status_var.set("Receiving router events")
                    router_line = f"[{ts}] {d.event.content}"
                    self.router_list.insert(0, router_line)
                    if self.router_list.size() > 200:
                        self.router_list.delete(200, tk.END)
        except queue.Empty:
            pass
        self.root.after(500, self._poll_decisions)

    def _update_context_labels(self):
        tl = self.context.threat_level()
        mode = self.context.mode()
        health = self.context.health_score()
        predicted = self.context.predicted_next_risk()

        self.threat_var.set(tl)
        self.mode_var.set(mode)
        self.health_var.set(f"{health:.1f}")
        self.predicted_risk_var.set(f"{predicted:.2f}")

        self.root.after(1000, self._update_context_labels)

# =========================
# 11. MAIN
# =========================

def main():
    if tk is None:
        print("tkinter not available. Install python3-tk or equivalent.")
        sys.exit(1)
    root = tk.Tk()
    BorgShieldGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

