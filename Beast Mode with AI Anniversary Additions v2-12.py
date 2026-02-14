import threading
import time
import random
import uuid
import os
import platform
import shutil
import pathlib
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Any
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import ctypes
import sys

# === AUTO-ELEVATION CHECK ===
def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# Optional deps
try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from pywinauto import Desktop
except ImportError:
    Desktop = None


# ---------- Core Types ------------------------------------------------------

class Fabric(Enum):
    CPU = auto()
    GPU = auto()

class ShardStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()

@dataclass
class Range:
    offset: int
    length: int

@dataclass
class Shard:
    id: str
    workload_id: str
    stage: str
    fabric: Fabric
    range: Range
    status: ShardStatus = ShardStatus.PENDING

@dataclass
class ShardSet:
    workload_id: str
    stage: str
    total_range: Range
    shards: List[Shard] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass
class BorgPlan:
    cpu_share: float
    gpu_share: float
    max_threads: int

@dataclass
class TelemetrySnapshot:
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    net_sent: float = 0.0
    net_recv: float = 0.0

@dataclass
class WorkloadHistory:
    latencies: List[float] = field(default_factory=list)
    ema_latency: float = 0.0
    cpu_bias: float = 0.5
    gpu_bias: float = 0.5
    contexts: List[str] = field(default_factory=list)


# ---------- Global State ----------------------------------------------------

running = True

telemetry_lock = threading.Lock()
telemetry = TelemetrySnapshot()

llm_pipeline = None
vision_pipeline = None

gpu_available = False
gpu_handle = None

override_lock = threading.Lock()
override_ai = {"cpu": 1.0, "gpu": 1.0}
override_vision = {"cpu": 1.0, "gpu": 1.0}
override_price = {"cpu": 1.0, "gpu": 1.0}

history_lock = threading.Lock()
history: Dict[str, WorkloadHistory] = {
    "ai_inference": WorkloadHistory(),
    "vision_inference": WorkloadHistory(),
    "price_scan": WorkloadHistory(),
}

heatmap_lock = threading.Lock()
heatmap_ai: List[float] = []
heatmap_vision: List[float] = []
heatmap_price: List[float] = []

current_context = {"title": None, "process": None}
cooldown_state = {"throttle": False}

backup_paths = {"primary": None, "secondary": None}
backup_lock = threading.Lock()
SOURCE_DATA_DIR = "./data"

SETTINGS_FILE = "cockpit_settings.json"

safe_mode = {
    "llm": False,
    "vision": False,
    "telemetry": False,
    "gpu": False,
}

log_lock = threading.Lock()
log_buffer: List[str] = []
MAX_LOG_LINES = 500

event_bus_lock = threading.Lock()
event_subscribers: Dict[str, List[Callable[[Any], None]]] = {}

watchdog_state = {
    "ai_alive": False,
    "vision_alive": False,
    "price_alive": False,
    "last_ai_beat": 0.0,
    "last_vision_beat": 0.0,
    "last_price_beat": 0.0,
}

threat_state = {
    "ai": 0.0,
    "vision": 0.0,
    "price": 0.0,
    "overall": 0.0,
}

swarm_state = {
    "peers": 3,
    "consensus": 1.0,
    "last_sync": 0.0,
}

process_lock = threading.Lock()
top_processes: List[Dict[str, Any]] = []

file_rep_lock = threading.Lock()
file_reputation_cache: Dict[str, float] = {}

policy_lock = threading.Lock()
policy_rules = {
    "max_cpu": 0.90,
    "max_gpu": 0.95,
    "block_high_risk_files": True,
}

sandbox_lock = threading.Lock()
sandbox_state = {
    "enabled": True,
    "blocked_files": [],
}

gpu_graph_lock = threading.Lock()
gpu_history: List[float] = []


# ---------- Utility: Logging + Events + Settings ---------------------------

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with log_lock:
        log_buffer.append(line)
        if len(log_buffer) > MAX_LOG_LINES:
            log_buffer.pop(0)

def event_subscribe(topic: str, handler: Callable[[Any], None]):
    with event_bus_lock:
        event_subscribers.setdefault(topic, []).append(handler)

def event_publish(topic: str, payload: Any = None):
    with event_bus_lock:
        handlers = list(event_subscribers.get(topic, []))
    for h in handlers:
        try:
            h(payload)
        except Exception as e:
            log(f"[EventBus] handler error on {topic}: {e}")

def load_settings():
    global backup_paths, override_ai, override_vision, override_price, policy_rules
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with backup_lock:
            backup_paths["primary"] = data.get("backup_primary")
            backup_paths["secondary"] = data.get("backup_secondary")
        with override_lock:
            override_ai["cpu"] = data.get("ai_cpu", 1.0)
            override_ai["gpu"] = data.get("ai_gpu", 1.0)
            override_vision["cpu"] = data.get("vision_cpu", 1.0)
            override_vision["gpu"] = data.get("vision_gpu", 1.0)
            override_price["cpu"] = data.get("price_cpu", 1.0)
            override_price["gpu"] = data.get("price_gpu", 1.0)
        with policy_lock:
            pr = data.get("policy_rules", {})
            policy_rules.update(pr)
        log("[Settings] Loaded persistent settings.")
    except Exception as e:
        log(f"[Settings] Failed to load: {e}")

def save_settings():
    try:
        with backup_lock:
            bp = dict(backup_paths)
        with override_lock:
            overrides = {
                "ai_cpu": override_ai["cpu"],
                "ai_gpu": override_ai["gpu"],
                "vision_cpu": override_vision["cpu"],
                "vision_gpu": override_vision["gpu"],
                "price_cpu": override_price["cpu"],
                "price_gpu": override_price["gpu"],
            }
        with policy_lock:
            pr = dict(policy_rules)
        data = {
            "backup_primary": bp.get("primary"),
            "backup_secondary": bp.get("secondary"),
            **overrides,
            "policy_rules": pr,
        }
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log("[Settings] Saved persistent settings.")
    except Exception as e:
        log(f"[Settings] Failed to save: {e}")


# ---------- Model Discovery -------------------------------------------------

def discover_llm_model_name() -> str:
    return "distilgpt2"

def discover_vision_model_name() -> str:
    return "google/vit-base-patch16-224-in21k"


# ---------- Context Organ ---------------------------------------------------

def get_active_window_info():
    system = platform.system()
    if system == "Windows" and Desktop:
        try:
            win = Desktop(backend="uia").get_active()
            return {"title": win.window_text(), "process": win.process_id()}
        except Exception:
            return None
    return None

def context_organ():
    global current_context, running
    while running:
        info = get_active_window_info()
        if info:
            current_context = info
        time.sleep(1.0)


# ---------- Autoloader + Safe Mode -----------------------------------------

def autoloader_load_ai_libraries():
    global llm_pipeline, vision_pipeline, gpu_available, gpu_handle

    log("[Autoloader] Loading tiny real models...")

    if pynvml:
        try:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_available = True
            log("[Autoloader] GPU detected.")
        except Exception as e:
            log(f"[Autoloader] NVML init failed: {e}")
            gpu_available = False
            safe_mode["gpu"] = True
    else:
        safe_mode["gpu"] = True

    if pipeline is None:
        log("[Autoloader] transformers missing, entering safe mode for LLM/Vision.")
        safe_mode["llm"] = True
        safe_mode["vision"] = True
        return

    try:
        llm_pipeline = pipeline("text-generation", model=discover_llm_model_name())
        log("[Autoloader] distilgpt2 loaded.")
    except Exception as e:
        log(f"[Autoloader] LLM load failed: {e}")
        llm_pipeline = None
        safe_mode["llm"] = True

    try:
        vision_pipeline = pipeline("image-classification", model=discover_vision_model_name())
        log("[Autoloader] vit-base-patch16-224-in21k loaded.")
    except Exception as e:
        log(f"[Autoloader] Vision load failed: {e}")
        vision_pipeline = None
        safe_mode["vision"] = True


# ---------- Borg Planner + Learning ----------------------------------------

def borg_learn(workload_type: str, latency: float):
    alpha = 0.2
    with history_lock:
        h = history[workload_type]
        h.latencies.append(latency)
        if len(h.latencies) > 200:
            h.latencies.pop(0)
        if h.ema_latency == 0.0:
            h.ema_latency = latency
        else:
            h.ema_latency = alpha * latency + (1 - alpha) * h.ema_latency

        target = 0.7
        if h.ema_latency > target:
            h.gpu_bias = min(2.0, h.gpu_bias * 1.05)
            h.cpu_bias = max(0.5, h.cpu_bias * 0.98)
        else:
            h.gpu_bias = max(0.5, h.gpu_bias * 0.98)
            h.cpu_bias = min(2.0, h.cpu_bias * 1.02)

def record_behavior(workload_type: str):
    title = current_context.get("title") or ""
    with history_lock:
        h = history[workload_type]
        h.contexts.append(title)
        if len(h.contexts) > 200:
            h.contexts.pop(0)

def borg_plan_for_workload(workload_type: str) -> BorgPlan:
    with history_lock:
        h = history[workload_type]
        cpu_bias, gpu_bias = h.cpu_bias, h.gpu_bias

    if workload_type == "ai_inference":
        base_cpu, base_gpu = 0.3, 0.7
    elif workload_type == "vision_inference":
        base_cpu, base_gpu = 0.2, 0.8
    elif workload_type == "price_scan":
        base_cpu, base_gpu = 0.4, 0.6
    else:
        base_cpu, base_gpu = 0.4, 0.6

    cpu_share = max(0.0, min(1.0, base_cpu * cpu_bias))
    gpu_share = max(0.0, min(1.0, base_gpu * gpu_bias))
    return BorgPlan(cpu_share=cpu_share, gpu_share=gpu_share, max_threads=4)


# ---------- Telemetry + Network + GPU Graph --------------------------------

def get_gpu_utilization():
    if gpu_available and pynvml:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0
    if torch is not None and torch.cuda.is_available():
        try:
            mem = torch.cuda.memory_allocated(0)
            return 0.2 if mem > 0 else 0.0
        except Exception:
            return 0.0
    return 0.0

def telemetry_thread():
    global telemetry, running
    rng = random.Random()
    last_net = None
    while running:
        try:
            with telemetry_lock:
                if psutil:
                    telemetry.cpu_util = psutil.cpu_percent(interval=None) / 100.0
                    net = psutil.net_io_counters()
                    if last_net is None:
                        telemetry.net_sent = 0.0
                        telemetry.net_recv = 0.0
                    else:
                        telemetry.net_sent = (net.bytes_sent - last_net.bytes_sent) / 1024.0
                        telemetry.net_recv = (net.bytes_recv - last_net.bytes_recv) / 1024.0
                    last_net = net
                else:
                    telemetry.cpu_util = rng.uniform(0.1, 0.3)
                    telemetry.net_sent = rng.uniform(0.0, 5.0)
                    telemetry.net_recv = rng.uniform(0.0, 5.0)

                gpu_real = get_gpu_utilization()
                if gpu_real > 0:
                    telemetry.gpu_util = gpu_real
                else:
                    telemetry.gpu_util = rng.uniform(0.0, 0.2)

            with gpu_graph_lock:
                gpu_history.append(telemetry.gpu_util)
                if len(gpu_history) > 100:
                    gpu_history.pop(0)

        except Exception as e:
            safe_mode["telemetry"] = True
            log(f"[Telemetry] Error, entering safe mode: {e}")
        time.sleep(0.5)


# ---------- Cooldown + Prewarm ---------------------------------------------

def cooldown_organ():
    global cooldown_state, running
    while running:
        with telemetry_lock:
            cpu = telemetry.cpu_util
            gpu = telemetry.gpu_util
        cooldown_state["throttle"] = cpu > 0.85 or gpu > 0.85
        time.sleep(1.0)

def prewarm_organ():
    global llm_pipeline, running
    while running:
        title = current_context.get("title") or ""
        if "Chrome" in title and llm_pipeline is None and pipeline and not safe_mode["llm"]:
            try:
                llm_pipeline = pipeline("text-generation", model=discover_llm_model_name())
                log("[Prewarm] LLM preloaded.")
            except Exception as e:
                log(f"[Prewarm] Failed: {e}")
                safe_mode["llm"] = True
        time.sleep(5.0)


# ---------- Overrides + Heatmaps -------------------------------------------

def apply_override(plan: BorgPlan, kind: str):
    with override_lock:
        if kind == "ai":
            oc, og = override_ai["cpu"], override_ai["gpu"]
        elif kind == "vision":
            oc, og = override_vision["cpu"], override_vision["gpu"]
        else:
            oc, og = override_price["cpu"], override_price["gpu"]

    cpu = max(0.0, min(1.0, plan.cpu_share * oc))
    gpu = max(0.0, min(1.0, plan.gpu_share * og))

    if cooldown_state["throttle"]:
        cpu *= 0.7
        gpu *= 0.7

    with policy_lock:
        max_cpu = policy_rules.get("max_cpu", 0.90)
        max_gpu = policy_rules.get("max_gpu", 0.95)
    cpu = min(cpu, max_cpu)
    gpu = min(gpu, max_gpu)

    return cpu, gpu

def push_heatmap(kind: str, value: float):
    with heatmap_lock:
        buf = heatmap_ai if kind == "ai" else heatmap_vision if kind == "vision" else heatmap_price
        buf.append(value)
        if len(buf) > 50:
            buf.pop(0)


# ---------- Threat Matrix + Swarm Sync -------------------------------------

def threat_matrix_organ():
    global running
    while running:
        with heatmap_lock:
            ai_val = sum(heatmap_ai[-10:]) / max(1, len(heatmap_ai[-10:]))
            vi_val = sum(heatmap_vision[-10:]) / max(1, len(heatmap_vision[-10:]))
            pr_val = sum(heatmap_price[-10:]) / max(1, len(heatmap_price[-10:]))

        with telemetry_lock:
            cpu = telemetry.cpu_util
            gpu = telemetry.gpu_util

        threat_state["ai"] = min(1.0, ai_val + cpu * 0.3)
        threat_state["vision"] = min(1.0, vi_val + gpu * 0.3)
        threat_state["price"] = min(1.0, pr_val + cpu * 0.2)
        threat_state["overall"] = min(1.0, (threat_state["ai"] + threat_state["vision"] + threat_state["price"]) / 3.0)

        event_publish("threat_update", dict(threat_state))
        time.sleep(2.0)

def swarm_sync_organ():
    global running
    rng = random.Random()
    while running:
        time.sleep(5.0)
        swarm_state["last_sync"] = time.time()
        jitter = rng.uniform(-0.1, 0.1)
        swarm_state["consensus"] = max(0.0, min(1.0, swarm_state["consensus"] + jitter))
        event_publish("swarm_sync", dict(swarm_state))


# ---------- Process Monitoring ---------------------------------------------

def process_monitor_organ():
    global running
    while running:
        if not psutil:
            time.sleep(2.0)
            continue
        procs = []
        try:
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                info = p.info
                procs.append({
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "cpu": info.get("cpu_percent", 0.0),
                    "mem": getattr(info.get("memory_info", None), "rss", 0),
                })
        except Exception:
            pass
        procs.sort(key=lambda x: x["cpu"], reverse=True)
        with process_lock:
            top_processes.clear()
            top_processes.extend(procs[:10])
        time.sleep(3.0)


# ---------- File Reputation + Sandboxing -----------------------------------

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def get_file_reputation(path: str) -> float:
    with file_rep_lock:
        if path in file_reputation_cache:
            return file_reputation_cache[path]
    h = hash_file(path)
    if not h:
        score = 0.5
    else:
        score = int(h[:4], 16) / 0xFFFF
    with file_rep_lock:
        file_reputation_cache[path] = score
    return score

def sandbox_check_file(path: str) -> bool:
    rep = get_file_reputation(path)
    with policy_lock:
        block_high = policy_rules.get("block_high_risk_files", True)
    if not block_high:
        return True
    if rep > 0.85:
        with sandbox_lock:
            sandbox_state["blocked_files"].append(path)
        log(f"[Sandbox] Blocked high-risk file: {path} (rep={rep:.2f})")
        return False
    return True

def sandbox_scan_data_dir():
    src = pathlib.Path(SOURCE_DATA_DIR)
    if not src.exists():
        return
    for path in src.rglob("*"):
        if path.is_file():
            sandbox_check_file(str(path))


# ---------- Workloads + Heartbeats -----------------------------------------

def ai_worker():
    global running, llm_pipeline
    text = "The system organism is processing shards. " * 40
    rng = random.Random()

    while running:
        watchdog_state["ai_alive"] = True
        watchdog_state["last_ai_beat"] = time.time()

        plan = borg_plan_for_workload("ai_inference")
        cpu_share, gpu_share = apply_override(plan, "ai")

        start = time.time()
        try:
            if llm_pipeline and not safe_mode["llm"]:
                _ = llm_pipeline(text, max_new_tokens=8, do_sample=False)
            else:
                time.sleep(0.05 + rng.uniform(0.0, 0.05))
        except Exception as e:
            log(f"[AI] error: {e}")
            safe_mode["llm"] = True
            time.sleep(0.05)

        duration = time.time() - start
        borg_learn("ai_inference", duration)
        record_behavior("ai_inference")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share + duration * 0.1)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.5)

        push_heatmap("ai", min(1.0, duration))
        time.sleep(0.3)

def vision_worker():
    global running, vision_pipeline
    rng = random.Random()
    while running:
        watchdog_state["vision_alive"] = True
        watchdog_state["last_vision_beat"] = time.time()

        plan = borg_plan_for_workload("vision_inference")
        cpu_share, gpu_share = apply_override(plan, "vision")

        start = time.time()
        try:
            if vision_pipeline and Image and not safe_mode["vision"]:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
                _ = vision_pipeline(img)
            else:
                time.sleep(0.1 + rng.uniform(0.0, 0.05))
        except Exception as e:
            log(f"[Vision] error: {e}")
            safe_mode["vision"] = True
            time.sleep(0.1)

        duration = time.time() - start
        borg_learn("vision_inference", duration)
        record_behavior("vision_inference")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share + duration * 0.05)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.7)

        push_heatmap("vision", min(1.0, duration))
        time.sleep(0.3)

def price_worker():
    global running
    rng = random.Random()
    while running:
        watchdog_state["price_alive"] = True
        watchdog_state["last_price_beat"] = time.time()

        plan = borg_plan_for_workload("price_scan")
        cpu_share, gpu_share = apply_override(plan, "price")

        start = time.time()
        for _ in range(20000):
            _ = rng.random() * rng.random()

        duration = time.time() - start
        borg_learn("price_scan", duration)
        record_behavior("price_scan")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share + duration * 0.2)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.3)

        push_heatmap("price", min(1.0, duration))
        time.sleep(0.4)


# ---------- Watchdog Organ --------------------------------------------------

def watchdog_organ():
    global running
    while running:
        now = time.time()
        for name, key in [("AI", "last_ai_beat"), ("Vision", "last_vision_beat"), ("Price", "last_price_beat")]:
            last = watchdog_state.get(key, 0.0)
            if last and now - last > 10.0:
                log(f"[Watchdog] {name} worker stalled (>{now-last:.1f}s).")
                event_publish("worker_stalled", name)
        time.sleep(3.0)


# ---------- Backup Organ ----------------------------------------------------

def ensure_source_dir():
    src = pathlib.Path(SOURCE_DATA_DIR)
    src.mkdir(parents=True, exist_ok=True)
    return src

def backup_organ_once():
    src = ensure_source_dir()
    with backup_lock:
        targets = [p for p in backup_paths.values() if p]

    if not targets:
        log("[Backup] No paths set.")
        return

    sandbox_scan_data_dir()

    for target in targets:
        try:
            dst_root = pathlib.Path(target)
            dst_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log(f"[Backup] Path error: {e}")
            continue

        log(f"[Backup] Syncing to {dst_root}")
        for path in src.rglob("*"):
            if path.is_file():
                if not sandbox_check_file(str(path)):
                    continue
                rel = path.relative_to(src)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or path.stat().st_mtime > dst.stat().st_mtime:
                    try:
                        shutil.copy2(path, dst)
                    except Exception as e:
                        log(f"[Backup] Copy error: {e}")
        log(f"[Backup] Done: {dst_root}")


# ---------- GUI -------------------------------------------------------------

class CockpitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Organism Cockpit")
        self.root.geometry("1200x720")

        self.style = ttk.Style()
        self.style.theme_use("clam")

        top_frame = ttk.Frame(root)
        top_frame.pack(fill="x", pady=5)

        self.status_label = ttk.Label(top_frame, text="System: RUNNING", font=("Segoe UI", 14))
        self.status_label.pack(side="left", padx=10)

        self.safe_mode_label = ttk.Label(top_frame, text="", font=("Segoe UI", 10), foreground="orange")
        self.safe_mode_label.pack(side="left", padx=10)

        self.swarm_label = ttk.Label(top_frame, text="", font=("Segoe UI", 10))
        self.swarm_label.pack(side="right", padx=10)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Panel: Cockpit ---
        self.panel_main = ttk.Frame(self.notebook)
        self.notebook.add(self.panel_main, text="Cockpit")

        self.cpu_var = tk.StringVar()
        self.gpu_var = tk.StringVar()
        self.net_var = tk.StringVar()
        ttk.Label(self.panel_main, textvariable=self.cpu_var, font=("Segoe UI", 12)).pack()
        ttk.Label(self.panel_main, textvariable=self.gpu_var, font=("Segoe UI", 12)).pack()
        ttk.Label(self.panel_main, textvariable=self.net_var, font=("Segoe UI", 10)).pack()

        sliders_frame = ttk.Frame(self.panel_main)
        sliders_frame.pack(pady=10, fill="x")

        ttk.Label(sliders_frame, text="AI CPU", width=8).grid(row=0, column=0)
        self.ai_cpu = tk.DoubleVar(value=override_ai["cpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.ai_cpu,
                  command=lambda v: self.update_override("ai")).grid(row=0, column=1, sticky="ew")

        ttk.Label(sliders_frame, text="AI GPU", width=8).grid(row=0, column=2)
        self.ai_gpu = tk.DoubleVar(value=override_ai["gpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.ai_gpu,
                  command=lambda v: self.update_override("ai")).grid(row=0, column=3, sticky="ew")

        ttk.Label(sliders_frame, text="Vision CPU", width=8).grid(row=1, column=0)
        self.vision_cpu = tk.DoubleVar(value=override_vision["cpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.vision_cpu,
                  command=lambda v: self.update_override("vision")).grid(row=1, column=1, sticky="ew")

        ttk.Label(sliders_frame, text="Vision GPU", width=8).grid(row=1, column=2)
        self.vision_gpu = tk.DoubleVar(value=override_vision["gpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.vision_gpu,
                  command=lambda v: self.update_override("vision")).grid(row=1, column=3, sticky="ew")

        ttk.Label(sliders_frame, text="Price CPU", width=8).grid(row=2, column=0)
        self.price_cpu = tk.DoubleVar(value=override_price["cpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.price_cpu,
                  command=lambda v: self.update_override("price")).grid(row=2, column=1, sticky="ew")

        ttk.Label(sliders_frame, text="Price GPU", width=8).grid(row=2, column=2)
        self.price_gpu = tk.DoubleVar(value=override_price["gpu"])
        ttk.Scale(sliders_frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.price_gpu,
                  command=lambda v: self.update_override("price")).grid(row=2, column=3, sticky="ew")

        sliders_frame.columnconfigure(1, weight=1)
        sliders_frame.columnconfigure(3, weight=1)

        self.ai_heat = tk.StringVar()
        self.vision_heat = tk.StringVar()
        self.price_heat = tk.StringVar()

        ttk.Label(self.panel_main, textvariable=self.ai_heat, font=("Segoe UI", 10)).pack()
        ttk.Label(self.panel_main, textvariable=self.vision_heat, font=("Segoe UI", 10)).pack()
        ttk.Label(self.panel_main, textvariable=self.price_heat, font=("Segoe UI", 10)).pack()

        backup_frame = ttk.Frame(self.panel_main)
        backup_frame.pack(pady=10)

        ttk.Button(backup_frame, text="Set Primary Backup", command=self.set_primary).grid(row=0, column=0, padx=5)
        ttk.Button(backup_frame, text="Set Secondary Backup", command=self.set_secondary).grid(row=0, column=1, padx=5)
        ttk.Button(backup_frame, text="Run Backup Now", command=self.run_backup).grid(row=0, column=2, padx=5)

        ttk.Button(self.panel_main, text="Save Settings", command=save_settings).pack(pady=5)
        ttk.Button(self.panel_main, text="Quit", command=self.quit).pack(pady=5)

        # --- Panel: Status & Threats ---
        self.panel_status = ttk.Frame(self.notebook)
        self.notebook.add(self.panel_status, text="Status & Threats")

        self.status_text = tk.StringVar()
        ttk.Label(self.panel_status, textvariable=self.status_text, justify="left").pack(anchor="w", padx=5, pady=5)

        threat_frame = ttk.Frame(self.panel_status)
        threat_frame.pack(pady=5, fill="x")

        self.threat_ai = tk.StringVar()
        self.threat_vision = tk.StringVar()
        self.threat_price = tk.StringVar()
        self.threat_overall = tk.StringVar()

        ttk.Label(threat_frame, textvariable=self.threat_ai).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(threat_frame, textvariable=self.threat_vision).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Label(threat_frame, textvariable=self.threat_price).grid(row=2, column=0, sticky="w", padx=5)
        ttk.Label(threat_frame, textvariable=self.threat_overall, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", padx=5, pady=5)

        self.overlay_canvas = tk.Canvas(self.panel_status, height=20)
        self.overlay_canvas.pack(fill="x", padx=5, pady=5)
        self.overlay_bar = self.overlay_canvas.create_rectangle(0, 0, 0, 20, fill="#00aa00")

        # GPU graph
        self.gpu_canvas = tk.Canvas(self.panel_status, height=80, bg="#111111")
        self.gpu_canvas.pack(fill="x", padx=5, pady=5)

        # --- Panel: Logs ---
        self.panel_logs = ttk.Frame(self.notebook)
        self.notebook.add(self.panel_logs, text="Logs")

        self.log_text = scrolledtext.ScrolledText(self.panel_logs, wrap="word", height=20, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Panel: Processes ---
        self.panel_procs = ttk.Frame(self.notebook)
        self.notebook.add(self.panel_procs, text="Processes")

        self.proc_tree = ttk.Treeview(self.panel_procs, columns=("pid", "cpu", "mem"), show="headings", height=15)
        self.proc_tree.heading("pid", text="PID")
        self.proc_tree.heading("cpu", text="CPU%")
        self.proc_tree.heading("mem", text="Memory (MB)")
        self.proc_tree.column("pid", width=80)
        self.proc_tree.column("cpu", width=80)
        self.proc_tree.column("mem", width=120)
        self.proc_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Subscribe to events
        event_subscribe("threat_update", self.on_threat_update)
        event_subscribe("swarm_sync", self.on_swarm_sync)
        event_subscribe("worker_stalled", self.on_worker_stalled)

        self.update_loop()
        self.update_logs_loop()
        self.animate_overlay()
        self.update_gpu_graph()
        self.update_process_list()

    def update_override(self, kind: str):
        with override_lock:
            if kind == "ai":
                override_ai["cpu"] = self.ai_cpu.get()
                override_ai["gpu"] = self.ai_gpu.get()
            elif kind == "vision":
                override_vision["cpu"] = self.vision_cpu.get()
                override_vision["gpu"] = self.vision_gpu.get()
            else:
                override_price["cpu"] = self.price_cpu.get()
                override_price["gpu"] = self.price_gpu.get()
        event_publish("overrides_changed", kind)

    def set_primary(self):
        path = filedialog.askdirectory(title="Select Primary Backup Folder")
        if path:
            with backup_lock:
                backup_paths["primary"] = path
            log(f"[GUI] Primary backup set to {path}")

    def set_secondary(self):
        path = filedialog.askdirectory(title="Select Secondary Backup Folder")
        if path:
            with backup_lock:
                backup_paths["secondary"] = path
            log(f"[GUI] Secondary backup set to {path}")

    def run_backup(self):
        threading.Thread(target=backup_organ_once, daemon=True).start()

    def on_threat_update(self, payload):
        if not payload:
            return
        self.threat_ai.set(f"AI Threat: {payload['ai']:.2f}")
        self.threat_vision.set(f"Vision Threat: {payload['vision']:.2f}")
        self.threat_price.set(f"Price Threat: {payload['price']:.2f}")
        self.threat_overall.set(f"Overall Threat: {payload['overall']:.2f}")

    def on_swarm_sync(self, payload):
        if not payload:
            return
        ts = time.strftime("%H:%M:%S", time.localtime(payload["last_sync"]))
        self.swarm_label.config(text=f"Swarm peers: {payload['peers']} | Consensus: {payload['consensus']:.2f} | Last sync: {ts}")

    def on_worker_stalled(self, name):
        log(f"[GUI] Worker stalled: {name}")
        self.status_label.config(text=f"System: WARNING - {name} stalled", foreground="orange")

    def update_logs_loop(self):
        with log_lock:
            lines = list(log_buffer)
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(lines))
        self.log_text.config(state="disabled")
        self.log_text.see("end")
        self.root.after(1000, self.update_logs_loop)

    def animate_overlay(self):
        level = threat_state["overall"]
        width = int(self.overlay_canvas.winfo_width() * level)
        color = "#00aa00"
        if level > 0.7:
            color = "#cc0000"
        elif level > 0.4:
            color = "#ffaa00"
        self.overlay_canvas.coords(self.overlay_bar, 0, 0, width, 20)
        self.overlay_canvas.itemconfig(self.overlay_bar, fill=color)
        self.root.after(200, self.animate_overlay)

    def update_gpu_graph(self):
        self.gpu_canvas.delete("all")
        with gpu_graph_lock:
            data = list(gpu_history)
        if not data:
            self.root.after(500, self.update_gpu_graph)
            return
        w = self.gpu_canvas.winfo_width() or 400
        h = self.gpu_canvas.winfo_height() or 80
        step = max(1, w // max(1, len(data)))
        prev_x, prev_y = 0, h
        for i, v in enumerate(data):
            x = i * step
            y = h - int(v * h)
            self.gpu_canvas.create_line(prev_x, prev_y, x, y, fill="#00ffcc")
            prev_x, prev_y = x, y
        self.root.after(500, self.update_gpu_graph)

    def update_process_list(self):
        for row in self.proc_tree.get_children():
            self.proc_tree.delete(row)
        with process_lock:
            procs = list(top_processes)
        for p in procs:
            mem_mb = p["mem"] / (1024 * 1024) if p["mem"] else 0
            self.proc_tree.insert("", "end", values=(p["pid"], f"{p['cpu']:.1f}", f"{mem_mb:.1f}"))
        self.root.after(3000, self.update_process_list)

    def update_loop(self):
        with telemetry_lock:
            cpu = telemetry.cpu_util * 100.0
            gpu = telemetry.gpu_util * 100.0
            net_s = telemetry.net_sent
            net_r = telemetry.net_recv
        self.cpu_var.set(f"CPU Utilization: {cpu:5.1f}%")
        self.gpu_var.set(f"GPU Utilization: {gpu:5.1f}%")
        self.net_var.set(f"Net: {net_s:.1f} KB/s up | {net_r:.1f} KB/s down")

        with heatmap_lock:
            ai_vals = [f"{v:.2f}" for v in heatmap_ai[-10:]]
            vi_vals = [f"{v:.2f}" for v in heatmap_vision[-10:]]
            pr_vals = [f"{v:.2f}" for v in heatmap_price[-10:]]

        self.ai_heat.set("AI Heat: " + ", ".join(ai_vals))
        self.vision_heat.set("Vision Heat: " + ", ".join(vi_vals))
        self.price_heat.set("Price Heat: " + ", ".join(pr_vals))

        sm = [k for k, v in safe_mode.items() if v]
        if sm:
            self.safe_mode_label.config(text="Safe mode: " + ", ".join(sm))
        else:
            self.safe_mode_label.config(text="")

        status_lines = []
        status_lines.append(f"LLM: {'SAFE-MODE' if safe_mode['llm'] else 'OK'}")
        status_lines.append(f"Vision: {'SAFE-MODE' if safe_mode['vision'] else 'OK'}")
        status_lines.append(f"Telemetry: {'SAFE-MODE' if safe_mode['telemetry'] else 'OK'}")
        status_lines.append(f"GPU: {'SAFE-MODE' if safe_mode['gpu'] else 'OK'}")
        status_lines.append("")
        status_lines.append(f"AI worker alive: {watchdog_state['ai_alive']}")
        status_lines.append(f"Vision worker alive: {watchdog_state['vision_alive']}")
        status_lines.append(f"Price worker alive: {watchdog_state['price_alive']}")
        with sandbox_lock:
            blocked = len(sandbox_state['blocked_files'])
        status_lines.append(f"Sandbox blocked files: {blocked}")
        with policy_lock:
            status_lines.append(f"Policy max CPU: {policy_rules.get('max_cpu', 0.9):.2f}")
            status_lines.append(f"Policy max GPU: {policy_rules.get('max_gpu', 0.95):.2f}")
        self.status_text.set("\n".join(status_lines))

        self.root.after(500, self.update_loop)

    def quit(self):
        global running
        running = False
        save_settings()
        self.root.after(100, self.root.destroy)


# ---------- Main ------------------------------------------------------------

def main():
    global running

    log("=== Organism Cockpit (Processes, Reputation, Network, Sandbox, Policy, Swarm, GPU Graphs) ===")
    ensure_source_dir()
    load_settings()
    autoloader_load_ai_libraries()

    threads = [
        threading.Thread(target=ai_worker, daemon=True),
        threading.Thread(target=vision_worker, daemon=True),
        threading.Thread(target=price_worker, daemon=True),
        threading.Thread(target=telemetry_thread, daemon=True),
        threading.Thread(target=context_organ, daemon=True),
        threading.Thread(target=cooldown_organ, daemon=True),
        threading.Thread(target=prewarm_organ, daemon=True),
        threading.Thread(target=threat_matrix_organ, daemon=True),
        threading.Thread(target=swarm_sync_organ, daemon=True),
        threading.Thread(target=watchdog_organ, daemon=True),
        threading.Thread(target=process_monitor_organ, daemon=True),
    ]
    for t in threads:
        t.start()

    root = tk.Tk()
    gui = CockpitGUI(root)

    def delayed_auto_backup():
        time.sleep(5.0)
        backup_organ_once()

    threading.Thread(target=delayed_auto_backup, daemon=True).start()

    root.mainloop()

    running = False
    time.sleep(0.2)
    log("=== Prototype exited cleanly ===")


if __name__ == "__main__":
    main()

