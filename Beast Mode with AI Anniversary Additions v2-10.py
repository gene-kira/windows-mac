import threading
import time
import random
import uuid
import os
import platform
import subprocess
import shutil
import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional
import tkinter as tk
from tkinter import ttk, filedialog
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

baseline_cpu = 0.0  # learned at startup


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


# ---------- Autoloader ------------------------------------------------------

def autoloader_load_ai_libraries():
    global llm_pipeline, vision_pipeline, gpu_available, gpu_handle

    print("[Autoloader] Loading tiny real models...")

    if pynvml:
        try:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_available = True
            print("[Autoloader] GPU detected.")
        except Exception as e:
            print("[Autoloader] NVML init failed:", e)
            gpu_available = False

    if pipeline is None:
        print("[Autoloader] transformers missing, running fake mode.")
        return

    try:
        llm_pipeline = pipeline("text-generation", model=discover_llm_model_name())
        print("[Autoloader] distilgpt2 loaded.")
    except Exception as e:
        print(f"[Autoloader] LLM load failed: {e}")
        llm_pipeline = None

    try:
        vision_pipeline = pipeline("image-classification", model=discover_vision_model_name())
        print("[Autoloader] vit-base-patch16-224-in21k loaded.")
    except Exception as e:
        print(f"[Autoloader] Vision load failed: {e}")
        vision_pipeline = None


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


# ---------- Telemetry + Baseline -------------------------------------------

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
    while running:
        with telemetry_lock:
            if psutil:
                telemetry.cpu_util = psutil.cpu_percent(interval=None) / 100.0
            else:
                telemetry.cpu_util = rng.uniform(0.1, 0.3)

            gpu_real = get_gpu_utilization()
            if gpu_real > 0:
                telemetry.gpu_util = gpu_real
            else:
                telemetry.gpu_util = rng.uniform(0.0, 0.2)

        time.sleep(0.5)

def measure_baseline_cpu():
    global baseline_cpu
    if not psutil:
        baseline_cpu = 0.3
        print(f"[Baseline] psutil missing, default baseline CPU={baseline_cpu*100:.1f}%")
        return
    samples = []
    print("[Baseline] Measuring baseline CPU for 3 seconds...")
    for _ in range(6):
        samples.append(psutil.cpu_percent(interval=0.5) / 100.0)
    baseline_cpu = sum(samples) / len(samples) if samples else 0.3
    print(f"[Baseline] Baseline CPU={baseline_cpu*100:.1f}%")

def dynamic_sleep(base_delay=0.2):
    with telemetry_lock:
        current = telemetry.cpu_util
    # If baseline not yet measured, just use base delay
    if baseline_cpu <= 0.0:
        return base_delay

    # Hard cap: do not allow more than +10% absolute over baseline
    # If we are already above that, slow down aggressively
    max_allowed = min(1.0, baseline_cpu + 0.10)
    if current > max_allowed:
        return base_delay + 0.6

    # If close to cap, slow down a bit
    if current > baseline_cpu + 0.05:
        return base_delay + 0.3

    # If well below baseline, we can be a bit more responsive
    if current < baseline_cpu - 0.10:
        return max(0.05, base_delay - 0.1)

    # Normal region
    return base_delay


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
        if "Chrome" in title and llm_pipeline is None and pipeline:
            try:
                llm_pipeline = pipeline("text-generation", model=discover_llm_model_name())
                print("[Prewarm] LLM preloaded.")
            except Exception:
                pass
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

    return cpu, gpu

def push_heatmap(kind: str, value: float):
    with heatmap_lock:
        buf = heatmap_ai if kind == "ai" else heatmap_vision if kind == "vision" else heatmap_price
        buf.append(value)
        if len(buf) > 50:
            buf.pop(0)


# ---------- Workloads -------------------------------------------------------

def ai_worker():
    global running, llm_pipeline
    text = "The system organism is processing shards. " * 20  # slightly lighter

    while running:
        plan = borg_plan_for_workload("ai_inference")
        cpu_share, gpu_share = apply_override(plan, "ai")

        start = time.time()
        if llm_pipeline:
            try:
                _ = llm_pipeline(text, max_new_tokens=6, do_sample=False)
            except Exception as e:
                print("[AI] error:", e)
        else:
            time.sleep(0.03)

        duration = time.time() - start
        borg_learn("ai_inference", duration)
        record_behavior("ai_inference")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share * 0.6)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.4)

        push_heatmap("ai", min(1.0, duration))
        time.sleep(dynamic_sleep(0.35))

def vision_worker():
    global running, vision_pipeline
    while running:
        plan = borg_plan_for_workload("vision_inference")
        cpu_share, gpu_share = apply_override(plan, "vision")

        start = time.time()
        if vision_pipeline and Image:
            try:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
                _ = vision_pipeline(img)
            except Exception as e:
                print("[Vision] error:", e)
        else:
            time.sleep(0.06)

        duration = time.time() - start
        borg_learn("vision_inference", duration)
        record_behavior("vision_inference")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share * 0.5)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.6)

        push_heatmap("vision", min(1.0, duration))
        time.sleep(dynamic_sleep(0.4))

def price_worker():
    global running
    rng = random.Random()
    while running:
        plan = borg_plan_for_workload("price_scan")
        cpu_share, gpu_share = apply_override(plan, "price")

        start = time.time()
        # lighter numeric load
        for _ in range(8000):
            _ = rng.random() * rng.random()

        duration = time.time() - start
        borg_learn("price_scan", duration)
        record_behavior("price_scan")

        with telemetry_lock:
            telemetry.cpu_util = max(telemetry.cpu_util, cpu_share * 0.5)
            telemetry.gpu_util = max(telemetry.gpu_util, gpu_share * 0.2)

        push_heatmap("price", min(1.0, duration))
        time.sleep(dynamic_sleep(0.45))


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
        print("[Backup] No paths set.")
        return

    for target in targets:
        try:
            dst_root = pathlib.Path(target)
            dst_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print("[Backup] Path error:", e)
            continue

        print("[Backup] Syncing to", dst_root)
        for path in src.rglob("*"):
            if path.is_file():
                rel = path.relative_to(src)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or path.stat().st_mtime > dst.stat().st_mtime:
                    try:
                        shutil.copy2(path, dst)
                    except Exception as e:
                        print("[Backup] Copy error:", e)
        print("[Backup] Done:", dst_root)


# ---------- GUI -------------------------------------------------------------

class CockpitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Organism Cockpit")
        self.root.geometry("800x520")

        ttk.Label(root, text="System: RUNNING", font=("Segoe UI", 14)).pack(pady=5)

        self.cpu_var = tk.StringVar()
        self.gpu_var = tk.StringVar()
        ttk.Label(root, textvariable=self.cpu_var, font=("Segoe UI", 12)).pack()
        ttk.Label(root, textvariable=self.gpu_var, font=("Segoe UI", 12)).pack()

        frame = ttk.Frame(root)
        frame.pack(pady=10, fill="x")

        # AI sliders
        ttk.Label(frame, text="AI CPU", width=8).grid(row=0, column=0)
        self.ai_cpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.ai_cpu,
                  command=lambda v: self.update_override("ai")).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="AI GPU", width=8).grid(row=0, column=2)
        self.ai_gpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.ai_gpu,
                  command=lambda v: self.update_override("ai")).grid(row=0, column=3, sticky="ew")

        # Vision sliders
        ttk.Label(frame, text="Vision CPU", width=8).grid(row=1, column=0)
        self.vision_cpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.vision_cpu,
                  command=lambda v: self.update_override("vision")).grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Vision GPU", width=8).grid(row=1, column=2)
        self.vision_gpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.vision_gpu,
                  command=lambda v: self.update_override("vision")).grid(row=1, column=3, sticky="ew")

        # Price sliders
        ttk.Label(frame, text="Price CPU", width=8).grid(row=2, column=0)
        self.price_cpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.price_cpu,
                  command=lambda v: self.update_override("price")).grid(row=2, column=1, sticky="ew")

        ttk.Label(frame, text="Price GPU", width=8).grid(row=2, column=2)
        self.price_gpu = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.2, to=2.0, orient="horizontal",
                  variable=self.price_gpu,
                  command=lambda v: self.update_override("price")).grid(row=2, column=3, sticky="ew")

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        # Text-based heat tracks
        self.ai_heat = tk.StringVar()
        self.vision_heat = tk.StringVar()
        self.price_heat = tk.StringVar()

        ttk.Label(root, textvariable=self.ai_heat, font=("Segoe UI", 10)).pack()
        ttk.Label(root, textvariable=self.vision_heat, font=("Segoe UI", 10)).pack()
        ttk.Label(root, textvariable=self.price_heat, font=("Segoe UI", 10)).pack()

        # Backup controls
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Set Primary Backup", command=self.set_primary).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Set Secondary Backup", command=self.set_secondary).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Run Backup Now", command=self.run_backup).grid(row=0, column=2, padx=5)

        ttk.Button(root, text="Quit", command=self.quit).pack(pady=10)

        self.update_loop()

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

    def update_loop(self):
        with telemetry_lock:
            cpu = telemetry.cpu_util * 100.0
            gpu = telemetry.gpu_util * 100.0
        self.cpu_var.set(f"CPU Utilization: {cpu:5.1f}%")
        self.gpu_var.set(f"GPU Utilization: {gpu:5.1f}%")

        with heatmap_lock:
            ai_vals = [f"{v:.2f}" for v in heatmap_ai[-10:]]
            vi_vals = [f"{v:.2f}" for v in heatmap_vision[-10:]]
            pr_vals = [f"{v:.2f}" for v in heatmap_price[-10:]]

        self.ai_heat.set("AI Heat: " + ", ".join(ai_vals))
        self.vision_heat.set("Vision Heat: " + ", ".join(vi_vals))
        self.price_heat.set("Price Heat: " + ", ".join(pr_vals))

        self.root.after(500, self.update_loop)

    def set_primary(self):
        path = filedialog.askdirectory(title="Select Primary Backup Folder")
        if path:
            with backup_lock:
                backup_paths["primary"] = path

    def set_secondary(self):
        path = filedialog.askdirectory(title="Select Secondary Backup Folder")
        if path:
            with backup_lock:
                backup_paths["secondary"] = path

    def run_backup(self):
        threading.Thread(target=backup_organ_once, daemon=True).start()

    def quit(self):
        global running
        running = False
        self.root.after(100, self.root.destroy)


# ---------- Main ------------------------------------------------------------

def main():
    global running

    print("=== Organism Prototype (Tkinter, Tiny Real Models, Auto Backup, Elevated, Fluid Governor) ===")
    ensure_source_dir()

    # Measure baseline CPU BEFORE starting workers
    measure_baseline_cpu()

    autoloader_load_ai_libraries()

    threads = [
        threading.Thread(target=ai_worker, daemon=True),
        threading.Thread(target=vision_worker, daemon=True),
        threading.Thread(target=price_worker, daemon=True),
        threading.Thread(target=telemetry_thread, daemon=True),
        threading.Thread(target=context_organ, daemon=True),
        threading.Thread(target=cooldown_organ, daemon=True),
        threading.Thread(target=prewarm_organ, daemon=True),
    ]
    for t in threads:
        t.start()

    root = tk.Tk()
    gui = CockpitGUI(root)

    # Auto-backup 5 seconds after GUI loads (non-blocking)
    def delayed_auto_backup():
        time.sleep(5.0)
        backup_organ_once()

    threading.Thread(target=delayed_auto_backup, daemon=True).start()

    root.mainloop()

    running = False
    time.sleep(0.2)
    print("=== Prototype exited cleanly ===")


if __name__ == "__main__":
    main()

