import os
import sys
import time
import json
import threading
import platform
import logging
import subprocess
import numpy as np

# Optional extras
try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


# ========================= LOGGING =========================

logging.basicConfig(
    filename="beast.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def beast_log(msg: str):
    print(f"[BEAST] {msg}", flush=True)
    logging.info(msg)


# ========================= SAFE GPU DETECTION =========================

def safe_gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


# ========================= CONFIG / PROFILES =========================

DEFAULT_CONFIG = {
    "profile": "balanced",
    "gui": {"enabled": True},
    "organism_targets": {},
    "enable_gpu": False,
}

PROFILES = {
    "balanced": {
        "organisms": [
            {"name": "AlphaCPU", "type": "cpu", "target_util": 0.3},
            {"name": "BetaCPU", "type": "cpu", "target_util": 0.3},
            {"name": "GammaGPU", "type": "gpu", "target_util": 0.3},
        ]
    },
    "full": {
        "organisms": [
            {"name": "AlphaCPU", "type": "cpu", "target_util": 0.8},
            {"name": "BetaCPU", "type": "cpu", "target_util": 0.8},
            {"name": "GammaGPU", "type": "gpu", "target_util": 0.8},
        ]
    },
    "silent": {
        "organisms": [
            {"name": "AlphaCPU", "type": "cpu", "target_util": 0.05},
        ]
    },
}


def load_config(path="beast_config.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            if "organism_targets" not in cfg:
                cfg["organism_targets"] = {}
            if "enable_gpu" not in cfg:
                cfg["enable_gpu"] = False
            return cfg
        except Exception:
            pass
    cfg = DEFAULT_CONFIG.copy()
    cfg["organism_targets"] = {}
    return cfg


def save_config(cfg, path="beast_config.json"):
    try:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ========================= CORE WORKLOADS =========================

def cpu_vector_cycle(size: int = 300_000) -> float:
    x = np.random.rand(size).astype("float64")
    y = np.random.rand(size).astype("float64")
    s = np.sin(x)
    c = np.cos(y)
    out = s * c
    return float(out.sum())


def gpu_vector_cycle(size: int = 300_000, gpu_enabled: bool = False) -> float:
    if not gpu_enabled:
        return cpu_vector_cycle(size)

    if not safe_gpu_available():
        return cpu_vector_cycle(size)

    try:
        import cupy as cp
        x = cp.random.rand(size, dtype=cp.float64)
        y = cp.random.rand(size, dtype=cp.float64)
        s = cp.sin(x)
        c = cp.cos(y)
        out = s * c
        return float(out.sum().get())
    except Exception:
        return cpu_vector_cycle(size)


# ========================= ORGANISMS =========================

class Organism:
    def __init__(self, name: str, kind: str, target_util: float):
        self.name = name
        self.kind = kind  # "cpu" or "gpu"
        self.target_util = max(0.0, min(target_util, 1.0))  # 0–1 from slider
        self.last_util = 0.0
        self.last_temp = None

    def set_from_slider(self, v: float):
        self.target_util = max(0.0, min(v, 1.0))


# ========================= GLOBAL ENGINE =========================

class BeastEngine:
    def __init__(self, organisms, cfg):
        self.organisms = organisms
        self.cfg = cfg
        self.gpu_enabled = bool(cfg.get("enable_gpu", False))
        self.cycle = 0
        self.last_cpu_util = 0.0
        self.last_temp = None
        self.running = True

        # Hybrid governor parameters
        self.min_iters = 1
        self.max_iters = 80  # crank this up if you want more bite
        self.base_vec_size = 300_000

        self.lock = threading.Lock()

    def _measure_cpu(self):
        if not PSUTIL:
            return None, None
        util = psutil.cpu_percent(interval=None)
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for _, entries in temps.items():
                    if entries:
                        temp = entries[0].current
                        break
        except Exception:
            pass
        return util, temp

    def _compute_global_load(self) -> float:
        # Global load L in [0,1] = average of organism sliders
        if not self.organisms:
            return 0.0
        s = sum(o.target_util for o in self.organisms)
        return s / len(self.organisms)

    def _hybrid_iterations(self, L: float) -> int:
        # Hybrid absolute + additive on CPU:
        # abs_floor = 0.3 * L
        # additive = 0.7 * L
        # factor = abs_floor + additive = L
        # iterations = min_iters + factor * (max_iters - min_iters)
        factor = max(0.0, min(L, 1.0))
        iters = int(self.min_iters + factor * (self.max_iters - self.min_iters))
        return max(self.min_iters, iters)

    def worker_loop(self):
        while self.running:
            self.cycle += 1

            # 1) Compute global load from sliders
            L = self._compute_global_load()
            iters = self._hybrid_iterations(L)

            # 2) Burn CPU/GPU according to global iterations
            for _ in range(iters):
                for o in self.organisms:
                    if o.kind == "cpu":
                        cpu_vector_cycle(self.base_vec_size)
                    elif o.kind == "gpu":
                        gpu_vector_cycle(self.base_vec_size, self.gpu_enabled)

            # 3) Telemetry (display only, not used for control)
            util, temp = self._measure_cpu()
            with self.lock:
                if util is not None:
                    self.last_cpu_util = util
                self.last_temp = temp

            beast_log(
                f"Cycle {self.cycle} | L={L:.2f} | iters={iters} | "
                f"CPU={self.last_cpu_util:.1f}% | Temp={self.last_temp if self.last_temp is not None else 'N/A'}C"
            )

            # Small sleep to avoid total lockup
            time.sleep(0.2)


# ========================= GUI COCKPIT =========================

class BeastGUI:
    def __init__(self, engine: BeastEngine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.root = tk.Tk()
        self.root.title("Beast Global CPU+GPU Cockpit (Hybrid Governor)")
        self.info_label = None
        self.sliders = {}
        self.gpu_var = tk.BooleanVar(value=self.engine.gpu_enabled)
        self.update_interval_ms = 1000
        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        self.info_label = ttk.Label(frame, text="Initializing…", anchor="w", justify="left")
        self.info_label.pack(fill="x", pady=(0, 10))

        org_frame = ttk.LabelFrame(frame, text="Organism Weights (0–100%)")
        org_frame.pack(fill="x")

        for o in self.engine.organisms:
            row = ttk.Frame(org_frame)
            row.pack(fill="x", pady=2)

            name_label = ttk.Label(row, text=f"{o.name} ({o.kind})", width=18)
            name_label.pack(side="left")

            val_label = ttk.Label(row, text=f"{int(o.target_util * 100)}%", width=5)
            val_label.pack(side="right")

            slider = ttk.Scale(
                row,
                from_=0,
                to=100,
                orient="horizontal",
                command=lambda val, org=o, lab=val_label: self._on_slider(org, val, lab),
            )
            slider.set(o.target_util * 100)
            slider.pack(side="right", fill="x", expand=True, padx=5)

            self.sliders[o.name] = slider

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(10, 0))

        save_btn = ttk.Button(btn_frame, text="Save Settings", command=self._save_settings)
        save_btn.pack(side="left", padx=5)

        reset_btn = ttk.Button(btn_frame, text="Reset Weights", command=self._reset_targets)
        reset_btn.pack(side="left", padx=5)

        gpu_check = ttk.Checkbutton(
            frame,
            text="Enable GPU work (if available)",
            variable=self.gpu_var,
            command=self._toggle_gpu
        )
        gpu_check.pack(fill="x", pady=(10, 0))

    def _on_slider(self, org: Organism, val, label_widget):
        try:
            v = float(val)
        except:
            return
        label_widget.config(text=f"{int(v)}%")
        mapped = max(0.0, min(v / 100.0, 1.0))
        org.set_from_slider(mapped)
        self.cfg["organism_targets"][org.name] = mapped
        save_config(self.cfg)

    def _save_settings(self):
        for o in self.engine.organisms:
            self.cfg["organism_targets"][o.name] = o.target_util
        self.cfg["enable_gpu"] = self.engine.gpu_enabled
        save_config(self.cfg)
        beast_log("Settings saved from GUI")

    def _reset_targets(self):
        for o in self.engine.organisms:
            o.set_from_slider(0.0)
            slider = self.sliders.get(o.name)
            if slider is not None:
                slider.set(0)
        self.cfg["organism_targets"] = {}
        save_config(self.cfg)
        beast_log("Weights reset to 0 (silent)")

    def _toggle_gpu(self):
        enabled = bool(self.gpu_var.get())
        self.engine.gpu_enabled = enabled
        self.cfg["enable_gpu"] = enabled
        save_config(self.cfg)
        beast_log(f"GPU enabled set to {enabled}")

    def update(self):
        with self.engine.lock:
            util = self.engine.last_cpu_util
            temp = self.engine.last_temp

        L = self.engine._compute_global_load()
        iters = self.engine._hybrid_iterations(L)

        lines = [
            f"Cycle: {self.engine.cycle}",
            f"Global Load L: {L:.2f}",
            f"Hybrid iters: {iters}",
            f"CPU Util: {util:.1f}%" if util is not None else "CPU Util: N/A",
            f"Temp: {temp if temp is not None else 'N/A'}C",
            f"GPU Enabled: {self.engine.gpu_enabled}",
        ]
        for o in self.engine.organisms:
            lines.append(f"{o.name} [{o.kind}] weight={int(o.target_util*100)}%")

        self.info_label.config(text="\n".join(lines))
        self.root.after(self.update_interval_ms, self.update)

    def run(self):
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.engine.running = False
        self.root.destroy()


# ========================= MAIN =========================

def main():
    cfg = load_config()
    profile_name = cfg.get("profile", "balanced")
    profile = PROFILES.get(profile_name, PROFILES["balanced"])

    beast_log(f"Starting BEAST GLOBAL HYBRID | profile={profile_name}")
    beast_log(
        f"Platform: {platform.system()} | psutil={PSUTIL} | "
        f"tkinter={TK_AVAILABLE} | gpu_enabled={cfg.get('enable_gpu', False)}"
    )

    organisms = []
    for o in profile["organisms"]:
        base_target = o["target_util"]
        override = cfg["organism_targets"].get(o["name"])
        target = override if override is not None else base_target
        organisms.append(Organism(o["name"], o["type"], target))

    engine = BeastEngine(organisms, cfg)

    worker = threading.Thread(target=engine.worker_loop, daemon=True)
    worker.start()

    gui_enabled = cfg.get("gui", {}).get("enabled", True) and TK_AVAILABLE

    if gui_enabled:
        gui = BeastGUI(engine, cfg)
        gui.run()
    else:
        try:
            while engine.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            engine.running = False


if __name__ == "__main__":
    main()

