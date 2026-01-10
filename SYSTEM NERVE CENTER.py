#!/usr/bin/env python3
# ============================================================
# SYSTEM NERVE CENTER – HybridBrain + Organs + Reboot Memory
# - Multi-horizon prediction, regime detection, meta-confidence
# - Pattern memory, reinforcement, meta-states
# - Multi-engine voting Best-Guess (EWMA, trend, variance, etc.)
# - SelfIntegrityOrgan
# - Organs: DeepRam, Network, Disk, Thermal (live)
# - ONNX / Movidius stub hook
# - Tkinter GUI: Brain Cortex + Reboot Memory + micro-chart
# ============================================================

import sys
import subprocess
import threading
import time
import math
import json
import os
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple

# ------------------------ AutoLoader --------------------------

def ensure_package(pkg, alias=None):
    try:
        __import__(alias or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(alias or pkg)

ensure_package("psutil", "psutil")
import psutil  # type: ignore

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    print("Tkinter is required for GUI.", file=sys.stderr)
    sys.exit(1)

# ONNX (Movidius / NPU hook)
try:
    ensure_package("onnxruntime", "onnxruntime")
    import onnxruntime as ort  # type: ignore
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

# ------------------------ Utility -----------------------------

def utc_ts() -> float:
    return time.time()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ------------------------ Rolling windows ----------------------

class RollingWindow:
    def __init__(self, maxlen: int = 300):
        from collections import deque
        self.values = deque(maxlen=maxlen)

    def push(self, v: float):
        self.values.append(v)

    def last(self) -> float:
        return self.values[-1] if self.values else 0.0

    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    def var(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return statistics.pvariance(self.values)

    def slope(self) -> float:
        n = len(self.values)
        if n < 3:
            return 0.0
        xs = list(range(n))
        ys = list(self.values)
        xm = sum(xs) / n
        ym = sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        return num / den if den > 0 else 0.0

    def ewma(self, alpha: float) -> float:
        if not self.values:
            return 0.0
        ew = self.values[0]
        for v in list(self.values)[1:]:
            ew = alpha * v + (1 - alpha) * ew
        return ew

# ------------------------ Brain data models --------------------

@dataclass
class BestGuessState:
    value: float = 0.0
    confidence: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)  # engines
    regime: str = "stable"

@dataclass
class PatternMemoryEntry:
    pattern_id: str
    overload_count: int = 0
    stable_count: int = 0
    beast_win_count: int = 0

@dataclass
class PatternMemory:
    patterns: Dict[str, PatternMemoryEntry] = field(default_factory=dict)

    def update(self, pattern_id: str, outcome: str):
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = PatternMemoryEntry(pattern_id)
        entry = self.patterns[pattern_id]
        if outcome == "overload":
            entry.overload_count += 1
        elif outcome == "stable":
            entry.stable_count += 1
        elif outcome == "beast_win":
            entry.beast_win_count += 1

@dataclass
class JudgmentState:
    good: int = 0
    bad: int = 0
    bias: float = 0.0
    frozen: bool = False

    def confidence(self) -> float:
        total = self.good + self.bad
        return self.good / total if total else 0.0

@dataclass
class MetaState:
    name: str = "Sentinel"  # Hyper-Flow / Deep-Dream / Sentinel / Recovery-Flow
    momentum: float = 0.0   # 0..1 inertia

@dataclass
class CortexState:
    risk: float = 0.0
    opportunity: float = 0.0
    environment: str = "CALM"
    mission: str = "PROTECT"
    anticipation: str = "Baseline"

@dataclass
class CollectiveState:
    risk: float = 0.0
    density: float = 0.0
    agreement: float = 1.0
    divergence: float = 0.0

@dataclass
class IntegrityState:
    sensor_fresh: bool = True
    organs_present: bool = True
    prediction_drift: float = 0.0
    integrity_score: float = 1.0  # 0..1

@dataclass
class HybridBrain:
    best_guess: BestGuessState = field(default_factory=BestGuessState)
    judgment: JudgmentState = field(default_factory=JudgmentState)
    pattern_memory: PatternMemory = field(default_factory=PatternMemory)
    meta_state: MetaState = field(default_factory=MetaState)
    cortex: CortexState = field(default_factory=CortexState)
    collective: CollectiveState = field(default_factory=CollectiveState)
    integrity: IntegrityState = field(default_factory=IntegrityState)
    dynamic_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "risk_high": 0.7,
        "risk_critical": 0.9,
    })
    reinforcement: Dict[str, float] = field(default_factory=lambda: {
        "stability": 0.5,
        "performance": 0.5,
    })
    baseline_risk: float = 0.2
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_heatmap: Dict[str, float] = field(default_factory=dict)  # which engines contributed
    stance: str = "Balanced"  # Conservative / Balanced / Beast

# ------------------------ Organs (live data) ------------------

class DeepRamOrgan:
    def __init__(self):
        self.usage = 0.0
        self.health = 1.0
        self.risk = 0.0

    def tick(self):
        mem = psutil.virtual_memory()
        self.usage = mem.percent / 100.0
        self.risk = self.usage
        self.health = clamp01(1.0 - self.risk)

class NetworkWatcherOrgan:
    def __init__(self):
        self.bytes = 0
        self.risk = 0.0
        self.health = 1.0
        self._last = None

    def tick(self):
        c = psutil.net_io_counters()
        total = c.bytes_sent + c.bytes_recv
        if self._last is None:
            delta = 0.0
        else:
            delta = max(0.0, total - self._last)
        self._last = total
        self.bytes = delta
        self.risk = clamp01(delta / (1024 * 1024))
        self.health = clamp01(1.0 - self.risk)

class DiskOrgan:
    def __init__(self):
        self.bytes = 0
        self.risk = 0.0
        self.health = 1.0
        self._last = None

    def tick(self):
        io = psutil.disk_io_counters()
        total = io.read_bytes + io.write_bytes
        if self._last is None:
            delta = 0.0
        else:
            delta = max(0.0, total - self._last)
        self._last = total
        self.bytes = delta
        self.risk = clamp01(delta / (1024 * 1024))
        self.health = clamp01(1.0 - self.risk)

class ThermalOrgan:
    def __init__(self):
        self.temp = 40.0
        self.risk = 0.0
        self.health = 1.0

    def tick(self):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for _, entries in temps.items():
                    if entries:
                        self.temp = entries[0].current
                        break
        except Exception:
            pass
        self.risk = clamp01((self.temp - 40.0) / 40.0) if self.temp else 0.0
        self.health = clamp01(1.0 - self.risk)

# Stub organs
class GPUCacheOrgan: pass
class VRAMOrgan: pass
class AICoachOrgan: pass
class SwarmNodeOrgan: pass
class BackupEngineOrgan: pass
class Back4BloodAnalyzer:
    def __init__(self):
        self.last_metrics: Dict[str, Any] = {}

    def ingest(self, metrics: Dict[str, Any]):
        self.last_metrics = metrics  # e.g. FPS, ping, etc.

# ------------------------ Self-Integrity Organ ----------------

class SelfIntegrityOrgan:
    def __init__(self, predictor_windows: Dict[str, RollingWindow], organs: Dict[str, Any], brain: HybridBrain):
        self.predictor_windows = predictor_windows
        self.organs = organs
        self.brain = brain
        self.last_forecast: Optional[float] = None

    def tick(self):
        sensor_fresh = all(w.values for w in self.predictor_windows.values())
        organs_present = all(self.organs.values())
        drift = 0.0
        if self.last_forecast is not None:
            drift = abs(self.brain.best_guess.value - self.last_forecast)
        self.last_forecast = self.brain.best_guess.value

        score = 1.0
        if not sensor_fresh:
            score -= 0.2
        if not organs_present:
            score -= 0.3
        score -= clamp01(drift)

        self.brain.integrity.sensor_fresh = sensor_fresh
        self.brain.integrity.organs_present = organs_present
        self.brain.integrity.prediction_drift = drift
        self.brain.integrity.integrity_score = clamp01(score)

# ------------------------ Prediction engine -------------------

class PredictionEngine:
    def __init__(self, brain: HybridBrain):
        self.brain = brain
        self.cpu = RollingWindow(240)
        self.mem = RollingWindow(240)
        self.disk = RollingWindow(240)
        self.net = RollingWindow(240)
        self.onnx_session = None
        if HAS_ONNX:
            # Load your Movidius/ONNX model here if desired
            self.onnx_session = None

    def ingest_sample(self):
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        io = psutil.disk_io_counters()
        disk_total = io.read_bytes + io.write_bytes
        netc = psutil.net_io_counters()
        net_total = netc.bytes_sent + netc.bytes_recv

        if self.disk.values:
            disk_delta = max(0.0, disk_total - self.disk.last())
        else:
            disk_delta = 0.0
        if self.net.values:
            net_delta = max(0.0, net_total - self.net.last())
        else:
            net_delta = 0.0

        disk_norm = min(1.0, disk_delta / (1024 * 1024))
        net_norm = min(1.0, net_delta / (1024 * 1024))

        self.cpu.push(cpu)
        self.mem.push(mem)
        self.disk.push(disk_norm)
        self.net.push(net_norm)

    def _regime(self, mean: float, slope: float, var: float) -> str:
        if var < 0.002 and abs(slope) < 0.001:
            return "stable"
        if slope > 0.01 and var < 0.02:
            return "rising"
        if var > 0.05:
            return "chaotic"
        if slope < -0.01:
            return "cooling"
        return "stable"

    def _engine_votes(self) -> Tuple[float, Dict[str, float]]:
        if not self.cpu.values:
            return 0.0, {}

        cpu_last = self.cpu.last()
        cpu_mean = self.cpu.mean()
        cpu_var = self.cpu.var()
        cpu_slope = self.cpu.slope()
        turbulence = clamp01(math.sqrt(cpu_var) * 5.0)
        baseline = self.brain.baseline_risk

        ewma_short = self.cpu.ewma(0.5)
        ewma_med = self.cpu.ewma(0.2)
        ewma_long = self.cpu.ewma(0.05)

        dev = cpu_last - baseline

        # Each engine produces a vote ∈ [0,1]
        votes = {}
        votes["ewma_short"] = clamp01(ewma_short)
        votes["trend"] = clamp01(0.5 + cpu_slope)
        votes["variance"] = clamp01(cpu_var * 10.0)
        votes["turbulence"] = turbulence
        votes["baseline_dev"] = clamp01(abs(dev))
        votes["reinforcement"] = self.brain.reinforcement.get("stability", 0.5)

        # ONNX model vote stub
        votes["onnx_model"] = 0.5
        if self.onnx_session:
            # fill with model output
            pass

        # Weighted vote
        weights = {
            "ewma_short": 0.25,
            "trend": 0.15,
            "variance": 0.1,
            "turbulence": 0.15,
            "baseline_dev": 0.1,
            "reinforcement": 0.15,
            "onnx_model": 0.1,
        }
        total = 0.0
        wsum = 0.0
        for k, v in votes.items():
            w = weights.get(k, 0.0)
            total += w * v
            wsum += w
        best_guess = total / wsum if wsum > 0 else 0.0
        return clamp01(best_guess), votes

    def update_forecast(self):
        if not self.cpu.values:
            return

        best_val, components = self._engine_votes()
        cpu_var = self.cpu.var()
        cpu_slope = self.cpu.slope()
        turbulence = clamp01(math.sqrt(cpu_var) * 5.0)
        regime = self._regime(self.cpu.mean(), cpu_slope, cpu_var)

        # Meta-confidence: low variance & slope & turbulence
        v_term = 1.0 - clamp01(cpu_var / 0.05)
        s_term = 1.0 - clamp01(abs(cpu_slope) / 0.05)
        t_term = 1.0 - turbulence
        r_term = self.brain.reinforcement.get("stability", 0.5)
        meta_conf = clamp01(0.35 * v_term + 0.2 * s_term + 0.25 * t_term + 0.2 * r_term)

        bg = self.brain.best_guess
        bg.value = best_val
        bg.confidence = meta_conf
        bg.components = components
        bg.regime = regime

        # Cortex risk/opportunity
        self.brain.cortex.risk = clamp01(best_val * (1.0 + turbulence))
        self.brain.cortex.opportunity = clamp01(1.0 - self.brain.cortex.risk)
        if self.brain.cortex.risk > 0.8:
            self.brain.cortex.environment = "DANGER"
            self.brain.cortex.anticipation = "Preemptive dampening"
        elif self.brain.cortex.risk > 0.5:
            self.brain.cortex.environment = "TENSE"
            self.brain.cortex.anticipation = "Watch spikes"
        else:
            self.brain.cortex.environment = "CALM"
            self.brain.cortex.anticipation = "Normal"

        # Collective
        self.brain.collective.risk = self.brain.cortex.risk

        # Record history for micro-chart
        self.brain.prediction_history.append({
            "ts": utc_ts(),
            "best": best_val,
            "turbulence": turbulence,
            "regime": regime,
        })
        if len(self.brain.prediction_history) > 300:
            self.brain.prediction_history.pop(0)

        # Reasoning heatmap (normalize component weights)
        max_comp = max(components.values()) if components else 1.0
        self.brain.reasoning_heatmap = {
            k: (v / max_comp if max_comp else 0.0) for k, v in components.items()
        }

# ------------------------ Nerve Center app --------------------

class NerveCenterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("System Nerve Center – HybridBrain")
        self.geometry("1200x720")

        self.brain = HybridBrain()
        self.predictor = PredictionEngine(self.brain)

        # Organs
        self.organs = {
            "deep_ram": DeepRamOrgan(),
            "network": NetworkWatcherOrgan(),
            "disk": DiskOrgan(),
            "thermal": ThermalOrgan(),
            "gpu_cache": GPUCacheOrgan(),
            "vram": VRAMOrgan(),
            "ai_coach": AICoachOrgan(),
            "swarm": SwarmNodeOrgan(),
            "backup": BackupEngineOrgan(),
        }
        self.back4blood = Back4BloodAnalyzer()

        self.integrity_organ = SelfIntegrityOrgan(
            predictor_windows={"cpu": self.predictor.cpu, "mem": self.predictor.mem},
            organs=self.organs,
            brain=self.brain,
        )

        # Reboot memory
        self.reboot_memory_path: str = ""
        self.reboot_autoload: bool = False

        self._build_gui()
        self._load_settings()
        if self.reboot_autoload:
            self._try_load_reboot_memory()

        self._start_background()
        self._schedule_gui_refresh()

    # ---------------- GUI build ----------------

    def _build_gui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_cortex = ttk.Frame(notebook)
        self.tab_reboot = ttk.Frame(notebook)
        self.tab_reason = ttk.Frame(notebook)

        notebook.add(self.tab_cortex, text="Brain Cortex")
        notebook.add(self.tab_reboot, text="Reboot Memory")
        notebook.add(self.tab_reason, text="Reasoning Heatmap")

        self._build_cortex_tab(self.tab_cortex)
        self._build_reboot_tab(self.tab_reboot)
        self._build_reason_tab(self.tab_reason)

    def _build_cortex_tab(self, parent):
        frame_top = ttk.LabelFrame(parent, text="Brain State")
        frame_top.pack(fill="x", padx=10, pady=5)

        self.lbl_meta_state = ttk.Label(frame_top, text="Meta-state: Sentinel")
        self.lbl_meta_state.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.lbl_stance = ttk.Label(frame_top, text="Stance: Balanced")
        self.lbl_stance.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        self.lbl_best = ttk.Label(frame_top, text="Best-Guess: 0.00 (conf 0.00)")
        self.lbl_best.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.lbl_env = ttk.Label(frame_top, text="Env: CALM | Risk: 0.00 | Integrity: 1.00")
        self.lbl_env.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        frame_chart = ttk.LabelFrame(parent, text="Prediction Micro-chart")
        frame_chart.pack(fill="x", padx=10, pady=10)
        self.canvas_chart = tk.Canvas(frame_chart, bg="#111111", height=180)
        self.canvas_chart.pack(fill="x", padx=5, pady=5)

    def _build_reboot_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Reboot Memory Persistence")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="SMB / UNC Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame, text="Pick SMB Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame, text="Test SMB Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame,
            text="Load memory from SMB on startup",
            variable=self.var_reboot_autoload,
            command=self.cmd_toggle_autoload,
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame, text="Status: Ready", anchor="w", fg="#00cc66")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    def _build_reason_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Reasoning Heatmap")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_heatmap = tk.Text(frame, bg="#050505", fg="#d0d0d0", font=("Consolas", 9))
        self.txt_heatmap.pack(fill="both", expand=True, padx=5, pady=5)

    # ---------------- Reboot memory ----------------

    def _get_reboot_files(self) -> Tuple[str, str]:
        base = self.reboot_memory_path or self.entry_reboot_path.get().strip()
        if not base:
            base = os.path.join(os.path.expanduser("~"), "nerve_center_memory.json")
        if base.endswith(".json"):
            state_path = base
            meta_path = base.replace(".json", "_meta.json")
        else:
            state_path = os.path.join(base, "nerve_center_state.json")
            meta_path = os.path.join(base, "nerve_center_meta.json")
        return state_path, meta_path

    def cmd_pick_reboot_path(self):
        path = filedialog.asksaveasfilename(
            title="Pick reboot memory file or path",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)
            self.reboot_memory_path = path
            self._set_status("Status: Path selected", ok=True)

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self._set_status("Status: No path set", ok=False)
            return
        directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
        if os.path.exists(directory):
            self._set_status("Status: Path OK", ok=True)
        else:
            self._set_status("Status: Path does not exist", ok=False)

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            path = os.path.join(os.path.expanduser("~"), "nerve_center_memory.json")
            self.entry_reboot_path.insert(0, path)
        self.reboot_memory_path = path

        state_path, meta_path = self._get_reboot_files()
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
        except Exception:
            pass

        state = {
            "brain": asdict(self.brain),
            "organs": {k: self._organ_state(v) for k, v in self.organs.items()},
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        meta = {"ts": time.time(), "version": "1.0"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._set_status(f"Status: Memory saved to {state_path}", ok=True)
        self._save_settings()

    def _organ_state(self, organ):
        return {k: v for k, v in organ.__dict__.items() if not callable(v)}

    def _try_load_reboot_memory(self):
        state_path, _ = self._get_reboot_files()
        if not os.path.exists(state_path):
            self._set_status("Status: No memory file found", ok=False)
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # brain
            brain_data = data.get("brain", {})
            # naive restore: only a subset for safety
            self.brain.baseline_risk = brain_data.get("baseline_risk", 0.2)
            self.brain.dynamic_thresholds = brain_data.get("dynamic_thresholds", self.brain.dynamic_thresholds)
            self.brain.reinforcement = brain_data.get("reinforcement", self.brain.reinforcement)
            self.brain.pattern_memory = PatternMemory(
                patterns={
                    k: PatternMemoryEntry(**v) for k, v in brain_data.get("pattern_memory", {}).get("patterns", {}).items()
                }
            )
            self._set_status(f"Status: Memory loaded from {state_path}", ok=True)
        except Exception as e:
            self._set_status(f"Status: Load failed: {e}", ok=False)

    def cmd_toggle_autoload(self):
        self.reboot_autoload = self.var_reboot_autoload.get()
        self._set_status(f"Status: Autoload {'enabled' if self.reboot_autoload else 'disabled'}", ok=True)
        self._save_settings()

    def _set_status(self, text: str, ok: bool = True):
        self.lbl_reboot_status.config(text=text, fg="#00cc66" if ok else "#ff5555")

    def _save_settings(self):
        s = {
            "reboot_autoload": self.reboot_autoload,
            "reboot_path": self.entry_reboot_path.get().strip(),
        }
        path = os.path.join(os.path.expanduser("~"), ".nerve_center_settings.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        path = os.path.join(os.path.expanduser("~"), ".nerve_center_settings.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.reboot_autoload = bool(s.get("reboot_autoload", False))
            self.var_reboot_autoload.set(self.reboot_autoload)
            p = s.get("reboot_path", "")
            if p:
                self.entry_reboot_path.insert(0, p)
                self.reboot_memory_path = p
        except Exception:
            pass

    # ---------------- background loops ----------------

    def _start_background(self):
        t = threading.Thread(target=self._bg_loop, daemon=True)
        t.start()

    def _bg_loop(self):
        while True:
            try:
                self.predictor.ingest_sample()
                for o in self.organs.values():
                    if hasattr(o, "tick"):
                        o.tick()
                self.predictor.update_forecast()
                self.integrity_organ.tick()
            except Exception as e:
                print(f"[BG] error: {e}", file=sys.stderr)
            time.sleep(1.0)

    def _schedule_gui_refresh(self):
        self._refresh_gui()
        self.after(500, self._schedule_gui_refresh)

    def _refresh_gui(self):
        b = self.brain
        self.lbl_meta_state.config(text=f"Meta-state: {b.meta_state.name}")
        self.lbl_stance.config(text=f"Stance: {b.stance}")
        self.lbl_best.config(text=f"Best-Guess: {b.best_guess.value:.2f} (conf {b.best_guess.confidence:.2f})")
        self.lbl_env.config(
            text=f"Env: {b.cortex.environment} | Risk: {b.cortex.risk:.2f} | Integrity: {b.integrity.integrity_score:.2f}"
        )
        self._draw_micro_chart()
        self._update_heatmap()

    def _draw_micro_chart(self):
        canvas = self.canvas_chart
        canvas.delete("all")
        w = int(canvas.winfo_width() or 800)
        h = int(canvas.winfo_height() or 180)
        ph = self.brain.prediction_history[-120:]
        if not ph:
            return

        def norm(v):
            return clamp01(v)

        vals = [norm(p["best"]) for p in ph]
        if not vals:
            return

        n = len(ph)
        x_step = max(1, w // max(1, n - 1))

        def build_points(key, scale=1.0):
            pts = []
            for i, p in enumerate(ph):
                val = norm(float(p.get(key, 0.0))) * scale
                x = i * x_step
                y = h - int(val * (h - 10)) - 5
                pts.append((x, y))
            return pts

        def line(points, color, width=1):
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Use "best" as medium and baseline as brain.baseline_risk
        baseline_val = self.brain.baseline_risk

        # Short ~ local last
        line(build_points("best"), "#00ffff", 1)  # reuse as short view

        # Medium
        line(build_points("best"), "#ffff00", 1)

        # Long/baseline: flat line
        y_base = h - int(baseline_val * (h - 10)) - 5
        canvas.create_line(0, y_base, w, y_base, fill="#888888", width=1)

        # Stance-colored medium (orange)
        line(build_points("best"), "#ff8800", 1)

        # Best-Guess explicit (magenta, thicker)
        line(build_points("best"), "#ff00ff", 2)

        canvas.create_line(0, h - 5, w, h - 5, fill="#555555")
        canvas.create_text(5, 10, text="High risk", anchor="nw", fill="#aaaaaa", font=("TkDefaultFont", 8))
        canvas.create_text(5, h - 15, text="Low risk", anchor="nw", fill="#aaaaaa", font=("TkDefaultFont", 8))

    def _update_heatmap(self):
        hm = self.brain.reasoning_heatmap
        self.txt_heatmap.delete("1.0", "end")
        if not hm:
            self.txt_heatmap.insert("end", "No reasoning data yet.\n")
            return
        self.txt_heatmap.insert("end", "Reasoning Heatmap (normalized engine influence):\n\n")
        for k, v in sorted(hm.items(), key=lambda kv: kv[1], reverse=True):
            bar = "#" * int(v * 20)
            self.txt_heatmap.insert("end", f"{k:15s} [{bar:<20s}] {v:.2f}\n")

# ---------------- main ----------------

if __name__ == "__main__":
    app = NerveCenterApp()
    app.mainloop()

