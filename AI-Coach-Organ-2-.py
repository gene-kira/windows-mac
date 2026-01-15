import math
import time
import json
import random
import sys
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

def safe_import(name, alias=None):
    try:
        module = __import__(name)
        return module if alias is None else getattr(module, alias)
    except Exception:
        return None

psutil = safe_import("psutil")
pynvml = safe_import("pynvml")
platform = safe_import("platform")

if pynvml:
    try:
        pynvml.nvmlInit()
    except Exception:
        pynvml = None

def ensure_admin():
    if os.name != "nt":
        return
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        is_admin = False
    if not is_admin:
        import ctypes
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
        sys.exit(0)

class PersistenceManager:
    def __init__(self, local_path: Path, smb_path: Path):
        self.local_path = local_path
        self.smb_path = smb_path

    def save(self, state: dict):
        data = json.dumps(state, indent=2)
        for p in (self.local_path, self.smb_path):
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(data, encoding="utf-8")
            except Exception as e:
                print(f"[WARN] Save failed for {p}: {e}")

    def load(self) -> Optional[dict]:
        for p in (self.smb_path, self.local_path):
            try:
                if p.exists():
                    return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
        return None

class ReplicaNPU:
    def __init__(self, cores=8, frequency_ghz=1.2):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.cycles = 0
        self.energy = 0.0

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        assert len(v1) == len(v2)
        chunk = max(1, math.ceil(len(v1) / self.cores))
        result = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            result += partial
        return result

    def matmul(self, A, B):
        result = [[0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                col = [B[k][j] for k in range(len(B))]
                result[i][j] = self.vector_mac(A[i], col)
        return result

    def relu(self, x):
        self.cycles += 1
        return max(0.0, x)

    def activate(self, tensor, mode="relu"):
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                if mode == "relu":
                    tensor[i][j] = self.relu(tensor[i][j])
        return tensor

@dataclass
class EWMAState:
    value: float = 0.0
    alpha: float = 0.3
    initialized: bool = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

class ForesightHelper:
    def __init__(self):
        self.ewma_short = EWMAState(alpha=0.5)
        self.ewma_med = EWMAState(alpha=0.3)
        self.ewma_long = EWMAState(alpha=0.1)
        self.history: List[float] = []
        self.regime_baseline = 0.0

    def observe(self, x: float):
        self.history.append(x)
        if len(self.history) > 512:
            self.history.pop(0)
        s = self.ewma_short.update(x)
        m = self.ewma_med.update(x)
        l = self.ewma_long.update(x)
        self.regime_baseline = l
        trend = self._estimate_trend()
        return {"short": s, "medium": m, "long": l, "baseline": self.regime_baseline, "trend": trend}

    def _estimate_trend(self, window: int = 20) -> float:
        if len(self.history) < 2:
            return 0.0
        data = self.history[-window:]
        n = len(data)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(data) / n
        num = sum((xs[i] - mean_x) * (data[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n)) or 1.0
        return num / den

class BaseOrgan:
    def __init__(self, name: str):
        self.name = name
        self.health = 1.0
        self.last_metric = 0.0

    def update(self):
        raise NotImplementedError

    def micro_recovery(self):
        self.health = min(1.0, self.health + 0.001)

class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRAM")
        self.usage_percent = 0.0

    def update(self):
        if psutil:
            vm = psutil.virtual_memory()
            self.usage_percent = vm.percent / 100.0
        else:
            self.usage_percent = random.uniform(0.2, 0.8)
        self.last_metric = self.usage_percent
        if self.usage_percent > 0.9:
            self.health -= 0.01
        else:
            self.health += 0.002
        self.health = max(0.0, min(1.0, self.health))

class BackupEngineOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("BackupEngine")
        self.last_backup_ok = True

    def update(self):
        self.last_metric = 1.0 if self.last_backup_ok else 0.0
        if not self.last_backup_ok:
            self.health -= 0.02
        else:
            self.health += 0.002
        self.health = max(0.0, min(1.0, self.health))

class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("NetworkWatcher")
        self.bytes_sent = 0
        self.bytes_recv = 0

    def update(self):
        if psutil:
            io = psutil.net_io_counters()
            self.bytes_sent = io.bytes_sent
            self.bytes_recv = io.bytes_recv
            self.last_metric = min(1.0, (self.bytes_sent + self.bytes_recv) / (1024 * 1024 * 1024))
        else:
            self.last_metric = random.uniform(0.0, 0.5)
        self.health = min(1.0, self.health + 0.001)

class GPUCacheOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("GPUCache")
        self.cache_pressure = 0.0

    def update(self):
        self.cache_pressure = random.uniform(0.1, 0.7)
        self.last_metric = self.cache_pressure
        self.health = min(1.0, self.health + 0.001)

class ThermalOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Thermal")
        self.temp_c = 40.0

    def update(self):
        if psutil and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                all_temps = [t.current for arr in temps.values() for t in arr]
                self.temp_c = sum(all_temps) / len(all_temps) if all_temps else 40.0
            except Exception:
                self.temp_c = 40.0
        else:
            self.temp_c = 35.0 + random.uniform(0, 30)
        self.last_metric = self.temp_c / 100.0
        if self.temp_c > 85:
            self.health -= 0.02
        else:
            self.health += 0.002
        self.health = max(0.0, min(1.0, self.health))

class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Disk")
        self.read_bytes = 0
        self.write_bytes = 0

    def update(self):
        if psutil:
            try:
                io = psutil.disk_io_counters()
                self.read_bytes = io.read_bytes
                self.write_bytes = io.write_bytes
                self.last_metric = min(1.0, (self.read_bytes + self.write_bytes) / (10 * 1024 * 1024 * 1024))
            except Exception:
                self.last_metric = 0.0
        else:
            self.last_metric = random.uniform(0.0, 0.5)
        self.health = min(1.0, self.health + 0.001)

class VRAMOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("VRAM")
        self.usage_percent = 0.0

    def update(self):
        if pynvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.usage_percent = mem.used / mem.total
            except Exception:
                self.usage_percent = random.uniform(0.1, 0.7)
        else:
            self.usage_percent = random.uniform(0.1, 0.7)
        self.last_metric = self.usage_percent
        if self.usage_percent > 0.9:
            self.health -= 0.01
        else:
            self.health += 0.002
        self.health = max(0.0, min(1.0, self.health))

class AICoachOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("AICoach")
        self.suggestions: List[str] = []

    def update(self):
        self.last_metric = self.health
        self.health = min(1.0, self.health + 0.001)

class SwarmNodeOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("SwarmNode")
        self.node_count = 1

    def update(self):
        self.last_metric = min(1.0, self.node_count / 10.0)
        self.health = min(1.0, self.health + 0.001)

class Back4BloodAnalyzer(BaseOrgan):
    def __init__(self):
        super().__init__("Back4BloodAnalyzer")
        self.game_running = False
        self.game_cpu = 0.0
        self.game_mem = 0.0
        self.proc = None

    def update(self):
        self.game_running = False
        self.game_cpu = 0.0
        self.game_mem = 0.0
        self.proc = None
        if psutil:
            for p in psutil.process_iter(["name", "cpu_percent", "memory_percent"]):
                name = (p.info.get("name") or "").lower()
                if "back4blood" in name or "back 4 blood" in name:
                    self.game_running = True
                    self.game_cpu = p.info.get("cpu_percent", 0.0) / 100.0
                    self.game_mem = p.info.get("memory_percent", 0.0) / 100.0
                    self.proc = p
                    break
        self.last_metric = (self.game_cpu + self.game_mem) / 2.0
        if self.game_running:
            self.health = min(1.0, self.health + 0.002)
        else:
            self.health = max(0.0, self.health - 0.001)

class PredictionBus:
    def __init__(self):
        self.current_risk: float = 0.0
        self.bottlenecks: Dict[str, float] = {}
        self.deep_ram_target: float = 0.7
        self.per_source_scaling: Dict[str, float] = {}

    def update_from_organs(self, organs: List[BaseOrgan]):
        self.bottlenecks.clear()
        self.per_source_scaling.clear()
        for o in organs:
            self.per_source_scaling[o.name] = 1.0
            if o.last_metric > 0.8:
                self.bottlenecks[o.name] = o.last_metric
        if self.bottlenecks:
            self.current_risk = min(1.0, max(self.bottlenecks.values()))
        else:
            self.current_risk = 0.2

class DecisionLog:
    def __init__(self, max_entries=200):
        self.entries: List[str] = []
        self.max_entries = max_entries

    def add(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.entries.append(line)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

class DecisionEngine:
    def __init__(self, log: DecisionLog):
        self.stance = "Balanced"
        self.log = log
        self.last_action_ts = 0.0

    def _set_process_priority(self, proc, stance: str):
        if not psutil:
            return
        try:
            if os.name == "nt":
                if stance == "Conservative":
                    proc.nice(psutil.IDLE_PRIORITY_CLASS)
                elif stance == "Balanced":
                    proc.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    proc.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                if stance == "Conservative":
                    proc.nice(10)
                elif stance == "Balanced":
                    proc.nice(0)
                else:
                    proc.nice(-5)
        except Exception as e:
            self.log.add(f"Priority change failed: {e}")

    def decide(self, brain, prediction_bus: PredictionBus, organs: List[BaseOrgan]):
        risk = prediction_bus.current_risk
        ram = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        thermal = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        vram = next((o for o in organs if isinstance(o, VRAMOrgan)), None)
        b4b = next((o for o in organs if isinstance(o, Back4BloodAnalyzer)), None)
        ram_use = ram.usage_percent if ram else 0.0
        temp_norm = (thermal.temp_c / 100.0) if thermal else 0.4
        vram_use = vram.usage_percent if vram else 0.0
        if (ram_use > 0.9 or temp_norm > 0.85 or vram_use > 0.9 or risk > 0.7):
            new_stance = "Conservative"
        elif (risk < 0.4 and temp_norm < 0.7 and ram_use < 0.8 and brain.reinforcement_score > 0.5):
            new_stance = "Beast"
        else:
            new_stance = "Balanced"
        if new_stance != self.stance:
            self.log.add(f"Stance change: {self.stance} -> {new_stance} (risk={risk:.2f}, RAM={ram_use:.2f}, T={temp_norm:.2f})")
            self.stance = new_stance
            self._apply_actions(new_stance, b4b)
        if self.stance == "Conservative":
            prediction_bus.deep_ram_target = 0.6
        elif self.stance == "Balanced":
            prediction_bus.deep_ram_target = 0.75
        else:
            prediction_bus.deep_ram_target = 0.9
        for name in prediction_bus.per_source_scaling.keys():
            if self.stance == "Conservative":
                prediction_bus.per_source_scaling[name] = 0.7
            elif self.stance == "Balanced":
                prediction_bus.per_source_scaling[name] = 1.0
            else:
                prediction_bus.per_source_scaling[name] = 1.3

    def _apply_actions(self, stance: str, b4b: Optional[Back4BloodAnalyzer]):
        now = time.time()
        if now - self.last_action_ts < 5.0:
            return
        self.last_action_ts = now
        self.log.add(f"Applying stance actions for {stance}")
        if psutil:
            self_proc = psutil.Process(os.getpid())
            self._set_process_priority(self_proc, stance)
            if b4b and b4b.proc:
                self._set_process_priority(b4b.proc, stance)

class HybridBrain:
    def __init__(self, npu: ReplicaNPU, foresight: ForesightHelper):
        self.npu = npu
        self.foresight = foresight
        self.meta_state = "Baseline"
        self.stance = "Balanced"
        self.meta_conf = 0.5
        self.model_integrity = 1.0
        self.reinforcement_score = 0.0
        self.last_predictions = {
            "short": 0.0,
            "medium": 0.0,
            "long": 0.0,
            "baseline": 0.0,
            "best_guess": 0.0,
            "meta_conf": 0.5,
        }
        self.last_reasoning: List[str] = []
        self.last_heatmap: Dict = {}
        self.fingerprint: Dict = {}
        self.mode_profiles: Dict[str, Dict] = {}

    def update(self, organs: List[BaseOrgan], decision_engine: DecisionEngine, prediction_bus: PredictionBus):
        deep_ram = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        thermal = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        disk = next((o for o in organs if isinstance(o, DiskOrgan)), None)
        vram = next((o for o in organs if isinstance(o, VRAMOrgan)), None)
        net = next((o for o in organs if isinstance(o, NetworkWatcherOrgan)), None)
        ram_load = deep_ram.usage_percent if deep_ram else 0.0
        temp_norm = (thermal.temp_c / 100.0) if thermal else 0.4
        disk_load = disk.last_metric if disk else 0.0
        vram_load = vram.usage_percent if vram else 0.0
        net_load = net.last_metric if net else 0.0
        strain = 0.35 * ram_load + 0.25 * temp_norm + 0.2 * disk_load + 0.1 * vram_load + 0.1 * net_load
        f = self.foresight.observe(strain)
        short = f["short"]
        med = f["medium"]
        long = f["long"]
        baseline = f["baseline"]
        trend = f["trend"]
        vec = [[short, med, long, baseline, trend]]
        W = [
            [0.6, 0.3, 0.1, -0.1, 0.2],
            [0.2, 0.4, 0.3, 0.0, 0.1],
        ]
        out = self.npu.matmul(vec, list(map(list, zip(*W))))
        out = self.npu.activate(out, mode="relu")
        raw_best = out[0][0]
        raw_alt = out[0][1]
        best_guess = max(0.0, min(1.0, raw_best / 2.0))
        alt_signal = max(0.0, min(1.0, raw_alt / 2.0))
        horizon_spread = max(short, med, long) - min(short, med, long)
        trend_mag = abs(trend)
        self.meta_conf = max(0.0, min(1.0, 1.0 - horizon_spread - 0.2 * trend_mag))
        if best_guess < 0.3:
            self.meta_state = "Calm Waters"
        elif best_guess < 0.6:
            self.meta_state = "Balanced Flow"
        else:
            self.meta_state = "Event Horizon"
        self.stance = decision_engine.stance
        if organs:
            self.model_integrity = sum(o.health for o in organs) / len(organs)
        else:
            self.model_integrity = 1.0
        reward = 1.0 - strain
        self.reinforcement_score = 0.9 * self.reinforcement_score + 0.1 * reward
        self.fingerprint = {
            "ram": ram_load,
            "temp": temp_norm,
            "disk": disk_load,
            "vram": vram_load,
            "net": net_load,
            "strain": strain,
        }
        self.mode_profiles[self.meta_state] = self.fingerprint.copy()
        self.last_predictions = {
            "short": short,
            "medium": med,
            "long": long,
            "baseline": baseline,
            "best_guess": best_guess,
            "meta_conf": self.meta_conf,
        }
        self.last_reasoning = [
            f"Strain={strain:.3f} (RAM={ram_load:.2f}, TEMP={temp_norm:.2f}, DISK={disk_load:.2f}, VRAM={vram_load:.2f}, NET={net_load:.2f})",
            f"EWMA short/med/long = {short:.3f}/{med:.3f}/{long:.3f}",
            f"Baseline={baseline:.3f}, Trend={trend:.4f}",
            f"NPU best_guess={best_guess:.3f}, alt_signal={alt_signal:.3f}",
            f"Meta-State={self.meta_state}, Stance={self.stance}, Meta-Conf={self.meta_conf:.3f}",
            f"Model Integrity={self.model_integrity:.3f}, Reinforcement={self.reinforcement_score:.3f}",
        ]
        self.last_heatmap = {
            "strain": strain,
            "ram_load": ram_load,
            "temp_norm": temp_norm,
            "disk_load": disk_load,
            "vram_load": vram_load,
            "net_load": net_load,
            "trend": trend,
            "best_guess_contributors": {
                "short": short,
                "medium": med,
                "long": long,
                "baseline": baseline,
                "trend": trend,
                "weights": {
                    "short": 0.6,
                    "medium": 0.3,
                    "long": 0.1,
                    "baseline": -0.1,
                    "trend": 0.2,
                },
            },
        }
        prediction_bus.current_risk = best_guess

    def to_state(self) -> dict:
        return {
            "meta_state": self.meta_state,
            "stance": self.stance,
            "meta_conf": self.meta_conf,
            "model_integrity": self.model_integrity,
            "reinforcement_score": self.reinforcement_score,
            "foresight_history": self.foresight.history,
            "ewma_short": asdict(self.foresight.ewma_short),
            "ewma_med": asdict(self.foresight.ewma_med),
            "ewma_long": asdict(self.foresight.ewma_long),
        }

    def from_state(self, state: dict):
        self.meta_state = state.get("meta_state", self.meta_state)
        self.stance = state.get("stance", self.stance)
        self.meta_conf = state.get("meta_conf", self.meta_conf)
        self.model_integrity = state.get("model_integrity", self.model_integrity)
        self.reinforcement_score = state.get("reinforcement_score", self.reinforcement_score)
        self.foresight.history = state.get("foresight_history", [])
        for name in ("ewma_short", "ewma_med", "ewma_long"):
            d = state.get(name)
            if d:
                ew = getattr(self.foresight, name)
                ew.value = d.get("value", ew.value)
                ew.alpha = d.get("alpha", ew.alpha)
                ew.initialized = d.get("initialized", ew.initialized)

class BrainCortexPanel:
    def __init__(self, root, brain: HybridBrain, organs: List[BaseOrgan],
                 prediction_bus: PredictionBus, decision_engine: DecisionEngine,
                 decision_log: DecisionLog, persistence: PersistenceManager):
        self.root = root
        self.brain = brain
        self.organs = organs
        self.prediction_bus = prediction_bus
        self.decision_engine = decision_engine
        self.decision_log = decision_log
        self.persistence = persistence
        self.local_path_var = tk.StringVar(value=str(persistence.local_path.parent.parent))
        self.smb_path_var = tk.StringVar(value=str(persistence.smb_path.parent.parent))
        self.stance_override_var = tk.StringVar(value="Auto")
        self.meta_state_override_var = tk.StringVar(value="Auto")
        self.autosave_interval_sec = 10
        self.last_autosave = 0.0
        self._build_gui()
        self._start_loop()

    def _build_gui(self):
        self.root.title("MagicBox – Event Horizon Cortex")
        self.root.configure(bg="#111111")
        top = tk.Frame(self.root, bg="#111111")
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.lbl_health = tk.Label(top, text="Health:", fg="#00ff00", bg="#111111")
        self.lbl_health.pack(side=tk.LEFT, padx=5)
        self.lbl_risk = tk.Label(top, text="Risk:", fg="#ff00ff", bg="#111111")
        self.lbl_risk.pack(side=tk.LEFT, padx=5)
        self.lbl_meta_conf = tk.Label(top, text="Meta-Conf:", fg="#ffffff", bg="#111111")
        self.lbl_meta_conf.pack(side=tk.LEFT, padx=5)
        self.lbl_integrity = tk.Label(top, text="Integrity:", fg="#ffffff", bg="#111111")
        self.lbl_integrity.pack(side=tk.LEFT, padx=5)
        self.lbl_meta_state = tk.Label(top, text="Meta-State:", fg="#00ffff", bg="#111111")
        self.lbl_meta_state.pack(side=tk.LEFT, padx=5)
        self.lbl_stance = tk.Label(top, text="Stance:", fg="#ffff66", bg="#111111")
        self.lbl_stance.pack(side=tk.LEFT, padx=5)
        mid = tk.Frame(self.root, bg="#111111")
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_chart = tk.Canvas(mid, width=400, height=200, bg="#000000", highlightthickness=0)
        self.canvas_chart.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right = tk.Frame(mid, bg="#111111")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.txt_reason = tk.Text(right, width=60, height=15, bg="#000000", fg="#ffffff")
        self.txt_reason.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_log = tk.Text(right, width=60, height=10, bg="#050505", fg="#aaaaaa")
        self.txt_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        bottom = tk.Frame(self.root, bg="#111111")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Label(bottom, text="Local Backup:", fg="#ffffff", bg="#111111").pack(side=tk.LEFT)
        self.entry_local = tk.Entry(bottom, textvariable=self.local_path_var, width=25)
        self.entry_local.pack(side=tk.LEFT, padx=3)
        btn_local = tk.Button(bottom, text="Choose Local", command=self._choose_local_folder)
        btn_local.pack(side=tk.LEFT, padx=3)
        tk.Label(bottom, text="SMB Backup:", fg="#ffffff", bg="#111111").pack(side=tk.LEFT)
        self.entry_smb = tk.Entry(bottom, textvariable=self.smb_path_var, width=25)
        self.entry_smb.pack(side=tk.LEFT, padx=3)
        btn_smb = tk.Button(bottom, text="Choose SMB", command=self._choose_smb_folder)
        btn_smb.pack(side=tk.LEFT, padx=3)
        btn_save = tk.Button(bottom, text="Save Now", command=self._save_now)
        btn_save.pack(side=tk.LEFT, padx=5)
        tk.Label(bottom, text="Stance Override:", fg="#ffffff", bg="#111111").pack(side=tk.LEFT, padx=5)
        self.cmb_stance = tk.OptionMenu(bottom, self.stance_override_var, "Auto", "Conservative", "Balanced", "Beast")
        self.cmb_stance.config(bg="#222222", fg="#ffffff")
        self.cmb_stance.pack(side=tk.LEFT)
        tk.Label(bottom, text="Meta-State Override:", fg="#ffffff", bg="#111111").pack(side=tk.LEFT, padx=5)
        self.cmb_meta = tk.OptionMenu(bottom, self.meta_state_override_var, "Auto", "Calm Waters", "Balanced Flow", "Event Horizon")
        self.cmb_meta.config(bg="#222222", fg="#ffffff")
        self.cmb_meta.pack(side=tk.LEFT)

    def _choose_local_folder(self):
        folder = filedialog.askdirectory(title="Select Local Backup Folder")
        if folder:
            self.local_path_var.set(folder)
            base = Path(folder) / "EventHorizon"
            self.persistence.local_path = base / "event_horizon_local.json"
            self.decision_log.add(f"Local backup path set to: {self.persistence.local_path}")

    def _choose_smb_folder(self):
        folder = filedialog.askdirectory(title="Select SMB / Network Backup Folder")
        if folder:
            self.smb_path_var.set(folder)
            base = Path(folder) / "EventHorizon"
            self.persistence.smb_path = base / "event_horizon_smb.json"
            self.decision_log.add(f"SMB backup path set to: {self.persistence.smb_path}")

    def _save_now(self):
        state = {"brain": self.brain.to_state()}
        self.persistence.save(state)
        self.decision_log.add("Manual Save Now triggered")

    def _start_loop(self):
        self._tick()
        self.root.after(500, self._start_loop)

    def _tick(self):
        for o in self.organs:
            o.update()
            o.micro_recovery()
        self.prediction_bus.update_from_organs(self.organs)
        if self.stance_override_var.get() != "Auto":
            self.decision_engine.stance = self.stance_override_var.get()
        else:
            self.decision_engine.decide(self.brain, self.prediction_bus, self.organs)
        if self.meta_state_override_var.get() != "Auto":
            self.brain.meta_state = self.meta_state_override_var.get()
        self.brain.update(self.organs, self.decision_engine, self.prediction_bus)
        now = time.time()
        if now - self.last_autosave > self.autosave_interval_sec:
            self.last_autosave = now
            state = {"brain": self.brain.to_state()}
            self.persistence.save(state)
            self.decision_log.add("Autosave completed")
        self._update_gui()

    def _update_gui(self):
        p = self.brain.last_predictions
        health = self.brain.model_integrity
        risk = p["best_guess"]
        self.lbl_health.config(text=f"Health: {health:.2f}")
        self.lbl_risk.config(text=f"Risk: {risk:.2f}")
        self.lbl_meta_conf.config(text=f"Meta-Conf: {p['meta_conf']:.2f}")
        self.lbl_integrity.config(text=f"Integrity: {self.brain.model_integrity:.2f}")
        self.lbl_meta_state.config(text=f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.config(text=f"Stance: {self.decision_engine.stance}")
        self._draw_chart()
        self._update_reasoning()
        self._update_log()

    def _draw_chart(self):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])
        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#111111", outline="")
        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        long = p["long"]
        baseline = p["baseline"]
        best_guess = p["best_guess"]

        def y_from_val(v):
            v = max(0.0, min(1.0, v))
            return h - int(v * (h - 10)) - 5

        x_short = w * 0.2
        x_med = w * 0.5
        x_long = w * 0.8
        y_short = y_from_val(short)
        y_med = y_from_val(med)
        y_long = y_from_val(long)
        y_base = y_from_val(baseline)
        y_best = y_from_val(best_guess)
        self.canvas_chart.create_line(0, y_base, w, y_base, fill="#555555", dash=(2, 2))
        self.canvas_chart.create_line(x_short, y_short, x_med, y_med, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_med, y_med, x_long, y_long, fill="#00ccff", width=2)
        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.decision_engine.stance, "#ffffff")
        self.canvas_chart.create_line(x_short, y_med, x_long, y_med, fill=stance_color, width=1)
        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)
        self.canvas_chart.create_text(
            5, 5, anchor="nw", fill="#aaaaaa",
            text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)"
        )

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")
        self.txt_reason.insert(tk.END, "\nHeatmap:\n")
        for k, v in self.brain.last_heatmap.items():
            if k == "best_guess_contributors":
                continue
            self.txt_reason.insert(tk.END, f"  {k}: {v}\n")
        self.txt_reason.insert(tk.END, "\nBest-Guess Contributors:\n")
        contrib = self.brain.last_heatmap.get("best_guess_contributors", {})
        for k, v in contrib.items():
            if k == "weights":
                continue
            self.txt_reason.insert(tk.END, f"  {k}: {v:.3f}\n")
        weights = contrib.get("weights", {})
        if weights:
            self.txt_reason.insert(tk.END, "\nWeights:\n")
            for k, w in weights.items():
                self.txt_reason.insert(tk.END, f"  {k}: {w:.3f}\n")

    def _update_log(self):
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.insert(tk.END, "Decision Log:\n")
        for line in self.decision_log.entries:
            self.txt_log.insert(tk.END, f"{line}\n")

def main():
    ensure_admin()
    default_local_root = Path(r"C:\EventHorizonLocal")
    default_smb_root = Path(r"\\SERVER\Share\EventHorizonSMB")
    local_path = default_local_root / "EventHorizon" / "event_horizon_local.json"
    smb_path = default_smb_root / "EventHorizon" / "event_horizon_smb.json"
    persistence = PersistenceManager(local_path, smb_path)
    npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
    foresight = ForesightHelper()
    brain = HybridBrain(npu, foresight)
    state = persistence.load()
    if state and "brain" in state:
        brain.from_state(state["brain"])
    organs: List[BaseOrgan] = [
        DeepRamOrgan(),
        BackupEngineOrgan(),
        NetworkWatcherOrgan(),
        GPUCacheOrgan(),
        ThermalOrgan(),
        DiskOrgan(),
        VRAMOrgan(),
        AICoachOrgan(),
        SwarmNodeOrgan(),
        Back4BloodAnalyzer(),
    ]
    prediction_bus = PredictionBus()
    decision_log = DecisionLog()
    decision_engine = DecisionEngine(decision_log)
    root = tk.Tk()
    panel = BrainCortexPanel(root, brain, organs, prediction_bus, decision_engine, decision_log, persistence)
    root.mainloop()

if __name__ == "__main__":
    main()

