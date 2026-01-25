"""
Complete System Optimizer – Borg Assimilation Console
Event Horizon / Altered States – Dreaming, Predictive, Fully Layered
Cross‑Platform (Windows / macOS / Linux)

Layers:
- Stable baseline: CPU-only NPU, no GPU, no workers, no threads
- GPU toggle in GUI (user-controlled, safe CuPy import, warm-start)
- Multi-core worker pool (no GUI calls from workers)
- PowerManager thread (no GUI calls, only CoreManager)
- VRAM detection
- Deep NPU–scheduler integration (predictive Beast Mode + worker scaling)
- NPU Dream Cycles (self-training when idle)
- Predictive Storage Routing (NPU hints for drive preference)
- Neural Telemetry Panel (plasticity/integrity/confidence waves)
- Assimilation Log (scrolling event log)

If you want CUDA:
  1. Install NVIDIA drivers
  2. Install the CUDA Toolkit (e.g. 12.x)
  3. Install a matching CuPy wheel, e.g.:
       pip install cupy-cuda12x
Then use the GUI toggle "Enable GPU" to activate it.
"""

import sys
import os
import platform
import json
import math
import random
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Any

import psutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QPushButton, QTextEdit, QSlider
)
from PyQt5.QtCore import QTimer, Qt

OS_NAME = platform.system().lower()

# =========================
# GPU backend (toggle-based)
# =========================

_gpu_checked = False
_gpu_available = False
_gpu_status_message = (
    "GPU: OFFLINE – Click 'Enable GPU' after installing NVIDIA drivers, "
    "CUDA Toolkit, and a matching CuPy wheel (e.g. cupy-cuda12x)."
)
_cupy = None


def try_enable_gpu():
    """
    Called only when user presses the GUI toggle.
    """
    global _gpu_checked, _gpu_available, _gpu_status_message, _cupy
    _gpu_checked = True
    try:
        import cupy as cp
        _ = cp.zeros((1, 1))
        _cupy = cp
        _gpu_available = True
        _gpu_status_message = "GPU: ONLINE (CuPy + CUDA active)"
    except Exception as e:
        _gpu_available = False
        _cupy = None
        _gpu_status_message = (
            "GPU: FAILED TO INITIALIZE – still in CPU mode.\n"
            "Check NVIDIA drivers, CUDA Toolkit, and CuPy install.\n"
            f"Reason: {repr(e)}"
        )


def get_gpu_status_message():
    return _gpu_status_message


# =========================
# Probes
# =========================

@dataclass
class CpuInfo:
    physical_cores: int
    logical_cores: int
    per_core_load: list


@dataclass
class TopologyMap:
    cpu: CpuInfo


class CpuProbe:
    def scan(self) -> CpuInfo:
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical
        per_core = psutil.cpu_percent(percpu=True)
        return CpuInfo(
            physical_cores=physical,
            logical_cores=logical,
            per_core_load=per_core,
        )


class ProbeManager:
    def __init__(self):
        self.cpu_probe = CpuProbe()

    def scan_all(self) -> TopologyMap:
        return TopologyMap(cpu=self.cpu_probe.scan())


# =========================
# Storage manager
# =========================

class StorageManager:
    def __init__(self, state_file: str = "optimizer_storage_state.json"):
        home = os.path.expanduser("~")
        self.state_file = os.path.join(home, state_file)
        self.partitions = self._scan_partitions()
        self.usage_history = defaultdict(list)
        self.hints = {}
        self._load_state()

    def _scan_partitions(self):
        try:
            parts = psutil.disk_partitions(all=False)
        except Exception:
            return []
        result = []
        for p in parts:
            mount = p.mountpoint
            opts = (p.opts or "").lower()
            is_network = (
                "remote" in opts
                or "network" in opts
                or mount.startswith("//")
                or mount.startswith("\\\\")
            )
            result.append({"mount": mount, "is_network": is_network})
        return result

    def _get_free_space(self, mount):
        try:
            usage = psutil.disk_usage(mount)
            return usage.free
        except Exception:
            return 0

    def _candidate_drives(self):
        local_candidates = []
        smb_candidates = []
        for p in self.partitions:
            mount = p["mount"]
            if p["is_network"]:
                smb_candidates.append(mount)
            else:
                local_candidates.append(mount)
        local_candidates.sort(key=lambda m: self._get_free_space(m), reverse=True)
        smb_candidates.sort(key=lambda m: self._get_free_space(m), reverse=True)
        return local_candidates, smb_candidates

    def choose_storage_root(self, npu_hint: float | None = None):
        """
        Predictive storage routing:
        - If NPU hint is high, prefer first local drive (fastest)
        - If NPU hint is low, allow network drive to be chosen more often
        """
        local, smb = self._candidate_drives()
        if not local and not smb:
            return os.getcwd()

        if npu_hint is None:
            if local:
                return local[0]
            return smb[0]

        if npu_hint >= 0.5:
            if local:
                return local[0]
            if smb:
                return smb[0]
        else:
            if smb:
                return smb[0]
            if local:
                return local[0]
        return os.getcwd()

    def get_top_two_drives(self):
        local, smb = self._candidate_drives()
        drives = local + smb
        if not drives:
            return [None, None]
        if len(drives) == 1:
            return [drives[0], None]
        return drives[:2]

    def record_write(self, mountpoint: str, bytes_written: int):
        self.usage_history[mountpoint].append(bytes_written)

    def get_usage_stats(self):
        return {m: sum(history) for m, history in self.usage_history.items()}

    def get_efficiency_score(self):
        stats = self.get_usage_stats()
        if not stats:
            return 100
        values = list(stats.values())
        max_val = max(values)
        min_val = min(values)
        if max_val == 0:
            return 100
        imbalance = (max_val - min_val) / max_val
        score = int((1 - imbalance) * 100)
        return score

    def allocate_path_for_file(self, filename: str, subdir: str = "optimizer_data", npu_hint: float | None = None):
        root = self.choose_storage_root(npu_hint=npu_hint)
        target_dir = os.path.join(root, subdir)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            target_dir = os.getcwd()
        return os.path.join(target_dir, filename)

    def _load_state(self):
        if not os.path.exists(self.state_file):
            self.hints["gaming_read_ahead_mb"] = 100
            self.hints["beast_on_threshold"] = 85
            self.hints["beast_off_threshold"] = 50
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.usage_history = defaultdict(list, data.get("usage_history", {}))
            self.hints = data.get("hints", {})
            self.hints["gaming_read_ahead_mb"] = self.hints.get("gaming_read_ahead_mb", 100)
            self.hints["beast_on_threshold"] = data.get("beast_on_threshold", 85)
            self.hints["beast_off_threshold"] = data.get("beast_off_threshold", 50)
        except Exception:
            self.usage_history = defaultdict(list)
            self.hints = {
                "gaming_read_ahead_mb": 100,
                "beast_on_threshold": 85,
                "beast_off_threshold": 50,
            }

    def save_state(self):
        data = {
            "usage_history": dict(self.usage_history),
            "hints": self.hints,
            "beast_on_threshold": self.hints.get("beast_on_threshold", 85),
            "beast_off_threshold": self.hints.get("beast_off_threshold", 50),
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass


# =========================
# Refactored ReplicaNPU (auto‑adapting per head)
# =========================

class ReplicaNPU:
    def __init__(
        self,
        cores=8,
        frequency_ghz=1.2,
        memory_size=32,
        plasticity_decay=0.0005,
        integrity_threshold=0.4,
    ):
        self.cores = cores
        self.frequency_ghz = frequency_ghz

        self.cycles = 0
        self.energy = 0.0

        self.memory = deque(maxlen=memory_size)

        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay

        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False

        # heads[name] stores input_dim as well
        self.heads = {}
        self.symbolic_bias = {}
        self.instruction_queue = deque()

        self.gpu_enabled = False  # flipped by CoreManager when GPU is enabled

    def enable_gpu(self):
        global _gpu_available
        if _gpu_available:
            self.gpu_enabled = True

    def disable_gpu(self):
        self.gpu_enabled = False

    def schedule(self, fn, *args):
        self.instruction_queue.append((fn, args))

    def tick(self, budget=64):
        executed = 0
        while self.instruction_queue and executed < budget:
            fn, args = self.instruction_queue.popleft()
            fn(*args)
            executed += 1
        self.plasticity = max(0.1, self.plasticity - self.plasticity_decay)

    # ---------- core math ----------

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        if self.gpu_enabled and _cupy is not None and len(v1) > 0:
            x = _cupy.asarray(v1, dtype=_cupy.float32)
            y = _cupy.asarray(v2, dtype=_cupy.float32)
            self.cycles += len(v1)
            self.energy += 0.001 * len(v1)
            return float(_cupy.dot(x, y))
        if not v1:
            return 0.0

        chunk = math.ceil(len(v1) / self.cores)
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    # ---------- head management ----------

    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "input_dim": input_dim,
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=64),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _adapt_input_for_head(self, x, head):
        dim = head["input_dim"]
        if len(x) == dim:
            return list(x)
        if len(x) < dim:
            return list(x) + [0.0] * (dim - len(x))
        return list(x[:dim])

    def _predict_head(self, head, x, name):
        x_adapted = self._adapt_input_for_head(x, head)
        y = self.vector_mac(x_adapted, head["w"])
        y += head["b"]
        y += self._symbolic_modulation(name)
        head["history"].append(y)
        self.memory.append(y)
        return y

    # ---------- public API ----------

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            if name not in self.heads:
                continue
            head = self.heads[name]
            x_adapted = self._adapt_input_for_head(x, head)
            pred = self._predict_head(head, x_adapted, name)
            error = target - pred
            weighted_error = (
                error * head["risk"] * self.plasticity * self.model_integrity
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x_adapted[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    # ---------- persistence ----------

    def save_state(self, path):
        state = {
            "heads": {
                k: {
                    "input_dim": v["input_dim"],
                    "w": v["w"],
                    "b": v["b"],
                    "lr": v["lr"],
                    "risk": v["risk"],
                    "organ": v["organ"],
                    "history": list(v["history"]),
                }
                for k, v in self.heads.items()
            },
            "plasticity": self.plasticity,
            "energy": self.energy,
            "cycles": self.cycles,
            "model_integrity": self.model_integrity,
            "frozen": self.frozen,
            "symbolic_bias": self.symbolic_bias,
            "gpu_enabled": self.gpu_enabled,
        }
        try:
            with open(path, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def load_state(self, path):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except Exception:
            return
        self.heads = {}
        for k, v in state.get("heads", {}).items():
            input_dim = v.get("input_dim", len(v.get("w", [])))
            self.heads[k] = {
                "input_dim": input_dim,
                "w": v["w"],
                "b": v["b"],
                "lr": v["lr"],
                "risk": v["risk"],
                "organ": v["organ"],
                "history": deque(v["history"], maxlen=64),
            }
        self.plasticity = state.get("plasticity", 1.0)
        self.energy = state.get("energy", 0.0)
        self.cycles = state.get("cycles", 0)
        self.model_integrity = state.get("model_integrity", 1.0)
        self.frozen = state.get("frozen", False)
        self.symbolic_bias = state.get("symbolic_bias", {})
        self.gpu_enabled = False  # re-enabled only via toggle

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {
                k: round(self.confidence(k), 3) for k in self.heads
            },
        }


# =========================
# CPU worker pool
# =========================

class CpuDronePool:
    def __init__(self, reserved_cores: int = 2):
        total = psutil.cpu_count(logical=False) or 1
        if total > 1:
            self.reserved_cores = min(reserved_cores, total - 1)
        else:
            self.reserved_cores = 0
        self.total_cores = total
        self.workers = max(1, self.total_cores - self.reserved_cores)
        self.pool = ProcessPoolExecutor(max_workers=self.workers)
        self.lock = threading.Lock()

    def submit(self, func: Callable[..., Any], *args, **kwargs):
        with self.lock:
            return self.pool.submit(func, *args, **kwargs)

    def resize(self, new_workers: int):
        with self.lock:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.workers = max(1, new_workers)
            self.pool = ProcessPoolExecutor(max_workers=self.workers)

    def shutdown(self):
        with self.lock:
            try:
                self.pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass


# =========================
# PowerManager
# =========================

class PowerManager:
    def __init__(
        self,
        core_manager: "CoreManager",
        cpu_pool: CpuDronePool,
        min_workers=1,
        max_workers=None,
        check_interval=2.0,
        beast_on_threshold=85,
        beast_off_threshold=50,
        beast_sustain_checks=1,  # more responsive
    ):
        self.core_manager = core_manager
        self.cpu_pool = cpu_pool
        self.min_workers = min_workers
        self.max_workers = max_workers or cpu_pool.workers
        self.check_interval = check_interval
        self.base_beast_on = beast_on_threshold
        self.base_beast_off = beast_off_threshold
        self.beast_sustain_checks = beast_sustain_checks

        self._running = False
        self._thread: threading.Thread | None = None
        self._high_load_count = 0
        self._low_load_count = 0

    def _loop(self):
        while self._running:
            try:
                load = psutil.cpu_percent(interval=None)

                npu_stats = self.core_manager.npu.stats()
                integrity = npu_stats["integrity"]
                avg_conf = 0.5
                if npu_stats["confidence"]:
                    avg_conf = sum(npu_stats["confidence"].values()) / len(npu_stats["confidence"])

                # base thresholds come from sliders (0–100)
                on_thresh = self.base_beast_on - int(10 * (avg_conf * integrity))
                off_thresh = self.base_beast_off + int(10 * (1 - avg_conf * integrity))

                # clamp only to 0–100 so sliders are respected
                on_thresh = max(0, min(100, on_thresh))
                off_thresh = max(0, min(100, off_thresh))

                if not self.core_manager.beast_mode:
                    if load >= on_thresh:
                        self._high_load_count += 1
                        if self._high_load_count >= self.beast_sustain_checks:
                            self.core_manager.enter_beast_mode()
                            self._high_load_count = 0
                    else:
                        self._high_load_count = 0
                else:
                    if load <= off_thresh:
                        self._low_load_count += 1
                        if self._low_load_count >= self.beast_sustain_checks:
                            self.core_manager.exit_beast_mode()
                            self._low_load_count = 0
                    else:
                        self._low_load_count = 0

                if not self.core_manager.beast_mode:
                    if load < 40 and self.cpu_pool.workers > self.min_workers:
                        self.cpu_pool.resize(self.cpu_pool.workers - 1)
                    elif load > 80 and self.cpu_pool.workers < self.max_workers:
                        self.cpu_pool.resize(self.cpu_pool.workers + 1)

            except Exception:
                pass

            time.sleep(self.check_interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# =========================
# CoreManager
# =========================

class CoreManager:
    def __init__(self, default_reserved_cores: int = 2):
        self.probes = ProbeManager()
        self.topology = self.probes.scan_all()

        self.storage = StorageManager()

        # load beast thresholds from storage
        on_t = self.storage.hints.get("beast_on_threshold", 85)
        off_t = self.storage.hints.get("beast_off_threshold", 50)

        self.beast_mode = False

        self.cpu_pool = CpuDronePool(reserved_cores=default_reserved_cores)
        self.default_reserved_cores = default_reserved_cores

        # Assimilation log buffer
        self.log_buffer = deque(maxlen=200)

        # NPU + state
        self.npu_state_path = self.storage.allocate_path_for_file(
            "replica_npu_state.json", subdir="optimizer_npu"
        )
        self.npu = ReplicaNPU(cores=8, frequency_ghz=1.2)
        self.npu.add_head("cpu_short", 3, lr=0.03, risk=1.2, organ="cortex")
        self.npu.add_head("cpu_long", 3, lr=0.01, risk=0.8, organ="planner")
        self.npu.add_head("storage_pref", 2, lr=0.02, risk=1.0, organ="io")
        self.npu.load_state(self.npu_state_path)

        self.power_manager = PowerManager(
            core_manager=self,
            cpu_pool=self.cpu_pool,
            min_workers=1,
            max_workers=self.cpu_pool.workers,
            check_interval=2.0,
            beast_on_threshold=on_t,
            beast_off_threshold=off_t,
        )
        self.power_manager.start()

        self._recent_loads = deque(maxlen=16)
        self._telemetry_plasticity = deque(maxlen=60)
        self._telemetry_integrity = deque(maxlen=60)
        self._telemetry_confidence = deque(maxlen=60)

        # Dream cycle control
        self._last_dream_time = time.time()
        self._dream_interval = 15.0
        self._dream_batch = 8

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_buffer.append(line)

    def get_log_text(self) -> str:
        return "\n".join(self.log_buffer)

    def refresh_topology(self):
        self.topology = self.probes.scan_all()
        return self.topology

    def set_reserved_cores(self, reserved: int):
        total = self.cpu_pool.total_cores
        if total > 1:
            self.cpu_pool.reserved_cores = min(reserved, total - 1)
        else:
            self.cpu_pool.reserved_cores = 0
        new_workers = max(1, total - self.cpu_pool.reserved_cores)
        self.cpu_pool.resize(new_workers)
        self.log(f"Adjusted reserved cores to {self.cpu_pool.reserved_cores}, workers={self.cpu_pool.workers}")

    def enter_beast_mode(self):
        if self.beast_mode:
            return
        self.beast_mode = True
        self.set_reserved_cores(0)
        self.npu.set_symbolic_bias("cpu_short", 0.1)
        self.log("BEAST MODE ENGAGED – full assimilation")

    def exit_beast_mode(self):
        if not self.beast_mode:
            return
        self.beast_mode = False
        self.set_reserved_cores(self.default_reserved_cores)
        self.npu.set_symbolic_bias("cpu_short", 0.0)
        self.log("Beast mode disengaged – returning to adaptive")

    def get_power_plan(self):
        if OS_NAME == "windows":
            try:
                import subprocess
                output = subprocess.check_output(
                    ["powercfg", "/GETACTIVESCHEME"],
                    shell=True,
                    text=True
                )
                return output.strip()
            except Exception:
                return "Unknown"
        elif OS_NAME == "linux":
            try:
                import subprocess
                gov = subprocess.check_output(
                    "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                    shell=True,
                    text=True
                )
                return f"Governor: {gov.strip()}"
            except Exception:
                return "Unknown"
        elif OS_NAME == "darwin":
            return "macOS Balanced (default)"
        return "Unknown"

    def get_memory_stats(self):
        vm = psutil.virtual_memory()
        ram_used = vm.used
        ram_total = vm.total

        vram_used = None
        vram_total = None
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                shell=True,
                text=True
            )
            line = out.strip().splitlines()[0]
            used_str, total_str = [x.strip() for x in line.split(",")]
            vram_used = int(used_str) * 1024 * 1024
            vram_total = int(total_str) * 1024 * 1024
        except Exception:
            vram_used = None
            vram_total = None

        return ram_used, ram_total, vram_used, vram_total

    def npu_step(self, total_load, eff_score):
        self._recent_loads.append(total_load)
        avg_load = sum(self._recent_loads) / len(self._recent_loads) if self._recent_loads else total_load

        x = [
            avg_load / 100.0,
            self.cpu_pool.workers / max(1, self.cpu_pool.total_cores),
            eff_score / 100.0,
        ]
        preds = self.npu.predict(x)
        targets = {
            "cpu_short": total_load / 100.0,
            "cpu_long": (avg_load / 100.0) * 0.8 + (eff_score / 100.0) * 0.2,
        }
        self.npu.learn(x, targets)

        conf_short = self.npu.confidence("cpu_short")
        conf_long = self.npu.confidence("cpu_long")
        integrity = (conf_short + conf_long) / 2.0
        self.npu.check_integrity(external_integrity=integrity)
        self.npu.micro_recovery()

        pred_short = preds.get("cpu_short", total_load / 100.0)

        if not self.beast_mode:
            if pred_short > 0.8 and self.cpu_pool.workers < self.cpu_pool.total_cores:
                self.cpu_pool.resize(self.cpu_pool.workers + 1)
                self.log(f"NPU scaled workers up to {self.cpu_pool.workers} (pred_short={pred_short:.2f})")
            elif pred_short < 0.3 and self.cpu_pool.workers > 1:
                self.cpu_pool.resize(self.cpu_pool.workers - 1)
                self.log(f"NPU scaled workers down to {self.cpu_pool.workers} (pred_short={pred_short:.2f})")

        stats = self.npu.stats()
        self._telemetry_plasticity.append(stats["plasticity"])
        self._telemetry_integrity.append(stats["integrity"])
        if stats["confidence"]:
            avg_conf = sum(stats["confidence"].values()) / len(stats["confidence"])
        else:
            avg_conf = 0.5
        self._telemetry_confidence.append(avg_conf)

        return preds

    def get_telemetry(self):
        return (
            list(self._telemetry_plasticity),
            list(self._telemetry_integrity),
            list(self._telemetry_confidence),
        )

    def submit_cpu_task(self, func: Callable[..., Any], *args, **kwargs):
        return self.cpu_pool.submit(func, *args, **kwargs)

    def enable_gpu_for_npu(self):
        try_enable_gpu()
        if _gpu_available:
            self.npu.enable_gpu()
            self.log("GPU warm-start: NPU switched to GPU mode")
        else:
            self.npu.disable_gpu()
            self.log("GPU enable failed – staying in CPU mode")

    def dream_cycle(self):
        now = time.time()
        if now - self._last_dream_time < self._dream_interval:
            return
        self._last_dream_time = now

        load = psutil.cpu_percent(interval=None)
        if load > 25:
            return

        self.log("NPU entering dream cycle (synthetic self-training)")
        for _ in range(self._dream_batch):
            fake_load = random.uniform(0.1, 0.9)
            fake_eff = random.uniform(0.5, 1.0)
            fake_workers_ratio = random.uniform(0.2, 1.0)

            x3 = [fake_load, fake_workers_ratio, fake_eff]
            self.npu.predict(x3)
            self.npu.learn(
                x3,
                {
                    "cpu_short": fake_load,
                    "cpu_long": fake_load * 0.7 + fake_eff * 0.3,
                },
            )

            x2 = [fake_eff, fake_load]
            self.npu.predict(x2)
            self.npu.learn(x2, {"storage_pref": fake_eff})

        self.log("NPU dream cycle complete")

    def predictive_storage_hint(self) -> float:
        if "storage_pref" not in self.npu.heads:
            return 0.7
        eff = self.storage.get_efficiency_score() / 100.0
        load = psutil.cpu_percent(interval=None) / 100.0
        x = [eff, load]
        preds = self.npu.predict(x)
        val = preds.get("storage_pref", 0.7)
        return max(0.0, min(1.0, (val + 1.0) / 2.0))

    def shutdown(self):
        self.power_manager.stop()
        self.storage.save_state()
        self.npu.save_state(self.npu_state_path)
        self.cpu_pool.shutdown()
        self.log("CoreManager shutdown complete")


# =========================
# Example heavy task
# =========================

def example_heavy_task(x: int) -> int:
    s = 0
    for i in range(500_000):
        s += (x + i) % 7
    return s


# =========================
# GUI
# =========================

class AssimilationConsole(QMainWindow):
    def __init__(self, core_manager: CoreManager):
        super().__init__()
        self.core_manager = core_manager
        self.setWindowTitle("Complete System Optimizer – Borg Assimilation Console")
        self.setMinimumSize(1300, 780)
        self._init_ui()
        self._init_timers()

    def _init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #050807; }
            #TitleLabel {
                color: #7CFF7C; font-size: 22px; font-weight: bold; letter-spacing: 3px;
            }
            #SubtitleLabel {
                color: #4ACF4A; font-size: 11px; letter-spacing: 2px;
            }
            #ModeLabel, #BeastLabel {
                color: #7CFF7C; font-size: 12px;
                padding: 4px 8px; border: 1px solid #2FAF2F;
                border-radius: 4px; background-color: rgba(0, 64, 0, 120);
            }
            #Separator {
                color: #1A3F1A; background-color: #1A3F1A; max-height: 2px;
            }
            #PanelTitle {
                color: #7CFF7C; font-size: 13px; font-weight: bold;
                margin-top: 8px; margin-bottom: 4px;
            }
            #CoreLabel {
                color: #A8EFA8; font-size: 11px; min-width: 60px;
            }
            #CoreBar {
                border: 1px solid #2FAF2F; border-radius: 3px; background-color: #020403;
            }
            QProgressBar::chunk { background-color: #2FAF2F; }
            #InfoLabel { color: #A8EFA8; font-size: 11px; }
            #StatusLabel { color: #4ACF4A; font-size: 11px; }
            QPushButton {
                color: #7CFF7C; border: 1px solid #2FAF2F; border-radius: 4px;
                padding: 4px 8px; background-color: rgba(0, 32, 0, 160);
            }
            QPushButton:hover {
                background-color: rgba(0, 64, 0, 200);
            }
            QTextEdit {
                background-color: #020403; color: #A8EFA8; border: 1px solid #2FAF2F;
                font-size: 10px;
            }
        """)

        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # =========================
        # HEADER WITH DEDICATION
        # =========================

        header = QHBoxLayout()
        title_label = QLabel("COMPLETE SYSTEM OPTIMIZER")
        title_label.setObjectName("TitleLabel")

        subtitle_label = QLabel("BORG COLLECTIVE RESOURCE ASSIMILATION – ALTERED STATES / DREAM CYCLES")
        subtitle_label.setObjectName("SubtitleLabel")

        dedication_label = QLabel(
            "Thanks to all of my Steam Buds — k‑6 (aka me), Neelieodd (Emmy), "
            "El Guapo, Terrynia.\n"
            "All of these codes were game‑tested on Back 4 Blood — "
            "thank you for a great game!"
        )
        dedication_label.setObjectName("SubtitleLabel")
        dedication_label.setWordWrap(True)

        header_text = QVBoxLayout()
        header_text.addWidget(title_label)
        header_text.addWidget(subtitle_label)
        header_text.addWidget(dedication_label)

        header.addLayout(header_text)
        header.addStretch()

        self.mode_label = QLabel("MODE: ADAPTIVE")
        self.mode_label.setObjectName("ModeLabel")
        header.addWidget(self.mode_label)

        self.beast_label = QLabel("BEAST MODE: OFF")
        self.beast_label.setObjectName("BeastLabel")
        header.addWidget(self.beast_label)

        main_layout.addLayout(header)

        # =========================
        # Separator
        # =========================

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("Separator")
        main_layout.addWidget(sep)

        # =========================
        # MAIN PANELS
        # =========================

        middle = QHBoxLayout()

        # CPU PANEL
        cpu_panel = QVBoxLayout()
        cpu_title = QLabel("CPU CORE MATRIX")
        cpu_title.setObjectName("PanelTitle")
        cpu_panel.addWidget(cpu_title)

        self.core_bars = []
        cpu_info = self.core_manager.topology.cpu
        for idx in range(cpu_info.logical_cores):
            row = QHBoxLayout()
            label = QLabel(f"Core {idx}")
            label.setObjectName("CoreLabel")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setObjectName("CoreBar")
            row.addWidget(label)
            row.addWidget(bar)
            cpu_panel.addLayout(row)
            self.core_bars.append(bar)

        middle.addLayout(cpu_panel, 2)

        # CENTER PANEL
        center_panel = QVBoxLayout()

        sys_title = QLabel("SYSTEM & STORAGE")
        sys_title.setObjectName("PanelTitle")
        center_panel.addWidget(sys_title)

        self.total_cores_label = QLabel()
        self.total_cores_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.total_cores_label)

        self.reserved_cores_label = QLabel()
        self.reserved_cores_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.reserved_cores_label)

        self.active_workers_label = QLabel()
        self.active_workers_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.active_workers_label)

        self.powerplan_label = QLabel()
        self.powerplan_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.powerplan_label)

        self.drive1_label = QLabel()
        self.drive1_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.drive1_label)

        self.drive2_label = QLabel()
        self.drive2_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.drive2_label)

        self.efficiency_label = QLabel()
        self.efficiency_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.efficiency_label)

        # Beast Mode Threshold Sliders
        self.beast_slider_label = QLabel(
            f"Beast Mode Threshold: {self.core_manager.power_manager.base_beast_on}%"
        )
        self.beast_slider_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.beast_slider_label)

        self.beast_slider = QSlider(Qt.Horizontal)
        self.beast_slider.setMinimum(0)
        self.beast_slider.setMaximum(100)
        self.beast_slider.setValue(self.core_manager.power_manager.base_beast_on)
        self.beast_slider.valueChanged.connect(self.on_beast_slider_changed)
        center_panel.addWidget(self.beast_slider)

        self.beast_off_slider_label = QLabel(
            f"Beast Mode Exit Threshold: {self.core_manager.power_manager.base_beast_off}%"
        )
        self.beast_off_slider_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.beast_off_slider_label)

        self.beast_off_slider = QSlider(Qt.Horizontal)
        self.beast_off_slider.setMinimum(0)
        self.beast_off_slider.setMaximum(100)
        self.beast_off_slider.setValue(self.core_manager.power_manager.base_beast_off)
        self.beast_off_slider.valueChanged.connect(self.on_beast_off_slider_changed)
        center_panel.addWidget(self.beast_off_slider)

        mem_title = QLabel("MEMORY UTILIZATION")
        mem_title.setObjectName("PanelTitle")
        center_panel.addWidget(mem_title)

        self.ram_label = QLabel()
        self.ram_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.ram_label)

        self.vram_label = QLabel()
        self.vram_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.vram_label)

        gpu_title = QLabel("GPU STATUS")
        gpu_title.setObjectName("PanelTitle")
        center_panel.addWidget(gpu_title)

        self.gpu_status_label = QLabel(get_gpu_status_message())
        self.gpu_status_label.setObjectName("InfoLabel")
        self.gpu_status_label.setWordWrap(True)
        center_panel.addWidget(self.gpu_status_label)

        self.gpu_button = QPushButton("Enable GPU")
        self.gpu_button.clicked.connect(self.on_enable_gpu_clicked)
        center_panel.addWidget(self.gpu_button)

        npu_title = QLabel("ALTERED STATES – NPU STATUS")
        npu_title.setObjectName("PanelTitle")
        center_panel.addWidget(npu_title)

        self.npu_plasticity_label = QLabel()
        self.npu_plasticity_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.npu_plasticity_label)

        self.npu_integrity_label = QLabel()
        self.npu_integrity_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.npu_integrity_label)

        self.npu_energy_label = QLabel()
        self.npu_energy_label.setObjectName("InfoLabel")
        center_panel.addWidget(self.npu_energy_label)

        self.npu_conf_label = QLabel()
        self.npu_conf_label.setObjectName("InfoLabel")
        self.npu_conf_label.setWordWrap(True)
        center_panel.addWidget(self.npu_conf_label)

        middle.addLayout(center_panel, 2)

        # RIGHT PANEL
        right_panel = QVBoxLayout()

        tele_title = QLabel("NEURAL TELEMETRY – PLASTICITY / INTEGRITY / CONFIDENCE")
        tele_title.setObjectName("PanelTitle")
        right_panel.addWidget(tele_title)

        self.telemetry_label = QLabel()
        self.telemetry_label.setObjectName("InfoLabel")
        self.telemetry_label.setWordWrap(True)
        right_panel.addWidget(self.telemetry_label)

        log_title = QLabel("ASSIMILATION LOG")
        log_title.setObjectName("PanelTitle")
        right_panel.addWidget(log_title)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        right_panel.addWidget(self.log_view, 1)

        middle.addLayout(right_panel, 2)

        main_layout.addLayout(middle)

        # BOTTOM STATUS BAR
        bottom = QHBoxLayout()
        self.status_label = QLabel("COLLECTIVE STATUS: IDLE – AWAITING ASSIMILATION")
        self.status_label.setObjectName("StatusLabel")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        self.cpu_load_label = QLabel("CPU: 0%")
        self.cpu_load_label.setObjectName("StatusLabel")
        bottom.addWidget(self.cpu_load_label)
        main_layout.addLayout(bottom)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # =========================
    # Timers
    # =========================

    def _init_timers(self):
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start()

    # =========================
    # Utility
    # =========================

    def _format_bytes(self, b):
        if b is None:
            return "Unknown"
        units = ["B", "KB", "MB", "GB", "TB"]
        val = float(b)
        idx = 0
        while val >= 1024 and idx < len(units) - 1:
            val /= 1024.0
            idx += 1
        return f"{val:.1f} {units[idx]}"

    def on_enable_gpu_clicked(self):
        self.core_manager.enable_gpu_for_npu()
        self.gpu_status_label.setText(get_gpu_status_message())
        self._refresh_log_view()

    def on_beast_slider_changed(self, value):
        pm = self.core_manager.power_manager
        pm.base_beast_on = value
        self.beast_slider_label.setText(f"Beast Mode Threshold: {value}%")
        self.core_manager.storage.hints["beast_on_threshold"] = value
        self.core_manager.storage.save_state()
        self.core_manager.log(f"Beast Mode ON threshold set to {value}%")
        self._refresh_log_view()

    def on_beast_off_slider_changed(self, value):
        pm = self.core_manager.power_manager
        pm.base_beast_off = value
        self.beast_off_slider_label.setText(f"Beast Mode Exit Threshold: {value}%")
        self.core_manager.storage.hints["beast_off_threshold"] = value
        self.core_manager.storage.save_state()
        self.core_manager.log(f"Beast Mode OFF threshold set to {value}%")
        self._refresh_log_view()

    def _render_telemetry_wave(self, values, width=30, height=6, char="█"):
        if not values:
            return "(no data)"
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        scaled = [
            int((v - vmin) / (vmax - vmin) * (height - 1)) for v in values[-width:]
        ]
        lines = []
        for h in reversed(range(height)):
            row = []
            for s in scaled:
                row.append(char if s >= h else " ")
            lines.append("".join(row))
        return "\n".join(lines)

    def _refresh_log_view(self):
        self.log_view.setPlainText(self.core_manager.get_log_text())
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    # =========================
    # Main Update Loop
    # =========================

    def update_stats(self):
        topo = self.core_manager.refresh_topology()
        cpu_info = topo.cpu
        per_core = cpu_info.per_core_load
        for i, bar in enumerate(self.core_bars):
            if i < len(per_core):
                bar.setValue(int(per_core[i]))
            else:
                bar.setValue(0)

        total_load = psutil.cpu_percent(interval=None)
        self.cpu_load_label.setText(f"CPU: {int(total_load)}%")

        self.total_cores_label.setText(
            f"Physical Cores: {cpu_info.physical_cores}  Logical: {cpu_info.logical_cores}"
        )
        self.reserved_cores_label.setText(
            f"Reserved Cores (AI): {self.core_manager.cpu_pool.reserved_cores}"
        )
        self.active_workers_label.setText(
            f"Active Drone Workers: {self.core_manager.cpu_pool.workers}"
        )

        self.powerplan_label.setText(
            f"Power Plan: {self.core_manager.get_power_plan()}"
        )

        drive1, drive2 = self.core_manager.storage.get_top_two_drives()
        self.drive1_label.setText(f"Primary Storage Root: {drive1}")
        self.drive2_label.setText(f"Secondary Storage Root: {drive2}")

        eff = self.core_manager.storage.get_efficiency_score()
        self.efficiency_label.setText(f"Data Efficiency: {eff}%")

        ram_used, ram_total, vram_used, vram_total = self.core_manager.get_memory_stats()
        self.ram_label.setText(
            f"System RAM: {self._format_bytes(ram_used)} / {self._format_bytes(ram_total)}"
        )
        if vram_used is not None and vram_total is not None:
            self.vram_label.setText(
                f"VRAM: {self._format_bytes(vram_used)} / {self._format_bytes(vram_total)}"
            )
        else:
            self.vram_label.setText("VRAM: Unknown / Unknown")

        self.gpu_status_label.setText(get_gpu_status_message())

        self.core_manager.npu_step(total_load, eff)
        npu_stats = self.core_manager.npu.stats()
        self.npu_plasticity_label.setText(
            f"NPU Plasticity: {npu_stats['plasticity']}"
        )
        self.npu_integrity_label.setText(
            f"NPU Integrity: {npu_stats['integrity']}  Frozen: {npu_stats['frozen']}"
        )
        self.npu_energy_label.setText(
            f"NPU Energy Units: {npu_stats['energy_units']}  Cycles: {npu_stats['cycles']}"
        )

        conf = npu_stats["confidence"]
        if conf:
            avg_conf = sum(conf.values()) / len(conf)
            self.npu_conf_label.setText(
                f"NPU Confidence (avg): {avg_conf:.3f}  Heads: {conf}"
            )
        else:
            self.npu_conf_label.setText("NPU Confidence: N/A")

        plast, integ, confv = self.core_manager.get_telemetry()
        wave_p = self._render_telemetry_wave(plast, char="▇")
        wave_i = self._render_telemetry_wave(integ, char="▇")
        wave_c = self._render_telemetry_wave(confv, char="▇")
        self.telemetry_label.setText(
            f"Plasticity\n{wave_p}\n\nIntegrity\n{wave_i}\n\nConfidence\n{wave_c}"
        )

        self.core_manager.dream_cycle()

        hint = self.core_manager.predictive_storage_hint()
        if random.random() < 0.02:
            self.core_manager.log(f"Storage preference hint={hint:.2f}")

        self._refresh_log_view()

        if self.core_manager.beast_mode:
            self.beast_label.setText("BEAST MODE: ON")
            self.mode_label.setText("MODE: FULL ASSIMILATION")
        else:
            self.beast_label.setText("BEAST MODE: OFF")
            self.mode_label.setText("MODE: ADAPTIVE")

        if total_load < 20:
            self.status_label.setText("COLLECTIVE STATUS: IDLE – AWAITING ASSIMILATION")
        elif total_load < 70:
            self.status_label.setText("COLLECTIVE STATUS: OPTIMAL ASSIMILATION IN PROGRESS")
        else:
            self.status_label.setText("COLLECTIVE STATUS: HIGH LOAD – COLLECTIVE RESPONDING")


# =========================
# Main
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = CoreManager(default_reserved_cores=2)
    window = AssimilationConsole(manager)
    window.show()
    exit_code = app.exec_()
    manager.shutdown()
    sys.exit(exit_code)

