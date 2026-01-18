#!/usr/bin/env python3
import sys
import subprocess
import importlib
import traceback
import threading
import time
import os
import random
from dataclasses import dataclass

# -------------------------------------------------------------------
# Auto-installer / autoloader
# -------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "PyQt5",
    "psutil",
    "pyyaml",
    "numpy",
    "pynvml",
    "pycuda",   # optional; will fail gracefully if CUDA not present
]

def install_package(pkg: str):
    print(f"[AUTOLOADER] Installing missing package: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure_packages(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"[AUTOLOADER] Package already available: {pkg}")
        except ImportError:
            try:
                install_package(pkg)
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install {pkg}: {e}")

def safe_import(name: str, required: bool = False):
    try:
        return importlib.import_module(name)
    except ImportError:
        try:
            install_package(name)
            return importlib.import_module(name)
        except Exception as e:
            print(f"[AUTOLOADER] Could not import {name}: {e}")
            if required:
                raise
            return None

ensure_packages(REQUIRED_PACKAGES)

from PyQt5 import QtWidgets, QtCore
psutil = safe_import("psutil")
yaml = safe_import("pyyaml")
np = safe_import("numpy")
pynvml = safe_import("pynvml")
pycuda = safe_import("pycuda")
pycuda_driver = None
pycuda_autoinit = None
if pycuda is not None:
    pycuda_driver = safe_import("pycuda.driver")
    pycuda_autoinit = safe_import("pycuda.autoinit")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def decide_ram_cache_bytes(default_bytes=2 * 1024 * 1024 * 1024):
    try:
        import psutil as _ps
        total = _ps.virtual_memory().total
        if total <= 8 * 1024 * 1024 * 1024:
            return min(default_bytes, 2 * 1024 * 1024 * 1024)
        return default_bytes
    except Exception:
        return default_bytes

def normalize(value, min_v, max_v):
    if value is None:
        return 0.0
    if max_v == min_v:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

# -------------------------------------------------------------------
# State vector
# -------------------------------------------------------------------

@dataclass
class CacheStateVector:
    temp_c: float | None = None
    power_mode: str | None = None      # "AC", "DC", or None
    gpu_load: float | None = None      # %
    disk_read_bw: float | None = None  # bytes/ms
    disk_write_bw: float | None = None # bytes/ms
    vram_frag: float = 0.0             # 0–1
    blocks: int = 0
    blocks_trend: float = 0.0
    appetite: float = 1.0

# -------------------------------------------------------------------
# Tiny sequence model (in-process "neural" predictor) + persistence
# -------------------------------------------------------------------

class TinySequenceModel:
    """
    Minimal 2-layer MLP over a flattened sequence of state features.
    Input:  seq_len x feat_dim → flattened
    Output: scalar (predicted next blocks)
    """

    def __init__(self, seq_len=32, feat_dim=5, hidden=16):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.in_dim = seq_len * feat_dim
        self.hidden = hidden

        rng = np.random.default_rng()
        self.W1 = rng.normal(0, 0.01, size=(self.in_dim, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.01, size=(hidden, 1))
        self.b2 = np.zeros(1, dtype=np.float32)

    def _forward(self, X):
        # X: (batch, in_dim)
        H = X @ self.W1 + self.b1
        H = np.maximum(H, 0.0)  # ReLU
        Y = H @ self.W2 + self.b2
        return H, Y  # Y: (batch, 1)

    def predict(self, seq):
        """
        seq: (seq_len, feat_dim)
        """
        x = np.asarray(seq, dtype=np.float32).reshape(1, -1)
        _, y = self._forward(x)
        return float(y[0, 0])

    def train_on_batch(self, X_seq, y_target, lr=1e-4):
        """
        X_seq: (batch, seq_len, feat_dim)
        y_target: (batch,)
        """
        if len(X_seq) == 0:
            return

        X = np.asarray(X_seq, dtype=np.float32).reshape(len(X_seq), -1)
        y = np.asarray(y_target, dtype=np.float32).reshape(-1, 1)

        H, y_pred = self._forward(X)
        # MSE loss
        diff = (y_pred - y)
        loss = float(np.mean(diff ** 2))

        # Backprop
        dY = 2.0 * diff / len(X)
        dW2 = H.T @ dY
        db2 = dY.sum(axis=0)

        dH = dY @ self.W2.T
        dH[H <= 0] = 0.0

        dW1 = X.T @ dH
        db1 = dH.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return loss

    def save_weights(self, path: str):
        try:
            np.savez(
                path,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
                seq_len=self.seq_len,
                feat_dim=self.feat_dim,
                hidden=self.hidden,
            )
        except Exception as e:
            print(f"[Ramnuke][MODEL] Failed to save weights: {e}")

    def load_weights(self, path: str):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            # Optional: restore dims if present
            if "seq_len" in data:
                self.seq_len = int(data["seq_len"])
            if "feat_dim" in data:
                self.feat_dim = int(data["feat_dim"])
            if "hidden" in data:
                self.hidden = int(data["hidden"])
            return True
        except Exception as e:
            print(f"[Ramnuke][MODEL] Failed to load weights: {e}")
            return False

# -------------------------------------------------------------------
# Predictors
# -------------------------------------------------------------------

class BasePredictor:
    def update(self, value: float): ...
    def predict_next(self) -> float | None: ...
    def is_anomalous(self, value: float) -> bool: ...

class TelemetryPredictor(BasePredictor):
    def __init__(self, alpha=0.3, window=50, base_sigma=3.0):
        self.alpha = alpha
        self.window = window
        self.base_sigma = base_sigma
        self.history = []
        self.ewma = None
        self.trend = 0.0
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.appetite = 1.0
        self.sigma = base_sigma

    def update(self, value: float):
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.pop(0)

        if self.ewma is None:
            self.ewma = value
        else:
            self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

        if len(self.history) >= 2:
            self.trend = (self.history[-1] - self.history[0]) / (len(self.history) - 1)
        else:
            self.trend = 0.0

        arr = np.array(self.history, dtype=float)
        self.baseline_mean = float(arr.mean())
        self.baseline_std = float(arr.std()) if len(arr) > 1 else 0.0

        self._auto_tune()

    def _auto_tune(self):
        vol = self.baseline_std
        if vol > 10:
            self.sigma = min(self.base_sigma * 2.0, 10.0)
            self.window = max(20, self.window - 5)
            self.alpha = min(0.5, self.alpha + 0.05)
        elif vol > 0:
            self.sigma = max(self.base_sigma * 0.7, 1.0)
            self.window = min(200, self.window + 2)
            self.alpha = max(0.1, self.alpha - 0.02)

        trend_mag = abs(self.trend)
        if trend_mag > 1000:
            self.appetite = min(2.0, self.appetite + 0.1)
        elif trend_mag < 100:
            self.appetite = max(0.5, self.appetite - 0.05)

    def predict_next(self):
        if self.ewma is None:
            return None
        return self.ewma + self.appetite * self.trend

    def is_anomalous(self, value: float):
        if self.baseline_std == 0:
            return False
        z = abs(value - self.baseline_mean) / self.baseline_std
        return z > self.sigma

class NeuralPredictor(BasePredictor):
    def __init__(self, model=None):
        self.model = model
        self.history = []
        self.fallback = TelemetryPredictor()

    def update(self, value: float):
        self.history.append(value)
        if len(self.history) > 128:
            self.history.pop(0)
        self.fallback.update(value)

    def predict_next(self):
        if self.model is None:
            return self.fallback.predict_next()
        seq_len = self.model.seq_len
        hist = self.history[-seq_len:]
        if len(hist) < seq_len:
            hist = [hist[0]] * (seq_len - len(hist)) + hist
        seq = np.zeros((seq_len, self.model.feat_dim), dtype=np.float32)
        for i, v in enumerate(hist):
            seq[i, 0] = v
        return self.model.predict(seq)

    def is_anomalous(self, value: float):
        return self.fallback.is_anomalous(value)

# -------------------------------------------------------------------
# VRAM cache manager
# -------------------------------------------------------------------

class VRAMCacheManager:
    VRAM_TARGET_BYTES = 8 * 1024 * 1024 * 1024
    RAM_TARGET_BYTES  = decide_ram_cache_bytes()
    BLOCK_SIZE        = 256 * 1024
    MAX_BLOCKS        = 1_000_000

    def __init__(self, logger, pycuda_driver=None, pycuda_autoinit=None):
        self.log = logger
        self.pycuda_driver = pycuda_driver
        self.pycuda_autoinit = pycuda_autoinit

        self.vram_available = False
        self.vram_buffer = None
        self.ram_buffer = None

        self.block_map = {}
        self.lru_list = []
        self.lock = threading.Lock()
        self.running = False

        self.worker_threads = []
        self.target_workers = 1
        self.vram_bias = 1.0
        self.dynamic_appetite = 1.0

        self.prev_disk_stats = None

        self.dynamic_block_size = self.BLOCK_SIZE
        self.last_block_id = None
        self.sequential_hits = 0

        self.policy_weights = {
            "temp": 0.25,
            "gpu": 0.25,
            "frag": 0.20,
            "trend": 0.15,
            "disk": 0.15,
        }
        self.policy_learning_rate = 0.01
        self.appetite_curve_sharpness = 8.0

        self.last_profile_time = 0.0
        self.profile_interval = 300.0  # seconds

        self.log("[Ramnuke][CACHE] VRAMCacheManager constructed.")

    # ---------------- Core init ----------------

    def _vram_stress_test(self):
        if not (self.pycuda_driver and self.pycuda_autoinit and np is not None):
            return False
        try:
            import pycuda.gpuarray as gpuarray
            self.log("[Ramnuke][CACHE][VRAM] Running VRAM stress test...")
            test = gpuarray.zeros(1024 * 1024, dtype=np.float32)
            test.fill(3.14159)
            host = test.get()
            if not np.allclose(host, 3.14159):
                self.log("[Ramnuke][CACHE][VRAM] Stress test FAILED: pattern mismatch.")
                return False
            self.log("[Ramnuke][CACHE][VRAM] Stress test PASSED.")
            return True
        except Exception as e:
            self.log(f"[Ramnuke][CACHE][VRAM] Stress test error: {e}")
            return False

    def initialize(self):
        self.log("[Ramnuke][CACHE] Initializing VRAM + RAM tiers...")
        self.vram_available = False

        if self.pycuda_driver and self.pycuda_autoinit and np is not None:
            try:
                import pycuda.gpuarray as gpuarray
                blocks = self.VRAM_TARGET_BYTES // self.BLOCK_SIZE
                floats_per_block = self.BLOCK_SIZE // 4
                total_floats = blocks * floats_per_block
                self.log(f"[Ramnuke][CACHE] Attempting {self.VRAM_TARGET_BYTES/1e9:.1f} GB VRAM allocation...")
                self.vram_buffer = gpuarray.zeros(total_floats, dtype=np.float32)
                if self._vram_stress_test():
                    self.vram_available = True
                    self.log("[Ramnuke][CACHE] VRAM allocation + stress test OK. VRAM tier ACTIVE.")
                else:
                    self.log("[Ramnuke][CACHE] VRAM stress test failed → RAM-only mode.")
                    self.vram_available = False
                    self.vram_buffer = None
            except Exception as e:
                self.log(f"[Ramnuke][CACHE] VRAM allocation failed → RAM-only mode.")
                self.log(f"[Ramnuke][CACHE] Reason: {e}")
                self.vram_available = False
                self.vram_buffer = None
        else:
            self.log("[Ramnuke][CACHE] No CUDA backend detected → RAM-only mode.")
            self.vram_available = False
            self.vram_buffer = None

        try:
            self.log(f"[Ramnuke][CACHE] Allocating {self.RAM_TARGET_BYTES/1e9:.1f} GB system RAM...")
            self.ram_buffer = np.zeros(self.RAM_TARGET_BYTES // 4, dtype=np.float32)
            self.log("[Ramnuke][CACHE] RAM tier ready.")
        except Exception as e:
            self.log(f"[Ramnuke][CACHE] RAM allocation failed: {e}")
            self.log(traceback.format_exc())

        if self.vram_available:
            self.log("[Ramnuke][CACHE] MODE: Hybrid VRAM + RAM")
        else:
            self.log("[Ramnuke][CACHE] MODE: RAM-ONLY (VRAM unavailable)")

    def _block_offset(self, block_id):
        return block_id * (self.BLOCK_SIZE // 4)

    # ---------------- I/O ----------------

    def update_io_pattern(self, block_id: int):
        if self.last_block_id is None:
            self.last_block_id = block_id
            return
        if block_id == self.last_block_id + 1:
            self.sequential_hits += 1
        else:
            self.sequential_hits = max(0, self.sequential_hits - 1)
        self.last_block_id = block_id

        score = max(0.0, min(1.0, self.sequential_hits / 100.0))
        self.adapt_block_size(score)

    def adapt_block_size(self, io_pattern_score: float):
        min_bs = 64 * 1024
        max_bs = 1024 * 1024
        new_bs = int(min_bs + (max_bs - min_bs) * io_pattern_score)
        if new_bs < 128 * 1024:
            new_bs = 64 * 1024
        elif new_bs < 512 * 1024:
            new_bs = 256 * 1024
        else:
            new_bs = 1024 * 1024
        self.dynamic_block_size = new_bs

    def read_block(self, block_id):
        with self.lock:
            self.update_io_pattern(block_id)

            if block_id in self.block_map:
                tier, offset = self.block_map[block_id]
                self._touch_lru(block_id)
                if tier == "VRAM" and self.vram_available:
                    return self.vram_buffer[offset : offset + (self.BLOCK_SIZE // 4)]
                else:
                    return self.ram_buffer[offset : offset + (self.BLOCK_SIZE // 4)]

            data = self._load_from_disk(block_id)
            self._insert_block(block_id, data)
            self._read_ahead(block_id)
            return data

    def _insert_block(self, block_id, data):
        offset = self._block_offset(block_id)

        if self.vram_available and self.vram_bias > 0.0:
            try:
                self.vram_buffer[offset : offset + len(data)] = data
                self.block_map[block_id] = ("VRAM", offset)
                self._touch_lru(block_id)
                return
            except Exception:
                self.log("[Ramnuke][CACHE] VRAM write failed → demoting to RAM-only.")
                self.vram_available = False
                self.vram_buffer = None

        self.ram_buffer[offset : offset + len(data)] = data
        self.block_map[block_id] = ("RAM", offset)
        self._touch_lru(block_id)

    def _evict_lru(self):
        if not self.lru_list:
            return
        victim = self.lru_list.pop(0)
        if victim in self.block_map:
            del self.block_map[victim]
        self.log(f"[Ramnuke][CACHE] Evicted block {victim} (LRU).")

    def _touch_lru(self, block_id):
        if block_id in self.lru_list:
            self.lru_list.remove(block_id)
        self.lru_list.append(block_id)

    def _read_ahead(self, block_id):
        for i in range(1, int(self.READ_AHEAD_BLOCKS) + 1):
            nb = block_id + i
            if nb not in self.block_map:
                data = self._load_from_disk(nb)
                self._insert_block(nb, data)

    def _load_from_disk(self, block_id):
        size = self.BLOCK_SIZE // 4
        return np.zeros(size, dtype=np.float32)

    # ---------------- Telemetry & environment ----------------

    def telemetry_snapshot(self):
        with self.lock:
            blocks = len(self.block_map)
            vram_blocks = sum(1 for t, _ in self.block_map.values() if t == "VRAM")
            ram_blocks = blocks - vram_blocks
        return {
            "blocks": blocks,
            "vram_blocks": vram_blocks,
            "ram_blocks": ram_blocks,
        }

    def _gpu_temperature(self):
        try:
            if pynvml:
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                pynvml.nvmlShutdown()
                return temp
        except:
            pass
        return None

    def _gpu_load(self):
        try:
            if pynvml:
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                pynvml.nvmlShutdown()
                return util.gpu
        except:
            pass
        return None

    def _power_state(self):
        try:
            if psutil:
                batt = psutil.sensors_battery()
                if batt is None:
                    return None
                return "AC" if batt.power_plugged else "DC"
        except:
            pass
        return None

    def _disk_stats(self):
        try:
            if psutil:
                d = psutil.disk_io_counters()
                return d.read_bytes, d.write_bytes, d.read_time, d.write_time
        except:
            pass
        return None

    def disk_io_physics(self, prev_stats, curr_stats):
        if not prev_stats or not curr_stats:
            return None
        rb0, wb0, rt0, wt0 = prev_stats
        rb1, wb1, rt1, wt1 = curr_stats
        d_rb = rb1 - rb0
        d_wb = wb1 - wb0
        d_rt = max(1, rt1 - rt0)
        d_wt = max(1, wt1 - wt0)
        read_bw = d_rb / d_rt
        write_bw = d_wb / d_wt
        read_lat = d_rt / max(1, d_rb)
        write_lat = d_wt / max(1, d_wb)
        return {
            "read_bw": read_bw,
            "write_bw": write_bw,
            "read_lat": read_lat,
            "write_lat": write_lat,
        }

    def vram_fragmentation_score(self):
        if not self.vram_available or not self.block_map:
            return 0.0
        vram_blocks = sum(1 for t, _ in self.block_map.values() if t == "VRAM")
        total_blocks = len(self.block_map)
        if total_blocks == 0:
            return 0.0
        pressure = vram_blocks / total_blocks
        return min(1.0, pressure)

    def _try_vram_resurrect(self):
        if self.vram_available:
            return
        if not (self.pycuda_driver and self.pycuda_autoinit and np is not None):
            return
        try:
            import pycuda.gpuarray as gpuarray
            self.log("[Ramnuke][CACHE][VRAM] Probing for VRAM resurrection...")
            test_arr = gpuarray.zeros(1024, dtype=np.float32)
            del test_arr
            self.log("[Ramnuke][CACHE][VRAM] Probe successful → reinitializing VRAM.")
            self.initialize()
        except Exception as e:
            self.log(f"[Ramnuke][CACHE][VRAM] Resurrection probe failed: {e}")

    def _cpu_cache_pressure(self):
        return None

    def _numa_locality_score(self):
        return None

    def _pcie_bandwidth_pressure(self):
        return None

    # ---------------- Policy helpers ----------------

    def fused_pressure(self, state: CacheStateVector):
        w = self.policy_weights
        return (
            w["temp"]  * normalize(state.temp_c, 40, 90) +
            w["gpu"]   * normalize(state.gpu_load, 0, 100) +
            w["frag"]  * state.vram_frag +
            w["trend"] * normalize(state.blocks_trend, -5000, 5000) +
            w["disk"]  * normalize(state.disk_read_bw or 0, 0, 50000)
        )

    def appetite_curve(self, pressure):
        k = self.appetite_curve_sharpness
        return 0.5 + 1.5 * (1 / (1 + np.exp(k * (pressure - 0.5))))

    def read_ahead_curve(self, appetite):
        return int(1 + (appetite ** 2) * 16)

    def worker_curve(self, appetite):
        if appetite < 0.8:
            return 1
        if appetite < 1.2:
            return 2
        return 4

    def vram_bias_curve(self, state: CacheStateVector):
        frag_penalty = 1.0 - state.vram_frag
        temp_penalty = 1.0 if (state.temp_c or 0) < 75 else 0.5
        gpu_penalty  = 1.0 if (state.gpu_load or 0) < 80 else 0.6
        return frag_penalty * temp_penalty * gpu_penalty

    def disk_prewarm_factor(self, state: CacheStateVector):
        if state.disk_read_bw is None:
            return 0.0
        if state.disk_read_bw > 30000:
            return 1.0
        if state.disk_read_bw > 10000:
            return 0.5
        return 0.0

    def learn_policy(self, state: CacheStateVector, pressure_before: float, pressure_after: float):
        delta = pressure_after - pressure_before
        lr = self.policy_learning_rate
        if abs(delta) < 0.01:
            return

        sign = -1 if delta > 0 else 1
        grad = {
            "temp":  normalize(state.temp_c, 40, 90),
            "gpu":   normalize(state.gpu_load, 0, 100),
            "frag":  state.vram_frag,
            "trend": normalize(state.blocks_trend, -5000, 5000),
            "disk":  normalize(state.disk_read_bw or 0, 0, 50000),
        }
        for k in self.policy_weights:
            self.policy_weights[k] += sign * lr * grad[k]

        s = sum(self.policy_weights.values())
        if s > 0:
            for k in self.policy_weights:
                self.policy_weights[k] /= s

        if delta > 0:
            self.appetite_curve_sharpness = max(2.0, self.appetite_curve_sharpness - 0.1)
        else:
            self.appetite_curve_sharpness = min(16.0, self.appetite_curve_sharpness + 0.1)

    def master_policy(self, state: CacheStateVector):
        pressure_before = self.fused_pressure(state)
        appetite = self.appetite_curve(pressure_before)
        read_ahead = self.read_ahead_curve(appetite)
        workers = self.worker_curve(appetite)
        vram_bias = self.vram_bias_curve(state)
        prewarm = self.disk_prewarm_factor(state)
        pressure_after = self.fused_pressure(state)
        self.learn_policy(state, pressure_before, pressure_after)
        return appetite, read_ahead, workers, vram_bias, prewarm

    def apply_master_policy(self, appetite, read_ahead, workers, vram_bias):
        self.dynamic_appetite = max(0.3, min(2.0, appetite))
        self.READ_AHEAD_BLOCKS = max(1, int(read_ahead))
        self.target_workers = workers
        self.vram_bias = max(0.0, min(1.0, vram_bias))

    # ---------------- Self profiling ----------------

    def self_profile(self):
        try:
            start = time.time()
            buf = np.zeros(10_000_000, dtype=np.float32)
            buf += 1.0
            elapsed = time.time() - start
            bw = buf.nbytes / max(1e-6, elapsed)
            self.log(f"[Ramnuke][PROFILE] RAM bandwidth≈{bw/1e9:.2f} GB/s")
        except Exception as e:
            self.log(f"[Ramnuke][PROFILE] Failed: {e}")

    # ---------------- Workers & watchdog ----------------

    def worker_loop(self, idx: int):
        self.log(f"[Ramnuke][CACHE][WORKER {idx}] started.")
        while self.running:
            time.sleep(1.0)
        self.log(f"[Ramnuke][CACHE][WORKER {idx}] stopped.")

    def adjust_workers(self):
        while len(self.worker_threads) < self.target_workers:
            idx = len(self.worker_threads)
            t = threading.Thread(target=self.worker_loop, args=(idx,), daemon=True)
            self.worker_threads.append(t)
            t.start()
            self.log(f"[Ramnuke][CACHE] Spawned worker thread {idx} (target={self.target_workers}).")

    def watchdog_loop(self):
        self.log("[Ramnuke][CACHE] Watchdog loop started.")
        counter = 0
        while self.running:
            try:
                snap = self.telemetry_snapshot()
                frag = self.vram_fragmentation_score()
                self.log(
                    f"[Ramnuke][CACHE][WD] blocks={snap['blocks']} "
                    f"VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']} "
                    f"mode={'VRAM+RAM' if self.vram_available else 'RAM-ONLY'} "
                    f"frag≈{frag:.2f}"
                )
                stats = self._disk_stats()
                if stats and self.prev_disk_stats:
                    phys = self.disk_io_physics(self.prev_disk_stats, stats)
                    if phys:
                        self.log(
                            f"[Ramnuke][CACHE][WD][DISK] read_bw≈{phys['read_bw']:.1f} B/ms "
                            f"write_bw≈{phys['write_bw']:.1f} B/ms "
                            f"read_lat≈{phys['read_lat']:.6f} ms/B "
                            f"write_lat≈{phys['write_lat']:.6f} ms/B"
                        )
                self.prev_disk_stats = stats
                counter += 1
                if counter % 12 == 0:
                    self._try_vram_resurrect()

                now = time.time()
                if now - self.last_profile_time > self.profile_interval:
                    self.self_profile()
                    self.last_profile_time = now
            except Exception as e:
                self.log(f"[Ramnuke][CACHE][WD] error: {e}")
            time.sleep(5)

    # ---------------- Public control ----------------

    def start(self):
        self.log("[Ramnuke][CACHE] Starting cache manager...")
        self.initialize()
        self.running = True
        t = threading.Thread(target=self.watchdog_loop, daemon=True)
        t.start()
        self.adjust_workers()
        self.log("[Ramnuke][CACHE] Cache manager is active.")

    def stop(self):
        self.running = False
        self.log("[Ramnuke][CACHE] Cache manager stopped.")

# -------------------------------------------------------------------
# App-level: CachedFileInterface + DirectoryPreloader
# -------------------------------------------------------------------

class CachedFileInterface:
    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.block_size = cache_manager.BLOCK_SIZE

    def _block_id_for(self, path, offset):
        return (hash(os.path.abspath(path)) & 0x7FFFFFFF) ^ (offset // self.block_size)

    def read(self, path, offset, length):
        data = bytearray()
        remaining = length
        while remaining > 0:
            block_id = self._block_id_for(path, offset)
            block = self.cache.read_block(block_id)
            block_bytes = block.view(np.uint8)
            block_off = offset % self.block_size
            take = min(remaining, self.block_size - block_off)
            data.extend(block_bytes[block_off:block_off+take])
            offset += take
            remaining -= take
        return bytes(data)

    def read_all(self, path):
        size = os.path.getsize(path)
        return self.read(path, 0, size)

class DirectoryPreloader:
    def __init__(self, cache_file_iface, logger=print, exts=None, max_file_size_mb=2048):
        self.cf = cache_file_iface
        self.log = logger
        self.exts = [e.lower() for e in (exts or [])]
        self.max_file_size = max_file_size_mb * 1024 * 1024

    def _wanted(self, path):
        if not self.exts:
            return True
        _, ext = os.path.splitext(path)
        return ext.lower() in self.exts

    def preload_dir(self, root):
        root = os.path.abspath(root)
        self.log(f"[Ramnuke][PRELOAD] Scanning {root}")
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                if not self._wanted(full):
                    continue
                try:
                    size = os.path.getsize(full)
                    if size > self.max_file_size:
                        self.log(f"[Ramnuke][PRELOAD] Skipping large file {full} ({size/1e6:.1f} MB)")
                        continue
                    self.log(f"[Ramnuke][PRELOAD] Caching {full} ({size/1e6:.1f} MB)")
                    _ = self.cf.read_all(full)
                except Exception as e:
                    self.log(f"[Ramnuke][PRELOAD] Failed {full}: {e}")

# -------------------------------------------------------------------
# Training engine (online learner + 4h weight persistence)
# -------------------------------------------------------------------

class TrainingEngine:
    """
    Background trainer that uses state_history to train TinySequenceModel
    to predict next block count from recent state windows.
    Also periodically saves weights to disk (4h interval).
    """

    def __init__(self, gui, model: TinySequenceModel, weights_path: str,
                 interval_sec=60, batch_size=64, autosave_interval_sec=4*3600):
        self.gui = gui
        self.model = model
        self.interval_sec = interval_sec
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.autosave_interval_sec = autosave_interval_sec
        self.last_save_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            try:
                time.sleep(self.interval_sec)
                self.train_step()
                now = time.time()
                if now - self.last_save_time >= self.autosave_interval_sec:
                    self.model.save_weights(self.weights_path)
                    self.last_save_time = now
                    self.gui.log(f"[Ramnuke][TRAIN] Auto-saved model weights → {self.weights_path}")
            except Exception as e:
                self.gui.log(f"[Ramnuke][TRAIN] Error in training loop: {e}")
                self.gui.log(traceback.format_exc())

    def build_sample(self, idx, seq_len):
        hist = self.gui.state_history
        if idx + 1 >= len(hist):
            return None, None

        start = max(0, idx - seq_len + 1)
        window = hist[start:idx+1]
        if len(window) < seq_len:
            window = [window[0]] * (seq_len - len(window)) + list(window)

        X = np.zeros((seq_len, self.model.feat_dim), dtype=np.float32)
        for i, st in enumerate(window):
            X[i, 0] = float(st.blocks)
            X[i, 1] = float(st.gpu_load or 0.0)
            X[i, 2] = float(st.temp_c or 0.0)
            X[i, 3] = float(st.disk_read_bw or 0.0)
            X[i, 4] = float(st.vram_frag or 0.0)

        y = float(hist[idx+1].blocks)
        return X, y

    def train_step(self):
        hist = self.gui.state_history
        if len(hist) < self.model.seq_len + 2:
            return

        idxs = list(range(len(hist) - 1))
        random.shuffle(idxs)
        idxs = idxs[:self.batch_size]

        X_batch = []
        y_batch = []
        for idx in idxs:
            X, y = self.build_sample(idx, self.model.seq_len)
            if X is None:
                continue
            X_batch.append(X)
            y_batch.append(y)

        if not X_batch:
            return

        loss = self.model.train_on_batch(X_batch, y_batch, lr=1e-5)
        self.gui.log(f"[Ramnuke][TRAIN] batch={len(X_batch)} loss={loss:.2f}")

# -------------------------------------------------------------------
# GUI cockpit
# -------------------------------------------------------------------

class AutoLoaderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ramnuke – VRAM/RAM Control System")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)

        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_gpu_info = QtWidgets.QPushButton("GPU / VRAM Info")
        self.btn_gpu_info.clicked.connect(self.handle_gpu_info)

        self.btn_preload = QtWidgets.QPushButton("Preload Directory into Ramnuke Cache")
        self.btn_preload.clicked.connect(self.handle_preload_dir)

        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)

        controls_layout.addWidget(self.btn_gpu_info)
        controls_layout.addWidget(self.btn_preload)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_exit)

        self.info_label = QtWidgets.QLabel(
            "Ramnuke – VRAM/RAM cybernetic control system with predictive policy, learning engine, and Altered States overrides."
        )

        self.override_group = QtWidgets.QGroupBox("Altered States – Manual Override")
        override_layout = QtWidgets.QFormLayout()

        self.slider_appetite = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_appetite.setRange(50, 200)
        self.slider_appetite.setValue(100)

        self.slider_horizon = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_horizon.setRange(10, 200)
        self.slider_horizon.setValue(50)

        self.slider_damp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_damp.setRange(5, 50)
        self.slider_damp.setValue(30)

        self.slider_readahead = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_readahead.setRange(1, 64)
        self.slider_readahead.setValue(8)

        override_layout.addRow("Appetite", self.slider_appetite)
        override_layout.addRow("Horizon", self.slider_horizon)
        override_layout.addRow("Dampening", self.slider_damp)
        override_layout.addRow("Read-Ahead", self.slider_readahead)
        self.override_group.setLayout(override_layout)

        self.state_panel = QtWidgets.QLabel()
        self.state_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.state_panel.setStyleSheet("font-family: Consolas, monospace;")

        main_layout.addWidget(self.info_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.override_group)
        main_layout.addWidget(self.state_panel)
        main_layout.addWidget(self.log_view)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.background_tick)

        self.cache = None
        self.file_cache = None
        self.preloader = None

        # Tiny sequence model + neural predictor
        self.model_weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ramnuke_seq_model.npz",
        )
        self.sequence_model = TinySequenceModel(seq_len=32, feat_dim=5, hidden=16)
        if self.sequence_model.load_weights(self.model_weights_path):
            self.log(f"[Ramnuke][MODEL] Loaded model weights from {self.model_weights_path}")
        else:
            self.log(f"[Ramnuke][MODEL] No existing weights, starting fresh.")

        self.cache_blocks_predictor = NeuralPredictor(model=self.sequence_model)
        self.cpu_predictor = TelemetryPredictor()
        self.gpu_predictor = TelemetryPredictor()

        self.state_history = []
        self.max_state_history = 5000

        # Training engine with 4h autosave
        self.training_engine = TrainingEngine(
            self,
            self.sequence_model,
            weights_path=self.model_weights_path,
            interval_sec=60,
            batch_size=64,
            autosave_interval_sec=4 * 3600,
        )

        self.log("[Ramnuke][SYSTEM] GUI initialized.")
        self.log_loaded_modules()

        QtCore.QTimer.singleShot(500, self.auto_start_engine)

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)
        print(msg)

    def log_loaded_modules(self):
        self.log("[Ramnuke][AUTOLOADER] Module status:")
        self.log("  PyQt5: OK")
        self.log(f"  psutil: {'OK' if psutil else 'MISSING'}")
        self.log(f"  pyyaml: {'OK' if yaml else 'MISSING'}")
        self.log(f"  numpy: {'OK' if np is not None else 'MISSING'}")
        self.log(f"  pynvml: {'OK' if pynvml else 'MISSING'}")
        self.log(f"  pycuda: {'OK' if pycuda else 'MISSING'}")

    def auto_start_engine(self):
        self.log("[Ramnuke][SYSTEM] Auto-starting caching engine...")
        self.start_caching_engine()
        self.timer.start()

    def handle_gpu_info(self):
        self.log("[Ramnuke][ACTION] GPU / VRAM Info requested.")
        if not pynvml:
            self.log("[Ramnuke][GPU] pynvml not available.")
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self.log(f"[Ramnuke][GPU] {count} GPU(s) detected.")
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.log(
                    f"[Ramnuke][GPU {i}] {name} | VRAM: {mem.total/1e9:.2f} GB total, "
                    f"{mem.used/1e9:.2f} GB used, {mem.free/1e9:.2f} GB free"
                )
            pynvml.nvmlShutdown()
        except Exception as e:
            self.log(f"[Ramnuke][GPU] Failed to query: {e}")
            self.log(traceback.format_exc())

    def handle_preload_dir(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select directory to preload", ""
        )
        if not root:
            return
        self.log(f"[Ramnuke][ACTION] Preloading directory: {root}")
        threading.Thread(
            target=self.preloader.preload_dir,
            args=(root,),
            daemon=True,
        ).start()

    def start_caching_engine(self):
        try:
            self.cache = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            self.cache.start()
            self.file_cache = CachedFileInterface(self.cache)
            self.preloader = DirectoryPreloader(
                self.file_cache,
                logger=self.log,
                exts=[".bin", ".pt", ".pth", ".onnx", ".npy", ".npz", ".pak", ".pak2"],
                max_file_size_mb=2048,
            )
            self.log("[Ramnuke][ENGINE] VRAM/RAM cache manager + app-level interfaces are active.")
        except Exception as e:
            self.log(f"[Ramnuke][ENGINE] Failed to start cache manager: {e}")
            self.log(traceback.format_exc())

    def background_tick(self):
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self.cpu_predictor.update(cpu)
            cpu_next = self.cpu_predictor.predict_next() or cpu
            self.log(
                f"[Ramnuke][TICK] CPU {cpu:.1f}% (EH next≈{cpu_next:.1f}% ) | "
                f"RAM {mem.used/1e9:.2f}/{mem.total/1e9:.2f} GB"
            )

        if not self.cache:
            return

        snap = self.cache.telemetry_snapshot()
        blocks = snap["blocks"]
        self.cache_blocks_predictor.update(blocks)
        blocks_next = self.cache_blocks_predictor.predict_next() or blocks
        anomalous = self.cache_blocks_predictor.is_anomalous(blocks)
        regime_flag = "ANOM" if anomalous else "OK"

        temp = self.cache._gpu_temperature()
        power = self.cache._power_state()
        gpu_load = self.cache._gpu_load()
        if gpu_load is not None:
            self.gpu_predictor.update(gpu_load)
            gpu_next = self.gpu_predictor.predict_next() or gpu_load
        else:
            gpu_next = None

        stats = self.cache._disk_stats()
        disk_read_bw = None
        disk_write_bw = None
        if stats and self.cache.prev_disk_stats:
            phys = self.cache.disk_io_physics(self.cache.prev_disk_stats, stats)
            if phys:
                disk_read_bw = phys["read_bw"]
                disk_write_bw = phys["write_bw"]
        self.cache.prev_disk_stats = stats

        vram_frag = self.cache.vram_fragmentation_score()

        app_override = self.slider_appetite.value() / 100.0
        hor_override = self.slider_horizon.value()
        damp_override = self.slider_damp.value() / 100.0
        ra_override = self.slider_readahead.value()

        self.cache_blocks_predictor.fallback.appetite = app_override
        self.cache_blocks_predictor.fallback.window = hor_override
        self.cache_blocks_predictor.fallback.alpha = damp_override

        gpu_future = gpu_next if gpu_next is not None else gpu_load
        if gpu_future is not None and gpu_future > 80:
            self.cache_blocks_predictor.fallback.appetite *= 0.9

        trend = self.cache_blocks_predictor.fallback.trend
        appetite_for_state = self.cache_blocks_predictor.fallback.appetite

        state = CacheStateVector(
            temp_c=temp,
            power_mode=power,
            gpu_load=gpu_load,
            disk_read_bw=disk_read_bw,
            disk_write_bw=disk_write_bw,
            vram_frag=vram_frag,
            blocks=blocks,
            blocks_trend=trend,
            appetite=appetite_for_state,
        )

        self.state_history.append(state)
        if len(self.state_history) > self.max_state_history:
            self.state_history.pop(0)

        appetite, ra, workers, vram_bias, prewarm = self.cache.master_policy(state)

        ra = min(ra, ra_override)

        self.cache.apply_master_policy(appetite, ra, workers, vram_bias)
        self.cache.adjust_workers()

        self.log(
            f"[Ramnuke][TICK][CACHE] blocks={blocks} (EH next≈{blocks_next:.1f}) "
            f"VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']} "
            f"[regime={regime_flag}]"
        )
        self.log(
            f"[Ramnuke][TICK][POLICY] app={appetite:.2f} RA={ra} workers={workers} "
            f"vram_bias={vram_bias:.2f} temp={temp}C gpu={gpu_load}% "
            f"gpu_next={gpu_next} frag={vram_frag:.2f} "
            f"disk_read_bw={disk_read_bw} prewarm={prewarm}"
        )

        heat = (
            f"State Vector / Policy\n"
            f"  temp: {temp} C\n"
            f"  power: {power}\n"
            f"  gpu: {gpu_load}% (next≈{gpu_next})\n"
            f"  blocks: {blocks} trend≈{trend:.1f}\n"
            f"  frag: {vram_frag:.2f}\n"
            f"  disk_read_bw: {disk_read_bw}\n"
            f"  appetite: {appetite:.2f}\n"
            f"  read_ahead: {ra}\n"
            f"  workers: {workers}\n"
            f"  vram_bias: {vram_bias:.2f}\n"
        )
        self.state_panel.setText(heat)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = AutoLoaderGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

