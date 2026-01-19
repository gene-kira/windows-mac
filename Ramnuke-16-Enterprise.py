import os
import sys
import time
import json
import random
import threading
import traceback
from enum import Enum

import numpy as np

from PyQt5 import QtWidgets, QtCore

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import pycuda.driver as pycuda_driver
    import pycuda.autoinit as pycuda_autoinit
except ImportError:
    pycuda_driver = None
    pycuda_autoinit = None

try:
    import zmq
except ImportError:
    zmq = None

# Optional LLM backends
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None


# -------------------------------------------------------------------
# Helpers / stubs
# -------------------------------------------------------------------

def decide_ram_cache_bytes():
    # simple heuristic: 4 GB or 1/4 of total, whichever smaller
    if not psutil:
        return 4 * 1024 * 1024 * 1024
    total = psutil.virtual_memory().total
    return int(min(4 * 1024 * 1024 * 1024, total // 4))


def normalize(x, lo, hi):
    if x is None:
        return 0.0
    if hi == lo:
        return 0.0
    v = (x - lo) / float(hi - lo)
    return max(0.0, min(1.0, v))


class RaidMode(Enum):
    RAID0 = "RAID0"
    RAID1 = "RAID1"


class MetaState(Enum):
    SENTINEL = "SENTINEL"
    HYPER_FLOW = "HYPER_FLOW"
    DEEP_DREAM = "DEEP_DREAM"
    RECOVERY_FLOW = "RECOVERY_FLOW"


class CacheStateVector:
    def __init__(
        self,
        temp_c=None,
        power_mode=None,
        gpu_load=None,
        disk_read_bw=None,
        disk_write_bw=None,
        vram_frag=0.0,
        blocks=0,
        blocks_trend=0.0,
        appetite=1.0,
    ):
        self.temp_c = temp_c
        self.power_mode = power_mode
        self.gpu_load = gpu_load
        self.disk_read_bw = disk_read_bw
        self.disk_write_bw = disk_write_bw
        self.vram_frag = vram_frag
        self.blocks = blocks
        self.blocks_trend = blocks_trend
        self.appetite = appetite


# -------------------------------------------------------------------
# Tiny sequence model + predictors
# -------------------------------------------------------------------

class TinySequenceModel:
    def __init__(self, seq_len=32, feat_dim=5, hidden=16):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden = hidden
        # simple linear weights as placeholder
        self.W = np.random.randn(feat_dim, 1).astype(np.float32)

    def load_weights(self, path):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self.W = data["W"]
            return True
        except Exception:
            return False

    def save_weights(self, path):
        try:
            np.savez(path, W=self.W)
        except Exception:
            pass

    def train_on_batch(self, X_batch, y_batch, lr=1e-5):
        # X_batch: list of (seq_len, feat_dim)
        # y_batch: list of scalars
        if not X_batch:
            return 0.0
        X = np.stack([x[-1] for x in X_batch], axis=0)  # last step only
        y = np.array(y_batch, dtype=np.float32).reshape(-1, 1)
        pred = X @ self.W
        err = pred - y
        loss = float((err ** 2).mean())
        grad = (X.T @ err) / X.shape[0]
        self.W -= lr * grad
        return loss

    def predict(self, X_seq):
        # X_seq: (seq_len, feat_dim)
        x = X_seq[-1]
        return float(x @ self.W).item()


class TelemetryPredictor:
    def __init__(self, window=30, alpha=0.3):
        self.window = window
        self.alpha = alpha
        self.values = []
        self.ewma = None
        self.trend = 0.0
        self.appetite = 1.0

    def update(self, v):
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)
        if self.ewma is None:
            self.ewma = float(v)
        else:
            self.ewma = self.alpha * float(v) + (1 - self.alpha) * self.ewma
        if len(self.values) >= 2:
            self.trend = self.values[-1] - self.values[-2]

    def predict_next(self):
        if self.ewma is None:
            return None
        return self.ewma + self.trend


class NeuralPredictor:
    def __init__(self, model=None):
        self.model = model
        self.fallback = TelemetryPredictor()
        self.last_pred = None

    def update(self, value: float):
        self.fallback.update(value)

    def predict_next(self):
        pred = self.fallback.predict_next()
        self.last_pred = pred
        return pred

    def is_anomalous(self, current_value: float, threshold: float = 3.0) -> bool:
        pred = self.last_pred
        if pred is None:
            pred = self.predict_next() or current_value
        err = abs(current_value - pred)
        scale = max(1.0, self.fallback.window * (1.0 - self.fallback.alpha))
        return err > threshold * scale


# -------------------------------------------------------------------
# HybridBrain
# -------------------------------------------------------------------

class HybridBrain:
    def __init__(self, gui, model: TinySequenceModel):
        self.gui = gui
        self.model = model

        # horizons in ticks (3s per tick approx)
        self.horizons = {
            "1s": 1,
            "5s": 2,
            "30s": 10,
            "120s": 40,
        }

        self.behavior_fingerprint = {
            "overload_patterns": 0.0,
            "stable_patterns": 0.0,
            "beast_wins": 0.0,
        }

        self.reinforcement_memory = []
        self.max_reinf = 500

        self.meta_state = MetaState.SENTINEL
        self.dynamic_thresholds = {
            "overload": 0.8,
            "stable": 0.3,
        }

        self.baseline_turbulence = 0.0
        self.baseline_variance = 0.0

    def _extract_series(self):
        hist = self.gui.state_history
        if not hist:
            return []
        return [s.blocks for s in hist]

    def _multi_horizon_predictions(self):
        series = self._extract_series()
        if not series:
            return {}

        preds = {}
        arr = np.array(series, dtype=float)
        if len(arr) < 4:
            base = arr[-1]
            for k in self.horizons:
                preds[k] = base
            return preds

        ewma = arr.copy()
        alpha = 0.3
        for i in range(1, len(arr)):
            ewma[i] = alpha * arr[i] + (1 - alpha) * ewma[i - 1]

        last = ewma[-1]
        trend = (ewma[-1] - ewma[max(0, len(ewma) - 5)]) / max(
            1, min(5, len(ewma) - 1)
        )

        for name, steps in self.horizons.items():
            preds[name] = last + trend * steps

        return preds

    def _meta_confidence(self, preds):
        series = self._extract_series()
        if len(series) < 10:
            return 0.3

        arr = np.array(series, dtype=float)
        var = float(arr.var())
        self.baseline_variance = 0.9 * self.baseline_variance + 0.1 * var

        diffs = np.diff(arr)
        turbulence = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0
        self.baseline_turbulence = 0.9 * self.baseline_turbulence + 0.1 * turbulence

        trend_stability = 1.0 / (1.0 + turbulence)
        var_norm = 1.0 / (1.0 + var / (self.baseline_variance + 1e-6))

        reinf = 0.5
        if self.reinforcement_memory:
            reinf = sum(self.reinforcement_memory) / len(self.reinforcement_memory)

        conf = (
            0.25 * trend_stability
            + 0.25 * var_norm
            + 0.25 * reinf
            + 0.25 * (1.0 / (1.0 + self.baseline_turbulence))
        )
        return max(0.0, min(1.0, conf))

    def _update_behavior_fingerprint(self, state: CacheStateVector, preds):
        if not preds:
            return
        current = state.blocks
        future_30 = preds.get("30s", current)
        delta = future_30 - current

        if delta > 5000:
            self.behavior_fingerprint["overload_patterns"] += 0.01
        elif delta < -2000:
            self.behavior_fingerprint["stable_patterns"] += 0.01
        else:
            self.behavior_fingerprint["beast_wins"] += 0.005

        for k in self.behavior_fingerprint:
            self.behavior_fingerprint[k] = max(
                0.0, min(1.0, self.behavior_fingerprint[k])
            )

    def _update_dynamic_thresholds(self):
        bf = self.behavior_fingerprint
        overload_bias = bf["overload_patterns"]
        stable_bias = bf["stable_patterns"]

        self.dynamic_thresholds["overload"] = 0.6 + 0.4 * overload_bias
        self.dynamic_thresholds["stable"] = 0.2 + 0.3 * stable_bias

    def _select_meta_state(self, preds, confidence):
        if not preds:
            self.meta_state = MetaState.SENTINEL
            return

        current = self.gui.state_history[-1].blocks if self.gui.state_history else 0
        future_30 = preds.get("30s", current)
        delta = future_30 - current

        overload_th = self.dynamic_thresholds["overload"]
        stable_th = self.dynamic_thresholds["stable"]

        rel = 0.0
        if current > 0:
            rel = delta / max(1.0, current)
        rel = max(-1.0, min(1.0, rel))

        if rel > overload_th and confidence > 0.5:
            self.meta_state = MetaState.HYPER_FLOW
        elif rel < -stable_th and confidence > 0.5:
            self.meta_state = MetaState.DEEP_DREAM
        elif confidence < 0.3:
            self.meta_state = MetaState.RECOVERY_FLOW
        else:
            self.meta_state = MetaState.SENTINEL

    def _predictive_risk_dampening(self, appetite, read_ahead, workers):
        if self.meta_state == MetaState.HYPER_FLOW:
            return appetite * 1.1, read_ahead + 2, min(8, workers + 1)
        if self.meta_state == MetaState.DEEP_DREAM:
            return appetite * 0.8, max(1, read_ahead - 2), max(1, workers - 1)
        if self.meta_state == MetaState.RECOVERY_FLOW:
            return appetite * 0.7, max(1, read_ahead - 3), max(1, workers - 2)
        return appetite, read_ahead, workers

    def reinforce(self, reward: float):
        self.reinforcement_memory.append(reward)
        if len(self.reinforcement_memory) > self.max_reinf:
            self.reinforcement_memory.pop(0)

    def decide(
        self,
        base_appetite,
        base_ra,
        base_workers,
        base_vram_bias,
        state: CacheStateVector,
    ):
        preds = self._multi_horizon_predictions()
        confidence = self._meta_confidence(preds)
        self._update_behavior_fingerprint(state, preds)
        self._update_dynamic_thresholds()
        self._select_meta_state(preds, confidence)

        appetite, ra, workers = self._predictive_risk_dampening(
            base_appetite, base_ra, base_workers
        )

        if confidence < 0.3:
            appetite *= 0.9
            ra = max(1, ra - 1)
        elif confidence > 0.7:
            appetite *= 1.05

        appetite = max(0.3, min(2.5, appetite))
        ra = max(1, min(64, int(ra)))
        workers = max(1, min(8, workers))

        return appetite, ra, workers, base_vram_bias, preds, confidence, self.meta_state


# -------------------------------------------------------------------
# ClusterAgent (ZeroMQ)
# -------------------------------------------------------------------

class ClusterAgent:
    def __init__(self, gui, node_id: str, bind_addr: str = None, peers=None):
        self.gui = gui
        self.node_id = node_id
        self.bind_addr = bind_addr
        self.peers = peers or []
        self.running = True

        if zmq is None:
            self.gui.log("[Ramnuke][CLUSTER] zmq not available, cluster disabled.")
            self.ctx = None
            self.pub = None
            self.sub = None
            self.cluster_pressures = {}
            self.cluster_hot_embeddings = {}
            return

        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        if self.bind_addr:
            self.pub.bind(self.bind_addr)

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        for p in self.peers:
            self.sub.connect(p)

        self.cluster_pressures = {}
        self.cluster_hot_embeddings = {}

        self.thread_tx = threading.Thread(target=self.tx_loop, daemon=True)
        self.thread_rx = threading.Thread(target=self.rx_loop, daemon=True)
        self.thread_tx.start()
        self.thread_rx.start()

    def tx_loop(self):
        if self.pub is None:
            return
        while self.running:
            try:
                state = self.gui.state_history[-1] if self.gui.state_history else None
                if state is None:
                    time.sleep(1.0)
                    continue
                payload = {
                    "node_id": self.node_id,
                    "pressure": float(getattr(self.gui, "last_pressure", 0.0)),
                    "appetite": float(getattr(self.gui, "last_appetite", 1.0)),
                    "hot_embeddings": list(getattr(self.gui, "hot_embeddings", [])),
                    "ts": time.time(),
                }
                self.pub.send_json(payload)
            except Exception as e:
                self.gui.log(f"[Ramnuke][CLUSTER][TX] error: {e}")
            time.sleep(2.0)

    def rx_loop(self):
        if self.sub is None:
            return
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(1000))
                if self.sub in socks and socks[self.sub] == zmq.POLLIN:
                    msg = self.sub.recv_json()
                    nid = msg.get("node_id")
                    if not nid or nid == self.node_id:
                        continue
                    self.cluster_pressures[nid] = msg.get("pressure", 0.0)
                    self.cluster_hot_embeddings[nid] = msg.get("hot_embeddings", [])
            except Exception as e:
                self.gui.log(f"[Ramnuke][CLUSTER][RX] error: {e}")

    def cluster_pressure(self) -> float:
        if not self.cluster_pressures:
            return 0.0
        return sum(self.cluster_pressures.values()) / len(self.cluster_pressures)

    def shared_hot_embeddings(self):
        out = set()
        for lst in self.cluster_hot_embeddings.values():
            for x in lst:
                out.add(x)
        return list(out)


# -------------------------------------------------------------------
# MemoryCompressionOrgan
# -------------------------------------------------------------------

class MemoryCompressionOrgan:
    def __init__(self, gui, cache_manager, interval_sec=60):
        self.gui = gui
        self.cache = cache_manager
        self.interval_sec = interval_sec
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self.gui.log(f"[Ramnuke][MEMCOMP] error: {e}")
                self.gui.log(traceback.format_exc())
            time.sleep(self.interval_sec)

    def run_cycle(self):
        # simple heuristic: if many RAM blocks and VRAM low, compress cold RAM blocks
        if not self.cache:
            return
        snap = self.cache.telemetry_snapshot()
        blocks = snap["blocks"]
        ram_blocks = snap["ram_blocks"]
        if blocks == 0:
            return
        ram_ratio = ram_blocks / max(1, blocks)
        if ram_ratio < 0.5:
            return
        # pretend to compress: just log for now
        self.gui.log(
            f"[Ramnuke][MEMCOMP] High RAM ratio ({ram_ratio:.2f}), "
            f"considering compression of cold blocks."
        )


# -------------------------------------------------------------------
# EmbeddingPrefetchOrgan
# -------------------------------------------------------------------

class EmbeddingPrefetchOrgan:
    def __init__(self, gui, cache_manager, embed_dim=128):
        self.gui = gui
        self.cache = cache_manager
        self.embed_dim = embed_dim
        self.local_hot_embeddings = set()
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            try:
                self.update_hot_embeddings()
                self.prefetch()
            except Exception as e:
                self.gui.log(f"[Ramnuke][EMB] error: {e}")
                self.gui.log(traceback.format_exc())
            time.sleep(5.0)

    def update_hot_embeddings(self):
        # placeholder: derive "hot embeddings" from recent blocks
        if not self.cache:
            return
        snap = self.cache.telemetry_snapshot()
        blocks = snap["blocks"]
        # treat block IDs as embedding IDs for now
        hot_ids = list(range(max(0, blocks - 32), blocks))
        self.local_hot_embeddings = set(hot_ids)
        self.gui.hot_embeddings = list(self.local_hot_embeddings)

    def prefetch(self):
        # cluster-aware: merge local + cluster hot embeddings
        cluster_agent = getattr(self.cache, "cluster_agent", None)
        shared = set()
        if cluster_agent is not None:
            shared = set(cluster_agent.shared_hot_embeddings())
        all_hot = self.local_hot_embeddings | shared
        # just log for now
        if all_hot:
            self.gui.log(
                f"[Ramnuke][EMB] Prefetching embeddings for {len(all_hot)} hot IDs "
                f"(local={len(self.local_hot_embeddings)}, cluster={len(shared)})"
            )


# -------------------------------------------------------------------
# LLMAdvisorOrgan (small LLM, GPU-accelerated if possible)
# -------------------------------------------------------------------

class LLMAdvisorOrgan:
    def __init__(self, gui, model_path: str, backend="onnx", interval_sec=30):
        self.gui = gui
        self.model_path = model_path
        self.backend = backend
        self.interval_sec = interval_sec
        self.running = True
        self.last_advice = ""
        self.last_appetite_delta = 0.0

        self.session = None
        self.device = "cpu"
        self._init_backend()

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def _init_backend(self):
        try:
            if self.backend == "onnx" and ort is not None:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
                self.gui.log(f"[Ramnuke][LLM] ONNX session ready on {self.device}.")
            elif self.backend == "torch" and torch is not None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.session = torch.jit.load(self.model_path, map_location=self.device)
                self.session.eval()
                self.gui.log(f"[Ramnuke][LLM] Torch model ready on {self.device}.")
            else:
                self.gui.log("[Ramnuke][LLM] No suitable backend, advisor disabled.")
        except Exception as e:
            self.gui.log(f"[Ramnuke][LLM] Failed to init backend: {e}")
            self.session = None

    def loop(self):
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self.gui.log(f"[Ramnuke][LLM] error: {e}")
                self.gui.log(traceback.format_exc())
            time.sleep(self.interval_sec)

    def _build_state_summary(self):
        if not self.gui.state_history:
            return "no_state"
        s = self.gui.state_history[-1]
        return json.dumps(
            {
                "temp": s.temp_c,
                "gpu": s.gpu_load,
                "blocks": s.blocks,
                "trend": s.blocks_trend,
                "frag": s.vram_frag,
            }
        )

    def _infer_appetite_delta(self, summary: str) -> float:
        # This is a placeholder: real implementation would tokenize summary
        # and run through the LLM. Here we just map simple heuristics.
        if self.session is None:
            # fallback heuristic
            try:
                data = json.loads(summary)
                gpu = data.get("gpu") or 0
                frag = data.get("frag") or 0
                if gpu > 90 or frag > 0.8:
                    return -0.1
                if gpu < 30 and frag < 0.3:
                    return 0.05
            except Exception:
                pass
            return 0.0

        # Example ONNX/Torch path would go here; we keep it abstract.
        return 0.0

    def run_cycle(self):
        summary = self._build_state_summary()
        delta = self._infer_appetite_delta(summary)
        self.last_appetite_delta = delta
        if delta > 0:
            self.last_advice = "LLM: environment looks safe, slight appetite increase."
        elif delta < 0:
            self.last_advice = "LLM: environment looks risky, dampen appetite."
        else:
            self.last_advice = "LLM: neutral stance."
        self.gui.log(f"[Ramnuke][LLM] advisory appetite_delta={delta:+.2f}")


# -------------------------------------------------------------------
# VRAM cache manager
# -------------------------------------------------------------------

class VRAMCacheManager:
    VRAM_TARGET_BYTES = 8 * 1024 * 1024 * 1024
    RAM_TARGET_BYTES = decide_ram_cache_bytes()
    BLOCK_SIZE = 256 * 1024
    MAX_BLOCKS = 1_000_000

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
        self.profile_interval = 300.0

        self.READ_AHEAD_BLOCKS = 4

        self.health_score = 1.0
        self.muted = False
        self.error_count = 0

        self.cluster_agent = None  # wired from GUI

        self.log("[Ramnuke][CACHE] VRAMCacheManager constructed.")

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
                self.log(
                    f"[Ramnuke][CACHE] Attempting {self.VRAM_TARGET_BYTES/1e9:.1f} GB VRAM allocation..."
                )
                self.vram_buffer = gpuarray.zeros(total_floats, dtype=np.float32)
                if self._vram_stress_test():
                    self.vram_available = True
                    self.log(
                        "[Ramnuke][CACHE] VRAM allocation + stress test OK. VRAM tier ACTIVE."
                    )
                else:
                    self.log(
                        "[Ramnuke][CACHE] VRAM stress test failed → RAM-only mode."
                    )
                    self.vram_available = False
                    self.vram_buffer = None
            except Exception as e:
                self.log("[Ramnuke][CACHE] VRAM allocation failed → RAM-only mode.")
                self.log(f"[Ramnuke][CACHE] Reason: {e}")
                self.vram_available = False
                self.vram_buffer = None
        else:
            self.log("[Ramnuke][CACHE] No CUDA backend detected → RAM-only mode.")
            self.vram_available = False
            self.vram_buffer = None

        try:
            self.log(
                f"[Ramnuke][CACHE] Allocating {self.RAM_TARGET_BYTES/1e9:.1f} GB system RAM..."
            )
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
                    return self.vram_buffer[
                        offset : offset + (self.BLOCK_SIZE // 4)
                    ]
                else:
                    return self.ram_buffer[
                        offset : offset + (self.BLOCK_SIZE // 4)
                    ]

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
                self.error_count += 1

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
                temp = pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU
                )
                pynvml.nvmlShutdown()
                return temp
        except Exception:
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
        except Exception:
            pass
        return None

    def _power_state(self):
        try:
            if psutil:
                batt = psutil.sensors_battery()
                if batt is None:
                    return None
                return "AC" if batt.power_plugged else "DC"
        except Exception:
            pass
        return None

    def _disk_stats(self):
        try:
            if psutil:
                d = psutil.disk_io_counters()
                return d.read_bytes, d.write_bytes, d.read_time, d.write_time
        except Exception:
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
            self.error_count += 1

    def fused_pressure(self, state: CacheStateVector):
        w = self.policy_weights
        return (
            w["temp"] * normalize(state.temp_c, 40, 90)
            + w["gpu"] * normalize(state.gpu_load, 0, 100)
            + w["frag"] * state.vram_frag
            + w["trend"] * normalize(state.blocks_trend, -5000, 5000)
            + w["disk"] * normalize(state.disk_read_bw or 0, 0, 50000)
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
        gpu_penalty = 1.0 if (state.gpu_load or 0) < 80 else 0.6
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
            "temp": normalize(state.temp_c, 40, 90),
            "gpu": normalize(state.gpu_load, 0, 100),
            "frag": state.vram_frag,
            "trend": normalize(state.blocks_trend, -5000, 5000),
            "disk": normalize(state.disk_read_bw or 0, 0, 50000),
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

        # cluster-aware adjustment
        cluster_agent = getattr(self, "cluster_agent", None)
        if cluster_agent is not None:
            cpress = cluster_agent.cluster_pressure()
            pressure_before = 0.7 * pressure_before + 0.3 * cpress

        appetite = self.appetite_curve(pressure_before)
        read_ahead = self.read_ahead_curve(appetite)
        workers = self.worker_curve(appetite)
        vram_bias = self.vram_bias_curve(state)
        prewarm = self.disk_prewarm_factor(state)
        pressure_after = self.fused_pressure(state)
        self.learn_policy(state, pressure_before, pressure_after)
        return appetite, read_ahead, workers, vram_bias, prewarm

    def apply_master_policy(self, appetite, read_ahead, workers, vram_bias):
        if self.muted:
            self.dynamic_appetite = 0.3
            self.READ_AHEAD_BLOCKS = 1
            self.target_workers = 0
            self.vram_bias = 0.0
            return
        self.dynamic_appetite = max(0.3, min(2.0, appetite))
        self.READ_AHEAD_BLOCKS = max(1, int(read_ahead))
        self.target_workers = workers
        self.vram_bias = max(0.0, min(1.0, vram_bias))

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
            self.log(
                f"[Ramnuke][CACHE] Spawned worker thread {idx} (target={self.target_workers})."
            )

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

                self.health_score = max(0.0, 1.0 - 0.05 * self.error_count)
                if self.health_score < 0.4 and not self.muted:
                    self.muted = True
                    self.log("[Ramnuke][CACHE] Spine muted due to low health.")
            except Exception as e:
                self.log(f"[Ramnuke][CACHE][WD] error: {e}")
            time.sleep(5)

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
# RAID cache manager (6-way)
# -------------------------------------------------------------------

class RaidCacheManager:
    def __init__(
        self,
        logger,
        mode: RaidMode,
        dev_a: VRAMCacheManager,
        dev_b: VRAMCacheManager,
        dev_c: VRAMCacheManager,
        dev_d: VRAMCacheManager,
        dev_e: VRAMCacheManager,
        dev_f: VRAMCacheManager,
    ):
        self.log = logger
        self.mode = mode
        self.a = dev_a
        self.b = dev_b
        self.c = dev_c
        self.d = dev_d
        self.e = dev_e
        self.f = dev_f
        self.devs = [self.a, self.b, self.c, self.d, self.e, self.f]
        self.running = False

    def start(self):
        self.log(f"[Ramnuke][RAID] Starting RAID {self.mode.value} over 6 spines (A–F)")
        for d in self.devs:
            d.start()
        self.running = True

    def stop(self):
        self.running = False
        for d in self.devs:
            d.stop()
        self.log("[Ramnuke][RAID] Stopped.")

    def _map_block_raid0(self, block_id: int):
        idx = block_id % 6
        local_id = block_id // 6
        return self.devs[idx], local_id

    def _map_block_raid1(self, block_id: int):
        return self.devs, block_id

    def read_block(self, block_id: int):
        if self.mode == RaidMode.RAID0:
            dev, local_id = self._map_block_raid0(block_id)
            if dev.muted:
                self.log(
                    f"[Ramnuke][RAID] Primary spine muted for block {block_id}, searching fallback."
                )
                last_exc = None
                for alt in self.devs:
                    if alt.muted:
                        continue
                    try:
                        return alt.read_block(local_id)
                    except Exception as e:
                        last_exc = e
                if last_exc:
                    raise last_exc
            return dev.read_block(local_id)

        devs, local_id = self._map_block_raid1(block_id)
        last_exc = None
        for d in devs:
            if d.muted:
                continue
            try:
                return d.read_block(local_id)
            except Exception as e:
                last_exc = e
                self.log(
                    f"[Ramnuke][RAID] read failed on device for block {block_id}: {e}"
                )
        raise last_exc if last_exc else RuntimeError("RAID1 read failed")

    def telemetry_snapshot(self):
        snaps = [d.telemetry_snapshot() for d in self.devs]
        return {
            "blocks": sum(s["blocks"] for s in snaps),
            "vram_blocks": sum(s["vram_blocks"] for s in snaps),
            "ram_blocks": sum(s["ram_blocks"] for s in snaps),
        }

    def vram_fragmentation_score(self):
        scores = [d.vram_fragmentation_score() for d in self.devs]
        return sum(scores) / len(scores)


# -------------------------------------------------------------------
# App-level: CachedFileInterface + DirectoryPreloader
# -------------------------------------------------------------------

class CachedFileInterface:
    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.block_size = getattr(cache_manager, "BLOCK_SIZE", 256 * 1024)

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
            data.extend(block_bytes[block_off : block_off + take])
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
                        self.log(
                            f"[Ramnuke][PRELOAD] Skipping large file {full} ({size/1e6:.1f} MB)"
                        )
                        continue
                    self.log(
                        f"[Ramnuke][PRELOAD] Caching {full} ({size/1e6:.1f} MB)"
                    )
                    _ = self.cf.read_all(full)
                except Exception as e:
                    self.log(f"[Ramnuke][PRELOAD] Failed {full}: {e}")


# -------------------------------------------------------------------
# Training engine
# -------------------------------------------------------------------

class TrainingEngine:
    def __init__(
        self,
        gui,
        model: TinySequenceModel,
        weights_path: str,
        interval_sec=60,
        batch_size=64,
        autosave_interval_sec=4 * 3600,
    ):
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
                    self.gui.log(
                        f"[Ramnuke][TRAIN] Auto-saved model weights → {self.weights_path}"
                    )
            except Exception as e:
                self.gui.log(f"[Ramnuke][TRAIN] Error in training loop: {e}")
                self.gui.log(traceback.format_exc())

    def build_sample(self, idx, seq_len):
        hist = self.gui.state_history
        if idx + 1 >= len(hist):
            return None, None

        start = max(0, idx - seq_len + 1)
        window = hist[start : idx + 1]
        if len(window) < seq_len:
            window = [window[0]] * (seq_len - len(window)) + list(window)

        X = np.zeros((seq_len, self.model.feat_dim), dtype=np.float32)
        for i, st in enumerate(window):
            X[i, 0] = float(st.blocks)
            X[i, 1] = float(st.gpu_load or 0.0)
            X[i, 2] = float(st.temp_c or 0.0)
            X[i, 3] = float(st.disk_read_bw or 0.0)
            X[i, 4] = float(st.vram_frag or 0.0)

        y = float(hist[idx + 1].blocks)
        return X, y

    def train_step(self):
        hist = self.gui.state_history
        if len(hist) < self.model.seq_len + 2:
            return

        idxs = list(range(len(hist) - 1))
        random.shuffle(idxs)
        idxs = idxs[: self.batch_size]

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
# Self-Integrity Organ
# -------------------------------------------------------------------

class SelfIntegrityOrgan:
    def __init__(self, gui, model, cpu_pred, gpu_pred, cache_pred, interval_sec=90):
        self.gui = gui
        self.model = model
        self.cpu_pred = cpu_pred
        self.gpu_pred = gpu_pred
        self.cache_pred = cache_pred
        self.interval_sec = interval_sec
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            try:
                self.check_sensors()
                self.check_metrics_consistency()
                self.check_organs()
                self.check_prediction_drift()
                self.check_model_degradation()
            except Exception as e:
                self.gui.log(f"[Ramnuke][INTEGRITY] error: {e}")
                self.gui.log(traceback.format_exc())
            time.sleep(self.interval_sec)

    def check_sensors(self):
        if not self.gui.state_history:
            return
        last = self.gui.state_history[-1]
        if last.gpu_load is None:
            self.gui.log("[Ramnuke][INTEGRITY] GPU load sensor missing or stale.")
        if last.temp_c is None:
            self.gui.log("[Ramnuke][INTEGRITY] GPU temperature sensor missing or stale.")
        if last.disk_read_bw is None and last.disk_write_bw is None:
            self.gui.log("[Ramnuke][INTEGRITY] Disk I/O metrics missing or stale.")

    def check_metrics_consistency(self):
        if not self.gui.state_history:
            return
        s = self.gui.state_history[-1]
        if s.gpu_load is not None and s.gpu_load > 90 and (
            s.temp_c is None or s.temp_c < 20
        ):
            self.gui.log(
                "[Ramnuke][INTEGRITY] Inconsistent GPU metrics: high load with low/unknown temperature."
            )
        if s.blocks < 0:
            self.gui.log(
                "[Ramnuke][INTEGRITY] Negative block count detected (logic bug?)."
            )

    def check_organs(self):
        cache = self.gui.cache
        if cache is None or not getattr(cache, "running", False):
            self.gui.log("[Ramnuke][INTEGRITY] Cache manager organ not running.")
        if (
            not getattr(self.gui, "training_engine", None)
            or not self.gui.training_engine.running
        ):
            self.gui.log("[Ramnuke][INTEGRITY] Training engine organ not running.")
        if self.model is None:
            self.gui.log("[Ramnuke][INTEGRITY] Sequence model organ missing.")
        if len(self.gui.state_history) == 0:
            self.gui.log("[Ramnuke][INTEGRITY] No state history accumulated yet.")

    def check_prediction_drift(self):
        errs = getattr(self.gui, "pred_error_history", [])
        if len(errs) < 50:
            return
        avg_err = sum(errs) / len(errs)
        if avg_err > 5000:
            self.gui.log(
                f"[Ramnuke][INTEGRITY] Prediction drift: avg block error≈{avg_err:.1f}"
            )

    def check_model_degradation(self):
        errs = getattr(self.gui, "pred_error_history", [])
        if len(errs) < 100:
            return
        mid = len(errs) // 2
        old = errs[:mid]
        new = errs[mid:]
        old_avg = sum(old) / max(1, len(old))
        new_avg = sum(new) / max(1, len(new))
        if new_avg > old_avg * 1.5 and new_avg > 2000:
            self.gui.log(
                f"[Ramnuke][INTEGRITY] Model degradation: old_err≈{old_avg:.1f}, new_err≈{new_avg:.1f}"
            )


# -------------------------------------------------------------------
# GUI cockpit
# -------------------------------------------------------------------

class AutoLoaderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Ramnuke – 6‑Spine RAID VRAM/RAM Control System (HybridBrain, Cluster, LLM)"
        )
        self.resize(1400, 950)

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
            "Ramnuke – 6‑spine RAID VRAM/RAM organism with HybridBrain, "
            "predictive policy, learning engine, self‑integrity, cluster sync, "
            "LLM advisor, memory compression, and embedding prefetch."
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

        self.raid_panel = QtWidgets.QLabel()
        self.raid_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.raid_panel.setStyleSheet("font-family: Consolas, monospace;")

        self.raid_layout_panel = QtWidgets.QLabel()
        self.raid_layout_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.raid_layout_panel.setStyleSheet("font-family: Consolas, monospace;")

        self.brain_panel = QtWidgets.QLabel()
        self.brain_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.brain_panel.setStyleSheet("font-family: Consolas, monospace;")

        # New tab for embeddings + LLM advisor
        self.tabs = QtWidgets.QTabWidget()
        self.main_status_tab = QtWidgets.QWidget()
        self.emb_tab = QtWidgets.QWidget()

        # Main status tab layout
        main_tab_layout = QtWidgets.QVBoxLayout(self.main_status_tab)
        main_tab_layout.addWidget(self.info_label)
        main_tab_layout.addLayout(controls_layout)
        main_tab_layout.addWidget(self.override_group)
        main_tab_layout.addWidget(self.state_panel)
        main_tab_layout.addWidget(self.brain_panel)
        main_tab_layout.addWidget(self.raid_panel)
        main_tab_layout.addWidget(self.raid_layout_panel)

        # Embedding / advisor tab
        emb_layout = QtWidgets.QVBoxLayout(self.emb_tab)
        self.emb_panel = QtWidgets.QLabel()
        self.emb_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.emb_panel.setStyleSheet("font-family: Consolas, monospace;")
        self.llm_panel = QtWidgets.QLabel()
        self.llm_panel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.llm_panel.setStyleSheet("font-family: Consolas, monospace;")
        emb_layout.addWidget(QtWidgets.QLabel("Embedding Prefetch / Cluster Similarity"))
        emb_layout.addWidget(self.emb_panel)
        emb_layout.addWidget(QtWidgets.QLabel("LLM Advisor"))
        emb_layout.addWidget(self.llm_panel)

        self.tabs.addTab(self.main_status_tab, "Core Status")
        self.tabs.addTab(self.emb_tab, "Embeddings / Advisor")

        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.log_view)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.background_tick)

        self.cache = None
        self.file_cache = None
        self.preloader = None

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

        self.pred_error_history = []

        self.training_engine = TrainingEngine(
            self,
            self.sequence_model,
            weights_path=self.model_weights_path,
            interval_sec=60,
            batch_size=64,
            autosave_interval_sec=4 * 3600,
        )

        self.integrity = SelfIntegrityOrgan(
            gui=self,
            model=self.sequence_model,
            cpu_pred=self.cpu_predictor,
            gpu_pred=self.gpu_predictor,
            cache_pred=self.cache_blocks_predictor,
        )

        self.hybrid_brain = HybridBrain(self, self.sequence_model)

        # cluster + organs
        self.cluster_agent = None
        self.memcomp = None
        self.emb_organ = None
        self.llm_advisor = None

        self.last_pressure = 0.0
        self.last_appetite = 1.0
        self.hot_embeddings = []

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
        self.log(f"  numpy: {'OK' if np is not None else 'MISSING'}")
        self.log(f"  pynvml: {'OK' if pynvml else 'MISSING'}")
        self.log(f"  pycuda: {'OK' if pycuda_driver else 'MISSING'}")
        self.log(f"  zmq: {'OK' if zmq else 'MISSING'}")
        self.log(f"  onnxruntime: {'OK' if ort else 'MISSING'}")
        self.log(f"  torch: {'OK' if torch else 'MISSING'}")

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
            cache_a = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            cache_b = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            cache_c = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            cache_d = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            cache_e = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )
            cache_f = VRAMCacheManager(
                logger=self.log,
                pycuda_driver=pycuda_driver,
                pycuda_autoinit=pycuda_autoinit,
            )

            raid_mode = RaidMode.RAID0  # or RaidMode.RAID1

            self.cache = RaidCacheManager(
                logger=self.log,
                mode=raid_mode,
                dev_a=cache_a,
                dev_b=cache_b,
                dev_c=cache_c,
                dev_d=cache_d,
                dev_e=cache_e,
                dev_f=cache_f,
            )
            self.cache.start()

            # cluster agent attached to first spine (others share its view)
            cache_a.cluster_agent = ClusterAgent(
                gui=self,
                node_id="node-A",
                bind_addr="tcp://0.0.0.0:5555",
                peers=[],  # add peer PUB endpoints in real cluster
            )
            self.cluster_agent = cache_a.cluster_agent

            self.file_cache = CachedFileInterface(self.cache)
            self.preloader = DirectoryPreloader(
                self.file_cache,
                logger=self.log,
                exts=[
                    ".bin",
                    ".pt",
                    ".pth",
                    ".onnx",
                    ".npy",
                    ".npz",
                    ".pak",
                    ".pak2",
                ],
                max_file_size_mb=2048,
            )

            # memory compression + embedding prefetch organs
            self.memcomp = MemoryCompressionOrgan(self, cache_a)
            self.emb_organ = EmbeddingPrefetchOrgan(self, self.cache)

            # LLM advisor (point to your real model path)
            llm_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "ramnuke_llm.onnx",
            )
            self.llm_advisor = LLMAdvisorOrgan(
                gui=self,
                model_path=llm_model_path,
                backend="onnx",
                interval_sec=30,
            )

            self.log(
                f"[Ramnuke][ENGINE] RAID {raid_mode.value} (6-way) cache manager + app-level interfaces are active."
            )
        except Exception as e:
            self.log(f"[Ramnuke][ENGINE] Failed to start RAID cache manager: {e}")
            self.log(traceback.format_exc())

    def _update_raid_layout_visualizer(self):
        if not isinstance(self.cache, RaidCacheManager):
            self.raid_layout_panel.setText("")
            return
        dev_letters = ["A", "B", "C", "D", "E", "F"]
        lines = ["RAID Layout (RAID0 mapping, first 24 blocks):"]
        for block_id in range(24):
            idx = block_id % len(self.cache.devs)
            lines.append(f"  block {block_id:3d} → spine {dev_letters[idx]}")
        self.raid_layout_panel.setText("\n".join(lines))

    def _update_raid_health_panel(self):
        if not isinstance(self.cache, RaidCacheManager):
            self.raid_panel.setText("")
            return
        dev_letters = ["A", "B", "C", "D", "E", "F"]
        lines = ["Spine Health:"]
        for i, dev in enumerate(self.cache.devs):
            snap = dev.telemetry_snapshot()
            frag = dev.vram_fragmentation_score()
            status = "MUTED" if dev.muted else "ACTIVE"
            lines.append(
                f"  {dev_letters[i]}: {status} | health={dev.health_score:.2f} "
                f"| blocks={snap['blocks']} VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']} "
                f"| frag={frag:.2f}"
            )
        self.raid_panel.setText("\n".join(lines))

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

        err = abs(blocks_next - blocks)
        self.pred_error_history.append(err)
        if len(self.pred_error_history) > 200:
            self.pred_error_history.pop(0)

        temp = None
        gpu_load = None
        power = None
        disk_read_bw = None
        disk_write_bw = None

        if isinstance(self.cache, RaidCacheManager):
            dev0 = self.cache.devs[0]
            temp = dev0._gpu_temperature()
            power = dev0._power_state()
            gpu_load = dev0._gpu_load()
            stats = dev0._disk_stats()
            if stats and dev0.prev_disk_stats:
                phys = dev0.disk_io_physics(dev0.prev_disk_stats, stats)
                if phys:
                    disk_read_bw = phys["read_bw"]
                    disk_write_bw = phys["write_bw"]
            dev0.prev_disk_stats = stats
            vram_frag = self.cache.vram_fragmentation_score()
        else:
            temp = self.cache._gpu_temperature()
            power = self.cache._power_state()
            gpu_load = self.cache._gpu_load()
            stats = self.cache._disk_stats()
            if stats and self.cache.prev_disk_stats:
                phys = self.cache.disk_io_physics(self.cache.prev_disk_stats, stats)
                if phys:
                    disk_read_bw = phys["read_bw"]
                    disk_write_bw = phys["write_bw"]
            self.cache.prev_disk_stats = stats
            vram_frag = self.cache.vram_fragmentation_score()

        if gpu_load is not None:
            self.gpu_predictor.update(gpu_load)
            gpu_next = self.gpu_predictor.predict_next() or gpu_load
        else:
            gpu_next = None

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

        if isinstance(self.cache, RaidCacheManager):
            base_appetite, base_ra, base_workers, base_vram_bias, prewarm = (
                self.cache.devs[0].master_policy(state)
            )
        else:
            base_appetite, base_ra, base_workers, base_vram_bias, prewarm = (
                self.cache.master_policy(state)
            )

        # LLM advisory influence
        llm_delta = 0.0
        if self.llm_advisor is not None:
            llm_delta = self.llm_advisor.last_appetite_delta
        base_appetite *= (1.0 + llm_delta)

        app2, ra2, workers2, vram_bias2, preds, conf, meta_state = (
            self.hybrid_brain.decide(
                base_appetite, base_ra, base_workers, base_vram_bias, state
            )
        )

        if isinstance(self.cache, RaidCacheManager):
            for dev in self.cache.devs:
                dev.apply_master_policy(app2, ra2, workers2, vram_bias2)
                dev.adjust_workers()
        else:
            self.cache.apply_master_policy(app2, ra2, workers2, vram_bias2)
            self.cache.adjust_workers()

        ra_final = min(ra2, ra_override)

        self.last_pressure = (
            self.cache.devs[0].fused_pressure(state)
            if isinstance(self.cache, RaidCacheManager)
            else self.cache.fused_pressure(state)
        )
        self.last_appetite = app2

        self.log(
            f"[Ramnuke][TICK][CACHE] blocks={blocks} (EH next≈{blocks_next:.1f}) "
            f"VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']} "
            f"[regime={regime_flag}]"
        )
        self.log(
            f"[Ramnuke][TICK][POLICY] base_app={base_appetite:.2f}→app={app2:.2f} "
            f"RA={ra_final} workers={workers2} vram_bias={vram_bias2:.2f} "
            f"temp={temp}C gpu={gpu_load}% gpu_next={gpu_next} "
            f"frag={vram_frag:.2f} disk_read_bw={disk_read_bw} prewarm={prewarm} "
            f"llm_delta={llm_delta:+.2f}"
        )

        heat = (
            "State Vector / Policy\n"
            f"  temp: {temp} C\n"
            f"  power: {power}\n"
            f"  gpu: {gpu_load}% (next≈{gpu_next})\n"
            f"  blocks: {blocks} trend≈{trend:.1f}\n"
            f"  frag: {vram_frag:.2f}\n"
            f"  disk_read_bw: {disk_read_bw}\n"
            f"  appetite: {app2:.2f}\n"
            f"  read_ahead: {ra_final}\n"
            f"  workers: {workers2}\n"
            f"  vram_bias: {vram_bias2:.2f}\n"
        )
        self.state_panel.setText(heat)

        preds_str = ", ".join(f"{k}:{v:.0f}" for k, v in (preds or {}).items())
        brain_text = (
            "HybridBrain\n"
            f"  meta_state: {meta_state.value}\n"
            f"  confidence: {conf:.2f}\n"
            f"  multi-horizon: {preds_str}\n"
            f"  thresholds: overload={self.hybrid_brain.dynamic_thresholds['overload']:.2f}, "
            f"stable={self.hybrid_brain.dynamic_thresholds['stable']:.2f}\n"
            f"  fingerprint: overload={self.hybrid_brain.behavior_fingerprint['overload_patterns']:.2f}, "
            f"stable={self.hybrid_brain.behavior_fingerprint['stable_patterns']:.2f}, "
            f"beast={self.hybrid_brain.behavior_fingerprint['beast_wins']:.2f}\n"
        )
        self.brain_panel.setText(brain_text)

        # Embedding / advisor tab text
        cluster_press = (
            self.cluster_agent.cluster_pressure() if self.cluster_agent else 0.0
        )
        shared_hot = (
            len(self.cluster_agent.shared_hot_embeddings())
            if self.cluster_agent
            else 0
        )
        emb_text = (
            "Embedding Prefetch / Cluster\n"
            f"  local_hot: {len(self.hot_embeddings)}\n"
            f"  cluster_hot: {shared_hot}\n"
            f"  cluster_pressure: {cluster_press:.3f}\n"
        )
        self.emb_panel.setText(emb_text)

        llm_text = (
            "LLM Advisor\n"
            f"  last_delta: {self.llm_advisor.last_appetite_delta:+.2f}\n"
            f"  last_advice: {self.llm_advisor.last_advice}\n"
        ) if self.llm_advisor else "LLM Advisor\n  status: disabled\n"
        self.llm_panel.setText(llm_text)

        self._update_raid_health_panel()
        self._update_raid_layout_visualizer()


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

