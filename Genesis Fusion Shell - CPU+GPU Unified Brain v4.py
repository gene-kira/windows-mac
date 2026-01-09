#!/usr/bin/env python3
"""
Genesis Fusion Organism - Tkinter Cockpit with Predictive Deep RAM Cache

Tabs:
- Turbo Booster (Twin Cache Organ: VRAM / RAM / Deep RAM, Movidius, sources, camera, SMB)
- Fusion Overview (CPU/GPU/MEM + FusionMode A/B/C)
- Fusion Insights (predictive, altered states, trends, deep RAM commentary)
- Memory Storage (unlimited local/SMB paths)

Dependencies:
    pip install psutil watchdog opencv-python pynvml
    # Optional acceleration:
    pip install cupy-cuda12x openvino

Uses pure tkinter + ttk (standard light theme).
"""

import sys
import os
import math
import time
import threading
import queue
import socket
import json
from collections import defaultdict, deque
import subprocess

# ------------- Auto-install cv2 if missing -------------
try:
    import cv2
except ImportError:
    try:
        print("[auto-install] OpenCV not found. Installing opencv-python...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        import cv2
        print("[auto-install] OpenCV installed.")
    except Exception as e:
        print(f"[auto-install] Failed to install OpenCV: {e}")
        raise

# ------------- Core deps -------------
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import cupy as cp
    GPU_BACKEND_AVAILABLE = True
except Exception:
    cp = None
    GPU_BACKEND_AVAILABLE = False

try:
    from openvino.runtime import Core as OVCore
    MOVIDIUS_AVAILABLE = True
except Exception:
    OVCore = None
    MOVIDIUS_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from enum import Enum, auto

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tkinter.scrolledtext import ScrolledText


# ===========================
# Capability tiers
# ===========================

class CapabilityTier:
    FULL = "full"        # GPU + Movidius
    GPU_ONLY = "gpu_only"
    RAM_ONLY = "ram_only"


CAPABILITY_TIER = CapabilityTier.RAM_ONLY
RAM_ONLY_MODE = True


def detect_capability_tier():
    global CAPABILITY_TIER, RAM_ONLY_MODE

    gpu_ok = False
    vram_ok = False
    movi_ok = False

    if GPU_BACKEND_AVAILABLE and cp is not None:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            if total and total > 0:
                vram_ok = True
        except Exception:
            vram_ok = False

    gpu_ok = GPU_BACKEND_AVAILABLE and vram_ok
    movi_ok = MOVIDIUS_AVAILABLE and (OVCore is not None)

    if gpu_ok and movi_ok:
        CAPABILITY_TIER = CapabilityTier.FULL
    elif gpu_ok and not movi_ok:
        CAPABILITY_TIER = CapabilityTier.GPU_ONLY
    else:
        CAPABILITY_TIER = CapabilityTier.RAM_ONLY

    RAM_ONLY_MODE = (CAPABILITY_TIER == CapabilityTier.RAM_ONLY)
    print(f"[capability] Detected hardware tier: {CAPABILITY_TIER}")


detect_capability_tier()


# ===========================
# Cache modes and telemetry
# ===========================

class CacheMode:
    CALM = "calm"
    AGGRESSIVE = "aggressive"
    PROTECTIVE = "protective"
    ALTERED = "altered"


class BlockTelemetry:
    def __init__(self, size_bytes):
        self.size_bytes = size_bytes
        self.access_count = 0
        self.last_access = time.time()
        self.created_at = self.last_access

    def touch(self):
        self.access_count += 1
        self.last_access = time.time()

    def score(self, now=None):
        if now is None:
            now = time.time()
        age = now - self.created_at
        idle = now - self.last_access
        return (self.access_count + 1) / (1.0 + idle) * math.log1p(age + 1.0)


class CacheBlock:
    def __init__(self, size_bytes, backend, handle):
        self.size_bytes = size_bytes
        self.backend = backend  # "vram" or "ram"
        self.handle = handle
        self.telemetry = BlockTelemetry(size_bytes)


# ===========================
# Core VRAM/RAM/Deep RAM cache
# ===========================

class VramRamCacheCore:
    def __init__(self, mode=CacheMode.CALM):
        self.mode = mode
        self.base_vram_threshold = 0.30
        self.base_vram_migrate_back = 0.25

        self.blocks = {}
        self.next_id = 1
        self._lock = threading.RLock()

        self._vram_total = self._detect_vram_total()
        self._vram_used = 0

        # Normal RAM cache usage (hot/warm)
        self._ram_used = 0

        # Deep RAM cache usage (cold / bulk)
        self._deep_ram_used = 0
        self._deep_blocks = set()  # block_ids that are “deep”

        # Hook to read FusionMode (A/B/C)
        self.fusion_mode_provider = lambda: None  # returns FusionModeEnum or None

        # For predictive deep RAM expansion/shrink commentary
        self.last_deep_state = None  # "low" / "medium" / "high"

    def _detect_vram_total(self):
        if RAM_ONLY_MODE:
            return None
        if not GPU_BACKEND_AVAILABLE:
            return None
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return int(total)
        except Exception:
            return None

    def current_thresholds(self):
        if self.mode == CacheMode.CALM:
            return self.base_vram_threshold, self.base_vram_migrate_back
        if self.mode == CacheMode.AGGRESSIVE:
            return 0.60, 0.50
        if self.mode == CacheMode.PROTECTIVE:
            return 0.20, 0.15
        if self.mode == CacheMode.ALTERED:
            return 0.45, 0.30
        return self.base_vram_threshold, self.base_vram_migrate_back

    def get_stats(self):
        with self._lock:
            if self._vram_total and not RAM_ONLY_MODE:
                vram_used = self._vram_used
            else:
                vram_used = 0
            vram_pct = vram_used / self._vram_total if self._vram_total and not RAM_ONLY_MODE else 0.0

            virtual_mem = psutil.virtual_memory()
            ram_total = virtual_mem.total
            ram_used_hot = self._ram_used
            ram_used_deep = self._deep_ram_used
            ram_used_total = ram_used_hot + ram_used_deep
            ram_pct = ram_used_total / ram_total if ram_total > 0 else 0.0

            vthr, mthr = self.current_thresholds()

            gpu_backend = None
            if (not RAM_ONLY_MODE and GPU_BACKEND_AVAILABLE and cp is not None and self._vram_total):
                gpu_backend = "CuPy"

            return {
                "vram_total": self._vram_total if not RAM_ONLY_MODE else None,
                "vram_used": vram_used,
                "vram_pct": vram_pct,
                "ram_total": ram_total,
                "ram_used": ram_used_total,
                "ram_hot_used": ram_used_hot,
                "ram_deep_used": ram_used_deep,
                "ram_pct": ram_pct,
                "gpu_backend": gpu_backend,
                "mode": self.mode,
                "vram_threshold": vthr,
                "vram_migrate_back": mthr,
            }

    def _vram_pct(self):
        if not self._vram_total or RAM_ONLY_MODE:
            return 0.0
        return self._vram_used / self._vram_total

    def can_use_vram(self):
        if RAM_ONLY_MODE:
            return False
        if not self._vram_total:
            return False
        vthr, _ = self.current_thresholds()
        return self._vram_pct() < vthr

    def _should_use_deep_ram(self, size_bytes: int) -> bool:
        """
        Decide if this allocation should go to deep RAM instead of 'hot' RAM.
        Uses:
          - current fusion mode (A/B/C)
          - system memory pressure
          - allocation size
        """
        vm = psutil.virtual_memory()
        mem_pct = vm.percent
        mode = self.fusion_mode_provider()

        # Small blocks → stay hot
        if size_bytes < 8 * 1024 * 1024:
            return False

        # If system is already under pressure, don’t be greedy
        if mem_pct > 90:
            return False

        if mode is None:
            appetite = 1
        elif mode.name == "A_CONSERVATIVE":
            appetite = 0
        elif mode.name == "B_BALANCED":
            appetite = 1
        else:
            appetite = 2  # C_BEAST

        if appetite == 0:
            return mem_pct < 60
        if appetite == 1:
            return mem_pct < 75
        return mem_pct < 90

    def allocate(self, size_bytes):
        with self._lock:
            backend = "ram"
            handle = None
            is_deep = False

            # Tier 1: VRAM
            if not RAM_ONLY_MODE and GPU_BACKEND_AVAILABLE and self.can_use_vram():
                try:
                    handle = cp.empty((size_bytes,), dtype=cp.uint8)
                    backend = "vram"
                    self._vram_used += size_bytes
                except Exception:
                    backend = "ram"
                    handle = None

            # Tier 2/3: RAM vs Deep RAM
            if backend == "ram":
                is_deep = self._should_use_deep_ram(size_bytes)
                handle = bytearray(size_bytes)
                if is_deep:
                    self._deep_ram_used += size_bytes
                else:
                    self._ram_used += size_bytes

            block_id = self.next_id
            self.next_id += 1
            blk = CacheBlock(size_bytes, backend, handle)
            self.blocks[block_id] = blk
            if backend == "ram" and is_deep:
                self._deep_blocks.add(block_id)
            return block_id

    def free(self, block_id):
        with self._lock:
            blk = self.blocks.pop(block_id, None)
            if blk is None:
                return False

            if blk.backend == "vram":
                try:
                    self._vram_used -= blk.size_bytes
                    del blk.handle
                    if cp is not None:
                        cp._default_memory_pool.free_all_blocks()
                except Exception:
                    pass
            else:
                if block_id in self._deep_blocks:
                    self._deep_blocks.remove(block_id)
                    self._deep_ram_used -= blk.size_bytes
                else:
                    self._ram_used -= blk.size_bytes
                del blk.handle
            return True

    def touch_block(self, block_id):
        with self._lock:
            blk = self.blocks.get(block_id)
            if blk:
                blk.telemetry.touch()

    def migrate_ram_to_vram_if_possible(self, logger=None):
        if RAM_ONLY_MODE:
            return
        if not GPU_BACKEND_AVAILABLE or not self._vram_total:
            return

        with self._lock:
            _, mthr = self.current_thresholds()
            current_pct = self._vram_pct()
            if current_pct >= mthr:
                return

            now = time.time()
            candidates = []
            for bid, blk in self.blocks.items():
                if blk.backend != "ram":
                    continue
                score = blk.telemetry.score(now)
                candidates.append((score, bid))
            candidates.sort(reverse=True)

            vthr, _ = self.current_thresholds()

            for score, bid in candidates:
                blk = self.blocks[bid]
                needed = blk.size_bytes
                new_pct = (self._vram_used + needed) / self._vram_total
                if new_pct >= vthr:
                    continue
                try:
                    new_handle = cp.empty((blk.size_bytes,), dtype=cp.uint8)
                    if bid in self._deep_blocks:
                        self._deep_ram_used -= blk.size_bytes
                        self._deep_blocks.remove(bid)
                    else:
                        self._ram_used -= blk.size_bytes
                    self._vram_used += blk.size_bytes

                    blk.backend = "vram"
                    blk.handle = new_handle
                    if logger:
                        logger(f"[predictive] RAM -> VRAM migration for hot block {bid} (score={score:.2f})")
                    break
                except Exception as e:
                    if logger:
                        logger(f"[predictive] Failed migration for block {bid}: {e}")
                    continue

    def _force_block_to_vram(self, block_id, logger=None):
        if RAM_ONLY_MODE:
            return
        if not GPU_BACKEND_AVAILABLE or not self._vram_total:
            return
        blk = self.blocks.get(block_id)
        if not blk or blk.backend == "vram":
            return
        vthr, _ = self.current_thresholds()
        needed = blk.size_bytes
        new_pct = (self._vram_used + needed) / self._vram_total
        if new_pct >= vthr:
            return
        try:
            new_handle = cp.empty((blk.size_bytes,), dtype=cp.uint8)
            if block_id in self._deep_blocks:
                self._deep_ram_used -= blk.size_bytes
                self._deep_blocks.remove(block_id)
            else:
                self._ram_used -= blk.size_bytes
            self._vram_used += blk.size_bytes
            blk.backend = "vram"
            blk.handle = new_handle
            if logger:
                logger(f"[policy] Forced block {block_id} to VRAM")
        except Exception as e:
            if logger:
                logger(f"[policy] Failed to force block {block_id} to VRAM: {e}")

    def _force_block_to_ram(self, block_id, logger=None):
        blk = self.blocks.get(block_id)
        if not blk or blk.backend == "ram":
            return
        try:
            new_handle = bytearray(blk.size_bytes)
            self._vram_used -= blk.size_bytes
            self._ram_used += blk.size_bytes
            blk.backend = "ram"
            blk.handle = new_handle
            if logger:
                logger(f"[policy] Forced block {block_id} to RAM")
        except Exception as e:
            if logger:
                logger(f"[policy] Failed to force block {block_id} to RAM: {e}")

    def rebalance_deep_ram(self, logger=None):
        """
        Periodic: shrink deep RAM usage based on fusion mode and system memory pressure.
        Also updates an internal deep-state level for insights.
        """
        with self._lock:
            vm = psutil.virtual_memory()
            mem_pct = vm.percent
            mode = self.fusion_mode_provider()

            if mode is None:
                appetite = 1
            elif mode.name == "A_CONSERVATIVE":
                appetite = 0
            elif mode.name == "B_BALANCED":
                appetite = 1
            else:
                appetite = 2  # C_BEAST

            # High pressure or conservative + medium pressure => shrink
            high_pressure = mem_pct > 90 or (appetite == 0 and mem_pct > 75)

            if high_pressure and self._deep_blocks:
                now = time.time()
                candidates = []
                for bid in self._deep_blocks:
                    blk = self.blocks.get(bid)
                    if not blk:
                        continue
                    score = blk.telemetry.score(now)
                    candidates.append((score, bid))
                candidates.sort()  # coldest first

                freed = 0
                target_bytes = self._deep_ram_used * 0.25
                for score, bid in candidates:
                    blk = self.blocks.get(bid)
                    if not blk:
                        continue
                    freed += blk.size_bytes
                    self.free(bid)
                    if logger:
                        logger(f"[deep] Freed deep block {bid} (score={score:.2f}) due to pressure {mem_pct:.1f}%")
                    if freed >= target_bytes:
                        break

            # Update deep-state for insights (low / medium / high)
            if vm.total > 0:
                deep_ratio = self._deep_ram_used / vm.total
            else:
                deep_ratio = 0.0

            if deep_ratio < 0.05:
                state = "low"
            elif deep_ratio < 0.15:
                state = "medium"
            else:
                state = "high"

            self.last_deep_state = state


# ===========================
# GPU-fronted ingestion
# ===========================

class GpuCacheEngine:
    def __init__(self, core: VramRamCacheCore):
        self.core = core

    def ingest_to_gpu(self, data_bytes: bytes) -> int:
        size = len(data_bytes)
        block_id = self.core.allocate(size)
        blk = self.core.blocks[block_id]
        if blk.backend == "vram" and cp is not None and not RAM_ONLY_MODE:
            try:
                gpu_arr = blk.handle
                gpu_arr[:size] = cp.asarray(data_bytes, dtype=cp.uint8)
            except Exception:
                blk.backend = "ram"
                blk.handle = bytearray(data_bytes)
                self.core._vram_used -= blk.size_bytes
                if block_id in self.core._deep_blocks:
                    self.core._deep_ram_used += blk.size_bytes
                else:
                    self.core._ram_used += blk.size_bytes
        else:
            blk.handle[:size] = data_bytes
        blk.telemetry.touch()
        return block_id

    def stats(self):
        return self.core.get_stats()


# ===========================
# Movidius engine
# ===========================

class MovidiusEngine:
    def __init__(self, model_path=None):
        self.available = MOVIDIUS_AVAILABLE and (OVCore is not None)
        self.model_path = model_path or os.environ.get("MOVIDIUS_MODEL", "")
        self.device_name = "MYRIAD"
        self.core = None
        self.compiled_model = None
        self.input_name = None
        self.output_name = None
        self.job_queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._jobs_submitted = 0
        self._jobs_completed = 0
        self._last_error = None
        self._avg_latency = 0.0
        self._latency_samples = 0

        if self.available and self.model_path and CAPABILITY_TIER == CapabilityTier.FULL:
            try:
                self._init_openvino()
            except Exception as e:
                self.available = False
                self._last_error = f"OpenVINO init error: {e}"

        self.worker.start()

    def _init_openvino(self):
        self.core = OVCore()
        model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(model, self.device_name)
        inputs = self.compiled_model.inputs
        outputs = self.compiled_model.outputs
        self.input_name = inputs[0].get_any_name()
        self.output_name = outputs[0].get_any_name()

    def is_available(self):
        return self.available and CAPABILITY_TIER == CapabilityTier.FULL

    def submit_job(self, features: dict, callback):
        if not self.is_available():
            return
        self._jobs_submitted += 1
        self.job_queue.put((features, callback))

    def _worker_loop(self):
        while True:
            features, callback = self.job_queue.get()
            start = time.time()
            try:
                result = self._run_inference(features)
                latency = time.time() - start
                self._latency_samples += 1
                self._avg_latency = (
                    self._avg_latency * (self._latency_samples - 1) + latency
                ) / self._latency_samples
                self._jobs_completed += 1
                callback(features, result)
            except Exception as e:
                self._last_error = str(e)
                callback(features, {"error": str(e)})

    def _run_inference(self, features: dict) -> dict:
        if self.is_available() and self.compiled_model is not None:
            import numpy as np
            vec = [
                float(features.get("size_bytes", 0)),
                float(features.get("access_count", 0)),
                float(features.get("age", 0.0)),
                float(features.get("idle", 0.0)),
                float(features.get("vram_pct", 0.0)),
                float(features.get("ram_pct", 0.0)),
            ]
            arr = np.array([vec], dtype=np.float32)
            infer_req = self.compiled_model.create_infer_request()
            infer_req.set_tensor(self.input_name, arr)
            infer_req.infer()
            out = infer_req.get_tensor(self.output_name).data
            heat_score = float(out[0][0]) if out.size >= 1 else 0.5
            evict_score = float(out[0][1]) if out.size >= 2 else 0.1
            anomaly_score = float(out[0][2]) if out.size >= 3 else 0.0
        else:
            access = float(features.get("access_count", 0))
            age = float(features.get("age", 0.0))
            idle = float(features.get("idle", 0.0))
            vram_pct = float(features.get("vram_pct", 0.0))
            heat_score = (access + 1) / (1.0 + idle) * math.log1p(age + 1.0)
            heat_score = 1.0 - math.exp(-heat_score / 10.0)
            evict_score = float(max(0.0, 1.0 - heat_score))
            anomaly_score = 0.0

        tier_recommendation = "vram" if heat_score >= 0.5 and not RAM_ONLY_MODE else "ram"
        if anomaly_score > 0.8:
            recommended_mode = CacheMode.PROTECTIVE
        elif heat_score > 0.7 and vram_pct < 0.5:
            recommended_mode = CacheMode.AGGRESSIVE
        else:
            recommended_mode = CacheMode.CALM

        return {
            "heat_score": float(heat_score),
            "evict_score": float(evict_score),
            "anomaly_score": float(anomaly_score),
            "tier_recommendation": tier_recommendation,
            "recommended_mode": recommended_mode,
        }

    def stats(self):
        backend = None
        if self.is_available():
            backend = "OpenVINO/MYRIAD"
        return {
            "available": self.is_available(),
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "queue_depth": self.job_queue.qsize(),
            "avg_latency": self._avg_latency,
            "last_error": self._last_error,
            "backend": backend,
        }


# ===========================
# Twin-organ coordinator
# ===========================

class TwinOrganCoordinator:
    def __init__(self, cache_core: VramRamCacheCore, movi_engine: MovidiusEngine, logger=None):
        self.cache_core = cache_core
        self.gpu = GpuCacheEngine(cache_core)
        self.movi = movi_engine
        self.logger = logger or (lambda msg: None)

    def ingest(self, data_bytes: bytes) -> int:
        block_id = self.gpu.ingest_to_gpu(data_bytes)
        features = self._extract_features_for_block(block_id)
        if CAPABILITY_TIER == CapabilityTier.FULL and self.movi.is_available():
            self.movi.submit_job(features, callback=self._on_inference_result)
        return block_id

    def _extract_features_for_block(self, block_id):
        with self.cache_core._lock:
            blk = self.cache_core.blocks.get(block_id)
            if blk is None:
                return {}
            tel = blk.telemetry
            now = time.time()
            stats = self.cache_core.get_stats()
            return {
                "block_id": block_id,
                "size_bytes": blk.size_bytes,
                "access_count": tel.access_count,
                "age": now - tel.created_at,
                "idle": now - tel.last_access,
                "mode": self.cache_core.mode,
                "vram_pct": stats["vram_pct"],
                "ram_pct": stats["ram_pct"],
            }

    def _on_inference_result(self, features, result):
        if "error" in result:
            self.logger(f"[Movidius error] {result['error']}")
            return

        block_id = features.get("block_id")
        if block_id is None:
            return

        heat = result.get("heat_score", 0.0)
        tier = result.get("tier_recommendation")
        new_mode = result.get("recommended_mode")

        if new_mode:
            self.cache_core.mode = new_mode
            self.logger(f"[Movidius] Mode set to {new_mode} (heat={heat:.2f}, tier={tier})")

        with self.cache_core._lock:
            blk = self.cache_core.blocks.get(block_id)
            if blk:
                blk.telemetry.access_count += heat * 2

        if tier == "vram":
            self.cache_core._force_block_to_vram(block_id, logger=self.logger)
        elif tier == "ram":
            self.cache_core._force_block_to_ram(block_id, logger=self.logger)

    def stats(self):
        s = self.cache_core.get_stats()
        ms = self.movi.stats()
        s["movidius"] = ms
        return s


# ===========================
# Brain / Physics engine
# ===========================

class Brain:
    def __init__(self, history_seconds=30, snapshot_interval=1.0):
        self.history = deque(maxlen=int(history_seconds / snapshot_interval))
        self.pressure = defaultdict(float)

    def record_snapshot(self, source_stats, cache_stats):
        self.history.append({
            "time": time.time(),
            "sources": source_stats,
            "cache": cache_stats,
        })
        self._update_pressure(source_stats)

    def _update_pressure(self, source_stats):
        alpha = 0.3
        rates = source_stats.get("rate", {})
        for src, rate in rates.items():
            prev = self.pressure[src]
            self.pressure[src] = (1 - alpha) * prev + alpha * rate

    def get_source_scale(self, source_name):
        p = self.pressure.get(source_name, 0.0)
        norm = p / (100 * 1024)
        norm = max(-1.0, min(1.0, norm))
        scale = 1.0 + 0.5 * norm
        return max(0.5, min(2.0, scale))

    def recommend_mode(self, current_mode, cache_stats):
        vram_pct = cache_stats["vram_pct"]
        ram_pct = cache_stats["ram_pct"]
        total_pressure = sum(self.pressure.values())

        if ram_pct > 0.9 or vram_pct > 0.95:
            return CacheMode.PROTECTIVE
        if total_pressure > 2 * 1024 * 1024 and vram_pct < 0.6:
            return CacheMode.AGGRESSIVE
        if total_pressure < 0.1 * 1024 * 1024 and vram_pct < 0.2:
            return CacheMode.CALM
        return current_mode


# ===========================
# Ingestion bus
# ===========================

class IngestionBus:
    def __init__(self, coordinator: TwinOrganCoordinator, brain: Brain, logger=None):
        self.coordinator = coordinator
        self.brain = brain
        self.logger = logger or (lambda m: None)
        self._lock = threading.Lock()
        self._running = True

        self.source_bytes_total = defaultdict(int)
        self.source_bytes_window = defaultdict(int)
        self.source_rate = defaultdict(float)
        self._last_rate_calc = time.time()

    def stop(self):
        with self._lock:
            self._running = False

    def ingest(self, source_name: str, payload: bytes):
        with self._lock:
            if not self._running:
                return None
        size = len(payload)
        block_id = self.coordinator.ingest(payload)

        self.source_bytes_total[source_name] += size
        self.source_bytes_window[source_name] += size

        self.logger(f"[{source_name}] Ingested {size} bytes as block {block_id}")
        return block_id

    def update_rates_and_feed_brain(self, cache_stats):
        now = time.time()
        dt = now - self._last_rate_calc
        if dt <= 0:
            return
        for src, window_bytes in self.source_bytes_window.items():
            self.source_rate[src] = window_bytes / dt
        self.source_bytes_window.clear()
        self._last_rate_calc = now

        snapshot = {
            "bytes_total": dict(self.source_bytes_total),
            "rate": dict(self.source_rate),
        }
        self.brain.record_snapshot(snapshot, cache_stats)

    def stats(self):
        with self._lock:
            return {
                "bytes_total": dict(self.source_bytes_total),
                "rate": dict(self.source_rate),
            }


# ===========================
# Adapters
# ===========================

class FileIngestionHandler(FileSystemEventHandler):
    def __init__(self, bus: IngestionBus, watch_name: str):
        self.bus = bus
        self.watch_name = watch_name

    def on_created(self, event):
        if event.is_directory:
            return
        try:
            with open(event.src_path, "rb") as f:
                data = f.read()
            name = f"{self.watch_name}:{os.path.basename(event.src_path)}"
            self.bus.ingest(name, data)
        except Exception as e:
            self.bus.logger(f"[file] error reading {event.src_path}: {e}")


def start_file_adapter(bus: IngestionBus, path: str, watch_name="file_drop"):
    handler = FileIngestionHandler(bus, watch_name)
    observer = Observer()
    observer.schedule(handler, path, recursive=False)
    observer.daemon = True
    observer.start()
    bus.logger(f"[file] watching folder: {path}")
    return observer


def start_tcp_ingest_server(bus: IngestionBus, host="0.0.0.0", port=5000, source_name="tcp_ingest"):
    def worker():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(5)
        bus.logger(f"[net] TCP ingest server on {host}:{port}")
        while True:
            conn, addr = srv.accept()
            label = f"{source_name}@{addr[0]}:{addr[1]}"
            bus.logger(f"[net] connection from {addr}")
            with conn:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    bus.ingest(label, chunk)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def start_system_net_adapter(bus: IngestionBus, brain: Brain, source_name="system_net"):
    def worker():
        prev = psutil.net_io_counters()
        bus.logger("[system_net] system-wide network monitor started")
        while True:
            time.sleep(1.0)
            curr = psutil.net_io_counters()
            delta = (curr.bytes_sent - prev.bytes_sent) + (curr.bytes_recv - prev.bytes_recv)
            prev = curr
            if delta <= 0:
                continue
            size = min(delta, 64 * 1024)
            scale = brain.get_source_scale(source_name)
            size = int(size * scale)
            size = max(256, min(size, 64 * 1024))
            payload = bytes([0x5A]) * size
            bus.ingest(source_name, payload)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def start_system_metrics_adapter(bus: IngestionBus,
                                 brain: Brain,
                                 cpu_source="cpu",
                                 ram_source="ram",
                                 disk_source="disk",
                                 gpu_source="gpu_vram"):
    def worker():
        prev_disk = psutil.disk_io_counters()
        prev_gpu_used = None
        if GPU_BACKEND_AVAILABLE and not RAM_ONLY_MODE:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                prev_gpu_used = total - free
            except Exception:
                prev_gpu_used = None

        bus.logger("[metrics] system metrics monitor started")

        while True:
            time.sleep(1.0)

            cpu_pct = psutil.cpu_percent(interval=None)
            base_cpu_size = int(max(1_024, min(16 * 1024, cpu_pct * 256)))
            cpu_scale = brain.get_source_scale(cpu_source)
            cpu_size = int(base_cpu_size * cpu_scale)
            bus.ingest(cpu_source, bytes([0xC1]) * cpu_size)

            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            base_ram_size = int(max(1_024, min(16 * 1024, ram_pct * 256)))
            ram_scale = brain.get_source_scale(ram_source)
            ram_size = int(base_ram_size * ram_scale)
            bus.ingest(ram_source, bytes([0xA1]) * ram_size)

            curr_disk = psutil.disk_io_counters()
            disk_delta = (curr_disk.read_bytes - prev_disk.read_bytes) + \
                         (curr_disk.write_bytes - prev_disk.write_bytes)
            prev_disk = curr_disk
            if disk_delta > 0:
                base_disk_size = int(max(1_024, min(32 * 1024, disk_delta)))
                disk_scale = brain.get_source_scale(disk_source)
                disk_size = int(base_disk_size * disk_scale)
                bus.ingest(disk_source, bytes([0xD1]) * disk_size)

            if GPU_BACKEND_AVAILABLE and not RAM_ONLY_MODE:
                try:
                    free, total = cp.cuda.runtime.memGetInfo()
                    used = total - free
                    if prev_gpu_used is None:
                        prev_gpu_used = used
                    delta_gpu = used - prev_gpu_used
                    prev_gpu_used = used
                    if delta_gpu != 0:
                        base_gpu_size = int(max(1_024, min(16 * 1024, abs(delta_gpu))))
                        gpu_scale = brain.get_source_scale(gpu_source)
                        gpu_size = int(base_gpu_size * gpu_scale)
                        bus.ingest(gpu_source, bytes([0x61]) * gpu_size)
                except Exception:
                    pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def start_camera_adapter(bus: IngestionBus, stop_event: threading.Event, camera_index=0, source_name="camera_0"):
    def worker():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            bus.logger(f"[camera] cannot open camera {camera_index}")
            return
        bus.logger(f"[camera] started camera {camera_index}")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            data = buf.tobytes()
            bus.ingest(source_name, data)
        cap.release()
        bus.logger(f"[camera] camera {camera_index} stopped")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def start_smb_adapter(bus: IngestionBus, brain: Brain, smb_path: str, source_name: str, interval=3.0):
    smb_path = os.path.normpath(smb_path)

    def worker():
        bus.logger(f"[smb] starting SMB scanner for {smb_path} as {source_name}")
        seen = {}
        while True:
            time.sleep(interval)
            try:
                if not os.path.isdir(smb_path):
                    bus.logger(f"[smb] path not available: {smb_path}")
                    continue
                for entry in os.scandir(smb_path):
                    if not entry.is_file():
                        continue
                    full_path = entry.path
                    try:
                        st = entry.stat()
                    except Exception:
                        continue
                    key = full_path
                    cur = (st.st_size, st.st_mtime)
                    prev = seen.get(key)
                    if prev is not None and prev == cur:
                        continue
                    seen[key] = cur
                    max_read = 8 * 1024 * 1024
                    try:
                        with open(full_path, "rb") as f:
                            data = f.read(max_read)
                        scale = brain.get_source_scale(source_name)
                        if scale < 1.0:
                            new_len = max(1, int(len(data) * scale))
                            data = data[:new_len]
                        src_label = f"{source_name}:{os.path.basename(full_path)}"
                        bus.ingest(src_label, data)
                    except Exception as e:
                        bus.logger(f"[smb] error reading {full_path}: {e}")
            except Exception as e:
                bus.logger(f"[smb] scanner error for {smb_path}: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


# ===========================
# Fusion mode + Insights + Memory
# ===========================

class FusionModeEnum(Enum):
    A_CONSERVATIVE = auto()
    B_BALANCED = auto()
    C_BEAST = auto()


class FusionModeController:
    def __init__(self):
        self.current_mode = FusionModeEnum.B_BALANCED
        self.recent_errors = 0
        self.recent_throttles = 0
        self.recent_successful_offloads = 0
        self.recent_failed_offloads = 0
        self.last_mode_change = time.time()

    def register_error(self):
        self.recent_errors += 1

    def register_throttle(self):
        self.recent_throttles += 1

    def register_offload_result(self, success: bool):
        if success:
            self.recent_successful_offloads += 1
        else:
            self.recent_failed_offloads += 1

    def decay_counters(self):
        self.recent_errors = max(0, self.recent_errors - 1)
        self.recent_throttles = max(0, self.recent_throttles - 1)
        self.recent_successful_offloads = max(0, self.recent_successful_offloads - 1)
        self.recent_failed_offloads = max(0, self.recent_failed_offloads - 1)

    def update_mode(self, cpu_temp: float, gpu_temp: float,
                    cpu_load: float, gpu_load: float,
                    context: str):
        safety_risk = 0
        safety_risk += self.recent_errors * 3
        safety_risk += self.recent_throttles * 2

        if cpu_temp > 80 or gpu_temp > 80:
            safety_risk += 3
        if cpu_temp > 90 or gpu_temp > 90:
            safety_risk += 5

        demand = 0
        if context in ("GAMING", "RENDERING"):
            demand += 4
        elif context in ("HEAVY_COMPUTE",):
            demand += 5
        elif context in ("BROWSING", "IDLE"):
            demand += 1

        if cpu_load > 80:
            demand += 2
        if gpu_load < 50 and cpu_load > 70:
            demand += 2

        confidence = self.recent_successful_offloads - self.recent_failed_offloads

        now = time.time()
        min_mode_hold = 30
        if now - self.last_mode_change < min_mode_hold:
            return

        new_mode = FusionModeEnum.B_BALANCED

        if safety_risk >= 5:
            new_mode = FusionModeEnum.A_CONSERVATIVE
        else:
            if demand >= 6 and confidence >= 0:
                new_mode = FusionModeEnum.C_BEAST
            elif demand >= 3:
                new_mode = FusionModeEnum.B_BALANCED
            else:
                new_mode = FusionModeEnum.A_CONSERVATIVE if confidence < 3 else FusionModeEnum.B_BALANCED

        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.last_mode_change = now


class SystemState(Enum):
    CALM = "Calm"
    STRAINED = "Strained"
    OVERLOADED = "Overloaded"


class InsightsEngine:
    def __init__(self):
        self.insights = []
        self.last_state = None
        self.last_deep_comment_state = None  # low / medium / high

    def _trend(self, data):
        if len(data) < 4:
            return 0.0
        y = data[-10:] if len(data) >= 10 else data[:]
        n = len(y)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(y) / n
        num = sum((xs[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        return num / den

    def _classify_state(self, cpu, gpu, mem):
        if cpu < 40 and gpu < 30 and mem < 70:
            return SystemState.CALM
        if cpu < 80 and gpu < 80 and mem < 90:
            return SystemState.STRAINED
        return SystemState.OVERLOADED

    def update(self, cpu_history, gpu_history, mem_history,
               mode: FusionModeEnum,
               deep_state: str):
        if not cpu_history or not mem_history:
            return

        cpu_now = cpu_history[-1]
        mem_now = mem_history[-1]
        gpu_now = gpu_history[-1] if gpu_history else 0.0

        cpu_trend = self._trend(cpu_history)
        gpu_trend = self._trend(gpu_history) if gpu_history else 0.0
        mem_trend = self._trend(mem_history)

        state = self._classify_state(cpu_now, gpu_now, mem_now)

        if state != self.last_state:
            if state == SystemState.CALM:
                self._record("State shift: Calm — resources comfortably within limits.")
            elif state == SystemState.STRAINED:
                self._record("State shift: Strained — elevated but manageable load.")
            elif state == SystemState.OVERLOADED:
                self._record("State shift: Overloaded — CPU/GPU/MEM pressures are high.")
            self.last_state = state

        horizon = 30

        def project(value, slope, steps):
            return value + slope * steps

        cpu_future = max(0.0, min(100.0, project(cpu_now, cpu_trend, horizon)))
        mem_future = max(0.0, min(100.0, project(mem_now, mem_trend, horizon)))
        gpu_future = max(0.0, min(100.0, project(gpu_now, gpu_trend, horizon)))

        if cpu_trend > 0.5 and cpu_future > 85:
            self._record("Prediction: CPU load rising, may exceed 85% soon.")
        if mem_trend > 0.3 and mem_future > 90:
            self._record("Prediction: Memory pressure building, may exceed 90%.")
        if gpu_trend > 0.5 and gpu_future > 80:
            self._record("Prediction: GPU usage rising, may exceed 80%.")

        if mode == FusionModeEnum.C_BEAST and state == SystemState.CALM:
            self._record("Observation: Beast Mode in Calm state — safe to push hard.")
        elif mode == FusionModeEnum.A_CONSERVATIVE and state == SystemState.OVERLOADED:
            self._record("Observation: Conservative Mode in Overloaded state — prioritizing stability.")

        # Deep RAM commentary (predictive / best guess)
        if deep_state != self.last_deep_comment_state:
            if deep_state == "low":
                self._record(
                    "Deep RAM cache: light footprint. System has ample headroom; "
                    "expect future expansions when idle memory appears."
                )
            elif deep_state == "medium":
                self._record(
                    "Deep RAM cache: moderate footprint. Balancing between throughput and safety. "
                    "Best guess: memory pyramid is healthy."
                )
            elif deep_state == "high":
                self._record(
                    "Deep RAM cache: large footprint. Cache is aggressively using idle memory. "
                    "Best guess: if pressure rises, expect automatic shrink."
                )
            self.last_deep_comment_state = deep_state

    def _record(self, text):
        ts = time.strftime("%H:%M:%S")
        self.insights.append(f"[{ts}] {text}")
        if len(self.insights) > 300:
            self.insights.pop(0)

    def get_insights_text(self):
        if not self.insights:
            return "No insights yet. Let the system run under load."
        return "\n".join(self.insights)


# ===========================
# MemoryManager
# ===========================

CONFIG_FILE = "memory_paths.json"


def is_smb_path(path: str) -> bool:
    return path.strip().startswith("\\\\")


def get_path_type(path: str) -> str:
    return "SMB" if is_smb_path(path) else "Local"


class MemoryManager:
    def __init__(self, logger=None):
        self.logger = logger or (lambda m: None)
        self.paths = []
        self._load_config()

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.paths = data
                    for entry in self.paths:
                        entry.setdefault("type", get_path_type(entry.get("path", "")))
                        entry.setdefault("status", "Unknown")
                        entry.setdefault("last_write", "N/A")
                self.logger(f"MemoryManager: loaded {len(self.paths)} paths.")
            except Exception as e:
                self.logger(f"MemoryManager: load error: {e}")
        else:
            self.logger("MemoryManager: no config found, starting new.")

    def _save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.paths, f, indent=2)
            self.logger("MemoryManager: saved config.")
        except Exception as e:
            self.logger(f"MemoryManager: save error: {e}")

    def get_paths(self):
        return self.paths

    def add_path(self, path: str):
        path = path.strip()
        if not path:
            return
        if any(p["path"].lower() == path.lower() for p in self.paths):
            self.logger(f"MemoryManager: duplicate path {path}")
            return
        self.paths.append({
            "path": path,
            "type": get_path_type(path),
            "status": "Unknown",
            "last_write": "N/A",
        })
        self._save_config()
        self.logger(f"MemoryManager: added {path}")

    def remove_path(self, idx: int):
        if 0 <= idx < len(self.paths):
            removed = self.paths.pop(idx)
            self._save_config()
            self.logger(f"MemoryManager: removed {removed['path']}")

    def edit_path(self, idx: int, new_path: str):
        if 0 <= idx < len(self.paths):
            new_path = new_path.strip()
            if not new_path:
                return
            self.paths[idx]["path"] = new_path
            self.paths[idx]["type"] = get_path_type(new_path)
            self.paths[idx]["status"] = "Unknown"
            self.paths[idx]["last_write"] = "N/A"
            self._save_config()
            self.logger(f"MemoryManager: edited to {new_path}")

    def test_path(self, idx: int) -> bool:
        if not (0 <= idx < len(self.paths)):
            return False
        entry = self.paths[idx]
        base = entry["path"]
        try:
            if not is_smb_path(base):
                if not os.path.exists(base):
                    os.makedirs(base, exist_ok=True)
            test_file = os.path.join(base, f"genesis_test_{int(time.time())}.tmp")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            entry["status"] = "Online"
            entry["last_write"] = time.strftime("%H:%M:%S")
            self._save_config()
            self.logger(f"MemoryManager: test OK {base}")
            return True
        except Exception as e:
            entry["status"] = "Error"
            self._save_config()
            self.logger(f"MemoryManager: test FAIL {base}: {e}")
            return False

    def save_snapshot(self, payload: str):
        if not self.paths:
            self.logger("MemoryManager: no paths configured, skip snapshot.")
            return
        filename = time.strftime("genesis_memory_%Y%m%d.log")
        for entry in self.paths:
            base = entry["path"]
            try:
                if not is_smb_path(base):
                    if not os.path.exists(base):
                        os.makedirs(base, exist_ok=True)
                full_path = os.path.join(base, filename)
                with open(full_path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {payload}\n")
                entry["status"] = "Online"
                entry["last_write"] = time.strftime("%H:%M:%S")
                self.logger(f"MemoryManager: wrote snapshot to {full_path}")
            except Exception as e:
                entry["status"] = "Error"
                self.logger(f"MemoryManager: write fail {base}: {e}")
        self._save_config()


# ===========================
# GPU telemetry helper
# ===========================

class GPUInfo:
    def __init__(self):
        self.available = NVML_AVAILABLE
        self.handle = None
        self.name = "No GPU"
        if self.available:
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count > 0:
                    self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.name = pynvml.nvmlDeviceGetName(self.handle).decode("utf-8", errors="ignore")
                else:
                    self.available = False
            except Exception:
                self.available = False

    def usage(self) -> float:
        if not self.available or self.handle is None:
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def temp(self) -> float:
        if not self.available or self.handle is None:
            return 0.0
        try:
            return float(pynvml.nvmlDeviceGetTemperature(
                self.handle,
                pynvml.NVML_TEMPERATURE_GPU
            ))
        except Exception:
            return 0.0

    def shutdown(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ===========================
# Tkinter Tabs
# ===========================

class TurboBoosterTab(ttk.Frame):
    def __init__(self, parent, orchestrator, cache_core, bus, brain, log_func):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.cache_core = cache_core
        self.bus = bus
        self.brain = brain
        self.log = log_func

        self._allocated_ids = []
        self.camera_running = False
        self.camera_thread = None
        self.camera_stop_event = None
        self.camera_index = 0
        self._smb_sources = []

        self._build_ui()
        self._schedule_update()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        tier_text = {
            CapabilityTier.FULL: "Full (GPU + Movidius)",
            CapabilityTier.GPU_ONLY: "GPU-only",
            CapabilityTier.RAM_ONLY: "RAM-only",
        }.get(CAPABILITY_TIER, CAPABILITY_TIER)

        self.lbl_tier = ttk.Label(self, text=f"Hardware Tier: {tier_text}")
        self.lbl_tier.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # VRAM bar
        self.lbl_vram = ttk.Label(self, text="VRAM: not available")
        self.pb_vram = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.lbl_vram.grid(row=1, column=0, sticky="we", padx=5)
        self.pb_vram.grid(row=2, column=0, sticky="we", padx=5)

        # RAM (hot+deep) bar
        self.lbl_ram = ttk.Label(self, text="RAM Cache: 0 / 0")
        self.pb_ram = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.lbl_ram.grid(row=3, column=0, sticky="we", padx=5)
        self.pb_ram.grid(row=4, column=0, sticky="we", padx=5)

        # Dedicated DEEP RAM bar
        self.lbl_deep = ttk.Label(self, text="Deep RAM: 0 MB")
        self.pb_deep = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.lbl_deep.grid(row=5, column=0, sticky="we", padx=5)
        self.pb_deep.grid(row=6, column=0, sticky="we", padx=5)

        self.lbl_status = ttk.Label(self, text="Status: idle")
        self.lbl_backend = ttk.Label(self, text="GPU backend: none")
        self.lbl_movi = ttk.Label(self, text="Movidius: none")
        self.lbl_brain = ttk.Label(self, text="Brain: observing")
        self.lbl_camera = ttk.Label(self, text="Camera: OFF")

        self.lbl_status.grid(row=7, column=0, sticky="w", padx=5)
        self.lbl_backend.grid(row=8, column=0, sticky="w", padx=5)
        self.lbl_movi.grid(row=9, column=0, sticky="w", padx=5)
        self.lbl_brain.grid(row=10, column=0, sticky="w", padx=5)
        self.lbl_camera.grid(row=11, column=0, sticky="w", padx=5)

        mode_frame = ttk.Frame(self)
        mode_frame.grid(row=12, column=0, sticky="we", padx=5, pady=5)
        ttk.Label(mode_frame, text="Cache Mode:").pack(side="left")
        self.mode_var = tk.StringVar(value=self.cache_core.mode)
        self.cbo_mode = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=[CacheMode.CALM, CacheMode.AGGRESSIVE, CacheMode.PROTECTIVE, CacheMode.ALTERED],
            state="readonly",
            width=15,
        )
        self.cbo_mode.pack(side="left", padx=5)
        self.cbo_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=13, column=0, sticky="we", padx=5, pady=5)
        ttk.Button(btn_frame, text="Test ingest 64MB",
                   command=lambda: self._ingest_test(64 * 1024 * 1024)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Test ingest 256MB",
                   command=lambda: self._ingest_test(256 * 1024 * 1024)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Free all test blocks",
                   command=self._free_all).pack(side="left", padx=2)

        cam_frame = ttk.Frame(self)
        cam_frame.grid(row=14, column=0, sticky="we", padx=5, pady=5)
        self.btn_camera = ttk.Button(cam_frame, text="Start Camera", command=self._toggle_camera)
        ttk.Label(cam_frame, text="Camera control:").pack(side="left")
        self.btn_camera.pack(side="left", padx=5)

        smb_frame = ttk.Frame(self)
        smb_frame.grid(row=15, column=0, sticky="we", padx=5, pady=5)
        ttk.Label(smb_frame, text="SMB:").pack(side="left")
        ttk.Button(smb_frame, text="Add SMB Source", command=self._add_smb_source).pack(side="left", padx=5)

        ttk.Label(self, text="Sources:").grid(row=16, column=0, sticky="w", padx=5)
        self.txt_sources = ScrolledText(self, height=8, wrap="none")
        self.txt_sources.grid(row=17, column=0, sticky="nsew", padx=5)
        self.rowconfigure(17, weight=1)

        ttk.Label(self, text="Event log:").grid(row=18, column=0, sticky="w", padx=5)
        self.txt_log = ScrolledText(self, height=6, wrap="word")
        self.txt_log.grid(row=19, column=0, sticky="nsew", padx=5, pady=(0, 5))

    def _schedule_update(self):
        self._update()
        self.after(1000, self._schedule_update)

    def _on_mode_change(self, event=None):
        mode = self.mode_var.get()
        self.cache_core.mode = mode
        self.log(f"[ui] Cache mode set to {mode}")

    def _ingest_test(self, size_bytes: int):
        self.log(f"[ui] Test ingest {size_bytes // (1024*1024)} MB")
        payload = bytes([0xEE]) * size_bytes
        block_id = self.orchestrator.ingest(payload)
        self._allocated_ids.append(block_id)

    def _free_all(self):
        self.log("[ui] Free all test blocks")
        for bid in self._allocated_ids:
            self.cache_core.free(bid)
        self._allocated_ids.clear()

    def _toggle_camera(self):
        if not self.camera_running:
            self.camera_stop_event = threading.Event()
            self.camera_thread = start_camera_adapter(self.bus, self.camera_stop_event,
                                                      camera_index=self.camera_index,
                                                      source_name=f"camera_{self.camera_index}")
            self.camera_running = True
            self.btn_camera.configure(text="Stop Camera")
            self.lbl_camera.configure(text=f"Camera: ON (index {self.camera_index})")
            self.log(f"[camera] Started camera index {self.camera_index}")
        else:
            if self.camera_stop_event is not None:
                self.camera_stop_event.set()
            self.camera_running = False
            self.btn_camera.configure(text="Start Camera")
            self.lbl_camera.configure(text="Camera: OFF")
            self.log(f"[camera] Stopped camera index {self.camera_index}")

    def _add_smb_source(self):
        path = filedialog.askdirectory(title="Select SMB/Network Folder", initialdir="\\\\")
        if not path:
            return
        path = os.path.normpath(path)
        source_name = f"smb{len(self._smb_sources)+1}"
        self._smb_sources.append((path, source_name))
        self.log(f"[smb] Added SMB source {source_name} -> {path}")
        start_smb_adapter(self.bus, self.brain, path, source_name, interval=3.0)

    def _update(self):
        try:
            self.cache_core.migrate_ram_to_vram_if_possible(logger=self.log)
            cache_stats = self.orchestrator.stats()
            self.bus.update_rates_and_feed_brain(cache_stats)

            self.cache_core.rebalance_deep_ram(logger=self.log)

            if self.cache_core.mode == CacheMode.ALTERED:
                new_mode = self.brain.recommend_mode(self.cache_core.mode, cache_stats)
                if new_mode != self.cache_core.mode:
                    self.cache_core.mode = new_mode
                    self.mode_var.set(new_mode)
                    self.log(f"[Brain] Mode auto-changed to {new_mode}")

            self._update_ui_from_stats(cache_stats, self.bus.stats())
        except Exception as e:
            self.log(f"[timer] error: {e}")

    def _update_ui_from_stats(self, cache_stats, bus_stats):
        vram_total = cache_stats.get("vram_total")
        vram_used = cache_stats.get("vram_used", 0)
        vram_pct = cache_stats.get("vram_pct", 0.0)
        ram_total = cache_stats.get("ram_total", 0)
        ram_used = cache_stats.get("ram_used", 0)
        ram_pct = cache_stats.get("ram_pct", 0.0)
        ram_hot = cache_stats.get("ram_hot_used", 0)
        ram_deep = cache_stats.get("ram_deep_used", 0)

        if vram_total:
            self.lbl_vram.configure(
                text=f"VRAM: {vram_used/1024/1024:.1f} MB / {vram_total/1024/1024:.1f} MB ({vram_pct*100:.1f}%)"
            )
            self.pb_vram["value"] = vram_pct * 100
        else:
            self.lbl_vram.configure(text="VRAM: not available")
            self.pb_vram["value"] = 0

        self.lbl_ram.configure(
            text=(
                f"RAM Cache: {ram_used/1024/1024:.1f} MB / {ram_total/1024/1024:.1f} MB "
                f"({ram_pct*100:.2f}%)  "
                f"[hot={ram_hot/1024/1024:.1f} MB, deep={ram_deep/1024/1024:.1f} MB]"
            )
        )
        self.pb_ram["value"] = ram_pct * 100

        # Deep RAM bar shows ratio of deep usage vs total RAM
        deep_ratio = (ram_deep / ram_total) if ram_total > 0 else 0.0
        self.pb_deep["value"] = deep_ratio * 100
        self.lbl_deep.configure(
            text=f"Deep RAM: {ram_deep/1024/1024:.1f} MB ({deep_ratio*100:.1f}% of system RAM)"
        )

        # Color-coded indicator on Deep RAM label:
        if deep_ratio < 0.05:
            self.lbl_deep.configure(foreground="black")
        elif deep_ratio < 0.15:
            self.lbl_deep.configure(foreground="orange")
        else:
            self.lbl_deep.configure(foreground="red")

        mode = cache_stats.get("mode")
        vthr = cache_stats.get("vram_threshold", 0.3)

        if CAPABILITY_TIER == CapabilityTier.RAM_ONLY:
            status = f"RAM-only mode ({mode})"
        elif CAPABILITY_TIER == CapabilityTier.GPU_ONLY:
            status = f"GPU-only mode ({mode})"
        else:
            if vram_total and vram_pct < vthr:
                status = f"VRAM-first ({mode}, below {vthr*100:.0f}%)"
            else:
                status = f"Spillover to RAM ({mode})"

        self.lbl_status.configure(text=f"Status: {status}")

        backend = cache_stats.get("gpu_backend") or "none"
        self.lbl_backend.configure(text=f"GPU backend: {backend}")

        movi = cache_stats.get("movidius", {})
        if CAPABILITY_TIER != CapabilityTier.FULL:
            self.lbl_movi.configure(text="Movidius: disabled")
        else:
            movi_backend = movi.get("backend") or "none"
            qd = movi.get("queue_depth", 0)
            jl = movi.get("jobs_completed", 0)
            lat = movi.get("avg_latency", 0.0)
            err = movi.get("last_error")
            txt = f"Movidius: {movi_backend}, jobs={jl}, queue={qd}, avg_latency={lat*1000:.1f} ms"
            if err:
                txt += f" (err: {err})"
            self.lbl_movi.configure(text=txt)

        total_map = bus_stats.get("bytes_total", {})
        rate_map = bus_stats.get("rate", {})
        if total_map:
            lines = []
            for src in sorted(total_map.keys()):
                total_mb = total_map[src] / (1024**2)
                rate_kb = rate_map.get(src, 0.0) / 1024.0
                p = self.brain.pressure.get(src, 0.0) / 1024.0
                lines.append(f"{src}: {total_mb:.2f} MB, {rate_kb:.1f} KB/s, pressure={p:.1f}")
            if len(lines) > 100:
                lines = lines[-100:]
            text = "\n".join(lines)
        else:
            text = "(no data yet)"
        self.txt_sources.delete("1.0", "end")
        self.txt_sources.insert("1.0", text)

        total_pressure = sum(self.brain.pressure.values())
        if total_pressure < 0.1 * 1024 * 1024:
            brain_state = "calm water"
        elif total_pressure < 2 * 1024 * 1024:
            brain_state = "flowing"
        else:
            brain_state = "rising flood"
        self.lbl_brain.configure(text=f"Brain: {brain_state}")


class FusionOverviewTab(ttk.Frame):
    def __init__(self, parent, fusion_ctrl, gpu_info, insights_engine, cache_core, log_func):
        super().__init__(parent)
        self.fusion_ctrl = fusion_ctrl
        self.gpu_info = gpu_info
        self.insights_engine = insights_engine
        self.cache_core = cache_core
        self.log = log_func

        self.cpu_history = []
        self.gpu_history = []
        self.mem_history = []

        self._build_ui()
        self._schedule_update()
        self._schedule_decay()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        self.lbl_cpu = ttk.Label(self, text="CPU: -- %")
        self.lbl_mem = ttk.Label(self, text="Memory: -- %")
        self.lbl_gpu = ttk.Label(self, text="GPU: -- %")
        self.lbl_cpu_temp = ttk.Label(self, text="CPU Temp: -- °C")
        self.lbl_gpu_temp = ttk.Label(self, text="GPU Temp: -- °C")
        self.lbl_mode = ttk.Label(self, text="Fusion Mode: --")
        self.lbl_context = ttk.Label(self, text="Context: --")

        row = 0
        for w in [self.lbl_cpu, self.lbl_mem, self.lbl_gpu,
                  self.lbl_cpu_temp, self.lbl_gpu_temp,
                  self.lbl_mode, self.lbl_context]:
            w.grid(row=row, column=0, sticky="w", padx=5, pady=3)
            row += 1

    def _schedule_update(self):
        self._update()
        self.after(1000, self._schedule_update)

    def _schedule_decay(self):
        self.fusion_ctrl.decay_counters()
        self.after(60000, self._schedule_decay)

    def _cpu_temp(self):
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                for t in entries:
                    label = (t.label or "").lower()
                    if "cpu" in label or "core" in label:
                        return float(t.current)
            return 0.0
        except Exception:
            return 0.0

    def _estimate_context(self, cpu, gpu):
        if cpu < 20 and gpu < 10:
            return "IDLE"
        if gpu > 50 and cpu > 40:
            return "GAMING"
        if cpu > 80 and gpu > 30:
            return "HEAVY_COMPUTE"
        if cpu < 60 and gpu < 40:
            return "BROWSING"
        return "HEAVY_COMPUTE"

    def _update(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu = self.gpu_info.usage() if self.gpu_info else 0.0
        cpu_temp = self._cpu_temp()
        gpu_temp = self.gpu_info.temp() if self.gpu_info else 0.0

        self.cpu_history.append(cpu)
        self.mem_history.append(mem)
        self.gpu_history.append(gpu)
        self.cpu_history = self.cpu_history[-120:]
        self.mem_history = self.mem_history[-120:]
        self.gpu_history = self.gpu_history[-120:]

        context = self._estimate_context(cpu, gpu)
        self.fusion_ctrl.update_mode(cpu_temp, gpu_temp, cpu, gpu, context)

        # Deep state from cache core
        deep_state = self.cache_core.last_deep_state or "low"
        self.insights_engine.update(self.cpu_history, self.gpu_history, self.mem_history,
                                    self.fusion_ctrl.current_mode, deep_state)

        self.lbl_cpu.configure(text=f"CPU: {cpu:.1f} %")
        self.lbl_mem.configure(text=f"Memory: {mem:.1f} % used")
        self.lbl_gpu.configure(text=f"GPU: {gpu:.1f} %")
        self.lbl_cpu_temp.configure(text=f"CPU Temp: {cpu_temp:.1f} °C")
        self.lbl_gpu_temp.configure(text=f"GPU Temp: {gpu_temp:.1f} °C")
        self.lbl_mode.configure(text=f"Fusion Mode: {self.fusion_ctrl.current_mode.name}")
        self.lbl_context.configure(text=f"Context: {context}")


class FusionInsightsTab(ttk.Frame):
    def __init__(self, parent, insights_engine):
        super().__init__(parent)
        self.insights_engine = insights_engine
        self._build_ui()
        self._schedule_update()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text="Fusion Insights (predictive cortex):").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.txt = ScrolledText(self, wrap="word")
        self.txt.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.rowconfigure(1, weight=1)

    def _schedule_update(self):
        self._update()
        self.after(5000, self._schedule_update)

    def _update(self):
        text = self.insights_engine.get_insights_text()
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self.txt.see("end")


class MemoryStorageTab(ttk.Frame):
    def __init__(self, parent, memory_mgr, log_func):
        super().__init__(parent)
        self.memory_mgr = memory_mgr
        self.log = log_func
        self._build_ui()
        self._refresh_table()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self, columns=("path", "type", "status", "last_write"), show="headings")
        self.tree.heading("path", text="Path")
        self.tree.heading("type", text="Type")
        self.tree.heading("status", text="Status")
        self.tree.heading("last_write", text="Last Write")
        self.tree.column("path", width=400, anchor="w")
        self.tree.column("type", width=80, anchor="center")
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("last_write", width=100, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, sticky="we", padx=5, pady=5)
        ttk.Button(btn_frame, text="Add Path", command=self._add_path).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove_path).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Edit", command=self._edit_path).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Test Path", command=self._test_path).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Save Snapshot Now", command=self._snapshot_now).pack(side="left", padx=2)

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, entry in enumerate(self.memory_mgr.get_paths()):
            self.tree.insert(
                "", "end", iid=str(i),
                values=(
                    entry.get("path", ""),
                    entry.get("type", ""),
                    entry.get("status", ""),
                    entry.get("last_write", "N/A"),
                )
            )

    def _selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _add_path(self):
        path = filedialog.askdirectory(title="Select folder or network path")
        if not path:
            return
        self.memory_mgr.add_path(path)
        self._refresh_table()

    def _remove_path(self):
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Remove Path", "Select a path first.")
            return
        self.memory_mgr.remove_path(idx)
        self._refresh_table()

    def _edit_path(self):
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Edit Path", "Select a path first.")
            return
        entry = self.memory_mgr.get_paths()[idx]
        old = entry["path"]
        new = simpledialog.askstring("Edit Path", "Update path:", initialvalue=old)
        if new:
            self.memory_mgr.edit_path(idx, new)
            self._refresh_table()

    def _test_path(self):
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Test Path", "Select a path first.")
            return
        ok = self.memory_mgr.test_path(idx)
        self._refresh_table()
        if ok:
            messagebox.showinfo("Test Path", "Path is online and writable.")
        else:
            messagebox.showwarning("Test Path", "Path is not writable or offline.")

    def _snapshot_now(self):
        payload = "Manual snapshot from MemoryStorage tab."
        self.memory_mgr.save_snapshot(payload)
        self._refresh_table()
        messagebox.showinfo("Snapshot", "Snapshot saved to all online paths.")


# ===========================
# Main window
# ===========================

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genesis Fusion Organism - Tkinter Cockpit")
        self.geometry("1300x800")

        self.log_widget = ScrolledText(self, height=6, wrap="word")
        self.log_widget.pack(side="bottom", fill="x")

        def logger(msg: str):
            ts = time.strftime("%H:%M:%S")
            self.log_widget.insert("end", f"[{ts}] {msg}\n")
            self.log_widget.see("end")

        self.cache_core = VramRamCacheCore()
        self.movi_engine = MovidiusEngine()
        self.brain = Brain()
        self.coordinator = TwinOrganCoordinator(self.cache_core, self.movi_engine, logger=logger)
        self.ingestion_bus = IngestionBus(self.coordinator, self.brain, logger=logger)
        self.gpu_info = GPUInfo()
        self.fusion_ctrl = FusionModeController()
        self.insights_engine = InsightsEngine()
        self.memory_mgr = MemoryManager(logger=logger)

        # Wire fusion mode into cache core for Deep RAM decisions
        self.cache_core.fusion_mode_provider = lambda: self.fusion_ctrl.current_mode

        ingest_folder = os.path.abspath("./ingest_watch")
        os.makedirs(ingest_folder, exist_ok=True)
        start_file_adapter(self.ingestion_bus, ingest_folder, "file_drop")
        start_tcp_ingest_server(self.ingestion_bus)
        start_system_net_adapter(self.ingestion_bus, self.brain)
        start_system_metrics_adapter(self.ingestion_bus, self.brain)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True)

        self.tab_turbo = TurboBoosterTab(self.notebook, self.coordinator, self.cache_core,
                                         self.ingestion_bus, self.brain, log_func=logger)
        self.tab_overview = FusionOverviewTab(self.notebook, self.fusion_ctrl,
                                              self.gpu_info, self.insights_engine,
                                              self.cache_core, log_func=logger)
        self.tab_insights = FusionInsightsTab(self.notebook, self.insights_engine)
        self.tab_memory = MemoryStorageTab(self.notebook, self.memory_mgr, log_func=logger)

        self.notebook.add(self.tab_turbo, text="Turbo Booster")
        self.notebook.add(self.tab_overview, text="Fusion Overview")
        self.notebook.add(self.tab_insights, text="Fusion Insights")
        self.notebook.add(self.tab_memory, text="Memory Storage")

        self._status_label = ttk.Label(self, text="")
        self._status_label.pack(side="bottom", fill="x")
        self._schedule_status()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _schedule_status(self):
        self._status_label.configure(
            text=f"Tier={CAPABILITY_TIER}, FusionMode={self.fusion_ctrl.current_mode.name}"
        )
        self.after(2000, self._schedule_status)

    def on_close(self):
        self.ingestion_bus.stop()
        try:
            self.gpu_info.shutdown()
        except Exception:
            pass
        self.destroy()


# ===========================
# Entry point
# ===========================

def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()

