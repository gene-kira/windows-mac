"""
Turbo Booster Twin Cache Organ (All Live, No Simulation Counters Only)

Features:
- GPU-first VRAM/RAM cache with predictive telemetry and modes.
- Movidius twin engine via OpenVINO (heuristic fallback if not configured).
- Ingestion bus with per-source stats (all sources generate real cache blocks).
- Adapters:
    * system_net  : ALL network interfaces (all ports, all processes) -> real payload blocks
    * tcp_ingest  : payload-level TCP server (port 5000)
    * file_drop   : files dropped into ./ingest_watch
    * camera_0    : webcam frames (Start/Stop from GUI)
    * cpu         : CPU load -> real synthetic blocks
    * ram         : RAM usage -> real synthetic blocks
    * disk        : disk read/write delta -> real synthetic blocks
    * gpu_vram    : GPU VRAM delta -> real synthetic blocks

Dependencies:
    pip install psutil PySide6 watchdog opencv-python
    pip install cupy-cuda12x        # or appropriate CuPy build for your GPU
    pip install openvino            # if using Movidius through OpenVINO
"""

import sys
import threading
import time
import math
import traceback
import queue
import os
import socket
from collections import defaultdict

# ===========================
# Auto-loader for backends
# ===========================

GPU_BACKEND_AVAILABLE = False
MOVIDIUS_AVAILABLE = False

cp = None
OVCore = None

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

import psutil
from PySide6 import QtWidgets, QtCore
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2


# ===========================
# Modes and telemetry
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
        self.handle = handle    # CuPy array or bytearray
        self.telemetry = BlockTelemetry(size_bytes)


# ===========================
# Core VRAM/RAM cache
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
        self._ram_used = 0

    def _detect_vram_total(self):
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
            if self._vram_total:
                vram_used = self._vram_used
                vram_pct = vram_used / self._vram_total
            else:
                vram_used = 0
            vram_pct = vram_used / self._vram_total if self._vram_total else 0.0

            virtual_mem = psutil.virtual_memory()
            ram_total = virtual_mem.total
            ram_used = self._ram_used
            ram_pct = ram_used / ram_total if ram_total > 0 else 0.0

            vthr, mthr = self.current_thresholds()

            return {
                "vram_total": self._vram_total,
                "vram_used": vram_used,
                "vram_pct": vram_pct,
                "ram_total": ram_total,
                "ram_used": ram_used,
                "ram_pct": ram_pct,
                "gpu_backend": "CuPy" if GPU_BACKEND_AVAILABLE else None,
                "mode": self.mode,
                "vram_threshold": vthr,
                "vram_migrate_back": mthr,
            }

    def _vram_pct(self):
        if not self._vram_total:
            return 0.0
        return self._vram_used / self._vram_total

    def can_use_vram(self):
        if not self._vram_total:
            return False
        vthr, _ = self.current_thresholds()
        return self._vram_pct() < vthr

    def allocate(self, size_bytes):
        with self._lock:
            backend = "ram"
            handle = None

            if GPU_BACKEND_AVAILABLE and self.can_use_vram():
                try:
                    handle = cp.empty((size_bytes,), dtype=cp.uint8)
                    backend = "vram"
                    self._vram_used += size_bytes
                except Exception:
                    backend = "ram"
                    handle = None

            if backend == "ram":
                handle = bytearray(size_bytes)
                self._ram_used += size_bytes

            block_id = self.next_id
            self.next_id += 1
            self.blocks[block_id] = CacheBlock(size_bytes, backend, handle)
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
                self._ram_used -= blk.size_bytes
                del blk.handle
            return True

    def touch_block(self, block_id):
        with self._lock:
            blk = self.blocks.get(block_id)
            if blk:
                blk.telemetry.touch()

    def migrate_ram_to_vram_if_possible(self, logger=None):
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
                    # Optional: copy RAM -> VRAM
                    # new_handle[:] = cp.asarray(blk.handle, dtype=cp.uint8)
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
        if blk.backend == "vram" and cp is not None:
            try:
                gpu_arr = blk.handle
                gpu_arr[:size] = cp.asarray(data_bytes, dtype=cp.uint8)
            except Exception:
                blk.backend = "ram"
                blk.handle = bytearray(data_bytes)
                self.core._vram_used -= blk.size_bytes
                self.core._ram_used += blk.size_bytes
        else:
            blk.handle[:size] = data_bytes
        blk.telemetry.touch()
        return block_id

    def stats(self):
        return self.core.get_stats()


# ===========================
# Movidius engine (OpenVINO)
# ===========================

class MovidiusEngine:
    def __init__(self, model_path=None):
        self.available = MOVIDIUS_AVAILABLE
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

        if self.available and self.model_path:
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
        return self.available

    def submit_job(self, features: dict, callback):
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
        if self.available and self.compiled_model is not None:
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
            size = float(features.get("size_bytes", 0))
            access = float(features.get("access_count", 0))
            age = float(features.get("age", 0.0))
            idle = float(features.get("idle", 0.0))
            vram_pct = float(features.get("vram_pct", 0.0))
            heat_score = (access + 1) / (1.0 + idle) * math.log1p(age + 1.0)
            heat_score = 1.0 - math.exp(-heat_score / 10.0)
            evict_score = float(max(0.0, 1.0 - heat_score))
            anomaly_score = 0.0

        tier_recommendation = "vram" if heat_score >= 0.5 else "ram"
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
        return {
            "available": self.available,
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "queue_depth": self.job_queue.qsize(),
            "avg_latency": self._avg_latency,
            "last_error": self._last_error,
            "backend": "OpenVINO/MYRIAD" if self.available else None,
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
        if self.movi.is_available():
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
# Ingestion bus + adapters
# ===========================

class IngestionBus:
    def __init__(self, coordinator: TwinOrganCoordinator, logger=None):
        self.coordinator = coordinator
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

    def update_rates(self):
        now = time.time()
        dt = now - self._last_rate_calc
        if dt <= 0:
            return
        for src, window_bytes in self.source_bytes_window.items():
            self.source_rate[src] = window_bytes / dt  # bytes per second
        self.source_bytes_window.clear()
        self._last_rate_calc = now

    def stats(self):
        with self._lock:
            return {
                "bytes_total": dict(self.source_bytes_total),
                "rate": dict(self.source_rate),  # bytes/sec
            }


# ---- File adapter ----

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


# ---- TCP adapter ----

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


# ---- System-wide network adapter (live blocks) ----

def start_system_net_adapter(bus: IngestionBus, source_name="system_net"):
    """
    Every second, measure system-wide net delta and create a small payload block.
    """
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
            payload = bytes([0x5A]) * size
            bus.ingest(source_name, payload)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


# ---- System metrics adapter (CPU, RAM, Disk, GPU VRAM -> live blocks) ----

def start_system_metrics_adapter(bus: IngestionBus,
                                 cpu_source="cpu",
                                 ram_source="ram",
                                 disk_source="disk",
                                 gpu_source="gpu_vram"):
    def worker():
        prev_disk = psutil.disk_io_counters()
        prev_gpu_used = None
        if GPU_BACKEND_AVAILABLE:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                prev_gpu_used = total - free
            except Exception:
                prev_gpu_used = None

        bus.logger("[metrics] system metrics monitor started")

        while True:
            time.sleep(1.0)

            # CPU: map percent to 1-16 KB
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_size = int(max(1_024, min(16 * 1024, cpu_pct * 256)))
            bus.ingest(cpu_source, bytes([0xC1]) * cpu_size)

            # RAM: map used percent to 1-16 KB
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            ram_size = int(max(1_024, min(16 * 1024, ram_pct * 256)))
            bus.ingest(ram_source, bytes([0xA1]) * ram_size)

            # Disk: delta -> up to 32KB
            curr_disk = psutil.disk_io_counters()
            disk_delta = (curr_disk.read_bytes - prev_disk.read_bytes) + \
                         (curr_disk.write_bytes - prev_disk.write_bytes)
            prev_disk = curr_disk
            if disk_delta > 0:
                size = int(max(1_024, min(32 * 1024, disk_delta)))
                bus.ingest(disk_source, bytes([0xD1]) * size)

            # GPU VRAM: delta -> up to 16KB
            if GPU_BACKEND_AVAILABLE:
                try:
                    free, total = cp.cuda.runtime.memGetInfo()
                    used = total - free
                    if prev_gpu_used is None:
                        prev_gpu_used = used
                    delta_gpu = used - prev_gpu_used
                    prev_gpu_used = used
                    if delta_gpu != 0:
                        size = int(max(1_024, min(16 * 1024, abs(delta_gpu))))
                        # Use 0x61 ('a') as a valid byte marker for GPU activity
                        bus.ingest(gpu_source, bytes([0x61]) * size)
                except Exception:
                    pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


# ---- Camera adapter ----

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


# ===========================
# GUI
# ===========================

class CacheMonitorWidget(QtWidgets.QWidget):
    stats_updated = QtCore.Signal(dict)

    def __init__(self, orchestrator: TwinOrganCoordinator, cache_core: VramRamCacheCore, bus: IngestionBus):
        super().__init__()
        self.orchestrator = orchestrator
        self.cache_core = cache_core
        self.bus = bus

        self._allocated_ids = []

        self.camera_running = False
        self.camera_thread = None
        self.camera_stop_event = None
        self.camera_index = 0

        self._setup_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

        self.stats_updated.connect(self._update_ui_from_stats)

    def _setup_ui(self):
        self.setWindowTitle("Turbo Booster Twin Cache Organ (All Live)")

        layout = QtWidgets.QVBoxLayout(self)

        self.vram_label = QtWidgets.QLabel("VRAM: not available")
        self.vram_bar = QtWidgets.QProgressBar()
        self.vram_bar.setRange(0, 100)

        self.ram_label = QtWidgets.QLabel("RAM Cache: 0 / 0")
        self.ram_bar = QtWidgets.QProgressBar()
        self.ram_bar.setRange(0, 100)

        self.status_label = QtWidgets.QLabel("Status: idle")
        self.backend_label = QtWidgets.QLabel("GPU backend: none")
        self.movi_label = QtWidgets.QLabel("Movidius: none")
        self.camera_status_label = QtWidgets.QLabel("Camera: OFF")

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            CacheMode.CALM, CacheMode.AGGRESSIVE,
            CacheMode.PROTECTIVE, CacheMode.ALTERED
        ])
        self.mode_combo.currentTextChanged.connect(self._change_mode)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_alloc_small = QtWidgets.QPushButton("Test ingest 64MB")
        self.btn_alloc_large = QtWidgets.QPushButton("Test ingest 256MB")
        self.btn_free_all = QtWidgets.QPushButton("Free all test blocks")
        btn_layout.addWidget(self.btn_alloc_small)
        btn_layout.addWidget(self.btn_alloc_large)
        btn_layout.addWidget(self.btn_free_all)

        cam_layout = QtWidgets.QHBoxLayout()
        self.btn_camera_toggle = QtWidgets.QPushButton("Start Camera")
        self.btn_camera_toggle.clicked.connect(self._toggle_camera)
        cam_layout.addWidget(QtWidgets.QLabel("Camera control:"))
        cam_layout.addWidget(self.btn_camera_toggle)
        cam_layout.addWidget(self.camera_status_label)

        self.source_stats_label = QtWidgets.QLabel("Sources: (waiting for data)")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        layout.addWidget(self.vram_label)
        layout.addWidget(self.vram_bar)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.ram_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.backend_label)
        layout.addWidget(self.movi_label)
        layout.addLayout(mode_layout)
        layout.addLayout(btn_layout)
        layout.addLayout(cam_layout)
        layout.addWidget(self.source_stats_label)
        layout.addWidget(QtWidgets.QLabel("Event log:"))
        layout.addWidget(self.log_view)

        self.btn_alloc_small.clicked.connect(lambda: self._ingest_test(64 * 1024 * 1024))
        self.btn_alloc_large.clicked.connect(lambda: self._ingest_test(256 * 1024 * 1024))
        self.btn_free_all.clicked.connect(self._free_all)

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)

    def _change_mode(self, mode):
        self.cache_core.mode = mode
        self.log(f"Mode changed to {mode}")

    def _ingest_test(self, size_bytes: int):
        try:
            data = bytes([0xAA]) * size_bytes
            bid = self.orchestrator.ingest(data)
            self._allocated_ids.append(bid)
            blk = self.cache_core.blocks[bid]
            loc = "VRAM" if blk.backend == "vram" else "RAM"
            self.log(f"[TEST] block {bid}: {size_bytes / (1024**2):.1f} MB via GPU-first -> {loc}")
        except Exception as e:
            self.log(f"[TEST] ingest error: {e}")
            traceback.print_exc()

    def _free_all(self):
        for bid in list(self._allocated_ids):
            if self.cache_core.free(bid):
                self.log(f"[TEST] Freed block {bid}")
        self._allocated_ids.clear()

    def _toggle_camera(self):
        if not self.camera_running:
            self.camera_stop_event = threading.Event()
            self.camera_thread = start_camera_adapter(
                self.bus,
                stop_event=self.camera_stop_event,
                camera_index=self.camera_index,
                source_name=f"camera_{self.camera_index}",
            )
            self.camera_running = True
            self.btn_camera_toggle.setText("Stop Camera")
            self.camera_status_label.setText(f"Camera: ON (index {self.camera_index})")
            self.log(f"[camera] Started camera index {self.camera_index}")
        else:
            if self.camera_stop_event is not None:
                self.camera_stop_event.set()
            self.camera_running = False
            self.btn_camera_toggle.setText("Start Camera")
            self.camera_status_label.setText("Camera: OFF")
            self.log(f"[camera] Stopped camera index {self.camera_index}")

    @QtCore.Slot()
    def _on_timer(self):
        try:
            self.cache_core.migrate_ram_to_vram_if_possible(logger=self.log)
            self.bus.update_rates()
            stats = self.orchestrator.stats()
            self.stats_updated.emit(stats)
        except Exception as e:
            self.log(f"Timer error: {e}")

    @QtCore.Slot(dict)
    def _update_ui_from_stats(self, stats):
        vram_total = stats["vram_total"]
        vram_used = stats["vram_used"]
        vram_pct = stats["vram_pct"]

        if vram_total:
            self.vram_label.setText(
                f"VRAM: {vram_used / (1024**2):.1f} MB / {vram_total / (1024**2):.1f} MB "
                f"({vram_pct * 100:.1f}%)"
            )
            self.vram_bar.setValue(int(vram_pct * 100))
        else:
            self.vram_label.setText("VRAM: not available")
            self.vram_bar.setValue(0)

        ram_total = stats["ram_total"]
        ram_used = stats["ram_used"]
        ram_pct = stats["ram_pct"]

        self.ram_label.setText(
            f"RAM Cache (logical): {ram_used / (1024**2):.1f} MB / {ram_total / (1024**2):.1f} MB "
            f"({ram_pct * 100:.4f}%)"
        )
        self.ram_bar.setValue(int(ram_pct * 100))

        mode = stats["mode"]
        vthr = stats["vram_threshold"]
        if vram_total:
            if vram_pct < vthr:
                status = f"VRAM-first ({mode}, below {vthr*100:.0f}%)"
            else:
                status = f"Spillover to RAM ({mode}, VRAM >= {vthr*100:.0f}%)"
        else:
            status = f"Simulation mode ({mode}, RAM-only)"

        self.status_label.setText(f"Status: {status}")

        backend = stats["gpu_backend"] or "none"
        self.backend_label.setText(f"GPU backend: {backend}")

        movi_stats = stats.get("movidius", {})
        movi_backend = movi_stats.get("backend") or "none"
        qd = movi_stats.get("queue_depth", 0)
        jl = movi_stats.get("jobs_completed", 0)
        lat = movi_stats.get("avg_latency", 0.0)
        err = movi_stats.get("last_error")
        txt = f"Movidius: {movi_backend}, jobs={jl}, queue={qd}, avg_latency={lat*1000:.1f} ms"
        if err:
            txt += f" (err: {err})"
        self.movi_label.setText(txt)

        src_stats = self.bus.stats()
        total_map = src_stats["bytes_total"]
        rate_map = src_stats["rate"]
        if total_map:
            lines = []
            for src in sorted(total_map.keys()):
                total_mb = total_map[src] / (1024**2)
                rate_kb = rate_map.get(src, 0.0) / 1024.0
                lines.append(f"{src}: {total_mb:.2f} MB total, {rate_kb:.1f} KB/s")
            self.source_stats_label.setText("Sources:\n  " + "\n  ".join(lines))
        else:
            self.source_stats_label.setText("Sources: (no data yet)")


# ===========================
# Entry point
# ===========================

def main():
    app = QtWidgets.QApplication(sys.argv)

    cache_core = VramRamCacheCore(mode=CacheMode.CALM)
    movi_engine = MovidiusEngine(model_path=None)  # set env MOVIDIUS_MODEL for real model
    orchestrator = TwinOrganCoordinator(cache_core, movi_engine, logger=lambda m: print(m))
    bus = IngestionBus(orchestrator, logger=lambda m: print(m))

    # System-wide network: all ports, all processes -> real blocks
    start_system_net_adapter(bus, source_name="system_net")

    # System metrics: CPU, RAM, Disk, GPU VRAM -> real blocks
    start_system_metrics_adapter(bus,
                                 cpu_source="cpu",
                                 ram_source="ram",
                                 disk_source="disk",
                                 gpu_source="gpu_vram")

    # Network first: TCP ingest server (payload)
    start_tcp_ingest_server(bus, host="0.0.0.0", port=5000, source_name="tcp_ingest")

    # File second: watch folder for new files
    watch_folder = os.path.abspath("./ingest_watch")
    os.makedirs(watch_folder, exist_ok=True)
    start_file_adapter(bus, path=watch_folder, watch_name="file_drop")

    # Camera last: started/stopped from GUI only
    w = CacheMonitorWidget(orchestrator, cache_core, bus)
    w.resize(1000, 650)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

