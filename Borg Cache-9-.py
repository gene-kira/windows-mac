#!/usr/bin/env python
"""
The Borg Cache - Model A, Wave 1 + Wave 2 + Wave 3 (Aggressive, Mesh, Minimal Logging)
- Single monolithic file
- Queen Borg with:
  - Predictive cache governor + pressure trend history
  - Behavior memory + reinforcement scoring
  - Stability scoring 3.0 (local + mesh-aware)
- Unified VRAM/RAM/SSD cache logic
- BorgIO block-level RAM cache with:
  - async flush
  - write coalescing
  - per-file workload profiling (sequential vs random)
  - hot-region tracking and pinning
  - predictive prefetch 3.0 (per-file adaptive)
  - integrity scanner (aggressive)
  - operation log for crash replay
- Coolant engine with optional NVIDIA VRAM telemetry
- Telemetry:
  - Legacy dual-buffer shared memory + file fallback
  - Segmented shared memory (coolant/cache/borgio/meta) with compression
- Hive / Cockpit / Hybrid modes
- Heartbeats:
  - Hive heartbeat
  - Flush worker heartbeat + resurrection++
- Crash replay:
  - Last 30s telemetry window
  - Last 100 BorgIO ops
- Wave 3:
  - ML prefetch model (Markov-style)
  - Behavior clustering engine
  - Mesh networking (UDP discovery 49300, TCP sync 49301)
  - Remote cockpit server (TCP 49302)
  - Mesh-aware stability scoring
  - Persistent RAM cache + model + clusters
  - Mesh-level crash replay aggregation
- Minimal logging
"""

import os
import sys
import time
import threading
import queue
import json
import platform
import argparse
import mmap
import zlib
import socket
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

# ---------------- Dependency checks ----------------

def ensure_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

if not ensure_import("psutil"):
    print("psutil is required. Install with: pip install psutil")
    sys.exit(1)

import psutil

HAS_PYNVML = ensure_import("pynvml")
if HAS_PYNVML:
    import pynvml

HAS_MSVCRT = os.name == "nt"
if HAS_MSVCRT:
    import msvcrt


# ---------------- GPU probe ----------------

def get_nvidia_vram_usage_mb() -> Optional[Tuple[int, int]]:
    if not HAS_PYNVML:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem.used // (1024 * 1024)
        total_mb = mem.total // (1024 * 1024)
        pynvml.nvmlShutdown()
        return used_mb, total_mb
    except Exception:
        return None


# ---------------- Coolant engine ----------------

@dataclass
class CoolantState:
    cpu_temp: float
    ram_temp: float
    disk_temp: float
    gpu_temp: float | None
    overall_pressure: float
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None


class CoolantEngine:
    def sample(self) -> CoolantState:
        cpu = psutil.cpu_percent(interval=0.05)
        ram = psutil.virtual_memory().percent

        disk_parts = psutil.disk_partitions(all=False)
        disk_pressures = []
        for p in disk_parts:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                disk_pressures.append(usage.percent)
            except PermissionError:
                continue
        disk = sum(disk_pressures) / len(disk_pressures) if disk_pressures else 0

        vram_info = get_nvidia_vram_usage_mb()
        gpu = None
        vram_used = None
        vram_total = None
        if vram_info is not None:
            vram_used, vram_total = vram_info
            gpu = (vram_used / max(vram_total, 1)) * 100.0

        overall = (cpu + ram + disk) / 3

        return CoolantState(
            cpu_temp=cpu,
            ram_temp=ram,
            disk_temp=disk,
            gpu_temp=gpu,
            overall_pressure=overall,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
        )


# ---------------- System scan ----------------

@dataclass
class SystemScanResult:
    os: str
    cpu: str
    ram_total: int
    disks: Dict[str, Any]
    gpu_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemScanner:
    def detect_gpu_mode(self) -> str:
        if HAS_PYNVML:
            return "discrete"
        if platform.system().lower() == "windows":
            return "integrated"
        return "none"

    def scan(self) -> SystemScanResult:
        os_name = platform.platform()
        cpu_name = platform.processor()
        ram_total = psutil.virtual_memory().total

        disks: Dict[str, Any] = {}
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except PermissionError:
                continue
            disks[part.device] = {
                "mount": part.mountpoint,
                "fstype": part.fstype,
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
            }

        gpu_mode = self.detect_gpu_mode()

        return SystemScanResult(
            os=os_name,
            cpu=cpu_name,
            ram_total=ram_total,
            disks=disks,
            gpu_mode=gpu_mode,
            metadata={"note": "Initial system scan complete"},
        )


# ---------------- Hive identity & config ----------------

@dataclass
class HiveIdentity:
    hive_name: str = "The Borg Cache"
    version: str = "1.8.0-wave3-aggressive-mesh"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "hive_name": self.hive_name,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ConfigManager:
    def __init__(self, path: str = "borg_cache_config.json"):
        self.path = path
        self.data: Dict[str, Any] = {}

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
        except Exception:
            pass

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# ---------------- Unified cache engine ----------------

@dataclass
class CacheState:
    vram_alloc_mb: int
    ram_alloc_mb: int
    ssd_alloc_mb: int
    total_cache_mb: int
    gpu_mode: str
    metadata: Dict[str, Any]


class CacheEngine:
    def __init__(self, gpu_mode: str = "none"):
        self.gpu_mode = gpu_mode
        self.vram_alloc_mb = 0
        self.ram_alloc_mb = 0
        self.ssd_alloc_mb = 0

    def apply_allocation(self, vram_mb: int, ram_mb: int, ssd_mb: int):
        if self.gpu_mode == "none":
            self.vram_alloc_mb = 0
            self.ram_alloc_mb = ram_mb + vram_mb
        elif self.gpu_mode == "integrated":
            self.vram_alloc_mb = vram_mb
            self.ram_alloc_mb = ram_mb
        else:
            self.vram_alloc_mb = vram_mb
            self.ram_alloc_mb = ram_mb
        self.ssd_alloc_mb = ssd_mb

    def get_state(self) -> CacheState:
        total = self.vram_alloc_mb + self.ram_alloc_mb + self.ssd_alloc_mb
        return CacheState(
            vram_alloc_mb=self.vram_alloc_mb,
            ram_alloc_mb=self.ram_alloc_mb,
            ssd_alloc_mb=self.ssd_alloc_mb,
            total_cache_mb=total,
            gpu_mode=self.gpu_mode,
            metadata={"note": "Unified cache pool controlled by Queen Borg"},
        )


# ---------------- BorgIO block cache ----------------

BLOCK_SIZE = 4096


@dataclass
class FileProfile:
    path: str
    reads: int = 0
    writes: int = 0
    sequential_hits: int = 0
    random_hits: int = 0
    last_block_index: Optional[int] = None
    hot_blocks: Dict[int, int] = field(default_factory=dict)

    def record_access(self, block_index: int):
        if self.last_block_index is None:
            self.last_block_index = block_index
        else:
            if block_index == self.last_block_index + 1:
                self.sequential_hits += 1
            else:
                self.random_hits += 1
            self.last_block_index = block_index

        self.hot_blocks[block_index] = self.hot_blocks.get(block_index, 0) + 1
        if len(self.hot_blocks) > 256:
            coldest = min(self.hot_blocks.items(), key=lambda kv: kv[1])[0]
            self.hot_blocks.pop(coldest, None)

    @property
    def is_sequential_heavy(self) -> bool:
        total = self.sequential_hits + self.random_hits
        if total < 16:
            return False
        return self.sequential_hits / total > 0.7

    @property
    def is_random_heavy(self) -> bool:
        total = self.sequential_hits + self.random_hits
        if total < 16:
            return False
        return self.random_hits / total > 0.7

    def is_block_hot(self, block_index: int) -> bool:
        hits = self.hot_blocks.get(block_index, 0)
        return hits >= 4


class BorgIO:
    def __init__(self, queen, max_cache_bytes: int = 128 * 1024 * 1024):
        self.queen = queen
        self.max_cache_bytes = max_cache_bytes

        self.cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.current_bytes = 0

        self.flush_queue: "queue.Queue[Tuple[str, int, bytes]]" = queue.Queue()
        self.flush_running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        self.flush_heartbeat = 0.0

        self.stats = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }

        self.prefetch_aggressiveness = 0
        self.async_flush_enabled = True

        self.last_block: Dict[str, int] = {}
        self.file_profiles: Dict[str, FileProfile] = {}
        self.lock = threading.Lock()

        self.coalesce_buffer: Dict[Tuple[str, int], bytes] = {}
        self.coalesce_max_delay = 0.05
        self.coalesce_last_flush = time.time()

        self.integrity_running = True
        self.integrity_thread = threading.Thread(target=self._integrity_scanner, daemon=True)
        self.integrity_thread.start()
        self.integrity_last_check = 0.0

        self.op_log: List[Dict[str, Any]] = []
        self.op_log_max = 100

    def ensure_flush_worker(self):
        if not self.flush_thread.is_alive() and self.flush_running:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()
        now = time.time()
        if now - self.flush_heartbeat > 1.0:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def resize_cache(self, new_size_bytes: int):
        with self.lock:
            self.max_cache_bytes = max(new_size_bytes, BLOCK_SIZE)
            self._evict_if_needed()

    def _touch(self, key: Tuple[str, int]):
        self.cache[key]["last_access"] = time.time()

    def _evict_if_needed(self):
        while self.current_bytes > self.max_cache_bytes and self.cache:
            candidates = []
            pinned = []
            for (path, block_index), entry in self.cache.items():
                prof = self.file_profiles.get(path)
                is_hot = prof.is_block_hot(block_index) if prof else False
                if is_hot:
                    pinned.append(((path, block_index), entry))
                else:
                    candidates.append(((path, block_index), entry))

            if candidates:
                oldest_key = min(candidates, key=lambda kv: kv[1]["last_access"])[0]
            else:
                oldest_key = min(self.cache.items(), key=lambda kv: kv[1]["last_access"])[0]

            entry = self.cache.pop(oldest_key)
            self.current_bytes -= entry["size"]
            self.stats["evictions"] += 1

    def _compress_block(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)

    def _decompress_block(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _put_block(self, path: str, block_index: int, data: bytes):
        size = len(data)
        if size > self.max_cache_bytes:
            return
        key = (path, block_index)
        compressed = self._compress_block(data)
        if key in self.cache:
            self.current_bytes -= self.cache[key]["size"]
        self.cache[key] = {
            "data": compressed,
            "size": len(compressed),
            "last_access": time.time(),
        }
        self.current_bytes += len(compressed)
        self._evict_if_needed()

    def _get_block(self, path: str, block_index: int) -> Optional[bytes]:
        key = (path, block_index)
        if key in self.cache:
            self._touch(key)
            self.stats["cache_hits"] += 1
            return self._decompress_block(self.cache[key]["data"])
        self.stats["cache_misses"] += 1
        return None

    def _read_block_from_disk(self, path: str, block_index: int) -> Optional[bytes]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            with p.open("rb") as f:
                f.seek(block_index * BLOCK_SIZE)
                return f.read(BLOCK_SIZE)
        except Exception:
            return None

    def _write_block_to_disk(self, path: str, block_index: int, data: bytes):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if p.exists() else "w+b"
        with p.open(mode) as f:
            f.seek(block_index * BLOCK_SIZE)
            f.write(data)

    def _flush_worker(self):
        while self.flush_running:
            self.flush_heartbeat = time.time()
            try:
                path, block_index, data = self.flush_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._write_block_to_disk(path, block_index, data)
            except Exception:
                pass

    def _enqueue_flush(self, path: str, block_index: int, data: bytes):
        self.flush_queue.put((path, block_index, data))

    def _get_profile(self, key_path: str) -> FileProfile:
        prof = self.file_profiles.get(key_path)
        if prof is None:
            prof = FileProfile(path=key_path)
            self.file_profiles[key_path] = prof
        return prof

    def _predictive_prefetch(self, key_path: str, block_index: int, prof: FileProfile):
        effective_prefetch = self.prefetch_aggressiveness

        if prof.is_random_heavy:
            effective_prefetch = 0
        elif prof.is_sequential_heavy and self.prefetch_aggressiveness > 0:
            effective_prefetch = min(2, self.prefetch_aggressiveness + 1)

        if effective_prefetch <= 0:
            return

        last = self.last_block.get(key_path, block_index)
        if block_index != last + 1:
            return

        count = 1 if effective_prefetch == 1 else 4
        for offset in range(1, count + 1):
            idx = block_index + offset
            if (key_path, idx) in self.cache:
                continue
            pre = self._read_block_from_disk(key_path, idx)
            if pre is not None:
                self._put_block(key_path, idx, pre)
                p = self._get_profile(key_path)
                p.reads += 1
                p.record_access(idx)
                self._log_op("prefetch", key_path, idx, len(pre))

    def _flush_coalesce_buffer_if_needed(self, force: bool = False):
        now = time.time()
        if not force and (now - self.coalesce_last_flush) < self.coalesce_max_delay:
            return
        if not self.coalesce_buffer:
            self.coalesce_last_flush = now
            return
        for (path, block_index), data in list(self.coalesce_buffer.items()):
            self._enqueue_flush(path, block_index, data)
        self.coalesce_buffer.clear()
        self.coalesce_last_flush = now

    def _log_op(self, kind: str, path: str, block_index: int, size: int):
        entry = {
            "t": time.time(),
            "kind": kind,
            "path": path,
            "block": block_index,
            "size": size,
        }
        self.op_log.append(entry)
        if len(self.op_log) > self.op_log_max:
            self.op_log = self.op_log[-self.op_log_max:]

    def read_block(self, path: str, block_index: int) -> Optional[bytes]:
        key_path = str(Path(path).resolve())
        with self.lock:
            self.stats["reads"] += 1
            cached = self._get_block(key_path, block_index)
            prof = self._get_profile(key_path)
            prof.reads += 1
            prof.record_access(block_index)
            if cached is not None:
                self.last_block[key_path] = block_index
                self._log_op("read_hit", key_path, block_index, len(cached))
                return cached

        data = self._read_block_from_disk(key_path, block_index)
        if data is None:
            return None

        with self.lock:
            self._put_block(key_path, block_index, data)
            self.last_block[key_path] = block_index
            prof = self._get_profile(key_path)
            prof.reads += 1
            prof.record_access(block_index)
            self._predictive_prefetch(key_path, block_index, prof)
            self._log_op("read_miss", key_path, block_index, len(data))

        return data

    def write_block(self, path: str, block_index: int, data: bytes, async_flush: Optional[bool] = None):
        key_path = str(Path(path).resolve())
        with self.lock:
            self.stats["writes"] += 1
            self._put_block(key_path, block_index, data)
            prof = self._get_profile(key_path)
            prof.writes += 1
            prof.record_access(block_index)
            self._log_op("write", key_path, block_index, len(data))

            use_async = self.async_flush_enabled if async_flush is None else async_flush
            if use_async:
                self.coalesce_buffer[(key_path, block_index)] = data
                self._flush_coalesce_buffer_if_needed(force=False)
            else:
                self._write_block_to_disk(key_path, block_index, data)

    def read_text(self, path: str, encoding: str = "utf-8") -> Optional[str]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = p.read_bytes()
        except Exception:
            return None
        key_path = str(p.resolve())
        with self.lock:
            prof = self._get_profile(key_path)
            for i in range(0, len(data), BLOCK_SIZE):
                block_idx = i // BLOCK_SIZE
                self._put_block(key_path, block_idx, data[i:i + BLOCK_SIZE])
                prof.reads += 1
                prof.record_access(block_idx)
                self._log_op("read_text_block", key_path, block_idx, min(BLOCK_SIZE, len(data) - i))
        return data.decode(encoding, errors="ignore")

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> bool:
        data = text.encode(encoding)
        p = Path(path)
        key_path = str(p.resolve())
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        except Exception:
            return False
        with self.lock:
            prof = self._get_profile(key_path)
            for i in range(0, len(data), BLOCK_SIZE):
                block_idx = i // BLOCK_SIZE
                self._put_block(key_path, block_idx, data[i:i + BLOCK_SIZE])
                prof.writes += 1
                prof.record_access(block_idx)
                self._log_op("write_text_block", key_path, block_idx, min(BLOCK_SIZE, len(data) - i))
        return True

    def snapshot_stats(self) -> Dict[str, Any]:
        with self.lock:
            seq_heavy = 0
            rnd_heavy = 0
            hot_files: List[Tuple[str, int]] = []
            for prof in self.file_profiles.values():
                if prof.is_sequential_heavy:
                    seq_heavy += 1
                elif prof.is_random_heavy:
                    rnd_heavy += 1
                total_access = prof.reads + prof.writes
                if total_access > 0:
                    hot_files.append((prof.path, total_access))

            hot_files.sort(key=lambda kv: kv[1], reverse=True)
            top_hot = [{"path": p, "accesses": a} for p, a in hot_files[:10]]

            return {
                "reads": self.stats["reads"],
                "writes": self.stats["writes"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "evictions": self.stats["evictions"],
                "current_bytes": self.current_bytes,
                "max_cache_bytes": self.max_cache_bytes,
                "entries": len(self.cache),
                "files_sequential_heavy": seq_heavy,
                "files_random_heavy": rnd_heavy,
                "top_hot_files": top_hot,
            }

    def _integrity_scanner(self):
        while self.integrity_running:
            time.sleep(1.0)
            with self.lock:
                self.integrity_last_check = time.time()
                if not self.cache:
                    continue
                keys = list(self.cache.keys())
                sample_keys = keys[: min(8, len(keys))]
                for key in sample_keys:
                    entry = self.cache.get(key)
                    if not entry:
                        continue
                    try:
                        data = self._decompress_block(entry["data"])
                        _ = len(data)
                    except Exception:
                        path, block_index = key
                        self.cache.pop(key, None)
                        self.current_bytes -= entry["size"]
                        fresh = self._read_block_from_disk(path, block_index)
                        if fresh is not None:
                            self._put_block(path, block_index, fresh)

    def shutdown(self):
        with self.lock:
            self._flush_coalesce_buffer_if_needed(force=True)
        self.flush_running = False
        self.integrity_running = False
        self.flush_thread.join(timeout=2.0)
        self.integrity_thread.join(timeout=2.0)


# ---------------- Telemetry bus (legacy dual-buffer) ----------------

@dataclass
class TelemetryConfig:
    mapping_name: str = "Global\\BorgCacheTelemetry"
    size: int = 64 * 1024
    file_path: str = "borg_cache_telemetry.json"


class TelemetryBus:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self._mmap: Optional[mmap.mmap] = None
        self._lock = threading.Lock()
        self._init_shared_memory()

    def _init_shared_memory(self):
        if os.name != "nt":
            self._mmap = None
            return
        try:
            self._mmap = mmap.mmap(
                -1,
                self.config.size,
                tagname=self.config.mapping_name
            )
            if self._mmap.read(1) == b"\x00":
                self._write_raw(self._empty_state())
        except Exception:
            self._mmap = None

    def _empty_state(self) -> bytes:
        state = {
            "active": 0,
            "buffers": [
                {"timestamp": 0, "payload": {}},
                {"timestamp": 0, "payload": {}},
            ],
        }
        raw = json.dumps(state).encode("utf-8")
        if len(raw) > self.config.size:
            raw = raw[: self.config.size]
        return raw.ljust(self.config.size, b"\x00")

    def _write_raw(self, data: bytes):
        if not self._mmap:
            return
        self._mmap.seek(0)
        self._mmap.write(data)
        self._mmap.flush()

    def _read_raw(self) -> Optional[bytes]:
        if not self._mmap:
            return None
        self._mmap.seek(0)
        return self._mmap.read(self.config.size)

    def publish(self, payload: Dict[str, Any]):
        now = time.time()
        with self._lock:
            if self._mmap:
                try:
                    raw = self._read_raw()
                    if not raw:
                        raw = self._empty_state()
                    txt = raw.rstrip(b"\x00").decode("utf-8", errors="ignore") or "{}"
                    try:
                        state = json.loads(txt)
                    except Exception:
                        state = {
                            "active": 0,
                            "buffers": [
                                {"timestamp": 0, "payload": {}},
                                {"timestamp": 0, "payload": {}},
                            ],
                        }
                    active = state.get("active", 0)
                    next_idx = 1 - int(active)
                    buffers = state.get("buffers", [
                        {"timestamp": 0, "payload": {}},
                        {"timestamp": 0, "payload": {}},
                    ])
                    buffers[next_idx] = {"timestamp": now, "payload": payload}
                    state["buffers"] = buffers
                    state["active"] = next_idx
                    raw_out = json.dumps(state).encode("utf-8")
                    if len(raw_out) > self.config.size:
                        raw_out = raw_out[: self.config.size]
                    self._write_raw(raw_out.ljust(self.config.size, b"\x00"))
                except Exception:
                    pass

        try:
            with open(self.config.file_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": now, "payload": payload}, f, indent=2)
        except Exception:
            pass

    def read(self) -> Optional[Dict[str, Any]]:
        best_payload = None
        best_ts = 0.0

        if self._mmap:
            try:
                raw = self._read_raw()
                if raw:
                    txt = raw.rstrip(b"\x00").decode("utf-8", errors="ignore") or "{}"
                    state = json.loads(txt)
                    buffers = state.get("buffers", [])
                    for buf in buffers:
                        ts = buf.get("timestamp", 0)
                        if ts > best_ts:
                            best_ts = ts
                            best_payload = buf.get("payload")
            except Exception:
                pass

        try:
            if os.path.exists(self.config.file_path):
                with open(self.config.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ts = data.get("timestamp", 0)
                if ts > best_ts:
                    best_ts = ts
                    best_payload = data.get("payload")
        except Exception:
            pass

        if best_payload is None:
            return None
        best_payload["_telemetry_timestamp"] = best_ts
        return best_payload


# ---------------- Segmented telemetry (Wave 2) ----------------

class SegmentedTelemetry:
    def __init__(self, base_name: str = "Global\\BorgCache", size: int = 32 * 1024):
        self.base_name = base_name
        self.size = size
        self._segments: Dict[str, mmap.mmap] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._init_segments(["Coolant", "Cache", "BorgIO", "Meta"])

    def _init_segments(self, names: List[str]):
        if os.name != "nt":
            return
        for n in names:
            tag = f"{self.base_name}_{n}"
            try:
                mm = mmap.mmap(-1, self.size, tagname=tag)
                self._segments[n] = mm
                self._locks[n] = threading.Lock()
            except Exception:
                continue

    def _write_segment(self, name: str, payload: Dict[str, Any]):
        mm = self._segments.get(name)
        if not mm:
            return
        lock = self._locks[name]
        with lock:
            try:
                raw = json.dumps({
                    "t": time.time(),
                    "payload": payload,
                }).encode("utf-8")
                comp = zlib.compress(raw, level=9)
                if len(comp) > self.size:
                    comp = comp[: self.size]
                mm.seek(0)
                mm.write(comp.ljust(self.size, b"\x00"))
                mm.flush()
            except Exception:
                pass

    def _read_segment(self, name: str) -> Optional[Dict[str, Any]]:
        mm = self._segments.get(name)
        if not mm:
            return None
        lock = self._locks[name]
        with lock:
            try:
                mm.seek(0)
                raw = mm.read(self.size)
                raw = raw.rstrip(b"\x00")
                if not raw:
                    return None
                decomp = zlib.decompress(raw)
                obj = json.loads(decomp.decode("utf-8", errors="ignore"))
                return obj
            except Exception:
                return None

    def publish_all(self, coolant: Dict[str, Any], cache: Dict[str, Any],
                    borgio: Dict[str, Any], meta: Dict[str, Any]):
        self._write_segment("Coolant", coolant)
        self._write_segment("Cache", cache)
        self._write_segment("BorgIO", borgio)
        self._write_segment("Meta", meta)

    def read_all(self) -> Optional[Dict[str, Any]]:
        c = self._read_segment("Coolant")
        ca = self._read_segment("Cache")
        b = self._read_segment("BorgIO")
        m = self._read_segment("Meta")
        if not any([c, ca, b, m]):
            return None
        return {
            "coolant": c["payload"] if c else {},
            "cache": ca["payload"] if ca else {},
            "borg_io": b["payload"] if b else {},
            "meta": m["payload"] if m else {},
            "_telemetry_timestamp": max(
                x["t"] for x in [c, ca, b, m] if x is not None
            ),
        }


# ---------------- Crash replay (Wave 2 + Mesh) ----------------

class CrashReplay:
    def __init__(self, telemetry_window: int = 30, path: str = "crash_replay.json", ops_path: str = "crash_ops.log"):
        self.telemetry_window = telemetry_window
        self.path = path
        self.ops_path = ops_path
        self.buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def record_snapshot(self, snapshot: Dict[str, Any]):
        with self.lock:
            self.buffer.append({"t": time.time(), "snapshot": snapshot})
            cutoff = time.time() - self.telemetry_window
            self.buffer = [x for x in self.buffer if x["t"] >= cutoff]

    def save_on_crash(self, ops: List[Dict[str, Any]]):
        try:
            with self.lock:
                data = {
                    "saved_at": time.time(),
                    "window_seconds": self.telemetry_window,
                    "snapshots": self.buffer,
                }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            with open(self.ops_path, "w", encoding="utf-8") as f:
                for op in ops:
                    f.write(json.dumps(op) + "\n")
        except Exception:
            pass

    def load_last(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# ---------------- Wave 3: ML Prefetch Model ----------------

class MLPrefetchModel:
    def __init__(self):
        self.transitions: Dict[str, Dict[int, Dict[int, int]]] = {}
        self.lock = threading.Lock()

    def _key(self, path: str) -> str:
        return path

    def observe(self, path: str, prev_block: Optional[int], block: int):
        if prev_block is None:
            return
        k = self._key(path)
        with self.lock:
            file_map = self.transitions.setdefault(k, {})
            trans = file_map.setdefault(prev_block, {})
            trans[block] = trans.get(block, 0) + 1

    def predict_next_blocks(self, path: str, block: int, max_n: int = 4) -> List[int]:
        k = self._key(path)
        with self.lock:
            file_map = self.transitions.get(k)
            if not file_map:
                return []
            trans = file_map.get(block)
            if not trans:
                return []
            sorted_targets = sorted(trans.items(), key=lambda kv: kv[1], reverse=True)
            return [b for b, _ in sorted_targets[:max_n]]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"transitions": self.transitions}

    def load(self, data: Dict[str, Any]):
        with self.lock:
            self.transitions = data.get("transitions", {})


# ---------------- Wave 3: Behavior Clustering ----------------

class ClusterEngine:
    def __init__(self):
        self.current_cluster: str = "unknown"
        self.lock = threading.Lock()

    def classify(self, stats: Dict[str, Any], coolant: CoolantState) -> str:
        reads = stats["reads"]
        writes = stats["writes"]
        seq = stats.get("files_sequential_heavy", 0)
        rnd = stats.get("files_random_heavy", 0)
        pressure = coolant.overall_pressure

        cluster = "mixed"
        if seq > rnd * 2 and reads > writes:
            cluster = "gaming_or_streaming"
        elif writes > reads * 2:
            cluster = "editing_or_build"
        elif rnd > seq * 2:
            cluster = "random_io"
        if pressure > 80:
            cluster = cluster + "_high_pressure"

        with self.lock:
            self.current_cluster = cluster
        return cluster

    def get_cluster(self) -> str:
        with self.lock:
            return self.current_cluster

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"cluster": self.current_cluster}

    def load(self, data: Dict[str, Any]):
        with self.lock:
            self.current_cluster = data.get("cluster", "unknown")


# ---------------- Wave 3: Mesh Manager ----------------

class MeshManager:
    UDP_PORT = 49300
    TCP_PORT = 49301

    def __init__(self, identity: HiveIdentity):
        self.identity = identity
        self.node_id = f"{socket.gethostname()}-{os.getpid()}"
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.running = True

        self.udp_thread = threading.Thread(target=self._udp_broadcast_loop, daemon=True)
        self.udp_thread.start()

        self.udp_listener_thread = threading.Thread(target=self._udp_listen_loop, daemon=True)
        self.udp_listener_thread.start()

        self.tcp_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
        self.tcp_thread.start()

        self.last_mesh_snapshot: Dict[str, Any] = {}

    def _udp_broadcast_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)
        while self.running:
            try:
                payload = {
                    "node_id": self.node_id,
                    "hive_name": self.identity.hive_name,
                    "version": self.identity.version,
                    "tcp_port": self.TCP_PORT,
                    "t": time.time(),
                }
                data = json.dumps(payload).encode("utf-8")
                sock.sendto(data, ("255.255.255.255", self.UDP_PORT))
            except Exception:
                pass
            time.sleep(2.0)

    def _udp_listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.UDP_PORT))
        except Exception:
            return
        sock.settimeout(1.0)
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            try:
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                node_id = msg.get("node_id")
                if not node_id or node_id == self.node_id:
                    continue
                with self.lock:
                    self.peers[node_id] = {
                        "addr": addr[0],
                        "tcp_port": msg.get("tcp_port", self.TCP_PORT),
                        "hive_name": msg.get("hive_name"),
                        "version": msg.get("version"),
                        "last_seen": msg.get("t", time.time()),
                    }
            except Exception:
                continue

    def _tcp_server_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.TCP_PORT))
            sock.listen(5)
        except Exception:
            return
        sock.settimeout(1.0)
        while self.running:
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle_tcp_client, args=(conn,), daemon=True).start()

    def _handle_tcp_client(self, conn: socket.socket):
        try:
            data = conn.recv(65536)
            if not data:
                conn.close()
                return
            msg = json.loads(data.decode("utf-8", errors="ignore"))
            if msg.get("type") == "mesh_sync":
                with self.lock:
                    self.last_mesh_snapshot = msg.get("payload", {})
                reply = json.dumps({"type": "mesh_ack"}).encode("utf-8")
                conn.sendall(reply)
        except Exception:
            pass
        finally:
            conn.close()

    def sync_with_peers(self, local_snapshot: Dict[str, Any]):
        with self.lock:
            peers = list(self.peers.items())
        for node_id, info in peers:
            addr = info.get("addr")
            port = info.get("tcp_port", self.TCP_PORT)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect((addr, port))
                msg = {
                    "type": "mesh_sync",
                    "from": self.node_id,
                    "payload": local_snapshot,
                }
                sock.sendall(json.dumps(msg).encode("utf-8"))
                try:
                    _ = sock.recv(4096)
                except Exception:
                    pass
                sock.close()
            except Exception:
                continue

    def get_mesh_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.last_mesh_snapshot)

    def get_peers(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.peers)

    def shutdown(self):
        self.running = False


# ---------------- Wave 3: Remote Cockpit Server ----------------

class RemoteCockpitServer:
    PORT = 49302

    def __init__(self, queen_get_overview_callable):
        self.get_overview = queen_get_overview_callable
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.PORT))
            sock.listen(5)
        except Exception:
            return
        sock.settimeout(1.0)
        while self.running:
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket):
        try:
            data = conn.recv(4096)
            if not data:
                conn.close()
                return
            try:
                req = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                req = {}
            cmd = req.get("cmd", "overview")
            if cmd == "overview":
                ov = self.get_overview()
                conn.sendall(json.dumps({"type": "overview", "payload": ov}).encode("utf-8"))
            else:
                conn.sendall(json.dumps({"type": "error", "msg": "unknown_cmd"}).encode("utf-8"))
        except Exception:
            pass
        finally:
            conn.close()

    def shutdown(self):
        self.running = False


# ---------------- Wave 3: Persistence Engine ----------------

class PersistenceEngine:
    def __init__(self, cache_path: str = "borg_cache_ram.bin",
                 model_path: str = "borg_cache_model.json",
                 cluster_path: str = "borg_cache_cluster.json"):
        self.cache_path = cache_path
        self.model_path = model_path
        self.cluster_path = cluster_path

    def save_cache(self, io: BorgIO):
        try:
            with io.lock:
                entries = []
                for (path, block_index), entry in io.cache.items():
                    try:
                        data = io._decompress_block(entry["data"])
                    except Exception:
                        continue
                    entries.append({
                        "path": path,
                        "block": block_index,
                        "data": data.hex(),
                    })
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries}, f)
        except Exception:
            pass

    def load_cache(self, io: BorgIO):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            with io.lock:
                for e in entries:
                    path = e["path"]
                    block = e["block"]
                    raw = bytes.fromhex(e["data"])
                    io._put_block(path, block, raw)
        except Exception:
            pass

    def save_model(self, model: MLPrefetchModel):
        try:
            snap = model.snapshot()
            with open(self.model_path, "w", encoding="utf-8") as f:
                json.dump(snap, f)
        except Exception:
            pass

    def load_model(self, model: MLPrefetchModel):
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model.load(data)
        except Exception:
            pass

    def save_cluster(self, cluster: ClusterEngine):
        try:
            snap = cluster.snapshot()
            with open(self.cluster_path, "w", encoding="utf-8") as f:
                json.dump(snap, f)
        except Exception:
            pass

    def load_cluster(self, cluster: ClusterEngine):
        if not os.path.exists(self.cluster_path):
            return
        try:
            with open(self.cluster_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cluster.load(data)
        except Exception:
            pass


# ---------------- Queen Borg ----------------

@dataclass
class LLMProfile:
    name: str
    kind: str
    endpoint: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueenBorg:
    def __init__(self, config_path: str = "borg_cache_config.json"):
        self.identity = HiveIdentity()
        self.config = ConfigManager(config_path)
        self.config.load()

        self.scanner = SystemScanner()
        self.scan_result = self.scanner.scan()
        self.gpu_mode = self.scan_result.gpu_mode

        self.coolant = CoolantEngine()
        self.cache = CacheEngine(gpu_mode=self.gpu_mode)
        self.llms: Dict[str, LLMProfile] = {}

        self._bootstrap_llms()

        self.io = BorgIO(self, max_cache_bytes=128 * 1024 * 1024)

        self.ml_model = MLPrefetchModel()
        self.cluster_engine = ClusterEngine()
        self.persistence = PersistenceEngine()
        self.mesh = MeshManager(self.identity)
        self.remote_cockpit = RemoteCockpitServer(self.get_overview)

        self._load_policy_memory()
        self._load_borgio_stats()
        self.persistence.load_model(self.ml_model)
        self.persistence.load_cluster(self.cluster_engine)
        self.persistence.load_cache(self.io)

        self.pressure_history: List[float] = []
        self.pressure_history_max = 60
        self.last_cache_config_score: float = 0.0

        self.crash_replay = CrashReplay()

        self.hive_heartbeat = 0.0

        self.update_cache_strategy()

        self.last_tick_time = time.time()

    def _bootstrap_llms(self):
        system = platform.system().lower()
        if system == "windows":
            self.register_llm(
                LLMProfile(
                    name="windows_copilot",
                    kind="system",
                    endpoint=None,
                    metadata={"note": "Windows-native AI assistant"},
                )
            )
        if not self.llms:
            self.register_llm(
                LLMProfile(
                    name="downloaded_llm",
                    kind="downloaded",
                    endpoint="path/to/downloaded/model",
                    metadata={"note": "Auto-installed fallback model"},
                )
            )

    def register_llm(self, profile: LLMProfile):
        self.llms[profile.name] = profile

    def list_llms(self) -> Dict[str, Any]:
        return {
            name: {
                "kind": p.kind,
                "endpoint": p.endpoint,
                "metadata": p.metadata,
            }
            for name, p in self.llms.items()
        }

    def _score_current_config(self, stats: Dict[str, Any], pressure_trend: float,
                              mesh_snapshot: Dict[str, Any]) -> float:
        reads = stats["reads"]
        misses = stats["cache_misses"]
        evictions = stats["evictions"]
        if reads <= 0:
            base_score = 0.0
        else:
            miss_rate = misses / reads
            eviction_rate = evictions / max(reads, 1)
            penalty = miss_rate + eviction_rate * 0.5 + max(pressure_trend, 0) / 200.0
            base_score = max(0.0, 1.0 - penalty)

        mesh_anomaly_rate = float(mesh_snapshot.get("anomaly_rate", 0.0))
        mesh_stability = float(mesh_snapshot.get("stability", 1.0))
        mesh_penalty = mesh_anomaly_rate * 0.3 + (1.0 - mesh_stability) * 0.3

        score = max(0.0, base_score - mesh_penalty)
        return score

    def _remember_config_outcome(self, score: float, trend: float):
        mem = self.config.get("behavior_memory", {})
        history = mem.get("scores", [])
        history.append({"t": time.time(), "score": score, "trend": trend})
        if len(history) > 200:
            history = history[-200:]
        mem["scores"] = history
        mem["last_score"] = score
        self.config.set("behavior_memory", mem)

    def _load_behavior_memory(self):
        mem = self.config.get("behavior_memory", {})
        self.last_cache_config_score = float(mem.get("last_score", 0.0))

    def _update_pressure_history(self, pressure: float):
        self.pressure_history.append(pressure)
        if len(self.pressure_history) > self.pressure_history_max:
            self.pressure_history = self.pressure_history[-self.pressure_history_max:]

    def _pressure_trend(self) -> float:
        if len(self.pressure_history) < 10:
            return 0.0
        recent = self.pressure_history[-5:]
        older = self.pressure_history[:5]
        if not older:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(older) / len(older))

    def update_cache_strategy(self):
        c = self.coolant.sample()
        pressure = c.overall_pressure
        self._update_pressure_history(pressure)
        trend = self._pressure_trend()

        ram_total_mb = self.scan_result.ram_total // (1024 * 1024)

        if self.gpu_mode == "integrated":
            max_vram = ram_total_mb // 8
        elif self.gpu_mode == "discrete":
            max_vram = 2048
        else:
            max_vram = 0

        if c.vram_used_mb is not None and c.vram_total_mb is not None:
            vram_free_mb = max(c.vram_total_mb - c.vram_used_mb, 0)
            max_vram = min(max_vram, vram_free_mb // 2)

        if pressure < 30:
            vram = min(max_vram, 1024)
            ram = min(ram_total_mb // 2, 4096)
            ssd = 4096
        elif pressure < 60:
            vram = min(max_vram, 512)
            ram = min(ram_total_mb // 3, 2048)
            ssd = 3072
        elif pressure < 80:
            vram = min(max_vram, 256)
            ram = min(ram_total_mb // 4, 1024)
            ssd = 2048
        else:
            vram = min(max_vram, 128)
            ram = min(ram_total_mb // 8, 512)
            ssd = 1024

        if trend > 5.0:
            ram = max(int(ram * 0.7), 256)
        elif trend < -5.0:
            ram = min(int(ram * 1.2), ram_total_mb // 2)

        cluster = self.cluster_engine.get_cluster()
        if "random_io" in cluster:
            ram = max(int(ram * 0.8), 256)
        if "gaming_or_streaming" in cluster:
            ram = min(int(ram * 1.1), ram_total_mb // 2)

        self.cache.apply_allocation(vram, ram, ssd)
        self.io.resize_cache(ram * 1024 * 1024)
        self.config.set("last_cache_state", self.cache.get_state().__dict__)

    def _load_policy_memory(self):
        policy = self.config.get("policy_memory", {})
        prefetch = policy.get("prefetch_aggressiveness")
        async_flush = policy.get("async_flush_enabled")
        if prefetch is not None:
            self.io.prefetch_aggressiveness = prefetch
        if async_flush is not None:
            self.io.async_flush_enabled = async_flush
        self._load_behavior_memory()

    def _save_policy_memory(self):
        policy = {
            "prefetch_aggressiveness": self.io.prefetch_aggressiveness,
            "async_flush_enabled": self.io.async_flush_enabled,
        }
        self.config.set("policy_memory", policy)

    def _load_borgio_stats(self):
        stats = self.config.get("borg_io_stats")
        if isinstance(stats, dict):
            for k in ("reads", "writes", "cache_hits", "cache_misses", "evictions"):
                if k in stats and isinstance(stats[k], int):
                    self.io.stats[k] = stats[k]

    def _tune_borg_io(self, coolant: CoolantState):
        stats = self.io.snapshot_stats()
        reads = stats["reads"]
        misses = stats["cache_misses"]
        evictions = stats["evictions"]
        writes = stats["writes"]

        miss_rate = (misses / reads) if reads > 0 else 0.0

        if miss_rate > 0.5:
            self.io.prefetch_aggressiveness = max(self.io.prefetch_aggressiveness, 2)
        elif miss_rate > 0.2:
            self.io.prefetch_aggressiveness = max(self.io.prefetch_aggressiveness, 1)
        else:
            if self.io.prefetch_aggressiveness > 0:
                self.io.prefetch_aggressiveness -= 1

        if evictions > 100:
            self.io.prefetch_aggressiveness = max(self.io.prefetch_aggressiveness - 1, 0)

        if writes > reads * 2:
            self.io.async_flush_enabled = True
        elif writes < max(reads / 4, 1):
            self.io.async_flush_enabled = True

        cluster = self.cluster_engine.classify(stats, coolant)
        mesh_snapshot = self.mesh.get_mesh_snapshot()
        trend = self._pressure_trend()
        score = self._score_current_config(stats, trend, mesh_snapshot)
        self._remember_config_outcome(score, trend)

        self.config.set("borg_io_stats", stats)
        self._save_policy_memory()

    def apply_operator_controls(self):
        controls = self.config.get("operator_controls", {})
        if not isinstance(controls, dict):
            return

        if "prefetch" in controls:
            try:
                v = int(controls["prefetch"])
                self.io.prefetch_aggressiveness = max(0, min(2, v))
            except Exception:
                pass

        if "async_flush" in controls:
            try:
                self.io.async_flush_enabled = bool(controls["async_flush"])
            except Exception:
                pass

        if "cache_limit_mb" in controls:
            try:
                mb = int(controls["cache_limit_mb"])
                mb = max(64, mb)
                self.io.resize_cache(mb * 1024 * 1024)
            except Exception:
                pass

    def _update_ml_model_from_ops(self):
        with self.io.lock:
            for op in self.io.op_log:
                if op["kind"] in ("read_hit", "read_miss", "prefetch"):
                    path = op["path"]
                    block = op["block"]
                    prev = block - 1
                    self.ml_model.observe(path, prev, block)

    def _mesh_local_snapshot(self) -> Dict[str, Any]:
        stats = self.io.snapshot_stats()
        coolant = self.coolant.sample()
        cluster = self.cluster_engine.get_cluster()
        anomaly_rate = 0.0
        if stats["reads"] > 0:
            anomaly_rate = stats["cache_misses"] / stats["reads"]
        stability = max(0.0, 1.0 - anomaly_rate)
        return {
            "node": self.mesh.node_id,
            "cluster": cluster,
            "anomaly_rate": anomaly_rate,
            "stability": stability,
            "hot_files": stats.get("top_hot_files", []),
            "t": time.time(),
        }

    def background_tick(self):
        c = self.coolant.sample()
        self._update_pressure_history(c.overall_pressure)
        self.update_cache_strategy()
        self.apply_operator_controls()
        self._tune_borg_io(c)
        self.io.ensure_flush_worker()
        self._update_ml_model_from_ops()
        local_mesh_snapshot = self._mesh_local_snapshot()
        self.mesh.sync_with_peers(local_mesh_snapshot)
        self.last_tick_time = time.time()
        self.hive_heartbeat = self.last_tick_time

    def get_overview(self) -> Dict[str, Any]:
        c = self.coolant.sample()
        cache_state = self.cache.get_state()
        controls = {
            "prefetch": self.io.prefetch_aggressiveness,
            "async_flush": self.io.async_flush_enabled,
            "cache_limit_mb": self.io.max_cache_bytes // (1024 * 1024),
        }
        stats = self.io.snapshot_stats()
        cluster = self.cluster_engine.get_cluster()
        mesh_snapshot = self.mesh.get_mesh_snapshot()
        peers = self.mesh.get_peers()
        return {
            "identity": self.identity.snapshot(),
            "scan": {
                "os": self.scan_result.os,
                "cpu": self.scan_result.cpu,
                "ram_total": self.scan_result.ram_total,
                "disks": self.scan_result.disks,
                "gpu_mode": self.scan_result.gpu_mode,
                "metadata": self.scan_result.metadata,
            },
            "coolant": {
                "cpu_temp": c.cpu_temp,
                "ram_temp": c.ram_temp,
                "disk_temp": c.disk_temp,
                "gpu_temp": c.gpu_temp,
                "overall_pressure": c.overall_pressure,
                "vram_used_mb": c.vram_used_mb,
                "vram_total_mb": c.vram_total_mb,
            },
            "cache": {
                "vram_alloc_mb": cache_state.vram_alloc_mb,
                "ram_alloc_mb": cache_state.ram_alloc_mb,
                "ssd_alloc_mb": cache_state.ssd_alloc_mb,
                "total_cache_mb": cache_state.total_cache_mb,
                "gpu_mode": cache_state.gpu_mode,
                "metadata": cache_state.metadata,
            },
            "llms": self.list_llms(),
            "borg_io": stats,
            "controls": controls,
            "hive": {
                "last_tick_time": self.last_tick_time,
                "heartbeat": self.hive_heartbeat,
                "cluster": cluster,
            },
            "mesh": {
                "local": self._mesh_local_snapshot(),
                "last_snapshot": mesh_snapshot,
                "peers": peers,
            },
        }

    def shutdown(self):
        self.persistence.save_cache(self.io)
        self.persistence.save_model(self.ml_model)
        self.persistence.save_cluster(self.cluster_engine)
        self.io.shutdown()
        self.mesh.shutdown()
        self.remote_cockpit.shutdown()


# ---------------- Operator control helpers ----------------

def save_controls_to_config(config: ConfigManager, queen: QueenBorg):
    controls = {
        "prefetch": queen.io.prefetch_aggressiveness,
        "async_flush": queen.io.async_flush_enabled,
        "cache_limit_mb": queen.io.max_cache_bytes // (1024 * 1024),
    }
    config.set("operator_controls", controls)


def reset_controls_in_config(config: ConfigManager):
    config.set("operator_controls", {})


# ---------------- Modes ----------------

def run_hive(telemetry: TelemetryBus, seg_telemetry: SegmentedTelemetry,
             crash_replay: CrashReplay, interval: float = 0.5):
    queen = QueenBorg()
    print("The Borg Cache hive (Wave 3, Mesh) is running. Ctrl+C to stop.")
    try:
        while True:
            queen.background_tick()
            ov = queen.get_overview()

            telemetry.publish(ov)

            seg_telemetry.publish_all(
                coolant=ov["coolant"],
                cache=ov["cache"],
                borgio=ov["borg_io"],
                meta={
                    "identity": ov["identity"],
                    "controls": ov["controls"],
                    "hive": ov["hive"],
                    "mesh": ov["mesh"],
                },
            )

            crash_replay.record_snapshot(ov)

            time.sleep(interval)
    except KeyboardInterrupt:
        queen.shutdown()
        print("\nHive shutdown complete.")
    except Exception:
        try:
            crash_replay.save_on_crash(queen.io.op_log)
        except Exception:
            pass
        queen.shutdown()
        print("\nHive crashed; crash replay saved (if possible).")


def render_overview(ov: Dict[str, Any], hive_online: bool):
    os.system("cls" if os.name == "nt" else "clear")
    idt = ov["identity"]
    print(f"{idt['hive_name']} v{idt['version']}")
    print(f"GPU mode: {ov['scan']['gpu_mode']}")
    print("--- Hive ---")
    print(f"Status: {'ONLINE' if hive_online else 'OFFLINE'}")
    hive = ov.get("hive", {})
    print(f"Cluster: {hive.get('cluster', 'unknown')}")
    print("--- Coolant ---")
    c = ov["coolant"]
    print(f"CPU: {c.get('cpu_temp', 0):.1f}  RAM: {c.get('ram_temp', 0):.1f}  DISK: {c.get('disk_temp', 0):.1f}")
    print(f"VRAM: {c.get('vram_used_mb', 0)} / {c.get('vram_total_mb', 0)} MB  Pressure: {c.get('overall_pressure', 0):.1f}")
    print("--- Unified Cache ---")
    cache = ov["cache"]
    print(f"VRAM: {cache.get('vram_alloc_mb', 0)} MB  RAM: {cache.get('ram_alloc_mb', 0)} MB  SSD: {cache.get('ssd_alloc_mb', 0)} MB")
    print(f"Total: {cache.get('total_cache_mb', 0)} MB")
    print("--- BorgIO ---")
    b = ov["borg_io"]
    print(f"Reads: {b.get('reads', 0)}  Writes: {b.get('writes', 0)}  Hits: {b.get('cache_hits', 0)}  Misses: {b.get('cache_misses', 0)}")
    print(f"Evictions: {b.get('evictions', 0)}  Current: {b.get('current_bytes', 0)} / {b.get('max_cache_bytes', 0)} bytes  Entries: {b.get('entries', 0)}")
    print(f"Seq-heavy files: {b.get('files_sequential_heavy', 0)}  Random-heavy files: {b.get('files_random_heavy', 0)}")
    top_hot = b.get("top_hot_files", [])
    if top_hot:
        print("--- Top hot files ---")
        for item in top_hot:
            print(f"{item['path']}  ({item['accesses']} accesses)")
    print("--- Controls ---")
    ctrl = ov.get("controls", {})
    print(f"Prefetch: {ctrl.get('prefetch', 'n/a')}  AsyncFlush: {ctrl.get('async_flush', 'n/a')}  CacheLimit: {ctrl.get('cache_limit_mb', 'n/a')} MB")
    mesh = ov.get("mesh", {})
    peers = mesh.get("peers", {})
    print("--- Mesh ---")
    print(f"Peers: {len(peers)}")
    for nid, info in peers.items():
        print(f"{nid} @ {info.get('addr')}:{info.get('tcp_port')}  last_seen={info.get('last_seen', 0):.0f}")


def run_cockpit_telemetry(telemetry: TelemetryBus, seg_telemetry: SegmentedTelemetry,
                          interval: float = 0.5, offline_timeout: float = 5.0):
    print("The Borg Cache cockpit (telemetry mode, Wave 3). Ctrl+C to exit.")
    try:
        while True:
            seg = seg_telemetry.read_all()
            if seg is not None:
                meta = seg.get("meta", {})
                ov = {
                    "identity": meta.get("identity", {"hive_name": "The Borg Cache", "version": "segmented"}),
                    "scan": {"gpu_mode": "unknown"},
                    "coolant": seg.get("coolant", {}),
                    "cache": seg.get("cache", {}),
                    "borg_io": seg.get("borg_io", {}),
                    "controls": meta.get("controls", {}),
                    "hive": meta.get("hive", {}),
                    "mesh": meta.get("mesh", {}),
                }
                ts = seg.get("_telemetry_timestamp", 0)
                hive_online = (time.time() - ts) < offline_timeout
                render_overview(ov, hive_online)
                time.sleep(interval)
                continue

            ov = telemetry.read()
            if ov is None:
                os.system("cls" if os.name == "nt" else "clear")
                print("The Borg Cache cockpit (telemetry mode, Wave 3).")
                print("Hive: OFFLINE (no telemetry yet)")
                time.sleep(interval)
                continue

            ts = ov.get("_telemetry_timestamp", 0)
            hive_online = (time.time() - ts) < offline_timeout
            render_overview(ov, hive_online)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nCockpit exit.")


def cockpit_handle_keys(queen: QueenBorg):
    if not HAS_MSVCRT:
        return
    if not msvcrt.kbhit():
        return
    key = msvcrt.getch()
    try:
        ch = key.decode().lower()
    except Exception:
        return

    if ch == "1":
        queen.io.prefetch_aggressiveness = min(2, queen.io.prefetch_aggressiveness + 1)
    elif ch == "2":
        queen.io.prefetch_aggressiveness = max(0, queen.io.prefetch_aggressiveness - 1)
    elif ch == "3":
        queen.io.async_flush_enabled = not queen.io.async_flush_enabled
    elif ch == "4":
        queen.io.resize_cache(queen.io.max_cache_bytes + 256 * 1024 * 1024)
    elif ch == "5":
        queen.io.resize_cache(max(64 * 1024 * 1024, queen.io.max_cache_bytes - 256 * 1024 * 1024))
    elif ch == "s":
        save_controls_to_config(queen.config, queen)
    elif ch == "r":
        reset_controls_in_config(queen.config)


def run_cockpit_hybrid(interval: float = 0.5):
    queen = QueenBorg()
    print("The Borg Cache cockpit (Hybrid mode, Wave 3). Ctrl+C to exit.")
    if HAS_MSVCRT:
        print("Controls: [1]++prefetch [2]--prefetch [3]toggle async [4]++cache [5]--cache [S]save [R]reset")
    else:
        print("Keyboard controls disabled (non-Windows).")
    try:
        while True:
            queen.background_tick()
            ov = queen.get_overview()
            render_overview(ov, hive_online=True)
            if HAS_MSVCRT:
                cockpit_handle_keys(queen)
            time.sleep(interval)
    except KeyboardInterrupt:
        queen.shutdown()
        print("\nHive shutdown complete.")


# ---------------- Entry ----------------

def main():
    parser = argparse.ArgumentParser(description="The Borg Cache - Model A, Wave 3 (Aggressive, Mesh)")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "hive", "cockpit"],
        default="hybrid",
        help="hybrid (default): single-process; hive: headless producer; cockpit: telemetry viewer",
    )
    parser.add_argument(
        "--telemetry-file",
        default="borg_cache_telemetry.json",
        help="Telemetry JSON file path (for hive/cockpit modes)",
    )
    parser.add_argument(
        "--telemetry-name",
        default="Global\\BorgCacheTelemetry",
        help="Shared memory mapping name (Windows only, legacy bus)",
    )
    args = parser.parse_args()

    telemetry_cfg = TelemetryConfig(
        mapping_name=args.telemetry_name,
        file_path=args.telemetry_file,
    )
    telemetry = TelemetryBus(telemetry_cfg)
    seg_telemetry = SegmentedTelemetry()
    crash_replay = CrashReplay()

    if args.mode == "hybrid":
        run_cockpit_hybrid()
    elif args.mode == "hive":
        run_hive(telemetry, seg_telemetry, crash_replay)
    elif args.mode == "cockpit":
        run_cockpit_telemetry(telemetry, seg_telemetry)


if __name__ == "__main__":
    main()

