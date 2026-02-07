#!/usr/bin/env python
"""
The Borg Cache - Model A, Wave 3.2
Aggressive + Mesh + Defensive + Extended ML (long-range, cross-file, multi-file, deep-pattern scoring)
- Single monolithic file
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
import hmac
import hashlib
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
    version: str = "1.9.1-wave3.2-extended-ml"
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

        # Wave 3: per-file block history for n-gram model
        self.block_history: Dict[str, List[int]] = {}

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

    def _get_history(self, key_path: str) -> List[int]:
        hist = self.block_history.get(key_path)
        if hist is None:
            hist = []
            self.block_history[key_path] = hist
        return hist

    def _update_history(self, key_path: str, block_index: int):
        hist = self._get_history(key_path)
        hist.append(block_index)
        if len(hist) > 64:
            del hist[:-64]

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
            pass  # we still allow ML/global predictions even if not strictly sequential

        # Heuristic sequential prefetch
        if block_index == last + 1:
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
                    self._update_history(key_path, idx)
                    self._log_op("prefetch", key_path, idx, len(pre))

        # Wave 3: per-file ML n-gram predictions (multi-order)
        cluster = self.queen.cluster_engine.get_cluster()
        hist = self._get_history(key_path)
        ml_candidates = self.queen.ml_model.predict_next_blocks(
            key_path, hist, cluster=cluster, max_n=4
        )

        # Wave 3.2: global cross-file sequence predictions
        global_candidates = self.queen.global_seq_model.predict_next(max_n=4)

        # Merge candidates: (path, block) pairs
        merged: List[Tuple[str, int]] = []
        for idx in ml_candidates:
            merged.append((key_path, idx))
        for pth, blk in global_candidates:
            merged.append((pth, blk))

        # Deduplicate
        seen = set()
        final_candidates: List[Tuple[str, int]] = []
        for pth, blk in merged:
            if (pth, blk) in seen:
                continue
            seen.add((pth, blk))
            final_candidates.append((pth, blk))

        # Wave 3.2: deep pattern scoring to filter/rank
        scored: List[Tuple[float, str, int]] = []
        for pth, blk in final_candidates:
            score = self.queen.deep_pattern_engine.score(
                pth, blk, cluster, self.stats
            )
            scored.append((score, pth, blk))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Prefetch top few with positive score
        for score, pth, idx in scored[:8]:
            if score <= 0:
                continue
            if (pth, idx) in self.cache:
                continue
            pre = self._read_block_from_disk(pth, idx)
            if pre is not None:
                self._put_block(pth, idx, pre)
                p = self._get_profile(pth)
                p.reads += 1
                p.record_access(idx)
                self._update_history(pth, idx)
                kind = "ml_prefetch" if pth == key_path else "global_prefetch"
                self._log_op(kind, pth, idx, len(pre))

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
            self._update_history(key_path, block_index)
            self.last_block[key_path] = block_index
            # Wave 3.2: feed global sequence model
            self.queen.global_seq_model.observe(key_path, block_index, self.queen.cluster_engine.get_cluster())
            if cached is not None:
                self._log_op("read_hit", key_path, block_index, len(cached))
                return cached

        data = self._read_block_from_disk(key_path, block_index)
        if data is None:
            return None

        with self.lock:
            self._put_block(key_path, block_index, data)
            prof = self._get_profile(key_path)
            prof.reads += 1
            prof.record_access(block_index)
            self._update_history(key_path, block_index)
            self.last_block[key_path] = block_index
            self.queen.global_seq_model.observe(key_path, block_index, self.queen.cluster_engine.get_cluster())
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
            self._update_history(key_path, block_index)
            self.last_block[key_path] = block_index
            self.queen.global_seq_model.observe(key_path, block_index, self.queen.cluster_engine.get_cluster())
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
                self._update_history(key_path, block_idx)
                self.last_block[key_path] = block_idx
                self.queen.global_seq_model.observe(key_path, block_idx, self.queen.cluster_engine.get_cluster())
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
                self._update_history(key_path, block_idx)
                self.last_block[key_path] = block_idx
                self.queen.global_seq_model.observe(key_path, block_idx, self.queen.cluster_engine.get_cluster())
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


# ---------------- Crash replay ----------------

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


# ---------------- Wave 3: ML Prefetch Model (multi-order n-gram + decay + cluster-aware) ----------------

class MLPrefetchModel:
    """
    Multi-order n-gram model:
    - orders: [1, 2, 3, 4]
    - backoff: try highest order first, then lower
    """
    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        # transitions[(file_key, order, history_tuple)] = {next_block: weight}
        self.transitions: Dict[Tuple[str, int, Tuple[int, ...]], Dict[int, int]] = {}
        self.lock = threading.Lock()

    def _file_key(self, path: str, cluster: str | None = None) -> str:
        directory = os.path.dirname(path)
        return f"{cluster or 'global'}::{directory}"

    def observe(self, path: str, history: List[int], next_block: int, cluster: str | None = None):
        if not history:
            return
        fk = self._file_key(path, cluster)
        with self.lock:
            for order in range(1, self.max_order + 1):
                if len(history) < order:
                    continue
                key = (fk, order, tuple(history[-order:]))
                file_map = self.transitions.setdefault(key, {})
                file_map[next_block] = file_map.get(next_block, 0) + 1

    def predict_next_blocks(self, path: str, history: List[int], cluster: str | None = None, max_n: int = 4) -> List[int]:
        if not history:
            return []
        fk = self._file_key(path, cluster)
        with self.lock:
            scores: Dict[int, float] = {}
            for order in range(self.max_order, 0, -1):
                if len(history) < order:
                    continue
                key = (fk, order, tuple(history[-order:]))
                trans = self.transitions.get(key)
                if not trans:
                    continue
                weight_factor = 1.0 + (order - 1) * 0.3
                for blk, w in trans.items():
                    scores[blk] = scores.get(blk, 0.0) + w * weight_factor
            if not scores:
                return []
            sorted_targets = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            return [b for b, _ in sorted_targets[:max_n]]

    def decay(self, factor: float = 0.9, min_weight: int = 1):
        with self.lock:
            to_delete = []
            for key, trans in self.transitions.items():
                for blk, w in list(trans.items()):
                    new_w = int(w * factor)
                    if new_w < min_weight:
                        del trans[blk]
                    else:
                        trans[blk] = new_w
                if not trans:
                    to_delete.append(key)
            for key in to_delete:
                del self.transitions[key]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"max_order": self.max_order, "transitions": self.transitions}

    def load(self, data: Dict[str, Any]):
        with self.lock:
            self.max_order = int(data.get("max_order", 4))
            self.transitions = data.get("transitions", {})


# ---------------- Wave 3.2: Global cross-file sequence model ----------------

class GlobalSequenceModel:
    """
    Tracks cross-file temporal sequences:
    - sequence of (file_key, block)
    - learns transitions between (file_key, block) pairs
    - predicts next (file_key, block) across files
    """
    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self.sequence: List[Tuple[str, int]] = []
        # transitions[(file_key, block)] = { (next_file_key, next_block): weight }
        self.transitions: Dict[Tuple[str, int], Dict[Tuple[str, int], int]] = {}
        self.lock = threading.Lock()

    def observe(self, path: str, block: int, cluster: str | None = None):
        fk = self._file_key(path, cluster)
        with self.lock:
            if self.sequence:
                prev_fk, prev_blk = self.sequence[-1]
                key = (prev_fk, prev_blk)
                nxt = (fk, block)
                m = self.transitions.setdefault(key, {})
                m[nxt] = m.get(nxt, 0) + 1
            self.sequence.append((fk, block))
            if len(self.sequence) > self.max_len:
                self.sequence = self.sequence[-self.max_len:]

    def _file_key(self, path: str, cluster: str | None = None) -> str:
        directory = os.path.dirname(path)
        return f"{cluster or 'global'}::{directory}"

    def predict_next(self, max_n: int = 4) -> List[Tuple[str, int]]:
        with self.lock:
            if not self.sequence:
                return []
            last_fk, last_blk = self.sequence[-1]
            key = (last_fk, last_blk)
            trans = self.transitions.get(key)
            if not trans:
                return []
            sorted_targets = sorted(trans.items(), key=lambda kv: kv[1], reverse=True)
            return [(fk, blk) for (fk, blk), _ in sorted_targets[:max_n]]


# ---------------- Wave 3.2: Deep pattern scoring engine ----------------

class DeepPatternEngine:
    """
    Lightweight "deep" scoring:
    - feature vector: [norm_block, log_reads, log_writes, miss_rate, cluster_hash]
    - single hidden layer with tanh, trained online with simple gradient-like updates
    This is not a full DL framework, but enough to capture richer patterns.
    """
    def __init__(self, hidden_size: int = 8, lr: float = 0.01):
        self.hidden_size = hidden_size
        self.lr = lr
        self.input_size = 5
        # weights: input -> hidden, hidden -> output
        self.W1 = [[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.W2 = [0.0 for _ in range(self.hidden_size)]
        self.b2 = 0.0
        self.lock = threading.Lock()

    def _tanh(self, x: float) -> float:
        # simple tanh approximation
        import math
        return math.tanh(x)

    def _features(self, path: str, block: int, cluster: str, stats: Dict[str, Any]) -> List[float]:
        import math
        norm_block = 1.0 / (1.0 + math.exp(-block / 1024.0))  # squashed
        reads = stats.get("reads", 0)
        writes = stats.get("writes", 0)
        misses = stats.get("cache_misses", 0)
        miss_rate = (misses / reads) if reads > 0 else 0.0
        f_reads = math.log(1 + reads)
        f_writes = math.log(1 + writes)
        ch = float(abs(hash(cluster)) % 1000) / 1000.0
        return [norm_block, f_reads, f_writes, miss_rate, ch]

    def score(self, path: str, block: int, cluster: str, stats: Dict[str, Any]) -> float:
        x = self._features(path, block, cluster, stats)
        with self.lock:
            h = []
            for i in range(self.hidden_size):
                s = self.b1[i]
                for j in range(self.input_size):
                    s += self.W1[i][j] * x[j]
                h.append(self._tanh(s))
            out = self.b2
            for i in range(self.hidden_size):
                out += self.W2[i] * h[i]
        return out

    def train_positive(self, path: str, block: int, cluster: str, stats: Dict[str, Any]):
        # simple positive reinforcement: push score up
        x = self._features(path, block, cluster, stats)
        with self.lock:
            h_raw = []
            h = []
            for i in range(self.hidden_size):
                s = self.b1[i]
                for j in range(self.input_size):
                    s += self.W1[i][j] * x[j]
                h_raw.append(s)
                h.append(self._tanh(s))
            out = self.b2
            for i in range(self.hidden_size):
                out += self.W2[i] * h[i]
            # target = +1
            err = 1.0 - out
            # update W2, b2
            for i in range(self.hidden_size):
                self.W2[i] += self.lr * err * h[i]
            self.b2 += self.lr * err
            # backprop to W1, b1 (approx)
            for i in range(self.hidden_size):
                dh = (1.0 - (self._tanh(h_raw[i]) ** 2)) * self.W2[i] * err
                for j in range(self.input_size):
                    self.W1[i][j] += self.lr * dh * x[j]
                self.b1[i] += self.lr * dh


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


# ---------------- Wave 3.1: Defensive Mesh Manager ----------------

class MeshManager:
    UDP_PORT = 49300
    TCP_PORT = 49301

    def __init__(self, identity: HiveIdentity, shared_secret: str = "borg_mesh_secret"):
        self.identity = identity
        self.node_id = f"{socket.gethostname()}-{os.getpid()}"
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.running = True

        self.shared_secret = shared_secret.encode("utf-8")
        self.conn_semaphore = threading.BoundedSemaphore(value=16)
        self.last_sync: Dict[str, float] = {}
        self.sync_cooldown = 2.0
        self.trust_scores: Dict[str, float] = {}

        self.udp_thread = threading.Thread(target=self._udp_broadcast_loop, daemon=True)
        self.udp_thread.start()

        self.udp_listener_thread = threading.Thread(target=self._udp_listen_loop, daemon=True)
        self.udp_listener_thread.start()

        self.tcp_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
        self.tcp_thread.start()

        self.last_mesh_snapshot: Dict[str, Any] = {}

    def _sign(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hmac.new(self.shared_secret, raw, hashlib.sha256).hexdigest()

    def _verify(self, payload: Dict[str, Any], signature: str) -> bool:
        expected = self._sign(payload)
        return hmac.compare_digest(expected, signature)

    def _update_trust(self, node_id: str, delta: float):
        with self.lock:
            cur = self.trust_scores.get(node_id, 0.0)
            cur += delta
            cur = max(-5.0, min(5.0, cur))
            self.trust_scores[node_id] = cur

    def _get_trust(self, node_id: str) -> float:
        with self.lock:
            return self.trust_scores.get(node_id, 0.0)

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
                sig = self._sign(payload)
                msg = {"payload": payload, "sig": sig}
                data = json.dumps(msg).encode("utf-8")
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
                payload = msg.get("payload", {})
                sig = msg.get("sig", "")
                node_id = payload.get("node_id")
                if not node_id or node_id == self.node_id:
                    continue
                if not self._verify(payload, sig):
                    self._update_trust(node_id, -1.0)
                    continue
                self._update_trust(node_id, 0.1)
                with self.lock:
                    self.peers[node_id] = {
                        "addr": addr[0],
                        "tcp_port": payload.get("tcp_port", self.TCP_PORT),
                        "hive_name": payload.get("hive_name"),
                        "version": payload.get("version"),
                        "last_seen": payload.get("t", time.time()),
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
            if not self.conn_semaphore.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=self._handle_tcp_client, args=(conn,), daemon=True).start()

    def _handle_tcp_client(self, conn: socket.socket):
        try:
            conn.settimeout(1.0)
            data = conn.recv(65536)
            if not data:
                conn.close()
                return
            msg = json.loads(data.decode("utf-8", errors="ignore"))
            if msg.get("type") == "mesh_sync":
                payload = msg.get("payload", {})
                sig = msg.get("sig", "")
                from_id = msg.get("from")
                if not from_id:
                    return
                if not self._verify(payload, sig):
                    self._update_trust(from_id, -1.0)
                    return
                if self._get_trust(from_id) < -2.0:
                    return
                with self.lock:
                    self.last_mesh_snapshot = payload
                self._update_trust(from_id, 0.2)
                reply = json.dumps({"type": "mesh_ack"}).encode("utf-8")
                conn.sendall(reply)
        except Exception:
            pass
        finally:
            conn.close()
            self.conn_semaphore.release()

    def sync_with_peers(self, local_snapshot: Dict[str, Any]):
        now = time.time()
        with self.lock:
            peers = list(self.peers.items())
        for node_id, info in peers:
            if self._get_trust(node_id) < -2.0:
                continue
            last = self.last_sync.get(node_id, 0)
            if now - last < self.sync_cooldown:
                continue
            self.last_sync[node_id] = now
            addr = info.get("addr")
            port = info.get("tcp_port", self.TCP_PORT)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect((addr, port))
                payload = local_snapshot
                sig = self._sign(payload)
                msg = {
                    "type": "mesh_sync",
                    "from": self.node_id,
                    "payload": payload,
                    "sig": sig,
                }
                sock.sendall(json.dumps(msg).encode("utf-8"))
                try:
                    _ = sock.recv(4096)
                except Exception:
                    pass
                sock.close()
                self._update_trust(node_id, 0.1)
            except Exception:
                self._update_trust(node_id, -0.1)
                continue

    def get_mesh_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.last_mesh_snapshot)

    def get_peers(self) -> Dict[str, Any]:
        with self.lock:
            peers = dict(self.peers)
            trusts = dict(self.trust_scores)
        for nid, info in peers.items():
            info["trust"] = trusts.get(nid, 0.0)
        return peers

    def shutdown(self):
        self.running = False


# ---------------- Remote Cockpit Server ----------------

class RemoteCockpitServer:
    PORT = 49302

    def __init__(self, queen_get_overview_callable):
        self.get_overview = queen_get_overview_callable
        self.running = True
        self.client_last: Dict[str, float] = {}
        self.client_cooldown = 0.2
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
            threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def _handle_client(self, conn: socket.socket, addr):
        try:
            conn.settimeout(1.0)
            ip = addr[0]
            now = time.time()
            last = self.client_last.get(ip, 0)
            if now - last < self.client_cooldown:
                conn.close()
                return
            self.client_last[ip] = now

            data = conn.recv(8192)
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


# ---------------- Persistence Engine ----------------

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

        self.ml_model = MLPrefetchModel(max_order=4)
        self.global_seq_model = GlobalSequenceModel(max_len=64)
        self.deep_pattern_engine = DeepPatternEngine(hidden_size=8, lr=0.01)
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
        self.last_decay_time = time.time()

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
                if op["kind"] in ("read_hit", "read_miss", "prefetch", "ml_prefetch", "global_prefetch"):
                    path = op["path"]
                    block = op["block"]
                    key_path = path
                    hist = self.io._get_history(key_path)
                    cluster = self.cluster_engine.get_cluster()
                    self.ml_model.observe(path, hist, block, cluster=cluster)
                    # Deep pattern reinforcement for successful accesses
                    self.deep_pattern_engine.train_positive(path, block, cluster, self.io.stats)

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

        now = time.time()
        if now - self.last_decay_time > 5.0:
            self.ml_model.decay(factor=0.9, min_weight=1)
            self.last_decay_time = now

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
    print("The Borg Cache hive (Wave 3.2, Extended ML) is running. Ctrl+C to stop.")
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
        print(f"{nid} @ {info.get('addr')}:{info.get('tcp_port')}  trust={info.get('trust', 0):.1f}  last_seen={info.get('last_seen', 0):.0f}")


def run_cockpit_telemetry(telemetry: TelemetryBus, seg_telemetry: SegmentedTelemetry,
                          interval: float = 0.5, offline_timeout: float = 5.0):
    print("The Borg Cache cockpit (telemetry mode, Wave 3.2). Ctrl+C to exit.")
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
                print("The Borg Cache cockpit (telemetry mode, Wave 3.2).")
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
    print("The Borg Cache cockpit (Hybrid mode, Wave 3.2). Ctrl+C to exit.")
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
    parser = argparse.ArgumentParser(description="The Borg Cache - Model A, Wave 3.2 (Extended ML, Mesh, Defensive)")
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

