#!/usr/bin/env python
"""
The Borg Cache - Model A + Controls + Predictive Profiles
- Queen Borg
- Unified VRAM/RAM/SSD cache logic
- BorgIO block-level RAM cache with async flush, predictive prefetch, anomaly stats
- Per-file workload profiling (sequential vs random)
- Hot-region tracking and pinning
- Coolant engine with optional NVIDIA VRAM telemetry
- Self-healing workers and persistent policy memory
- Telemetry bus: shared memory dual-buffer + file fallback
- Hive / Cockpit / Hybrid modes
- Operator controls: prefetch, async flush, cache size
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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ---------------- Dependency checks ----------------

def ensure_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

# Required
if not ensure_import("psutil"):
    print("psutil is required. Install with: pip install psutil")
    sys.exit(1)

import psutil

HAS_PYNVML = ensure_import("pynvml")

if HAS_PYNVML:
    import pynvml

# Optional for keyboard controls (Windows)
HAS_MSVCRT = os.name == "nt"
if HAS_MSVCRT:
    import msvcrt


# ---------------- GPU probe ----------------

def get_nvidia_vram_usage_mb() -> Optional[tuple[int, int]]:
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
        cpu = psutil.cpu_percent(interval=0.1)
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
    version: str = "1.5.0"
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
        except Exception as e:
            print(f"[CONFIG] Failed to save config: {e}")

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

import zlib

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

        self.cache: Dict[tuple[str, int], Dict[str, Any]] = {}
        self.current_bytes = 0

        self.flush_queue: "queue.Queue[tuple[str, int, bytes]]" = queue.Queue()
        self.flush_running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()

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

    def ensure_flush_worker(self):
        if not self.flush_thread.is_alive() and self.flush_running:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def resize_cache(self, new_size_bytes: int):
        with self.lock:
            self.max_cache_bytes = max(new_size_bytes, BLOCK_SIZE)
            self._evict_if_needed()

    def _touch(self, key: tuple[str, int]):
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
        return zlib.compress(data)

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
        except Exception as e:
            print(f"[BORG_IO] Failed to read block {block_index} from {path}: {e}")
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
            try:
                path, block_index, data = self.flush_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._write_block_to_disk(path, block_index, data)
            except Exception as e:
                print(f"[BORG_IO] Flush error for {path} block {block_index}: {e}")

    def _enqueue_flush(self, path: str, block_index: int, data: bytes):
        self.flush_queue.put((path, block_index, data))

    def read_block(self, path: str, block_index: int) -> Optional[bytes]:
        key_path = str(Path(path).resolve())
        with self.lock:
            self.stats["reads"] += 1
            cached = self._get_block(key_path, block_index)
            if cached is not None:
                self.last_block[key_path] = block_index
                prof = self.file_profiles.get(key_path)
                if prof is None:
                    prof = FileProfile(path=key_path)
                    self.file_profiles[key_path] = prof
                prof.reads += 1
                prof.record_access(block_index)
                return cached

        data = self._read_block_from_disk(key_path, block_index)
        if data is None:
            return None

        with self.lock:
            self._put_block(key_path, block_index, data)
            self.last_block[key_path] = block_index

            prof = self.file_profiles.get(key_path)
            if prof is None:
                prof = FileProfile(path=key_path)
                self.file_profiles[key_path] = prof
            prof.reads += 1
            prof.record_access(block_index)

            effective_prefetch = self.prefetch_aggressiveness
            if prof.is_random_heavy:
                effective_prefetch = 0
            elif prof.is_sequential_heavy and self.prefetch_aggressiveness > 0:
                effective_prefetch = min(2, self.prefetch_aggressiveness + 1)

            if effective_prefetch > 0:
                last = self.last_block.get(key_path, block_index)
                if block_index == last + 1:
                    count = 1 if effective_prefetch == 1 else 4
                    for offset in range(1, count + 1):
                        idx = block_index + offset
                        if (key_path, idx) not in self.cache:
                            pre = self._read_block_from_disk(key_path, idx)
                            if pre is not None:
                                self._put_block(key_path, idx, pre)

        return data

    def write_block(self, path: str, block_index: int, data: bytes, async_flush: Optional[bool] = None):
        key_path = str(Path(path).resolve())
        with self.lock:
            self.stats["writes"] += 1
            self._put_block(key_path, block_index, data)

            prof = self.file_profiles.get(key_path)
            if prof is None:
                prof = FileProfile(path=key_path)
                self.file_profiles[key_path] = prof
            prof.writes += 1
            prof.record_access(block_index)

        use_async = self.async_flush_enabled if async_flush is None else async_flush
        if use_async:
            self._enqueue_flush(key_path, block_index, data)
        else:
            self._write_block_to_disk(key_path, block_index, data)

    def read_text(self, path: str, encoding: str = "utf-8") -> Optional[str]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = p.read_bytes()
        except Exception as e:
            print(f"[BORG_IO] Failed to read {path}: {e}")
            return None
        key_path = str(p.resolve())
        with self.lock:
            for i in range(0, len(data), BLOCK_SIZE):
                block_idx = i // BLOCK_SIZE
                self._put_block(key_path, block_idx, data[i:i + BLOCK_SIZE])
                prof = self.file_profiles.get(key_path)
                if prof is None:
                    prof = FileProfile(path=key_path)
                    self.file_profiles[key_path] = prof
                prof.reads += 1
                prof.record_access(block_idx)
        return data.decode(encoding, errors="ignore")

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> bool:
        data = text.encode(encoding)
        p = Path(path)
        key_path = str(p.resolve())
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        except Exception as e:
            print(f"[BORG_IO] Failed to write {path}: {e}")
            return False
        with self.lock:
            for i in range(0, len(data), BLOCK_SIZE):
                block_idx = i // BLOCK_SIZE
                self._put_block(key_path, block_idx, data[i:i + BLOCK_SIZE])
                prof = self.file_profiles.get(key_path)
                if prof is None:
                    prof = FileProfile(path=key_path)
                    self.file_profiles[key_path] = prof
                prof.writes += 1
                prof.record_access(block_idx)
        return True

    def snapshot_stats(self) -> Dict[str, Any]:
        with self.lock:
            seq_heavy = 0
            rnd_heavy = 0
            hot_files: list[tuple[str, int]] = []
            for prof in self.file_profiles.values():
                if prof.is_sequential_heavy:
                    seq_heavy += 1
                elif prof.is_random_heavy:
                    rnd_heavy += 1
                total_access = prof.reads + prof.writes
                if total_access > 0:
                    hot_files.append((prof.path, total_access))

            hot_files.sort(key=lambda kv: kv[1], reverse=True)
            top_hot = [{"path": p, "accesses": a} for p, a in hot_files[:5]]

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

    def shutdown(self):
        self.flush_running = False
        self.flush_thread.join(timeout=2.0)


# ---------------- Telemetry bus ----------------

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

        self._load_policy_memory()
        self._load_borgio_stats()

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

    def update_cache_strategy(self):
        c = self.coolant.sample()
        pressure = c.overall_pressure
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

    def _tune_borg_io(self):
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

    def background_tick(self):
        self.update_cache_strategy()
        self.apply_operator_controls()
        self._tune_borg_io()
        self.io.ensure_flush_worker()
        self.last_tick_time = time.time()

    def get_overview(self) -> Dict[str, Any]:
        c = self.coolant.sample()
        cache_state = self.cache.get_state()
        controls = {
            "prefetch": self.io.prefetch_aggressiveness,
            "async_flush": self.io.async_flush_enabled,
            "cache_limit_mb": self.io.max_cache_bytes // (1024 * 1024),
        }
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
            "borg_io": self.io.snapshot_stats(),
            "controls": controls,
            "hive": {
                "last_tick_time": self.last_tick_time,
            },
        }

    def shutdown(self):
        self.io.shutdown()


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

def run_hive(telemetry: TelemetryBus, interval: float = 2.0):
    queen = QueenBorg()
    print("The Borg Cache hive is running. Ctrl+C to stop.")
    try:
        while True:
            queen.background_tick()
            ov = queen.get_overview()
            telemetry.publish(ov)
            time.sleep(interval)
    except KeyboardInterrupt:
        queen.shutdown()
        print("\nHive shutdown complete.")


def render_overview(ov: Dict[str, Any], hive_online: bool):
    os.system("cls" if os.name == "nt" else "clear")
    idt = ov["identity"]
    print(f"{idt['hive_name']} v{idt['version']}")
    print(f"GPU mode: {ov['scan']['gpu_mode']}")
    print("--- Hive ---")
    print(f"Status: {'ONLINE' if hive_online else 'OFFLINE'}")
    print("--- Coolant ---")
    c = ov["coolant"]
    print(f"CPU: {c['cpu_temp']:.1f}  RAM: {c['ram_temp']:.1f}  DISK: {c['disk_temp']:.1f}")
    print(f"VRAM: {c['vram_used_mb']} / {c['vram_total_mb']} MB  Pressure: {c['overall_pressure']:.1f}")
    print("--- Unified Cache ---")
    cache = ov["cache"]
    print(f"VRAM: {cache['vram_alloc_mb']} MB  RAM: {cache['ram_alloc_mb']} MB  SSD: {cache['ssd_alloc_mb']} MB")
    print(f"Total: {cache['total_cache_mb']} MB")
    print("--- BorgIO ---")
    b = ov["borg_io"]
    print(f"Reads: {b['reads']}  Writes: {b['writes']}  Hits: {b['cache_hits']}  Misses: {b['cache_misses']}")
    print(f"Evictions: {b['evictions']}  Current: {b['current_bytes']} / {b['max_cache_bytes']} bytes  Entries: {b['entries']}")
    print(f"Seq-heavy files: {b.get('files_sequential_heavy', 0)}  Random-heavy files: {b.get('files_random_heavy', 0)}")
    top_hot = b.get("top_hot_files", [])
    if top_hot:
        print("--- Top hot files ---")
        for item in top_hot:
            print(f"{item['path']}  ({item['accesses']} accesses)")
    print("--- Controls ---")
    ctrl = ov.get("controls", {})
    print(f"Prefetch: {ctrl.get('prefetch', 'n/a')}  AsyncFlush: {ctrl.get('async_flush', 'n/a')}  CacheLimit: {ctrl.get('cache_limit_mb', 'n/a')} MB")


def run_cockpit_telemetry(telemetry: TelemetryBus, interval: float = 2.0, offline_timeout: float = 10.0):
    print("The Borg Cache cockpit (telemetry mode). Ctrl+C to exit.")
    try:
        while True:
            ov = telemetry.read()
            if ov is None:
                os.system("cls" if os.name == "nt" else "clear")
                print("The Borg Cache cockpit (telemetry mode).")
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


def run_cockpit_hybrid(interval: float = 2.0):
    queen = QueenBorg()
    print("The Borg Cache cockpit (Hybrid mode). Ctrl+C to exit.")
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
    parser = argparse.ArgumentParser(description="The Borg Cache - Model A + Predictive Profiles")
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
        help="Shared memory mapping name (Windows only)",
    )
    args = parser.parse_args()

    telemetry_cfg = TelemetryConfig(
        mapping_name=args.telemetry_name,
        file_path=args.telemetry_file,
    )
    telemetry = TelemetryBus(telemetry_cfg)

    if args.mode == "hybrid":
        run_cockpit_hybrid()
    elif args.mode == "hive":
        run_hive(telemetry)
    elif args.mode == "cockpit":
        run_cockpit_telemetry(telemetry)


if __name__ == "__main__":
    main()

