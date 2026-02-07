#!/usr/bin/env python
"""
The Borg Cache - Hybrid (automatic but observable) hive
- Queen Borg
- Unified VRAM/RAM/SSD cache logic
- BorgIO block-level RAM cache with async flush, predictive prefetch, anomaly stats
- Coolant engine with optional NVIDIA VRAM telemetry
- Optional UI automation watcher (uiautomation) to adapt behavior by active app
- Self-healing workers and persistent policy memory
"""

import os
import sys
import time
import threading
import queue
import json
import platform
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

# Optional
HAS_PYNVML = ensure_import("pynvml")
HAS_UIA = ensure_import("uiautomation")
HAS_PYTHONCOM = ensure_import("pythoncom")

if HAS_PYNVML:
    import pynvml
if HAS_UIA:
    import uiautomation as uia
if HAS_PYTHONCOM:
    import pythoncom


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
    version: str = "1.2.0"
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
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            with open(self.path, "w") as f:
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


class BorgIO:
    """
    Block-level RAM cache organ with:
    - block cache
    - async flush worker (self-healing)
    - pattern-aware prefetch
    - anomaly stats
    - Queen-controlled tuning
    """

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

        self.prefetch_aggressiveness = 0  # 0=off,1=light,2=aggressive
        self.async_flush_enabled = True

        self.last_block: Dict[str, int] = {}
        self.lock = threading.Lock()

    # ---- self-healing flush worker ----

    def ensure_flush_worker(self):
        if not self.flush_thread.is_alive() and self.flush_running:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    # ---- dynamic resizing ----

    def resize_cache(self, new_size_bytes: int):
        with self.lock:
            self.max_cache_bytes = max(new_size_bytes, BLOCK_SIZE)
            self._evict_if_needed()

    def _touch(self, key: tuple[str, int]):
        self.cache[key]["last_access"] = time.time()

    def _evict_if_needed(self):
        while self.current_bytes > self.max_cache_bytes and self.cache:
            oldest_key = min(
                self.cache.items(), key=lambda kv: kv[1]["last_access"]
            )[0]
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
                return cached

        data = self._read_block_from_disk(key_path, block_index)
        if data is None:
            return None

        with self.lock:
            self._put_block(key_path, block_index, data)
            self.last_block[key_path] = block_index

            if self.prefetch_aggressiveness > 0:
                last = self.last_block.get(key_path, block_index)
                if block_index == last + 1:
                    count = 1 if self.prefetch_aggressiveness == 1 else 4
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
        return True

    def snapshot_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "reads": self.stats["reads"],
                "writes": self.stats["writes"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "evictions": self.stats["evictions"],
                "current_bytes": self.current_bytes,
                "max_cache_bytes": self.max_cache_bytes,
                "entries": len(self.cache),
            }

    def shutdown(self):
        self.flush_running = False
        self.flush_thread.join(timeout=2.0)


# ---------------- UI Automation watcher ----------------

class UIAutomationWatcher(threading.Thread):
    """
    Optional: watches active window title and notifies Queen.
    Self-healing is handled by Queen if thread dies.
    """

    def __init__(self, queen, interval: float = 2.0):
        super().__init__(daemon=True)
        self.queen = queen
        self.interval = interval
        self.running = True

    def run(self):
        if not HAS_UIA or not HAS_PYTHONCOM:
            return

        pythoncom.CoInitialize()
        try:
            last_title = None
            while self.running:
                try:
                    win = uia.GetForegroundControl()
                    title = win.Name
                    if title != last_title:
                        last_title = title
                        self.queen.on_active_window_changed(title)
                except Exception:
                    pass
                time.sleep(self.interval)
        finally:
            pythoncom.CoUninitialize()

    def stop(self):
        self.running = False


# ---------------- Queen Borg ----------------

@dataclass
class LLMProfile:
    name: str
    kind: str
    endpoint: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueenBorg:
    def __init__(self):
        self.identity = HiveIdentity()
        self.config = ConfigManager()
        self.config.load()

        self.scanner = SystemScanner()
        self.scan_result = self.scanner.scan()
        self.gpu_mode = self.scan_result.gpu_mode

        self.coolant = CoolantEngine()
        self.cache = CacheEngine(gpu_mode=self.gpu_mode)
        self.llms: Dict[str, LLMProfile] = {}

        self._bootstrap_llms()

        self.io = BorgIO(self, max_cache_bytes=128 * 1024 * 1024)

        self.ui_watcher: Optional[UIAutomationWatcher] = None
        self._start_ui_watcher_if_possible()

        # load prior policy memory if any
        self._load_policy_memory()

        self.update_cache_strategy()

    # ---- LLMs ----

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

    # ---- UI watcher management ----

    def _start_ui_watcher_if_possible(self):
        if HAS_UIA and HAS_PYTHONCOM:
            self.ui_watcher = UIAutomationWatcher(self)
            self.ui_watcher.start()

    def _ensure_ui_watcher(self):
        if self.ui_watcher and not self.ui_watcher.is_alive():
            self._start_ui_watcher_if_possible()

    # ---- Active window influence ----

    def on_active_window_changed(self, title: str):
        t = title.lower()
        if any(x in t for x in ["visual studio", "code", "editor", "notepad"]):
            self.io.prefetch_aggressiveness = 2
        elif any(x in t for x in ["chrome", "edge", "firefox", "browser"]):
            self.io.prefetch_aggressiveness = 1
        else:
            self.io.prefetch_aggressiveness = 0

    # ---- Cache strategy ----

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

    # ---- Policy memory ----

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

    # ---- BorgIO tuning based on anomalies ----

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
        elif writes < reads / 4:
            self.io.async_flush_enabled = True

        self.config.set("borg_io_stats", stats)
        self._save_policy_memory()

    # ---- Background tick ----

    def background_tick(self):
        self.update_cache_strategy()
        self._tune_borg_io()
        self.io.ensure_flush_worker()
        self._ensure_ui_watcher()

    # ---- Snapshots ----

    def get_overview(self) -> Dict[str, Any]:
        c = self.coolant.sample()
        cache_state = self.cache.get_state()
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
        }

    # ---- Shutdown ----

    def shutdown(self):
        if self.ui_watcher:
            self.ui_watcher.stop()
        self.io.shutdown()


# ---------------- Simple text cockpit / service ----------------

def run_cockpit():
    queen = QueenBorg()
    print("The Borg Cache cockpit (Hybrid mode). Ctrl+C to exit.")
    try:
        while True:
            queen.background_tick()
            ov = queen.get_overview()
            os.system("cls" if os.name == "nt" else "clear")
            idt = ov["identity"]
            print(f"{idt['hive_name']} v{idt['version']}")
            print(f"GPU mode: {ov['scan']['gpu_mode']}")
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
            print(f"Prefetch: {queen.io.prefetch_aggressiveness}  AsyncFlush: {queen.io.async_flush_enabled}")
            time.sleep(2)
    except KeyboardInterrupt:
        queen.shutdown()
        print("\nHive shutdown complete.")


if __name__ == "__main__":
    run_cockpit()

