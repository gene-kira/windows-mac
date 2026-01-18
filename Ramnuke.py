#!/usr/bin/env python3
import sys
import subprocess
import importlib
import traceback
import threading
import time
import os

# -------------------------------------------------------------------
# 1. Auto-installer / autoloader
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
# 2. VRAMCacheManager (8 GB VRAM + 2 GB RAM, LRU + read-ahead)
# -------------------------------------------------------------------

class VRAMCacheManager:
    VRAM_TARGET_BYTES = 8 * 1024 * 1024 * 1024      # 8 GB
    RAM_TARGET_BYTES  = 2 * 1024 * 1024 * 1024      # 2 GB
    BLOCK_SIZE        = 256 * 1024                  # 256 KB
    READ_AHEAD_BLOCKS = 8
    MAX_BLOCKS        = 1_000_000                   # logical cap for map

    def __init__(self, logger, pycuda_driver=None, pycuda_autoinit=None):
        self.log = logger
        self.pycuda_driver = pycuda_driver
        self.pycuda_autoinit = pycuda_autoinit

        self.vram_available = False
        self.vram_buffer = None
        self.ram_buffer = None

        # block_id -> ("VRAM"/"RAM", offset)
        self.block_map = {}
        # LRU list: most recent at end
        self.lru_list = []
        self.lock = threading.Lock()
        self.running = False

        self.log("[CACHE] VRAMCacheManager constructed.")

    # ---------------- Initialization ----------------
    def initialize(self):
        self.log("[CACHE] Initializing VRAM + RAM tiers...")

        # VRAM
        if self.pycuda_driver and self.pycuda_autoinit and np is not None:
            try:
                import pycuda.gpuarray as gpuarray
                blocks = self.VRAM_TARGET_BYTES // self.BLOCK_SIZE
                floats_per_block = self.BLOCK_SIZE // 4
                total_floats = blocks * floats_per_block
                self.log(f"[CACHE] Allocating {self.VRAM_TARGET_BYTES/1e9:.1f} GB VRAM...")
                self.vram_buffer = gpuarray.zeros(total_floats, dtype=np.float32)
                self.vram_available = True
                self.log("[CACHE] VRAM allocation successful.")
            except Exception as e:
                self.log(f"[CACHE] VRAM allocation failed: {e}")
                self.log(traceback.format_exc())
                self.vram_available = False
        else:
            self.log("[CACHE] pycuda not available; VRAM tier disabled.")
            self.vram_available = False

        # RAM
        try:
            self.log(f"[CACHE] Allocating {self.RAM_TARGET_BYTES/1e9:.1f} GB RAM...")
            self.ram_buffer = np.zeros(self.RAM_TARGET_BYTES // 4, dtype=np.float32)
            self.log("[CACHE] RAM tier ready.")
        except Exception as e:
            self.log(f"[CACHE] RAM allocation failed: {e}")
            self.log(traceback.format_exc())

    def _block_offset(self, block_id):
        return block_id * (self.BLOCK_SIZE // 4)

    # ---------------- Core operations ----------------
    def read_block(self, block_id):
        with self.lock:
            if block_id in self.block_map:
                tier, offset = self.block_map[block_id]
                self._touch_lru(block_id)
                if tier == "VRAM" and self.vram_available:
                    return self.vram_buffer[offset : offset + (self.BLOCK_SIZE // 4)]
                else:
                    return self.ram_buffer[offset : offset + (self.BLOCK_SIZE // 4)]

            # miss
            data = self._load_from_disk(block_id)
            self._insert_block(block_id, data)
            self._read_ahead(block_id)
            return data

    def _insert_block(self, block_id, data):
        if len(self.block_map) >= self.MAX_BLOCKS:
            self._evict_lru()

        offset = self._block_offset(block_id)

        if self.vram_available:
            try:
                self.vram_buffer[offset : offset + len(data)] = data
                self.block_map[block_id] = ("VRAM", offset)
                self._touch_lru(block_id)
                return
            except Exception:
                self.log("[CACHE] VRAM write failed; falling back to RAM.")

        self.ram_buffer[offset : offset + len(data)] = data
        self.block_map[block_id] = ("RAM", offset)
        self._touch_lru(block_id)

    def _evict_lru(self):
        if not self.lru_list:
            return
        victim = self.lru_list.pop(0)
        if victim in self.block_map:
            del self.block_map[victim]
        self.log(f"[CACHE] Evicted block {victim} (LRU).")

    def _touch_lru(self, block_id):
        if block_id in self.lru_list:
            self.lru_list.remove(block_id)
        self.lru_list.append(block_id)

    def _read_ahead(self, block_id):
        for i in range(1, self.READ_AHEAD_BLOCKS + 1):
            nb = block_id + i
            if nb not in self.block_map:
                data = self._load_from_disk(nb)
                self._insert_block(nb, data)

    def _load_from_disk(self, block_id):
        # Placeholder: real implementation would map block_id to file/offset.
        # For now, just return zeros.
        size = self.BLOCK_SIZE // 4
        return np.zeros(size, dtype=np.float32)

    # ---------------- Telemetry & watchdog ----------------
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

    def watchdog_loop(self):
        self.log("[CACHE] Watchdog loop started.")
        while self.running:
            try:
                snap = self.telemetry_snapshot()
                self.log(
                    f"[CACHE][WD] blocks={snap['blocks']} "
                    f"VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']}"
                )
            except Exception as e:
                self.log(f"[CACHE][WD] error: {e}")
            time.sleep(5)

    # ---------------- Public control ----------------
    def start(self):
        self.log("[CACHE] Starting cache manager...")
        self.initialize()
        self.running = True
        t = threading.Thread(target=self.watchdog_loop, daemon=True)
        t.start()
        self.log("[CACHE] Cache manager is active.")

    def stop(self):
        self.running = False
        self.log("[CACHE] Cache manager stopped.")

# -------------------------------------------------------------------
# 3. Application-level: CachedFileInterface + DirectoryPreloader
# -------------------------------------------------------------------

class CachedFileInterface:
    """
    App-level interface that maps file offsets to cache blocks
    and uses VRAMCacheManager underneath.
    """
    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.block_size = cache_manager.BLOCK_SIZE

    def _block_id_for(self, path, offset):
        # Stable-ish mapping: hash(path) + block index
        return (hash(os.path.abspath(path)) & 0x7FFFFFFF) ^ (offset // self.block_size)

    def read(self, path, offset, length):
        """
        Read 'length' bytes from 'path' starting at 'offset',
        using VRAM cache for block-aligned chunks.
        """
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
        """Convenience: read entire file through the cache."""
        size = os.path.getsize(path)
        return self.read(path, 0, size)


class DirectoryPreloader:
    """
    Walks a directory and preloads files into the VRAM cache.
    Great for ML datasets, model weights, or game assets you care about.
    """
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
        self.log(f"[PRELOAD] Scanning {root}")
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                if not self._wanted(full):
                    continue
                try:
                    size = os.path.getsize(full)
                    if size > self.max_file_size:
                        self.log(f"[PRELOAD] Skipping large file {full} ({size/1e6:.1f} MB)")
                        continue
                    self.log(f"[PRELOAD] Caching {full} ({size/1e6:.1f} MB)")
                    _ = self.cf.read_all(full)
                except Exception as e:
                    self.log(f"[PRELOAD] Failed {full}: {e}")

# -------------------------------------------------------------------
# 4. GUI cockpit (Windows, fully autonomous)
# -------------------------------------------------------------------

class AutoLoaderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Cleaner – VRAM Cache Cockpit")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)

        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_gpu_info = QtWidgets.QPushButton("GPU / VRAM Info")
        self.btn_gpu_info.clicked.connect(self.handle_gpu_info)

        self.btn_preload = QtWidgets.QPushButton("Preload Directory into VRAM Cache")
        self.btn_preload.clicked.connect(self.handle_preload_dir)

        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)

        controls_layout.addWidget(self.btn_gpu_info)
        controls_layout.addWidget(self.btn_preload)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_exit)

        self.info_label = QtWidgets.QLabel(
            "Autonomous VRAM/RAM cache organism: auto-installs, auto-starts, preloads directories, self-monitors."
        )

        main_layout.addWidget(self.info_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.log_view)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.background_tick)

        self.cache = None
        self.file_cache = None
        self.preloader = None

        self.log("[SYSTEM] GUI initialized.")
        self.log_loaded_modules()

        # Autostart the caching engine after GUI is up
        QtCore.QTimer.singleShot(500, self.auto_start_engine)

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)
        print(msg)

    def log_loaded_modules(self):
        self.log("[AUTOLOADER] Module status:")
        self.log("  PyQt5: OK")
        self.log(f"  psutil: {'OK' if psutil else 'MISSING'}")
        self.log(f"  pyyaml: {'OK' if yaml else 'MISSING'}")
        self.log(f"  numpy: {'OK' if np is not None else 'MISSING'}")
        self.log(f"  pynvml: {'OK' if pynvml else 'MISSING'}")
        self.log(f"  pycuda: {'OK' if pycuda else 'MISSING'}")

    def auto_start_engine(self):
        self.log("[SYSTEM] Auto-starting caching engine...")
        self.start_caching_engine()
        self.timer.start()

    def handle_gpu_info(self):
        self.log("[ACTION] GPU / VRAM Info requested.")
        if not pynvml:
            self.log("[GPU] pynvml not available.")
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self.log(f"[GPU] {count} GPU(s) detected.")
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.log(
                    f"[GPU {i}] {name} | VRAM: {mem.total/1e9:.2f} GB total, "
                    f"{mem.used/1e9:.2f} GB used, {mem.free/1e9:.2f} GB free"
                )
            pynvml.nvmlShutdown()
        except Exception as e:
            self.log(f"[GPU] Failed to query: {e}")
            self.log(traceback.format_exc())

    def handle_preload_dir(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select directory to preload", ""
        )
        if not root:
            return
        self.log(f"[ACTION] Preloading directory: {root}")
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
            self.log("[ENGINE] VRAM/RAM cache manager + app-level interfaces are active.")
        except Exception as e:
            self.log(f"[ENGINE] Failed to start cache manager: {e}")
            self.log(traceback.format_exc())

    def background_tick(self):
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self.log(
                f"[TICK] CPU {cpu:.1f}% | RAM {mem.used/1e9:.2f}/{mem.total/1e9:.2f} GB"
            )
        if self.cache:
            snap = self.cache.telemetry_snapshot()
            self.log(
                f"[TICK][CACHE] blocks={snap['blocks']} "
                f"VRAM={snap['vram_blocks']} RAM={snap['ram_blocks']}"
            )

# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = AutoLoaderGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

