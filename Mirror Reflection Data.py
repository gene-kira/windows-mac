#!/usr/bin/env python3
"""
Autonomous Telemetry Dashboard
- Per-plugin enable/disable
- Threshold alerts
- Overlay (always-on-top) mode
- Auto-loader for psutil, py-cpuinfo, pynvml
- Adaptive sampling + manual override
- CSV logging with rotating session files
- Graceful shutdown and thread-safe updates
"""
import sys
import os
import time
import csv
import threading
import queue
import importlib
import subprocess
import platform
from datetime import datetime
from collections import deque

# ---------------------------
# Auto-loader for libraries
# ---------------------------
def ensure(lib_name, package_name=None):
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        pkg = package_name or lib_name
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            importlib.invalidate_caches()
            importlib.import_module(lib_name)
            return True
        except Exception:
            return False

HAS_PSUTIL   = ensure("psutil")
HAS_CPUINFO  = ensure("cpuinfo", "py-cpuinfo")
HAS_PYNVML   = ensure("pynvml")

# Tkinter (built-in for most Python distributions)
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except Exception as e:
    print("Tkinter not available:", e)
    sys.exit(1)

import random  # fallback demo values

# ---------------------------
# Utility helpers
# ---------------------------
def safe_get(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def fmt_bytes(n):
    try:
        for unit in ["B","KB","MB","GB","TB","PB"]:
            if n < 1024.0:
                return f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} EB"
    except Exception:
        return "n/a"

def rolling_file(base_dir="telemetry_logs"):
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"session_{ts}.csv")

# ---------------------------
# Collector plugins
# ---------------------------
class CollectorBase:
    name = "base"
    def collect(self):
        return {}

class CPUCollector(CollectorBase):
    name = "cpu"
    def __init__(self):
        self.last_freq = None

    def collect(self):
        data = {}
        if HAS_PSUTIL:
            import psutil
            data["cpu_percent_per_core"] = psutil.cpu_percent(percpu=True)
            data["cpu_percent_total"] = psutil.cpu_percent()
            freq = safe_get(lambda: psutil.cpu_freq(), None)
            if freq:
                data["cpu_freq_current_mhz"] = freq.current
                data["cpu_freq_min_mhz"] = freq.min
                data["cpu_freq_max_mhz"] = freq.max
            # Load average (Unix only)
            if hasattr(os, "getloadavg"):
                try:
                    la1, la5, la15 = os.getloadavg()
                    data["load_avg_1m"] = la1
                    data["load_avg_5m"] = la5
                    data["load_avg_15m"] = la15
                except Exception:
                    pass
            # Temps
            temps = safe_get(lambda: psutil.sensors_temperatures(), {})
            if temps:
                flat = {}
                for k, arr in temps.items():
                    if arr:
                        t = arr[0]
                        flat[f"temp_{k}"] = getattr(t, "current", None)
                data["temps"] = flat
            # Battery (laptops)
            batt = safe_get(lambda: psutil.sensors_battery(), None)
            if batt:
                data["battery_percent"] = batt.percent
                data["battery_plugged"] = bool(batt.power_plugged)
        else:
            # Fallback for demo
            data["cpu_percent_total"] = random.uniform(5, 50)
            data["cpu_percent_per_core"] = [random.uniform(5, 50) for _ in range(4)]
        # Static CPU info
        if HAS_CPUINFO:
            from cpuinfo import get_cpu_info
            info = safe_get(get_cpu_info, {})
            if info:
                data["cpu_brand"] = info.get("brand_raw")
                data["arch"] = info.get("arch")
        else:
            data["cpu_brand"] = platform.processor()
            data["arch"] = platform.machine()
        return data

class MemoryCollector(CollectorBase):
    name = "memory"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            import psutil
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            data["mem_total"] = vm.total
            data["mem_used"] = vm.used
            data["mem_available"] = vm.available
            data["mem_percent"] = vm.percent
            data["swap_total"] = sm.total
            data["swap_used"] = sm.used
            data["swap_percent"] = sm.percent
        return data

class DiskCollector(CollectorBase):
    name = "disk"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            import psutil
            io = psutil.disk_io_counters()
            if io:
                data["disk_read_bytes"] = io.read_bytes
                data["disk_write_bytes"] = io.write_bytes
                data["disk_read_count"] = io.read_count
                data["disk_write_count"] = io.write_count
        return data

class NetCollector(CollectorBase):
    name = "network"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            import psutil
            io = psutil.net_io_counters()
            if io:
                data["net_bytes_sent"] = io.bytes_sent
                data["net_bytes_recv"] = io.bytes_recv
                data["net_packets_sent"] = io.packets_sent
                data["net_packets_recv"] = io.packets_recv
        return data

class ProcessCollector(CollectorBase):
    name = "process"
    def collect(self):
        data = {}
        if HAS_PSUTIL:
            import psutil
            procs = []
            for p in psutil.process_iter(attrs=["pid","name","cpu_percent","memory_info"]):
                try:
                    info = p.info
                    procs.append({
                        "pid": info["pid"],
                        "name": info.get("name",""),
                        "cpu": info.get("cpu_percent", 0.0),
                        "rss": getattr(info.get("memory_info"), "rss", 0)
                    })
                except Exception:
                    continue
            procs.sort(key=lambda x: x["cpu"], reverse=True)
            data["top_processes"] = procs[:8]
        return data

class GPUCollector(CollectorBase):
    name = "gpu"
    def __init__(self):
        self.nv_ok = False
        if HAS_PYNVML:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.nv_ok = True
            except Exception:
                self.nv_ok = False

    def collect(self):
        data = {}
        if self.nv_ok:
            nv = self.pynvml
            try:
                count = nv.nvmlDeviceGetCount()
                gpus = []
                for i in range(count):
                    h = nv.nvmlDeviceGetHandleByIndex(i)
                    name = nv.nvmlDeviceGetName(h).decode()
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                    gpus.append({
                        "index": i,
                        "name": name,
                        "gpu_util": util.gpu,
                        "mem_util": (mem.used / mem.total * 100.0) if mem.total else 0.0,
                        "mem_total": mem.total,
                        "mem_used": mem.used,
                        "temp_c": temp
                    })
                data["gpus"] = gpus
            except Exception:
                pass
        return data

# ---------------------------
# Telemetry engine
# ---------------------------
DEFAULT_THRESHOLDS = {
    "cpu_total_percent": 85.0,
    "mem_percent": 90.0,
    "disk_write_bps": 100 * 1024 * 1024,  # 100 MB/s
    "disk_read_bps": 100 * 1024 * 1024,   # 100 MB/s
    "net_send_bps": 50 * 1024 * 1024,     # 50 MB/s
    "net_recv_bps": 50 * 1024 * 1024,     # 50 MB/s
    "temp_c": 85.0,                        # GPU/CPU temps threshold
}

class TelemetryEngine:
    def __init__(self, interval=1.0, adaptive=True, enabled_plugins=None, thresholds=None):
        self.interval = interval
        self.adaptive = adaptive
        self.enabled_plugins = enabled_plugins or {
            "cpu": True,
            "memory": True,
            "disk": True,
            "network": True,
            "process": True,
            "gpu": True,
        }
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()

        # Instantiate collectors
        self.collectors_map = {
            "cpu": CPUCollector(),
            "memory": MemoryCollector(),
            "disk": DiskCollector(),
            "network": NetCollector(),
            "process": ProcessCollector(),
            "gpu": GPUCollector(),
        }

        self._stop = threading.Event()
        self._thread = None
        self.out_queue = queue.Queue(maxsize=64)
        self.last_cpu_util = deque(maxlen=5)
        self.logging_enabled = False
        self.log_file = None
        self.csv_writer = None

        # For rate calculations (disk/net)
        self._prev_disk = None
        self._prev_net = None
        self._prev_time = None

        self.alerts = []  # latest alerts

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_log()

    def set_interval(self, interval):
        self.interval = max(0.2, float(interval))

    def set_adaptive(self, adaptive):
        self.adaptive = bool(adaptive)

    def update_enabled_plugins(self, mapping):
        self.enabled_plugins.update(mapping)

    def update_thresholds(self, mapping):
        self.thresholds.update(mapping)

    def enable_logging(self, enable=True):
        self.logging_enabled = bool(enable)
        if enable and not self.log_file:
            self._open_log()
        elif not enable:
            self._close_log()

    def _open_log(self):
        try:
            self.log_file_path = rolling_file()
            self.log_file = open(self.log_file_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.log_file)
            # header written with first row
        except Exception:
            self.log_file = None
            self.csv_writer = None

    def _close_log(self):
        try:
            if self.log_file:
                self.log_file.close()
        finally:
            self.log_file = None
            self.csv_writer = None

    def _flatten_row(self, snapshot):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cpu_total": snapshot.get("cpu", {}).get("cpu_percent_total"),
            "mem_percent": snapshot.get("memory", {}).get("mem_percent"),
            "disk_read": snapshot.get("disk", {}).get("disk_read_bytes"),
            "disk_write": snapshot.get("disk", {}).get("disk_write_bytes"),
            "net_sent": snapshot.get("network", {}).get("net_bytes_sent"),
            "net_recv": snapshot.get("network", {}).get("net_bytes_recv"),
            "load1": snapshot.get("cpu", {}).get("load_avg_1m"),
            "gpu0_util": None,
        }
        g = snapshot.get("gpu", {}).get("gpus", [])
        if g:
            row["gpu0_util"] = g[0].get("gpu_util")
        return row

    def _check_alerts(self, snapshot, deltas):
        alerts = []
        thr = self.thresholds
        cpu_total = snapshot.get("cpu", {}).get("cpu_percent_total", 0.0)
        if cpu_total is not None and cpu_total >= thr["cpu_total_percent"]:
            alerts.append(f"CPU total {cpu_total:.1f}% >= {thr['cpu_total_percent']:.1f}%")

        mem_percent = snapshot.get("memory", {}).get("mem_percent", 0.0)
        if mem_percent is not None and mem_percent >= thr["mem_percent"]:
            alerts.append(f"Memory {mem_percent:.1f}% >= {thr['mem_percent']:.1f}%")

        # Disk/network deltas are bytes per second
        if deltas:
            dw = deltas.get("disk_write_bps", 0.0)
            dr = deltas.get("disk_read_bps", 0.0)
            ns = deltas.get("net_send_bps", 0.0)
            nr = deltas.get("net_recv_bps", 0.0)
            if dw >= thr["disk_write_bps"]:
                alerts.append(f"Disk write {fmt_bytes(dw)}/s >= {fmt_bytes(thr['disk_write_bps'])}/s")
            if dr >= thr["disk_read_bps"]:
                alerts.append(f"Disk read {fmt_bytes(dr)}/s >= {fmt_bytes(thr['disk_read_bps'])}/s")
            if ns >= thr["net_send_bps"]:
                alerts.append(f"Net send {fmt_bytes(ns)}/s >= {fmt_bytes(thr['net_send_bps'])}/s")
            if nr >= thr["net_recv_bps"]:
                alerts.append(f"Net recv {fmt_bytes(nr)}/s >= {fmt_bytes(thr['net_recv_bps'])}/s")

        # Temps
        temps = snapshot.get("cpu", {}).get("temps", {}) or {}
        for k, v in temps.items():
            if v is not None and v >= thr["temp_c"]:
                alerts.append(f"{k} {v:.1f}°C >= {thr['temp_c']:.1f}°C")

        # GPU temps/util
        gpus = snapshot.get("gpu", {}).get("gpus", []) or []
        for g in gpus:
            t = g.get("temp_c")
            if t is not None and t >= thr["temp_c"]:
                alerts.append(f"GPU{g.get('index')} {t:.1f}°C >= {thr['temp_c']:.1f}°C")

        self.alerts = alerts

    def _calc_deltas(self, snapshot, now):
        deltas = {}
        # Disk read/write BPS
        if self._prev_disk and self._prev_time and "disk" in snapshot:
            dt = max(0.001, now - self._prev_time)
            d_prev = self._prev_disk
            d_curr = snapshot["disk"]
            if d_prev and d_curr:
                if d_prev.get("disk_write_bytes") is not None and d_curr.get("disk_write_bytes") is not None:
                    deltas["disk_write_bps"] = (d_curr["disk_write_bytes"] - d_prev["disk_write_bytes"]) / dt
                if d_prev.get("disk_read_bytes") is not None and d_curr.get("disk_read_bytes") is not None:
                    deltas["disk_read_bps"] = (d_curr["disk_read_bytes"] - d_prev["disk_read_bytes"]) / dt

        # Net send/recv BPS
        if self._prev_net and self._prev_time and "network" in snapshot:
            dt = max(0.001, now - self._prev_time)
            n_prev = self._prev_net
            n_curr = snapshot["network"]
            if n_prev and n_curr:
                if n_prev.get("net_bytes_sent") is not None and n_curr.get("net_bytes_sent") is not None:
                    deltas["net_send_bps"] = (n_curr["net_bytes_sent"] - n_prev["net_bytes_sent"]) / dt
                if n_prev.get("net_bytes_recv") is not None and n_curr.get("net_bytes_recv") is not None:
                    deltas["net_recv_bps"] = (n_curr["net_bytes_recv"] - n_prev["net_bytes_recv"]) / dt
        return deltas

    def _run(self):
        header_written = False
        while not self._stop.is_set():
            start = time.time()
            snapshot = {}
            for key, collector in self.collectors_map.items():
                if not self.enabled_plugins.get(key, True):
                    continue
                try:
                    snapshot[key] = collector.collect()
                except Exception:
                    snapshot[key] = {}
            # Adaptive interval based on recent CPU load
            cpu_total = snapshot.get("cpu", {}).get("cpu_percent_total", 0.0)
            self.last_cpu_util.append(cpu_total if cpu_total is not None else 0.0)
            if self.adaptive and len(self.last_cpu_util) >= 3:
                avg = sum(self.last_cpu_util) / len(self.last_cpu_util)
                if avg > 75:
                    target = max(0.5, self.interval * 1.5)
                elif avg < 20:
                    target = max(0.2, self.interval * 0.8)
                else:
                    target = self.interval
            else:
                target = self.interval

            now = time.time()
            deltas = self._calc_deltas(snapshot, now)
            self._check_alerts(snapshot, deltas)

            # store prev for rate calc
            self._prev_time = now
            self._prev_disk = snapshot.get("disk", None)
            self._prev_net = snapshot.get("network", None)

            # Enqueue snapshot with deltas + alerts
            snapshot["_deltas"] = deltas
            snapshot["_alerts"] = list(self.alerts)
            try:
                self.out_queue.put(snapshot, timeout=0.1)
            except queue.Full:
                pass

            # Logging
            if self.logging_enabled and self.csv_writer:
                row = self._flatten_row(snapshot)
                if not header_written:
                    self.csv_writer.writerow(list(row.keys()))
                    header_written = True
                self.csv_writer.writerow(list(row.values()))
                try:
                    self.log_file.flush()
                except Exception:
                    pass

            elapsed = time.time() - start
            remaining = max(0.0, target - elapsed)
            self._stop.wait(remaining)

# ---------------------------
# Settings dialog
# ---------------------------
class SettingsDialog(tk.Toplevel):
    def __init__(self, master, engine: TelemetryEngine, on_apply):
        super().__init__(master)
        self.title("Settings")
        self.resizable(False, False)
        self.engine = engine
        self.on_apply = on_apply

        # Overlay mode toggle
        self.overlay_var = tk.BooleanVar(value=bool(master.attributes("-topmost")))
        overlay_frame = ttk.LabelFrame(self, text="Overlay mode")
        overlay_frame.pack(fill="x", padx=8, pady=8)
        ttk.Checkbutton(overlay_frame, text="Always on top", variable=self.overlay_var).pack(anchor="w", padx=8, pady=4)

        # Plugin toggles
        plugins_frame = ttk.LabelFrame(self, text="Enabled plugins")
        plugins_frame.pack(fill="x", padx=8, pady=8)
        self.plugin_vars = {}
        for key in ["cpu","memory","disk","network","process","gpu"]:
            var = tk.BooleanVar(value=self.engine.enabled_plugins.get(key, True))
            ttk.Checkbutton(plugins_frame, text=key, variable=var).pack(anchor="w", padx=8, pady=2)
            self.plugin_vars[key] = var

        # Thresholds
        thr = self.engine.thresholds
        thr_frame = ttk.LabelFrame(self, text="Alert thresholds")
        thr_frame.pack(fill="x", padx=8, pady=8)

        self.thr_vars = {
            "cpu_total_percent": tk.DoubleVar(value=thr["cpu_total_percent"]),
            "mem_percent": tk.DoubleVar(value=thr["mem_percent"]),
            "disk_write_bps": tk.DoubleVar(value=thr["disk_write_bps"]),
            "disk_read_bps": tk.DoubleVar(value=thr["disk_read_bps"]),
            "net_send_bps": tk.DoubleVar(value=thr["net_send_bps"]),
            "net_recv_bps": tk.DoubleVar(value=thr["net_recv_bps"]),
            "temp_c": tk.DoubleVar(value=thr["temp_c"]),
        }

        def add_thr_row(parent, label, var, unit_hint=""):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=4)
            ttk.Label(row, text=label).pack(side="left")
            entry = ttk.Entry(row, width=18, textvariable=var)
            entry.pack(side="left", padx=8)
            if unit_hint:
                ttk.Label(row, text=unit_hint).pack(side="left")

        add_thr_row(thr_frame, "CPU total %", self.thr_vars["cpu_total_percent"])
        add_thr_row(thr_frame, "Memory %", self.thr_vars["mem_percent"])
        add_thr_row(thr_frame, "Disk write B/s", self.thr_vars["disk_write_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Disk read B/s", self.thr_vars["disk_read_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Net send B/s", self.thr_vars["net_send_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Net recv B/s", self.thr_vars["net_recv_bps"], "(bytes/sec)")
        add_thr_row(thr_frame, "Temperature °C", self.thr_vars["temp_c"])

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=8)
        ttk.Button(btns, text="Apply", command=self.apply).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def apply(self):
        plugin_mapping = {k: v.get() for k, v in self.plugin_vars.items()}
        thr_mapping = {k: v.get() for k, v in self.thr_vars.items()}
        overlay = self.overlay_var.get()
        try:
            self.on_apply(plugin_mapping, thr_mapping, overlay)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}")
        self.destroy()

# ---------------------------
# GUI dashboard
# ---------------------------
class Dashboard(tk.Frame):
    def __init__(self, master, engine: TelemetryEngine):
        super().__init__(master)
        self.engine = engine
        self.master = master
        self.master.title("Autonomous Telemetry Dashboard")
        self.pack(fill="both", expand=True)

        # Top controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=8, pady=8)

        self.start_btn = ttk.Button(control_frame, text="Start", command=self.on_start)
        self.stop_btn  = ttk.Button(control_frame, text="Stop", command=self.on_stop)
        self.log_btn   = ttk.Button(control_frame, text="Enable logging", command=self.on_toggle_log)
        self.adapt_var = tk.BooleanVar(value=True)
        self.adapt_chk = ttk.Checkbutton(control_frame, text="Adaptive interval", variable=self.adapt_var, command=self.on_adaptive_toggle)
        self.settings_btn = ttk.Button(control_frame, text="Settings", command=self.on_settings)

        self.start_btn.pack(side="left", padx=4)
        self.stop_btn.pack(side="left", padx=4)
        self.log_btn.pack(side="left", padx=4)
        self.adapt_chk.pack(side="left", padx=12)
        self.settings_btn.pack(side="right", padx=4)

        # Interval slider
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", padx=8)
        ttk.Label(slider_frame, text="Sampling interval (sec)").pack(side="left")
        self.interval_var = tk.DoubleVar(value=self.engine.interval)
        self.interval_slider = ttk.Scale(slider_frame, from_=0.2, to=5.0, variable=self.interval_var, command=self.on_interval_change)
        self.interval_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.interval_read = ttk.Label(slider_frame, text=f"{self.engine.interval:.2f}s")
        self.interval_read.pack(side="right", padx=4)

        # Status line + alerts banner
        self.status_var = tk.StringVar(value="Idle")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.pack(fill="x", padx=8, pady=4)

        self.alert_canvas = tk.Canvas(self, height=36, bg="#202020", highlightthickness=0)
        self.alert_canvas.pack(fill="x", padx=8)
        self.alert_banner_text_id = None

        # Canvas charts
        charts_frame = ttk.Frame(self)
        charts_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.cpu_canvas = tk.Canvas(charts_frame, width=480, height=160, bg="#101820", highlightthickness=0)
        self.mem_canvas = tk.Canvas(charts_frame, width=480, height=160, bg="#101820", highlightthickness=0)
        self.cpu_canvas.pack(side="top", fill="x", expand=True)
        self.mem_canvas.pack(side="top", fill="x", expand=True, pady=(8,0))

        # Right panel with metrics
        right_frame = ttk.Frame(self)
        right_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.metrics_tree = ttk.Treeview(right_frame, columns=("value",), show="tree headings")
        self.metrics_tree.heading("#0", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("#0", width=260)
        self.metrics_tree.column("value", width=280)
        self.metrics_tree.pack(fill="both", expand=True)

        # Log file path display
        self.log_path_var = tk.StringVar(value="Logging: disabled")
        ttk.Label(self, textvariable=self.log_path_var).pack(fill="x", padx=8, pady=(0,8))

        # periodic UI update
        self.after(200, self.poll_engine)

    # Control handlers
    def on_start(self):
        self.engine.start()
        self.status_var.set("Collecting telemetry...")
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])

    def on_stop(self):
        self.engine.stop()
        self.status_var.set("Stopped")
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])

    def on_toggle_log(self):
        enabled = not self.engine.logging_enabled
        self.engine.enable_logging(enabled)
        if enabled and hasattr(self.engine, "log_file_path"):
            self.log_path_var.set(f"Logging: {self.engine.log_file_path}")
            self.log_btn.configure(text="Disable logging")
        else:
            self.log_path_var.set("Logging: disabled")
            self.log_btn.configure(text="Enable logging")

    def on_adaptive_toggle(self):
        self.engine.set_adaptive(self.adapt_var.get())

    def on_interval_change(self, _evt=None):
        val = float(self.interval_var.get())
        self.interval_read.configure(text=f"{val:.2f}s")
        self.engine.set_interval(val)

    def on_settings(self):
        def apply_settings(plugin_mapping, thr_mapping, overlay_mode):
            self.engine.update_enabled_plugins(plugin_mapping)
            self.engine.update_thresholds(thr_mapping)
            try:
                self.master.attributes("-topmost", overlay_mode)
            except Exception:
                pass
        SettingsDialog(self.master, self.engine, on_apply=apply_settings)

    # UI update logic
    def poll_engine(self):
        try:
            snapshot = self.engine.out_queue.get_nowait()
            self.render(snapshot)
        except queue.Empty:
            pass
        self.after(200, self.poll_engine)

    def render(self, s):
        self.draw_alerts(s.get("_alerts", []))
        self.draw_cpu_chart(s.get("cpu", {}))
        self.draw_mem_chart(s.get("memory", {}))
        self.update_metrics_tree(s)

    def draw_alerts(self, alerts):
        self.alert_canvas.delete("all")
        if alerts:
            self.alert_canvas.configure(bg="#3d0000")
            text = " | ".join(alerts[:4]) + (" ..." if len(alerts) > 4 else "")
            self.alert_canvas.create_text(12, 18, text=f"ALERT: {text}", anchor="w", fill="#ffcccb", font=("Arial", 12, "bold"))
            self.status_var.set("Alerts active")
        else:
            self.alert_canvas.configure(bg="#202020")
            self.alert_canvas.create_text(12, 18, text="No alerts", anchor="w", fill="#9aa0a6", font=("Arial", 11))
            if "Collecting" in self.status_var.get() or "Stopped" in self.status_var.get():
                pass
            else:
                self.status_var.set("Idle")

    def draw_cpu_chart(self, cpu):
        canvas = self.cpu_canvas
        canvas.delete("all")
        w = int(canvas["width"]); h = int(canvas["height"])
        canvas.create_text(8, 12, text="CPU per-core % and total", anchor="w", fill="#9aa0a6")
        total = cpu.get("cpu_percent_total", 0.0) or 0.0
        per_core = cpu.get("cpu_percent_per_core", []) or []
        n = max(1, len(per_core))
        bar_w = max(12, (w - 48) // n)
        for i, v in enumerate(per_core):
            x0 = 16 + i * bar_w
            y0 = h - 24
            bar_h = int((h - 56) * (float(v) / 100.0))
            canvas.create_rectangle(x0, y0 - bar_h, x0 + bar_w - 6, y0, fill="#00c853", outline="")
            canvas.create_text(x0 + bar_w//2 - 4, y0 + 12, text=str(i), fill="#9aa0a6")
        # total usage gauge
        gauge_w = 320
        canvas.create_rectangle(16, 28, 16 + gauge_w, 44, outline="#37474f")
        fill_w = int(gauge_w * (total / 100.0))
        fill_color = "#ffab00" if total < 75 else "#ff6d00"
        canvas.create_rectangle(16, 28, 16 + fill_w, 44, fill=fill_color, outline="")
        canvas.create_text(16 + gauge_w + 8, 36, text=f"{total:.1f}%", anchor="w", fill="#9aa0a6")

    def draw_mem_chart(self, mem):
        canvas = self.mem_canvas
        canvas.delete("all")
        w = int(canvas["width"]); h = int(canvas["height"])
        canvas.create_text(8, 12, text="Memory and swap", anchor="w", fill="#9aa0a6")
        total = mem.get("mem_total", 0) or 1
        used = mem.get("mem_used", 0) or 0
        percent = mem.get("mem_percent", 0.0) or 0.0
        swap_total = mem.get("swap_total", 0) or 1
        swap_used = mem.get("swap_used", 0) or 0
        # memory usage bar
        bar_w = w - 32
        canvas.create_rectangle(16, 28, 16 + bar_w, 44, outline="#37474f")
        canvas.create_rectangle(16, 28, 16 + int(bar_w * (used / total)), 44, fill="#2962ff", outline="")
        canvas.create_text(16 + bar_w + 8, 36, text=f"{percent:.1f}%", anchor="w", fill="#9aa0a6")
        # swap bar
        canvas.create_rectangle(16, 64, 16 + bar_w, 80, outline="#37474f")
        canvas.create_rectangle(16, 64, 16 + int(bar_w * (swap_used / swap_total)), 80, fill="#d81b60", outline="")
        # labels
        canvas.create_text(16, 100, anchor="w", fill="#9aa0a6",
                           text=f"Mem: {fmt_bytes(total)} total, {fmt_bytes(used)} used")
        canvas.create_text(16, 120, anchor="w", fill="#9aa0a6",
                           text=f"Swap: {fmt_bytes(swap_total)} total, {fmt_bytes(swap_used)} used")

    def update_metrics_tree(self, s):
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        # CPU quick stats
        cpu = s.get("cpu", {})
        self._add_metric("CPU total %", f"{float(cpu.get('cpu_percent_total', 0.0) or 0.0):.1f}")
        if "cpu_freq_current_mhz" in cpu:
            self._add_metric("CPU freq (MHz)", f"{cpu.get('cpu_freq_current_mhz', 0):.0f}")
        if "load_avg_1m" in cpu:
            self._add_metric("Load avg 1m", f"{cpu.get('load_avg_1m', 0):.2f}")
        # Temps
        temps = cpu.get("temps", {}) or {}
        for k, v in temps.items():
            if v is not None:
                self._add_metric(f"{k}", f"{v:.1f} °C")
        # Battery
        if "battery_percent" in cpu:
            bat = f"{cpu.get('battery_percent', 0):.0f}%"
            plug = "plugged" if cpu.get("battery_plugged") else "on battery"
            self._add_metric("Battery", f"{bat} ({plug})")

        # Memory
        mem = s.get("memory", {})
        self._add_metric("Mem used", fmt_bytes(mem.get("mem_used", 0)))
        self._add_metric("Mem avail", fmt_bytes(mem.get("mem_available", 0)))

        # Disk & Net
        disk = s.get("disk", {})
        net = s.get("network", {})
        self._add_metric("Disk read", fmt_bytes(disk.get("disk_read_bytes", 0)))
        self._add_metric("Disk write", fmt_bytes(disk.get("disk_write_bytes", 0)))
        self._add_metric("Net sent", fmt_bytes(net.get("net_bytes_sent", 0)))
        self._add_metric("Net recv", fmt_bytes(net.get("net_bytes_recv", 0)))

        # Rates
        deltas = s.get("_deltas", {}) or {}
        if deltas:
            self._add_metric("Disk read rate", f"{fmt_bytes(deltas.get('disk_read_bps', 0))}/s")
            self._add_metric("Disk write rate", f"{fmt_bytes(deltas.get('disk_write_bps', 0))}/s")
            self._add_metric("Net send rate", f"{fmt_bytes(deltas.get('net_send_bps', 0))}/s")
            self._add_metric("Net recv rate", f"{fmt_bytes(deltas.get('net_recv_bps', 0))}/s")

        # GPU
        gpu = s.get("gpu", {}).get("gpus", [])
        if gpu:
            g0 = gpu[0]
            self._add_metric(f"GPU0 {g0.get('name','')}", f"{g0.get('gpu_util',0)}% util, {g0.get('temp_c','n/a')} °C")
            self._add_metric("GPU0 mem", f"{fmt_bytes(g0.get('mem_used',0))}/{fmt_bytes(g0.get('mem_total',0))}")

        # Top processes
        procs = s.get("process", {}).get("top_processes", [])
        for p in procs:
            name = p.get("name","proc")
            cpuv = p.get("cpu", 0.0)
            rss = p.get("rss", 0)
            self._add_metric(f"PID {p['pid']} {name}", f"{cpuv:.1f}% | {fmt_bytes(rss)}")

    def _add_metric(self, key, value):
        self.metrics_tree.insert("", "end", text=key, values=(value,))

# ---------------------------
# App bootstrap
# ---------------------------
def main():
    root = tk.Tk()
    # Style
    try:
        style = ttk.Style(root)
        if platform.system() == "Windows":
            style.theme_use("vista")
        else:
            style.theme_use("clam")
    except Exception:
        pass

    engine = TelemetryEngine(interval=1.0, adaptive=True)
    app = Dashboard(root, engine)

    # Initial button states
    app.stop_btn.state(["disabled"])
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, engine))
    root.geometry("1024x720")
    root.mainloop()

def on_close(root, engine):
    try:
        engine.stop()
    except Exception:
        pass
    root.destroy()

if __name__ == "__main__":
    main()

