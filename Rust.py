#!/usr/bin/env python3
"""
Cross-platform Laser-like Telemetry Monitor
- Live CPU, Memory, GPU (NVIDIA via GPUtil/NVML), Disk, Network, Processes
- LeafShield: protected processes/services
- AnomalyDetector: non-destructive scoring (CPU, network, disk IO, path, parent anomalies, memory)
- Custom rules: allow/deny patterns per path or name
- Ports focus: flags burst SYNs/unusual bindings (best-effort via psutil)
- Pulse modes: gentle vs aggressive sampling cadences
- GUI overlays: color-coded rows for scan results
- Auto-loader for Python libs
"""

import sys
import subprocess
import importlib

# ----------------------------
# AUTO-LOADER FOR PY LIBRARIES
# ----------------------------
REQUIRED_PY_LIBS = [
    "psutil",
    "matplotlib",
    "GPUtil",      # NVIDIA GPU generic
    "pynvml",      # NVIDIA NVML deeper metrics
]

def ensure_libs(lib_names):
    for name in lib_names:
        try:
            importlib.import_module(name)
        except ImportError:
            print(f"[AUTOLOADER] Installing missing library: {name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])

# Tkinter first (may need OS package on Linux)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("[ERROR] Tkinter not available.")
    print("Linux: install via `sudo apt install python3-tk` (Debian/Ubuntu) or your distro's equivalent.")
    print("Windows/macOS: install Python from python.org with Tcl/Tk included.")
    raise

# Install Python libraries
ensure_libs(REQUIRED_PY_LIBS)

# Imports after auto-loader
import psutil
import time
import os
import platform
from collections import deque
from threading import Thread, Event
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# GPU libs may fail if no GPU; handle gracefully
try:
    import GPUtil
except Exception:
    GPUtil = None

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None


# ----------------------------
# UTILITIES
# ----------------------------
def format_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    step = 1024.0
    i = 0
    v = float(n)
    while v >= step and i < len(units)-1:
        v /= step
        i += 1
    return f"{v:.2f} {units[i]}"

def safe_get_temps():
    try:
        temps = psutil.sensors_temperatures()
        return temps or {}
    except Exception:
        return {}

def safe_gpu_info():
    info = {"available": False, "items": []}
    # GPUtil first
    try:
        gpus = GPUtil.getGPUs() if GPUtil else []
        if gpus:
            info["available"] = True
            for g in gpus:
                info["items"].append({
                    "id": g.id,
                    "name": g.name,
                    "load": g.load * 100.0,
                    "mem_used": g.memoryUsed,
                    "mem_total": g.memoryTotal,
                    "temp": g.temperature,
                    "uuid": getattr(g, "uuid", ""),
                })
            return info
    except Exception:
        pass
    # NVML fallback
    if pynvml:
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                info["available"] = True
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    name_raw = pynvml.nvmlDeviceGetName(h)
                    name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    info["items"].append({
                        "id": i,
                        "name": name,
                        "load": float(util.gpu),
                        "mem_used": int(mem.used) / (1024**2),
                        "mem_total": int(mem.total) / (1024**2),
                        "temp": float(temp),
                        "uuid": "",
                    })
        except Exception:
            pass
    return info


# ----------------------------
# LASER METAPHOR: LEAF SHIELD + ANOMALY DETECTOR + RULES
# ----------------------------
class LeafShield:
    """
    Protects critical processes/services ('leaf') the scanner must not touch.
    Non-destructive: only filters/overlooks during anomaly surfacing.
    """
    def __init__(self, extra_names=None, extra_pids=None):
        self.os_name = platform.system()
        self.protected_names = set(self._default_names())
        if extra_names:
            self.protected_names.update({n.lower() for n in extra_names})
        self.protected_pids = set(extra_pids or [])

    def _default_names(self):
        base = [
            # Minimal cross-platform core
            "system", "system idle process", "init", "launchd", "services", "lsass",
            "csrss", "wininit", "winlogon", "smss", "explorer", "dwm",
            "dock", "windowserver", "kernel_task",
        ]
        return [n.lower() for n in base]

    def is_protected(self, proc):
        try:
            name = (proc.info.get("name") or proc.name() or "").lower()
        except Exception:
            name = ""
        return proc.pid in self.protected_pids or name in self.protected_names


class CustomRules:
    """
    Allow/Deny patterns that adjust anomaly scoring.
    Patterns apply to process name and exe path (case-insensitive).
    """
    def __init__(self):
        self.allow_names = set()
        self.deny_names = set()
        self.allow_paths = set()
        self.deny_paths = set()

    def load_from_strings(self, allow_names_str, deny_names_str, allow_paths_str, deny_paths_str):
        self.allow_names = {s.strip().lower() for s in allow_names_str.split(",") if s.strip()}
        self.deny_names = {s.strip().lower() for s in deny_names_str.split(",") if s.strip()}
        self.allow_paths = {s.strip().lower() for s in allow_paths_str.split(",") if s.strip()}
        self.deny_paths = {s.strip().lower() for s in deny_paths_str.split(",") if s.strip()}

    def adjust_score(self, name, exe, score, reasons):
        n = (name or "").lower()
        e = (exe or "").lower()
        if n in self.allow_names or any(e.startswith(p) for p in self.allow_paths):
            reasons.append("Rule allow -10")
            score = max(0, score - 10)
        if n in self.deny_names or any(e.startswith(p) for p in self.deny_paths):
            reasons.append("Rule deny +15")
            score += 15
        return score


class PortsFocus:
    """
    Best-effort port anomaly: scans inet connections and flags
    - Burst SYNs (approx via connections with SYN_SENT/SYN_RECV)
    - Unusual local bindings (high ephemeral usage with LISTEN)
    """
    def __init__(self):
        self.last_conn_sample = []
        self.syn_burst_threshold = 10
        self.listen_threshold = 20  # high number of LISTEN sockets

    def analyze(self):
        syn_sent = 0
        syn_recv = 0
        listen_count = 0
        by_proc = {}
        try:
            conns = psutil.net_connections(kind="inet")
            for c in conns:
                if c.status == "SYN_SENT":
                    syn_sent += 1
                elif c.status == "SYN_RECV":
                    syn_recv += 1
                elif c.status == "LISTEN":
                    listen_count += 1
                pid = c.pid or -1
                by_proc.setdefault(pid, {"syn": 0, "listen": 0, "est": 0})
                if c.status in ("SYN_SENT", "SYN_RECV"):
                    by_proc[pid]["syn"] += 1
                elif c.status == "LISTEN":
                    by_proc[pid]["listen"] += 1
                elif c.status == "ESTABLISHED":
                    by_proc[pid]["est"] += 1
        except Exception:
            pass
        summary = {
            "syn_sent": syn_sent,
            "syn_recv": syn_recv,
            "listen_count": listen_count,
            "by_proc": by_proc,
            "syn_burst": (syn_sent + syn_recv) >= self.syn_burst_threshold,
            "listen_overload": listen_count >= self.listen_threshold,
        }
        return summary


class AnomalyDetector:
    """
    Scores processes for suspicious patterns ('rust'):
    - CPU spikes + sustained high load
    - Excessive network connections or rapid outbound bursts (PortsFocus integrated)
    - Unusual disk writes
    - Suspicious parent/child relationships
    - Executable path anomalies (temp folders, user profile binaries, hidden)
    - Memory footprint heuristic
    Non-destructive: produces a score and rationale; does not alter processes.
    """
    def __init__(self, shield: LeafShield, rules: CustomRules, ports_focus: PortsFocus):
        self.shield = shield
        self.rules = rules
        self.ports_focus = ports_focus

    def _path_suspicion(self, exe):
        if not exe:
            return 0
        exe_low = exe.lower()
        score = 0
        suspicious_roots = [
            os.path.expanduser("~").lower(),
            os.path.join(os.path.expanduser("~"), "appdata").lower(),  # windows
            "/tmp", "/var/tmp", "/private/tmp",
        ]
        if any(exe_low.startswith(r) for r in suspicious_roots):
            score += 15
        if os.path.basename(exe_low).startswith("."):
            score += 10
        if exe_low.endswith((".tmp", ".dat", ".log")):
            score += 10
        return score

    def _parent_anomaly(self, proc):
        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if not parent:
                return 0
            pname = (parent.name() or "").lower()
            if pname in ("powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe", "python.exe", "node.exe"):
                return 10
            if pname in ("bash", "sh", "zsh"):
                return 5
        except Exception:
            pass
        return 0

    def _connection_pressure(self, pid, ports_summary):
        score = 0
        # Per-proc counts from PortsFocus
        stats = ports_summary["by_proc"].get(pid, {})
        syn = stats.get("syn", 0)
        listen = stats.get("listen", 0)
        est = stats.get("est", 0)
        if syn >= 10:
            score += 20
        elif syn >= 5:
            score += 10
        if listen >= 20:
            score += 10
        if est >= 50:
            score += 10
        return score

    def _io_pressure(self, pid):
        score = 0
        try:
            p = psutil.Process(pid)
            io = p.io_counters() if hasattr(p, "io_counters") else None
            if io and io.write_bytes > 10 * 1024 * 1024:  # >10MB since start
                score += 10
            if io and io.read_bytes > 50 * 1024 * 1024:
                score += 5
        except Exception:
            pass
        return score

    def _cpu_pressure(self, proc):
        try:
            cpu = proc.cpu_percent(interval=None)
            if cpu >= 80:
                return 20
            if cpu >= 50:
                return 10
        except Exception:
            pass
        return 0

    def score_process(self, proc, ports_summary):
        # Protected leaf
        if self.shield.is_protected(proc):
            return {"score": 0, "level": "protected", "reasons": ["LeafShield: protected"], "name": proc.info.get("name", ""), "pid": proc.pid}

        reasons = []
        score = 0

        # CPU pressure
        s = self._cpu_pressure(proc)
        if s: reasons.append(f"CPU pressure +{s}")
        score += s

        # Path suspicion
        exe = ""
        try:
            exe = proc.info.get("exe") or proc.exe()
        except Exception:
            exe = ""
        s = self._path_suspicion(exe)
        if s: reasons.append(f"Path suspicion +{s}")
        score += s

        # Parent anomaly
        s = self._parent_anomaly(proc)
        if s: reasons.append(f"Parent anomaly +{s}")
        score += s

        # Connection pressure
        s = self._connection_pressure(proc.pid, ports_summary)
        if s: reasons.append(f"Connection pressure +{s}")
        score += s

        # IO pressure
        s = self._io_pressure(proc.pid)
        if s: reasons.append(f"IO pressure +{s}")
        score += s

        # Memory footprint heuristic
        try:
            rss = proc.info.get("memory_info").rss if proc.info.get("memory_info") else proc.memory_info().rss
            if rss > 800 * 1024 * 1024:  # >800MB
                reasons.append("Large RSS +10")
                score += 10
        except Exception:
            pass

        # Apply custom rules adjustments
        name = proc.info.get("name", "")
        score = self.rules.adjust_score(name, exe, score, reasons)

        # Normalize and classify
        level = "normal"
        if score >= 45:
            level = "suspect"
        elif score >= 20:
            level = "watch"

        return {"score": score, "level": level, "reasons": reasons, "name": name, "pid": proc.pid, "exe": exe}


# ----------------------------
# MAIN GUI CLASS
# ----------------------------
class TelemetryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser-like Telemetry Monitor (LeafShield + Rules + PortsFocus)")
        self.root.geometry("1380x900")

        # Pulse mode & cadence
        self.pulse_mode = tk.StringVar(value="gentle")
        self.update_ms = 1000  # gentle default

        # Histories
        self.cpu_history = deque(maxlen=240)
        self.net_history_sent = deque(maxlen=240)
        self.net_history_recv = deque(maxlen=240)
        self.last_net_io = psutil.net_io_counters()

        # Shield, rules, ports focus, detector
        self.shield = LeafShield(extra_names=["python", "python.exe", "system", "kernel_task", "windowserver"])
        self.rules = CustomRules()
        self.ports_focus = PortsFocus()
        self.detector = AnomalyDetector(self.shield, self.rules, self.ports_focus)

        # Notebook (tabs)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)

        # Create tabs
        self.tab_overview = ttk.Frame(self.tabs)
        self.tab_cpu = ttk.Frame(self.tabs)
        self.tab_memory = ttk.Frame(self.tabs)
        self.tab_gpu = ttk.Frame(self.tabs)
        self.tab_disks = ttk.Frame(self.tabs)
        self.tab_network = ttk.Frame(self.tabs)
        self.tab_processes = ttk.Frame(self.tabs)
        self.tab_scan = ttk.Frame(self.tabs)

        for tab, name in [
            (self.tab_overview, "Overview"),
            (self.tab_cpu, "CPU"),
            (self.tab_memory, "Memory"),
            (self.tab_gpu, "GPU"),
            (self.tab_disks, "Disks"),
            (self.tab_network, "Network"),
            (self.tab_processes, "Processes"),
            (self.tab_scan, "Laser Scan"),
        ]:
            self.tabs.add(tab, text=name)

        # Build tabs
        self._build_overview()
        self._build_cpu_tab()
        self._build_memory_tab()
        self._build_gpu_tab()
        self._build_disks_tab()
        self._build_network_tab()
        self._build_processes_tab()
        self._build_scan_tab()

        # Start updates
        self._schedule_updates()

    # ------------------------
    # TAB BUILDERS
    # ------------------------
    def _build_overview(self):
        frame = self.tab_overview

        self.lbl_os = ttk.Label(frame, text=f"OS: {platform.system()} {platform.release()}")
        self.lbl_os.pack(anchor="w", padx=12, pady=6)

        self.lbl_cpu = ttk.Label(frame, text="CPU: —")
        self.lbl_mem = ttk.Label(frame, text="Memory: —")
        self.lbl_swap = ttk.Label(frame, text="Swap: —")
        self.lbl_gpu = ttk.Label(frame, text="GPU: —")
        self.lbl_disk = ttk.Label(frame, text="Disk: —")
        self.lbl_net = ttk.Label(frame, text="Network: —")
        self.lbl_temp = ttk.Label(frame, text="Temps: —")

        for lbl in (self.lbl_cpu, self.lbl_mem, self.lbl_swap, self.lbl_gpu, self.lbl_disk, self.lbl_net, self.lbl_temp):
            lbl.pack(anchor="w", padx=12, pady=8)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.set_title("CPU Usage History (%)")
        ax.set_ylim(0, 100)
        self.overview_cpu_ax = ax
        self.overview_cpu_line, = ax.plot([], [], "r-")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="x", padx=12, pady=8)
        self.overview_cpu_canvas = canvas

    def _build_cpu_tab(self):
        frame = self.tab_cpu

        self.core_frame = ttk.Frame(frame)
        self.core_frame.pack(fill="x", padx=12, pady=12)

        self.core_bars = []
        self.core_labels = []
        for i in range(psutil.cpu_count(logical=True)):
            bar = ttk.Progressbar(self.core_frame, orient="horizontal", mode="determinate", length=250, maximum=100)
            bar.grid(row=i, column=1, padx=8, pady=4, sticky="w")
            lbl = ttk.Label(self.core_frame, text=f"Core {i}: —")
            lbl.grid(row=i, column=0, padx=8, pady=4, sticky="w")
            self.core_bars.append(bar)
            self.core_labels.append(lbl)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title("CPU Usage (%)")
        ax.set_ylim(0, 100)
        self.cpu_ax = ax
        self.cpu_line, = ax.plot([], [], "b-")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="x", padx=12, pady=8)
        self.cpu_canvas = canvas

    def _build_memory_tab(self):
        frame = self.tab_memory
        self.mem_overview = ttk.Label(frame, text="—")
        self.mem_overview.pack(anchor="w", padx=12, pady=8)

        self.mem_text = tk.Text(frame, height=22, width=140)
        self.mem_text.pack(fill="both", padx=12, pady=8)

    def _build_gpu_tab(self):
        frame = self.tab_gpu
        self.gpu_text = tk.Text(frame, height=25, width=140)
        self.gpu_text.pack(fill="both", padx=12, pady=12)

    def _build_disks_tab(self):
        frame = self.tab_disks
        self.disk_text = tk.Text(frame, height=25, width=140)
        self.disk_text.pack(fill="both", padx=12, pady=12)

    def _build_network_tab(self):
        frame = self.tab_network
        ttk.Label(frame, text="Active Network Connections").pack(anchor="w", padx=12, pady=6)
        self.net_text = tk.Text(frame, height=18, width=140)
        self.net_text.pack(fill="both", padx=12, pady=6)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title("Network Throughput (KB/s)")
        ax.set_ylim(0, 1024)
        ax.set_ylabel("KB/s")
        ax.set_xlabel("Samples")
        self.net_ax = ax
        self.net_line_sent, = ax.plot([], [], "g-", label="Sent")
        self.net_line_recv, = ax.plot([], [], "m-", label="Recv")
        ax.legend(loc="upper right")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="x", padx=12, pady=6)
        self.net_canvas = canvas

    def _build_processes_tab(self):
        frame = self.tab_processes
        top = ttk.Frame(frame)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Top Processes by CPU").grid(row=0, column=0, sticky="w", padx=12, pady=6)
        ttk.Label(top, text="Top Processes by Memory").grid(row=0, column=1, sticky="w", padx=12, pady=6)

        self.proc_cpu_text = tk.Text(top, height=30, width=70)
        self.proc_mem_text = tk.Text(top, height=30, width=70)
        self.proc_cpu_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        self.proc_mem_text.grid(row=1, column=1, sticky="nsew", padx=12, pady=6)

        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

    def _build_scan_tab(self):
        frame = self.tab_scan

        controls = ttk.Frame(frame)
        controls.pack(fill="x", padx=12, pady=8)

        ttk.Label(controls, text="Pulse mode:").grid(row=0, column=0, sticky="w")
        self.pulse_combo = ttk.Combobox(controls, values=["gentle", "aggressive"], textvariable=self.pulse_mode, width=12)
        self.pulse_combo.grid(row=0, column=1, sticky="w", padx=8)
        self.pulse_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_pulse_mode())

        ttk.Label(controls, text="LeafShield names (comma-separated):").grid(row=1, column=0, sticky="w")
        self.leaf_entry = ttk.Entry(controls, width=60)
        self.leaf_entry.grid(row=1, column=1, sticky="w", padx=8)
        self.leaf_entry.insert(0, ", ".join(sorted(self.shield.protected_names)))
        ttk.Button(controls, text="Update LeafShield", command=self._update_leafshield).grid(row=1, column=2, padx=8)

        ttk.Label(controls, text="Allow names:").grid(row=2, column=0, sticky="w")
        self.allow_names_entry = ttk.Entry(controls, width=40)
        self.allow_names_entry.grid(row=2, column=1, sticky="w", padx=8)

        ttk.Label(controls, text="Deny names:").grid(row=2, column=2, sticky="w")
        self.deny_names_entry = ttk.Entry(controls, width=40)
        self.deny_names_entry.grid(row=2, column=3, sticky="w", padx=8)

        ttk.Label(controls, text="Allow paths:").grid(row=3, column=0, sticky="w")
        self.allow_paths_entry = ttk.Entry(controls, width=40)
        self.allow_paths_entry.grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(controls, text="Deny paths:").grid(row=3, column=2, sticky="w")
        self.deny_paths_entry = ttk.Entry(controls, width=40)
        self.deny_paths_entry.grid(row=3, column=3, sticky="w", padx=8)

        ttk.Button(controls, text="Apply Rules", command=self._apply_rules).grid(row=4, column=0, padx=8, pady=4)

        ttk.Label(frame, text="Laser Scan: Protected vs. Suspect vs. Watch vs. Normal (non-destructive)").pack(anchor="w", padx=12, pady=8)

        columns = ("pid", "name", "level", "score", "reasons")
        self.scan_tree = ttk.Treeview(frame, columns=columns, show="headings", height=26)
        for col, width in zip(columns, (80, 220, 140, 80, 700)):
            self.scan_tree.heading(col, text=col.capitalize())
            self.scan_tree.column(col, width=width, anchor="w")
        self.scan_tree.pack(fill="both", expand=True, padx=12, pady=8)

        # Color tags via style map are limited; use tags per item
        self.scan_tree.tag_configure("suspect", foreground="#b00000")
        self.scan_tree.tag_configure("watch", foreground="#9a6b00")
        self.scan_tree.tag_configure("normal", foreground="#006400")
        self.scan_tree.tag_configure("protected", foreground="#004d99")

        self.scan_status = ttk.Label(frame, text="—")
        self.scan_status.pack(anchor="w", padx=12, pady=6)

        # Ports focus summary box
        self.ports_summary_text = tk.Text(frame, height=8, width=140)
        self.ports_summary_text.pack(fill="x", padx=12, pady=6)

    def _apply_pulse_mode(self):
        mode = self.pulse_mode.get()
        if mode == "gentle":
            self.update_ms = 1000
        else:
            self.update_ms = 500
        self.scan_status.config(text=f"Pulse mode set to {mode} ({self.update_ms} ms)")
        # restart scheduling loop
        self._restart_schedule()

    def _update_leafshield(self):
        try:
            names = [n.strip().lower() for n in self.leaf_entry.get().split(",") if n.strip()]
            self.shield.protected_names = set(names)
            self.scan_status.config(text=f"LeafShield updated: {len(names)} protected names.")
        except Exception as e:
            self.scan_status.config(text=f"LeafShield update failed: {e}")

    def _apply_rules(self):
        try:
            self.rules.load_from_strings(
                self.allow_names_entry.get(),
                self.deny_names_entry.get(),
                self.allow_paths_entry.get(),
                self.deny_paths_entry.get(),
            )
            self.scan_status.config(text="Rules applied.")
        except Exception as e:
            self.scan_status.config(text=f"Rules application failed: {e}")

    # ------------------------
    # UPDATE LOOP
    # ------------------------
    def _restart_schedule(self):
        # Cancel and reschedule a single after loop (simpler than threads)
        try:
            self.root.after_cancel(self._after_id)
        except Exception:
            pass
        self._schedule_updates()

    def _schedule_updates(self):
        self._update_all()
        self._after_id = self.root.after(self.update_ms, self._schedule_updates)

    def _update_all(self):
        self._update_overview()
        self._update_cpu()
        self._update_memory()
        self._update_gpu()
        self._update_disks()
        self._update_network()
        self._update_processes()
        self._update_scan()

    def _update_overview(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        self.cpu_history.append(cpu_percent)
        self.overview_cpu_line.set_xdata(range(len(self.cpu_history)))
        self.overview_cpu_line.set_ydata(self.cpu_history)
        self.overview_cpu_ax.set_xlim(0, len(self.cpu_history))
        self.overview_cpu_canvas.draw()

        temps = safe_get_temps()
        temp_strs = []
        for k, vs in temps.items():
            if not vs: continue
            readings = ", ".join([f"{v.label or k}:{v.current}°C" for v in vs[:3]])
            temp_strs.append(readings)
        temp_summary = "; ".join(temp_strs) if temp_strs else "No sensor data"

        disk_parts = psutil.disk_partitions(all=False)
        disk_usages = []
        for p in disk_parts:
            try:
                u = psutil.disk_usage(p.mountpoint)
                disk_usages.append(f"{p.device} {format_bytes(u.used)}/{format_bytes(u.total)} ({u.percent}%)")
            except Exception:
                pass
        disk_summary = " | ".join(disk_usages) if disk_usages else "No disks"

        net = psutil.net_io_counters()
        net_summary = f"Sent {format_bytes(net.bytes_sent)}, Recv {format_bytes(net.bytes_recv)}"

        gpu_info = safe_gpu_info()
        if gpu_info["available"] and gpu_info["items"]:
            g0 = gpu_info["items"][0]
            gpu_summary = f"{g0['name']} Load {g0['load']:.1f}% Mem {g0['mem_used']:.0f}/{g0['mem_total']:.0f} MB Temp {g0['temp']:.0f}°C"
        else:
            gpu_summary = "No GPU detected"

        self.lbl_cpu.config(text=f"CPU: {cpu_percent:.1f}%")
        self.lbl_mem.config(text=f"Memory: {mem.percent:.1f}% ({format_bytes(mem.used)}/{format_bytes(mem.total)})")
        self.lbl_swap.config(text=f"Swap: {swap.percent:.1f}% ({format_bytes(swap.used)}/{format_bytes(swap.total)})")
        self.lbl_gpu.config(text=f"GPU: {gpu_summary}")
        self.lbl_disk.config(text=f"Disk: {disk_summary}")
        self.lbl_net.config(text=f"Network: {net_summary}")
        self.lbl_temp.config(text=f"Temps: {temp_summary}")

    def _update_cpu(self):
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        for i, val in enumerate(per_core):
            if i < len(self.core_bars):
                self.core_bars[i]["value"] = val
                self.core_labels[i].config(text=f"Core {i}: {val:.1f}%")

        self.cpu_line.set_xdata(range(len(self.cpu_history)))
        self.cpu_line.set_ydata(self.cpu_history)
        self.cpu_ax.set_xlim(0, len(self.cpu_history))
        self.cpu_canvas.draw()

    def _update_memory(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        text = (
            f"Memory: {mem.percent:.1f}% ({format_bytes(mem.used)}/{format_bytes(mem.total)})\n"
            f"Available: {format_bytes(mem.available)} | Cached: {format_bytes(getattr(mem, 'cached', 0))}\n"
            f"Swap: {swap.percent:.1f}% ({format_bytes(swap.used)}/{format_bytes(swap.total)})\n\n"
            "Top 20 processes by RSS:\n"
        )
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "memory_info"]):
            try:
                rss = p.info["memory_info"].rss
                procs.append((rss, p.info["pid"], p.info["name"]))
            except Exception:
                continue
        procs.sort(reverse=True)
        lines = []
        for rss, pid, name in procs[:20]:
            lines.append(f"PID {pid:<6} RSS {format_bytes(rss):>10}  {name}")
        self.mem_overview.config(text=text)
        self.mem_text.delete("1.0", tk.END)
        self.mem_text.insert(tk.END, text + "\n".join(lines))

    def _update_gpu(self):
        info = safe_gpu_info()
        self.gpu_text.delete("1.0", tk.END)
        if info["available"] and info["items"]:
            for g in info["items"]:
                self.gpu_text.insert(tk.END,
                    f"GPU {g['id']}: {g['name']}\n"
                    f"  Load: {g['load']:.1f}%\n"
                    f"  Mem: {g['mem_used']:.0f}/{g['mem_total']:.0f} MB\n"
                    f"  Temp: {g['temp']:.0f}°C\n"
                    f"  UUID: {g['uuid']}\n\n"
                )
        else:
            self.gpu_text.insert(tk.END, "No GPU detected or unsupported driver.\n")

    def _update_disks(self):
        self.disk_text.delete("1.0", tk.END)
        self.disk_text.insert(tk.END, "Disk partitions and usage:\n\n")
        for p in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(p.mountpoint)
                self.disk_text.insert(tk.END,
                    f"{p.device} ({p.mountpoint})  {format_bytes(u.used)}/{format_bytes(u.total)}  {u.percent}%\n"
                )
            except Exception:
                continue

        self.disk_text.insert(tk.END, "\nDisk IO counters:\n\n")
        try:
            io = psutil.disk_io_counters(perdisk=True)
            for dev, stats in io.items():
                self.disk_text.insert(tk.END,
                    f"{dev}: reads {stats.read_count} ({format_bytes(stats.read_bytes)}), "
                    f"writes {stats.write_count} ({format_bytes(stats.write_bytes)}), "
                    f"read_time {stats.read_time} ms, write_time {stats.write_time} ms\n"
                )
        except Exception:
            self.disk_text.insert(tk.END, "No disk IO counters available.\n")

    def _update_network(self):
        self.net_text.delete("1.0", tk.END)
        try:
            conns = psutil.net_connections(kind="inet")
            for c in conns[:250]:
                laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "—"
                raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "—"
                self.net_text.insert(tk.END, f"{c.status:<12} {laddr:<22} -> {raddr:<22}  PID {str(c.pid or '—')}\n")
        except Exception:
            self.net_text.insert(tk.END, "Unable to read connections (permissions or OS limits).\n")

        now_io = psutil.net_io_counters()
        dt_bytes_sent = max(0, now_io.bytes_sent - self.last_net_io.bytes_sent)
        dt_bytes_recv = max(0, now_io.bytes_recv - self.last_net_io.bytes_recv)
        self.last_net_io = now_io

        kbps_sent = dt_bytes_sent / 1024.0
        kbps_recv = dt_bytes_recv / 1024.0
        self.net_history_sent.append(kbps_sent)
        self.net_history_recv.append(kbps_recv)

        self.net_line_sent.set_xdata(range(len(self.net_history_sent)))
        self.net_line_sent.set_ydata(self.net_history_sent)
        self.net_line_recv.set_xdata(range(len(self.net_history_recv)))
        self.net_line_recv.set_ydata(self.net_history_recv)

        max_kbps = max([1.0, *(self.net_history_sent or [1.0]), *(self.net_history_recv or [1.0])])
        self.net_ax.set_ylim(0, max(256, min(8192, max_kbps * 1.2)))
        self.net_ax.set_xlim(0, max(len(self.net_history_sent), len(self.net_history_recv)))
        self.net_canvas.draw()

    def _update_processes(self):
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info"]):
            procs.append(p)

        # Trigger CPU percent refresh
        for p in procs:
            try:
                _ = p.cpu_percent(interval=None)
            except Exception:
                pass

        # Slight delay if aggressive to capture delta better
        if self.update_ms <= 500:
            time.sleep(0.02)
        else:
            time.sleep(0.04)

        cpu_rank, mem_rank = [], []
        for p in procs:
            try:
                cpu = p.cpu_percent(interval=None)
                mem_rss = p.info["memory_info"].rss
                cpu_rank.append((cpu, p.info["pid"], p.info["name"]))
                mem_rank.append((mem_rss, p.info["pid"], p.info["name"]))
            except Exception:
                continue

        cpu_rank.sort(reverse=True)
        mem_rank.sort(reverse=True)

        self.proc_cpu_text.delete("1.0", tk.END)
        self.proc_mem_text.delete("1.0", tk.END)

        self.proc_cpu_text.insert(tk.END, "Top 30 by CPU (%):\n\n")
        for cpu, pid, name in cpu_rank[:30]:
            self.proc_cpu_text.insert(tk.END, f"PID {pid:<6} CPU {cpu:>6.1f}%  {name}\n")

        self.proc_mem_text.insert(tk.END, "Top 30 by Memory (RSS):\n\n")
        for rss, pid, name in mem_rank[:30]:
            self.proc_mem_text.insert(tk.END, f"PID {pid:<6} RSS {format_bytes(rss):>10}  {name}\n")

    def _update_scan(self):
        # Ports summary first (global context for scoring)
        ports_summary = self.ports_focus.analyze()

        # Show ports global summary
        self.ports_summary_text.delete("1.0", tk.END)
        self.ports_summary_text.insert(tk.END,
            f"Ports summary: SYN_SENT {ports_summary['syn_sent']}, SYN_RECV {ports_summary['syn_recv']}, LISTEN {ports_summary['listen_count']}\n"
            f"SYN burst: {ports_summary['syn_burst']} | Listen overload: {ports_summary['listen_overload']}\n"
            "Top per-PID connection bursts (approx):\n"
        )
        # Show top syn/listen counts
        pid_stats = sorted(ports_summary["by_proc"].items(), key=lambda kv: (kv[1].get("syn",0), kv[1].get("listen",0), kv[1].get("est",0)), reverse=True)[:10]
        for pid, stats in pid_stats:
            self.ports_summary_text.insert(tk.END, f"PID {pid:<6} SYN {stats.get('syn',0):<4} LISTEN {stats.get('listen',0):<4} EST {stats.get('est',0):<4}\n")

        start = time.time()
        results = []
        for p in psutil.process_iter(attrs=["pid", "name", "exe", "memory_info"]):
            try:
                res = self.detector.score_process(p, ports_summary)
                results.append(res)
            except Exception:
                continue

        order = {"suspect": 0, "watch": 1, "normal": 2, "protected": 3}
        results.sort(key=lambda r: (order.get(r["level"], 4), -r["score"], r["name"]))

        for item in self.scan_tree.get_children():
            self.scan_tree.delete(item)
        for r in results[:600]:
            level_sym = {"suspect": "⚠", "watch": "◑", "normal": "•", "protected": "🛡"}.get(r["level"], "•")
            reasons = "; ".join(r["reasons"]) if r["reasons"] else ""
            tag = r["level"] if r["level"] in ("suspect","watch","normal","protected") else "normal"
            self.scan_tree.insert("", "end", values=(r["pid"], r["name"], f"{level_sym} {r['level']}", r["score"], reasons), tags=(tag,))

        elapsed = (time.time() - start) * 1000.0
        self.scan_status.config(text=f"Laser scan: {len(results)} items | {elapsed:.1f} ms | Non-destructive (LeafShield + Rules + PortsFocus)")

# ----------------------------
# ENTRY POINT
# ----------------------------
def main():
    root = tk.Tk()
    app = TelemetryGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

