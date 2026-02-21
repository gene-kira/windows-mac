import os
import sys
import time
import json
import math
import socket
import threading
import platform
import multiprocessing
import concurrent.futures
import numpy as np

# Optional extras
try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


# ========================= CONFIG / PROFILES =========================

DEFAULT_CONFIG = {
    "profile": "balanced",  # "full", "balanced", "silent", "thermal_safe"
    "daemon_mode": False,
    "distributed": {
        "enabled": False,
        "port": 50555,
        "broadcast_interval": 2.0
    },
    "gui": {
        "enabled": True
    }
}

PROFILES = {
    "full": {
        "organisms": [
            {"name": "Alpha", "target_util": 0.9},
            {"name": "Beta", "target_util": 0.9}
        ]
    },
    "balanced": {
        "organisms": [
            {"name": "Alpha", "target_util": 0.5},
            {"name": "Beta", "target_util": 0.6}
        ]
    },
    "silent": {
        "organisms": [
            {"name": "Alpha", "target_util": 0.2}
        ]
    },
    "thermal_safe": {
        "organisms": [
            {"name": "Alpha", "target_util": 0.4}
        ]
    }
}


def load_config(path="beast_config.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg, path="beast_config.json"):
    try:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ========================= LOGGING =========================

def beast_log(msg: str):
    print(f"[BEAST] {msg}", flush=True)


# ========================= CORE WORKLOAD =========================

def beast_vector_cycle(size: int) -> float:
    x = np.random.rand(size).astype("float64")
    y = np.random.rand(size).astype("float64")
    s = np.sin(x)
    c = np.cos(y)
    out = s * c
    return float(out.sum())


# ========================= AFFINITY / NUMA =========================

def set_affinity_for_current_process(core_ids):
    if not PSUTIL:
        return
    try:
        p = psutil.Process(os.getpid())
        system = platform.system().lower()
        if system in ("windows", "linux"):
            p.cpu_affinity(core_ids)
    except Exception:
        pass


def detect_numa_groups():
    """
    Best-effort NUMA grouping:
    - If psutil has cpu_count(logical=False), we still only get counts.
    - We approximate by splitting logical cores into 2 groups if many cores.
    """
    total_cores = max(1, multiprocessing.cpu_count())
    if total_cores <= 4:
        return [list(range(total_cores))]
    half = total_cores // 2
    return [list(range(0, half)), list(range(half, total_cores))]


# ========================= TEMPERATURE =========================

def get_cpu_temperature():
    if not PSUTIL:
        return None
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        # pick first available sensor
        for name, entries in temps.items():
            if entries:
                return entries[0].current
    except Exception:
        return None
    return None


# ========================= ORGANISM / SWARM =========================

class BeastOrganism:
    def __init__(self, name: str, target_util: float = 0.5, size: int = 1_000_000):
        self.name = name
        self.target_util = max(0.05, min(target_util, 1.0))
        self.size = size
        self.last_util = 0.0
        self.last_temp = None
        self.core_group = []  # list of core IDs assigned

    def adjust(self, measured_util: float, temp: float | None):
        if measured_util <= 0:
            return

        # Temperature-aware governor
        if temp is not None and temp > 80.0:
            # too hot, back off aggressively
            self.size = max(200_000, int(self.size * 0.7))
        else:
            target = self.target_util * 100
            diff = measured_util - target
            if diff > 5:
                self.size = max(200_000, int(self.size * 0.85))
            elif diff < -5:
                self.size = min(10_000_000, int(self.size * 1.15))

        self.last_util = measured_util
        self.last_temp = temp


def worker_entry(args):
    core_ids, name, size = args
    if core_ids:
        set_affinity_for_current_process(core_ids)
    res = beast_vector_cycle(size)
    return core_ids, name, size, res


class BeastSwarm:
    def __init__(self, organisms, cfg):
        self.organisms = organisms
        self.cfg = cfg
        self.cycle = 0
        self.total_cores = max(1, multiprocessing.cpu_count())
        self.numa_groups = detect_numa_groups()
        self.per_core_usage = [0.0] * self.total_cores
        self.lock = threading.Lock()

        if PSUTIL:
            psutil.cpu_percent(interval=None)

    def assign_core_groups(self):
        """
        Core-pinned multi-organism:
        - Distribute NUMA groups across organisms.
        """
        flat_groups = self.numa_groups
        # simple round-robin assignment of groups to organisms
        for i, org in enumerate(self.organisms):
            org.core_group = []
        idx = 0
        for group in flat_groups:
            org = self.organisms[idx % len(self.organisms)]
            org.core_group.extend(group)
            idx += 1

        # if more organisms than groups, some may share
        if len(self.organisms) > len(flat_groups):
            # just ensure each has at least one core
            all_cores = list(range(self.total_cores))
            for i, org in enumerate(self.organisms):
                if not org.core_group:
                    org.core_group.append(all_cores[i % len(all_cores)])

    def telemetry(self):
        if not PSUTIL:
            return "Telemetry unavailable (psutil not installed)"
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        with self.lock:
            self.per_core_usage = per_core
        lines = []
        for i, v in enumerate(per_core):
            bar = "#" * int(v / 5)
            lines.append(f"Core {i:02d}: {v:5.1f}% | {bar}")
        return "\n".join(lines)

    def run_cycle(self):
        self.cycle += 1
        start = time.time()

        jobs = []
        # For each organism, spawn one worker per core in its group
        for org in self.organisms:
            if not org.core_group:
                continue
            for core_id in org.core_group:
                jobs.append(([core_id], org.name, org.size))

        if not jobs:
            return

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(worker_entry, jobs))

        end = time.time()
        elapsed = end - start
        util = psutil.cpu_percent(interval=None) if PSUTIL else -1
        temp = get_cpu_temperature()

        for org in self.organisms:
            org.adjust(util, temp)

        return results, util, temp, elapsed


# ========================= DISTRIBUTED BEAST =========================

class BeastNetwork:
    def __init__(self, cfg):
        self.enabled = cfg.get("distributed", {}).get("enabled", False)
        self.port = cfg.get("distributed", {}).get("port", 50555)
        self.interval = cfg.get("distributed", {}).get("broadcast_interval", 2.0)
        self.stop_flag = threading.Event()
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        hostname = socket.gethostname()
        while not self.stop_flag.is_set():
            msg = f"BEAST_HEARTBEAT {hostname}"
            try:
                sock.sendto(msg.encode("utf-8"), ("255.255.255.255", self.port))
            except Exception:
                pass
            time.sleep(self.interval)


# ========================= GUI COCKPIT =========================

class BeastGUI:
    def __init__(self, swarm: BeastSwarm):
        self.swarm = swarm
        self.root = tk.Tk()
        self.root.title("Beast CPU Cockpit")
        self.core_bars = []
        self.info_label = None
        self._build_ui()
        self.update_interval_ms = 1000

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        self.info_label = ttk.Label(frame, text="Initializing…", anchor="w", justify="left")
        self.info_label.pack(fill="x", pady=(0, 10))

        cores_frame = ttk.Frame(frame)
        cores_frame.pack(fill="both", expand=True)

        for i in range(self.swarm.total_cores):
            row = ttk.Frame(cores_frame)
            row.pack(fill="x")
            lbl = ttk.Label(row, text=f"Core {i:02d}", width=8)
            lbl.pack(side="left")
            bar = ttk.Progressbar(row, orient="horizontal", mode="determinate", maximum=100)
            bar.pack(side="left", fill="x", expand=True, padx=5)
            self.core_bars.append(bar)

    def update(self):
        with self.swarm.lock:
            per_core = self.swarm.per_core_usage[:]

        for i, v in enumerate(per_core):
            if i < len(self.core_bars):
                self.core_bars[i]["value"] = v

        lines = [f"Cycle: {self.swarm.cycle}"]
        for org in self.swarm.organisms:
            lines.append(
                f"{org.name}: target={int(org.target_util*100)}% "
                f"size={org.size:,} last_util={org.last_util:.1f}% "
                f"temp={org.last_temp if org.last_temp is not None else 'N/A'}"
            )
        self.info_label.config(text="\n".join(lines))

        self.root.after(self.update_interval_ms, self.update)

    def run(self):
        self.update()
        self.root.mainloop()


# ========================= DAEMON MODE =========================

def daemonize_if_requested(cfg):
    if not cfg.get("daemon_mode", False):
        return
    if platform.system().lower() == "windows":
        # No real fork-daemon on Windows; just continue in foreground
        return
    # Unix-style daemonization
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "r") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(os.devnull, "a+") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())


# ========================= MAIN LOOP =========================

def main():
    cfg = load_config()
    profile_name = cfg.get("profile", "balanced")
    profile = PROFILES.get(profile_name, PROFILES["balanced"])

    daemonize_if_requested(cfg)

    beast_log(f"Starting BEAST TOTAL UNIFIED | profile={profile_name}")
    beast_log(f"Platform: {platform.system()} | psutil={PSUTIL} | tkinter={TK_AVAILABLE}")

    organisms = []
    for o in profile["organisms"]:
        organisms.append(BeastOrganism(o["name"], o["target_util"]))

    swarm = BeastSwarm(organisms, cfg)
    swarm.assign_core_groups()

    net = BeastNetwork(cfg)
    net.start()

    gui_enabled = cfg.get("gui", {}).get("enabled", True) and TK_AVAILABLE

    if gui_enabled:
        # Run swarm in background thread, GUI in main thread
        def swarm_thread():
            while True:
                result = swarm.run_cycle()
                if result is None:
                    time.sleep(0.5)
                    continue
                _, util, temp, elapsed = result
                print("\n" + "=" * 60)
                beast_log(
                    f"Cycle {swarm.cycle} | CPU={util if util>=0 else 'N/A'}% "
                    f"| Temp={temp if temp is not None else 'N/A'}C | Time={elapsed:.3f}s"
                )
                print("-" * 60)
                print(swarm.telemetry())
                print("=" * 60)

        t = threading.Thread(target=swarm_thread, daemon=True)
        t.start()

        gui = BeastGUI(swarm)
        gui.run()
    else:
        # Console-only mode
        while True:
            result = swarm.run_cycle()
            if result is None:
                time.sleep(0.5)
                continue
            _, util, temp, elapsed = result
            print("\n" + "=" * 60)
            beast_log(
                f"Cycle {swarm.cycle} | CPU={util if util>=0 else 'N/A'}% "
                f"| Temp={temp if temp is not None else 'N/A'}C | Time={elapsed:.3f}s"
            )
            print("-" * 60)
            print(swarm.telemetry())
            print("=" * 60)


if __name__ == "__main__":
    main()

