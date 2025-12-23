import importlib
import os
import threading
import time
import platform
import psutil
import queue
import datetime
import json
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================
# CONFIG / CONSTANTS
# ============================================================

CONFIG_FILE = "borgnet_config.json"

# Network drive health states
DRIVE_STATUS_ONLINE = "ONLINE"
DRIVE_STATUS_OFFLINE = "OFFLINE"
DRIVE_STATUS_FAILING = "FAILING"
DRIVE_STATUS_UNKNOWN = "UNKNOWN"


# ============================================================
# CONFIG PERSISTENCE
# ============================================================

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[Config] Failed to save config: {e}")


# ============================================================
# AUTOLOADER
# ============================================================

class AutoLoader:
    """
    Auto-loads Python modules from a given folder.
    Used for optional plugins / extensions.
    """
    def __init__(self, module_folder="modules"):
        self.module_folder = module_folder
        self.modules = {}

    def load_all(self):
        if not os.path.exists(self.module_folder):
            os.makedirs(self.module_folder)

        for file in os.listdir(self.module_folder):
            if file.endswith(".py") and not file.startswith("_"):
                name = file[:-3]
                try:
                    module = importlib.import_module(f"{self.module_folder}.{name}")
                    self.modules[name] = module
                    print(f"[AutoLoader] Loaded module: {name}")
                except Exception as e:
                    print(f"[AutoLoader] Failed to load {name}: {e}")

    def get(self, name):
        return self.modules.get(name, None)


# ============================================================
# SYSTEM INVENTORY
# ============================================================

class SystemInventory:
    """
    Scans hardware and OS details of the host system.
    """
    def __init__(self):
        self.info = {}

    def scan(self):
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        self.info = {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "cores": psutil.cpu_count(logical=True),
            "memory_total": vm.total,
            "memory_available": vm.available,
            "disk_total": disk.total,
            "disk_free": disk.free,
            "network_ifaces": list(psutil.net_if_addrs().keys())
        }
        return self.info


# ============================================================
# MEMORY ENGINE (TRAFFIC + BEHAVIOR + PERSISTENCE)
# ============================================================

class MemoryEngine:
    """
    Long-lived memory of network behavior & system patterns.
    Not packet-level data, just patterns & endpoints.
    """
    def __init__(self, persist_file="borgnet_memory.json"):
        self.persist_file = persist_file
        self.visited_sites = set()
        self.connection_patterns = []  # (laddr, raddr, family, type)
        self.system_history = []       # snapshots of system state

        # Endpoint behavior
        self.endpoint_frequency = defaultdict(int)
        self.endpoint_last_seen = {}           # endpoint -> ISO timestamp string
        self.endpoint_hours = defaultdict(set) # endpoint -> set of hours when seen

        self._lock = threading.Lock()

        self._load()

    def record_site(self, site):
        with self._lock:
            self.visited_sites.add(site)

    def record_connection(self, conn_tuple):
        """
        conn_tuple: (laddr, raddr, family, type)
        """
        _, raddr, _, _ = conn_tuple
        endpoint_key = f"{raddr.ip}:{raddr.port}"
        now = datetime.datetime.now()
        hour = now.hour

        with self._lock:
            self.connection_patterns.append(conn_tuple)
            self.endpoint_frequency[endpoint_key] += 1
            self.endpoint_last_seen[endpoint_key] = now.isoformat()
            self.endpoint_hours[endpoint_key].add(hour)

    def record_system_state(self, state):
        with self._lock:
            self.system_history.append(state)

    def snapshot(self):
        """
        Returns a safe snapshot for GUI / other components.
        """
        with self._lock:
            return {
                "visited_sites": list(self.visited_sites),
                "connection_count": len(self.connection_patterns),
                "endpoint_frequency": dict(self.endpoint_frequency),
                "system_history_len": len(self.system_history),
                "endpoint_last_seen": dict(self.endpoint_last_seen),
                "endpoint_hours": {
                    ep: list(hours) for ep, hours in self.endpoint_hours.items()
                },
            }

    def _load(self):
        if not os.path.exists(self.persist_file):
            return
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.visited_sites = set(data.get("visited_sites", []))
            self.endpoint_frequency = defaultdict(
                int, data.get("endpoint_frequency", {})
            )
            print("[MemoryEngine] Loaded persisted memory.")
        except Exception as e:
            print(f"[MemoryEngine] Failed to load memory: {e}")

    def save(self):
        try:
            snap = self.snapshot()
            data = {
                "visited_sites": snap["visited_sites"],
                "endpoint_frequency": snap["endpoint_frequency"],
                # last_seen/hours are runtime-only and will rebuild over time
            }
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("[MemoryEngine] Memory persisted.")
        except Exception as e:
            print(f"[MemoryEngine] Failed to save memory: {e}")


# ============================================================
# THREAT DETECTOR (ANOMALY / "ROGUE-LIKE" FLAGGING + SUMMARY)
# ============================================================

class ThreatDetector(threading.Thread):
    """
    Anomaly detection to spot "rogue-like" connections:
    - New endpoints (never seen before).
    - Rare endpoints (very low frequency).
    - Endpoints contacted at unusual times based on prior history.

    Severity:
    - INFO      : new endpoint
    - WATCH     : rare endpoint
    - ATTENTION : unusual-time access (high severity)
    """
    def __init__(self, memory_engine, alert_queue):
        super().__init__(daemon=True)
        self.memory = memory_engine
        self.alert_queue = alert_queue
        self.running = True

        self.known_endpoints = set()
        self.rare_endpoints_flagged = set()
        self.unusual_time_flagged = set()

        # Tunable thresholds
        self.rare_threshold = 2          # <= 2 sightings considered "rare"
        self.min_baseline_for_time = 3   # need at least 3 sightings to build pattern

        # Stats for session summary
        self.alert_log = []
        self.stats_lock = threading.Lock()
        self.new_count = 0
        self.rare_count = 0
        self.unusual_time_count = 0

    def _log_alert(self, level, message):
        text = f"[{level}] {message}"
        self.alert_queue.put(text)
        with self.stats_lock:
            self.alert_log.append(text)
            if level == "INFO":
                self.new_count += 1
            elif level == "WATCH":
                self.rare_count += 1
            elif level == "ATTENTION":
                self.unusual_time_count += 1

    def run(self):
        while self.running:
            try:
                snap = self.memory.snapshot()
                freq_map = snap["endpoint_frequency"]
                hours_map = snap["endpoint_hours"]
                last_seen_map = snap["endpoint_last_seen"]

                now = datetime.datetime.now()
                current_hour = now.hour

                for endpoint, freq in freq_map.items():
                    hours_list = hours_map.get(endpoint, [])
                    last_seen = last_seen_map.get(endpoint, "unknown")

                    # 1) New endpoint
                    if endpoint not in self.known_endpoints and freq == 1:
                        self.known_endpoints.add(endpoint)
                        self._log_alert(
                            "INFO",
                            f"New endpoint observed (possible rogue): {endpoint} "
                            f"(first seen at {last_seen})"
                        )
                        continue

                    # 2) Rare endpoint
                    if (
                        endpoint not in self.rare_endpoints_flagged
                        and 1 < freq <= self.rare_threshold
                    ):
                        self.rare_endpoints_flagged.add(endpoint)
                        self._log_alert(
                            "WATCH",
                            f"Rare endpoint (possible rogue): {endpoint} "
                            f"(seen {freq} times, last seen at {last_seen})"
                        )

                    # 3) Unusual-time endpoint (HIGH severity)
                    if (
                        endpoint not in self.unusual_time_flagged
                        and freq >= self.min_baseline_for_time
                        and len(hours_list) > 0
                        and current_hour not in hours_list
                    ):
                        self.unusual_time_flagged.add(endpoint)
                        self._log_alert(
                            "ATTENTION",
                            f"Endpoint contacted at unusual time (possible rogue): {endpoint} "
                            f"(normal hours: {sorted(hours_list)}, current hour: {current_hour})"
                        )

            except Exception as e:
                self._log_alert("INFO", f"ThreatDetector error: {e}")
            time.sleep(3)

    def stop(self):
        self.running = False

    def get_summary(self):
        """
        Returns a dictionary of session-level anomaly stats and the last few alerts.
        """
        with self.stats_lock:
            return {
                "new_endpoints": self.new_count,
                "rare_endpoints": self.rare_count,
                "unusual_time_endpoints": self.unusual_time_count,
                "total_alerts": len(self.alert_log),
                "recent_alerts": self.alert_log[-10:],  # last 10 alerts
            }

    def has_high_severity_alerts(self):
        """
        Returns True if any high-severity (ATTENTION) alerts have been seen this run.
        """
        with self.stats_lock:
            return self.unusual_time_count > 0


# ============================================================
# NETWORK OBSERVER (SAFE, PASSIVE)
# ============================================================

class NetworkObserver(threading.Thread):
    """
    Passive observer of OS-level connections via psutil.
    No packet interception; just metadata.
    """
    def __init__(self, memory_engine, config):
        super().__init__(daemon=True)
        self.memory = memory_engine
        self.config = config
        self.running = True

    def run(self):
        while self.running:
            try:
                interval = self.config.settings["background_scan_interval"]
                conns = psutil.net_connections()
                for c in conns:
                    if c.raddr and c.laddr:
                        conn_tuple = (c.laddr, c.raddr, c.family, c.type)
                        self.memory.record_connection(conn_tuple)
                time.sleep(interval)
            except Exception as e:
                print(f"[NetworkObserver] Error: {e}")
                time.sleep(5)

    def stop(self):
        self.running = False


# ============================================================
# SIMULATION ENGINE
# ============================================================

class SimulationEngine(threading.Thread):
    """
    Runs lightweight simulations based on historical patterns.
    Placeholder for future advanced behavior.
    """
    def __init__(self, memory_engine, config):
        super().__init__(daemon=True)
        self.memory = memory_engine
        self.config = config
        self.running = True

    def run(self):
        while self.running:
            try:
                snap = self.memory.snapshot()
                conn_count = snap["connection_count"]
                if conn_count > 0:
                    # Placeholder for future optimization simulations
                    pass
                time.sleep(5)
            except Exception as e:
                print(f"[SimulationEngine] Error: {e}")
                time.sleep(5)

    def stop(self):
        self.running = False


# ============================================================
# TIME-AWARE BRAIN
# ============================================================

class TimeAwareBrain(threading.Thread):
    """
    Adjusts config based on time of day / usage patterns.
    Example: 'gaming hours' in evenings.
    """
    def __init__(self, config, gaming_manager):
        super().__init__(daemon=True)
        self.config = config
        self.gaming_manager = gaming_manager
        self.running = True

    def run(self):
        while self.running:
            now = datetime.datetime.now()
            hour = now.hour

            if 18 <= hour <= 23:
                self.config.set_mode("gaming")
                self.gaming_manager.enable()
            else:
                self.config.set_mode("balanced")
                self.gaming_manager.disable()

            time.sleep(60)

    def stop(self):
        self.running = False


# ============================================================
# GAMING MODE MANAGER
# ============================================================

class GamingModeManager:
    """
    Manages 'gaming' vs 'normal' behavior flags.
    Stub for future QoS / prioritization logic.
    """
    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()

    def enable(self):
        with self._lock:
            self.enabled = True

    def disable(self):
        with self._lock:
            self.enabled = False

    def is_enabled(self):
        with self._lock:
            return self.enabled

    def toggle(self):
        with self._lock:
            self.enabled = not self.enabled
            return self.enabled


# ============================================================
# ADAPTIVE CONFIGURATION ENGINE
# ============================================================

class AdaptiveConfig:
    """
    Dynamic configuration that adapts to system resources,
    time of day, and modes (e.g., gaming).
    """
    def __init__(self):
        self.settings = {
            "mode": "balanced",            # "balanced" or "gaming"
            "gui_refresh_rate": 1.0,       # seconds
            "background_scan_interval": 5, # seconds
            "memory_safety_margin_mb": 500
        }
        self._lock = threading.Lock()

    def adapt_to_system(self, system_info):
        with self._lock:
            available_mb = system_info["memory_available"] / (1024 * 1024)
            margin = self.settings["memory_safety_margin_mb"]

            if available_mb < margin:
                self.settings["gui_refresh_rate"] = 2.0
                self.settings["background_scan_interval"] = 7
            else:
                self.settings["gui_refresh_rate"] = 1.0
                self.settings["background_scan_interval"] = 5

    def set_mode(self, mode):
        if mode not in ("balanced", "gaming"):
            return
        with self._lock:
            self.settings["mode"] = mode
            if mode == "gaming":
                self.settings["gui_refresh_rate"] = min(
                    self.settings["gui_refresh_rate"], 1.0
                )
            else:
                self.settings["gui_refresh_rate"] = max(
                    self.settings["gui_refresh_rate"], 1.0
                )

    def snapshot(self):
        with self._lock:
            return dict(self.settings)


# ============================================================
# SHARED STATE SYNC / COMPARISON (DUAL NETWORK DRIVES)
# ============================================================

class SharedStateSync(threading.Thread):
    """
    Periodically writes this node's state to a shared directory and,
    if Queen, compares Queen vs Backup states to monitor divergence.
    Uses primary drive if possible; falls back to secondary if needed.
    Tracks health for both drives.
    """
    def __init__(self, node, get_primary_dir_func, get_secondary_dir_func):
        super().__init__(daemon=True)
        self.node = node
        self.get_primary_dir = get_primary_dir_func
        self.get_secondary_dir = get_secondary_dir_func
        self.running = True

        self._status_lock = threading.Lock()
        self._primary_status = DRIVE_STATUS_UNKNOWN
        self._secondary_status = DRIVE_STATUS_UNKNOWN

    def _set_primary_status(self, status):
        with self._status_lock:
            self._primary_status = status

    def _set_secondary_status(self, status):
        with self._status_lock:
            self._secondary_status = status

    def get_statuses(self):
        with self._status_lock:
            return self._primary_status, self._secondary_status

    def _state_file_path(self, shared_dir):
        safe_name = self.node.name.replace(" ", "_")
        return os.path.join(shared_dir, f"{safe_name}_state.json")

    def _write_state_to_dir(self, shared_dir, primary=True):
        """
        Write and verify state in the given directory.
        Returns True on success, False on failure.
        """
        try:
            if not shared_dir or not os.path.exists(shared_dir):
                if primary:
                    self._set_primary_status(DRIVE_STATUS_OFFLINE)
                else:
                    self._set_secondary_status(DRIVE_STATUS_OFFLINE)
                return False

            mem_snap = self.node.memory.snapshot()
            cfg_snap = self.node.config.snapshot()
            state = {
                "name": self.node.name,
                "is_queen": self.node.is_queen,
                "config": cfg_snap,
                "memory_summary": {
                    "visited_sites": len(mem_snap["visited_sites"]),
                    "connection_count": mem_snap["connection_count"],
                    "known_endpoints": len(mem_snap["endpoint_frequency"]),
                    "system_history_len": mem_snap["system_history_len"],
                },
                "timestamp": datetime.datetime.now().isoformat()
            }

            fpath = self._state_file_path(shared_dir)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    _ = json.load(f)
            except Exception:
                if primary:
                    self._set_primary_status(DRIVE_STATUS_FAILING)
                else:
                    self._set_secondary_status(DRIVE_STATUS_FAILING)
                return False

            if primary:
                self._set_primary_status(DRIVE_STATUS_ONLINE)
            else:
                self._set_secondary_status(DRIVE_STATUS_ONLINE)
            return True

        except Exception as e:
            print(f"[SharedStateSync] Error writing state: {e}")
            if primary:
                self._set_primary_status(DRIVE_STATUS_FAILING)
            else:
                self._set_secondary_status(DRIVE_STATUS_FAILING)
            return False

    def _compare_states_as_queen(self, shared_dir):
        """
        Queen reads all *_state.json files and compares key metrics.
        """
        try:
            if not shared_dir or not os.path.exists(shared_dir):
                return

            files = [f for f in os.listdir(shared_dir) if f.endswith("_state.json")]
            states = []
            for fname in files:
                fpath = os.path.join(shared_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        states.append(json.load(f))
                except Exception:
                    continue

            if len(states) < 2:
                return  # need at least Queen + one other

            base = states[0]
            for other in states[1:]:
                if base["memory_summary"] != other["memory_summary"]:
                    self.node.alert_queue.put(
                        "[INFO] Shared state comparison: differences detected between "
                        f"{base['name']} and {other['name']}"
                    )

        except Exception as e:
            print(f"[SharedStateSync] Error comparing states: {e}")

    def run(self):
        while self.running and self.node.is_running():
            primary_dir = self.get_primary_dir()
            secondary_dir = self.get_secondary_dir()

            # First try primary
            used_dir = None
            if primary_dir:
                ok_primary = self._write_state_to_dir(primary_dir, primary=True)
                if ok_primary:
                    used_dir = primary_dir

            # If primary failed, try secondary
            if used_dir is None and secondary_dir:
                ok_secondary = self._write_state_to_dir(secondary_dir, primary=False)
                if ok_secondary:
                    used_dir = secondary_dir

            # If Queen and we used some directory, compare states
            if used_dir and self.node.is_queen:
                self._compare_states_as_queen(used_dir)

            time.sleep(10)

    def stop(self):
        self.running = False


# ============================================================
# QUEEN + DRONE ARCHITECTURE
# ============================================================

class BorgNode:
    """
    Core node: can act as Queen or Drone.
    Monitors system and network, adapts config, runs simulations.
    """
    def __init__(
        self,
        is_queen=False,
        name="Node",
        persist_file="borgnet_memory.json",
        get_primary_dir_func=None,
        get_secondary_dir_func=None
    ):
        self.is_queen = is_queen
        self.name = name

        self.inventory = SystemInventory()
        self.config = AdaptiveConfig()
        self.memory = MemoryEngine(persist_file=persist_file)
        self.gaming_manager = GamingModeManager()

        self.alert_queue = queue.Queue()

        self.observer = NetworkObserver(self.memory, self.config)
        self.simulator = SimulationEngine(self.memory, self.config)
        self.time_brain = TimeAwareBrain(self.config, self.gaming_manager)
        self.threat_detector = ThreatDetector(self.memory, self.alert_queue)
        self.shared_sync = SharedStateSync(self, get_primary_dir_func, get_secondary_dir_func)

        self._running = False

    def start(self):
        system_info = self.inventory.scan()
        self.config.adapt_to_system(system_info)

        print(f"[{self.name}] System inventory collected:")
        for k, v in system_info.items():
            print(f"  {k}: {v}")

        self._running = True
        self.observer.start()
        self.simulator.start()
        self.time_brain.start()
        self.threat_detector.start()
        self.shared_sync.start()

    def stop(self):
        self._running = False
        self.observer.stop()
        self.simulator.stop()
        self.time_brain.stop()
        self.threat_detector.stop()
        self.shared_sync.stop()
        self.memory.save()

    def is_running(self):
        return self._running


class BorgQueen(BorgNode):
    """
    Queen node: coordinates drones conceptually.
    """
    def __init__(
        self,
        name="Queen",
        persist_file="borgnet_queen_memory.json",
        get_primary_dir_func=None,
        get_secondary_dir_func=None
    ):
        super().__init__(
            is_queen=True,
            name=name,
            persist_file=persist_file,
            get_primary_dir_func=get_primary_dir_func,
            get_secondary_dir_func=get_secondary_dir_func
        )
        self.drones = []

    def register_drone(self, drone):
        self.drones.append(drone)
        print(f"[{self.name}] Registered drone: {drone.name}")


# ============================================================
# TKINTER GUI (BORG-THEMED PANEL + STATUS)
# ============================================================

class BorgGUI:
    """
    Tkinter-based GUI:
    - Borg-themed network drive panel
    - Shows primary & secondary path + ONLINE/OFFLINE/FAILING
    - Lets user pick primary & secondary drives
    - Shows Borg alert when high-severity anomalies exist
    """

    def __init__(self, root, queen, shared_dirs_ref, config_ref):
        self.root = root
        self.queen = queen
        self.shared_dirs_ref = shared_dirs_ref  # dict with {"primary": ..., "secondary": ...}
        self.config_ref = config_ref            # config dict

        self.root.title("BorgNet Guardian")
        self.root.geometry("800x550")

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        # Main frame
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Top: Borg summary
        summary_frame = ttk.LabelFrame(main, text="BorgNet Node Status")
        summary_frame.pack(fill=tk.X, pady=5)

        self.node_label = ttk.Label(summary_frame, text="")
        self.node_label.pack(anchor="w")

        self.mode_label = ttk.Label(summary_frame, text="")
        self.mode_label.pack(anchor="w")

        self.gaming_label = ttk.Label(summary_frame, text="")
        self.gaming_label.pack(anchor="w")

        # Network drive Borg-themed panel
        drive_frame = ttk.LabelFrame(main, text="Collective Storage Node")
        drive_frame.pack(fill=tk.X, pady=5)

        # Primary
        ttk.Label(drive_frame, text="Primary Path:").grid(row=0, column=0, sticky="w")
        self.primary_path_var = tk.StringVar(value="(none)")
        self.primary_status_var = tk.StringVar(value=DRIVE_STATUS_UNKNOWN)

        self.primary_path_label = ttk.Label(drive_frame, textvariable=self.primary_path_var)
        self.primary_path_label.grid(row=0, column=1, sticky="w", padx=5)

        self.primary_button = ttk.Button(
            drive_frame,
            text="Select Primary Drive",
            command=self._select_primary
        )
        self.primary_button.grid(row=0, column=2, padx=10)

        ttk.Label(drive_frame, text="Primary Status:").grid(row=1, column=0, sticky="w")
        self.primary_status_label = ttk.Label(drive_frame, textvariable=self.primary_status_var)
        self.primary_status_label.grid(row=1, column=1, sticky="w", padx=5)

        # Secondary
        ttk.Label(drive_frame, text="Secondary Path:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.secondary_path_var = tk.StringVar(value="(none)")
        self.secondary_status_var = tk.StringVar(value=DRIVE_STATUS_UNKNOWN)

        self.secondary_path_label = ttk.Label(drive_frame, textvariable=self.secondary_path_var)
        self.secondary_path_label.grid(row=2, column=1, sticky="w", padx=5, pady=(10, 0))

        self.secondary_button = ttk.Button(
            drive_frame,
            text="Select Secondary Drive",
            command=self._select_secondary
        )
        self.secondary_button.grid(row=2, column=2, padx=10, pady=(10, 0))

        ttk.Label(drive_frame, text="Secondary Status:").grid(row=3, column=0, sticky="w")
        self.secondary_status_label = ttk.Label(drive_frame, textvariable=self.secondary_status_var)
        self.secondary_status_label.grid(row=3, column=1, sticky="w", padx=5)

        # Alerts panel
        alerts_frame = ttk.LabelFrame(main, text="Alerts / Rogue-like Activity")
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.alerts_text = tk.Text(alerts_frame, height=10, wrap="word")
        self.alerts_text.pack(fill=tk.BOTH, expand=True)

        # Borg high-severity banner
        self.borg_banner_var = tk.StringVar(value="")
        self.borg_banner_label = ttk.Label(
            main,
            textvariable=self.borg_banner_var,
            foreground="red",
            font=("TkDefaultFont", 10, "bold")
        )
        self.borg_banner_label.pack(fill=tk.X, pady=5)

        # Bottom controls (gaming toggle + report)
        controls_frame = ttk.Frame(main)
        controls_frame.pack(fill=tk.X, pady=5)

        self.gaming_button = ttk.Button(
            controls_frame,
            text="Toggle Gaming Mode",
            command=self._toggle_gaming_mode
        )
        self.gaming_button.pack(side=tk.LEFT, padx=5)

        self.report_button = ttk.Button(
            controls_frame,
            text="Show Anomaly Report",
            command=self._show_report
        )
        self.report_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = ttk.Button(
            controls_frame,
            text="Quit",
            command=self._quit
        )
        self.quit_button.pack(side=tk.RIGHT, padx=5)

    def _select_primary(self):
        path = filedialog.askdirectory(title="Select Primary BorgNet Shared Network Drive")
        if path:
            self.shared_dirs_ref["primary"] = path
            self.primary_path_var.set(path)
            self.config_ref["primary_shared_dir"] = path
            save_config(self.config_ref)

    def _select_secondary(self):
        path = filedialog.askdirectory(title="Select Secondary BorgNet Shared Network Drive")
        if path:
            self.shared_dirs_ref["secondary"] = path
            self.secondary_path_var.set(path)
            self.config_ref["secondary_shared_dir"] = path
            save_config(self.config_ref)

    def _toggle_gaming_mode(self):
        state = self.queen.gaming_manager.toggle()
        mode = "ENABLED" if state else "DISABLED"
        messagebox.showinfo("Gaming Mode", f"Gaming mode {mode}")

    def _show_report(self):
        summary = self.queen.threat_detector.get_summary()
        lines = [
            "=== Anomaly / Rogue-like Activity Summary ===",
            f"New endpoints:          {summary['new_endpoints']}",
            f"Rare endpoints:         {summary['rare_endpoints']}",
            f"Unusual-time endpoints: {summary['unusual_time_endpoints']}",
            f"Total alerts:           {summary['total_alerts']}",
            "",
            "Recent alerts:"
        ]
        lines.extend(summary["recent_alerts"])
        report_text = "\n".join(lines)
        messagebox.showinfo("Anomaly Report", report_text)

    def _quit(self):
        self.root.quit()

    def _schedule_update(self):
        self._update_gui()
        self.root.after(1000, self._schedule_update)

    def _update_gui(self):
        if not self.queen.is_running():
            return

        cfg = self.queen.config.snapshot()
        self.node_label.config(text=f"Node: {self.queen.name} (Queen)")
        self.mode_label.config(text=f"Mode: {cfg['mode']}")
        self.gaming_label.config(
            text=f"Gaming Mode: {'ON' if self.queen.gaming_manager.is_enabled() else 'OFF'}"
        )

        # Update paths from shared ref
        primary_path = self.shared_dirs_ref.get("primary") or "(none)"
        secondary_path = self.shared_dirs_ref.get("secondary") or "(none)"
        self.primary_path_var.set(primary_path)
        self.secondary_path_var.set(secondary_path)

        # Drive statuses from shared_sync
        primary_status, secondary_status = self.queen.shared_sync.get_statuses()
        self.primary_status_var.set(primary_status)
        self.secondary_status_var.set(secondary_status)

        # Color coding
        def color_for_status(status):
            if status == DRIVE_STATUS_ONLINE:
                return "green"
            if status == DRIVE_STATUS_OFFLINE:
                return "orange"
            if status == DRIVE_STATUS_FAILING:
                return "red"
            return "black"

        self.primary_status_label.config(foreground=color_for_status(primary_status))
        self.secondary_status_label.config(foreground=color_for_status(secondary_status))

        # Alerts: drain and append
        new_alerts = []
        while not self.queen.alert_queue.empty():
            try:
                new_alerts.append(self.queen.alert_queue.get_nowait())
            except queue.Empty:
                break

        if new_alerts:
            for a in new_alerts:
                self.alerts_text.insert(tk.END, a + "\n")
            self.alerts_text.see(tk.END)

        # Borg banner for high severity
        if self.queen.threat_detector.has_high_severity_alerts():
            self.borg_banner_var.set("WE ARE THE BORG — anomaly detected")
        else:
            self.borg_banner_var.set("")


# ============================================================
# MAIN
# ============================================================

def main():
    # Load config (for shared dirs)
    cfg = load_config()
    shared_dirs_ref = {
        "primary": cfg.get("primary_shared_dir", ""),
        "secondary": cfg.get("secondary_shared_dir", "")
    }

    def get_primary_dir():
        return shared_dirs_ref.get("primary", "")

    def get_secondary_dir():
        return shared_dirs_ref.get("secondary", "")

    # Auto-loader (plugins, optional)
    loader = AutoLoader()
    loader.load_all()

    # Create Queen and Backup nodes
    queen = BorgQueen(
        name="BorgNet-Queen",
        persist_file="borgnet_queen_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir
    )
    queen.start()

    backup_drone = BorgNode(
        is_queen=False,
        name="BorgNet-Backup",
        persist_file="borgnet_backup_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir
    )
    queen.register_drone(backup_drone)
    backup_drone.start()

    # Build and run Tk GUI in main thread
    root = tk.Tk()
    gui = BorgGUI(root, queen, shared_dirs_ref, cfg)

    try:
        root.mainloop()
    finally:
        print("\n[Main] Shutting down BorgNet Guardian...")
        queen.stop()
        backup_drone.stop()

        # Final anomaly summary on shutdown
        summary = queen.threat_detector.get_summary()
        print("\n=== Final Anomaly / Rogue-like Activity Summary ===")
        print(f"New endpoints:         {summary['new_endpoints']}")
        print(f"Rare endpoints:        {summary['rare_endpoints']}")
        print(f"Unusual-time endpoints:{summary['unusual_time_endpoints']}")
        print(f"Total alerts:          {summary['total_alerts']}")
        print("===================================================")
        time.sleep(1)
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()

