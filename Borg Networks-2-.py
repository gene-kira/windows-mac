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
# MEMORY ENGINE
# ============================================================

class MemoryEngine:
    def __init__(self, persist_file="borgnet_memory.json"):
        self.persist_file = persist_file
        self.visited_sites = set()
        self.connection_patterns = []  # (laddr, raddr, family, type)
        self.system_history = []

        self.endpoint_frequency = defaultdict(int)
        self.endpoint_last_seen = {}
        self.endpoint_hours = defaultdict(set)

        self._lock = threading.Lock()
        self._load()

    def record_site(self, site):
        with self._lock:
            self.visited_sites.add(site)

    def record_connection(self, conn_tuple):
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
            }
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("[MemoryEngine] Memory persisted.")
        except Exception as e:
            print(f"[MemoryEngine] Failed to save memory: {e}")


# ============================================================
# THREAT DETECTOR
# ============================================================

class ThreatDetector(threading.Thread):
    def __init__(self, memory_engine, alert_queue):
        super().__init__(daemon=True)
        self.memory = memory_engine
        self.alert_queue = alert_queue
        self.running = True

        self.known_endpoints = set()
        self.rare_endpoints_flagged = set()
        self.unusual_time_flagged = set()

        self.rare_threshold = 2
        self.min_baseline_for_time = 3

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

                    if endpoint not in self.known_endpoints and freq == 1:
                        self.known_endpoints.add(endpoint)
                        self._log_alert(
                            "INFO",
                            f"New endpoint observed (possible rogue): {endpoint} "
                            f"(first seen at {last_seen})"
                        )
                        continue

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
        with self.stats_lock:
            return {
                "new_endpoints": self.new_count,
                "rare_endpoints": self.rare_count,
                "unusual_time_endpoints": self.unusual_time_count,
                "total_alerts": len(self.alert_log),
                "recent_alerts": self.alert_log[-10:],
            }

    def has_high_severity_alerts(self):
        with self.stats_lock:
            return self.unusual_time_count > 0


# ============================================================
# NETWORK OBSERVER
# ============================================================

class NetworkObserver(threading.Thread):
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
# ADAPTIVE CONFIG
# ============================================================

class AdaptiveConfig:
    def __init__(self):
        self.settings = {
            "mode": "balanced",
            "gui_refresh_rate": 1.0,
            "background_scan_interval": 5,
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
# SHARED STATE SYNC (DUAL DRIVES + RESYNC)
# ============================================================

class SharedStateSync(threading.Thread):
    """
    - Writes node state to active drive (primary preferred, secondary fallback)
    - Tracks drive health
    - As Queen: compares states to detect differences
    - When a drive transitions OFFLINE/FAILING -> ONLINE, resyncs it with latest collective state
    """
    def __init__(self, node, get_primary_dir_func, get_secondary_dir_func, shared_registry):
        super().__init__(daemon=True)
        self.node = node
        self.get_primary_dir = get_primary_dir_func
        self.get_secondary_dir = get_secondary_dir_func
        self.running = True

        self._status_lock = threading.Lock()
        self._primary_status = DRIVE_STATUS_UNKNOWN
        self._secondary_status = DRIVE_STATUS_UNKNOWN
        self._last_primary_status = DRIVE_STATUS_UNKNOWN
        self._last_secondary_status = DRIVE_STATUS_UNKNOWN

        self.shared_registry = shared_registry
        self.registry_lock = shared_registry["_lock"]

    def _set_primary_status(self, status):
        with self._status_lock:
            self._last_primary_status = self._primary_status
            self._primary_status = status

    def _set_secondary_status(self, status):
        with self._status_lock:
            self._last_secondary_status = self._secondary_status
            self._secondary_status = status

    def get_statuses(self):
        with self._status_lock:
            return self._primary_status, self._secondary_status

    def _state_file_path(self, shared_dir, node_name=None):
        safe_name = (node_name or self.node.name).replace(" ", "_")
        return os.path.join(shared_dir, f"{safe_name}_state.json")

    def _write_state_to_dir(self, shared_dir, primary=True):
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

            fpath = self._state_file_path(shared_dir, self.node.name)
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

            with self.registry_lock:
                self.shared_registry["nodes"][self.node.name] = state

            return True

        except Exception as e:
            print(f"[SharedStateSync] Error writing state: {e}")
            if primary:
                self._set_primary_status(DRIVE_STATUS_FAILING)
            else:
                self._set_secondary_status(DRIVE_STATUS_FAILING)
            return False

    def _compare_states_as_queen(self, used_dir):
        try:
            if not used_dir or not os.path.exists(used_dir):
                return

            files = [f for f in os.listdir(used_dir) if f.endswith("_state.json")]
            states = []
            for fname in files:
                fpath = os.path.join(used_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        states.append(json.load(f))
                except Exception:
                    continue

            if len(states) < 2:
                return

            base = states[0]
            for other in states[1:]:
                if base["memory_summary"] != other["memory_summary"]:
                    self.node.alert_queue.put(
                        "[INFO] Shared state comparison: differences detected between "
                        f"{base['name']} and {other['name']}"
                    )

            with self.registry_lock:
                for s in states:
                    self.shared_registry["nodes"][s["name"]] = s

        except Exception as e:
            print(f"[SharedStateSync] Error comparing states: {e}")

    def _resync_drive_from_registry(self, target_dir):
        """
        When a drive comes back ONLINE, repopulate it with latest states
        from shared_registry so it is up to date.
        """
        if not target_dir or not os.path.exists(target_dir):
            return

        try:
            with self.registry_lock:
                states = list(self.shared_registry["nodes"].values())

            for s in states:
                name = s.get("name", "Unknown")
                fpath = self._state_file_path(target_dir, name)
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(s, f, indent=2)
                except Exception as e:
                    print(f"[SharedStateSync] Error resyncing {name} to {target_dir}: {e}")
        except Exception as e:
            print(f"[SharedStateSync] Resync error: {e}")

    def _check_resync_triggers(self, primary_dir, secondary_dir):
        with self._status_lock:
            p_now = self._primary_status
            p_prev = self._last_primary_status
            s_now = self._secondary_status
            s_prev = self._last_secondary_status

        # Primary: was offline/failing, now online
        if p_now == DRIVE_STATUS_ONLINE and p_prev in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING):
            self.node.alert_queue.put("[INFO] Primary drive came back ONLINE, resyncing from collective.")
            self._resync_drive_from_registry(primary_dir)

        # Secondary: was offline/failing, now online
        if s_now == DRIVE_STATUS_ONLINE and s_prev in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING):
            self.node.alert_queue.put("[INFO] Secondary drive came back ONLINE, resyncing from collective.")
            self._resync_drive_from_registry(secondary_dir)

    def run(self):
        while self.running and self.node.is_running():
            primary_dir = self.get_primary_dir()
            secondary_dir = self.get_secondary_dir()

            used_dir = None
            if primary_dir:
                ok_primary = self._write_state_to_dir(primary_dir, primary=True)
                if ok_primary:
                    used_dir = primary_dir

            if used_dir is None and secondary_dir:
                ok_secondary = self._write_state_to_dir(secondary_dir, primary=False)
                if ok_secondary:
                    used_dir = secondary_dir

            self._check_resync_triggers(primary_dir, secondary_dir)

            if used_dir and self.node.is_queen:
                self._compare_states_as_queen(used_dir)

            time.sleep(10)

    def stop(self):
        self.running = False


# ============================================================
# QUEEN + NODES
# ============================================================

class BorgNode:
    def __init__(
        self,
        is_queen=False,
        name="Node",
        persist_file="borgnet_memory.json",
        get_primary_dir_func=None,
        get_secondary_dir_func=None,
        shared_registry=None
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
        self.shared_sync = SharedStateSync(
            self, get_primary_dir_func, get_secondary_dir_func, shared_registry
        )

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
    def __init__(
        self,
        name="Queen",
        persist_file="borgnet_queen_memory.json",
        get_primary_dir_func=None,
        get_secondary_dir_func=None,
        shared_registry=None
    ):
        super().__init__(
            is_queen=True,
            name=name,
            persist_file=persist_file,
            get_primary_dir_func=get_primary_dir_func,
            get_secondary_dir_func=get_secondary_dir_func,
            shared_registry=shared_registry
        )
        self.drones = []

    def register_drone(self, drone):
        self.drones.append(drone)
        print(f"[{self.name}] Registered node: {drone.name}")


# ============================================================
# TKINTER GUI
# ============================================================

class BorgGUI:
    def __init__(self, root, queen, node_list, shared_dirs_ref, config_ref, shared_registry):
        self.root = root
        self.queen = queen
        self.node_list = node_list
        self.shared_dirs_ref = shared_dirs_ref
        self.config_ref = config_ref
        self.shared_registry = shared_registry
        self.registry_lock = shared_registry["_lock"]

        self.root.title("BorgNet Guardian")
        self.root.geometry("1000x650")

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main)
        top_frame.pack(fill=tk.X, pady=5)

        summary_frame = ttk.LabelFrame(top_frame, text="BorgNet Node Status")
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.node_label = ttk.Label(summary_frame, text="")
        self.node_label.pack(anchor="w")

        self.mode_label = ttk.Label(summary_frame, text="")
        self.mode_label.pack(anchor="w")

        self.gaming_label = ttk.Label(summary_frame, text="")
        self.gaming_label.pack(anchor="w")

        health_frame = ttk.LabelFrame(top_frame, text="Collective Health Score")
        health_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.health_score_var = tk.StringVar(value="N/A")
        self.health_detail_var = tk.StringVar(value="")

        ttk.Label(health_frame, text="Score:").pack(anchor="w")
        self.health_score_label = ttk.Label(health_frame, textvariable=self.health_score_var)
        self.health_score_label.pack(anchor="w")

        ttk.Label(health_frame, text="Summary:").pack(anchor="w")
        self.health_detail_label = ttk.Label(health_frame, textvariable=self.health_detail_var, wraplength=300)
        self.health_detail_label.pack(anchor="w")

        drive_frame = ttk.LabelFrame(main, text="Collective Storage Node")
        drive_frame.pack(fill=tk.X, pady=5)

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

        middle_frame = ttk.Frame(main)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        alerts_frame = ttk.LabelFrame(middle_frame, text="Alerts / Rogue-like Activity")
        alerts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.alerts_text = tk.Text(alerts_frame, height=10, wrap="word")
        self.alerts_text.pack(fill=tk.BOTH, expand=True)

        nodes_frame = ttk.LabelFrame(middle_frame, text="Collective Node States")
        nodes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("name", "role", "conn", "endpoints", "timestamp")
        self.nodes_tree = ttk.Treeview(nodes_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.nodes_tree.heading(col, text=col.capitalize())
        self.nodes_tree.pack(fill=tk.BOTH, expand=True)

        self.borg_banner_var = tk.StringVar(value="")
        self.borg_banner_label = ttk.Label(
            main,
            textvariable=self.borg_banner_var,
            foreground="red",
            font=("TkDefaultFont", 10, "bold")
        )
        self.borg_banner_label.pack(fill=tk.X, pady=5)

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

        self.endpoint_button = ttk.Button(
            controls_frame,
            text="Show Endpoint Details",
            command=self._show_endpoints
        )
        self.endpoint_button.pack(side=tk.LEFT, padx=5)

        self.diff_button = ttk.Button(
            controls_frame,
            text="Compare Queen vs Backup",
            command=self._show_diff
        )
        self.diff_button.pack(side=tk.LEFT, padx=5)

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

    def _show_endpoints(self):
        snap = self.queen.memory.snapshot()
        freq = snap["endpoint_frequency"]
        last_seen = snap["endpoint_last_seen"]
        hours = snap["endpoint_hours"]

        lines = []
        for ep, count in sorted(freq.items(), key=lambda x: x[0]):
            ls = last_seen.get(ep, "unknown")
            hrs = sorted(hours.get(ep, []))
            lines.append(
                f"{ep:25}  count={count:4}  last={ls}  hours={hrs}"
            )

        text = "\n".join(lines) if lines else "No endpoints recorded yet."

        win = tk.Toplevel(self.root)
        win.title("Endpoint Details")
        txt = tk.Text(win, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, text)

    def _show_diff(self):
        with self.shared_registry["_lock"]:
            queen_state = self.shared_registry["nodes"].get("BorgNet-Queen")
            backup_state = self.shared_registry["nodes"].get("BorgNet-Backup")

        if not queen_state or not backup_state:
            messagebox.showinfo(
                "State Comparison",
                "Queen or Backup state not yet available from shared storage."
            )
            return

        q_mem = queen_state["memory_summary"]
        b_mem = backup_state["memory_summary"]

        lines = [
            "=== Queen vs Backup Memory Summary ===",
            "",
            f"Queen:  {queen_state['name']}",
            f"Backup: {backup_state['name']}",
            "",
            f"Visited sites:    Queen={q_mem['visited_sites']}  Backup={b_mem['visited_sites']}",
            f"Connections:      Queen={q_mem['connection_count']}  Backup={b_mem['connection_count']}",
            f"Known endpoints:  Queen={q_mem['known_endpoints']}  Backup={b_mem['known_endpoints']}",
            f"History snapshots:Queen={q_mem['system_history_len']}  Backup={b_mem['system_history_len']}",
        ]

        messagebox.showinfo("State Comparison", "\n".join(lines))

    def _quit(self):
        self.root.quit()

    def _schedule_update(self):
        self._update_gui()
        self.root.after(1000, self._schedule_update)

    def _compute_health(self):
        p_status, s_status = self.queen.shared_sync.get_statuses()
        summary = self.queen.threat_detector.get_summary()

        score = 100
        details = []

        if p_status != DRIVE_STATUS_ONLINE:
            score -= 20
            details.append(f"Primary drive {p_status}")
        if s_status != DRIVE_STATUS_ONLINE:
            score -= 10
            details.append(f"Secondary drive {s_status}")

        if summary["unusual_time_endpoints"] > 0:
            score -= 30
            details.append("High-severity anomalies present")

        if summary["rare_endpoints"] > 0:
            score -= 10
            details.append("Rare endpoints present")

        if summary["new_endpoints"] > 0:
            score -= 5
            details.append("New endpoints observed")

        if score < 0:
            score = 0

        if not details:
            details.append("Collective stable")

        return score, "; ".join(details)

    def _update_nodes_table(self):
        for row in self.nodes_tree.get_children():
            self.nodes_tree.delete(row)

        with self.shared_registry["_lock"]:
            states = list(self.shared_registry["nodes"].values())

        for s in states:
            name = s.get("name", "Unknown")
            role = "Queen" if s.get("is_queen") else "Worker"
            mem = s.get("memory_summary", {})
            conn = mem.get("connection_count", 0)
            endpoints = mem.get("known_endpoints", 0)
            ts = s.get("timestamp", "unknown")

            self.nodes_tree.insert(
                "",
                tk.END,
                values=(name, role, conn, endpoints, ts)
            )

    def _update_gui(self):
        if not self.queen.is_running():
            return

        cfg = self.queen.config.snapshot()
        self.node_label.config(text=f"Node: {self.queen.name} (Main Queen)")
        self.mode_label.config(text=f"Mode: {cfg['mode']}")
        self.gaming_label.config(
            text=f"Gaming Mode: {'ON' if self.queen.gaming_manager.is_enabled() else 'OFF'}"
        )

        primary_path = self.shared_dirs_ref.get("primary") or "(none)"
        secondary_path = self.shared_dirs_ref.get("secondary") or "(none)"
        self.primary_path_var.set(primary_path)
        self.secondary_path_var.set(secondary_path)

        primary_status, secondary_status = self.queen.shared_sync.get_statuses()
        self.primary_status_var.set(primary_status)
        self.secondary_status_var.set(secondary_status)

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

        if self.queen.threat_detector.has_high_severity_alerts():
            self.borg_banner_var.set("WE ARE THE BORG — anomaly detected")
        else:
            self.borg_banner_var.set("")

        score, detail = self._compute_health()
        self.health_score_var.set(str(score))
        self.health_detail_var.set(detail)

        self._update_nodes_table()


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = load_config()
    shared_dirs_ref = {
        "primary": cfg.get("primary_shared_dir", ""),
        "secondary": cfg.get("secondary_shared_dir", "")
    }

    def get_primary_dir():
        return shared_dirs_ref.get("primary", "")

    def get_secondary_dir():
        return shared_dirs_ref.get("secondary", "")

    shared_registry = {
        "_lock": threading.Lock(),
        "nodes": {}
    }

    loader = AutoLoader()
    loader.load_all()

    queen = BorgQueen(
        name="BorgNet-Queen",
        persist_file="borgnet_queen_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry
    )
    queen.start()

    backup_worker = BorgNode(
        is_queen=False,
        name="BorgNet-Backup",
        persist_file="borgnet_backup_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry
    )
    queen.register_drone(backup_worker)
    backup_worker.start()

    queen1 = BorgNode(
        is_queen=True,
        name="BorgNet-Queen-1",
        persist_file="borgnet_queen1_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry
    )
    queen.register_drone(queen1)
    queen1.start()

    queen2 = BorgNode(
        is_queen=True,
        name="BorgNet-Queen-2",
        persist_file="borgnet_queen2_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry
    )
    queen.register_drone(queen2)
    queen2.start()

    node_list = [queen, backup_worker, queen1, queen2]

    root = tk.Tk()
    gui = BorgGUI(root, queen, node_list, shared_dirs_ref, cfg, shared_registry)

    try:
        root.mainloop()
    finally:
        print("\n[Main] Shutting down BorgNet Guardian...")
        for node in node_list:
            node.stop()

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

