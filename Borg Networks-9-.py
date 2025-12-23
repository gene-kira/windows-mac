import os
import sys
import platform

# ============================================================
# AUTO-ELEVATION CHECK (Windows only)
# ============================================================

def ensure_admin():
    try:
        if platform.system() == "Windows":
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",
                    sys.executable,
                    f'"{script}" {params}',
                    None,
                    1
                )
                sys.exit()
    except Exception as e:
        print(f"[BorgNet] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# ============================================================
# ORIGINAL IMPORTS
# ============================================================

import importlib
import threading
import time
import psutil
import queue
import datetime
import json
from collections import defaultdict, deque
import socket
import base64
import hashlib
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

DEFAULT_BORG_PORT = 45678
DEFAULT_BORG_KEY = "borg-shared-secret"  # override in config for real use

ANOMALY_FORECAST_WINDOW = 10  # seconds window for short-term anomaly forecast
DRIVE_TREND_WINDOW = 30       # number of samples for drive trend
HEALTH_HISTORY_WINDOW = 30    # predictive health trend history length
SEQUENCE_WINDOW = 5           # length of sequence memory for endpoint sequences


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
# MEMORY ENGINE + SEQUENCE ANALYZER
# ============================================================

class MemoryEngine:
    """
    Stores:
      - endpoints, frequencies, hours, last-seen
      - connection list
      - a rolling sequence window of endpoints for sequence anomaly detection
    """
    def __init__(self, persist_file="borgnet_memory.json"):
        self.persist_file = persist_file
        self.visited_sites = set()
        self.connection_patterns = []  # (laddr, raddr, family, type)
        self.system_history = []

        self.endpoint_frequency = defaultdict(int)
        self.endpoint_last_seen = {}
        self.endpoint_hours = defaultdict(set)

        # For simple sequence-based anomaly detection
        self.sequence_window = deque(maxlen=SEQUENCE_WINDOW)
        self.transition_counts = defaultdict(int)  # (from, to) -> count

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

            # sequence history (simple endpoint sequence)
            self.sequence_window.append(endpoint_key)
            if len(self.sequence_window) >= 2:
                prev = self.sequence_window[-2]
                cur = self.sequence_window[-1]
                self.transition_counts[(prev, cur)] += 1

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
                "transition_counts": {
                    f"{a}|{b}": c for (a, b), c in self.transition_counts.items()
                }
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
# THREAT DETECTOR (EVENT STREAM FOR PREDICTION)
# ============================================================

class ThreatDetector(threading.Thread):
    """
    Detects anomalies and maintains counters. Also emits
    an "event stream" of anomaly deltas for predictive layers.
    """
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

        # For predictive model: per-interval deltas
        self.delta_history = deque(maxlen=50)  # list of dicts with deltas + timestamp

        # Sequence anomaly stream
        self.sequence_anomalies = 0
        self.sequence_anomaly_history = deque(maxlen=50)

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

    def _check_sequence_anomalies(self, snap):
        transition_counts = snap.get("transition_counts", {})
        endpoint_frequency = snap.get("endpoint_frequency", {})
        anomalies_in_tick = 0
        for key, count in transition_counts.items():
            if count == 1:
                a, b = key.split("|", 1)
                if endpoint_frequency.get(a, 0) > 3 and endpoint_frequency.get(b, 0) > 3:
                    anomalies_in_tick += 1
        if anomalies_in_tick > 0:
            with self.stats_lock:
                self.sequence_anomalies += anomalies_in_tick
                self.sequence_anomaly_history.append({
                    "time": datetime.datetime.now().isoformat(),
                    "count": anomalies_in_tick
                })
            self._log_alert("WATCH",
                            f"Sequence anomalies detected: {anomalies_in_tick} new rare transition(s)")

    def run(self):
        while self.running:
            try:
                snap = self.memory.snapshot()
                freq_map = snap["endpoint_frequency"]
                hours_map = snap["endpoint_hours"]
                last_seen_map = snap["endpoint_last_seen"]

                now = datetime.datetime.now()
                current_hour = now.hour

                interval_new = 0
                interval_rare = 0
                interval_unusual = 0

                for endpoint, freq in freq_map.items():
                    hours_list = hours_map.get(endpoint, [])
                    last_seen = last_seen_map.get(endpoint, "unknown")

                    if endpoint not in self.known_endpoints and freq == 1:
                        self.known_endpoints.add(endpoint)
                        interval_new += 1
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
                        interval_rare += 1
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
                        interval_unusual += 1
                        self._log_alert(
                            "ATTENTION",
                            f"Endpoint contacted at unusual time (possible rogue): {endpoint} "
                            f"(normal hours: {sorted(hours_list)}, current hour: {current_hour})"
                        )

                self._check_sequence_anomalies(snap)

                now_ts = datetime.datetime.now().isoformat()
                deltas = {
                    "time": now_ts,
                    "new": interval_new,
                    "rare": interval_rare,
                    "unusual": interval_unusual,
                    "sequence": self.sequence_anomaly_history[-1]["count"] if self.sequence_anomaly_history else 0
                }
                with self.stats_lock:
                    self.delta_history.append(deltas)

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
                "sequence_anomalies": self.sequence_anomalies,
                "total_alerts": len(self.alert_log),
                "recent_alerts": self.alert_log[-10:],
            }

    def has_high_severity_alerts(self):
        with self.stats_lock:
            return self.unusual_time_count > 0

    def get_delta_history(self):
        with self.stats_lock:
            return list(self.delta_history)


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
# SIMULATION ENGINE (placeholder)
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
# SHARED STATE SYNC (DUAL DRIVES + RESYNC + LATENCY)
# ============================================================

class SharedStateSync(threading.Thread):
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

        self._lat_lock = threading.Lock()
        self._primary_latencies = []
        self._secondary_latencies = []

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

    def _record_latency(self, primary, duration):
        with self._lat_lock:
            lst = self._primary_latencies if primary else self._secondary_latencies
            lst.append(duration)
            if len(lst) > DRIVE_TREND_WINDOW:
                del lst[0]

    def get_drive_latency_stats(self):
        with self._lat_lock:
            def stats(lst):
                if not lst:
                    return 0.0, 0.0, 0.0
                avg = sum(lst) / len(lst)
                recent = lst[-1]
                var = sum((x - avg) ** 2 for x in lst) / len(lst)
                return avg, recent, var

            p_avg, p_recent, p_var = stats(self._primary_latencies)
            s_avg, s_recent, s_var = stats(self._secondary_latencies)
            return {
                "primary_avg": p_avg,
                "primary_recent": p_recent,
                "primary_var": p_var,
                "secondary_avg": s_avg,
                "secondary_recent": s_recent,
                "secondary_var": s_var,
            }

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

            start = time.time()

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

            duration = time.time() - start
            self._record_latency(primary, duration)

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

        if p_now == DRIVE_STATUS_ONLINE and p_prev in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING):
            self.node.alert_queue.put("[INFO] Primary drive came back ONLINE, resyncing from collective.")
            self._resync_drive_from_registry(primary_dir)

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
# BORG COMM ENGINE (LAN COMMUNICATION)
# ============================================================

class BorgCommEngine(threading.Thread):
    """
    Simple LAN comms engine:
    - UDP unicast with XOR-style "encryption" (demo-level)
    - Nodes exchange JSON messages:
        - status_update
        - prediction_update
        - alert
    """
    def __init__(self, node_name, cfg, message_handler):
        super().__init__(daemon=True)
        self.node_name = node_name
        self.cfg = cfg
        self.message_handler = message_handler
        self.running = True

        self.port = int(cfg.get("borg_port", DEFAULT_BORG_PORT))
        self.key = cfg.get("borg_key", DEFAULT_BORG_KEY)
        self.key_bytes = hashlib.sha256(self.key.encode("utf-8")).digest()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))

        self.peers = cfg.get("borg_peers", [])  # list of IPs or hostnames
        self._lock = threading.Lock()
        self.last_rx_times = {}

    def _xor_encrypt(self, data: bytes) -> bytes:
        kb = self.key_bytes
        return bytes([b ^ kb[i % len(kb)] for i, b in enumerate(data)])

    def _xor_decrypt(self, data: bytes) -> bytes:
        return self._xor_encrypt(data)

    def send_message(self, msg_type, payload, target="broadcast"):
        msg = {
            "type": msg_type,
            "from": self.node_name,
            "to": target,
            "timestamp": datetime.datetime.now().isoformat(),
            "payload": payload,
        }
        try:
            raw = json.dumps(msg).encode("utf-8")
            enc = self._xor_encrypt(raw)
            blob = base64.b64encode(enc)

            targets = []
            if target == "broadcast":
                targets = self.peers
            else:
                targets = self.peers

            for host in targets:
                try:
                    self.sock.sendto(blob, (host, self.port))
                except Exception as e:
                    print(f"[BorgCommEngine:{self.node_name}] send error to {host}: {e}")
        except Exception as e:
            print(f"[BorgCommEngine:{self.node_name}] send_message error: {e}")

    def run(self):
        while self.running:
            try:
                self.sock.settimeout(1.0)
                try:
                    data, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue

                if not data:
                    continue

                try:
                    enc = base64.b64decode(data)
                    raw = self._xor_decrypt(enc)
                    msg = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue

                src = msg.get("from", "?")
                with self._lock:
                    self.last_rx_times[src] = datetime.datetime.now().isoformat()

                self.message_handler(msg, addr)

            except Exception as e:
                print(f"[BorgCommEngine:{self.node_name}] recv error: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

    def get_peer_status(self):
        with self._lock:
            return dict(self.last_rx_times)


# ============================================================
# PREDICTIVE AI ENGINE (ANOMALY, DRIVE, HIVE-AWARE)
# ============================================================

class PredictiveAIEngine(threading.Thread):
    """
    Enhanced predictive engine:
      - Uses ThreatDetector delta history for anomaly trend & forecast
      - Uses SharedStateSync latency history for drive trend & forecast
      - Uses shared_registry + LAN info for hive-level collective prediction
      - Tracks health score history for trend
      - Performs light self-reflection based on realized anomalies
    """
    def __init__(self, node, shared_registry):
        super().__init__(daemon=True)
        self.node = node
        self.shared_registry = shared_registry
        self.registry_lock = shared_registry["_lock"]
        self.running = True

        self._lock = threading.Lock()
        self._summary = {
            "anomaly_risk": "LOW",
            "anomaly_trend": "STABLE",
            "anomaly_forecast": "Stable",
            "drive_risk": "LOW",
            "drive_trend": "STABLE",
            "drive_forecast": "Stable",
            "collective_risk_score": 100,
            "hive_anomaly_risk": "LOW",
            "hive_drive_risk": "LOW",
            "hive_trend": "STABLE",
            "health_trend": "STABLE",
            "forecast": "",
            "gaming_condition": "OK",
            "notes": []
        }

        self._risk_history = deque(maxlen=HEALTH_HISTORY_WINDOW)
        self._self_reflection_history = deque(maxlen=50)
        self._sensitivity_factor = 1.0

    def _compute_anomaly_prediction(self):
        deltas = self.node.threat_detector.get_delta_history()
        if not deltas:
            return "LOW", "STABLE", "No anomaly activity yet."

        total_new = sum(d["new"] for d in deltas)
        total_rare = sum(d["rare"] for d in deltas)
        total_unusual = sum(d["unusual"] for d in deltas)
        total_seq = sum(d.get("sequence", 0) for d in deltas)

        length = max(1, len(deltas))
        avg_unusual = total_unusual / length
        avg_seq = total_seq / length

        recent = deltas[-min(3, length):]
        recent_new = sum(d["new"] for d in recent) / len(recent)
        recent_rare = sum(d["rare"] for d in recent) / len(recent)
        recent_unusual = sum(d["unusual"] for d in recent) / len(recent)
        recent_seq = sum(d.get("sequence", 0) for d in recent) / len(recent)

        base_score = (
            recent_new * 5 +
            recent_rare * 8 +
            recent_unusual * 12 +
            recent_seq * 10
        ) * self._sensitivity_factor

        if base_score >= 40:
            level = "HIGH"
        elif base_score >= 20:
            level = "MEDIUM"
        else:
            level = "LOW"

        prev_half = deltas[:length // 2]
        later_half = deltas[length // 2:]
        if prev_half and later_half:
            prev_intensity = sum(d["new"] + d["rare"] + d["unusual"] + d.get("sequence", 0) for d in prev_half) / len(prev_half)
            later_intensity = sum(d["new"] + d["rare"] + d["unusual"] + d.get("sequence", 0) for d in later_half) / len(later_half)
            diff = later_intensity - prev_intensity
            if diff > 0.5:
                trend = "RISING"
            elif diff < -0.5:
                trend = "FALLING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        if trend == "RISING":
            forecast = "Anomaly pressure increasing; short-term risk may escalate."
        elif trend == "FALLING":
            forecast = "Anomaly pressure easing; short-term risk may soften."
        else:
            forecast = "Anomaly activity stable in recent window."

        if avg_unusual > 0:
            forecast += " Persistent unusual-time endpoints observed."
        if avg_seq > 0:
            forecast += " Sequence anomalies present."

        return level, trend, forecast.strip()

    def _compute_drive_prediction(self):
        p_status, s_status = self.node.shared_sync.get_statuses()
        lat_stats = self.node.shared_sync.get_drive_latency_stats()

        def drive_score(status, avg, recent, var):
            score = 0
            notes = []
            if status in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING):
                score += 40
                notes.append(f"{status} status.")
            if avg > 0.3:
                score += 20
                notes.append(f"High average latency ({avg:.3f}s).")
            if recent > 0.5:
                score += 15
                notes.append(f"Recent latency spike ({recent:.3f}s).")
            if var > 0.05:
                score += 10
                notes.append(f"High latency variance ({var:.4f}).")
            return score, notes

        p_score, p_notes = drive_score(
            p_status,
            lat_stats["primary_avg"],
            lat_stats["primary_recent"],
            lat_stats["primary_var"]
        )
        s_score, s_notes = drive_score(
            s_status,
            lat_stats["secondary_avg"],
            lat_stats["secondary_recent"],
            lat_stats["secondary_var"]
        )

        total_score = (p_score + s_score) * self._sensitivity_factor
        notes = []
        if p_score > 0:
            notes.append(f"Primary: {'; '.join(p_notes)}")
        if s_score > 0:
            notes.append(f"Secondary: {'; '.join(s_notes)}")

        if total_score >= 60:
            level = "HIGH"
        elif total_score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        if total_score >= 60:
            trend = "RISING"
            forecast = "Drive subsystem under stress; pre-failure behavior likely if trend continues."
        elif total_score >= 30:
            trend = "SLIGHT_RISE"
            forecast = "Drive subsystem shows early warning signs; monitor latency and status closely."
        else:
            trend = "STABLE"
            forecast = "Drive subsystem stable based on recent samples."

        return level, trend, forecast, notes

    def _compute_collective_prediction(self, anomaly_level, drive_level):
        with self.registry_lock:
            states = dict(self.shared_registry["nodes"])

        if not states:
            return 100, "LOW", "LOW", "STABLE", ["No shared state yet; hive appears idle."]

        anomaly_scores = []
        drive_scores = []
        node_notes = []

        def level_to_num(lvl):
            if lvl == "HIGH":
                return 2
            if lvl == "MEDIUM":
                return 1
            if lvl == "LOW":
                return 0
            return 1

        for name, st in states.items():
            pred = st.get("remote_prediction", {}).get("prediction", {})
            if not pred and name == self.node.name:
                pred = self.get_summary()

            if pred:
                anomaly_scores.append(level_to_num(pred.get("anomaly_risk", "LOW")))
                drive_scores.append(level_to_num(pred.get("drive_risk", "LOW")))

        if anomaly_scores:
            avg_anomaly = sum(anomaly_scores) / len(anomaly_scores)
        else:
            avg_anomaly = level_to_num(anomaly_level)

        if drive_scores:
            avg_drive = sum(drive_scores) / len(drive_scores)
        else:
            avg_drive = level_to_num(drive_level)

        hive_anomaly_risk = "LOW"
        if avg_anomaly >= 1.5:
            hive_anomaly_risk = "HIGH"
        elif avg_anomaly >= 0.5:
            hive_anomaly_risk = "MEDIUM"

        hive_drive_risk = "LOW"
        if avg_drive >= 1.5:
            hive_drive_risk = "HIGH"
        elif avg_drive >= 0.5:
            hive_drive_risk = "MEDIUM"

        base_score = 100
        if hive_anomaly_risk == "HIGH":
            base_score -= 35
        elif hive_anomaly_risk == "MEDIUM":
            base_score -= 20

        if hive_drive_risk == "HIGH":
            base_score -= 35
        elif hive_drive_risk == "MEDIUM":
            base_score -= 20

        if base_score < 0:
            base_score = 0

        hive_trend = "STABLE"
        if hive_anomaly_risk == "HIGH" or hive_drive_risk == "HIGH":
            hive_trend = "RISING"
        elif hive_anomaly_risk == "LOW" and hive_drive_risk == "LOW":
            hive_trend = "STABLE"

        if not node_notes:
            node_notes.append("Hive predictions aggregated from local and remote nodes.")

        return base_score, hive_anomaly_risk, hive_drive_risk, hive_trend, node_notes

    def _compute_health_trend(self, collective_score):
        self._risk_history.append(collective_score)
        if len(self._risk_history) < 5:
            return "STABLE"

        first = self._risk_history[0]
        last = self._risk_history[-1]
        delta = last - first

        if delta > 10:
            return "IMPROVING"
        elif delta < -10:
            return "DETERIORATING"
        else:
            return "STABLE"

    def _compute_gaming_condition(self, anomaly_risk, drive_risk):
        cfg = self.node.config.snapshot()
        mode = cfg.get("mode", "balanced")
        if mode != "gaming":
            return "N/A"

        if anomaly_risk == "HIGH" or drive_risk == "HIGH":
            return "POOR"
        if anomaly_risk == "MEDIUM" or drive_risk == "MEDIUM":
            return "DEGRADED"
        return "GOOD"

    def _update_self_reflection(self, anomaly_level):
        summary = self.node.threat_detector.get_summary()
        realized_intensity = (
            summary["new_endpoints"] +
            summary["rare_endpoints"] +
            2 * summary["unusual_time_endpoints"]
        )

        self._self_reflection_history.append({
            "time": datetime.datetime.now().isoformat(),
            "anomaly_level": anomaly_level,
            "realized_intensity": realized_intensity
        })

        recent = list(self._self_reflection_history)[-5:]
        if not recent:
            return

        high_preds = sum(1 for r in recent if r["anomaly_level"] == "HIGH")
        total_realized = sum(r["realized_intensity"] for r in recent)

        if high_preds >= 3 and total_realized < 3:
            self._sensitivity_factor = max(0.7, self._sensitivity_factor - 0.05)
        elif high_preds <= 1 and total_realized > 10:
            self._sensitivity_factor = min(1.3, self._sensitivity_factor + 0.05)

    def run(self):
        while self.running and self.node.is_running():
            try:
                anomaly_level, anomaly_trend, anomaly_forecast = self._compute_anomaly_prediction()
                drive_level, drive_trend, drive_forecast, drive_notes = self._compute_drive_prediction()
                collective_score, hive_anomaly, hive_drive, hive_trend, hive_notes = \
                    self._compute_collective_prediction(anomaly_level, drive_level)
                health_trend = self._compute_health_trend(collective_score)
                gaming_condition = self._compute_gaming_condition(anomaly_level, drive_level)

                global_forecast = anomaly_forecast + " " + drive_forecast

                notes = []
                notes.append(f"Hive anomaly risk: {hive_anomaly}")
                notes.append(f"Hive drive risk: {hive_drive}")
                notes.extend(drive_notes)
                notes.extend(hive_notes)

                self._update_self_reflection(anomaly_level)

                with self._lock:
                    self._summary = {
                        "anomaly_risk": anomaly_level,
                        "anomaly_trend": anomaly_trend,
                        "anomaly_forecast": anomaly_forecast,
                        "drive_risk": drive_level,
                        "drive_trend": drive_trend,
                        "drive_forecast": drive_forecast,
                        "collective_risk_score": collective_score,
                        "hive_anomaly_risk": hive_anomaly,
                        "hive_drive_risk": hive_drive,
                        "hive_trend": hive_trend,
                        "health_trend": health_trend,
                        "forecast": global_forecast.strip(),
                        "gaming_condition": gaming_condition,
                        "notes": notes[:8]
                    }
            except Exception as e:
                with self._lock:
                    self._summary = {
                        "anomaly_risk": "UNKNOWN",
                        "anomaly_trend": "UNKNOWN",
                        "anomaly_forecast": f"Predictive error: {e}",
                        "drive_risk": "UNKNOWN",
                        "drive_trend": "UNKNOWN",
                        "drive_forecast": "UNKNOWN",
                        "collective_risk_score": 0,
                        "hive_anomaly_risk": "UNKNOWN",
                        "hive_drive_risk": "UNKNOWN",
                        "hive_trend": "UNKNOWN",
                        "health_trend": "UNKNOWN",
                        "forecast": f"Predictive engine error: {e}",
                        "gaming_condition": "UNKNOWN",
                        "notes": []
                    }
            time.sleep(10)

    def stop(self):
        self.running = False

    def get_summary(self):
        with self._lock:
            return dict(self._summary)


# ============================================================
# QUEEN + NODES (WITH PREDICTIVE + COMM)
# ============================================================

class BorgNode:
    def __init__(
        self,
        is_queen=False,
        name="Node",
        persist_file="borgnet_memory.json",
        get_primary_dir_func=None,
        get_secondary_dir_func=None,
        shared_registry=None,
        cfg=None
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
        self.predictive_engine = PredictiveAIEngine(self, shared_registry)

        self.shared_registry = shared_registry
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.comm_engine = BorgCommEngine(
            node_name=self.name,
            cfg=self.cfg,
            message_handler=self._handle_borg_message
        )

        self._running = False

    def _handle_borg_message(self, msg, addr):
        try:
            msg_type = msg.get("type")
            src = msg.get("from", "?")
            payload = msg.get("payload", {})

            if src == self.name:
                return

            if msg_type in ("status_update", "prediction_update"):
                with self.shared_registry["_lock"]:
                    node_state = self.shared_registry["nodes"].get(src, {})
                    if msg_type == "status_update":
                        node_state["remote_status"] = payload
                    elif msg_type == "prediction_update":
                        node_state["remote_prediction"] = payload
                    self.shared_registry["nodes"][src] = node_state

            elif msg_type == "alert":
                text = payload.get("message", "")
                if text:
                    self.alert_queue.put(f"[REMOTE:{src}] {text}")

        except Exception as e:
            print(f"[{self.name}] Borg message handler error: {e}")

    def _broadcast_status(self):
        mem_snap = self.memory.snapshot()
        cfg_snap = self.config.snapshot()
        payload = {
            "name": self.name,
            "is_queen": self.is_queen,
            "memory_summary": {
                "visited_sites": len(mem_snap["visited_sites"]),
                "connection_count": mem_snap["connection_count"],
                "known_endpoints": len(mem_snap["endpoint_frequency"]),
                "system_history_len": mem_snap["system_history_len"],
            },
            "config": cfg_snap,
        }
        self.comm_engine.send_message("status_update", payload, target="broadcast")

    def _broadcast_prediction(self):
        pred = self.predictive_engine.get_summary()
        payload = {
            "name": self.name,
            "prediction": pred
        }
        self.comm_engine.send_message("prediction_update", payload, target="broadcast")

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
        self.predictive_engine.start()
        self.comm_engine.start()

        def _broadcaster():
            while self.is_running():
                try:
                    self._broadcast_status()
                    self._broadcast_prediction()
                except Exception as e:
                    print(f"[{self.name}] broadcast error: {e}")
                time.sleep(10)

        self._broadcast_thread = threading.Thread(target=_broadcaster, daemon=True)
        self._broadcast_thread.start()

    def stop(self):
        self._running = False
        self.observer.stop()
        self.simulator.stop()
        self.time_brain.stop()
        self.threat_detector.stop()
        self.shared_sync.stop()
        self.predictive_engine.stop()
        self.comm_engine.stop()
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
        shared_registry=None,
        cfg=None
    ):
        super().__init__(
            is_queen=True,
            name=name,
            persist_file=persist_file,
            get_primary_dir_func=get_primary_dir_func,
            get_secondary_dir_func=get_secondary_dir_func,
            shared_registry=shared_registry,
            cfg=cfg
        )
        self.drones = []

    def register_drone(self, drone):
        self.drones.append(drone)
        print(f"[{self.name}] Registered node: {drone.name}")

# ============================================================
# TKINTER GUI (Light-compressed, 780x780, global fonts, scrollbars)
# ============================================================

class BorgGUI:
    def __init__(self, root, queen, node_list, shared_dirs_ref, config_ref, shared_registry, cfg):
        self.root = root
        self.queen = queen
        self.node_list = node_list
        self.shared_dirs_ref = shared_dirs_ref
        self.config_ref = config_ref
        self.shared_registry = shared_registry
        self.registry_lock = shared_registry["_lock"]
        self.cfg = cfg

        self.root.title("BorgNet Guardian — Predictive Hive")
        self.root.geometry("780x780")

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main)
        top_frame.pack(fill=tk.X, pady=4)

        summary_frame = ttk.LabelFrame(top_frame, text="BorgNet Node Status")
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.node_label = ttk.Label(summary_frame, text="")
        self.node_label.pack(anchor="w")

        self.mode_label = ttk.Label(summary_frame, text="")
        self.mode_label.pack(anchor="w")

        self.gaming_label = ttk.Label(summary_frame, text="")
        self.gaming_label.pack(anchor="w")

        health_frame = ttk.LabelFrame(top_frame, text="Collective Health Score")
        health_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.health_score_var = tk.StringVar(value="N/A")
        self.health_detail_var = tk.StringVar(value="")
        self.health_trend_var = tk.StringVar(value="STABLE")

        ttk.Label(health_frame, text="Score:").pack(anchor="w")
        self.health_score_label = ttk.Label(health_frame, textvariable=self.health_score_var)
        self.health_score_label.pack(anchor="w")

        ttk.Label(health_frame, text="Trend:").pack(anchor="w")
        self.health_trend_label = ttk.Label(health_frame, textvariable=self.health_trend_var)
        self.health_trend_label.pack(anchor="w")

        ttk.Label(health_frame, text="Summary:").pack(anchor="w")
        self.health_detail_label = ttk.Label(
            health_frame,
            textvariable=self.health_detail_var,
            wraplength=220
        )
        self.health_detail_label.pack(anchor="w")

        predictive_frame = ttk.LabelFrame(top_frame, text="Predictive Intelligence")
        predictive_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pred_anomaly_var = tk.StringVar(value="LOW")
        self.pred_anomaly_trend_var = tk.StringVar(value="STABLE")
        self.pred_anomaly_forecast_var = tk.StringVar(value="")

        self.pred_drive_var = tk.StringVar(value="LOW")
        self.pred_drive_trend_var = tk.StringVar(value="STABLE")
        self.pred_drive_forecast_var = tk.StringVar(value="")

        self.pred_collective_var = tk.StringVar(value="100")
        self.pred_hive_anomaly_var = tk.StringVar(value="LOW")
        self.pred_hive_drive_var = tk.StringVar(value="LOW")
        self.pred_hive_trend_var = tk.StringVar(value="STABLE")

        self.pred_forecast_var = tk.StringVar(value="")
        self.pred_gaming_var = tk.StringVar(value="N/A")
        self.pred_notes_var = tk.StringVar(value="")

        ttk.Label(predictive_frame, text="Anomaly Risk:").pack(anchor="w")
        self.pred_anomaly_label = ttk.Label(predictive_frame, textvariable=self.pred_anomaly_var)
        self.pred_anomaly_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Anomaly Trend:").pack(anchor="w")
        self.pred_anomaly_trend_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_anomaly_trend_var
        )
        self.pred_anomaly_trend_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Anomaly Forecast:").pack(anchor="w")
        self.pred_anomaly_forecast_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_anomaly_forecast_var,
            wraplength=220
        )
        self.pred_anomaly_forecast_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Drive Risk:").pack(anchor="w")
        self.pred_drive_label = ttk.Label(predictive_frame, textvariable=self.pred_drive_var)
        self.pred_drive_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Drive Trend:").pack(anchor="w")
        self.pred_drive_trend_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_drive_trend_var
        )
        self.pred_drive_trend_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Drive Forecast:").pack(anchor="w")
        self.pred_drive_forecast_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_drive_forecast_var,
            wraplength=220
        )
        self.pred_drive_forecast_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Collective Risk Score:").pack(anchor="w")
        self.pred_collective_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_collective_var
        )
        self.pred_collective_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Hive Anomaly Risk:").pack(anchor="w")
        self.pred_hive_anomaly_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_hive_anomaly_var
        )
        self.pred_hive_anomaly_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Hive Drive Risk:").pack(anchor="w")
        self.pred_hive_drive_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_hive_drive_var
        )
        self.pred_hive_drive_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Hive Trend:").pack(anchor="w")
        self.pred_hive_trend_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_hive_trend_var
        )
        self.pred_hive_trend_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Global Forecast:").pack(anchor="w")
        self.pred_forecast_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_forecast_var,
            wraplength=220
        )
        self.pred_forecast_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Gaming Condition:").pack(anchor="w")
        self.pred_gaming_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_gaming_var
        )
        self.pred_gaming_label.pack(anchor="w")

        ttk.Label(predictive_frame, text="Notes:").pack(anchor="w")
        self.pred_notes_label = ttk.Label(
            predictive_frame,
            textvariable=self.pred_notes_var,
            wraplength=220
        )
        self.pred_notes_label.pack(anchor="w")

        drive_frame = ttk.LabelFrame(main, text="Collective Storage Node")
        drive_frame.pack(fill=tk.X, pady=4)

        ttk.Label(drive_frame, text="Primary Path:").grid(row=0, column=0, sticky="w")
        self.primary_path_var = tk.StringVar(value="(none)")
        self.primary_status_var = tk.StringVar(value=DRIVE_STATUS_UNKNOWN)

        self.primary_path_label = ttk.Label(drive_frame, textvariable=self.primary_path_var)
        self.primary_path_label.grid(row=0, column=1, sticky="w", padx=4)

        self.primary_button = ttk.Button(
            drive_frame,
            text="Select Primary",
            command=self._select_primary
        )
        self.primary_button.grid(row=0, column=2, padx=4)

        ttk.Label(drive_frame, text="Primary Status:").grid(row=1, column=0, sticky="w")
        self.primary_status_label = ttk.Label(
            drive_frame,
            textvariable=self.primary_status_var
        )
        self.primary_status_label.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(drive_frame, text="Secondary Path:").grid(
            row=2,
            column=0,
            sticky="w",
            pady=(4, 0)
        )
        self.secondary_path_var = tk.StringVar(value="(none)")
        self.secondary_status_var = tk.StringVar(value=DRIVE_STATUS_UNKNOWN)

        self.secondary_path_label = ttk.Label(
            drive_frame,
            textvariable=self.secondary_path_var
        )
        self.secondary_path_label.grid(row=2, column=1, sticky="w", padx=4, pady=(4, 0))

        self.secondary_button = ttk.Button(
            drive_frame,
            text="Select Secondary",
            command=self._select_secondary
        )
        self.secondary_button.grid(row=2, column=2, padx=4, pady=(4, 0))

        ttk.Label(drive_frame, text="Secondary Status:").grid(row=3, column=0, sticky="w")
        self.secondary_status_label = ttk.Label(
            drive_frame,
            textvariable=self.secondary_status_var
        )
        self.secondary_status_label.grid(row=3, column=1, sticky="w", padx=4)

        middle_frame = ttk.Frame(main)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        alerts_frame = ttk.LabelFrame(middle_frame, text="Alerts / Rogue-like Activity")
        alerts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        # Alerts text + vertical scrollbar (always visible)
        alerts_inner = ttk.Frame(alerts_frame)
        alerts_inner.pack(fill=tk.BOTH, expand=True)

        alerts_scroll = ttk.Scrollbar(alerts_inner, orient="vertical")
        self.alerts_text = tk.Text(
            alerts_inner,
            height=10,
            wrap="word",
            yscrollcommand=alerts_scroll.set
        )
        alerts_scroll.config(command=self.alerts_text.yview)

        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        nodes_frame = ttk.LabelFrame(
            middle_frame,
            text="Collective Node States (Local + LAN)"
        )
        nodes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("name", "role", "conn", "endpoints", "timestamp")
        self.nodes_tree = ttk.Treeview(
            nodes_frame,
            columns=columns,
            show="headings",
            height=10
        )
        for col in columns:
            self.nodes_tree.heading(col, text=col.capitalize())

        # Light compression on columns
        self.nodes_tree.column("name", width=120, anchor="w")
        self.nodes_tree.column("role", width=70, anchor="w")
        self.nodes_tree.column("conn", width=60, anchor="center")
        self.nodes_tree.column("endpoints", width=80, anchor="center")
        self.nodes_tree.column("timestamp", width=150, anchor="w")

        # Treeview + vertical scrollbar
        nodes_inner = ttk.Frame(nodes_frame)
        nodes_inner.pack(fill=tk.BOTH, expand=True)

        nodes_scroll = ttk.Scrollbar(nodes_inner, orient="vertical")
        nodes_scroll.config(command=self.nodes_tree.yview)
        self.nodes_tree.configure(yscrollcommand=nodes_scroll.set)

        self.nodes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nodes_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        comm_frame = ttk.LabelFrame(main, text="BorgComm LAN Status")
        comm_frame.pack(fill=tk.X, pady=4)

        self.comm_peers_var = tk.StringVar(value="")
        ttk.Label(comm_frame, text="Last messages from peers:").pack(anchor="w")
        self.comm_peers_label = ttk.Label(
            comm_frame,
            textvariable=self.comm_peers_var,
            wraplength=500
        )
        self.comm_peers_label.pack(anchor="w")

        self.borg_banner_var = tk.StringVar(value="")
        self.borg_banner_label = ttk.Label(
            main,
            textvariable=self.borg_banner_var,
            foreground="red",
            font=("TkDefaultFont", 10, "bold")
        )
        self.borg_banner_label.pack(fill=tk.X, pady=4)

        controls_frame = ttk.Frame(main)
        controls_frame.pack(fill=tk.X, pady=4)

        self.gaming_button = ttk.Button(
            controls_frame,
            text="Toggle Gaming Mode",
            command=self._toggle_gaming_mode
        )
        self.gaming_button.pack(side=tk.LEFT, padx=4)

        self.report_button = ttk.Button(
            controls_frame,
            text="Show Anomaly Report",
            command=self._show_report
        )
        self.report_button.pack(side=tk.LEFT, padx=4)

        self.endpoint_button = ttk.Button(
            controls_frame,
            text="Show Endpoint Details",
            command=self._show_endpoints
        )
        self.endpoint_button.pack(side=tk.LEFT, padx=4)

        self.diff_button = ttk.Button(
            controls_frame,
            text="Compare Queen vs Backup",
            command=self._show_diff
        )
        self.diff_button.pack(side=tk.LEFT, padx=4)

        self.quit_button = ttk.Button(
            controls_frame,
            text="Quit",
            command=self._quit
        )
        self.quit_button.pack(side=tk.RIGHT, padx=4)

    def _select_primary(self):
        path = filedialog.askdirectory(
            title="Select Primary BorgNet Shared Network Drive"
        )
        if path:
            self.shared_dirs_ref["primary"] = path
            self.primary_path_var.set(path)
            self.config_ref["primary_shared_dir"] = path
            save_config(self.config_ref)

    def _select_secondary(self):
        path = filedialog.askdirectory(
            title="Select Secondary BorgNet Shared Network Drive"
        )
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
            f"Sequence anomalies:     {summary['sequence_anomalies']}",
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

        if summary["sequence_anomalies"] > 0:
            score -= 20
            details.append("Sequence anomalies present")

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
            states = dict(self.shared_registry["nodes"])

        for s in states.values():
            name = s.get("name", "Unknown")
            role = "Queen" if s.get("is_queen") else "Worker"
            mem = s.get("memory_summary", {})
            conn = mem.get("connection_count", 0)
            endpoints = mem.get("known_endpoints", 0)
            ts = s.get("timestamp", s.get("remote_status", {}).get("timestamp", "unknown"))

            self.nodes_tree.insert(
                "",
                tk.END,
                values=(name, role, conn, endpoints, ts)
            )

    def _update_predictive_panel(self):
        summary = self.queen.predictive_engine.get_summary()
        self.pred_anomaly_var.set(summary.get("anomaly_risk", "UNKNOWN"))
        self.pred_anomaly_trend_var.set(summary.get("anomaly_trend", "UNKNOWN"))
        self.pred_anomaly_forecast_var.set(summary.get("anomaly_forecast", ""))

        self.pred_drive_var.set(summary.get("drive_risk", "UNKNOWN"))
        self.pred_drive_trend_var.set(summary.get("drive_trend", "UNKNOWN"))
        self.pred_drive_forecast_var.set(summary.get("drive_forecast", ""))

        self.pred_collective_var.set(str(summary.get("collective_risk_score", 0)))
        self.pred_hive_anomaly_var.set(summary.get("hive_anomaly_risk", "UNKNOWN"))
        self.pred_hive_drive_var.set(summary.get("hive_drive_risk", "UNKNOWN"))
        self.pred_hive_trend_var.set(summary.get("hive_trend", "UNKNOWN"))

        self.pred_forecast_var.set(summary.get("forecast", ""))
        self.pred_gaming_var.set(summary.get("gaming_condition", "N/A"))
        notes = summary.get("notes", [])
        self.pred_notes_var.set(" | ".join(notes))

        def color_for_risk(level):
            if level == "HIGH":
                return "red"
            if level == "MEDIUM":
                return "orange"
            if level == "LOW":
                return "green"
            return "black"

        self.pred_anomaly_label.config(
            foreground=color_for_risk(self.pred_anomaly_var.get())
        )
        self.pred_drive_label.config(
            foreground=color_for_risk(self.pred_drive_var.get())
        )

    def _update_comm_panel(self):
        peer_status = self.queen.comm_engine.get_peer_status()
        if not peer_status:
            self.comm_peers_var.set("No BorgComm peers heard yet.")
            return
        parts = [f"{name}: {ts}" for name, ts in peer_status.items()]
        self.comm_peers_var.set(" | ".join(parts))

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

        pred = self.queen.predictive_engine.get_summary()
        self.health_trend_var.set(pred.get("health_trend", "STABLE"))
        self.health_detail_var.set(detail)

        self._update_nodes_table()
        self._update_predictive_panel()
        self._update_comm_panel()


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = load_config()
    shared_dirs_ref = {
        "primary": cfg.get("primary_shared_dir", ""),
        "secondary": cfg.get("secondary_shared_dir", "")
    }

    if "borg_peers" not in cfg:
        cfg["borg_peers"] = ["127.0.0.1"]
    if "borg_port" not in cfg:
        cfg["borg_port"] = DEFAULT_BORG_PORT
    if "borg_key" not in cfg:
        cfg["borg_key"] = DEFAULT_BORG_KEY
    save_config(cfg)

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
        shared_registry=shared_registry,
        cfg=cfg
    )
    queen.start()

    backup_worker = BorgNode(
        is_queen=False,
        name="BorgNet-Backup",
        persist_file="borgnet_backup_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry,
        cfg=cfg
    )
    queen.register_drone(backup_worker)
    backup_worker.start()

    queen1 = BorgNode(
        is_queen=True,
        name="BorgNet-Queen-1",
        persist_file="borgnet_queen1_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry,
        cfg=cfg
    )
    queen.register_drone(queen1)
    queen1.start()

    queen2 = BorgNode(
        is_queen=True,
        name="BorgNet-Queen-2",
        persist_file="borgnet_queen2_memory.json",
        get_primary_dir_func=get_primary_dir,
        get_secondary_dir_func=get_secondary_dir,
        shared_registry=shared_registry,
        cfg=cfg
    )
    queen.register_drone(queen2)
    queen2.start()

    node_list = [queen, backup_worker, queen1, queen2]

    root = tk.Tk()

    # Global font scaling (F1)
    root.option_add("*Font", "TkDefaultFont 9")
    root.option_add("*Label.Font", "TkDefaultFont 9")
    root.option_add("*Button.Font", "TkDefaultFont 9")
    root.option_add("*Entry.Font", "TkDefaultFont 9")
    root.option_add("*Text.Font", "TkDefaultFont 9")
    root.option_add("*Treeview.Font", "TkDefaultFont 9")
    root.option_add("*Treeview.Heading.Font", "TkDefaultFont 10")
    root.option_add("*Labelframe.Label.Font", "TkDefaultFont 10")

    gui = BorgGUI(root, queen, node_list, shared_dirs_ref, cfg, shared_registry, cfg)

    try:
        root.mainloop()
    finally:
        print("\n[Main] Shutting down BorgNet Guardian — Predictive Hive...")
        for node in node_list:
            node.stop()

        summary = queen.threat_detector.get_summary()
        print("\n=== Final Anomaly / Rogue-like Activity Summary ===")
        print(f"New endpoints:         {summary['new_endpoints']}")
        print(f"Rare endpoints:        {summary['rare_endpoints']}")
        print(f"Unusual-time endpoints:{summary['unusual_time_endpoints']}")
        print(f"Sequence anomalies:    {summary['sequence_anomalies']}")
        print(f"Total alerts:          {summary['total_alerts']}")
        print("===================================================")
        time.sleep(1)
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()

