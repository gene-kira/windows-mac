import os
import sys
import platform
import threading
import time
import psutil
import queue
import datetime
import json
from collections import defaultdict, deque
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

ANOMALY_FORECAST_WINDOW = 10
DRIVE_TREND_WINDOW = 30
HEALTH_HISTORY_WINDOW = 30
SEQUENCE_WINDOW = 5


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
        self.connection_patterns = []
        self.system_history = []

        self.endpoint_frequency = defaultdict(int)
        self.endpoint_last_seen = {}
        self.endpoint_hours = defaultdict(set)

        self.sequence_window = deque(maxlen=SEQUENCE_WINDOW)
        self.transition_counts = defaultdict(int)

        self._lock = threading.Lock()
        self._load()

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
        self.delta_history = deque(maxlen=50)

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
            self._log_alert(
                "WATCH",
                f"Sequence anomalies detected: {anomalies_in_tick} new rare transition(s)"
            )

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
                    "sequence": self.sequence_anomaly_history[-1]["count"]
                    if self.sequence_anomaly_history else 0
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
# NETWORK OBSERVER / SIMULATION / TIME BRAIN / GAMING
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


class SimulationEngine(threading.Thread):
    def __init__(self, memory_engine, config):
        super().__init__(daemon=True)
        self.memory = memory_engine
        self.config = config
        self.running = True

    def run(self):
        while self.running:
            try:
                _ = self.memory.snapshot()
                time.sleep(5)
            except Exception as e:
                print(f"[SimulationEngine] Error: {e}")
                time.sleep(5)

    def stop(self):
        self.running = False


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

    def snapshot(self):
        with self._lock:
            return dict(self.settings)


# ============================================================
# SHARED STATE SYNC (LOCAL JSON ONLY)
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

            time.sleep(10)

    def stop(self):
        self.running = False


# ============================================================
# PREDICTIVE AI ENGINE (LOCAL ONLY)
# ============================================================

class PredictiveAIEngine(threading.Thread):
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
        self._sensitivity_factor = 1.0

    def _compute_anomaly_prediction(self):
        deltas = self.node.threat_detector.get_delta_history()
        if not deltas:
            return "LOW", "STABLE", "No anomaly activity yet."

        length = max(1, len(deltas))
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
            prev_intensity = sum(
                d["new"] + d["rare"] + d["unusual"] + d.get("sequence", 0)
                for d in prev_half
            ) / len(prev_half)
            later_intensity = sum(
                d["new"] + d["rare"] + d["unusual"] + d.get("sequence", 0)
                for d in later_half
            ) / len(later_half)
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
            forecast = "Drive subsystem shows early warning signs; monitor latency."
        else:
            trend = "STABLE"
            forecast = "Drive subsystem stable based on recent samples."

        return level, trend, forecast, notes

    def _compute_collective_prediction(self, anomaly_level, drive_level):
        def level_to_num(lvl):
            if lvl == "HIGH":
                return 2
            if lvl == "MEDIUM":
                return 1
            if lvl == "LOW":
                return 0
            return 1

        anomaly_val = level_to_num(anomaly_level)
        drive_val = level_to_num(drive_level)

        hive_anomaly_risk = "LOW"
        if anomaly_val >= 2:
            hive_anomaly_risk = "HIGH"
        elif anomaly_val >= 1:
            hive_anomaly_risk = "MEDIUM"

        hive_drive_risk = "LOW"
        if drive_val >= 2:
            hive_drive_risk = "HIGH"
        elif drive_val >= 1:
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

        node_notes = ["Local hive only; no LAN peers integrated."]
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
        self.cfg = cfg or {}

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
        self.predictive_engine.start()

    def stop(self):
        self._running = False
        self.observer.stop()
        self.simulator.stop()
        self.time_brain.stop()
        self.threat_detector.stop()
        self.shared_sync.stop()
        self.predictive_engine.stop()
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
# TKINTER GUI WITH BULLETPROOF C-RED HUD
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

        self._hud_scanline_pos = 0
        self._hud_scanline_direction = 1

        self._build_layout()
        self._schedule_update()
        self._schedule_hud_scanline()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        # ====== COLLECTIVE STORAGE NODE ======
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

        # ====== C-RED MACHINE-VISION HUD (FULL WIDTH, HYBRID) ======
        hud_frame = ttk.LabelFrame(main, text="Machine-Vision HUD")
        hud_frame.pack(fill=tk.X, pady=4)

        hud_top = ttk.Frame(hud_frame)
        hud_top.pack(fill=tk.X, pady=(0, 2))

        mono_font = ("Courier New", 9)

        self.hud_static_var = tk.StringVar(value="HUD initializing...")
        self.hud_static_label = tk.Label(
            hud_top,
            textvariable=self.hud_static_var,
            fg="#FF3B3B",
            bg="black",
            font=mono_font,
            anchor="w",
            justify="left"
        )
        self.hud_static_label.pack(fill=tk.X)

        hud_bottom = ttk.Frame(hud_frame)
        hud_bottom.pack(fill=tk.BOTH, expand=True)

        self.hud_text = tk.Text(
            hud_bottom,
            height=6,
            wrap="none",
            fg="#FF3B3B",
            bg="black",
            insertbackground="#FF3B3B",
            font=mono_font
        )
        hud_scroll = ttk.Scrollbar(hud_bottom, orient="vertical", command=self.hud_text.yview)
        self.hud_text.configure(yscrollcommand=hud_scroll.set)

        self.hud_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hud_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.hud_scanline_canvas = tk.Canvas(
            self.hud_text,
            highlightthickness=0,
            bg="black",   # explicit safe color
            bd=0
        )
        self.hud_scanline_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ====== TOP ROW: Node Status / Health / Predictive ======
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

        # ====== MIDDLE: Alerts + Nodes Table ======
        middle_frame = ttk.Frame(main)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        alerts_frame = ttk.LabelFrame(middle_frame, text="Alerts / Rogue-like Activity")
        alerts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

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
            text="Collective Node States (Local Only)"
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

        self.nodes_tree.column("name", width=120, anchor="w")
        self.nodes_tree.column("role", width=70, anchor="w")
        self.nodes_tree.column("conn", width=60, anchor="center")
        self.nodes_tree.column("endpoints", width=80, anchor="center")
        self.nodes_tree.column("timestamp", width=150, anchor="w")

        nodes_inner = ttk.Frame(nodes_frame)
        nodes_inner.pack(fill=tk.BOTH, expand=True)

        nodes_scroll = ttk.Scrollbar(nodes_inner, orient="vertical")
        nodes_scroll.config(command=self.nodes_tree.yview)
        self.nodes_tree.configure(yscrollcommand=nodes_scroll.set)

        self.nodes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nodes_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ====== COMM STATUS (LOCAL STUB) ======
        comm_frame = ttk.LabelFrame(main, text="BorgComm LAN Status (Local Stub)")
        comm_frame.pack(fill=tk.X, pady=4)

        self.comm_peers_var = tk.StringVar(value="LAN communication not enabled in this build.")
        ttk.Label(comm_frame, textvariable=self.comm_peers_var, wraplength=500).pack(anchor="w")

        # ====== BORG BANNER ======
        self.borg_banner_var = tk.StringVar(value="")
        self.borg_banner_label = ttk.Label(
            main,
            textvariable=self.borg_banner_var,
            foreground="red",
            font=("TkDefaultFont", 10, "bold")
        )
        self.borg_banner_label.pack(fill=tk.X, pady=4)

        # ====== CONTROLS ======
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

    def _quit(self):
        self.root.quit()

    def _schedule_update(self):
        self._update_gui()
        self.root.after(1000, self._schedule_update)

    def _schedule_hud_scanline(self):
        self._update_hud_scanline()
        self.root.after(70, self._schedule_hud_scanline)

    def _update_hud_scanline(self):
        try:
            self.hud_scanline_canvas.delete("all")
            w = self.hud_scanline_canvas.winfo_width()
            h = self.hud_scanline_canvas.winfo_height()
            if w <= 0 or h <= 0:
                return

            step = max(2, h // 20)
            self._hud_scanline_pos += self._hud_scanline_direction * step

            if self._hud_scanline_pos >= h:
                self._hud_scanline_pos = h
                self._hud_scanline_direction = -1
            elif self._hud_scanline_pos <= 0:
                self._hud_scanline_pos = 0
                self._hud_scanline_direction = 1

            y1 = self._hud_scanline_pos
            y2 = min(h, y1 + step * 2)

            self.hud_scanline_canvas.create_rectangle(
                0, y1, w, y2,
                fill="#330000",
                outline=""
            )
        except Exception:
            pass

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
            ts = s.get("timestamp", "unknown")

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

    def _update_hud_panel(self):
        summary = self.queen.threat_detector.get_summary()
        pred = self.queen.predictive_engine.get_summary()
        lat_stats = self.queen.shared_sync.get_drive_latency_stats()
        p_status, s_status = self.queen.shared_sync.get_statuses()

        static_lines = []
        static_lines.append(
            f"ANOM: new={summary['new_endpoints']} "
            f"rare={summary['rare_endpoints']} "
            f"ut={summary['unusual_time_endpoints']} "
            f"seq={summary['sequence_anomalies']}"
        )
        static_lines.append(
            f"PRED: anom={pred.get('anomaly_risk','?')}({pred.get('anomaly_trend','?')}) "
            f"drive={pred.get('drive_risk','?')}({pred.get('drive_trend','?')}) "
            f"hive={pred.get('hive_anomaly_risk','?')}/{pred.get('hive_drive_risk','?')}"
        )
        static_lines.append(
            f"DRIVE: P[{p_status}] avg={lat_stats['primary_avg']:.3f}s "
            f"S[{s_status}] avg={lat_stats['secondary_avg']:.3f}s"
        )
        static_lines.append(
            f"HIVE SCORE={pred.get('collective_risk_score',0)} "
            f"trend={pred.get('health_trend','?')} "
            f"gaming={pred.get('gaming_condition','N/A')}"
        )

        self.hud_static_var.set(" | ".join(static_lines))

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        feed_line = (
            f"[{ts}] anom:new={summary['new_endpoints']} "
            f"rare={summary['rare_endpoints']} "
            f"ut={summary['unusual_time_endpoints']} "
            f"seq={summary['sequence_anomalies']} "
            f"pred={pred.get('anomaly_risk','?')}/{pred.get('drive_risk','?')} "
            f"hive={pred.get('hive_anomaly_risk','?')}/{pred.get('hive_drive_risk','?')}"
        )

        self.hud_text.insert(tk.END, feed_line + "\n")
        max_lines = 200
        current = int(self.hud_text.index("end-1c").split(".")[0])
        if current > max_lines:
            self.hud_text.delete("1.0", f"{current - max_lines}.0")
        self.hud_text.see(tk.END)

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
        self._update_hud_panel()


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = load_config()
    shared_dirs_ref = {
        "primary": cfg.get("primary_shared_dir", ""),
        "secondary": cfg.get("secondary_shared_dir", "")
    }

    save_config(cfg)

    def get_primary_dir():
        return shared_dirs_ref.get("primary", "")

    def get_secondary_dir():
        return shared_dirs_ref.get("secondary", "")

    shared_registry = {
        "_lock": threading.Lock(),
        "nodes": {}
    }

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

    node_list = [queen, backup_worker]

    root = tk.Tk()

    # Force safe default theme to avoid weird system colors
    style = ttk.Style()
    try:
        style.theme_use("default")
    except Exception:
        pass

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


