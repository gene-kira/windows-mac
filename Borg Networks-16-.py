import os
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

CONFIG_FILE = "borgnet_config.json"

DRIVE_STATUS_ONLINE = "ONLINE"
DRIVE_STATUS_OFFLINE = "OFFLINE"
DRIVE_STATUS_FAILING = "FAILING"
DRIVE_STATUS_UNKNOWN = "UNKNOWN"

DRIVE_TREND_WINDOW = 30
HEALTH_HISTORY_WINDOW = 30
SEQUENCE_WINDOW = 5


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except:
        pass


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
                "endpoint_hours": {ep: list(h) for ep, h in self.endpoint_hours.items()},
                "transition_counts": {f"{a}|{b}": c for (a, b), c in self.transition_counts.items()}
            }

    def _load(self):
        if not os.path.exists(self.persist_file):
            return
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.visited_sites = set(data.get("visited_sites", []))
            self.endpoint_frequency = defaultdict(int, data.get("endpoint_frequency", {}))
        except:
            pass

    def save(self):
        try:
            snap = self.snapshot()
            data = {
                "visited_sites": snap["visited_sites"],
                "endpoint_frequency": snap["endpoint_frequency"],
            }
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except:
            pass


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
        anomalies = 0
        for key, count in transition_counts.items():
            if count == 1:
                a, b = key.split("|", 1)
                if endpoint_frequency.get(a, 0) > 3 and endpoint_frequency.get(b, 0) > 3:
                    anomalies += 1
        if anomalies > 0:
            with self.stats_lock:
                self.sequence_anomalies += anomalies
                self.sequence_anomaly_history.append({
                    "time": datetime.datetime.now().isoformat(),
                    "count": anomalies
                })
            self._log_alert("WATCH", f"Sequence anomalies detected: {anomalies}")

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
                        self._log_alert("INFO", f"New endpoint observed: {endpoint} ({last_seen})")
                        continue
                    if endpoint not in self.rare_endpoints_flagged and 1 < freq <= self.rare_threshold:
                        self.rare_endpoints_flagged.add(endpoint)
                        interval_rare += 1
                        self._log_alert("WATCH", f"Rare endpoint: {endpoint} ({freq})")
                    if endpoint not in self.unusual_time_flagged and freq >= self.min_baseline_for_time:
                        if current_hour not in hours_list:
                            self.unusual_time_flagged.add(endpoint)
                            interval_unusual += 1
                            self._log_alert("ATTENTION", f"Unusual time: {endpoint}")
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
            except:
                pass
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
            except:
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
            except:
                time.sleep(5)

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


class AdaptiveGuesserB2:
    def __init__(self):
        self.bias = 0.0
        self.confidence = 0.5
        self.history = deque(maxlen=20)
        self.learning_rate = 0.35
        self.correction_rate = 0.55

    def guess(self, signal_strength):
        weighted = signal_strength + self.bias
        threshold = max(0.1, 1.0 - self.confidence)
        decision = weighted >= threshold
        self.history.append(("guess", decision, weighted, threshold))
        return decision

    def update(self, was_correct):
        if was_correct:
            self.bias += self.learning_rate
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.bias -= self.correction_rate
            self.confidence = max(0.1, self.confidence - 0.15)
        self.bias = max(-2.0, min(2.0, self.bias))
        self.history.append(("update", was_correct, self.bias, self.confidence))


class HybridAdaptiveEngine:
    def __init__(self):
        self.mode = "stability"
        self.volatility = 0.0
        self.trust = 0.5
        self._history = deque(maxlen=30)

    def update_from_anomaly_deltas(self, deltas):
        if not deltas:
            self.volatility = 0.0
            return
        recent = deltas[-min(5, len(deltas)):]
        intensity_values = [
            d["new"] + d["rare"] + d["unusual"] + d.get("sequence", 0)
            for d in recent
        ]
        avg_intensity = sum(intensity_values) / len(intensity_values)
        if len(intensity_values) > 1:
            diffs = [
                abs(intensity_values[i] - intensity_values[i - 1])
                for i in range(1, len(intensity_values))
            ]
            avg_diff = sum(diffs) / len(diffs)
        else:
            avg_diff = 0.0
        self.volatility = max(0.0, min(1.0, (avg_intensity + avg_diff) / 10.0))
        self._history.append(("anomaly", self.volatility))

    def update_trust(self, last_guess_correct):
        delta = 0.1 if last_guess_correct else -0.15
        self.trust = max(0.0, min(1.0, self.trust + delta))
        self._history.append(("trust", self.trust, last_guess_correct))

    def choose_mode(self):
        if self.volatility > 0.4:
            self.mode = "reflex"
        elif self.volatility < 0.2:
            self.mode = "stability"

    def get_tuning(self):
        if self.mode == "reflex":
            return {
                "sensitivity_factor": 1.3,
                "anomaly_learning_rate": 0.45,
                "anomaly_correction_rate": 0.65,
                "drive_learning_rate": 0.45,
                "drive_correction_rate": 0.65,
            }
        else:
            return {
                "sensitivity_factor": 0.8,
                "anomaly_learning_rate": 0.20,
                "anomaly_correction_rate": 0.35,
                "drive_learning_rate": 0.20,
                "drive_correction_rate": 0.35,
            }


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
            "mode": "stability",
            "volatility": 0.0,
            "trust": 0.5,
            "notes": []
        }
        self._risk_history = deque(maxlen=HEALTH_HISTORY_WINDOW)
        self._sensitivity_factor = 1.0
        self.anomaly_guesser = AdaptiveGuesserB2()
        self.drive_guesser = AdaptiveGuesserB2()
        self.hybrid = HybridAdaptiveEngine()

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
        signal = min(1.0, base_score / 40.0)
        decision_high = self.anomaly_guesser.guess(signal)
        if decision_high and level != "HIGH":
            level = "HIGH"
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
        total_unusual = sum(d["unusual"] for d in deltas)
        total_seq = sum(d.get("sequence", 0) for d in deltas)
        actual_high = (total_unusual > 0 or total_seq > 0)
        self.anomaly_guesser.update(actual_high)
        self.hybrid.update_trust(actual_high)
        return level, trend, forecast.strip()

    def _compute_drive_prediction(self):
        p_status, s_status = self.node.shared_sync.get_statuses()
        lat = self.node.shared_sync.get_drive_latency_stats()
        def score(status, avg, recent, var):
            sc = 0
            notes = []
            if status in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING):
                sc += 40
                notes.append(status)
            if avg > 0.3:
                sc += 20
                notes.append(f"avg {avg:.3f}")
            if recent > 0.5:
                sc += 15
                notes.append(f"recent {recent:.3f}")
            if var > 0.05:
                sc += 10
                notes.append(f"var {var:.4f}")
            return sc, notes
        p_score, p_notes = score(
            p_status,
            lat["primary_avg"],
            lat["primary_recent"],
            lat["primary_var"]
        )
        s_score, s_notes = score(
            s_status,
            lat["secondary_avg"],
            lat["secondary_recent"],
            lat["secondary_var"]
        )
        total = (p_score + s_score) * self._sensitivity_factor
        notes = []
        if p_score > 0:
            notes.append("Primary: " + "; ".join(p_notes))
        if s_score > 0:
            notes.append("Secondary: " + "; ".join(s_notes))
        if total >= 60:
            level = "HIGH"
        elif total >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"
        signal = min(1.0, total / 60.0)
        decision_high = self.drive_guesser.guess(signal)
        if decision_high and level != "HIGH":
            level = "HIGH"
        if total >= 60:
            trend = "RISING"
            forecast = "Drive subsystem under stress; pre-failure behavior likely."
        elif total >= 30:
            trend = "SLIGHT_RISE"
            forecast = "Drive subsystem shows early warning signs."
        else:
            trend = "STABLE"
            forecast = "Drive subsystem stable."
        actual_high = (
            p_status in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING) or
            s_status in (DRIVE_STATUS_OFFLINE, DRIVE_STATUS_FAILING) or
            total >= 60
        )
        self.drive_guesser.update(actual_high)
        self.hybrid.update_trust(actual_high)
        return level, trend, forecast, notes

    def _compute_collective_prediction(self, anomaly_level, drive_level):
        with self.registry_lock:
            nodes = list(self.shared_registry["nodes"].values())
        if not nodes:
            score = 100
            hive_anomaly = anomaly_level
            hive_drive = drive_level
            hive_trend = "STABLE"
            hive_notes = []
            return score, hive_anomaly, hive_drive, hive_trend, hive_notes
        anomaly_vals = []
        drive_vals = []
        for n in nodes:
            mem = n.get("memory_summary", {})
            c = mem.get("connection_count", 0)
            e = mem.get("known_endpoints", 0)
            if c > 200 or e > 150:
                anomaly_vals.append(2)
            elif c > 100 or e > 80:
                anomaly_vals.append(1)
            else:
                anomaly_vals.append(0)
            if c > 200:
                drive_vals.append(2)
            elif c > 100:
                drive_vals.append(1)
            else:
                drive_vals.append(0)
        avg_anom = sum(anomaly_vals) / len(anomaly_vals)
        avg_drive = sum(drive_vals) / len(drive_vals)
        hive_anomaly = "HIGH" if avg_anom >= 1.5 else "MEDIUM" if avg_anom >= 0.5 else "LOW"
        hive_drive = "HIGH" if avg_drive >= 1.5 else "MEDIUM" if avg_drive >= 0.5 else "LOW"
        score = 100 - int((avg_anom * 20) + (avg_drive * 20))
        score = max(0, min(100, score))
        if len(self._risk_history) > 1:
            prev = self._risk_history[-1]
            diff = score - prev
            if diff > 5:
                hive_trend = "IMPROVING"
            elif diff < -5:
                hive_trend = "WORSENING"
            else:
                hive_trend = "STABLE"
        else:
            hive_trend = "STABLE"
        self._risk_history.append(score)
        hive_notes = []
        if hive_anomaly != "LOW":
            hive_notes.append(f"Hive anomaly={hive_anomaly}")
        if hive_drive != "LOW":
            hive_notes.append(f"Hive drive={hive_drive}")
        return score, hive_anomaly, hive_drive, hive_trend, hive_notes

    def _compute_health_trend(self, score):
        if len(self._risk_history) < 2:
            return "STABLE"
        prev = self._risk_history[-2]
        if score > prev + 5:
            return "IMPROVING"
        if score < prev - 5:
            return "DECLINING"
        return "STABLE"

    def _compute_gaming_condition(self, anomaly_risk, drive_risk):
        mode = self.node.config.settings["mode"]
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
                deltas = self.node.threat_detector.get_delta_history()
                self.hybrid.update_from_anomaly_deltas(deltas)
                self.hybrid.choose_mode()
                tuning = self.hybrid.get_tuning()
                self._sensitivity_factor = tuning["sensitivity_factor"]
                self.anomaly_guesser.learning_rate = tuning["anomaly_learning_rate"]
                self.anomaly_guesser.correction_rate = tuning["anomaly_correction_rate"]
                self.drive_guesser.learning_rate = tuning["drive_learning_rate"]
                self.drive_guesser.correction_rate = tuning["drive_correction_rate"]
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
                        "mode": self.hybrid.mode,
                        "volatility": self.hybrid.volatility,
                        "trust": self.hybrid.trust,
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
                        "mode": "stability",
                        "volatility": 0.0,
                        "trust": 0.0,
                        "notes": []
                    }
            time.sleep(10)

    def stop(self):
        self.running = False

    def get_summary(self):
        with self._lock:
            return dict(self._summary)


class GameAwarenessEngine(threading.Thread):
    def __init__(self, node, game_process_names=None):
        super().__init__(daemon=True)
        self.node = node
        self.running = True
        self.game_process_names = game_process_names or ["Back4Blood", "back4blood", "back4blood.exe"]
        self._lock = threading.Lock()
        self.mode = "OFFLINE"
        self.total_active_seconds = 0.0
        self.last_check_time = time.time()
        self.learning_confidence = 0.0
        self.progress = 0.0
        self._activity_bias = 0.0
        self._activity_confidence = 0.5
        self._learning_threshold_seconds = 60.0 * 30.0

    def _detect_game_process(self):
        try:
            for proc in psutil.process_iter(["name"]):
                name = (proc.info.get("name") or "").lower()
                for target in self.game_process_names:
                    if target.lower() in name:
                        return True
        except:
            pass
        return False

    def _guess_game_active(self, raw_detected):
        signal = 1.0 if raw_detected else 0.0
        weighted = signal + self._activity_bias
        threshold = max(0.1, 1.0 - self._activity_confidence)
        decision = (weighted >= threshold)
        was_correct = (decision == raw_detected)
        if was_correct:
            self._activity_bias += 0.2
            self._activity_confidence = min(1.0, self._activity_confidence + 0.05)
        else:
            self._activity_bias -= 0.3
            self._activity_confidence = max(0.1, self._activity_confidence - 0.07)
        self._activity_bias = max(-2.0, min(2.0, self._activity_bias))
        return decision

    def _update_learning_state(self, game_active, dt):
        if game_active:
            self.total_active_seconds += dt
        try:
            j_snap = self.node.judgment_engine.snapshot()
            j_conf = j_snap.get("judgment_confidence", 0.0)
        except:
            j_conf = 0.0
        base_progress = min(1.0, self.total_active_seconds / self._learning_threshold_seconds)
        self.progress = max(0.0, min(1.0, 0.5 * base_progress + 0.5 * j_conf))
        self.learning_confidence = self.progress
        if not game_active:
            if self.learning_confidence >= 0.95:
                self.mode = "MISSION_ACCOMPLISHED"
            else:
                self.mode = "OFFLINE"
        else:
            if self.learning_confidence >= 0.95:
                self.mode = "MISSION_ACCOMPLISHED"
            else:
                self.mode = "LEARNING"

    def run(self):
        while self.running and self.node.is_running():
            now = time.time()
            dt = now - self.last_check_time
            self.last_check_time = now
            raw_detected = self._detect_game_process()
            game_active = self._guess_game_active(raw_detected)
            self._update_learning_state(game_active, dt)
            time.sleep(5.0)

    def stop(self):
        self.running = False

    def snapshot(self):
        with self._lock:
            return {
                "mode": self.mode,
                "total_active_seconds": self.total_active_seconds,
                "learning_confidence": self.learning_confidence,
                "progress": self.progress,
                "activity_confidence": self._activity_confidence,
            }


class JudgmentEngine(threading.Thread):
    def __init__(self, node):
        super().__init__(daemon=True)
        self.node = node
        self.running = True
        self._lock = threading.Lock()
        self.learning_from_player = True
        self.learning_from_bots = True
        self.judgment_confidence = 0.0
        self.sample_count = 0
        self.good_outcome_count = 0
        self.bad_outcome_count = 0
        self._episodes = []
        self._input_buffer = deque(maxlen=500)
        self._bot_event_buffer = deque(maxlen=500)

    def record_player_input(self, event):
        with self._lock:
            self._input_buffer.append(event)

    def record_bot_event(self, event):
        with self._lock:
            self._bot_event_buffer.append(event)

    def snapshot(self):
        with self._lock:
            return {
                "judgment_confidence": self.judgment_confidence,
                "sample_count": self.sample_count,
                "good_outcome_count": self.good_outcome_count,
                "bad_outcome_count": self.bad_outcome_count,
                "learning_from_player": self.learning_from_player,
                "learning_from_bots": self.learning_from_bots,
            }

    def _build_episode_from_buffer(self):
        with self._lock:
            events = list(self._input_buffer)
        if len(events) < 10:
            return None
        times = [e.get("timestamp", 0) for e in events]
        t_min, t_max = min(times), max(times)
        dt = max(0.001, t_max - t_min)
        intensity = len(events) / dt
        if intensity < 1.0:
            return None
        episode = {
            "start_time": t_min,
            "end_time": t_max,
            "intensity": intensity,
            "actions_count": len(events),
            "outcome": "UNKNOWN",
        }
        return episode

    def _evaluate_episode_outcome(self, episode):
        import random
        if episode["intensity"] < 3.0:
            prob_good = 0.7
        elif episode["intensity"] < 6.0:
            prob_good = 0.5
        else:
            prob_good = 0.3
        episode["outcome"] = "GOOD" if random.random() < prob_good else "BAD"
        return episode["outcome"]

    def _update_judgment_from_episode(self, episode):
        outcome = episode["outcome"]
        with self._lock:
            self._episodes.append(episode)
            self.sample_count += 1
            if outcome == "GOOD":
                self.good_outcome_count += 1
            elif outcome == "BAD":
                self.bad_outcome_count += 1
            total = max(1, self.good_outcome_count + self.bad_outcome_count)
            imbalance = abs(self.good_outcome_count - self.bad_outcome_count) / total
            size_factor = min(1.0, self.sample_count / 100.0)
            self.judgment_confidence = max(
                0.0, min(1.0, 0.3 * imbalance + 0.7 * size_factor)
            )

    def run(self):
        while self.running and self.node.is_running():
            try:
                episode = self._build_episode_from_buffer()
                if episode is not None:
                    self._evaluate_episode_outcome(episode)
                    self._update_judgment_from_episode(episode)
                time.sleep(5.0)
            except:
                time.sleep(5.0)

    def stop(self):
        self.running = False


class SituationalAwarenessEngine(threading.Thread):
    def __init__(self, node):
        super().__init__(daemon=True)
        self.node = node
        self.running = True
        self._lock = threading.Lock()
        self.current_mission = "STABILITY"
        self.mission_weights = {"PROTECT": 0.0, "STABILITY": 0.0, "LEARN": 0.0, "OPTIMIZE": 0.0}
        self.environment = "CALM"
        self.unexpected_event = "None"
        self.event_severity = 0
        self.opportunity_score = 0.0
        self.risk_score = 0.0
        self.plan_adjustment = "None"
        self.guessing_factor = 0.0
        self.outcome_confidence = 0.0
        self.anticipation = "None"
        self.short_history = deque(maxlen=12)
        self.mid_history = deque(maxlen=60)
        self.long_history = deque(maxlen=720)
        self._last_threat_level = 0.0
        self._last_stability_risk = 0.0
        self._last_env = "CALM"
        self._decision_history = deque(maxlen=50)

    def _risk_level_to_num(self, lvl):
        if lvl == "HIGH":
            return 2
        if lvl == "MEDIUM":
            return 1
        if lvl == "LOW":
            return 0
        return 1

    def _compute_situation_vector(self):
        th = self.node.threat_detector.get_summary()
        pred = self.node.predictive_engine.get_summary()
        ga = self.node.game_awareness.snapshot()
        js = self.node.judgment_engine.snapshot()
        lat = self.node.shared_sync.get_drive_latency_stats()
        p_status, s_status = self.node.shared_sync.get_statuses()

        unusual = th.get("unusual_time_endpoints", 0)
        seq = th.get("sequence_anomalies", 0)
        anomaly_risk_num = self._risk_level_to_num(pred.get("anomaly_risk", "LOW"))
        threat_level = min(1.0, (unusual * 0.2 + seq * 0.3 + anomaly_risk_num * 0.3))

        volatility = pred.get("volatility", 0.0)
        drive_risk_num = self._risk_level_to_num(pred.get("drive_risk", "LOW"))
        drive_fail = 1 if (p_status in (DRIVE_STATUS_FAILING, DRIVE_STATUS_OFFLINE) or
                           s_status in (DRIVE_STATUS_FAILING, DRIVE_STATUS_OFFLINE)) else 0
        stability_risk = min(1.0, volatility * 0.5 + drive_risk_num * 0.3 + drive_fail * 0.6)

        game_mode = ga.get("mode", "OFFLINE")
        game_prog = ga.get("progress", 0.0)
        j_conf = js.get("judgment_confidence", 0.0)

        learning_opportunity = 0.0
        if game_mode in ("LEARNING", "MISSION_ACCOMPLISHED"):
            learning_opportunity += game_prog * 0.6 + (1.0 - j_conf) * 0.4
        learning_opportunity *= max(0.0, 1.0 - threat_level)

        collective = pred.get("collective_risk_score", 100)
        eff_base = max(0.0, (collective - 60) / 40.0)
        efficiency_pressure = eff_base * max(0.0, 1.0 - threat_level) * max(0.0, 1.0 - stability_risk)

        snapshot = {
            "time": datetime.datetime.now().isoformat(),
            "threat": threat_level,
            "stability": stability_risk,
            "learning": learning_opportunity,
            "efficiency": efficiency_pressure,
            "volatility": volatility,
            "anomaly_risk_num": anomaly_risk_num,
            "drive_risk_num": drive_risk_num,
        }

        self.short_history.append(snapshot)
        self.mid_history.append(snapshot)
        self.long_history.append(snapshot)

        return {
            "threat_level": threat_level,
            "stability_risk": stability_risk,
            "learning_opportunity": learning_opportunity,
            "efficiency_pressure": efficiency_pressure,
            "volatility": volatility,
            "anomaly_risk_num": anomaly_risk_num,
            "drive_risk_num": drive_risk_num,
        }

    def _detect_environment(self, vec):
        t = vec["threat_level"]
        s = vec["stability_risk"]
        if t > 0.6 or s > 0.6:
            return "DANGER"
        if t > 0.3 or s > 0.3:
            return "TENSE"
        return "CALM"

    def _detect_unexpected_event(self, vec):
        events = []
        if vec["threat_level"] - self._last_threat_level > 0.4:
            events.append(("Threat spike", 80))
        if vec["stability_risk"] - self._last_stability_risk > 0.4:
            events.append(("Stability collapse", 85))
        if vec["learning_opportunity"] > 0.7 and vec["threat_level"] < 0.2:
            events.append(("Learning window", 40))
        if vec["efficiency_pressure"] > 0.7 and vec["stability_risk"] < 0.3:
            events.append(("Optimization window", 35))
        if self.node.game_awareness.snapshot().get("mode") == "LEARNING":
            events.append(("Game learning active", 40))
        if not events:
            return "None", 0
        ev = max(events, key=lambda x: x[1])
        return ev

    def _compute_opportunity(self, vec):
        base = vec["learning_opportunity"] * 0.6 + vec["efficiency_pressure"] * 0.4
        if self.environment == "CALM":
            base *= 1.2
        if self.environment == "DANGER":
            base *= 0.5
        return min(1.0, base)

    def _compute_risk(self, vec):
        base = vec["threat_level"] * 0.6 + vec["stability_risk"] * 0.4
        if self.environment == "DANGER":
            base *= 1.3
        return min(1.0, base)

    def _choose_mission(self, opp, risk):
        w_protect = risk * 1.2
        w_stability = (1.0 - abs(risk - opp)) * 0.8
        w_learn = opp * 1.1
        w_opt = opp * 0.7 * (1.0 - risk)
        weights = {
            "PROTECT": w_protect,
            "STABILITY": w_stability,
            "LEARN": w_learn,
            "OPTIMIZE": w_opt
        }
        mission = max(weights.items(), key=lambda x: x[1])[0]
        return mission, weights

    def _compute_plan_adjustment(self, mission, event, severity):
        if mission == "PROTECT":
            if severity > 60:
                return "Immediate defensive shift"
            return "Heightened caution"
        if mission == "STABILITY":
            if event != "None":
                return "Rebalance systems"
            return "Maintain equilibrium"
        if mission == "LEARN":
            if event == "Learning window":
                return "Accelerate learning"
            return "Normal learning"
        if mission == "OPTIMIZE":
            if event == "Optimization window":
                return "Boost efficiency"
            return "Standard optimization"
        return "None"

    def _compute_anticipation(self, vec):
        if vec["threat_level"] > 0.6:
            return "Prepare for escalation"
        if vec["learning_opportunity"] > 0.6:
            return "Prepare for learning surge"
        if vec["efficiency_pressure"] > 0.6:
            return "Prepare for optimization"
        return "None"

    def run(self):
        while self.running and self.node.is_running():
            try:
                vec = self._compute_situation_vector()
                self.environment = self._detect_environment(vec)
                event, severity = self._detect_unexpected_event(vec)
                opp = self._compute_opportunity(vec)
                risk = self._compute_risk(vec)
                mission, weights = self._choose_mission(opp, risk)
                plan = self._compute_plan_adjustment(mission, event, severity)
                anticipation = self._compute_anticipation(vec)
                with self._lock:
                    self.current_mission = mission
                    self.mission_weights = weights
                    self.unexpected_event = event
                    self.event_severity = severity
                    self.opportunity_score = opp
                    self.risk_score = risk
                    self.plan_adjustment = plan
                    self.anticipation = anticipation
                self._last_threat_level = vec["threat_level"]
                self._last_stability_risk = vec["stability_risk"]
                time.sleep(5)
            except:
                time.sleep(5)

    def stop(self):
        self.running = False

    def snapshot(self):
        with self._lock:
            return {
                "mission": self.current_mission,
                "weights": dict(self.mission_weights),
                "environment": self.environment,
                "unexpected_event": self.unexpected_event,
                "event_severity": self.event_severity,
                "opportunity_score": self.opportunity_score,
                "risk_score": self.risk_score,
                "plan_adjustment": self.plan_adjustment,
                "anticipation": self.anticipation,
            }


class BackupPathManager:
    def __init__(self):
        self._lock = threading.Lock()
        cfg = load_config()
        self.primary = cfg.get("primary_backup_path", "C:/BorgNetPrimary")
        self.secondary = cfg.get("secondary_backup_path", "D:/BorgNetSecondary")

    def get_primary(self):
        with self._lock:
            return self.primary

    def get_secondary(self):
        with self._lock:
            return self.secondary

    def set_primary(self, path):
        with self._lock:
            self.primary = path
            cfg = load_config()
            cfg["primary_backup_path"] = path
            save_config(cfg)

    def set_secondary(self, path):
        with self._lock:
            self.secondary = path
            cfg = load_config()
            cfg["secondary_backup_path"] = path
            save_config(cfg)


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
            except:
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
        except:
            if primary:
                self._set_primary_status(DRIVE_STATUS_FAILING)
            else:
                self._set_secondary_status(DRIVE_STATUS_FAILING)
            return False

    def force_sync_once(self):
        primary_dir = self.get_primary_dir()
        secondary_dir = self.get_secondary_dir()
        used = None
        if primary_dir:
            if self._write_state_to_dir(primary_dir, primary=True):
                used = primary_dir
        if used is None and secondary_dir:
            self._write_state_to_dir(secondary_dir, primary=False)

    def run(self):
        while self.running and self.node.is_running():
            self.force_sync_once()
            time.sleep(10)

    def stop(self):
        self.running = False


class BorgNode:
    def __init__(self, name, is_queen, config, memory,
                 threat_detector, network_observer,
                 simulation_engine, predictive_engine,
                 game_awareness, judgment_engine,
                 situational_engine, shared_sync):
        self.name = name
        self.is_queen = is_queen
        self.config = config
        self.memory = memory
        self.threat_detector = threat_detector
        self.network_observer = network_observer
        self.simulation_engine = simulation_engine
        self.predictive_engine = predictive_engine
        self.game_awareness = game_awareness
        self.judgment_engine = judgment_engine
        self.situational_engine = situational_engine
        self.shared_sync = shared_sync
        self._running = True

    def start(self):
        self.threat_detector.start()
        self.network_observer.start()
        self.simulation_engine.start()
        self.predictive_engine.start()
        self.game_awareness.start()
        self.judgment_engine.start()
        self.situational_engine.start()
        self.shared_sync.start()

    def stop(self):
        self._running = False
        self.threat_detector.stop()
        self.network_observer.stop()
        self.simulation_engine.stop()
        self.predictive_engine.stop()
        self.game_awareness.stop()
        self.judgment_engine.stop()
        self.situational_engine.stop()
        self.shared_sync.stop()

    def is_running(self):
        return self._running


class BorgQueen:
    def __init__(self, registry):
        self.registry = registry
        self._lock = registry["_lock"]

    def get_all_nodes(self):
        with self._lock:
            return dict(self.registry["nodes"])

    def get_node(self, name):
        with self._lock:
            return self.registry["nodes"].get(name)


class BorgGUI:
    def __init__(self, root, node, shared_registry, backup_manager):
        self.root = root
        self.node = node
        self.shared_registry = shared_registry
        self.backup_manager = backup_manager
        self.queen = BorgQueen(shared_registry)

        self.root.title("BorgNet Situational Hive")
        self.root.geometry("1300x800")
        self.root.configure(bg="#101015")

        style = ttk.Style()
        try:
            style.theme_use("default")
        except:
            pass
        style.configure("TNotebook", background="#101015", borderwidth=0)
        style.configure("TNotebook.Tab", background="#202030", foreground="#cccccc", padding=6)
        style.map("TNotebook.Tab", background=[("selected", "#303048")])
        style.configure("TFrame", background="#101015")

        self._build()

    def _build(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.tab_overview = ttk.Frame(self.tabs)
        self.tab_predict = ttk.Frame(self.tabs)
        self.tab_threats = ttk.Frame(self.tabs)
        self.tab_game = ttk.Frame(self.tabs)
        self.tab_judgment = ttk.Frame(self.tabs)
        self.tab_situational = ttk.Frame(self.tabs)
        self.tab_borgnet = ttk.Frame(self.tabs)
        self.tab_collective = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_overview, text="Overview")
        self.tabs.add(self.tab_predict, text="Predictive AI")
        self.tabs.add(self.tab_threats, text="Threats")
        self.tabs.add(self.tab_game, text="Game Awareness")
        self.tabs.add(self.tab_judgment, text="Judgment")
        self.tabs.add(self.tab_situational, text="Situational Cortex")
        self.tabs.add(self.tab_borgnet, text="BorgNet Status")
        self.tabs.add(self.tab_collective, text="Collective Storage")

        self._build_overview_tab()
        self._build_predict_tab()
        self._build_threats_tab()
        self._build_game_tab()
        self._build_judgment_tab()
        self._build_situational_tab()
        self._build_borgnet_tab()
        self._build_collective_tab()

        self._update()

    def _make_labeled_text(self, parent, label_text):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=6, pady=4)
        lbl = tk.Label(frame, text=label_text, fg="#88c0ff", bg="#101015", anchor="w")
        lbl.pack(fill="x")
        text = tk.Text(frame, bg="#050509", fg="#a9b1d6", height=8)
        text.pack(fill="both", expand=True)
        text.configure(font=("Consolas", 9))
        return text

    def _build_overview_tab(self):
        top = ttk.Frame(self.tab_overview)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True)

        self.overview_status = self._make_labeled_text(left, "System / Memory Snapshot")
        self.overview_threat = self._make_labeled_text(left, "Threat Summary")

        self.overview_predict = self._make_labeled_text(right, "Predictive AI Summary")
        self.overview_situational = self._make_labeled_text(right, "Situational Summary")

    def _build_predict_tab(self):
        self.predict_text = self._make_labeled_text(self.tab_predict, "Predictive Engine State")

    def _build_threats_tab(self):
        self.threat_text = self._make_labeled_text(self.tab_threats, "Threat Detector State")

    def _build_game_tab(self):
        frame = ttk.Frame(self.tab_game)
        frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.game_text = tk.Text(frame, bg="#050509", fg="#a9b1d6")
        self.game_text.pack(fill="both", expand=True)
        self.game_text.configure(font=("Consolas", 9))

    def _build_judgment_tab(self):
        self.judgment_text = self._make_labeled_text(self.tab_judgment, "Judgment Engine State")

    def _build_situational_tab(self):
        self.situational_text = self._make_labeled_text(self.tab_situational, "Situational Cortex State")

    def _build_borgnet_tab(self):
        outer = ttk.Frame(self.tab_borgnet)
        outer.pack(fill="both", expand=True, padx=6, pady=4)

        top = ttk.Frame(outer)
        top.pack(side="top", fill="x", pady=4)

        tk.Label(top, text="Primary Backup Path:", fg="#88c0ff", bg="#101015").grid(row=0, column=0, sticky="w")
        self.primary_entry = tk.Entry(top, bg="#050509", fg="#a9b1d6", width=60)
        self.primary_entry.grid(row=0, column=1, padx=4, pady=2, sticky="we")
        self.primary_entry.insert(0, self.backup_manager.get_primary())
        btn_browse_primary = tk.Button(top, text="Browse...", command=self._browse_primary)
        btn_browse_primary.grid(row=0, column=2, padx=4, pady=2)

        tk.Label(top, text="Secondary Backup Path:", fg="#88c0ff", bg="#101015").grid(row=1, column=0, sticky="w")
        self.secondary_entry = tk.Entry(top, bg="#050509", fg="#a9b1d6", width=60)
        self.secondary_entry.grid(row=1, column=1, padx=4, pady=2, sticky="we")
        self.secondary_entry.insert(0, self.backup_manager.get_secondary())
        btn_browse_secondary = tk.Button(top, text="Browse...", command=self._browse_secondary)
        btn_browse_secondary.grid(row=1, column=2, padx=4, pady=2)

        btn_test = tk.Button(top, text="Test Write", command=self._test_write_paths)
        btn_test.grid(row=2, column=1, sticky="w", pady=4)

        btn_force_sync = tk.Button(top, text="Force Sync Now", command=self._force_sync_now)
        btn_force_sync.grid(row=2, column=2, sticky="w", pady=4)

        top.columnconfigure(1, weight=1)

        self.borgnet_text = tk.Text(outer, bg="#050509", fg="#a9b1d6")
        self.borgnet_text.pack(fill="both", expand=True)
        self.borgnet_text.configure(font=("Consolas", 9))

    def _build_collective_tab(self):
        outer = ttk.Frame(self.tab_collective)
        outer.pack(fill="both", expand=True, padx=6, pady=4)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        lbl_nodes = tk.Label(left, text="Hive Nodes", fg="#88c0ff", bg="#101015", anchor="w")
        lbl_nodes.pack(fill="x")

        self.node_listbox = tk.Listbox(left, bg="#050509", fg="#a9b1d6", height=20)
        self.node_listbox.pack(fill="y", expand=False)
        self.node_listbox.configure(font=("Consolas", 9))
        self.node_listbox.bind("<<ListboxSelect>>", self._on_node_selected)

        self.collective_detail = tk.Text(right, bg="#050509", fg="#a9b1d6")
        self.collective_detail.pack(fill="both", expand=True)
        self.collective_detail.configure(font=("Consolas", 9))

    def _safe_json(self, obj):
        try:
            return json.dumps(obj, indent=2)
        except Exception as e:
            return f"<< JSON error: {e} >>\n{str(obj)}"

    def _browse_primary(self):
        path = filedialog.askdirectory(title="Select Primary Backup Directory")
        if path:
            self.primary_entry.delete(0, tk.END)
            self.primary_entry.insert(0, path)
            self.backup_manager.set_primary(path)

    def _browse_secondary(self):
        path = filedialog.askdirectory(title="Select Secondary Backup Directory")
        if path:
            self.secondary_entry.delete(0, tk.END)
            self.secondary_entry.insert(0, path)
            self.backup_manager.set_secondary(path)

    def _test_write_paths(self):
        results = []
        for label, path in [("Primary", self.backup_manager.get_primary()),
                            ("Secondary", self.backup_manager.get_secondary())]:
            if not path:
                results.append(f"{label}: No path set")
                continue
            try:
                if not os.path.exists(path):
                    results.append(f"{label}: Path does not exist")
                    continue
                test_file = os.path.join(path, "borgnet_test.tmp")
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write("borgnet test")
                os.remove(test_file)
                results.append(f"{label}: OK")
            except Exception as e:
                results.append(f"{label}: ERROR - {e}")
        messagebox.showinfo("Test Write Results", "\n".join(results))

    def _force_sync_now(self):
        try:
            self.node.shared_sync.force_sync_once()
            messagebox.showinfo("Force Sync", "Manual sync triggered.")
        except Exception as e:
            messagebox.showerror("Force Sync Error", str(e))

    def _on_node_selected(self, event):
        try:
            sel = self.node_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            name = self.node_listbox.get(idx)
        except:
            return
        try:
            node_state = self.queen.get_node(name)
        except:
            node_state = None
        self.collective_detail.delete("1.0", tk.END)
        if not node_state:
            self.collective_detail.insert(tk.END, f"No state for node '{name}'")
        else:
            self.collective_detail.insert(tk.END, self._safe_json(node_state))

    def _update_collective_view(self):
        try:
            nodes = self.queen.get_all_nodes()
        except:
            nodes = {}
        existing = set(self.node_listbox.get(0, tk.END))
        incoming = set(nodes.keys())
        if existing != incoming:
            self.node_listbox.delete(0, tk.END)
            for n in sorted(incoming):
                self.node_listbox.insert(tk.END, n)

    def _update_borgnet_status(self, pred_snap, drive_stats, primary_status, secondary_status):
        status_payload = {
            "node_name": self.node.name,
            "is_queen": self.node.is_queen,
            "mode": self.node.config.snapshot().get("mode"),
            "collective_risk_score": pred_snap.get("collective_risk_score"),
            "hive_anomaly_risk": pred_snap.get("hive_anomaly_risk"),
            "hive_drive_risk": pred_snap.get("hive_drive_risk"),
            "hive_trend": pred_snap.get("hive_trend"),
            "health_trend": pred_snap.get("health_trend"),
            "primary_status": primary_status,
            "secondary_status": secondary_status,
            "drive_latency": drive_stats,
            "backup_paths": {
                "primary": self.backup_manager.get_primary(),
                "secondary": self.backup_manager.get_secondary(),
            },
            "forecast": pred_snap.get("forecast"),
            "notes": pred_snap.get("notes", []),
        }
        self.borgnet_text.delete("1.0", tk.END)
        self.borgnet_text.insert(tk.END, self._safe_json(status_payload))

    def _build_judgment_tab(self):
        self.judgment_text = self._make_labeled_text(self.tab_judgment, "Judgment Engine State")

    def _build_situational_tab(self):
        self.situational_text = self._make_labeled_text(self.tab_situational, "Situational Cortex State")

    def _build_predict_tab(self):
        self.predict_text = self._make_labeled_text(self.tab_predict, "Predictive Engine State")

    def _build_threats_tab(self):
        self.threat_text = self._make_labeled_text(self.tab_threats, "Threat Detector State")

    def _build_game_tab(self):
        frame = ttk.Frame(self.tab_game)
        frame.pack(fill="both", expand=True, padx=6, pady=4)
        self.game_text = tk.Text(frame, bg="#050509", fg="#a9b1d6")
        self.game_text.pack(fill="both", expand=True)
        self.game_text.configure(font=("Consolas", 9))

    def _build_collective_tab(self):
        outer = ttk.Frame(self.tab_collective)
        outer.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)
        lbl_nodes = tk.Label(left, text="Hive Nodes", fg="#88c0ff", bg="#101015", anchor="w")
        lbl_nodes.pack(fill="x")
        self.node_listbox = tk.Listbox(left, bg="#050509", fg="#a9b1d6", height=20)
        self.node_listbox.pack(fill="y", expand=False)
        self.node_listbox.configure(font=("Consolas", 9))
        self.node_listbox.bind("<<ListboxSelect>>", self._on_node_selected)
        self.collective_detail = tk.Text(right, bg="#050509", fg="#a9b1d6")
        self.collective_detail.pack(fill="both", expand=True)
        self.collective_detail.configure(font=("Consolas", 9))

    def _update(self):
        if not self.node.is_running():
            return

        try:
            mem_snap = self.node.memory.snapshot()
            threat_snap = self.node.threat_detector.get_summary()
            pred_snap = self.node.predictive_engine.get_summary()
            situ_snap = self.node.situational_engine.snapshot()
            game_snap = self.node.game_awareness.snapshot()
            judge_snap = self.node.judgment_engine.snapshot()
            cfg_snap = self.node.config.snapshot()
            drive_stats = self.node.shared_sync.get_drive_latency_stats()
            primary_status, secondary_status = self.node.shared_sync.get_statuses()
        except Exception:
            self.root.after(1000, self._update)
            return

        self.overview_status.delete("1.0", tk.END)
        self.overview_status.insert(tk.END, self._safe_json({
            "node": self.node.name,
            "mode": cfg_snap.get("mode"),
            "memory": {
                "connections": mem_snap.get("connection_count"),
                "known_endpoints": len(mem_snap.get("endpoint_frequency", {})),
                "system_history_len": mem_snap.get("system_history_len"),
            }
        }))

        self.overview_threat.delete("1.0", tk.END)
        self.overview_threat.insert(tk.END, self._safe_json(threat_snap))

        self.overview_predict.delete("1.0", tk.END)
        self.overview_predict.insert(tk.END, self._safe_json(pred_snap))

        self.overview_situational.delete("1.0", tk.END)
        self.overview_situational.insert(tk.END, self._safe_json(situ_snap))

        self.predict_text.delete("1.0", tk.END)
        self.predict_text.insert(tk.END, self._safe_json(pred_snap))

        self.threat_text.delete("1.0", tk.END)
        self.threat_text.insert(tk.END, self._safe_json(threat_snap))

        self.game_text.delete("1.0", tk.END)
        self.game_text.insert(tk.END, self._safe_json(game_snap))

        self.judgment_text.delete("1.0", tk.END)
        self.judgment_text.insert(tk.END, self._safe_json(judge_snap))

        self.situational_text.delete("1.0", tk.END)
        self.situational_text.insert(tk.END, self._safe_json(situ_snap))

        self._update_borgnet_status(pred_snap, drive_stats, primary_status, secondary_status)
        self._update_collective_view()

        self.root.after(1000, self._update)


def main():
    cfg = AdaptiveConfig()
    mem = MemoryEngine()
    alert_q = queue.Queue()
    threat = ThreatDetector(mem, alert_q)
    net = NetworkObserver(mem, cfg)
    sim = SimulationEngine(mem, cfg)

    shared_registry = {"nodes": {}, "_lock": threading.Lock()}
    backup_manager = BackupPathManager()

    def get_primary():
        return backup_manager.get_primary()

    def get_secondary():
        return backup_manager.get_secondary()

    node = BorgNode(
        "Node1", True, cfg, mem,
        None, None, None, None, None, None, None, None
    )

    sync = SharedStateSync(node, get_primary, get_secondary, shared_registry)
    pred = PredictiveAIEngine(node, shared_registry)
    judge = JudgmentEngine(node)
    game = GameAwarenessEngine(node)
    situ = SituationalAwarenessEngine(node)

    node.threat_detector = threat
    node.network_observer = net
    node.simulation_engine = sim
    node.predictive_engine = pred
    node.game_awareness = game
    node.judgment_engine = judge
    node.situational_engine = situ
    node.shared_sync = sync

    node.start()

    root = tk.Tk()
    gui = BorgGUI(root, node, shared_registry, backup_manager)
    root.mainloop()

    node.stop()


if __name__ == "__main__":
    main()

