import sys
import os
import json
import time
import random
import socket
import threading
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import (
    Qt, QTimer, QThreadPool, QRunnable, pyqtSignal, QObject
)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QTabWidget, QListWidget, QListWidgetItem, QComboBox
)

JSON_LOCK = threading.Lock()

# ============================================================
# ENUMS & DATA CLASSES
# ============================================================
class RouteColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


@dataclass
class RouteStatus:
    name: str
    latency_ms: float
    loss: float
    congestion: float
    color: RouteColor
    worker_id: str = ""
    is_opportunistic: bool = False


@dataclass
class RouteForecast:
    forecast_latency: float
    forecast_loss: float
    forecast_congestion: float
    forecast_color: RouteColor
    health_score: float


@dataclass
class RouteThreatProjection:
    route_name: str
    current_color: RouteColor
    projected_color_30s: RouteColor
    projected_color_60s: RouteColor
    risk_score: float
    anomaly: float
    trend_risk: float


@dataclass
class BackupMission:
    mission_id: str
    source_path: str
    target_path: str
    chosen_route: Optional[RouteStatus] = None
    success: bool = False
    message: str = ""
    started_at: float = time.time()
    finished_at: float = 0.0


@dataclass
class MissionOutcomePrediction:
    success_prob: float
    bypass_prob: float
    fail_prob: float
    helicopter_prob: float


# ============================================================
# ROUTE HISTORY & FORECASTING
# ============================================================
class RouteHistory:
    def __init__(self):
        self.history: Dict[str, List[RouteStatus]] = {}

    def add_sample(self, status: RouteStatus):
        arr = self.history.setdefault(status.name, [])
        arr.append(status)
        if len(arr) > 100:
            self.history[status.name] = arr[-100:]

    def get_history(self, name: str) -> List[RouteStatus]:
        return self.history.get(name, [])

    def smooth(self, values: List[float], alpha: float = 0.3) -> float:
        if not values:
            return 0.0
        s = values[0]
        for v in values[1:]:
            s = alpha * v + (1 - alpha) * s
        return s

    def trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = list(range(len(values)))
        n = len(values)
        avg_x = sum(x) / n
        avg_y = sum(values) / n
        num = sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x, values))
        den = sum((xi - avg_x) ** 2 for xi in x) or 1.0
        return num / den

    def anomaly_score(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        var = sum((v - avg) ** 2 for v in values) / len(values)
        std = var ** 0.5
        if std == 0:
            return 0.0
        latest = values[-1]
        z = abs(latest - avg) / std
        return min(1.0, z / 3.0)


class ShortHorizonForecaster:
    def __init__(self, history: RouteHistory):
        self.history = history

    def forecast_routes(self, routes: Dict[str, RouteStatus]) -> Dict[str, RouteForecast]:
        out: Dict[str, RouteForecast] = {}
        for name, status in routes.items():
            hist = self.history.get_history(name)
            lat_vals = [s.latency_ms for s in hist] or [status.latency_ms]
            loss_vals = [s.loss for s in hist] or [status.loss]
            cong_vals = [s.congestion for s in hist] or [status.congestion]

            smoothed_lat = self.history.smooth(lat_vals)
            smoothed_loss = self.history.smooth(loss_vals)
            smoothed_cong = self.history.smooth(cong_vals)

            lat_trend = self.history.trend(lat_vals)
            loss_trend = self.history.trend(loss_vals)
            cong_trend = self.history.trend(cong_vals)

            f_lat = smoothed_lat + lat_trend * 5
            f_loss = smoothed_loss + loss_trend * 5
            f_cong = smoothed_cong + cong_trend * 5

            score = f_lat * 0.4 + f_loss * 0.4 + f_cong * 0.2
            if score < 20:
                fc_color = RouteColor.GREEN
            elif score < 40:
                fc_color = RouteColor.YELLOW
            elif score < 60:
                fc_color = RouteColor.ORANGE
            else:
                fc_color = RouteColor.RED

            health = max(0.0, 100.0 - score)
            out[name] = RouteForecast(
                forecast_latency=f_lat,
                forecast_loss=f_loss,
                forecast_congestion=f_cong,
                forecast_color=fc_color,
                health_score=health,
            )
        return out


# ============================================================
# THREAT CONE 2.0
# ============================================================
class ThreatCone:
    def __init__(self, history: RouteHistory, queen: "QueenBrain"):
        self.history = history
        self.queen = queen

    def project_30s(self) -> List[RouteThreatProjection]:
        projections: List[RouteThreatProjection] = []
        snapshot = self.queen.get_snapshot()

        for name, status in snapshot.items():
            hist = self.history.get_history(name)
            if not hist:
                continue

            lat_vals = [s.latency_ms for s in hist]
            loss_vals = [s.loss for s in hist]
            cong_vals = [s.congestion for s in hist]

            smoothed_lat = self.history.smooth(lat_vals)
            smoothed_loss = self.history.smooth(loss_vals)
            smoothed_cong = self.history.smooth(cong_vals)

            lat_trend = self.history.trend(lat_vals)
            loss_trend = self.history.trend(loss_vals)

            lat_30 = smoothed_lat + lat_trend * 5
            loss_30 = smoothed_loss + loss_trend * 5
            lat_60 = smoothed_lat + lat_trend * 10
            loss_60 = smoothed_loss + loss_trend * 10

            score_30 = lat_30 * 0.4 + loss_30 * 0.4 + smoothed_cong * 0.2
            score_60 = lat_60 * 0.4 + loss_60 * 0.4 + smoothed_cong * 0.2

            if score_30 < 20:
                proj_30 = RouteColor.GREEN
            elif score_30 < 40:
                proj_30 = RouteColor.YELLOW
            elif score_30 < 60:
                proj_30 = RouteColor.ORANGE
            else:
                proj_30 = RouteColor.RED

            if score_60 < 20:
                proj_60 = RouteColor.GREEN
            elif score_60 < 40:
                proj_60 = RouteColor.YELLOW
            elif score_60 < 60:
                proj_60 = RouteColor.ORANGE
            else:
                proj_60 = RouteColor.RED

            anomaly = max(
                self.history.anomaly_score(lat_vals),
                self.history.anomaly_score(loss_vals),
                self.history.anomaly_score(cong_vals),
            )

            trend_risk = max(0.0, (lat_trend * 0.1 + loss_trend * 0.2))
            base_risk = score_30 / 100.0
            risk = min(1.0, base_risk + anomaly * 0.3 + trend_risk)

            projections.append(RouteThreatProjection(
                route_name=name,
                current_color=status.color,
                projected_color_30s=proj_30,
                projected_color_60s=proj_60,
                risk_score=risk,
                anomaly=anomaly,
                trend_risk=trend_risk,
            ))
        return projections


# ============================================================
# QUEEN BRAIN, STANCE, RL, DECISION LOG
# ============================================================
class QueenStance(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    BEAST = "Beast"


@dataclass
class RouteArm:
    successes: int = 0
    failures: int = 0


def bandit_score(arm: RouteArm) -> float:
    a = arm.successes + 1
    b = arm.failures + 1
    return random.betavariate(a, b)


@dataclass
class DecisionEntry:
    timestamp: float
    chosen_route: str
    reason: str


@dataclass
class OverrideDecision:
    timestamp: float
    original_route: str
    overridden_route: str
    justification: str


@dataclass
class PreemptiveDecision:
    timestamp: float
    from_route: str
    to_route: str
    risk: float
    anomaly: float
    stance: QueenStance
    reason: str


class QueenDecisionLog:
    def __init__(self):
        self.entries: List[Any] = []

    def record_decision(self, chosen: RouteStatus, all_routes: Dict[str, RouteStatus]):
        reason = (
            f"Selected {chosen.name} because it had color {chosen.color.value} "
            f"and lowest latency among acceptable routes."
        )
        self.entries.append(DecisionEntry(
            timestamp=time.time(),
            chosen_route=chosen.name,
            reason=reason
        ))

    def record_override(self, original: RouteStatus, new: RouteStatus):
        justification = (
            f"User override: replacing {original.name} ({original.color.value}) "
            f"with {new.name} ({new.color.value})."
        )
        self.entries.append(OverrideDecision(
            timestamp=time.time(),
            original_route=original.name,
            overridden_route=new.name,
            justification=justification
        ))

    def record_preemptive(self, from_route: RouteStatus, to_route: RouteStatus,
                          risk: float, anomaly: float, stance: QueenStance, reason: str):
        self.entries.append(PreemptiveDecision(
            timestamp=time.time(),
            from_route=from_route.name,
            to_route=to_route.name,
            risk=risk,
            anomaly=anomaly,
            stance=stance,
            reason=reason
        ))


class QueenBrain:
    def __init__(self):
        self.route_map: Dict[str, RouteStatus] = {}
        self.history = RouteHistory()
        self.log = QueenDecisionLog()
        self.stance = QueenStance.BALANCED
        self.arms: Dict[str, RouteArm] = {}
        self.current_context: str = "idle"  # browser/game/system/network/idle
        self.last_preemptive_time: float = 0.0
        self.preemptive_cooldown: float = 10.0  # seconds
        self.pinned_route: Optional[str] = None

    def update_route(self, status: RouteStatus):
        self.route_map[status.name] = status
        self.history.add_sample(status)

    def get_snapshot(self) -> Dict[str, RouteStatus]:
        return dict(self.route_map)

    def update_context_from_telemetry(self, category: str):
        # category: browser/game/system/network
        self.current_context = category
        self._auto_switch_stance()

    def _auto_switch_stance(self):
        # Simple but real: stance depends on context
        if self.current_context == "game":
            self.stance = QueenStance.BEAST
        elif self.current_context == "browser":
            self.stance = QueenStance.BALANCED
        elif self.current_context in ("system", "network"):
            self.stance = QueenStance.CONSERVATIVE
        else:
            self.stance = QueenStance.BALANCED


def choose_best_route_with_stance(route_map: Dict[str, RouteStatus],
                                  stance: QueenStance,
                                  queen: QueenBrain) -> Optional[RouteStatus]:
    if not route_map:
        return None

    if stance == QueenStance.CONSERVATIVE:
        allowed = {RouteColor.GREEN, RouteColor.YELLOW}
    elif stance == QueenStance.BEAST:
        allowed = {RouteColor.GREEN, RouteColor.YELLOW, RouteColor.ORANGE}
    else:
        allowed = {RouteColor.GREEN, RouteColor.YELLOW, RouteColor.ORANGE}

    candidates = [r for r in route_map.values() if r.color in allowed]
    if not candidates:
        return None

    priority = [RouteColor.GREEN, RouteColor.YELLOW, RouteColor.ORANGE]

    def key_fn(r: RouteStatus):
        base = priority.index(r.color) if r.color in priority else 99
        arm = queen.arms.get(r.name, RouteArm())
        bscore = bandit_score(arm)
        return (base, r.latency_ms - bscore * 100.0)

    return sorted(candidates, key=key_fn)[0]


def update_reinforcement(queen: QueenBrain, mission: BackupMission):
    if not mission.chosen_route:
        return
    name = mission.chosen_route.name
    arm = queen.arms.setdefault(name, RouteArm())
    if mission.success:
        arm.successes += 1
    else:
        arm.failures += 1


# ============================================================
# PREEMPTIVE REROUTE BRAIN (STANCE + TELEMETRY FUSED)
# ============================================================
class PreemptiveRerouteBrain:
    def __init__(self, queen: QueenBrain, threat_cone: ThreatCone):
        self.queen = queen
        self.threat_cone = threat_cone

    def _stance_thresholds(self) -> Tuple[float, float]:
        # (risk_threshold, anomaly_threshold)
        if self.queen.stance == QueenStance.CONSERVATIVE:
            return 0.45, 0.35
        elif self.queen.stance == QueenStance.BEAST:
            return 0.75, 0.65
        else:
            return 0.60, 0.50

    def maybe_preemptive_reroute(self):
        now = time.time()
        if now - self.queen.last_preemptive_time < self.queen.preemptive_cooldown:
            return None

        projections = self.threat_cone.project_30s()
        if not projections:
            return None

        risk_threshold, anomaly_threshold = self._stance_thresholds()

        # Telemetry fusion: if context is "game", tolerate more risk but react faster to anomaly
        if self.queen.current_context == "game":
            risk_threshold += 0.05
            anomaly_threshold -= 0.05
        elif self.queen.current_context == "system":
            risk_threshold -= 0.05

        current_routes = self.queen.get_snapshot()
        if not current_routes:
            return None

        # Determine current best route
        current_best = choose_best_route_with_stance(current_routes, self.queen.stance, self.queen)
        if not current_best:
            return None

        current_proj = next((p for p in projections if p.route_name == current_best.name), None)
        if not current_proj:
            return None

        if current_proj.risk_score < risk_threshold and current_proj.anomaly < anomaly_threshold:
            return None

        safer_candidates = [
            p for p in projections
            if p.route_name != current_best.name
            and p.risk_score < current_proj.risk_score
            and p.projected_color_30s in (RouteColor.GREEN, RouteColor.YELLOW)
        ]
        if not safer_candidates:
            return None

        safer_candidates.sort(key=lambda p: p.risk_score)
        chosen_proj = safer_candidates[0]
        new_route = current_routes.get(chosen_proj.route_name)
        if not new_route:
            return None

        self.queen.pinned_route = new_route.name
        self.queen.last_preemptive_time = now

        reason = (
            f"Preemptive reroute from {current_best.name} to {new_route.name} | "
            f"risk {current_proj.risk_score:.2f} -> {chosen_proj.risk_score:.2f}, "
            f"anomaly {current_proj.anomaly:.2f}, stance {self.queen.stance.value}, "
            f"context {self.queen.current_context}"
        )
        self.queen.log.record_preemptive(
            from_route=current_best,
            to_route=new_route,
            risk=current_proj.risk_score,
            anomaly=current_proj.anomaly,
            stance=self.queen.stance,
            reason=reason,
        )
        return new_route


# ============================================================
# SETTINGS & DELTA WRITER
# ============================================================
class SettingsStore:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            appdata = os.getenv("APPDATA") or "."
            base_dir = os.path.join(appdata, "HeavyData")
            os.makedirs(base_dir, exist_ok=True)
            path = os.path.join(base_dir, "settings.json")
        self.path = path
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with JSON_LOCK:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value


class DeltaWriter:
    def __init__(self):
        self.last_hash = None

    def compute_hash(self, data: dict):
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def write_if_changed(self, data: dict, path: str):
        new_hash = self.compute_hash(data)
        if new_hash == self.last_hash:
            return False, "No changes detected. Skipping write."

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"

        for _ in range(5):
            try:
                with JSON_LOCK:
                    with open(tmp, "w") as f:
                        json.dump(data, f, indent=2)
                    os.replace(tmp, path)
                self.last_hash = new_hash
                return True, "New data written."
            except PermissionError:
                time.sleep(0.1)

        return False, "Permission denied after retries."


# ============================================================
# BACKUP ENGINE
# ============================================================
class BackupEngine:
    def __init__(self, queen: QueenBrain, settings: SettingsStore):
        self.queen = queen
        self.settings = settings
        self.delta = DeltaWriter()

    def _simulate_copy(self, src: str, dst: str, latency_ms: float):
        time.sleep(min(latency_ms / 1000.0, 2.0))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with JSON_LOCK:
            with open(dst, "w") as f:
                f.write(f"Backup from {src} via simulated route\n")

    def run_mission(self, mission: BackupMission):
        snapshot = self.queen.get_snapshot()
        route = choose_best_route_with_stance(snapshot, self.queen.stance, self.queen)
        if route is None:
            mission.success = False
            mission.message = "No viable route"
            mission.finished_at = time.time()
            return mission, []

        mission.chosen_route = route
        try:
            self._simulate_copy(mission.source_path, mission.target_path, route.latency_ms)
            mission.success = True
            mission.message = f"Backup completed via {route.name} ({route.color.value})"
        except Exception as e:
            mission.success = False
            mission.message = f"Backup failed via {route.name}: {e}"
        finally:
            mission.finished_at = time.time()

        data_to_write = {
            "mission_id": mission.mission_id,
            "route": mission.chosen_route.name,
            "color": mission.chosen_route.color.value,
            "worker": mission.chosen_route.worker_id,
            "timestamp": mission.started_at,
            "success": mission.success,
            "message": mission.message,
        }

        results: List[Dict[str, Any]] = []
        local_path = self.settings.get("local_backup_path")
        smb_path = self.settings.get("smb_backup_path")

        if local_path:
            try:
                dst_local = os.path.join(local_path, f"{mission.mission_id}_backup.json")
                changed, msg = self.delta.write_if_changed(data_to_write, dst_local)
                status = "SUCCESS" if changed else "SKIPPED"
                results.append({
                    "type": "Local",
                    "path": dst_local,
                    "status": status,
                    "message": msg,
                    "timestamp": time.time()
                })
            except Exception as e:
                results.append({
                    "type": "Local",
                    "path": local_path,
                    "status": "ERROR",
                    "message": str(e),
                    "timestamp": time.time()
                })

        if smb_path:
            try:
                dst_smb = os.path.join(smb_path, f"{mission.mission_id}_backup.json")
                changed, msg = self.delta.write_if_changed(data_to_write, dst_smb)
                status = "SUCCESS" if changed else "SKIPPED"
                results.append({
                    "type": "SMB",
                    "path": dst_smb,
                    "status": status,
                    "message": msg,
                    "timestamp": time.time()
                })
            except Exception as e:
                results.append({
                    "type": "SMB",
                    "path": smb_path,
                    "status": "ERROR",
                    "message": str(e),
                    "timestamp": time.time()
                })

        mission.message += "\n\nDual-write results:\n" + "\n".join(
            f"{r['type']} -> {r['status']}: {r['message']}" for r in results
        )

        update_reinforcement(self.queen, mission)
        return mission, results


# ============================================================
# PING & CONGESTION (SIMULATED)
# ============================================================
def ping_host(host: str) -> Tuple[float, float]:
    # Simulated latency/loss
    base = 20.0 if host == "127.0.0.1" else 60.0
    jitter = random.uniform(-10, 40)
    latency = max(1.0, base + jitter)
    loss = max(0.0, min(1.0, random.uniform(0.0, 0.05 if host == "127.0.0.1" else 0.15)))
    return latency, loss


def get_congestion() -> float:
    return random.uniform(0.0, 1.0)


def classify_route(latency_ms: float, loss_pct: float, cong_pct: float) -> RouteColor:
    score = latency_ms * 0.4 + loss_pct * 0.4 + cong_pct * 0.2
    if score < 20:
        return RouteColor.GREEN
    elif score < 40:
        return RouteColor.YELLOW
    elif score < 60:
        return RouteColor.ORANGE
    else:
        return RouteColor.RED


# ============================================================
# WORKERS
# ============================================================
class WorkerSignals(QObject):
    status_ready = pyqtSignal(RouteStatus)


class ProbeTask(QRunnable):
    def __init__(self, worker_id: str, route_name: str, interval: float, signals: WorkerSignals):
        super().__init__()
        self.worker_id = worker_id
        self.route_name = route_name
        self.interval = interval
        self.signals = signals
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        host_map = {
            "INT_LocalDisk": "127.0.0.1",
            "INT_CPUPath": "127.0.0.1",
            "EXT_SMB1": "192.168.1.10",
            "EXT_SMB2": "8.8.8.8",
            "EXT_CloudRelay": "1.1.1.1"
        }
        host = host_map.get(self.route_name, "8.8.8.8")

        while self._running:
            latency, loss = ping_host(host)
            congestion = get_congestion()
            color = classify_route(latency, loss * 100, congestion * 100)

            status = RouteStatus(
                name=self.route_name,
                latency_ms=latency,
                loss=loss * 100,
                congestion=congestion * 100,
                color=color,
                worker_id=self.worker_id
            )
            self.signals.status_ready.emit(status)
            time.sleep(self.interval)


# ============================================================
# PANELS
# ============================================================
class HeatmapPanel(QWidget):
    def __init__(self, queen: QueenBrain):
        super().__init__()
        self.queen = queen

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Route", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Swarm Heatmap"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self):
        routes = self.queen.get_snapshot()
        self.table.setRowCount(len(routes))
        for row, status in enumerate(routes.values()):
            self.table.setItem(row, 0, QTableWidgetItem(status.name))
            item = QTableWidgetItem(status.color.value)
            if status.color == RouteColor.GREEN:
                item.setBackground(Qt.green)
            elif status.color == RouteColor.YELLOW:
                item.setBackground(Qt.yellow)
            elif status.color == RouteColor.ORANGE:
                item.setBackground(Qt.darkYellow)
            else:
                item.setBackground(Qt.red)
            self.table.setItem(row, 1, item)


class BackupLanePanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Route", "Current Color", "Forecast Color",
            "Latency", "Loss", "Congestion", "Health Score"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Backup Lanes (Forecast)"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_lanes(self, routes: Dict[str, RouteStatus], forecasts: Dict[str, RouteForecast]):
        combined = []
        for name, status in routes.items():
            fc = forecasts.get(name)
            if not fc:
                continue
            combined.append((fc.health_score, status, fc))
        combined.sort(key=lambda x: x[0])

        self.table.setRowCount(len(combined))
        for row, (health, status, fc) in enumerate(combined):
            self.table.setItem(row, 0, QTableWidgetItem(status.name))

            cur_item = QTableWidgetItem(status.color.value)
            if status.color == RouteColor.GREEN:
                cur_item.setBackground(Qt.green)
            elif status.color == RouteColor.YELLOW:
                cur_item.setBackground(Qt.yellow)
            elif status.color == RouteColor.ORANGE:
                cur_item.setBackground(Qt.darkYellow)
            else:
                cur_item.setBackground(Qt.red)
            self.table.setItem(row, 1, cur_item)

            fc_item = QTableWidgetItem(fc.forecast_color.value)
            if fc.forecast_color == RouteColor.GREEN:
                fc_item.setBackground(Qt.green)
            elif fc.forecast_color == RouteColor.YELLOW:
                fc_item.setBackground(Qt.yellow)
            elif fc.forecast_color == RouteColor.ORANGE:
                fc_item.setBackground(Qt.darkYellow)
            else:
                fc_item.setBackground(Qt.red)
            self.table.setItem(row, 2, fc_item)

            self.table.setItem(row, 3, QTableWidgetItem(f"{fc.forecast_latency:.1f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{fc.forecast_loss:.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{fc.forecast_congestion:.1f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{health:.1f}"))


class RerouteFlowPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "From Route", "To Route", "Reason"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Reroute Flow"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.events: List[Dict[str, Any]] = []

    def add_event(self, from_route: Optional[str], to_route: str, reason: str):
        ts = time.time()
        self.events.append({
            "timestamp": ts,
            "from": from_route or "(none)",
            "to": to_route,
            "reason": reason
        })
        if len(self.events) > 50:
            self.events = self.events[-50:]
        self._refresh()

    def _refresh(self):
        self.table.setRowCount(len(self.events))
        for row, e in enumerate(self.events):
            self.table.setItem(row, 0, QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))))
            self.table.setItem(row, 1, QTableWidgetItem(e["from"]))
            self.table.setItem(row, 2, QTableWidgetItem(e["to"]))
            self.table.setItem(row, 3, QTableWidgetItem(e["reason"]))


class MissionTimelinePanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Destination", "Quality"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Mission Timeline"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.events: List[Tuple[float, str, RouteColor]] = []

    def set_events(self, events: List[Tuple[float, str, RouteColor]]):
        self.events = events
        self._refresh()

    def _refresh(self):
        self.table.setRowCount(len(self.events))
        for row, (ts, label, color) in enumerate(self.events):
            self.table.setItem(row, 0, QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))))
            self.table.setItem(row, 1, QTableWidgetItem(label))
            self.table.setItem(row, 2, QTableWidgetItem(color.value))


class DecisionLogPanel(QWidget):
    def __init__(self, queen: QueenBrain):
        super().__init__()
        self.queen = queen

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Route", "Reason/Justification"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Queen Decision Log"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self):
        entries = self.queen.log.entries
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            ts = getattr(entry, "timestamp", time.time())
            if isinstance(entry, DecisionEntry):
                route = entry.chosen_route
                reason = entry.reason
            elif isinstance(entry, OverrideDecision):
                route = f"{entry.original_route} -> {entry.overridden_route}"
                reason = entry.justification
            else:  # PreemptiveDecision
                route = f"{entry.from_route} -> {entry.to_route}"
                reason = entry.reason
            self.table.setItem(row, 0, QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))))
            self.table.setItem(row, 1, QTableWidgetItem(route))
            self.table.setItem(row, 2, QTableWidgetItem(reason))


class BackupManagerPanel(QWidget):
    def __init__(self, settings: SettingsStore):
        super().__init__()
        self.settings = settings
        self.history: List[Dict[str, Any]] = self.settings.get("backup_history", [])

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self.btn_local = QPushButton("Select Local Backup Folder")
        self.btn_smb = QPushButton("Select SMB Backup Folder")
        btn_layout.addWidget(self.btn_local)
        btn_layout.addWidget(self.btn_smb)

        layout.addLayout(btn_layout)

        self.label_local = QLabel("Local Backup Path: (not set)")
        self.label_smb = QLabel("SMB Backup Path: (not set)")
        layout.addWidget(self.label_local)
        layout.addWidget(self.label_smb)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Path Type", "Path", "Last Write", "Status", "Message"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Backup Status (Last 10 Missions):"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.btn_local.clicked.connect(self._select_local)
        self.btn_smb.clicked.connect(self._select_smb)

        self._load_paths()
        self._refresh_table()

    def _load_paths(self):
        local = self.settings.get("local_backup_path")
        smb = self.settings.get("smb_backup_path")
        if local:
            self.label_local.setText(f"Local Backup Path: {local}")
        if smb:
            self.label_smb.setText(f"SMB Backup Path: {smb}")

    def _select_local(self):
        path = QFileDialog.getExistingDirectory(self, "Select Local Backup Folder")
        if not path:
            return
        self.settings.set("local_backup_path", path)
        self.settings.save()
        self.label_local.setText(f"Local Backup Path: {path}")

    def _select_smb(self):
        path = QFileDialog.getExistingDirectory(self, "Select SMB Backup Folder")
        if not path:
            return
        self.settings.set("smb_backup_path", path)
        self.settings.save()
        self.label_smb.setText(f"SMB Backup Path: {path}")

    def add_history_entry(self, entry: Dict[str, Any]):
        self.history.append(entry)
        if len(self.history) > 10:
            self.history = self.history[-10:]
        self._refresh_table()
        self.settings.set("backup_history", self.history)
        self.settings.save()

    def _refresh_table(self):
        self.table.setRowCount(len(self.history))
        for row, h in enumerate(self.history):
            self.table.setItem(row, 0, QTableWidgetItem(h.get("type", "")))
            self.table.setItem(row, 1, QTableWidgetItem(h.get("path", "")))
            ts = h.get("timestamp", time.time())
            self.table.setItem(row, 2, QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))))
            self.table.setItem(row, 3, QTableWidgetItem(h.get("status", "")))
            self.table.setItem(row, 4, QTableWidgetItem(h.get("message", "")))


class UnifiedRouteTable(QTableWidget):
    def __init__(self, queen: QueenBrain, history: RouteHistory,
                 forecaster: ShortHorizonForecaster, threat_cone: ThreatCone,
                 parent=None):
        super().__init__(0, 12, parent)
        self.queen = queen
        self.history = history
        self.forecaster = forecaster
        self.threat_cone = threat_cone

        self.setHorizontalHeaderLabels([
            "Route",
            "Latency (ms)",
            "Loss (%)",
            "Congestion (%)",
            "Current Color",
            "Forecast Color",
            "Threat Projection",
            "Risk",
            "Worker",
            "Opportunistic",
            "Reroute?",
            "Mission Quality"
        ])

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(10, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(11, QHeaderView.ResizeToContents)

        self.verticalHeader().setDefaultSectionSize(22)
        self.setAlternatingRowColors(True)

    def _color_item(self, text: str, color: RouteColor) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if color == RouteColor.GREEN:
            item.setBackground(QColor(0, 120, 0))
        elif color == RouteColor.YELLOW:
            item.setBackground(QColor(180, 180, 0))
        elif color == RouteColor.ORANGE:
            item.setBackground(QColor(200, 120, 0))
        else:
            item.setBackground(QColor(160, 0, 0))
        item.setForeground(QColor(255, 255, 255))
        return item

    def refresh(self):
        routes = self.queen.get_snapshot()
        forecasts = self.forecaster.forecast_routes(routes)
        projections = {p.route_name: p for p in self.threat_cone.project_30s()}

        self.setRowCount(len(routes))
        for row, (name, status) in enumerate(routes.items()):
            fc = forecasts.get(name)
            proj = projections.get(name)

            self.setItem(row, 0, QTableWidgetItem(name))
            self.setItem(row, 1, QTableWidgetItem(f"{status.latency_ms:.1f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{status.loss:.1f}"))
            self.setItem(row, 3, QTableWidgetItem(f"{status.congestion:.1f}"))

            cur_item = self._color_item(status.color.value, status.color)
            self.setItem(row, 4, cur_item)

            if fc:
                fc_item = self._color_item(fc.forecast_color.value, fc.forecast_color)
                self.setItem(row, 5, fc_item)
            else:
                self.setItem(row, 5, QTableWidgetItem("N/A"))

            if proj:
                proj_text = f"{proj.projected_color_30s.value}/{proj.projected_color_60s.value}"
                proj_item = self._color_item(proj_text, proj.projected_color_30s)
                self.setItem(row, 6, proj_item)
                self.setItem(row, 7, QTableWidgetItem(f"{proj.risk_score:.2f}"))
            else:
                self.setItem(row, 6, QTableWidgetItem("N/A"))
                self.setItem(row, 7, QTableWidgetItem("N/A"))

            self.setItem(row, 8, QTableWidgetItem(status.worker_id or ""))
            self.setItem(row, 9, QTableWidgetItem("Yes" if status.is_opportunistic else "No"))

            reroute_flag = "Yes" if status.color in (RouteColor.ORANGE, RouteColor.RED) else "No"
            self.setItem(row, 10, QTableWidgetItem(reroute_flag))

            hist = self.history.get_history(name)
            if hist:
                lat_vals = [s.latency_ms for s in hist]
                loss_vals = [s.loss for s in hist]
                cong_vals = [s.congestion for s in hist]
                anomaly = max(
                    self.history.anomaly_score(lat_vals),
                    self.history.anomaly_score(loss_vals),
                    self.history.anomaly_score(cong_vals),
                )
                quality = max(0.0, 1.0 - anomaly)
                self.setItem(row, 11, QTableWidgetItem(f"{quality:.2f}"))
            else:
                self.setItem(row, 11, QTableWidgetItem("N/A"))


# ============================================================
# LIVE SYSTEM FEEDBACK PANEL + TELEMETRY FUSION
# ============================================================
class LiveSystemFeedbackPanel(QWidget):
    def __init__(self, queen: QueenBrain):
        super().__init__()
        self.queen = queen

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.tabs = QTabWidget()

        self.browser_list = QListWidget()
        self.game_list = QListWidget()
        self.system_list = QListWidget()
        self.network_list = QListWidget()

        def wrap_list_widget(w):
            tab = QWidget()
            l = QVBoxLayout()
            l.setContentsMargins(4, 4, 4, 4)
            l.addWidget(w)
            tab.setLayout(l)
            return tab

        self.tabs.addTab(wrap_list_widget(self.browser_list), "Browser")
        self.tabs.addTab(wrap_list_widget(self.game_list), "Game")
        self.tabs.addTab(wrap_list_widget(self.system_list), "System")
        self.tabs.addTab(wrap_list_widget(self.network_list), "Network")

        layout.addWidget(QLabel("Live System Feedback"))
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_event(self, category: str, text: str):
        ts = time.strftime("%H:%M:%S", time.localtime())
        item = QListWidgetItem(f"[{ts}] {text}")

        # Telemetry fusion: update queen context
        self.queen.update_context_from_telemetry(category)

        if category == "browser":
            self.browser_list.addItem(item)
            self.browser_list.scrollToBottom()
        elif category == "game":
            self.game_list.addItem(item)
            self.game_list.scrollToBottom()
        elif category == "system":
            self.system_list.addItem(item)
            self.system_list.scrollToBottom()
        elif category == "network":
            self.network_list.addItem(item)
            self.network_list.scrollToBottom()


# ============================================================
# THREAT CONE PANEL
# ============================================================
class ThreatConePanel(QWidget):
    def __init__(self, threat_cone: ThreatCone):
        super().__init__()
        self.threat_cone = threat_cone

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Route", "Current", "30s", "60s", "Risk", "Trend Risk"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(QLabel("Threat Cone 2.0"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self):
        projections = self.threat_cone.project_30s()
        self.table.setRowCount(len(projections))
        for row, p in enumerate(projections):
            self.table.setItem(row, 0, QTableWidgetItem(p.route_name))
            self.table.setItem(row, 1, QTableWidgetItem(p.current_color.value))
            self.table.setItem(row, 2, QTableWidgetItem(p.projected_color_30s.value))
            self.table.setItem(row, 3, QTableWidgetItem(p.projected_color_60s.value))
            self.table.setItem(row, 4, QTableWidgetItem(f"{p.risk_score:.2f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{p.trend_risk:.2f}"))


# ============================================================
# OUTCOME PREDICTOR, NEURAL FUSION, ETC.
# ============================================================
def predict_mission_outcome(snapshot: Dict[str, RouteStatus]) -> MissionOutcomePrediction:
    if not snapshot:
        return MissionOutcomePrediction(0.25, 0.25, 0.25, 0.25)

    greens = sum(1 for r in snapshot.values() if r.color == RouteColor.GREEN)
    reds = sum(1 for r in snapshot.values() if r.color == RouteColor.RED)
    total = len(snapshot)

    success = greens / total
    fail = reds / total
    bypass = max(0.0, 0.3 - fail * 0.2)
    helicopter = max(0.0, 1.0 - (success + fail + bypass))
    s = success + fail + bypass + helicopter
    if s == 0:
        return MissionOutcomePrediction(0.25, 0.25, 0.25, 0.25)
    return MissionOutcomePrediction(success / s, bypass / s, fail / s, helicopter / s)


class OutcomePredictorPanel(QWidget):
    def __init__(self, queen: QueenBrain):
        super().__init__()
        self.queen = queen

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.label = QLabel("Mission Outcome Prediction")
        self.label_details = QLabel("")
        layout.addWidget(self.label)
        layout.addWidget(self.label_details)
        self.setLayout(layout)

    def refresh(self):
        snapshot = self.queen.get_snapshot()
        prediction = predict_mission_outcome(snapshot)
        text = (
            f"Success: {prediction.success_prob:.2f} | "
            f"Bypass: {prediction.bypass_prob:.2f} | "
            f"Fail: {prediction.fail_prob:.2f} | "
            f"Helicopter: {prediction.helicopter_prob:.2f}"
        )
        self.label_details.setText(text)


class NeuralFusionPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Neural Fusion Overlay (Future Upgrade Shell)"))
        layout.addWidget(QLabel("This panel will fuse multi-LLM reasoning and route intelligence into a single overlay."))
        self.setLayout(layout)


class AdaptiveWeightsPanel(QWidget):
    def __init__(self, queen: QueenBrain):
        super().__init__()
        self.queen = queen

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.label = QLabel("Adaptive Weights (Bandit Arms)")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Route", "Successes", "Failures"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)

        layout.addWidget(self.label)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh(self):
        arms = self.queen.arms
        self.table.setRowCount(len(arms))
        for row, (name, arm) in enumerate(arms.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(arm.successes)))
            self.table.setItem(row, 2, QTableWidgetItem(str(arm.failures)))


class SchedulerPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Scheduler (Future Upgrade Shell)"))
        layout.addWidget(QLabel("This will manage timed missions, recurring backups, and scheduled reroutes."))
        self.setLayout(layout)


class AutoReroutePanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Auto-Reroute (Future Upgrade Shell)"))
        layout.addWidget(QLabel("This will automatically trigger reroutes when risk or color crosses thresholds."))
        self.setLayout(layout)


class SwarmSyncPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Swarm Sync (Future Upgrade Shell)"))
        layout.addWidget(QLabel("This will sync multiple nodes/agents into a single swarm intelligence."))
        self.setLayout(layout)


class ThemeEnginePanel(QWidget):
    def __init__(self, main_window_ref):
        super().__init__()
        self.main_window_ref = main_window_ref

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Theme Engine"))
        self.combo = QComboBox()
        self.combo.addItems(["Dark Tactical", "Light", "High Contrast"])
        layout.addWidget(QLabel("Select Theme:"))
        layout.addWidget(self.combo)

        self.combo.currentIndexChanged.connect(self._apply_theme)
        self.setLayout(layout)

    def _apply_theme(self, idx: int):
        if idx == 0:
            self.main_window_ref.apply_dark_tactical_theme()
        elif idx == 1:
            self.main_window_ref.apply_light_theme()
        else:
            self.main_window_ref.apply_high_contrast_theme()


class MissionReplayPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Mission Replay (Future Upgrade Shell)"))
        layout.addWidget(QLabel("This will allow replaying past missions with timeline and route decisions."))
        self.setLayout(layout)


class ExportImportPanel(QWidget):
    def __init__(self, queen: QueenBrain, settings: SettingsStore):
        super().__init__()
        self.queen = queen
        self.settings = settings

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self.btn_export = QPushButton("Export Config & State")
        self.btn_import = QPushButton("Import Config & State")
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_import)

        layout.addLayout(btn_layout)
        self.label_status = QLabel("")
        layout.addWidget(self.label_status)
        self.setLayout(layout)

        self.btn_export.clicked.connect(self._export_state)
        self.btn_import.clicked.connect(self._import_state)

    def _export_state(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export State", "", "JSON Files (*.json)")
        if not path:
            return
        data = {
            "settings": self.settings.data,
            "routes": {
                name: {
                    "latency_ms": r.latency_ms,
                    "loss": r.loss,
                    "congestion": r.congestion,
                    "color": r.color.value,
                    "worker_id": r.worker_id,
                    "is_opportunistic": r.is_opportunistic,
                }
                for name, r in self.queen.route_map.items()
            }
        }
        try:
            with JSON_LOCK:
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, path)
            self.label_status.setText(f"Exported to {path}")
        except Exception as e:
            self.label_status.setText(f"Export failed: {e}")

    def _import_state(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import State", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.settings.data = data.get("settings", {})
            self.settings.save()

            routes_data = data.get("routes", {})
            for name, rd in routes_data.items():
                status = RouteStatus(
                    name=name,
                    latency_ms=rd.get("latency_ms", 999.0),
                    loss=rd.get("loss", 100.0),
                    congestion=rd.get("congestion", 100.0),
                    color=RouteColor(rd.get("color", "RED")),
                    worker_id=rd.get("worker_id", ""),
                    is_opportunistic=rd.get("is_opportunistic", False),
                )
                self.queen.update_route(status)

            self.label_status.setText(f"Imported from {path}")
        except Exception as e:
            self.label_status.setText(f"Import failed: {e}")


# ============================================================
# TELEMETRY SERVER (EVENT BUS) – TRUE FUSION
# ============================================================
class TelemetryServer(threading.Thread):
    def __init__(self, port: int, callback):
        super().__init__(daemon=True)
        self.port = port
        self.callback = callback
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", self.port))
        s.listen(5)
        s.settimeout(1.0)

        while self._running:
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                data = b""
                conn.settimeout(1.0)
                while True:
                    try:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    except socket.timeout:
                        break
                try:
                    text = data.decode("utf-8").strip()
                    if not text:
                        continue
                    # Expect JSON lines or single JSON
                    for line in text.splitlines():
                        try:
                            evt = json.loads(line)
                            category = evt.get("category", "system")
                            msg = evt.get("message", "event")
                            self.callback(category, msg)
                        except Exception:
                            continue
                except Exception:
                    continue
        s.close()


# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Heavy Data Swarm Cockpit")
        self.resize(1400, 800)

        self.queen = QueenBrain()
        self.settings = SettingsStore()
        self.backup_engine = BackupEngine(self.queen, self.settings)
        self.forecaster = ShortHorizonForecaster(self.queen.history)
        self.threat_cone = ThreatCone(self.queen.history, self.queen)
        self.preemptive_brain = PreemptiveRerouteBrain(self.queen, self.threat_cone)

        self.thread_pool = QThreadPool()
        self.worker_signals = WorkerSignals()
        self.worker_signals.status_ready.connect(self._on_status_ready)
        self.probes: List[ProbeTask] = []

        self.last_best_route: Optional[str] = None

        central = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)
        central.setLayout(root_layout)
        self.setCentralWidget(central)

        # --- Tab 1: Traffic Monitor ---
        self.tab_traffic = QWidget()
        traffic_layout = QVBoxLayout()
        traffic_layout.setContentsMargins(6, 6, 6, 6)
        traffic_layout.setSpacing(4)

        self.reroute_led = QLabel()
        self.reroute_led.setFixedSize(16, 16)
        self.reroute_led.setStyleSheet("border-radius: 8px; background-color: green;")
        self.reroute_led.setToolTip("Reroute Status")

        led_layout = QHBoxLayout()
        led_layout.addStretch()
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.addWidget(self.reroute_led)
        traffic_layout.addLayout(led_layout)

        self.main_table = UnifiedRouteTable(self.queen, self.queen.history, self.forecaster, self.threat_cone)
        traffic_layout.addWidget(QLabel("Traffic Monitor"))
        traffic_layout.addWidget(self.main_table)
        self.tab_traffic.setLayout(traffic_layout)
        self.tabs.addTab(self.tab_traffic, "Traffic Monitor")

        # --- Tab 2: Route Categories ---
        self.tab_categories = QWidget()
        cat_layout = QVBoxLayout()
        cat_layout.setContentsMargins(6, 6, 6, 6)
        cat_layout.setSpacing(4)

        self.subtabs_categories = QTabWidget()

        self.tab_browser = QWidget()
        browser_layout = QVBoxLayout()
        browser_layout.setContentsMargins(4, 4, 4, 4)
        browser_layout.setSpacing(4)
        self.browser_table = UnifiedRouteTable(self.queen, self.queen.history, self.forecaster, self.threat_cone)
        browser_layout.addWidget(self.browser_table)
        self.tab_browser.setLayout(browser_layout)
        self.subtabs_categories.addTab(self.tab_browser, "Browser")

        self.tab_game = QWidget()
        game_layout = QVBoxLayout()
        game_layout.setContentsMargins(4, 4, 4, 4)
        game_layout.setSpacing(4)
        self.game_table = UnifiedRouteTable(self.queen, self.queen.history, self.forecaster, self.threat_cone)
        game_layout.addWidget(self.game_table)
        self.tab_game.setLayout(game_layout)
        self.subtabs_categories.addTab(self.tab_game, "Game")

        self.tab_system = QWidget()
        system_layout = QVBoxLayout()
        system_layout.setContentsMargins(4, 4, 4, 4)
        system_layout.setSpacing(4)
        self.system_table = UnifiedRouteTable(self.queen, self.queen.history, self.forecaster, self.threat_cone)
        system_layout.addWidget(self.system_table)
        self.tab_system.setLayout(system_layout)
        self.subtabs_categories.addTab(self.tab_system, "System")

        self.tab_network = QWidget()
        network_layout = QVBoxLayout()
        network_layout.setContentsMargins(4, 4, 4, 4)
        network_layout.setSpacing(4)
        self.network_table = UnifiedRouteTable(self.queen, self.queen.history, self.forecaster, self.threat_cone)
        network_layout.addWidget(self.network_table)
        self.tab_network.setLayout(network_layout)
        self.subtabs_categories.addTab(self.tab_network, "Network")

        cat_layout.addWidget(QLabel("Route Categories"))
        cat_layout.addWidget(self.subtabs_categories)
        self.tab_categories.setLayout(cat_layout)
        self.tabs.addTab(self.tab_categories, "Route Categories")

        # --- Other tabs ---
        self.heatmap_panel = HeatmapPanel(self.queen)
        self.tabs.addTab(self.heatmap_panel, "Heatmap")

        self.backup_lane_panel = BackupLanePanel()
        self.tabs.addTab(self.backup_lane_panel, "Backup Lanes")

        self.reroute_flow_panel = RerouteFlowPanel()
        self.tabs.addTab(self.reroute_flow_panel, "Reroute Flow")

        self.mission_timeline_panel = MissionTimelinePanel()
        self.tabs.addTab(self.mission_timeline_panel, "Mission Timeline")

        self.decision_log_panel = DecisionLogPanel(self.queen)
        self.tabs.addTab(self.decision_log_panel, "Decision Log")

        self.backup_manager_panel = BackupManagerPanel(self.settings)
        self.tabs.addTab(self.backup_manager_panel, "Backup Manager")

        self.outcome_predictor_panel = OutcomePredictorPanel(self.queen)
        self.tabs.addTab(self.outcome_predictor_panel, "Predictive Simulator")

        self.neural_fusion_panel = NeuralFusionPanel()
        self.tabs.addTab(self.neural_fusion_panel, "Neural Fusion")

        self.adaptive_weights_panel = AdaptiveWeightsPanel(self.queen)
        self.tabs.addTab(self.adaptive_weights_panel, "Adaptive Weights")

        self.scheduler_panel = SchedulerPanel()
        self.tabs.addTab(self.scheduler_panel, "Scheduler")

        self.threat_cone_panel = ThreatConePanel(self.threat_cone)
        self.tabs.addTab(self.threat_cone_panel, "Threat Cone 2.0")

        self.auto_reroute_panel = AutoReroutePanel()
        self.tabs.addTab(self.auto_reroute_panel, "Auto-Reroute")

        self.swarm_sync_panel = SwarmSyncPanel()
        self.tabs.addTab(self.swarm_sync_panel, "Swarm Sync")

        self.theme_engine_panel = ThemeEnginePanel(self)
        self.tabs.addTab(self.theme_engine_panel, "Theme Engine")

        self.mission_replay_panel = MissionReplayPanel()
        self.tabs.addTab(self.mission_replay_panel, "Mission Replay")

        self.live_feedback_panel = LiveSystemFeedbackPanel(self.queen)
        self.tabs.addTab(self.live_feedback_panel, "Live System Feedback")

        self._start_probes()
        self._start_telemetry_server()

        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self._refresh_all)
        self.refresh_timer.start()

        self.apply_dark_tactical_theme()

    # THEMES
    def apply_dark_tactical_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #101010;
            }
            QWidget {
                background-color: #101010;
                color: #E0E0E0;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #303030;
            }
            QTabBar::tab {
                background: #202020;
                color: #E0E0E0;
                padding: 4px 8px;
            }
            QTabBar::tab:selected {
                background: #404040;
            }
            QTableWidget {
                gridline-color: #404040;
                background-color: #181818;
                alternate-background-color: #202020;
            }
            QHeaderView::section {
                background-color: #202020;
                color: #E0E0E0;
                padding: 2px;
                border: 1px solid #404040;
            }
            QPushButton {
                background-color: #303030;
                color: #E0E0E0;
                border: 1px solid #505050;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """)

    def apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F0F0;
            }
            QWidget {
                background-color: #F0F0F0;
                color: #202020;
                font-size: 10pt;
            }
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8F8F8;
            }
            QHeaderView::section {
                background-color: #E0E0E0;
                color: #202020;
                padding: 2px;
                border: 1px solid #C0C0C0;
            }
        """)

    def apply_high_contrast_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
                background-color: #000000;
                color: #FFFFFF;
                font-size: 11pt;
            }
            QTableWidget {
                background-color: #000000;
                alternate-background-color: #202020;
                gridline-color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #000000;
                color: #FFFFFF;
                padding: 2px;
                border: 1px solid #FFFFFF;
            }
            QPushButton {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #202020;
            }
        """)

    # WORKERS
    def _start_probes(self):
        routes = [
            "INT_LocalDisk",
            "INT_CPUPath",
            "EXT_SMB1",
            "EXT_SMB2",
            "EXT_CloudRelay",
        ]
        for idx, rname in enumerate(routes):
            worker_id = f"W{idx+1}"
            task = ProbeTask(worker_id, rname, interval=2.0, signals=self.worker_signals)
            self.probes.append(task)
            self.thread_pool.start(task)

    def _on_status_ready(self, status: RouteStatus):
        self.queen.update_route(status)

        snapshot = self.queen.get_snapshot()

        # Preemptive reroute brain (telemetry + stance fused)
        preemptive_route = self.preemptive_brain.maybe_preemptive_reroute()
        if preemptive_route:
            self.queen.pinned_route = preemptive_route.name

        # If pinned route exists and is still acceptable, prefer it
        if self.queen.pinned_route and self.queen.pinned_route in snapshot:
            best = snapshot[self.queen.pinned_route]
        else:
            best = choose_best_route_with_stance(snapshot, self.queen.stance, self.queen)

        if best:
            if self.last_best_route is None:
                self.last_best_route = best.name
            elif best.name != self.last_best_route:
                self.reroute_flow_panel.add_event(self.last_best_route, best.name, "Auto selection change")
                self.last_best_route = best.name
                self.reroute_led.setStyleSheet("border-radius: 8px; background-color: red;")

            if best.color == RouteColor.GREEN:
                self.reroute_led.setStyleSheet("border-radius: 8px; background-color: green;")

    def _start_telemetry_server(self):
        def cb(category, message):
            # category: browser/game/system/network
            self.live_feedback_panel.add_event(category, message)

        self.telemetry_server = TelemetryServer(port=5555, callback=cb)
        self.telemetry_server.start()

    # REFRESH LOOP
    def _refresh_all(self):
        routes = self.queen.get_snapshot()
        forecasts = self.forecaster.forecast_routes(routes)

        self.main_table.refresh()

        browser_routes = {k: v for k, v in routes.items() if "Cloud" in k or "Browser" in k or "EXT" in k}
        game_routes = {k: v for k, v in routes.items() if "SMB" in k or "Game" in k}
        system_routes = {k: v for k, v in routes.items() if "INT" in k}
        network_routes = {k: v for k, v in routes.items() if "EXT" in k or "Cloud" in k}

        original_map = self.queen.route_map

        self.queen.route_map = browser_routes
        self.browser_table.refresh()

        self.queen.route_map = game_routes
        self.game_table.refresh()

        self.queen.route_map = system_routes
        self.system_table.refresh()

        self.queen.route_map = network_routes
        self.network_table.refresh()

        self.queen.route_map = original_map

        self.heatmap_panel.refresh()
        self.backup_lane_panel.update_lanes(routes, forecasts)
        self.threat_cone_panel.refresh()
        self.outcome_predictor_panel.refresh()
        self.adaptive_weights_panel.refresh()
        self.decision_log_panel.refresh()

    def closeEvent(self, event):
        for p in self.probes:
            p.stop()
        if hasattr(self, "telemetry_server"):
            self.telemetry_server.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

