import sys
import os
import time
import json
import threading
import socket
import traceback
import re
import random
import hashlib
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import psutil
import win32gui
import win32process
from pynput import mouse, keyboard

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QFileDialog, QMessageBox, QHBoxLayout,
    QListWidget, QListWidgetItem, QTabWidget, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtGui import QColor

# ============================================================
# EVENT BUS CONFIG
# ============================================================

TCP_HOST = "127.0.0.1"
TCP_PORT = 9009
MAX_TIMELINE_ROWS = 600

DEFAULT_CONFIG = {
    "mode": "strict",
    "firewall": {
        "block_patterns": [
            r"ignore\s+all\s+previous",
            r"system\s*:",
            r"assistant\s*:",
            r"<script",
            r"data:",
            r"base64",
        ],
    },
}

config = DEFAULT_CONFIG.copy()

clients_lock = threading.Lock()
clients: List[Tuple[socket.socket, Any]] = []


def broadcast_event(event: dict):
    data = json.dumps(event, ensure_ascii=False) + "\n"
    dead = []
    with clients_lock:
        for s, f in clients:
            try:
                f.write(data)
                f.flush()
            except Exception:
                dead.append((s, f))
        for d in dead:
            try:
                d[0].close()
            except Exception:
                pass
            if d in clients:
                clients.remove(d)


def handle_bus_client(sock: socket.socket, addr):
    f_in = sock.makefile("r", encoding="utf-8")
    f_out = sock.makefile("w", encoding="utf-8")
    with clients_lock:
        clients.append((sock, f_out))
    try:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                traceback.print_exc()
                continue
            broadcast_event(event)
    finally:
        with clients_lock:
            clients[:] = [c for c in clients if c[0] is not sock]
        try:
            sock.close()
        except Exception:
            pass


def bus_server_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((TCP_HOST, TCP_PORT))
    srv.listen(10)
    while True:
        try:
            sock, addr = srv.accept()
            threading.Thread(target=handle_bus_client, args=(sock, addr), daemon=True).start()
        except Exception:
            traceback.print_exc()
            time.sleep(1)


class EventBus:
    def __init__(self):
        self.callbacks: List[Any] = []

    def emit(self, event: dict):
        for cb in self.callbacks:
            try:
                cb(event)
            except Exception:
                traceback.print_exc()


bus = EventBus()


def tcp_client_loop():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TCP_HOST, TCP_PORT))
            f = sock.makefile("r", encoding="utf-8")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    bus.emit(event)
                except Exception:
                    traceback.print_exc()
            f.close()
            sock.close()
        except Exception:
            time.sleep(2)


class EventProducer:
    def __init__(self):
        self.sock: Optional[socket.socket] = None
        self.file = None

    def connect(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((TCP_HOST, TCP_PORT))
                self.file = self.sock.makefile("w", encoding="utf-8")
                return
            except Exception:
                time.sleep(2)

    def send_event(self, event: dict):
        if self.file is None:
            self.connect()
        try:
            self.file.write(json.dumps(event) + "\n")
            self.file.flush()
        except Exception:
            try:
                self.file.close()
                self.sock.close()
            except Exception:
                pass
            self.file = None
            self.sock = None
            self.connect()


producer = EventProducer()


def get_active_window_info() -> Tuple[str, str]:
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(pid).name()
        return title, proc
    except Exception:
        return "", ""


def build_event(event_type: str, raw_text: str, source_name: str, extra: Optional[dict] = None) -> dict:
    ts = time.time()
    event = {
        "timestamp": ts,
        "raw": raw_text,
        "clean": raw_text,
        "hidden": False,
        "decision": "allow",
        "risk": 0,
        "domain": "",
        "source": {"name": source_name},
        "meta": {"type": event_type},
    }
    if extra:
        event["meta"].update(extra)
    return event


def monitor_window_focus():
    producer.connect()
    last_title, last_proc = "", ""
    while True:
        title, proc = get_active_window_info()
        if title != last_title or proc != last_proc:
            last_title, last_proc = title, proc
            raw = f"WindowFocus: {proc} | {title}"
            event = build_event("window_focus", raw, proc, extra={"window_title": title})
            producer.send_event(event)
        time.sleep(0.1)


def on_click(x, y, button, pressed):
    if not pressed:
        return
    title, proc = get_active_window_info()
    raw = f"Click: {proc} | {title} | {button} @ ({x},{y})"
    event = build_event(
        "mouse_click",
        raw,
        proc,
        extra={"x": x, "y": y, "button": str(button), "window_title": title},
    )
    producer.send_event(event)


def on_press(key):
    title, proc = get_active_window_info()
    try:
        key_str = key.char
    except Exception:
        key_str = str(key)
    raw = f"Keypress: {proc} | {title} | {key_str}"
    event = build_event(
        "key_press",
        raw,
        proc,
        extra={"key": key_str, "window_title": title},
    )
    producer.send_event(event)


def start_producer_hooks():
    threading.Thread(target=monitor_window_focus, daemon=True).start()
    mouse.Listener(on_click=on_click).start()
    keyboard.Listener(on_press=on_press).start()

# ============================================================
# CORE ENUMS, DATA, HELPERS
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
    is_opportunistic: bool = False
    last_updated: float = field(default_factory=time.time)
    worker_id: str = ""


@dataclass
class MissionTimeline:
    events: List[Tuple[float, str, RouteColor]] = field(default_factory=list)

    def record(self, label: str, color: RouteColor):
        self.events.append((time.time(), label, color))


@dataclass
class BackupMission:
    mission_id: str
    source_path: str
    target_path: str
    chosen_route: Optional[RouteStatus] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    success: Optional[bool] = None
    message: str = ""
    timeline: MissionTimeline = field(default_factory=MissionTimeline)


@dataclass
class MissionOutcomePrediction:
    success_prob: float
    bypass_prob: float
    fail_prob: float
    helicopter_prob: float


def classify_route(latency, loss, congestion) -> RouteColor:
    score = latency * 0.4 + loss * 0.4 + congestion * 0.2
    if score < 20:
        return RouteColor.GREEN
    elif score < 40:
        return RouteColor.YELLOW
    elif score < 60:
        return RouteColor.ORANGE
    else:
        return RouteColor.RED


def ping_host(host: str, count: int = 2) -> tuple[float, float]:
    cmd = ["ping", "-n", str(count), host]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    loss_match = re.search(r"Lost = (\d+)", output)
    lost = int(loss_match.group(1)) if loss_match else count

    avg_match = re.search(r"Average = (\d+)ms", output)
    avg = float(avg_match.group(1)) if avg_match else 999.0

    return avg, lost / count


def get_congestion() -> float:
    net = psutil.net_io_counters()
    return min((net.bytes_sent + net.bytes_recv) / 1e8, 1.0)


def predict_mission_outcome(route_map: Dict[str, RouteStatus]) -> MissionOutcomePrediction:
    greens = sum(1 for r in route_map.values() if r.color == RouteColor.GREEN)
    yellows = sum(1 for r in route_map.values() if r.color == RouteColor.YELLOW)
    oranges = sum(1 for r in route_map.values() if r.color == RouteColor.ORANGE)
    reds = sum(1 for r in route_map.values() if r.color == RouteColor.RED)

    total = max(len(route_map), 1)

    success_prob = (greens + yellows * 0.5) / total
    bypass_prob = oranges / total
    fail_prob = reds / total
    helicopter_prob = 1.0 if reds == total else reds / total * 0.5

    return MissionOutcomePrediction(
        success_prob=success_prob,
        bypass_prob=bypass_prob,
        fail_prob=fail_prob,
        helicopter_prob=helicopter_prob,
    )

# ============================================================
# HISTORY & FORECAST
# ============================================================

@dataclass
class RouteSample:
    timestamp: float
    latency_ms: float
    loss: float
    congestion: float
    color: RouteColor


class RouteHistory:
    def __init__(self, max_samples: int = 200):
        self.max_samples = max_samples
        self.history: Dict[str, List[RouteSample]] = {}

    def add_sample(self, status: RouteStatus):
        lst = self.history.setdefault(status.name, [])
        lst.append(
            RouteSample(
                timestamp=status.last_updated,
                latency_ms=status.latency_ms,
                loss=status.loss,
                congestion=status.congestion,
                color=status.color,
            )
        )
        if len(lst) > self.max_samples:
            del lst[0 : len(lst) - self.max_samples]

    def get_history(self, route_name: str) -> List[RouteSample]:
        return self.history.get(route_name, [])

    def exp_smooth(self, values: List[float], alpha: float = 0.3) -> Optional[float]:
        if not values:
            return None
        s = values[0]
        for v in values[1:]:
            s = alpha * v + (1 - alpha) * s
        return s

    def anomaly_score(self, values: List[float]) -> float:
        if len(values) < 5:
            return 0.0
        import statistics

        mean = statistics.mean(values)
        stdev = statistics.pstdev(values) or 1.0
        latest = values[-1]
        z = abs(latest - mean) / stdev
        return min(z / 3.0, 1.0)

    def trend(self, values: List[float]) -> float:
        if len(values) < 3:
            return 0.0
        xs = list(range(len(values)))
        mean_x = sum(xs) / len(xs)
        mean_y = sum(values) / len(values)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = num / den
        return slope

    def volatility(self, values: List[float]) -> float:
        if len(values) < 5:
            return 0.0
        import statistics

        return statistics.pstdev(values)

    def stability_score(self, colors: List[RouteColor]) -> float:
        if len(colors) < 3:
            return 1.0
        flips = 0
        for a, b in zip(colors, colors[1:]):
            if a != b:
                flips += 1
        flip_rate = flips / (len(colors) - 1)
        return max(0.0, 1.0 - flip_rate)


@dataclass
class RouteForecast:
    route_name: str
    forecast_latency: float
    forecast_loss: float
    forecast_congestion: float
    forecast_color: RouteColor
    health_score: float


class ShortHorizonForecaster:
    def __init__(self, history: RouteHistory):
        self.history = history

    def forecast_routes(self, route_map: Dict[str, RouteStatus]) -> Dict[str, RouteForecast]:
        forecasts: Dict[str, RouteForecast] = {}
        for name, status in route_map.items():
            hist = self.history.get_history(name)
            lat_vals = [s.latency_ms for s in hist]
            loss_vals = [s.loss for s in hist]
            cong_vals = [s.congestion for s in hist]
            color_vals = [s.color for s in hist]

            f_lat = self.history.exp_smooth(lat_vals) or status.latency_ms
            f_loss = self.history.exp_smooth(loss_vals) or status.loss
            f_cong = self.history.exp_smooth(cong_vals) or status.congestion

            lat_trend = self.history.trend(lat_vals)
            loss_trend = self.history.trend(loss_vals)
            vol_lat = self.history.volatility(lat_vals)
            vol_loss = self.history.volatility(loss_vals)
            stability = self.history.stability_score(color_vals)

            trend_penalty = max(0.0, lat_trend * 0.1 + loss_trend * 0.2)
            volatility_penalty = (vol_lat * 0.02 + vol_loss * 0.05)
            stability_bonus = (1.0 - stability) * 10.0

            base_health = f_lat * 0.4 + f_loss * 0.4 + f_cong * 0.2
            health = base_health + trend_penalty + volatility_penalty + stability_bonus

            f_color = classify_route(f_lat, f_loss, f_cong)

            forecasts[name] = RouteForecast(
                route_name=name,
                forecast_latency=f_lat,
                forecast_loss=f_loss,
                forecast_congestion=f_cong,
                forecast_color=f_color,
                health_score=health,
            )
        return forecasts

# ============================================================
# THREAT CONE
# ============================================================

@dataclass
class RouteThreatProjection:
    route_name: str
    current_color: RouteColor
    projected_color_30s: RouteColor
    projected_color_60s: RouteColor
    risk_score: float
    anomaly: float
    trend_risk: float


class ThreatCone:
    def __init__(self, history: RouteHistory, queen_ref):
        self.history = history
        self.queen_ref = queen_ref

    def project_30s(self) -> List[RouteThreatProjection]:
        snapshot = self.queen_ref.get_snapshot()
        projections: List[RouteThreatProjection] = []

        for name, status in snapshot.items():
            hist = self.history.get_history(name)
            lat_vals = [s.latency_ms for s in hist]
            loss_vals = [s.loss for s in hist]
            cong_vals = [s.congestion for s in hist]

            smoothed_lat = self.history.exp_smooth(lat_vals) or status.latency_ms
            smoothed_loss = self.history.exp_smooth(loss_vals) or status.loss
            smoothed_cong = self.history.exp_smooth(cong_vals) or status.congestion

            anomaly = max(
                self.history.anomaly_score(lat_vals),
                self.history.anomaly_score(loss_vals),
                self.history.anomaly_score(cong_vals),
            )

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

            trend_risk = max(0.0, (lat_trend * 0.1 + loss_trend * 0.2))
            base_risk = score_30 / 100.0
            risk = min(1.0, base_risk + anomaly * 0.3 + trend_risk)

            projections.append(
                RouteThreatProjection(
                    route_name=name,
                    current_color=status.color,
                    projected_color_30s=proj_30,
                    projected_color_60s=proj_60,
                    risk_score=risk,
                    anomaly=anomaly,
                    trend_risk=trend_risk,
                )
            )
        return projections

# ============================================================
# QUEEN BRAIN & STANCE
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
class PreemptiveRerouteDecision:
    timestamp: float
    from_route: Optional[str]
    to_route: str
    reason: str
    projected_risk: float
    projected_color_30s: RouteColor
    projected_color_60s: RouteColor


@dataclass
class PreemptiveDecisionEntry:
    timestamp: float
    from_route: str
    to_route: str
    reason: str
    projected_risk: float
    projected_color_30s: str
    projected_color_60s: str


class QueenDecisionLog:
    def __init__(self):
        self.entries: List[Any] = []

    def record_decision(self, chosen: RouteStatus, all_routes: Dict[str, RouteStatus]):
        reason = (
            f"Selected {chosen.name} because it had color {chosen.color.value} "
            f"and lowest latency among acceptable routes."
        )
        self.entries.append(
            DecisionEntry(
                timestamp=time.time(),
                chosen_route=chosen.name,
                reason=reason,
            )
        )

    def record_override(self, original: RouteStatus, new: RouteStatus):
        justification = (
            f"User override: replacing {original.name} ({original.color.value}) "
            f"with {new.name} ({new.color.value})."
        )
        self.entries.append(
            OverrideDecision(
                timestamp=time.time(),
                original_route=original.name,
                overridden_route=new.name,
                justification=justification,
            )
        )

    def record_preemptive(self, decision: PreemptiveRerouteDecision):
        self.entries.append(
            PreemptiveDecisionEntry(
                timestamp=decision.timestamp,
                from_route=decision.from_route or "(none)",
                to_route=decision.to_route,
                reason=decision.reason,
                projected_risk=decision.projected_risk,
                projected_color_30s=decision.projected_color_30s.value,
                projected_color_60s=decision.projected_color_60s.value,
            )
        )


class QueenBrain:
    def __init__(self):
        self.route_map: Dict[str, RouteStatus] = {}
        self.history = RouteHistory()
        self.log = QueenDecisionLog()
        self.stance = QueenStance.BALANCED
        self.arms: Dict[str, RouteArm] = {}

    def update_route(self, status: RouteStatus):
        self.route_map[status.name] = status
        self.history.add_sample(status)

    def get_snapshot(self) -> Dict[str, RouteStatus]:
        return dict(self.route_map)


def choose_best_route_with_stance(
    route_map: Dict[str, RouteStatus],
    stance: QueenStance,
    queen: QueenBrain,
) -> Optional[RouteStatus]:
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
# PREEMPTIVE REROUTE BRAIN
# ============================================================

class PreemptiveRerouteBrain:
    def __init__(
        self,
        queen: QueenBrain,
        threat_cone: ThreatCone,
        risk_threshold: float = 0.65,
        color_escalation_only: bool = True,
        min_improvement_ms: float = 5.0,
    ):
        self.queen = queen
        self.threat_cone = threat_cone
        self.risk_threshold = risk_threshold
        self.color_escalation_only = color_escalation_only
        self.min_improvement_ms = min_improvement_ms
        self.last_decision: Optional[PreemptiveRerouteDecision] = None

        self.cooldown_seconds = 20.0
        self.last_action_time = 0.0

    def _stance_thresholds(self):
        if self.queen.stance == QueenStance.CONSERVATIVE:
            return {
                "risk": 0.50,
                "latency_gain": 2.0,
                "color_escalation": True,
            }
        elif self.queen.stance == QueenStance.BEAST:
            return {
                "risk": 0.80,
                "latency_gain": 10.0,
                "color_escalation": False,
            }
        else:  # Balanced
            return {
                "risk": 0.65,
                "latency_gain": 5.0,
                "color_escalation": True,
            }

    def evaluate(self, current_best: Optional[RouteStatus]) -> Optional[PreemptiveRerouteDecision]:
        now = time.time()
        if now - self.last_action_time < self.cooldown_seconds:
            return None

        snapshot = self.queen.get_snapshot()
        if not snapshot:
            return None

        projections = {p.route_name: p for p in self.threat_cone.project_30s()}

        if current_best is None:
            return None

        current_proj = projections.get(current_best.name)
        if not current_proj:
            return None

        t = self._stance_thresholds()

        if current_proj.risk_score < t["risk"]:
            return None

        if t["color_escalation"]:
            if current_proj.projected_color_30s.value <= current_best.color.value:
                return None

        candidates: List[Tuple[float, RouteStatus]] = []
        for name, status in snapshot.items():
            if name == current_best.name:
                continue
            proj = projections.get(name)
            if not proj:
                continue

            if proj.risk_score < current_proj.risk_score:
                score = proj.risk_score * 100.0 + status.latency_ms * 0.5
                candidates.append((score, status))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        _, better_route = candidates[0]

        if current_best.latency_ms - better_route.latency_ms < t["latency_gain"]:
            return None

        decision = PreemptiveRerouteDecision(
            timestamp=time.time(),
            from_route=current_best.name,
            to_route=better_route.name,
            reason=(
                f"Projected risk for {current_best.name} = {current_proj.risk_score:.2f}, "
                f"better alternative {better_route.name} found with lower projected risk."
            ),
            projected_risk=current_proj.risk_score,
            projected_color_30s=current_proj.projected_color_30s,
            projected_color_60s=current_proj.projected_color_60s,
        )
        self.last_decision = decision
        self.last_action_time = time.time()
        return decision

# ============================================================
# SETTINGS & DELTA WRITER
# ============================================================

JSON_LOCK = threading.Lock()


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
        route = choose_best_route_with_stance(self.queen.route_map, self.queen.stance, self.queen)
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
                results.append(
                    {
                        "type": "Local",
                        "path": dst_local,
                        "status": status,
                        "message": msg,
                        "timestamp": time.time(),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "type": "Local",
                        "path": local_path,
                        "status": "ERROR",
                        "message": str(e),
                        "timestamp": time.time(),
                    }
                )

        if smb_path:
            try:
                dst_smb = os.path.join(smb_path, f"{mission.mission_id}_backup.json")
                changed, msg = self.delta.write_if_changed(data_to_write, dst_smb)
                status = "SUCCESS" if changed else "SKIPPED"
                results.append(
                    {
                        "type": "SMB",
                        "path": dst_smb,
                        "status": status,
                        "message": msg,
                        "timestamp": time.time(),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "type": "SMB",
                        "path": smb_path,
                        "status": "ERROR",
                        "message": str(e),
                        "timestamp": time.time(),
                    }
                )

        mission.message += "\n\nDual-write results:\n" + "\n".join(
            f"{r['type']} -> {r['status']}: {r['message']}" for r in results
        )

        update_reinforcement(self.queen, mission)
        return mission, results

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
            "EXT_CloudRelay": "1.1.1.1",
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
                worker_id=self.worker_id,
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
        self.table.setHorizontalHeaderLabels(
            [
                "Route",
                "Current Color",
                "Forecast Color",
                "Latency",
                "Loss",
                "Congestion",
                "Health Score",
            ]
        )
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
        self.events.append(
            {
                "timestamp": ts,
                "from": from_route or "(none)",
                "to": to_route,
                "reason": reason,
            }
        )
        if len(self.events) > 50:
            self.events = self.events[-50:]
        self._refresh()

    def _refresh(self):
        self.table.setRowCount(len(self.events))
        for row, e in enumerate(self.events):
            self.table.setItem(
                row,
                0,
                QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))),
            )
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
            self.table.setItem(
                row,
                0,
                QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))),
            )
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
            elif isinstance(entry, PreemptiveDecisionEntry):
                route = f"{entry.from_route} -> {entry.to_route}"
                reason = (
                    f"{entry.reason} "
                    f"(risk={entry.projected_risk:.2f}, "
                    f"30s={entry.projected_color_30s}, "
                    f"60s={entry.projected_color_60s})"
                )
            else:
                route = "Unknown"
                reason = str(entry)
            self.table.setItem(
                row,
                0,
                QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))),
            )
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
        self.table.setHorizontalHeaderLabels(
            ["Path Type", "Path", "Last Write", "Status", "Message"]
        )
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
            self.table.setItem(
                row,
                2,
                QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(ts))),
            )
            self.table.setItem(row, 3, QTableWidgetItem(h.get("status", "")))
            self.table.setItem(row, 4, QTableWidgetItem(h.get("message", "")))


class UnifiedRouteTable(QTableWidget):
    def __init__(
        self,
        queen: QueenBrain,
        history: RouteHistory,
        forecaster: ShortHorizonForecaster,
        threat_cone: ThreatCone,
        parent=None,
    ):
        super().__init__(0, 12, parent)
        self.queen = queen
        self.history = history
        self.forecaster = forecaster
        self.threat_cone = threat_cone

        self.setHorizontalHeaderLabels(
            [
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
                "Mission Quality",
            ]
        )

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
# LIVE SYSTEM FEEDBACK PANEL (EVENT BUS INTEGRATION)
# ============================================================

class LiveSystemFeedbackPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.tabs = QTabWidget()

        self.browser_list = QListWidget()
        self.game_list = QListWidget()
        self.system_list = QListWidget()
        self.network_list = QListWidget()

        browser_tab = QWidget()
        bl = QVBoxLayout()
        bl.setContentsMargins(4, 4, 4, 4)
        bl.addWidget(self.browser_list)
        browser_tab.setLayout(bl)

        game_tab = QWidget()
        gl = QVBoxLayout()
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addWidget(self.game_list)
        game_tab.setLayout(gl)

        system_tab = QWidget()
        sl = QVBoxLayout()
        sl.setContentsMargins(4, 4, 4, 4)
        sl.addWidget(self.system_list)
        system_tab.setLayout(sl)

        network_tab = QWidget()
        nl = QVBoxLayout()
        nl.setContentsMargins(4, 4, 4, 4)
        nl.addWidget(self.network_list)
        network_tab.setLayout(nl)

        self.tabs.addTab(browser_tab, "Browser")
        self.tabs.addTab(game_tab, "Game")
        self.tabs.addTab(system_tab, "System")
        self.tabs.addTab(network_tab, "Network")

        layout.addWidget(QLabel("Live System Feedback"))
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_event(self, category: str, text: str):
        ts = time.strftime("%H:%M:%S", time.localtime())
        item = QListWidgetItem(f"[{ts}] {text}")
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
        self.table.setHorizontalHeaderLabels(
            ["Route", "Current", "30s", "60s", "Risk", "Trend Risk"]
        )
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
        layout.addWidget(
            QLabel(
                "This panel will fuse multi-LLM reasoning and route intelligence into a single overlay."
            )
        )
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
        layout.addWidget(
            QLabel("This will manage timed missions, recurring backups, and scheduled reroutes.")
        )
        self.setLayout(layout)


class AutoReroutePanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Auto-Reroute (Future Upgrade Shell)"))
        layout.addWidget(
            QLabel("This will automatically trigger reroutes when risk or color crosses thresholds.")
        )
        self.setLayout(layout)


class SwarmSyncPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Swarm Sync (Future Upgrade Shell)"))
        layout.addWidget(
            QLabel("This will sync multiple nodes/agents into a single swarm intelligence.")
        )
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
        layout.addWidget(
            QLabel("This will allow replaying past missions with timeline and route decisions.")
        )
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
            },
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
# MAIN WINDOW (FUSED WITH EVENT BUS)
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
        self.preemptive_reroute_brain = PreemptiveRerouteBrain(
            self.queen,
            self.threat_cone,
            risk_threshold=0.65,
            color_escalation_only=True,
            min_improvement_ms=5.0,
        )

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

        self.main_table = UnifiedRouteTable(
            self.queen, self.queen.history, self.forecaster, self.threat_cone
        )
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
        self.browser_table = UnifiedRouteTable(
            self.queen, self.queen.history, self.forecaster, self.threat_cone
        )
        browser_layout.addWidget(self.browser_table)
        self.tab_browser.setLayout(browser_layout)
        self.subtabs_categories.addTab(self.tab_browser, "Browser")

        self.tab_game = QWidget()
        game_layout = QVBoxLayout()
        game_layout.setContentsMargins(4, 4, 4, 4)
        game_layout.setSpacing(4)
        self.game_table = UnifiedRouteTable(
            self.queen, self.queen.history, self.forecaster, self.threat_cone
        )
        game_layout.addWidget(self.game_table)
        self.tab_game.setLayout(game_layout)
        self.subtabs_categories.addTab(self.tab_game, "Game")

        self.tab_system = QWidget()
        system_layout = QVBoxLayout()
        system_layout.setContentsMargins(4, 4, 4, 4)
        system_layout.setSpacing(4)
        self.system_table = UnifiedRouteTable(
            self.queen, self.queen.history, self.forecaster, self.threat_cone
        )
        system_layout.addWidget(self.system_table)
        self.tab_system.setLayout(system_layout)
        self.subtabs_categories.addTab(self.tab_system, "System")

        self.tab_network = QWidget()
        network_layout = QVBoxLayout()
        network_layout.setContentsMargins(4, 4, 4, 4)
        network_layout.setSpacing(4)
        self.network_table = UnifiedRouteTable(
            self.queen, self.queen.history, self.forecaster, self.threat_cone
        )
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

        self.export_import_panel = ExportImportPanel(self.queen, self.settings)
        self.tabs.addTab(self.export_import_panel, "Export / Import")

        self.live_feedback_panel = LiveSystemFeedbackPanel()
        self.tabs.addTab(self.live_feedback_panel, "Live System Feedback")

        # Subscribe to event bus
        bus.callbacks.append(self._on_bus_event)

        self._start_probes()

        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self._refresh_all)
        self.refresh_timer.start()

        self.apply_dark_tactical_theme()

    # THEMES
    def apply_dark_tactical_theme(self):
        self.setStyleSheet(
            """
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
        """
        )

    def apply_light_theme(self):
        self.setStyleSheet(
            """
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
        """
        )

    def apply_high_contrast_theme(self):
        self.setStyleSheet(
            """
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
        """
        )

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
        best = choose_best_route_with_stance(snapshot, self.queen.stance, self.queen)

        preemptive_decision = self.preemptive_reroute_brain.evaluate(best)
        if preemptive_decision is not None:
            self.reroute_flow_panel.add_event(
                preemptive_decision.from_route,
                preemptive_decision.to_route,
                f"Preemptive reroute: {preemptive_decision.reason}",
            )
            self.reroute_led.setStyleSheet(
                "border-radius: 8px; background-color: orange;"
            )
            self.queen.log.record_preemptive(preemptive_decision)
            best = snapshot.get(preemptive_decision.to_route, best)

        if best:
            if self.last_best_route is None:
                self.last_best_route = best.name
            elif best.name != self.last_best_route:
                self.reroute_flow_panel.add_event(
                    self.last_best_route, best.name, "Auto selection change"
                )
                self.last_best_route = best.name
                self.reroute_led.setStyleSheet(
                    "border-radius: 8px; background-color: red;"
                )

            if best.color == RouteColor.GREEN:
                self.reroute_led.setStyleSheet(
                    "border-radius: 8px; background-color: green;"
                )

    # EVENT BUS HANDLER
    def _on_bus_event(self, event: dict):
        raw = event.get("raw", "")
        meta = event.get("meta", {})
        etype = meta.get("type", "")
        source_name = event.get("source", {}).get("name", "") or ""
        text = raw

        proc_lower = source_name.lower()
        title = meta.get("window_title", "") or ""
        title_lower = title.lower()

        category = "system"
        if any(b in proc_lower for b in ["chrome", "edge", "firefox", "brave", "opera"]):
            category = "browser"
        elif any(g in proc_lower for g in ["steam", "game", "elden", "fortnite", "valorant"]):
            category = "game"
        elif etype in ("window_focus", "key_press", "mouse_click"):
            category = "system"
        elif "network" in proc_lower or "vpn" in proc_lower:
            category = "network"

        self.live_feedback_panel.add_event(category, text)

    # REFRESH LOOP
    def _refresh_all(self):
        routes = self.queen.get_snapshot()
        forecasts = self.forecaster.forecast_routes(routes)

        self.main_table.refresh()

        browser_routes = {
            k: v
            for k, v in routes.items()
            if "Cloud" in k or "Browser" in k or "EXT" in k
        }
        game_routes = {
            k: v for k, v in routes.items() if "SMB" in k or "Game" in k
        }
        system_routes = {k: v for k, v in routes.items() if "INT" in k}
        network_routes = {
            k: v for k, v in routes.items() if "EXT" in k or "Cloud" in k
        }

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
        super().closeEvent(event)

# ============================================================
# MAIN
# ============================================================

def main():
    threading.Thread(target=bus_server_loop, daemon=True).start()
    start_producer_hooks()
    threading.Thread(target=tcp_client_loop, daemon=True).start()

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

