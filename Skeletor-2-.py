from __future__ import annotations
import json
import math
import os
import random
import threading
import time
import traceback
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Tuple, Callable, Optional

# =========================
# Optional psutil loader (real system metrics)
# =========================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# =========================
# Utilities: ring buffer, atomic write, structured logging
# =========================
def atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

def log_jsonl(path: str, event: dict) -> None:
    line = json.dumps(event)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[float] = []
    def append(self, x: float):
        self.data.append(x)
        if len(self.data) > self.capacity:
            excess = len(self.data) - self.capacity
            self.data = self.data[excess:]
    def to_list(self) -> List[float]:
        return list(self.data)
    def tail(self, n: int) -> List[float]:
        return self.data[-n:] if n < len(self.data) else list(self.data)
    def __len__(self): return len(self.data)

# =========================
# Core model with multi-valued illumination
# =========================
class Illumination(Enum):
    DARK = 0
    DIM = 1
    AMBIENT = 2
    BRIGHT = 3
    LIGHT = 4

@dataclass
class Thresholds:
    low_on: float = 0.30
    mid_on: float = 0.50
    high_on: float = 0.70
    low_off: float = 0.25
    mid_off: float = 0.45
    high_off: float = 0.65
    def clamp(self):
        for name in ("low_on","mid_on","high_on","low_off","mid_off","high_off"):
            val = getattr(self, name)
            setattr(self, name, max(0.0, min(1.0, val)))

@dataclass
class Signal:
    intensity: float
    def update(self, noise: float) -> None:
        self.intensity = min(1.0, max(0.0, self.intensity + random.uniform(-noise, noise)))
    def classify(self, prev: Illumination, t: Thresholds) -> Illumination:
        x = self.intensity
        if prev in (Illumination.LIGHT, Illumination.BRIGHT):
            if x <= t.high_off:
                if x <= t.mid_off:
                    return Illumination.AMBIENT if x > t.low_off else Illumination.DARK
                return Illumination.DIM
            return Illumination.LIGHT if x >= t.high_on else Illumination.BRIGHT
        elif prev in (Illumination.DARK, Illumination.DIM):
            if x >= t.mid_on:
                return Illumination.LIGHT if x >= t.high_on else Illumination.BRIGHT
            return Illumination.DARK if x <= t.low_off else (Illumination.DIM if x <= t.mid_off else Illumination.AMBIENT)
        else:
            if x >= t.high_on: return Illumination.LIGHT
            if x >= t.mid_on: return Illumination.BRIGHT
            if x <= t.low_off: return Illumination.DARK
            if x <= t.mid_off: return Illumination.DIM
            return Illumination.AMBIENT

# =========================
# Strategies (mutation/reversion)
# =========================
@dataclass
class StrategyPack:
    name: str
    vote: Callable[[List[Illumination]], Illumination]
    parity: Callable[[List[Illumination]], bool]
    blend: Callable[[Illumination, Illumination, float], Illumination]

def vote_majority(states: List[Illumination]) -> Illumination:
    counts: Dict[Illumination, int] = {s: 0 for s in Illumination}
    for s in states: counts[s] += 1
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], -abs(kv[0].value - Illumination.AMBIENT.value)), reverse=True)
    return ordered[0][0]

def vote_weighted(states: List[Illumination]) -> Illumination:
    weights = {Illumination.DARK: 1.0, Illumination.DIM: 1.1, Illumination.AMBIENT: 1.3, Illumination.BRIGHT: 1.15, Illumination.LIGHT: 1.1}
    score = {s: 0.0 for s in Illumination}
    for s in states: score[s] += weights[s]
    ordered = sorted(score.items(), key=lambda kv: (kv[1], -abs(kv[0].value - Illumination.AMBIENT.value)), reverse=True)
    return ordered[0][0]

def parity_even_brights(states: List[Illumination]) -> bool:
    brights = sum(1 for s in states if s in (Illumination.BRIGHT, Illumination.LIGHT))
    return (brights % 2) == 0

def parity_ambient_guard(states: List[Illumination]) -> bool:
    ambient = sum(1 for s in states if s == Illumination.AMBIENT)
    brights = sum(1 for s in states if s in (Illumination.BRIGHT, Illumination.LIGHT))
    return ambient >= 1 and brights <= max(1, len(states)//2)

def blend_linear(a: Illumination, b: Illumination, w: float = 0.5) -> Illumination:
    v = a.value * w + b.value * (1 - w)
    nearest = min(Illumination, key=lambda s: abs(s.value - v))
    return nearest

def blend_bias_ambient(a: Illumination, b: Illumination, w: float = 0.5) -> Illumination:
    v = a.value * w + b.value * (1 - w)
    v = (v + Illumination.AMBIENT.value) / 2.0
    nearest = min(Illumination, key=lambda s: abs(s.value - v))
    return nearest

DEFAULT_STRATEGY = StrategyPack("majority+even_brights+linear", vote_majority, parity_even_brights, blend_linear)
ROBUST_STRATEGY  = StrategyPack("weighted+ambient_guard+bias_ambient", vote_weighted, parity_ambient_guard, blend_bias_ambient)

# =========================
# Protection & anomalies
# =========================
@dataclass
class ProtectedState:
    state: Illumination
    confidence: float
    anomalies: List[str]
    parity_ok: bool
    voted_from: List[Illumination]
    strategy: str

def anomaly_score(series: List[float]) -> Tuple[float, List[str], float]:
    if not series: return 0.0, ["empty_series"], 0.5
    mean = sum(series)/len(series)
    var = sum((x-mean)**2 for x in series) / max(1, len(series)-1)
    std = math.sqrt(var)
    anomalies = []
    if std > 0.15: anomalies.append(f"high_jitter(std={std:.3f})")
    if any(x < 0.02 for x in series): anomalies.append("near_dark_floor")
    if any(x > 0.98 for x in series): anomalies.append("near_light_ceiling")
    drift = abs(mean - 0.5)
    if drift > 0.25: anomalies.append(f"mean_drift({mean:.2f})")
    score = min(1.0, (std/0.3)*0.7 + (drift/0.5)*0.3)
    return score, anomalies, mean

def protect(states: List[Illumination], intensities: List[float], strat: StrategyPack) -> ProtectedState:
    s = strat.vote(states) if states else Illumination.AMBIENT
    parity_ok = strat.parity(states) if states else True
    jitter, anomalies, _ = anomaly_score(intensities)
    agreement = (states.count(s)/max(1, len(states))) if states else 0.5
    conf = max(0.0, min(1.0, 0.5*agreement + 0.3*(1.0 if parity_ok else 0.0) + 0.2*(1.0 - jitter)))
    return ProtectedState(s, conf, anomalies, parity_ok, states, strat.name)

# =========================
# Thought model & reasoner
# =========================
@dataclass
class Thought:
    label: str
    conclusion: Illumination
    confidence: float
    evidence: Dict[str, Illumination]
    notes: List[str]

class Reasoner:
    def __init__(self, blend: Callable[[Illumination, Illumination, float], Illumination]):
        self.blend = blend
    def rule_cpu_ready(self, facts: Dict[str, Illumination]) -> Optional[Thought]:
        if facts.get("cpu") in (Illumination.BRIGHT, Illumination.LIGHT) and facts.get("intent") in (Illumination.BRIGHT, Illumination.LIGHT):
            return Thought("cpu_ready", Illumination.LIGHT, 0.9, {"cpu":facts["cpu"],"intent":facts["intent"]}, ["cpu_load_and_intent_align"])
        return None
    def rule_disk_caution(self, facts: Dict[str, Illumination]) -> Optional[Thought]:
        if facts.get("context") == Illumination.AMBIENT and facts.get("disk") in (Illumination.BRIGHT, Illumination.LIGHT):
            blended = self.blend(Illumination.BRIGHT, Illumination.AMBIENT, 0.6)
            return Thought("disk_caution", blended, 0.72, {"context":facts["context"],"disk":facts["disk"]}, ["ambient_context_requires_stabilization"])
        return None
    def rule_net_trust_conflict(self, facts: Dict[str, Illumination]) -> Optional[Thought]:
        if facts.get("net") in (Illumination.BRIGHT, Illumination.LIGHT) and facts.get("trust") in (Illumination.DARK, Illumination.DIM):
            return Thought("net_trust_conflict", Illumination.DIM, 0.75, {"net":facts["net"], "trust":facts["trust"]}, ["high_net_activity_but_low_trust"])
        return None
    def rule_mem_pressure(self, facts: Dict[str, Illumination]) -> Optional[Thought]:
        if facts.get("mem") in (Illumination.BRIGHT, Illumination.LIGHT):
            return Thought("mem_pressure", Illumination.AMBIENT, 0.7, {"mem":facts["mem"]}, ["memory_pressure_detected"])
        return None
    def rule_cpu_dim_rise(self, facts: Dict[str, Illumination]) -> Optional[Thought]:
        if facts.get("cpu") == Illumination.DIM and facts.get("intent") in (Illumination.BRIGHT, Illumination.LIGHT):
            blended = self.blend(Illumination.DIM, Illumination.BRIGHT, 0.4)
            return Thought("cpu_rising", blended, 0.68, {"cpu":facts["cpu"],"intent":facts["intent"]}, ["intent_supports_cpu_rise"])
        return None
    def aggregate(self, thoughts: List[Thought]) -> Optional[Thought]:
        if not thoughts: return None
        vals = [t.conclusion.value for t in thoughts]
        avg = sum(vals)/len(vals)
        nearest = min(Illumination, key=lambda s: abs(s.value - avg))
        conf = max(0.5, sum(t.confidence for t in thoughts)/len(thoughts) - 0.08)
        ev: Dict[str, Illumination] = {}
        for t in thoughts: ev.update(t.evidence)
        return Thought("aggregate", nearest, conf, ev, ["aggregated_conflicts_resolved_toward_context"])
    def reason(self, facts: Dict[str, Illumination]) -> List[Thought]:
        ts: List[Thought] = []
        for rule in (self.rule_cpu_ready, self.rule_disk_caution, self.rule_net_trust_conflict, self.rule_mem_pressure, self.rule_cpu_dim_rise):
            t = rule(facts)
            if t: ts.append(t)
        agg = self.aggregate(ts)
        if agg: ts.append(agg)
        return ts

# =========================
# Profiles & persistence
# =========================
@dataclass
class ChannelProfile:
    jitter_avg: float = 0.0
    drift_avg: float = 0.0
    imbalance_avg: float = 0.0
    samples: int = 0
    thresholds: Thresholds = field(default_factory=Thresholds)
    alpha: float = 0.15
    def update(self, jitter: float, mean: float, imbalance: float):
        anomaly_density = min(1.0, jitter/0.3 + abs(mean-0.5)/0.5 + imbalance)
        self.alpha = 0.10 + 0.15 * min(1.0, anomaly_density)  # 0.10..0.25
        self.samples += 1
        a = self.alpha
        self.jitter_avg = (1 - a)*self.jitter_avg + a*jitter
        self.drift_avg  = (1 - a)*self.drift_avg  + a*abs(mean - 0.5)
        self.imbalance_avg = (1 - a)*self.imbalance_avg + a*imbalance

@dataclass
class PersistentState:
    profiles: Dict[str, ChannelProfile] = field(default_factory=dict)
    strategy_name: str = DEFAULT_STRATEGY.name
    version: int = 4

# =========================
# Metrics provider adapter
# =========================
class MetricsProvider:
    def __init__(self):
        self.last_net = None
        self.last_disk = None
        self.last_ts = None
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
    def sample(self) -> Dict[str, float]:
        ts = time.time()
        metrics: Dict[str, float] = {}
        if PSUTIL_AVAILABLE:
            try:
                cpu = psutil.cpu_percent(interval=None) / 100.0
                vm = psutil.virtual_memory()
                mem = vm.percent / 100.0
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                if self.last_disk and self.last_ts:
                    dt = max(1e-6, ts - self.last_ts)
                    rdiff = max(0, disk_io.read_bytes - self.last_disk.read_bytes)
                    wdiff = max(0, disk_io.write_bytes - self.last_disk.write_bytes)
                    disk_rate = (rdiff + wdiff) / dt
                else:
                    disk_rate = 0.0
                if self.last_net and self.last_ts:
                    dt = max(1e-6, ts - self.last_ts)
                    rdiff = max(0, net_io.bytes_recv - self.last_net.bytes_recv)
                    sdiff = max(0, net_io.bytes_sent - self.last_net.bytes_sent)
                    net_rate = (rdiff + sdiff) / dt
                else:
                    net_rate = 0.0
                disk_norm = min(1.0, disk_rate / (50_000_000))
                net_norm  = min(1.0, net_rate  / (25_000_000))
                proc_count = len(psutil.pids())
                proc_norm = min(1.0, proc_count / 2000.0)
                try:
                    load1, load5, load15 = os.getloadavg()
                    cores = psutil.cpu_count(logical=True) or 1
                    load_norm = min(1.0, load1 / max(1, cores))
                except Exception:
                    load_norm = cpu
                metrics = {"cpu": float(cpu),"mem": float(mem),"disk": float(disk_norm),
                           "net": float(net_norm),"proc": float(proc_norm),"load": float(load_norm)}
                self.last_ts = ts; self.last_disk = disk_io; self.last_net = net_io
            except Exception:
                metrics = self.synthetic_fallback()
        else:
            metrics = self.synthetic_fallback()
        return metrics
    def synthetic_fallback(self) -> Dict[str, float]:
        def walk(v): return min(1.0, max(0.0, v + random.uniform(-0.05, 0.05)))
        base = getattr(self, "_fallback_state", {"cpu":0.4,"mem":0.55,"disk":0.2,"net":0.15,"proc":0.1,"load":0.35})
        for k in list(base.keys()): base[k] = walk(base[k])
        self._fallback_state = base
        return dict(base)

# =========================
# Simple forecasting utilities
# =========================
def ewma_forecast(series: List[float], alpha: float = 0.3, steps: int = 30) -> List[float]:
    if not series:
        return [0.5]*steps
    f = series[-1]
    # EWMA seed with last value; then extend assuming decay toward current trend
    hist = list(series[-60:])
    for x in hist:
        f = alpha*x + (1-alpha)*f
    out = [f]
    for _ in range(steps-1):
        f = alpha*out[-1] + (1-alpha)*out[-1]  # keeps level; you can add drift later
        out.append(f)
    return [min(1.0, max(0.0, v)) for v in out]

def linear_trend_forecast(series: List[float], steps: int = 30) -> List[float]:
    n = len(series)
    if n < 2:
        return [series[-1] if series else 0.5]*steps
    # Least squares on last W points
    W = min(80, n)
    xs = list(range(W))
    ys = series[-W:]
    mean_x = sum(xs)/W; mean_y = sum(ys)/W
    denom = sum((x-mean_x)**2 for x in xs) or 1e-6
    slope = sum((x-mean_x)*(y-mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope*mean_x
    out = []
    for k in range(1, steps+1):
        y = intercept + slope*(W-1 + k)
        out.append(min(1.0, max(0.0, y)))
    return out

def forecast_risk_from_jitter(series: List[float], window: int = 60) -> float:
    if len(series) < 3:
        return 0.0
    tail = series[-window:] if len(series) >= window else series
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail) / max(1, len(tail)-1)
    std = math.sqrt(var)
    # Normalize risk 0..1 relative to jitter threshold ~0.15
    return max(0.0, min(1.0, std / 0.25))

# =========================
# Autonomous engine (with prediction)
# =========================
class AutonomousLightDark:
    def __init__(self, channels: List[str], state_path: str = "light_dark_state.json", log_path: str = "light_dark_telemetry.jsonl"):
        self.channels = channels
        self.state_path = state_path
        self.log_path = log_path

        self.history: Dict[str, RingBuffer] = {ch: RingBuffer(4096) for ch in channels}
        self.last_states: Dict[str, Illumination] = {ch: Illumination.AMBIENT for ch in channels}
        self.conf_trends: Dict[str, RingBuffer] = {ch: RingBuffer(1024) for ch in channels}
        self.anomaly_density: RingBuffer = RingBuffer(1024)

        self.noise: float = 0.01
        self.rate_hz: int = 10
        self.running: bool = True

        self.persistent = self._load_state()
        self.profiles: Dict[str, ChannelProfile] = {ch: self.persistent.profiles.get(ch, ChannelProfile()) for ch in channels}
        self.thresholds: Dict[str, Thresholds] = {ch: self.profiles[ch].thresholds for ch in channels}
        self.strategy: StrategyPack = DEFAULT_STRATEGY if self.persistent.strategy_name == DEFAULT_STRATEGY.name else ROBUST_STRATEGY

        self.loop_ms: float = 0.0
        self.bottleneck_ms: float = 0.0
        self.health_score: float = 1.0
        self.strategy_timeline: List[Tuple[float, str, str]] = []
        self.threshold_timeline: List[Tuple[float, str, Thresholds]] = []
        self.thought_log: List[Dict] = []

        self._listeners: List[Callable[[dict], None]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.reasoner = Reasoner(self.strategy.blend)
        self.metrics = MetricsProvider()

        # Prediction buffers
        self.pred_series: Dict[str, List[float]] = {ch: [] for ch in channels}
        self.pred_conf: Dict[str, List[float]] = {ch: [] for ch in channels}
        self.pred_anomaly_risk: float = 0.0

    def register_listener(self, cb: Callable[[dict], None]): self._listeners.append(cb)

    def _emit(self, payload: dict):
        for cb in list(self._listeners):
            try:
                cb(payload)
            except Exception as e:
                self._log_error("listener_error", e)

    def _load_state(self) -> PersistentState:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
                    ps = PersistentState()
                    ps.strategy_name = data.get("strategy_name", DEFAULT_STRATEGY.name)
                    ps.version = data.get("version", 4)
                    for ch, p in data.get("profiles", {}).items():
                        th = Thresholds(**p.get("thresholds", {}))
                        prof = ChannelProfile(
                            jitter_avg=p.get("jitter_avg",0.0),
                            drift_avg=p.get("drift_avg",0.0),
                            imbalance_avg=p.get("imbalance_avg",0.0),
                            samples=p.get("samples",0),
                            thresholds=th,
                            alpha=p.get("alpha",0.15)
                        )
                        ps.profiles[ch] = prof
                    return ps
        except Exception as e:
            self._log_error("state_load_error", e)
        return PersistentState()

    def _save_state(self) -> None:
        try:
            obj = {
                "profiles": {ch: {"jitter_avg":p.jitter_avg,"drift_avg":p.drift_avg,"imbalance_avg":p.imbalance_avg,"samples":p.samples,"thresholds":asdict(p.thresholds),"alpha":p.alpha} for ch,p in self.profiles.items()},
                "strategy_name": self.strategy.name,
                "version": 4
            }
            atomic_write_json(self.state_path, obj)
        except Exception as e:
            self._log_error("state_save_error", e)

    def _log_error(self, kind: str, e: Exception):
        event = {"ts": time.time(), "kind": kind, "error": str(e), "trace": traceback.format_exc()}
        try:
            log_jsonl(self.log_path, event)
        except Exception:
            pass

    def _log_snapshot(self, kind: str, payload: dict):
        event = {"ts": time.time(), "kind": kind, "payload": payload}
        try:
            log_jsonl(self.log_path, event)
        except Exception:
            pass

    def start(self): self.thread.start()
    def stop(self):
        self.running = False
        self.thread.join(timeout=3.0)
        self._save_state()

    def _run(self):
        backoff = 0.0
        while self.running:
            t0 = time.time()
            try:
                # Sample real metrics
                m = self.metrics.sample()

                # Map to intensities + derived channels
                intensities_map = {
                    "cpu": m["cpu"], "mem": m["mem"], "disk": m["disk"], "net": m["net"], "proc": m["proc"], "load": m["load"],
                    "intent": min(1.0, 0.3 + 0.7*m["cpu"]),
                    "context": 1.0 - abs(0.5 - (m["mem"]+m["load"])/2.0),
                    "trust": 1.0 - m["net"]
                }

                # Ingest and classify
                for ch in self.channels:
                    val = intensities_map.get(ch, 0.5)
                    sig = Signal(intensity=val); sig.update(self.noise)
                    self.history[ch].append(sig.intensity)
                    self.last_states[ch] = sig.classify(self.last_states[ch], self.thresholds[ch])

                # Protection
                states = [self.last_states[ch] for ch in self.channels]
                intensities: List[float] = []
                for ch in self.channels: intensities += self.history[ch].tail(256)
                protected = protect(states, intensities, self.strategy)

                # Health & trends
                anomaly_penalty = 0.12 * len(protected.anomalies)
                self.health_score = max(0.0, min(1.0, 0.7*protected.confidence + 0.3*(1.0 - anomaly_penalty)))
                for ch in self.channels:
                    st = self.last_states[ch].value
                    conf_val = 1.0 - (abs(st - Illumination.AMBIENT.value) / Illumination.LIGHT.value)
                    self.conf_trends[ch].append(conf_val)
                self.anomaly_density.append(anomaly_penalty)

                # Predictions (short horizon)
                horizon = 30  # ~3s at 10 Hz
                self.pred_series = {}
                for ch in self.channels:
                    s = self.history[ch].tail(200)
                    # Hybrid forecast: average EWMA and linear trend for stability
                    f1 = ewma_forecast(s, alpha=0.35, steps=horizon)
                    f2 = linear_trend_forecast(s, steps=horizon)
                    self.pred_series[ch] = [min(1.0, max(0.0, 0.6*a + 0.4*b)) for a, b in zip(f1, f2)]
                    # Project confidence by mapping predicted states around ambient
                    self.pred_conf[ch] = [1.0 - (abs(v - 0.5) / 0.5) * 0.5 for v in self.pred_series[ch]]
                # Predict anomaly risk from recent jitter
                self.pred_anomaly_risk = forecast_risk_from_jitter(self.anomaly_density.to_list(), window=60)

                # Judge–repair + thoughts
                thoughts = self._judge_repair_and_think(protected)

                # Timing
                t1 = time.time()
                self.loop_ms = (t1 - t0) * 1000.0
                self.bottleneck_ms = max(self.bottleneck_ms * 0.95, self.loop_ms)

                # Emit payload
                payload = {
                    "ts": t1,
                    "states": {ch: self.last_states[ch].name for ch in self.channels},
                    "series": {ch: self.history[ch].tail(150) for ch in self.channels},
                    "thresholds": {ch: self.thresholds[ch] for ch in self.channels},
                    "protected": protected,
                    "strategy": self.strategy.name,
                    "health": self.health_score,
                    "loop_ms": self.loop_ms,
                    "bottleneck_ms": self.bottleneck_ms,
                    "strategy_timeline": list(self.strategy_timeline[-200:]),
                    "threshold_timeline": [(t, ch, th) for (t, ch, th) in self.threshold_timeline[-200:]],
                    "thoughts": thoughts,
                    "thought_log": list(self.thought_log[-200:]),
                    "conf_trends": {ch: self.conf_trends[ch].tail(100) for ch in self.channels},
                    "anomaly_bar": self.anomaly_density.tail(100),
                    "pred_series": self.pred_series,
                    "pred_conf": self.pred_conf,
                    "pred_anomaly_risk": self.pred_anomaly_risk
                }
                self._emit(payload)
                self._log_snapshot("status_snapshot", {"health": self.health_score, "strategy": self.strategy.name, "loop_ms": self.loop_ms})

                # Rate control
                elapsed = time.time() - t0
                time.sleep(max(0, (1.0/self.rate_hz) - elapsed + backoff))
                backoff = max(0.0, backoff - 0.001)

            except Exception as e:
                backoff = min(0.25, backoff + 0.05)
                self._log_error("loop_error", e)
                time.sleep(0.02 + backoff)

    def _judge_repair_and_think(self, protected: ProtectedState) -> List[Dict]:
        # Global metrics
        series_all: List[float] = []
        for ch in self.channels: series_all += self.history[ch].tail(256)
        jitter_g, anomalies_g, mean_g = anomaly_score(series_all)

        # Predictive pre-switch: if forecasted risk high, pre-mutate
        pre_switch_to_robust = (self.pred_anomaly_risk > 0.6)
        pre_switch_to_default = (self.pred_anomaly_risk < 0.2)

        # Adaptive thresholds per channel
        changed_thr = 0
        now = time.time()
        for ch in self.channels:
            series = self.history[ch].tail(256)
            jitter_c, anomalies_c, mean_c = anomaly_score(series)
            self.profiles[ch].update(jitter_c, mean_c, imbalance=0.0)
            th = self.thresholds[ch]
            new_th = Thresholds(th.low_on, th.mid_on, th.high_on, th.low_off, th.mid_off, th.high_off)
            if jitter_c > 0.22:
                new_th.low_on = min(0.55, new_th.low_on + 0.02)
                new_th.mid_on = min(0.65, new_th.mid_on + 0.02)
                new_th.high_on = min(0.88, new_th.high_on + 0.02)
            if abs(mean_c - 0.5) > 0.25:
                if mean_c > 0.5:
                    new_th.low_off = min(0.5, new_th.low_off + 0.02)
                    new_th.mid_off = min(0.6, new_th.mid_off + 0.02)
                else:
                    new_th.high_off = max(0.45, new_th.high_off - 0.02)
            new_th.clamp()
            if (new_th.low_on, new_th.mid_on, new_th.high_on, new_th.low_off, new_th.mid_off, new_th.high_off) != (th.low_on, th.mid_on, th.high_on, th.low_off, th.mid_off, th.high_off):
                self.thresholds[ch] = new_th
                self.profiles[ch].thresholds = new_th
                self.threshold_timeline.append((now, ch, new_th))
                if len(self.threshold_timeline) > 1000:
                    self.threshold_timeline = self.threshold_timeline[-1000:]
                changed_thr += 1

        # Strategy mutation/reversion (reactive + predictive)
        want_robust_reactive = jitter_g > 0.26 or "mean_drift" in " ".join(anomalies_g)
        want_default_reactive = jitter_g < 0.12
        strat_changed = None
        if (want_robust_reactive or pre_switch_to_robust) and self.strategy.name != ROBUST_STRATEGY.name:
            self.strategy = ROBUST_STRATEGY
            self.reasoner = Reasoner(self.strategy.blend)
            self.strategy_timeline.append((now, "mutate", self.strategy.name))
            if len(self.strategy_timeline) > 1000:
                self.strategy_timeline = self.strategy_timeline[-1000:]
            strat_changed = "mutated_to_robust" + ("_predictive" if pre_switch_to_robust else "_reactive")
        elif (want_default_reactive or pre_switch_to_default) and self.strategy.name != DEFAULT_STRATEGY.name:
            self.strategy = DEFAULT_STRATEGY
            self.reasoner = Reasoner(self.strategy.blend)
            self.strategy_timeline.append((now, "revert", self.strategy.name))
            if len(self.strategy_timeline) > 1000:
                self.strategy_timeline = self.strategy_timeline[-1000:]
            strat_changed = "reverted_to_default" + ("_predictive" if pre_switch_to_default else "_reactive")

        # Auto-tune
        if self.health_score < 0.6:
            self.noise = max(0.0, self.noise - 0.002)
            self.rate_hz = max(3, int(self.rate_hz * 0.9))
        elif self.health_score > 0.85:
            self.noise = min(0.05, self.noise + 0.001)
            self.rate_hz = min(30, int(self.rate_hz * 1.05))

        # Facts
        facts = {
            "cpu": self.last_states.get("cpu", Illumination.AMBIENT),
            "mem": self.last_states.get("mem", Illumination.AMBIENT),
            "disk": self.last_states.get("disk", Illumination.AMBIENT),
            "net": self.last_states.get("net", Illumination.AMBIENT),
            "proc": self.last_states.get("proc", Illumination.AMBIENT),
            "load": self.last_states.get("load", Illumination.AMBIENT),
            "intent": self.last_states.get("intent", Illumination.AMBIENT),
            "context": self.last_states.get("context", Illumination.AMBIENT),
            "trust": self.last_states.get("trust", Illumination.AMBIENT),
        }
        rule_thoughts = self.reasoner.reason(facts)

        # Predictive thoughts
        predictive_notes = [f"pred_anomaly_risk={self.pred_anomaly_risk:.2f}"]
        # Example: if predicted CPU series climbs >0.75, warn of imminent mem_pressure
        cpu_pred = self.pred_series.get("cpu", [])
        mem_pred = self.pred_series.get("mem", [])
        if cpu_pred and max(cpu_pred) > 0.75:
            predictive_notes.append("forecast_cpu_high")
        if mem_pred and max(mem_pred) > 0.8:
            predictive_notes.append("forecast_mem_pressure")

        # Corrective proposal thought
        proposal_notes = []
        if changed_thr: proposal_notes.append(f"threshold_updates={changed_thr}")
        if strat_changed: proposal_notes.append(strat_changed)
        proposal_thought = Thought(
            "judge_repair_predictive",
            Illumination.LIGHT if self.health_score >= 0.75 else Illumination.AMBIENT if self.health_score >= 0.5 else Illumination.DARK,
            max(0.5, min(0.95, 0.6 + 0.4*(self.health_score))),
            {"health": Illumination.LIGHT if self.health_score>=0.85 else Illumination.AMBIENT if self.health_score>=0.6 else Illumination.DARK},
            ["global_jitter={:.3f}".format(jitter_g)] + proposal_notes + predictive_notes
        )

        thoughts_serialized = [self._serialize_thought(t) for t in rule_thoughts + [proposal_thought]]
        self.thought_log.extend(thoughts_serialized)
        if len(self.thought_log) > 1500:
            self.thought_log = self.thought_log[-1500:]

        try:
            self._log_snapshot("thoughts", {"count": len(thoughts_serialized)})
        except Exception:
            pass

        return thoughts_serialized

    @staticmethod
    def _serialize_thought(t: Thought) -> Dict:
        return {
            "label": t.label,
            "conclusion": t.conclusion.name,
            "confidence": t.confidence,
            "evidence": {k: v.name for k, v in t.evidence.items()},
            "notes": list(t.notes),
            "ts": time.time()
        }

# =========================
# Compact scientific GUI (≈50% smaller) with predictive overlays
# =========================
import tkinter as tk
from tkinter import ttk

class ScientificGUI:
    def __init__(self, engine: AutonomousLightDark):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Secondary Skeleton Monitor (Predictive, Compact)")
        self.root.geometry("750x490")
        self.root.configure(bg="#111111")

        # Status bar
        self.status_text = tk.StringVar(value="Initializing…")
        top = ttk.Frame(self.root); top.pack(fill="x", padx=4, pady=4)
        ttk.Label(top, textvariable=self.status_text).pack(side="left")

        # Tabs
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True, padx=4, pady=4)
        self.overview = ttk.Frame(self.tabs); self.analytics = ttk.Frame(self.tabs); self.thoughts_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.overview, text="Overview"); self.tabs.add(self.analytics, text="Analytics"); self.tabs.add(self.thoughts_tab, text="Thoughts")

        # Overview: tiles
        self.tile_frame = ttk.Frame(self.overview); self.tile_frame.pack(fill="x", padx=4, pady=4)
        self.tile_vars: Dict[str, tk.StringVar] = {ch: tk.StringVar(value="AMBIENT") for ch in engine.channels}
        for i, ch in enumerate(engine.channels):
            card = ttk.LabelFrame(self.tile_frame, text=ch)
            card.grid(row=0, column=i % 6, padx=3, pady=3, sticky="nsew")
            ttk.Label(card, textvariable=self.tile_vars[ch], font=("Consolas", 8)).pack(padx=4, pady=4)
            self.tile_frame.grid_columnconfigure(i % 6, weight=1)

        # Summary
        self.summary = ttk.Frame(self.overview); self.summary.pack(fill="x", padx=4, pady=4)
        self.summary_prot = tk.StringVar(value="Protection: —")
        self.summary_health = tk.StringVar(value="Health: —")
        self.summary_strategy = tk.StringVar(value="Strategy: —")
        ttk.Label(self.summary, textvariable=self.summary_prot, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_health, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_strategy, font=("Consolas", 8)).pack(anchor="w")

        # Charts grid (series + confidence, compact)
        self.chart_grid = ttk.Frame(self.overview); self.chart_grid.pack(fill="both", expand=True, padx=4, pady=4)
        self.series_canvases: Dict[str, tk.Canvas] = {}
        self.conf_canvases: Dict[str, tk.Canvas] = {}
        for i, ch in enumerate(engine.channels):
            c_series = tk.Canvas(self.chart_grid, width=210, height=105, bg="#1b1b1b", highlightthickness=0)
            c_series.grid(row=(i // 3)*2, column=i % 3, padx=3, pady=3, sticky="nsew")
            self.series_canvases[ch] = c_series
            c_conf = tk.Canvas(self.chart_grid, width=210, height=45, bg="#151515", highlightthickness=0)
            c_conf.grid(row=(i // 3)*2+1, column=i % 3, padx=3, pady=3, sticky="nsew")
            self.conf_canvases[ch] = c_conf
        for r in range(((len(engine.channels)+2)//3)*2): self.chart_grid.grid_rowconfigure(r, weight=1)
        for c in range(3): self.chart_grid.grid_columnconfigure(c, weight=1)

        # Analytics: timelines + heatmap + anomaly bar (compact)
        self.analytics_top = ttk.Frame(self.analytics); self.analytics_top.pack(fill="x", padx=4, pady=4)
        self.timeline_text = tk.StringVar(value="Timelines: —")
        ttk.Label(self.analytics_top, textvariable=self.timeline_text, font=("Consolas", 8)).pack(anchor="w")
        self.timeline_canvas = tk.Canvas(self.analytics, width=740, height=60, bg="#1b1b1b", highlightthickness=0)
        self.timeline_canvas.pack(fill="x", padx=4, pady=4)
        self.heat_canvas = tk.Canvas(self.analytics, width=740, height=140, bg="#1b1b1b", highlightthickness=0)
        self.heat_canvas.pack(fill="x", padx=4, pady=4)
        self.anomaly_canvas = tk.Canvas(self.analytics, width=740, height=45, bg="#151515", highlightthickness=0)
        self.anomaly_canvas.pack(fill="x", padx=4, pady=4)

        # Thoughts: table + narrative
        self.th_tree = ttk.Treeview(self.thoughts_tab, columns=("label","conclusion","confidence","evidence","notes","ts"), show="headings")
        for col in ("label","conclusion","confidence","evidence","notes","ts"):
            self.th_tree.heading(col, text=col)
            self.th_tree.column(col, width=90 if col!="evidence" else 160, stretch=True)
        self.th_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.th_narrative = tk.Text(self.thoughts_tab, height=6)
        self.th_narrative.pack(fill="x", expand=False, padx=4, pady=4)

        # Bind engine
        self.payload: Optional[dict] = None
        engine.register_listener(self.update_data)
        self.root.after(120, self.refresh)

    def run(self):
        self.engine.start()
        try: self.root.mainloop()
        finally: self.engine.stop()

    def update_data(self, payload: dict): self.payload = payload

    def refresh(self):
        if self.payload:
            self.status_text.set(f"Loop={self.payload['loop_ms']:.1f}ms | Bottleneck={self.payload['bottleneck_ms']:.1f}ms | Strategy={self.payload['strategy']} | Health={self.payload['health']:.2f}")
            for ch, st in self.payload["states"].items(): self.tile_vars[ch].set(st)
            prot = self.payload["protected"]
            anomalies = ", ".join(prot.anomalies) if prot.anomalies else "none"
            voted = "[" + ", ".join(s.name for s in prot.voted_from) + "]"
            self.summary_prot.set(f"Protection: {prot.state.name} conf={prot.confidence:.2f} parity={prot.parity_ok} strategy={prot.strategy} anomalies={anomalies} from={voted}")
            self.summary_health.set(f"Health: {self.payload['health']:.2f}")
            self.summary_strategy.set(f"Strategy: {self.payload['strategy']}")

            # Series + confidence + prediction overlays
            for ch in self.engine.channels:
                self.draw_series(self.series_canvases[ch], self.payload["series"].get(ch, []),
                                 self.payload["thresholds"][ch],
                                 self.payload["pred_series"].get(ch, []))
                self.draw_confidence(self.conf_canvases[ch], self.payload["conf_trends"].get(ch, []),
                                     self.payload["pred_conf"].get(ch, []))

            # Timelines, heatmap, anomaly bar
            strat_events = len(self.payload["strategy_timeline"]); thr_events = len(self.payload["threshold_timeline"])
            self.timeline_text.set(f"Timelines: strategy={strat_events} thresholds={thr_events} | risk={self.payload.get('pred_anomaly_risk',0.0):.2f}")
            self.draw_timeline(self.timeline_canvas, self.payload["strategy_timeline"])
            self.draw_heatmap(self.heat_canvas, self.payload["series"])
            self.draw_anomaly_bar(self.anomaly_canvas, self.payload["anomaly_bar"])

            # Thoughts
            self.update_thoughts(self.payload["thoughts"])
            self.update_narrative(self.payload["thought_log"])

        self.root.after(120, self.refresh)

    def update_thoughts(self, thoughts: List[Dict]):
        self.th_tree.delete(*self.th_tree.get_children())
        for t in thoughts:
            ev = ";".join(f"{k}={v}" for k,v in t["evidence"].items())
            notes = ",".join(t["notes"])
            ts = "{:.3f}".format(t["ts"])
            self.th_tree.insert("", "end", values=(t["label"], t["conclusion"], f"{t['confidence']:.2f}", ev, notes, ts))

    def update_narrative(self, log_items: List[Dict]):
        self.th_narrative.delete("1.0", "end")
        for t in log_items[-50:]:
            ev = " ".join(f"{k}={v}" for k,v in t["evidence"].items())
            notes = ";".join(t["notes"])
            line = f"[{t['ts']:.3f}] {t['label']} -> {t['conclusion']} (conf={t['confidence']:.2f}) | {ev} | {notes}\n"
            self.th_narrative.insert("end", line)
        self.th_narrative.see("end")

    def draw_series(self, canvas: tk.Canvas, series: List[float], th: Thresholds, pred: List[float]):
        w = int(canvas.winfo_width() or 210); h = int(canvas.winfo_height() or 105)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        for y in [h*0.3, h*0.6]: canvas.create_line(0, y, w, y, fill="#333333")
        if series:
            n = len(series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-12)) - 6) for val in series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#66ccff", width=2)
        # Prediction overlay (dashed)
        if pred:
            n2 = len(pred)
            xs2 = [int(i*(w-20)/max(1, n2-1)) + 6 for i in range(n2)]
            ys2 = [int(h - (val*(h-12)) - 6) for val in pred]
            for i in range(1, n2):
                canvas.create_line(xs2[i-1], ys2[i-1], xs2[i], ys2[i], fill="#ffaa44", dash=(3,2), width=2)
        def yv(v): return h - int(v*(h-12)) - 6
        canvas.create_line(6, yv(th.low_on),  w-6, yv(th.low_on),  fill="#88aa44", dash=(4,3))
        canvas.create_line(6, yv(th.mid_on),  w-6, yv(th.mid_on),  fill="#44cc88", dash=(4,3))
        canvas.create_line(6, yv(th.high_on), w-6, yv(th.high_on), fill="#ffaa44", dash=(4,3))
        canvas.create_line(6, yv(th.low_off), w-6, yv(th.low_off), fill="#aa88ff", dash=(4,3))
        canvas.create_line(6, yv(th.mid_off), w-6, yv(th.mid_off), fill="#8888aa", dash=(4,3))
        canvas.create_line(6, yv(th.high_off),w-6, yv(th.high_off),fill="#ff8888", dash=(4,3))

    def draw_confidence(self, canvas: tk.Canvas, conf_series: List[float], pred_conf: List[float]):
        w = int(canvas.winfo_width() or 210); h = int(canvas.winfo_height() or 45)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        if conf_series:
            n = len(conf_series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-10)) - 5) for val in conf_series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#44dd88", width=2)
        if pred_conf:
            n2 = len(pred_conf)
            xs2 = [int(i*(w-20)/max(1, n2-1)) + 6 for i in range(n2)]
            ys2 = [int(h - (val*(h-10)) - 5) for val in pred_conf]
            for i in range(1, n2): canvas.create_line(xs2[i-1], ys2[i-1], xs2[i], ys2[i], fill="#ddaa44", dash=(3,2), width=2)
        canvas.create_line(6, h-6, w-6, h-6, fill="#333333")

    def draw_timeline(self, canvas: tk.Canvas, events: List[Tuple[float, str, str]]):
        w = int(canvas.winfo_width() or 740); h = int(canvas.winfo_height() or 60)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        if not events: return
        t_min = events[0][0]; t_max = events[-1][0]; span = max(1e-3, t_max - t_min)
        for ts, action, strat in events:
            x = int((ts - t_min)/span * (w-12)) + 6
            color = "#ffaa44" if action == "mutate" else "#44ccff"
            canvas.create_line(x, 6, x, h-6, fill=color)
            canvas.create_text(x+3, 8, text=f"{action}:{strat}", anchor="nw", fill="#cccccc", font=("Consolas", 7))

    def draw_heatmap(self, canvas: tk.Canvas, series_map: Dict[str, List[float]]):
        canvas.delete("all")
        w = int(canvas.winfo_width() or 740); h = int(canvas.winfo_height() or 140)
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        channels = list(series_map.keys()); rows = len(channels)
        cols = max((len(v) for v in series_map.values()), default=0)
        if rows == 0 or cols == 0: return
        all_vals = [val for vs in series_map.values() for val in vs] or [0.0]
        vmin, vmax = min(all_vals), max(all_vals); rng = max(1e-6, vmax - vmin)
        cell_w = max(2, int(w / cols)); cell_h = max(10, int(h / rows))
        for r, ch in enumerate(channels):
            vals = series_map[ch]
            if len(vals) < cols: vals = ([vals[0]] * (cols - len(vals))) + vals
            for c in range(cols):
                v = (vals[c] - vmin) / rng
                if v < 0.35: color = f"#{int(30+v*100):02x}44cc"
                elif v > 0.65: color = f"#ff{int(120+v*100):02x}66"
                else: color = f"#aa66{int(120+v*100):02x}"
                x0 = c * cell_w; y0 = r * cell_h
                canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h, fill=color, outline="")

    def draw_anomaly_bar(self, canvas: tk.Canvas, anomaly_series: List[float]):
        w = int(canvas.winfo_width() or 740); h = int(canvas.winfo_height() or 45)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        if not anomaly_series: return
        n = len(anomaly_series); bw = max(2, int((w-12)/n)); x = 6
        for val in anomaly_series:
            height = int(val * (h-10))
            color = "#ff6666" if val > 0.2 else "#ffaa44" if val > 0.1 else "#66cc88"
            canvas.create_rectangle(x, h-6-height, x+bw, h-6, fill=color, outline="")
            x += bw

# =========================
# Wiring it up
# =========================
def main():
    channels = ["cpu", "mem", "disk", "net", "proc", "load", "intent", "context", "trust"]
    engine = AutonomousLightDark(channels)
    gui = ScientificGUI(engine)
    gui.run()

if __name__ == "__main__":
    main()

