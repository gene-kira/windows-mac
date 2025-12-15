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
    version: int = 7

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
# Forecasting utilities (tri-horizon) and risk/ETA
# =========================
def ewma(series: List[float], alpha: float = 0.35) -> float:
    if not series: return 0.5
    f = series[0]
    for x in series:
        f = alpha*x + (1-alpha)*f
    return f

def residual_std(series: List[float], fitted: List[float]) -> float:
    if not series or len(series) != len(fitted): return 0.0
    res = [y - yhat for y, yhat in zip(series, fitted)]
    mean_r = sum(res)/len(res)
    var_r = sum((r - mean_r)**2 for r in res) / max(1, len(res)-1)
    return math.sqrt(var_r)

def linear_fit(ys: List[float]) -> Tuple[float, float]:
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(ys)/n
    denom = sum((x-mx)**2 for x in xs) or 1e-6
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope*mx
    return intercept, slope

def forecast_short(series: List[float], steps: int = 60) -> Tuple[List[float], List[float], List[float]]:
    # EWMA level + linear correction on last 100 points
    tail = series[-100:] if len(series) >= 100 else series
    level = ewma(tail, 0.45)
    intercept, slope = linear_fit(tail)
    fitted = [intercept + slope*i for i in range(len(tail))]
    rstd = residual_std(tail, fitted)
    f = [min(1.0, max(0.0, level + slope*k*0.35)) for k in range(1, steps+1)]
    band = [min(0.4, rstd*(1 + k/steps)*0.8) for k in range(1, steps+1)]
    lo = [max(0.0, y - b) for y, b in zip(f, band)]
    hi = [min(1.0, y + b) for y, b in zip(f, band)]
    return f, lo, hi

def forecast_mid(series: List[float], steps: int = 120) -> Tuple[List[float], List[float], List[float]]:
    tail = series[-240:] if len(series) >= 240 else series
    intercept, slope = linear_fit(tail)
    fitted = [intercept + slope*i for i in range(len(tail))]
    rstd = residual_std(tail, fitted)
    f = []
    for k in range(1, steps+1):
        y = intercept + slope*(len(tail)-1 + k)
        f.append(min(1.0, max(0.0, y)))
    band = [min(0.5, rstd*(1 + k/steps)) for k in range(1, steps+1)]
    lo = [max(0.0, y - b) for y, b in zip(f, band)]
    hi = [min(1.0, y + b) for y, b in zip(f, band)]
    return f, lo, hi

def forecast_long(series: List[float], steps: int = 300) -> Tuple[List[float], List[float], List[float]]:
    # Regime-aware: if volatility high, damp slope; else trust slope more
    tail = series[-600:] if len(series) >= 600 else series
    intercept, slope = linear_fit(tail)
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail) / max(1, len(tail)-1)
    std = math.sqrt(var)
    slope_damped = slope * (0.5 if std > 0.2 else 0.9)
    f = []
    for k in range(1, steps+1):
        y = intercept + slope_damped*(len(tail)-1 + k)
        f.append(min(1.0, max(0.0, y)))
    band_base = min(0.6, (std + abs(slope)*0.5))
    band = [min(0.6, band_base*(1 + 0.6*k/steps)) for k in range(1, steps+1)]
    lo = [max(0.0, y - b) for y, b in zip(f, band)]
    hi = [min(1.0, y + b) for y, b in zip(f, band)]
    return f, lo, hi

def risk_from_jitter(series: List[float], window: int = 240) -> float:
    if len(series) < 3:
        return 0.0
    tail = series[-window:] if len(series) >= window else series
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail) / max(1, len(tail)-1)
    std = math.sqrt(var)
    return max(0.0, min(1.0, std / 0.25))

def predict_eta_to_level(series: List[float], target: float, hz: float = 10.0) -> Optional[float]:
    if len(series) < 3: return None
    intercept, slope = linear_fit(series[-300:] if len(series)>=300 else series)
    if abs(slope) < 1e-6: return None
    steps = (target - intercept) / slope
    if steps < 0: return None
    return steps / hz

# =========================
# Causality detection (lagged correlations)
# =========================
def lagged_corr(a: List[float], b: List[float], max_lag: int = 60) -> Tuple[int, float]:
    # Find lag where a leads b with max correlation (positive lags mean a precedes b)
    best_lag = 0; best_corr = 0.0
    for lag in range(1, max_lag+1):
        a_cut = a[-(lag+200):-lag] if len(a) >= lag+200 else a[:-lag] if lag < len(a) else []
        b_cut = b[-200:] if len(b) >= 200 else b
        if not a_cut or not b_cut or len(a_cut) != len(b_cut): continue
        ma = sum(a_cut)/len(a_cut); mb = sum(b_cut)/len(b_cut)
        sa = math.sqrt(sum((x-ma)**2 for x in a_cut) / max(1, len(a_cut)-1)) or 1e-6
        sb = math.sqrt(sum((y-mb)**2 for y in b_cut) / max(1, len(b_cut)-1)) or 1e-6
        cov = sum((x-ma)*(y-mb) for x,y in zip(a_cut, b_cut)) / max(1, len(a_cut)-1)
        corr = cov / (sa*sb)
        if abs(corr) > abs(best_corr):
            best_corr = corr; best_lag = lag
    return best_lag, best_corr

# =========================
# Preemptive strategy stack
# =========================
@dataclass
class StrategyPlan:
    action: str  # "mutate" or "revert"
    eta_s: float
    reason: str
    created_ts: float
    cooldown_s: float = 10.0

# =========================
# Autonomous engine (advanced prediction + causality + strategy stack)
# =========================
class AutonomousLightDark:
    def __init__(self, channels: List[str], state_path: str = "light_dark_state.json", log_path: str = "light_dark_telemetry.jsonl"):
        self.channels = channels
        self.state_path = state_path
        self.log_path = log_path

        self.history: Dict[str, RingBuffer] = {ch: RingBuffer(12000) for ch in channels}
        self.last_states: Dict[str, Illumination] = {ch: Illumination.AMBIENT for ch in channels}
        self.conf_trends: Dict[str, RingBuffer] = {ch: RingBuffer(4096) for ch in channels}
        self.anomaly_density: RingBuffer = RingBuffer(4096)

        self.noise: float = 0.006
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

        # Tri-horizon predictions
        self.pred_horizons = {"S": 60, "M": 120, "L": 300}
        self.pred: Dict[str, Dict[str, List[float]]] = {ch: {"S":[], "M":[], "L":[]} for ch in channels}
        self.bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {ch: {"S":([],[]), "M":([],[]), "L":([],[])} for ch in channels}
        self.pred_conf: Dict[str, float] = {ch: 0.5 for ch in channels}
        self.pred_anomaly_risk: float = 0.0
        self.pred_strategy_eta_s: Optional[float] = None

        # Causality map
        self.causal_links: List[Tuple[str, str, int, float]] = []  # (source, target, lag, corr)

        # Strategy plan stack
        self.plan_stack: List[StrategyPlan] = []

        # Future thoughts and scenarios
        self.future_thoughts: List[Dict] = []
        self.scenarios: List[Dict] = []

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
                    ps.version = data.get("version", 7)
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
                "version": 7
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
                m = self.metrics.sample()

                intensities_map = {
                    "cpu": m["cpu"], "mem": m["mem"], "disk": m["disk"], "net": m["net"], "proc": m["proc"], "load": m["load"],
                    "intent": min(1.0, 0.3 + 0.7*m["cpu"]),
                    "context": 1.0 - abs(0.5 - (m["mem"]+m["load"])/2.0),
                    "trust": 1.0 - m["net"]
                }

                for ch in self.channels:
                    val = intensities_map.get(ch, 0.5)
                    sig = Signal(intensity=val); sig.update(self.noise)
                    self.history[ch].append(sig.intensity)
                    self.last_states[ch] = sig.classify(self.last_states[ch], self.thresholds[ch])

                states = [self.last_states[ch] for ch in self.channels]
                intensities: List[float] = []
                for ch in self.channels: intensities += self.history[ch].tail(256)
                protected = protect(states, intensities, self.strategy)

                anomaly_penalty = 0.12 * len(protected.anomalies)
                self.health_score = max(0.0, min(1.0, 0.7*protected.confidence + 0.3*(1.0 - anomaly_penalty)))
                for ch in self.channels:
                    st = self.last_states[ch].value
                    conf_val = 1.0 - (abs(st - Illumination.AMBIENT.value) / Illumination.LIGHT.value)
                    self.conf_trends[ch].append(conf_val)
                self.anomaly_density.append(anomaly_penalty)

                # Tri-horizon predictions
                self.pred_conf = {}
                for ch in self.channels:
                    series = self.history[ch].tail(1200)
                    fS, loS, hiS = forecast_short(series, steps=self.pred_horizons["S"])
                    fM, loM, hiM = forecast_mid(series, steps=self.pred_horizons["M"])
                    fL, loL, hiL = forecast_long(series, steps=self.pred_horizons["L"])
                    self.pred[ch]["S"] = fS; self.bands[ch]["S"] = (loS, hiS)
                    self.pred[ch]["M"] = fM; self.bands[ch]["M"] = (loM, hiM)
                    self.pred[ch]["L"] = fL; self.bands[ch]["L"] = (loL, hiL)
                    # Confidence from short-range near ambient + band tightness
                    tightness = max(1e-6, sum((hi - lo) for lo, hi in zip(loS, hiS)) / len(loS))
                    conf = max(0.0, min(1.0, 1.0 - tightness))
                    self.pred_conf[ch] = conf
                self.pred_anomaly_risk = risk_from_jitter(self.anomaly_density.to_list(), window=240)

                # Predict ETA to robust switch based on anomaly density crossing 0.2
                self.pred_strategy_eta_s = predict_eta_to_level(self.anomaly_density.to_list(), target=0.2, hz=self.rate_hz)

                # Causality detection (selected pairs)
                self.causal_links = []
                pairs = [("cpu","load"), ("net","trust"), ("disk","mem")]
                for a, b in pairs:
                    la, lb = self.history[a].to_list(), self.history[b].to_list()
                    lag, corr = lagged_corr(la, lb, max_lag=60)
                    if abs(corr) > 0.25:
                        self.causal_links.append((a, b, lag, corr))

                # Build/refresh strategy plan stack (preemptive)
                now = time.time()
                self._refresh_plans(now)

                # Future thoughts
                self.future_thoughts = []
                def add_future(label, concl, p, notes): self.future_thoughts.append({"label": label, "conclusion": concl, "confidence": p, "notes": notes, "ts": now})
                cpuS = self.pred.get("cpu", {}).get("S", [])
                memM = self.pred.get("mem", {}).get("M", [])
                netM = self.pred.get("net", {}).get("M", [])
                if cpuS and max(cpuS) > 0.85: add_future("cpu_forecast_high", "BRIGHT", 0.78, ["short_horizon_peak"])
                if memM and max(memM) > 0.9: add_future("mem_forecast_pressure", "AMBIENT", 0.76, ["mid_horizon_pressure"])
                if netM and max(netM) > 0.8 and self.pred_anomaly_risk > 0.5: add_future("net_forecast_instability", "DIM", 0.74, ["mid_horizon_instability"])

                # Scenarios
                self.scenarios = simulate_scenarios({"cpu": m["cpu"], "mem": m["mem"], "net": m["net"], "load": m["load"]})

                # Judge–repair + thoughts
                thoughts = self._judge_repair_and_think(protected)

                t1 = time.time()
                self.loop_ms = (t1 - t0) * 1000.0
                self.bottleneck_ms = max(self.bottleneck_ms * 0.95, self.loop_ms)

                payload = {
                    "ts": t1,
                    "states": {ch: self.last_states[ch].name for ch in self.channels},
                    "series": {ch: self.history[ch].tail(220) for ch in self.channels},
                    "thresholds": {ch: self.thresholds[ch] for ch in self.channels},
                    "protected": protected,
                    "strategy": self.strategy.name,
                    "health": self.health_score,
                    "loop_ms": self.loop_ms,
                    "bottleneck_ms": self.bottleneck_ms,
                    "strategy_timeline": list(self.strategy_timeline[-200:]),
                    "threshold_timeline": [(t, ch, th) for (t, ch, th) in self.threshold_timeline[-200:]],
                    "thoughts": thoughts,
                    "thought_log": list(self.thought_log[-250:]),
                    "conf_trends": {ch: self.conf_trends[ch].tail(140) for ch in self.channels},
                    "anomaly_bar": self.anomaly_density.tail(140),
                    "pred": self.pred,
                    "bands": self.bands,
                    "pred_conf": self.pred_conf,
                    "pred_anomaly_risk": self.pred_anomaly_risk,
                    "pred_strategy_eta_s": self.pred_strategy_eta_s,
                    "future_thoughts": list(self.future_thoughts[-40:]),
                    "scenarios": list(self.scenarios[-10:]),
                    "causal_links": list(self.causal_links),
                    "plans": [asdict(p) for p in self.plan_stack[-10:]]
                }
                self._emit(payload)
                self._log_snapshot("status_snapshot", {"health": self.health_score, "strategy": self.strategy.name, "loop_ms": self.loop_ms})

                elapsed = time.time() - t0
                time.sleep(max(0, (1.0/self.rate_hz) - elapsed + backoff))
                backoff = max(0.0, backoff - 0.001)

            except Exception as e:
                backoff = min(0.25, backoff + 0.05)
                self._log_error("loop_error", e)
                time.sleep(0.02 + backoff)

    def _refresh_plans(self, now: float):
        # Clear expired
        self.plan_stack = [p for p in self.plan_stack if (now - p.created_ts) < p.cooldown_s]
        # Add new plan if risk high and no active mutate plan
        high_risk = self.pred_anomaly_risk > 0.6
        eta = self.pred_strategy_eta_s
        has_mutate = any(p.action == "mutate" for p in self.plan_stack)
        has_revert = any(p.action == "revert" for p in self.plan_stack)
        if high_risk and eta is not None and not has_mutate:
            self.plan_stack.append(StrategyPlan("mutate", eta, "predicted_risk_high", now))
        elif (self.pred_anomaly_risk < 0.2) and not has_revert:
            self.plan_stack.append(StrategyPlan("revert", 5.0, "risk_low", now))

    def _apply_plans(self, now: float) -> Optional[str]:
        # Execute plans whose ETA reached
        if not self.plan_stack: return None
        executed = None
        remaining = []
        for p in self.plan_stack:
            due = (now - p.created_ts) >= max(0.0, p.eta_s)
            if due and executed is None:
                if p.action == "mutate" and self.strategy.name != ROBUST_STRATEGY.name:
                    self.strategy = ROBUST_STRATEGY
                    self.reasoner = Reasoner(self.strategy.blend)
                    self.strategy_timeline.append((now, "mutate", self.strategy.name))
                    executed = "mutated_to_robust_predictive_plan"
                elif p.action == "revert" and self.strategy.name != DEFAULT_STRATEGY.name:
                    self.strategy = DEFAULT_STRATEGY
                    self.reasoner = Reasoner(self.strategy.blend)
                    self.strategy_timeline.append((now, "revert", self.strategy.name))
                    executed = "reverted_to_default_predictive_plan"
            else:
                remaining.append(p)
        self.plan_stack = remaining[-10:]
        return executed

    def _judge_repair_and_think(self, protected: ProtectedState) -> List[Dict]:
        series_all: List[float] = []
        for ch in self.channels: series_all += self.history[ch].tail(256)
        jitter_g, anomalies_g, mean_g = anomaly_score(series_all)

        pre_switch_to_robust = (self.pred_anomaly_risk > 0.6)
        pre_switch_to_default = (self.pred_anomaly_risk < 0.2)

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

        want_robust_reactive = jitter_g > 0.26 or "mean_drift" in " ".join(anomalies_g)
        want_default_reactive = jitter_g < 0.12
        strat_changed = None
        # Try predictive plan execution first
        executed_plan = self._apply_plans(now)
        if executed_plan:
            strat_changed = executed_plan
        else:
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

        if self.health_score < 0.6:
            self.noise = max(0.0, self.noise - 0.002)
            self.rate_hz = max(3, int(self.rate_hz * 0.9))
        elif self.health_score > 0.85:
            self.noise = min(0.05, self.noise + 0.001)
            self.rate_hz = min(30, int(self.rate_hz * 1.05))

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

        predictive_notes = [f"pred_anomaly_risk={self.pred_anomaly_risk:.2f}"]
        if self.pred_strategy_eta_s is not None:
            predictive_notes.append(f"strategy_eta_s≈{self.pred_strategy_eta_s:.1f}")
        # causality notes
        for a,b,lag,corr in self.causal_links[:4]:
            predictive_notes.append(f"{a}->{b}(lag={lag},corr={corr:.2f})")

        cpuS = self.pred.get("cpu", {}).get("S", [])
        memM = self.pred.get("mem", {}).get("M", [])
        if cpuS and max(cpuS) > 0.85:
            predictive_notes.append("forecast_cpu_high")
        if memM and max(memM) > 0.9:
            predictive_notes.append("forecast_mem_pressure")

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
        if len(self.thought_log) > 2500:
            self.thought_log = self.thought_log[-2500:]

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
# Scenario simulation hooks
# =========================
def simulate_scenarios(current: Dict[str, float]) -> List[Dict]:
    cpu = current.get("cpu", 0.5)
    mem = current.get("mem", 0.5)
    net = current.get("net", 0.2)
    load = current.get("load", cpu)
    scenarios = []
    cpu1 = min(1.0, cpu + 0.30); load1 = min(1.0, (load + cpu1) / 2.0)
    risk1 = 0.5 * cpu1 + 0.25 * load1 + 0.25 * max(0.0, mem - 0.6)
    scenarios.append({"name": "cpu_burst_30", "risk": min(1.0, risk1), "notes": "cpu+load stress"})
    net2 = min(1.0, net * 2.0); trust2 = 1.0 - net2
    risk2 = 0.6 * net2 + 0.4 * (1.0 - trust2)
    scenarios.append({"name": "net_double", "risk": min(1.0, risk2), "notes": "net surge → trust down"})
    mem3 = min(1.0, mem + 0.20)
    risk3 = 0.7 * mem3 + 0.3 * load
    scenarios.append({"name": "mem_leak", "risk": min(1.0, risk3), "notes": "mem pressure"})
    return scenarios

# =========================
# Compact scientific GUI with tri-horizon overlays, ETA markers, risk meters, causal ribbons, scenarios, future thoughts
# =========================
import tkinter as tk
from tkinter import ttk

class ScientificGUI:
    def __init__(self, engine: AutonomousLightDark):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Predictive Skeleton Monitor (Tri-horizon, Causality, Preemptive)")
        self.root.geometry("820x600")
        self.root.configure(bg="#111111")

        # Status bar
        self.status_text = tk.StringVar(value="Initializing…")
        top = ttk.Frame(self.root); top.pack(fill="x", padx=4, pady=4)
        ttk.Label(top, textvariable=self.status_text).pack(side="left")

        # Tabs
        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True, padx=4, pady=4)
        self.overview = ttk.Frame(self.tabs); self.analytics = ttk.Frame(self.tabs); self.thoughts_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.overview, text="Overview"); self.tabs.add(self.analytics, text="Analytics"); self.tabs.add(self.thoughts_tab, text="Thoughts")

        # Overview: tiles + risk meters (value + trend)
        self.tile_frame = ttk.Frame(self.overview); self.tile_frame.pack(fill="x", padx=4, pady=4)
        self.tile_vars: Dict[str, tk.StringVar] = {ch: tk.StringVar(value="AMBIENT") for ch in engine.channels}
        self.gauges: Dict[str, tk.Canvas] = {}
        for i, ch in enumerate(engine.channels):
            card = ttk.LabelFrame(self.tile_frame, text=ch); card.grid(row=0, column=i % 6, padx=3, pady=3, sticky="nsew")
            row_inner = ttk.Frame(card); row_inner.pack(fill="x")
            ttk.Label(row_inner, textvariable=self.tile_vars[ch], font=("Consolas", 8)).pack(side="left", padx=4, pady=4)
            g = tk.Canvas(row_inner, width=80, height=14, bg="#181818", highlightthickness=0); g.pack(side="right", padx=4, pady=4)
            self.gauges[ch] = g
            self.tile_frame.grid_columnconfigure(i % 6, weight=1)

        # Summary + predicted ETA
        self.summary = ttk.Frame(self.overview); self.summary.pack(fill="x", padx=4, pady=4)
        self.summary_prot = tk.StringVar(value="Protection: —")
        self.summary_health = tk.StringVar(value="Health: —")
        self.summary_strategy = tk.StringVar(value="Strategy: —")
        self.summary_eta = tk.StringVar(value="Predicted strategy ETA: —")
        ttk.Label(self.summary, textvariable=self.summary_prot, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_health, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_strategy, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_eta, font=("Consolas", 8)).pack(anchor="w")

        # Charts grid (series + tri-horizon + ETA markers)
        self.chart_grid = ttk.Frame(self.overview); self.chart_grid.pack(fill="both", expand=True, padx=4, pady=4)
        self.series_canvases: Dict[str, tk.Canvas] = {}
        self.conf_canvases: Dict[str, tk.Canvas] = {}
        for i, ch in enumerate(engine.channels):
            c_series = tk.Canvas(self.chart_grid, width=240, height=120, bg="#1b1b1b", highlightthickness=0)
            c_series.grid(row=(i // 3)*2, column=i % 3, padx=3, pady=3, sticky="nsew"); self.series_canvases[ch] = c_series
            c_conf = tk.Canvas(self.chart_grid, width=240, height=48, bg="#151515", highlightthickness=0)
            c_conf.grid(row=(i // 3)*2+1, column=i % 3, padx=3, pady=3, sticky="nsew"); self.conf_canvases[ch] = c_conf
        for r in range(((len(engine.channels)+2)//3)*2): self.chart_grid.grid_rowconfigure(r, weight=1)
        for c in range(3): self.chart_grid.grid_columnconfigure(c, weight=1)

        # Analytics: timelines + heatmap + anomaly bar + scenarios + future thoughts + causality
        self.analytics_top = ttk.Frame(self.analytics); self.analytics_top.pack(fill="x", padx=4, pady=4)
        self.timeline_text = tk.StringVar(value="Timelines: —")
        ttk.Label(self.analytics_top, textvariable=self.timeline_text, font=("Consolas", 8)).pack(anchor="w")
        self.timeline_canvas = tk.Canvas(self.analytics, width=780, height=60, bg="#1b1b1b", highlightthickness=0)
        self.timeline_canvas.pack(fill="x", padx=4, pady=4)
        self.heat_canvas = tk.Canvas(self.analytics, width=780, height=140, bg="#1b1b1b", highlightthickness=0)
        self.heat_canvas.pack(fill="x", padx=4, pady=4)
        self.anomaly_canvas = tk.Canvas(self.analytics, width=780, height=45, bg="#151515", highlightthickness=0)
        self.anomaly_canvas.pack(fill="x", padx=4, pady=4)

        ttk.Label(self.analytics, text="Predicted events (next horizon):", font=("Consolas", 8)).pack(anchor="w", padx=4)
        self.future_list = tk.Listbox(self.analytics, height=6)
        self.future_list.pack(fill="x", padx=4, pady=2)

        ttk.Label(self.analytics, text="Scenario simulations (risk projection):", font=("Consolas", 8)).pack(anchor="w", padx=4, pady=(8,2))
        self.scenario_list = tk.Listbox(self.analytics, height=5)
        self.scenario_list.pack(fill="x", padx=4, pady=2)

        ttk.Label(self.analytics, text="Causality (lagged correlations):", font=("Consolas", 8)).pack(anchor="w", padx=4, pady=(8,2))
        self.causality_list = tk.Listbox(self.analytics, height=6)
        self.causality_list.pack(fill="x", padx=4, pady=2)

        # Thoughts: table + narrative
        self.th_tree = ttk.Treeview(self.thoughts_tab, columns=("label","conclusion","confidence","evidence","notes","ts"), show="headings")
        for col in ("label","conclusion","confidence","evidence","notes","ts"):
            self.th_tree.heading(col, text=col)
            self.th_tree.column(col, width=100 if col!="evidence" else 180, stretch=True)
        self.th_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.th_narrative = tk.Text(self.thoughts_tab, height=8)
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
            risk = self.payload.get('pred_anomaly_risk', 0.0)
            self.status_text.set(f"Loop={self.payload['loop_ms']:.1f}ms | Bottleneck={self.payload['bottleneck_ms']:.1f}ms | Strategy={self.payload['strategy']} | Health={self.payload['health']:.2f} | Risk={risk:.2f}")
            for ch, st in self.payload["states"].items(): self.tile_vars[ch].set(st)
            prot = self.payload["protected"]
            anomalies = ", ".join(prot.anomalies) if prot.anomalies else "none"
            voted = "[" + ", ".join(s.name for s in prot.voted_from) + "]"
            self.summary_prot.set(f"Protection: {prot.state.name} conf={prot.confidence:.2f} parity={prot.parity_ok} strategy={prot.strategy} anomalies={anomalies} from={voted}")
            self.summary_health.set(f"Health: {self.payload['health']:.2f}")
            self.summary_strategy.set(f"Strategy: {self.payload['strategy']}")
            eta = self.payload.get("pred_strategy_eta_s", None)
            self.summary_eta.set(f"Predicted strategy ETA: {'~'+str(round(eta,1))+'s' if eta is not None else '—'}")

            # Risk meters
            for ch, g in self.gauges.items():
                self.draw_gauge(g, self.payload["pred_conf"].get(ch, 0.5))

            # Series + confidence + tri-horizon overlays + ETA markers
            for ch in self.engine.channels:
                self.draw_series(self.series_canvases[ch],
                                 self.payload["series"].get(ch, []),
                                 self.payload["thresholds"][ch],
                                 self.payload["pred"].get(ch, {}),
                                 self.payload["bands"].get(ch, {}))
                self.draw_confidence(self.conf_canvases[ch], self.payload["conf_trends"].get(ch, []))

            strat_events = len(self.payload["strategy_timeline"]); thr_events = len(self.payload["threshold_timeline"])
            self.timeline_text.set(f"Timelines: strategy={strat_events} thresholds={thr_events}")
            self.draw_timeline(self.timeline_canvas, self.payload["strategy_timeline"])
            self.draw_heatmap(self.heat_canvas, self.payload["series"])
            self.draw_anomaly_bar(self.anomaly_canvas, self.payload["anomaly_bar"])

            self.update_thoughts(self.payload["thoughts"])
            self.update_narrative(self.payload["thought_log"])
            self.update_future(self.payload.get("future_thoughts", []))
            self.update_scenarios(self.payload.get("scenarios", []))
            self.update_causality(self.payload.get("causal_links", []))

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
        for t in log_items[-70:]:
            ev = " ".join(f"{k}={v}" for k,v in t["evidence"].items())
            notes = ";".join(t["notes"])
            line = f"[{t['ts']:.3f}] {t['label']} -> {t['conclusion']} (conf={t['confidence']:.2f}) | {ev} | {notes}\n"
            self.th_narrative.insert("end", line)
        self.th_narrative.see("end")

    def update_future(self, future_items: List[Dict]):
        self.future_list.delete(0, "end")
        for item in future_items[-14:]:
            label = item.get("label","")
            concl = item.get("conclusion","")
            conf = item.get("confidence",0.0)
            notes = ";".join(item.get("notes",[]))
            self.future_list.insert("end", f"{label} -> {concl} (p≈{conf:.2f}) | {notes}")

    def update_scenarios(self, scenarios: List[Dict]):
        self.scenario_list.delete(0, "end")
        for sc in scenarios[-10:]:
            self.scenario_list.insert("end", f"{sc['name']} | risk≈{sc['risk']:.2f} | {sc['notes']}")

    def update_causality(self, links: List[Tuple[str,str,int,float]]):
        self.causality_list.delete(0, "end")
        for a,b,lag,corr in links[-10:]:
            self.causality_list.insert("end", f"{a} → {b} | lag={lag}s | corr≈{corr:.2f}")

    def draw_gauge(self, canvas: tk.Canvas, val: float):
        canvas.delete("all")
        w = int(canvas.winfo_width() or 80); h = int(canvas.winfo_height() or 14)
        canvas.create_rectangle(0, 0, w, h, fill="#181818", outline="")
        bar_w = int(val * w)
        color = "#66cc88" if val >= 0.66 else "#ffaa44" if val >= 0.33 else "#ff6666"
        canvas.create_rectangle(0, 0, bar_w, h, fill=color, outline="")
        # Trend arrow (simple: color hint)
        canvas.create_rectangle(w-12, 2, w-2, h-2, fill=color, outline="")

    def draw_series(self, canvas: tk.Canvas, series: List[float], th: Thresholds,
                    pred_map: Dict[str, List[float]], bands_map: Dict[str, Tuple[List[float], List[float]]]):
        w = int(canvas.winfo_width() or 240); h = int(canvas.winfo_height() or 120)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        for y in [h*0.3, h*0.6]: canvas.create_line(0, y, w, y, fill="#333333")
        # live series
        if series:
            n = len(series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-12)) - 6) for val in series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#66ccff", width=2)
        # thresholds
        def yv(v): return h - int(v*(h-12)) - 6
        canvas.create_line(6, yv(th.low_on),  w-6, yv(th.low_on),  fill="#88aa44", dash=(4,3))
        canvas.create_line(6, yv(th.mid_on),  w-6, yv(th.mid_on),  fill="#44cc88", dash=(4,3))
        canvas.create_line(6, yv(th.high_on), w-6, yv(th.high_on), fill="#ffaa44", dash=(4,3))
        canvas.create_line(6, yv(th.low_off), w-6, yv(th.low_off), fill="#aa88ff", dash=(4,3))
        canvas.create_line(6, yv(th.mid_off), w-6, yv(th.mid_off), fill="#8888aa", dash=(4,3))
        canvas.create_line(6, yv(th.high_off),w-6, yv(th.high_off),fill="#ff8888", dash=(4,3))
        # tri-horizon overlays
        styles = {"S":("#ffaa44",(3,2),2), "M":("#44dd88",(4,3),2), "L":("#ff6666",(5,3),2)}
        for horiz in ("S","M","L"):
            pred = pred_map.get(horiz, [])
            lo, hi = bands_map.get(horiz, ([], []))
            if pred:
                n2 = len(pred)
                xs2 = [int(i*(w-20)/max(1, n2-1)) + 6 for i in range(n2)]
                ys2 = [int(h - (val*(h-12)) - 6) for val in pred]
                color, dash, width = styles[horiz]
                for i in range(1, n2):
                    canvas.create_line(xs2[i-1], ys2[i-1], xs2[i], ys2[i], fill=color, dash=dash, width=width)
            if lo and hi and len(lo) == len(hi):
                nB = len(lo)
                xsB = [int(i*(w-20)/max(1, nB-1)) + 6 for i in range(nB)]
                ysLo = [int(h - (val*(h-12)) - 6) for val in lo]
                ysHi = [int(h - (val*(h-12)) - 6) for val in hi]
                band_color = "#444444" if horiz=="L" else "#555555" if horiz=="M" else "#666666"
                for i in range(1, nB):
                    canvas.create_rectangle(xsB[i-1], min(ysLo[i-1], ysHi[i-1]), xsB[i], max(ysLo[i-1], ysHi[i-1]),
                                            fill=band_color, outline="")
        # ETA markers: where mid-horizon band crosses high_on (upside) or high_off (downside)
        loM, hiM = bands_map.get("M", ([], []))
        if loM and hiM:
            idx_up = next((i for i,(l,h) in enumerate(zip(loM,hiM)) if l >= th.high_on or h >= th.high_on), None)
            idx_down = next((i for i,(l,h) in enumerate(zip(loM,hiM)) if l <= th.high_off or h <= th.high_off), None)
            def x_at(i, n): return int(i*(w-20)/max(1, n-1)) + 6
            if idx_up is not None:
                x = x_at(idx_up, len(hiM)); canvas.create_line(x, 6, x, h-6, fill="#ffaa44"); canvas.create_text(x+3, 8, text="ETA↑", anchor="nw", fill="#cccccc", font=("Consolas", 7))
            if idx_down is not None:
                x = x_at(idx_down, len(hiM)); canvas.create_line(x, 6, x, h-6, fill="#44ccff"); canvas.create_text(x+3, 8, text="ETA↓", anchor="nw", fill="#cccccc", font=("Consolas", 7))

    def draw_confidence(self, canvas: tk.Canvas, conf_series: List[float]):
        w = int(canvas.winfo_width() or 240); h = int(canvas.winfo_height() or 48)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        if conf_series:
            n = len(conf_series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-10)) - 5) for val in conf_series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#44dd88", width=2)
        canvas.create_line(6, h-6, w-6, h-6, fill="#333333")

    def draw_timeline(self, canvas: tk.Canvas, events: List[Tuple[float, str, str]]):
        w = int(canvas.winfo_width() or 780); h = int(canvas.winfo_height() or 60)
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
        w = int(canvas.winfo_width() or 780); h = int(canvas.winfo_height() or 140)
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
        w = int(canvas.winfo_width() or 780); h = int(canvas.winfo_height() or 45)
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

