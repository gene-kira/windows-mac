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
from typing import Dict, List, Tuple, Callable, Optional, Set
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future

# =========================
# Optional psutil loader (real system metrics)
# =========================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# =========================
# Utilities: atomic write, async logger, ring buffer
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

class AsyncLogger:
    def __init__(self, path: str, capacity: int = 10000):
        self.path = path
        self.q: Queue = Queue(maxsize=capacity)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def _run(self):
        while self.running:
            try:
                event = self.q.get(timeout=0.25)
            except Empty:
                continue
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass
    def submit(self, kind: str, payload: dict):
        try:
            self.q.put_nowait({"ts": time.time(), "kind": kind, "payload": payload})
        except Exception:
            pass
    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass

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
# Core model
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
# Strategies
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
# Thought model
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
    regime: str = "stable"
    def update(self, jitter: float, mean: float, imbalance: float):
        anomaly_density = min(1.0, jitter/0.3 + abs(mean-0.5)/0.5 + imbalance)
        self.alpha = 0.10 + 0.15 * min(1.0, anomaly_density)
        self.samples += 1
        a = self.alpha
        self.jitter_avg = (1 - a)*self.jitter_avg + a*jitter
        self.drift_avg  = (1 - a)*self.drift_avg  + a*abs(mean - 0.5)
        self.imbalance_avg = (1 - a)*self.imbalance_avg + a*imbalance

@dataclass
class PersistentState:
    profiles: Dict[str, ChannelProfile] = field(default_factory=dict)
    strategy_name: str = DEFAULT_STRATEGY.name
    version: int = 10

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
# Change-point detection (CUSUM)
# =========================
@dataclass
class ChangePointState:
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    threshold: float = 0.08
    drift: float = 0.0
    regime: str = "stable"

def update_cusum(cps: ChangePointState, x: float, mean_ref: float = 0.5, k: float = 0.02) -> ChangePointState:
    dev = x - mean_ref
    cps.pos_sum = max(0.0, cps.pos_sum + dev - k)
    cps.neg_sum = max(0.0, cps.neg_sum - dev - k)
    cps.drift = 0.9*cps.drift + 0.1*dev
    if cps.pos_sum > cps.threshold or cps.neg_sum > cps.threshold:
        cps.regime = "spike"
        cps.pos_sum *= 0.5; cps.neg_sum *= 0.5
    else:
        cps.regime = "drift" if abs(cps.drift) > 0.1 else "stable"
    return cps

# =========================
# Kalman (level+slope)
# =========================
@dataclass
class KalmanState:
    level: float = 0.5
    slope: float = 0.0
    P11: float = 0.2
    P12: float = 0.0
    P22: float = 0.2
    q_level: float = 1e-3
    q_slope: float = 1e-4
    r_obs: float = 3e-3

def kalman_update(ks: KalmanState, z: float) -> KalmanState:
    level_pred = ks.level + ks.slope
    slope_pred = ks.slope
    P11 = ks.P11 + ks.q_level + ks.P22 + ks.q_slope + 2*ks.P12
    P12 = ks.P12 + ks.q_slope + ks.P22
    P22 = ks.P22 + ks.q_slope
    y = z - level_pred
    S = P11 + ks.r_obs
    K1 = P11 / S
    K2 = P12 / S
    ks.level = level_pred + K1*y
    ks.slope = slope_pred + K2*y
    ks.P11 = (1 - K1) * P11
    ks.P12 = (1 - K1) * P12
    ks.P22 = P22 - K2 * P12
    ks.level = min(1.0, max(0.0, ks.level))
    ks.slope = min(0.2, max(-0.2, ks.slope))
    return ks

def kalman_forecast(ks: KalmanState, steps: int, band_scale: float = 0.1) -> Tuple[List[float], List[float], List[float]]:
    f = []; lo = []; hi = []
    l = ks.level; s = ks.slope
    for k in range(1, steps+1):
        y = l + s*k
        y = min(1.0, max(0.0, y))
        f.append(y)
        b = min(0.5, band_scale*(1 + k/steps))
        lo.append(max(0.0, y - b))
        hi.append(min(1.0, y + b))
    return f, lo, hi

# =========================
# Forecasting (EWMA + linear + Holt-Winters-lite)
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
    if not ys: return 0.5, 0.0
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(ys)/n
    denom = sum((x-mx)**2 for x in xs) or 1e-6
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope*mx
    return intercept, slope

def forecast_short(series: List[float], steps: int = 60) -> Tuple[List[float], List[float], List[float]]:
    tail = series[-120:] if len(series) >= 120 else series
    level = ewma(tail, 0.5)
    intercept, slope = linear_fit(tail)
    fitted = [intercept + slope*i for i in range(len(tail))]
    rstd = residual_std(tail, fitted)
    f = [min(1.0, max(0.0, level + slope*k*0.35)) for k in range(1, steps+1)]
    band = [min(0.4, rstd*(1 + k/steps)*0.8) for k in range(1, steps+1)]
    lo = [max(0.0, y - b) for y, b in zip(f, band)]
    hi = [min(1.0, y + b) for y, b in zip(f, band)]
    return f, lo, hi

def forecast_mid(series: List[float], steps: int = 120) -> Tuple[List[float], List[float], List[float]]:
    tail = series[-300:] if len(series) >= 300 else series
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
    tail = series[-800:] if len(series) >= 800 else series
    intercept, slope = linear_fit(tail)
    mean = sum(tail)/len(tail) if tail else 0.5
    var = sum((x-mean)**2 for x in tail) / max(1, len(tail)-1) if tail else 0.0
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

# Holt-Winters-lite (additive, no seasonality length learn; simple smoothing)
def holt_winters(series: List[float], steps: int, alpha: float = 0.5, beta: float = 0.2) -> Tuple[List[float], List[float], List[float]]:
    if not series:
        f = [0.5]*steps; band = [0.2]*steps
        lo = [max(0.0, y - b) for y,b in zip(f,band)]
        hi = [min(1.0, y + b) for y,b in zip(f,band)]
    else:
        level = series[0]; trend = 0.0
        for x in series:
            prev_level = level
            level = alpha*x + (1-alpha)*(level + trend)
            trend = beta*(level - prev_level) + (1-beta)*trend
        f = []; band = []
        for k in range(1, steps+1):
            y = min(1.0, max(0.0, level + trend * k))
            f.append(y)
            b = min(0.5, 0.15*(1 + k/steps))
            band.append(b)
        lo = [max(0.0, y - b) for y,b in zip(f,band)]
        hi = [min(1.0, y + b) for y,b in zip(f,band)]
    return f, lo, hi

def risk_from_jitter(series: List[float], window: int = 300) -> float:
    if len(series) < 3:
        return 0.0
    tail = series[-window:] if len(series) >= window else series
    mean = sum(tail)/len(tail)
    var = sum((x-mean)**2 for x in tail) / max(1, len(tail)-1)
    std = math.sqrt(var)
    return max(0.0, min(1.0, std / 0.25))

# =========================
# Hazard-based ETA
# =========================
def hazard_eta(target: float, lo: List[float], hi: List[float], hz: float = 10.0) -> Optional[Tuple[float, float]]:
    if not lo or not hi or len(lo) != len(hi): return None
    idx = None
    for i, (l, h) in enumerate(zip(lo, hi)):
        if l <= target <= h:
            idx = i; break
    if idx is None: return None
    window = max(1, min(10, len(lo)//12))
    slice_lo = lo[max(0, idx-window):min(len(lo), idx+window)]
    slice_hi = hi[max(0, idx-window):min(len(hi), idx+window)]
    tight = max(1e-6, sum(h - l for l, h in zip(slice_lo, slice_hi)) / len(slice_lo))
    conf = max(0.0, min(1.0, 1.0 - tight))
    seconds = (idx+1) / hz
    return seconds, conf

# =========================
# Backtesting calibrator
# =========================
@dataclass
class BacktestCal:
    target_coverage: float = 0.8
    scaling: float = 1.0
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    def update_coverage(self, realized: List[float], lo: List[float], hi: List[float]):
        if not realized or not lo or not hi: return
        n = min(len(realized), len(lo), len(hi))
        covered = sum(1 for i in range(n) if lo[i] <= realized[i] <= hi[i])
        cov = covered / max(1, n)
        if cov < self.target_coverage - 0.05:
            self.scaling = min(1.6, self.scaling * 1.08)
        elif cov > self.target_coverage + 0.05:
            self.scaling = max(0.6, self.scaling * 0.94)
    def scale_bands(self, lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
        return ([max(0.0, 0.5 + (l-0.5)*self.scaling) for l in lo],
                [min(1.0, 0.5 + (h-0.5)*self.scaling) for h in hi])
    def update_alert_quality(self, predicted_events: List[Tuple[str, float]], realized_events: Set[str]):
        for label, conf in predicted_events:
            if label in realized_events:
                self.tp += 1
            else:
                self.fp += 1
        for label in realized_events:
            if label not in {p[0] for p in predicted_events}:
                self.fn += 1
        denom_p = max(1, self.tp + self.fp)
        denom_r = max(1, self.tp + self.fn)
        self.precision = self.tp / denom_p
        self.recall = self.tp / denom_r

# =========================
# Causality (lagged correlations + Granger-lite)
# =========================
def lagged_corr(a: List[float], b: List[float], max_lag: int = 60) -> Tuple[int, float]:
    best_lag = 0; best_corr = 0.0
    for lag in range(1, max_lag+1):
        a_cut = a[-(lag+240):-lag] if len(a) >= lag+240 else a[:-lag] if lag < len(a) else []
        b_cut = b[-240:] if len(b) >= 240 else b
        if not a_cut or not b_cut or len(a_cut) != len(b_cut): continue
        ma = sum(a_cut)/len(a_cut); mb = sum(b_cut)/len(b_cut)
        sa = math.sqrt(sum((x-ma)**2 for x in a_cut) / max(1, len(a_cut)-1)) or 1e-6
        sb = math.sqrt(sum((y-mb)**2 for y in b_cut) / max(1, len(b_cut)-1)) or 1e-6
        cov = sum((x-ma)*(y-mb) for x,y in zip(a_cut, b_cut)) / max(1, len(a_cut)-1)
        corr = cov / (sa*sb)
        if abs(corr) > abs(best_corr):
            best_corr = corr; best_lag = lag
    return best_lag, best_corr

def granger_lite(a: List[float], b: List[float], lag: int) -> float:
    if len(a) < lag+10 or len(b) < lag+10: return 0.0
    aL = a[:-lag]; bL = b[lag:]
    mb = sum(bL)/len(bL)
    var_b = sum((x-mb)**2 for x in bL) / max(1, len(bL)-1)
    ma = sum(aL)/len(aL)
    denom = sum((x-ma)**2 for x in aL) or 1e-6
    beta = sum((x-ma)*(y-mb) for x,y in zip(aL, bL)) / denom
    alpha = mb - beta*ma
    residuals = [y - (alpha + beta*x) for x,y in zip(aL, bL)]
    mr = sum(residuals)/len(residuals)
    var_r = sum((r-mr)**2 for r in residuals) / max(1, len(residuals)-1)
    improvement = max(0.0, min(1.0, (var_b - var_r) / max(1e-6, var_b)))
    return improvement

# Directed lead–lag graph
@dataclass
class LeadLagEdge:
    lag_s: int
    corr: float
    gain: float
    confidence: float

class LeadLagGraph:
    def __init__(self):
        self.edges: Dict[str, Dict[str, LeadLagEdge]] = {}
        self.decay: float = 0.98
        self.min_conf: float = 0.05
    def update_edge(self, src: str, dst: str, lag: int, corr: float, gain: float):
        conf_inc = max(0.0, min(1.0, abs(corr) * 0.5 + gain * 0.5))
        d = self.edges.setdefault(src, {})
        e = d.get(dst)
        if e:
            e.confidence = min(1.0, e.confidence * self.decay + conf_inc * (1.0 - self.decay))
            e.lag_s = lag; e.corr = corr; e.gain = gain
        else:
            d[dst] = LeadLagEdge(lag_s=lag, corr=corr, gain=gain, confidence=conf_inc)
    def prune(self):
        for src in list(self.edges.keys()):
            for dst in list(self.edges[src].keys()):
                self.edges[src][dst].confidence *= self.decay
                if self.edges[src][dst].confidence < self.min_conf:
                    del self.edges[src][dst]
            if not self.edges[src]:
                del self.edges[src]
    def propagate_eta(self, start: str, target_threshold: float, preds: Dict[str, Dict[str, List[float]]], bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]], hz: float = 10.0) -> Optional[Tuple[str, float, float]]:
        # BFS over graph, accumulate lag; pick path with highest minimum confidence
        best = None
        visited = set([start])
        queue: List[Tuple[str, float, float]] = [(start, 0.0, 1.0)]  # node, lag_s, path_conf
        while queue:
            node, lag_s, path_conf = queue.pop(0)
            # If we have mid-horizon bands for node, estimate an ETA to threshold
            lo, hi = bands.get(node, {}).get("M", ([], []))
            if lo and hi:
                eta_pack = hazard_eta(target_threshold, lo, hi, hz)
                if eta_pack:
                    eta_s, conf = eta_pack
                    total_eta = lag_s + eta_s
                    total_conf = min(path_conf, conf)
                    if best is None or total_conf > best[2]:
                        best = (node, total_eta, total_conf)
            for nxt, edge in self.edges.get(node, {}).items():
                if nxt in visited: continue
                visited.add(nxt)
                queue.append((nxt, lag_s + edge.lag_s, min(path_conf, edge.confidence)))
        return best

# =========================
# Scenario planner with adaptive weights
# =========================
@dataclass
class ScenarioWeights:
    cpu_burst: float = 1.0
    net_double: float = 1.0
    mem_leak: float = 1.0
    def adjust(self, outcomes: Dict[str, bool]):
        for name, hit in outcomes.items():
            if name == "cpu_burst_30":
                self.cpu_burst = min(2.2, self.cpu_burst * (1.08 if hit else 0.95))
            elif name == "net_double":
                self.net_double = min(2.2, self.net_double * (1.08 if hit else 0.95))
            elif name == "mem_leak":
                self.mem_leak = min(2.2, self.mem_leak * (1.08 if hit else 0.95))

def simulate_scenarios(current: Dict[str, float], w: ScenarioWeights) -> List[Dict]:
    cpu = current.get("cpu", 0.5)
    mem = current.get("mem", 0.5)
    net = current.get("net", 0.2)
    load = current.get("load", cpu)
    scenarios = []
    cpu1 = min(1.0, cpu + 0.30); load1 = min(1.0, (load + cpu1) / 2.0)
    risk1 = (0.5 * cpu1 + 0.25 * load1 + 0.25 * max(0.0, mem - 0.6)) * w.cpu_burst
    scenarios.append({"name": "cpu_burst_30", "risk": min(1.0, risk1), "notes": "cpu+load stress"})
    net2 = min(1.0, net * 2.0); trust2 = 1.0 - net2
    risk2 = (0.6 * net2 + 0.4 * (1.0 - trust2)) * w.net_double
    scenarios.append({"name": "net_double", "risk": min(1.0, risk2), "notes": "net surge → trust down"})
    mem3 = min(1.0, mem + 0.20)
    risk3 = (0.7 * mem3 + 0.3 * load) * w.mem_leak
    scenarios.append({"name": "mem_leak", "risk": min(1.0, risk3), "notes": "mem pressure"})
    return scenarios

# =========================
# Preemptive strategy stack
# =========================
@dataclass
class StrategyPlan:
    action: str  # "mutate" or "revert"
    eta_s: float
    reason: str
    created_ts: float
    cooldown_s: float = 12.0
    min_dwell_s: float = 10.0

# =========================
# Ensemble + Bayesian risk fusion
# =========================
@dataclass
class EnsembleWeights:
    kalman: float = 0.4
    ewma_linear: float = 0.35
    holt_winters: float = 0.25
    def normalize(self):
        s = max(1e-6, self.kalman + self.ewma_linear + self.holt_winters)
        self.kalman /= s; self.ewma_linear /= s; self.holt_winters /= s

def ensemble_blend(preds: List[List[float]], weights: EnsembleWeights) -> List[float]:
    weights.normalize()
    n = min(len(p) for p in preds) if preds else 0
    out = []
    for i in range(n):
        y = weights.kalman * preds[0][i] + weights.ewma_linear * preds[1][i] + weights.holt_winters * preds[2][i]
        out.append(min(1.0, max(0.0, y)))
    return out

def fuse_risk(volatility: float, band_tightness: float, causality_gain: float, backtest_precision: float) -> float:
    # Bayesian-style fusion (heuristic): higher volatility and wider bands raise risk; higher causality/backtest lower uncertainty
    base = min(1.0, 0.5*volatility + 0.5*band_tightness)
    gain_bonus = 0.3 * causality_gain + 0.3 * backtest_precision
    # Confidence acts to reduce effective risk by stabilizing forecasts
    effective = max(0.0, min(1.0, base - 0.25*gain_bonus))
    return effective

# =========================
# Feature extraction (slope, curvature, seasonality hint)
# =========================
@dataclass
class Features:
    slope: float = 0.0
    curvature: float = 0.0
    seasonality_strength: float = 0.0

def extract_features(series: List[float]) -> Features:
    if len(series) < 5:
        return Features()
    intercept, slope = linear_fit(series)
    # curvature: compare early, mid, late slopes
    n = len(series); third = n // 3 or 1
    _, s1 = linear_fit(series[:third])
    _, s2 = linear_fit(series[third:2*third])
    _, s3 = linear_fit(series[2*third:])
    curvature = (s3 - s1) - (s2 - s1)
    # seasonality hint: autocorrelation at small lag
    lag = min(20, n//5)
    if lag >= 3:
        xs = series[:-lag]; ys = series[lag:]
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
        sx = math.sqrt(sum((x-mx)**2 for x in xs) / max(1, len(xs)-1)) or 1e-6
        sy = math.sqrt(sum((y-my)**2 for y in ys) / max(1, len(ys)-1)) or 1e-6
        cov = sum((x-mx)*(y-my) for x,y in zip(xs, ys)) / max(1, len(xs)-1)
        ac = cov / (sx*sy)
    else:
        ac = 0.0
    return Features(slope=slope, curvature=curvature, seasonality_strength=max(0.0, ac))

# =========================
# Counterfactual simulator
# =========================
def counterfactual_strategy(th: Thresholds, action: str) -> Thresholds:
    new_th = Thresholds(**asdict(th))
    if action == "mutate":
        new_th.high_on = min(0.92, new_th.high_on + 0.02)
        new_th.mid_on = min(0.68, new_th.mid_on + 0.01)
        new_th.high_off = min(0.75, new_th.high_off + 0.01)
    elif action == "revert":
        new_th.high_on = max(0.68, new_th.high_on - 0.02)
        new_th.mid_on = max(0.50, new_th.mid_on - 0.01)
        new_th.high_off = max(0.60, new_th.high_off - 0.01)
    new_th.clamp()
    return new_th

def expected_risk_after_action(series: List[float], bands_lo: List[float], bands_hi: List[float], action: str, risk_base: float) -> float:
    # Penalize or reward risk depending on action and band width near front
    if not bands_lo or not bands_hi: return risk_base
    front = min(20, len(bands_lo))
    width = sum(bands_hi[i]-bands_lo[i] for i in range(front)) / max(1, front)
    if action == "mutate":
        return max(0.0, min(1.0, risk_base * (0.85 if width < 0.25 else 0.92)))
    else:
        return max(0.0, min(1.0, risk_base * (1.05 if width > 0.25 else 0.98)))

# =========================
# Autonomous engine (ultra-predictive)
# =========================
class AutonomousLightDark:
    def __init__(self, channels: List[str], state_path: str = "light_dark_state.json", log_path: str = "light_dark_telemetry.jsonl"):
        self.channels = channels
        self.state_path = state_path
        self.logger = AsyncLogger(log_path)

        self.history: Dict[str, RingBuffer] = {ch: RingBuffer(18000) for ch in channels}
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
        self.last_strategy_change_ts: float = 0.0

        self._listeners: List[Callable[[dict], None]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.reasoner = Reasoner(self.strategy.blend)
        self.metrics = MetricsProvider()

        self.executor = ThreadPoolExecutor(max_workers=10)

        self.pred_horizons = {"S": 60, "M": 120, "L": 300}
        self.pred: Dict[str, Dict[str, List[float]]] = {ch: {"S":[], "M":[], "L":[]} for ch in channels}
        self.bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {ch: {"S":([],[]), "M":([],[]), "L":([],[])} for ch in channels}
        self.pred_conf: Dict[str, float] = {ch: 0.5 for ch in channels}
        self.pred_anomaly_risk: float = 0.0
        self.pred_strategy_eta_s: Optional[float] = None
        self.pred_strategy_eta_conf: float = 0.0

        self.kalman: Dict[str, KalmanState] = {ch: KalmanState() for ch in channels}
        self.cusum: Dict[str, ChangePointState] = {ch: ChangePointState() for ch in channels}
        self.calibrator = BacktestCal()
        self.leadlag = LeadLagGraph()
        self.causal_links: List[Tuple[str, str, int, float, float]] = []  # (src,dst,lag,corr,gain)

        self.plan_stack: List[StrategyPlan] = []
        self.future_thoughts: List[Dict] = []
        self.scenarios: List[Dict] = []
        self.scenario_weights = ScenarioWeights()

        self.ensemble_weights: Dict[str, EnsembleWeights] = {ch: EnsembleWeights() for ch in channels}
        self.features: Dict[str, Features] = {ch: Features() for ch in channels}

        self.futures_forecast: Dict[str, Dict[str, Future]] = {ch: {} for ch in channels}
        self.future_causality: Optional[Future] = None
        self.future_scenarios: Optional[Future] = None

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
                    ps.version = data.get("version", 10)
                    for ch, p in data.get("profiles", {}).items():
                        th = Thresholds(**p.get("thresholds", {}))
                        prof = ChannelProfile(
                            jitter_avg=p.get("jitter_avg",0.0),
                            drift_avg=p.get("drift_avg",0.0),
                            imbalance_avg=p.get("imbalance_avg",0.0),
                            samples=p.get("samples",0),
                            thresholds=th,
                            alpha=p.get("alpha",0.15),
                            regime=p.get("regime","stable")
                        )
                        ps.profiles[ch] = prof
                    return ps
        except Exception as e:
            self._log_error("state_load_error", e)
        return PersistentState()

    def _save_state(self) -> None:
        try:
            obj = {
                "profiles": {ch: {"jitter_avg":p.jitter_avg,"drift_avg":p.drift_avg,"imbalance_avg":p.imbalance_avg,"samples":p.samples,"thresholds":asdict(p.thresholds),"alpha":p.alpha,"regime":p.regime} for ch,p in self.profiles.items()},
                "strategy_name": self.strategy.name,
                "version": 10
            }
            atomic_write_json(self.state_path, obj)
        except Exception as e:
            self._log_error("state_save_error", e)

    def _log_error(self, kind: str, e: Exception):
        self.logger.submit(kind, {"error": str(e), "trace": traceback.format_exc()})

    def _log_snapshot(self, kind: str, payload: dict):
        self.logger.submit(kind, payload)

    def start(self): self.thread.start()
    def stop(self):
        self.running = False
        self.thread.join(timeout=3.0)
        self._save_state()
        self.executor.shutdown(wait=False)
        self.logger.stop()

    # ---------- Threaded tasks ----------
    def _schedule_forecasts(self):
        for ch in self.channels:
            series = self.history[ch].tail(2000)
            if series:
                self.kalman[ch] = kalman_update(self.kalman[ch], series[-1])
                self.features[ch] = extract_features(series)
            wts = self.ensemble_weights[ch]
            # Short: Kalman + short + HW blend
            def f_short():
                fk, lok, hik = kalman_forecast(self.kalman[ch], self.pred_horizons["S"], band_scale=0.08)
                fS, loS, hiS = forecast_short(series, steps=self.pred_horizons["S"])
                fh, loh, hih = holt_winters(series, steps=self.pred_horizons["S"], alpha=0.55, beta=0.25)
                ff = ensemble_blend([fk, fS, fh], wts)
                lo2 = [min(1.0, max(0.0, ensemble_blend([[lok[i]],[loS[i]],[loh[i]]], wts)[0])) for i in range(min(len(lok),len(loS),len(loh)))]
                hi2 = [min(1.0, max(0.0, ensemble_blend([[hik[i]],[hiS[i]],[hih[i]]], wts)[0])) for i in range(min(len(hik),len(hiS),len(hih)))]
                return ff, lo2, hi2
            def f_mid():
                return forecast_mid(series, steps=self.pred_horizons["M"])
            def f_long():
                return forecast_long(series, steps=self.pred_horizons["L"])
            self.futures_forecast[ch]["S"] = self.executor.submit(f_short)
            self.futures_forecast[ch]["M"] = self.executor.submit(f_mid)
            self.futures_forecast[ch]["L"] = self.executor.submit(f_long)

    def _collect_forecasts(self):
        for ch in self.channels:
            for horiz in ("S","M","L"):
                fut = self.futures_forecast.get(ch, {}).get(horiz)
                if fut and fut.done():
                    try:
                        f, lo, hi = fut.result()
                        lo, hi = self.calibrator.scale_bands(lo, hi)
                        self.pred[ch][horiz] = f
                        self.bands[ch][horiz] = (lo, hi)
                        if horiz == "S" and lo and hi:
                            tightness = max(1e-6, sum((hi_i - lo_i) for lo_i, hi_i in zip(lo, hi)) / len(lo))
                            self.pred_conf[ch] = max(0.0, min(1.0, 1.0 - tightness))
                    except Exception as e:
                        self._log_error("forecast_error", e)

    def _schedule_causality(self):
        pairs = [("cpu","load"), ("net","trust"), ("disk","mem"), ("cpu","mem"), ("load","mem")]
        data = {k: self.history[k].to_list() for k,_ in pairs}
        def compute():
            links = []
            for a,b in pairs:
                la, lb = data.get(a, []), data.get(b, [])
                lag, corr = lagged_corr(la, lb, max_lag=60)
                gain = granger_lite(la, lb, lag) if lag > 0 else 0.0
                if abs(corr) > 0.25 and gain > 0.06:
                    links.append((a, b, lag, corr, gain))
            return links
        self.future_causality = self.executor.submit(compute)

    def _collect_causality(self):
        fut = self.future_causality
        if fut and fut.done():
            try:
                self.causal_links = fut.result()
                for src,dst,lag,corr,gain in self.causal_links:
                    self.leadlag.update_edge(src, dst, lag, corr, gain)
                self.leadlag.prune()
            except Exception as e:
                self._log_error("causality_error", e)

    def _schedule_scenarios(self, metrics: Dict[str, float]):
        self.future_scenarios = self.executor.submit(simulate_scenarios, {"cpu": metrics["cpu"], "mem": metrics["mem"], "net": metrics["net"], "load": metrics["load"]}, self.scenario_weights)

    def _collect_scenarios(self):
        fut = self.future_scenarios
        if fut and fut.done():
            try:
                self.scenarios = fut.result()
            except Exception as e:
                self._log_error("scenario_error", e)

    # ---------- Main loop ----------
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

                # Update signals + classification + change-point
                for ch in self.channels:
                    val = intensities_map.get(ch, 0.5)
                    sig = Signal(intensity=val); sig.update(self.noise)
                    self.history[ch].append(sig.intensity)
                    self.last_states[ch] = sig.classify(self.last_states[ch], self.thresholds[ch])
                    cp = self.cusum[ch] = update_cusum(self.cusum[ch], sig.intensity, mean_ref=0.5, k=0.02)
                    self.profiles[ch].regime = cp.regime
                    th = self.thresholds[ch]
                    if cp.regime == "spike":
                        th.high_on = min(0.92, th.high_on + 0.01); th.high_off = min(0.75, th.high_off + 0.01)
                    elif cp.regime == "drift":
                        th.mid_on = min(0.65, th.mid_on + 0.005); th.mid_off = min(0.6, th.mid_off + 0.005)
                    th.clamp()

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

                # Threaded computations
                self._schedule_forecasts()
                self._schedule_causality()
                self._schedule_scenarios(m)
                self._collect_forecasts()
                self._collect_causality()
                self._collect_scenarios()

                # Backtesting calibrator: coverage using short bands
                for ch in self.channels:
                    realized = self.history[ch].tail(len(self.pred[ch]["S"]))
                    loS, hiS = self.bands[ch]["S"]
                    self.calibrator.update_coverage(realized, loS, hiS)

                # Risk fusion (Bayesian-style)
                volatility = risk_from_jitter(self.anomaly_density.to_list(), window=300)
                tightness_avg = sum(max(0.0, (self.bands[ch]["S"][1][i] - self.bands[ch]["S"][0][i]) if self.bands[ch]["S"][0] and i < len(self.bands[ch]["S"][0]) else 0.0)
                                    for ch in self.channels for i in range(min(10, len(self.bands[ch]["S"][0]))))
                denom = sum(1 for ch in self.channels if self.bands[ch]["S"][0]) * max(1, min(10, max(len(self.bands[ch]["S"][0]) for ch in self.channels if self.bands[ch]["S"][0])))
                band_tightness = min(1.0, tightness_avg / max(1, denom))
                causality_gain = min(1.0, sum(edge.gain for src in self.leadlag.edges for edge in self.leadlag.edges[src].values()) / max(1, sum(len(v) for v in self.leadlag.edges.values())))
                self.pred_anomaly_risk = fuse_risk(volatility, band_tightness, causality_gain, self.calibrator.precision)

                # Hazard ETA on load high_on via lead-lag propagation
                bands_load_M = self.bands.get("load", {}).get("M", ([], []))
                eta_pack = hazard_eta(self.thresholds["load"].high_on, bands_load_M[0], bands_load_M[1], hz=self.rate_hz) if bands_load_M[0] else None
                if eta_pack:
                    self.pred_strategy_eta_s, self.pred_strategy_eta_conf = eta_pack
                else:
                    # try propagation from CPU to load, etc.
                    prop = self.leadlag.propagate_eta("cpu", self.thresholds["load"].high_on, self.pred, self.bands, hz=self.rate_hz)
                    if prop:
                        _, eta_s, eta_conf = prop
                        self.pred_strategy_eta_s, self.pred_strategy_eta_conf = eta_s, eta_conf
                    else:
                        self.pred_strategy_eta_s, self.pred_strategy_eta_conf = None, 0.0

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

                # Alert quality update
                realized_labels: Set[str] = set()
                if max(self.history["cpu"].tail(60)) > 0.85: realized_labels.add("cpu_forecast_high")
                if max(self.history["mem"].tail(120)) > 0.9: realized_labels.add("mem_forecast_pressure")
                self.calibrator.update_alert_quality([(f["label"], f["confidence"]) for f in self.future_thoughts], realized_labels)

                thoughts = self._judge_repair_and_think(protected)

                t1 = time.time()
                self.loop_ms = (t1 - t0) * 1000.0
                self.bottleneck_ms = max(self.bottleneck_ms * 0.95, self.loop_ms)

                # Counterfactuals for plan arbitration
                cf_mutate_th = counterfactual_strategy(self.thresholds["load"], "mutate")
                cf_revert_th = counterfactual_strategy(self.thresholds["load"], "revert")
                loM, hiM = self.bands.get("load", {}).get("M", ([],[]))
                risk_mut = expected_risk_after_action(self.history["load"].tail(200), loM, hiM, "mutate", self.pred_anomaly_risk)
                risk_rev = expected_risk_after_action(self.history["load"].tail(200), loM, hiM, "revert", self.pred_anomaly_risk)

                payload = {
                    "ts": t1,
                    "states": {ch: self.last_states[ch].name for ch in self.channels},
                    "series": {ch: self.history[ch].tail(240) for ch in self.channels},
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
                    "conf_trends": {ch: self.conf_trends[ch].tail(160) for ch in self.channels},
                    "anomaly_bar": self.anomaly_density.tail(160),
                    "pred": self.pred,
                    "bands": self.bands,
                    "pred_conf": self.pred_conf,
                    "pred_anomaly_risk": self.pred_anomaly_risk,
                    "pred_strategy_eta_s": self.pred_strategy_eta_s,
                    "pred_strategy_eta_conf": self.pred_strategy_eta_conf,
                    "future_thoughts": list(self.future_thoughts[-40:]),
                    "scenarios": list(self.scenarios[-10:]),
                    "causal_links": list(self.causal_links),
                    "leadlag": {src: {dst: asdict(edge) for dst,edge in edges.items()} for src,edges in self.leadlag.edges.items()},
                    "plans": [asdict(p) for p in self.plan_stack[-10:]],
                    "calibration": {"scaling": self.calibrator.scaling, "precision": self.calibrator.precision, "recall": self.calibrator.recall},
                    "counterfactuals": {"mutate_risk": risk_mut, "revert_risk": risk_rev}
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
        self.plan_stack = [p for p in self.plan_stack if (now - p.created_ts) < p.cooldown_s]
        high_risk = self.pred_anomaly_risk > 0.65
        low_risk = self.pred_anomaly_risk < 0.18
        eta, eta_conf = self.pred_strategy_eta_s, self.pred_strategy_eta_conf
        has_mutate = any(p.action == "mutate" for p in self.plan_stack)
        has_revert = any(p.action == "revert" for p in self.plan_stack)
        dwell_ok = (now - self.last_strategy_change_ts) > 10.0
        if high_risk and eta is not None and eta_conf > 0.55 and not has_mutate and dwell_ok:
            self.plan_stack.append(StrategyPlan("mutate", eta, "hazard_eta_high", now))
        elif low_risk and not has_revert and dwell_ok:
            self.plan_stack.append(StrategyPlan("revert", 6.0, "risk_low", now))

    def _apply_plans(self, now: float) -> Optional[str]:
        if not self.plan_stack: return None
        executed = None
        remaining = []
        for p in self.plan_stack:
            if (now - self.last_strategy_change_ts) < p.min_dwell_s:
                remaining.append(p); continue
            due = (now - p.created_ts) >= max(0.0, p.eta_s)
            if due and executed is None:
                if p.action == "mutate" and self.strategy.name != ROBUST_STRATEGY.name:
                    self.strategy = ROBUST_STRATEGY
                    self.reasoner = Reasoner(self.strategy.blend)
                    self.strategy_timeline.append((now, "mutate", self.strategy.name))
                    self.last_strategy_change_ts = now
                    executed = "mutated_to_robust_predictive_plan"
                elif p.action == "revert" and self.strategy.name != DEFAULT_STRATEGY.name:
                    self.strategy = DEFAULT_STRATEGY
                    self.reasoner = Reasoner(self.strategy.blend)
                    self.strategy_timeline.append((now, "revert", self.strategy.name))
                    self.last_strategy_change_ts = now
                    executed = "reverted_to_default_predictive_plan"
            else:
                remaining.append(p)
        self.plan_stack = remaining[-10:]
        return executed

    def _judge_repair_and_think(self, protected: ProtectedState) -> List[Dict]:
        series_all: List[float] = []
        for ch in self.channels: series_all += self.history[ch].tail(256)
        jitter_g, anomalies_g, mean_g = anomaly_score(series_all)

        pre_switch_to_robust = (self.pred_anomaly_risk > 0.65)
        pre_switch_to_default = (self.pred_anomaly_risk < 0.18)

        changed_thr = 0
        now = time.time()
        for ch in self.channels:
            series = self.history[ch].tail(256)
            jitter_c, anomalies_c, mean_c = anomaly_score(series)
            self.profiles[ch].update(jitter_c, mean_c, imbalance=0.0)
            th = self.thresholds[ch]
            new_th = Thresholds(th.low_on, th.mid_on, th.high_on, th.low_off, th.mid_off, th.high_off)
            regime = self.profiles[ch].regime
            if jitter_c > 0.22 or regime == "spike":
                new_th.low_on = min(0.55, new_th.low_on + 0.02)
                new_th.mid_on = min(0.65, new_th.mid_on + 0.02)
                new_th.high_on = min(0.90, new_th.high_on + 0.02)
            if abs(mean_c - 0.5) > 0.25 or regime == "drift":
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

        want_robust_reactive = jitter_g > 0.28 or "mean_drift" in " ".join(anomalies_g)
        want_default_reactive = jitter_g < 0.12
        strat_changed = None

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
                self.last_strategy_change_ts = now
                strat_changed = "mutated_to_robust" + ("_predictive" if pre_switch_to_robust else "_reactive")
            elif (want_default_reactive or pre_switch_to_default) and self.strategy.name != DEFAULT_STRATEGY.name:
                self.strategy = DEFAULT_STRATEGY
                self.reasoner = Reasoner(self.strategy.blend)
                self.strategy_timeline.append((now, "revert", self.strategy.name))
                if len(self.strategy_timeline) > 1000:
                    self.strategy_timeline = self.strategy_timeline[-1000:]
                self.last_strategy_change_ts = now
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
            predictive_notes.append(f"strategy_eta_s≈{self.pred_strategy_eta_s:.1f}(p≈{self.pred_strategy_eta_conf:.2f})")
        for a,b,lag,corr,imp in self.causal_links[:5]:
            predictive_notes.append(f"{a}->{b}(lag={lag},corr={corr:.2f},gain={imp:.2f})")

        cpuS = self.pred.get("cpu", {}).get("S", [])
        memM = self.pred.get("mem", {}).get("M", [])
        if cpuS and max(cpuS) > 0.85: predictive_notes.append("forecast_cpu_high")
        if memM and max(memM) > 0.9: predictive_notes.append("forecast_mem_pressure")

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
        if len(self.thought_log) > 3200:
            self.thought_log = self.thought_log[-3200:]
        try:
            self._log_snapshot("thoughts", {"count": len(thoughts_serialized)})
        except Exception:
            pass
        return thoughts_serialized

    @staticmethod
    def _serialize_thought(t: Thought) -> Dict:
        return {"label": t.label, "conclusion": t.conclusion.name, "confidence": t.confidence,
                "evidence": {k: v.name for k, v in t.evidence.items()}, "notes": list(t.notes), "ts": time.time()}

# =========================
# GUI (Queue-decoupled, 480ms refresh, ensemble/lead-lag/counterfactual visuals)
# =========================
import tkinter as tk
from tkinter import ttk

class ScientificGUI:
    def __init__(self, engine: AutonomousLightDark):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Predictive Skeleton Monitor (Ensemble, Lead-Lag, Counterfactual)")
        self.root.geometry("1020x720")
        self.root.configure(bg="#111111")

        self.payload_q: Queue = Queue(maxsize=32)
        engine.register_listener(lambda payload: self._enqueue_payload(payload))

        self.status_text = tk.StringVar(value="Initializing…")
        top = ttk.Frame(self.root); top.pack(fill="x", padx=4, pady=4)
        ttk.Label(top, textvariable=self.status_text).pack(side="left")

        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True, padx=4, pady=4)
        self.overview = ttk.Frame(self.tabs); self.analytics = ttk.Frame(self.tabs); self.thoughts_tab = ttk.Frame(self.tabs); self.graph_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.overview, text="Overview"); self.tabs.add(self.analytics, text="Analytics"); self.tabs.add(self.thoughts_tab, text="Thoughts"); self.tabs.add(self.graph_tab, text="Lead-Lag")

        self.tile_frame = ttk.Frame(self.overview); self.tile_frame.pack(fill="x", padx=4, pady=4)
        self.tile_vars: Dict[str, tk.StringVar] = {ch: tk.StringVar(value="AMBIENT") for ch in engine.channels}
        self.gauges: Dict[str, tk.Canvas] = {}
        for i, ch in enumerate(engine.channels):
            card = ttk.LabelFrame(self.tile_frame, text=ch); card.grid(row=0, column=i % 6, padx=3, pady=3, sticky="nsew")
            row_inner = ttk.Frame(card); row_inner.pack(fill="x")
            ttk.Label(row_inner, textvariable=self.tile_vars[ch], font=("Consolas", 8)).pack(side="left", padx=4, pady=4)
            g = tk.Canvas(row_inner, width=90, height=16, bg="#181818", highlightthickness=0); g.pack(side="right", padx=4, pady=4)
            self.gauges[ch] = g
            self.tile_frame.grid_columnconfigure(i % 6, weight=1)

        self.toggle_frame = ttk.Frame(self.overview); self.toggle_frame.pack(fill="x", padx=4, pady=(0,6))
        self.show_S = tk.BooleanVar(value=True); self.show_M = tk.BooleanVar(value=True); self.show_L = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toggle_frame, text="Short (S)", variable=self.show_S).pack(side="left", padx=6)
        ttk.Checkbutton(self.toggle_frame, text="Mid (M)", variable=self.show_M).pack(side="left", padx=6)
        ttk.Checkbutton(self.toggle_frame, text="Long (L)", variable=self.show_L).pack(side="left", padx=6)

        self.summary = ttk.Frame(self.overview); self.summary.pack(fill="x", padx=4, pady=4)
        self.summary_prot = tk.StringVar(value="Protection: —")
        self.summary_health = tk.StringVar(value="Health: —")
        self.summary_strategy = tk.StringVar(value="Strategy: —")
        self.summary_eta = tk.StringVar(value="Predicted strategy ETA: —")
        self.summary_quality = tk.StringVar(value="Alert quality: —")
        self.summary_cf = tk.StringVar(value="Counterfactual: —")
        ttk.Label(self.summary, textvariable=self.summary_prot, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_health, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_strategy, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_eta, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_quality, font=("Consolas", 8)).pack(anchor="w")
        ttk.Label(self.summary, textvariable=self.summary_cf, font=("Consolas", 8)).pack(anchor="w")

        self.chart_grid = ttk.Frame(self.overview); self.chart_grid.pack(fill="both", expand=True, padx=4, pady=4)
        self.series_canvases: Dict[str, tk.Canvas] = {}
        self.conf_canvases: Dict[str, tk.Canvas] = {}
        self.agree_canvases: Dict[str, tk.Canvas] = {}
        for i, ch in enumerate(engine.channels):
            c_series = tk.Canvas(self.chart_grid, width=300, height=140, bg="#1b1b1b", highlightthickness=0)
            c_series.grid(row=(i // 3)*3, column=i % 3, padx=3, pady=3, sticky="nsew"); self.series_canvases[ch] = c_series
            c_conf = tk.Canvas(self.chart_grid, width=300, height=50, bg="#151515", highlightthickness=0)
            c_conf.grid(row=(i // 3)*3+1, column=i % 3, padx=3, pady=3, sticky="nsew"); self.conf_canvases[ch] = c_conf
            c_ag = tk.Canvas(self.chart_grid, width=300, height=16, bg="#121212", highlightthickness=0)
            c_ag.grid(row=(i // 3)*3+2, column=i % 3, padx=3, pady=3, sticky="nsew"); self.agree_canvases[ch] = c_ag
        for r in range(((len(engine.channels)+2)//3)*3): self.chart_grid.grid_rowconfigure(r, weight=1)
        for c in range(3): self.chart_grid.grid_columnconfigure(c, weight=1)

        self.analytics_top = ttk.Frame(self.analytics); self.analytics_top.pack(fill="x", padx=4, pady=4)
        self.timeline_text = tk.StringVar(value="Timelines: —")
        ttk.Label(self.analytics_top, textvariable=self.timeline_text, font=("Consolas", 8)).pack(anchor="w")
        self.timeline_canvas = tk.Canvas(self.analytics, width=980, height=60, bg="#1b1b1b", highlightthickness=0)
        self.timeline_canvas.pack(fill="x", padx=4, pady=4)
        self.heat_canvas = tk.Canvas(self.analytics, width=980, height=160, bg="#1b1b1b", highlightthickness=0)
        self.heat_canvas.pack(fill="x", padx=4, pady=4)
        self.anomaly_canvas = tk.Canvas(self.analytics, width=980, height=45, bg="#151515", highlightthickness=0)
        self.anomaly_canvas.pack(fill="x", padx=4, pady=4)
        ttk.Label(self.analytics, text="Predicted events:", font=("Consolas", 8)).pack(anchor="w", padx=4)
        self.future_list = tk.Listbox(self.analytics, height=6)
        self.future_list.pack(fill="x", padx=4, pady=2)
        ttk.Label(self.analytics, text="Scenarios:", font=("Consolas", 8)).pack(anchor="w", padx=4, pady=(8,2))
        self.scenario_list = tk.Listbox(self.analytics, height=5)
        self.scenario_list.pack(fill="x", padx=4, pady=2)
        ttk.Label(self.analytics, text="Alert quality (precision/recall):", font=("Consolas", 8)).pack(anchor="w", padx=4, pady=(8,2))
        self.quality_canvas = tk.Canvas(self.analytics, width=980, height=40, bg="#151515", highlightthickness=0)
        self.quality_canvas.pack(fill="x", padx=4, pady=4)

        ttk.Label(self.graph_tab, text="Lead–Lag Map (edges with confidence)", font=("Consolas", 9)).pack(anchor="w", padx=4, pady=4)
        self.graph_canvas = tk.Canvas(self.graph_tab, width=980, height=400, bg="#1b1b1b", highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        self.th_tree = ttk.Treeview(self.thoughts_tab, columns=("label","conclusion","confidence","evidence","notes","ts"), show="headings")
        for col in ("label","conclusion","confidence","evidence","notes","ts"):
            self.th_tree.heading(col, text=col)
            self.th_tree.column(col, width=120 if col!="evidence" else 240, stretch=True)
        self.th_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.th_narrative = tk.Text(self.thoughts_tab, height=8)
        self.th_narrative.pack(fill="x", expand=False, padx=4, pady=4)

        self.payload_latest: Optional[dict] = None
        self.root.after(480, self.refresh)
        self.engine.start()

    def _enqueue_payload(self, payload: dict):
        try:
            while not self.payload_q.empty():
                self.payload_q.get_nowait()
            self.payload_q.put_nowait(payload)
        except Exception:
            pass

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.engine.stop()

    def refresh(self):
        try:
            self.payload_latest = self.payload_q.get_nowait()
        except Empty:
            pass
        payload = self.payload_latest
        if payload:
            risk = payload.get('pred_anomaly_risk', 0.0)
            self.status_text.set(f"Loop={payload['loop_ms']:.1f}ms | Bottleneck={payload['bottleneck_ms']:.1f}ms | Strategy={payload['strategy']} | Health={payload['health']:.2f} | Risk={risk:.2f}")
            for ch, st in payload["states"].items(): self.tile_vars[ch].set(st)
            prot = payload["protected"]
            anomalies = ", ".join(prot.anomalies) if prot.anomalies else "none"
            voted = "[" + ", ".join(s.name for s in prot.voted_from) + "]"
            self.summary_prot.set(f"Protection: {prot.state.name} conf={prot.confidence:.2f} parity={prot.parity_ok} strategy={prot.strategy} anomalies={anomalies} from={voted}")
            self.summary_health.set(f"Health: {payload['health']:.2f}")
            self.summary_strategy.set(f"Strategy: {payload['strategy']}")
            eta = payload.get("pred_strategy_eta_s", None)
            etac = payload.get("pred_strategy_eta_conf", 0.0)
            self.summary_eta.set(f"Predicted strategy ETA: {'~'+str(round(eta,1))+'s' if eta is not None else '—'} (p≈{etac:.2f})")
            cal = payload.get("calibration", {})
            self.summary_quality.set(f"Alert quality: precision≈{cal.get('precision',0.0):.2f} | recall≈{cal.get('recall',0.0):.2f} | band_scale≈{cal.get('scaling',1.0):.2f}")
            cf = payload.get("counterfactuals", {})
            self.summary_cf.set(f"Counterfactuals: mutate→risk≈{cf.get('mutate_risk', float('nan')):.2f} | revert→risk≈{cf.get('revert_risk', float('nan')):.2f}")

            for ch, g in self.gauges.items():
                self.draw_gauge(g, payload["pred_conf"].get(ch, 0.5))

            for ch in self.engine.channels:
                preds = payload["pred"].get(ch, {})
                bands = payload["bands"].get(ch, {})
                self.draw_series(self.series_canvases[ch],
                                 payload["series"].get(ch, []),
                                 payload["thresholds"][ch],
                                 preds, bands,
                                 show_S=self.show_S.get(), show_M=self.show_M.get(), show_L=self.show_L.get())
                self.draw_confidence(self.conf_canvases[ch], payload["conf_trends"].get(ch, []))
                self.draw_agreement(self.agree_canvases[ch], preds)

            strat_events = len(payload["strategy_timeline"]); thr_events = len(payload["threshold_timeline"])
            self.timeline_text.set(f"Timelines: strategy={strat_events} thresholds={thr_events}")
            self.draw_timeline(self.timeline_canvas, payload["strategy_timeline"])
            self.draw_heatmap(self.heat_canvas, payload["series"])
            self.draw_anomaly_bar(self.anomaly_canvas, payload["anomaly_bar"])
            self.draw_quality(self.quality_canvas, payload.get("calibration", {}))
            self.draw_leadlag(self.graph_canvas, payload.get("leadlag", {}))

            self.update_thoughts(payload["thoughts"])
            self.update_narrative(payload["thought_log"])
            self.update_future(payload.get("future_thoughts", []))
            self.update_scenarios(payload.get("scenarios", []))
        self.root.after(480, self.refresh)

    def update_thoughts(self, thoughts: List[Dict]):
        self.th_tree.delete(*self.th_tree.get_children())
        for t in thoughts:
            ev = ";".join(f"{k}={v}" for k,v in t["evidence"].items())
            notes = ",".join(t["notes"])
            ts = "{:.3f}".format(t["ts"])
            self.th_tree.insert("", "end", values=(t["label"], t["conclusion"], f"{t['confidence']:.2f}", ev, notes, ts))

    def update_narrative(self, log_items: List[Dict]):
        self.th_narrative.delete("1.0", "end")
        for t in log_items[-80:]:
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

    def draw_quality(self, canvas: tk.Canvas, cal: Dict[str, float]):
        p = cal.get("precision", 0.0); r = cal.get("recall", 0.0)
        scale = cal.get("scaling", 1.0)
        canvas.delete("all")
        w = int(canvas.winfo_width() or 980); h = int(canvas.winfo_height() or 40)
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        canvas.create_text(8, 8, text=f"Precision", anchor="nw", fill="#cccccc", font=("Consolas", 8))
        canvas.create_rectangle(80, 8, 80 + int(p*(w-160)), 18, fill="#66cc88", outline="")
        canvas.create_text(8, 22, text=f"Recall", anchor="nw", fill="#cccccc", font=("Consolas", 8))
        canvas.create_rectangle(80, 22, 80 + int(r*(w-160)), 32, fill="#ffaa44", outline="")
        canvas.create_text(w-160, 8, text=f"Band scale≈{scale:.2f}", anchor="nw", fill="#cccccc", font=("Consolas", 8))

    def draw_gauge(self, canvas: tk.Canvas, val: float):
        canvas.delete("all")
        w = int(canvas.winfo_width() or 90); h = int(canvas.winfo_height() or 16)
        canvas.create_rectangle(0, 0, w, h, fill="#181818", outline="")
        bar_w = int(val * w)
        color = "#66cc88" if val >= 0.66 else "#ffaa44" if val >= 0.33 else "#ff6666"
        canvas.create_rectangle(0, 0, bar_w, h, fill=color, outline="")
        canvas.create_rectangle(w-14, 2, w-2, h-2, fill=color, outline="")

    def draw_series(self, canvas: tk.Canvas, series: List[float], th: Thresholds,
                    pred_map: Dict[str, List[float]], bands_map: Dict[str, Tuple[List[float], List[float]]],
                    show_S: bool, show_M: bool, show_L: bool):
        w = int(canvas.winfo_width() or 300); h = int(canvas.winfo_height() or 140)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        for y in [h*0.3, h*0.6]: canvas.create_line(0, y, w, y, fill="#333333")
        if series:
            n = len(series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-12)) - 6) for val in series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#66ccff", width=2)
        def yv(v): return h - int(v*(h-12)) - 6
        canvas.create_line(6, yv(th.low_on),  w-6, yv(th.low_on),  fill="#88aa44", dash=(4,3))
        canvas.create_line(6, yv(th.mid_on),  w-6, yv(th.mid_on),  fill="#44cc88", dash=(4,3))
        canvas.create_line(6, yv(th.high_on), w-6, yv(th.high_on), fill="#ffaa44", dash=(4,3))
        canvas.create_line(6, yv(th.low_off), w-6, yv(th.low_off), fill="#aa88ff", dash=(4,3))
        canvas.create_line(6, yv(th.mid_off), w-6, yv(th.mid_off), fill="#8888aa", dash=(4,3))
        canvas.create_line(6, yv(th.high_off),w-6, yv(th.high_off),fill="#ff8888", dash=(4,3))
        styles = {"S":("#ffaa44",(3,2),2), "M":("#44dd88",(4,3),2), "L":("#ff6666",(5,3),2)}
        for horiz, show in (("S",show_S),("M",show_M),("L",show_L)):
            if not show: continue
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
        # ETA markers (mid horizon)
        loM, hiM = bands_map.get("M", ([], []))
        if show_M and loM and hiM:
            idx_up = next((i for i,(l,h) in enumerate(zip(loM,hiM)) if l >= th.high_on or h >= th.high_on), None)
            idx_down = next((i for i,(l,h) in enumerate(zip(loM,hiM)) if l <= th.high_off or h <= th.high_off), None)
            def x_at(i, n): return int(i*(w-20)/max(1, n-1)) + 6
            if idx_up is not None:
                x = x_at(idx_up, len(hiM)); canvas.create_line(x, 6, x, h-6, fill="#ffaa44"); canvas.create_text(x+3, 8, text="ETA↑", anchor="nw", fill="#cccccc", font=("Consolas", 7))
            if idx_down is not None:
                x = x_at(idx_down, len(hiM)); canvas.create_line(x, 6, x, h-6, fill="#44ccff"); canvas.create_text(x+3, 8, text="ETA↓", anchor="nw", fill="#cccccc", font=("Consolas", 7))

    def draw_confidence(self, canvas: tk.Canvas, conf_series: List[float]):
        w = int(canvas.winfo_width() or 300); h = int(canvas.winfo_height() or 50)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        if conf_series:
            n = len(conf_series)
            xs = [int(i*(w-20)/max(1, n-1)) + 6 for i in range(n)]
            ys = [int(h - (val*(h-10)) - 5) for val in conf_series]
            for i in range(1, n): canvas.create_line(xs[i-1], ys[i-1], xs[i], ys[i], fill="#44dd88", width=2)
        canvas.create_line(6, h-6, w-6, h-6, fill="#333333")

    def draw_agreement(self, canvas: tk.Canvas, preds: Dict[str, List[float]]):
        w = int(canvas.winfo_width() or 300); h = int(canvas.winfo_height() or 16)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#121212", outline="")
        S, M, L = preds.get("S", []), preds.get("M", []), preds.get("L", [])
        agree = 0.5
        if S and M and L:
            n = min(len(S), len(M), len(L), 60)
            agree = 1.0 - (sum(abs(S[i]-M[i]) + abs(M[i]-L[i]) + abs(S[i]-L[i]) for i in range(n)) / (3*n))
            agree = max(0.0, min(1.0, agree))
        color = "#66cc88" if agree >= 0.66 else "#ffaa44" if agree >= 0.33 else "#ff6666"
        canvas.create_rectangle(2, 2, int(2 + agree*(w-4)), h-2, fill=color, outline="")
        canvas.create_text(w-80, 2, text=f"agree≈{agree:.2f}", anchor="nw", fill="#cccccc", font=("Consolas", 8))

    def draw_timeline(self, canvas: tk.Canvas, events: List[Tuple[float, str, str]]):
        w = int(canvas.winfo_width() or 980); h = int(canvas.winfo_height() or 60)
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
        w = int(canvas.winfo_width() or 980); h = int(canvas.winfo_height() or 160)
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
        w = int(canvas.winfo_width() or 980); h = int(canvas.winfo_height() or 45)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, w, h, fill="#151515", outline="")
        if not anomaly_series: return
        n = len(anomaly_series); bw = max(2, int((w-12)/n)); x = 6
        for val in anomaly_series:
            height = int(val * (h-10))
            color = "#ff6666" if val > 0.2 else "#ffaa44" if val > 0.1 else "#66cc88"
            canvas.create_rectangle(x, h-6-height, x+bw, h-6, fill=color, outline="")
            x += bw

    def draw_leadlag(self, canvas: tk.Canvas, graph: Dict[str, Dict[str, Dict]]):
        canvas.delete("all")
        w = int(canvas.winfo_width() or 980); h = int(canvas.winfo_height() or 400)
        canvas.create_rectangle(0, 0, w, h, fill="#1b1b1b", outline="")
        nodes = list(graph.keys())
        for src, edges in graph.items():
            for dst in edges.keys():
                if dst not in nodes: nodes.append(dst)
        if not nodes: return
        # Place nodes in circle
        R = min(w,h)//2 - 40
        cx, cy = w//2, h//2
        pos: Dict[str, Tuple[int,int]] = {}
        for i, node in enumerate(nodes):
            ang = 2*math.pi*i/len(nodes)
            x = int(cx + R*math.cos(ang)); y = int(cy + R*math.sin(ang))
            pos[node] = (x, y)
        # Draw edges
        for src, edges in graph.items():
            for dst, ed in edges.items():
                x1,y1 = pos.get(src, (cx,cy)); x2,y2 = pos.get(dst, (cx,cy))
                conf = ed.get("confidence", 0.0)
                width = max(1, int(1 + conf*4))
                color = "#66cc88" if conf >= 0.66 else "#ffaa44" if conf >= 0.33 else "#ff6666"
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width, arrow="last")
                canvas.create_text((x1+x2)//2, (y1+y2)//2, text=f"lag={ed.get('lag_s',0)} gain={ed.get('gain',0.0):.2f}", fill="#cccccc", font=("Consolas", 7))
        # Draw nodes
        for node, (x,y) in pos.items():
            canvas.create_oval(x-12,y-12,x+12,y+12, fill="#2a2a2a", outline="#cccccc")
            canvas.create_text(x, y-18, text=node, fill="#cccccc", font=("Consolas", 8))

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

