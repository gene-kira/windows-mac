from __future__ import annotations
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Tuple, Callable, Optional, Set
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# =========================
# Optional psutil loader (system + process metrics)
# =========================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# =========================
# Utilities: async logger, ring buffer
# =========================
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
    version: int = 11

# =========================
# Metrics provider (psutil + synthetic fallback)
# =========================
class MetricsProvider:
    def __init__(self, game_hint_names: Optional[List[str]] = None):
        self.last_net = None
        self.last_disk = None
        self.last_ts = None
        self.game_hint_names = (game_hint_names or [
            "steam","epic","riot","battle.net","blizzard","valorant","league",
            "fortnite","cod","warzone","cs2","csgo","minecraft","roblox","pubg","apex"
        ])
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
    def _find_game_pressure(self) -> float:
        if not PSUTIL_AVAILABLE: return 0.0
        score = 0.0
        try:
            for p in psutil.process_iter(["name","exe","cmdline","cpu_percent","memory_info"]):
                name = (p.info.get("name") or "").lower()
                exe = (p.info.get("exe") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if any(h in name or h in exe or h in cmd for h in self.game_hint_names):
                    cpu = (p.info.get("cpu_percent") or 0.0) / 100.0
                    mem = (getattr(p.info.get("memory_info"), "rss", 0) / (2.5 * 1024 * 1024 * 1024))
                    score += min(1.0, 0.6*cpu + 0.4*min(1.0, mem))
        except Exception:
            pass
        return max(0.0, min(1.0, score))
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
                disk_norm = min(1.0, disk_rate / 50_000_000)
                net_norm  = min(1.0, net_rate  / 25_000_000)
                proc_count = len(psutil.pids())
                proc_norm = min(1.0, proc_count / 2000.0)
                try:
                    load1, _, _ = os.getloadavg()
                    cores = psutil.cpu_count(logical=True) or 1
                    load_norm = min(1.0, load1 / max(1, cores))
                except Exception:
                    load_norm = cpu
                game_pressure = self._find_game_pressure()
                intent = min(1.0, 0.4*game_pressure + 0.6*cpu)
                metrics = {
                    "cpu": float(cpu),"mem": float(mem),"disk": float(disk_norm),
                    "net": float(net_norm),"proc": float(proc_norm),"load": float(load_norm),
                    "intent": float(intent)
                }
                self.last_ts = ts; self.last_disk = disk_io; self.last_net = net_io
            except Exception:
                metrics = self.synthetic_fallback()
        else:
            metrics = self.synthetic_fallback()
        return metrics
    def synthetic_fallback(self) -> Dict[str, float]:
        def walk(v): return min(1.0, max(0.0, v + random.uniform(-0.05, 0.05)))
        base = getattr(self, "_fallback_state", {"cpu":0.4,"mem":0.55,"disk":0.2,"net":0.15,"proc":0.1,"load":0.35,"intent":0.4})
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
    cps.regime = "spike" if (cps.pos_sum > cps.threshold or cps.neg_sum > cps.threshold) else ("drift" if abs(cps.drift) > 0.1 else "stable")
    if cps.regime == "spike":
        cps.pos_sum *= 0.5; cps.neg_sum *= 0.5
    return cps

# =========================
# Kalman (level + slope)
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

def holt_winters(series: List[float], steps: int, alpha: float = 0.55, beta: float = 0.25) -> Tuple[List[float], List[float], List[float]]:
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
# Causality (lagged correlations + Granger-lite) + Lead–Lag graph
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
    def propagate_eta(self, start: str, target_threshold: float, bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]], hz: float = 10.0) -> Optional[Tuple[str, float, float]]:
        best = None
        visited = set([start])
        queue: List[Tuple[str, float, float]] = [(start, 0.0, 1.0)]
        while queue:
            node, lag_s, path_conf = queue.pop(0)
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
# Prometheus-style Probe (scan tunnels)
# =========================
@dataclass
class ProbeResult:
    path: List[str]
    eta_s: float
    confidence: float
    hops: int
    target_threshold: float

class Probe:
    def __init__(self, graph: LeadLagGraph, bands: Dict[str, Dict[str, Tuple[List[float], List[float]]]], hz: float = 10.0):
        self.graph = graph
        self.bands = bands
        self.hz = hz
    def _eta_local(self, node: str, target_threshold: float) -> Optional[Tuple[float, float]]:
        lo, hi = self.bands.get(node, {}).get("M", ([], []))
        if not lo or not hi:
            return None
        return hazard_eta(target_threshold, lo, hi, hz=self.hz)
    def scan(self, start: str, target_threshold: float, max_depth: int = 5) -> Optional[ProbeResult]:
        best: Optional[ProbeResult] = None
        visited: Set[str] = set()
        def dfs(node: str, depth: int, acc_eta: float, acc_conf: float, path: List[str]):
            nonlocal best
            visited.add(node)
            path.append(node)
            local_eta = self._eta_local(node, target_threshold)
            if local_eta:
                eta_s, eta_conf = local_eta
                total_eta = acc_eta + eta_s
                total_conf = min(acc_conf, eta_conf)
                candidate = ProbeResult(path=list(path), eta_s=total_eta, confidence=total_conf, hops=len(path)-1, target_threshold=target_threshold)
                if best is None or candidate.confidence > best.confidence or (abs(candidate.confidence - best.confidence) < 1e-6 and candidate.eta_s < best.eta_s):
                    best = candidate
            if depth >= max_depth:
                path.pop(); visited.remove(node); return
            for nxt, edge in self.graph.edges.get(node, {}).items():
                if nxt in visited: continue
                next_conf = min(acc_conf, edge.confidence)
                next_eta = acc_eta + max(0.0, edge.lag_s)
                dfs(nxt, depth+1, next_eta, next_conf, path)
            path.pop(); visited.remove(node)
        dfs(start, 0, 0.0, 1.0, [])
        return best

# =========================
# Scenario planner
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
# Policy & counterfactual helpers
# =========================
@dataclass
class StrategyPlan:
    action: str
    eta_s: float
    reason: str
    created_ts: float
    cooldown_s: float = 12.0
    min_dwell_s: float = 10.0

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

def fuse_risk(volatility: float, band_tightness: float, causality_gain: float, precision: float, recall: float) -> float:
    base = min(1.0, 0.45*volatility + 0.55*band_tightness)
    gain_bonus = 0.2 * causality_gain + 0.2 * precision + 0.1 * recall
    effective = max(0.0, min(1.0, base - 0.35*gain_bonus))
    return effective

@dataclass
class Features:
    slope: float = 0.0
    curvature: float = 0.0
    seasonality_strength: float = 0.0
    burst_density: float = 0.0

def extract_features(series: List[float]) -> Features:
    if len(series) < 5:
        return Features()
    intercept, slope = linear_fit(series)
    n = len(series); third = n // 3 or 1
    _, s1 = linear_fit(series[:third])
    _, s2 = linear_fit(series[third:2*third])
    _, s3 = linear_fit(series[2*third:])
    curvature = (s3 - s1) - (s2 - s1)
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
    tail = series[-200:] if len(series) >= 200 else series
    bd = sum(1 for x in tail if x > 0.8) / max(1, len(tail))
    return Features(slope=slope, curvature=curvature, seasonality_strength=max(0.0, ac), burst_density=bd)

def counterfactual_thresholds(th: Thresholds, action: str, regime: str) -> Thresholds:
    new_th = Thresholds(**asdict(th))
    if action == "mutate":
        inc = 0.02 if regime != "spike" else 0.01
        new_th.high_on = min(0.92, new_th.high_on + inc)
        new_th.mid_on = min(0.68, new_th.mid_on + inc*0.5)
        new_th.high_off = min(0.75, new_th.high_off + inc*0.5)
    elif action == "revert":
        dec = 0.02 if regime == "stable" else 0.01
        new_th.high_on = max(0.68, new_th.high_on - dec)
        new_th.mid_on = max(0.50, new_th.mid_on - dec*0.5)
        new_th.high_off = max(0.60, new_th.high_off - dec*0.5)
    new_th.clamp()
    return new_th

def expected_risk(series: List[float], lo: List[float], hi: List[float], base_risk: float, action: str) -> float:
    if not lo or not hi: return base_risk
    front = min(25, len(lo))
    width = sum(hi[i]-lo[i] for i in range(front)) / max(1, front)
    disagree_penalty = 0.0
    if width > 0.25: disagree_penalty += 0.08
    if action == "mutate":
        return max(0.0, min(1.0, base_risk * (0.85 if width < 0.25 else 0.92) - disagree_penalty))
    elif action == "revert":
        return max(0.0, min(1.0, base_risk * (1.05 if width > 0.25 else 0.98) + disagree_penalty*0.5))
    else:
        return base_risk

@dataclass
class PolicyChoice:
    action: str
    eta_s: float
    expected_risk: float
    margin_vs_next: float

# =========================
# Engine
# =========================
class AutonomousLightDark:
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.logger = AsyncLogger("telemetry.jsonl")
        self.history = {ch: RingBuffer(2000) for ch in channels}
        self.last_states = {ch: Illumination.AMBIENT for ch in channels}
        self.thresholds = {ch: Thresholds() for ch in channels}
        self.rate_hz = 2   # slowed loop
        self.executor = ThreadPoolExecutor(max_workers=12)
        self.metrics = MetricsProvider()
        self.reasoner = Reasoner(blend_linear)
        self.graph = LeadLagGraph()
        self.probe = Probe(self.graph, {}, hz=self.rate_hz)
        self.scenario_weights = ScenarioWeights()
        self.backtest = BacktestCal()
        self._listeners: List[Callable[[dict], None]] = []
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def register_listener(self, cb: Callable[[dict], None]):
        self._listeners.append(cb)

    def _emit(self, payload: dict):
        for cb in self._listeners:
            try: cb(payload)
            except Exception: pass

    def _run(self):
        while self.running:
            try:
                metrics = self.metrics.sample()
                states = {}
                for ch, val in metrics.items():
                    sig = Signal(val)
                    self.history[ch].append(val)
                    states[ch] = sig.classify(self.last_states[ch], self.thresholds[ch])
                    self.last_states[ch] = states[ch]

                prot = protect(list(states.values()), list(metrics.values()), DEFAULT_STRATEGY)
                thoughts = self.reasoner.reason(states)
                scenarios = simulate_scenarios(metrics, self.scenario_weights)
                probe_result = self.probe.scan("cpu", 0.7)

                payload = {
                    "ts": time.time(),
                    "metrics": metrics,
                    "states": {ch: states[ch].name for ch in states},
                    "protected": asdict(prot),
                    "thoughts": [asdict(t) for t in thoughts],
                    "scenarios": scenarios,
                    "probe": asdict(probe_result) if probe_result else None
                }
                self._emit(payload)
                time.sleep(1.0/self.rate_hz)
            except Exception as e:
                self.logger.submit("error", {"err": str(e)})
                time.sleep(1)

# =========================
# Text-only GUI
# =========================
import tkinter as tk
from tkinter import ttk

class ScientificGUI:
    def __init__(self, engine: AutonomousLightDark):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Predictive Skeleton Monitor")
        self.root.geometry("900x500")
        self.payload_q = Queue()
        engine.register_listener(lambda p: self._enqueue_payload(p))

        # Labels for different info
        self.status = tk.StringVar(value="Initializing…")
        ttk.Label(self.root, textvariable=self.status).pack(anchor="w", padx=6, pady=6)

        self.summary = tk.StringVar(value="Summary: —")
        ttk.Label(self.root, textvariable=self.summary).pack(anchor="w", padx=6, pady=6)

        self.probe_info = tk.StringVar(value="Probe: —")
        ttk.Label(self.root, textvariable=self.probe_info).pack(anchor="w", padx=6, pady=6)

        self.scenario_info = tk.StringVar(value="Scenarios: —")
        ttk.Label(self.root, textvariable=self.scenario_info).pack(anchor="w", padx=6, pady=6)

        self.metrics_info = tk.StringVar(value="Metrics: —")
        ttk.Label(self.root, textvariable=self.metrics_info).pack(anchor="w", padx=6, pady=6)

        self.thoughts_info = tk.StringVar(value="Thoughts: —")
        ttk.Label(self.root, textvariable=self.thoughts_info).pack(anchor="w", padx=6, pady=6)

        self.root.after(1000, self.refresh)

    def _enqueue_payload(self, p):
        while not self.payload_q.empty():
            self.payload_q.get_nowait()
        self.payload_q.put(p)

    def refresh(self):
        try:
            payload = self.payload_q.get_nowait()
        except Empty:
            payload = None
        if payload:
            self.status.set(f"Loop ts={payload['ts']:.1f}")
            self.summary.set(f"States: {payload['states']} | Protected: {payload['protected']}")
            probe = payload.get("probe")
            if probe:
                path = "→".join(probe.get("path", []))
                self.probe_info.set(f"Probe path: {path} | ETA: {probe.get('eta_s',0):.1f}s | Conf: {probe.get('confidence',0):.2f}")
            else:
                self.probe_info.set("Probe: none")
            scenarios = payload.get("scenarios", [])
            scen_text = "; ".join([f"{s['name']} risk={s['risk']:.2f}" for s in scenarios])
            self.scenario_info.set(f"Scenarios: {scen_text}")
            metrics = payload.get("metrics", {})
            self.metrics_info.set(f"Metrics: {metrics}")
            thoughts = payload.get("thoughts", [])
            thought_text = "; ".join([f"{t['label']}→{t['conclusion']}({t['confidence']:.2f})" for t in thoughts])
            self.thoughts_info.set(f"Thoughts: {thought_text}")
        self.root.after(1000, self.refresh)

    def run(self):
        self.root.mainloop()

# =========================
# Main
# =========================
def main():
    channels = ["cpu","mem","disk","net","proc","load","intent"]
    engine = AutonomousLightDark(channels)
    gui = ScientificGUI(engine)
    gui.run()

if __name__ == "__main__":
    main()

