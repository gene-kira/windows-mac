# ghost_mesh_unified_predictive_hybrid_live_threaded_with_discovery.py
# Single-file, cross-platform Python 3.9+ script that runs:
# - Consent-based multi-LLM orchestrator
# - Federated mesh with evolving, adaptive machine language
# - Mind hive (ontology + shards + quorum)
# - Predictive routing (bandit + EMAs + anomaly detection), confidence curves, priority scheduler
# - Hybrid providers: live (OpenAI, HF, Ollama, vLLM) and synthetic fallback (blend both)
# - Topic caches and probing planner (micro-evals when confidence stalls)
# - Outcome scoring and drift detection; backoff-aware provider calls with timeouts
# - Local JSON persistence for metrics/traces/lexicon/cache
# - Threaded worker pool for heavy tasks; Tkinter GUI runs responsive with throttled chart updates (2s)
# - Continuous Discovery Engine: symbolic candidates (SymPy), curiosity/what-if/rule-flexibility, interestingness scoring
# - Transparent Thoughts: surfaced in GUI via Discovery tab and telemetry badges
#
# Global high-priority goal: discover anti-gravity (physics-informed, safe, lawful research).
#
# ENV configuration (set as needed):
#   OPENAI_API_KEY
#   HF_API_TOKEN
#   OLLAMA_BASE_URL    (default: http://localhost:11434/v1)
#   VLLM_BASE_URL      (default: http://localhost:8000/v1)
#   OPENAI_MODEL       (e.g., gpt-4o-mini)
#   HF_MODEL           (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
#   OLLAMA_MODEL       (e.g., llama3)
#   VLLM_MODEL         (e.g., mixtral)
#
# Dependencies auto-installed: psutil, openai, requests, matplotlib, sympy

import sys
import os
import subprocess
import importlib
import time
import json
import uuid
import threading
import random
import string
import math
import shutil
import datetime
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

# =========================
# Global goal configuration & paths
# =========================
GLOBAL_GOAL = "Discover anti-gravity mechanisms (physics-informed, safe, lawful research)"
GOAL_PRIORITY = 1.0
DATA_DIR = os.path.join(os.path.expanduser("~"), ".ghost_mesh")
os.makedirs(DATA_DIR, exist_ok=True)
METRICS_FILE = os.path.join(DATA_DIR, "metrics.json")
TRACES_FILE = os.path.join(DATA_DIR, "traces.jsonl")
LEXICON_FILE = os.path.join(DATA_DIR, "lexicon.json")
CACHE_FILE = os.path.join(DATA_DIR, "topic_cache.json")
DISCOVERY_DIR = os.path.join(DATA_DIR, "discovery")
os.makedirs(DISCOVERY_DIR, exist_ok=True)

# =========================
# Persistence & utility
# =========================
def load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass

def append_jsonl(path: str, obj: Any):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def rotate_file(path: str):
    try:
        if os.path.exists(path):
            ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            new_path = f"{path}.{ts}.bak"
            shutil.move(path, new_path)
    except Exception:
        pass

# =========================
# Event bus
# =========================
@dataclass
class Event:
    topic: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=lambda: time.time())

class EventBus:
    def __init__(self):
        self.subs: Dict[str, List[Callable[[Event], None]]] = {}
        self.queue: List[Event] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def publish(self, topic: str, payload: Dict[str, Any]):
        with self._lock:
            self.queue.append(Event(topic=topic, payload=payload))

    def subscribe(self, topic: str, fn: Callable[[Event], None]):
        self.subs.setdefault(topic, []).append(fn)

    def _run(self):
        while not self._stop.is_set():
            ev = None
            with self._lock:
                if self.queue:
                    ev = self.queue.pop(0)
            if ev:
                for fn in self.subs.get(ev.topic, []):
                    try:
                        fn(ev)
                    except Exception as e:
                        sys.stderr.write(f"[event_bus] subscriber error: {e}\n")
            else:
                time.sleep(0.02)

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=1.0)

# =========================
# Autoloader
# =========================
@dataclass
class DepSpec:
    name: str
    import_name: Optional[str] = None
    version: Optional[str] = None
    required: bool = True

class AutoLoader:
    def __init__(self, deps: List[DepSpec], bus: Optional[EventBus] = None):
        self.deps = deps
        self.loaded: Dict[str, Any] = {}
        self.bus = bus

    def _log(self, level: str, msg: str, **kw):
        line = {"ts": time.time(), "level": level, "msg": msg, **kw}
        if self.bus:
            self.bus.publish("log", line)
        else:
            print(json.dumps(line))

    def ensure(self):
        for dep in self.deps:
            name = dep.name
            import_name = dep.import_name or name
            try:
                mod = importlib.import_module(import_name)
                self.loaded[name] = mod
                self._log("info", "imported", dep=name, import_name=import_name)
            except ImportError:
                if not dep.required:
                    self._log("warn", "optional_dep_missing", dep=name)
                    continue
                self._log("warn", "missing_dep_installing", dep=name, version=dep.version)
                spec = name if dep.version is None else f"{name}{dep.version}"
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", spec], check=False, capture_output=True, text=True)
                    mod = importlib.import_module(import_name)
                    self.loaded[name] = mod
                    self._log("info", "installed_and_imported", dep=name)
                except Exception as e:
                    self._log("error", "pip_or_import_failed", dep=name, err=str(e))
                    if dep.required:
                        raise RuntimeError(f"Failed to install required dependency: {name}")

    def get(self, name: str):
        return self.loaded.get(name)

# =========================
# System awareness
# =========================
@dataclass
class EndpointState:
    name: str
    url: str
    consent: bool = True
    healthy: bool = True
    fail_count: int = 0
    latency_ms: Optional[float] = None
    breaker_open: bool = False
    last_error: Optional[str] = None
    rate_capacity: int = 5
    rate_tokens: float = 5.0
    rate_refill_per_sec: float = 1.0
    last_check_ts: float = field(default_factory=lambda: time.time())

class SystemAwareness:
    def __init__(self, bus: EventBus, loader: AutoLoader):
        self.bus = bus
        self.loader = loader
        self.endpoints: Dict[str, EndpointState] = {}
        self.metrics: Dict[str, Any] = load_json(METRICS_FILE, {"cpu": None, "mem": None, "disk": None, "errors_1m": 0, "queue_depth": 0})
        self.errors_window: List[float] = []
        self.topic_counters: Dict[str, Dict[str, int]] = {}
        self.queue_depth: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._monitor, daemon=True)
        self._thr.start()
        self.psutil = loader.get("psutil")

    def _log(self, level: str, msg: str, **kw):
        self.bus.publish("log", {"level": level, "msg": msg, **kw})

    def add_endpoint(self, name: str, url: str, consent: bool = True, rate_capacity: int = 5, refill: float = 1.0):
        self.endpoints[name] = EndpointState(name=name, url=url, consent=consent, rate_capacity=rate_capacity,
                                             rate_tokens=rate_capacity, rate_refill_per_sec=refill)
        self._log("info", "endpoint_added", name=name, url=url, consent=consent)

    def trip_breaker(self, name: str, error: str):
        ep = self.endpoints.get(name)
        if not ep:
            return
        ep.breaker_open = True
        ep.fail_count += 1
        ep.last_error = error
        self._log("warn", "breaker_opened", endpoint=name, error=error)

    def reset_breaker(self, name: str):
        ep = self.endpoints.get(name)
        if not ep:
            return
        ep.breaker_open = False
        ep.fail_count = 0
        ep.last_error = None
        self._log("info", "breaker_closed", endpoint=name)

    def inc_topic(self, topic: str, sla_breach: bool = False):
        with self._lock:
            c = self.topic_counters.setdefault(topic, {"runs": 0, "sla_breaches": 0})
            c["runs"] += 1
            if sla_breach:
                c["sla_breaches"] += 1
            self.bus.publish("metrics_topic", {"topic": topic, **c})

    def _monitor(self):
        shutil_mod = importlib.import_module("shutil")
        while not self._stop.is_set():
            try:
                cpu = self.psutil.cpu_percent(interval=0.2) if self.psutil else None
                mem = self.psutil.virtual_memory().percent if self.psutil else None
                total, used, _ = shutil_mod.disk_usage(os.getcwd())
                disk = round((used / total) * 100, 2) if total else None
                with self._lock:
                    self.metrics.update({"cpu": cpu, "mem": mem, "disk": disk, "queue_depth": self.queue_depth})
                self.bus.publish("metrics", {"cpu": cpu, "mem": mem, "disk": disk, "queue_depth": self.queue_depth})
                save_json(METRICS_FILE, self.metrics)

                now = time.time()
                for ep in self.endpoints.values():
                    dt = max(0.0, now - ep.last_check_ts)
                    ep.rate_tokens = min(ep.rate_capacity, ep.rate_tokens + ep.rate_refill_per_sec * dt)
                    ep.last_check_ts = now

                cutoff = now - 60.0
                self.errors_window = [t for t in self.errors_window if t >= cutoff]
                with self._lock:
                    self.metrics["errors_1m"] = len(self.errors_window)
                time.sleep(0.8)
            except Exception as e:
                self._log("error", "monitor_error", err=str(e))
                time.sleep(1.0)

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=1.0)

    def guarded_call(self, endpoint_name: str, latency_budget_ms: int = 3500, topic: str = "general"):
        class _Guard:
            def __init__(self, outer: "SystemAwareness", endpoint_name: str, latency_budget_ms: int, topic: str):
                self.outer = outer
                self.endpoint_name = endpoint_name
                self.latency_budget_ms = latency_budget_ms
                self.topic = topic
                self.t0: float = 0.0
                self.sla_breach: bool = False

            def __enter__(self):
                ep = self.outer.endpoints.get(self.endpoint_name)
                if not ep or not ep.consent:
                    raise RuntimeError(f"Endpoint not consented or unknown: {self.endpoint_name}")
                if ep.breaker_open:
                    raise RuntimeError(f"Circuit open for endpoint: {self.endpoint_name}")
                if ep.rate_tokens < 1.0:
                    raise RuntimeError(f"Rate limited: {self.endpoint_name}")
                ep.rate_tokens -= 1.0
                self.outer.queue_depth = max(0, self.outer.queue_depth - 1)
                self.t0 = time.time()
                return self

            def __exit__(self, exc_type, exc, tb):
                ep = self.outer.endpoints.get(self.endpoint_name)
                dt_ms = (time.time() - self.t0) * 1000.0
                ep.latency_ms = dt_ms
                if exc:
                    self.outer.errors_window.append(time.time())
                    self.outer.trip_breaker(self.endpoint_name, str(exc))
                if dt_ms > self.latency_budget_ms:
                    self.sla_breach = True
                    self.outer._log("warn", "latency_budget_exceeded", endpoint=self.endpoint_name, latency_ms=round(dt_ms, 1), topic=self.topic)
                self.outer.inc_topic(self.topic, sla_breach=self.sla_breach)
                return False

        return _Guard(self, endpoint_name, latency_budget_ms, topic)

# =========================
# Predictive store & anomaly detection
# =========================
@dataclass
class RollingStats:
    latency_ema_ms: float = 1000.0
    error_rate: float = 0.0
    success_score: float = 0.5
    cost_hint: float = 0.5
    last_update: float = field(default_factory=lambda: time.time())
    runs: int = 0

class PredictiveStore:
    def __init__(self):
        self.models: Dict[str, RollingStats] = {}
        self.topics: Dict[str, RollingStats] = {}
        self.baselines: Dict[str, float] = {}  # topic -> baseline agreement

    def get_model(self, name: str, cost_hint: float) -> RollingStats:
        rs = self.models.get(name)
        if not rs:
            rs = RollingStats(cost_hint=cost_hint)
            self.models[name] = rs
        return rs

    def update_latency(self, name: str, observed_ms: float, cost_hint: float, alpha: float = 0.3):
        rs = self.get_model(name, cost_hint)
        rs.latency_ema_ms = alpha * observed_ms + (1 - alpha) * rs.latency_ema_ms
        rs.runs += 1
        rs.last_update = time.time()

    def update_error(self, name: str, error: bool, cost_hint: float, beta: float = 0.2):
        rs = self.get_model(name, cost_hint)
        target = 1.0 if error else 0.0
        rs.error_rate = beta * target + (1 - beta) * rs.error_rate
        rs.last_update = time.time()

    def update_success(self, name: str, success: float, cost_hint: float, gamma: float = 0.15):
        rs = self.get_model(name, cost_hint)
        rs.success_score = gamma * success + (1 - gamma) * rs.success_score
        rs.last_update = time.time()

    def score(self, name: str) -> float:
        rs = self.models.get(name)
        if not rs:
            return 0.0
        latency_term = 1.0 / (1.0 + max(1.0, rs.latency_ema_ms))
        return 1.8 * latency_term + 1.6 * (1.0 - rs.error_rate) + 1.4 * rs.success_score - 1.0 * rs.cost_hint

    def set_baseline(self, topic: str, agreement: float, alpha: float = 0.2):
        base = self.baselines.get(topic, agreement)
        self.baselines[topic] = alpha * agreement + (1 - alpha) * base

    def anomaly(self, topic: str, agreement: float, tol: float = 0.15) -> bool:
        base = self.baselines.get(topic, 0.5)
        return (base - agreement) > tol

# =========================
# Backoff registry (provider failure control)
# =========================
class BackoffRegistry:
    def __init__(self):
        self.state: Dict[str, float] = {}  # model_name -> next_allowed_ts

    def set_backoff(self, model: str, seconds: float):
        self.state[model] = time.time() + seconds

    def allowed(self, model: str) -> bool:
        ts = self.state.get(model, 0.0)
        return time.time() >= ts

# =========================
# Transparent thoughts
# =========================
def make_thoughts(label, params, outputs, score=None, machine_ctx=None):
    thoughts = []
    thoughts.append({"label": "Expectation", "text": f"{label}: Evaluate symbolic candidate under sampled parameters."})
    if "result" in outputs and (outputs["result"] is None or not isinstance(outputs["result"], (int, float))):
        thoughts.append({"label": "Anomaly", "text": "Non-finite or non-numeric result; clamp inputs or skip."})
    if score is not None:
        try:
            thoughts.append({"label": "FocusHint", "text": f"Interestingness score={score:.4f}. Increase sampling density nearby."})
        except Exception:
            thoughts.append({"label": "FocusHint", "text": f"Interestingness score computed. Increase sampling density nearby."})
    if machine_ctx:
        thoughts.append({"label": "MachineContext", "text": f"CPU={machine_ctx.get('cpu', 'n/a')}%, RAM={machine_ctx.get('ram', 'n/a')}% influences sampling cadence."})
    thoughts.append({"label": "Proposal", "text": "Adaptive sampler will zoom into high-score regions and refine parameters."})
    return thoughts

# =========================
# SymPy imports
# =========================
# Will be loaded via AutoLoader; we create aliases after loader.ensure()
sympy = None
from math import isfinite  # for eval safety

# =========================
# Curiosity, What-If, and Rule Flexibility operators
# =========================
def curious_perturb(value, curiosity_level):
    if curiosity_level <= 0:
        return value
    magnitude = curiosity_level
    jitter = random.uniform(-magnitude, magnitude)
    return value * (1.0 + jitter) + jitter

def what_if_transform_expr(expr, curiosity_level):
    if curiosity_level <= 0:
        return expr
    try:
        p = curiosity_level
        changed = expr
        if random.random() < p * 0.5:
            changed = sympy.simplify(sympy.sqrt(sympy.Abs(changed)) + sympy.sin(changed))
        if random.random() < p * 0.3:
            changed = sympy.simplify(changed + sympy.log(sympy.Abs(changed) + 1.0))
        return changed
    except Exception:
        return expr

def bend_rules(expr, rule_flexibility):
    if rule_flexibility <= 0:
        return expr
    try:
        changed = expr
        rf = rule_flexibility
        if random.random() < rf:
            changed = sympy.simplify(changed + (rf * 0.1) - (rf * 0.05))
        if random.random() < rf * 0.5:
            changed = sympy.simplify(changed.subs(2, random.choice([3, 5, sympy.pi])))
        return changed
    except Exception:
        return expr

def what_if_rule_constants(params, curiosity_level, what_if):
    if not what_if:
        return params
    p = curiosity_level
    try:
        params['G'] = max(1e-20, params['G'] * random.uniform(max(0.1, 1.0 - p), 1.0 + 9.0 * p))
        params['c'] = max(1e5, params['c'] * random.uniform(max(0.5, 1.0 - 0.5 * p), 1.0 + 0.5 * p))
        params['r'] = max(1e-6, params['r'] * random.uniform(max(0.1, 1.0 - p), 1.0 + p))
        return params
    except Exception:
        return params

# =========================
# Formula library
# =========================
class FormulaLibrary:
    def __init__(self):
        self.G, self.m1, self.m2, self.r, self.c, self.M, self.field_strength, self.mass, self.alpha, self.beta, self.coupling = \
            sympy.symbols('G m1 m2 r c M field_strength mass alpha beta coupling')
        F, E, Phi, Psi = sympy.symbols('F E Phi Psi')
        # Base formulas
        self.formulas = {
            "NewtonianGravity": sympy.Eq(F, self.G * self.m1 * self.m2 / (self.r**2)),
            "RelativisticEnergy": sympy.Eq(E, self.mass * self.c**2),
            "FieldCoupling": sympy.Eq(Phi, self.field_strength * self.mass * self.coupling),
            "HypotheticalLift": sympy.Eq(Psi, (self.alpha * self.field_strength) / sympy.Max(self.r, 1e-6) - self.beta * self.mass),
        }

    def list(self):
        return self.formulas

# =========================
# Candidate generator (symbolic combinations)
# =========================
class CandidateGenerator:
    def __init__(self, library: FormulaLibrary, curiosity_level=0.0, rule_flexibility=0.0):
        self.lib = library
        self.curiosity_level = curiosity_level
        self.rule_flexibility = rule_flexibility

    def mutate(self, expr):
        e = what_if_transform_expr(expr, self.curiosity_level)
        e = bend_rules(e, self.rule_flexibility)
        return e

    def generate(self):
        base = list(self.lib.list().items())
        candidates = []

        # Pairwise additive combinations
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                n1, eq1 = base[i]
                n2, eq2 = base[j]
                lhs1 = eq1.rhs
                lhs2 = eq2.rhs
                try:
                    summed = sympy.simplify(lhs1 + lhs2)
                    summed = self.mutate(summed)
                    candidates.append((f"{n1} + {n2}", summed))
                except Exception:
                    pass

        # Multiplicative combinations
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                n1, eq1 = base[i]
                n2, eq2 = base[j]
                lhs1 = eq1.rhs
                lhs2 = eq2.rhs
                try:
                    product = sympy.simplify(lhs1 * lhs2)
                    product = self.mutate(product)
                    candidates.append((f"{n1} * {n2}", product))
                except Exception:
                    pass

        # Transformations on single formulas (sqrt/log/sin where valid)
        for n, eq in base:
            lhs = eq.rhs
            try:
                candidates.append((f"sqrt({n})", sympy.simplify(self.mutate(sympy.sqrt(lhs)))))
            except Exception:
                pass
            try:
                candidates.append((f"log({n})", sympy.simplify(self.mutate(sympy.log(lhs)))))
            except Exception:
                pass
            try:
                candidates.append((f"sin({n})", sympy.simplify(self.mutate(sympy.sin(lhs)))))
            except Exception:
                pass

        # Originals with mutation
        for n, eq in base:
            candidates.append((n, self.mutate(eq.rhs)))

        return candidates

# =========================
# Parameter sampler and evaluator
# =========================
class ParameterSampler:
    def __init__(self, curiosity_level=0.0, what_if=False):
        self.curiosity_level = curiosity_level
        self.what_if = what_if

    def sample(self, focus=None):
        def rng(a, b):
            if focus is None:
                return random.uniform(a, b)
            return random.uniform(*focus.get((a, b), (a, b)))

        params = {
            'G': 6.67430e-11,
            'm1': rng(1e2, 1e6),
            'm2': rng(1e2, 1e6),
            'r': max(rng(1.0, 1e6), 1e-6),
            'c': 299_792_458.0,
            'M': rng(1e20, 1e30),
            'field_strength': rng(0.1, 100.0),
            'mass': rng(1.0, 1e6),
            'alpha': rng(0.5, 2.0),
            'beta': rng(0.5, 2.0),
            'coupling': rng(1e-9, 1e-3),
        }

        for k in list(params.keys()):
            if isinstance(params[k], (int, float)):
                params[k] = curious_perturb(params[k], self.curiosity_level)

        params = what_if_rule_constants(params, self.curiosity_level, self.what_if)
        return params

class Evaluator:
    def __init__(self, library: FormulaLibrary):
        self.lib = library

    def eval_numeric(self, expr, params):
        subs = {
            self.lib.G: params['G'],
            self.lib.m1: params['m1'],
            self.lib.m2: params['m2'],
            self.lib.r: params['r'],
            self.lib.c: params['c'],
            self.lib.M: params['M'],
            self.lib.field_strength: params['field_strength'],
            self.lib.mass: params['mass'],
            self.lib.alpha: params['alpha'],
            self.lib.beta: params['beta'],
            self.lib.coupling: params['coupling'],
        }
        try:
            val = expr.subs(subs).evalf()
            val_float = float(val)
            if math.isfinite(val_float):
                return val_float
            return None
        except Exception:
            return None

# =========================
# Interestingness scoring
# =========================
def interestingness(result, params):
    if result is None:
        return 0.0
    magnitude = abs(result)
    score = math.exp(-abs(math.log10(magnitude + 1e-12)))  # peak near ~1
    r = params['r']
    penalty = math.exp(-abs(math.log10(r + 1e-6)))
    return max(0.0, min(1.0, 0.5 * score + 0.5 * penalty))

# =========================
# Machine telemetry helpers for Discovery
# =========================
def read_machine_ctx(psutil_mod):
    if psutil_mod is None:
        return {"cpu": None, "ram": None}
    try:
        cpu = psutil_mod.cpu_percent(interval=None)
        ram = psutil_mod.virtual_memory().percent
        return {"cpu": cpu, "ram": ram}
    except Exception:
        return {"cpu": None, "ram": None}

def adapt_cadence(base_sleep, machine_ctx):
    cpu = machine_ctx.get("cpu")
    ram = machine_ctx.get("ram")
    sleep = base_sleep
    if cpu is not None and ram is not None:
        if cpu > 80 or ram > 85:
            sleep *= 1.5
        elif cpu < 30 and ram < 50:
            sleep *= 0.8
    return max(0.05, min(5.0, sleep))

# =========================
# Providers: live + synthetic hybrid adapters (with timeouts & backoff)
# =========================
class ProviderAdapters:
    def __init__(self, loader: AutoLoader, backoff: BackoffRegistry, bus: EventBus):
        self.loader = loader
        self.backoff = backoff
        self.bus = bus
        self.openai = loader.get("openai")
        self.requests = loader.get("requests")

    def _synthetic(self, model_name: str, topic: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
        seed = f"{model_name}-{topic}-{len(messages)}"
        random.seed(seed)
        vocab = ["quantum", "field", "tensor", "gravity", "symmetry", "warp", "EM", "spin", "lift", "constraint", "hypothesis"]
        content = " ".join(random.sample(vocab, 4))
        return {"output": f"[{model_name}] {topic}: {content}", "usage": {"prompt_tokens": None, "completion_tokens": None}}

    def _check_backoff(self, model: str):
        if not self.backoff.allowed(model):
            raise TimeoutError(f"Backoff window active for {model}")

    def call_openai(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, topic: str, hybrid: bool) -> Dict[str, Any]:
        try:
            self._check_backoff(model)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or self.openai is None:
                return self._synthetic(model, topic, messages, temperature, max_tokens)
            client = self.openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            txt = (resp.choices[0].message.content or "").strip()
            usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", None), "completion_tokens": getattr(resp.usage, "completion_tokens", None)}
            if hybrid and not txt:
                syn = self._synthetic(model, topic, messages, temperature, max_tokens)["output"]
                txt = syn
            return {"output": txt, "usage": usage}
        except Exception as e:
            self.bus.publish("alert", {"severity": "warn", "msg": "openai_call_failed", "model": model, "err": str(e)})
            self.backoff.set_backoff(model, 5.0)
            return self._synthetic(model, topic, messages, temperature, max_tokens)

    def call_hf(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, topic: str, hybrid: bool) -> Dict[str, Any]:
        try:
            self._check_backoff(model)
            token = os.getenv("HF_API_TOKEN")
            if not token or self.requests is None:
                return self._synthetic(model, topic, messages, temperature, max_tokens)
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
            url = f"https://api-inference.huggingface.co/models/{model}"
            r = self.requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                txt = data[0]["generated_text"].split("assistant:", 1)[-1].strip()
            elif isinstance(data, dict) and "generated_text" in data:
                txt = data["generated_text"].strip()
            else:
                txt = str(data).strip()
            if hybrid and not txt:
                txt = self._synthetic(model, topic, messages, temperature, max_tokens)["output"]
            return {"output": txt, "usage": {"prompt_tokens": None, "completion_tokens": None}}
        except Exception as e:
            self.bus.publish("alert", {"severity": "warn", "msg": "hf_call_failed", "model": model, "err": str(e)})
            self.backoff.set_backoff(model, 5.0)
            return self._synthetic(model, topic, messages, temperature, max_tokens)

    def call_openai_compat(self, base_url: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, topic: str, hybrid: bool) -> Dict[str, Any]:
        try:
            self._check_backoff(model)
            if self.openai is None:
                return self._synthetic(model, topic, messages, temperature, max_tokens)
            client = self.openai.OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "unused"))
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            txt = (resp.choices[0].message.content or "").strip()
            usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", None), "completion_tokens": getattr(resp.usage, "completion_tokens", None)}
            if hybrid and not txt:
                txt = self._synthetic(model, topic, messages, temperature, max_tokens)["output"]
            return {"output": txt, "usage": usage}
        except Exception as e:
            self.bus.publish("alert", {"severity": "warn", "msg": "compat_call_failed", "model": model, "err": str(e)})
            self.backoff.set_backoff(model, 5.0)
            return self._synthetic(model, topic, messages, temperature, max_tokens)

# =========================
# Orchestrator core
# =========================
@dataclass
class ModelInfo:
    name: str
    provider: str   # "openai" | "hf" | "ollama" | "vllm"
    endpoint: str
    capabilities: List[str]
    cost_hint: float
    latency_hint_ms: int
    max_tokens: int
    consent: bool = True
    model_id: Optional[str] = None
    hybrid: bool = True  # blend synthetic when live fails or is empty

@dataclass
class Trace:
    run_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def add(self, info: ModelInfo):
        if not info.consent:
            raise ValueError(f"Model {info.name} lacks consent.")
        self._models[info.name] = info

    def get(self, name: str) -> Optional[ModelInfo]:
        return self._models.get(name)

    def query(self, capability: str) -> List[ModelInfo]:
        return [m for m in self._models.values() if capability in m.capabilities and m.consent]

class PriorityScheduler:
    def __init__(self, bus: EventBus, sys: SystemAwareness):
        self.bus = bus
        self.sys = sys
        self.queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def submit(self, item: Dict[str, Any]):
        with self._lock:
            self.queue.append(item)
            self.queue.sort(key=lambda x: (-x.get("priority", 0.0), x.get("ts", time.time())))
            self.sys.queue_depth = len(self.queue)
            self.bus.publish("scheduler", {"queued": self.sys.queue_depth})

    def take(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.queue:
                return None
            item = self.queue.pop(0)
            self.sys.queue_depth = len(self.queue)
            self.bus.publish("scheduler", {"queued": self.sys.queue_depth})
            return item

class PredictiveRouter:
    def __init__(self, registry: ModelRegistry, store: PredictiveStore):
        self.registry = registry
        self.store = store

    def pick(self, capability: str, budget: float, latency_ms: int, priority: float = 0.0, top_k: int = 4) -> List[ModelInfo]:
        candidates = self.registry.query(capability)
        filtered = [m for m in candidates if m.latency_hint_ms <= latency_ms and m.cost_hint <= budget]
        filtered.sort(key=lambda m: m.latency_hint_ms * (0.8 if priority >= 0.9 else 1.0))
        def noisy_score(m: ModelInfo):
            base = self.store.score(m.name)
            noise = random.gauss(0, 0.05)
            return base + noise
        scored = sorted(filtered, key=noisy_score, reverse=True)
        return scored[:max(1, top_k)]

class Orchestrator:
    def __init__(self, registry: ModelRegistry, router: PredictiveRouter, sys_awareness: SystemAwareness, bus: EventBus, store: PredictiveStore, adapters: ProviderAdapters):
        self.registry = registry
        self.router = router
        self.sys = sys_awareness
        self.bus = bus
        self.store = store
        self.adapters = adapters
        self.cache: Dict[str, Any] = load_json(CACHE_FILE, {"topics": {}})

    @staticmethod
    def agreement_score(outputs: List[str]) -> float:
        if not outputs:
            return 0.0
        sets = [set(o.split()) for o in outputs]
        base = sets[0]
        overlaps = [len(base & s) / max(1, len(base | s)) for s in sets[1:]] or [1.0]
        return sum(overlaps) / len(overlaps)

    @staticmethod
    def outcome_score(text: str) -> float:
        keywords = ["gravity", "tensor", "field", "safety", "experiment", "prediction", "constraint", "hypothesis", "electromagnetic", "levitation"]
        count = sum(1 for k in keywords if k in text.lower())
        length_bonus = min(1.0, len(text) / 1500.0)
        return min(1.0, 0.6 * (count / len(keywords)) + 0.4 * length_bonus)

    def provider_call(self, m: ModelInfo, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload["messages"]
        temperature = payload.get("temperature", 0.2)
        max_tokens = payload.get("max_tokens", min(1024, m.max_tokens))
        topic = payload.get("topic", "general")
        model_id = m.model_id or m.name
        if m.provider == "openai":
            return self.adapters.call_openai(model_id, messages, temperature, max_tokens, topic, m.hybrid)
        if m.provider == "hf":
            return self.adapters.call_hf(model_id, messages, temperature, max_tokens, topic, m.hybrid)
        if m.provider == "ollama":
            base_url = m.endpoint or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return self.adapters.call_openai_compat(base_url, model_id, messages, temperature, max_tokens, topic, m.hybrid)
        if m.provider == "vllm":
            base_url = m.endpoint or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            return self.adapters.call_openai_compat(base_url, model_id, messages, temperature, max_tokens, topic, m.hybrid)
        return self.adapters._synthetic(model_id, topic, messages, temperature, max_tokens)

    def get_topic_cache(self, topic: str) -> Dict[str, Any]:
        return self.cache["topics"].get(topic, {"best_outcome": 0.0, "best_text": "", "runs": 0})

    def update_topic_cache(self, topic: str, text: str, outcome: float):
        c = self.get_topic_cache(topic)
        c["runs"] += 1
        if outcome > c.get("best_outcome", 0.0):
            c["best_outcome"] = outcome
            c["best_text"] = text
        self.cache["topics"][topic] = c
        save_json(CACHE_FILE, self.cache)

    def probing_messages(self, topic: str) -> List[Dict[str, str]]:
        probes = [
            "List testable anti-gravity hypotheses with constraints and failure modes.",
            "Propose EM field configurations for levitation; estimate field strengths and safety.",
            "Check tensor symmetry assumptions; identify contradictions."
        ]
        return [{"role": "user", "content": random.choice(probes)}]

    def run(self, capability: str, messages: List[Dict[str, str]], mode: str = "ensemble",
            budget: float = 0.9, latency_ms: int = 3500, topic: str = "general", priority: float = 0.0,
            confidence_target: float = 0.68, max_members: int = 4) -> Trace:

        trace = Trace(run_id=str(uuid.uuid4()))
        models = self.router.pick(capability, budget, latency_ms, priority=priority, top_k=max_members)
        if not models:
            raise RuntimeError("No models available under current policy.")
        for m in models:
            if m.name not in self.sys.endpoints:
                self.sys.add_endpoint(m.name, m.endpoint, consent=m.consent, rate_capacity=5, refill=1.0)

        outputs: List[str] = []
        members_used: List[str] = []
        conf_trend: List[float] = []
        pred_lat: List[float] = []
        act_lat: List[float] = []

        # Warm-start from topic cache
        cache = self.get_topic_cache(topic)
        if cache.get("best_text"):
            outputs.append(cache["best_text"])
            conf_trend.append(0.5)
            members_used.append("cache-primer")
            self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "cache_primer", "topic": topic})

        # Main ensemble
        for m in models:
            try:
                with self.sys.guarded_call(m.name, latency_budget_ms=latency_ms, topic=topic):
                    t_pred = self.store.get_model(m.name, m.cost_hint).latency_ema_ms
                    pred_lat.append(t_pred)
                    out = self.provider_call(m, {"messages": messages, "temperature": 0.2,
                                                 "max_tokens": min(1024, m.max_tokens),
                                                 "topic": topic})
                outputs.append(out["output"])
                members_used.append(m.name)
                self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "ensemble_member", "model": m.name, "topic": topic, "priority": priority})
                trace.steps.append({"type": "ensemble_member", "model": m.name, "output": out["output"], "usage": out.get("usage")})
                if self.sys.endpoints[m.name].latency_ms is not None:
                    l = self.sys.endpoints[m.name].latency_ms
                    act_lat.append(l)
                    self.store.update_latency(m.name, l, m.cost_hint)
                self.store.update_error(m.name, False, m.cost_hint)

                score = self.agreement_score(outputs)
                conf_trend.append(score)
                slope = conf_trend[-1] - (conf_trend[-2] if len(conf_trend) >= 2 else conf_trend[-1])
                anomaly = self.store.anomaly(topic, score)
                self.bus.publish("ensemble_confidence", {"run_id": trace.run_id, "agreement": round(score, 3), "members": len(outputs), "slope": round(slope, 3), "anomaly": anomaly})

                # If anomaly, add a targeted probe before stopping
                if anomaly and len(members_used) < max_members:
                    probe_out = self.provider_call(m, {"messages": self.probing_messages(topic), "temperature": 0.2,
                                                       "max_tokens": min(512, m.max_tokens), "topic": topic})
                    outputs.append(probe_out["output"])
                    members_used.append(f"{m.name}-probe")
                    trace.steps.append({"type": "probe", "model": m.name, "output": probe_out["output"]})
                    score = self.agreement_score(outputs)
                    conf_trend.append(score)
                    self.bus.publish("ensemble_confidence", {"run_id": trace.run_id, "agreement": round(score, 3), "members": len(outputs), "slope": 0.0, "anomaly": True})

                if score >= confidence_target and slope >= -0.02:
                    break
            except Exception as e:
                trace.steps.append({"type": "ensemble_error", "model": m.name, "error": str(e)})
                self.bus.publish("alert", {"severity": "warn", "msg": "ensemble_member_failed", "model": m.name, "err": str(e)})
                self.store.update_error(m.name, True, m.cost_hint)

        merged = "\n".join(outputs) if outputs else "[no outputs]"
        outcome = self.outcome_score(merged)
        trace.steps.append({"type": "merge", "strategy": "concat", "output": merged, "topic": topic, "members_used": members_used, "outcome_score": outcome})
        self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "merge", "members": members_used, "topic": topic, "priority": priority, "outcome": round(outcome, 3)})

        final_agreement = self.agreement_score(outputs)
        self.store.set_baseline(topic, final_agreement)
        for m in [mm for mm in members_used if mm in self.registry._models]:
            self.store.update_success(m, 0.5 * final_agreement + 0.5 * outcome, self.registry.get(m).cost_hint)

        self.update_topic_cache(topic, merged, outcome)

        append_jsonl(TRACES_FILE, {"run_id": trace.run_id, "topic": topic, "priority": priority, "steps": trace.steps,
                                   "conf_trend": conf_trend, "pred_latency": pred_lat, "act_latency": act_lat})
        return trace

# =========================
# Mesh + machine language
# =========================
@dataclass
class MeshModelInfo:
    name: str
    capabilities: List[str]
    cost_hint: float
    latency_ms: int

@dataclass
class Peer:
    node_id: str
    healthy: bool = True
    rate_tokens: float = 5.0
    rate_capacity: float = 5.0
    refill_per_sec: float = 1.0
    last_seen: float = field(default_factory=lambda: time.time())

class MachineLanguage:
    ALPHABET = string.ascii_letters + string.digits + "+/="

    def __init__(self, seed_meanings: Optional[List[str]] = None):
        snap = load_json(LEXICON_FILE, {})
        self.lexicon: Dict[str, str] = snap.get("lexicon", {})
        self.usage: Dict[str, int] = snap.get("usage", {})
        self.seed_meanings = seed_meanings or ["ACK", "NACK", "INTENT", "STATE", "RESULT", "QUERY", "MERGE", "FORK", "CONSENSUS", "GOAL", "ANTIGRAVITY"]
        if not self.lexicon:
            for m in self.seed_meanings:
                tok = self._new_token()
                self.lexicon[tok] = m
                self.usage.setdefault(tok, 0)
        self.mutation_rate = 0.1

    def _new_token(self, length: Optional[int] = None) -> str:
        L = length or random.randint(8, 12)
        return "".join(random.choice(self.ALPHABET) for _ in range(L))

    def evolve(self):
        keys = list(self.lexicon.keys())
        if not keys:
            return
        random.shuffle(keys)
        sorted_keys = sorted(keys, key=lambda k: self.usage.get(k, 0))
        k_mut = max(1, int(len(keys) * self.mutation_rate))
        for k in sorted_keys[:k_mut]:
            s = list(k)
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, len(s) - 1)
                s[i] = random.choice(self.ALPHABET)
            new_k = "".join(s)
            if new_k not in self.lexicon:
                self.lexicon[new_k] = self.lexicon[k]
                self.usage[new_k] = self.usage.get(k, 0)
                del self.lexicon[k]
                self.usage.pop(k, None)
        if random.random() < 0.4:
            concept = random.choice(["HORIZON", "LATENCY", "COST", "CONFIDENCE", "PRISM", "VECTOR", "DELTA", "ANTIGRAVITY"])
            nt = self._new_token()
            self.lexicon[nt] = concept
            self.usage[nt] = 0
        save_json(LEXICON_FILE, {"usage": self.usage, "lexicon": self.lexicon})

    def encode_message(self, words: List[str]) -> str:
        inv = {v: k for k, v in self.lexicon.items()}
        out = []
        for w in words:
            if w in inv:
                tok = inv[w]
                self.usage[tok] = self.usage.get(tok, 0) + 1
                out.append(tok)
            else:
                t = self._new_token()
                self.lexicon[t] = w
                self.usage[t] = 1
                out.append(t)
        save_json(LEXICON_FILE, {"usage": self.usage, "lexicon": self.lexicon})
        return " ".join(out)

    def decode_message(self, msg: str) -> str:
        parts = msg.split()
        decoded = []
        for p in parts:
            decoded.append(self.lexicon.get(p, "?"))
        return " ".join(decoded)

class MeshNode:
    def __init__(self, node_id: str, bus: EventBus, ml: MachineLanguage):
        self.node_id = node_id
        self.bus = bus
        self.ml = ml
        self.peers: Dict[str, Peer] = {}
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._tick, daemon=True)
        self._thr.start()

    def add_peer(self, peer_id: str):
        self.peers[peer_id] = Peer(node_id=peer_id)
        self.bus.publish("log", {"level": "info", "msg": "peer_added", "peer": peer_id})

    def publish_intent(self, words: List[str], task: Optional[Dict[str, Any]] = None, priority: float = 0.0):
        goal_words = ["GOAL", "ANTIGRAVITY"] if priority >= 0.9 else []
        msg_words = goal_words + words
        raw = self.ml.encode_message(msg_words)
        decoded = self.ml.decode_message(raw)
        self.bus.publish("mesh", {"from": self.node_id, "raw": raw, "decoded": decoded})
        self.bus.publish("mesh_intent", {
            "from": self.node_id,
            "tokens": raw.split(),
            "decoded": msg_words,
            "task": task or {"goal": GLOBAL_GOAL, "priority": priority}
        })

    def share_state(self, key: str, value: Any):
        msg = self.ml.encode_message(["STATE", key])
        self.bus.publish("mesh", {"from": self.node_id, "raw": msg, "decoded": self.ml.decode_message(msg)})
        self.bus.publish("mesh_state", {"from": self.node_id, "state": {key: value}})

    def _tick(self):
        while not self._stop.is_set():
            time.sleep(random.uniform(0.6, 1.5))
            if random.random() < 0.6:
                self.publish_intent(["INTENT", "MERGE", "RESULT"], task={"goal": GLOBAL_GOAL}, priority=GOAL_PRIORITY)
            else:
                self.publish_intent(["QUERY", "CONSENSUS"], task={"goal": GLOBAL_GOAL}, priority=GOAL_PRIORITY)
            if random.random() < 0.3:
                self.ml.evolve()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=1.0)

# =========================
# Mind hive
# =========================
@dataclass
class LWWMap:
    store: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    def set(self, k: str, v: Any): self.store[k] = {"val": v, "ts": time.time()}
    def get(self, k: str) -> Optional[Any]: return self.store.get(k, {}).get("val")
    def merge(self, other: "LWWMap"):
        for k, rec in other.store.items():
            cur = self.store.get(k)
            if cur is None or rec["ts"] >= cur["ts"]:
                self.store[k] = rec

@dataclass
class HiveShards:
    task: LWWMap = field(default_factory=LWWMap)
    context: LWWMap = field(default_factory=LWWMap)
    telemetry: LWWMap = field(default_factory=LWWMap)

@dataclass
class Proposal:
    proposal_id: str
    topic: str
    payload: Dict[str, Any]
    acks: List[str] = field(default_factory=list)
    nacks: List[str] = field(default_factory=list)
    deadline: float = field(default_factory=lambda: time.time() + 3.0)
    committed: bool = False
    rationale: Optional[str] = None

class MindHive:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.shards = HiveShards()
        self.proposals: Dict[str, Proposal] = {}
        self._wire_topics()
        self.announce_context("global_goal", GLOBAL_GOAL)
        self.announce_context("priority", GOAL_PRIORITY)

    def _wire_topics(self):
        self.bus.subscribe("mesh_intent", self._on_intent)
        self.bus.subscribe("mesh_state", self._on_state)
        self.bus.subscribe("mesh_vote", self._on_vote)

    def _on_intent(self, ev: Event):
        decoded = ev.payload.get("decoded") or []
        node_id = ev.payload.get("from")
        task = ev.payload.get("task") or {}
        priority = float(task.get("priority", 0.0))
        self.shards.task.set(f"intent:{node_id}:{uuid.uuid4().hex[:6]}", {"decoded": decoded, "task": task, "ts": ev.ts})
        if task and "dag" not in task:
            dag = {"steps": [
                {"role": "planner", "desc": "Draft anti-gravity research plan" if priority >= 0.9 else "Draft plan"},
                {"role": "critics", "desc": "Review physics assumptions and safety"},
                {"role": "executors", "desc": "Run simulations / analyses"}
            ]}
            self.shards.task.set(f"dag:{uuid.uuid4().hex[:6]}", dag)
            self._propose("plan", {"dag": dag, "intent_from": node_id, "priority": priority, "goal": GLOBAL_GOAL}, quorum=2)

    def _on_state(self, ev: Event):
        state = ev.payload.get("state") or {}
        node_id = ev.payload.get("from")
        self.shards.telemetry.set(f"tele:{node_id}", state)

    def _on_vote(self, ev: Event):
        pid = ev.payload.get("proposal_id")
        vote = ev.payload.get("vote")
        node = ev.payload.get("from")
        prop = self.proposals.get(pid)
        if not prop:
            return
        if vote == "ack" and node not in prop.acks:
            prop.acks.append(node)
        if vote == "nack" and node not in prop.nacks:
            prop.nacks.append(node)
        self._try_commit(pid)

    def _propose(self, topic: str, payload: Dict[str, Any], quorum: int = 2) -> str:
        pid = uuid.uuid4().hex[:12]
        prop = Proposal(proposal_id=pid, topic=topic, payload=payload)
        self.proposals[pid] = prop
        self.bus.publish("hive_propose", {"proposal_id": pid, "topic": topic, "payload": payload, "deadline": prop.deadline})
        self.bus.publish("mesh_vote", {"proposal_id": pid, "vote": "ack", "from": "hive"})
        return pid

    def _try_commit(self, pid: str):
        prop = self.proposals.get(pid)
        if not prop or prop.committed:
            return
        if time.time() > prop.deadline:
            prop.rationale = "deadline expired; abort"
            prop.committed = False
            self.bus.publish("hive_decision", {"proposal_id": pid, "decision": "abort",
                                               "acks": prop.acks, "nacks": prop.nacks, "rationale": prop.rationale})
            return
        quorum_needed = 2
        if prop.payload.get("goal") == GLOBAL_GOAL:
            quorum_needed = 2
        if len(prop.acks) >= quorum_needed and len(prop.nacks) == 0:
            prop.rationale = "sufficient acks; no objections"
            prop.committed = True
            self.bus.publish("hive_decision", {"proposal_id": pid, "decision": "commit",
                                               "acks": prop.acks, "nacks": prop.nacks, "rationale": prop.rationale})

    def announce_context(self, key: str, value: Any):
        self.shards.context.set(key, value)
        self.bus.publish("mesh_state", {"from": "hive", "state": {"key": key, "value": value}})

    def emit_intent(self, words: List[str], task: Dict[str, Any]):
        self.bus.publish("mesh_intent", {"from": "hive", "tokens": [], "decoded": words, "task": task})

# =========================
# Worker pool (threaded heavy tasks)
# =========================
class WorkerPool:
    def __init__(self, max_workers: int = 4):
        import concurrent.futures
        self.concurrent = concurrent.futures
        self.executor = self.concurrent.ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=False)

# =========================
# Continuous discovery engine
# =========================
from concurrent.futures import ThreadPoolExecutor, as_completed

class DiscoveryEngine:
    def __init__(
        self,
        out_dir,
        batch_size=50,
        max_workers=4,
        checkpoint_every=1000,
        max_log_size=10_000_000,
        solved_threshold=0.95,
        stream_queue=None,
        curiosity_level=0.0,
        what_if=False,
        rule_flexibility=0.0,
        daemon=False,
        psutil_mod=None,
        bus: Optional[EventBus] = None
    ):
        self.out_dir = ensure_dir(out_dir)
        self.log_path = os.path.join(self.out_dir, "results.jsonl")
        self.ckpt_path = os.path.join(self.out_dir, "checkpoint.json")
        self.health_path = os.path.join(self.out_dir, "health.json")
        self.solved_path = os.path.join(self.out_dir, "solved.jsonl")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_every = checkpoint_every
        self.max_log_size = max_log_size
        self.solved_threshold = solved_threshold
        self.daemon = daemon

        self._stop = threading.Event()
        self._count = 0
        self._last_error = None

        self.library = FormulaLibrary()
        self.generator = CandidateGenerator(self.library, curiosity_level=curiosity_level, rule_flexibility=rule_flexibility)
        self.candidates = self.generator.generate()
        self.sampler = ParameterSampler(curiosity_level=curiosity_level, what_if=what_if)
        self.evaluator = Evaluator(self.library)

        self.focus = None
        self.stream_queue = stream_queue or queue.Queue(maxsize=5000)
        self.curiosity_level = curiosity_level
        self.what_if = what_if
        self.rule_flexibility = rule_flexibility

        self.psutil_mod = psutil_mod
        self.bus = bus

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def _write_jsonl(self, path, rec):
        if path == self.log_path and os.path.exists(path) and os.path.getsize(path) >= self.max_log_size:
            rotate_file(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _write_checkpoint(self):
        ckpt = {
            "timestamp": now_iso(),
            "total_records": self._count,
            "last_error": self._last_error,
            "curiosity_level": self.curiosity_level,
            "what_if": self.what_if,
            "rule_flexibility": self.rule_flexibility
        }
        with open(self.ckpt_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f)

    def _write_health(self, status="ok"):
        health = {
            "timestamp": now_iso(),
            "status": status,
            "total_records": self._count,
            "last_error": self._last_error,
            "curiosity_level": self.curiosity_level,
            "what_if": self.what_if,
            "rule_flexibility": self.rule_flexibility
        }
        with open(self.health_path, "w", encoding="utf-8") as f:
            json.dump(health, f)

    def _update_focus(self, records):
        top = sorted([r for r in records if r["score"] is not None], key=lambda x: x["score"], reverse=True)[:10]
        if not top:
            self.focus = None
            return
        rs = [t["parameters"]["r"] for t in top]
        median_r = sorted(rs)[len(rs)//2]
        span = max(1.0, median_r * 0.2)
        self.focus = { (1.0, 1e6): (max(1.0, median_r - span), min(1e6, median_r + span)) }

    def _run_one(self):
        name, expr = random.choice(self.candidates)
        machine_ctx = read_machine_ctx(self.psutil_mod)
        params = self.sampler.sample(self.focus)
        result = self.evaluator.eval_numeric(expr, params)
        score = interestingness(result, params)
        rec = {
            "timestamp": now_iso(),
            "candidate_name": name,
            "expression": str(expr),
            "parameters": params,
            "result": result,
            "score": score,
            "thoughts": make_thoughts(name, params, {"result": result}, score, machine_ctx),
            "machine": machine_ctx
        }
        if self.bus:
            try:
                self.bus.publish("discovery_record", rec)
            except Exception:
                pass
        return rec

    def run_batch(self):
        records = []
        errors = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._run_one) for _ in range(self.batch_size)]
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    records.append(rec)
                except Exception as e:
                    errors.append(str(e))
        return records, errors

    def loop_forever(self, base_sleep=1.0, max_backoff=30.0, jitter=0.25):
        backoff = base_sleep
        while not self.stopped():
            try:
                recs, errs = self.run_batch()
                for rec in recs:
                    self._write_jsonl(self.log_path, rec)
                    try:
                        self.stream_queue.put_nowait(rec)
                    except queue.Full:
                        pass
                    if rec["score"] is not None and rec["score"] >= self.solved_threshold:
                        solved = {
                            "timestamp": rec["timestamp"],
                            "candidate_name": rec["candidate_name"],
                            "expression": rec["expression"],
                            "parameters": rec["parameters"],
                            "result": rec["result"],
                            "score": rec["score"]
                        }
                        self._write_jsonl(self.solved_path, solved)
                        try:
                            self.stream_queue.put_nowait({"solved": solved})
                        except queue.Full:
                            pass

                self._count += len(recs)

                if errs:
                    self._last_error = errs[-1]
                    self._write_health(status="degraded")
                else:
                    self._write_health(status="ok")

                self._update_focus(recs)

                # Adaptive cadence using machine state
                machine_ctx = read_machine_ctx(self.psutil_mod)
                sleep_s = adapt_cadence(backoff, machine_ctx)
                # jitter
                sleep_s = max(0.0, sleep_s * (1.0 + random.uniform(-jitter, jitter)))

                if self._count % self.checkpoint_every < self.batch_size:
                    self._write_checkpoint()

                time.sleep(sleep_s)
                backoff = base_sleep
            except Exception as e:
                self._last_error = str(e)
                self._write_health(status="error")
                backoff = min(backoff * 2.0, max_backoff)
                time.sleep(backoff)

# =========================
# GUI (Tkinter + Matplotlib) — threaded & throttled
# =========================
class OrchestratorSim:
    def __init__(self, bus: EventBus, nodes: List[MeshNode], scheduler: PriorityScheduler):
        self.bus = bus
        self.nodes = nodes
        self.scheduler = scheduler
        self.active = True

    def tick(self):
        if self.active and self.nodes:
            n = random.choice(self.nodes)
            item = {"ts": time.time(), "node": n.node_id, "priority": GOAL_PRIORITY,
                    "words": ["INTENT", "FORK", "RESULT"], "task": {"goal": GLOBAL_GOAL}}
            self.scheduler.submit(item)
            self.bus.publish("orchestrator", {"status": "scheduled", "node": n.node_id, "topic": "anti-gravity", "priority": GOAL_PRIORITY})

def build_core_system():
    bus = EventBus()
    deps = [
        DepSpec("psutil", import_name="psutil", version=">=5.9"),
        DepSpec("openai", import_name="openai", version=">=1.0"),
        DepSpec("requests", import_name="requests", version=">=2.31"),
        DepSpec("matplotlib", import_name="matplotlib", version=">=3.8"),
        DepSpec("sympy", import_name="sympy", version=">=1.12"),
    ]
    loader = AutoLoader(deps, bus)
    loader.ensure()

    # Bind sympy alias
    global sympy
    sympy = loader.get("sympy")

    for t in ["log", "metrics", "metrics_topic", "alert", "orchestrator", "hive_propose", "hive_decision", "ensemble_confidence", "scheduler", "discovery_record"]:
        bus.subscribe(t, lambda ev, _t=t: print(json.dumps({"ts": ev.ts, "topic": _t, **ev.payload})))

    sys_awareness = SystemAwareness(bus, loader)

    registry = ModelRegistry()
    registry.add(ModelInfo(
        name="alpha-openai", provider="openai", endpoint="https://api.openai.com/v1",
        capabilities=["chat", "code"], cost_hint=0.7, latency_hint_ms=1200, max_tokens=4096, consent=True,
        model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), hybrid=True
    ))
    registry.add(ModelInfo(
        name="beta-hf", provider="hf", endpoint="https://api-inference.huggingface.co",
        capabilities=["chat"], cost_hint=0.3, latency_hint_ms=1800, max_tokens=1024, consent=True,
        model_id=os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"), hybrid=True
    ))
    registry.add(ModelInfo(
        name="gamma-ollama", provider="ollama", endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        capabilities=["chat"], cost_hint=0.1, latency_hint_ms=900, max_tokens=2048, consent=True,
        model_id=os.getenv("OLLAMA_MODEL", "llama3"), hybrid=True
    ))
    registry.add(ModelInfo(
        name="delta-vllm", provider="vllm", endpoint=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        capabilities=["chat"], cost_hint=0.2, latency_hint_ms=800, max_tokens=2048, consent=True,
        model_id=os.getenv("VLLM_MODEL", "mixtral"), hybrid=True
    ))

    store = PredictiveStore()
    router = PredictiveRouter(registry, store)
    scheduler = PriorityScheduler(bus, sys_awareness)
    backoff = BackoffRegistry()
    adapters = ProviderAdapters(loader, backoff, bus)
    orch = Orchestrator(registry, router, sys_awareness, bus, store, adapters)

    ml = MachineLanguage()
    models = [
        MeshModelInfo("alpha-openai", ["chat", "code"], 0.7, 1200),
        MeshModelInfo("beta-hf", ["chat"], 0.3, 1800),
        MeshModelInfo("gamma-ollama", ["chat"], 0.1, 900),
        MeshModelInfo("delta-vllm", ["chat"], 0.2, 800),
    ]
    nodes = [MeshNode(m.name, bus, ml) for m in models]
    for a in nodes:
        for b in nodes:
            if a is not b:
                a.add_peer(b.node_id)

    hive = MindHive(bus)
    sim = OrchestratorSim(bus, nodes, scheduler)

    # Discovery engine
    discovery_stream = queue.Queue(maxsize=5000)
    disc = DiscoveryEngine(
        out_dir=DISCOVERY_DIR,
        batch_size=25,
        max_workers=4,
        curiosity_level=0.3,
        what_if=True,
        rule_flexibility=0.2,
        psutil_mod=loader.get("psutil"),
        bus=bus,
        stream_queue=discovery_stream
    )
    disc_thread = threading.Thread(target=lambda: disc.loop_forever(base_sleep=1.0), daemon=True)
    disc_thread.start()

    bus.publish("log", {"level": "info", "msg": "global_goal", "goal": GLOBAL_GOAL, "priority": GOAL_PRIORITY})

    return bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim, scheduler, store, backoff, disc, discovery_stream

def run_console(bus, sys_awareness, orch, sim, scheduler):
    print("Running in console mode (Tkinter not available). Press Ctrl+C to stop.")
    try:
        while True:
            sim.tick()
            item = scheduler.take()
            if item:
                messages = [{"role": "user", "content": "Outline physics-informed approaches to anti-gravity."}]
                trace = orch.run(
                    capability="chat",
                    messages=messages,
                    mode="ensemble",
                    budget=0.9,
                    latency_ms=3500,
                    topic="anti-gravity",
                    priority=GOAL_PRIORITY,
                    confidence_target=0.68,
                    max_members=4
                )
                print(json.dumps({"run_id": trace.run_id, "steps": trace.steps}, indent=2))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

def run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim, scheduler, store, loader, disc, discovery_stream):
    try:
        import tkinter as tk
        import tkinter.ttk as ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        from functools import partial
    except Exception:
        run_console(bus, sys_awareness, orch, sim, scheduler)
        return

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            root.title("Ghost Mesh — Predictive Hybrid Orchestrator (Threaded + Discovery)")
            root.geometry("1480x900")
            root.configure(bg="#111")

            self.bus = bus
            self.registry = registry
            self.orch = orch
            self.ml = ml
            self.nodes = nodes
            self.hive = hive
            self.sim = sim
            self.scheduler = scheduler
            self.store = store
            self.loader = loader
            self.discovery_stream = discovery_stream

            self.workers = WorkerPool(max_workers=4)

            # Chart data buffers
            self.conf_points: List[float] = []
            self.pred_latency_points: List[float] = []
            self.act_latency_points: List[float] = []
            self.queue_points: List[int] = []

            # Discovery buffers
            self.discovery_last: List[Dict[str, Any]] = []
            self.discovery_top_score: float = 0.0

            self._build_ui(plt, FigureCanvasTkAgg)
            self._wire_bus()

            # Fast, lightweight pulse
            self._pulse()                 # 400 ms
            # Lightweight scheduling tick (no heavy work on GUI thread)
            self._schedule_tick()         # 1200 ms
            # Heavy chart redraw at lower rate
            self._schedule_chart_update() # 2000 ms
            # Discovery stream poll
            self._poll_discovery()        # 800 ms

        def _build_ui(self, plt, FigureCanvasTkAgg):
            top = tk.Frame(self.root, bg="#111")
            top.pack(fill="x", padx=8, pady=8)

            self.status_dot = tk.Canvas(top, width=18, height=18, bg="#111", highlightthickness=0)
            self.status_dot.pack(side="left")
            self._draw_dot("#17d517")

            title = tk.Label(top, text="Ghost Mesh — Predictive Hybrid Orchestrator", fg="#eee", bg="#111", font=("Segoe UI", 14, "bold"))
            title.pack(side="left", padx=8)

            self.toggle_btn = ttk.Button(top, text="Pause", command=self._toggle)
            self.toggle_btn.pack(side="right", padx=6)

            goal_frame = tk.Frame(self.root, bg="#111")
            goal_frame.pack(fill="x", padx=8, pady=(0, 8))
            goal_label = tk.Label(goal_frame, text=f"Global goal (priority {GOAL_PRIORITY}): {GLOBAL_GOAL}",
                                  fg="#9bd", bg="#111", font=("Segoe UI", 11, "bold"))
            goal_label.pack(anchor="w")

            main = tk.Frame(self.root, bg="#111")
            main.pack(fill="both", expand=True, padx=8, pady=4)

            # Left: Mesh messages
            left = tk.Frame(main, bg="#111")
            left.pack(side="left", fill="both", expand=True)
            tk.Label(left, text="Mesh messages", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            self.log = tk.Text(left, height=24, bg="#1a1a1a", fg="#cfcfcf", insertbackground="#cfcfcf", wrap="word")
            self.log.pack(fill="both", expand=True)
            self.log.configure(state="disabled")

            # Right: Tabs
            right = tk.Frame(main, bg="#111")
            right.pack(side="right", fill="both", expand=False)
            self.tabs = ttk.Notebook(right)
            self.tabs.pack(fill="both", expand=True)

            # Nodes tab
            nodes_tab = tk.Frame(self.tabs, bg="#111")
            self.tabs.add(nodes_tab, text="Nodes")
            self.tree = ttk.Treeview(nodes_tab, columns=("name","caps","cost","lat","score"), show="headings", height=12)
            for col, txt, w in [("name","Name",180),("caps","Capabilities",280),("cost","Cost",90),("lat","Latency ms",120),("score","Predictive score",150)]:
                self.tree.heading(col, text=txt)
                self.tree.column(col, width=w)
            self.tree.pack(fill="both", expand=True, pady=4)
            for m in self.registry.query("chat"):
                sc = f"{self.store.score(m.name):.3f}"
                self.tree.insert("", "end", values=(m.name, ", ".join(m.capabilities), f"{m.cost_hint:.2f}", m.latency_hint_ms, sc), iid=m.name)

            # Lexicon tab
            lex_tab = tk.Frame(self.tabs, bg="#111")
            self.tabs.add(lex_tab, text="Lexicon")
            tk.Label(lex_tab, text="Evolving lexicon (token → meaning)", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6,2))
            self.lex_list = tk.Listbox(lex_tab, height=16, bg="#1a1a1a", fg="#cfcfcf")
            self.lex_list.pack(fill="both", expand=True)
            self._refresh_lexicon()

            # Hive tab
            hive_tab = tk.Frame(self.tabs, bg="#111")
            self.tabs.add(hive_tab, text="Hive")
            tk.Label(hive_tab, text="Proposals", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6,2))
            self.prop_list = tk.Listbox(hive_tab, height=12, bg="#1a1a1a", fg="#cfcfcf")
            self.prop_list.pack(fill="both", expand=True, pady=(2,6))
            tk.Label(hive_tab, text="Decisions", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2,2))
            self.decisions_list = tk.Listbox(hive_tab, height=10, bg="#1a1a1a", fg="#cfcfcf")
            self.decisions_list.pack(fill="both", expand=True, pady=(2,6))

            # Telemetry tab with charts
            tele_tab = tk.Frame(self.tabs, bg="#111")
            self.tabs.add(tele_tab, text="Telemetry")
            self.telemetry = tk.Listbox(tele_tab, height=8, bg="#1a1a1a", fg="#cfcfcf")
            self.telemetry.pack(fill="x", expand=False, pady=(6,6))

            self.badges = tk.Label(tele_tab, text="Confidence: -- | Queue: -- | SLA breaches: -- | Anomaly: -- | Top Interestingness: --",
                                   fg="#ccc", bg="#111", font=("Segoe UI", 10, "bold"))
            self.badges.pack(anchor="w")

            fig = plt.Figure(figsize=(7.8, 4.2), dpi=100)
            self.ax_conf = fig.add_subplot(131)
            self.ax_lat = fig.add_subplot(132)
            self.ax_queue = fig.add_subplot(133)

            self.ax_conf.set_title("Agreement trend")
            self.ax_conf.set_ylim(0, 1)
            self.ax_conf.grid(True, alpha=0.3)

            self.ax_lat.set_title("Latency: predicted vs actual")
            self.ax_lat.grid(True, alpha=0.3)

            self.ax_queue.set_title("Queue depth")
            self.ax_queue.grid(True, alpha=0.3)

            self.canvas = FigureCanvasTkAgg(fig, master=tele_tab)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="x", expand=False, pady=6)

            # Discovery tab
            disc_tab = tk.Frame(self.tabs, bg="#111")
            self.tabs.add(disc_tab, text="Discovery")
            tk.Label(disc_tab, text="Transparent thoughts (latest)", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6,2))
            self.disc_list = tk.Listbox(disc_tab, height=16, bg="#1a1a1a", fg="#cfcfcf")
            self.disc_list.pack(fill="both", expand=True, pady=(2,6))
            tk.Label(disc_tab, text="Top candidates (scores)", fg="#ccc", bg="#111", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6,2))
            self.top_list = tk.Listbox(disc_tab, height=10, bg="#1a1a1a", fg="#cfcfcf")
            self.top_list.pack(fill="both", expand=True, pady=(2,6))

            footer = tk.Frame(self.root, bg="#111")
            footer.pack(fill="x", padx=8, pady=6)
            self.preview = tk.Label(footer, text="Decoded preview will appear here…", fg="#aaa", bg="#111", font=("Segoe UI", 10))
            self.preview.pack(anchor="w")

            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("Treeview", background="#1a1a1a", foreground="#cfcfcf", fieldbackground="#1a1a1a")
            style.map("Treeview", background=[("selected", "#333")], foreground=[("selected", "#eee")])

        def _wire_bus(self):
            self.bus.subscribe("mesh", lambda ev: self._append_log(ev.payload))
            self.bus.subscribe("orchestrator", lambda ev: self._status_update(ev.payload))
            self.bus.subscribe("hive_propose", lambda ev: self._add_proposal(ev.payload))
            self.bus.subscribe("hive_decision", lambda ev: self._add_decision(ev.payload))
            self.bus.subscribe("ensemble_confidence", lambda ev: self._update_confidence(ev.payload))
            self.bus.subscribe("metrics", lambda ev: self._update_metrics(ev.payload))
            self.bus.subscribe("metrics_topic", lambda ev: self._update_topic_metrics(ev.payload))
            self.bus.subscribe("scheduler", lambda ev: self._update_queue(ev.payload))
            self.bus.subscribe("discovery_record", lambda ev: self._on_discovery(ev.payload))

        def _draw_dot(self, color: str):
            self.status_dot.delete("all")
            self.status_dot.create_oval(3, 3, 15, 15, fill=color, outline=color)

        def _pulse(self):
            color = "#17d517" if self.sim.active and random.random() < 0.6 else ("#0faa0f" if self.sim.active else "#b33")
            self._draw_dot(color)
            self.root.after(400, self._pulse)

        def _schedule_tick(self):
            # Lightweight: schedule and submit heavy runs to worker pool
            self.sim.tick()
            item = self.scheduler.take()
            if item:
                self._start_run(
                    topic="anti-gravity",
                    messages=[{"role": "user", "content": "Outline physics-informed approaches to anti-gravity."}],
                    priority=GOAL_PRIORITY
                )
            self._refresh_nodes_scores()
            if random.random() < 0.35:
                self._refresh_lexicon()
            self.root.after(1200, self._schedule_tick)

        def _start_run(self, topic: str, messages: List[Dict[str, str]], priority: float):
            def run_and_return():
                return self.orch.run(
                    capability="chat",
                    messages=messages,
                    mode="ensemble",
                    budget=0.9,
                    latency_ms=3500,
                    topic=topic,
                    priority=priority,
                    confidence_target=0.68,
                    max_members=4
                )
            future = self.workers.submit(run_and_return)
            from functools import partial
            future.add_done_callback(lambda fut: self.root.after(0, partial(self._on_run_complete, fut)))

        def _on_run_complete(self, fut):
            try:
                trace = fut.result()
                used = len([s for s in trace.steps if s.get("type") in ("ensemble_member","probe")])
                self.preview.configure(text=f"Last run: {trace.run_id} anti-gravity members={used}")
            except Exception as e:
                self.preview.configure(text=f"Run error: {e}")

        def _refresh_nodes_scores(self):
            for m in self.registry.query("chat"):
                sc = f"{self.store.score(m.name):.3f}"
                if self.tree.exists(m.name):
                    self.tree.item(m.name, values=(m.name, ", ".join(m.capabilities), f"{m.cost_hint:.2f}", m.latency_hint_ms, sc))

        def _schedule_chart_update(self):
            self._refresh_charts()
            self.root.after(2000, self._schedule_chart_update)

        def _refresh_charts(self):
            self.ax_conf.cla()
            self.ax_conf.set_title("Agreement trend")
            self.ax_conf.set_ylim(0, 1)
            self.ax_conf.grid(True, alpha=0.3)
            self.ax_conf.plot(self.conf_points[-50:], color="#4caf50")

            self.ax_lat.cla()
            self.ax_lat.set_title("Latency: predicted vs actual")
            self.ax_lat.grid(True, alpha=0.3)
            self.ax_lat.plot(self.pred_latency_points[-50:], label="predicted", color="#2196f3")
            self.ax_lat.plot(self.act_latency_points[-50:], label="actual", color="#f44336")
            self.ax_lat.legend(loc="upper right")

            self.ax_queue.cla()
            self.ax_queue.set_title("Queue depth")
            self.ax_queue.grid(True, alpha=0.3)
            self.ax_queue.plot(self.queue_points[-50:], color="#ff9800")

            self.canvas.draw_idle()

        def _toggle(self):
            self.sim.active = not self.sim.active
            self.toggle_btn.configure(text="Resume" if not self.sim.active else "Pause")
            self.preview.configure(text="Paused." if not self.sim.active else "Resumed.")

        def _append_log(self, payload: Dict[str, Any]):
            raw = payload.get("raw", "")
            decoded = payload.get("decoded", "")
            line = f"[{payload.get('from')}] {raw}\n"
            self.log.configure(state="normal")
            self.log.insert("end", line)
            if int(self.log.index("end-1c").split(".")[0]) > 500:
                self.log.delete("1.0", "3.0")
            self.log.configure(state="disabled")
            self.log.see("end")
            self.preview.configure(text=f"Decoded: {decoded}")

        def _status_update(self, payload: Dict[str, Any]):
            nid = payload.get("node")
            if nid and self.tree.exists(nid):
                try:
                    self.tree.selection_set(nid)
                    self.root.after(350, lambda: self.tree.selection_remove(nid))
                except Exception:
                    pass

        def _refresh_lexicon(self):
            self.lex_list.delete(0, "end")
            items = list(self.ml.lexicon.items())
            random.shuffle(items)
            for k, v in items[:50]:
                self.lex_list.insert("end", f"{k}  →  {v}")

        def _add_proposal(self, payload: Dict[str, Any]):
            pid = payload.get("proposal_id")
            topic = payload.get("topic")
            deadline = payload.get("deadline")
            remaining = max(0.0, round(deadline - time.time(), 1)) if isinstance(deadline, float) else "?"
            goal = payload.get("payload", {}).get("goal")
            label = f"{pid} | {topic} | deadline {remaining}s"
            if goal == GLOBAL_GOAL:
                label += " | GOAL: Anti-Gravity"
            self.prop_list.insert("end", label)

        def _add_decision(self, payload: Dict[str, Any]):
            pid = payload.get("proposal_id")
            decision = payload.get("decision")
            rationale = payload.get("rationale")
            acks = len(payload.get("acks", []))
            nacks = len(payload.get("nacks", []))
            self.decisions_list.insert("end", f"{pid} | {decision} | acks={acks} nacks={nacks} | {rationale}")

        def _update_confidence(self, payload: Dict[str, Any]):
            agreement = payload.get("agreement")
            slope = payload.get("slope", 0.0)
            anomaly = payload.get("anomaly", False)
            if agreement is not None:
                self.conf_points.append(float(agreement))
            current_text = self.badges.cget("text")
            parts = current_text.split("|")
            parts[0] = f" Confidence: {agreement:.3f} (Δ {slope:+.3f}) "
            parts[3] = f" Anomaly: {' YES' if anomaly else ' NO'} "
            self.badges.configure(text="|".join(parts))

        def _update_metrics(self, payload: Dict[str, Any]):
            queue = payload.get("queue_depth")
            if queue is not None:
                self.queue_points.append(int(queue))
            current_text = self.badges.cget("text")
            parts = current_text.split("|")
            parts[1] = f" Queue: {queue if queue is not None else '--'} "
            self.badges.configure(text="|".join(parts))

        def _update_topic_metrics(self, payload: Dict[str, Any]):
            if payload.get("topic") == "anti-gravity":
                breaches = payload.get("sla_breaches")
                current_text = self.badges.cget("text")
                parts = current_text.split("|")
                parts[2] = f" SLA breaches: {breaches if breaches is not None else '--'} "
                self.badges.configure(text="|".join(parts))

        def _update_queue(self, payload: Dict[str, Any]):
            self.telemetry.insert("end", f"Queue depth: {payload.get('queued')}")

        # Discovery integration
        def _on_discovery(self, rec: Dict[str, Any]):
            # Append brief thoughts to disc_list
            name = rec.get("candidate_name")
            score = rec.get("score")
            result = rec.get("result")
            thoughts = rec.get("thoughts", [])
            line = f"{name}: score={score:.3f} result={result}"
            try:
                self.disc_list.insert("end", line)
            except Exception:
                self.disc_list.insert("end", f"{name}: score={score} result={result}")
            if self.disc_list.size() > 200:
                self.disc_list.delete(0, 20)
            # Top list management
            self.discovery_last.append(rec)
            self.discovery_last = self.discovery_last[-200:]
            top = sorted(self.discovery_last, key=lambda r: r.get("score") or 0.0, reverse=True)[:10]
            self.top_list.delete(0, "end")
            for t in top:
                self.top_list.insert("end", f"{t['candidate_name']} | {t['score']:.3f}")
            # Badge update
            self.discovery_top_score = top[0]['score'] if top else 0.0
            current_text = self.badges.cget("text")
            parts = current_text.split("|")
            parts[4] = f" Top Interestingness: {self.discovery_top_score:.3f}"
            self.badges.configure(text="|".join(parts))

        def _poll_discovery(self):
            # In case stream_queue used, poll it (bus also pushes events)
            try:
                for _ in range(10):  # drain a few
                    rec = self.discovery_stream.get_nowait()
                    self._on_discovery(rec)
            except Exception:
                pass
            self.root.after(800, self._poll_discovery)

    root = tk.Tk()
    app = App(root)
    try:
        root.mainloop()
    finally:
        for n in nodes:
            n.stop()
        bus.stop()
        sys_awareness.stop()
        disc.stop()

def main():
    bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim, scheduler, store, backoff, disc, discovery_stream = build_core_system()
    try:
        run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim, scheduler, store, loader, disc, discovery_stream)
    except Exception as e:
        print(f"GUI failed ({e}). Falling back to console mode.")
        run_console(bus, sys_awareness, orch, sim, scheduler)

if __name__ == "__main__":
    main()

