# ghost_mesh_unified_predictive_hybrid_live_threaded.py
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
# Dependencies auto-installed: psutil, openai, requests, matplotlib

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

# =========================
# Persistence utilities
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
        psutil = self.loader.get("psutil")
        shutil = importlib.import_module("shutil")
        while not self._stop.is_set():
            try:
                cpu = psutil.cpu_percent(interval=0.2) if psutil else None
                mem = psutil.virtual_memory().percent if psutil else None
                total, used, _ = shutil.disk_usage(os.getcwd())
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

        merged = "\n\n".join(outputs) if outputs else "[no outputs]"
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
    ]
    loader = AutoLoader(deps, bus)
    loader.ensure()

    for t in ["log", "metrics", "metrics_topic", "alert", "orchestrator", "hive_propose", "hive_decision", "ensemble_confidence", "scheduler"]:
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

    bus.publish("log", {"level": "info", "msg": "global_goal", "goal": GLOBAL_GOAL, "priority": GOAL_PRIORITY})

    return bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim, scheduler, store, backoff

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

def run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim, scheduler, store, loader):
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
            root.title("Ghost Mesh — Predictive Hybrid Orchestrator (Threaded)")
            root.geometry("1360x840")
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

            self.workers = WorkerPool(max_workers=4)

            # Chart data buffers
            self.conf_points: List[float] = []
            self.pred_latency_points: List[float] = []
            self.act_latency_points: List[float] = []
            self.queue_points: List[int] = []

            self._build_ui(plt, FigureCanvasTkAgg)
            self._wire_bus()

            # Fast, lightweight pulse
            self._pulse()                 # 400 ms
            # Lightweight scheduling tick (no heavy work on GUI thread)
            self._schedule_tick()         # 1200 ms
            # Heavy chart redraw at lower rate
            self._schedule_chart_update() # 2000 ms

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
            self.log = tk.Text(left, height=28, bg="#1a1a1a", fg="#cfcfcf", insertbackground="#cfcfcf", wrap="word")
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

            self.badges = tk.Label(tele_tab, text="Confidence: -- | Queue: -- | SLA breaches: -- | Anomaly: --",
                                   fg="#ccc", bg="#111", font=("Segoe UI", 10, "bold"))
            self.badges.pack(anchor="w")

            fig = plt.Figure(figsize=(7.0, 4.2), dpi=100)
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
            self.root.after(1200, self._schedule_tick)  # 1.2s cadence, no heavy work on GUI thread

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
            self._refresh_charts()  # Heavy redraw isolated to slower cadence
            self.root.after(2000, self._schedule_chart_update)  # 2s cadence

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
            parts[3] = f" Anomaly: {'YES' if anomaly else 'NO'}"
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

    root = tk.Tk()
    app = App(root)
    try:
        root.mainloop()
    finally:
        for n in nodes:
            n.stop()
        bus.stop()
        sys_awareness.stop()

def main():
    bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim, scheduler, store, backoff = build_core_system()
    try:
        run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim, scheduler, store, loader)
    except Exception as e:
        print(f"GUI failed ({e}). Falling back to console mode.")
        run_console(bus, sys_awareness, orch, sim, scheduler)

if __name__ == "__main__":
    main()

