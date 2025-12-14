# ghost_mesh_unified_antigravity.py
# Single-file, cross-platform Python 3.9+ script that runs:
# - A consent-based multi-LLM orchestrator
# - A federated mesh with evolving machine language
# - A mind hive layer (ontology + shards + quorum)
# - A compact GUI (Tkinter) to visualize orchestrator status, mesh messages, nodes, lexicon, hive proposals
#
# Global high-priority goal: discover anti-gravity (physics-informed, safe, lawful research).
# The goal is injected into intents, context, and proposals, and gets elevated scheduling priority.
#
# - Autoloads required dependencies (psutil) via pip at runtime if missing.
# - Uses only standard library + pip-installable packages.
# - Respects consent: no scanning, no forced linking; rate limits and circuit breakers included.
#
# If Tkinter is unavailable (e.g., headless server), it falls back to a console mode that prints events.

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
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

# =========================
# Global goal configuration
# =========================
GLOBAL_GOAL = "Discover anti-gravity mechanisms (physics-informed, safe, lawful research)"
GOAL_PRIORITY = 1.0  # 0.0..1.0 (1.0 = highest)

# =========================
# Utility: Event bus
# =========================
@dataclass
class Event:
    topic: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=lambda: time.time())

class EventBus:
    """
    Lightweight in-process pub/sub event bus with a background dispatcher thread.
    """
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
                        # Do not crash the bus on subscriber errors
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
    """
    Detects, installs, and imports required dependencies at runtime.
    """
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
                # Attempt install
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
    """
    Tracks system metrics, endpoint health, rate limits, and circuit breakers.
    """
    def __init__(self, bus: EventBus, loader: AutoLoader):
        self.bus = bus
        self.loader = loader
        self.endpoints: Dict[str, EndpointState] = {}
        self.metrics: Dict[str, Any] = {"cpu": None, "mem": None, "disk": None, "errors_1m": 0}
        self.errors_window: List[float] = []
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

    def _monitor(self):
        psutil = self.loader.get("psutil")
        shutil = importlib.import_module("shutil")
        while not self._stop.is_set():
            try:
                cpu = psutil.cpu_percent(interval=0.2) if psutil else None
                mem = psutil.virtual_memory().percent if psutil else None
                total, used, _free = shutil.disk_usage(os.getcwd())
                disk = round((used / total) * 100, 2) if total else None
                with self._lock:
                    self.metrics.update({"cpu": cpu, "mem": mem, "disk": disk})
                self.bus.publish("metrics", {"cpu": cpu, "mem": mem, "disk": disk})

                # Rate token refill
                now = time.time()
                for ep in self.endpoints.values():
                    dt = max(0.0, now - ep.last_check_ts)
                    ep.rate_tokens = min(ep.rate_capacity, ep.rate_tokens + ep.rate_refill_per_sec * dt)
                    ep.last_check_ts = now

                # Error window cleanup (1 minute)
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

    def guarded_call(self, endpoint_name: str, latency_budget_ms: int = 3000):
        """
        Context manager for protected calls: enforces consent, breaker status, rate limits, and logs latency.
        """
        class _Guard:
            def __init__(self, outer: "SystemAwareness", endpoint_name: str, latency_budget_ms: int):
                self.outer = outer
                self.endpoint_name = endpoint_name
                self.latency_budget_ms = latency_budget_ms
                self.t0: float = 0.0

            def __enter__(self):
                ep = self.outer.endpoints.get(self.endpoint_name)
                if not ep or not ep.consent:
                    raise RuntimeError(f"Endpoint not consented or unknown: {self.endpoint_name}")
                if ep.breaker_open:
                    raise RuntimeError(f"Circuit open for endpoint: {self.endpoint_name}")
                if ep.rate_tokens < 1.0:
                    raise RuntimeError(f"Rate limited: {self.endpoint_name}")
                ep.rate_tokens -= 1.0
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
                    self.outer._log("warn", "latency_budget_exceeded", endpoint=self.endpoint_name, latency_ms=round(dt_ms, 1))
                return False

        return _Guard(self, endpoint_name, latency_budget_ms)

# =========================
# Orchestrator
# =========================
@dataclass
class ModelInfo:
    name: str
    provider: str
    endpoint: str
    capabilities: List[str]
    cost_hint: float
    latency_hint_ms: int
    max_tokens: int
    consent: bool = True

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

class Adapter:
    def __init__(self, call_fn: Callable[[ModelInfo, Dict[str, Any]], Dict[str, Any]]):
        self.call = call_fn

def _synthetic_call(model: ModelInfo, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthetic adapter: simulates latency and returns a placeholder reply.
    Replace with real provider SDK calls that respect TOS and consent.
    """
    # Slight latency reduction for high-priority anti-gravity tasks (simulated scheduling preference)
    base_delay = min(model.latency_hint_ms, 1500) / 1000.0
    delay = base_delay * (0.75 if payload.get("priority", 0.0) >= 0.9 else 1.0)
    time.sleep(delay)
    topic = payload.get("topic", "general")
    return {
        "model": model.name,
        "output": f"[{model.name}] reply ({topic})",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200}
    }

ADAPTERS = {
    "openai": Adapter(_synthetic_call),
    "huggingface": Adapter(_synthetic_call),
    "custom": Adapter(_synthetic_call),
}

class Router:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def pick(self, capability: str, budget: float, latency_ms: int, priority: float = 0.0) -> List[ModelInfo]:
        candidates = self.registry.query(capability)
        # High-priority tasks prefer lower latency and cost more aggressively
        ranked = sorted(candidates, key=lambda m: (m.latency_hint_ms * (0.8 if priority >= 0.9 else 1.0), m.cost_hint))
        return [m for m in ranked if m.latency_hint_ms <= latency_ms and m.cost_hint <= budget]

class Orchestrator:
    """
    Routes or ensembles model calls under policy, with detailed event logging.
    """
    def __init__(self, registry: ModelRegistry, router: Router, sys_awareness: SystemAwareness, bus: EventBus):
        self.registry = registry
        self.router = router
        self.sys = sys_awareness
        self.bus = bus

    def run(self, capability: str, messages: List[Dict[str, str]], mode: str = "ensemble",
            budget: float = 0.5, latency_ms: int = 2000, topic: str = "general", priority: float = 0.0) -> Trace:
        trace = Trace(run_id=str(uuid.uuid4()))
        models = self.router.pick(capability, budget, latency_ms, priority=priority)
        if not models:
            raise RuntimeError("No models available under current policy.")
        for m in models[:3]:
            if m.name not in self.sys.endpoints:
                self.sys.add_endpoint(m.name, m.endpoint, consent=m.consent, rate_capacity=5, refill=1.0)

        if mode == "route":
            m = models[0]
            with self.sys.guarded_call(m.name, latency_budget_ms=latency_ms):
                out = ADAPTERS[m.provider].call(m, {"messages": messages, "temperature": 0.2,
                                                    "max_tokens": min(1024, m.max_tokens),
                                                    "topic": topic, "priority": priority})
            trace.steps.append({"type": "single", "model": m.name, "output": out["output"], "usage": out["usage"]})
            self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "route", "model": m.name, "topic": topic, "priority": priority})
            return trace

        outputs = []
        members = []
        for m in models[:3]:
            members.append(m.name)
            try:
                with self.sys.guarded_call(m.name, latency_budget_ms=latency_ms):
                    out = ADAPTERS[m.provider].call(m, {"messages": messages, "temperature": 0.2,
                                                        "max_tokens": min(1024, m.max_tokens),
                                                        "topic": topic, "priority": priority})
                trace.steps.append({"type": "ensemble_member", "model": m.name, "output": out["output"], "usage": out["usage"]})
                outputs.append(out["output"])
                self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "ensemble_member", "model": m.name, "topic": topic, "priority": priority})
            except Exception as e:
                trace.steps.append({"type": "ensemble_error", "model": m.name, "error": str(e)})
                self.bus.publish("alert", {"severity": "warn", "msg": "ensemble_member_failed", "model": m.name, "err": str(e)})

        merged = "\n\n".join(outputs) if outputs else "[no outputs]"
        trace.steps.append({"type": "merge", "strategy": "concat", "output": merged, "topic": topic})
        self.bus.publish("orchestrator", {"run_id": trace.run_id, "mode": "merge", "members": members, "topic": topic, "priority": priority})
        return trace

# =========================
# Mesh + evolving machine language
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
    """
    Evolving machine-native lexicon: tokens mutate over time; nodes communicate via these tokens.
    """
    ALPHABET = string.ascii_letters + string.digits + "+/="

    def __init__(self, seed_meanings: Optional[List[str]] = None):
        self.lexicon: Dict[str, str] = {}
        self.seed_meanings = seed_meanings or ["ACK", "NACK", "INTENT", "STATE", "RESULT", "QUERY", "MERGE", "FORK", "CONSENSUS", "GOAL"]
        for m in self.seed_meanings:
            self.lexicon[self._new_token()] = m
        self.mutation_rate = 0.15

    def _new_token(self, length: Optional[int] = None) -> str:
        L = length or random.randint(8, 12)
        return "".join(random.choice(self.ALPHABET) for _ in range(L))

    def evolve(self):
        keys = list(self.lexicon.keys())
        if not keys:
            return
        random.shuffle(keys)
        k_mut = max(1, int(len(keys) * self.mutation_rate))
        for k in keys[:k_mut]:
            s = list(k)
            for _ in range(random.randint(1, 2)):
                i = random.randint(0, len(s) - 1)
                s[i] = random.choice(self.ALPHABET)
            new_k = "".join(s)
            if new_k not in self.lexicon:
                self.lexicon[new_k] = self.lexicon[k]
                del self.lexicon[k]
        # Occasionally add a new concept
        if random.random() < 0.4:
            concept = random.choice(["DIVERGENCE", "HORIZON", "LATENCY", "COST", "CONFIDENCE", "PRISM", "VECTOR", "DELTA", "ANTIGRAVITY"])
            self.lexicon[self._new_token()] = concept

    def encode_message(self, words: List[str]) -> str:
        inv = {v: k for k, v in self.lexicon.items()}
        out = []
        for w in words:
            if w in inv:
                out.append(inv[w])
            else:
                t = self._new_token()
                self.lexicon[t] = w
                out.append(t)
        return " ".join(out)

    def decode_message(self, msg: str) -> str:
        parts = msg.split()
        decoded = []
        for p in parts:
            decoded.append(self.lexicon.get(p, "?"))
        return " ".join(decoded)

class MeshNode:
    """
    Mesh node that gossips intents and shares state via the event bus.
    """
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
        # Inject GOAL tag and goal text into intent words
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
            # Alternate intents; anti-gravity gets high priority pulses
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
# Mind hive (ontology, shards, quorum)
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
    task: LWWMap = field(default_factory=LWWMap)      # DAGs, steps, evidence
    context: LWWMap = field(default_factory=LWWMap)   # constraints, shared ontology items
    telemetry: LWWMap = field(default_factory=LWWMap) # health, rates, latencies summaries

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
    """
    Federated hive layer: collects intents and state, proposes plans, and decides via quorum.
    Anti-gravity goal is announced in context and prioritized in planning.
    """
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.shards = HiveShards()
        self.proposals: Dict[str, Proposal] = {}
        self._wire_topics()
        # Announce global goal in context at startup
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
        # Record intent
        self.shards.task.set(f"intent:{node_id}:{uuid.uuid4().hex[:6]}", {"decoded": decoded, "task": task, "ts": ev.ts})
        # Propose a plan, weighted by priority
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
        # Bootstrap with a local ack (for demo)
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
        # Simple quorum rule: commit if acks >= 2 and no nacks
        if len(prop.acks) >= 2 and len(prop.nacks) == 0:
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
# GUI (Tkinter) with fallback to console mode
# =========================
class OrchestratorSim:
    """
    Drives periodic orchestrator-like activity through the mesh for demo purposes.
    """
    def __init__(self, bus: EventBus, nodes: List[MeshNode]):
        self.bus = bus
        self.nodes = nodes
        self.active = True

    def tick(self):
        if self.active and self.nodes:
            n = random.choice(self.nodes)
            n.publish_intent(["INTENT", "FORK", "RESULT"], task={"goal": GLOBAL_GOAL, "dag": {"steps": [{"role": "planner"}]}}, priority=GOAL_PRIORITY)
            self.bus.publish("orchestrator", {"status": "running", "node": n.node_id, "topic": "anti-gravity", "priority": GOAL_PRIORITY})

def build_core_system():
    """
    Build bus, autoloader, system awareness, orchestrator, mesh, and hive.
    """
    bus = EventBus()
    loader = AutoLoader([DepSpec("psutil", import_name="psutil", version=">=5.9")], bus)
    loader.ensure()

    # Log subscribers (console visibility)
    for t in ["log", "metrics", "alert", "orchestrator", "hive_propose", "hive_decision"]:
        bus.subscribe(t, lambda ev, _t=t: print(json.dumps({"ts": ev.ts, "topic": _t, **ev.payload})))

    sys_awareness = SystemAwareness(bus, loader)

    registry = ModelRegistry()
    registry.add(ModelInfo(
        name="alpha-chat", provider="openai", endpoint="https://api.alpha.chat",
        capabilities=["chat", "code"], cost_hint=0.4, latency_hint_ms=900, max_tokens=4096, consent=True
    ))
    registry.add(ModelInfo(
        name="beta-hf", provider="huggingface", endpoint="https://api.hf.inference",
        capabilities=["chat"], cost_hint=0.2, latency_hint_ms=1200, max_tokens=2048, consent=True
    ))
    registry.add(ModelInfo(
        name="gamma-local", provider="custom", endpoint="http://localhost:8000/v1/chat",
        capabilities=["chat"], cost_hint=0.05, latency_hint_ms=600, max_tokens=1024, consent=True
    ))

    router = Router(registry)
    orch = Orchestrator(registry, router, sys_awareness, bus)

    ml = MachineLanguage()
    models = [
        MeshModelInfo("alpha-chat", ["chat", "code"], 0.4, 900),
        MeshModelInfo("beta-hf", ["chat"], 0.2, 1200),
        MeshModelInfo("gamma-local", ["chat"], 0.05, 600),
    ]
    nodes = [MeshNode(m.name, bus, ml) for m in models]
    for a in nodes:
        for b in nodes:
            if a is not b:
                a.add_peer(b.node_id)

    hive = MindHive(bus)
    sim = OrchestratorSim(bus, nodes)

    # Broadcast global goal event for visibility
    bus.publish("log", {"level": "info", "msg": "global_goal", "goal": GLOBAL_GOAL, "priority": GOAL_PRIORITY})

    return bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim

def run_console(bus, sys_awareness, orch, sim):
    """
    Console fallback loop if Tkinter is not available.
    """
    print("Running in console mode (Tkinter not available). Press Ctrl+C to stop.")
    try:
        t0 = time.time()
        while True:
            # Drive orchestrator demo every ~1.2s
            sim.tick()
            # Occasionally perform a high-priority anti-gravity run (synthetic)
            if time.time() - t0 > 5.0:
                t0 = time.time()
                trace = orch.run(
                    capability="chat",
                    messages=[{"role": "user", "content": "Outline physics-informed approaches to anti-gravity."}],
                    mode="ensemble",
                    budget=0.5,
                    latency_ms=2000,
                    topic="anti-gravity",
                    priority=GOAL_PRIORITY
                )
                print(json.dumps({"run_id": trace.run_id, "steps": trace.steps}, indent=2))
            time.sleep(1.2)
    except KeyboardInterrupt:
        pass

def run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim):
    """
    Tkinter GUI showing mesh messages, nodes, lexicon, and hive proposals/decisions.
    """
    try:
        import tkinter as tk
        import tkinter.ttk as ttk
    except Exception:
        run_console(bus, sys_awareness, orch, sim)
        return

    class App:
        def __init__(self, root: tk.Tk):
            self.root = root
            root.title("Ghost Mesh — Orchestrator and Hive (Anti-Gravity Priority)")
            root.geometry("1180x720")
            root.configure(bg="#111")

            self.bus = bus
            self.registry = registry
            self.orch = orch
            self.ml = ml
            self.nodes = nodes
            self.hive = hive
            self.sim = sim

            self._build_ui()
            self._wire_bus()

            # Pulse and tick loops
            self._pulse()
            self._tick()

        def _build_ui(self):
            top = tk.Frame(self.root, bg="#111")
            top.pack(fill="x", padx=8, pady=8)

            self.status_dot = tk.Canvas(top, width=18, height=18, bg="#111", highlightthickness=0)
            self.status_dot.pack(side="left")
            self._draw_dot("#17d517")

            title = tk.Label(top, text="Ghost Mesh — Orchestrator and Hive", fg="#eee", bg="#111", font=("Segoe UI", 14, "bold"))
            title.pack(side="left", padx=8)

            self.toggle_btn = ttk.Button(top, text="Pause", command=self._toggle)
            self.toggle_btn.pack(side="right", padx=6)

            # Global goal banner
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
            self.log = tk.Text(left, height=26, bg="#1a1a1a", fg="#cfcfcf", insertbackground="#cfcfcf", wrap="word")
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
            self.tree = ttk.Treeview(nodes_tab, columns=("name","caps","cost","lat"), show="headings", height=12)
            for col, txt, w in [("name","Name",180),("caps","Capabilities",260),("cost","Cost",80),("lat","Latency ms",110)]:
                self.tree.heading(col, text=txt)
                self.tree.column(col, width=w)
            self.tree.pack(fill="both", expand=True, pady=4)
            for m in self.registry.query("chat"):
                self.tree.insert("", "end", values=(m.name, ", ".join(m.capabilities), f"{m.cost_hint:.2f}", m.latency_hint_ms), iid=m.name)

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

            # Footer: decoded preview
            footer = tk.Frame(self.root, bg="#111")
            footer.pack(fill="x", padx=8, pady=6)
            self.preview = tk.Label(footer, text="Decoded preview will appear here…", fg="#aaa", bg="#111", font=("Segoe UI", 10))
            self.preview.pack(anchor="w")

            # Style tweaks
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

        def _draw_dot(self, color: str):
            self.status_dot.delete("all")
            self.status_dot.create_oval(3, 3, 15, 15, fill=color, outline=color)

        def _pulse(self):
            # Status pulse (green when active, red when paused)
            color = "#17d517" if self.sim.active and random.random() < 0.6 else ("#0faa0f" if self.sim.active else "#b33")
            self._draw_dot(color)
            self.root.after(400, self._pulse)

        def _tick(self):
            # Drive orchestrator simulation and refresh lexicon periodically
            self.sim.tick()
            if random.random() < 0.35:
                self._refresh_lexicon()
            # Periodically run high-priority anti-gravity orchestration
            if random.random() < 0.5:
                try:
                    trace = self.orch.run(
                        capability="chat",
                        messages=[{"role": "user", "content": "Outline physics-informed approaches to anti-gravity."}],
                        mode="ensemble",
                        budget=0.5,
                        latency_ms=2000,
                        topic="anti-gravity",
                        priority=GOAL_PRIORITY
                    )
                    self.preview.configure(text=f"Last run: {trace.run_id} topic=anti-gravity")
                except Exception as e:
                    self.preview.configure(text=f"Orchestrator error: {e}")
            self.root.after(1200, self._tick)

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
            # Trim very old lines to avoid unbounded growth
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
            # Highlight anti-gravity plans
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

    root = tk.Tk()
    app = App(root)
    try:
        root.mainloop()
    finally:
        # Cleanup: stop node threads and bus/awareness
        for n in nodes:
            n.stop()
        bus.stop()
        sys_awareness.stop()

def main():
    bus, loader, sys_awareness, registry, router, orch, ml, nodes, hive, sim = build_core_system()

    # Try GUI; if import fails, console fallback occurs inside run_gui
    try:
        run_gui(bus, sys_awareness, registry, orch, ml, nodes, hive, sim)
    except Exception as e:
        print(f"GUI failed ({e}). Falling back to console mode.")
        run_console(bus, sys_awareness, orch, sim)

if __name__ == "__main__":
    main()

