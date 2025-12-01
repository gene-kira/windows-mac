#!/usr/bin/env python3
# autonomous_codex_engine.py
# Safe, transparent, autonomous engine with:
# - Adaptive Codex Mutation (purge logic rewritten from predictive patterns)
# - Ghost sync response (telemetry retention shortening, phantom node insertion)
# - Swarm-wide codex sync (evolving rules across nodes)
# - Autoloader for common libraries

import time
import json
import signal
import threading
import queue
import hashlib
import os
import random
import socket
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ---------- Autoloader ----------

def autoload(modules: List[str]) -> Dict[str, bool]:
    status = {}
    for m in modules:
        try:
            __import__(m)
            status[m] = True
        except Exception:
            status[m] = False
    return status

# Attempt imports commonly used in CV/ML pipelines.
AUTOLOAD_STATUS = autoload(["numpy", "cv2", "torch", "scipy", "psutil"])
# Access optional modules defensively.
np = None
cv2 = None
torch = None
psutil = None
try:
    import numpy as np
except Exception:
    pass
try:
    import cv2
except Exception:
    pass
try:
    import torch
except Exception:
    pass
try:
    import psutil
except Exception:
    pass

# ---------- Configuration ----------

@dataclass
class Config:
    tick_hz: int = 30
    max_backoff_ms: int = 2000
    audit_log_path: str = "audit.log.jsonl"
    telemetry_enabled: bool = True
    telemetry_retention_max: int = 1024            # Dynamic; shortened on ghost sync
    consent_required: bool = True
    enable_anomaly_detection: bool = True
    enable_watchdog: bool = True
    device_check_interval_s: int = 5
    manifest_path: str = "manifest.hash"
    ethics_policy_path: str = "ethics.policy.json"
    dashboard_enabled: bool = True

    # Memory & Learning
    memory_path: str = "memory.jsonl"
    memory_index_path: str = "memory.index.json"
    policy_path: str = "policy.json"
    replay_capacity: int = 5000
    epsilon_start: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    reward_decay: float = 0.995
    actions: List[str] = field(default_factory=lambda: [
        "idle",
        "track_soft",
        "track_hard",
        "reticle_cycle",
        "multi_lock",
        "xray_scout",
        "defer_and_log"
    ])

    # Codex Mutation & Swarm Sync
    codex_path: str = "codex.rules.json"
    swarm_sync_enabled: bool = True
    swarm_port: int = 47653
    swarm_peers: List[str] = field(default_factory=list)  # e.g., ["192.168.1.12","192.168.1.13"]
    codex_sync_interval_s: int = 8
    codex_mutation_interval_ticks: int = 120
    ghost_sync_threshold: float = 0.9  # anomaly threshold for ghost sync
    phantom_node_enabled: bool = False

# ---------- Telemetry & Audit ----------

class Telemetry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._q = queue.Queue(maxsize=self.cfg.telemetry_retention_max)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._enabled = self.cfg.telemetry_enabled
        if self._enabled:
            self._thread.start()

    def set_retention(self, maxsize: int):
        # Rebuild queue with new maxsize, keeping recent items
        items = []
        while not self._q.empty():
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        self._q = queue.Queue(maxsize=maxsize)
        for pkt in items[-maxsize:]:
            try:
                self._q.put_nowait(pkt)
            except queue.Full:
                break

    def emit(self, label: str, data: Dict):
        if not self._enabled:
            return
        packet = {"ts": time.time(), "label": label, "data": data}
        try:
            self._q.put_nowait(packet)
        except queue.Full:
            # Drop oldest to prioritize recency
            try:
                _ = self._q.get_nowait()
                self._q.put_nowait(packet)
            except queue.Empty:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                pkt = self._q.get(timeout=0.25)
                print(f"[TEL] {pkt['label']}: {pkt['data']}")
            except queue.Empty:
                pass

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

class AuditLog:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._f = open(self.cfg.audit_log_path, "a", encoding="utf-8")

    def write(self, event: str, detail: Dict):
        record = {"ts": time.time(), "event": event, "detail": detail}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# ---------- Manifest & Integrity ----------

def compute_manifest_hash(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except FileNotFoundError:
            h.update(f"missing:{p}".encode("utf-8"))
    return h.hexdigest()

def verify_manifest(cfg: Config, paths: List[str], audit: AuditLog) -> bool:
    current = compute_manifest_hash(paths)
    try:
        with open(cfg.manifest_path, "r", encoding="utf-8") as f:
            expected = f.read().strip()
    except FileNotFoundError:
        audit.write("manifest_missing", {"path": cfg.manifest_path, "current": current})
        return False
    ok = (current == expected)
    audit.write("manifest_verify", {"ok": ok, "expected": expected, "current": current})
    return ok

# ---------- Ethics & Consent ----------

class EthicsGuard:
    def __init__(self, cfg: Config, audit: AuditLog):
        self.cfg = cfg
        self.audit = audit
        self.policy = {"allow_actions": ["idle","track_soft","reticle_cycle","defer_and_log"],
                       "deny_actions": ["covert_data_collection","track_hard","xray_scout","multi_lock"]}
        self._load_policy()

    def _load_policy(self):
        try:
            with open(self.cfg.ethics_policy_path, "r", encoding="utf-8") as f:
                self.policy = json.load(f)
        except FileNotFoundError:
            self.audit.write("ethics_policy_default_used", {"path": self.cfg.ethics_policy_path})

    def check(self, action: str, context: Dict) -> bool:
        if action in self.policy.get("deny_actions", []):
            self.audit.write("ethics_denied", {"action": action, "context": context})
            return False
        if action not in self.policy.get("allow_actions", []):
            self.audit.write("ethics_unlisted_action", {"action": action, "context": context})
            return False
        self.audit.write("ethics_allowed", {"action": action, "context": context})
        return True

    def require_consent(self, consent_flag: bool) -> bool:
        ok = (consent_flag or not self.cfg.consent_required)
        self.audit.write("consent_check", {"required": self.cfg.consent_required, "ok": ok})
        return ok

# ---------- Device/Resource Watchdog ----------

class Watchdog:
    def __init__(self, cfg: Config, tel: Telemetry, audit: AuditLog):
        self.cfg = cfg
        self.tel = tel
        self.audit = audit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.cfg.enable_watchdog:
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            # Use psutil if available
            mem_ok = True
            gpu = "unknown"
            cam = "ok"
            if psutil:
                vmem = psutil.virtual_memory()
                mem_ok = vmem.percent < 85
            status = {"camera": cam, "gpu": gpu, "memory_ok": mem_ok, "autoload": AUTOLOAD_STATUS}
            self.tel.emit("watchdog", status)
            self.audit.write("watchdog_tick", status)
            time.sleep(self.cfg.device_check_interval_s)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

# ---------- Memory (Persistent, Append-only) ----------

class MemoryStore:
    def __init__(self, cfg: Config, audit: AuditLog):
        self.cfg = cfg
        self.audit = audit
        os.makedirs(os.path.dirname(self.cfg.memory_path) or ".", exist_ok=True)
        self._f = open(self.cfg.memory_path, "a", encoding="utf-8")
        self._index = {"count": 0, "labels": {}, "last_hash": ""}  # simple index
        self._load_index()

    def _load_index(self):
        try:
            with open(self.cfg.memory_index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except FileNotFoundError:
            self._persist_index()

    def _persist_index(self):
        with open(self.cfg.memory_index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f)

    def append(self, episode: Dict):
        payload = json.dumps(episode, sort_keys=True)
        ep_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        record = {"ts": time.time(), "hash": ep_hash, "episode": episode}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        # Update index
        label = episode.get("label", "generic")
        self._index["count"] += 1
        self._index["labels"][label] = self._index["labels"].get(label, 0) + 1
        self._index["last_hash"] = ep_hash
        self._persist_index()
        self.audit.write("memory_append", {"hash": ep_hash, "label": label})

    def query_recent(self, label: Optional[str] = None, limit: int = 100) -> List[Dict]:
        result = []
        try:
            with open(self.cfg.memory_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            lines = lines[-limit:]
            for line in reversed(lines):
                rec = json.loads(line)
                ep = rec["episode"]
                if label is None or ep.get("label") == label:
                    result.append(ep)
        except FileNotFoundError:
            pass
        return list(reversed(result))

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# ---------- Learning (Contextual Bandit) ----------

@dataclass
class PolicyState:
    q_values: Dict[str, float] = field(default_factory=dict)  # action -> value
    epsilon: float = 0.2
    reward_ma: float = 0.0

class PolicyLearner:
    def __init__(self, cfg: Config, audit: AuditLog):
        self.cfg = cfg
        self.audit = audit
        self.state = PolicyState(epsilon=self.cfg.epsilon_start)
        self._load()
        for a in self.cfg.actions:
            self.state.q_values.setdefault(a, 0.0)

    def _load(self):
        try:
            with open(self.cfg.policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.state = PolicyState(q_values=data.get("q_values", {}),
                                         epsilon=data.get("epsilon", self.cfg.epsilon_start),
                                         reward_ma=data.get("reward_ma", 0.0))
        except FileNotFoundError:
            self._persist()

    def _persist(self):
        with open(self.cfg.policy_path, "w", encoding="utf-8") as f:
            json.dump({
                "q_values": self.state.q_values,
                "epsilon": self.state.epsilon,
                "reward_ma": self.state.reward_ma
            }, f)

    def select_action(self, context: Dict) -> str:
        if random.random() < self.state.epsilon:
            a = random.choice(self.cfg.actions)
            self.audit.write("policy_explore", {"action": a, "epsilon": self.state.epsilon})
        else:
            a = max(self.state.q_values.items(), key=lambda kv: kv[1])[0]
            self.audit.write("policy_exploit", {"action": a, "q": self.state.q_values[a]})
        self.state.epsilon = max(self.cfg.epsilon_min, self.state.epsilon * self.cfg.epsilon_decay)
        self._persist()
        return a

    def update(self, action: str, reward: float):
        old_q = self.state.q_values.get(action, 0.0)
        lr = 0.1
        new_q = old_q + lr * (reward - old_q)
        self.state.q_values[action] = new_q
        self.state.reward_ma = self.cfg.reward_decay * self.state.reward_ma + (1 - self.cfg.reward_decay) * reward
        self.audit.write("policy_update", {"action": action, "old_q": old_q, "new_q": new_q, "reward": reward, "reward_ma": self.state.reward_ma})
        self._persist()

# ---------- Motion Detection (Predictable Interface) ----------

class MotionDetector:
    def __init__(self, tel: Telemetry, audit: AuditLog, cfg: Config):
        self.tel = tel
        self.audit = audit
        self.cfg = cfg
        self.phantom_node = self.cfg.phantom_node_enabled

    def set_phantom(self, enabled: bool):
        self.phantom_node = enabled
        self.tel.emit("phantom_node_set", {"enabled": enabled})
        self.audit.write("phantom_node_set", {"enabled": enabled})

    def detect_motion(self, frame) -> Tuple[List[Dict], Dict]:
        # Always return (events, stats)
        # Placeholder predictive frame generation and phantom blending:
        events = []
        stats = {
            "frame_time_ms": 12.0,
            "frame_drop_rate": 0.01,
            "frame_jitter": 0.05
        }
        # If phantom node enabled, synthesize a low-confidence "phantom" event:
        if self.phantom_node:
            events.append({"id": "phantom", "bbox": [0,0,0,0], "label": "unknown", "confidence": 0.05, "phantom": True})
            stats["frame_jitter"] = min(1.0, stats["frame_jitter"] + 0.02)

        self.tel.emit("detect_motion", {"count": len(events), "stats": stats})
        self.audit.write("detect_motion", {"count": len(events), "stats": stats})
        return events, stats

# ---------- Anomaly Detection ----------

@dataclass
class AnomalyStats:
    recent_scores: List[float] = field(default_factory=list)
    threshold: float = 0.85
    window: int = 30

class AnomalyDetector:
    def __init__(self, cfg: Config, tel: Telemetry, audit: AuditLog):
        self.cfg = cfg
        self.tel = tel
        self.audit = audit
        self.stats = AnomalyStats(threshold=self.cfg.ghost_sync_threshold)

    def score(self, events: List[Dict], stats: Dict) -> float:
        jitter = stats.get("frame_jitter", 0.0)
        drop = stats.get("frame_drop_rate", 0.0)
        unknown = sum(1 for e in events if e.get("label") == "unknown")
        s = min(1.0, 0.5 * jitter + 0.4 * drop + 0.1 * (unknown / max(1, len(events))))
        return s

    def update(self, events: List[Dict], stats: Dict) -> Optional[float]:
        if not self.cfg.enable_anomaly_detection:
            return None
        s = self.score(events, stats)
        self.stats.recent_scores.append(s)
        if len(self.stats.recent_scores) > self.stats.window:
            self.stats.recent_scores.pop(0)
        if s >= self.stats.threshold:
            self.tel.emit("anomaly", {"score": s})
            self.audit.write("anomaly_trigger", {"score": s, "stats": stats})
        return s

# ---------- Dashboard (Placeholder) ----------

class Dashboard:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def update(self, state: Dict):
        if not self.cfg.dashboard_enabled:
            return
        print(f"[DASH] {state.get('status','idle')} | events={state.get('event_count',0)} | action={state.get('action')} | reward={state.get('reward')} | anomaly={state.get('anomaly_score')} | ghost={state.get('ghost_sync')}")

# ---------- Codex (Adaptive Mutation & Swarm Sync) ----------

@dataclass
class CodexState:
    version: int = 1
    rules: Dict[str, Dict] = field(default_factory=lambda: {
        "purge_logic": {
            "retain_episodes": 5000,
            "retain_label_priority": {"tick": 0.5, "anomaly": 0.9, "ethics": 0.8}
        },
        "predictive_patterns": {
            "jitter_spike_threshold": 0.15,
            "drop_spike_threshold": 0.12
        }
    })
    last_mutation_ts: float = 0.0

class CodexManager:
    def __init__(self, cfg: Config, audit: AuditLog, tel: Telemetry):
        self.cfg = cfg
        self.audit = audit
        self.tel = tel
        self.state = CodexState()
        self._load()
        self._stop = threading.Event()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True) if self.cfg.swarm_sync_enabled else None

    def _load(self):
        try:
            with open(self.cfg.codex_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.state = CodexState(version=data.get("version", 1),
                                        rules=data.get("rules", self.state.rules),
                                        last_mutation_ts=data.get("last_mutation_ts", 0.0))
                self.audit.write("codex_loaded", {"version": self.state.version})
        except FileNotFoundError:
            self._persist()
            self.audit.write("codex_default_created", {"version": self.state.version})

    def _persist(self):
        with open(self.cfg.codex_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": self.state.version,
                "rules": self.state.rules,
                "last_mutation_ts": self.state.last_mutation_ts
            }, f)

    def mutate_from_patterns(self, recent_scores: List[float], recent_stats: List[Dict]):
        # Derive trends and rewrite purge logic
        if not recent_stats:
            return
        avg_jitter = sum(s.get("frame_jitter", 0.0) for s in recent_stats) / len(recent_stats)
        avg_drop = sum(s.get("frame_drop_rate", 0.0) for s in recent_stats) / len(recent_stats)
        jitter_thr = self.state.rules["predictive_patterns"]["jitter_spike_threshold"]
        drop_thr = self.state.rules["predictive_patterns"]["drop_spike_threshold"]

        # Mutation policy: if average exceeds thresholds, reduce retention and prioritize anomaly episodes
        mutated = False
        if avg_jitter > jitter_thr or avg_drop > drop_thr:
            retain = self.state.rules["purge_logic"]["retain_episodes"]
            new_retain = max(1000, int(retain * 0.8))
            self.state.rules["purge_logic"]["retain_episodes"] = new_retain
            self.state.rules["purge_logic"]["retain_label_priority"]["anomaly"] = min(1.0, 0.95)
            mutated = True

        if mutated:
            self.state.version += 1
            self.state.last_mutation_ts = time.time()
            self._persist()
            self.tel.emit("codex_mutation", {"version": self.state.version, "retain": self.state.rules["purge_logic"]["retain_episodes"]})
            self.audit.write("codex_mutation", {
                "version": self.state.version,
                "avg_jitter": avg_jitter,
                "avg_drop": avg_drop,
                "rules": self.state.rules["purge_logic"]
            })

    def start_sync(self):
        if self._sync_thread:
            self._sync_thread.start()

    def stop_sync(self):
        self._stop.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)

    def _sync_loop(self):
        # UDP broadcast/peer sync (simple, local-network-friendly)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)
        addr = ('<broadcast>', self.cfg.swarm_port)
        while not self._stop.is_set():
            try:
                packet = json.dumps({
                    "type": "codex_sync",
                    "version": self.state.version,
                    "rules": self.state.rules
                }).encode("utf-8")
                sock.sendto(packet, addr)
                # Listen briefly for peer updates
                try:
                    data, _ = sock.recvfrom(65535)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") == "codex_sync":
                        peer_ver = msg.get("version", 0)
                        if peer_ver > self.state.version:
                            self.state.version = peer_ver
                            self.state.rules = msg.get("rules", self.state.rules)
                            self._persist()
                            self.tel.emit("codex_sync_received", {"version": peer_ver})
                            self.audit.write("codex_sync_received", {"peer_version": peer_ver})
                except socket.timeout:
                    pass
            except Exception as e:
                # Log and continue
                self.audit.write("codex_sync_error", {"error": str(e)})
            time.sleep(self.cfg.codex_sync_interval_s)

# ---------- Purge Manager ----------

class PurgeManager:
    def __init__(self, cfg: Config, memory: MemoryStore, audit: AuditLog, tel: Telemetry, codex: CodexManager):
        self.cfg = cfg
        self.memory = memory
        self.audit = audit
        self.tel = tel
        self.codex = codex

    def apply_purge(self):
        # Apply purge based on codex rules (retain window only; append-only file)
        rules = self.codex.state.rules.get("purge_logic", {})
        retain_n = int(rules.get("retain_episodes", self.cfg.replay_capacity))
        try:
            with open(self.cfg.memory_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= retain_n:
                return
            # Keep most recent retain_n
            new_lines = lines[-retain_n:]
            with open(self.cfg.memory_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            self.audit.write("purge_apply", {"kept": retain_n, "dropped": len(lines) - retain_n})
            self.tel.emit("purge", {"kept": retain_n, "dropped": len(lines) - retain_n})
        except FileNotFoundError:
            pass

# ---------- Autonomy Planner ----------

@dataclass
class Task:
    name: str
    priority: int
    guard: Dict
    action: str

class AutonomyPlanner:
    def __init__(self, cfg: Config, audit: AuditLog):
        self.cfg = cfg
        self.audit = audit
        self.tasks: List[Task] = [
            Task(name="safe_idle", priority=10, guard={"event_count_min": 0}, action="idle"),
            Task(name="soft_track_when_events", priority=50, guard={"event_count_min": 1}, action="track_soft"),
            Task(name="reticle_cycle_for_variation", priority=40, guard={"event_count_min": 1}, action="reticle_cycle"),
            Task(name="defer_on_anomaly", priority=100, guard={"anomaly_min": self.cfg.ghost_sync_threshold}, action="defer_and_log"),
        ]

    def propose(self, context: Dict) -> str:
        def guard_ok(task: Task) -> bool:
            ec = context.get("event_count", 0)
            an = context.get("anomaly", 0.0)
            if "event_count_min" in task.guard and ec < task.guard["event_count_min"]:
                return False
            if "anomaly_min" in task.guard and an < task.guard["anomaly_min"]:
                return False
            return True

        valid = [t for t in self.tasks if guard_ok(t)]
        if not valid:
            self.audit.write("planner_default", {"context": context})
            return "idle"
        best = max(valid, key=lambda t: t.priority)
        self.audit.write("planner_select", {"task": best.name, "action": best.action, "context": context})
        return best.action

# ---------- Engine ----------

class SafeAutonomousCodexEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tel = Telemetry(cfg)
        self.audit = AuditLog(cfg)
        self.ethics = EthicsGuard(cfg, self.audit)
        self.watchdog = Watchdog(cfg, self.tel, self.audit)
        self.memory = MemoryStore(cfg, self.audit)
        self.learner = PolicyLearner(cfg, self.audit)
        self.detector = MotionDetector(self.tel, self.audit, cfg)
        self.anomaly = AnomalyDetector(cfg, self.tel, self.audit)
        self.dashboard = Dashboard(cfg)
        self.planner = AutonomyPlanner(cfg, self.audit)
        self.codex = CodexManager(cfg, self.audit, self.tel)
        self.purge = PurgeManager(cfg, self.memory, self.audit, self.tel, self.codex)
        self._running = False
        self._backoff_ms = 0
        self._tick_count = 0
        self._recent_stats_buffer: List[Dict] = []
        self.codex.start_sync()

    def start(self, consent_flag: bool):
        if not self.ethics.require_consent(consent_flag):
            print("Consent not granted; engine will not start.")
            self.audit.write("engine_refused_no_consent", {})
            return
        self._running = True
        self.watchdog.start()
        self.audit.write("engine_start", {"tick_hz": self.cfg.tick_hz})
        self._loop()

    def _compute_reward(self, events: List[Dict], stats: Dict, action: str, anomaly_score: Optional[float]) -> float:
        stable = max(0.0, 1.0 - stats.get("frame_drop_rate", 0.0)) * max(0.0, 1.0 - stats.get("frame_jitter", 0.0))
        has_events = 1.0 if len(events) > 0 else 0.0
        action_bonus = {
            "idle": 0.0,
            "track_soft": 0.2 * has_events,
            "reticle_cycle": 0.1 * has_events,
            "defer_and_log": 0.05,
            "multi_lock": 0.3 * has_events,
            "track_hard": 0.35 * has_events,
            "xray_scout": 0.15
        }.get(action, 0.0)
        anomaly_penalty = 0.5 * (anomaly_score or 0.0)
        reward = stable + action_bonus - anomaly_penalty
        return max(-1.0, min(1.0, reward))

    def _execute_action(self, action: str, events: List[Dict], stats: Dict):
        ctx = {"event_count": len(events), "stats": stats}
        if not self.ethics.check(action, ctx):
            self.audit.write("action_blocked_ethics", {"action": action})
            return False
        self.tel.emit("action", {"action": action, "event_count": len(events)})
        self.audit.write("action_execute", {"action": action, "event_count": len(events)})
        return True

    def _ghost_sync_response(self):
        # Shorten telemetry retention and enable phantom node in detector
        new_retention = max(128, int(self.cfg.telemetry_retention_max * 0.5))
        self.tel.set_retention(new_retention)
        self.detector.set_phantom(True)
        self.audit.write("ghost_sync_response", {"telemetry_retention": new_retention, "phantom_node_enabled": True})
        self.tel.emit("ghost_sync", {"telemetry_retention": new_retention, "phantom_node_enabled": True})

    def _loop(self):
        tick_dt = 1.0 / max(1, self.cfg.tick_hz)
        while self._running:
            t0 = time.time()
            try:
                frame = None  # integrate camera grab if available
                events, stats = self.detector.detect_motion(frame)
                a_score = self.anomaly.update(events, stats)
                ghost_detected = (a_score or 0.0) >= self.cfg.ghost_sync_threshold

                # Buffer stats for codex mutation
                self._recent_stats_buffer.append(stats)
                if len(self._recent_stats_buffer) > 64:
                    self._recent_stats_buffer.pop(0)

                # Ghost sync response
                if ghost_detected:
                    self._ghost_sync_response()

                context = {"event_count": len(events), "anomaly": a_score or 0.0}
                plan_action = self.planner.propose(context)
                policy_action = self.learner.select_action(context)
                chosen = policy_action if self.ethics.check(policy_action, context) else plan_action

                ok = self._execute_action(chosen, events, stats)
                reward = self._compute_reward(events, stats, chosen, a_score) if ok else -0.2

                # Learning update
                self.learner.update(chosen, reward)

                # Memory episode
                episode = {
                    "label": "tick",
                    "events_count": len(events),
                    "stats": stats,
                    "anomaly": a_score,
                    "action": chosen,
                    "reward": reward,
                    "ghost_sync": ghost_detected
                }
                self.memory.append(episode)

                # Dashboard
                self.dashboard.update({
                    "status": "running",
                    "event_count": len(events),
                    "action": chosen,
                    "reward": reward,
                    "anomaly_score": a_score,
                    "ghost_sync": ghost_detected
                })

                # Periodic codex mutation based on predictive patterns
                self._tick_count += 1
                if self._tick_count % self.cfg.codex_mutation_interval_ticks == 0:
                    self.codex.mutate_from_patterns(
                        recent_scores=[],
                        recent_stats=self._recent_stats_buffer
                    )
                    # Apply purge according to updated codex
                    self.purge.apply_purge()

                self._backoff_ms = 0
            except RecoverableError as e:
                self.audit.write("recoverable_error", {"error": str(e)})
                self._backoff_ms = min(self.cfg.max_backoff_ms, max(100, self._backoff_ms * 2 or 100))
                self.tel.emit("backoff", {"ms": self._backoff_ms})
                time.sleep(self._backoff_ms / 1000.0)
            except Exception as e:
                self.audit.write("fatal_error", {"error": str(e)})
                self._backoff_ms = min(self.cfg.max_backoff_ms, max(250, self._backoff_ms * 2 or 250))
                self.tel.emit("error", {"message": str(e), "backoff_ms": self._backoff_ms})
                time.sleep(self._backoff_ms / 1000.0)

            t1 = time.time()
            sleep_left = tick_dt - (t1 - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    def stop(self):
        if not self._running:
            return
        self._running = False
        self.watchdog.stop()
        self.codex.stop_sync()
        self.tel.stop()
        self.memory.close()
        self.audit.write("engine_stop", {})
        self.audit.close()
        print("Engine stopped cleanly.")

# ---------- Errors ----------

class RecoverableError(Exception):
    pass

# ---------- Graceful Shutdown ----------

_engine_instance: Optional[SafeAutonomousCodexEngine] = None

def _signal_handler(signum, frame):
    global _engine_instance
    if _engine_instance:
        _engine_instance.stop()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------- Main ----------

def main():
    cfg = Config()
    audit = AuditLog(cfg)
    manifest_ok = verify_manifest(cfg, paths=["autonomous_codex_engine.py", cfg.ethics_policy_path, cfg.codex_path], audit=audit)
    if not manifest_ok:
        print("Manifest verification failed; proceed cautiously.")
    audit.close()

    global _engine_instance
    _engine_instance = SafeAutonomousCodexEngine(cfg)
    _engine_instance.start(consent_flag=True)

if __name__ == "__main__":
    main()

