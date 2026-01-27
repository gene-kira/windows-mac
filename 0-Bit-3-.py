# ============================
# PART 1 – CORE + BORG SPINE
# ============================

import importlib
import subprocess
import sys
import platform
import time
import json
import random
import os
import threading
import queue
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

# ---------- Universal autoloader ----------

class AutoLoader:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.os = platform.system().lower()

    def install(self, package: str):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            try:
                subprocess.call(
                    [sys.executable, "-m", "ensurepip"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                subprocess.call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

    def load(self, package: str, import_name: str = None):
        name = import_name or package
        if name in self.cache:
            return self.cache[name]
        try:
            module = importlib.import_module(name)
        except ImportError:
            self.install(package)
            module = importlib.import_module(name)
        self.cache[name] = module
        return module


autoload = AutoLoader()

tk = autoload.load("tkinter", "tkinter")
ttk = autoload.load("tkinter.ttk", "tkinter.ttk")
psutil = autoload.load("psutil")
try:
    np = autoload.load("numpy")
except Exception:
    np = None

# ---------- Telemetry ----------

class Telemetry:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, event_type: str, payload: Dict[str, Any]):
        self.events.append({
            "t": time.time(),
            "type": event_type,
            "payload": payload,
        })


telemetry = Telemetry()

# ============================
# Hybrid Brain – core state
# ============================

META_STATES = ["HYPER_FLOW", "DEEP_DREAM", "SENTINEL", "RECOVERY_FLOW"]
MISSIONS = ["AUTO", "PROTECT", "STABILITY", "LEARN", "OPTIMIZE"]
ENV_STATES = ["CALM", "TENSE", "DANGER"]

@dataclass
class HorizonPrediction:
    horizon: float
    value: float
    confidence: float
    engines: Dict[str, float] = field(default_factory=dict)

@dataclass
class PatternFingerprint:
    pattern_id: str
    label: str
    count: int = 0

@dataclass
class ReinforcementMemory:
    good: int = 0
    bad: int = 0
    notes: List[str] = field(default_factory=list)

@dataclass
class HybridBrainState:
    meta_state: str = "SENTINEL"
    stance: str = "BALANCED"
    volatility: float = 0.0
    trust: float = 0.5
    cognitive_load: float = 0.0
    mission: str = "AUTO"
    environment: str = "CALM"
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    anomaly_risk: float = 0.0
    drive_risk: float = 0.0
    hive_risk: float = 0.0
    collective_health: float = 1.0
    health_trend: float = 0.0
    prediction_horizon: str = "MEDIUM"
    meta_confidence: float = 0.5
    bias_drift: float = 0.0
    judgment_confidence: float = 0.5
    sample_count: int = 0
    meta_state_momentum: float = 0.0
    mesh_risk: float = 0.0
    mesh_hostile_ratio: float = 0.0
    mesh_entropy: float = 0.0

@dataclass
class AutoTuningState:
    appetite: float = 1.0
    threshold_scale: float = 1.0
    horizon_bias: float = 0.0
    dampening_scale: float = 1.0
    cache_aggressiveness: float = 1.0
    thread_expansion: float = 1.0
    last_calibration_ts: float = 0.0

class HybridBrain:
    def __init__(self):
        self.state = HybridBrainState()
        self.history: List[float] = []
        self.horizons = [1.0, 5.0, 30.0, 120.0]
        self.reinforcement = ReinforcementMemory()
        self.patterns: Dict[str, PatternFingerprint] = {}
        self.baseline_risk: float = 0.2
        self.reasoning_tail: List[str] = []
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self.auto = AutoTuningState()
        self.best_guess: float = 0.0
        self.best_guess_conf: float = 0.0
        self.last_calibration_day: Optional[int] = None
        self.last_meta_state: str = self.state.meta_state

    # ---------- core ingest ----------

    def ingest_metrics(self, cpu: float, mem: float, disk: float, net: float) -> None:
        risk = (cpu * 0.4 + mem * 0.3 + disk * 0.2 + net * 0.1) / 100.0
        self.history.append(risk)
        if len(self.history) > 600:
            self.history.pop(0)
        self.state.cognitive_load = min(1.0, (cpu + mem) / 200.0)
        self.state.risk_score = risk
        self._update_volatility()
        self._update_meta_state_with_momentum()
        self._update_trust()
        self._update_collective_stub()
        self._auto_calibration()
        self._append_reason(f"Ingest metrics: cpu={cpu:.1f}, mem={mem:.1f}, disk={disk:.1f}, net={net:.1f}, risk={risk:.3f}")

    def _update_volatility(self):
        if len(self.history) < 5:
            self.state.volatility = 0.0
            return
        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        self.state.volatility = min(1.0, var * 50.0)

    def _meta_state_target(self) -> str:
        v = self.state.volatility
        r = self.state.risk_score + self.state.mesh_risk * 0.4
        if r < 0.3 and v < 0.2:
            return "DEEP_DREAM"
        elif r < 0.5 and v < 0.5:
            return "HYPER_FLOW"
        elif r < 0.7:
            return "SENTINEL"
        else:
            return "RECOVERY_FLOW"

    def _update_meta_state_with_momentum(self):
        target = self._meta_state_target()
        if target == self.state.meta_state:
            self.state.meta_state_momentum = min(1.0, self.state.meta_state_momentum + 0.05)
            return
        can_switch = False
        if self.state.meta_state == "HYPER_FLOW" and target == "SENTINEL":
            can_switch = self.state.volatility < 0.4 and self.state.mesh_risk < 0.6
        elif self.state.meta_state == "SENTINEL" and target == "HYPER_FLOW":
            can_switch = self.state.risk_score < 0.5 and self.state.mesh_hostile_ratio < 0.3
        elif self.state.meta_state == "RECOVERY_FLOW" and target == "DEEP_DREAM":
            can_switch = self.state.volatility < 0.2 and self.state.mesh_risk < 0.4
        else:
            can_switch = True

        if can_switch:
            self.state.meta_state = target
            self.state.meta_state_momentum = 0.0
            self._append_reason(f"Meta-state transition: {self.last_meta_state} -> {target}")
            self.last_meta_state = target
        else:
            self.state.meta_state_momentum = max(0.0, self.state.meta_state_momentum - 0.05)

    def _update_trust(self):
        self.state.trust = max(
            0.0,
            min(
                1.0,
                0.7 - self.state.volatility * 0.4
                + (1.0 - abs(self.state.risk_score - self.baseline_risk)) * 0.3
                - self.state.mesh_hostile_ratio * 0.3
            ),
        )

    def _update_collective_stub(self):
        mesh_factor = 1.0 - self.state.mesh_risk
        self.state.collective_health = max(0.0, (1.0 - self.state.risk_score + mesh_factor) / 2.0)
        if len(self.history) > 1:
            self.state.health_trend = (self.history[-1] - self.history[0]) / max(1, len(self.history))

    # ---------- regime + engines ----------

    def _regime(self) -> str:
        v = self.state.volatility
        if len(self.history) < 5:
            return "UNKNOWN"
        trend = self.history[-1] - self.history[0]
        if v < 0.1:
            return "LOW_VAR"
        if v > 0.5:
            return "HIGH_VAR"
        if trend > 0.1:
            return "RISING"
        if trend < -0.1:
            return "COOLING"
        return "NORMAL"

    def _ewma(self, alpha: float) -> float:
        if not self.history:
            return self.baseline_risk
        value = self.history[0]
        for x in self.history[1:]:
            value = alpha * x + (1 - alpha) * value
        return value

    def _trend_engine(self) -> float:
        if len(self.history) < 2:
            return self.baseline_risk
        return max(0.0, min(1.0, self.history[-1] + (self.history[-1] - self.history[0]) * 0.5))

    def _variance_engine(self) -> float:
        if len(self.history) < 5:
            return self.baseline_risk
        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        return max(0.0, min(1.0, self.state.risk_score + var * 0.3))

    def _turbulence_engine(self) -> float:
        return max(0.0, min(1.0, self.state.volatility))

    def _model_stub(self) -> float:
        if not self.history:
            return self.baseline_risk
        recent = self.history[-1]
        return max(0.0, min(1.0, recent + self.state.volatility * 0.2))

    def _meta_confidence_components(self) -> Tuple[float, float, float, float, float]:
        if len(self.history) < 5:
            return 0.3, 0.0, 0.0, 0.5, 0.5
        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        trend = self.history[-1] - self.history[0]
        turbulence = min(1.0, var * 50.0)
        sensor_noise = 0.1
        reinforcement_factor = 0.5 + 0.5 * (self.reinforcement.good - self.reinforcement.bad) / max(
            1, self.reinforcement.good + self.reinforcement.bad
        )
        return var, trend, turbulence, sensor_noise, reinforcement_factor

    # ---------- auto-tuning / calibration ----------

    def _auto_calibration(self):
        now = time.time()
        day = int(now // 86400)
        if self.last_calibration_day is None:
            self.last_calibration_day = day
            self.auto.last_calibration_ts = now
            return
        if day != self.last_calibration_day:
            self.last_calibration_day = day
            if self.history:
                self.baseline_risk = sum(self.history) / len(self.history)
            total = max(1, self.reinforcement.good + self.reinforcement.bad)
            bias = (self.reinforcement.good - self.reinforcement.bad) / total
            self.auto.threshold_scale = 1.0 - 0.2 * bias
            self.auto.dampening_scale = 1.0 + 0.2 * (1.0 - self.state.trust)
            self.auto.horizon_bias = 0.1 * bias
            self.auto.cache_aggressiveness = 1.0 + 0.1 * bias
            self.auto.thread_expansion = 1.0 + 0.1 * (1.0 - self.state.risk_score)
            self._append_reason("Auto-calibration: baselines and thresholds recomputed")

    # ---------- reinforcement / patterns ----------

    def record_pattern(self, pattern_id: str, label: str):
        fp = self.patterns.get(pattern_id)
        if not fp:
            fp = PatternFingerprint(pattern_id=pattern_id, label=label, count=0)
            self.patterns[pattern_id] = fp
        fp.count += 1
        self._append_reason(f"Pattern {pattern_id} -> {label}, count={fp.count}")

    def reinforce(self, good: bool, note: str = ""):
        if good:
            self.reinforcement.good += 1
        else:
            self.reinforcement.bad += 1
        if note:
            self.reinforcement.notes.append(note)
        self.state.sample_count += 1
        total = max(1, self.reinforcement.good + self.reinforcement.bad)
        self.state.judgment_confidence = self.reinforcement.good / total
        self.state.bias_drift = (self.reinforcement.good - self.reinforcement.bad) / total
        self._append_reason(f"Reinforce: good={good}, note={note}")

    def reset_bias(self):
        self.reinforcement = ReinforcementMemory()
        self.state.bias_drift = 0.0
        self.state.judgment_confidence = 0.5
        self._append_reason("Bias reset")

    # ---------- stance / mission / env ----------

    def set_stance(self, stance: str):
        self.state.stance = stance
        self._append_reason(f"Stance set to {stance}")

    def set_mission(self, mission: str):
        if mission in MISSIONS:
            self.state.mission = mission
            self._append_reason(f"Mission set to {mission}")

    def set_environment(self, env: str):
        if env in ENV_STATES:
            self.state.environment = env
            self._append_reason(f"Environment set to {env}")

    def set_prediction_horizon_mode(self, mode: str):
        if mode in ("SHORT", "MEDIUM", "LONG"):
            self.state.prediction_horizon = mode
            self._append_reason(f"Prediction horizon mode set to {mode}")

    def override_meta_state(self, meta_state: str):
        if meta_state in META_STATES:
            self.state.meta_state = meta_state
            self._append_reason(f"Meta-state override -> {meta_state}")

    # ---------- snapshots ----------

    def snapshot(self):
        self.last_snapshot = {
            "state": asdict(self.state),
            "reinforcement": asdict(self.reinforcement),
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
            "baseline_risk": self.baseline_risk,
            "auto": asdict(self.auto),
            "best_guess": self.best_guess,
            "best_guess_conf": self.best_guess_conf,
        }
        self._append_reason("Snapshot taken")
        return self.last_snapshot

    def rollback(self):
        if not self.last_snapshot:
            return
        s = self.last_snapshot
        self.state = HybridBrainState(**s["state"])
        self.reinforcement = ReinforcementMemory(**s["reinforcement"])
        self.patterns = {k: PatternFingerprint(**v) for k, v in s["patterns"].items()}
        self.baseline_risk = s["baseline_risk"]
        self.auto = AutoTuningState(**s["auto"])
        self.best_guess = s["best_guess"]
        self.best_guess_conf = s["best_guess_conf"]
        self._append_reason("Rollback to last snapshot")

    def export_json(self) -> str:
        snap = self.snapshot()
        return json.dumps(snap, indent=2)

    def _append_reason(self, msg: str):
        ts = time.strftime("%H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        self.reasoning_tail.append(line)
        if len(self.reasoning_tail) > 200:
            self.reasoning_tail.pop(0)

# ============================
# Borg Mesh Spine
# ============================

BORG_MESH_CONFIG = {
    "max_corridors": 10000,
    "unknown_bias": 0.4,
}

def privacy_filter(snippet: str):
    return snippet, 0

class DummyGuardian:
    def disassemble(self, snippet: str):
        entropy = min(1.0, max(0.0, len(snippet) / 1024.0))
        return {"entropy": entropy, "pattern_flags": []}

    def _pii_count(self, snippet: str) -> int:
        return 0

    def reassemble(self, url: str, snippet: str, raw_pii_hits: int = 0):
        return {"status": "SAFE_FOR_TRAVEL"}

class DummyMemory:
    def __init__(self):
        self.mesh_events: List[Dict[str, Any]] = []

    def record_mesh_event(self, evt: Dict[str, Any]):
        self.mesh_events.append(evt)

class DummyComms:
    def send_secure(self, channel: str, msg: str, profile: str):
        telemetry.log("BORG_COMMS", {"channel": channel, "msg": msg, "profile": profile})

class BorgMesh:
    def __init__(self, memory: DummyMemory, comms: DummyComms, guardian: DummyGuardian, brain: HybridBrain):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]
        self.brain = brain
        self.hostile_count = 0
        self.total_enforced = 0

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: List[str]):
        risk = self._risk(snippet)
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0, "status": "UNKNOWN"})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            if len(self.edges) < self.max_corridors:
                self.edges.add((url, l))
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "discover", "url": url, "risk": risk, "links": len(links)}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:discover", f"{url} risk={risk} links={len(links)}", "Default")
        self.brain.record_pattern("mesh:discover", "stability")
        telemetry.log("BORG_DISCOVER", evt)

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "build", "url": url}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        telemetry.log("BORG_BUILD", evt)
        return True

    def enforce(self, url: str, snippet: str):
        if url not in self.nodes:
            return False
        verdict = self.guardian.reassemble(url, privacy_filter(snippet or "")[0],
                                           raw_pii_hits=self.guardian._pii_count(snippet or ""))
        status = verdict.get("status", "HOSTILE")
        self.nodes[url]["state"] = "enforced"
        self.nodes[url]["status"] = status
        self.total_enforced += 1
        if status != "SAFE_FOR_TRAVEL":
            self.hostile_count += 1
            self.nodes[url]["risk"] = max(50, self.nodes[url].get("risk", 0))
            self.brain.record_pattern("mesh:hostile", "overload")
            self.brain.reinforce(False, "mesh hostile")
        else:
            self.nodes[url]["risk"] = 0
            self.brain.reinforce(True, "mesh safe")
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "enforce", "url": url, "status": status}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:enforce", f"{url} status={status}", "Default")
        telemetry.log("BORG_ENFORCE", evt)
        self._update_brain_mesh_view()
        return True

    def _update_brain_mesh_view(self):
        total = max(1, self.total_enforced)
        hostile_ratio = self.hostile_count / total
        avg_risk = 0.0
        if self.nodes:
            avg_risk = sum(n.get("risk", 0) for n in self.nodes.values()) / (len(self.nodes) * 100.0)
        entropy = min(1.0, len(self.edges) / max(1, self.max_corridors))
        self.brain.state.mesh_risk = avg_risk
        self.brain.state.mesh_hostile_ratio = hostile_ratio
        self.brain.state.mesh_entropy = entropy
        if hostile_ratio > 0.4 or avg_risk > 0.6:
            self.brain.set_environment("DANGER")
        elif hostile_ratio > 0.2:
            self.brain.set_environment("TENSE")
        else:
            self.brain.set_environment("CALM")

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        return {
            "total": total,
            "discovered": discovered,
            "built": built,
            "enforced": enforced,
            "corridors": len(self.edges),
            "hostile_ratio": self.brain.state.mesh_hostile_ratio,
            "mesh_risk": self.brain.state.mesh_risk,
        }

class BorgScanner(threading.Thread):
    def __init__(self, mesh: BorgMesh, in_events: queue.Queue, out_ops: queue.Queue, label="SCANNER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                ev = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            unseen_links = [l for l in ev["links"] if l not in self.mesh.nodes and random.random() < BORG_MESH_CONFIG["unknown_bias"]]
            self.mesh.discover(ev["url"], ev.get("snippet", ""), unseen_links or ev["links"])
            self.out_ops.put(("build", ev["url"]))
            time.sleep(random.uniform(0.2, 0.6))

class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if op == "build":
                if self.mesh.build(url):
                    self.ops_q.put(("enforce", url))
            elif op == "enforce":
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))

class BorgEnforcer(threading.Thread):
    def __init__(self, mesh: BorgMesh, label="ENFORCER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    self.mesh.enforce(url, snippet="")
            time.sleep(1.2)

# ============================
# PART 2 – PREDICTION + ORGANS
# ============================

class HybridBrain(HybridBrain):  # extend
    def predict_multi_horizon(self) -> List[HorizonPrediction]:
        var, trend, turbulence, noise, reinf = self._meta_confidence_components()
        regime = self._regime()
        preds: List[HorizonPrediction] = []

        for h in self.horizons:
            if regime == "LOW_VAR":
                alpha = 0.2
            elif regime == "HIGH_VAR":
                alpha = 0.6
            elif regime == "RISING":
                alpha = 0.5
            elif regime == "COOLING":
                alpha = 0.3
            else:
                alpha = 0.4

            ewma_val = self._ewma(alpha)
            trend_val = self._trend_engine()
            var_val = self._variance_engine()
            turb_val = self._turbulence_engine()
            model_val = self._model_stub()
            mesh_val = max(0.0, min(1.0, self.state.mesh_risk + self.state.mesh_hostile_ratio * 0.5))

            base_conf = max(0.1, 1.0 - turbulence - noise)
            model_conf = min(1.0, base_conf + reinf * 0.2)
            ewma_conf = 1.0 - model_conf

            horizon_factor = max(0.2, 1.0 - (h / 120.0) * (0.5 + turbulence))
            model_conf *= horizon_factor
            ewma_conf *= horizon_factor

            w_ewma = ewma_conf
            w_trend = 0.4 * horizon_factor
            w_var = 0.3 * horizon_factor
            w_turb = 0.2 * horizon_factor
            w_model = model_conf
            w_reinf = 0.3 * reinf
            w_mesh = 0.5 + self.state.mesh_hostile_ratio * 0.5

            engines = {
                "ewma": ewma_val,
                "trend": trend_val,
                "variance": var_val,
                "turbulence": turb_val,
                "model": model_val,
                "baseline": self.baseline_risk,
                "mesh": mesh_val,
            }

            total_w = w_ewma + w_trend + w_var + w_turb + w_model + w_reinf + w_mesh
            if total_w <= 0:
                blended = self.baseline_risk
            else:
                blended = (
                    ewma_val * w_ewma
                    + trend_val * w_trend
                    + var_val * w_var
                    + turb_val * w_turb
                    + model_val * w_model
                    + self.baseline_risk * w_reinf
                    + mesh_val * w_mesh
                ) / total_w

            total_conf = max(0.1, min(1.0, base_conf * horizon_factor))
            preds.append(HorizonPrediction(horizon=h, value=blended, confidence=total_conf, engines=engines))

        if preds:
            self.state.meta_confidence = sum(p.confidence for p in preds) / len(preds)
            self.state.anomaly_risk = preds[0].value
            self.state.drive_risk = preds[-1].value
            self.state.hive_risk = (self.state.anomaly_risk + self.state.drive_risk + self.state.mesh_risk) / 3.0

            weights = [1.0, 0.8, 0.5, 0.3]
            num = sum(p.value * w for p, w in zip(preds, weights))
            den = sum(weights)
            self.best_guess = num / den
            self.best_guess_conf = self.state.meta_confidence
        self._append_reason(
            f"Predict multi-horizon: regime={regime}, meta_conf={self.state.meta_confidence:.2f}, best_guess={self.best_guess:.3f}, mesh_risk={self.state.mesh_risk:.2f}"
        )
        return preds

    def predictive_dampening(self, preds: List[HorizonPrediction]) -> Dict[str, Any]:
        micro = next((p for p in preds if abs(p.horizon - 5.0) < 1e-3), None)
        macro = next((p for p in preds if abs(p.horizon - 120.0) < 1e-3), None)
        scale = self.auto.dampening_scale
        mesh_amp = 1.0 + self.state.mesh_hostile_ratio
        actions = {
            "shrink_deep_ram": False,
            "reduce_ingestion": False,
            "slow_threads": False,
            "prepare_fallback": False,
        }
        if micro and micro.value * scale * mesh_amp > 0.6:
            actions["shrink_deep_ram"] = True
            actions["reduce_ingestion"] = True
        if macro and macro.value * scale * mesh_amp > 0.5:
            actions["slow_threads"] = True
            actions["prepare_fallback"] = True
        self._append_reason(
            f"Dampening: micro={micro.value if micro else None}, macro={macro.value if macro else None}, mesh_amp={mesh_amp:.2f}, actions={actions}"
        )
        return actions

# ---------- Organs ----------

@dataclass
class OrganState:
    name: str
    health: float = 1.0
    risk: float = 0.0
    load: float = 0.0
    notes: str = ""

class BaseOrgan:
    def __init__(self, name: str):
        self.state = OrganState(name=name)

    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        pass

class DeepRamOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        risk = metrics.get("mem", 0.0) / 100.0
        appetite = brain.auto.appetite
        self.state.load = risk * appetite
        self.state.risk = risk
        if risk > 0.8:
            self.state.health -= 0.01
            self.state.notes = "Deep RAM pressure"
        else:
            self.state.health = min(1.0, self.state.health + 0.005)

class BackupEngineOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        self.state.load = 0.2
        self.state.risk = 0.1
        self.state.notes = "Snapshots ready"

class NetworkWatcherOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        net = metrics.get("net", 0.0) / 100.0
        self.state.load = net
        self.state.risk = net * 0.7
        if net > 0.7:
            self.state.notes = "Network pulse high"
        else:
            self.state.notes = ""

class GPUCacheOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        self.state.load = 0.3 * brain.auto.cache_aggressiveness
        self.state.risk = 0.2

class ThermalOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        temp = metrics.get("temp", 50.0)
        risk = max(0.0, (temp - 60.0) / 40.0)
        self.state.load = temp / 100.0
        self.state.risk = risk
        if risk > 0.5:
            self.state.notes = "Thermal stress"
        else:
            self.state.notes = ""

class DiskOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        disk = metrics.get("disk", 0.0) / 100.0
        self.state.load = disk
        self.state.risk = disk * 0.6

class VRAMOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        self.state.load = 0.4
        self.state.risk = 0.3

class AICoachOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        self.state.load = 0.1
        self.state.risk = 0.1
        self.state.notes = "Coaching thresholds"

class SwarmNodeOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        self.state.load = 1.0 - brain.state.collective_health
        self.state.risk = self.state.load

class SelfIntegrityOrgan(BaseOrgan):
    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        drift = abs(brain.state.health_trend)
        if drift > 0.1 or brain.state.meta_confidence < 0.3:
            self.state.health -= 0.01
            self.state.risk += 0.02
            self.state.notes = "Integrity drift – safe mode bias"
        else:
            self.state.health = min(1.0, self.state.health + 0.005)
            self.state.risk = max(0.0, self.state.risk - 0.01)

class BorgMeshOrgan(BaseOrgan):
    def __init__(self, name: str, mesh: BorgMesh):
        super().__init__(name)
        self.mesh = mesh

    def update(self, brain: HybridBrain, metrics: Dict[str, float]):
        st = self.mesh.stats()
        self.state.load = min(1.0, st["corridors"] / max(1, self.mesh.max_corridors))
        self.state.risk = st["mesh_risk"]
        self.state.notes = f"Nodes={st['total']} H={st['hostile_ratio']:.2f}"

# ---------- Organism ----------

class Organism:
    def __init__(self):
        self.brain = HybridBrain()
        self.memory = DummyMemory()
        self.comms = DummyComms()
        self.guardian = DummyGuardian()
        self.mesh = BorgMesh(self.memory, self.comms, self.guardian, self.brain)

        self.organs: List[BaseOrgan] = [
            DeepRamOrgan("DeepRAM"),
            BackupEngineOrgan("BackupEngine"),
            NetworkWatcherOrgan("NetworkWatcher"),
            GPUCacheOrgan("GPUCache"),
            ThermalOrgan("Thermal"),
            DiskOrgan("Disk"),
            VRAMOrgan("VRAM"),
            AICoachOrgan("AICoach"),
            SwarmNodeOrgan("SwarmNode"),
            SelfIntegrityOrgan("SelfIntegrity"),
            BorgMeshOrgan("BorgMesh", self.mesh),
        ]
        self.predictions: List[HorizonPrediction] = []

        self.mesh_in_q: queue.Queue = queue.Queue()
        self.mesh_ops_q: queue.Queue = queue.Queue()
        self.mesh_scanner = BorgScanner(self.mesh, self.mesh_in_q, self.mesh_ops_q)
        self.mesh_worker = BorgWorker(self.mesh, self.mesh_ops_q)
        self.mesh_enforcer = BorgEnforcer(self.mesh)
        self.mesh_scanner.start()
        self.mesh_worker.start()
        self.mesh_enforcer.start()

    def sample_metrics(self) -> Dict[str, float]:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            net = random.uniform(0, 100)
            temp = 50.0
        except Exception:
            cpu = mem = disk = net = 0.0
            temp = 50.0
        return {"cpu": cpu, "mem": mem, "disk": disk, "net": net, "temp": temp}

    def tick(self):
        metrics = self.sample_metrics()
        self.brain.ingest_metrics(metrics["cpu"], metrics["mem"], metrics["disk"], metrics["net"])
        self.predictions = self.brain.predict_multi_horizon()
        damp = self.brain.predictive_dampening(self.predictions)
        for organ in self.organs:
            organ.update(self.brain, metrics)
        telemetry.log("TICK", {"metrics": metrics, "damp": damp})

    def inject_mesh_event(self, url: str, snippet: str, links: List[str]):
        self.mesh_in_q.put({"url": url, "snippet": snippet, "links": links})

organism = Organism()

# ============================
# PART 3 – GUI / NERVE CENTER
# ============================

class MissionControlGUI:
    REFRESH_MS = 1000

    def __init__(self, root, organism: Organism):
        self.root = root
        self.org = organism
        self._configure_theme()

        self.root.title("Autonomous Cipher Engine – Event Horizon Nerve Center")

        self.main = ttk.Frame(root, padding=5)
        self.main.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(self.main)
        self.notebook.pack(fill="both", expand=True)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_altered = ttk.Frame(self.notebook)
        self.tab_reboot = ttk.Frame(self.notebook)
        self.tab_mesh = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="Mission Control")
        self.notebook.add(self.tab_altered, text="Altered States")
        self.notebook.add(self.tab_mesh, text="Borg Mesh Cortex")
        self.notebook.add(self.tab_reboot, text="Reboot Memory")

        self._build_tab_main()
        self._build_tab_altered()
        self._build_tab_mesh()
        self._build_tab_reboot()

        self.refresh()

    def _configure_theme(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#050814"
        fg = "#c7f0ff"
        accent = "#00ffc8"
        style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", background=bg, foreground=accent)
        style.configure("TLabelframe.Label", background=bg, foreground=accent)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background="#0b1020", foreground=fg)
        style.map("TButton",
                  background=[("active", "#101830")],
                  foreground=[("active", accent)])
        self.colors = {
            "bg": bg,
            "fg": fg,
            "accent": accent,
            "danger": "#ff4b81",
            "ok": "#00ffc8",
        }
        self.root.configure(bg=bg)

    # ---------- TAB MAIN ----------

    def _build_tab_main(self):
        top = ttk.Frame(self.tab_main)
        top.pack(side="top", fill="x")

        self._build_hybrid_brain_panel(top)
        self._build_judgment_panel(top)

        mid = ttk.Frame(self.tab_main)
        mid.pack(side="top", fill="x")

        self._build_situational_panel(mid)
        self._build_predictive_panel(mid)

        bottom = ttk.Frame(self.tab_main)
        bottom.pack(side="top", fill="both", expand=True)

        self._build_collective_panel(bottom)
        self._build_command_bar(bottom)
        self._build_dialogue(bottom)

        self._build_organs_panel(self.tab_main)
        self._build_prediction_chart(self.tab_main)
        self._build_reasoning_heatmap(self.tab_main)

    def _build_hybrid_brain_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Hybrid Brain Core")
        frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        self.mode_label = ttk.Label(frame, text="Meta-state: ?")
        self.mode_label.pack(anchor="w")

        self.vol_label = ttk.Label(frame, text="Volatility: ?")
        self.vol_label.pack(anchor="w")

        self.trust_label = ttk.Label(frame, text="Trust: ?")
        self.trust_label.pack(anchor="w")

        self.load_label = ttk.Label(frame, text="Cognitive load: ?")
        self.load_label.pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)

        ttk.Button(btns, text="Stability", command=lambda: self.org.brain.override_meta_state("SENTINEL")).pack(side="left", padx=2)
        ttk.Button(btns, text="Reflex", command=lambda: self.org.brain.override_meta_state("HYPER_FLOW")).pack(side="left", padx=2)
        ttk.Button(btns, text="Exploration", command=lambda: self.org.brain.override_meta_state("DEEP_DREAM")).pack(side="left", padx=2)
        ttk.Button(btns, text="Recovery", command=lambda: self.org.brain.override_meta_state("RECOVERY_FLOW")).pack(side="left", padx=2)

        btns2 = ttk.Frame(frame)
        btns2.pack(fill="x", pady=3)
        ttk.Button(btns2, text="More cautious", command=lambda: self.org.brain.set_stance("CAUTIOUS")).pack(side="left", padx=2)
        ttk.Button(btns2, text="Balanced", command=lambda: self.org.brain.set_stance("BALANCED")).pack(side="left", padx=2)
        ttk.Button(btns2, text="More assertive", command=lambda: self.org.brain.set_stance("ASSERTIVE")).pack(side="left", padx=2)

    def _build_judgment_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Judgment Engine")
        frame.pack(side="right", fill="x", expand=True, padx=5, pady=5)

        self.jconf_label = ttk.Label(frame, text="Judgment confidence: ?")
        self.jconf_label.pack(anchor="w")

        self.bias_label = ttk.Label(frame, text="Bias drift: ?")
        self.bias_label.pack(anchor="w")

        self.samples_label = ttk.Label(frame, text="Samples: ?")
        self.samples_label.pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Reinforce good", command=lambda: self.org.brain.reinforce(True, "manual good")).pack(side="left", padx=2)
        ttk.Button(btns, text="Correct bad", command=lambda: self.org.brain.reinforce(False, "manual bad")).pack(side="left", padx=2)
        ttk.Button(btns, text="Reset bias", command=self.org.brain.reset_bias).pack(side="left", padx=2)

    def _build_situational_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Situational Awareness Cortex")
        frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        self.mission_label = ttk.Label(frame, text="Mission: AUTO")
        self.mission_label.pack(anchor="w")

        self.env_label = ttk.Label(frame, text="Environment: CALM")
        self.env_label.pack(anchor="w")

        self.risk_label = ttk.Label(frame, text="Risk: ?")
        self.risk_label.pack(anchor="w")

        self.opp_label = ttk.Label(frame, text="Opportunity: ?")
        self.opp_label.pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Force PROTECT", command=lambda: self.org.brain.set_mission("PROTECT")).pack(side="left", padx=2)
        ttk.Button(btns, text="Force LEARN", command=lambda: self.org.brain.set_mission("LEARN")).pack(side="left", padx=2)
        ttk.Button(btns, text="Force OPTIMIZE", command=lambda: self.org.brain.set_mission("OPTIMIZE")).pack(side="left", padx=2)
        ttk.Button(btns, text="AUTO", command=lambda: self.org.brain.set_mission("AUTO")).pack(side="left", padx=2)

    def _build_predictive_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Predictive Intelligence")
        frame.pack(side="right", fill="x", expand=True, padx=5, pady=5)

        self.meta_conf_label = ttk.Label(frame, text="Meta-confidence: ?")
        self.meta_conf_label.pack(anchor="w")

        self.anom_label = ttk.Label(frame, text="Anomaly risk: ?")
        self.anom_label.pack(anchor="w")

        self.drive_label = ttk.Label(frame, text="Drive risk: ?")
        self.drive_label.pack(anchor="w")

        self.hive_label = ttk.Label(frame, text="Hive risk: ?")
        self.hive_label.pack(anchor="w")

        self.best_guess_label = ttk.Label(frame, text="Best-Guess: ?")
        self.best_guess_label.pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Short horizon", command=lambda: self.org.brain.set_prediction_horizon_mode("SHORT")).pack(side="left", padx=2)
        ttk.Button(btns, text="Medium", command=lambda: self.org.brain.set_prediction_horizon_mode("MEDIUM")).pack(side="left", padx=2)
        ttk.Button(btns, text="Long", command=lambda: self.org.brain.set_prediction_horizon_mode("LONG")).pack(side="left", padx=2)

    def _build_collective_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Collective Health & Hive Influence")
        frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.coll_label = ttk.Label(frame, text="Collective health: ?")
        self.coll_label.pack(anchor="w")

        self.trend_label = ttk.Label(frame, text="Health trend: ?")
        self.trend_label.pack(anchor="w")

        self.mesh_risk_label = ttk.Label(frame, text="Mesh risk: ?")
        self.mesh_risk_label.pack(anchor="w")

        self.mesh_hostile_label = ttk.Label(frame, text="Mesh hostile ratio: ?")
        self.mesh_hostile_label.pack(anchor="w")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Trust hive more", command=lambda: None).pack(side="left", padx=2)
        ttk.Button(btns, text="Trust hive less", command=lambda: None).pack(side="left", padx=2)
        ttk.Button(btns, text="Isolate node", command=lambda: None).pack(side="left", padx=2)

    def _build_command_bar(self, parent):
        frame = ttk.Labelframe(parent, text="Command Bar")
        frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        ttk.Button(frame, text="Stabilize system", command=lambda: self.org.brain.set_mission("STABILITY")).pack(fill="x", pady=1)
        ttk.Button(frame, text="High-alert mode", command=lambda: self.org.brain.set_mission("PROTECT")).pack(fill="x", pady=1)
        ttk.Button(frame, text="Begin learning cycle", command=lambda: self.org.brain.set_mission("LEARN")).pack(fill="x", pady=1)
        ttk.Button(frame, text="Optimize performance", command=lambda: self.org.brain.set_mission("OPTIMIZE")).pack(fill="x", pady=1)
        ttk.Button(frame, text="Purge anomaly memory", command=lambda: self.org.brain.history.clear()).pack(fill="x", pady=1)
        ttk.Button(frame, text="Snapshot brain state", command=self._snapshot).pack(fill="x", pady=1)
        ttk.Button(frame, text="Rollback to previous", command=self.org.brain.rollback).pack(fill="x", pady=1)
        ttk.Button(frame, text="Export JSON", command=self._export_json).pack(fill="x", pady=1)

    def _build_dialogue(self, parent):
        frame = ttk.Labelframe(parent, text="ASI Dialogue / Reasoning Tail")
        frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.dialog_text = tk.Text(frame, height=10, wrap="word",
                                   bg=self.colors["bg"], fg=self.colors["fg"],
                                   insertbackground=self.colors["accent"])
        self.dialog_text.pack(fill="both", expand=True)

    def _build_organs_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Organs")
        frame.pack(side="bottom", fill="both", expand=True, padx=5, pady=5)

        self.org_list = tk.Listbox(frame, height=8,
                                   bg=self.colors["bg"], fg=self.colors["fg"],
                                   selectbackground=self.colors["accent"])
        self.org_list.pack(fill="both", expand=True)

    def _build_prediction_chart(self, parent):
        frame = ttk.Labelframe(parent, text="Prediction Micro-Chart")
        frame.pack(side="bottom", fill="x", padx=5, pady=5)

        self.canvas_chart = tk.Canvas(frame, height=80, bg="#050814", highlightthickness=0)
        self.canvas_chart.pack(fill="x", expand=True)

    def _build_reasoning_heatmap(self, parent):
        frame = ttk.Labelframe(parent, text="Reasoning Heatmap")
        frame.pack(side="bottom", fill="x", padx=5, pady=5)

        self.heatmap_label = ttk.Label(frame, text="Engines / Organs / Patterns contribution")
        self.heatmap_label.pack(anchor="w")

    # ---------- TAB ALTERED STATES ----------

    def _build_tab_altered(self):
        frame = ttk.Labelframe(self.tab_altered, text="Altered States Cortex")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.altered_label = ttk.Label(frame, text="Meta-state, horizon bias, appetite, thread expansion, mesh influence")
        self.altered_label.pack(anchor="w", pady=5)

        self.altered_state_text = tk.Text(frame, height=12, wrap="word",
                                          bg=self.colors["bg"], fg=self.colors["fg"],
                                          insertbackground=self.colors["accent"])
        self.altered_state_text.pack(fill="both", expand=True)

    # ---------- TAB MESH CORTEX ----------

    def _build_tab_mesh(self):
        frame = ttk.Labelframe(self.tab_mesh, text="Borg Mesh Cortex")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.mesh_stats_label = ttk.Label(frame, text="Mesh stats: ?")
        self.mesh_stats_label.pack(anchor="w")

        self.mesh_events_text = tk.Text(frame, height=10, wrap="word",
                                        bg=self.colors["bg"], fg=self.colors["fg"],
                                        insertbackground=self.colors["accent"])
        self.mesh_events_text.pack(fill="both", expand=True, pady=5)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Snapshot Mesh", command=self._snapshot_mesh).pack(side="left", padx=2)
        ttk.Button(btns, text="Inject Test Corridor", command=self._inject_test_mesh).pack(side="left", padx=2)

    # ---------- TAB REBOOT MEMORY ----------

    def _build_tab_reboot(self):
        frame_reboot = ttk.LabelFrame(self.tab_reboot, text="Reboot Memory Persistence")
        frame_reboot.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame_reboot, text="SMB / UNC Path or local folder:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame_reboot, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame_reboot, text="Pick Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame_reboot, text="Test Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame_reboot, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame_reboot,
            text="Load memory from path on startup",
            variable=self.var_reboot_autoload
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame_reboot, text="Status: Ready", anchor="w", fg="#00cc66", bg=self.colors["bg"])
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    # ---------- Reboot memory commands ----------

    def cmd_pick_reboot_path(self):
        from tkinter import filedialog
        path = filedialog.askdirectory()
        if path:
            self.entry_reboot_path.delete(0, "end")
            self.entry_reboot_path.insert(0, path)

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self.lbl_reboot_status.config(text="Status: No path set", fg="#ff4b81")
            return
        ok = os.path.isdir(path)
        self.lbl_reboot_status.config(
            text=f"Status: {'OK' if ok else 'Not accessible'}",
            fg="#00cc66" if ok else "#ff4b81"
        )

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self.lbl_reboot_status.config(text="Status: No path set", fg="#ff4b81")
            return
        try:
            os.makedirs(path, exist_ok=True)
            state = {
                "brain": json.loads(self.org.brain.export_json()),
                "organs": [asdict(o.state) for o in self.org.organs],
                "history": self.org.brain.history[-300:],
                "mesh_nodes": self.org.mesh.nodes,
                "mesh_edges": list(self.org.mesh.edges),
            }
            with open(os.path.join(path, "organism_state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self.lbl_reboot_status.config(text="Status: Saved", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Error {e}", fg="#ff4b81")

    def try_autoload_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not (self.var_reboot_autoload.get() and path):
            return
        try:
            state_path = os.path.join(path, "organism_state.json")
            if not os.path.isfile(state_path):
                return
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            brain_snap = data.get("brain", {})
            self.org.brain.last_snapshot = brain_snap
            self.org.brain.rollback()
            organ_states = data.get("organs", [])
            for o, s in zip(self.org.organs, organ_states):
                o.state = OrganState(**s)
            self.org.brain.history = data.get("history", [])
            self.org.mesh.nodes = data.get("mesh_nodes", {})
            self.org.mesh.edges = set(tuple(e) for e in data.get("mesh_edges", []))
            self.lbl_reboot_status.config(text="Status: Loaded from path", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Load error {e}", fg="#ff4b81")

    # ---------- Commands ----------

    def _snapshot(self):
        self.org.brain.snapshot()
        self._append_dialog("Snapshot captured.")

    def _export_json(self):
        data = self.org.brain.export_json()
        self._append_dialog("JSON snapshot:\n" + data)

    def _append_dialog(self, msg: str):
        self.dialog_text.insert("end", msg + "\n")
        self.dialog_text.see("end")

    def _snapshot_mesh(self):
        st = self.org.mesh.stats()
        self._append_dialog(f"Mesh snapshot: {st}")

    def _inject_test_mesh(self):
        url = f"http://node-{random.randint(1,9999)}.local"
        links = [f"http://node-{random.randint(1,9999)}.local" for _ in range(3)]
        self.org.inject_mesh_event(url, "test snippet", links)
        self._append_dialog(f"Injected mesh event: {url}")

    # ---------- Refresh ----------

    def refresh(self):
        self.org.tick()
        s = self.org.brain.state

        self.mode_label.config(text=f"Meta-state: {s.meta_state} / stance={s.stance}")
        self.vol_label.config(text=f"Volatility: {s.volatility:.2f}")
        self.trust_label.config(text=f"Trust: {s.trust:.2f}")
        self.load_label.config(text=f"Cognitive load: {s.cognitive_load:.2f}")

        self.jconf_label.config(text=f"Judgment confidence: {s.judgment_confidence:.2f}")
        self.bias_label.config(text=f"Bias drift: {s.bias_drift:.2f}")
        self.samples_label.config(text=f"Samples: {s.sample_count}")

        self.mission_label.config(text=f"Mission: {s.mission}")
        self.env_label.config(text=f"Environment: {s.environment}")
        self.risk_label.config(text=f"Risk: {s.risk_score:.2f}")
        self.opp_label.config(text=f"Opportunity: {s.opportunity_score:.2f}")

        self.meta_conf_label.config(text=f"Meta-confidence: {s.meta_confidence:.2f}")
        self.anom_label.config(text=f"Anomaly risk: {s.anomaly_risk:.2f}")
        self.drive_label.config(text=f"Drive risk: {s.drive_risk:.2f}")
        self.hive_label.config(text=f"Hive risk: {s.hive_risk:.2f}")
        self.best_guess_label.config(text=f"Best-Guess: {self.org.brain.best_guess:.2f} (conf {self.org.brain.best_guess_conf:.2f})")

        self.coll_label.config(text=f"Collective health: {s.collective_health:.2f}")
        self.trend_label.config(text=f"Health trend: {s.health_trend:.3f}")
        self.mesh_risk_label.config(text=f"Mesh risk: {s.mesh_risk:.2f}")
        self.mesh_hostile_label.config(text=f"Mesh hostile ratio: {s.mesh_hostile_ratio:.2f}")

        self.org_list.delete(0, "end")
        for organ in self.org.organs:
            st = organ.state
            line = f"{st.name}: H={st.health:.2f} R={st.risk:.2f} L={st.load:.2f} {st.notes}"
            self.org_list.insert("end", line)

        self.dialog_text.delete("1.0", "end")
        for line in self.org.brain.reasoning_tail[-15:]:
            self.dialog_text.insert("end", line + "\n")
        self.dialog_text.see("end")

        self._update_altered_states_view()
        self._update_prediction_chart()
        self._update_reasoning_heatmap()
        self._update_mesh_view()

        self.root.after(self.REFRESH_MS, self.refresh)

    def _update_altered_states_view(self):
        self.altered_state_text.delete("1.0", "end")
        b = self.org.brain
        s = b.state
        auto = b.auto
        lines = [
            f"Meta-state: {s.meta_state}",
            f"Momentum: {s.meta_state_momentum:.2f}",
            f"Horizon mode: {s.prediction_horizon}",
            f"Appetite: {auto.appetite:.2f}",
            f"Threshold scale: {auto.threshold_scale:.2f}",
            f"Horizon bias: {auto.horizon_bias:.2f}",
            f"Dampening scale: {auto.dampening_scale:.2f}",
            f"Cache aggressiveness: {auto.cache_aggressiveness:.2f}",
            f"Thread expansion: {auto.thread_expansion:.2f}",
            f"Mesh risk: {s.mesh_risk:.2f}",
            f"Mesh hostile ratio: {s.mesh_hostile_ratio:.2f}",
            f"Mesh entropy: {s.mesh_entropy:.2f}",
        ]
        self.altered_state_text.insert("end", "\n".join(lines))

    def _update_prediction_chart(self):
        c = self.canvas_chart
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 80

        preds = self.org.predictions
        if not preds:
            return

        horizons = [p.horizon for p in preds]
        values = [p.value for p in preds]
        min_h, max_h = min(horizons), max(horizons)
        min_v, max_v = 0.0, 1.0

        def map_x(hv):
            return 10 + (hv - min_h) / max(1e-6, max_h - min_h) * (w - 20)

        def map_y(v):
            return h - 10 - (v - min_v) / max(1e-6, max_v - min_v) * (h - 20)

        base_y = map_y(self.org.brain.baseline_risk)
        c.create_line(0, base_y, w, base_y, fill="#444444", dash=(2, 2))

        colors = ["#00ffc8", "#00aaff", "#ffaa00", "#ff4b81"]
        for i, p in enumerate(preds):
            x = map_x(p.horizon)
            y = map_y(p.value)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colors[i % len(colors)], outline="")
            if i > 0:
                p_prev = preds[i - 1]
                c.create_line(map_x(p_prev.horizon), map_y(p_prev.value), x, y, fill=colors[i % len(colors)])

        if len(preds) >= 2:
            mid = preds[1]
            stance = self.org.brain.state.stance
            stance_color = {"CAUTIOUS": "#00ffc8", "BALANCED": "#ffaa00", "ASSERTIVE": "#ff4b81"}.get(stance, "#ffaa00")
            c.create_line(0, map_y(mid.value), w, map_y(mid.value), fill=stance_color)

        bg_y = map_y(self.org.brain.best_guess)
        c.create_line(0, bg_y, w, bg_y, fill="#ffffff")

    def _update_reasoning_heatmap(self):
        preds = self.org.predictions
        if not preds:
            self.heatmap_label.config(text="Reasoning Heatmap: no predictions yet")
            return
        last = preds[-1]
        engines = last.engines
        items = sorted(engines.items(), key=lambda kv: kv[1], reverse=True)
        txt = "Reasoning Heatmap – engines: " + ", ".join(f"{k}:{v:.2f}" for k, v in items)
        self.heatmap_label.config(text=txt)

    def _update_mesh_view(self):
        st = self.org.mesh.stats()
        self.mesh_stats_label.config(
            text=f"Mesh stats – nodes={st['total']} disc={st['discovered']} built={st['built']} "
                 f"enforced={st['enforced']} corridors={st['corridors']} hostile={st['hostile_ratio']:.2f} "
                 f"mesh_risk={st['mesh_risk']:.2f}"
        )
        self.mesh_events_text.delete("1.0", "end")
        for evt in self.org.memory.mesh_events[-20:]:
            line = f"{evt['time']} {evt['type']} {evt.get('url','')} {evt.get('status','')} risk={evt.get('risk','')}\n"
            self.mesh_events_text.insert("end", line)
        self.mesh_events_text.see("end")

# ============================
# PART 4 – AUGMENTATION + MAIN
# ============================

class MovidiusInferenceEngine:
    """
    Stub for ONNX/Movidius model.
    Mesh-aware: trains on combined system + mesh risk windows.
    """
    def __init__(self):
        self.model_loaded = False

    def train_stub(self, windows: List[List[float]], labels: List[float]):
        telemetry.log("MOVIDIUS_TRAIN", {"samples": len(windows)})

    def predict_stub(self, window: List[float]) -> float:
        if not window:
            return 0.5
        return max(0.0, min(1.0, sum(window) / len(window)))


movidius_engine = MovidiusInferenceEngine()

class AugmentationEngine:
    """
    Mesh-aware augmentation.
    Hooks into brain lifecycle for extra hints and ONNX training windows.
    """
    def __init__(self, organism: Organism):
        self.org = organism
        self.train_buffer: List[Tuple[List[float], float]] = []

    def on_tick(self):
        b = self.org.brain
        hist = b.history[-30:]
        if len(hist) >= 10:
            mesh_sig = [b.state.mesh_risk, b.state.mesh_hostile_ratio, b.state.mesh_entropy]
            window = hist[-10:] + mesh_sig
            pred = movidius_engine.predict_stub(window)
            b.baseline_risk = 0.9 * b.baseline_risk + 0.1 * pred
            self.train_buffer.append((window, b.state.risk_score))
            if len(self.train_buffer) >= 50:
                windows = [w for w, _ in self.train_buffer]
                labels = [l for _, l in self.train_buffer]
                movidius_engine.train_stub(windows, labels)
                self.train_buffer.clear()
        if b.state.mesh_hostile_ratio > 0.4:
            b.auto.dampening_scale = min(2.0, b.auto.dampening_scale + 0.05)
            b.auto.appetite = max(0.5, b.auto.appetite - 0.02)
        else:
            b.auto.dampening_scale = max(0.8, b.auto.dampening_scale - 0.01)
            b.auto.appetite = min(1.2, b.auto.appetite + 0.01)

augmentation = AugmentationEngine(organism)

_orig_tick = organism.tick
def _aug_tick():
    _orig_tick()
    augmentation.on_tick()
organism.tick = _aug_tick

# ---------- Main ----------

if __name__ == "__main__":
    root = tk.Tk()
    gui = MissionControlGUI(root, organism)
    gui.try_autoload_reboot_memory()
    root.mainloop()

