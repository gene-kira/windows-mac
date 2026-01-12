#!/usr/bin/env python3
# ============================================================
# SYSTEM NERVE CENTER – HybridBrain + Tri-Stance + MultiGame v4.1 "Event Horizon"
# - Multi-engine Best-Guess with regime-aware, drift-aware fusion
# - Back 4 Blood + World War Z + other games + world pattern + live events
# - Pattern memory + Horizon pattern memory (future risk projection)
# - Outcome modeling (overload / stable / beast_win prediction)
# - Meta-state momentum (Sentinel / Hyper-Flow / Deep-Dream / Recovery-Flow)
# - Tri-Stance decision engine (Conservative / Balanced / Beast)
# - Predictive dampening + micro-recovery loops
# - SelfIntegrityOrgan with drift-aware self-correction
# - Movidius/ONNX training hook stub + GUI
# - Back4BloodAnalyzer + Back4BloodPhysicsOrgan from JSON metrics or fallback
# - InteractionCoachOrgan: multi-game human↔AI training ground (episodes with game_id)
# - GameDetector: multi-game detection + profiles + fallback "other_game"
# - GameAIConversationBridge: explicit Game AI ↔ System AI protocol
# - Conversation Tab: last 20 JSON-like exchanges
# - Horizon Patterns Tab: future-risk maps by game/time/stance
# - Full B4B memory persistence (metrics, physics, episodes, game state)
# - World foresight via pattern memory + live events + Event Horizon projections
# ============================================================

import sys
import subprocess
import threading
import time
import math
import json
import os
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple

# ------------------------ AutoLoader --------------------------

def ensure_package(pkg, alias=None):
    try:
        __import__(alias or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(alias or pkg)

ensure_package("psutil", "psutil")
import psutil  # type: ignore

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    print("Tkinter is required for GUI.", file=sys.stderr)
    sys.exit(1)

# ONNX (Movidius / NPU hook)
try:
    ensure_package("onnxruntime", "onnxruntime")
    import onnxruntime as ort  # type: ignore
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

# ------------------------ Utility -----------------------------

def utc_ts() -> float:
    return time.time()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def local_time_components() -> Tuple[int, int]:
    t = time.localtime()
    return t.tm_hour, t.tm_wday  # 0=Monday

# ------------------------ Rolling windows ----------------------

class RollingWindow:
    def __init__(self, maxlen: int = 300):
        from collections import deque
        self.values = deque(maxlen=maxlen)

    def push(self, v: float):
        self.values.append(v)

    def last(self) -> float:
        return self.values[-1] if self.values else 0.0

    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    def var(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return statistics.pvariance(self.values)

    def slope(self) -> float:
        n = len(self.values)
        if n < 3:
            return 0.0
        xs = list(range(n))
        ys = list(self.values)
        xm = sum(xs) / n
        ym = sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        return num / den if den > 0 else 0.0

    def ewma(self, alpha: float) -> float:
        if not self.values:
            return 0.0
        ew = self.values[0]
        for v in list(self.values)[1:]:
            ew = alpha * v + (1 - alpha) * ew
        return ew

# ------------------------ Brain data models --------------------

@dataclass
class BestGuessState:
    value: float = 0.0
    confidence: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    regime: str = "stable"

@dataclass
class PatternMemoryEntry:
    pattern_id: str
    overload_count: int = 0
    stable_count: int = 0
    beast_win_count: int = 0

@dataclass
class HorizonPatternEntry:
    pattern_id: str
    future_overload: int = 0
    future_stable: int = 0
    future_beast_win: int = 0
    avg_future_risk: float = 0.5
    samples: int = 0

@dataclass
class PatternMemory:
    patterns: Dict[str, PatternMemoryEntry] = field(default_factory=dict)
    horizon_patterns: Dict[str, HorizonPatternEntry] = field(default_factory=dict)

    def update(self, pattern_id: str, outcome: str):
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = PatternMemoryEntry(pattern_id)
        e = self.patterns[pattern_id]
        if outcome == "overload":
            e.overload_count += 1
        elif outcome == "stable":
            e.stable_count += 1
        elif outcome == "beast_win":
            e.beast_win_count += 1

    def update_horizon(self, pattern_id: str, future_risk: float):
        if pattern_id not in self.horizon_patterns:
            self.horizon_patterns[pattern_id] = HorizonPatternEntry(pattern_id)
        e = self.horizon_patterns[pattern_id]
        band = "stable"
        if future_risk > 0.8:
            band = "overload"
        elif future_risk > 0.4:
            band = "beast_win"
        if band == "overload":
            e.future_overload += 1
        elif band == "stable":
            e.future_stable += 1
        else:
            e.future_beast_win += 1
        e.samples += 1
        e.avg_future_risk += (future_risk - e.avg_future_risk) / max(1, e.samples)

@dataclass
class JudgmentState:
    good: int = 0
    bad: int = 0
    bias: float = 0.0
    frozen: bool = False

    def confidence(self) -> float:
        total = self.good + self.bad
        return self.good / total if total else 0.0

@dataclass
class MetaState:
    name: str = "Sentinel"
    momentum: float = 0.0   # 0..1 inertia

@dataclass
class CortexState:
    risk: float = 0.0
    opportunity: float = 0.0
    environment: str = "CALM"
    mission: str = "PROTECT"
    anticipation: str = "Baseline"

@dataclass
class CollectiveState:
    risk: float = 0.0
    density: float = 0.0
    agreement: float = 1.0
    divergence: float = 0.0

@dataclass
class IntegrityState:
    sensor_fresh: bool = True
    organs_present: bool = True
    prediction_drift: float = 0.0
    integrity_score: float = 1.0

@dataclass
class HybridBrain:
    best_guess: BestGuessState = field(default_factory=BestGuessState)
    judgment: JudgmentState = field(default_factory=JudgmentState)
    pattern_memory: PatternMemory = field(default_factory=PatternMemory)
    meta_state: MetaState = field(default_factory=MetaState)
    cortex: CortexState = field(default_factory=CortexState)
    collective: CollectiveState = field(default_factory=CollectiveState)
    integrity: IntegrityState = field(default_factory=IntegrityState)
    dynamic_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "risk_high": 0.7,
        "risk_critical": 0.9,
    })
    reinforcement: Dict[str, float] = field(default_factory=lambda: {
        "stability": 0.5,
        "performance": 0.5,
    })
    baseline_risk: float = 0.2
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_heatmap: Dict[str, float] = field(default_factory=dict)
    stance: str = "Balanced"
    current_game_id: Optional[str] = None
    # Conversation state
    game_ai_alignment: float = 0.5
    game_ai_pressure: float = 0.0
    # Outcome model
    predicted_overload: float = 0.0
    predicted_stable: float = 0.0
    predicted_beast_win: float = 0.0

# ------------------------ Organs + micro-recovery -------------

class DeepRamOrgan:
    def __init__(self):
        self.usage = 0.0
        self.health = 1.0
        self.risk = 0.0
        self.appetite = 1.0

    def tick(self):
        mem = psutil.virtual_memory()
        self.usage = mem.percent / 100.0
        self.risk = self.usage
        self.health = clamp01(1.0 - self.risk)

    def apply_recovery(self, stance: str, global_risk: float):
        if stance == "Conservative" or global_risk > 0.8:
            self.appetite = max(0.3, self.appetite - 0.1)
        elif stance == "Beast" and global_risk < 0.5:
            self.appetite = min(1.5, self.appetite + 0.05)
        else:
            self.appetite = clamp01(self.appetite)

class NetworkWatcherOrgan:
    def __init__(self):
        self.bytes = 0
        self.risk = 0.0
        self.health = 1.0
        self._last = None
        self.rate_limit = 1.0

    def tick(self):
        c = psutil.net_io_counters()
        total = c.bytes_sent + c.bytes_recv
        if self._last is None:
            delta = 0.0
        else:
            delta = max(0.0, total - self._last)
        self._last = total
        self.bytes = delta
        self.risk = clamp01(delta / (1024 * 1024))
        self.health = clamp01(1.0 - self.risk)

    def apply_recovery(self, stance: str, global_risk: float):
        if stance == "Conservative" or global_risk > 0.7:
            self.rate_limit = max(0.3, self.rate_limit - 0.1)
        elif stance == "Beast" and global_risk < 0.5:
            self.rate_limit = min(1.5, self.rate_limit + 0.05)
        else:
            self.rate_limit = clamp01(self.rate_limit)

class DiskOrgan:
    def __init__(self):
        self.bytes = 0
        self.risk = 0.0
        self.health = 1.0
        self._last = None
        self.stagger_factor = 1.0

    def tick(self):
        io = psutil.disk_io_counters()
        total = io.read_bytes + io.write_bytes
        if self._last is None:
            delta = 0.0
        else:
            delta = max(0.0, total - self._last)
        self._last = total
        self.bytes = delta
        self.risk = clamp01(delta / (1024 * 1024))
        self.health = clamp01(1.0 - self.risk)

    def apply_recovery(self, stance: str, global_risk: float):
        if stance == "Conservative" or global_risk > 0.7:
            self.stagger_factor = min(3.0, self.stagger_factor + 0.1)
        elif stance == "Beast" and global_risk < 0.5:
            self.stagger_factor = max(1.0, self.stagger_factor - 0.05)

class ThermalOrgan:
    def __init__(self):
        self.temp = 40.0
        self.risk = 0.0
        self.health = 1.0
        self.cooldown_factor = 1.0

    def tick(self):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for _, entries in temps.items():
                    if entries:
                        self.temp = entries[0].current
                        break
        except Exception:
            pass
        self.risk = clamp01((self.temp - 40.0) / 40.0) if self.temp else 0.0
        self.health = clamp01(1.0 - self.risk)

    def apply_recovery(self, stance: str, global_risk: float):
        if global_risk > 0.7 or self.risk > 0.5:
            self.cooldown_factor = min(2.0, self.cooldown_factor + 0.1)
        else:
            self.cooldown_factor = max(1.0, self.cooldown_factor - 0.05)

class GPUCacheOrgan:
    def apply_recovery(self, stance: str, global_risk: float): pass

class VRAMOrgan:
    def apply_recovery(self, stance: str, global_risk: float): pass

class AICoachOrgan:
    def apply_recovery(self, stance: str, global_risk: float): pass

class SwarmNodeOrgan:
    def apply_recovery(self, stance: str, global_risk: float): pass

class BackupEngineOrgan:
    def apply_recovery(self, stance: str, global_risk: float): pass

# ------------------------ Back4Blood Analyzer -----------------

class Back4BloodAnalyzer:
    def __init__(self, brain: HybridBrain):
        self.brain = brain
        self.metrics_path: Optional[str] = None
        self.last_seen_ts: Optional[str] = None
        self.last_metrics: Dict[str, Any] = {}

    def set_metrics_path(self, path: str):
        self.metrics_path = path

    def tick(self):
        if not self.metrics_path or not os.path.exists(self.metrics_path):
            return
        try:
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        ts = str(data.get("timestamp", ""))
        if self.last_seen_ts == ts:
            return

        self.last_seen_ts = ts
        self.last_metrics = data

        pattern_id = f"b4b:{data.get('fps','?')}:{data.get('ping','?')}"
        raw_outcome = str(data.get("outcome", "stable")).lower()

        if raw_outcome in ("overload", "stable", "beast_win"):
            outcome = raw_outcome
        elif raw_outcome in ("victory", "win", "success"):
            outcome = "beast_win"
        elif raw_outcome in ("defeat", "loss", "wipe"):
            outcome = "overload"
        else:
            outcome = "stable"

        self.brain.pattern_memory.update(pattern_id, outcome)

# ------------------------ Back4Blood Physics Organ ------------

class Back4BloodPhysicsOrgan:
    def __init__(self, analyzer: Back4BloodAnalyzer, game_detector: Optional["GameDetector"] = None):
        self.analyzer = analyzer
        self.game_detector = game_detector
        self.physics_stress: float = 0.0
        self.movement_intensity: float = 0.0
        self.server_stress: float = 0.0
        self.outcome: str = "unknown"

    def tick(self):
        m = self.analyzer.last_metrics

        if (not m) and self.game_detector and self.game_detector.game_running and \
           (self.game_detector.current_game_id == "b4b"):
            cpu = psutil.cpu_percent() / 100.0
            mem = psutil.virtual_memory().percent / 100.0
            net = psutil.net_io_counters()
            net_rate = min(1.0, (net.bytes_sent + net.bytes_recv) / (1024 * 1024))

            self.physics_stress = clamp01(0.5 * cpu + 0.3 * mem + 0.2 * net_rate)
            self.movement_intensity = clamp01(0.5 + 0.5 * cpu)
            self.server_stress = clamp01(net_rate)
            self.outcome = "stable"
            return

        if not m:
            self.physics_stress = 0.0
            self.movement_intensity = 0.0
            self.server_stress = 0.0
            self.outcome = "unknown"
            return

        fps = float(m.get("fps", 120.0))
        ping = float(m.get("ping", 40.0))
        cpu_load = float(m.get("cpu_load", 0.5))
        mem_load = float(m.get("mem_load", 0.5))
        outcome = str(m.get("outcome", "stable")).lower()

        fps_norm = clamp01(max(0.0, 200.0 - fps) / 200.0)
        load_norm = clamp01(0.5 * cpu_load + 0.5 * mem_load)
        self.physics_stress = clamp01(0.6 * fps_norm + 0.4 * load_norm)
        self.movement_intensity = clamp01((fps / 240.0) * (0.5 + 0.5 * cpu_load))
        self.server_stress = clamp01(ping / 200.0)
        self.outcome = outcome

# ------------------------ Interaction Coach Organ -------------  

@dataclass
class InteractionEpisode:
    game_id: str
    ts_start: float
    ts_end: float
    human_intensity: float
    physics_stress_avg: float
    server_stress_avg: float
    stance_history: List[str]
    restoration_used: bool
    outcome: str

class InteractionCoachOrgan:
    def __init__(self, brain: HybridBrain,
                 b4b_analyzer: Back4BloodAnalyzer,
                 b4b_physics: Back4BloodPhysicsOrgan,
                 game_detector: "GameDetector"):
        self.brain = brain
        self.b4b_analyzer = b4b_analyzer
        self.b4b_physics = b4b_physics
        self.game_detector = game_detector
        self.current: Optional[InteractionEpisode] = None
        self.episodes: List[InteractionEpisode] = []
        self._stance_trace: List[str] = []
        self._stress_trace: List[float] = []
        self._server_trace: List[float] = []

    def start_if_needed(self):
        if self.current is None and self.game_detector.game_running:
            now = utc_ts()
            gid = self.game_detector.current_game_id or "unknown"
            self.current = InteractionEpisode(
                game_id=gid,
                ts_start=now,
                ts_end=now,
                human_intensity=0.0,
                physics_stress_avg=0.0,
                server_stress_avg=0.0,
                stance_history=[],
                restoration_used=False,
                outcome="unknown",
            )
            self._stance_trace.clear()
            self._stress_trace.clear()
            self._server_trace.clear()

    def tick(self, restoration_active: bool):
        if self.current is None:
            self.start_if_needed()
            if self.current is None:
                return

        self._stance_trace.append(self.brain.stance)
        self._stress_trace.append(self.b4b_physics.physics_stress)
        self._server_trace.append(self.b4b_physics.server_stress)
        self.current.ts_end = utc_ts()
        if restoration_active:
            self.current.restoration_used = True

        outcome = self.b4b_physics.outcome
        if outcome in ("overload", "stable", "beast_win", "victory", "defeat"):
            self._finalize_episode(outcome)

    def _finalize_episode(self, outcome: str):
        if self.current is None:
            return
        self.current.outcome = outcome
        self.current.stance_history = list(self._stance_trace)
        self.current.physics_stress_avg = statistics.mean(self._stress_trace) if self._stress_trace else 0.0
        self.current.server_stress_avg = statistics.mean(self._server_trace) if self._server_trace else 0.0
        self.current.human_intensity = self.current.physics_stress_avg
        self.episodes.append(self.current)

        if outcome in ("stable", "victory", "beast_win"):
            self.brain.reinforcement["stability"] = clamp01(self.brain.reinforcement["stability"] + 0.02)
        if outcome in ("overload", "defeat"):
            self.brain.reinforcement["stability"] = clamp01(self.brain.reinforcement["stability"] - 0.02)

        self.current = None
        self._stance_trace.clear()
        self._stress_trace.clear()
        self._server_trace.clear()

# ------------------------ Game AI conversation bridge ---------

class GameAIConversationBridge:
    def __init__(self, brain: HybridBrain, game_detector: "GameDetector",
                 b4b_physics: Back4BloodPhysicsOrgan):
        self.brain = brain
        self.game_detector = game_detector
        self.b4b_physics = b4b_physics
        self.last_game_ai_msg: Dict[str, Any] = {}
        self.last_system_ai_reply: Dict[str, Any] = {}
        self.transcript: List[Tuple[str, Dict[str, Any]]] = []  # ("GAME_AI"/"SYSTEM_AI", message)
        self.max_messages: int = 20

    def _append_transcript(self, speaker: str, msg: Dict[str, Any]):
        self.transcript.append((speaker, msg))
        if len(self.transcript) > self.max_messages:
            self.transcript.pop(0)

    def build_game_ai_message(self) -> Dict[str, Any]:
        gid = self.game_detector.current_game_id or "none"
        risk = self.brain.cortex.risk
        stance = self.brain.stance
        physics_stress = self.b4b_physics.physics_stress
        server_stress = self.b4b_physics.server_stress
        outcome = self.b4b_physics.outcome

        msg = {
            "source": "game_ai",
            "game_id": gid,
            "ts": utc_ts(),
            "perception": {
                "player_risk": risk,
                "player_stance": stance,
            },
            "environment": {
                "physics_stress": physics_stress,
                "server_stress": server_stress,
                "recent_outcome": outcome,
            },
            "intent": self._infer_game_intent(physics_stress, server_stress, outcome),
        }
        self.last_game_ai_msg = msg
        self._append_transcript("GAME_AI", msg)
        return msg

    def _infer_game_intent(self, physics_stress: float, server_stress: float,
                           outcome: str) -> str:
        if physics_stress > 0.7 or server_stress > 0.7:
            return "aggressive"
        if outcome in ("overload", "defeat"):
            return "pressure_player"
        if outcome in ("beast_win", "victory"):
            return "recover_and_scale"
        return "neutral"

    def build_system_ai_reply(self, game_msg: Dict[str, Any]) -> Dict[str, Any]:
        gid = game_msg.get("game_id", "none")
        risk = self.brain.cortex.risk
        stance = self.brain.stance
        meta = self.brain.meta_state.name
        integrity = self.brain.integrity.integrity_score
        world_bias = self.brain.best_guess.value

        if risk > 0.8 or integrity < 0.6:
            recommendation = "ease_off"
        elif risk < 0.4 and integrity > 0.8:
            recommendation = "intensify"
        else:
            recommendation = "maintain"

        reply = {
            "source": "system_ai",
            "game_id": gid,
            "ts": utc_ts(),
            "state": {
                "stance": stance,
                "meta_state": meta,
                "risk": risk,
                "integrity": integrity,
                "world_risk_bias": world_bias,
            },
            "recommendation": recommendation,
        }
        self.last_system_ai_reply = reply
        self._append_transcript("SYSTEM_AI", reply)
        return reply

    def conversation_step(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        game_msg = self.build_game_ai_message()
        sys_reply = self.build_system_ai_reply(game_msg)
        return game_msg, sys_reply

    def alignment_score(self) -> float:
        intent = (self.last_game_ai_msg.get("intent") or "neutral").lower()
        rec = (self.last_system_ai_reply.get("recommendation") or "maintain").lower()

        if intent == "aggressive" and rec == "intensify":
            return 1.0
        if intent == "pressure_player" and rec in ("ease_off", "maintain"):
            return 0.3
        if intent == "recover_and_scale" and rec == "ease_off":
            return 0.8
        if intent == "neutral" and rec == "maintain":
            return 0.9
        return 0.5

# ------------------------ Game Detector (multi-game) ----------  

GAME_PROFILES: Dict[str, Dict[str, float]] = {
    "b4b": {
        "baseline_risk": 0.3,
    },
    "worldwarz": {
        "baseline_risk": 0.35,
    },
    "other_game": {
        "baseline_risk": 0.35,
    },
    "generic": {
        "baseline_risk": 0.3,
    },
}

GAME_DISPLAY_NAMES: Dict[str, str] = {
    "b4b": "Back 4 Blood",
    "worldwarz": "World War Z",
    "other_game": "Other Game",
}

class GameDetector:
    PLATFORM_STEAM = "steam"
    PLATFORM_EPIC = "epic"

    GAME_MAP = {
        "back4blood.exe": ("b4b", PLATFORM_STEAM),
        "back4bloodsteam.exe": ("b4b", PLATFORM_STEAM),
        "b4b.exe": ("b4b", PLATFORM_STEAM),
        "b4bshipping.exe": ("b4b", PLATFORM_STEAM),

        "wwz.exe": ("worldwarz", PLATFORM_EPIC),
        "worldwarz.exe": ("worldwarz", PLATFORM_EPIC),
    }

    def __init__(self):
        self.game_running: bool = False
        self.current_game_id: Optional[str] = None
        self.platform: Optional[str] = None
        self.proc_info: Optional[Dict[str, Any]] = None
        self.last_state: bool = False

    def tick(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        found = False
        info = None
        game_id = None
        platform = None

        for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_info"]):
            try:
                name = p.info["name"].lower()
            except Exception:
                continue
            if name in self.GAME_MAP:
                found = True
                game_id, platform = self.GAME_MAP[name]
                info = {
                    "pid": p.info["pid"],
                    "cpu": p.info["cpu_percent"],
                    "mem": p.info["memory_info"].rss,
                    "name": name,
                }
                break

        if not found:
            for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_info"]):
                try:
                    name = p.info["name"].lower()
                    cpu = p.info["cpu_percent"]
                except Exception:
                    continue

                if cpu and cpu > 10 and not any(sys_name in name for sys_name in (
                    "system", "idle", "explorer", "svchost", "chrome", "firefox", "discord"
                )):
                    found = True
                    game_id = "other_game"
                    platform = "unknown"
                    info = {
                        "pid": p.info["pid"],
                        "cpu": cpu,
                        "mem": p.info["memory_info"].rss,
                        "name": name,
                    }
                    break

        self.game_running = found
        self.current_game_id = game_id
        self.platform = platform
        self.proc_info = info
        return found, info

# ------------------------ Self-Integrity Organ ----------------

class SelfIntegrityOrgan:
    def __init__(self, predictor_windows: Dict[str, RollingWindow], organs: Dict[str, Any], brain: HybridBrain):
        self.predictor_windows = predictor_windows
        self.organs = organs
        self.brain = brain
        self.last_forecast: Optional[float] = None

    def tick(self):
        sensor_fresh = all(w.values for w in self.predictor_windows.values())
        organs_present = all(self.organs.values())
        drift = 0.0
        if self.last_forecast is not None:
            drift = abs(self.brain.best_guess.value - self.last_forecast)
        self.last_forecast = self.brain.best_guess.value

        score = 1.0
        if not sensor_fresh:
            score -= 0.2
        if not organs_present:
            score -= 0.3
        score -= clamp01(drift)

        self.brain.integrity.sensor_fresh = sensor_fresh
        self.brain.integrity.organs_present = organs_present
        self.brain.integrity.prediction_drift = drift
        self.brain.integrity.integrity_score = clamp01(score)

# ------------------------ Movidius / ONNX Trainer -------------  

class MovidiusTrainer:
    def __init__(self):
        self.last_train_ts: Optional[float] = None

    def train_from_logs(self, dataset_path: str, output_model_path: str):
        self.last_train_ts = time.time()
        print(f"[MovidiusTrainer] Training stub: {dataset_path} -> {output_model_path}")

# ------------------------ Prediction engine -------------------

class PredictionEngine:
    def __init__(self, brain: HybridBrain):
        self.brain = brain
        self.cpu = RollingWindow(240)
        self.mem = RollingWindow(240)
        self.disk = RollingWindow(240)
        self.net = RollingWindow(240)
        self.onnx_session = None
        self.trainer = MovidiusTrainer()
        self.b4b_provider: Optional[Back4BloodPhysicsOrgan] = None
        if HAS_ONNX:
            self.onnx_session = None

    def ingest_sample(self):
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        io = psutil.disk_io_counters()
        disk_total = io.read_bytes + io.write_bytes
        netc = psutil.net_io_counters()
        net_total = netc.bytes_sent + netc.bytes_recv

        if self.disk.values:
            disk_delta = max(0.0, disk_total - self.disk.last())
        else:
            disk_delta = 0.0
        if self.net.values:
            net_delta = max(0.0, net_total - self.net.last())
        else:
            net_delta = 0.0

        disk_norm = min(1.0, disk_delta / (1024 * 1024))
        net_norm = min(1.0, net_delta / (1024 * 1024))

        self.cpu.push(cpu)
        self.mem.push(mem)
        self.disk.push(disk_norm)
        self.net.push(net_norm)

    def _regime(self, mean: float, slope: float, var: float) -> str:
        if var < 0.002 and abs(slope) < 0.001:
            return "stable"
        if slope > 0.01 and var < 0.02:
            return "rising"
        if var > 0.05:
            return "chaotic"
        if slope < -0.01:
            return "cooling"
        return "stable"

    def _onnx_vote(self, features: List[float]) -> float:
        if not self.onnx_session:
            return 0.5
        return 0.5

    def _b4b_outcome_score(self) -> float:
        if not self.b4b_provider:
            return 0.5
        outcome = self.b4b_provider.outcome
        mapping = {
            "beast_win": 0.0,
            "victory": 0.3,
            "stable": 0.3,
            "defeat": 0.8,
            "overload": 1.0,
        }
        return mapping.get(outcome, 0.5)

    def _pattern_influence(self) -> float:
        total_overload = sum(e.overload_count for e in self.brain.pattern_memory.patterns.values())
        total_stable = sum(e.stable_count for e in self.brain.pattern_memory.patterns.values())
        total_win = sum(e.beast_win_count for e in self.brain.pattern_memory.patterns.values())
        raw = (total_overload * 1.0 - total_stable * 0.5 - total_win * 1.0) / 50.0
        return clamp01(0.5 + raw)

    def _world_pattern_score(self, cpu_mean: float, net_mean: float) -> float:
        hour, weekday = local_time_components()
        cpu_band = int(cpu_mean * 10)
        net_band = int(net_mean * 10)
        pattern_id = f"world:{weekday}:{hour}:{cpu_band}:{net_band}"
        entry = self.brain.pattern_memory.patterns.get(pattern_id)
        if not entry:
            return 0.5
        raw = (entry.overload_count - entry.stable_count * 0.5 - entry.beast_win_count * 0.8) / 10.0
        centered = 0.5 + raw
        return max(0.2, min(0.8, centered))

    def _live_event_score(self):
        hour, weekday = local_time_components()
        if hour in (18, 19, 20, 21):
            return 0.7
        if hour == 3:
            return 0.8
        if weekday in (5, 6):
            return 0.6
        return 0.3

    def _event_horizon_vote(self, cpu_mean: float, net_mean: float) -> float:
        hour, weekday = local_time_components()
        game_id = getattr(self.brain, "current_game_id", "none") or "none"
        stance = self.brain.stance
        meta = self.brain.meta_state.name
        cpu_band = int(cpu_mean * 10)
        net_band = int(net_mean * 10)
        pattern_id = f"horizon:{game_id}:{stance}:{meta}:{weekday}:{hour}:{cpu_band}:{net_band}"
        entry = self.brain.pattern_memory.horizon_patterns.get(pattern_id)
        if not entry or entry.samples < 3:
            return 0.5
        return clamp01(entry.avg_future_risk)

    def _outcome_model(self, votes: Dict[str, float]) -> Tuple[float, float, float]:
        b4b_stress = votes.get("b4b_stress", 0.0)
        turb = votes.get("turbulence", 0.0)
        baseline_dev = votes.get("baseline_dev", 0.0)
        world_pattern = votes.get("world_pattern", 0.5)
        game_ai_pressure = votes.get("game_ai_pressure", 0.5)

        overload = clamp01(0.35 * b4b_stress + 0.25 * turb + 0.2 * baseline_dev + 0.2 * game_ai_pressure)
        stable = clamp01(1.0 - overload * 0.8 - abs(world_pattern - 0.3) * 0.2)
        beast = clamp01((1.0 - overload) * 0.5 + self.brain.reinforcement.get("performance", 0.5) * 0.5)

        total = overload + stable + beast
        if total <= 1e-6:
            return 0.33, 0.34, 0.33
        return overload / total, stable / total, beast / total

    def _engine_votes(self) -> Tuple[float, Dict[str, float], str, Dict[str, float]]:
        if not self.cpu.values:
            return 0.0, {}, "stable", {}

        cpu_last = self.cpu.last()
        cpu_mean = self.cpu.mean()
        cpu_var = self.cpu.var()
        cpu_slope = self.cpu.slope()
        net_mean = self.net.mean()
        turbulence = clamp01(math.sqrt(cpu_var) * 5.0)
        baseline = self.brain.baseline_risk

        ewma_short = self.cpu.ewma(0.5)
        ewma_mid = self.cpu.ewma(0.2)
        ewma_long = self.cpu.ewma(0.05)
        dev = cpu_last - baseline

        features = [
            cpu_last,
            self.mem.last(),
            self.disk.last(),
            self.net.last(),
            cpu_var,
            cpu_slope,
        ]
        model_vote = self._onnx_vote(features)
        pattern_score = self._pattern_influence()
        integrity_penalty = 1.0 - self.brain.integrity.integrity_score
        world_pattern_score = self._world_pattern_score(cpu_mean, net_mean)
        horizon_vote = self._event_horizon_vote(cpu_mean, net_mean)

        meta_bias_map = {
            "Sentinel": 0.5,
            "Hyper-Flow": 0.2,
            "Deep-Dream": 0.3,
            "Recovery-Flow": 0.8,
        }
        meta_bias = meta_bias_map.get(self.brain.meta_state.name, 0.5)

        votes: Dict[str, float] = {}
        votes["ewma_short"] = clamp01(ewma_short)
        votes["ewma_mid"] = clamp01(ewma_mid)
        votes["ewma_long"] = clamp01(ewma_long)
        votes["trend"] = clamp01(0.5 + cpu_slope)
        votes["variance"] = clamp01(cpu_var * 10.0)
        votes["turbulence"] = turbulence
        votes["baseline_dev"] = clamp01(abs(dev))
        votes["reinforcement"] = self.brain.reinforcement.get("stability", 0.5)
        votes["onnx_model"] = model_vote
        votes["pattern_memory"] = pattern_score
        votes["integrity"] = integrity_penalty
        votes["meta_state"] = meta_bias
        votes["world_pattern"] = world_pattern_score
        votes["event_horizon"] = horizon_vote

        if self.b4b_provider:
            votes["b4b_stress"] = self.b4b_provider.physics_stress
            votes["b4b_server"] = self.b4b_provider.server_stress
            votes["b4b_outcome"] = self._b4b_outcome_score()
            predicted_outcome = clamp01(
                0.4 * self.b4b_provider.physics_stress +
                0.3 * self.b4b_provider.server_stress +
                0.3 * self.b4b_provider.movement_intensity
            )
            votes["b4b_predicted_outcome"] = predicted_outcome
        else:
            votes["b4b_stress"] = 0.0
            votes["b4b_server"] = 0.0
            votes["b4b_outcome"] = 0.5
            votes["b4b_predicted_outcome"] = 0.5

        game_id = getattr(self.brain, "current_game_id", None)
        if game_id:
            game_profile_bias = GAME_PROFILES.get(game_id, GAME_PROFILES["generic"]).get("baseline_risk", 0.3)
        else:
            game_profile_bias = 0.3
        votes["game_profile_bias"] = clamp01(game_profile_bias)

        if game_id:
            game_load_risk = clamp01(self.cpu.last() * 0.5 + self.mem.last() * 0.5)
        else:
            game_load_risk = 0.0
        votes["game_load_risk"] = game_load_risk

        if game_id:
            game_pattern_id = f"game:{game_id}:{int(cpu_mean*10)}:{int(net_mean*10)}"
            entry = self.brain.pattern_memory.patterns.get(game_pattern_id)
            if entry:
                raw = (entry.overload_count - entry.stable_count * 0.5 - entry.beast_win_count * 0.8) / 10.0
                centered = 0.5 + raw
                game_pattern_score = max(0.2, min(0.8, centered))
            else:
                game_pattern_score = 0.5
        else:
            game_pattern_score = 0.5
        votes["game_pattern"] = game_pattern_score

        live_event_bias = self._live_event_score()
        votes["live_event_bias"] = live_event_bias

        hour, weekday = local_time_components()
        cpu_band = int(cpu_mean * 10)
        net_band = int(net_mean * 10)
        live_pattern_id = f"live:{weekday}:{hour}:{cpu_band}:{net_band}"
        entry = self.brain.pattern_memory.patterns.get(live_pattern_id)
        if entry:
            raw = (entry.overload_count - entry.stable_count * 0.5 - entry.beast_win_count * 0.8) / 10.0
            centered = 0.5 + raw
            live_pattern_score = max(0.2, min(0.8, centered))
        else:
            live_pattern_score = 0.5
        votes["live_event_pattern"] = live_pattern_score

        other_game_active = 1.0 if game_id == "other_game" else 0.0
        votes["other_game_active"] = other_game_active

        game_ai_pressure = getattr(self.brain, "game_ai_pressure", 0.5)
        game_ai_alignment = getattr(self.brain, "game_ai_alignment", 0.5)
        votes["game_ai_pressure"] = clamp01(game_ai_pressure)
        votes["game_ai_alignment"] = clamp01(game_ai_alignment)

        overload_p, stable_p, beast_p = self._outcome_model(votes)
        votes["outcome_overload"] = overload_p
        votes["outcome_stable"] = stable_p
        votes["outcome_beast_win"] = beast_p

        regime = self._regime(cpu_mean, cpu_slope, cpu_var)

        regime_pred: Dict[str, float] = {}
        regime_pred["stable"] = clamp01(0.6 * ewma_mid + 0.4 * self.brain.reinforcement.get("stability", 0.5))
        regime_pred["rising"] = clamp01(0.5 * votes["trend"] + 0.3 * votes["variance"] + 0.2 * votes["baseline_dev"])
        regime_pred["chaotic"] = clamp01(0.5 * votes["turbulence"] + 0.3 * model_vote + 0.2 * integrity_penalty)
        regime_pred["cooling"] = clamp01(0.6 * ewma_long + 0.4 * votes["baseline_dev"])

        regime_weights = {
            "stable": 0.35,
            "rising": 0.25,
            "chaotic": 0.25,
            "cooling": 0.15,
        }

        weights = {
            "ewma_short": 0.08,
            "ewma_mid": 0.10,
            "ewma_long": 0.05,
            "trend": 0.08,
            "variance": 0.06,
            "turbulence": 0.08,
            "baseline_dev": 0.06,
            "reinforcement": 0.08,
            "onnx_model": 0.06,
            "pattern_memory": 0.06,
            "integrity": 0.05,
            "meta_state": 0.04,
            "world_pattern": 0.07,
            "event_horizon": 0.08,
            "b4b_stress": 0.05,
            "b4b_server": 0.03,
            "b4b_outcome": 0.04,
            "b4b_predicted_outcome": 0.06,
            "game_profile_bias": 0.05,
            "game_load_risk": 0.05,
            "game_pattern": 0.06,
            "live_event_bias": 0.05,
            "live_event_pattern": 0.06,
            "other_game_active": 0.03,
            "game_ai_pressure": 0.05,
            "game_ai_alignment": 0.04,
            "outcome_overload": 0.07,
            "outcome_stable": 0.04,
            "outcome_beast_win": 0.04,
        }

        drift = self.brain.integrity.prediction_drift
        drift_clamped = clamp01(drift)

        total = 0.0
        wsum = 0.0
        for k, v in votes.items():
            w = weights.get(k, 0.0)
            if k in ("variance", "turbulence", "baseline_dev", "pattern_memory"):
                w *= (1.0 - 0.5 * drift_clamped)
            if k in ("event_horizon", "world_pattern", "live_event_pattern"):
                w *= (0.7 + 0.3 * drift_clamped)
            total += w * v
            wsum += w
        base_best = total / wsum if wsum > 0 else 0.0
        base_best = clamp01(base_best)

        rpred = regime_pred.get(regime, base_best)
        rweight = regime_weights.get(regime, 0.25)
        fused = clamp01(rweight * rpred + (1.0 - rweight) * base_best)

        return fused, votes, regime, regime_pred

    def _update_meta_state_momentum(self, regime: str):
        risk = self.brain.cortex.risk
        integrity = self.brain.integrity.integrity_score
        pressure = self.brain.game_ai_pressure
        alignment = self.brain.game_ai_alignment

        target = self.brain.meta_state.name

        if risk > 0.8 or integrity < 0.6:
            target = "Recovery-Flow"
        elif regime == "chaotic" and pressure > 0.6:
            target = "Recovery-Flow"
        elif regime in ("rising", "stable") and risk < 0.4 and integrity > 0.8 and alignment > 0.6:
            target = "Hyper-Flow"
        elif regime == "cooling" and risk < 0.3:
            target = "Sentinel"
        else:
            target = self.brain.meta_state.name

        if target == self.brain.meta_state.name:
            self.brain.meta_state.momentum = clamp01(self.brain.meta_state.momentum * 0.95 + 0.05)
        else:
            self.brain.meta_state.momentum = clamp01(self.brain.meta_state.momentum + 0.1)
            if self.brain.meta_state.momentum > 0.7:
                self.brain.meta_state.name = target
                self.brain.meta_state.momentum = 0.3

    def update_forecast(self):
        if not self.cpu.values:
            return

        best_val, components, regime, regime_pred = self._engine_votes()
        cpu_var = self.cpu.var()
        cpu_slope = self.cpu.slope()
        turbulence = clamp01(math.sqrt(cpu_var) * 5.0)

        v_term = 1.0 - clamp01(cpu_var / 0.05)
        s_term = 1.0 - clamp01(abs(cpu_slope) / 0.05)
        t_term = 1.0 - turbulence
        r_term = self.brain.reinforcement.get("stability", 0.5)
        meta_conf = clamp01(0.35 * v_term + 0.2 * s_term + 0.25 * t_term + 0.2 * r_term)

        bg = self.brain.best_guess
        bg.value = best_val
        bg.confidence = meta_conf
        bg.components = components
        bg.regime = regime

        self.brain.cortex.risk = clamp01(best_val * (1.0 + turbulence))
        self.brain.cortex.opportunity = clamp01(1.0 - self.brain.cortex.risk)

        overload_p = components.get("outcome_overload", 0.33)
        stable_p = components.get("outcome_stable", 0.33)
        beast_p = components.get("outcome_beast_win", 0.33)
        self.brain.predicted_overload = overload_p
        self.brain.predicted_stable = stable_p
        self.brain.predicted_beast_win = beast_p

        if overload_p > 0.7:
            self.brain.cortex.environment = "DANGER"
            self.brain.cortex.anticipation = "Imminent overload"
        elif beast_p > 0.6 and self.brain.cortex.risk < 0.5:
            self.brain.cortex.environment = "FLOW"
            self.brain.cortex.anticipation = "Push performance"
        elif self.brain.cortex.risk > 0.5:
            self.brain.cortex.environment = "TENSE"
            self.brain.cortex.anticipation = "Watch spikes"
        else:
            self.brain.cortex.environment = "CALM"
            self.brain.cortex.anticipation = "Normal"

        self.brain.collective.risk = self.brain.cortex.risk

        self.brain.prediction_history.append({
            "ts": utc_ts(),
            "best": best_val,
            "turbulence": turbulence,
            "regime": regime,
        })
        if len(self.brain.prediction_history) > 300:
            self.brain.prediction_history.pop(0)

        max_comp = max(components.values()) if components else 1.0
        self.brain.reasoning_heatmap = {
            k: (v / max_comp if max_comp else 0.0) for k, v in components.items()
        }

        self._update_meta_state_momentum(regime)

# ------------------------ Tri-Stance Engine -------------------

class TriStanceEngine:
    def __init__(self, brain: HybridBrain):
        self.brain = brain

    def tick(self):
        risk = self.brain.cortex.risk
        integrity = self.brain.integrity.integrity_score
        regime = self.brain.best_guess.regime
        stance = self.brain.stance

        if risk > 0.8 or integrity < 0.6 or regime == "chaotic":
            new = "Conservative"
        elif risk < 0.4 and integrity > 0.8 and regime in ("stable", "rising"):
            new = "Beast"
        else:
            new = "Balanced"

        if stance != new:
            self.brain.stance = new

# ------------------------ Restoration Protocol ----------------

class RestorationProtocol:
    def __init__(self, brain: HybridBrain, organs: Dict[str, Any]):
        self.brain = brain
        self.organs = organs
        self.active = False

    def engage(self):
        self.active = True
        self.brain.stance = "Conservative"
        self.brain.meta_state.name = "Recovery-Flow"
        self.brain.meta_state.momentum = 1.0
        self.brain.reinforcement["stability"] = 1.0
        self.brain.reinforcement["performance"] = 0.0

    def tick(self):
        if not self.active:
            return

        self.brain.dynamic_thresholds["risk_high"] = 0.5
        self.brain.dynamic_thresholds["risk_critical"] = 0.7

        stale = []
        for pid, entry in self.brain.pattern_memory.patterns.items():
            if entry.overload_count == 0 and entry.stable_count == 0 and entry.beast_win_count == 0:
                stale.append(pid)
        for pid in stale:
            del self.brain.pattern_memory.patterns[pid]

        for o in self.organs.values():
            if hasattr(o, "apply_recovery"):
                o.apply_recovery("Conservative", self.brain.cortex.risk)

        if self.brain.cortex.risk < 0.3 and self.brain.integrity.integrity_score > 0.8:
            self.active = False
            self.brain.meta_state.name = "Sentinel"
            self.brain.stance = "Balanced"
            self.brain.dynamic_thresholds["risk_high"] = 0.7
            self.brain.dynamic_thresholds["risk_critical"] = 0.9

# ------------------------ Nerve Center GUI --------------------

class NerveCenterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("System Nerve Center – HybridBrain Tri-Stance (Event Horizon)")
        self.geometry("1350x780")

        self.brain = HybridBrain()
        self.predictor = PredictionEngine(self.brain)
        self.tri_stance = TriStanceEngine(self.brain)

        self.organs = {
            "deep_ram": DeepRamOrgan(),
            "network": NetworkWatcherOrgan(),
            "disk": DiskOrgan(),
            "thermal": ThermalOrgan(),
            "gpu_cache": GPUCacheOrgan(),
            "vram": VRAMOrgan(),
            "ai_coach": AICoachOrgan(),
            "swarm": SwarmNodeOrgan(),
            "backup": BackupEngineOrgan(),
        }

        self.game_detector = GameDetector()
        self.back4blood_analyzer = Back4BloodAnalyzer(self.brain)
        self.back4blood_physics = Back4BloodPhysicsOrgan(self.back4blood_analyzer, self.game_detector)

        self.interaction_coach = InteractionCoachOrgan(
            self.brain,
            self.back4blood_analyzer,
            self.back4blood_physics,
            self.game_detector,
        )

        self.game_ai_bridge = GameAIConversationBridge(
            self.brain,
            self.game_detector,
            self.back4blood_physics,
        )

        self.predictor.b4b_provider = self.back4blood_physics

        self.integrity_organ = SelfIntegrityOrgan(
            predictor_windows={"cpu": self.predictor.cpu, "mem": self.predictor.mem},
            organs=self.organs,
            brain=self.brain,
        )

        self.restoration = RestorationProtocol(self.brain, self.organs)

        self.reboot_memory_path: str = ""
        self.reboot_autoload: bool = False

        self.last_horizon_key: Optional[str] = None

        self._build_gui()
        self._load_settings()
        if self.reboot_autoload:
            self._try_load_reboot_memory()

        self.conv_freeze = False

        self._start_background()
        self._schedule_gui_refresh()

    def _build_gui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = tk.Frame(self, bg="#111111", height=40)
        top.pack(fill="x", side="top")
        self.lbl_top_status = tk.Label(
            top,
            text="Nerve Center Online (Event Horizon)",
            fg="#00ff99",
            bg="#111111",
            font=("Consolas", 12, "bold"),
        )
        self.lbl_top_status.pack(side="left", padx=10)

        self.btn_restoration = tk.Button(
            top,
            text="Serve Again Mode",
            bg="#661111",
            fg="#ffdddd",
            command=self._engage_restoration,
        )
        self.btn_restoration.pack(side="right", padx=10)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_cortex = ttk.Frame(notebook)
        self.tab_reboot = ttk.Frame(notebook)
        self.tab_reason = ttk.Frame(notebook)
        self.tab_b4b = ttk.Frame(notebook)
        self.tab_onnx = ttk.Frame(notebook)
        self.tab_conv = ttk.Frame(notebook)
        self.tab_horizon = ttk.Frame(notebook)

        notebook.add(self.tab_cortex, text="Brain Cortex")
        notebook.add(self.tab_reboot, text="Reboot Memory")
        notebook.add(self.tab_reason, text="Reasoning Heatmap")
        notebook.add(self.tab_b4b, text="Back4Blood / Games")
        notebook.add(self.tab_onnx, text="ONNX / Movidius")
        notebook.add(self.tab_conv, text="Conversation")
        notebook.add(self.tab_horizon, text="Horizon Patterns")

        self._build_cortex_tab(self.tab_cortex)
        self._build_reboot_tab(self.tab_reboot)
        self._build_reason_tab(self.tab_reason)
        self._build_b4b_tab(self.tab_b4b)
        self._build_onnx_tab(self.tab_onnx)
        self._build_conv_tab(self.tab_conv)
        self._build_horizon_tab(self.tab_horizon)

    def _build_cortex_tab(self, parent):
        frame_top = ttk.LabelFrame(parent, text="Brain State")
        frame_top.pack(fill="x", padx=10, pady=5)

        self.lbl_meta_state = ttk.Label(frame_top, text="Meta-state: Sentinel")
        self.lbl_meta_state.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.lbl_stance = ttk.Label(frame_top, text="Stance: Balanced")
        self.lbl_stance.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        self.lbl_best = ttk.Label(frame_top, text="Best-Guess: 0.00 (conf 0.00)")
        self.lbl_best.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.lbl_env = ttk.Label(frame_top, text="Env: CALM | Risk: 0.00 | Integrity: 1.00")
        self.lbl_env.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        self.lbl_conv = ttk.Label(frame_top, text="AI Link: neutral (0.50)")
        self.lbl_conv.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        self.lbl_outcome = ttk.Label(frame_top, text="Outcome: overload=0.00 stable=0.00 beast=0.00")
        self.lbl_outcome.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        frame_chart = ttk.LabelFrame(parent, text="Prediction Micro-chart")
        frame_chart.pack(fill="x", padx=10, pady=10)
        self.canvas_chart = tk.Canvas(frame_chart, bg="#111111", height=180)
        self.canvas_chart.pack(fill="x", padx=5, pady=5)

    def _build_reboot_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Reboot Memory Persistence")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="SMB / UNC Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame, text="Pick SMB Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame, text="Test SMB Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame,
            text="Load memory from SMB on startup",
            variable=self.var_reboot_autoload,
            command=self.cmd_toggle_autoload,
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame, text="Status: Ready", anchor="w", fg="#00cc66")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    def _build_reason_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Reasoning Heatmap")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_heatmap = tk.Text(frame, bg="#050505", fg="#d0d0d0", font=("Consolas", 9))
        self.txt_heatmap.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_b4b_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Game Analyzer, Physics & Interaction Coach")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="B4B Metrics JSON Path:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.entry_b4b_path = ttk.Entry(frame, width=60)
        self.entry_b4b_path.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(frame, text="Browse", command=self._pick_b4b_path).grid(row=0, column=2, padx=5, pady=3)

        self.lbl_b4b_last = ttk.Label(frame, text="Last B4B metrics: (none)")
        self.lbl_b4b_last.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=3)

        self.lbl_b4b_phys = ttk.Label(frame, text="Physics: stress=0.00, move=0.00, server=0.00")
        self.lbl_b4b_phys.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=3)

        self.lbl_b4b_interact = ttk.Label(frame, text="Episodes: 0, Stability reinforcement=0.50")
        self.lbl_b4b_interact.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=3)

        self.lbl_b4b_game = ttk.Label(frame, text="Game: Not running", foreground="#ff5555")
        self.lbl_b4b_game.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=3)

    def _build_onnx_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="ONNX / Movidius Training")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Dataset/log path:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.entry_ds_path = ttk.Entry(frame, width=50)
        self.entry_ds_path.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(frame, text="Browse", command=self._pick_ds_path).grid(row=0, column=2, padx=5, pady=3)

        ttk.Label(frame, text="Output ONNX path:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.entry_onnx_out = ttk.Entry(frame, width=50)
        self.entry_onnx_out.grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(frame, text="Browse", command=self._pick_onnx_out).grid(row=1, column=2, padx=5, pady=3)

        self.btn_train_onnx = ttk.Button(frame, text="Train ONNX Model", command=self._train_onnx)
        self.btn_train_onnx.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.lbl_onnx_status = ttk.Label(frame, text="Status: Waiting", foreground="#00cc66")
        self.lbl_onnx_status.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=5)

    def _build_conv_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Game AI ↔ System AI Conversation (last 20 messages)")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.txt_conv = tk.Text(frame, bg="#050505", fg="#d0d0d0", font=("Consolas", 9))
        self.txt_conv.pack(fill="both", expand=True, padx=5, pady=5)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", padx=5, pady=5)

        self.btn_conv_freeze = ttk.Button(btn_frame, text="Freeze", command=self._toggle_conv_freeze)
        self.btn_conv_freeze.pack(side="left", padx=5)

    def _build_horizon_tab(self, tab):
        frame = ttk.LabelFrame(tab, text="Event Horizon Future‑Risk Map")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.txt_horizon = tk.Text(
            frame,
            bg="#050505",
            fg="#d0d0d0",
            font=("Consolas", 9)
        )
        self.txt_horizon.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(
            frame,
            text="Shows predicted future‑risk by game / stance / meta / time."
        ).pack(anchor="w", padx=5, pady=5)

    def _toggle_conv_freeze(self):
        self.conv_freeze = not self.conv_freeze
        self.btn_conv_freeze.config(text="Unfreeze" if self.conv_freeze else "Freeze")

    def _engage_restoration(self):
        self.restoration.engage()
        self.lbl_top_status.config(text="Serve Again Mode: Restoration Active", fg="#ffdd55")

    def _pick_b4b_path(self):
        path = filedialog.askopenfilename(
            title="Select Back4Blood metrics JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.entry_b4b_path.delete(0, tk.END)
            self.entry_b4b_path.insert(0, path)
            self.back4blood_analyzer.set_metrics_path(path)

    def _pick_ds_path(self):
        path = filedialog.askopenfilename(
            title="Select dataset/log file",
            filetypes=[("All files", "*.*")]
        )
        if path:
            self.entry_ds_path.delete(0, tk.END)
            self.entry_ds_path.insert(0, path)

    def _pick_onnx_out(self):
        path = filedialog.asksaveasfilename(
            title="Select output ONNX path",
            defaultextension=".onnx",
            filetypes=[("ONNX models", "*.onnx"), ("All files", "*.*")]
        )
        if path:
            self.entry_onnx_out.delete(0, tk.END)
            self.entry_onnx_out.insert(0, path)

    def _train_onnx(self):
        ds = self.entry_ds_path.get().strip()
        out = self.entry_onnx_out.get().strip()
        if not ds or not out:
            self.lbl_onnx_status.config(text="Status: Missing paths", foreground="#ff5555")
            return
        try:
            self.predictor.trainer.train_from_logs(ds, out)
            self.lbl_onnx_status.config(
                text=f"Status: Training stub called at {time.strftime('%H:%M:%S')}",
                foreground="#00cc66",
            )
        except Exception as e:
            self.lbl_onnx_status.config(text=f"Status: Error {e}", foreground="#ff5555")

    def _get_reboot_files(self) -> Tuple[str, str]:
        base = self.reboot_memory_path or self.entry_reboot_path.get().strip()
        if not base:
            base = os.path.join(os.path.expanduser("~"), "nerve_center_memory.json")
        if base.endswith(".json"):
            state_path = base
            meta_path = base.replace(".json", "_meta.json")
        else:
            state_path = os.path.join(base, "nerve_center_state.json")
            meta_path = os.path.join(base, "nerve_center_meta.json")
        return state_path, meta_path

    def cmd_pick_reboot_path(self):
        path = filedialog.asksaveasfilename(
            title="Pick reboot memory file or path",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)
            self.reboot_memory_path = path
            self._set_status("Status: Path selected", ok=True)

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            self._set_status("Status: No path set", ok=False)
            return
        directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
        if os.path.exists(directory):
            self._set_status("Status: Path OK", ok=True)
        else:
            self._set_status("Status: Path does not exist", ok=False)

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            path = os.path.join(os.path.expanduser("~"), "nerve_center_memory.json")
            self.entry_reboot_path.insert(0, path)
        self.reboot_memory_path = path

        state_path, meta_path = self._get_reboot_files()
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
        except Exception:
            pass

        state: Dict[str, Any] = {
            "brain": asdict(self.brain),
            "organs": {k: self._organ_state(v) for k, v in self.organs.items()},
        }

        b4b_state: Dict[str, Any] = {
            "last_metrics": self.back4blood_analyzer.last_metrics,
            "physics": {
                "physics_stress": self.back4blood_physics.physics_stress,
                "movement_intensity": self.back4blood_physics.movement_intensity,
                "server_stress": self.back4blood_physics.server_stress,
                "outcome": self.back4blood_physics.outcome,
            },
            "episodes": [
                asdict(ep) for ep in self.interaction_coach.episodes
            ],
            "game_state": {
                "running": self.game_detector.game_running,
                "proc_info": self.game_detector.proc_info,
                "game_id": self.game_detector.current_game_id,
                "platform": self.game_detector.platform,
            },
        }
        state["b4b"] = b4b_state

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        meta = {"ts": time.time(), "version": "4.1-event-horizon-horizon-tab"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._set_status(f"Status: Memory saved to {state_path}", ok=True)
        self._save_settings()

    def _organ_state(self, organ):
        return {k: v for k, v in organ.__dict__.items() if not callable(v)}

    def _try_load_reboot_memory(self):
        state_path, _ = self._get_reboot_files()
        if not os.path.exists(state_path):
            self._set_status("Status: No memory file found", ok=False)
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            brain_data = data.get("brain", {})
            self.brain.baseline_risk = brain_data.get("baseline_risk", 0.2)
            self.brain.dynamic_thresholds = brain_data.get("dynamic_thresholds", self.brain.dynamic_thresholds)
            self.brain.reinforcement = brain_data.get("reinforcement", self.brain.reinforcement)

            pm = brain_data.get("pattern_memory", {})
            pat = pm.get("patterns", {})
            horizon = pm.get("horizon_patterns", {})
            self.brain.pattern_memory = PatternMemory(
                patterns={k: PatternMemoryEntry(**v) for k, v in pat.items()},
                horizon_patterns={k: HorizonPatternEntry(**v) for k, v in horizon.items()},
            )

            b4b_data = data.get("b4b", {})
            if b4b_data:
                self.back4blood_analyzer.last_metrics = b4b_data.get("last_metrics", {})
                phys = b4b_data.get("physics", {})
                self.back4blood_physics.physics_stress = float(phys.get("physics_stress", 0.0))
                self.back4blood_physics.movement_intensity = float(phys.get("movement_intensity", 0.0))
                self.back4blood_physics.server_stress = float(phys.get("server_stress", 0.0))
                self.back4blood_physics.outcome = phys.get("outcome", "unknown")

                eps_raw = b4b_data.get("episodes", [])
                episodes: List[InteractionEpisode] = []
                for e in eps_raw:
                    try:
                        episodes.append(InteractionEpisode(**e))
                    except Exception:
                        continue
                self.interaction_coach.episodes = episodes

                game_state = b4b_data.get("game_state", {})
                self.game_detector.game_running = bool(game_state.get("running", False))
                self.game_detector.proc_info = game_state.get("proc_info", None)
                self.game_detector.current_game_id = game_state.get("game_id", None)
                self.game_detector.platform = game_state.get("platform", None)
                self.game_detector.last_state = self.game_detector.game_running
                self.brain.current_game_id = self.game_detector.current_game_id

            self._set_status(f"Status: Memory loaded from {state_path}", ok=True)
        except Exception as e:
            self._set_status(f"Status: Load failed: {e}", ok=False)

    def cmd_toggle_autoload(self):
        self.reboot_autoload = self.var_reboot_autoload.get()
        self._set_status(f"Status: Autoload {'enabled' if self.reboot_autoload else 'disabled'}", ok=True)
        self._save_settings()

    def _set_status(self, text: str, ok: bool = True):
        self.lbl_reboot_status.config(text=text, fg="#00cc66" if ok else "#ff5555")

    def _save_settings(self):
        s = {
            "reboot_autoload": self.reboot_autoload,
            "reboot_path": self.entry_reboot_path.get().strip(),
        }
        path = os.path.join(os.path.expanduser("~"), ".nerve_center_settings_event_horizon.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        path = os.path.join(os.path.expanduser("~"), ".nerve_center_settings_event_horizon.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.reboot_autoload = bool(s.get("reboot_autoload", False))
            self.var_reboot_autoload.set(self.reboot_autoload)
            p = s.get("reboot_path", "")
            if p:
                self.entry_reboot_path.insert(0, p)
                self.reboot_memory_path = p
        except Exception:
            pass

    def _start_background(self):
        t = threading.Thread(target=self._bg_loop, daemon=True)
        t.start()

    def _current_horizon_key(self) -> str:
        hour, weekday = local_time_components()
        cpu_mean = self.predictor.cpu.mean()
        net_mean = self.predictor.net.mean()
        cpu_band = int(cpu_mean * 10)
        net_band = int(net_mean * 10)
        gid = self.game_detector.current_game_id or "none"
        stance = self.brain.stance
        meta = self.brain.meta_state.name
        return f"horizon:{gid}:{stance}:{meta}:{weekday}:{hour}:{cpu_band}:{net_band}"

    def _bg_loop(self):
        while True:
            try:
                running, info = self.game_detector.tick()
                if running:
                    gid = self.game_detector.current_game_id or "generic"
                    self.brain.current_game_id = gid
                    if not self.game_detector.last_state:
                        profile = GAME_PROFILES.get(gid, GAME_PROFILES["generic"])
                        self.brain.baseline_risk = profile.get("baseline_risk", self.brain.baseline_risk)
                        disp = GAME_DISPLAY_NAMES.get(gid, gid)
                        print(f"[GameDetector] Game {disp} detected — profile loaded, auto-training engaged")
                        self.lbl_top_status.config(
                            text=f"Game {disp} detected — Training Active",
                            fg="#55ff55",
                        )
                else:
                    if self.game_detector.last_state:
                        print("[GameDetector] Game closed — training paused")
                        self.lbl_top_status.config(
                            text="Game closed — Training Paused",
                            fg="#ffaa00",
                        )
                    self.brain.current_game_id = None
                self.game_detector.last_state = running

                self.back4blood_analyzer.tick()
                self.back4blood_physics.tick()

                hour, weekday = local_time_components()
                cpu_mean = self.predictor.cpu.mean()
                net_mean = self.predictor.net.mean()
                cpu_band = int(cpu_mean * 10)
                net_band = int(net_mean * 10)
                if self.brain.cortex.risk > 0.8:
                    world_outcome = "overload"
                elif self.brain.cortex.risk < 0.4:
                    world_outcome = "stable"
                else:
                    world_outcome = "beast_win"

                world_pattern_id = f"world:{weekday}:{hour}:{cpu_band}:{net_band}"
                self.brain.pattern_memory.update(world_pattern_id, world_outcome)

                if self.game_detector.game_running:
                    gid = self.game_detector.current_game_id or "generic"
                    game_pattern_id = f"game:{gid}:{cpu_band}:{net_band}"
                    self.brain.pattern_memory.update(game_pattern_id, world_outcome)

                live_pattern_id = f"live:{weekday}:{hour}:{cpu_band}:{net_band}"
                self.brain.pattern_memory.update(live_pattern_id, world_outcome)

                for o in self.organs.values():
                    if hasattr(o, "tick"):
                        o.tick()

                self.predictor.ingest_sample()
                self.predictor.update_forecast()

                current_key = self._current_horizon_key()
                if self.last_horizon_key is not None and self.last_horizon_key != current_key:
                    self.brain.pattern_memory.update_horizon(self.last_horizon_key, self.brain.cortex.risk)
                self.last_horizon_key = current_key

                self.integrity_organ.tick()
                self.tri_stance.tick()
                self.restoration.tick()

                self.interaction_coach.tick(self.restoration.active)

                game_msg, sys_reply = self.game_ai_bridge.conversation_step()
                intent = (game_msg.get("intent") or "neutral").lower()
                if intent in ("aggressive", "pressure_player"):
                    pressure = 0.8
                elif intent == "recover_and_scale":
                    pressure = 0.4
                else:
                    pressure = 0.5
                self.brain.game_ai_pressure = pressure
                self.brain.game_ai_alignment = self.game_ai_bridge.alignment_score()

                stance = self.brain.stance
                risk = self.brain.cortex.risk
                for o in self.organs.values():
                    if hasattr(o, "apply_recovery"):
                        o.apply_recovery(stance, risk)

            except Exception as e:
                print(f"[BG] error: {e}", file=sys.stderr)
            time.sleep(1.0)

    def _schedule_gui_refresh(self):
        self._refresh_gui()
        self.after(500, self._schedule_gui_refresh)

    def _refresh_gui(self):
        b = self.brain
        self.lbl_meta_state.config(text=f"Meta-state: {b.meta_state.name} (mom {b.meta_state.momentum:.2f})")
        self.lbl_stance.config(text=f"Stance: {b.stance}")
        self.lbl_best.config(text=f"Best-Guess: {b.best_guess.value:.2f} (conf {b.best_guess.confidence:.2f})")
        self.lbl_env.config(
            text=f"Env: {b.cortex.environment} | Risk: {b.cortex.risk:.2f} | Integrity: {b.integrity.integrity_score:.2f}"
        )
        align = self.brain.game_ai_alignment
        if align > 0.7:
            conv_text = f"AI Link: aligned ({align:.2f})"
        elif align < 0.4:
            conv_text = f"AI Link: tense ({align:.2f})"
        else:
            conv_text = f"AI Link: neutral ({align:.2f})"
        self.lbl_conv.config(text=conv_text)

        self.lbl_outcome.config(
            text=f"Outcome: overload={b.predicted_overload:.2f} stable={b.predicted_stable:.2f} beast={b.predicted_beast_win:.2f}"
        )

        top_color = "#00ff99"
        if self.restoration.active:
            top_color = "#ffdd55"
        elif b.cortex.risk > 0.8:
            top_color = "#ff4444"
        elif b.cortex.risk > 0.5:
            top_color = "#ffaa00"
        self.lbl_top_status.config(fg=top_color)
        self._draw_micro_chart()
        self._update_heatmap()
        self._update_b4b_panel()
        self._update_conv_panel()
        self._update_horizon_panel()

    def _draw_micro_chart(self):
        canvas = self.canvas_chart
        canvas.delete("all")
        w = int(canvas.winfo_width() or 800)
        h = int(canvas.winfo_height() or 180)
        ph = self.brain.prediction_history[-120:]
        if not ph:
            return

        def norm(v): return clamp01(v)
        n = len(ph)
        x_step = max(1, w // max(1, n - 1))

        def build_points(key):
            pts = []
            for i, p in enumerate(ph):
                val = norm(float(p.get(key, 0.0)))
                x = i * x_step
                y = h - int(val * (h - 10)) - 5
                pts.append((x, y))
            return pts

        def line(points, color, width=1):
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        baseline_val = self.brain.baseline_risk
        pts_best = build_points("best")

        line(pts_best, "#00ffff", 1)
        line(pts_best, "#ffff00", 1)
        y_base = h - int(baseline_val * (h - 10)) - 5
        canvas.create_line(0, y_base, w, y_base, fill="#888888", width=1)
        line(pts_best, "#ff8800", 1)
        line(pts_best, "#ff00ff", 2)

        canvas.create_line(0, h - 5, w, h - 5, fill="#555555")
        canvas.create_text(5, 10, text="High risk", anchor="nw", fill="#aaaaaa", font=("TkDefaultFont", 8))
        canvas.create_text(5, h - 15, text="Low risk", anchor="nw", fill="#aaaaaa", font=("TkDefaultFont", 8))

    def _update_heatmap(self):
        hm = self.brain.reasoning_heatmap
        self.txt_heatmap.delete("1.0", "end")
        if not hm:
            self.txt_heatmap.insert("end", "No reasoning data yet.\n")
            return
        self.txt_heatmap.insert("end", "Reasoning Heatmap (normalized engine influence):\n\n")
        for k, v in sorted(hm.items(), key=lambda kv: kv[1], reverse=True):
            bar = "#" * int(v * 20)
            self.txt_heatmap.insert("end", f"{k:25s} [{bar:<20s}] {v:.2f}\n")

    def _update_b4b_panel(self):
        m = self.back4blood_analyzer.last_metrics
        if not m:
            self.lbl_b4b_last.config(text="Last B4B metrics: (none)")
            self.lbl_b4b_phys.config(text="Physics: stress=0.00, move=0.00, server=0.00")
            self.lbl_b4b_interact.config(text="Episodes: 0, Stability reinforcement=0.50")
        else:
            out = m.get("outcome", "unknown")
            fps = m.get("fps", "?")
            ping = m.get("ping", "?")
            self.lbl_b4b_last.config(
                text=f"Last B4B metrics: outcome={out}, fps={fps}, ping={ping}"
            )
            p = self.back4blood_physics
            self.lbl_b4b_phys.config(
                text=f"Physics: stress={p.physics_stress:.2f}, move={p.movement_intensity:.2f}, server={p.server_stress:.2f}"
            )
            coach = self.interaction_coach
            eps = len(coach.episodes)
            stab_bias = self.brain.reinforcement.get("stability", 0.5)
            self.lbl_b4b_interact.config(
                text=f"Episodes: {eps}, Stability reinforcement={stab_bias:.2f}"
            )

        if self.game_detector.game_running:
            info = self.game_detector.proc_info or {}
            gid = self.game_detector.current_game_id or "unknown"
            disp = GAME_DISPLAY_NAMES.get(gid, gid)
            self.lbl_b4b_game.config(
                text=f"Game: {disp} RUNNING (pid={info.get('pid','?')}, cpu={info.get('cpu','?')}%)",
                foreground="#00cc66",
            )
        else:
            self.lbl_b4b_game.config(text="Game: Not running", foreground="#ff5555")

    def _update_conv_panel(self):
        if self.conv_freeze:
            return
        self.txt_conv.delete("1.0", "end")
        for speaker, msg in self.game_ai_bridge.transcript:
            ts = msg.get("ts", utc_ts())
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
            self.txt_conv.insert("end", f"[{ts_str}] {speaker} →\n")
            self.txt_conv.insert("end", json.dumps(msg, indent=2) + "\n\n")
        self.txt_conv.see("end")

    def _update_horizon_panel(self):
        pm = self.brain.pattern_memory.horizon_patterns
        self.txt_horizon.delete("1.0", "end")

        if not pm:
            self.txt_horizon.insert("end", "No horizon patterns learned yet.\n")
            return

        self.txt_horizon.insert("end", "Horizon Pattern Memory (future‑risk projections):\n\n")

        sorted_items = sorted(
            pm.items(),
            key=lambda kv: kv[1].avg_future_risk,
            reverse=True
        )

        for pid, entry in sorted_items:
            bar = "#" * int(entry.avg_future_risk * 20)
            self.txt_horizon.insert(
                "end",
                f"{pid}\n"
                f"  future_risk={entry.avg_future_risk:.2f} [{bar:<20s}]\n"
                f"  overload={entry.future_overload}  "
                f"stable={entry.future_stable}  "
                f"beast={entry.future_beast_win}  "
                f"samples={entry.samples}\n\n"
            )

# ---------------- main ----------------

if __name__ == "__main__":
    app = NerveCenterApp()
    app.mainloop()

