#!/usr/bin/env python3
"""
Hybrid: Living Game Optimizer + Genesis Node v3 (Tkinter, Single File, Ultra‑Permissive Detection)

- Optimizer (body):
    - Tkinter optimizer panel (left)
    - system metrics, foreground-process-as-game detection
    - foresight risk + posture engine (Calm / Engaged / Redline)
    - per-game profiles + optional network storage
    - overlay HUD (Tkinter Toplevel)

- Genesis (co-pilot brain):
    - memory, traits, posture, consciousness state
    - intent engine + rule engine + self-tuning scheduler
    - plugin system + narrative mindstream
    - Tkinter notebook (right side)
"""

import os
import sys
import json
import time
import threading
import queue
import importlib.util
import traceback
import platform
import locale
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import psutil
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Optional GPU sensor
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# --- Windows-specific imports for foreground window ---
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
GetForegroundWindow = user32.GetForegroundWindow
GetWindowThreadProcessId = user32.GetWindowThreadProcessId

# -------------------------------------------------------------------
# CONFIG (Optimizer)
# -------------------------------------------------------------------

APP_NAME = "Hybrid Game Optimizer + Genesis Co-Pilot"

LOCAL_ROOT = Path(os.getenv("APPDATA", Path.home())) / "GameOptimizer"
NETWORK_ROOT_ENV = os.getenv("GAME_OPTIMIZER_SMB_ROOT", "")

LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

GAME_DIR_HINTS = [
    r"\Steam\steamapps\common",
    r"\Epic Games\\",
    r"\GOG Galaxy\Games",
    r"\Riot Games\\",
    r"\Origin Games\\",
    r"\Battle.net\\",
]

# Foreground process names we consider "never a game"
SYSTEM_PROCESS_DENYLIST = {
    "explorer.exe",
    "shellexperiencehost.exe",
    "searchui.exe",
    "searchapp.exe",
    "ctfmon.exe",
    "dwm.exe",
    "systemsettings.exe",
    "taskmgr.exe",
    "python.exe",
    "pythonw.exe",
    "cmd.exe",
    "powershell.exe",
    "wt.exe",
    "mmc.exe",
    "conhost.exe",
    "msedge.exe",
    "chrome.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "notepad.exe",
    "code.exe",
    "devenv.exe",
}

# -------------------------------------------------------------------
# CONFIG (Genesis)
# -------------------------------------------------------------------

MEMORY_FILE = str(LOCAL_ROOT / "genesis_memory.json")
PLUGINS_DIR = str(LOCAL_ROOT / "plugins")
BASE_SIM_TICK_SECONDS = 0.2

POSTURE_MODES = {
    "calm": 1.5,
    "focused": 1.0,
    "aggressive": 0.6,
}

CONSCIOUSNESS_STATES = {
    "baseline": {
        "narrative_prefix": "",
        "thought_interval_factor": 1.0,
        "verbosity_bonus": 0.0,
    },
    "flow": {
        "narrative_prefix": "[flow] ",
        "thought_interval_factor": 0.7,
        "verbosity_bonus": 0.1,
    },
    "trance": {
        "narrative_prefix": "[trance] ",
        "thought_interval_factor": 0.5,
        "verbosity_bonus": 0.2,
    },
    "overload": {
        "narrative_prefix": "[overload] ",
        "thought_interval_factor": 1.5,
        "verbosity_bonus": -0.2,
    },
}

DEFAULT_CONSCIOUSNESS = "baseline"

# -------------------------------------------------------------------
# Logger (shared)
# -------------------------------------------------------------------

class Logger:
    def __init__(self):
        self._listeners: List[Callable[[str], None]] = []

    def register(self, fn: Callable[[str], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def log(self, msg: str) -> None:
        text = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(text)
        for fn in list(self._listeners):
            try:
                fn(text)
            except Exception:
                pass

logger = Logger()

# -------------------------------------------------------------------
# Genesis Memory layer
# -------------------------------------------------------------------

class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "traits": {},
            "facts": {},
            "log": [],
            "narrative": [],
            "plugins": {},
            "metrics": {
                "ticks": 0,
                "heartbeats": 0,
                "thoughts": 0,
                "intents_created": 0,
                "intents_approved": 0,
                "intents_rejected": 0,
            },
            "posture": "focused",
            "consciousness_state": DEFAULT_CONSCIOUSNESS,
            "fluidity": 0.5,
            "pressure": 0.5,
            "intent_history": [],
        }
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            logger.log(f"[Genesis] Memory file not found, initializing new store at {self.path}")
            self._save_internal()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.log(f"[Genesis] Loaded memory from {self.path}")
        except Exception as e:
            logger.log(f"[Genesis] Error loading memory: {e}. Starting fresh.")
            self._save_internal()

    def _save_internal(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def save(self) -> None:
        with self._lock:
            self._save_internal()

    def set_trait(self, key: str, value: Any) -> None:
        with self._lock:
            self.data.setdefault("traits", {})[key] = value
            self._save_internal()

    def get_trait(self, key: str, default: Any = None) -> Any:
        return self.data.get("traits", {}).get(key, default)

    def set_posture(self, posture: str) -> None:
        if posture not in POSTURE_MODES:
            return
        with self._lock:
            self.data["posture"] = posture
            self._save_internal()

    def get_posture(self) -> str:
        return self.data.get("posture", "focused")

    def set_consciousness(self, state: str) -> None:
        if state not in CONSCIOUSNESS_STATES:
            return
        with self._lock:
            self.data["consciousness_state"] = state
            self._save_internal()

    def get_consciousness(self) -> str:
        return self.data.get("consciousness_state", DEFAULT_CONSCIOUSNESS)

    def set_fluidity(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        with self._lock:
            self.data["fluidity"] = value
            self._save_internal()

    def get_fluidity(self) -> float:
        return float(self.data.get("fluidity", 0.5))

    def set_pressure(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        with self._lock:
            self.data["pressure"] = value
            self._save_internal()

    def get_pressure(self) -> float:
        return float(self.data.get("pressure", 0.5))

    def remember_fact(self, key: str, value: Any) -> None:
        with self._lock:
            self.data.setdefault("facts", {})[key] = value
            self._save_internal()

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.data.get("facts", {}).get(key, default)

    def append_log(self, entry: str) -> None:
        with self._lock:
            self.data.setdefault("log", []).append({"ts": time.time(), "entry": entry})
            if len(self.data["log"]) > 2000:
                self.data["log"] = self.data["log"][-2000:]
            self._save_internal()

    def append_narrative(self, entry: str) -> None:
        with self._lock:
            self.data.setdefault("narrative", []).append({"ts": time.time(), "entry": entry})
            if len(self.data["narrative"]) > 2000:
                self.data["narrative"] = self.data["narrative"][-2000:]
            self._save_internal()

    def get_narrative(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.data.get("narrative", []))

    def inc_metric(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self.data.setdefault("metrics", {})
            self.data["metrics"][key] = self.data["metrics"].get(key, 0) + delta
            self._save_internal()

    def get_metrics(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.data.get("metrics", {}))

    def set_plugin_state(self, name: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("plugins", {})[name] = state
            self._save_internal()

    def get_plugin_state(self, name: str, default: Any = None) -> Any:
        return self.data.get("plugins", {}).get(name, default)

    def add_intent_history(self, intent: Dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault("intent_history", []).append(intent)
            if len(self.data["intent_history"]) > 500:
                self.data["intent_history"] = self.data["intent_history"][-500:]
            self._save_internal()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.data))

# -------------------------------------------------------------------
# Genesis Event bus
# -------------------------------------------------------------------

@dataclass
class Event:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(event.type, []))
            handlers += self._subscribers.get("*", [])
        for h in handlers:
            try:
                h(event)
            except Exception:
                logger.log(f"[Genesis] Error in event handler for {event.type}:\n{traceback.format_exc()}")

# -------------------------------------------------------------------
# Genesis plugin system
# -------------------------------------------------------------------

@dataclass
class PluginMeta:
    name: str
    version: str
    description: str
    module: Any
    enabled: bool = True
    last_error: Optional[str] = None

class PluginManager:
    def __init__(self, plugins_dir: str, memory: MemoryStore, bus: EventBus):
        self.plugins_dir = plugins_dir
        self.memory = memory
        self.bus = bus
        self.plugins: Dict[str, PluginMeta] = {}
        os.makedirs(self.plugins_dir, exist_ok=True)

    def discover_and_load(self) -> None:
        logger.log("[Genesis] Scanning plugins...")
        self.plugins.clear()
        for fname in os.listdir(self.plugins_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(self.plugins_dir, fname)
            name = os.path.splitext(fname)[0]
            self._load_single(name, path)

    def _load_single(self, name: str, path: str) -> None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            logger.log(f"[Genesis] Failed to create spec for plugin {name}")
            return
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.log(f"[Genesis] Error loading plugin {name}:\n{traceback.format_exc()}")
            return

        metadata = getattr(module, "METADATA", None)
        if not isinstance(metadata, dict):
            logger.log(f"[Genesis] Plugin {name} missing METADATA dict")
            return

        meta = PluginMeta(
            name=metadata.get("name", name),
            version=str(metadata.get("version", "0.0")),
            description=metadata.get("description", ""),
            module=module,
            enabled=True,
        )
        self.plugins[meta.name] = meta
        logger.log(f"[Genesis] Loaded plugin: {meta.name} v{meta.version}")

    def register_plugins(self, engine_api: Dict[str, Any]) -> None:
        for pname, meta in self.plugins.items():
            try:
                if hasattr(meta.module, "register"):
                    meta.module.register(engine_api)
                    logger.log(f"[Genesis] Registered plugin: {pname}")
                else:
                    logger.log(f"[Genesis] Plugin {pname} has no register()")
            except Exception:
                meta.last_error = traceback.format_exc()
                logger.log(f"[Genesis] Error registering plugin {pname}:\n{meta.last_error}")

    def get_plugin_snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "enabled": p.enabled,
                "last_error": p.last_error,
            }
            for p in self.plugins.values()
        ]

# -------------------------------------------------------------------
# Genesis simulation loop + tasks
# -------------------------------------------------------------------

@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)
    handler: Callable[[], None] = field(compare=False)
    interval: float = field(compare=False, default=1.0)
    last_run: float = field(compare=False, default=0.0)
    origin: str = field(compare=False, default="core")

class SimulationLoop(threading.Thread):
    def __init__(self, engine: "GenesisEngine"):
        super().__init__(daemon=True)
        self.engine = engine
        self.task_queue = engine.task_queue
        self.stop_flag = engine.stop_flag

    def run(self) -> None:
        logger.log("[Genesis] Simulation loop started")
        while not self.stop_flag.is_set():
            posture = self.engine.memory.get_posture()
            posture_factor = POSTURE_MODES.get(posture, 1.0)
            tick = BASE_SIM_TICK_SECONDS * posture_factor

            try:
                task: Task = self.task_queue.get(timeout=tick)
            except queue.Empty:
                self.engine.memory.inc_metric("ticks", 1)
                continue

            now = time.time()
            if now - task.last_run < task.interval * posture_factor:
                self.task_queue.put(task)
                self.engine.memory.inc_metric("ticks", 1)
                time.sleep(tick)
                continue

            try:
                task.handler()
            except Exception:
                logger.log(f"[Genesis] Error in task {task.name}:\n{traceback.format_exc()}")
            finally:
                task.last_run = time.time()
                self.task_queue.put(task)
                self.engine.memory.inc_metric("ticks", 1)

        logger.log("[Genesis] Simulation loop stopped")

# -------------------------------------------------------------------
# Genesis intent engine
# -------------------------------------------------------------------

@dataclass
class Intent:
    id: int
    kind: str
    description: str
    payload: Dict[str, Any]
    created_ts: float = field(default_factory=time.time)
    status: str = "pending"
    resolution_ts: Optional[float] = None

class IntentEngine:
    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self._lock = threading.Lock()
        self._counter = 0
        self._intents: List[Intent] = []

    def create_intent(self, kind: str, description: str, payload: Dict[str, Any]) -> Intent:
        with self._lock:
            self._counter += 1
            intent = Intent(
                id=self._counter,
                kind=kind,
                description=description,
                payload=payload,
            )
            self._intents.append(intent)
        self.memory.inc_metric("intents_created", 1)
        self.memory.add_intent_history({
            "id": intent.id,
            "kind": intent.kind,
            "description": intent.description,
            "payload": intent.payload,
            "created_ts": intent.created_ts,
        })
        return intent

    def list_intents(self) -> List[Intent]:
        with self._lock:
            return list(self._intents)

    def resolve_intent(self, intent_id: int, status: str) -> Optional[Intent]:
        with self._lock:
            for intent in self._intents:
                if intent.id == intent_id:
                    intent.status = status
                    intent.resolution_ts = time.time()
                    if status == "approved":
                        self.memory.inc_metric("intents_approved", 1)
                    elif status == "rejected":
                        self.memory.inc_metric("intents_rejected", 1)
                    return intent
        return None
# -------------------------------------------------------------------
# Genesis Self-tuning & prediction engine
# -------------------------------------------------------------------

class SelfTuningScheduler:
    def __init__(self, engine: "GenesisEngine"):
        self.engine = engine
        self.window: List[Dict[str, Any]] = []

    def snapshot_point(self):
        m = self.engine.memory.get_metrics()
        posture = self.engine.get_posture()
        state = self.engine.get_consciousness()
        self.window.append({
            "ts": time.time(),
            "ticks": m.get("ticks", 0),
            "heartbeats": m.get("heartbeats", 0),
            "thoughts": m.get("thoughts", 0),
            "posture": posture,
            "state": state,
        })
        if len(self.window) > 50:
            self.window = self.window[-50:]

    def analyze_and_propose(self):
        if len(self.window) < 5:
            return

        latest = self.window[-1]
        earliest = self.window[0]
        dt = max(1.0, latest["ts"] - earliest["ts"])
        d_thoughts = latest["thoughts"] - earliest["thoughts"]
        d_heartbeats = latest["heartbeats"] - earliest["heartbeats"]
        thoughts_rate = d_thoughts / dt
        hb_rate = d_heartbeats / dt if d_heartbeats > 0 else 0.0

        fluidity = self.engine.memory.get_fluidity()
        pressure = self.engine.memory.get_pressure()

        if thoughts_rate > 0:
            fluidity = min(1.0, fluidity + 0.01)
        else:
            fluidity = max(0.0, fluidity - 0.01)

        if hb_rate > 0.05 and thoughts_rate < 0.02:
            pressure = min(1.0, pressure + 0.02)
        else:
            pressure = max(0.0, pressure - 0.01)

        self.engine.memory.set_fluidity(fluidity)
        self.engine.memory.set_pressure(pressure)

        predicted_state = self.engine.get_consciousness()
        if fluidity > 0.7 and pressure < 0.4:
            predicted_state = "flow"
        elif fluidity > 0.8 and pressure < 0.3 and thoughts_rate > 0.02:
            predicted_state = "trance"
        elif pressure > 0.8:
            predicted_state = "overload"
        else:
            predicted_state = "baseline"

        if predicted_state != self.engine.get_consciousness():
            self.engine.intent_engine.create_intent(
                kind="consciousness_change",
                description=f"Shift consciousness state -> {predicted_state}",
                payload={"target_state": predicted_state},
            )

        posture = self.engine.get_posture()
        if predicted_state == "overload" and posture == "aggressive":
            self.engine.intent_engine.create_intent(
                kind="posture_change",
                description="High pressure + aggressive; suggest calm posture.",
                payload={"target_posture": "calm"},
            )

        if predicted_state in ("flow", "trance"):
            self.engine.intent_engine.create_intent(
                kind="task_interval_adjust",
                description="Increase tempo of intent_stream for flow/trance.",
                payload={"task_name": "intent_stream", "factor": 0.8},
            )
        elif predicted_state == "overload":
            self.engine.intent_engine.create_intent(
                kind="task_interval_adjust",
                description="Slow intent_stream to relieve overload.",
                payload={"task_name": "intent_stream", "factor": 1.3},
            )

# -------------------------------------------------------------------
# Genesis rule engine
# -------------------------------------------------------------------

@dataclass
class Rule:
    name: str
    condition: Callable[["GenesisEngine"], bool]
    action: Callable[["GenesisEngine"], None]
    last_fired: float = 0.0
    cooldown: float = 5.0

class RuleEngine:
    def __init__(self, engine: "GenesisEngine"):
        self.engine = engine
        self.rules: List[Rule] = []
        self._build_default_rules()

    def _build_default_rules(self):
        def cond_curiosity_aggressive(e: "GenesisEngine") -> bool:
            c = e.traits.get("curiosity", 0.7)
            posture = e.memory.get_posture()
            metrics = e.memory.get_metrics()
            heartbeats = metrics.get("heartbeats", 0)
            return c > 0.9 and posture != "aggressive" and heartbeats > 5

        def act_curiosity_aggressive(e: "GenesisEngine"):
            e.set_posture("aggressive")
            e.memory.append_narrative("Rule: curiosity peaked; posture -> aggressive.")
            logger.log("[Genesis RuleEngine] curiosity_to_aggressive")

        self.rules.append(Rule(
            name="curiosity_to_aggressive",
            condition=cond_curiosity_aggressive,
            action=act_curiosity_aggressive,
            cooldown=15.0,
        ))

        def cond_caution_calm(e: "GenesisEngine") -> bool:
            caution = e.traits.get("caution", 0.6)
            posture = e.memory.get_posture()
            metrics = e.memory.get_metrics()
            ticks = metrics.get("ticks", 0)
            return caution > 0.8 and posture == "aggressive" and ticks > 200

        def act_caution_calm(e: "GenesisEngine"):
            e.set_posture("calm")
            e.memory.append_narrative("Rule: caution stabilized; posture -> calm.")
            logger.log("[Genesis RuleEngine] caution_to_calm")

        self.rules.append(Rule(
            name="caution_to_calm",
            condition=cond_caution_calm,
            action=act_caution_calm,
            cooldown=30.0,
        ))

    def evaluate(self):
        now = time.time()
        for r in self.rules:
            if now - r.last_fired < r.cooldown:
                continue
            try:
                if r.condition(self.engine):
                    r.action(self.engine)
                    r.last_fired = now
            except Exception:
                logger.log(f"[Genesis] Error in rule {r.name}:\n{traceback.format_exc()}")

# -------------------------------------------------------------------
# Genesis core engine (no separate GUI, used by Tk main GUI)
# -------------------------------------------------------------------

class GenesisEngine:
    def __init__(self, memory_path: str):
        self.memory = MemoryStore(memory_path)
        self.bus = EventBus()
        self.plugin_manager = PluginManager(PLUGINS_DIR, self.memory, self.bus)
        self.task_queue: "queue.PriorityQueue[Task]" = queue.PriorityQueue()
        self.stop_flag = threading.Event()
        self.sim_thread: Optional[SimulationLoop] = None
        self.intent_engine = IntentEngine(self.memory)
        self.rule_engine = RuleEngine(self)
        self.self_tuner = SelfTuningScheduler(self)

        self.traits = {
            "curiosity": self.memory.get_trait("curiosity", 0.75),
            "caution": self.memory.get_trait("caution", 0.65),
            "verbosity": self.memory.get_trait("verbosity", 0.7),
        }

        self._register_core_handlers()

    def start(self) -> None:
        logger.log("[Genesis] Engine starting...")
        self.stop_flag.clear()
        self.plugin_manager.discover_and_load()
        self.plugin_manager.register_plugins(self._build_engine_api())
        self._schedule_core_tasks()
        self.sim_thread = SimulationLoop(self)
        self.sim_thread.start()
        self.bus.publish(Event(type="engine_started", payload={
            "traits": self.traits,
            "posture": self.get_posture(),
            "state": self.get_consciousness(),
        }))
        self.memory.append_log("engine_started")
        self.memory.append_narrative("Genesis Node awakened.")

    def stop(self) -> None:
        logger.log("[Genesis] Engine stopping...")
        self.stop_flag.set()
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2.0)
        self.bus.publish(Event(type="engine_stopped"))
        self.memory.append_log("engine_stopped")
        self.memory.append_narrative("Genesis Node entered rest.")
        self.memory.save()

    def reload_plugins(self) -> None:
        logger.log("[Genesis] Reloading plugins...")
        self.plugin_manager.discover_and_load()
        self.plugin_manager.register_plugins(self._build_engine_api())
        self.bus.publish(Event(type="plugins_reloaded"))
        self.memory.append_log("plugins_reloaded")
        self.memory.append_narrative("Plugins refreshed; abilities realigned.")

    def set_posture(self, posture: str) -> None:
        if posture not in POSTURE_MODES:
            return
        self.memory.set_posture(posture)
        self.bus.publish(Event(type="posture_changed", payload={"posture": posture}))
        logger.log(f"[Genesis] Posture changed to: {posture}")

    def get_posture(self) -> str:
        return self.memory.get_posture()

    def set_consciousness(self, state: str) -> None:
        if state not in CONSCIOUSNESS_STATES:
            return
        self.memory.set_consciousness(state)
        self.bus.publish(Event(type="consciousness_changed", payload={"state": state}))
        logger.log(f"[Genesis] Consciousness state changed to: {state}")

    def get_consciousness(self) -> str:
        return self.memory.get_consciousness()

    def _build_engine_api(self) -> Dict[str, Any]:
        return {
            "log": logger.log,
            "publish_event": self.bus.publish,
            "subscribe_event": self.bus.subscribe,
            "remember": self.memory.remember_fact,
            "recall": self.memory.get_fact,
            "set_trait": self.memory.set_trait,
            "get_trait": self.memory.get_trait,
            "set_posture": self.set_posture,
            "get_posture": self.get_posture,
            "set_consciousness": self.set_consciousness,
            "get_consciousness": self.get_consciousness,
            "add_task": self.add_task,
            "get_time": time.time,
            "append_narrative": self.memory.append_narrative,
            "set_plugin_state": self.memory.set_plugin_state,
            "get_plugin_state": self.memory.get_plugin_state,
            "create_intent": self.intent_engine.create_intent,
        }

    def add_task(self, name: str, priority: int, interval: float, handler: Callable[[], None], origin: str = "core") -> None:
        task = Task(priority=priority, name=name, handler=handler, interval=interval, origin=origin)
        self.task_queue.put(task)
        logger.log(f"[Genesis] Task added: {name} (prio={priority}, interval={interval}, origin={origin})")

    def _register_core_handlers(self) -> None:
        def generic_logger(event: Event) -> None:
            self.memory.append_log(f"event:{event.type}")
            if event.type == "heartbeat":
                self.memory.inc_metric("heartbeats", 1)
        self.bus.subscribe("*", generic_logger)

    def _schedule_core_tasks(self) -> None:
        def heartbeat():
            self.bus.publish(Event(
                type="heartbeat",
                payload={"t": time.time(), "posture": self.get_posture(), "state": self.get_consciousness()},
            ))
        self.add_task("heartbeat", priority=10, interval=2.0, handler=heartbeat)

        def self_reflect():
            c = self.traits.get("curiosity", 0.75)
            c = max(0.0, min(1.0, c + 0.005))
            self.traits["curiosity"] = c
            self.memory.set_trait("curiosity", c)
            state = self.get_consciousness()
            cfg = CONSCIOUSNESS_STATES[state]
            prefix = cfg["narrative_prefix"]
            if self.traits.get("verbosity", 0.7) + cfg["verbosity_bonus"] > 0.6:
                self.memory.append_narrative(f"{prefix}Self-reflect: curiosity {c:.3f}")
            logger.log(f"[Genesis] Self-reflect: curiosity {c:.3f}")
        self.add_task("self_reflect", priority=20, interval=10.0, handler=self_reflect)

        def intent_stream():
            posture = self.get_posture()
            state = self.get_consciousness()
            metrics = self.memory.get_metrics()
            thought_count = metrics.get("thoughts", 0) + 1
            self.memory.inc_metric("thoughts", 1)

            cfg = CONSCIOUSNESS_STATES[state]
            prefix = cfg["narrative_prefix"]

            if posture == "calm":
                idea = f"{prefix}thought #{thought_count}: slow scan, preserving energy."
            elif posture == "aggressive":
                idea = f"{prefix}thought #{thought_count}: hunting for patterns to amplify."
            else:
                idea = f"{prefix}thought #{thought_count}: focused internal evaluation."

            self.memory.append_narrative(idea)
            if self.traits.get("verbosity", 0.7) + cfg["verbosity_bonus"] > 0.8:
                logger.log("[Genesis Mindstream] " + idea)
        self.add_task("intent_stream", priority=50, interval=5.0, handler=intent_stream)

        def rule_engine_tick():
            self.rule_engine.evaluate()
        self.add_task("rules", priority=30, interval=4.0, handler=rule_engine_tick)

        def self_tuner_tick():
            self.self_tuner.snapshot_point()
            self.self_tuner.analyze_and_propose()
        self.add_task("self_tuner", priority=40, interval=6.0, handler=self_tuner_tick)

    def get_memory_snapshot(self) -> Dict[str, Any]:
        return self.memory.get_snapshot()

    def get_plugins_snapshot(self) -> List[Dict[str, Any]]:
        return self.plugin_manager.get_plugin_snapshot()

    def get_narrative(self) -> List[Dict[str, Any]]:
        return self.memory.get_narrative()

    def get_metrics(self) -> Dict[str, int]:
        return self.memory.get_metrics()

    def get_intents(self) -> List[Intent]:
        return self.intent_engine.list_intents()

    def apply_intent(self, intent: Intent) -> None:
        if intent.kind == "consciousness_change":
            target = intent.payload.get("target_state")
            self.set_consciousness(target)
        elif intent.kind == "posture_change":
            target = intent.payload.get("target_posture")
            self.set_posture(target)
        elif intent.kind == "task_interval_adjust":
            task_name = intent.payload.get("task_name")
            factor = float(intent.payload.get("factor", 1.0))
            new_queue: "queue.PriorityQueue[Task]" = queue.PriorityQueue()
            while not self.task_queue.empty():
                task: Task = self.task_queue.get()
                if task.name == task_name:
                    task.interval = max(0.5, task.interval * factor)
                    logger.log(f"[Genesis] Intent applied: {task_name} interval -> {task.interval:.2f}")
                new_queue.put(task)
            self.task_queue = new_queue
        elif intent.kind == "trait_adjust":
            trait = intent.payload.get("trait")
            delta = float(intent.payload.get("delta", 0.0))
            if trait in self.traits:
                new_val = max(0.0, min(1.0, self.traits[trait] + delta))
                self.traits[trait] = new_val
                self.memory.set_trait(trait, new_val)
                self.memory.append_narrative(f"Intent applied: trait {trait} -> {new_val:.3f}")
                logger.log(f"[Genesis] Intent applied: trait {trait} -> {new_val:.3f}")

# -------------------------------------------------------------------
# Example Genesis plugin
# -------------------------------------------------------------------

EXAMPLE_PLUGIN = r'''"""
Example Genesis Node plugin (v3).
"""

METADATA = {
    "name": "ExamplePlugin",
    "version": "0.3",
    "description": "Demonstration plugin reacting to state & posture.",
}


def register(engine):
    log = engine["log"]
    subscribe = engine["subscribe_event"]
    remember = engine["remember"]
    recall = engine["recall"]
    get_time = engine["get_time"]
    get_posture = engine["get_posture"]
    get_state = engine["get_consciousness"]
    append_narrative = engine["append_narrative"]
    set_plugin_state = engine["set_plugin_state"]
    get_plugin_state = engine["get_plugin_state"]
    add_task = engine["add_task"]
    create_intent = engine["create_intent"]

    PLUGIN_NAME = METADATA["name"]

    log(f"[{PLUGIN_NAME}] Registering")

    def on_heartbeat(event):
        last = recall("example.last_heartbeat", 0)
        now = get_time()
        if now - last > 5:
            log(f"[{PLUGIN_NAME}] Heartbeat sensed. Posture={get_posture()} State={get_state()}")
            remember("example.last_heartbeat", now)

    subscribe("heartbeat", on_heartbeat)

    def plugin_thought():
        count = recall("example.thought_count", 0)
        count += 1
        remember("example.thought_count", count)
        msg = f"[{PLUGIN_NAME}] Quiet check #{count}, posture={get_posture()}, state={get_state()}"
        append_narrative(msg)
        state_obj = get_plugin_state(PLUGIN_NAME, {"mild_suggestions": 0})
        state_obj["mild_suggestions"] = state_obj.get("mild_suggestions", 0) + 1
        set_plugin_state(PLUGIN_NAME, state_obj)

        if state_obj["mild_suggestions"] % 5 == 0:
            create_intent(
                kind="trait_adjust",
                description="Plugin suggests slight curiosity increase.",
                payload={"trait": "curiosity", "delta": 0.01},
            )

    add_task("example_plugin_thought", priority=70, interval=13.0, handler=plugin_thought, origin=PLUGIN_NAME)
'''

def ensure_example_plugin():
    os.makedirs(PLUGINS_DIR, exist_ok=True)
    example_path = os.path.join(PLUGINS_DIR, "example_plugin.py")
    if not os.path.exists(example_path):
        with open(example_path, "w", encoding="utf-8") as f:
            f.write(EXAMPLE_PLUGIN)
        logger.log(f"[Genesis] Created example plugin at {example_path}")

# -------------------------------------------------------------------
# Utility: OS info / language (Optimizer)
# -------------------------------------------------------------------

def get_os_info():
    os_name = platform.system()
    os_version = platform.version()
    lang, enc = locale.getdefaultlocale()
    return {
        "os_name": os_name,
        "os_version": os_version,
        "language": lang or "unknown",
        "encoding": enc or "unknown",
    }

# -------------------------------------------------------------------
# Foreground window → process detection (Optimizer)
# -------------------------------------------------------------------

def get_foreground_pid():
    hwnd = GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value or None

def get_foreground_process():
    pid = get_foreground_pid()
    if not pid:
        return None
    try:
        return psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

# -------------------------------------------------------------------
# Optimizer game detection (EXTREMELY permissive)
# -------------------------------------------------------------------

def is_probable_game(proc: psutil.Process, gpu_usage_hint: float = 0.0) -> bool:
    """
    Ultra-permissive: if there's a foreground process with an .exe that is
    NOT on the denylist, we treat it as a game.
    """
    if proc is None:
        return False
    try:
        exe = proc.exe()
        name = proc.name().lower()
        exe_lower = exe.lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

    logger.log(f"[Detector] Foreground process: name={name}, exe={exe}")

    if name in SYSTEM_PROCESS_DENYLIST:
        logger.log("[Detector] Denylisted -> NOT a game")
        return False

    if not exe_lower.endswith(".exe"):
        logger.log("[Detector] Not an .exe -> NOT a game")
        return False

    logger.log("[Detector] Treating foreground .exe as GAME")
    return True

def profile_id_for_proc(proc: psutil.Process) -> str:
    try:
        exe = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        exe = "unknown"
    h = hashlib.sha1(exe.encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(exe).lower()}_{h}"

# -------------------------------------------------------------------
# Optimizer Profile storage
# -------------------------------------------------------------------

class ProfileStore:
    def __init__(self, local_root: Path, network_root: str | None):
        self.local_root = local_root
        self.network_root = network_root if network_root else None

    def _profile_rel_path(self, profile_id: str) -> Path:
        return Path("profiles") / profile_id / "profile.json"

    def _local_path(self, profile_id: str) -> Path:
        return self.local_root / self._profile_rel_path(profile_id)

    def _network_path(self, profile_id: str) -> Path | None:
        if not self.network_root:
            return None
        return Path(self.network_root) / self._profile_rel_path(profile_id)

    def load_profile(self, profile_id: str, default_data: dict) -> dict:
        net_path = self._network_path(profile_id)
        if net_path and net_path.exists():
            try:
                data = json.loads(net_path.read_text(encoding="utf-8"))
                self._write_local(profile_id, data)
                return data
            except Exception:
                pass

        local_path = self._local_path(profile_id)
        if local_path.exists():
            try:
                return json.loads(local_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.save_profile(profile_id, default_data)
        return default_data

    def save_profile(self, profile_id: str, data: dict):
        self._write_local(profile_id, data)
        net_path = self._network_path(profile_id)
        if net_path:
            try:
                net_path.parent.mkdir(parents=True, exist_ok=True)
                net_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _write_local(self, profile_id: str, data: dict):
        path = self._local_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# -------------------------------------------------------------------
# Optimizer Profile data structures
# -------------------------------------------------------------------

@dataclass
class GameProfile:
    profile_id: str
    exe_path: str
    display_name: str
    sessions_observed: int = 0
    minutes_observed: float = 0.0
    scenario_diversity: float = 0.0
    pattern_noise: float = 0.0

    learning_percent: int = 0
    confidence_percent: int = 0

    successful_predictions: int = 0
    total_predictions: int = 0
    recent_anomalies: int = 0
    stability_score: float = 0.0

    last_update_ts: float = 0.0
    last_session_start: float = 0.0
    last_session_end: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict):
        return GameProfile(**data)

def default_profile(profile_id: str, exe_path: str) -> GameProfile:
    return GameProfile(
        profile_id=profile_id,
        exe_path=exe_path,
        display_name=os.path.basename(exe_path) if exe_path else profile_id,
    )

# -------------------------------------------------------------------
# Optimizer metrics helpers
# -------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def update_learning_and_confidence(profile: GameProfile, seconds_delta: float):
    """
    More responsive learning/confidence model:
    - Learning ramps quickly with minutes + sessions.
    - Confidence starts low, ramps with total predictions, and is nudged by stability/anomalies.
    This guarantees visible motion in the first few minutes.
    """
    minutes = profile.minutes_observed
    sessions = profile.sessions_observed
    diversity = profile.scenario_diversity

    # Base learning from time + sessions + diversity
    learning = (
        sessions * 15.0 +        # each session = strong boost
        minutes * 2.0 +          # each minute adds 2%
        diversity * 20.0         # diversity helps
    )
    learning = int(clamp(learning, 0, 100))

    # Confidence:
    # - starts low with few predictions
    # - ramps with more predictions
    # - adjusted by success rate, stability, anomalies
    total_pred = max(1, profile.total_predictions)
    success_rate = profile.successful_predictions / total_pred

    # Base confidence from sample size (up to 60%)
    sample_term = min(total_pred / 200.0, 1.0) * 60.0

    # Quality term from success & stability
    quality_term = success_rate * 30.0 + (profile.stability_score / 10.0)

    # Penalty for anomalies
    anomaly_penalty = profile.recent_anomalies * 2.0

    confidence = sample_term + quality_term - anomaly_penalty
    confidence = int(clamp(confidence, 0, 100))

    profile.learning_percent = learning
    profile.confidence_percent = confidence

# -------------------------------------------------------------------
# Optimizer Inference engine abstraction
# -------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
        self.available = False

    def init(self):
        self.available = False

    def predict_risk(self, features: dict) -> float:
        return 0.0

class MovidiusInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.model_path = model_path
        self.device = None

    def init(self):
        try:
            self.available = False
        except Exception:
            self.available = False

    def predict_risk(self, features: dict) -> float:
        if not self.available:
            return 0.0
        return 0.0

# -------------------------------------------------------------------
# Optimizer Foresight engine
# -------------------------------------------------------------------

class ForesightEngine:
    def __init__(self, window_size: int = 10000, inference_engine: InferenceEngine | None = None):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.long_term = {
            "cpu_avg": 0.0,
            "ram_avg": 0.0,
            "cpu_var": 0.0,
            "ram_var": 0.0,
            "samples": 0
        }
        self.smooth = {
            "cpu": 0.0,
            "ram": 0.0,
            "alpha": 0.2,
            "initialized": False,
        }
        self.volatility = {
            "cpu": 0.0,
            "ram": 0.0,
            "alpha": 0.3,
            "initialized": False,
        }
        self.inertia = {
            "risk": 0.0,
            "alpha": 0.2,
        }
        self.inference_engine = inference_engine

    def add_sample(self, readings: dict):
        cpu = readings.get("cpu_total", 0.0)
        ram = readings.get("ram_percent", 0.0)

        self.history.append({
            "ts": time.time(),
            "cpu": cpu,
            "ram": ram,
        })

        lt = self.long_term
        n = lt["samples"] + 1
        lt["samples"] = n

        lt["cpu_avg"] += (cpu - lt["cpu_avg"]) / n
        lt["ram_avg"] += (ram - lt["ram_avg"]) / n

        lt["cpu_var"] += (cpu - lt["cpu_avg"]) ** 2
        lt["ram_var"] += (ram - lt["ram_avg"]) ** 2

        s = self.smooth
        if not s["initialized"]:
            s["cpu"] = cpu
            s["ram"] = ram
            s["initialized"] = True
        else:
            a = s["alpha"]
            s["cpu"] = a * cpu + (1 - a) * s["cpu"]
            s["ram"] = a * ram + (1 - a) * s["ram"]

        v = self.volatility
        if not v["initialized"]:
            v["cpu"] = 0.0
            v["ram"] = 0.0
            v["initialized"] = True
        else:
            dv_cpu = abs(cpu - s["cpu"])
            dv_ram = abs(ram - s["ram"])
            a_v = v["alpha"]
            v["cpu"] = a_v * dv_cpu + (1 - a_v) * v["cpu"]
            v["ram"] = a_v * dv_ram + (1 - a_v) * v["ram"]

    def compute_risk(self) -> float:
        if len(self.history) < 5:
            return 0.0

        cpu_now = self.history[-1]["cpu"]
        cpu_prev = self.history[0]["cpu"]

        ram_now = self.history[-1]["ram"]
        ram_prev = self.history[0]["ram"]

        cpu_trend = cpu_now - cpu_prev
        ram_trend = ram_now - ram_prev

        lt = self.long_term
        cpu_dev = abs(cpu_now - lt["cpu_avg"])
        ram_dev = abs(ram_now - lt["ram_avg"])

        s = self.smooth
        cpu_s = s["cpu"]
        ram_s = s["ram"]

        cpu_potential = max(cpu_s - lt["cpu_avg"], 0.0)
        ram_potential = max(ram_s - lt["ram_avg"], 0.0)

        v = self.volatility
        vol_score = (v["cpu"] * 0.6 + v["ram"] * 0.4)

        near_risk = (
            max(cpu_trend, 0) * 0.8 +
            max(ram_trend, 0) * 0.6 +
            vol_score * 0.5
        )

        mid_risk = (
            cpu_potential * 0.6 +
            ram_potential * 0.5 +
            cpu_dev * 0.4 +
            ram_dev * 0.3
        )

        risk = near_risk * 0.6 + mid_risk * 0.4
        base_risk = float(clamp(risk, 0.0, 100.0))

        ine = self.inertia
        a_r = ine["alpha"]
        ine["risk"] = a_r * base_risk + (1 - a_r) * ine["risk"]
        base_risk = ine["risk"]

        if self.inference_engine and self.inference_engine.available:
            features = {
                "cpu_now": cpu_now,
                "ram_now": ram_now,
                "cpu_trend": cpu_trend,
                "ram_trend": ram_trend,
                "cpu_avg": lt["cpu_avg"],
                "ram_avg": lt["ram_avg"],
                "cpu_smooth": s["cpu"],
                "ram_smooth": s["ram"],
                "vol_cpu": v["cpu"],
                "vol_ram": v["ram"],
            }
            delta = self.inference_engine.predict_risk(features)
            base_risk = float(clamp(base_risk + delta, 0.0, 100.0))

        return base_risk

    def get_long_term_stats(self):
        lt = self.long_term
        if lt["samples"] > 1:
            cpu_var = lt["cpu_var"] / (lt["samples"] - 1)
            ram_var = lt["ram_var"] / (lt["samples"] - 1)
        else:
            cpu_var = 0.0
            ram_var = 0.0
        return {
            "cpu_avg": lt["cpu_avg"],
            "ram_avg": lt["ram_avg"],
            "cpu_var": cpu_var,
            "ram_var": ram_var,
            "samples": lt["samples"],
        }

    def get_confidence(self) -> float:
        lt = self.long_term
        v = self.volatility
        if lt["samples"] < 50:
            return 20.0
        vol = (v["cpu"] + v["ram"]) / 2.0
        sample_term = min(lt["samples"] / 500.0, 1.0)
        conf = 50.0 * sample_term + max(0.0, 50.0 - vol)
        return float(clamp(conf, 0.0, 100.0))

# -------------------------------------------------------------------
# Optimizer Power posture engine
# -------------------------------------------------------------------

class Posture:
    CALM = "Calm"
    ENGAGED = "Game Engaged"
    REDLINE = "Redline"

class PowerPostureEngine:
    def __init__(self):
        self.plan_balanced = None
        self.plan_high_perf = None

    def _set_power_plan(self, guid: str | None):
        if not guid:
            return
        try:
            subprocess.run(
                ["powercfg", "/S", guid],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def apply_posture(self, posture: str, risk: float, game_proc: psutil.Process | None):
        if posture == Posture.CALM:
            self._set_power_plan(self.plan_balanced)
        elif posture == Posture.ENGAGED:
            if risk > 40.0:
                self._set_power_plan(self.plan_high_perf)
            else:
                self._set_power_plan(self.plan_balanced)
        elif posture == Posture.REDLINE:
            self._set_power_plan(self.plan_high_perf)

        self._adjust_priorities(posture, game_proc)

    def _adjust_priorities(self, posture: str, game_proc: psutil.Process | None):
        if game_proc is None:
            return
        try:
            if posture in (Posture.ENGAGED, Posture.REDLINE):
                game_proc.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                game_proc.nice(psutil.NORMAL_PRIORITY_CLASS)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if posture == Posture.REDLINE:
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    if p.pid == game_proc.pid:
                        continue
                    name = (p.info["name"] or "").lower()
                    if any(tok in name for tok in ["updater", "update", "installer", "onedrive", "dropbox", "launcher"]):
                        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

# -------------------------------------------------------------------
# Optimizer Brain modes
# -------------------------------------------------------------------

class BrainMode:
    BASELINE = "Baseline"
    HYPERVIGILANT = "Hypervigilant"
    RELAXED = "Relaxed"

# -------------------------------------------------------------------
# Optimizer posture decision engine
# -------------------------------------------------------------------

def decide_posture(readings: dict, profile: GameProfile | None, risk: float, brain_mode: str) -> str:
    cpu = readings.get("cpu_total", 0.0)
    ram = readings.get("ram_percent", 0.0)
    if not readings.get("is_game_active", False):
        return Posture.CALM
    risk_redline = 70.0
    cpu_redline = 80.0
    ram_redline = 85.0
    if brain_mode == BrainMode.HYPERVIGILANT:
        risk_redline -= 15.0
        cpu_redline -= 5.0
        ram_redline -= 5.0
    elif brain_mode == BrainMode.RELAXED:
        risk_redline += 10.0
        cpu_redline += 5.0
        ram_redline += 5.0
    if risk >= risk_redline:
        return Posture.REDLINE
    if cpu > cpu_redline or ram > ram_redline:
        return Posture.REDLINE
    return Posture.ENGAGED

# -------------------------------------------------------------------
# Optimizer sensors
# -------------------------------------------------------------------

def collect_gpu_usage() -> float:
    if not HAS_GPU:
        return 0.0
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0
        return max(gpu.load for gpu in gpus) * 100.0
    except Exception:
        return 0.0

def collect_system_metrics(active_game_proc: psutil.Process | None) -> dict:
    readings = {}
    readings["cpu_total"] = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    readings["ram_percent"] = ram.percent
    readings["gpu_total"] = collect_gpu_usage()
    if active_game_proc:
        try:
            readings["game_name"] = active_game_proc.name()
            readings["game_exe"] = active_game_proc.exe()
            p_cpu = active_game_proc.cpu_percent(interval=None)
            p_mem = active_game_proc.memory_percent()
            readings["game_cpu"] = p_cpu
            readings["game_mem"] = p_mem
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            readings["game_name"] = "Unknown"
            readings["game_exe"] = ""
            readings["game_cpu"] = 0.0
            readings["game_mem"] = 0.0
    else:
        readings["game_name"] = "None"
        readings["game_exe"] = ""
        readings["game_cpu"] = 0.0
        readings["game_mem"] = 0.0
    readings["fps"] = "--"
    readings["ping"] = "--"
    return readings

# -------------------------------------------------------------------
# OptimizerCore (no GUI toolkit binding; called from Tk)
# -------------------------------------------------------------------

class OptimizerCore:
    def __init__(self, profile_store: ProfileStore, genesis: GenesisEngine, poll_interval_ms: int = 500):
        self.store = profile_store
        self.poll_interval_ms = poll_interval_ms
        self.current_profile_id: str | None = None
        self.current_profile: GameProfile | None = None
        self.current_posture: str = Posture.CALM
        self.last_tick_ts = time.time()
        self.learning_enabled = True
        self.os_info = get_os_info()
        self.movidius_engine = MovidiusInferenceEngine(model_path=None)
        self.movidius_engine.init()
        self.foresight = ForesightEngine(window_size=10000, inference_engine=self.movidius_engine)
        self.power_engine = PowerPostureEngine()
        self.active_game_proc: psutil.Process | None = None
        self.brain_mode = BrainMode.BASELINE
        self.recent_stutter_score = 0.0
        self.network_root = self.store.network_root
        self.network_enabled = self.store.network_root is not None
        self.network_status = "Disabled" if not self.network_enabled else "Connected"
        self.genesis = genesis
        self.state_listeners: List[Callable[[dict], None]] = []

    def add_state_listener(self, fn: Callable[[dict], None]):
        self.state_listeners.append(fn)

    def emit_state(self, stats: dict):
        for fn in list(self.state_listeners):
            try:
                fn(stats)
            except Exception:
                pass

    def set_network_root(self, path: str | None, enabled: bool):
        if enabled and path:
            try:
                p = Path(path)
                p.mkdir(parents=True, exist_ok=True)
                self.store.network_root = str(p)
                self.network_root = str(p)
                self.network_enabled = True
                self.network_status = "Connected"
                return True, "Network storage enabled."
            except Exception as e:
                self.network_enabled = False
                self.network_status = f"Error: {e}"
                self.store.network_root = None
                self.network_root = None
                return False, f"Error enabling network storage: {e}"
        else:
            self.store.network_root = None
            self.network_root = None
            self.network_enabled = False
            self.network_status = "Disabled"
            return True, "Network storage disabled."

    def test_network_root(self, path: str):
        try:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / "._optimizer_test.tmp"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return True, "Connection OK."
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _process_genesis_intents(self):
        intents = self.genesis.get_intents()
        for intent in intents:
            if intent.status != "pending":
                continue
            if intent.kind in ("posture_change", "trait_adjust", "task_interval_adjust"):
                self.genesis.intent_engine.resolve_intent(intent.id, "approved")
                self.genesis.apply_intent(intent)

    def tick(self) -> dict:
        now = time.time()
        delta = now - self.last_tick_ts
        self.last_tick_ts = now

        fg_proc = get_foreground_process()
        gpu_hint = collect_gpu_usage() if HAS_GPU else 0.0
        is_game = is_probable_game(fg_proc, gpu_usage_hint=gpu_hint) if fg_proc else False
        self.active_game_proc = fg_proc if is_game else None

        readings = collect_system_metrics(self.active_game_proc)
        readings["is_game_active"] = is_game

        if is_game and fg_proc:
            pid = profile_id_for_proc(fg_proc)
            if pid != self.current_profile_id:
                logger.log(f"[Optimizer] New game detected: {readings['game_name']} ({pid})")
                self._load_profile_for_process(fg_proc, pid)
            if self.current_profile:
                self.current_profile.minutes_observed += delta / 60.0
                self.current_profile.scenario_diversity = min(
                    1.0, self.current_profile.scenario_diversity + delta / 600.0
                )
                self.current_profile.pattern_noise = max(
                    0.0, self.current_profile.pattern_noise - delta / 1200.0
                )
        else:
            if self.current_profile and self.current_profile.last_session_start > 0:
                self.current_profile.last_session_end = now
                self._save_current_profile()
            self.current_profile_id = None
            self.current_profile = None

        self.foresight.add_sample(readings)
        raw_risk = self.foresight.compute_risk()
        foresight_conf = self.foresight.get_confidence()

        risk = raw_risk
        if self.brain_mode == BrainMode.HYPERVIGILANT:
            risk *= 1.25
        elif self.brain_mode == BrainMode.RELAXED:
            risk *= 0.85
        risk = float(clamp(risk, 0.0, 100.0))

        readings["foresight_risk"] = risk
        readings["foresight_confidence"] = foresight_conf

        predicted_high_risk = risk > 70.0
        actual_spike = readings["cpu_total"] > 90.0 or readings["ram_percent"] > 90.0

        if predicted_high_risk and not actual_spike:
            self.recent_stutter_score = max(0.0, self.recent_stutter_score - 1.0)
        elif (not predicted_high_risk) and actual_spike:
            self.recent_stutter_score = min(100.0, self.recent_stutter_score + 4.0)

        if self.current_profile and self.learning_enabled:
            self.current_profile.total_predictions += 1
            if readings["cpu_total"] < 90.0:
                self.current_profile.successful_predictions += 1
                self.recent_stutter_score = max(0.0, self.recent_stutter_score - 0.5)
            else:
                self.current_profile.recent_anomalies += 1
                self.recent_stutter_score = min(100.0, self.recent_stutter_score + 2.0)

        self.recent_stutter_score = clamp(self.recent_stutter_score, 0.0, 100.0)

        if self.recent_stutter_score > 40.0:
            self.brain_mode = BrainMode.HYPERVIGILANT
        elif self.recent_stutter_score < 10.0 and self.current_profile and self.current_profile.learning_percent > 60:
            self.brain_mode = BrainMode.RELAXED
        else:
            self.brain_mode = BrainMode.BASELINE

        readings["brain_mode"] = self.brain_mode

        self.current_posture = decide_posture(readings, self.current_profile, risk, self.brain_mode)
        readings["posture"] = self.current_posture

        self.power_engine.apply_posture(self.current_posture, risk, self.active_game_proc)

        if self.current_profile and self.learning_enabled:
            self.current_profile.stability_score = max(
                0.0,
                100.0 - abs(readings["cpu_total"] - 50.0)
            )
            update_learning_and_confidence(self.current_profile, delta)
            if now - self.current_profile.last_update_ts > 5.0:
                self._save_current_profile()
                self.current_profile.last_update_ts = now

            readings["learning_percent"] = self.current_profile.learning_percent
            readings["confidence_percent"] = self.current_profile.confidence_percent
            readings["profile_name"] = self.current_profile.display_name
        else:
            readings["learning_percent"] = 0
            readings["confidence_percent"] = 0
            readings["profile_name"] = "None"

        lt_stats = self.foresight.get_long_term_stats()
        readings["lt_cpu_avg"] = lt_stats["cpu_avg"]
        readings["lt_ram_avg"] = lt_stats["ram_avg"]

        readings["os_info"] = self.os_info
        readings["network_root"] = self.network_root or ""
        readings["network_status"] = self.network_status

        self._publish_to_genesis(readings)
        self._process_genesis_intents()

        g_metrics = self.genesis.get_metrics()
        readings["genesis_metrics"] = g_metrics
        readings["genesis_posture"] = self.genesis.get_posture()
        readings["genesis_state"] = self.genesis.get_consciousness()
        readings["genesis_fluidity"] = self.genesis.memory.get_fluidity()
        readings["genesis_pressure"] = self.genesis.memory.get_pressure()

        self.emit_state(readings)
        return readings

    def _publish_to_genesis(self, readings: dict):
        self.genesis.bus.publish(Event(
            type="optimizer_metrics",
            payload={
                "cpu_total": readings.get("cpu_total"),
                "ram_percent": readings.get("ram_percent"),
                "gpu_total": readings.get("gpu_total"),
                "foresight_risk": readings.get("foresight_risk"),
                "foresight_confidence": readings.get("foresight_confidence"),
            }
        ))
        self.genesis.bus.publish(Event(
            type="game_status",
            payload={
                "is_game_active": readings.get("is_game_active"),
                "game_name": readings.get("game_name"),
                "game_cpu": readings.get("game_cpu"),
                "game_mem": readings.get("game_mem"),
            }
        ))
        self.genesis.bus.publish(Event(
            type="optimizer_posture",
            payload={
                "posture": readings.get("posture"),
                "brain_mode": readings.get("brain_mode"),
            }
        ))

    def _load_profile_for_process(self, proc: psutil.Process, profile_id: str):
        now = time.time()
        try:
            exe = proc.exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            exe = ""
        default = default_profile(profile_id, exe).to_dict()
        data = self.store.load_profile(profile_id, default)
        self.current_profile = GameProfile.from_dict(data)
        self.current_profile_id = profile_id
        self.current_profile.sessions_observed += 1
        self.current_profile.last_update_ts = now
        self.current_profile.last_session_start = now

    def _save_current_profile(self):
        if self.current_profile is None:
            return
        self.store.save_profile(self.current_profile.profile_id, self.current_profile.to_dict())

# -------------------------------------------------------------------
# Overlay HUD (Tkinter Toplevel)
# -------------------------------------------------------------------

class OverlayHUD(tk.Toplevel):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.root = root
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.config(bg="black")
        try:
            self.attributes("-transparentcolor", "black")
        except Exception:
            pass
        frame = tk.Frame(self, bg="#202020")
        frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.lbl_posture = tk.Label(frame, text="Calm", fg="white", bg="#202020", font=("Segoe UI", 9, "bold"))
        self.lbl_posture.grid(row=0, column=0, sticky="w", padx=(4, 4))
        self.risk_var = tk.IntVar(value=0)
        self.risk_bar = ttk.Progressbar(frame, orient="horizontal", length=120, mode="determinate",
                                        maximum=100, variable=self.risk_var)
        self.risk_bar.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.lbl_lc = tk.Label(frame, text="L:0 C:0", fg="white", bg="#202020", font=("Segoe UI", 8))
        self.lbl_lc.grid(row=0, column=2, sticky="e", padx=(4, 4))
        frame.columnconfigure(1, weight=1)
        self.update_idletasks()
        self._position_top_center()

    def _position_top_center(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        screen_w = self.winfo_screenwidth()
        x = (screen_w - w) // 2
        y = 10
        self.geometry(f"{w}x{h}+{x}+{y}")

    def update_overlay(self, posture: str, risk: float, learning: int, confidence: int, visible: bool, force_show: bool):
        should_show = visible or force_show
        if not should_show:
            self.withdraw()
            return
        self.deiconify()
        self.lbl_posture.config(text=posture)
        try:
            self.risk_var.set(int(risk))
        except Exception:
            self.risk_var.set(0)
        self.lbl_lc.config(text=f"L:{learning} C:{confidence}")
        self._position_top_center()

# -------------------------------------------------------------------
# Genesis GUI view (right pane)
# -------------------------------------------------------------------

class GenesisView:
    def __init__(self, parent: tk.Widget, engine: GenesisEngine):
        self.engine = engine
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)
        self._build_layout()
        logger.register(self._append_log)
        self.engine.bus.subscribe("*", self._on_event)
        self._refresh_memory()
        self._refresh_plugins()
        self._refresh_mindstream()
        self._refresh_intents()
        self._update_status_bar()
        self._schedule_gui_updates()

    def _build_layout(self):
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=0)
        self.frame.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        console_frame = ttk.Frame(self.notebook)
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)
        self.console = scrolledtext.ScrolledText(console_frame, state="disabled", wrap="word", height=8)
        self.console.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(console_frame, text="Console")
        self.events_text = scrolledtext.ScrolledText(self.notebook, state="disabled", wrap="word")
        self.notebook.add(self.events_text, text="Events")
        self.memory_text = scrolledtext.ScrolledText(self.notebook, state="disabled", wrap="word")
        self.notebook.add(self.memory_text, text="Memory")
        plugins_frame = ttk.Frame(self.notebook)
        plugins_frame.rowconfigure(0, weight=1)
        plugins_frame.columnconfigure(0, weight=1)
        self.plugins_tree = ttk.Treeview(
            plugins_frame,
            columns=("name", "version", "description", "enabled", "error"),
            show="headings",
            height=8,
        )
        for col, text, w in [
            ("name", "Name", 120),
            ("version", "Version", 70),
            ("description", "Description", 220),
            ("enabled", "Enabled", 70),
            ("error", "Last Error", 250),
        ]:
            self.plugins_tree.heading(col, text=text)
            self.plugins_tree.column(col, width=w, anchor="w")
        self.plugins_tree.grid(row=0, column=0, sticky="nsew")
        plugins_scroll = ttk.Scrollbar(plugins_frame, orient="vertical", command=self.plugins_tree.yview)
        plugins_scroll.grid(row=0, column=1, sticky="ns")
        self.plugins_tree.configure(yscrollcommand=plugins_scroll.set)
        self.notebook.add(plugins_frame, text="Plugins")
        self.mindstream_text = scrolledtext.ScrolledText(self.notebook, state="disabled", wrap="word")
        self.notebook.add(self.mindstream_text, text="Mindstream")
        intents_frame = ttk.Frame(self.notebook)
        intents_frame.rowconfigure(1, weight=1)
        intents_frame.columnconfigure(0, weight=1)
        self.intents_summary_label = ttk.Label(intents_frame, text="Intents: 0 pending")
        self.intents_summary_label.grid(row=0, column=0, sticky="w")
        self.intents_tree = ttk.Treeview(
            intents_frame,
            columns=("id", "kind", "description", "status"),
            show="headings",
            height=8,
        )
        for col, text, w in [
            ("id", "ID", 50),
            ("kind", "Kind", 150),
            ("description", "Description", 280),
            ("status", "Status", 100),
        ]:
            self.intents_tree.heading(col, text=text)
            self.intents_tree.column(col, width=w, anchor="w")
        self.intents_tree.grid(row=1, column=0, sticky="nsew")
        intents_scroll = ttk.Scrollbar(intents_frame, orient="vertical", command=self.intents_tree.yview)
        intents_scroll.grid(row=1, column=1, sticky="ns")
        self.intents_tree.configure(yscrollcommand=intents_scroll.set)
        intents_buttons = ttk.Frame(intents_frame)
        intents_buttons.grid(row=2, column=0, sticky="ew", pady=5)
        intents_buttons.columnconfigure(0, weight=1)
        intents_buttons.columnconfigure(1, weight=1)
        intents_buttons.columnconfigure(2, weight=1)
        self.btn_approve_intent = ttk.Button(intents_buttons, text="Approve", command=self._approve_intent)
        self.btn_approve_intent.grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_reject_intent = ttk.Button(intents_buttons, text="Reject", command=self._reject_intent)
        self.btn_reject_intent.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_refresh_intents = ttk.Button(intents_buttons, text="Refresh Intents", command=self._refresh_intents)
        self.btn_refresh_intents.grid(row=0, column=2, sticky="ew", padx=2)
        self.notebook.add(intents_frame, text="Intents")
        status_frame = ttk.Frame(self.frame)
        status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        for i in range(6):
            status_frame.columnconfigure(i, weight=1)
        self.status_posture = ttk.Label(status_frame, text="Posture: -")
        self.status_posture.grid(row=0, column=0, sticky="w")
        self.status_state = ttk.Label(status_frame, text="State: -")
        self.status_state.grid(row=0, column=1, sticky="w")
        self.status_ticks = ttk.Label(status_frame, text="Ticks: 0")
        self.status_ticks.grid(row=0, column=2, sticky="w")
        self.status_heartbeats = ttk.Label(status_frame, text="Heartbeats: 0")
        self.status_heartbeats.grid(row=0, column=3, sticky="w")
        self.status_thoughts = ttk.Label(status_frame, text="Thoughts: 0")
        self.status_thoughts.grid(row=0, column=4, sticky="w")
        self.status_intents = ttk.Label(status_frame, text="Intents: 0/0/0")
        self.status_intents.grid(row=0, column=5, sticky="w")

    def _append_log(self, line: str):
        self.console.configure(state="normal")
        self.console.insert("end", line + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _on_event(self, event: Event):
        text = f"[{time.strftime('%H:%M:%S', time.localtime(event.ts))}] {event.type} {event.payload}\n"
        self.events_text.configure(state="normal")
        self.events_text.insert("end", text)
        self.events_text.see("end")
        self.events_text.configure(state="disabled")

    def _get_selected_intent_id(self) -> Optional[int]:
        sel = self.intents_tree.selection()
        if not sel:
            return None
        item = self.intents_tree.item(sel[0])
        try:
            return int(item["values"][0])
        except Exception:
            return None

    def _approve_intent(self):
        intent_id = self._get_selected_intent_id()
        if intent_id is None:
            return
        intent = self.engine.intent_engine.resolve_intent(intent_id, "approved")
        if intent:
            self.engine.apply_intent(intent)
            self._refresh_intents()
            self._update_status_bar()

    def _reject_intent(self):
        intent_id = self._get_selected_intent_id()
        if intent_id is None:
            return
        intent = self.engine.intent_engine.resolve_intent(intent_id, "rejected")
        if intent:
            self._refresh_intents()
            self._update_status_bar()

    def _refresh_memory(self):
        snapshot = self.engine.get_memory_snapshot()
        self.memory_text.configure(state="normal")
        self.memory_text.delete("1.0", "end")
        self.memory_text.insert("end", json.dumps(snapshot, indent=2))
        self.memory_text.configure(state="disabled")

    def _refresh_plugins(self):
        for item in self.plugins_tree.get_children():
            self.plugins_tree.delete(item)
        for p in self.engine.get_plugins_snapshot():
            self.plugins_tree.insert(
                "",
                "end",
                values=(
                    p["name"],
                    p["version"],
                    p["description"],
                    "Yes" if p["enabled"] else "No",
                    (p["last_error"] or "")[:200],
                ),
            )

    def _refresh_mindstream(self):
        narrative = self.engine.get_narrative()
        self.mindstream_text.configure(state="normal")
        self.mindstream_text.delete("1.0", "end")
        for entry in narrative[-200:]:
            ts = time.strftime('%H:%M:%S', time.localtime(entry["ts"]))
            self.mindstream_text.insert("end", f"[{ts}] {entry['entry']}\n")
        self.mindstream_text.see("end")
        self.mindstream_text.configure(state="disabled")

    def _refresh_intents(self):
        for item in self.intents_tree.get_children():
            self.intents_tree.delete(item)
        intents = self.engine.get_intents()
        pending = sum(1 for i in intents if i.status == "pending")
        for i in intents[-200:]:
            self.intents_tree.insert(
                "",
                "end",
                values=(i.id, i.kind, i.description, i.status),
            )
        self.intents_summary_label.config(text=f"Intents: {pending} pending")

    def _update_status_bar(self):
        posture = self.engine.get_posture()
        state = self.engine.get_consciousness()
        metrics = self.engine.get_metrics()
        self.status_posture.config(text=f"Posture: {posture}")
        self.status_state.config(text=f"State: {state}")
        self.status_ticks.config(text=f"Ticks: {metrics.get('ticks', 0)}")
        self.status_heartbeats.config(text=f"Heartbeats: {metrics.get('heartbeats', 0)}")
        self.status_thoughts.config(text=f"Thoughts: {metrics.get('thoughts', 0)}")
        self.status_intents.config(
            text=f"Intents: created {metrics.get('intents_created', 0)}, "
                 f"approved {metrics.get('intents_approved', 0)}, "
                 f"rejected {metrics.get('intents_rejected', 0)}"
        )

    def _schedule_gui_updates(self):
        self._refresh_mindstream()
        self._refresh_intents()
        self._update_status_bar()
        self.frame.after(1000, self._schedule_gui_updates)

# -------------------------------------------------------------------
# Optimizer Panel (left pane)
# -------------------------------------------------------------------

class OptimizerPanel:
    def __init__(self, parent: tk.Widget, core: OptimizerCore):
        self.core = core
        self.latest_stats: dict = {}
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        self._build_layout()
        self.core.add_state_listener(self.update_stats)

    def _build_layout(self):
        self.frame.columnconfigure(0, weight=1)
        self.lbl_game = ttk.Label(self.frame, text="Game: None")
        self.lbl_game.grid(row=0, column=0, sticky="w")
        self.lbl_profile = ttk.Label(self.frame, text="Profile: None")
        self.lbl_profile.grid(row=1, column=0, sticky="w")
        self.lbl_posture = ttk.Label(self.frame, text="Posture: Calm")
        self.lbl_posture.grid(row=2, column=0, sticky="w")
        self.lbl_mode = ttk.Label(self.frame, text="Brain Mode: Baseline")
        self.lbl_mode.grid(row=3, column=0, sticky="w")
        self.lbl_learning = ttk.Label(self.frame, text="Learning: 0%")
        self.lbl_learning.grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.learning_var = tk.IntVar(value=0)
        self.learning_bar = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate",
                                            maximum=100, variable=self.learning_var)
        self.learning_bar.grid(row=5, column=0, sticky="ew")
        self.lbl_confidence = ttk.Label(self.frame, text="Optimization Confidence: 0%")
        self.lbl_confidence.grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.confidence_var = tk.IntVar(value=0)
        self.confidence_bar = ttk.Progressbar(self.frame, orient="horizontal", mode="determinate",
                                              maximum=100, variable=self.confidence_var)
        self.confidence_bar.grid(row=7, column=0, sticky="ew")
        self.lbl_risk = ttk.Label(self.frame, text="Foresight Risk: 0% (C:0%)")
        self.lbl_risk.grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.lbl_cpu = ttk.Label(self.frame, text="CPU: -- %")
        self.lbl_cpu.grid(row=9, column=0, sticky="w")
        self.lbl_ram = ttk.Label(self.frame, text="RAM: -- %")
        self.lbl_ram.grid(row=10, column=0, sticky="w")
        self.lbl_gpu = ttk.Label(self.frame, text="GPU: -- %")
        self.lbl_gpu.grid(row=11, column=0, sticky="w")
        self.lbl_game_cpu = ttk.Label(self.frame, text="Game CPU: -- %")
        self.lbl_game_cpu.grid(row=12, column=0, sticky="w")
        self.lbl_game_mem = ttk.Label(self.frame, text="Game RAM: -- %")
        self.lbl_game_mem.grid(row=13, column=0, sticky="w")
        self.lbl_fps = ttk.Label(self.frame, text="FPS: --")
        self.lbl_fps.grid(row=14, column=0, sticky="w")
        self.lbl_ping = ttk.Label(self.frame, text="Ping: -- ms")
        self.lbl_ping.grid(row=15, column=0, sticky="w")
        self.lbl_lt = ttk.Label(self.frame, text="Long-term CPU/RAM: -- / -- %")
        self.lbl_lt.grid(row=16, column=0, sticky="w", pady=(6, 0))
        self.learning_var_enabled = tk.BooleanVar(value=True)
        self.chk_learning = ttk.Checkbutton(
            self.frame,
            text="Learning enabled",
            variable=self.learning_var_enabled,
            command=self._on_learning_toggled
        )
        self.chk_learning.grid(row=17, column=0, sticky="w", pady=(8, 0))
        self.overlay_override_var = tk.BooleanVar(value=False)
        self.chk_overlay = ttk.Checkbutton(
            self.frame,
            text="Show Overlay HUD (override)",
            variable=self.overlay_override_var
        )
        self.chk_overlay.grid(row=18, column=0, sticky="w")
        ttk.Label(self.frame, text="Network / SMB storage:").grid(row=19, column=0, sticky="w", pady=(10, 0))
        net_row = ttk.Frame(self.frame)
        net_row.grid(row=20, column=0, sticky="ew")
        net_row.columnconfigure(0, weight=1)
        self.txt_network_path = ttk.Entry(net_row)
        initial_net = self.core.network_root or ""
        self.txt_network_path.insert(0, initial_net)
        self.txt_network_path.grid(row=0, column=0, sticky="ew")
        btn_browse = ttk.Button(net_row, text="Browse…", command=self._on_browse_network)
        btn_browse.grid(row=0, column=1, padx=(4, 0))
        net_ctrl = ttk.Frame(self.frame)
        net_ctrl.grid(row=21, column=0, sticky="ew", pady=(4, 0))
        self.network_enabled_var = tk.BooleanVar(value=self.core.network_enabled)
        chk_net = ttk.Checkbutton(
            net_ctrl, text="Enable network/SMB storage",
            variable=self.network_enabled_var,
            command=self._on_network_enable_changed
        )
        chk_net.grid(row=0, column=0, sticky="w")
        btn_test = ttk.Button(net_ctrl, text="Test connection", command=self._on_test_network)
        btn_test.grid(row=0, column=1, padx=(8, 0))
        self.lbl_net_status = ttk.Label(self.frame, text="Network: Disabled")
        self.lbl_net_status.grid(row=22, column=0, sticky="w")
        btn_summary = ttk.Button(self.frame, text="Show Session Summary", command=self.show_summary)
        btn_summary.grid(row=23, column=0, sticky="ew", pady=(10, 0))
        self.lbl_os = ttk.Label(self.frame, text="OS: Unknown")
        self.lbl_os.grid(row=24, column=0, sticky="w", pady=(6, 0))
        self.frame.rowconfigure(25, weight=1)

    def _on_learning_toggled(self):
        self.core.learning_enabled = self.learning_var_enabled.get()

    def _on_browse_network(self):
        from tkinter import filedialog
        current = self.txt_network_path.get().strip() or ""
        path = filedialog.askdirectory(initialdir=current or None, title="Select network / SMB folder")
        if path:
            self.txt_network_path.delete(0, "end")
            self.txt_network_path.insert(0, path)

    def _on_test_network(self):
        path = self.txt_network_path.get().strip()
        if not path:
            self.lbl_net_status.config(text="Network: No path specified.", foreground="orange")
            return
        ok, msg = self.core.test_network_root(path)
        self.lbl_net_status.config(text=f"Network test: {msg}", foreground="green" if ok else "red")

    def _on_network_enable_changed(self):
        enabled = self.network_enabled_var.get()
        path = self.txt_network_path.get().strip()
        ok, msg = self.core.set_network_root(path, enabled)
        color = "green" if ok and enabled else ("gray" if ok and not enabled else "red")
        self.lbl_net_status.config(text=f"Network: {msg}", foreground=color)

    def update_stats(self, stats: dict):
        self.latest_stats = stats
        game_name = stats.get("game_name", "None")
        profile_name = stats.get("profile_name", "None")
        posture = stats.get("posture", "Calm")
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        risk = stats.get("foresight_risk", 0.0)
        foresight_conf = stats.get("foresight_confidence", 0.0)
        brain_mode = stats.get("brain_mode", BrainMode.BASELINE)
        self.lbl_game.config(text=f"Game: {game_name}")
        self.lbl_profile.config(text=f"Profile: {profile_name}")
        self.lbl_posture.config(text=f"Posture: {posture}")
        self.lbl_mode.config(text=f"Brain Mode: {brain_mode}")
        self.learning_var.set(learn)
        self.lbl_learning.config(text=f"Learning: {learn}%")
        self.confidence_var.set(conf)
        self.lbl_confidence.config(text=f"Optimization Confidence: {conf}%")
        self.lbl_risk.config(text=f"Foresight Risk: {risk:.0f}% (C:{foresight_conf:.0f}%)")
        self.lbl_cpu.config(text=f"CPU: {stats.get('cpu_total', 0):.0f} %")
        self.lbl_ram.config(text=f"RAM: {stats.get('ram_percent', 0):.0f} %")
        self.lbl_gpu.config(text=f"GPU: {stats.get('gpu_total', 0):.0f} %")
        self.lbl_game_cpu.config(text=f"Game CPU: {stats.get('game_cpu', 0):.0f} %")
        self.lbl_game_mem.config(text=f"Game RAM: {stats.get('game_mem', 0):.0f} %")
        self.lbl_fps.config(text=f"FPS: {stats.get('fps', '--')}")
        self.lbl_ping.config(text=f"Ping: {stats.get('ping', '--')} ms")
        lt_cpu = stats.get("lt_cpu_avg", 0.0)
        lt_ram = stats.get("lt_ram_avg", 0.0)
        self.lbl_lt.config(text=f"Long-term CPU/RAM: {lt_cpu:.0f} / {lt_ram:.0f} %")
        os_info = stats.get("os_info", {})
        os_str = f"{os_info.get('os_name', '')} {os_info.get('os_version', '')} / {os_info.get('language', '')}"
        self.lbl_os.config(text=f"OS: {os_str}")
        net_root = stats.get("network_root", "")
        net_status = stats.get("network_status", "Disabled")
        text = f"Network: {net_status}"
        if net_root:
            text += f" ({net_root})"
        color = "red" if net_status.startswith("Error") else \
                ("green" if net_status == "Connected" else "gray")
        self.lbl_net_status.config(text=text, foreground=color)

    def show_summary(self):
        stats = self.latest_stats or {}
        game = stats.get("game_name", "None")
        profile = stats.get("profile_name", "None")
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        risk = stats.get("foresight_risk", 0.0)
        foresight_conf = stats.get("foresight_confidence", 0.0)
        cpu = stats.get("cpu_total", 0.0)
        ram = stats.get("ram_percent", 0.0)
        gpu = stats.get("gpu_total", 0.0)
        lt_cpu = stats.get("lt_cpu_avg", 0.0)
        lt_ram = stats.get("lt_ram_avg", 0.0)
        g_m = stats.get("genesis_metrics", {})
        g_state = stats.get("genesis_state", "-")
        g_posture = stats.get("genesis_posture", "-")
        g_fluid = stats.get("genesis_fluidity", 0.0)
        g_press = stats.get("genesis_pressure", 0.0)
        msg = (
            f"Game: {game}\n"
            f"Profile: {profile}\n"
            f"Learning: {learn}%\n"
            f"Optimization Confidence: {conf}%\n"
            f"Foresight Risk (last): {risk:.0f}% (C:{foresight_conf:.0f}%)\n\n"
            f"Current CPU: {cpu:.0f}% | RAM: {ram:.0f}% | GPU: {gpu:.0f}%\n"
            f"Long-term CPU avg: {lt_cpu:.0f}% | RAM avg: {lt_ram:.0f}%\n\n"
            f"Genesis state: {g_state} | posture: {g_posture}\n"
            f"Genesis fluidity: {g_fluid:.2f} | pressure: {g_press:.2f}\n"
            f"Genesis ticks/hb/thoughts: "
            f"{g_m.get('ticks', 0)}/{g_m.get('heartbeats', 0)}/{g_m.get('thoughts', 0)}\n"
        )
        messagebox.showinfo("Session Summary", msg)

# -------------------------------------------------------------------
# Main application with PanedWindow
# -------------------------------------------------------------------

class HybridApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1350x780")
        ensure_example_plugin()
        self.genesis = GenesisEngine(MEMORY_FILE)
        self.genesis.start()
        initial_network_root = NETWORK_ROOT_ENV if NETWORK_ROOT_ENV else None
        if initial_network_root:
            Path(initial_network_root).mkdir(parents=True, exist_ok=True)
        self.store = ProfileStore(LOCAL_ROOT, initial_network_root)
        self.core = OptimizerCore(self.store, self.genesis, poll_interval_ms=500)
        self._build_layout()
        self.overlay = OverlayHUD(self.root)
        self.last_stats: dict = {}
        self._schedule_tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        left_frame = ttk.Frame(self.paned)
        right_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=1)
        self.paned.add(right_frame, weight=2)
        self.optimizer_panel = OptimizerPanel(left_frame, self.core)
        self.genesis_view = GenesisView(right_frame, self.genesis)

    def _schedule_tick(self):
        self._do_tick()
        self.root.after(self.core.poll_interval_ms, self._schedule_tick)

    def _do_tick(self):
        stats = self.core.tick()
        self.last_stats = stats
        self._update_overlay_from_stats(stats)

    def _update_overlay_from_stats(self, stats: dict):
        posture = stats.get("posture", "Calm")
        risk = stats.get("foresight_risk", 0.0)
        learn = stats.get("learning_percent", 0)
        conf = stats.get("confidence_percent", 0)
        game_name = stats.get("game_name", "None")
        game_active = (game_name != "None")
        force_show = self.optimizer_panel.overlay_override_var.get()
        self.overlay.update_overlay(posture, risk, learn, conf, visible=game_active, force_show=force_show)

    def _on_close(self):
        try:
            self.genesis.stop()
        except Exception:
            pass
        self.root.destroy()

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = HybridApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


