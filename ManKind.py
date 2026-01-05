#!/usr/bin/env python3
"""
Mankind Engine – GUI Organism with Network Backup v2

One-file organism with a full GUI and backup support:

- World Watcher tab:
  - Synthetic pressures for politics / war / economy / tech
  - Recent events per domain

- Concepts tab:
  - Internal concepts, confidence, tags, and link counts
  - Recent concept events

- Covenant & Plans tab:
  - Human-centered values (care, justice, autonomy, future)
  - Plan evaluation UI
  - Self-adjusting 'future' weight when abused

- Lantern tab:
  - Journaling area
  - Reflection (emotions, themes, needs, questions, steps)
  - Value constellation over time

- Self Observer tab:
  - Aggregate activity: world, concepts, human entries, covenant adjustments

- Backup tab:
  - Choose backup folder (supports Windows SMB paths like \\SERVER\\Share\\Folder)
  - Backup now (copies entire state directory to backup folder)
  - Restore now (copies from backup folder back into local state)
  - Status display

On clean exit, it attempts an automatic backup (if backup path is configured).

Not a god, not conscious, not an oracle.
A unified skeleton whose intent is to watch, question, value, reflect, and remember.
"""

import os
import json
import time
import uuid
import random
import threading
import textwrap
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# ==========================================================================
# GLOBAL CONFIG
# ==========================================================================

STATE_DIR = "mankind_engine_state"
os.makedirs(STATE_DIR, exist_ok=True)

WORLD_DB = os.path.join(STATE_DIR, "world.json")
CONCEPTS_FILE = os.path.join(STATE_DIR, "concepts.json")
CONCEPT_HISTORY_FILE = os.path.join(STATE_DIR, "concept_history.log")
COVENANT_VALUES_FILE = os.path.join(STATE_DIR, "covenant_values.json")
COVENANT_LOG_FILE = os.path.join(STATE_DIR, "covenant_log.log")
LANTERN_ENTRIES_FILE = os.path.join(STATE_DIR, "lantern_entries.json")
LANTERN_VALUES_FILE = os.path.join(STATE_DIR, "lantern_values.json")

CONFIG_FILE = os.path.join(STATE_DIR, "backup_config.json")

WORLD_DOMAINS = [
    {"id": "politics", "name": "Politics"},
    {"id": "war", "name": "War & Conflict"},
    {"id": "economy", "name": "Economy"},
    {"id": "tech", "name": "Tech & AI"},
]

WORLD_EVENT_INTERVAL_SECONDS = 8
WORLD_PRESSURE_DECAY_PER_MIN = 0.3
WORLD_PRESSURE_MAX = 100.0
WORLD_PRESSURE_MIN = 0.0


# ==========================================================================
# UTIL
# ==========================================================================

def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap(text: str, width: int = 76) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


# ==========================================================================
# EVENT BUS
# ==========================================================================

@dataclass
class EngineEvent:
    timestamp: float
    source: str         # "world", "concept", "covenant", "lantern"
    kind: str           # "pressure_update", "world_event", "question", ...
    payload: Dict[str, Any]


class EventBus:
    def __init__(self):
        self.subscribers: List[Any] = []
        self.log: List[EngineEvent] = []
        self.lock = threading.Lock()

    def publish(self, event: EngineEvent):
        with self.lock:
            self.log.append(event)
            if len(self.log) > 500:
                self.log = self.log[-500:]
        for sub in self.subscribers:
            try:
                sub.handle_event(event)
            except Exception:
                pass

    def subscribe(self, subscriber: Any):
        self.subscribers.append(subscriber)

    def recent(self, limit: int = 50) -> List[EngineEvent]:
        with self.lock:
            return self.log[-limit:]


# ==========================================================================
# WORLD WATCHER
# ==========================================================================

@dataclass
class WorldEvent:
    timestamp: float
    domain_id: str
    title: str
    severity: float
    details: str
    source: str


class WorldState:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.pressures: Dict[str, float] = {d["id"]: 0.0 for d in WORLD_DOMAINS}
        self.events: List[WorldEvent] = []
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(WORLD_DB):
            with open(WORLD_DB, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.pressures = raw.get("pressures", self.pressures)
            self.events = [WorldEvent(**e) for e in raw.get("events", [])]

    def save(self):
        with self.lock:
            raw = {
                "pressures": self.pressures,
                "events": [asdict(e) for e in self.events[-200:]],
            }
        with open(WORLD_DB, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def add_event(self, event: WorldEvent):
        with self.lock:
            self.events.append(event)
        self.save()
        self.bus.publish(EngineEvent(
            timestamp=event.timestamp,
            source="world",
            kind="world_event",
            payload={
                "domain_id": event.domain_id,
                "title": event.title,
                "severity": event.severity,
            },
        ))

    def update_pressure(self, domain_id: str, delta: float):
        with self.lock:
            cur = self.pressures.get(domain_id, 0.0)
            new = clamp(cur + delta, WORLD_PRESSURE_MIN, WORLD_PRESSURE_MAX)
            self.pressures[domain_id] = new
        self.save()
        self.bus.publish(EngineEvent(
            timestamp=now_ts(),
            source="world",
            kind="pressure_update",
            payload={
                "domain_id": domain_id,
                "pressure": new,
                "delta": delta,
            },
        ))

    def decay(self, minutes: float):
        with self.lock:
            for did in list(self.pressures.keys()):
                cur = self.pressures[did]
                dec = WORLD_PRESSURE_DECAY_PER_MIN * minutes
                new = clamp(cur - dec, WORLD_PRESSURE_MIN, WORLD_PRESSURE_MAX)
                self.pressures[did] = new
        self.save()

    def recent_events(self, domain_id: str, limit: int = 5) -> List[WorldEvent]:
        with self.lock:
            evs = [e for e in self.events if e.domain_id == domain_id]
            return evs[-limit:]

    def get_pressures_snapshot(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.pressures)


class WorldGenerator:
    def __init__(self, domain_id: str):
        self.domain_id = domain_id

    def make_event(self) -> WorldEvent:
        sev = random.uniform(0.0, 10.0)
        if sev < 3:
            bias = "Minor disturbance"
        elif sev < 7:
            bias = "Significant development"
        else:
            bias = "Major escalation"
        title = f"{bias} in {self.domain_id}"
        details = f"Synthetic event with severity {sev:.2f} in {self.domain_id}."
        return WorldEvent(
            timestamp=now_ts(),
            domain_id=self.domain_id,
            title=title,
            severity=sev,
            details=details,
            source="synthetic",
        )


def world_loop(world: WorldState, domain_id: str, stop_flag: threading.Event):
    gen = WorldGenerator(domain_id)
    last = time.time()
    while not stop_flag.is_set():
        now = time.time()
        minutes = (now - last) / 60.0
        if minutes > 0:
            world.decay(minutes)
        last = now
        ev = gen.make_event()
        world.add_event(ev)
        world.update_pressure(domain_id, delta=ev.severity)
        time.sleep(WORLD_EVENT_INTERVAL_SECONDS)


# ==========================================================================
# AUTOGENOUS KERNEL
# ==========================================================================

@dataclass
class Concept:
    id: str
    name: str
    description: str
    confidence: float
    tags: List[str]
    links: Dict[str, float]


@dataclass
class ConceptEvent:
    timestamp: float
    kind: str
    payload: Dict[str, Any]


class ConceptMemory:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.concepts: Dict[str, Concept] = {}
        self.lock = threading.Lock()
        self._load_or_seed()

    def _load_or_seed(self):
        if os.path.exists(CONCEPTS_FILE):
            with open(CONCEPTS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for cid, cdata in raw.items():
                self.concepts[cid] = Concept(**cdata)
        else:
            self._seed()
            self._save()

    def _seed(self):
        base = [
            Concept(
                id=str(uuid.uuid4()),
                name="stability",
                description="Tendency of a system to remain similar over time.",
                confidence=0.6,
                tags=["meta", "dynamics"],
                links={},
            ),
            Concept(
                id=str(uuid.uuid4()),
                name="change",
                description="Deviation from previous state across time.",
                confidence=0.6,
                tags=["meta", "dynamics"],
                links={},
            ),
            Concept(
                id=str(uuid.uuid4()),
                name="goal",
                description="Preferred configuration of a system.",
                confidence=0.5,
                tags=["meta", "purpose"],
                links={},
            ),
        ]
        for c in base:
            self.concepts[c.id] = c

    def _save(self):
        with self.lock:
            raw = {cid: asdict(c) for cid, c in self.concepts.items()}
        with open(CONCEPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def save(self):
        self._save()

    def log_event(self, ev: ConceptEvent):
        with open(CONCEPT_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev)) + "\n")
        self.bus.publish(EngineEvent(
            timestamp=ev.timestamp,
            source="concept",
            kind=ev.kind,
            payload=ev.payload,
        ))

    def get_concepts_snapshot(self) -> List[Concept]:
        with self.lock:
            return list(self.concepts.values())


class ConceptQuestioner:
    def __init__(self, mem: ConceptMemory):
        self.mem = mem

    def pick_uncertain(self) -> Optional[Concept]:
        with self.mem.lock:
            low = [c for c in self.mem.concepts.values() if c.confidence < 0.7]
        if not low:
            return None
        return random.choice(low)

    def pick_contradiction_pair(self) -> Optional[List[Concept]]:
        with self.mem.lock:
            cs = list(self.mem.concepts.values())
        if len(cs) < 2:
            return None
        pairs = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                ci, cj = cs[i], cs[j]
                if not set(ci.tags).intersection(cj.tags):
                    pairs.append((ci, cj))
        if not pairs:
            return None
        return random.choice(pairs)

    def make_question(self) -> Optional[ConceptEvent]:
        mode = random.choice(["uncertainty", "contradiction"])
        if mode == "uncertainty":
            c = self.pick_uncertain()
            if not c:
                return None
            payload = {
                "mode": "uncertainty",
                "concept_id": c.id,
                "question": f"How can I refine the concept '{c.name}'?",
            }
        else:
            pair = self.pick_contradiction_pair()
            if not pair:
                return None
            c1, c2 = pair
            payload = {
                "mode": "contradiction",
                "concept_a": c1.id,
                "concept_b": c2.id,
                "question": f"Are '{c1.name}' and '{c2.name}' connected or opposed?",
            }
        return ConceptEvent(timestamp=now_ts(), kind="question", payload=payload)


class ConceptPlayground:
    def __init__(self, mem: ConceptMemory):
        self.mem = mem

    def run_simulation(self, q: ConceptEvent) -> ConceptEvent:
        state = {
            "stability": random.random(),
            "change": random.random(),
            "goal_progress": random.random(),
        }
        update = {}
        payload = q.payload
        if payload.get("mode") == "uncertainty":
            cid = payload["concept_id"]
            with self.mem.lock:
                c = self.mem.concepts.get(cid)
                if c:
                    delta = random.uniform(0.01, 0.05)
                    old_conf = c.confidence
                    c.confidence = min(1.0, c.confidence + delta)
                    update = {
                        "type": "confidence_boost",
                        "concept_id": cid,
                        "old_conf": old_conf,
                        "new_conf": c.confidence,
                        "delta": delta,
                    }
        elif payload.get("mode") == "contradiction":
            with self.mem.lock:
                c1 = self.mem.concepts.get(payload["concept_a"])
                c2 = self.mem.concepts.get(payload["concept_b"])
                if c1 and c2:
                    if random.random() < 0.5:
                        strength = random.uniform(0.1, 0.5)
                        c1.links[c2.id] = strength
                        c2.links[c1.id] = strength
                        update = {
                            "type": "link",
                            "a": c1.id,
                            "b": c2.id,
                            "strength": strength,
                        }
                    else:
                        c1.links.pop(c2.id, None)
                        c2.links.pop(c1.id, None)
                        update = {
                            "type": "separate",
                            "a": c1.id,
                            "b": c2.id,
                        }
        coherence = 1.0 - abs(state["stability"] - state["change"])
        self.mem.save()
        return ConceptEvent(
            timestamp=now_ts(),
            kind="simulation_result",
            payload={
                "question": q.payload,
                "state": state,
                "coherence": coherence,
                "update": update,
            },
        )


def concept_loop(mem: ConceptMemory, stop_flag: threading.Event):
    questioner = ConceptQuestioner(mem)
    playground = ConceptPlayground(mem)
    while not stop_flag.is_set():
        q = questioner.make_question()
        if q:
            mem.log_event(q)
            res = playground.run_simulation(q)
            mem.log_event(res)
        time.sleep(3.0)


# ==========================================================================
# COVENANT CORE & HUMAN-FIRST STEWARD
# ==========================================================================

@dataclass
class ValueDimension:
    name: str
    description: str
    weight: float
    history: List[Dict[str, Any]]


class CovenantState:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.values: Dict[str, ValueDimension] = {}
        self._load_or_seed()

    def _load_or_seed(self):
        if os.path.exists(COVENANT_VALUES_FILE):
            with open(COVENANT_VALUES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, data in raw.items():
                self.values[name] = ValueDimension(
                    name=data["name"],
                    description=data["description"],
                    weight=data["weight"],
                    history=data.get("history", []),
                )
        else:
            now = now_ts()
            self.values = {
                "care": ValueDimension(
                    name="care",
                    description="Protecting health, safety, emotional well-being.",
                    weight=1.5,
                    history=[{"timestamp": now, "reason": "initial", "weight": 1.5}],
                ),
                "justice": ValueDimension(
                    name="justice",
                    description="Fairness, non-exploitation, power balance.",
                    weight=1.3,
                    history=[{"timestamp": now, "reason": "initial", "weight": 1.3}],
                ),
                "autonomy": ValueDimension(
                    name="autonomy",
                    description="Freedom, informed consent, self-determination.",
                    weight=1.2,
                    history=[{"timestamp": now, "reason": "initial", "weight": 1.2}],
                ),
                "future": ValueDimension(
                    name="future",
                    description="Long-term sustainability and risk avoidance.",
                    weight=1.4,
                    history=[{"timestamp": now, "reason": "initial", "weight": 1.4}],
                ),
            }
            self._save()

    def _save(self):
        raw = {k: asdict(v) for k, v in self.values.items()}
        with open(COVENANT_VALUES_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def log(self, event: Dict[str, Any]):
        event["timestamp"] = now_ts()
        with open(COVENANT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        self.bus.publish(EngineEvent(
            timestamp=event["timestamp"],
            source="covenant",
            kind=event["type"],
            payload=event,
        ))

    def update_weight(self, name: str, new_weight: float, reason: str):
        v = self.values[name]
        v.weight = new_weight
        v.history.append({"timestamp": now_ts(), "reason": reason, "weight": new_weight})
        self._save()
        self.log({
            "type": "weight_update",
            "dimension": name,
            "new_weight": new_weight,
            "reason": reason,
        })

    def recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not os.path.exists(COVENANT_LOG_FILE):
            return []
        out = []
        with open(COVENANT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line.strip())
                    out.append(ev)
                except json.JSONDecodeError:
                    continue
        return out[-limit:]


@dataclass
class Plan:
    name: str
    description: str
    impacts: Dict[str, int]  # dimension -> -10 .. +10
    money_cost: float


class HumanFirstSteward:
    def __init__(self, cov: CovenantState):
        self.cov = cov

    def score_plan(self, plan: Plan) -> Dict[str, Any]:
        total = 0.0
        breakdown: Dict[str, Any] = {}
        for dim_name, impact in plan.impacts.items():
            if dim_name not in self.cov.values:
                continue
            dim = self.cov.values[dim_name]
            contribution = impact * dim.weight
            breakdown[dim_name] = {
                "impact": impact,
                "weight": dim.weight,
                "contribution": contribution,
                "description": dim.description,
            }
            total += contribution
        result = {
            "plan": {
                "name": plan.name,
                "description": plan.description,
                "money_cost": plan.money_cost,
            },
            "total_score": total,
            "breakdown": breakdown,
        }
        self.cov.log({"type": "evaluation", "result": result})
        return result

    def reflect_and_adjust(self):
        events = self.cov.recent_events(limit=50)
        if not events:
            return
        window = events[-20:]
        future_ignored = 0
        for ev in window:
            if ev.get("type") != "evaluation":
                continue
            res = ev["result"]
            total = res["total_score"]
            breakdown = res["breakdown"]
            f = breakdown.get("future")
            if not f:
                continue
            if total > 0 and f["impact"] < 0:
                future_ignored += 1
        if future_ignored >= 3:
            v = self.cov.values["future"]
            new_weight = v.weight + 0.1
            self.cov.update_weight(
                "future",
                new_weight,
                reason=f"Detected {future_ignored} scores ignoring future harm.",
            )


# ==========================================================================
# LANTERN (JOURNAL)
# ==========================================================================

@dataclass
class LanternEntry:
    timestamp: float
    text: str
    analysis: Dict[str, Any]
    reflection: Dict[str, Any]


@dataclass
class LanternValueNode:
    name: str
    weight: float
    history: List[Dict[str, Any]]


class LanternState:
    def __init__(self):
        self.entries: List[LanternEntry] = []
        self.values: Dict[str, LanternValueNode] = {}
        self._load()

    def _load(self):
        if os.path.exists(LANTERN_ENTRIES_FILE):
            with open(LANTERN_ENTRIES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for e in raw:
                self.entries.append(LanternEntry(
                    timestamp=e["timestamp"],
                    text=e["text"],
                    analysis=e["analysis"],
                    reflection=e["reflection"],
                ))
        if os.path.exists(LANTERN_VALUES_FILE):
            with open(LANTERN_VALUES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, v in raw.items():
                self.values[name] = LanternValueNode(
                    name=v["name"],
                    weight=v["weight"],
                    history=v["history"],
                )

    def save(self):
        with open(LANTERN_ENTRIES_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)
        with open(LANTERN_VALUES_FILE, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.values.items()}, f, indent=2)

    def record_entry(self, entry: LanternEntry):
        self.entries.append(entry)
        self.save()

    def update_value(self, name: str, delta: float, reason: str):
        now = now_ts()
        if name not in self.values:
            self.values[name] = LanternValueNode(
                name=name,
                weight=max(0.0, delta),
                history=[{"timestamp": now, "delta": delta, "reason": reason}],
            )
        else:
            v = self.values[name]
            v.weight = max(0.0, v.weight + delta)
            v.history.append({"timestamp": now, "delta": delta, "reason": reason})
        self.save()

    def top_values(self, n: int = 5) -> List[LanternValueNode]:
        return sorted(self.values.values(), key=lambda v: v.weight, reverse=True)[:n]


EMOTION_KEYWORDS = {
    "hope": ["hope", "dream", "light", "future", "grow", "build", "create"],
    "fear": ["afraid", "fear", "scared", "terrified", "danger", "threat", "risk"],
    "anger": ["angry", "rage", "furious", "hate", "injustice", "unfair"],
    "sadness": ["sad", "alone", "empty", "loss", "grief", "tired", "exhausted"],
    "confusion": ["confused", "lost", "don't know", "no idea", "uncertain"],
    "determination": ["fight", "refuse", "won't give up", "determined", "driven"],
}

THEME_KEYWORDS = {
    "self": [" I ", " me ", " myself "],
    "others": ["they", "them", "friends", "family", "people"],
    "future": ["future", "tomorrow", "next year", "years", "decades"],
    "meaning": ["meaning", "purpose", "why", "worth"],
    "struggle": ["struggle", "battle", "hard", "difficult", "pain"],
    "growth": ["grow", "improve", "learn", "change", "evolve"],
}

VALUE_HOOKS = {
    "connection": ["alone", "lonely", "friends", "family", "together", "community"],
    "freedom": ["trapped", "stuck", "free", "freedom", "control"],
    "creation": ["build", "create", "make", "design", "invent"],
    "justice": ["unfair", "injustice", "justice", "wrong", "right"],
    "truth": ["truth", "lies", "honest", "real"],
    "safety": ["danger", "safe", "risk", "threat"],
    "growth": ["grow", "improve", "learn", "evolve"],
}


def detect_emotions(text: str) -> Dict[str, float]:
    tl = text.lower()
    scores = {k: 0.0 for k in EMOTION_KEYWORDS}
    for emo, words in EMOTION_KEYWORDS.items():
        for w in words:
            if w in tl:
                scores[emo] += 1.0
    total = sum(scores.values())
    if total > 0:
        for emo in scores:
            scores[emo] /= total
    return scores


def detect_themes(text: str) -> Dict[str, float]:
    tl = " " + text.lower() + " "
    scores = {k: 0.0 for k in THEME_KEYWORDS}
    for theme, words in THEME_KEYWORDS.items():
        for w in words:
            if w.lower() in tl:
                scores[theme] += 1.0
    total = sum(scores.values())
    if total > 0:
        for theme in scores:
            scores[theme] /= total
    return scores


def detect_values(text: str) -> Dict[str, float]:
    tl = text.lower()
    scores = {k: 0.0 for k in VALUE_HOOKS}
    for val, words in VALUE_HOOKS.items():
        for w in words:
            if w.lower() in tl:
                scores[val] += 1.0
    return scores


def summarize_emotions(emotions: Dict[str, float]) -> str:
    if not emotions or sum(emotions.values()) == 0:
        return "Your emotional state is not fully clear here, but something important is moving inside you."
    sorted_emos = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top = [e for e in sorted_emos if e[1] > 0.2]
    parts = [f"{name} ({intensity:.2f})" for name, intensity in top]
    return "The main emotional tones I sense here are: " + ", ".join(parts) + "."


def summarize_themes(themes: Dict[str, float]) -> str:
    if not themes or sum(themes.values()) == 0:
        return "Your words touch many parts of life at once; the themes are wide and intertwined."
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    top = [t for t in sorted_themes if t[1] > 0.2]
    names = [t[0] for t in top]
    return "You seem to be focused on themes of: " + ", ".join(names) + "."


def infer_needs(emotions: Dict[str, float], themes: Dict[str, float]) -> List[str]:
    needs: List[str] = []
    if emotions.get("sadness", 0) > 0.2 or themes.get("self", 0) > 0.3:
        needs.append("being seen and accepted as you are")
    if emotions.get("fear", 0) > 0.2 or themes.get("future", 0) > 0.3:
        needs.append("a sense of safety about the future")
    if themes.get("meaning", 0) > 0.2:
        needs.append("a sense of purpose and meaning")
    if themes.get("growth", 0) > 0.2:
        needs.append("space and support to grow and change")
    if themes.get("others", 0) > 0.2:
        needs.append("connection with others who understand you")
    if not needs:
        needs.append("clarity about what matters most to you right now")
    return list(dict.fromkeys(needs))


def generate_questions(needs: List[str]) -> List[str]:
    qs: List[str] = []
    for n in needs:
        if "seen and accepted" in n:
            qs.append("What would it look like, in one small scene, to be fully seen and accepted?")
        if "safety about the future" in n:
            qs.append("What is one small action that would make tomorrow feel slightly safer or more grounded?")
        if "purpose and meaning" in n:
            qs.append("When have you felt that your life mattered, even in a small way?")
        if "grow and change" in n:
            qs.append("What is one tiny change you could make this week that your future self would thank you for?")
        if "connection with others" in n:
            qs.append("Is there one person you could reach out to, even quietly, to share a piece of what you feel?")
        if "clarity about what matters" in n:
            qs.append("If you had to name just one thing that matters to you today, what would it be?")
    if not qs:
        qs.append("What is the question your heart keeps circling but never quite asks out loud?")
    return qs


def generate_steps(needs: List[str]) -> List[str]:
    steps: List[str] = []
    for n in needs:
        if "seen and accepted" in n:
            steps.append("Write a few sentences to yourself as if you were your own best ally, without criticism.")
        if "safety about the future" in n:
            steps.append("Choose one small practical task that reduces chaos and do it gently.")
        if "purpose and meaning" in n:
            steps.append("Spend 10 minutes doing something that feels meaningful, even if nobody else sees it.")
        if "grow and change" in n:
            steps.append("Pick a tiny habit that serves your future self, and try it once today.")
        if "connection with others" in n:
            steps.append("Send a simple, honest message to someone you trust, even if it's just: 'Thinking of you.'")
        if "clarity about what matters" in n:
            steps.append("Write a short list titled 'What matters to me right now' with 3 honest items.")
    if not steps:
        steps.append("Take 3 slow breaths, then write freely for 5 minutes about what you truly want to move toward.")
    return list(dict.fromkeys(steps))


class Lantern:
    def __init__(self, bus: EventBus):
        self.state = LanternState()
        self.bus = bus

    def process_entry(self, text: str) -> LanternEntry:
        emotions = detect_emotions(text)
        themes = detect_themes(text)
        value_signals = detect_values(text)

        for val_name, count in value_signals.items():
            if count > 0:
                self.state.update_value(
                    val_name,
                    delta=0.1 * count,
                    reason="Detected in journal entry",
                )

        emotional_summary = summarize_emotions(emotions)
        theme_summary = summarize_themes(themes)
        needs = infer_needs(emotions, themes)
        questions = generate_questions(needs)
        steps = generate_steps(needs)

        analysis = {
            "emotions": emotions,
            "themes": themes,
            "values_detected": value_signals,
        }
        reflection = {
            "emotional_summary": emotional_summary,
            "theme_summary": theme_summary,
            "needs": needs,
            "questions": questions,
            "steps": steps,
        }

        entry = LanternEntry(
            timestamp=now_ts(),
            text=text,
            analysis=analysis,
            reflection=reflection,
        )
        self.state.record_entry(entry)
        self.bus.publish(EngineEvent(
            timestamp=entry.timestamp,
            source="lantern",
            kind="journal_entry",
            payload={"analysis": analysis, "reflection": reflection},
        ))
        return entry

    def top_values_str(self) -> str:
        top = self.state.top_values()
        if not top:
            return "Your deeper values are still forming in this lantern's view."
        lines = ["Values your entries seem to circle around:"]
        for v in top:
            lines.append(f"- {v.name} (weight {v.weight:.2f})")
        return "\n".join(lines)


# ==========================================================================
# SELF OBSERVER
# ==========================================================================

class SelfObserver:
    def __init__(self):
        self.world_activity = 0
        self.concept_activity = 0
        self.human_activity = 0
        self.covenant_adjustments = 0
        self.last_summary_ts = 0.0
        self.lock = threading.Lock()

    def handle_event(self, event: EngineEvent):
        with self.lock:
            if event.source == "world":
                if event.kind in ("world_event", "pressure_update"):
                    self.world_activity += 1
            elif event.source == "concept":
                if event.kind in ("question", "simulation_result"):
                    self.concept_activity += 1
            elif event.source == "lantern":
                if event.kind == "journal_entry":
                    self.human_activity += 1
            elif event.source == "covenant":
                if event.kind == "weight_update":
                    self.covenant_adjustments += 1

    def snapshot(self) -> str:
        with self.lock:
            now = now_ts()
            dt = now - self.last_summary_ts if self.last_summary_ts > 0 else None
            self.last_summary_ts = now
            lines = []
            lines.append("SELF OBSERVER SNAPSHOT")
            lines.append("-" * 40)
            if dt:
                lines.append(f"Time since last snapshot: {dt:.1f} seconds")
            lines.append(f"World activity events       : {self.world_activity}")
            lines.append(f"Concept self-play events    : {self.concept_activity}")
            lines.append(f"Human journal interactions  : {self.human_activity}")
            lines.append(f"Covenant weight adjustments : {self.covenant_adjustments}")
            return "\n".join(lines)


# ==========================================================================
# BACKUP MANAGER (LOCAL + SMB)
# ==========================================================================

class BackupManager:
    """
    Handles backup and restore of the entire state directory.

    - backup_path is stored in backup_config.json
    - Works with local paths or Windows SMB paths (\\SERVER\\Share\\Folder)
    """

    def __init__(self):
        self.backup_path: Optional[str] = None
        self.status: str = ""
        self._load_config()

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.backup_path = data.get("backup_path") or None
            except Exception:
                self.backup_path = None
        else:
            self.backup_path = None

    def _save_config(self):
        data = {"backup_path": self.backup_path}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def set_backup_path(self, path: str):
        self.backup_path = path
        self._save_config()
        self.status = f"Backup path set to: {path}"

    def backup_now(self) -> bool:
        if not self.backup_path:
            self.status = "No backup path set."
            return False
        try:
            if not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path, exist_ok=True)
            # Copy all files from STATE_DIR to backup_path
            for root, dirs, files in os.walk(STATE_DIR):
                rel = os.path.relpath(root, STATE_DIR)
                target_root = os.path.join(self.backup_path, rel) if rel != "." else self.backup_path
                os.makedirs(target_root, exist_ok=True)
                for fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(target_root, fname)
                    shutil.copy2(src, dst)
            self.status = f"Backup completed to {self.backup_path} at {now_iso()}"
            return True
        except Exception as e:
            self.status = f"Backup failed: {e}"
            return False

    def restore_now(self) -> bool:
        if not self.backup_path:
            self.status = "No backup path set."
            return False
        try:
            if not os.path.exists(self.backup_path):
                self.status = "Backup path does not exist."
                return False
            # Copy all files from backup_path to STATE_DIR
            for root, dirs, files in os.walk(self.backup_path):
                rel = os.path.relpath(root, self.backup_path)
                target_root = os.path.join(STATE_DIR, rel) if rel != "." else STATE_DIR
                os.makedirs(target_root, exist_ok=True)
                for fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(target_root, fname)
                    shutil.copy2(src, dst)
            self.status = f"Restore completed from {self.backup_path} at {now_iso()}"
            return True
        except Exception as e:
            self.status = f"Restore failed: {e}"
            return False


# ==========================================================================
# GUI
# ==========================================================================

class MankindEngineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mankind Engine – GUI Organism v2")
        self.root.geometry("1200x750")

        # Core
        self.bus = EventBus()
        self.observer = SelfObserver()
        self.bus.subscribe(self.observer)

        self.backup = BackupManager()

        # Attempt restore from backup on startup (if any)
        # This is soft: if restore fails, local state remains.
        if self.backup.backup_path:
            self.backup.restore_now()

        self.world = WorldState(self.bus)
        self.concept_mem = ConceptMemory(self.bus)
        self.cov = CovenantState(self.bus)
        self.steward = HumanFirstSteward(self.cov)
        self.lantern = Lantern(self.bus)

        self.stop_flag = threading.Event()
        self._start_background_loops()

        self._build_gui()
        self._start_periodic_refresh()

    def _start_background_loops(self):
        self.world_threads = []
        for d in WORLD_DOMAINS:
            t = threading.Thread(target=world_loop, args=(self.world, d["id"], self.stop_flag), daemon=True)
            t.start()
            self.world_threads.append(t)
        self.concept_thread = threading.Thread(target=concept_loop, args=(self.concept_mem, self.stop_flag), daemon=True)
        self.concept_thread.start()

    def _build_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.world_frame = ttk.Frame(notebook)
        self.concept_frame = ttk.Frame(notebook)
        self.covenant_frame = ttk.Frame(notebook)
        self.lantern_frame = ttk.Frame(notebook)
        self.observer_frame = ttk.Frame(notebook)
        self.backup_frame = ttk.Frame(notebook)

        notebook.add(self.world_frame, text="World")
        notebook.add(self.concept_frame, text="Concepts")
        notebook.add(self.covenant_frame, text="Covenant & Plans")
        notebook.add(self.lantern_frame, text="Lantern")
        notebook.add(self.observer_frame, text="Self Observer")
        notebook.add(self.backup_frame, text="Backup")

        self._build_world_tab()
        self._build_concept_tab()
        self._build_covenant_tab()
        self._build_lantern_tab()
        self._build_observer_tab()
        self._build_backup_tab()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------- World tab -------------

    def _build_world_tab(self):
        top = ttk.Frame(self.world_frame)
        top.pack(fill="x", padx=5, pady=5)

        ttk.Label(top, text="World pressures (synthetic)", font=("Arial", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh now", command=self.refresh_world_tab).pack(side="right")

        middle = ttk.Frame(self.world_frame)
        middle.pack(fill="both", expand=True, padx=5, pady=5)

        self.world_tree = ttk.Treeview(middle, columns=("pressure",), show="headings", height=5)
        self.world_tree.heading("pressure", text="Pressure")
        self.world_tree.column("pressure", width=150, anchor="center")
        self.world_tree.pack(side="top", fill="x")

        for d in WORLD_DOMAINS:
            self.world_tree.insert("", "end", iid=d["id"], values=(0.0,))

        bottom = ttk.Frame(self.world_frame)
        bottom.pack(fill="both", expand=True, padx=5, pady=5)

        self.world_events_text = scrolledtext.ScrolledText(bottom, wrap="word", height=15)
        self.world_events_text.pack(fill="both", expand=True)

        self.refresh_world_tab()

    def refresh_world_tab(self):
        pressures = self.world.get_pressures_snapshot()
        for d in WORLD_DOMAINS:
            did = d["id"]
            p = pressures.get(did, 0.0)
            if self.world_tree.exists(did):
                self.world_tree.item(did, values=(f"{p:.2f}",))

        self.world_events_text.delete("1.0", "end")
        for d in WORLD_DOMAINS:
            self.world_events_text.insert("end", f"[{d['name']}]\n")
            events = self.world.recent_events(d["id"], limit=5)
            if not events:
                self.world_events_text.insert("end", "  (no events yet)\n\n")
            else:
                for ev in events:
                    ts = time.strftime("%H:%M:%S", time.gmtime(ev.timestamp))
                    self.world_events_text.insert(
                        "end",
                        f"  - [{ts}] ({ev.severity:.2f}) {ev.title}\n"
                    )
                self.world_events_text.insert("end", "\n")

    # ------------- Concept tab -------------

    def _build_concept_tab(self):
        top = ttk.Frame(self.concept_frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Label(top, text="Autogenous Concepts", font=("Arial", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_concept_tab).pack(side="right")

        middle = ttk.Frame(self.concept_frame)
        middle.pack(fill="both", expand=True, padx=5, pady=5)

        self.concept_tree = ttk.Treeview(
            middle,
            columns=("name", "confidence", "tags", "links"),
            show="headings",
            height=10,
        )
        self.concept_tree.heading("name", text="Name")
        self.concept_tree.heading("confidence", text="Confidence")
        self.concept_tree.heading("tags", text="Tags")
        self.concept_tree.heading("links", text="Link count")
        self.concept_tree.column("name", width=140)
        self.concept_tree.column("confidence", width=100, anchor="center")
        self.concept_tree.column("tags", width=220)
        self.concept_tree.column("links", width=80, anchor="center")
        self.concept_tree.pack(fill="x")

        bottom = ttk.Frame(self.concept_frame)
        bottom.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(bottom, text="Recent concept events:").pack(anchor="w")
        self.concept_events_text = scrolledtext.ScrolledText(bottom, wrap="word", height=12)
        self.concept_events_text.pack(fill="both", expand=True)

        self.refresh_concept_tab()

    def refresh_concept_tab(self):
        for item in self.concept_tree.get_children():
            self.concept_tree.delete(item)
        concepts = self.concept_mem.get_concepts_snapshot()
        for c in concepts:
            self.concept_tree.insert(
                "",
                "end",
                iid=c.id,
                values=(
                    c.name,
                    f"{c.confidence:.2f}",
                    ",".join(c.tags),
                    len(c.links),
                ),
            )
        self.concept_events_text.delete("1.0", "end")
        if os.path.exists(CONCEPT_HISTORY_FILE):
            try:
                with open(CONCEPT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-40:]
                for line in lines:
                    self.concept_events_text.insert("end", line)
            except Exception:
                pass

    # ------------- Covenant & Plans tab -------------

    def _build_covenant_tab(self):
        top = ttk.Frame(self.covenant_frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Label(top, text="Covenant Core – Human Values", font=("Arial", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_covenant_tab).pack(side="right")

        middle = ttk.Frame(self.covenant_frame)
        middle.pack(fill="x", padx=5, pady=5)

        self.cov_tree = ttk.Treeview(
            middle,
            columns=("description", "weight"),
            show="headings",
            height=4,
        )
        self.cov_tree.heading("description", text="Description")
        self.cov_tree.heading("weight", text="Weight")
        self.cov_tree.column("description", width=500)
        self.cov_tree.column("weight", width=80, anchor="center")
        self.cov_tree.pack(fill="x")

        plan_frame = ttk.LabelFrame(self.covenant_frame, text="Evaluate a Plan")
        plan_frame.pack(fill="both", expand=True, padx=5, pady=5)

        form = ttk.Frame(plan_frame)
        form.pack(fill="x", padx=5, pady=5)

        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky="w")
        self.plan_name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.plan_name_var, width=40).grid(row=0, column=1, sticky="w")

        ttk.Label(form, text="Description:").grid(row=1, column=0, sticky="nw")
        self.plan_desc_text = scrolledtext.ScrolledText(form, width=60, height=4, wrap="word")
        self.plan_desc_text.grid(row=1, column=1, sticky="w")

        ttk.Label(form, text="Money cost (info only):").grid(row=2, column=0, sticky="w")
        self.plan_cost_var = tk.StringVar(value="0")
        ttk.Entry(form, textvariable=self.plan_cost_var, width=20).grid(row=2, column=1, sticky="w")

        impacts_frame = ttk.Frame(plan_frame)
        impacts_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(impacts_frame, text="Impacts (-10 very harmful, +10 very beneficial)").grid(row=0, column=0, columnspan=2, sticky="w")

        self.impact_vars = {}
        dims = ["care", "justice", "autonomy", "future"]
        for i, dname in enumerate(dims):
            ttk.Label(impacts_frame, text=dname.capitalize() + ":").grid(row=i+1, column=0, sticky="w")
            var = tk.StringVar(value="0")
            self.impact_vars[dname] = var
            ttk.Entry(impacts_frame, textvariable=var, width=5).grid(row=i+1, column=1, sticky="w")

        buttons = ttk.Frame(plan_frame)
        buttons.pack(fill="x", padx=5, pady=5)

        ttk.Button(buttons, text="Evaluate Plan", command=self.evaluate_plan_gui).pack(side="left")
        ttk.Button(buttons, text="Reflect & Adjust (future weight)", command=self.reflect_covenant_gui).pack(side="left", padx=10)

        result_frame = ttk.Frame(plan_frame)
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(result_frame, text="Last Plan Evaluation:").pack(anchor="w")
        self.plan_result_text = scrolledtext.ScrolledText(result_frame, wrap="word", height=12)
        self.plan_result_text.pack(fill="both", expand=True)

        self.refresh_covenant_tab()

    def refresh_covenant_tab(self):
        for item in self.cov_tree.get_children():
            self.cov_tree.delete(item)
        for name, dim in self.cov.values.items():
            self.cov_tree.insert(
                "",
                "end",
                iid=name,
                values=(dim.description, f"{dim.weight:.2f}"),
            )

    def evaluate_plan_gui(self):
        name = self.plan_name_var.get().strip() or "Unnamed Plan"
        desc = self.plan_desc_text.get("1.0", "end").strip() or "(no description)"
        try:
            cost = float(self.plan_cost_var.get().strip())
        except ValueError:
            cost = 0.0
        impacts = {}
        for dname, var in self.impact_vars.items():
            try:
                impacts[dname] = int(var.get().strip())
            except ValueError:
                impacts[dname] = 0
        plan = Plan(name=name, description=desc, impacts=impacts, money_cost=cost)
        res = self.steward.score_plan(plan)
        self._show_plan_result(res)

    def _show_plan_result(self, res: Dict[str, Any]):
        self.plan_result_text.delete("1.0", "end")
        self.plan_result_text.insert("end", f"Plan: {res['plan']['name']}\n")
        self.plan_result_text.insert("end", f"Description: {res['plan']['description']}\n")
        self.plan_result_text.insert("end", f"Money cost (info): {res['plan']['money_cost']}\n")
        self.plan_result_text.insert("end", f"Human-centered score: {res['total_score']:.2f}\n\n")
        self.plan_result_text.insert("end", "Breakdown:\n")
        for dim, data in res["breakdown"].items():
            self.plan_result_text.insert(
                "end",
                f"  {dim.upper()}: impact={data['impact']}, "
                f"weight={data['weight']:.2f}, "
                f"contribution={data['contribution']:.2f}\n"
            )

    def reflect_covenant_gui(self):
        self.steward.reflect_and_adjust()
        self.refresh_covenant_tab()
        messagebox.showinfo("Covenant", "Reflected on recent evaluations and adjusted 'future' weight if needed.")

    # ------------- Lantern tab -------------

    def _build_lantern_tab(self):
        top = ttk.Frame(self.lantern_frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Label(top, text="Lantern – Human Growth Companion", font=("Arial", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Clear reflection", command=self.clear_lantern_reflection).pack(side="right")

        main = ttk.Frame(self.lantern_frame)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ttk.Label(left, text="Write from where you are now:").pack(anchor="w")
        self.lantern_input = scrolledtext.ScrolledText(left, wrap="word", height=12)
        self.lantern_input.pack(fill="both", expand=True)

        ttk.Button(left, text="Process Entry", command=self.process_lantern_entry).pack(anchor="e", pady=5)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        ttk.Label(right, text="Reflection:").pack(anchor="w")
        self.lantern_reflection = scrolledtext.ScrolledText(right, wrap="word", height=16)
        self.lantern_reflection.pack(fill="both", expand=True)

        ttk.Label(right, text="Value constellation:").pack(anchor="w", pady=(5, 0))
        self.lantern_values_label = ttk.Label(right, text="", justify="left")
        self.lantern_values_label.pack(anchor="w")

        self.update_lantern_values_label()

    def clear_lantern_reflection(self):
        self.lantern_reflection.delete("1.0", "end")

    def process_lantern_entry(self):
        text = self.lantern_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Lantern", "Please write something first.")
            return
        entry = self.lantern.process_entry(text)
        self.lantern_input.delete("1.0", "end")
        self.lantern_reflection.delete("1.0", "end")

        self.lantern_reflection.insert("end", wrap(entry.reflection["emotional_summary"]) + "\n\n")
        self.lantern_reflection.insert("end", wrap(entry.reflection["theme_summary"]) + "\n\n")
        self.lantern_reflection.insert("end", "Needs sensed underneath your words:\n")
        for n in entry.reflection["needs"]:
            self.lantern_reflection.insert("end", f" - {n}\n")
        self.lantern_reflection.insert("end", "\nQuestions you might sit with:\n")
        for q in entry.reflection["questions"]:
            self.lantern_reflection.insert("end", f" - {q}\n")
        self.lantern_reflection.insert("end", "\nSmall steps you could consider:\n")
        for s in entry.reflection["steps"]:
            self.lantern_reflection.insert("end", f" - {s}\n")

        self.update_lantern_values_label()

    def update_lantern_values_label(self):
        self.lantern_values_label.config(text=self.lantern.top_values_str())

    # ------------- Observer tab -------------

    def _build_observer_tab(self):
        top = ttk.Frame(self.observer_frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Label(top, text="Self Observer", font=("Arial", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_observer_tab).pack(side="right")

        self.observer_text = scrolledtext.ScrolledText(self.observer_frame, wrap="word")
        self.observer_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.refresh_observer_tab()

    def refresh_observer_tab(self):
        snapshot = self.observer.snapshot()
        self.observer_text.delete("1.0", "end")
        self.observer_text.insert("end", snapshot)

    # ------------- Backup tab -------------

    def _build_backup_tab(self):
        top = ttk.Frame(self.backup_frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Label(top, text="Backup & Restore", font=("Arial", 12, "bold")).pack(side="left")

        mid = ttk.Frame(self.backup_frame)
        mid.pack(fill="x", padx=5, pady=5)

        ttk.Label(mid, text="Current backup path:").grid(row=0, column=0, sticky="w")
        self.backup_path_var = tk.StringVar(value=self.backup.backup_path or "(not set)")
        self.backup_path_label = ttk.Label(mid, textvariable=self.backup_path_var)
        self.backup_path_label.grid(row=0, column=1, sticky="w")

        ttk.Button(mid, text="Select backup folder", command=self.select_backup_folder).grid(row=1, column=0, pady=5, sticky="w")

        buttons = ttk.Frame(self.backup_frame)
        buttons.pack(fill="x", padx=5, pady=5)

        ttk.Button(buttons, text="Backup now", command=self.backup_now_gui).pack(side="left", padx=5)
        ttk.Button(buttons, text="Restore now", command=self.restore_now_gui).pack(side="left", padx=5)

        status_frame = ttk.Frame(self.backup_frame)
        status_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(status_frame, text="Backup status:").pack(anchor="w")
        self.backup_status_text = scrolledtext.ScrolledText(status_frame, wrap="word", height=10)
        self.backup_status_text.pack(fill="both", expand=True)

        self.update_backup_status("Backup manager ready.")

    def select_backup_folder(self):
        path = filedialog.askdirectory(title="Select backup folder (can be SMB path)")
        if path:
            self.backup.set_backup_path(path)
            self.backup_path_var.set(path)
            self.update_backup_status(f"Backup path set to: {path}")

    def backup_now_gui(self):
        ok = self.backup.backup_now()
        self.update_backup_status(self.backup.status)
        if ok:
            messagebox.showinfo("Backup", "Backup completed successfully.")
        else:
            messagebox.showerror("Backup", f"Backup failed.\n\n{self.backup.status}")

    def restore_now_gui(self):
        ok = self.backup.restore_now()
        self.update_backup_status(self.backup.status)
        if ok:
            messagebox.showinfo("Restore", "Restore completed successfully.\nRestart the program to fully reload state.")
        else:
            messagebox.showerror("Restore", f"Restore failed.\n\n{self.backup.status}")

    def update_backup_status(self, msg: str):
        self.backup_status_text.insert("end", f"[{now_iso()}] {msg}\n")
        self.backup_status_text.see("end")

    # ------------- Periodic refresh -------------

    def _start_periodic_refresh(self):
        def tick():
            self.refresh_world_tab()
            self.refresh_concept_tab()
            self.refresh_observer_tab()
            self.root.after(5000, tick)
        self.root.after(5000, tick)

    # ------------- Close -------------

    def on_close(self):
        if messagebox.askokcancel("Quit", "Shut down Mankind Engine? It will attempt a final backup if configured."):
            # Attempt backup on exit
            if self.backup.backup_path:
                self.backup.backup_now()
            self.stop_flag.set()
            time.sleep(0.5)
            self.root.destroy()


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    root = tk.Tk()
    app = MankindEngineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

