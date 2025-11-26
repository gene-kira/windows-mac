import os
import json
import random
import asyncio
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------- Simulation Engine ---------------- #

SYLLABLES = ["ga", "ru", "ko", "mi", "ta", "so", "ne", "da", "zi", "lu", "ha", "po"]

@dataclass
class Config:
    objects: List[str]
    num_agents_per_tribe: int = 8
    rounds_between_report: int = 800
    invention_prob: float = 0.4
    success_reward: float = 1.2
    failure_penalty: float = 0.3
    decay: float = 0.0008
    agent_variance: float = 0.2
    symbolic_recursion_prob: float = 0.1
    meaning_drift_interval: int = 800
    tribe_contact_round: int = 2500

def invent_grunt():
    n = random.choice([2, 3])
    return "".join(random.sample(SYLLABLES, n))

class Agent:
    def __init__(self, name: str, objects: List[str], variance: float):
        self.name = name
        self.lexicon: Dict[str, Counter] = {obj: Counter() for obj in objects}
        self.variance = variance

    def known_names(self, obj: str):
        return self.lexicon[obj]

    def choose_name_to_speak(self, obj: str) -> Optional[str]:
        names = self.known_names(obj)
        if not names:
            return None
        total = sum(names.values())
        r = random.random() * total
        cumulative = 0.0
        for w, weight in names.items():
            cumulative += max(0.0, weight * (1.0 + random.uniform(-self.variance, self.variance)))
            if r <= cumulative:
                return w
        return max(names.items(), key=lambda x: x[1])[0]

    def hear(self, obj: str, word: str, success: bool, cfg: Config):
        if success:
            self.lexicon[obj][word] += cfg.success_reward
        else:
            self.lexicon[obj][word] = max(0.0, self.lexicon[obj][word] - cfg.failure_penalty)
        for w in list(self.lexicon[obj].keys()):
            self.lexicon[obj][w] = max(0.0, self.lexicon[obj][w] * (1.0 - cfg.decay))
            if self.lexicon[obj][w] < 1e-6:
                del self.lexicon[obj][w]

    def invent(self, obj: str, cfg: Config) -> str:
        word = invent_grunt()
        self.lexicon[obj][word] += cfg.success_reward * 0.6
        return word

@dataclass
class Event:
    type: str
    payload: dict

@dataclass
class Simulation:
    cfg: Config
    agents: List[Agent] = field(default_factory=list)
    tribe1: List[Agent] = field(default_factory=list)
    tribe2: List[Agent] = field(default_factory=list)
    round_idx: int = 0
    running: bool = False
    success_history: List[int] = field(default_factory=list)
    on_event: Optional[Callable[[Event], None]] = None

    def init_agents(self):
        self.tribe1 = [Agent(f"T1_A{i}", self.cfg.objects, self.cfg.agent_variance) for i in range(self.cfg.num_agents_per_tribe)]
        self.tribe2 = [Agent(f"T2_A{i}", self.cfg.objects, self.cfg.agent_variance) for i in range(self.cfg.num_agents_per_tribe)]
        self.agents = self.tribe1 + self.tribe2

    def speak(self, speaker: Agent, listener: Agent, obj: str) -> Tuple[bool, str]:
        word = speaker.choose_name_to_speak(obj)
        if word is None or random.random() < self.cfg.invention_prob:
            word = speaker.invent(obj, self.cfg)
        if random.random() < self.cfg.symbolic_recursion_prob and speaker.known_names(obj):
            other_word = random.choice(list(speaker.known_names(obj).keys()))
            word = word + "-" + other_word
        understands = listener.known_names(obj).get(word, 0.0) > 0.2
        speaker.hear(obj, word, success=understands, cfg=self.cfg)
        listener.hear(obj, word, success=understands, cfg=self.cfg)
        return understands, word

    def consensus_score(self) -> Dict[str, float]:
        scores = {}
        for obj in self.cfg.objects:
            modal = Counter()
            for a in self.agents:
                names = a.known_names(obj)
                if names:
                    top = max(names.items(), key=lambda x: x[1])[0]
                    modal[top] += 1
            scores[obj] = (modal.most_common(1)[0][1] / len(self.agents)) if modal else 0.0
        return scores

    def final_snapshot(self) -> Dict[str, List[Tuple[str, int]]]:
        final = {}
        for obj in self.cfg.objects:
            tally = Counter()
            for a in self.agents:
                names = a.known_names(obj)
                if names:
                    top = max(names.items(), key=lambda x: x[1])[0]
                    tally[top] += 1
            final[obj] = tally.most_common(3)
        return final

    async def run(self):
        self.running = True
        while self.running:
            self.round_idx += 1
            obj = random.choice(self.cfg.objects)

            if self.round_idx < self.cfg.tribe_contact_round:
                tribe = self.tribe1 if random.random() < 0.5 else self.tribe2
                speaker, listener = random.sample(tribe, 2)
            else:
                speaker, listener = random.sample(self.agents, 2)

            success, word = self.speak(speaker, listener, obj)
            self.success_history.append(1 if success else 0)

            if self.on_event:
                self.on_event(Event("interaction", {
                    "round": self.round_idx,
                    "obj": obj,
                    "speaker": speaker.name,
                    "listener": listener.name,
                    "word": word,
                    "success": success
                }))

            if self.round_idx % self.cfg.meaning_drift_interval == 0:
                o1, o2 = random.sample(self.cfg.objects, 2)
                for a in self.agents:
                    a.lexicon[o1], a.lexicon[o2] = a.lexicon[o2], a.lexicon[o1]
                if self.on_event:
                    self.on_event(Event("meaning_drift", {"round": self.round_idx, "swap": [o1, o2]}))

            if self.round_idx % self.cfg.rounds_between_report == 0:
                scores = self.consensus_score()
                avg_success = sum(self.success_history[-self.cfg.rounds_between_report:]) / self.cfg.rounds_between_report
                if self.on_event:
                    self.on_event(Event("report", {"round": self.round_idx, "avg_success": avg_success, "scores": scores}))

            await asyncio.sleep(0.01)

    def stop(self):
        self.running = False

# ---------------- Persistence ---------------- #

DDL = """
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""

class Store:
    def __init__(self, path="events.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(DDL)
        self.conn.commit()

    def log(self, type: str, payload_json: str):
        self.conn.execute(
            "INSERT INTO events(ts, type, payload) VALUES(?, ?, ?)",
            (datetime.utcnow().isoformat(), type, payload_json)
        )
        self.conn.commit()

# ---------------- FastAPI Server ---------------- #

OBJECTS = ["river", "dog", "frog", "finger", "fire", "stone", "tree", "sun", "moon", "star"]

cfg = Config(objects=OBJECTS)
sim = Simulation(cfg=cfg)
sim.init_agents()
store = Store()

app = FastAPI(title="Live Naming Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws

# ---------------- WebSocket Hub ---------------- #

class Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def remove(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        data = json.dumps(msg)
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except WebSocketDisconnect:
                dead.append(ws)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.remove(d)

hub = Hub()

# Wire simulation events to broadcast + persist
def on_event(ev: Event):
    payload_json = json.dumps(ev.payload)
    store.log(ev.type, payload_json)
    asyncio.create_task(hub.broadcast({"type": ev.type, "payload": ev.payload}))

sim.on_event = on_event

# ---------------- FastAPI Lifecycle ---------------- #

@app.on_event("startup")
async def startup():
    asyncio.create_task(sim.run())

@app.on_event("shutdown")
async def shutdown():
    sim.stop()

# ---------------- REST Endpoints ---------------- #

@app.get("/status")
def status():
    return {
        "running": sim.running,
        "round": sim.round_idx,
        "agents": [a.name for a in sim.agents],
        "objects": sim.cfg.objects
    }

@app.post("/stop")
def stop():
    sim.stop()
    return {"ok": True}

@app.post("/start")
async def start():
    if not sim.running:
        asyncio.create_task(sim.run())
    return {"ok": True}

@app.get("/snapshot")
def snapshot():
    return sim.final_snapshot()

@app.get("/consensus")
def consensus():
    return sim.consensus_score()

# Human-in-the-loop: let a user speak as an agent
@app.post("/speak")
def human_speak(agent: str, obj: str, word: str):
    found = next((a for a in sim.agents if a.name == agent), None)
    if not found:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    listeners = [a for a in sim.agents if a.name != agent]
    listener = random.choice(listeners)
    understands = listener.known_names(obj).get(word, 0.0) > 0.2
    found.hear(obj, word, success=True, cfg=sim.cfg)
    listener.hear(obj, word, success=understands, cfg=sim.cfg)
    on_event(Event("human_interaction", {
        "round": sim.round_idx,
        "obj": obj,
        "speaker": agent,
        "listener": listener.name,
        "word": word,
        "success": understands
    }))
    return {"success": understands}

# ---------------- WebSocket Endpoint ---------------- #

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                cmd = json.loads(msg)
                if cmd.get("type") == "set":
                    key, val = cmd["key"], cmd["value"]
                    if hasattr(sim.cfg, key):
                        setattr(sim.cfg, key, val)
                        await ws.send_text(json.dumps({"type":"ack","key":key,"value":val}))
            except Exception:
                pass
    except WebSocketDisconnect:
        hub.remove(ws)

# ---------------- Entrypoint ---------------- #

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

