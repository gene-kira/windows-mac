#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golem Mode v4 (freeport + robust node id): Autonomous + Replicated + Durable + Tkinter GUI
- Autonomy: observe/contain automatically, quorum-gated purge
- Replication: signed HTTP gossip, vector clocks, ethics-aware tie-breaks
- Durability: SQLite-backed ledger, events, policies, threats
- GUI: Tkinter dashboard with actions
- Node ID: optional; auto-generated if missing
- Port: supports freeport via --port 0 (binds to any available port)
"""

import os
import sys
import json
import hmac
import base64
import hashlib
import random
import threading
import argparse
import sqlite3
import time
import socket
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from http.server import HTTPServer, BaseHTTPRequestHandler

# Tkinter (stdlib)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    print("Tkinter not available. Install tkinter (e.g., sudo apt-get install python3-tk).")
    sys.exit(1)

# -----------------------------
# Utilities and cryptographic helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256b(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256(data: Any) -> str:
    return sha256b(json.dumps(data, sort_keys=True).encode("utf-8"))

def short_rand(n: int = 6) -> str:
    return base64.urlsafe_b64encode(os.urandom(9)).decode("ascii").rstrip("=")[:n]

def auto_node_id() -> str:
    host = socket.gethostname()
    return f"{host}-{short_rand(6)}"

def random_id(prefix: str = "id") -> str:
    return f"{prefix}_{base64.urlsafe_b64encode(os.urandom(9)).decode('ascii').rstrip('=')}"

def sign(secret: bytes, payload: Dict[str, Any]) -> str:
    return hmac.new(secret, json.dumps(payload, sort_keys=True).encode("utf-8"), hashlib.sha256).hexdigest()

def verify(secret: bytes, payload: Dict[str, Any], sig: str) -> bool:
    return hmac.compare_digest(sign(secret, payload), sig)

def jitter(value: float, pct: float = 0.1) -> float:
    return value * (1.0 + random.uniform(-pct, pct))

# -----------------------------
# Data schemas
# -----------------------------

@dataclass
class Threat:
    id: str
    node_id: str
    indicators: List[str]
    risk: float
    resurrection_score: float
    narrative: str
    glyph: str
    status: str  # observed | contained | purged | released
    vclock: Dict[str, int]

@dataclass
class Event:
    id: str
    time: str
    node_id: str
    type: str
    payload: Dict[str, Any]
    payload_hash: str
    signature: str
    glyph: str
    parent_event_id: Optional[str] = None

@dataclass
class Policy:
    id: str
    scope: str  # global | group | node
    precedence: int
    actions_allowed: List[str]
    ethics_rules: List[str]
    version: int
    signer_id: str
    vclock: Dict[str, int]

@dataclass
class Persona:
    node_id: str
    skin_profile: str
    blend_score: float
    decoy_channels: List[str]
    constraints: List[str]

@dataclass
class NodeMeta:
    id: str
    role: str
    trust_score: float
    sync_state: str  # fresh | stale | isolated
    last_glyph: str
    heartbeat: float
    resource_usage: float
    time_drift_ms: int

# -----------------------------
# Glyph engine
# -----------------------------

class GlyphEngine:
    GLYPHS = ["Truth", "Vigilance", "Resurrection", "Containment", "Release", "Harmony", "Ash"]

    def choose(self, context: Dict[str, Any]) -> str:
        if context.get("type") in ("detect", "signal"): return "Vigilance"
        if context.get("action") == "observe": return "Truth"
        if context.get("action") == "contain": return "Containment"
        if context.get("action") == "purge": return "Ash"
        if context.get("action") == "release": return "Release"
        if context.get("resurrection", 0) > 0.6: return "Resurrection"
        return random.choice(self.GLYPHS)

    def narrate(self, subject: str, action: str, detail: str) -> str:
        return f"{subject} enacts {action}: {detail}."

# -----------------------------
# Durable storage (SQLite)
# -----------------------------

class DurableStore:
    def __init__(self, path: str):
        self.path = path
        self._ensure_db()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS ledger(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_hash TEXT NOT NULL,
            time TEXT NOT NULL,
            record_json TEXT NOT NULL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY,
            time TEXT,
            node_id TEXT,
            type TEXT,
            payload_json TEXT,
            payload_hash TEXT,
            signature TEXT,
            glyph TEXT,
            parent_event_id TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS policies(
            id TEXT PRIMARY KEY,
            scope TEXT,
            precedence INTEGER,
            actions_allowed_json TEXT,
            ethics_rules_json TEXT,
            version INTEGER,
            signer_id TEXT,
            vclock_json TEXT,
            updated_at TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS threats(
            id TEXT PRIMARY KEY,
            node_id TEXT,
            indicators_json TEXT,
            risk REAL,
            resurrection_score REAL,
            narrative TEXT,
            glyph TEXT,
            status TEXT,
            vclock_json TEXT,
            updated_at TEXT
        )""")
        conn.commit(); conn.close()

    def append_ledger(self, record: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT chain_hash FROM ledger ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        prev_hash = row["chain_hash"] if row else ""
        payload = {"prev": prev_hash, "rec": record}
        chain_hash = sha256(payload)
        stamped = {"chain_hash": chain_hash, "time": now_iso(), "record_json": json.dumps(record, sort_keys=True)}
        cur.execute("INSERT INTO ledger(chain_hash,time,record_json) VALUES(?,?,?)",
                    (stamped["chain_hash"], stamped["time"], stamped["record_json"]))
        conn.commit(); conn.close()
        return stamped

    def ledger_tip(self) -> str:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT chain_hash FROM ledger ORDER BY id DESC LIMIT 1")
        row = cur.fetchone(); conn.close()
        return row["chain_hash"] if row else ""

    def ledger_len(self) -> int:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM ledger")
        row = cur.fetchone(); conn.close()
        return int(row["c"])

    # Events
    def store_event(self, ev: Event):
        conn = self._connect(); cur = conn.cursor()
        cur.execute("""INSERT OR IGNORE INTO events(id, time, node_id, type, payload_json, payload_hash, signature, glyph, parent_event_id)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (ev.id, ev.time, ev.node_id, ev.type, json.dumps(ev.payload, sort_keys=True),
                     ev.payload_hash, ev.signature, ev.glyph, ev.parent_event_id))
        conn.commit(); conn.close()

    def tail_events(self, count: int = 50) -> List[Event]:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT * FROM events ORDER BY time DESC LIMIT ?", (count,))
        rows = cur.fetchall(); conn.close()
        out = []
        for r in rows:
            out.append(Event(
                id=r["id"], time=r["time"], node_id=r["node_id"], type=r["type"],
                payload=json.loads(r["payload_json"]), payload_hash=r["payload_hash"], signature=r["signature"],
                glyph=r["glyph"], parent_event_id=r["parent_event_id"]
            ))
        return out

    # Policies
    def upsert_policy(self, p: Policy):
        conn = self._connect(); cur = conn.cursor()
        cur.execute("""INSERT INTO policies(id, scope, precedence, actions_allowed_json, ethics_rules_json, version, signer_id, vclock_json, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         scope=excluded.scope,
                         precedence=excluded.precedence,
                         actions_allowed_json=excluded.actions_allowed_json,
                         ethics_rules_json=excluded.ethics_rules_json,
                         version=excluded.version,
                         signer_id=excluded.signer_id,
                         vclock_json=excluded.vclock_json,
                         updated_at=excluded.updated_at
                    """, (p.id, p.scope, p.precedence,
                          json.dumps(p.actions_allowed, sort_keys=True),
                          json.dumps(p.ethics_rules, sort_keys=True),
                          p.version, p.signer_id,
                          json.dumps(p.vclock, sort_keys=True), now_iso()))
        conn.commit(); conn.close()

    def get_policies(self) -> Dict[str, Policy]:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT * FROM policies")
        rows = cur.fetchall(); conn.close()
        out = {}
        for r in rows:
            out[r["id"]] = Policy(
                id=r["id"], scope=r["scope"], precedence=r["precedence"],
                actions_allowed=json.loads(r["actions_allowed_json"]),
                ethics_rules=json.loads(r["ethics_rules_json"]),
                version=r["version"], signer_id=r["signer_id"],
                vclock=json.loads(r["vclock_json"])
            )
        return out

    # Threats
    def upsert_threat(self, t: Threat):
        conn = self._connect(); cur = conn.cursor()
        cur.execute("""INSERT INTO threats(id, node_id, indicators_json, risk, resurrection_score, narrative, glyph, status, vclock_json, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         node_id=excluded.node_id,
                         indicators_json=excluded.indicators_json,
                         risk=excluded.risk,
                         resurrection_score=excluded.resurrection_score,
                         narrative=excluded.narrative,
                         glyph=excluded.glyph,
                         status=excluded.status,
                         vclock_json=excluded.vclock_json,
                         updated_at=excluded.updated_at
                    """, (t.id, t.node_id, json.dumps(t.indicators, sort_keys=True), t.risk,
                          t.resurrection_score, t.narrative, t.glyph, t.status, json.dumps(t.vclock, sort_keys=True), now_iso()))
        conn.commit(); conn.close()

    def get_threats(self) -> List[Threat]:
        conn = self._connect(); cur = conn.cursor()
        cur.execute("SELECT * FROM threats ORDER BY updated_at DESC")
        rows = cur.fetchall(); conn.close()
        out = []
        for r in rows:
            out.append(Threat(
                id=r["id"], node_id=r["node_id"],
                indicators=json.loads(r["indicators_json"]),
                risk=r["risk"], resurrection_score=r["resurrection_score"],
                narrative=r["narrative"], glyph=r["glyph"], status=r["status"],
                vclock=json.loads(r["vclock_json"])
            ))
        return out

# -----------------------------
# Vector clock
# -----------------------------

def vclock_increment(vc: Dict[str, int], node_id: str) -> Dict[str, int]:
    vc = dict(vc or {})
    vc[node_id] = vc.get(node_id, 0) + 1
    return vc

def vclock_compare(a: Dict[str, int], b: Dict[str, int]) -> int:
    a_dom = False; b_dom = False
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        av = a.get(k, 0); bv = b.get(k, 0)
        if av < bv: b_dom = True
        if av > bv: a_dom = True
    if a_dom and not b_dom: return 1
    if b_dom and not a_dom: return -1
    return 0

# -----------------------------
# Event bus
# -----------------------------

class EventBus:
    def __init__(self, secret: bytes, store: DurableStore):
        self.secret = secret
        self.store = store
        self.subscribers: List[Callable[[Event], None]] = []
        self.lock = threading.Lock()

    def publish(self, node_id: str, event_type: str, payload: Dict[str, Any], glyph: str, parent: Optional[str] = None) -> Event:
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        ev = Event(
            id=random_id("evt"), time=now_iso(), node_id=node_id, type=event_type,
            payload=payload, payload_hash=sha256b(payload_bytes),
            signature=hmac.new(self.secret, payload_bytes, hashlib.sha256).hexdigest(),
            glyph=glyph, parent_event_id=parent
        )
        with self.lock:
            self.store.store_event(ev)
        for sub in self.subscribers:
            try: sub(ev)
            except Exception: pass
        return ev

    def tail(self, count: int = 50) -> List[Event]:
        return self.store.tail_events(count)

    def subscribe(self, handler: Callable[[Event], None]):
        self.subscribers.append(handler)

# -----------------------------
# Codex (policy registry)
# -----------------------------

class Codex:
    def __init__(self, bus: EventBus, glyphs: GlyphEngine, store: DurableStore, node_id: str):
        self.bus = bus
        self.glyphs = glyphs
        self.store = store
        self.node_id = node_id
        self.policies: Dict[str, Policy] = self.store.get_policies()
        self.lock = threading.Lock()

    def broadcast(self, policy: Policy):
        with self.lock:
            policy.vclock = vclock_increment(policy.vclock, self.node_id)
            self.policies[policy.id] = policy
            self.store.upsert_policy(policy)
        self.bus.publish("codex", "policy_broadcast", {"policy": policy.__dict__}, glyph=self.glyphs.choose({"type": "signal"}))

    def merge_snapshot(self, incoming: Dict[str, Dict[str, Any]]):
        with self.lock:
            for pid, pdata in incoming.items():
                incoming_p = Policy(**pdata)
                local_p = self.policies.get(pid)
                if not local_p:
                    self.policies[pid] = incoming_p
                    self.store.upsert_policy(incoming_p)
                else:
                    cmp = vclock_compare(local_p.vclock, incoming_p.vclock)
                    if cmp == -1:
                        self.policies[pid] = incoming_p
                        self.store.upsert_policy(incoming_p)
                    elif cmp == 0:
                        if incoming_p.version > local_p.version or \
                           (incoming_p.version == local_p.version and incoming_p.precedence >= local_p.precedence):
                            self.policies[pid] = incoming_p
                            self.store.upsert_policy(incoming_p)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return {pid: p.__dict__ for pid, p in self.policies.items()}

# -----------------------------
# Node agent
# -----------------------------

class NodeAgent:
    def __init__(self, node: NodeMeta, bus: EventBus, store: DurableStore, glyphs: GlyphEngine):
        self.node = node
        self.bus = bus
        self.store = store
        self.glyphs = glyphs
        self.threats: Dict[str, Threat] = {}
        for t in self.store.get_threats():
            self.threats[t.id] = t
        self.lock = threading.Lock()

    def detect(self) -> Optional[Threat]:
        if random.random() < 0.35:
            t_id = random_id("thr")
            risk = random.random()
            resurrection = max(0.0, min(1.0, jitter(risk, 0.3) * 0.7))
            indicators = random.sample(["proc_anomaly", "net_beacon", "file_mut", "policy_violation"], k=random.randint(1, 3))
            glyph = self.glyphs.choose({"type": "detect", "resurrection": resurrection})
            narrative = self.glyphs.narrate(self.node.id, "detect", f"signals={indicators}, risk={risk:.2f}, resurrection={resurrection:.2f}")
            thr = Threat(
                id=t_id, node_id=self.node.id, indicators=indicators,
                risk=risk, resurrection_score=resurrection,
                narrative=narrative, glyph=glyph, status="observed",
                vclock={self.node.id: 1}
            )
            with self.lock:
                self.threats[thr.id] = thr
                self.store.upsert_threat(thr)
            self.bus.publish(self.node.id, "threat_detected", {"threat": thr.__dict__}, glyph=glyph)
            self.node.last_glyph = glyph
            return thr
        return None

    def observe(self, thr_id: str):
        with self.lock:
            thr = self.threats.get(thr_id)
            if not thr: return
            thr.status = "observed"
            thr.vclock = vclock_increment(thr.vclock, self.node.id)
            self.store.upsert_threat(thr)
        self.bus.publish(self.node.id, "observe_started", {"thr_id": thr_id}, glyph="Truth")

    def contain(self, thr_id: str):
        with self.lock:
            thr = self.threats.get(thr_id)
            if not thr: return
            thr.status = "contained"
            thr.vclock = vclock_increment(thr.vclock, self.node.id)
            self.store.upsert_threat(thr)
        self.bus.publish(self.node.id, "contain_started", {"thr_id": thr_id}, glyph="Containment")

    def purge_execute(self, thr_id: str, quorum_note: str):
        with self.lock:
            thr = self.threats.get(thr_id)
            if not thr: return
            thr.status = "purged"
            thr.vclock = vclock_increment(thr.vclock, self.node.id)
            self.store.upsert_threat(thr)
        self.bus.publish(self.node.id, "purge_executed", {"thr_id": thr_id, "quorum": quorum_note}, glyph="Ash")

    def release(self, thr_id: str):
        with self.lock:
            thr = self.threats.get(thr_id)
            if not thr: return
            thr.status = "released"
            thr.vclock = vclock_increment(thr.vclock, self.node.id)
            self.store.upsert_threat(thr)
        self.bus.publish(self.node.id, "release_executed", {"thr_id": thr_id}, glyph="Release")

    def merge_threats(self, incoming: List[Dict[str, Any]]):
        with self.lock:
            for td in incoming:
                it = Threat(**td)
                lt = self.threats.get(it.id)
                if not lt:
                    self.threats[it.id] = it
                    self.store.upsert_threat(it)
                else:
                    cmp = vclock_compare(lt.vclock, it.vclock)
                    if cmp == -1:
                        self.threats[it.id] = it
                        self.store.upsert_threat(it)
                    elif cmp == 0:
                        rank = {"purged": 3, "contained": 2, "observed": 1, "released": 0}
                        if rank.get(it.status, 0) > rank.get(lt.status, 0) or it.risk > lt.risk:
                            self.threats[it.id] = it
                            self.store.upsert_threat(it)

# -----------------------------
# Autonomous controller
# -----------------------------

class AutonomousController:
    def __init__(self, agent: NodeAgent, bus: EventBus, codex: Codex):
        self.agent = agent
        self.bus = bus
        self.codex = codex
        self.auto_observe_threshold = 0.55
        self.auto_contain_threshold = 0.75
        self.auto_purge_threshold = 0.92
        self.enable_auto_purge = True

    def tick(self):
        n = self.agent.node
        n.heartbeat = round(jitter(n.heartbeat, 0.05), 2)
        n.resource_usage = round(jitter(n.resource_usage, 0.05), 2)
        if n.resource_usage > 0.85: n.sync_state = "stale"
        if random.random() < 0.25:
            self.bus.publish(n.id, "heartbeat", {"hb": n.heartbeat, "res": n.resource_usage}, glyph="Harmony")
        thr = self.agent.detect()
        if thr:
            if thr.risk >= self.auto_observe_threshold: self.agent.observe(thr.id)
            if thr.risk >= self.auto_contain_threshold: self.agent.contain(thr.id)
            if self.enable_auto_purge and thr.risk >= self.auto_purge_threshold:
                self.bus.publish(n.id, "purge_requested",
                                 {"thr_id": thr.id, "node_id": n.id, "status": thr.status, "risk": thr.risk},
                                 glyph="Ash")

# -----------------------------
# Replication: secure gossip over HTTP
# -----------------------------

class PeerDirectory:
    def __init__(self, self_url: str):
        self.self_url = self_url
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def set_peers(self, urls: List[str]):
        with self.lock:
            for url in urls:
                if url and url != self.self_url:
                    self.peers[url] = {"reachable": False, "last_ok": None}

    def mark(self, url: str, ok: bool):
        with self.lock:
            if url in self.peers:
                self.peers[url]["reachable"] = ok
                self.peers[url]["last_ok"] = now_iso()

    def list(self) -> List[str]:
        with self.lock:
            return list(self.peers.keys())

class GossipReplicator:
    def __init__(self, app, secret: bytes, peers: PeerDirectory):
        self.app = app
        self.secret = secret
        self.peers = peers
        self.running = False

    def _post(self, url: str, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = {"payload": payload, "sig": sign(self.secret, payload)}
        req = Request(url + path, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=3.5) as resp:
                data = resp.read()
                self.peers.mark(url, ok=True)
                return json.loads(data.decode("utf-8"))
        except Exception:
            self.peers.mark(url, ok=False)
            return None

    def gossip_tick(self):
        policies = self.app.codex.snapshot()
        events = [e.__dict__ for e in self.app.bus.tail(25)]
        threats = [t.__dict__ for t in self.app.agent.threats.values()]
        payload = {
            "node": self.app.node.id,
            "policies": policies,
            "events": events,
            "threats": threats,
            "ledger_tip": self.app.store.ledger_tip(),
            "time": now_iso()
        }
        for peer in self.peers.list():
            self._post(peer, "/gossip", payload)

    def start(self, interval_sec: float = 2.0):
        if self.running: return
        self.running = True
        def loop():
            while self.running:
                self.gossip_tick()
                time.sleep(interval_sec)
        threading.Thread(target=loop, daemon=True).start()

# -----------------------------
# HTTP server: gossip and quorum endpoints
# -----------------------------

class GolemHTTPHandler(BaseHTTPRequestHandler):
    app = None
    secret = None

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length > 0 else b"{}"
        try: return json.loads(data.decode("utf-8"))
        except Exception: return {}

    def _write_json(self, obj: Dict[str, Any], code: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self._read_json()
        payload = body.get("payload", {})
        sig = body.get("sig", "")
        if not verify(GolemHTTPHandler.secret, payload, sig):
            return self._write_json({"error": "invalid signature"}, 401)

        if self.path == "/gossip":
            self.app.merge_gossip(payload)
            return self._write_json({"ok": True})

        if self.path == "/quorum/purge_vote":
            status = payload.get("status", "observed")
            risk = float(payload.get("risk", 0.0))
            vote = bool(risk >= self.app.auto.auto_purge_threshold and status in ("observed", "contained"))
            self.app.bus.publish(self.app.node.id, "purge_vote_cast",
                                 {"thr_id": payload.get("thr_id"), "source_node": payload.get("node_id"), "vote": vote}, glyph="Truth")
            return self._write_json({"vote": vote})

        return self._write_json({"error": "unknown path"}, 404)

    def log_message(self, fmt, *args):
        pass

# -----------------------------
# App container
# -----------------------------

class GolemApp:
    def __init__(self, node_id: str, port: int, peers: List[str], db_path: str):
        self.secret = os.urandom(32)
        self.store = DurableStore(db_path)
        self.bus = EventBus(secret=self.secret, store=self.store)
        self.glyphs = GlyphEngine()
        self.node = NodeMeta(
            id=node_id, role="guardian", trust_score=round(random.uniform(0.7, 0.95), 2),
            sync_state="fresh", last_glyph="Harmony", heartbeat=round(jitter(1.0, 0.2), 2),
            resource_usage=round(random.uniform(0.15, 0.75), 2), time_drift_ms=random.randint(-25, 25)
        )
        self.codex = Codex(bus=self.bus, glyphs=self.glyphs, store=self.store, node_id=node_id)
        self.agent = NodeAgent(node=self.node, bus=self.bus, store=self.store, glyphs=self.glyphs)
        self.auto = AutonomousController(agent=self.agent, bus=self.bus, codex=self.codex)
        self.persona = Persona(node_id=node_id, skin_profile=random.choice(["clay", "basalt", "terracotta"]),
                               blend_score=round(random.uniform(0.5, 0.9), 2),
                               decoy_channels=random.sample(["honeypot", "canary", "echo"], k=random.randint(1, 3)),
                               constraints=["no-protected-impersonation", "no-coercion"])
        self.self_url = f"http://localhost:{port}"  # updated after server binds
        self.peers = PeerDirectory(self.self_url)
        self.peers.set_peers(peers)
        self.gossip = GossipReplicator(self, self.secret, self.peers)
        self.bus.subscribe(self._on_event)

        pol = Policy(
            id=random_id("pol"), scope="global", precedence=1,
            actions_allowed=["observe", "contain", "release", "purge"],
            ethics_rules=["quorum-purge", "containment-first", "minimal-capture"],
            version=1, signer_id="admin_root", vclock={self.node.id: 1}
        )
        self.codex.broadcast(pol)

    def _on_event(self, ev: Event):
        if ev.type == "purge_requested":
            thr_id = ev.payload.get("thr_id")
            node_id = ev.payload.get("node_id")
            status = ev.payload.get("status", "observed")
            risk = float(ev.payload.get("risk", 0.0))
            votes = 1; total = 1
            for peer in self.peers.list():
                resp = self._post_peer(peer, "/quorum/purge_vote",
                                       {"thr_id": thr_id, "node_id": node_id, "status": status, "risk": risk})
                total += 1
                if resp and resp.get("vote"): votes += 1
            if votes >= max(2, (total // 2) + 1):
                self.agent.purge_execute(thr_id, quorum_note=f"votes={votes}/{total}")
            else:
                self.bus.publish(self.node.id, "purge_quorum_failed",
                                 {"thr_id": thr_id, "votes": votes, "total": total}, glyph="Vigilance")

    def _post_peer(self, url: str, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = {"payload": payload, "sig": sign(self.secret, payload)}
        req = Request(url + path, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=3.5) as resp:
                data = resp.read()
                self.peers.mark(url, ok=True)
                return json.loads(data.decode("utf-8"))
        except Exception:
            self.peers.mark(url, ok=False)
            return None

    def merge_gossip(self, payload: Dict[str, Any]):
        self.codex.merge_snapshot(payload.get("policies", {}))
        for ev in payload.get("events", []):
            try:
                eobj = Event(**ev)
                self.store.store_event(eobj)
            except Exception:
                pass
        self.agent.merge_threats(payload.get("threats", []))
        peer_tip = payload.get("ledger_tip", "")
        if peer_tip and peer_tip != self.store.ledger_tip():
            self.bus.publish(self.node.id, "ledger_divergence",
                             {"peer_tip": peer_tip, "self_tip": self.store.ledger_tip()}, glyph="Vigilance")

    def start_autonomy(self):
        def loop():
            while True:
                self.auto.tick()
                time.sleep(1.0)
        threading.Thread(target=loop, daemon=True).start()

    def start_gossip(self):
        self.gossip.start(interval_sec=2.0)

    def start_http(self, port: int) -> int:
        GolemHTTPHandler.app = self
        GolemHTTPHandler.secret = self.secret
        server = HTTPServer(("0.0.0.0", port), GolemHTTPHandler)
        bound_port = server.server_address[1]  # actual port (supports freeport when port=0)
        self.self_url = f"http://localhost:{bound_port}"
        self.peers.self_url = self.self_url
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.bus.publish(self.node.id, "server_started", {"port": bound_port}, glyph="Truth")
        return bound_port

    def render_snapshot(self) -> Dict[str, Any]:
        threats = [{
            "id": t.id, "node_id": t.node_id, "risk": round(t.risk, 2),
            "resurrection": round(t.resurrection_score, 2),
            "status": t.status, "glyph": t.glyph
        } for t in self.agent.threats.values()]
        events = [{"time": e.time, "node_id": e.node_id, "type": e.type, "glyph": e.glyph} for e in self.bus.tail(12)]
        peers = [{"url": url} for url in self.peers.list()]
        return {
            "node": self.node.id, "time": now_iso(),
            "heartbeat": round(self.node.heartbeat, 2), "resource": round(self.node.resource_usage, 2),
            "sync": self.node.sync_state, "ledger_len": self.store.ledger_len(),
            "ledger_tip": self.store.ledger_tip() or "none",
            "self_url": self.self_url,
            "threats": threats, "events": events, "peers": peers,
            "policies": [{**p.__dict__, "vclock": p.vclock} for p in self.codex.policies.values()]
        }

# -----------------------------
# Tkinter GUI
# -----------------------------

class GolemGUI(tk.Tk):
    def __init__(self, app: GolemApp):
        super().__init__()
        self.title(f"Golem Mode v4 • {app.node.id}")
        self.geometry("1000x700")
        self.app = app

        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=8, pady=8)
        self.status_lbl = ttk.Label(status_frame, text="Status", font=("Segoe UI", 12, "bold"))
        self.status_lbl.pack(side="left")

        peers_frame = ttk.LabelFrame(self, text="Peers")
        peers_frame.pack(side="left", fill="y", padx=8, pady=8)
        self.peers_list = tk.Listbox(peers_frame, height=12, width=32)
        self.peers_list.pack(fill="both", expand=True)

        threats_frame = ttk.LabelFrame(self, text="Threats")
        threats_frame.pack(side="top", fill="both", padx=8, pady=8)
        cols = ("id", "node", "risk", "resurrect", "status", "glyph")
        self.threats = ttk.Treeview(threats_frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (240, 140, 90, 120, 140, 100)):
            self.threats.heading(c, text=c.capitalize())
            self.threats.column(c, width=w, anchor="w")
        self.threats.pack(fill="both", expand=True)

        actions_frame = ttk.Frame(threats_frame)
        actions_frame.pack(fill="x", pady=6)
        ttk.Button(actions_frame, text="Observe", command=self.observe_selected).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Contain", command=self.contain_selected).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Release", command=self.release_selected).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Purge (quorum)", command=self.purge_selected).pack(side="left", padx=4)

        events_frame = ttk.LabelFrame(self, text="Events (tail)")
        events_frame.pack(side="bottom", fill="both", padx=8, pady=8)
        ecols = ("time", "node", "type", "glyph")
        self.events = ttk.Treeview(events_frame, columns=ecols, show="headings", height=10)
        for c, w in zip(ecols, (220, 140, 260, 140)):
            self.events.heading(c, text=c.capitalize())
            self.events.column(c, width=w, anchor="w")
        self.events.pack(fill="both", expand=True)

        self.after(500, self.refresh)

    def refresh(self):
        snap = self.app.render_snapshot()
        self.status_lbl.configure(
            text=f"Node {snap['node']} • {snap['time']} • HB {snap['heartbeat']} • Res {snap['resource']} • Sync {snap['sync']} • Ledger {snap['ledger_len']} • URL {snap['self_url']}"
        )
        self.peers_list.delete(0, tk.END)
        for p in snap["peers"]:
            self.peers_list.insert(tk.END, p["url"])

        for i in self.threats.get_children():
            self.threats.delete(i)
        for t in snap["threats"]:
            halo = "⚠️" if t["resurrection"] > 0.6 else "•"
            self.threats.insert("", "end", values=(t["id"], t["node_id"], t["risk"], f"{t['resurrection']} {halo}", t["status"], t["glyph"]))

        for i in self.events.get_children():
            self.events.delete(i)
        for e in snap["events"]:
            self.events.insert("", "end", values=(e["time"], e["node_id"], e["type"], e["glyph"]))

        self.after(1000, self.refresh)

    def _selected_thr(self) -> Optional[str]:
        sel = self.threats.selection()
        if not sel: return None
        vals = self.threats.item(sel[0], "values")
        return vals[0] if vals else None

    def observe_selected(self):
        tid = self._selected_thr()
        if tid: self.app.agent.observe(tid)

    def contain_selected(self):
        tid = self._selected_thr()
        if tid: self.app.agent.contain(tid)

    def release_selected(self):
        tid = self._selected_thr()
        if tid: self.app.agent.release(tid)

    def purge_selected(self):
        tid = self._selected_thr()
        if not tid: return
        t = self.app.agent.threats.get(tid)
        risk = t.risk if t else 0.95
        status = t.status if t else "observed"
        self.app.bus.publish(self.app.node.id, "purge_requested",
                             {"thr_id": tid, "node_id": self.app.node.id, "status": status, "risk": risk},
                             glyph="Ash")
        messagebox.showinfo("Quorum", "Purge requested. Awaiting peer votes...")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Golem Mode v4 (freeport + robust node id)")
    p.add_argument("--node-id", required=False, help="Unique node id (auto-generated if omitted)")
    p.add_argument("--port", type=int, default=0, help="HTTP server port (0 selects any free port)")
    p.add_argument("--peers", type=str, default="", help="Comma-separated peer URLs (e.g., http://localhost:8082,http://localhost:8083)")
    p.add_argument("--db", type=str, default="", help="SQLite file path (defaults to golem_<node>.db)")
    return p.parse_args()

def main():
    args = parse_args()
    node_id = args.node_id or auto_node_id()
    # Temporary DB path depends on node id
    db_path = args.db or f"golem_{node_id}.db"
    peers = [u.strip() for u in args.peers.split(",") if u.strip()]

    app = GolemApp(node_id=node_id, port=args.port, peers=peers, db_path=db_path)
    bound_port = app.start_http(args.port)  # supports freeport; returns actual port
    # Update peers that might have used placeholder ports
    app.start_autonomy()
    app.start_gossip()

    print(f"Golem node '{node_id}' listening on {app.self_url}, DB={db_path}")
    gui = GolemGUI(app)
    gui.mainloop()

if __name__ == "__main__":
    main()

