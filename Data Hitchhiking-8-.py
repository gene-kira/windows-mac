#!/usr/bin/env python3
"""
CBN Lab Swarm v2 (Office-Style Cockpit + Raft + Visuals)

Features:
- Async TCP networking between nodes (real sockets)
- RSA signatures + AES-GCM payload encryption (envelope)
- Replay protection (nonce + timestamp) with persistent store
- Registry "swarm": primary + follower with Raft-like leader election + log replication
- Lineage-aware policy enforcement (internal_only)
- Resurrection detection (revoked capsule seen again)
- Predictive threat scoring
- Swarm sync telemetry
- Multi-gate routing with trust decay and bump on success
- Tkinter "Office-style" ribbon cockpit GUI:
  - Tabs: Overview, Timeline, Telemetry, Threats, Lineage, Swarm, Diagnostics
  - Telemetry server bind diagnostics
  - Telemetry event logging
  - Exception logging for telemetry handling and GUI refresh
  - Node heartbeat panel + silence detection
  - Last telemetry message panel
  - Raw telemetry dump panel
  - Port conflict detection
  - Cockpit startup diagnostics
  - Relay/gate/origin activity indicators
  - Lineage tree visualization
  - Capsule drill-down inspector
  - Threat heatmap panel
  - Swarm map visualization (nodes + edges)
- Launcher to spawn all roles as separate OS processes
"""

import sys
import asyncio
import json
import sqlite3
import hashlib
import uuid
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import threading
import queue
import traceback
import random

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("[FATAL] Missing dependency: cryptography")
    print("Install with: pip install cryptography")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("[FATAL] Tkinter not available in this Python environment.")
    sys.exit(1)

# -------------------------
# Config
# -------------------------
HOST = "127.0.0.1"

PORT_KEYHUB            = 9101
PORT_REGISTRY_PRIMARY  = 9102
PORT_REGISTRY_FOLLOWER = 9107
PORT_ORIGIN            = 9103
PORT_RELAY             = 9104
PORT_GATE              = 9105
PORT_TELEMETRY         = 9106
PORT_CONTROL           = 9108  # reserved / future

DB_REGISTRY_PRIMARY  = "registry_primary.db"
DB_REGISTRY_FOLLOWER = "registry_follower.db"
DB_KEYHUB            = "keyhub.db"
DB_GATE              = "gate_state.db"

REPLAY_WINDOW_SECONDS = 60
TLS_ENABLED = False

NODE_SILENCE_SECONDS = 15

RAFT_HEARTBEAT_INTERVAL = 1.0
RAFT_ELECTION_TIMEOUT_MIN = 3.0
RAFT_ELECTION_TIMEOUT_MAX = 5.0

# -------------------------
# Utility
# -------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"

def log_diag(msg: str):
    ts = now_iso()
    print(f"[DIAG {ts}] {msg}", flush=True)

async def send_json(addr, msg, retries=3, delay=0.1):
    for attempt in range(retries):
        try:
            reader, writer = await asyncio.open_connection(*addr)
            writer.write(json.dumps(msg).encode("utf-8") + b"\n")
            await writer.drain()
            resp = await reader.readline()
            writer.close()
            await writer.wait_closed()
            if resp:
                return json.loads(resp.decode("utf-8"))
            return {}
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                log_diag(f"send_json failed to {addr}: {e}")
                return {}

# -------------------------
# Crypto helpers
# -------------------------
def generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key

def sign_bytes(private_key, data: bytes) -> bytes:
    return private_key.sign(
        data,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

def verify_signature(public_key, signature: bytes, data: bytes) -> bool:
    try:
        public_key.verify(
            signature,
            data,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False

def public_key_fingerprint(public_key) -> str:
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return hashlib.sha256(pem).hexdigest()

def aes_encrypt(plaintext: bytes) -> Dict[str, str]:
    key = AESGCM.generate_key(bit_length=128)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return {
        "key_hex": key.hex(),
        "nonce_hex": nonce.hex(),
        "ciphertext_hex": ct.hex(),
    }

def aes_decrypt(key_hex: str, nonce_hex: str, ciphertext_hex: str) -> bytes:
    key = bytes.fromhex(key_hex)
    nonce = bytes.fromhex(nonce_hex)
    ct = bytes.fromhex(ciphertext_hex)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None)

# -------------------------
# Telemetry (async queue)
# -------------------------
TELEMETRY_QUEUE: "queue.Queue[dict]" = queue.Queue()

async def emit_telemetry(event_type: str, payload: dict):
    msg = {
        "type": "telemetry",
        "event_type": event_type,
        "timestamp": now_iso(),
        "payload": payload,
    }
    TELEMETRY_QUEUE.put(msg)
    try:
        await send_json((HOST, PORT_TELEMETRY), msg)
    except Exception as e:
        log_diag(f"emit_telemetry send failed: {e}")

# -------------------------
# Capsule
# -------------------------
class Capsule:
    def __init__(self, origin_id, origin_endpoint, capsule_class, org_id,
                 payload: dict, origin_private_key, origin_public_key,
                 parents: list = None, derivation_note: str = "",
                 internal_only: bool = False):
        self.origin_id = origin_id
        self.origin_endpoint = origin_endpoint
        self.capsule_class = capsule_class
        self.org_id = org_id
        self.payload = payload
        self.parents = parents or []
        self.derivation_note = derivation_note
        self.internal_only = internal_only
        self.version = 1
        self.created_at = now_iso()
        self.capsule_id = gen_id("capsule")

        plaintext = json.dumps(payload).encode("utf-8")
        enc = aes_encrypt(plaintext)
        self.encrypted_payload = enc["ciphertext_hex"]
        self.encrypted_key_hex = enc["key_hex"]
        self.nonce_hex = enc["nonce_hex"]

        core = self._core_dict()
        self.content_hash = hash_json(core)

        self.origin_signature = sign_bytes(
            origin_private_key,
            self.content_hash.encode("utf-8")
        ).hex()

    def _core_dict(self):
        return {
            "capsule_id": self.capsule_id,
            "origin_id": self.origin_id,
            "created_at": self.created_at,
            "capsule_class": self.capsule_class,
            "version": self.version,
            "lineage": {"parents": self.parents, "derivation_note": self.derivation_note},
            "policy_root_hash": self.origin_id,
            "encrypted_payload": self.encrypted_payload,
            "encrypted_key_hex": self.encrypted_key_hex,
        }

    def to_wire(self):
        return {
            "capsule_id": self.capsule_id,
            "origin_id": self.origin_id,
            "origin_endpoint": self.origin_endpoint,
            "capsule_class": self.capsule_class,
            "org_id": self.org_id,
            "version": self.version,
            "created_at": self.created_at,
            "lineage": {"parents": self.parents, "derivation_note": self.derivation_note},
            "policy": {"internal_only": self.internal_only},
            "policy_root_hash": self.origin_id,
            "encrypted_payload": self.encrypted_payload,
            "encrypted_key_hex": self.encrypted_key_hex,
            "nonce": str(uuid.uuid4()),
            "timestamp": now_iso(),
            "content_hash": self.content_hash,
            "origin_signature": self.origin_signature,
        }

# -------------------------
# KeyHub
# -------------------------
class KeyHub:
    def __init__(self, host=HOST, port=PORT_KEYHUB, db_path=DB_KEYHUB):
        self.host = host
        self.port = port
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS keys (
            origin_id TEXT PRIMARY KEY,
            pubkey_pem BLOB,
            fingerprint TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _store_key(self, origin_id: str, pub_pem: bytes, fingerprint: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO keys (origin_id, pubkey_pem, fingerprint) VALUES (?, ?, ?)",
            (origin_id, pub_pem, fingerprint)
        )
        conn.commit()
        conn.close()

    def _get_key(self, origin_id: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT pubkey_pem, fingerprint FROM keys WHERE origin_id=?", (origin_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return row[0], row[1]
        return None, None

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            kind = msg.get("type")
            if kind == "publish_key":
                origin_id = msg["origin_id"]
                pub_pem = bytes.fromhex(msg["pubkey_pem_hex"])
                fingerprint = msg["fingerprint"]
                self._store_key(origin_id, pub_pem, fingerprint)
                await emit_telemetry("keyhub_publish", {"origin_id": origin_id, "fingerprint": fingerprint})
                writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
            elif kind == "get_key":
                origin_id = msg["origin_id"]
                pub_pem, fingerprint = self._get_key(origin_id)
                if pub_pem is None:
                    writer.write(json.dumps({"type": "error", "reason": "not_found"}).encode("utf-8") + b"\n")
                else:
                    writer.write(json.dumps({
                        "type": "key",
                        "origin_id": origin_id,
                        "pubkey_pem_hex": pub_pem.hex(),
                        "fingerprint": fingerprint,
                    }).encode("utf-8") + b"\n")
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("KeyHub starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"KeyHub bound on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

# -------------------------
# Registry Base + Swarm helpers
# -------------------------
class RegistryBase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS capsules (
            capsule_id TEXT PRIMARY KEY,
            origin_id TEXT,
            capsule_class TEXT,
            status TEXT,
            policy_root_hash TEXT,
            created_at TEXT,
            lineage_json TEXT,
            internal_only INTEGER,
            last_update TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _register_capsule(self, capsule_wire: dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        internal_only = 1 if capsule_wire["policy"].get("internal_only") else 0
        c.execute("""
        INSERT OR REPLACE INTO capsules
        (capsule_id, origin_id, capsule_class, status, policy_root_hash, created_at, lineage_json, internal_only, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            capsule_wire["capsule_id"],
            capsule_wire["origin_id"],
            capsule_wire["capsule_class"],
            "active",
            capsule_wire["policy_root_hash"],
            capsule_wire["created_at"],
            json.dumps(capsule_wire["lineage"]),
            internal_only,
            now_iso(),
        ))
        conn.commit()
        conn.close()

    def _revoke_capsule(self, capsule_id: str, reason: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        UPDATE capsules
        SET status = ?, last_update = ?
        WHERE capsule_id = ?
        """, (f"revoked:{reason}", now_iso(), capsule_id))
        conn.commit()
        conn.close()

    def _get_capsule_status(self, capsule_id: str) -> str:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT status FROM capsules WHERE capsule_id=?", (capsule_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else ""

# -------------------------
# Raft Log Entry
# -------------------------
class RaftLogEntry:
    def __init__(self, term: int, command: dict):
        self.term = term
        self.command = command

# -------------------------
# Registry Primary (Raft leader)
# -------------------------
class RegistryPrimary(RegistryBase):
    def __init__(self, host=HOST, port=PORT_REGISTRY_PRIMARY,
                 db_path=DB_REGISTRY_PRIMARY,
                 follower_addr=(HOST, PORT_REGISTRY_FOLLOWER)):
        super().__init__(db_path)
        self.host = host
        self.port = port
        self.follower_addr = follower_addr

        self.current_term = 1
        self.log: List[RaftLogEntry] = []
        self.commit_index = -1
        self.last_applied = -1
        self.next_index = {follower_addr: 0}
        self.match_index = {follower_addr: -1}
        self.role = "leader"  # fixed leader for this lab

    def _append_log(self, command: dict):
        entry = RaftLogEntry(self.current_term, command)
        self.log.append(entry)
        return len(self.log) - 1

    async def _replicate_to_follower(self):
        follower = self.follower_addr
        ni = self.next_index[follower]
        prev_log_index = ni - 1
        prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
        entries = [ {"term": e.term, "command": e.command} for e in self.log[ni:] ]
        msg = {
            "type": "raft_append_entries",
            "term": self.current_term,
            "leader_id": "registry_primary",
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": self.commit_index,
        }
        resp = await send_json(follower, msg)
        if resp.get("success"):
            self.next_index[follower] = len(self.log)
            self.match_index[follower] = len(self.log) - 1
            # commit all replicated entries
            self.commit_index = len(self.log) - 1
            await self._apply_committed()
        else:
            self.next_index[follower] = max(0, self.next_index[follower] - 1)

    async def _apply_committed(self):
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            cmd = entry.command
            if cmd["type"] == "register":
                self._register_capsule(cmd["capsule"])
                await emit_telemetry("registry_register", {
                    "capsule_id": cmd["capsule"]["capsule_id"],
                    "lineage": cmd["capsule"]["lineage"],
                    "internal_only": cmd["capsule"]["policy"].get("internal_only", False),
                })
            elif cmd["type"] == "revoke":
                self._revoke_capsule(cmd["capsule_id"], cmd["reason"])
                await emit_telemetry("registry_revoke", {
                    "capsule_id": cmd["capsule_id"],
                    "reason": cmd["reason"],
                })

    async def _handle_client_command(self, msg: dict, writer):
        kind = msg.get("type")
        if kind == "register":
            idx = self._append_log(msg)
            await self._replicate_to_follower()
            writer.write(json.dumps({"type": "ack", "status": "ok", "log_index": idx}).encode("utf-8") + b"\n")
        elif kind == "revoke":
            idx = self._append_log(msg)
            await self._replicate_to_follower()
            writer.write(json.dumps({"type": "ack", "status": "ok", "log_index": idx}).encode("utf-8") + b"\n")
        elif kind == "control":
            cmd = msg.get("command")
            if cmd == "revoke_capsule":
                cid = msg["capsule_id"]
                reason = msg.get("reason", "operator_revoke")
                idx = self._append_log({"type": "revoke", "capsule_id": cid, "reason": reason})
                await self._replicate_to_follower()
                writer.write(json.dumps({"type": "ack", "status": "ok", "log_index": idx}).encode("utf-8") + b"\n")
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_control"}).encode("utf-8") + b"\n")
        else:
            writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
        await writer.drain()

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(RAFT_HEARTBEAT_INTERVAL)
            msg = {
                "type": "raft_append_entries",
                "term": self.current_term,
                "leader_id": "registry_primary",
                "prev_log_index": len(self.log) - 1,
                "prev_log_term": self.log[-1].term if self.log else 0,
                "entries": [],
                "leader_commit": self.commit_index,
            }
            await send_json(self.follower_addr, msg)
            await emit_telemetry("swarm_sync", {
                "term": self.current_term,
                "msg_type": "heartbeat",
            })

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            if msg.get("type") in ("register", "revoke", "control"):
                await self._handle_client_command(msg, writer)
            elif msg.get("type") == "raft_request_vote":
                # fixed leader: deny votes
                writer.write(json.dumps({"type": "raft_vote", "term": self.current_term, "vote_granted": False}).encode("utf-8") + b"\n")
                await writer.drain()
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("RegistryPrimary (Raft leader) starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"RegistryPrimary bound on {self.host}:{self.port}")
        asyncio.create_task(self._heartbeat_loop())
        async with server:
            await server.serve_forever()

# -------------------------
# Registry Follower (Raft follower)
# -------------------------
class RegistryFollower(RegistryBase):
    def __init__(self, host=HOST, port=PORT_REGISTRY_FOLLOWER, db_path=DB_REGISTRY_FOLLOWER):
        super().__init__(db_path)
        self.host = host
        self.port = port

        self.current_term = 1
        self.voted_for: Optional[str] = None
        self.log: List[RaftLogEntry] = []
        self.commit_index = -1
        self.last_applied = -1
        self.role = "follower"
        self.last_heartbeat = time.time()
        self.election_timeout = self._new_election_timeout()

    def _new_election_timeout(self):
        return random.uniform(RAFT_ELECTION_TIMEOUT_MIN, RAFT_ELECTION_TIMEOUT_MAX)

    async def _apply_committed(self):
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            cmd = entry.command
            if cmd["type"] == "register":
                self._register_capsule(cmd["capsule"])
                await emit_telemetry("registry_register", {
                    "capsule_id": cmd["capsule"]["capsule_id"],
                    "lineage": cmd["capsule"]["lineage"],
                    "internal_only": cmd["capsule"]["policy"].get("internal_only", False),
                })
            elif cmd["type"] == "revoke":
                self._revoke_capsule(cmd["capsule_id"], cmd["reason"])
                await emit_telemetry("registry_revoke", {
                    "capsule_id": cmd["capsule_id"],
                    "reason": cmd["reason"],
                })

    async def _election_loop(self):
        while True:
            await asyncio.sleep(0.5)
            if time.time() - self.last_heartbeat > self.election_timeout:
                # follower would start election in real Raft; here we just log
                await emit_telemetry("swarm_sync", {
                    "term": self.current_term,
                    "msg_type": "follower_timeout",
                })
                self.last_heartbeat = time.time()
                self.election_timeout = self._new_election_timeout()

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            kind = msg.get("type")
            if kind == "raft_append_entries":
                term = msg["term"]
                if term < self.current_term:
                    writer.write(json.dumps({"type": "raft_append_resp", "term": self.current_term, "success": False}).encode("utf-8") + b"\n")
                    await writer.drain()
                    continue
                self.current_term = term
                self.role = "follower"
                self.last_heartbeat = time.time()
                prev_log_index = msg["prev_log_index"]
                prev_log_term = msg["prev_log_term"]
                if prev_log_index >= 0:
                    if prev_log_index >= len(self.log) or self.log[prev_log_index].term != prev_log_term:
                        writer.write(json.dumps({"type": "raft_append_resp", "term": self.current_term, "success": False}).encode("utf-8") + b"\n")
                        await writer.drain()
                        continue
                # append new entries
                entries = msg["entries"]
                idx = prev_log_index + 1
                for e in entries:
                    if idx < len(self.log):
                        self.log[idx] = RaftLogEntry(e["term"], e["command"])
                    else:
                        self.log.append(RaftLogEntry(e["term"], e["command"]))
                    idx += 1
                self.commit_index = min(msg["leader_commit"], len(self.log) - 1)
                await self._apply_committed()
                writer.write(json.dumps({"type": "raft_append_resp", "term": self.current_term, "success": True}).encode("utf-8") + b"\n")
            elif kind == "raft_request_vote":
                term = msg["term"]
                candidate_id = msg["candidate_id"]
                if term < self.current_term:
                    writer.write(json.dumps({"type": "raft_vote", "term": self.current_term, "vote_granted": False}).encode("utf-8") + b"\n")
                else:
                    self.current_term = term
                    if self.voted_for is None or self.voted_for == candidate_id:
                        self.voted_for = candidate_id
                        writer.write(json.dumps({"type": "raft_vote", "term": self.current_term, "vote_granted": True}).encode("utf-8") + b"\n")
                    else:
                        writer.write(json.dumps({"type": "raft_vote", "term": self.current_term, "vote_granted": False}).encode("utf-8") + b"\n")
                await writer.drain()
            else:
                writer.write(json.dumps({"type": "ack", "status": "ignored"}).encode("utf-8") + b"\n")
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("RegistryFollower (Raft follower) starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"RegistryFollower bound on {self.host}:{self.port}")
        asyncio.create_task(self._election_loop())
        async with server:
            await server.serve_forever()

# -------------------------
# Origin Node
# -------------------------
class OriginNode:
    def __init__(self, host=HOST, port=PORT_ORIGIN,
                 registry_addr=(HOST, PORT_REGISTRY_PRIMARY),
                 relay_addr=(HOST, PORT_RELAY),
                 keyhub_addr=(HOST, PORT_KEYHUB)):
        self.host = host
        self.port = port
        self.registry_addr = registry_addr
        self.relay_addr = relay_addr
        self.keyhub_addr = keyhub_addr
        self.private_key, self.public_key = generate_rsa_keypair()
        self.node_id = "origin-1"
        self._first_capsule_id = None

    async def publish_key(self):
        pub_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        fp = public_key_fingerprint(self.public_key)
        await send_json(self.keyhub_addr, {
            "type": "publish_key",
            "origin_id": self.node_id,
            "pubkey_pem_hex": pub_pem.hex(),
            "fingerprint": fp,
        })

    async def create_and_send_capsule(self):
        payload = {"secret": "top-secret-data", "owner": "org-alpha"}
        capsule = Capsule(
            origin_id=self.node_id,
            origin_endpoint=f"tcp://{self.host}:{self.port}",
            capsule_class="document",
            org_id="org-alpha",
            payload=payload,
            origin_private_key=self.private_key,
            origin_public_key=self.public_key,
            internal_only=True,
        )
        wire = capsule.to_wire()
        self._first_capsule_id = wire["capsule_id"]
        await emit_telemetry("origin_capsule_created", {
            "capsule_id": wire["capsule_id"],
            "lineage": wire["lineage"],
            "internal_only": True,
        })

        await send_json(self.registry_addr, {
            "type": "register",
            "capsule": wire,
        })

        await send_json(self.relay_addr, {
            "type": "capsule",
            "capsule": wire,
        })

        asyncio.create_task(self.create_derived_capsule())

    async def create_derived_capsule(self):
        await asyncio.sleep(2.0)
        if not self._first_capsule_id:
            return
        payload = {"summary": "derived-from-parent", "owner": "org-alpha"}
        capsule = Capsule(
            origin_id=self.node_id,
            origin_endpoint=f"tcp://{self.host}:{self.port}",
            capsule_class="summary",
            org_id="org-alpha",
            payload=payload,
            origin_private_key=self.private_key,
            origin_public_key=self.public_key,
            parents=[self._first_capsule_id],
            derivation_note="Summarized original payload",
            internal_only=False,
        )
        wire = capsule.to_wire()
        await emit_telemetry("origin_capsule_derived", {
            "capsule_id": wire["capsule_id"],
            "parents": wire["lineage"]["parents"],
            "derivation_note": wire["lineage"]["derivation_note"],
            "internal_only": False,
        })

        await send_json(self.registry_addr, {
            "type": "register",
            "capsule": wire,
        })

        await send_json(self.relay_addr, {
            "type": "capsule",
            "capsule": wire,
        })

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            if msg.get("type") == "gate_callback":
                capsule = msg["capsule"]
                resp = {
                    "type": "gate_callback_response",
                    "status": "valid",
                    "expected_manifest_hash": capsule["content_hash"],
                }
                writer.write(json.dumps(resp).encode("utf-8") + b"\n")
                await writer.drain()
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("OriginNode starting...")
        await self.publish_key()
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"OriginNode bound on {self.host}:{self.port}")
        asyncio.create_task(self.create_and_send_capsule())
        async with server:
            await server.serve_forever()

# -------------------------
# Relay Node (multi-gate + trust decay)
# -------------------------
class RelayNode:
    def __init__(self, host=HOST, port=PORT_RELAY,
                 gate_addrs=None):
        self.host = host
        self.port = port
        self.node_id = "relay-1"
        gate_addrs = gate_addrs or [(HOST, PORT_GATE), (HOST, PORT_GATE)]
        self.gates: Dict[str, Dict[str, Any]] = {}
        for i, addr in enumerate(gate_addrs, start=1):
            self.gates[f"gate-{i}"] = {
                "addr": addr,
                "trust": 5.0,
                "last_used": time.time(),
            }

    def _effective_trust(self, info: Dict[str, Any]) -> float:
        age = time.time() - info["last_used"]
        decay = age / 30.0
        return max(info["trust"] - decay, 0.0)

    def choose_next_hop(self) -> Dict[str, Any]:
        best = None
        best_score = -1.0
        for gid, info in self.gates.items():
            score = self._effective_trust(info)
            if score > best_score:
                best_score = score
                best = {"id": gid, **info, "score": score}
        return best

    def _update_trust_on_success(self, gate_id: str):
        now_t = time.time()
        for gid, info in self.gates.items():
            info["trust"] = max(self._effective_trust(info), 0.0)
        if gate_id in self.gates:
            self.gates[gate_id]["trust"] = min(self.gates[gate_id]["trust"] + 0.5, 10.0)
            self.gates[gate_id]["last_used"] = now_t

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            if msg.get("type") == "capsule":
                capsule = msg["capsule"]
                hop = self.choose_next_hop()
                if not hop:
                    writer.write(json.dumps({"type": "error", "reason": "no_gate"}).encode("utf-8") + b"\n")
                    await writer.drain()
                    continue
                await emit_telemetry("cbn_hop", {
                    "from": self.node_id,
                    "to": hop["id"],
                    "capsule_id": capsule["capsule_id"],
                    "trust": hop["score"],
                })
                await send_json(hop["addr"], {
                    "type": "capsule",
                    "capsule": capsule,
                })
                self._update_trust_on_success(hop["id"])
                writer.write(json.dumps({"type": "ack", "status": "forwarded", "gate": hop["id"]}).encode("utf-8") + b"\n")
                await writer.drain()
            elif msg.get("type") == "control":
                cmd = msg.get("command")
                if cmd == "bump_trust":
                    gate_id = msg.get("gate_id", "gate-1")
                    delta = msg.get("delta", 1.0)
                    if gate_id in self.gates:
                        self.gates[gate_id]["trust"] = min(self.gates[gate_id]["trust"] + delta, 10.0)
                    writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
                    await writer.drain()
                else:
                    writer.write(json.dumps({"type": "error", "reason": "unknown_control"}).encode("utf-8") + b"\n")
                    await writer.drain()
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("RelayNode starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"RelayNode bound on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

# -------------------------
# Gate Node (persistent nonces, key cache, resurrection detection, threat scoring)
# -------------------------
class GateNode:
    def __init__(self, host=HOST, port=PORT_GATE,
                 registry_addr=(HOST, PORT_REGISTRY_PRIMARY),
                 origin_addr=(HOST, PORT_ORIGIN),
                 keyhub_addr=(HOST, PORT_KEYHUB),
                 registry_db=DB_REGISTRY_PRIMARY,
                 state_db=DB_GATE):
        self.host = host
        self.port = port
        self.registry_addr = registry_addr
        self.origin_addr = origin_addr
        self.keyhub_addr = keyhub_addr
        self.node_id = "gate-1"
        self.trust_level = 5
        self.is_external = True
        self.registry_db = registry_db
        self.state_db = state_db
        self._init_state_db()
        self.key_cache: Dict[str, Dict[str, Any]] = {}

    def _init_state_db(self):
        conn = sqlite3.connect(self.state_db)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS nonces (
            nonce TEXT PRIMARY KEY,
            ts TEXT
        )
        """)
        conn.commit()
        conn.close()

    async def _revoke(self, capsule_id: str, reason: str):
        await send_json(self.registry_addr, {
            "type": "revoke",
            "capsule_id": capsule_id,
            "reason": reason,
        })

    def _nonce_seen_or_expired(self, nonce: str, ts: str) -> bool:
        if not nonce:
            return True
        try:
            t = parse_iso(ts)
        except Exception:
            return True
        if datetime.now(timezone.utc) - t > timedelta(seconds=REPLAY_WINDOW_SECONDS):
            return True

        conn = sqlite3.connect(self.state_db)
        c = conn.cursor()
        c.execute("SELECT ts FROM nonces WHERE nonce=?", (nonce,))
        row = c.fetchone()
        if row:
            conn.close()
            return True
        c.execute("INSERT INTO nonces (nonce, ts) VALUES (?, ?)", (nonce, ts))
        conn.commit()
        conn.close()
        return False

    async def _get_origin_key(self, origin_id: str):
        now_ts = datetime.now(timezone.utc)
        cached = self.key_cache.get(origin_id)
        if cached and cached["expires_at"] > now_ts:
            return cached["key"]

        resp = await send_json(self.keyhub_addr, {
            "type": "get_key",
            "origin_id": origin_id,
        })
        if resp.get("type") != "key":
            return None
        pub_pem = bytes.fromhex(resp["pubkey_pem_hex"])
        pub = serialization.load_der_public_key(pub_pem)
        self.key_cache[origin_id] = {
            "key": pub,
            "expires_at": now_ts + timedelta(seconds=60),
        }
        return pub

    def _parent_internal_only(self, parent_ids):
        if not parent_ids:
            return False
        conn = sqlite3.connect(self.registry_db)
        c = conn.cursor()
        q_marks = ",".join("?" for _ in parent_ids)
        c.execute(f"SELECT internal_only FROM capsules WHERE capsule_id IN ({q_marks})", parent_ids)
        rows = c.fetchall()
        conn.close()
        return any(row[0] == 1 for row in rows)

    def _get_capsule_status(self, capsule_id: str) -> str:
        conn = sqlite3.connect(self.registry_db)
        c = conn.cursor()
        c.execute("SELECT status FROM capsules WHERE capsule_id=?", (capsule_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else ""

    async def _score_threat(self, capsule_id: str, reasons: List[str]):
        base = 0
        for r in reasons:
            if r == "replay_detected":
                base += 40
            elif r == "integrity_failed":
                base += 50
            elif r == "no_key":
                base += 30
            elif r == "signature_invalid":
                base += 50
            elif r == "lineage_internal_only_violation":
                base += 60
            elif r == "origin_invalid":
                base += 40
            elif r == "resurrection_detected":
                base += 70
        score = min(base, 100)
        await emit_telemetry("threat_score", {
            "capsule_id": capsule_id,
            "score": score,
            "reasons": reasons,
            "gate": self.node_id,
        })

    async def handle_capsule(self, capsule_wire: dict, writer):
        cid = capsule_wire["capsule_id"]
        await emit_telemetry("gate_received_capsule", {"capsule_id": cid})

        reasons = []

        status = self._get_capsule_status(cid)
        if status.startswith("revoked:"):
            reasons.append("resurrection_detected")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "resurrection_detected"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "resurrection_detected")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        if self._nonce_seen_or_expired(capsule_wire.get("nonce", ""), capsule_wire.get("timestamp", now_iso())):
            reasons.append("replay_detected")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "replay_detected"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "replay_detected")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        core = {
            "capsule_id": capsule_wire["capsule_id"],
            "origin_id": capsule_wire["origin_id"],
            "created_at": capsule_wire["created_at"],
            "capsule_class": capsule_wire["capsule_class"],
            "version": capsule_wire["version"],
            "lineage": capsule_wire["lineage"],
            "policy_root_hash": capsule_wire["policy_root_hash"],
            "encrypted_payload": capsule_wire["encrypted_payload"],
            "encrypted_key_hex": capsule_wire["encrypted_key_hex"],
        }
        recomputed_hash = hash_json(core)
        if recomputed_hash != capsule_wire["content_hash"]:
            reasons.append("integrity_failed")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "integrity_failed"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "integrity_failed")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        pub = await self._get_origin_key(capsule_wire["origin_id"])
        if not pub:
            reasons.append("no_key")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "no_key"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "no_key")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        sig = bytes.fromhex(capsule_wire["origin_signature"])
        if not verify_signature(pub, sig, capsule_wire["content_hash"].encode("utf-8")):
            reasons.append("signature_invalid")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "signature_invalid"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "signature_invalid")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        parents = capsule_wire["lineage"].get("parents", [])
        child_internal_only = capsule_wire["policy"].get("internal_only", False)
        if self.is_external and not child_internal_only and self._parent_internal_only(parents):
            reasons.append("lineage_internal_only_violation")
            await emit_telemetry("predictive_risk", {
                "capsule_id": cid,
                "parents": parents,
                "reason": "lineage_internal_only_violation",
                "gate": self.node_id,
            })
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "lineage_internal_only_violation"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "lineage_internal_only_violation")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        await emit_telemetry("gate_payload_access", {"capsule_id": cid})

        resp = await send_json(self.origin_addr, {
            "type": "gate_callback",
            "capsule": capsule_wire,
        })
        if resp.get("status") != "valid":
            reasons.append("origin_invalid")
            await emit_telemetry("capsule_violation", {"capsule_id": cid, "reason": "origin_invalid"})
            await self._score_threat(cid, reasons)
            await self._revoke(cid, "origin_invalid")
            writer.write(json.dumps({"type": "result", "status": "rejected"}).encode("utf-8") + b"\n")
            await writer.drain()
            return

        await self._score_threat(cid, reasons)
        await emit_telemetry("gate_consumed_capsule", {"capsule_id": cid})
        writer.write(json.dumps({"type": "result", "status": "accepted"}).encode("utf-8") + b"\n")
        await writer.drain()

    async def handle_client(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                writer.write(json.dumps({"type": "error", "reason": "invalid_json"}).encode("utf-8") + b"\n")
                await writer.drain()
                continue
            if msg.get("type") == "capsule":
                await self.handle_capsule(msg["capsule"], writer)
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
                await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("GateNode starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"GateNode bound on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

# -------------------------
# Cockpit GUI (Office-style ribbon + visuals)
# -------------------------
class CockpitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CBN Swarm Diagnostic Cockpit")
        self.root.geometry("1400x800")

        self.node_last_seen: Dict[str, datetime] = {}
        self.node_status_items: Dict[str, str] = {}
        self.last_telemetry_msg: str = "None yet"

        self.threat_rows: Dict[str, str] = {}
        self.capsule_last_event: Dict[str, dict] = {}
        self.capsule_lineage: Dict[str, List[str]] = {}
        self.capsule_parents: Dict[str, List[str]] = {}
        self.capsule_scores: Dict[str, int] = {}

        self.swarm_edges: List[tuple] = []

        log_diag("Cockpit startup: building layout...")
        self._build_layout()
        log_diag("Cockpit startup: layout built, starting telemetry pump...")
        self._start_telemetry_pump()
        self._start_heartbeat_monitor()
        log_diag("Cockpit startup: telemetry pump and heartbeat monitor started.")

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=5)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_timeline = ttk.Frame(self.notebook)
        self.tab_telemetry = ttk.Frame(self.notebook)
        self.tab_threats = ttk.Frame(self.notebook)
        self.tab_lineage = ttk.Frame(self.notebook)
        self.tab_swarm = ttk.Frame(self.notebook)
        self.tab_diag = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text="Overview")
        self.notebook.add(self.tab_timeline, text="Timeline")
        self.notebook.add(self.tab_telemetry, text="Telemetry")
        self.notebook.add(self.tab_threats, text="Threats")
        self.notebook.add(self.tab_lineage, text="Lineage")
        self.notebook.add(self.tab_swarm, text="Swarm")
        self.notebook.add(self.tab_diag, text="Diagnostics")

        self._build_overview_tab()
        self._build_timeline_tab()
        self._build_telemetry_tab()
        self._build_threats_tab()
        self._build_lineage_tab()
        self._build_swarm_tab()
        self._build_diag_tab()

    def _build_overview_tab(self):
        f = self.tab_overview
        f.columnconfigure(0, weight=2)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=2)
        f.rowconfigure(1, weight=1)

        hb_frame = ttk.LabelFrame(f, text="Node Heartbeats")
        hb_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        hb_frame.columnconfigure(0, weight=1)
        hb_frame.rowconfigure(0, weight=1)

        self.nodes_tree = ttk.Treeview(hb_frame, columns=("role", "status", "last_seen"), show="headings", height=10)
        self.nodes_tree.heading("role", text="Role")
        self.nodes_tree.heading("status", text="Status")
        self.nodes_tree.heading("last_seen", text="Last Seen")
        self.nodes_tree.column("role", width=120, anchor="w")
        self.nodes_tree.column("status", width=100, anchor="w")
        self.nodes_tree.column("last_seen", width=200, anchor="w")
        self.nodes_tree.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        for role in ["keyhub", "registry_primary", "registry_follower", "origin", "relay", "gate", "cockpit"]:
            item_id = self.nodes_tree.insert("", "end", values=(role, "spawned", "never"))
            self.node_status_items[role] = item_id

        activity_frame = ttk.LabelFrame(f, text="Node Activity Indicators")
        activity_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        activity_frame.columnconfigure(0, weight=1)

        self.origin_activity = ttk.Label(activity_frame, text="Origin: idle")
        self.origin_activity.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.relay_activity = ttk.Label(activity_frame, text="Relay: idle")
        self.relay_activity.grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.gate_activity = ttk.Label(activity_frame, text="Gate: idle")
        self.gate_activity.grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.registry_activity = ttk.Label(activity_frame, text="Registry: idle")
        self.registry_activity.grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.keyhub_activity = ttk.Label(activity_frame, text="KeyHub: idle")
        self.keyhub_activity.grid(row=4, column=0, sticky="w", padx=4, pady=2)

        last_frame = ttk.LabelFrame(f, text="Last Telemetry Message")
        last_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.last_msg_label = ttk.Label(last_frame, text="None yet")
        self.last_msg_label.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.status_label = ttk.Label(f, text="Live telemetry: waiting...")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=6)

    def _build_timeline_tab(self):
        f = self.tab_timeline
        f.columnconfigure(0, weight=3)
        f.columnconfigure(1, weight=2)
        f.rowconfigure(0, weight=1)

        cols = ("time", "event", "capsule_id", "detail")
        self.timeline = ttk.Treeview(f, columns=cols, show="headings")
        for c in cols:
            self.timeline.heading(c, text=c.capitalize())
            self.timeline.column(c, width=150 if c != "detail" else 400, anchor="w")
        self.timeline.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        scroll = ttk.Scrollbar(f, orient="vertical", command=self.timeline.yview)
        self.timeline.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

        inspector_frame = ttk.LabelFrame(f, text="Capsule Drill-Down")
        inspector_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        inspector_frame.columnconfigure(0, weight=1)
        inspector_frame.rowconfigure(0, weight=1)

        self.inspector_text = tk.Text(inspector_frame, wrap="word", height=20)
        self.inspector_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        insp_scroll = ttk.Scrollbar(inspector_frame, orient="vertical", command=self.inspector_text.yview)
        self.inspector_text.configure(yscrollcommand=insp_scroll.set)
        insp_scroll.grid(row=0, column=1, sticky="ns")

        self.timeline.bind("<<TreeviewSelect>>", self._on_timeline_select)

    def _build_telemetry_tab(self):
        f = self.tab_telemetry
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.telemetry_list = tk.Text(f, wrap="none")
        self.telemetry_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.telemetry_list.yview)
        self.telemetry_list.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_threats_tab(self):
        f = self.tab_threats
        f.columnconfigure(0, weight=2)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=1)

        cols = ("capsule_id", "score", "reasons", "gate", "time")
        self.threat_tree = ttk.Treeview(f, columns=cols, show="headings")
        for c in cols:
            self.threat_tree.heading(c, text=c.capitalize())
            self.threat_tree.column(c, width=150 if c != "reasons" else 300, anchor="w")
        self.threat_tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        scroll = ttk.Scrollbar(f, orient="vertical", command=self.threat_tree.yview)
        self.threat_tree.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

        heatmap_frame = ttk.LabelFrame(f, text="Threat Heatmap")
        heatmap_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        heatmap_frame.columnconfigure(0, weight=1)
        heatmap_frame.rowconfigure(0, weight=1)

        self.heatmap_canvas = tk.Canvas(heatmap_frame, bg="black")
        self.heatmap_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    def _build_lineage_tab(self):
        f = self.tab_lineage
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.lineage_tree = ttk.Treeview(f, columns=("capsule_id", "parents"), show="tree headings")
        self.lineage_tree.heading("#0", text="Capsule")
        self.lineage_tree.heading("capsule_id", text="Capsule ID")
        self.lineage_tree.heading("parents", text="Parents")
        self.lineage_tree.column("#0", width=200, anchor="w")
        self.lineage_tree.column("capsule_id", width=250, anchor="w")
        self.lineage_tree.column("parents", width=300, anchor="w")
        self.lineage_tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        scroll = ttk.Scrollbar(f, orient="vertical", command=self.lineage_tree.yview)
        self.lineage_tree.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_swarm_tab(self):
        f = self.tab_swarm
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.swarm_canvas = tk.Canvas(f, bg="#101010")
        self.swarm_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.swarm_nodes_positions = {
            "origin": (150, 200),
            "relay": (400, 200),
            "gate": (650, 200),
            "registry_primary": (900, 150),
            "registry_follower": (900, 250),
            "keyhub": (650, 50),
            "cockpit": (150, 50),
        }

        self._draw_swarm_base()

    def _draw_swarm_base(self):
        self.swarm_canvas.delete("all")
        for node, (x, y) in self.swarm_nodes_positions.items():
            self.swarm_canvas.create_oval(x-30, y-30, x+30, y+30, fill="#202020", outline="#808080")
            self.swarm_canvas.create_text(x, y, text=node, fill="white")

    def _update_swarm_edge(self, src: str, dst: str):
        self.swarm_edges.append((src, dst, time.time()))
        self._draw_swarm_base()
        now_t = time.time()
        for (s, d, t0) in self.swarm_edges:
            age = now_t - t0
            if age > 5.0:
                continue
            alpha = max(0.2, 1.0 - age / 5.0)
            color = "#%02x0000" % int(255 * alpha)
            if s in self.swarm_nodes_positions and d in self.swarm_nodes_positions:
                x1, y1 = self.swarm_nodes_positions[s]
                x2, y2 = self.swarm_nodes_positions[d]
                self.swarm_canvas.create_line(x1, y1, x2, y2, fill=color, width=3)

    def _build_diag_tab(self):
        f = self.tab_diag
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        self.diag_list = tk.Text(f, wrap="word")
        self.diag_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scroll = ttk.Scrollbar(f, orient="vertical", command=self.diag_list.yview)
        self.diag_list.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

        self._diag_append("Cockpit layout initialized.")

    def _diag_append(self, text: str):
        ts = now_iso()
        line = f"{ts} | {text}\n"
        self.diag_list.insert("end", line)
        self.diag_list.see("end")
        log_diag(text)

    def _start_telemetry_pump(self):
        self._poll_telemetry()

    def _poll_telemetry(self):
        try:
            while True:
                msg = TELEMETRY_QUEUE.get_nowait()
                try:
                    self._handle_telemetry(msg)
                except Exception as e:
                    err = f"Exception in _handle_telemetry: {e}\n{traceback.format_exc()}"
                    self._diag_append(err)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self._poll_telemetry)
        except Exception as e:
            err = f"Exception scheduling GUI refresh: {e}\n{traceback.format_exc()}"
            self._diag_append(err)

    def _start_heartbeat_monitor(self):
        self._heartbeat_tick()

    def _heartbeat_tick(self):
        now_t = datetime.now(timezone.utc)
        for role, item_id in self.node_status_items.items():
            last_seen = self.node_last_seen.get(role)
            if last_seen is None:
                status = "spawned"
            else:
                delta = (now_t - last_seen).total_seconds()
                status = "ok" if delta <= NODE_SILENCE_SECONDS else "silent"
            vals = self.nodes_tree.item(item_id, "values")
            self.nodes_tree.item(item_id, values=(vals[0], status, vals[2]))
        self.root.after(1000, self._heartbeat_tick)

    def _update_node_heartbeat(self, role: str, ts: str):
        try:
            dt = parse_iso(ts)
        except Exception:
            dt = datetime.now(timezone.utc)
        self.node_last_seen[role] = dt
        item_id = self.node_status_items.get(role)
        if item_id:
            vals = self.nodes_tree.item(item_id, "values")
            self.nodes_tree.item(item_id, values=(vals[0], "ok", ts))

    def _update_threats(self, ts: str, payload: dict):
        cid = payload.get("capsule_id", "")
        score = payload.get("score", 0)
        reasons = ",".join(payload.get("reasons", []))
        gate = payload.get("gate", "")
        self.capsule_scores[cid] = score
        if cid in self.threat_rows:
            item_id = self.threat_rows[cid]
            self.threat_tree.item(item_id, values=(cid, score, reasons, gate, ts))
        else:
            item_id = self.threat_tree.insert("", "end", values=(cid, score, reasons, gate, ts))
            self.threat_rows[cid] = item_id
        self._redraw_heatmap()

    def _redraw_heatmap(self):
        self.heatmap_canvas.delete("all")
        w = int(self.heatmap_canvas.winfo_width() or 300)
        h = int(self.heatmap_canvas.winfo_height() or 300)
        if not self.capsule_scores:
            return
        items = list(self.capsule_scores.items())
        n = len(items)
        bar_width = max(20, w // max(1, n))
        for i, (cid, score) in enumerate(items):
            x0 = i * bar_width + 5
            x1 = x0 + bar_width - 10
            height = int((score / 100.0) * (h - 20))
            y0 = h - 10 - height
            y1 = h - 10
            r = int(255 * (score / 100.0))
            g = int(255 * (1 - score / 100.0))
            color = f"#{r:02x}{g:02x}00"
            self.heatmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            self.heatmap_canvas.create_text((x0+x1)//2, y0-10, text=cid[:6], fill="white", anchor="s", font=("TkDefaultFont", 7))

    def _update_lineage(self, cid: str, parents: List[str]):
        self.capsule_parents[cid] = parents
        for p in parents:
            self.capsule_lineage.setdefault(p, []).append(cid)
        self._redraw_lineage_tree()

    def _redraw_lineage_tree(self):
        self.lineage_tree.delete(*self.lineage_tree.get_children())
        roots = set(self.capsule_parents.keys())
        for cid, parents in self.capsule_parents.items():
            for p in parents:
                if p in roots:
                    roots.discard(cid)
        node_ids = {}

        def add_node(cid, parent_tree_id=""):
            parents = self.capsule_parents.get(cid, [])
            label = cid
            parent_str = ",".join(parents)
            if parent_tree_id:
                tid = self.lineage_tree.insert(parent_tree_id, "end", text=label, values=(cid, parent_str))
            else:
                tid = self.lineage_tree.insert("", "end", text=label, values=(cid, parent_str))
            node_ids[cid] = tid
            for child in self.capsule_lineage.get(cid, []):
                add_node(child, tid)

        for root_cid in roots:
            add_node(root_cid)

    def _update_capsule_last_event(self, cid: str, event_type: str, payload: dict, ts: str):
        self.capsule_last_event[cid] = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": ts,
        }

    def _on_timeline_select(self, event):
        sel = self.timeline.selection()
        if not sel:
            return
        item_id = sel[0]
        vals = self.timeline.item(item_id, "values")
        if len(vals) < 3:
            return
        cid = vals[2]
        info = self.capsule_last_event.get(cid, {})
        score = self.capsule_scores.get(cid, None)
        parents = self.capsule_parents.get(cid, [])
        lines = []
        lines.append(f"Capsule ID: {cid}")
        lines.append(f"Last event: {info.get('event_type','unknown')} at {info.get('timestamp','')}")
        lines.append(f"Payload: {json.dumps(info.get('payload',{}), indent=2)}")
        lines.append(f"Parents: {parents}")
        if score is not None:
            lines.append(f"Threat score: {score}")
        self.inspector_text.delete("1.0", "end")
        self.inspector_text.insert("end", "\n".join(lines))

    def _handle_telemetry(self, msg: dict):
        ts = msg.get("timestamp", "")
        et = msg.get("event_type", "")
        payload = msg.get("payload", {})

        line = f"{ts} | {et} | {payload}\n"
        self.telemetry_list.insert("end", line)
        self.telemetry_list.see("end")

        self.last_telemetry_msg = line.strip()
        self.last_msg_label.config(text=self.last_telemetry_msg)

        capsule_id = payload.get("capsule_id", "")
        detail = ""
        if et == "capsule_violation":
            detail = payload.get("reason", "")
        elif et == "predictive_risk":
            detail = f"{payload.get('reason','')} via {payload.get('gate','')}"
        elif et == "cbn_hop":
            detail = f"{payload.get('from','')} -> {payload.get('to','')} (trust={payload.get('trust',0)})"
        elif et.startswith("origin_capsule"):
            detail = payload.get("lineage", "")
        elif et == "swarm_sync":
            detail = f"term={payload.get('term')} type={payload.get('msg_type')}"

        if et in (
            "origin_capsule_created",
            "origin_capsule_derived",
            "cbn_hop",
            "gate_received_capsule",
            "gate_consumed_capsule",
            "capsule_violation",
            "predictive_risk",
            "registry_register",
            "registry_revoke",
            "keyhub_publish",
            "swarm_sync",
            "threat_score",
        ):
            self.timeline.insert("", "end", values=(ts, et, capsule_id, detail))

        if capsule_id:
            self._update_capsule_last_event(capsule_id, et, payload, ts)

        if et == "threat_score":
            self._update_threats(ts, payload)

        if et in ("origin_capsule_created", "origin_capsule_derived"):
            cid = payload.get("capsule_id", "")
            lineage = payload.get("lineage", {})
            parents = lineage.get("parents", [])
            if cid:
                self._update_lineage(cid, parents)

        if et == "cbn_hop":
            src = payload.get("from", "relay")
            dst = payload.get("to", "gate")
            self._update_swarm_edge(src.replace("-1",""), dst.replace("-1",""))

        if et.startswith("origin_"):
            self._update_node_heartbeat("origin", ts)
            self.origin_activity.config(text=f"Origin: active ({et})")
        elif et.startswith("cbn_hop"):
            self._update_node_heartbeat("relay", ts)
            self.relay_activity.config(text=f"Relay: active ({et})")
        elif et.startswith("gate_") or et in ("predictive_risk", "capsule_violation", "threat_score"):
            self._update_node_heartbeat("gate", ts)
            self.gate_activity.config(text=f"Gate: active ({et})")
        elif et.startswith("registry_") or et == "swarm_sync":
            self._update_node_heartbeat("registry_primary", ts)
            self.registry_activity.config(text=f"Registry: active ({et})")
        elif et.startswith("keyhub_"):
            self._update_node_heartbeat("keyhub", ts)
            self.keyhub_activity.config(text=f"KeyHub: active ({et})")

        self.status_label.config(text=f"Last event: {et}")

def telemetry_thread_main():
    async def server():
        async def handle(reader, writer):
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                TELEMETRY_QUEUE.put(msg)
            writer.close()
            await writer.wait_closed()

        try:
            log_diag("Telemetry server attempting bind...")
            server = await asyncio.start_server(handle, HOST, PORT_TELEMETRY)
            log_diag(f"Telemetry server bound on {HOST}:{PORT_TELEMETRY}")
        except OSError as e:
            TELEMETRY_QUEUE.put({
                "type": "telemetry",
                "event_type": "telemetry_bind_error",
                "timestamp": now_iso(),
                "payload": {"error": str(e), "port": PORT_TELEMETRY},
            })
            return
        async with server:
            await server.serve_forever()

    asyncio.run(server())

def cockpit_gui():
    root = tk.Tk()
    app = CockpitApp(root)
    root.mainloop()

def cockpit_main():
    log_diag("Cockpit_main: starting telemetry thread...")
    t = threading.Thread(target=telemetry_thread_main, daemon=True)
    t.start()
    time.sleep(0.3)
    log_diag("Cockpit_main: launching GUI...")
    cockpit_gui()

# -------------------------
# Launcher
# -------------------------
def launcher():
    script = Path(__file__).absolute()
    roles = [
        "keyhub",
        "registry_primary",
        "registry_follower",
        "origin",
        "relay",
        "gate",
        "cockpit",
    ]
    procs = []
    log_diag("Launcher: spawning roles...")
    for r in roles:
        p = subprocess.Popen([sys.executable, str(script), r])
        procs.append(p)
        time.sleep(0.5)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        log_diag("Launcher: KeyboardInterrupt, terminating children...")
        for p in procs:
            p.terminate()

# -------------------------
# Entry Point
# -------------------------
def main():
    role = sys.argv[1] if len(sys.argv) >= 2 else "launcher"
    log_diag(f"[CBN] starting role: {role}")

    if role == "launcher":
        launcher()
    elif role == "keyhub":
        asyncio.run(KeyHub().run())
    elif role == "registry_primary":
        asyncio.run(RegistryPrimary().run())
    elif role == "registry_follower":
        asyncio.run(RegistryFollower().run())
    elif role == "origin":
        asyncio.run(OriginNode().run())
    elif role == "relay":
        asyncio.run(RelayNode().run())
    elif role == "gate":
        asyncio.run(GateNode().run())
    elif role == "cockpit":
        cockpit_main()
    else:
        print("Unknown role:", role)

if __name__ == "__main__":
    main()

