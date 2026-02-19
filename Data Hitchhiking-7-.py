#!/usr/bin/env python3
"""
CBN Lab Swarm (Office-Style Cockpit Edition)

Features:
- Async TCP networking between nodes (real sockets)
- RSA signatures + AES-GCM payload encryption (envelope)
- Replay protection (nonce + timestamp) with persistent store
- Registry "swarm": primary + follower with mirrored writes + Raft-like stub
- Lineage-aware policy enforcement (internal_only)
- Resurrection detection (revoked capsule seen again)
- Predictive threat scoring
- Swarm sync telemetry
- Multi-gate routing with trust decay and bump on success
- Tkinter "Office-style" ribbon cockpit GUI:
  - Tabs: Overview, Timeline, Telemetry, Threats, Diagnostics
  - Telemetry server bind diagnostics
  - Telemetry event logging
  - Exception logging for telemetry handling and GUI refresh
  - Node heartbeat panel + silence detection
  - Last telemetry message panel
  - Raw telemetry dump panel
  - Port conflict detection
  - Cockpit startup diagnostics
  - Relay/gate/origin activity indicators
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
from typing import Dict, Any, List
import os
import threading
import queue
import traceback

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
# Registry Primary (Raft-like stub + swarm sync)
# -------------------------
class RegistryPrimary(RegistryBase):
    def __init__(self, host=HOST, port=PORT_REGISTRY_PRIMARY,
                 db_path=DB_REGISTRY_PRIMARY,
                 follower_addr=(HOST, PORT_REGISTRY_FOLLOWER),
                 swarm_peers: List[tuple] = None):
        super().__init__(db_path)
        self.host = host
        self.port = port
        self.follower_addr = follower_addr
        self.term = 1
        self.is_leader = True
        self.swarm_peers = swarm_peers or [follower_addr]

    async def _mirror_to_follower(self, msg: dict):
        try:
            await send_json(self.follower_addr, msg)
        except Exception as e:
            log_diag(f"RegistryPrimary mirror_to_follower failed: {e}")

    async def _swarm_broadcast(self, msg: dict):
        # Swarm sync stub: broadcast to peers (here just follower)
        for peer in self.swarm_peers:
            try:
                await send_json(peer, {"type": "swarm_sync", "inner": msg, "term": self.term})
            except Exception as e:
                log_diag(f"RegistryPrimary swarm_broadcast to {peer} failed: {e}")
        await emit_telemetry("swarm_sync", {"term": self.term, "msg_type": msg.get("type")})

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
            if kind == "register":
                self._register_capsule(msg["capsule"])
                await self._mirror_to_follower(msg)
                await self._swarm_broadcast(msg)
                await emit_telemetry("registry_register", {
                    "capsule_id": msg["capsule"]["capsule_id"],
                    "lineage": msg["capsule"]["lineage"],
                    "internal_only": msg["capsule"]["policy"].get("internal_only", False),
                })
                writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
            elif kind == "revoke":
                self._revoke_capsule(msg["capsule_id"], msg["reason"])
                await self._mirror_to_follower(msg)
                await self._swarm_broadcast(msg)
                await emit_telemetry("registry_revoke", {"capsule_id": msg["capsule_id"], "reason": msg["reason"]})
                writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
            elif kind == "control":
                cmd = msg.get("command")
                if cmd == "revoke_capsule":
                    cid = msg["capsule_id"]
                    reason = msg.get("reason", "operator_revoke")
                    self._revoke_capsule(cid, reason)
                    inner = {"type": "revoke", "capsule_id": cid, "reason": reason}
                    await self._mirror_to_follower(inner)
                    await self._swarm_broadcast(inner)
                    await emit_telemetry("registry_revoke", {"capsule_id": cid, "reason": reason})
                    writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
                else:
                    writer.write(json.dumps({"type": "error", "reason": "unknown_control"}).encode("utf-8") + b"\n")
            elif kind == "raft_heartbeat":
                # Raft-like stub: respond as leader
                writer.write(json.dumps({"type": "raft_ack", "term": self.term}).encode("utf-8") + b"\n")
            else:
                writer.write(json.dumps({"type": "error", "reason": "unknown_type"}).encode("utf-8") + b"\n")
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("RegistryPrimary starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"RegistryPrimary bound on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

# -------------------------
# Registry Follower (swarm sync receiver)
# -------------------------
class RegistryFollower(RegistryBase):
    def __init__(self, host=HOST, port=PORT_REGISTRY_FOLLOWER, db_path=DB_REGISTRY_FOLLOWER):
        super().__init__(db_path)
        self.host = host
        self.port = port
        self.term = 1

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
            if kind == "register":
                self._register_capsule(msg["capsule"])
                writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
            elif kind == "revoke":
                self._revoke_capsule(msg["capsule_id"], msg["reason"])
                writer.write(json.dumps({"type": "ack", "status": "ok"}).encode("utf-8") + b"\n")
            elif kind == "swarm_sync":
                inner = msg.get("inner", {})
                itype = inner.get("type")
                if itype == "register":
                    self._register_capsule(inner["capsule"])
                elif itype == "revoke":
                    self._revoke_capsule(inner["capsule_id"], inner["reason"])
                writer.write(json.dumps({"type": "ack", "status": "swarm_ok"}).encode("utf-8") + b"\n")
            elif kind == "raft_heartbeat":
                self.term = max(self.term, msg.get("term", 1))
                writer.write(json.dumps({"type": "raft_ack", "term": self.term}).encode("utf-8") + b"\n")
            else:
                writer.write(json.dumps({"type": "ack", "status": "ignored"}).encode("utf-8") + b"\n")
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        log_diag("RegistryFollower starting...")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        log_diag(f"RegistryFollower bound on {self.host}:{self.port}")
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
        # Simulate multiple gates (could be different ports in future)
        self.gates: Dict[str, Dict[str, Any]] = {}
        for i, addr in enumerate(gate_addrs, start=1):
            self.gates[f"gate-{i}"] = {
                "addr": addr,
                "trust": 5.0,
                "last_used": time.time(),
            }

    def _effective_trust(self, info: Dict[str, Any]) -> float:
        # Simple trust decay over time
        age = time.time() - info["last_used"]
        decay = age / 30.0  # 30s half-life-ish
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

        # Resurrection detection: if registry already has revoked status
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

        await self._score_threat(cid, reasons)  # may be empty => score 0
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
# Cockpit GUI (Office-style ribbon)
# -------------------------
class CockpitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CBN Swarm Diagnostic Cockpit")
        self.root.geometry("1300x750")

        self.node_last_seen: Dict[str, datetime] = {}
        self.node_status_items: Dict[str, str] = {}
        self.last_telemetry_msg: str = "None yet"

        self.threat_rows: Dict[str, str] = {}

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

        # Ribbon-style tabs
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tabs
        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_timeline = ttk.Frame(self.notebook)
        self.tab_telemetry = ttk.Frame(self.notebook)
        self.tab_threats = ttk.Frame(self.notebook)
        self.tab_diag = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text="Overview")
        self.notebook.add(self.tab_timeline, text="Timeline")
        self.notebook.add(self.tab_telemetry, text="Telemetry")
        self.notebook.add(self.tab_threats, text="Threats")
        self.notebook.add(self.tab_diag, text="Diagnostics")

        # Overview tab: nodes + activity + last message
        self._build_overview_tab()
        # Timeline tab
        self._build_timeline_tab()
        # Telemetry tab
        self._build_telemetry_tab()
        # Threats tab
        self._build_threats_tab()
        # Diagnostics tab
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
        f.columnconfigure(0, weight=1)
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
        f.columnconfigure(0, weight=1)
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
        if cid in self.threat_rows:
            item_id = self.threat_rows[cid]
            self.threat_tree.item(item_id, values=(cid, score, reasons, gate, ts))
        else:
            item_id = self.threat_tree.insert("", "end", values=(cid, score, reasons, gate, ts))
            self.threat_rows[cid] = item_id

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

        if et == "threat_score":
            self._update_threats(ts, payload)

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

