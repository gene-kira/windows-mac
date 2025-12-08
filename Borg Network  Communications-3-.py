# ============================================================
# Borg Mesh + Mirror Defense — Live GUI + Dual Network Scanners
# Persistent memory (with optional JSON), queen board agents,
# External web auto-expansion (no cap), Local Borg internet (dynamic ports),
# Adaptive Codex Mutation, Zero-Trust Guard, Timed Purge Policies,
# Sensitive-data encryption + GUI chameleon masking, and short-lived decoy telemetry.
# ============================================================

# ---------------------- Imports -----------------------------
import os
import json
import random
import time
import threading
import queue
import datetime
import ssl
import socket
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Any, Optional

# GUI
import tkinter as tk
from tkinter import ttk

# ---------------------- Config ------------------------------
BORG_MESH_CONFIG = {
    "max_corridors": 60_000,
    "unknown_bias": 0.6,
    "crawl_timeout": 6,
    "crawl_delay": 0.7,
    "max_body_bytes": 256_000,
    "user_agent": "BorgMesh/1.7 (+respectful-crawler; admin-controlled)",
    "memory_flush_interval": 15,
    "ghost_retention": 5
}

# Persistence options (enable/disable JSON persistence)
PERSISTENCE = {
    "enabled": True,
    "auto_save_on_checkpoint": True,
    "path": "memory.json"
}

# External web seed allow-list (auto-expands with no cap)
ALLOWED_DOMAINS: Set[str] = {
    "example.com",
    "iana.org",
}

# Local Borg internet seeds: hostnames only (ports discovered dynamically)
LOCAL_HOST_SEEDS: List[str] = [
    "localhost",
    "127.0.0.1",
    # "intranet.local",
]

# Local port scan configuration:
LOCAL_PORTS: Optional[List[int]] = None  # None => 1..65535
LOCAL_MAX_CONCURRENT_PROBES = 200

# ---------------------- Data Tags & Retention Policies ----------------------
RETENTION = {
    "NETWORK_IDENTIFIERS": 30,    # IP/MAC-like identifiers: purge in 30s
    "FAKE_TELEMETRY": 30,         # decoy telemetry purges in 30s after leaving system
    "PERSONAL_DATA": 86400,       # 1 day
    "SYSTEM_INFO": 30,            # short-lived system info
    "DEFAULT": 600                # 10 minutes
}

TAGS = {
    "PERSONAL": "PERSONAL_DATA",
    "BIOMETRIC": "PERSONAL_DATA",
    "SYSTEM": "SYSTEM_INFO",
    "FAKE": "FAKE_TELEMETRY",
    "NET_ID": "NETWORK_IDENTIFIERS"
}

# ---------------------- Bridge & Core Systems --------------------
class Bridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._messages: List[Dict[str, Any]] = []

    def emit(self, channel: str, message: str, tags: List[str] = None):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        record = {"time": ts, "channel": channel, "tags": tags or [], "message": message}
        with self._lock:
            self._messages.append(record)
        print(f"[{ts}] [{channel}] {' '.join(tags or [])} :: {message}")

    def get_recent(self, channel: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._messages[-limit:]
        if channel is None:
            return data
        return [m for m in data if m["channel"] == channel]

class Responder:
    def quarantine(self, path: str):
        print(f"QUARANTINE: {path} (non-destructive placeholder)")

class Trust:
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
    def update(self, key: str, scopes: List[str]):
        self.bridge.emit("trust", f"update {key} scopes={scopes}", ["trust"])

class BorgCommsRouter:
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
    def send_secure(self, topic: str, message: str, channel: str = "Default"):
        self.bridge.emit(channel, f"{topic} :: {message}", ["comms"])

# ---------------------- Zero-Trust Guard --------------------
class ZeroTrustGuard:
    """
    Minimal zero-trust policy engine:
    - Deny-by-default: agents must be explicitly allowed.
    - Ops: "scan:web", "scan:local", "scan:mesh", "build", "enforce".
    - Sensitive ops require scopes (e.g., 'patrol').
    """
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self._lock = threading.Lock()
        self.allowlist: Dict[str, Set[str]] = {}
        self.scopes: Dict[str, Set[str]] = {}

    def allow(self, agent_label: str, op: str):
        with self._lock:
            self.allowlist.setdefault(agent_label, set()).add(op)
            self.bridge.emit("ztrust", f"allow {agent_label} op={op}", ["policy"])

    def grant_scope(self, agent_label: str, scope: str):
        with self._lock:
            self.scopes.setdefault(agent_label, set()).add(scope)
            self.bridge.emit("ztrust", f"scope {agent_label} +{scope}", ["policy"])

    def can(self, agent_label: str, op: str, required_scope: Optional[str] = None) -> bool:
        with self._lock:
            ok = op in self.allowlist.get(agent_label, set())
            scoped = True if required_scope is None else (required_scope in self.scopes.get(agent_label, set()))
        if not ok or not scoped:
            self.bridge.emit("ztrust", f"DENY {agent_label} op={op} scope={required_scope}", ["deny"])
        return ok and scoped

# ---------------------- Encryption (Mirror + Reverse Polarity) ----------------
class MirrorCipher:
    """
    Conceptual symmetric cipher (demo-level):
    - "Mirror": reverse bytes
    - "Reverse polarity": XOR with 0xAA
    Replace with proven crypto for production.
    """
    def __init__(self, key: int = 0xAA):
        self.key = key
    def encrypt(self, s: str) -> bytes:
        b = s.encode("utf-8", errors="ignore")[::-1]
        return bytes([(x ^ self.key) for x in b])
    def decrypt(self, data: bytes) -> str:
        b = bytes([(x ^ self.key) for x in data])[::-1]
        return b.decode("utf-8", errors="ignore")

# ---------------------- Purge Manager -----------------------
class PurgeManager:
    """
    Timed purges for sensitive data classes.
    Ephemeral store: id -> encrypted data, tag, expiry.
    """
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self._lock = threading.Lock()
        self.store: Dict[str, Dict[str, Any]] = {}
        self.cipher = MirrorCipher()
        self.running = True
        threading.Thread(target=self._reaper, daemon=True).start()

    def add(self, payload_id: str, plaintext: str, tag: str):
        ttl = RETENTION.get(tag, RETENTION["DEFAULT"])
        enc = self.cipher.encrypt(plaintext)
        with self._lock:
            self.store[payload_id] = {"data": enc, "tag": tag, "expires": time.time() + ttl}
        self.bridge.emit("purge", f"store {payload_id} tag={tag} ttl={ttl}s", ["store"])

    def get(self, payload_id: str, decrypt: bool = False) -> Optional[str]:
        with self._lock:
            rec = self.store.get(payload_id)
        if not rec:
            return None
        if decrypt:
            return MirrorCipher().decrypt(rec["data"])
        return "<masked>"

    def _reaper(self):
        while self.running:
            now = time.time()
            expired = []
            with self._lock:
                for k, v in list(self.store.items()):
                    if v["expires"] <= now:
                        expired.append(k)
                for k in expired:
                    self.store.pop(k, None)
            for k in expired:
                self.bridge.emit("purge", f"destroy {k}", ["self_destruct"])
            time.sleep(0.8)

    def stop(self):
        self.running = False

# ---------------------- Persistent Memory -------------------
class MemoryManager:
    """
    Tracks mesh + sensitive buffer references:
    - events, visited/built/enforced
    - web_domains, local_hosts/services
    - codex_rules
    - telemetry retention (ghost mode adjusts)
    - JSON persistence optional
    - sensitive_index: tag -> [payload_ids]
    """
    def __init__(self, bridge: Bridge, purge: PurgeManager):
        self.bridge = bridge
        self.purge = purge
        self._lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self.visited_urls: Set[str] = set()
        self.built_urls: Set[str] = set()
        self.enforced_urls: Set[str] = set()
        self.web_domains: Set[str] = set(ALLOWED_DOMAINS)
        self.local_hosts: Set[str] = set(LOCAL_HOST_SEEDS)
        self.local_services: Set[str] = set()
        self.codex_rules: List[str] = []
        self._last_flush = time.time()
        self._flush_interval = BORG_MESH_CONFIG["memory_flush_interval"]
        self._ghost_until = 0.0
        self.sensitive_index: Dict[str, List[str]] = {
            "PERSONAL_DATA": [],
            "NETWORK_IDENTIFIERS": [],
            "FAKE_TELEMETRY": [],
            "SYSTEM_INFO": []
        }

    def record_mesh_event(self, evt: Dict[str, Any]):
        with self._lock:
            self.events.append(evt)

    def mark_visited(self, url: str):
        with self._lock:
            self.visited_urls.add(url)

    def mark_built(self, url: str):
        with self._lock:
            self.built_urls.add(url)

    def mark_enforced(self, url: str):
        with self._lock:
            self.enforced_urls.add(url)

    def add_domain(self, domain: str):
        with self._lock:
            if domain not in self.web_domains:
                self.web_domains.add(domain)
                self.bridge.emit("net", f"domain added {domain}", ["domain"])

    def add_local_host(self, host: str):
        with self._lock:
            if host not in self.local_hosts:
                self.local_hosts.add(host)
                self.bridge.emit("local", f"host added {host}", ["host"])

    def add_local_service(self, url: str):
        with self._lock:
            if url not in self.local_services:
                self.local_services.add(url)
                self.bridge.emit("local", f"service added {url}", ["service"])

    def add_codex_rule(self, rule: str):
        with self._lock:
            self.codex_rules.append(rule)
        self.bridge.emit("codex", f"rule added: {rule}", ["codex"])

    def register_sensitive(self, payload_id: str, tag: str):
        with self._lock:
            self.sensitive_index.setdefault(tag, []).append(payload_id)
        self.bridge.emit("purge", f"index {payload_id} -> {tag}", ["index"])

    def set_telemetry_retention(self, seconds: int, ghost_mode: bool = False):
        with self._lock:
            self._flush_interval = max(1, seconds)
            if ghost_mode:
                self._ghost_until = time.time() + max(1, seconds) * 3
        mode = "ghost" if ghost_mode else "normal"
        self.bridge.emit("memory", f"telemetry retention -> {seconds}s ({mode})", ["retention"])

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return self.events[-limit:]

    def checkpoint(self):
        if time.time() > self._ghost_until and self._flush_interval != BORG_MESH_CONFIG["memory_flush_interval"]:
            self.set_telemetry_retention(BORG_MESH_CONFIG["memory_flush_interval"], ghost_mode=False)
        if time.time() - self._last_flush >= self._flush_interval:
            with self._lock:
                summary = {
                    "events": len(self.events),
                    "visited": len(self.visited_urls),
                    "built": len(self.built_urls),
                    "enforced": len(self.enforced_urls),
                    "domains": len(self.web_domains),
                    "local_hosts": len(self.local_hosts),
                    "local_services": len(self.local_services),
                    "codex_rules": len(self.codex_rules),
                    "sensitive_buffers": {k: len(v) for k, v in self.sensitive_index.items()}
                }
            self.bridge.emit("memory", f"checkpoint {summary}", ["checkpoint"])
            self._last_flush = time.time()
            if PERSISTENCE["enabled"] and PERSISTENCE.get("auto_save_on_checkpoint", False):
                try:
                    self.save_to_file(PERSISTENCE["path"])
                    self.bridge.emit("memory", f"autosaved -> {PERSISTENCE['path']}", ["persist"])
                except Exception as e:
                    self.bridge.emit("memory", f"autosave failed: {e}", ["persist_error"])

    # ---------------------- JSON Persistence ----------------
    def save_to_file(self, filename: str):
        with self._lock:
            data = {
                "events": self.events,
                "visited_urls": list(self.visited_urls),
                "built_urls": list(self.built_urls),
                "enforced_urls": list(self.enforced_urls),
                "web_domains": list(self.web_domains),
                "local_hosts": list(self.local_hosts),
                "local_services": list(self.local_services),
                "codex_rules": self.codex_rules,
                "_flush_interval": self._flush_interval
            }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            with self._lock:
                self.events = data.get("events", [])
                self.visited_urls = set(data.get("visited_urls", []))
                self.built_urls = set(data.get("built_urls", []))
                self.enforced_urls = set(data.get("enforced_urls", []))
                self.web_domains = set(data.get("web_domains", list(ALLOWED_DOMAINS)))
                self.local_hosts = set(data.get("local_hosts", list(LOCAL_HOST_SEEDS)))
                self.local_services = set(data.get("local_services", []))
                self.codex_rules = data.get("codex_rules", [])
                self._flush_interval = int(data.get("_flush_interval", BORG_MESH_CONFIG["memory_flush_interval"]))
            self.bridge.emit("memory", f"loaded from {filename}", ["persist"])
        except FileNotFoundError:
            self.bridge.emit("memory", f"no persistence file at {filename} (starting fresh)", ["persist"])
        except Exception as e:
            self.bridge.emit("memory", f"load failed: {e}", ["persist_error"])

# ---------------------- SecurityGuardian --------------------
class SecurityGuardian:
    def disassemble(self, snippet: str) -> Dict[str, Any]:
        entropy = min(1.0, (len(set(snippet)) / 50.0) if snippet else 0.1)
        flags = []
        lower = snippet.lower()
        if any(k in lower for k in ["password", "ssn", "secret", "token", "apikey", "license", "address", "phone"]):
            flags.append("pii_like")
        if "http" in lower:
            flags.append("url")
        return {"entropy": entropy, "pattern_flags": flags}

    def _pii_count(self, snippet: str) -> int:
        lower = snippet.lower()
        tokens = ["password", "ssn", "secret", "token", "apikey", "license", "address", "phone"]
        return sum(lower.count(t) for t in tokens)

    def reassemble(self, url: str, sanitized: str, raw_pii_hits: int = 0) -> Dict[str, Any]:
        status = "SAFE_FOR_TRAVEL" if raw_pii_hits == 0 else "HOSTILE"
        return {"url": url, "status": status, "pii_hits": raw_pii_hits}

def privacy_filter(text: str):
    sanitized = (text or "")
    for token in ["password", "secret", "apikey", "token", "ssn", "license", "address", "phone"]:
        sanitized = sanitized.replace(token, "***")
    return sanitized, {"sanitized": True}

# ---------------------- Event Type --------------------------
@dataclass
class MeshEvent:
    url: str
    snippet: str
    links: List[str]

# ============================================================
# Borg Mesh — overlay
# ============================================================
class BorgMesh:
    def __init__(self, memory: MemoryManager, comms: BorgCommsRouter, guardian: SecurityGuardian):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]
        self._lock = threading.Lock()

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: List[str]):
        with self._lock:
            if url in self.nodes and self.nodes[url]["state"] in ("discovered", "built", "enforced"):
                for l in links[:20]:
                    if len(self.edges) >= self.max_corridors:
                        break
                    self.edges.add((url, l))
                return
        risk = self._risk(snippet)
        with self._lock:
            node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
            node["state"] = "discovered"
            node["risk"] = risk
            node["seen"] += 1
            self.nodes[url] = node
            for l in links[:20]:
                if len(self.edges) >= self.max_corridors:
                    break
                self.edges.add((url, l))
        self.memory.mark_visited(url)
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "discover", "url": url, "risk": risk, "links": len(links)}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:discover", f"{url} risk={risk} links={len(links)}", "Default")

    def build(self, url: str) -> bool:
        with self._lock:
            if url not in self.nodes:
                return False
            if self.nodes[url]["state"] in ("built", "enforced"):
                return True
            self.nodes[url]["state"] = "built"
        self.memory.mark_built(url)
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "type": "build", "url": url}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        return True

    def enforce(self, url: str, snippet: str) -> bool:
        with self._lock:
            if url not in self.nodes:
                return False
        verdict = self.guardian.reassemble(
            url,
            privacy_filter(snippet or "")[0],
            raw_pii_hits=self.guardian._pii_count(snippet or "")
        )
        status = verdict.get("status", "HOSTILE")
        with self._lock:
            self.nodes[url]["state"] = "enforced"
            self.nodes[url]["risk"] = 0 if status == "SAFE_FOR_TRAVEL" else max(50, self.nodes[url]["risk"])
        self.memory.mark_enforced(url)
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "type": "enforce", "url": url, "status": status}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:enforce", f"{url} status={status}", "Default")
        return True

    def add_phantom_node(self, label: str):
        with self._lock:
            url = f"phantom://{label}"
            self.nodes[url] = {"state": "discovered", "risk": 0, "seen": 1, "phantom": True}
        self.memory.record_mesh_event({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "phantom", "url": url, "status": "virtual"
        })
        self.comms.send_secure("mesh:phantom", f"{url} spawned", "Default")

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self.nodes)
            discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
            built = sum(1 for n in self.nodes.values() if n["state"] == "built")
            enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
            corridors = len(self.edges)
        return {"total": total, "discovered": discovered, "built": built, "enforced": enforced, "corridors": corridors}

# ============================================================
# Borg Roles — scanners, builders, workers, enforcers
# ============================================================
class BorgScanner(threading.Thread):
    def __init__(self, mesh: BorgMesh, in_events: queue.Queue, out_ops: queue.Queue, zt: ZeroTrustGuard, label="SCANNER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.zt = zt
        self.running = True
    def stop(self):
        self.running = False
    def run(self):
        while self.running:
            try:
                ev: MeshEvent = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self.zt.can(self.label, "scan:mesh"):
                time.sleep(0.2)
                continue
            unseen_links = [l for l in ev.links if l not in self.mesh.nodes and random.random() < BORG_MESH_CONFIG["unknown_bias"]]
            self.mesh.discover(ev.url, ev.snippet, unseen_links or ev.links)
            if ev.url not in self.mesh.nodes or self.mesh.nodes[ev.url]["state"] == "discovered":
                if self.zt.can(self.label, "build"):
                    self.out_ops.put(("build", ev.url))
            time.sleep(random.uniform(0.2, 0.6))

class BorgBuilder(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, zt: ZeroTrustGuard, label="BUILDER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.zt = zt
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
            if op == "build" and self.zt.can(self.label, "build"):
                if self.mesh.build(url):
                    if self.zt.can(self.label, "enforce"):
                        self.ops_q.put(("enforce", url))
            else:
                self.ops_q.put((op, url))
            time.sleep(random.uniform(0.2, 0.5))

class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, zt: ZeroTrustGuard, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.zt = zt
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
            if op == "build" and self.zt.can(self.label, "build"):
                if self.mesh.build(url):
                    if self.zt.can(self.label, "enforce"):
                        self.ops_q.put(("enforce", url))
            elif op == "enforce" and self.zt.can(self.label, "enforce", required_scope="patrol"):
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))

class BorgEnforcer(threading.Thread):
    def __init__(self, mesh: BorgMesh, guardian: SecurityGuardian, zt: ZeroTrustGuard, label="ENFORCER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.guardian = guardian
        self.zt = zt
        self.label = label
        self.running = True
    def stop(self):
        self.running = False
    def run(self):
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    if self.zt.can(self.label, "enforce", required_scope="patrol"):
                        self.mesh.enforce(url, snippet="")
            time.sleep(1.2)

# ============================================================
# MirrorDefense + MirrorHook
# ============================================================
class MirrorDefense:
    def __init__(self, bridge: Bridge, responder: Responder, trust: Trust, lists: Dict[str, Any], threshold: int = 50):
        self.bridge = bridge
        self.responder = responder
        self.trust = trust
        self.lists = lists
        self.threshold = threshold
    def evaluate(self, analysis: Dict[str, Any]):
        status = analysis.get("status", "")
        pos = analysis.get("positives", 0)
        neg = analysis.get("negatives", 0)
        voids = analysis.get("voids", 0)
        unity = analysis.get("unity", 0)
        if "oscillating" in status and (pos + neg) > self.threshold:
            self.bridge.emit("mirror", "Oscillation threshold exceeded — auto-quarantine engaged", ["mirror"])
            suspect = os.path.join(os.getcwd(), "suspicious.tmp")
            if os.path.isfile(suspect):
                self.responder.quarantine(suspect)
        if unity > 0:
            self.bridge.emit("mirror", "Synthesis detected — alert admin", ["mirror"])
            self.trust.update("mirror:unity", ["beacon"])
        if status == "positive dominance":
            self.bridge.emit("mirror", "Positive dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:posdom", ["fs"])
        elif status == "negative dominance":
            self.bridge.emit("mirror", "Negative dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:negdom", ["fs"])
        if status == "void equilibrium" and voids > self.threshold // 2:
            self.bridge.emit("mirror", "Void equilibrium — possible covert channel", ["mirror"])

class MirrorHook:
    def __init__(self, defense: MirrorDefense, bridge: Bridge):
        self.defense = defense
        self.bridge = bridge
        self.enabled = True
    def submit(self, analysis: Dict[str, Any]):
        if not self.enabled or not isinstance(analysis, dict):
            return
        self.bridge.emit("mirror", f"analysis status={analysis.get('status','unknown')}", ["trace"])
        try:
            self.defense.evaluate(analysis)
        except Exception as e:
            self.bridge.emit("mirror", f"defense error {e}", ["error"])

# ============================================================
# HTML parsing helpers
# ============================================================
class HTMLLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: List[str] = []
        self._title: str = ""
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for (k, v) in attrs:
                if k.lower() == "href" and v:
                    try:
                        url = urllib.parse.urljoin(self.base_url, v)
                        self.links.append(url)
                    except Exception:
                        pass
    def handle_data(self, data):
        if len(self._title) < 120:
            self._title += data.strip()[:120-len(self._title)]
    @property
    def title(self):
        return self._title.strip()

def parse_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        domain = (parsed.hostname or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

# ============================================================
# External web fetcher (polite)
# ============================================================
def fetch_page(url: str) -> Tuple[str, List[str]]:
    req = urllib.request.Request(url, headers={"User-Agent": BORG_MESH_CONFIG["user_agent"]})
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=BORG_MESH_CONFIG["crawl_timeout"], context=context) as resp:
            raw = resp.read(BORG_MESH_CONFIG["max_body_bytes"])
            body = raw.decode("utf-8", errors="ignore")
            parser = HTMLLinkParser(url); parser.feed(body)
            snippet = parser.title or body[:180]
            clean_links = []
            for l in parser.links:
                p = urllib.parse.urlparse(l)
                if p.scheme in ("http", "https") and parse_domain(l):
                    clean_links.append(l)
            return snippet, clean_links
    except Exception as e:
        return f"error fetching: {e}", []

# ============================================================
# Local fetcher (same-host links)
# ============================================================
def fetch_local(url: str) -> Tuple[str, List[str]]:
    req = urllib.request.Request(url, headers={"User-Agent": BORG_MESH_CONFIG["user_agent"]})
    try:
        with urllib.request.urlopen(req, timeout=BORG_MESH_CONFIG["crawl_timeout"]) as resp:
            raw = resp.read(BORG_MESH_CONFIG["max_body_bytes"])
            body = raw.decode("utf-8", errors="ignore")
            parser = HTMLLinkParser(url); parser.feed(body)
            snippet = parser.title or body[:180]
            clean_links = []
            base_host = parse_domain(url)
            for l in parser.links:
                if urllib.parse.urlparse(l).scheme in ("http", "https"):
                    if parse_domain(l) == base_host:
                        clean_links.append(l)
            return snippet, clean_links
    except Exception as e:
        return f"error fetching: {e}", []

# ============================================================
# Network scanners
# ============================================================
class NetworkScanner(threading.Thread):
    def __init__(self, memory: MemoryManager, in_events: queue.Queue, bridge: Bridge, zt: ZeroTrustGuard, purge: PurgeManager, label="NET_SCANNER"):
        super().__init__(daemon=True)
        self.memory = memory
        self.in_events = in_events
        self.bridge = bridge
        self.zt = zt
        self.purge = purge
        self.label = label
        self.running = True
        self.frontier: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()

    def add_seed(self, url: str):
        p = urllib.parse.urlparse(url); d = parse_domain(url)
        if p.scheme in ("http", "https") and d:
            self.frontier.put(url)
            self.memory.add_domain(d)
            nid = f"netid:{d}:{int(time.time())}"
            self.purge.add(nid, f"domain={d}", "NETWORK_IDENTIFIERS")
            self.memory.register_sensitive(nid, "NETWORK_IDENTIFIERS")
            self.bridge.emit("net", f"seed added {url}", ["seed"])
        else:
            self.bridge.emit("net", f"seed rejected {url}", ["reject"])

    def stop(self):
        self.running = False

    def run(self):
        for d in list(self.memory.web_domains)[:6]:
            self.frontier.put(f"https://{d}/")
        while self.running:
            try:
                url = self.frontier.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.2)
                continue
            if not self.zt.can(self.label, "scan:web"):
                time.sleep(0.2)
                continue
            with self._lock:
                if url in self.memory.visited_urls:
                    continue
            snippet, links = fetch_page(url)
            time.sleep(BORG_MESH_CONFIG["crawl_delay"])
            self.memory.mark_visited(url)
            self.in_events.put(MeshEvent(url=url, snippet=snippet, links=links))
            for l in links[:20]:
                d = parse_domain(l)
                if not d:
                    continue
                self.memory.add_domain(d)
                with self._lock:
                    if l not in self.memory.visited_urls:
                        self.frontier.put(l)
            self.memory.checkpoint()

class LocalNetworkScanner(threading.Thread):
    def __init__(self, memory: MemoryManager, in_events: queue.Queue, bridge: Bridge, zt: ZeroTrustGuard, purge: PurgeManager, label="LOCAL_SCANNER"):
        super().__init__(daemon=True)
        self.memory = memory
        self.in_events = in_events
        self.bridge = bridge
        self.zt = zt
        self.purge = purge
        self.label = label
        self.running = True
        self.frontier: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self.port_list = list(range(1, 65536)) if LOCAL_PORTS is None else list(LOCAL_PORTS)

    def add_host(self, host: str):
        host = host.strip()
        if host:
            self.memory.add_local_host(host)
            self.frontier.put(host)
            nid = f"netid:{host}:{int(time.time())}"
            self.purge.add(nid, f"host={host}", "NETWORK_IDENTIFIERS")
            self.memory.register_sensitive(nid, "NETWORK_IDENTIFIERS")
            self.bridge.emit("local", f"host seeded {host}", ["seed"])

    def stop(self):
        self.running = False

    def _probe_port(self, host: str, port: int, results: List[str]):
        if not self.zt.can(self.label, "scan:local"):
            return
        try:
            with socket.create_connection((host, port), timeout=2.0):
                url = f"http://{host}:{port}/"
                snippet, links = fetch_local(url)
                if not snippet.startswith("error fetching"):
                    results.append(url)
                    time.sleep(0.05)
                    self.memory.mark_visited(url)
                    self.memory.add_local_service(url)
                    self.in_events.put(MeshEvent(url=url, snippet=snippet, links=links))
        except Exception:
            pass

    def run(self):
        for h in list(self.memory.local_hosts):
            self.frontier.put(h)
        while self.running:
            try:
                host = self.frontier.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.2)
                continue
            results: List[str] = []
            threads: List[threading.Thread] = []
            sem = threading.Semaphore(LOCAL_MAX_CONCURRENT_PROBES)
            def worker(port):
                with sem:
                    self._probe_port(host, port, results)
            for port in self.port_list:
                if not self.running:
                    break
                t = threading.Thread(target=worker, args=(port,), daemon=True)
                threads.append(t); t.start(); time.sleep(0.002)
            for t in threads:
                t.join(timeout=0.5)
            for service_url in results:
                snippet, links = fetch_local(service_url)
                time.sleep(BORG_MESH_CONFIG["crawl_delay"])
                self.memory.mark_visited(service_url)
                self.in_events.put(MeshEvent(url=service_url, snippet=snippet, links=links))
                base_host = parse_domain(service_url)
                for l in links[:20]:
                    if parse_domain(l) == base_host and l not in self.memory.visited_urls:
                        self.memory.mark_visited(l)
                        self.in_events.put(MeshEvent(url=l, snippet="local link", links=[]))
            self.memory.checkpoint()

# ============================================================
# Adaptive Codex Mutation — ghost sync
# ============================================================
class SymbolicLanguage:
    def __init__(self):
        self.grammar: Dict[str, Any] = {"rules": []}
    def evolve(self, token: str):
        rule = f"phantom_node::{token}"
        self.grammar["rules"].append(rule)
        return rule

class SyncAPI:
    def __init__(self, memory: MemoryManager, bridge: Bridge):
        self.memory = memory; self.bridge = bridge
    def propagate_grammar(self, grammar: Dict[str, Any]):
        for r in grammar.get("rules", []):
            self.memory.add_codex_rule(r)
        self.bridge.emit("codex", f"grammar propagated ({len(grammar.get('rules', []))} rules)", ["sync"])

class AdaptiveCodexDaemon(threading.Thread):
    def __init__(self, bridge: Bridge, memory: MemoryManager, mesh: BorgMesh, purge: PurgeManager, label="CODEX_DAEMON"):
        super().__init__(daemon=True)
        self.bridge = bridge; self.memory = memory; self.mesh = mesh; self.purge = purge
        self.label = label; self.running = True
        self.language = SymbolicLanguage(); self.sync_api = SyncAPI(memory, bridge)
        self._token_seq = 0
    def stop(self): self.running = False
    def run(self):
        while self.running:
            mirror_msgs = self.bridge.get_recent(channel="mirror", limit=10)
            ghost_detected = any(("Oscillation threshold" in m["message"]) or ("oscillating" in m["message"].lower()) for m in mirror_msgs)
            if ghost_detected:
                self.memory.set_telemetry_retention(BORG_MESH_CONFIG["ghost_retention"], ghost_mode=True)
                label = f"ghost-{datetime.datetime.now().strftime('%H%M%S')}"
                self.mesh.add_phantom_node(label)
                self._token_seq += 1
                new_rule = self.language.evolve(f"{label}-{self._token_seq}")
                self.memory.add_codex_rule(new_rule)
                self.sync_api.propagate_grammar(self.language.grammar)
                decoy_id = f"fake:{int(time.time())}"
                self.purge.add(decoy_id, "decoy_telemetry=yes", "FAKE_TELEMETRY")
                self.memory.register_sensitive(decoy_id, "FAKE_TELEMETRY")
                self.bridge.emit("codex", "ghost sync: retention short, phantom, grammar, decoy emitted", ["ghost"])
            self._token_seq += 1
            token = f"pulse-{self._token_seq}"
            new_rule = self.language.evolve(token)
            self.memory.add_codex_rule(new_rule)
            self.sync_api.propagate_grammar(self.language.grammar)
            time.sleep(10)

# ============================================================
# Live GUI — chameleon masking for sensitive data
# ============================================================
class MeshGUI(tk.Tk):
    def __init__(self, mesh: BorgMesh, memory: MemoryManager, bridge: Bridge,
                 hook: MirrorHook = None,
                 net_scanner: NetworkScanner = None,
                 local_scanner: LocalNetworkScanner = None,
                 codex_daemon: AdaptiveCodexDaemon = None,
                 purge: PurgeManager = None,
                 in_events: queue.Queue = None,
                 out_ops: queue.Queue = None,
                 guardian: SecurityGuardian = None):
        super().__init__()
        self.mesh = mesh; self.memory = memory; self.bridge = bridge
        self.hook = hook; self.net_scanner = net_scanner; self.local_scanner = local_scanner
        self.codex_daemon = codex_daemon; self.purge = purge
        self.in_events = in_events; self.out_ops = out_ops; self.guardian = guardian

        self.title("Borg Mesh Codex Dashboard")
        self.geometry("1400x980")

        # Mesh stats
        self.stats_frame = ttk.LabelFrame(self, text="Mesh Stats"); self.stats_frame.pack(fill="x", padx=10, pady=5)
        self.stats_label = ttk.Label(self.stats_frame, text="", font=("Consolas", 11)); self.stats_label.pack(padx=5, pady=5)

        # Memory summary
        self.mem_frame = ttk.LabelFrame(self, text="Memory Summary"); self.mem_frame.pack(fill="x", padx=10, pady=5)
        self.mem_label = ttk.Label(self.mem_frame, text="", font=("Consolas", 10)); self.mem_label.pack(padx=5, pady=5)

        # Events
        self.events_frame = ttk.LabelFrame(self, text="Recent Mesh Events"); self.events_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.events_box = tk.Listbox(self.events_frame, font=("Consolas", 10)); self.events_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Mirror
        self.alerts_frame = ttk.LabelFrame(self, text="MirrorDefense Alerts"); self.alerts_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.alerts_box = tk.Listbox(self.alerts_frame, font=("Consolas", 10)); self.alerts_box.pack(fill="both", expand=True, padx=5, pady=5)

        # External web scanner
        self.net_frame = ttk.LabelFrame(self, text="External Web Scanner (no domain cap)"); self.net_frame.pack(fill="x", padx=10, pady=5)
        self.net_seed_entry = ttk.Entry(self.net_frame, width=60); self.net_seed_entry.pack(side="left", padx=5, pady=5)
        self.net_seed_entry.insert(0, "https://example.com")
        self.net_seed_btn = ttk.Button(self.net_frame, text="Add web seed", command=self._add_web_seed); self.net_seed_btn.pack(side="left", padx=5, pady=5)
        self.net_info_label = ttk.Label(self.net_frame, text=f"Domains auto-expand (no cap)", font=("Consolas", 9)); self.net_info_label.pack(side="left", padx=10)

        # Local Borg internet
        self.local_frame = ttk.LabelFrame(self, text="Local Borg Internet (dynamic ports)"); self.local_frame.pack(fill="x", padx=10, pady=5)
        self.local_seed_entry = ttk.Entry(self.local_frame, width=40); self.local_seed_entry.pack(side="left", padx=5, pady=5)
        self.local_seed_entry.insert(0, "localhost")
        self.local_seed_btn = ttk.Button(self.local_frame, text="Add host", command=self._add_local_host); self.local_seed_btn.pack(side="left", padx=5, pady=5)
        port_info = "ports: all (1..65535)" if LOCAL_PORTS is None else f"ports: {sorted(set(LOCAL_PORTS))[:10]}..."
        self.local_info_label = ttk.Label(self.local_frame, text=f"{port_info} | concurrency={LOCAL_MAX_CONCURRENT_PROBES}", font=("Consolas", 9)); self.local_info_label.pack(side="left", padx=10)

        # Queen board
        self.queen_frame = ttk.LabelFrame(self, text="Queen Board — Agent Management"); self.queen_frame.pack(fill="x", padx=10, pady=5)
        self.add_scanner_btn = ttk.Button(self.queen_frame, text="Add Scanner", command=self._add_scanner); self.add_scanner_btn.pack(side="left", padx=5, pady=5)
        self.add_builder_btn = ttk.Button(self.queen_frame, text="Add Builder", command=self._add_builder); self.add_builder_btn.pack(side="left", padx=5, pady=5)
        self.add_worker_btn = ttk.Button(self.queen_frame, text="Add Worker", command=self._add_worker); self.add_worker_btn.pack(side="left", padx=5, pady=5)
        self.add_enforcer_btn = ttk.Button(self.queen_frame, text="Add Enforcer", command=self._add_enforcer); self.add_enforcer_btn.pack(side="left", padx=5, pady=5)
        self.retire_btn = ttk.Button(self.queen_frame, text="Retire selected", command=self._retire_selected); self.retire_btn.pack(side="left", padx=5, pady=5)
        self.agents_list = tk.Listbox(self.queen_frame, height=6, width=80, font=("Consolas", 10)); self.agents_list.pack(side="left", padx=10, pady=5)

        # Codex
        self.codex_frame = ttk.LabelFrame(self, text="Adaptive Codex Mutation"); self.codex_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.codex_box = tk.Listbox(self.codex_frame, font=("Consolas", 10)); self.codex_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Sensitive buffers (masked view)
        self.sens_frame = ttk.LabelFrame(self, text="Sensitive Buffers (Chameleon-masked)"); self.sens_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.sens_box = tk.Listbox(self.sens_frame, font=("Consolas", 10)); self.sens_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Controls
        self.ctrl_frame = ttk.LabelFrame(self, text="Controls"); self.ctrl_frame.pack(fill="x", padx=10, pady=5)
        self.seed_btn = ttk.Button(self.ctrl_frame, text="Mirror burst", command=self._seed_burst); self.seed_btn.pack(side="left", padx=5, pady=5)
        if PERSISTENCE["enabled"]:
            self.save_btn = ttk.Button(self.ctrl_frame, text="Save memory to JSON", command=self._save_memory); self.save_btn.pack(side="left", padx=5, pady=5)

        self.after(700, self.refresh)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._stop_callables: List[Any] = []
        self._agents: List[Dict[str, Any]] = []

    def attach_stoppers(self, stoppers: List[Any]):
        self._stop_callables = stoppers

    def _add_web_seed(self):
        url = self.net_seed_entry.get().strip()
        if self.net_scanner: self.net_scanner.add_seed(url)

    def _add_local_host(self):
        host = self.local_seed_entry.get().strip()
        if self.local_scanner: self.local_scanner.add_host(host)

    def _save_memory(self):
        try:
            self.memory.save_to_file(PERSISTENCE["path"])
            self.bridge.emit("memory", f"manual save -> {PERSISTENCE['path']}", ["persist"])
        except Exception as e:
            self.bridge.emit("memory", f"manual save failed: {e}", ["persist_error"])

    def _seed_burst(self):
        if self.hook:
            self.hook.submit({"status": "oscillating", "positives": 5, "negatives": 4, "voids": 0, "unity": 0, "entropy": "high"})
            self.hook.submit({"status": "positive dominance", "positives": 8, "negatives": 1, "voids": 0, "unity": 0, "entropy": "low"})
            self.hook.submit({"status": "void equilibrium", "positives": 0, "negatives": 0, "voids": 6, "unity": 0, "entropy": "low"})
            self.hook.submit({"status": "stable", "positives": 1, "negatives": 1, "voids": 0, "unity": 1, "entropy": "low"})

    def _register_agent(self, agent_type: str, obj, stop_callable, label: str):
        self._agents.append({"type": agent_type, "obj": obj, "stop": stop_callable, "label": label})
        self.agents_list.insert(tk.END, f"{agent_type} :: {label}")
        self.bridge.emit("queen", f"added {label}", ["agent"])

    def _add_scanner(self):
        s = BorgScanner(self.mesh, self.in_events, self.out_ops, self.zt, label=f"SCANNER-{len(self._agents)+1}")
        s.start(); self._register_agent("Scanner", s, s.stop, s.label)

    def _add_builder(self):
        b = BorgBuilder(self.mesh, self.out_ops, self.zt, label=f"BUILDER-{len(self._agents)+1}")
        b.start(); self._register_agent("Builder", b, b.stop, b.label)

    def _add_worker(self):
        w = BorgWorker(self.mesh, self.out_ops, self.zt, label=f"WORKER-{len(self._agents)+1}")
        w.start(); self._register_agent("Worker", w, w.stop, w.label)

    def _add_enforcer(self):
        e = BorgEnforcer(self.mesh, self.guardian, self.zt, label=f"ENFORCER-{len(self._agents)+1}")
        e.start(); self._register_agent("Enforcer", e, e.stop, e.label)

    def _retire_selected(self):
        idx = self.agents_list.curselection()
        if not idx: return
        i = idx[0]; agent = self._agents[i]
        try: agent["stop"]()
        except Exception: pass
        self.agents_list.delete(i); self._agents.pop(i)
        self.bridge.emit("queen", f"retired {agent['label']}", ["agent"])

    def refresh(self):
        stats = self.mesh.stats()
        self.stats_label.config(text=(f"Total: {stats['total']} | Discovered: {stats['discovered']} | "
                                      f"Built: {stats['built']} | Enforced: {stats['enforced']} | Corridors: {stats['corridors']}"))
        mem_text = (f"Visited: {len(self.memory.visited_urls)} | Built: {len(self.memory.built_urls)} | "
                    f"Enforced: {len(self.memory.enforced_urls)} | Web domains: {len(self.memory.web_domains)} | "
                    f"Local hosts: {len(self.memory.local_hosts)} | Local services: {len(self.memory.local_services)} | "
                    f"Codex rules: {len(self.memory.codex_rules)}")
        self.mem_label.config(text=mem_text)

        self.events_box.delete(0, tk.END)
        for evt in self.memory.recent_events(limit=60):
            self.events_box.insert(tk.END, f"{evt['time']} :: {evt['type']} :: {evt['url']} :: status={evt.get('status','')} :: risk={evt.get('risk','')}")

        self.alerts_box.delete(0, tk.END)
        for msg in self.bridge.get_recent(channel="mirror", limit=60):
            self.alerts_box.insert(tk.END, f"{msg['time']} :: {' '.join(msg['tags'])} :: {msg['message']}")

        self.codex_box.delete(0, tk.END)
        for rule in self.memory.codex_rules[-60:]:
            self.codex_box.insert(tk.END, rule)

        # Chameleon-masked sensitive buffers
        self.sens_box.delete(0, tk.END)
        for tag, ids in self.memory.sensitive_index.items():
            for pid in ids[-40:]:
                self.sens_box.insert(tk.END, f"{tag} :: {pid} :: <masked> :: ttl<= {RETENTION.get(tag, RETENTION['DEFAULT'])}s")

        self.after(900, self.refresh)

    def _on_close(self):
        for stop in self._stop_callables:
            try: stop()
            except Exception: pass
        for a in self._agents:
            try: a["stop"]()
            except Exception: pass
        if PERSISTENCE["enabled"]:
            try:
                self.memory.save_to_file(PERSISTENCE["path"])
                self.bridge.emit("memory", f"saved on exit -> {PERSISTENCE['path']}", ["persist"])
            except Exception as e:
                self.bridge.emit("memory", f"save on exit failed: {e}", ["persist_error"])
        time.sleep(0.3)
        self.destroy()

    def attach_zero_trust(self, zt: ZeroTrustGuard):
        self.zt = zt

# ============================================================
# Background feeders + MirrorTicker
# ============================================================
class MeshFeeder(threading.Thread):
    def __init__(self, in_events: queue.Queue, label="FEEDER"):
        super().__init__(daemon=True)
        self.in_events = in_events; self.label = label; self.running = True
        self._domains = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "tau", "lambda"]
    def stop(self): self.running = False
    def run(self):
        while self.running:
            base = random.choice(self._domains)
            url = f"https://{base}.example/{random.randint(1, 999)}"
            links = [f"https://{random.choice(self._domains)}.example/{random.randint(1, 999)}" for _ in range(random.randint(1, 3))]
            snippet = random.choice(["public page","contains secret token","docs http links","user guide","apikey embedded","password reset page"])
            self.in_events.put(MeshEvent(url=url, snippet=snippet, links=links))
            time.sleep(random.uniform(0.8, 1.6))

class MirrorTicker(threading.Thread):
    def __init__(self, hook: MirrorHook, label="MIRROR_TICK"):
        super().__init__(daemon=True)
        self.hook = hook; self.label = label; self.running = True
    def stop(self): self.running = False
    def run(self):
        statuses = [("oscillating",4,4,0,0,"high"),("positive dominance",6,0,0,0,"low"),
                    ("negative dominance",0,6,0,0,"low"),("void equilibrium",0,0,6,0,"low"),("stable",1,1,0,1,"low")]
        while self.running:
            s = random.choice(statuses)
            self.hook.submit({"status": s[0], "positives": s[1], "negatives": s[2], "voids": s[3], "unity": s[4], "entropy": s[5]})
            time.sleep(random.uniform(1.8, 3.2))

# ============================================================
# Main — wire everything and launch the live GUI
# ============================================================
def main():
    bridge = Bridge()
    purge = PurgeManager(bridge)
    memory = MemoryManager(bridge, purge)

    if PERSISTENCE["enabled"]:
        memory.load_from_file(PERSISTENCE["path"])

    comms = BorgCommsRouter(bridge)
    guardian = SecurityGuardian()
    zt = ZeroTrustGuard(bridge)

    # Grant minimal ops explicitly (deny-by-default)
    for agent in ["SCANNER-BASE","WORKER-BASE","ENFORCER-BASE","LOCAL_SCANNER","NET_SCANNER","CODEX_DAEMON"]:
        zt.allow(agent, "scan:mesh")
        zt.allow(agent, "build")
        zt.allow(agent, "enforce")
        zt.grant_scope(agent, "patrol")
    zt.allow("NET_SCANNER", "scan:web")
    zt.allow("LOCAL_SCANNER", "scan:local")

    mesh = BorgMesh(memory, comms, guardian)

    in_events: queue.Queue = queue.Queue()
    out_ops: queue.Queue = queue.Queue()

    # Base agents
    scanner = BorgScanner(mesh, in_events, out_ops, zt, label="SCANNER-BASE")
    worker = BorgWorker(mesh, out_ops, zt, label="WORKER-BASE")
    enforcer = BorgEnforcer(mesh, guardian, zt, label="ENFORCER-BASE")

    responder = Responder()
    trust = Trust(bridge)
    lists = {"allow": [], "deny": []}
    defense = MirrorDefense(bridge, responder, trust, lists, threshold=5)
    hook = MirrorHook(defense, bridge)

    feeder = MeshFeeder(in_events, label="FEEDER")
    ticker = MirrorTicker(hook, label="MIRROR_TICK")

    net_scanner = NetworkScanner(memory, in_events, bridge, zt, purge, label="NET_SCANNER")
    local_scanner = LocalNetworkScanner(memory, in_events, bridge, zt, purge, label="LOCAL_SCANNER")

    codex_daemon = AdaptiveCodexDaemon(bridge, memory, mesh, purge, label="CODEX_DAEMON")

    # Start threads
    scanner.start(); worker.start(); enforcer.start()
    feeder.start(); ticker.start()
    net_scanner.start(); local_scanner.start(); codex_daemon.start()

    # GUI
    app = MeshGUI(mesh, memory, bridge, hook, net_scanner, local_scanner, codex_daemon, purge, in_events, out_ops, guardian)
    app.attach_zero_trust(zt)
    app.attach_stoppers([scanner.stop, worker.stop, enforcer.stop, feeder.stop, ticker.stop, net_scanner.stop, local_scanner.stop, codex_daemon.stop])
    app.mainloop()

    # After GUI close, join threads briefly
    scanner.join(timeout=1.0); worker.join(timeout=1.0); enforcer.join(timeout=1.0)
    feeder.join(timeout=1.0); ticker.join(timeout=1.0)
    net_scanner.join(timeout=1.0); local_scanner.join(timeout=1.0); codex_daemon.join(timeout=1.0)
    purge.stop()

if __name__ == "__main__":
    main()

