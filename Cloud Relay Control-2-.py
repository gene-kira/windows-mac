import sys
import threading
import time
import subprocess
import importlib
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random
import json
import os
import socket
import ssl
from urllib.parse import urlparse

# -----------------------------
# AUTOLOADER
# -----------------------------

REQUIRED_LIBS = [
    "PyQt5",
    "websockets",
    "requests",
    "uiautomation",
]

def ensure_libs():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

from PyQt5 import QtWidgets, QtCore, QtGui
import requests
import asyncio
import websockets
import uiautomation as auto

try:
    import dns.resolver
    HAVE_DNS = True
except Exception:
    HAVE_DNS = False

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------

class Protocol(Enum):
    TCP = "TCP"
    WEBSOCKET = "WebSocket"
    HTTP = "HTTP"

class RelayMode(Enum):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    CUSTOM = "CUSTOM"
    AUTO = "AUTO"

class TrustColor(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"

# TLS / AUTH CONFIG
TLS_ENABLED = False          # set True if you have cert/key
TLS_CERT_FILE = "server.crt"
TLS_KEY_FILE = "server.key"

AUTH_REQUIRED = True
AUTH_TOKEN = "secret-token-123"   # shared token for demo

# Ports
RELAY_HOST = "0.0.0.0"
RELAY_PORT = 9000
WS_RELAY_PORT = 8765
HTTP_RELAY_PORT = 5000

# Default local endpoints
LOCAL_TCP_HOST = "127.0.0.1"
LOCAL_TCP_PORT = RELAY_PORT
LOCAL_WS_URL = f"ws://127.0.0.1:{WS_RELAY_PORT}/relay"
LOCAL_HTTP_URL = f"http://127.0.0.1:{HTTP_RELAY_PORT}/relay"

# Example “cloud” endpoints (pretend these are remote; still loopback for demo)
CLOUD_TCP_HOST = "127.0.0.1"
CLOUD_TCP_PORT = RELAY_PORT
CLOUD_WS_URL = f"ws://127.0.0.1:{WS_RELAY_PORT}/cloud"
CLOUD_HTTP_URL = f"http://127.0.0.1:{HTTP_RELAY_PORT}/cloud"

# Internet candidates (multi‑relay load balancing pool)
INTERNET_CANDIDATES = [
    (f"tcp://127.0.0.1:{RELAY_PORT}", Protocol.TCP),
    (LOCAL_WS_URL, Protocol.WEBSOCKET),
    (LOCAL_HTTP_URL, Protocol.HTTP),
]

CONGESTION_MED = 80.0
CONGESTION_HIGH = 200.0

RELAYS_DB_FILE = "relays.json"

CLUSTER_ID = "default-cluster"
NODE_ID = f"node-{socket.gethostname()}"
GOSSIP_PORT = 49123
GOSSIP_INTERVAL = 15

DEFAULT_APP_PREFS = {
    "Chrome": {"mode": "CLOUD", "protocol": "HTTP"},
    "Edge": {"mode": "CLOUD", "protocol": "HTTP"},
    "Steam": {"mode": "AUTO", "protocol": "TCP"},
    "Game": {"mode": "AUTO", "protocol": "TCP"},
}

DEFAULT_AUTO_REOPT = {
    "enabled": True,
    "interval_minutes": 5,
    "min_score_improvement": 10
}

DEFAULT_DISCOVERY_CONFIG = {
    "dns_srv_domain": "localhost",
    "api_directory_url": "",
}

# -----------------------------
# PERSISTENCE / MEMORY
# -----------------------------

def load_relays_db():
    base = {
        "known_relays": [],
        "last_used": None,
        "app_prefs": DEFAULT_APP_PREFS.copy(),
        "auto_reoptimize": DEFAULT_AUTO_REOPT.copy(),
        "cluster": {
            "cluster_id": CLUSTER_ID,
            "node_id": NODE_ID,
        },
        "discovery": DEFAULT_DISCOVERY_CONFIG.copy(),
    }
    if not os.path.exists(RELAYS_DB_FILE):
        return base
    try:
        with open(RELAYS_DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception:
        return base
    for k, v in base.items():
        if k not in db:
            db[k] = v
    for r in db["known_relays"]:
        r.setdefault("pinned", False)
        r.setdefault("reputation", 50.0)
        r.setdefault("secure", False)
        r.setdefault("auth_token", None)
        r.setdefault("predicted_degrade", False)
        r.setdefault("resurrected", False)
        r.setdefault("last_dead", None)
    if "discovery" not in db:
        db["discovery"] = DEFAULT_DISCOVERY_CONFIG.copy()
    else:
        for k, v in DEFAULT_DISCOVERY_CONFIG.items():
            db["discovery"].setdefault(k, v)
    return db

def save_relays_db(db):
    try:
        with open(RELAYS_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
    except Exception:
        pass

def find_known_relay(db, endpoint, protocol: Protocol):
    for r in db["known_relays"]:
        if r["endpoint"] == endpoint and r["protocol"] == protocol.value:
            return r
    return None

def update_known_relay(db, endpoint, protocol: Protocol, score, trust: str, secure=False, auth_token=None):
    found = find_known_relay(db, endpoint, protocol)
    now = time.time()
    if found is None:
        found = {
            "endpoint": endpoint,
            "protocol": protocol.value,
            "score": score,
            "trust": trust,
            "last_seen": now,
            "success_count": 0,
            "failure_count": 0,
            "pinned": False,
            "reputation": 50.0,
            "secure": bool(secure),
            "auth_token": auth_token,
            "predicted_degrade": False,
            "resurrected": False,
            "last_dead": None,
        }
        db["known_relays"].append(found)
    else:
        if found.get("last_dead") and score > 0:
            found["resurrected"] = True
            found["last_dead"] = None
        found["score"] = score
        found["trust"] = trust
        found["last_seen"] = now
        found.setdefault("pinned", False)
        found.setdefault("reputation", 50.0)
        found.setdefault("secure", bool(secure))
        found.setdefault("auth_token", auth_token)
        found.setdefault("predicted_degrade", False)
        found.setdefault("resurrected", False)
    return db

def record_relay_result(db, endpoint, protocol: Protocol, success: bool):
    r = find_known_relay(db, endpoint, protocol)
    now = time.time()
    if r is None:
        return db
    if success:
        r["success_count"] = r.get("success_count", 0) + 1
    else:
        r["failure_count"] = r.get("failure_count", 0) + 1
        if r["failure_count"] > 3:
            r["last_dead"] = now
            r["predicted_degrade"] = False
    r["last_seen"] = now
    return db

def get_historical_stability(db, endpoint, protocol: Protocol):
    r = find_known_relay(db, endpoint, protocol)
    if not r:
        return 0.0
    succ = r.get("success_count", 0)
    fail = r.get("failure_count", 0)
    total = succ + fail
    if total == 0:
        return 0.0
    return succ / total

# -----------------------------
# STATUS MODEL
# -----------------------------

@dataclass
class RelayStatus:
    mode: RelayMode = RelayMode.LOCAL
    protocol: Protocol = Protocol.TCP
    endpoint: str = f"tcp://{LOCAL_TCP_HOST}:{LOCAL_TCP_PORT}"
    trust: TrustColor = TrustColor.GREEN
    latency_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_contact: str = "-"
    autoloader_status: str = "Ready"
    log_lines: list = field(default_factory=list)

    loss_percent: float = 0.0
    congestion_level: str = "LOW"
    avg_latency_ms: float = 0.0
    _latency_window: deque = field(default_factory=lambda: deque(maxlen=20), repr=False)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 500:
            self.log_lines = self.log_lines[-500:]

    def update_latency_stats(self, latency_ms: float):
        self._latency_window.append(latency_ms)
        if self._latency_window:
            self.avg_latency_ms = sum(self._latency_window) / len(self._latency_window)
        else:
            self.avg_latency_ms = latency_ms

        if self.avg_latency_ms >= CONGESTION_HIGH:
            self.congestion_level = "HIGH"
        elif self.avg_latency_ms >= CONGESTION_MED:
            self.congestion_level = "MEDIUM"
        else:
            self.congestion_level = "LOW"

    def update_loss_stats(self):
        total = self.success_count + self.failure_count
        if total > 0:
            self.loss_percent = (self.failure_count / total) * 100.0
        else:
            self.loss_percent = 0.0

# -----------------------------
# PREDICTION + REPUTATION
# -----------------------------

class PredictionEngine:
    def __init__(self):
        self.history = {}

    def update(self, endpoint, protocol: Protocol, latency, loss):
        key = (endpoint, protocol.value)
        dq = self.history.get(key)
        if dq is None:
            dq = deque(maxlen=30)
            self.history[key] = dq
        dq.append((latency, loss))

    def predict_degrade(self, endpoint, protocol: Protocol):
        key = (endpoint, protocol.value)
        dq = self.history.get(key)
        if not dq or len(dq) < 5:
            return False
        latencies = [x[0] for x in dq]
        losses = [x[1] for x in dq]
        if latencies[-1] > latencies[0] + 50 and losses[-1] > losses[0] + 5:
            return True
        return False

class ReputationManager:
    def update_reputation(self, relay_record, predicted_degrade: bool):
        rep = relay_record.get("reputation", 50.0)
        succ = relay_record.get("success_count", 0)
        fail = relay_record.get("failure_count", 0)
        total = succ + fail
        stability = succ / total if total > 0 else 0.5

        rep = 30 + stability * 70
        if fail > 10:
            rep -= 10
        if predicted_degrade:
            rep -= 15

        rep = max(0.0, min(100.0, rep))
        relay_record["reputation"] = rep

# -----------------------------
# GOSSIP / SWARM SYNC
# -----------------------------

class GossipNode(QtCore.QObject):
    gossip_log = QtCore.pyqtSignal(str)

    def __init__(self, relays_db, parent=None):
        super().__init__(parent)
        self.relays_db = relays_db
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", GOSSIP_PORT))
        sock.settimeout(1.0)

        last_broadcast = 0
        while not self._stop.is_set():
            now = time.time()
            if now - last_broadcast > GOSSIP_INTERVAL:
                last_broadcast = now
                self._broadcast(sock)
            try:
                data, addr = sock.recvfrom(65535)
                self._handle_gossip(data, addr)
            except socket.timeout:
                pass
            except Exception:
                pass

        sock.close()

    def _broadcast(self, sock):
        best = None
        for r in self.relays_db.get("known_relays", []):
            if best is None or r.get("score", 0) > best.get("score", 0):
                best = r
        payload = {
            "cluster_id": self.relays_db.get("cluster", {}).get("cluster_id", CLUSTER_ID),
            "node_id": self.relays_db.get("cluster", {}).get("node_id", NODE_ID),
            "best": {
                "endpoint": best.get("endpoint") if best else None,
                "protocol": best.get("protocol") if best else None,
                "score": best.get("score") if best else 0,
                "reputation": best.get("reputation") if best else 0,
            },
            "known_relays": [
                {
                    "endpoint": r.get("endpoint"),
                    "protocol": r.get("protocol"),
                    "score": r.get("score", 0),
                    "reputation": r.get("reputation", 50.0),
                    "trust": r.get("trust", "YELLOW"),
                }
                for r in self.relays_db.get("known_relays", [])
            ],
        }
        try:
            sock.sendto(json.dumps(payload).encode("utf-8"), ("255.255.255.255", GOSSIP_PORT))
            self.gossip_log.emit(f"[GOSSIP] Broadcast best relay: {payload['best']}")
        except Exception:
            pass

    def _handle_gossip(self, data, addr):
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            return
        if msg.get("cluster_id") != self.relays_db.get("cluster", {}).get("cluster_id", CLUSTER_ID):
            return
        node_id = msg.get("node_id")
        if node_id == self.relays_db.get("cluster", {}).get("node_id", NODE_ID):
            return

        best = msg.get("best") or {}
        ep = best.get("endpoint")
        proto = best.get("protocol")
        if ep and proto:
            self.gossip_log.emit(f"[GOSSIP] From {node_id}: {ep} ({proto}) score={best.get('score')} rep={best.get('reputation')}")
            for r in self.relays_db.get("known_relays", []):
                if r.get("endpoint") == ep and r.get("protocol") == proto:
                    peer_rep = best.get("reputation", 50.0)
                    r["reputation"] = (r.get("reputation", 50.0) * 3 + peer_rep) / 4
                    break

        for peer_r in msg.get("known_relays", []):
            ep2 = peer_r.get("endpoint")
            proto2 = peer_r.get("protocol")
            if not ep2 or not proto2:
                continue
            try:
                protocol_enum = Protocol(proto2)
            except Exception:
                continue
            existing = find_known_relay(self.relays_db, ep2, protocol_enum)
            if existing is None:
                self.relays_db = update_known_relay(
                    self.relays_db,
                    ep2,
                    protocol_enum,
                    peer_r.get("score", 0),
                    peer_r.get("trust", "YELLOW"),
                )
            else:
                existing["score"] = max(existing.get("score", 0), peer_r.get("score", 0))
                existing["reputation"] = (existing.get("reputation", 50.0) + peer_r.get("reputation", 50.0)) / 2
        save_relays_db(self.relays_db)

# -----------------------------
# RELAY VALIDATOR
# -----------------------------

class RelayValidator:
    @staticmethod
    def validate(endpoint: str, protocol: Protocol):
        if not endpoint or not isinstance(endpoint, str):
            return False, "Empty endpoint"
        if not isinstance(protocol, Protocol):
            return False, "Invalid protocol enum"

        if protocol == Protocol.TCP:
            if not endpoint.startswith("tcp://"):
                return False, "TCP endpoint must start with tcp://"
            try:
                host_port = endpoint[len("tcp://"):]
                host, port_str = host_port.split(":")
                host = host.strip()
                port = int(port_str)
                if not host:
                    return False, "TCP host empty"
                if port <= 0 or port > 65535:
                    return False, "TCP port out of range"
            except Exception:
                return False, "Malformed TCP endpoint"
            return True, None

        if protocol in (Protocol.HTTP, Protocol.WEBSOCKET):
            try:
                parsed = urlparse(endpoint)
                if protocol == Protocol.HTTP and parsed.scheme not in ("http", "https"):
                    return False, "HTTP URL must be http/https"
                if protocol == Protocol.WEBSOCKET and parsed.scheme not in ("ws", "wss"):
                    return False, "WS URL must be ws/wss"
                if not parsed.hostname:
                    return False, "URL host empty"
                if parsed.port is not None and (parsed.port <= 0 or parsed.port > 65535):
                    return False, "URL port out of range"
            except Exception:
                return False, "Malformed URL"
            return True, None

        return False, "Unknown protocol"

# -----------------------------
# RELAY CLIENT
# -----------------------------

class RelayClient(QtCore.QObject):
    status_updated = QtCore.pyqtSignal(object)
    auto_escape_triggered = QtCore.pyqtSignal()

    def __init__(self, status: RelayStatus, relays_db, prediction_engine: PredictionEngine, reputation_mgr: ReputationManager):
        super().__init__()
        self.status = status
        self.relays_db = relays_db
        self.prediction_engine = prediction_engine
        self.reputation_mgr = reputation_mgr
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def set_mode_and_protocol(self, mode: RelayMode, protocol: Protocol, endpoint: str):
        with self._lock:
            self.status.mode = mode
            self.status.protocol = protocol
            self.status.endpoint = endpoint
            self.status.success_count = 0
            self.status.failure_count = 0
            self.status.latency_ms = 0.0
            self.status.loss_percent = 0.0
            self.status.avg_latency_ms = 0.0
            self.status.congestion_level = "LOW"
            self.status.trust = TrustColor.GREEN if mode == RelayMode.LOCAL else TrustColor.YELLOW
            self.status._latency_window.clear()
            self.status.last_contact = "-"
            self.status.log(f"Switched to {mode.value} / {protocol.value} @ {endpoint}")
            self.relays_db["last_used"] = {
                "endpoint": endpoint,
                "protocol": protocol.value,
                "mode": mode.value,
            }
            save_relays_db(self.relays_db)
        self.status_updated.emit(self.status)

    def _loop(self):
        while not self._stop.is_set():
            start = time.time()
            try:
                with self._lock:
                    mode = self.status.mode
                    protocol = self.status.protocol
                    endpoint = self.status.endpoint

                self._contact_once(mode, protocol, endpoint)
                latency = (time.time() - start) * 1000.0

                with self._lock:
                    self.status.latency_ms = latency
                    self.status.success_count += 1
                    self.status.last_contact = time.strftime("%H:%M:%S")
                    self.status.update_latency_stats(latency)
                    self.status.update_loss_stats()

                    self.relays_db = record_relay_result(self.relays_db, endpoint, protocol, True)

                    self.prediction_engine.update(endpoint, protocol, latency, self.status.loss_percent)
                    r = find_known_relay(self.relays_db, endpoint, protocol)
                    if r:
                        predicted = self.prediction_engine.predict_degrade(endpoint, protocol)
                        r["predicted_degrade"] = predicted
                        self.reputation_mgr.update_reputation(r, predicted)

                    save_relays_db(self.relays_db)

                    if mode == RelayMode.LOCAL:
                        self.status.trust = TrustColor.GREEN
                    else:
                        if self.status.loss_percent > 20.0 or self.status.congestion_level == "HIGH":
                            self.status.trust = TrustColor.ORANGE
                        if self.status.loss_percent > 40.0:
                            self.status.trust = TrustColor.RED
                        if self.status.loss_percent <= 5.0 and self.status.congestion_level == "LOW":
                            self.status.trust = TrustColor.GREEN

                self.status_updated.emit(self.status)

            except Exception as e:
                with self._lock:
                    self.status.failure_count += 1
                    self.status.update_loss_stats()
                    self.relays_db = record_relay_result(self.relays_db, self.status.endpoint, self.status.protocol, False)
                    r = find_known_relay(self.relays_db, self.status.endpoint, self.status.protocol)
                    if r:
                        self.reputation_mgr.update_reputation(r, r.get("predicted_degrade", False))
                    save_relays_db(self.relays_db)
                    if self.status.failure_count < 3:
                        self.status.trust = TrustColor.ORANGE
                    else:
                        self.status.trust = TrustColor.RED
                    self.status.log(f"Contact error: {e}")

                    if (
                        self.status.failure_count >= 3
                        and self.status.success_count == 0
                        and self.status.loss_percent >= 100.0
                        and (self.status.latency_ms == 0.0 or self.status.avg_latency_ms == 0.0)
                        and self.status.last_contact == "-"
                    ):
                        self.status.log("[AUTO-ESCAPE] Dead relay detected — triggering escape.")
                        self.auto_escape_triggered.emit()

                self.status_updated.emit(self.status)

            time.sleep(2.0)

    def _contact_once(self, mode: RelayMode, protocol: Protocol, endpoint: str):
        if protocol == Protocol.TCP:
            self._contact_tcp(endpoint)
        elif protocol == Protocol.WEBSOCKET:
            self._contact_ws(endpoint)
        elif protocol == Protocol.HTTP:
            self._contact_http(endpoint)

    def _contact_tcp(self, endpoint: str):
        if not endpoint.startswith("tcp://"):
            raise ValueError("Invalid TCP endpoint format")
        host_port = endpoint[len("tcp://"):]
        host, port_str = host_port.split(":")
        port = int(port_str)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        if TLS_ENABLED:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            s = context.wrap_socket(s, server_hostname=host)
        start = time.time()
        try:
            s.connect((host, port))
            if AUTH_REQUIRED:
                s.sendall(f"AUTH {AUTH_TOKEN}\n".encode("utf-8"))
                resp = s.recv(1024)
                if b"OK" not in resp:
                    raise RuntimeError("Auth failed")
            s.sendall(b"PING")
            _ = s.recv(1024)
        finally:
            s.close()
        _ = (time.time() - start) * 1000.0

    def _contact_http(self, url: str):
        headers = {}
        if AUTH_REQUIRED:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        r = requests.get(url, timeout=1.0, headers=headers)
        if not r.ok:
            raise RuntimeError(f"HTTP status {r.status_code}")
        if r.text.strip() != "PONG":
            raise RuntimeError("Unexpected HTTP response")

    def _contact_ws(self, url: str):
        async def ping_ws():
            async with websockets.connect(url, ping_timeout=1.0, extra_headers={"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_REQUIRED else None) as ws:
                await ws.send("PING")
                resp = await ws.recv()
                if resp.strip() != "PONG":
                    raise RuntimeError("Unexpected WS response")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ping_ws())
        finally:
            loop.close()

# -----------------------------
# DISCOVERY (DNS / API / PROMETHEUS)
# -----------------------------

class DiscoveredRelay:
    def __init__(self, endpoint, protocol, latency_ms, loss_percent, congestion_level, score, trust, source, secure=False, auth_token=None):
        self.endpoint = endpoint
        self.protocol = protocol
        self.latency_ms = latency_ms
        self.loss_percent = loss_percent
        self.congestion_level = congestion_level
        self.score = score
        self.trust = trust
        self.source = source
        self.secure = secure
        self.auth_token = auth_token

class AutoSearchWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(list)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, relays_db, parent=None, prometheus_mode=True):
        super().__init__(parent)
        self._stop = threading.Event()
        self.relays_db = relays_db
        self.prometheus_mode = prometheus_mode

    def stop(self):
        self._stop.set()

    @QtCore.pyqtSlot()
    def run(self):
        results = []
        try:
            self.progress.emit("Scanning local network for TCP relays...")
            results.extend(self._scan_local_tcp())
            if self._stop.is_set():
                self.finished.emit(results); return

            self.progress.emit("Probing internet candidates (multi‑relay pool)...")
            results.extend(self._probe_internet_candidates())
            if self._stop.is_set():
                self.finished.emit(results); return

            self.progress.emit("DNS SRV discovery...")
            results.extend(self._dns_srv_discovery())
            if self._stop.is_set():
                self.finished.emit(results); return

            self.progress.emit("API directory discovery...")
            results.extend(self._api_directory_discovery())
            if self._stop.is_set():
                self.finished.emit(results); return

            self.progress.emit("Heuristic discovery...")
            results.extend(self._heuristic_discovery())
            if self._stop.is_set():
                self.finished.emit(results); return

            if self.prometheus_mode:
                self.progress.emit("Prometheus‑style cave scanning...")
                results.extend(self._prometheus_cave_scan())
        finally:
            for r in results:
                self.relays_db = update_known_relay(
                    self.relays_db, r.endpoint, r.protocol, r.score, r.trust.value, secure=r.secure, auth_token=r.auth_token
                )
            save_relays_db(self.relays_db)
            self.finished.emit(results)

    def _scan_local_tcp(self):
        discovered = []
        ports_to_try = [LOCAL_TCP_PORT]
        for port in ports_to_try:
            if self._stop.is_set():
                break
            endpoint = f"tcp://127.0.0.1:{port}"
            try:
                latency = self._probe_tcp(endpoint)
                loss = 0.0
                congestion = "LOW" if latency < 80 else "MEDIUM" if latency < 200 else "HIGH"
                hist = get_historical_stability(self.relays_db, endpoint, Protocol.TCP)
                score, trust = self._score_candidate(endpoint, Protocol.TCP, latency, loss, congestion, hist, secure=TLS_ENABLED)
                discovered.append(DiscoveredRelay(endpoint, Protocol.TCP, latency, loss, congestion, score, trust, "LOCAL", secure=TLS_ENABLED))
                self.progress.emit(f"Local: {endpoint} ({latency:.1f} ms, score {score:.1f})")
            except Exception:
                continue
        return discovered

    def _probe_internet_candidates(self):
        discovered = []
        for endpoint, protocol in INTERNET_CANDIDATES:
            if self._stop.is_set():
                break
            try:
                if protocol == Protocol.TCP:
                    latency = self._probe_tcp(endpoint)
                elif protocol == Protocol.HTTP:
                    latency = self._probe_http(endpoint)
                else:
                    latency = self._probe_ws(endpoint)
                loss = random.uniform(0.0, 5.0)
                congestion = "LOW" if latency < 80 else "MEDIUM" if latency < 200 else "HIGH"
                hist = get_historical_stability(self.relays_db, endpoint, protocol)
                secure = endpoint.startswith("https://") or endpoint.startswith("wss://") or TLS_ENABLED
                score, trust = self._score_candidate(endpoint, protocol, latency, loss, congestion, hist, secure=secure)
                discovered.append(DiscoveredRelay(endpoint, protocol, latency, loss, congestion, score, trust, "INTERNET", secure=secure))
                self.progress.emit(f"Internet: {endpoint} ({latency:.1f} ms, score {score:.1f})")
            except Exception:
                continue
        return discovered

    def _dns_srv_discovery(self):
        discovered = []
        if not HAVE_DNS:
            self.progress.emit("DNS library not available; skipping SRV.")
            return discovered

        discovery_cfg = self.relays_db.get("discovery", DEFAULT_DISCOVERY_CONFIG)
        domain = discovery_cfg.get("dns_srv_domain", DEFAULT_DISCOVERY_CONFIG["dns_srv_domain"]).strip()
        if not domain:
            self.progress.emit("DNS SRV domain not configured; skipping.")
            return discovered

        srv_name = f"_relay._tcp.{domain}"
        try:
            answers = dns.resolver.resolve(srv_name, "SRV")
        except Exception:
            self.progress.emit(f"No SRV records for {srv_name}.")
            return discovered

        for rdata in answers:
            if self._stop.is_set():
                break
            host = str(rdata.target).rstrip(".")
            port = int(rdata.port)
            endpoint = f"tcp://{host}:{port}"
            try:
                latency = self._probe_tcp(endpoint)
                loss = random.uniform(0.0, 10.0)
                congestion = "LOW" if latency < 80 else "MEDIUM" if latency < 200 else "HIGH"
                hist = get_historical_stability(self.relays_db, endpoint, Protocol.TCP)
                score, trust = self._score_candidate(endpoint, Protocol.TCP, latency, loss, congestion, hist, secure=TLS_ENABLED)
                discovered.append(DiscoveredRelay(endpoint, Protocol.TCP, latency, loss, congestion, score, trust, "DNS-SRV", secure=TLS_ENABLED))
                self.progress.emit(f"DNS‑SRV: {endpoint} ({latency:.1f} ms, score {score:.1f})")
            except Exception:
                continue
        return discovered

    def _api_directory_discovery(self):
        discovered = []
        discovery_cfg = self.relays_db.get("discovery", DEFAULT_DISCOVERY_CONFIG)
        api_url = discovery_cfg.get("api_directory_url", DEFAULT_DISCOVERY_CONFIG["api_directory_url"]).strip()
        if not api_url:
            self.progress.emit("API directory URL not configured; skipping.")
            return discovered

        try:
            resp = requests.get(api_url, timeout=2.0)
            if not resp.ok:
                self.progress.emit(f"API directory HTTP {resp.status_code}")
                return discovered
            data = resp.json()
        except Exception:
            self.progress.emit("API directory unreachable.")
            return discovered

        for item in data:
            if self._stop.is_set():
                break
            endpoint = item.get("endpoint")
            proto_name = item.get("protocol", "TCP")
            secure = bool(item.get("secure", False))
            auth_token = item.get("auth_token")
            try:
                protocol = Protocol(proto_name)
            except Exception:
                continue
            try:
                if protocol == Protocol.TCP:
                    latency = self._probe_tcp(endpoint)
                elif protocol == Protocol.HTTP:
                    latency = self._probe_http(endpoint)
                else:
                    latency = self._probe_ws(endpoint)
                loss = random.uniform(0.0, 10.0)
                congestion = "LOW" if latency < 80 else "MEDIUM" if latency < 200 else "HIGH"
                hist = get_historical_stability(self.relays_db, endpoint, protocol)
                score, trust = self._score_candidate(endpoint, protocol, latency, loss, congestion, hist, secure=secure)
                discovered.append(DiscoveredRelay(endpoint, protocol, latency, loss, congestion, score, trust, "API", secure=secure, auth_token=auth_token))
                self.progress.emit(f"API: {endpoint} ({latency:.1f} ms, score {score:.1f})")
            except Exception:
                continue
        return discovered

    def _heuristic_discovery(self):
        discovered = []
        return discovered

    def _prometheus_cave_scan(self):
        discovered = []
        hosts = ["127.0.0.1"]
        ports = [RELAY_PORT]
        max_probes = 3
        probes_done = 0
        for host in hosts:
            for port in ports:
                if self._stop.is_set() or probes_done >= max_probes:
                    return discovered
                endpoint = f"tcp://{host}:{port}"
                probes_done += 1
                try:
                    latency = self._probe_tcp(endpoint)
                    loss = random.uniform(0.0, 5.0)
                    congestion = "LOW" if latency < 80 else "MEDIUM" if latency < 200 else "HIGH"
                    hist = get_historical_stability(self.relays_db, endpoint, Protocol.TCP)
                    score, trust = self._score_candidate(endpoint, Protocol.TCP, latency, loss, congestion, hist, secure=TLS_ENABLED)
                    discovered.append(DiscoveredRelay(endpoint, Protocol.TCP, latency, loss, congestion, score, trust, "PROMETHEUS", secure=TLS_ENABLED))
                    self.progress.emit(f"Prometheus: {endpoint} ({latency:.1f} ms, score {score:.1f})")
                except Exception:
                    continue
        return discovered

    def _probe_tcp(self, endpoint: str) -> float:
        if not endpoint.startswith("tcp://"):
            raise ValueError("Invalid TCP endpoint format")
        host_port = endpoint[len("tcp://"):]
        host, port_str = host_port.split(":")
        port = int(port_str)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        if TLS_ENABLED:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            s = context.wrap_socket(s, server_hostname=host)
        start = time.time()
        try:
            s.connect((host, port))
            if AUTH_REQUIRED:
                s.sendall(f"AUTH {AUTH_TOKEN}\n".encode("utf-8"))
                resp = s.recv(1024)
                if b"OK" not in resp:
                    raise RuntimeError("Auth failed")
            s.sendall(b"PING")
            _ = s.recv(1024)
        finally:
            s.close()
        return (time.time() - start) * 1000.0

    def _probe_http(self, url: str) -> float:
        start = time.time()
        headers = {}
        if AUTH_REQUIRED:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        r = requests.get(url, timeout=1.0, headers=headers)
        if not r.ok:
            raise RuntimeError(f"HTTP status {r.status_code}")
        if r.text.strip() != "PONG":
            raise RuntimeError("Unexpected HTTP response")
        return (time.time() - start) * 1000.0

    def _probe_ws(self, url: str) -> float:
        async def ping_ws():
            start = time.time()
            async with websockets.connect(url, ping_timeout=1.0, extra_headers={"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_REQUIRED else None) as ws:
                await ws.send("PING")
                resp = await ws.recv()
                if resp.strip() != "PONG":
                    raise RuntimeError("Unexpected WS response")
            return (time.time() - start) * 1000.0
        loop = asyncio.new_event_loop()
        try:
            latency = loop.run_until_complete(ping_ws())
        finally:
            loop.close()
        return latency

    def _score_candidate(self, endpoint: str, protocol: Protocol, latency_ms: float, loss_percent: float, congestion_level: str, stability: float, secure: bool):
        score = max(0.0, 100.0 - latency_ms * 0.3)
        score -= loss_percent * 0.8
        if congestion_level == "MEDIUM":
            score -= 10.0
        elif congestion_level == "HIGH":
            score -= 25.0
        score += stability * 20.0
        if secure:
            score += 5.0
        score = max(0.0, min(100.0, score))

        if score >= 80:
            trust = TrustColor.GREEN
        elif score >= 60:
            trust = TrustColor.YELLOW
        elif score >= 40:
            trust = TrustColor.ORANGE
        else:
            trust = TrustColor.RED

        return score, trust

# -----------------------------
# TIMED AUTO RE‑OPTIMIZER
# -----------------------------

class AutoReoptimizer(QtCore.QObject):
    log_event = QtCore.pyqtSignal(str)
    reopt_decision = QtCore.pyqtSignal(dict)
    status_update = QtCore.pyqtSignal(dict)

    def __init__(self, relays_db, status: RelayStatus, parent=None):
        super().__init__(parent)
        self.relays_db = relays_db
        self.status = status
        cfg = self.relays_db.get("auto_reoptimize", DEFAULT_AUTO_REOPT)
        self.enabled = cfg.get("enabled", True)
        self.interval_minutes = cfg.get("interval_minutes", 5)
        self.min_score_improvement = float(cfg.get("min_score_improvement", 10))
        self.next_run_seconds = self.interval_minutes * 60
        self.last_run_time = None
        self.last_best_score = 0.0
        self.last_improvement = 0.0
        self.last_swap_endpoint = "-"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._lock = threading.Lock()

    def start(self):
        if not self.enabled:
            return
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    @QtCore.pyqtSlot(bool, int, int)
    def update_settings(self, enabled: bool, interval_minutes: int, min_improvement: int):
        with self._lock:
            self.enabled = enabled
            self.interval_minutes = max(1, int(interval_minutes))
            self.min_score_improvement = float(min_improvement)
            self.next_run_seconds = self.interval_minutes * 60
            self.relays_db["auto_reoptimize"] = {
                "enabled": self.enabled,
                "interval_minutes": self.interval_minutes,
                "min_score_improvement": self.min_score_improvement,
            }
            save_relays_db(self.relays_db)
        self.log_event.emit(
            f"[AUTO-REOPT] Settings: enabled={self.enabled}, interval={self.interval_minutes} min, min_improvement={self.min_score_improvement}"
        )

    @QtCore.pyqtSlot()
    def manual_optimize(self):
        self.log_event.emit("[AUTO-REOPT] Manual optimization triggered.")
        self._run_optimization_cycle(manual=True)

    def _current_relay_pinned(self) -> bool:
        ep = self.status.endpoint
        proto = self.status.protocol.value
        for r in self.relays_db.get("known_relays", []):
            if r.get("endpoint") == ep and r.get("protocol") == proto:
                return bool(r.get("pinned", False))
        return False

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                if not self.enabled:
                    self.next_run_seconds = self.interval_minutes * 60
            for _ in range(self.interval_minutes * 60):
                if self._stop.is_set():
                    return
                with self._lock:
                    if not self.enabled:
                        self.next_run_seconds = self.interval_minutes * 60
                    else:
                        self.next_run_seconds = max(0, self.next_run_seconds - 1)
                    self._emit_status()
                time.sleep(1)
                if self._stop.is_set():
                    return
                if not self.enabled:
                    continue
            if self._stop.is_set():
                return
            if self.enabled:
                self._run_optimization_cycle(manual=False)
                with self._lock:
                    self.next_run_seconds = self.interval_minutes * 60

    def _run_optimization_cycle(self, manual: bool):
        tag = "MANUAL" if manual else "AUTO-REOPT"

        if not manual and self._current_relay_pinned():
            self.log_event.emit(f"[{tag}] Current relay pinned; skipping.")
            return

        self.log_event.emit(f"[{tag}] Running optimization...")
        results = self._run_headless_search()
        if not results:
            self.log_event.emit(f"[{tag}] No candidates.")
            return

        sorted_res = sorted(results, key=lambda r: r.score, reverse=True)
        top = sorted_res[:3]
        best = top[0]
        best_score = best.score
        current_score = self._estimate_current_score()
        self.last_run_time = time.strftime("%H:%M:%S")
        self.last_best_score = best_score
        self.last_improvement = best_score - current_score
        self._emit_status()

        self.log_event.emit(f"[{tag}] Best: {best.endpoint} ({best.protocol.value}, score {best_score:.1f})")
        self.log_event.emit(f"[{tag}] Current estimated score: {current_score:.1f}")

        if best_score >= current_score + self.min_score_improvement and best.trust != TrustColor.RED:
            chosen = self._choose_balanced(top)
            decision = {
                "endpoint": chosen.endpoint,
                "protocol": chosen.protocol.value,
                "mode": RelayMode.AUTO.value,
                "score": chosen.score,
            }
            self.last_swap_endpoint = chosen.endpoint
            self.log_event.emit(f"[{tag}] Improvement +{best_score - current_score:.1f}. Hot‑swapping to {chosen.endpoint}")
            self._emit_status()
            self.reopt_decision.emit(decision)
        else:
            self.log_event.emit(f"[{tag}] No significant improvement; keeping current.")
            self._emit_status()

    def _run_headless_search(self):
        results_container = []
        def finished(results):
            results_container.extend(results)
        worker = AutoSearchWorker(self.relays_db, prometheus_mode=True)
        worker.finished.connect(finished)
        worker.run()
        return results_container

    def _estimate_current_score(self):
        latency = self.status.avg_latency_ms or self.status.latency_ms or 100.0
        loss = self.status.loss_percent
        congestion = self.status.congestion_level
        score = max(0.0, 100.0 - latency * 0.3)
        score -= loss * 0.8
        if congestion == "MEDIUM":
            score -= 10.0
        elif congestion == "HIGH":
            score -= 25.0
        return max(0.0, min(100.0, score))

    def _choose_balanced(self, candidates):
        best = None
        best_weight = -1
        for c in candidates:
            r = find_known_relay(self.relays_db, c.endpoint, c.protocol)
            rep = r.get("reputation", 50.0) if r else 50.0
            weight = c.score * 0.7 + rep * 0.3
            if weight > best_weight:
                best_weight = weight
                best = c
        return best or candidates[0]

    def _emit_status(self):
        data = {
            "last_run": self.last_run_time or "-",
            "next_run_in": self.next_run_seconds,
            "last_best_score": self.last_best_score,
            "last_improvement": self.last_improvement,
            "last_swap_endpoint": self.last_swap_endpoint,
            "enabled": self.enabled,
            "interval_minutes": self.interval_minutes,
            "min_score_improvement": self.min_score_improvement,
        }
        self.status_update.emit(data)

# -----------------------------
# UIAUTOMATION WATCHER
# -----------------------------

class UIAutomationWatcher(QtCore.QObject):
    ui_event = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        last_title = ""
        while not self._stop.is_set():
            try:
                window = auto.GetForegroundControl()
                title = window.Name if window else ""
                if title != last_title:
                    last_title = title
                    self.ui_event.emit(title)
            except Exception:
                pass
            time.sleep(1.0)

# -----------------------------
# HUD ALERT OVERLAY (ANIMATED)
# -----------------------------

class HUDAlertOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.alerts = []
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def show_alert(self, text: str, color: QtGui.QColor, duration_ms: int = 2000, style: str = "normal"):
        alert = {
            "text": text,
            "color": color,
            "created": time.time(),
            "duration": duration_ms / 1000.0,
            "style": style,
        }
        self.alerts.append(alert)
        self.update()

    def _tick(self):
        now = time.time()
        self.alerts = [a for a in self.alerts if now - a["created"] < a["duration"]]
        if self.alerts:
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        if not self.alerts:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        center_x = self.width() // 2
        top_y = int(self.height() * 0.08)

        now = time.time()
        alert_height = 40
        spacing = 8

        for idx, alert in enumerate(self.alerts):
            age = now - alert["created"]
            t = age / alert["duration"]
            t = max(0.0, min(1.0, t))

            alpha = int(255 * (1.0 - t))
            base_color = alert["color"]
            color = QtGui.QColor(base_color.red(), base_color.green(), base_color.blue(), alpha)

            pulse = 1.0 + 0.1 * (1.0 - t)
            if alert["style"] == "red-shake":
                offset_x = int(4 * (1.0 - t) * (1 if int(age * 20) % 2 == 0 else -1))
            elif alert["style"] == "green-pulse":
                offset_x = 0
                pulse = 1.0 + 0.2 * (1.0 - t)
            elif alert["style"] == "blue-swoop":
                offset_x = int(10 * (1.0 - t))
            else:
                offset_x = 0

            width = int(420 * pulse)
            height = int(alert_height * pulse)
            x = center_x - width // 2 + offset_x
            y = top_y + idx * (alert_height + spacing)

            rect = QtCore.QRect(x, y, width, height)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(rect, 10, 10)

            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, alpha)))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(rect, QtCore.Qt.AlignCenter, alert["text"])

# -----------------------------
# STATUS LIGHT (WITH SHAKE)
# -----------------------------

class StatusLight(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(80, 80)
        self._color = QtGui.QColor("gray")
        self._base_color = self._color
        self._flash_timer = QtCore.QTimer(self)
        self._flash_timer.timeout.connect(self._end_flash)
        self._shaking = False
        self._shake_phase = 0
        self._shake_timer = QtCore.QTimer(self)
        self._shake_timer.timeout.connect(self._tick_shake)

    def set_trust(self, trust: TrustColor):
        mapping = {
            TrustColor.GREEN: QtGui.QColor(0, 200, 0),
            TrustColor.YELLOW: QtGui.QColor(220, 220, 0),
            TrustColor.ORANGE: QtGui.QColor(255, 140, 0),
            TrustColor.RED: QtGui.QColor(220, 0, 0),
        }
        self._base_color = mapping.get(trust, QtGui.QColor("gray"))
        if not self._flash_timer.isActive():
            self._color = self._base_color
            self.update()

    def flash(self, color: QtGui.QColor, duration_ms: int = 1000, shake: bool = False):
        self._color = color
        self._flash_timer.start(duration_ms)
        if shake:
            self._shaking = True
            self._shake_phase = 0
            self._shake_timer.start(50)
        self.update()

    def _end_flash(self):
        self._flash_timer.stop()
        self._color = self._base_color
        self._shaking = False
        self._shake_timer.stop()
        self.update()

    def _tick_shake(self):
        self._shake_phase += 1
        if self._shake_phase > 20:
            self._shaking = False
            self._shake_timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(QtCore.Qt.NoPen)
        radius = min(self.width(), self.height()) - 4

        dx = 0
        if self._shaking:
            dx = 2 if self._shake_phase % 2 == 0 else -2

        painter.translate(dx, 0)
        painter.drawEllipse(2, 2, radius, radius)

# -----------------------------
# AUTO SEARCH DIALOG
# -----------------------------

class AutoSearchDialog(QtWidgets.QDialog):
    def __init__(self, relays_db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Auto‑Search: External Relays")
        self.resize(800, 450)

        self.worker = AutoSearchWorker(relays_db, prometheus_mode=True)
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self.on_progress)
        self.thread.started.connect(self.worker.run)

        layout = QtWidgets.QVBoxLayout(self)

        self.progress_label = QtWidgets.QLabel("Starting scan...")
        layout.addWidget(self.progress_label)

        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Endpoint", "Protocol", "Latency", "Loss %", "Congestion", "Score", "Trust", "Source", "Secure"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect to Selected")
        self.btn_best = QtWidgets.QPushButton("Auto‑Select Best")
        self.btn_cancel = QtWidgets.QPushButton("Close")
        btn_layout.addWidget(self.btn_connect)
        btn_layout.addWidget(self.btn_best)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.btn_cancel.clicked.connect(self.reject)

        self._results = []
        self._selected = None

        self.table.doubleClicked.connect(self._on_double_click)
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_best.clicked.connect(self._on_best_clicked)

        self.thread.start()

    def on_progress(self, msg: str):
        self.progress_label.setText(msg)

    def on_finished(self, results: list):
        self._results = results
        self.progress_label.setText(f"Scan complete. {len(results)} candidates found.")
        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(0)
        for r in self._results:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(r.endpoint))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(r.protocol.value))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{r.latency_ms:.1f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{r.loss_percent:.1f}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(r.congestion_level))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{r.score:.1f}"))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(r.trust.value))
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(r.source))
            self.table.setItem(row, 8, QtWidgets.QTableWidgetItem("YES" if r.secure else "NO"))

    def _on_double_click(self):
        self._select_current()
        if self._selected:
            self.accept()

    def _on_connect_clicked(self):
        self._select_current()
        if self._selected:
            self.accept()

    def _on_best_clicked(self):
        if not self._results:
            self._selected = None
            return
        best = max(self._results, key=lambda r: r.score)
        self._selected = best
        self.accept()

    def _select_current(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._results):
            self._selected = None
            return
        self._selected = self._results[row]

    def selected_relay(self):
        return self._selected

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)

# -----------------------------
# MAIN WINDOW / COCKPIT
# -----------------------------

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Relay Control Cockpit (TLS / Swarm)")

        self.relays_db = load_relays_db()
        self.status = RelayStatus()
        self.prediction_engine = PredictionEngine()
        self.reputation_mgr = ReputationManager()
        self.client = RelayClient(self.status, self.relays_db, self.prediction_engine, self.reputation_mgr)
        self.client.status_updated.connect(self.on_status_updated)
        self.client.auto_escape_triggered.connect(self.on_auto_escape_triggered)

        self.ui_watcher = UIAutomationWatcher()
        self.ui_watcher.ui_event.connect(self.on_ui_event)

        self.reoptimizer = AutoReoptimizer(self.relays_db, self.status)
        self.reoptimizer.log_event.connect(self.on_reopt_log)
        self.reoptimizer.reopt_decision.connect(self.on_reopt_decision)
        self.reoptimizer.status_update.connect(self.on_reopt_status)

        self.gossip = GossipNode(self.relays_db)
        self.gossip.gossip_log.connect(self.on_reopt_log)

        self._build_ui()
        self._wire_events()

        last = self.relays_db.get("last_used")
        if last:
            mode = RelayMode(last.get("mode", "LOCAL"))
            protocol = Protocol(last.get("protocol", "TCP"))
            endpoint = last.get("endpoint", f"tcp://{LOCAL_TCP_HOST}:{LOCAL_TCP_PORT}")
        else:
            mode = RelayMode.LOCAL
            protocol = Protocol.TCP
            endpoint = f"tcp://{LOCAL_TCP_HOST}:{LOCAL_TCP_PORT}"

        self.client.set_mode_and_protocol(mode, protocol, endpoint)
        self.client.start()
        self.ui_watcher.start()
        self.reoptimizer.start()
        self.gossip.start()
        self.refresh_known_relays_panel()
        self._load_discovery_fields()

    def _build_ui(self):
        root_layout = QtWidgets.QHBoxLayout(self)

        left_layout = QtWidgets.QVBoxLayout()

        group = QtWidgets.QGroupBox("RELAY CONTROL")
        group_layout = QtWidgets.QVBoxLayout(group)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_local = QtWidgets.QPushButton("LOCAL")
        self.btn_cloud = QtWidgets.QPushButton("CLOUD")
        self.btn_custom = QtWidgets.QPushButton("CUSTOM")
        self.btn_auto = QtWidgets.QPushButton("AUTO‑SEARCH")
        for b in (self.btn_local, self.btn_cloud, self.btn_custom, self.btn_auto):
            b.setCheckable(True)
            b.setMinimumHeight(40)
            b.setStyleSheet("font-weight: bold; font-size: 14px;")
        btn_layout.addWidget(self.btn_local)
        btn_layout.addWidget(self.btn_cloud)
        btn_layout.addWidget(self.btn_custom)
        btn_layout.addWidget(self.btn_auto)

        center_layout = QtWidgets.QHBoxLayout()
        self.status_light = StatusLight()

        metrics_layout = QtWidgets.QFormLayout()
        self.lbl_mode = QtWidgets.QLabel("-")
        self.lbl_protocol = QtWidgets.QLabel("-")
        self.lbl_endpoint = QtWidgets.QLabel("-")
        self.lbl_latency = QtWidgets.QLabel("-")
        self.lbl_avg_latency = QtWidgets.QLabel("-")
        self.lbl_congestion = QtWidgets.QLabel("LOW")
        self.lbl_loss = QtWidgets.QLabel("0.0 %")
        self.lbl_success = QtWidgets.QLabel("0")
        self.lbl_failure = QtWidgets.QLabel("0")
        self.lbl_last = QtWidgets.QLabel("-")
        self.lbl_auto = QtWidgets.QLabel("Ready")

        metrics_layout.addRow("Mode:", self.lbl_mode)
        metrics_layout.addRow("Protocol:", self.lbl_protocol)
        metrics_layout.addRow("Endpoint:", self.lbl_endpoint)
        metrics_layout.addRow("Latency (ms):", self.lbl_latency)
        metrics_layout.addRow("Avg Latency (ms):", self.lbl_avg_latency)
        metrics_layout.addRow("Congestion:", self.lbl_congestion)
        metrics_layout.addRow("Loss:", self.lbl_loss)
        metrics_layout.addRow("Success:", self.lbl_success)
        metrics_layout.addRow("Failure:", self.lbl_failure)
        metrics_layout.addRow("Last Contact:", self.lbl_last)
        metrics_layout.addRow("Autoloader:", self.lbl_auto)

        center_layout.addWidget(self.status_light)
        center_layout.addLayout(metrics_layout)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setStyleSheet(
            "background-color: #111; color: #0f0; font-family: Consolas; font-size: 11px;"
        )

        group_layout.addLayout(btn_layout)
        group_layout.addLayout(center_layout)
        group_layout.addWidget(self.log_view)

        relays_group = QtWidgets.QGroupBox("KNOWN RELAYS COCKPIT")
        relays_layout = QtWidgets.QVBoxLayout(relays_group)

        self.relays_table = QtWidgets.QTableWidget(0, 12)
        self.relays_table.setHorizontalHeaderLabels(
            ["Pinned", "Endpoint", "Protocol", "Score", "Trust", "Stability", "Success", "Failure", "Last Seen", "Rep", "Pred", "Res"]
        )
        self.relays_table.horizontalHeader().setStretchLastSection(True)
        relays_layout.addWidget(self.relays_table)

        left_layout.addWidget(group)
        left_layout.addWidget(relays_group)

        right_layout = QtWidgets.QVBoxLayout()

        reopt_group = QtWidgets.QGroupBox("AUTO‑REOPT SETTINGS")
        reopt_layout = QtWidgets.QVBoxLayout(reopt_group)

        self.chk_reopt_enable = QtWidgets.QCheckBox("Enable Auto‑Reopt")
        cfg = self.relays_db.get("auto_reoptimize", DEFAULT_AUTO_REOPT)
        self.chk_reopt_enable.setChecked(cfg.get("enabled", True))

        interval_layout = QtWidgets.QHBoxLayout()
        lbl_interval = QtWidgets.QLabel("Re‑optimize every (min):")
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 60)
        self.spin_interval.setValue(int(cfg.get("interval_minutes", 5)))
        interval_layout.addWidget(lbl_interval)
        interval_layout.addWidget(self.spin_interval)

        thresh_layout = QtWidgets.QHBoxLayout()
        lbl_thresh = QtWidgets.QLabel("Min score improvement:")
        self.spin_threshold = QtWidgets.QSpinBox()
        self.spin_threshold.setRange(1, 50)
        self.spin_threshold.setValue(int(cfg.get("min_score_improvement", 10)))
        thresh_layout.addWidget(lbl_thresh)
        thresh_layout.addWidget(self.spin_threshold)

        self.btn_reopt_now = QtWidgets.QPushButton("Optimize Now")

        status_form = QtWidgets.QFormLayout()
        self.lbl_reopt_last_run = QtWidgets.QLabel("-")
        self.lbl_reopt_next_run = QtWidgets.QLabel("-")
        self.lbl_reopt_last_best = QtWidgets.QLabel("0.0")
        self.lbl_reopt_last_improv = QtWidgets.QLabel("0.0")
        self.lbl_reopt_last_swap = QtWidgets.QLabel("-")

        status_form.addRow("Last run:", self.lbl_reopt_last_run)
        status_form.addRow("Next run in (s):", self.lbl_reopt_next_run)
        status_form.addRow("Last best score:", self.lbl_reopt_last_best)
        status_form.addRow("Last improvement:", self.lbl_reopt_last_improv)
        status_form.addRow("Last hot‑swap:", self.lbl_reopt_last_swap)

        reopt_layout.addWidget(self.chk_reopt_enable)
        reopt_layout.addLayout(interval_layout)
        reopt_layout.addLayout(thresh_layout)
        reopt_layout.addWidget(self.btn_reopt_now)
        reopt_layout.addSpacing(10)
        reopt_layout.addLayout(status_form)

        discovery_group = QtWidgets.QGroupBox("DISCOVERY CONFIG")
        discovery_layout = QtWidgets.QFormLayout(discovery_group)

        self.edit_dns_srv_domain = QtWidgets.QLineEdit()
        self.edit_api_directory_url = QtWidgets.QLineEdit()
        self.btn_save_discovery = QtWidgets.QPushButton("Save Discovery Settings")

        discovery_layout.addRow("DNS SRV domain:", self.edit_dns_srv_domain)
        discovery_layout.addRow("API directory URL:", self.edit_api_directory_url)
        discovery_layout.addRow(self.btn_save_discovery)

        right_layout.addWidget(reopt_group)
        right_layout.addWidget(discovery_group)
        right_layout.addStretch()

        root_layout.addLayout(left_layout, 3)
        root_layout.addLayout(right_layout, 1)

        self.setLayout(root_layout)
        self.resize(1300, 700)

        self.hud_overlay = HUDAlertOverlay(self)
        self.hud_overlay.setGeometry(self.rect())
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self and event.type() == QtCore.QEvent.Resize:
            self.hud_overlay.setGeometry(self.rect())
        return super().eventFilter(obj, event)

    def _wire_events(self):
        self.btn_local.clicked.connect(self.on_local_clicked)
        self.btn_cloud.clicked.connect(self.on_cloud_clicked)
        self.btn_custom.clicked.connect(self.on_custom_clicked)
        self.btn_auto.clicked.connect(self.on_auto_clicked)

        self.chk_reopt_enable.toggled.connect(self.on_reopt_settings_changed)
        self.spin_interval.valueChanged.connect(self.on_reopt_settings_changed)
        self.spin_threshold.valueChanged.connect(self.on_reopt_settings_changed)
        self.btn_reopt_now.clicked.connect(self.on_reopt_now_clicked)

        self.btn_save_discovery.clicked.connect(self.on_save_discovery_clicked)

    def _load_discovery_fields(self):
        discovery_cfg = self.relays_db.get("discovery", DEFAULT_DISCOVERY_CONFIG)
        self.edit_dns_srv_domain.setText(discovery_cfg.get("dns_srv_domain", DEFAULT_DISCOVERY_CONFIG["dns_srv_domain"]))
        self.edit_api_directory_url.setText(discovery_cfg.get("api_directory_url", DEFAULT_DISCOVERY_CONFIG["api_directory_url"]))

    def _set_button_state(self, active: QtWidgets.QPushButton):
        for b in (self.btn_local, self.btn_cloud, self.btn_custom, self.btn_auto):
            b.setChecked(b is active)
            if b is active:
                b.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #444; color: #fff;")
            else:
                b.setStyleSheet("font-weight: bold; font-size: 14px;")

    def _switch_with_validation(self, mode: RelayMode, protocol: Protocol, endpoint: str, source: str):
        ok, reason = RelayValidator.validate(endpoint, protocol)
        if not ok:
            msg = f"[VALIDATOR] Rejected {endpoint} ({protocol.value}) from {source}: {reason}"
            self.status.log(msg)
            self.log_view.setPlainText("\n".join(self.status.log_lines))
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
            self.hud_overlay.show_alert("VALIDATOR: Invalid relay rejected", QtGui.QColor(255, 255, 0), 1500, "yellow")
            self.status_light.flash(QtGui.QColor(255, 255, 0), 1000, shake=False)
            return False
        self.client.set_mode_and_protocol(mode, protocol, endpoint)
        style = "green-pulse" if mode != RelayMode.LOCAL else "blue-swoop"
        self.hud_overlay.show_alert(f"Connected to: {endpoint}", QtGui.QColor(0, 255, 0), 1200, style)
        self.status_light.flash(QtGui.QColor(0, 255, 0), 800, shake=False)
        return True

    def on_local_clicked(self):
        self._set_button_state(self.btn_local)
        endpoint = f"tcp://{LOCAL_TCP_HOST}:{LOCAL_TCP_PORT}"
        self._switch_with_validation(RelayMode.LOCAL, Protocol.TCP, endpoint, "LOCAL")

    def on_cloud_clicked(self):
        self._set_button_state(self.btn_cloud)
        protocol, ok = self._ask_protocol_dialog(default=Protocol.HTTP)
        if not ok:
            return
        if protocol == Protocol.TCP:
            endpoint = f"tcp://{CLOUD_TCP_HOST}:{CLOUD_TCP_PORT}"
        elif protocol == Protocol.WEBSOCKET:
            endpoint = CLOUD_WS_URL
        else:
            endpoint = CLOUD_HTTP_URL
        self._switch_with_validation(RelayMode.CLOUD, protocol, endpoint, "CLOUD")

    def on_custom_clicked(self):
        self._set_button_state(self.btn_custom)
        protocol, ok = self._ask_protocol_dialog(default=Protocol.TCP)
        if not ok:
            return
        if protocol == Protocol.TCP:
            text, ok2 = QtWidgets.QInputDialog.getText(self, "Custom TCP Endpoint", "tcp://host:port", text=f"tcp://127.0.0.1:{RELAY_PORT}")
        elif protocol == Protocol.WEBSOCKET:
            text, ok2 = QtWidgets.QInputDialog.getText(self, "Custom WebSocket URL", "ws://...", text=LOCAL_WS_URL)
        else:
            text, ok2 = QtWidgets.QInputDialog.getText(self, "Custom HTTP URL", "http://...", text=LOCAL_HTTP_URL)
        if not ok2 or not text:
            return
        self._switch_with_validation(RelayMode.CUSTOM, protocol, text, "CUSTOM")

    def on_auto_clicked(self):
        self._set_button_state(self.btn_auto)
        dlg = AutoSearchDialog(self.relays_db, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected = dlg.selected_relay()
            if selected:
                if self._switch_with_validation(RelayMode.AUTO, selected.protocol, selected.endpoint, "AUTO-SEARCH"):
                    self.refresh_known_relays_panel()
                else:
                    self.hud_overlay.show_alert("Relay rejected — invalid or unsafe", QtGui.QColor(255, 255, 0), 1500, "yellow")

    def _ask_protocol_dialog(self, default: Protocol = Protocol.TCP):
        items = ["TCP", "WebSocket", "HTTP"]
        default_index = items.index(default.value)
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select Protocol", "Protocol:", items, default_index, False)
        if not ok:
            return default, False
        mapping = {"TCP": Protocol.TCP, "WebSocket": Protocol.WEBSOCKET, "HTTP": Protocol.HTTP}
        return mapping[item], True

    def on_reopt_settings_changed(self):
        enabled = self.chk_reopt_enable.isChecked()
        interval = self.spin_interval.value()
        threshold = self.spin_threshold.value()
        self.reoptimizer.update_settings(enabled, interval, threshold)
        if enabled:
            self.reoptimizer.start()

    def on_reopt_now_clicked(self):
        self.reoptimizer.manual_optimize()

    def on_save_discovery_clicked(self):
        dns_domain = self.edit_dns_srv_domain.text().strip()
        api_url = self.edit_api_directory_url.text().strip()
        if "discovery" not in self.relays_db:
            self.relays_db["discovery"] = DEFAULT_DISCOVERY_CONFIG.copy()
        self.relays_db["discovery"]["dns_srv_domain"] = dns_domain
        self.relays_db["discovery"]["api_directory_url"] = api_url
        save_relays_db(self.relays_db)
        self.status.log(f"[DISCOVERY] Updated DNS SRV domain='{dns_domain}', API directory='{api_url}'")
        self.log_view.setPlainText("\n".join(self.status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def on_pin_changed(self, endpoint: str, protocol: str, pinned: bool):
        for r in self.relays_db.get("known_relays", []):
            if r.get("endpoint") == endpoint and r.get("protocol") == protocol:
                r["pinned"] = bool(pinned)
                break
        save_relays_db(self.relays_db)
        self.status.log(f"[PIN] {endpoint} ({protocol}) pinned={pinned}")
        self.log_view.setPlainText("\n".join(self.status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def refresh_known_relays_panel(self):
        relays = self.relays_db.get("known_relays", [])
        self.relays_table.setRowCount(0)
        for r in relays:
            row = self.relays_table.rowCount()
            self.relays_table.insertRow(row)

            pinned_chk = QtWidgets.QCheckBox()
            pinned_chk.setChecked(bool(r.get("pinned", False)))
            endpoint = r["endpoint"]
            protocol = r["protocol"]
            pinned_chk.stateChanged.connect(
                lambda state, ep=endpoint, pr=protocol: self.on_pin_changed(ep, pr, state == QtCore.Qt.Checked)
            )
            self.relays_table.setCellWidget(row, 0, pinned_chk)

            endpoint_item = QtWidgets.QTableWidgetItem(endpoint)
            protocol_item = QtWidgets.QTableWidgetItem(protocol)

            score_bar = QtWidgets.QProgressBar()
            score_bar.setRange(0, 100)
            score_bar.setValue(int(r.get("score", 0)))
            score_bar.setFormat(f'{int(r.get("score", 0))}')

            trust_item = QtWidgets.QTableWidgetItem(r.get("trust", "YELLOW"))

            succ = r.get("success_count", 0)
            fail = r.get("failure_count", 0)
            total = succ + fail
            stability = int((succ / total) * 100) if total > 0 else 0
            stability_bar = QtWidgets.QProgressBar()
            stability_bar.setRange(0, 100)
            stability_bar.setValue(stability)
            stability_bar.setFormat(f'{stability}%')

            success_item = QtWidgets.QTableWidgetItem(str(succ))
            failure_item = QtWidgets.QTableWidgetItem(str(fail))
            last_seen_ts = r.get("last_seen", 0)
            last_seen_str = time.strftime("%H:%M:%S", time.localtime(last_seen_ts)) if last_seen_ts else "-"
            last_seen_item = QtWidgets.QTableWidgetItem(last_seen_str)

            rep_item = QtWidgets.QTableWidgetItem(f"{r.get('reputation', 50.0):.1f}")
            pred_item = QtWidgets.QTableWidgetItem("YES" if r.get("predicted_degrade") else "NO")
            res_item = QtWidgets.QTableWidgetItem("YES" if r.get("resurrected") else "NO")

            if r.get("predicted_degrade"):
                pred_item.setBackground(QtGui.QColor(255, 200, 0))
            if r.get("resurrected"):
                res_item.setBackground(QtGui.QColor(0, 200, 255))

            self.relays_table.setCellWidget(row, 0, pinned_chk)
            self.relays_table.setItem(row, 1, endpoint_item)
            self.relays_table.setItem(row, 2, protocol_item)
            self.relays_table.setCellWidget(row, 3, score_bar)
            self.relays_table.setItem(row, 4, trust_item)
            self.relays_table.setCellWidget(row, 5, stability_bar)
            self.relays_table.setItem(row, 6, success_item)
            self.relays_table.setItem(row, 7, failure_item)
            self.relays_table.setItem(row, 8, last_seen_item)
            self.relays_table.setItem(row, 9, rep_item)
            self.relays_table.setItem(row, 10, pred_item)
            self.relays_table.setItem(row, 11, res_item)

    @QtCore.pyqtSlot(object)
    def on_status_updated(self, status: RelayStatus):
        self.lbl_mode.setText(status.mode.value)
        self.lbl_protocol.setText(status.protocol.value)
        self.lbl_endpoint.setText(status.endpoint)
        self.lbl_latency.setText(f"{status.latency_ms:.1f}")
        self.lbl_avg_latency.setText(f"{status.avg_latency_ms:.1f}")
        self.lbl_congestion.setText(status.congestion_level)
        self.lbl_loss.setText(f"{status.loss_percent:.1f} %")
        self.lbl_success.setText(str(status.success_count))
        self.lbl_failure.setText(str(status.failure_count))
        self.lbl_last.setText(status.last_contact)
        self.lbl_auto.setText(status.autoloader_status)
        self.status_light.set_trust(status.trust)

        self.log_view.setPlainText("\n".join(status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

        self.refresh_known_relays_panel()

    @QtCore.pyqtSlot(str)
    def on_ui_event(self, title: str):
        if not title:
            return
        self.status.log(f"Active window: {title}")
        self.log_view.setPlainText("\n".join(self.status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        self._maybe_apply_app_preference(title)

    @QtCore.pyqtSlot(str)
    def on_reopt_log(self, msg: str):
        self.status.log(msg)
        self.log_view.setPlainText("\n".join(self.status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    @QtCore.pyqtSlot(dict)
    def on_reopt_decision(self, decision: dict):
        mode = RelayMode(decision["mode"])
        protocol = Protocol(decision["protocol"])
        endpoint = decision["endpoint"]
        if not self._switch_with_validation(mode, protocol, endpoint, "AUTO-REOPT"):
            self.hud_overlay.show_alert("VALIDATOR: Invalid relay rejected", QtGui.QColor(255, 255, 0), 1500, "yellow")

    @QtCore.pyqtSlot(dict)
    def on_reopt_status(self, data: dict):
        self.lbl_reopt_last_run.setText(data.get("last_run", "-"))
        self.lbl_reopt_next_run.setText(str(data.get("next_run_in", 0)))
        self.lbl_reopt_last_best.setText(f"{data.get('last_best_score', 0.0):.1f}")
        self.lbl_reopt_last_improv.setText(f"{data.get('last_improvement', 0.0):.1f}")
        self.lbl_reopt_last_swap.setText(data.get("last_swap_endpoint", "-"))

    @QtCore.pyqtSlot()
    def on_auto_escape_triggered(self):
        self.hud_overlay.show_alert("AUTO‑ESCAPE: Dead relay detected — switching…", QtGui.QColor(255, 0, 0), 2500, "red-shake")
        self.status_light.flash(QtGui.QColor(255, 0, 0), 1500, shake=True)
        self.status.log("[AUTO-ESCAPE] Dead relay detected — running emergency scan.")
        self.log_view.setPlainText("\n".join(self.status.log_lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        self._run_auto_escape_scan()

    def _run_auto_escape_scan(self):
        worker = AutoSearchWorker(self.relays_db, prometheus_mode=True)
        results_container = []
        def finished(results):
            results_container.extend(results)
        worker.finished.connect(finished)
        worker.run()

        if not results_container:
            self.hud_overlay.show_alert("NO VALID RELAYS FOUND — Retrying…", QtGui.QColor(255, 165, 0), 2000, "orange")
            return

        sorted_res = sorted(results_container, key=lambda r: r.score, reverse=True)
        for candidate in sorted_res:
            if self._switch_with_validation(RelayMode.AUTO, candidate.protocol, candidate.endpoint, "AUTO-ESCAPE"):
                return

        self.hud_overlay.show_alert("NO VALID RELAYS FOUND — Retrying…", QtGui.QColor(255, 165, 0), 2000, "orange")

    def _maybe_apply_app_preference(self, title: str):
        prefs = self.relays_db.get("app_prefs", {})
        title_l = title.lower()
        for pattern, pref in prefs.items():
            if pattern.lower() in title_l:
                mode_name = pref.get("mode", "AUTO")
                proto_name = pref.get("protocol", "TCP")
                try:
                    mode = RelayMode(mode_name)
                    protocol = Protocol(proto_name)
                except Exception:
                    continue
                best = self._pick_best_known_relay(protocol)
                if not best:
                    self.status.log(f"[APP] {pattern}: no suitable relay for {protocol.value}")
                    return
                self.status.log(f"[APP] {pattern} matched. Switching to {mode.value}/{protocol.value} @ {best['endpoint']}")
                self.log_view.setPlainText("\n".join(self.status.log_lines))
                self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
                self._switch_with_validation(mode, protocol, best["endpoint"], "APP-PREF")
                return

    def _pick_best_known_relay(self, protocol: Protocol):
        candidates = [
            r for r in self.relays_db.get("known_relays", [])
            if r.get("protocol") == protocol.value
        ]
        if not candidates:
            return None
        candidates = [r for r in candidates if r.get("trust", "YELLOW") != "RED"] or candidates
        return max(candidates, key=lambda r: (r.get("score", 0), r.get("reputation", 50.0)))

    def closeEvent(self, event):
        self.client.stop()
        self.ui_watcher.stop()
        self.reoptimizer.stop()
        self.gossip.stop()
        super().closeEvent(event)

# -----------------------------
# RELAY SERVERS (TCP / WS / HTTP)
# -----------------------------

def handle_relay_client(conn, addr):
    try:
        if AUTH_REQUIRED:
            data = conn.recv(1024)
            if not data.startswith(b"AUTH "):
                conn.close()
                return
            token = data.decode("utf-8").strip().split(" ", 1)[1]
            if token != AUTH_TOKEN:
                conn.sendall(b"ERR auth\n")
                conn.close()
                return
            conn.sendall(b"OK\n")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            if data.strip() == b"PING":
                conn.sendall(b"PONG")
            else:
                conn.sendall(b"OK")
    except:
        pass
    finally:
        conn.close()

def relay_server_main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((RELAY_HOST, RELAY_PORT))
    s.listen(50)
    if TLS_ENABLED:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(TLS_CERT_FILE, TLS_KEY_FILE)
        s = context.wrap_socket(s, server_side=True)
    while True:
        conn, addr = s.accept()
        t = threading.Thread(target=handle_relay_client, args=(conn, addr), daemon=True)
        t.start()

async def ws_handler(websocket, path):
    if AUTH_REQUIRED:
        token = websocket.request_headers.get("Authorization", "")
        if not token.startswith("Bearer "):
            await websocket.close()
            return
        if token.split(" ", 1)[1] != AUTH_TOKEN:
            await websocket.close()
            return
    try:
        async for message in websocket:
            if message.strip() == "PING":
                await websocket.send("PONG")
            else:
                await websocket.send("OK")
    except:
        pass

def ws_server_main():
    async def run():
        async with websockets.serve(ws_handler, "0.0.0.0", WS_RELAY_PORT):
            await asyncio.Future()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())

from http.server import BaseHTTPRequestHandler, HTTPServer

class HTTPRelayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if AUTH_REQUIRED:
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != AUTH_TOKEN:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"UNAUTHORIZED")
                return
        if self.path.startswith("/relay") or self.path.startswith("/cloud"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"PONG")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"NOT FOUND")

def http_server_main():
    server = HTTPServer(("0.0.0.0", HTTP_RELAY_PORT), HTTPRelayHandler)
    server.serve_forever()

def start_servers_in_background():
    t1 = threading.Thread(target=relay_server_main, daemon=True)
    t1.start()
    t2 = threading.Thread(target=ws_server_main, daemon=True)
    t2.start()
    t3 = threading.Thread(target=http_server_main, daemon=True)
    t3.start()

# -----------------------------
# ENTRY POINT
# -----------------------------

def main():
    start_servers_in_background()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

