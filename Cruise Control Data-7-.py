"""
Borg Routing Organism v3 + Qt Cockpit
- Auto Elevation + Auto Install + Auto Restart
- Crypto policy + Temporal Prediction + Peer Roles + Status API + Backoff + Clean Shutdown
- Multi-threaded DNS organ
- Qt/PySide6 cockpit talking to /atlas, /peers, /prediction, /mode, /backoff
"""

# ============================================================
#  AUTO-ELEVATION CHECK (Windows only)
# ============================================================

import os, sys, platform, ctypes

def ensure_admin():
    try:
        if platform.system().lower() == "windows":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",
                    sys.executable,
                    f'"{script}" {params}',
                    None,
                    1
                )
                sys.exit()
    except Exception as e:
        print(f"[Elevation] Failed to elevate: {e}")
        sys.exit()

ensure_admin()

# ============================================================
#  AUTO-INSTALL + AUTO-RESTART LOADER
# ============================================================

import importlib
import subprocess

def autoload_with_restart(modules: dict, critical: set):
    loaded = {}
    installed_now = set()

    for mod, pip_name in modules.items():
        try:
            loaded[mod] = importlib.import_module(mod)
        except ImportError:
            try:
                print(f"[AUTOLOADER] Installing missing dependency: {pip_name}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                loaded[mod] = importlib.import_module(mod)
                installed_now.add(mod)
                print(f"[AUTOLOADER] Installed and loaded: {mod}")
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install {pip_name}: {e}")
                loaded[mod] = None
        except Exception as e:
            print(f"[AUTOLOADER] Unexpected error importing {mod}: {e}")
            loaded[mod] = None

    if installed_now & critical:
        print("[AUTOLOADER] Critical dependency installed, restarting organism...")
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    return loaded

modules = autoload_with_restart(
    {
        "socket": "socket",
        "threading": "threading",
        "queue": "queue",
        "time": "time",
        "dataclasses": "dataclasses",
        "typing": "typing",
        "struct": "struct",
        "json": "json",
        "os": "os",
        "collections": "collections",
        "secrets": "secrets",
        "uiautomation": "uiautomation",
        "pythoncom": "pywin32",
        "nacl": "pynacl",
        "http.server": "http",
        "socketserver": "socketserver",
        "signal": "signal",
    },
    critical={"nacl", "pythoncom", "uiautomation"},
)

# ============================================================
#  IMPORTS (AFTER AUTOLOADER)
# ============================================================

socket = modules["socket"]
threading = modules["threading"]
queue = modules["queue"]
time = modules["time"]
struct = modules["struct"]
json = modules["json"]
os = modules["os"]
collections = modules["collections"]
secrets = modules["secrets"]
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import http.server
import socketserver
import signal

uiautomation = modules["uiautomation"]
pythoncom = modules["pythoncom"]
nacl = modules["nacl"]

USE_STRONG_CRYPTO = nacl is not None
CRYPTO_MODE = "auto"

if USE_STRONG_CRYPTO:
    from nacl import signing, exceptions, secret, utils
else:
    signing = None
    exceptions = None
    secret = None
    utils = None

# ============================================================
#  DATA MODELS
# ============================================================

@dataclass
class RouteCandidate:
    ip: str
    port: int
    family: int
    last_latency_ms: float
    last_checked: float
    healthy: bool
    source: str

@dataclass
class DomainRoute:
    domain: str
    primary_v4: Optional[RouteCandidate] = None
    backups_v4: List[RouteCandidate] = field(default_factory=list)
    primary_v6: Optional[RouteCandidate] = None
    backups_v6: List[RouteCandidate] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

@dataclass
class PeerInfo:
    node_id: str
    public_key: bytes
    host: str
    port: int
    initial_reputation: float
    role: str = "leaf"

# ============================================================
#  PEER REGISTRY & REPUTATION
# ============================================================

class PeerRegistry:
    def __init__(self, peers: Dict[str, PeerInfo]):
        self.peers = peers

    def get_peer(self, node_id: str) -> Optional[PeerInfo]:
        return self.peers.get(node_id)

class PeerReputation:
    def __init__(self):
        self.score: Dict[str, float] = defaultdict(lambda: 0.0)
        self.first_seen: Dict[str, float] = {}

    def set_initial(self, node_id: str, value: float):
        self.score[node_id] = value
        if node_id not in self.first_seen:
            self.first_seen[node_id] = time.time()

    def reward(self, node_id: str, amount: float = 1.0):
        self.score[node_id] += amount

    def penalize(self, node_id: str, amount: float = 1.0):
        self.score[node_id] -= amount

    def is_trusted(self, node_id: str, threshold: float = -5.0) -> bool:
        return self.score[node_id] > threshold

    def can_relay(self, peer: PeerInfo) -> bool:
        if peer.role not in ("relay", "authority"):
            return False
        return self.score[peer.node_id] >= 0.0

    def can_author(self, peer: PeerInfo) -> bool:
        if peer.role != "authority":
            return False
        age = time.time() - self.first_seen.get(peer.node_id, time.time())
        return self.score[peer.node_id] >= 1.0 and age > 60.0

# ============================================================
#  TRUST BUNDLE CREATOR & LOADER
# ============================================================

DEFAULT_TRUST_BUNDLE = "borg_trust_bundle_v3.json"

def create_default_trust_bundle(path: str):
    if USE_STRONG_CRYPTO:
        signing_key = signing.SigningKey.generate()
        verify_key = signing_key.verify_key
        priv_hex = signing_key.encode().hex()
        pub_hex = verify_key.encode().hex()
        swarm_key = secrets.token_bytes(32).hex()
    else:
        key_bytes = secrets.token_bytes(32)
        priv_hex = key_bytes.hex()
        pub_hex = key_bytes.hex()
        swarm_key = secrets.token_bytes(32).hex()

    node_id = "node-local-single-v3"

    bundle = {
        "node": {
            "node_id": node_id,
            "private_key": priv_hex,
            "public_key": pub_hex,
            "listen_ip": "0.0.0.0",
            "sync_port": 5055,
            "enable_ui_automation": False
        },
        "peers": [],
        "settings": {
            "probe_interval_sec": 20,
            "max_predictive_domains": 10,
            "max_peers": 32,
            "swarm_key": swarm_key,
            "predictor_path": "borg_predictor_v3.json",
            "crypto_mode": "auto",
            "status_port": 8053
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

def load_trust_bundle(path: str):
    if not os.path.exists(path):
        create_default_trust_bundle(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_cfg = data["node"]
    peers_cfg = data.get("peers", [])
    settings = data.get("settings", {})

    node_id = node_cfg["node_id"]
    priv_key_hex = node_cfg["private_key"]
    pub_key_hex = node_cfg["public_key"]
    listen_ip = node_cfg.get("listen_ip", "0.0.0.0")
    sync_port = node_cfg.get("sync_port", 5055)
    enable_ui_automation = node_cfg.get("enable_ui_automation", False)

    private_key = bytes.fromhex(priv_key_hex)
    public_key = bytes.fromhex(pub_key_hex)

    peers: Dict[str, PeerInfo] = {}
    reputation = PeerReputation()

    for p in peers_cfg:
        pid = p["node_id"]
        pk = bytes.fromhex(p["public_key"])
        role = p.get("role", "leaf")
        info = PeerInfo(
            node_id=pid,
            public_key=pk,
            host=p["host"],
            port=p["port"],
            initial_reputation=p.get("initial_reputation", 0.0),
            role=role,
        )
        peers[pid] = info
        reputation.set_initial(pid, info.initial_reputation)

    registry = PeerRegistry(peers)

    swarm_key_hex = settings.get("swarm_key")
    if not swarm_key_hex:
        swarm_key_hex = secrets.token_bytes(32).hex()
        settings["swarm_key"] = swarm_key_hex

    return {
        "node_id": node_id,
        "private_key_hex": priv_key_hex,
        "public_key": public_key,
        "listen_ip": listen_ip,
        "sync_port": sync_port,
        "peer_registry": registry,
        "peer_reputation": reputation,
        "settings": settings,
        "enable_ui_automation": enable_ui_automation,
    }

# ============================================================
#  SIGNING & ENCRYPTION
# ============================================================

def pseudo_sign(private_key: bytes, message: bytes) -> bytes:
    key = private_key or b"\x00"
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(message)])

def pseudo_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    expected = pseudo_sign(public_key, message)
    return expected == signature

def sign_update(node_id: str, private_key_hex: str, update_payload: dict) -> dict:
    body = {
        "node_id": node_id,
        "timestamp": time.time(),
        "payload": update_payload,
    }
    msg = json.dumps(body, sort_keys=True).encode("utf-8")

    if USE_STRONG_CRYPTO:
        signing_key = signing.SigningKey(bytes.fromhex(private_key_hex))
        sig = signing_key.sign(msg).signature
    else:
        sig = pseudo_sign(bytes.fromhex(private_key_hex), msg)

    body["signature"] = sig.hex()
    return body

def verify_update(peer_registry: PeerRegistry, reputation: PeerReputation, body: dict) -> Optional[dict]:
    node_id = body.get("node_id")
    sig_hex = body.get("signature")
    payload = body.get("payload")
    ts = body.get("timestamp")

    if not node_id or not sig_hex or payload is None:
        return None

    peer = peer_registry.get_peer(node_id)
    if not peer or not reputation.is_trusted(node_id):
        return None

    msg = json.dumps({
        "node_id": node_id,
        "timestamp": ts,
        "payload": payload,
    }, sort_keys=True).encode("utf-8")
    sig = bytes.fromhex(sig_hex)

    if USE_STRONG_CRYPTO:
        verify_key = signing.VerifyKey(peer.public_key)
        try:
            verify_key.verify(msg, sig)
        except exceptions.BadSignatureError:
            return None
    else:
        if not pseudo_verify(peer.public_key, msg, sig):
            return None

    return payload

class SwarmCrypto:
    def __init__(self, swarm_key_hex: str):
        self.swarm_key_hex = swarm_key_hex
        if USE_STRONG_CRYPTO:
            self.box = secret.SecretBox(bytes.fromhex(swarm_key_hex))
        else:
            self.box = None

    def encrypt(self, body: dict) -> bytes:
        plaintext = json.dumps(body).encode("utf-8")
        if USE_STRONG_CRYPTO:
            nonce = utils.random(secret.SecretBox.NONCE_SIZE)
            ciphertext = self.box.encrypt(plaintext, nonce).ciphertext
            return nonce + ciphertext
        else:
            return plaintext

    def decrypt(self, data: bytes) -> Optional[dict]:
        try:
            if USE_STRONG_CRYPTO:
                nonce = data[:secret.SecretBox.NONCE_SIZE]
                ciphertext = data[secret.SecretBox.NONCE_SIZE:]
                plaintext = self.box.decrypt(ciphertext, nonce)
                return json.loads(plaintext.decode("utf-8"))
            else:
                return json.loads(data.decode("utf-8"))
        except Exception:
            return None

# ============================================================
#  PREDICTIVE DOMAIN TRACKER
# ============================================================

class PredictiveDomainTracker:
    def __init__(self, path="borg_predictor_v3.json", max_history: int = 5000):
        self.path = path
        self.counts_global = defaultdict(int)
        self.counts_by_hour = defaultdict(int)
        self.counts_by_dow = defaultdict(int)
        self.max_history = max_history
        self.total_events = 0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cg = data.get("counts_global", {})
                ch = data.get("counts_by_hour", {})
                cd = data.get("counts_by_dow", {})
                for k, v in cg.items():
                    self.counts_global[k] = int(v)
                for k, v in ch.items():
                    hour, domain = k.split("|", 1)
                    self.counts_by_hour[(int(hour), domain)] = int(v)
                for k, v in cd.items():
                    dow, domain = k.split("|", 1)
                    self.counts_by_dow[(int(dow), domain)] = int(v)
                self.total_events = sum(self.counts_global.values())
            except Exception:
                pass

    def _save(self):
        try:
            cg = dict(self.counts_global)
            ch = {f"{h}|{d}": c for (h, d), c in self.counts_by_hour.items()}
            cd = {f"{w}|{d}": c for (w, d), c in self.counts_by_dow.items()}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "counts_global": cg,
                    "counts_by_hour": ch,
                    "counts_by_dow": cd
                }, f)
        except Exception:
            pass

    def record(self, domain: str):
        now = time.localtime()
        hour = now.tm_hour
        dow = now.tm_wday

        self.counts_global[domain] += 1
        self.counts_by_hour[(hour, domain)] += 1
        self.counts_by_dow[(dow, domain)] += 1
        self.total_events += 1

        if self.total_events > self.max_history:
            for k in list(self.counts_global.keys()):
                self.counts_global[k] = max(1, self.counts_global[k] // 2)
            for k in list(self.counts_by_hour.keys()):
                self.counts_by_hour[k] = max(1, self.counts_by_hour[k] // 2)
            for k in list(self.counts_by_dow.keys()):
                self.counts_by_dow[k] = max(1, self.counts_by_dow[k] // 2)
            self.total_events = sum(self.counts_global.values())
        self._save()

    def top_global(self, n: int = 10) -> List[str]:
        return [d for d, _ in sorted(self.counts_global.items(), key=lambda x: x[1], reverse=True)[:n]]

    def top_for_current_hour(self, n: int = 10) -> List[str]:
        now = time.localtime()
        hour = now.tm_hour
        items = [(d, c) for (h, d), c in self.counts_by_hour.items() if h == hour]
        items.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in items[:n]]

    def top_for_current_dow(self, n: int = 10) -> List[str]:
        now = time.localtime()
        dow = now.tm_wday
        items = [(d, c) for (w, d), c in self.counts_by_dow.items() if w == dow]
        items.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in items[:n]]

# ============================================================
#  QUEEN ROUTER
# ============================================================

class QueenRouter:
    def __init__(self, node_id: str, private_key_hex: str,
                 peer_registry: PeerRegistry, reputation: PeerReputation,
                 swarm_crypto: SwarmCrypto,
                 listen_ip: str = "0.0.0.0", sync_port: int = 5055,
                 predictor_path: str = "borg_predictor_v3.json",
                 max_predictive_domains: int = 10):
        self.node_id = node_id
        self.private_key_hex = private_key_hex
        self.peer_registry = peer_registry
        self.reputation = reputation
        self.swarm_crypto = swarm_crypto

        self._atlas: Dict[str, DomainRoute] = {}
        self._updates = queue.Queue()
        self._lock = threading.RLock()
        self._running = False

        self.listen_ip = listen_ip
        self.sync_port = sync_port

        self.predictor = PredictiveDomainTracker(path=predictor_path)
        self.max_predictive_domains = max_predictive_domains

    def start(self):
        self._running = True
        threading.Thread(target=self._update_loop, daemon=True).start()
        threading.Thread(target=self._sync_listener, daemon=True).start()

    def stop(self):
        self._running = False

    def _update_loop(self):
        while self._running:
            try:
                update, meta = self._updates.get(timeout=1.0)
                self._apply_update(update, from_peer=meta["from_peer"], peer_id=meta.get("peer_id"))
            except queue.Empty:
                continue

    def _apply_update(self, update, from_peer: bool, peer_id: Optional[str]):
        domain = update["domain"]
        candidates = update["candidates"]

        with self._lock:
            existing = self._atlas.get(domain)
            if not existing:
                existing = DomainRoute(domain=domain)
                self._atlas[domain] = existing

            v4_list: Dict[str, RouteCandidate] = {}
            v6_list: Dict[str, RouteCandidate] = {}

            def add_candidate(c: RouteCandidate):
                if c.family == socket.AF_INET:
                    v4_list[c.ip] = c
                elif c.family == socket.AF_INET6:
                    v6_list[c.ip] = c

            if existing.primary_v4:
                add_candidate(existing.primary_v4)
            for c in existing.backups_v4:
                add_candidate(c)
            if existing.primary_v6:
                add_candidate(existing.primary_v6)
            for c in existing.backups_v6:
                add_candidate(c)

            for c in candidates:
                ip = c["ip"]
                port = c["port"]
                family = c["family"]
                last_checked = c["last_checked"]
                latency = c["last_latency_ms"]
                healthy = c["healthy"]
                source = c.get("source", "local")

                rc = RouteCandidate(
                    ip=ip,
                    port=port,
                    family=family,
                    last_latency_ms=latency,
                    last_checked=last_checked,
                    healthy=healthy,
                    source=source
                )

                if family == socket.AF_INET:
                    existing_c = v4_list.get(ip)
                    if not existing_c or last_checked > existing_c.last_checked:
                        v4_list[ip] = rc
                else:
                    existing_c = v6_list.get(ip)
                    if not existing_c or last_checked > existing_c.last_checked:
                        v6_list[ip] = rc

            def sort_and_promote(cmap: Dict[str, RouteCandidate]) -> Tuple[Optional[RouteCandidate], List[RouteCandidate]]:
                all_c = list(cmap.values())
                all_c = [c for c in all_c if c.healthy]
                all_c.sort(key=lambda x: x.last_latency_ms)
                if not all_c:
                    return None, []
                local = [c for c in all_c if c.source == "local"]
                if local:
                    primary = local[0]
                    backups = local[1:] + [c for c in all_c if c.source == "peer"]
                else:
                    primary = None
                    backups = all_c
                return primary, backups

            primary_v4, backups_v4 = sort_and_promote(v4_list)
            primary_v6, backups_v6 = sort_and_promote(v6_list)

            existing.primary_v4 = primary_v4 or existing.primary_v4
            existing.backups_v4 = backups_v4
            existing.primary_v6 = primary_v6 or existing.primary_v6
            existing.backups_v6 = backups_v6
            existing.last_updated = time.time()

        if not from_peer:
            self._broadcast_update(update)
        else:
            if peer_id:
                self.reputation.reward(peer_id, 0.1)

    def submit_update(self, update, from_peer: bool = False, peer_id: Optional[str] = None):
        self._updates.put((update, {"from_peer": from_peer, "peer_id": peer_id}))

    def resolve(self, domain: str, family: int) -> Optional[str]:
        with self._lock:
            route = self._atlas.get(domain)
            if not route:
                return None
            if family == socket.AF_INET:
                if route.primary_v4 and route.primary_v4.healthy:
                    self.predictor.record(domain)
                    return route.primary_v4.ip
            else:
                if route.primary_v6 and route.primary_v6.healthy:
                    self.predictor.record(domain)
                    return route.primary_v6.ip
            return None

    def snapshot_atlas(self):
        with self._lock:
            out = {}
            for d, r in self._atlas.items():
                out[d] = {
                    "primary_v4": r.primary_v4.ip if r.primary_v4 else None,
                    "primary_v6": r.primary_v6.ip if r.primary_v6 else None,
                    "backups_v4": [c.ip for c in r.backups_v4],
                    "backups_v6": [c.ip for c in r.backups_v6],
                    "last_updated": r.last_updated,
                }
            return out

    def _sync_listener(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.listen_ip, self.sync_port))
        s.listen(5)
        while self._running:
            try:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_peer_conn, args=(conn,), daemon=True).start()
            except Exception:
                time.sleep(0.1)
        try:
            s.close()
        except Exception:
            pass

    def _handle_peer_conn(self, conn: socket.socket):
        with conn:
            try:
                data = conn.recv(65535)
                if not data:
                    return
                body = self.swarm_crypto.decrypt(data)
                if body is None:
                    return
                node_id = body.get("node_id")
                peer = self.peer_registry.get_peer(node_id) if node_id else None
                if not peer or not self.reputation.can_author(peer):
                    return
                payload = verify_update(self.peer_registry, self.reputation, body)
                if payload is None:
                    return
                self.submit_update(payload, from_peer=True, peer_id=node_id)
            except Exception:
                return

    def _broadcast_update(self, update):
        if not self.peer_registry.peers:
            return
        body = sign_update(self.node_id, self.private_key_hex, update)
        payload = self.swarm_crypto.encrypt(body)

        for peer_id, info in self.peer_registry.peers.items():
            if not self.reputation.can_relay(info):
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect((info.host, info.port))
                s.sendall(payload)
                s.close()
            except Exception:
                continue

# ============================================================
#  WORKER SCANNER (WITH BACKOFF)
# ============================================================

class WorkerScanner:
    def __init__(self, queen: QueenRouter, base_domains: List[str],
                 interval_sec: int = 15, max_predictive_domains: int = 10):
        self.queen = queen
        self.base_domains = base_domains
        self.interval_sec = interval_sec
        self.max_predictive_domains = max_predictive_domains
        self._running = False

        self.last_probe_time: Dict[str, float] = {}
        self.failure_count: Dict[str, int] = defaultdict(int)
        self.next_allowed_time: Dict[str, float] = defaultdict(lambda: 0.0)

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _can_probe(self, domain: str) -> bool:
        now = time.time()
        return now >= self.next_allowed_time[domain]

    def _record_result(self, domain: str, success: bool):
        now = time.time()
        self.last_probe_time[domain] = now
        if success:
            self.failure_count[domain] = 0
            self.next_allowed_time[domain] = now + self.interval_sec
        else:
            self.failure_count[domain] += 1
            backoff = min(300, self.interval_sec * (2 ** self.failure_count[domain]))
            self.next_allowed_time[domain] = now + backoff

    def _probe_domain(self, domain: str):
        candidates = []
        success_any = False
        try:
            infos = socket.getaddrinfo(domain, 443, proto=socket.IPPROTO_TCP)
            for family, socktype, proto, canonname, sockaddr in infos:
                ip = sockaddr[0]
                port = sockaddr[1]
                start = time.time()
                s = socket.socket(family, socktype, proto)
                s.settimeout(1.0)
                try:
                    s.connect((ip, port))
                    latency = (time.time() - start) * 1000.0
                    healthy = True
                    success_any = True
                except Exception:
                    latency = float("inf")
                    healthy = False
                finally:
                    s.close()
                candidates.append({
                    "ip": ip,
                    "port": port,
                    "family": family,
                    "last_latency_ms": latency,
                    "last_checked": time.time(),
                    "healthy": healthy,
                    "source": "local",
                })
        except Exception:
            pass
        self._record_result(domain, success_any)
        return {"domain": domain, "candidates": candidates}

    def _loop(self):
        while self._running:
            predictive_global = self.queen.predictor.top_global(self.max_predictive_domains)
            predictive_hour = self.queen.predictor.top_for_current_hour(self.max_predictive_domains)
            predictive_dow = self.queen.predictor.top_for_current_dow(self.max_predictive_domains)

            domains = set(self.base_domains)
            domains.update(predictive_global)
            domains.update(predictive_hour)
            domains.update(predictive_dow)

            for d in domains:
                if not self._running:
                    break
                if not self._can_probe(d):
                    continue
                update = self._probe_domain(d)
                self.queen.submit_update(update, from_peer=False)
            time.sleep(self.interval_sec)

    def snapshot_backoff(self):
        out = {}
        for d in set(list(self.last_probe_time.keys()) + list(self.next_allowed_time.keys())):
            out[d] = {
                "last_probe": self.last_probe_time.get(d),
                "next_allowed": self.next_allowed_time.get(d),
                "failure_count": self.failure_count.get(d, 0),
            }
        return out

# ============================================================
#  OPTIONAL UI AUTOMATION PREDICTOR
# ============================================================

class UIAutomationWatcher:
    def __init__(self, queen: QueenRouter, interval_sec: int = 5, enabled: bool = False):
        self.queen = queen
        self.interval_sec = interval_sec
        self._running = False
        self.enabled = enabled

    def start(self):
        if not self.enabled:
            return
        if uiautomation is None or pythoncom is None:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        pythoncom.CoInitialize()
        try:
            while self._running:
                try:
                    window = uiautomation.GetForegroundControl()
                    name = window.Name or ""
                    with self.queen._lock:
                        for domain in list(self.queen._atlas.keys()):
                            if domain in name:
                                self.queen.predictor.record(domain)
                except Exception:
                    pass
                time.sleep(self.interval_sec)
        finally:
            pythoncom.CoUninitialize()

# ============================================================
#  MULTI-THREADED DNS SERVER
# ============================================================

TOP_DOMAINS = {
    "yahoo.com",
    "google.com",
    "youtube.com",
    "microsoft.com",
}

class LocalDNSServer:
    def __init__(self, queen: QueenRouter, listen_ip="127.0.0.1", port=53, max_workers: int = 64):
        self.queen = queen
        self.listen_ip = listen_ip
        self.port = port
        self._running = False
        self._sock = None
        self._max_workers = max_workers
        self._active_workers = 0
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        threading.Thread(target=self._serve, daemon=True).start()

    def stop(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

    def _parse_query(self, data: bytes) -> Tuple[str, int]:
        i = 12
        labels = []
        length = data[i]
        while length != 0:
            i += 1
            labels.append(data[i:i+length].decode())
            i += length
            length = data[i]
        domain = ".".join(labels)
        qtype = struct.unpack("!H", data[i+1:i+3])[0]
        return domain, qtype

    def _build_response(self, query: bytes, ip: str, qtype: int) -> bytes:
        tid = query[:2]
        flags = b"\x81\x80"
        qdcount = b"\x00\x01"
        ancount = b"\x00\x01"
        nscount = b"\x00\x00"
        arcount = b"\x00\x00"
        header = tid + flags + qdcount + ancount + nscount + arcount
        question = query[12:]
        answer_name = b"\xc0\x0c"
        if qtype == 28:
            answer_type = b"\x00\x1c"
            rdata = socket.inet_pton(socket.AF_INET6, ip)
        else:
            answer_type = b"\x00\x01"
            rdata = socket.inet_aton(ip)
        answer_class = b"\x00\x01"
        ttl = b"\x00\x00\x00\x3c"
        rdlength = struct.pack("!H", len(rdata))
        answer = answer_name + answer_type + answer_class + ttl + rdlength + rdata
        return header + question + answer

    def _handle_request(self, data: bytes, addr):
        try:
            domain, qtype = self._parse_query(data)

            if qtype == 28:
                ip = self.queen.resolve(domain, socket.AF_INET6)
            else:
                ip = self.queen.resolve(domain, socket.AF_INET)

            if ip:
                response = self._build_response(data, ip, qtype)
                self._sock.sendto(response, addr)
                return

            try:
                if qtype == 28:
                    infos = socket.getaddrinfo(domain, 443, socket.AF_INET6, socket.SOCK_STREAM)
                    ip = infos[0][4][0]
                    response = self._build_response(data, ip, qtype)
                else:
                    upstream_ip = socket.gethostbyname(domain)
                    response = self._build_response(data, upstream_ip, qtype)
                self._sock.sendto(response, addr)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            with self._lock:
                self._active_workers -= 1

    def _serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.listen_ip, self.port))
        self._sock = sock

        while self._running:
            try:
                data, addr = sock.recvfrom(512)
            except Exception:
                if not self._running:
                    break
                time.sleep(0.05)
                continue

            with self._lock:
                if self._active_workers >= self._max_workers:
                    # Drop packet under extreme load to avoid thread explosion
                    continue
                self._active_workers += 1

            t = threading.Thread(target=self._handle_request, args=(data, addr), daemon=True)
            t.start()

        try:
            sock.close()
        except Exception:
            pass

# ============================================================
#  STATUS HTTP/JSON SERVER
# ============================================================

class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/atlas":
            data = self.server.organism["queen"].snapshot_atlas()
        elif self.path == "/peers":
            reg: PeerRegistry = self.server.organism["peer_registry"]
            rep: PeerReputation = self.server.organism["peer_reputation"]
            peers = {}
            for pid, info in reg.peers.items():
                peers[pid] = {
                    "host": info.host,
                    "port": info.port,
                    "role": info.role,
                    "reputation": rep.score.get(pid, 0.0),
                    "first_seen": rep.first_seen.get(pid, None),
                }
            data = peers
        elif self.path == "/prediction":
            pred: PredictiveDomainTracker = self.server.organism["queen"].predictor
            data = {
                "top_global": pred.top_global(20),
                "top_hour": pred.top_for_current_hour(20),
                "top_dow": pred.top_for_current_dow(20),
            }
        elif self.path == "/mode":
            data = {
                "crypto_mode": CRYPTO_MODE,
                "use_strong_crypto": USE_STRONG_CRYPTO,
                "ui_automation_enabled": self.server.organism["enable_ui_automation"],
            }
        elif self.path == "/backoff":
            worker = self.server.organism["worker"]
            data = worker.snapshot_backoff()
        else:
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

class StatusServer:
    def __init__(self, organism: dict, port: int = 8053):
        self.organism = organism
        self.port = port
        self._running = False
        self._httpd = None

    def start(self):
        self._running = True
        def run():
            with socketserver.TCPServer(("127.0.0.1", self.port), StatusHandler) as httpd:
                httpd.organism = self.organism
                self._httpd = httpd
                while self._running:
                    httpd.handle_request()
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self._running = False
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
        except Exception:
            pass

# ============================================================
#  ORGANISM MAIN
# ============================================================

def run_organism():
    global USE_STRONG_CRYPTO, CRYPTO_MODE

    trust_path = DEFAULT_TRUST_BUNDLE
    bundle = load_trust_bundle(trust_path)

    CRYPTO_MODE = bundle["settings"].get("crypto_mode", "auto")

    if CRYPTO_MODE == "compat":
        if USE_STRONG_CRYPTO:
            print("[Crypto] Policy: COMPAT. Forcing weak mode even though PyNaCl is available.")
        USE_STRONG_CRYPTO = False
    elif CRYPTO_MODE == "strong_only":
        if not USE_STRONG_CRYPTO:
            print("[Crypto] Policy: STRONG_ONLY but PyNaCl is unavailable. Exiting.")
            sys.exit(1)
        else:
            print("[Crypto] Policy: STRONG_ONLY and PyNaCl is available.")
    else:
        if USE_STRONG_CRYPTO:
            print("[Crypto] Policy: AUTO, using STRONG mode (PyNaCl present).")
        else:
            print("[Crypto] Policy: AUTO, falling back to COMPAT (PyNaCl missing).")

    swarm_crypto = SwarmCrypto(bundle["settings"]["swarm_key"])

    queen = QueenRouter(
        node_id=bundle["node_id"],
        private_key_hex=bundle["private_key_hex"],
        peer_registry=bundle["peer_registry"],
        reputation=bundle["peer_reputation"],
        swarm_crypto=swarm_crypto,
        listen_ip=bundle["listen_ip"],
        sync_port=bundle["sync_port"],
        predictor_path=bundle["settings"].get("predictor_path", "borg_predictor_v3.json"),
        max_predictive_domains=bundle["settings"].get("max_predictive_domains", 10),
    )
    queen.start()

    worker = WorkerScanner(
        queen,
        base_domains=list(TOP_DOMAINS),
        interval_sec=bundle["settings"].get("probe_interval_sec", 20),
        max_predictive_domains=bundle["settings"].get("max_predictive_domains", 10),
    )
    worker.start()

    ui_watcher = UIAutomationWatcher(
        queen,
        interval_sec=5,
        enabled=bundle["enable_ui_automation"],
    )
    ui_watcher.start()

    dns_server = LocalDNSServer(queen)
    dns_server.start()

    organism = {
        "queen": queen,
        "peer_registry": bundle["peer_registry"],
        "peer_reputation": bundle["peer_reputation"],
        "enable_ui_automation": bundle["enable_ui_automation"],
        "worker": worker,
    }
    status_port = bundle["settings"].get("status_port", 8053)
    status_server = StatusServer(organism, port=status_port)
    status_server.start()

    print("Borg Routing Organism v3 running.")
    print("Local DNS on 127.0.0.1:53 (A + AAAA, multi-threaded)")
    print(f"Trust bundle: {trust_path}")
    print(f"Status API: http://127.0.0.1:{status_port}/(atlas|peers|prediction|mode|backoff)")
    if USE_STRONG_CRYPTO:
        print("Crypto mode: STRONG (Ed25519 + encrypted P2P)")
    else:
        print("Crypto mode: COMPAT (pseudo-sign + unencrypted P2P)")
    print("Press Ctrl+C to exit gracefully.")

    def handle_sigint(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping organism...")
        worker.stop()
        dns_server.stop()
        ui_watcher.stop()
        status_server.stop()
        queen.stop()
        try:
            queen.predictor._save()
        except Exception:
            pass
        time.sleep(1)
        print("[Shutdown] Done.")

# ============================================================
#  QT COCKPIT
# ============================================================

def run_cockpit():
    import requests
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QPalette, QFont
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
        QTableWidgetItem, QLabel, QHeaderView, QGroupBox, QGridLayout
    )

    STATUS_BASE = "http://127.0.0.1:8053"

    def safe_get(path, timeout=0.5):
        url = STATUS_BASE + path
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def set_dark_palette(app: QApplication):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(15, 15, 15))
        palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(30, 30, 30))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    def make_group(title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                margin-top: 6px;
                color: #ccc;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px 0 3px;
            }
        """)
        return box

    def color_item(text: str, color: QColor) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setForeground(color)
        return item

    class AtlasPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.table = QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(["Domain", "Primary v4", "Primary v6", "Backups v4", "Backups v6", "Backoff"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)
            self.table.setShowGrid(True)
            self.table.setSelectionMode(self.table.NoSelection)
            self.table.setEditTriggers(self.table.NoEditTriggers)
            layout.addWidget(self.table)

        def update_data(self, atlas: dict, backoff: dict):
            if atlas is None:
                self.table.setRowCount(0)
                return

            domains = sorted(atlas.keys(), key=lambda d: atlas[d].get("last_updated", 0), reverse=True)
            self.table.setRowCount(len(domains))

            now = time.time()
            for row, d in enumerate(domains):
                r = atlas[d]
                p4 = r.get("primary_v4")
                p6 = r.get("primary_v6")
                b4 = r.get("backups_v4", [])
                b6 = r.get("backups_v6", [])
                ts = r.get("last_updated", 0)
                age = now - ts

                bo = backoff.get(d, {}) if backoff else {}
                last_probe = bo.get("last_probe")
                next_allowed = bo.get("next_allowed")
                failure_count = bo.get("failure_count", 0)

                tip = f"Last updated: {time.ctime(ts) if ts else '-'}\n"
                tip += f"Last probe: {time.ctime(last_probe) if last_probe else '-'}\n"
                tip += f"Next allowed: {time.ctime(next_allowed) if next_allowed else '-'}\n"
                tip += f"Failure count: {failure_count}"

                def ip_color(ip):
                    if not ip:
                        return QColor(200, 80, 80)
                    if age < 30:
                        return QColor(80, 200, 120)
                    elif age < 120:
                        return QColor(220, 180, 80)
                    else:
                        return QColor(200, 80, 80)

                in_backoff = next_allowed is not None and next_allowed > now + 1

                dom_text = ("⏱ " if in_backoff else "") + d
                dom_item = color_item(dom_text, QColor(200, 200, 200))
                dom_item.setToolTip(tip)
                self.table.setItem(row, 0, dom_item)

                self.table.setItem(row, 1, color_item(p4 or "-", ip_color(p4)))
                self.table.setItem(row, 2, color_item(p6 or "-", ip_color(p6)))
                self.table.setItem(row, 3, color_item(str(len(b4)), QColor(150, 150, 200)))
                self.table.setItem(row, 4, color_item(str(len(b6)), QColor(150, 150, 200)))
                self.table.setItem(row, 5, color_item("BACKOFF" if in_backoff else "-", QColor(220, 180, 80) if in_backoff else QColor(100, 100, 100)))

    class PeersPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.table = QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(["Node", "Role", "Reputation", "First Seen", "Relay?", "Authority?"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)
            self.table.setShowGrid(True)
            self.table.setSelectionMode(self.table.NoSelection)
            self.table.setEditTriggers(self.table.NoEditTriggers)
            layout.addWidget(self.table)

        def update_data(self, peers: dict):
            if peers is None:
                self.table.setRowCount(0)
                return

            ids = sorted(peers.keys())
            self.table.setRowCount(len(ids))
            now = time.time()

            for row, pid in enumerate(ids):
                info = peers[pid]
                short_id = pid[:10] + "…" if len(pid) > 10 else pid
                role = info.get("role", "leaf")
                rep = info.get("reputation", 0.0)
                first_seen = info.get("first_seen", None)
                age_str = "-"
                if first_seen:
                    age = now - first_seen
                    if age < 120:
                        age_str = f"{int(age)}s"
                    elif age < 7200:
                        age_str = f"{int(age/60)}m"
                    else:
                        age_str = f"{int(age/3600)}h"

                tip = f"Node ID: {pid}\nReputation: {rep:.4f}\nFirst seen: {time.ctime(first_seen) if first_seen else '-'}"

                node_item = color_item(short_id, QColor(200, 200, 200))
                node_item.setToolTip(tip)
                self.table.setItem(row, 0, node_item)

                if role == "leaf":
                    rc = QColor(160, 160, 160)
                elif role == "relay":
                    rc = QColor(80, 160, 220)
                else:
                    rc = QColor(220, 190, 80)
                self.table.setItem(row, 1, color_item(role, rc))

                rep_color = QColor(80, 200, 120) if rep >= 0 else QColor(200, 80, 80)
                self.table.setItem(row, 2, color_item(f"{rep:.2f}", rep_color))
                self.table.setItem(row, 3, color_item(age_str, QColor(180, 180, 180)))

                relay_ok = role in ("relay", "authority") and rep >= 0
                auth_ok = role == "authority" and rep >= 1.0

                self.table.setItem(row, 4, color_item("●" if relay_ok else "○",
                                                      QColor(80, 160, 220) if relay_ok else QColor(80, 80, 80)))
                self.table.setItem(row, 5, color_item("●" if auth_ok else "○",
                                                      QColor(220, 190, 80) if auth_ok else QColor(80, 80, 80)))

    class PredictionPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QGridLayout(self)

            self.global_box, self.global_table = self._make_table("Top Global")
            self.hour_box, self.hour_table = self._make_table("Top Hour")
            self.dow_box, self.dow_table = self._make_table("Top Day-of-Week")

            layout.addWidget(self.global_box, 0, 0)
            layout.addWidget(self.hour_box, 0, 1)
            layout.addWidget(self.dow_box, 1, 0, 1, 2)

        def _make_table(self, title):
            box = make_group(title)
            v = QVBoxLayout(box)
            table = QTableWidget(0, 1)
            table.setHorizontalHeaderLabels(["Domain"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setShowGrid(True)
            table.setSelectionMode(table.NoSelection)
            table.setEditTriggers(table.NoEditTriggers)
            v.addWidget(table)
            return box, table

        def update_data(self, prediction: dict):
            if prediction is None:
                for t in (self.global_table, self.hour_table, self.dow_table):
                    t.setRowCount(0)
                return

            def fill(table, items):
                table.setRowCount(len(items))
                for i, d in enumerate(items):
                    item = color_item(d, QColor(200, 200, 200))
                    item.setToolTip(f"Rank: {i+1}")
                    table.setItem(i, 0, item)

            fill(self.global_table, prediction.get("top_global", []))
            fill(self.hour_table, prediction.get("top_hour", []))
            fill(self.dow_table, prediction.get("top_dow", []))

    class ModePanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)

            self.crypto_label = QLabel("Crypto: -")
            self.crypto_label.setAlignment(Qt.AlignLeft)
            self.crypto_label.setStyleSheet("color: #ccc; font-weight: bold;")

            self.details_label = QLabel("")
            self.details_label.setAlignment(Qt.AlignLeft)
            self.details_label.setStyleSheet("color: #aaa;")

            self.ui_label = QLabel("UI Automation: -")
            self.ui_label.setAlignment(Qt.AlignLeft)
            self.ui_label.setStyleSheet("color: #ccc;")

            layout.addWidget(self.crypto_label)
            layout.addWidget(self.details_label)
            layout.addWidget(self.ui_label)
            layout.addStretch(1)

        def update_data(self, mode: dict):
            if mode is None:
                self.crypto_label.setText("Crypto: unavailable")
                self.details_label.setText("")
                self.ui_label.setText("UI Automation: unknown")
                return

            crypto_mode = mode.get("crypto_mode", "auto")
            strong = mode.get("use_strong_crypto", False)
            ui_auto = mode.get("ui_automation_enabled", False)

            if crypto_mode == "strong_only":
                if strong:
                    text = "Crypto: STRONG_ONLY (Ed25519 + encrypted P2P)"
                    color = "#4caf50"
                else:
                    text = "Crypto: STRONG_ONLY (but strong crypto unavailable!)"
                    color = "#f44336"
            elif crypto_mode == "compat":
                text = "Crypto: COMPAT (pseudo-sign + unencrypted P2P)"
                color = "#ff9800"
            else:
                if strong:
                    text = "Crypto: AUTO (using STRONG)"
                    color = "#4caf50"
                else:
                    text = "Crypto: AUTO (falling back to COMPAT)"
                    color = "#ff9800"

            explain = {
                "strong_only": "Requires PyNaCl. Fails fast if strong crypto unavailable.",
                "compat": "Always uses pseudo-sign + unencrypted P2P.",
                "auto": "Uses strong crypto if available, else compat.",
            }.get(crypto_mode, "Unknown mode.")

            self.crypto_label.setText(text)
            self.crypto_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.crypto_label.setToolTip(explain)

            self.details_label.setText(f"Strong crypto active: {'yes' if strong else 'no'}")
            self.ui_label.setText(f"UI Automation: {'enabled' if ui_auto else 'disabled'}")

    class CockpitWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Borg Routing Cockpit v3")
            self.resize(1600, 900)

            font = QFont("Consolas", 9)
            self.setFont(font)

            main_layout = QHBoxLayout(self)

            atlas_box = make_group("Atlas (Routes)")
            atlas_layout = QVBoxLayout(atlas_box)
            self.atlas_panel = AtlasPanel()
            atlas_layout.addWidget(self.atlas_panel)

            peers_box = make_group("Peers (Swarm)")
            peers_layout = QVBoxLayout(peers_box)
            self.peers_panel = PeersPanel()
            peers_layout.addWidget(self.peers_panel)

            pred_box = make_group("Prediction Engine")
            pred_layout = QVBoxLayout(pred_box)
            self.pred_panel = PredictionPanel()
            pred_layout.addWidget(self.pred_panel)

            mode_box = make_group("System Mode & Health")
            mode_layout = QVBoxLayout(mode_box)
            self.mode_panel = ModePanel()
            mode_layout.addWidget(self.mode_panel)

            main_layout.addWidget(atlas_box, 3)
            main_layout.addWidget(peers_box, 3)
            main_layout.addWidget(pred_box, 3)
            main_layout.addWidget(mode_box, 2)

            self.timer_atlas = QTimer(self)
            self.timer_atlas.timeout.connect(self.refresh_atlas)
            self.timer_atlas.start(1000)

            self.timer_peers = QTimer(self)
            self.timer_peers.timeout.connect(self.refresh_peers)
            self.timer_peers.start(2000)

            self.timer_pred = QTimer(self)
            self.timer_pred.timeout.connect(self.refresh_prediction)
            self.timer_pred.start(3000)

            self.timer_mode = QTimer(self)
            self.timer_mode.timeout.connect(self.refresh_mode)
            self.timer_mode.start(5000)

        def refresh_atlas(self):
            atlas = safe_get("/atlas")
            backoff = safe_get("/backoff")
            self.atlas_panel.update_data(atlas, backoff or {})

        def refresh_peers(self):
            data = safe_get("/peers")
            self.peers_panel.update_data(data)

        def refresh_prediction(self):
            data = safe_get("/prediction")
            self.pred_panel.update_data(data)

        def refresh_mode(self):
            data = safe_get("/mode")
            self.mode_panel.update_data(data)

    app = QApplication(sys.argv)
    set_dark_palette(app)
    w = CockpitWindow()
    w.show()
    sys.exit(app.exec())

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if "--cockpit" in sys.argv:
        run_cockpit()
    else:
        run_organism()

