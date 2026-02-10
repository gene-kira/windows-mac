"""
Borg Routing Organism v2 - Auto Elevation + Auto Install + Auto Restart
Queen + Workers + DNS (A/AAAA) + P2P + Trust + (Ed25519+Encrypted OR Fallback) + Reputation + Persistent Prediction + Optional UI Automation
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
    """
    modules: { module_name: pip_name }
    critical: set of module_names that, if installed now, should trigger a restart
    """
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

    # If any critical module was just installed, restart the script
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

uiautomation = modules["uiautomation"]
pythoncom = modules["pythoncom"]
nacl = modules["nacl"]

USE_STRONG_CRYPTO = nacl is not None

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
    family: int          # socket.AF_INET or socket.AF_INET6
    last_latency_ms: float
    last_checked: float
    healthy: bool
    source: str          # "local" or "peer"

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


# ============================================================
#  PEER REGISTRY & REPUTATION
# ============================================================

class PeerRegistry:
    def __init__(self, peers: Dict[str, PeerInfo]):
        self.peers = peers  # node_id -> PeerInfo

    def get_peer(self, node_id: str) -> Optional[PeerInfo]:
        return self.peers.get(node_id)


class PeerReputation:
    def __init__(self):
        self.score: Dict[str, float] = defaultdict(lambda: 0.0)

    def set_initial(self, node_id: str, value: float):
        self.score[node_id] = value

    def reward(self, node_id: str, amount: float = 1.0):
        self.score[node_id] += amount

    def penalize(self, node_id: str, amount: float = 1.0):
        self.score[node_id] -= amount

    def is_trusted(self, node_id: str, threshold: float = -5.0) -> bool:
        return self.score[node_id] > threshold


# ============================================================
#  TRUST BUNDLE CREATOR & LOADER
# ============================================================

DEFAULT_TRUST_BUNDLE = "borg_trust_bundle_v2_auto_restart.json"

def create_default_trust_bundle(path: str):
    """
    Creates a minimal self-only trust bundle if none exists.
    If strong crypto is available, uses Ed25519 + swarm key.
    If not, uses random bytes as pseudo keys.
    """
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

    node_id = "node-local-single-v2-auto-restart"

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
            "predictor_path": "borg_predictor_v2_auto_restart.json"
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
        info = PeerInfo(
            node_id=pid,
            public_key=pk,
            host=p["host"],
            port=p["port"],
            initial_reputation=p.get("initial_reputation", 0.0),
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
#  SIGNING & ENCRYPTION (AUTO MODE)
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
#  PREDICTIVE DOMAIN TRACKER (PERSISTENT)
# ============================================================

class PredictiveDomainTracker:
    def __init__(self, path="borg_predictor_v2_auto_restart.json", max_history: int = 1000):
        self.path = path
        self.counts = defaultdict(int)
        self.max_history = max_history
        self.total_events = 0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                counts = data.get("counts", {})
                for k, v in counts.items():
                    self.counts[k] = int(v)
                self.total_events = sum(self.counts.values())
            except Exception:
                pass

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"counts": self.counts}, f)
        except Exception:
            pass

    def record(self, domain: str):
        self.counts[domain] += 1
        self.total_events += 1
        if self.total_events > self.max_history:
            for k in list(self.counts.keys()):
                self.counts[k] = max(1, self.counts[k] // 2)
            self.total_events = sum(self.counts.values())
        self._save()

    def top_domains(self, n: int = 10) -> List[str]:
        return [d for d, _ in sorted(self.counts.items(), key=lambda x: x[1], reverse=True)[:n]]


# ============================================================
#  QUEEN ROUTER
# ============================================================

class QueenRouter:
    def __init__(self, node_id: str, private_key_hex: str,
                 peer_registry: PeerRegistry, reputation: PeerReputation,
                 swarm_crypto: SwarmCrypto,
                 listen_ip: str = "0.0.0.0", sync_port: int = 5055,
                 predictor_path: str = "borg_predictor_v2_auto_restart.json",
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

    # ---------------- P2P SYNC ----------------

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

    def _handle_peer_conn(self, conn: socket.socket):
        with conn:
            try:
                data = conn.recv(65535)
                if not data:
                    return
                body = self.swarm_crypto.decrypt(data)
                if body is None:
                    return
                payload = verify_update(self.peer_registry, self.reputation, body)
                if payload is None:
                    return
                peer_id = body.get("node_id")
                self.submit_update(payload, from_peer=True, peer_id=peer_id)
            except Exception:
                return

    def _broadcast_update(self, update):
        if not self.peer_registry.peers:
            return
        body = sign_update(self.node_id, self.private_key_hex, update)
        payload = self.swarm_crypto.encrypt(body)

        for peer_id, info in self.peer_registry.peers.items():
            if not self.reputation.is_trusted(peer_id):
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
#  WORKER SCANNER
# ============================================================

class WorkerScanner:
    def __init__(self, queen: QueenRouter, base_domains: List[str], interval_sec: int = 15, max_predictive_domains: int = 10):
        self.queen = queen
        self.base_domains = base_domains
        self.interval_sec = interval_sec
        self.max_predictive_domains = max_predictive_domains
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _probe_domain(self, domain: str):
        candidates = []
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
        return {"domain": domain, "candidates": candidates}

    def _loop(self):
        while self._running:
            predictive_top = self.queen.predictor.top_domains(self.max_predictive_domains)
            domains = list(set(self.base_domains + predictive_top))
            for d in domains:
                update = self._probe_domain(d)
                self.queen.submit_update(update, from_peer=False)
            time.sleep(self.interval_sec)


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
#  DNS SERVER (A + AAAA)
# ============================================================

TOP_DOMAINS = {
    "yahoo.com",
    "google.com",
    "youtube.com",
    "microsoft.com",
}

class LocalDNSServer:
    def __init__(self, queen: QueenRouter, listen_ip="127.0.0.1", port=53):
        self.queen = queen
        self.listen_ip = listen_ip
        self.port = port

    def start(self):
        threading.Thread(target=self._serve, daemon=True).start()

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
        if qtype == 28:  # AAAA
            answer_type = b"\x00\x1c"
            rdata = socket.inet_pton(socket.AF_INET6, ip)
        else:            # A
            answer_type = b"\x00\x01"
            rdata = socket.inet_aton(ip)
        answer_class = b"\x00\x01"
        ttl = b"\x00\x00\x00\x3c"
        rdlength = struct.pack("!H", len(rdata))
        answer = answer_name + answer_type + answer_class + ttl + rdlength + rdata
        return header + question + answer

    def _serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.listen_ip, self.port))

        while True:
            try:
                data, addr = sock.recvfrom(512)
                domain, qtype = self._parse_query(data)

                if qtype == 28:
                    ip = self.queen.resolve(domain, socket.AF_INET6)
                else:
                    ip = self.queen.resolve(domain, socket.AF_INET)

                if ip:
                    response = self._build_response(data, ip, qtype)
                    sock.sendto(response, addr)
                    continue

                try:
                    if qtype == 28:
                        infos = socket.getaddrinfo(domain, 443, socket.AF_INET6, socket.SOCK_STREAM)
                        ip = infos[0][4][0]
                        response = self._build_response(data, ip, qtype)
                    else:
                        upstream_ip = socket.gethostbyname(domain)
                        response = self._build_response(data, upstream_ip, qtype)
                    sock.sendto(response, addr)
                except Exception:
                    pass
            except Exception:
                time.sleep(0.1)


# ============================================================
#  MAIN BOOTSTRAP
# ============================================================

def main():
    trust_path = DEFAULT_TRUST_BUNDLE
    bundle = load_trust_bundle(trust_path)

    swarm_crypto = SwarmCrypto(bundle["settings"]["swarm_key"])

    queen = QueenRouter(
        node_id=bundle["node_id"],
        private_key_hex=bundle["private_key_hex"],
        peer_registry=bundle["peer_registry"],
        reputation=bundle["peer_reputation"],
        swarm_crypto=swarm_crypto,
        listen_ip=bundle["listen_ip"],
        sync_port=bundle["sync_port"],
        predictor_path=bundle["settings"].get("predictor_path", "borg_predictor_v2_auto_restart.json"),
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

    print("Borg Routing Organism v2 (auto elevation + auto install + auto restart) running.")
    print("Local DNS on 127.0.0.1:53 (A + AAAA)")
    print(f"Trust bundle: {trust_path}")
    if USE_STRONG_CRYPTO:
        print("Crypto mode: STRONG (Ed25519 + encrypted P2P, PyNaCl active)")
    else:
        print("Crypto mode: COMPAT (pseudo-sign + unencrypted P2P, PyNaCl unavailable)")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

