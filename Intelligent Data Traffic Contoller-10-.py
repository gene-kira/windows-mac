#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, platform, pathlib
def is_admin():
    try:
        return os.getuid() == 0
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
if platform.system() == "Windows" and not is_admin():
    import ctypes
    script_path = os.path.abspath(sys.argv[0])
    params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}" {params}', None, 1)
    sys.exit()

STATE_DIR = os.path.join(str(pathlib.Path.home()), ".autonomous_cop")
os.makedirs(STATE_DIR, exist_ok=True)

import subprocess, importlib, time, threading, queue, random, socket, json, signal, math, datetime, struct
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from time import monotonic

def auto_load(lib: str, pip_name: Optional[str] = None):
    try:
        return importlib.import_module(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or lib])
        return importlib.import_module(lib)

asyncio = auto_load("asyncio")
tkinter = auto_load("tkinter")
matplotlib = auto_load("matplotlib"); matplotlib.use("TkAgg")
plt = auto_load("matplotlib.pyplot")
FigureCanvasTkAgg = auto_load("matplotlib.backends.backend_tkagg").FigureCanvasTkAgg
aiohttp = auto_load("aiohttp"); from aiohttp import web
numpy = auto_load("numpy")
websockets = auto_load("websockets")
psutil = auto_load("psutil")
cryptography = auto_load("cryptography")
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

@dataclass
class TokenBucket:
    rate: float
    burst: float
    tokens: float = field(default=0.0)
    last: float = field(default_factory=monotonic)
    def __post_init__(self): self.tokens = self.burst
    def allow(self, cost: float = 1.0) -> bool:
        now = monotonic(); elapsed = now - self.last
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate); self.last = now
        if self.tokens >= cost: self.tokens -= cost; return True
        return False

@dataclass
class PriorityLane:
    name: str
    weight: int
    bucket: TokenBucket
    q: deque = field(default_factory=deque)
    holds: bool = False

@dataclass
class Endpoint:
    protocol: str
    host: str
    port: int
    path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouteInfo:
    endpoint: Endpoint
    health: float = 1.0
    weight: float = 1.0
    errors: int = 0
    latency_ms_p50: float = 30.0
    in_use: bool = True

@dataclass
class MutationRecord:
    timestamp: float
    before: Dict[str, Any]
    after: Dict[str, Any]
    reward: float
    accepted: bool

@dataclass
class Strategy:
    name: str
    scheduler_weights: Dict[str, int]
    rate_guards: Dict[str, float]
    router_bias: float
    proto_behaviors: Dict[str, Dict[str, Any]]

class SecureStore:
    def __init__(self, path=os.path.join(STATE_DIR, "traffic_cop_state.bin"), key_path=os.path.join(STATE_DIR, "traffic_cop_key.bin")):
        self.path = path; self.key_path = key_path
        self.key = self._load_or_create_key(); self.fernet = Fernet(self.key)
    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path): return open(self.key_path, "rb").read()
        key = Fernet.generate_key(); open(self.key_path, "wb").write(key); return key
    def save(self, data: Dict[str, Any]):
        token = self.fernet.encrypt(json.dumps(data).encode("utf-8"))
        tmp = self.path + ".tmp"; open(tmp, "wb").write(token); os.replace(tmp, self.path)
        try: os.chmod(self.path, 0o600)
        except: pass
    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path): return None
        try:
            blob = self.fernet.decrypt(open(self.path, "rb").read())
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

class SystemBackbone:
    def __init__(self):
        self.inventory = self.get_inventory()
        self.metrics = {}
        self.lock = threading.Lock()
        self.last_disk = psutil.disk_io_counters(); self.last_net = psutil.net_io_counters(); self.last_time = time.time()
    def get_inventory(self) -> Dict[str, Any]:
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "network_interfaces": list(psutil.net_if_addrs().keys()),
        }
    def sample_metrics(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=0.10)
        mem = psutil.virtual_memory(); disk = psutil.disk_io_counters(); net = psutil.net_io_counters()
        now = time.time(); dt = max(0.0001, now - self.last_time)
        disk_read_mb_s = ((disk.read_bytes - self.last_disk.read_bytes) / (1024**2)) / dt
        disk_write_mb_s = ((disk.write_bytes - self.last_disk.write_bytes) / (1024**2)) / dt
        net_sent_mb_s = ((net.bytes_sent - self.last_net.bytes_sent) / (1024**2)) / dt
        net_recv_mb_s = ((net.bytes_recv - self.last_net.bytes_recv) / (1024**2)) / dt
        with self.lock:
            self.metrics = {
                "cpu_percent": round(cpu_percent, 2),
                "memory_used_gb": round(mem.used / (1024**3), 2),
                "memory_percent": round(mem.percent, 2),
                "disk_read_mb_s": round(disk_read_mb_s, 3),
                "disk_write_mb_s": round(disk_write_mb_s, 3),
                "net_sent_mb_s": round(net_sent_mb_s, 3),
                "net_recv_mb_s": round(net_recv_mb_s, 3),
            }
            self.last_disk, self.last_net, self.last_time = disk, net, now
        return self.metrics

class PortManager:
    def __init__(self, scan_ranges: List[Tuple[int,int]] = [(8000, 8100), (9000, 9050)]):
        self.scan_ranges = scan_ranges
        self.available_out_ports: Dict[str, List[int]] = {"tcp": [], "udp": []}
        self.lock = threading.Lock()
    def _is_port_free(self, host: str, port: int, proto: str) -> bool:
        try:
            if proto == "tcp":
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.2)
                res = s.connect_ex((host, port)); s.close(); return res != 0
            elif proto == "udp":
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try: s.bind((host, port)); s.close(); return True
                except: s.close(); return False
        except: return False
    def scan(self, host: str = "0.0.0.0"):
        with self.lock:
            self.available_out_ports = {"tcp": [], "udp": []}
            for start, end in self.scan_ranges:
                for p in range(start, end+1):
                    if self._is_port_free(host, p, "tcp"): self.available_out_ports["tcp"].append(p)
                    if self._is_port_free(host, p, "udp"): self.available_out_ports["udp"].append(p)
    def next_out_port(self, proto: str) -> Optional[int]:
        with self.lock:
            ports = self.available_out_ports.get(proto, [])
            return ports[0] if ports else None

class Router:
    def __init__(self, routes: List[RouteInfo]):
        self.routes = routes; self.lock = threading.Lock()
    def choose(self) -> Optional[RouteInfo]:
        with self.lock:
            best, score_best = None, -1.0
            for info in self.routes:
                if not info.in_use: continue
                score = info.health * info.weight
                if score > score_best:
                    score_best = score; best = info
            return best
    def record_success(self, route: RouteInfo, latency_ms: float):
        with self.lock:
            route.latency_ms_p50 = 0.85 * route.latency_ms_p50 + 0.15 * latency_ms
            route.health = min(1.0, route.health + 0.02)
            if route.errors > 0: route.errors = max(0, route.errors - 1)
    def record_error(self, route: RouteInfo):
        with self.lock:
            route.errors += 1
            route.health = max(0.1, route.health - 0.06)
            if route.errors > 10: route.weight = max(0.1, route.weight - 0.1)
    def widen_under_pressure(self, pressure: int, bias: float = 0.1):
        with self.lock:
            if pressure > 2000:
                for r in self.routes:
                    r.in_use = True
                    r.weight = min(2.5, r.weight + bias)
            elif pressure < 800:
                for r in self.routes:
                    r.weight = max(0.5, r.weight - bias/2)

class TelemetryBus:
    def __init__(self): self.q = queue.Queue(maxsize=100000)
    def emit(self, event: Dict[str, Any]):
        try: self.q.put_nowait(event)
        except queue.Full:
            try: self.q.get_nowait()
            except: pass
            try: self.q.put_nowait(event)
            except: pass

class ClientPools:
    def __init__(self):
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()
    async def get_http(self) -> aiohttp.ClientSession:
        async with self.lock:
            if self.http_session is None or self.http_session.closed:
                self.http_session = aiohttp.ClientSession()
            return self.http_session
    async def close(self):
        async with self.lock:
            if self.http_session and not self.http_session.closed:
                await self.http_session.close()

class ProtoClient:
    def __init__(self, pools: ClientPools):
        self.pools = pools
    async def send(self, endpoint: Endpoint, payload: bytes, behavior: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Optional[str]]:
        t0 = monotonic()
        try:
            proto = endpoint.protocol
            if proto in ("http","https"):
                scheme = "https" if proto == "https" else "http"
                url = f"{scheme}://{endpoint.host}:{endpoint.port}{endpoint.path or '/'}"
                session = await self.pools.get_http()
                async with session.post(url, data=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    await resp.read()
                return True, (monotonic()-t0)*1000.0, None
            elif proto == "ws":
                url = f"ws://{endpoint.host}:{endpoint.port}{endpoint.path or '/'}"
                async with websockets.connect(url, max_queue=2) as ws:
                    await ws.send(payload.decode("utf-8", errors="ignore"))
                return True, (monotonic()-t0)*1000.0, None
            elif proto == "tcp":
                try:
                    reader, writer = await asyncio.open_connection(endpoint.host, endpoint.port)
                except OSError:
                    await asyncio.sleep(random.uniform(0.005, 0.03))
                    reader, writer = await asyncio.open_connection(endpoint.host, endpoint.port)
                writer.write(payload); await writer.drain()
                writer.close(); await writer.wait_closed()
                return True, (monotonic()-t0)*1000.0, None
            elif proto == "udp":
                loop = asyncio.get_running_loop()
                local_port = random.randint(20000, 60000)
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: asyncio.DatagramProtocol(),
                    local_addr=("0.0.0.0", local_port),
                    remote_addr=(endpoint.host, endpoint.port))
                transport.sendto(payload); transport.close()
                return True, (monotonic()-t0)*1000.0, None
            elif proto == "xproto":
                bs = behavior or {}
                chunk = int(bs.get("chunk_size", 256)); jitter_ms = float(bs.get("jitter_ms", 2.0)); retries = int(bs.get("retries", 0))
                loop = asyncio.get_running_loop()
                local_port = random.randint(20000, 60000)
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: asyncio.DatagramProtocol(),
                    local_addr=("0.0.0.0", local_port),
                    remote_addr=(endpoint.host, endpoint.port))
                for i in range(0, len(payload), chunk):
                    transport.sendto(payload[i:i+chunk]); await asyncio.sleep(jitter_ms / 1000.0)
                transport.close()
                for _ in range(retries): await asyncio.sleep(jitter_ms / 1000.0)
                return True, (monotonic()-t0)*1000.0, None
            else:
                return False, (monotonic()-t0)*1000.0, f"Unknown protocol {proto}"
        except Exception as e:
            await asyncio.sleep(random.uniform(0.005, 0.03))
            return False, (monotonic()-t0)*1000.0, str(e)

class Forecaster:
    def __init__(self):
        self.pressure_hist = deque(maxlen=800)
        self.latency_hist = deque(maxlen=800)
        self.alpha_p = 0.35; self.beta_p = 0.20
        self.alpha_l = 0.40; self.beta_l = 0.25
        self.route_lat_hist: Dict[str, deque] = {}
        self.ewma_p = None
        self.ewma_l = None
        self.ewma_alpha = 0.3
        self.hour_pressure_sum = [0.0]*24
        self.hour_pressure_count = [0]*24
        self.hour_latency_sum = [0.0]*24
        self.hour_latency_count = [0]*24
    def update(self, pressure: float, latency_sample: Optional[float], route_latencies: Optional[Dict[str, float]] = None):
        self.pressure_hist.append(pressure)
        if latency_sample is not None: self.latency_hist.append(latency_sample)
        if self.ewma_p is None: self.ewma_p = float(pressure)
        else: self.ewma_p = (1 - self.ewma_alpha) * self.ewma_p + self.ewma_alpha * pressure
        if latency_sample is not None:
            if self.ewma_l is None: self.ewma_l = float(latency_sample)
            else: self.ewma_l = (1 - self.ewma_alpha) * self.ewma_l + self.ewma_alpha * latency_sample
        if route_latencies:
            for rid, lat in route_latencies.items():
                dq = self.route_lat_hist.setdefault(rid, deque(maxlen=400))
                dq.append(lat)
        hour = datetime.datetime.now().hour
        self.hour_pressure_sum[hour] += pressure; self.hour_pressure_count[hour] += 1
        if latency_sample is not None:
            self.hour_latency_sum[hour] += latency_sample; self.hour_latency_count[hour] += 1
    def holt_next(self, series: List[float], alpha: float, beta: float) -> Tuple[float, float, float]:
        if not series: return 0.0, 0.0, 0.0
        level = series[0]; trend = 0.0
        for x in series[1:]:
            prev_level = level
            level = alpha * x + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return level + trend, level + 5 * trend, level
    def time_mods(self) -> Dict[str, float]:
        now = datetime.datetime.now()
        hour = now.hour; weekday = now.weekday(); month = now.month
        hour_mult = 1.0 + 0.3 * math.exp(-((hour-10)**2)/18) + 0.25 * math.exp(-((hour-18)**2)/12)
        day_mult = 1.15 if weekday < 5 else 0.9
        season_mult = 1.15 if month in (11,12) else (1.05 if month in (6,7) else 1.0)
        avg_p_all = (sum(self.hour_pressure_sum) / max(1, sum(self.hour_pressure_count))) if sum(self.hour_pressure_count) else None
        avg_l_all = (sum(self.hour_latency_sum) / max(1, sum(self.hour_latency_count))) if sum(self.hour_latency_count) else None
        avg_p_hr = (self.hour_pressure_sum[hour] / max(1, self.hour_pressure_count[hour])) if self.hour_pressure_count[hour] else None
        avg_l_hr = (self.hour_latency_sum[hour] / max(1, self.hour_latency_count[hour])) if self.hour_latency_count[hour] else None
        learned_p_mult = (avg_p_hr / avg_p_all) if (avg_p_all and avg_p_hr) else 1.0
        learned_l_mult = (avg_l_hr / avg_l_all) if (avg_l_all and avg_l_hr) else 1.0
        learned_mult = 0.5 * learned_p_mult + 0.5 * learned_l_mult
        return {"hour_mult": hour_mult, "day_mult": day_mult, "season_mult": season_mult, "learned_mult": learned_mult}
    def confidence_band(self, series: deque, mult: float = 2.0) -> Tuple[float, float]:
        if not series: return 0.0, 0.0
        arr = numpy.array(series); mu = arr.mean(); sigma = arr.std() or 1.0
        return mu - mult * sigma, mu + mult * sigma
    def forecast(self) -> Dict[str, Any]:
        p_next, p_5s, _ = self.holt_next(list(self.pressure_hist), self.alpha_p, self.beta_p)
        if self.latency_hist:
            l_next, l_5s, _ = self.holt_next(list(self.latency_hist), self.alpha_l, self.beta_l)
        else:
            l_next, l_5s = (0.0, 0.0)
        mods = self.time_mods()
        base_p_factor = mods["hour_mult"] * mods["day_mult"] * mods["season_mult"]
        p_factor = base_p_factor * mods["learned_mult"]
        l_factor = (0.5 * mods["hour_mult"] + 0.5 * mods["day_mult"]) * mods["learned_mult"]
        route_next: Dict[str, float] = {}
        for rid, dq in self.route_lat_hist.items():
            rn, _, _ = self.holt_next(list(dq), self.alpha_l, self.beta_l)
            route_next[rid] = max(0.0, rn * l_factor)
        p_lo, p_hi = self.confidence_band(self.pressure_hist, 1.8)
        l_lo, l_hi = self.confidence_band(self.latency_hist, 1.8) if self.latency_hist else (0.0, 0.0)
        return {
            "pressure_next": max(0.0, p_next * p_factor),
            "latency_next": max(0.0, l_next * l_factor),
            "pressure_5s": max(0.0, p_5s * p_factor),
            "latency_5s": max(0.0, l_5s * l_factor),
            "route_latency_next": route_next,
            "mods": mods,
            "ewma_pressure": self.ewma_p or 0.0,
            "ewma_latency": self.ewma_l or 0.0,
            "pressure_ci": (p_lo, p_hi),
            "latency_ci": (l_lo, l_hi),
        }
    def anomaly(self) -> Dict[str, bool]:
        def spike(series: deque, k=2.5):
            if len(series) < 40: return False
            arr = numpy.array(series); mu = arr.mean(); sigma = arr.std() or 1.0
            return arr[-1] > mu + k * sigma
        return {"pressure_spike": spike(self.pressure_hist), "latency_spike": spike(self.latency_hist)}

class StrategyBandit:
    def __init__(self, base: Strategy):
        self.base = base
        self.arms: Dict[str, Dict[str, Any]] = {}
        self.epsilon = 0.12
    def register(self, strat: Strategy):
        self.arms[strat.name] = {"strategy": strat, "value": 0.0, "count": 0}
    def choose(self) -> Strategy:
        if random.random() < self.epsilon or not self.arms:
            return self._random_candidate()
        best = max(self.arms.values(), key=lambda x: x["value"])
        return best["strategy"]
    def update(self, strat_name: str, reward: float):
        arm = self.arms.get(strat_name)
        if not arm: return
        arm["count"] += 1
        arm["value"] = arm["value"] + (reward - arm["value"]) / arm["count"]
    def _random_candidate(self) -> Strategy:
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        sw = dict(self.base.scheduler_weights)
        sw["P0"] = clamp(sw.get("P0", 5) + random.choice([-1,0,1]), 4, 8)
        sw["P1"] = clamp(sw.get("P1", 3) + random.choice([-1,0,1]), 2, 6)
        sw["P2"] = clamp(sw.get("P2", 2) + random.choice([-1,0,1]), 1, 4)
        sw["P3"] = clamp(sw.get("P3", 1) + random.choice([0,1]), 1, 2)
        guards = dict(self.base.rate_guards)
        guards["p0_floor_rate"] = clamp(guards.get("p0_floor_rate", 240) + random.choice([-10,0,10]), 220, 480)
        guards["p3_ceiling_rate"] = clamp(guards.get("p3_ceiling_rate", 70) + random.choice([-10,0,10]), 20, 120)
        proto_behaviors = dict(self.base.proto_behaviors)
        proto_behaviors["xproto"] = {"chunk_size": random.choice([128,256,512]),
                                     "jitter_ms": random.choice([1.0,2.0,5.0]),
                                     "retries": random.choice([0,1,2])}
        router_bias = round(random.choice([0.05,0.1,0.15]), 2)
        name = f"rl_{int(time.time())}_{random.randint(1000,9999)}"
        strat = Strategy(name, sw, guards, router_bias, proto_behaviors)
        self.register(strat); return strat

@dataclass
class LiquorBotConfig:
    bot_id: str
    position: str
    interval_s: float = 1.0
    jitter_ms: int = 50

class LiquorBot:
    def __init__(self, cfg: LiquorBotConfig, telemetry: TelemetryBus, pools: ClientPools):
        self.cfg = cfg
        self.telemetry = telemetry
        self.pools = pools
        self.running = True
        self.rng = random.Random(hash(cfg.bot_id) & 0xffffffff)
        self.lat_hist = deque(maxlen=200)
        self.net_hist = deque(maxlen=200)
        self.err_hist = deque(maxlen=200)
    async def _probe_net(self):
        t0 = monotonic()
        try:
            net = psutil.net_io_counters()
            sent = net.bytes_sent; recv = net.bytes_recv
            await asyncio.sleep(0.05)
            net2 = psutil.net_io_counters()
            d_sent = max(0, net2.bytes_sent - sent); d_recv = max(0, net2.bytes_recv - recv)
            mbps = (d_sent + d_recv) / (1024**2) / max(0.001, (monotonic() - t0))
            self.net_hist.append(mbps)
            self.telemetry.emit({"event":"liquor_net","bot":self.cfg.bot_id,"pos":self.cfg.position,"mbps":mbps})
        except Exception as e:
            self.err_hist.append(1)
            self.telemetry.emit({"event":"liquor_error","bot":self.cfg.bot_id,"pos":self.cfg.position,"error":str(e)})
    async def _probe_latency(self):
        t0 = monotonic()
        try:
            loop = asyncio.get_running_loop()
            local_port = self.rng.randint(20000, 60000)
            transport, _ = await loop.create_datagram_endpoint(lambda: asyncio.DatagramProtocol(),
                                                               local_addr=("0.0.0.0", local_port),
                                                               remote_addr=("127.0.0.1", 8050))
            transport.sendto(os.urandom(64))
            await asyncio.sleep(0.01 + self.rng.random()*0.02)
            transport.close()
            lat_ms = (monotonic() - t0) * 1000.0
            self.lat_hist.append(lat_ms)
            self.telemetry.emit({"event":"liquor_lat","bot":self.cfg.bot_id,"pos":self.cfg.position,"lat_ms":lat_ms})
        except Exception as e:
            self.err_hist.append(1)
            self.telemetry.emit({"event":"liquor_error","bot":self.cfg.bot_id,"pos":self.cfg.position,"error":str(e)})
    async def run(self):
        while self.running:
            try:
                await asyncio.gather(self._probe_net(), self._probe_latency())
            except Exception as e:
                self.err_hist.append(1)
                self.telemetry.emit({"event":"liquor_error","bot":self.cfg.bot_id,"pos":self.cfg.position,"error":str(e)})
            await asyncio.sleep(max(0.1, self.cfg.interval_s + (self.rng.randint(-self.cfg.jitter_ms, self.cfg.jitter_ms)/1000.0)))

# --- User activity monitor ---
class UserActivityMonitor:
    def __init__(self, telemetry: TelemetryBus, poll_s: float = 1.5):
        self.telemetry = telemetry
        self.poll_s = poll_s
        self.running = True
        self.force_user_priority = False
        self.detected_apps: List[str] = []
        self.signals: Dict[str, Any] = {"user_active": False, "apps": []}
        self._targets = [
            "steam","skype","teams","discord","zoom","obs","twitch",
            "chrome","firefox","edge","opera","epicgames","battle.net",
            "valorant","fortnite","csgo","minecraft","roblox","league","dota"
        ]
    def snapshot(self) -> Dict[str, Any]:
        return {"user_active": self.signals.get("user_active", False), "apps": list(self.detected_apps), "forced": self.force_user_priority}
    def set_forced(self, on: bool):
        self.force_user_priority = on
        self.telemetry.emit({"event":"user_priority_toggle","forced": on, "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    def _detect(self):
        apps = set()
        try:
            for p in psutil.process_iter(attrs=["name","exe","cmdline","username","cpu_percent"]):
                name = (p.info.get("name") or "").lower()
                exe = (p.info.get("exe") or "").lower()
                cmd = " ".join([str(x).lower() for x in (p.info.get("cmdline") or [])])
                if any(t in name or t in exe or t in cmd for t in self._targets):
                    apps.add(name or exe or cmd)
                # crude video hint: browser with gpu process or high cpu
                if ("chrome" in name or "chromium" in name or "edge" in name or "firefox" in name) and (p.info.get("cpu_percent", 0.0) > 15.0):
                    apps.add(name)
        except Exception:
            pass
        self.detected_apps = sorted(list(apps))[:8]
        active = bool(self.detected_apps)
        return active
    async def run(self):
        while self.running:
            active = self._detect()
            user_active = self.force_user_priority or active
            self.signals["user_active"] = user_active
            self.signals["apps"] = list(self.detected_apps)
            self.telemetry.emit({"event":"user_activity","user_active": user_active, "apps": list(self.detected_apps),
                                 "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            await asyncio.sleep(self.poll_s)

class TrafficCop:
    def __init__(self, telemetry: TelemetryBus, router: Router, portmgr: PortManager,
                 backbone: SystemBackbone, base_strategy: Strategy, store: SecureStore, user_monitor: UserActivityMonitor):
        self.telemetry = telemetry; self.router = router; self.portmgr = portmgr; self.backbone = backbone
        self.store = store; self.pools = ClientPools(); self.client = ProtoClient(self.pools)
        self.forecaster = Forecaster()
        self.user_monitor = user_monitor

        persisted = self.store.load() or {}
        strat_dict = persisted.get("strategy")
        if strat_dict: base_strategy = Strategy(**strat_dict)

        sw = base_strategy.scheduler_weights
        self.lanes: Dict[str, PriorityLane] = {
            "P0": PriorityLane("P0", sw.get("P0", 5), TokenBucket(rate=base_strategy.rate_guards.get("p0_floor_rate", 240), burst=400)),
            "P1": PriorityLane("P1", sw.get("P1", 3), TokenBucket(rate=160, burst=240)),
            "P2": PriorityLane("P2", sw.get("P2", 2), TokenBucket(rate=100, burst=160)),
            "P3": PriorityLane("P3", sw.get("P3", 1), TokenBucket(rate=base_strategy.rate_guards.get("p3_ceiling_rate", 70), burst=120)),
        }
        self.strategy = base_strategy

        self.params = persisted.get("params") or {
            "pressure_onhold_threshold": 2400,
            "pressure_relief_threshold": 1300,
            "cpu_soft_limit": 75.0, "cpu_hard_limit": 90.0,
            "mem_soft_limit": 80.0, "mem_hard_limit": 92.0,
            "net_soft_mb_s": 50.0, "net_hard_mb_s": 100.0,
            "disk_soft_mb_s": 80.0, "disk_hard_mb_s": 150.0,
            "forecast_pressure_warn": 2200, "forecast_latency_warn_ms": 80.0,
            "ephemeral_pressure_limit": 300,
            "concurrency_max": 220, "concurrency_min": 20,
            "concurrency_decay": 0.8, "concurrency_growth": 1.06,
            "user_priority_min_reserve": 0.20  # reserve 20% concurrency headroom when user is active
        }

        self.running = True
        self.mutations: List[MutationRecord] = []
        self.concurrency = self.params["concurrency_min"] * 2
        self.error_window = deque(maxlen=200)
        self.addr_in_use_hits = deque(maxlen=200)
        self.sema = asyncio.Semaphore(self.concurrency)

        self.bandit = StrategyBandit(base_strategy)
        self.baseline_score = persisted.get("baseline_score", 0.0)
        self.promotion_margin = 40.0
        self.promotion_consecutive = 2
        self.consecutive_beats = 0

        self.watch_latency = deque([0.0]*40, maxlen=120)
        self.watch_errors = deque([0]*10, maxlen=120)
        self.last_promotion_time: Optional[float] = None
        self.last_strategy_before_promotion: Optional[Strategy] = None

        self._conc_smooth = float(self.concurrency)

    def persist(self, baseline_score: Optional[float] = None):
        data = {"strategy": self.strategy.__dict__, "params": self.params}
        if baseline_score is not None: data["baseline_score"] = baseline_score
        self.store.save(data)

    def classify(self, item: Dict[str, Any]) -> str:
        # user-priority: enforce P0 for user-facing items
        if item.get("userApp") or item.get("sla") or item.get("transactional"): return "P0"
        if item.get("userFacing"): return "P1"
        if item.get("bulk"): return "P3"
        return "P2"

    def ingest(self, item_stream_iterable):
        for item in item_stream_iterable:
            lane_name = self.classify(item)
            self.lanes[lane_name].q.append(item)
            self.telemetry.emit({"event":"enqueue","lane":lane_name})

    async def dispatch(self, item: Dict[str, Any], route: RouteInfo):
        async with self.sema:
            payload = (item.get("payload") or "data").encode("utf-8", errors="ignore")
            behavior = self.strategy.proto_behaviors.get(route.endpoint.protocol, {})
            ok, latency_ms, err = await self.client.send(route.endpoint, payload, behavior)
            rid = f"{route.endpoint.protocol}:{route.endpoint.host}:{route.endpoint.port}"
            if ok:
                self.router.record_success(route, latency_ms)
                self.telemetry.emit({"event":"success","route":rid,"latency_ms":latency_ms})
                self.forecaster.update(pressure=0, latency_sample=latency_ms, route_latencies={rid: latency_ms})
                self.watch_latency.append(latency_ms)
            else:
                self.router.record_error(route)
                self.error_window.append(1); self.watch_errors.append(1)
                if err and ("10048" in err or "address already in use" in err.lower()): self.addr_in_use_hits.append(1)
                self.telemetry.emit({"event":"error","route":rid,"error":err})

    async def scheduler(self):
        order = ["P0","P1","P2","P3"]
        while self.running:
            forecast = self.forecaster.forecast()
            route_next = forecast.get("route_latency_next", {})
            fast_rids = sorted(route_next.items(), key=lambda kv: kv[1])[:2]
            fast_set = {rid for rid, _ in fast_rids}
            for lane_name in order:
                lane = self.lanes[lane_name]
                if lane.holds and lane_name != "P0": continue
                budget = min(lane.weight, max(1, int(self.sema._value)))
                for _ in range(budget):
                    if lane.q and lane.bucket.allow():
                        item = lane.q.popleft()
                        route = self.router.choose()
                        if route:
                            rid = f"{route.endpoint.protocol}:{route.endpoint.host}:{route.endpoint.port}"
                            asyncio.create_task(self.dispatch(item, route))
                            self.telemetry.emit({"event":"dispatch","lane":lane_name,"route":rid})
            await asyncio.sleep(0.002)

    async def predictive_shaping(self, forecast: Dict[str, Any], pressure: int, lb_mbps: float, user_active: bool):
        p_next = forecast["pressure_next"]; l_next = forecast["latency_next"]
        anomalies = self.forecaster.anomaly()
        p_ci_hi = forecast.get("pressure_ci", (0.0, p_next))[1]
        l_ci_hi = forecast.get("latency_ci", (0.0, l_next))[1]
        lb_factor = 1.0 + min(0.5, lb_mbps / 100.0)
        risk_p = max(p_ci_hi, p_next) * lb_factor
        risk_l = max(l_ci_hi, l_next)

        if user_active:
            # Hard bias toward user lanes
            self.lanes["P0"].weight = min(9, self.lanes["P0"].weight + 2)
            self.lanes["P1"].weight = min(7, self.lanes["P1"].weight + 1)
            self.lanes["P2"].bucket.rate = max(60.0, self.lanes["P2"].bucket.rate * 0.9)
            self.lanes["P3"].bucket.rate = max(18.0, self.lanes["P3"].bucket.rate * 0.8)
            self.lanes["P3"].holds = True
        elif (risk_p > self.params["forecast_pressure_warn"] or
              risk_l > self.params["forecast_latency_warn_ms"] or
              anomalies["pressure_spike"]):
            self.lanes["P0"].weight = min(8, self.lanes["P0"].weight + 1)
            self.lanes["P1"].weight = min(6, self.lanes["P1"].weight + 1)
            self.lanes["P3"].bucket.rate = max(18.0, self.lanes["P3"].bucket.rate * 0.9)
            self.router.widen_under_pressure(int(max(p_next, p_ci_hi)), bias=self.strategy.router_bias)
            self.telemetry.emit({"event":"predictive_act","p_next":p_next,"l_next":l_next,
                                 "ci":{"p_hi":p_ci_hi,"l_hi":l_ci_hi},"mods":forecast.get("mods",{}),
                                 "anomaly":anomalies,"lb_mbps":lb_mbps,"user_active":user_active})

        if pressure > self.params["ephemeral_pressure_limit"]:
            self.lanes["P2"].bucket.rate = max(50.0, self.lanes["P2"].bucket.rate * 0.95)
            self.lanes["P3"].bucket.rate = max(18.0, self.lanes["P3"].bucket.rate * 0.9)

    def _adjust_concurrency(self, calm: bool, user_active: bool):
        addr_hits = sum(self.addr_in_use_hits)
        if addr_hits > 0:
            target = max(self.params["concurrency_min"], int(self.concurrency * 0.7)); self.addr_in_use_hits.clear()
        elif user_active:
            # keep headroom reserved for real-time user tasks
            target = min(self.params["concurrency_max"], int(self.concurrency * 1.02))
        elif calm:
            target = min(self.params["concurrency_max"], int(self.concurrency * self.params["concurrency_growth"]))
        else:
            target = max(self.params["concurrency_min"], int(self.concurrency * self.params["concurrency_decay"]))
        reserve = self.params["user_priority_min_reserve"] if user_active else 0.0
        self._conc_smooth = 0.7 * self._conc_smooth + 0.3 * target
        self.concurrency = int(round(self._conc_smooth))
        effective_conc = int(self.concurrency * (1.0 - reserve))
        effective_conc = max(self.params["concurrency_min"], effective_conc)
        delta = effective_conc - self.sema._value
        if delta > 0:
            for _ in range(delta): self.sema.release()
        elif delta < 0:
            self.sema._value = max(self.params["concurrency_min"], effective_conc)

    def _rollback_watchdog(self):
        if self.last_promotion_time is None: return
        if len(self.watch_latency) < 40: return
        avg_lat = sum(self.watch_latency)/len(self.watch_latency)
        err_density = sum(self.watch_errors)
        degrade_lat = (avg_lat*8.0) > (self.baseline_score + 30.0)
        degrade_err = err_density > max(10, len(self.watch_errors)//4)
        if degrade_lat or degrade_err:
            if self.last_strategy_before_promotion:
                self.apply_strategy(self.last_strategy_before_promotion, baseline_score=self.baseline_score, auto=False)
                self.telemetry.emit({"event":"rollback","reason":{"lat":degrade_lat,"err":degrade_err}})
                self.last_promotion_time = None
                self.watch_latency.clear(); self.watch_errors.clear()

    def _liquor_bot_mbps_avg(self) -> float:
        try:
            if liquor_bot_hub:
                snap = liquor_bot_hub.snapshot()
                vals = [v.get("mbps", 0.0) for v in snap.values() if isinstance(v, dict)]
                return float(sum(vals) / max(1, len(vals))) if vals else 0.0
        except Exception:
            pass
        return 0.0

    async def adaptive(self):
        while self.running:
            depths = {k: len(v.q) for k,v in self.lanes.items()}
            pressure = sum(depths.values())
            sysm = self.backbone.sample_metrics()
            cpu, memp = sysm["cpu_percent"], sysm["memory_percent"]
            net_total = sysm["net_sent_mb_s"] + sysm["net_recv_mb_s"]
            disk_total = sysm["disk_read_mb_s"] + sysm["disk_write_mb_s"]

            avg_lat = (sum(self.watch_latency) / len(self.watch_latency)) if self.watch_latency else None
            self.forecaster.update(pressure=pressure, latency_sample=avg_lat)
            forecast = self.forecaster.forecast()
            p_next = forecast["pressure_next"]; l_next = forecast["latency_next"]

            # Holds under pressure
            for name, lane in self.lanes.items():
                lane.holds = (name != "P0") and (pressure > self.params["pressure_onhold_threshold"])
            if pressure < self.params["pressure_relief_threshold"]:
                for lane in self.lanes.values(): lane.holds = False

            overload_soft = (cpu > self.params["cpu_soft_limit"] or memp > self.params["mem_soft_limit"] or
                             net_total > self.params["net_soft_mb_s"] or disk_total > self.params["disk_soft_mb_s"])
            overload_hard = (cpu > self.params["cpu_hard_limit"] or memp > self.params["mem_hard_limit"] or
                             net_total > self.params["net_hard_mb_s"] or disk_total > self.params["disk_hard_mb_s"])

            if overload_hard:
                self.lanes["P3"].bucket.rate = max(10.0, self.lanes["P3"].bucket.rate * 0.6)
                self.lanes["P2"].bucket.rate = max(30.0, self.lanes["P2"].bucket.rate * 0.75)
                self.lanes["P1"].bucket.rate = min(220.0, self.lanes["P1"].bucket.rate * 0.95)
                self.lanes["P0"].bucket.rate = max(self.strategy.rate_guards.get("p0_floor_rate", 240), self.lanes["P0"].bucket.rate)
                self.lanes["P3"].holds = True
            elif overload_soft:
                self.lanes["P3"].bucket.rate = max(20.0, self.lanes["P3"].bucket.rate * 0.85)
                self.lanes["P2"].bucket.rate = max(50.0, self.lanes["P2"].bucket.rate * 0.9)
                self.lanes["P0"].bucket.rate = max(self.strategy.rate_guards.get("p0_floor_rate", 240), self.lanes["P0"].bucket.rate)

            self.lanes["P0"].bucket.rate = max(self.strategy.rate_guards.get("p0_floor_rate", 240), self.lanes["P0"].bucket.rate)
            self.lanes["P3"].bucket.rate = min(self.strategy.rate_guards.get("p3_ceiling_rate", 70), self.lanes["P3"].bucket.rate)

            lb_mbps = self._liquor_bot_mbps_avg()
            user_state = user_monitor.snapshot()
            user_active = bool(user_state.get("user_active", False))
            await self.predictive_shaping(forecast, pressure, lb_mbps, user_active)

            calm = (sum(self.error_window) < max(5, len(self.error_window) // 10)) and (pressure < self.params["pressure_onhold_threshold"])
            self._adjust_concurrency(calm, user_active)

            self.router.widen_under_pressure(pressure, bias=self.strategy.router_bias)
            self._rollback_watchdog()

            ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.telemetry.emit({"event":"adapt","depths":depths,"pressure":pressure,
                                 "forecast":forecast,"pressure_next":p_next,"latency_next":l_next,
                                 "sysmetrics":sysm,"overload_soft":overload_soft,"overload_hard":overload_hard,
                                 "concurrency":self.concurrency,"lb_mbps":lb_mbps,
                                 "user_active": user_active, "user_apps": user_state.get("apps", []),
                                 "adapt_ts": ts_now})
            await asyncio.sleep(0.5)

    def apply_strategy(self, new_strategy: Strategy, baseline_score: Optional[float] = None, auto: bool = False):
        before = {"strategy": self.strategy.name,
                  "weights": {k: v.weight for k, v in self.lanes.items()},
                  "guards": dict(self.strategy.rate_guards)}
        self.last_strategy_before_promotion = self.strategy
        for k, lane in self.lanes.items(): lane.weight = new_strategy.scheduler_weights.get(k, lane.weight)
        self.strategy = new_strategy
        after = {"strategy": self.strategy.name,
                 "weights": {k: v.weight for k, v in self.lanes.items()},
                 "guards": dict(self.strategy.rate_guards)}
        self.mutations.append(MutationRecord(time.time(), before, after, reward=baseline_score or 0.0, accepted=True))
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.telemetry.emit({"event":"auto_promotion" if auto else "promotion","new_strategy":self.strategy.name,"promo_ts":ts})
        self.persist(baseline_score)
        self.last_promotion_time = time.time()
        self.watch_latency.clear(); self.watch_errors.clear()

class SecondBrain:
    def __init__(self, telemetry: TelemetryBus, backbone: SystemBackbone, base_strategy: Strategy, cop: TrafficCop, store: SecureStore):
        self.telemetry = telemetry; self.backbone = backbone; self.cop = cop; self.store = store
        self.bandit = cop.bandit
        self.history: List[Dict[str, Any]] = []
        self.running = True
    async def evaluate(self, strat: Strategy) -> Dict[str, Any]:
        endpoints = [Endpoint("udp","127.0.0.1",8050), Endpoint("tcp","127.0.0.1",8051), Endpoint("xproto","127.0.0.1",8052)]
        client = ProtoClient(ClientPools())
        payload = os.urandom(2048)
        latencies, errors = [], 0
        for ep in endpoints:
            ok, lat_ms, err = await client.send(ep, payload, strat.proto_behaviors.get(ep.protocol, {}))
            latencies.append(lat_ms); errors += 0 if ok else 1
        sysm = self.backbone.sample_metrics()
        avg_lat = numpy.mean(latencies) if latencies else 50.0
        penalty_sys = (max(0, sysm["cpu_percent"]-75.0) + max(0, sysm["memory_percent"]-80.0) +
                       max(0, (sysm['net_sent_mb_s']+sysm['net_recv_mb_s'])-50.0) + max(0, (sysm['disk_read_mb_s']+sysm['disk_write_mb_s'])-80.0))
        score = (1000.0 - avg_lat*8.0) - (errors*50.0) - (penalty_sys*2.0)
        result = {"strategy": strat.name, "avg_latency_ms": avg_lat, "errors": errors, "score": score}
        self.telemetry.emit({"event":"sim_result","result":result})
        return result
    async def run(self):
        while self.running:
            candidate = self.bandit.choose()
            res = await self.evaluate(candidate)
            self.history.append(res)
            self.bandit.update(candidate.name, res["score"])
            if res["score"] > self.cop.baseline_score + self.cop.promotion_margin:
                self.cop.consecutive_beats += 1
                if self.cop.consecutive_beats >= self.cop.promotion_consecutive:
                    self.cop.apply_strategy(candidate, baseline_score=res["score"], auto=True)
                    self.cop.baseline_score = res["score"]; self.cop.consecutive_beats = 0
                    self.store.save({"strategy": self.cop.strategy.__dict__, "params": self.cop.params,
                                     "baseline_score": self.cop.baseline_score, "mutations": [asdict(m) for m in self.cop.mutations]})
            else:
                self.cop.consecutive_beats = 0
            await asyncio.sleep(1.0)

class GlassNet(asyncio.DatagramProtocol):
    def __init__(self, telemetry: TelemetryBus, cop: TrafficCop,
                 group_host="239.12.12.12", group_port=42042,
                 seed_secret_path=os.path.join(STATE_DIR, "glassnet_seed.bin")):
        self.telemetry = telemetry; self.cop = cop
        self.group_host = group_host; self.group_port = group_port
        self.transport = None; self.loop = None
        self.node_id = f"node_{random.randint(100000,999999)}"
        self.priv = x25519.X25519PrivateKey.generate(); self.pub = self.priv.public_key().public_bytes_raw()
        self.sessions: Dict[bytes, Dict[str, Any]] = {}
        self.seed_secret = self._load_or_create_seed(seed_secret_path)
        self.last_discovery = 0.0; self.last_gossip = 0.0
        self.quorum_window = deque(maxlen=36)
        self.trust: Dict[str, float] = {}
        self.peer_latest: Dict[str, Dict[str, Any]] = {}
        self.peer_bots: Dict[str, Dict[str, Any]] = {}

    def _load_or_create_seed(self, path: str) -> bytes:
        try:
            if os.path.exists(path): return open(path, "rb").read()
            secret = os.urandom(32); open(path, "wb").write(secret); return secret
        except Exception: return os.urandom(32)

    def _group_key_for_window(self) -> AESGCM:
        now = datetime.datetime.now()
        slot = int((now.minute // 2))
        salt = f"{now.year}-{now.month}-{now.day}-{now.hour}-{slot}".encode("utf-8")
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"glassnet-group").derive(self.seed_secret)
        return AESGCM(key)

    def _encrypt_with_nonce(self, aes: AESGCM, payload: bytes, ctr: int) -> bytes:
        nonce = struct.pack("!12s", bytes(f"{ctr:012d}", "ascii"))
        ct = aes.encrypt(nonce, payload, b"glassnet"); return nonce + ct

    def _decrypt_with_nonce(self, aes: AESGCM, ct: bytes) -> Optional[bytes]:
        if len(ct) < 13: return None
        nonce, body = ct[:12], ct[12:]
        try: return aes.decrypt(nonce, body, b"glassnet")
        except Exception: return None

    def connection_made(self, transport):
        self.transport = transport; self.loop = asyncio.get_running_loop(); self._join_multicast()

    def _join_multicast(self):
        sock = self.transport.get_extra_info('socket')
        try:
            mreq = struct.pack("4sl", socket.inet_aton(self.group_host), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            try: sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
            except: pass
        except Exception as e:
            self.telemetry.emit({"event":"p2p_error","error":f"multicast_join:{e}"})

    def _send_group(self, data: bytes):
        try: self.transport.sendto(data, (self.group_host, self.group_port))
        except Exception as e: self.telemetry.emit({"event":"p2p_error","error":f"send_group:{e}"})

    def datagram_received(self, data, addr):
        group_aes = self._group_key_for_window()
        blob = self._decrypt_with_nonce(group_aes, data)
        if blob:
            try: obj = json.loads(blob.decode("utf-8"))
            except Exception: return
            sym = obj.get("sym","")
            if sym == "🔍":
                try:
                    peer_pub = bytes.fromhex(obj["pub"]); node = obj.get("node","unknown")
                except Exception: return
                if peer_pub == self.pub: return
                if peer_pub not in self.sessions:
                    aes = self._derive_session(peer_pub)
                    self.sessions[peer_pub] = {"aes": aes, "nonce_ctr": 1, "node": node, "trust": 0.5, "addr": addr}
                    ack = {"sym":"🧪","node": self.node_id, "addr": addr}
                    frame = self._encrypt_with_nonce(group_aes, json.dumps(ack).encode("utf-8"), ctr=random.randint(1, 1_000_000))
                    self._send_group(frame)
                    self.telemetry.emit({"event":"p2p_session","peer":node})
            elif sym == "🧪":
                self.telemetry.emit({"event":"p2p_ack","from":obj.get("node","unknown")})
            return
        for peer_pub, sess in list(self.sessions.items()):
            blob = self._decrypt_with_nonce(sess["aes"], data)
            if blob:
                try:
                    info = json.loads(blob.decode("utf-8"))
                    if info.get("sym") == "🫥":
                        state = info.get("state", {})
                        self.peer_latest[state.get("node","unknown")] = state
                        bots = state.get("liquor_bots", {})
                        node = state.get("node","unknown")
                        if bots: self.peer_bots[node] = bots
                        self._merge_gossip(state, sess)
                        return
                except Exception:
                    continue

    def error_received(self, exc):
        self.telemetry.emit({"event":"p2p_error","error":str(exc)})

    def _derive_session(self, peer_pub_bytes: bytes) -> AESGCM:
        peer_pub = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)
        shared = self.priv.exchange(peer_pub)
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"glassnet-peer").derive(shared)
        return AESGCM(key)

    def _gossip_state(self) -> Dict[str, Any]:
        depths = {k: len(v.q) for k,v in self.cop.lanes.items()}
        pressure = sum(depths.values())
        forecast = self.cop.forecaster.forecast()
        with self.cop.router.lock:
            routes = [{
                "rid": f"{r.endpoint.protocol}:{r.endpoint.host}:{r.endpoint.port}",
                "lat_p50": r.latency_ms_p50,
                "health": r.health,
                "weight": r.weight,
                "errors": r.errors
            } for r in self.cop.router.routes]
        lane_weights = {k: v.weight for k, v in self.cop.lanes.items()}
        return {
            "node": self.node_id, "ts": time.time(),
            "depths": depths, "pressure": pressure, "forecast": forecast,
            "routes": routes, "lane_weights": lane_weights, "concurrency": self.cop.concurrency,
            "liquor_bots": liquor_bot_hub.snapshot() if liquor_bot_hub else {},
        }

    def _merge_gossip(self, info: Dict[str, Any], sess: Dict[str, Any]):
        try:
            peer_node = info.get("node","unknown")
            f = info.get("forecast", {})
            p_next = float(f.get("pressure_next", 0.0))
            self.quorum_window.append({"node": peer_node, "pressure_next": p_next, "ts": info.get("ts", time.time())})
            if len(self.quorum_window) >= 2:
                recent = list(self.quorum_window)[-2]
                realized = sum({k: len(v.q) for k,v in self.cop.lanes.items()}.values())
                error = abs(recent["pressure_next"] - realized)
                delta = -0.01 if error > 1000 else 0.01
                sess["trust"] = min(1.0, max(0.0, sess["trust"] + delta))
                self.trust[peer_node] = sess["trust"]
            t = sess["trust"]
            peer_weights = info.get("lane_weights", {})
            for k, w_peer in peer_weights.items():
                if k in self.cop.lanes:
                    w_local = self.cop.lanes[k].weight
                    self.cop.lanes[k].weight = int(round((0.75 * w_local) + (0.25 * t * w_peer)))
            peer_routes = {r["rid"]: r for r in info.get("routes", [])}
            with self.cop.router.lock:
                for r in self.cop.router.routes:
                    rid = f"{r.endpoint.protocol}:{r.endpoint.host}:{r.endpoint.port}"
                    pr = peer_routes.get(rid)
                    if pr:
                        r.health = max(r.health, pr["health"])
                        r.latency_ms_p50 = min(r.latency_ms_p50, pr["lat_p50"])
                        r.weight = (r.weight * 0.8) + (pr["weight"] * 0.2 * t)
            quorum = [x for x in self.quorum_window if time.time() - x["ts"] < 5.0]
            if quorum:
                hi = sum(1 for x in quorum if x["pressure_next"] > self.cop.params["forecast_pressure_warn"])
                if hi >= max(2, len(quorum)//2):
                    self.cop.router.widen_under_pressure(int(p_next), bias=self.cop.strategy.router_bias)
            peer_conc = int(info.get("concurrency", self.cop.concurrency))
            self.cop.concurrency = max(self.cop.params["concurrency_min"], int(0.6 * self.cop.concurrency + 0.4 * peer_conc))
            self.cop.sema._value = max(self.cop.params["concurrency_min"], self.cop.concurrency)
            self.telemetry.emit({"event":"p2p_merge","peer":peer_node,"trust":sess["trust"],"p_next":p_next,"peer_conc":peer_conc})
        except Exception as e:
            self.telemetry.emit({"event":"p2p_error","error":f"merge:{e}"})

    async def start(self, host="0.0.0.0"):
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(lambda: self, local_addr=(host, self.group_port), reuse_port=True)
        asyncio.create_task(self._periodic())

    async def _periodic(self):
        while True:
            try:
                await self._broadcast_discovery()
                await self._broadcast_gossip()
            except Exception as e:
                self.telemetry.emit({"event":"p2p_error","error":str(e)})
            await asyncio.sleep(1.5)

    async def _broadcast_discovery(self):
        now = monotonic()
        if now - self.last_discovery < 2.0: return
        self.last_discovery = now
        aes = self._group_key_for_window()
        disc = {"sym":"🔍","node": self.node_id, "pub": self.pub.hex()}
        frame = self._encrypt_with_nonce(aes, json.dumps(disc).encode("utf-8"), ctr=random.randint(1, 1_000_000))
        self._send_group(frame)

    async def _broadcast_gossip(self):
        now = monotonic()
        if now - self.last_gossip < 0.8: return
        self.last_gossip = now
        state = {"sym":"🫥","state": self._gossip_state()}
        payload = json.dumps(state).encode("utf-8")
        for sess in list(self.sessions.values()):
            try:
                ctr = sess["nonce_ctr"]; sess["nonce_ctr"] += 1
                frame = self._encrypt_with_nonce(sess["aes"], payload, ctr)
                self._send_group(frame)
            except Exception as e:
                self.telemetry.emit({"event":"p2p_error","error":f"gossip:{e}"})

class LiquorBotHub:
    def __init__(self, telemetry: TelemetryBus, pools: ClientPools, configs: List[LiquorBotConfig]):
        self.telemetry = telemetry
        self.pools = pools
        self.configs = configs
        self.bots: List[LiquorBot] = [LiquorBot(cfg, telemetry, pools) for cfg in configs]
        self.latest: Dict[str, Dict[str, Any]] = {}
        self.running = True
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.latest)
    async def run(self):
        async def bot_task(bot: LiquorBot):
            await bot.run()
        def listener():
            while self.running:
                try:
                    e = self.telemetry.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                ev = e.get("event")
                if ev in ("liquor_net","liquor_lat","liquor_error"):
                    bid = e.get("bot"); pos = e.get("pos")
                    rec = self.latest.setdefault(bid, {"pos": pos})
                    rec.update({k:v for k,v in e.items() if k not in ("event","bot","pos")})
        threading.Thread(target=listener, daemon=True).start()
        await asyncio.gather(*[bot_task(b) for b in self.bots])

class MetricsServer:
    def __init__(self, cop: TrafficCop, router: Router, backbone: SystemBackbone, sim: SecondBrain, user_monitor: UserActivityMonitor):
        self.cop, self.router, self.backbone, self.sim, self.user_monitor = cop, router, backbone, sim, user_monitor
        self.app = web.Application(); self.app.add_routes([web.get('/metrics', self.handle_metrics)])
    async def handle_metrics(self, request):
        lines = []
        depths = {k: len(v.q) for k,v in self.cop.lanes.items()}
        pressure = sum(depths.values())
        sysm = self.backbone.metrics or self.backbone.sample_metrics()
        lines.append(f"traffic_pressure {pressure}")
        for name, depth in depths.items(): lines.append(f"traffic_lane_depth{{lane=\"{name}\"}} {depth}")
        with self.router.lock:
            for r in self.router.routes:
                ep = r.endpoint; rid = f"{ep.protocol}:{ep.host}:{ep.port}"
                lines.append(f"traffic_route_latency_ms{{route=\"{rid}\"}} {r.latency_ms_p50}")
                lines.append(f"traffic_route_health{{route=\"{rid}\"}} {r.health}")
                lines.append(f"traffic_route_errors{{route=\"{rid}\"}} {r.errors}")
                lines.append(f"traffic_route_weight{{route=\"{rid}\"}} {r.weight}")
        lines += [
            f"system_cpu_percent {sysm['cpu_percent']}",
            f"system_memory_percent {sysm['memory_percent']}",
            f"system_memory_used_gb {sysm['memory_used_gb']}",
            f"system_disk_read_mb_s {sysm['disk_read_mb_s']}",
            f"system_disk_write_mb_s {sysm['disk_write_mb_s']}",
            f"system_net_sent_mb_s {sysm['net_sent_mb_s']}",
            f"system_net_recv_mb_s {sysm['net_recv_mb_s']}",
        ]
        forecast = self.cop.forecaster.forecast()
        lines += [
            f"forecast_pressure_next {forecast['pressure_next']}",
            f"forecast_latency_next {forecast['latency_next']}",
            f"forecast_pressure_5s {forecast['pressure_5s']}",
            f"forecast_latency_5s {forecast['latency_5s']}",
            f"forecast_ewma_pressure {forecast['ewma_pressure']}",
            f"forecast_ewma_latency {forecast['ewma_latency']}",
        ]
        user_state = self.user_monitor.snapshot()
        lines.append(f"user_active {1 if user_state.get('user_active') else 0}")
        lines.append(f"user_priority_forced {1 if user_state.get('forced') else 0}")
        lines.append(f"adaptive_concurrency {self.cop.concurrency}")
        lines.append(f"strategy_name{{name=\"{self.cop.strategy.name}\"}} 1")
        lines.append(f"sim_live_baseline_score {self.cop.baseline_score}")
        text = "\n".join(lines) + "\n"
        return web.Response(text=text, content_type="text/plain")
    def start(self, host="0.0.0.0", port=9100):
        threading.Thread(target=lambda: web.run_app(self.app, host=host, port=port), daemon=True).start()

class WebScanner:
    def __init__(self, telemetry: TelemetryBus, endpoints: List[Endpoint], pools: ClientPools):
        self.telemetry = telemetry; self.endpoints = endpoints; self.running = True; self.pools = pools
    async def probe_http(self, endpoint: Endpoint):
        t0 = monotonic(); url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}{endpoint.path or '/'}"
        try:
            session = await self.pools.get_http()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp: await resp.text()
            self.telemetry.emit({"event":"scan_success","url":url,"latency_ms":(monotonic()-t0)*1000.0,"status":resp.status})
        except Exception as e:
            self.telemetry.emit({"event":"scan_error","url":url,"error":str(e)})
    async def run(self):
        while self.running:
            tasks = []
            for ep in self.endpoints:
                if ep.protocol in ("http","https"): tasks.append(asyncio.create_task(self.probe_http(ep)))
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(3.0)
    async def manual_scan(self):
        tasks = []
        for ep in self.endpoints:
            if ep.protocol in ("http","https"): tasks.append(asyncio.create_task(self.probe_http(ep)))
        if tasks: await asyncio.gather(*tasks, return_exceptions=True)

class TrafficGUI:
    def __init__(self, cop: TrafficCop, telemetry: TelemetryBus, backbone: SystemBackbone,
                 sim: SecondBrain, net: GlassNet, scanner: WebScanner, bot_hub: LiquorBotHub,
                 user_monitor: UserActivityMonitor):
        self.cop, self.telemetry, self.backbone, self.sim, self.net, self.scanner, self.bot_hub, self.user_monitor = cop, telemetry, backbone, sim, net, scanner, bot_hub, user_monitor
        self.root = tkinter.Tk()
        self.root.title("Predictive Cop (User Priority + RL + GlassNet + LiquorBots)")
        try: self.root.geometry("980x620")
        except: pass

        top_frame = tkinter.Frame(self.root); top_frame.pack(fill="x")
        self.live_label = tkinter.Label(top_frame, text=f"Live strategy: {self.cop.strategy.name}", font=("Arial", 9))
        self.live_label.pack(side="left", padx=6)
        self.baseline_label = tkinter.Label(top_frame, text=f"Baseline score: {self.cop.baseline_score:.2f}", font=("Arial", 9))
        self.baseline_label.pack(side="left", padx=6)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_promo_label = tkinter.Label(top_frame, text=f"Last auto-promotion: {now_str}", font=("Arial", 9))
        self.last_promo_label.pack(side="left", padx=6)
        self.last_adjust_label = tkinter.Label(top_frame, text=f"Last adaptive adjustment: {now_str}", font=("Arial", 9))
        self.last_adjust_label.pack(side="left", padx=6)

        scan_btn = tkinter.Button(top_frame, text="Manual scan", font=("Arial", 9), command=self._manual_scan)
        scan_btn.pack(side="right", padx=8)

        # User priority controls
        user_frame = tkinter.Frame(self.root); user_frame.pack(fill="x", pady=4)
        self.user_status_label = tkinter.Label(user_frame, text="User priority: inactive | apps: []", font=("Arial", 9))
        self.user_status_label.pack(side="left", padx=6)
        self.user_toggle_btn = tkinter.Button(user_frame, text="Force user priority (ON/OFF)", font=("Arial", 9),
                                              command=self._toggle_user_priority)
        self.user_toggle_btn.pack(side="right", padx=8)

        inv_text = tkinter.Text(self.root, height=4, font=("Courier New", 8))
        inv_text.insert("1.0", json.dumps(self.backbone.inventory, indent=2))
        inv_text.configure(state="disabled"); inv_text.pack(fill="x")

        main_frame = tkinter.Frame(self.root); main_frame.pack(fill="both", expand=True)
        left_frame = tkinter.Frame(main_frame); left_frame.pack(side="left", fill="both", expand=True)
        right_frame = tkinter.Frame(main_frame); right_frame.pack(side="right", fill="y")

        self.fig, self.axs = plt.subplots(2, 3, figsize=(6.6, 4.0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame); self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.maxlen = 600
        self.pressure_hist = deque([0]*40, maxlen=self.maxlen)
        self.latency_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.cpu_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.mem_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.net_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.disk_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.p_next_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.l_next_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.conc_hist = deque([self.cop.concurrency]*40, maxlen=self.maxlen)
        self.p_ewma_hist = deque([0.0]*40, maxlen=self.maxlen)
        self.l_ewma_hist = deque([0.0]*40, maxlen=self.maxlen)

        self.axs[0][0].set_title("CPU %", fontsize=9); self.line_cpu, = self.axs[0][0].plot([], [], color="darkblue", linewidth=1.0)
        self.axs[0][1].set_title("Memory %", fontsize=9); self.line_mem, = self.axs[0][1].plot([], [], color="teal", linewidth=1.0)
        self.axs[0][2].set_title("Net MB/s", fontsize=9); self.line_net, = self.axs[0][2].plot([], [], color="purple", linewidth=1.0)
        self.axs[1][0].set_title("Pressure & forecast", fontsize=9)
        self.line_press, = self.axs[1][0].plot([], [], label="pressure", color="black", linewidth=1.0)
        self.line_pnext, = self.axs[1][0].plot([], [], label="next", color="orange", linewidth=1.0)
        self.line_pewma, = self.axs[1][0].plot([], [], label="ewma", color="gray", linewidth=1.0)
        self.axs[1][0].legend(fontsize=7)
        self.axs[1][1].set_title("Latency & forecast", fontsize=9)
        self.line_lat, = self.axs[1][1].plot([], [], label="live", color="green", linewidth=1.0)
        self.line_lnext, = self.axs[1][1].plot([], [], label="next", color="brown", linewidth=1.0)
        self.line_lewma, = self.axs[1][1].plot([], [], label="ewma", color="gray", linewidth=1.0)
        self.axs[1][1].legend(fontsize=7)
        self.axs[1][2].set_title("Adaptive concurrency", fontsize=9)
        self.line_conc, = self.axs[1][2].plot([], [], label="concurrency", color="red", linewidth=1.0)
        self.axs[1][2].legend(fontsize=7)
        for ax in (self.axs[0][0], self.axs[0][1], self.axs[0][2], self.axs[1][0], self.axs[1][1], self.axs[1][2]):
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.tick_params(labelsize=7)

        tkinter.Label(right_frame, text="LAN peers", font=("Arial", 10, "bold")).pack(anchor="w", padx=6, pady=(6,2))
        self.peer_list = tkinter.Listbox(right_frame, height=10, font=("Arial", 9)); self.peer_list.pack(fill="y", padx=6, pady=4)
        self.peer_info = tkinter.Text(right_frame, height=10, width=30, font=("Courier New", 8)); self.peer_info.pack(fill="y", padx=6, pady=4)
        self.peer_info.configure(state="disabled")
        self.peer_list.bind("<<ListboxSelect>>", self._on_peer_select)

        tkinter.Label(right_frame, text="Liquor bots", font=("Arial", 10, "bold")).pack(anchor="w", padx=6, pady=(8,2))
        self.bot_list = tkinter.Listbox(right_frame, height=12, font=("Arial", 9)); self.bot_list.pack(fill="y", padx=6, pady=4)
        self.bot_info = tkinter.Text(right_frame, height=12, width=30, font=("Courier New", 8)); self.bot_info.pack(fill="y", padx=6, pady=4)
        self.bot_info.configure(state="disabled")
        self.bot_list.bind("<<ListboxSelect>>", self._on_bot_select)

        self.root.after(50, self._poll_telemetry)
        self.root.after(200, self._render)
        self.root.after(800, self._refresh_peer_panel)
        self.root.after(900, self._refresh_bot_panel)

    def _toggle_user_priority(self):
        current = self.user_monitor.force_user_priority
        self.user_monitor.set_forced(not current)

    def _manual_scan(self):
        try:
            asyncio.run(self.scanner.manual_scan())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.create_task(self.scanner.manual_scan())

    def _poll_telemetry(self):
        sysm = self.backbone.sample_metrics()
        self.cpu_hist.append(sysm["cpu_percent"]); self.mem_hist.append(sysm["memory_percent"])
        self.net_hist.append(sysm["net_sent_mb_s"] + sysm["net_recv_mb_s"])
        self.disk_hist.append(sysm["disk_read_mb_s"] + sysm["disk_write_mb_s"])
        while True:
            try: e = self.telemetry.q.get_nowait()
            except queue.Empty: break
            ev = e.get("event")
            if ev == "adapt":
                self.pressure_hist.append(e.get("pressure", 0))
                self.p_next_hist.append(float(e.get("pressure_next", 0.0)))
                self.l_next_hist.append(float(e.get("latency_next", 0.0)))
                f = e.get("forecast", {})
                if f:
                    self.p_ewma_hist.append(float(f.get("ewma_pressure", 0.0)))
                    self.l_ewma_hist.append(float(f.get("ewma_latency", 0.0)))
                self.conc_hist.append(int(e.get("concurrency", self.cop.concurrency)))
                ts = e.get("adapt_ts")
                if ts:
                    self.last_adjust_label.config(text=f"Last adaptive adjustment: {ts}")
                ua = e.get("user_active", False)
                apps = e.get("user_apps", [])
                self.user_status_label.config(text=f"User priority: {'active' if ua else 'inactive'} | apps: {apps}")
            elif ev == "user_activity":
                ua = e.get("user_active", False); apps = e.get("apps", [])
                self.user_status_label.config(text=f"User priority: {'active' if ua else 'inactive'} | apps: {apps}")
            elif ev == "user_priority_toggle":
                forced = e.get("forced", False)
                st = self.user_monitor.snapshot()
                self.user_status_label.config(text=f"User priority: {'active' if st.get('user_active') else 'inactive'} (forced {forced}) | apps: {st.get('apps', [])}")
            elif ev == "success":
                self.latency_hist.append(float(e.get("latency_ms", 0.0)))
            elif ev in ("auto_promotion","promotion"):
                ts = e.get("promo_ts", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                self.live_label.config(text=f"Live strategy: {self.cop.strategy.name}")
                self.last_promo_label.config(text=f"Last auto-promotion: {ts}")
                self.baseline_label.config(text=f"Baseline score: {self.cop.baseline_score:.2f}")
        self.root.after(120, self._poll_telemetry)

    def _update_line(self, line, data, ax, ylim_pad=0.08):
        if not data: data = [0]
        x = range(len(data))
        line.set_data(list(x), list(data))
        ax.set_xlim(0, max(40, len(data)))
        ymin = min(data); ymax = max(data)
        if ymax == ymin: ymax = ymin + 1.0
        pad = (ymax - ymin) * ylim_pad
        ax.set_ylim(ymin - pad, ymax + pad)

    def _render(self):
        self._update_line(self.line_cpu, list(self.cpu_hist), self.axs[0][0])
        self._update_line(self.line_mem, list(self.mem_hist), self.axs[0][1])
        self._update_line(self.line_net, list(self.net_hist), self.axs[0][2])
        self._update_line(self.line_press, list(self.pressure_hist), self.axs[1][0])
        self._update_line(self.line_pnext, list(self.p_next_hist), self.axs[1][0])
        self._update_line(self.line_pewma, list(self.p_ewma_hist), self.axs[1][0])
        self._update_line(self.line_lat, list(self.latency_hist), self.axs[1][1])
        self._update_line(self.line_lnext, list(self.l_next_hist), self.axs[1][1])
        self._update_line(self.line_lewma, list(self.l_ewma_hist), self.axs[1][1])
        self._update_line(self.line_conc, list(self.conc_hist), self.axs[1][2])
        self.fig.tight_layout(); self.canvas.draw()
        self.root.after(220, self._render)

    def _refresh_peer_panel(self):
        peers = sorted([(node, self.net.trust.get(node, 0.5)) for node in self.net.peer_latest.keys()], key=lambda x: -x[1])
        self.peer_list.delete(0, tkinter.END)
        for node, trust in peers:
            self.peer_list.insert(tkinter.END, f"{node}  (trust {trust:.2f})")
        self.root.after(1000, self._refresh_peer_panel)

    def _refresh_bot_panel(self):
        local_bots = self.bot_hub.snapshot()
        peer_bots = {}
        for node, bots in self.net.peer_bots.items():
            for bid, sample in bots.items():
                peer_bots[f"{node}:{bid}"] = sample
        self.bot_list.delete(0, tkinter.END)
        for bid in sorted(local_bots.keys()):
            self.bot_list.insert(tkinter.END, f"{bid} (local)")
        for key in sorted(peer_bots.keys()):
            self.bot_list.insert(tkinter.END, f"{key} (peer)")
        self.root.after(1000, self._refresh_bot_panel)

    def _on_peer_select(self, _):
        sel = self.peer_list.curselection()
        if not sel: return
        label = self.peer_list.get(sel[0])
        node = label.split()[0]
        state = self.net.peer_latest.get(node, {})
        text = {
            "node": node,
            "pressure": state.get("pressure", 0),
            "forecast_pressure_next": state.get("forecast", {}).get("pressure_next", 0.0),
            "forecast_latency_next": state.get("forecast", {}).get("latency_next", 0.0),
            "concurrency": state.get("concurrency", 0),
            "lane_weights": state.get("lane_weights", {}),
            "routes": state.get("routes", [])[:5],
            "liquor_bots": state.get("liquor_bots", {}),
        }
        self.peer_info.configure(state="normal")
        self.peer_info.delete("1.0", tkinter.END)
        self.peer_info.insert("1.0", json.dumps(text, indent=2))
        self.peer_info.configure(state="disabled")

    def _on_bot_select(self, _):
        sel = self.bot_list.curselection()
        if not sel: return
        label = self.bot_list.get(sel[0])
        if "(local)" in label:
            bid = label.split()[0]
            sample = self.bot_hub.snapshot().get(bid, {})
            pos = sample.get("pos","unknown")
            info = {"bot_id": bid, "pos": pos, "mbps": sample.get("mbps", 0.0), "lat_ms": sample.get("lat_ms", 0.0)}
        else:
            key = label.split()[0]
            node, bid = key.split(":")
            sample = self.net.peer_bots.get(node, {}).get(bid, {})
            info = {"node": node, "bot_id": bid, "pos": sample.get("pos","unknown"), "mbps": sample.get("mbps", 0.0), "lat_ms": sample.get("lat_ms", 0.0)}
        self.bot_info.configure(state="normal")
        self.bot_info.delete("1.0", tkinter.END)
        self.bot_info.insert("1.0", json.dumps(info, indent=2))
        self.bot_info.configure(state="disabled")

    def start(self): self.root.mainloop()

def synthetic_stream():
    rng = random.Random(321)
    while True:
        kind = rng.random(); item = {"key":"default", "payload":"demo-data"}
        # Simulate some user-app items periodically
        if rng.random() < 0.05: item.update({"userApp": True, "transactional": True})
        elif kind < 0.08: item.update({"sla":True,"transactional":True})
        elif kind < 0.35: item.update({"userFacing":True})
        elif kind < 0.70: item.update({"bulk":True})
        yield item; time.sleep(rng.uniform(0.0005, 0.004))

def build_routes(portmgr: PortManager) -> List[RouteInfo]:
    portmgr.scan(); routes: List[RouteInfo] = []
    routes.append(RouteInfo(Endpoint("http","httpbin.org",80,"/post"), weight=1.0))
    routes.append(RouteInfo(Endpoint("https","httpbin.org",443,"/post"), weight=0.9))
    tcp_p1 = portmgr.next_out_port("tcp") or 8051
    udp_p1 = portmgr.next_out_port("udp") or 8050
    x_p1 = portmgr.next_out_port("udp") or 8052
    routes.append(RouteInfo(Endpoint("tcp","127.0.0.1",tcp_p1), weight=0.6))
    routes.append(RouteInfo(Endpoint("udp","127.0.0.1",udp_p1), weight=0.6))
    routes.append(RouteInfo(Endpoint("xproto","127.0.0.1",x_p1), weight=0.5))
    return routes

def base_strategy() -> Strategy:
    return Strategy(
        name="base",
        scheduler_weights={"P0":5,"P1":3,"P2":2,"P3":1},
        rate_guards={"p0_floor_rate":240,"p3_ceiling_rate":70},
        router_bias=0.1,
        proto_behaviors={"xproto":{"chunk_size":256,"jitter_ms":2.0,"retries":0}}
    )

def start_async_tasks(cop: TrafficCop, scanner: WebScanner, sim: SecondBrain, net: GlassNet, bot_hub: LiquorBotHub, user_monitor: UserActivityMonitor):
    async def runner():
        await net.start(host="0.0.0.0")
        asyncio.create_task(bot_hub.run())
        asyncio.create_task(user_monitor.run())
        await asyncio.gather(cop.scheduler(), cop.adaptive(), sim.run(), scanner.run())
    threading.Thread(target=lambda: asyncio.run(runner()), daemon=True).start()

liquor_bot_hub: Optional[LiquorBotHub] = None
user_monitor: Optional[UserActivityMonitor] = None

def main():
    telemetry = TelemetryBus()
    store = SecureStore()
    backbone = SystemBackbone()
    portmgr = PortManager()
    routes = build_routes(portmgr)
    router = Router(routes)
    strat = base_strategy()
    global user_monitor
    user_monitor = UserActivityMonitor(telemetry, poll_s=1.5)
    cop = TrafficCop(telemetry, router, portmgr, backbone, strat, store, user_monitor)
    sim = SecondBrain(telemetry, backbone, strat, cop, store)
    net = GlassNet(telemetry, cop, group_host="239.12.12.12", group_port=42042)
    MetricsServer(cop, router, backbone, sim, user_monitor).start(host="0.0.0.0", port=9100)
    scan_targets = [Endpoint("https","httpbin.org",443,"/get")]
    scanner = WebScanner(telemetry, scan_targets, cop.pools)

    bot_configs = [
        LiquorBotConfig("lb-edge-west", "edge-west", interval_s=0.8),
        LiquorBotConfig("lb-core-1", "core-router-1", interval_s=1.0),
        LiquorBotConfig("lb-db-subnet", "db-subnet", interval_s=1.2),
        LiquorBotConfig("lb-app-east", "app-east", interval_s=0.9),
    ]
    global liquor_bot_hub
    liquor_bot_hub = LiquorBotHub(telemetry, cop.pools, bot_configs)

    start_async_tasks(cop, scanner, sim, net, liquor_bot_hub, user_monitor)

    def ingest_thread():
        for item in synthetic_stream(): cop.ingest([item])
    threading.Thread(target=ingest_thread, daemon=True).start()

    def handle_exit(signum, frame):
        try:
            store.save({"strategy": cop.strategy.__dict__, "params": cop.params,
                        "baseline_score": cop.baseline_score, "mutations": [asdict(m) for m in cop.mutations]})
            print("State saved. Exiting.")
        finally:
            os._exit(0)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, handle_exit)

    gui = TrafficGUI(cop, telemetry, backbone, sim, net, scanner, liquor_bot_hub, user_monitor); gui.start()

if __name__ == "__main__":
    main()

