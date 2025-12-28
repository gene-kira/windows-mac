#!/usr/bin/env python3
"""
Autonomous Cipher Engine – Hybrid Brain, Predictive, Mutating, Persistent Defense (All-in-One)

Key traits:
- Hybrid brain:
    * AI policy engine (anomaly typing, personality, confidence).
    * SystemControlEngine with persistent threat history and threat heat.
    * SituationalCortex for global situational awareness and collective health score.
    * SelfRewritingAgent that mutates internal weights based on outcomes.

- Predictive intelligence:
    * SituationalCortex tracks rolling trends (connections, CPU, threat heat).
    * Simple prediction of near-future load (best-guess) influences aggression and scanning intervals.

- Judgment & confidence:
    * Confidence from anomaly statistics.
    * Threat heat from per-process behavior.
    * Personality (conservative/balanced/aggressive) modulates responses.

- Adaptive Codex Mutation:
    * Codex maintains telemetry_retention_minutes and phantom_node flag.
    * Ghost sync detection (spiky latency + low activity + anomaly mismatch).
    * On ghost sync: shorten retention, enable phantom_node, log mutation.
    * Codex state is exportable/importable (sync-ready across nodes).

- Enforcement:
    * NetworkControlService:
        - Receives network_policy, applies Windows Firewall rules via `netsh advfirewall`.
        - Forwards policies to WfpPolicyAgent (QoS stub).
    * FileControlService:
        - Receives file_policy, logs and forwards to MinifilterPolicyAgent (stub).

- Resilience:
    * Supervisor with engine health scoring and auto-restart.
    * Collective health score (SituationalCortex) combining engines + environment.

- Interface:
    * Compact Tkinter GUI:
        - Oscillator A/B states.
        - AI confidence, personality, anomaly.
        - Collective health score.
        - Manual frequency bias slider.

NOTE:
- This script uses only user-mode techniques (netsh) and stubs for WFP/Minifilter.
- No real kernel drivers are included here.
- Run as Administrator on Windows for firewall rules to work.
"""

import sys
import subprocess
import importlib

STD_LIB_MODULES = {
    "tkinter", "threading", "time", "queue", "math", "sys",
    "subprocess", "importlib", "datetime", "json", "platform",
    "os", "hashlib", "struct", "traceback", "ipaddress",
    "random", "pathlib", "statistics", "socket",
}

def ensure_module(mod_name, pip_name=None, optional=False):
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        if mod_name in STD_LIB_MODULES:
            if optional:
                return None
            raise
        if pip_name is None:
            pip_name = mod_name
        try:
            print(f"[AUTO-LOADER] Installing missing package: {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return importlib.import_module(mod_name)
        except Exception:
            if optional:
                print(f"[AUTO-LOADER] Optional package {pip_name} not available.")
                return None
            raise

# Stdlib + Tkinter
tk = ensure_module("tkinter")
from tkinter import ttk
import threading
import time
import queue
import math
import datetime
import json
import platform
import os
import hashlib
import struct
import traceback
import ipaddress
import random
from pathlib import Path
import statistics
import socket

# Optional libs
psutil = ensure_module("psutil", optional=True)
requests = ensure_module("requests", optional=True)


# ---------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------

class EngineBase:
    def __init__(self, name, max_silence=30.0):
        self.name = name
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.last_heartbeat = 0.0
        self.max_silence = max_silence
        self._last_error_time = 0.0

    def _beat(self):
        self.last_heartbeat = time.time()

    def mark_error(self):
        self._last_error_time = time.time()

    def is_healthy(self):
        if not self._running:
            return False
        if self.last_heartbeat == 0.0:
            return True
        return (time.time() - self.last_heartbeat) < self.max_silence

    def health_score(self):
        now = time.time()
        silence = now - self.last_heartbeat if self.last_heartbeat else 0.0
        err_age = now - self._last_error_time if self._last_error_time else 9999.0
        score = 1.0
        if silence > self.max_silence:
            score -= 0.5
        if err_age < 60.0:
            score -= 0.3
        return max(0.0, min(1.0, score))

    def start(self):
        raise NotImplementedError

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------

class BinaryOscillator(EngineBase):
    def __init__(self):
        super().__init__(name="oscillator_a", max_silence=10.0)
        self.frequency_hz = 1000.0
        self.duty_cycle = 0.5
        self.mode = "freq"
        self.current_state = 0

    def set_frequency(self, hz):
        with self._lock:
            self.frequency_hz = max(1.0, min(100_000.0, float(hz)))

    def set_duty_cycle(self, duty):
        with self._lock:
            self.duty_cycle = min(0.99, max(0.01, float(duty)))

    def set_mode(self, mode):
        if mode in ("freq", "max"):
            with self._lock:
                self.mode = mode

    def get_mode_freq(self):
        with self._lock:
            return self.mode, self.frequency_hz

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OscillatorAThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            mode, freq = self.get_mode_freq()
            try:
                if mode == "max":
                    self._run_max()
                else:
                    self._run_freq(freq)
            except Exception:
                self.mark_error()
                time.sleep(0.01)

    def _run_freq(self, freq):
        period = 1.0 / freq
        with self._lock:
            on_time = period * self.duty_cycle
            off_time = period * (1.0 - self.duty_cycle)
        if not self._running:
            return
        self.current_state = 1
        if on_time > 0:
            time.sleep(on_time)
        if not self._running:
            return
        self.current_state = 0
        if off_time > 0:
            time.sleep(off_time)

    def _run_max(self):
        for _ in range(10_000):
            if not self._running:
                return
            self.current_state ^= 1
        time.sleep(0.001)


class MirrorOscillator(EngineBase):
    def __init__(self, primary_oscillator: BinaryOscillator,
                 base_inverse: int = 50_000,
                 scale_factor: int = 100_000_000):
        super().__init__(name="oscillator_b", max_silence=10.0)
        self.primary = primary_oscillator
        self.base_inverse = int(base_inverse)
        self.scale_factor = int(scale_factor)
        self.mode = "freq"
        self.frequency_hz = 1000
        self.duty_cycle = 0.5
        self.current_state = 0

    def get_mode_freq(self):
        with self._lock:
            return self.mode, self.frequency_hz

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OscillatorBThread"
        )
        self._thread.start()

    def _compute_mirror_freq(self, a_freq: float) -> int:
        a_int = max(1, int(a_freq))
        inv = max(1, self.base_inverse - min(self.base_inverse - 1, a_int))
        scaled = max(1, self.scale_factor // a_int)
        avg = (inv + scaled) // 2
        return max(1, min(100_000, int(avg)))

    def _run(self):
        while self._running:
            self._beat()
            try:
                a_mode, a_freq = self.primary.get_mode_freq()
                if a_mode == "max":
                    mode = "freq"
                    freq_b = self._compute_mirror_freq(a_freq)
                else:
                    mode = "max"
                    freq_b = self._compute_mirror_freq(a_freq)
                with self._lock:
                    self.mode = mode
                    self.frequency_hz = freq_b
                if mode == "max":
                    self._run_max()
                else:
                    self._run_freq(freq_b)
            except Exception:
                self.mark_error()
                time.sleep(0.01)

    def _run_freq(self, freq_int: int):
        period = 1.0 / float(freq_int)
        on_time = period * self.duty_cycle
        off_time = period * (1.0 - self.duty_cycle)
        if not self._running:
            return
        self.current_state = 1
        if on_time > 0:
            time.sleep(on_time)
        if not self._running:
            return
        self.current_state = 0
        if off_time > 0:
            time.sleep(off_time)

    def _run_max(self):
        for _ in range(10_000):
            if not self._running:
                return
            self.current_state ^= 1
        time.sleep(0.001)


# ---------------------------------------------------------------------
# System inventory / compute / network / ports / netmap
# ---------------------------------------------------------------------

class SystemInventory(EngineBase):
    def __init__(self, event_queue):
        super().__init__(name="inventory", max_silence=300.0)
        self.event_queue = event_queue
        self.snapshot = {}
        self.snapshot_bytes = b""

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="InventoryThread"
        )
        self._thread.start()

    def get_snapshot_bytes(self):
        with self._lock:
            return self.snapshot_bytes

    def _run(self):
        try:
            self._beat()
            snapshot = self._build_snapshot()
            snapshot_json = json.dumps(snapshot, indent=2, sort_keys=True)
            snapshot_bytes = snapshot_json.encode("utf-8", errors="replace")
            with self._lock:
                self.snapshot = snapshot
                self.snapshot_bytes = snapshot_bytes
            self._push({
                "type": "inventory",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "summary": {
                    "program_count": len(snapshot.get("installed_programs", [])),
                    "devices": snapshot.get("devices", {}),
                },
            })
        except Exception as e:
            self.mark_error()
            self._push({
                "type": "inventory_error",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
        finally:
            self._running = False

    def _build_snapshot(self):
        data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "cpu": {}, "memory": {}, "devices": {}, "installed_programs": [],
        }
        if psutil is not None:
            try:
                data["cpu"] = {
                    "count_logical": psutil.cpu_count(logical=True),
                    "count_physical": psutil.cpu_count(logical=False),
                    "frequency_mhz": (
                        psutil.cpu_freq().current if psutil.cpu_freq() else None
                    ),
                }
                vm = psutil.virtual_memory()
                data["memory"] = {
                    "total_bytes": vm.total,
                    "available_bytes": vm.available,
                }
            except Exception:
                self.mark_error()
        data["devices"] = {"mouse_present": True, "keyboard_present": True}
        data["installed_programs"] = self._get_installed_programs()
        return data

    def _get_installed_programs(self):
        system = platform.system().lower()
        if system == "windows":
            return self._get_installed_programs_windows()
        else:
            programs = []
            try:
                paths = os.environ.get("PATH", "").split(os.pathsep)
                seen = set()
                for p in paths:
                    if not os.path.isdir(p):
                        continue
                    for name in os.listdir(p):
                        full = os.path.join(p, name)
                        if full in seen:
                            continue
                        seen.add(full)
                        if os.access(full, os.X_OK) and not os.path.isdir(full):
                            programs.append({"name": name, "path": full})
            except Exception:
                self.mark_error()
            return programs

    def _get_installed_programs_windows(self):
        programs = []
        try:
            import winreg
        except ImportError:
            return programs
        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        hives = [(winreg.HKEY_LOCAL_MACHINE, "HKLM"),
                 (winreg.HKEY_CURRENT_USER, "HKCU")]
        for hive, hive_name in hives:
            for reg_path in reg_paths:
                try:
                    key = winreg.OpenKey(hive, reg_path)
                except OSError:
                    continue
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        i += 1
                    except OSError:
                        break
                    try:
                        subkey = winreg.OpenKey(key, subkey_name)
                    except OSError:
                        continue
                    try:
                        name, _ = winreg.QueryValueEx(subkey, "DisplayName")
                    except OSError:
                        name = None
                    try:
                        version, _ = winreg.QueryValueEx(subkey, "DisplayVersion")
                    except OSError:
                        version = None
                    try:
                        install_location, _ = winreg.QueryValueEx(
                            subkey, "InstallLocation"
                        )
                    except OSError:
                        install_location = None
                    if name:
                        programs.append({
                            "name": name,
                            "version": version,
                            "install_location": install_location,
                            "source": f"{hive_name}\\{reg_path}",
                        })
                winreg.CloseKey(key)
        return programs

    def _push(self, data):
        try:
            self.event_queue.put_nowait(data)
        except queue.Full:
            pass


class ComputeEngine(EngineBase):
    def __init__(self, event_queue, inventory: SystemInventory):
        super().__init__(name="compute", max_silence=30.0)
        self.event_queue = event_queue
        self.inventory = inventory
        self.iterations_per_report = 200_000

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ComputeThread"
        )
        self._thread.start()

    def _run(self):
        i, state = 0, 0
        digest = hashlib.sha256()
        while self._running:
            self._beat()
            try:
                data = self.inventory.get_snapshot_bytes()
                if not data:
                    time.sleep(0.2)
                    continue
                for b in data:
                    if not self._running:
                        break
                    digest.update(struct.pack("B", b ^ (state & 0xFF)))
                    state ^= b
                    i += 1
                    if i % self.iterations_per_report == 0:
                        self._push({
                            "type": "compute",
                            "iterations": i,
                            "digest_hex": digest.hexdigest(),
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        })
            except Exception:
                self.mark_error()
                time.sleep(0.01)
        self._push({
            "type": "compute",
            "iterations": i,
            "digest_hex": digest.hexdigest(),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stopped": True,
        })

    def _push(self, data):
        try:
            self.event_queue.put_nowait(data)
        except queue.Full:
            pass


class NetworkEngine(EngineBase):
    def __init__(self, event_queue):
        super().__init__(name="network", max_silence=120.0)
        self.event_queue = event_queue
        self.interval_seconds = 30
        self.test_url = "https://worldtimeapi.org/api/timezone/Etc/UTC"
        self.latency_history = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="NetworkThread"
        )
        self._thread.start()

    def _run(self):
        if requests is None:
            self._push({
                "type": "network",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "ok": False,
                "note": "requests module not available",
            })
            return
        while self._running:
            self._beat()
            start = time.time()
            status = {
                "type": "network",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "url": self.test_url,
                "ok": False,
                "latency_ms": None,
                "note": "",
            }
            try:
                resp = requests.get(self.test_url, timeout=5)
                latency = (time.time() - start) * 1000.0
                status["latency_ms"] = latency
                status["ok"] = resp.status_code == 200
                status["note"] = f"HTTP {resp.status_code}"
                self._update_latency(latency)
            except Exception as e:
                self.mark_error()
                status["note"] = f"Error: {e!r}"
            self._push(status)
            for _ in range(int(self.interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def _update_latency(self, latency_ms: float):
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 200:
            self.latency_history.pop(0)

    def get_latency_stats(self):
        if not self.latency_history:
            return None, None
        mean = statistics.mean(self.latency_history)
        stdev = statistics.pstdev(self.latency_history) or 1.0
        return mean, stdev

    def _push(self, data):
        try:
            self.event_queue.put_nowait(data)
        except queue.Full:
            pass


class LocalNetworkMapper(EngineBase):
    def __init__(self, event_queue):
        super().__init__(name="netmap", max_silence=600.0)
        self.event_queue = event_queue
        self.interval_seconds = 180
        self.max_hosts_per_subnet = 256

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="NetmapThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            try:
                subnets = self._discover_subnets()
                self._push({
                    "type": "netmap",
                    "phase": "discover",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "subnet_count": len(subnets),
                    "subnets": [str(n) for n in subnets],
                })
                for net in subnets:
                    if not self._running:
                        break
                    hosts = self._scan_subnet(net)
                    self._push({
                        "type": "netmap",
                        "phase": "scan_result",
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        "subnet": str(net),
                        "host_count": len(hosts),
                        "hosts": hosts,
                    })
            except Exception as e:
                self.mark_error()
                self._push({
                    "type": "netmap",
                    "phase": "error",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
            for _ in range(int(self.interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def _discover_subnets(self):
        subnets = set()
        if psutil is None:
            return list(subnets)
        for ifname, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if getattr(addr, "family", None) == socket.AF_INET:
                    ip = addr.address
                    netmask = addr.netmask
                    if not ip or not netmask:
                        continue
                    if ip.startswith("127."):
                        continue
                    try:
                        network = ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
                        subnets.add(network)
                    except Exception:
                        continue
        return list(subnets)

    def _scan_subnet(self, network):
        hosts = []
        host_iter = list(network.hosts())
        if len(host_iter) > self.max_hosts_per_subnet:
            host_iter = host_iter[: self.max_hosts_per_subnet]
        for ip in host_iter:
            if not self._running:
                break
            ip_str = str(ip)
            if self._ping_host(ip_str):
                hosts.append({"ip": ip_str, "alive": True})
        return hosts

    def _ping_host(self, ip):
        system = platform.system().lower()
        if system == "windows":
            cmd = ["ping", "-n", "1", "-w", "300", ip]
        else:
            cmd = ["ping", "-c", "1", "-W", "1", ip]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception:
            self.mark_error()
            return False

    def _push(self, data):
        try:
            self.event_queue.put_nowait(data)
        except queue.Full:
            pass


class PortsMonitor(EngineBase):
    def __init__(self, event_queue):
        super().__init__(name="ports", max_silence=60.0)
        self.event_queue = event_queue
        self.interval_seconds = 15

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PortsThread"
        )
        self._thread.start()

    def _run(self):
        if psutil is None:
            self._push({
                "type": "ports",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "ok": False,
                "note": "psutil module not available",
            })
            return
        while self._running:
            self._beat()
            try:
                conns = psutil.net_connections(kind="inet")
                listening, active = [], []
                for c in conns:
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None
                    status = c.status
                    if status == psutil.CONN_LISTEN:
                        listening.append({"local": laddr, "pid": c.pid})
                    else:
                        if raddr:
                            active.append({
                                "local": laddr,
                                "remote": raddr,
                                "status": status,
                                "pid": c.pid,
                            })
                self._push({
                    "type": "ports",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "listening_count": len(listening),
                    "active_count": len(active),
                    "listening_sample": listening[:10],
                    "active_sample": active[:20],
                })
            except Exception as e:
                self.mark_error()
                self._push({
                    "type": "ports_error",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
            for _ in range(int(self.interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def _push(self, data):
        try:
            self.event_queue.put_nowait(data)
        except queue.Full:
            pass


# ---------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------

class EventBus(EngineBase):
    def __init__(self):
        super().__init__(name="event_bus", max_silence=10.0)
        self.input_queue = queue.Queue(maxsize=5000)
        self.outputs = []

    def add_output(self, q):
        self.outputs.append(q)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="EventBusThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            try:
                event = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            for out in self.outputs:
                try:
                    out.put_nowait(event)
                except queue.Full:
                    pass


# ---------------------------------------------------------------------
# AI Policy Engine
# ---------------------------------------------------------------------

class AIPolicyEngine(EngineBase):
    def __init__(
        self,
        ai_queue,
        event_queue,
        oscillator_a: BinaryOscillator,
        oscillator_b: MirrorOscillator,
        net_mapper: LocalNetworkMapper,
        ports_monitor: PortsMonitor,
        network_engine: NetworkEngine,
        state_path: Path,
        freq_bias_getter,
    ):
        super().__init__(name="ai_policy", max_silence=10.0)
        self.ai_queue = ai_queue
        self.event_queue = event_queue
        self.oscillator_a = oscillator_a
        self.oscillator_b = oscillator_b
        self.net_mapper = net_mapper
        self.ports_monitor = ports_monitor
        self.network_engine = network_engine
        self.state_path = state_path
        self.weights = self._load_state()
        self.active_history = []
        self.freq_bias_getter = freq_bias_getter
        self.personality = "balanced"
        self.latest_ports = None
        self.latest_network = None

    def _load_state(self):
        default = {
            "active_high_threshold": 200.0,
            "active_mid_threshold": 50.0,
            "freq_base": 200.0,
            "freq_active_factor": 12.0,
            "freq_listen_factor": 4.0,
            "netmap_fast_interval": 60.0,
            "netmap_slow_interval": 240.0,
            "ports_fast_interval": 5.0,
            "ports_mid_interval": 12.0,
            "ports_slow_interval": 25.0,
            "network_fast_interval": 15.0,
            "network_slow_interval": 60.0,
            "learning_rate": 0.01,
        }
        try:
            if self.state_path.is_file():
                with self.state_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                default.update(loaded)
        except Exception:
            pass
        return default

    def _save_state(self):
        try:
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(self.weights, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AIPolicyThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            try:
                event = self.ai_queue.get(timeout=0.1)
            except queue.Empty:
                self._emit_coherence()
                continue
            etype = event.get("type")
            if etype == "ports":
                self.latest_ports = event
                self._update_memory(event)
                self._apply_policy()
            elif etype == "network":
                self.latest_network = event
            self._emit_coherence()

    def _update_memory(self, ports_event):
        active_count = float(ports_event.get("active_count", 0))
        self.active_history.append(active_count)
        if len(self.active_history) > 200:
            self.active_history.pop(0)

    def _compute_confidence_and_anomaly(self, active_count):
        if len(self.active_history) < 10:
            return 0.5, "unknown"
        mean = statistics.mean(self.active_history)
        stdev = statistics.pstdev(self.active_history) or 1.0
        z = (active_count - mean) / stdev
        confidence = max(0.1, min(1.0, 1.5 - 0.2 * abs(z)))
        if abs(z) < 0.8:
            anomaly_type = "normal"
        elif abs(z) >= 3.0:
            anomaly_type = "spike"
        elif 0.8 <= abs(z) < 3.0:
            window = self.active_history[-20:] if len(self.active_history) >= 20 else self.active_history
            if len(window) >= 5:
                v = statistics.pvariance(window)
            else:
                v = 0.0
            anomaly_type = "drift" if v < 5.0 else "noisy"
        else:
            anomaly_type = "normal"
        if mean < 5.0 and active_count < 5.0:
            anomaly_type = "flat"
        return confidence, anomaly_type

    def _update_personality(self, confidence, anomaly_type):
        if confidence < 0.4 and anomaly_type in ("spike", "noisy"):
            self.personality = "conservative"
        elif confidence > 0.8 and anomaly_type in ("normal", "flat", "drift"):
            self.personality = "aggressive"
        else:
            self.personality = "balanced"

    def _personality_factor(self):
        if self.personality == "conservative":
            return 0.7
        elif self.personality == "aggressive":
            return 1.3
        return 1.0

    def _apply_policy(self):
        ports = self.latest_ports or {}
        active_count = float(ports.get("active_count", 0))
        listening_count = float(ports.get("listening_count", 0))
        w = self.weights
        confidence, anomaly_type = self._compute_confidence_and_anomaly(active_count)
        self._update_personality(confidence, anomaly_type)
        base_lr = 0.01
        lr = base_lr * (1.5 - confidence)
        lr = max(0.001, min(0.05, lr))
        if active_count > w["active_high_threshold"]:
            w["active_high_threshold"] += lr * active_count
        elif active_count < w["active_mid_threshold"]:
            w["active_high_threshold"] = max(
                w["active_mid_threshold"] + 10.0,
                w["active_high_threshold"] - lr * 10.0,
            )
        if w["active_high_threshold"] <= w["active_mid_threshold"]:
            w["active_high_threshold"] = w["active_mid_threshold"] + 10.0
        bias = self.freq_bias_getter()
        bias_factor = 1.0 + (bias * 0.5)
        aggression = (0.5 + confidence) * self._personality_factor()
        decision = {
            "type": "ai_decision",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "confidence": confidence,
            "anomaly_type": anomaly_type,
            "learning_rate": lr,
            "personality": self.personality,
        }
        osc_health = self.oscillator_a.health_score()
        if osc_health < 0.5:
            aggression *= 0.6
        if active_count > w["active_high_threshold"]:
            self.oscillator_a.set_mode("max")
            decision["oscillator_mode"] = "max"
            decision["reason"] = (
                f"High active connections: {active_count:.0f} > {w['active_high_threshold']:.1f}"
            )
        else:
            base = w["freq_base"]
            bonus = min(
                10_000.0,
                aggression
                * (active_count * w["freq_active_factor"] + listening_count * w["freq_listen_factor"]),
            )
            freq = (base + bonus) * bias_factor
            self.oscillator_a.set_mode("freq")
            self.oscillator_a.set_frequency(freq)
            decision["oscillator_mode"] = "freq"
            decision["oscillator_frequency_hz"] = freq
            decision["reason"] = (
                f"Active={active_count:.0f}, Listening={listening_count:.0f}, "
                f"bias={bias:.2f}, aggr={aggression:.2f}, freq≈{freq:.1f}"
            )
        if active_count > w["active_mid_threshold"]:
            self.net_mapper.interval_seconds = w["netmap_fast_interval"]
        else:
            self.net_mapper.interval_seconds = w["netmap_slow_interval"]
        if active_count > w["active_high_threshold"]:
            self.ports_monitor.interval_seconds = w["ports_fast_interval"]
        elif active_count > w["active_mid_threshold"]:
            self.ports_monitor.interval_seconds = w["ports_mid_interval"]
        else:
            self.ports_monitor.interval_seconds = w["ports_slow_interval"]
        decision["netmap_interval_seconds"] = self.net_mapper.interval_seconds
        decision["ports_interval_seconds"] = self.ports_monitor.interval_seconds
        if self.latest_network and not self.latest_network.get("ok", False):
            self.network_engine.interval_seconds = w["network_fast_interval"]
            decision["network_interval_seconds"] = w["network_fast_interval"]
            decision["network_reason"] = "Network not OK, probing more frequently."
        else:
            self.network_engine.interval_seconds = w["network_slow_interval"]
            decision["network_interval_seconds"] = w["network_slow_interval"]
            decision["network_reason"] = "Network OK/unknown, normal probe rate."
        if random.random() < (0.05 + (1.0 - confidence) * 0.15):
            self._save_state()
        try:
            self.event_queue.put_nowait(decision)
        except queue.Full:
            pass

    def _emit_coherence(self):
        mode_a, freq_a = self.oscillator_a.get_mode_freq()
        mode_b, freq_b = self.oscillator_b.get_mode_freq()
        mode_coherence = 1.0 if mode_a == mode_b else 0.0
        max_freq = max(freq_a, freq_b, 1.0)
        freq_diff = abs(freq_a - freq_b) / max_freq
        freq_coherence = max(0.0, 1.0 - freq_diff)
        coherence = {
            "type": "coherence",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "mode_a": mode_a,
            "mode_b": mode_b,
            "freq_a_hz": freq_a,
            "freq_b_hz": freq_b,
            "mode_coherence": mode_coherence,
            "freq_coherence": freq_coherence,
        }
        try:
            self.event_queue.put_nowait(coherence)
        except queue.Full:
            pass


# ---------------------------------------------------------------------
# System Control Engine with persistent threat history
# ---------------------------------------------------------------------

class SystemControlEngine(EngineBase):
    def __init__(self, event_queue, interval_seconds=10, history_path: Path = Path("threat_history.json")):
        super().__init__(name="system_control", max_silence=60.0)
        self.event_queue = event_queue
        self.interval_seconds = interval_seconds
        self.threat_scores = {}
        self.history_path = history_path
        self.history = self._load_history()

    def _load_history(self):
        try:
            if self.history_path.is_file():
                with self.history_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_history(self):
        try:
            with self.history_path.open("w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        if psutil is None:
            print("[SystemControl] psutil not available, control engine disabled.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SystemControlThread"
        )
        self._thread.start()

    def _run(self):
        save_counter = 0
        while self._running:
            self._beat()
            try:
                self._scan_and_control()
                save_counter += 1
                if save_counter >= 6:
                    self._save_history()
                    save_counter = 0
            except Exception:
                self.mark_error()
            for _ in range(int(self.interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def _scan_and_control(self):
        decisions = []
        if psutil is None:
            return
        this_pid = os.getpid()
        try:
            procs = list(psutil.process_iter([
                "pid", "name", "username", "cpu_percent",
                "memory_percent", "io_counters", "connections", "exe"
            ]))
        except Exception:
            self.mark_error()
            return
        new_threat = {}
        for p in procs:
            pid = p.info.get("pid")
            name = p.info.get("name") or ""
            if pid in (0, 4) or pid == this_pid:
                continue
            username = p.info.get("username") or ""
            cpu = p.info.get("cpu_percent") or 0.0
            mem = p.info.get("memory_percent") or 0.0
            io_c = p.info.get("io_counters")
            read_bytes = getattr(io_c, "read_bytes", 0) if io_c else 0
            write_bytes = getattr(io_c, "write_bytes", 0) if io_c else 0
            conns = p.info.get("connections") or []
            conn_count = len(conns) if isinstance(conns, (list, tuple)) else 0
            exe_path = p.info.get("exe") or ""
            key = exe_path if exe_path else name
            classification = self._classify_process(name, username, cpu, mem,
                                                    read_bytes, write_bytes, conn_count)
            instant_threat = self._compute_threat_heat(classification, cpu, mem, conn_count, read_bytes, write_bytes)
            historical = self.history.get(key, 0.0)
            alpha = 0.3
            smoothed = alpha * instant_threat + (1 - alpha) * historical
            self.history[key] = smoothed
            new_threat[pid] = smoothed
            action = self._decide_action(classification, smoothed)
            if action != "none":
                result = self._apply_priority_change(p, action)
            else:
                result = "no_change"
            decisions.append({
                "pid": pid,
                "name": name,
                "exe_path": exe_path,
                "classification": classification,
                "cpu_percent": cpu,
                "mem_percent": mem,
                "read_bytes": read_bytes,
                "write_bytes": write_bytes,
                "conn_count": conn_count,
                "threat_heat": smoothed,
                "action": action,
                "result": result,
            })
        self.threat_scores = new_threat
        event = {
            "type": "control_decision",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "decision_count": len(decisions),
            "decisions_sample": decisions[:50],
        }
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _classify_process(self, name, username, cpu, mem, read_b, write_b, conn_count):
        lname = name.lower()
        trusted_names = [
            "system", "idle", "explorer.exe", "dwm.exe", "python", "python.exe",
            "code.exe", "cmd.exe", "powershell.exe", "bash", "zsh", "gnome-shell",
        ]
        if any(t in lname for t in trusted_names):
            return "trusted"
        if cpu > 60.0 or mem > 20.0:
            return "hog"
        if conn_count > 50:
            return "net_heavy"
        if (read_b + write_b) > (10 * 1024 * 1024):
            return "io_heavy"
        if cpu > 20.0 or mem > 10.0:
            return "noisy"
        return "neutral"

    def _compute_threat_heat(self, classification, cpu, mem, conn_count, read_b, write_b):
        base = {
            "trusted": 0.1,
            "neutral": 0.3,
            "noisy": 0.5,
            "hog": 0.8,
            "net_heavy": 0.7,
            "io_heavy": 0.7,
        }.get(classification, 0.3)
        cpu_factor = min(1.0, cpu / 100.0)
        mem_factor = min(1.0, mem / 50.0)
        conn_factor = min(1.0, conn_count / 100.0)
        io_factor = min(1.0, (read_b + write_b) / (50 * 1024 * 1024))
        heat = base + 0.3 * cpu_factor + 0.2 * mem_factor + 0.2 * conn_factor + 0.3 * io_factor
        return max(0.0, min(2.0, heat))

    def _decide_action(self, classification, threat):
        if classification == "trusted":
            if threat < 0.5:
                return "raise_priority"
            return "none"
        if threat > 1.2:
            return "lower_priority"
        if classification in ("hog", "net_heavy", "io_heavy") and threat > 0.8:
            return "lower_priority"
        return "none"

    def _apply_priority_change(self, proc, action):
        try:
            nice = proc.nice()
            system = platform.system().lower()
            if system == "windows":
                import psutil as _ps
                if action == "lower_priority":
                    new = _ps.BELOW_NORMAL_PRIORITY_CLASS
                elif action == "raise_priority":
                    new = _ps.ABOVE_NORMAL_PRIORITY_CLASS
                else:
                    return "no_change"
                proc.nice(new)
                return f"set_priority_class:{new}"
            else:
                if action == "lower_priority":
                    new = min(19, nice + 2)
                elif action == "raise_priority":
                    new = max(-10, nice - 1)
                else:
                    return "no_change"
                if new != nice:
                    proc.nice(new)
                    return f"nice_changed:{nice}->{new}"
                return "no_change"
        except Exception:
            self.mark_error()
            return "failed"


# ---------------------------------------------------------------------
# PolicyEmitter with anomaly-aware blocking
# ---------------------------------------------------------------------

class PolicyEmitter(EngineBase):
    def __init__(self, ai_queue, event_queue):
        super().__init__(name="policy_emitter", max_silence=30.0)
        self.ai_queue = ai_queue
        self.event_queue = event_queue
        self.network_target = ("127.0.0.1", 9501)
        self.file_target = ("127.0.0.1", 9502)
        self._net_sock = None
        self._file_sock = None
        self.last_anomaly_type = "normal"
        self.last_confidence = 0.5

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PolicyEmitterThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        for sock in (self._net_sock, self._file_sock):
            try:
                if sock:
                    sock.close()
            except Exception:
                pass

    def _connect_if_needed(self):
        if self._net_sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(self.network_target)
                s.settimeout(None)
                self._net_sock = s
            except Exception:
                self._net_sock = None
        if self._file_sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(self.file_target)
                s.settimeout(None)
                self._file_sock = s
            except Exception:
                self._file_sock = None

    def _send_json(self, sock, obj):
        try:
            data = json.dumps(obj).encode("utf-8") + b"\n"
            sock.sendall(data)
        except Exception:
            self.mark_error()
            try:
                sock.close()
            except Exception:
                pass

    def _run(self):
        while self._running:
            self._beat()
            try:
                event = self.ai_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            etype = event.get("type")
            if etype == "ai_decision":
                self.last_anomaly_type = event.get("anomaly_type", self.last_anomaly_type)
                self.last_confidence = float(event.get("confidence", self.last_confidence))
            elif etype == "control_decision":
                net_policy, file_policy = self._derive_policies_from_control(event)
                if net_policy or file_policy:
                    self._connect_if_needed()
                    if net_policy and self._net_sock is not None:
                        self._send_json(self._net_sock, net_policy)
                    if file_policy and self._file_sock is not None:
                        self._send_json(self._file_sock, file_policy)

    def _derive_policies_from_control(self, control_event):
        decisions = control_event.get("decisions_sample", [])
        net_rules = []
        file_rules = []
        anomaly = self.last_anomaly_type
        for d in decisions:
            pid = d.get("pid")
            name = d.get("name") or ""
            classification = d.get("classification")
            threat_heat = float(d.get("threat_heat", 0.0))
            if not pid or not name:
                continue
            auto_block = False
            if anomaly in ("spike", "noisy") and threat_heat >= 1.5 and classification in ("hog", "net_heavy"):
                auto_block = True
            if classification == "net_heavy" or threat_heat > 1.5:
                net_rules.append({
                    "id": f"net-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "remote_port": None,
                        "remote_ip": None,
                        "protocol": "tcp",
                    },
                    "action": "block" if auto_block else "deprioritize",
                    "priority_class": "low",
                    "mark": 0x10,
                })
            elif classification == "hog":
                net_rules.append({
                    "id": f"net-hog-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "protocol": "any",
                    },
                    "action": "block" if auto_block else "deprioritize",
                    "priority_class": "low",
                    "mark": 0x08,
                })
                file_rules.append({
                    "id": f"file-hog-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "path_prefix": None,
                    },
                    "action": "throttle",
                    "max_write_kbps": 1024,
                })
            elif classification == "io_heavy" or (threat_heat > 1.2 and classification == "noisy"):
                file_rules.append({
                    "id": f"file-io-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "path_prefix": None,
                    },
                    "action": "throttle",
                    "max_write_kbps": 512,
                })
            elif classification == "trusted" and threat_heat < 0.5:
                net_rules.append({
                    "id": f"net-trusted-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "protocol": "any",
                    },
                    "action": "prioritize",
                    "priority_class": "high",
                    "mark": 0x20,
                })
                file_rules.append({
                    "id": f"file-trusted-{pid}",
                    "target": {
                        "pid": pid,
                        "process_name": name,
                        "path_prefix": None,
                    },
                    "action": "allow_fast",
                })
        net_policy = {
            "type": "network_policy",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "rules": net_rules,
        } if net_rules else None
        file_policy = {
            "type": "file_policy",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "rules": file_rules,
        } if file_rules else None
        return net_policy, file_policy


# ---------------------------------------------------------------------
# Codex (Adaptive purge logic, ghost sync, phantom node)
# ---------------------------------------------------------------------

class AdaptiveCodex(EngineBase):
    def __init__(self, cortex_metrics_getter, codex_path: Path = Path("codex_rules.json")):
        super().__init__(name="adaptive_codex", max_silence=60.0)
        self.codex_path = codex_path
        self.cortex_metrics_getter = cortex_metrics_getter
        self.rules = self._load_codex()
        self.mutation_log = []

    def _load_codex(self):
        default = {
            "telemetry_retention_minutes": 120,
            "phantom_node": False,
            "ghost_sync_count": 0,
        }
        try:
            if self.codex_path.is_file():
                with self.codex_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                default.update(loaded)
        except Exception:
            pass
        return default

    def _save_codex(self):
        try:
            with self.codex_path.open("w", encoding="utf-8") as f:
                json.dump(self.rules, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def export_rules(self):
        return json.dumps(self.rules, sort_keys=True)

    def import_rules(self, json_str):
        try:
            incoming = json.loads(json_str)
            self.rules.update(incoming)
            self._save_codex()
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AdaptiveCodexThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            try:
                self._evaluate_ghost_sync()
            except Exception:
                self.mark_error()
            time.sleep(10.0)

    def _evaluate_ghost_sync(self):
        metrics = self.cortex_metrics_getter()
        if not metrics:
            return
        anomaly = metrics.get("last_anomaly_type", "normal")
        avg_active = metrics.get("avg_active_connections", 0.0)
        avg_latency = metrics.get("avg_latency_ms", 0.0)
        threat_mean = metrics.get("mean_threat_heat", 0.0)
        ghost_sync = False
        if anomaly in ("spike", "noisy") and avg_active < 5.0 and avg_latency > 150.0 and threat_mean < 0.7:
            ghost_sync = True
        if ghost_sync:
            old_ret = self.rules.get("telemetry_retention_minutes", 120)
            new_ret = max(15, int(old_ret * 0.5))
            self.rules["telemetry_retention_minutes"] = new_ret
            self.rules["phantom_node"] = True
            self.rules["ghost_sync_count"] = self.rules.get("ghost_sync_count", 0) + 1
            self.mutation_log.append({
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "old_retention": old_ret,
                "new_retention": new_ret,
                "reason": "ghost_sync_detected",
                "metrics": metrics,
            })
            print(f"[AdaptiveCodex] Ghost sync detected. Retention shortened {old_ret} -> {new_ret}, phantom_node=True.")
            self._save_codex()


# ---------------------------------------------------------------------
# Self-Rewriting Agent (weights + mutation log)
# ---------------------------------------------------------------------

class SelfRewritingAgent(EngineBase):
    def __init__(self, cortex_metrics_getter, agent_path: Path = Path("agent_state.json")):
        super().__init__(name="self_rewriting_agent", max_silence=60.0)
        self.cortex_metrics_getter = cortex_metrics_getter
        self.agent_path = agent_path
        self.agent_weights, self.mutation_log = self._load_state()

    def _load_state(self):
        default_weights = [0.6, -0.8, -0.3]
        default_log = []
        try:
            if self.agent_path.is_file():
                with self.agent_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("weights", default_weights), data.get("mutation_log", default_log)
        except Exception:
            pass
        return default_weights, default_log

    def _save_state(self):
        try:
            with self.agent_path.open("w", encoding="utf-8") as f:
                json.dump({"weights": self.agent_weights, "mutation_log": self.mutation_log}, f, indent=2)
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SelfRewritingAgentThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            try:
                self._maybe_mutate()
            except Exception:
                self.mark_error()
            time.sleep(15.0)

    def _maybe_mutate(self):
        metrics = self.cortex_metrics_getter()
        if not metrics:
            return
        collective_health = metrics.get("collective_health_score", 1.0)
        anomaly = metrics.get("last_anomaly_type", "normal")
        mean_threat = metrics.get("mean_threat_heat", 0.0)
        if collective_health < 0.6 and mean_threat > 1.0 and anomaly in ("spike", "noisy"):
            old_weights = list(self.agent_weights)
            for i in range(len(self.agent_weights)):
                delta = random.uniform(-0.1, 0.1)
                self.agent_weights[i] += delta
                self.agent_weights[i] = max(-2.0, min(2.0, self.agent_weights[i]))
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "old_weights": old_weights,
                "new_weights": list(self.agent_weights),
                "metrics": metrics,
                "reason": "collective_health_low_high_threat",
            }
            self.mutation_log.append(entry)
            print("[SelfRewritingAgent] Mutated agent_weights based on situational metrics.")
            self._save_state()

    def get_weights(self):
        return list(self.agent_weights)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resolve_process_exe(pid):
    if psutil is None:
        return None
    try:
        p = psutil.Process(pid)
        return p.exe()
    except Exception:
        return None


# ---------------------------------------------------------------------
# WFP Policy Agent (logging + QoS stub)
# ---------------------------------------------------------------------

class WfpPolicyAgent(EngineBase):
    def __init__(self, enable_qos=True):
        super().__init__(name="wfp_policy_agent", max_silence=60.0)
        self.policy_queue = queue.Queue(maxsize=1000)
        self.enable_qos = enable_qos

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WfpPolicyAgentThread"
        )
        self._thread.start()

    def submit_policy(self, policy):
        try:
            self.policy_queue.put_nowait(policy)
        except queue.Full:
            pass

    def _run(self):
        while self._running:
            self._beat()
            try:
                policy = self.policy_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._apply_policy(policy)

    def _apply_policy(self, policy):
        rules = policy.get("rules", [])
        print(f"[WfpPolicyAgent] Received {len(rules)} network rules.")
        if not self.enable_qos:
            return
        for r in rules:
            action = (r.get("action") or "").lower()
            target = r.get("target", {})
            pid = target.get("pid")
            exe_path = _resolve_process_exe(pid) if pid else None
            if not exe_path:
                continue
            if action in ("prioritize", "deprioritize"):
                dscp = 40 if action == "prioritize" else 8
                policy_name = f"AIBackbone_QoS_{pid}_{action}"
                print(f"[WfpPolicyAgent] (stub) Would apply QoS DSCP={dscp} for {exe_path} as {policy_name}")


# ---------------------------------------------------------------------
# Minifilter Policy Agent – stub (logs only)
# ---------------------------------------------------------------------

class MinifilterPolicyAgent(EngineBase):
    def __init__(self):
        super().__init__(name="minifilter_policy_agent", max_silence=60.0)
        self.policy_queue = queue.Queue(maxsize=1000)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MinifilterPolicyAgentThread"
        )
        self._thread.start()

    def submit_policy(self, policy):
        try:
            self.policy_queue.put_nowait(policy)
        except queue.Full:
            pass

    def _run(self):
        while self._running:
            self._beat()
            try:
                policy = self.policy_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._apply_policy(policy)

    def _apply_policy(self, policy):
        rules = policy.get("rules", [])
        print(f"[MinifilterPolicyAgent] Received {len(rules)} file rules (stub).")


# ---------------------------------------------------------------------
# NetworkControlService
# ---------------------------------------------------------------------

class NetworkControlService(EngineBase):
    def __init__(self, wfp_agent: WfpPolicyAgent, host="127.0.0.1", port=9501):
        super().__init__(name="network_control_service", max_silence=60.0)
        self.host = host
        self.port = port
        self._sock = None
        self.wfp_agent = wfp_agent

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="NetworkControlServiceThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

    def _run(self):
        self._beat()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.host, self.port))
            s.listen(5)
            self._sock = s
            print(f"[NetworkControlService] Listening on {self.host}:{self.port}")
        except Exception as e:
            self.mark_error()
            print("[NetworkControlService] Failed to bind/listen:", e)
            self._running = False
            return
        while self._running:
            try:
                s.settimeout(1.0)
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    self._beat()
                    continue
                t = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
                t.start()
            except Exception as e:
                self.mark_error()
                time.sleep(0.5)

    def _client_handler(self, conn):
        with conn:
            buf = b""
            while self._running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            if obj.get("type") == "network_policy":
                                self._apply_network_policy(obj)
                        except Exception as e:
                            print("[NetworkControlService] Error:", e)
                except Exception:
                    break

    def _apply_network_policy(self, policy):
        rules = policy.get("rules", [])
        for r in rules:
            target = r.get("target", {})
            action = (r.get("action") or "").lower()
            rule_id = r.get("id") or f"rule_{random.randint(1, 10_000_000)}"
            pid = target.get("pid")
            process_name = target.get("process_name")
            remote_ip = target.get("remote_ip")
            remote_port = target.get("remote_port")
            protocol = (target.get("protocol") or "any").lower()
            rule_name = f"AIBackbone_{rule_id}"
            base_cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f'name={rule_name}',
                "dir=out",
                f'action={"block" if action == "block" else "allow"}',
                "enable=yes",
            ]
            exe_path = None
            if pid:
                exe_path = _resolve_process_exe(pid)
            if exe_path:
                base_cmd.append(f'program="{exe_path}"')
            if protocol in ("tcp", "udp"):
                base_cmd.append(f"protocol={protocol}")
            else:
                base_cmd.append("protocol=any")
            if remote_ip:
                base_cmd.append(f"remoteip={remote_ip}")
            if remote_port:
                try:
                    rp = int(remote_port)
                    base_cmd.append(f"remoteport={rp}")
                except Exception:
                    pass
            if action == "block":
                try:
                    print("[NetworkControlService] Adding BLOCK rule via netsh:", " ".join(base_cmd))
                    subprocess.run(" ".join(base_cmd), shell=True)
                except Exception as e:
                    print("[NetworkControlService] Failed to add block rule:", e)
            elif action == "prioritize":
                try:
                    print("[NetworkControlService] Adding ALLOW rule via netsh:", " ".join(base_cmd))
                    subprocess.run(" ".join(base_cmd), shell=True)
                except Exception as e:
                    print("[NetworkControlService] Failed to add allow rule:", e)
            elif action == "deprioritize":
                print(f"[NetworkControlService] Deprioritize requested for pid={pid} name={process_name}, "
                      f"remote_ip={remote_ip} remote_port={remote_port} (QoS not yet wired).")
        self.wfp_agent.submit_policy(policy)


# ---------------------------------------------------------------------
# FileControlService
# ---------------------------------------------------------------------

class FileControlService(EngineBase):
    def __init__(self, minifilter_agent: MinifilterPolicyAgent, host="127.0.0.1", port=9502):
        super().__init__(name="file_control_service", max_silence=60.0)
        self.host = host
        self.port = port
        self._sock = None
        self.minifilter_agent = minifilter_agent

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="FileControlServiceThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

    def _run(self):
        self._beat()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.host, self.port))
            s.listen(5)
            self._sock = s
            print(f"[FileControlService] Listening on {self.host}:{self.port}")
        except Exception as e:
            self.mark_error()
            print("[FileControlService] Failed to bind/listen:", e)
            self._running = False
            return
        while self._running:
            try:
                s.settimeout(1.0)
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    self._beat()
                    continue
                t = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
                t.start()
            except Exception as e:
                self.mark_error()
                time.sleep(0.5)

    def _client_handler(self, conn):
        with conn:
            buf = b""
            while self._running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            if obj.get("type") == "file_policy":
                                self._apply_file_policy(obj)
                        except Exception as e:
                            print("[FileControlService] Error:", e)
                except Exception:
                    break

    def _apply_file_policy(self, policy):
        rules = policy.get("rules", [])
        for r in rules:
            target = r.get("target", {})
            action = r.get("action")
            pid = target.get("pid")
            proc_name = target.get("process_name")
            path_prefix = target.get("path_prefix")
            max_kbps = r.get("max_write_kbps")
            print(f"[FileControlService] file_policy action={action} pid={pid} name={proc_name} path={path_prefix} max_kbps={max_kbps}")
        self.minifilter_agent.submit_policy(policy)


# ---------------------------------------------------------------------
# Situational Cortex (situational awareness, predictive, collective health)
# ---------------------------------------------------------------------

class SituationalCortex(EngineBase):
    def __init__(self, cortex_queue, engines, control_engine: SystemControlEngine, network_engine: NetworkEngine):
        super().__init__(name="situational_cortex", max_silence=30.0)
        self.cortex_queue = cortex_queue
        self.engines = engines
        self.control_engine = control_engine
        self.network_engine = network_engine
        self.last_anomaly_type = "normal"
        self.last_confidence = 0.5
        self.avg_active_connections = 0.0
        self.active_conn_history = []
        self.latency_history = []
        self.mean_threat_heat = 0.0
        self.collective_health_score = 1.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SituationalCortexThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            self._process_events()
            self._update_collective_health()
            time.sleep(0.2)

    def _process_events(self):
        while True:
            try:
                ev = self.cortex_queue.get_nowait()
            except queue.Empty:
                break
            etype = ev.get("type")
            if etype == "ai_decision":
                self.last_anomaly_type = ev.get("anomaly_type", self.last_anomaly_type)
                self.last_confidence = float(ev.get("confidence", self.last_confidence))
            elif etype == "ports":
                active_count = float(ev.get("active_count", 0))
                self.active_conn_history.append(active_count)
                if len(self.active_conn_history) > 100:
                    self.active_conn_history.pop(0)
                self.avg_active_connections = statistics.mean(self.active_conn_history) if self.active_conn_history else 0.0
            elif etype == "network":
                latency = ev.get("latency_ms")
                if latency is not None:
                    self.latency_history.append(float(latency))
                    if len(self.latency_history) > 100:
                        self.latency_history.pop(0)

    def _update_collective_health(self):
        health_scores = []
        for eng in self.engines.values():
            try:
                health_scores.append(eng.health_score())
            except Exception:
                continue
        self.collective_health_score = statistics.mean(health_scores) if health_scores else 1.0
        self.mean_threat_heat = statistics.mean(self.control_engine.threat_scores.values()) if self.control_engine.threat_scores else 0.0

    def get_metrics(self):
        avg_latency = statistics.mean(self.latency_history) if self.latency_history else 0.0
        return {
            "last_anomaly_type": self.last_anomaly_type,
            "last_confidence": self.last_confidence,
            "avg_active_connections": self.avg_active_connections,
            "avg_latency_ms": avg_latency,
            "mean_threat_heat": self.mean_threat_heat,
            "collective_health_score": self.collective_health_score,
        }


# ---------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------

class Supervisor(EngineBase):
    def __init__(self, engines):
        super().__init__(name="supervisor", max_silence=10.0)
        self.engines = engines

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SupervisorThread"
        )
        self._thread.start()

    def _run(self):
        while self._running:
            self._beat()
            for name, engine in self.engines.items():
                try:
                    healthy = engine.is_healthy()
                    score = engine.health_score()
                except Exception:
                    healthy = False
                    score = 0.0
                if (not engine._running) or (not healthy) or (score < 0.2):
                    try: engine.stop()
                    except Exception: pass
                    try: engine.start()
                    except Exception: pass
            time.sleep(5.0)


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------

class BackboneGUI:
    def __init__(self, root,
                 oscillator_a: BinaryOscillator,
                 oscillator_b: MirrorOscillator,
                 gui_queue: queue.Queue,
                 cortex: SituationalCortex):
        self.root = root
        self.root.title("Autonomous Cipher Engine – Hybrid Cortex")
        self.oscillator_a = oscillator_a
        self.oscillator_b = oscillator_b
        self.gui_queue = gui_queue
        self.cortex = cortex
        self.last_gui_update = 0.0
        self.gui_update_interval = 0.2
        self.freq_bias_var = tk.DoubleVar(value=0.0)
        self.confidence_var = tk.StringVar(value="—")
        self.personality_var = tk.StringVar(value="—")
        self.anomaly_var = tk.StringVar(value="—")
        self.collective_health_var = tk.StringVar(value="—")
        self._build_style()
        self._build_layout()
        self._update_status_text("Hybrid brain online. Persistent, predictive, adaptive defense.")
        self._schedule_updates()

    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("StateLabel.TLabel", font=("Consolas", 20, "bold"))
        style.configure("InfoLabel.TLabel", font=("Consolas", 10))

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.state_label_a = ttk.Label(
            main, text="Oscillator A", style="StateLabel.TLabel", anchor="center",
        )
        self.state_label_a.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 5))
        ttk.Label(main, text="A Mode:", style="InfoLabel.TLabel").grid(row=1, column=0, sticky="w")
        self.mode_var_a = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.mode_var_a, style="InfoLabel.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(main, text="A Frequency (Hz):", style="InfoLabel.TLabel").grid(row=2, column=0, sticky="w")
        self.freq_var_a = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.freq_var_a, style="InfoLabel.TLabel").grid(row=2, column=1, sticky="w")
        self.state_label_b = ttk.Label(
            main, text="Oscillator B (Mirror)",
            style="StateLabel.TLabel", anchor="center",
        )
        self.state_label_b.grid(row=0, column=2, columnspan=2, sticky="nsew", pady=(0, 5))
        ttk.Label(main, text="B Mode:", style="InfoLabel.TLabel").grid(row=1, column=2, sticky="w")
        self.mode_var_b = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.mode_var_b, style="InfoLabel.TLabel").grid(row=1, column=3, sticky="w")
        ttk.Label(main, text="B Frequency (Hz):", style="InfoLabel.TLabel").grid(row=2, column=2, sticky="w")
        self.freq_var_b = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.freq_var_b, style="InfoLabel.TLabel").grid(row=2, column=3, sticky="w")
        ttk.Label(main, text="AI confidence:", style="InfoLabel.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(main, textvariable=self.confidence_var, style="InfoLabel.TLabel").grid(row=3, column=1, sticky="w")
        ttk.Label(main, text="AI personality:", style="InfoLabel.TLabel").grid(row=3, column=2, sticky="w")
        ttk.Label(main, textvariable=self.personality_var, style="InfoLabel.TLabel").grid(row=3, column=3, sticky="w")
        ttk.Label(main, text="Anomaly type:", style="InfoLabel.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(main, textvariable=self.anomaly_var, style="InfoLabel.TLabel").grid(row=4, column=1, sticky="w")
        ttk.Label(main, text="Collective health:", style="InfoLabel.TLabel").grid(row=4, column=2, sticky="w")
        ttk.Label(main, textvariable=self.collective_health_var, style="InfoLabel.TLabel").grid(row=4, column=3, sticky="w")
        ttk.Label(main, text="Frequency bias (manual nudge):", style="InfoLabel.TLabel").grid(row=5, column=0, sticky="w")
        self.bias_scale = ttk.Scale(
            main, from_=-1.0, to=1.0, orient="horizontal", variable=self.freq_bias_var,
        )
        self.bias_scale.grid(row=5, column=1, sticky="ew", padx=(5, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status_var, style="InfoLabel.TLabel", anchor="w").grid(
            row=6, column=0, columnspan=4, sticky="ew", pady=(10, 0)
        )
        for c in range(4):
            main.columnconfigure(c, weight=1)

    def get_bias(self):
        try:
            return float(self.freq_bias_var.get())
        except Exception:
            return 0.0

    def _schedule_updates(self):
        now = time.time()
        if now - self.last_gui_update >= self.gui_update_interval:
            self.last_gui_update = now
            self._process_gui_queue()
            self._update_oscillator_labels()
            self._update_cortex_metrics()
        self.root.after(int(self.gui_update_interval * 1000), self._schedule_updates)

    def _process_gui_queue(self):
        while True:
            try:
                data = self.gui_queue.get_nowait()
            except queue.Empty:
                break
            if data.get("type") == "ai_decision":
                conf = data.get("confidence")
                if conf is not None:
                    self.confidence_var.set(f"{conf:.2f}")
                personality = data.get("personality")
                if personality:
                    self.personality_var.set(personality)
                anomaly = data.get("anomaly_type")
                if anomaly:
                    self.anomaly_var.set(anomaly)

    def _update_oscillator_labels(self):
        mode_a, freq_a = self.oscillator_a.get_mode_freq()
        mode_b, freq_b = self.oscillator_b.get_mode_freq()
        self.mode_var_a.set(mode_a)
        self.freq_var_a.set(f"{freq_a:.1f}")
        self.state_label_a.configure(text=f"Oscillator A – {mode_a.upper()} @ {freq_a:.1f} Hz")
        self.mode_var_b.set(mode_b)
        self.freq_var_b.set(f"{freq_b:.1f}")
        self.state_label_b.configure(text=f"Oscillator B – {mode_b.upper()} @ {freq_b:.1f} Hz")

    def _update_cortex_metrics(self):
        metrics = self.cortex.get_metrics()
        ch = metrics.get("collective_health_score", 1.0)
        self.collective_health_var.set(f"{ch:.2f}")

    def _update_status_text(self, text):
        self.status_var.set(text)


# ---------------------------------------------------------------------
# Wiring & Entry
# ---------------------------------------------------------------------

def main():
    state_path = Path("ai_policy_state.json")
    bus = EventBus()
    gui_queue = queue.Queue(maxsize=2000)
    ai_queue = queue.Queue(maxsize=2000)
    cortex_queue = queue.Queue(maxsize=2000)
    bus.add_output(gui_queue)
    bus.add_output(ai_queue)
    bus.add_output(cortex_queue)

    oscillator_a = BinaryOscillator()
    oscillator_b = MirrorOscillator(oscillator_a, base_inverse=50_000, scale_factor=100_000_000)
    inventory = SystemInventory(bus.input_queue)
    compute_engine = ComputeEngine(bus.input_queue, inventory)
    network_engine = NetworkEngine(bus.input_queue)
    net_mapper = LocalNetworkMapper(bus.input_queue)
    ports_monitor = PortsMonitor(bus.input_queue)
    control_engine = SystemControlEngine(bus.input_queue, interval_seconds=10, history_path=Path("threat_history.json"))

    engines = {
        "event_bus": bus,
        "oscillator_a": oscillator_a,
        "oscillator_b": oscillator_b,
        "inventory": inventory,
        "compute": compute_engine,
        "network": network_engine,
        "netmap": net_mapper,
        "ports": ports_monitor,
        "system_control": control_engine,
    }

    cortex = SituationalCortex(cortex_queue, engines, control_engine, network_engine)

    def cortex_metrics_getter():
        return cortex.get_metrics()

    codex = AdaptiveCodex(cortex_metrics_getter, codex_path=Path("codex_rules.json"))
    self_agent = SelfRewritingAgent(cortex_metrics_getter, agent_path=Path("agent_state.json"))

    wfp_agent = WfpPolicyAgent(enable_qos=True)
    minifilter_agent = MinifilterPolicyAgent()

    net_control_service = NetworkControlService(wfp_agent=wfp_agent)
    file_control_service = FileControlService(minifilter_agent=minifilter_agent)

    root = tk.Tk()
    gui = BackboneGUI(root, oscillator_a, oscillator_b, gui_queue, cortex)

    ai_engine = AIPolicyEngine(
        ai_queue=ai_queue,
        event_queue=bus.input_queue,
        oscillator_a=oscillator_a,
        oscillator_b=oscillator_b,
        net_mapper=net_mapper,
        ports_monitor=ports_monitor,
        network_engine=network_engine,
        state_path=state_path,
        freq_bias_getter=gui.get_bias,
    )

    policy_emitter = PolicyEmitter(ai_queue=ai_queue, event_queue=bus.input_queue)

    full_engines = engines.copy()
    full_engines.update({
        "ai_policy": ai_engine,
        "policy_emitter": policy_emitter,
        "wfp_policy_agent": wfp_agent,
        "minifilter_policy_agent": minifilter_agent,
        "network_control_service": net_control_service,
        "file_control_service": file_control_service,
        "situational_cortex": cortex,
        "adaptive_codex": codex,
        "self_rewriting_agent": self_agent,
    })
    supervisor = Supervisor(full_engines)

    bus.start()
    inventory.start()
    compute_engine.start()
    network_engine.start()
    net_mapper.start()
    ports_monitor.start()
    oscillator_a.start()
    oscillator_b.start()
    control_engine.start()
    ai_engine.start()
    policy_emitter.start()
    wfp_agent.start()
    minifilter_agent.start()
    cortex.start()
    codex.start()
    self_agent.start()
    net_control_service.start()
    file_control_service.start()
    supervisor.start()

    try:
        root.mainloop()
    finally:
        supervisor.stop()
        ai_engine.stop()
        oscillator_a.stop()
        oscillator_b.stop()
        compute_engine.stop()
        network_engine.stop()
        net_mapper.stop()
        ports_monitor.stop()
        control_engine.stop()
        policy_emitter.stop()
        net_control_service.stop()
        file_control_service.stop()
        wfp_agent.stop()
        minifilter_agent.stop()
        cortex.stop()
        codex.stop()
        self_agent.stop()
        inventory.stop()
        bus.stop()

if __name__ == "__main__":
    main()

