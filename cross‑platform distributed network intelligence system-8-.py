#!/usr/bin/env python3
"""
Borg Universal Cockpit - Windows / Linux / macOS
Now with:
- Per-process flow mapping
- ML-style flow classification scaffold
- Deeper UI automation (intent + app hints)
- Multi-NIC routing scaffold
- Basic threat detection
- Extended macOS backend skeleton for PF divert integration
- Auto-elevation on Windows (PyInstaller-safe)
- Safe-mode fallback when elevation is missing/denied
- Admin/safe-mode indicator in the cockpit
- BernoulliDataEngine (hybrid: per-flow + global)
- Bernoulli-driven routing preference
- Cross-platform routing manipulation organ (real when configured)
- DualPersonalityBot (guardian/rogue persona logger)
- SwarmSyncOrgan (node presence + basic state broadcast)
- NIC discovery + auto-routing configuration
"""

import sys
import os
import platform
import subprocess
import importlib
import threading
import time
import queue
from collections import deque, defaultdict
import socket
import random
import json

# ---------------------------
# Global privilege state
# ---------------------------

IS_ADMIN = False
SAFE_MODE = False  # when True, backends won't start; cockpit runs in monitor-only mode

# ---------------------------
# Elevation / privilege handling
# ---------------------------

def _is_windows_admin():
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _windows_auto_elevate():
    """
    Windows-only auto-elevation.
    - If not admin, relaunch with 'runas'.
    - PyInstaller-safe (handles frozen exe).
    - On failure, enters SAFE_MODE instead of exiting hard.
    """
    global IS_ADMIN, SAFE_MODE
    import ctypes

    if _is_windows_admin():
        IS_ADMIN = True
        return

    try:
        if getattr(sys, "frozen", False):
            script = sys.executable
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
        else:
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])

        print("[Borg] Elevation required. Relaunching as administrator...")
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script}" {params}', None, 1
        )

        if ret <= 32:
            print(f"[Borg] Elevation failed (code={ret}). Entering SAFE MODE.")
            SAFE_MODE = True
            IS_ADMIN = False
            return

        sys.exit(0)
    except Exception as e:
        print(f"[Borg] Elevation failed: {e}. Entering SAFE MODE.")
        SAFE_MODE = True
        IS_ADMIN = False


def _posix_priv_hint():
    """
    Linux/macOS privilege hint.
    - If not root, we warn and enter SAFE_MODE.
    - Operator can still run cockpit, but backends won't start.
    """
    global IS_ADMIN, SAFE_MODE
    try:
        euid = os.geteuid()
    except AttributeError:
        return

    if euid == 0:
        IS_ADMIN = True
        return

    print("[Borg] Not running as root. NFQUEUE/PF divert may fail.")
    print("[Borg] Hint: run with 'sudo' or as root for full interception.")
    SAFE_MODE = True
    IS_ADMIN = False


def ensure_privileges():
    """
    Cross-platform privilege handler.
    - Windows: auto-elevate, SAFE_MODE on failure.
    - Linux/macOS: hint + SAFE_MODE if not root.
    """
    os_name = platform.system().lower()
    if os_name == "windows":
        _windows_auto_elevate()
    elif os_name in ("linux", "darwin"):
        _posix_priv_hint()
    else:
        print("[Borg] Unknown OS privilege model. Running in SAFE MODE.")
        global SAFE_MODE, IS_ADMIN
        SAFE_MODE = True
        IS_ADMIN = False


ensure_privileges()

# ---------------------------
# Autoloader
# ---------------------------

_loaded_modules = {}


def ensure_module(import_name: str, pip_name: str, required: bool = True):
    try:
        module = importlib.import_module(import_name)
        _loaded_modules[import_name] = module
        return module
    except ImportError:
        print(f"[AUTOLOADER] Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            module = importlib.import_module(import_name)
            _loaded_modules[import_name] = module
            return module
        except Exception as e:
            print(f"[AUTOLOADER] Failed to install {pip_name}: {e}")
            if required:
                raise
            return None


import tkinter as tk
from tkinter import ttk

ensure_module("numpy", "numpy", required=False)
ensure_module("matplotlib", "matplotlib", required=True)
ensure_module("matplotlib.backends.backend_tkagg", "matplotlib", required=True)
ensure_module("psutil", "psutil", required=False)
ensure_module("netifaces", "netifaces", required=False)

np = _loaded_modules.get("numpy", None)
matplotlib = _loaded_modules["matplotlib"]
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
psutil = _loaded_modules.get("psutil", None)
netifaces = _loaded_modules.get("netifaces", None)

# ---------------------------
# Game Signature Database
# ---------------------------

GAME_SIGNATURES = {
    27015: "Source Engine",
    3074: "Xbox Live / CoD",
    7777: "Unreal / Fortnite",
    25565: "Minecraft",
    5000: "TestGame",
}

GAME_PROCESS_HINTS = [
    "steam", "epic", "battle.net", "fortnite", "cod", "callofduty",
    "valorant", "csgo", "cs2", "overwatch", "minecraft"
]


def match_game_signature(port: int):
    return GAME_SIGNATURES.get(port, None)


# ---------------------------
# Data Frequency Engine
# ---------------------------

class DataFrequencyEngine:
    def __init__(self):
        self.input_rate_hz = 10.0
        self.output_rate_hz = 50.0

        self.running = False
        self.consumer_thread = None
        self.autotune_thread = None

        self.event_queue = queue.Queue()
        self.produced_count = 0
        self.emitted_count = 0
        self.interpolated_count = 0

        self.lock = threading.Lock()

        self.log_buffer = deque(maxlen=400)

        self.graph_window_sec = 10.0
        self.graph_lock = threading.Lock()
        self.input_events_times = deque()
        self.output_events_times = deque()

        self.auto_tune_enabled = True
        self.target_queue_depth = 5
        self.auto_tune_gain = 1.0

        self.external_producer = True
        self.consumer_callbacks = []

        self.autonomous_mode = True

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self.log_buffer.append(line)

    def set_rates(self, input_rate_hz: float, output_rate_hz: float):
        with self.lock:
            self.input_rate_hz = max(0.1, input_rate_hz)
            self.output_rate_hz = max(0.1, output_rate_hz)
        self.log(
            f"Configured: input={self.input_rate_hz:.2f} Hz, output={self.output_rate_hz:.2f} Hz"
        )

    def set_auto_tune(self, enabled: bool):
        self.auto_tune_enabled = enabled
        self.log(f"Auto-tune {'enabled' if enabled else 'disabled'}.")

    def set_target_queue_depth(self, depth: int):
        self.target_queue_depth = max(0, depth)
        self.log(f"Target queue depth set to {self.target_queue_depth}.")

    def register_consumer(self, callback):
        self.consumer_callbacks.append(callback)
        self.log("Registered consumer callback.")

    def start(self):
        if self.running:
            self.log("Already running.")
            return
        self.running = True
        self.produced_count = 0
        self.emitted_count = 0
        self.interpolated_count = 0

        self.log("Using external producer (backends).")

        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        self.autotune_thread = threading.Thread(target=self._autotune_loop, daemon=True)
        self.autotune_thread.start()

        self.log("Data Frequency Engine started.")

    def stop(self):
        self.running = False
        self.log("Stopping...")
        if self.consumer_thread:
            self.consumer_thread.join(timeout=1.0)
        if self.autotune_thread:
            self.autotune_thread.join(timeout=1.0)
        self.log("Stopped.")

    def _consumer_loop(self):
        last_ts = None
        while self.running:
            with self.lock:
                period = 1.0 / self.output_rate_hz
            time.sleep(period)

            try:
                ts = None
                while True:
                    ts = self.event_queue.get_nowait()
            except queue.Empty:
                ts = None

            now = time.time()
            if ts is not None:
                last_ts = ts
                self.emitted_count += 1
                self._record_output_event(now)
            else:
                if last_ts is not None:
                    self.interpolated_count += 1
                    self.emitted_count += 1
                    self._record_output_event(now)

            for cb in self.consumer_callbacks:
                try:
                    cb(now)
                except Exception as e:
                    self.log(f"Consumer callback error: {e}")

    def _autotune_loop(self):
        while self.running:
            time.sleep(0.5)
            if not self.auto_tune_enabled:
                continue

            qsize = self.event_queue.qsize()
            error = qsize - self.target_queue_depth

            with self.lock:
                self.output_rate_hz = max(
                    0.1,
                    self.output_rate_hz + self.auto_tune_gain * (-error)
                )

                if self.autonomous_mode:
                    if qsize < 2:
                        self.output_rate_hz *= 1.05
                    if qsize > 50:
                        self.output_rate_hz *= 0.90

    def _record_input_event(self, ts: float):
        with self.graph_lock:
            self.input_events_times.append(ts)
            self._trim_graph_buffers(ts)

    def _record_output_event(self, ts: float):
        with self.graph_lock:
            self.output_events_times.append(ts)
            self._trim_graph_buffers(ts)

    def _trim_graph_buffers(self, now: float):
        cutoff = now - self.graph_window_sec
        while self.input_events_times and self.input_events_times[0] < cutoff:
            self.input_events_times.popleft()
        while self.output_events_times and self.output_events_times[0] < cutoff:
            self.output_events_times.popleft()

    def get_graph_series(self):
        with self.graph_lock:
            now = time.time()
            cutoff = now - self.graph_window_sec
            xs_in = [t - cutoff for t in self.input_events_times]
            ys_in = [1.0] * len(xs_in)
            xs_out = [t - cutoff for t in self.output_events_times]
            ys_out = [1.0] * len(xs_out)
        return xs_in, ys_in, xs_out, ys_out

    def get_snapshot(self):
        with self.lock:
            in_rate = self.input_rate_hz
            out_rate = self.output_rate_hz
        snapshot = {
            "input_rate_hz": in_rate,
            "output_rate_hz": out_rate,
            "produced": self.produced_count,
            "emitted": self.emitted_count,
            "interpolated": self.interpolated_count,
            "queue_size": self.event_queue.qsize(),
            "auto_tune_enabled": self.auto_tune_enabled,
            "target_queue_depth": self.target_queue_depth,
            "log_lines": list(self.log_buffer),
            "autonomous_mode": self.autonomous_mode,
        }
        return snapshot


engine = DataFrequencyEngine()

# ---------------------------
# Bernoulli Data Engine (hybrid)
# ---------------------------

class BernoulliDataEngine:
    def __init__(self):
        self.global_velocity = 0.0
        self.global_pressure = 0.0
        self.global_kinetic = 0.0
        self.global_bernoulli = 0.0
        self.global_state = "unknown"

    def compute_flow_metrics(self, flow, queue_depth):
        v = flow["packets_c2s"] + flow["packets_s2c"]
        avg_rtt = 0.0
        if flow["rtt_samples"]:
            avg_rtt = sum(flow["rtt_samples"]) / len(flow["rtt_samples"])
        flow["avg_rtt"] = avg_rtt

        P = (1.0 / (v + 1e-6)) + 0.5 * avg_rtt + 0.1 * queue_depth
        K = 0.5 * (v ** 2)
        B = K - P

        if B > 5000:
            state = "stable"
        elif B > 1000:
            state = "turbulent"
        elif B > 0:
            state = "choking"
        else:
            state = "starving"

        return {
            "velocity": v,
            "pressure": P,
            "kinetic": K,
            "bernoulli": B,
            "state": state,
        }

    def compute_global(self, all_flows, queue_depth):
        total_v = 0.0
        total_P = 0.0
        total_K = 0.0

        for f in all_flows.values():
            v = f["packets_c2s"] + f["packets_s2c"]
            avg_rtt = f.get("avg_rtt", 0.0)
            P = (1.0 / (v + 1e-6)) + 0.5 * avg_rtt + 0.1 * queue_depth
            K = 0.5 * (v ** 2)

            total_v += v
            total_P += P
            total_K += K

        self.global_velocity = total_v
        self.global_pressure = total_P
        self.global_kinetic = total_K
        self.global_bernoulli = total_K - total_P

        if self.global_bernoulli > 20000:
            self.global_state = "stable"
        elif self.global_bernoulli > 5000:
            self.global_state = "turbulent"
        elif self.global_bernoulli > 0:
            self.global_state = "choking"
        else:
            self.global_state = "starving"


bernoulli_engine = BernoulliDataEngine()

# ---------------------------
# Cross-platform routing manipulation organ
# ---------------------------

class RoutingManipulator:
    """
    Cross-platform routing organ.
    - Provides real routing manipulation when configured.
    - Defaults to logging-only until interfaces/policies are explicitly set.
    """

    def __init__(self, os_name: str):
        self.os_name = os_name
        self.enabled = False
        self.config = {
            "windows": {
                "game_if": None,
                "bulk_if": None,
            },
            "linux": {
                "game_table": None,
                "bulk_table": None,
                "fwmark_game": None,
                "fwmark_bulk": None,
            },
            "darwin": {
                "pf_anchor": None,
                "game_if": None,
                "bulk_if": None,
            },
        }

    # --- Configuration helpers ---

    def configure_windows(self, game_if: str, bulk_if: str):
        self.config["windows"]["game_if"] = game_if
        self.config["windows"]["bulk_if"] = bulk_if
        self.enabled = True
        engine.log(f"[RoutingManipulator] Windows configured: game_if={game_if}, bulk_if={bulk_if}")

    def configure_linux(self, game_table: int, bulk_table: int, fwmark_game: int, fwmark_bulk: int):
        self.config["linux"]["game_table"] = game_table
        self.config["linux"]["bulk_table"] = bulk_table
        self.config["linux"]["fwmark_game"] = fwmark_game
        self.config["linux"]["fwmark_bulk"] = fwmark_bulk
        self.enabled = True
        engine.log(
            f"[RoutingManipulator] Linux configured: game_table={game_table}, "
            f"bulk_table={bulk_table}, fwmark_game={fwmark_game}, fwmark_bulk={fwmark_bulk}"
        )

    def configure_darwin(self, pf_anchor: str, game_if: str, bulk_if: str):
        self.config["darwin"]["pf_anchor"] = pf_anchor
        self.config["darwin"]["game_if"] = game_if
        self.config["darwin"]["bulk_if"] = bulk_if
        self.enabled = True
        engine.log(
            f"[RoutingManipulator] macOS configured: anchor={pf_anchor}, "
            f"game_if={game_if}, bulk_if={bulk_if}"
        )

    # --- Policy application ---

    def apply_flow_policy(self, flow, bernoulli, tag: str):
        """
        Called per-flow decision point.
        Uses tag + Bernoulli metrics to prefer low-pressure, stable paths.
        Real manipulation only happens if enabled and configured.
        """
        if not self.enabled:
            return

        if self.os_name == "windows":
            self._apply_windows(flow, bernoulli, tag)
        elif self.os_name == "linux":
            self._apply_linux(flow, bernoulli, tag)
        elif self.os_name == "darwin":
            self._apply_darwin(flow, bernoulli, tag)

    def _apply_windows(self, flow, bernoulli, tag: str):
        cfg = self.config["windows"]
        game_if = cfg["game_if"]
        bulk_if = cfg["bulk_if"]
        if not game_if or not bulk_if:
            return

        iface = game_if if tag.startswith("game") else bulk_if
        try:
            cmd = ["netsh", "interface", "ipv4", "set", "interface", iface,
                   "metric=10" if tag.startswith("game") else "metric=50"]
            engine.log(f"[RoutingManipulator] Windows route preference: {cmd}")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            engine.log(f"[RoutingManipulator] Windows routing error: {e}")

    def _apply_linux(self, flow, bernoulli, tag: str):
        cfg = self.config["linux"]
        game_table = cfg["game_table"]
        bulk_table = cfg["bulk_table"]
        fwmark_game = cfg["fwmark_game"]
        fwmark_bulk = cfg["fwmark_bulk"]
        if None in (game_table, bulk_table, fwmark_game, fwmark_bulk):
            return

        mark = fwmark_game if tag.startswith("game") else fwmark_bulk
        table = game_table if tag.startswith("game") else bulk_table

        try:
            cmd_rule = ["ip", "rule", "add", "fwmark", str(mark), "lookup", str(table)]
            engine.log(f"[RoutingManipulator] Linux policy rule: {cmd_rule}")
            subprocess.run(cmd_rule, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            engine.log(f"[RoutingManipulator] Linux routing error: {e}")

    def _apply_darwin(self, flow, bernoulli, tag: str):
        cfg = self.config["darwin"]
        anchor = cfg["pf_anchor"]
        game_if = cfg["game_if"]
        bulk_if = cfg["bulk_if"]
        if not anchor or not game_if or not bulk_if:
            return

        iface = game_if if tag.startswith("game") else bulk_if
        try:
            rule = f"pass out on {iface} all"
            cmd = ["pfctl", "-a", anchor, "-f", "-"]
            engine.log(f"[RoutingManipulator] macOS PF rule via anchor={anchor}, iface={iface}")
            subprocess.run(cmd, input=rule.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            engine.log(f"[RoutingManipulator] macOS routing error: {e}")


routing_manipulator = RoutingManipulator(platform.system().lower())

# ---------------------------
# Swarm Sync Organ
# ---------------------------

class SwarmSyncOrgan:
    """
    Very lightweight swarm presence + state broadcaster.
    - UDP broadcast on a fixed port.
    - Shares node_id, OS, SAFE_MODE, global Bernoulli state.
    - Receives other nodes' beacons and keeps a small view.
    """

    def __init__(self, engine: DataFrequencyEngine, port: int = 49337):
        self.engine = engine
        self.port = port
        self.node_id = f"{socket.gethostname()}-{os.getpid()}"
        self.os_name = platform.system().lower()
        self.running = False
        self.thread_tx = None
        self.thread_rx = None
        self.swarm_view_lock = threading.Lock()
        self.swarm_view = {}  # node_id -> last_state

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread_tx = threading.Thread(target=self._tx_loop, daemon=True)
        self.thread_rx = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread_tx.start()
        self.thread_rx.start()
        self.engine.log("[SwarmSync] Started swarm presence broadcast/receive.")

    def stop(self):
        self.running = False
        self.engine.log("[SwarmSync] Stopped.")

    def _tx_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while self.running:
            try:
                payload = {
                    "node_id": self.node_id,
                    "os": self.os_name,
                    "safe_mode": SAFE_MODE,
                    "admin": IS_ADMIN,
                    "global_state": bernoulli_engine.global_state,
                    "global_bernoulli": bernoulli_engine.global_bernoulli,
                    "ts": time.time(),
                }
                data = json.dumps(payload).encode("utf-8")
                sock.sendto(data, ("255.255.255.255", self.port))
            except Exception as e:
                self.engine.log(f"[SwarmSync] TX error: {e}")
            time.sleep(5.0)

    def _rx_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.port))
        except Exception as e:
            self.engine.log(f"[SwarmSync] RX bind error: {e}")
            return

        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                node_id = payload.get("node_id")
                if not node_id or node_id == self.node_id:
                    continue
                with self.swarm_view_lock:
                    self.swarm_view[node_id] = {
                        "os": payload.get("os"),
                        "safe_mode": payload.get("safe_mode"),
                        "admin": payload.get("admin"),
                        "global_state": payload.get("global_state"),
                        "global_bernoulli": payload.get("global_bernoulli"),
                        "ts": payload.get("ts"),
                        "addr": addr[0],
                    }
            except Exception as e:
                self.engine.log(f"[SwarmSync] RX error: {e}")

    def get_swarm_snapshot(self):
        with self.swarm_view_lock:
            return dict(self.swarm_view)


swarm_sync = SwarmSyncOrgan(engine)

# ---------------------------
# Routing Brain + Threat Detection + Multi-NIC scaffold
# ---------------------------

class RoutingBrain:
    def __init__(self):
        self.targets = [
            {"name": "LowLatency", "host": "0.0.0.0", "weight": 1.0, "role": "game"},
            {"name": "BulkPath", "host": "0.0.0.0", "weight": 1.0, "role": "bulk"},
        ]
        self.ui_intent = "unknown"
        self.ui_app_hint = "unknown"
        self.ui_lock = threading.Lock()

        self.threat_lock = threading.Lock()
        self.threat_events = deque(maxlen=200)
        self.flow_anomalies = defaultdict(int)

    def choose_target(self, flow):
        tag = flow.get("tag", "unknown")
        bern = flow.get("bernoulli", {})
        b_score = bern.get("bernoulli", 0.0)
        pressure = bern.get("pressure", 0.0)

        if tag.startswith("game"):
            best = None
            best_score = float("-inf")
            for t in self.targets:
                if t["role"] == "game":
                    score = b_score - pressure
                    if score > best_score:
                        best_score = score
                        best = t
            if best is not None:
                if routing_manipulator:
                    routing_manipulator.apply_flow_policy(flow, bern, tag)
                return best

        if tag.startswith("bulk"):
            for t in self.targets:
                if t["role"] == "bulk":
                    if routing_manipulator:
                        routing_manipulator.apply_flow_policy(flow, bern, tag)
                    return t

        if routing_manipulator:
            routing_manipulator.apply_flow_policy(flow, bern, tag)
        return self.targets[0]

    def set_ui_intent(self, intent: str, app_hint: str = "unknown"):
        with self.ui_lock:
            self.ui_intent = intent
            self.ui_app_hint = app_hint

    def get_ui_intent(self):
        with self.ui_lock:
            return self.ui_intent, self.ui_app_hint

    def report_threat(self, flow_key, reason: str):
        ts = time.time()
        with self.threat_lock:
            self.threat_events.append((ts, flow_key, reason))
            self.flow_anomalies[flow_key] += 1
        engine.log(f"[THREAT] {flow_key} -> {reason}")

    def get_threat_snapshot(self):
        with self.threat_lock:
            return list(self.threat_events), dict(self.flow_anomalies)


routing_brain = RoutingBrain()

# ---------------------------
# ML-style classifier scaffold
# ---------------------------

class FlowClassifier:
    def __init__(self):
        pass

    def classify(self, features: dict, ui_intent: str, ui_app_hint: str, proc_name: str):
        avg_size = features.get("avg_size", 0.0)
        rate = features.get("rate", 0.0)
        total_bytes = features.get("bytes", 0)
        rtt = features.get("rtt", 0.0)
        port = features.get("port", 0)
        bern = features.get("bernoulli", 0.0)
        pressure = features.get("pressure", 0.0)
        velocity = features.get("velocity", 0.0)

        proc_lower = (proc_name or "").lower()
        ui_intent = ui_intent or "unknown"
        ui_app_hint = ui_app_hint or "unknown"

        if match_game_signature(port):
            return "game:signature"

        if any(h in proc_lower for h in GAME_PROCESS_HINTS):
            return "game:process"

        if ui_intent == "game" or ui_app_hint == "game":
            if rate > 5 and avg_size < 800 and bern > 0 and pressure < 5:
                return "game:ui"

        if ui_intent == "browser":
            return "browser"

        if ui_intent == "bulk":
            return "bulk"

        if rate > 40 and avg_size < 200 and bern > 1000:
            return "game:pattern"

        if total_bytes > 10 * 1024 * 1024 and rtt > 0.1:
            return "bulk"

        if velocity < 2 and pressure > 10:
            return "non-game"

        return "non-game"


flow_classifier = FlowClassifier()

# ---------------------------
# Flow classification helpers
# ---------------------------

def classify_flow(flow, port_hint=None):
    avg_size = flow["bytes_c2s"] / max(flow["packets_c2s"], 1)
    rate = flow["packets_c2s"]
    rtt = 0.0
    if flow["rtt_samples"]:
        rtt = sum(flow["rtt_samples"]) / len(flow["rtt_samples"])
    total_bytes = flow["bytes_c2s"] + flow["bytes_s2c"]
    ui_intent, ui_app_hint = routing_brain.get_ui_intent()
    proc_name = flow.get("process_name", "")

    features = {
        "avg_size": avg_size,
        "rate": rate,
        "bytes": total_bytes,
        "rtt": rtt,
        "port": port_hint or 0,
    }

    bern = flow.get("bernoulli", None)
    if bern:
        features["bernoulli"] = bern.get("bernoulli", 0.0)
        features["pressure"] = bern.get("pressure", 0.0)
        features["velocity"] = bern.get("velocity", 0.0)

    label = flow_classifier.classify(features, ui_intent, ui_app_hint, proc_name)

    if label.startswith("game"):
        flow["tag"] = "game"
    elif label == "bulk":
        flow["tag"] = "bulk"
    elif label == "browser":
        flow["tag"] = "browser"
    else:
        flow["tag"] = "non-game"


# ---------------------------
# Per-process mapping helper
# ---------------------------

def map_flow_to_process(local_ip, local_port, remote_ip, remote_port, proto):
    if psutil is None:
        return ""
    try:
        conns = psutil.net_connections(kind="inet")
        for c in conns:
            if c.laddr and c.raddr:
                if (
                    c.laddr.ip == local_ip
                    and c.laddr.port == local_port
                    and c.raddr.ip == remote_ip
                    and c.raddr.port == remote_port
                ):
                    try:
                        p = psutil.Process(c.pid)
                        return p.name()
                    except Exception:
                        return ""
    except Exception:
        return ""
    return ""


# ---------------------------
# NIC discovery + auto-routing configuration
# ---------------------------

def discover_nics():
    """
    Returns a list of NICs with basic metadata.
    Uses netifaces if available, otherwise falls back to psutil.
    """
    nics = []
    if netifaces is not None:
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            ipv4 = addrs.get(netifaces.AF_INET, [])
            for a in ipv4:
                ip = a.get("addr")
                if ip and not ip.startswith("127."):
                    nics.append({"name": iface, "ip": ip})
    elif psutil is not None:
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    nics.append({"name": name, "ip": a.address})
    return nics


def auto_configure_routing_manipulator():
    """
    Simple heuristic:
    - If multiple NICs exist, pick:
        - game_if = NIC with lowest name sort
        - bulk_if = NIC with highest name sort
    - For Linux, use fixed tables/fwmarks (requires external ip rule/ip route setup).
    - For macOS, use a default anchor name.
    """
    os_name = platform.system().lower()
    nics = discover_nics()
    if not nics:
        engine.log("[AutoRouting] No NICs discovered; routing manipulator stays disabled.")
        return

    if os_name == "windows":
        if len(nics) >= 2:
            sorted_nics = sorted(nics, key=lambda x: x["name"])
            game_if = sorted_nics[0]["name"]
            bulk_if = sorted_nics[-1]["name"]
            routing_manipulator.configure_windows(game_if, bulk_if)
            engine.log(f"[AutoRouting] Windows auto-config: game_if={game_if}, bulk_if={bulk_if}")
        else:
            engine.log("[AutoRouting] Only one NIC found; Windows routing manipulator not configured.")
    elif os_name == "linux":
        # These are example values; operator must ensure tables exist.
        game_table = 100
        bulk_table = 200
        fwmark_game = 10
        fwmark_bulk = 20
        routing_manipulator.configure_linux(game_table, bulk_table, fwmark_game, fwmark_bulk)
        engine.log(
            "[AutoRouting] Linux auto-config: "
            f"game_table={game_table}, bulk_table={bulk_table}, "
            f"fwmark_game={fwmark_game}, fwmark_bulk={fwmark_bulk}"
        )
    elif os_name == "darwin":
        if len(nics) >= 2:
            sorted_nics = sorted(nics, key=lambda x: x["name"])
            game_if = sorted_nics[0]["name"]
            bulk_if = sorted_nics[-1]["name"]
            anchor = "borg/cockpit"
            routing_manipulator.configure_darwin(anchor, game_if, bulk_if)
            engine.log(
                f"[AutoRouting] macOS auto-config: anchor={anchor}, "
                f"game_if={game_if}, bulk_if={bulk_if}"
            )
        else:
            engine.log("[AutoRouting] Only one NIC found; macOS routing manipulator not configured.")
    else:
        engine.log("[AutoRouting] Unsupported OS for auto-routing configuration.")


# ---------------------------
# FlowBackend interface
# ---------------------------

class FlowBackend:
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def tick(self, now_ts: float):
        raise NotImplementedError

    def get_flows(self):
        raise NotImplementedError

    def set_flow_bypass(self, flow_id, bypass: bool):
        pass


# ---------------------------
# Windows Backend (WinDivert / pydivert) - hardened against WinError 87
# ---------------------------

class WindowsBackend(FlowBackend):
    def __init__(self, engine: DataFrequencyEngine, routing_brain: RoutingBrain):
        self.engine = engine
        self.routing_brain = routing_brain
        self.running = False
        self.thread = None
        self.flow_stats = {}

        if SAFE_MODE:
            self.pydivert = None
        else:
            self.pydivert = ensure_module("pydivert", "pydivert", required=False)

    def _ensure_flow(self, key, src, dst, proto):
        if key not in self.flow_stats:
            self.flow_stats[key] = {
                "bytes_c2s": 0,
                "bytes_s2c": 0,
                "packets_c2s": 0,
                "packets_s2c": 0,
                "tag": "unknown",
                "bypass": False,
                "rtt_samples": [],
                "last_send_ts": None,
                "last_recv_ts": None,
                "process_name": "",
                "proto": proto,
                "src": src,
                "dst": dst,
                "avg_rtt": 0.0,
                "bernoulli": None,
            }
        return self.flow_stats[key]

    def start(self):
        if SAFE_MODE or not IS_ADMIN:
            self.engine.log("[WindowsBackend] SAFE MODE or non-admin: backend not started.")
            return
        if self.pydivert is None:
            self.engine.log("[WindowsBackend] pydivert not available; backend disabled.")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.engine.log("[WindowsBackend] Started WinDivert capture on all TCP/UDP.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.engine.log("[WindowsBackend] Stopped.")

    def _capture_loop(self):
        from pydivert import WinDivert
        global SAFE_MODE

        flt = "true"

        try:
            with WinDivert(flt) as w:
                for packet in w:
                    if not self.running:
                        break

                    ts = time.time()
                    self.engine.event_queue.put(ts)
                    self.engine.produced_count += 1
                    self.engine._record_input_event(ts)

                    proto = packet.protocol
                    src = (packet.src_addr, packet.src_port)
                    dst = (packet.dst_addr, packet.dst_port)

                    if packet.is_outbound:
                        key = ("out", proto, src, dst)
                    else:
                        key = ("in", proto, dst, src)

                    flow = self._ensure_flow(key, src, dst, proto)

                    length = len(packet.raw)
                    if packet.is_outbound:
                        flow["bytes_c2s"] += length
                        flow["packets_c2s"] += 1
                        flow["last_send_ts"] = ts
                        port_hint = src[1]
                    else:
                        flow["bytes_s2c"] += length
                        flow["packets_s2c"] += 1
                        flow["last_recv_ts"] = ts
                        if flow["last_send_ts"] is not None:
                            rtt = ts - flow["last_send_ts"]
                            flow["rtt_samples"].append(rtt)
                            if len(flow["rtt_samples"]) > 50:
                                flow["rtt_samples"].pop(0)
                        port_hint = dst[1]

                    if not flow["process_name"]:
                        try:
                            flow["process_name"] = map_flow_to_process(
                                src[0], src[1], dst[0], dst[1], proto
                            )
                        except Exception:
                            pass

                    flow["bernoulli"] = bernoulli_engine.compute_flow_metrics(
                        flow, self.engine.event_queue.qsize()
                    )

                    classify_flow(flow, port_hint=port_hint)
                    self._threat_check(flow, key)

                    try:
                        w.send(packet)
                    except Exception as e:
                        self.engine.log(f"[WindowsBackend] reinject error: {e}")

        except OSError as e:
            winerr = getattr(e, "winerror", None)
            if winerr == 87:
                self.engine.log("[WindowsBackend] WinDivert WinError 87 (parameter incorrect).")
                self.engine.log("[WindowsBackend] Likely driver/DLL mismatch or OS/AV blocking.")
            else:
                self.engine.log(f"[WindowsBackend] WinDivert OSError: {e}")
            SAFE_MODE = True
            self.engine.log("[WindowsBackend] Entering SAFE MODE; backend disabled.")
        except Exception as e:
            self.engine.log(f"[WindowsBackend] WinDivert error: {e}")
            SAFE_MODE = True
            self.engine.log("[WindowsBackend] Entering SAFE MODE; backend disabled.")

    def _threat_check(self, flow, key):
        total_bytes = flow["bytes_c2s"] + flow["bytes_s2c"]
        if total_bytes > 100 * 1024 * 1024:
            routing_brain.report_threat(key, "High volume flow")

        if flow["rtt_samples"]:
            avg_rtt = sum(flow["rtt_samples"]) / len(flow["rtt_samples"])
            if avg_rtt > 0.5:
                routing_brain.report_threat(key, "High RTT flow")

        bern = flow.get("bernoulli", {})
        if bern:
            if bern.get("state") in ("choking", "starving"):
                routing_brain.report_threat(key, f"Bernoulli instability: {bern.get('state')}")

    def tick(self, now_ts: float):
        bernoulli_engine.compute_global(self.flow_stats, engine.event_queue.qsize())

    def get_flows(self):
        return self.flow_stats

    def set_flow_bypass(self, flow_id, bypass: bool):
        if flow_id in self.flow_stats:
            self.flow_stats[flow_id]["bypass"] = bypass


# ---------------------------
# Linux Backend (NFQUEUE)
# ---------------------------

class LinuxBackend(FlowBackend):
    def __init__(self, engine: DataFrequencyEngine, routing_brain: RoutingBrain):
        self.engine = engine
        self.routing_brain = routing_brain
        self.running = False
        self.thread = None
        self.flow_stats = {}
        self.packet_queue = queue.Queue()

        self.nfq_mod = ensure_module("netfilterqueue", "netfilterqueue", required=False)
        self.scapy_mod = ensure_module("scapy.all", "scapy", required=False)

    def _ensure_flow(self, key, src, dst, proto):
        if key not in self.flow_stats:
            self.flow_stats[key] = {
                "bytes_c2s": 0,
                "bytes_s2c": 0,
                "packets_c2s": 0,
                "packets_s2c": 0,
                "tag": "unknown",
                "bypass": False,
                "rtt_samples": [],
                "last_send_ts": None,
                "last_recv_ts": None,
                "process_name": "",
                "proto": proto,
                "src": src,
                "dst": dst,
                "avg_rtt": 0.0,
                "bernoulli": None,
            }
        return self.flow_stats[key]

    def start(self):
        if SAFE_MODE:
            self.engine.log("[LinuxBackend] SAFE MODE: backend not started (no root).")
            return
        if self.nfq_mod is None or self.scapy_mod is None:
            self.engine.log("[LinuxBackend] netfilterqueue/scapy not available; backend disabled.")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._nfqueue_loop, daemon=True)
        self.thread.start()
        self.engine.log("[LinuxBackend] Started NFQUEUE capture (configure iptables externally).")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.engine.log("[LinuxBackend] Stopped.")

    def _nfqueue_loop(self):
        from netfilterqueue import NetfilterQueue
        from scapy.all import IP, TCP, UDP

        def cb(pkt):
            if not self.running:
                pkt.accept()
                return

            ts = time.time()
            self.engine.event_queue.put(ts)
            self.engine.produced_count += 1
            self.engine._record_input_event(ts)

            try:
                sc = IP(pkt.get_payload())
            except Exception:
                pkt.accept()
                return

            proto = sc.proto
            src_port = 0
            dst_port = 0
            if TCP in sc:
                src_port = sc[TCP].sport
                dst_port = sc[TCP].dport
            elif UDP in sc:
                src_port = sc[UDP].sport
                dst_port = sc[UDP].dport

            src = (sc.src, src_port)
            dst = (sc.dst, dst_port)

            key = ("flow", proto, src, dst)
            flow = self._ensure_flow(key, src, dst, proto)

            length = len(sc)
            flow["bytes_c2s"] += length
            flow["packets_c2s"] += 1
            flow["last_send_ts"] = ts

            if not flow["process_name"]:
                try:
                    flow["process_name"] = map_flow_to_process(
                        src[0], src[1], dst[0], dst[1], proto
                    )
                except Exception:
                    pass

            flow["bernoulli"] = bernoulli_engine.compute_flow_metrics(
                flow, self.engine.event_queue.qsize()
            )

            classify_flow(flow, port_hint=src_port)
            self._threat_check(flow, key)

            self.packet_queue.put((pkt, flow))

        nfq = NetfilterQueue()
        try:
            nfq.bind(0, cb)
            nfq.run()
        except Exception as e:
            self.engine.log(f"[LinuxBackend] NFQUEUE error: {e}")
        finally:
            nfq.unbind()

    def _threat_check(self, flow, key):
        total_bytes = flow["bytes_c2s"] + flow["bytes_s2c"]
        if total_bytes > 100 * 1024 * 1024:
            routing_brain.report_threat(key, "High volume flow")

        bern = flow.get("bernoulli", {})
        if bern:
            if bern.get("state") in ("choking", "starving"):
                routing_brain.report_threat(key, f"Bernoulli instability: {bern.get('state')}")

    def tick(self, now_ts: float):
        drained = 0
        max_per_tick = 200
        while drained < max_per_tick:
            try:
                pkt, flow = self.packet_queue.get_nowait()
            except queue.Empty:
                break
            try:
                pkt.accept()
            except Exception as e:
                self.engine.log(f"[LinuxBackend] accept error: {e}")
            drained += 1

        bernoulli_engine.compute_global(self.flow_stats, engine.event_queue.qsize())

    def get_flows(self):
        return self.flow_stats

    def set_flow_bypass(self, flow_id, bypass: bool):
        if flow_id in self.flow_stats:
            self.flow_stats[flow_id]["bypass"] = bypass


# ---------------------------
# macOS Backend (Scapy sniff + PF divert scaffold)
# ---------------------------

class MacOSBackend(FlowBackend):
    def __init__(self, engine: DataFrequencyEngine, routing_brain: RoutingBrain):
        self.engine = engine
        self.routing_brain = routing_brain
        self.running = False
        self.thread = None
        self.flow_stats = {}

        self.scapy_mod = ensure_module("scapy.all", "scapy", required=False)

    def _ensure_flow(self, key, src, dst, proto):
        if key not in self.flow_stats:
            self.flow_stats[key] = {
                "bytes_c2s": 0,
                "bytes_s2c": 0,
                "packets_c2s": 0,
                "packets_s2c": 0,
                "tag": "unknown",
                "bypass": False,
                "rtt_samples": [],
                "last_send_ts": None,
                "last_recv_ts": None,
                "process_name": "",
                "proto": proto,
                "src": src,
                "dst": dst,
                "avg_rtt": 0.0,
                "bernoulli": None,
            }
        return self.flow_stats[key]

    def start(self):
        if SAFE_MODE:
            self.engine.log("[MacOSBackend] SAFE MODE: backend not started (no root).")
            return
        if self.scapy_mod is None:
            self.engine.log("[MacOSBackend] scapy not available; backend disabled.")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()
        self.engine.log("[MacOSBackend] Started Scapy sniff (monitoring only).")

    def stop(self):
        self.running = False
        self.engine.log("[MacOSBackend] Stopped.")

    def _sniff_loop(self):
        from scapy.all import sniff, IP, TCP, UDP

        def cb(pkt):
            if not self.running:
                return

            ts = time.time()
            self.engine.event_queue.put(ts)
            self.engine.produced_count += 1
            self.engine._record_input_event(ts)

            if IP not in pkt:
                return

            ip = pkt[IP]
            proto = ip.proto
            src_port = 0
            dst_port = 0
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            src = (ip.src, src_port)
            dst = (ip.dst, dst_port)
            key = ("flow", proto, src, dst)
            flow = self._ensure_flow(key, src, dst, proto)

            length = len(pkt)
            flow["bytes_c2s"] += length
            flow["packets_c2s"] += 1
            flow["last_send_ts"] = ts

            if not flow["process_name"]:
                try:
                    flow["process_name"] = map_flow_to_process(
                        src[0], src[1], dst[0], dst[1], proto
                    )
                except Exception:
                    pass

            flow["bernoulli"] = bernoulli_engine.compute_flow_metrics(
                flow, self.engine.event_queue.qsize()
            )

            classify_flow(flow, port_hint=src_port)

        sniff(filter="ip", prn=cb, store=False)

    def tick(self, now_ts: float):
        bernoulli_engine.compute_global(self.flow_stats, engine.event_queue.qsize())

    def get_flows(self):
        return self.flow_stats

    def set_flow_bypass(self, flow_id, bypass: bool):
        if flow_id in self.flow_stats:
            self.flow_stats[flow_id]["bypass"] = bypass


# ---------------------------
# UIAutomation Backend (deeper)
# ---------------------------

class UIAutomationBackend:
    def __init__(self, engine: DataFrequencyEngine, routing_brain: RoutingBrain):
        self.engine = engine
        self.routing_brain = routing_brain
        self.running = False
        self.thread = None
        self.os_name = platform.system().lower()

        self.pywinauto = None
        if self.os_name == "windows":
            self.pywinauto = ensure_module("pywinauto", "pywinauto", required=False)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.engine.log("[UIAutomation] Started UI intent sensing.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.engine.log("[UIAutomation] Stopped.")

    def _loop(self):
        while self.running:
            try:
                intent, app_hint = self._detect_intent()
                if intent:
                    self.routing_brain.set_ui_intent(intent, app_hint)
            except Exception as e:
                self.engine.log(f"[UIAutomation] error: {e}")
            time.sleep(1.0)

    def _detect_intent(self):
        if self.os_name == "windows" and self.pywinauto is not None:
            try:
                from pywinauto import Desktop
                desk = Desktop(backend="uia")
                win = desk.get_active()
                title = (win.window_text() or "").lower()
                app_hint = "unknown"

                if any(k in title for k in GAME_PROCESS_HINTS):
                    return "game", "game_launcher"

                if any(k in title for k in ["chrome", "firefox", "edge", "safari", "browser"]):
                    return "browser", "browser"

                if any(k in title for k in ["download", "torrent", "qbittorrent", "utorrent"]):
                    return "bulk", "downloader"

                if any(k in title for k in ["netflix", "youtube", "twitch", "prime video"]):
                    return "bulk", "streaming"

                return "other", app_hint
            except Exception:
                return "unknown", "unknown"

        return "unknown", "unknown"


# ---------------------------
# DualPersonalityBot helpers
# ---------------------------

def adaptive_mutation(mode: str):
    engine.log(f"[DualBot] Adaptive mutation invoked: {mode}")

def generate_decoy():
    return f"decoy-{int(time.time())}"

def compliance_auditor(items):
    return f"OK ({len(items)} items)"

def reverse_mirror_encrypt(s: str) -> str:
    return s[::-1]

def camouflage(s: str, style: str) -> str:
    return f"{style}:{s}"

def random_glyph_stream() -> str:
    glyphs = "☉☼★✦✧✩✪✫✬✭✮✯"
    return "".join(random.choice(glyphs) for _ in range(32))


class DualPersonalityBot:
    def __init__(self, cb):
        self.cb = cb
        self.run = True
        self.mode = "guardian"  # "guardian" or "rogue"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []

    def switch_mode(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"🔺 Personality switched to {self.mode.upper()}")

    def guardian_behavior(self):
        adaptive_mutation("ghost sync")
        decoy = generate_decoy()
        self.cb(f"🕊️ Guardian audit: {decoy}")
        self.cb(f"🔱 Compliance: {compliance_auditor([decoy])}")

    def rogue_behavior(self):
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy), "alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"

        self.rogue_weights = [
            w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights
        ]
        self.rogue_log.append(self.rogue_weights)

        score = sum(self.rogue_weights) / len(self.rogue_weights)

        self.cb("💀⚔️ Rogue escalation initiated")
        self.cb(f"🜏 Rogue pattern: {unusual_pattern}")
        self.cb(f"📊 Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.run:
            if self.mode == "guardian":
                self.guardian_behavior()
            else:
                self.rogue_behavior()
            time.sleep(10)


# ---------------------------
# Tkinter GUI
# ---------------------------

class BorgFrequencyGUI:
    def __init__(self, root, engine: DataFrequencyEngine, backend: FlowBackend, backend_label: str):
        self.root = root
        self.engine = engine
        self.backend = backend
        self.backend_label = backend_label

        self.root.title(f"Borg Universal Cockpit ({backend_label})")
        self.root.geometry("1400x800")

        self._build_layout()
        self._start_polling()

    def _build_layout(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=5, pady=5)

        controls_frame = ttk.LabelFrame(top_frame, text="Governor Control")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        ttk.Label(controls_frame, text="Input rate (Hz):").grid(row=0, column=0, sticky="w")
        self.input_rate_var = tk.StringVar(value="10.0")
        ttk.Entry(controls_frame, textvariable=self.input_rate_var, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(controls_frame, text="Output rate (Hz):").grid(row=1, column=0, sticky="w")
        self.output_rate_var = tk.StringVar(value="50.0")
        ttk.Entry(controls_frame, textvariable=self.output_rate_var, width=10).grid(row=1, column=1, sticky="w")

        ttk.Button(controls_frame, text="Apply Rates", command=self.on_apply_rates).grid(row=2, column=0, columnspan=2, pady=4)

        self.auto_tune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Auto-tune output",
            variable=self.auto_tune_var,
            command=self.on_toggle_autotune
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(controls_frame, text="Target queue depth:").grid(row=4, column=0, sticky="w")
        self.target_queue_var = tk.StringVar(value="5")
        ttk.Entry(controls_frame, textvariable=self.target_queue_var, width=10).grid(row=4, column=1, sticky="w")

        self.autotune_status_var = tk.StringVar(value="")
        ttk.Label(controls_frame, textvariable=self.autotune_status_var, wraplength=250).grid(row=5, column=0, columnspan=2, sticky="w", pady=4)

        self.autonomous_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_frame,
            text="Autonomous Mode",
            variable=self.autonomous_var,
            command=self.on_toggle_autonomous
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)

        backend_frame = ttk.LabelFrame(top_frame, text="Backend / Privilege / Bernoulli / Swarm")
        backend_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.backend_label_var = tk.StringVar(value=f"Active backend: {self.backend_label}")
        self.privilege_var = tk.StringVar(value=self._privilege_text())
        self.global_bern_var = tk.StringVar(value="Global Bernoulli: n/a")
        self.global_state_var = tk.StringVar(value="Global Stability: unknown")
        self.swarm_var = tk.StringVar(value="Swarm nodes: 0")

        ttk.Label(backend_frame, textvariable=self.backend_label_var).grid(row=0, column=0, sticky="w")
        ttk.Label(backend_frame, textvariable=self.privilege_var).grid(row=1, column=0, sticky="w")
        ttk.Label(backend_frame, textvariable=self.global_bern_var).grid(row=2, column=0, sticky="w")
        ttk.Label(backend_frame, textvariable=self.global_state_var).grid(row=3, column=0, sticky="w")
        ttk.Label(backend_frame, textvariable=self.swarm_var).grid(row=4, column=0, sticky="w")

        graph_frame = ttk.LabelFrame(top_frame, text="Frequency Graph (events over last 10s)")
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time window (s)")
        self.ax.set_ylabel("Events")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 2)
        self.ax.grid(True)

        (self.line_in,) = self.ax.plot([], [], label="Input events", color="tab:blue")
        (self.line_out,) = self.ax.plot([], [], label="Output events", color="tab:orange")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        telemetry_frame = ttk.LabelFrame(bottom_frame, text="Telemetry")
        telemetry_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.produced_var = tk.StringVar(value="Produced: 0")
        self.emitted_var = tk.StringVar(value="Emitted: 0")
        self.interp_var = tk.StringVar(value="Interpolated: 0")
        self.queue_var = tk.StringVar(value="Queue size: 0")
        self.status_var = tk.StringVar(value="")
        self.intent_var = tk.StringVar(value="UI intent: unknown")
        self.threats_var = tk.StringVar(value="Threats: 0")

        ttk.Label(telemetry_frame, textvariable=self.produced_var).pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.emitted_var).pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.interp_var).pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.queue_var).pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.intent_var).pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.threats_var).pack(anchor="w")
        ttk.Separator(telemetry_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(telemetry_frame, text="Status:").pack(anchor="w")
        ttk.Label(telemetry_frame, textvariable=self.status_var, wraplength=250).pack(anchor="w")

        log_frame = ttk.LabelFrame(bottom_frame, text="Borg Log Console")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, wrap="none", height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state="disabled")

        flows_frame = ttk.LabelFrame(bottom_frame, text="Flows")
        flows_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.flows_list = tk.Text(flows_frame, wrap="none", height=15)
        self.flows_list.pack(fill=tk.BOTH, expand=True)
        self.flows_list.configure(state="disabled")

        if SAFE_MODE:
            self._show_safe_mode_popup()

    def _privilege_text(self):
        if IS_ADMIN and not SAFE_MODE:
            return "Privilege: ADMIN (full interception)"
        if SAFE_MODE and not IS_ADMIN:
            return "Privilege: SAFE MODE (no backend; cockpit only)"
        if SAFE_MODE and IS_ADMIN:
            return "Privilege: ADMIN + SAFE MODE (manual override)"
        return "Privilege: UNKNOWN"

    def _show_safe_mode_popup(self):
        try:
            from tkinter import messagebox
            messagebox.showwarning(
                "Borg Safe Mode",
                "Running in SAFE MODE.\n\n"
                "- Backends are not started.\n"
                "- No packet interception.\n\n"
                "Run as Administrator (Windows) or with sudo (Linux/macOS) for full functionality."
            )
        except Exception:
            pass

    def on_apply_rates(self):
        try:
            in_rate = float(self.input_rate_var.get())
            out_rate = float(self.output_rate_var.get())
        except ValueError:
            self._append_log_line("[GUI] Invalid rate values.")
            return
        engine.set_rates(in_rate, out_rate)

    def on_toggle_autotune(self):
        engine.set_auto_tune(self.auto_tune_var.get())

    def on_toggle_autonomous(self):
        engine.autonomous_mode = self.autonomous_var.get()

    def _start_polling(self):
        self._poll_engine()
        self._poll_graph()
        self.root.after(500, self._start_polling)

    def _poll_engine(self):
        snap = engine.get_snapshot()

        self.input_rate_var.set(f"{snap['input_rate_hz']:.2f}")
        self.output_rate_var.set(f"{snap['output_rate_hz']:.2f}")

        self.produced_var.set(f"Produced: {snap['produced']}")
        self.emitted_var.set(f"Emitted: {snap['emitted']}")
        self.interp_var.set(f"Interpolated: {snap['interpolated']}")
        self.queue_var.set(f"Queue size: {snap['queue_size']}")

        self.auto_tune_var.set(snap["auto_tune_enabled"])
        self.target_queue_var.set(str(snap["target_queue_depth"]))
        self.autonomous_var.set(snap["autonomous_mode"])

        self.autotune_status_var.set(
            f"Auto-tune: {'ON' if snap['auto_tune_enabled'] else 'OFF'}, "
            f"target q={snap['target_queue_depth']}, "
            f"out={snap['output_rate_hz']:.2f} Hz, "
            f"Autonomous: {'ON' if snap['autonomous_mode'] else 'OFF'}"
        )

        if snap["log_lines"]:
            self.status_var.set(snap["log_lines"][-1])

        ui_intent, ui_app_hint = routing_brain.get_ui_intent()
        self.intent_var.set(f"UI intent: {ui_intent} ({ui_app_hint})")

        threats, anomalies = routing_brain.get_threat_snapshot()
        self.threats_var.set(f"Threats: {len(threats)}")

        self.privilege_var.set(self._privilege_text())

        self.global_bern_var.set(
            f"Global Bernoulli: {bernoulli_engine.global_bernoulli:.1f} "
            f"(P={bernoulli_engine.global_pressure:.2f}, V={bernoulli_engine.global_velocity:.1f})"
        )
        self.global_state_var.set(f"Global Stability: {bernoulli_engine.global_state}")

        swarm_snapshot = swarm_sync.get_swarm_snapshot()
        self.swarm_var.set(f"Swarm nodes: {len(swarm_snapshot)}")

        self._set_log_text("\n".join(snap["log_lines"]))
        self._update_flows_panel()

    def _poll_graph(self):
        xs_in, ys_in, xs_out, ys_out = engine.get_graph_series()

        if xs_in or xs_out:
            max_x = max(xs_in + xs_out)
        else:
            max_x = 10.0

        self.ax.set_xlim(0, max(10.0, max_x))
        self.ax.set_ylim(0, 2)

        self.line_in.set_data(xs_in, ys_in)
        self.line_out.set_data(xs_out, ys_out)

        self.canvas.draw_idle()

    def _update_flows_panel(self):
        flows = self.backend.get_flows()
        text = ""
        for key, flow in flows.items():
            rtt = 0.0
            if flow["rtt_samples"]:
                rtt = sum(flow["rtt_samples"]) / len(flow["rtt_samples"])
            bern = flow.get("bernoulli", {})
            b_score = bern.get("bernoulli", 0.0)
            b_state = bern.get("state", "unknown")
            pressure = bern.get("pressure", 0.0)
            velocity = bern.get("velocity", 0.0)

            text += (
                f"{key} | tag={flow['tag']} | proc={flow.get('process_name','')} | "
                f"bypass={flow['bypass']} | "
                f"c2s={flow['packets_c2s']}pkts/{flow['bytes_c2s']}B | "
                f"s2c={flow['packets_s2c']}pkts/{flow['bytes_s2c']}B | "
                f"rtt~={rtt*1000:.1f}ms | "
                f"Bern={b_score:.1f} ({b_state}) P={pressure:.3f} V={velocity:.1f}\n"
            )

        self.flows_list.configure(state="normal")
        self.flows_list.delete("1.0", tk.END)
        self.flows_list.insert(tk.END, text)
        self.flows_list.configure(state="disabled")
        self.flows_list.see(tk.END)

    def _set_log_text(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        if text:
            self.log_text.insert(tk.END, text)
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def _append_log_line(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)


if __name__ == "__main__":
    os_name = platform.system().lower()
    backend_label = ""
    backend: FlowBackend

    if os_name == "windows":
        backend = WindowsBackend(engine, routing_brain)
        backend_label = "Windows / WinDivert"
    elif os_name == "linux":
        backend = LinuxBackend(engine, routing_brain)
        backend_label = "Linux / NFQUEUE"
    elif os_name == "darwin":
        backend = MacOSBackend(engine, routing_brain)
        backend_label = "macOS / Scapy+PF scaffold"
    else:
        raise RuntimeError("Unsupported OS for universal interceptor")

    engine.autonomous_mode = True

    # Auto-configure routing manipulator based on NIC discovery
    auto_configure_routing_manipulator()

    if not SAFE_MODE:
        backend.start()
    else:
        engine.log("[CORE] SAFE MODE: backends not started; cockpit running in monitor-only mode.")

    engine.register_consumer(backend.tick)
    engine.start()

    ui_backend = UIAutomationBackend(engine, routing_brain)
    ui_backend.start()

    swarm_sync.start()

    dual_bot = DualPersonalityBot(engine.log)
    dual_bot.start()

    root = tk.Tk()
    app = BorgFrequencyGUI(root, engine, backend, backend_label)
    root.mainloop()

