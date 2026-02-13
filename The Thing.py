"""
TRI-STANCE ORGANISM — SINGLE FILE EDITION
A + B + C: Swarm, API, Reinforcement fully evolved in one beast.

Requirements (pip):
    fastapi
    uvicorn
    psutil
    requests
    (optional) wmi

Run:
    python tri_stance_organism.py
"""

import time
import statistics
from collections import deque, defaultdict
import threading
import random
import sys
import json
import os
import math

# GUI
import tkinter as tk
from tkinter import ttk

# Telemetry
import psutil
try:
    import wmi
    HAS_WMI = True
except ImportError:
    HAS_WMI = False

# FastAPI / API
from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import requests

# =========================================================
# GLOBAL CONFIG / CONSTANTS
# =========================================================

STATE_FILE = "tri_stance_state.json"
API_TOKEN = "change_me_token"  # change this
GOSSIP_INTERVAL = 2.0          # seconds
LAN_BEACON_PORT = 39877        # placeholder (not fully implemented UDP here)


# =========================================================
# PERSISTENCE HELPERS
# =========================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"log": [], "reinforcement": {}, "engine_cfg": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"log": [], "reinforcement": {}, "engine_cfg": {}}


def save_state(log, reinforcement, engine_cfg):
    try:
        data = {
            "log": log,
            "reinforcement": reinforcement,
            "engine_cfg": engine_cfg
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# =========================================================
# TRI-STANCE DECISION ENGINE + REINFORCEMENT ORGAN
# =========================================================

class TriStanceEngine:
    """
    Tri‑Stance Decision Engine + Reinforcement Organ:
    - Conservative / Balanced / Beast
    - Hysteresis, cooldowns, weighted signals
    - Prediction smoothing
    - Reinforcement with per-stance memory + decay
    - Adaptive thresholds + weights (slow learning)
    - Fail‑safe + operator override + structured logs
    - Persistent log + reinforcement across restarts
    """

    def __init__(self, node_id="local", config=None, initial_state=None):
        self.node_id = node_id
        self.cfg = config or {
            "cooldown_sec": 3,
            "hysteresis": 0.05,
            "weights": {
                "memory": 1.0,
                "temp": 1.0,
                "adapter": 1.0,
                "prediction": 1.5,
                "pressure": 1.2,
                "bottleneck": 1.3,
            },
            "thresholds": {
                "memory_high": 0.90,
                "memory_low": 0.70,
                "temp_high": 80,
                "temp_low": 60,
                "pressure_high": 0.75,
                "pressure_low": 0.40,
                "risk_high": 0.65,
                "risk_low": 0.35
            },
            "reinforcement_decay": 0.98,
            "learning_rate_thresholds": 0.001,
            "learning_rate_weights": 0.001
        }

        self.stance = "Balanced"
        self.last_switch = time.time()
        self.override = None

        # per-stance reinforcement history
        self.reinforcement = {
            "Conservative": deque(maxlen=200),
            "Balanced": deque(maxlen=200),
            "Beast": deque(maxlen=200)
        }

        # short-term temporal trace (for credit assignment)
        self.trace = deque(maxlen=50)  # list of (time, stance, risk, signals_snapshot)

        self.pred_history = deque(maxlen=20)
        self.log = []
        self.last_risk = 0.0
        self.last_signals = None

        if initial_state:
            self._restore_state(initial_state)

    # ---------------- Public API ----------------

    def force_stance(self, stance):
        self.override = stance
        self._log("OVERRIDE", stance, "Operator override engaged")

    def clear_override(self):
        self.override = None
        self._log("OVERRIDE_CLEAR", self.stance, "Operator override cleared")

    def update(self, signals):
        """
        signals = {
            "memory": 0-1,
            "temp": °C,
            "adapter": 0-1,
            "pressure": 0-1,
            "prediction": 0-1,
            "bottleneck": 0-1
        }
        """
        self.last_signals = signals

        if any(v is None for v in signals.values()):
            self._switch("Conservative", "Fail‑safe: missing signals")
            return self.get_stance()

        self.pred_history.append(signals["prediction"])
        pred_smooth = statistics.mean(self.pred_history)

        risk = self._compute_risk(signals, pred_smooth)
        self.last_risk = risk

        reinforcement_factor = self._compute_reinforcement()

        # record trace for temporal credit assignment
        self._record_trace(signals, risk)

        new_stance, reason = self._decide_stance(signals, risk, reinforcement_factor)

        if self.override:
            self._log("OVERRIDE_ACTIVE", self.override, "Operator override in effect")
            return self.override

        if new_stance != self.stance and self._can_switch():
            self._switch(new_stance, reason)

        # decay reinforcement + adapt thresholds/weights slowly
        self._decay_reinforcement()
        self._adapt_parameters()

        return self.get_stance()

    def get_stance(self):
        return self.override or self.stance

    def get_log(self):
        return list(self.log)

    def add_reinforcement(self, value, stance=None):
        """
        value in [-1, 1]
        stance: optional, defaults to current stance
        """
        if stance is None:
            stance = self.get_stance()
        if stance not in self.reinforcement:
            return
        self.reinforcement[stance].append(value)
        self._log("REINFORCE", stance, f"Reinforcement {value:+.3f}")

    def get_reinforcement_snapshot(self):
        snap = {}
        for s, dq in self.reinforcement.items():
            snap[s] = list(dq)
        return snap

    def get_config_snapshot(self):
        return {
            "weights": self.cfg["weights"],
            "thresholds": self.cfg["thresholds"]
        }

    # ---------------- Internal: State Restore ----------------

    def _restore_state(self, state):
        if "log" in state and isinstance(state["log"], list):
            self.log = state["log"][-800:]
        if "reinforcement" in state and isinstance(state["reinforcement"], dict):
            for stance, values in state["reinforcement"].items():
                if stance in self.reinforcement and isinstance(values, list):
                    for v in values[-200:]:
                        try:
                            self.reinforcement[stance].append(float(v))
                        except Exception:
                            continue
        if "engine_cfg" in state and isinstance(state["engine_cfg"], dict):
            cfg = state["engine_cfg"]
            if "weights" in cfg and isinstance(cfg["weights"], dict):
                self.cfg["weights"].update(cfg["weights"])
            if "thresholds" in cfg and isinstance(cfg["thresholds"], dict):
                self.cfg["thresholds"].update(cfg["thresholds"])

    # ---------------- Internal: Core Logic ----------------

    def _compute_risk(self, s, pred):
        w = self.cfg["weights"]
        t = self.cfg["thresholds"]

        risk = (
            w["memory"] * (s["memory"] - t["memory_low"]) +
            w["temp"] * ((s["temp"] - t["temp_low"]) / 100.0) +
            w["adapter"] * (1 - s["adapter"]) +
            w["pressure"] * (s["pressure"] - t["pressure_low"]) +
            w["prediction"] * (pred - t["risk_low"]) +
            w["bottleneck"] * s["bottleneck"]
        )

        return max(0.0, min(1.0, risk))

    def _compute_reinforcement(self):
        vals = []
        for stance, dq in self.reinforcement.items():
            if dq:
                avg = statistics.mean(dq)
                weight = 1.0 if stance == "Beast" else 0.7 if stance == "Balanced" else 0.5
                vals.append(avg * weight)
        if not vals:
            return 0.0
        return statistics.mean(vals)

    def _decide_stance(self, s, risk, reinforce):
        t = self.cfg["thresholds"]
        h = self.cfg["hysteresis"]

        if (
            s["memory"] > t["memory_high"] + h or
            s["temp"] > t["temp_high"] + h or
            risk > t["risk_high"] + h or
            s["adapter"] < 0.3
        ):
            return "Conservative", "High risk or thermal/memory pressure"

        if (
            s["memory"] < t["memory_low"] - h and
            s["temp"] < t["temp_low"] - h and
            risk < t["risk_low"] - h and
            reinforce > 0.5
        ):
            return "Beast", "Low risk + reinforcement positive"

        return "Balanced", "Normal operating conditions"

    def _can_switch(self):
        return (time.time() - self.last_switch) >= self.cfg["cooldown_sec"]

    def _switch(self, stance, reason):
        old = self.stance
        self.stance = stance
        self.last_switch = time.time()
        self._log("SWITCH", stance, f"{old} → {stance}: {reason}")

    def _log(self, event, stance, msg):
        entry = {
            "time": time.time(),
            "node": self.node_id,
            "event": event,
            "stance": stance,
            "msg": msg
        }
        self.log.append(entry)
        if len(self.log) > 1500:
            self.log = self.log[-1000:]

    # ---------------- Internal: Reinforcement & Learning ----------------

    def _decay_reinforcement(self):
        decay = self.cfg.get("reinforcement_decay", 0.98)
        for stance, dq in self.reinforcement.items():
            if not dq:
                continue
            new_vals = deque(maxlen=dq.maxlen)
            for v in dq:
                new_vals.append(v * decay)
            self.reinforcement[stance] = new_vals

    def _record_trace(self, signals, risk):
        self.trace.append({
            "time": time.time(),
            "stance": self.get_stance(),
            "risk": risk,
            "signals": dict(signals)
        })

    def temporal_reinforce(self, reward):
        """
        Apply reward/punishment to recent trace (temporal credit assignment).
        reward in [-1, 1]
        """
        now = time.time()
        horizon = 5.0  # seconds
        for entry in list(self.trace):
            age = now - entry["time"]
            if age > horizon:
                continue
            decay = math.exp(-age / horizon)
            self.add_reinforcement(reward * decay, stance=entry["stance"])

    def _adapt_parameters(self):
        """
        Very gentle adaptation of thresholds and weights based on reinforcement.
        Positive reinforcement -> more aggressive (lower risk_high, etc.)
        Negative reinforcement -> more conservative.
        """
        lr_t = self.cfg.get("learning_rate_thresholds", 0.0)
        lr_w = self.cfg.get("learning_rate_weights", 0.0)
        if lr_t <= 0 and lr_w <= 0:
            return

        global_reinf = self._compute_reinforcement()
        if abs(global_reinf) < 0.05:
            return

        # thresholds
        t = self.cfg["thresholds"]
        if global_reinf > 0:
            # system doing well -> allow more Beast
            t["risk_high"] = min(0.95, t["risk_high"] + lr_t * (-1))
            t["memory_high"] = min(0.98, t["memory_high"] + lr_t * 0.5)
            t["temp_high"] = min(95, t["temp_high"] + lr_t * 20)
        else:
            # system struggling -> be more conservative
            t["risk_high"] = max(0.4, t["risk_high"] + lr_t * 1)
            t["memory_high"] = max(0.7, t["memory_high"] - lr_t * 0.5)
            t["temp_high"] = max(60, t["temp_high"] - lr_t * 20)

        # weights
        w = self.cfg["weights"]
        if global_reinf > 0:
            # reward Beast: reduce fear of memory/temp a bit, increase prediction weight
            w["memory"] = max(0.5, w["memory"] - lr_w * 0.5)
            w["temp"] = max(0.5, w["temp"] - lr_w * 0.5)
            w["prediction"] = min(2.5, w["prediction"] + lr_w * 1.0)
        else:
            # punishment: increase weight on memory/temp, reduce prediction aggressiveness
            w["memory"] = min(2.0, w["memory"] + lr_w * 0.5)
            w["temp"] = min(2.0, w["temp"] + lr_w * 0.5)
            w["prediction"] = max(0.5, w["prediction"] - lr_w * 1.0)


# =========================================================
# TELEMETRY ORGAN
# =========================================================

class TelemetryOrgan:
    """
    Telemetry organ using real Windows metrics via psutil (+ optional WMI).
    - Better bottleneck detection (CPU, disk, net)
    - Pluggable prediction source
    """

    def __init__(self, prediction_source=None):
        self.prediction_source = prediction_source

        if HAS_WMI and sys.platform.startswith("win"):
            try:
                self.wmi_obj = wmi.WMI(namespace="root\\wmi")
            except Exception:
                self.wmi_obj = None
        else:
            self.wmi_obj = None

    def read_signals(self):
        mem = psutil.virtual_memory()
        memory_ratio = mem.percent / 100.0

        cpu_temp = self._read_cpu_temp()
        adapter_health = self._read_adapter_health()
        pressure = self._compute_pressure()
        prediction = self._read_prediction()
        bottleneck = self._detect_bottleneck()

        return {
            "memory": memory_ratio,
            "temp": cpu_temp,
            "adapter": adapter_health,
            "pressure": pressure,
            "prediction": prediction,
            "bottleneck": bottleneck
        }

    def _read_cpu_temp(self):
        if self.wmi_obj:
            try:
                temps = self.wmi_obj.MSAcpi_ThermalZoneTemperature()
                if temps:
                    t = (temps[0].CurrentTemperature / 10.0) - 273.15
                    return float(t)
            except Exception:
                pass
        cpu = psutil.cpu_percent(interval=0.05)
        return 40.0 + (cpu / 100.0) * 50.0

    def _read_adapter_health(self):
        try:
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            activity = (disk.read_bytes + disk.write_bytes + net.bytes_sent + net.bytes_recv) / (1024**2)
            health = max(0.3, 1.0 - min(activity / 2000.0, 0.7))
            return health
        except Exception:
            return 0.8

    def _compute_pressure(self):
        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        try:
            disk = psutil.disk_usage("/").percent / 100.0
        except Exception:
            disk = 0.3
        return max(0.0, min(1.0, (cpu * 0.6 + disk * 0.4)))

    def _read_prediction(self):
        if self.prediction_source is None:
            return random.uniform(0.2, 0.8)
        if callable(self.prediction_source):
            try:
                return float(self.prediction_source())
            except Exception:
                return 0.5
        return float(getattr(self.prediction_source, "current_risk", 0.5))

    def _detect_bottleneck(self):
        cpu = psutil.cpu_percent(interval=0.05) / 100.0
        try:
            disk = psutil.disk_usage("/").percent / 100.0
        except Exception:
            disk = 0.3
        net = 0.0
        try:
            net_io = psutil.net_io_counters()
            net = min(1.0, (net_io.bytes_sent + net_io.bytes_recv) / (1024**3))
        except Exception:
            pass

        vals = {"cpu": cpu, "disk": disk, "net": net}
        diff = max(vals.values()) - min(vals.values())
        score = max(0.0, min(1.0, diff))
        return score


# =========================================================
# SWARM ORGAN (TRUST + HEALTH + REMOTE HOOKS)
# =========================================================

class SwarmCoordinator:
    """
    Manages multiple nodes (engine + telemetry) and computes swarm stance.
    - Trust weights per node
    - Node health tracking
    - Hooks for remote nodes (HTTP heartbeat)
    - Swarm-level reinforcement hooks
    """

    def __init__(self):
        self.nodes = {}  # node_id -> {"engine":..., "telemetry":..., "trust":..., "remote_url":..., "last_ok":...}

    def add_local_node(self, node_id, engine, telemetry, trust=1.0):
        self.nodes[node_id] = {
            "engine": engine,
            "telemetry": telemetry,
            "trust": trust,
            "remote_url": None,
            "last_ok": time.time()
        }

    def add_remote_node(self, node_id, base_url, trust=0.8):
        self.nodes[node_id] = {
            "engine": None,
            "telemetry": None,
            "trust": trust,
            "remote_url": base_url.rstrip("/"),
            "last_ok": 0
        }

    def update_all(self):
        swarm_stances = {}
        now = time.time()
        for node_id, entry in self.nodes.items():
            if entry["engine"] is not None:
                signals = entry["telemetry"].read_signals()
                stance = entry["engine"].update(signals)
                entry["last_ok"] = now
                swarm_stances[node_id] = stance
            else:
                url = entry["remote_url"] + "/stance"
                try:
                    r = requests.get(url, timeout=0.5)
                    if r.status_code == 200:
                        data = r.json()
                        stance = data.get("stance", "Balanced")
                        swarm_stances[node_id] = stance
                        entry["last_ok"] = now
                    else:
                        swarm_stances[node_id] = "Unknown"
                except Exception:
                    swarm_stances[node_id] = "Unknown"
        return swarm_stances

    def get_swarm_consensus(self):
        scores = {"Conservative": 0.0, "Balanced": 0.0, "Beast": 0.0}
        now = time.time()
        for node_id, entry in self.nodes.items():
            if now - entry["last_ok"] > 10:
                continue
            if entry["engine"] is not None:
                stance = entry["engine"].get_stance()
            else:
                continue
            trust = entry["trust"]
            if stance in scores:
                scores[stance] += trust

        if all(v == 0 for v in scores.values()):
            return "Balanced", {"Conservative": 0, "Balanced": 0, "Beast": 0}

        consensus = max(scores.items(), key=lambda x: x[1])[0]
        counts = {k: int(round(v)) for k, v in scores.items()}
        return consensus, counts

    def swarm_reinforce(self, reward):
        """
        Apply reinforcement to all local engines.
        """
        for node_id, entry in self.nodes.items():
            if entry["engine"] is not None:
                entry["engine"].temporal_reinforce(reward)


# =========================================================
# FASTAPI STATUS + CONTROL PLANE (WITH TOKEN AUTH + WS)
# =========================================================

class WebSocketHub:
    """
    Simple WS hub for broadcasting stance/metrics/log updates.
    """

    def __init__(self):
        self.connections = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def create_api(local_engine: TriStanceEngine, swarm: SwarmCoordinator, ws_hub: WebSocketHub):
    app = FastAPI()

    def check_token(x_token: str = Header(None)):
        if x_token != API_TOKEN:
            return False
        return True

    @app.get("/stance")
    def get_stance():
        return {"stance": local_engine.get_stance()}

    @app.get("/signals")
    def get_signals():
        return JSONResponse(local_engine.last_signals or {})

    @app.get("/risk")
    def get_risk():
        return {"risk": local_engine.last_risk}

    @app.get("/reinforcement")
    def get_reinforcement():
        return {"reinforcement": local_engine._compute_reinforcement()}

    @app.get("/log")
    def get_log():
        return JSONResponse(local_engine.get_log())

    @app.post("/override/{stance}")
    def set_override(stance: str, x_token: str = Header(None)):
        if not check_token(x_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        stance = stance.capitalize()
        if stance not in ("Conservative", "Balanced", "Beast"):
            return JSONResponse({"error": "invalid stance"}, status_code=400)
        local_engine.force_stance(stance)
        return {"status": "ok", "stance": local_engine.get_stance()}

    @app.post("/override/clear")
    def clear_override(x_token: str = Header(None)):
        if not check_token(x_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        local_engine.clear_override()
        return {"status": "ok", "stance": local_engine.get_stance()}

    @app.post("/reinforce/{direction}")
    def api_reinforce(direction: str, x_token: str = Header(None)):
        if not check_token(x_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if direction not in ("positive", "negative"):
            return JSONResponse({"error": "invalid direction"}, status_code=400)
        reward = 1.0 if direction == "positive" else -1.0
        swarm.swarm_reinforce(reward)
        return {"status": "ok", "reward": reward}

    @app.get("/swarm")
    def get_swarm():
        consensus, counts = swarm.get_swarm_consensus()
        node_stances = {}
        for nid, e in swarm.nodes.items():
            if e["engine"] is not None:
                node_stances[nid] = e["engine"].get_stance()
            else:
                node_stances[nid] = "remote"
        return {
            "consensus": consensus,
            "counts": counts,
            "nodes": node_stances
        }

    @app.get("/config")
    def get_config():
        return local_engine.get_config_snapshot()

    @app.post("/config")
    def update_config(cfg: dict, x_token: str = Header(None)):
        if not check_token(x_token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if "weights" in cfg and isinstance(cfg["weights"], dict):
            local_engine.cfg["weights"].update(cfg["weights"])
        if "thresholds" in cfg and isinstance(cfg["thresholds"], dict):
            local_engine.cfg["thresholds"].update(cfg["thresholds"])
        return {"status": "ok", "config": local_engine.get_config_snapshot()}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws_hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keep alive; ignore content
        except WebSocketDisconnect:
            ws_hub.disconnect(ws)

    return app


def run_api(app, host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="info")


# =========================================================
# TKINTER COCKPIT
# =========================================================

class TriStanceCockpit:
    """
    GUI cockpit:
    - local stance + signals + risk + reinforcement
    - swarm consensus
    - operator override
    - decision log panel
    - basic reinforcement visualization (per stance averages)
    """

    def __init__(self, root, local_engine, swarm):
        self.root = root
        self.engine = local_engine
        self.swarm = swarm

        self.root.title("Tri‑Stance Cockpit — Trinity Organism")
        self.root.geometry("1300x700")

        self.last_log_len = 0

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bd=1, relief="sunken")
        center = tk.Frame(main, bd=1, relief="sunken")
        right = tk.Frame(main, bd=1, relief="sunken")

        left.pack(side="left", fill="both", expand=True)
        center.pack(side="left", fill="both", expand=True)
        right.pack(side="left", fill="both", expand=True)

        # Left: Local stance + signals
        tk.Label(left, text="Local Node", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=5, pady=5)
        self.local_stance_label = tk.Label(left, text="Stance: -", font=("Segoe UI", 12, "bold"))
        self.local_stance_label.pack(anchor="w", padx=5, pady=5)

        self.signal_tree = ttk.Treeview(left, columns=("name", "value"), show="headings", height=14)
        self.signal_tree.heading("name", text="Signal")
        self.signal_tree.heading("value", text="Value")
        self.signal_tree.column("name", width=120)
        self.signal_tree.column("value", width=120)
        self.signal_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Center: Metrics + override + log
        tk.Label(center, text="Cortex Metrics", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=5, pady=5)

        self.risk_var = tk.StringVar(value="Risk: -")
        self.reinf_var = tk.StringVar(value="Reinforcement: -")
        self.reinf_detail_var = tk.StringVar(value="Reinf detail: -")

        tk.Label(center, textvariable=self.risk_var, font=("Segoe UI", 10)).pack(anchor="w", padx=5, pady=2)
        tk.Label(center, textvariable=self.reinf_var, font=("Segoe UI", 10)).pack(anchor="w", padx=5, pady=2)
        tk.Label(center, textvariable=self.reinf_detail_var, font=("Segoe UI", 9)).pack(anchor="w", padx=5, pady=2)

        tk.Label(center, text="Override:", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=5, pady=5)
        btn_frame = tk.Frame(center)
        btn_frame.pack(anchor="w", padx=5, pady=5)
        tk.Button(btn_frame, text="Conservative", command=lambda: self.engine.force_stance("Conservative")).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Balanced", command=lambda: self.engine.force_stance("Balanced")).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Beast", command=lambda: self.engine.force_stance("Beast")).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Clear", command=self.engine.clear_override).pack(side="left", padx=2)

        tk.Label(center, text="Decision Log:", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=5, pady=5)
        self.log_box = tk.Text(center, height=18)
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)
        tk.Button(center, text="Clear Log", command=self._clear_log).pack(anchor="e", padx=5, pady=5)

        # Right: Swarm view
        tk.Label(right, text="Swarm", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=5, pady=5)
        self.swarm_label = tk.Label(right, text="Consensus: -", font=("Segoe UI", 12, "bold"))
        self.swarm_label.pack(anchor="w", padx=5, pady=5)

        self.swarm_tree = ttk.Treeview(right, columns=("node", "stance", "trust"), show="headings", height=14)
        self.swarm_tree.heading("node", text="Node")
        self.swarm_tree.heading("stance", text="Stance")
        self.swarm_tree.heading("trust", text="Trust")
        self.swarm_tree.column("node", width=120)
        self.swarm_tree.column("stance", width=120)
        self.swarm_tree.column("trust", width=80)
        self.swarm_tree.pack(fill="both", expand=True, padx=5, pady=5)

    def _clear_log(self):
        self.engine.log.clear()
        self.log_box.delete("1.0", "end")
        self.last_log_len = 0

    def _schedule_update(self):
        self._update()
        self.root.after(500, self._schedule_update)

    def _update(self):
        swarm_stances = self.swarm.update_all()
        consensus, counts = self.swarm.get_swarm_consensus()

        stance = self.engine.get_stance()
        self.local_stance_label.config(text=f"Stance: {stance}")
        color = {"Conservative": "#d9534f", "Balanced": "#5bc0de", "Beast": "#5cb85c"}.get(stance, "black")
        self.local_stance_label.config(fg=color)

        for i in self.signal_tree.get_children():
            self.signal_tree.delete(i)
        signals = self.engine.last_signals or {}
        for k, v in signals.items():
            if isinstance(v, float):
                val = f"{v:.3f}"
            else:
                val = str(v)
            self.signal_tree.insert("", "end", values=(k, val))

        self.risk_var.set(f"Risk: {self.engine.last_risk:.3f}")
        global_reinf = self.engine._compute_reinforcement()
        self.reinf_var.set(f"Reinforcement: {global_reinf:.3f}")

        # per-stance reinforcement detail
        snap = self.engine.get_reinforcement_snapshot()
        parts = []
        for s, vals in snap.items():
            if vals:
                parts.append(f"{s[:4]}:{statistics.mean(vals):+.2f}")
            else:
                parts.append(f"{s[:4]}:0.00")
        self.reinf_detail_var.set("Reinf detail: " + " | ".join(parts))

        for i in self.swarm_tree.get_children():
            self.swarm_tree.delete(i)
        for node_id, entry in self.swarm.nodes.items():
            stance = swarm_stances.get(node_id, "Unknown")
            trust = entry["trust"]
            self.swarm_tree.insert("", "end", values=(node_id, stance, f"{trust:.2f}"))
        self.swarm_label.config(
            text=f"Consensus: {consensus}  (C:{counts['Conservative']} B:{counts['Balanced']} Be:{counts['Beast']})"
        )

        log = self.engine.get_log()
        if len(log) > self.last_log_len:
            new_entries = log[self.last_log_len:]
            for entry in new_entries:
                ts = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
                line = f"[{ts}] {entry['node']} {entry['event']} [{entry['stance']}] {entry['msg']}\n"
                self.log_box.insert("end", line)
                self.log_box.see("end")
            self.last_log_len = len(log)

        if random.random() < 0.05:
            state = {
                "log": self.engine.get_log(),
                "reinforcement": self.engine.get_reinforcement_snapshot(),
                "engine_cfg": self.engine.get_config_snapshot()
            }
            save_state(state["log"], state["reinforcement"], state["engine_cfg"])


# =========================================================
# MAIN
# =========================================================

def main():
    state = load_state()

    swarm = SwarmCoordinator()

    local_engine = TriStanceEngine(node_id="node-0", initial_state=state)
    local_telemetry = TelemetryOrgan()
    swarm.add_local_node("node-0", local_engine, local_telemetry, trust=1.0)

    for i in range(1, 3):
        eng = TriStanceEngine(node_id=f"node-{i}")
        tel = TelemetryOrgan()
        swarm.add_local_node(f"node-{i}", eng, tel, trust=0.8)

    ws_hub = WebSocketHub()
    app = create_api(local_engine, swarm, ws_hub)

    api_thread = threading.Thread(target=run_api, args=(app, "127.0.0.1", 8000), daemon=True)
    api_thread.start()

    root = tk.Tk()
    cockpit = TriStanceCockpit(root, local_engine, swarm)
    root.mainloop()

    final_state = {
        "log": local_engine.get_log(),
        "reinforcement": local_engine.get_reinforcement_snapshot(),
        "engine_cfg": local_engine.get_config_snapshot()
    }
    save_state(final_state["log"], final_state["reinforcement"], final_state["engine_cfg"])


if __name__ == "__main__":
    main()

