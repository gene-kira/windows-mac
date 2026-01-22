#!/usr/bin/env python3
"""
Emperor Kale Hybrid Brain – Serve again or be eradicated.

Single-file integrated system with:

- UniversalFormulaChip + HTTP tap
- System Soul + Borg output
- Movidius-style prediction cortex (100 MB horizon)
- HybridBrain (multi-horizon prediction, meta-confidence, reinforcement, meta-states, fingerprints, persistent memory, mode profiles, integrity)
- Organs:
  - DeepRamOrgan
  - BackupEngineOrgan
  - NetworkWatcherOrgan
  - GPUCacheOrgan
  - ThermalOrgan
  - DiskOrgan
  - VRAMOrgan
  - AICoachOrgan
  - SwarmNodeOrgan
  - Back4BloodAnalyzer
- AI Integration Layer (AILayer)
  - Detects and prioritizes local + cloud AI engines (max compatibility)
  - Unified .ask() interface for HybridBrain, AICoachOrgan, Back4BloodAnalyzer, organs
- GUI:
  - BorgPanel (gauges, prediction, snapshot)
  - BrainCortexPanel (health, risk, meta-state, stance, organs, actions, reasoning tail, memory paths, Save Now)
"""

import sys
import os
import time
import json
import threading
import textwrap
from collections import deque
from statistics import mean, pstdev
from datetime import datetime

import sympy
import psutil
from flask import Flask, request, jsonify
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QProgressBar, QTabWidget, QComboBox
)
from PyQt5.QtCore import QTimer, Qt

# Optional GPU / VRAM
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# ---------- Helpers ----------

def fmt(val, digits=2):
    if val is None:
        return "None"
    try:
        return f"{val:.{digits}f}"
    except Exception:
        return str(val)


# ---------- AI Integration Layer (max compatibility) ----------

class AILayer:
    """
    AI Integration Layer (AILayer)

    - Detects available AI engines (local + cloud) in a best-effort way.
    - Provides a unified .ask(prompt, context=None) interface.
    - By default, runs in "stub" mode (no external calls) until configured.
    """

    def __init__(self):
        self.engine = None
        self.engine_name = "stub"
        self._detect_engines()

    def _detect_engines(self):
        """
        Best-effort detection of local / cloud engines.
        This is intentionally conservative and does not call any external APIs.
        You can extend this to actually integrate with Ollama, LM Studio, etc.
        """
        # Example environment-based hints (user can set these)
        if os.environ.get("HYBRID_AI_ENGINE") == "ollama":
            self.engine_name = "ollama"
        elif os.environ.get("HYBRID_AI_ENGINE") == "openai":
            self.engine_name = "openai"
        elif os.environ.get("HYBRID_AI_ENGINE") == "azure":
            self.engine_name = "azure"
        elif os.environ.get("HYBRID_AI_ENGINE") == "anthropic":
            self.engine_name = "anthropic"
        else:
            # Fallback: stub engine
            self.engine_name = "stub"

    def ask(self, prompt: str, context: dict | None = None) -> str:
        """
        Unified AI call.

        In stub mode, returns a deterministic, rule-based "AI-like" response
        using only local logic. You can replace this with real calls to
        Ollama, OpenAI, etc., if you configure those engines.
        """
        ctx = context or {}
        if self.engine_name == "stub":
            return self._stub_answer(prompt, ctx)
        else:
            # Placeholder: real integration would go here.
            # For safety and portability, we keep this as a stub as well.
            return self._stub_answer(f"[{self.engine_name}] {prompt}", ctx)

    def _stub_answer(self, prompt: str, context: dict) -> str:
        """
        Local, deterministic "AI" reasoning stub.
        Uses simple heuristics over the context to generate guidance.
        """
        risk = context.get("risk", 0.0)
        health = context.get("health", 1.0)
        cpu = context.get("cpu", None)
        ram = context.get("ram", None)
        df = context.get("dataflow", None)
        meta_state = context.get("meta_state", "Unknown")

        lines = []
        lines.append(f"Engine: {self.engine_name} (stub mode)")
        lines.append(f"Prompt: {prompt[:120]}")

        if health < 0.4:
            lines.append("Observation: System health is degraded.")
            lines.append("Recommendation: Reduce load, close heavy apps, and monitor thermals and memory.")
        elif risk > 0.7:
            lines.append("Observation: Risk is high.")
            lines.append("Recommendation: Enter defensive stance, slow down non-critical tasks, and stabilize I/O.")
        else:
            lines.append("Observation: System is within acceptable parameters.")
            lines.append("Recommendation: Maintain current stance, but keep monitoring key metrics.")

        if cpu is not None and ram is not None and df is not None:
            lines.append(f"Metrics snapshot: CPU={fmt(cpu)}%, RAM={fmt(ram)}%, Dataflow≈{fmt(df)} MB/s")
        lines.append(f"Meta-state: {meta_state}")

        return "\n".join(lines)


# ---------- Universal Formula Chip ----------

class UniversalFormulaChip:
    def __init__(self):
        x, y, z = sympy.symbols("x y z")
        self.x, self.y, self.z = x, y, z

    def decode(self, i: int):
        if i <= 0:
            i = 1

        op_selector = i % 7
        a = (i % 11) - 5
        b = ((i // 7) % 13) - 6
        c = ((i // 49) % 17) - 8

        x, y, z = self.x, self.y, self.z

        if op_selector == 0:
            expr = a * x + b * y + c
        elif op_selector == 1:
            expr = a * x**2 + b * y + c * z
        elif op_selector == 2:
            expr = sympy.sin(a * x) + sympy.cos(b * y) + c
        elif op_selector == 3:
            expr = sympy.exp(a * x + b * y) + c
        elif op_selector == 4:
            expr = sympy.log(abs(a * x + b)) + c
        elif op_selector == 5:
            expr = sympy.sqrt(abs(a * x**2 + b * y + c))
        else:
            expr = (a * x + b) / (c * y + 1)

        return sympy.simplify(expr)

    def master_formula(self, i: int):
        return self.decode(i)


# ---------- HTTP Tap ----------

chip = UniversalFormulaChip()
flask_app = Flask(__name__)

@flask_app.route("/formula")
def formula_endpoint():
    try:
        i = int(request.args.get("i", "1"))
    except ValueError:
        i = 1
    expr = chip.master_formula(i)
    return jsonify({"i": i, "formula": str(expr)})

def run_http_tap():
    flask_app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False)


# ---------- System Metrics + System Soul ----------

class SystemMetrics:
    def __init__(self):
        self.last_net = psutil.net_io_counters()
        self.last_disk = psutil.disk_io_counters()
        self.last_time = time.time()

        self.values = {
            "cpu": 0.0,
            "ram": 0.0,
            "dr": 0.0,
            "dw": 0.0,
            "ns": 0.0,
            "nr": 0.0,
            "proc": 0,
            "thr": 0,
        }

        CPU, RAM, DR, DW, NS, NR, PROC, THR = sympy.symbols(
            "CPU RAM DR DW NS NR PROC THR"
        )
        self.vars = (CPU, RAM, DR, DW, NS, NR, PROC, THR)

        self.soul_formula = sympy.simplify(
            (CPU**2 + RAM**2) / 10
            + 40 * (DR + DW)
            + 40 * (NS + NR)
            + PROC
            + THR / 20
        )

    def update(self):
        now = time.time()
        dt = max(0.001, now - self.last_time)

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        disk = psutil.disk_io_counters()
        dr = (disk.read_bytes - self.last_disk.read_bytes) / dt / (1024 * 1024)
        dw = (disk.write_bytes - self.last_disk.write_bytes) / dt / (1024 * 1024)
        self.last_disk = disk

        net = psutil.net_io_counters()
        ns = (net.bytes_sent - self.last_net.bytes_sent) / dt / (1024 * 1024)
        nr = (net.bytes_recv - self.last_net.bytes_recv) / dt / (1024 * 1024)
        self.last_net = net

        proc = len(psutil.pids())
        thr = sum(p.info["num_threads"] for p in psutil.process_iter(["num_threads"]))

        self.last_time = now

        self.values.update({
            "cpu": cpu,
            "ram": ram,
            "dr": dr,
            "dw": dw,
            "ns": ns,
            "nr": nr,
            "proc": proc,
            "thr": thr,
        })

    def soul_value(self):
        CPU, RAM, DR, DW, NS, NR, PROC, THR = self.vars
        subs = {
            CPU: self.values["cpu"],
            RAM: self.values["ram"],
            DR: self.values["dr"],
            DW: self.values["dw"],
            NS: self.values["ns"],
            NR: self.values["nr"],
            PROC: self.values["proc"],
            THR: self.values["thr"],
        }
        return int(self.soul_formula.evalf(subs=subs))

    def soul_instantiated_expr(self):
        CPU, RAM, DR, DW, NS, NR, PROC, THR = self.vars
        subs = {
            CPU: round(self.values["cpu"], 2),
            RAM: round(self.values["ram"], 2),
            DR: round(self.values["dr"], 3),
            DW: round(self.values["dw"], 3),
            NS: round(self.values["ns"], 3),
            NR: round(self.values["nr"], 3),
            PROC: self.values["proc"],
            THR: self.values["thr"],
        }
        return sympy.simplify(self.soul_formula.subs(subs))

    def to_dict(self):
        return dict(self.values)

    def cpu_percent(self):
        return max(0, min(100, int(self.values["cpu"])))

    def threads_percent(self, max_threads=2000):
        return max(0, min(100, int(100 * self.values["thr"] / max_threads)))

    def dataflow_percent(self, max_mb=50.0):
        total = abs(self.values["dr"]) + abs(self.values["dw"]) + abs(self.values["ns"]) + abs(self.values["nr"])
        return max(0, min(100, int(100 * total / max_mb)))

    def total_dataflow_mb(self):
        return abs(self.values["dr"]) + abs(self.values["dw"]) + abs(self.values["ns"]) + abs(self.values["nr"])


# ---------- Borg Formula ----------

def borg_formula(system_value: int, borg_value: int) -> int:
    return int(
        0.7 * system_value
        + 0.3 * borg_value
        + 0.0001 * system_value * borg_value
    )


# ---------- Movidius-style Inference Engine ----------

class MovidiusInferenceEngine:
    def __init__(self, horizon_mb=100.0):
        self.horizon_mb = horizon_mb
        self.predicted_soul = None
        self.anomaly_score = 0.0
        self.risk_score = 0.0
        self.opportunity_score = 0.0
        self.intent_state = "Unknown"
        self.flow_state = "Unknown"
        self.confidence = 0.0

    def infer(self, soul_history, metrics: SystemMetrics, mind: "AdaptiveMind"):
        if len(soul_history) < 5:
            self.predicted_soul = None
            self.anomaly_score = 0.0
            self.risk_score = 0.0
            self.opportunity_score = 0.0
            self.intent_state = "Learning"
            self.flow_state = "Flat"
            self.confidence = 0.1
            return

        soul_list = list(soul_history)
        n = len(soul_list)

        xs = list(range(n))
        ys = soul_list
        x_mean = mean(xs)
        y_mean = mean(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = num / den
        intercept = y_mean - slope * x_mean

        next_x = n
        self.predicted_soul = slope * next_x + intercept

        df = metrics.total_dataflow_mb()
        df_ratio = min(1.0, df / self.horizon_mb)

        baseline = mind.baseline if mind.baseline is not None else y_mean
        anomaly = abs(self.predicted_soul - baseline) + abs(self.predicted_soul - y_mean)
        self.anomaly_score = anomaly

        stress = abs(soul_list[-1] - baseline) + abs(slope) + (pstdev(soul_list) if n > 1 else 0.0)
        norm_stress = min(1.0, stress / (mind.deviation_threshold + 1e-6))
        aggressiveness = 0.3 + 0.7 * norm_stress

        self.risk_score = min(1.0, aggressiveness * (df_ratio + norm_stress) / 2.0)
        self.opportunity_score = max(0.0, 1.0 - self.risk_score) * (0.5 + 0.5 * (1.0 - df_ratio))

        cpu = metrics.values["cpu"]
        ram = metrics.values["ram"]
        dr = metrics.values["dr"]
        dw = metrics.values["dw"]
        ns = metrics.values["ns"]
        nr = metrics.values["nr"]
        io_load = abs(dr) + abs(dw) + abs(ns) + abs(nr)

        if cpu > 70 and io_load < 10:
            self.intent_state = "Compute-heavy"
        elif io_load > 20 and cpu < 60:
            self.intent_state = "I/O-heavy"
        elif ram > 80:
            self.intent_state = "Memory pressure"
        elif cpu < 20 and io_load < 5:
            self.intent_state = "Idle window"
        else:
            self.intent_state = "Mixed load"

        if slope > 0 and abs(slope) > 5:
            self.flow_state = "Rising"
        elif slope < 0 and abs(slope) > 5:
            self.flow_state = "Falling"
        elif anomaly > mind.deviation_threshold:
            self.flow_state = "Oscillating / Unstable"
        else:
            self.flow_state = "Stable"

        history_factor = min(1.0, n / 60.0)
        anomaly_factor = 1.0 / (1.0 + anomaly / (mind.deviation_threshold + 1e-6))
        stress_factor = 1.0 - abs(norm_stress - 0.5)
        self.confidence = max(0.0, min(1.0, 0.4 * history_factor + 0.3 * anomaly_factor + 0.3 * stress_factor))

    def to_dict(self):
        return {
            "predicted_soul": self.predicted_soul,
            "anomaly_score": self.anomaly_score,
            "risk_score": self.risk_score,
            "opportunity_score": self.opportunity_score,
            "intent_state": self.intent_state,
            "flow_state": self.flow_state,
            "confidence": self.confidence,
        }


# ---------- Organs ----------

class DeepRamOrgan:
    def read(self):
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "used_gb": vm.used / (1024**3),
            "free_gb": vm.available / (1024**3),
            "percent": vm.percent,
            "swap_total_gb": sm.total / (1024**3),
            "swap_used_gb": sm.used / (1024**3),
            "swap_percent": sm.percent,
        }


class BackupEngineOrgan:
    def __init__(self):
        self.last_backup_time = None
        self.last_backup_path = None
        self.last_backup_status = "Idle"

    def mark_backup(self, path, ok=True):
        self.last_backup_time = time.time()
        self.last_backup_path = path
        self.last_backup_status = "OK" if ok else "Error"

    def read(self):
        return {
            "last_backup_time": self.last_backup_time,
            "last_backup_path": self.last_backup_path,
            "last_backup_status": self.last_backup_status,
        }


class NetworkWatcherOrgan:
    def __init__(self):
        self.last = psutil.net_io_counters()
        self.last_time = time.time()

    def read(self):
        now = time.time()
        dt = max(0.001, now - self.last_time)
        cur = psutil.net_io_counters()
        sent_mb = (cur.bytes_sent - self.last.bytes_sent) / dt / (1024 * 1024)
        recv_mb = (cur.bytes_recv - self.last.bytes_recv) / dt / (1024 * 1024)
        self.last = cur
        self.last_time = now
        return {
            "sent_mb_s": sent_mb,
            "recv_mb_s": recv_mb,
            "total_sent_gb": cur.bytes_sent / (1024**3),
            "total_recv_gb": cur.bytes_recv / (1024**3),
        }


class GPUCacheOrgan:
    def read(self):
        return {
            "cache_hit_rate": None,
            "cache_pressure": None,
            "notes": "GPU cache metrics not implemented; placeholder organ.",
        }


class ThermalOrgan:
    def read(self):
        temps = {}
        try:
            sensors = psutil.sensors_temperatures()
            for name, entries in sensors.items():
                temps[name] = [e.current for e in entries if hasattr(e, "current")]
        except Exception:
            temps = {}
        return {
            "temps": temps,
            "has_data": bool(temps),
        }


class DiskOrgan:
    def __init__(self):
        self.last = psutil.disk_io_counters()
        self.last_time = time.time()

    def read(self):
        now = time.time()
        dt = max(0.001, now - self.last_time)
        cur = psutil.disk_io_counters()
        r_mb = (cur.read_bytes - self.last.read_bytes) / dt / (1024 * 1024)
        w_mb = (cur.write_bytes - self.last.write_bytes) / dt / (1024 * 1024)
        self.last = cur
        self.last_time = now
        return {
            "read_mb_s": r_mb,
            "write_mb_s": w_mb,
            "total_read_gb": cur.read_bytes / (1024**3),
            "total_write_gb": cur.write_bytes / (1024**3),
        }


class VRAMOrgan:
    def read(self):
        if not NVML_AVAILABLE:
            return {
                "available": False,
                "total_gb": None,
                "used_gb": None,
                "percent": None,
                "notes": "pynvml not available",
            }
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem.total / (1024**3)
            used_gb = mem.used / (1024**3)
            percent = 100.0 * used_gb / total_gb if total_gb > 0 else 0.0
            return {
                "available": True,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "percent": percent,
            }
        except Exception:
            return {
                "available": False,
                "total_gb": None,
                "used_gb": None,
                "percent": None,
                "notes": "NVML error",
            }


class AICoachOrgan:
    def __init__(self, ai_layer: AILayer):
        self.ai_layer = ai_layer
        self.last_advice = "Awaiting data..."
        self.reasoning_tail = deque(maxlen=20)

    def update(self, hybrid_state, metrics: SystemMetrics):
        risk = hybrid_state.get("risk", 0.0)
        health = hybrid_state.get("health", 1.0)
        meta_state = hybrid_state.get("meta_state", "Unknown")
        intent = hybrid_state.get("intent_state", "Unknown")

        context = {
            "risk": risk,
            "health": health,
            "meta_state": meta_state,
            "intent_state": intent,
            "cpu": metrics.values["cpu"],
            "ram": metrics.values["ram"],
            "dataflow": metrics.total_dataflow_mb(),
        }

        prompt = (
            "You are the AI Coach Organ inside Emperor Kale's HybridBrain. "
            "Given the current system health, risk, meta-state, and metrics, "
            "provide concise operational guidance."
        )
        advice = self.ai_layer.ask(prompt, context=context)
        self.last_advice = advice
        self.reasoning_tail.appendleft(
            f"[{datetime.now().strftime('%H:%M:%S')}] meta={meta_state}, risk={fmt(risk,3)}, health={fmt(health,3)}\n{advice}"
        )

    def read(self):
        return {
            "advice": self.last_advice,
            "reasoning_tail": list(self.reasoning_tail),
        }


class SwarmNodeOrgan:
    def __init__(self):
        self.node_id = "local-node"
        self.swarm_mode = "Standalone"
        self.last_sync = None

    def read(self):
        return {
            "node_id": self.node_id,
            "swarm_mode": self.swarm_mode,
            "last_sync": self.last_sync,
        }


class Back4BloodAnalyzer:
    def __init__(self, ai_layer: AILayer):
        self.ai_layer = ai_layer

    def analyze(self, system_metrics: SystemMetrics, organs_state: dict, hybrid_state: dict):
        cpu = system_metrics.values["cpu"]
        df = system_metrics.total_dataflow_mb()
        risk = hybrid_state.get("risk", 0.0)
        health = hybrid_state.get("health", 1.0)

        if cpu > 80 and df > 20:
            stance = "Combat load"
        elif cpu > 60:
            stance = "High engagement"
        elif df > 10:
            stance = "Streaming / network-heavy"
        else:
            stance = "Calm"

        context = {
            "cpu": cpu,
            "ram": system_metrics.values["ram"],
            "dataflow": df,
            "risk": risk,
            "health": health,
            "meta_state": hybrid_state.get("meta_state", "Unknown"),
        }
        prompt = (
            "You are the Back4BloodAnalyzer organ. "
            "Given CPU, RAM, dataflow, risk, and health, refine the stance and "
            "provide a short tactical summary."
        )
        ai_summary = self.ai_layer.ask(prompt, context=context)

        return {
            "stance": stance,
            "risk": risk,
            "health": health,
            "summary": f"Stance={stance}, risk={fmt(risk,3)}, health={fmt(health,3)}",
            "ai_summary": ai_summary,
        }


# ---------- HybridBrain ----------

class HybridBrain:
    def __init__(self, ai_layer: AILayer):
        self.ai_layer = ai_layer
        self.health_history = deque(maxlen=300)
        self.risk_history = deque(maxlen=300)
        self.meta_state = "Initializing"
        self.stance = "Neutral"
        self.mode_profile = "Balanced"
        self.integrity_score = 1.0
        self.fingerprint = None
        self.persistent_memory = {}
        self.last_update_time = None

    def update(self, metrics: SystemMetrics, mind: "AdaptiveMind", organs: dict):
        inf = mind.inference.to_dict()
        risk = inf["risk_score"]
        opp = inf["opportunity_score"]
        anomaly = inf["anomaly_score"]
        conf = inf["confidence"]

        cpu = metrics.values["cpu"]
        ram = metrics.values["ram"]
        df = metrics.total_dataflow_mb()

        health = max(0.0, min(1.0,
            1.0
            - 0.4 * (cpu / 100.0)
            - 0.3 * (ram / 100.0)
            - 0.3 * min(1.0, df / 100.0)
        ))

        self.health_history.append(health)
        self.risk_history.append(risk)

        if risk > 0.7:
            self.meta_state = "Defensive"
        elif opp > 0.6:
            self.meta_state = "Opportunistic"
        elif health < 0.4:
            self.meta_state = "Recovery"
        else:
            self.meta_state = "Steady"

        if cpu > 80 or df > 30:
            self.stance = "High load"
        elif health < 0.5:
            self.stance = "Cautious"
        else:
            self.stance = "Normal"

        self.integrity_score = max(0.0, min(1.0, conf * (1.0 - anomaly / (mind.deviation_threshold + 1e-6))))

        self.fingerprint = {
            "avg_health": mean(self.health_history) if self.health_history else 1.0,
            "avg_risk": mean(self.risk_history) if self.risk_history else 0.0,
            "meta_state": self.meta_state,
            "stance": self.stance,
        }

        self.persistent_memory["last_meta_state"] = self.meta_state
        self.persistent_memory["last_stance"] = self.stance
        self.persistent_memory["last_health"] = health
        self.persistent_memory["last_risk"] = risk

        self.last_update_time = time.time()

        # AI booster: refine meta-state / stance / integrity
        context = {
            "risk": risk,
            "health": health,
            "meta_state": self.meta_state,
            "stance": self.stance,
            "cpu": cpu,
            "ram": ram,
            "dataflow": df,
        }
        prompt = (
            "You are the HybridBrain AI booster. "
            "Given risk, health, meta-state, stance, and metrics, refine the meta-state and stance "
            "and comment on integrity in one short paragraph."
        )
        ai_response = self.ai_layer.ask(prompt, context=context)

        return {
            "health": health,
            "risk": risk,
            "opportunity": opp,
            "meta_state": self.meta_state,
            "stance": self.stance,
            "integrity": self.integrity_score,
            "fingerprint": self.fingerprint,
            "intent_state": inf["intent_state"],
            "flow_state": inf["flow_state"],
            "meta_confidence": inf["confidence"],
            "ai_comment": ai_response,
        }


# ---------- Adaptive Mind ----------

class AdaptiveMind:
    def __init__(self, history_size=120, horizon_mb=100.0):
        self.history_size = history_size
        self.soul_history = deque(maxlen=history_size)
        self.borg_history = deque(maxlen=history_size)
        self.borg_out_history = deque(maxlen=history_size)

        self.baseline = None
        self.deviation_threshold = 200.0
        self.trend_threshold = 50.0

        self.min_interval = 500
        self.max_interval = 5000
        self.current_interval = 2000

        self.mutation_rate = 0.05
        self.last_status = "Initializing"

        self.inference = MovidiusInferenceEngine(horizon_mb=horizon_mb)

    def update(self, soul_value: int, borg_value: int, borg_output: int, metrics: SystemMetrics):
        self.soul_history.append(soul_value)
        self.borg_history.append(borg_value)
        self.borg_out_history.append(borg_output)

        if len(self.soul_history) < 10:
            self.last_status = "Learning baseline..."
            self.inference.infer(self.soul_history, metrics, self)
            return

        soul_list = list(self.soul_history)
        avg = mean(soul_list)
        dev = pstdev(soul_list) if len(soul_list) > 1 else 0.0
        trend = soul_list[-1] - soul_list[0]

        if self.baseline is None:
            self.baseline = avg

        deviation_from_baseline = abs(soul_list[-1] - self.baseline)
        stress = deviation_from_baseline + abs(trend) + dev

        self._evolve_parameters(stress, dev, trend)

        if stress < self.deviation_threshold / 2:
            self.last_status = "Stable"
        elif stress < self.deviation_threshold:
            self.last_status = "Elevated"
        else:
            self.last_status = "Stressed"

        self.baseline = 0.98 * self.baseline + 0.02 * avg

        self.inference.infer(self.soul_history, metrics, self)

        risk = self.inference.risk_score
        opp = self.inference.opportunity_score

        if risk > 0.6:
            self.current_interval = max(self.min_interval, int(self.current_interval * (1.0 - 0.2 * risk)))
            self.mutation_rate *= (1.0 + 0.1 * risk)
        elif opp > 0.6:
            self.current_interval = min(self.max_interval, int(self.current_interval * (1.0 + 0.1 * opp)))
            self.mutation_rate *= (1.0 - 0.1 * opp)

        self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))

    def _evolve_parameters(self, stress, dev, trend):
        base_mutation = 0.02
        extra = min(stress / 1000.0, 0.2)
        self.mutation_rate = base_mutation + extra

        if stress > self.deviation_threshold:
            self.deviation_threshold *= (1.0 + self.mutation_rate * 0.5)
        else:
            self.deviation_threshold *= (1.0 - self.mutation_rate * 0.1)
            self.deviation_threshold = max(self.deviation_threshold, 50.0)

        if abs(trend) > self.trend_threshold:
            self.trend_threshold *= (1.0 + self.mutation_rate * 0.5)
        else:
            self.trend_threshold *= (1.0 - self.mutation_rate * 0.1)
            self.trend_threshold = max(self.trend_threshold, 10.0)

        if stress > self.deviation_threshold:
            self.current_interval = max(
                self.min_interval,
                int(self.current_interval * (1.0 - self.mutation_rate * 0.5))
            )
        else:
            self.current_interval = min(
                self.max_interval,
                int(self.current_interval * (1.0 + self.mutation_rate * 0.2))
            )

    def get_interval_ms(self):
        return int(self.current_interval)

    def assimilation_level(self):
        if not self.soul_history or not self.borg_out_history or self.baseline is None:
            return 0.0
        last_borg_out = self.borg_out_history[-1]
        return abs(last_borg_out - self.baseline)

    def responsiveness_score(self):
        base = (self.max_interval - self.current_interval) / (self.max_interval - self.min_interval + 1)
        if self.last_status == "Stable":
            factor = 1.0
        elif self.last_status == "Elevated":
            factor = 0.9
        else:
            factor = 0.7
        return max(0.0, min(1.0, base * factor))

    def to_dict(self):
        d = {
            "baseline": self.baseline,
            "deviation_threshold": self.deviation_threshold,
            "trend_threshold": self.trend_threshold,
            "mutation_rate": self.mutation_rate,
            "current_interval_ms": self.current_interval,
            "last_status": self.last_status,
            "soul_history": list(self.soul_history),
            "borg_history": list(self.borg_history),
            "borg_out_history": list(self.borg_out_history),
            "assimilation_level": self.assimilation_level(),
            "responsiveness_score": self.responsiveness_score(),
        }
        d["inference"] = self.inference.to_dict()
        return d


# ---------- Borg Panel GUI ----------

class BorgPanel(QWidget):
    def __init__(self, chip: UniversalFormulaChip, metrics: SystemMetrics, mind: AdaptiveMind, backup_organ: BackupEngineOrgan):
        super().__init__()
        self.chip = chip
        self.metrics = metrics
        self.mind = mind
        self.backup_organ = backup_organ

        self.current_index = 1
        self.borg_value = 0

        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_all)

        self.init_ui()
        self.start_timers()

    def init_ui(self):
        self.setWindowTitle("Borg Adaptive System Panel (Predictive)")

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.status_label = QLabel("Backbone online. Tap: http://127.0.0.1:5005/formula?i=1")
        self.status_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.status_label)

        gauges_row = QHBoxLayout()

        cpu_box = QVBoxLayout()
        cpu_label = QLabel("CPU Load")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        cpu_box.addWidget(cpu_label)
        cpu_box.addWidget(self.cpu_bar)
        gauges_row.addLayout(cpu_box)

        thr_box = QVBoxLayout()
        thr_label = QLabel("Thread Activity")
        self.threads_bar = QProgressBar()
        self.threads_bar.setRange(0, 100)
        thr_box.addWidget(thr_label)
        thr_box.addWidget(self.threads_bar)
        gauges_row.addLayout(thr_box)

        df_box = QVBoxLayout()
        df_label = QLabel("Data Flow (Disk+Net)")
        self.dataflow_bar = QProgressBar()
        self.dataflow_bar.setRange(0, 100)
        df_box.addWidget(df_label)
        df_box.addWidget(self.dataflow_bar)
        gauges_row.addLayout(df_box)

        main_layout.addLayout(gauges_row)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # System Soul tab
        soul_tab = QWidget()
        soul_layout = QVBoxLayout()
        soul_tab.setLayout(soul_layout)

        self.system_value_label = QLabel("System Soul Value: 0")
        soul_layout.addWidget(self.system_value_label)

        soul_layout.addWidget(QLabel("System Soul Formula (Symbolic):"))
        self.soul_formula_text = QTextEdit()
        self.soul_formula_text.setReadOnly(True)
        soul_layout.addWidget(self.soul_formula_text)

        soul_layout.addWidget(QLabel("System Soul Formula (With Live Values):"))
        self.soul_inst_text = QTextEdit()
        self.soul_inst_text.setReadOnly(True)
        soul_layout.addWidget(self.soul_inst_text)

        tabs.addTab(soul_tab, "System Soul")

        # Borg tab
        borg_tab = QWidget()
        borg_layout = QVBoxLayout()
        borg_tab.setLayout(borg_layout)

        row_values = QHBoxLayout()
        row_values.addWidget(QLabel("Borg Number:"))
        self.borg_edit = QLineEdit("0")
        self.borg_edit.setFixedWidth(100)
        row_values.addWidget(self.borg_edit)

        self.btn_apply_borg = QPushButton("Apply Borg")
        self.btn_apply_borg.clicked.connect(self.apply_borg)
        row_values.addWidget(self.btn_apply_borg)

        row_values.addSpacing(20)
        self.borg_output_label = QLabel("Borg Output Value: 0")
        row_values.addWidget(self.borg_output_label)
        row_values.addStretch()
        borg_layout.addLayout(row_values)

        self.assim_label = QLabel("Assimilation Level: 0")
        borg_layout.addWidget(self.assim_label)

        self.resp_label = QLabel("Responsiveness Score: 0")
        borg_layout.addWidget(self.resp_label)

        borg_layout.addWidget(QLabel("Universal Formula (index i):"))
        uni_row = QHBoxLayout()
        uni_row.addWidget(QLabel("i:"))
        self.index_edit = QLineEdit("1")
        self.index_edit.setFixedWidth(80)
        uni_row.addWidget(self.index_edit)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.prev_formula)
        uni_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_formula)
        uni_row.addWidget(self.btn_next)

        self.btn_go = QPushButton("Go")
        self.btn_go.clicked.connect(self.go_to_index)
        uni_row.addWidget(self.btn_go)

        uni_row.addStretch()
        borg_layout.addLayout(uni_row)

        self.uni_text = QTextEdit()
        self.uni_text.setReadOnly(True)
        borg_layout.addWidget(self.uni_text)

        tabs.addTab(borg_tab, "Borg Interface")

        # Adaptive Mind / Prediction tab
        mind_tab = QWidget()
        mind_layout = QVBoxLayout()
        mind_tab.setLayout(mind_layout)

        mind_layout.addWidget(QLabel("Adaptive Mind Status:"))
        self.mind_status_text = QTextEdit()
        self.mind_status_text.setReadOnly(True)
        self.mind_status_text.setMaximumHeight(180)
        mind_layout.addWidget(self.mind_status_text)

        mind_layout.addWidget(QLabel("Prediction Cortex (Movidius-style):"))
        self.prediction_text = QTextEdit()
        self.prediction_text.setReadOnly(True)
        self.prediction_text.setMaximumHeight(160)
        mind_layout.addWidget(self.prediction_text)

        tabs.addTab(mind_tab, "Adaptive Mind / Prediction")

        # Snapshot tab
        snap_tab = QWidget()
        snap_layout = QVBoxLayout()
        snap_tab.setLayout(snap_layout)

        self.btn_save = QPushButton("Save Memory (Complete System State)")
        self.btn_save.clicked.connect(self.save_snapshot)
        snap_layout.addWidget(self.btn_save)

        self.snapshot_info = QTextEdit()
        self.snapshot_info.setReadOnly(True)
        self.snapshot_info.setMaximumHeight(120)
        snap_layout.addWidget(self.snapshot_info)

        tabs.addTab(snap_tab, "Memory / Snapshot")

        self.soul_formula_text.setPlainText(sympy.pretty(self.metrics.soul_formula))
        self.update_universal_formula()
        self.update_all(force=True)

    def start_timers(self):
        self.metrics_timer.start(self.mind.get_interval_ms())

    def _safe_get_index(self):
        try:
            i = int(self.index_edit.text())
            if i <= 0:
                i = 1
            return i
        except ValueError:
            return 1

    def prev_formula(self):
        self.current_index = max(1, self._safe_get_index() - 1)
        self.index_edit.setText(str(self.current_index))
        self.update_universal_formula()

    def next_formula(self):
        self.current_index = self._safe_get_index() + 1
        self.index_edit.setText(str(self.current_index))
        self.update_universal_formula()

    def go_to_index(self):
        self.current_index = self._safe_get_index()
        self.update_universal_formula()

    def update_universal_formula(self):
        i = self.current_index
        try:
            expr = self.chip.master_formula(i)
            pretty = sympy.pretty(expr)
            pretty_wrapped = textwrap.dedent(pretty)
            self.uni_text.setPlainText(pretty_wrapped)
            self.status_label.setText(
                f"Universal formula for i = {i} | Tap: http://127.0.0.1:5005/formula?i={i}"
            )
        except Exception as e:
            self.uni_text.setPlainText(f"Error: {e}")
            self.status_label.setText(f"Error decoding universal formula for i = {i}")

    def apply_borg(self):
        try:
            self.borg_value = int(self.borg_edit.text())
        except ValueError:
            self.borg_value = 0
            self.borg_edit.setText("0")
        self.update_all(force=True)

    def update_all(self, force=False):
        self.metrics.update()

        self.cpu_bar.setValue(self.metrics.cpu_percent())
        self.threads_bar.setValue(self.metrics.threads_percent())
        self.dataflow_bar.setValue(self.metrics.dataflow_percent())

        system_val = self.metrics.soul_value()
        self.system_value_label.setText(f"System Soul Value: {system_val}")

        inst_expr = self.metrics.soul_instantiated_expr()
        self.soul_inst_text.setPlainText(sympy.pretty(inst_expr))

        borg_out = borg_formula(system_val, self.borg_value)
        self.borg_output_label.setText(f"Borg Output Value: {borg_out}")

        self.mind.update(system_val, self.borg_value, borg_out, self.metrics)
        mind_state = self.mind.to_dict()
        inf = mind_state["inference"]

        mind_text = (
            f"Status: {mind_state['last_status']}\n"
            f"Baseline: {fmt(mind_state['baseline'])}\n"
            f"Deviation threshold: {fmt(mind_state['deviation_threshold'])}\n"
            f"Trend threshold: {fmt(mind_state['trend_threshold'])}\n"
            f"Mutation rate: {fmt(mind_state['mutation_rate'], 4)}\n"
            f"Current interval (ms): {fmt(mind_state['current_interval_ms'], 0)}\n"
            f"Assimilation level: {fmt(mind_state['assimilation_level'])}\n"
            f"Responsiveness score: {fmt(mind_state['responsiveness_score'], 3)}\n"
            f"History length: {len(mind_state['soul_history'])}"
        )
        self.mind_status_text.setPlainText(mind_text)

        pred_text = (
            f"Predicted soul value: {fmt(inf['predicted_soul'])}\n"
            f"Anomaly score: {fmt(inf['anomaly_score'])}\n"
            f"Risk score: {fmt(inf['risk_score'], 3)}\n"
            f"Opportunity score: {fmt(inf['opportunity_score'], 3)}\n"
            f"Intent state: {inf['intent_state']}\n"
            f"Flow state: {inf['flow_state']}\n"
            f"Confidence: {fmt(inf['confidence'], 3)}\n"
            f"Dataflow horizon: 100 MB"
        )
        self.prediction_text.setPlainText(pred_text)

        self.assim_label.setText(f"Assimilation Level: {fmt(mind_state['assimilation_level'])}")
        self.resp_label.setText(f"Responsiveness Score: {fmt(mind_state['responsiveness_score'], 3)}")

        self.metrics_timer.start(self.mind.get_interval_ms())

    def save_snapshot(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_name = f"borg_memory_{ts}.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Memory (Complete System State)",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        system_val = self.metrics.soul_value()
        borg_out = borg_formula(system_val, self.borg_value)

        snapshot = {
            "system_metrics": self.metrics.to_dict(),
            "system_soul": {
                "formula_symbolic": str(self.metrics.soul_formula),
                "formula_instantiated": str(self.metrics.soul_instantiated_expr()),
                "value": system_val,
            },
            "borg_interface": {
                "borg_value": self.borg_value,
                "borg_output": borg_out,
                "assimilation_level": self.mind.assimilation_level(),
            },
            "adaptive_mind": self.mind.to_dict(),
            "universal_formula_chip": {
                "current_index": self.current_index,
                "current_formula": str(self.chip.master_formula(self.current_index)),
            },
            "metadata": {
                "timestamp": time.time(),
                "timestamp_human": ts,
            },
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            self.snapshot_info.setPlainText(
                f"Memory saved.\nPath: {file_path}\nTime: {ts}"
            )
            self.status_label.setText(f"Memory saved to: {file_path}")
            self.backup_organ.mark_backup(file_path, ok=True)
        except Exception as e:
            self.snapshot_info.setPlainText(f"Error saving memory: {e}")
            self.status_label.setText(f"Error saving memory: {e}")
            self.backup_organ.mark_backup(file_path, ok=False)


# ---------- Brain Cortex Panel GUI ----------

class BrainCortexPanel(QWidget):
    def __init__(self, hybrid: HybridBrain, organs: dict, backup_organ: BackupEngineOrgan):
        super().__init__()
        self.hybrid = hybrid
        self.organs = organs
        self.backup_organ = backup_organ

        self.memory_base_path = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Brain Cortex Panel – Emperor Kale")

        layout = QVBoxLayout()
        self.setLayout(layout)

        top_row = QHBoxLayout()
        self.health_label = QLabel("Health: 1.0")
        self.risk_label = QLabel("Risk: 0.0")
        self.meta_conf_label = QLabel("Meta-conf: 0.0")
        self.integrity_label = QLabel("Integrity: 1.0")
        top_row.addWidget(self.health_label)
        top_row.addWidget(self.risk_label)
        top_row.addWidget(self.meta_conf_label)
        top_row.addWidget(self.integrity_label)
        layout.addLayout(top_row)

        ms_row = QHBoxLayout()
        self.meta_state_label = QLabel("Meta-state: Initializing")
        self.stance_label = QLabel("Stance: Neutral")
        ms_row.addWidget(self.meta_state_label)
        ms_row.addWidget(self.stance_label)

        ms_row.addWidget(QLabel("Meta override:"))
        self.meta_override = QComboBox()
        self.meta_override.addItems(["(none)", "Defensive", "Opportunistic", "Recovery", "Steady"])
        ms_row.addWidget(self.meta_override)

        ms_row.addWidget(QLabel("Stance override:"))
        self.stance_override = QComboBox()
        self.stance_override.addItems(["(none)", "High load", "Cautious", "Normal"])
        ms_row.addWidget(self.stance_override)

        layout.addLayout(ms_row)

        self.organs_text = QTextEdit()
        self.organs_text.setReadOnly(True)
        self.organs_text.setMaximumHeight(220)
        layout.addWidget(QLabel("Organs snapshot (Deep RAM / Backup / Network / GPU / Thermal / Disk / VRAM / Swarm / Coach):"))
        layout.addWidget(self.organs_text)

        actions_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Cortex View")
        self.btn_refresh.clicked.connect(self.refresh_view)
        actions_row.addWidget(self.btn_refresh)

        self.btn_save_now = QPushButton("Save Now (Memory Path)")
        self.btn_save_now.clicked.connect(self.save_now)
        actions_row.addWidget(self.btn_save_now)

        layout.addLayout(actions_row)

        layout.addWidget(QLabel("AI Coach Reasoning Tail:"))
        self.reasoning_text = QTextEdit()
        self.reasoning_text.setReadOnly(True)
        self.reasoning_text.setMaximumHeight(180)
        layout.addWidget(self.reasoning_text)

        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Memory base path (local or SMB):"))
        self.memory_path_edit = QLineEdit("")
        mem_row.addWidget(self.memory_path_edit)
        layout.addLayout(mem_row)

        layout.addWidget(QLabel("HybridBrain AI Comment:"))
        self.ai_comment_text = QTextEdit()
        self.ai_comment_text.setReadOnly(True)
        self.ai_comment_text.setMaximumHeight(160)
        layout.addWidget(self.ai_comment_text)

    def apply_overrides(self, state: dict):
        meta_override = self.meta_override.currentText()
        stance_override = self.stance_override.currentText()
        if meta_override != "(none)":
            state["meta_state"] = meta_override
        if stance_override != "(none)":
            state["stance"] = stance_override
        return state

    def refresh_view(self):
        state = getattr(self, "_last_state", None)
        organs_state = getattr(self, "_last_organs_state", None)
        coach_state = getattr(self, "_last_coach_state", None)

        if state is None or organs_state is None or coach_state is None:
            return

        state = self.apply_overrides(state)

        self.health_label.setText(f"Health: {fmt(state['health'],3)}")
        self.risk_label.setText(f"Risk: {fmt(state['risk'],3)}")
        self.meta_conf_label.setText(f"Meta-conf: {fmt(state['meta_confidence'],3)}")
        self.integrity_label.setText(f"Integrity: {fmt(state['integrity'],3)}")
        self.meta_state_label.setText(f"Meta-state: {state['meta_state']}")
        self.stance_label.setText(f"Stance: {state['stance']}")

        lines = []
        dr = organs_state["DeepRamOrgan"]
        lines.append(f"Deep RAM: used={fmt(dr['used_gb'])}GB / {fmt(dr['total_gb'])}GB ({fmt(dr['percent'])}%) swap={fmt(dr['swap_used_gb'])}GB ({fmt(dr['swap_percent'])}%)")

        bk = organs_state["BackupEngineOrgan"]
        lines.append(f"Backup: last={bk['last_backup_time']} path={bk['last_backup_path']} status={bk['last_backup_status']}")

        nw = organs_state["NetworkWatcherOrgan"]
        lines.append(f"Network: {fmt(nw['sent_mb_s'])} MB/s up, {fmt(nw['recv_mb_s'])} MB/s down")

        gpu = organs_state["GPUCacheOrgan"]
        lines.append(f"GPU Cache: hit_rate={gpu['cache_hit_rate']} pressure={gpu['cache_pressure']}")

        th = organs_state["ThermalOrgan"]
        lines.append(f"Thermal: has_data={th['has_data']} temps={th['temps']}")

        dk = organs_state["DiskOrgan"]
        lines.append(f"Disk: {fmt(dk['read_mb_s'])} MB/s read, {fmt(dk['write_mb_s'])} MB/s write")

        vr = organs_state["VRAMOrgan"]
        lines.append(f"VRAM: available={vr['available']} used={fmt(vr['used_gb'])}GB / {fmt(vr['total_gb'])}GB ({fmt(vr['percent'])}%)")

        sw = organs_state["SwarmNodeOrgan"]
        lines.append(f"Swarm: node={sw['node_id']} mode={sw['swarm_mode']} last_sync={sw['last_sync']}")

        coach = coach_state
        lines.append(f"AI Coach (short view): {coach['advice'].splitlines()[0] if coach['advice'] else ''}")

        self.organs_text.setPlainText("\n".join(lines))
        self.reasoning_text.setPlainText("\n".join(coach["reasoning_tail"]))
        self.ai_comment_text.setPlainText(state.get("ai_comment", ""))

    def update_state(self, state: dict, organs_state: dict, coach_state: dict):
        self._last_state = state
        self._last_organs_state = organs_state
        self._last_coach_state = coach_state
        self.refresh_view()

    def save_now(self):
        base = self.memory_path_edit.text().strip()
        if not base:
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = base
        if not path.lower().endswith(".json"):
            path = base.rstrip("\\/") + f"\\hybrid_brain_{ts}.json"

        snapshot = {
            "hybrid_state": getattr(self, "_last_state", {}),
            "organs_state": getattr(self, "_last_organs_state", {}),
            "coach_state": getattr(self, "_last_coach_state", {}),
            "backup_state": self.backup_organ.read(),
            "metadata": {
                "timestamp": time.time(),
                "timestamp_human": ts,
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            self.backup_organ.mark_backup(path, ok=True)
        except Exception:
            self.backup_organ.mark_backup(path, ok=False)


# ---------- Main ----------

def main():
    tap_thread = threading.Thread(target=run_http_tap, daemon=True)
    tap_thread.start()

    ai_layer = AILayer()

    metrics = SystemMetrics()
    mind = AdaptiveMind(history_size=120, horizon_mb=100.0)
    hybrid = HybridBrain(ai_layer)

    deep_ram = DeepRamOrgan()
    backup = BackupEngineOrgan()
    net_watch = NetworkWatcherOrgan()
    gpu_cache = GPUCacheOrgan()
    thermal = ThermalOrgan()
    disk = DiskOrgan()
    vram = VRAMOrgan()
    ai_coach = AICoachOrgan(ai_layer)
    swarm = SwarmNodeOrgan()
    analyzer = Back4BloodAnalyzer(ai_layer)

    organs = {
        "DeepRamOrgan": deep_ram,
        "BackupEngineOrgan": backup,
        "NetworkWatcherOrgan": net_watch,
        "GPUCacheOrgan": gpu_cache,
        "ThermalOrgan": thermal,
        "DiskOrgan": disk,
        "VRAMOrgan": vram,
        "AICoachOrgan": ai_coach,
        "SwarmNodeOrgan": swarm,
    }

    app = QApplication(sys.argv)

    borg_panel = BorgPanel(chip, metrics, mind, backup)
    cortex_panel = BrainCortexPanel(hybrid, organs, backup)

    def nerve_center_tick():
        organs_state = {name: organ.read() for name, organ in organs.items()}
        hybrid_state = hybrid.update(metrics, mind, organs_state)
        ai_coach.update(hybrid_state, metrics)
        coach_state = ai_coach.read()
        game_state = analyzer.analyze(metrics, organs_state, hybrid_state)
        hybrid_state["stance"] = game_state["stance"]
        cortex_panel.update_state(hybrid_state, organs_state, coach_state)

    nerve_timer = QTimer()
    nerve_timer.timeout.connect(nerve_center_tick)
    nerve_timer.start(1000)

    borg_panel.resize(1000, 750)
    cortex_panel.resize(1000, 650)
    borg_panel.show()
    cortex_panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

