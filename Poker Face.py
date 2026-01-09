#!/usr/bin/env python3
"""
MagicBox Mission Control – Predictive Tri‑Stance Engine (Tkinter edition)

Features:
- Autoloader for dependencies (psutil, numpy)
- Live CPU/RAM sensing
- Multi‑horizon prediction (1s, 5s, 30s, 120s) via rolling history
- Meta‑confidence fusion (variance, trend stability, noise)
- Risk scoring and simple anomaly intuition
- Tri‑Stance engine: Conservative / Balanced / Beast, self‑switching
- Hybrid Brain / Judgment / Situational / Predictive / Collective panes
- Command bar + ASI dialogue window
"""

import sys
import subprocess
import importlib
import threading
import time
from collections import deque

# ========= AUTLOADER FOR LIBRARIES =========

def ensure_lib(mod_name, pip_name=None):
    """
    Try importing a module. If it fails, pip‑install it and import again.
    Old‑guy friendly: does it automatically, prints progress.
    """
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        pip_target = pip_name or mod_name
        print(f"[Autoloader] Installing {pip_target}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_target])
        return importlib.import_module(mod_name)

psutil = ensure_lib("psutil")
np = ensure_lib("numpy")

import tkinter as tk
from tkinter import ttk

# ========= CORE STATE & PREDICTION ENGINE =========

class ResourceHistory:
    """
    Tracks recent CPU load history and produces multi‑horizon forecasts.
    Horizons are in seconds; we approximate by averaging windows of samples.
    """
    def __init__(self, max_seconds=180, sample_interval=1.0):
        self.sample_interval = sample_interval
        self.max_samples = int(max_seconds / sample_interval)
        self.cpu_history = deque(maxlen=self.max_samples)
        self.ram_history = deque(maxlen=self.max_samples)

    def add_sample(self, cpu_percent, ram_percent):
        self.cpu_history.append(cpu_percent)
        self.ram_history.append(ram_percent)

    def _window_avg(self, data, seconds):
        if not data:
            return 0.0
        n = int(seconds / self.sample_interval)
        if n <= 0:
            return float(data[-1])
        if n > len(data):
            n = len(data)
        window = list(data)[-n:]
        return float(sum(window) / len(window))

    def horizons(self):
        """
        Returns a dict of multi‑horizon forecasts for CPU and RAM.
        Very simple: moving averages, but structured for upgrade to ML later.
        """
        horizon_seconds = [1, 5, 30, 120]
        cpu_forecast = {}
        ram_forecast = {}
        for h in horizon_seconds:
            cpu_forecast[h] = self._window_avg(self.cpu_history, h)
            ram_forecast[h] = self._window_avg(self.ram_history, h)
        return cpu_forecast, ram_forecast

    def cpu_variance(self):
        if len(self.cpu_history) < 3:
            return 0.0
        arr = np.array(self.cpu_history, dtype=float)
        return float(np.var(arr))

    def trend_stability(self):
        """
        Estimate trend stability as 1 / (1 + |slope|) for CPU.
        Uses simple linear regression on last N samples.
        """
        n = len(self.cpu_history)
        if n < 5:
            return 0.5
        y = np.array(self.cpu_history[-30:], dtype=float) if n >= 30 else np.array(self.cpu_history, dtype=float)
        x = np.arange(len(y))
        # slope via least squares
        denom = np.sum((x - x.mean())**2)
        if denom == 0:
            return 0.5
        slope = np.sum((x - x.mean()) * (y - y.mean())) / denom
        stability = 1.0 / (1.0 + abs(slope) / 10.0)
        # clamp 0..1
        return max(0.0, min(1.0, stability))


class JudgmentMemory:
    """
    Keeps track of reinforcement: how often our stance decisions
    led to stable vs unstable outcomes.
    """
    def __init__(self):
        self.good = 0
        self.bad = 0
        self.bias = 0.0  # positive -> optimistic, negative -> pessimistic

    def reinforce_good(self):
        self.good += 1
        self.bias += 0.05
        self.bias = min(self.bias, 1.0)

    def reinforce_bad(self):
        self.bad += 1
        self.bias -= 0.05
        self.bias = max(self.bias, -1.0)

    def reset_bias(self):
        self.bias = 0.0

    def ratio(self):
        total = self.good + self.bad
        if total == 0:
            return 0.5
        return self.good / total

    def judgment_confidence(self):
        """
        Confidence grows with sample size and how decisive the ratio is.
        """
        total = self.good + self.bad
        if total == 0:
            return 0.1
        ratio = self.ratio()
        distance = abs(ratio - 0.5) * 2.0  # 0..1
        size_factor = min(1.0, total / 100.0)
        return distance * size_factor


class TriStanceEngine:
    """
    Conservative / Balanced / Beast stance engine.

    Uses:
    - Current CPU/RAM
    - Multi‑horizon forecasts
    - Variance / trend
    - Judgment bias
    To pick a stance and compute a risk score + volatility.
    """
    def __init__(self, history: ResourceHistory, memory: JudgmentMemory):
        self.history = history
        self.memory = memory
        self.current_stance = "Balanced"
        self.mode_override = None  # "Conservative", "Balanced", "Beast"
        self.risk_tolerance = 0.5  # 0..1, higher = more risk tolerant

    def set_override(self, stance: str | None):
        self.mode_override = stance

    def set_risk_tolerance(self, value: float):
        self.risk_tolerance = max(0.0, min(1.0, value))

    def compute_meta_confidence(self, cpu_forecast):
        """
        Meta‑confidence fusion:
        - lower variance => higher confidence
        - more stable trend => higher confidence
        - more consistent horizons => higher confidence
        - modulated by reinforcement bias
        """
        var = self.history.cpu_variance()
        variance_score = 1.0 / (1.0 + var / 200.0)  # rough scaling
        stability = self.history.trend_stability()

        # horizon consistency: how similar are 1s, 5s, 30s, 120s
        vals = np.array(list(cpu_forecast.values()), dtype=float)
        if len(vals) == 0:
            horizon_score = 0.5
        else:
            h_var = np.var(vals)
            horizon_score = 1.0 / (1.0 + h_var / 200.0)

        base = (variance_score + stability + horizon_score) / 3.0
        bias_adj = 0.5 + self.memory.bias / 2.0  # 0..1
        confidence = base * 0.7 + bias_adj * 0.3
        return max(0.0, min(1.0, confidence)), var

    def compute_risk(self, cpu_forecast, ram_forecast):
        """
        Predictive risk dampening skeleton:
        Risk increases with:
        - High forecasted loads
        - Rising trend
        - High variance
        - Low meta‑confidence
        """
        cpu1 = cpu_forecast[1]
        cpu30 = cpu_forecast[30]
        ram30 = ram_forecast[30]

        # simple trend: 30s vs 5s
        cpu5 = cpu_forecast[5]
        trend = cpu30 - cpu5

        meta_conf, var = self.compute_meta_confidence(cpu_forecast)

        # base risk from forecasted utilization
        util_risk = max(cpu30, ram30) / 100.0
        trend_risk = max(0.0, trend / 50.0)
        variance_risk = min(1.0, var / 300.0)
        confidence_buffer = 1.0 - meta_conf

        raw_risk = 0.3 * util_risk + 0.2 * trend_risk + 0.2 * variance_risk + 0.3 * confidence_buffer
        # adjust by risk tolerance (more tolerant => lower displayed risk)
        adjusted = raw_risk * (1.0 - self.risk_tolerance * 0.4)
        return max(0.0, min(1.0, adjusted)), meta_conf, var, trend

    def decide_stance(self, cpu_now, ram_now, cpu_forecast, ram_forecast):
        risk, meta_conf, var, trend = self.compute_risk(cpu_forecast, ram_forecast)

        if self.mode_override:
            stance = self.mode_override
        else:
            # Conservative conditions
            if (cpu_now > 85 or ram_now > 90 or
                cpu_forecast[30] > 80 or
                risk > 0.7):
                stance = "Conservative"
            # Beast conditions
            elif (cpu_now < 60 and ram_now < 80 and
                  cpu_forecast[30] < 75 and
                  risk < 0.5 and
                  trend <= 0.0):
                stance = "Beast"
            else:
                stance = "Balanced"

        self.current_stance = stance

        # Volatility as a function of variance and trend magnitude
        volatility = min(1.0, (var / 200.0) + abs(trend) / 50.0)

        # Cognitive load proxy: how much we're doing (here: combine cpu/ram and stance)
        stance_factor = {"Conservative": 0.7, "Balanced": 1.0, "Beast": 1.3}[stance]
        cog_load = min(1.0, (cpu_now / 100.0 * 0.6 + ram_now / 100.0 * 0.4) * stance_factor)

        # Environment classification
        if risk > 0.75:
            env = "DANGER"
        elif risk > 0.5:
            env = "TENSE"
        else:
            env = "CALM"

        # Opportunity: inverse of risk but boosted by confidence
        opportunity = max(0.0, min(1.0, (1.0 - risk) * 0.7 + meta_conf * 0.3))

        return {
            "stance": stance,
            "risk": risk,
            "volatility": volatility,
            "meta_conf": meta_conf,
            "cog_load": cog_load,
            "env": env,
            "opportunity": opportunity,
            "cpu_trend": trend,
        }


# ========= ORGANISM CONTROLLER =========

class OrganismCore:
    """
    Glue between sensors, prediction engine, reinforcement memory, and GUI.
    """
    def __init__(self, sample_interval=1.0):
        self.history = ResourceHistory(max_seconds=180, sample_interval=sample_interval)
        self.memory = JudgmentMemory()
        self.engine = TriStanceEngine(self.history, self.memory)
        self.sample_interval = sample_interval

        # High‑level meta‑state
        self.mission = "AUTO"
        self.prediction_horizon_mode = "Medium"  # Short / Medium / Long
        self.collective_health = 0.9
        self.hive_density = 0.1
        self.node_agreement = 0.8

        self.running = False
        self.lock = threading.Lock()
        self.latest_snapshot = {}

    def _sensor_sample(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        return cpu, ram

    def tick(self):
        """
        One sensor+decision step. Called repeatedly in a background thread.
        """
        cpu, ram = self._sensor_sample()
        self.history.add_sample(cpu, ram)

        cpu_forecast, ram_forecast = self.history.horizons()
        decision = self.engine.decide_stance(cpu, ram, cpu_forecast, ram_forecast)

        # Fake collective health drift toward calmness unless risk high
        risk = decision["risk"]
        if risk > 0.7:
            self.collective_health -= 0.01
        else:
            self.collective_health += 0.005
        self.collective_health = max(0.0, min(1.0, self.collective_health))

        # Fake hive dynamics
        self.hive_density = max(0.0, min(1.0, self.hive_density + np.random.uniform(-0.02, 0.02)))
        self.node_agreement = max(0.0, min(1.0, self.node_agreement + np.random.uniform(-0.01, 0.01)))

        with self.lock:
            self.latest_snapshot = {
                "cpu_now": cpu,
                "ram_now": ram,
                "cpu_forecast": cpu_forecast,
                "ram_forecast": ram_forecast,
                "decision": decision,
                "collective_health": self.collective_health,
                "hive_density": self.hive_density,
                "node_agreement": self.node_agreement,
                "judgment_confidence": self.memory.judgment_confidence(),
                "good_outcomes": self.memory.good,
                "bad_outcomes": self.memory.bad,
            }

    def get_snapshot(self):
        with self.lock:
            return dict(self.latest_snapshot)

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _run_loop(self):
        while self.running:
            try:
                self.tick()
            except Exception as e:
                print("[Core] Error in tick:", e)
            time.sleep(self.sample_interval)

    # Reinforcement hooks
    def reinforce_good(self):
        self.memory.reinforce_good()

    def reinforce_bad(self):
        self.memory.reinforce_bad()

    def reset_bias(self):
        self.memory.reset_bias()

    # Mission control
    def set_mission(self, mission):
        self.mission = mission

    def set_prediction_horizon_mode(self, mode):
        self.prediction_horizon_mode = mode


# ========= TKINTER GUI =========

class MissionControlGUI:
    def __init__(self, root, core: OrganismCore):
        self.root = root
        self.core = core

        self.root.title("MagicBox Mission Control – Autonomous Cipher Engine")
        self.root.geometry("1250x800")

        # Simple ttk theming (MagicBox-ish)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#20252f")
        style.configure("TLabel", background="#20252f", foreground="#e0e6f5", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#f6d365")
        style.configure("Value.TLabel", font=("Consolas", 11), foreground="#a5ffb2")
        style.configure("Danger.Value.TLabel", font=("Consolas", 11), foreground="#ff7b7b")
        style.configure("TButton", font=("Segoe UI", 9), padding=4)
        style.configure("Panel.TLabelframe", background="#242a35", foreground="#f6d365", font=("Segoe UI", 10, "bold"))
        style.configure("Panel.TLabelframe.Label", foreground="#f6d365")

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Top row: Hybrid Brain, Judgment, Situational
        top_row = ttk.Frame(main)
        top_row.pack(fill="x", expand=False)

        self._build_hybrid_brain_panel(top_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self._build_judgment_panel(top_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self._build_situational_panel(top_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Middle row: Predictive panel + Collective
        mid_row = ttk.Frame(main)
        mid_row.pack(fill="x", expand=False)

        self._build_predictive_panel(mid_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self._build_collective_panel(mid_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Command bar + Dialogue
        bottom_row = ttk.Frame(main)
        bottom_row.pack(fill="both", expand=True)

        self._build_command_bar(bottom_row).pack(side="left", fill="y", expand=False, padx=4, pady=4)
        self._build_dialogue_panel(bottom_row).pack(side="left", fill="both", expand=True, padx=4, pady=4)

    # ----- Panels -----

    def _build_hybrid_brain_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Hybrid Brain Core", style="Panel.TLabelframe")

        self.mode_var = tk.StringVar(value="Balanced")
        self.volatility_var = tk.StringVar(value="0.00")
        self.trust_var = tk.StringVar(value="0.00")
        self.cog_load_var = tk.StringVar(value="0.00")

        ttk.Label(frame, text="Mode:", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.mode_var, style="Value.TLabel").grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Volatility:", style="Header.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.volatility_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Trust (meta‑confidence):", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.trust_var, style="Value.TLabel").grid(row=2, column=1, sticky="w")

        ttk.Label(frame, text="Cognitive Load:", style="Header.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.cog_load_var, style="Value.TLabel").grid(row=3, column=1, sticky="w")

        # Controls
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, sticky="we", pady=(8, 0))

        ttk.Button(btn_frame, text="Auto Mode", command=lambda: self.core.engine.set_override(None)).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Stability", command=lambda: self.core.engine.set_override("Conservative")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="Reflex", command=lambda: self.core.engine.set_override("Balanced")).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(btn_frame, text="Exploration", command=lambda: self.core.engine.set_override("Beast")).grid(row=0, column=3, padx=2, pady=2)

        return frame

    def _build_judgment_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Judgment Engine", style="Panel.TLabelframe")

        self.judgment_conf_var = tk.StringVar(value="0.00")
        self.good_bad_var = tk.StringVar(value="0 / 0")

        ttk.Label(frame, text="Judgment Confidence:", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.judgment_conf_var, style="Value.TLabel").grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Good / Bad Outcomes:", style="Header.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.good_bad_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(8, 0))

        ttk.Button(btn_frame, text="Reinforce (Good)", command=self._reinforce_good).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Correct (Bad)", command=self._reinforce_bad).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="Reset Bias", command=self._reset_bias).grid(row=0, column=2, padx=2, pady=2)

        return frame

    def _build_situational_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Situational Awareness Cortex", style="Panel.TLabelframe")

        self.mission_var = tk.StringVar(value="AUTO")
        self.env_var = tk.StringVar(value="CALM")
        self.opportunity_var = tk.StringVar(value="0.00")
        self.risk_var = tk.StringVar(value="0.00")

        ttk.Label(frame, text="Mission:", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.mission_var, style="Value.TLabel").grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Environment:", style="Header.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.env_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Opportunity:", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.opportunity_var, style="Value.TLabel").grid(row=2, column=1, sticky="w")

        ttk.Label(frame, text="Risk Score:", style="Header.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.risk_var, style="Value.TLabel").grid(row=3, column=1, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(8, 0))

        ttk.Button(btn_frame, text="Force PROTECT", command=lambda: self._set_mission("PROTECT")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Force LEARN", command=lambda: self._set_mission("LEARN")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="Force OPTIMIZE", command=lambda: self._set_mission("OPTIMIZE")).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(btn_frame, text="Return to AUTO", command=lambda: self._set_mission("AUTO")).grid(row=0, column=3, padx=2, pady=2)

        return frame

    def _build_predictive_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Predictive Intelligence", style="Panel.TLabelframe")

        self.cpu1_var = tk.StringVar(value="0.0")
        self.cpu5_var = tk.StringVar(value="0.0")
        self.cpu30_var = tk.StringVar(value="0.0")
        self.cpu120_var = tk.StringVar(value="0.0")
        self.anomaly_risk_var = tk.StringVar(value="0.00")

        ttk.Label(frame, text="Forecast CPU (1s / 5s / 30s / 120s):", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.cpu1_var, style="Value.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.cpu5_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(frame, textvariable=self.cpu30_var, style="Value.TLabel").grid(row=1, column=2, sticky="w")
        ttk.Label(frame, textvariable=self.cpu120_var, style="Value.TLabel").grid(row=1, column=3, sticky="w")

        ttk.Label(frame, text="Anomaly Risk (intuition):", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.anomaly_risk_var, style="Danger.Value.TLabel").grid(row=2, column=1, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=(8, 0))

        ttk.Button(btn_frame, text="Short Horizon", command=lambda: self._set_horizon("Short")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Medium Horizon", command=lambda: self._set_horizon("Medium")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="Long Horizon", command=lambda: self._set_horizon("Long")).grid(row=0, column=2, padx=2, pady=2)

        return frame

    def _build_collective_panel(self, parent):
        frame = ttk.Labelframe(parent, text="Collective Health & Hive", style="Panel.TLabelframe")

        self.collective_health_var = tk.StringVar(value="0.90")
        self.hive_density_var = tk.StringVar(value="0.10")
        self.node_agreement_var = tk.StringVar(value="0.80")

        ttk.Label(frame, text="Collective Health:", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.collective_health_var, style="Value.TLabel").grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Hive Density:", style="Header.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.hive_density_var, style="Value.TLabel").grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Node Agreement:", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.node_agreement_var, style="Value.TLabel").grid(row=2, column=1, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(8, 0))

        ttk.Button(btn_frame, text="Aggressive Sync", command=lambda: self._append_dialogue("Hive sync set to Aggressive.\n")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Conservative Sync", command=lambda: self._append_dialogue("Hive sync set to Conservative.\n")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="Local‑Only", command=lambda: self._append_dialogue("Node set to Local‑Only.\n")).grid(row=0, column=2, padx=2, pady=2)

        return frame

    def _build_command_bar(self, parent):
        frame = ttk.Labelframe(parent, text="Command Bar", style="Panel.TLabelframe")

        commands = [
            ("Stabilize System", self._cmd_stabilize),
            ("High‑Alert Mode", self._cmd_high_alert),
            ("Begin Learning Cycle", self._cmd_learning),
            ("Optimize Performance", self._cmd_optimize),
            ("Purge Anomaly Memory", self._cmd_purge),
            ("Rebuild Predictive Model", self._cmd_rebuild),
            ("Reset Situational Cortex", self._cmd_reset_situational),
            ("Snapshot Brain State", self._cmd_snapshot),
            ("Rollback to Previous", self._cmd_rollback),
        ]

        for i, (label, cmd) in enumerate(commands):
            ttk.Button(frame, text=label, command=cmd).grid(row=i, column=0, padx=4, pady=3, sticky="we")

        frame.grid_columnconfigure(0, weight=1)
        return frame

    def _build_dialogue_panel(self, parent):
        frame = ttk.Labelframe(parent, text="ASI Dialogue", style="Panel.TLabelframe")

        # Buttons for questions
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(0, 4))

        ttk.Button(btn_frame, text="Why this mission?", command=self._ask_why_mission).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="What next?", command=self._ask_what_next).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="What uncertain?", command=self._ask_uncertain).grid(row=0, column=2, padx=2, pady=2)

        ttk.Button(btn_frame, text="What do you need?", command=self._ask_need).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="Explain reasoning", command=self._ask_reasoning).grid(row=1, column=1, padx=2, pady=2)

        # Text area
        self.dialogue = tk.Text(frame, height=14, wrap="word", bg="#151821", fg="#e0e6f5", insertbackground="#f6d365")
        self.dialogue.pack(fill="both", expand=True)
        self._append_dialogue("ASI: Online. Tri‑Stance predictive engine initialized.\n")

        return frame

    # ----- UI callbacks -----

    def _reinforce_good(self):
        self.core.reinforce_good()
        self._append_dialogue("Operator: Marking recent behavior as GOOD.\n")

    def _reinforce_bad(self):
        self.core.reinforce_bad()
        self._append_dialogue("Operator: Marking recent behavior as BAD.\n")

    def _reset_bias(self):
        self.core.reset_bias()
        self._append_dialogue("Operator: Judgment bias reset.\n")

    def _set_mission(self, mission):
        self.core.set_mission(mission)
        self.mission_var.set(mission)
        self._append_dialogue(f"Operator: Mission override -> {mission}\n")

    def _set_horizon(self, mode):
        self.core.set_prediction_horizon_mode(mode)
        self._append_dialogue(f"Operator: Prediction horizon -> {mode}.\n")

    # Command bar implementations (skeleton behaviors)

    def _cmd_stabilize(self):
        self.core.engine.set_override("Conservative")
        self.core.engine.set_risk_tolerance(0.1)
        self._append_dialogue("Command: Stabilize system. Entering Conservative stance with low risk tolerance.\n")

    def _cmd_high_alert(self):
        self.core.engine.set_risk_tolerance(0.0)
        self._append_dialogue("Command: High‑alert mode. Maximum risk aversion.\n")

    def _cmd_learning(self):
        self._append_dialogue("Command: Begin learning cycle. Reinforcement signals prioritized.\n")

    def _cmd_optimize(self):
        self.core.engine.set_risk_tolerance(0.8)
        self.core.engine.set_override("Beast")
        self._append_dialogue("Command: Optimize performance. Beast stance encouraged with high risk tolerance.\n")

    def _cmd_purge(self):
        self._append_dialogue("Command: Purge anomaly memory (not yet wired to persistent store).\n")

    def _cmd_rebuild(self):
        self._append_dialogue("Command: Rebuild predictive model (future: retrain / re‑fit ML model).\n")

    def _cmd_reset_situational(self):
        self._set_mission("AUTO")
        self._append_dialogue("Command: Reset situational cortex to AUTO.\n")

    def _cmd_snapshot(self):
        snap = self.core.get_snapshot()
        self._append_dialogue(f"Snapshot: {snap}\n")

    def _cmd_rollback(self):
        self._append_dialogue("Command: Rollback to previous state (placeholder – requires snapshot memory).\n")

    # Dialogue questions

    def _ask_why_mission(self):
        snap = self.core.get_snapshot()
        mission = self.core.mission
        reason = "balancing risk and opportunity based on current predictions."
        self._append_dialogue(f"ASI: Mission='{mission}' chosen because {reason}\n")

    def _ask_what_next(self):
        snap = self.core.get_snapshot()
        decision = snap.get("decision", {})
        stance = decision.get("stance", "Balanced")
        risk = decision.get("risk", 0.0)
        self._append_dialogue(
            f"ASI: Next, I plan to maintain '{stance}' stance while risk={risk:.2f} remains within current tolerance.\n"
        )

    def _ask_uncertain(self):
        snap = self.core.get_snapshot()
        decision = snap.get("decision", {})
        meta_conf = decision.get("meta_conf", 0.0)
        volatility = decision.get("volatility", 0.0)
        self._append_dialogue(
            f"ASI: My uncertainty is highest when meta‑confidence={meta_conf:.2f} and volatility={volatility:.2f} spike together.\n"
        )

    def _ask_need(self):
        snap = self.core.get_snapshot()
        self._append_dialogue("ASI: I need more reinforcement signals to refine stance thresholds.\n")

    def _ask_reasoning(self):
        snap = self.core.get_snapshot()
        decision = snap.get("decision", {})
        risk = decision.get("risk", 0.0)
        trend = decision.get("cpu_trend", 0.0)
        stance = decision.get("stance", "Balanced")
        self._append_dialogue(
            f"ASI: I chose '{stance}' because predicted trend={trend:.2f} and risk={risk:.2f} under current tolerance.\n"
        )

    # ----- Periodic refresh -----

    def _schedule_update(self):
        self._refresh_from_core()
        self.root.after(1000, self._schedule_update)

    def _refresh_from_core(self):
        snap = self.core.get_snapshot()
        if not snap:
            return
        decision = snap["decision"]
        stance = decision["stance"]
        volatility = decision["volatility"]
        meta_conf = decision["meta_conf"]
        cog_load = decision["cog_load"]
        env = decision["env"]
        opportunity = decision["opportunity"]
        risk = decision["risk"]

        self.mode_var.set(stance)
        self.volatility_var.set(f"{volatility:.2f}")
        self.trust_var.set(f"{meta_conf:.2f}")
        self.cog_load_var.set(f"{cog_load:.2f}")

        self.env_var.set(env)
        self.opportunity_var.set(f"{opportunity:.2f}")
        self.risk_var.set(f"{risk:.2f}")

        self.judgment_conf_var.set(f"{snap['judgment_confidence']:.2f}")
        self.good_bad_var.set(f"{snap['good_outcomes']} / {snap['bad_outcomes']}")

        cpu_forecast = snap["cpu_forecast"]
        self.cpu1_var.set(f"{cpu_forecast.get(1, 0.0):.1f}%")
        self.cpu5_var.set(f"{cpu_forecast.get(5, 0.0):.1f}%")
        self.cpu30_var.set(f"{cpu_forecast.get(30, 0.0):.1f}%")
        self.cpu120_var.set(f"{cpu_forecast.get(120, 0.0):.1f}%")

        # crude anomaly risk = volatility * (1 - meta_conf)
        anomaly_risk = volatility * (1.0 - meta_conf)
        self.anomaly_risk_var.set(f"{anomaly_risk:.2f}")

        self.collective_health_var.set(f"{snap['collective_health']:.2f}")
        self.hive_density_var.set(f"{snap['hive_density']:.2f}")
        self.node_agreement_var.set(f"{snap['node_agreement']:.2f}")

    def _append_dialogue(self, text):
        self.dialogue.insert("end", text)
        self.dialogue.see("end")


# ========= ENTRY POINT =========

def main():
    core = OrganismCore(sample_interval=1.0)
    core.start()

    root = tk.Tk()
    gui = MissionControlGUI(root, core)

    def on_close():
        core.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

