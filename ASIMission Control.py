import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import datetime
import queue

# Try to use real system metrics, else fallback to simulated
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


# ─────────────────────────────────────────────
# Action bus and system tuner (intent → action)
# ─────────────────────────────────────────────

class ActionBus:
    def __init__(self):
        self.q = queue.Queue()

    def publish(self, action_type, payload=None):
        self.q.put({"type": action_type, "payload": payload or {}, "ts": time.time()})

    def try_get(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


class SystemTuner(threading.Thread):
    """
    Consumes actions and prints them.
    In a real system, this would adjust OS settings, backup behavior, etc.
    """
    def __init__(self, action_bus: ActionBus):
        super().__init__(daemon=True)
        self.bus = action_bus
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            action = self.bus.try_get()
            if action:
                self._handle_action(action)
            time.sleep(0.1)

    def _handle_action(self, action):
        print(f"[SystemTuner] {action['type']}  payload={action['payload']}")


# ─────────────────────────────────────────────
# System & Web collectors
# ─────────────────────────────────────────────

class SystemSnapshotCollector(threading.Thread):
    """
    Collects system state periodically: CPU, memory, disk, net.
    Uses psutil if available, otherwise simulates values.
    """
    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._lock = threading.Lock()
        self._running = True
        self._snapshot = {}

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return self._running

    def run(self):
        while self.is_running():
            snap = self._collect_once()
            with self._lock:
                self._snapshot = snap
            time.sleep(self.interval)

    def _collect_once(self):
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            net = psutil.net_io_counters()
            net_sent = net.bytes_sent
            net_recv = net.bytes_recv
        else:
            cpu = random.uniform(5, 80)
            mem = random.uniform(20, 90)
            disk = random.uniform(30, 95)
            net_sent = random.randint(10_000, 5_000_000)
            net_recv = random.randint(10_000, 5_000_000)

        return {
            "cpu_percent": cpu,
            "mem_percent": mem,
            "disk_percent": disk,
            "net_bytes_sent": net_sent,
            "net_bytes_recv": net_recv,
        }

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)


class WebSnapshotCollector(threading.Thread):
    """
    Simulated 'web' collector.
    You can swap fetch_fn later to hit real APIs.
    """
    def __init__(self, interval=10.0, fetch_fn=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.fetch_fn = fetch_fn or self._default_fetch
        self._lock = threading.Lock()
        self._running = True
        self._snapshot = {}

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return self._running

    def run(self):
        while self.is_running():
            try:
                snap = self.fetch_fn()
            except Exception as e:
                snap = {"error": str(e)}
            with self._lock:
                self._snapshot = snap
            time.sleep(self.interval)

    def _default_fetch(self):
        # Simulated global threat & noise
        return {
            "global_threat_level": random.uniform(0.0, 1.0),  # 0..1
            "global_traffic_intensity": random.uniform(0.0, 1.0),
            "event_flags": {
                "security_incident": random.random() < 0.05,
                "major_outage": random.random() < 0.03,
            },
        }

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)


# ─────────────────────────────────────────────
# BrainCore – wired to system + web
# ─────────────────────────────────────────────

class BrainCore(threading.Thread):
    """
    Real-time 'brain' that:
    - reads system + web collectors
    - synthesizes anomaly/drive/hive risk
    - decides mission, environment, plan
    - estimates early warning, horizon, focus
    - simulates judgment confidence based on "outcomes"
    """

    def __init__(self, system_collector: SystemSnapshotCollector,
                 web_collector: WebSnapshotCollector,
                 action_bus: ActionBus,
                 interval=2.0):
        super().__init__(daemon=True)
        self.system = system_collector
        self.web = web_collector
        self.actions = action_bus
        self.interval = interval

        self._lock = threading.Lock()
        self._running = True

        # Core exposed state (matches GUI expectations)
        self.mode = "stability"
        self.mode_override = "AUTO"

        self.mission = "STABILITY"
        self.mission_override = "AUTO"

        self.risk_tolerance = 0.5

        self.collective_health = 80
        self.health_trend = "STABLE"

        self.anomaly_risk = "LOW"
        self.drive_risk = "LOW"
        self.hive_risk = "LOW"

        self.volatility = 0.2
        self.trust = 0.7

        self.judgment_confidence = 0.6
        self.judgment_samples = 100
        self.judgment_good = 70
        self.judgment_bad = 30

        self.opportunity_score = 0.4
        self.risk_score = 0.3
        self.environment = "CALM"
        self.plan = "Maintain balance"
        self.anticipation = "None"

        self.early_warning = 0.2
        self.stability_horizon = "MEDIUM"
        self.recommended_focus = "STABILITY"

        self._last_health = self.collective_health

    # ── external control (GUI) ──────────────────────

    def set_mode_override(self, mode):
        with self._lock:
            self.mode_override = mode
        self.actions.publish("SET_MODE_OVERRIDE", {"mode": mode})

    def set_mission_override(self, mission):
        with self._lock:
            self.mission_override = mission
        self.actions.publish("SET_MISSION_OVERRIDE", {"mission": mission})

    def set_risk_tolerance(self, value):
        v = max(0.0, min(1.0, float(value)))
        with self._lock:
            self.risk_tolerance = v
        self.actions.publish("SET_RISK_TOLERANCE", {"value": v})

    # ── thread control ──────────────────────────────

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return self._running

    # ── main loop ───────────────────────────────────

    def run(self):
        while self.is_running():
            self._tick()
            time.sleep(self.interval)

    def _tick(self):
        sys_snap = self.system.snapshot() or {}
        web_snap = self.web.snapshot() or {}

        cpu = float(sys_snap.get("cpu_percent", 0.0))
        mem = float(sys_snap.get("mem_percent", 0.0))
        disk = float(sys_snap.get("disk_percent", 0.0))
        net_sent = float(sys_snap.get("net_bytes_sent", 0.0))
        net_recv = float(sys_snap.get("net_bytes_recv", 0.0))

        g_threat = float(web_snap.get("global_threat_level", 0.0))
        g_traffic = float(web_snap.get("global_traffic_intensity", 0.0))
        flags = web_snap.get("event_flags", {}) or {}
        sec_incident = bool(flags.get("security_incident", False))
        major_outage = bool(flags.get("major_outage", False))

        # Normalize system signals into 0..1
        cpu_n = min(1.0, cpu / 100.0)
        mem_n = min(1.0, mem / 100.0)
        disk_n = min(1.0, disk / 100.0)
        net_n = min(1.0, (net_sent + net_recv) / (10_000_000.0 + 1.0))

        # System stress ~ anomaly risk proxy
        system_stress = 0.4 * cpu_n + 0.4 * mem_n + 0.2 * net_n
        # "Drive" risk ~ disk usage + global traffic + outages
        drive_stress = 0.5 * disk_n + 0.3 * g_traffic + (0.2 if major_outage else 0.0)
        hive_pressure = 0.5 * g_threat + 0.5 * g_traffic + (0.2 if sec_incident else 0.0)

        system_stress = max(0.0, min(1.0, system_stress))
        drive_stress = max(0.0, min(1.0, drive_stress))
        hive_pressure = max(0.0, min(1.0, hive_pressure))

        with self._lock:
            # MODE
            if self.mode_override != "AUTO":
                self.mode = self.mode_override.lower()
            else:
                if system_stress > 0.7 or drive_stress > 0.7 or hive_pressure > 0.7:
                    self.mode = "reflex"
                elif system_stress < 0.3 and drive_stress < 0.3 and hive_pressure < 0.3:
                    self.mode = "stability"
                else:
                    self.mode = random.choice(["stability", "reflex", "exploration"])

            # VOLATILITY from combined stress
            combined = (system_stress + drive_stress + hive_pressure) / 3.0
            noise = random.uniform(-0.05, 0.05)
            self.volatility = max(0.0, min(1.0, 0.7 * self.volatility + 0.3 * (combined + noise)))

            # Anomaly risk from system_stress + global_threat
            anomaly_signal = 0.6 * system_stress + 0.4 * g_threat
            if anomaly_signal > 0.7:
                self.anomaly_risk = "HIGH"
            elif anomaly_signal > 0.4:
                self.anomaly_risk = "MEDIUM"
            else:
                self.anomaly_risk = "LOW"

            # Drive risk from drive_stress
            if drive_stress > 0.7:
                self.drive_risk = "HIGH"
            elif drive_stress > 0.4:
                self.drive_risk = "MEDIUM"
            else:
                self.drive_risk = "LOW"

            # Hive risk from hive_pressure
            if hive_pressure > 0.7:
                self.hive_risk = "HIGH"
            elif hive_pressure > 0.4:
                self.hive_risk = "MEDIUM"
            else:
                self.hive_risk = "LOW"

            # TRUST: decreases with volatility + threat
            risk_factor = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}[self.anomaly_risk]
            target_trust = max(0.1, 1.0 - (self.volatility * 0.5 + risk_factor * 0.5))
            self.trust = max(0.0, min(1.0, 0.8 * self.trust + 0.2 * target_trust))

            # JUDGMENT EVOLUTION (simulated outcomes)
            self.judgment_samples += random.randint(1, 6)
            good_add = random.randint(0, 4)
            bad_add = random.randint(0, 3)
            self.judgment_good += good_add
            self.judgment_bad += bad_add
            total = max(1, self.judgment_good + self.judgment_bad)
            good_ratio = self.judgment_good / total

            # More good outcomes when system_stress is controlled
            target_conf = min(
                1.0,
                0.3 + good_ratio * 0.5 + (self.judgment_samples / 600.0) * 0.2
            )
            self.judgment_confidence = max(
                0.0, min(1.0, 0.8 * self.judgment_confidence + 0.2 * target_conf)
            )

            # Situational scores (opportunity vs risk)
            self.risk_score = max(0.0, min(1.0, (system_stress + drive_stress + hive_pressure) / 3.0))
            # More learning opportunity when risk low, web traffic stable
            self.opportunity_score = max(
                0.0,
                min(1.0, (1.0 - self.risk_score) * 0.6 + (1.0 - abs(g_traffic - 0.5)) * 0.4)
            )

            # Environment by risk
            if self.risk_score > 0.7:
                self.environment = "DANGER"
            elif self.risk_score > 0.4:
                self.environment = "TENSE"
            else:
                self.environment = "CALM"

            # MISSION
            if self.mission_override != "AUTO":
                self.mission = self.mission_override
            else:
                if self.risk_score > 0.7:
                    self.mission = "PROTECT"
                elif self.opportunity_score > 0.6 and self.environment == "CALM":
                    self.mission = "LEARN"
                elif self.environment == "TENSE":
                    self.mission = random.choice(["STABILITY", "PROTECT"])
                else:
                    self.mission = random.choice(["STABILITY", "OPTIMIZE"])

            # PLAN
            if self.mission == "PROTECT":
                self.plan = "Shield systems and reduce exposure"
            elif self.mission == "LEARN":
                self.plan = "Exploit windows for learning"
            elif self.mission == "OPTIMIZE":
                self.plan = "Tune performance under constraints"
            else:
                self.plan = "Maintain balanced posture"

            # ANTICIPATION
            if self.environment == "DANGER":
                self.anticipation = "Prepare for escalation"
            elif self.opportunity_score > 0.6:
                self.anticipation = "Prepare for learning surge"
            elif sec_incident:
                self.anticipation = "Anticipate security ripple effects"
            else:
                self.anticipation = "None"

            # COLLECTIVE HEALTH (0..100), inverted risk + volatility
            risk_combo = (system_stress + drive_stress + hive_pressure) / 3.0
            health_raw = 100 - int(risk_combo * 70 + self.volatility * 30)
            health_raw += int((self.risk_tolerance - 0.5) * 10)
            health_raw = max(0, min(100, health_raw))

            self.health_trend = "STABLE"
            if health_raw > self._last_health + 2:
                self.health_trend = "IMPROVING"
            elif health_raw < self._last_health - 2:
                self.health_trend = "DECLINING"
            self._last_health = self.collective_health = health_raw

            # META – early warning & horizon & focus
            warning = 0.0
            if self.health_trend == "DECLINING":
                warning += 0.4
            if self.anomaly_risk == "HIGH":
                warning += 0.3
            if self.drive_risk == "HIGH":
                warning += 0.2
            if sec_incident or major_outage:
                warning += 0.2
            self.early_warning = max(0.0, min(1.0, warning))

            if self.collective_health > 80:
                self.stability_horizon = "LONG"
            elif self.collective_health > 50:
                self.stability_horizon = "MEDIUM"
            else:
                self.stability_horizon = "SHORT"

            if self.anomaly_risk == "HIGH" or self.drive_risk == "HIGH":
                self.recommended_focus = "PROTECT"
            elif self.mission == "LEARN":
                self.recommended_focus = "LEARN"
            elif self.mission == "OPTIMIZE":
                self.recommended_focus = "OPTIMIZE"
            else:
                self.recommended_focus = "STABILITY"

            # Publish some actions
            if self.early_warning > 0.7:
                self.actions.publish("RAISE_ALERT", {
                    "reason": "High early warning",
                    "anomaly_risk": self.anomaly_risk,
                    "drive_risk": self.drive_risk,
                    "health": self.collective_health,
                    "environment": self.environment,
                })

            if self.recommended_focus == "PROTECT" and self.mode != "reflex":
                self.actions.publish("SUGGEST_REFLEX_MODE", {
                    "current_mode": self.mode,
                    "environment": self.environment,
                })

    # ── snapshot for GUI ────────────────────────────

    def snapshot(self):
        with self._lock:
            return {
                "mode": self.mode,
                "mode_override": self.mode_override,
                "mission": self.mission,
                "mission_override": self.mission_override,
                "risk_tolerance": self.risk_tolerance,
                "collective_health": self.collective_health,
                "health_trend": self.health_trend,
                "anomaly_risk": self.anomaly_risk,
                "drive_risk": self.drive_risk,
                "hive_risk": self.hive_risk,
                "volatility": self.volatility,
                "trust": self.trust,
                "judgment_confidence": self.judgment_confidence,
                "judgment_samples": self.judgment_samples,
                "judgment_good": self.judgment_good,
                "judgment_bad": self.judgment_bad,
                "environment": self.environment,
                "opportunity_score": self.opportunity_score,
                "risk_score": self.risk_score,
                "plan": self.plan,
                "anticipation": self.anticipation,
                "early_warning": self.early_warning,
                "stability_horizon": self.stability_horizon,
                "recommended_focus": self.recommended_focus,
            }


# ─────────────────────────────────────────────
# Mission Control GUI
# ─────────────────────────────────────────────

class MissionControlGUI:
    def __init__(self, root, brain: BrainCore):
        self.root = root
        self.brain = brain

        self.root.title("ASI Mission Control – Hybrid Brain Console")
        self.root.geometry("1200x720")
        self.root.configure(bg="#05060a")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#05060a")
        style.configure("TLabel", background="#05060a", foreground="#c0c8ff")
        style.configure("Header.TLabel", font=("Consolas", 14, "bold"), foreground="#88c0ff")
        style.configure("Card.TFrame", background="#10121a", relief="ridge", borderwidth=1)
        style.configure("CardTitle.TLabel", background="#10121a", foreground="#a5b4ff", font=("Consolas", 11, "bold"))

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=4)

        self.lbl_title = ttk.Label(
            top,
            text="ASI HYBRID BRAIN – MISSION CONTROL",
            style="Header.TLabel",
        )
        self.lbl_title.pack(side="left")

        self.lbl_clock = ttk.Label(top, text="", style="TLabel")
        self.lbl_clock.pack(side="right")

        # Main grid
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        main.columnconfigure(0, weight=1, uniform="col")
        main.columnconfigure(1, weight=1, uniform="col")
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self._build_hybrid_brain(main)
        self._build_judgment(main)
        self._build_situational(main)
        self._build_predictive(main)

    def _build_card(self, parent, title, row, col):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(col, weight=1)
        lbl = ttk.Label(frame, text=title, style="CardTitle.TLabel")
        lbl.pack(anchor="w", padx=6, pady=4)
        sep = ttk.Separator(frame, orient="horizontal")
        sep.pack(fill="x", padx=4, pady=2)
        return frame

    def _build_hybrid_brain(self, parent):
        frame = self._build_card(parent, "HYBRID BRAIN CORE", 0, 0)

        self.lbl_mode = ttk.Label(frame, text="Mode: ?")
        self.lbl_mode.pack(anchor="w", padx=8, pady=(6, 2))

        frm_mode = ttk.Frame(frame, style="Card.TFrame")
        frm_mode.pack(fill="x", padx=8, pady=2)

        ttk.Label(frm_mode, text="Mode override:", width=14).pack(side="left")
        self.mode_var = tk.StringVar(value="AUTO")
        for txt in ["AUTO", "stability", "reflex", "exploration"]:
            ttk.Radiobutton(
                frm_mode,
                text=txt,
                value=txt,
                variable=self.mode_var,
                command=self._on_mode_change
            ).pack(side="left", padx=2)

        frm_vol = ttk.Frame(frame, style="Card.TFrame")
        frm_vol.pack(fill="x", padx=8, pady=4)

        self.volatility_var = tk.DoubleVar(value=0.0)
        self.trust_var = tk.DoubleVar(value=0.0)

        ttk.Label(frm_vol, text="Volatility:").grid(row=0, column=0, sticky="w")
        ttk.Progressbar(frm_vol, maximum=1.0, variable=self.volatility_var).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(frm_vol, text="Trust:").grid(row=1, column=0, sticky="w")
        ttk.Progressbar(frm_vol, maximum=1.0, variable=self.trust_var).grid(row=1, column=1, sticky="we", padx=4)
        frm_vol.columnconfigure(1, weight=1)

        frm_risk_tol = ttk.Frame(frame, style="Card.TFrame")
        frm_risk_tol.pack(fill="x", padx=8, pady=4)

        ttk.Label(frm_risk_tol, text="Risk tolerance:").grid(row=0, column=0, sticky="w")
        self.risk_tol_var = tk.DoubleVar(value=0.5)
        scale = ttk.Scale(
            frm_risk_tol,
            from_=0.0,
            to=1.0,
            variable=self.risk_tol_var,
            command=self._on_risk_tolerance_change
        )
        scale.grid(row=0, column=1, sticky="we", padx=4)
        frm_risk_tol.columnconfigure(1, weight=1)

        self.lbl_risk_tol = ttk.Label(frm_risk_tol, text="0.50")
        self.lbl_risk_tol.grid(row=0, column=2, sticky="e", padx=4)

        frm_health = ttk.Frame(frame, style="Card.TFrame")
        frm_health.pack(fill="x", padx=8, pady=4)

        self.health_var = tk.DoubleVar(value=80)
        ttk.Label(frm_health, text="Collective health:").grid(row=0, column=0, sticky="w")
        ttk.Progressbar(frm_health, maximum=100, variable=self.health_var).grid(row=0, column=1, sticky="we", padx=4)
        self.lbl_health_trend = ttk.Label(frm_health, text="Trend: ?")
        self.lbl_health_trend.grid(row=1, column=0, columnspan=2, sticky="w")
        frm_health.columnconfigure(1, weight=1)

    def _build_judgment(self, parent):
        frame = self._build_card(parent, "JUDGMENT & CONFIDENCE", 1, 0)

        self.lbl_j_conf = ttk.Label(frame, text="Confidence: ?")
        self.lbl_j_conf.pack(anchor="w", padx=8, pady=(6, 2))

        self.j_conf_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(frame, maximum=1.0, variable=self.j_conf_var).pack(fill="x", padx=8, pady=2)

        self.lbl_j_samples = ttk.Label(frame, text="Samples: ?")
        self.lbl_j_samples.pack(anchor="w", padx=8, pady=2)

        frm_outcomes = ttk.Frame(frame, style="Card.TFrame")
        frm_outcomes.pack(fill="x", padx=8, pady=4)

        self.lbl_j_good = ttk.Label(frm_outcomes, text="Good: ?")
        self.lbl_j_good.grid(row=0, column=0, sticky="w")
        self.lbl_j_bad = ttk.Label(frm_outcomes, text="Bad: ?")
        self.lbl_j_bad.grid(row=0, column=1, sticky="w")

        self.j_good_ratio_var = tk.DoubleVar(value=0.0)
        ttk.Label(frame, text="Good outcome ratio:").pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Progressbar(frame, maximum=1.0, variable=self.j_good_ratio_var).pack(fill="x", padx=8, pady=2)

    def _build_situational(self, parent):
        frame = self._build_card(parent, "SITUATIONAL AWARENESS CORTEX", 0, 1)

        self.lbl_mission = ttk.Label(frame, text="Mission: ?")
        self.lbl_mission.pack(anchor="w", padx=8, pady=(6, 2))

        frm_mission = ttk.Frame(frame, style="Card.TFrame")
        frm_mission.pack(fill="x", padx=8, pady=2)

        ttk.Label(frm_mission, text="Mission override:", width=14).pack(side="left")
        self.mission_var = tk.StringVar(value="AUTO")
        for txt in ["AUTO", "PROTECT", "STABILITY", "LEARN", "OPTIMIZE"]:
            ttk.Radiobutton(
                frm_mission,
                text=txt,
                value=txt,
                variable=self.mission_var,
                command=self._on_mission_change
            ).pack(side="left", padx=2)

        self.lbl_env = ttk.Label(frame, text="Environment: ?")
        self.lbl_env.pack(anchor="w", padx=8, pady=(6, 2))

        frm_scores = ttk.Frame(frame, style="Card.TFrame")
        frm_scores.pack(fill="x", padx=8, pady=4)

        self.opp_var = tk.DoubleVar(value=0.0)
        self.risk_var = tk.DoubleVar(value=0.0)

        ttk.Label(frm_scores, text="Opportunity:").grid(row=0, column=0, sticky="w")
        ttk.Progressbar(frm_scores, maximum=1.0, variable=self.opp_var).grid(row=0, column=1, sticky="we", padx=4)

        ttk.Label(frm_scores, text="Risk:").grid(row=1, column=0, sticky="w")
        ttk.Progressbar(frm_scores, maximum=1.0, variable=self.risk_var).grid(row=1, column=1, sticky="we", padx=4)

        frm_scores.columnconfigure(1, weight=1)

        self.lbl_plan = ttk.Label(frame, text="Plan: ?")
        self.lbl_plan.pack(anchor="w", padx=8, pady=2)

        self.lbl_anticipation = ttk.Label(frame, text="Anticipation: ?")
        self.lbl_anticipation.pack(anchor="w", padx=8, pady=2)

    def _build_predictive(self, parent):
        frame = self._build_card(parent, "PREDICTIVE INTELLIGENCE & COLLECTIVE HEALTH", 1, 1)

        self.lbl_anom = ttk.Label(frame, text="Anomaly risk: ?")
        self.lbl_anom.pack(anchor="w", padx=8, pady=(6, 2))

        self.lbl_drive = ttk.Label(frame, text="Drive risk: ?")
        self.lbl_drive.pack(anchor="w", padx=8, pady=2)

        self.lbl_hive = ttk.Label(frame, text="Hive risk: ?")
        self.lbl_hive.pack(anchor="w", padx=8, pady=2)

        self.ew_var = tk.DoubleVar(value=0.0)
        ttk.Label(frame, text="Early warning:").pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Progressbar(frame, maximum=1.0, variable=self.ew_var).pack(fill="x", padx=8, pady=2)

        self.lbl_horizon = ttk.Label(frame, text="Stability horizon: ?")
        self.lbl_horizon.pack(anchor="w", padx=8, pady=2)

        self.lbl_focus = ttk.Label(frame, text="Recommended focus: ?")
        self.lbl_focus.pack(anchor="w", padx=8, pady=2)

    # ── Callbacks ────────────────────────────────────

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.brain.set_mode_override(mode)

    def _on_mission_change(self):
        mission = self.mission_var.get()
        self.brain.set_mission_override(mission)

    def _on_risk_tolerance_change(self, _value):
        val = self.risk_tol_var.get()
        self.lbl_risk_tol.config(text=f"{val:.2f}")
        self.brain.set_risk_tolerance(val)

    # ── Periodic GUI update ─────────────────────────

    def _schedule_update(self):
        self._update()
        self.root.after(500, self._schedule_update)

    def _update(self):
        snap = self.brain.snapshot()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lbl_clock.config(text=now)

        # Hybrid brain
        self.lbl_mode.config(
            text=f"Mode: {snap['mode']} (override: {snap['mode_override']})"
        )
        self.volatility_var.set(snap["volatility"])
        self.trust_var.set(snap["trust"])
        self.health_var.set(snap["collective_health"])
        self.lbl_health_trend.config(
            text=f"Trend: {snap['health_trend']}   Health: {snap['collective_health']}"
        )

        self.risk_tol_var.set(snap["risk_tolerance"])
        self.lbl_risk_tol.config(text=f"{snap['risk_tolerance']:.2f}")

        # Judgment
        self.lbl_j_conf.config(
            text=f"Confidence: {snap['judgment_confidence']:.2f}"
        )
        self.j_conf_var.set(snap["judgment_confidence"])
        self.lbl_j_samples.config(
            text=f"Samples: {snap['judgment_samples']}"
        )
        self.lbl_j_good.config(text=f"Good: {snap['judgment_good']}")
        self.lbl_j_bad.config(text=f"Bad: {snap['judgment_bad']}")
        total = max(1, snap["judgment_good"] + snap["judgment_bad"])
        self.j_good_ratio_var.set(snap["judgment_good"] / total)

        # Situational
        self.lbl_mission.config(
            text=f"Mission: {snap['mission']} (override: {snap['mission_override']})"
        )
        self.lbl_env.config(text=f"Environment: {snap['environment']}")
        self.opp_var.set(snap["opportunity_score"])
        self.risk_var.set(snap["risk_score"])
        self.lbl_plan.config(text=f"Plan: {snap['plan']}")
        self.lbl_anticipation.config(text=f"Anticipation: {snap['anticipation']}")

        # Predictive / Collective
        self.lbl_anom.config(text=f"Anomaly risk: {snap['anomaly_risk']}")
        self.lbl_drive.config(text=f"Drive risk: {snap['drive_risk']}")
        self.lbl_hive.config(text=f"Hive risk: {snap['hive_risk']}")
        self.ew_var.set(snap["early_warning"])
        self.lbl_horizon.config(text=f"Stability horizon: {snap['stability_horizon']}")
        self.lbl_focus.config(text=f"Recommended focus: {snap['recommended_focus']}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    actions = ActionBus()
    tuner = SystemTuner(actions)
    tuner.start()

    sys_col = SystemSnapshotCollector(interval=2.0)
    web_col = WebSnapshotCollector(interval=5.0)
    sys_col.start()
    web_col.start()

    brain = BrainCore(sys_col, web_col, actions, interval=2.0)
    brain.start()

    root = tk.Tk()
    gui = MissionControlGUI(root, brain)

    try:
        root.mainloop()
    finally:
        brain.stop()
        tuner.stop()
        sys_col.stop()
        web_col.stop()


if __name__ == "__main__":
    main()

