#!/usr/bin/env python3
# ============================================================
# GUARDIAN NERVE CENTER — HybridBrain + Organs + Zero Trust
# PyQt5 MagicBox / Military Console Edition (single file)
# ============================================================

import os, sys, platform, ctypes, json, time, threading, random, math, datetime
from collections import deque

# ---------------------- Auto-loader -------------------------
def ensure_packages():
    import subprocess
    pkgs = ["psutil", "pyttsx3", "PyQt5", "pynvml"]
    for p in pkgs:
        try:
            __import__(p if p != "PyQt5" else "PyQt5.QtWidgets")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

ensure_packages()

import psutil
import pyttsx3
from PyQt5 import QtWidgets, QtCore, QtGui
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ============================================================
# Elevation (Windows)
# ============================================================
def ensure_admin():
    if platform.system() != "Windows":
        return
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{a}"' for a in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Guardian] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# ============================================================
# ReplicaNPU — predictive core
# ============================================================
class ReplicaNPU:
    def __init__(self, cores=8, frequency_ghz=1.2, memory_size=16,
                 plasticity_decay=0.0005, integrity_threshold=0.4):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.cycles = 0
        self.energy = 0.0
        self.memory = deque(maxlen=memory_size)
        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay
        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False
        self.heads = {}
        self.symbolic_bias = {}

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        assert len(v1) == len(v2)
        chunk = max(1, math.ceil(len(v1) / self.cores))
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    def add_head(self, name, input_dim, lr=0.01, risk=1.0):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "history": deque(maxlen=32),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"] + self._symbolic_modulation(name)
        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            error = target - pred
            weighted_error = error * head["risk"] * self.plasticity * self.model_integrity
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {k: round(self.confidence(k), 3) for k in self.heads},
        }

# ============================================================
# Organs — real-time sensors
# ============================================================
class BaseOrgan:
    def __init__(self, name):
        self.name = name
        self.health = 1.0
        self.risk = 0.0

    def update(self):
        pass

    def micro_recovery(self):
        self.risk = max(0.0, self.risk - 0.01)
        self.health = min(1.0, self.health + 0.005)

class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRAM")
        self.usage = 0.0

    def update(self):
        mem = psutil.virtual_memory()
        self.usage = mem.percent / 100.0
        self.risk = self.usage
        self.health = 1.0 - self.usage

class ThermalOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Thermal")
        self.temp = 0.0

    def update(self):
        try:
            temps = psutil.sensors_temperatures()
            all_t = []
            for arr in temps.values():
                for t in arr:
                    if t.current:
                        all_t.append(t.current)
            self.temp = max(all_t) if all_t else 40.0
        except Exception:
            self.temp = 40.0
        self.risk = min(1.0, max(0.0, (self.temp - 50.0) / 40.0))
        self.health = 1.0 - self.risk

class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Disk")
        self.io = 0.0
        self._last = psutil.disk_io_counters()

    def update(self):
        now = psutil.disk_io_counters()
        delta = (now.read_bytes + now.write_bytes) - (self._last.read_bytes + self._last.write_bytes)
        self._last = now
        self.io = delta / (1024 * 1024)
        self.risk = min(1.0, self.io / 200.0)
        self.health = 1.0 - self.risk

class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Network")
        self.connections = 0

    def update(self):
        conns = psutil.net_connections(kind="inet")
        self.connections = len([c for c in conns if c.status == "ESTABLISHED"])
        self.risk = min(1.0, self.connections / 200.0)
        self.health = 1.0 - self.risk

class GPUCacheOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("GPUCache")
        self.util = 0.0

    def update(self):
        if not NVML_AVAILABLE:
            self.util = 0.0
            self.health = 1.0
            self.risk = 0.0
            return
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            self.util = util.gpu
        except Exception:
            self.util = 0.0
        self.risk = min(1.0, self.util / 100.0)
        self.health = 1.0 - self.risk

class VRAMOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("VRAM")
        self.used = 0.0

    def update(self):
        if not NVML_AVAILABLE:
            self.used = 0.0
            self.health = 1.0
            self.risk = 0.0
            return
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.used = mem.used / mem.total
        except Exception:
            self.used = 0.0
        self.risk = self.used
        self.health = 1.0 - self.used

class AICoachOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("AICoach")
        self.coaching_level = 0.5

    def update(self):
        self.risk = 1.0 - self.coaching_level
        self.health = self.coaching_level

class SwarmNodeOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("SwarmNode")
        self.agreement = 0.8

    def update(self):
        self.risk = 1.0 - self.agreement
        self.health = self.agreement

# ============================================================
# Zero Trust + Data TTL
# ============================================================
class ZeroTrustGuardian:
    def __init__(self):
        self.personal_data_store = []
        self.fake_telemetry = []
        self.mac_ip_store = []
        self.lock = threading.Lock()

    def _now(self):
        return datetime.datetime.utcnow()

    def store_personal(self, payload):
        with self.lock:
            self.personal_data_store.append(
                {"data": payload, "expires": self._now() + datetime.timedelta(days=1)}
            )

    def store_mac_ip(self, payload):
        with self.lock:
            self.mac_ip_store.append(
                {"data": payload, "expires": self._now() + datetime.timedelta(seconds=30)}
            )

    def emit_fake_telemetry(self, payload):
        with self.lock:
            self.fake_telemetry.append(
                {"data": payload, "expires": self._now() + datetime.timedelta(seconds=30)}
            )

    def purge_expired(self):
        with self.lock:
            now = self._now()
            self.personal_data_store = [x for x in self.personal_data_store if x["expires"] > now]
            self.fake_telemetry = [x for x in self.fake_telemetry if x["expires"] > now]
            self.mac_ip_store = [x for x in self.mac_ip_store if x["expires"] > now]

    def zero_trust_check(self, identity):
        allowed = {"system_core", "authorized_user"}
        if identity not in allowed:
            raise PermissionError(f"Zero Trust: {identity} blocked")

# ============================================================
# BorgMesh + MirrorDefense (minimal stubs)
# ============================================================
class BorgMesh:
    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def discover(self, url, risk):
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node

    def enforce(self, url):
        if url in self.nodes:
            self.nodes[url]["state"] = "enforced"

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        return {"total": total, "discovered": discovered, "enforced": enforced}

class MirrorDefense:
    def __init__(self):
        self.last_status = "idle"

    def evaluate(self, analysis):
        status = analysis.get("status", "unknown")
        self.last_status = status

# ============================================================
# HybridBrain — tri-stance, meta-states, best-guess
# ============================================================
class HybridBrain:
    META_STATES = ["Hyper-Flow", "Sentinel", "Recovery-Flow", "Deep-Dream"]
    STANCES = ["Conservative", "Balanced", "Beast"]

    def __init__(self):
        self.npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
        self.npu.add_head("short", 3, lr=0.05, risk=1.5)
        self.npu.add_head("medium", 3, lr=0.03, risk=1.0)
        self.npu.add_head("long", 3, lr=0.02, risk=0.7)
        self.meta_state = "Sentinel"
        self.stance = "Balanced"
        self.last_predictions = {
            "short": 0.0,
            "medium": 0.0,
            "long": 0.0,
            "baseline": 0.0,
            "best_guess": 0.0,
            "meta_conf": 0.5,
        }
        self.pattern_memory = []
        self.last_reasoning = deque(maxlen=12)
        self.baseline = 0.3

    def _regime(self, risk):
        if risk < 0.3:
            return "stable"
        elif risk < 0.6:
            return "rising"
        else:
            return "chaotic"

    def _meta_confidence(self, preds):
        vals = [preds["short"], preds["medium"], preds["long"]]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return max(0.0, min(1.0, 1.0 - var))

    def _best_guess(self, preds, meta_conf, regime):
        weights = {
            "stable": (0.2, 0.4, 0.4),
            "rising": (0.3, 0.5, 0.2),
            "chaotic": (0.5, 0.4, 0.1),
        }[regime]
        bg = (
            preds["short"] * weights[0]
            + preds["medium"] * weights[1]
            + preds["long"] * weights[2]
        )
        return (bg * meta_conf) + (self.baseline * (1.0 - meta_conf))

    def _update_stance(self, risk, organs):
        ram = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        therm = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        mem_high = ram.usage > 0.9 if ram else False
        temp_high = therm.temp > 80 if therm else False
        if mem_high or temp_high or risk > 0.7:
            self.stance = "Conservative"
        elif risk < 0.4:
            self.stance = "Beast"
        else:
            self.stance = "Balanced"

    def _update_meta_state(self, risk):
        if risk > 0.8:
            self.meta_state = "Sentinel"
        elif risk > 0.6:
            self.meta_state = "Recovery-Flow"
        elif risk < 0.3:
            self.meta_state = "Hyper-Flow"
        else:
            self.meta_state = "Deep-Dream"

    def update(self, organs):
        ram = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        therm = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        net = next((o for o in organs if isinstance(o, NetworkWatcherOrgan)), None)
        x = [
            ram.usage if ram else 0.3,
            (therm.temp / 100.0) if therm else 0.4,
            (net.connections / 200.0) if net else 0.1,
        ]
        preds = self.npu.predict(x)
        meta_conf = self._meta_confidence(preds)
        regime = self._regime(preds["medium"])
        best_guess = self._best_guess(preds, meta_conf, regime)
        self.baseline = 0.9 * self.baseline + 0.1 * preds["medium"]
        self._update_stance(best_guess, organs)
        self._update_meta_state(best_guess)
        self.last_predictions = {
            "short": preds["short"],
            "medium": preds["medium"],
            "long": preds["long"],
            "baseline": self.baseline,
            "best_guess": best_guess,
            "meta_conf": meta_conf,
        }
        self.pattern_memory.append(
            {"x": x, "preds": preds, "regime": regime, "best": best_guess}
        )
        self.last_reasoning.append(
            f"Regime={regime}, Stance={self.stance}, MetaState={self.meta_state}, Best={best_guess:.2f}"
        )
        self.npu.micro_recovery()
        self.npu.check_integrity(external_integrity=1.0)

# ============================================================
# Reboot Memory (SMB / local)
# ============================================================
class RebootMemoryManager:
    def __init__(self):
        self.path = ""
        self.autoload = False

    def save(self, brain: HybridBrain, organs, path):
        state = {
            "brain": brain.last_predictions,
            "meta_state": brain.meta_state,
            "stance": brain.stance,
            "baseline": brain.baseline,
            "organs": {o.name: {"health": o.health, "risk": o.risk} for o in organs},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, brain: HybridBrain, organs, path):
        if not os.path.isfile(path):
            return False
        with open(path, "r") as f:
            state = json.load(f)
        brain.meta_state = state.get("meta_state", brain.meta_state)
        brain.stance = state.get("stance", brain.stance)
        brain.baseline = state.get("baseline", brain.baseline)
        return True

# ============================================================
# Voice Engine (pyttsx3)
# ============================================================
def init_voice():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    # pick a stable voice if you want; here we just log
    for v in voices:
        print("[Voice]", v.id)
    return engine

# ============================================================
# GUI — MagicBox / ASI Console
# ============================================================
class GuardianUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardian Nerve Center — ASI Defense Console")
        self.resize(1200, 720)
        self.brain = HybridBrain()
        self.organs = [
            DeepRamOrgan(),
            ThermalOrgan(),
            DiskOrgan(),
            NetworkWatcherOrgan(),
            GPUCacheOrgan(),
            VRAMOrgan(),
            AICoachOrgan(),
            SwarmNodeOrgan(),
        ]
        self.zero_trust = ZeroTrustGuardian()
        self.borg = BorgMesh()
        self.mirror = MirrorDefense()
        self.reboot_mgr = RebootMemoryManager()
        self.voice = init_voice()
        self._build_ui()
        self._start_timer()

    # ---------------- UI Layout ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("⚔️ Guardian Nerve Center — Military‑Grade ASI Oversight")
        title.setStyleSheet("color:#00f7ff; font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self._build_tab_overview()
        self._build_tab_brain()
        self._build_tab_threat()
        self._build_tab_reboot()

    def _build_tab_overview(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Nerve Center")
        lay = QtWidgets.QVBoxLayout(tab)

        grid = QtWidgets.QGridLayout()
        lay.addLayout(grid)

        self.lbl_meta_state = QtWidgets.QLabel("Meta-State: Sentinel")
        self.lbl_stance = QtWidgets.QLabel("Stance: Balanced")
        self.lbl_meta_conf = QtWidgets.QLabel("Meta-Confidence: 0.00")
        self.lbl_current_risk = QtWidgets.QLabel("Current Risk: 0.00")
        self.lbl_integrity = QtWidgets.QLabel("Model Integrity: 1.00")

        for w in [self.lbl_meta_state, self.lbl_stance, self.lbl_meta_conf,
                  self.lbl_current_risk, self.lbl_integrity]:
            w.setStyleSheet("color:#ffffff; font-size:12px;")

        grid.addWidget(self.lbl_meta_state, 0, 0)
        grid.addWidget(self.lbl_stance, 0, 1)
        grid.addWidget(self.lbl_meta_conf, 1, 0)
        grid.addWidget(self.lbl_current_risk, 1, 1)
        grid.addWidget(self.lbl_integrity, 2, 0)

        self.chart = QtWidgets.QLabel()
        self.chart.setFixedHeight(180)
        self.chart.setMinimumWidth(600)
        lay.addWidget(self.chart)

        self.txt_reason = QtWidgets.QTextEdit()
        self.txt_reason.setReadOnly(True)
        self.txt_reason.setStyleSheet("background:#050711; color:#00ff99; font-family:Consolas;")
        lay.addWidget(self.txt_reason)

        cmd_bar = QtWidgets.QHBoxLayout()
        lay.addLayout(cmd_bar)

        self.btn_stabilize = QtWidgets.QPushButton("Stabilize System")
        self.btn_high_alert = QtWidgets.QPushButton("High‑Alert Mode")
        self.btn_learn = QtWidgets.QPushButton("Begin Learning Cycle")
        self.btn_optimize = QtWidgets.QPushButton("Optimize Performance")
        self.btn_purge = QtWidgets.QPushButton("Purge Anomaly Memory")

        for b in [self.btn_stabilize, self.btn_high_alert, self.btn_learn,
                  self.btn_optimize, self.btn_purge]:
            b.setStyleSheet("background:#111827; color:#e5e7eb;")
            cmd_bar.addWidget(b)

        self.btn_stabilize.clicked.connect(self._cmd_stabilize)
        self.btn_high_alert.clicked.connect(self._cmd_high_alert)
        self.btn_learn.clicked.connect(self._cmd_learn)
        self.btn_optimize.clicked.connect(self._cmd_optimize)
        self.btn_purge.clicked.connect(self._cmd_purge)

    def _build_tab_brain(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Hybrid Brain")
        lay = QtWidgets.QVBoxLayout(tab)

        self.lbl_brain_stats = QtWidgets.QLabel("Brain Stats")
        self.lbl_brain_stats.setStyleSheet("color:#ffffff; font-family:Consolas;")
        lay.addWidget(self.lbl_brain_stats)

        self.txt_dialogue = QtWidgets.QTextEdit()
        self.txt_dialogue.setReadOnly(True)
        self.txt_dialogue.setStyleSheet("background:#050711; color:#93c5fd; font-family:Consolas;")
        lay.addWidget(self.txt_dialogue)

        ask_layout = QtWidgets.QHBoxLayout()
        lay.addLayout(ask_layout)
        self.entry_question = QtWidgets.QLineEdit()
        self.entry_question.setPlaceholderText("Ask the ASI: Why did you choose this mission?")
        self.btn_ask = QtWidgets.QPushButton("Ask")
        self.btn_ask.setStyleSheet("background:#111827; color:#e5e7eb;")
        ask_layout.addWidget(self.entry_question)
        ask_layout.addWidget(self.btn_ask)
        self.btn_ask.clicked.connect(self._cmd_ask)

    def _build_tab_threat(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Threat Matrix")
        lay = QtWidgets.QVBoxLayout(tab)

        self.lbl_borg = QtWidgets.QLabel("Borg Mesh: 0 nodes")
        self.lbl_borg.setStyleSheet("color:#ffffff;")
        lay.addWidget(self.lbl_borg)

        self.lbl_mirror = QtWidgets.QLabel("MirrorDefense: idle")
        self.lbl_mirror.setStyleSheet("color:#ffffff;")
        lay.addWidget(self.lbl_mirror)

        self.txt_threat = QtWidgets.QTextEdit()
        self.txt_threat.setReadOnly(True)
        self.txt_threat.setStyleSheet("background:#050711; color:#f97316; font-family:Consolas;")
        lay.addWidget(self.txt_threat)

    def _build_tab_reboot(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Reboot Memory")
        lay = QtWidgets.QVBoxLayout(tab)

        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.entry_reboot_path = QtWidgets.QLineEdit()
        form.addRow("SMB / UNC Path:", self.entry_reboot_path)

        btn_layout = QtWidgets.QHBoxLayout()
        lay.addLayout(btn_layout)
        self.btn_save_reboot = QtWidgets.QPushButton("Save Memory for Reboot")
        self.btn_save_reboot.setStyleSheet("background:#111827; color:#e5e7eb;")
        btn_layout.addWidget(self.btn_save_reboot)

        self.chk_autoload = QtWidgets.QCheckBox("Load memory from SMB on startup")
        lay.addWidget(self.chk_autoload)

        self.lbl_reboot_status = QtWidgets.QLabel("Status: Ready")
        self.lbl_reboot_status.setStyleSheet("color:#22c55e;")
        lay.addWidget(self.lbl_reboot_status)

        self.btn_save_reboot.clicked.connect(self._cmd_save_reboot)

    # ---------------- Commands ----------------
    def _cmd_stabilize(self):
        self.brain.stance = "Conservative"
        self._speak("Stabilizing system. Conservative stance engaged.")

    def _cmd_high_alert(self):
        self.brain.meta_state = "Sentinel"
        self._speak("High alert mode. Sentinel meta state active.")

    def _cmd_learn(self):
        self._speak("Learning cycle initiated. Reinforcement memory engaged.")

    def _cmd_optimize(self):
        self.brain.stance = "Beast"
        self._speak("Optimization mode. Beast stance unleashed.")

    def _cmd_purge(self):
        self.brain.pattern_memory.clear()
        self._speak("Anomaly memory purged. Clean slate.")

    def _cmd_ask(self):
        q = self.entry_question.text().strip()
        if not q:
            return
        resp = f"Intent: Protect.\nConfidence: {self.brain.last_predictions['meta_conf']:.2f}\n" \
               f"Risk: {self.brain.last_predictions['best_guess']:.2f}\n" \
               f"Reasoning: {list(self.brain.last_reasoning)[-1] if self.brain.last_reasoning else 'N/A'}"
        self.txt_dialogue.append(f"YOU: {q}\nASI: {resp}\n")
        self._speak("Response delivered in console.")

    def _cmd_save_reboot(self):
        path = self.entry_reboot_path.text().strip()
        if not path:
            self.lbl_reboot_status.setText("Status: Path required")
            self.lbl_reboot_status.setStyleSheet("color:#f97316;")
            return
        try:
            self.reboot_mgr.save(self.brain, self.organs, path)
            self.lbl_reboot_status.setText("Status: Saved")
            self.lbl_reboot_status.setStyleSheet("color:#22c55e;")
        except Exception as e:
            self.lbl_reboot_status.setText(f"Status: Error {e}")
            self.lbl_reboot_status.setStyleSheet("color:#f97316;")

    def _speak(self, text):
        try:
            self.voice.say(text)
            self.voice.runAndWait()
        except Exception:
            pass

    # ---------------- Timer / Update Loop ----------------
    def _start_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def _tick(self):
        self.zero_trust.purge_expired()
        for o in self.organs:
            o.update()
            o.micro_recovery()
        self.brain.update(self.organs)
        self._update_borg_mirror()
        self._update_overview()
        self._update_brain_tab()
        self._update_threat_tab()
        self._draw_chart()

    def _update_borg_mirror(self):
        # Simple real-time discovery from network connections
        conns = psutil.net_connections(kind="inet")
        for c in conns:
            if c.raddr and c.status == "ESTABLISHED":
                url = f"{c.raddr.ip}:{c.raddr.port}"
                risk = min(100, c.raddr.port / 10)
                self.borg.discover(url, risk)
        self.mirror.evaluate({"status": "oscillating" if len(conns) > 50 else "calm"})

    def _update_overview(self):
        p = self.brain.last_predictions
        self.lbl_meta_state.setText(f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.setText(f"Stance: {self.brain.stance}")
        self.lbl_meta_conf.setText(f"Meta-Confidence: {p['meta_conf']:.2f}")
        self.lbl_current_risk.setText(f"Current Risk: {p['best_guess']:.2f}")
        self.lbl_integrity.setText(f"Model Integrity: {self.brain.npu.model_integrity:.2f}")
        self.txt_reason.clear()
        self.txt_reason.append("Reasoning Tail:")
        for line in self.brain.last_reasoning:
            self.txt_reason.append(f"  - {line}")

    def _update_brain_tab(self):
        stats = self.brain.npu.stats()
        txt = json.dumps(stats, indent=2)
        self.lbl_brain_stats.setText(f"Brain Stats:\n{txt}")

    def _update_threat_tab(self):
        s = self.borg.stats()
        self.lbl_borg.setText(f"Borg Mesh: {s['total']} nodes, {s['enforced']} enforced")
        self.lbl_mirror.setText(f"MirrorDefense: {self.mirror.last_status}")
        self.txt_threat.setPlainText(
            f"Zero Trust Personal Records: {len(self.zero_trust.personal_data_store)}\n"
            f"Fake Telemetry: {len(self.zero_trust.fake_telemetry)}\n"
            f"MAC/IP Store: {len(self.zero_trust.mac_ip_store)}"
        )

    def _draw_chart(self):
        w, h = 600, 180
        img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor("#050711"))
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        preds = self.brain.last_predictions
        short = preds["short"]
        med = preds["medium"]
        longp = preds["long"]
        base = preds["baseline"]
        best = preds["best_guess"]

        def y(v):
            v_clamp = max(0.0, min(1.0, v))
            return h - int(v_clamp * (h - 10)) - 5

        x_short, x_med, x_long = int(w * 0.2), int(w * 0.5), int(w * 0.8)
        y_short, y_med, y_long = y(short), y(med), y(longp)
        y_base, y_best = y(base), y(best)

        pen = QtGui.QPen(QtGui.QColor("#555555"))
        pen.setStyle(QtCore.Qt.DashLine)
        p.setPen(pen)
        p.drawLine(0, y_base, w, y_base)

        pen = QtGui.QPen(QtGui.QColor("#00f7ff"))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(x_short, y_short, x_med, y_med)
        p.drawLine(x_med, y_med, x_long, y_long)

        stance_color = {
            "Conservative": "#22c55e",
            "Balanced": "#eab308",
            "Beast": "#ef4444",
        }.get(self.brain.stance, "#ffffff")
        pen = QtGui.QPen(QtGui.QColor(stance_color))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(x_short, y_med, x_long, y_med)

        pen = QtGui.QPen(QtGui.QColor("#ff00ff"))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(0, y_best, w, y_best)

        p.setPen(QtGui.QColor("#9ca3af"))
        p.drawText(5, 15, "Short/Med/Long (cyan), Baseline (gray), Best‑Guess (magenta)")
        p.end()
        self.chart.setPixmap(QtGui.QPixmap.fromImage(img))

# ============================================================
# Main
# ============================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = GuardianUI()
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

