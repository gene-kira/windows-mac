#!/usr/bin/env python3
# ============================================================
#  K A L E   G U A R D I A N   N E R V E   C E N T E R
#  HybridBrain + ReplicaNPU + Organs + Borg hooks
#  Tkinter MagicBox + optional PyQt5 holo-face
# ============================================================

import os, sys, math, time, threading, json, platform, queue, random, subprocess

# ---------------------- Auto-loader --------------------------
def ensure_packages(pkgs):
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

ensure_packages([
    "psutil", "pyttsx3", "watchdog", "PyQt5"
])

import psutil
import pyttsx3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PyQt5 import QtWidgets, QtGui, QtCore

# ---------------------- Elevation (Windows) ------------------
def ensure_admin():
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{a}"' for a in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Guardian] Elevation failed: {e}")
        # still continue, but warn
ensure_admin()

# ============================================================
# ReplicaNPU — Neural Processing Unit
# ============================================================
from collections import deque

class ReplicaNPU:
    def __init__(self, cores=8, frequency_ghz=1.2, memory_size=32):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.cycles = 0
        self.energy = 0.0
        self.memory = deque(maxlen=memory_size)
        self.heads = {}
        self.symbolic_bias = {}
        self.plasticity = 1.0
        self.plasticity_decay = 0.0005
        self.model_integrity = 1.0
        self.integrity_threshold = 0.4
        self.frozen = False

    # --- low-level ops ---
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

    # --- heads ---
    def add_head(self, name, input_dim, lr=0.01, risk=1.0):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "history": deque(maxlen=32),
        }

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def _symbolic_mod(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"] + self._symbolic_mod(name)
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
        errs = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            err = target - pred
            weighted = err * head["risk"] * self.plasticity * self.model_integrity
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted
            self.energy += 0.005
            errs[name] = err
        self.plasticity = max(0.1, self.plasticity - self.plasticity_decay)
        return errs

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

    def stats(self):
        t = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "time_s": t,
            "energy": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {k: round(self.confidence(k), 3) for k in self.heads},
        }

# ============================================================
# Organs — real data only
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

class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRAM")
        self.usage = 0.0

    def update(self):
        mem = psutil.virtual_memory()
        self.usage = mem.percent / 100.0
        self.risk = self.usage
        self.health = 1.0 - self.usage

class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Network")
        self.bytes_sent = 0
        self.bytes_recv = 0

    def update(self):
        c = psutil.net_io_counters()
        self.bytes_sent = c.bytes_sent
        self.bytes_recv = c.bytes_recv
        # simple heuristic: high throughput => higher risk
        total = self.bytes_sent + self.bytes_recv
        self.risk = min(1.0, total / (1024 * 1024 * 50))  # 50MB window
        self.health = 1.0 - self.risk

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
                    all_t.append(t.current)
            self.temp = max(all_t) if all_t else 40.0
        except Exception:
            self.temp = 40.0
        self.risk = min(1.0, max(0.0, (self.temp - 50) / 40.0))
        self.health = 1.0 - self.risk

class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Disk")
        self.read = 0
        self.write = 0

    def update(self):
        io = psutil.disk_io_counters()
        self.read = io.read_bytes
        self.write = io.write_bytes
        total = self.read + self.write
        self.risk = min(1.0, total / (1024 * 1024 * 200))  # 200MB window
        self.health = 1.0 - self.risk

# ============================================================
# HybridBrain — multi-horizon + best-guess + stance
# ============================================================
class HybridBrain:
    def __init__(self):
        self.npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
        self.npu.add_head("short", 4, lr=0.05, risk=1.5)
        self.npu.add_head("medium", 4, lr=0.03, risk=1.0)
        self.npu.add_head("long", 4, lr=0.02, risk=0.7)
        self.meta_state = "Sentinel"
        self.stance = "Balanced"
        self.last_predictions = {
            "short": 0.0, "medium": 0.0, "long": 0.0,
            "baseline": 0.0, "best_guess": 0.0, "meta_conf": 0.5
        }
        self.last_reasoning = []

    def _features_from_organs(self, organs):
        # CPU, RAM, NET risk, THERMAL risk
        cpu = psutil.cpu_percent() / 100.0
        ram = next(o for o in organs if isinstance(o, DeepRamOrgan)).usage
        net = next(o for o in organs if isinstance(o, NetworkWatcherOrgan)).risk
        therm = next(o for o in organs if isinstance(o, ThermalOrgan)).risk
        return [cpu, ram, net, therm]

    def _baseline(self, feats):
        # simple EWMA-like baseline
        return sum(feats) / len(feats)

    def _meta_conf(self):
        cs = [self.npu.confidence(h) for h in self.npu.heads]
        return sum(cs) / len(cs) if cs else 0.5

    def _best_guess(self, preds):
        # multi-engine voting: short/med/long + baseline
        vals = [preds["short"], preds["medium"], preds["long"], preds["baseline"]]
        # weight medium & baseline higher
        weights = [0.2, 0.4, 0.2, 0.2]
        return sum(v * w for v, w in zip(vals, weights))

    def _update_stance(self, feats, best_guess):
        cpu, ram, net, therm = feats
        risk = max(cpu, ram, net, therm, best_guess)
        if risk > 0.85:
            self.stance = "Conservative"
        elif risk < 0.55:
            self.stance = "Beast"
        else:
            self.stance = "Balanced"

    def _update_meta_state(self, best_guess):
        if best_guess > 0.9:
            self.meta_state = "Hyper-Flow"
        elif best_guess > 0.7:
            self.meta_state = "Sentinel"
        elif best_guess > 0.5:
            self.meta_state = "Recovery-Flow"
        else:
            self.meta_state = "Deep-Dream"

    def update(self, organs):
        feats = self._features_from_organs(organs)
        baseline = self._baseline(feats)
        preds = self.npu.predict(feats)
        preds["baseline"] = baseline
        best = self._best_guess(preds)
        meta_conf = self._meta_conf()
        self._update_stance(feats, best)
        self._update_meta_state(best)
        self.last_predictions = {
            "short": preds["short"],
            "medium": preds["medium"],
            "long": preds["long"],
            "baseline": baseline,
            "best_guess": best,
            "meta_conf": meta_conf,
        }
        self.last_reasoning = [
            f"CPU/RAM/NET/THERM = {', '.join(f'{v:.2f}' for v in feats)}",
            f"Baseline={baseline:.2f}, BestGuess={best:.2f}",
            f"MetaConf={meta_conf:.2f}, Stance={self.stance}, MetaState={self.meta_state}",
        ]

# ============================================================
# BorgMesh + MirrorDefense — stubs wired for future fusion
# ============================================================
class BorgMesh:
    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def stats(self):
        return {
            "total": len(self.nodes),
            "corridors": len(self.edges),
        }

class MirrorDefense:
    def __init__(self):
        self.last_status = "idle"

    def evaluate(self, analysis: dict):
        self.last_status = analysis.get("status", "unknown")

class MirrorHook:
    def __init__(self, defense: MirrorDefense):
        self.defense = defense

    def submit(self, analysis: dict):
        if not isinstance(analysis, dict):
            return
        self.defense.evaluate(analysis)

# ============================================================
# Ingestion watcher — real FS events, glyph-ready
# ============================================================
class IngestionHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory:
            self.callback("created", event.src_path)

def start_ingestion_watcher(path, callback):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    handler = IngestionHandler(callback)
    obs = Observer()
    obs.schedule(handler, path, recursive=False)
    obs.start()
    return obs

# ============================================================
# PyQt5 Holographic Face — optional diagnostics window
# ============================================================
class HoloFaceWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASI Holographic Sentinel")
        self.resize(400, 300)
        self.setStyleSheet("background-color: #050510; color: #00f7ff;")
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("HOLOGRAPHIC ASI FACE ONLINE")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.label)
        self.stream = QtWidgets.QLabel("∞ DATA STREAMS")
        self.stream.setAlignment(QtCore.Qt.AlignCenter)
        self.stream.setStyleSheet("font-size: 12px; color: #8888ff;")
        layout.addWidget(self.stream)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._pulse)
        self.timer.start(500)

    def _pulse(self):
        dots = "." * random.randint(1, 6)
        self.stream.setText(f"STREAMING{dots}")

def launch_holo_face():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = HoloFaceWindow()
    w.show()
    threading.Thread(target=app.exec_, daemon=True).start()

# ============================================================
# Tkinter MagicBox Nerve Center
# ============================================================
class GuardianApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kale Guardian Nerve Center — MagicBox Edition")
        self.root.geometry("1100x700")
        self.root.configure(bg="#05060b")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#05060b")
        style.configure("TLabel", background="#05060b", foreground="#d0e0ff")
        style.configure("TButton", background="#101320", foreground="#d0e0ff")
        style.configure("TLabelframe", background="#05060b", foreground="#d0e0ff")

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)

        self.brain = HybridBrain()
        self.organs = [
            DeepRamOrgan(),
            NetworkWatcherOrgan(),
            ThermalOrgan(),
            DiskOrgan(),
        ]
        self.mesh = BorgMesh()
        self.mirror_defense = MirrorDefense()
        self.mirror_hook = MirrorHook(self.mirror_defense)

        self.ingest_obs = start_ingestion_watcher("./ingest_zone", self._on_ingest_event)

        self._build_gui()
        self._start_update_loop()

    # ---------------- GUI layout ----------------
    def _build_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(main)
        right.pack(side="right", fill="y")

        # Brain panel
        frame_brain = ttk.LabelFrame(left, text="Hybrid Brain Core")
        frame_brain.pack(fill="x", pady=5)
        self.lbl_meta_state = ttk.Label(frame_brain, text="Meta-State: Sentinel")
        self.lbl_meta_state.pack(anchor="w")
        self.lbl_stance = ttk.Label(frame_brain, text="Stance: Balanced")
        self.lbl_stance.pack(anchor="w")
        self.lbl_meta_conf = ttk.Label(frame_brain, text="Meta-Confidence: 0.00")
        self.lbl_meta_conf.pack(anchor="w")
        self.lbl_model_integrity = ttk.Label(frame_brain, text="Model Integrity: 1.00")
        self.lbl_model_integrity.pack(anchor="w")
        self.lbl_current_risk = ttk.Label(frame_brain, text="Current Risk: 0.00")
        self.lbl_current_risk.pack(anchor="w")

        # Prediction chart
        frame_chart = ttk.LabelFrame(left, text="Predictive Intelligence")
        frame_chart.pack(fill="both", expand=True, pady=5)
        self.canvas_chart = tk.Canvas(frame_chart, width=600, height=220,
                                      bg="#05060b", highlightthickness=0)
        self.canvas_chart.pack(fill="both", expand=True)

        # Reasoning tail
        frame_reason = ttk.LabelFrame(left, text="Reasoning Tail")
        frame_reason.pack(fill="both", expand=True, pady=5)
        self.txt_reason = tk.Text(frame_reason, height=8, bg="#05060b",
                                  fg="#a0ffb0", insertbackground="#a0ffb0")
        self.txt_reason.pack(fill="both", expand=True)

        # Right side: controls
        frame_cmd = ttk.LabelFrame(right, text="Command Bar")
        frame_cmd.pack(fill="x", pady=5)
        ttk.Button(frame_cmd, text="Stabilize System",
                   command=lambda: self._speak("Stabilizing system")).pack(fill="x", pady=2)
        ttk.Button(frame_cmd, text="High-Alert Mode",
                   command=lambda: self._speak("Entering high alert mode")).pack(fill="x", pady=2)
        ttk.Button(frame_cmd, text="Begin Learning Cycle",
                   command=lambda: self._speak("Learning cycle engaged")).pack(fill="x", pady=2)
        ttk.Button(frame_cmd, text="Purge Anomaly Memory",
                   command=lambda: self._speak("Anomaly memory purged")).pack(fill="x", pady=2)

        frame_holo = ttk.LabelFrame(right, text="Holographic ASI")
        frame_holo.pack(fill="x", pady=5)
        ttk.Button(frame_holo, text="Launch Holo Face", command=launch_holo_face).pack(fill="x", pady=2)

        frame_reboot = ttk.LabelFrame(right, text="Reboot Memory")
        frame_reboot.pack(fill="x", pady=5)
        ttk.Label(frame_reboot, text="SMB / UNC Path:").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame_reboot, width=40)
        self.entry_reboot_path.pack(anchor="w", pady=2)
        ttk.Button(frame_reboot, text="Pick Path", command=self.cmd_pick_reboot_path).pack(fill="x", pady=2)
        ttk.Button(frame_reboot, text="Save Memory for Reboot",
                   command=self.cmd_save_reboot_memory).pack(fill="x", pady=2)
        self.var_reboot_autoload = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_reboot, text="Load memory from SMB on startup",
                        variable=self.var_reboot_autoload).pack(anchor="w", pady=2)
        self.lbl_reboot_status = tk.Label(frame_reboot, text="Status: Ready",
                                          fg="#00cc66", bg="#05060b")
        self.lbl_reboot_status.pack(anchor="w", pady=2)

        frame_mesh = ttk.LabelFrame(right, text="Borg Mesh / MirrorDefense")
        frame_mesh.pack(fill="x", pady=5)
        self.lbl_mesh_stats = ttk.Label(frame_mesh, text="Mesh: 0 nodes / 0 corridors")
        self.lbl_mesh_stats.pack(anchor="w")
        self.lbl_mirror_status = ttk.Label(frame_mesh, text="Mirror: idle")
        self.lbl_mirror_status.pack(anchor="w")

    # ---------------- SMB memory ----------------
    def cmd_pick_reboot_path(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path:
            messagebox.showerror("Error", "Please specify SMB / UNC path.")
            return
        try:
            os.makedirs(path, exist_ok=True)
            state = {
                "brain": self.brain.last_predictions,
                "meta_state": self.brain.meta_state,
                "stance": self.brain.stance,
            }
            with open(os.path.join(path, "guardian_state.json"), "w") as f:
                json.dump(state, f, indent=2)
            self.lbl_reboot_status.config(text="Status: Saved", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Error {e}", fg="#ff4444")

    # ---------------- Ingestion callback ----------------
    def _on_ingest_event(self, kind, path):
        # Real FS event — you can route into glyph map / MirrorDefense here
        self._speak("Ingestion event detected")
        self.mirror_hook.submit({"status": "oscillating", "positives": 1, "negatives": 1})

    # ---------------- Voice ----------------
    def _speak(self, text):
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run, daemon=True).start()

    # ---------------- Update loop ----------------
    def _start_update_loop(self):
        self._tick()
        self.root.after(1000, self._start_update_loop)

    def _tick(self):
        for o in self.organs:
            o.update()
            o.micro_recovery()
        self.brain.update(self.organs)
        self._update_gui()

    def _update_gui(self):
        p = self.brain.last_predictions
        self.lbl_meta_state.config(text=f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.config(text=f"Stance: {self.brain.stance}")
        self.lbl_meta_conf.config(text=f"Meta-Confidence: {p['meta_conf']:.2f}")
        self.lbl_model_integrity.config(text=f"Model Integrity: {self.brain.npu.model_integrity:.2f}")
        current_risk = max(p["short"], p["medium"], p["long"], p["best_guess"])
        self.lbl_current_risk.config(text=f"Current Risk: {current_risk:.2f}")

        ms = self.mesh.stats()
        self.lbl_mesh_stats.config(text=f"Mesh: {ms['total']} nodes / {ms['corridors']} corridors")
        self.lbl_mirror_status.config(text=f"Mirror: {self.mirror_defense.last_status}")

        self._draw_chart()
        self._update_reasoning()

    def _draw_chart(self):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])
        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#05060b", outline="")

        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        longp = p["long"]
        baseline = p["baseline"]
        best = p["best_guess"]

        def y(v):
            v_clamped = max(0.0, min(1.0, v))
            return h - int(v_clamped * (h - 10)) - 5

        x_s, x_m, x_l = w * 0.2, w * 0.5, w * 0.8
        y_s, y_m, y_l = y(short), y(med), y(longp)
        y_b = y(baseline)
        y_best = y(best)

        # Baseline
        self.canvas_chart.create_line(0, y_b, w, y_b, fill="#555555", dash=(2, 2))
        # Short/Med/Long
        self.canvas_chart.create_line(x_s, y_s, x_m, y_m, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_m, y_m, x_l, y_l, fill="#00ccff", width=2)
        # Stance-colored medium band
        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.brain.stance, "#ffffff")
        self.canvas_chart.create_line(x_s, y_m, x_l, y_m, fill=stance_color, width=1)
        # Best-Guess
        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)

        self.canvas_chart.create_text(
            5, 5, anchor="nw", fill="#aaaaaa",
            text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)"
        )

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")

# ============================================================
# Entry point
# ============================================================
def main():
    root = tk.Tk()
    app = GuardianApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

