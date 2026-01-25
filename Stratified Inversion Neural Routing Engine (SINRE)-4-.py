import sys
import subprocess
import importlib
import math
import random
import os
import platform
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass

# =========================
# Silent autoloader
# =========================

def ensure_package(pkg_name, import_name=None):
    mod_name = import_name or pkg_name
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        print(f"[AUTOLOADER] {pkg_name} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        print(f"[AUTOLOADER] {pkg_name} installed.")
        return importlib.import_module(mod_name)

psutil = ensure_package("psutil", "psutil")
QtCore = ensure_package("PyQt5", "PyQt5.QtCore")
QtGui = ensure_package("PyQt5", "PyQt5.QtGui")
QtWidgets = ensure_package("PyQt5", "PyQt5.QtWidgets")

# Optional GPU telemetry
try:
    GPUtil = ensure_package("GPUtil", "GPUtil")
except Exception:
    GPUtil = None

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
)

# =========================
# Storage root helpers
# =========================

def get_sinre_storage_root():
    system = platform.system().lower()

    if system == "windows":
        # Try local drives D:..Z:
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:/"
            if os.path.exists(root) and os.access(root, os.W_OK):
                return os.path.join(root, "SINRE")

        # Simple SMB-style roots (customize if you want)
        smb_candidates = [
            r"\\SINRE-SHARE",
            r"\\NETWORK-STORAGE",
        ]
        for smb in smb_candidates:
            if os.path.exists(smb) and os.access(smb, os.W_OK):
                return os.path.join(smb, "SINRE")

        # Fallback to C:
        return os.path.join("C:/", "SINRE")

    # Non-Windows: use home directory
    home = os.path.expanduser("~")
    return os.path.join(home, "SINRE")

def ensure_sinre_storage():
    root = get_sinre_storage_root()
    os.makedirs(root, exist_ok=True)
    return root

# =========================
# Tumblr Effect Organ
# =========================

class TumblrEffectOrgan:
    def invert(self, x):
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            return x[::-1]
        if isinstance(x, (list, tuple)):
            inv = [self.invert(v) for v in reversed(x)]
            return type(x)(inv)
        if isinstance(x, dict):
            new_d = {}
            for k, v in x.items():
                try:
                    new_d[self.invert(v)] = self.invert(k)
                except TypeError:
                    new_d[k] = self.invert(v)
            return new_d
        return x

# =========================
# 3D Fence Stack Organ (60 × rows × cols)
# =========================

class FenceStackOrgan:
    """
    3D fence:
    - layers: number of stacked fences (e.g. 60)
    - rows, cols: size of each fence (e.g. 6x6)
    Nodes are addressed as (layer, row, col).
    """

    def __init__(self, layers=60, rows=6, cols=6):
        self.layers = layers
        self.rows = rows
        self.cols = cols
        self.grid = [
            [[True for _ in range(cols)] for _ in range(rows)]
            for _ in range(layers)
        ]

    def set_node(self, z, r, c, active):
        if 0 <= z < self.layers and 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[z][r][c] = active

    def _neighbors(self, z, r, c):
        for dz, dr, dc in [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]:
            nz, nr, nc = z + dz, r + dr, c + dc
            if (
                0 <= nz < self.layers
                and 0 <= nr < self.rows
                and 0 <= nc < self.cols
                and self.grid[nz][nr][nc]
            ):
                yield (nz, nr, nc)

    def route(self, in_layer, in_row, out_layer, out_row):
        from collections import deque

        if not (
            0 <= in_layer < self.layers
            and 0 <= out_layer < self.layers
            and 0 <= in_row < self.rows
            and 0 <= out_row < self.rows
        ):
            return []

        start = (in_layer, in_row, 0)
        goal = (out_layer, out_row, self.cols - 1)

        if not self.grid[start[0]][start[1]][start[2]] or not self.grid[goal[0]][goal[1]][goal[2]]:
            return []

        q = deque([start])
        came_from = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nb in self._neighbors(*cur):
                if nb not in came_from:
                    came_from[nb] = cur
                    q.append(nb)

        if goal not in came_from:
            return []

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def route_value(self, value, in_layer, in_row, out_layer, out_row):
        path = self.route(in_layer, in_row, out_layer, out_row)
        if not path:
            return None, []
        return value, path

# =========================
# Replica-style NPU
# =========================

@dataclass
class HeadConfig:
    input_dim: int
    lr: float
    risk: float
    name: str

class ReplicaNPU:
    def __init__(
        self,
        cores=4,
        frequency_ghz=1.0,
        memory_size=64,
        plasticity_decay=0.0005,
        integrity_threshold=0.4,
    ):
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
        if not v1:
            return 0.0
        chunk = math.ceil(len(v1) / self.cores)
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    def add_head(self, name, input_dim, lr=0.01, risk=1.0):
        self.heads[name] = {
            "input_dim": input_dim,
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "history": deque(maxlen=64),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _adapt_input_for_head(self, x, head):
        dim = head["input_dim"]
        if len(x) == dim:
            return list(x)
        if len(x) < dim:
            return list(x) + [0.0] * (dim - len(x))
        return list(x[:dim])

    def _predict_head(self, head, x, name):
        x_adapted = self._adapt_input_for_head(x, head)
        y = self.vector_mac(x_adapted, head["w"])
        y += head["b"]
        y += self._symbolic_modulation(name)
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
            if name not in self.heads:
                continue
            head = self.heads[name]
            x_adapted = self._adapt_input_for_head(x, head)
            pred = self._predict_head(head, x_adapted, name)
            error = target - pred
            weighted_error = (
                error * head["risk"] * self.plasticity * self.model_integrity
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x_adapted[i]
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

    def check_integrity(self):
        if not self.heads:
            self.model_integrity = 1.0
            self.frozen = False
            return
        confs = [self.confidence(k) for k in self.heads]
        self.model_integrity = sum(confs) / len(confs)
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
            "confidence": {
                k: round(self.confidence(k), 3) for k in self.heads
            },
        }

# =========================
# Integrated Engine (LIVE SYSTEM)
# =========================

class IntegratedEngine:
    """
    LIVE ENGINE:
    - Uses psutil for real telemetry
    - TumblrEffectOrgan for inside-out transforms
    - ReplicaNPU for prediction/learning
    - FenceStackOrgan (60×6×6) for 3D routing
    - Berserk mode for full Beast behavior
    """

    def __init__(self, layers=60, rows=6, cols=6):
        self.tumblr = TumblrEffectOrgan()
        self.npu = ReplicaNPU(cores=4, frequency_ghz=1.0)
        self.fence = FenceStackOrgan(layers=layers, rows=rows, cols=cols)

        self.npu.add_head("cpu_short", input_dim=4, lr=0.03, risk=1.2)
        self.npu.add_head("cpu_long", input_dim=4, lr=0.01, risk=0.8)

        self.recent_loads = deque(maxlen=32)
        self.berserk_mode = False

    def set_berserk_mode(self, enabled):
        self.berserk_mode = enabled

    def get_live_telemetry(self):
        cpu_load = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        mem_used = mem.percent
        freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        efficiency = max(0.0, 100.0 - abs(cpu_load - 50))

        # CPU temperature
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                if entries:
                    cpu_temp = entries[0].current
                    break
        except Exception:
            cpu_temp = None

        # GPU telemetry (if GPUtil available)
        gpu_load = None
        gpu_temp = None
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_load = gpu.load * 100.0
                    gpu_temp = gpu.temperature
            except Exception:
                gpu_load = None
                gpu_temp = None

        return cpu_load, mem_used, freq, efficiency, cpu_temp, gpu_load, gpu_temp

    def step(self):
        cpu_load, mem_used, freq, efficiency, cpu_temp, gpu_load, gpu_temp = self.get_live_telemetry()

        self.recent_loads.append(cpu_load)
        avg_load = sum(self.recent_loads) / len(self.recent_loads)

        x = [
            avg_load / 100.0,
            mem_used / 100.0,
            efficiency / 100.0,
            (freq / 5000.0) if freq else 0.0,
        ]

        x_inverted = self.tumblr.invert(x)

        # Berserk: crank plasticity, learning rate, risk, and effective frequency
        if self.berserk_mode:
            self.npu.plasticity = min(1.0, self.npu.plasticity + 0.05)
            for head in self.npu.heads.values():
                head["lr"] *= 1.2
                head["risk"] *= 1.3
            self.npu.frequency_ghz *= 1.5

        preds = self.npu.predict(x_inverted)
        targets = {
            "cpu_short": cpu_load / 100.0,
            "cpu_long": (avg_load / 100.0) * 0.6 + (efficiency / 100.0) * 0.4,
        }
        self.npu.learn(x_inverted, targets)
        self.npu.check_integrity()
        self.npu.micro_recovery()

        pred_short = preds.get("cpu_short", cpu_load / 100.0)

        layers = self.fence.layers
        rows = self.fence.rows

        in_layer = int(pred_short * (layers - 1))
        out_layer = int((1.0 - pred_short) * (layers - 1))

        in_row = int((cpu_load / 100.0) * (rows - 1))
        out_row = int((efficiency / 100.0) * (rows - 1))

        value, path = self.fence.route_value(
            pred_short,
            in_layer,
            in_row,
            out_layer,
            out_row
        )

        return {
            "cpu_load": cpu_load,
            "mem_used": mem_used,
            "freq": freq,
            "efficiency": efficiency,
            "cpu_temp": cpu_temp,
            "gpu_load": gpu_load,
            "gpu_temp": gpu_temp,
            "input_vector": x,
            "inverted_vector": x_inverted,
            "predictions": preds,
            "pred_short": pred_short,
            "fence_in": (in_layer, in_row),
            "fence_out": (out_layer, out_row),
            "fence_path": path,
            "npu_stats": self.npu.stats(),
        }

# =========================
# Glitch‑corrupted ASCII cube frames
# =========================

ASCII_CUBE_FRAMES = [
    # Frame 0 – normal
    (
        "    +----+   \n"
        "   /    /|   \n"
        "  +----+ |   \n"
        "  |    | +   \n"
        "  |    |/    \n"
        "  +----+     "
    ),
    # Frame 1 – slight corruption
    (
        "    +--#-+   \n"
        "   /   %/|   \n"
        "  +--?#+ |   \n"
        "  |   !| +   \n"
        "  |   ~|/    \n"
        "  +----+     "
    ),
    # Frame 2 – heavy corruption
    (
        "    +X@@X+   \n"
        "   /@@@/|    \n"
        "  +@@@+ |    \n"
        "  |@@@| +    \n"
        "  |@@@|/     \n"
        "  +X@@X+     "
    ),
    # Frame 3 – glitch smear
    (
        "    +====+   \n"
        "   ///// /|  \n"
        "  +====+ |   \n"
        "  |////| +   \n"
        "  |////|/    \n"
        "  +====+     "
    ),
    # Frame 4 – distortion
    (
        "    +----+   \n"
        "   /    /|   \n"
        "  +----+ |   \n"
        "  |  * | +   \n"
        "  | *  |/    \n"
        "  +----+     "
    ),
    # Frame 5 – meltdown
    (
        "    +----+   \n"
        "   /XXXX/|   \n"
        "  +XXXX+ |   \n"
        "  |XXXX| +   \n"
        "  |XXXX|/    \n"
        "  +----+     "
    ),
    # Frame 6 – static burst
    (
        "    +----+   \n"
        "   /%%%%/|   \n"
        "  +%%%%+ |   \n"
        "  |%%%%| +   \n"
        "  |%%%%|/    \n"
        "  +----+     "
    ),
    # Frame 7 – corrupted geometry
    (
        "    +----+   \n"
        "   /    /|   \n"
        "  +----+ |   \n"
        "  |  \\ | +   \n"
        "  | \\  |/    \n"
        "  +----+     "
    ),
    # Frame 8 – full glitch
    (
        "    +####+   \n"
        "   /####/|   \n"
        "  +####+ |   \n"
        "  |####| +   \n"
        "  |####|/    \n"
        "  +####+     "
    ),
    # Frame 9 – recovery wobble
    (
        "    +----+   \n"
        "   /    /|   \n"
        "  +----+ |   \n"
        "  |    | +   \n"
        "  |    |/    \n"
        "  +----+     "
    ),
]

# =========================
# Borg Console GUI
# =========================

class BorgConsole(QMainWindow):
    def __init__(self, engine: IntegratedEngine):
        super().__init__()
        self.engine = engine

        self.setWindowTitle("Stratified Inversion Neural Routing Engine – LIVE SYSTEM")
        self.setMinimumSize(1400, 900)

        self._cube_index = 0

        self._init_ui()
        self._init_timer()

    def _init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #000000; }
            QLabel { color: #00FF66; font-size: 14px; }
            QTextEdit {
                background-color: #001100;
                color: #00FF66;
                border: 1px solid #00AA44;
                font-family: Consolas;
                font-size: 12px;
            }
        """)

        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Logo
        logo = QLabel("⭐ STRATIFIED INVERSION NEURAL ROUTING ENGINE – LIVE SYSTEM ⭐")
        logo.setStyleSheet("""
            color: #00FFAA;
            font-size: 22px;
            font-weight: bold;
            font-family: Consolas;
            padding: 6px;
        """)
        layout.addWidget(logo, alignment=Qt.AlignCenter)

        # Status strip
        self.status_strip = QLabel("STATUS: ONLINE")
        self.status_strip.setStyleSheet("""
            color: #000000;
            background-color: #00FF66;
            font-size: 14px;
            font-weight: bold;
            font-family: Consolas;
            padding: 4px 10px;
        """)
        layout.addWidget(self.status_strip)

        # ASCII cube
        self.cube_display = QTextEdit()
        self.cube_display.setReadOnly(True)
        self.cube_display.setFixedHeight(120)
        self.cube_display.setStyleSheet("""
            background-color: #000000;
            color: #00FF66;
            border: 1px solid #00AA44;
            font-family: Consolas;
            font-size: 12px;
        """)
        layout.addWidget(self.cube_display)

        # Beast Mode toggle
        self.beast_button = QtWidgets.QPushButton("BEAST MODE: OFF")
        self.beast_button.setCheckable(True)
        self.beast_button.setStyleSheet("""
            QPushButton {
                background-color: #330000;
                color: #FF6666;
                font-weight: bold;
                font-family: Consolas;
                padding: 4px 10px;
            }
            QPushButton:checked {
                background-color: #FF0000;
                color: #FFFFFF;
            }
        """)
        self.beast_button.toggled.connect(self._toggle_beast_mode)
        layout.addWidget(self.beast_button)

        # Main panels
        panels = QHBoxLayout()
        layout.addLayout(panels)

        # LEFT: live telemetry
        left = QVBoxLayout()
        panels.addLayout(left)

        self.cpu_label = QLabel("CPU Load: 0%")
        self.mem_label = QLabel("Memory Used: 0%")
        self.freq_label = QLabel("CPU Frequency: 0 MHz")
        self.eff_label = QLabel("Efficiency: 0")

        self.cpu_temp_label = QLabel("CPU Temp: N/A")
        self.gpu_label = QLabel("GPU Load: N/A")
        self.gpu_temp_label = QLabel("GPU Temp: N/A")

        left.addWidget(self.cpu_label)
        left.addWidget(self.mem_label)
        left.addWidget(self.freq_label)
        left.addWidget(self.eff_label)
        left.addWidget(self.cpu_temp_label)
        left.addWidget(self.gpu_label)
        left.addWidget(self.gpu_temp_label)

        # MIDDLE: NPU + Tumblr
        mid = QVBoxLayout()
        panels.addLayout(mid)

        self.input_label = QLabel("Input Vector: []")
        self.inv_label = QLabel("Tumblr Inverted: []")
        self.pred_label = QLabel("Predictions: {}")

        mid.addWidget(self.input_label)
        mid.addWidget(self.inv_label)
        mid.addWidget(self.pred_label)

        # RIGHT: 3D Fence
        right = QVBoxLayout()
        panels.addLayout(right)

        self.fence_in_label = QLabel("Fence Input: (0,0)")
        self.fence_out_label = QLabel("Fence Output: (0,0)")
        self.fence_path_label = QLabel("Fence Path: []")

        right.addWidget(self.fence_in_label)
        right.addWidget(self.fence_out_label)
        right.addWidget(self.fence_path_label)

        # NPU stats
        self.stats_label = QLabel("NPU Stats: {}")
        layout.addWidget(self.stats_label)

        # Log window
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _init_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_console)
        self.timer.start()

    def _toggle_beast_mode(self, checked):
        self.engine.set_berserk_mode(checked)
        if checked:
            self.beast_button.setText("BEAST MODE: BERSERK")
        else:
            self.beast_button.setText("BEAST MODE: OFF")

    def update_console(self):
        result = self.engine.step()

        # Live telemetry
        self.cpu_label.setText(f"CPU Load: {result['cpu_load']:.1f}%")
        self.mem_label.setText(f"Memory Used: {result['mem_used']:.1f}%")
        self.freq_label.setText(f"CPU Frequency: {result['freq']:.0f} MHz")
        self.eff_label.setText(f"Efficiency: {result['efficiency']:.1f}")

        cpu_temp = result.get("cpu_temp")
        gpu_load = result.get("gpu_load")
        gpu_temp = result.get("gpu_temp")

        if cpu_temp is not None:
            self.cpu_temp_label.setText(f"CPU Temp: {cpu_temp:.1f} °C")
        else:
            self.cpu_temp_label.setText("CPU Temp: N/A")

        if gpu_load is not None:
            self.gpu_label.setText(f"GPU Load: {gpu_load:.1f}%")
        else:
            self.gpu_label.setText("GPU Load: N/A")

        if gpu_temp is not None:
            self.gpu_temp_label.setText(f"GPU Temp: {gpu_temp:.1f} °C")
        else:
            self.gpu_temp_label.setText("GPU Temp: N/A")

        # NPU + Tumblr
        self.input_label.setText(f"Input Vector: {result['input_vector']}")
        self.inv_label.setText(f"Tumblr Inverted: {result['inverted_vector']}")
        self.pred_label.setText(f"Predictions: {result['predictions']}")

        # 3D Fence
        self.fence_in_label.setText(f"Fence Input: {result['fence_in']}")
        self.fence_out_label.setText(f"Fence Output: {result['fence_out']}")
        self.fence_path_label.setText(f"Fence Path: {result['fence_path']}")

        # NPU stats
        stats = result["npu_stats"]
        self.stats_label.setText(f"NPU Stats: {stats}")

        # Rotating glitch‑corrupted ASCII cube (faster in Berserk)
        step = 3 if self.engine.berserk_mode else 1
        self._cube_index += step
        frame = ASCII_CUBE_FRAMES[self._cube_index % len(ASCII_CUBE_FRAMES)]
        self.cube_display.setPlainText(frame)

        # Status strip
        status, color = self._compute_status(stats, result)
        self.status_strip.setText(f"STATUS: {status}")
        self.status_strip.setStyleSheet(f"""
            color: #000000;
            background-color: {color};
            font-size: 14px;
            font-weight: bold;
            font-family: Consolas;
            padding: 4px 10px;
        """)

        # Log
        self.log.append(
            f"[LIVE] CPU={result['cpu_load']:.1f}% | MEM={result['mem_used']:.1f}% | "
            f"Route={result['fence_path']} | STATUS={status}"
        )

        # Save state
        self._save_state(result, status)

    def _compute_status(self, stats, result):
        if self.engine.berserk_mode:
            return "BERSERK", "#FF0033"

        integrity = stats.get("integrity", 1.0)
        plasticity = stats.get("plasticity", 1.0)
        fence_path = result.get("fence_path", [])

        if integrity < 0.5:
            return "RECOVERY", "#FFCC00"
        if plasticity > 0.8:
            return "LEARNING", "#00AAFF"
        if fence_path:
            return "ROUTING", "#00FF66"
        return "ONLINE", "#00FFAA"

    def _save_state(self, result, status):
        try:
            root = ensure_sinre_storage()
            snapshot = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": status,
                "cpu_load": result.get("cpu_load"),
                "mem_used": result.get("mem_used"),
                "freq": result.get("freq"),
                "efficiency": result.get("efficiency"),
                "cpu_temp": result.get("cpu_temp"),
                "gpu_load": result.get("gpu_load"),
                "gpu_temp": result.get("gpu_temp"),
                "predictions": result.get("predictions"),
                "fence_in": result.get("fence_in"),
                "fence_out": result.get("fence_out"),
                "npu_stats": result.get("npu_stats"),
            }
            path = os.path.join(root, "sinre_state.log")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
        except Exception:
            pass

# =========================
# Main launcher
# =========================

def main():
    app = QApplication(sys.argv)
    engine = IntegratedEngine(layers=60, rows=6, cols=6)
    window = BorgConsole(engine)
    window.show()  # normal window, not fullscreen
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

