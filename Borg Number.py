#!/usr/bin/env python3
"""
Borg Adaptive System Panel

- UniversalFormulaChip backbone (master formula engine)
- Local HTTP tap: /formula?i=...
- System Soul formula from live system metrics
- Borg number + Borg output
- Adaptive "mind" that evaluates and adjusts itself (internally only)
- Rolling history, evolving thresholds, and mutation-like behavior
- PyQt5 GUI with Save Snapshot (SMB/local) via Windows file dialog
"""

import importlib
import sys
import threading
import time
import textwrap
import json
from collections import deque
from statistics import mean, pstdev

# ---------- Autoloader ----------

def auto_import(module_name, friendly_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        label = friendly_name or module_name
        print(f"[FATAL] Missing required library: {label}")
        print(f"Install it with:  pip install {module_name}")
        sys.exit(1)

sympy = auto_import("sympy", "Sympy (symbolic math)")
flask = auto_import("flask", "Flask (HTTP tap)")
psutil = auto_import("psutil", "psutil (system metrics)")
PyQt5 = auto_import("PyQt5", "PyQt5 (GUI)")
from flask import Flask, request, jsonify
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt

symbols = sympy.symbols
sin = sympy.sin
cos = sympy.cos
exp = sympy.exp
log = sympy.log
sqrt = sympy.sqrt
Integer = sympy.Integer


# ---------- Universal Formula Backbone ----------

class UniversalFormulaChip:
    def __init__(self):
        self.x, self.y, self.z = symbols("x y z")

    def decode(self, i: int):
        if i <= 0:
            i = 1

        op_selector = i % 7
        a = Integer(i % 11 - 5)
        b = Integer((i // 7) % 13 - 6)
        c = Integer((i // 49) % 17 - 8)

        x, y, z = self.x, self.y, self.z

        if op_selector == 0:
            expr = a * x + b * y + c
        elif op_selector == 1:
            expr = a * x**2 + b * y + c * z
        elif op_selector == 2:
            expr = sin(a * x) + cos(b * y) + c
        elif op_selector == 3:
            expr = exp(a * x + b * y) + c
        elif op_selector == 4:
            expr = log(abs(a * x + b)) + c
        elif op_selector == 5:
            expr = sqrt(abs(a * x**2 + b * y + c))
        else:
            expr = (a * x + b) / (c * y + 1)

        return sympy.simplify(expr)

    def master_formula(self, i: int):
        return self.decode(i)


# ---------- HTTP Tap ----------

chip = UniversalFormulaChip()
app = Flask(__name__)

@app.route("/formula")
def formula_endpoint():
    try:
        i = int(request.args.get("i", "1"))
    except ValueError:
        i = 1
    expr = chip.master_formula(i)
    return jsonify({"i": i, "formula": str(expr)})

def run_http_tap():
    app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False)


# ---------- System Metrics + System Soul ----------

class SystemMetrics:
    def __init__(self):
        self.last_net = psutil.net_io_counters()
        self.last_disk = psutil.disk_io_counters()
        self.last_time = time.time()

        self.cpu = 0.0
        self.ram = 0.0
        self.disk_read = 0.0
        self.disk_write = 0.0
        self.net_sent = 0.0
        self.net_recv = 0.0
        self.proc_count = 0
        self.thread_count = 0

        self.CPU, self.RAM, self.DR, self.DW, self.NS, self.NR, self.PROC, self.THR = symbols(
            "CPU RAM DR DW NS NR PROC THR"
        )

        # System Soul formula: nonlinear, weighted
        self.soul_formula = sympy.simplify(
            (self.CPU**2 + self.RAM**2) / 10
            + 40 * (self.DR + self.DW)
            + 40 * (self.NS + self.NR)
            + self.PROC
            + self.THR / 20
        )

    def update(self):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1.0

        self.cpu = psutil.cpu_percent(interval=None)
        self.ram = psutil.virtual_memory().percent

        disk = psutil.disk_io_counters()
        self.disk_read = (disk.read_bytes - self.last_disk.read_bytes) / dt / (1024 * 1024)
        self.disk_write = (disk.write_bytes - self.last_disk.write_bytes) / dt / (1024 * 1024)
        self.last_disk = disk

        net = psutil.net_io_counters()
        self.net_sent = (net.bytes_sent - self.last_net.bytes_sent) / dt / (1024 * 1024)
        self.net_recv = (net.bytes_recv - self.last_net.bytes_recv) / dt / (1024 * 1024)
        self.last_net = net

        self.proc_count = len(psutil.pids())
        self.thread_count = sum(p.info["num_threads"] for p in psutil.process_iter(["num_threads"]))

        self.last_time = now

    def soul_value(self):
        subs = {
            self.CPU: self.cpu,
            self.RAM: self.ram,
            self.DR: self.disk_read,
            self.DW: self.disk_write,
            self.NS: self.net_sent,
            self.NR: self.net_recv,
            self.PROC: self.proc_count,
            self.THR: self.thread_count,
        }
        val = self.soul_formula.evalf(subs=subs)
        return int(val)

    def soul_instantiated_expr(self):
        subs = {
            self.CPU: round(self.cpu, 2),
            self.RAM: round(self.ram, 2),
            self.DR: round(self.disk_read, 3),
            self.DW: round(self.disk_write, 3),
            self.NS: round(self.net_sent, 3),
            self.NR: round(self.net_recv, 3),
            self.PROC: self.proc_count,
            self.THR: self.thread_count,
        }
        return sympy.simplify(self.soul_formula.subs(subs))

    def to_dict(self):
        return {
            "cpu": self.cpu,
            "ram": self.ram,
            "disk_read": self.disk_read,
            "disk_write": self.disk_write,
            "net_sent": self.net_sent,
            "net_recv": self.net_recv,
            "proc_count": self.proc_count,
            "thread_count": self.thread_count,
        }


# ---------- Borg Formula ----------

def borg_formula(system_value: int, borg_value: int) -> int:
    # Borg as evolutionary catalyst: blend + interaction
    return int(
        0.7 * system_value
        + 0.3 * borg_value
        + 0.0001 * system_value * borg_value
    )


# ---------- Adaptive Mind (Internal Only) ----------

class AdaptiveMind:
    """
    Self-contained adaptive layer:
    - keeps rolling history of soul values
    - tracks baseline, deviation, trend
    - adjusts its own thresholds and sampling interval
    - never touches OS; only internal behavior
    """

    def __init__(self, history_size=120):
        self.history_size = history_size
        self.soul_history = deque(maxlen=history_size)
        self.borg_history = deque(maxlen=history_size)
        self.borg_out_history = deque(maxlen=history_size)

        # Baseline and thresholds evolve over time
        self.baseline = None
        self.deviation_threshold = 200.0  # starting point
        self.trend_threshold = 50.0       # starting point

        # Sampling interval in ms (adaptive)
        self.min_interval = 500
        self.max_interval = 5000
        self.current_interval = 2000

        # Mutation-like parameters
        self.mutation_rate = 0.05  # will change dynamically
        self.last_status = "Initializing"

    def update(self, soul_value: int, borg_value: int, borg_output: int):
        self.soul_history.append(soul_value)
        self.borg_history.append(borg_value)
        self.borg_out_history.append(borg_output)

        if len(self.soul_history) < 10:
            self.last_status = "Learning baseline..."
            return

        soul_list = list(self.soul_history)
        avg = mean(soul_list)
        dev = pstdev(soul_list) if len(soul_list) > 1 else 0.0
        trend = soul_list[-1] - soul_list[0]

        if self.baseline is None:
            self.baseline = avg

        deviation_from_baseline = abs(soul_list[-1] - self.baseline)

        # Simple "stress" score
        stress = deviation_from_baseline + abs(trend) + dev

        # Evolve thresholds and mutation rate based on stress
        self._evolve_parameters(stress, dev, trend)

        # Decide status
        if stress < self.deviation_threshold / 2:
            self.last_status = "Stable"
        elif stress < self.deviation_threshold:
            self.last_status = "Elevated"
        else:
            self.last_status = "Stressed"

        # Slowly adjust baseline toward current average
        self.baseline = 0.98 * self.baseline + 0.02 * avg

    def _evolve_parameters(self, stress, dev, trend):
        # Dynamic mutation rate: higher under stress
        base_mutation = 0.02
        extra = min(stress / 1000.0, 0.2)
        self.mutation_rate = base_mutation + extra

        # Mutate deviation threshold
        if stress > self.deviation_threshold:
            self.deviation_threshold *= (1.0 + self.mutation_rate * 0.5)
        else:
            self.deviation_threshold *= (1.0 - self.mutation_rate * 0.1)
            self.deviation_threshold = max(self.deviation_threshold, 50.0)

        # Mutate trend threshold
        if abs(trend) > self.trend_threshold:
            self.trend_threshold *= (1.0 + self.mutation_rate * 0.5)
        else:
            self.trend_threshold *= (1.0 - self.mutation_rate * 0.1)
            self.trend_threshold = max(self.trend_threshold, 10.0)

        # Adjust sampling interval: faster under stress, slower when stable
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
        # Simple heuristic: how much Borg output differs from soul baseline
        if not self.soul_history or not self.borg_out_history or self.baseline is None:
            return 0.0
        last_borg_out = self.borg_out_history[-1]
        return abs(last_borg_out - self.baseline)

    def responsiveness_score(self):
        # Higher when interval is low and status is stable/elevated
        base = (self.max_interval - self.current_interval) / (self.max_interval - self.min_interval + 1)
        if self.last_status == "Stable":
            factor = 1.0
        elif self.last_status == "Elevated":
            factor = 0.9
        else:
            factor = 0.7
        return max(0.0, min(1.0, base * factor))

    def to_dict(self):
        return {
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


# ---------- PyQt5 GUI ----------

def fmt(val, digits=2):
    if val is None:
        return "None"
    try:
        return f"{val:.{digits}f}"
    except Exception:
        return str(val)


class BorgPanel(QWidget):
    def __init__(self, chip: UniversalFormulaChip, metrics: SystemMetrics, mind: AdaptiveMind):
        super().__init__()
        self.chip = chip
        self.metrics = metrics
        self.mind = mind

        self.current_index = 1
        self.borg_value = 0

        self.init_ui()
        self.start_timers()

    def init_ui(self):
        self.setWindowTitle("Borg Adaptive System Panel")

        layout = QVBoxLayout()

        # Universal formula controls + Save
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Universal formula index (i):"))
        self.index_edit = QLineEdit("1")
        self.index_edit.setFixedWidth(80)
        top_row.addWidget(self.index_edit)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.prev_formula)
        top_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_formula)
        top_row.addWidget(self.btn_next)

        self.btn_go = QPushButton("Go")
        self.btn_go.clicked.connect(self.go_to_index)
        top_row.addWidget(self.btn_go)

        top_row.addStretch()

        self.btn_save = QPushButton("Save Snapshot")
        self.btn_save.clicked.connect(self.save_snapshot)
        top_row.addWidget(self.btn_save)

        layout.addLayout(top_row)

        # System Soul formula (symbolic)
        layout.addWidget(QLabel("System Soul Formula (Symbolic):"))
        self.soul_formula_text = QTextEdit()
        self.soul_formula_text.setReadOnly(True)
        layout.addWidget(self.soul_formula_text)

        # System Soul formula instantiated
        layout.addWidget(QLabel("System Soul Formula (With Live Values):"))
        self.soul_inst_text = QTextEdit()
        self.soul_inst_text.setReadOnly(True)
        layout.addWidget(self.soul_inst_text)

        # System Soul value and Borg value/output
        row_values = QHBoxLayout()

        self.system_value_label = QLabel("System Soul Value: 0")
        row_values.addWidget(self.system_value_label)

        row_values.addSpacing(20)

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
        layout.addLayout(row_values)

        # Adaptive mind status
        layout.addWidget(QLabel("Adaptive Mind Status:"))
        self.mind_status_text = QTextEdit()
        self.mind_status_text.setReadOnly(True)
        self.mind_status_text.setMaximumHeight(140)
        layout.addWidget(self.mind_status_text)

        # Universal formula display
        layout.addWidget(QLabel("Universal Formula (index i):"))
        self.uni_text = QTextEdit()
        self.uni_text.setReadOnly(True)
        layout.addWidget(self.uni_text)

        # Status
        self.status_label = QLabel("Backbone online. Tap: http://127.0.0.1:5005/formula?i=1")
        self.status_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Initial text
        self.soul_formula_text.setPlainText(sympy.pretty(self.metrics.soul_formula))
        self.update_universal_formula()
        self.update_system_and_borg(force=True)

    def start_timers(self):
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_system_and_borg)
        self.metrics_timer.start(self.mind.get_interval_ms())

    # ---- Universal formula controls ----

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

    # ---- Borg controls ----

    def apply_borg(self):
        try:
            self.borg_value = int(self.borg_edit.text())
        except ValueError:
            self.borg_value = 0
            self.borg_edit.setText("0")
        self.update_system_and_borg(force=True)

    # ---- System + Borg + Mind update ----

    def update_system_and_borg(self, force=False):
        self.metrics.update()

        system_val = self.metrics.soul_value()
        self.system_value_label.setText(f"System Soul Value: {system_val}")

        inst_expr = self.metrics.soul_instantiated_expr()
        self.soul_inst_text.setPlainText(sympy.pretty(inst_expr))

        borg_out = borg_formula(system_val, self.borg_value)
        self.borg_output_label.setText(f"Borg Output Value: {borg_out}")

        # Update adaptive mind
        self.mind.update(system_val, self.borg_value, borg_out)
        mind_state = self.mind.to_dict()

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

        # Adjust timer interval adaptively
        self.metrics_timer.start(self.mind.get_interval_ms())

    # ---- Save Snapshot ----

    def save_snapshot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        system_val = self.metrics.soul_value()
        borg_out = borg_formula(system_val, self.borg_value)

        snapshot = {
            "system_metrics": self.metrics.to_dict(),
            "system_soul_value": system_val,
            "borg_value": self.borg_value,
            "borg_output": borg_out,
            "adaptive_mind": self.mind.to_dict(),
            "universal_index": self.current_index,
            "universal_formula": str(self.chip.master_formula(self.current_index)),
            "timestamp": time.time(),
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            self.status_label.setText(f"Snapshot saved to: {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error saving snapshot: {e}")


# ---------- Main ----------

def main():
    tap_thread = threading.Thread(target=run_http_tap, daemon=True)
    tap_thread.start()

    metrics = SystemMetrics()
    mind = AdaptiveMind(history_size=120)

    app = QApplication(sys.argv)
    panel = BorgPanel(chip, metrics, mind)
    panel.resize(900, 700)
    panel.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

