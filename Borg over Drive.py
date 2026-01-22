#!/usr/bin/env python3
"""
Borg Adaptive System Panel

- UniversalFormulaChip backbone (master formula engine)
- Local HTTP tap: /formula?i=...
- System Soul formula from live system metrics
- Borg number + Borg output
- Adaptive "mind" that evaluates and adjusts itself (internally only)
- Rolling history, evolving thresholds, and mutation-like behavior
- CPU / Threads / Data Flow gauges
- PyQt5 GUI with Save Memory (SMB/local) via Windows file dialog
"""

import sys
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
    QProgressBar, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt


# ---------- Helper ----------

def fmt(val, digits=2):
    if val is None:
        return "None"
    try:
        return f"{val:.{digits}f}"
    except Exception:
        return str(val)


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


# ---------- Borg Formula ----------

def borg_formula(system_value: int, borg_value: int) -> int:
    return int(
        0.7 * system_value
        + 0.3 * borg_value
        + 0.0001 * system_value * borg_value
    )


# ---------- Adaptive Mind ----------

class AdaptiveMind:
    def __init__(self, history_size=120):
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
        stress = deviation_from_baseline + abs(trend) + dev

        self._evolve_parameters(stress, dev, trend)

        if stress < self.deviation_threshold / 2:
            self.last_status = "Stable"
        elif stress < self.deviation_threshold:
            self.last_status = "Elevated"
        else:
            self.last_status = "Stressed"

        self.baseline = 0.98 * self.baseline + 0.02 * avg

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


# ---------- GUI ----------

class BorgPanel(QWidget):
    def __init__(self, chip: UniversalFormulaChip, metrics: SystemMetrics, mind: AdaptiveMind):
        super().__init__()
        self.chip = chip
        self.metrics = metrics
        self.mind = mind

        self.current_index = 1
        self.borg_value = 0

        # Create timer first so it's always available
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_all)

        self.init_ui()
        self.start_timers()

    def init_ui(self):
        self.setWindowTitle("Borg Adaptive System Panel")

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

        # Adaptive Mind tab
        mind_tab = QWidget()
        mind_layout = QVBoxLayout()
        mind_tab.setLayout(mind_layout)

        mind_layout.addWidget(QLabel("Adaptive Mind Status:"))
        self.mind_status_text = QTextEdit()
        self.mind_status_text.setReadOnly(True)
        self.mind_status_text.setMaximumHeight(160)
        mind_layout.addWidget(self.mind_status_text)

        tabs.addTab(mind_tab, "Adaptive Mind")

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

    # Universal formula controls

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

    # Borg controls

    def apply_borg(self):
        try:
            self.borg_value = int(self.borg_edit.text())
        except ValueError:
            self.borg_value = 0
            self.borg_edit.setText("0")
        self.update_all(force=True)

    # Main update loop

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

        self.assim_label.setText(f"Assimilation Level: {fmt(mind_state['assimilation_level'])}")
        self.resp_label.setText(f"Responsiveness Score: {fmt(mind_state['responsiveness_score'], 3)}")

        self.metrics_timer.start(self.mind.get_interval_ms())

    # Save Memory / Snapshot

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
        except Exception as e:
            self.snapshot_info.setPlainText(f"Error saving memory: {e}")
            self.status_label.setText(f"Error saving memory: {e}")


# ---------- Main ----------

def main():
    tap_thread = threading.Thread(target=run_http_tap, daemon=True)
    tap_thread.start()

    metrics = SystemMetrics()
    mind = AdaptiveMind(history_size=120)

    app = QApplication(sys.argv)
    panel = BorgPanel(chip, metrics, mind)
    panel.resize(1000, 700)
    panel.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

