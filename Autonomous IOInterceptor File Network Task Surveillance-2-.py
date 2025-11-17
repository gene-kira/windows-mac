import os, psutil, subprocess, threading, time, sys
from sympy import Eq, symbols, sympify
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QHBoxLayout, QTextEdit, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer

# === Thread-Safe Log Emitter ===
class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

# === Memory Stub ===
class Memory:
    def store_case(self, constraints, target, result):
        print(f"[Memory] Stored case for {target}: {result}")

# === IOInterceptor ===
class IOInterceptor:
    def __init__(self):
        self.last_snapshot = {}

    def capture(self):
        data = {}
        data.update(self.scan_files())
        data.update(self.scan_network())
        data.update(self.scan_tasks())
        self.last_snapshot = data
        return data

    def scan_files(self):
        suspicious_paths = [
            "C:\\Windows\\Temp\\DiagTrack",
            "C:\\ProgramData\\Microsoft\\Feedback",
            "C:\\Windows\\System32\\LogFiles\\WMI"
        ]
        result = {}
        for path in suspicious_paths:
            key = f"file_exists_{path.replace(':','').replace('\\','_')}"
            result[symbols(key)] = os.path.exists(path)
        return result

    def scan_network(self):
        result = {}
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.raddr:
                ip = conn.raddr.ip
                key = f"packet_destination_{ip.replace('.','_')}"
                result[symbols(key)] = True
        return result

    def scan_tasks(self):
        result = {}
        try:
            output = subprocess.check_output("schtasks", shell=True).decode()
            result[symbols("DiagTrack_task_active")] = "DiagTrack" in output or "Feedback" in output
        except Exception:
            result[symbols("DiagTrack_task_active")] = False
        return result

# === Telemetry Stub ===
class Telemetry:
    def snapshot(self):
        return {
            symbols("telemetry_level"): 3,
            symbols("error_reporting"): 1
        }

    def purge(self, state):
        print("🔻 Purging telemetry based on:", state)

# === ReasonerDaemon ===
class ReasonerDaemon:
    def __init__(self, telemetry, memory, log_emitter):
        self.constraints = []
        self.variables = {}
        self.telemetry = telemetry
        self.memory = memory
        self.io_interceptor = IOInterceptor()
        self.log_emitter = log_emitter
        self.running = False
        self.interval = 5

    def emit(self, msg):
        self.log_emitter.log_signal.emit(msg)

    def add_constraint(self, expr: str):
        if "=" not in expr:
            self.emit(f"error: Missing '=' in constraint: {expr}")
            return False
        parts = expr.split("=")
        if len(parts) != 2:
            self.emit(f"error: Invalid format: {expr}")
            return False
        left, right = parts
        try:
            syms = {str(s): symbols(str(s)) for s in sympify(expr).free_symbols}
            eq = Eq(sympify(left.strip(), syms), sympify(right.strip(), syms))
            self.constraints.append(eq)
            self.emit(f"add_constraint: {expr}")
            return True
        except Exception as e:
            self.emit(f"error: Parse failed: {expr} → {e}")
            return False

    def evaluate(self, inputs: dict):
        for eq in self.constraints:
            try:
                result = eq.subs(inputs)
                if not result:
                    return False
            except Exception:
                return False
        return True

    def enforce(self):
        inputs = {**self.telemetry.snapshot(), **self.io_interceptor.capture()}
        if not self.evaluate(inputs):
            self.emit(f"violation: {inputs}")
            self.memory.store_case(self.constraints, "violation", inputs)
            self.telemetry.purge(inputs)

    def start(self):
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.running:
            self.enforce()
            time.sleep(self.interval)

# === GUI Overlay ===
class ConstraintDebugger(QWidget):
    def __init__(self, reasoner):
        super().__init__()
        self.reasoner = reasoner
        self.setWindowTitle("Codex Purge Shell: Constraint Matrix")
        self.setGeometry(100, 100, 1000, 700)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.init_constraint_table()
        self.init_symbol_inspector()
        self.init_mutation_log()
        self.init_constraint_composer()
        self.init_intercept_panel()
        self.init_violation_pulse()
        self.init_error_panel()
        self.setup_timer()

    def init_constraint_table(self):
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Constraint", "Status", "Source"])
        self.layout.addWidget(QLabel("🧩 Constraint Matrix"))
        self.layout.addWidget(self.table)

    def refresh_constraints(self):
        self.table.setRowCount(len(self.reasoner.constraints))
        inputs = {**self.reasoner.telemetry.snapshot(), **self.reasoner.io_interceptor.last_snapshot}
        for i, eq in enumerate(self.reasoner.constraints):
            self.table.setItem(i, 0, QTableWidgetItem(str(eq)))
            status = "✓" if eq.subs(inputs) else "✗"
            self.table.setItem(i, 1, QTableWidgetItem(status))
            self.table.setItem(i, 2, QTableWidgetItem("Daemon"))

    def init_symbol_inspector(self):
        self.layout.addWidget(QLabel("🔍 Symbol Inspector"))
        self.symbols = QTextEdit()
        self.symbols.setReadOnly(True)
        self.layout.addWidget(self.symbols)

    def update_symbols(self):
        self.symbols.setText("\n".join(f"{k}: {v}" for k, v in self.reasoner.variables.items()))

    def init_mutation_log(self):
        self.layout.addWidget(QLabel("📜 Mutation Log"))
        self.mutations = QTextEdit()
        self.mutations.setReadOnly(True)
        self.layout.addWidget(self.mutations)

    def append_log(self, message):
        self.mutations.append(message)
        self.mutations.ensureCursorVisible()

    def init_constraint_composer(self):
        self.layout.addWidget(QLabel("🧠 Add Constraint"))
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_constraint)
        row.addWidget(self.input)
        row.addWidget(self.add_btn)
        self.layout.addLayout(row)

    def add_constraint(self):
        expr = self.input.text()
        success = self.reasoner.add_constraint(expr)
        if success:
            self.refresh_constraints()
        self.update_errors()

    def init_intercept_panel(self):
        self.intercept_box = QGroupBox("🔻 Intercepted Data Streams")
        self.intercept_layout = QGridLayout()
        self.intercept_box.setLayout(self.intercept_layout)
        self.layout.addWidget(self.intercept_box)

    def refresh_intercepts(self):
        snapshot = self.reasoner.io_interceptor.last_snapshot
        for i in reversed(range(self.intercept_layout.count())):
            self.intercept_layout.itemAt(i).widget().setParent(None)
        for i, (sym, val) in enumerate(snapshot.items()):
            label = QLabel(f"{sym}: {'✓' if val else '✗'}")
            label.setStyleSheet("color: green;" if val else "color: red;")
            self.intercept_layout.addWidget(label, i, 0)

    def init_violation_pulse(self):
        self.pulse = QLabel("✓ All constraints satisfied.")
        self.pulse.setStyleSheet("color: green; font-weight: bold;")
        self.layout.addWidget(self.pulse)

    def update_pulse(self):
        inputs = {**self.reasoner.telemetry.snapshot(), **self.reasoner.io_interceptor.capture()}
        if not self.reasoner.evaluate(inputs):
            self.pulse.setText("⚠️ Constraint Violation Detected!")
            self.pulse.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pulse.setText("✓ All constraints satisfied.")
            self.pulse.setStyleSheet("color: green; font-weight: bold;")

    def init_error_panel(self):
        self.layout.addWidget(QLabel("❌ Constraint Errors"))
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.layout.addWidget(self.error_log)

    def update_errors(self):
        errors = [line for line in self.mutations.toPlainText().splitlines() if line.startswith("error")]
        self.error_log.setText("\n".join(errors))

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start(5000)  # Refresh every 5 seconds

    def refresh_all(self):
        self.refresh_constraints()
        self.update_symbols()
        self.refresh_intercepts()
        self.update_pulse()
        self.update_errors()
# === Launch Everything ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    log_emitter = LogEmitter()
    reasoner = ReasonerDaemon(Telemetry(), Memory(), log_emitter)

    gui = ConstraintDebugger(reasoner)
    log_emitter.log_signal.connect(gui.append_log)

    # Preload symbolic constraints
    reasoner.add_constraint("telemetry_level <= 1")
    reasoner.add_constraint("DiagTrack_task_active = False")
    reasoner.add_constraint("error_reporting = 0")
    reasoner.add_constraint("file_exists_C_Windows_Temp_DiagTrack = False")

    reasoner.start()
    gui.show()
    sys.exit(app.exec_())



        
        

