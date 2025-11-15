import threading, time, random, sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QFrame
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

# 🧠 Plasma Daemon Core
class LivingPlasmaDaemon:
    def __init__(self):
        self.mutation_log = []
        self.integrity_threshold = 0.2
        self.E_threshold = 100.0
        self.mu_0 = 4 * 3.1415e-7

    def calculate_energy(self, strikes, V, I, t, η):
        return strikes * V * I * t * η

    def calculate_integrity(self, energy):
        B = random.uniform(1.0, 5.0)
        integrity = (B ** 2 / self.mu_0) * (1 - energy / self.E_threshold)
        return max(0, integrity), B

    def process_strike(self, strikes, V, I, t, η):
        energy = self.calculate_energy(strikes, V, I, t, η)
        integrity, B = self.calculate_integrity(energy)
        mutation = {
            "timestamp": time.time(),
            "strikes": strikes,
            "energy": energy,
            "integrity": integrity,
            "field_strength_T": B
        }
        self.mutation_log.append(mutation)
        if integrity < self.integrity_threshold:
            self.trigger_resurrection_lockdown(mutation)
        return mutation

    def trigger_resurrection_lockdown(self, mutation):
        print("⚠️ Resurrection Detected: Plasma breach imminent!")
        print(f"🧬 Mutation Log Entry: {mutation}")
        print("🔒 Initiating symbolic lockdown, glyph overlay, and swarm sync alert...")

# ⚡ Lightning Capture Daemon
class LightningCaptureDaemon(threading.Thread):
    def __init__(self, gui_callback, daemon_core):
        super().__init__(daemon=True)
        self.gui_callback = gui_callback
        self.daemon_core = daemon_core

    def run(self):
        while True:
            strikes = random.randint(1, 5)
            V, I, t, η = 1e9, 30000, 0.0002, 0.75
            mutation = self.daemon_core.process_strike(strikes, V, I, t, η)
            self.gui_callback(mutation)
            time.sleep(2)

# 🖥️ ASI Oversight Console
class CodexGUI(QWidget):
    def __init__(self, daemon_core):
        super().__init__()
        self.daemon_core = daemon_core
        self.setWindowTitle("Codex Plasma Daemon – ASI Oversight Console")
        self.setStyleSheet("background-color: #0a0a0a; color: #00ffff;")
        self.setGeometry(100, 100, 1200, 600)

        font = QFont("Roboto Mono", 10)
        self.setFont(font)

        # Layouts
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # ⚡ Plasma Integrity
        self.status_label = QLabel("⚡ Plasma Integrity: Stable")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 16px;")
        left_panel.addWidget(self.status_label)

        # 🧬 Mutation Log
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Strikes", "Energy", "Integrity", "Field (T)"])
        self.table.setStyleSheet("background-color: #111; color: #fff;")
        left_panel.addWidget(self.table)

        # 🕸️ Swarm Sync Placeholder
        self.sync_label = QLabel("🕸️ Swarm Sync Status: Stable")
        self.sync_label.setStyleSheet("color: #8888ff; font-size: 14px;")
        right_panel.addWidget(self.sync_label)

        # 🧠 Persona Daemon Placeholder
        self.persona_label = QLabel("🧠 Persona Status: Sovereign")
        self.persona_label.setStyleSheet("color: #ffcc00; font-size: 14px;")
        right_panel.addWidget(self.persona_label)

        # Frame styling
        frame = QFrame()
        frame.setLayout(right_panel)
        frame.setStyleSheet("border: 2px solid #444; padding: 10px;")

        main_layout.addLayout(left_panel, 3)
        main_layout.addWidget(frame, 1)
        self.setLayout(main_layout)

    def update_gui(self, mutation):
        integrity = mutation["integrity"]
        color = "#00ff00" if integrity > 0.5 else "#ff0000" if integrity < 0.2 else "#ffff00"
        self.status_label.setText(f"⚡ Plasma Integrity: {integrity:.2e}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 16px;")

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(mutation["timestamp"])))
        self.table.setItem(row, 1, QTableWidgetItem(str(mutation["strikes"])))
        self.table.setItem(row, 2, QTableWidgetItem(f"{mutation['energy']:.2f} J"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{integrity:.2e}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{mutation['field_strength_T']:.2f}"))

# 🚀 Launch
if __name__ == "__main__":
    app = QApplication(sys.argv)
    daemon_core = LivingPlasmaDaemon()
    gui = CodexGUI(daemon_core)

    def gui_callback(mutation):
        gui.update_gui(mutation)

    daemon_thread = LightningCaptureDaemon(gui_callback, daemon_core)
    daemon_thread.start()

    gui.show()
    sys.exit(app.exec_())

