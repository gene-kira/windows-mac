import sys, math, time, threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QGridLayout,
    QVBoxLayout, QHBoxLayout, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor

# 🔁 Reverse Polarity Logic
def reverse_polarity(input_str):
    return ''.join(chr(~ord(c) & 0xFF) for c in input_str)

# 💓 Biometric Entropy Simulation
def synthetic_pulse(t):
    heart = math.sin(t * 2 * math.pi / 1.0)
    breath = math.sin(t * 2 * math.pi / 0.2)
    neural = math.sin(t * 2 * math.pi / 40)
    return heart + breath + neural

def entropy_seed():
    t = time.time()
    return int((synthetic_pulse(t) + 1) * 1000) % 256

# 🎨 Glyph Mutation Stream
class GlyphStreamWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.glyphs = ["⟡", "⧫", "⟴", "✶", "∇", "Ψ", "Σ", "⨁"]
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(QFont("Fira Code", 24))
        for i, glyph in enumerate(self.glyphs):
            color = QColor(0, entropy_seed(), 255)
            painter.setPen(color)
            painter.drawText(50 + i * 40, 100, glyph)

# 🔧 Packet Mutation Thread (Simulated)
def packet_mutation_loop(log_callback, active_flag):
    while active_flag["running"]:
        time.sleep(5)
        sample = "Hello from ASI"
        mutated = reverse_polarity(sample)
        log_callback(f"Mutated Packet: {mutated} | Entropy: {entropy_seed()}")

# 🧬 Main Console GUI
class GlyphSbitConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlyphSbit Codex Console")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #0A0A0A;")
        self.system_active = {"running": False}
        self.init_ui()
        self.init_watchdog()

    def init_ui(self):
        layout = QVBoxLayout()

        # 🔘 ON/OFF Panel
        control_layout = QHBoxLayout()
        self.status_label = QLabel("Status: ⧫ DORMANT")
        self.status_label.setStyleSheet("color: #FF0033; font-size: 16px;")

        on_button = QPushButton("⟡ ON")
        on_button.setStyleSheet("background-color: #00F0FF; font-weight: bold;")
        on_button.clicked.connect(self.activate_system)

        off_button = QPushButton("⧫ OFF")
        off_button.setStyleSheet("background-color: #FF0033; font-weight: bold;")
        off_button.clicked.connect(self.deactivate_system)

        control_layout.addWidget(on_button)
        control_layout.addWidget(off_button)
        control_layout.addWidget(self.status_label)
        layout.addLayout(control_layout)

        # 🧬 Scientific Grid Panels
        grid = QGridLayout()
        grid.setSpacing(15)

        panels = [
            ("Codex Input Panel", "#00F0FF"),
            ("Glyph Mutation Stream", "#9B00FF"),
            ("Biometric Resonance Simulator", "#FF0033"),
            ("Swarm-Sync Telemetry", "#00F0FF"),
            ("Forensic Mutation Log", "#9B00FF"),
            ("TTL Vaporizer Trigger", "#FF0033")
        ]

        for i, (title, color) in enumerate(panels):
            panel = QFrame()
            panel.setStyleSheet("background-color: #1A1A1A;")
            label = QLabel(title)
            label.setStyleSheet(f"color: {color};")
            label.setAlignment(Qt.AlignCenter)
            panel_layout = QVBoxLayout()
            panel_layout.addWidget(label)
            if title == "Glyph Mutation Stream":
                panel_layout.addWidget(GlyphStreamWidget())
            panel.setLayout(panel_layout)
            grid.addWidget(panel, i // 3, i % 3)

        layout.addLayout(grid)

        # 🧿 Console Log
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        self.console_log.setText("Autoloader initialized. Dependencies resolved.")
        layout.addWidget(self.console_log)

        self.setLayout(layout)

    def init_watchdog(self):
        self.watchdog = QTimer()
        self.watchdog.setInterval(5000)
        self.watchdog.timeout.connect(self.send_heartbeat)

    def activate_system(self):
        self.system_active["running"] = True
        self.status_label.setText("Status: ⟡ ACTIVE")
        self.status_label.setStyleSheet("color: #00F0FF; font-size: 16px;")
        self.console_log.append("System Activated. Entropy Seed Injected.")
        self.watchdog.start()
        threading.Thread(target=packet_mutation_loop, args=(self.console_log.append, self.system_active), daemon=True).start()

    def deactivate_system(self):
        self.system_active["running"] = False
        self.status_label.setText("Status: ⧫ DORMANT")
        self.status_label.setStyleSheet("color: #FF0033; font-size: 16px;")
        self.console_log.append("System Halted. Glyphs Frozen.")
        self.watchdog.stop()

    def send_heartbeat(self):
        if self.system_active["running"]:
            self.console_log.append("Watchdog heartbeat: system integrity stable.")

# 🧪 Entry Point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    console = GlyphSbitConsole()
    console.show()
    sys.exit(app.exec_())

