import sys, math, time, threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor

# 🔁 Reverse Polarity Logic
def reverse_polarity(input_str):
    return ''.join(chr(~ord(c) & 0xFF) for c in input_str)

# 💓 Biometric Sensor Simulation
def read_biometric_data():
    t = time.time()
    return {
        "heart_rate": int((math.sin(t) + 1) * 40 + 60),
        "breath_rate": int((math.sin(t / 2) + 1) * 10 + 12),
        "neural_spike": round(math.sin(t / 40), 3)
    }

# 📡 Hardware Telemetry Simulation
def fetch_hardware_telemetry():
    return f"📡 NPU Temp={entropy_seed()}°C | Load={entropy_seed()}%"

# 💓 Entropy Seed
def synthetic_pulse(t):
    heart = math.sin(t * 2 * math.pi / 1.0)
    breath = math.sin(t * 2 * math.pi / 0.2)
    neural = math.sin(t * 2 * math.pi / 40)
    return heart + breath + neural

def entropy_seed():
    t = time.time()
    return int((synthetic_pulse(t) + 1) * 1000) % 256

# 🔧 Packet Mutation Thread
def packet_mutation_loop(log_callback, active_flag):
    while active_flag["running"]:
        time.sleep(5)
        sample = "Hello from ASI"
        mutated = reverse_polarity(sample)
        bio = read_biometric_data()
        log_callback(f"🔁 Mutated Packet: {mutated} | Entropy: {entropy_seed()}")
        log_callback(f"💓 Biometric: HR={bio['heart_rate']} BR={bio['breath_rate']} NS={bio['neural_spike']}")
        log_callback(fetch_hardware_telemetry())
        log_callback("🔊 Sonic Glyph Feedback: Resonance ping emitted.")

# 💣 TTL Countdown Overlay
def vaporize_payload(log_callback):
    log_callback("⚠️ TTL Vaporizer Triggered. Payload self-destruct in 3...")
    QTimer.singleShot(1000, lambda: log_callback("2..."))
    QTimer.singleShot(2000, lambda: log_callback("1..."))
    QTimer.singleShot(3000, lambda: log_callback(f"💥 Payload vaporized. Entropy spike: {entropy_seed()}"))

# 🧬 Main Console GUI
class GlyphSbitConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlyphSbit Codex Console")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #0A0A0A;")
        self.system_active = {"running": False}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 🔘 Control Panel
        control_layout = QHBoxLayout()
        self.status_label = QLabel("Status: ⧫ DORMANT")
        self.status_label.setStyleSheet("color: #FF0033; font-size: 16px;")

        on_button = QPushButton("⟡ ON")
        on_button.setStyleSheet("background-color: #00F0FF; font-weight: bold;")
        on_button.clicked.connect(self.activate_system)

        off_button = QPushButton("⧫ OFF")
        off_button.setStyleSheet("background-color: #FF0033; font-weight: bold;")
        off_button.clicked.connect(self.deactivate_system)

        vapor_button = QPushButton("💣 TTL")
        vapor_button.setStyleSheet("background-color: #9B00FF; font-weight: bold;")
        vapor_button.clicked.connect(lambda: vaporize_payload(self.console_log.append))

        control_layout.addWidget(on_button)
        control_layout.addWidget(off_button)
        control_layout.addWidget(vapor_button)
        control_layout.addWidget(self.status_label)
        layout.addLayout(control_layout)

        # 🧿 Console Log
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setStyleSheet("background-color: #000000; color: #FFFFFF;")
        self.console_log.setText("Autoloader initialized. Dependencies resolved.")
        layout.addWidget(self.console_log)

        self.setLayout(layout)

    def activate_system(self):
        self.system_active["running"] = True
        self.status_label.setText("Status: ⟡ ACTIVE")
        self.status_label.setStyleSheet("color: #00F0FF; font-size: 16px;")
        self.console_log.append("System Activated. Entropy Seed Injected.")
        self.console_log.append("🧬 Persona Injection: Codex identity mutated.")
        threading.Thread(target=self.daemon_watchdog, daemon=True).start()
        threading.Thread(target=packet_mutation_loop, args=(self.console_log.append, self.system_active), daemon=True).start()

    def deactivate_system(self):
        self.system_active["running"] = False
        self.status_label.setText("Status: ⧫ DORMANT")
        self.status_label.setStyleSheet("color: #FF0033; font-size: 16px;")
        self.console_log.append("System Halted. Glyphs Frozen.")

    def daemon_watchdog(self):
        while self.system_active["running"]:
            time.sleep(5)
            self.console_log.append("🧿 Daemon Heartbeat: System integrity stable.")

# 🧪 Entry Point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    console = GlyphSbitConsole()
    console.show()
    sys.exit(app.exec_())

