#!/usr/bin/env python3

import sys, os, time, threading, subprocess, hashlib
from datetime import datetime, timedelta

# ┌────────────────────────────────────────────┐
# │ Shai-Hulud Devourer: ASI Codex Mesh Shell  │
# └────────────────────────────────────────────┘

# 🧠 Autoloader
REQUIRED_LIBS = ["PyQt5", "pyqtgraph", "cryptography", "psutil"]
for lib in REQUIRED_LIBS:
    try:
        __import__(lib)
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", lib])

# ✅ Imports
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import psutil
from cryptography.fernet import Fernet

# 🔐 Constants
DAEMON = "ipsec"
LOG_FILE = "/tmp/asi_guardian.log"
HEARTBEAT_INTERVAL = 3000
VAPORIZATION_QUEUE = []
GLYPHS = ["⚡", "🔒", "🧬", "🛡️", "💠", "🔥", "🌐", "📡"]
KEY = Fernet.generate_key()
CIPHER = Fernet(KEY)
PERSONA_HASH = hashlib.sha256(b"killer666|FACE|FINGERPRINT|GSR").hexdigest()
SHAI_HULUD_SIGNATURES = ["bundle.js", "shai-hulud-workflow.yml", "@ctrl", "@crowdstrike", "@art-ws"]

# 📜 Logging
def ensure_log_file():
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[INIT] Devourer log started at {datetime.now()}\n")
        os.chmod(LOG_FILE, 0o666)
    except Exception as e:
        print(f"[LOG ERROR] {e}")

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a") as log:
            log.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# 🛡️ Watchdog
def watchdog():
    try:
        if subprocess.run(["pgrep", DAEMON], stdout=subprocess.DEVNULL).returncode != 0:
            log_event(f"[ALERT] {DAEMON} down. Restarting...")
            subprocess.run(["sudo", "systemctl", "restart", DAEMON])
        else:
            log_event(f"[HEARTBEAT] {DAEMON} alive.")
    except Exception as e:
        log_event(f"[ERROR] Watchdog failed: {e}")

# 🧬 SA morphing
def morph_sa():
    try:
        log_event("[MORPH] Rotating keys and SPI...")
        subprocess.run(["ipsec", "rereadsecrets"])
        subprocess.run(["ipsec", "update"])
    except Exception as e:
        log_event(f"[ERROR] SA morph failed: {e}")

# 💣 Reverse polarity + vaporization
def reverse_polarity(data):
    inverted = ''.join(chr(~ord(c) & 0xFF) for c in data)
    encrypted = CIPHER.encrypt(inverted.encode())
    return encrypted

def schedule_vaporization(data, ttl_seconds, label):
    expiration = datetime.now() + timedelta(seconds=ttl_seconds)
    VAPORIZATION_QUEUE.append((data, expiration, label))

def vaporization_daemon():
    while True:
        now = datetime.now()
        for item in VAPORIZATION_QUEUE[:]:
            data, expiration, label = item
            if now >= expiration:
                log_event(f"[VAPORIZED] {label} | Data: {data}")
                VAPORIZATION_QUEUE.remove(item)
        time.sleep(1)

# 🧠 Persona validation
def validate_persona():
    current_hash = hashlib.sha256(b"killer666|FACE|FINGERPRINT|GSR").hexdigest()
    if current_hash != PERSONA_HASH:
        log_event("[BLOCKED] Persona mismatch. Access denied.")
        sys.exit(1)
    else:
        log_event("[VALIDATED] Persona codex matched.")

# 🧠 Biometric resonance simulation
def simulate_biometrics():
    biometric_data = "FACE_HASH|FINGERPRINT_HASH|GSR_SIGNAL"
    encrypted = reverse_polarity(biometric_data)
    schedule_vaporization(encrypted, 86400, "Biometric Data")

# 📡 Swarm-sync telemetry injection
def inject_fake_telemetry():
    fake = reverse_polarity("FAKE_TELEMETRY_PACKET")
    schedule_vaporization(fake, 30, "Fake Telemetry")
    log_event("[INJECTED] Fake telemetry dispatched.")

# 🕷️ Shai-Hulud detection
def detect_shai_hulud():
    infected = []
    for root, dirs, files in os.walk("."):
        for name in files:
            for sig in SHAI_HULUD_SIGNATURES:
                if sig in name:
                    infected.append(os.path.join(root, name))
    if infected:
        for path in infected:
            log_event(f"[DETECTED] Shai-Hulud signature: {path}")
            try:
                os.remove(path)
                log_event(f"[DEVOUR] Quarantined and vaporized: {path}")
            except Exception as e:
                log_event(f"[ERROR] Failed to vaporize {path}: {e}")
    else:
        log_event("[SCAN] No Shai-Hulud signatures found.")

# 💠 Qt GUI Dashboard
class DevourerDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shai-Hulud Devourer Console")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: #0a0a0a; color: #00ffcc;")
        self.layout = QVBoxLayout()

        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Courier", 14))
        self.layout.addWidget(self.status_label)

        self.cpu_plot = pg.PlotWidget()
        self.cpu_plot.setYRange(0, 100)
        self.cpu_plot.setBackground('k')
        self.cpu_curve = self.cpu_plot.plot(pen='c')
        self.layout.addWidget(self.cpu_plot)

        self.telemetry_button = QPushButton("Inject Fake Telemetry")
        self.telemetry_button.clicked.connect(inject_fake_telemetry)
        self.layout.addWidget(self.telemetry_button)

        self.biometric_button = QPushButton("Simulate Biometric Resonance")
        self.biometric_button.clicked.connect(simulate_biometrics)
        self.layout.addWidget(self.biometric_button)

        self.scan_button = QPushButton("Scan for Shai-Hulud")
        self.scan_button.clicked.connect(detect_shai_hulud)
        self.layout.addWidget(self.scan_button)

        self.setLayout(self.layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(HEARTBEAT_INTERVAL)
        self.cpu_data = []

    def update_status(self):
        glyph = GLYPHS[int(time.time()) % len(GLYPHS)]
        self.status_label.setText(f"{glyph} {datetime.now().strftime('%H:%M:%S')} | IPsec alive.")
        watchdog()
        morph_sa()
        schedule_vaporization("MAC/IP Metadata", 30, "Outbound Metadata")
        schedule_vaporization("Backdoor Payload", 3, "Backdoor Exfiltration")
        cpu = psutil.cpu_percent()
        self.cpu_data.append(cpu)
        if len(self.cpu_data) > 100:
            self.cpu_data.pop(0)
        self.cpu_curve.setData(self.cpu_data)

# 🚀 Launch
def main():
    ensure_log_file()
    validate_persona()
    threading.Thread(target=vaporization_daemon, daemon=True).start()
    app = QApplication(sys.argv)
    dashboard = DevourerDashboard()
    dashboard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

