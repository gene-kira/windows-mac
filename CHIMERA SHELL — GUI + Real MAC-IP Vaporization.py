# CHIMERA SHELL — GUI + Real MAC/IP Vaporization
import sys, time, uuid, random, socket, hashlib, json, platform, subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QProgressBar, QPushButton
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer

# 🧯 Real MAC/IP Vaporization
def get_mac_ip():
    try:
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0, 8*6, 8)][::-1])
        ip = socket.gethostbyname(socket.gethostname())
    except:
        mac, ip = "00:00:00:00:00:00", "127.0.0.1"
    return mac, ip

def vaporize_mac_ip():
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.call(["sudo", "ifconfig", "eth0", "down"])
            subprocess.call(["sudo", "ifconfig", "eth0", "hw", "ether", "00:11:22:33:44:55"])
            subprocess.call(["sudo", "ifconfig", "eth0", "up"])
            subprocess.call(["sudo", "dhclient", "-r"])
            subprocess.call(["sudo", "dhclient"])
        elif system == "Windows":
            subprocess.call(["ipconfig", "/release"], shell=True)
            subprocess.call(["ipconfig", "/renew"], shell=True)
        return "MAC/IP vaporized successfully."
    except Exception as e:
        return f"Vaporization failed: {str(e)}"

# 🧠 Capsule + Persona Injection
class Capsule:
    def __init__(self, payload):
        self.id = str(uuid.uuid4())
        self.payload = payload
        self.glyph = hashlib.sha256((payload + str(time.time())).encode()).hexdigest()

def decode_capsule(capsule):
    return {
        "persona": f"persona_{random.randint(1,5)}",
        "mutation": [uuid.uuid4().hex[:8] for _ in range(3)],
        "swarm": f"node_{random.randint(100,999)}"
    }

# 🖥️ GUI Console
class ChimeraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CHIMERA — ASI Oversight")
        self.setGeometry(100, 100, 900, 650)
        self.export_log = []
        self.mac, self.ip = get_mac_ip()

        # GUI Styling
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(15, 15, 30))
        pal.setColor(QPalette.WindowText, QColor(0, 255, 180))
        self.setPalette(pal)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("CHIMERA: Autonomous ASI Oversight", font=QFont("Consolas", 20, QFont.Bold), alignment=Qt.AlignCenter))

        self.feed = QTextEdit(); self.feed.setFont(QFont("Courier", 10)); self.feed.setStyleSheet("background-color:#111;color:#0f0;"); layout.addWidget(self.feed)
        self.country = QTextEdit(); self.country.setFont(QFont("Courier", 10)); self.country.setStyleSheet("background-color:#111;color:#f00;"); layout.addWidget(self.country)
        self.persona = QTextEdit(); self.persona.setFont(QFont("Courier", 10)); self.persona.setStyleSheet("background-color:#111;color:#ff0;"); layout.addWidget(self.persona)

        self.bar = QProgressBar(); self.bar.setMaximum(100); self.bar.setValue(0); self.bar.setStyleSheet("QProgressBar::chunk { background-color: #0ff; }"); layout.addWidget(self.bar)
        self.status = QLabel("Status: Initializing..."); self.status.setFont(QFont("Consolas", 10)); self.status.setStyleSheet("color: #0ff;"); layout.addWidget(self.status)

        self.vaporize_btn = QPushButton(f"Vaporize MAC/IP ({self.mac} / {self.ip})")
        self.vaporize_btn.setFont(QFont("Consolas", 10)); self.vaporize_btn.setStyleSheet("background-color:#222;color:#f0f0f0;")
        self.vaporize_btn.clicked.connect(self.vaporize_mac_ip_gui)
        layout.addWidget(self.vaporize_btn)

        self.setLayout(layout)

        # Start mutation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_mutation_cycle)
        self.timer.start(5000)

    def vaporize_mac_ip_gui(self):
        result = vaporize_mac_ip()
        self.status.setText(f"Vaporization: {result}")
        self.vaporize_btn.setText("Vaporize MAC/IP (done)")

    def run_mutation_cycle(self):
        try:
            telemetry = {
                "cpu": random.randint(0, 100),
                "ram": random.randint(0, 100),
                "temp": round(random.uniform(30, 90), 2),
                "timestamp": time.time()
            }
            capsule = Capsule(str(telemetry))
            decoded = decode_capsule(capsule)
            timestamp = time.strftime('%H:%M:%S')

            self.feed.append(f"[{timestamp}] Capsule {capsule.id} → Glyph {capsule.glyph[:12]}...")
            self.country.append(f"[{timestamp}] Outbound: {self.ip} → TTL 30s")
            self.persona.append(f"[{timestamp}] Persona: {decoded['persona']} | Swarm: {decoded['swarm']}")
            self.status.setText(f"Status: Capsule {capsule.id} mutated at {timestamp}")

            current = self.bar.value()
            self.bar.setValue(0 if current >= 100 else current + random.randint(5, 15))

            # Trim buffers
            for box in [self.feed, self.country, self.persona]:
                if box.document().blockCount() > 100:
                    box.clear()

            # Export every 50 cycles
            self.export_log.append({
                "timestamp": timestamp,
                "capsule_id": capsule.id,
                "glyph": capsule.glyph,
                "persona": decoded['persona'],
                "swarm": decoded['swarm']
            })
            if len(self.export_log) >= 50:
                with open("chimera_capsules.json", "w") as f:
                    json.dump(self.export_log[-50:], f, indent=2)
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")

# 🚀 INIT
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ChimeraGUI()
    gui.show()
    sys.exit(app.exec_())

