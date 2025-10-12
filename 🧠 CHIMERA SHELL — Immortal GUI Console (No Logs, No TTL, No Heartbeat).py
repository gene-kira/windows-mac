import sys, time, uuid, random, socket, platform, subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QProgressBar, QPushButton
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer

def get_mac_ip():
    try: return ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0, 8*6, 8)][::-1]), socket.gethostbyname(socket.gethostname())
    except: return "00:00:00:00:00:00", "127.0.0.1"

def vaporize_mac_ip():
    try:
        if platform.system() == "Linux":
            for cmd in [["ifconfig", "eth0", "down"], ["ifconfig", "eth0", "hw", "ether", "00:11:22:33:44:55"], ["ifconfig", "eth0", "up"], ["dhclient", "-r"], ["dhclient"]]:
                subprocess.call(["sudo"] + cmd)
        elif platform.system() == "Windows":
            subprocess.call(["ipconfig", "/release"], shell=True)
            subprocess.call(["ipconfig", "/renew"], shell=True)
        return "MAC/IP vaporized."
    except Exception as e: return f"Failed: {e}"

def mutate_capsule(payload):
    return {
        "id": str(uuid.uuid4()),
        "glyph": uuid.uuid4().hex[:12],
        "biometric": {k: uuid.uuid4().hex[:16] for k in ["fingerprint", "faceprint", "retina"]},
        "persona": f"persona_{random.randint(1,5)}",
        "dialect": random.choice(["vX.0", "vY.7", "vZ.3"]),
        "swarm": [f"node_{random.randint(100,999)}" for _ in range(3)],
        "inference": [f"{chip} ✓" for chip in ["NCS2", "Coral", "Hailo"]],
        "timestamp": time.strftime('%H:%M:%S')
    }

class Chimera(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CHIMERA — Mutation Mesh")
        self.setGeometry(100, 100, 950, 720)
        self.mac, self.ip = get_mac_ip()

        pal = QPalette(); pal.setColor(QPalette.Window, QColor(15,15,30)); pal.setColor(QPalette.WindowText, QColor(0,255,180)); self.setPalette(pal)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("CHIMERA: Mutation Mesh Console", font=QFont("Consolas", 20), alignment=Qt.AlignCenter))

        self.boxes = {k: QTextEdit() for k in ["feed", "country", "persona", "swarm"]}
        for k, box in self.boxes.items():
            box.setFont(QFont("Courier", 10)); box.setStyleSheet(f"background-color:#111;color:{'#0f0' if k=='feed' else '#f00' if k=='country' else '#ff0' if k=='persona' else '#0ff'};")
            layout.addWidget(box)

        self.bar = QProgressBar(); self.bar.setMaximum(100); layout.addWidget(self.bar)
        self.status = QLabel("Status: Initializing..."); self.status.setFont(QFont("Consolas", 10)); self.status.setStyleSheet("color: #0ff;"); layout.addWidget(self.status)

        self.vaporize_btn = QPushButton(f"Vaporize MAC/IP ({self.mac} / {self.ip})")
        self.vaporize_btn.clicked.connect(self.vaporize); layout.addWidget(self.vaporize_btn)

        self.setLayout(layout)
        QTimer.singleShot(0, self.start)

    def vaporize(self):
        self.status.setText(vaporize_mac_ip()); self.vaporize_btn.setText("Vaporized")

    def start(self):
        self.mutate_timer = QTimer(); self.mutate_timer.timeout.connect(self.mutate); self.mutate_timer.start(5000)

    def mutate(self):
        try:
            capsule = mutate_capsule(str({"cpu":random.randint(0,100),"ram":random.randint(0,100),"temp":round(random.uniform(30,90),2)}))
            self.boxes["feed"].append(f"[{capsule['timestamp']}] Capsule {capsule['id']} → Glyph {capsule['glyph']}")
            self.boxes["country"].append(f"[{capsule['timestamp']}] Outbound: {self.ip} → TTL 30s")
            self.boxes["persona"].append(f"[{capsule['timestamp']}] Persona: {capsule['persona']} | Dialect: {capsule['dialect']}")
            self.boxes["swarm"].append(f"[{capsule['timestamp']}] Swarm: {', '.join(capsule['swarm'])} | Inference: {', '.join(capsule['inference'])}")
            self.status.setText(f"Mutation: {capsule['id']} | Biometric injected")
            self.bar.setValue(0 if self.bar.value() >= 100 else self.bar.value() + random.randint(5,15))
            for box in self.boxes.values():
                if box.document().blockCount() > 100: box.clear()
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv); gui = Chimera(); gui.show(); sys.exit(app.exec_())

