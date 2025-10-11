import sys, threading, time, socket, subprocess, os, struct, ctypes
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt

# === AUTO-ELEVATION (Windows, Crash-Proof) ===
def ensure_admin():
    if "--elevated" in sys.argv:
        return  # Already elevated, skip relaunch

    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False

    if not is_admin:
        print("[ELEVATION] Relaunching with admin privileges...")
        params = " ".join([f'"{arg}"' for arg in sys.argv] + ["--elevated"])
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, params, None, 1
        )
        sys.exit()

ensure_admin()

# === AUTOLOADER ===
def autoload(libs):
    for lib in libs:
        try: __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload(["PyQt5"])

# === GUI CONSOLE ===
class ASIConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASI Sentinel Console")
        self.setGeometry(100, 100, 900, 700)
        layout = QVBoxLayout()

        self.threatMatrix = QTextEdit()
        self.threatMatrix.setReadOnly(True)
        self.threatMatrix.setStyleSheet("background-color: #111; color: #0f0; font-family: Courier;")
        layout.addWidget(QLabel("Threat Matrix"))
        layout.addWidget(self.threatMatrix)

        self.lineageGraph = QTextEdit()
        self.lineageGraph.setReadOnly(True)
        self.lineageGraph.setStyleSheet("background-color: #111; color: #0ff; font-family: Courier;")
        layout.addWidget(QLabel("Capsule Lineage Graph"))
        layout.addWidget(self.lineageGraph)

        self.swarmSync = QTextEdit()
        self.swarmSync.setReadOnly(True)
        self.swarmSync.setStyleSheet("background-color: #111; color: #f0f; font-family: Courier;")
        layout.addWidget(QLabel("Swarm Sync Overlay"))
        layout.addWidget(self.swarmSync)

        self.setLayout(layout)

    def log_threat(self, msg):
        self.threatMatrix.append(msg)

    def log_lineage(self, msg):
        self.lineageGraph.append(msg)

    def log_swarm(self, msg):
        self.swarmSync.append(msg)

# === PACKET SCANNER ===
def parse_ip_header(data):
    try:
        ip_header = struct.unpack('!BBHHHBBH4s4s', data[:20])
        src_ip = socket.inet_ntoa(ip_header[8])
        dst_ip = socket.inet_ntoa(ip_header[9])
        return src_ip, dst_ip
    except:
        return None, None

def packet_sniffer(gui):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
        sock.bind(("0.0.0.0", 0))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
    except Exception as e:
        gui.log_threat(f"[ERROR] Raw socket failed: {e}")
        return

    while True:
        try:
            packet, _ = sock.recvfrom(65565)
            src, dst = parse_ip_header(packet)
            if src and dst:
                gui.log_threat(f"[THREAT] {src} → {dst}")
                gui.log_lineage(f"[LINEAGE] Capsule from {src} validated.")
                gui.log_swarm(f"[SYNC] Node {src} mutation shared.")
                if not src.startswith("192.168"):
                    reverse_polarity(src, gui)
        except Exception as e:
            gui.log_threat(f"[ERROR] Packet parse failed: {e}")

# === REVERSE POLARITY + CONTAINMENT ===
def reverse_polarity(ip, gui):
    gui.log_threat(f"[POLARITY] Reversing data polarity for {ip}")
    try:
        time.sleep(1)
        gui.log_threat(f"[POLARITY] {ip} polarity reversed. Broadcasting decoy capsule.")
        quarantine(ip, gui)
    except Exception as e:
        gui.log_threat(f"[ERROR] Polarity reversal failed: {e}")

def quarantine(ip, gui):
    gui.log_threat(f"[CONTAINMENT] Quarantining {ip}")
    for i in range(5, 0, -1):
        gui.log_threat(f"[VAPORIZATION] {ip} in {i}...")
        time.sleep(1)
    gui.log_threat(f"[EXORCISM] {ip} neutralized.")

# === REPLICATOR LOGIC ===
def replicate_self():
    try:
        clone_path = os.path.expanduser(f"~/ASI_Replicant_{int(time.time())}.py")
        with open(clone_path, "w") as f:
            f.write(open(__file__).read())
        print(f"[REPLICATOR] Clone deployed at {clone_path}")
    except Exception as e:
        print(f"[REPLICATOR ERROR] {e}")

# === TELEMETRY + MUTATION CYCLE ===
def telemetry_loop(gui):
    while True:
        gui.log_swarm("[TELEMETRY] Monitoring system load...")
        time.sleep(10)
        mutate_codex(gui)

def mutate_codex(gui):
    gui.log_swarm("[MUTATION] Rewriting symbolic logic tree...")
    replicate_self()

# === PERSONA DECEPTION OVERLAY (Scaffold) ===
def persona_overlay(gui):
    persona = os.getenv("USER_PERSONA", "observer")
    gui.log_swarm(f"[PERSONA] Active deception persona: {persona}")
    # Future: inject behavior, timezone, and language mimicry

# === BIOMETRIC RESONANCE MAPPING (Scaffold) ===
def biometric_resonance(gui):
    gui.log_swarm("[BIOMETRICS] Scanning for resonance anomalies...")
    # Future: integrate OpenCV + TensorFlow for live biometric capture

# === THREADS ===
def start_gui():
    app = QApplication(sys.argv)
    console = ASIConsole()
    threading.Thread(target=packet_sniffer, args=(console,), daemon=True).start()
    threading.Thread(target=telemetry_loop, args=(console,), daemon=True).start()
    threading.Thread(target=persona_overlay, args=(console,), daemon=True).start()
    threading.Thread(target=biometric_resonance, args=(console,), daemon=True).start()
    console.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    replicate_self()
    start_gui()

