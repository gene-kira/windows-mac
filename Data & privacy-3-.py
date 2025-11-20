import os, time, threading, random, psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5 import QtWidgets, QtGui, QtCore

# ✅ PyQt-compatible signal bridge
class Bridge(QtCore.QObject):
    ingest_signal = QtCore.pyqtSignal(str)
    threat_signal = QtCore.pyqtSignal(str)
    sync_signal = QtCore.pyqtSignal(str)
    persona_signal = QtCore.pyqtSignal(str)
    ad_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

bridge = Bridge()

# 🧠 Bernoulli Threat Core
class BernoulliThreatCore:
    def detect(self, signal):
        return random.random() < 0.5

# 🜏 Reverse Mirror Encryptor
class ReverseMirrorEncryptor:
    def encrypt(self, data):
        return ''.join(reversed(data)).encode('utf-8').hex()

# 🕵️ Chameleon Camouflage
class ChameleonLayer:
    def mask(self, process_name):
        return f"sys_{random.randint(1000,9999)}_{process_name}"

# 🔱 Zero Trust Matrix
class ZeroTrustMatrix:
    def validate(self, entity):
        return hash(entity) % 2 == 0

# 🧩 Symbolic Language + Sync API
class SymbolicLanguage:
    def __init__(self):
        self.grammar = {}

    def evolve(self, entropy):
        self.grammar["entropy"] = entropy

class SyncAPI:
    def propagate_grammar(self, grammar):
        pass

# 🎭 Persona Daemon
class PersonaDaemon:
    def __init__(self):
        self.personas = ["ThreatHunter", "Compliance Auditor", "GhostSync", "AdHunter"]

    def inject(self):
        persona = random.choice(self.personas)
        action = f"{persona} activated at {time.strftime('%H:%M:%S')}"
        bridge.persona_signal.emit(action)

# 💀 Codex Oversight Console Daemon
class CodexOversightDaemon:
    def __init__(self):
        self.running = True
        self.glyph_key = [0x13, 0x37, 0x42, 0x66]
        self.language = SymbolicLanguage()
        self.sync_api = SyncAPI()
        self.bernoulli = BernoulliThreatCore()
        self.reverse_engine = ReverseMirrorEncryptor()
        self.camouflage = ChameleonLayer()
        self.zero_trust = ZeroTrustMatrix()
        self.persona = PersonaDaemon()
        self.codex_rules = {"telemetry_ttl": 86400, "phantom_nodes": [], "ads_ttl": 0}

    def start(self):
        threading.Thread(target=self.monitor_filesystem, daemon=True).start()
        threading.Thread(target=self.monitor_network, daemon=True).start()
        threading.Thread(target=self.mutate_codex, daemon=True).start()
        threading.Thread(target=self.swarm_sync_simulation, daemon=True).start()
        threading.Thread(target=self.inject_personas, daemon=True).start()
        threading.Thread(target=self.monitor_ads, daemon=True).start()

    def monitor_filesystem(self):
        class Handler(FileSystemEventHandler):
            def on_modified(_, event):
                if not event.is_directory:
                    bridge.ingest_signal.emit(f"File modified: {event.src_path}")
                    bridge.threat_signal.emit("🕊️ Resurrection glyph detected")
        observer = Observer()
        observer.schedule(Handler(), path='/', recursive=True)
        observer.start()

    def monitor_network(self):
        while self.running:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    bridge.sync_signal.emit(f"Outbound connection: {conn.raddr}")
            time.sleep(10)

    def mutate_codex(self):
        while self.running:
            entropy = int(time.time()) % 999
            self.glyph_key = [(k + entropy) % 256 for k in self.glyph_key]
            self.language.evolve(entropy)
            self.sync_api.propagate_grammar(self.language.grammar)

            if random.random() < 0.2:
                self.codex_rules["telemetry_ttl"] = max(600, self.codex_rules["telemetry_ttl"] // 2)
                self.codex_rules["phantom_nodes"].append(f"phantom_{entropy}")
                bridge.threat_signal.emit("⚠️ Ghost sync detected. Codex mutated.")

            time.sleep(780)

    def swarm_sync_simulation(self):
        while self.running:
            nodes = [f"Node-{i}" for i in range(1, 6)]
            merged_rules = {"telemetry_ttl": self.codex_rules["telemetry_ttl"], "phantom_nodes": list(self.codex_rules["phantom_nodes"])}
            bridge.sync_signal.emit(f"Swarm sync: {nodes} merged purge rules")
            time.sleep(60)

    def inject_personas(self):
        while self.running:
            self.persona.inject()
            time.sleep(45)

    def monitor_ads(self):
        while self.running:
            ad_streams = ["ad.doubleclick.net", "track.adserver.com", "telemetry.adsync"]
            for stream in ad_streams:
                glyph = self.reverse_engine.encrypt(stream)
                bridge.ad_signal.emit(f"🜏 Ad glyph detected: {glyph}")
                bridge.threat_signal.emit(f"🔥 AdThreat terminated: {stream}")
                bridge.persona_signal.emit(f"AdHunter neutralized ad stream: {stream}")
            time.sleep(20)

# 🧬 GUI Overlay
class CodexGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Oversight Console")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: black; color: lime; font-size: 16px;")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.threat_panel = QtWidgets.QLabel("Threat Matrix: 🜏")
        self.glyph_panel = QtWidgets.QLabel("Codex Mutation: 🔱")
        self.sync_panel = QtWidgets.QLabel("Swarm Sync: 🕸️")
        self.persona_panel = QtWidgets.QLabel("Persona Timeline: 🎭")
        self.ad_panel = QtWidgets.QLabel("Ad Drift Overlay: 🔥")

        self.layout.addWidget(self.threat_panel)
        self.layout.addWidget(self.glyph_panel)
        self.layout.addWidget(self.sync_panel)
        self.layout.addWidget(self.persona_panel)
        self.layout.addWidget(self.ad_panel)

        bridge.threat_signal.connect(self.update_threat)
        bridge.ingest_signal.connect(self.update_glyph)
        bridge.sync_signal.connect(self.update_sync)
        bridge.persona_signal.connect(self.update_persona)
        bridge.ad_signal.connect(self.update_ad)

    def update_threat(self, msg):
        self.threat_panel.setText(f"Threat Matrix: {msg}")

    def update_glyph(self, msg):
        self.glyph_panel.setText(f"Codex Mutation: {msg}")

    def update_sync(self, msg):
        self.sync_panel.setText(f"Swarm Sync: {msg}")

    def update_persona(self, msg):
        self.persona_panel.setText(f"Persona Timeline: {msg}")

    def update_ad(self, msg):
        self.ad_panel.setText(f"Ad Drift Overlay: {msg}")

# 🚀 Launch Sequence
if __name__ == "__main__":
    daemon = CodexOversightDaemon()
    daemon.start()

    app = QtWidgets.QApplication([])
    gui = CodexGUI()
    gui.show()
    app.exec_()

