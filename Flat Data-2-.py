# ─── AUTLOADER ────────────────────────────────────────────────────────────────
import importlib, subprocess, sys
def autoload(modules):
    for mod in modules:
        try: importlib.import_module(mod)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", mod])
autoload(["psutil", "watchdog", "PyQt6", "base91"])

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import os, time, threading, random, base91
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QTextEdit, QPushButton, QGridLayout
)
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import pyqtSignal, QObject, QTimer

# ─── SIGNAL BRIDGE ────────────────────────────────────────────────────────────
class TelemetryBridge(QObject):
    ingest_signal = pyqtSignal(str)
    sync_signal = pyqtSignal(str)
    threat_signal = pyqtSignal(str)

bridge = TelemetryBridge()

# ─── SYMBOLIC LANGUAGE ENGINE ─────────────────────────────────────────────────
class SymbolicLanguage:
    def __init__(self):
        self.glyphs = ['🜏', '⚔️', '🛡️', '💀', '🔱', '🕊️']
        self.grammar = {}

    def evolve(self, entropy_seed):
        for g in self.glyphs:
            self.grammar[g] = ''.join(random.choices(self.glyphs, k=(entropy_seed % 5) + 1))

    def translate(self, encoded):
        return ''.join(self.grammar.get(c, c) for c in encoded)

# ─── DEVOURER DAEMON ─────────────────────────────────────────────────────────
class DevourerDaemon:
    def __init__(self):
        self.running = True
        self.glyph_key = [0x13, 0x37, 0x42, 0x66]
        self.language = SymbolicLanguage()

    def start(self):
        threading.Thread(target=self.monitor_filesystem, daemon=True).start()
        threading.Thread(target=self.monitor_network, daemon=True).start()
        threading.Thread(target=self.mutate_glyph_language, daemon=True).start()

    def monitor_filesystem(self):
        class Handler(FileSystemEventHandler):
            def on_modified(_, event):
                if not event.is_directory:
                    msg = f"File modified: {event.src_path}"
                    bridge.ingest_signal.emit(msg)
                    bridge.threat_signal.emit("🕊️ Resurrection glyph detected")
        observer = Observer()
        observer.schedule(Handler(), path='/', recursive=True)
        observer.start()

    def monitor_network(self):
        while self.running:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    msg = f"Outbound connection: {conn.raddr}"
                    bridge.sync_signal.emit(msg)
            time.sleep(10)

    def mutate_glyph_language(self):
        while self.running:
            entropy = int(time.time()) % 999
            self.glyph_key = [(k + entropy) % 256 for k in self.glyph_key]
            self.language.evolve(entropy)
            time.sleep(780)

# ─── THREAT MATRIX GRID ──────────────────────────────────────────────────────
class ThreatMatrixGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout()
        self.cells = [[QLabel("🜏") for _ in range(12)] for _ in range(12)]
        for i in range(12):
            for j in range(12):
                self.grid.addWidget(self.cells[i][j], i, j)
        self.setLayout(self.grid)
        self.timer = QTimer()
        self.timer.timeout.connect(self.pulse)
        self.timer.start(3000)
        bridge.threat_signal.connect(self.trigger_glyph_storm)

    def pulse(self):
        for i in range(12):
            for j in range(12):
                glyph = random.choices(["🜏", "💀", "🔱", "🕊️"], weights=[85, 5, 5, 5])[0]
                self.cells[i][j].setText(glyph)

    def trigger_glyph_storm(self, _):
        for _ in range(3):
            self.pulse()

# ─── GUI CONSOLE ─────────────────────────────────────────────────────────────
class ASIConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Devourer Console")
        self.resize(1000, 700)
        self.setAutoFillBackground(True)
        self.apply_camouflage()

        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_ingest_panel(), "🜏 Ingest")
        self.tabs.addTab(self.create_sync_panel(), "🔱 Sync")
        self.tabs.addTab(self.create_threat_panel(), "💀 Threat Matrix")
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def apply_camouflage(self):
        hue = random.randint(0, 359)
        color = QColor.fromHsv(hue, 255, 180)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        self.setPalette(palette)

    def create_ingest_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🜏 Flat Data Ingest Stream"))
        self.ingest_log = QTextEdit()
        self.ingest_log.setReadOnly(True)
        layout.addWidget(self.ingest_log)
        panel.setLayout(layout)
        bridge.ingest_signal.connect(self.log_ingest)
        return panel

    def create_sync_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🔱 Swarm Sync Simulation"))
        self.sync_log = QTextEdit()
        self.sync_log.setReadOnly(True)
        layout.addWidget(self.sync_log)
        panel.setLayout(layout)
        bridge.sync_signal.connect(self.log_sync)
        return panel

    def create_threat_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("💀 Threat Matrix Drift"))
        self.threat_grid = ThreatMatrixGrid()
        layout.addWidget(self.threat_grid)
        panel.setLayout(layout)
        return panel

    def log_ingest(self, msg): self.ingest_log.append(f"🜏 {msg}")
    def log_sync(self, msg): self.sync_log.append(f"🔱 {msg}")

# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    daemon = DevourerDaemon()
    daemon.start()
    app = QApplication(sys.argv)
    console = ASIConsole()
    console.show()
    sys.exit(app.exec())

