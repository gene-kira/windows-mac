# 🚀 Autoloader
import subprocess, sys
for pkg in ["PyQt5", "requests", "psutil"]:
    try: __import__(pkg if pkg != "PyQt5" else "PyQt5.QtWidgets")
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🧠 Imports
import threading, time, random, sqlite3, requests, psutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# 🌐 Detect active internet ports
def get_active_ports():
    ports = set()
    for c in psutil.net_connections(kind='inet'):
        if c.status == psutil.CONN_ESTABLISHED and c.laddr:
            ports.add(str(c.laddr.port))
    return sorted(ports)

# 🧬 Daemon Core
class PlasmaDaemon:
    def __init__(self):
        self.lock = threading.Lock()
        self.db = sqlite3.connect("mutations.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS mutations (
            timestamp REAL, port TEXT, strikes INT, energy REAL, integrity REAL, field_strength_T REAL)""")
        self.gui_callback = None
        self.mu_0, self.E_threshold, self.integrity_threshold = 4e-7 * 3.1415, 100.0, 0.2

    def process(self, port, s, V, I, t, η):
        E = s * V * I * t * η
        B = random.uniform(1.0, 5.0)
        integrity = max(0, (B**2 / self.mu_0) * (1 - E / self.E_threshold))
        m = dict(timestamp=time.time(), port=port, strikes=s, energy=E, integrity=integrity, field_strength_T=B)
        with self.lock:
            self.db.execute("INSERT INTO mutations VALUES (?, ?, ?, ?, ?, ?)", tuple(m.values()))
            self.db.commit()
        try: requests.post("http://localhost:5000/sync", json=m)
        except: pass
        if integrity < self.integrity_threshold and self.gui_callback:
            self.gui_callback("glyph", m)
        return m

# ⚡ Daemon Thread
class LightningDaemon(threading.Thread):
    def __init__(self, port, cb, core):
        super().__init__(daemon=True)
        self.port, self.cb, self.core = port, cb, core

    def run(self):
        while True:
            m = self.core.process(self.port, random.randint(1, 5), 1e9, 30000, 0.0002, 0.75)
            self.cb("mutation", m); time.sleep(2)

# 🌐 Signal Bridge
class SignalBridge(QObject):
    mutation_signal = pyqtSignal(dict)
    glyph_signal = pyqtSignal(dict)

# 🌌 Plasma Ring
class PlasmaRing(QWidget):
    def __init__(self): super().__init__(); self.integrity = 1.0; self.setMinimumSize(300, 300)
    def update_integrity(self, v): self.integrity = v; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        glow = max(0.2, min(1.0, self.integrity))
        p.setPen(QPen(QColor.fromHsvF(0.6 * glow, 1.0, 1.0), 4))
        c = self.rect().center(); r = 100 + 50 * glow; p.drawEllipse(c, r, r)

# 🕸️ Swarm Sync Map
class SyncMap(QGraphicsView):
    def __init__(self):
        super().__init__(); self.scene = QGraphicsScene(); self.setScene(self.scene)
        self.nodes = [QGraphicsEllipseItem(QRectF(x * 60, y * 60, 40, 40)) for x in range(5) for y in range(3)]
        for n in self.nodes: n.setBrush(QColor("#444")); self.scene.addItem(n)
    def pulse(self): [n.setBrush(QColor("#00ffff")) for n in self.nodes]

# 🖥️ GUI
class CodexGUI(QWidget):
    def __init__(self, core):
        super().__init__()
        self.setWindowTitle("Codex Plasma Daemon – ASI Oversight Console")
        self.setGeometry(100, 100, 1400, 700)
        self.setStyleSheet("background:#0a0a0a; color:#00ffff;"); self.setFont(QFont("Roboto Mono", 10))
        self.bridge = SignalBridge()
        self.bridge.mutation_signal.connect(self.update)
        self.bridge.glyph_signal.connect(self.show_glyph)

        self.status = QLabel("⚡ Plasma Integrity: Stable"); self.status.setStyleSheet("color:#00ff00; font-size:16px;")
        self.table = QTableWidget(0, 6); self.table.setHorizontalHeaderLabels(["Time", "Port", "Strikes", "Energy", "Integrity", "Field (T)"])
        self.table.setStyleSheet("background:#111; color:#fff;")
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#111; color:#00ffff;")
        self.ring = PlasmaRing(); self.sync = SyncMap(); self.sync.setFixedSize(400, 200)
        self.glyph_scene = QGraphicsScene(); self.glyph_view = QGraphicsView(self.glyph_scene)
        self.glyph_view.setFixedSize(400, 200); self.glyph_view.setStyleSheet("background:#111; border:2px solid #00ffff;")

        left = QVBoxLayout(); [left.addWidget(w) for w in [self.status, self.table, self.log]]
        right = QVBoxLayout(); [right.addWidget(w) for w in [self.ring, self.sync, self.glyph_view]]
        layout = QHBoxLayout(); layout.addLayout(left, 3); layout.addLayout(right, 2); self.setLayout(layout)

    def update(self, m):
        i = m["integrity"]
        color = "#00ff00" if i > 0.5 else "#ff0000" if i < 0.2 else "#ffff00"
        self.status.setText(f"⚡ Plasma Integrity: {i:.2e}")
        self.status.setStyleSheet(f"color:{color}; font-size:16px;")
        r = self.table.rowCount(); self.table.insertRow(r)
        for j, k in enumerate(["timestamp", "port", "strikes", "energy", "integrity", "field_strength_T"]):
            val = f"{m[k]:.2f}" if isinstance(m[k], float) else str(m[k])
            self.table.setItem(r, j, QTableWidgetItem(val))
        self.ring.update_integrity(i); self.sync.pulse()
        self.log.append(f"⏱ {m['timestamp']:.2f} | Port {m['port']} | ⚡ {m['strikes']} | E={m['energy']:.2f} J | Integrity={i:.2e}")

    def show_glyph(self, m):
        self.glyph_scene.clear()
        glyph = QGraphicsTextItem("☢ Resurrection Glyph Activated ☢")
        glyph.setDefaultTextColor(QColor("#ff00ff")); glyph.setFont(QFont("Roboto Mono", 16, QFont.Bold)); glyph.setPos(50, 80)
        self.glyph_scene.addItem(glyph)

# 🚀 Launch
if __name__ == "__main__":
    app = QApplication(sys.argv)
    core = PlasmaDaemon(); gui = CodexGUI(core)

    def callback(event, m):
        if event == "mutation": gui.bridge.mutation_signal.emit(m)
        elif event == "glyph": gui.bridge.glyph_signal.emit(m)

    core.gui_callback = callback
    for port in get_active_ports():
        LightningDaemon(port, callback, core).start()

    gui.show(); sys.exit(app.exec_())

