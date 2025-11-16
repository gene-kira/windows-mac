import subprocess, sys
for p in ["PyQt5", "requests", "psutil"]:
    try: __import__(p if p != "PyQt5" else "PyQt5.QtWidgets")
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", p])

import threading, time, random, sqlite3, requests, psutil, socket
from PyQt5.QtWidgets import *; from PyQt5.QtGui import *; from PyQt5.QtCore import *

def enforce():
    cmds = [
        'reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\DataCollection" /v AllowTelemetry /t REG_DWORD /d 0 /f',
        'schtasks /delete /tn "Microsoft\\Windows\\Feedback\\Siuf\\DmClient" /f',
        'schtasks /delete /tn "Microsoft\\Windows\\Feedback\\Siuf\\DmClientOnScenarioDownload" /f',
        'reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Windows Error Reporting" /v Disabled /t REG_DWORD /d 1 /f'
    ]
    for cmd in cmds:
        try: subprocess.run(cmd, shell=True)
        except: pass

def get_ports():
    return [{
        "port": str(c.laddr.port),
        "proto": "TCP" if c.type == socket.SOCK_STREAM else "UDP",
        "remote": c.raddr.ip if c.raddr else "N/A"
    } for c in psutil.net_connections(kind='inet') if c.status == psutil.CONN_ESTABLISHED and c.laddr]

def is_suspicious(p, r):
    p = int(p)
    return p in [4444, 5555, 31337] or p > 49152 or r.startswith(("10.", "192.168."))

class PlasmaDaemon:
    def __init__(self):
        self.lock = threading.Lock()
        self.db = sqlite3.connect("mutations.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS mutations (
            timestamp REAL, port TEXT, proto TEXT, remote TEXT, strikes INT,
            energy REAL, integrity REAL, field_strength_T REAL, suspicious INT)""")
        self.gui_callback = None
        self.mu_0, self.E_th, self.i_th = 4e-7 * 3.1415, 100.0, 0.2

    def process(self, port, proto, remote, strikes, V, I, t, η):
        E = strikes * V * I * t * η
        B = random.uniform(1.0, 5.0)
        i = max(0, (B**2 / self.mu_0) * (1 - E / self.E_th))
        m = dict(timestamp=time.time(), port=port, proto=proto, remote=remote, strikes=strikes,
                 energy=E, integrity=i, field_strength_T=B, suspicious=int(is_suspicious(port, remote)))
        with self.lock:
            self.db.execute("INSERT INTO mutations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(m.values()))
            self.db.commit()
        try: requests.post("http://localhost:5000/sync", json=m)
        except: pass
        if i < self.i_th and self.gui_callback:
            self.gui_callback("glyph", m)
        return m

class LightningDaemon(threading.Thread):
    def __init__(self, info, cb, core):
        super().__init__(daemon=True)
        self.i, self.cb, self.core = info, cb, core

    def run(self):
        while True:
            m = self.core.process(self.i["port"], self.i["proto"], self.i["remote"],
                                  random.randint(1, 5), 1e9, 30000, 0.0002, 0.75)
            self.cb("mutation", m)
            time.sleep(2)

class SignalBridge(QObject):
    mutation_signal = pyqtSignal(dict)
    glyph_signal = pyqtSignal(dict)

class PlasmaRing(QWidget):
    def __init__(self):
        super().__init__()
        self.i = 1.0
        self.setMinimumSize(300, 300)

    def update_integrity(self, v):
        self.i = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        g = max(0.2, min(1.0, self.i))
        p.setPen(QPen(QColor.fromHsvF(0.6 * g, 1.0, 1.0), 4))
        c = self.rect().center()
        r = 100 + 50 * g
        p.drawEllipse(c, r, r)

class SyncMap(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.nodes = [QGraphicsEllipseItem(QRectF(x * 60, y * 60, 40, 40)) for x in range(5) for y in range(3)]
        for n in self.nodes:
            n.setBrush(QColor("#444"))
            self.scene.addItem(n)

    def pulse(self):
        for n in self.nodes:
            n.setBrush(QColor("#00ffff"))
class CodexGUI(QWidget):
    def __init__(self, core):
        super().__init__()
        self.setWindowTitle("Codex Purge Shell – ASI Oversight Console")
        self.setGeometry(100, 100, 1400, 700)
        self.setStyleSheet("background:#0a0a0a; color:#00ffff;")
        self.setFont(QFont("Roboto Mono", 10))
        self.bridge = SignalBridge()
        self.bridge.mutation_signal.connect(self.update)
        self.bridge.glyph_signal.connect(self.show_glyph)
        self.mutations = []

        self.status = QLabel("⚡ Plasma Integrity: Stable")
        self.status.setStyleSheet("color:#00ff00; font-size:16px;")
        self.diag = QLabel("🧼 Diagnostic Status: Telemetry Locked, 0 Suspicious Ports")
        self.diag.setStyleSheet("color:#ff00ff; font-size:14px;")
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Time", "Port", "Proto", "Remote", "Strikes", "Energy", "Integrity", "Field (T)", "⚠️"])
        self.table.setStyleSheet("background:#111; color:#fff;")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#111; color:#00ffff;")
        self.ring = PlasmaRing()
        self.sync = SyncMap()
        self.sync.setFixedSize(400, 200)
        self.glyph_scene = QGraphicsScene()
        self.glyph_view = QGraphicsView(self.glyph_scene)
        self.glyph_view.setFixedSize(400, 200)
        self.glyph_view.setStyleSheet("background:#111; border:2px solid #00ffff;")

        left = QVBoxLayout()
        for w in [self.status, self.diag, self.table, self.log]:
            left.addWidget(w)
        right = QVBoxLayout()
        for w in [self.ring, self.sync, self.glyph_view]:
            right.addWidget(w)
        layout = QHBoxLayout()
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)
        self.setLayout(layout)

    def update(self, m):
        self.mutations.append(m)
        i = m["integrity"]
        color = "#00ff00" if i > 0.5 else "#ff0000" if i < 0.2 else "#ffff00"
        self.status.setText(f"⚡ Plasma Integrity: {i:.2e}")
        self.status.setStyleSheet(f"color:{color}; font-size:16px;")
        suspicious_count = sum(x["suspicious"] for x in self.mutations)
        self.diag.setText(f"🧼 Diagnostic Status: Telemetry Locked, {suspicious_count} Suspicious Ports")
        r = self.table.rowCount()
        self.table.insertRow(r)
        for j, k in enumerate(["timestamp", "port", "proto", "remote", "strikes", "energy", "integrity", "field_strength_T", "suspicious"]):
            val = "⚠️" if k == "suspicious" and m[k] else (f"{m[k]:.2f}" if isinstance(m[k], float) else str(m[k]))
            self.table.setItem(r, j, QTableWidgetItem(val))
        self.ring.update_integrity(i)
        self.sync.pulse()
        self.log.append(
            f"⏱ {m['timestamp']:.2f} | Port {m['port']} ({m['proto']}) → {m['remote']} | "
            f"⚡ {m['strikes']} | E={m['energy']:.2f} J | Integrity={i:.2e} | "
            f"{'⚠️ SUSPICIOUS' if m['suspicious'] else 'OK'}"
        )

    def show_glyph(self, m):
        self.glyph_scene.clear()
        glyph = QGraphicsTextItem("☢ Resurrection Glyph Activated ☢")
        glyph.setDefaultTextColor(QColor("#ff00ff"))
        glyph.setFont(QFont("Roboto Mono", 16, QFont.Bold))
        glyph.setPos(50, 80)
        self.glyph_scene.addItem(glyph)

# 🚀 Launch
if __name__ == "__main__":
    enforce()
    app = QApplication(sys.argv)
    core = PlasmaDaemon()
    gui = CodexGUI(core)

    def callback(event, m):
        if event == "mutation":
            gui.bridge.mutation_signal.emit(m)
        elif event == "glyph":
            gui.bridge.glyph_signal.emit(m)

    core.gui_callback = callback
    for info in get_ports():
        LightningDaemon(info, callback, core).start()

    gui.show()
    sys.exit(app.exec_())


