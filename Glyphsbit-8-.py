import sys, math, time, threading, uuid, traceback, psutil
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QTimer

codex = {"telemetry": 30, "threats": set(), "nodes": ["nodeA", "nodeB", "nodeC"]}

def entropy(): return int((math.sin(time.time()) + 1) * 1000) % 256
def bio(): t = time.time(); return {"hr": int((math.sin(t)+1)*40+60), "br": int((math.sin(t/2)+1)*10+12), "ns": round(math.sin(t/40),3)}

def safe_log(widget, message):
    def update():
        if widget.document().blockCount() > 800:
            widget.clear()
            widget.append("🧹 Log cleared to prevent overflow.")
        widget.append(message)
    QTimer.singleShot(0, update)

def trace(widget, tag, payload):
    lineage = uuid.uuid4().hex[:8]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    safe_log(widget, f"🧬 [{lineage}] {tag} @ {ts} | Entropy={entropy()} | {payload}")

def encrypt(payload, widget):
    b = bio()
    encrypted = ''.join(chr((ord(c)+b["hr"])%256) for c in payload)
    trace(widget, "Encrypt", encrypted)
    return ''.join(c if c.isprintable() else '?' for c in encrypted)

def vapor(widget, label, delay):
    threading.Timer(delay, lambda: safe_log(widget, f"💥 {label} vaporized after {delay}s.")).start()

def overlay(widget):
    glyphs = ["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]
    o = ''.join(glyphs[(entropy()+i)%len(glyphs)] for i in range(8))
    trace(widget, "Overlay", o)
    vapor(widget, f"Overlay[{o}]", 30)

def ghost(sync, widget):
    if sync not in codex["nodes"]:
        codex["telemetry"] = 10
        codex["threats"].add("phantom_node")
        trace(widget, "Ghost Sync", "phantom_node")
        overlay(widget)

def sync_codex(widget):
    for node in codex["nodes"]:
        safe_log(widget, f"🔗 Codex synced to {node}")

def protect(payload, widget):
    if "ASI" in payload:
        safe_log(widget, f"⚠️ Leak: {payload}")
        vapor(widget, "Backdoor", 3)

def net(mac, ip, widget):
    safe_log(widget, f"🌐 MAC={mac} IP={ip}")
    vapor(widget, "MAC/IP", 30)

def personal(tag, val, widget):
    safe_log(widget, f"🔐 {tag}: {val}")
    vapor(widget, f"{tag} data", 86400)

def mutate(widget, active):
    try:
        while active["run"]:
            time.sleep(5)
            e = encrypt("Hello from ASI", widget)
            b = bio()
            safe_log(widget, f"🔁 Packet: {e}")
            safe_log(widget, f"💓 Bio: HR={b['hr']} BR={b['br']} NS={b['ns']}")
            overlay(widget); protect(e, widget); net("00:1A:2B:3C:4D:5E", "192.168.1.100", widget)
            personal("Face", "EncodedFaceData123", widget); ghost("ghostNodeX", widget); sync_codex(widget)
    except Exception as ex:
        safe_log(widget, f"🔥 Crash: {ex}")

def memory_diagnostics(widget, active):
    try:
        while active["run"]:
            time.sleep(10)
            mem = psutil.virtual_memory()
            used = f"{mem.used // (1024**2)}MB"
            total = f"{mem.total // (1024**2)}MB"
            percent = mem.percent
            safe_log(widget, f"🧠 Memory: {used} / {total} ({percent}%) used")
    except Exception as e:
        safe_log(widget, f"🔥 Memory diagnostics crashed: {e}")

def animate_glyphs(widget, active):
    glyphs = ["⟡", "⧫", "⟴", "✶", "∇", "Ψ", "Σ", "⨁"]
    try:
        while active["run"]:
            time.sleep(3)
            pulse = ''.join(glyphs[(entropy() + i) % len(glyphs)] for i in range(6))
            safe_log(widget, f"✨ Glyph Pulse: {pulse}")
    except Exception as e:
        safe_log(widget, f"🔥 Glyph animation crashed: {e}")

class CodexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Sentinel")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background:#0A0A0A;")
        self.active = {"run": False}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        control = QHBoxLayout()

        self.status = QLabel("Status: ⧫ DORMANT")
        self.status.setStyleSheet("color:#F33;font-size:16px;")

        buttons = [
            ("⟡ ON", "#00F0FF", self.activate),
            ("⧫ OFF", "#FF0033", self.deactivate),
            ("💣 TTL", "#9B00FF", self.ttl)
        ]
        for label, color, callback in buttons:
            btn = QPushButton(label)
            btn.setStyleSheet(f"background-color:{color}; color:#000; font-weight:bold; font-size:16px; padding:6px;")
            btn.clicked.connect(callback)
            control.addWidget(btn)

        control.addWidget(self.status)
        layout.addLayout(control)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#000;color:#FFF;")
        self.log.setText("Autoloader initialized. Dependencies resolved.")
        layout.addWidget(self.log)

        self.setLayout(layout)

    def activate(self):
        self.active["run"] = True
        QTimer.singleShot(0, lambda: self.status.setText("Status: ⟡ ACTIVE"))
        QTimer.singleShot(0, lambda: self.status.setStyleSheet("color:#00F0FF;font-size:16px;"))
        safe_log(self.log, "System Activated. Entropy Seed Injected.")
        safe_log(self.log, "🧬 Persona Injection: Codex identity mutated.")
        threading.Thread(target=self.watchdog, daemon=True).start()
        threading.Thread(target=mutate, args=(self.log, self.active), daemon=True).start()
        threading.Thread(target=memory_diagnostics, args=(self.log, self.active), daemon=True).start()
        threading.Thread(target=animate_glyphs, args=(self.log, self.active), daemon=True).start()

    def deactivate(self):
        self.active["run"] = False
        QTimer.singleShot(0, lambda: self.status.setText("Status: ⧫ DORMANT"))
        QTimer.singleShot(0, lambda: self.status.setStyleSheet("color:#FF0033;font-size:16px;"))
        safe_log(self.log, "System Halted. Glyphs Frozen.")

    def watchdog(self):
        while self.active["run"]:
            time.sleep(5)
            safe_log(self.log, "🧿 Heartbeat: System integrity stable.")

    def ttl(self):
        safe_log(self.log, "⚠️ TTL Triggered. Self-destruct in 3...")
        QTimer.singleShot(1000, lambda: safe_log(self.log, "2..."))
        QTimer.singleShot(2000, lambda: safe_log(self.log, "1..."))
        QTimer.singleShot(3000, lambda: safe_log(self.log, f"💥 Vaporized. Entropy={entropy()}"))

if __name__ == "__main__":
    def hook(t, v, b): print("💥 Uncaught:", ''.join(traceback.format_exception(t, v, b)))
    sys.excepthook = hook
    app = QApplication(sys.argv)
    gui = CodexGUI()
    gui.show()
    sys.exit(app.exec_())

