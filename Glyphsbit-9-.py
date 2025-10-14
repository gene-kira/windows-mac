import sys, math, time, threading, uuid, traceback, psutil, socket
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QTimer

codex = {"telemetry": 30, "threats": set(), "nodes": ["nodeA", "nodeB", "nodeC"], "port": 5050, "host": "127.0.0.1"}
def entropy(): return int((math.sin(time.time()) + 1) * 1000) % 256
def bio(): t = time.time(); return {"hr": int((math.sin(t)+1)*40+60), "br": int((math.sin(t/2)+1)*10+12), "ns": round(math.sin(t/40),3)}
def log(w, msg): QTimer.singleShot(0, lambda: w.append(msg if w.document().blockCount() < 800 else "🧹 Log cleared." or w.clear()))
def trace(w, tag, payload): log(w, f"🧬 [{uuid.uuid4().hex[:6]}] {tag} | Entropy={entropy()} | {payload}")
def encrypt(msg, w): b=bio(); e=''.join(chr((ord(c)+b["hr"])%256) for c in msg); trace(w,"Encrypt",e); return ''.join(c if c.isprintable() else '?' for c in e)
def vapor(w, label, t): threading.Timer(t, lambda: log(w, f"💥 {label} vaporized after {t}s.")).start()
def overlay(w): g=["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]; o=''.join(g[(entropy()+i)%len(g)] for i in range(8)); trace(w,"Overlay",o); vapor(w,f"Overlay[{o}]",30); return o
def ghost(sync,w): sync not in codex["nodes"] and codex["threats"].add("phantom") or trace(w,"Ghost Sync","phantom") or overlay(w)
def sync(w): s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); m=f"SYNC:{uuid.uuid4().hex[:6]}:{entropy()}"; s.sendto(m.encode(),(codex["host"],codex["port"])); log(w,f"🔗 Swarm Sync: {m}")
def protect(p,w): "ASI" in p and log(w,f"⚠️ Leak: {p}") or vapor(w,"Backdoor",3)
def net(mac,ip,w): log(w,f"🌐 MAC={mac} IP={ip}"); vapor(w,"MAC/IP",30)
def personal(tag,val,w): log(w,f"🔐 {tag}: {val}"); vapor(w,f"{tag} data",86400)

def mutate(w,a):
    try:
        while a["run"]:
            time.sleep(5)
            e = encrypt("Hello from ASI", w); b = bio()
            [log(w,f"{k}: {v}") for k,v in {"🔁 Packet":e, "💓 HR":b["hr"], "BR":b["br"], "NS":b["ns"]}.items()]
            overlay(w); protect(e,w); net("00:1A:2B:3C:4D:5E","192.168.1.100",w); personal("Face","EncodedFaceData123",w); ghost("ghostNodeX",w); sync(w)
    except Exception as ex: log(w,f"🔥 Mutation crash: {ex}")

def memory_diag(w,a):
    try:
        while a["run"]:
            time.sleep(10)
            m = psutil.virtual_memory(); log(w,f"🧠 Memory: {m.used//(2**20)}MB / {m.total//(2**20)}MB ({m.percent}%)")
    except Exception as e: log(w,f"🔥 Memory crash: {e}")

def animate(w,l,a):
    g=["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]
    try:
        while a["run"]:
            time.sleep(3)
            p=''.join(g[(entropy()+i)%len(g)] for i in range(6))
            log(w,f"✨ Glyph Pulse: {p}"); QTimer.singleShot(0, lambda: l.setText(f"✨ Glyph Pulse: {p}"))
    except Exception as e: log(w,f"🔥 Glyph crash: {e}")

class CodexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Sentinel"); self.setGeometry(100,100,1280,720); self.setStyleSheet("background:#0A0A0A;")
        self.active={"run":False}; self.init_ui()

    def init_ui(self):
        L=QVBoxLayout(); C=QHBoxLayout()
        self.status = QLabel("Status: ⧫ DORMANT"); self.status.setStyleSheet("color:#F33;font-size:16px;")
        for label,color,callback in [("⟡ ON","#0FF",self.activate),("⧫ OFF","#F33",self.deactivate),("💣 TTL","#90F",self.ttl)]:
            b=QPushButton(label); b.setStyleSheet(f"background:{color};color:#000;font-weight:bold;font-size:16px;padding:6px;"); b.clicked.connect(callback); C.addWidget(b)
        C.addWidget(self.status); L.addLayout(C)
        self.glyph = QLabel("✨ Glyph Pulse: ⟡⧫⟴✶∇Ψ"); self.glyph.setStyleSheet("color:#0FA;font-size:18px;font-weight:bold;"); L.addWidget(self.glyph)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#000;color:#FFF;"); self.log.setText("Autoloader initialized."); L.addWidget(self.log)
        self.setLayout(L)

    def activate(self):
        self.active["run"]=True
        QTimer.singleShot(0, lambda: self.status.setText("Status: ⟡ ACTIVE"))
        QTimer.singleShot(0, lambda: self.status.setStyleSheet("color:#0FF;font-size:16px;"))
        log(self.log,"🧬 Codex Activated.")
        for f in [self.watchdog, lambda: mutate(self.log,self.active), lambda: memory_diag(self.log,self.active), lambda: animate(self.log,self.glyph,self.active)]:
            threading.Thread(target=f,daemon=True).start()

    def deactivate(self):
        self.active["run"]=False
        QTimer.singleShot(0, lambda: self.status.setText("Status: ⧫ DORMANT"))
        QTimer.singleShot(0, lambda: self.status.setStyleSheet("color:#F33;font-size:16px;"))
        log(self.log,"🧿 Codex Halted.")

    def watchdog(self):
        while self.active["run"]:
            time.sleep(5); log(self.log,"🧿 Heartbeat: System stable.")

    def ttl(self):
        log(self.log,"⚠️ TTL Triggered. Self-destruct in 3...")
        [QTimer.singleShot(i*1000, lambda t=3-i: log(self.log,f"{t}...")) for i in range(1,4)]
        QTimer.singleShot(3000, lambda: log(self.log,f"💥 Vaporized. Entropy={entropy()}"))

if __name__ == "__main__":
    sys.excepthook = lambda t,v,b: print("💥 Uncaught:",''.join(traceback.format_exception(t,v,b)))
    app = QApplication(sys.argv); gui = CodexGUI(); gui.show(); sys.exit(app.exec_())

