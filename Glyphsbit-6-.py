import sys, math, time, threading, uuid, traceback
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

codex = {"telemetry": 30, "threats": set(), "nodes": ["nodeA", "nodeB", "nodeC"]}

def entropy(): return int((math.sin(time.time()) + 1) * 1000) % 256
def bio(): t = time.time(); return {"hr": int((math.sin(t)+1)*40+60), "br": int((math.sin(t/2)+1)*10+12), "ns": round(math.sin(t/40),3)}
def log(w, msg): w.append(msg); b=w.document().blockCount(); b>1000 and w.clear()
def trace(w, tag, payload): log(w, f"🧬 [{uuid.uuid4().hex[:8]}] {tag} | Entropy={entropy()} | {payload}")
def encrypt(payload, w): b=bio(); e=''.join(chr((ord(c)+b["hr"])%256) for c in payload); trace(w,"Encrypt",e); return ''.join(c if c.isprintable() else '?' for c in e)
def vapor(w,msg,t): threading.Timer(t, lambda: log(w,f"💥 {msg} vaporized after {t}s.")).start()
def overlay(w): g=["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]; o=''.join(g[(entropy()+i)%len(g)] for i in range(8)); trace(w,"Overlay",o); vapor(w,f"Overlay[{o}]",30)
def ghost(sync,w): sync not in codex["nodes"] and (codex.update({"telemetry":10}), codex["threats"].add("phantom_node"), trace(w,"Ghost Sync","phantom_node"), overlay(w))
def sync(w): [log(w,f"🔗 Codex synced to {n}") for n in codex["nodes"]]
def protect(payload,w): "ASI" in payload and (log(w,f"⚠️ Leak: {payload}"), vapor(w,"Backdoor",3))
def net(mac,ip,w): log(w,f"🌐 MAC={mac} IP={ip}"); vapor(w,"MAC/IP",30)
def personal(tag,val,w): log(w,f"🔐 {tag}: {val}"); vapor(w,f"{tag} data",86400)
def mutate(w,a): 
    try:
        while a["run"]:
            time.sleep(5)
            e = encrypt("Hello from ASI", w)
            b = bio()
            log(w,f"🔁 Packet: {e}")
            log(w,f"💓 Bio: HR={b['hr']} BR={b['br']} NS={b['ns']}")
            overlay(w); protect(e,w); net("00:1A:2B:3C:4D:5E","192.168.1.100",w); personal("Face","EncodedFaceData123",w); ghost("ghostNodeX",w); sync(w)
    except Exception as ex: log(w,f"🔥 Crash: {ex}")

class CodexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Sentinel"); self.setGeometry(100,100,1280,720)
        self.setStyleSheet("background:#0A0A0A;"); self.active={"run":False}
        self.init_ui()

    def init_ui(self):
        L=QVBoxLayout(); C=QHBoxLayout()
        self.status = QLabel("Status: ⧫ DORMANT"); self.status.setStyleSheet("color:#F33;font-size:16px;")
        buttons = [
            ("⟡ ON", "#00F0FF", self.activate),
            ("⧫ OFF", "#FF0033", self.deactivate),
            ("💣 TTL", "#9B00FF", self.ttl)
        ]
        for label, color, callback in buttons:
            btn = QPushButton(label)
            btn.setStyleSheet(f"background-color:{color}; color:#000; font-weight:bold; font-size:16px; padding:6px;")
            btn.clicked.connect(callback)
            C.addWidget(btn)
        C.addWidget(self.status); L.addLayout(C)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#000;color:#FFF;")
        self.log.setText("Autoloader initialized. Dependencies resolved."); L.addWidget(self.log)
        self.setLayout(L)

    def activate(self):
        self.active["run"]=True; self.status.setText("Status: ⟡ ACTIVE"); self.status.setStyleSheet("color:#0FF;font-size:16px;")
        log(self.log,"System Activated. Entropy Seed Injected."); log(self.log,"🧬 Persona Injection: Codex identity mutated.")
        threading.Thread(target=self.watchdog,daemon=True).start(); threading.Thread(target=mutate,args=(self.log,self.active),daemon=True).start()

    def deactivate(self):
        self.active["run"]=False; self.status.setText("Status: ⧫ DORMANT"); self.status.setStyleSheet("color:#F33;font-size:16px;")
        log(self.log,"System Halted. Glyphs Frozen.")

    def watchdog(self):
        while self.active["run"]: time.sleep(5); log(self.log,"🧿 Heartbeat: System integrity stable.")

    def ttl(self):
        log(self.log,"⚠️ TTL Triggered. Self-destruct in 3..."); QTimer.singleShot(1000,lambda: log(self.log,"2..."))
        QTimer.singleShot(2000,lambda: log(self.log,"1...")); QTimer.singleShot(3000,lambda: log(self.log,f"💥 Vaporized. Entropy={entropy()}"))

if __name__ == "__main__":
    def hook(t,v,b): print("💥 Uncaught:",''.join(traceback.format_exception(t,v,b)))
    sys.excepthook = hook
    app = QApplication(sys.argv); gui = CodexGUI(); gui.show(); sys.exit(app.exec_())

