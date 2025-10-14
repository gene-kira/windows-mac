import sys, math, time, threading, uuid, traceback, psutil, socket
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer

codex = {"nodes":["nodeA","nodeB"],"port":5050,"host":"127.0.0.1","ttl":180}
def entropy(): return int((math.sin(time.time())+1)*1000)%256
def bio(): t=time.time(); return {"hr":int((math.sin(t)+1)*40+60),"br":int((math.sin(t/2)+1)*10+12),"ns":round(math.sin(t/40),3)}
def log(w,m): QTimer.singleShot(0,lambda: w.append(m if w.document().blockCount()<800 else "🧹 Log cleared." or w.clear()))
def trace(w,t,p): log(w,f"🧬 [{uuid.uuid4().hex[:6]}] {t} | Entropy={entropy()} | {p}")
def encrypt(msg,w): b=bio(); e=''.join(chr((ord(c)+b["hr"])%256) for c in msg); trace(w,"Encrypt",e); return ''.join(c if c.isprintable() else '?' for c in e)
def overlay(w): g=["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]; o=''.join(g[(entropy()+i)%len(g)] for i in range(8)); trace(w,"Overlay",o); return o
def sync(w): s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); m=f"SYNC:{uuid.uuid4().hex[:6]}:{entropy()}"; s.sendto(m.encode(),(codex["host"],codex["port"])); log(w,f"🔗 Swarm Sync: {m}")

def codex_immunity(w,b):
    log(w,f"🛡️ Codex Immunity Activated. HR={b['hr']} BR={b['br']} NS={b['ns']}")
    trace(w,"Immunity Sync",f"Entropy={entropy()} | Glyph={overlay(w)}")
    sync(w)
    log(w,"🧬 Mutation threads hardened. Swarm notified.")

def devour(w,b,a):
    glyph = overlay(w)
    log(w,f"🐍 Devourer v2: Glyph collapse {glyph}")
    codex_immunity(w,b)
    a["run"] = False
    log(w,"💣 Mutation threads terminated. TTL override engaged.")
    [QTimer.singleShot(i*1000,lambda t=3-i: log(w,f"{t}...")) for i in range(1,4)]
    QTimer.singleShot(3000,lambda: log(w,f"💥 Codex vaporized. Entropy={entropy()}"))

def detect_shai_hulud(w,a):
    try:
        while a["run"]:
            time.sleep(5)
            b = bio()
            if b["ns"] > 0.9 or entropy() > 240:
                trace(w,"Shai-Hulud Breach",f"NS={b['ns']} Entropy={entropy()}")
                devour(w,b,a)
    except Exception as e: log(w,f"🔥 Shai-Hulud detection crashed: {e}")

def mutate(w,a):
    try:
        while a["run"]:
            time.sleep(5)
            e=encrypt("GlyphChain::"+overlay(w),w); b=bio()
            [log(w,f"{k}: {v}") for k,v in {"🔁 Packet":e,"💓 HR":b["hr"],"BR":b["br"],"NS":b["ns"]}.items()]
            sync(w)
    except Exception as ex: log(w,f"🔥 Mutation crash: {ex}")

def memory_diag(w,a): 
    try:
        while a["run"]:
            time.sleep(10)
            m=psutil.virtual_memory(); log(w,f"🧠 Memory: {m.used//(2**20)}MB / {m.total//(2**20)}MB ({m.percent}%)")
    except Exception as e: log(w,f"🔥 Memory crash: {e}")

def animate(w,l,a):
    g=["⟡","⧫","⟴","✶","∇","Ψ","Σ","⨁"]
    try:
        while a["run"]:
            time.sleep(3)
            p=''.join(g[(entropy()+i)%len(g)] for i in range(6))
            log(w,f"✨ Glyph Pulse: {p}"); QTimer.singleShot(0,lambda: l.setText(f"✨ Glyph Pulse: {p}"))
    except Exception as e: log(w,f"🔥 Glyph crash: {e}")

def ttl_auto(w,a):
    time.sleep(codex["ttl"])
    a["run"]=False
    log(w,"⚠️ TTL expired. Self-destruct initiated.")
    [QTimer.singleShot(i*1000,lambda t=3-i: log(w,f"{t}...")) for i in range(1,4)]
    QTimer.singleShot(3000,lambda: log(w,f"💥 Codex vaporized. Entropy={entropy()}"))

class CodexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codex Sentinel"); self.setGeometry(100,100,1280,720); self.setStyleSheet("background:#0A0A0A;")
        self.active={"run":False}; self.init_ui()

    def init_ui(self):
        L=QVBoxLayout(); C=QHBoxLayout()
        self.status=QLabel("Status: ⧫ DORMANT"); self.status.setStyleSheet("color:#F33;font-size:16px;")
        for label,color,callback in [("⟡ ON","#0FF",self.activate),("⧫ OFF","#F33",self.deactivate)]:
            b=QPushButton(label); b.setStyleSheet(f"background:{color};color:#000;font-weight:bold;font-size:16px;padding:6px;"); b.clicked.connect(callback); C.addWidget(b)
        C.addWidget(self.status); L.addLayout(C)
        self.glyph=QLabel("✨ Glyph Pulse: ⟡⧫⟴✶∇Ψ"); self.glyph.setStyleSheet("color:#0FA;font-size:18px;font-weight:bold;"); L.addWidget(self.glyph)
        self.log=QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#000;color:#FFF;"); self.log.setText("Autoloader initialized."); L.addWidget(self.log)
        self.setLayout(L)

    def activate(self):
        self.active["run"]=True
        QTimer.singleShot(0,lambda: self.status.setText("Status: ⟡ ACTIVE"))
        QTimer.singleShot(0,lambda: self.status.setStyleSheet("color:#0FF;font-size:16px;"))
        log(self.log,"🧬 Codex Activated.")
        for f in [self.watchdog,lambda: mutate(self.log,self.active),lambda: memory_diag(self.log,self.active),lambda: animate(self.log,self.glyph,self.active),lambda: ttl_auto(self.log,self.active),lambda: detect_shai_hulud(self.log,self.active)]:
            threading.Thread(target=f,daemon=True).start()

    def deactivate(self):
        self.active["run"]=False
        QTimer.singleShot(0,lambda: self.status.setText("Status: ⧫ DORMANT"))
        QTimer.singleShot(0,lambda: self.status.setStyleSheet("color:#F33;font-size:16px;"))
        log(self.log,"🧿 Codex Halted.")

    def watchdog(self):
        while self.active["run"]: time.sleep(5); log(self.log,"🧿 Heartbeat: System stable.")

if __name__ == "__main__":
    sys.excepthook=lambda t,v,b: print("💥 Uncaught:",''.join(traceback.format_exception(t,v,b)))
    app=QApplication(sys.argv); gui=CodexGUI(); gui.show(); sys.exit(app.exec_())

