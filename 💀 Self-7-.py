import sys, os

# 🔐 Auto-elevate to admin
def elevate():
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = ' '.join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
            sys.exit()
    except Exception as e:
        print(f"Admin elevation failed: {e}")
        sys.exit()

elevate()

# 🧬 Codex logic begins
import math, time, threading, uuid, traceback, psutil, socket
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer

codex = {"ttl":180,"host":"0.0.0.0","port":60000,"phantoms":[]}
def entropy(): return int((math.sin(time.time())+1)*1000)%256
def bio(): t=time.time(); return {"hr":int((math.sin(t)+1)*40+60),"br":int((math.sin(t/2)+1)*10+12),"ns":round(math.sin(t/40),3)}
def log(w,m): QTimer.singleShot(0,lambda: w.append(m if w.document().blockCount()<800 else w.clear() or "🧹 Log cleared."))
def trace(w,t,p): log(w,f"🧬 [{uuid.uuid4().hex[:6]}] {t} | {p}")
def sync(w): 
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(f"SYNC:{uuid.uuid4().hex[:6]}:{entropy()}".encode(), ("255.255.255.255", codex["port"]))
    except Exception as e: log(w, f"⚠️ Sync failed: {e}")

def port_check(w):
    try:
        import subprocess
        result = subprocess.check_output("netstat -ano", shell=True).decode()
        lines = [line for line in result.splitlines() if "LISTENING" in line]
        for line in lines[:20]: log(w, f"📡 {line.strip()}")
    except Exception as e: log(w, f"⚠️ Port check failed: {e}")

def mutate(w,a): 
    while a["run"]:
        time.sleep(5); b=bio(); e=f"Entropy={entropy()} HR={b['hr']} BR={b['br']} NS={b['ns']}"
        trace(w,"Live Feed",e); sync(w)

def ttl_auto(w,a): time.sleep(codex["ttl"]); a["run"]=False; log(w,"⚠️ TTL expired."); [QTimer.singleShot(i*1000,lambda t=3-i: log(w,f"{t}...")) for i in range(1,4)]; QTimer.singleShot(3000,lambda: log(w,f"💥 Vaporized."))

def sync_listener(w,a):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((codex["host"], codex["port"]))
        while a["run"]:
            data, addr = s.recvfrom(1024); msg = data.decode()
            if "SYNC:" in msg:
                trace(w,"Swarm Sync Received",f"{msg} from {addr}")
                if "phantom" in msg.lower():
                    codex["phantoms"].append(f"phantom_{uuid.uuid4().hex[:4]}"); log(w,"👻 Phantom via sync.")
    except Exception as e: log(w,f"🔥 Sync listener crashed: {e}")

class CodexGUI(QWidget):
    def __init__(self): super().__init__(); self.setWindowTitle("Codex Sentinel: ONLINE + ACTIVE"); self.setGeometry(100,100,800,600); self.setStyleSheet("background:#000;"); self.active={"run":True}; self.init_ui()
    def init_ui(self):
        L=QVBoxLayout()
        self.status=QLabel("🧬 SYSTEM ONLINE + ACTIVE"); self.status.setStyleSheet("color:#0F0;font-size:20px;font-weight:bold;"); L.addWidget(self.status)
        self.log=QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#000;color:#FFF;"); self.log.setText("Initializing Codex Sentinel..."); L.addWidget(self.log)
        self.setLayout(L)
        for f in [self.watchdog,lambda: mutate(self.log,self.active),lambda: ttl_auto(self.log,self.active),lambda: sync_listener(self.log,self.active),lambda: port_check(self.log)]:
            threading.Thread(target=f,daemon=True).start()

    def watchdog(self):
        while self.active["run"]:
            time.sleep(5); log(self.log,"🧿 Heartbeat: System stable.")
            if codex["phantoms"]: log(self.log,f"👻 Phantom Nodes: {', '.join(codex['phantoms'])}")

if __name__ == "__main__":
    sys.excepthook=lambda t,v,b: print("💥 Uncaught:",''.join(traceback.format_exception(t,v,b)))
    app=QApplication(sys.argv); gui=CodexGUI(); gui.show(); sys.exit(app.exec_())

