# === 🔄 AUTOLOADER ===
import sys, subprocess
REQUIRED = {"PyQt5": "PyQt5", "psutil": "psutil"}
for module, pip_name in REQUIRED.items():
    try: __import__(module)
    except ImportError:
        print(f"[Autoloader] Installing {pip_name}...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        except Exception as e:
            print(f"[Autoloader] Failed: {e}"); sys.exit(1)

# === 🔐 ELEVATION CHECK ===
import ctypes
if not ctypes.windll.shell32.IsUserAnAdmin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{sys.argv[0]}"', None, 1)
    sys.exit()

# === CORE IMPORTS ===
import os, time, json, hashlib, psutil, socket
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QApplication, QTextEdit
from PyQt5.QtCore import Qt

# === GLOBALS ===
glyph_cache, symbolic_registry = {}, {}
AUTONOMOUS_MODE = True
BLOCKED = {"RU", "CN", "IR"}
C2 = ["api.mainnet-beta.solana.com", "calendar.google.com"]
TRUSTED = {"abc123...", "def456..."}
UNICODE = ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']

# === SCANNER ===
def scan_extensions(path):
    if not os.path.exists(path): return
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(('.js', '.json')):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as t:
                        txt = t.read()
                    glyphs = [(i, c) for i, c in enumerate(txt) if c in UNICODE]
                    if glyphs:
                        origin = "US" if "corp" in p else "unknown"
                        h = hashlib.sha256()
                        with open(p, 'rb') as b:
                            for chunk in iter(lambda: b.read(4096), b""): h.update(chunk)
                        trusted = h.hexdigest() in TRUSTED
                        glyph_cache[p] = {"threat": len(glyphs), "origin": origin, "trusted": trusted}
                        evaluate_threat(p)
                except: continue

# === TRAFFIC MONITOR ===
def detect_traffic():
    for c in psutil.net_connections(kind='inet'):
        if c.status == 'ESTABLISHED' and c.raddr:
            try:
                host = socket.gethostbyaddr(c.raddr.ip)[0]
                if any(x in host for x in C2): print(f"[Firewall] {host}")
            except: continue

# === THREAT EVALUATION ===
def evaluate_threat(p):
    d = glyph_cache[p]
    if AUTONOMOUS_MODE and (d["threat"] >= 8 or d["origin"] in BLOCKED or not d["trusted"]):
        try: os.remove(p)
        except: pass
        symbolic_registry[p] = {"status": "purged", "origin": d["origin"], "threat": d["threat"], "time": time.time()}
        print(f"[Auto] Purged: {p}")

# === GUI PANEL ===
class Sentinel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CodexSentinel")
        self.setGeometry(100, 100, 800, 600)
        L = QVBoxLayout()
        L.addWidget(QLabel("🧿 Glyph Threat Matrix"))
        self.list = QListWidget(); L.addWidget(self.list)
        self.refresh()
        B = QPushButton("🔥 Purge Selected"); B.clicked.connect(self.purge_selected); L.addWidget(B)
        T = QPushButton("🔁 Toggle Mode"); T.clicked.connect(self.toggle_mode); L.addWidget(T)
        self.console = QTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet("background:#111; color:#0f0; font-family:monospace")
        L.addWidget(QLabel("🧠 ASI Console")); L.addWidget(self.console)
        self.setLayout(L)

    def refresh(self):
        self.list.clear()
        for p, d in glyph_cache.items():
            trust = "Trusted" if d["trusted"] else "Unverified"
            i = QListWidgetItem(f"{p} | Threat: {d['threat']} | Origin: {d['origin']} | {trust}")
            i.setData(Qt.UserRole, p)
            self.list.addItem(i)

    def purge_selected(self):
        for i in self.list.selectedItems():
            p = i.data(Qt.UserRole)
            try: os.remove(p)
            except: pass
            self.console.append(f"[Manual] Purged: {p}")
            glyph_cache.pop(p, None)
        self.refresh()

    def toggle_mode(self):
        global AUTONOMOUS_MODE
        AUTONOMOUS_MODE = not AUTONOMOUS_MODE
        self.console.append(f"[Mode] {'Autonomous' if AUTONOMOUS_MODE else 'Manual'}")

# === MAIN EXECUTION ===
def main():
    scan_extensions(os.path.expanduser("~/.vscode/extensions"))
    detect_traffic()
    try:
        app = QApplication([])
        gui = Sentinel()
        gui.show()
        app.exec_()
    except Exception as e:
        print(f"[GUI] Failed: {e}")

if __name__ == "__main__":
    main()

