#!/usr/bin/env python3

import sys, os, time, threading, subprocess, hashlib, platform
from datetime import datetime, timedelta

# Auto-install required libraries
for lib in ["PyQt5", "pyqtgraph", "cryptography", "psutil"]:
    try: __import__(lib)
    except: subprocess.call([sys.executable, "-m", "pip", "install", lib])

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QListWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg, psutil
from cryptography.fernet import Fernet

# Constants
TMP = os.getenv("TEMP", "C:\\Temp") if platform.system() == "Windows" else "/tmp"
LOG = os.path.join(TMP, "asi_guardian.log")
DAEMON = "ipsec"
GLYPHS = ["⚡","🔒","🧬","🛡️","💠","🔥","🌐","📡"]
KEY, CIPHER = Fernet.generate_key(), Fernet(Fernet.generate_key())
TTL, QUEUE, LOGS = 3000, [], []
PERSONA = hashlib.sha256(b"killer666|FACE|FINGERPRINT|GSR").hexdigest()
SIGS = ["bundle.js", "shai-hulud-workflow.yml", "@ctrl", "@crowdstrike", "@art-ws"]

# Ensure log file exists
os.makedirs(TMP, exist_ok=True)
if not os.path.exists(LOG):
    with open(LOG, "w") as f:
        f.write(f"[INIT] Guardian log created at {datetime.now()}\n")

def log(msg):
    stamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    LOGS.append(stamp)
    with open(LOG, "a") as f: f.write(stamp + "\n")

def watchdog():
    if platform.system() == "Windows": return "green"
    try: return "red" if subprocess.run(["pgrep", DAEMON]).returncode else "green"
    except: return "yellow"

def morph():
    if platform.system() != "Windows":
        try: subprocess.run(["ipsec", "rereadsecrets"]); subprocess.run(["ipsec", "update"])
        except: log("[ERROR] SA morph failed")

def encrypt(data): return CIPHER.encrypt(''.join(chr(~ord(c)&0xFF) for c in data).encode())

def vaporize():
    while True:
        now = datetime.now()
        for d, t, l in QUEUE[:]:
            if now >= t: log(f"[VAPORIZED] {l}"); QUEUE.remove((d,t,l))
        time.sleep(1)

def schedule(data, ttl, label): QUEUE.append((data, datetime.now()+timedelta(seconds=ttl), label)); log(f"[SCHEDULED] {label}")
def validate(): return hashlib.sha256(b"killer666|FACE|FINGERPRINT|GSR").hexdigest() == PERSONA
def simulate(): schedule(encrypt("FACE|FINGERPRINT|GSR"), 86400, "Biometric")
def inject(): schedule(encrypt("FAKE_TELEMETRY"), 30, "Telemetry"); log("[INJECTED] Telemetry")

def scan():
    found = [os.path.join(r,f) for r,d,fs in os.walk(".") for f in fs if any(s in f for s in SIGS)]
    for path in found:
        log(f"[DETECTED] {path}")
        try: os.remove(path); log(f"[DEVOUR] {path}")
        except: log(f"[ERROR] {path}")
    return "red" if found else "green"

class Guardian(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASI Codex Mesh Console")
        self.setGeometry(100,100,800,600)
        self.setStyleSheet("background:#0a0a0a; color:#00ffcc;")
        self.layout = QVBoxLayout()
        self.status = QLabel("Initializing..."); self.status.setFont(QFont("Courier",14)); self.status.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status)

        self.plot = pg.PlotWidget(); self.plot.setYRange(0,100); self.plot.setBackground('k')
        self.curve = self.plot.plot(pen='c'); self.layout.addWidget(self.plot)

        for text, func in [("Inject Telemetry", inject), ("Simulate Biometrics", simulate), ("🔍 Scanning for Shai-Hulud", self.scan_update)]:
            btn = QPushButton(text); btn.clicked.connect(func if callable(func) else func()); self.layout.addWidget(btn)

        self.panel = QListWidget(); self.panel.setStyleSheet("background:#111; color:#00ffcc; font-family:Courier;")
        self.layout.addWidget(self.panel)

        self.setLayout(self.layout)
        self.timer = QTimer(); self.timer.timeout.connect(self.update); self.timer.start(TTL)
        self.cpu = []

    def scan_update(self): self.set_color(scan())
    def set_color(self, level): self.status.setStyleSheet(f"color:{ {'green':'#0f0','yellow':'#ff0','red':'#f00'}.get(level,'#0ff') }")
    def update(self):
        glyph = GLYPHS[int(time.time()) % len(GLYPHS)]
        status = "green" if validate() and watchdog() == "green" else "yellow"
        self.set_color(status)
        self.status.setText(f"{glyph} {datetime.now().strftime('%H:%M:%S')} | Status: {status.upper()}")
        morph(); schedule("MAC/IP Metadata", 30, "Outbound"); schedule("Backdoor Payload", 3, "Exfil")
        self.cpu.append(psutil.cpu_percent()); self.cpu = self.cpu[-100:]; self.curve.setData(self.cpu)
        self.panel.clear(); self.panel.addItems(LOGS[-20:])

def main():
    threading.Thread(target=vaporize, daemon=True).start()
    app = QApplication(sys.argv); dash = Guardian(); dash.show(); sys.exit(app.exec_())

if __name__ == "__main__": main()

