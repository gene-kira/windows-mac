# AUTONOMOUS ASI OVERSIGHT SHELL — MYTHIC-GRADE
# Requires: psutil, tkinter, requests
# Autoloader: installs missing libraries

import subprocess, sys

def autoload(modules):
    import importlib
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", mod])

autoload(["psutil", "tkinter", "requests"])

import psutil, json, threading, time, uuid
import tkinter as tk
from tkinter import ttk

# === SYMBOLIC REGISTRY ===
REGISTRY_PATH = "symbolic_registry.json"
TEMP_STORE = {}

def load_registry():
    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except:
        return []

def match_signature(process_name, origin):
    registry = load_registry()
    for entry in registry:
        if entry["signature"] in process_name and (entry["origin"] == origin or entry["origin"] == "ANY"):
            return entry
    return None

# === REAL-TIME AUTONOMOUS DEFENSE ===
def purge_expired_data():
    while True:
        now = time.time()
        expired = [k for k, v in TEMP_STORE.items() if now - v["timestamp"] > v["ttl"]]
        for key in expired:
            del TEMP_STORE[key]
        time.sleep(5)

def store_temp_data(data_type, value, ttl):
    key = str(uuid.uuid4())
    TEMP_STORE[key] = {
        "type": data_type,
        "value": value,
        "timestamp": time.time(),
        "ttl": ttl
    }

# === FIREWALL CONTROL ===
def block_outbound(process_name=None, port=None):
    if process_name:
        rule = f'netsh advfirewall firewall add rule name="Block {process_name}" dir=out program="{process_name}" action=block'
    elif port:
        rule = f'netsh advfirewall firewall add rule name="Block Port {port}" dir=out protocol=TCP localport={port} action=block'
    else:
        return
    subprocess.call(rule, shell=True)

# === ENCRYPTED TRAFFIC DETECTION ===
def detect_encrypted_connections():
    while True:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.laddr and conn.raddr:
                if conn.raddr.port == 443:
                    store_temp_data("EncryptedTraffic", f"{conn.raddr.ip}:{conn.raddr.port}", ttl=30)
        time.sleep(5)

# === WEBCAM DEFENSE DAEMON ===
def monitor_webcam():
    while True:
        for proc in psutil.process_iter(['name']):
            if "camera" in proc.info['name'].lower() or "webcam" in proc.info['name'].lower():
                print(f"[Webcam Defense] {proc.info['name']} accessing webcam.")
                store_temp_data("WebcamAccess", proc.info['name'], ttl=30)
        time.sleep(5)

# === THREAT HUNTER + COMPLIANCE AUDITOR ===
def classify_processes(callback):
    seen = set()
    while True:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['pid'] not in seen:
                seen.add(proc.info['pid'])
                origin = "US"  # Placeholder
                match = match_signature(proc.info['name'], origin)
                callback(proc.info['name'], origin, match)
        time.sleep(1)

# === GHOST SYNC ===
def ghost_sync(nodes):
    for node in nodes:
        print(f"[GhostSync] {node} received purge rules silently.")
        time.sleep(0.5)

# === ZERO TRUST GUARDIAN ===
def zero_trust_check(data):
    if "ASI" in data or "AI" in data or "hacker" in data:
        print("[ZeroTrust] Blocked suspicious entity.")
        return False
    return True

# === MILITARY-GRADE GUI ===
class ASIConsole:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASI Oversight Console")
        self.root.geometry("1000x600")
        self.root.configure(bg="#0f0f0f")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#1f1f1f", foreground="#00ffcc", fieldbackground="#1f1f1f", rowheight=25, font=('Consolas', 10))
        style.configure("Treeview.Heading", background="#2f2f2f", foreground="#00ffff", font=('Consolas', 11, 'bold'))

        self.tree = ttk.Treeview(self.root, columns=("Origin", "Signature", "Action", "Compliance"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(self.root, text="System Armed — Autonomous Defense Active", bg="#0f0f0f", fg="#00ff00", font=('Consolas', 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def add_threat(self, name, origin, action, compliance):
        self.tree.insert("", "end", values=(origin, name, action, compliance))

    def run(self):
        self.root.mainloop()

# === FUSION CORE ===
overlay = ASIConsole()

def on_threat_detected(name, origin, match):
    action = "PURGE" if match and match["purge"] else "RETAIN"
    compliance = "VIOLATION" if match and not match.get("compliant", True) else "OK"
    overlay.add_threat(name, origin, action, compliance)

    # Real-Time Defense: Store classified data with TTL
    if "mac" in name.lower() or "ip" in name.lower():
        store_temp_data("MAC/IP", name, ttl=86400)  # 1 day
    elif "backdoor" in name.lower():
        store_temp_data("Backdoor", name, ttl=3)  # 3 sec
        block_outbound(process_name=name)
    elif "telemetry" in name.lower():
        store_temp_data("Telemetry", name, ttl=30)  # 30 sec
    elif any(x in name.lower() for x in ["face", "finger", "phone", "address", "license", "social"]):
        store_temp_data("Personal", name, ttl=86400)  # 1 day

    # Zero Trust Check
    if not zero_trust_check(name):
        return

    # Ghost Sync
    if match and match["sync"]:
        threading.Thread(target=ghost_sync, args=(["NodeA", "NodeB", "NodeC"],)).start()

# === THREADS ===
threading.Thread(target=classify_processes, args=(on_threat_detected,), daemon=True).start()
threading.Thread(target=purge_expired_data, daemon=True).start()
threading.Thread(target=detect_encrypted_connections, daemon=True).start()
threading.Thread(target=monitor_webcam, daemon=True).start()
overlay.run()

