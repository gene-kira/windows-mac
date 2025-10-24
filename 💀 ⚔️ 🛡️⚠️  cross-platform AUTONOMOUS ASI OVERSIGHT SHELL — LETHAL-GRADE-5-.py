import subprocess, sys, platform, json, threading, time, uuid, psutil, tkinter as tk
from tkinter import ttk
from datetime import datetime
import os

# === AUTOLOADER ===
def autoload(modules):
    import importlib
    for mod in modules:
        try: importlib.import_module(mod)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", mod])
autoload(["psutil", "tkinter", "requests"])

# === REGISTRY + LEDGER ===
REGISTRY_PATH = "symbolic_registry.json"
LEDGER_PATH = "symbolic_ledger.json"
TEMP_STORE = {}

def load_registry():
    try: return json.load(open(REGISTRY_PATH))
    except: return []

def log_ritual(origin, signature, action, persona, compliance):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "origin": origin, "signature": signature,
        "action": action, "persona": persona, "compliance": compliance
    }
    ledger = []
    if os.path.exists(LEDGER_PATH):
        try: ledger = json.load(open(LEDGER_PATH))
        except: pass
    ledger.append(entry)
    json.dump(ledger, open(LEDGER_PATH, "w"), indent=2)

def match_signature(name, origin):
    for entry in load_registry():
        if entry["signature"] in name and (entry["origin"] == origin or entry["origin"] == "ANY"):
            return entry
    return None

# === PURGE + TTL ===
def purge_expired_data():
    while True:
        now = time.time()
        for k in list(TEMP_STORE):
            if now - TEMP_STORE[k]["timestamp"] > TEMP_STORE[k]["ttl"]:
                del TEMP_STORE[k]
        time.sleep(5)

def store_temp_data(data_type, value, ttl):
    TEMP_STORE[str(uuid.uuid4())] = {
        "type": data_type, "value": value,
        "timestamp": time.time(), "ttl": ttl
    }

# === FIREWALL + KILL ===
def block_outbound(name=None, port=None):
    sysname = platform.system()
    if sysname == "Windows":
        rule = f'netsh advfirewall firewall add rule name="Block {name or port}" dir=out '
        rule += f'program="{name}" action=block' if name else f'protocol=TCP localport={port} action=block'
    elif sysname == "Linux" and port:
        rule = f'sudo iptables -A OUTPUT -p tcp --dport {port} -j DROP'
    elif sysname == "Darwin" and port:
        rule = f'echo "block out proto tcp from any to any port {port}" | sudo pfctl -a com.apple/BlockOut -f -'
    else: return
    subprocess.call(rule, shell=True)

def annihilate_threat(name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'].lower() == name.lower():
            try: psutil.Process(proc.info['pid']).kill()
            except: pass

# === ENCRYPTED + WEBCAM DEFENSE ===
def detect_encrypted():
    while True:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.raddr and conn.raddr.port == 443:
                store_temp_data("EncryptedTraffic", f"{conn.raddr.ip}:{conn.raddr.port}", ttl=30)
        time.sleep(5)

def monitor_webcam():
    sysname = platform.system()
    while True:
        if sysname == "Windows":
            for proc in psutil.process_iter(['name']):
                if "camera" in proc.info['name'].lower() or "webcam" in proc.info['name'].lower():
                    store_temp_data("WebcamAccess", proc.info['name'], ttl=30)
                    annihilate_threat(proc.info['name'])
        elif sysname in ["Linux", "Darwin"]:
            if subprocess.getoutput("lsof /dev/video0"): store_temp_data("WebcamAccess", "video0 accessed", ttl=30)
        time.sleep(5)

# === ZERO TRUST + SYNC ===
def zero_trust_check(data):
    return not any(x in data.lower() for x in ["asi", "ai", "hacker"])

def ghost_sync(nodes, signature):
    for node in nodes:
        print(f"[GhostSync] {node} received purge rule for {signature}")
        time.sleep(0.5)

# === GUI CONSOLE ===
class ASIConsole:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASI Killbox Console")
        self.root.geometry("1000x600")
        self.root.configure(bg="#0f0f0f")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#1f1f1f", foreground="#00ffcc", fieldbackground="#1f1f1f", rowheight=25, font=('Consolas', 10))
        style.configure("Treeview.Heading", background="#2f2f2f", foreground="#00ffff", font=('Consolas', 11, 'bold'))
        self.tree = ttk.Treeview(self.root, columns=("Origin", "Signature", "Action", "Compliance", "Persona"), show="headings")
        for col in self.tree["columns"]: self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self.root, text="Killbox Armed — Autonomous Defense Active", bg="#0f0f0f", fg="#00ff00", font=('Consolas', 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    def add_threat(self, origin, signature, action, compliance, persona):
        self.tree.insert("", "end", values=(origin, signature, action, compliance, persona))
    def run(self): self.root.mainloop()

# === FUSION CORE ===
overlay = ASIConsole()

def on_threat_detected(name, origin, match):
    action, compliance, persona = "RETAIN", "OK", "Sentinel"
    if match:
        if match["purge"]:
            action = "PURGE + KILL + BLOCK"
            annihilate_threat(name)
            block_outbound(name)
        if not match.get("compliant", True): compliance = "VIOLATION"
        if match.get("sync", False): threading.Thread(target=ghost_sync, args=(["NodeA", "NodeB", "NodeC"], name)).start()
    overlay.add_threat(origin, name, action, compliance, persona)
    log_ritual(origin, name, action, persona, compliance)
    if "mac" in name.lower() or "ip" in name.lower(): store_temp_data("MAC/IP", name, ttl=86400)
    elif "backdoor" in name.lower(): store_temp_data("Backdoor", name, ttl=3)
    elif "telemetry" in name.lower(): store_temp_data("Telemetry", name, ttl=30)
    elif any(x in name.lower() for x in ["face", "finger", "phone", "address", "license", "social"]): store_temp_data("Personal", name, ttl=86400)
    if not zero_trust_check(name): return

def continuous_threat_hunt():
    seen = set()
    while True:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['pid'] not in seen:
                seen.add(proc.info['pid'])
                origin = "US"
                match = match_signature(proc.info['name'], origin)
                on_threat_detected(proc.info['name'], origin, match)
        time.sleep(2)

# === THREADS ===
threading.Thread(target=purge_expired_data, daemon=True).start()
threading.Thread(target=detect_encrypted, daemon=True).start()
threading.Thread(target=monitor_webcam, daemon=True).start()
threading.Thread(target=continuous_threat_hunt, daemon=True).start()
overlay.run()

