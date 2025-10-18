# 🔧 Autoloader: Ensure required libraries are present
import subprocess
import sys

def ensure(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[Autoloader] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["tkinter"]:
    ensure(pkg)

# 🧿 Codex Sentinel Daemon
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta

# 🧬 Codex Rules
class Codex:
    def __init__(self):
        self.retention_hours = 24
        self.phantom_nodes = set()
        self.allowed_countries = {"US", "CA", "UK"}

    def should_purge(self, threat):
        if threat["source_node"] in self.phantom_nodes:
            return True
        return threat["timestamp"] < datetime.now() - timedelta(hours=self.retention_hours)

    def should_block(self, origin):
        return origin not in self.allowed_countries

    def mutate_on_ghost_sync(self):
        self.retention_hours = max(1, self.retention_hours // 2)
        self.phantom_nodes.add("ghost-sync")
        print("[Codex] Mutation triggered. Retention shortened.")

# 🎭 Persona Engine
class PersonaEngine:
    def __init__(self):
        self.personas = []

    def inject(self, name):
        self.personas.append({"name": name, "status": "active"})
        print(f"[Persona] {name} has entered the ritual.")

# 🌐 Swarm Sync
class SwarmSync:
    def __init__(self):
        self.nodes = {}

    def broadcast(self, message):
        for node in self.nodes:
            self.nodes[node] = message
        print(f"[Swarm] Broadcast: {message}")

    def detect_phantom(self, node_id):
        if self.nodes.get(node_id) == "ghost-sync":
            print("[Swarm] Phantom node detected.")
            return True
        return False

# 🔥 Purge Daemon
class PurgeDaemon:
    def __init__(self, codex):
        self.codex = codex

    def evaluate(self, threats):
        for threat in threats:
            if self.codex.should_purge(threat):
                try:
                    os.remove(threat["path"])
                    print(f"[Purge] {threat['path']} purged.")
                except Exception as e:
                    print(f"[Error] {e}")

# 🖥️ GUI Shell
class CodexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Sentinel")
        self.codex = Codex()
        self.persona_engine = PersonaEngine()
        self.swarm = SwarmSync()
        self.purge_daemon = PurgeDaemon(self.codex)
        self.quarantine = []

        self.setup_gui()
        self.start_monitoring()

    def setup_gui(self):
        self.status = tk.StringVar()
        self.status.set("🟢 Idle")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 14)).pack(pady=10)
        ttk.Button(self.root, text="Inject ThreatHunter", command=lambda: self.persona_engine.inject("ThreatHunter")).pack()
        ttk.Button(self.root, text="Trigger Ghost Sync", command=self.trigger_ghost_sync).pack()
        ttk.Button(self.root, text="Manual Purge", command=self.manual_purge).pack()

    def start_monitoring(self):
        path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\WebCache")
        def monitor():
            seen = set()
            while True:
                try:
                    for fname in os.listdir(path):
                        fpath = os.path.join(path, fname)
                        if fpath not in seen:
                            seen.add(fpath)
                            threat = {
                                "path": fpath,
                                "source_node": "local",
                                "timestamp": datetime.now(),
                                "type": "telemetry",
                                "origin": "RU"
                            }
                            if self.codex.should_block(threat["origin"]):
                                self.quarantine.append(threat)
                                self.status.set("🟡 Quarantine")
                                print(f"[Quarantine] {fpath}")
                    time.sleep(5)
                except Exception as e:
                    print(f"[Monitor Error] {e}")
        threading.Thread(target=monitor, daemon=True).start()

    def manual_purge(self):
        self.purge_daemon.evaluate(self.quarantine)
        self.quarantine.clear()
        self.status.set("🟢 Idle")

    def trigger_ghost_sync(self):
        self.swarm.nodes["node-1"] = "ghost-sync"
        if self.swarm.detect_phantom("node-1"):
            self.codex.mutate_on_ghost_sync()
            self.status.set("🧿 Phantom Node")

# 🧬 Launch Ritual
if __name__ == "__main__":
    root = tk.Tk()
    app = CodexGUI(root)
    root.mainloop()

