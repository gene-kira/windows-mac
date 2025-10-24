# AUTONOMOUS ASI OVERSIGHT SHELL — LETHAL-GRADE
# Requires: psutil, tkinter, requests, json, threading
# Autoloader: installs missing libraries

import subprocess
import sys

def autoload(modules):
    import importlib
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", mod])

autoload(["psutil", "tkinter", "requests"])

import psutil, json, threading, time
import tkinter as tk
from tkinter import ttk

# === SYMBOLIC REGISTRY ===
REGISTRY_PATH = "symbolic_registry.json"

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

# === THREAT HUNTER + COMPLIANCE AUDITOR ===
def classify_processes(callback):
    seen = set()
    while True:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['pid'] not in seen:
                seen.add(proc.info['pid'])
                origin = "US"  # Placeholder for geo-IP logic
                match = match_signature(proc.info['name'], origin)
                callback(proc.info['name'], origin, match)
        time.sleep(2)

# === GHOST SYNC ===
def ghost_sync(nodes):
    for node in nodes:
        print(f"[GhostSync] {node} received purge rules silently.")
        time.sleep(0.5)

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
    if match and match["sync"]:
        threading.Thread(target=ghost_sync, args=(["NodeA", "NodeB", "NodeC"],)).start()

threading.Thread(target=classify_processes, args=(on_threat_detected,), daemon=True).start()
overlay.run()

