# codex_gui_resolver.py

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧠 CodexNet GUI Resolver + Swarm Sync + Autoloader + Cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 🔁 Autoloader: ensure all libraries are present
import subprocess
import sys
import os

REQUIRED_LIBRARIES = [
    "tkinter", "json", "datetime", "threading"
]

PIP_PACKAGES = {
    "tkinter": "tk"
}

def ensure_libraries():
    for lib in REQUIRED_LIBRARIES:
        try:
            __import__(lib)
        except ImportError:
            if lib in PIP_PACKAGES:
                print(f"[AUTOLOADER] Installing: {lib}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", PIP_PACKAGES[lib]])
            else:
                print(f"[AUTOLOADER] ERROR: Missing '{lib}' with no pip mapping.")
                sys.exit(1)

ensure_libraries()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 Imports (after autoloader)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
from datetime import datetime
from threading import Thread

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📜 Embedded Symbolic Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYMBOLIC_REGISTRY = {
    "🜁_AetherNode": {
        "ip": "192.168.1.10",
        "origin": "internal",
        "threat_level": "none",
        "last_sync": None
    },
    "🜂_PurgeDaemon": {
        "ip": "192.168.1.20",
        "origin": "internal",
        "threat_level": "low",
        "last_sync": None
    },
    "🜃_AuditGlyph": {
        "ip": "192.168.1.30",
        "origin": "internal",
        "threat_level": "none",
        "last_sync": None
    },
    "🜄_IngestDaemon": {
        "ip": "192.168.1.40",
        "origin": "external",
        "threat_level": "medium",
        "last_sync": None
    },
    "🝃_CodexMutator": {
        "ip": "192.168.1.50",
        "origin": "internal",
        "threat_level": "low",
        "last_sync": None
    },
    "🝞_PersonaGate": {
        "ip": "192.168.1.60",
        "origin": "external",
        "threat_level": "high",
        "last_sync": None
    },
    "🝓_EventBus": {
        "ip": "192.168.1.70",
        "origin": "internal",
        "threat_level": "none",
        "last_sync": None
    },
    "🝖_ThreatMatrix": {
        "ip": "192.168.1.80",
        "origin": "internal",
        "threat_level": "critical",
        "last_sync": None
    },
    "🝗_CountryFilter": {
        "ip": "192.168.1.90",
        "origin": "external",
        "threat_level": "medium",
        "last_sync": None
    },
    "🝙_SwarmSyncCore": {
        "ip": "192.168.1.100",
        "origin": "internal",
        "threat_level": "none",
        "last_sync": None
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧬 Persona Clearance Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSONA_CLEARANCE = {
    "admin": list(SYMBOLIC_REGISTRY.keys()),
    "observer": [k for k, v in SYMBOLIC_REGISTRY.items() if v["origin"] == "internal"],
    "quarantined": []
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 Symbolic Resolver with Cache + Threat Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CACHE_PATH = "codex_cache.json"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        CACHE = json.load(f)
else:
    CACHE = {}

def resolve_symbol(symbol, persona):
    if symbol in CACHE:
        return {
            "status": "cached",
            "symbol": symbol,
            "address": CACHE[symbol]["ip"],
            "resolved_by": CACHE[symbol]["resolved_by"],
            "timestamp": CACHE[symbol]["timestamp"]
        }

    if symbol not in SYMBOLIC_REGISTRY:
        return {"status": "error", "message": "Unknown symbol"}
    if symbol not in PERSONA_CLEARANCE.get(persona, []):
        return {"status": "blocked", "message": "Clearance denied"}

    threat = SYMBOLIC_REGISTRY[symbol]["threat_level"]
    if threat == "critical" and persona != "admin":
        return {"status": "blocked", "message": "Threat level too high"}

    CACHE[symbol] = {
        "ip": SYMBOLIC_REGISTRY[symbol]["ip"],
        "resolved_by": persona,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(CACHE, f, indent=2)

    return {
        "status": "resolved",
        "symbol": symbol,
        "address": SYMBOLIC_REGISTRY[symbol]["ip"],
        "origin": SYMBOLIC_REGISTRY[symbol]["origin"],
        "threat_level": threat,
        "timestamp": CACHE[symbol]["timestamp"]
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔄 Swarm Sync Ritual
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sync_node(symbol):
    if symbol not in SYMBOLIC_REGISTRY:
        return {"status": "error", "message": "Unknown node"}
    SYMBOLIC_REGISTRY[symbol]["last_sync"] = datetime.utcnow().isoformat()
    return {
        "status": "synced",
        "node": symbol,
        "ip": SYMBOLIC_REGISTRY[symbol]["ip"],
        "timestamp": SYMBOLIC_REGISTRY[symbol]["last_sync"]
    }

def daemon_loop():
    while True:
        for symbol in SYMBOLIC_REGISTRY:
            sync_node(symbol)
        update_swarm_panel()
        time.sleep(30)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🖥️ GUI Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

root = tk.Tk()
root.title("🧠 CodexNet Resolver + Swarm Sync")
root.geometry("800x600")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔮 Symbolic Resolver Panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tk.Label(root, text="Symbolic Name:").pack()
symbol_entry = ttk.Combobox(root, values=list(SYMBOLIC_REGISTRY.keys()))
symbol_entry.pack()

tk.Label(root, text="Persona:").pack()
persona_entry = ttk.Combobox(root, values=list(PERSONA_CLEARANCE.keys()))
persona_entry.pack()

output_box = tk.Text(root, height=10, width=95)
output_box.pack(pady=10)

def on_resolve():
    symbol = symbol_entry.get()
    persona = persona_entry.get()
    result = resolve_symbol(symbol, persona)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, json.dumps(result, indent=2))
    if result["status"] not in ["resolved", "cached"]:
        messagebox.showwarning("Resolution Failed", result["message"])

ttk.Button(root, text="Resolve", command=on_resolve).pack(pady=5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧬 Swarm Sync Panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tk.Label(root, text="🧬 Swarm Node Status").pack(pady=10)
swarm_frame = tk.Frame(root)
swarm_frame.pack()

swarm_labels = {}

def update_swarm_panel():
    for widget in swarm_frame.winfo_children():
        widget.destroy()
    for symbol, data in SYMBOLIC_REGISTRY.items():
        text = f"{symbol} → {data['ip']} | Origin: {data['origin']} | Threat: {data['threat_level']} | Last Sync: {data.get('last_sync', 'Pending')}"
        lbl = tk.Label(swarm_frame, text=text, anchor="w", width=110)
        lbl.pack()
        swarm_labels[symbol] = lbl

Thread(target=daemon_loop, daemon=True).start()
root.mainloop()

