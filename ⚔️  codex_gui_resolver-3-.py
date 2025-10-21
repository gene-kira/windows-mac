# codex_gui_resolver.py

import subprocess, sys, os, json, time, tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from threading import Thread

# ━━━━━━━━━━━ AUTOLOADER ━━━━━━━━━━━
for lib in ["tkinter", "json", "datetime", "threading"]:
    try: __import__(lib)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])

# ━━━━━━━━━━━ REGISTRY + CACHE ━━━━━━━━━━━
REGISTRY = {
    "🜁_AetherNode": {"ip": "192.168.1.10", "origin": "internal", "threat_level": "none"},
    "🜂_PurgeDaemon": {"ip": "192.168.1.20", "origin": "internal", "threat_level": "low"},
    "🜃_AuditGlyph": {"ip": "192.168.1.30", "origin": "internal", "threat_level": "none"},
    "🜄_IngestDaemon": {"ip": "192.168.1.40", "origin": "external", "threat_level": "medium"},
    "🝃_CodexMutator": {"ip": "192.168.1.50", "origin": "internal", "threat_level": "low"},
    "🝞_PersonaGate": {"ip": "192.168.1.60", "origin": "external", "threat_level": "high"},
    "🝓_EventBus": {"ip": "192.168.1.70", "origin": "internal", "threat_level": "none"},
    "🝖_ThreatMatrix": {"ip": "192.168.1.80", "origin": "internal", "threat_level": "critical"},
    "🝗_CountryFilter": {"ip": "192.168.1.90", "origin": "external", "threat_level": "medium"},
    "🝙_SwarmSyncCore": {"ip": "192.168.1.100", "origin": "internal", "threat_level": "none"}
}
CLEARANCE = {
    "admin": list(REGISTRY),
    "observer": [k for k, v in REGISTRY.items() if v["origin"] == "internal"],
    "quarantined": []
}
DEFAULT_PATH = "codex_cache.json"
CACHE = json.load(open(DEFAULT_PATH)) if os.path.exists(DEFAULT_PATH) else {}

# ━━━━━━━━━━━ GUI SETUP ━━━━━━━━━━━
root = tk.Tk(); root.title("CodexNet Resolver"); root.geometry("850x650")
tk.Label(root, text="Select Drive:").pack()
drive_var = tk.StringVar(); ttk.Combobox(root, textvariable=drive_var, values=["C:\\", "D:\\", "E:\\"]).pack()
tk.Label(root, text="Symbolic Name:").pack()
symbol_entry = ttk.Combobox(root, values=list(REGISTRY)); symbol_entry.pack()
tk.Label(root, text="Persona:").pack()
persona_entry = ttk.Combobox(root, values=list(CLEARANCE)); persona_entry.pack()
output_box = tk.Text(root, height=10, width=100); output_box.pack(pady=10)

# ━━━━━━━━━━━ RESOLUTION + CACHE ━━━━━━━━━━━
def get_path(): return os.path.join(drive_var.get(), "codex_cache.json") if os.path.exists(drive_var.get()) else DEFAULT_PATH
def flush(): json.dump(CACHE, open(get_path(), "w"), indent=2)
def resolve():
    s, p = symbol_entry.get(), persona_entry.get()
    if s in CACHE: r = {"status": "cached", **CACHE[s], "symbol": s}
    elif s not in REGISTRY: r = {"status": "error", "message": "Unknown symbol"}
    elif s not in CLEARANCE.get(p, []): r = {"status": "blocked", "message": "Clearance denied"}
    elif REGISTRY[s]["threat_level"] == "critical" and p != "admin": r = {"status": "blocked", "message": "Threat level too high"}
    else:
        CACHE[s] = {"ip": REGISTRY[s]["ip"], "resolved_by": p, "timestamp": datetime.utcnow().isoformat()}
        r = {"status": "resolved", **CACHE[s], "symbol": s, "origin": REGISTRY[s]["origin"], "threat_level": REGISTRY[s]["threat_level"]}
    output_box.delete("1.0", tk.END); output_box.insert(tk.END, json.dumps(r, indent=2))
    if r["status"] not in ["resolved", "cached"]: messagebox.showwarning("Resolution Failed", r["message"])

ttk.Button(root, text="Resolve", command=resolve).pack(pady=5)
ttk.Button(root, text="Save Cache", command=flush).pack(pady=5)

# ━━━━━━━━━━━ SWARM PANEL ━━━━━━━━━━━
tk.Label(root, text="Swarm Node Status").pack(pady=10)
swarm_frame = tk.Frame(root); swarm_frame.pack()
def update_swarm():
    for w in swarm_frame.winfo_children(): w.destroy()
    for k, v in REGISTRY.items():
        sync = v.get("last_sync", "Pending")
        lbl = tk.Label(swarm_frame, text=f"{k} → {v['ip']} | Origin: {v['origin']} | Threat: {v['threat_level']} | Last Sync: {sync}", anchor="w", width=110)
        lbl.pack()

# ━━━━━━━━━━━ DAEMON + SHUTDOWN ━━━━━━━━━━━
def daemon():
    last = time.time()
    while True:
        for k in REGISTRY: REGISTRY[k]["last_sync"] = datetime.utcnow().isoformat()
        update_swarm()
        if time.time() - last >= 900: flush(); last = time.time()
        time.sleep(30)

def shutdown(): flush(); root.destroy()
root.protocol("WM_DELETE_WINDOW", shutdown)
Thread(target=daemon, daemon=True).start()
root.mainloop()

