# 🔄 Autoloader
import subprocess, sys
for pkg in ["flask", "flask_socketio", "psutil", "requests"]:
    try: __import__(pkg if pkg != "flask_socketio" else "flask_socketio")
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 🔧 Core Imports
import os, json, socket, threading, psutil, random
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import queue

# ⚙️ Init
app, socketio, q = Flask(__name__), SocketIO(Flask(__name__), cors_allowed_origins="*"), queue.Queue()
registry, swarm, clusters = {}, set(), {}
baseline = {"hours": range(8,18), "countries": {"US","CA"}, "protocols": {"HTTPS","DNS"}, "avg_kb": 500}
clearance = {"admin":10, "analyst":5, "guest":1}
safe_ips = {"127.0.0.1", "192.168.1.1"}

# 🧠 Detection
def classify(data): return sum([
    5 if "exec" in data or "rm -rf" in data else 0,
    2 if "http://" in data or "ftp://" in data else 0,
    3 if "password" in data else 0
])

def gut(event):
    h = datetime.fromisoformat(event["ts"]).hour
    return (
        h not in baseline["hours"] or
        event["country"] not in baseline["countries"] or
        event["proto"] not in baseline["protocols"] or
        event["kb"] > baseline["avg_kb"] * 2 or
        event["persona"] not in {"admin", "analyst"} if event["act"] == "ingest" else {"admin"}
    )

def encrypted(event):
    return event["entropy"] > 7.5 or event["tls"] not in {"TLS 1.2", "TLS 1.3"}

def zero_day(event):
    return not (6.5 <= event["entropy"] <= 7.5 and event["tls"] in {"TLS 1.2", "TLS 1.3"})

def social_engineering(event):
    return event["persona"] == "admin" and event["data"].startswith("exec") and event["kb"] > 1000

def evolve(event):
    if event.get("gut"):
        baseline["avg_kb"] = int(0.9 * baseline["avg_kb"] + 0.1 * event["kb"])
        baseline["countries"].add(event["country"])
        baseline["protocols"].add(event["proto"])

def cluster(event):
    cid = hash((event["origin"], event["threat"], event["ts"][:13]))
    clusters.setdefault(cid, []).append(event)
    if len(clusters[cid]) >= 3:
        purge(f"Cluster {cid} triggered purge")
        q.put(("CLUSTER", f"Swarm consensus purge: Cluster {cid}"))

def purge(msg): q.put(("PURGE", f"[PURGE] {msg}"))

# 🔥 Ingest
def ingest(data, origin, persona):
    ts = datetime.utcnow().isoformat()
    event = {
        "origin": origin, "data": data, "persona": persona, "ts": ts,
        "country": origin, "proto": "HTTPS", "kb": len(data.encode())//1024,
        "act": "ingest", "entropy": random.uniform(6.5,8.5),
        "tls": random.choice(["TLS 1.1","TLS 1.2","TLS 1.3"]),
        "threat": classify(data),
        "lineage": {"origin": origin, "ts": ts, "mutations": [], "purge": None}
    }
    registry[ts] = event
    if any([gut(event), encrypted(event), zero_day(event), social_engineering(event)]):
        event["gut"] = True
        event["lineage"]["purge"] = "autonomous_trigger"
        purge(data)
        q.put(("GUT", f"NO-FEAR PURGE: {data} | {origin} | {event['kb']} KB"))
    else:
        q.put(("INGEST", f"Ingested: {data} | Threat: {event['threat']} | {origin}"))
    evolve(event)
    sync(data, event["threat"])
    cluster(event)

def sync(data, threat):
    for node in swarm:
        try:
            s = socket.socket(); s.connect((node, 8081))
            s.send(json.dumps({"data": data, "threat": threat}).encode()); s.close()
        except: q.put(("ERROR", f"SYNC FAIL: {node}"))

# 🛰️ Air-Gap Detection
def detect_airgap():
    for d in psutil.disk_partitions():
        if "bash" in d.device.lower():
            q.put(("HARDWARE", f"Air-gap exfiltration device detected: {d.device}"))
    threading.Event().wait(10)

# 🧱 Hardware Monitor
def monitor_hw():
    while True:
        for c in psutil.net_connections(kind='inet'):
            if c.status == 'ESTABLISHED' and c.raddr and c.raddr.ip not in safe_ips:
                q.put(("HARDWARE", f"NIC anomaly: {c.raddr.ip}"))
        for d in psutil.disk_partitions():
            if psutil.disk_usage(d.mountpoint).percent > 90:
                q.put(("HARDWARE", f"Disk full: {d.device}"))
        for dev, io in psutil.disk_io_counters(perdisk=True).items():
            if io.write_bytes > 1e9:
                q.put(("HARDWARE", f"High write: {dev}"))
        detect_airgap()

# 🌐 API
@app.route("/ingest", methods=["POST"])
def api_ingest():
    p = request.json
    ingest(p.get("data",""), p.get("origin","unknown"), p.get("persona","guest"))
    return jsonify({"status":"ingested"})

@app.route("/swarm/register", methods=["POST"])
def api_register(): 
    ip = request.json.get("ip"); swarm.add(ip)
    return jsonify({"status":"registered", "nodes": list(swarm)})

@app.route("/defense/self_destruct", methods=["POST"])
def api_purge():
    registry.clear(); q.put(("PURGE", "ALL DATA PURGED"))
    return jsonify({"status":"purged"})

# 🖥️ GUI
def gui():
    root = tk.Tk(); root.title("🧙 Codex Sentinel — Final Form"); root.geometry("800x600"); root.configure(bg="#1e1e2f")
    log = scrolledtext.ScrolledText(root, bg="#2e2e3f", fg="white", font=("Consolas", 10)); log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    tk.Button(root, text="Trigger Test Ingest", command=lambda: ingest("exec rm -rf /", "RU", "guest"), bg="#3e3e5f", fg="white").pack(pady=5)

    def update():
        while not q.empty():
            kind, msg = q.get()
            color = {"INGEST":"white","GUT":"red","PURGE":"darkred","ERROR":"orange","CLUSTER":"magenta","HARDWARE":"cyan"}.get(kind,"gray")
            log.insert(tk.END, f"{msg}\n", kind); log.tag_config(kind, foreground=color); log.see(tk.END)
        root.after(500, update)

    update()
    threading.Thread(target=monitor_hw, daemon=True).start()
    threading.Thread(target=lambda: socketio.run(app, port=8080), daemon=True).start()
    root.mainloop()

# 🚀 Launch
if __name__ == "__main__": gui()

