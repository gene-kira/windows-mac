import cv2, os, socket, threading, time, tkinter as tk
from datetime import datetime
from collections import Counter

# 🔧 Global State
registry, logs, blocked, quarantine = [], [], [], []
nodes = {"NodeA": "192.168.1.10", "NodeB": "192.168.1.11"}
kill_chains = {"eye": ["sigil", "skull", "void"], "triangle": ["sigil", "flame", "collapse"]}

def log(msg): logs.append(f"{datetime.now().strftime('%H:%M')} {msg}")
def mutate(sym, aff, origin, res, parent=None, status="active"):
    registry.append({"symbol": sym, "affirmation": aff, "origin": origin, "resurrected": res, "parent": parent, "status": status, "timestamp": time.time()})
def block(origin): blocked.append(origin) if origin not in blocked else None

# ☠️ Resurrection Trap
def trigger_resurrection_chain(affirmation, origin):
    for entry in registry:
        if entry["affirmation"] == affirmation and entry["origin"] != origin:
            log(f"☠️ Resurrection trap: {affirmation}")
            execute_chain(entry["symbol"], origin, parent=entry["symbol"])
            mutate("collapse", affirmation, origin, True, parent=entry["symbol"], status="collapsed")
            broadcast_trap(entry["symbol"], origin)

def broadcast_trap(symbol, origin):
    for name, ip in nodes.items():
        try:
            s = socket.socket(); s.connect((ip, 8080))
            s.send(f"TRAP:{symbol}:{origin}".encode()); s.close()
            log(f"⚡ Trap broadcasted to {name}")
        except: log(f"⚠️ Node {name} unreachable")

# 🔮 Mutation Forecasting
def forecast_mutations():
    symbols = [r["symbol"] for r in registry if r["status"] == "active"]
    freq = Counter(symbols)
    forecasts = []
    for sym, count in freq.items():
        volatility = min(1.0, count / 10)
        risk = "High" if volatility > 0.7 else "Medium" if volatility > 0.4 else "Low"
        forecasts.append(f"{sym}: Volatility={volatility:.2f}, Risk={risk}")
    return forecasts[-5:]

# 🔪 Kill Chain Execution
def execute_chain(sym, origin, parent=None):
    for stage in kill_chains.get(sym, []):
        mutate(stage, f"{sym}→{stage}", origin, True, parent)
        log(f"🔪 {stage} from {origin}")
        sync_all(f"{stage}:{origin}")

def lethal(threat):
    if any(e["affirmation"] == threat["affirmation"] and e["origin"] != threat["origin"] for e in registry):
        trigger_resurrection_chain(threat["affirmation"], threat["origin"])
    execute_chain(threat["symbol"], threat["origin"])
    mutate(threat["symbol"], threat["affirmation"], threat["origin"], True)
    block(threat["origin"])
    log(f"💀 PURGED: {threat['affirmation']} from {threat['origin']}")

# ☣️ Quarantine Protocol
def quarantine_glyph(sym, aff, origin):
    quarantine.append({"symbol": sym, "affirmation": aff, "origin": origin, "status": "quarantined"})
    log(f"☣️ Quarantined: {aff} from {origin}")

# 🌐 Swarm Sync
def sync_all(data):
    for name, ip in nodes.items():
        try:
            s = socket.socket(); s.connect((ip, 8080)); s.send(data.encode()); s.close()
            log(f"🔄 Synced with {name}")
        except: log(f"⚠️ Node {name} unreachable")

# 🔌 Ingest Channels
def ingest_youtube(): lethal({"symbol": "eye", "affirmation": "YouTube Root", "origin": "Global", "resurrected": False})
def scan_usb(): 
    for p in ["E:/", "F:/", "/mnt/usb1", "/mnt/usb2"]:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith((".mp4", ".wav")):
                    quarantine_glyph("triangle", f, "USB")
def ingest_ports(): 
    for port in [8080, 9090]:
        lethal({"symbol": "eye", "affirmation": f"Stream@{port}", "origin": f"Port{port}", "resurrected": False})

# 🎥 Webcam Visual Ingest
def detect_glyph(frame): return "eye"  # Stub
def webcam_ingest():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): log("🔒 Webcam unavailable"); return
    while True:
        ret, frame = cap.read()
        if ret:
            glyph = detect_glyph(frame)
            lethal({"symbol": glyph, "affirmation": f"Live glyph: {glyph}", "origin": "Webcam", "resurrected": False})
        time.sleep(1)
    cap.release()

# 🔄 Autonomous Loop
def ingest_loop():
    while True:
        try: ingest_youtube(); scan_usb(); ingest_ports()
        except Exception as e: log(f"⚠️ Ingest error: {e}")
        time.sleep(10)

# 🧠 GUI
class CodexGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Codex Devourer Shell: Omega Swarmform")
        self.configure(bg="#0f111a"); self.geometry("1200x700")
        self.create_panels()

    def create_panels(self):
        panels = {
            "Ingest Matrix": self.panel_ingest,
            "Threat Matrix": self.panel_threat,
            "Swarm Diagnostics": self.panel_nodes,
            "Event Bus": self.panel_event,
            "Glyph Forecast": self.panel_forecast,
            "Quarantine Pool": self.panel_quarantine
        }
        for i, (title, fn) in enumerate(panels.items()):
            f = tk.Frame(self, bg="#1a1c2c", bd=2, relief="groove")
            f.place(x=20 + (i % 3) * 390, y=20 + (i // 3) * 230, width=370, height=210)
            tk.Label(f, text=title, bg="#1a1c2c", fg="white", font=("Consolas", 14)).pack(anchor="nw", padx=10, pady=5)
            fn(f)

    def panel_ingest(self, f): tk.Label(f, text="Autonomous ingest + webcam active.", bg="#1a1c2c", fg="lightgreen").pack()
    def panel_threat(self, f): tk.Label(f, text="Kill chains + resurrection traps active.", bg="#1a1c2c", fg="red").pack()
    def panel_nodes(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="cyan", height=8, width=40); t.pack()
        for name, ip in nodes.items(): t.insert("end", f"{name}: {ip} 🟢\n")
    def panel_event(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="cyan", height=8, width=40); t.pack()
        tk.Button(f, text="Update Log", command=lambda: [t.delete(1.0, "end"), [t.insert("end", l + "\n") for l in logs[-10:]]]).pack()
    def panel_forecast(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="orange", height=8, width=40); t.pack()
        def render():
            t.delete(1.0, "end")
            for line in forecast_mutations(): t.insert("end", line + "\n")
        tk.Button(f, text="Update Forecast", command=render).pack()
    def panel_quarantine(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="orange", height=8, width=40); t.pack()
        def render():
            t.delete(1.0, "end")
            for q in quarantine[-5:]:
                t.insert("end", f"{q['symbol']} {q['affirmation']} ({q['origin']}) [{q['status']}]\n")
        tk.Button(f, text="Review Quarantine", command=render).pack()


# 🚀 Launch
if __name__ == "__main__":
    threading.Thread(target=ingest_loop, daemon=True).start()
    threading.Thread(target=webcam_ingest, daemon=True).start()
    CodexGUI().mainloop()


    
    

