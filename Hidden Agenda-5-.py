import cv2, os, socket, threading, time, tkinter as tk
from pytube import YouTube
from datetime import datetime

# 🔧 Global State
registry, logs, blocked = [], [], []
kill_chains = {"eye": ["sigil", "skull", "void"], "triangle": ["sigil", "flame", "collapse"]}

def log(msg): logs.append(f"{datetime.now().strftime('%H:%M')} {msg}")
def mutate(sym, aff, origin, res): registry.append({"symbol": sym, "affirmation": aff, "origin": origin, "resurrected": res})
def block(origin): blocked.append(origin) if origin not in blocked else None

# 🔪 Kill Chain Execution
def execute_chain(sym, origin):
    for stage in kill_chains.get(sym, []):
        mutate(stage, f"{sym}→{stage}", origin, True)
        log(f"🔪 {stage} from {origin}")
        sync("127.0.0.1", f"{stage}:{origin}")

def lethal(threat):
    if any(e["affirmation"] == threat["affirmation"] and e["origin"] != threat["origin"] for e in registry):
        mutate("collapse", threat["affirmation"], threat["origin"], True)
        log(f"🧬 COLLAPSE: {threat['affirmation']}")
    execute_chain(threat["symbol"], threat["origin"])
    mutate(threat["symbol"], threat["affirmation"], threat["origin"], True)
    block(threat["origin"])
    log(f"💀 PURGED: {threat['affirmation']} from {threat['origin']}")

def sync(ip, data):
    try:
        s = socket.socket(); s.connect((ip, 8080)); s.send(data.encode()); s.recv(1024); s.close()
        log(f"🔄 Synced with {ip}: {data}")
    except: log(f"⚠️ Sync failed with {ip}")

# 🔌 Ingest Channels
def ingest_youtube(): lethal({"symbol": "eye", "affirmation": "YouTube Root", "origin": "Global", "resurrected": False})
def scan_usb(): 
    for p in ["E:/", "F:/", "/mnt/usb1", "/mnt/usb2"]:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith((".mp4", ".wav")):
                    lethal({"symbol": "triangle", "affirmation": f, "origin": "USB", "resurrected": False})
def ingest_ports(): 
    for port in [8080, 9090]:
        lethal({"symbol": "eye", "affirmation": f"Stream@{port}", "origin": f"Port{port}", "resurrected": False})

# 🎥 Webcam Visual Ingest
def detect_glyph(frame): return "eye"  # Stub for glyph detection
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
        self.title("Codex Devourer Shell"); self.configure(bg="#0f111a"); self.geometry("1200x700")
        self.create_panels()

    def create_panels(self):
        panels = {
            "Ingest Matrix": self.panel_ingest,
            "Threat Matrix": self.panel_threat,
            "Country Filter": self.panel_country,
            "Event Bus": self.panel_event,
            "Symbol Registry": self.panel_registry
        }
        for i, (title, fn) in enumerate(panels.items()):
            f = tk.Frame(self, bg="#1a1c2c", bd=2, relief="groove")
            f.place(x=20 + (i % 3) * 390, y=20 + (i // 3) * 230, width=370, height=210)
            tk.Label(f, text=title, bg="#1a1c2c", fg="white", font=("Consolas", 14)).pack(anchor="nw", padx=10, pady=5)
            fn(f)

    def panel_ingest(self, f): tk.Label(f, text="Autonomous ingest + webcam active.", bg="#1a1c2c", fg="lightgreen").pack()
    def panel_threat(self, f): tk.Label(f, text="Kill chains auto-executed.", bg="#1a1c2c", fg="red").pack()
    def panel_country(self, f):
        lbl = tk.Label(f, text="Blocked: None", bg="#1a1c2c", fg="white"); lbl.pack()
        tk.Button(f, text="Refresh", command=lambda: lbl.config(text=f"Blocked: {', '.join(blocked)}")).pack()
    def panel_event(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="cyan", height=8, width=40); t.pack()
        tk.Button(f, text="Update Log", command=lambda: [t.delete(1.0, tk.END), [t.insert(t.END, l + "\n") for l in logs[-10:]]]).pack()
    def panel_registry(self, f):
        t = tk.Text(f, bg="#1a1c2c", fg="white", height=8, width=40); t.pack()
        tk.Button(f, text="Update Registry", command=lambda: [t.delete(1.0, tk.END), [t.insert(t.END, f"{r['symbol']} {r['affirmation']} ({r['origin']})\n") for r in registry[-5:]]]).pack()

# 🚀 Launch
if __name__ == "__main__":
    threading.Thread(target=ingest_loop, daemon=True).start()
    threading.Thread(target=webcam_ingest, daemon=True).start()
    CodexGUI().mainloop()

