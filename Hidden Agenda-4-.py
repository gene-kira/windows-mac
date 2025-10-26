import subprocess, sys, importlib, os, socket, cv2, tkinter as tk, threading, time
from tkinter import ttk
from datetime import datetime
from pytube import YouTube

# 🔄 Autoloader
required_packages = {
    "opencv-python": "cv2",
    "pytube": "pytube",
    "pydub": "pydub",
    "requests": "requests",
    "numpy": "numpy"
}

def codex_autoload(package_map):
    for pip_name, import_name in package_map.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

codex_autoload(required_packages)

# 🧬 Symbolic Registry
symbolic_registry = []
event_log = []
blocked_countries = []

def mutate_registry(symbol, affirmation, origin, resurrected):
    symbolic_registry.append({
        "symbol": symbol,
        "affirmation": affirmation,
        "origin": origin,
        "resurrected": resurrected
    })

def log_event(text):
    timestamp = datetime.now().strftime("%H:%M")
    event_log.append(f"{timestamp} {text}")

def block_ip(origin):
    blocked_countries.append(origin)

# 🧨 Lethal Execution
def lethal_execute(threat):
    if detect_resurrection(threat["affirmation"], threat["origin"]):
        collapse_mutation(threat)
    mutate_registry(threat["symbol"], threat["affirmation"], threat["origin"], True)
    block_ip(threat["origin"])
    log_event(f"💀 OBLIVION PURGE: {threat['affirmation']} from {threat['origin']}")

def detect_resurrection(affirmation, origin):
    for entry in symbolic_registry:
        if entry["affirmation"] == affirmation and entry["origin"] != origin:
            log_event(f"⚠️ Resurrection trap triggered: {affirmation}")
            return True
    return False

def collapse_mutation(threat):
    lineage = [e for e in symbolic_registry if e["affirmation"] == threat["affirmation"]]
    for ancestor in lineage:
        mutate_registry("sigil", ancestor["affirmation"], threat["origin"], True)
    log_event(f"🧬 COLLAPSE: {threat['affirmation']} lineage purged")

# 🔌 Ingest Channels
def ingest_youtube(url):
    yt = YouTube(url)
    meta = {
        "symbol": "eye",
        "affirmation": yt.title,
        "origin": "YouTube",
        "resurrected": False
    }
    log_event(f"📡 Ingested YouTube: {yt.title}")
    lethal_execute(meta)

def scan_usb_mounts():
    usb_paths = ["E:/", "F:/", "/mnt/usb1", "/mnt/usb2"]
    for path in usb_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith((".mp4", ".wav")):
                    meta = {
                        "symbol": "usb",
                        "affirmation": file,
                        "origin": "USB",
                        "resurrected": False
                    }
                    log_event(f"📁 USB Ingest: {file} from {path}")
                    lethal_execute(meta)

def ingest_network_streams():
    ports = [8080, 9090]
    for port in ports:
        meta = {
            "symbol": "net",
            "affirmation": f"Stream@{port}",
            "origin": f"Port{port}",
            "resurrected": False
        }
        log_event(f"🌐 Port Ingest: {port}")
        lethal_execute(meta)

# 🛡️ Webcam Defense
def monitor_webcam():
    cap = cv2.VideoCapture(0)
    status = cap.isOpened()
    cap.release()
    log_event("🛡️ Webcam Accessed" if status else "🔒 Webcam Blocked")
    return status

# 🔄 Autonomous Ingest Loop
def autonomous_ingest_loop():
    while True:
        try:
            scan_usb_mounts()
            ingest_network_streams()
            # Future: ingest_youtube_feed_batch()
        except Exception as e:
            log_event(f"⚠️ Ingest error: {str(e)}")
        time.sleep(10)

# 🧠 GUI Shell
class CodexShellGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Codex Devourer Shell: Oblivion Protocol")
        self.configure(bg="#0f111a")
        self.geometry("1200x700")
        self.webcam_status = monitor_webcam()
        self.create_panels()

    def create_panels(self):
        panels = {
            "Ingest Matrix": self.panel_ingest,
            "Threat Matrix": self.panel_threat,
            "Country Filter": self.panel_country,
            "Event Bus": self.panel_event,
            "Persona Status": self.panel_persona,
            "Symbol Registry": self.panel_registry
        }
        for i, (title, method) in enumerate(panels.items()):
            frame = tk.Frame(self, bg="#1a1c2c", bd=2, relief="groove")
            frame.place(x=20 + (i % 3) * 390, y=20 + (i // 3) * 230, width=370, height=210)
            label = tk.Label(frame, text=title, bg="#1a1c2c", fg="white", font=("Consolas", 14))
            label.pack(anchor="nw", padx=10, pady=5)
            method(frame)

    def panel_ingest(self, frame):
        url_entry = tk.Entry(frame, width=40)
        url_entry.pack(pady=5)
        tk.Button(frame, text="Manual YouTube Ingest", command=lambda: ingest_youtube(url_entry.get())).pack(pady=2)
        tk.Label(frame, text="Autonomous ingest running...", bg="#1a1c2c", fg="lightgreen").pack()

    def panel_threat(self, frame):
        tk.Label(frame, text="Threats auto-purged on ingest.", bg="#1a1c2c", fg="red").pack()

    def panel_country(self, frame):
        display = tk.Label(frame, text="Blocked: None", bg="#1a1c2c", fg="white")
        display.pack()
        def update():
            if blocked_countries:
                display.config(text=f"Blocked: {', '.join(blocked_countries)}")
        tk.Button(frame, text="Refresh", command=update).pack()

    def panel_event(self, frame):
        log_text = tk.Text(frame, bg="#1a1c2c", fg="cyan", height=8, width=40)
        log_text.pack()
        def refresh():
            log_text.delete(1.0, tk.END)
            for entry in event_log[-10:]:
                log_text.insert(tk.END, entry + "\n")
        tk.Button(frame, text="Update Log", command=refresh).pack()

    def panel_persona(self, frame):
        status = "🟢 Webcam Active" if self.webcam_status else "🔴 Webcam Blocked"
        tk.Label(frame, text=f"Exposure: 42%\n{status}", bg="#1a1c2c", fg="lightblue").pack()

    def panel_registry(self, frame):
        text = tk.Text(frame, bg="#1a1c2c", fg="white", height=8, width=40)
        text.pack()
        def refresh():
            text.delete(1.0, tk.END)
            for entry in symbolic_registry[-5:]:
                line = f"{entry['symbol']} {entry['affirmation']} ({entry['origin']})\n"
                text.insert(tk.END, line)
        tk.Button(frame, text="Update Registry", command=refresh).pack()

# 🚀 Launch Shell
if __name__ == "__main__":
    threading.Thread(target=autonomous_ingest_loop, daemon=True).start()
    app = CodexShellGUI()
    app.mainloop()

