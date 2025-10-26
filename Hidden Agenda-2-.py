# 🔄 Autoloader + Symbolic Module Checker
import subprocess, sys, importlib, os, socket, cv2, tkinter as tk
from tkinter import ttk
from datetime import datetime

# 🧿 Package map: pip name → import name
required_packages = {
    "opencv-python": "cv2",
    "pytube": "pytube",
    "pydub": "pydub",
    "requests": "requests",
    "numpy": "numpy"
}

def codex_autoload(package_map):
    status = {}
    for pip_name, import_name in package_map.items():
        try:
            importlib.import_module(import_name)
            status[import_name] = "✅ Loaded"
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                importlib.import_module(import_name)
                status[import_name] = "✅ Installed"
            except Exception:
                status[import_name] = "❌ Failed"
    return status

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

# 🔥 Ingest Daemon
def ingest_video(url):
    from pytube import YouTube
    yt = YouTube(url)
    metadata = {
        "title": yt.title,
        "author": yt.author,
        "origin": "US",
        "keywords": yt.keywords
    }
    log_event(f"Ingested YouTube: {yt.title}")
    return metadata

def purge_logic(metadata):
    if any(k in metadata["keywords"] for k in ["mind-control", "manipulation", "subliminal"]):
        mutate_registry("skull", "Manipulative intent", metadata["origin"], False)
        log_event(f"Purged: {metadata['title']}")
        return "Purged"
    return "Benign"

# 🔌 USB Ingest
def scan_usb_mounts():
    usb_paths = ["E:/", "F:/", "/mnt/usb1", "/mnt/usb2"]
    ingested = []
    for path in usb_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith((".mp4", ".wav")):
                    ingested.append({
                        "source": path,
                        "file": file,
                        "origin": "USB",
                        "keywords": ["usb-ingest"]
                    })
                    log_event(f"USB Ingest: {file} from {path}")
    return ingested

# 🛡️ Webcam Defense
def monitor_webcam():
    cap = cv2.VideoCapture(0)
    status = cap.isOpened()
    cap.release()
    log_event("Webcam Accessed" if status else "Webcam Blocked")
    return status

# 🌐 Swarm Sync
def start_node_listener(port):
    s = socket.socket()
    s.bind(('0.0.0.0', port))
    s.listen(1)
    log_event(f"Node listening on port {port}")
    while True:
        conn, addr = s.accept()
        data = conn.recv(1024)
        if data:
            log_event(f"Sync from {addr}: {data.decode()}")
            conn.send(b"ACK")
        conn.close()

# 🧠 GUI Shell
class CodexShellGUI(tk.Tk):
    def __init__(self, module_status):
        super().__init__()
        self.title("Codex Sentinel Shell")
        self.configure(bg="#0f111a")
        self.geometry("1200x700")
        self.module_status = module_status
        self.webcam_status = monitor_webcam()
        self.create_panels()

    def create_panels(self):
        panels = {
            "Ingest Matrix": self.panel_ingest,
            "Threat Matrix": self.panel_threat,
            "Country Filter": self.panel_country,
            "Event Bus": self.panel_event,
            "Persona Status": self.panel_persona,
            "Symbol Registry": self.panel_registry,
            "Module Checker": self.panel_modules
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
        result = tk.Label(frame, text="", bg="#1a1c2c", fg="lightgreen")
        result.pack()
        def ingest():
            url = url_entry.get()
            meta = ingest_video(url)
            status = purge_logic(meta)
            result.config(text=f"{meta['title']} → {status}")
        tk.Button(frame, text="Ingest YouTube", command=ingest).pack(pady=5)
        tk.Button(frame, text="Scan USB", command=lambda: scan_usb_mounts()).pack(pady=5)

    def panel_threat(self, frame):
        tk.Label(frame, text="Threats will appear after ingest.", bg="#1a1c2c", fg="orange").pack()

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
        for entry in event_log[-10:]:
            log_text.insert(tk.END, entry + "\n")

    def panel_persona(self, frame):
        status = "🟢 Webcam Active" if self.webcam_status else "🔴 Webcam Blocked"
        tk.Label(frame, text=f"Exposure: 42%\n{status}", bg="#1a1c2c", fg="lightblue").pack()

    def panel_registry(self, frame):
        text = tk.Text(frame, bg="#1a1c2c", fg="white", height=8, width=40)
        text.pack()
        for entry in symbolic_registry[-5:]:
            line = f"{entry['symbol']} {entry['affirmation']} ({entry['origin']})\n"
            text.insert(tk.END, line)

    def panel_modules(self, frame):
        for mod, status in self.module_status.items():
            color = "lightgreen" if "✅" in status else "red"
            tk.Label(frame, text=f"{mod}: {status}", bg="#1a1c2c", fg=color).pack(anchor="w", padx=10)

# 🚀 Launch Shell
if __name__ == "__main__":
    module_status = codex_autoload(required_packages)
    app = CodexShellGUI(module_status)
    app.mainloop()

