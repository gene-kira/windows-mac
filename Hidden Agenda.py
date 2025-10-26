# 🔄 Autoloader + Symbolic Module Checker
import subprocess
import sys
import importlib
import tkinter as tk
from tkinter import ttk

# 🧿 Package map: pip name → import name
required_packages = {
    "opencv-python": "cv2",
    "pytube": "pytube",
    "pydub": "pydub",
    "requests": "requests",
    "numpy": "numpy"
}

# 🔥 Codex Autoloader
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

def mutate_registry(symbol, affirmation, origin, resurrected):
    symbolic_registry.append({
        "symbol": symbol,
        "affirmation": affirmation,
        "origin": origin,
        "resurrected": resurrected
    })

# 🔥 Ingest Daemon
def ingest_video(url):
    from pytube import YouTube
    yt = YouTube(url)
    metadata = {
        "title": yt.title,
        "author": yt.author,
        "origin": "US",  # Placeholder
        "keywords": yt.keywords
    }
    return metadata

# 🛡️ Devourer Mode
def purge_logic(metadata):
    if any(k in metadata["keywords"] for k in ["mind-control", "manipulation", "subliminal"]):
        mutate_registry("skull", "Manipulative intent", metadata["origin"], False)
        return "Purged"
    return "Benign"

# 🧠 Codex GUI Shell
class CodexShellGUI(tk.Tk):
    def __init__(self, module_status):
        super().__init__()
        self.title("Codex Sentinel Shell")
        self.configure(bg="#0f111a")
        self.geometry("1200x700")
        self.module_status = module_status
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
        tk.Button(frame, text="Ingest", command=ingest).pack(pady=5)

    def panel_threat(self, frame):
        tk.Label(frame, text="Threats will appear after ingest.", bg="#1a1c2c", fg="orange").pack()

    def panel_country(self, frame):
        tk.Label(frame, text="Allowed: US\nBlocked: UK, IN", bg="#1a1c2c", fg="white").pack()

    def panel_event(self, frame):
        tk.Label(frame, text="Event Log:\n11:52 Ingest US\n11:50 Purge", bg="#1a1c2c", fg="cyan").pack()

    def panel_persona(self, frame):
        tk.Label(frame, text="Exposure: 42%", bg="#1a1c2c", fg="lightblue").pack()

    def panel_registry(self, frame):
        tk.Label(frame, text="Symbols:\n👁️ You are powerful\n🔺 Buy chips", bg="#1a1c2c", fg="white").pack()

    def panel_modules(self, frame):
        for mod, status in self.module_status.items():
            color = "lightgreen" if "✅" in status else "red"
            tk.Label(frame, text=f"{mod}: {status}", bg="#1a1c2c", fg=color).pack(anchor="w", padx=10)

# 🚀 Launch Shell
if __name__ == "__main__":
    module_status = codex_autoload(required_packages)
    app = CodexShellGUI(module_status)
    app.mainloop()

