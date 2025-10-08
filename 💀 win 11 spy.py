import subprocess
import sys

# 📦 Autoloader for required libraries
def autoload_libraries():
    required = ["psutil"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            print(f"📦 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

import tkinter as tk
from tkinter import ttk, messagebox
import winreg
import platform
import psutil
import json
import os

# 🔍 Registry Scanner
def scan_registry_keys(paths):
    registry_data = {}
    for root, subkey in paths:
        try:
            with winreg.OpenKey(root, subkey) as key:
                i = 0
                while True:
                    try:
                        name, value, _ = winreg.EnumValue(key, i)
                        registry_data[f"{subkey}\\{name}"] = value
                        i += 1
                    except OSError:
                        break
        except FileNotFoundError:
            continue
    return registry_data

# ⚙️ Settings URIs
def get_settings_uris():
    return [
        "ms-settings:privacy-location",
        "ms-settings:taskbar",
        "ms-settings:windowsupdate",
        "ms-settings:display",
        "ms-settings:notifications",
        "ms-settings:appsfeatures",
        "ms-settings:network-status",
        "ms-settings:signinoptions",
        "ms-settings:bluetooth",
        "ms-settings:storage"
    ]

# 🧬 Services
def list_services():
    result = subprocess.run(["sc", "query"], capture_output=True, text=True)
    return result.stdout

# ⏳ Scheduled Tasks
def list_scheduled_tasks():
    result = subprocess.run(["schtasks", "/query", "/fo", "LIST", "/v"], capture_output=True, text=True)
    return result.stdout

# 🧠 System Info
def get_system_info():
    return {
        "OS": platform.system(),
        "Version": platform.version(),
        "Architecture": platform.machine(),
        "CPU": platform.processor(),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Uptime (hrs)": round((psutil.boot_time() - psutil.boot_time()) / 3600, 2)
    }

# 💾 Save Snapshot
def save_snapshot(data):
    with open("control_snapshot.json", "w") as f:
        json.dump(data, f, indent=4)

# 🧿 GUI Overlay
def load_controls():
    try:
        with open("control_snapshot.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", "control_snapshot.json not found.")
        return {}

def create_dynamic_tab(frame, title, data):
    ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack(anchor="w")
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for i, (key, value) in enumerate(data.items()):
        ttk.Label(scroll_frame, text=key).grid(row=i, column=0, sticky="w")
        entry = ttk.Entry(scroll_frame, width=60)
        entry.insert(0, str(value))
        entry.grid(row=i, column=1)

def create_text_tab(frame, title, content):
    ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack(anchor="w")
    text = tk.Text(frame, wrap="word", height=20)
    text.insert("1.0", content)
    text.pack(fill="both", expand=True)

def build_gui(data):
    root = tk.Tk()
    root.title("🧿 Windows 11 Symbolic Control Overlay")
    root.geometry("1000x700")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    for key, value in data.items():
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=f"🔹 {key.capitalize()}")
        if isinstance(value, dict):
            create_dynamic_tab(tab, key, value)
        else:
            create_text_tab(tab, key, value)

    root.mainloop()

# 🧪 Main Ritual
def main():
    registry_paths = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Policies"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies")
    ]

    data = {
        "registry": scan_registry_keys(registry_paths),
        "settings_uris": {uri: "Launchable" for uri in get_settings_uris()},
        "services": {"Raw Output": list_services()},
        "scheduled_tasks": {"Raw Output": list_scheduled_tasks()},
        "system_info": get_system_info()
    }

    save_snapshot(data)
    build_gui(data)

if __name__ == "__main__":
    main()

