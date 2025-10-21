import os, sys, ctypes, subprocess, winreg, tkinter as tk, time, hashlib, threading, uuid, json
from pynput import keyboard, mouse
import cv2

# Elevation
def elevate():
    try: import win32com.shell.shell as shell
    except: subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"]); import win32com.shell.shell as shell
    if not ctypes.windll.shell32.IsUserAnAdmin():
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=' '.join([f'"{a}"' for a in sys.argv])); sys.exit()

# Symbolic tag
def tag(prefix, content):
    h = hashlib.sha256(content.encode()).hexdigest()[:6]
    t = time.strftime('%H%M%S')
    return f"{prefix}::{h}_{t}"

# Ghost Mode
class Ghost:
    def __init__(self):
        self.ids = {}
        for d in ["keyboard", "mouse", "webcam"]:
            self.ids[d] = f"{d}_{uuid.uuid4().hex[:8]}"

# Trace Registry
class Trace:
    def __init__(self):
        self.log = []
    def add(self, device, event):
        self.log.append({"device": device, "event": event, "symbol": tag(device, event)})
    def purge(self):
        self.log.clear()
    def dump(self):
        return json.dumps(self.log, indent=2)

# Threat Matrix
class Threat:
    def __init__(self):
        self.streams = {}
    def add(self, key, value):
        level = "HIGH" if "camera" in value else "MEDIUM" if "input" in value else "LOW"
        self.streams[key] = {"value": value, "symbol": tag("STREAM", key), "level": level}
    def purge(self):
        self.streams.clear()

# Registry helpers
def set_reg(root, path, name, value, reg_type=winreg.REG_DWORD):
    try:
        key = winreg.CreateKeyEx(root, path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, name, 0, reg_type, value)
        key.Close()
        return True
    except:
        return False

def del_key(root, path, subkey):
    try:
        winreg.DeleteKey(winreg.OpenKey(root, path, 0, winreg.KEY_ALL_ACCESS), subkey)
        return True
    except:
        return False

def purge_files(log):
    for d in [r"%LOCALAPPDATA%\Microsoft\Clipboard", r"%LOCALAPPDATA%\Packages"]:
        for r, _, fs in os.walk(os.path.expandvars(d)):
            for f in fs:
                if "ScreenClip" in f or "Clipboard" in r:
                    try: os.remove(os.path.join(r, f)); log(f"🧹 {f}")
                    except: log(f"⚠️ {f}")

# GUI launcher
def launch():
    ghost = Ghost()
    trace = Trace()
    threat = Threat()

    root = tk.Tk()
    root.title("Codex Phantom Shell")
    root.geometry("720x880")
    status = tk.StringVar(value="🧠 Monitoring...")
    logbox = tk.Text(root, height=10, width=85, font=("Consolas", 9))
    logbox.pack()

    def log(msg):
        status.set(msg)
        logbox.insert(tk.END, msg + "\n")
        logbox.see(tk.END)
        root.update_idletasks()

    tk.Label(root, text="🧿 Idle", font=("Consolas", 12), fg="darkgreen").pack()
    tk.Label(root, text="", font=("Consolas", 10), fg="darkred").pack()
    toggles = tk.Frame(root)
    toggles.pack()

    def toggle(setting):
        frame = tk.Frame(toggles)
        frame.pack()
        tk.Label(frame, text=f"{setting['label']} {tag('REG', setting['name'])}", font=("Consolas", 10), width=50, anchor="w").pack(side=tk.LEFT)
        for label, val in [("Allow", setting["on"]), ("Deny", setting["off"])]:
            tk.Button(frame, text=label, width=10,
                      command=lambda v=val: log(f"{setting['label']} {label}: {'✅' if set_reg(winreg.HKEY_CURRENT_USER, setting['path'], setting['name'], v, setting.get('type', winreg.REG_DWORD)) else '⚠️'}")
                      ).pack(side=tk.LEFT)

    settings = [
        {"label": "Clipboard History", "path": r"Software\\Microsoft\\Clipboard", "name": "EnableClipboardHistory", "on": 1, "off": 0},
        {"label": "Screen Clipping", "path": r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock", "name": "BlockScreenClip", "on": 0, "off": 1},
        {"label": "Tailored Experiences", "path": r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy", "name": "TailoredExperiencesWithDiagnosticDataEnabled", "on": 1, "off": 0},
        {"label": "Advertising ID", "path": r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo", "name": "Enabled", "on": 1, "off": 0},
        {"label": "Location Services", "path": r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location", "name": "Value", "on": "Allow", "off": "Deny", "type": winreg.REG_SZ}
    ]
    for s in settings:
        toggle(s)

    def scan():
        purge_files(log)
        base = r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"
        try: key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, base); i = 0; found = []
        except: found = []
        else:
            while True:
                try: found.append(winreg.EnumKey(key, i)); i += 1
                except: break
            key.Close()
        log(f"🔍 Found {len(found)} telemetry keys.")
        for sub in found:
            log(f"{'🧹 Purged' if del_key(winreg.HKEY_CURRENT_USER, base, sub) else '⚠️'}: {sub}")

    def start_input_daemon():
        def on_key(key):
            try: char = key.char if hasattr(key, 'char') and key.char else str(key)
            except: char = str(key)
            trace.add("keyboard", char)
            threat.add("keyboard_input", char)
            log(f"⌨️ {tag('KEY', char)} [MEDIUM]")

        def on_click(x, y, button, pressed):
            if pressed:
                event = f"{button} @({x},{y})"
                trace.add("mouse", event)
                threat.add("mouse_input", event)
                log(f"🖱️ {tag('MOUSE', event)} [MEDIUM]")

        def webcam_daemon():
            cap = cv2.VideoCapture(0)
            while True:
                ret, _ = cap.read()
                if ret:
                    trace.add("webcam", "frame")
                    threat.add("camera_access", "Webcam active")
                    log(f"📷 {tag('WEBCAM', 'frame')} [HIGH]")
                time.sleep(10)

        threading.Thread(target=lambda: keyboard.Listener(on_press=on_key).start(), daemon=True).start()
        threading.Thread(target=lambda: mouse.Listener(on_click=on_click).start(), daemon=True).start()
        threading.Thread(target=webcam_daemon, daemon=True).start()

    start_input_daemon()

    tk.Button(root, text="Export Trace Log", command=lambda: log(trace.dump()), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Purge All Traces", command=lambda: trace.purge() or threat.purge() or log("🔥 Traces purged."), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Show Ghost IDs", command=lambda: [log(f"{d}: {ghost.ids[d]}") for d in ghost.ids], font=("Consolas", 10)).pack(pady=5)
    tk.Label(root, textvariable=status, wraplength=640, font=("Consolas", 10), fg="blue").pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    elevate()
    launch()

