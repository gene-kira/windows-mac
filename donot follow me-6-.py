import tkinter as tk
from tkinter import messagebox, scrolledtext
import winreg, json, threading, time, os, ctypes, sys, subprocess

# === AUTO-ELEVATION ===
def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# === CONFIG ===
status = {}
status_labels = {}
settings_file = "codex_settings.json"

TRACKING_VECTORS = {
    "Activity History": [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\System", "EnableActivityFeed", 0),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\System", "PublishUserActivities", 0),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\System", "UploadUserActivities", 0)
    ],
    "Advertising ID": [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo", "Enabled", 0)
    ],
    "Tailored Experiences": [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Privacy", "TailoredExperiencesWithDiagnosticDataEnabled", 0)
    ],
    "Telemetry": [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection", "AllowTelemetry", 0),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection", "DisableTelemetryOptInSettingsUx", 1),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\CloudContent", "DisableWindowsConsumerFeatures", 1)
    ],
    "Feedback Hub": [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection", "DoNotShowFeedbackNotifications", 1)
    ]
}

TELEMETRY_TASKS = [
    r"\Microsoft\Windows\Application Experience\ProgramDataUpdater",
    r"\Microsoft\Windows\Customer Experience Improvement Program\Consolidator",
    r"\Microsoft\Windows\Autochk\Proxy"
]

TELEMETRY_SERVICES = [
    "DiagTrack",
    "dmwappushservice"
]

# === CORE FUNCTIONS ===
def log_event(message):
    with open("codex_purge_log.json", "a") as log:
        log.write(json.dumps({"time": time.ctime(), "event": message}) + "\n")
    log_view.insert(tk.END, f"{time.ctime()} — {message}\n")
    log_view.see(tk.END)

def purge_registry(root, path, name, value):
    try:
        key = winreg.CreateKeyEx(root, path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
        winreg.CloseKey(key)
        log_event(f"✅ Purged {name} in {path}")
    except Exception as e:
        log_event(f"❌ Error purging {name} in {path}: {e}")

def purge_vector(vector_name):
    for root, path, name, value in TRACKING_VECTORS[vector_name]:
        purge_registry(root, path, name, value)
    status[vector_name] = "Purged"
    log_event(f"🧬 {vector_name} purged with symbolic overlay")

def purge_all():
    for vector in TRACKING_VECTORS:
        purge_vector(vector)
    suppress_tasks_and_services()
    log_event("🧬 Full purge executed")

def suppress_tasks_and_services():
    for task in TELEMETRY_TASKS:
        try:
            subprocess.run(["schtasks", "/Change", "/TN", task, "/Disable"], check=True)
            log_event(f"⏱️ Disabled task: {task}")
        except Exception as e:
            log_event(f"❌ Failed to disable task {task}: {e}")
    for service in TELEMETRY_SERVICES:
        try:
            subprocess.run(["sc", "config", service, "start=", "disabled"], check=True)
            subprocess.run(["sc", "stop", service], check=True)
            log_event(f"⏱️ Disabled service: {service}")
        except Exception as e:
            log_event(f"❌ Failed to disable service {service}: {e}")

def check_clearance():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def swarm_sync():
    log_event("🕸️ Swarm sync triggered — node propagation visualized")

def update_status():
    for vector, label in status_labels.items():
        label.config(text=f"{vector}: {status.get(vector, 'Active')}")

def boot_purge():
    def run_boot():
        if autonomous_mode.get() and check_clearance():
            purge_all()
            swarm_sync()
            root.after(0, update_status)
        elif autonomous_mode.get():
            log_event("Autonomous mode blocked: insufficient privileges")
    threading.Thread(target=run_boot, daemon=True).start()

def monitor_resurrection():
    def run_monitor():
        while True:
            if autonomous_mode.get() and check_clearance():
                for vector_name, entries in TRACKING_VECTORS.items():
                    for root, path, name, expected_value in entries:
                        try:
                            key = winreg.OpenKey(root, path, 0, winreg.KEY_READ)
                            current_value, _ = winreg.QueryValueEx(key, name)
                            winreg.CloseKey(key)
                            if current_value != expected_value:
                                purge_registry(root, path, name, expected_value)
                                log_event(f"🧠 Resurrection detected: {vector_name} re-enabled. Threat matrix updated.")
                                status[vector_name] = "Purged"
                        except Exception as e:
                            log_event(f"Resurrection check failed for {name} in {path}: {e}")
                root.after(0, update_status)
            time.sleep(30)
    threading.Thread(target=run_monitor, daemon=True).start()

def confirm_and_purge(vector_name):
    if not check_clearance():
        messagebox.showerror("Access Denied", "You must run this as administrator.")
        return

    def safe_purge():
        purge_vector(vector_name)
        root.after(0, update_status)

    if messagebox.askyesno("Confirm Purge", f"⚠️ Confirm purge of {vector_name}?"):
        threading.Thread(target=safe_purge, daemon=True).start()

def manual_override():
    if check_clearance():
        def safe_override():
            purge_all()
            swarm_sync()
            root.after(0, update_status)
        threading.Thread(target=safe_override, daemon=True).start()
        messagebox.showinfo("Override", "Manual purge executed.")
    else:
        messagebox.showerror("Access Denied", "You must run this as administrator.")

def toggle_autonomous():
    mode = "enabled" if autonomous_mode.get() else "disabled"
    log_event(f"Autonomous mode {mode}")
    save_settings()
    messagebox.showinfo("Autonomous Mode", f"Autonomous mode {mode.capitalize()}.")

def save_settings():
    with open(settings_file, "w") as f:
        json.dump({"autonomous_mode": autonomous_mode.get()}, f)

def load_settings():
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                data = json.load(f)
                autonomous_mode.set(data.get("autonomous_mode", False))
        except:
            log_event("⚠️ Failed to load settings")

# === GUI SETUP ===
root = tk.Tk()
root.title("Codex Purge Panel")
root.geometry("520x750")

autonomous_mode = tk.BooleanVar(value=False)
load_settings()

tk.Label(root, text="🧿 Codex Purge Panel", font=("Arial", 16)).pack(pady=10)

for vector in TRACKING_VECTORS:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=vector, width=30, anchor="w").pack(side="left")
    tk.Button(frame, text="Purge", command=lambda v=vector: confirm_and_purge(v)).pack(side="right")
    status_labels[vector] = tk.Label(root, text=f"{vector}: Active", fg="red")
    status_labels[vector].pack()

tk.Button(root, text="Manual Override", command=manual_override).pack(pady=10)
tk.Checkbutton(root, text="Enable Fully Autonomous Mode", variable=autonomous_mode, command=toggle_autonomous).pack(pady=10)

# === Diagnostic Level Panel ===
def get_telemetry_level():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection", 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, "AllowTelemetry")
        winreg.CloseKey(key)
        return value
    except:
        return None

def update_diagnostic_panel():
    level = get_telemetry_level()
    if level == 0:
        diag_status.config(text="Diagnostic Level: 0 (None)", fg="green")
    elif level == 1:
        diag_status.config(text="Diagnostic Level: 1 (Basic)", fg="orange")
    elif level == 2:
        diag_status.config(text="Diagnostic Level: 2 (Enhanced)", fg="red")
    elif level == 3:
        diag_status.config(text="Diagnostic Level: 3 (Full)", fg="red")
    else:
        diag_status.config(text="Diagnostic Level: Unknown", fg="gray")

tk.Label(root, text="📊 Diagnostic Level Panel", font=("Arial", 12)).pack(pady=5)
diag_status = tk.Label(root, text="Diagnostic Level: Unknown", fg="gray")
diag_status.pack()
tk.Button(root, text="Force Lockdown", command=lambda: [purge_vector("Telemetry"), update_diagnostic_panel()]).pack(pady=5)

tk.Label(root, text="🔍 Live Log Viewer", font=("Arial", 12)).pack(pady=5)
log_view = scrolledtext.ScrolledText(root, width=60, height=15, state="normal")
log_view.pack(pady=5)

# === DAEMON THREADS ===
log_event("🌀 Codex Purge Shell started")
boot_purge()
monitor_resurrection()
update_diagnostic_panel()

try:
    root.mainloop()
except Exception as e:
    log_event(f"❌ GUI crash: {str(e)}")
    print(f"[Codex Sentinel] GUI crash: {e}")




