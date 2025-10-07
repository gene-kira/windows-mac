# 🔐 Lockdown Mode GUI Overlay: Military-Grade Edition
# Author: killer666
# Purpose: Tactical control of real-time data isolation, capsule execution, and symbolic cloaking

import tkinter as tk
from tkinter import ttk, messagebox
import time
import uuid
import psutil
import socket

# 🔘 Lockdown State
LOCKDOWN_ACTIVE = True
ALLOW_DATA_OUT = False
ALLOW_PERSONAL_DATA_OUT = False

# 🧬 Chameleon Cloak + Self-Destruct
def chameleon_cloak(data):
    return f"⧈{uuid.uuid4().hex[:8]}⧈"

def self_destruct(data, delay=5):
    time.sleep(delay)
    del data

# 🔍 Real-Time Telemetry
def get_live_telemetry():
    return {
        "CPU": f"{psutil.cpu_percent(interval=1)}%",
        "Memory": f"{psutil.virtual_memory().percent}%",
        "Entropy": uuid.uuid4().hex[:8]
    }

def get_personal_identifiers():
    return {
        "IP": socket.gethostbyname(socket.gethostname()),
        "MAC": ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0,8*6,8)]),
        "Hostname": socket.gethostname()
    }

def block_personal_data():
    if not ALLOW_PERSONAL_DATA_OUT:
        return {key: "⛔" for key in get_personal_identifiers()}
    return get_personal_identifiers()

# 🧪 Capsule Execution
def execute_capsule():
    if LOCKDOWN_ACTIVE and not ALLOW_DATA_OUT:
        messagebox.showwarning("Capsule Blocked", "Lockdown active. Capsule execution denied.")
        return
    cloaked = chameleon_cloak("Sensitive payload")
    messagebox.showinfo("Capsule Executed", f"Cloaked: {cloaked}\nSelf-destruct in 5s")
    self_destruct(cloaked)

# 🧭 GUI Setup
root = tk.Tk()
root.title("Lockdown Control Console")
root.geometry("700x500")
root.configure(bg="#0f1115")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Consolas", 11), padding=6, foreground="#ffffff", background="#2c2f36")
style.map("TButton", background=[("active", "#3e4149")])

# 🧿 Status Panel
status_frame = tk.LabelFrame(root, text="System Status", bg="#0f1115", fg="#00ffcc", font=("Consolas", 12))
status_frame.pack(fill="both", expand=True, padx=20, pady=10)

status_text = tk.StringVar()
status_label = tk.Label(status_frame, textvariable=status_text, justify="left", bg="#0f1115", fg="#eeeeee", font=("Consolas", 10))
status_label.pack(padx=10, pady=10)

def update_status():
    telemetry = get_live_telemetry()
    identifiers = block_personal_data()
    status_text.set(f"""
🔒 Lockdown: {'ACTIVE' if LOCKDOWN_ACTIVE else 'INACTIVE'}
📤 Data Out: {'ENABLED' if ALLOW_DATA_OUT else 'BLOCKED'}
🧍 Personal Data: {'ALLOWED' if ALLOW_PERSONAL_DATA_OUT else 'BLOCKED'}

🧠 Telemetry:
  CPU: {telemetry['CPU']}
  Memory: {telemetry['Memory']}
  Entropy: {telemetry['Entropy']}

🕵️ Identifiers:
  IP: {identifiers['IP']}
  MAC: {identifiers['MAC']}
  Hostname: {identifiers['Hostname']}
""")

# 🧿 Control Panel
control_frame = tk.LabelFrame(root, text="Command Center", bg="#0f1115", fg="#00ffcc", font=("Consolas", 12))
control_frame.pack(fill="x", padx=20, pady=5)

ttk.Button(control_frame, text="Toggle Lockdown", command=lambda: toggle("lockdown")).pack(side="left", padx=10, pady=10)
ttk.Button(control_frame, text="Toggle Data Out", command=lambda: toggle("data")).pack(side="left", padx=10, pady=10)
ttk.Button(control_frame, text="Toggle Personal Data", command=lambda: toggle("personal")).pack(side="left", padx=10, pady=10)
ttk.Button(control_frame, text="Execute Capsule", command=execute_capsule).pack(side="right", padx=10, pady=10)

def toggle(mode):
    global LOCKDOWN_ACTIVE, ALLOW_DATA_OUT, ALLOW_PERSONAL_DATA_OUT
    if mode == "lockdown":
        LOCKDOWN_ACTIVE = not LOCKDOWN_ACTIVE
    elif mode == "data":
        ALLOW_DATA_OUT = not ALLOW_DATA_OUT
    elif mode == "personal":
        ALLOW_PERSONAL_DATA_OUT = not ALLOW_PERSONAL_DATA_OUT
    update_status()

update_status()
root.mainloop()

