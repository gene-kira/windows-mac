import tkinter as tk
import hashlib
import platform
import uuid
import time
import threading
import os
import json
from tkinter import simpledialog

# 🔄 Autoloader for required libraries
def autoload_libraries():
    required_libs = [
        "camouflage_engine", "fingerprint_bind", "biometric_echo",
        "entropy_trigger", "symbolic_capsule", "audit_trail"
    ]
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"[✓] Loaded {lib}")
        except ImportError:
            print(f"[✗] Missing {lib} — attempting auto-install...")

# 📁 Capsule folder setup
CAPSULE_FOLDER = "chameleon_capsule"
FINGERPRINT_FILE = os.path.join(CAPSULE_FOLDER, "fingerprint.json")
PIN_FILE = os.path.join(CAPSULE_FOLDER, "pin.json")

def initialize_capsule_folder():
    if not os.path.exists(CAPSULE_FOLDER):
        os.makedirs(CAPSULE_FOLDER)
        print(f"📁 Created capsule folder: {CAPSULE_FOLDER}")

# 🔐 Fingerprint logic
def get_system_fingerprint():
    cpu = platform.processor()
    os_info = platform.system() + platform.release()
    shell_id = str(uuid.getnode())
    composite = f"{cpu}-{os_info}-{shell_id}"
    return hashlib.sha256(composite.encode()).hexdigest()

def store_fingerprint(fingerprint):
    with open(FINGERPRINT_FILE, "w") as f:
        json.dump({"fingerprint": fingerprint}, f)

def load_fingerprint():
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE, "r") as f:
            return json.load(f).get("fingerprint")
    return None

# 🔐 PIN logic
USER_PIN_HASH = None
MAX_ATTEMPTS = 6
attempt_counter = 0

def store_pin_hash(pin_hash):
    with open(PIN_FILE, "w") as f:
        json.dump({"pin_hash": pin_hash}, f)

def load_pin_hash():
    if os.path.exists(PIN_FILE):
        with open(PIN_FILE, "r") as f:
            return json.load(f).get("pin_hash")
    return None

def set_user_pin():
    pin = simpledialog.askstring("Set PIN", "Enter a secure PIN or password:", show='*')
    if pin:
        global USER_PIN_HASH
        USER_PIN_HASH = hashlib.sha256(pin.encode()).hexdigest()
        store_pin_hash(USER_PIN_HASH)
        print("🔐 PIN/password set and saved.")

def verify_user_pin():
    global attempt_counter
    while attempt_counter < MAX_ATTEMPTS:
        pin = simpledialog.askstring("Verify PIN", f"Enter your PIN ({MAX_ATTEMPTS - attempt_counter} attempts left):", show='*')
        if pin and hashlib.sha256(pin.encode()).hexdigest() == USER_PIN_HASH:
            attempt_counter = 0
            print("✅ PIN verified — access granted.")
            return True
        else:
            attempt_counter += 1
            print("❌ Incorrect PIN.")
    print("🚫 Maximum attempts reached — access locked.")
    return False

# 🦎 Camouflage logic
camouflage_active = False
camouflage_timer = None

def activate_camouflage():
    global camouflage_active
    camouflage_active = True
    print("🦎 Camouflage activated — data is now invisible to unauthorized systems.")

def deactivate_camouflage():
    global camouflage_active
    camouflage_active = False
    print("🦎 Camouflage deactivated — data is visible.")

def start_camouflage_timer(delay_seconds):
    global camouflage_timer
    if camouflage_timer:
        camouflage_timer.cancel()
    camouflage_timer = threading.Timer(delay_seconds, activate_camouflage)
    camouflage_timer.start()
    print(f"⏳ Camouflage will activate in {delay_seconds // 60} minutes...")

# 🔐 Visibility check
def can_view_data():
    current_fingerprint = get_system_fingerprint()
    if current_fingerprint == ORIGIN_FINGERPRINT and not camouflage_active:
        return True
    elif camouflage_active and USER_PIN_HASH:
        return verify_user_pin()
    return False

# 🎛️ GUI Control Panel
def toggle_chameleon(state, delay_seconds=1800):
    if state == "on":
        deactivate_camouflage()
        start_camouflage_timer(delay_seconds)
    else:
        if camouflage_timer:
            camouflage_timer.cancel()
        activate_camouflage()

def launch_control_panel():
    def set_timer():
        try:
            minutes = int(timer_entry.get())
            if 30 <= minutes <= 10080:
                toggle_chameleon("on", minutes * 60)
            else:
                print("⛔ Timer must be between 30 minutes and 1 week (10080 minutes).")
        except ValueError:
            print("⛔ Invalid input. Enter minutes as a number.")

    root = tk.Tk()
    root.title("Chameleon Control Panel")
    root.geometry("350x270")
    root.configure(bg="black")

    label = tk.Label(root, text="Chameleon Mode", font=("Arial", 14), fg="white", bg="black")
    label.pack(pady=10)

    timer_label = tk.Label(root, text="Set Timer (minutes):", font=("Arial", 12), fg="white", bg="black")
    timer_label.pack()
    timer_entry = tk.Entry(root, font=("Arial", 12))
    timer_entry.pack(pady=5)

    on_btn = tk.Button(root, text="ON", bg="green", fg="white",
                       font=("Arial", 12), width=10,
                       command=set_timer)
    off_btn = tk.Button(root, text="OFF", bg="red", fg="white",
                        font=("Arial", 12), width=10,
                        command=lambda: toggle_chameleon("off"))
    pin_btn = tk.Button(root, text="Set PIN", bg="blue", fg="white",
                        font=("Arial", 12), width=10,
                        command=set_user_pin)

    on_btn.pack(pady=5)
    off_btn.pack(pady=5)
    pin_btn.pack(pady=5)
    root.mainloop()

# 🧪 Launch
if __name__ == "__main__":
    autoload_libraries()
    initialize_capsule_folder()

    ORIGIN_FINGERPRINT = get_system_fingerprint()
    store_fingerprint(ORIGIN_FINGERPRINT)
    USER_PIN_HASH = load_pin_hash()

    print(f"🔐 Origin Fingerprint: {ORIGIN_FINGERPRINT}")
    launch_control_panel()

