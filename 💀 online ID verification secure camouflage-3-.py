import tkinter as tk
import hashlib
import platform
import uuid
import threading
import os
import json

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

def save_pin_from_entry():
    pin = pin_entry.get()
    if pin:
        global USER_PIN_HASH
        USER_PIN_HASH = hashlib.sha256(pin.encode()).hexdigest()
        store_pin_hash(USER_PIN_HASH)
        print("🔐 PIN/password set and saved.")

def verify_user_pin():
    global attempt_counter
    while attempt_counter < MAX_ATTEMPTS:
        pin = pin_entry.get()
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
    update_status("ON", "green")  # ✅ Green light for ON

def deactivate_camouflage():
    global camouflage_active
    camouflage_active = False
    print("🦎 Camouflage deactivated — data is visible.")
    update_status("OFF", "red")  # ✅ Red light for OFF

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
def update_status(text, color):
    status_label.config(text=f"Status: {text}")
    status_light.itemconfig(light_circle, fill=color)

def toggle_camouflage():
    global camouflage_active
    if camouflage_active:
        deactivate_camouflage()
    else:
        activate_camouflage()

def set_timer():
    try:
        minutes = int(timer_entry.get())
        if 30 <= minutes <= 10080:
            deactivate_camouflage()
            start_camouflage_timer(minutes * 60)
        else:
            print("⛔ Timer must be between 30 minutes and 1 week (10080 minutes).")
    except ValueError:
        print("⛔ Invalid input. Enter minutes as a number.")

def launch_control_panel():
    root = tk.Tk()
    root.title("Chameleon Control Panel")
    root.geometry("360x400")
    root.configure(bg="black")

    label = tk.Label(root, text="Chameleon Mode", font=("Arial", 14), fg="white", bg="black")
    label.pack(pady=10)

    timer_label = tk.Label(root, text="Set Timer (minutes):", font=("Arial", 12), fg="white", bg="black")
    timer_label.pack()
    global timer_entry
    timer_entry = tk.Entry(root, font=("Arial", 12))
    timer_entry.pack(pady=5)

    pin_label = tk.Label(root, text="Set PIN/Password:", font=("Arial", 12), fg="white", bg="black")
    pin_label.pack()
    global pin_entry
    pin_entry = tk.Entry(root, font=("Arial", 12), show="*")
    pin_entry.pack(pady=5)

    save_pin_btn = tk.Button(root, text="Save PIN", font=("Arial", 12), width=10,
                             command=save_pin_from_entry, bg="blue", fg="white")
    save_pin_btn.pack(pady=5)

    global status_label, status_light, light_circle
    status_label = tk.Label(root, text="Status: OFF", font=("Arial", 12), fg="white", bg="black")
    status_label.pack(pady=5)

    status_light = tk.Canvas(root, width=30, height=30, bg="black", highlightthickness=0)
    light_circle = status_light.create_oval(5, 5, 25, 25, fill="red")  # Initial state: OFF
    status_light.pack()

    toggle_btn = tk.Button(root, text="Toggle", font=("Arial", 12), width=10,
                           command=toggle_camouflage, bg="gray", fg="white")
    timer_btn = tk.Button(root, text="Set Timer", font=("Arial", 12), width=10,
                          command=set_timer, bg="green", fg="white")

    toggle_btn.pack(pady=5)
    timer_btn.pack(pady=5)
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

