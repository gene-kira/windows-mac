import os
import sys
import platform
import subprocess
import importlib
import random
import base64
import time
import tkinter as tk
from tkinter import ttk

# 🔺 Elevation Check (Windows only)
def is_admin():
    try:
        return os.getuid() == 0  # Unix-like systems
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()  # Windows fallback

if not is_admin():
    if platform.system() == "Windows":
        import ctypes
        script_path = os.path.abspath(sys.argv[0])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}"', None, 1)
        sys.exit()

# 🔥 Lethal Modules to Devour
LETHAL_MODULES = [
    "requests", "psutil", "cryptography", "scapy", "tkinter", "pywin32"
]

# 🧠 Autonomous Devourer Loader
def daemonized_loader(modules):
    for mod in modules:
        try:
            importlib.import_module(mod)
            log_glyph(mod, status="fused")
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", mod])
            try:
                importlib.import_module(mod)
                log_glyph(mod, status="resurrected")
            except ImportError:
                log_glyph(mod, status="failed")
                trigger_purge(mod)

def log_glyph(module, status):
    print(f"[{status.upper()}] 🔥 {module} → Glyph logged")

def trigger_purge(module):
    print(f"[PURGE] ⚠️ Failed to load {module}. Initiating fallback daemon escalation.")

# 🧬 Camouflage Encoder
def xor(data, key):
    return ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))

def generate_rune_key():
    return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(16))

def inject_glyph_noise(encoded):
    noise = ''.join(random.choice("~!@#$%^&*()_+-=[]{}|;:,.<>?") for _ in range(10))
    return encoded + noise

def chameleon_encode(data):
    b64 = base64.b64encode(data.encode()).decode()
    encoded = xor(b64, generate_rune_key())
    padded = inject_glyph_noise(encoded)
    return padded

# 🧪 Synthetic Decoy Generator
def random_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(random.randint(0, int(time.time()))))

def random_country():
    return random.choice(["US", "RU", "CN", "BR", "IN", "DE", "FR", "NG", "JP", "KR"])

def random_glyph_stream():
    return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(64))

def generate_decoy():
    return {
        "timestamp": random_time(),
        "origin": random_country(),
        "payload": random_glyph_stream()
    }

# 🕵️‍♂️ Outbound Filter Daemon
def is_sensitive(packet):
    return isinstance(packet, str) and len(packet) > 10

def mix_streams(real, decoy):
    return {
        "real": real,
        "decoy": decoy,
        "mode": "lethal-autonomous",
        "glyph": random_glyph_stream()
    }

def outbound_filter(packet):
    if is_sensitive(packet):
        real = chameleon_encode(packet)
        decoy = generate_decoy()
        return mix_streams(real, decoy)
    return {"packet": packet, "mode": "pass-through"}

# 🧪 Scapy Hook (optional, runs if scapy is available)
def start_packet_daemon():
    try:
        from scapy.all import sniff, Raw

        def cloak_packet(pkt):
            if pkt.haslayer(Raw):
                try:
                    payload = pkt[Raw].load.decode(errors="ignore")
                    if "session=" in payload:
                        cloaked = outbound_filter(payload)
                        print("🕶️ Cloaked Packet:")
                        print(cloaked)
                except Exception as e:
                    print(f"❌ Error cloaking packet: {e}")

        print("🧠 Scapy Daemon: Listening for outbound packets...")
        sniff(filter="tcp", prn=cloak_packet, store=0)
    except Exception as e:
        print(f"⚠️ Scapy daemon not started: {e}")

# 🧩 Minimalist GUI Panel
def launch_gui():
    root = tk.Tk()
    root.title("Codex Purge Shell")
    root.geometry("400x300")
    root.configure(bg="#1e1e1e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", foreground="#00ffcc", background="#1e1e1e", font=("Consolas", 10))

    ttk.Label(root, text="🧠 Codex Purge Shell").pack(pady=10)
    ttk.Label(root, text="✅ Running with Admin Privileges").pack(pady=5)
    ttk.Label(root, text="🔥 Modules Fused").pack(pady=5)
    ttk.Label(root, text="🕶️ Camouflage Active").pack(pady=5)

    ttk.Label(root, text="🧬 Glyph Stream Preview").pack(pady=10)
    glyph_label = ttk.Label(root, text=random_glyph_stream())
    glyph_label.pack()

    root.after(5000, lambda: glyph_label.config(text=random_glyph_stream()))
    root.mainloop()

# 💰 Safe Trigger Payload
def cash_after_yes():
    print("🔍 [Trigger] Entered cash_after_yes()")

    try:
        print("✅ Elevation confirmed. Modules fused.")
        print("💰 [CASH AFTER YES] → Codex Purge Shell entering autonomous devourer mode…")

        glyph = random_glyph_stream()
        print(f"🧬 Generated glyph stream: {glyph}")

        test_packet = "Diagnostic ping: origin=internal; status=probe"
        cloaked = outbound_filter(test_packet)
        print("🕶️ Outbound filter test result:")
        print(cloaked)

        print("🧩 Launching GUI Panel…")
        launch_gui()

    except Exception as e:
        print(f"❌ [ERROR] Crash inside cash_after_yes(): {e}")

# 🧨 Ritual Ignition
if __name__ == "__main__":
    print("🧠 Running with administrator privileges. Igniting daemon...")
    daemonized_loader(LETHAL_MODULES)

    sample_data = "User telemetry: session=42; origin=US; status=active"
    print("\n🕶️ Outbound Camouflage Activated:")
    cloaked = outbound_filter(sample_data)
    print(cloaked)

    cash_after_yes()

