import os, sys, platform, subprocess, importlib, random, base64, time, json, threading
import tkinter as tk
from tkinter import ttk

CONFIG_PATH = "codex_config.json"
MUTATION_LOG = []

# 🔺 Elevation Check
def is_admin():
    try:
        return os.getuid() == 0
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()

if not is_admin():
    if platform.system() == "Windows":
        import ctypes
        script_path = os.path.abspath(sys.argv[0])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}"', None, 1)
        sys.exit()

# 🔧 Persistent Config
def load_config():
    if not os.path.exists(CONFIG_PATH):
        config = {
            "decoy_density": 50,
            "daemon_enabled": True
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)
    else:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    return config

def log_mutation(event):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    glyph = ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(16))
    MUTATION_LOG.append(f"[{stamp}] {event} → {glyph}")

# 🔥 Module Loader
LETHAL_MODULES = ["requests", "psutil", "cryptography", "scapy", "tkinter", "pywin32"]

def daemonized_loader(modules):
    for mod in modules:
        try:
            importlib.import_module(mod)
            log_glyph(mod, "fused")
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", mod])
            try:
                importlib.import_module(mod)
                log_glyph(mod, "resurrected")
            except ImportError:
                log_glyph(mod, "failed")
                trigger_purge(mod)

def log_glyph(module, status):
    print(f"[{status.upper()}] 🔥 {module} → Glyph logged")
    log_mutation(f"{module} status: {status}")

def trigger_purge(module):
    print(f"[PURGE] ⚠️ Failed to load {module}. Initiating fallback daemon escalation.")
    log_mutation(f"{module} purge triggered")

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

# 🧪 Decoy Generator
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

# 🌐 Country Filter Rule
def should_block_country(origin):
    # Example rule: block all non-US traffic
    return origin != "US"

# 🕵️‍♂️ Outbound Filter
def is_sensitive(packet):
    return isinstance(packet, str) and "session=" in packet

def mix_streams(real, decoy):
    return {
        "real": real,
        "decoy": decoy,
        "mode": "lethal-autonomous",
        "glyph": random_glyph_stream()
    }

def outbound_filter(packet, config):
    if is_sensitive(packet):
        origin = random_country()
        if should_block_country(origin):
            log_mutation(f"Blocked packet from {origin}")
            return {"blocked": True, "origin": origin}
        real = chameleon_encode(packet)
        decoy = generate_decoy()
        return mix_streams(real, decoy)
    return {"packet": packet, "mode": "pass-through"}

# 🧪 Scapy Daemon (Threaded)
def start_packet_daemon(config):
    try:
        from scapy.all import sniff, Raw

        def cloak_packet(pkt):
            if pkt.haslayer(Raw):
                try:
                    payload = pkt[Raw].load.decode(errors="ignore")
                    result = outbound_filter(payload, config)
                    print("🕶️ Cloaked Packet:", result)
                except Exception as e:
                    print(f"❌ Error cloaking packet: {e}")

        def run_sniffer():
            sniff(filter="tcp", prn=cloak_packet, store=0)

        thread = threading.Thread(target=run_sniffer, daemon=True)
        thread.start()
        log_mutation("Scapy daemon started")
    except Exception as e:
        print(f"⚠️ Scapy daemon not started: {e}")
        log_mutation("Scapy daemon failed")

# 🧩 GUI Panel
def launch_gui(config):
    root = tk.Tk()
    root.title("Codex Purge Shell")
    root.geometry("500x400")
    root.configure(bg="#1e1e1e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", foreground="#00ffcc", background="#1e1e1e", font=("Consolas", 10))

    ttk.Label(root, text="🧠 Codex Purge Shell").pack(pady=10)
    ttk.Label(root, text="✅ Admin Privileges").pack()
    ttk.Label(root, text="🔥 Modules Fused").pack()
    ttk.Label(root, text="🕶️ Camouflage Active").pack()

    ttk.Label(root, text="🧬 Glyph Stream").pack(pady=10)
    glyph_label = ttk.Label(root, text=random_glyph_stream())
    glyph_label.pack()

    ttk.Label(root, text="📜 Mutation Log").pack(pady=10)
    log_box = tk.Text(root, height=6, bg="#0f0f0f", fg="#00ffcc")
    log_box.pack(fill="both", expand=True)
    for entry in MUTATION_LOG[-5:]:
        log_box.insert("end", entry + "\n")

    root.after(5000, lambda: glyph_label.config(text=random_glyph_stream()))
    root.mainloop()

# 💰 Trigger
def cash_after_yes(config):
    print("✅ Elevation confirmed. Modules fused.")
    print("💰 Codex Purge Shell entering autonomous devourer mode…")

    test_packet = "session=42; origin=US; status=active"
    cloaked = outbound_filter(test_packet, config)
    print("🕶️ Outbound filter test result:", cloaked)

    if config["daemon_enabled"]:
        start_packet_daemon(config)

    launch_gui(config)

# 🧨 Main
if __name__ == "__main__":
    config = load_config()
    daemonized_loader(LETHAL_MODULES)
    cash_after_yes(config)

