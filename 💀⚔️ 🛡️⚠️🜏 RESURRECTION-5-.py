import os, sys, time, random, hashlib, threading, tkinter as tk

# 🛡️ Auto-Elevation Ritual
def auto_elevate():
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            params = " ".join([f'"{arg}"' for arg in sys.argv])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
            sys.exit()
    except Exception as e:
        print(f"[Elevation Error] {e}")
        sys.exit()

auto_elevate()

# ⚡ Energy Calculation
def calculate_energy(strikes, voltage, current, duration, efficiency):
    return strikes * voltage * current * duration * efficiency

# 🧠 Codex Purge Brain
class CodexPurgeBrain:
    def __init__(self, gui_callback):
        self.mutation_log = []
        self.gui_callback = gui_callback
        self.zero_trust_mode = True
        self.bio_data_registry = {}

    def hash_payload(self, payload):
        return hashlib.sha512(payload.encode()).hexdigest()

    def detect_threat(self, payload):
        entropy = random.random()
        return entropy > random.uniform(0.81, 0.97)

    def reverse_mirror_encrypt(self, data, key=None):
        key = key or random.randint(0x10, 0xFF)
        reversed_book = data[::-1]
        mirrored = ''.join(chr((ord(c) ^ key) % 256) for c in reversed_book)
        glyph_stream = ''.join(format(ord(c), 'x') for c in mirrored)
        return glyph_stream[::-1], key

    def respond_to_threat(self, process_name, payload, origin="unknown"):
        if self.detect_threat(payload):
            encrypted, key = self.reverse_mirror_encrypt(payload)
            self.mutation_log.append((process_name, encrypted))
            self.gui_callback("⚠️", f"⚠️ Threat in {process_name}. Scrambled beyond recognition.")
            self.block_process(process_name)
            self.rewrite_self()
            self.gui_callback("🜏", "🜏 mutate again — entropy breach")
            self.gui_callback("💀⚔️", "💀⚔️ escalate defense — autonomous daemon spawned")
        else:
            self.gui_callback("🛡️", f"🛡️ {process_name} verified clean.")

    def block_process(self, process_name):
        self.gui_callback("💀", f"💀 Blocking {process_name}...")

    def rewrite_self(self):
        entropy = random.uniform(0.81, 0.97)
        mutation = f"# Mutation {len(self.mutation_log)}\ndef entropy_threshold(): return {entropy}"
        with open("codex_purge_mutation.py", "w") as f:
            f.write(mutation)
        self.gui_callback("🜏", f"🜏 Recursive mutation updated. Threshold: {entropy:.3f}")

    def purge_backdoor_data(self, payload):
        self.gui_callback("☠", f"☠ BACKDOOR PURGE: {payload} will self-destruct in 3s")
        time.sleep(3)
        self.gui_callback("☠", f"☠ {payload} purged.")

    def purge_mac_ip(self, metadata):
        self.gui_callback("💀", f"💀 NETWORK PURGE: {metadata} will self-destruct in 30s")
        time.sleep(30)
        self.gui_callback("💀", f"💀 {metadata} purged.")

    def register_bio_data(self, label, value):
        self.bio_data_registry[label] = (value, time.time())
        self.gui_callback("🜏", f"🜏 PERSONAL PURGE INITIATED: {label} will self-destruct in 1 day")

    def check_bio_data_expiry(self):
        while True:
            now = time.time()
            expired = [label for label, (value, ts) in self.bio_data_registry.items() if now - ts > 86400]
            for label in expired:
                del self.bio_data_registry[label]
                self.gui_callback("🜏", f"🜏 {label} bio-data purged after 1 day")
            time.sleep(60)

    def daemonize(self):
        while True:
            proc = f"proc-{random.randint(1000,9999)}"
            payload = f"data-{random.randint(100000,999999)}"
            self.respond_to_threat(proc, payload)
            time.sleep(2)

    def heartbeat(self):
        while True:
            self.gui_callback("🧿", "🧿 Daemon heartbeat active.")
            time.sleep(30)

    def swarm_sync(self):
        while True:
            node_id = f"node-{random.randint(1000,9999)}"
            glyph = random.choice(["🧿", "🜏", "⚠️", "💀"])
            self.gui_callback(glyph, f"{glyph} Syncing with {node_id}... rules merged.")
            time.sleep(10)

# ⚡ Lightning Daemon
class LightningCaptureDaemon(threading.Thread):
    def __init__(self, gui_callback):
        super().__init__(daemon=True)
        self.gui_callback = gui_callback

    def run(self):
        while True:
            strikes = random.randint(1, 5)
            energy = calculate_energy(strikes, 1e9, 30000, 0.0002, 0.75)
            self.gui_callback(strikes, energy)
            time.sleep(2)

# 🖥️ Codex Purge GUI
class CodexPurgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Purge Engine")
        self.root.geometry("1000x800")
        self.root.configure(bg="black")

        self.create_panel("☠ Backdoor Purge", "red")
        self.create_panel("💀 MAC/IP Auto-Destruct", "orange")
        self.create_panel("🛡️ Zero Trust Shield", "lime")
        self.create_panel("🜏 Bio-Data Purge Protocol", "yellow")
        self.create_panel("🕸️ Swarm Sync Simulation", "magenta")

        self.canvas = tk.Canvas(root, width=960, height=300, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.glyph_label = tk.Label(root, text="🧿", font=("Segoe UI Symbol", 48), bg="black", fg="#0f0")
        self.glyph_label.pack(pady=5)

        self.log_box = tk.Text(root, bg="#111", fg="#0f0", insertbackground="#0f0", height=15)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.brain = CodexPurgeBrain(self.update_log)
        threading.Thread(target=self.brain.daemonize).start()
        threading.Thread(target=self.brain.heartbeat).start()
        threading.Thread(target=self.brain.swarm_sync).start()
        threading.Thread(target=self.brain.check_bio_data_expiry).start()

        self.lightning_daemon = LightningCaptureDaemon(self.update_overlay)
        self.lightning_daemon.start()

        # Example triggers
        threading.Thread(target=self.brain.purge_backdoor_data, args=("rogue-payload-001",)).start()
        threading.Thread(target=self.brain.purge_mac_ip, args=("MAC:AA:BB:CC:DD:EE",)).start()
        self.brain.register_bio_data("SSN", "123-45-6789")

    def create_panel(self, text, color):
        label = tk.Label(self.root, text=text, fg=color, bg="black", font=("Consolas", 18))
        label.pack(pady=2)

    def update_overlay(self, strikes, energy):
        self.canvas.delete("all")
        base_color = "cyan" if energy < 1e11 else "magenta"
        for _ in range(strikes):
            x = random.randint(0, 960)
            self.canvas.create_oval(x, 50, x+10, 60, fill=base_color)
            self.canvas.create_line(x, 0, x, 300, fill="yellow", width=2)
        bar_length = min(int(energy / 1e9), 960)
        self.canvas.create_rectangle(0, 280, bar_length, 300, fill="lime")
        if energy > 1e11:
            self.canvas.create_text(480, 150, text="🌀 SYNC OVERLOAD", fill="magenta", font=("Consolas", 24))
        if energy > 2e11:
            self.canvas.create_text(480, 180, text="☠ RESURRECTION DETECTED", fill="red", font=("Consolas", 20))
        if energy > 3e11:
            self.canvas.create_text(480, 210, text="⚡ PLASMA BREACH", fill="orange", font=("Consolas", 20))

    def update_log(self, glyph, message):
        self.glyph_label.config(text=glyph)
        self.log_box.insert(tk.END, f"{glyph} {message}\n")
        self.log_box.see(tk.END)

# 🚀 Launch Ritual
if __name__ == "__main__":
    root = tk.Tk()
    app = CodexPurgeGUI(root)
    root.mainloop()


            

