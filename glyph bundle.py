import importlib, subprocess, sys, threading, time, uuid, requests, tkinter as tk, os
from tkinter import ttk
import psutil

# ⚙️ AUTLOADER
REQUIRED_LIBRARIES = ["requests", "uuid", "threading", "time", "json", "tkinter", "psutil"]

def autoload():
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"[🜏] {lib} loaded")
        except ImportError:
            print(f"[⚠️] {lib} missing")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                print(f"[🕊️] {lib} installed")
            except Exception as e:
                print(f"[💀] Failed to install {lib}: {e}")

# 🧬 GLYPH BUNDLE
class GlyphBundle:
    def __init__(self, payloads):
        self.bundle_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.payloads = payloads
        self.status = "queued"
        self.skip_flag = False

# 🌐 SWARM NODE
class SwarmNode:
    def __init__(self, node_id, persona, endpoint):
        self.node_id = node_id
        self.persona = persona
        self.endpoint = endpoint
        self.last_sync = 0

# 🌐 SYNC ENGINE
class SwarmSyncEngine:
    def __init__(self):
        self.nodes = []
        self.synced_bundles = {}

    def register_node(self, node):
        self.nodes.append(node)

    def propagate(self, bundle):
        for node in self.nodes:
            if bundle.bundle_id not in self.synced_bundles.get(node.node_id, []):
                try:
                    requests.post(node.endpoint, json={"bundle_id": bundle.bundle_id, "payloads": bundle.payloads})
                    self.synced_bundles.setdefault(node.node_id, []).append(bundle.bundle_id)
                    node.last_sync = time.time()
                except Exception as e:
                    print(f"[SYNC ERROR] Node {node.node_id}: {e}")

# 🧠 TRANSMISSION DAEMON
class CodexTransmissionDaemon:
    def __init__(self, sync_engine, gui):
        self.lock = threading.Lock()
        self.queue = []
        self.sync_engine = sync_engine
        self.gui = gui

    def ingest(self, frame):
        with self.lock:
            self.queue.append(frame)
            if len(self.queue) >= 4:
                bundle = GlyphBundle(self.queue[:4])
                self.queue = self.queue[4:]
                self.transmit(bundle)

    def transmit(self, bundle):
        if self.detect_resurrection(bundle):
            bundle.skip_flag = True
            self.gui.update_resurrection(bundle.bundle_id)
        else:
            bundle.status = "transmitted"
            self.sync_engine.propagate(bundle)
        self.gui.update_glyph(bundle.bundle_id, bundle.skip_flag)

    def detect_resurrection(self, bundle):
        return any("error" in line.lower() or "fail" in line.lower() for line in bundle.payloads)

# 🖥️ GUI PANEL
class CodexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Transmission Panel")
        self.root.geometry("900x600")
        self.setup_panels()

    def setup_panels(self):
        self.glyph_frame = ttk.LabelFrame(self.root, text="Glyph Pulse")
        self.glyph_frame.pack(fill="x", padx=10, pady=5)
        self.glyph_label = ttk.Label(self.glyph_frame, text="🜏 Waiting for bundle...")
        self.glyph_label.pack()

        self.node_frame = ttk.LabelFrame(self.root, text="Swarm Node Map")
        self.node_frame.pack(fill="x", padx=10, pady=5)
        self.node_list = tk.Listbox(self.node_frame)
        self.node_list.pack(fill="x")

        self.port_frame = ttk.LabelFrame(self.root, text="Port Controls")
        self.port_frame.pack(fill="x", padx=10, pady=5)
        self.port_status = ttk.Label(self.port_frame, text="Ports: Scanning...")
        self.port_status.pack()

        self.res_frame = ttk.LabelFrame(self.root, text="Resurrection Ring")
        self.res_frame.pack(fill="x", padx=10, pady=5)
        self.res_label = ttk.Label(self.res_frame, text="🕊️ No resurrection detected")
        self.res_label.pack()

    def update_glyph(self, bundle_id, skip_flag):
        pulse = "⚔️" if skip_flag else "🛡️"
        self.glyph_label.config(text=f"{pulse} Bundle {bundle_id} processed")

    def update_nodes(self, nodes):
        self.node_list.delete(0, tk.END)
        for node in nodes:
            self.node_list.insert(tk.END, f"{node.node_id} [{node.persona}]")

    def update_resurrection(self, bundle_id):
        self.res_label.config(text=f"💀 Resurrection triggered for {bundle_id}")

    def update_ports(self):
        ports = [conn.laddr.port for conn in psutil.net_connections() if conn.status == 'LISTEN']
        self.port_status.config(text=f"Ports: {', '.join(map(str, ports))} ✅")

# 🔍 REAL-TIME LOG WATCHER
def watch_logs(daemon, log_path):
    if not os.path.exists(log_path):
        print(f"[⚠️] Log file not found: {log_path}")
        return
    with open(log_path, "r", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        buffer = []
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            buffer.append(line.strip())
            if len(buffer) == 4:
                daemon.ingest(buffer)
                buffer = []

# 🔁 PORT MONITOR LOOP
def monitor_ports(gui):
    while True:
        gui.update_ports()
        time.sleep(5)

# 🛡️ SELF-HEALING WATCHDOG
def watchdog(gui):
    while True:
        try:
            if not gui.root.winfo_exists():
                print("[💀] GUI closed. Restarting...")
                os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as e:
            print(f"[⚠️] Watchdog error: {e}")
        time.sleep(10)

# 🚀 SYSTEM BOOT
if __name__ == "__main__":
    autoload()

    root = tk.Tk()
    gui = CodexGUI(root)

    sync_engine = SwarmSyncEngine()
    node1 = SwarmNode("node_alpha", "guardian", "http://localhost:5001/sync")
    node2 = SwarmNode("node_beta", "observer", "http://localhost:5002/sync")
    sync_engine.register_node(node1)
    sync_engine.register_node(node2)
    gui.update_nodes(sync_engine.nodes)

    daemon = CodexTransmissionDaemon(sync_engine, gui)

    log_path = "C:/Windows/System32/LogFiles/Firewall/pfirewall.log" if os.name == "nt" else "/var/log/syslog"

    threading.Thread(target=watch_logs, args=(daemon, log_path), daemon=True).start()
    threading.Thread(target=monitor_ports, args=(gui,), daemon=True).start()
    threading.Thread(target=watchdog, args=(gui,), daemon=True).start()

    root.mainloop()

