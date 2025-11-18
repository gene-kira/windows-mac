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

# 🧠 SYMBOLIC DRIFT TRACKER
class SymbolicDriftTracker:
    def __init__(self):
        self.glyph_history = {}

    def record(self, node_id, glyph_signature):
        self.glyph_history.setdefault(node_id, []).append((time.time(), glyph_signature))

    def get_drift(self, node_id):
        return self.glyph_history.get(node_id, [])

# 🛡️ THREAT SIGNATURE DIFF VIEWER
class ThreatSignatureDiffViewer:
    def __init__(self):
        self.node_signatures = {}

    def update_signature(self, node_id, signature):
        self.node_signatures[node_id] = signature

    def compare(self):
        if not self.node_signatures:
            return {}
        base = list(self.node_signatures.values())[0]
        diffs = {}
        for node, sig in self.node_signatures.items():
            diffs[node] = [s for s in sig if s not in base]
        return diffs

# 🕊️ PEACE DAEMON SYNC
class PeaceDaemonSync:
    def __init__(self):
        self.unity_glyph = "🕊️"

    def propagate(self, nodes):
        for node in nodes:
            print(f"[UNITY] Propagating {self.unity_glyph} to {node.node_id}")

# 🗳️ GLYPH CONSENSUS ENGINE
class GlyphConsensusEngine:
    def __init__(self):
        self.proposals = {}  # proposal_id → {glyph, votes}
        self.accepted = []

    def propose(self, node_id, glyph):
        proposal_id = str(uuid.uuid4())
        self.proposals[proposal_id] = {"glyph": glyph, "votes": {node_id}}
        return proposal_id

    def vote(self, node_id, proposal_id):
        if proposal_id in self.proposals:
            self.proposals[proposal_id]["votes"].add(node_id)

    def tally(self, threshold=2):
        for pid, data in list(self.proposals.items()):
            if len(data["votes"]) >= threshold:
                self.accepted.append(data["glyph"])
                del self.proposals[pid]

    def get_status(self):
        return {
            "proposals": {pid: len(data["votes"]) for pid, data in self.proposals.items()},
            "accepted": self.accepted
        }

# 🧠 TRANSMISSION DAEMON
class CodexTransmissionDaemon:
    def __init__(self, sync_engine, gui, drift_tracker, diff_viewer, consensus_engine):
        self.lock = threading.Lock()
        self.queue = []
        self.sync_engine = sync_engine
        self.gui = gui
        self.drift_tracker = drift_tracker
        self.diff_viewer = diff_viewer
        self.consensus_engine = consensus_engine

    def ingest(self, frame):
        with self.lock:
            self.queue.append(frame)
            if len(self.queue) >= 4:
                bundle = GlyphBundle(self.queue[:4])
                self.queue = self.queue[4:]
                self.transmit(bundle)

    def transmit(self, bundle):
        glyph_signature = hash("".join(bundle.payloads))
        self.drift_tracker.record("local_node", glyph_signature)
        self.diff_viewer.update_signature("local_node", bundle.payloads)

        proposal_id = self.consensus_engine.propose("local_node", bundle.payloads)
        self.consensus_engine.vote("local_node", proposal_id)

        if self.detect_resurrection(bundle):
            bundle.skip_flag = True
            self.gui.update_resurrection(bundle.bundle_id)
        else:
            bundle.status = "transmitted"
            self.sync_engine.propagate(bundle)
        self.gui.update_glyph(bundle.bundle_id, bundle.skip_flag)

    def detect_resurrection(self, bundle):
        return any("error" in line.lower() or "fail" in line.lower() for line in bundle.payloads)

class CodexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Transmission Panel")
        self.root.geometry("1100x800")
        self.setup_panels()

    def setup_panels(self):
        self.glyph_label = self._add_panel("Glyph Pulse", "🜏 Waiting for bundle...")
        self.node_list = self._add_list_panel("Swarm Node Map")
        self.port_status = self._add_panel("Port Controls", "Ports: Scanning...")
        self.res_label = self._add_panel("Resurrection Ring", "🕊️ No resurrection detected")
        self.drift_label = self._add_panel("Symbolic Drift Tracker", "No drift recorded")
        self.diff_label = self._add_panel("Threat Signature Diff Viewer", "No differences detected")
        self.consensus_label = self._add_panel("Glyph Consensus Status", "No proposals yet")

    def _add_panel(self, title, default_text):
        frame = ttk.LabelFrame(self.root, text=title)
        frame.pack(fill="x", padx=10, pady=5)
        label = ttk.Label(frame, text=default_text)
        label.pack()
        return label

    def _add_list_panel(self, title):
        frame = ttk.LabelFrame(self.root, text=title)
        frame.pack(fill="x", padx=10, pady=5)
        listbox = tk.Listbox(frame)
        listbox.pack(fill="x")
        return listbox

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

    def update_drift(self, drift):
        self.drift_label.config(text=f"Drift: {len(drift)} glyph mutations")

    def update_diff(self, diffs):
        if not diffs:
            self.diff_label.config(text="No differences detected")
        else:
            text = "\n".join([f"{node}: {len(diff)} differences" for node, diff in diffs.items()])
            self.diff_label.config(text=text)

    def update_consensus(self, status):
        proposals = status["proposals"]
        accepted = status["accepted"]
        text = f"Proposals: {len(proposals)} | Accepted: {len(accepted)}"
        self.consensus_label.config(text=text)

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

def monitor_ports(gui):
    while True:
        gui.update_ports()
        time.sleep(5)

def monitor_drift(gui, tracker):
    while True:
        drift = tracker.get_drift("local_node")
        gui.update_drift(drift)
        time.sleep(10)

def monitor_diff(gui, viewer):
    while True:
        diffs = viewer.compare()
        gui.update_diff(diffs)
        time.sleep(10)

def monitor_consensus(gui, engine):
    while True:
        engine.tally()
        status = engine.get_status()
        gui.update_consensus(status)
        time.sleep(15)

def peace_sync_loop(syncer, nodes):
    while True:
        syncer.propagate(nodes)
        time.sleep(60)

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
    drift_tracker = SymbolicDriftTracker()
    diff_viewer = ThreatSignatureDiffViewer()
    peace_sync = PeaceDaemonSync()
    consensus_engine = GlyphConsensusEngine()

    node1 = SwarmNode("node_alpha", "guardian", "http://localhost:5001/sync")
    node2 = SwarmNode("node_beta", "observer", "http://localhost:5002/sync")
    sync_engine.register_node(node1)
    sync_engine.register_node(node2)
    gui.update_nodes(sync_engine.nodes)

    daemon = CodexTransmissionDaemon(sync_engine, gui, drift_tracker, diff_viewer, consensus_engine)

    log_path = "C:/Windows/System32/LogFiles/Firewall/pfirewall.log" if os.name == "nt" else "/var/log/syslog"

    threading.Thread(target=watch_logs, args=(daemon, log_path), daemon=True).start()
    threading.Thread(target=monitor_ports, args=(gui,), daemon=True).start()
    threading.Thread(target=monitor_drift, args=(gui, drift_tracker), daemon=True).start()
    threading.Thread(target=monitor_diff, args=(gui, diff_viewer), daemon=True).start()
    threading.Thread(target=monitor_consensus, args=(gui, consensus_engine), daemon=True).start()
    threading.Thread(target=peace_sync_loop, args=(peace_sync, sync_engine.nodes), daemon=True).start()
    threading.Thread(target=watchdog, args=(gui,), daemon=True).start()

    root.mainloop()

