# ——————————————————————————————————————————————————————————————
#  THE GATEKEEPER — Event Horizon Port Nerve Center v2.3
#  Tabbed Cockpit • Persistent Memory • Localhost Trust
#  1‑Minute GUI Refresh • 1‑Hour Port Rotation
#  AI Anomaly Brain (auto Aggressive/Balanced)
#  Boost Mode + Micro‑Batch GUI Updates + Multi‑core Scoring
#  Dynamic Memory Path Organ + Manual Backup Override (Local/SMB)
#  Scrollable Tabs / 30‑row views
# ——————————————————————————————————————————————————————————————

import importlib
import subprocess
import sys
import threading
import time
import queue
import logging
import ipaddress
import collections
import random
import datetime
import json
import os
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

# ——— CONFIG ————————————————————————————————————————————————
LOG_FILE = "gatekeeper.log"
POLL_INTERVAL = 3              # engine scan interval (seconds)
LEARNING_DURATION = 60         # auto-learning window (seconds)
ROTATION_INTERVAL = 3600       # port rotation interval (seconds) = 1 hour

COMMON_PORTS = {
    20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 137, 138, 139,
    143, 161, 162, 389, 443, 445, 514, 587, 631, 993, 995, 1433,
    1521, 3306, 3389, 5432, 5900, 6379, 8080
}

ALLOWED_PORTS_BY_PROGRAM = {}
LEARNING_START_TIME = time.time()
LEARNING_LOCKED = False

TRUSTED_LOCAL_SERVICES = {
    "python.exe", "python", "node.exe", "node", "dotnet.exe", "dotnet"
}

telemetry_queue = queue.Queue()
last_rotation_check = time.time()

# multi-core scoring pool
SCORING_EXECUTOR = ThreadPoolExecutor(max_workers=None)

# ——— AUTOLOADER ——————————————————————————————————————————————
def autoload(lib):
    try:
        return importlib.import_module(lib)
    except ImportError:
        print(f"[GATEKEEPER AUTOLOADER] Installing missing library: {lib}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        return importlib.import_module(lib)

psutil = autoload("psutil")

def load_tk():
    tk_mod = autoload("tkinter")
    from tkinter import ttk, filedialog, messagebox
    return tk_mod, ttk, filedialog, messagebox

# ——— LOGGING ————————————————————————————————————————————————
logger = logging.getLogger("Gatekeeper")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
formatter = logging.Formatter(
    "%(asctime)s [GATEKEEPER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ——— HELPERS ————————————————————————————————————————————————
def classify_port(port):
    return "INPUT_ONLY" if port in COMMON_PORTS else "OUTPUT_ONLY"

def classify_remote_ip(ip):
    if not ip:
        return "unknown"
    try:
        obj = ipaddress.ip_address(ip)
        if obj.is_private:
            return "private"
        if obj.is_loopback:
            return "loopback"
        if obj.is_multicast:
            return "multicast"
        return "public"
    except ValueError:
        return "invalid"

def is_overseas(ip):
    return classify_remote_ip(ip) == "public"

def program_key(entry):
    return entry["program_path"] or entry["program"]

# ——— ROTATING OUTPUT PORT AUTHORITY ————————————————
class OutputPortAuthority:
    def __init__(self, pool_size=20, min_port=20000, max_port=60000):
        self.pool_size = pool_size
        self.min_port = min_port
        self.max_port = max_port
        self.current_ports = set()
        self.last_rotation = None
        self.rotate_ports()

    def rotate_ports(self):
        self.current_ports = set(
            random.sample(range(self.min_port, self.max_port), self.pool_size)
        )
        self.last_rotation = datetime.datetime.now()
        logger.info(f"[ROTATION] New output ports: {sorted(self.current_ports)}")

    def is_allowed_output(self, port):
        return port in self.current_ports

OUTPUT_AUTHORITY = OutputPortAuthority()

# ——— PREDICTIVE BRAIN ————————————————————————————————
class PredictiveBrain:
    def __init__(self):
        self.history = collections.defaultdict(
            lambda: collections.deque(maxlen=200)
        )

    def record(self, key, port, score):
        self.history[key].append((port, score))

    def predict(self, key, port):
        h = self.history.get(key)
        if not h:
            return 50
        same = [ts for p, ts in h if p == port]
        all_ts = [ts for _, ts in h]
        if not all_ts:
            return 50
        avg_all = sum(all_ts) / len(all_ts)
        if same:
            avg_same = sum(same) / len(same)
            return max(0, min(100, (avg_all + avg_same) / 2))
        return max(0, min(100, avg_all + 20))

PREDICTIVE_BRAIN = PredictiveBrain()

def trust_score(entry):
    ip = entry["remote_ip"]
    prog = (entry["program"] or "").lower()
    cls = classify_remote_ip(ip)

    if ip == "127.0.0.1":
        return 100
    if cls == "private" and prog in TRUSTED_LOCAL_SERVICES:
        return 90
    if cls == "private":
        return 75
    if cls == "loopback":
        return 80
    if cls == "public":
        return 40
    return 50

def threat_score(entry):
    score = 0
    ip = entry["remote_ip"]
    prog = (entry["program"] or "").lower()
    port = entry["port"]
    cls = classify_remote_ip(ip)

    if entry["role"] == "OUTPUT_ONLY":
        score += 10
    if cls == "public":
        score += 30
    if port >= 49152:
        score += 15
    if any(x in prog for x in ["powershell", "cmd", "python", "wscript", "cscript"]):
        score += 20

    key = program_key(entry)
    PREDICTIVE_BRAIN.record(key, port, score)
    predicted = PREDICTIVE_BRAIN.predict(key, port)
    return max(0, min(100, (score + predicted) / 2))

# ——— ANOMALY BRAIN ————————————————————————————————
class AnomalyBrain:
    def __init__(self):
        self.mode = "BALANCED"
        self.recent_anomalies = collections.deque(maxlen=200)

    def record_anomaly(self, severity):
        self.recent_anomalies.append((time.time(), severity))
        self._update_mode()

    def _update_mode(self):
        now = time.time()
        window = 600
        recent = [s for t, s in self.recent_anomalies if now - t <= window]
        rate = len(recent)
        if rate <= 3:
            new_mode = "AGGRESSIVE"
        else:
            new_mode = "BALANCED"
        if new_mode != self.mode:
            logger.info(f"[ANOMALY BRAIN] Mode change: {self.mode} → {new_mode}")
            self.mode = new_mode

    def detect_anomaly(self, entry, tscore, trust):
        cls = classify_remote_ip(entry["remote_ip"])
        key = program_key(entry)

        if self.mode == "AGGRESSIVE":
            high_thr = 60
            med_thr = 40
        else:
            high_thr = 75
            med_thr = 55

        reasons = []

        if tscore >= high_thr:
            reasons.append(f"High threat score {tscore}")
        elif tscore >= med_thr:
            reasons.append(f"Elevated threat score {tscore}")

        if tscore >= 70 and trust >= 80:
            reasons.append("High threat but high trust (mismatch)")
        if tscore <= 40 and trust <= 40:
            reasons.append("Low threat but low trust (suspicious baseline)")

        if cls == "public" and trust < 60 and tscore >= med_thr:
            reasons.append("Overseas connection with low trust")

        hist = PREDICTIVE_BRAIN.history.get(key)
        if hist:
            ports_seen = {p for p, _ in hist}
            if entry["port"] not in ports_seen and entry["port"] >= 49152:
                reasons.append("New high ephemeral port for this program")

        if not reasons:
            return False, None, None

        if any("High threat" in r or "Overseas" in r for r in reasons):
            severity = "HIGH"
        elif any("Elevated" in r or "mismatch" in r for r in reasons):
            severity = "MEDIUM"
        else:
            severity = "LOW"

        reason_str = "; ".join(reasons)
        self.record_anomaly(severity)
        return True, severity, reason_str

ANOMALY_BRAIN = AnomalyBrain()

# ——— MEMORY ORGAN v2.3 ——————————————————————————————
class GatekeeperMemory:
    def __init__(self):
        self.current_path = None
        self.file = None
        self.smb_paths = [
            r"\\SERVER\GatekeeperMemory",
            r"\\192.168.1.10\GatekeeperMemory"
        ]
        # modes: "AUTO", "MANUAL_LOCAL", "MANUAL_SMB"
        self.mode = "AUTO"
        self.manual_path = None
        self.find_best_path(initial=True)

    def test_write(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, "test.tmp")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return True
        except Exception:
            return False

    def find_best_path(self, initial=False):
        if self.mode != "AUTO":
            # manual override: do not auto-switch
            if self.manual_path:
                self.current_path = self.manual_path
                self.file = os.path.join(self.manual_path, "gatekeeper_memory.json")
            return self.current_path

        # 1. Local drives D: → Z:
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            drive = f"{letter}:\\Gatekeeper"
            if os.path.exists(root) and self.test_write(drive):
                return self.switch_path(drive, initial)

        # 2. SMB paths
        for smb in self.smb_paths:
            if self.test_write(smb):
                return self.switch_path(smb, initial)

        # 3. Fallback to C:
        fallback = "C:\\Gatekeeper"
        os.makedirs(fallback, exist_ok=True)
        return self.switch_path(fallback, initial)

    def switch_path(self, new_path, initial):
        old_path = self.current_path
        self.current_path = new_path
        self.file = os.path.join(new_path, "gatekeeper_memory.json")

        if initial:
            logger.info(f"[MEMORY] Using storage path: {new_path}")
            return new_path

        if old_path != new_path:
            logger.info(f"[MIGRATION] Memory path changed: {old_path} → {new_path}")

            try:
                old_file = os.path.join(old_path, "gatekeeper_memory.json") if old_path else None
                if old_file and os.path.exists(old_file):
                    import shutil
                    shutil.copy2(old_file, self.file)
                    logger.info("[MIGRATION] Memory file migrated successfully.")
            except Exception as e:
                logger.error(f"[MIGRATION] Failed to migrate memory file: {e}")

        return new_path

    def set_manual_path(self, path, mode_label, messagebox=None):
        self.mode = mode_label
        self.manual_path = path
        self.current_path = path
        self.file = os.path.join(path, "gatekeeper_memory.json")
        logger.info(f"[MEMORY] Manual override set: mode={mode_label}, path={path}")
        if messagebox:
            messagebox.showinfo(
                "Gatekeeper Backup Override",
                f"Backup mode set to {mode_label}.\nPath:\n{path}\n\nAuto-switching disabled."
            )

    def set_auto_mode(self, messagebox=None):
        self.mode = "AUTO"
        self.manual_path = None
        self.find_best_path(initial=False)
        logger.info(f"[MEMORY] Backup mode set to AUTO. Dynamic memory organ re-enabled.")
        if messagebox:
            messagebox.showinfo(
                "Gatekeeper Backup Mode",
                f"Backup mode set to AUTOMATIC.\nDynamic memory path organ re-enabled.\nCurrent path:\n{self.current_path}"
            )

    def save(self):
        if self.mode == "AUTO":
            self.find_best_path(initial=False)
        else:
            if not self.manual_path:
                logger.error("[MEMORY] Manual mode active but no manual_path set.")
                return
            if not os.path.exists(self.manual_path):
                logger.error(f"[MEMORY] Manual backup path not available: {self.manual_path}")
                return
            self.current_path = self.manual_path
            self.file = os.path.join(self.manual_path, "gatekeeper_memory.json")

        try:
            data = {
                "allowed_ports": {k: list(v) for k, v in ALLOWED_PORTS_BY_PROGRAM.items()},
                "predictive": {k: list(v) for k, v in PREDICTIVE_BRAIN.history.items()},
                "rotation": list(OUTPUT_AUTHORITY.current_ports),
                "rotation_time": OUTPUT_AUTHORITY.last_rotation.isoformat()
                    if OUTPUT_AUTHORITY.last_rotation else None,
                "locked": LEARNING_LOCKED,
                "anomaly_mode": ANOMALY_BRAIN.mode,
                "backup_mode": self.mode,
                "manual_path": self.manual_path
            }
            with open(self.file, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"[MEMORY] Saved to {self.file}")
        except Exception as e:
            logger.error(f"[MEMORY] Save failed: {e}")

    def load(self):
        # initial path selection (AUTO by default)
        self.find_best_path(initial=True)

        if not self.file or not os.path.exists(self.file):
            logger.info("[MEMORY] No memory file found.")
            return

        try:
            with open(self.file) as f:
                data = json.load(f)

            for k, v in data.get("allowed_ports", {}).items():
                ALLOWED_PORTS_BY_PROGRAM[k] = set(v)

            for k, v in data.get("predictive", {}).items():
                PREDICTIVE_BRAIN.history[k] = collections.deque(v, maxlen=200)

            rotation_ports = data.get("rotation", [])
            if rotation_ports:
                OUTPUT_AUTHORITY.current_ports = set(rotation_ports)

            rotation_time = data.get("rotation_time")
            if rotation_time:
                OUTPUT_AUTHORITY.last_rotation = datetime.datetime.fromisoformat(rotation_time)

            global LEARNING_LOCKED
            LEARNING_LOCKED = data.get("locked", False)

            mode = data.get("anomaly_mode")
            if mode in ("AGGRESSIVE", "BALANCED"):
                ANOMALY_BRAIN.mode = mode

            # backup mode restore
            bmode = data.get("backup_mode", "AUTO")
            mpath = data.get("manual_path", None)
            self.mode = bmode
            self.manual_path = mpath
            if self.mode == "AUTO":
                self.find_best_path(initial=False)
            else:
                if self.manual_path:
                    self.current_path = self.manual_path
                    self.file = os.path.join(self.manual_path, "gatekeeper_memory.json")

            logger.info(f"[MEMORY] Loaded from {self.file} (mode={self.mode}, manual_path={self.manual_path})")
        except Exception as e:
            logger.error(f"[MEMORY] Load failed: {e}")

MEMORY = GatekeeperMemory()

# ——— POLICY ENGINE ——————————————————————————————————————
def update_learning_lock():
    global LEARNING_LOCKED
    if not LEARNING_LOCKED and (time.time() - LEARNING_START_TIME) > LEARNING_DURATION:
        LEARNING_LOCKED = True
        logger.info("[GATEKEEPER] Auto-learning window ended. Ports are now locked.")

def apply_policy(entry):
    if entry["remote_ip"] == "127.0.0.1":
        logger.info(
            f"[LOCALHOST] {entry['program']} (PID {entry['pid']}) "
            f"{entry['local_ip']} → 127.0.0.1 port {entry['port']} (friendly)"
        )
        return

    update_learning_lock()

    if entry["direction"] == "REMOTE":
        if not OUTPUT_AUTHORITY.is_allowed_output(entry["port"]):
            logger.warning(
                f"[ROTATION VIOLATION] {entry['program']} (PID {entry['pid']}) "
                f"used forbidden output port {entry['port']} "
                f"(allowed: {sorted(OUTPUT_AUTHORITY.current_ports)})"
            )
            telemetry_queue.put({"violation": entry})
            return

    key = program_key(entry)
    port = entry["port"]

    if key not in ALLOWED_PORTS_BY_PROGRAM:
        ALLOWED_PORTS_BY_PROGRAM[key] = set()

    if not LEARNING_LOCKED:
        if port not in ALLOWED_PORTS_BY_PROGRAM[key]:
            ALLOWED_PORTS_BY_PROGRAM[key].add(port)
            logger.info(
                f"[LEARN] {key} now allowed port {port}. "
                f"Set: {sorted(ALLOWED_PORTS_BY_PROGRAM[key])}"
            )
        return

    if port not in ALLOWED_PORTS_BY_PROGRAM[key]:
        logger.warning(
            f"[VIOLATION] {entry['program']} (PID {entry['pid']}) "
            f"used unauthorized port {port} "
            f"{entry['local_ip']} → {entry['remote_ip']}"
        )
        telemetry_queue.put({"violation": entry})

# ——— MULTI-CORE SCORING TASK ——————————————————————————————
def score_and_analyze(entry):
    t = threat_score(entry)
    tr = trust_score(entry)
    is_anom, sev, reason = ANOMALY_BRAIN.detect_anomaly(entry, t, tr)
    anomaly_record = None
    if is_anom:
        anomaly_record = {
            "entry": entry,
            "threat": t,
            "trust": tr,
            "severity": sev,
            "reason": reason,
            "timestamp": time.time()
        }
    entry["threat"] = t
    entry["trust"] = tr
    return entry, anomaly_record

# ——— ENGINE THREAD ——————————————————————————————————————
def port_policy_engine():
    global last_rotation_check
    logger.info("[GATEKEEPER] Engine online.")
    last_save = 0

    while True:
        now = time.time()

        if now - last_rotation_check >= ROTATION_INTERVAL:
            OUTPUT_AUTHORITY.rotate_ports()
            last_rotation_check = now

        try:
            conns = psutil.net_connections(kind='inet')
        except Exception as e:
            logger.error(f"[ENGINE] Error reading connections: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        snapshot = []
        for c in conns:
            l = c.laddr if c.laddr else None
            r = c.raddr if c.raddr else None

            lport = l.port if l else None
            rport = r.port if r else None
            lip = l.ip if l else None
            rip = r.ip if r else None

            try:
                p = psutil.Process(c.pid) if c.pid else None
                name = p.name() if p else "Unknown"
                path = p.exe() if p else ""
            except Exception:
                name = "Unknown"
                path = ""

            if lport:
                entry = {
                    "direction": "LOCAL",
                    "port": lport,
                    "role": classify_port(lport),
                    "status": str(c.status),
                    "pid": c.pid,
                    "local_ip": lip,
                    "remote_ip": rip,
                    "program": name,
                    "program_path": path
                }
                snapshot.append(entry)
                apply_policy(entry)

            if rport:
                entry = {
                    "direction": "REMOTE",
                    "port": rport,
                    "role": classify_port(rport),
                    "status": str(c.status),
                    "pid": c.pid,
                    "local_ip": lip,
                    "remote_ip": rip,
                    "program": name,
                    "program_path": path
                }
                snapshot.append(entry)
                apply_policy(entry)

        enriched_snapshot = []
        futures = SCORING_EXECUTOR.map(score_and_analyze, snapshot)
        for enriched, anomaly_record in futures:
            enriched_snapshot.append(enriched)
            if anomaly_record is not None:
                try:
                    telemetry_queue.put({"anomaly": anomaly_record}, timeout=0.1)
                except queue.Full:
                    pass

        try:
            telemetry_queue.put(enriched_snapshot, timeout=1)
        except queue.Full:
            pass

        if now - last_save > 30:
            MEMORY.save()
            last_save = now

        time.sleep(POLL_INTERVAL)

# ——— SCROLLABLE FRAME ————————————————————————————————
def make_scrollable_frame(parent, tk, ttk):
    container = ttk.Frame(parent)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    container.pack(fill="both", expand=True)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return container, scrollable_frame

# ——— GUI — TABBED COCKPIT + BOOST MODE + BACKUP TAB ——————
class PortPolicyCockpit:
    def __init__(self, root, ttk, tk, filedialog, messagebox):
        self.root = root
        self.ttk = ttk
        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox

        root.title("THE GATEKEEPER — Event Horizon Port Nerve Center v2.3")

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{int(sw*0.90)}x{int(sh*0.90)}")

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)

        # raw tab frames
        self.tab_dashboard = ttk.Frame(self.tabs)
        self.tab_intel = ttk.Frame(self.tabs)
        self.tab_local = ttk.Frame(self.tabs)
        self.tab_overseas = ttk.Frame(self.tabs)
        self.tab_violations = ttk.Frame(self.tabs)
        self.tab_anomaly = ttk.Frame(self.tabs)
        self.tab_logs = ttk.Frame(self.tabs)
        self.tab_backup = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_dashboard, text="Dashboard")
        self.tabs.add(self.tab_intel, text="Intelligence Core")
        self.tabs.add(self.tab_local, text="Local Traffic")
        self.tabs.add(self.tab_overseas, text="Overseas Traffic")
        self.tabs.add(self.tab_violations, text="Violations")
        self.tabs.add(self.tab_anomaly, text="Anomaly Detection")
        self.tabs.add(self.tab_logs, text="Logs")
        self.tabs.add(self.tab_backup, text="Backup Settings")

        # scrollable wrappers
        _, dash_frame = make_scrollable_frame(self.tab_dashboard, tk, ttk)
        _, intel_frame = make_scrollable_frame(self.tab_intel, tk, ttk)
        _, local_frame = make_scrollable_frame(self.tab_local, tk, ttk)
        _, overseas_frame = make_scrollable_frame(self.tab_overseas, tk, ttk)
        _, viol_frame = make_scrollable_frame(self.tab_violations, tk, ttk)
        _, anomaly_frame = make_scrollable_frame(self.tab_anomaly, tk, ttk)
        _, logs_frame = make_scrollable_frame(self.tab_logs, tk, ttk)
        _, backup_frame = make_scrollable_frame(self.tab_backup, tk, ttk)

        # Dashboard
        self.status_label = ttk.Label(dash_frame, text="ENGINE ONLINE")
        self.status_label.pack(fill="x")

        self.refresh_timer_label = ttk.Label(dash_frame, text="GUI Refresh in: 60s")
        self.refresh_timer_label.pack(fill="x")

        self.rotation_timer_label = ttk.Label(dash_frame, text="Port Rotation in: 60m 0s")
        self.rotation_timer_label.pack(fill="x")

        self.rotation_label = ttk.Label(dash_frame, text="")
        self.rotation_label.pack(fill="x")

        columns = (
            "direction", "program", "path", "local_ip", "remote_ip",
            "port", "role", "status", "pid", "threat", "trust"
        )
        self.tree = ttk.Treeview(dash_frame, columns=columns, show="headings", height=30)
        for col in columns:
            self.tree.heading(col, text=col.upper())
            self.tree.column(col, width=120, anchor="center")
        tree_scroll = ttk.Scrollbar(dash_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # Intelligence Core
        self.ai_text = tk.Text(intel_frame, height=4, state="disabled")
        self.ai_text.pack(fill="x")

        self.graph_canvas = tk.Canvas(intel_frame, height=120, bg="black")
        self.graph_canvas.pack(fill="x")
        self.graph_points = collections.deque(maxlen=100)

        self.anomaly_mode_label = ttk.Label(
            intel_frame,
            text=f"Anomaly Brain Mode: {ANOMALY_BRAIN.mode}"
        )
        self.anomaly_mode_label.pack(fill="x")

        # Local Traffic
        self.local_text = tk.Text(local_frame, height=20, state="disabled")
        local_scroll = ttk.Scrollbar(local_frame, orient="vertical", command=self.local_text.yview)
        self.local_text.configure(yscrollcommand=local_scroll.set)
        self.local_text.pack(side="left", fill="both", expand=True)
        local_scroll.pack(side="right", fill="y")

        # Overseas Traffic
        self.overseas_text = tk.Text(overseas_frame, height=20, state="disabled")
        overseas_scroll = ttk.Scrollbar(overseas_frame, orient="vertical", command=self.overseas_text.yview)
        self.overseas_text.configure(yscrollcommand=overseas_scroll.set)
        self.overseas_text.pack(side="left", fill="both", expand=True)
        overseas_scroll.pack(side="right", fill="y")

        # Violations
        self.violations_text = tk.Text(viol_frame, height=20, state="disabled")
        viol_scroll = ttk.Scrollbar(viol_frame, orient="vertical", command=self.violations_text.yview)
        self.violations_text.configure(yscrollcommand=viol_scroll.set)
        self.violations_text.pack(side="left", fill="both", expand=True)
        viol_scroll.pack(side="right", fill="y")

        # Anomaly Detection
        self.anomaly_summary = ttk.Label(
            anomaly_frame,
            text="Anomalies detected will appear below."
        )
        self.anomaly_summary.pack(fill="x")

        anomaly_cols = (
            "time", "severity", "program", "pid",
            "local_ip", "remote_ip", "port", "threat", "trust", "reason"
        )
        self.anomaly_tree = ttk.Treeview(
            anomaly_frame, columns=anomaly_cols, show="headings", height=15
        )
        for col in anomaly_cols:
            self.anomaly_tree.heading(col, text=col.upper())
            self.anomaly_tree.column(col, width=120, anchor="center")
        anomaly_scroll = ttk.Scrollbar(anomaly_frame, orient="vertical", command=self.anomaly_tree.yview)
        self.anomaly_tree.configure(yscrollcommand=anomaly_scroll.set)
        self.anomaly_tree.pack(side="left", fill="both", expand=True)
        anomaly_scroll.pack(side="right", fill="y")

        # Logs
        self.log_label = ttk.Label(logs_frame, text=f"Audit Log: {LOG_FILE}")
        self.log_label.pack(fill="x")

        self.log_text = tk.Text(logs_frame, height=25, state="disabled")
        log_scroll = ttk.Scrollbar(logs_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        # Backup Settings
        self.backup_mode_var = self.tk.StringVar(value=MEMORY.mode)
        mode_frame = ttk.LabelFrame(backup_frame, text="Backup Mode")
        mode_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            mode_frame, text="Automatic (Dynamic Memory Path)",
            variable=self.backup_mode_var, value="AUTO",
            command=self.on_backup_mode_change
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode_frame, text="Manual Local Backup",
            variable=self.backup_mode_var, value="MANUAL_LOCAL",
            command=self.on_backup_mode_change
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode_frame, text="Manual SMB Backup",
            variable=self.backup_mode_var, value="MANUAL_SMB",
            command=self.on_backup_mode_change
        ).pack(anchor="w")

        btn_frame = ttk.Frame(backup_frame)
        btn_frame.pack(fill="x", pady=5)

        self.btn_choose_local = ttk.Button(
            btn_frame, text="Choose Local Backup Folder",
            command=self.choose_local_backup
        )
        self.btn_choose_local.pack(fill="x", pady=2)

        self.btn_choose_smb = ttk.Button(
            btn_frame, text="Choose SMB Backup Folder",
            command=self.choose_smb_backup
        )
        self.btn_choose_smb.pack(fill="x", pady=2)

        self.backup_status_label = ttk.Label(
            backup_frame,
            text=self.get_backup_status_text()
        )
        self.backup_status_label.pack(fill="x", pady=5)

        # timers
        self.next_refresh_time = time.time() + 60

        # micro-batch state
        self.pending_rows = collections.deque()
        self.batch_size = 50
        self.batch_delay = 10
        self.last_graph_score = None

        self.root.after(60000, self.update_from_engine)
        self.update_timers()
        self.load_log_file()

    # backup helpers
    def get_backup_status_text(self):
        return (
            f"Backup Mode: {MEMORY.mode}\n"
            f"Manual Path: {MEMORY.manual_path}\n"
            f"Current Path: {MEMORY.current_path}\n"
            f"Auto-switching: {'Enabled' if MEMORY.mode == 'AUTO' else 'Disabled'}"
        )

    def on_backup_mode_change(self):
        mode = self.backup_mode_var.get()
        if mode == "AUTO":
            MEMORY.set_auto_mode(self.messagebox)
        elif mode == "MANUAL_LOCAL":
            # wait for user to choose path
            pass
        elif mode == "MANUAL_SMB":
            # wait for user to choose path
            pass
        self.backup_status_label.config(text=self.get_backup_status_text())

    def choose_local_backup(self):
        path = self.filedialog.askdirectory(title="Choose Local Backup Folder")
        if not path:
            return
        MEMORY.set_manual_path(path, "MANUAL_LOCAL", self.messagebox)
        self.backup_mode_var.set("MANUAL_LOCAL")
        self.backup_status_label.config(text=self.get_backup_status_text())

    def choose_smb_backup(self):
        path = self.filedialog.askdirectory(title="Choose SMB/Network Backup Folder")
        if not path:
            return
        MEMORY.set_manual_path(path, "MANUAL_SMB", self.messagebox)
        self.backup_mode_var.set("MANUAL_SMB")
        self.backup_status_label.config(text=self.get_backup_status_text())

    # GUI update loop
    def update_from_engine(self):
        updated = False
        last_score = None

        try:
            while True:
                item = telemetry_queue.get_nowait()

                if isinstance(item, dict) and "violation" in item:
                    self.add_violation(item["violation"])
                    continue

                if isinstance(item, dict) and "anomaly" in item:
                    self.add_anomaly(item["anomaly"])
                    continue

                if isinstance(item, list):
                    last_score = self.prepare_table(item)
                    updated = True

        except queue.Empty:
            pass

        if updated:
            mode = "LOCKED" if LEARNING_LOCKED else "LEARNING"
            self.status_label.config(
                text=f"ENGINE ONLINE • MODE: {mode}"
            )
            self.rotation_label.config(
                text=f"Output Ports: {sorted(OUTPUT_AUTHORITY.current_ports)}"
            )
            self.anomaly_mode_label.config(
                text=f"Anomaly Brain Mode: {ANOMALY_BRAIN.mode}"
            )
            if last_score is not None:
                self.last_graph_score = last_score

            self.next_refresh_time = time.time() + 60

        self.root.after(60000, self.update_from_engine)

    def update_timers(self):
        now = time.time()

        refresh_remaining = int(self.next_refresh_time - now)
        if refresh_remaining < 0:
            refresh_remaining = 0
        self.refresh_timer_label.config(text=f"GUI Refresh in: {refresh_remaining}s")

        global last_rotation_check
        rotation_next = last_rotation_check + ROTATION_INTERVAL
        rotation_remaining = int(rotation_next - now)
        if rotation_remaining < 0:
            rotation_remaining = 0

        mins = rotation_remaining // 60
        secs = rotation_remaining % 60
        self.rotation_timer_label.config(
            text=f"Port Rotation in: {mins}m {secs}s"
        )

        if refresh_remaining <= 3:
            self.batch_size = 200
            self.batch_delay = 1
        else:
            self.batch_size = 75
            self.batch_delay = 5

        self.root.after(1000, self.update_timers)

    def prepare_table(self, snapshot):
        for i in self.tree.get_children():
            self.tree.delete(i)

        self.local_text.config(state="normal")
        self.local_text.delete("1.0", "end")
        self.local_text.config(state="disabled")

        self.overseas_text.config(state="normal")
        self.overseas_text.delete("1.0", "end")
        self.overseas_text.config(state="disabled")

        self.pending_rows.clear()
        last_score = None

        for e in snapshot:
            t = e.get("threat", threat_score(e))
            last_score = t
            self.pending_rows.append(e)

        self.insert_rows_batch()
        return last_score

    def insert_rows_batch(self):
        count = 0
        while self.pending_rows and count < self.batch_size:
            e = self.pending_rows.popleft()
            t = e.get("threat", threat_score(e))
            trust = e.get("trust", trust_score(e))

            tag = "normal"
            if e["remote_ip"] == "127.0.0.1":
                tag = "localhost"
            elif is_overseas(e["remote_ip"]):
                tag = "overseas"

            self.tree.insert(
                "",
                "end",
                values=(
                    e["direction"], e["program"], e["program_path"],
                    e["local_ip"], e["remote_ip"], e["port"],
                    e["role"], e["status"], e["pid"], t, trust
                ),
                tags=(tag,)
            )

            if tag == "localhost":
                self.add_local(e)
            if tag == "overseas":
                self.add_overseas(e)

            count += 1

        if self.pending_rows:
            self.root.after(self.batch_delay, self.insert_rows_batch)
        else:
            if self.last_graph_score is not None:
                self.update_graph(self.last_graph_score)

    def add_local(self, e):
        self.local_text.config(state="normal")
        self.local_text.insert(
            "end",
            f"{e['program']} (PID {e['pid']}) → {e['remote_ip']} port {e['port']}\n"
        )
        self.local_text.see("end")
        self.local_text.config(state="disabled")

    def add_overseas(self, e):
        self.overseas_text.config(state="normal")
        self.overseas_text.insert(
            "end",
            f"{e['program']} (PID {e['pid']}) → {e['remote_ip']} port {e['port']}\n"
        )
        self.overseas_text.see("end")
        self.overseas_text.config(state="disabled")

    def add_violation(self, entry):
        msg = (
            f"{entry['program']} (PID {entry['pid']}) "
            f"→ unauthorized port {entry['port']} "
            f"{entry['local_ip']} → {entry['remote_ip']} "
            f"role={entry['role']} status={entry['status']} "
            f"threat={entry.get('threat', threat_score(entry))}\n"
        )
        self.violations_text.config(state="normal")
        self.violations_text.insert("end", msg)
        self.violations_text.see("end")
        self.violations_text.config(state="disabled")

    def add_anomaly(self, record):
        e = record["entry"]
        t = record["threat"]
        tr = record["trust"]
        sev = record["severity"]
        reason = record["reason"]
        ts = datetime.datetime.fromtimestamp(record["timestamp"]).strftime("%H:%M:%S")

        self.anomaly_tree.insert(
            "",
            "end",
            values=(
                ts,
                sev,
                e["program"],
                e["pid"],
                e["local_ip"],
                e["remote_ip"],
                e["port"],
                t,
                tr,
                reason
            )
        )

        self.add_ai_event(
            f"[{sev}] {e['program']} PID {e['pid']} port {e['port']} → {e['remote_ip']} :: {reason}"
        )

    def update_graph(self, score):
        self.graph_points.append(score)
        self.graph_canvas.delete("all")

        if not self.graph_points:
            return

        w = int(self.graph_canvas.winfo_width() or 1)
        h = int(self.graph_canvas.winfo_height() or 1)
        n = len(self.graph_points)

        prev_x = None
        prev_y = None

        for i, s in enumerate(self.graph_points):
            x = int(i * w / max(1, n - 1))
            y = int(h - (s / 100) * h)

            if prev_x is not None:
                self.graph_canvas.create_line(prev_x, prev_y, x, y, fill="lime", width=2)

            prev_x, prev_y = x, y

    def add_ai_event(self, msg):
        self.ai_text.config(state="normal")
        self.ai_text.insert("end", msg + "\n")
        self.ai_text.see("end")
        self.ai_text.config(state="disabled")

    def load_log_file(self):
        if not os.path.exists(LOG_FILE):
            return
        try:
            with open(LOG_FILE, "r") as f:
                content = f.read()
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", content)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        except Exception:
            pass

# ——— MAIN ————————————————————————————————————————————————
def main():
    headless = "--headless" in sys.argv

    MEMORY.load()

    engine_thread = threading.Thread(
        target=port_policy_engine,
        daemon=True
    )
    engine_thread.start()

    if headless:
        logger.info("[GATEKEEPER] Running in headless mode.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("[GATEKEEPER] Shutdown.")
        return

    tk, ttk, filedialog, messagebox = load_tk()
    root = tk.Tk()
    app = PortPolicyCockpit(root, ttk, tk, filedialog, messagebox)
    root.mainloop()

if __name__ == "__main__":
    main()

