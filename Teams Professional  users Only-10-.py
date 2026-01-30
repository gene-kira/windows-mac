import psutil
import threading
import time
import tkinter as tk
from tkinter import ttk
import os
import json
from datetime import datetime
import statistics

# ---------- CONFIG ----------

CHECK_INTERVAL = 2.0

APPDATA = os.getenv("APPDATA") or os.getcwd()
LISTS_FILE = os.path.join(APPDATA, "sentinel_lists.json")
LOG_FILE = os.path.join(APPDATA, "sentinel_events.log")
BEHAVIOR_FILE = os.path.join(APPDATA, "sentinel_behavior.json")
STATS_FILE = os.path.join(APPDATA, "sentinel_stats.json")

ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()

BEHAVIOR_DB = {}
STATS_DB = {
    "history": []
}

WHITELIST_NAMES = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
}

LIST_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
BEHAVIOR_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()

BASE_LOCKDOWN_SCORE_THRESHOLD = 70
MIN_LOCKDOWN_THRESHOLD = 40
MAX_LOCKDOWN_THRESHOLD = 90

MAX_SCORE_HISTORY = 20
MAX_CPU_HISTORY = 20
MOMENTUM_WINDOW = 5
NEW_KEY_AGE_SECONDS = 60
NEW_FILE_AGE_SECONDS = 120


# ---------- HELPERS ----------

def norm_path(p: str | None) -> str | None:
    if not p:
        return None
    return os.path.normpath(p).replace("\\", "/").lower()


# ---------- LOGGING ----------

def log_event(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    with LOG_LOCK:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass


# ---------- LIST STORAGE ----------

def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE
    with LIST_LOCK:
        if not os.path.exists(LISTS_FILE):
            return
        try:
            with open(LISTS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_lists.json unreadable, starting with empty lists.")
            return

    # Normalize everything on load
    ALLOWLIST = set(norm_path(x) for x in d.get("allow", []) if x)
    BLOCKLIST = set(norm_path(x) for x in d.get("block", []) if x)
    RADIOACTIVE = set(norm_path(x) for x in d.get("radioactive", []) if x)


def save_lists():
    data = {
        "allow": sorted(ALLOWLIST),
        "block": sorted(BLOCKLIST),
        "radioactive": sorted(RADIOACTIVE),
    }

    tmp_file = LISTS_FILE + ".tmp"
    with LIST_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, LISTS_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save lists: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ---------- BEHAVIOR STORAGE ----------

def load_behavior():
    global BEHAVIOR_DB
    with BEHAVIOR_LOCK:
        if not os.path.exists(BEHAVIOR_FILE):
            BEHAVIOR_DB = {}
            return
        try:
            with open(BEHAVIOR_FILE, "r", encoding="utf-8") as f:
                BEHAVIOR_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_behavior.json unreadable, starting fresh.")
            BEHAVIOR_DB = {}
            return

        for key, rec in BEHAVIOR_DB.items():
            rec.setdefault("cluster_dir", "")
            rec.setdefault("cluster_name_pattern", "")
            rec.setdefault("cluster_key", "")
            rec.setdefault("score_history", [])
            rec.setdefault("cpu_history", [])
            rec.setdefault("file_size", None)
            rec.setdefault("file_mtime", None)
            rec.setdefault("parent_type", None)
            rec.setdefault("first_seen", time.time())
            rec.setdefault("last_seen", rec["first_seen"])
            rec.setdefault("launch_count", 0)
            rec.setdefault("total_cpu", 0.0)
            rec.setdefault("max_cpu", 0.0)
            rec.setdefault("has_network", False)
            rec.setdefault("parent_pids", [])
            rec.setdefault("user_label", None)
            rec.setdefault("last_score", 0)
            rec.setdefault("fingerprints", [])


def save_behavior():
    tmp_file = BEHAVIOR_FILE + ".tmp"
    with BEHAVIOR_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(BEHAVIOR_DB, f, indent=2)
            os.replace(tmp_file, BEHAVIOR_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save behavior DB: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ---------- STATS STORAGE ----------

def load_stats():
    global STATS_DB
    with STATS_LOCK:
        if not os.path.exists(STATS_FILE):
            STATS_DB = {"history": []}
            return
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                STATS_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_stats.json unreadable, starting fresh.")
            STATS_DB = {"history": []}


def save_stats():
    tmp_file = STATS_FILE + ".tmp"
    with STATS_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(STATS_DB, f, indent=2)
            os.replace(tmp_file, STATS_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save stats DB: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ---------- BRAIN ----------

class SentinelBrain:
    def classify(self, proc):
        try:
            name = proc.name()
            exe_raw = proc.exe() or ""
            exe = norm_path(exe_raw) or f"<no-path:{proc.pid}>"
            name_l = name.lower()
            cpu = proc.cpu_percent(interval=None)
            ppid = proc.ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        if exe in BLOCKLIST or name_l in BLOCKLIST:
            status = "BLOCKED"
        elif exe in ALLOWLIST:
            status = "ALLOW"
        elif exe in RADIOACTIVE:
            status = "RADIOACTIVE"
        elif name_l in WHITELIST_NAMES:
            status = "TRUSTED"
        else:
            status = "SUSPICIOUS"

        threat = self.threat_level(status)

        return {
            "pid": proc.pid,
            "ppid": ppid,
            "name": name,
            "name_l": name_l,
            "cpu": cpu,
            "path": exe,
            "status": status,
            "threat": threat,
            "score": 0,
            "anomalies": [],
            "fingerprint": [],
            "matched_bad_fp": False,
            "momentum": 0.0,
            "age_seconds": 0.0,
            "cluster_stats": {},
        }

    @staticmethod
    def threat_level(status):
        if status in ("BLOCKED", "RADIOACTIVE"):
            return "HIGH"
        if status == "SUSPICIOUS":
            return "MEDIUM"
        if status in ("TRUSTED", "ALLOW"):
            return "LOW"
        return "UNKNOWN"

    @staticmethod
    def compute_fingerprint(info, behavior_record, extra_features):
        fp = []
        path = info["path"]
        name_l = info["name_l"]
        cpu = info["cpu"]
        status = info["status"]

        lower_path = path.lower()
        if any(x in lower_path for x in ["/temp/", "\\temp\\", "\\appdata\\", "/appdata/"]):
            fp.append("temp_path")

        base_name = os.path.basename(path)
        if not base_name or base_name.startswith("<no-path:"):
            base_name = name_l
        if len(base_name) > 10:
            consonants = sum(1 for c in base_name if c.lower() in "bcdfghjklmnpqrstvwxyz")
            if consonants / max(1, len(base_name)) > 0.6:
                fp.append("random_name")

        if cpu > 50:
            fp.append("high_cpu")

        if behavior_record:
            launches = behavior_record.get("launch_count", 0)
            max_cpu = behavior_record.get("max_cpu", 0.0)
            has_net = behavior_record.get("has_network", False)
            first_seen = behavior_record.get("first_seen", 0)
            last_seen = behavior_record.get("last_seen", 0)
            lifetime = last_seen - first_seen if last_seen and first_seen else 0

            if launches <= 1 and lifetime < 10:
                fp.append("ephemeral")

            if has_net:
                fp.append("has_net")

            if cpu > max_cpu * 2 and max_cpu > 20:
                fp.append("cpu_spike")

        if status in ("BLOCKED", "RADIOACTIVE"):
            fp.append("user_bad")
        elif status in ("ALLOW", "TRUSTED"):
            fp.append("user_good")

        if extra_features.get("night_time"):
            fp.append("night_run")
        if extra_features.get("sibling_burst"):
            fp.append("sibling_burst")
        if extra_features.get("new_file"):
            fp.append("new_file")
        if extra_features.get("temp_net"):
            fp.append("temp_net")
        if extra_features.get("parent_type"):
            fp.append(f"parent_{extra_features['parent_type']}")

        return sorted(set(fp))

    @staticmethod
    def compute_momentum(score_history):
        if not score_history:
            return 0.0
        window = score_history[-MOMENTUM_WINDOW:]
        if not window:
            return 0.0
        weights = list(range(1, len(window) + 1))
        total_w = sum(weights)
        return sum(s * w for s, w in zip(window, weights)) / total_w

    @staticmethod
    def compute_score(info, behavior_record, bad_fingerprints, cluster_stats, now):
        status = info["status"]

        # Hard override: ALLOW means score 0, LOW, no anomalies
        if status == "ALLOW":
            return 0, "LOW", [], [], False, 0.0, 0.0, None

        score = 0
        anomalies = []
        matched_bad_fp = False

        path = info["path"]
        name_l = info["name_l"]
        cpu = info["cpu"]

        lower_path = path.lower()
        in_temp = any(x in lower_path for x in ["/temp/", "\\temp\\", "\\appdata\\", "/appdata/"])
        if in_temp:
            score += 20
            anomalies.append("Runs from temp/appdata")

        base_name = os.path.basename(path)
        if not base_name or base_name.startswith("<no-path:"):
            base_name = name_l
        if len(base_name) > 10:
            consonants = sum(1 for c in base_name if c.lower() in "bcdfghjklmnpqrstvwxyz")
            if consonants / max(1, len(base_name)) > 0.6:
                score += 15
                anomalies.append("Random-looking name")

        if cpu > 50:
            score += 15
            anomalies.append(f"High CPU: {cpu:.1f}%")
        if cpu > 80:
            score += 10

        launches = behavior_record.get("launch_count", 0) if behavior_record else 0
        max_cpu = behavior_record.get("max_cpu", 0.0) if behavior_record else 0.0
        has_net = behavior_record.get("has_network", False) if behavior_record else False
        first_seen = behavior_record.get("first_seen", now) if behavior_record else now
        last_seen = behavior_record.get("last_seen", now) if behavior_record else now
        lifetime = last_seen - first_seen if last_seen and first_seen else 0
        age_seconds = now - first_seen if first_seen else 0

        file_size = behavior_record.get("file_size") if behavior_record else None
        file_mtime = behavior_record.get("file_mtime") if behavior_record else None
        new_file = False
        if file_mtime:
            file_age = now - file_mtime
            if file_age < NEW_FILE_AGE_SECONDS:
                new_file = True
                score += 10
                anomalies.append("Recently created binary on disk")

        if age_seconds < NEW_KEY_AGE_SECONDS and status == "SUSPICIOUS":
            score += 10
            anomalies.append("Early-life suspicious process")

        if launches <= 1 and lifetime < 10:
            score += 10
            anomalies.append("Ephemeral process")

        if has_net:
            score += 25
            anomalies.append("Has network activity")

        if cpu > max_cpu * 2 and max_cpu > 20:
            score += 15
            anomalies.append("CPU spike anomaly")

        if status == "BLOCKED":
            score += 40
        elif status == "RADIOACTIVE":
            score += 25
        elif status == "SUSPICIOUS":
            score += 10
        elif status == "TRUSTED":
            score -= 20

        user_label = behavior_record.get("user_label") if behavior_record else None
        if user_label == "block":
            score += 30
        elif user_label == "radio":
            score += 15
        elif user_label == "allow":
            score -= 25

        extra_features = {}

        hour = datetime.fromtimestamp(now).hour
        if hour < 6 or hour > 22:
            extra_features["night_time"] = True

        if in_temp and has_net:
            extra_features["temp_net"] = True

        parent_type = behavior_record.get("parent_type") if behavior_record else None
        if parent_type:
            extra_features["parent_type"] = parent_type

        sibling_burst = False
        cluster_key = behavior_record.get("cluster_key") if behavior_record else None
        if cluster_key and cluster_key in cluster_stats:
            cstat = cluster_stats[cluster_key]
            if cstat["recent_members"] >= 5:
                sibling_burst = True
                score += 10
                anomalies.append("Cluster burst: many similar processes recently")
        if sibling_burst:
            extra_features["sibling_burst"] = True

        if new_file:
            extra_features["new_file"] = True

        fp = SentinelBrain.compute_fingerprint(info, behavior_record, extra_features)

        for bad_fp in bad_fingerprints:
            overlap = len(set(fp) & set(bad_fp))
            if overlap >= 3:
                score += 25
                anomalies.append("Matched bad fingerprint from previous block")
                matched_bad_fp = True
                break

        score_history = behavior_record.get("score_history", []) if behavior_record else []
        momentum = SentinelBrain.compute_momentum(score_history + [score])
        if momentum > 60 and score < momentum:
            score += 10
            anomalies.append("Rising threat momentum")
        elif momentum > 40:
            score += 5
            anomalies.append("Elevated threat momentum")

        if cluster_key and cluster_key in cluster_stats:
            cstat = cluster_stats[cluster_key]
            if cstat["bad_count"] >= 2:
                score += 15
                anomalies.append("Cluster has multiple bad members")
            if cstat["avg_score"] >= 60:
                score += 10
                anomalies.append("Cluster average score is high")

        if score < 0:
            score = 0
        if score > 100:
            score = 100

        if score >= 70:
            threat = "HIGH"
        elif score >= 40:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        return score, threat, anomalies, fp, matched_bad_fp, momentum, age_seconds, cluster_key


# ---------- GUI ----------

class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel – Predictive Guardian")

        self.brain = SentinelBrain()
        self.running = True
        self.lockdown = tk.BooleanVar(value=False)
        self.simulate_kills = tk.BooleanVar(value=False)

        self.sort_column = "score"
        self.sort_reverse = True

        self.monitor_thread = None
        self.watchdog_thread = None

        self.current_rows = []
        self.current_net_rows = []

        self.dynamic_lockdown_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD

        self._build_ui()
        self._build_context_menu()
        self.refresh_lists_ui()

        self.start_monitor_thread()
        self.start_watchdog_thread()

    # ---------- THREAD MANAGEMENT ----------

    def start_monitor_thread(self):
        t = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread = t
        t.start()
        log_event("Monitor thread started.")

    def start_watchdog_thread(self):
        t = threading.Thread(target=self.watchdog_loop, daemon=True)
        self.watchdog_thread = t
        t.start()
        log_event("Watchdog thread started.")

    # ---------- UI BUILD ----------

    def _build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x")

        self.lockdown_check = ttk.Checkbutton(
            top_frame,
            text="Lockdown Mode (adaptive score-based kill)",
            variable=self.lockdown,
            command=self.on_lockdown_toggle,
        )
        self.lockdown_check.pack(side="left", padx=5, pady=2)

        self.simulate_check = ttk.Checkbutton(
            top_frame,
            text="Simulate Pre-emptive Kills",
            variable=self.simulate_kills,
            command=self.on_simulate_toggle,
        )
        self.simulate_check.pack(side="left", padx=5, pady=2)

        self.threshold_label = ttk.Label(top_frame, text="Threshold: ?")
        self.threshold_label.pack(side="left", padx=(10, 2))

        ttk.Label(top_frame, text="Filter:").pack(side="left", padx=(10, 2))
        self.filter_var = tk.StringVar(value="ALL")
        self.filter_combo = ttk.Combobox(
            top_frame,
            textvariable=self.filter_var,
            values=["ALL", "HIGH", "MEDIUM", "LOW"],
            state="readonly",
            width=8,
        )
        self.filter_combo.current(0)
        self.filter_combo.pack(side="left", padx=2)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_process_view())

        whatif_frame = ttk.Frame(self.root)
        whatif_frame.pack(fill="x", pady=(2, 4))

        ttk.Label(whatif_frame, text="What-if kills:").pack(side="left", padx=(5, 2))
        self.whatif_50 = ttk.Label(whatif_frame, text="T50: 0")
        self.whatif_50.pack(side="left", padx=5)
        self.whatif_60 = ttk.Label(whatif_frame, text="T60: 0")
        self.whatif_60.pack(side="left", padx=5)
        self.whatif_70 = ttk.Label(whatif_frame, text="T70: 0")
        self.whatif_70.pack(side="left", padx=5)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        proc_frame = ttk.Frame(notebook)
        notebook.add(proc_frame, text="Processes")

        pane = ttk.PanedWindow(proc_frame, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane)
        pane.add(left, weight=3)

        self.tree = ttk.Treeview(
            left,
            columns=("name", "pid", "cpu", "status", "threat", "score", "path"),
            show="headings",
        )
        for c in ("name", "pid", "cpu", "status", "threat", "score", "path"):
            self.tree.heading(c, text=c.upper(), command=lambda col=c: self.on_tree_heading_click(col))
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<Double-1>", self.show_explanation_popup)

        self.tree.tag_configure("HIGH", foreground="red")
        self.tree.tag_configure("MEDIUM", foreground="orange")
        self.tree.tag_configure("LOW", foreground="green")
        self.tree.tag_configure("UNKNOWN", foreground="gray")

        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        self.allow_box = self._make_box(right, "ALLOWLIST")
        self.block_box = self._make_box(right, "BLOCKLIST")
        self.radio_box = self._make_box(right, "RADIOACTIVE")

        ttk.Button(
            right, text="Remove Selected",
            command=self.remove_selected_from_lists
        ).pack(fill="x", pady=5)

        net_frame = ttk.Frame(notebook)
        notebook.add(net_frame, text="Network")

        self.net_tree = ttk.Treeview(
            net_frame,
            columns=("laddr", "raddr", "status", "pid", "name"),
            show="headings",
        )
        for c in ("laddr", "raddr", "status", "pid", "name"):
            self.net_tree.heading(c, text=c.upper())
        self.net_tree.pack(fill="both", expand=True)

    def _make_box(self, parent, title):
        f = ttk.LabelFrame(parent, text=title)
        f.pack(fill="both", expand=True, padx=5, pady=5)

        lb = tk.Listbox(f)
        lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(f, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        return lb

    # ---------- CONTEXT MENU ----------

    def _build_context_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="➕ Add to Allowlist", command=self.add_selected_allow)
        self.menu.add_command(label="⛔ Add to Blocklist", command=self.add_selected_block)
        self.menu.add_command(label="☢ Add to Radioactive", command=self.add_selected_radio)

    def show_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.menu.tk_popup(event.x_root, event.y_root)
            self.menu.grab_release()

    # ---------- EXPLANATION POPUP ----------

    def show_explanation_popup(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0], "values")
        if not item or len(item) < 7:
            return

        name, pid, cpu, status, threat, score, path = item
        pid = int(pid)
        row = None
        for r in self.current_rows:
            if r["pid"] == pid and r["path"] == path:
                row = r
                break
        if not row:
            return

        anomalies = row.get("anomalies", [])
        fp = row.get("fingerprint", [])
        momentum = row.get("momentum", 0.0)
        age_seconds = row.get("age_seconds", 0.0)
        cluster_stats = row.get("cluster_stats", {})

        win = tk.Toplevel(self.root)
        win.title(f"Explanation – {name} (PID {pid})")
        win.geometry("550x450")

        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True)

        text.insert("end", f"Name: {name}\n")
        text.insert("end", f"PID: {pid}\n")
        text.insert("end", f"Path: {path}\n")
        text.insert("end", f"Status: {status}\n")
        text.insert("end", f"Threat: {threat}\n")
        text.insert("end", f"Score: {score}\n")
        text.insert("end", f"CPU: {cpu}%\n")
        text.insert("end", f"Age: {age_seconds:.1f} seconds\n")
        text.insert("end", f"Momentum: {momentum:.1f}\n\n")

        text.insert("end", "Cluster stats:\n")
        if cluster_stats:
            text.insert("end", f"  Members: {cluster_stats.get('members', 0)}\n")
            text.insert("end", f"  Bad count: {cluster_stats.get('bad_count', 0)}\n")
            text.insert("end", f"  Avg score: {cluster_stats.get('avg_score', 0.0):.1f}\n")
            text.insert("end", f"  Recent members: {cluster_stats.get('recent_members', 0)}\n\n")
        else:
            text.insert("end", "  (none)\n\n")

        text.insert("end", "Anomalies:\n")
        if anomalies:
            for a in anomalies:
                text.insert("end", f"  - {a}\n")
        else:
            text.insert("end", "  (none)\n")

        text.insert("end", "\nFingerprint:\n")
        if fp:
            for f in fp:
                text.insert("end", f"  - {f}\n")
        else:
            text.insert("end", "  (none)\n")

        text.config(state="disabled")

    # ---------- SORTING / FILTERING ----------

    def on_tree_heading_click(self, column):
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        self.refresh_process_view()

    # ---------- SELECTION HELPERS ----------

    def _selected_values(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.item(sel[0], "values")

    def _selected_path(self):
        v = self._selected_values()
        if not v or len(v) < 7:
            return None
        return v[6] if v[6] else None

    def _selected_name_pid(self):
        v = self._selected_values()
        if not v or len(v) < 2:
            return None, None
        name = v[0]
        try:
            pid = int(v[1])
        except (TypeError, ValueError):
            pid = None
        return name, pid

    def _key_for_behavior(self, path, name_l):
        if path and not path.startswith("<no-path:"):
            return path
        return name_l

    # ---------- LIST OPERATIONS (FULLY PATCHED) ----------

    def add_selected_allow(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return

        name_l = name.lower()
        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        ALLOWLIST.add(norm)
        BLOCKLIST.discard(norm)
        RADIOACTIVE.discard(norm)
        save_lists()

        log_event(f"ALLOW OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "allow"
            rec["fingerprints"] = []
            rec["last_score"] = 0
            rec["score_history"] = []
            rec["cpu_history"] = []
            BEHAVIOR_DB[key] = rec

        save_behavior()

        self.refresh_lists_ui()
        self.refresh_process_view()

    def add_selected_block(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()

        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        BLOCKLIST.add(norm)
        ALLOWLIST.discard(norm)
        RADIOACTIVE.discard(norm)
        save_lists()

        log_event(f"BLOCK OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "block"
            fps = set(tuple(fp) for fp in rec.get("fingerprints", []))
            rec["fingerprints"] = [list(x) for x in fps]
            rec["last_score"] = max(rec.get("last_score", 0), 90)
            BEHAVIOR_DB[key] = rec
        save_behavior()

        if pid is not None:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                log_event(f"BLOCK: Immediately terminated PID {pid} ({name}) key={key}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                log_event(f"BLOCK: Failed to terminate PID {pid}: {e}")

        self.refresh_lists_ui()
        self.refresh_process_view()

    def add_selected_radio(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()

        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        RADIOACTIVE.add(norm)
        ALLOWLIST.discard(norm)
        BLOCKLIST.discard(norm)
        save_lists()

        log_event(f"RADIO OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "radio"
            rec["last_score"] = max(rec.get("last_score", 0), 70)
            BEHAVIOR_DB[key] = rec
        save_behavior()

        self.refresh_lists_ui()
        self.refresh_process_view()

    def refresh_lists_ui(self):
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            lb.delete(0, "end")
            for item in sorted(s):
                lb.insert("end", item)

    def remove_selected_from_lists(self):
        changed = False
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            sel = lb.curselection()
            if sel:
                val = lb.get(sel[0])
                if val in s:
                    s.discard(val)
                    log_event(f"Removed from list: {val}")
                    changed = True
        if changed:
            save_lists()
            self.refresh_lists_ui()
            self.refresh_process_view()

    # ---------- AUTO-FILL LOGIC (USING NORMALIZED PATHS) ----------

    def auto_update_lists(self, info):
        p = info["path"]
        if not p or p.startswith("<no-path:"):
            return False

        np = norm_path(p)
        changed = False

        if info["status"] == "TRUSTED" and np not in ALLOWLIST:
            ALLOWLIST.add(np)
            RADIOACTIVE.discard(np)
            BLOCKLIST.discard(np)
            changed = True
            log_event(f"AUTO: Trusted path added to ALLOWLIST: {np}")

        elif info["status"] == "SUSPICIOUS" and np not in RADIOACTIVE and np not in ALLOWLIST:
            try:
                proc = psutil.Process(info["pid"])
                if proc.create_time() > time.time() - 3:
                    return changed
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

            RADIOACTIVE.add(np)
            changed = True
            log_event(f"AUTO: Suspicious path added to RADIOACTIVE: {np}")

        if changed:
            save_lists()
        return changed

    # ---------- LOCKDOWN MODE ----------

    def on_lockdown_toggle(self):
        state = "ON" if self.lockdown.get() else "OFF"
        log_event(f"Lockdown mode toggled: {state}")

    def on_simulate_toggle(self):
        state = "ON" if self.simulate_kills.get() else "OFF"
        log_event(f"Simulate pre-emptive kills toggled: {state}")

    def enforce_lockdown(self, info):
        if not self.lockdown.get():
            return

        status = info["status"]
        path = info["path"]
        pid = info["pid"]
        score = info.get("score", 0)
        anomalies = info.get("anomalies", [])
        matched_bad_fp = info.get("matched_bad_fp", False)

        if status in ("ALLOW", "TRUSTED"):
            return

        np = norm_path(path) if path and not path.startswith("<no-path:") else None
        if np and np in ALLOWLIST:
            return

        if score < self.dynamic_lockdown_threshold and status not in ("BLOCKED", "RADIOACTIVE"):
            return

        reasons = ", ".join(anomalies) if anomalies else "no anomalies listed"

        if self.simulate_kills.get():
            log_event(
                f"SIMULATED KILL: would have terminated PID {pid} ({info['name']}) "
                f"path={path} status={status} score={score} "
                f"threshold={self.dynamic_lockdown_threshold} reasons=[{reasons}]"
            )
            return

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            msg = (
                f"LOCKDOWN: Terminated PID {pid} ({info['name']}) "
                f"path={path} status={status} score={score} "
                f"threshold={self.dynamic_lockdown_threshold}"
            )
            if matched_bad_fp:
                msg += " (Matched bad fingerprint from previous block)"
            log_event(msg)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            log_event(f"LOCKDOWN: Failed to terminate PID {pid}: {e}")

    # ---------- BEHAVIOR TRACKING / FINGERPRINTS / CLUSTERS ----------

    def update_behavior(self, proc, info, bad_fingerprints, cluster_stats, now):
        path = info["path"]
        name_l = info["name_l"]
        key = self._key_for_behavior(path, name_l)

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {
                "first_seen": now,
                "last_seen": now,
                "launch_count": 0,
                "total_cpu": 0.0,
                "max_cpu": 0.0,
                "has_network": False,
                "parent_pids": [],
                "user_label": None,
                "last_score": 0,
                "fingerprints": [],
                "cluster_dir": "",
                "cluster_name_pattern": "",
                "cluster_key": "",
                "score_history": [],
                "cpu_history": [],
                "file_size": None,
                "file_mtime": None,
                "parent_type": None,
            })

            if rec.get("file_size") is None and not path.startswith("<no-path:"):
                try:
                    st = os.stat(path)
                    rec["file_size"] = st.st_size
                    rec["file_mtime"] = st.st_mtime
                except OSError:
                    rec["file_size"] = None
                    rec["file_mtime"] = None

            rec["last_seen"] = now
            rec["launch_count"] = rec.get("launch_count", 0) + 1
            rec["total_cpu"] = rec.get("total_cpu", 0.0) + info["cpu"]
            rec["max_cpu"] = max(rec.get("max_cpu", 0.0), info["cpu"])

            parent_pids = set(rec.get("parent_pids", []))
            parent_pids.add(info["ppid"])
            rec["parent_pids"] = list(parent_pids)

            parent_type = rec.get("parent_type")
            if not parent_type:
                try:
                    pproc = psutil.Process(info["ppid"])
                    pname = pproc.name().lower()
                    if any(x in pname for x in ["chrome", "edge", "firefox", "brave", "opera"]):
                        parent_type = "browser"
                    elif any(x in pname for x in ["winword", "excel", "powerpnt"]):
                        parent_type = "office"
                    elif "explorer" in pname:
                        parent_type = "shell"
                    else:
                        parent_type = "other"
                    rec["parent_type"] = parent_type
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    rec["parent_type"] = "unknown"

            if not path.startswith("<no-path:"):
                rec["cluster_dir"] = os.path.dirname(path)
            base_name = os.path.basename(path) or name_l
            rec["cluster_name_pattern"] = self.name_pattern(base_name)
            cluster_key = f"{rec['cluster_dir']}|{rec['cluster_name_pattern']}"
            rec["cluster_key"] = cluster_key

            BEHAVIOR_DB[key] = rec

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            score, threat, anomalies, fp, matched_bad_fp, momentum, age_seconds, cluster_key = \
                self.brain.compute_score(info, rec, bad_fingerprints, cluster_stats, now)

            rec["last_score"] = score

            sh = rec.get("score_history", [])
            sh.append(score)
            if len(sh) > MAX_SCORE_HISTORY:
                sh = sh[-MAX_SCORE_HISTORY:]
            rec["score_history"] = sh

            ch = rec.get("cpu_history", [])
            ch.append(info["cpu"])
            if len(ch) > MAX_CPU_HISTORY:
                ch = ch[-MAX_CPU_HISTORY:]
            rec["cpu_history"] = ch

            fps = set(tuple(x) for x in rec.get("fingerprints", []))
            fps.add(tuple(fp))
            rec["fingerprints"] = [list(x) for x in fps]

            BEHAVIOR_DB[key] = rec

        info["score"] = score
        info["threat"] = threat
        info["anomalies"] = anomalies
        info["fingerprint"] = fp
        info["matched_bad_fp"] = matched_bad_fp
        info["momentum"] = momentum
        info["age_seconds"] = age_seconds
        info["cluster_stats"] = cluster_stats.get(cluster_key, {})

    @staticmethod
    def name_pattern(name):
        pattern = []
        for c in name:
            if c.isalpha():
                pattern.append("A")
            elif c.isdigit():
                pattern.append("9")
            else:
                pattern.append("_")
        return "".join(pattern)

    def mark_network_activity(self, pid_to_key):
        with BEHAVIOR_LOCK:
            for key in pid_to_key.values():
                rec = BEHAVIOR_DB.get(key)
                if not rec:
                    continue
                if not rec.get("has_network", False):
                    rec["has_network"] = True
                    BEHAVIOR_DB[key] = rec

    def collect_bad_fingerprints(self):
        bad_fps = []
        with BEHAVIOR_LOCK:
            for key, rec in BEHAVIOR_DB.items():
                label = rec.get("user_label")
                last_score = rec.get("last_score", 0)
                if label in ("block", "radio") or last_score >= 80:
                    for fp in rec.get("fingerprints", []):
                        if fp:
                            bad_fps.append(fp)
        return bad_fps

    def compute_cluster_stats(self, now):
        clusters = {}
        with BEHAVIOR_LOCK:
            for key, rec in BEHAVIOR_DB.items():
                ckey = rec.get("cluster_key")
                if not ckey:
                    continue
                c = clusters.setdefault(ckey, {
                    "members": 0,
                    "bad_count": 0,
                    "score_sum": 0.0,
                    "score_count": 0,
                    "recent_members": 0,
                })
                c["members"] += 1
                score = rec.get("last_score", 0)
                c["score_sum"] += score
                c["score_count"] += 1
                label = rec.get("user_label")
                if label in ("block", "radio") or score >= 80:
                    c["bad_count"] += 1
                if now - rec.get("last_seen", now) < 60:
                    c["recent_members"] += 1

        for ckey, c in clusters.items():
            if c["score_count"] > 0:
                c["avg_score"] = c["score_sum"] / c["score_count"]
            else:
                c["avg_score"] = 0.0

        return clusters

    # ---------- TEMPORAL ANOMALIES / ADAPTIVE THRESHOLD ----------

    def update_stats_and_threshold(self, rows):
        ts = time.time()
        proc_count = len(rows)
        high_count = sum(1 for r in rows if r["threat"] == "HIGH")
        net_count = 0
        scores = [r["score"] for r in rows]
        avg_score = statistics.mean(scores) if scores else 0.0

        with STATS_LOCK:
            hist = STATS_DB.get("history", [])
            hist.append({
                "ts": ts,
                "proc_count": proc_count,
                "high_count": high_count,
                "net_count": net_count,
                "avg_score": avg_score,
            })
            if len(hist) > 200:
                hist = hist[-200:]
            STATS_DB["history"] = hist

        save_stats()

        with STATS_LOCK:
            hist = STATS_DB.get("history", [])
            if len(hist) < 20:
                self.dynamic_lockdown_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD
                self.threshold_label.config(text=f"Threshold: {int(self.dynamic_lockdown_threshold)}")
                return

            high_vals = [h["high_count"] for h in hist]
            avg_vals = [h["avg_score"] for h in hist]

            high_avg = statistics.mean(high_vals)
            high_std = statistics.pstdev(high_vals) if len(high_vals) > 1 else 0.0

            current_high = high_count
            hostile_factor = 0.0
            if high_std > 0 and current_high > high_avg + 2 * high_std:
                hostile_factor += 0.3
            elif current_high > high_avg + high_std:
                hostile_factor += 0.15

            avg_avg = statistics.mean(avg_vals)
            avg_std = statistics.pstdev(avg_vals) if len(avg_vals) > 1 else 0.0
            current_avg = avg_score
            if avg_std > 0 and current_avg > avg_avg + 2 * avg_std:
                hostile_factor += 0.3
            elif current_avg > avg_avg + avg_std:
                hostile_factor += 0.15

            hostile_factor = min(1.0, hostile_factor)
            new_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD - hostile_factor * 20
            new_threshold = max(MIN_LOCKDOWN_THRESHOLD, min(MAX_LOCKDOWN_THRESHOLD, new_threshold))
            self.dynamic_lockdown_threshold = new_threshold

        self.threshold_label.config(text=f"Threshold: {int(self.dynamic_lockdown_threshold)}")

        if hostile_factor >= 0.5:
            log_event(
                f"GLOBAL ALERT: Hostile pattern detected. "
                f"high_count={high_count}, avg_score={avg_score:.1f}, "
                f"threshold={self.dynamic_lockdown_threshold:.1f}"
            )

    # ---------- WHAT-IF DASHBOARD ----------

    def update_whatif_dashboard(self, rows):
        def count_kills(threshold):
            c = 0
            for r in rows:
                status = r["status"]
                path = r["path"]
                score = r["score"]
                if status in ("ALLOW", "TRUSTED"):
                    continue
                np = norm_path(path) if path and not path.startswith("<no-path:") else None
                if np and np in ALLOWLIST:
                    continue
                if score >= threshold:
                    c += 1
            return c

        t50 = count_kills(50)
        t60 = count_kills(60)
        t70 = count_kills(70)

        self.whatif_50.config(text=f"T50: {t50}")
        self.whatif_60.config(text=f"T60: {t60}")
        self.whatif_70.config(text=f"T70: {t70}")

    # ---------- MONITOR LOOP ----------

    def monitor_loop(self):
        while self.running:
            updated = False
            rows = []
            net_rows = []
            proc_cache = {}
            pid_to_key = {}

            now = time.time()
            bad_fps = self.collect_bad_fingerprints()
            cluster_stats = self.compute_cluster_stats(now)

            for proc in psutil.process_iter():
                try:
                    info = self.brain.classify(proc)
                    if not info:
                        continue

                    self.update_behavior(proc, info, bad_fps, cluster_stats, now)

                    key = self._key_for_behavior(info["path"], info["name_l"])
                    pid_to_key[info["pid"]] = key

                    proc_cache[proc.pid] = info["name"]

                    if self.auto_update_lists(info):
                        updated = True

                    self.enforce_lockdown(info)

                    rows.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            try:
                for c in psutil.net_connections(kind="inet"):
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""
                    status = c.status
                    pid = c.pid or 0
                    name = proc_cache.get(pid, "")
                    net_rows.append({
                        "laddr": laddr,
                        "raddr": raddr,
                        "status": status,
                        "pid": pid,
                        "name": name,
                    })
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass

            self.mark_network_activity(pid_to_key)
            save_behavior()

            self.update_stats_and_threshold(rows)

            self.root.after(0, self.update_ui, rows, net_rows, updated)
            time.sleep(CHECK_INTERVAL)

    def update_ui(self, rows, net_rows, updated):
        self.current_rows = rows
        self.current_net_rows = net_rows
        self.refresh_process_view()
        self.refresh_network_view()
        self.update_whatif_dashboard(rows)
        if updated:
            self.refresh_lists_ui()

    def refresh_process_view(self):
        rows = list(self.current_rows)

        filt = self.filter_var.get()
        if filt != "ALL":
            rows = [r for r in rows if r["threat"] == filt]

        def sort_key(r):
            val = r.get(self.sort_column)
            if self.sort_column in ("pid", "cpu", "score"):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0
            return str(val)

        rows = sorted(rows, key=sort_key, reverse=self.sort_reverse)

        self.tree.delete(*self.tree.get_children())
        for i in rows:
            tag = i["threat"] if i["threat"] in ("HIGH", "MEDIUM", "LOW") else "UNKNOWN"
            self.tree.insert(
                "",
                "end",
                values=(
                    i["name"],
                    i["pid"],
                    f"{i['cpu']:.1f}",
                    i["status"],
                    i["threat"],
                    i["score"],
                    i["path"],
                ),
                tags=(tag,),
            )

    def refresh_network_view(self):
        rows = list(self.current_net_rows)
        self.net_tree.delete(*self.net_tree.get_children())
        for n in rows:
            self.net_tree.insert(
                "",
                "end",
                values=(
                    n["laddr"],
                    n["raddr"],
                    n["status"],
                    n["pid"],
                    n["name"],
                ),
            )

    # ---------- WATCHDOG ----------

    def watchdog_loop(self):
        while self.running:
            time.sleep(5.0)

            if not os.path.exists(LISTS_FILE):
                log_event("WATCHDOG: Lists file missing, recreating.")
                save_lists()

            with LIST_LOCK:
                try:
                    if os.path.exists(LISTS_FILE):
                        with open(LISTS_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Lists file corrupted, resetting.")
                    try:
                        os.remove(LISTS_FILE)
                    except OSError:
                        pass
                    save_lists()

            with BEHAVIOR_LOCK:
                try:
                    if os.path.exists(BEHAVIOR_FILE):
                        with open(BEHAVIOR_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Behavior DB corrupted, resetting.")
                    try:
                        os.remove(BEHAVIOR_FILE)
                    except OSError:
                        pass
                    save_behavior()

            with STATS_LOCK:
                try:
                    if os.path.exists(STATS_FILE):
                        with open(STATS_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Stats DB corrupted, resetting.")
                    try:
                        os.remove(STATS_FILE)
                    except OSError:
                        pass
                    save_stats()

            if self.monitor_thread and not self.monitor_thread.is_alive():
                log_event("WATCHDOG: Monitor thread not alive, restarting.")
                self.start_monitor_thread()


# ---------- MAIN ----------

def main():
    os.makedirs(APPDATA, exist_ok=True)
    print("Using lists file:", os.path.abspath(LISTS_FILE))
    print("Using log file:", os.path.abspath(LOG_FILE))
    print("Using behavior file:", os.path.abspath(BEHAVIOR_FILE))
    print("Using stats file:", os.path.abspath(STATS_FILE))
    load_lists()
    load_behavior()
    load_stats()
    root = tk.Tk()
    app = SentinelGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(app, root))
    root.mainloop()


def on_close(app, root):
    app.running = False
    log_event("Sentinel shutting down.")
    root.destroy()


if __name__ == "__main__":
    main()

