import psutil
import threading
import time
import tkinter as tk
from tkinter import ttk
import os
import json
from datetime import datetime

# ---------- CONFIG ----------

CHECK_INTERVAL = 2.0

APPDATA = os.getenv("APPDATA") or os.getcwd()
LISTS_FILE = os.path.join(APPDATA, "sentinel_lists.json")
LOG_FILE = os.path.join(APPDATA, "sentinel_events.log")
BEHAVIOR_FILE = os.path.join(APPDATA, "sentinel_behavior.json")

ALLOWLIST = set()
BLOCKLIST = set()      # holds paths or names (lowercased)
RADIOACTIVE = set()

# Behavior DB: key -> record
# key is usually path; if no path, falls back to name
BEHAVIOR_DB = {}

WHITELIST_NAMES = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
}

LIST_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
BEHAVIOR_LOCK = threading.Lock()

# Score thresholds
LOCKDOWN_SCORE_THRESHOLD = 70
AUTO_KILL_SCORE_THRESHOLD = 90  # even without lockdown, if we ever want that later


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

    ALLOWLIST = set(d.get("allow", []))
    BLOCKLIST = set(d.get("block", []))
    RADIOACTIVE = set(d.get("radioactive", []))


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


# ---------- BRAIN ----------

class SentinelBrain:
    def classify(self, proc):
        try:
            name = proc.name()
            exe = (proc.exe() or "").replace("\\", "/").lower()
            name_l = name.lower()
            cpu = proc.cpu_percent(interval=None)
            ppid = proc.ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        if not exe:
            exe = f"<no-path:{proc.pid}>"

        # BLOCKLIST matches by path OR name
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

        # Initial threat label; will be refined by scoring engine
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
    def compute_score(info, behavior_record):
        score = 0
        anomalies = []

        path = info["path"]
        name_l = info["name_l"]
        cpu = info["cpu"]
        status = info["status"]

        # 1. Path-based heuristics
        lower_path = path.lower()
        if any(x in lower_path for x in ["/temp/", "\\temp\\", "\\appdata\\", "/appdata/"]):
            score += 20
            anomalies.append("Runs from temp/appdata")

        # 2. Name randomness heuristic
        base_name = os.path.basename(path)
        if not base_name or base_name.startswith("<no-path:"):
            base_name = name_l
        if len(base_name) > 10:
            consonants = sum(1 for c in base_name if c.lower() in "bcdfghjklmnpqrstvwxyz")
            if consonants / max(1, len(base_name)) > 0.6:
                score += 15
                anomalies.append("Random-looking name")

        # 3. CPU behavior
        if cpu > 50:
            score += 15
            anomalies.append(f"High CPU: {cpu:.1f}%")
        if cpu > 80:
            score += 10

        # 4. Behavior history
        if behavior_record:
            launches = behavior_record.get("launch_count", 0)
            max_cpu = behavior_record.get("max_cpu", 0.0)
            has_net = behavior_record.get("has_network", False)
            first_seen = behavior_record.get("first_seen", 0)
            last_seen = behavior_record.get("last_seen", 0)

            lifetime = last_seen - first_seen if last_seen and first_seen else 0

            if launches <= 1 and lifetime < 10:
                score += 10
                anomalies.append("Ephemeral process")

            if has_net:
                score += 25
                anomalies.append("Has network activity")

            if cpu > max_cpu * 2 and max_cpu > 20:
                score += 15
                anomalies.append("CPU spike anomaly")

        # 5. Status-based weighting
        if status == "BLOCKED":
            score += 40
        elif status == "RADIOACTIVE":
            score += 25
        elif status == "SUSPICIOUS":
            score += 10
        elif status in ("TRUSTED", "ALLOW"):
            score -= 20

        # 6. User label influence
        user_label = behavior_record.get("user_label") if behavior_record else None
        if user_label == "block":
            score += 30
        elif user_label == "radio":
            score += 15
        elif user_label == "allow":
            score -= 25

        # Clamp score
        if score < 0:
            score = 0
        if score > 100:
            score = 100

        # Map to threat level
        if score >= 70:
            threat = "HIGH"
        elif score >= 40:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        return score, threat, anomalies


# ---------- GUI ----------

class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel – Predictive Guardian")

        self.brain = SentinelBrain()
        self.running = True
        self.lockdown = tk.BooleanVar(value=False)

        self.sort_column = "score"
        self.sort_reverse = True

        self.monitor_thread = None
        self.watchdog_thread = None

        self.current_rows = []
        self.current_net_rows = []

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
            text="Lockdown Mode (score-based pre-emptive kill)",
            variable=self.lockdown,
            command=self.on_lockdown_toggle,
        )
        self.lockdown_check.pack(side="left", padx=5, pady=2)

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

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # Processes tab
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

        # Network tab
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

    # ---------- LIST OPERATIONS + LEARNING MODULE ----------

    def add_selected_allow(self):
        path = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()
        key = self._key_for_behavior(path, name_l)

        if path:
            ALLOWLIST.add(path)
            RADIOACTIVE.discard(path)
        save_lists()
        log_event(f"Added to ALLOWLIST: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "allow"
            BEHAVIOR_DB[key] = rec
        save_behavior()

        self.refresh_lists_ui()
        self.refresh_process_view()

    def add_selected_block(self):
        path = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()

        if path and not path.startswith("<no-path:"):
            key = path
        else:
            key = name_l

        BLOCKLIST.add(key)
        save_lists()
        log_event(f"Added to BLOCKLIST: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "block"
            BEHAVIOR_DB[key] = rec
        save_behavior()

        # Immediate kill of the selected process
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
        path = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()
        key = self._key_for_behavior(path, name_l)

        if path:
            RADIOACTIVE.add(path)
            ALLOWLIST.discard(path)
        save_lists()
        log_event(f"Added to RADIOACTIVE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "radio"
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

    # ---------- AUTO-FILL LOGIC ----------

    def auto_update_lists(self, info):
        p = info["path"]
        if not p:
            return False

        changed = False

        if info["status"] == "TRUSTED" and p not in ALLOWLIST:
            ALLOWLIST.add(p)
            RADIOACTIVE.discard(p)
            changed = True
            log_event(f"AUTO: Trusted path added to ALLOWLIST: {p}")

        elif info["status"] == "SUSPICIOUS" and p not in RADIOACTIVE:
            try:
                proc = psutil.Process(info["pid"])
                if proc.create_time() > time.time() - 3:
                    return changed
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

            RADIOACTIVE.add(p)
            changed = True
            log_event(f"AUTO: Suspicious path added to RADIOACTIVE: {p}")

        if changed:
            save_lists()
        return changed

    # ---------- LOCKDOWN MODE / PRE-EMPTIVE ENGINE ----------

    def on_lockdown_toggle(self):
        state = "ON" if self.lockdown.get() else "OFF"
        log_event(f"Lockdown mode toggled: {state}")

    def enforce_lockdown(self, info):
        if not self.lockdown.get():
            return

        status = info["status"]
        path = info["path"]
        pid = info["pid"]
        score = info.get("score", 0)

        if status in ("ALLOW", "TRUSTED"):
            return
        if path in ALLOWLIST:
            return

        if score < LOCKDOWN_SCORE_THRESHOLD and status not in ("BLOCKED", "RADIOACTIVE"):
            return

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            log_event(
                f"LOCKDOWN: Terminated PID {pid} ({info['name']}) "
                f"path={path} status={status} score={score}"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            log_event(f"LOCKDOWN: Failed to terminate PID {pid}: {e}")

    # ---------- BEHAVIOR TRACKING / ANOMALY DETECTION ----------

    def update_behavior(self, proc, info):
        path = info["path"]
        name_l = info["name_l"]
        key = self._key_for_behavior(path, name_l)
        now = time.time()

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
            })

            # Update basic behavior
            rec["last_seen"] = now
            rec["launch_count"] = rec.get("launch_count", 0) + 1
            rec["total_cpu"] = rec.get("total_cpu", 0.0) + info["cpu"]
            rec["max_cpu"] = max(rec.get("max_cpu", 0.0), info["cpu"])

            parent_pids = set(rec.get("parent_pids", []))
            parent_pids.add(info["ppid"])
            rec["parent_pids"] = list(parent_pids)

            # Network flag will be updated later when we process connections
            BEHAVIOR_DB[key] = rec

        # Compute score and anomalies
        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            score, threat, anomalies = self.brain.compute_score(info, rec)
            rec["last_score"] = score
            BEHAVIOR_DB[key] = rec

        info["score"] = score
        info["threat"] = threat
        info["anomalies"] = anomalies

    def mark_network_activity(self, pid_to_key):
        with BEHAVIOR_LOCK:
            for key in pid_to_key.values():
                rec = BEHAVIOR_DB.get(key)
                if not rec:
                    continue
                if not rec.get("has_network", False):
                    rec["has_network"] = True
                    BEHAVIOR_DB[key] = rec

    # ---------- MONITOR LOOP ----------

    def monitor_loop(self):
        while self.running:
            updated = False
            rows = []
            net_rows = []
            proc_cache = {}
            pid_to_key = {}

            for proc in psutil.process_iter():
                try:
                    info = self.brain.classify(proc)
                    if not info:
                        continue

                    # Behavior tracking + scoring
                    self.update_behavior(proc, info)

                    key = self._key_for_behavior(info["path"], info["name_l"])
                    pid_to_key[info["pid"]] = key

                    proc_cache[proc.pid] = info["name"]

                    if self.auto_update_lists(info):
                        updated = True

                    self.enforce_lockdown(info)

                    rows.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Network connections
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

            # Mark behavior records that have network activity
            self.mark_network_activity(pid_to_key)
            save_behavior()

            self.root.after(0, self.update_ui, rows, net_rows, updated)
            time.sleep(CHECK_INTERVAL)

    def update_ui(self, rows, net_rows, updated):
        self.current_rows = rows
        self.current_net_rows = net_rows
        self.refresh_process_view()
        self.refresh_network_view()
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

            # Lists file health
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

            # Behavior DB health
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

            # Monitor thread health
            if self.monitor_thread and not self.monitor_thread.is_alive():
                log_event("WATCHDOG: Monitor thread not alive, restarting.")
                self.start_monitor_thread()


# ---------- MAIN ----------

def main():
    os.makedirs(APPDATA, exist_ok=True)
    print("Using lists file:", os.path.abspath(LISTS_FILE))
    print("Using log file:", os.path.abspath(LOG_FILE))
    print("Using behavior file:", os.path.abspath(BEHAVIOR_FILE))
    load_lists()
    load_behavior()
    root = tk.Tk()
    app = SentinelGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(app, root))
    root.mainloop()


def on_close(app, root):
    app.running = False    # signal threads to stop
    log_event("Sentinel shutting down.")
    root.destroy()


if __name__ == "__main__":
    main()

