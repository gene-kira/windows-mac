import sys
import os
import subprocess
import threading
import time
import datetime
import json
import tkinter as tk
from tkinter import ttk, messagebox

# ---------- ADMIN ELEVATION ----------
if sys.platform == "win32":
    import ctypes

    def ensure_admin():
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = False

        if not is_admin:
            print("Time Boss needs admin privileges. Requesting elevation...")
            script = sys.executable
            params = " ".join([f'"{arg}"' for arg in sys.argv])
            try:
                ctypes.windll.shell32.ShellExecuteW(None, "runas", script, params, None, 1)
                sys.exit(0)
            except Exception as e:
                print("Failed to elevate:", e)
                sys.exit(1)

    ensure_admin()

# ---------- SIMPLE AUTOLOADER FOR PSUTIL ----------
try:
    import psutil
except ImportError:
    answer = input("psutil not found. Install it now? [y/N]: ").strip().lower()
    if answer == "y":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    else:
        print("psutil is required. Exiting.")
        sys.exit(1)

# ---------- LOG/SETTINGS PATH ----------
APPDATA_DIR = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
LOG_FILE = os.path.join(APPDATA_DIR, "time_boss_log.txt")
SETTINGS_FILE = os.path.join(APPDATA_DIR, "time_boss_settings.json")

# ---------- DEFAULT SETTINGS ----------
DEFAULT_SETTINGS = {
    "drift_warning": 1.0,
    "drift_severe": 5.0,
    "drift_critical": 10.0,
    "monitor_interval": 2.0,
    "sync_check_interval": 60.0,
    "clock_fight_window": 60.0,
    "clock_fight_threshold": 3,
    "auto_correct": False,
    "auto_kill": False
}

# ---------- SETTINGS MANAGEMENT ----------
def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = DEFAULT_SETTINGS.copy()
        s.update(data)
        return s
    except Exception:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

# ---------- LOGGING ----------
def log_event(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except PermissionError:
        print(f"Permission denied writing log: {LOG_FILE}")

# ---------- TIME ENGINE ----------
class TimeEngine:
    def __init__(self, settings):
        self.settings = settings
        self.last_wall = datetime.datetime.now()
        self.last_mono = time.monotonic()
        self.drift_history = []
        self.lock = threading.Lock()

    def sample(self):
        now_wall = datetime.datetime.now()
        now_mono = time.monotonic()
        expected_wall = self.last_wall + datetime.timedelta(seconds=(now_mono - self.last_mono))
        drift = (now_wall - expected_wall).total_seconds()
        self.last_wall = now_wall
        self.last_mono = now_mono

        with self.lock:
            self.drift_history.append((now_wall, drift))
            cutoff = now_wall - datetime.timedelta(minutes=10)
            self.drift_history = [d for d in self.drift_history if d[0] >= cutoff]
        return drift

    def classify_drift(self, drift):
        ad = abs(drift)
        if ad >= self.settings["drift_critical"]:
            return "CRITICAL"
        elif ad >= self.settings["drift_severe"]:
            return "SEVERE"
        elif ad >= self.settings["drift_warning"]:
            return "WARNING"
        else:
            return "NORMAL"

    def detect_clock_fight(self):
        with self.lock:
            if len(self.drift_history) < 5:
                return False
            now = self.drift_history[-1][0]
            window = self.settings["clock_fight_window"]
            recent = [d for d in self.drift_history if (now - d[0]).total_seconds() <= window]

        if len(recent) < 5:
            return False

        signs = []
        for _, drift in recent:
            if drift > 0.5:
                signs.append(1)
            elif drift < -0.5:
                signs.append(-1)
            else:
                signs.append(0)

        flips = 0
        last = signs[0]
        for s in signs[1:]:
            if s != 0 and last != 0 and s != last:
                flips += 1
            if s != 0:
                last = s
        return flips >= self.settings["clock_fight_threshold"]

# ---------- SYNC ENGINE ----------
class SyncEngine:
    def __init__(self):
        self.status = "Unknown"
        self.last_sync = "Unknown"
        self.server = "Unknown"
        self.last_check_time = None
        self.lock = threading.Lock()

    def _run_cmd(self, cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
            return out
        except Exception as e:
            return f"ERROR: {e}"

    def check_status(self):
        status_out = self._run_cmd("w32tm /query /status")
        config_out = self._run_cmd("w32tm /query /configuration")
        status = "Unknown"
        last_sync = "Unknown"
        server = "Unknown"

        if "ERROR" in status_out:
            status = "ERROR"
        else:
            for line in status_out.splitlines():
                line = line.strip()
                if line.startswith("Stratum:"):
                    status = "OK"
                if line.startswith("Last Successful Sync Time:"):
                    last_sync = line.split(":", 1)[1].strip()

        if "ERROR" not in config_out:
            for line in config_out.splitlines():
                line = line.strip()
                if "NtpServer" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        server = parts[1].strip()

        with self.lock:
            self.status = status
            self.last_sync = last_sync
            self.server = server
            self.last_check_time = datetime.datetime.now()

    def force_sync(self):
        out = self._run_cmd("w32tm /resync")
        log_event(f"FORCE SYNC: {out.strip()}")
        return out

# ---------- HOUND DOG ENGINE ----------
class HoundDog:
    WHITELIST = [
        "System", "Idle", "explorer.exe", "svchost.exe", "csrss.exe",
        "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe", "smss.exe"
    ]

    def __init__(self):
        self.snapshots = []
        self.lock = threading.Lock()

    def take_snapshot(self, label="normal"):
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "exe"]):
            try:
                info = p.info
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": info["cpu_percent"],
                    "exe": info.get("exe") or "",
                })
            except Exception:
                continue
        ts = datetime.datetime.now()
        with self.lock:
            self.snapshots.append((ts, label, procs))
            self.snapshots = self.snapshots[-10:]

    def find_suspects(self):
        with self.lock:
            if len(self.snapshots) < 2:
                return []

            _, _, before_procs = self.snapshots[-2]
            _, _, after_procs = self.snapshots[-1]

        before_map = {(p["pid"], p["exe"]): p for p in before_procs}
        after_map = {(p["pid"], p["exe"]): p for p in after_procs}

        suspects = []

        for key, ap in after_map.items():
            bp = before_map.get(key)
            score = 0
            reasons = []

            if bp is None:
                score += 3
                reasons.append("New process")
            else:
                if ap["cpu"] - bp["cpu"] > 10:
                    score += 2
                    reasons.append("CPU spike")

            if ap["name"] in self.WHITELIST:
                score = 0

            if score > 0:
                suspects.append({
                    "pid": ap["pid"],
                    "name": ap["name"],
                    "exe": ap["exe"],
                    "score": score,
                    "reasons": ", ".join(reasons),
                })

        suspects.sort(key=lambda x: x["score"], reverse=True)
        return suspects

    def auto_kill(self, suspects):
        for s in suspects:
            try:
                p = psutil.Process(s["pid"])
                p.terminate()
                log_event(f"AUTO-KILL: Terminated {s['name']} (PID {s['pid']})")
            except Exception as e:
                log_event(f"AUTO-KILL FAILED: {s['name']} (PID {s['pid']}): {e}")

# ---------- GUI ----------
class TimeBossGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Boss")
        self.settings = load_settings()
        self.engine = TimeEngine(self.settings)
        self.sync_engine = SyncEngine()
        self.hound = HoundDog()
        self.running = True

        self._build_ui()
        threading.Thread(target=self.monitor_loop, daemon=True).start()
        threading.Thread(target=self.sync_loop, daemon=True).start()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # --- Live Clock Tab ---
        live_tab = ttk.Frame(notebook)
        notebook.add(live_tab, text="Live Clock")
        lf = ttk.LabelFrame(live_tab, text="Current Status")
        lf.pack(fill="both", expand=True, padx=10, pady=10)

        self.time_var = tk.StringVar(value="System Time: --")
        self.mono_var = tk.StringVar(value="Monotonic: --")
        self.drift_var = tk.StringVar(value="Drift: --")
        self.severity_var = tk.StringVar(value="Severity: NORMAL")
        self.clock_fight_var = tk.StringVar(value="Clock Fight: No")

        ttk.Label(lf, textvariable=self.time_var).pack(anchor="w", pady=2)
        ttk.Label(lf, textvariable=self.mono_var).pack(anchor="w", pady=2)
        ttk.Label(lf, textvariable=self.drift_var).pack(anchor="w", pady=2)
        ttk.Label(lf, textvariable=self.severity_var).pack(anchor="w", pady=2)
        ttk.Label(lf, textvariable=self.clock_fight_var).pack(anchor="w", pady=2)

        self.severity_bar = ttk.Progressbar(lf, mode="determinate", maximum=100)
        self.severity_bar.pack(fill="x", pady=5)

        # --- Drift History Tab ---
        history_tab = ttk.Frame(notebook)
        notebook.add(history_tab, text="Drift History")
        hf = ttk.LabelFrame(history_tab, text="Recent Drift Events")
        hf.pack(fill="both", expand=True, padx=10, pady=10)
        self.history_text = tk.Text(hf, height=15, wrap="none")
        self.history_text.pack(fill="both", expand=True)

        # --- Sync Tab ---
        sync_tab = ttk.Frame(notebook)
        notebook.add(sync_tab, text="Time Sync")
        sf = ttk.LabelFrame(sync_tab, text="Windows Time Service")
        sf.pack(fill="both", expand=True, padx=10, pady=10)

        self.sync_status_var = tk.StringVar(value="Status: Unknown")
        self.sync_server_var = tk.StringVar(value="Server: Unknown")
        self.sync_last_var = tk.StringVar(value="Last Sync: Unknown")
        self.sync_checked_var = tk.StringVar(value="Last Checked: Never")

        ttk.Label(sf, textvariable=self.sync_status_var).pack(anchor="w", pady=2)
        ttk.Label(sf, textvariable=self.sync_server_var).pack(anchor="w", pady=2)
        ttk.Label(sf, textvariable=self.sync_last_var).pack(anchor="w", pady=2)
        ttk.Label(sf, textvariable=self.sync_checked_var).pack(anchor="w", pady=2)

        ttk.Button(sf, text="Force Sync Now", command=self.force_sync).pack(anchor="w", pady=5)

        # --- Hound Dog Tab ---
        hound_tab = ttk.Frame(notebook)
        notebook.add(hound_tab, text="Hound Dog")
        hdf = ttk.LabelFrame(hound_tab, text="Suspected Processes Around Drift")
        hdf.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("name", "pid", "score", "reasons", "exe")
        self.hound_tree = ttk.Treeview(hdf, columns=columns, show="headings", height=10)
        headers = ["Name", "PID", "Score", "Reasons", "Path"]
        widths = [140, 60, 60, 200, 400]
        for col, text, w in zip(columns, headers, widths):
            self.hound_tree.heading(col, text=text)
            self.hound_tree.column(col, width=w)
        self.hound_tree.pack(fill="both", expand=True)

        # --- Settings Tab ---
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")

        stf = ttk.LabelFrame(settings_tab, text="Drift Thresholds (seconds)")
        stf.pack(fill="x", padx=10, pady=10)

        self.warn_var = tk.DoubleVar(value=self.settings["drift_warning"])
        self.severe_var = tk.DoubleVar(value=self.settings["drift_severe"])
        self.critical_var = tk.DoubleVar(value=self.settings["drift_critical"])

        row = 0
        ttk.Label(stf, text="Warning:").grid(row=row, column=0, sticky="w")
        ttk.Entry(stf, textvariable=self.warn_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(stf, text="Severe:").grid(row=row, column=0, sticky="w")
        ttk.Entry(stf, textvariable=self.severe_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(stf, text="Critical:").grid(row=row, column=0, sticky="w")
        ttk.Entry(stf, textvariable=self.critical_var, width=10).grid(row=row, column=1, sticky="w")

        optf = ttk.LabelFrame(settings_tab, text="Options")
        optf.pack(fill="x", padx=10, pady=10)
        self.auto_correct_var = tk.BooleanVar(value=self.settings["auto_correct"])
        self.auto_kill_var = tk.BooleanVar(value=self.settings.get("auto_kill", False))
        ttk.Checkbutton(optf, text="Auto-correct (force sync on critical drift)", variable=self.auto_correct_var).pack(anchor="w")
        ttk.Checkbutton(optf, text="Auto-Kill Suspicious Processes", variable=self.auto_kill_var).pack(anchor="w")

        ttk.Button(settings_tab, text="Save Settings", command=self.save_settings).pack(anchor="e", padx=10, pady=10)
        ttk.Button(self.root, text="Quit", command=self.on_quit).pack(anchor="e", padx=10, pady=5)

    # ---------- GUI HELPERS ----------
    def append_history(self, msg):
        self.history_text.insert("end", msg + "\n")
        self.history_text.see("end")

    def update_live_view(self, drift, severity, clock_fight):
        now = datetime.datetime.now()
        self.time_var.set(f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.mono_var.set(f"Monotonic: {self.engine.last_mono:.3f} s")
        self.drift_var.set(f"Drift: {drift:+.3f} s")
        self.severity_var.set(f"Severity: {severity}")
        self.clock_fight_var.set(f"Clock Fight: {'YES' if clock_fight else 'No'}")
        ad = min(abs(drift), self.settings["drift_critical"] * 2)
        pct = (ad / (self.settings["drift_critical"] * 2)) * 100.0
        self.severity_bar["value"] = pct

    def update_sync_view(self):
        with self.sync_engine.lock:
            status = self.sync_engine.status
            last_sync = self.sync_engine.last_sync
            server = self.sync_engine.server
            checked = self.sync_engine.last_check_time
        self.sync_status_var.set(f"Status: {status}")
        self.sync_server_var.set(f"Server: {server}")
        self.sync_last_var.set(f"Last Sync: {last_sync}")
        self.sync_checked_var.set(f"Last Checked: {checked.strftime('%Y-%m-%d %H:%M:%S')}" if checked else "Last Checked: Never")

    # ---------- NEW: Highlight most suspicious process ----------
    def update_hound_view(self, suspects):
        self.hound_tree.delete(*self.hound_tree.get_children())
        if not suspects:
            return
        max_score = max(s["score"] for s in suspects)
        for s in suspects:
            tags = ()
            if s["score"] == max_score:
                tags = ("most_suspicious",)
            self.hound_tree.insert(
                "", "end",
                values=(s["name"], s["pid"], s["score"], s["reasons"], s["exe"]),
                tags=tags
            )
        self.hound_tree.tag_configure("most_suspicious", background="#FFCCCC")

    # ---------- BACKGROUND LOOPS ----------
    def monitor_loop(self):
        while self.running:
            self.hound.take_snapshot(label="before")
            drift = self.engine.sample()
            severity = self.engine.classify_drift(drift)
            clock_fight = self.engine.detect_clock_fight()
            msg = f"DRIFT {drift:+.3f} s [{severity}]"
            if clock_fight:
                msg += " CLOCK FIGHT DETECTED"
            log_event(msg)

            suspects = None
            if severity in ("SEVERE", "CRITICAL") or clock_fight:
                self.hound.take_snapshot(label="after")
                suspects = self.hound.find_suspects()
                if suspects:
                    log_event(f"HOUND DOG: {len(suspects)} suspects detected.")
                    if self.auto_kill_var.get():
                        self.hound.auto_kill(suspects)

            def gui_update():
                self.update_live_view(drift, severity, clock_fight)
                self.append_history(msg)
                if suspects is not None:
                    self.update_hound_view(suspects)
            self.root.after(0, gui_update)

            if severity == "CRITICAL" and self.auto_correct_var.get():
                out = self.sync_engine.force_sync()
                log_event("AUTO-CORRECT: Forced sync due to critical drift.")
                def gui_sync_msg():
                    self.append_history("AUTO-CORRECT: Forced sync due to critical drift.")
                    self.append_history(out.strip())
                self.root.after(0, gui_sync_msg)

            time.sleep(self.settings["monitor_interval"])

    def sync_loop(self):
        while self.running:
            self.sync_engine.check_status()
            self.root.after(0, self.update_sync_view)
            time.sleep(self.settings["sync_check_interval"])

    # ---------- ACTIONS ----------
    def force_sync(self):
        out = self.sync_engine.force_sync()
        messagebox.showinfo("Time Boss", out)
        self.update_sync_view()

    def save_settings(self):
        self.settings["drift_warning"] = float(self.warn_var.get())
        self.settings["drift_severe"] = float(self.severe_var.get())
        self.settings["drift_critical"] = float(self.critical_var.get())
        self.settings["auto_correct"] = bool(self.auto_correct_var.get())
        self.settings["auto_kill"] = bool(self.auto_kill_var.get())
        save_settings(self.settings)
        messagebox.showinfo("Time Boss", "Settings saved.")

    def on_quit(self):
        self.running = False
        self.root.destroy()

# ---------- MAIN ----------
def main():
    root = tk.Tk()
    TimeBossGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
