"""
HANNIBAL BACKBONE GUARDIAN

Single-file organism:
- Auto-load dependencies
- Persistent memory across reboots
- System + software inventory
- Risk + trend analysis
- Burst guardian: boosts active loading app with all cores
- Per-app learning: burst profiles
- Hannibal-style verdicts & opportunities
- Tkinter GUI for visibility
"""

import importlib
import subprocess
import sys
import os
import json
import threading
import time

# ------------------ AUTO-LOADER ------------------ #

REQUIRED_LIBS = [
    "psutil",
    "platform",
]

def ensure_libs():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[AUTO-LOADER] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

import psutil
import platform
import uuid
from pathlib import Path

# Try Windows-specific imports for foreground detection
try:
    import win32gui
    import win32process
except ImportError:
    win32gui = None
    win32process = None

import tkinter as tk
from tkinter import ttk

# ------------------ PERSISTENT MEMORY ------------------ #

APP_DIR = Path(os.path.expanduser("~")) / ".hannibal_guardian"
APP_DIR.mkdir(exist_ok=True)

STATE_FILE = APP_DIR / "guardian_state.json"

DEFAULT_STATE = {
    "schema_version": 1,
    "machine_id": None,
    "runs": 0,
    "risk_history": [],
    "burst_profiles": {},      # per-process stats
    "app_policies": {},        # per app name policy overrides (future use)
}

def load_state():
    if not STATE_FILE.exists():
        state = DEFAULT_STATE.copy()
        state["machine_id"] = str(uuid.uuid4())
        return state

    try:
        with STATE_FILE.open("r") as f:
            data = json.load(f)
    except Exception:
        data = DEFAULT_STATE.copy()
        data["machine_id"] = str(uuid.uuid4())
        return data

    if "schema_version" not in data or data["schema_version"] != DEFAULT_STATE["schema_version"]:
        upgraded = DEFAULT_STATE.copy()
        for k in upgraded.keys():
            if k in data:
                upgraded[k] = data[k]
        data = upgraded

    if not data.get("machine_id"):
        data["machine_id"] = str(uuid.uuid4())

    return data

def save_state(state):
    try:
        with STATE_FILE.open("w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[STATE] Failed to save state: {e}")

STATE = load_state()
STATE["runs"] = STATE.get("runs", 0) + 1
save_state(STATE)

# ------------------ MACHINE FINGERPRINT ------------------ #

def get_machine_fingerprint():
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
    }

MACHINE_FINGERPRINT = get_machine_fingerprint()

# ------------------ SYSTEM & SOFTWARE INVENTORY ------------------ #

def get_system_inventory():
    info = {
        "os": MACHINE_FINGERPRINT["os"],
        "machine": MACHINE_FINGERPRINT["machine"],
        "processor": MACHINE_FINGERPRINT["processor"],
        "cpu_count_logical": MACHINE_FINGERPRINT["cpu_count_logical"],
        "cpu_count_physical": MACHINE_FINGERPRINT["cpu_count_physical"],
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "boot_time": psutil.boot_time(),
    }
    return info

def get_software_inventory():
    try:
        import pkg_resources
        installed = sorted([str(d) for d in pkg_resources.working_set])
    except Exception:
        installed = []
    return {
        "python_version": sys.version.split()[0],
        "packages": installed
    }

# ------------------ RISK + TREND + HANNIBAL VERDICT ------------------ #

def analyze_weak_points(sys_inv, soft_inv):
    weak = []

    if sys_inv["ram_total_gb"] < 4:
        weak.append("Low RAM (< 4GB): memory pressure risk.")

    if sys_inv["cpu_count_physical"] is not None and sys_inv["cpu_count_physical"] < 4:
        weak.append("Low physical core count: parallel load may choke.")

    if "pip" not in " ".join(soft_inv["packages"]).lower():
        weak.append("pip or tooling not clearly visible: dependency management may be fragile.")

    if len(soft_inv["packages"]) < 5:
        weak.append("Very few Python packages: environment underpowered for advanced analysis.")

    return weak

def analyze_opportunities(sys_inv, soft_inv):
    ops = []

    if sys_inv["ram_total_gb"] >= 8:
        ops.append("High enough RAM: safe to run heavier predictive models and caching.")
    else:
        ops.append("Optimize for lightweight agents and focused monitoring.")

    if len(soft_inv["packages"]) > 20:
        ops.append("Rich Python ecosystem: leverage specialized libraries for ML, visualization, and security.")
    else:
        ops.append("Curate a minimal, hardened package set for reliability.")

    ops.append("Schedule periodic scans and trend analysis instead of one-off checks.")

    return ops

def hannibal_judgment(weak_points, opportunities):
    if not weak_points and opportunities:
        return "Calm. Environment is stable. Exploit strengths ruthlessly."
    if len(weak_points) <= 2:
        return "Manageable vulnerabilities. Exploit opportunities while reinforcing weak links."
    if len(weak_points) <= 5:
        return "Too many exposed ribs. Prioritize structural reinforcement before ambitious moves."
    return "This house is made of glass. Do not play with stones; rebuild the foundation."

def risk_score(weak_points, sys_inv):
    base = len(weak_points) * 10
    if sys_inv["ram_total_gb"] < 4:
        base += 10
    if sys_inv["cpu_count_physical"] and sys_inv["cpu_count_physical"] < 4:
        base += 5
    return min(100, base)

def update_risk_history(score):
    history = STATE.get("risk_history", [])
    history.append({"score": score, "run": STATE["runs"]})
    history = history[-200:]
    STATE["risk_history"] = history
    save_state(STATE)
    return history

def predict_trend(history):
    if len(history) < 5:
        return "Insufficient data. Observing your breathing."
    scores = [h["score"] for h in history]
    first, last = scores[0], scores[-1]
    delta = last - first
    if delta > 10:
        return "Risk rising across reboots. Adaptations are failing."
    if delta < -10:
        return "Risk falling across reboots. Reinforcements are working."
    return "Risk stable with oscillations. This organism lives in tension."

def what_if(sys_inv, scenario):
    sys_clone = dict(sys_inv)
    sys_clone.update(scenario)
    soft_inv = get_software_inventory()
    weak = analyze_weak_points(sys_clone, soft_inv)
    score = risk_score(weak, sys_clone)
    verdict = hannibal_judgment(weak, analyze_opportunities(sys_clone, soft_inv))
    return {
        "hypothetical_sys": sys_clone,
        "weak_points": weak,
        "risk_score": score,
        "verdict": verdict
    }

def full_scan():
    sys_inv = get_system_inventory()
    soft_inv = get_software_inventory()
    weak = analyze_weak_points(sys_inv, soft_inv)
    ops = analyze_opportunities(sys_inv, soft_inv)
    score = risk_score(weak, sys_inv)
    history = update_risk_history(score)
    trend = predict_trend(history)
    verdict = hannibal_judgment(weak, ops)
    return {
        "system": sys_inv,
        "software": soft_inv,
        "weak_points": weak,
        "opportunities": ops,
        "risk_score": score,
        "trend": trend,
        "verdict": verdict,
        "runs": STATE["runs"]
    }

# ------------------ BURST GUARDIAN (BOOST ACTIVE LOADING APP) ------------------ #

CPU_HIGH_THRESHOLD = 40      # percent, considered “loading”
CPU_LOW_THRESHOLD = 10       # below this, loading done
SAMPLE_INTERVAL = 0.5        # seconds
STABLE_CYCLES_DONE = 5       # how many low cycles mark “done”
BOOST_DURATION_MAX = 30      # default hard cap seconds for a boost phase

class BurstGuardian:
    def __init__(self):
        self.current_boost_pid = None
        self.boost_start_time = None
        self.low_cycles = 0
        self.original_states = {}
        self.current_proc_name = "unknown"
        self.current_max_boost = BOOST_DURATION_MAX

    # ---- foreground detection ---- #
    def get_foreground_pid(self):
        if win32gui is None or win32process is None:
            return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            tid, pid = win32process.GetWindowThreadProcessId(hwnd)
            return pid
        except Exception:
            return None

    # ---- state capture / restore ---- #
    def capture_state(self, proc):
        try:
            if proc.pid in self.original_states:
                return
            nice = proc.nice()
            affinity = proc.cpu_affinity()
            self.original_states[proc.pid] = (nice, affinity)
        except Exception:
            pass

    def restore_state(self, pid):
        state = self.original_states.pop(pid, None)
        if not state:
            return
        try:
            proc = psutil.Process(pid)
            nice, affinity = state
            proc.nice(nice)
            proc.cpu_affinity(affinity)
        except Exception:
            pass

    # ---- learning: record burst outcome ---- #
    def record_burst_outcome(self, proc_name, duration, reason):
        name = proc_name.lower()
        profiles = STATE.get("burst_profiles", {})
        prof = profiles.get(name, {
            "total_bursts": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "last_reason": "",
        })
        prof["total_bursts"] += 1
        prof["total_time"] += duration
        prof["average_time"] = prof["total_time"] / max(1, prof["total_bursts"])
        prof["last_reason"] = reason
        profiles[name] = prof
        STATE["burst_profiles"] = profiles
        save_state(STATE)

    # ---- boost & end ---- #
    def boost_process(self, proc):
        self.capture_state(proc)
        name = proc.name().lower()
        policies = STATE.get("app_policies", {})
        policy = policies.get(name, {})
        max_boost = policy.get("max_boost_seconds", BOOST_DURATION_MAX)

        try:
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            all_cores = list(range(psutil.cpu_count(logical=True)))
            proc.cpu_affinity(all_cores)

            self.current_boost_pid = proc.pid
            self.boost_start_time = time.time()
            self.low_cycles = 0
            self.current_proc_name = name
            self.current_max_boost = max_boost

            print(f"[BOOST] Elevating {name} (PID {proc.pid}) for up to {max_boost}s.")
        except Exception as e:
            print(f"[BOOST] Failed to boost process: {e}")

    def end_boost(self, reason="completed"):
        if self.current_boost_pid is None:
            return
        pid = self.current_boost_pid
        duration = 0.0
        if self.boost_start_time:
            duration = time.time() - self.boost_start_time

        print(f"[BOOST-END] PID {pid} ({self.current_proc_name}) - {reason}, duration {duration:.2f}s")
        self.record_burst_outcome(self.current_proc_name, duration, reason)
        self.restore_state(pid)

        self.current_boost_pid = None
        self.boost_start_time = None
        self.low_cycles = 0
        self.current_proc_name = "unknown"
        self.current_max_boost = BOOST_DURATION_MAX

    # ---- main loop ---- #
    def run(self):
        print("[BURST] Burst guardian starting.")
        while True:
            try:
                self.tick()
            except Exception as e:
                print(f"[BURST-ERROR] Tick failed: {e}")
            time.sleep(SAMPLE_INTERVAL)

    def tick(self):
        fg_pid = self.get_foreground_pid()

        if fg_pid is None:
            if self.current_boost_pid is not None:
                self.end_boost("no_foreground")
            return

        try:
            fg_proc = psutil.Process(fg_pid)
        except psutil.NoSuchProcess:
            if self.current_boost_pid == fg_pid:
                self.end_boost("process_gone")
            return

        cpu_usage = fg_proc.cpu_percent(interval=None)

        if self.current_boost_pid is None:
            if cpu_usage >= CPU_HIGH_THRESHOLD:
                self.boost_process(fg_proc)
            return

        if fg_pid != self.current_boost_pid:
            self.end_boost("focus_changed")
            return

        if self.boost_start_time and (time.time() - self.boost_start_time) > self.current_max_boost:
            self.end_boost("time_cap")
            return

        if cpu_usage < CPU_LOW_THRESHOLD:
            self.low_cycles += 1
        else:
            self.low_cycles = 0

        if self.low_cycles >= STABLE_CYCLES_DONE:
            self.end_boost("cpu_low_stable")

# ------------------ GUI ------------------ #

def format_burst_profiles():
    profiles = STATE.get("burst_profiles", {})
    if not profiles:
        return "No burst history yet."
    lines = []
    for name, p in profiles.items():
        lines.append(
            f"{name}: bursts={p['total_bursts']}, avg={p['average_time']:.1f}s, last_reason={p['last_reason']}"
        )
    return "\n".join(lines)

class GuardianGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HANNIBAL BACKBONE GUARDIAN")
        self.root.geometry("1000x700")

        self.build_layout()
        self.refresh_data()

    def build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        for i in range(3):
            main.columnconfigure(i, weight=1)
        for i in range(6):
            main.rowconfigure(i, weight=1)

        self.risk_label = ttk.Label(main, text="Risk: ?", font=("Consolas", 16, "bold"))
        self.risk_label.grid(row=0, column=0, sticky="w")

        self.trend_label = ttk.Label(main, text="Trend: ?", font=("Consolas", 10))
        self.trend_label.grid(row=0, column=1, sticky="w")

        self.verdict_label = ttk.Label(main, text="Verdict: ?", font=("Consolas", 10))
        self.verdict_label.grid(row=0, column=2, sticky="w")

        ttk.Label(main, text="Weak Points", font=("Consolas", 12, "bold")).grid(row=1, column=0, sticky="nw")
        self.weak_text = tk.Text(main, height=10)
        self.weak_text.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(main, text="Areas of Opportunity", font=("Consolas", 12, "bold")).grid(row=1, column=1, sticky="nw")
        self.ops_text = tk.Text(main, height=10)
        self.ops_text.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(main, text="System Inventory", font=("Consolas", 12, "bold")).grid(row=1, column=2, sticky="nw")
        self.sys_text = tk.Text(main, height=10)
        self.sys_text.grid(row=2, column=2, sticky="nsew", padx=5, pady=5)

        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        for i in range(4):
            controls.columnconfigure(i, weight=1)

        ttk.Label(controls, text="What if RAM (GB):").grid(row=0, column=0, sticky="e")
        self.ram_entry = ttk.Entry(controls, width=10)
        self.ram_entry.grid(row=0, column=1, sticky="w")

        self.whatif_button = ttk.Button(controls, text="Run What-If", command=self.handle_what_if)
        self.whatif_button.grid(row=0, column=2, sticky="w", padx=10)

        self.whatif_result = ttk.Label(controls, text="What-If Verdict: -")
        self.whatif_result.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

        self.refresh_button = ttk.Button(controls, text="Rescan Now", command=self.refresh_data)
        self.refresh_button.grid(row=0, column=3, sticky="e", padx=10)

        ttk.Label(main, text="Learned Burst Profiles & Memory", font=("Consolas", 12, "bold")).grid(row=4, column=0, sticky="nw")
        self.memory_text = tk.Text(main, height=8)
        self.memory_text.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

    def refresh_data(self):
        data = full_scan()

        self.risk_label.config(text=f"Risk: {data['risk_score']}/100 (Run #{data['runs']})")
        self.trend_label.config(text=f"Trend: {data['trend']}")
        self.verdict_label.config(text=f"Verdict: {data['verdict']}")

        self.weak_text.delete("1.0", tk.END)
        if data["weak_points"]:
            for w in data["weak_points"]:
                self.weak_text.insert(tk.END, "- " + w + "\n")
        else:
            self.weak_text.insert(tk.END, "No obvious weak points.\n")

        self.ops_text.delete("1.0", tk.END)
        for o in data["opportunities"]:
            self.ops_text.insert(tk.END, "- " + o + "\n")

        self.sys_text.delete("1.0", tk.END)
        for k, v in data["system"].items():
            self.sys_text.insert(tk.END, f"{k}: {v}\n")

        self.memory_text.delete("1.0", tk.END)
        self.memory_text.insert(tk.END, format_burst_profiles() + "\n\n")
        self.memory_text.insert(tk.END, f"Machine ID: {STATE.get('machine_id')}\n")
        self.memory_text.insert(tk.END, f"Total Runs: {STATE.get('runs')}\n")

        self.root.after(10000, self.refresh_data)

    def handle_what_if(self):
        try:
            ram_val = float(self.ram_entry.get())
        except ValueError:
            self.whatif_result.config(text="What-If Verdict: invalid RAM value.")
            return

        sys_inv = get_system_inventory()
        result = what_if(sys_inv, {"ram_total_gb": ram_val})
        text = f"Risk {result['risk_score']}/100 -> {result['verdict']}"
        self.whatif_result.config(text="What-If Verdict: " + text)

# ------------------ ENTRY POINT ------------------ #

def start_burst_guardian():
    guardian = BurstGuardian()
    t = threading.Thread(target=guardian.run, daemon=True)
    t.start()
    return guardian

def main():
    start_burst_guardian()
    root = tk.Tk()
    app = GuardianGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

