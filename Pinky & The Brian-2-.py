"""
HANNIBAL BACKBONE GUARDIAN – SELF-ADAPTING

This single-file organism provides:

- Auto-load dependencies
- Persistent memory across reboots
- System + software + process inventory
- Anomaly detection
- Rule engine (rules.json) for weak points and opportunities
- Auto-tuning of rules based on correlation with risk/anomaly
- Burst guardian: boosts active foreground "loading" app with all cores
- Per-app learning: burst profiles (stats)
- Per-app policies (app_policies.json): boost_on, max_boost_seconds, cpu_threshold
- Auto-tuning per-app burst policies based on history
- Hannibal-style verdicts
- What-if engine (RAM scenario)
- Rolling log file
- Tkinter GUI:
  - Risk, trend, anomaly, verdict
  - Weak points, opportunities
  - System inventory
  - Top processes
  - Learned burst memory
  - Per-app policy editor
  - Mutation log (recent auto-changes)

Target: Windows ( burst + foreground detection use Win32 APIs ).
"""

import importlib
import subprocess
import sys
import os
import json
import threading
import time
from collections import deque

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

try:
    import win32gui
    import win32process
except ImportError:
    win32gui = None
    win32process = None

import tkinter as tk
from tkinter import ttk

# ------------------ FILE LOCATIONS ------------------ #

APP_DIR = Path(os.path.expanduser("~")) / ".hannibal_guardian"
APP_DIR.mkdir(exist_ok=True)

STATE_FILE = APP_DIR / "guardian_state.json"
POLICY_FILE = APP_DIR / "app_policies.json"
RULES_FILE = APP_DIR / "rules.json"
LOG_FILE = APP_DIR / "guardian.log"

# ------------------ LOGGING ------------------ #

MAX_LOG_LINES = 2000

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        if LOG_FILE.exists():
            with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            lines.append(line + "\n")
            lines = lines[-MAX_LOG_LINES:]
        else:
            lines = [line + "\n"]
        with LOG_FILE.open("w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass

def tail_log(max_lines=200):
    if not LOG_FILE.exists():
        return "No mutations/logs yet."
    try:
        with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return "Unable to read log."

# ------------------ PERSISTENT STATE & CONFIG ------------------ #

DEFAULT_STATE = {
    "schema_version": 2,
    "machine_id": None,
    "runs": 0,
    "risk_history": [],
    "burst_profiles": {},      # per process
    "anomaly_history": [],
}

DEFAULT_POLICIES = {
    # name -> { "boost_on": bool, "max_boost_seconds": int, "cpu_threshold": int }
}

DEFAULT_RULES = {
    "weak_points": [
        {
            "id": "low_ram",
            "condition": "sys_inv['ram_total_gb'] < 4",
            "weight": 15,
            "message": "Low RAM (< 4GB): memory pressure risk."
        },
        {
            "id": "low_physical_cores",
            "condition": "sys_inv['cpu_count_physical'] is not None and sys_inv['cpu_count_physical'] < 4",
            "weight": 10,
            "message": "Low physical core count: parallel load may choke."
        },
        {
            "id": "few_packages",
            "condition": "len(soft_inv['packages']) < 5",
            "weight": 8,
            "message": "Very few Python packages: environment underpowered for advanced analysis."
        },
    ],
    "opportunities": [
        {
            "id": "high_ram",
            "condition": "sys_inv['ram_total_gb'] >= 8",
            "message": "High enough RAM: safe to run heavier predictive models and caching."
        },
        {
            "id": "multi_core",
            "condition": "sys_inv['cpu_count_physical'] is not None and sys_inv['cpu_count_physical'] >= 4",
            "message": "Multiple physical cores: ideal for burst-mode resource boosting and parallel analysis."
        },
        {
            "id": "python_ecosystem_rich",
            "condition": "len(soft_inv['packages']) > 20",
            "message": "Rich Python ecosystem: leverage specialized libraries for ML, visualization, and security."
        },
        {
            "id": "python_ecosystem_minimal",
            "condition": "len(soft_inv['packages']) <= 20",
            "message": "Curate a minimal, hardened package set for reliability."
        },
    ]
}

def load_json(path, default):
    if not path.exists():
        return json.loads(json.dumps(default))
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return json.loads(json.dumps(default))
    merged = json.loads(json.dumps(default))
    if isinstance(merged, dict) and isinstance(data, dict):
        for k, v in data.items():
            merged[k] = v
        return merged
    return data

def save_json(path, data):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"[SAVE] Failed to save {path.name}: {e}")

STATE = load_json(STATE_FILE, DEFAULT_STATE)
if not STATE.get("machine_id"):
    STATE["machine_id"] = str(uuid.uuid4())
STATE["runs"] = STATE.get("runs", 0) + 1
save_json(STATE_FILE, STATE)

APP_POLICIES = load_json(POLICY_FILE, DEFAULT_POLICIES)
RULES = load_json(RULES_FILE, DEFAULT_RULES)
save_json(RULES_FILE, RULES)

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

# ------------------ INVENTORY ------------------ #

def get_system_inventory():
    vm = psutil.virtual_memory()
    info = {
        "os": MACHINE_FINGERPRINT["os"],
        "machine": MACHINE_FINGERPRINT["machine"],
        "processor": MACHINE_FINGERPRINT["processor"],
        "cpu_count_logical": MACHINE_FINGERPRINT["cpu_count_logical"],
        "cpu_count_physical": MACHINE_FINGERPRINT["cpu_count_physical"],
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_percent": vm.percent,
        "cpu_usage": psutil.cpu_percent(interval=0.1),
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

def get_top_processes(limit=8):
    procs = []
    for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info"]):
        try:
            info = p.info
            mem_mb = info["memory_info"].rss / (1024**2)
            procs.append((info["name"], info["pid"], info["cpu_percent"], mem_mb))
        except Exception:
            continue
    procs.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return procs[:limit]

# ------------------ RULE ENGINE ------------------ #

def evaluate_rules(sys_inv, soft_inv):
    weak_points = []
    opportunities = []
    weak_total_weight = 0

    safe_globals = {"__builtins__": {}}
    local_ctx = {"sys_inv": sys_inv, "soft_inv": soft_inv}

    for rule in RULES.get("weak_points", []):
        cond = rule.get("condition")
        weight = rule.get("weight", 5)
        msg = rule.get("message", rule.get("id", "weak_point"))
        if not cond:
            continue
        try:
            if eval(cond, safe_globals, local_ctx):
                weak_points.append(msg)
                weak_total_weight += weight
        except Exception as e:
            log(f"[RULE] Weak rule {rule.get('id')} failed: {e}")

    for rule in RULES.get("opportunities", []):
        cond = rule.get("condition")
        msg = rule.get("message", rule.get("id", "opportunity"))
        if not cond:
            continue
        try:
            if eval(cond, safe_globals, local_ctx):
                opportunities.append(msg)
        except Exception as e:
            log(f"[RULE] Opp rule {rule.get('id')} failed: {e}")

    return weak_points, opportunities, weak_total_weight

def auto_tune_rules(risk_score_value, anomaly_score_value, weak_weight):
    high_risk = risk_score_value > 70 or anomaly_score_value > 70
    low_risk = risk_score_value < 30 and anomaly_score_value < 30

    for rule in RULES.get("weak_points", []):
        w = rule.get("weight", 5)
        if high_risk and weak_weight < 40:
            new_w = min(30, w + 1)
            if new_w != w:
                rule["weight"] = new_w
                log(f"[RULE-MUTATION] Increased weight of weak rule {rule.get('id')} to {new_w}")
        elif low_risk and weak_weight > 10:
            new_w = max(1, w - 1)
            if new_w != w:
                rule["weight"] = new_w
                log(f"[RULE-MUTATION] Decreased weight of weak rule {rule.get('id')} to {new_w}")
    save_json(RULES_FILE, RULES)

# ------------------ ANOMALY ENGINE ------------------ #

class AnomalyEngine:
    def __init__(self, window=60):
        self.cpu_window = deque(maxlen=window)
        self.mem_window = deque(maxlen=window)

    def update(self, sys_inv):
        self.cpu_window.append(sys_inv["cpu_usage"])
        self.mem_window.append(sys_inv["ram_used_gb"])

    def score(self):
        if len(self.cpu_window) < 10:
            return 0
        cpu_vals = list(self.cpu_window)
        mem_vals = list(self.mem_window)
        cpu_delta = max(cpu_vals) - min(cpu_vals)
        mem_delta = max(mem_vals) - min(mem_vals)

        score = 0
        if cpu_delta > 50:
            score += 40
        elif cpu_delta > 30:
            score += 25
        elif cpu_delta > 15:
            score += 10

        if mem_delta > 4:
            score += 40
        elif mem_delta > 2:
            score += 25
        elif mem_delta > 1:
            score += 10

        return min(100, score)

ANOMALY_ENGINE = AnomalyEngine()

# ------------------ RISK & VERDICT ------------------ #

def hannibal_judgment(weak_points_count, opportunities_count, anomaly_score):
    if anomaly_score > 70:
        return "System behavior is erratic. Freeze ambitions, trace anomalies, reinforce defenses."
    if weak_points_count == 0 and opportunities_count > 0:
        return "Calm. Environment is stable. Exploit strengths ruthlessly."
    if weak_points_count <= 2:
        return "Manageable vulnerabilities. Exploit opportunities while reinforcing weak links."
    if weak_points_count <= 5:
        return "Too many exposed ribs. Prioritize structural reinforcement before ambitious moves."
    return "This house is made of glass. Do not play with stones; rebuild the foundation."

def risk_score(weak_weight, sys_inv, anomaly_score):
    base = weak_weight
    if sys_inv["ram_total_gb"] < 4:
        base += 10
    if sys_inv["cpu_count_physical"] and sys_inv["cpu_count_physical"] < 4:
        base += 5
    base += int(anomaly_score * 0.4)
    return min(100, base)

def update_risk_history(score):
    history = STATE.get("risk_history", [])
    history.append({"score": score, "run": STATE["runs"]})
    history = history[-200:]
    STATE["risk_history"] = history
    save_json(STATE_FILE, STATE)
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

# ------------------ WHAT-IF ENGINE ------------------ #

def what_if(sys_inv, scenario):
    sys_clone = dict(sys_inv)
    sys_clone.update(scenario)
    soft_inv = get_software_inventory()
    weak_points, opportunities, weak_weight = evaluate_rules(sys_clone, soft_inv)
    anomaly_score = ANOMALY_ENGINE.score()
    score = risk_score(weak_weight, sys_clone, anomaly_score)
    verdict = hannibal_judgment(len(weak_points), len(opportunities), anomaly_score)
    return {
        "hypothetical_sys": sys_clone,
        "weak_points": weak_points,
        "risk_score": score,
        "verdict": verdict
    }

# ------------------ FULL SCAN ------------------ #

def full_scan():
    sys_inv = get_system_inventory()
    soft_inv = get_software_inventory()
    ANOMALY_ENGINE.update(sys_inv)
    anomaly_score = ANOMALY_ENGINE.score()

    weak_points, opportunities, weak_weight = evaluate_rules(sys_inv, soft_inv)
    score = risk_score(weak_weight, sys_inv, anomaly_score)
    history = update_risk_history(score)
    trend = predict_trend(history)
    verdict = hannibal_judgment(len(weak_points), len(opportunities), anomaly_score)

    auto_tune_rules(score, anomaly_score, weak_weight)

    top_procs = get_top_processes()

    return {
        "system": sys_inv,
        "software": soft_inv,
        "weak_points": weak_points,
        "opportunities": opportunities,
        "risk_score": score,
        "trend": trend,
        "verdict": verdict,
        "runs": STATE["runs"],
        "anomaly_score": anomaly_score,
        "top_processes": top_procs,
    }

# ------------------ BURST GUARDIAN ------------------ #

CPU_HIGH_DEFAULT = 40
CPU_LOW_THRESHOLD = 10
SAMPLE_INTERVAL = 0.5
STABLE_CYCLES_DONE = 5
BOOST_DURATION_DEFAULT = 30

class BurstGuardian:
    def __init__(self):
        self.current_boost_pid = None
        self.boost_start_time = None
        self.low_cycles = 0
        self.original_states = {}
        self.current_proc_name = "unknown"
        self.current_max_boost = BOOST_DURATION_DEFAULT
        self.current_cpu_threshold = CPU_HIGH_DEFAULT

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

    def get_policy_for(self, name):
        name = name.lower()
        policy = APP_POLICIES.get(name, {
            "boost_on": True,
            "max_boost_seconds": BOOST_DURATION_DEFAULT,
            "cpu_threshold": CPU_HIGH_DEFAULT,
        })
        APP_POLICIES[name] = policy
        save_json(POLICY_FILE, APP_POLICIES)
        return policy

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
        save_json(STATE_FILE, STATE)

        self.auto_tune_policy(name, prof)

    def auto_tune_policy(self, name, prof):
        policy = self.get_policy_for(name)
        avg_time = prof.get("average_time", 0)

        if avg_time > 20 and policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT) < 60:
            policy["max_boost_seconds"] = min(60, policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT) + 5)
            log(f"[POLICY-MUTATION] Increased max_boost_seconds for {name} to {policy['max_boost_seconds']}")
        elif avg_time < 5 and policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT) > 10:
            policy["max_boost_seconds"] = max(10, policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT) - 5)
            log(f"[POLICY-MUTATION] Decreased max_boost_seconds for {name} to {policy['max_boost_seconds']}")

        APP_POLICIES[name] = policy
        save_json(POLICY_FILE, APP_POLICIES)

    def boost_process(self, proc):
        self.capture_state(proc)
        name = proc.name().lower()
        policy = self.get_policy_for(name)

        if not policy.get("boost_on", True):
            return

        max_boost = policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT)
        cpu_threshold = policy.get("cpu_threshold", CPU_HIGH_DEFAULT)

        try:
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            all_cores = list(range(psutil.cpu_count(logical=True)))
            proc.cpu_affinity(all_cores)

            self.current_boost_pid = proc.pid
            self.boost_start_time = time.time()
            self.low_cycles = 0
            self.current_proc_name = name
            self.current_max_boost = max_boost
            self.current_cpu_threshold = cpu_threshold

            log(f"[BOOST] Elevating {name} (PID {proc.pid}) up to {max_boost}s, threshold {cpu_threshold}%.")
        except Exception as e:
            log(f"[BOOST] Failed to boost process: {e}")

    def end_boost(self, reason="completed"):
        if self.current_boost_pid is None:
            return
        pid = self.current_boost_pid
        duration = 0.0
        if self.boost_start_time:
            duration = time.time() - self.boost_start_time

        log(f"[BOOST-END] PID {pid} ({self.current_proc_name}) - {reason}, duration {duration:.2f}s")
        self.record_burst_outcome(self.current_proc_name, duration, reason)
        self.restore_state(pid)

        self.current_boost_pid = None
        self.boost_start_time = None
        self.low_cycles = 0
        self.current_proc_name = "unknown"
        self.current_max_boost = BOOST_DURATION_DEFAULT
        self.current_cpu_threshold = CPU_HIGH_DEFAULT

    def run(self):
        log("[BURST] Burst guardian starting.")
        while True:
            try:
                self.tick()
            except Exception as e:
                log(f"[BURST-ERROR] Tick failed: {e}")
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
        name = fg_proc.name().lower()
        policy = self.get_policy_for(name)
        cpu_threshold = policy.get("cpu_threshold", CPU_HIGH_DEFAULT)

        if self.current_boost_pid is None:
            if policy.get("boost_on", True) and cpu_usage >= cpu_threshold:
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

# ------------------ GUI HELPERS ------------------ #

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

def format_top_processes(top_procs):
    if not top_procs:
        return "No process data."
    lines = []
    for name, pid, cpu, mem in top_procs:
        lines.append(f"{name} (PID {pid}) - CPU: {cpu:.1f}%, RAM: {mem:.1f} MB")
    return "\n".join(lines)

def policy_for_display(name):
    name = name.lower()
    policy = APP_POLICIES.get(name, {
        "boost_on": True,
        "max_boost_seconds": BOOST_DURATION_DEFAULT,
        "cpu_threshold": CPU_HIGH_DEFAULT,
    })
    APP_POLICIES[name] = policy
    save_json(POLICY_FILE, APP_POLICIES)
    return policy

# ------------------ GUI ------------------ #

class GuardianGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HANNIBAL BACKBONE GUARDIAN")
        self.root.geometry("1350x850")

        self.selected_policy_app = tk.StringVar(value="")

        self.build_layout()
        self.refresh_data()

    def build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        for i in range(3):
            main.columnconfigure(i, weight=1)
        for i in range(8):
            main.rowconfigure(i, weight=1)

        self.risk_label = ttk.Label(main, text="Risk: ?", font=("Consolas", 16, "bold"))
        self.risk_label.grid(row=0, column=0, sticky="w")

        self.trend_label = ttk.Label(main, text="Trend: ?", font=("Consolas", 10))
        self.trend_label.grid(row=0, column=1, sticky="w")

        self.verdict_label = ttk.Label(main, text="Verdict: ?", font=("Consolas", 10))
        self.verdict_label.grid(row=0, column=2, sticky="w")

        ttk.Label(main, text="Weak Points", font=("Consolas", 12, "bold")).grid(row=1, column=0, sticky="nw")
        self.weak_text = tk.Text(main, height=8)
        self.weak_text.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)

        ttk.Label(main, text="Areas of Opportunity", font=("Consolas", 12, "bold")).grid(row=1, column=1, sticky="nw")
        self.ops_text = tk.Text(main, height=8)
        self.ops_text.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)

        ttk.Label(main, text="System Inventory", font=("Consolas", 12, "bold")).grid(row=1, column=2, sticky="nw")
        self.sys_text = tk.Text(main, height=8)
        self.sys_text.grid(row=2, column=2, sticky="nsew", padx=4, pady=4)

        ttk.Label(main, text="Top Processes", font=("Consolas", 12, "bold")).grid(row=3, column=0, sticky="nw")
        self.proc_text = tk.Text(main, height=8)
        self.proc_text.grid(row=4, column=0, sticky="nsew", padx=4, pady=4)

        ttk.Label(main, text="Learned Burst Profiles & Memory", font=("Consolas", 12, "bold")).grid(row=3, column=1, sticky="nw")
        self.memory_text = tk.Text(main, height=8)
        self.memory_text.grid(row=4, column=1, sticky="nsew", padx=4, pady=4)

        controls = ttk.Frame(main)
        controls.grid(row=3, column=2, sticky="new", pady=4)
        for i in range(2):
            controls.columnconfigure(i, weight=1)

        ttk.Label(controls, text="What if RAM (GB):").grid(row=0, column=0, sticky="e")
        self.ram_entry = ttk.Entry(controls, width=10)
        self.ram_entry.grid(row=0, column=1, sticky="w")

        self.whatif_button = ttk.Button(controls, text="Run What-If", command=self.handle_what_if)
        self.whatif_button.grid(row=1, column=0, sticky="w", pady=4)

        self.whatif_result = ttk.Label(controls, text="What-If Verdict: -")
        self.whatif_result.grid(row=1, column=1, sticky="w", pady=4)

        self.refresh_button = ttk.Button(controls, text="Rescan Now", command=self.refresh_data)
        self.refresh_button.grid(row=2, column=0, sticky="w", pady=4)

        self.anomaly_label = ttk.Label(controls, text="Anomaly: 0/100")
        self.anomaly_label.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(main, text="Per-App Policy Editor", font=("Consolas", 12, "bold")).grid(row=5, column=0, sticky="nw")
        policy_frame = ttk.Frame(main)
        policy_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=4)
        for i in range(6):
            policy_frame.columnconfigure(i, weight=1)

        ttk.Label(policy_frame, text="App Name:").grid(row=0, column=0, sticky="e")
        self.policy_app_entry = ttk.Entry(policy_frame, textvariable=self.selected_policy_app, width=25)
        self.policy_app_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(policy_frame, text="Boost On:").grid(row=0, column=2, sticky="e")
        self.policy_boost_var = tk.BooleanVar(value=True)
        self.policy_boost_check = ttk.Checkbutton(policy_frame, variable=self.policy_boost_var)
        self.policy_boost_check.grid(row=0, column=3, sticky="w")

        ttk.Label(policy_frame, text="Max Boost Seconds:").grid(row=1, column=0, sticky="e")
        self.policy_max_boost_entry = ttk.Entry(policy_frame, width=10)
        self.policy_max_boost_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(policy_frame, text="CPU Threshold (%):").grid(row=1, column=2, sticky="e")
        self.policy_cpu_thresh_entry = ttk.Entry(policy_frame, width=10)
        self.policy_cpu_thresh_entry.grid(row=1, column=3, sticky="w")

        self.policy_load_button = ttk.Button(policy_frame, text="Load Policy", command=self.load_policy_from_name)
        self.policy_load_button.grid(row=0, column=4, sticky="w", padx=4)

        self.policy_save_button = ttk.Button(policy_frame, text="Save Policy", command=self.save_policy_from_gui)
        self.policy_save_button.grid(row=1, column=4, sticky="w", padx=4)

        self.policy_status = ttk.Label(policy_frame, text="Policy: -")
        self.policy_status.grid(row=2, column=0, columnspan=5, sticky="w")

        ttk.Label(main, text="Mutation & Event Log", font=("Consolas", 12, "bold")).grid(row=5, column=2, sticky="nw")
        self.log_text = tk.Text(main, height=10)
        self.log_text.grid(row=6, column=2, sticky="nsew", padx=4, pady=4)

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

        self.proc_text.delete("1.0", tk.END)
        self.proc_text.insert(tk.END, format_top_processes(data["top_processes"]) + "\n")

        self.memory_text.delete("1.0", tk.END)
        self.memory_text.insert(tk.END, format_burst_profiles() + "\n\n")
        self.memory_text.insert(tk.END, f"Machine ID: {STATE.get('machine_id')}\n")
        self.memory_text.insert(tk.END, f"Total Runs: {STATE.get('runs')}\n")

        self.anomaly_label.config(text=f"Anomaly: {data['anomaly_score']}/100")

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, tail_log(120))

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

    def load_policy_from_name(self):
        name = self.selected_policy_app.get().strip().lower()
        if not name:
            self.policy_status.config(text="Policy: enter an app name first.")
            return
        policy = policy_for_display(name)
        self.policy_boost_var.set(policy.get("boost_on", True))
        self.policy_max_boost_entry.delete(0, tk.END)
        self.policy_max_boost_entry.insert(0, str(policy.get("max_boost_seconds", BOOST_DURATION_DEFAULT)))
        self.policy_cpu_thresh_entry.delete(0, tk.END)
        self.policy_cpu_thresh_entry.insert(0, str(policy.get("cpu_threshold", CPU_HIGH_DEFAULT)))
        self.policy_status.config(text=f"Policy: loaded for {name}")

    def save_policy_from_gui(self):
        name = self.selected_policy_app.get().strip().lower()
        if not name:
            self.policy_status.config(text="Policy: enter an app name first.")
            return

        try:
            max_boost = int(self.policy_max_boost_entry.get())
        except ValueError:
            self.policy_status.config(text="Policy: max boost seconds must be integer.")
            return

        try:
            cpu_thresh = int(self.policy_cpu_thresh_entry.get())
        except ValueError:
            self.policy_status.config(text="Policy: CPU threshold must be integer.")
            return

        APP_POLICIES[name] = {
            "boost_on": self.policy_boost_var.get(),
            "max_boost_seconds": max_boost,
            "cpu_threshold": cpu_thresh,
        }
        save_json(POLICY_FILE, APP_POLICIES)
        self.policy_status.config(text=f"Policy: saved for {name}")
        log(f"[POLICY] Updated policy for {name}: boost_on={self.policy_boost_var.get()}, max={max_boost}, cpu_thresh={cpu_thresh}")

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

