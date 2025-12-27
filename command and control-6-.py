#!/usr/bin/env python3
"""
Hybrid Exoskeleton v2.3 – Admin-elevated, mission-aware guardian

Mode A: Always elevates to Administrator on startup (Windows).
E1 message: "Administrative privileges required — elevating now…"

Core features preserved:
- Autoloader (psutil)
- GUI (processes, network, alerts, logs)
- Brain loop (time-aware, idle-aware)
- Persistent memory (JSON)
- Network whitelist with mission-critical flags
"""

import sys
import os
import subprocess
import importlib
import threading
import queue
import time
import traceback
import platform
import json
import random
from collections import deque
from datetime import datetime

# ==============================
# ADMIN ELEVATION (MODE A, E1)
# ==============================

def ensure_admin():
    """
    Mode A: always elevate to admin on Windows at startup.

    Behavior:
    - If on non-Windows: do nothing.
    - If already admin: continue.
    - If not admin:
        - Show minimal popup: "Administrative privileges required — elevating now..."
        - Relaunch self with 'runas' (UAC)
        - Exit current process.
    """
    if os.name != "nt":
        return  # Only elevate on Windows

    try:
        import ctypes
        # Check if already admin
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = False

        if is_admin:
            return

        # E1 message: minimal, direct
        MB_OK = 0x0
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                "Administrative privileges required — elevating now…",
                "Hybrid Exoskeleton",
                MB_OK
            )
        except Exception:
            # If MessageBox fails, just skip the message and elevate anyway
            pass

        # Relaunch self with elevation
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                params,
                None,
                1
            )
        except Exception as e:
            # If elevation fails, print to stderr and continue (non-admin)
            sys.stderr.write(f"[ELEVATION] Failed to elevate: {e}\n")
            return

        # If we successfully triggered elevation, exit this process.
        sys.exit(0)

    except Exception as e:
        # Fail-safe: log to stderr, continue non-admin
        sys.stderr.write(f"[ELEVATION] Unexpected error: {e}\n")
        return

# ==============================
# CONFIG
# ==============================

REQUIRED_MODULES = ["psutil"]

IDLE_CPU_THRESHOLD = 20.0
SCAN_INTERVAL_IDLE = 3.0
SCAN_INTERVAL_BUSY = 10.0

RULES_FILE = "exo_rules.json"
BRAIN_STATE_FILE = "exo_brain_state.json"
NETWORK_RESOURCES_FILE = "network_resources.json"

CREATIVE_MODE = True
DEFAULT_NETWORK_ENABLED = True
HOT_USES_THRESHOLD = 3

# Process names that count as mission-critical (substring match, case-insensitive)
CRITICAL_PROCESS_NAMES = ["nginx", "postgres", "mysqld", "redis", "sshd", "explorer"]

# ==============================
# AUTOLOADER
# ==============================

class AutoLoader:
    def __init__(self, modules, log_func):
        self.modules = modules
        self.log = log_func
        self.loaded_modules = {}
        self.errors = {}

    def ensure_all(self):
        for name in self.modules:
            self._ensure_module(name)

    def _ensure_module(self, name):
        self.log(f"[AUTOLOADER] Checking module: {name}")
        try:
            mod = importlib.import_module(name)
            self.loaded_modules[name] = mod
            self.log(f"[AUTOLOADER] Loaded: {name}")
        except ImportError:
            self.log(f"[AUTOLOADER] Missing: {name} — installing via pip...")
            if self._install_module(name):
                try:
                    mod = importlib.import_module(name)
                    self.loaded_modules[name] = mod
                    self.log(f"[AUTOLOADER] Installed and loaded: {name}")
                except Exception as e:
                    msg = f"[AUTOLOADER] Failed to import {name} after install: {e}"
                    self.log(msg)
                    self.errors[name] = msg
            else:
                msg = f"[AUTOLOADER] pip install failed for {name}"
                self.log(msg)
                self.errors[name] = msg
        except Exception as e:
            msg = f"[AUTOLOADER] Unexpected error loading {name}: {e}"
            self.log(msg)
            self.errors[name] = msg

    def _install_module(self, name):
        try:
            cmd = [sys.executable, "-m", "pip", "install", name]
            self.log(f"[AUTOLOADER] Running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode == 0:
                self.log(f"[AUTOLOADER] pip install succeeded: {name}")
                return True
            else:
                self.log(f"[AUTOLOADER] pip install error for {name}:\n{proc.stdout}")
                return False
        except Exception as e:
            self.log(f"[AUTOLOADER] Exception during pip install for {name}: {e}")
            return False

# ==============================
# PERSISTENT MEMORY
# ==============================

class BrainMemory:
    def __init__(self, log_func):
        self.log = log_func
        self.state = {
            "run_count": 0,
            "last_efficiency": None,
            "best_efficiency": None,
            "efficiency_history": [],
            "rule_stats": {},
            "network_usage": {},           # path -> {uses, last_info}
            "network_enabled": DEFAULT_NETWORK_ENABLED,
            "last_network_snapshot": None,
            "ignored_alert_signatures": [],
        }
        self._load()

    def _load(self):
        try:
            with open(BRAIN_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.update(data)
            self.log(f"[MEMORY] Loaded brain state. Runs={self.state.get('run_count')}")
        except FileNotFoundError:
            self.log("[MEMORY] No previous brain state; starting fresh.")
        except Exception as e:
            self.log(f"[MEMORY] Error loading brain state: {e}")

    def save(self):
        try:
            with open(BRAIN_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            self.log("[MEMORY] Brain state saved.")
        except Exception as e:
            self.log(f"[MEMORY] Failed to save brain state: {e}")

    def register_startup(self):
        self.state["run_count"] = self.state.get("run_count", 0) + 1
        self.save()

    def record_efficiency(self, eff):
        self.state["last_efficiency"] = eff
        best = self.state.get("best_efficiency")
        if best is None or eff > best:
            self.state["best_efficiency"] = eff
        hist = self.state.get("efficiency_history", [])
        hist.append(eff)
        self.state["efficiency_history"] = hist[-200:]

    def record_rule_outcome(self, rule_id, delta):
        stats = self.state["rule_stats"].setdefault(rule_id, {"uses": 0, "avg_delta": 0.0})
        n = stats["uses"]
        avg = stats["avg_delta"]
        new_avg = (avg * n + delta) / (n + 1)
        stats["uses"] = n + 1
        stats["avg_delta"] = new_avg

    def record_network_usage(self, path, info):
        stats = self.state["network_usage"].setdefault(path, {"uses": 0, "last_info": None})
        stats["uses"] += 1
        stats["last_info"] = info

    def get_last_network_snapshot(self):
        return self.state.get("last_network_snapshot")

    def set_last_network_snapshot(self, snap):
        self.state["last_network_snapshot"] = snap

    def add_ignored_alert_signature(self, sig_hash: int):
        if sig_hash not in self.state["ignored_alert_signatures"]:
            self.state["ignored_alert_signatures"].append(sig_hash)

    def is_alert_signature_ignored(self, sig_hash: int):
        return sig_hash in self.state.get("ignored_alert_signatures", [])

    def get_network_enabled(self):
        return bool(self.state.get("network_enabled", DEFAULT_NETWORK_ENABLED))

    def set_network_enabled(self, enabled: bool):
        self.state["network_enabled"] = bool(enabled)

    def get_summary_text(self):
        best = self.state.get("best_efficiency")
        last = self.state.get("last_efficiency")
        runs = self.state.get("run_count")
        return f"Runs: {runs} | Last: {last if last is not None else 'N/A'} | Best: {best if best is not None else 'N/A'}"

# ==============================
# NETWORK MANAGER
# ==============================

class NetworkManager:
    """
    Each resource:
      { "path": "...", "label": "...", "read_only": true, "mission_critical": false }
    """

    def __init__(self, log_func, memory: BrainMemory):
        self.log = log_func
        self.memory = memory
        self.resources = []
        self._load_resources()

    def _load_resources(self):
        try:
            with open(NETWORK_RESOURCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                norm = []
                for r in data:
                    if not isinstance(r, dict) or "path" not in r:
                        continue
                    nr = {
                        "path": r["path"],
                        "label": r.get("label", r["path"]),
                        "read_only": bool(r.get("read_only", True)),
                        "mission_critical": bool(r.get("mission_critical", False)),
                    }
                    norm.append(nr)
                self.resources = norm
                self.log(f"[NETWORK] Loaded {len(self.resources)} resources.")
            else:
                self.log("[NETWORK] network_resources.json invalid; expected list.")
        except FileNotFoundError:
            self.log("[NETWORK] No network_resources.json; starting empty.")
            self.resources = []
            self._save_resources()
        except Exception as e:
            self.log(f"[NETWORK] Error loading resources: {e}")
            self.resources = []

    def _save_resources(self):
        try:
            with open(NETWORK_RESOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.resources, f, indent=2)
            self.log(f"[NETWORK] Saved {len(self.resources)} resources.")
        except Exception as e:
            self.log(f"[NETWORK] Failed to save network resources: {e}")

    def add_resource(self, path, label=None, read_only=True, mission_critical=False):
        if not path:
            return False, "Empty path"
        label = label or path
        for r in self.resources:
            if r.get("path") == path:
                return False, "Resource already exists"
        entry = {
            "path": path,
            "label": label,
            "read_only": bool(read_only),
            "mission_critical": bool(mission_critical),
        }
        self.resources.append(entry)
        self._save_resources()
        self.log(f"[NETWORK] Added resource: {label} -> {path} (mission={mission_critical})")
        return True, "Added"

    def retire_resource(self, path):
        before = len(self.resources)
        self.resources = [r for r in self.resources if r.get("path") != path]
        if len(self.resources) < before:
            self._save_resources()
            self.log(f"[NETWORK] Retired resource: {path}")
            return True
        return False

    def list_local_mounts(self):
        mounts = []
        try:
            if os.name == "nt":
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    d = f"{letter}:\\"
                    if os.path.exists(d):
                        mounts.append(d)
            else:
                if os.path.exists("/proc/mounts"):
                    with open("/proc/mounts", "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                mounts.append(parts[1])
        except Exception as e:
            self.log(f"[NETWORK] Error enumerating mounts: {e}")
        return sorted(set(mounts))

    def probe_whitelisted_resources(self, single_path=None):
        results = {}
        resources = self.resources
        if single_path:
            resources = [r for r in resources if r.get("path") == single_path]
        for res in resources:
            path = res.get("path")
            label = res.get("label", path)
            mc = res.get("mission_critical", False)
            try:
                accessible = os.path.exists(path)
                info = {"accessible": accessible, "mission_critical": mc}
                if accessible:
                    try:
                        entries = os.listdir(path)
                        info["top_level_count"] = len(entries)
                        total_bytes = 0
                        for name in entries[:20]:
                            try:
                                p = os.path.join(path, name)
                                if os.path.isfile(p):
                                    total_bytes += os.path.getsize(p)
                            except Exception:
                                continue
                        info["sample_bytes"] = total_bytes
                    except Exception as e:
                        info["error_listing"] = str(e)
                results[label] = info
                self.log(f"[NETWORK] Probe {label}: {info}")
                self.memory.record_network_usage(path, info)
            except Exception as e:
                self.log(f"[NETWORK] Error probing {path}: {e}")
                results[label] = {"accessible": False, "error": str(e), "mission_critical": mc}
        return results

    def get_resource_by_label(self, label):
        for r in self.resources:
            if r.get("label") == label:
                return r
        return None

    def build_snapshot(self, probe_results):
        mounts = self.list_local_mounts()
        resources = {}
        for res in self.resources:
            label = res.get("label", res.get("path"))
            info = probe_results.get(label)
            if info is not None:
                resources[label] = info
        return {"mounts": mounts, "resources": resources}

    @staticmethod
    def diff_snapshots(old, new):
        if old is None:
            return {"mounts_added": [], "mounts_removed": [], "resources_changed": {}}
        changes = {"mounts_added": [], "mounts_removed": [], "resources_changed": {}}
        old_mounts = set(old.get("mounts", []))
        new_mounts = set(new.get("mounts", []))
        changes["mounts_added"] = sorted(new_mounts - old_mounts)
        changes["mounts_removed"] = sorted(old_mounts - new_mounts)
        old_res = old.get("resources", {})
        new_res = new.get("resources", {})
        labels = set(old_res.keys()) | set(new_res.keys())
        for lab in labels:
            o = old_res.get(lab)
            n = new_res.get(lab)
            if o != n:
                changes["resources_changed"][lab] = {"old": o, "new": n}
        return changes

    @staticmethod
    def has_meaningful_change(changes):
        return bool(changes["mounts_added"] or changes["mounts_removed"] or changes["resources_changed"])

# ==============================
# RULE ENGINE (stable)
# ==============================

class OptimizationRuleEngine:
    def __init__(self, log_func):
        self.log = log_func
        self.rules = []
        self.history = deque(maxlen=200)
        self._load_rules()

    def _load_rules(self):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            self.log(f"[RULES] Loaded {len(self.rules)} rules.")
        except FileNotFoundError:
            self._seed_default_rules()
            self._save_rules()
        except Exception as e:
            self.log(f"[RULES] Error loading rules: {e}")
            self._seed_default_rules()

    def _save_rules(self):
        try:
            with open(RULES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.rules, f, indent=2)
            self.log(f"[RULES] Saved {len(self.rules)} rules.")
        except Exception as e:
            self.log(f"[RULES] Failed to save rules: {e}")

    def _seed_default_rules(self):
        self.rules = [
            {"id": "daytime_backoff", "description": "During daytime and high CPU, back off.", "cpu_gt": 60, "mem_gt": 0, "when_idle_only": False, "hour_start": 8, "hour_end": 20, "action": "backoff_scans"},
            {"id": "night_deep_analysis", "description": "At night and idle, analyze deeper.", "cpu_gt": 0, "mem_gt": 0, "when_idle_only": True, "hour_start": 0, "hour_end": 6, "action": "focus_heavy_procs"},
            {"id": "weekend_chill", "description": "Weekend: be less aggressive.", "cpu_gt": 50, "mem_gt": 0, "when_idle_only": False, "weekend_only": True, "action": "extra_backoff"},
        ]

    def evaluate(self, metrics, process_rows, idle_mode, time_ctx, network_info=None):
        cpu = metrics.get("cpu", 0.0)
        mem = metrics.get("mem", 0.0)
        hour = time_ctx["hour"]
        is_weekend = time_ctx["is_weekend"]

        actions = set()
        explanations = []
        used_rule_ids = []

        for rule in self.rules:
            if idle_mode is False and rule.get("when_idle_only", False):
                continue
            hs = rule.get("hour_start", 0)
            he = rule.get("hour_end", 24)
            if not (hs <= hour < he):
                continue
            if rule.get("weekend_only", False) and not is_weekend:
                continue
            if cpu > rule.get("cpu_gt", 0) and mem > rule.get("mem_gt", 0):
                act = rule.get("action")
                actions.add(act)
                used_rule_ids.append(rule["id"])
                explanations.append(
                    f"[RULE:{rule['id']}] {rule['description']} "
                    f"(hour={hour}, cpu={cpu:.1f}, mem={mem:.1f})"
                )

        suggested_interval = None
        highlight_pids = []
        if "focus_heavy_procs" in actions and process_rows:
            top = process_rows[:5]
            highlight_pids = [int(r[0]) for r in top]
        if "backoff_scans" in actions or "extra_backoff" in actions:
            suggested_interval = "busy"

        return {
            "highlight_pids": highlight_pids,
            "interval_hint": suggested_interval,
            "explanations": explanations,
            "used_rule_ids": used_rule_ids,
        }

    def record_outcome(self, eff_before, eff_after, used_rule_ids):
        if not used_rule_ids:
            return
        self.history.append((eff_before, eff_after, used_rule_ids))
        # Evolution disabled for stability; can be re-enabled later.

# ==============================
# GUI
# ==============================

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import simpledialog, messagebox

class AddResourceDialog(simpledialog.Dialog):
    def __init__(self, parent, title="Add Network Resource"):
        self.path = ""
        self.label = ""
        self.read_only = True
        self.mission_critical = False
        super().__init__(parent, title=title)

    def body(self, master):
        ttk.Label(master, text="Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_entry = ttk.Entry(master, width=60)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(master, text="Label (optional):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.label_entry = ttk.Entry(master, width=40)
        self.label_entry.grid(row=1, column=1, padx=5, pady=5)

        self.read_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(master, text="Read-only", variable=self.read_only_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5
        )

        self.mission_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(master, text="Mission-critical", variable=self.mission_var).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=5
        )
        return self.path_entry

    def apply(self):
        self.path = self.path_entry.get().strip()
        self.label = self.label_entry.get().strip()
        self.read_only = bool(self.read_only_var.get())
        self.mission_critical = bool(self.mission_var.get())

class ExoGUI:
    def __init__(self, root, task_queue, control_queue):
        self.root = root
        self.task_queue = task_queue
        self.control_queue = control_queue

        self.show_watched_only = tk.BooleanVar(value=False)
        self.net_sort_col = None
        self.net_sort_reverse = False

        self.alert_active = False
        self.last_alert_details = None

        self.root.title("Hybrid Exoskeleton Console")
        self.root.geometry("1450x820")

        self._build_layout()
        self._poll_queue()
        self._blink_alert()

    def _build_layout(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(0, 10))

        self.efficiency_var = tk.StringVar(value="System Efficiency: N/A")
        ttk.Label(top, textvariable=self.efficiency_var).pack(side=tk.LEFT, padx=10)

        self.time_var = tk.StringVar(value="Time: N/A")
        ttk.Label(top, textvariable=self.time_var).pack(side=tk.LEFT, padx=10)

        self.memory_var = tk.StringVar(value="Memory: N/A")
        ttk.Label(top, textvariable=self.memory_var).pack(side=tk.LEFT, padx=10)

        creative_text = "Creative mode: ON" if CREATIVE_MODE else "Creative mode: OFF"
        ttk.Label(top, text=creative_text, foreground="blue").pack(side=tk.LEFT, padx=10)

        self.network_state_var = tk.StringVar(
            value="Network: ON" if DEFAULT_NETWORK_ENABLED else "Network: OFF"
        )
        self.network_state_label = ttk.Label(
            top,
            textvariable=self.network_state_var,
            foreground="green" if DEFAULT_NETWORK_ENABLED else "red"
        )
        self.network_state_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="Toggle Network", command=self._cmd_toggle_network).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Stop Brain", command=self._cmd_stop_brain).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Start Brain", command=self._cmd_start_brain).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="System Inventory", command=self._cmd_inventory).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="Run Autoloader", command=self._cmd_run_autoload).pack(side=tk.RIGHT, padx=5)

        alert_frame = ttk.Frame(self.root)
        alert_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.alert_var = tk.StringVar(value="No alerts")
        self.alert_label = tk.Label(
            alert_frame, textvariable=self.alert_var, bg="black", fg="white", padx=10, pady=3
        )
        self.alert_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(alert_frame, text="Acknowledge", command=self._cmd_ack_alert).pack(side=tk.LEFT, padx=5)
        ttk.Button(alert_frame, text="View Details", command=self._cmd_view_alert).pack(side=tk.LEFT, padx=5)
        ttk.Button(alert_frame, text="Keep", command=self._cmd_keep_alert).pack(side=tk.LEFT, padx=5)
        ttk.Button(alert_frame, text="Follow", command=self._cmd_follow_alert).pack(side=tk.LEFT, padx=5)
        ttk.Button(alert_frame, text="Kill", command=self._cmd_kill_alert).pack(side=tk.LEFT, padx=5)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: processes
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)
        proc_frame = ttk.LabelFrame(left_frame, text="Processes & Efficiency")
        proc_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.proc_tree = ttk.Treeview(
            proc_frame,
            columns=("pid", "cpu", "mem", "eff", "status", "mission"),
            show="headings"
        )
        for col, text in [
            ("pid", "PID"),
            ("cpu", "CPU %"),
            ("mem", "MEM %"),
            ("eff", "Efficiency"),
            ("status", "Status"),
            ("mission", "Mission"),
        ]:
            self.proc_tree.heading(col, text=text)
        self.proc_tree.column("pid", width=70, anchor=tk.CENTER)
        self.proc_tree.column("cpu", width=70, anchor=tk.CENTER)
        self.proc_tree.column("mem", width=70, anchor=tk.CENTER)
        self.proc_tree.column("eff", width=110, anchor=tk.CENTER)
        self.proc_tree.column("status", width=260, anchor=tk.W)
        self.proc_tree.column("mission", width=100, anchor=tk.CENTER)
        self.proc_tree.pack(fill=tk.BOTH, expand=True)

        # Right: network + log
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)

        net_frame = ttk.LabelFrame(right_frame, text="Network Resources (whitelist)")
        net_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        net_ctrl = ttk.Frame(net_frame)
        net_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(net_ctrl, text="Add Resource", command=self._on_add_resource).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(
            net_ctrl,
            text="Show watched only",
            variable=self.show_watched_only,
            command=self._on_toggle_show_watched,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(net_ctrl, text="Refresh Probe", command=self._cmd_refresh_network).pack(side=tk.LEFT, padx=3)

        self.net_tree = ttk.Treeview(
            net_frame,
            columns=("label", "path", "accessible", "top_count", "sample_bytes", "uses", "mission"),
            show="headings"
        )
        for col, text in [
            ("label", "Label"),
            ("path", "Path"),
            ("accessible", "Accessible"),
            ("top_count", "Top-level count"),
            ("sample_bytes", "Sample bytes"),
            ("uses", "Uses"),
            ("mission", "Mission"),
        ]:
            self.net_tree.heading(col, text=text, command=lambda c=col: self._on_net_heading_click(c))
        self.net_tree.column("label", width=140, anchor=tk.W)
        self.net_tree.column("path", width=320, anchor=tk.W)
        self.net_tree.column("accessible", width=80, anchor=tk.CENTER)
        self.net_tree.column("top_count", width=100, anchor=tk.CENTER)
        self.net_tree.column("sample_bytes", width=110, anchor=tk.CENTER)
        self.net_tree.column("uses", width=80, anchor=tk.CENTER)
        self.net_tree.column("mission", width=100, anchor=tk.CENTER)
        self.net_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.net_tree.tag_configure("hot", background="#ffefc6")
        self.net_tree.tag_configure("cold", background="#f0f0f0")
        self.net_tree.bind("<Button-3>", self._on_net_right_click)
        self.net_tree.bind("<Control-Button-1>", self._on_net_right_click)

        log_frame = ttk.LabelFrame(right_frame, text="Activity Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_widget = ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

    def _poll_queue(self):
        try:
            while True:
                msg = self.task_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _blink_alert(self):
        if self.alert_active:
            bg = self.alert_label.cget("bg")
            self.alert_label.configure(bg="red" if bg != "red" else "black")
        else:
            self.alert_label.configure(bg="black")
        self.root.after(500, self._blink_alert)

    def _handle_message(self, msg):
        mtype = msg.get("type")
        if mtype == "log":
            self._append_log(msg.get("text", ""))
        elif mtype == "status":
            self.status_var.set(msg.get("text", ""))
        elif mtype == "efficiency":
            self.efficiency_var.set(msg.get("text", ""))
        elif mtype == "inventory":
            txt = msg.get("text", "")
            self._append_log("\n--- SYSTEM INVENTORY ---\n" + txt + "\n-------------------------\n")
        elif mtype == "process_snapshot":
            self._update_process_table(msg.get("data", []))
        elif mtype == "time_context":
            self.time_var.set(msg.get("text", ""))
        elif mtype == "memory_summary":
            self.memory_var.set(msg.get("text", ""))
        elif mtype == "network_state":
            en = msg.get("enabled", True)
            self.network_state_var.set("Network: ON" if en else "Network: OFF")
            self.network_state_label.configure(foreground="green" if en else "red")
        elif mtype == "network_table":
            self._update_network_table(msg.get("data", {}))
        elif mtype == "alert":
            self.alert_var.set(msg.get("text", ""))
            self.alert_active = True
            self.last_alert_details = msg.get("details")

    def _append_log(self, text):
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)

    def _update_process_table(self, rows):
        self.proc_tree.delete(*self.proc_tree.get_children())
        for row in rows:
            self.proc_tree.insert("", tk.END, values=row)

    def _update_network_table(self, network_info):
        self.net_tree.delete(*self.net_tree.get_children())
        rows = []
        for label, info in network_info.items():
            path = info.get("path", "")
            accessible = info.get("accessible", False)
            top_count = info.get("top_level_count", "")
            sample_bytes = info.get("sample_bytes", "")
            uses = int(info.get("uses", 0))
            mission = "Yes" if info.get("mission_critical", False) else "No"
            rows.append({
                "label": label,
                "path": path,
                "accessible": "Yes" if accessible else "No",
                "top_count": str(top_count),
                "sample_bytes": str(sample_bytes),
                "uses": uses,
                "mission": mission,
            })

        if self.show_watched_only.get():
            rows = [r for r in rows if r["uses"] > 0]

        if self.net_sort_col:
            key = self.net_sort_col
            rev = self.net_sort_reverse
            if key == "uses":
                rows.sort(key=lambda r: r["uses"], reverse=rev)
            else:
                rows.sort(key=lambda r: r.get(key, ""), reverse=rev)

        for r in rows:
            tag = "hot" if r["uses"] >= HOT_USES_THRESHOLD or r["mission"] == "Yes" else "cold"
            self.net_tree.insert(
                "", tk.END,
                values=(
                    r["label"], r["path"], r["accessible"],
                    r["top_count"], r["sample_bytes"],
                    str(r["uses"]), r["mission"]
                ),
                tags=(tag,)
            )

    # Alert actions
    def _cmd_ack_alert(self):
        self.alert_var.set("No alerts")
        self.alert_active = False
        self.last_alert_details = None

    def _cmd_view_alert(self):
        if not self.last_alert_details:
            return
        win = tk.Toplevel(self.root)
        win.title("Alert Details")
        txt = ScrolledText(win, wrap=tk.WORD, width=100, height=30)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, self.last_alert_details)
        txt.configure(state=tk.DISABLED)

    def _cmd_keep_alert(self):
        if not self.last_alert_details:
            return
        self.control_queue.put({"cmd": "alert_keep", "details": self.last_alert_details})

    def _cmd_follow_alert(self):
        if not self.last_alert_details:
            return
        self.control_queue.put({"cmd": "alert_follow", "details": self.last_alert_details})

    def _cmd_kill_alert(self):
        if not self.last_alert_details:
            return
        self.control_queue.put({"cmd": "alert_kill", "details": self.last_alert_details})

    # Top bar commands
    def _cmd_toggle_network(self):
        self.control_queue.put({"cmd": "toggle_network"})

    def _cmd_run_autoload(self):
        self.control_queue.put({"cmd": "run_autoload"})

    def _cmd_inventory(self):
        self.control_queue.put({"cmd": "inventory"})

    def _cmd_start_brain(self):
        self.control_queue.put({"cmd": "start_brain"})

    def _cmd_stop_brain(self):
        self.control_queue.put({"cmd": "stop_brain"})

    # Network panel commands
    def _on_add_resource(self):
        dlg = AddResourceDialog(self.root)
        if not dlg.path:
            return
        path = dlg.path
        label = dlg.label or path
        read_only = dlg.read_only
        mission = dlg.mission_critical
        accessible = os.path.exists(path)
        lines = [
            f"Path: {path}",
            f"Label: {label}",
            f"Accessible now: {'Yes' if accessible else 'No'}",
            f"Mission-critical: {'Yes' if mission else 'No'}",
            "",
            "Add this resource to the whitelist?"
        ]
        if not messagebox.askyesno("Confirm Add Resource", "\n".join(lines), parent=self.root):
            self._append_log(f"[GUI] Add resource cancelled: {path}")
            return
        self.control_queue.put({
            "cmd": "add_network_resource",
            "path": path,
            "label": label,
            "read_only": read_only,
            "mission_critical": mission
        })

    def _on_toggle_show_watched(self):
        self.control_queue.put({"cmd": "refresh_network_table"})

    def _cmd_refresh_network(self):
        self.control_queue.put({"cmd": "refresh_network_probe"})

    def _on_net_right_click(self, event):
        iid = self.net_tree.identify_row(event.y)
        if not iid:
            return
        vals = self.net_tree.item(iid, "values")
        if not vals:
            return
        label, path = vals[0], vals[1]
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Probe Now", command=lambda p=path: self._probe_now(p))
        menu.add_command(label="Retire/Ignore Resource", command=lambda p=path, l=label: self._retire_resource(p, l))
        menu.add_separator()
        menu.add_command(label="Open in File Manager", command=lambda p=path: self._open_in_file_manager(p))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _probe_now(self, path):
        self.control_queue.put({"cmd": "probe_single_resource", "path": path})

    def _retire_resource(self, path, label):
        if not messagebox.askyesno(
            "Retire Resource",
            f"Retire and remove resource:\n{label}\n{path} ?",
            parent=self.root
        ):
            return
        self.control_queue.put({"cmd": "retire_network_resource", "path": path})

    def _open_in_file_manager(self, path):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showinfo("Open Path", f"Unable to open path: {e}", parent=self.root)

    def _on_net_heading_click(self, col):
        if self.net_sort_col == col:
            self.net_sort_reverse = not self.net_sort_reverse
        else:
            self.net_sort_col = col
            self.net_sort_reverse = False
        self.control_queue.put({"cmd": "refresh_network_table"})

# ==============================
# BRAIN
# ==============================

class ExoBrain(threading.Thread):
    def __init__(self, task_queue, control_queue):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.control_queue = control_queue
        self.autoloader = AutoLoader(REQUIRED_MODULES, self._log)
        self.running = True
        self.brain_active = False
        self.psutil = None

        self.memory = BrainMemory(self._log)
        self.memory.register_startup()
        self.rule_engine = OptimizationRuleEngine(self._log)
        self.network_manager = NetworkManager(self._log, self.memory)

        self._last_efficiency = None
        self._last_network_alert_hash = None
        self._latest_change_struct = None

    def run(self):
        self._log("[BRAIN] Control thread started.")
        self._set_status("Ready")
        self._send_network_state()
        self._send_network_table()
        while self.running:
            try:
                cmd = self.control_queue.get()
                if not isinstance(cmd, dict):
                    continue
                action = cmd.get("cmd")
                if action == "run_autoload":
                    self._handle_autoload()
                elif action == "inventory":
                    self._handle_inventory()
                elif action == "start_brain":
                    self._handle_start_brain()
                elif action == "stop_brain":
                    self._handle_stop_brain()
                elif action == "toggle_network":
                    self._handle_toggle_network()
                elif action == "add_network_resource":
                    self._handle_add_network_resource(cmd)
                elif action == "refresh_network_probe":
                    self._handle_refresh_network_probe()
                elif action == "refresh_network_table":
                    self._handle_refresh_network_table()
                elif action == "probe_single_resource":
                    self._handle_probe_single_resource(cmd.get("path"))
                elif action == "retire_network_resource":
                    self._handle_retire_network_resource(cmd.get("path"))
                elif action == "alert_keep":
                    self._handle_alert_keep()
                elif action == "alert_follow":
                    self._handle_alert_follow()
                elif action == "alert_kill":
                    self._handle_alert_kill()
            except Exception as e:
                self._log(f"[BRAIN] Control loop error: {e}")
                self._log(traceback.format_exc())

    # --- Control handlers ---

    def _handle_autoload(self):
        self._set_status("Autoloading modules...")
        self._log("[BRAIN] Autoloader started.")
        self.autoloader.ensure_all()
        self.psutil = self.autoloader.loaded_modules.get("psutil")
        if self.autoloader.errors:
            self._log(f"[BRAIN] Autoloader errors: {list(self.autoloader.errors.keys())}")
        else:
            self._log("[BRAIN] Autoloader complete.")
        self._set_status("Ready")

    def _handle_inventory(self):
        self._set_status("Inventory scan")
        txt = self._build_inventory_snapshot()
        self._send_inventory(txt)
        self._set_status("Ready")

    def _handle_start_brain(self):
        if self.brain_active:
            self._log("[BRAIN] Brain already running.")
            return
        if not self.psutil:
            self._log("[BRAIN] psutil missing. Run autoloader first.")
            return
        self.brain_active = True
        threading.Thread(target=self._brain_loop, daemon=True).start()
        self._log("[BRAIN] Brain loop started.")
        self._send_memory_summary(self.memory.get_summary_text())
        self._send_network_state()
        self._set_status("Brain running")

    def _handle_stop_brain(self):
        if not self.brain_active:
            self._log("[BRAIN] Brain not running.")
            return
        self.brain_active = False
        self.memory.save()
        self._log("[BRAIN] Brain stopping; memory saved.")
        self._set_status("Stopping brain...")

    def _handle_toggle_network(self):
        current = self.memory.get_network_enabled()
        self.memory.set_network_enabled(not current)
        self.memory.save()
        self._log(f"[BRAIN] Network toggled to {not current}")
        self._send_network_state()
        self._send_network_table()

    def _handle_add_network_resource(self, cmd):
        path = cmd.get("path")
        label = cmd.get("label")
        read_only = cmd.get("read_only", True)
        mission = cmd.get("mission_critical", False)
        ok, msg = self.network_manager.add_resource(path, label, read_only, mission)
        self._log(f"[BRAIN] Add resource: {msg}")
        self._handle_refresh_network_probe()

    def _handle_refresh_network_probe(self):
        self._log("[BRAIN] Probing resources...")
        probe_results = self.network_manager.probe_whitelisted_resources()
        self._send_network_table_with_probe(probe_results)
        snap = self.network_manager.build_snapshot(probe_results)
        self._maybe_handle_network_change(snap)

    def _handle_refresh_network_table(self):
        self._send_network_table()

    def _handle_probe_single_resource(self, path):
        if not path:
            return
        self._log(f"[BRAIN] Probing single: {path}")
        probe_results = self.network_manager.probe_whitelisted_resources(single_path=path)
        self._send_network_table_with_probe(probe_results)

    def _handle_retire_network_resource(self, path):
        if not path:
            return
        if self.network_manager.retire_resource(path):
            self._log(f"[BRAIN] Retired resource: {path}")
        else:
            self._log(f"[BRAIN] Retire failed: {path}")
        self._send_network_table()

    def _handle_alert_keep(self):
        if not self._latest_change_struct:
            return
        signature = json.dumps(self._latest_change_struct, sort_keys=True)
        sig_hash = hash(signature)
        self.memory.add_ignored_alert_signature(sig_hash)
        self.memory.save()
        self._log("[BRAIN] Alert marked Keep (ignored).")

    def _handle_alert_follow(self):
        if not self._latest_change_struct:
            return
        labels = list(self._latest_change_struct.get("resources_changed", {}).keys())
        if labels:
            self._log(f"[BRAIN] Follow requested for: {', '.join(labels)}")

    def _handle_alert_kill(self):
        if not self._latest_change_struct:
            return
        labels = list(self._latest_change_struct.get("resources_changed", {}).keys())
        if not labels:
            return
        removed = []
        for res in list(self.network_manager.resources):
            if res.get("label") in labels:
                removed.append(res.get("path"))
        for p in removed:
            self.network_manager.retire_resource(p)
        if removed:
            self._log(f"[BRAIN] Kill applied; retired: {', '.join(removed)}")
            self._send_network_table()

    # --- Inventory & time ---

    def _build_inventory_snapshot(self):
        try:
            lines = []
            uname = platform.uname()
            lines.append(f"System: {uname.system} {uname.release} ({uname.version})")
            lines.append(f"Node: {uname.node}")
            lines.append(f"Machine: {uname.machine}")
            lines.append(f"Processor: {uname.processor or 'Unknown'}")
            if self.psutil:
                lines.append(f"Physical cores: {self.psutil.cpu_count(logical=False)}")
                lines.append(f"Logical cores: {self.psutil.cpu_count(logical=True)}")
                vm = self.psutil.virtual_memory()
                lines.append(f"Memory total: {vm.total / (1024**3):.2f} GB")
                disk = self.psutil.disk_usage('/')
                lines.append(f"Disk / total: {disk.total / (1024**3):.2f} GB")
                procs = list(self.psutil.process_iter(attrs=["name", "pid"]))
                lines.append(f"Running processes: {len(procs)}")
            mounts = self.network_manager.list_local_mounts()
            lines.append(f"Local mounts: {len(mounts)}")
            lines.append(f"Whitelisted resources: {len(self.network_manager.resources)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Inventory error: {e}"

    def _get_time_context(self):
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        month = now.month
        is_weekend = weekday >= 5
        if month in (12, 1, 2):
            season = "winter"
        elif month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        else:
            season = "autumn"
        txt = (
            f"Time: {now:%Y-%m-%d %H:%M:%S} | Hour: {hour} | "
            f"Weekday: {weekday} | Month: {month} | Season: {season}"
        )
        self.task_queue.put({"type": "time_context", "text": txt})
        return {"now": now, "hour": hour, "weekday": weekday,
                "month": month, "is_weekend": is_weekend, "season": season}

    # --- Alerts ---

    def _maybe_handle_network_change(self, snap):
        old = self.memory.get_last_network_snapshot()
        changes = self.network_manager.diff_snapshots(old, snap)
        self.memory.set_last_network_snapshot(snap)
        if not self.network_manager.has_meaningful_change(changes):
            return
        self._latest_change_struct = changes
        signature = json.dumps(changes, sort_keys=True)
        sig_hash = hash(signature)
        if self.memory.is_alert_signature_ignored(sig_hash):
            self._log("[NETWORK] Change detected but ignored (Keep).")
            return
        if sig_hash == self._last_network_alert_hash:
            return
        self._last_network_alert_hash = sig_hash

        mission_changed = False
        for label, diff in changes.get("resources_changed", {}).items():
            res = self.network_manager.get_resource_by_label(label)
            if res and res.get("mission_critical"):
                mission_changed = True
                break

        parts = []
        if changes["mounts_added"]:
            parts.append(f"+Mounts: {', '.join(changes['mounts_added'])}")
        if changes["mounts_removed"]:
            parts.append(f"-Mounts: {', '.join(changes['mounts_removed'])}")
        if changes["resources_changed"]:
            labs = ", ".join(changes["resources_changed"].keys())
            parts.append(f"Resources changed: {labs}")
        summary = " | ".join(parts) if parts else "Network topology changed."
        if mission_changed:
            summary = "MISSION-CRITICAL CHANGE: " + summary

        detail_lines = ["Network change detected:"]
        if changes["mounts_added"]:
            detail_lines.append(f"  Mounts added: {', '.join(changes['mounts_added'])}")
        if changes["mounts_removed"]:
            detail_lines.append(f"  Mounts removed: {', '.join(changes['mounts_removed'])}")
        if changes["resources_changed"]:
            detail_lines.append("  Resources changed:")
            for label, diff in changes["resources_changed"].items():
                detail_lines.append(f"    - {label}:")
                detail_lines.append(f"        Old: {diff['old']}")
                detail_lines.append(f"        New: {diff['new']}")
        details_str = "\n".join(detail_lines)

        self._log(f"[NETWORK] Change detected: {summary}")
        self.task_queue.put({
            "type": "alert",
            "text": f"Network change: {summary}",
            "details": details_str
        })

    # --- Brain loop ---

    def _brain_loop(self):
        self._log("[BRAIN] Entering loop.")
        cycle = 0
        while self.brain_active:
            try:
                cycle += 1
                time_ctx = self._get_time_context()
                cpu = self.psutil.cpu_percent(interval=0.3)
                mem = self.psutil.virtual_memory().percent
                idle = cpu < IDLE_CPU_THRESHOLD
                mode = "OPTIMIZING (idle)" if idle else "PAUSED (busy)"
                eff = max(0.0, 100.0 - (cpu * 0.6 + mem * 0.4))
                self._send_efficiency(f"System Efficiency: {eff:5.1f} | Mode: {mode}")

                proc_rows = self._build_process_snapshot(idle)
                self._send_process_snapshot(proc_rows)

                network_enabled = self.memory.get_network_enabled()
                network_info = {}
                if network_enabled and self.network_manager.resources:
                    probe_results = self.network_manager.probe_whitelisted_resources()
                    self._send_network_table_with_probe(probe_results)
                    snap = self.network_manager.build_snapshot(probe_results)
                    self._maybe_handle_network_change(snap)
                    for res in self.network_manager.resources:
                        label = res.get("label", res.get("path"))
                        path = res.get("path")
                        info = probe_results.get(label, {})
                        mem_usage = self.memory.state.get("network_usage", {}).get(path, {})
                        info["path"] = path
                        info["uses"] = mem_usage.get("uses", 0)
                        info["mission_critical"] = res.get("mission_critical", False)
                        network_info[label] = info
                else:
                    for res in self.network_manager.resources:
                        label = res.get("label", res.get("path"))
                        path = res.get("path")
                        accessible = os.path.exists(path)
                        mem_usage = self.memory.state.get("network_usage", {}).get(path, {})
                        network_info[label] = {
                            "accessible": accessible,
                            "top_level_count": mem_usage.get("last_info", {}).get("top_level_count", ""),
                            "sample_bytes": mem_usage.get("last_info", {}).get("sample_bytes", ""),
                            "uses": mem_usage.get("uses", 0),
                            "path": path,
                            "mission_critical": res.get("mission_critical", False),
                        }

                decision = self.rule_engine.evaluate(
                    {"cpu": cpu, "mem": mem, "efficiency": eff},
                    proc_rows,
                    idle,
                    time_ctx,
                    network_info
                )

                if self._last_efficiency is not None:
                    delta = eff - self._last_efficiency
                    self.rule_engine.record_outcome(
                        self._last_efficiency,
                        eff,
                        decision.get("used_rule_ids", [])
                    )
                    for rid in decision.get("used_rule_ids", []):
                        self.memory.record_rule_outcome(rid, delta)
                self._last_efficiency = eff
                self.memory.record_efficiency(eff)

                for line in decision.get("explanations", []):
                    self._log(line)

                self._log(f"[BRAIN] Cycle {cycle}: CPU={cpu:.1f} MEM={mem:.1f} | {mode}")

                if cycle % 10 == 0:
                    self.memory.save()
                    self._send_memory_summary(self.memory.get_summary_text())

                hint = decision.get("interval_hint")
                if hint == "busy":
                    time.sleep(SCAN_INTERVAL_BUSY)
                else:
                    time.sleep(SCAN_INTERVAL_IDLE if idle else SCAN_INTERVAL_BUSY)

            except Exception as e:
                self._log(f"[BRAIN] Error in loop: {e}")
                self._log(traceback.format_exc())
                time.sleep(2)

        self.memory.save()
        self._set_status("Ready")
        self._log("[BRAIN] Loop halted.")

    def _build_process_snapshot(self, idle_mode):
        rows = []
        if not self.psutil:
            return rows
        try:
            for p in self.psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
                info = p.info
                pid = info.get("pid", 0)
                name = info.get("name") or f"PID {pid}"
                cpu = info.get("cpu_percent") or 0.0
                mem = info.get("memory_percent") or 0.0
                weight_factor = 1.0 if idle_mode else 1.5
                eff = max(0.0, 100.0 - weight_factor * (cpu * 0.7 + mem * 0.3))
                status = "Analyzing (idle)" if idle_mode else "Monitoring (busy)"
                lower_name = name.lower()
                mission = any(key.lower() in lower_name for key in CRITICAL_PROCESS_NAMES)
                mission_text = "Yes" if mission else "No"
                rows.append([
                    pid,
                    f"{cpu:.1f}",
                    f"{mem:.1f}",
                    f"{eff:.1f}",
                    f"{status} ({name})",
                    mission_text
                ])
            rows.sort(key=lambda r: float(r[1]), reverse=True)
        except Exception as e:
            self._log(f"[BRAIN] Process snapshot error: {e}")
        return rows

    # send helpers
    def _send_network_state(self):
        enabled = self.memory.get_network_enabled()
        self.task_queue.put({"type": "network_state", "enabled": enabled})

    def _send_network_table(self):
        network_info = {}
        for res in self.network_manager.resources:
            label = res.get("label", res.get("path"))
            path = res.get("path")
            accessible = os.path.exists(path)
            mem_usage = self.memory.state.get("network_usage", {}).get(path, {})
            network_info[label] = {
                "accessible": accessible,
                "top_level_count": mem_usage.get("last_info", {}).get("top_level_count", ""),
                "sample_bytes": mem_usage.get("last_info", {}).get("sample_bytes", ""),
                "uses": mem_usage.get("uses", 0),
                "path": path,
                "mission_critical": res.get("mission_critical", False),
            }
        self.task_queue.put({"type": "network_table", "data": network_info})

    def _send_network_table_with_probe(self, probe_results):
        network_info = {}
        for res in self.network_manager.resources:
            label = res.get("label", res.get("path"))
            path = res.get("path")
            info = probe_results.get(label, {})
            mem_usage = self.memory.state.get("network_usage", {}).get(path, {})
            info["path"] = path
            info["uses"] = mem_usage.get("uses", 0)
            info["mission_critical"] = res.get("mission_critical", False)
            network_info[label] = info
        self.task_queue.put({"type": "network_table", "data": network_info})

    def _send_efficiency(self, text):
        self.task_queue.put({"type": "efficiency", "text": text})

    def _send_inventory(self, text):
        self.task_queue.put({"type": "inventory", "text": text})

    def _send_process_snapshot(self, data):
        self.task_queue.put({"type": "process_snapshot", "data": data})

    def _send_memory_summary(self, text):
        self.task_queue.put({"type": "memory_summary", "text": text})

    def _log(self, text):
        self.task_queue.put({"type": "log", "text": text})

    def _set_status(self, text):
        self.task_queue.put({"type": "status", "text": text})

# ==============================
# MAIN
# ==============================

def main():
    # Mode A: force admin elevation on Windows before anything else.
    ensure_admin()

    task_queue = queue.Queue()
    control_queue = queue.Queue()

    brain = ExoBrain(task_queue, control_queue)
    brain.start()

    root = tk.Tk()
    gui = ExoGUI(root, task_queue, control_queue)

    task_queue.put({
        "type": "log",
        "text": "[MEMORY] On boot: " + brain.memory.get_summary_text()
    })
    task_queue.put({
        "type": "memory_summary",
        "text": brain.memory.get_summary_text()
    })

    control_queue.put({"cmd": "run_autoload"})
    control_queue.put({"cmd": "inventory"})
    control_queue.put({"cmd": "start_brain"})

    root.mainloop()

if __name__ == "__main__":
    main()

