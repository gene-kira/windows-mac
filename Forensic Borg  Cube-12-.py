import os
import sys
import time
import json
import threading
import hashlib
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import List, Callable, Any, Dict

# =========================
#  AUTOINSTALL HELPERS
# =========================

def autoload(lib):
    try:
        return __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        return __import__(lib)

psutil = autoload("psutil")

try:
    import winreg
except ImportError:
    winreg = None

try:
    import win32evtlog
    import win32con
except ImportError:
    win32evtlog = None
    win32con = None

try:
    import wmi
except ImportError:
    wmi = None

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog

# ETW placeholder flag (real providers can be wired later)
etw_available = False


# =========================
#  EVENT BUS + EVENTS
# =========================

@dataclass
class ScanResult:
    status_text: str
    mode: str
    input_pct: float
    system_pct: float
    output_pct: float
    anomaly_logs: List[str]
    forensic_explanations: List[str]
    threat_matrix: List[str]
    defense_recs: List[Dict]


@dataclass
class InfoEvent:
    message: str
    mode: str


class EventBus:
    def __init__(self):
        self._subscribers: List[Callable[[Any], None]] = []

    def subscribe(self, cb: Callable[[Any], None]):
        self._subscribers.append(cb)

    def publish(self, event: Any):
        for cb in list(self._subscribers):
            cb(event)


# =========================
#  CONFIG + STORAGE
# =========================

class Config:
    SCAN_EXCLUDED_DIRS = [
        r"C:\Windows\WinSxS",
        r"C:\Windows\SoftwareDistribution",
        r"C:\Windows\Temp",
        r"C:\ProgramData\Microsoft\Windows Defender",
    ]
    SCAN_EXCLUDED_PATTERNS = [
        r"\$Recycle.Bin",
        r"\System Volume Information",
    ]
    MAX_FILES_PER_SCAN = 50000

    GUI_MAX_LOG_LINES = 2000
    AUTOSCAN_INTERVAL_SEC = 30

    SCORE_LOW = 10
    SCORE_MEDIUM = 30
    SCORE_HIGH = 60
    SCORE_CRITICAL = 90

    ENABLE_DEFENSE_RECOMMENDER = True

    QUARANTINE_SUBDIR = "Quarantine"

    JOURNAL_MAX_SIZE_MB = 50

    MAX_LIST_ROWS = 30


class StorageManager:
    def __init__(self):
        self.base_root = self.detect_storage_root()
        self.base_dir = os.path.join(self.base_root, "ForensicBorg")
        self.dirs = {
            "Baseline": os.path.join(self.base_dir, "Baseline"),
            "Scenarios": os.path.join(self.base_dir, "Scenarios"),
            "AutoScan": os.path.join(self.base_dir, "AutoScan"),
            "BorgState": os.path.join(self.base_dir, "BorgState"),
            "Reports": os.path.join(self.base_dir, "Reports"),
            "Logs": os.path.join(self.base_dir, "Logs"),
            "Quarantine": os.path.join(self.base_dir, Config.QUARANTINE_SUBDIR),
            "Defense": os.path.join(self.base_dir, "Defense"),
            "Journal": os.path.join(self.base_dir, "Journal"),
        }
        self.ensure_dirs()
        self.max_age_days = 180

    def detect_storage_root(self):
        try:
            parts = psutil.disk_partitions(all=False)
        except Exception:
            parts = []

        smb_candidates = []
        local_candidates = []
        d_drive = None

        for p in parts:
            mp = p.mountpoint
            fstype = (p.fstype or "").lower()
            opts = (p.opts or "").lower()

            if "remote" in opts or "network" in opts or "cifs" in fstype:
                smb_candidates.append(mp)
            else:
                if len(mp) >= 2 and mp[1] == ":":
                    letter = mp[0].upper()
                    if letter == "D":
                        d_drive = mp
                    elif letter != "C":
                        local_candidates.append(mp)

        for root in smb_candidates:
            if self.test_writable(root):
                return root

        if d_drive and self.test_writable(d_drive):
            return d_drive

        for root in local_candidates:
            if self.test_writable(root):
                return root

        c_root = "C:\\"
        if self.test_writable(c_root):
            return c_root

        return os.getcwd()

    def test_writable(self, root):
        try:
            test_dir = os.path.join(root, "ForensicBorg_test")
            os.makedirs(test_dir, exist_ok=True)
            test_file = os.path.join(test_dir, "test.tmp")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False

    def ensure_dirs(self):
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def cleanup_old_files(self):
        cutoff = time.time() - self.max_age_days * 24 * 3600
        for d in self.dirs.values():
            try:
                for fname in os.listdir(d):
                    fpath = os.path.join(d, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime < cutoff:
                            os.remove(fpath)
                    except Exception:
                        pass
            except Exception:
                pass

    def timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_json(self, subdir_key, prefix, data):
        self.cleanup_old_files()
        ts = self.timestamp()
        d = self.dirs[subdir_key]
        fname = f"{prefix}_{ts}.json"
        path = os.path.join(d, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        return path

    def save_text(self, subdir_key, prefix, text):
        self.cleanup_old_files()
        ts = self.timestamp()
        d = self.dirs[subdir_key]
        fname = f"{prefix}_{ts}.txt"
        path = os.path.join(d, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        return path

    def defense_file_path(self, name):
        return os.path.join(self.dirs["Defense"], name)

    def journal_path(self):
        return os.path.join(self.dirs["Journal"], "journal.jsonl")

    def rotate_journal_if_needed(self):
        path = self.journal_path()
        if not os.path.exists(path):
            return
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > Config.JOURNAL_MAX_SIZE_MB:
                ts = self.timestamp()
                new_name = os.path.join(self.dirs["Journal"], f"journal_{ts}.jsonl")
                os.rename(path, new_name)
        except Exception:
            pass


# =========================
#  SNAPSHOT + DIFF HELPERS
# =========================

class FileHashCache:
    def __init__(self):
        self.cache = {}

    def get_hash(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (path, st.st_size, st.st_mtime)
        if key in self.cache:
            return self.cache[key]
        h = self.compute_hash(path)
        self.cache[key] = h
        return h

    @staticmethod
    def compute_hash(path, block_size=65536):
        sha = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    data = f.read(block_size)
                    if not data:
                        break
                    sha.update(data)
            return sha.hexdigest()
        except (PermissionError, FileNotFoundError, IsADirectoryError):
            return None


def is_excluded_path(path):
    p = os.path.abspath(path)
    for ex in Config.SCAN_EXCLUDED_DIRS:
        if p.lower().startswith(os.path.abspath(ex).lower()):
            return True
    for pat in Config.SCAN_EXCLUDED_PATTERNS:
        if pat.lower() in p.lower():
            return True
    return False


def snapshot_directory(root_path, hash_cache: FileHashCache):
    state = {}
    root_path = os.path.abspath(root_path)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        if is_excluded_path(dirpath):
            dirnames[:] = []
            continue
        rel_dir = os.path.relpath(dirpath, root_path)
        for d in dirnames:
            rel_path = os.path.normpath(os.path.join(rel_dir, d))
            state[rel_path] = {"type": "dir"}
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            if is_excluded_path(full_path):
                continue
            rel_path = os.path.normpath(os.path.join(rel_dir, f))
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = None
            file_hash = hash_cache.get_hash(full_path)
            state[rel_path] = {"type": "file", "size": size, "hash": file_hash}
            count += 1
            if count >= Config.MAX_FILES_PER_SCAN:
                return state
    return state


def snapshot_processes():
    procs = {}
    for p in psutil.process_iter(attrs=["pid", "name", "exe", "username", "ppid", "cmdline"]):
        info = p.info
        key = f"{info.get('pid')}:{info.get('name')}"
        procs[key] = {
            "pid": info.get("pid"),
            "name": info.get("name"),
            "exe": info.get("exe"),
            "user": info.get("username"),
            "ppid": info.get("ppid"),
            "cmdline": " ".join(info.get("cmdline") or []),
        }
    return procs


def snapshot_registry():
    if winreg is None:
        return {}
    snapshot = {}
    PERSISTENCE_KEYS = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options"),
    ]

    def walk_key(root, path, depth=0, max_depth=4):
        key_id = f"{root}:{path}"
        snapshot[key_id] = {"values": [], "subkeys": []}
        try:
            key = winreg.OpenKey(root, path)
        except OSError:
            return
        try:
            i = 0
            while True:
                name, value, vtype = winreg.EnumValue(key, i)
                snapshot[key_id]["values"].append(f"{name}={value}")
                i += 1
        except OSError:
            pass
        if depth >= max_depth:
            return
        try:
            i = 0
            while True:
                sub = winreg.EnumKey(key, i)
                snapshot[key_id]["subkeys"].append(sub)
                walk_key(root, path + "\\" + sub, depth + 1, max_depth)
                i += 1
        except OSError:
            pass

    for root, path in PERSISTENCE_KEYS:
        walk_key(root, path)
    return snapshot


def snapshot_network():
    conns = psutil.net_connections(kind="inet")
    summary = {"total": len(conns), "by_status": {}, "by_laddr": {}}
    for c in conns:
        st = str(c.status)
        summary["by_status"][st] = summary["by_status"].get(st, 0) + 1
        la = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "?"
        summary["by_laddr"][la] = summary["by_laddr"].get(la, 0) + 1
    return summary


def threat_score(action, path, score_label, context, notes, proc_info=None, net_info=None):
    score = 0
    p = path.lower()
    if "startup" in p or "runonce" in p or "winlogon" in p:
        score += 40
    if "appdata" in p or "programdata" in p or "\\temp" in p:
        score += 20
    if any(ext in p for ext in [".exe", ".dll", ".sys", ".ps1", ".vbs", ".js"]):
        score += 30
    if "services" in p or "currentcontrolset" in p:
        score += 35
    if context == "likely_system":
        score += 10
    if context == "unknown":
        score += 15
    if "script" in " ".join(notes).lower():
        score += 10
    if proc_info:
        cmd = (proc_info.get("cmdline") or "").lower()
        if "powershell" in cmd or "wscript" in cmd or "cscript" in cmd:
            score += 25
        if " -enc " in cmd or " -encodedcommand " in cmd:
            score += 30
    if net_info:
        if net_info.get("total", 0) > 200:
            score += 10
    if score >= Config.SCORE_CRITICAL:
        severity = "CRITICAL"
    elif score >= Config.SCORE_HIGH:
        severity = "HIGH"
    elif score >= Config.SCORE_MEDIUM:
        severity = "MEDIUM"
    elif score >= Config.SCORE_LOW:
        severity = "LOW"
    else:
        severity = "INFO"
    return score, severity


# =========================
#  REALTIME HOOKS
# =========================

class RealtimeHooks:
    def __init__(self, engine, bus: EventBus):
        self.engine = engine
        self.bus = bus
        self.running = False
        self.threads = []

    def start(self):
        self.running = True
        if win32evtlog is not None:
            t = threading.Thread(target=self.watch_security_log, daemon=True)
            t.start()
            self.threads.append(t)
        if wmi is not None:
            t2 = threading.Thread(target=self.watch_wmi_process, daemon=True)
            t2.start()
            self.threads.append(t2)
        if etw_available:
            # real ETW watcher would be wired here
            pass

    def stop(self):
        self.running = False

    def watch_security_log(self):
        try:
            server = "localhost"
            logtype = "Security"
            hand = win32evtlog.OpenEventLog(server, logtype)
            flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            offset = 0
            while self.running:
                events = win32evtlog.ReadEventLog(hand, flags, offset)
                if not events:
                    time.sleep(2)
                    continue
                for ev_obj in events:
                    msg = f"Security Event ID={ev_obj.EventID} Source={ev_obj.SourceName}"
                    self.engine.handle_realtime_event(
                        domain="SECURITY",
                        action="SEC_EVENT",
                        path=f"EventID={ev_obj.EventID}",
                        extra_notes=[msg]
                    )
                time.sleep(2)
        except Exception:
            self.bus.publish(InfoEvent(
                message="Security log watcher failed; staying in safe-mode (no blocking).",
                mode="SAFE"
            ))

    def watch_wmi_process(self):
        try:
            c = wmi.WMI()
            watcher = c.Win32_Process.watch_for("creation")
            while self.running:
                new_proc = watcher()
                name = new_proc.Name
                pid = new_proc.ProcessId
                exe = new_proc.ExecutablePath or ""
                cmd = new_proc.CommandLine or ""
                path = exe or name
                notes = [f"WMI process start: {name} PID={pid}", f"CMD={cmd}"]
                self.engine.handle_realtime_event(
                    domain="PROC",
                    action="PROC START",
                    path=path,
                    extra_notes=notes
                )
        except Exception:
            self.bus.publish(InfoEvent(
                message="WMI process watcher failed; staying in safe-mode (no blocking).",
                mode="SAFE"
            ))


# =========================
#  ENGINE
# =========================

class ForensicEngine:
    def __init__(self, storage: StorageManager, bus: EventBus):
        self.storage = storage
        self.bus = bus

        self.baseline_fs = None
        self.baseline_proc = None
        self.baseline_reg = None
        self.baseline_net = None

        self.scenario_snapshots = []
        self.borg_state = {}

        self.last_fs = None
        self.last_proc = None
        self.last_reg = None
        self.last_net = None

        self.hash_cache = FileHashCache()

        self.auto_interval_sec = Config.AUTOSCAN_INTERVAL_SEC
        self.auto_thread = None
        self.auto_running = False

        self.recent_events = []

        self.whitelist = set()
        self.blocklist = set()
        self.killlist = set()
        self.load_defense_lists()

        self.load_borg_state_if_exists()

        self.hooks = RealtimeHooks(self, bus)

    # ---- journaling ----

    def journal(self, entry: Dict):
        self.storage.rotate_journal_if_needed()
        path = self.storage.journal_path()
        entry["ts"] = datetime.now().isoformat()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ---- defense lists ----

    def load_defense_lists(self):
        def load_set(name):
            path = self.storage.defense_file_path(name)
            s = set()
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                s.add(line)
                except Exception:
                    pass
            return s

        self.whitelist = load_set("whitelist.txt")
        self.blocklist = load_set("blocklist.txt")
        self.killlist = load_set("killlist.txt")

    def save_defense_lists(self):
        def save_set(name, s):
            path = self.storage.defense_file_path(name)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for item in sorted(s):
                        f.write(item + "\n")
            except Exception:
                pass

        save_set("whitelist.txt", self.whitelist)
        save_set("blocklist.txt", self.blocklist)
        save_set("killlist.txt", self.killlist)

    def record_decision(self, decision_type, rec):
        path = rec.get("path", "")
        if decision_type == "allow":
            self.whitelist.add(path)
            self.blocklist.discard(path)
            self.killlist.discard(path)
        elif decision_type == "block":
            self.blocklist.add(path)
            self.whitelist.discard(path)
            self.killlist.discard(path)
        elif decision_type == "kill":
            self.killlist.add(path)
            self.whitelist.discard(path)
            self.blocklist.discard(path)
        self.save_defense_lists()
        self.journal({
            "type": "decision",
            "decision": decision_type,
            "path": path,
            "severity": rec.get("severity"),
            "score": rec.get("score"),
            "notes": rec.get("notes", []),
        })

    # ---- borg state ----

    def load_borg_state_if_exists(self):
        path = os.path.join(self.storage.dirs["BorgState"], "borg_state.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.borg_state = json.load(f)
            except Exception:
                self.borg_state = {}
        else:
            self.borg_state = {}

    def save_borg_state(self):
        path = os.path.join(self.storage.dirs["BorgState"], "borg_state.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.borg_state, f, indent=2)
        except Exception:
            pass

    # ---- scan result publish ----

    def publish_scan_result(self, status_text, mode,
                            input_pct, system_pct, output_pct,
                            anomaly_logs, forensic_explanations,
                            threat_matrix, defense_recs):
        self.bus.publish(ScanResult(
            status_text=status_text,
            mode=mode,
            input_pct=input_pct,
            system_pct=system_pct,
            output_pct=output_pct,
            anomaly_logs=anomaly_logs,
            forensic_explanations=forensic_explanations,
            threat_matrix=threat_matrix,
            defense_recs=defense_recs,
        ))

    # ---- baseline / scenario ----

    def capture_baseline(self, root_dir):
        self.bus.publish(InfoEvent("Capturing baseline snapshot...", "BASELINE"))
        fs = snapshot_directory(root_dir, self.hash_cache)
        proc = snapshot_processes()
        reg = snapshot_registry()
        net = snapshot_network()

        self.baseline_fs = fs
        self.baseline_proc = proc
        self.baseline_reg = reg
        self.baseline_net = net

        data = {
            "root_dir": root_dir,
            "fs": fs,
            "proc": proc,
            "reg": reg,
            "net": net,
        }
        self.storage.save_json("Baseline", "baseline", data)
        self.borg_state["baseline_root"] = root_dir
        self.save_borg_state()
        self.bus.publish(InfoEvent("Baseline captured and saved.", "BASELINE"))

    def run_scenario_and_capture(self, root_dir, label):
        self.bus.publish(InfoEvent(f"Capturing scenario: {label}", "SCENARIO"))
        fs = snapshot_directory(root_dir, self.hash_cache)
        proc = snapshot_processes()
        reg = snapshot_registry()
        net = snapshot_network()
        snap = {
            "label": label,
            "root_dir": root_dir,
            "fs": fs,
            "proc": proc,
            "reg": reg,
            "net": net,
            "timestamp": datetime.now().isoformat(),
        }
        self.scenario_snapshots.append(snap)
        self.storage.save_json("Scenarios", f"scenario_{label}", snap)
        self.bus.publish(InfoEvent(f"Scenario '{label}' captured.", "SCENARIO"))

    # ---- diff / analyze ----

    def analyze_diffs(self):
        if not self.baseline_fs:
            self.bus.publish(InfoEvent("No baseline set. Capture baseline first.", "WARN"))
            return
        if not self.scenario_snapshots:
            self.bus.publish(InfoEvent("No scenarios captured.", "WARN"))
            return

        anomaly_logs = []
        forensic_explanations = []
        threat_matrix = []
        defense_recs = []

        for snap in self.scenario_snapshots:
            label = snap["label"]
            anomaly_logs.append(f"=== Scenario: {label} ===")
            forensic_explanations.append(f"Scenario {label} vs baseline:")

            # FS diff (simple)
            fs_new = snap["fs"]
            added = set(fs_new.keys()) - set(self.baseline_fs.keys())
            removed = set(self.baseline_fs.keys()) - set(fs_new.keys())
            for path in sorted(added):
                entry = fs_new[path]
                if entry.get("type") == "file":
                    msg = f"[FS+]{path} (new file)"
                    anomaly_logs.append(msg)
                    notes = [msg]
                    score, sev = threat_score("FS_NEW", path, "fs", "unknown", notes)
                    threat_matrix.append(f"[{sev}] FS_NEW {path} (score={score})")
                    if Config.ENABLE_DEFENSE_RECOMMENDER and score >= Config.SCORE_MEDIUM:
                        defense_recs.append({
                            "action": "FS_NEW",
                            "path": path,
                            "score": score,
                            "severity": sev,
                            "notes": notes,
                        })
            for path in sorted(removed):
                msg = f"[FS-]{path} (removed)"
                anomaly_logs.append(msg)

        self.publish_scan_result(
            status_text="Diff analysis complete.",
            mode="ANALYZE",
            input_pct=50,
            system_pct=70,
            output_pct=80,
            anomaly_logs=anomaly_logs,
            forensic_explanations=forensic_explanations,
            threat_matrix=threat_matrix,
            defense_recs=defense_recs,
        )

    # ---- report ----

    def export_report(self):
        lines = []
        lines.append("Forensic Borg Report")
        lines.append("====================")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")
        lines.append(f"Baseline root: {self.borg_state.get('baseline_root', 'N/A')}")
        lines.append(f"Scenarios: {len(self.scenario_snapshots)}")
        text = "\n".join(lines)
        path = self.storage.save_text("Reports", "report", text)
        self.bus.publish(InfoEvent(f"Report exported to: {path}", "REPORT"))

    # ---- autoscan ----

    def start_auto_scan(self):
        if self.auto_thread is not None:
            return
        self.auto_running = True
        self.auto_thread = threading.Thread(target=self.auto_scan_loop, daemon=True)
        self.auto_thread.start()
        self.bus.publish(InfoEvent(
            "Auto-scan loop armed (every 30 seconds).",
            "AUTO"
        ))

    def stop_auto_scan(self):
        self.auto_running = False
        self.hooks.stop()

    def auto_scan_loop(self):
        while self.auto_running:
            try:
                self.scan_cycle_once()
            except Exception as e:
                self.bus.publish(InfoEvent(
                    f"Auto-scan error: {e}",
                    "ERROR"
                ))
            time.sleep(self.auto_interval_sec)

    def scan_cycle_once(self):
        root_dir = self.borg_state.get("baseline_root", "C:\\")
        fs = snapshot_directory(root_dir, self.hash_cache)
        proc = snapshot_processes()
        reg = snapshot_registry()
        net = snapshot_network()

        anomaly_logs = []
        forensic_explanations = []
        threat_matrix = []
        defense_recs = []

        if self.last_fs is not None:
            added = set(fs.keys()) - set(self.last_fs.keys())
            for path in sorted(added):
                entry = fs[path]
                if entry.get("type") == "file":
                    msg = f"[AUTO FS+]{path} (new file since last scan)"
                    anomaly_logs.append(msg)
                    notes = [msg]
                    score, sev = threat_score("AUTO_FS_NEW", path, "fs", "unknown", notes)
                    threat_matrix.append(f"[{sev}] AUTO_FS_NEW {path} (score={score})")
                    if Config.ENABLE_DEFENSE_RECOMMENDER and score >= Config.SCORE_MEDIUM:
                        defense_recs.append({
                            "action": "AUTO_FS_NEW",
                            "path": path,
                            "score": score,
                            "severity": sev,
                            "notes": notes,
                        })

        self.last_fs = fs
        self.last_proc = proc
        self.last_reg = reg
        self.last_net = net

        self.storage.save_json("AutoScan", "autoscan", {
            "root_dir": root_dir,
            "fs": fs,
            "proc": proc,
            "reg": reg,
            "net": net,
            "timestamp": datetime.now().isoformat(),
        })

        self.publish_scan_result(
            status_text=f"Auto-scan completed at {datetime.now().strftime('%H:%M:%S')}",
            mode="AUTO",
            input_pct=40,
            system_pct=60,
            output_pct=70,
            anomaly_logs=anomaly_logs,
            forensic_explanations=forensic_explanations,
            threat_matrix=threat_matrix,
            defense_recs=defense_recs,
        )

    # ---- realtime events ----

    def handle_realtime_event(self, domain, action, path, extra_notes=None):
        extra_notes = extra_notes or []
        anomaly_logs = []
        forensic_explanations = []
        threat_matrix = []
        defense_recs = []

        msg = f"[RT] {domain} {action} :: {path}"
        anomaly_logs.append(msg)
        forensic_explanations.append(msg)

        score, sev = threat_score(action, path, domain, "unknown", extra_notes)
        threat_matrix.append(f"[{sev}] {domain} {action} {path} (score={score})")

        if Config.ENABLE_DEFENSE_RECOMMENDER and score >= Config.SCORE_MEDIUM:
            defense_recs.append({
                "action": f"{domain}_{action}",
                "path": path,
                "score": score,
                "severity": sev,
                "notes": extra_notes,
            })

        self.publish_scan_result(
            status_text=f"Realtime event: {domain} {action}",
            mode="REALTIME",
            input_pct=30,
            system_pct=50,
            output_pct=80,
            anomaly_logs=anomaly_logs,
            forensic_explanations=forensic_explanations,
            threat_matrix=threat_matrix,
            defense_recs=defense_recs,
        )


# =========================
#  TKINTER COCKPIT
# =========================

class TkCockpit:
    def __init__(self, root, engine: ForensicEngine, bus: EventBus, storage: StorageManager):
        self.root = root
        self.engine = engine
        self.bus = bus
        self.storage = storage

        self.root.title("Forensic Borg • Tk Cockpit")
        self.root.geometry("1100x750")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_dashboard = tk.Frame(self.notebook)
        self.tab_threats = tk.Frame(self.notebook)
        self.tab_defense = tk.Frame(self.notebook)
        self.tab_decisions = tk.Frame(self.notebook)
        self.tab_scenarios = tk.Frame(self.notebook)
        self.tab_logs = tk.Frame(self.notebook)

        self.notebook.add(self.tab_dashboard, text="📊 Dashboard")
        self.notebook.add(self.tab_threats, text="⚠️ Threat Matrix")
        self.notebook.add(self.tab_defense, text="🛡 Defense")
        self.notebook.add(self.tab_decisions, text="✔ Decisions")
        self.notebook.add(self.tab_scenarios, text="🧪 Scenarios")
        self.notebook.add(self.tab_logs, text="📜 Logs")

        self._build_dashboard_tab()
        self._build_threats_tab()
        self._build_defense_tab()
        self._build_decisions_tab()
        self._build_scenarios_tab()
        self._build_logs_tab()

        self.root_dir = "C:\\"
        self.defense_rows = []

        bus.subscribe(self.on_event)

    # --- build tabs ---

    def _build_dashboard_tab(self):
        banner = tk.Label(
            self.tab_dashboard,
            text="Bloodhound | Sherlock | Rubik | Borg | Defense Recommender (Recommend-only, Non-Destructive)",
            font=("Consolas", 10),
            fg="#00ff88"
        )
        banner.pack(pady=4)

        self.label = tk.Label(self.tab_dashboard, text="Idle", font=("Consolas", 12))
        self.label.pack(pady=3)

        bars_frame = tk.Frame(self.tab_dashboard)
        bars_frame.pack(pady=3)

        tk.Label(bars_frame, text="Input", font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        self.input_bar = ttk.Progressbar(bars_frame, length=200, mode="determinate")
        self.input_bar.grid(row=0, column=1, padx=4)

        tk.Label(bars_frame, text="System", font=("Consolas", 9)).grid(row=1, column=0, sticky="w")
        self.system_bar = ttk.Progressbar(bars_frame, length=200, mode="determinate")
        self.system_bar.grid(row=1, column=1, padx=4)

        tk.Label(bars_frame, text="Output", font=("Consolas", 9)).grid(row=2, column=0, sticky="w")
        self.output_bar = ttk.Progressbar(bars_frame, length=200, mode="determinate")
        self.output_bar.grid(row=2, column=1, padx=4)

        self.mode_label = tk.Label(self.tab_dashboard, text="Mode: N/A", font=("Consolas", 10))
        self.mode_label.pack(pady=2)

        self.path_label = tk.Label(self.tab_dashboard, text="Root: C:\\", font=("Consolas", 9))
        self.path_label.pack(pady=2)

        btn_frame = tk.Frame(self.tab_dashboard)
        btn_frame.pack(pady=6)

        tk.Button(btn_frame, text="Choose Root", command=self.choose_directory).grid(row=0, column=0, padx=4)
        tk.Button(btn_frame, text="Capture Baseline", command=self.capture_baseline).grid(row=0, column=1, padx=4)
        tk.Button(btn_frame, text="Capture Attack Scenario", command=self.capture_scenario).grid(row=0, column=2, padx=4)
        tk.Button(btn_frame, text="Analyze Diffs", command=self.engine.analyze_diffs).grid(row=0, column=3, padx=4)
        tk.Button(btn_frame, text="Export Report", command=self.engine.export_report).grid(row=0, column=4, padx=4)
        tk.Button(btn_frame, text="Save Output", command=self.save_output_manual).grid(row=0, column=5, padx=4)

    def _build_threats_tab(self):
        top = tk.Frame(self.tab_threats)
        top.pack(fill="both", expand=True)
        tk.Label(top, text="Threat Matrix (max 30 rows):", font=("Consolas", 10)).pack(anchor="w")
        self.threat_list = tk.Listbox(top, height=Config.MAX_LIST_ROWS, width=120, font=("Consolas", 9))
        scroll = ttk.Scrollbar(top, orient="vertical", command=self.threat_list.yview)
        self.threat_list.configure(yscrollcommand=scroll.set)
        self.threat_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._bind_mousewheel(self.threat_list)

    def _build_defense_tab(self):
        container = tk.Frame(self.tab_defense)
        container.pack(fill="both", expand=True)
        tk.Label(container, text="Defense Recommendations (Allow / Block / Kill, max 30 rows):",
                 font=("Consolas", 10)).pack(anchor="w")
        self.defense_canvas = tk.Canvas(container)
        self.defense_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.defense_canvas.yview)
        self.defense_inner = tk.Frame(self.defense_canvas)
        self.defense_inner.bind(
            "<Configure>",
            lambda e: self.defense_canvas.configure(scrollregion=self.defense_canvas.bbox("all"))
        )
        self.defense_canvas.create_window((0, 0), window=self.defense_inner, anchor="nw")
        self.defense_canvas.configure(yscrollcommand=self.defense_scrollbar.set)
        self.defense_canvas.pack(side="left", fill="both", expand=True)
        self.defense_scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel(self.defense_canvas)

    def _build_decisions_tab(self):
        frame = tk.Frame(self.tab_decisions)
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        def build_list(parent, title):
            f = tk.Frame(parent)
            f.pack(side="left", fill="both", expand=True, padx=2)
            tk.Label(f, text=title, font=("Consolas", 10)).pack(anchor="w")
            lb = tk.Listbox(f, height=Config.MAX_LIST_ROWS, width=40, font=("Consolas", 9))
            scroll = ttk.Scrollbar(f, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=scroll.set)
            lb.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            self._bind_mousewheel(lb)
            return lb

        self.allowed_list = build_list(frame, "Allowed (max 30):")
        self.blocked_list = build_list(frame, "Blocked (recommended, max 30):")
        self.kill_list = build_list(frame, "Kill-Flagged (recommended, max 30):")

    def _build_scenarios_tab(self):
        top = tk.Frame(self.tab_scenarios)
        top.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Label(top, text="Scenarios (timeline, max 30):", font=("Consolas", 10)).pack(anchor="w")
        self.scenario_list = tk.Listbox(top, height=Config.MAX_LIST_ROWS, width=100, font=("Consolas", 9))
        scroll = ttk.Scrollbar(top, orient="vertical", command=self.scenario_list.yview)
        self.scenario_list.configure(yscrollcommand=scroll.set)
        self.scenario_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._bind_mousewheel(self.scenario_list)

    def _build_logs_tab(self):
        top = tk.Frame(self.tab_logs)
        top.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Label(top, text="Logs (auto-trim):", font=("Consolas", 10)).pack(anchor="w")
        self.log = tk.Text(top, height=30, width=120, font=("Consolas", 9))
        scroll = ttk.Scrollbar(top, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._bind_mousewheel(self.log)

    # --- mousewheel ---

    def _bind_mousewheel(self, widget):
        def _on_mousewheel(event):
            delta = -1 * int(event.delta / 120)
            widget.yview_scroll(delta, "units")
            return "break"
        widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", _on_mousewheel))
        widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))

    # --- UI actions ---

    def choose_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.root_dir = path
            self.path_label.config(text=f"Root: {path}")
            self._status("Root directory selected.", "IDLE")

    def capture_baseline(self):
        self.engine.capture_baseline(self.root_dir)

    def capture_scenario(self):
        label = simpledialog.askstring("Scenario label", "Name this scenario:")
        if not label:
            label = f"Scenario_{len(self.engine.scenario_snapshots)+1}"
        self.engine.run_scenario_and_capture(self.root_dir, label)
        self.scenario_list.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} :: {label}")
        self._trim_listbox(self.scenario_list)

    def save_output_manual(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Save Output Log"
        )
        if not path:
            return
        try:
            log_text = self.log.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(log_text)
            self._status(f"Output saved to: {path}", "SAVE")
        except Exception as e:
            self._status(f"Error saving output: {e}", "ERROR")

    # --- event handling from EventBus ---

    def on_event(self, event):
        if isinstance(event, ScanResult):
            self._status(event.status_text, event.mode)
            for line in event.anomaly_logs:
                self.log.insert(tk.END, line + "\n")
            for line in event.forensic_explanations:
                self.log.insert(tk.END, line + "\n")
            self.log.see(tk.END)
            self._trim_log()

            for tentry in event.threat_matrix:
                self.threat_list.insert(tk.END, tentry)
                self._trim_listbox(self.threat_list)

            for rec in event.defense_recs:
                self._add_defense_row(rec)

            self._update_bars(event.input_pct, event.system_pct, event.output_pct)

        elif isinstance(event, InfoEvent):
            self._status(event.message, event.mode)

    # --- helpers ---

    def _status(self, text, mode):
        self.mode_label.config(text=f"Mode: {mode}")
        self.label.config(text=text)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self._trim_log()

    def _update_bars(self, i, s, o):
        self.input_bar['value'] = max(0, min(100, i))
        self.system_bar['value'] = max(0, min(100, s))
        self.output_bar['value'] = max(0, min(100, o))

    def _trim_log(self):
        lines = int(self.log.index('end-1c').split('.')[0])
        if lines > Config.GUI_MAX_LOG_LINES:
            self.log.delete("1.0", f"{lines - Config.GUI_MAX_LOG_LINES}.0")

    def _trim_listbox(self, lb):
        while lb.size() > Config.MAX_LIST_ROWS:
            lb.delete(0)

    def _add_defense_row(self, rec):
        frame = tk.Frame(self.defense_inner, bd=1, relief="groove", padx=2, pady=2)
        label_text = f"[{rec['severity']}] {rec['action']} :: {rec['path']} (score={rec['score']})"
        if rec.get("notes"):
            label_text += f" | {', '.join(rec['notes'])}"
        lbl = tk.Label(frame, text=label_text, font=("Consolas", 8), anchor="w", justify="left", wraplength=900)
        lbl.pack(side="top", fill="x")

        btn_frame = tk.Frame(frame)
        btn_frame.pack(side="top", anchor="w", pady=2)

        def on_allow():
            self.engine.record_decision("allow", rec)
            self.allowed_list.insert(tk.END, label_text)
            self._trim_listbox(self.allowed_list)
            frame.destroy()
            self.defense_rows.remove(frame)

        def on_block():
            self.engine.record_decision("block", rec)
            self.blocked_list.insert(tk.END, label_text)
            self._trim_listbox(self.blocked_list)
            frame.destroy()
            self.defense_rows.remove(frame)

        def on_kill():
            self.engine.record_decision("kill", rec)
            self.kill_list.insert(tk.END, label_text)
            self._trim_listbox(self.kill_list)
            frame.destroy()
            self.defense_rows.remove(frame)

        tk.Button(btn_frame, text="Allow", command=on_allow, font=("Consolas", 8)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Block", command=on_block, font=("Consolas", 8)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Kill", command=on_kill, font=("Consolas", 8)).pack(side="left", padx=2)

        frame.pack(fill="x", pady=2)
        self.defense_rows.append(frame)
        if len(self.defense_rows) > Config.MAX_LIST_ROWS:
            oldest = self.defense_rows.pop(0)
            oldest.destroy()


# =========================
#  MAIN
# =========================

def main():
    storage = StorageManager()
    bus = EventBus()
    engine = ForensicEngine(storage, bus)

    root = tk.Tk()
    cockpit = TkCockpit(root, engine, bus, storage)

    engine.start_auto_scan()
    engine.hooks.start()

    root.mainloop()
    engine.stop_auto_scan()


if __name__ == "__main__":
    main()

