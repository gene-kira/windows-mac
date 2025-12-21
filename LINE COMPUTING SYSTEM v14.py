# ============================================================
# LINE COMPUTING SYSTEM v12
# Autonomous, predictive, adaptive, defensive, personalized, self-scaling guardian
#
# Adds on top of v11:
# - SelfScaler module:
#     * Monitors risk, event volume, CPU load
#     * Dynamically adjusts:
#         - Swarm node count
#         - System poll intervals
#         - Network/FS poll intervals
#         - Event generator rate (sim mode)
#     * Acts on "modes":
#         - CALM (scale down / relax)
#         - NORMAL (baseline)
#         - STRESSED (tighten)
#         - CRITICAL (max defense, max sensitivity)
# ============================================================

# -----------------------------
# AUTO-LOAD REQUIRED LIBRARIES
# -----------------------------
import importlib

required_libs = [
    "time", "logging", "json", "os", "random",
    "threading", "queue", "tkinter", "tkinter.scrolledtext",
    "subprocess", "socket", "math", "hashlib"
]

for lib in required_libs:
    parts = lib.split(".")
    module = importlib.import_module(lib)
    if len(parts) == 1:
        globals()[lib] = module
    else:
        globals()[parts[-1]] = module

# Optional psutil for deeper system hooks
try:
    psutil = importlib.import_module("psutil")
except ImportError:
    psutil = None

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
SETTINGS_FILE = "line_settings.json"
RISK_HISTORY_FILE = "risk_history.json"

SYMBOL_OK = "✔"
SYMBOL_BAD = "✖"
SYMBOL_SYNC = "⇆"
SYMBOL_THREAT = "⚠"
SYMBOL_WATCHDOG = "👁"
SYMBOL_SYS = "⭑"
SYMBOL_PROFILE = "◆"
SYMBOL_STRICT = "⛔"
SYMBOL_PRED = "♜"
SYMBOL_FOLD = "➕"
SYMBOL_FP = "⌘"
SYMBOL_NET = "🌐"
SYMBOL_FS = "📁"
SYMBOL_REPLAY = "▶"
SYMBOL_SCALE = "⚙"

# -----------------------------
# SETTINGS MEMORY
# -----------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "expected_sequence": ["LEFT", "RIGHT"],
        "profile_sequence": [],
        "profile_metadata": {},
        "swarm_nodes": 3,
        "event_interval": 0.5,
        "watchdog_timeout": 5.0,
        "system_poll_interval": 2.0,

        "process_whitelist": [
            "explorer.exe", "SearchApp.exe", "ShellExperienceHost.exe", "svchost.exe",
            "winlogon.exe", "csrss.exe", "wininit.exe", "dwm.exe", "lsass.exe",
            "services.exe", "smss.exe", "System", "RuntimeBroker.exe", "chrome.exe",
            "msedge.exe", "firefox.exe", "notepad.exe", "python.exe", "Code.exe",
            "cmd.exe", "conhost.exe", "Taskmgr.exe"
        ],

        "process_blacklist": [
            "mimikatz.exe", "mimikat.exe", "lsassdump.exe", "procdump.exe",
            "ransom.exe", "encryptor.exe", "keylogger.exe", "netcat.exe",
            "nc.exe", "meterpreter.exe"
        ],

        "process_policies": {
            "POWERSHELL.EXE": "alert",
            "PWSH.EXE": "alert",
            "CMD.EXE": "log",
            "PYTHON.EXE": "log",
            "NODE.EXE": "log",
            "RANSOM.EXE": "kill",
            "MIMIKATZ.EXE": "kill",
            "PROCDUMP.EXE": "alert",
            "NETCAT.EXE": "alert",
            "NC.EXE": "alert",
            "DISCORD.EXE": "log",
            "STEAM.EXE": "log",
            "WSL.EXE": "log"
        },

        "strict_mode": False,

        "predictive": {
            "risk_threshold_strict": 70,
            "risk_threshold_warn": 40,
            "operator_idle_seconds": 600
        },

        "folding": {
            "unknown_threshold": 5
        },

        "auto_start_system": True,
        "auto_start_profiling": True,

        "network_monitor": {
            "enabled": True,
            "poll_interval": 5.0,
            "bytes_threshold_per_interval": 10000000
        },

        "fs_watcher": {
            "enabled": False,
            "paths": [],
            "poll_interval": 5.0,
            "max_files_per_path": 5000
        },

        # Self-scaling configuration
        "self_scaler": {
            "enabled": True,
            "interval": 5.0,
            "max_swarm_nodes": 7,
            "min_swarm_nodes": 1,
            "min_poll_interval": 0.5,
            "max_poll_interval": 10.0,
            "min_event_interval": 0.1,
            "max_event_interval": 1.0
        }
    }

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except Exception:
        pass

settings = load_settings()

# -----------------------------
# EVENT BUS (PUB/SUB)
# -----------------------------
class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.lock = threading.Lock()

    def subscribe(self, event_type, callback):
        with self.lock:
            self.subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event_type, data=None):
        with self.lock:
            callbacks = list(self.subscribers.get(event_type, []))
        for cb in callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                logging.error(f"EventBus error: {e}")

event_bus = EventBus()

# -----------------------------
# LIVE LOG BUFFER
# -----------------------------
class LiveLog:
    def __init__(self, max_entries=2000):
        self.buffer = []
        self.max_entries = max_entries
        self.lock = threading.Lock()

    def add(self, level, msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} | {level} | {msg}"
        with self.lock:
            print(entry)
            self.buffer.append(entry)
            if len(self.buffer) > self.max_entries:
                self.buffer.pop(0)
        event_bus.publish("log", entry)

    def dump(self):
        with self.lock:
            return "\n".join(self.buffer)

log = LiveLog()

# -----------------------------
# RISK HISTORY + REPLAY
# -----------------------------
class RiskHistoryManager:
    def __init__(self):
        self.history = []
        self.lock = threading.Lock()

    def record(self, ts, event, source, is_anomaly, risk):
        with self.lock:
            self.history.append({
                "ts": ts,
                "event": event,
                "source": source,
                "is_anomaly": is_anomaly,
                "risk": risk
            })
            if len(self.history) > 20000:
                self.history.pop(0)

    def export(self, path=RISK_HISTORY_FILE):
        with self.lock:
            data = list(self.history)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            log.add("INFO", f"{SYMBOL_REPLAY} Risk history exported to {path}")
        except Exception as e:
            log.add("ERROR", f"{SYMBOL_REPLAY} Risk history export failed: {e}")

    def import_(self, path=RISK_HISTORY_FILE):
        if not os.path.exists(path):
            log.add("WARNING", f"{SYMBOL_REPLAY} Risk history file {path} not found.")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            with self.lock:
                self.history = data
            log.add("INFO", f"{SYMBOL_REPLAY} Risk history imported from {path}, entries={len(self.history)}")
        except Exception as e:
            log.add("ERROR", f"{SYMBOL_REPLAY} Risk history import failed: {e}")

    def get_history(self):
        with self.lock:
            return list(self.history)

risk_history = RiskHistoryManager()

class ReplayController:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self, history):
        with self.lock:
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run, args=(history,), daemon=True)
        self.thread.start()
        log.add("INFO", f"{SYMBOL_REPLAY} Replay started ({len(history)} events).")

    def stop(self):
        with self.lock:
            self.running = False
        log.add("INFO", f"{SYMBOL_REPLAY} Replay stop requested.")

    def _run(self, history):
        replay_engine = PredictiveEngine(settings.get("predictive", {}))
        last_ts = None
        for entry in history:
            with self.lock:
                if not self.running:
                    break
            ts = entry.get("ts", time.time())
            event = entry.get("event", "UNKNOWN")
            source = entry.get("source", "replay")
            is_anomaly = entry.get("is_anomaly", False)
            risk = replay_engine.observe_event(event, source, is_anomaly, behavior_score=0.0)
            log.add("INFO", f"{SYMBOL_REPLAY} Replay: {event} src={source} risk={risk:.1f}")
            if last_ts is not None:
                dt = max(0.01, min(0.5, ts - last_ts))
                time.sleep(dt)
            else:
                time.sleep(0.05)
            last_ts = ts

replay_controller = ReplayController()

# -----------------------------
# THREAT GLYPH RENDERER
# -----------------------------
class ThreatGlypher:
    def __init__(self):
        self.map = {
            "PROC_BLACKLISTED": "☠",
            "PROC_WHITELISTED": "⚙",
            "PROC_NEW": "❓",
            "PROC_STORM": "🌪",
            "CPU_SPIKE": "🔥",
            "CPU_DROP": "❄",
            "FS_CHANGE": "📄",
            "FS_NEW": "🆕",
            "FS_DEL": "🗑",
            "NET_SPIKE": "📡",
            "NET_CONN": "🔌"
        }

    def glyph_for(self, event):
        return self.map.get(event, SYMBOL_THREAT)

glypher = ThreatGlypher()

# -----------------------------
# THREAT MATRIX
# -----------------------------
class ThreatMatrix:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def record(self, event, expected, source="line", action=None, score=None):
        with self.lock:
            self.events.append({
                "event": event,
                "expected": expected,
                "source": source,
                "action": action,
                "score": score,
                "timestamp": time.time()
            })
        g = glypher.glyph_for(event)
        extra = ""
        if action:
            extra += f" action={action}"
        if score is not None:
            extra += f" score={score:.1f}"
        glyph = f"{g} [{source}] {event} (expected {expected}){extra}"
        event_bus.publish("threat", glyph)

    def summary(self):
        with self.lock:
            lines = []
            for e in self.events:
                g = glypher.glyph_for(e["event"])
                extra = ""
                if e.get("action"):
                    extra += f" action={e['action']}"
                if e.get("score") is not None:
                    extra += f" score={e['score']:.1f}"
                lines.append(
                    f"{g} [{e['source']}] {e['event']} (expected {e['expected']}){extra}"
                )
            return lines

threat_matrix = ThreatMatrix()

# -----------------------------
# PROCESS BEHAVIOR MODEL
# -----------------------------
class ProcessBehaviorModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = {}

    def observe_start(self, pid, name):
        name_u = (name or "UNKNOWN").upper()
        now = time.time()
        with self.lock:
            fp = self.stats.setdefault(name_u, {
                "starts": 0,
                "last_start": None,
                "avg_lifetime": None,
                "last_pid": None
            })
            fp["starts"] += 1
            fp["last_start"] = now
            fp["last_pid"] = pid

    def observe_end(self, pid, name):
        name_u = (name or "UNKNOWN").upper()
        now = time.time()
        with self.lock:
            fp = self.stats.get(name_u)
            if not fp:
                return
            if fp.get("last_pid") != pid or fp.get("last_start") is None:
                return
            lifetime = now - fp["last_start"]
            old = fp.get("avg_lifetime")
            if old is None:
                fp["avg_lifetime"] = lifetime
            else:
                fp["avg_lifetime"] = (old * 0.8) + (lifetime * 0.2)

    def compute_deviation_score(self, pid, name):
        if not psutil:
            return 0.0
        name_u = (name or "UNKNOWN").upper()
        now = time.time()
        with self.lock:
            fp = self.stats.get(name_u)
        if not fp:
            return 0.0
        score = 0.0
        try:
            p = psutil.Process(int(pid))
            cpu = p.cpu_percent(interval=0.0)
            if cpu > 50:
                score += min((cpu - 50) / 2.0, 20.0)
            start_ts = fp.get("last_start")
            avg_life = fp.get("avg_lifetime")
            if start_ts and avg_life:
                life_now = now - start_ts
                if life_now > avg_life * 3:
                    score += 10.0
                elif life_now < avg_life * 0.3:
                    score += 5.0
        except Exception:
            pass
        return score

proc_behavior = ProcessBehaviorModel()

# -----------------------------
# COMMAND-LINE ENTROPY
# -----------------------------
def shannon_entropy(s):
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent

def compute_cmd_entropy(pid):
    if not psutil:
        return 0.0
    try:
        p = psutil.Process(int(pid))
        cmd = " ".join(p.cmdline())
        ent = shannon_entropy(cmd)
        if ent > 4.5:
            return min((ent - 4.5) * 5.0, 20.0)
        return 0.0
    except Exception:
        return 0.0

# -----------------------------
# PREDICTIVE ENGINE
# -----------------------------
class PredictiveEngine:
    def __init__(self, config):
        self.lock = threading.Lock()
        self.temporal_events = {}
        self.anomaly_history = []
        self.operator_activity = {}
        self.current_risk = 0.0

        self.risk_threshold_warn = config.get("risk_threshold_warn", 40)
        self.risk_threshold_strict = config.get("risk_threshold_strict", 70)
        self.operator_idle_seconds = config.get("operator_idle_seconds", 600)

        self.last_operator_activity_ts = time.time()

    def set_operator_activity(self):
        with self.lock:
            self.last_operator_activity_ts = time.time()

    def is_operator_idle(self):
        with self.lock:
            return (time.time() - self.last_operator_activity_ts) > self.operator_idle_seconds

    def _get_hour_bucket(self, ts=None):
        if ts is None:
            ts = time.time()
        lt = time.localtime(ts)
        return lt.tm_hour

    def observe_event(self, event_type, source, is_anomaly=False, behavior_score=0.0):
        now = time.time()
        hour = self._get_hour_bucket(now)

        with self.lock:
            self.temporal_events.setdefault(hour, {})
            self.temporal_events[hour][event_type] = self.temporal_events[hour].get(event_type, 0) + 1
            self.operator_activity[hour] = self.operator_activity.get(hour, 0) + 1
            self.last_operator_activity_ts = now

            rarity_score = self._compute_rarity(event_type, hour)
            base = 30.0 if is_anomaly else 5.0
            delta = base + rarity_score + behavior_score
            tod_weight = self._time_of_day_weight(hour)
            delta *= tod_weight

            if self.is_operator_idle():
                delta *= 1.3

            if not is_anomaly:
                delta = -min(delta, 5.0)

            self.current_risk *= 0.97
            self.current_risk += delta
            self.current_risk = max(0.0, min(100.0, self.current_risk))

            if is_anomaly:
                self.anomaly_history.append((now, event_type, self.current_risk))
                if len(self.anomaly_history) > 2000:
                    self.anomaly_history.pop(0)

            risk = self.current_risk

        risk_history.record(now, event_type, source, is_anomaly, risk)
        event_bus.publish("predictive_risk", {"risk": risk})
        return risk

    def _compute_rarity(self, event_type, hour):
        hour_map = self.temporal_events.get(hour, {})
        total = sum(hour_map.values())
        if total == 0:
            return 5.0
        freq = hour_map.get(event_type, 0)
        p = freq / max(total, 1)
        if p > 0.2:
            return 0.0
        elif p > 0.05:
            return 5.0
        elif p > 0.01:
            return 10.0
        else:
            return 20.0

    def _time_of_day_weight(self, hour):
        total_activity = sum(self.operator_activity.values()) or 1
        this_activity = self.operator_activity.get(hour, 0)
        activity_ratio = this_activity / total_activity
        if activity_ratio < 0.02:
            return 2.0
        elif activity_ratio < 0.05:
            return 1.5
        elif activity_ratio < 0.15:
            return 1.2
        else:
            return 1.0

    def get_risk(self):
        with self.lock:
            return self.current_risk

    def classify_risk_level(self):
        r = self.get_risk()
        if r >= self.risk_threshold_strict:
            return "HIGH"
        elif r >= self.risk_threshold_warn:
            return "WARN"
        else:
            return "NORMAL"

predictive_engine = PredictiveEngine(settings.get("predictive", {}))

# -----------------------------
# LINE ENGINE
# -----------------------------
class LineComputer:
    def __init__(self, expected_sequence, strict_mode=False):
        self.expected = expected_sequence
        self.index = 0
        self.lock = threading.Lock()
        self.last_event_time = time.time()
        self.strict_mode = strict_mode

    def set_expected_sequence(self, sequence):
        with self.lock:
            self.expected = list(sequence)
            self.index = 0
        log.add("INFO", f"{SYMBOL_PROFILE} Expected sequence updated. Length={len(sequence)}")
        event_bus.publish("line_updated", {"expected_sequence": list(sequence)})

    def get_expected_sequence(self):
        with self.lock:
            return list(self.expected)

    def set_strict_mode(self, enabled: bool):
        with self.lock:
            self.strict_mode = enabled
        mode_str = "ON" if enabled else "OFF"
        log.add("INFO", f"{SYMBOL_STRICT} Strict mode {mode_str}.")
        settings["strict_mode"] = enabled
        save_settings(settings)

    def is_strict_mode(self):
        with self.lock:
            return self.strict_mode

    def process(self, event, source="line", behavior_score=0.0):
        with self.lock:
            if not self.expected:
                expected_event = None
            else:
                expected_event = self.expected[self.index]
            self.last_event_time = time.time()
            is_match = (expected_event is None or event == expected_event)

        risk = predictive_engine.observe_event(
            event_type=event,
            source=source,
            is_anomaly=not is_match,
            behavior_score=behavior_score
        )

        if is_match:
            log.add("INFO", f"{SYMBOL_OK} [{source}] OK: {event} is in line. (risk={risk:.1f})")
            with self.lock:
                if self.expected:
                    self.index = (self.index + 1) % len(self.expected)
            event_bus.publish("line_ok", {"event": event, "expected": expected_event, "source": source, "risk": risk})
            return True
        else:
            log.add("WARNING", f"{SYMBOL_BAD} [{source}] ANOMALY: {event} is NOT in line! Expected {expected_event}. (risk={risk:.1f})")
            event_bus.publish("line_anomaly", {"event": event, "expected": expected_event, "source": source, "risk": risk})
            return False

    def fold_event_into_line(self, event):
        with self.lock:
            if event in self.expected:
                log.add("INFO", f"{SYMBOL_FOLD} Event {event} already in line, skipping fold.")
                return
            insert_pos = self.index
            self.expected.insert(insert_pos, event)
            log.add("INFO", f"{SYMBOL_FOLD} Folded event {event} into line at position {insert_pos}. New length={len(self.expected)}")
            saved_seq = list(self.expected)

        settings["expected_sequence"] = saved_seq
        save_settings(settings)
        event_bus.publish("line_updated", {"expected_sequence": saved_seq})

    def get_last_event_age(self):
        with self.lock:
            return time.time() - self.last_event_time

# -----------------------------
# FOLD MANAGER
# -----------------------------
class FoldManager:
    def __init__(self, engine: LineComputer, unknown_threshold=5):
        self.engine = engine
        self.unknown_threshold = unknown_threshold
        self.lock = threading.Lock()
        self.counts = {}
        self.last_anomaly = None
        event_bus.subscribe("line_anomaly", self._on_line_anomaly)

    def _make_key(self, event, source):
        return f"{event}|{source}"

    def _on_line_anomaly(self, event_type, data):
        if not data:
            return
        event = data.get("event")
        source = data.get("source")
        risk = data.get("risk", 0.0)
        with self.lock:
            self.last_anomaly = (event, source, risk)

    def register_anomaly_for_threshold(self, event, source, risk, is_whitelisted, is_blacklisted):
        if is_blacklisted:
            return
        if risk >= predictive_engine.risk_threshold_strict:
            return
        if is_whitelisted:
            log.add("INFO", f"{SYMBOL_FOLD} Auto-folding whitelisted event {event} from {source}.")
            self.engine.fold_event_into_line(event)
            return

        key = self._make_key(event, source)
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            c = self.counts[key]

        if c >= self.unknown_threshold:
            log.add("INFO", f"{SYMBOL_FOLD} Threshold reached for event {event} from {source} (count={c}), folding into line.")
            self.engine.fold_event_into_line(event)
        else:
            log.add("INFO", f"{SYMBOL_FOLD} Counting anomaly {event} from {source}: {c}/{self.unknown_threshold} before fold.")

    def manual_fold_last(self):
        with self.lock:
            la = self.last_anomaly
        if not la:
            log.add("INFO", f"{SYMBOL_FOLD} No last anomaly to fold.")
            return
        event, source, risk = la
        log.add("INFO", f"{SYMBOL_FOLD} MANUAL fold requested for last anomaly {event} from {source} (risk={risk:.1f}).")
        self.engine.fold_event_into_line(event)

fold_manager = None

# -----------------------------
# PROFILING MANAGER
# -----------------------------
class ProfileManager:
    def __init__(self):
        self.recording = False
        self.sequence = []
        self.lock = threading.Lock()
        self.start_time = None
        event_bus.subscribe("sys_raw_event", self._on_sys_raw_event)

    def _on_sys_raw_event(self, event_type, data):
        if not self.recording:
            return
        if not data:
            return
        event_name = data.get("event")
        if not event_name:
            return
        with self.lock:
            if not self.sequence or self.sequence[-1] != event_name:
                self.sequence.append(event_name)

    def start(self):
        with self.lock:
            self.sequence = []
            self.recording = True
            self.start_time = time.time()
        log.add("INFO", f"{SYMBOL_PROFILE} Profiling started. Learning normal system line.")

    def stop(self):
        with self.lock:
            self.recording = False
            start = self.start_time
            self.start_time = None
        if start:
            duration = time.time() - start
            log.add("INFO", f"{SYMBOL_PROFILE} Profiling stopped. Collected {len(self.sequence)} steps over {duration:.1f}s.")
        else:
            log.add("INFO", f"{SYMBOL_PROFILE} Profiling stopped.")

    def lock_profile_into_engine(self, engine: LineComputer):
        with self.lock:
            learned = list(self.sequence)
            start = self.start_time
        if not learned:
            log.add("WARNING", f"{SYMBOL_PROFILE} No profile sequence to lock. Skipping.")
            return

        engine.set_expected_sequence(learned)

        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "duration_seconds": None,
            "notes": "Baseline profile locked from system events."
        }
        if start:
            metadata["duration_seconds"] = time.time() - start

        settings["expected_sequence"] = learned
        settings["profile_sequence"] = learned
        settings["profile_metadata"] = metadata
        save_settings(settings)

        log.add("INFO", f"{SYMBOL_PROFILE} Profile locked. Engine now using learned system line.")
        log.add("INFO", f"{SYMBOL_PROFILE} Profile metadata: host={metadata['host']} created_at={metadata['created_at']}")

    def get_sequence(self):
        with self.lock:
            return list(self.sequence)

    def get_metadata(self):
        return settings.get("profile_metadata", {})

profile_manager = ProfileManager()

# -----------------------------
# SWARM
# -----------------------------
class SwarmNode:
    def __init__(self, node_id, sequence):
        self.node_id = node_id
        self.sequence = list(sequence)
        self.lock = threading.Lock()
        self.shared_risk = 0.0
        self.last_update_ts = time.time()

    def sync(self, other_node):
        with self.lock, other_node.lock:
            merged = list(dict.fromkeys(self.sequence + other_node.sequence))
            self.sequence = merged
            other_node.sequence = merged
            avg_risk = (self.shared_risk + other_node.shared_risk) / 2.0
            self.shared_risk = avg_risk
            other_node.shared_risk = avg_risk
            self.last_update_ts = time.time()
            other_node.last_update_ts = time.time()

        msg = f"{SYMBOL_SYNC} Node {self.node_id} synced with Node {other_node.node_id} (shared_risk={self.shared_risk:.1f})"
        log.add("INFO", msg)
        event_bus.publish("swarm_sync", msg)

class SwarmManager:
    def __init__(self, node_count, base_sequence):
        self.lock = threading.Lock()
        self.nodes = [SwarmNode(i + 1, base_sequence) for i in range(node_count)]
        self.running = False
        self.thread = None
        self.sync_interval = 2.0

        event_bus.subscribe("line_updated", self._on_line_updated)

    def _on_line_updated(self, event_type, data):
        if not data:
            return
        new_seq = data.get("expected_sequence") or []
        with self.lock:
            for node in self.nodes:
                with node.lock:
                    node.sequence = list(new_seq)

    def start(self, interval=2.0):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.sync_interval = interval
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.add("INFO", "Swarm manager started.")

    def stop(self):
        with self.lock:
            self.running = False
        log.add("INFO", "Swarm manager stopped.")

    def _run(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                interval = self.sync_interval
                nodes_copy = list(self.nodes)
            time.sleep(interval)
            if len(nodes_copy) < 2:
                continue
            import random
            a, b = random.sample(nodes_copy, 2)
            a.sync(b)
            current_risk = predictive_engine.get_risk()
            for n in nodes_copy:
                with n.lock:
                    n.shared_risk = (n.shared_risk * 0.8) + (current_risk * 0.2)

    def get_status_lines(self):
        lines = []
        now = time.time()
        with self.lock:
            nodes_copy = list(self.nodes)
        for node in nodes_copy:
            with node.lock:
                seq_str = ",".join(node.sequence)
                risk = node.shared_risk
                age = now - node.last_update_ts
            lines.append(f"Node {node.node_id}: [{seq_str}] shared_risk={risk:.1f} last_sync={age:.1f}s ago")
        return lines

    def get_node_count(self):
        with self.lock:
            return len(self.nodes)

    def set_node_count(self, new_count):
        with self.lock:
            current = len(self.nodes)
            if new_count == current or new_count <= 0:
                return
            if new_count > current:
                base_seq = self.nodes[0].sequence if self.nodes else []
                for i in range(current, new_count):
                    self.nodes.append(SwarmNode(i + 1, base_seq))
                log.add("INFO", f"{SYMBOL_SCALE} Swarm scaled up to {new_count} nodes.")
            else:
                self.nodes = self.nodes[:new_count]
                log.add("INFO", f"{SYMBOL_SCALE} Swarm scaled down to {new_count} nodes.")

    def set_interval(self, interval):
        with self.lock:
            self.sync_interval = interval

# -----------------------------
# WATCHDOG
# -----------------------------
class Watchdog:
    def __init__(self, engine: LineComputer, timeout=5.0, check_interval=1.0):
        self.engine = engine
        self.timeout = timeout
        self.check_interval = check_interval
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.add("INFO", f"{SYMBOL_WATCHDOG} Watchdog started (timeout={self.timeout}s).")

    def stop(self):
        self.running = False
        log.add("INFO", f"{SYMBOL_WATCHDOG} Watchdog stopped.")

    def _run(self):
        while self.running:
            time.sleep(self.check_interval)
            age = self.engine.get_last_event_age()
            if age > self.timeout:
                msg = f"{SYMBOL_WATCHDOG} Watchdog: no events for {age:.1f}s – potential stall."
                log.add("WARNING", msg)
                event_bus.publish("watchdog_alert", msg)

# -----------------------------
# EVENT GENERATOR (SIM)
# -----------------------------
class EventGenerator:
    def __init__(self, engine: LineComputer, interval=0.5, anomaly_rate=0.2):
        self.engine = engine
        self.interval = interval
        self.anomaly_rate = anomaly_rate
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.valid_events = ["LEFT", "RIGHT"]
        self.anomaly_events = ["ROCK", "BRANCH", "GLITCH"]

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.add("INFO", "Event generator started.")

    def stop(self):
        with self.lock:
            self.running = False
        log.add("INFO", "Event generator stopped.")

    def set_interval(self, interval):
        with self.lock:
            self.interval = interval
        log.add("INFO", f"{SYMBOL_SCALE} Event generator interval set to {interval:.2f}s.")

    def _run(self):
        import random
        while True:
            with self.lock:
                if not self.running:
                    break
                interval = self.interval
            time.sleep(interval)
            if random.random() < self.anomaly_rate:
                event = random.choice(self.anomaly_events)
                source = "sim-anomaly"
            else:
                event = random.choice(self.valid_events)
                source = "sim"
            self.engine.process(event, source=source)

# -----------------------------
# POLICY ENGINE
# -----------------------------
class ProcessPolicyEngine:
    def __init__(self, engine: LineComputer, policy_map):
        self.engine = engine
        self.policy_map = {k.upper(): v for k, v in (policy_map or {}).items()}

    def decide_action(self, name: str, default_action: str):
        key = (name or "").upper()
        return self.policy_map.get(key, default_action)

    def apply_action(self, pid: str, name: str, action: str, event_type: str, source: str,
                     is_whitelisted: bool, is_blacklisted: bool):
        behavior_score = 0.0
        if psutil and pid not in (None, "?", ""):
            behavior_score += proc_behavior.compute_deviation_score(pid, name)
            behavior_score += compute_cmd_entropy(pid)
            if behavior_score > 0:
                log.add("INFO", f"{SYMBOL_FP} Behavior+CLI deviation for {name} (pid={pid}) score={behavior_score:.1f}")

        match = self.engine.process(event_type, source=source, behavior_score=behavior_score)
        risk = predictive_engine.get_risk()

        if fold_manager is not None:
            fold_manager.register_anomaly_for_threshold(
                event=event_type,
                source=source,
                risk=risk,
                is_whitelisted=is_whitelisted or (action == "allow"),
                is_blacklisted=is_blacklisted
            )

        if risk >= predictive_engine.risk_threshold_strict and not self.engine.is_strict_mode():
            log.add("WARNING", f"{SYMBOL_PRED} Risk {risk:.1f} >= strict threshold, enabling strict mode.")
            self.engine.set_strict_mode(True)
        elif risk < predictive_engine.risk_threshold_warn and self.engine.is_strict_mode():
            log.add("INFO", f"{SYMBOL_PRED} Risk {risk:.1f} below warn threshold, strict mode can be relaxed (manual).")

        if action == "allow":
            action_effective = "log"
        else:
            action_effective = action

        if not self.engine.is_strict_mode():
            if action_effective == "alert":
                log.add("WARNING", f"{SYMBOL_STRICT} Policy ALERT on process {name} (pid={pid}) event={event_type}, risk={risk:.1f}")
            elif action_effective == "kill":
                log.add("WARNING", f"{SYMBOL_STRICT} Policy would KILL {name} (pid={pid}) [strict off, no kill], risk={risk:.1f}")
            threat_matrix.record(event_type, None if match else "LINE_MISMATCH",
                                 source=source, action=action_effective, score=risk)
            return

        if action_effective == "ignore":
            return

        if action_effective == "log":
            threat_matrix.record(event_type, None if match else "LINE_MISMATCH",
                                 source=source, action=action_effective, score=risk)
            return

        if action_effective == "alert":
            log.add("WARNING", f"{SYMBOL_STRICT} STRICT ALERT on process {name} (pid={pid}) event={event_type}, risk={risk:.1f}")
            threat_matrix.record(event_type, None if match else "LINE_MISMATCH",
                                 source=source, action=action_effective, score=risk)
            return

        if action_effective == "kill":
            killed = False
            if psutil:
                try:
                    pid_int = int(pid)
                    p = psutil.Process(pid_int)
                    p.terminate()
                    killed = True
                    log.add("WARNING", f"{SYMBOL_STRICT} STRICT KILL executed on {name} (pid={pid}) risk={risk:.1f}")
                except Exception as e:
                    log.add("ERROR", f"{SYMBOL_STRICT} Failed to kill {name} (pid={pid}): {e}")
            else:
                log.add("WARNING", f"{SYMBOL_STRICT} STRICT KILL requested for {name} (pid={pid}), but psutil not available. risk={risk:.1f}")
            threat_matrix.record(event_type, None if match else "LINE_MISMATCH",
                                 source=source,
                                 action=("kill-ok" if killed else "kill-failed"),
                                 score=risk)

# -----------------------------
# SYSTEM EVENT COLLECTOR
# -----------------------------
class SystemEventCollector:
    def __init__(self, engine: LineComputer, policy_engine: ProcessPolicyEngine,
                 poll_interval=2.0,
                 whitelist=None, blacklist=None):
        self.engine = engine
        self.policy_engine = policy_engine
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None

        self.known_procs = set()
        self.cpu_baseline = None

        self.whitelist = set((whitelist or []))
        self.blacklist = set((blacklist or []))

        net_cfg = settings.get("network_monitor", {})
        self.net_enabled = net_cfg.get("enabled", False)
        self.net_poll_interval = net_cfg.get("poll_interval", 5.0)
        self.net_bytes_threshold = net_cfg.get("bytes_threshold_per_interval", 10000000)
        self.last_net_check = time.time()
        self.last_net_bytes = None

        fs_cfg = settings.get("fs_watcher", {})
        self.fs_enabled = fs_cfg.get("enabled", False)
        self.fs_paths = fs_cfg.get("paths", [])
        self.fs_poll_interval = fs_cfg.get("poll_interval", 5.0)
        self.fs_max_files = fs_cfg.get("max_files_per_path", 5000)
        self.last_fs_check = time.time()
        self.fs_snapshots = {}

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.add("INFO", f"{SYMBOL_SYS} System collector started (interval={self.poll_interval}s).")

    def stop(self):
        self.running = False
        log.add("INFO", f"{SYMBOL_SYS} System collector stopped.")

    def set_poll_interval(self, interval):
        self.poll_interval = interval
        log.add("INFO", f"{SYMBOL_SCALE} System poll interval set to {interval:.2f}s.")

    def set_net_poll_interval(self, interval):
        self.net_poll_interval = interval
        log.add("INFO", f"{SYMBOL_SCALE} Network poll interval set to {interval:.2f}s.")

    def set_fs_poll_interval(self, interval):
        self.fs_poll_interval = interval
        log.add("INFO", f"{SYMBOL_SCALE} FS poll interval set to {interval:.2f}s.")

    def _run(self):
        self.known_procs = self._get_current_proc_keys()
        self.cpu_baseline = self._get_cpu_load()
        while self.running:
            time.sleep(self.poll_interval)
            self._check_processes()
            self._check_cpu()
            self._maybe_check_network()
            self._maybe_check_filesystem()

    def _get_current_proc_keys(self):
        keys = set()
        if psutil:
            for p in psutil.process_iter(["pid", "name"]):
                pid = p.info.get("pid")
                name = p.info.get("name") or "UNKNOWN"
                keys.add(f"{pid}:{name}")
        else:
            try:
                if os.name == "nt":
                    out = subprocess.check_output(["tasklist"])
                    for line in out.splitlines()[3:]:
                        parts = line.decode(errors="ignore").split()
                        if len(parts) >= 2:
                            name, pid = parts[0], parts[1]
                            keys.add(f"{pid}:{name}")
                else:
                    out = subprocess.check_output(["ps", "-eo", "pid,comm"])
                    for line in out.splitlines()[1:]:
                        parts = line.decode(errors="ignore").split(None, 1)
                        if len(parts) == 2:
                            pid, name = parts
                            keys.add(f"{pid}:{name}")
            except Exception:
                pass
        return keys

    def _get_cpu_load(self):
        if psutil:
            return psutil.cpu_percent(interval=None)
        else:
            return None

    def _check_processes(self):
        current = self._get_current_proc_keys()
        new_procs = current - self.known_procs
        dead_procs = self.known_procs - current
        self.known_procs = current

        for key in new_procs:
            pid, name = self._split_proc_key(key)
            if psutil:
                proc_behavior.observe_start(pid, name)

        for key in dead_procs:
            pid, name = self._split_proc_key(key)
            if psutil:
                proc_behavior.observe_end(pid, name)

        for key in new_procs:
            pid, name = self._split_proc_key(key)
            base_name_upper = (name or "").upper()
            wl_match = any(base_name_upper == n.upper() for n in self.whitelist)
            bl_match = any(base_name_upper == n.upper() for n in self.blacklist)

            if bl_match:
                event_type = "PROC_BLACKLISTED"
                default_action = "kill"
                source = f"sys-proc-bl:{name}"
            elif wl_match:
                event_type = "PROC_WHITELISTED"
                default_action = "allow"
                source = f"sys-proc-wl:{name}"
            else:
                event_type = "PROC_NEW"
                default_action = "alert" if self.engine.is_strict_mode() else "log"
                source = f"sys-proc:{name}"

            raw_payload = {"event": event_type, "name": name, "pid": pid, "source": source}
            event_bus.publish("sys_raw_event", raw_payload)

            action = self.policy_engine.decide_action(name, default_action)
            self.policy_engine.apply_action(pid, name, action, event_type, source,
                                            is_whitelisted=wl_match,
                                            is_blacklisted=bl_match)

        if len(dead_procs) > 0 and len(dead_procs) > 10:
            event_type = "PROC_STORM"
            raw_payload = {"event": event_type, "count": len(dead_procs), "source": "sys-proc-storm"}
            event_bus.publish("sys_raw_event", raw_payload)
            self.engine.process(event_type, source="sys-proc-storm")

    def _check_cpu(self):
        current = self._get_cpu_load()
        if current is None:
            return
        if self.cpu_baseline is None:
            self.cpu_baseline = current
            return

        if current > self.cpu_baseline + 50:
            event_type = "CPU_SPIKE"
            raw_payload = {"event": event_type, "value": current, "source": "sys-cpu"}
            event_bus.publish("sys_raw_event", raw_payload)
            self.engine.process(event_type, source="sys-cpu")
        elif current < self.cpu_baseline - 30:
            event_type = "CPU_DROP"
            raw_payload = {"event": event_type, "value": current, "source": "sys-cpu"}
            event_bus.publish("sys_raw_event", raw_payload)
            self.engine.process(event_type, source="sys-cpu")

    def _maybe_check_network(self):
        if not self.net_enabled or not psutil:
            return
        now = time.time()
        if now - self.last_net_check < self.net_poll_interval:
            return
        self.last_net_check = now
        counters = psutil.net_io_counters()
        if self.last_net_bytes is None:
            self.last_net_bytes = counters.bytes_sent + counters.bytes_recv
            return
        current_bytes = counters.bytes_sent + counters.bytes_recv
        delta = current_bytes - self.last_net_bytes
        self.last_net_bytes = current_bytes
        if delta > self.net_bytes_threshold:
            event_type = "NET_SPIKE"
            raw_payload = {"event": event_type, "bytes": delta, "source": "sys-net"}
            event_bus.publish("sys_raw_event", raw_payload)
            self.engine.process(event_type, source="sys-net")

    def _maybe_check_filesystem(self):
        if not self.fs_enabled or not self.fs_paths:
            return
        now = time.time()
        if now - self.last_fs_check < self.fs_poll_interval:
            return
        self.last_fs_check = now

        for path in self.fs_paths:
            snap_prev = self.fs_snapshots.get(path)
            snap_now = self._snapshot_path(path)
            if snap_now is None:
                continue
            if snap_prev is None:
                self.fs_snapshots[path] = snap_now
                continue
            added = snap_now - snap_prev
            removed = snap_prev - snap_now
            self.fs_snapshots[path] = snap_now
            if added:
                event_type = "FS_NEW"
                raw_payload = {"event": event_type, "count": len(added), "source": f"fs:{path}"}
                event_bus.publish("sys_raw_event", raw_payload)
                self.engine.process(event_type, source=f"fs:{path}")
            if removed:
                event_type = "FS_DEL"
                raw_payload = {"event": event_type, "count": len(removed), "source": f"fs:{path}"}
                event_bus.publish("sys_raw_event", raw_payload)
                self.engine.process(event_type, source=f"fs:{path}")
            if added or removed:
                event_type = "FS_CHANGE"
                raw_payload = {"event": event_type, "source": f"fs:{path}"}
                event_bus.publish("sys_raw_event", raw_payload)
                self.engine.process(event_type, source=f"fs:{path}")

    def _snapshot_path(self, path):
        files = set()
        try:
            count = 0
            for root, dirs, filenames in os.walk(path):
                for f in filenames:
                    full = os.path.join(root, f)
                    h = hashlib.sha1(full.encode("utf-8", errors="ignore")).hexdigest()
                    files.add(h)
                    count += 1
                    if count > self.fs_max_files:
                        return files
            return files
        except Exception as e:
            log.add("ERROR", f"{SYMBOL_FS} FS snapshot failed for {path}: {e}")
            return None

    def _split_proc_key(self, key):
        try:
            pid, name = key.split(":", 1)
            return pid, name
        except ValueError:
            return "?", key

# -----------------------------
# SELF-SCALER
# -----------------------------
class SelfScaler:
    """
    Watches:
      - Predictive risk
      - Event volume (from risk history)
      - CPU load (if psutil)
    Adjusts:
      - swarm node count
      - system poll interval
      - net/fs poll intervals
      - event generator interval (sim mode)
    Modes:
      - CALM
      - NORMAL
      - STRESSED
      - CRITICAL
    """
    def __init__(self, swarm: SwarmManager, sys_collector: SystemEventCollector,
                 generator: EventGenerator, cfg: dict):
        self.swarm = swarm
        self.sys_collector = sys_collector
        self.generator = generator
        self.cfg = cfg or {}
        self.enabled = self.cfg.get("enabled", True)
        self.interval = self.cfg.get("interval", 5.0)
        self.max_swarm = self.cfg.get("max_swarm_nodes", 7)
        self.min_swarm = self.cfg.get("min_swarm_nodes", 1)
        self.min_poll = self.cfg.get("min_poll_interval", 0.5)
        self.max_poll = self.cfg.get("max_poll_interval", 10.0)
        self.min_event_interval = self.cfg.get("min_event_interval", 0.1)
        self.max_event_interval = self.cfg.get("max_event_interval", 1.0)

        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_mode = "NORMAL"

    def start(self):
        if not self.enabled:
            log.add("INFO", f"{SYMBOL_SCALE} SelfScaler disabled by config.")
            return
        with self.lock:
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.add("INFO", f"{SYMBOL_SCALE} SelfScaler started (interval={self.interval}s).")

    def stop(self):
        with self.lock:
            self.running = False
        log.add("INFO", f"{SYMBOL_SCALE} SelfScaler stopped.")

    def _run(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                interval = self.interval
            time.sleep(interval)
            self._scale_once()

    def _scale_once(self):
        risk = predictive_engine.get_risk()
        level = predictive_engine.classify_risk_level()
        cpu = self._get_cpu()
        mode = self._classify_mode(risk, level, cpu)

        if mode != self.last_mode:
            log.add("INFO", f"{SYMBOL_SCALE} SelfScaler mode change: {self.last_mode} -> {mode}")
            self.last_mode = mode

        self._scale_swarm(mode)
        self._scale_polling(mode)
        self._scale_generator(mode)

    def _get_cpu(self):
        if not psutil:
            return None
        try:
            return psutil.cpu_percent(interval=0.0)
        except Exception:
            return None

    def _classify_mode(self, risk, level, cpu):
        # Simple heuristic: combine risk + cpu
        if level == "HIGH" or (cpu is not None and cpu > 85):
            return "CRITICAL"
        if level == "WARN" or (cpu is not None and cpu > 70):
            return "STRESSED"
        if risk < 15 and (cpu is None or cpu < 30):
            return "CALM"
        return "NORMAL"

    def _scale_swarm(self, mode):
        current_nodes = self.swarm.get_node_count()
        if mode == "CRITICAL":
            target = min(self.max_swarm, current_nodes + 2)
        elif mode == "STRESSED":
            target = min(self.max_swarm, current_nodes + 1)
        elif mode == "CALM":
            target = max(self.min_swarm, current_nodes - 1)
        else:
            target = current_nodes
        if target != current_nodes:
            self.swarm.set_node_count(target)

        # Sync interval
        if mode == "CRITICAL":
            self.swarm.set_interval(1.0)
        elif mode == "STRESSED":
            self.swarm.set_interval(2.0)
        elif mode == "CALM":
            self.swarm.set_interval(4.0)
        else:
            self.swarm.set_interval(2.5)

    def _scale_polling(self, mode):
        if mode == "CRITICAL":
            sys_interval = max(self.min_poll, 1.0)
            net_interval = max(self.min_poll, 2.0)
            fs_interval = max(self.min_poll, 2.0)
        elif mode == "STRESSED":
            sys_interval = max(self.min_poll, 1.5)
            net_interval = max(self.min_poll, 3.0)
            fs_interval = max(self.min_poll, 3.0)
        elif mode == "CALM":
            sys_interval = min(self.max_poll, 4.0)
            net_interval = min(self.max_poll, 8.0)
            fs_interval = min(self.max_poll, 8.0)
        else:  # NORMAL
            sys_interval = 2.0
            net_interval = 5.0
            fs_interval = 5.0

        self.sys_collector.set_poll_interval(sys_interval)
        self.sys_collector.set_net_poll_interval(net_interval)
        self.sys_collector.set_fs_poll_interval(fs_interval)

    def _scale_generator(self, mode):
        if mode == "CRITICAL":
            interval = self.min_event_interval
        elif mode == "STRESSED":
            interval = max(self.min_event_interval, 0.25)
        elif mode == "CALM":
            interval = min(self.max_event_interval, 0.8)
        else:
            interval = 0.5
        self.generator.set_interval(interval)

# -----------------------------
# GUI COCKPIT
# -----------------------------
class LineGUI:
    def __init__(self, root, engine: LineComputer, swarm: SwarmManager,
                 watchdog: Watchdog, generator: EventGenerator,
                 sys_collector: SystemEventCollector, profiler: ProfileManager,
                 fold_mgr: FoldManager):
        self.root = root
        self.engine = engine
        self.swarm = swarm
        self.watchdog = watchdog
        self.generator = generator
        self.sys_collector = sys_collector
        self.profiler = profiler
        self.fold_mgr = fold_mgr

        self.root.title("Line Computing Cockpit")
        self.root.geometry("1300x780")

        self.ui_queue = queue.Queue()
        self.current_risk = 0.0

        self._build_layout()
        self._wire_events()
        self._start_polling_ui_queue()

    def _build_layout(self):
        top_frame = tkinter.Frame(self.root)
        top_frame.pack(side=tkinter.TOP, fill=tkinter.X)

        btn_frame = tkinter.Frame(top_frame)
        btn_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        profile_frame = tkinter.Frame(top_frame)
        profile_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        strict_frame = tkinter.Frame(top_frame)
        strict_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        fold_frame = tkinter.Frame(top_frame)
        fold_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        replay_frame = tkinter.Frame(top_frame)
        replay_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        meta_frame = tkinter.Frame(top_frame)
        meta_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        risk_frame = tkinter.Frame(top_frame)
        risk_frame.pack(side=tkinter.LEFT, padx=5, pady=5)

        status_frame = tkinter.Frame(top_frame)
        status_frame.pack(side=tkinter.RIGHT, padx=5, pady=5)

        self.btn_start = tkinter.Button(btn_frame, text="Start System", command=self.start_system)
        self.btn_stop = tkinter.Button(btn_frame, text="Stop System", command=self.stop_system)
        self.btn_start.pack(side=tkinter.LEFT, padx=5)
        self.btn_stop.pack(side=tkinter.LEFT, padx=5)

        self.btn_profile_start = tkinter.Button(profile_frame, text="Start Profiling", command=self.start_profiling)
        self.btn_profile_stop = tkinter.Button(profile_frame, text="Stop Profiling", command=self.stop_profiling)
        self.btn_profile_lock = tkinter.Button(profile_frame, text="Lock Profile", command=self.lock_profile)
        self.btn_profile_start.pack(side=tkinter.LEFT, padx=3)
        self.btn_profile_stop.pack(side=tkinter.LEFT, padx=3)
        self.btn_profile_lock.pack(side=tkinter.LEFT, padx=3)

        self.strict_var = tkinter.BooleanVar(value=self.engine.is_strict_mode())
        self.chk_strict = tkinter.Checkbutton(strict_frame, text="Strict Mode", variable=self.strict_var,
                                              command=self.toggle_strict_mode)
        self.chk_strict.pack(side=tkinter.LEFT, padx=5)

        self.btn_manual_fold = tkinter.Button(fold_frame, text="Fold Last Anomaly", command=self.manual_fold_last)
        self.btn_manual_fold.pack(side=tkinter.LEFT, padx=5)

        self.btn_export_risk = tkinter.Button(replay_frame, text="Export History", command=self.export_risk)
        self.btn_import_risk = tkinter.Button(replay_frame, text="Import History", command=self.import_risk)
        self.btn_replay_start = tkinter.Button(replay_frame, text="Start Replay", command=self.start_replay)
        self.btn_replay_stop = tkinter.Button(replay_frame, text="Stop Replay", command=self.stop_replay)
        self.btn_export_risk.pack(side=tkinter.LEFT, padx=3)
        self.btn_import_risk.pack(side=tkinter.LEFT, padx=3)
        self.btn_replay_start.pack(side=tkinter.LEFT, padx=3)
        self.btn_replay_stop.pack(side=tkinter.LEFT, padx=3)

        self.label_profile_meta = tkinter.Label(meta_frame, text="Profile: none", fg="blue")
        self.label_profile_meta.pack(side=tkinter.LEFT, padx=5)
        self._update_profile_metadata_label()

        self.label_risk = tkinter.Label(risk_frame, text="Risk: 0.0 (NORMAL)", fg="green")
        self.label_risk.pack(side=tkinter.LEFT, padx=5)

        self.label_status = tkinter.Label(status_frame, text="Status: IDLE", fg="gray")
        self.label_status.pack(side=tkinter.RIGHT)

        main_frame = tkinter.Frame(self.root)
        main_frame.pack(fill=tkinter.BOTH, expand=True)

        log_frame = tkinter.LabelFrame(main_frame, text="Live Log")
        log_frame.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True, padx=5, pady=5)

        self.txt_log = scrolledtext.ScrolledText(log_frame, wrap=tkinter.WORD, height=20)
        self.txt_log.pack(fill=tkinter.BOTH, expand=True)

        right_frame = tkinter.Frame(main_frame)
        right_frame.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True)

        threat_frame = tkinter.LabelFrame(right_frame, text="Threat Matrix")
        threat_frame.pack(fill=tkinter.BOTH, expand=True, padx=5, pady=5)

        self.list_threats = tkinter.Listbox(threat_frame, height=10)
        self.list_threats.pack(fill=tkinter.BOTH, expand=True)

        swarm_frame = tkinter.LabelFrame(right_frame, text="Swarm Dashboard")
        swarm_frame.pack(fill=tkinter.BOTH, expand=True, padx=5, pady=5)

        self.list_swarm = tkinter.Listbox(swarm_frame, height=10)
        self.list_swarm.pack(fill=tkinter.BOTH, expand=True)

    def _wire_events(self):
        event_bus.subscribe("log", self._on_log_event)
        event_bus.subscribe("threat", self._on_threat_event)
        event_bus.subscribe("swarm_sync", self._on_swarm_event)
        event_bus.subscribe("watchdog_alert", self._on_watchdog_event)
        event_bus.subscribe("predictive_risk", self._on_risk_event)
        self.root.after(1000, self._refresh_swarm_status)

    def _on_log_event(self, event_type, data):
        self.ui_queue.put(("log", data))

    def _on_threat_event(self, event_type, data):
        self.ui_queue.put(("threat", data))

    def _on_swarm_event(self, event_type, data):
        self.ui_queue.put(("swarm_log", data))

    def _on_watchdog_event(self, event_type, data):
        self.ui_queue.put(("log", data))

    def _on_risk_event(self, event_type, data):
        if not data:
            return
        self.ui_queue.put(("risk", data))

    def _start_polling_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "threat":
                    self._add_threat(payload)
                elif kind == "swarm_log":
                    self._append_log(payload)
                elif kind == "risk":
                    self._update_risk(payload.get("risk", 0.0))
        except queue.Empty:
            pass
        self.root.after(100, self._start_polling_ui_queue)

    def _append_log(self, line):
        self.txt_log.insert(tkinter.END, line + "\n")
        self.txt_log.see(tkinter.END)

    def _add_threat(self, line):
        self.list_threats.insert(tkinter.END, line)
        self.list_threats.see(tkinter.END)

    def _refresh_swarm_status(self):
        self.list_swarm.delete(0, tkinter.END)
        for line in self.swarm.get_status_lines():
            self.list_swarm.insert(tkinter.END, line)
        self.root.after(1000, self._refresh_swarm_status)

    def _update_profile_metadata_label(self):
        meta = profile_manager.get_metadata()
        if not meta:
            txt = "Profile: none"
        else:
            created = meta.get("created_at", "?")
            host = meta.get("host", "?")
            txt = f"Profile: {created} @ {host}"
        self.label_profile_meta.config(text=txt)

    def _update_risk(self, risk):
        self.current_risk = risk
        level = predictive_engine.classify_risk_level()
        if level == "HIGH":
            color = "red"
        elif level == "WARN":
            color = "orange"
        else:
            color = "green"
        self.label_risk.config(text=f"Risk: {risk:.1f} ({level})", fg=color)

    def start_system(self):
        self.generator.start()
        self.watchdog.start()
        self.swarm.start(interval=3.0)
        self.sys_collector.start()
        self.label_status.config(text="Status: RUNNING", fg="green")

    def stop_system(self):
        self.generator.stop()
        self.watchdog.stop()
        self.swarm.stop()
        self.sys_collector.stop()
        self.label_status.config(text="Status: STOPPED", fg="red")

    def start_profiling(self):
        profile_manager.start()
        self._append_log(f"{SYMBOL_PROFILE} Profiling requested from GUI.")

    def stop_profiling(self):
        profile_manager.stop()
        seq_len = len(profile_manager.get_sequence())
        self._append_log(f"{SYMBOL_PROFILE} Profiling stopped. Steps learned: {seq_len}")

    def lock_profile(self):
        profile_manager.stop()
        profile_manager.lock_profile_into_engine(self.engine)
        self._update_profile_metadata_label()
        self._append_log(f"{SYMBOL_PROFILE} Profile locked into engine as new expected sequence.")

    def toggle_strict_mode(self):
        enabled = self.strict_var.get()
        self.engine.set_strict_mode(enabled)

    def manual_fold_last(self):
        if self.fold_mgr:
            self.fold_mgr.manual_fold_last()

    def export_risk(self):
        risk_history.export()

    def import_risk(self):
        risk_history.import_()

    def start_replay(self):
        hist = risk_history.get_history()
        if not hist:
            self._append_log(f"{SYMBOL_REPLAY} No history to replay.")
            return
        replay_controller.start(hist)

    def stop_replay(self):
        replay_controller.stop()

# -----------------------------
# MAIN
# -----------------------------
def main():
    global fold_manager

    engine = LineComputer(
        settings.get("expected_sequence", ["LEFT", "RIGHT"]),
        strict_mode=settings.get("strict_mode", False)
    )

    fold_manager = FoldManager(
        engine,
        unknown_threshold=settings.get("folding", {}).get("unknown_threshold", 5)
    )

    swarm = SwarmManager(settings.get("swarm_nodes", 3), engine.get_expected_sequence())

    watchdog = Watchdog(
        engine,
        timeout=settings.get("watchdog_timeout", 5.0),
        check_interval=1.0
    )

    generator = EventGenerator(
        engine,
        interval=settings.get("event_interval", 0.5),
        anomaly_rate=0.3
    )

    policy_engine = ProcessPolicyEngine(
        engine,
        settings.get("process_policies", {})
    )

    sys_collector = SystemEventCollector(
        engine,
        policy_engine=policy_engine,
        poll_interval=settings.get("system_poll_interval", 2.0),
        whitelist=settings.get("process_whitelist", []),
        blacklist=settings.get("process_blacklist", [])
    )

    scaler_cfg = settings.get("self_scaler", {})
    self_scaler = SelfScaler(
        swarm=swarm,
        sys_collector=sys_collector,
        generator=generator,
        cfg=scaler_cfg
    )

    root = tkinter.Tk()
    gui = LineGUI(root, engine, swarm, watchdog, generator, sys_collector, profile_manager, fold_manager)

    if settings.get("auto_start_system", True):
        gui.start_system()
    if settings.get("auto_start_profiling", True):
        profile_manager.start()
        log.add("INFO", f"{SYMBOL_PROFILE} Auto-profiling enabled on startup.")

    self_scaler.start()

    try:
        root.mainloop()
    finally:
        generator.stop()
        watchdog.stop()
        swarm.stop()
        sys_collector.stop()
        self_scaler.stop()
        replay_controller.stop()
        save_settings(settings)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

