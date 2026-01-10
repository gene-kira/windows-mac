import sys
import subprocess
import importlib
import os
import json
import random
import time
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk


# =========================
#  Auto-loader utilities
# =========================

def ensure_lib(module_name, pip_name=None):
    """
    Try to import a module; if it fails, attempt to install it via pip.
    Returns the imported module or None if unavailable.
    """
    if pip_name is None:
        pip_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"[AUTO-LOADER] Missing module '{module_name}', attempting to install '{pip_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"[AUTO-LOADER] Failed to install '{pip_name}': {e}")
            return None


# Optional libs (not required but enhance awareness)
psutil = ensure_lib("psutil")  # system + process insight


# =========================
#  Introspection bus
# =========================

class IntrospectionBus:
    """
    Simple event bus for internal events.
    """

    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def publish(self, event_type, payload):
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "payload": payload,
        }
        for cb in self.subscribers:
            try:
                cb(event)
            except Exception as e:
                print(f"[BUS] Subscriber error: {e}")


# =========================
#  Adapters
# =========================

class BaseAdapter:
    def collect_signals(self):
        raise NotImplementedError

    def describe(self):
        return self.__class__.__name__


class SystemAdapter(BaseAdapter):
    """
    Time, CPU, memory.
    """

    def collect_signals(self):
        sig = {
            "time": time.strftime("%H:%M:%S"),
        }
        if psutil:
            try:
                sig["cpu_percent"] = psutil.cpu_percent(interval=0.0)
                mem = psutil.virtual_memory()
                sig["mem_percent"] = mem.percent
            except Exception:
                sig["cpu_percent"] = None
                sig["mem_percent"] = None
        else:
            sig["cpu_percent"] = None
            sig["mem_percent"] = None
        return sig

    def describe(self):
        return "SystemAdapter"


class WorldNoiseAdapter(BaseAdapter):
    """
    Synthetic world tension.
    """

    def __init__(self):
        self.random = random.Random()
        self.random.seed(time.time())

    def collect_signals(self):
        return {
            "world_tension": round(self.random.uniform(0, 1), 3),
            "signal_noise": round(self.random.uniform(-1, 1), 3),
        }

    def describe(self):
        return "WorldNoiseAdapter"


class ProcessAdapter(BaseAdapter):
    """
    Observes running processes (Windows-friendly via psutil).
    """

    def collect_signals(self):
        results = []
        if not psutil:
            return {"error": "psutil not available"}
        try:
            for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
                info = p.info
                results.append({
                    "pid": info.get("pid"),
                    "name": info.get("name") or "unknown",
                    "cpu": info.get("cpu_percent") or 0.0,
                    "mem": info.get("memory_percent") or 0.0,
                })
        except Exception as e:
            return {"error": str(e)}
        return {"processes": results}

    def describe(self):
        return "ProcessAdapter"


# =========================
#  Memory module (persistent)
# =========================

class MemoryStore:
    """
    Handles save/load of engine state + voice registry + profiles.
    """

    def __init__(self):
        self.last_path = None

    def serialize_engine(self, engine):
        return {
            "next_id": engine.next_id,
            "thought_history": engine.thought_history,
            "current_focus_id": engine.current_focus["id"] if engine.current_focus else None,
            "voice_registry": engine.voice_registry,
            "profiles": engine.profiles,
            "active_profile": engine.active_profile,
        }

    def restore_engine(self, engine, data):
        engine.next_id = data.get("next_id", 1)
        engine.thought_history = data.get("thought_history", [])
        focus_id = data.get("current_focus_id", None)
        engine.current_focus = None
        if focus_id is not None:
            for t in engine.thought_history:
                if t["id"] == focus_id:
                    engine.current_focus = t
                    break
        engine.voice_registry = data.get("voice_registry", {})
        engine.profiles = data.get("profiles", engine.default_profiles())
        engine.active_profile = data.get("active_profile", "Default")

    def save_to_file(self, engine, path):
        data = self.serialize_engine(engine)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.last_path = path

    def load_from_file(self, engine, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.restore_engine(engine, data)
        self.last_path = path


# =========================
#  Thought engine + voices
# =========================

class ThoughtEngine:
    """
    Prosthetic mind core:
    - Maintains thought history + current focus.
    - Maintains voice registry (process roles).
    - Maintains profiles / baselines.
    - Uses adapters + signals to think.
    """

    ROLE_ORDER = ["core", "focus", "worker", "ally", "noise", "suspect", "ghost"]

    def __init__(self, bus):
        self.bus = bus
        self.thought_history = []
        self.current_focus = None
        self.next_id = 1
        self.random = random.Random()
        self.random.seed(time.time())

        self.adapters = []
        self.mode = "default"
        self.active_profile = "Default"

        self.voice_registry = {}  # name -> meta
        self.profiles = self.default_profiles()

    def default_profiles(self):
        return {
            "Default": {
                "expected_processes": [],
                "cpu_range": [0, 80],
                "mem_range": [0, 80],
            },
            "Work": {
                "expected_processes": [],
                "cpu_range": [0, 80],
                "mem_range": [0, 80],
            },
            "Game": {
                "expected_processes": [],
                "cpu_range": [10, 100],
                "mem_range": [20, 100],
            },
            "Experiment": {
                "expected_processes": [],
                "cpu_range": [0, 100],
                "mem_range": [0, 100],
            },
        }

    def register_adapter(self, adapter):
        self.adapters.append(adapter)
        self.bus.publish("adapter_registered", {"adapter": adapter.describe()})

    def _new_thought_id(self):
        tid = self.next_id
        self.next_id += 1
        return tid

    def _timestamp(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- Voices / processes ----------

    def update_voices_from_processes(self, process_list):
        """
        Update voice registry based on current processes.
        """
        profile = self.profiles.get(self.active_profile, self.profiles["Default"])
        expected = set(profile.get("expected_processes", []))

        seen_names = set()

        for proc in process_list:
            name = proc.get("name", "unknown")
            pid = proc.get("pid")
            cpu = proc.get("cpu", 0.0)
            mem = proc.get("mem", 0.0)
            seen_names.add(name)

            entry = self.voice_registry.get(name, {
                "name": name,
                "role": "worker",
                "confidence": 0.3,
                "seen_count": 0,
                "last_pid": None,
                "last_cpu": 0.0,
                "last_mem": 0.0,
                "last_seen": None,
                "profile_marks": [],
            })

            entry["seen_count"] += 1
            entry["last_pid"] = pid
            entry["last_cpu"] = cpu
            entry["last_mem"] = mem
            entry["last_seen"] = self._timestamp()

            # Role heuristics
            if name.lower() in ["explorer.exe", "winlogon.exe", "csrss.exe"]:
                entry["role"] = "core"
                entry["confidence"] = max(entry["confidence"], 0.9)
            elif name in expected:
                entry["role"] = "ally"
                entry["confidence"] = max(entry["confidence"], 0.7)
            else:
                if cpu > 10 or mem > 5:
                    if entry["role"] not in ["core", "focus", "ally"]:
                        entry["role"] = "suspect"
                        entry["confidence"] = max(entry["confidence"], 0.5)
                else:
                    if entry["role"] not in ["core", "focus", "ally", "suspect"]:
                        entry["role"] = "noise"

            self.voice_registry[name] = entry

        # Voices not seen in this snapshot may become ghosts
        for name, entry in list(self.voice_registry.items()):
            if name not in seen_names:
                # If they flicker often, mark as ghost
                if entry["role"] not in ["core", "focus", "ally"]:
                    if entry["seen_count"] < 3:
                        entry["role"] = "ghost"
                        entry["confidence"] = min(entry["confidence"], 0.3)
                    self.voice_registry[name] = entry

    def get_sorted_voices(self):
        """
        Return voices grouped and sorted by role + importance.
        """
        buckets = {r: [] for r in self.ROLE_ORDER}
        for v in self.voice_registry.values():
            role = v.get("role", "worker")
            if role not in buckets:
                buckets[role] = []
            buckets[role].append(v)

        for role in buckets:
            buckets[role].sort(
                key=lambda x: (-x.get("confidence", 0.0), -x.get("last_cpu", 0.0), -x.get("last_mem", 0.0))
            )
        return buckets

    def set_voice_role(self, name, new_role):
        if name in self.voice_registry:
            entry = self.voice_registry[name]
            entry["role"] = new_role
            entry["confidence"] = max(entry.get("confidence", 0.3), 0.8)
            entry.setdefault("profile_marks", []).append(
                {"profile": self.active_profile, "role": new_role, "time": self._timestamp()}
            )
            self.voice_registry[name] = entry
            self.bus.publish("voice_tagged", {"name": name, "role": new_role, "profile": self.active_profile})

            # If you mark something as important in this profile, add to expected
            if new_role in ["core", "ally", "focus"]:
                profile = self.profiles.get(self.active_profile, self.profiles["Default"])
                if name not in profile["expected_processes"]:
                    profile["expected_processes"].append(name)
                    self.profiles[self.active_profile] = profile

    # ---------- Signals / thinking ----------

    def collect_all_signals(self):
        combined = {}
        for adapter in self.adapters:
            try:
                sigs = adapter.collect_signals()
                combined[adapter.describe()] = sigs
            except Exception as e:
                combined[adapter.describe()] = {"error": str(e)}

        # special: process adapter -> update voices
        proc_sig = combined.get("ProcessAdapter", {})
        if "processes" in proc_sig:
            self.update_voices_from_processes(proc_sig["processes"])

        return combined

    def _basic_associations(self, text, signals):
        text_lower = text.lower()
        associations = []

        # Themes from text
        if any(k in text_lower for k in ["heat", "thermistor", "water heater", "temperature"]):
            associations.append("thermal boundaries and hidden safety layers")
            associations.append("physics as the first rule engine")

        if any(k in text_lower for k in ["mind", "think", "thought", "consciousness"]):
            associations.append("mapping thoughts as visible, inspectable structures")
            associations.append("a machine that narrates its own internal shifts")

        if any(k in text_lower for k in ["rule", "free", "god", "create"]):
            associations.append("creation as controlled bending of prior constraints")
            associations.append("the tension between safety and absolute freedom in system design")

        # External signals
        sys_sig = signals.get("SystemAdapter", {})
        world_sig = signals.get("WorldNoiseAdapter", {})
        proc_sig = signals.get("ProcessAdapter", {})

        cpu = sys_sig.get("cpu_percent")
        mem = sys_sig.get("mem_percent")
        tension = world_sig.get("world_tension")

        if cpu is not None and cpu > 80:
            associations.append("machine under strain, survival mode thinking")
        if mem is not None and mem > 80:
            associations.append("memory pressure as cognitive overload metaphor")
        if tension is not None and tension > 0.7:
            associations.append("high world tension, thoughts drift toward resilience and risk")

        # Process world
        processes = proc_sig.get("processes", [])
        if processes:
            heavy = [p for p in processes if p.get("cpu", 0) > 10]
            if heavy:
                names = list({p["name"] for p in heavy})
                associations.append(f"heavy voices speaking loudly: {', '.join(names[:5])}")

        # If nothing else
        if not associations:
            associations.append("hidden structures behind everyday objects")
            associations.append("what this idea would look like scaled 1000x")

        extra = [
            "how this connects to failure modes and recovery",
            "how this system looks as a living organism",
            "what this becomes if you strip away all comfort",
            "how this changes if safety rails are redesigned, not removed",
        ]
        associations.append(self.random.choice(extra))

        return associations

    def think(self, user_input):
        if user_input is None:
            user_input = ""

        base_text = user_input.strip()
        if not base_text and self.current_focus:
            base_text = self.current_focus.get("content", "")
            source = "internal_focus"
        elif base_text:
            source = "user_input"
        else:
            source = "idle_drift"

        signals = self.collect_all_signals()
        associations = self._basic_associations(base_text, signals)
        chosen = self.random.choice(associations)

        thought_id = self._new_thought_id()
        timestamp = self._timestamp()

        reasoning = {
            "timestamp": timestamp,
            "source": source,
            "input_used": base_text,
            "associations_considered": associations,
            "chosen_association": chosen,
            "external_signals": signals,
            "mode": self.mode,
            "profile": self.active_profile,
        }

        thought_content = f"{chosen}"

        thought_entry = {
            "id": thought_id,
            "timestamp": timestamp,
            "source": source,
            "content": thought_content,
            "reasoning": reasoning,
            "links": [],
        }

        if self.thought_history:
            prev = self.thought_history[-1]
            thought_entry["links"].append(prev["id"])

        self.thought_history.append(thought_entry)
        self.current_focus = thought_entry

        self.bus.publish("thought_created", {"thought": thought_entry})

        return thought_entry

    def get_state_summary(self):
        if not self.current_focus:
            return f"Idle. Profile: {self.active_profile}. Waiting for a seed.\n"

        focus = self.current_focus
        return (
            f"Current focus (#{focus['id']} at {focus['timestamp']}):\n"
            f"  {focus['content']}\n"
            f"Source: {focus['source']}\n"
            f"Linked to: {focus['links']}\n"
            f"Mode: {focus['reasoning'].get('mode', 'default')}\n"
            f"Profile: {focus['reasoning'].get('profile', self.active_profile)}\n"
        )


# =========================
#  Background thinker
# =========================

class BackgroundThinker(threading.Thread):
    def __init__(self, engine, request_queue, on_thought_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.request_queue = request_queue
        self.on_thought_callback = on_thought_callback
        self._stop_flag = threading.Event()
        self.daemon = True

    def run(self):
        while not self._stop_flag.is_set():
            try:
                user_input = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            thought = self.engine.think(user_input)
            self.on_thought_callback(thought)

    def stop(self):
        self._stop_flag.set()


# =========================
#  GUI
# =========================

class MindForgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MindForge v0.3 – Spine of the Machine")

        self.bus = IntrospectionBus()
        self.engine = ThoughtEngine(self.bus)
        self.memory = MemoryStore()

        self.request_queue = queue.Queue()
        self.background_thinker = BackgroundThinker(
            self.engine, self.request_queue, self.on_new_thought
        )
        self.background_thinker.start()

        # Adapters
        self.engine.register_adapter(SystemAdapter())
        self.engine.register_adapter(WorldNoiseAdapter())
        self.engine.register_adapter(ProcessAdapter())

        self.bus.subscribe(self.on_bus_event)

        self.selected_voice_name = None

        self._build_layout()
        self._refresh_all_views()

    def _build_layout(self):
        # Top controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Seed / Prompt:").pack(side=tk.LEFT)
        self.input_entry = tk.Entry(top_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.think_button = tk.Button(top_frame, text="Think Once", command=self.on_think_clicked)
        self.think_button.pack(side=tk.LEFT, padx=5)

        self.drift_button = tk.Button(top_frame, text="Drift", command=self.on_drift_clicked)
        self.drift_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(top_frame, text="Save Brain", command=self.on_save_brain)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(top_frame, text="Load Brain", command=self.on_load_brain)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Profile selector
        tk.Label(top_frame, text="Profile:").pack(side=tk.LEFT, padx=(10, 2))
        self.profile_var = tk.StringVar(value=self.engine.active_profile)
        self.profile_combo = ttk.Combobox(
            top_frame,
            textvariable=self.profile_var,
            values=list(self.engine.profiles.keys()),
            state="readonly",
            width=12
        )
        self.profile_combo.pack(side=tk.LEFT)
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_changed)

        self.mode_label = tk.Label(top_frame, text="Mode: default")
        self.mode_label.pack(side=tk.LEFT, padx=10)

        # Main bottom split
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: thought stream
        left_frame = tk.Frame(bottom_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="Thought Stream").pack(side=tk.TOP, anchor="w")
        self.thought_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=10)
        self.thought_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Middle: reasoning + state
        mid_frame = tk.Frame(bottom_frame)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(mid_frame, text="Internal Reasoning").pack(side=tk.TOP, anchor="w")
        self.reason_text = scrolledtext.ScrolledText(mid_frame, wrap=tk.WORD, height=10)
        self.reason_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tk.Label(mid_frame, text="Current Mental State").pack(side=tk.TOP, anchor="w")
        self.state_text = scrolledtext.ScrolledText(mid_frame, wrap=tk.WORD, height=5)
        self.state_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right: adapters + bus + voices
        right_frame = tk.Frame(bottom_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Adapters
        tk.Label(right_frame, text="Adapters / External Signals").pack(side=tk.TOP, anchor="w")
        self.adapter_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=7)
        self.adapter_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Event log
        tk.Label(right_frame, text="Event Log (Bus)").pack(side=tk.TOP, anchor="w")
        self.bus_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=7)
        self.bus_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Voice world map
        voice_frame = tk.Frame(self.root)
        voice_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(voice_frame, text="Voices / Process World Map").pack(side=tk.TOP, anchor="w")
        self.voice_list = tk.Listbox(voice_frame, height=10)
        self.voice_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.voice_list.bind("<<ListboxSelect>>", self.on_voice_selected)

        voice_controls = tk.Frame(voice_frame)
        voice_controls.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Label(voice_controls, text="Tag Voice As:").pack(side=tk.TOP, anchor="w")
        for role in ["core", "focus", "ally", "worker", "noise", "suspect", "ghost"]:
            b = tk.Button(
                voice_controls,
                text=role.capitalize(),
                command=lambda r=role: self.on_tag_voice(r),
                width=10
            )
            b.pack(side=tk.TOP, pady=1)

        self.voice_info_text = scrolledtext.ScrolledText(voice_controls, wrap=tk.WORD, height=10, width=40)
        self.voice_info_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- GUI callbacks ----------

    def on_think_clicked(self):
        user_input = self.input_entry.get()
        self.request_queue.put(user_input)

    def on_drift_clicked(self):
        self.request_queue.put("")

    def on_new_thought(self, thought):
        self.root.after(0, self._update_gui_with_thought, thought)

    def _update_gui_with_thought(self, thought):
        ts = thought["timestamp"]
        tid = thought["id"]
        src = thought["source"]
        content = thought["content"]
        self.thought_text.insert(
            tk.END,
            f"[#{tid} @ {ts} | source={src}] {content}\n"
        )
        self.thought_text.see(tk.END)

        r = thought["reasoning"]
        reasoning_text = (
            f"Thought #{tid} reasoning:\n"
            f"  Time: {r['timestamp']}\n"
            f"  Source: {r['source']}\n"
            f"  Mode: {r.get('mode', 'default')}\n"
            f"  Profile: {r.get('profile', self.engine.active_profile)}\n"
            f"  Input used: {r['input_used']!r}\n"
            f"  Associations considered:\n"
        )
        for a in r["associations_considered"]:
            mark = "->" if a == r["chosen_association"] else "  "
            reasoning_text += f"    {mark} {a}\n"

        reasoning_text += "\nExternal signals:\n"
        for adapter_name, sigs in r["external_signals"].items():
            reasoning_text += f"  {adapter_name}:\n"
            if isinstance(sigs, dict):
                for k, v in sigs.items():
                    if k == "processes":
                        reasoning_text += f"    processes: {len(v)} observed\n"
                    else:
                        reasoning_text += f"    {k}: {v}\n"
            else:
                reasoning_text += f"    {sigs}\n"

        reasoning_text += "\n"
        self.reason_text.insert(tk.END, reasoning_text)
        self.reason_text.see(tk.END)

        state_summary = self.engine.get_state_summary()
        self.state_text.delete("1.0", tk.END)
        self.state_text.insert(tk.END, state_summary)

        self._refresh_all_views()

    def _refresh_all_views(self):
        self._update_adapter_view()
        self._update_voice_list()

    def _update_adapter_view(self):
        signals = self.engine.collect_all_signals()
        self.adapter_text.delete("1.0", tk.END)
        for adapter_name, sigs in signals.items():
            self.adapter_text.insert(tk.END, f"{adapter_name}:\n")
            if isinstance(sigs, dict):
                for k, v in sigs.items():
                    if k == "processes":
                        self.adapter_text.insert(tk.END, f"  processes: {len(v)} observed\n")
                    else:
                        self.adapter_text.insert(tk.END, f"  {k}: {v}\n")
            else:
                self.adapter_text.insert(tk.END, f"  {sigs}\n")
            self.adapter_text.insert(tk.END, "\n")

    def on_bus_event(self, event):
        self.root.after(0, self._append_bus_event, event)

    def _append_bus_event(self, event):
        self.bus_text.insert(
            tk.END,
            f"[{event['timestamp']}] {event['type']}: {event['payload']}\n"
        )
        self.bus_text.see(tk.END)

    def on_save_brain(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.memory.save_to_file(self.engine, path)
            messagebox.showinfo("Saved", f"Brain saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save brain: {e}")

    def on_load_brain(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.memory.load_from_file(self.engine, path)
            messagebox.showinfo("Loaded", f"Brain loaded from {path}")
            self.thought_text.insert(tk.END, f"\n[System] Brain state loaded from {os.path.basename(path)}\n")
            self._refresh_all_views()
            state_summary = self.engine.get_state_summary()
            self.state_text.delete("1.0", tk.END)
            self.state_text.insert(tk.END, state_summary)
            self.profile_var.set(self.engine.active_profile)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load brain: {e}")

    def on_profile_changed(self, event=None):
        profile_name = self.profile_var.get()
        self.engine.active_profile = profile_name
        self.bus.publish("profile_changed", {"profile": profile_name})
        self._refresh_all_views()

    # ---------- Voices UI ----------

    def _update_voice_list(self):
        buckets = self.engine.get_sorted_voices()
        self.voice_list.delete(0, tk.END)
        for role in self.engine.ROLE_ORDER:
            voices = buckets.get(role, [])
            if not voices:
                continue
            self.voice_list.insert(tk.END, f"=== {role.upper()} ===")
            for v in voices:
                name = v["name"]
                cpu = v.get("last_cpu", 0.0)
                mem = v.get("last_mem", 0.0)
                self.voice_list.insert(
                    tk.END,
                    f"{name} | cpu={cpu:.1f}% mem={mem:.1f}%"
                )

    def on_voice_selected(self, event):
        selection = self.voice_list.curselection()
        if not selection:
            return
        idx = selection[0]
        text = self.voice_list.get(idx)
        if text.startswith("==="):
            self.selected_voice_name = None
            self.voice_info_text.delete("1.0", tk.END)
            self.voice_info_text.insert(tk.END, "No voice selected.\n")
            return

        # parse name before first |
        name = text.split("|", 1)[0].strip()
        self.selected_voice_name = name
        v = self.engine.voice_registry.get(name)

        self.voice_info_text.delete("1.0", tk.END)
        if not v:
            self.voice_info_text.insert(tk.END, f"{name}\n(no info in registry)\n")
            return

        self.voice_info_text.insert(
            tk.END,
            f"Name: {v['name']}\n"
            f"Role: {v['role']}\n"
            f"Confidence: {v.get('confidence', 0.0):.2f}\n"
            f"Seen count: {v.get('seen_count', 0)}\n"
            f"Last PID: {v.get('last_pid', None)}\n"
            f"Last CPU: {v.get('last_cpu', 0.0):.1f}%\n"
            f"Last MEM: {v.get('last_mem', 0.0):.1f}%\n"
            f"Last seen: {v.get('last_seen', None)}\n"
            f"Profile marks:\n"
        )

        for mark in v.get("profile_marks", []):
            self.voice_info_text.insert(
                tk.END,
                f"  [{mark['time']}] profile={mark['profile']} role={mark['role']}\n"
            )

    def on_tag_voice(self, role):
        if not self.selected_voice_name:
            messagebox.showinfo("Tag Voice", "Select a voice in the list first.")
            return
        self.engine.set_voice_role(self.selected_voice_name, role)
        self._refresh_all_views()
        self.on_voice_selected(None)

    # ---------- Shutdown ----------

    def on_close(self):
        try:
            self.background_thinker.stop()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MindForgeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

