#!/usr/bin/env python3
# neural_core_v3_tk.py
# Phase 7 (Tkinter, predictive+adaptive):
# - Event-driven organs
# - Copilot-first brain (stub) + local LLM fallback (Ollama / LM Studio)
# - Predictive Mode Drift + sequence-aware forecasting
# - Meta-Reasoning Organ
# - Self-Healing Organ Wrapper + ResilienceOrgan
# - Adaptive Critique Depth + Outcome Feedback (self-tuning)
# - Behavioral Profiling Engine
# - Predictive Keyboard Behavior (pywin32-based, NO PYNPUT)
# - Reasoning Template Organ (prepares prompts per intent + trend)
# - Time-of-day Session Profile Organ
# - Clarification Organ (asks questions when unsure)
# - Persona-based Brain (per-intent personas)
# - Smart Autoloader (install + smart update) with log
# - Dependency Status / Autoloader panel
# - pyperclip-safe clipboard
# - Tkinter GUI (no PySide6)

import threading
import time
import sys
import subprocess
from collections import Counter, deque, defaultdict
from datetime import datetime

import psutil
import requests

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# ============================================================
#  OPTIONAL CLIPBOARD SUPPORT (pyperclip-safe)
# ============================================================

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except Exception:
    PYPERCLIP_AVAILABLE = False

# ============================================================
#  OPTIONAL WINDOWS KEYBOARD SUPPORT (pywin32-safe)
# ============================================================

try:
    import win32api
    import win32con
    WIN32_AVAILABLE = True
except Exception:
    win32api = None
    win32con = None
    WIN32_AVAILABLE = False


# ============================================================
#  LLM HELPERS (Copilot stub + Ollama + LM Studio)
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"


def call_copilot_stub(prompt: str, system: str | None = None, max_tokens: int = 512):
    msg = (
        "[COPILOT PLACEHOLDER]\n"
        "This is where a real Copilot API call would go.\n"
        "Replace call_copilot_stub() with your real integration."
    )
    return msg, False


def call_ollama_llm(prompt: str, system: str | None = None, max_tokens: int = 512):
    if system:
        full_prompt = f"System:\n{system}\n\nUser:\n{prompt}"
    else:
        full_prompt = prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "options": {"num_predict": max_tokens},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        return str(data)
    except Exception as e:
        return f"[LOCAL LLM ERROR: {e}]"


def call_lmstudio_llm(prompt: str, system: str | None = None, max_tokens: int = 512):
    headers = {"Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "lmstudio",
        "messages": messages,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(LMSTUDIO_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        return str(data)
    except Exception as e:
        return f"[LOCAL LLM ERROR (LM Studio)]: {e}"


def llm_critique(answer: str, backend: str):
    prompt = f"""
Critique the following answer. Identify weaknesses, missing details, unclear logic, and improvements.

Answer:
{answer}
""".strip()

    if backend == "ollama":
        return call_ollama_llm(prompt, system="Critique engine", max_tokens=512)
    elif backend == "lmstudio":
        return call_lmstudio_llm(prompt, system="Critique engine", max_tokens=512)
    return answer


def llm_refine(original: str, critique: str, backend: str):
    prompt = f"""
Refine the original answer using the critique. Improve clarity, detail, and correctness.

Original:
{original}

Critique:
{critique}
""".strip()

    if backend == "ollama":
        return call_ollama_llm(prompt, system="Refinement engine", max_tokens=700)
    elif backend == "lmstudio":
        return call_lmstudio_llm(prompt, system="Refinement engine", max_tokens=700)
    return original


def detect_local_llm():
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": "ping", "options": {"num_predict": 5}},
            timeout=2,
        )
        if r.status_code == 200:
            return "ollama", "Ollama detected on localhost:11434"
    except Exception:
        pass

    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=2)
        if r.status_code == 200:
            return "lmstudio", "LM Studio detected on localhost:1234"
    except Exception:
        pass

    return None, "No local LLM detected."


# ============================================================
#  EVENT BUS (with self-healing wrapper)
# ============================================================

class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.lock = threading.Lock()

    def subscribe(self, event_type, handler, organ_name="UnknownOrgan"):
        def wrapped(payload):
            try:
                handler(payload)
            except Exception as e:
                print(f"[EVENT ERROR] {event_type} in {organ_name}: {e}")
                self.emit("ORGAN_FAILURE", {"organ": organ_name, "event": event_type, "error": str(e)})

        with self.lock:
            self.subscribers.setdefault(event_type, []).append(wrapped)

    def emit(self, event_type, payload=None):
        with self.lock:
            handlers = list(self.subscribers.get(event_type, []))
        for h in handlers:
            h(payload)


# ============================================================
#  ORGAN BASE + REGISTRY
# ============================================================

class Organ:
    def __init__(self, name, bus, state):
        self.name = name
        self.bus = bus
        self.state = state

    def register(self):
        pass


class OrganRegistry:
    def __init__(self):
        self.organs = []
        self.organ_by_name = {}

    def add(self, organ):
        self.organs.append(organ)
        self.organ_by_name[organ.name] = organ

    def register_all(self):
        for organ in self.organs:
            organ.register()

    def get(self, name):
        return self.organ_by_name.get(name)


# ============================================================
#  STATE CORE (now with personas, templates, feedback, session)
# ============================================================

class StateCore:
    def __init__(self):
        self.flags = {
            "clipboard_enabled": False,
            "keyboard_enabled": True,
        }

        self.health = "Unknown"

        self.keyboard_buffer = ""
        self.keyboard_last_time = time.time()
        self.keyboard_lock = threading.Lock()

        self.last_intent_info = None
        self.last_mode_trend = "No mode history yet."
        self.last_context_summary = "No context yet."

        self.llm_backend = None
        self.llm_status = "Unknown"

        self.critique_depth = 1
        self.behavior_profile = "No profile yet."
        self.typing_stats = {
            "total_chars": 0,
            "total_inputs": 0,
            "avg_len": 0.0,
            "code_ratio": 0.0,
            "last_sources": deque(maxlen=20),
        }

        self.dependencies = {
            "psutil": {"package": "psutil", "status": "unknown", "last_error": None},
            "requests": {"package": "requests", "status": "unknown", "last_error": None},
            "win32api": {"package": "pywin32", "status": "unknown", "last_error": None},
            "pyperclip": {"package": "pyperclip", "status": "unknown", "last_error": None},
        }

        # Personas per intent
        self.personas = {
            "CODE": """
You are a surgical coding assistant.
Priorities:
- Correctness
- Edge cases
- Robustness
- Performance awareness
Style:
- Concrete examples
- Step-by-step reasoning
- Explicit assumptions
            """.strip(),
            "DEBUG": """
You are a diagnostic engine.
Priorities:
- Hypothesis generation
- Systematic elimination
- Error pattern recognition
Style:
- Ask what is unknown
- Offer multiple hypotheses
- Propose small test steps
            """.strip(),
            "DOCS": """
You are an explainer.
Priorities:
- Clarity
- Layered explanations
- Good structure
Style:
- From simple to advanced
- Use analogies
- Summarize key points
            """.strip(),
            "RESEARCH": """
You are a research synthesizer.
Priorities:
- Tradeoffs
- Alternatives
- Uncertainty
Style:
- Compare options
- Highlight risks
- Suggest next experiments
            """.strip(),
            "CHAT": """
You are a concise, helpful collaborator.
Priorities:
- Relevance
- Brevity
- Supportive tone
Style:
- Direct answers
- Minimal fluff
- Offer options when unclear
            """.strip(),
            "UNKNOWN": """
Fallback mode.
Be conservative, ask clarifying questions, and avoid overconfident claims.
            """.strip(),
        }

        # Prepared reasoning templates per intent/mode
        self.prepared_prompts = {}
        self.last_mode_sequence = deque(maxlen=20)
        self.mode_transition_counts = defaultdict(Counter)
        self.forecast_next_mode = "UNKNOWN"

        # Session profile (time-of-day)
        self.session_profile = {
            "time_bucket": "unknown",
            "dominant_intents": Counter(),
        }

        # Outcome feedback: how often config leads to corrections vs "good"
        # key: (intent, critique_depth) -> {"good": n, "bad": m}
        self.outcome_stats = defaultdict(lambda: {"good": 0, "bad": 0})

        # For ClarificationOrgan
        self.pending_clarification = False
        self.last_user_input_raw = ""
        self.last_answer_was_weak = False


# ============================================================
#  INPUT MANAGER (pyperclip + pywin32 keyboard)
# ============================================================

class InputManager(Organ):
    def __init__(self, bus, state):
        super().__init__("InputManager", bus, state)
        self.hotword = "@@"  # keyboard trigger

    def register(self):
        threading.Thread(target=self.console_loop, daemon=True).start()
        threading.Thread(target=self.clipboard_loop, daemon=True).start()
        threading.Thread(target=self.keyboard_poll_loop, daemon=True).start()
        threading.Thread(target=self.keyboard_idle_monitor, daemon=True).start()

    def emit_text(self, text, source):
        self.state.last_user_input_raw = text
        self.bus.emit("INPUT_TEXT", {"text": text, "source": source})

    def console_loop(self):
        while True:
            try:
                text = input("You> ").strip()
                if text:
                    self.emit_text(text, "console")
            except Exception:
                break

    def clipboard_loop(self):
        if not PYPERCLIP_AVAILABLE:
            print("[Clipboard] pyperclip not installed — clipboard disabled.")
            return

        last_clip = ""
        while True:
            try:
                if self.state.flags["clipboard_enabled"]:
                    clip = pyperclip.paste()
                    if clip and clip != last_clip:
                        last_clip = clip
                        self.emit_text(clip, "clipboard")
                time.sleep(1)
            except Exception:
                time.sleep(1)

    def keyboard_poll_loop(self, poll_interval=0.02):
        global WIN32_AVAILABLE, win32api, win32con

        if not sys.platform.lower().startswith("win"):
            print("[Keyboard] Non-Windows platform detected — keyboard monitoring disabled.")
            return

        pressed = set()

        while True:
            try:
                if not self.state.flags["keyboard_enabled"]:
                    time.sleep(poll_interval)
                    continue

                if not WIN32_AVAILABLE:
                    try:
                        import win32api as _wapi
                        import win32con as _wcon
                        win32api = _wapi
                        win32con = _wcon
                        WIN32_AVAILABLE = True
                        print("[Keyboard] win32api successfully imported after autoloader.")
                    except Exception:
                        time.sleep(1.0)
                        continue

                now = time.time()

                vkeys = []
                vkeys.extend(range(0x30, 0x3A))   # 0-9
                vkeys.extend(range(0x41, 0x5B))   # A-Z
                vkeys.append(win32con.VK_SPACE)
                vkeys.append(win32con.VK_RETURN)

                for vk in vkeys:
                    state = win32api.GetAsyncKeyState(vk)
                    if state & 0x8000:
                        if vk not in pressed:
                            pressed.add(vk)
                            ch = self._vk_to_char(vk)
                            if ch is not None:
                                with self.state.keyboard_lock:
                                    self.state.keyboard_last_time = now
                                    self.state.keyboard_buffer += ch
                                    self.bus.emit("KEYBOARD_ACTIVITY", {"char": ch, "time": now})

                                    buf = self.state.keyboard_buffer
                                    if self.hotword in buf:
                                        idx = buf.rfind(self.hotword)
                                        clean = buf[:idx]
                                        sentence = self._extract_last_sentence(clean)
                                        if sentence:
                                            self.emit_text(sentence, "keyboard_hotword")
                                        self.state.keyboard_buffer = ""
                    else:
                        if vk in pressed:
                            pressed.remove(vk)

                time.sleep(poll_interval)
            except Exception:
                time.sleep(poll_interval)

    def _vk_to_char(self, vk):
        if win32api is None or win32con is None:
            return None

        if vk == win32con.VK_SPACE:
            return " "
        if vk == win32con.VK_RETURN:
            return "\n"

        shift = win32api.GetKeyState(win32con.VK_SHIFT) < 0

        if 0x30 <= vk <= 0x39:
            base = "0123456789"[vk - 0x30]
            return base

        if 0x41 <= vk <= 0x5A:
            ch = chr(vk)
            return ch if shift else ch.lower()

        return None

    def keyboard_idle_monitor(self, idle_threshold=0.8, poll=0.3):
        while True:
            try:
                if not self.state.flags["keyboard_enabled"]:
                    time.sleep(poll)
                    continue

                now = time.time()
                with self.state.keyboard_lock:
                    buf = self.state.keyboard_buffer
                    last = self.state.keyboard_last_time

                    if buf and any(ch in buf for ch in [".", "?", "!", "\n"]):
                        if (now - last) > idle_threshold:
                            sentence = self._extract_last_sentence(buf)
                            if sentence:
                                self.emit_text(sentence, "keyboard_idle")
                                self.bus.emit("KEYBOARD_PREDICTIVE_EVENT", {
                                    "type": "idle_sentence",
                                    "sentence": sentence,
                                    "time": now,
                                })
                            self.state.keyboard_buffer = ""

                time.sleep(poll)
            except Exception:
                time.sleep(poll)

    def _extract_last_sentence(self, text):
        for sep in [".", "?", "!", "\n"]:
            text = text.replace(sep, sep + "|")
        parts = [p.strip() for p in text.split("|") if p.strip()]
        return parts[-1] if parts else text.strip()


# ============================================================
#  INTENT ROUTER
# ============================================================

class IntentRouterOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("IntentRouterOrgan", bus, state)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_input, self.name)

    def on_input(self, payload):
        text = payload["text"].lower()

        if any(k in text for k in ["def ", "class ", "import ", "function", "python", "tkinter", "pywin32"]):
            intent = "CODE"
            conf = 0.8
            expl = "Detected code-like keywords."
        elif any(k in text for k in ["bug", "exception", "crash", "traceback", "error"]):
            intent = "DEBUG"
            conf = 0.8
            expl = "Debugging-related keywords."
        elif any(k in text for k in ["doc", "explain", "describe", "documentation", "guide", "tutorial"]):
            intent = "DOCS"
            conf = 0.75
            expl = "Documentation-related keywords."
        elif any(k in text for k in ["research", "study", "paper", "article", "compare", "tradeoff"]):
            intent = "RESEARCH"
            conf = 0.75
            expl = "Research-related keywords."
        else:
            intent = "CHAT"
            conf = 0.55
            expl = "General conversation / ambiguous."

        info = {
            "intent": intent,
            "confidence": conf,
            "explanation": expl,
            "source": payload["source"],
        }

        self.state.last_intent_info = info
        self.bus.emit("INTENT_DETECTED", info)


# ============================================================
#  PREDICTION ORGAN (topics + mode drift + sequence forecasting)
# ============================================================

class PredictionOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("PredictionOrgan", bus, state)
        self.tokens = Counter()
        self.modes = Counter()
        self.mode_history = deque(maxlen=50)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_text, self.name)
        self.bus.subscribe("INTENT_DETECTED", self.on_intent, self.name)

    def on_text(self, payload):
        text = payload["text"].lower()
        for t in text.split():
            if len(t) >= 4:
                self.tokens[t] += 1

        if not self.tokens:
            self.bus.emit("PREDICTION_UPDATED", "Not enough data.")
            return

        top = self.tokens.most_common(5)
        msg = "Likely topics: " + ", ".join(f"{w} ({c})" for w, c in top)
        self.bus.emit("PREDICTION_UPDATED", msg)

    def on_intent(self, payload):
        mode = payload["intent"]
        self.modes[mode] += 1
        self.mode_history.append(mode)

        # Maintain sequence-based transition stats
        if self.state.last_mode_sequence:
            prev = self.state.last_mode_sequence[-1]
            self.state.mode_transition_counts[prev][mode] += 1
        self.state.last_mode_sequence.append(mode)

        # Trend summary
        top = self.modes.most_common(3)
        trend = ", ".join(f"{m} ({c})" for m, c in top)
        current = mode
        msg = f"Current mode: {current} | Trend: {trend}"
        self.state.last_mode_trend = msg
        self.bus.emit("MODE_TREND_UPDATED", msg)

        # Simple drift forecast
        code_like = self.modes["CODE"] + self.modes["DEBUG"]
        chat_like = self.modes["CHAT"] + self.modes["DOCS"]
        if code_like > chat_like:
            drift = "CODE/DEBUG drift"
        else:
            drift = "CHAT/DOCS drift"

        # Sequence-aware next-mode forecast
        next_mode = "UNKNOWN"
        transitions = self.state.mode_transition_counts[current]
        if transitions:
            next_mode = transitions.most_common(1)[0][0]
        self.state.forecast_next_mode = next_mode

        self.bus.emit("MODE_DRIFT_FORECAST", {
            "current": current,
            "forecast": drift,
            "trend": trend,
            "next_mode": next_mode,
        })


# ============================================================
#  CONTEXT MEMORY
# ============================================================

class ContextMemoryOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("ContextMemoryOrgan", bus, state)
        self.topics = Counter()
        self.entities = Counter()
        self.actions = deque(maxlen=20)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_text, self.name)
        self.bus.subscribe("BRAIN_OUTPUT", self.on_brain, self.name)

    def on_text(self, payload):
        self._observe(payload["text"], "USER")
        self._emit()

    def on_brain(self, text):
        self._observe(text, "BRAIN")
        self._emit()

    def _observe(self, text, role):
        for t in text.split():
            if len(t) >= 5:
                self.topics[t.lower()] += 1

        for token in text.split():
            if token.istitle() and len(token) >= 3:
                self.entities[token.strip(",.!?")] += 1

        ts = time.strftime("%H:%M:%S")
        self.actions.append(f"{ts} - Observed {role} text.")

    def _emit(self):
        top_topics = self.topics.most_common(5)
        top_entities = self.entities.most_common(5)

        summary = (
            "Recent focus: "
            + ", ".join(k for k, _ in top_topics[:3])
            if top_topics else "No clear focus."
        )

        self.state.last_context_summary = summary

        lines = [
            "=== Context Memory ===",
            "Topics: " + ", ".join(f"{k} ({v})" for k, v in top_topics),
            "Entities: " + ", ".join(f"{k} ({v})" for k, v in top_entities),
            "",
            "Recent actions:",
        ]
        lines.extend("  - " + a for a in self.actions)
        lines.append("")
        lines.append("Summary:")
        lines.append("  " + summary)

        self.bus.emit("CONTEXT_UPDATED", "\n".join(lines))


# ============================================================
#  BEHAVIORAL PROFILING ENGINE
# ============================================================

class BehavioralProfileOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("BehavioralProfileOrgan", bus, state)
        self.last_key_time = None
        self.key_intervals = deque(maxlen=100)
        self.input_lengths = deque(maxlen=100)
        self.code_inputs = 0
        self.total_inputs = 0

    def register(self):
        self.bus.subscribe("KEYBOARD_ACTIVITY", self.on_key, self.name)
        self.bus.subscribe("INPUT_TEXT", self.on_input_text, self.name)

    def on_key(self, payload):
        t = payload["time"]
        if self.last_key_time is not None:
            interval = t - self.last_key_time
            self.key_intervals.append(interval)
        self.last_key_time = t

    def on_input_text(self, payload):
        text = payload["text"]
        source = payload["source"]
        length = len(text)

        self.input_lengths.append(length)
        self.state.typing_stats["total_chars"] += length
        self.state.typing_stats["total_inputs"] += 1
        self.state.typing_stats["last_sources"].append(source)

        is_code_like = any(k in text.lower() for k in ["def ", "class ", "import ", "{", "}", ";"])
        if is_code_like:
            self.code_inputs += 1
        self.total_inputs += 1

        avg_len = sum(self.input_lengths) / len(self.input_lengths) if self.input_lengths else 0
        code_ratio = self.code_inputs / self.total_inputs if self.total_inputs else 0

        self.state.typing_stats["avg_len"] = avg_len
        self.state.typing_stats["code_ratio"] = code_ratio

        typing_speed = 0.0
        if self.key_intervals:
            avg_interval = sum(self.key_intervals) / len(self.key_intervals)
            typing_speed = 1.0 / avg_interval if avg_interval > 0 else 0.0

        profile = (
            f"Avg input length: {avg_len:.1f} chars | "
            f"Code ratio: {code_ratio:.2f} | "
            f"Estimated typing speed: {typing_speed:.2f} keys/sec"
        )
        self.state.behavior_profile = profile
        self.bus.emit("BEHAVIOR_PROFILE_UPDATED", profile)


# ============================================================
#  SESSION PROFILE ORGAN (time-of-day + dominant intents)
# ============================================================

class SessionProfileOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("SessionProfileOrgan", bus, state)
        self.intent_counts = Counter()

    def register(self):
        self.bus.subscribe("INTENT_DETECTED", self.on_intent, self.name)
        threading.Thread(target=self.loop, daemon=True).start()

    def _current_bucket(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late-night"

    def on_intent(self, payload):
        intent = payload["intent"]
        self.intent_counts[intent] += 1

    def loop(self):
        while True:
            bucket = self._current_bucket()
            self.state.session_profile["time_bucket"] = bucket
            self.state.session_profile["dominant_intents"] = self.intent_counts.copy()
            time.sleep(30)


# ============================================================
#  ANOMALY DETECTOR
# ============================================================

class AnomalyDetectorOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("AnomalyDetectorOrgan", bus, state)
        self.last = None
        self.anomalies = deque(maxlen=20)

    def register(self):
        self.bus.subscribe("BRAIN_OUTPUT", self.on_brain, self.name)

    def on_brain(self, text):
        msgs = []

        if len(text) < 40:
            msgs.append("Output very short.")
            self.state.last_answer_was_weak = True
        else:
            self.state.last_answer_was_weak = False

        if "[LOCAL LLM ERROR" in text:
            msgs.append("Local LLM error detected.")

        if "[COPILOT PLACEHOLDER]" in text:
            msgs.append("Copilot stub used.")

        if self.last:
            a = set(self.last.split())
            b = set(text.split())
            if a and b:
                sim = len(a & b) / len(a | b)
                if sim > 0.9:
                    msgs.append(f"High similarity to previous output ({sim:.2f}).")

        self.last = text

        anomaly_payload = None
        if msgs:
            ts = time.strftime("%H:%M:%S")
            entry = f"{ts} - " + "; ".join(msgs)
            self.anomalies.append(entry)
            anomaly_payload = entry

        if not self.anomalies:
            self.bus.emit("ANOMALY_UPDATED", "No anomalies detected.")
        else:
            lines = ["=== Anomalies ==="]
            lines.extend("  - " + a for a in self.anomalies)
            self.bus.emit("ANOMALY_UPDATED", "\n".join(lines))

        if anomaly_payload:
            self.bus.emit("ANOMALY_EVENT", {"message": anomaly_payload})


# ============================================================
#  LLM STATUS ORGAN
# ============================================================

class LLMStatusOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("LLMStatusOrgan", bus, state)

    def register(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            backend, detail = detect_local_llm()
            self.state.llm_backend = backend
            self.state.llm_status = detail

            self.bus.emit("LLM_STATUS_UPDATED", {
                "backend": backend,
                "detail": detail,
                "status": detail,
            })

            time.sleep(15)


# ============================================================
#  META-REASONING ORGAN (adaptive critique depth)
# ============================================================

class MetaReasoningOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("MetaReasoningOrgan", bus, state)
        self.last_anomaly_time = None

    def register(self):
        self.bus.subscribe("ANOMALY_EVENT", self.on_anomaly, self.name)
        self.bus.subscribe("BEHAVIOR_PROFILE_UPDATED", self.on_behavior, self.name)

    def on_anomaly(self, payload):
        self.last_anomaly_time = time.time()
        self.state.critique_depth = 2

    def on_behavior(self, payload):
        code_ratio = self.state.typing_stats.get("code_ratio", 0.0)
        if code_ratio > 0.5:
            self.state.critique_depth = max(self.state.critique_depth, 2)
        else:
            if self.last_anomaly_time and (time.time() - self.last_anomaly_time) > 120:
                self.state.critique_depth = 1


# ============================================================
#  REASONING TEMPLATE ORGAN (prepares prompts per intent + trend)
# ============================================================

class ReasoningTemplateOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("ReasoningTemplateOrgan", bus, state)

    def register(self):
        self.bus.subscribe("MODE_TREND_UPDATED", self.on_trend, self.name)
        self.bus.subscribe("INTENT_DETECTED", self.on_intent, self.name)

    def on_trend(self, payload):
        # payload is plain text; trend already in state.last_mode_trend
        self._rebuild_templates()

    def on_intent(self, payload):
        self._rebuild_templates()

    def _rebuild_templates(self):
        intent_info = self.state.last_intent_info or {"intent": "UNKNOWN", "confidence": 0.0}
        intent = intent_info["intent"]
        bucket = self.state.session_profile.get("time_bucket", "unknown")
        forecast_next = self.state.forecast_next_mode

        base = f"""
Session time bucket: {bucket}
Current mode: {intent}
Forecast next mode: {forecast_next}
Mode trend: {self.state.last_mode_trend}
Behavior profile: {self.state.behavior_profile}
Context summary: {self.state.last_context_summary}
        """.strip()

        templates = {}

        templates["CODE"] = base + """

You must:
- Ask for missing environment details if needed.
- Consider edge cases, error paths, and performance.
- Show minimal but complete examples.
"""
        templates["DEBUG"] = base + """

You must:
- Form 2-3 concrete hypotheses about the failure.
- Ask clarifying questions if crucial data is missing.
- Propose small, testable steps.
"""
        templates["DOCS"] = base + """

You must:
- Start with a high-level explanation.
- Then show a more detailed, structured breakdown.
- End with a quick summary.
"""
        templates["RESEARCH"] = base + """

You must:
- Compare at least two options.
- Highlight tradeoffs and uncertainties.
- Suggest follow-up experiments or checks.
"""
        templates["CHAT"] = base + """

You must:
- Answer briefly but clearly.
- Only expand if explicitly asked.
"""

        templates["UNKNOWN"] = base + """

You must:
- Be conservative.
- Ask at least one clarifying question before committing to a detailed answer.
"""

        self.state.prepared_prompts = templates


# ============================================================
#  OUTCOME FEEDBACK ORGAN (self-tuning behavior)
# ============================================================

class OutcomeFeedbackOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("OutcomeFeedbackOrgan", bus, state)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_input, self.name)

    def on_input(self, payload):
        text = payload["text"].lower()

        # Look for signals of dissatisfaction
        bad_markers = ["that's wrong", "not what i meant", "try again", "no,", "incorrect", "this is wrong"]
        good_markers = ["perfect", "that works", "exactly", "nice", "good", "thanks"]

        intent = self.state.last_intent_info["intent"] if self.state.last_intent_info else "UNKNOWN"
        depth = self.state.critique_depth

        key = (intent, depth)

        if any(m in text for m in bad_markers):
            self.state.outcome_stats[key]["bad"] += 1
        elif any(m in text for m in good_markers):
            self.state.outcome_stats[key]["good"] += 1

        # Adjust global critique depth gently based on aggregate stats
        stats = self.state.outcome_stats[key]
        total = stats["good"] + stats["bad"]
        if total >= 5:  # only react when we have some data
            bad_rate = stats["bad"] / total
            if bad_rate > 0.6 and self.state.critique_depth < 3:
                self.state.critique_depth += 1
            elif bad_rate < 0.25 and self.state.critique_depth > 1:
                self.state.critique_depth -= 1


# ============================================================
#  CLARIFICATION ORGAN (asks when unsure)
# ============================================================

class ClarificationOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("ClarificationOrgan", bus, state)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_input, self.name)

    def on_input(self, payload):
        text = payload["text"]
        intent_info = self.state.last_intent_info

        if self.state.pending_clarification:
            # The user is answering a clarification question; clear the flag.
            self.state.pending_clarification = False
            return

        if not intent_info:
            return

        intent = intent_info["intent"]
        conf = intent_info["confidence"]

        # Heuristics for when to ask:
        is_short = len(text.split()) <= 3
        ambiguous_intent = (intent == "CHAT" and conf < 0.6)
        weak_prev = self.state.last_answer_was_weak

        if is_short or ambiguous_intent or weak_prev:
            q = "Before I go deep: what are you actually trying to achieve here? (Goal / context / constraints?)"
            self.state.pending_clarification = True
            self.bus.emit("BRAIN_OUTPUT", "[CLARIFICATION] " + q)


# ============================================================
#  SMART AUTOLOADER ORGAN (install + smart update)
# ============================================================

class AutoloaderOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("AutoloaderOrgan", bus, state)
        self.failure_counts = {}

    def register(self):
        threading.Thread(target=self.initial_scan, daemon=True).start()
        self.bus.subscribe("ORGAN_FAILURE", self.on_organ_failure, self.name)

    def initial_scan(self):
        for mod_name, info in self.state.dependencies.items():
            pkg = info["package"]
            try:
                __import__(mod_name)
                info["status"] = "ok"
                info["last_error"] = None
                self.bus.emit("DEPENDENCY_STATUS", {"module": mod_name, "status": "ok"})
            except Exception as e:
                info["status"] = "missing"
                info["last_error"] = str(e)
                self.bus.emit("DEPENDENCY_STATUS", {"module": mod_name, "status": "missing", "error": str(e)})
                self.install_dependency(mod_name, pkg)

    def install_dependency(self, mod_name, pkg):
        self.bus.emit("DEPENDENCY_INSTALL_STARTED", {"module": mod_name, "package": pkg})
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            __import__(mod_name)
            self.state.dependencies[mod_name]["status"] = "ok"
            self.state.dependencies[mod_name]["last_error"] = None
            self.bus.emit("DEPENDENCY_INSTALL_SUCCESS", {"module": mod_name, "package": pkg})
        except Exception as e:
            self.state.dependencies[mod_name]["status"] = "failed"
            self.state.dependencies[mod_name]["last_error"] = str(e)
            self.bus.emit("DEPENDENCY_INSTALL_FAILED", {
                "module": mod_name,
                "package": pkg,
                "error": str(e),
            })

    def update_dependency(self, mod_name, pkg):
        self.bus.emit("DEPENDENCY_UPDATE_STARTED", {"module": mod_name, "package": pkg})
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
            __import__(mod_name)
            self.state.dependencies[mod_name]["status"] = "ok"
            self.state.dependencies[mod_name]["last_error"] = None
            self.bus.emit("DEPENDENCY_UPDATE_SUCCESS", {"module": mod_name, "package": pkg})
        except Exception as e:
            self.state.dependencies[mod_name]["status"] = "update_failed"
            self.state.dependencies[mod_name]["last_error"] = str(e)
            self.bus.emit("DEPENDENCY_UPDATE_FAILED", {
                "module": mod_name,
                "package": pkg,
                "error": str(e),
            })

    def on_organ_failure(self, payload):
        error = payload.get("error", "")
        organ_name = payload.get("organ", "UnknownOrgan")

        if "No module named" in error or "ModuleNotFoundError" in error or "ImportError" in error:
            target = None
            parts = error.split("'")
            if len(parts) >= 2:
                target = parts[1]

            if target and target in self.state.dependencies:
                info = self.state.dependencies[target]
                status = info["status"]

                self.failure_counts[target] = self.failure_counts.get(target, 0) + 1
                count = self.failure_counts[target]

                if status in ("missing", "failed"):
                    self.install_dependency(target, info["package"])
                else:
                    if count >= 3:
                        self.update_dependency(target, info["package"])
        else:
            pass


# ============================================================
#  BRAIN ORGAN (persona-based, templates, adaptive depth)
# ============================================================

class BrainOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("BrainOrgan", bus, state)

    def register(self):
        self.bus.subscribe("INPUT_TEXT", self.on_input, self.name)

    def on_input(self, payload):
        # ClarificationOrgan might have just asked something; if so, skip heavy reasoning once.
        if self.state.pending_clarification:
            return

        text = payload["text"]
        source = payload["source"]

        intent_info = self.state.last_intent_info or {"intent": "UNKNOWN", "confidence": 0.0, "explanation": "None"}
        intent = intent_info["intent"]
        conf = intent_info["confidence"]
        expl = intent_info["explanation"]

        persona = self.state.personas.get(intent, self.state.personas["UNKNOWN"])
        template = self.state.prepared_prompts.get(intent, "")

        # Compose system prompt
        system_prompt = f"""
You are my personal neural core.

=== Persona (mode: {intent}) ===
{persona}

=== Session & Context ===
{template}

Critique depth setting: {self.state.critique_depth}

Rules:
- Respect the persona priorities.
- Use the session/context information.
- When uncertain, ask focused clarifying questions instead of guessing.
        """.strip()

        draft, ok = call_copilot_stub(text, system=system_prompt, max_tokens=600)
        used_backend = "copilot_stub"

        if not ok:
            backend = self.state.llm_backend
            if backend is None:
                self.bus.emit("LLM_MISSING", {
                    "message": "No local LLM detected. Install LM Studio or Ollama.",
                })
            else:
                used_backend = backend
                if backend == "ollama":
                    draft = call_ollama_llm(text, system=system_prompt, max_tokens=600)
                elif backend == "lmstudio":
                    draft = call_lmstudio_llm(text, system=system_prompt, max_tokens=600)
                else:
                    draft = call_ollama_llm(text, system=system_prompt, max_tokens=600)

        critique_depth = self.state.critique_depth
        if used_backend in ("ollama", "lmstudio"):
            critique = llm_critique(draft, backend=used_backend)
            refined = llm_refine(draft, critique, backend=used_backend)
            if critique_depth > 1:
                critique2 = llm_critique(refined, backend=used_backend)
                refined2 = llm_refine(refined, critique2, backend=used_backend)
                refined = refined2 + "\n\n[Meta] Second-pass refinement applied."
        else:
            critique = "[CRITIQUE] Not available for this backend (stub)."
            refined = draft

        output_lines = []
        output_lines.append(f"[BRAIN] Source: {source}")
        output_lines.append(f"[BRAIN] Backend: {used_backend}")
        output_lines.append(f"[BRAIN] Intent: {intent} (conf={conf:.2f})")
        output_lines.append(f"[BRAIN] Critique depth: {critique_depth}")
        output_lines.append("")
        output_lines.append("=== DRAFT ===")
        output_lines.append(draft)
        output_lines.append("")
        output_lines.append("=== CRITIQUE ===")
        output_lines.append(critique)
        output_lines.append("")
        output_lines.append("=== REFINED ===")
        output_lines.append(refined)

        self.bus.emit("BRAIN_OUTPUT", "\n".join(output_lines))


# ============================================================
#  DIAGNOSTICS ORGAN
# ============================================================

class DiagnosticsOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("DiagnosticsOrgan", bus, state)

    def register(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            self.state.health = f"CPU: {cpu:.1f}% | RAM: {ram:.1f}%"
            self.bus.emit("HEALTH_UPDATED", self.state.health)
            time.sleep(2)


# ============================================================
#  RESILIENCE ORGAN
# ============================================================

class ResilienceOrgan(Organ):
    def __init__(self, bus, state):
        super().__init__("ResilienceOrgan", bus, state)

    def register(self):
        self.bus.subscribe("ORGAN_FAILURE", self.on_failure, self.name)

    def on_failure(self, payload):
        organ = payload.get("organ", "UnknownOrgan")
        error = payload.get("error", "Unknown error")
        msg = f"[RESILIENCE] Organ failure detected: {organ} | {error}"
        self.bus.emit("BRAIN_OUTPUT", msg)


# ============================================================
#  GUI ORGAN (Tkinter)
# ============================================================

class GUIOrgan(Organ):
    def __init__(self, root, bus, state):
        super().__init__("GUIOrgan", bus, state)
        self.root = root
        self.root.title("NEURAL CORE V3 — Tkinter organism (predictive+adaptive)")
        self.root.geometry("1800x900")
        self._build_ui()

    def register(self):
        self.bus.subscribe("BRAIN_OUTPUT", self.on_brain_output, self.name)
        self.bus.subscribe("HEALTH_UPDATED", self.on_health_updated, self.name)
        self.bus.subscribe("PREDICTION_UPDATED", self.on_prediction_updated, self.name)
        self.bus.subscribe("MODE_TREND_UPDATED", self.on_mode_trend_updated, self.name)
        self.bus.subscribe("MODE_DRIFT_FORECAST", self.on_mode_drift, self.name)
        self.bus.subscribe("CONTEXT_UPDATED", self.on_context_updated, self.name)
        self.bus.subscribe("INTENT_DETECTED", self.on_intent_detected, self.name)
        self.bus.subscribe("ANOMALY_UPDATED", self.on_anomaly_updated, self.name)
        self.bus.subscribe("LLM_STATUS_UPDATED", self.on_llm_status_updated, self.name)
        self.bus.subscribe("LLM_MISSING", self.on_llm_missing, self.name)
        self.bus.subscribe("BEHAVIOR_PROFILE_UPDATED", self.on_behavior_profile, self.name)
        self.bus.subscribe("DEPENDENCY_STATUS", self.on_dependency_status, self.name)
        self.bus.subscribe("DEPENDENCY_INSTALL_STARTED", self.on_dependency_install_started, self.name)
        self.bus.subscribe("DEPENDENCY_INSTALL_SUCCESS", self.on_dependency_install_success, self.name)
        self.bus.subscribe("DEPENDENCY_INSTALL_FAILED", self.on_dependency_install_failed, self.name)
        self.bus.subscribe("DEPENDENCY_UPDATE_STARTED", self.on_dependency_update_started, self.name)
        self.bus.subscribe("DEPENDENCY_UPDATE_SUCCESS", self.on_dependency_update_success, self.name)
        self.bus.subscribe("DEPENDENCY_UPDATE_FAILED", self.on_dependency_update_failed, self.name)

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=4)
        main.columnconfigure(2, weight=3)
        main.rowconfigure(0, weight=1)

        # Left column
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Mind Stream").grid(row=0, column=0, sticky="w")
        self.stream = ScrolledText(left, wrap="word", height=20)
        self.stream.grid(row=1, column=0, sticky="nsew")

        self.health_label = ttk.Label(left, text="Health: Unknown")
        self.health_label.grid(row=2, column=0, sticky="w", pady=(5, 0))

        self.clip_label = ttk.Label(left, text="Clipboard: OFF")
        self.clip_label.grid(row=3, column=0, sticky="w")

        clip_btn_frame = ttk.Frame(left)
        clip_btn_frame.grid(row=4, column=0, sticky="w")
        self.clip_btn_on = ttk.Button(clip_btn_frame, text="Clipboard ON", command=self.enable_clipboard)
        self.clip_btn_off = ttk.Button(clip_btn_frame, text="Clipboard OFF", command=self.disable_clipboard)
        self.clip_btn_on.pack(side="left")
        self.clip_btn_off.pack(side="left", padx=(5, 0))

        self.key_label = ttk.Label(left, text="Keyboard Monitor: ON")
        self.key_label.grid(row=5, column=0, sticky="w", pady=(5, 0))

        key_btn_frame = ttk.Frame(left)
        key_btn_frame.grid(row=6, column=0, sticky="w")
        self.key_btn_on = ttk.Button(key_btn_frame, text="Keyboard ON", command=self.enable_keyboard)
        self.key_btn_off = ttk.Button(key_btn_frame, text="Keyboard OFF", command=self.disable_keyboard)
        self.key_btn_on.pack(side="left")
        self.key_btn_off.pack(side="left", padx=(5, 0))

        # Center column
        center = ttk.Frame(main)
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        for r in range(10):
            center.rowconfigure(r, weight=1)

        ttk.Label(center, text="Prediction / Topics").grid(row=0, column=0, sticky="w")
        self.prediction_view = ScrolledText(center, wrap="word", height=4)
        self.prediction_view.grid(row=1, column=0, sticky="nsew")

        ttk.Label(center, text="Mode Trend").grid(row=2, column=0, sticky="w")
        self.mode_view = ScrolledText(center, wrap="word", height=4)
        self.mode_view.grid(row=3, column=0, sticky="nsew")

        ttk.Label(center, text="Mode Drift Forecast + Next Mode").grid(row=4, column=0, sticky="w")
        self.mode_drift_view = ScrolledText(center, wrap="word", height=4)
        self.mode_drift_view.grid(row=5, column=0, sticky="nsew")

        ttk.Label(center, text="Behavior Profile").grid(row=6, column=0, sticky="w")
        self.behavior_view = ScrolledText(center, wrap="word", height=4)
        self.behavior_view.grid(row=7, column=0, sticky="nsew")

        ttk.Label(center, text="Context Memory").grid(row=8, column=0, sticky="w")
        self.context_view = ScrolledText(center, wrap="word", height=8)
        self.context_view.grid(row=9, column=0, sticky="nsew")

        # Right column
        right = ttk.Frame(main)
        right.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        for r in range(8):
            right.rowconfigure(r, weight=1)

        ttk.Label(right, text="Intent Router").grid(row=0, column=0, sticky="w")
        self.intent_view = ScrolledText(right, wrap="word", height=6)
        self.intent_view.grid(row=1, column=0, sticky="nsew")

        ttk.Label(right, text="Anomaly Detector").grid(row=2, column=0, sticky="w")
        self.anomaly_view = ScrolledText(right, wrap="word", height=6)
        self.anomaly_view.grid(row=3, column=0, sticky="nsew")

        ttk.Label(right, text="LLM Status / Install").grid(row=4, column=0, sticky="w")
        self.llm_status_view = ScrolledText(right, wrap="word", height=6)
        self.llm_status_view.grid(row=5, column=0, sticky="nsew")

        ttk.Label(right, text="Dependency Status / Autoloader").grid(row=6, column=0, sticky="w")
        self.dependency_view = ScrolledText(right, wrap="word", height=6)
        self.dependency_view.grid(row=7, column=0, sticky="nsew")

    # ---- GUI update helpers (thread-safe using .after) ----

    def _append_text(self, widget, text):
        def inner():
            widget.insert("end", text)
            widget.insert("end", "\n")
            widget.see("end")
        self.root.after(0, inner)

    def _set_text(self, widget, text):
        def inner():
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.see("end")
        self.root.after(0, inner)

    def _set_label(self, label, text):
        def inner():
            label.config(text=text)
        self.root.after(0, inner)

    # ---- Event handlers ----

    def on_brain_output(self, text):
        self._append_text(self.stream, text)

    def on_health_updated(self, text):
        self._set_label(self.health_label, f"Health: {text}")

    def on_prediction_updated(self, text):
        self._set_text(self.prediction_view, text)

    def on_mode_trend_updated(self, text):
        self._set_text(self.mode_view, text)

    def on_mode_drift(self, payload):
        if not payload:
            return
        current = payload["current"]
        forecast = payload["forecast"]
        trend = payload["trend"]
        next_mode = payload.get("next_mode", "UNKNOWN")
        txt = f"Current: {current}\nDrift: {forecast}\nTrend: {trend}\nSequence-forecast next mode: {next_mode}"
        self._set_text(self.mode_drift_view, txt)

    def on_context_updated(self, text):
        self._set_text(self.context_view, text)

    def on_intent_detected(self, payload):
        if not payload:
            return
        intent = payload["intent"]
        conf = payload["confidence"]
        expl = payload["explanation"]
        src = payload["source"]
        txt = f"Source: {src}\nIntent: {intent} (conf={conf:.2f})\nExplanation: {expl}"
        self._set_text(self.intent_view, txt)

    def on_anomaly_updated(self, text):
        self._set_text(self.anomaly_view, text)

    def on_llm_status_updated(self, payload):
        if not payload:
            return
        status = payload["status"]
        lines = []
        lines.append(f"Local LLM Status: {status}")
        lines.append("")
        lines.append("Recommended Local LLM Runtimes:")
        lines.append(" - LM Studio (https://lmstudio.ai)")
        lines.append(" - Ollama (https://ollama.com)")
        lines.append("")
        lines.append("Helpful Guides:")
        lines.append(" - Run Ollama locally (freeCodeCamp)")
        lines.append(" - Running LLMs locally (GitHub: di37/running-llms-locally)")
        self._set_text(self.llm_status_view, "\n".join(lines))

    def on_llm_missing(self, payload):
        def inner():
            existing = self.llm_status_view.get("1.0", "end").rstrip("\n")
            extra = "\n\n[LLM MISSING] No local LLM detected. Install LM Studio or Ollama."
            self.llm_status_view.delete("1.0", "end")
            self.llm_status_view.insert("1.0", existing + extra)
            self.llm_status_view.see("end")
        self.root.after(0, inner)

    def on_behavior_profile(self, text):
        self._set_text(self.behavior_view, text)

    def on_dependency_status(self, payload):
        if not payload:
            return
        module = payload.get("module")
        status = payload.get("status")
        error = payload.get("error")
        line = f"[STATUS] {module}: {status}"
        if error:
            line += f" (error: {error})"
        self._append_text(self.dependency_view, line)

    def on_dependency_install_started(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        line = f"[INSTALL] Starting install of {pkg} for module {module}..."
        self._append_text(self.dependency_view, line)

    def on_dependency_install_success(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        line = f"[INSTALL] Success: {pkg} for module {module}"
        self._append_text(self.dependency_view, line)

    def on_dependency_install_failed(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        error = payload["error"]
        line = f"[INSTALL] FAILED: {pkg} for module {module} (error: {error})"
        self._append_text(self.dependency_view, line)

    def on_dependency_update_started(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        line = f"[UPDATE] Starting update of {pkg} for module {module}..."
        self._append_text(self.dependency_view, line)

    def on_dependency_update_success(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        line = f"[UPDATE] Success: {pkg} for module {module}"
        self._append_text(self.dependency_view, line)

    def on_dependency_update_failed(self, payload):
        module = payload["module"]
        pkg = payload["package"]
        error = payload["error"]
        line = f"[UPDATE] FAILED: {pkg} for module {module} (error: {error})"
        self._append_text(self.dependency_view, line)

    # ---- Controls ----

    def enable_clipboard(self):
        if not PYPERCLIP_AVAILABLE:
            self._append_text(self.stream, "[Clipboard] pyperclip is not installed. Clipboard monitoring disabled.")
            self._set_label(self.clip_label, "Clipboard: UNAVAILABLE")
            return
        self.state.flags["clipboard_enabled"] = True
        self._set_label(self.clip_label, "Clipboard: ON")

    def disable_clipboard(self):
        self.state.flags["clipboard_enabled"] = False
        self._set_label(self.clip_label, "Clipboard: OFF")

    def enable_keyboard(self):
        self.state.flags["keyboard_enabled"] = True
        self._set_label(self.key_label, "Keyboard Monitor: ON")

    def disable_keyboard(self):
        self.state.flags["keyboard_enabled"] = False
        self._set_label(self.key_label, "Keyboard Monitor: OFF")


# ============================================================
#  MAIN
# ============================================================

def main():
    bus = EventBus()
    state = StateCore()
    registry = OrganRegistry()

    root = tk.Tk()
    gui = GUIOrgan(root, bus, state)

    registry.add(gui)
    registry.add(InputManager(bus, state))
    registry.add(IntentRouterOrgan(bus, state))
    registry.add(PredictionOrgan(bus, state))
    registry.add(ContextMemoryOrgan(bus, state))
    registry.add(BehavioralProfileOrgan(bus, state))
    registry.add(SessionProfileOrgan(bus, state))
    registry.add(AnomalyDetectorOrgan(bus, state))
    registry.add(LLMStatusOrgan(bus, state))
    registry.add(MetaReasoningOrgan(bus, state))
    registry.add(ReasoningTemplateOrgan(bus, state))
    registry.add(OutcomeFeedbackOrgan(bus, state))
    registry.add(ClarificationOrgan(bus, state))
    registry.add(ResilienceOrgan(bus, state))
    registry.add(AutoloaderOrgan(bus, state))
    registry.add(BrainOrgan(bus, state))
    registry.add(DiagnosticsOrgan(bus, state))

    registry.register_all()

    root.mainloop()


if __name__ == "__main__":
    main()

