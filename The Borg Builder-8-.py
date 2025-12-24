#!/usr/bin/env python
# neural_core.py
# Dark sci-fi neural core with:
# - Cross-platform design (Windows / Linux / macOS)
# - Local LLM via Ollama
# - Optional remote LLM connectors (config-driven)
# - Prediction Organ (topic + mode forecasting)
# - Context Memory Organ (live situation awareness)
# - Intent Router Organ (classifies input intent)
# - Anomaly Detector Organ (flags weird output patterns)
# - Keyboard Monitor Organ (hybrid: sentence boundaries + hotword)
# - Clipboard Listener Organ (toggle, 1s poll, only new text)
# - Self-critique pipeline with meta-reasoning (second-pass refinement on anomalies/weakness)
# - Diagnostics + Safe mode Organ (CPU/RAM/disk + network aware)
# - Web Probe Organ + GUI button (URL -> fetch -> extract -> distill)
# - GUI: health, safe mode, net, intent, anomalies, context, prediction, mind stream, evolution log

import sys
import subprocess
import importlib
import os
import json
import platform
import shutil
import time
import textwrap
from pathlib import Path
import threading
from collections import Counter, deque
import socket
from urllib.parse import urlparse
from html import unescape
import queue

# ========= AUTO-INSTALL REQUIRED PACKAGES =========

REQUIRED_PACKAGES = [
    "psutil",
    "PySide6",
    "requests",
    "pyperclip",
    "pynput",  # for keyboard monitor
]

def ensure_packages():
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[*] Missing package '{pkg}', installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print("[*] All required packages are present.\n")

ensure_packages()

# Now safe to import external packages
import psutil
import requests
import pyperclip
from pynput import keyboard
from PySide6 import QtWidgets, QtCore, QtGui


# ========= CONFIG / PATHS =========

APP_DIR = Path.home() / ".neural_core"
CONFIG_FILE = APP_DIR / "config.json"
MODELS_DIR = APP_DIR / "models"
VECTOR_DB_DIR = APP_DIR / "vector_db"
RAW_KNOWLEDGE_FILE = APP_DIR / "raw_knowledge.jsonl"
DISTILLED_DIR = APP_DIR / "distilled"
LOG_DIR = APP_DIR / "logs"
EVOLUTION_LOG = LOG_DIR / "evolution.log"
PREDICTION_LOG = LOG_DIR / "prediction.log"
DIAGNOSTICS_LOG = LOG_DIR / "diagnostics.log"
CONTEXT_MEMORY_FILE = APP_DIR / "context_memory.json"

for p in [APP_DIR, MODELS_DIR, VECTOR_DB_DIR, DISTILLED_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ========= GLOBAL STATE (SAFE MODE, HEALTH, CLIPBOARD, QUEUE, KEYBOARD) =========

STATE = {
    "safe_mode": False,
    "last_health_report": "Unknown",
    "clipboard_listener_enabled": False,
    "last_clipboard_text": "",
    "keyboard_monitor_enabled": True,  # always on for now; can wire to GUI later
}

INPUT_QUEUE: "queue.Queue[str]" = queue.Queue()


# ========= HELPER: HTML -> TEXT =========

def extract_text_from_html(html: str) -> str:
    import re
    html = re.sub(r"(?is)<(script|style).*?>.*?(</\\1>)", "", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ========= PROBE ORGAN =========

def detect_gpu():
    has_nvidia = shutil.which("nvidia-smi") is not None
    cuda_available = False
    if has_nvidia:
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            cuda_available = True
        except Exception:
            cuda_available = False
    return {
        "has_nvidia": has_nvidia,
        "cuda_available": cuda_available,
    }


def check_network(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def probe_environment():
    cpu_count = os.cpu_count() or 1
    ram_bytes = psutil.virtual_memory().total
    disk = psutil.disk_usage("/")

    gpu_info = detect_gpu()
    net_ok = check_network()

    env = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_count": cpu_count,
        "ram_gb": ram_bytes / (1024 ** 3),
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_free_gb": disk.free / (1024 ** 3),
        "network_available": net_ok,
        "gpu": gpu_info,
    }
    return env


# ========= BLUEPRINT ORGAN =========

def choose_blueprint(env: dict) -> dict:
    ram = env["ram_gb"]
    cpu = env["cpu_count"]
    gpu = env["gpu"]

    if gpu.get("cuda_available"):
        model_size = "medium"
        backend = "gpu"
    elif ram >= 24 and cpu >= 8:
        model_size = "medium"
        backend = "cpu"
    elif ram >= 12:
        model_size = "small"
        backend = "cpu"
    else:
        model_size = "tiny"
        backend = "cpu"

    return {
        "model_size": model_size,
        "backend": backend,
        "max_threads": max(1, cpu - 1),
        "use_disk_cache": True,
        "profile": "balanced",
    }


# ========= BUILDER ORGAN =========

def model_path_from_blueprint(blueprint: dict) -> Path:
    size = blueprint["model_size"]
    return MODELS_DIR / f"llm_{size}.bin"


def download_model_if_needed(path: Path, blueprint: dict):
    if path.exists():
        print(f"[*] Model already present: {path}")
        return
    print(f"[*] Preparing model stub for size={blueprint['model_size']}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")
    print("[*] Placeholder model file created.\n")


def ensure_built(cfg: dict):
    blueprint = cfg["blueprint"]
    model_path = model_path_from_blueprint(blueprint)
    download_model_if_needed(model_path, blueprint)
    cfg["model_path"] = str(model_path)
    print(f"[*] Model path set to {model_path}")


# ========= LOCAL LLM VIA OLLAMA (RESILIENT) =========

def call_local_llm(prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
    if system:
        full_prompt = f"System:\n{system}\n\nUser:\n{prompt}"
    else:
        full_prompt = prompt

    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "options": {"num_predict": max_tokens}
    }

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        return str(data)
    except Exception as e:
        return f"[LOCAL LLM ERROR: {e}]"


def call_local_llm_critique(answer: str) -> str:
    prompt = f"""
You are the self-critique subsystem of my neural core.

Given the following answer, do a critical review:
- List weaknesses, missing edge cases, and potential failure modes.
- Point out any unclear or hand-wavy reasoning.
- Suggest what should be improved or expanded.

Answer:
{answer}

Return just the critique.
""".strip()
    return call_local_llm(prompt, system="Self-critique engine", max_tokens=512)


def call_local_llm_refine(original_answer: str, critique: str, extra_mode: str = "") -> str:
    mode_tail = f"\nExtra mode: {extra_mode}" if extra_mode else ""
    prompt = f"""
You are the refinement subsystem of my neural core.

You are given an original answer and a critique of that answer.
Task:
- Improve the answer using the critique.
- Preserve what is good.
- Fix weaknesses, fill in missing details, and tighten logic.
- Make it clearer and more robust.
{mode_tail}

Original answer:
{original_answer}

Critique:
{critique}

Return the refined answer only.
""".strip()
    return call_local_llm(prompt, system="Refinement engine", max_tokens=700)


# ========= EXTERNAL LLM CONNECTORS =========

class LLMConnector:
    def __init__(self, name: str, base_url: str, api_key: str | None = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key

    def call(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text") or data.get("response") or str(data)


CONNECTORS: list[LLMConnector] = []


def load_llm_connectors_from_config(cfg: dict):
    llm_cfgs = cfg.get("remote_llms", [])
    for llm in llm_cfgs:
        try:
            CONNECTORS.append(
                LLMConnector(
                    name=llm["name"],
                    base_url=llm["base_url"],
                    api_key=llm.get("api_key")
                )
            )
        except KeyError:
            continue


# ========= ORCHESTRATOR =========

def fanout_to_llms(prompt: str, system: str | None = None, task: str = "chat") -> dict:
    results: dict[str, str] = {}
    results["local"] = call_local_llm(prompt, system=system)
    for conn in CONNECTORS:
        try:
            results[conn.name] = conn.call(prompt, system=system)
        except Exception as e:
            results[conn.name] = f"[error calling {conn.name}: {e}]"
    return results


def fuse_answers(answers: dict[str, str]) -> str:
    parts = []
    for name, text in answers.items():
        parts.append(f"[{name.upper()}]\n{text}\n")
    return "\n".join(parts)


# ========= DISTILLATION PIPELINE =========

def add_raw_knowledge(record: dict):
    with open(RAW_KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_knowledge() -> list[dict]:
    if not RAW_KNOWLEDGE_FILE.exists():
        return []
    records = []
    with open(RAW_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def distill_topic(topic: str, records: list[dict]) -> Path:
    combined = "\n\n".join(
        f"[{r['source_type']}::{r['source_name']}]\n{r['content']}"
        for r in records
    )

    prompt = f"""
You are distilling knowledge for my personal system.

Topic: {topic}

You will be given multiple noisy, overlapping, and possibly conflicting notes
from different sources (external LLMs, local files, prior sessions).

Task:
- Extract core principles, patterns, and best practices.
- Resolve obvious contradictions or mark uncertainties.
- Remove redundancy.
- Produce a clear, reusable distilled note I can use as a reference.

Raw material:
{combined}

Return only the distilled note.
""".strip()

    distilled_text = call_local_llm(prompt, max_tokens=1024)
    filename = DISTILLED_DIR / f"{topic.replace(' ', '_')}_{int(time.time())}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(distilled_text)
    return filename


# ========= EVOLVER LOGGING =========

def log_evolution_event(title: str, body: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(EVOLUTION_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {title}\n{textwrap.indent(body, '  ')}\n\n")


# ========= PREDICTION ORGAN (UPGRADED: MODES) =========

class PredictionEngine:
    def __init__(self, max_history: int = 100):
        self.history = deque(maxlen=max_history)
        self.counter = Counter()
        self.mode_history = deque(maxlen=50)
        self.mode_counter = Counter()

    def _tokenize(self, text: str):
        delimiters = ".,;:!?()[]{}<>\"'\\/-_"
        tmp = text.lower()
        for d in delimiters:
            tmp = tmp.replace(d, " ")
        tokens = [t for t in tmp.split() if len(t) >= 4]
        return tokens

    def observe(self, user_text: str):
        self.history.append(user_text)
        tokens = self._tokenize(user_text)
        self.counter.update(tokens)
        self._log_prediction_state()

    def observe_mode(self, mode: str):
        mode = mode.upper()
        self.mode_history.append(mode)
        self.mode_counter.update([mode])

    def predict_top_topics(self, k: int = 5) -> list[str]:
        if not self.counter:
            return []
        return [w for w, _ in self.counter.most_common(k)]

    def predict_mode_trend(self) -> str:
        if not self.mode_counter:
            return "No mode history yet."
        top_modes = self.mode_counter.most_common(3)
        current = self.mode_history[-1] if self.mode_history else "UNKNOWN"
        parts = [f"{m} ({c})" for m, c in top_modes]
        return f"Current mode: {current} | Trend: " + ", ".join(parts)

    def _log_prediction_state(self):
        topics = self.predict_top_topics(5)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] topics={topics}\n")


PREDICTOR = PredictionEngine()


# ========= WEB PROBE ORGAN =========

class WebProbe:
    def __init__(self):
        self.last_status = "unknown"
        self.last_url = None
        self.last_fetch_ok = None
        self.last_error = None

    def connectivity_status(self) -> str:
        ok = check_network()
        self.last_status = "online" if ok else "offline"
        return self.last_status

    def fetch_text(self, url: str, timeout: float = 10.0) -> str | None:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            self.last_url = url
            self.last_fetch_ok = True
            self.last_error = None
            return resp.text
        except Exception as e:
            self.last_url = url
            self.last_fetch_ok = False
            self.last_error = str(e)
            return None


WEB_PROBE = WebProbe()


# ========= DIAGNOSTICS + SAFE MODE ORGAN =========

def log_diagnostics_event(body: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(DIAGNOSTICS_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {body}\n")


def run_diagnostics_cycle() -> str:
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    issues = []

    if ram.percent > 90:
        issues.append(f"High RAM usage: {ram.percent:.1f}%")
    if cpu > 95:
        issues.append(f"High CPU usage: {cpu:.1f}%")
    if disk.free / (1024**3) < 2:
        issues.append(f"Low disk free: {disk.free / (1024**3):.1f} GB")

    net = "online" if check_network() else "offline"

    enter_safe = bool(issues)

    if enter_safe and not STATE["safe_mode"]:
        STATE["safe_mode"] = True
        log_diagnostics_event("Entering SAFE MODE due to: " + "; ".join(issues))
    elif not issues and STATE["safe_mode"]:
        STATE["safe_mode"] = False
        log_diagnostics_event("Exiting SAFE MODE; resources back to normal.")

    mode_label = "SAFE MODE" if STATE["safe_mode"] else "NORMAL"

    report_lines = [
        f"Mode: {mode_label}",
        f"CPU: {cpu:.1f}%",
        f"RAM: {ram.percent:.1f}%",
        f"Disk Free: {disk.free / (1024**3):.1f} / {disk.total / (1024**3):.1f} GB",
        f"Network: {net}",
    ]
    if issues:
        report_lines.append("Issues: " + "; ".join(issues))
    else:
        report_lines.append("Issues: none detected.")

    report = "\n".join(report_lines)
    log_diagnostics_event(report)
    STATE["last_health_report"] = report
    return report


# ========= CONTEXT MEMORY ORGAN =========

class ContextMemoryOrgan:
    def __init__(self, max_recent_actions: int = 20):
        self.active_topics = Counter()
        self.entities = Counter()
        self.goals = Counter()
        self.recent_actions = deque(maxlen=max_recent_actions)
        self.summary = "No context yet."
        self._load_from_disk()

    def _tokenize(self, text: str):
        delimiters = ".,;:!?()[]{}<>\"'\\/-_"
        tmp = text.lower()
        for d in delimiters:
            tmp = tmp.replace(d, " ")
        tokens = [t for t in tmp.split() if len(t) >= 4]
        return tokens

    def _detect_topics(self, text: str):
        tokens = self._tokenize(text)
        return tokens

    def _detect_entities(self, text: str):
        entities = set()
        for token in text.split():
            if token.istitle() and len(token) >= 3:
                entities.add(token.strip(",.?!:;()[]{}"))
        keywords = ["organ", "engine", "guardian", "clipboard", "distillation", "prediction", "diagnostics", "context"]
        for kw in keywords:
            if kw in text.lower():
                entities.add(kw)
        return list(entities)

    def _detect_goals(self, text: str):
        hints = []
        lower = text.lower()
        prefixes = ["want to", "trying to", "goal is", "need to", "aim is", "plan is"]
        for p in prefixes:
            if p in lower:
                hints.append(p)
        return hints

    def record_action(self, description: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.recent_actions.append(f"{ts} - {description}")

    def update(self, user_text: str, draft: str, critique: str, refined: str):
        combined = f"USER:\n{user_text}\n\nREFINED:\n{refined}"
        topics = self._detect_topics(combined)
        self.active_topics.update(topics)

        entities = self._detect_entities(combined)
        self.entities.update(entities)

        goals = self._detect_goals(user_text)
        self.goals.update(goals)

        self.record_action("Processed interaction with context memory update.")
        self._update_summary_with_llm()
        self._save_to_disk()

    def _update_summary_with_llm(self):
        top_topics = [w for w, _ in self.active_topics.most_common(6)]
        top_entities = [w for w, _ in self.entities.most_common(8)]
        top_goals = [w for w, _ in self.goals.most_common(5)]

        actions_snippet = "\n".join(list(self.recent_actions)[-5:])

        prompt = f"""
You are the context memory organ of a personal neural core system.

You are given:
- Active topics (keywords)
- Entities (names, tools, concepts)
- Goal hints
- Recent actions log

Summarize in 1–3 short paragraphs:
- What this system seems to be working on recently
- What the main themes and goals are
- Any notable shifts or patterns in activity

Active topics:
{top_topics}

Entities:
{top_entities}

Goal hints:
{top_goals}

Recent actions:
{actions_snippet}

Return just the summary.
""".strip()

        summary = call_local_llm(prompt, system="Context memory summarizer", max_tokens=300)
        self.summary = summary if summary else "No summary available."

    def render(self) -> str:
        top_topics = self.active_topics.most_common(6)
        top_entities = self.entities.most_common(8)
        top_goals = self.goals.most_common(5)

        def format_counter(cnt_list):
            return ", ".join(f"{k} ({v})" for k, v in cnt_list) if cnt_list else "None"

        text = []
        text.append("=== Context Memory ===")
        text.append("Active Topics:")
        text.append("  " + (format_counter(top_topics) if top_topics else "  None"))
        text.append("")
        text.append("Entities:")
        text.append("  " + (format_counter(top_entities) if top_entities else "  None"))
        text.append("")
        text.append("Goal Hints:")
        text.append("  " + (format_counter(top_goals) if top_goals else "  None"))
        text.append("")
        text.append("Recent Actions:")
        if self.recent_actions:
            for ra in self.recent_actions:
                text.append("  - " + ra)
        else:
            text.append("  None")
        text.append("")
        text.append("Summary:")
        text.append("  " + (self.summary.strip() if self.summary else "No summary yet."))

        return "\n".join(text)

    def _save_to_disk(self):
        data = {
            "active_topics": dict(self.active_topics),
            "entities": dict(self.entities),
            "goals": dict(self.goals),
            "recent_actions": list(self.recent_actions),
            "summary": self.summary,
        }
        try:
            with open(CONTEXT_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_from_disk(self):
        if not CONTEXT_MEMORY_FILE.exists():
            return
        try:
            with open(CONTEXT_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_topics = Counter(data.get("active_topics", {}))
            self.entities = Counter(data.get("entities", {}))
            self.goals = Counter(data.get("goals", {}))
            self.recent_actions = deque(data.get("recent_actions", []), maxlen=self.recent_actions.maxlen)
            self.summary = data.get("summary", "No context yet.")
        except Exception:
            pass


CONTEXT_MEMORY = ContextMemoryOrgan()


# ========= INTENT ROUTER ORGAN =========

class IntentRouterOrgan:
    def __init__(self):
        self.last_intent = "OTHER"
        self.last_confidence = 0.0
        self.last_explanation = ""

    def classify(self, text: str) -> dict:
        prompt = f"""
You are an intent classifier for a personal neural core.

Given the input text, classify the user's primary intent into EXACTLY one of:
- CODE
- DOCS
- CHAT
- DEBUG
- RESEARCH
- OTHER

Also provide:
- confidence score between 0 and 1
- a one-sentence explanation

Return JSON ONLY, with keys:
- "intent": string
- "confidence": float
- "explanation": string

Input text:
{text}
""".strip()

        raw = call_local_llm(prompt, system="Intent router", max_tokens=256)
        intent = "OTHER"
        confidence = 0.0
        explanation = ""

        try:
            data = json.loads(raw)
            intent = str(data.get("intent", "OTHER")).upper()
            confidence = float(data.get("confidence", 0.0))
            explanation = str(data.get("explanation", ""))
        except Exception:
            lower = text.lower()
            if any(k in lower for k in ["def ", "class ", "import ", "function", "code", "python", "error:"]):
                intent = "CODE"
            elif any(k in lower for k in ["doc", "documentation", "explain", "describe"]):
                intent = "DOCS"
            elif any(k in lower for k in ["bug", "stack trace", "exception"]):
                intent = "DEBUG"
            elif any(k in lower for k in ["research", "paper", "study", "article"]):
                intent = "RESEARCH"
            else:
                intent = "CHAT"
            confidence = 0.4
            explanation = "Fallback heuristic classification."

        self.last_intent = intent
        self.last_confidence = confidence
        self.last_explanation = explanation
        return {
            "intent": intent,
            "confidence": confidence,
            "explanation": explanation,
        }

    def choose_system_prompt(self, base_prompt: str, intent: str) -> str:
        intent = intent.upper()
        if intent == "CODE":
            return base_prompt + "\n\nYou are in CODE mode: focus on code, be precise, include examples."
        if intent == "DOCS":
            return base_prompt + "\n\nYou are in DOCS mode: explain clearly and concisely, focus on structure."
        if intent == "DEBUG":
            return base_prompt + "\n\nYou are in DEBUG mode: look for bugs, edge cases, and failure modes."
        if intent == "RESEARCH":
            return base_prompt + "\n\nYou are in RESEARCH mode: compare ideas, highlight uncertainties and tradeoffs."
        if intent == "CHAT":
            return base_prompt + "\n\nYou are in CHAT mode: keep it conversational but still precise."
        return base_prompt


INTENT_ROUTER = IntentRouterOrgan()


# ========= ANOMALY DETECTOR ORGAN =========

class AnomalyDetectorOrgan:
    def __init__(self, max_history: int = 10):
        self.last_refined = None
        self.history = deque(maxlen=max_history)
        self.anomalies = deque(maxlen=20)

    def _similarity(self, a: str, b: str) -> float:
        a_set = set(a.split())
        b_set = set(b.split())
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union

    def analyze(self, user: str, draft: str, refined: str) -> str | None:
        anomaly_msgs = []

        if "[LOCAL LLM ERROR" in refined or "error" in refined.lower():
            anomaly_msgs.append("Refined output contains an error marker.")
        if len(refined.strip()) < 40:
            anomaly_msgs.append("Refined output is suspiciously short.")
        if self.last_refined:
            sim = self._similarity(self.last_refined, refined)
            if sim > 0.9:
                anomaly_msgs.append(f"Refined output is highly similar to previous (similarity={sim:.2f}).")

        self.last_refined = refined
        self.history.append({
            "user": user,
            "draft": draft[:400],
            "refined": refined[:400],
        })

        if anomaly_msgs:
            msg = "; ".join(anomaly_msgs)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            entry = f"{ts} - {msg}"
            self.anomalies.append(entry)
            return entry
        return None

    def render(self) -> str:
        if not self.anomalies:
            return "No anomalies detected."
        lines = ["=== Anomalies ==="]
        for a in self.anomalies:
            lines.append("  - " + a)
        return "\n".join(lines)


ANOMALY_DETECTOR = AnomalyDetectorOrgan()


# ========= META-REASONING ORGAN (second-pass refinement) =========

class MetaReasoningOrgan:
    """
    Decides whether to perform an extra refinement pass.
    Uses anomaly presence + basic heuristics.
    """
    def should_second_pass(self, refined: str, anomaly: str | None) -> bool:
        if anomaly:
            return True
        if len(refined.strip()) < 80:
            return True
        if "[LOCAL LLM ERROR" in refined:
            return True
        return False


META_REASONER = MetaReasoningOrgan()


# ========= GUI (OBSERVER) =========

class ThoughtStreamModel(QtCore.QObject):
    new_thought = QtCore.Signal(str)
    prediction_update = QtCore.Signal(str)
    health_update = QtCore.Signal(str)
    safe_mode_update = QtCore.Signal(bool)
    critique_cycle_update = QtCore.Signal(str)
    context_update = QtCore.Signal(str)
    intent_update = QtCore.Signal(str)
    anomaly_update = QtCore.Signal(str)
    prediction_mode_update = QtCore.Signal(str)

    def append(self, text: str):
        self.new_thought.emit(text)

    def update_prediction(self, text: str):
        self.prediction_update.emit(text)

    def update_prediction_mode(self, text: str):
        self.prediction_mode_update.emit(text)

    def update_health(self, text: str):
        self.health_update.emit(text)

    def update_safe_mode(self, flag: bool):
        self.safe_mode_update.emit(flag)

    def update_critique_cycle(self, text: str):
        self.critique_cycle_update.emit(text)

    def update_context(self, text: str):
        self.context_update.emit(text)

    def update_intent(self, text: str):
        self.intent_update.emit(text)

    def update_anomaly(self, text: str):
        self.anomaly_update.emit(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, thought_model: ThoughtStreamModel):
        super().__init__()
        self.thought_model = thought_model
        self.setWindowTitle("NEURAL CORE OBSERVATORY")
        self.resize(1750, 980)
        self._init_ui()
        self._apply_dark_sci_fi_theme()

        self.thought_model.new_thought.connect(self._on_new_thought)
        self.thought_model.prediction_update.connect(self._on_prediction_update)
        self.thought_model.prediction_mode_update.connect(self._on_prediction_mode_update)
        self.thought_model.health_update.connect(self._on_health_update)
        self.thought_model.safe_mode_update.connect(self._on_safe_mode_update)
        self.thought_model.critique_cycle_update.connect(self._on_critique_cycle_update)
        self.thought_model.context_update.connect(self._on_context_update)
        self.thought_model.intent_update.connect(self._on_intent_update)
        self.thought_model.anomaly_update.connect(self._on_anomaly_update)

        self._start_resource_timer()
        self._start_evolution_log_watcher()
        self._start_net_status_timer()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        # Left panel
        left_split = QtWidgets.QVBoxLayout()
        self.organs_list = QtWidgets.QListWidget()
        self.organs_list.addItems([
            "Probe Organ",
            "Blueprint Organ",
            "Builder Organ",
            "Brain Organ",
            "Distillation Organ",
            "Prediction Organ",
            "Diagnostics Organ",
            "Self-Critique Organ",
            "Context Memory Organ",
            "Intent Router Organ",
            "Anomaly Detector Organ",
            "Keyboard Monitor Organ",
            "Web Probe Organ",
            "Clipboard Listener Organ",
            "Observer Organ",
        ])
        self.organs_list.setFixedHeight(340)

        self.vitals_text = QtWidgets.QTextEdit()
        self.vitals_text.setReadOnly(True)

        self.prediction_text = QtWidgets.QTextEdit()
        self.prediction_text.setReadOnly(True)

        self.prediction_mode_text = QtWidgets.QTextEdit()
        self.prediction_mode_text.setReadOnly(True)

        self.health_text = QtWidgets.QTextEdit()
        self.health_text.setReadOnly(True)

        self.safe_mode_label = QtWidgets.QLabel("MODE: UNKNOWN")
        self.safe_mode_label.setAlignment(QtCore.Qt.AlignCenter)

        self.net_status_label = QtWidgets.QLabel("NET: unknown")
        self.net_status_label.setAlignment(QtCore.Qt.AlignCenter)

        self.web_probe_button = QtWidgets.QPushButton("Web Probe")
        self.web_probe_button.clicked.connect(self._on_web_probe_clicked)

        # Clipboard toggle (still GUI controlled)
        self.clipboard_status_label = QtWidgets.QLabel("Clipboard Listener: OFF")
        self.clipboard_status_label.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_clip_on = QtWidgets.QPushButton("◉ ON")
        self.btn_clip_off = QtWidgets.QPushButton("○ OFF")
        self.btn_clip_on.setCheckable(True)
        self.btn_clip_off.setCheckable(True)
        self.btn_clip_off.setChecked(True)

        self.btn_clip_on.clicked.connect(self._on_clip_on_clicked)
        self.btn_clip_off.clicked.connect(self._on_clip_off_clicked)

        clip_toggle_layout = QtWidgets.QHBoxLayout()
        clip_toggle_layout.addWidget(self.btn_clip_on)
        clip_toggle_layout.addWidget(self.btn_clip_off)

        left_split.addWidget(QtWidgets.QLabel("Organs"))
        left_split.addWidget(self.organs_list)
        left_split.addWidget(QtWidgets.QLabel("Resource Vitals"))
        left_split.addWidget(self.vitals_text)
        left_split.addWidget(QtWidgets.QLabel("Prediction / Likely Topics"))
        left_split.addWidget(self.prediction_text)
        left_split.addWidget(QtWidgets.QLabel("Mode Trend Forecast"))
        left_split.addWidget(self.prediction_mode_text)
        left_split.addWidget(self.web_probe_button)
        left_split.addWidget(QtWidgets.QLabel("Health / Diagnostics"))
        left_split.addWidget(self.health_text)
        left_split.addWidget(self.safe_mode_label)
        left_split.addWidget(self.net_status_label)
        left_split.addWidget(QtWidgets.QLabel("Clipboard Listener Control"))
        left_split.addWidget(self.clipboard_status_label)
        left_split.addLayout(clip_toggle_layout)

        # Center panel: mind + context + intent/anomaly
        center_vbox = QtWidgets.QVBoxLayout()
        center_vbox.addWidget(QtWidgets.QLabel("Mind Stream"))
        self.thought_view = QtWidgets.QTextEdit()
        self.thought_view.setReadOnly(True)
        center_vbox.addWidget(self.thought_view, 3)

        center_vbox.addWidget(QtWidgets.QLabel("Context Memory"))
        self.context_view = QtWidgets.QTextEdit()
        self.context_view.setReadOnly(True)
        center_vbox.addWidget(self.context_view, 2)

        intent_anomaly_split = QtWidgets.QHBoxLayout()
        intent_box = QtWidgets.QVBoxLayout()
        intent_box.addWidget(QtWidgets.QLabel("Intent Router"))
        self.intent_view = QtWidgets.QTextEdit()
        self.intent_view.setReadOnly(True)
        intent_box.addWidget(self.intent_view)

        anomaly_box = QtWidgets.QVBoxLayout()
        anomaly_box.addWidget(QtWidgets.QLabel("Anomaly Detector"))
        self.anomaly_view = QtWidgets.QTextEdit()
        self.anomaly_view.setReadOnly(True)
        anomaly_box.addWidget(self.anomaly_view)

        intent_anomaly_split.addLayout(intent_box)
        intent_anomaly_split.addLayout(anomaly_box)

        center_vbox.addLayout(intent_anomaly_split, 2)

        # Right panel: evolution + critique
        right_vbox = QtWidgets.QVBoxLayout()
        right_vbox.addWidget(QtWidgets.QLabel("Evolution Log"))
        self.evolution_view = QtWidgets.QTextEdit()
        self.evolution_view.setReadOnly(True)
        right_vbox.addWidget(self.evolution_view)

        right_vbox.addWidget(QtWidgets.QLabel("Critique Cycles"))
        self.critique_view = QtWidgets.QTextEdit()
        self.critique_view.setReadOnly(True)
        right_vbox.addWidget(self.critique_view)

        layout.addLayout(left_split, 3)
        layout.addLayout(center_vbox, 6)
        layout.addLayout(right_vbox, 3)

        self.setCentralWidget(central)
        self._update_clipboard_toggle_visual()

    def _apply_dark_sci_fi_theme(self):
        palette = self.palette()
        bg = QtGui.QColor(10, 10, 16)
        fg = QtGui.QColor(190, 255, 220)
        accent = QtGui.QColor(0, 255, 160)

        palette.setColor(QtGui.QPalette.Window, bg)
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15, 15, 25))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(20, 20, 35))
        palette.setColor(QtGui.QPalette.Text, fg)
        palette.setColor(QtGui.QPalette.WindowText, fg)
        palette.setColor(QtGui.QPalette.Highlight, accent)
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))

        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #05050A;
            }
            QTextEdit, QListWidget {
                background-color: #10101A;
                color: #CFFFE0;
                border: 1px solid #00FF99;
            }
            QLabel {
                color: #6CFFC6;
                font-weight: bold;
            }
            QPushButton {
                background-color: #10101A;
                color: #CFFFE0;
                border: 1px solid #00FF99;
                padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: #00FF99;
                color: #000000;
            }
        """)

    def _update_clipboard_toggle_visual(self):
        if STATE["clipboard_listener_enabled"]:
            self.clipboard_status_label.setText("Clipboard Listener: ON")
            self.clipboard_status_label.setStyleSheet("color: #6CFFC6; font-weight: bold;")
            self.btn_clip_on.setChecked(True)
            self.btn_clip_off.setChecked(False)
            self.btn_clip_on.setText("◉ ON")
            self.btn_clip_off.setText("○ OFF")
        else:
            self.clipboard_status_label.setText("Clipboard Listener: OFF")
            self.clipboard_status_label.setStyleSheet("color: #FFAA33; font-weight: bold;")
            self.btn_clip_on.setChecked(False)
            self.btn_clip_off.setChecked(True)
            self.btn_clip_on.setText("○ ON")
            self.btn_clip_off.setText("◉ OFF")

    def _on_clip_on_clicked(self):
        STATE["clipboard_listener_enabled"] = True
        self._update_clipboard_toggle_visual()
        self.thought_model.append("[CLIPBOARD] Listener enabled (ON).")

    def _on_clip_off_clicked(self):
        STATE["clipboard_listener_enabled"] = False
        self._update_clipboard_toggle_visual()
        self.thought_model.append("[CLIPBOARD] Listener disabled (OFF).")

    def _on_new_thought(self, text: str):
        self.thought_view.append(text)

    def _on_prediction_update(self, text: str):
        self.prediction_text.setPlainText(text)

    def _on_prediction_mode_update(self, text: str):
        self.prediction_mode_text.setPlainText(text)

    def _on_health_update(self, text: str):
        self.health_text.setPlainText(text)

    def _on_safe_mode_update(self, flag: bool):
        if flag:
            self.safe_mode_label.setText("MODE: SAFE MODE")
            self.safe_mode_label.setStyleSheet("color: #FF5050; font-weight: bold;")
        else:
            self.safe_mode_label.setText("MODE: NORMAL")
            self.safe_mode_label.setStyleSheet("color: #6CFFC6; font-weight: bold;")

    def _on_critique_cycle_update(self, text: str):
        self.critique_view.setPlainText(text)

    def _on_context_update(self, text: str):
        self.context_view.setPlainText(text)

    def _on_intent_update(self, text: str):
        self.intent_view.setPlainText(text)

    def _on_anomaly_update(self, text: str):
        self.anomaly_view.setPlainText(text)

    def _start_resource_timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._update_vitals)
        timer.start(1000)
        self._vitals_timer = timer

    def _update_vitals(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        used_gb = ram.used / (1024 ** 3)
        total_gb = ram.total / (1024 ** 3)
        txt = (
            f"CPU: {cpu:.1f}%\n"
            f"RAM: {ram.percent:.1f}% ({used_gb:.1f} / {total_gb:.1f} GB)\n"
        )
        self.vitals_text.setPlainText(txt)

    def _start_evolution_log_watcher(self):
        def watch():
            last_size = 0
            while True:
                if EVOLUTION_LOG.exists():
                    size = EVOLUTION_LOG.stat().st_size
                    if size != last_size:
                        with open(EVOLUTION_LOG, "r", encoding="utf-8") as f:
                            content = f.read()
                        last_size = size
                        self._set_evolution_text(content)
                time.sleep(2)

        threading.Thread(target=watch, daemon=True).start()

    def _start_net_status_timer(self):
        def tick():
            status = WEB_PROBE.connectivity_status()
            if status == "online":
                self.net_status_label.setText("NET: ONLINE")
                self.net_status_label.setStyleSheet("color: #6CFFC6; font-weight: bold;")
            else:
                self.net_status_label.setText("NET: OFFLINE")
                self.net_status_label.setStyleSheet("color: #FFAA33; font-weight: bold;")
        timer = QtCore.QTimer(self)
        timer.timeout.connect(tick)
        timer.start(5000)
        self._net_timer = timer

    @QtCore.Slot(str)
    def _set_evolution_text(self, text: str):
        def update():
            self.evolution_view.setPlainText(text)
        QtCore.QTimer.singleShot(0, update)

    def _on_web_probe_clicked(self):
        url, ok = QtWidgets.QInputDialog.getText(
            self,
            "Web Probe",
            "Enter URL to probe and distill:",
            QtWidgets.QLineEdit.Normal,
            ""
        )
        if not ok or not url.strip():
            return
        url = url.strip()
        self.thought_model.append(f"[WEB PROBE] Fetching: {url}")

        def worker():
            try:
                html = WEB_PROBE.fetch_text(url)
                if html is None:
                    self.thought_model.append(f"[WEB PROBE ERROR] {WEB_PROBE.last_error}")
                    return

                text = extract_text_from_html(html)
                if not text:
                    self.thought_model.append(f"[WEB PROBE] No text extracted from {url}")
                    return

                parsed = urlparse(url)
                topic_hint = parsed.netloc or "web_content"
                topic = f"web::{topic_hint}"

                add_raw_knowledge({
                    "source_type": "web",
                    "source_name": url,
                    "timestamp": time.time(),
                    "topic_hint": topic,
                    "content": text[:15000],
                })

                distilled_path = distill_topic(topic, [{
                    "source_type": "web",
                    "source_name": url,
                    "content": text[:8000],
                }])

                self.thought_model.append(
                    f"[WEB PROBE] Distilled {url} into {distilled_path.name}"
                )

                PREDICTOR.observe(text[:500])
                top_topics = PREDICTOR.predict_top_topics()
                pred_label = "Likely topics: " + ", ".join(top_topics) if top_topics else "Not enough data yet."
                self.thought_model.update_prediction(pred_label)
                mode_trend = PREDICTOR.predict_mode_trend()
                self.thought_model.update_prediction_mode(mode_trend)

            except Exception as e:
                self.thought_model.append(f"[WEB PROBE ERROR] {e}")

        threading.Thread(target=worker, daemon=True).start()


# ========= BRAIN LOOP (INTENT + ANOMALY + META-REASONING + CONTEXT) =========

def process_one_input(user: str, thought_model: ThoughtStreamModel, cfg: dict):
    if not user:
        return

    PREDICTOR.observe(user)
    top_topics = PREDICTOR.predict_top_topics()
    pred_label = "Likely topics: " + ", ".join(top_topics) if top_topics else "Not enough data yet."
    thought_model.update_prediction(pred_label)

    thought_model.append(f"[INPUT] {user}")

    intent_info = INTENT_ROUTER.classify(user)
    intent = intent_info["intent"]
    conf = intent_info["confidence"]
    expl = intent_info["explanation"]
    PREDICTOR.observe_mode(intent)
    mode_trend = PREDICTOR.predict_mode_trend()

    intent_text = (
        f"Intent: {intent} (confidence={conf:.2f})\n"
        f"Explanation: {expl}"
    )
    thought_model.update_intent(intent_text)
    thought_model.update_prediction_mode(mode_trend)

    base_system = "You are my neural core."
    routed_system = INTENT_ROUTER.choose_system_prompt(base_system, intent)

    if STATE["safe_mode"]:
        response = call_local_llm(user, system="Safe-mode neural core (minimal).")
        thought_model.append(f"[OUTPUT - SAFE]\n{response}")
        log_evolution_event("interaction_safe_mode", f"User: {user}\nResponse: {response[:300]}...")
        CONTEXT_MEMORY.record_action("Processed input in SAFE MODE.")
        thought_model.update_context(CONTEXT_MEMORY.render())
        return

    answers = fanout_to_llms(user, system=routed_system, task=intent)
    draft = fuse_answers(answers)
    thought_model.append(f"[DRAFT]\n{draft}")

    critique = call_local_llm_critique(draft)
    thought_model.append(f"[CRITIQUE]\n{critique}")

    refined = call_local_llm_refine(draft, critique)
    thought_model.append(f"[REFINED]\n{refined}")

    anomaly = ANOMALY_DETECTOR.analyze(user, draft, refined)

    if META_REASONER.should_second_pass(refined, anomaly):
        thought_model.append("[META] Second-pass refinement triggered due to anomaly or weak output.")
        critique2 = call_local_llm_critique(refined)
        refined2 = call_local_llm_refine(refined, critique2, extra_mode="second_pass")
        refined = refined2
        thought_model.append(f"[REFINED 2]\n{refined}")
        anomaly = ANOMALY_DETECTOR.analyze(user, draft, refined)

    critique_summary = (
        "=== Self-Critique Cycle ===\n"
        "DRAFT: first-pass answer from core.\n"
        "CRITIQUE: internal review of weaknesses.\n"
        "REFINED: improved answer using critique.\n"
        "META: second-pass refinement if anomalies/weaknesses detected.\n"
    )
    thought_model.update_critique_cycle(critique_summary)

    CONTEXT_MEMORY.update(user, draft, critique, refined)
    thought_model.update_context(CONTEXT_MEMORY.render())

    if anomaly:
        thought_model.update_anomaly(ANOMALY_DETECTOR.render())
        thought_model.append(f"[ANOMALY] {anomaly}")
    else:
        thought_model.update_anomaly(ANOMALY_DETECTOR.render())

    add_raw_knowledge({
        "source_type": "conversation",
        "source_name": "user_session",
        "timestamp": time.time(),
        "topic_hint": intent.lower(),
        "content": f"Q: {user}\nDRAFT: {draft}\nCRITIQUE: {critique}\nREFINED: {refined}",
    })

    log_evolution_event(
        "interaction",
        f"User: {user}\nIntent: {intent} (conf={conf:.2f})\nDraft: {draft[:200]}...\nRefined: {refined[:200]}..."
    )


def brain_loop(thought_model: ThoughtStreamModel, cfg: dict):
    thought_model.append("[SYSTEM] Brain online. Input sources: console, clipboard, keyboard monitor.")

    thought_model.update_context(CONTEXT_MEMORY.render())
    thought_model.update_intent("Intent: UNKNOWN\nExplanation: none yet.")
    thought_model.update_anomaly(ANOMALY_DETECTOR.render())
    thought_model.update_prediction_mode("No mode history yet.")

    def console_reader():
        while True:
            try:
                user = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                INPUT_QUEUE.put("__EXIT__")
                break
            INPUT_QUEUE.put(user)

    threading.Thread(target=console_reader, daemon=True).start()

    while True:
        user = INPUT_QUEUE.get()
        if user == "__EXIT__":
            thought_model.append("[SYSTEM] Brain shutting down (console exit).")
            break
        process_one_input(user, thought_model, cfg)


# ========= CLIPBOARD LISTENER =========

def clipboard_listener(thought_model: ThoughtStreamModel, interval_sec: float = 1.0):
    while True:
        try:
            if STATE["clipboard_listener_enabled"]:
                try:
                    text = pyperclip.paste()
                except Exception:
                    text = ""
                if text and text != STATE["last_clipboard_text"]:
                    STATE["last_clipboard_text"] = text
                    thought_model.append("[CLIPBOARD] New clipboard text captured into brain queue.")
                    INPUT_QUEUE.put(text)
            time.sleep(interval_sec)
        except Exception as e:
            thought_model.append(f"[CLIPBOARD ERROR] {e}")
            time.sleep(interval_sec)


# ========= KEYBOARD MONITOR ORGAN (HYBRID) =========

class KeyboardMonitorOrgan:
    """
    Hybrid:
    - Tracks typed characters into a buffer (global).
    - If user hits sentence boundary (. ? ! newline) and then pauses ~0.8s:
        -> send last sentence to brain.
    - If user types hotword '@@':
        -> send last sentence immediately, strip '@@'.
    """

    def __init__(self, thought_model: ThoughtStreamModel):
        self.thought_model = thought_model
        self.buffer = ""
        self.last_event_time = time.time()
        self.lock = threading.Lock()
        self.hotword = "@@"

    def _extract_last_sentence(self, text: str) -> str:
        for sep in [".", "?", "!", "\n"]:
            text = text.replace(sep, sep + "|")
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if not parts:
            return text.strip()
        return parts[-1].strip()

    def on_press(self, key):
        if not STATE["keyboard_monitor_enabled"]:
            return
        with self.lock:
            self.last_event_time = time.time()
            try:
                if hasattr(key, 'char') and key.char is not None:
                    self.buffer += key.char
                elif key == keyboard.Key.space:
                    self.buffer += " "
                elif key == keyboard.Key.enter:
                    self.buffer += "\n"
                else:
                    return
            except Exception:
                return

            if self.hotword in self.buffer:
                idx = self.buffer.rfind(self.hotword)
                clean = self.buffer[:idx]
                sentence = self._extract_last_sentence(clean)
                if sentence:
                    self.thought_model.append("[KEYBOARD] Hotword trigger: sending last sentence.")
                    INPUT_QUEUE.put(sentence)
                self.buffer = ""
                return

            if any(ch in self.buffer for ch in [".", "?", "!", "\n"]):
                # We'll let the timer thread handle the pause detection.
                pass

    def monitor_idle(self, idle_threshold: float = 0.8, poll: float = 0.3):
        while True:
            try:
                if not STATE["keyboard_monitor_enabled"]:
                    time.sleep(poll)
                    continue
                now = time.time()
                with self.lock:
                    if self.buffer and (now - self.last_event_time) > idle_threshold:
                        sentence = self._extract_last_sentence(self.buffer)
                        if sentence:
                            self.thought_model.append("[KEYBOARD] Idle pause after sentence; sending to brain.")
                            INPUT_QUEUE.put(sentence)
                        self.buffer = ""
                time.sleep(poll)
            except Exception as e:
                self.thought_model.append(f"[KEYBOARD ERROR] {e}")
                time.sleep(poll)


def start_keyboard_monitor(thought_model: ThoughtStreamModel):
    km = KeyboardMonitorOrgan(thought_model)

    listener = keyboard.Listener(on_press=km.on_press)
    listener.daemon = True
    listener.start()

    threading.Thread(target=km.monitor_idle, daemon=True).start()


# ========= CONFIG + MAIN =========

def load_or_init_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def first_time_setup(cfg: dict):
    print("[*] First run: mapping environment...")
    env = probe_environment()
    print("[*] Environment:", env)

    print("[*] Choosing blueprint...")
    blueprint = choose_blueprint(env)
    print("[*] Blueprint:", blueprint)

    cfg["environment"] = env
    cfg["blueprint"] = blueprint
    save_config(cfg)

    print("[*] Building system according to blueprint...")
    ensure_built(cfg)
    save_config(cfg)
    print("[*] Initial assembly complete.\n")


def main():
    cfg = load_or_init_config()
    if "blueprint" not in cfg:
        first_time_setup(cfg)
    else:
        print("[*] Existing blueprint found; skipping initial assembly.\n")

    load_llm_connectors_from_config(cfg)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    thought_model = ThoughtStreamModel()
    window = MainWindow(thought_model)
    window.show()

    thought_model.update_health(STATE["last_health_report"])
    thought_model.update_safe_mode(STATE["safe_mode"])

    t_brain = threading.Thread(target=brain_loop, args=(thought_model, cfg), daemon=True)
    t_brain.start()

    def diag_wrapper():
        while True:
            report = run_diagnostics_cycle()
            thought_model.update_health(report)
            thought_model.update_safe_mode(STATE["safe_mode"])
            time.sleep(10)

    t_diag = threading.Thread(target=diag_wrapper, daemon=True)
    t_diag.start()

    t_clip = threading.Thread(target=clipboard_listener, args=(thought_model, 1.0), daemon=True)
    t_clip.start()

    start_keyboard_monitor(thought_model)

    app.exec()


if __name__ == "__main__":
    main()

