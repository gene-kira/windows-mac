#!/usr/bin/env python
# neural_core.py
# One-file, dark sci-fi neural core with:
# - Cross-platform design (Windows / Linux / macOS)
# - Prediction organ (topic forecasting)
# - Self-critique answer pipeline (draft -> critique -> refined)
# - Diagnostics + safe mode organ (CPU/RAM/disk + network aware)
# - Web probe organ (internet connectivity + simple fetch)
# - GUI visuals for health, safe mode, net status, critique cycles

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
import socket  # for cross-platform network check

# ========= AUTO-INSTALL REQUIRED PACKAGES =========

REQUIRED_PACKAGES = [
    "psutil",
    "PySide6",
    "requests",
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

for p in [APP_DIR, MODELS_DIR, VECTOR_DB_DIR, DISTILLED_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ========= GLOBAL STATE (SAFE MODE, HEALTH) =========

STATE = {
    "safe_mode": False,
    "last_health_report": "Unknown",
}


# ========= PROBE ORGAN (ENVIRONMENT MAPPING) =========

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
    """
    Cross-platform network check.
    Tries to open a TCP connection to a public DNS server (no data sent).
    Works on Windows / Linux / macOS without shell commands.
    """
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


# ========= BLUEPRINT ORGAN (BODY PLAN) =========

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


# ========= BUILDER ORGAN (ASSEMBLY) =========

def model_path_from_blueprint(blueprint: dict) -> Path:
    size = blueprint["model_size"]
    return MODELS_DIR / f"llm_{size}.bin"  # placeholder


def download_model_if_needed(path: Path, blueprint: dict):
    if path.exists():
        print(f"[*] Model already present: {path}")
        return

    print(f"[*] Preparing model stub for size={blueprint['model_size']}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")  # placeholder stub file
    print("[*] Placeholder model file created.\n")


def ensure_built(cfg: dict):
    blueprint = cfg["blueprint"]
    model_path = model_path_from_blueprint(blueprint)
    download_model_if_needed(model_path, blueprint)
    cfg["model_path"] = str(model_path)
    print(f"[*] Model path set to {model_path}")


# ========= LLM STUBS (CORE + SELF-CRITIQUE) =========

def call_local_llm(prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
    """
    Placeholder for a local LLM used for chat / reasoning.
    Later, you can wire this into a real local model.
    """
    header = "[LOCAL LLM STUB]\n"
    if system:
        header += f"(system: {system})\n"
    return header + f"Response to: {prompt[:300]}..."


def call_local_llm_critique(answer: str) -> str:
    """
    Stub for self-critique: asks the model to list weaknesses / missing cases.
    """
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


def call_local_llm_refine(original_answer: str, critique: str) -> str:
    """
    Stub for refinement: improve the answer based on its critique.
    """
    prompt = f"""
You are the refinement subsystem of my neural core.

You are given an original answer and a critique of that answer.
Task:
- Improve the answer using the critique.
- Preserve what is good.
- Fix weaknesses, fill in missing details, and tighten logic.
- Make it clearer and more robust.

Original answer:
{original_answer}

Critique:
{critique}

Return the refined answer only.
""".strip()
    return call_local_llm(prompt, system="Refinement engine", max_tokens=512)


# ========= EXTERNAL LLM CONNECTORS (OPTIONAL, STUBBED) =========

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


CONNECTORS: list[LLMConnector] = []  # configure external models if desired


# ========= ORCHESTRATOR (LOCAL + MULTI-LLM) =========

def fanout_to_llms(prompt: str, system: str | None = None, use_external: bool = True) -> dict:
    results: dict[str, str] = {}
    local_answer = call_local_llm(prompt, system=system)
    results["local"] = local_answer

    if not use_external or not CONNECTORS:
        return results

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


# ========= DISTILLATION PIPELINE (SKELETON) =========

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


# ========= PREDICTION ORGAN =========

class PredictionEngine:
    """
    Simple v0 prediction engine:
    - Tracks recent inputs
    - Builds keyword frequency
    - Suggests likely next topics
    """
    def __init__(self, max_history: int = 100):
        self.history = deque(maxlen=max_history)
        self.counter = Counter()

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

    def predict_top_topics(self, k: int = 5) -> list[str]:
        if not self.counter:
            return []
        return [w for w, _ in self.counter.most_common(k)]

    def _log_prediction_state(self):
        topics = self.predict_top_topics(5)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] topics={topics}\n")


PREDICTOR = PredictionEngine()


# ========= WEB PROBE ORGAN (INTERNET-AWARE) =========

class WebProbe:
    """
    Minimal web probe organ.
    - Checks connectivity.
    - Optionally fetches simple text resources for future distillation.
    - Always runs controlled calls, never executes fetched code.
    """
    def __init__(self):
        self.last_status = "unknown"
        self.last_url = None
        self.last_fetch_ok = None
        self.last_error = None

    def connectivity_status(self) -> str:
        ok = check_network()
        self.last_status = "online" if ok else "offline"
        return self.last_status

    def fetch_text(self, url: str, timeout: float = 5.0) -> str | None:
        """
        Safe text fetcher. Returns the body as text or None on error.
        Does not execute anything, just fetches data.
        """
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
    """
    Run a lightweight diagnostics pass and update STATE['safe_mode'] if needed.
    Returns a human-readable health report string.
    """
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

    # Simple heuristic safe mode toggle
    enter_safe = False
    if issues:
        enter_safe = True

    # Update global STATE
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


# ========= DARK SCI-FI GUI (OBSERVER ORGAN) =========

class ThoughtStreamModel(QtCore.QObject):
    new_thought = QtCore.Signal(str)
    prediction_update = QtCore.Signal(str)
    health_update = QtCore.Signal(str)
    safe_mode_update = QtCore.Signal(bool)
    critique_cycle_update = QtCore.Signal(str)

    def append(self, text: str):
        self.new_thought.emit(text)

    def update_prediction(self, text: str):
        self.prediction_update.emit(text)

    def update_health(self, text: str):
        self.health_update.emit(text)

    def update_safe_mode(self, flag: bool):
        self.safe_mode_update.emit(flag)

    def update_critique_cycle(self, text: str):
        self.critique_cycle_update.emit(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, thought_model: ThoughtStreamModel):
        super().__init__()
        self.thought_model = thought_model
        self.setWindowTitle("NEURAL CORE OBSERVATORY")
        self.resize(1400, 850)
        self._init_ui()
        self._apply_dark_sci_fi_theme()

        self.thought_model.new_thought.connect(self._on_new_thought)
        self.thought_model.prediction_update.connect(self._on_prediction_update)
        self.thought_model.health_update.connect(self._on_health_update)
        self.thought_model.safe_mode_update.connect(self._on_safe_mode_update)
        self.thought_model.critique_cycle_update.connect(self._on_critique_cycle_update)

        self._start_resource_timer()
        self._start_evolution_log_watcher()
        self._start_net_status_timer()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        # Left: organs + vitals + prediction + health/safe mode/net
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
            "Web Probe Organ",
            "Observer Organ",
        ])
        self.organs_list.setFixedHeight(220)

        self.vitals_text = QtWidgets.QTextEdit()
        self.vitals_text.setReadOnly(True)

        self.prediction_text = QtWidgets.QTextEdit()
        self.prediction_text.setReadOnly(True)

        self.health_text = QtWidgets.QTextEdit()
        self.health_text.setReadOnly(True)

        self.safe_mode_label = QtWidgets.QLabel("MODE: UNKNOWN")
        self.safe_mode_label.setAlignment(QtCore.Qt.AlignCenter)

        self.net_status_label = QtWidgets.QLabel("NET: unknown")
        self.net_status_label.setAlignment(QtCore.Qt.AlignCenter)

        left_split.addWidget(QtWidgets.QLabel("Organs"))
        left_split.addWidget(self.organs_list)
        left_split.addWidget(QtWidgets.QLabel("Resource Vitals"))
        left_split.addWidget(self.vitals_text)
        left_split.addWidget(QtWidgets.QLabel("Prediction / Likely Topics"))
        left_split.addWidget(self.prediction_text)
        left_split.addWidget(QtWidgets.QLabel("Health / Diagnostics"))
        left_split.addWidget(self.health_text)
        left_split.addWidget(self.safe_mode_label)
        left_split.addWidget(self.net_status_label)

        # Center: thought stream
        center_vbox = QtWidgets.QVBoxLayout()
        center_vbox.addWidget(QtWidgets.QLabel("Mind Stream"))
        self.thought_view = QtWidgets.QTextEdit()
        self.thought_view.setReadOnly(True)
        center_vbox.addWidget(self.thought_view)

        # Right: evolution log + critique cycles
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
        layout.addLayout(center_vbox, 4)
        layout.addLayout(right_vbox, 3)

        self.setCentralWidget(central)

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
        """)

    def _on_new_thought(self, text: str):
        self.thought_view.append(text)

    def _on_prediction_update(self, text: str):
        self.prediction_text.setPlainText(text)

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
        timer.start(5000)  # every 5 seconds
        self._net_timer = timer

    @QtCore.Slot(str)
    def _set_evolution_text(self, text: str):
        def update():
            self.evolution_view.setPlainText(text)
        QtCore.QTimer.singleShot(0, update)


# ========= BRAIN LOOP (WITH SELF-CRITIQUE PIPELINE) =========

def brain_loop(thought_model: ThoughtStreamModel, cfg: dict):
    thought_model.append("[SYSTEM] Brain online. Type into this console. GUI is watching.")

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            thought_model.append("[SYSTEM] Brain shutting down.")
            break

        if user.lower() in {"exit", "quit"}:
            thought_model.append("[SYSTEM] Exit command received.")
            break

        if not user:
            continue

        # Prediction observe
        PREDICTOR.observe(user)
        top_topics = PREDICTOR.predict_top_topics()
        pred_label = "Likely topics: " + ", ".join(top_topics) if top_topics else "Not enough data yet."
        thought_model.update_prediction(pred_label)

        thought_model.append(f"[INPUT] {user}")

        # If safe mode, keep it lightweight
        if STATE["safe_mode"]:
            response = call_local_llm(user, system="Safe-mode neural core (minimal).")
            thought_model.append(f"[OUTPUT - SAFE]\n{response}")
            log_evolution_event("interaction_safe_mode", f"User: {user}\nResponse: {response[:300]}...")
            continue

        # Normal mode: full pipeline
        answers = fanout_to_llms(user, system="You are my neural core.")
        draft = fuse_answers(answers)
        thought_model.append(f"[DRAFT]\n{draft}")

        # Self-critique
        critique = call_local_llm_critique(draft)
        thought_model.append(f"[CRITIQUE]\n{critique}")

        # Refinement
        refined = call_local_llm_refine(draft, critique)
        thought_model.append(f"[REFINED]\n{refined}")

        # Update critique panel in GUI
        critique_summary = (
            "=== Self-Critique Cycle ===\n"
            "DRAFT: first-pass answer from core.\n"
            "CRITIQUE: internal review of weaknesses.\n"
            "REFINED: improved answer using critique.\n"
        )
        thought_model.update_critique_cycle(critique_summary)

        # Save raw knowledge
        add_raw_knowledge({
            "source_type": "conversation",
            "source_name": "user_session",
            "timestamp": time.time(),
            "topic_hint": "general",
            "content": f"Q: {user}\nDRAFT: {draft}\nCRITIQUE: {critique}\nREFINED: {refined}",
        })

        log_evolution_event(
            "interaction",
            f"User: {user}\nDraft: {draft[:200]}...\nCritique: {critique[:200]}...\nRefined: {refined[:200]}..."
        )


# ========= CONFIG LOAD/SAVE + MAIN ORCHESTRATION =========

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

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    thought_model = ThoughtStreamModel()
    window = MainWindow(thought_model)
    window.show()

    # Push initial health + mode to GUI
    thought_model.update_health(STATE["last_health_report"])
    thought_model.update_safe_mode(STATE["safe_mode"])

    # Brain loop
    t_brain = threading.Thread(target=brain_loop, args=(thought_model, cfg), daemon=True)
    t_brain.start()

    # Diagnostics loop
    def diag_wrapper():
        while True:
            report = run_diagnostics_cycle()
            thought_model.update_health(report)
            thought_model.update_safe_mode(STATE["safe_mode"])
            time.sleep(10)

    t_diag = threading.Thread(target=diag_wrapper, daemon=True)
    t_diag.start()

    app.exec()


if __name__ == "__main__":
    main()

