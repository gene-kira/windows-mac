#!/usr/bin/env python
# neural_core.py
# One-file self-assembling, dark sci-fi, LLM-on-steroids skeleton.
# Fixed so QApplication is created in the main thread.

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

for p in [APP_DIR, MODELS_DIR, VECTOR_DB_DIR, DISTILLED_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


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


def check_network():
    try:
        cmd = ["ping", "-c", "1", "8.8.8.8"] if platform.system() != "Windows" else ["ping", "-n", "1", "8.8.8.8"]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
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
        "profile": "balanced",  # later: safe / beast
    }


# ========= BUILDER ORGAN (ASSEMBLY) =========

def model_path_from_blueprint(blueprint: dict) -> Path:
    size = blueprint["model_size"]
    return MODELS_DIR / f"llm_{size}.bin"  # placeholder


def download_model_if_needed(path: Path, blueprint: dict):
    if path.exists():
        print(f"[*] Model already present: {path}")
        return

    print(f"[*] Downloading model for size={blueprint['model_size']} (placeholder)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: implement real download
    with open(path, "wb") as f:
        f.write(b"")  # placeholder
    print("[*] Placeholder model file created.\n")


def ensure_built(cfg: dict):
    blueprint = cfg["blueprint"]
    model_path = model_path_from_blueprint(blueprint)
    download_model_if_needed(model_path, blueprint)
    cfg["model_path"] = str(model_path)
    print(f"[*] Model path set to {model_path}")


# ========= LOCAL LLM STUB =========

def call_local_llm(prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
    header = "[LOCAL LLM STUB]\n"
    if system:
        header += f"(system: {system})\n"
    return header + f"Response to: {prompt[:300]}..."


# ========= EXTERNAL LLM CONNECTORS (STUB) =========

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


# Configure external LLMs here if you want
CONNECTORS: list[LLMConnector] = []
# Example (disabled):
# CONNECTORS.append(LLMConnector("remote_llm", "https://your-llm-endpoint", "API_KEY_HERE"))


# ========= ORCHESTRATOR (LOCAL + MULTI-LLM) =========

def fanout_to_llms(prompt: str, system: str | None = None, use_external: bool = True) -> dict:
    results: dict[str, str] = {}

    # Always include local core
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


# ========= EVOLVER (SELF-IMPROVEMENT LOGGING SKELETON) =========

def log_evolution_event(title: str, body: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(EVOLUTION_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {title}\n{textwrap.indent(body, '  ')}\n\n")


def propose_improvements(metrics: dict) -> list[dict]:
    # Placeholder: later, use metrics + LLM to generate proposals
    return []


def apply_proposal(proposal: dict):
    log_evolution_event("proposal_applied", str(proposal))


# ========= DARK SCI-FI GUI (OBSERVER ORGAN) =========

class ThoughtStreamModel(QtCore.QObject):
    new_thought = QtCore.Signal(str)

    def append(self, text: str):
        self.new_thought.emit(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, thought_model: ThoughtStreamModel):
        super().__init__()
        self.thought_model = thought_model
        self.setWindowTitle("NEURAL CORE OBSERVATORY")
        self.resize(1200, 800)
        self._init_ui()
        self._apply_dark_sci_fi_theme()

        self.thought_model.new_thought.connect(self._on_new_thought)

        self._start_resource_timer()
        self._start_evolution_log_watcher()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        # Left: organs + vitals
        left_split = QtWidgets.QVBoxLayout()
        self.organs_list = QtWidgets.QListWidget()
        self.organs_list.addItems([
            "Probe Organ",
            "Blueprint Organ",
            "Builder Organ",
            "Brain Organ",
            "Distillation Organ",
            "Evolver Organ",
            "Observer Organ",
        ])
        self.organs_list.setFixedHeight(150)

        self.vitals_text = QtWidgets.QTextEdit()
        self.vitals_text.setReadOnly(True)

        left_split.addWidget(QtWidgets.QLabel("Organs"))
        left_split.addWidget(self.organs_list)
        left_split.addWidget(QtWidgets.QLabel("Resource Vitals"))
        left_split.addWidget(self.vitals_text)

        # Center: thought stream
        center_vbox = QtWidgets.QVBoxLayout()
        center_vbox.addWidget(QtWidgets.QLabel("Mind Stream"))
        self.thought_view = QtWidgets.QTextEdit()
        self.thought_view.setReadOnly(True)
        center_vbox.addWidget(self.thought_view)

        # Right: evolution log
        right_vbox = QtWidgets.QVBoxLayout()
        right_vbox.addWidget(QtWidgets.QLabel("Evolution Log"))
        self.evolution_view = QtWidgets.QTextEdit()
        self.evolution_view.setReadOnly(True)
        right_vbox.addWidget(self.evolution_view)

        layout.addLayout(left_split, 2)
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

    def _start_resource_timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._update_vitals)
        timer.start(1000)
        self._vitals_timer = timer

    def _update_vitals(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        txt = (
            f"CPU: {cpu:.1f}%\n"
            f"RAM: {ram.percent:.1f}% ({ram.used / (1024**3):.1f} / {ram.total / (1024**3):.1f} GB)\n"
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

    @QtCore.Slot(str)
    def _set_evolution_text(self, text: str):
        def update():
            self.evolution_view.setPlainText(text)
        QtCore.QTimer.singleShot(0, update)


# ========= BRAIN LOOP (RUNS IN BACKGROUND THREAD) =========

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

        thought_model.append(f"[INPUT] {user}")

        answers = fanout_to_llms(user, system="You are my neural core.")
        fused = fuse_answers(answers)

        thought_model.append(f"[OUTPUT]\n{fused}")

        add_raw_knowledge({
            "source_type": "conversation",
            "source_name": "user_session",
            "timestamp": time.time(),
            "topic_hint": "general",  # TODO: infer better topics
            "content": f"Q: {user}\nA: {fused}",
        })

        log_evolution_event("interaction", f"User: {user}\nFused: {fused[:300]}...")


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

    # Create Qt application and GUI in MAIN THREAD
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    thought_model = ThoughtStreamModel()
    window = MainWindow(thought_model)
    window.show()

    # Start brain loop in BACKGROUND THREAD
    t = threading.Thread(target=brain_loop, args=(thought_model, cfg), daemon=True)
    t.start()

    # Enter Qt event loop (blocks main thread)
    app.exec()


if __name__ == "__main__":
    main()

