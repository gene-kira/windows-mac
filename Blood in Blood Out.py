import sys
import re
import hashlib
import json
import socket
import uuid
import math
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QTextEdit, QPushButton, QLabel, QFrame, QListWidget, QListWidgetItem,
    QDialog, QTreeWidget, QTreeWidgetItem, QLineEdit, QFileDialog, QComboBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor

try:
    from PyQt5.QtWidgets import QOpenGLWidget  # type: ignore
    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False


# ========================= CORE ORGANS =========================

class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def record_analysis(self, fingerprint: str, findings: List[str]):
        self._data.setdefault("analyses", {})
        self._data["analyses"][fingerprint] = {"findings": findings}
        self.save()

    def record_lineage(self, fingerprint: str, lineage: List[Tuple[str, str, str]]):
        self._data.setdefault("lineage", {})
        self._data["lineage"][fingerprint] = lineage
        self.save()

    def record_ai_lineage(self, fingerprint: str, lineage: List[Tuple[str, str]]):
        self._data.setdefault("ai_lineage", {})
        self._data["ai_lineage"][fingerprint] = lineage
        self.save()

    def get_all(self) -> Dict[str, Any]:
        return self._data


PII_PATTERNS = {
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "biometric": re.compile(r"\bbiometric_[A-Za-z0-9]+\b"),
}


class DataGuard:
    def __init__(self, secret_salt: str):
        self.secret_salt = secret_salt

    def _encode(self, value: str) -> str:
        h = hashlib.sha256((self.secret_salt + value).encode("utf-8")).hexdigest()
        return f"ENC#{h[:16]}"

    def protect_text(self, text: str) -> str:
        protected = text
        for label, pattern in PII_PATTERNS.items():
            def repl(m):
                return f"<{label.upper()}:{self._encode(m.group(0))}>"
            protected = pattern.sub(repl, protected)
        return protected

    def scan_summary(self, text: str) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for label, pattern in PII_PATTERNS.items():
            summary[label] = len(pattern.findall(text))
        return summary


@dataclass
class SystemIdentity:
    alias: str
    encoded_host: str
    encoded_ip: str
    encoded_mac: str


class SystemGuard:
    def __init__(self, secret_salt: str):
        self.secret_salt = secret_salt
        self._identity = self._build_identity()

    def _enc(self, value: str) -> str:
        h = hashlib.sha256((self.secret_salt + value).encode("utf-8")).hexdigest()
        return f"SYS#{h[:12]}"

    def _build_identity(self) -> SystemIdentity:
        host = socket.gethostname()
        try:
            ip = socket.gethostbyname(host)
        except Exception:
            ip = "0.0.0.0"
        mac = ":".join(f"{(uuid.getnode() >> ele) & 0xff:02x}"
                       for ele in range(0, 8 * 6, 8))[::-1]
        return SystemIdentity(
            alias=self._enc(host + ip + mac),
            encoded_host=self._enc(host),
            encoded_ip=self._enc(ip),
            encoded_mac=self._enc(mac),
        )

    @property
    def identity(self) -> SystemIdentity:
        return self._identity

    def scrub_text(self, text: str) -> str:
        t = text.replace(socket.gethostname(), self._identity.encoded_host)
        return t


class SecureLogger:
    def __init__(self, dataguard: DataGuard, systemguard: SystemGuard, sink=None):
        self.dg = dataguard
        self.sg = systemguard
        self.sink = sink

    def _protect(self, msg: str) -> str:
        msg = self.dg.protect_text(msg)
        msg = self.sg.scrub_text(msg)
        return msg

    def _emit(self, level: str, msg: str):
        safe = f"[{level}] {self._protect(msg)}"
        print(safe)
        if self.sink is not None:
            self.sink.add_log(safe)

    def info(self, msg: str):
        self._emit("INFO", msg)

    def warn(self, msg: str):
        self._emit("WARN", msg)

    def error(self, msg: str):
        self._emit("ERROR", msg)


# ========================= AI ENGINE =========================

class AIBackend(Enum):
    RULE_BASED = auto()
    LOCAL_MODEL = auto()
    REMOTE_API = auto()


@dataclass
class AIRewriteResult:
    rewritten: str
    safety_flags: List[str]
    confidence: float
    lineage: List[Tuple[str, str]]


class AIEngine:
    def __init__(self, logger: SecureLogger, memory: MemoryStore,
                 backend: AIBackend = AIBackend.RULE_BASED):
        self.logger = logger
        self.memory = memory
        self.backend = backend

        self.local_model_path: Optional[Path] = None
        self.local_model = None
        self.local_tokenizer = None

        self.remote_api_url: Optional[str] = None
        self.remote_api_key: Optional[str] = None

    def set_backend(self, backend: AIBackend):
        self.backend = backend
        self.logger.info(f"AIEngine backend set to {backend.name}")

    def set_local_model(self, path: Path):
        self.local_model_path = path
        self.logger.info(f"AIEngine local model path set to {path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.local_model = AutoModelForCausalLM.from_pretrained(str(path))
            self.logger.info("AIEngine local model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            self.local_model = None

    def set_remote_api(self, url: str, api_key: Optional[str] = None):
        self.remote_api_url = url
        self.remote_api_key = api_key
        self.logger.info(f"AIEngine remote API URL set to {url}")

    def auto_download_model(self, repo: str, target_path: Path):
        self.logger.info(f"Auto-download requested for model repo: {repo} -> {target_path}")
        # Hook for your own download logic.

    def _safety_filter(self, original: str, rewritten: str) -> Tuple[List[str], float]:
        flags: List[str] = []
        dangerous = ["eval(", "exec(", "os.system(", "subprocess.Popen("]
        for d in dangerous:
            if d in rewritten:
                flags.append(f"DANGEROUS_PRIMITIVE_RETAINED:{d}")
        confidence = max(0.0, 1.0 - 0.2 * len(flags))
        if rewritten.strip() == original.strip():
            flags.append("NO_CHANGE")
            confidence = min(confidence, 0.5)
        return flags, confidence

    def rewrite_defensive(self, code: str) -> AIRewriteResult:
        lineage: List[Tuple[str, str]] = []
        lineage.append(("Input", f"{len(code)} chars"))

        if self.backend == AIBackend.RULE_BASED:
            rewritten = self._rule_based_rewrite(code, lineage)
        elif self.backend == AIBackend.LOCAL_MODEL:
            rewritten = self._local_model_rewrite(code, lineage)
        elif self.backend == AIBackend.REMOTE_API:
            rewritten = self._remote_api_rewrite(code, lineage)
        else:
            self.logger.warn("AIEngine: unknown backend, falling back to rule-based.")
            rewritten = self._rule_based_rewrite(code, lineage)

        flags, confidence = self._safety_filter(code, rewritten)
        lineage.append(("SafetyFilter", f"flags={flags}, confidence={confidence:.2f}"))

        fp = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.memory.record_ai_lineage(fp, lineage)

        return AIRewriteResult(
            rewritten=rewritten,
            safety_flags=flags,
            confidence=confidence,
            lineage=lineage,
        )

    def _rule_based_rewrite(self, code: str, lineage: List[Tuple[str, str]]) -> str:
        self.logger.info("AIEngine: rule-based defensive rewrite.")
        hardened = code.replace("eval(", "raise RuntimeError('AI blocked eval('")
        hardened = hardened.replace("exec(", "raise RuntimeError('AI blocked exec('")
        lineage.append(("RuleBased", "Blocked eval/exec via simple replacements"))
        return "# AIEngine rule-based defensive rewrite\n" + hardened

    def _local_model_rewrite(self, code: str, lineage: List[Tuple[str, str]]) -> str:
        if self.local_model is None:
            self.logger.warn("AIEngine: local model not loaded, falling back to rule-based.")
            return self._rule_based_rewrite(code, lineage)
        try:
            from torch import no_grad
            prompt = (
                "Rewrite the following code to be safe, defensive, and secure. "
                "Remove dangerous primitives and replace them with safe alternatives.\n\n"
                f"Code:\n{code}\n\nRewritten:"
            )
            inputs = self.local_tokenizer(prompt, return_tensors="pt")
            with no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.2,
                    do_sample=False
                )
            rewritten = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            lineage.append(("LocalModel", "Generated defensive rewrite via local model"))
            return rewritten
        except Exception as e:
            self.logger.error(f"Local model rewrite failed: {e}")
            return self._rule_based_rewrite(code, lineage)

    def _remote_api_rewrite(self, code: str, lineage: List[Tuple[str, str]]) -> str:
        if not self.remote_api_url:
            self.logger.warn("AIEngine: remote API URL missing, falling back to rule-based.")
            return self._rule_based_rewrite(code, lineage)
        try:
            import requests
            payload = {"task": "rewrite_defensive", "code": code}
            headers = {}
            if self.remote_api_key:
                headers["Authorization"] = f"Bearer {self.remote_api_key}"
            resp = requests.post(self.remote_api_url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                rewritten = data.get("rewritten", "# Remote API returned no rewrite")
                lineage.append(("RemoteAPI", f"Called {self.remote_api_url}"))
                return rewritten
            else:
                self.logger.error(f"Remote API error: {resp.status_code}")
                return self._rule_based_rewrite(code, lineage)
        except Exception as e:
            self.logger.error(f"Remote API rewrite failed: {e}")
            return self._rule_based_rewrite(code, lineage)

    def describe(self) -> str:
        return f"AIEngine backend: {self.backend.name}"


# ========================= NUCLEAR + ANOMALY ENGINE =========================

class RiskLevel(Enum):
    SAFE = auto()
    SUSPICIOUS = auto()
    MALICIOUS = auto()


class PolicyMode(Enum):
    MONITOR = auto()
    STRICT = auto()
    NUCLEAR = auto()


class Category(Enum):
    CODE = auto()
    NETWORK = auto()
    PROCESS = auto()
    FILE = auto()
    PII = auto()


@dataclass
class AnalysisResult:
    risk: RiskLevel
    findings: List[str]


@dataclass
class TransformResult:
    hardened_code: str
    policy_summary: str
    lineage: List[Tuple[str, str, str]] = field(default_factory=list)


class AnomalyEngine:
    """
    Full anomaly organ:
    - Rolling history per category
    - Mean / std / z-score
    - Simple density anomaly label (cluster-like)
    """
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.history: List[Dict[str, int]] = []

    def update_from_findings(self, findings: List[str]):
        snapshot: Dict[str, int] = {c.name: 0 for c in Category}
        for f in findings:
            if "category:" in f:
                cat = f.split("category:", 1)[1].strip(" )")
                if cat in snapshot:
                    snapshot[cat] += 1
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _stats_for_category(self, cat: str) -> Tuple[int, float, float, float, str]:
        if not self.history:
            return 0, 0.0, 0.0, 0.0, "NORMAL"

        counts = [snap.get(cat, 0) for snap in self.history]
        latest = counts[-1]
        prev = counts[-2] if len(counts) > 1 else 0

        mean = sum(counts) / len(counts)
        var = sum((x - mean) ** 2 for x in counts) / max(1, len(counts) - 1)
        std = var ** 0.5
        z = (latest - mean) / std if std > 0 else 0.0

        anomaly = "NORMAL"
        if latest >= 3 and z >= 2.0:
            anomaly = "HOT_CLUSTER"
        elif latest == 0 and mean > 0:
            anomaly = "COLD_DROP"

        momentum = latest - prev
        heat = min(1.0, latest / 5.0)
        return latest, momentum, z, heat, anomaly

    def current_stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for cat in [c.name for c in Category]:
            count, momentum, z, heat, anomaly = self._stats_for_category(cat)
            stats[cat] = {
                "count": count,
                "momentum": momentum,
                "z": z,
                "heat": heat,
                "anomaly": anomaly,
            }
        return stats


class CodePurifier:
    def __init__(self, memory: MemoryStore, mode: PolicyMode = PolicyMode.MONITOR,
                 anomaly: Optional[AnomalyEngine] = None):
        self.memory = memory
        self.mode = mode
        self.anomaly = anomaly

        self._patterns: Dict[str, Tuple[str, Category]] = {
            r"\beval\s*\(": ("Use of eval() is dangerous.", Category.CODE),
            r"\bexec\s*\(": ("Use of exec() is dangerous.", Category.CODE),
            r"\bos\.system\s*\(": ("Spawning shell commands via os.system is dangerous.", Category.PROCESS),
            r"\bsubprocess\.Popen\b": ("Spawning processes via subprocess.Popen is dangerous.", Category.PROCESS),
            r"\bsocket\.socket\b": ("Network socket usage may be exfil/C2.", Category.NETWORK),
            r"\bopen\s*\(": ("File open may be data exfiltration.", Category.FILE),
        }

        self._rewrites: List[Tuple[re.Pattern, str, Category]] = [
            (re.compile(r"\beval\s*\("), "raise RuntimeError('Blocked eval('", Category.CODE),
            (re.compile(r"\bexec\s*\("), "raise RuntimeError('Blocked exec('", Category.CODE),
            (re.compile(r"\bos\.system\s*\("), "raise RuntimeError('Blocked os.system('", Category.PROCESS),
            (re.compile(r"\bsubprocess\.Popen\b"), "raise RuntimeError('Blocked subprocess.Popen('", Category.PROCESS),
        ]

    def _fingerprint(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def analyze(self, code: str) -> AnalysisResult:
        findings: List[str] = []
        for pattern, (desc, cat) in self._patterns.items():
            if re.search(pattern, code):
                findings.append(f"{desc} (pattern: {pattern}, category: {cat.name})")

        if any("os.system" in f or "subprocess" in f or "eval" in f or "exec" in f for f in findings):
            risk = RiskLevel.MALICIOUS
        elif findings:
            risk = RiskLevel.SUSPICIOUS
        else:
            risk = RiskLevel.SAFE

        if self.anomaly is not None:
            self.anomaly.update_from_findings(findings)

        fp = self._fingerprint(code)
        self.memory.record_analysis(fp, findings)
        return AnalysisResult(risk=risk, findings=findings)

    def _nuclear_block_detail(self, analysis: AnalysisResult) -> str:
        cats = set()
        for f in analysis.findings:
            if "category:" in f:
                cats.add(f.split("category:", 1)[1].strip(" )"))
        if not cats:
            return "Nuclear: suspicious code blocked (no category parsed)."
        return "Nuclear: blocked categories: " + ", ".join(sorted(cats))

    def transform(self, code: str, analysis: AnalysisResult) -> TransformResult:
        lineage: List[Tuple[str, str, str]] = []
        current = code
        lineage.append((Category.CODE.name, "Original", f"{len(current)} chars"))

        if self.mode == PolicyMode.NUCLEAR and analysis.risk != RiskLevel.SAFE:
            detail = self._nuclear_block_detail(analysis)
            lineage.append(("NUCLEAR", "Block", detail))
            summary = [
                f"Risk: {analysis.risk.name}",
                "Policy: NUCLEAR – suspicious code fully blocked.",
                detail,
            ]
            if analysis.findings:
                summary.append("Findings:")
                summary.extend(f"  - {f}" for f in analysis.findings)
            fp = self._fingerprint(code)
            self.memory.record_lineage(fp, lineage)
            return TransformResult(
                hardened_code="# NUCLEAR MODE: CODE BLOCKED",
                policy_summary="\n".join(summary),
                lineage=lineage,
            )

        for pattern, replacement, cat in self._rewrites:
            matches = pattern.findall(current)
            if matches:
                lineage.append((cat.name, "Rewrite",
                                f"{pattern.pattern} -> {replacement} ({len(matches)} occurrence(s))"))
                current = pattern.sub(replacement, current)

        lineage.append((Category.CODE.name, "Final", f"{len(current)} chars after rewrites"))

        policy_lines = [
            f"Risk: {analysis.risk.name}",
            f"Policy: {self.mode.name} – dangerous primitives rewritten and surfaced.",
        ]
        if analysis.findings:
            policy_lines.append("Findings:")
            policy_lines.extend(f"  - {f}" for f in analysis.findings)

        fp = self._fingerprint(code)
        self.memory.record_lineage(fp, lineage)

        return TransformResult(
            hardened_code=current,
            policy_summary="\n".join(policy_lines),
            lineage=lineage,
        )

    def purify(self, code: str) -> TransformResult:
        analysis = self.analyze(code)
        return self.transform(code, analysis)

    def set_mode(self, mode: PolicyMode):
        self.mode = mode


# ========================= UI / VISUAL ORGANS =========================

class ThemeManager:
    THEMES = {
        "dark": """
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QTextEdit { background-color: #2b2b2b; color: #ffffff; }
            QPushButton { background-color: #3a3a3a; color: #ffffff; }
            QListWidget { background-color: #252525; color: #ffffff; }
        """,
        "grass": """
            QWidget { background-color: #0f3d0f; color: #d0ffd0; }
            QTextEdit { background-color: #145214; color: #eaffea; }
            QPushButton { background-color: #1f7a1f; color: #ffffff; }
            QListWidget { background-color: #0f3d0f; color: #d0ffd0; }
        """,
        "nuclear": """
            QWidget { background-color: #000000; color: #39ff14; }
            QTextEdit { background-color: #001100; color: #39ff14; }
            QPushButton { background-color: #003300; color: #39ff14; }
            QListWidget { background-color: #000000; color: #39ff14; }
        """,
    }

    def apply(self, widget, theme_name: str):
        css = self.THEMES.get(theme_name.lower())
        if css:
            widget.setStyleSheet(css)


class PulsingLED(QWidget):
    def __init__(self, color=QColor(0, 255, 0)):
        super().__init__()
        self.color = color
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_phase)
        self.timer.start(50)

    def update_phase(self):
        self.phase = (self.phase + 5) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        intensity = (1 + math.sin(self.phase * math.pi / 180.0)) / 2.0
        c = QColor(self.color)
        c.setAlpha(int(150 + intensity * 105))
        painter.setBrush(c)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())


class MomentumArrow(QLabel):
    def set_momentum(self, value: int):
        if value > 0:
            self.setText("↑")
            self.setStyleSheet("color: red; font-size: 24px;")
        elif value < 0:
            self.setText("↓")
            self.setStyleSheet("color: green; font-size: 24px;")
        else:
            self.setText("→")
            self.setStyleSheet("color: yellow; font-size: 24px;")


class ThreatPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("Threat Panel")
        self.title.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout.addWidget(self.title)

        self.system_alias = QLabel("System Alias: —")
        layout.addWidget(self.system_alias)

        self.pii_counts = QLabel("PII Counts: —")
        layout.addWidget(self.pii_counts)

        self.blocked_patterns = QLabel("Recent Blocked Patterns: —")
        layout.addWidget(self.blocked_patterns)

        self.momentum = MomentumArrow("→")
        layout.addWidget(self.momentum)

        self.led = PulsingLED(QColor(255, 0, 0))
        self.led.setFixedSize(24, 24)
        layout.addWidget(self.led)

    def update_panel(self, alias: str, pii_summary: dict, blocked: list, momentum_value: int):
        self.system_alias.setText(f"System Alias: {alias}")
        pii_text = ", ".join(f"{k}: {v}" for k, v in pii_summary.items() if v > 0)
        self.pii_counts.setText(f"PII Counts: {pii_text or 'none'}")
        if blocked:
            self.blocked_patterns.setText(
                "Recent Blocked Patterns:\n  - " + "\n  - ".join(blocked)
            )
        else:
            self.blocked_patterns.setText("Recent Blocked Patterns: none")
        self.momentum.set_momentum(momentum_value)


class LogViewer(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def add_log(self, text: str):
        self.append(text)
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )


class LineagePopup(QDialog):
    def __init__(self, lineage_steps: List[Tuple[str, str, str]]):
        super().__init__()
        self.setWindowTitle("Lineage Tree")
        layout = QVBoxLayout()
        self.setLayout(layout)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Category", "Action", "Details"])
        layout.addWidget(tree)

        for cat, action, detail in lineage_steps:
            item = QTreeWidgetItem([cat, action, detail])
            tree.addTopLevelItem(item)


class AILineagePopup(QDialog):
    def __init__(self, lineage_steps: List[Tuple[str, str]]):
        super().__init__()
        self.setWindowTitle("AI Lineage")
        layout = QVBoxLayout()
        self.setLayout(layout)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Step", "Details"])
        layout.addWidget(tree)

        for step, detail in lineage_steps:
            item = QTreeWidgetItem([step, detail])
            tree.addTopLevelItem(item)


class GPUOverlayWidget(QOpenGLWidget if HAS_OPENGL else QWidget):
    def __init__(self):
        super().__init__()
        self.heat = 0.0
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    def set_heat(self, heat: float):
        self.heat = max(0.0, min(1.0, heat))
        self.update()

    def _tick(self):
        self.phase = (self.phase + 5.0) % 360.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        intensity = (1 + math.sin(self.phase * math.pi / 180.0)) / 2.0
        alpha = int(50 + 150 * self.heat * intensity)
        color = QColor(255, 0, 0, alpha)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())


class NuclearDashboardPanel(QWidget):
    """
    Per-category nuclear dashboard, driven by AnomalyEngine + AI confidence.
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("Nuclear Dashboard (Per Category)")
        self.title.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout.addWidget(self.title)

        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        self.labels: Dict[str, Dict[str, QLabel]] = {}
        row = 0
        for cat in Category:
            name = cat.name
            lbl_cat = QLabel(name)
            lbl_count = QLabel("Count: 0")
            lbl_mom = QLabel("Momentum: 0")
            lbl_z = QLabel("Z: 0.00")
            lbl_heat = QLabel("Heat: 0.00")
            lbl_anom = QLabel("Anomaly: NORMAL")
            self.grid.addWidget(lbl_cat, row, 0)
            self.grid.addWidget(lbl_count, row, 1)
            self.grid.addWidget(lbl_mom, row, 2)
            self.grid.addWidget(lbl_z, row, 3)
            self.grid.addWidget(lbl_heat, row, 4)
            self.grid.addWidget(lbl_anom, row, 5)
            self.labels[name] = {
                "count": lbl_count,
                "momentum": lbl_mom,
                "z": lbl_z,
                "heat": lbl_heat,
                "anomaly": lbl_anom,
            }
            row += 1

        self.ai_conf_label = QLabel("AI Confidence: —")
        layout.addWidget(self.ai_conf_label)

        self.ai_flags_label = QLabel("AI Flags: —")
        layout.addWidget(self.ai_flags_label)

    def update_stats(self, stats: Dict[str, Dict[str, Any]],
                     ai_conf: float, ai_flags: List[str]):
        for cat, vals in stats.items():
            if cat in self.labels:
                self.labels[cat]["count"].setText(f"Count: {vals['count']}")
                self.labels[cat]["momentum"].setText(f"Momentum: {vals['momentum']:.0f}")
                self.labels[cat]["z"].setText(f"Z: {vals['z']:.2f}")
                self.labels[cat]["heat"].setText(f"Heat: {vals['heat']:.2f}")
                self.labels[cat]["anomaly"].setText(f"Anomaly: {vals['anomaly']}")
        self.ai_conf_label.setText(f"AI Confidence: {ai_conf:.2f}")
        self.ai_flags_label.setText(f"AI Flags: {ai_flags or ['NONE']}")


# ========================= UIAUTOMATION ORGAN (STUB) =========================

class UIAutomationOrgan:
    def __init__(self, logger: SecureLogger):
        self.logger = logger

    def scan_active_windows(self):
        self.logger.info("UIAutomation: scan_active_windows() called (stub).")

    def apply_protective_overlays(self):
        self.logger.info("UIAutomation: apply_protective_overlays() called (stub).")


# ========================= COCKPIT =========================

class Cockpit(QWidget):
    def __init__(self):
        super().__init__()

        self.memory = MemoryStore(path=Path("memory/purifier_memory.json"))
        self.dataguard = DataGuard(secret_salt="your-secret-here")
        self.systemguard = SystemGuard(secret_salt="your-secret-here")
        self.log_viewer = LogViewer()
        self.logger = SecureLogger(self.dataguard, self.systemguard, sink=self.log_viewer)

        self.anomaly = AnomalyEngine()
        self.engine = CodePurifier(memory=self.memory, mode=PolicyMode.MONITOR,
                                   anomaly=self.anomaly)
        self.ai_engine = AIEngine(logger=self.logger, memory=self.memory,
                                  backend=AIBackend.RULE_BASED)
        self.ui_auto = UIAutomationOrgan(logger=self.logger)

        self.theme_manager = ThemeManager()
        self.last_threat_count = 0
        self._last_lineage: List[Tuple[str, str, str]] = []
        self._last_ai_lineage: List[Tuple[str, str]] = []
        self._last_ai_confidence: float = 0.0
        self._last_ai_flags: List[str] = []

        self.sound_threat = None
        self.sound_nuclear = None
        self.sound_pii = None

        self.setWindowTitle("Defensive Code Purifier – Anomaly Nuclear Cockpit")
        self.resize(1900, 1000)

        self._build_ui()
        self.theme_manager.apply(self, "dark")

    def _build_ui(self):
        root = QHBoxLayout()
        self.setLayout(root)

        nav = QListWidget()
        for name in ["Code", "Threats", "Logs", "AI Engine", "Memory", "Nuclear", "Lineage", "Automation"]:
            nav.addItem(QListWidgetItem(name))
        nav.currentRowChanged.connect(self.on_nav_changed)
        nav.setFixedWidth(160)
        root.addWidget(nav)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, stretch=1)

        self.overlay = GPUOverlayWidget()
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay.raise_()
        self.stack.installEventFilter(self)

        # Code page
        code_page = QWidget()
        code_layout = QVBoxLayout()
        code_page.setLayout(code_layout)

        top_bar = QHBoxLayout()
        btn_dark = QPushButton("Dark")
        btn_dark.clicked.connect(lambda: self.theme_manager.apply(self, "dark"))
        top_bar.addWidget(btn_dark)

        btn_grass = QPushButton("Grass")
        btn_grass.clicked.connect(lambda: self.theme_manager.apply(self, "grass"))
        top_bar.addWidget(btn_grass)

        btn_nuclear_theme = QPushButton("Nuclear Theme")
        btn_nuclear_theme.clicked.connect(lambda: self.theme_manager.apply(self, "nuclear"))
        top_bar.addWidget(btn_nuclear_theme)

        code_layout.addLayout(top_bar)

        mid = QHBoxLayout()
        self.raw_edit = QTextEdit()
        self.raw_edit.setPlaceholderText("Paste suspicious or unknown code here...")
        mid.addWidget(self.raw_edit, stretch=3)

        self.hardened_edit = QTextEdit()
        self.hardened_edit.setReadOnly(True)
        self.hardened_edit.setPlaceholderText("Hardened output will appear here...")
        mid.addWidget(self.hardened_edit, stretch=3)

        self.threat_panel = ThreatPanel()
        mid.addWidget(self.threat_panel, stretch=1)

        code_layout.addLayout(mid)

        bottom = QHBoxLayout()
        self.status_label = QLabel("Risk: —")
        self.status_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        bottom.addWidget(self.status_label, stretch=1)

        btn_purify = QPushButton("Purify")
        btn_purify.clicked.connect(self.on_purify)
        bottom.addWidget(btn_purify)

        btn_lineage = QPushButton("Lineage Tree")
        btn_lineage.clicked.connect(self.show_lineage)
        bottom.addWidget(btn_lineage)

        btn_ai_rewrite = QPushButton("AI Rewrite")
        btn_ai_rewrite.clicked.connect(self.on_ai_rewrite)
        bottom.addWidget(btn_ai_rewrite)

        btn_ai_lineage = QPushButton("AI Lineage")
        btn_ai_lineage.clicked.connect(self.show_ai_lineage)
        bottom.addWidget(btn_ai_lineage)

        code_layout.addLayout(bottom)
        self.stack.addWidget(code_page)

        # Threats page
        threats_page = QWidget()
        t_layout = QVBoxLayout()
        threats_page.setLayout(t_layout)
        self.threat_detail_label = QLabel("Threat details will appear here.")
        t_layout.addWidget(self.threat_detail_label)
        self.stack.addWidget(threats_page)

        # Logs page
        logs_page = QWidget()
        l_layout = QVBoxLayout()
        logs_page.setLayout(l_layout)
        l_layout.addWidget(self.log_viewer)
        self.stack.addWidget(logs_page)

        # AI Engine page
        ai_page = QWidget()
        ai_layout = QVBoxLayout()
        ai_page.setLayout(ai_layout)

        self.ai_status = QLabel(self.ai_engine.describe())
        ai_layout.addWidget(self.ai_status)

        ai_layout.addWidget(QLabel("AI Backend:"))
        self.ai_backend_combo = QComboBox()
        self.ai_backend_combo.addItems([b.name for b in AIBackend])
        self.ai_backend_combo.setCurrentText(self.ai_engine.backend.name)
        self.ai_backend_combo.currentTextChanged.connect(self.on_set_ai_backend)
        ai_layout.addWidget(self.ai_backend_combo)

        ai_layout.addWidget(QLabel("Local Model Path:"))
        self.ai_local_model_path_edit = QLineEdit("")
        ai_layout.addWidget(self.ai_local_model_path_edit)

        btn_browse_model = QPushButton("Browse Local Model...")
        btn_browse_model.clicked.connect(self.on_browse_local_model)
        ai_layout.addWidget(btn_browse_model)

        btn_load_model = QPushButton("Load Local Model")
        btn_load_model.clicked.connect(self.on_load_local_model)
        ai_layout.addWidget(btn_load_model)

        ai_layout.addWidget(QLabel("Auto-Download Repo (hook):"))
        self.ai_repo_edit = QLineEdit("your/model-repo")
        ai_layout.addWidget(self.ai_repo_edit)

        btn_auto_download = QPushButton("Auto-Download Model (hook)")
        btn_auto_download.clicked.connect(self.on_auto_download_model)
        ai_layout.addWidget(btn_auto_download)

        ai_layout.addWidget(QLabel("Remote API URL:"))
        self.ai_remote_url_edit = QLineEdit("")
        ai_layout.addWidget(self.ai_remote_url_edit)

        ai_layout.addWidget(QLabel("Remote API Key (optional):"))
        self.ai_remote_key_edit = QLineEdit("")
        ai_layout.addWidget(self.ai_remote_key_edit)

        btn_set_remote = QPushButton("Set Remote API")
        btn_set_remote.clicked.connect(self.on_set_remote_api)
        ai_layout.addWidget(btn_set_remote)

        self.ai_safety_label = QLabel("AI Safety: —")
        ai_layout.addWidget(self.ai_safety_label)

        self.ai_confidence_label = QLabel("AI Confidence: —")
        ai_layout.addWidget(self.ai_confidence_label)

        self.stack.addWidget(ai_page)

        # Memory page
        mem_page = QWidget()
        mem_layout = QVBoxLayout()
        mem_page.setLayout(mem_layout)

        self.mem_path_edit = QLineEdit(str(self.memory.path))
        mem_layout.addWidget(QLabel("Memory Path (local or SMB):"))
        mem_layout.addWidget(self.mem_path_edit)

        mem_btn_bar = QHBoxLayout()
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.on_browse_memory_path)
        mem_btn_bar.addWidget(btn_browse)

        btn_load_mem = QPushButton("Load Memory")
        btn_load_mem.clicked.connect(self.on_load_memory)
        mem_btn_bar.addWidget(btn_load_mem)

        btn_save_mem = QPushButton("Save Memory")
        btn_save_mem.clicked.connect(self.on_save_memory)
        mem_btn_bar.addWidget(btn_save_mem)

        mem_layout.addLayout(mem_btn_bar)

        self.mem_view = QTextEdit()
        self.mem_view.setReadOnly(True)
        mem_layout.addWidget(self.mem_view)

        self.stack.addWidget(mem_page)

        # Nuclear page
        nuc_page = QWidget()
        nuc_layout = QVBoxLayout()
        nuc_page.setLayout(nuc_layout)

        nuc_layout.addWidget(QLabel("Nuclear Mode Tier System"))

        btn_monitor = QPushButton("Mode: MONITOR")
        btn_monitor.clicked.connect(lambda: self.set_policy_mode(PolicyMode.MONITOR))
        nuc_layout.addWidget(btn_monitor)

        btn_strict = QPushButton("Mode: STRICT")
        btn_strict.clicked.connect(lambda: self.set_policy_mode(PolicyMode.STRICT))
        nuc_layout.addWidget(btn_strict)

        btn_nuclear = QPushButton("Mode: NUCLEAR")
        btn_nuclear.clicked.connect(lambda: self.set_policy_mode(PolicyMode.NUCLEAR))
        nuc_layout.addWidget(btn_nuclear)

        self.nuclear_status = QLabel("Current Mode: MONITOR")
        nuc_layout.addWidget(self.nuclear_status)

        self.nuclear_dashboard = NuclearDashboardPanel()
        nuc_layout.addWidget(self.nuclear_dashboard)

        self.stack.addWidget(nuc_page)

        # Lineage page
        lineage_page = QWidget()
        ln_layout = QVBoxLayout()
        lineage_page.setLayout(ln_layout)
        self.lineage_label = QLabel("Lineage summary will appear here.")
        ln_layout.addWidget(self.lineage_label)
        self.stack.addWidget(lineage_page)

        # Automation page
        auto_page = QWidget()
        auto_layout = QVBoxLayout()
        auto_page.setLayout(auto_layout)

        auto_layout.addWidget(QLabel("UI Automation Organ (stub)"))
        btn_scan_windows = QPushButton("Scan Active Windows (stub)")
        btn_scan_windows.clicked.connect(self.on_scan_windows)
        auto_layout.addWidget(btn_scan_windows)

        btn_apply_overlays = QPushButton("Apply Protective Overlays (stub)")
        btn_apply_overlays.clicked.connect(self.on_apply_overlays)
        auto_layout.addWidget(btn_apply_overlays)

        self.stack.addWidget(auto_page)

        nav.setCurrentRow(0)

    def eventFilter(self, obj, event):
        if obj is self.stack and event.type() == event.Resize:
            self.overlay.setGeometry(self.stack.geometry())
        return super().eventFilter(obj, event)

    def on_nav_changed(self, index: int):
        self.stack.setCurrentIndex(index)

    def play_sound(self, sound_obj):
        if sound_obj is not None:
            sound_obj.play()

    def set_policy_mode(self, mode: PolicyMode):
        self.engine.set_mode(mode)
        self.nuclear_status.setText(f"Current Mode: {mode.name}")
        if mode == PolicyMode.NUCLEAR:
            self.theme_manager.apply(self, "nuclear")
            self.status_label.setText("Risk: NUCLEAR MODE ENABLED")
            self.play_sound(self.sound_nuclear)
            self.log_viewer.add_log("[NUCLEAR] Nuclear mode enabled")
        else:
            self.theme_manager.apply(self, "dark")
            self.status_label.setText(f"Risk: Mode set to {mode.name}")
            self.log_viewer.add_log(f"[INFO] Policy mode set to {mode.name}")

    def on_purify(self):
        code = self.raw_edit.toPlainText().strip()
        if not code:
            return

        pii_summary = self.dataguard.scan_summary(code)
        if any(v > 0 for v in pii_summary.values()):
            self.play_sound(self.sound_pii)

        result = self.engine.purify(code)
        self._last_lineage = result.lineage

        self.hardened_edit.setPlainText(result.hardened_code)

        safe_policy = self.dataguard.protect_text(result.policy_summary)
        safe_policy = self.systemguard.scrub_text(safe_policy)
        self.status_label.setText(f"Risk: {safe_policy.splitlines()[0]}")

        blocked = result.policy_summary.splitlines()[2:]
        alias = self.systemguard.identity.alias

        current_threat_count = len(blocked)
        momentum_value = current_threat_count - self.last_threat_count
        self.last_threat_count = current_threat_count

        self.threat_panel.update_panel(alias, pii_summary, blocked, momentum_value)

        self.log_viewer.add_log(f"[INFO] Purified code; PII={pii_summary}, blocked={len(blocked)}")
        if blocked:
            self.play_sound(self.sound_threat)

        self.threat_detail_label.setText(safe_policy)
        self.lineage_label.setText(
            "\n".join(f"{cat} | {action} | {detail}" for cat, action, detail in result.lineage)
        )

        stats = self.anomaly.current_stats()
        global_heat = max(v["heat"] for v in stats.values()) if stats else 0.0
        self.overlay.set_heat(global_heat)
        self.nuclear_dashboard.update_stats(stats, self._last_ai_confidence, self._last_ai_flags)

    def on_ai_rewrite(self):
        code = self.raw_edit.toPlainText().strip()
        if not code:
            return

        ai_result = self.ai_engine.rewrite_defensive(code)
        self._last_ai_lineage = ai_result.lineage
        self._last_ai_confidence = ai_result.confidence
        self._last_ai_flags = ai_result.safety_flags

        self.hardened_edit.setPlainText(ai_result.rewritten)
        self.ai_safety_label.setText(f"AI Safety Flags: {ai_result.safety_flags or ['NONE']}")
        self.ai_confidence_label.setText(f"AI Confidence: {ai_result.confidence:.2f}")

        self.log_viewer.add_log(
            f"[AI] Rewrite applied; confidence={ai_result.confidence:.2f}, flags={ai_result.safety_flags}"
        )

        stats = self.anomaly.current_stats()
        global_heat = max(v["heat"] for v in stats.values()) if stats else 0.0
        self.overlay.set_heat(global_heat)
        self.nuclear_dashboard.update_stats(stats, self._last_ai_confidence, self._last_ai_flags)

    def on_set_ai_backend(self, name: str):
        try:
            backend = AIBackend[name]
            self.ai_engine.set_backend(backend)
            self.ai_status.setText(self.ai_engine.describe())
        except KeyError:
            self.log_viewer.add_log(f"[ERROR] Invalid AI backend: {name}")

    def on_browse_local_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Local Model Directory")
        if path:
            self.ai_local_model_path_edit.setText(path)

    def on_load_local_model(self):
        path_text = self.ai_local_model_path_edit.text().strip()
        if not path_text:
            return
        path = Path(path_text)
        self.ai_engine.set_local_model(path)
        self.ai_status.setText(self.ai_engine.describe())

    def on_auto_download_model(self):
        repo = self.ai_repo_edit.text().strip()
        if not repo:
            return
        target = Path(self.ai_local_model_path_edit.text().strip() or "models/auto")
        self.ai_engine.auto_download_model(repo, target)
        self.log_viewer.add_log(f"[AI] Auto-download hook called for repo={repo}, target={target}")

    def on_set_remote_api(self):
        url = self.ai_remote_url_edit.text().strip()
        key = self.ai_remote_key_edit.text().strip() or None
        if not url:
            return
        self.ai_engine.set_remote_api(url, key)
        self.ai_status.setText(self.ai_engine.describe())

    def show_lineage(self):
        if not self._last_lineage:
            return
        popup = LineagePopup(self._last_lineage)
        popup.exec_()

    def show_ai_lineage(self):
        if not self._last_ai_lineage:
            return
        popup = AILineagePopup(self._last_ai_lineage)
        popup.exec_()

    def on_browse_memory_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Memory File", str(self.memory.path))
        if path:
            self.mem_path_edit.setText(path)

    def on_load_memory(self):
        new_path = Path(self.mem_path_edit.text())
        self.memory = MemoryStore(path=new_path)
        self.engine.memory = self.memory
        self.mem_view.setPlainText(json.dumps(self.memory.get_all(), indent=2))
        self.log_viewer.add_log(f"[INFO] Memory loaded from {new_path}")

    def on_save_memory(self):
        new_path = Path(self.mem_path_edit.text())
        self.memory.path = new_path
        self.memory.save()
        self.mem_view.setPlainText(json.dumps(self.memory.get_all(), indent=2))
        self.log_viewer.add_log(f"[INFO] Memory saved to {new_path}")

    def on_scan_windows(self):
        self.ui_auto.scan_active_windows()

    def on_apply_overlays(self):
        self.ui_auto.apply_protective_overlays()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    cockpit = Cockpit()
    cockpit.show()
    sys.exit(app.exec_())

