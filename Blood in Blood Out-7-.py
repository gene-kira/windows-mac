import sys
import re
import os
import hashlib
import json
import socket
import uuid
import math
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QTextEdit, QPushButton, QLabel, QFrame, QListWidget, QListWidgetItem,
    QDialog, QTreeWidget, QTreeWidgetItem, QLineEdit, QFileDialog, QComboBox,
    QGridLayout, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QSize, QEvent
from PyQt5.QtGui import QPainter, QColor, QPen

# Try OpenGL widget; we’ll also enable software OpenGL at app level
try:
    from PyQt5.QtWidgets import QOpenGLWidget  # type: ignore
    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False

# Optional numeric / clustering backends
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    from cuml.cluster import DBSCAN as cuDBSCAN  # type: ignore
    HAS_CUML = True
except Exception:
    HAS_CUML = False

try:
    import numpy as np
    from sklearn.cluster import DBSCAN as skDBSCAN  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Optional crypto for encrypted memory
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

# Optional psutil for network/process info
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# Optional pywin32 for UI automation
try:
    import win32gui
    import win32process
    import win32con
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False


# ========================= ENCRYPTED MEMORY STORE =========================

class EncryptedMemoryStore:
    def __init__(self, path: Path, secret_salt: str):
        self.path = path
        self.secret_salt = secret_salt
        self._data: Dict[str, Any] = {}
        self._key = self._derive_key(secret_salt)
        self._fernet = Fernet(self._key) if HAS_CRYPTO else None
        self._load()

    def _derive_key(self, salt: str) -> bytes:
        h = hashlib.sha256(salt.encode("utf-8")).digest()
        if HAS_CRYPTO:
            import base64
            return base64.urlsafe_b64encode(h)
        else:
            return h

    def _encrypt(self, raw: bytes) -> bytes:
        if self._fernet is not None:
            return self._fernet.encrypt(raw)
        key = self._key
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))

    def _decrypt(self, enc: bytes) -> bytes:
        if self._fernet is not None:
            return self._fernet.decrypt(enc)
        key = self._key
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(enc))

    def _load(self):
        if self.path.exists():
            try:
                enc = self.path.read_bytes()
                raw = self._decrypt(enc)
                self._data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self._data, indent=2).encode("utf-8")
        enc = self._encrypt(raw)
        self.path.write_bytes(enc)

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


# ========================= DATA + SYSTEM GUARD =========================

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
        self._lock = threading.Lock()

    def _protect(self, msg: str) -> str:
        msg = self.dg.protect_text(msg)
        msg = self.sg.scrub_text(msg)
        return msg

    def _emit(self, level: str, msg: str):
        safe = f"[{level}] {self._protect(msg)}"
        with self._lock:
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
    LOCAL_HTTP = auto()
    REMOTE_OPENAI = auto()


@dataclass
class AIRewriteResult:
    rewritten: str
    safety_flags: List[str]
    confidence: float
    lineage: List[Tuple[str, str]]


class AIEngine:
    def __init__(self, logger: SecureLogger, memory: EncryptedMemoryStore,
                 backend: AIBackend = AIBackend.RULE_BASED):
        self.logger = logger
        self.memory = memory
        self.backend = backend

        self.local_http_url: Optional[str] = None
        self.local_http_model: Optional[str] = None

        self.remote_base_url: Optional[str] = None
        self.remote_api_key: Optional[str] = None
        self.remote_model: Optional[str] = None

        self.auto_mode: bool = True

    def set_backend(self, backend: AIBackend):
        self.backend = backend
        self.logger.info(f"AIEngine backend set to {backend.name}")

    def set_auto_mode(self, enabled: bool):
        self.auto_mode = enabled
        self.logger.info(f"AIEngine auto_mode={enabled}")

    def set_local_http(self, url: str, model: str):
        self.local_http_url = url
        self.local_http_model = model
        self.logger.info(f"AIEngine local HTTP set to {url} (model={model})")

    def set_remote_openai(self, base_url: str, api_key: str, model: str):
        self.remote_base_url = base_url.rstrip("/")
        self.remote_api_key = api_key
        self.remote_model = model
        self.logger.info(f"AIEngine remote OpenAI set to {base_url} (model={model})")

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

    def _choose_backend(self) -> AIBackend:
        if not self.auto_mode:
            return self.backend
        if self.local_http_url and self.local_http_model:
            return AIBackend.LOCAL_HTTP
        if self.remote_base_url and self.remote_api_key and self.remote_model:
            return AIBackend.REMOTE_OPENAI
        return AIBackend.RULE_BASED

    def rewrite_defensive(self, code: str) -> AIRewriteResult:
        lineage: List[Tuple[str, str]] = []
        lineage.append(("Input", f"{len(code)} chars"))

        backend = self._choose_backend()
        lineage.append(("BackendSelection", f"Chosen backend={backend.name} (auto={self.auto_mode})"))

        if backend == AIBackend.RULE_BASED:
            rewritten = self._rule_based_rewrite(code, lineage)
        elif backend == AIBackend.LOCAL_HTTP:
            rewritten = self._local_http_rewrite(code, lineage)
        elif backend == AIBackend.REMOTE_OPENAI:
            rewritten = self._remote_openai_rewrite(code, lineage)
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

    def _local_http_rewrite(self, code: str, lineage: List[Tuple[str, str]]) -> str:
        if not self.local_http_url or not self.local_http_model:
            self.logger.warn("AIEngine: local HTTP not configured, falling back to rule-based.")
            return self._rule_based_rewrite(code, lineage)
        try:
            import requests
            prompt = (
                "You are a defensive security code rewriter.\n"
                "Rewrite the following code to be safe, defensive, and secure.\n"
                "Remove dangerous primitives (eval, exec, os.system, subprocess, raw sockets) and replace them with safe alternatives.\n"
                "Return ONLY the rewritten code, no explanations.\n\n"
                f"Code:\n{code}\n\nRewritten code:\n"
            )
            payload = {
                "model": self.local_http_model,
                "messages": [
                    {"role": "system", "content": "You are a secure code rewriting assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            resp = requests.post(self.local_http_url, json=payload, timeout=30)
            if resp.status_code != 200:
                self.logger.error(f"Local HTTP error: {resp.status_code} {resp.text[:200]}")
                return self._rule_based_rewrite(code, lineage)
            data = resp.json()
            rewritten = data["choices"][0]["message"]["content"]
            lineage.append(("LocalHTTP", f"Called {self.local_http_url} model={self.local_http_model}"))
            return rewritten
        except Exception as e:
            self.logger.error(f"Local HTTP rewrite failed: {e}")
            return self._rule_based_rewrite(code, lineage)

    def _remote_openai_rewrite(self, code: str, lineage: List[Tuple[str, str]]) -> str:
        if not self.remote_base_url or not self.remote_api_key or not self.remote_model:
            self.logger.warn("AIEngine: remote OpenAI not configured, falling back to rule-based.")
            return self._rule_based_rewrite(code, lineage)
        try:
            import requests
            url = f"{self.remote_base_url}/chat/completions"
            prompt = (
                "You are a defensive security code rewriter.\n"
                "Rewrite the following code to be safe, defensive, and secure.\n"
                "Remove dangerous primitives (eval, exec, os.system, subprocess, raw sockets) and replace them with safe alternatives.\n"
                "Return ONLY the rewritten code, no explanations.\n\n"
                f"Code:\n{code}\n\nRewritten code:\n"
            )
            payload = {
                "model": self.remote_model,
                "messages": [
                    {"role": "system", "content": "You are a secure code rewriting assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            headers = {
                "Authorization": f"Bearer {self.remote_api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                self.logger.error(f"Remote OpenAI error: {resp.status_code} {resp.text[:200]}")
                return self._rule_based_rewrite(code, lineage)
            data = resp.json()
            rewritten = data["choices"][0]["message"]["content"]
            lineage.append(("RemoteOpenAI", f"Called {url} model={self.remote_model}"))
            return rewritten
        except Exception as e:
            self.logger.error(f"Remote OpenAI rewrite failed: {e}")
            return self._rule_based_rewrite(code, lineage)

    def describe(self) -> str:
        mode = "AUTO" if self.auto_mode else "MANUAL"
        return f"AIEngine backend: {self.backend.name} (mode={mode})"


# ========================= NUCLEAR + ANALYSIS =========================

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


# -------- DeepAnomalyOrgan (DBSCAN + predictive) --------

@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    density: float
    categories: List[str]
    volatility: float
    fingerprint: str


class DeepAnomalyOrgan:
    def __init__(self, max_history: int = 100, logger: Optional[SecureLogger] = None):
        self.max_history = max_history
        self.history: List[Dict[str, int]] = []
        self._vectors: List[List[float]] = []
        self._clusters: List[ClusterInfo] = []
        self._last_labels: Optional[List[int]] = None
        self.logger = logger
        self._lock = threading.Lock()

    def safe_update_from_findings(self, findings: List[str]):
        try:
            self.update_from_findings(findings)
        except Exception as e:
            if self.logger:
                self.logger.error(f"DeepAnomalyOrgan failure: {e}")

    def update_from_findings(self, findings: List[str]):
        snapshot: Dict[str, int] = {c.name: 0 for c in Category}
        for f in findings:
            if "category:" in f:
                cat = f.split("category:", 1)[1].strip(" )")
                if cat in snapshot:
                    snapshot[cat] += 1
        with self._lock:
            self.history.append(snapshot)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            vec = self._build_vector(snapshot)
            self._vectors.append(vec)
            if len(self._vectors) > self.max_history:
                self._vectors.pop(0)

        self._run_clustering()

    def _build_vector(self, snapshot: Dict[str, int]) -> List[float]:
        counts = [snapshot.get(c.name, 0) for c in Category]
        if len(self.history) > 1:
            prev = self.history[-2]
        else:
            prev = {c.name: 0 for c in Category}
        momentum = [snapshot.get(c.name, 0) - prev.get(c.name, 0) for c in Category]

        z_scores = []
        for c in Category:
            series = [h.get(c.name, 0) for h in self.history]
            mean = sum(series) / len(series)
            var = sum((x - mean) ** 2 for x in series) / max(1, len(series) - 1)
            std = var ** 0.5
            latest = series[-1]
            z = (latest - mean) / std if std > 0 else 0.0
            z_scores.append(z)

        return counts + z_scores + momentum

    def _run_clustering(self):
        with self._lock:
            if len(self._vectors) < 5:
                self._clusters = []
                self._last_labels = None
                return

            X = self._vectors[-50:]
        n_samples = len(X)

        labels = None

        if HAS_CUML and HAS_CUPY:
            try:
                X_gpu = cp.asarray(X, dtype=cp.float32)
                db = cuDBSCAN(eps=1.5, min_samples=3)
                labels_gpu = db.fit_predict(X_gpu)
                labels = cp.asnumpy(labels_gpu).tolist()
            except Exception:
                labels = None

        if labels is None and HAS_SKLEARN:
            try:
                X_np = np.asarray(X, dtype=np.float32)
                db = skDBSCAN(eps=1.5, min_samples=3)
                labels = db.fit_predict(X_np).tolist()
            except Exception:
                labels = None

        if labels is None:
            labels = self._pure_python_dbscan(X, eps=2.5, min_samples=3)

        with self._lock:
            self._last_labels = labels
            self._clusters = self._build_cluster_info(X, labels)

    def _pure_python_dbscan(self, X: List[List[float]], eps: float, min_samples: int) -> List[int]:
        n = len(X)
        labels = [-1] * n
        visited = [False] * n
        cluster_id = 0

        def dist(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        def region_query(idx):
            return [j for j in range(n) if dist(X[idx], X[j]) <= eps]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seeds = neighbors[:]
                while seeds:
                    j = seeds.pop()
                    if not visited[j]:
                        visited[j] = True
                        n2 = region_query(j)
                        if len(n2) >= min_samples:
                            for p in n2:
                                if p not in seeds:
                                    seeds.append(p)
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1
        return labels

    def _build_cluster_info(self, X: List[List[float]], labels: List[int]) -> List[ClusterInfo]:
        if not X or not labels:
            return []

        clusters: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            if lab == -1:
                continue
            clusters.setdefault(lab, []).append(idx)

        infos: List[ClusterInfo] = []
        for cid, idxs in clusters.items():
            size = len(idxs)
            if size == 0:
                continue
            if HAS_SKLEARN:
                arr = np.asarray([X[i] for i in idxs], dtype=np.float32)
                centroid = arr.mean(axis=0)
                diffs = arr - centroid
                var = (diffs ** 2).mean()
                volatility = float(var ** 0.5)
            else:
                arr = [X[i] for i in idxs]
                dim = len(arr[0])
                centroid = [sum(v[d] for v in arr) / size for d in range(dim)]
                var = sum(
                    sum((v[d] - centroid[d]) ** 2 for d in range(dim))
                    for v in arr
                ) / (size * dim)
                volatility = var ** 0.5

            density = size / max(1, len(X))
            cat_counts = {c.name: 0 for c in Category}
            for i in idxs:
                snap = self.history[-len(X) + i]
                for c in Category:
                    cat_counts[c.name] += snap.get(c.name, 0)
            top_cats = sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)
            categories = [name for name, cnt in top_cats if cnt > 0][:3]

            fp_src = ",".join(f"{x:.2f}" for x in centroid[:10])
            fingerprint = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:16]

            infos.append(
                ClusterInfo(
                    cluster_id=cid,
                    size=size,
                    density=float(density),
                    categories=categories,
                    volatility=float(volatility),
                    fingerprint=fingerprint,
                )
            )
        return infos

    def current_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            history = list(self.history)
        stats: Dict[str, Dict[str, Any]] = {}
        if not history:
            for c in Category:
                stats[c.name] = {
                    "count": 0,
                    "momentum": 0.0,
                    "z": 0.0,
                    "heat": 0.0,
                    "anomaly": "NORMAL",
                }
            return stats

        latest = history[-1]
        prev = history[-2] if len(history) > 1 else {c.name: 0 for c in Category}

        for c in Category:
            name = c.name
            series = [h.get(name, 0) for h in history]
            mean = sum(series) / len(series)
            var = sum((x - mean) ** 2 for x in series) / max(1, len(series) - 1)
            std = var ** 0.5
            last = series[-1]
            z = (last - mean) / std if std > 0 else 0.0
            momentum = last - prev.get(name, 0)
            heat = min(1.0, last / 5.0)

            anomaly = "NORMAL"
            if last >= 3 and z >= 2.0:
                anomaly = "HOT_CLUSTER"
            elif last == 0 and mean > 0:
                anomaly = "COLD_DROP"

            stats[name] = {
                "count": last,
                "momentum": momentum,
                "z": z,
                "heat": heat,
                "anomaly": anomaly,
            }
        return stats

    def current_clusters(self) -> List[ClusterInfo]:
        with self._lock:
            return list(self._clusters)

    def predictive_horizon(self) -> Dict[str, Any]:
        with self._lock:
            history = list(self.history)
        if not history:
            return {"short": "STABLE", "medium": "STABLE", "long": "STABLE"}

        total_series = [sum(h.values()) for h in history]
        if len(total_series) < 3:
            return {"short": "STABLE", "medium": "STABLE", "long": "STABLE"}

        def trend(series, window):
            if len(series) < window + 1:
                return 0.0
            recent = series[-window:]
            prev = series[-window - 1:-1]
            return (sum(recent) - sum(prev)) / window

        short_trend = trend(total_series, 3)
        med_trend = trend(total_series, 10) if len(total_series) >= 11 else short_trend
        long_trend = trend(total_series, 20) if len(total_series) >= 21 else med_trend

        def label(t):
            if t > 1.0:
                return "RISING"
            if t < -1.0:
                return "FALLING"
            return "STABLE"

        return {
            "short": label(short_trend),
            "medium": label(med_trend),
            "long": label(long_trend),
        }


class CodePurifier:
    def __init__(self, memory: EncryptedMemoryStore, mode: PolicyMode = PolicyMode.MONITOR,
                 anomaly: Optional[DeepAnomalyOrgan] = None, logger: Optional[SecureLogger] = None):
        self.memory = memory
        self.mode = mode
        self.anomaly = anomaly
        self.logger = logger

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

    def safe_analyze(self, code: str) -> AnalysisResult:
        try:
            return self.analyze(code)
        except Exception as e:
            if self.logger:
                self.logger.error(f"CodePurifier.analyze failure: {e}")
            return AnalysisResult(risk=RiskLevel.SAFE, findings=[])

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
            self.anomaly.safe_update_from_findings(findings)

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

    def safe_transform(self, code: str, analysis: AnalysisResult) -> TransformResult:
        try:
            return self.transform(code, analysis)
        except Exception as e:
            if self.logger:
                self.logger.error(f"CodePurifier.transform failure: {e}")
            return TransformResult(
                hardened_code="# ERROR: transform failure",
                policy_summary=f"Transform failure: {e}",
                lineage=[],
            )

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
        analysis = self.safe_analyze(code)
        return self.safe_transform(code, analysis)

    def set_mode(self, mode: PolicyMode):
        self.mode = mode


# ========================= SANDBOX ORGAN =========================

class SandboxOrgan:
    def __init__(self, logger: SecureLogger, anomaly: Optional[DeepAnomalyOrgan] = None):
        self.logger = logger
        self.anomaly = anomaly

    def safe_run(self, code: str):
        try:
            self.run(code)
        except Exception as e:
            self.logger.error(f"SandboxOrgan failure: {e}")

    def run(self, code: str):
        findings = []
        for line in code.splitlines():
            if line.strip().startswith("import "):
                findings.append(f"Dynamic import: {line.strip()} (category: CODE)")
            if "requests." in line or "http://" in line or "https://" in line:
                findings.append(f"Potential network call: {line.strip()} (category: NETWORK)")
        if self.anomaly is not None:
            self.anomaly.safe_update_from_findings(findings)
        if findings:
            self.logger.info(f"SandboxOrgan dynamic findings: {len(findings)}")


# ========================= NETWORK THREAT ORGAN (ACTIVE DEFENSE) =========================

class NetworkThreatOrgan:
    """
    Active network defense:
    - Uses psutil to inspect connections
    - Flags suspicious remote IPs/ports
    - Feeds findings into anomaly engine
    - Optionally kills processes / blocks connections (best-effort)
    """

    def __init__(self, logger: SecureLogger, anomaly: Optional[DeepAnomalyOrgan] = None):
        self.logger = logger
        self.anomaly = anomaly
        self.enabled = True
        self.active_defense = True
        self._lock = threading.Lock()
        self._last_snapshot: Dict[str, int] = {}
        self._suspicious_ports = {4444, 1337, 31337, 5555}
        self._high_port_threshold = 50000

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.logger.info(f"NetworkThreatOrgan enabled={enabled}")

    def set_active_defense(self, enabled: bool):
        self.active_defense = enabled
        self.logger.info(f"NetworkThreatOrgan active_defense={enabled}")

    def safe_scan(self):
        try:
            self.scan()
        except Exception as e:
            self.logger.error(f"NetworkThreatOrgan failure: {e}")

    def scan(self):
        if not self.enabled or not HAS_PSUTIL:
            return

        findings: List[str] = []
        suspicious_procs = set()

        try:
            conns = psutil.net_connections(kind="inet")
        except Exception as e:
            self.logger.error(f"psutil.net_connections failed: {e}")
            return

        for c in conns:
            laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "?:?"
            raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "?:?"
            status = c.status
            pid = c.pid

            if not c.raddr:
                continue

            rip = c.raddr.ip
            rport = c.raddr.port

            if rport in self._suspicious_ports or rport > self._high_port_threshold:
                findings.append(
                    f"Suspicious outbound connection to {rip}:{rport} (status={status}, category: NETWORK)"
                )
                if pid:
                    suspicious_procs.add(pid)

        if self.anomaly is not None and findings:
            self.anomaly.safe_update_from_findings(findings)

        if findings:
            self.logger.warn(f"NetworkThreatOrgan: {len(findings)} suspicious connections detected")

        if self.active_defense and suspicious_procs:
            for pid in suspicious_procs:
                try:
                    p = psutil.Process(pid)
                    name = p.name()
                    self.logger.warn(f"Active defense: terminating suspicious process PID={pid}, name={name}")
                    p.terminate()
                except Exception as e:
                    self.logger.error(f"Failed to terminate PID={pid}: {e}")


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

        self.network_summary = QLabel("Network: —")
        layout.addWidget(self.network_summary)

        self.momentum = MomentumArrow("→")
        layout.addWidget(self.momentum)

        self.led = PulsingLED(QColor(255, 0, 0))
        self.led.setFixedSize(24, 24)
        layout.addWidget(self.led)

    def update_panel(self, alias: str, pii_summary: dict, blocked: list,
                     momentum_value: int, network_summary: str):
        self.system_alias.setText(f"System Alias: {alias}")
        pii_text = ", ".join(f"{k}: {v}" for k, v in pii_summary.items() if v > 0)
        self.pii_counts.setText(f"PII Counts: {pii_text or 'none'}")
        if blocked:
            self.blocked_patterns.setText(
                "Recent Blocked Patterns:\n  - " + "\n  - ".join(blocked)
            )
        else:
            self.blocked_patterns.setText("Recent Blocked Patterns: none")
        self.network_summary.setText(f"Network: {network_summary}")
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
        color = QColor(0, 255, 0, alpha)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())


# -------- Tactical HUD Nuclear Dashboard --------

class RadarWidget(QOpenGLWidget if HAS_OPENGL else QWidget):
    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)
        self._blips: List[Tuple[float, float, float]] = []

    def sizeHint(self) -> QSize:
        return QSize(260, 260)

    def set_clusters(self, clusters: List[ClusterInfo]):
        blips: List[Tuple[float, float, float]] = []
        for c in clusters:
            r = min(1.0, max(0.1, c.density * 2.0))
            angle = (hash(c.fingerprint) % 360)
            intensity = min(1.0, 0.3 + c.volatility)
            blips.append((r, float(angle), float(intensity)))
        self._blips = blips
        self.update()

    def _tick(self):
        self.phase = (self.phase + 4.0) % 360.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        cx = rect.center().x()
        cy = rect.center().y()
        radius = min(rect.width(), rect.height()) / 2 - 10

        painter.fillRect(rect, QColor(0, 0, 0, 255))

        pen_grid = QPen(QColor(0, 120, 0))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        painter.setBrush(Qt.NoBrush)

        for r in [0.25, 0.5, 0.75, 1.0]:
            painter.drawEllipse(
                int(cx - radius * r),
                int(cy - radius * r),
                int(2 * radius * r),
                int(2 * radius * r),
            )

        for angle in range(0, 360, 30):
            rad = angle * math.pi / 180.0
            x2 = cx + radius * math.cos(rad)
            y2 = cy + radius * math.sin(rad)
            painter.drawLine(cx, cy, int(x2), int(y2))

        sweep_angle = self.phase
        rad = sweep_angle * math.pi / 180.0
        x2 = cx + radius * math.cos(rad)
        y2 = cy + radius * math.sin(rad)
        pen_sweep = QPen(QColor(0, 255, 0, 180))
        pen_sweep.setWidth(2)
        painter.setPen(pen_sweep)
        painter.drawLine(cx, cy, int(x2), int(y2))

        for r, angle_deg, intensity in self._blips:
            rad = angle_deg * math.pi / 180.0
            br = radius * r
            bx = cx + br * math.cos(rad)
            by = cy + br * math.sin(rad)
            alpha = int(80 + 175 * min(1.0, intensity))
            painter.setBrush(QColor(0, 255, 0, alpha))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(bx - 4), int(by - 4), 8, 8)


class DeepNuclearDashboardPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("Nuclear Dashboard – Tactical HUD")
        self.title.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout.addWidget(self.title)

        top = QHBoxLayout()
        layout.addLayout(top)

        self.radar = RadarWidget()
        self.radar.setMinimumHeight(220)
        top.addWidget(self.radar, stretch=2)

        right = QVBoxLayout()
        top.addLayout(right, stretch=3)

        self.grid = QGridLayout()
        right.addLayout(self.grid)

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

        self.cluster_label = QLabel("Clusters: —")
        right.addWidget(self.cluster_label)

        self.pred_label = QLabel("Predictive Horizon: —")
        right.addWidget(self.pred_label)

        self.ai_conf_label = QLabel("AI Confidence: —")
        layout.addWidget(self.ai_conf_label)

        self.ai_flags_label = QLabel("AI Flags: —")
        layout.addWidget(self.ai_flags_label)

        self.network_label = QLabel("Network Threats: —")
        layout.addWidget(self.network_label)

    def update_stats(self,
                     stats: Dict[str, Dict[str, Any]],
                     clusters: List[ClusterInfo],
                     predictions: Dict[str, Any],
                     ai_conf: float,
                     ai_flags: List[str],
                     network_summary: str):
        for cat, vals in stats.items():
            if cat in self.labels:
                self.labels[cat]["count"].setText(f"Count: {vals['count']}")
                self.labels[cat]["momentum"].setText(f"Momentum: {vals['momentum']:.0f}")
                self.labels[cat]["z"].setText(f"Z: {vals['z']:.2f}")
                self.labels[cat]["heat"].setText(f"Heat: {vals['heat']:.2f}")
                self.labels[cat]["anomaly"].setText(f"Anomaly: {vals['anomaly']}")

        if clusters:
            lines = []
            for c in clusters:
                cats = ",".join(c.categories) if c.categories else "—"
                lines.append(
                    f"Cluster {c.cluster_id}: size={c.size}, dens={c.density:.2f}, "
                    f"vol={c.volatility:.2f}, cats={cats}, fp={c.fingerprint}"
                )
            self.cluster_label.setText("Clusters:\n" + "\n".join(lines))
        else:
            self.cluster_label.setText("Clusters: none")

        self.radar.set_clusters(clusters)

        self.pred_label.setText(
            f"Predictive Horizon: short={predictions.get('short','?')}, "
            f"medium={predictions.get('medium','?')}, "
            f"long={predictions.get('long','?')}"
        )

        self.ai_conf_label.setText(f"AI Confidence: {ai_conf:.2f}")
        self.ai_flags_label.setText(f"AI Flags: {ai_flags or ['NONE']}")
        self.network_label.setText(f"Network Threats: {network_summary}")


# ========================= UIAUTOMATION ORGAN (ACTIVE DEFENSE) =========================

class UIAutomationOrgan:
    """
    Active UI defense (best-effort on Windows):
    - Enumerates windows
    - Detects suspicious titles
    - Optionally closes them
    """

    def __init__(self, logger: SecureLogger):
        self.logger = logger
        self.enabled = True
        self.active_defense = True
        self._suspicious_keywords = ["hacker", "ransom", "stealer", "keylogger"]

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.logger.info(f"UIAutomationOrgan enabled={enabled}")

    def set_active_defense(self, enabled: bool):
        self.active_defense = enabled
        self.logger.info(f"UIAutomationOrgan active_defense={enabled}")

    def safe_scan_active_windows(self):
        try:
            self.scan_active_windows()
        except Exception as e:
            self.logger.error(f"UIAutomationOrgan failure: {e}")

    def scan_active_windows(self):
        if not self.enabled:
            return
        if not HAS_WIN32:
            self.logger.info("UIAutomationOrgan: Win32 APIs not available; monitoring disabled.")
            return

        suspicious_handles = []

        def enum_handler(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            try:
                title = win32gui.GetWindowText(hwnd)
            except Exception:
                return
            if not title:
                return
            lower = title.lower()
            for kw in self._suspicious_keywords:
                if kw in lower:
                    suspicious_handles.append((hwnd, title))
                    break

        win32gui.EnumWindows(enum_handler, None)

        for hwnd, title in suspicious_handles:
            self.logger.warn(f"Suspicious window detected: '{title}' (HWND={hwnd})")
            if self.active_defense:
                try:
                    self.logger.warn(f"Active UI defense: closing window '{title}'")
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                except Exception as e:
                    self.logger.error(f"Failed to close window '{title}': {e}")

    def apply_protective_overlays(self):
        self.logger.info("UIAutomationOrgan: apply_protective_overlays() called (placeholder).")


# ========================= FILE WATCHER ORGAN (IMPROVED) =========================

class FileWatcherOrgan:
    def __init__(self, logger: SecureLogger, purifier: CodePurifier,
                 anomaly: DeepAnomalyOrgan, dataguard: DataGuard,
                 systemguard: SystemGuard, executor: ThreadPoolExecutor):
        self.logger = logger
        self.purifier = purifier
        self.anomaly = anomaly
        self.dataguard = dataguard
        self.systemguard = systemguard
        self.executor = executor

        self.watch_dir: Path = Path.cwd()
        self.enabled: bool = False
        self._seen_hashes: Dict[Path, str] = {}

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(3000)

    def set_directory(self, path: Path):
        self.watch_dir = path
        self._seen_hashes.clear()
        self.logger.info(f"FileWatcher: directory set to {path}")

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.logger.info(f"FileWatcher: enabled={enabled}")

    def _tick(self):
        if not self.enabled:
            return
        if not self.watch_dir.exists() or not self.watch_dir.is_dir():
            return

        for entry in self._iter_py_files(self.watch_dir):
            try:
                h = self._hash_file(entry)
            except Exception as e:
                self.logger.error(f"FileWatcher: failed to hash {entry}: {e}")
                continue
            last = self._seen_hashes.get(entry)
            if last is None or last != h:
                self._seen_hashes[entry] = h
                self.executor.submit(self._process_file_safe, entry)

    def _iter_py_files(self, root: Path):
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".py") and not name.endswith(".hardened.py"):
                    yield Path(dirpath) / name

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _process_file_safe(self, path: Path):
        try:
            self._process_file(path)
        except Exception as e:
            self.logger.error(f"FileWatcher: processing failure for {path}: {e}")

    def _process_file(self, path: Path):
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            self.logger.error(f"FileWatcher: failed to read {path}: {e}")
            return

        self.logger.info(f"FileWatcher: processing {path.name} in Nuclear Mode")

        old_mode = self.purifier.mode
        self.purifier.set_mode(PolicyMode.NUCLEAR)
        result = self.purifier.purify(code)
        self.purifier.set_mode(old_mode)

        hardened_path = path.with_suffix(".hardened.py")
        try:
            hardened_path.write_text(result.hardened_code, encoding="utf-8")
            self.logger.info(f"FileWatcher: wrote hardened file {hardened_path.name}")
        except Exception as e:
            self.logger.error(f"FileWatcher: failed to write {hardened_path}: {e}")

        safe_policy = self.dataguard.protect_text(result.policy_summary)
        safe_policy = self.systemguard.scrub_text(safe_policy)
        self.logger.info(f"FileWatcher Policy for {path.name}: {safe_policy.splitlines()[0]}")


# ========================= QT EVENT HELPER =========================

class _CallableEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, fn):
        super().__init__(self.EVENT_TYPE)
        self.fn = fn

    def invoke(self):
        self.fn()


# Patch QApplication to dispatch our callable events
class PatchedApplication(QApplication):
    def notify(self, receiver, event):
        if isinstance(event, _CallableEvent):
            event.invoke()
            return True
        return super().notify(receiver, event)


# ========================= COCKPIT =========================

class Cockpit(QWidget):
    def __init__(self):
        super().__init__()

        self.executor = ThreadPoolExecutor(max_workers=8)

        self.dataguard = DataGuard(secret_salt="your-secret-here")
        self.systemguard = SystemGuard(secret_salt="your-secret-here")
        self.memory = EncryptedMemoryStore(path=Path("memory/purifier_memory.enc"),
                                           secret_salt="your-secret-here")
        self.log_viewer = LogViewer()
        self.logger = SecureLogger(self.dataguard, self.systemguard, sink=self.log_viewer)

        self.anomaly = DeepAnomalyOrgan(logger=self.logger)
        self.engine = CodePurifier(memory=self.memory, mode=PolicyMode.MONITOR,
                                   anomaly=self.anomaly, logger=self.logger)
        self.ai_engine = AIEngine(logger=self.logger, memory=self.memory,
                                  backend=AIBackend.RULE_BASED)
        self.ui_auto = UIAutomationOrgan(logger=self.logger)
        self.sandbox = SandboxOrgan(logger=self.logger, anomaly=self.anomaly)
        self.network_organ = NetworkThreatOrgan(logger=self.logger, anomaly=self.anomaly)

        self.file_watcher = FileWatcherOrgan(
            logger=self.logger,
            purifier=self.engine,
            anomaly=self.anomaly,
            dataguard=self.dataguard,
            systemguard=self.systemguard,
            executor=self.executor,
        )

        self.theme_manager = ThemeManager()
        self.last_threat_count = 0
        self._last_lineage: List[Tuple[str, str, str]] = []
        self._last_ai_lineage: List[Tuple[str, str]] = []
        self._last_ai_confidence: float = 0.0
        self._last_ai_flags: List[str] = []
        self._last_network_summary: str = "—"

        self.sound_threat = None
        self.sound_nuclear = None
        self.sound_pii = None

        self._last_raw_code: str = ""
        self._last_auto_time: float = 0.0
        self._auto_debounce_ms: int = 800

        self.setWindowTitle("Defensive Code Purifier – Autonomous Nuclear Cockpit (Network + UI Defense)")
        self.resize(1900, 1000)

        self._build_ui()
        self.theme_manager.apply(self, "dark")

        self._auto_timer = QTimer()
        self._auto_timer.timeout.connect(self._auto_tick)
        self._auto_timer.start(500)

        self._network_timer = QTimer()
        self._network_timer.timeout.connect(self._network_tick)
        self._network_timer.start(4000)

        self._ui_auto_timer = QTimer()
        self._ui_auto_timer.timeout.connect(self._ui_auto_tick)
        self._ui_auto_timer.start(5000)

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

        right_panel = QVBoxLayout()

        self.hardened_edit = QTextEdit()
        self.hardened_edit.setReadOnly(True)
        self.hardened_edit.setPlaceholderText("Purifier hardened output...")
        right_panel.addWidget(self.hardened_edit, stretch=1)

        self.ai_output_edit = QTextEdit()
        self.ai_output_edit.setReadOnly(True)
        self.ai_output_edit.setPlaceholderText("AI rewritten output...")
        right_panel.addWidget(self.ai_output_edit, stretch=1)

        mid.addLayout(right_panel, stretch=3)

        self.threat_panel = ThreatPanel()
        mid.addWidget(self.threat_panel, stretch=1)

        code_layout.addLayout(mid)

        bottom = QHBoxLayout()
        self.status_label = QLabel("Risk: —")
        self.status_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        bottom.addWidget(self.status_label, stretch=1)

        btn_purify = QPushButton("Purify (Manual)")
        btn_purify.clicked.connect(self.on_purify)
        bottom.addWidget(btn_purify)

        btn_lineage = QPushButton("Lineage Tree")
        btn_lineage.clicked.connect(self.show_lineage)
        bottom.addWidget(btn_lineage)

        btn_ai_rewrite = QPushButton("AI Rewrite (Manual)")
        btn_ai_rewrite.clicked.connect(self.on_ai_rewrite)
        bottom.addWidget(btn_ai_rewrite)

        btn_ai_lineage = QPushButton("AI Lineage")
        btn_ai_lineage.clicked.connect(self.show_ai_lineage)
        bottom.addWidget(btn_ai_lineage)

        btn_sandbox = QPushButton("Run Sandbox (Static/Dynamic Scan)")
        btn_sandbox.clicked.connect(self.on_sandbox_run)
        bottom.addWidget(btn_sandbox)

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

        ai_layout.addWidget(QLabel("AI Backend (manual override):"))
        self.ai_backend_combo = QComboBox()
        self.ai_backend_combo.addItems([b.name for b in AIBackend])
        self.ai_backend_combo.setCurrentText(self.ai_engine.backend.name)
        self.ai_backend_combo.currentTextChanged.connect(self.on_set_ai_backend)
        ai_layout.addWidget(self.ai_backend_combo)

        self.ai_auto_check = QCheckBox("Auto-select best backend")
        self.ai_auto_check.setChecked(True)
        self.ai_auto_check.stateChanged.connect(self.on_toggle_ai_auto)
        ai_layout.addWidget(self.ai_auto_check)

        ai_layout.addWidget(QLabel("Local HTTP URL (OpenAI-style):"))
        self.ai_local_http_url_edit = QLineEdit("http://localhost:1234/v1/chat/completions")
        ai_layout.addWidget(self.ai_local_http_url_edit)

        ai_layout.addWidget(QLabel("Local HTTP Model:"))
        self.ai_local_http_model_edit = QLineEdit("your-local-model")
        ai_layout.addWidget(self.ai_local_http_model_edit)

        btn_set_local_http = QPushButton("Set Local HTTP Backend")
        btn_set_local_http.clicked.connect(self.on_set_local_http)
        ai_layout.addWidget(btn_set_local_http)

        ai_layout.addWidget(QLabel("Remote Base URL (OpenAI-style):"))
        self.ai_remote_base_url_edit = QLineEdit("https://api.openai.com/v1")
        ai_layout.addWidget(self.ai_remote_base_url_edit)

        ai_layout.addWidget(QLabel("Remote API Key:"))
        self.ai_remote_api_key_edit = QLineEdit("")
        ai_layout.addWidget(self.ai_remote_api_key_edit)

        ai_layout.addWidget(QLabel("Remote Model:"))
        self.ai_remote_model_edit = QLineEdit("gpt-4.1-mini")
        ai_layout.addWidget(self.ai_remote_model_edit)

        btn_set_remote_openai = QPushButton("Set Remote OpenAI Backend")
        btn_set_remote_openai.clicked.connect(self.on_set_remote_openai)
        ai_layout.addWidget(btn_set_remote_openai)

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
        mem_layout.addWidget(QLabel("Memory Path (encrypted):"))
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

        self.nuclear_dashboard = DeepNuclearDashboardPanel()
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

        auto_layout.addWidget(QLabel("UI Automation Organ (Active Defense)"))
        self.ui_auto_enable_check = QCheckBox("Enable UI Automation")
        self.ui_auto_enable_check.setChecked(True)
        self.ui_auto_enable_check.stateChanged.connect(self.on_toggle_ui_auto)
        auto_layout.addWidget(self.ui_auto_enable_check)

        self.ui_auto_active_check = QCheckBox("Active UI Defense (auto-close suspicious windows)")
        self.ui_auto_active_check.setChecked(True)
        self.ui_auto_active_check.stateChanged.connect(self.on_toggle_ui_auto_active)
        auto_layout.addWidget(self.ui_auto_active_check)

        btn_scan_windows = QPushButton("Scan Active Windows Now")
        btn_scan_windows.clicked.connect(self.on_scan_windows)
        auto_layout.addWidget(btn_scan_windows)

        btn_apply_overlays = QPushButton("Apply Protective Overlays (placeholder)")
        btn_apply_overlays.clicked.connect(self.on_apply_overlays)
        auto_layout.addWidget(btn_apply_overlays)

        auto_layout.addWidget(QLabel("File Watcher Organ (auto Nuclear Mode)"))
        self.watch_dir_edit = QLineEdit(str(Path.cwd()))
        auto_layout.addWidget(self.watch_dir_edit)

        btn_browse_watch = QPushButton("Browse Watch Directory...")
        btn_browse_watch.clicked.connect(self.on_browse_watch_dir)
        auto_layout.addWidget(btn_browse_watch)

        self.watch_enable_check = QCheckBox("Enable File Watcher")
        self.watch_enable_check.stateChanged.connect(self.on_toggle_file_watcher)
        auto_layout.addWidget(self.watch_enable_check)

        auto_layout.addWidget(QLabel("Network Threat Organ (Active Defense)"))
        self.net_enable_check = QCheckBox("Enable Network Monitoring")
        self.net_enable_check.setChecked(True)
        self.net_enable_check.stateChanged.connect(self.on_toggle_network)
        auto_layout.addWidget(self.net_enable_check)

        self.net_active_check = QCheckBox("Active Network Defense (terminate suspicious processes)")
        self.net_active_check.setChecked(True)
        self.net_active_check.stateChanged.connect(self.on_toggle_network_active)
        auto_layout.addWidget(self.net_active_check)

        self.stack.addWidget(auto_page)

        nav.setCurrentRow(0)

    def eventFilter(self, obj, event):
        if obj is self.stack and event.type() == QEvent.Resize:
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
            self.overlay.set_heat(0.3)
        else:
            self.theme_manager.apply(self, "dark")
            self.status_label.setText(f"Risk: Mode set to {mode.name}")
            self.log_viewer.add_log(f"[INFO] Policy mode set to {mode.name}")
            self.overlay.set_heat(0.0)

    def _auto_tick(self):
        code = self.raw_edit.toPlainText()
        if not code.strip():
            return

        now = time.time() * 1000
        if now - self._last_auto_time < self._auto_debounce_ms:
            return
        self._last_auto_time = now

        if code == self._last_raw_code:
            return
        self._last_raw_code = code

        self.executor.submit(self._auto_process_code_safe, code)

    def _auto_process_code_safe(self, code: str):
        try:
            self._auto_process_code(code)
        except Exception as e:
            self.logger.error(f"Auto process failure: {e}")

    def _auto_process_code(self, code: str):
        pii_summary = self.dataguard.scan_summary(code)
        if any(v > 0 for v in pii_summary.values()):
            self.play_sound(self.sound_pii)

        result = self.engine.purify(code)
        self._last_lineage = result.lineage

        safe_policy = self.dataguard.protect_text(result.policy_summary)
        safe_policy = self.systemguard.scrub_text(safe_policy)

        blocked = result.policy_summary.splitlines()[2:]
        alias = self.systemguard.identity.alias

        current_threat_count = len(blocked)
        momentum_value = current_threat_count - self.last_threat_count
        self.last_threat_count = current_threat_count

        stats = self.anomaly.current_stats()
        clusters = self.anomaly.current_clusters()
        predictions = self.anomaly.predictive_horizon()
        global_heat = max(v["heat"] for v in stats.values()) if stats else 0.0
        if self.engine.mode == PolicyMode.NUCLEAR:
            global_heat = max(global_heat, 0.3)

        ai_result: Optional[AIRewriteResult] = None
        if "Risk: SAFE" not in result.policy_summary:
            ai_result = self.ai_engine.rewrite_defensive(code)
            self._last_ai_lineage = ai_result.lineage
            self._last_ai_confidence = ai_result.confidence
            self._last_ai_flags = ai_result.safety_flags
            self.logger.info(
                f"AI-AUTO rewrite; confidence={ai_result.confidence:.2f}, flags={ai_result.safety_flags}"
            )

        network_summary = self._last_network_summary

        def update_ui():
            self.hardened_edit.setPlainText(result.hardened_code)
            if ai_result is not None:
                self.ai_output_edit.setPlainText(ai_result.rewritten)
                self.ai_safety_label.setText(f"AI Safety Flags: {ai_result.safety_flags or ['NONE']}")
                self.ai_confidence_label.setText(f"AI Confidence: {ai_result.confidence:.2f}")
            else:
                self.ai_output_edit.setPlainText("")
                self.ai_safety_label.setText("AI Safety: (not triggered)")
                self.ai_confidence_label.setText("AI Confidence: —")

            self.status_label.setText(f"Risk: {safe_policy.splitlines()[0]}")
            self.threat_panel.update_panel(alias, pii_summary, blocked, momentum_value, network_summary)
            self.threat_detail_label.setText(safe_policy)
            self.lineage_label.setText(
                "\n".join(f"{cat} | {action} | {detail}" for cat, action, detail in result.lineage)
            )

            if blocked:
                self.play_sound(self.sound_threat)

            self.overlay.set_heat(global_heat)
            self.nuclear_dashboard.update_stats(
                stats, clusters, predictions, self._last_ai_confidence, self._last_ai_flags, network_summary
            )

        QApplication.instance().postEvent(self, _CallableEvent(update_ui))

    def on_purify(self):
        code = self.raw_edit.toPlainText().strip()
        if not code:
            return
        self.executor.submit(self._manual_purify_safe, code)

    def _manual_purify_safe(self, code: str):
        try:
            self._manual_purify(code)
        except Exception as e:
            self.logger.error(f"Manual purify failure: {e}")

    def _manual_purify(self, code: str):
        pii_summary = self.dataguard.scan_summary(code)
        if any(v > 0 for v in pii_summary.values()):
            self.play_sound(self.sound_pii)

        result = self.engine.purify(code)
        self._last_lineage = result.lineage

        safe_policy = self.dataguard.protect_text(result.policy_summary)
        safe_policy = self.systemguard.scrub_text(safe_policy)

        blocked = result.policy_summary.splitlines()[2:]
        alias = self.systemguard.identity.alias

        current_threat_count = len(blocked)
        momentum_value = current_threat_count - self.last_threat_count
        self.last_threat_count = current_threat_count

        stats = self.anomaly.current_stats()
        clusters = self.anomaly.current_clusters()
        predictions = self.anomaly.predictive_horizon()
        global_heat = max(v["heat"] for v in stats.values()) if stats else 0.0
        if self.engine.mode == PolicyMode.NUCLEAR:
            global_heat = max(global_heat, 0.3)

        network_summary = self._last_network_summary

        def update_ui():
            self.hardened_edit.setPlainText(result.hardened_code)
            self.status_label.setText(f"Risk: {safe_policy.splitlines()[0]}")
            self.threat_panel.update_panel(alias, pii_summary, blocked, momentum_value, network_summary)
            self.threat_detail_label.setText(safe_policy)
            self.lineage_label.setText(
                "\n".join(f"{cat} | {action} | {detail}" for cat, action, detail in result.lineage)
            )
            if blocked:
                self.play_sound(self.sound_threat)
            self.overlay.set_heat(global_heat)
            self.nuclear_dashboard.update_stats(
                stats, clusters, predictions, self._last_ai_confidence, self._last_ai_flags, network_summary
            )

        QApplication.instance().postEvent(self, _CallableEvent(update_ui))

    def on_ai_rewrite(self):
        code = self.raw_edit.toPlainText().strip()
        if not code:
            return
        self.executor.submit(self._manual_ai_rewrite_safe, code)

    def _manual_ai_rewrite_safe(self, code: str):
        try:
            self._manual_ai_rewrite(code)
        except Exception as e:
            self.logger.error(f"Manual AI rewrite failure: {e}")

    def _manual_ai_rewrite(self, code: str):
        ai_result = self.ai_engine.rewrite_defensive(code)
        self._last_ai_lineage = ai_result.lineage
        self._last_ai_confidence = ai_result.confidence
        self._last_ai_flags = ai_result.safety_flags

        stats = self.anomaly.current_stats()
        clusters = self.anomaly.current_clusters()
        predictions = self.anomaly.predictive_horizon()
        global_heat = max(v["heat"] for v in stats.values()) if stats else 0.0
        if self.engine.mode == PolicyMode.NUCLEAR:
            global_heat = max(global_heat, 0.3)

        network_summary = self._last_network_summary

        def update_ui():
            self.ai_output_edit.setPlainText(ai_result.rewritten)
            self.ai_safety_label.setText(f"AI Safety Flags: {ai_result.safety_flags or ['NONE']}")
            self.ai_confidence_label.setText(f"AI Confidence: {ai_result.confidence:.2f}")
            self.log_viewer.add_log(
                f"[AI] Rewrite applied; confidence={ai_result.confidence:.2f}, flags={ai_result.safety_flags}"
            )
            self.overlay.set_heat(global_heat)
            self.nuclear_dashboard.update_stats(
                stats, clusters, predictions, self._last_ai_confidence, self._last_ai_flags, network_summary
            )

        QApplication.instance().postEvent(self, _CallableEvent(update_ui))

    def on_set_ai_backend(self, name: str):
        try:
            backend = AIBackend[name]
            self.ai_engine.set_backend(backend)
            self.ai_status.setText(self.ai_engine.describe())
        except KeyError:
            self.log_viewer.add_log(f"[ERROR] Invalid AI backend: {name}")

    def on_toggle_ai_auto(self, state: int):
        enabled = state == Qt.Checked
        self.ai_engine.set_auto_mode(enabled)
        self.ai_status.setText(self.ai_engine.describe())

    def on_set_local_http(self):
        url = self.ai_local_http_url_edit.text().strip()
        model = self.ai_local_http_model_edit.text().strip()
        if not url or not model:
            return
        self.ai_engine.set_local_http(url, model)
        self.ai_status.setText(self.ai_engine.describe())

    def on_set_remote_openai(self):
        base_url = self.ai_remote_base_url_edit.text().strip()
        api_key = self.ai_remote_api_key_edit.text().strip()
        model = self.ai_remote_model_edit.text().strip()
        if not base_url or not api_key or not model:
            return
        self.ai_engine.set_remote_openai(base_url, api_key, model)
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
        self.memory = EncryptedMemoryStore(path=new_path, secret_salt="your-secret-here")
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
        self.ui_auto.safe_scan_active_windows()

    def on_apply_overlays(self):
        self.ui_auto.apply_protective_overlays()

    def on_browse_watch_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Watch Directory", self.watch_dir_edit.text())
        if path:
            self.watch_dir_edit.setText(path)
            self.file_watcher.set_directory(Path(path))

    def on_toggle_file_watcher(self, state: int):
        enabled = state == Qt.Checked
        self.file_watcher.set_enabled(enabled)
        self.log_viewer.add_log(f"[WATCHER] Enabled={enabled}, dir={self.watch_dir_edit.text()}")

    def on_toggle_network(self, state: int):
        enabled = state == Qt.Checked
        self.network_organ.set_enabled(enabled)

    def on_toggle_network_active(self, state: int):
        enabled = state == Qt.Checked
        self.network_organ.set_active_defense(enabled)

    def on_toggle_ui_auto(self, state: int):
        enabled = state == Qt.Checked
        self.ui_auto.set_enabled(enabled)

    def on_toggle_ui_auto_active(self, state: int):
        enabled = state == Qt.Checked
        self.ui_auto.set_active_defense(enabled)

    def on_sandbox_run(self):
        code = self.raw_edit.toPlainText().strip()
        if not code:
            return
        self.executor.submit(self.sandbox.safe_run, code)

    def _network_tick(self):
        if not HAS_PSUTIL:
            return
        self.network_organ.safe_scan()
        # Build a short summary for UI
        try:
            conns = psutil.net_connections(kind="inet")
            suspicious = 0
            for c in conns:
                if not c.raddr:
                    continue
                rport = c.raddr.port
                if rport in self.network_organ._suspicious_ports or rport > self.network_organ._high_port_threshold:
                    suspicious += 1
            self._last_network_summary = f"{suspicious} suspicious connections" if suspicious else "clean"
        except Exception:
            self._last_network_summary = "error"

    def _ui_auto_tick(self):
        self.ui_auto.safe_scan_active_windows()

    def event(self, e):
        if isinstance(e, _CallableEvent):
            e.invoke()
            return True
        return super().event(e)


if __name__ == "__main__":
    # R-2: software OpenGL emulation hint
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
    app = PatchedApplication(sys.argv)
    cockpit = Cockpit()
    cockpit.show()
    sys.exit(app.exec_())

