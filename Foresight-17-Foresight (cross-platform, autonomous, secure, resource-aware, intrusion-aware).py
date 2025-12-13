#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foresight: anticipatory assistant with hierarchical routines, ensemble confidence, branching forecasts,
adaptive anomaly detection, context fusion, calibrated confidence, encrypted persistence, intrusion detection,
and safe auto-play — compact cross-platform Tk GUI.

Enhancements in this version:
- Hierarchical routines: discovers macro-routines and previews them from a trigger.
- Ensemble confidence: blends base predictor with workflow votes and context urgency.
- Branching forecasts: primary path + clean alternative path without mutating model state.
- Adaptive anomaly detection: rolling baselines per hour/day of week.
- Context fusion: optional external signals (deadlines, meetings, bursts).
- Feedback weighting: stronger adaptation on accept/skip for faster personalization.
- Semantic tags: work/play/comms/system bias by time/context.
- Routine preview in UI: quick look at the likely macro-routine when a trigger appears.
"""

import sys
import os
import json
import time
import threading
import datetime
import importlib
import subprocess
import base64
import locale
import uuid
from collections import defaultdict, deque, Counter
from typing import Dict, List, Tuple, Optional, Set

# ----------------------------
# Auto-loader
# ----------------------------
def auto_loader(libraries: List[str]) -> None:
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception as e:
                print(f"[foresight] Failed to install {lib}: {e}")

auto_loader(["matplotlib", "psutil", "cryptography"])

# Tkinter baseline
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
except Exception as e:
    print("[foresight] Tkinter not available.")
    print(e)
    sys.exit(1)

# Matplotlib (optional visual)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    plt = None
    FigureCanvasTkAgg = None

# System libraries
import psutil

# Cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# ----------------------------
# Cross-platform foreground window title (best-effort)
# ----------------------------
def get_foreground_window_title() -> Optional[str]:
    plat = sys.platform
    if plat.startswith("win"):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            hwnd = user32.GetForegroundWindow()
            if hwnd == 0:
                return None
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            return title if title else None
        except Exception:
            return None
    elif plat == "darwin":
        try:
            from AppKit import NSWorkspace
            active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            name = active_app.localizedName() if active_app else None
            return name.strip() if name else None
        except Exception:
            return None
    elif plat.startswith("linux"):
        try:
            import subprocess as sp
            out = sp.check_output(["wmctrl", "-lx"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
            lines = [l for l in out.splitlines() if l.strip()]
            if lines:
                parts = lines[-1].split()
                title = parts[-1] if parts else ""
                return title.strip() or None
        except Exception:
            pass
        try:
            import subprocess as sp
            out = sp.check_output(["xprop", "-root", "_NET_ACTIVE_WINDOW"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
            parts = out.split()
            wid = parts[-1]
            if wid and wid != "0x0":
                out2 = sp.check_output(["xprop", "-id", wid, "WM_NAME"], stderr=sp.DEVNULL).decode("utf-8", errors="ignore")
                if "=" in out2:
                    title = out2.split("=", 1)[1].strip().strip('"')
                    return title if title else None
        except Exception:
            return None
    return None

# ----------------------------
# SecurityGuardian: encryption + reputation + suspicion scoring
# ----------------------------
class SecurityGuardian:
    def __init__(self, salt_file: str = "foresight.salt"):
        self.salt_file = salt_file
        self.salt = self._load_or_create_salt()
        self.key = self._derive_key()
        self.fernet = Fernet(self.key)
        self.bad_markers = [
            "login", "verify", "password", "account locked", "free", "prize", "claim",
            "gift", "crypto", "investment", "urgent", "warning", "lottery", "sweepstakes"
        ]
        self.whitelist: Set[str] = set()
        self.watchlist: Set[str] = set()
        self.safe_daily: Dict[str, int] = defaultdict(int)

    def _load_or_create_salt(self) -> bytes:
        if os.path.exists(self.salt_file):
            try:
                with open(self.salt_file, "rb") as f:
                    return f.read()
            except Exception:
                pass
        salt = os.urandom(16)
        try:
            with open(self.salt_file, "wb") as f:
                f.write(salt)
        except Exception:
            pass
        return salt

    def _screen_fingerprint(self) -> str:
        try:
            root = tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return f"{w}x{h}"
        except Exception:
            return "unknown_screen"

    def _derive_key(self) -> bytes:
        try:
            hostname = os.uname().nodename
        except Exception:
            hostname = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown_host"
        mac = uuid.getnode()
        tz = time.tzname[0] if time.tzname else "unknown_tz"
        try:
            loc = locale.getdefaultlocale()[0] or "unknown_locale"
        except Exception:
            loc = "unknown_locale"
        screen = self._screen_fingerprint()
        fingerprint = f"{hostname}|{mac}|{tz}|{loc}|{screen}"
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=self.salt,
                         iterations=100_000, backend=default_backend())
        return base64.urlsafe_b64encode(kdf.derive(fingerprint.encode("utf-8")))

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data)

    def score_title_suspicion(self, title: str, hour: int, daily_count: int) -> float:
        t = (title or "").lower()
        score = 0.0
        for m in self.bad_markers:
            if m in t:
                score += 0.4
        if hour in (1, 2, 3, 4, 5):
            score += 0.2
        if daily_count == 0 and any(m in t for m in self.bad_markers):
            score += 0.3
        return min(score, 1.0)

    def score_process_suspicion(self, name: str, cpu: float, mem_mb: float,
                                hour: int, seen_before: bool, parent_name: Optional[str],
                                net_connections: int, baseline_net: float) -> float:
        lname = (name or "").lower()
        score = 0.0
        if not seen_before and lname not in {x.lower() for x in self.whitelist}:
            score += 0.35
        if cpu >= 25.0 or mem_mb >= 800.0:
            score += 0.35
        if hour in (1, 2, 3, 4, 5):
            score += 0.15
        suspicious_parents = {"powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe", "bash", "zsh", "sh", "osascript"}
        if parent_name and parent_name.lower() in suspicious_parents:
            score += 0.25
        if lname.endswith(".scr") or lname.endswith(".js") or lname.endswith(".vbs"):
            score += 0.25
        if baseline_net >= 0 and net_connections > baseline_net * 3 + 2:
            score += 0.3
        return min(score, 1.0)

# ----------------------------
# Data structures
# ----------------------------
class HabitStats:
    __slots__ = ("count", "last_ts", "tod_hist")
    def __init__(self, count: int = 0, last_ts: float = 0.0, tod_hist: Optional[Dict[int, int]] = None):
        self.count = count
        self.last_ts = last_ts
        self.tod_hist = tod_hist if tod_hist is not None else {h: 0 for h in range(24)}
    def to_dict(self) -> Dict:
        return {"count": self.count, "last_ts": self.last_ts, "tod_hist": self.tod_hist}
    @staticmethod
    def from_dict(d: Dict) -> "HabitStats":
        return HabitStats(
            count=int(d.get("count", 0)),
            last_ts=float(d.get("last_ts", 0.0)),
            tod_hist={int(k): int(v) for k, v in d.get("tod_hist", {h: 0 for h in range(24)}).items()}
        )

# ----------------------------
# External context fusion
# ----------------------------
class ContextManager:
    def __init__(self, path: str = "foresight_context.json"):
        self.path = path
        self.data: Dict = {}
        self.last_load = 0.0
        self.reload_interval = 30.0

    def load(self) -> None:
        try:
            if (time.time() - self.last_load) < self.reload_interval and self.data:
                return
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
            self.last_load = time.time()
        except Exception:
            self.data = {}

    def features(self) -> Dict[str, float]:
        self.load()
        deadline_min = float(self.data.get("deadline_minutes", 0.0))
        meeting_now = bool(self.data.get("meeting_now", False))
        network_burst = float(self.data.get("network_burst", 0.0))  # 0..1
        deadline_urgency = max(0.0, min(1.0, 1.0 - (deadline_min / 240.0))) if deadline_min > 0 else 0.0
        return {
            "deadline_urgency": deadline_urgency,
            "meeting_now": 1.0 if meeting_now else 0.0,
            "network_burst": max(0.0, min(1.0, network_burst)),
        }

# ----------------------------
# Confidence calibration
# ----------------------------
class ConfidenceCalibrator:
    def __init__(self, bins: int = 10, history_max: int = 3000):
        self.bins = bins
        self.bin_correct = [0] * bins
        self.bin_total = [0] * bins
        self.history = deque(maxlen=history_max)  # (raw_conf, correct)
        self.alpha = 0.6        # blend raw and reliability
        self.global_beta = [1.0, 1.0]  # Beta(a,b)

    def _bin_index(self, conf: float) -> int:
        conf = max(0.0, min(1.0, conf))
        idx = int(conf * self.bins)
        return min(self.bins - 1, idx)

    def update(self, raw_conf: float, correct: bool) -> None:
        idx = self._bin_index(raw_conf)
        self.bin_total[idx] += 1
        if correct:
            self.bin_correct[idx] += 1
            self.global_beta[0] += 1.0
        else:
            self.global_beta[1] += 1.0
        self.history.append((raw_conf, 1 if correct else 0))

    def reliability(self, raw_conf: float) -> float:
        idx = self._bin_index(raw_conf)
        total = self.bin_total[idx]
        a, b = self.global_beta
        global_rel = a / (a + b)
        if total == 0:
            return global_rel
        return (self.bin_correct[idx] + a * 0.1) / (total + (a + b) * 0.1)

    def calibrate(self, raw_conf: float) -> float:
        r = self.reliability(raw_conf)
        cal = self.alpha * raw_conf + (1 - self.alpha) * r
        return max(0.0, min(1.0, cal))

    def to_dict(self) -> Dict:
        return {
            "bins": self.bins,
            "bin_correct": self.bin_correct,
            "bin_total": self.bin_total,
            "alpha": self.alpha,
            "global_beta": self.global_beta
        }

    def from_dict(self, d: Dict) -> None:
        self.bins = int(d.get("bins", self.bins))
        self.bin_correct = list(d.get("bin_correct", self.bin_correct))
        self.bin_total = list(d.get("bin_total", self.bin_total))
        self.alpha = float(d.get("alpha", self.alpha))
        gb = d.get("global_beta", self.global_beta)
        if isinstance(gb, list) and len(gb) == 2:
            self.global_beta = [float(gb[0]), float(gb[1])]

# ----------------------------
# Workflow chains and hierarchical routines
# ----------------------------
class WorkflowChains:
    def __init__(self, min_len: int = 3, max_len: int = 8, top_k: int = 100):
        self.min_len = min_len
        self.max_len = max_len
        self.top_k = top_k
        self.chain_counts: Dict[Tuple[str, ...], int] = Counter()
        self.hierarchy: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
        self.last_build_ts = 0.0

    def ingest_log(self, events: List[Tuple[float, str]]) -> None:
        seq = [h for ts, h in events if isinstance(h, str) and not h.startswith("advisory:")]
        n = len(seq)
        cc = Counter()
        for L in range(self.min_len, self.max_len + 1):
            for i in range(0, max(0, n - L + 1)):
                window = tuple(seq[i:i+L])
                # lightweight namespace filter to reduce noise
                if not (window[0].startswith(("web:", "work:", "proc:", "app:"))):
                    continue
                cc[window] += 1
        self.chain_counts = Counter(dict(cc.most_common(self.top_k)))
        self._build_hierarchy()
        self.last_build_ts = time.time()

    def _build_hierarchy(self) -> None:
        hier = defaultdict(list)
        for chain, _ in self.chain_counts.items():
            trigger = chain[0]
            hier[trigger].append(chain)
        for k in hier:
            hier[k].sort(key=lambda c: (len(c), self.chain_counts.get(c, 0)), reverse=True)
        self.hierarchy = hier

    def next_in_chain(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str]) -> Optional[Tuple[str, float]]:
        if not prev1:
            return None
        tails = [
            (3, (prev3, prev2, prev1)),
            (2, (prev2, prev1)),
            (1, (prev1,))
        ]
        best = None
        best_score = 0.0
        for L, tail in tails:
            if any(t is None for t in tail):
                continue
            for chain, cnt in self.chain_counts.items():
                if len(chain) <= L:
                    continue
                if tuple(chain[:L]) == tail:
                    candidate = chain[L]
                    score = cnt * (1.0 + 0.25 * (L-1)) * (1.0 + 0.06 * len(chain))
                    if score > best_score:
                        best = candidate
                        best_score = score
        if best:
            max_cnt = max(self.chain_counts.values()) if self.chain_counts else 1
            conf_like = min(1.0, 0.6 + 0.4 * (best_score / (max_cnt + 1e-9)))
            return best, conf_like
        return None

    def routine_for_trigger(self, trigger: str) -> Optional[List[str]]:
        chains = self.hierarchy.get(trigger) or []
        return list(chains[0]) if chains else None

    def to_dict(self) -> Dict:
        return {
            "min_len": self.min_len,
            "max_len": self.max_len,
            "top_k": self.top_k,
            "chains": { "|".join(k): int(v) for k, v in self.chain_counts.items() },
            "last_build_ts": self.last_build_ts
        }

    def from_dict(self, d: Dict) -> None:
        self.min_len = int(d.get("min_len", self.min_len))
        self.max_len = int(d.get("max_len", self.max_len))
        self.top_k = int(d.get("top_k", self.top_k))
        chains = d.get("chains", {})
        cc = Counter()
        for k, v in chains.items():
            try:
                tup = tuple(k.split("|"))
                cc[tup] = int(v)
            except Exception:
                continue
        self.chain_counts = cc
        self._build_hierarchy()
        self.last_build_ts = float(d.get("last_build_ts", 0.0))

# ----------------------------
# Next-gen predictor with adaptive anomaly and semantic tags
# ----------------------------
class NextGenPredictor:
    def __init__(self):
        self.transitions1: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.transitions2: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.transitions3: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self.context_bias: Dict[Tuple[int, int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.season_month_bias: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # (month, dow)
        self.season_wom_bias: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))    # (week_of_month, dow)

        self.cluster_map = {
            "browser": ["chrome", "edge", "firefox", "brave", "opera", "safari"],
            "ide": ["code", "vscode", "pycharm", "idea", "sublime"],
            "office": ["word", "excel", "powerpoint", "outlook"],
            "terminal": ["powershell", "cmd", "bash", "zsh", "terminal"],
            "comms": ["teams", "slack", "discord", "zoom", "meet"],
            "play": ["steam", "epic", "game"],
            "system": ["settings", "taskmgr", "monitor", "process"]
        }

        self.tag_bias = {
            "work": 0.12,
            "comms": 0.10,
            "system": 0.06,
            "play": -0.08
        }

        self.cluster_weight = 0.15
        self.base_decay = 0.98
        self.adaptive_factor = 0.02
        self.lr = 0.06
        self.weights = {
            "sequence1": 1.0,
            "sequence2": 1.2,
            "sequence3": 1.25,
            "context": 1.0,
            "season_month": 0.6,
            "season_wom": 0.6,
            "tod": 0.8,
            "global": 0.7,
            "cluster": 0.3,
            "recency": 0.6
        }
        self._lock = threading.RLock()
        self.stability_counter = 0
        self.anomaly_baseline: Dict[Tuple[int,int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _habit_cluster(self, habit_key: Optional[str]) -> Optional[str]:
        if not habit_key:
            return None
        h = habit_key.lower()
        for cname, keywords in self.cluster_map.items():
            if any(k in h for k in keywords):
                return cname
        if habit_key.startswith("web:"):
            return "browser"
        if habit_key.startswith("work:"):
            return "office"
        if habit_key.startswith("proc:"):
            return "system"
        return None

    def _semantic_tag(self, habit_key: Optional[str], hour: int) -> Optional[str]:
        if not habit_key:
            return None
        cl = self._habit_cluster(habit_key)
        if cl in {"office", "ide", "terminal"}:
            return "work"
        if cl == "comms":
            return "comms"
        if cl == "system":
            return "system"
        if cl == "browser":
            return "work" if 8 <= hour <= 18 else "play"
        return None

    def _adaptive_decay(self) -> float:
        adj = max(-0.03, min(0.03, (self.stability_counter / 20.0) * self.adaptive_factor))
        return max(0.94, min(0.995, self.base_decay + adj))

    def update_sequence(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str], curr: str) -> None:
        if not curr:
            return
        with self._lock:
            decay = self._adaptive_decay()
            if prev1:
                for nxt in list(self.transitions1[prev1].keys()):
                    self.transitions1[prev1][nxt] *= decay
                self.transitions1[prev1][curr] += 1.0
                c1 = self._habit_cluster(prev1)
                c2 = self._habit_cluster(curr)
                if c1 and c2 and c1 == c2:
                    self.transitions1[prev1][curr] += self.cluster_weight
            if prev1 and prev2:
                key2 = (prev2, prev1)
                for nxt in list(self.transitions2[key2].keys()):
                    self.transitions2[key2][nxt] *= decay
                self.transitions2[key2][curr] += 1.0
            if prev1 and prev2 and prev3:
                key3 = (prev3, prev2, prev1)
                for nxt in list(self.transitions3[key3].keys()):
                    self.transitions3[key3][nxt] *= decay
                self.transitions3[key3][curr] += 1.0

    def _week_of_month(self, dt: datetime.datetime) -> int:
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return int((adjusted_dom - 1) / 7) + 1

    def update_context(self, dt: datetime.datetime, habit_key: str) -> None:
        hour = dt.hour
        dow = dt.weekday()
        weekend = 1 if dow >= 5 else 0
        month = dt.month
        wom = self._week_of_month(dt)
        with self._lock:
            self.context_bias[(hour, dow, weekend)][habit_key] += 1.0
            self.season_month_bias[(month, dow)][habit_key] += 1.0
            self.season_wom_bias[(wom, dow)][habit_key] += 1.0
            self.anomaly_baseline[(hour, dow)][habit_key] += 1.0

    def _tod_smoothing(self, hour: int, habits: Dict[str, HabitStats]) -> Dict[str, float]:
        kernel = {0: 0.25, 1: 0.5, 2: 0.25}
        scores = {}
        for hkey, stats in habits.items():
            val = 0.0
            for dk, w in kernel.items():
                hh = (hour - 1 + dk) % 24
                val += w * float(stats.tod_hist.get(hh, 0))
            scores[hkey] = val
        total = sum(scores.values()) or 1.0
        return {k: (v / total) for k, v in scores.items()}

    def _recency_scores(self, now_ts: float, habits: Dict[str, HabitStats]) -> Dict[str, float]:
        lam = 1.0 / 24.0
        scores = {}
        for hkey, stats in habits.items():
            dt_hours = max(0.0, (now_ts - (stats.last_ts or 0.0)) / 3600.0)
            scores[hkey] = (2.71828 ** (-lam * dt_hours))
        total = sum(scores.values()) or 1.0
        return {k: (v / total) for k, v in scores.items()}

    def _score_snapshot(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str],
                        dt: datetime.datetime, habits: Dict[str, HabitStats], now_ts: float,
                        ctx_features: Optional[Dict[str,float]] = None) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        hour = dt.hour
        dow = dt.weekday()
        weekend = 1 if dow >= 5 else 0
        month = dt.month
        wom = self._week_of_month(dt)

        seq1_scores = defaultdict(float)
        if prev1 and prev1 in self.transitions1:
            for nxt, cnt in self.transitions1[prev1].items():
                seq1_scores[nxt] += cnt

        seq2_scores = defaultdict(float)
        key2 = (prev2, prev1) if (prev2 and prev1) else None
        if key2 and key2 in self.transitions2:
            for nxt, cnt in self.transitions2[key2].items():
                seq2_scores[nxt] += cnt

        seq3_scores = defaultdict(float)
        key3 = (prev3, prev2, prev1) if (prev3 and prev2 and prev1) else None
        if key3 and key3 in self.transitions3:
            for nxt, cnt in self.transitions3[key3].items():
                seq3_scores[nxt] += cnt

        ctx_scores = defaultdict(float)
        for h, cnt in self.context_bias[(hour, dow, weekend)].items():
            ctx_scores[h] += cnt

        season_m_scores = defaultdict(float)
        for h, cnt in self.season_month_bias[(month, dow)].items():
            season_m_scores[h] += cnt

        season_w_scores = defaultdict(float)
        for h, cnt in self.season_wom_bias[(wom, dow)].items():
            season_w_scores[h] += cnt

        glob_scores = {h: s.count for h, s in habits.items()}
        tod_scores = self._tod_smoothing(hour, habits)
        recency_scores = self._recency_scores(now_ts, habits)

        def norm(d: Dict[str, float]) -> Dict[str, float]:
            total = float(sum(d.values()))
            if total <= 0.0:
                return {k: 0.0 for k in d.keys()}
            return {k: (v / total) for k, v in d.items()}

        seq1_n = norm(seq1_scores)
        seq2_n = norm(seq2_scores)
        seq3_n = norm(seq3_scores)
        ctx_n = norm(ctx_scores)
        season_m_n = norm(season_m_scores)
        season_w_n = norm(season_w_scores)
        glob_n = norm(glob_scores)

        scores = defaultdict(float)
        components = {
            "sequence1": {}, "sequence2": {}, "sequence3": {},
            "context": {}, "season_month": {}, "season_wom": {},
            "tod": {}, "global": {}, "cluster": {}, "recency": {}, "semantic": {}
        }
        for h in habits.keys():
            s = (self.weights["sequence1"] * seq1_n.get(h, 0.0)
                 + self.weights["sequence2"] * seq2_n.get(h, 0.0)
                 + self.weights["sequence3"] * seq3_n.get(h, 0.0)
                 + self.weights["context"]   * ctx_n.get(h, 0.0)
                 + self.weights["season_month"] * season_m_n.get(h, 0.0)
                 + self.weights["season_wom"]   * season_w_n.get(h, 0.0)
                 + self.weights["tod"]       * tod_scores.get(h, 0.0)
                 + self.weights["global"]    * glob_n.get(h, 0.0)
                 + self.weights["recency"]   * recency_scores.get(h, 0.0))
            c_prev = self._habit_cluster(prev1) if prev1 else None
            c_h = self._habit_cluster(h)
            if c_prev and c_h and c_prev == c_h:
                s += self.weights["cluster"] * 0.5
                components["cluster"][h] = self.weights["cluster"] * 0.5
            tag = self._semantic_tag(h, hour)
            if tag:
                bias = self.tag_bias.get(tag, 0.0)
                s += bias
                components["semantic"][h] = bias
            if ctx_features:
                if ctx_features.get("deadline_urgency", 0.0) > 0.0 and tag == "work":
                    s += 0.12 * ctx_features["deadline_urgency"]
                if ctx_features.get("meeting_now", 0.0) > 0.0 and (c_h == "comms" or tag == "comms"):
                    s += 0.18
                if ctx_features.get("network_burst", 0.0) > 0.6 and c_h == "browser":
                    s += 0.08
            scores[h] = s
            components["sequence1"][h] = self.weights["sequence1"] * seq1_n.get(h, 0.0)
            components["sequence2"][h] = self.weights["sequence2"] * seq2_n.get(h, 0.0)
            components["sequence3"][h] = self.weights["sequence3"] * seq3_n.get(h, 0.0)
            components["context"][h]   = self.weights["context"]   * ctx_n.get(h, 0.0)
            components["season_month"][h] = self.weights["season_month"] * season_m_n.get(h, 0.0)
            components["season_wom"][h]   = self.weights["season_wom"]   * season_w_n.get(h, 0.0)
            components["tod"][h]       = self.weights["tod"]       * tod_scores.get(h, 0.0)
            components["global"][h]    = self.weights["global"]    * glob_n.get(h, 0.0)
            components["recency"][h]   = self.weights["recency"]   * recency_scores.get(h, 0.0)

        return scores, components

    def _choose(self, scores: Dict[str, float]) -> Tuple[Optional[str], float]:
        if not scores:
            return None, 0.0
        total = sum(scores.values())
        if total <= 0.0:
            best = max(scores.items(), key=lambda x: x[1])[0]
            return best, 0.0
        sorted_vals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_h, best_s = sorted_vals[0]
        second_s = sorted_vals[1][1] if len(sorted_vals) > 1 else 0.0
        margin = best_s - second_s
        conf = max(0.05, min(1.0, (best_s / total) + 0.5 * (margin / (best_s + 1e-9))))
        return best_h, conf

    def _choose_top_two(self, scores: Dict[str, float]) -> Tuple[Optional[str], float, Optional[str], float]:
        if not scores:
            return None, 0.0, None, 0.0
        sorted_vals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_h, best_s = sorted_vals[0]
        second_h, second_s = (sorted_vals[1] if len(sorted_vals) > 1 else (None, 0.0))
        total = sum(scores.values()) or 1.0
        margin = best_s - second_s
        conf1 = max(0.05, min(1.0, (best_s / total) + 0.5 * (margin / (best_s + 1e-9))))
        conf2 = max(0.05, min(0.95, (second_s / total)))
        return best_h, conf1, second_h, conf2

    def anomaly_score(self, habit: str, components: Dict[str, float], hour: int, dow: int) -> float:
        baseline = self.anomaly_baseline[(hour, dow)].get(habit, 0.0)
        total = components.get("total", 0.0)
        norm_base = baseline / (baseline + 8.0)
        ratio = abs(total - norm_base) / (total + norm_base + 1e-9)
        return max(0.0, min(1.0, ratio))

    def predict(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str],
                dt: datetime.datetime, habits: Dict[str, HabitStats], now_ts: float,
                ctx_features: Optional[Dict[str,float]] = None) -> Tuple[Optional[str], float, Dict[str, float]]:
        try:
            scores, components = self._score_snapshot(prev1, prev2, prev3, dt, habits, now_ts, ctx_features)
            best_h, conf = self._choose(scores)
            explain = {k: v.get(best_h, 0.0) for k, v in components.items()} if best_h else {}
            if best_h:
                explain["total"] = scores.get(best_h, 0.0)
                explain["anomaly"] = self.anomaly_score(best_h, explain, dt.hour, dt.weekday())
            return best_h, conf, explain
        except Exception:
            return None, 0.0, {}

    def predict_plan(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str],
                     dt: datetime.datetime, habits: Dict[str, HabitStats], now_ts: float,
                     ctx_features: Optional[Dict[str,float]] = None, length: int = 14) -> List[Tuple[str, float]]:
        plan: List[Tuple[str, float]] = []
        p1, p2, p3 = prev1, prev2, prev3
        try:
            for _ in range(max(1, length)):
                scores, _ = self._score_snapshot(p1, p2, p3, dt, habits, now_ts, ctx_features)
                nxt, conf = self._choose(scores)
                if not nxt:
                    break
                plan.append((nxt, conf))
                p3, p2, p1 = p2, p1, nxt
            return plan
        except Exception:
            return plan

    def predict_top_two_step(self, prev1: Optional[str], prev2: Optional[str], prev3: Optional[str],
                             dt: datetime.datetime, habits: Dict[str, HabitStats], now_ts: float,
                             ctx_features: Optional[Dict[str,float]] = None) -> Tuple[Tuple[Optional[str], float], Tuple[Optional[str], float]]:
        try:
            scores, _ = self._score_snapshot(prev1, prev2, prev3, dt, habits, now_ts, ctx_features)
            b1, c1, b2, c2 = self._choose_top_two(scores)
            return (b1, c1), (b2, c2)
        except Exception:
            return (None, 0.0), (None, 0.0)

    def reward(self, matched: bool, strong: bool = False) -> None:
        with self._lock:
            delta = (self.lr * (1.6 if strong else 1.0)) if matched else (-self.lr * (1.6 if strong else 1.0))
            self.stability_counter = max(-20, min(20, self.stability_counter + (2 if matched else -3)))
            for k in self.weights:
                self.weights[k] = max(0.05, self.weights[k] + delta * (0.5 if k in {"global", "tod"} else 1.0))

    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "transitions1": {src: dict(dst) for src, dst in self.transitions1.items()},
                "transitions2": {"|".join(src): dict(dst) for src, dst in self.transitions2.items()},
                "transitions3": {"|".join(src): dict(dst) for src, dst in self.transitions3.items()},
                "context_bias": {f"{hour},{dow},{weekend}": dict(hs) for (hour, dow, weekend), hs in self.context_bias.items()},
                "season_month_bias": {f"{month},{dow}": dict(hs) for (month, dow), hs in self.season_month_bias.items()},
                "season_wom_bias": {f"{wom},{dow}": dict(hs) for (wom, dow), hs in self.season_wom_bias.items()},
                "weights": dict(self.weights),
                "cluster_weight": self.cluster_weight,
                "base_decay": self.base_decay,
                "adaptive_factor": self.adaptive_factor,
                "lr": self.lr,
                "stability_counter": self.stability_counter,
                "anomaly_baseline": {f"{h},{d}": dict(v) for (h,d), v in self.anomaly_baseline.items()}
            }

    def from_dict(self, d: Dict) -> None:
        with self._lock:
            t1 = d.get("transitions1", {})
            t2 = d.get("transitions2", {})
            t3 = d.get("transitions3", {})
            cb = d.get("context_bias", {})
            sm = d.get("season_month_bias", {})
            sw = d.get("season_wom_bias", {})
            self.transitions1 = defaultdict(lambda: defaultdict(float))
            for src, dst_map in t1.items():
                self.transitions1[src] = defaultdict(float, {k: float(v) for k, v in dst_map.items()})
            self.transitions2 = defaultdict(lambda: defaultdict(float))
            for k, dst_map in t2.items():
                try:
                    prev2, prev1 = k.split("|")
                    key = (prev2, prev1)
                except Exception:
                    continue
                self.transitions2[key] = defaultdict(float, {h: float(v) for h, v in dst_map.items()})
            self.transitions3 = defaultdict(lambda: defaultdict(float))
            for k, dst_map in t3.items():
                try:
                    prev3, prev2, prev1 = k.split("|")
                    key = (prev3, prev2, prev1)
                except Exception:
                    continue
                self.transitions3[key] = defaultdict(float, {h: float(v) for h, v in dst_map.items()})
            self.context_bias = defaultdict(lambda: defaultdict(float))
            for k, hs in cb.items():
                try:
                    hour_str, dow_str, weekend_str = k.split(",")
                    key = (int(hour_str), int(dow_str), int(weekend_str))
                except Exception:
                    continue
                self.context_bias[key] = defaultdict(float, {h: float(v) for h, v in hs.items()})
            self.season_month_bias = defaultdict(lambda: defaultdict(float))
            for k, hs in sm.items():
                try:
                    month_str, dow_str = k.split(",")
                    key = (int(month_str), int(dow_str))
                except Exception:
                    continue
                self.season_month_bias[key] = defaultdict(float, {h: float(v) for h, v in hs.items()})
            self.season_wom_bias = defaultdict(lambda: defaultdict(float))
            for k, hs in sw.items():
                try:
                    wom_str, dow_str = k.split(",")
                    key = (int(wom_str), int(dow_str))
                except Exception:
                    continue
                self.season_wom_bias[key] = defaultdict(float, {h: float(v) for h, v in hs.items()})
            w = d.get("weights", self.weights)
            self.weights = {k: float(v) for k, v in w.items()}
            self.cluster_weight = float(d.get("cluster_weight", self.cluster_weight))
            self.base_decay = float(d.get("base_decay", self.base_decay))
            self.adaptive_factor = float(d.get("adaptive_factor", self.adaptive_factor))
            self.lr = float(d.get("lr", self.lr))
            self.stability_counter = int(d.get("stability_counter", self.stability_counter))
            ab = d.get("anomaly_baseline", {})
            self.anomaly_baseline = defaultdict(lambda: defaultdict(float))
            for k, v in ab.items():
                try:
                    hh, dd = k.split(",")
                    key = (int(hh), int(dd))
                except Exception:
                    continue
                self.anomaly_baseline[key] = defaultdict(float, {h: float(s) for h, s in v.items()})

# ----------------------------
# System monitor (cross-platform) with network baselines
# ----------------------------
class SystemMonitor:
    def __init__(self):
        self.browser_markers = ["Chrome", "Edge", "Firefox", "Brave", "Opera", "Safari"]
        self.cpu_threshold = 2.0
        self.mem_threshold_mb = 100.0
        self.disabled_items: Set[str] = set()
        self._lock = threading.RLock()
        self.net_baseline: Dict[str, float] = defaultdict(lambda: -1.0)

    def normalize_habit(self, raw: str) -> str:
        raw = (raw or "").strip()
        return raw[:128] if raw else ""

    def derive_browser_title_habit(self, title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        for marker in self.browser_markers:
            sep = f" - {marker}"
            if sep in title:
                base = title.split(sep)[0].strip()
                if base:
                    return f"web:{base}"
        return f"app:{title}" if title else None

    def scan_foreground(self) -> Optional[str]:
        title = get_foreground_window_title()
        habit = self.derive_browser_title_habit(title)
        return self.normalize_habit(habit) if habit else None

    def scan_processes(self) -> List[Tuple[str, int, float, float, Optional[str], int, float]]:
        habits: List[Tuple[str, int, float, float, Optional[str], int, float]] = []
        for p in psutil.process_iter(attrs=["pid", "name"]):
            try:
                name = p.info.get("name") or ""
                if not name:
                    continue
                pid = int(p.info.get("pid"))
                parent_name = None
                try:
                    par = p.parent()
                    if par:
                        parent_name = par.name()
                except Exception:
                    parent_name = None

                key = self.normalize_habit(f"proc:{name}")
                with self._lock:
                    disabled = key in self.disabled_items
                cpu = p.cpu_percent(interval=0.0) or 0.0
                mem_mb = (p.memory_info().rss / (1024.0 * 1024.0)) if not disabled else 0.0

                net_conns = 0
                baseline = self.net_baseline.get(name, -1.0)
                try:
                    cons = p.connections(kind='inet')
                    net_conns = len(cons)
                    if baseline < 0:
                        self.net_baseline[name] = net_conns * 1.0
                    else:
                        self.net_baseline[name] = baseline * 0.95 + net_conns * 0.05
                    baseline = self.net_baseline[name]
                except Exception:
                    pass

                if disabled:
                    if cpu > self.cpu_threshold:
                        habits.append((key, pid, cpu, mem_mb, parent_name, net_conns, baseline))
                    continue

                if cpu >= self.cpu_threshold or mem_mb >= self.mem_threshold_mb or net_conns > (baseline * 2 + 2):
                    habits.append((key, pid, cpu, mem_mb, parent_name, net_conns, baseline))
                    lname = name.lower()
                    if any(tag in lname for tag in ["steam", "epic", "game"]):
                        habits.append((self.normalize_habit(f"play:{name}"), pid, cpu, mem_mb, parent_name, net_conns, baseline))
                    if any(tag in lname for tag in ["powershell", "cmd", "terminal", "vscode", "pycharm", "excel", "word", "outlook", "bash", "zsh", "teams", "slack"]):
                        habits.append((self.normalize_habit(f"work:{name}"), pid, cpu, mem_mb, parent_name, net_conns, baseline))
            except Exception:
                continue
        return habits

    def mark_disabled(self, items: List[str]) -> None:
        with self._lock:
            for it in items:
                if it:
                    self.disabled_items.add(it)

    def mark_enabled(self, items: List[str]) -> None:
        with self._lock:
            for it in items:
                if it and it in self.disabled_items:
                    self.disabled_items.remove(it)

# ----------------------------
# Intrusion alert container
# ----------------------------
class IntrusionAlert:
    def __init__(self, message: str, pid: Optional[int], habit_key: Optional[str], severity: float):
        self.message = message
        self.pid = pid
        self.habit_key = habit_key
        self.severity = severity
        self.timestamp = time.time()

# ----------------------------
# Foresight engine: encrypted cache, security, calibration, workflows, ensemble, auto-play
# ----------------------------
class Foresight:
    def __init__(self, cache_file: str = "foresight_cache.bin"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self.guard = SecurityGuardian()
        self.habits: Dict[str, HabitStats] = {}
        self.disabled_habits: Dict[str, HabitStats] = {}
        self.event_log: List[Tuple[float, str]] = []
        self.predictor = NextGenPredictor()
        self.calibrator = ConfidenceCalibrator()
        self.workflows = WorkflowChains()
        self.context = ContextManager()
        self.learning_enabled = True
        self.activity_score = 0.0
        self.last_review_day = None
        self.monitor = SystemMonitor()
        self.alert_queue: deque[IntrusionAlert] = deque(maxlen=200)
        self.action_map: Dict[str, Dict[str, str]] = {}  # {"habit_key": {"type": "cmd|open", "value": "..." }}
        # Auto-play settings and safelist
        self.auto_play_enabled: bool = False
        self.auto_play_threshold: float = 0.85
        self.auto_play_delay_sec: float = 4.0
        self.auto_safe_clusters: Set[str] = {"browser", "office", "comms"}
        self.auto_whitelist_habits: Set[str] = set()
        self._load_cache()

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        with self._lock:
            try:
                with open(self.cache_file, "rb") as f:
                    enc = f.read()
                raw = self.guard.decrypt(enc)
                data = json.loads(raw.decode("utf-8"))
                self.habits = {h: HabitStats.from_dict(s) for h, s in data.get("habits", {}).items()}
                self.disabled_habits = {h: HabitStats.from_dict(s) for h, s in data.get("disabled", {}).items()}
                self.event_log = [(float(ts), str(h)) for ts, h in data.get("event_log", [])][-60000:]
                self.last_review_day = data.get("last_review_day")
                self.guard.whitelist = set(data.get("whitelist", []))
                self.action_map = data.get("action_map", {})
                self.auto_play_enabled = bool(data.get("auto_play_enabled", self.auto_play_enabled))
                self.auto_play_threshold = float(data.get("auto_play_threshold", self.auto_play_threshold))
                self.auto_play_delay_sec = float(data.get("auto_play_delay_sec", self.auto_play_delay_sec))
                self.auto_safe_clusters = set(data.get("auto_safe_clusters", list(self.auto_safe_clusters)))
                self.auto_whitelist_habits = set(data.get("auto_whitelist_habits", list(self.auto_whitelist_habits)))
                self.monitor.mark_disabled(list(self.disabled_habits.keys()))
                pred_state = data.get("predictor_state")
                if pred_state:
                    self.predictor.from_dict(pred_state)
                calib_state = data.get("calibrator_state")
                if calib_state:
                    self.calibrator.from_dict(calib_state)
                wf_state = data.get("workflow_state")
                if wf_state:
                    self.workflows.from_dict(wf_state)
            except Exception:
                self.habits = {}
                self.disabled_habits = {}
                self.event_log = []
                self.last_review_day = None
                self.guard.whitelist = set()
                self.action_map = {}

    def _save_cache(self) -> None:
        with self._lock:
            data = {
                "habits": {h: s.to_dict() for h, s in self.habits.items()},
                "disabled": {h: s.to_dict() for h, s in self.disabled_habits.items()},
                "event_log": self.event_log[-60000:],
                "last_review_day": self.last_review_day,
                "whitelist": list(self.guard.whitelist),
                "predictor_state": self.predictor.to_dict(),
                "calibrator_state": self.calibrator.to_dict(),
                "workflow_state": self.workflows.to_dict(),
                "action_map": self.action_map,
                "auto_play_enabled": self.auto_play_enabled,
                "auto_play_threshold": self.auto_play_threshold,
                "auto_play_delay_sec": self.auto_play_delay_sec,
                "auto_safe_clusters": list(self.auto_safe_clusters),
                "auto_whitelist_habits": list(self.auto_whitelist_habits),
            }
            raw = json.dumps(data).encode("utf-8")
            enc = self.guard.encrypt(raw)
            tmp = self.cache_file + ".tmp"
            with open(tmp, "wb") as f:
                f.write(enc)
            os.replace(tmp, self.cache_file)

    def record_habit(self, habit: str, ts: Optional[float] = None) -> None:
        if not habit or not habit.strip():
            return
        habit = habit.strip()
        now = ts if ts is not None else time.time()
        dt = datetime.datetime.fromtimestamp(now)
        hour = dt.hour
        with self._lock:
            if habit in self.disabled_habits:
                self.habits[habit] = self.disabled_habits.pop(habit)
                self.monitor.mark_enabled([habit])
            stats = self.habits.get(habit)
            if stats is None:
                stats = HabitStats()
                self.habits[habit] = stats
            stats.count += 1
            stats.last_ts = now
            stats.tod_hist[hour] = stats.tod_hist.get(hour, 0) + 1
            prev1 = self.event_log[-1][1] if self.event_log else None
            prev2 = self.event_log[-2][1] if len(self.event_log) > 1 else None
            prev3 = self.event_log[-3][1] if len(self.event_log) > 2 else None
            prev1 = prev1 if isinstance(prev1, str) and not prev1.startswith("advisory:") else None
            prev2 = prev2 if isinstance(prev2, str) and not prev2.startswith("advisory:") else None
            prev3 = prev3 if isinstance(prev3, str) and not prev3.startswith("advisory:") else None
            self.predictor.update_sequence(prev1, prev2, prev3, habit)
            self.predictor.update_context(dt, habit)
            self.event_log.append((now, habit))
            if len(self.event_log) > 60000:
                self.event_log = self.event_log[-60000:]
            self.activity_score = min(1.0, self.activity_score + 0.08)
            if habit.startswith("web:"):
                today = dt.strftime("%Y-%m-%d")
                key = f"{today}|{habit}"
                self.guard.safe_daily[key] += 1
        if (time.time() - self.workflows.last_build_ts) > 45.0:
            self.workflows.ingest_log(self.event_log[-1000:])  # longer slice for better routines
        self._save_cache()

    def get_top_habits(self, n: int = 12) -> List[Tuple[str, int]]:
        with self._lock:
            return sorted(((h, s.count) for h, s in self.habits.items()),
                          key=lambda x: x[1], reverse=True)[:n]

    def _prev_context(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        prev1 = self.event_log[-1][1] if self.event_log else None
        prev2 = self.event_log[-2][1] if len(self.event_log) > 1 else None
        prev3 = self.event_log[-3][1] if len(self.event_log) > 2 else None
        prev1 = prev1 if isinstance(prev1, str) and not prev1.startswith("advisory:") else None
        prev2 = prev2 if isinstance(prev2, str) and not prev2.startswith("advisory:") else None
        prev3 = prev3 if isinstance(prev3, str) and not prev3.startswith("advisory:") else None
        return prev1, prev2, prev3

    def _ensemble(self, base_pred: Optional[str], base_conf: float,
                  chain_suggestion: Optional[Tuple[str, float]],
                  ctx_features: Dict[str,float]) -> Tuple[Optional[str], float]:
        pred = base_pred
        conf = base_conf
        if chain_suggestion:
            c_step, c_conf = chain_suggestion
            if base_pred == c_step:
                conf = min(1.0, 0.7 * base_conf + 0.3 * c_conf + 0.1)
            else:
                urgency = ctx_features.get("deadline_urgency", 0.0) + ctx_features.get("meeting_now", 0.0)
                # Penalize divergence unless chain is significantly stronger or urgency favors it
                if c_conf >= base_conf * (0.9 - 0.2 * urgency):
                    pred, conf = c_step, max(base_conf, c_conf)
                else:
                    conf = base_conf * 0.98
        return pred, conf

    def predict_next(self, when_ts: Optional[float] = None) -> Tuple[Optional[str], float, Dict[str, float]]:
        try:
            now_ts = when_ts if when_ts is not None else time.time()
            dt = datetime.datetime.fromtimestamp(now_ts)
            ctx_features = self.context.features()
            with self._lock:
                prev1, prev2, prev3 = self._prev_context()
            raw_pred, raw_conf, explain = self.predictor.predict(prev1, prev2, prev3, dt, self.habits, now_ts, ctx_features)
            chain_suggestion = self.workflows.next_in_chain(prev1, prev2, prev3)
            raw_pred, raw_conf = self._ensemble(raw_pred, raw_conf, chain_suggestion, ctx_features)
            cal_conf = self.calibrator.calibrate(raw_conf) if raw_pred else 0.0
            return raw_pred, cal_conf, explain
        except Exception:
            return None, 0.0, {}

    def predict_branch_plans(self, when_ts: Optional[float] = None, length: int = 14) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        try:
            now_ts = when_ts if when_ts is not None else time.time()
            dt = datetime.datetime.fromtimestamp(now_ts)
            ctx_features = self.context.features()
            with self._lock:
                prev1, prev2, prev3 = self._prev_context()
            # Primary plan
            primary_raw = self.predictor.predict_plan(prev1, prev2, prev3, dt, self.habits, now_ts, ctx_features, length=length)
            # Alternative path: compute top two for first decision and follow the second if available
            (top_step, top_conf), (alt_step, alt_conf) = self.predictor.predict_top_two_step(prev1, prev2, prev3, dt, self.habits, now_ts, ctx_features)
            alt_raw: List[Tuple[str, float]] = []
            if alt_step and (not primary_raw or alt_step != primary_raw[0][0]):
                # Build alt plan starting with alt_step then proceed
                alt_raw.append((alt_step, alt_conf))
                p1, p2, p3 = alt_step, prev1, prev2  # shift context by choosing alternative first
                alt_raw.extend(self.predictor.predict_plan(p1, p2, p3, dt, self.habits, now_ts, ctx_features, length=max(1, length-1)))
            # Calibrate
            primary = [(s, self.calibrator.calibrate(c)) for s, c in primary_raw]
            alt = [(s, self.calibrator.calibrate(c)) for s, c in alt_raw]
            return primary, alt
        except Exception:
            return [], []

    def predict_next_plan(self, when_ts: Optional[float] = None, length: int = 14) -> List[Tuple[str, float]]:
        primary, alt = self.predict_branch_plans(when_ts=when_ts, length=length)
        merged: List[Tuple[str, float]] = []
        for i, (s, c) in enumerate(primary):
            merged.append((s, c))
            if i == 1 and alt:
                for a_s, a_c in alt[:3]:
                    merged.append((f"{a_s} (alt)", max(0.0, min(1.0, a_c * 0.95))))
        return merged

    def routine_preview(self) -> Optional[List[str]]:
        with self._lock:
            prev1, _, _ = self._prev_context()
        if not prev1:
            return None
        return self.workflows.routine_for_trigger(prev1)

    def reinforce(self) -> None:
        with self._lock:
            if not self.event_log or not self.habits:
                return
            last_ts, last_habit = self.event_log[-1]
            dt = datetime.datetime.fromtimestamp(last_ts)
            prev1 = self.event_log[-2][1] if len(self.event_log) > 1 else None
            prev2 = self.event_log[-3][1] if len(self.event_log) > 2 else None
            prev3 = self.event_log[-4][1] if len(self.event_log) > 3 else None
            prev1 = prev1 if isinstance(prev1, str) and not prev1.startswith("advisory:") else None
            prev2 = prev2 if isinstance(prev2, str) and not prev2.startswith("advisory:") else None
            prev3 = prev3 if isinstance(prev3, str) and not prev3.startswith("advisory:") else None
            ctx_features = self.context.features()
            predicted, raw_conf, _ = self.predictor.predict(prev1, prev2, prev3, dt, self.habits, last_ts, ctx_features)
            correct = (predicted == last_habit)
            self.calibrator.update(raw_conf or 0.0, correct)
            self.predictor.reward(correct, strong=True)
            self.activity_score = max(0.0, self.activity_score - 0.02)
        self._save_cache()

    def daily_review(self) -> int:
        with self._lock:
            now = time.time()
            today_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            if self.last_review_day == today_str:
                return 0
            cutoff = now - 24 * 3600
            disabled_count = 0
            to_disable = []
            for habit, stats in list(self.habits.items()):
                if stats.last_ts < cutoff:
                    self.disabled_habits[habit] = self.habits.pop(habit)
                    to_disable.append(habit)
                    disabled_count += 1
            self.last_review_day = today_str
        self.monitor.mark_disabled(to_disable)
        self._save_cache()
        return disabled_count

    def assess_process(self, habit_key: str, pid: int, cpu: float, mem_mb: float,
                       parent_name: Optional[str], net_conns: int, baseline_net: float) -> Optional[IntrusionAlert]:
        hour = datetime.datetime.fromtimestamp(time.time()).hour
        with self._lock:
            seen_before = habit_key in self.habits
        name = habit_key.replace("proc:", "") if habit_key.startswith("proc:") else habit_key
        suspicion = self.guard.score_process_suspicion(
            name=name, cpu=cpu, mem_mb=mem_mb, hour=hour, seen_before=seen_before,
            parent_name=parent_name, net_connections=net_conns, baseline_net=baseline_net
        )
        if suspicion >= 0.7:
            msg = f"Unauthorized access suspected: {name} (PID {pid})"
            alert = IntrusionAlert(message=msg, pid=pid, habit_key=habit_key, severity=suspicion)
            with self._lock:
                self.alert_queue.append(alert)
                self.event_log.append((time.time(), f"advisory:{msg}"))
                if len(self.event_log) > 60000:
                    self.event_log = self.event_log[-60000:]
            return alert
        return None

    def ingest_passive_events(self) -> Tuple[int, Optional[str]]:
        if not self.learning_enabled:
            return 0, None
        ingested = 0
        advisory = None

        fg = self.monitor.scan_foreground()
        if fg:
            self.record_habit(fg)
            ingested += 1
            if fg.startswith("web:"):
                title = fg[4:]
                dt = datetime.datetime.now()
                hour = dt.hour
                today = dt.strftime("%Y-%m-%d")
                daily_count = self.guard.safe_daily.get(f"{today}|{fg}", 0)
                susp = self.guard.score_title_suspicion(title, hour, daily_count)
                if susp >= 0.7:
                    advisory = f"Suspicious site behavior: '{title}'. Proceed with caution."

        for habit_key, pid, cpu, mem_mb, parent_name, net_conns, baseline_net in self.monitor.scan_processes():
            self.record_habit(habit_key)
            ingested += 1
            self.assess_process(habit_key, pid, cpu, mem_mb, parent_name, net_conns, baseline_net)

        if ingested > 0:
            self.reinforce()
        return ingested, advisory

    # -------- Actions and auto-play settings --------
    def set_action(self, habit_key: str, action_type: str, value: str) -> None:
        with self._lock:
            self.action_map[habit_key] = {"type": action_type, "value": value}
        self._save_cache()

    def execute_action(self, habit_key: str) -> bool:
        try:
            with self._lock:
                spec = self.action_map.get(habit_key)
            if not spec:
                return False
            atype = spec.get("type")
            val = spec.get("value")
            if not atype or not val:
                return False
            plat = sys.platform
            if atype == "cmd":
                subprocess.Popen(val, shell=True)
                return True
            elif atype == "open":
                if plat.startswith("win"):
                    os.startfile(val)  # type: ignore[attr-defined]
                    return True
                elif plat == "darwin":
                    subprocess.Popen(["open", val])
                    return True
                else:
                    subprocess.Popen(["xdg-open", val])
                    return True
            return False
        except Exception:
            return False

    def set_auto_play(self, enabled: bool, threshold: float, delay_sec: float) -> None:
        with self._lock:
            self.auto_play_enabled = bool(enabled)
            self.auto_play_threshold = max(0.5, min(0.99, float(threshold)))
            self.auto_play_delay_sec = max(1.0, min(15.0, float(delay_sec)))
        self._save_cache()

    def set_auto_safelists(self, clusters: List[str], habits: List[str]) -> None:
        with self._lock:
            self.auto_safe_clusters = set(c.strip() for c in clusters if c)
            self.auto_whitelist_habits = set(h.strip() for h in habits if h)
        self._save_cache()

    def habit_cluster(self, habit_key: Optional[str]) -> Optional[str]:
        return self.predictor._habit_cluster(habit_key)

    def set_learning(self, enabled: bool) -> None:
        with self._lock:
            self.learning_enabled = bool(enabled)

    def kill_process(self, pid: int, habit_key: Optional[str] = None) -> bool:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except psutil.TimeoutExpired:
                p.kill()
            with self._lock:
                self.predictor.weights["global"] = max(0.05, self.predictor.weights["global"] - 0.05)
            return True
        except Exception:
            return False

    def allow_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.whitelist.add(habit_key)
            self.predictor.weights["sequence1"] += 0.02
            self.predictor.weights["sequence2"] += 0.02
            self.predictor.weights["sequence3"] += 0.02
        self._save_cache()

    def monitor_habit(self, habit_key: str) -> None:
        with self._lock:
            self.guard.watchlist.add(habit_key)
        self._save_cache()

# ----------------------------
# Background scheduler with adaptive auto-play
# ----------------------------
class Scheduler(threading.Thread):
    def __init__(self, foresight: Foresight, gui_ref, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.fs = foresight
        self.gui = gui_ref
        self.stop_event = stop_event
        self._auto_timer: Optional[threading.Timer] = None
        self._pending_auto_step: Optional[str] = None

    def _cancel_auto(self):
        if self._auto_timer:
            self._auto_timer.cancel()
            self._auto_timer = None
        self._pending_auto_step = None
        if self.gui:
            self.gui.status_var.set("Auto-play canceled.")

    def _auto_execute(self):
        step = self._pending_auto_step
        self._pending_auto_step = None
        self._auto_timer = None
        if not step:
            return
        executed = False
        cl = self.fs.habit_cluster(step)
        with self.fs._lock:
            safe_clusters = self.fs.auto_safe_clusters
            safe_habits = self.fs.auto_whitelist_habits
        if (cl and cl in safe_clusters) or (step in safe_habits):
            executed = self.fs.execute_action(step)
        self.fs.record_habit(step)
        self.fs.reinforce()
        if self.gui:
            self.gui._update_plan()
            msg = "Auto-executed and recorded." if executed else "Auto-recorded (execution not allowed)."
            self.gui.status_var.set(msg + " Plan refreshed.")

    def maybe_autoplay(self, suggestion: Optional[str], confidence: float):
        with self.fs._lock:
            enabled = self.fs.auto_play_enabled
            threshold = self.fs.auto_play_threshold
            delay_base = self.fs.auto_play_delay_sec
        if not enabled or not suggestion:
            return
        if confidence < threshold:
            return
        if confidence >= 0.97:
            delay = max(1.0, delay_base * 0.5)
        elif confidence >= 0.92:
            delay = max(1.5, delay_base * 0.7)
        else:
            delay = delay_base
        self._pending_auto_step = suggestion
        if self.gui:
            self.gui.status_var.set(f"Auto-play in {delay:.0f}s: {suggestion} (Conf: {confidence*100:.0f}%). Click 'Cancel auto' to stop.")
        if self._auto_timer:
            self._auto_timer.cancel()
        self._auto_timer = threading.Timer(delay, self._auto_execute)
        self._auto_timer.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                disabled_ct = self.fs.daily_review()
                with self.fs._lock:
                    active = self.fs.activity_score
                    learning = self.fs.learning_enabled
                base_interval = 2.0 if learning else 6.0
                interval = max(0.8, base_interval * (1.5 - active))
                ingested, advisory = self.fs.ingest_passive_events()
                if ingested == 0 and learning and disabled_ct == 0:
                    self.fs.reinforce()
                if advisory and self.gui:
                    with self.fs._lock:
                        self.fs.event_log.append((time.time(), f"advisory:{advisory}"))
                        if len(self.fs.event_log) > 60000:
                            self.fs.event_log = self.fs.event_log[-60000:]
                suggestion, conf, _ = self.fs.predict_next()
                self.maybe_autoplay(suggestion, conf)
                time.sleep(interval)
            except Exception:
                time.sleep(3.0)

# ----------------------------
# GUI (live assistant + forecasting + auto-play + safe lists + anomaly banner + routine preview)
# ----------------------------
class ForesightGUI:
    def __init__(self, foresight: Foresight, update_ms: int = 1000):
        self.fs = foresight
        self.update_ms = update_ms
        self.stop_event = threading.Event()
        self.root = tk.Tk()
        self.root.title("Foresight")
        try:
            self.root.call('tk', 'scaling', 1.0)
        except Exception:
            pass

        self.root.geometry("1080x600")
        self.root.minsize(980, 520)

        self.figure = None
        self.ax = None
        self.canvas = None
        self.explain_last: Dict[str, float] = {}
        self.execute_enabled = tk.BooleanVar(value=False)
        self.auto_enabled_var = tk.BooleanVar(value=self.fs.auto_play_enabled)
        self.auto_threshold_var = tk.DoubleVar(value=self.fs.auto_play_threshold)
        self.auto_delay_var = tk.DoubleVar(value=self.fs.auto_play_delay_sec)
        self._build_layout()

        self.scheduler = Scheduler(self.fs, self, self.stop_event)
        self.scheduler.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=6)
        container.pack(fill="both", expand=True)
        for c in range(4):
            container.columnconfigure(c, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(2, weight=0)

        top = ttk.Frame(container)
        top.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        for c in range(14):
            top.columnconfigure(c, weight=0)
        top.columnconfigure(0, weight=1)

        self.suggestion_var = tk.StringVar(value="Suggestion: —")
        self.confidence_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.suggestion_var).grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.confidence_var, foreground="#2f4f4f").grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="Why?", command=self._show_why).grid(row=0, column=2, sticky="e")
        ttk.Button(top, text="Preview routine", command=self._preview_routine).grid(row=0, column=3, sticky="e")

        self.learning_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Learning", variable=self.learning_var, command=self._toggle_learning).grid(row=0, column=4, sticky="e")
        ttk.Checkbutton(top, text="Execute actions", variable=self.execute_enabled).grid(row=0, column=5, sticky="e")

        ttk.Checkbutton(top, text="Auto-play", variable=self.auto_enabled_var, command=self._apply_auto).grid(row=0, column=6, sticky="e")
        ttk.Label(top, text="Threshold").grid(row=0, column=7, sticky="e")
        self.threshold_spin = ttk.Spinbox(top, from_=0.5, to=0.99, increment=0.01, textvariable=self.auto_threshold_var, width=5, command=self._apply_auto)
        self.threshold_spin.grid(row=0, column=8, sticky="e")

        ttk.Label(top, text="Delay (s)").grid(row=1, column=7, sticky="e")
        self.delay_spin = ttk.Spinbox(top, from_=1.0, to=15.0, increment=1.0, textvariable=self.auto_delay_var, width=5, command=self._apply_auto)
        self.delay_spin.grid(row=1, column=8, sticky="e")

        ttk.Button(top, text="Cancel auto", command=self._cancel_auto).grid(row=1, column=6, sticky="e")
        ttk.Button(top, text="Safe lists", command=self._edit_safe_lists).grid(row=1, column=5, sticky="e")

        self.advisory_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.advisory_var, foreground="#b22222").grid(row=1, column=0, columnspan=3, sticky="w")

        self.anomaly_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.anomaly_var, foreground="#8b0000").grid(row=1, column=3, columnspan=2, sticky="w")

        left = ttk.LabelFrame(container, text="Top habits", padding=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, height=18)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        midL = ttk.LabelFrame(container, text="Frequency", padding=6)
        midL.grid(row=1, column=1, sticky="nsew", padx=(4, 4))
        midL.columnconfigure(0, weight=1)
        midL.rowconfigure(0, weight=1)
        if plt is not None and FigureCanvasTkAgg is not None:
            self.figure = plt.Figure(figsize=(4.2, 2.5), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Frequency")
            self.ax.set_ylabel("Count")
            self.ax.tick_params(axis='x', rotation=35, labelsize=7)
            self.canvas = FigureCanvasTkAgg(self.figure, master=midL)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        midR = ttk.LabelFrame(container, text="Next steps (live assistant)", padding=6)
        midR.grid(row=1, column=2, sticky="nsew", padx=(4, 4))
        midR.columnconfigure(0, weight=1)
        midR.rowconfigure(0, weight=1)

        self.plan_tree = ttk.Treeview(midR, columns=("step", "confidence"), show="headings", height=18)
        self.plan_tree.heading("step", text="Step")
        self.plan_tree.heading("confidence", text="Confidence")
        self.plan_tree.column("step", width=300, anchor="w")
        self.plan_tree.column("confidence", width=100, anchor="center")
        self.plan_tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
        plan_scroll = ttk.Scrollbar(midR, orient="vertical", command=self.plan_tree.yview)
        plan_scroll.grid(row=0, column=3, sticky="ns")
        self.plan_tree.config(yscrollcommand=plan_scroll.set)

        ttk.Button(midR, text="Accept", command=self._accept_step).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(midR, text="Skip", command=self._skip_step).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(midR, text="Up", command=self._move_up).grid(row=1, column=2, sticky="ew", pady=(6, 0))

        ttk.Button(midR, text="Down", command=self._move_down).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(midR, text="Edit actions", command=self._edit_actions).grid(row=2, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(midR, text="Execute selected", command=self._execute_selected).grid(row=2, column=2, sticky="ew", pady=(4, 0))

        right = ttk.LabelFrame(container, text="Auto-play settings", padding=6)
        right.grid(row=1, column=3, sticky="nsew", padx=(4, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.auto_summary = tk.StringVar(value=self._format_auto_summary())
        ttk.Label(right, textvariable=self.auto_summary, justify="left").grid(row=0, column=0, sticky="nw")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", side="bottom")

    def _format_auto_summary(self) -> str:
        with self.fs._lock:
            clusters = ", ".join(sorted(self.fs.auto_safe_clusters)) or "—"
            habits = ", ".join(sorted(self.fs.auto_whitelist_habits)) or "—"
        return f"Enabled: {self.auto_enabled_var.get()}\nThreshold: {self.auto_threshold_var.get():.2f}\nDelay: {self.auto_delay_var.get():.1f}s\nSafe clusters: {clusters}\nSafe habits: {habits}"

    def _apply_auto(self):
        self.fs.set_auto_play(self.auto_enabled_var.get(), self.auto_threshold_var.get(), self.auto_delay_var.get())
        self.auto_summary.set(self._format_auto_summary())
        self.status_var.set("Auto-play settings updated.")

    def _cancel_auto(self):
        self.scheduler._cancel_auto()

    def _edit_safe_lists(self):
        clusters_str = simpledialog.askstring("Safe clusters", "Comma-separated clusters to auto-execute (e.g., browser,office,comms):")
        habits_str = simpledialog.askstring("Safe habits", "Comma-separated habit keys to auto-execute (e.g., app:Notepad, web:Docs):")
        clusters = [s.strip() for s in clusters_str.split(",")] if clusters_str else []
        habits = [s.strip() for s in habits_str.split(",")] if habits_str else []
        self.fs.set_auto_safelists(clusters, habits)
        self.auto_summary.set(self._format_auto_summary())
        self.status_var.set("Safe lists updated.")

    def _toggle_learning(self):
        self.fs.set_learning(self.learning_var.get())
        self.status_var.set("Learning is " + ("ON" if self.learning_var.get() else "OFF"))

    def _poll_advisory(self) -> Optional[str]:
        with self.fs._lock:
            for ts, h in reversed(self.fs.event_log[-50:]):
                if isinstance(h, str) and h.startswith("advisory:"):
                    return h[len("advisory:"):]
        return None

    def _poll_alert(self) -> Optional[IntrusionAlert]:
        with self.fs._lock:
            if self.fs.alert_queue:
                return self.fs.alert_queue.pop()
        return None

    def _show_intrusion_prompt(self, alert: IntrusionAlert):
        msg = f"{alert.message}\n\nChoose an action:\n- Kill: terminate the process\n- Allow: whitelist and ignore alerts for this item\n- Monitor: keep observing and alert if it escalates"
        resp = messagebox.askquestion("Intrusion detected", msg, icon="warning")
        if resp == "yes":
            success = (alert.pid is not None) and self.fs.kill_process(alert.pid, alert.habit_key)
            self.status_var.set("Process terminated." if success else "Failed to terminate process.")
            return
        resp2 = messagebox.askyesno("Allow or Monitor?", "Allow this item (Yes) or Monitor it (No)?")
        if resp2:
            if alert.habit_key:
                self.fs.allow_habit(alert.habit_key)
                self.status_var.set("Item whitelisted.")
        else:
            if alert.habit_key:
                self.fs.monitor_habit(alert.habit_key)
                self.status_var.set("Item added to watchlist.")

    def _show_why(self):
        explain = self.explain_last or {}
        if not explain or "total" not in explain:
            messagebox.showinfo("Why this prediction", "No explanation available yet.")
            return
        lines = [
            f"Total score: {explain.get('total', 0.0):.3f}",
            f"Sequence (1-step): {explain.get('sequence1', 0.0):.3f}",
            f"Sequence (2-step): {explain.get('sequence2', 0.0):.3f}",
            f"Sequence (3-step): {explain.get('sequence3', 0.0):.3f}",
            f"Context: {explain.get('context', 0.0):.3f}",
            f"Season (month): {explain.get('season_month', 0.0):.3f}",
            f"Season (week-of-month): {explain.get('season_wom', 0.0):.3f}",
            f"Hour-of-day: {explain.get('tod', 0.0):.3f}",
            f"Global: {explain.get('global', 0.0):.3f}",
            f"Cluster boost: {explain.get('cluster', 0.0)::.3f}",
            f"Recency: {explain.get('recency', 0.0):.3f}",
            f"Semantic tag bias: {explain.get('semantic', 0.0):.3f}",
            f"Anomaly score: {explain.get('anomaly', 0.0):.2f}",
        ]
        messagebox.showinfo("Why this prediction", "\n".join(lines))

    def _preview_routine(self):
        routine = self.fs.routine_preview()
        if not routine:
            messagebox.showinfo("Routine preview", "No routine recognized from current trigger.")
            return
        msg = "Recognized routine:\n\n" + "\n → ".join(routine)
        messagebox.showinfo("Routine preview", msg)

    def _update_plan(self):
        try:
            plan = self.fs.predict_next_plan(length=14)
            for item in self.plan_tree.get_children():
                self.plan_tree.delete(item)
            for step, conf in plan:
                self.plan_tree.insert("", "end", values=(step, f"{conf*100:.0f}%"))
        except Exception:
            pass

    def _selected_plan_item(self) -> Optional[str]:
        try:
            sel = self.plan_tree.selection()
            if not sel:
                return None
            vals = self.plan_tree.item(sel[0], "values")
            return vals[0] if vals else None
        except Exception:
            return None

    def _accept_step(self):
        step = self._selected_plan_item()
        if not step:
            self.status_var.set("No step selected.")
            return
        step_clean = step.replace(" (alt)", "")
        self.fs.record_habit(step_clean)
        self.fs.reinforce()
        executed = False
        if self.execute_enabled.get():
            executed = self.fs.execute_action(step_clean)
        self.status_var.set(("Accepted and executed." if executed else "Accepted.") + " Plan refreshed.")
        self._update_plan()

    def _skip_step(self):
        step = self._selected_plan_item()
        if not step:
            self.status_var.set("No step selected.")
            return
        self.fs.predictor.reward(matched=False, strong=True)
        for iid in self.plan_tree.get_children():
            vals = self.plan_tree.item(iid, "values")
            if vals and vals[0] == step:
                self.plan_tree.delete(iid)
                break
        self.status_var.set("Skipped. Predictor adjusted.")
        self._update_plan()
        self.fs._save_cache()

    def _move_up(self):
        sel = self.plan_tree.selection()
        if not sel:
            self.status_var.set("No step selected.")
            return
        iid = sel[0]
        prev_iid = self.plan_tree.prev(iid)
        if not prev_iid:
            return
        self.plan_tree.move(iid, self.plan_tree.parent(iid), self.plan_tree.index(prev_iid))
        self.status_var.set("Reordered: moved up.")

    def _move_down(self):
        sel = self.plan_tree.selection()
        if not sel:
            self.status_var.set("No step selected.")
            return
        iid = sel[0]
        next_iid = self.plan_tree.next(iid)
        if not next_iid:
            return
        self.plan_tree.move(iid, self.plan_tree.parent(iid), self.plan_tree.index(next_iid) + 1)
        self.status_var.set("Reordered: moved down.")

    def _edit_actions(self):
        habit_key = simpledialog.askstring("Action editor", "Habit key to bind (e.g., app:Notepad, web:Docs):")
        if not habit_key:
            return
        atype = simpledialog.askstring("Action type", "Type: 'cmd' to run, 'open' to open file/url/app:")
        if not atype or atype not in {"cmd", "open"}:
            messagebox.showerror("Invalid type", "Action type must be 'cmd' or 'open'.")
            return
        value = simpledialog.askstring("Action value", "Command or path/URL to open:")
        if not value:
            return
        self.fs.set_action(habit_key.strip(), atype.strip(), value.strip())
        self.status_var.set(f"Action bound: {habit_key} -> {atype} {value}")

    def _execute_selected(self):
        step = self._selected_plan_item()
        if not step:
            self.status_var.set("No step selected.")
            return
        ok = self.fs.execute_action(step.replace(" (alt)", ""))
        self.status_var.set("Executed." if ok else "No action bound or failed.")

    def _tick(self):
        try:
            result = self.fs.predict_next()
            if isinstance(result, tuple) and len(result) == 3:
                pred, conf, explain = result
            else:
                pred, conf, explain = None, 0.0, {}

            self.explain_last = explain or {}
            if pred:
                self.suggestion_var.set(f"Suggestion: {pred}")
                self.confidence_var.set(f"Confidence: {conf*100:.0f}%")
                anom = float(explain.get("anomaly", 0.0)) if explain else 0.0
                self.anomaly_var.set("" if anom < 0.35 else f"Outlier pattern predicted (anomaly {anom:.2f}).")
            else:
                self.suggestion_var.set("Suggestion: —")
                self.confidence_var.set("")
                self.explain_last = {}
                self.anomaly_var.set("")

            adv = self._poll_advisory()
            self.advisory_var.set(adv if adv else "")

            alert = self._poll_alert()
            if alert:
                self._show_intrusion_prompt(alert)

            self.listbox.delete(0, tk.END)
            for h, c in self.fs.get_top_habits(20):
                self.listbox.insert(tk.END, f"{h}  —  {c}")

            if self.canvas is not None and self.ax is not None:
                self.ax.clear()
                self.ax.set_title("Frequency")
                self.ax.set_ylabel("Count")
                data = self.fs.get_top_habits(10)
                if data:
                    habits = [h for h, _ in data]
                    counts = [c for _, c in data]
                    self.ax.bar(habits, counts, color="#66b3ff")
                    self.ax.tick_params(axis='x', rotation=35, labelsize=7)
                else:
                    self.ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
                self.canvas.draw()

            self._update_plan()

            with self.fs._lock:
                total = len(self.fs.habits)
                disabled_total = len(self.fs.disabled_habits)
                events = len(self.fs.event_log)
                act = self.fs.activity_score
                learn = self.fs.learning_enabled
            self.status_var.set(
                f"Active: {total} | Disabled: {disabled_total} | Events: {events} | Act: {act:.2f} | {'Learning ON' if learn else 'Learning OFF'}"
            )
        except Exception as e:
            self.status_var.set(f"Update error: {e}")

        self.root.after(self.update_ms, self._tick)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ----------------------------
# Entry point
# ----------------------------
def main():
    fs = Foresight(cache_file="foresight_cache.bin")
    gui = ForesightGUI(fs, update_ms=1000)
    gui.run()

if __name__ == "__main__":
    main()

