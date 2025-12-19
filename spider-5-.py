#!/usr/bin/env python3
"""
netmesh_mon_gui.py

Single-file NetMesh monitor + Tkinter GUI:

Engine:
- Autoloader for required libs
- IDS-style detectors
- Simple AI scoring (IsolationForest)
- Network scanner
- Active probes
- Mesh heartbeat + peer discovery
- Telemetry-driven self-improving detectors
- HTTP API for status, probes, detector toggling, host risk (/predict/hosts)

GUI:
- Dark theme
- Live alerts
- Signal stats
- Live mesh peers (via /status)
- Probe controls (HTTP + TCP)
- Detector toggles (via /detectors & /detectors/toggle)
- Alert-count bar graph (matplotlib if available)
- Host risk tab (via /predict/hosts)
"""

import sys, os, time, signal, subprocess, logging, json, threading, queue, hashlib, math, re, argparse, socket, ipaddress
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque

RUNNING = True

# =============================================================================
# Autoloader for required libraries
# =============================================================================

REQUIRED_LIBS = {
    "yaml": ["pyyaml", "pyyaml==6.0.2"],
    "requests": ["requests", "requests==2.32.3"],
    "scapy": ["scapy", "scapy==2.5.0"],
    "pyshark": ["pyshark", "pyshark==0.6"],
    "numpy": ["numpy", "numpy==2.1.2"],
    "pandas": ["pandas", "pandas==2.2.3"],
    "sklearn": ["scikit-learn", "scikit-learn==1.5.2"],
    "matplotlib": ["matplotlib"],  # optional for GUI graphs
}

def _install_package(pkg):
    for attempt in range(1, 4):
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(attempt * 0.5)
    return False

def _load_or_install(libname, candidates, log):
    import importlib
    for candidate in candidates:
        try:
            return importlib.import_module(libname)
        except Exception:
            pass
        log.info({"event": "autoloader_install_attempt", "package": candidate})
        if _install_package(candidate):
            try:
                return importlib.import_module(libname)
            except Exception:
                pass
    log.error({"event": "autoloader_failed", "library": libname})
    return None

def autoload_all(log):
    loaded = {}
    for libname, candidates in REQUIRED_LIBS.items():
        loaded[libname] = _load_or_install(libname, candidates, log)
    return loaded

# =============================================================================
# Default config
# =============================================================================

DEFAULT_CONFIG = {
    "service": {
        "interface": "auto",
        "backend": "auto",  # scapy | pyshark | auto
        "capture_filter": "",
        "max_queue": 5000,
        "heartbeat_sec": 10,
        "ai_retrain_interval_sec": 1800,
        "scan_interval_sec": 900,
        "scanner_timeout_sec": 0.7,
        "scanner_max_ports": 100,
        "mesh_port": 8787,
        "cidr": "auto",
        "broadcast_port": 9787,
        "broadcast_interval_sec": 15,
    },
    "logging": {
        "level": "INFO",
        "file_path": "logs/alerts.jsonl",
        "rotate_max_mb": 20,
        "rotate_keep": 5,
        "console": True,
        "webhook": {"enabled": False, "url": "", "headers": {}},
    },
    "alerts": {"min_severity": 2, "dedup_window_sec": 120},
    "plugins": {
        "ports_anomaly": {
            "enabled": True,
            "allowed_tcp": [80, 443, 22, 53],
            "allowed_udp": [53, 123],
        },
        "payload_signals": {
            "enabled": True,
            "keywords": [
                "cmd.exe",
                "powershell",
                "nc ",
                "wget ",
                "curl ",
                "sh -c",
                "python -c",
                "node -e",
            ],
            "base64_min_len": 24,
            "hex_min_len": 24,
        },
        "beaconing": {
            "enabled": True,
            "window_sec": 300,
            "min_repeats": 5,
            "jitter_tolerance_sec": 3,
        },
        "dns_suspicion": {
            "enabled": True,
            "entropy_threshold": 4.0,
            "long_label_len": 20,
        },
        "hidden_text_commands": {
            "enabled": True,
            "cmd_keywords": [
                "run",
                "exec",
                "execute",
                "cmd",
                "powershell",
                "bash",
                "sh -c",
                "wget",
                "curl",
                "nc ",
                "python -c",
                "node -e",
            ],
        },
    },
    "ai": {
        "enabled": True,
        "bootstrap_samples": 400,
        "contamination": 0.02,
        "weights": {"rule": 0.6, "iforest": 0.3, "baseline": 0.1},
        "baseline_scale": 3.0,
    },
    "self_improve": {
        "enabled": True,
        "extensions_dir": "extensions",
        "auto_load": True,
        "min_alerts": 40,
        "max_extensions": 20,
        "generation_interval_sec": 60,
    },
    "probes": {
        "queue_max": 1000,
        "rate_limit_per_host_sec": 5,
        "timeout_sec": 1.0,
    },
}

# =============================================================================
# Logging
# =============================================================================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time()),
            "level": record.levelname,
            "msg": record.msg if isinstance(record.msg, dict) else {"text": record.getMessage()},
        }
        return json.dumps(payload, ensure_ascii=False)

class WebhookHandler(logging.Handler):
    def __init__(self, url, headers=None, timeout=2, requests_mod=None):
        super().__init__()
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.requests = requests_mod

    def emit(self, record):
        if not self.requests:
            return
        try:
            payload = record.msg if isinstance(record.msg, dict) else {"text": record.getMessage()}
            self.requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        except Exception:
            pass

def get_logger(cfg, requests_mod=None):
    os.makedirs(os.path.dirname(cfg["file_path"]), exist_ok=True)
    log = logging.getLogger("netmesh")
    log.setLevel(getattr(logging, cfg["level"]))
    log.handlers.clear()

    fh = RotatingFileHandler(
        cfg["file_path"],
        maxBytes=cfg["rotate_max_mb"] * 1024 * 1024,
        backupCount=cfg["rotate_keep"],
    )
    fh.setFormatter(JsonFormatter())
    log.addHandler(fh)

    if cfg.get("console", True):
        ch = logging.StreamHandler()
        ch.setFormatter(JsonFormatter())
        log.addHandler(ch)

    wb = cfg.get("webhook", {})
    if wb.get("enabled") and requests_mod:
        log.addHandler(WebhookHandler(wb["url"], headers=wb.get("headers", {}), requests_mod=requests_mod))
    return log

def json_safe(val):
    try:
        return str(val)
    except Exception:
        return "<unserializable>"

# =============================================================================
# Config + shutdown helpers
# =============================================================================

def deep_copy(obj):
    return json.loads(json.dumps(obj))

def load_config(path, yaml_mod):
    cfg = deep_copy(DEFAULT_CONFIG)
    if path:
        if not yaml_mod:
            print("[config] yaml not available, using defaults")
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            user = yaml_mod.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg

def handle_shutdown(signum, frame):
    global RUNNING
    RUNNING = False

# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    def __init__(self, cfg, log):
        self.log = log
        self.min_severity = cfg.get("min_severity", 2)
        self.dedup_window = cfg.get("dedup_window_sec", 120)
        self.recent = {}
        self.counts = defaultdict(int)
        self.payload_samples = deque(maxlen=1500)
        self.dns_labels = deque(maxlen=1500)
        self.port_hits = defaultdict(int)
        self.beacon_pairs = defaultdict(int)
        self.alert_history = deque(maxlen=5000)

    def _fingerprint(self, ev):
        key = (
            json_safe(ev.get("signal", ""))
            + json_safe(ev.get("src", ""))
            + json_safe(ev.get("dst", ""))
            + json_safe(ev.get("extra", ""))
        )
        return hashlib.sha256(key.encode()).hexdigest()

    def _is_duplicate(self, fp):
        now = time.time()
        prev = self.recent.get(fp)
        if prev and (now - prev) < self.dedup_window:
            return True
        self.recent[fp] = now
        return False

    def ingest(self, ev):
        sev = int(ev.get("severity", 1))
        if sev < self.min_severity:
            return
        fp = self._fingerprint(ev)
        if self._is_duplicate(fp):
            return
        sig = ev.get("signal", "unknown")
        self.counts[sig] += 1
        self.alert_history.append({"ts": time.time(), "event": ev})

        if sig in ("keyword_in_payload", "base64_blob", "hex_blob", "hidden_text_command"):
            p = ev.get("payload", "")
            if p:
                self.payload_samples.append(str(p)[:1024])
        if sig == "suspicious_dns_label":
            for d in ev.get("details", []):
                lbl = d.get("label", "")
                if lbl:
                    self.dns_labels.append(lbl)
        if sig in ("unusual_tcp_port", "unusual_udp_port"):
            dp = ev.get("dst_port")
            if isinstance(dp, int):
                self.port_hits[dp] += 1
        if sig == "periodic_beaconing":
            key = (ev.get("src", ""), ev.get("dst", ""))
            self.beacon_pairs[key] += 1

        self.log.info({"event": "alert", "severity": sev, "data": ev})

    def flush(self):
        pass

# =============================================================================
# Host-level temporal profiling and risk scoring
# =============================================================================

class HostProfiler:
    """
    Maintains rolling statistics per host over time windows.
    Feeds HostRiskModel.
    """
    def __init__(self, log, window_sec=600):
        self.log = log
        self.window = window_sec
        self.history = defaultdict(deque)  # host -> deque of feature dicts

    def observe_packet(self, src, dst, length, anomaly=False, dns_entropy=None, dst_port=None, beacon=False):
        now = time.time()
        for host, role in ((src, "src"), (dst, "dst")):
            if not host:
                continue
            feats = {
                "ts": now,
                "packets": 1,
                "bytes": length or 0,
                "anomalies": 1 if anomaly and role == "src" else 0,
                "dns_entropy": dns_entropy if dns_entropy is not None and role == "src" else 0.0,
                "ports": {dst_port} if dst_port and role == "src" else set(),
                "beacon": 1 if beacon and role == "src" else 0,
            }
            self.history[host].append(feats)
            self._prune(host, now)

    def _prune(self, host, now):
        dq = self.history[host]
        cutoff = now - self.window
        while dq and dq[0]["ts"] < cutoff:
            dq.popleft()

    def snapshot_features(self):
        """
        Build aggregate feature vector per host.
        Returns dict: host -> feature dict.
        """
        snap = {}
        now = time.time()
        for host, dq in list(self.history.items()):
            self._prune(host, now)
            if not dq:
                continue
            packets = sum(x["packets"] for x in dq)
            bytes_ = sum(x["bytes"] for x in dq)
            anomalies = sum(x["anomalies"] for x in dq)
            beacons = sum(x["beacon"] for x in dq)
            dns_vals = [x["dns_entropy"] for x in dq if x["dns_entropy"] > 0]
            avg_dns = sum(dns_vals) / len(dns_vals) if dns_vals else 0.0
            ports = set()
            for x in dq:
                ports |= x["ports"]
            uniq_ports = len(ports)
            snap[host] = {
                "packets": packets,
                "bytes": bytes_,
                "anomalies": anomalies,
                "beacons": beacons,
                "avg_dns_entropy": avg_dns,
                "uniq_ports": uniq_ports,
                "last_ts": dq[-1]["ts"],
            }
        return snap

class HostRiskModel:
    """
    Uses snapshot features to compute host risk scores.
    If numpy/sklearn available, uses IsolationForest; otherwise uses heuristic.
    """
    def __init__(self, log, np_mod=None, IF_cls=None):
        self.log = log
        self.np = np_mod
        self.IF = IF_cls
        self.model = None
        self.hosts = []
        self.risks = {}  # host -> 0-100

    def update(self, snapshots):
        if not snapshots:
            self.risks = {}
            return
        self.hosts = list(snapshots.keys())
        X = []
        for h in self.hosts:
            f = snapshots[h]
            X.append([
                f["packets"],
                f["bytes"],
                f["anomalies"],
                f["beacons"],
                f["avg_dns_entropy"],
                f["uniq_ports"],
            ])
        # Heuristic fallback
        if self.np is None or self.IF is None:
            scores = []
            for feats in X:
                packets, bytes_, anomalies, beacons, avg_dns, uniq_ports = feats
                score = (
                    0.2 * math.log1p(packets) +
                    0.2 * math.log1p(bytes_) +
                    0.3 * anomalies +
                    0.2 * beacons +
                    0.05 * avg_dns +
                    0.05 * uniq_ports
                )
                scores.append(score)
            max_s = max(scores) if scores else 1.0
            self.risks = {
                h: int(min(100, (s / max_s) * 100.0))
                for h, s in zip(self.hosts, scores)
            }
            return
        # IsolationForest
        try:
            X_arr = self.np.array(X, dtype=float)
            if self.model is None or len(self.hosts) > len(self.risks) + 5:
                self.model = self.IF(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=42,
                )
                self.model.fit(X_arr)
            scores = -self.model.decision_function(X_arr)
            max_s = max(scores) if len(scores) else 1.0
            self.risks = {
                h: int(min(100, max(0, (s / max_s) * 100.0)))
                for h, s in zip(self.hosts, scores)
            }
        except Exception as e:
            self.log.error({"event": "host_risk_error", "error": str(e)})
            self.risks = {h: 10 for h in self.hosts}

    def get_risks(self):
        return self.risks

# =============================================================================
# Capture abstraction
# =============================================================================

class Capture:
    def __init__(self, cfg, log):
        self.log = log
        self.iface = cfg.get("interface", "auto")
        self.backend = cfg.get("backend", "auto")
        self.filter = cfg.get("capture_filter", "")
        self.q = queue.Queue(maxsize=cfg.get("max_queue", 5000))
        self.stop = False
        self.worker = None
        self._init_backend()

    def _init_backend(self):
        backend = self.backend
        if backend == "auto":
            try:
                import pyshark  # noqa
                backend = "pyshark"
            except Exception:
                backend = "scapy"
        self.backend = backend
        self.log.info({"event": "capture_backend", "backend": backend})

        if self.iface == "auto":
            self.iface = self._guess_iface()
        self.log.info({"event": "capture_interface", "iface": self.iface})

        if backend == "pyshark":
            self._start_pyshark()
        else:
            self._start_scapy()

    def _guess_iface(self):
        try:
            return list(socket.if_nameindex())[0][1]
        except Exception:
            return "eth0"

    def _start_pyshark(self):
        import pyshark
        def worker():
            while not self.stop and RUNNING:
                try:
                    cap = pyshark.LiveCapture(
                        interface=self.iface,
                        display_filter=None,
                        bpf_filter=self.filter or None,
                    )
                    for pkt in cap.sniff_continuously():
                        if self.stop or not RUNNING:
                            break
                        try:
                            self.q.put(pkt, timeout=0.1)
                        except queue.Full:
                            continue
                except Exception as e:
                    self.log.error({"event": "pyshark_error", "error": str(e)})
                    time.sleep(1)
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _start_scapy(self):
        from scapy.all import sniff

        def scapy_cb(pkt):
            try:
                self.q.put(pkt, timeout=0.1)
            except queue.Full:
                pass

        def worker():
            while not self.stop and RUNNING:
                try:
                    sniff(
                        iface=self.iface,
                        prn=scapy_cb,
                        store=False,
                        filter=self.filter or None,
                    )
                except Exception as e:
                    self.log.error({"event": "scapy_error", "error": str(e)})
                    time.sleep(1)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def next_packet(self, timeout=0.5):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.stop = True

# =============================================================================
# Detectors
# =============================================================================

DETECTOR_ENABLED = {}  # runtime toggles

class PortsAnomalyDetector:
    name = "ports_anomaly"
    def __init__(self, cfg, log):
        self.log = log
        self.allowed_tcp = set(cfg.get("allowed_tcp", []))
        self.allowed_udp = set(cfg.get("allowed_udp", []))

    def process(self, pkt):
        try:
            if hasattr(pkt, "transport_layer"):
                layer = pkt.transport_layer
                src = getattr(getattr(pkt, "ip", None), "src", "")
                dst = getattr(getattr(pkt, "ip", None), "dst", "")
                if layer == "TCP":
                    dport = int(getattr(pkt.tcp, "dstport", -1))
                    if dport > 0 and dport not in self.allowed_tcp:
                        return [{
                            "signal": "unusual_tcp_port",
                            "severity": 2,
                            "dst_port": dport,
                            "src": src,
                            "dst": dst,
                        }]
                elif layer == "UDP":
                    dport = int(getattr(pkt.udp, "dstport", -1))
                    if dport > 0 and dport not in self.allowed_udp:
                        return [{
                            "signal": "unusual_udp_port",
                            "severity": 2,
                            "dst_port": dport,
                            "src": src,
                            "dst": dst,
                        }]
            else:
                src = dst = ""
                if hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                    src = pkt["IP"].src
                    dst = pkt["IP"].dst
                if hasattr(pkt, "haslayer") and pkt.haslayer("TCP"):
                    dport = int(pkt["TCP"].dport)
                    if dport not in self.allowed_tcp:
                        return [{
                            "signal": "unusual_tcp_port",
                            "severity": 2,
                            "dst_port": dport,
                            "src": src,
                            "dst": dst,
                        }]
                if hasattr(pkt, "haslayer") and pkt.haslayer("UDP"):
                    dport = int(pkt["UDP"].dport)
                    if dport not in self.allowed_udp:
                        return [{
                            "signal": "unusual_udp_port",
                            "severity": 2,
                            "dst_port": dport,
                            "src": src,
                            "dst": dst,
                        }]
        except Exception:
            return None
        return None

class PayloadSignalsDetector:
    name = "payload_signals"
    def __init__(self, cfg, log):
        self.log = log
        self.keywords = set(cfg.get("keywords", []))
        self.b64_min = int(cfg.get("base64_min_len", 24))
        self.hex_min = int(cfg.get("hex_min_len", 24))
        self.b64rx = re.compile(r"[A-Za-z0-9+/=]{%d,}" % self.b64_min)
        self.hexrx = re.compile(r"[A-Fa-f0-9]{%d,}" % self.hex_min)

    def _payload(self, pkt):
        try:
            if hasattr(pkt, "highest_layer"):
                if hasattr(pkt, "tcp") and hasattr(pkt.tcp, "payload"):
                    return str(pkt.tcp.payload)
                if hasattr(pkt, "udp") and hasattr(pkt.udp, "payload"):
                    return str(pkt.udp.payload)
            else:
                if hasattr(pkt, "haslayer") and pkt.haslayer("Raw"):
                    return pkt["Raw"].load.decode(errors="ignore")
        except Exception:
            return ""
        return ""

    def process(self, pkt):
        p = self._payload(pkt)
        if not p:
            return None
        findings = []
        for kw in self.keywords:
            if kw in p:
                findings.append({
                    "signal": "keyword_in_payload",
                    "severity": 3,
                    "keyword": kw,
                    "payload": p[:256],
                })
        for m in self.b64rx.findall(p):
            findings.append({
                "signal": "base64_blob",
                "severity": 2,
                "len": len(m),
                "payload": m[:256],
            })
        for m in self.hexrx.findall(p):
            findings.append({
                "signal": "hex_blob",
                "severity": 2,
                "len": len(m),
                "payload": m[:256],
            })
        if not findings:
            return None

        src = dst = ""
        try:
            if hasattr(pkt, "ip"):
                src = getattr(pkt.ip, "src", "")
                dst = getattr(pkt.ip, "dst", "")
            elif hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                src = pkt["IP"].src
                dst = pkt["IP"].dst
        except Exception:
            pass
        for f in findings:
            f["src"] = src
            f["dst"] = dst
        return findings

class BeaconingDetector:
    name = "beaconing"
    def __init__(self, cfg, log):
        self.log = log
        self.win = int(cfg.get("window_sec", 300))
        self.min_rep = int(cfg.get("min_repeats", 5))
        self.jitter = int(cfg.get("jitter_tolerance_sec", 3))
        self.seen = defaultdict(list)

    def process(self, pkt):
        ts = time.time()
        src = dst = ""
        try:
            if hasattr(pkt, "ip"):
                src = getattr(pkt.ip, "src", "")
                dst = getattr(pkt.ip, "dst", "")
            elif hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                src = pkt["IP"].src
                dst = pkt["IP"].dst
        except Exception:
            return None
        if not src or not dst:
            return None
        key = (src, dst)
        arr = self.seen[key]
        arr.append(ts)
        cutoff = ts - self.win
        while arr and arr[0] < cutoff:
            arr.pop(0)
        if len(arr) >= self.min_rep:
            intervals = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
            if intervals:
                mean = sum(intervals) / len(intervals)
                devs = [abs(x - mean) for x in intervals]
                if sum(1 for d in devs if d <= self.jitter) >= max(1, len(devs)-1):
                    return [{
                        "signal": "periodic_beaconing",
                        "severity": 4,
                        "src": src,
                        "dst": dst,
                        "period_sec": round(mean, 2),
                    }]
        return None

def shannon_entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    probs = [v / len(s) for v in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

class DnsSuspicionDetector:
    name = "dns_suspicion"
    def __init__(self, cfg, log):
        self.log = log
        self.th = float(cfg.get("entropy_threshold", 4.0))
        self.long_label = int(cfg.get("long_label_len", 20))

    def process(self, pkt):
        domain = None
        try:
            if hasattr(pkt, "dns") and hasattr(pkt.dns, "qry_name"):
                domain = pkt.dns.qry_name
            elif hasattr(pkt, "haslayer") and pkt.haslayer("DNS"):
                domain = pkt["DNS"].qd.qname.decode()
        except Exception:
            domain = None
        if not domain:
            return None
        domain = str(domain)
        labels = domain.split(".")
        suspicious = []
        for L in labels:
            ent = shannon_entropy(L)
            if ent >= self.th or len(L) >= self.long_label:
                suspicious.append({"label": L, "entropy": round(ent, 2)})
        if suspicious:
            return [{
                "signal": "suspicious_dns_label",
                "severity": 3,
                "domain": domain,
                "details": suspicious,
            }]
        return None

class HiddenTextCommandsDetector:
    name = "hidden_text_commands"
    def __init__(self, cfg, log):
        self.log = log
        self.cmd_keywords = set(cfg.get("cmd_keywords", []))
        self.rx_zero_width = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
        self.rx_html_comment = re.compile(r"<!--(.*?)-->", re.DOTALL | re.IGNORECASE)
        self.rx_js_eval = re.compile(r"(eval|Function|atob|decodeURIComponent)\s*\(", re.IGNORECASE)
        self.rx_js_unicode = re.compile(r"\\u[0-9A-Fa-f]{4}")
        self.rx_hex_blob = re.compile(r"[A-Fa-f0-9]{24,}")
        self.max_report = 512

    def _payload(self, pkt):
        try:
            if hasattr(pkt, "highest_layer"):
                if hasattr(pkt, "http") and hasattr(pkt.http, "file_data"):
                    return str(pkt.http.file_data)
                if hasattr(pkt, "data") and hasattr(pkt.data, "data"):
                    return str(pkt.data.data)
            else:
                if hasattr(pkt, "haslayer") and pkt.haslayer("Raw"):
                    return pkt["Raw"].load.decode(errors="ignore")
        except Exception:
            return ""
        return ""

    def _deobfuscate(self, s):
        s = self.rx_zero_width.sub("", s)
        for m in self.rx_html_comment.finditer(s):
            content = m.group(1)
            if any(kw in content for kw in self.cmd_keywords):
                s += "\n" + content
        return s

    def process(self, pkt):
        p = self._payload(pkt)
        if not p:
            return None
        raw = p
        p = self._deobfuscate(p)
        hits = []
        if self.rx_js_eval.search(p) or self.rx_js_unicode.search(p):
            hits.append("obfuscated_js")
        if self.rx_hex_blob.search(p):
            hits.append("hex_obfuscation")
        lower = p.lower()
        for kw in self.cmd_keywords:
            if kw in lower:
                hits.append("cmd_keyword:" + kw)
        if not hits:
            return None

        src = dst = ""
        try:
            if hasattr(pkt, "ip"):
                src = getattr(pkt.ip, "src", "")
                dst = getattr(pkt.ip, "dst", "")
            elif hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                src = pkt["IP"].src
                dst = pkt["IP"].dst
        except Exception:
            pass

        return [{
            "signal": "hidden_text_command",
            "severity": 4,
            "src": src,
            "dst": dst,
            "indicators": hits,
            "payload": raw[:self.max_report],
        }]

def load_detectors(cfg_plugins, log):
    dets = []
    if cfg_plugins.get("ports_anomaly", {}).get("enabled", True):
        d = PortsAnomalyDetector(cfg_plugins["ports_anomaly"], log)
        dets.append(d); DETECTOR_ENABLED[d.name] = True
    if cfg_plugins.get("payload_signals", {}).get("enabled", True):
        d = PayloadSignalsDetector(cfg_plugins["payload_signals"], log)
        dets.append(d); DETECTOR_ENABLED[d.name] = True
    if cfg_plugins.get("beaconing", {}).get("enabled", True):
        d = BeaconingDetector(cfg_plugins["beaconing"], log)
        dets.append(d); DETECTOR_ENABLED[d.name] = True
    if cfg_plugins.get("dns_suspicion", {}).get("enabled", True):
        d = DnsSuspicionDetector(cfg_plugins["dns_suspicion"], log)
        dets.append(d); DETECTOR_ENABLED[d.name] = True
    if cfg_plugins.get("hidden_text_commands", {}).get("enabled", True):
        d = HiddenTextCommandsDetector(cfg_plugins["hidden_text_commands"], log)
        dets.append(d); DETECTOR_ENABLED[d.name] = True
    log.info({"event": "detectors_loaded", "count": len(dets)})
    return dets

# =============================================================================
# AI judge
# =============================================================================

class AIJudge:
    def __init__(self, cfg, log, np_mod, IsolationForest_cls):
        self.log = log
        self.np = np_mod
        self.IF = IsolationForest_cls
        self.enabled = bool(cfg.get("enabled", True) and self.IF and self.np is not None)
        self.bootstrap = int(cfg.get("bootstrap_samples", 400))
        self.contamination = float(cfg.get("contamination", 0.02))
        self.model = None
        self.features = []
        self.trained = False

    def _features_from_pkt(self, pkt):
        length = 0
        proto = 0
        src = dst = ""
        try:
            if hasattr(pkt, "length"):
                length = int(pkt.length)
            elif hasattr(pkt, "wirelen"):
                length = int(pkt.wirelen)
        except Exception:
            pass
        try:
            if hasattr(pkt, "ip"):
                src = getattr(pkt.ip, "src", "")
                dst = getattr(pkt.ip, "dst", "")
                proto = int(getattr(pkt.ip, "proto", 0))
            elif hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                ip = pkt["IP"]
                src = ip.src
                dst = ip.dst
                proto = int(ip.proto)
        except Exception:
            pass
        return [length, proto], f"{src}>{dst}"

    def observe(self, pkt):
        if not self.enabled:
            return None
        feats, fid = self._features_from_pkt(pkt)
        self.features.append((feats, fid, time.time()))
        if len(self.features) > self.bootstrap * 4:
            self.features = self.features[-self.bootstrap * 4 :]
        return feats, fid

    def maybe_train(self):
        if not self.enabled or self.trained:
            return
        if len(self.features) < self.bootstrap or self.np is None:
            return
        X = self.np.array([f[0] for f in self.features])
        try:
            self.model = self.IF(
                n_estimators=120,
                contamination=self.contamination,
                random_state=42,
            )
            self.model.fit(X)
            self.trained = True
            self.log.info({"event": "ai_trained", "samples": len(X)})
        except Exception as e:
            self.log.error({"event": "ai_train_error", "error": str(e)})
            self.enabled = False

    def score(self, obs):
        if not self.enabled or not self.trained or self.model is None:
            return {"iforest": 0.0}
        feats, fid = obs
        try:
            X = self.np.array(feats, dtype=float).reshape(1, -1)
            score = -float(self.model.decision_function(X)[0])
            return {"iforest": score, "fid": fid}
        except Exception as e:
            self.log.error({"event": "ai_score_error", "error": str(e)})
            return {"iforest": 0.0}

def ensemble_severity(rule_sev, ai_scores, weights, baseline_scale):
    if not ai_scores:
        return rule_sev, {"rule": rule_sev, "iforest": 0.0, "baseline": 0.0}
    w_rule = float(weights.get("rule", 0.6))
    w_if = float(weights.get("iforest", 0.3))
    w_base = float(weights.get("baseline", 0.1))
    ifor = float(ai_scores.get("iforest", 0.0))
    base = 1.0 if rule_sev >= 3 else 0.5
    raw = w_rule * rule_sev + w_if * ifor + w_base * base * baseline_scale
    sev = max(1, min(5, int(round(raw))))
    return sev, {"rule": rule_sev, "iforest": ifor, "baseline": base, "raw": raw}

# =============================================================================
# Active probes
# =============================================================================

class ProbeAPI:
    def __init__(self, cfg, log, pipeline, requests_mod):
        self.log = log
        self.pipeline = pipeline
        self.requests = requests_mod
        self.q = queue.Queue(maxsize=cfg.get("queue_max", 1000))
        self.timeout = float(cfg.get("timeout_sec", 1.0))
        self.rate = int(cfg.get("rate_limit_per_host_sec", 5))
        self.last_probe = {}
        self.stop = False

    def enqueue_http(self, url):
        self._enqueue({"type": "http", "url": url})

    def enqueue_tcp(self, host, port):
        self._enqueue({"type": "tcp", "host": host, "port": int(port)})

    def _enqueue(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self.log.error({"event": "probe_queue_full"})

    def _allow_host(self, host):
        now = time.time()
        t = self.last_probe.get(host, 0)
        if now - t < self.rate:
            return False
        self.last_probe[host] = now
        return True

    def worker(self):
        while not self.stop and RUNNING:
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job["type"] == "http":
                    self._do_http(job["url"])
                elif job["type"] == "tcp":
                    self._do_tcp(job["host"], job["port"])
            except Exception as e:
                self.log.error({"event": "probe_error", "error": str(e), "job": job})

    def _do_http(self, url):
        if not self.requests:
            return
        host = url.split("://")[-1].split("/")[0]
        if not self._allow_host(host):
            return
        try:
            r = self.requests.get(url, timeout=self.timeout)
            self.pipeline.ingest({
                "signal": "probe_http",
                "severity": 1,
                "url": url,
                "status": r.status_code,
                "len": len(r.content),
            })
        except Exception as e:
            self.pipeline.ingest({
                "signal": "probe_http_fail",
                "severity": 2,
                "url": url,
                "error": str(e),
            })

    def _do_tcp(self, host, port):
        if not self._allow_host(host):
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            try:
                if s.connect_ex((host, port)) == 0:
                    self.pipeline.ingest({
                        "signal": "probe_tcp_open",
                        "severity": 2,
                        "host": host,
                        "port": port,
                    })
            except Exception as e:
                self.pipeline.ingest({
                    "signal": "probe_tcp_error",
                    "severity": 2,
                    "host": host,
                    "port": port,
                    "error": str(e),
                })

    def close(self):
        self.stop = True

# =============================================================================
# Scanner
# =============================================================================

class NetworkScanner:
    def __init__(self, cfg, log, pipeline):
        self.log = log
        self.pipeline = pipeline
        self.interval = int(cfg.get("scan_interval_sec", 900))
        self.timeout = float(cfg.get("scanner_timeout_sec", 0.7))
        self.max_ports = int(cfg.get("scanner_max_ports", 100))
        self.last_scan = 0
        self.ports = [22, 80, 443, 3389, 8080, 8443][: self.max_ports]

    def _guess_cidr(self):
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            net = ipaddress.ip_network(ip + "/24", strict=False)
            return str(net)
        except Exception:
            return "192.168.1.0/24"

    def _iter_hosts(self, cidr):
        net = ipaddress.ip_network(cidr, strict=False)
        for ip in net.hosts():
            yield str(ip)

    def _scan_host(self, ip):
        open_ports = []
        for port in self.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                try:
                    if s.connect_ex((ip, port)) == 0:
                        open_ports.append(port)
                except Exception:
                    continue
        return open_ports

    def maybe_scan(self, cidr):
        now = time.time()
        if now - self.last_scan < self.interval:
            return
        self.last_scan = now
        cidr = cidr or self._guess_cidr()
        self.log.info({"event": "scanner_start", "cidr": cidr})
        for ip in self._iter_hosts(cidr):
            ports = self._scan_host(ip)
            if ports:
                self.pipeline.ingest({
                    "signal": "scan_open_ports",
                    "severity": 2,
                    "ip": ip,
                    "open_ports": ports,
                })
        self.log.info({"event": "scanner_done"})

# =============================================================================
# Self-improver + extensions
# =============================================================================

class SelfImprover:
    def __init__(self, cfg, log, pipeline, probes):
        self.log = log
        self.pipeline = pipeline
        self.probes = probes
        self.enabled = bool(cfg.get("enabled", True))
        self.dir = cfg.get("extensions_dir", "extensions")
        self.min_alerts = int(cfg.get("min_alerts", 40))
        self.max_ext = int(cfg.get("max_extensions", 20))
        self.interval = int(cfg.get("generation_interval_sec", 60))
        os.makedirs(self.dir, exist_ok=True)

    def _current_extensions(self):
        return [f for f in os.listdir(self.dir) if f.endswith(".py")]

    def _choose_signal(self):
        if not self.pipeline.counts:
            return None
        sig, cnt = max(self.pipeline.counts.items(), key=lambda kv: kv[1])
        if cnt < self.min_alerts:
            return None
        return sig

    def _write_extension(self, name, signal_name):
        fname = os.path.join(self.dir, f"{name}.py")
        if os.path.exists(fname):
            return
        tmpl = f'''"""
Auto-generated detector for signal pattern: {signal_name}
"""

class Detector:
    name = "ext_{name}"

    def __init__(self, cfg, log, probes=None):
        self.log = log
        self.probes = probes

    def process(self, pkt):
        # Placeholder: logic can be refined later
        return None
'''
        with open(fname, "w", encoding="utf-8") as f:
            f.write(tmpl)
        self.log.info({"event": "self_improve_new_ext", "file": fname})

    def propose_and_write(self):
        if not self.enabled:
            return
        if len(self._current_extensions()) >= self.max_ext:
            return
        sig = self._choose_signal()
        if not sig:
            return
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", sig)
        name = f"det_{safe}_{int(time.time())}"
        self._write_extension(name, sig)

def load_extensions(ext_dir, log, ext_cfg, probes):
    exts = []
    if not os.path.isdir(ext_dir):
        return exts
    for fname in os.listdir(ext_dir):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(ext_dir, fname)
        mod_name = f"ext_{fname[:-3]}"
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "Detector"):
                inst = mod.Detector(ext_cfg, log, probes=probes)
                exts.append(inst)
        except Exception as e:
            log.error({"event": "ext_load_error", "file": path, "error": str(e)})
    if exts:
        log.info({"event": "extensions_loaded", "count": len(exts)})
    return exts

# =============================================================================
# Mesh HTTP + broadcaster (API endpoints)
# =============================================================================

class MeshHTTPServer:
    def __init__(self, log, port, pipeline, scanner=None, probes=None, peers=None, detectors_ref=None, host_risk_model=None):
        self.log = log
        self.port = port
        self.pipeline = pipeline
        self.scanner = scanner
        self.probes = probes
        self.thread = None
        self.stop = False
        self.peers = peers if peers is not None else set()
        self.detectors_ref = detectors_ref
        self.host_risk_model = host_risk_model

    def start(self):
        import http.server, socketserver

        server_peers = self.peers
        pipeline = self.pipeline
        log = self.log
        probes = self.probes
        detectors_ref = self.detectors_ref
        host_risk_model = self.host_risk_model

        class Handler(http.server.BaseHTTPRequestHandler):
            def _read_json(self2):
                length = int(self2.headers.get("Content-Length", 0) or 0)
                if length <= 0:
                    return {}
                body = self2.rfile.read(length)
                try:
                    return json.loads(body.decode() or "{}")
                except Exception:
                    return {}

            def do_GET(self2):
                if self2.path == "/status":
                    body = {
                        "ts": int(time.time()),
                        "alerts_by_signal": dict(pipeline.counts),
                        "peers": list(server_peers),
                        "host_risks": host_risk_model.get_risks() if host_risk_model else {},
                    }
                    b = json.dumps(body).encode()
                    self2.send_response(200)
                    self2.send_header("Content-Type", "application/json")
                    self2.send_header("Content-Length", str(len(b)))
                    self2.end_headers()
                    self2.wfile.write(b)
                elif self2.path == "/detectors":
                    dets = []
                    if detectors_ref is not None:
                        for d in detectors_ref:
                            name = getattr(d, "name", "unknown")
                            dets.append({
                                "name": name,
                                "enabled": bool(DETECTOR_ENABLED.get(name, True)),
                            })
                    body = {"detectors": dets}
                    b = json.dumps(body).encode()
                    self2.send_response(200)
                    self2.send_header("Content-Type", "application/json")
                    self2.send_header("Content-Length", str(len(b)))
                    self2.end_headers()
                    self2.wfile.write(b)
                elif self2.path == "/predict/hosts":
                    risks = host_risk_model.get_risks() if host_risk_model else {}
                    body = {"host_risks": risks}
                    b = json.dumps(body).encode()
                    self2.send_response(200)
                    self2.send_header("Content-Type", "application/json")
                    self2.send_header("Content-Length", str(len(b)))
                    self2.end_headers()
                    self2.wfile.write(b)
                else:
                    self2.send_response(404)
                    self2.end_headers()

            def do_POST(self2):
                data = self2._read_json()
                path = self2.path
                if path == "/mesh/announce":
                    src = data.get("addr")
                    if src:
                        server_peers.add(src)
                        log.info({"event": "mesh_peer_added", "peer": src})
                    self2.send_response(200); self2.end_headers()
                elif path == "/probe/http":
                    url = data.get("url")
                    if url and probes:
                        probes.enqueue_http(url)
                    self2.send_response(200); self2.end_headers()
                elif path == "/probe/tcp":
                    host = data.get("host")
                    port = data.get("port")
                    if host and port and probes:
                        probes.enqueue_tcp(host, int(port))
                    self2.send_response(200); self2.end_headers()
                elif path == "/detectors/toggle":
                    name = data.get("name")
                    enabled = data.get("enabled")
                    if name is not None and enabled is not None:
                        DETECTOR_ENABLED[name] = bool(enabled)
                        log.info({"event": "detector_toggled", "name": name, "enabled": bool(enabled)})
                    self2.send_response(200); self2.end_headers()
                else:
                    self2.send_response(404)
                    self2.end_headers()

            def log_message(self2, fmt, *args):
                return

        def run():
            with socketserver.TCPServer(("", self.port), Handler) as httpd:
                self.log.info({"event": "mesh_http_start", "port": self.port})
                httpd.timeout = 1
                while not self.stop and RUNNING:
                    httpd.handle_request()
                self.log.info({"event": "mesh_http_stop"})

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop = True

class MeshBroadcaster:
    def __init__(self, log, port, mesh_port, interval):
        self.log = log
        self.port = port
        self.mesh_port = mesh_port
        self.interval = interval
        self.stop = False

    def announce_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        hostname = socket.gethostname()
        addr = socket.gethostbyname(hostname)
        def payload():
            return json.dumps({"addr": f"http://{addr}:{self.mesh_port}"}).encode()
        while not self.stop and RUNNING:
            try:
                sock.sendto(payload(), ("255.255.255.255", self.port))
            except Exception as e:
                self.log.error({"event": "mesh_broadcast_error", "error": str(e)})
            time.sleep(self.interval)
        sock.close()

    def listen_loop(self, peers_set):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.port))
        except Exception as e:
            self.log.error({"event": "mesh_listen_bind_error", "error": str(e)})
            return
        while not self.stop and RUNNING:
            try:
                data, _ = sock.recvfrom(2048)
                try:
                    msg = json.loads(data.decode())
                except Exception:
                    continue
                addr = msg.get("addr")
                if addr:
                    peers_set.add(addr)
            except Exception:
                time.sleep(0.5)
        sock.close()
# =============================================================================
# TKINTER GUI (Frontend)
# =============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess

try:
    import requests as gui_requests
except Exception:
    gui_requests = None

# matplotlib is optional; graphs disabled if missing
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None
    FigureCanvasTkAgg = None

class BackendController:
    def __init__(self, on_status):
        self.proc = None
        self.on_status = on_status
        self.stop_flag = False

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args=None):
        if self.is_running():
            self.on_status("Backend already running")
            return
        cmd = [sys.executable, sys.argv[0], "--engine"]
        if args:
            cmd += args
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.on_status("Backend started")
            threading.Thread(target=self._stderr_reader, daemon=True).start()
        except Exception as e:
            self.on_status(f"Failed to start backend: {e}")

    def _stderr_reader(self):
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            if self.stop_flag:
                break
            line = line.strip()
            if line:
                self.on_status(f"[backend] {line}")

    def stop(self):
        self.stop_flag = True
        if self.proc and self.is_running():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
                self.on_status("Backend stopped")
            except Exception:
                try:
                    self.proc.kill()
                    self.on_status("Backend killed")
                except Exception:
                    self.on_status("Failed to stop backend")
        self.proc = None
        self.stop_flag = False


class LogTailer:
    def __init__(self, path, queue_obj, on_status):
        self.path = path
        self.queue = queue_obj
        self.on_status = on_status
        self.stop_flag = False

    def start(self):
        self.stop_flag = False
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.stop_flag = True

    def _run(self):
        while not os.path.exists(self.path) and not self.stop_flag:
            time.sleep(0.5)
        if self.stop_flag:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)
                while not self.stop_flag:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.2)
                        f.seek(pos)
                        continue
                    try:
                        data = json.loads(line.strip())
                        self.queue.put(data)
                    except Exception:
                        self.on_status("Invalid JSON in log")
        except Exception as e:
            self.on_status(f"Log tailer error: {e}")


class NetmeshGUI(tk.Tk):
    def __init__(self, mesh_port=8787):
        super().__init__()
        self.title("NetMesh Monitor")
        self.geometry("1200x700")
        self.mesh_port = mesh_port

        self.backend = BackendController(self._status)
        self.log_queue = queue.Queue()
        self.log_tailer = LogTailer("logs/alerts.jsonl", self.log_queue, self._status)

        self.alert_counts = defaultdict(int)

        self._setup_dark_theme()
        self._build_ui()
        self._poll_log_queue()
        self._poll_status()

    def _setup_dark_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#1e1e1e"
        fg = "#f0f0f0"
        self.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background="#333333", foreground=fg)
        style.configure("TFrame", background=bg)
        style.configure("Treeview", background="#252526", foreground=fg, fieldbackground="#252526")
        style.map("TButton", background=[("active", "#444444")])

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=5, pady=5)

        self.btn_start = ttk.Button(top, text="Start", command=self.on_start)
        self.btn_start.pack(side="left", padx=3)

        self.btn_stop = ttk.Button(top, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=3)

        self.btn_reload = ttk.Button(top, text="Reload Extensions", command=self.on_reload)
        self.btn_reload.pack(side="left", padx=3)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=10)

        # Probe controls
        probe_frame = ttk.Frame(top)
        probe_frame.pack(side="right")
        ttk.Label(probe_frame, text="Probe HTTP URL:").grid(row=0, column=0, sticky="e")
        self.entry_probe_url = ttk.Entry(probe_frame, width=30)
        self.entry_probe_url.grid(row=0, column=1, padx=3)
        ttk.Button(probe_frame, text="Send", command=self.on_probe_http).grid(row=0, column=2, padx=3)

        ttk.Label(probe_frame, text="Probe TCP host:").grid(row=1, column=0, sticky="e")
        self.entry_probe_host = ttk.Entry(probe_frame, width=18)
        self.entry_probe_host.grid(row=1, column=1, sticky="w", padx=3)
        ttk.Label(probe_frame, text="port:").grid(row=1, column=2, sticky="e")
        self.entry_probe_port = ttk.Entry(probe_frame, width=6)
        self.entry_probe_port.grid(row=1, column=3, padx=3)
        ttk.Button(probe_frame, text="Send", command=self.on_probe_tcp).grid(row=1, column=4, padx=3)

        # Main split
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: alerts + graph
        left = ttk.Frame(main)
        ttk.Label(left, text="Alerts").pack(anchor="w")
        self.txt_alerts = scrolledtext.ScrolledText(left, wrap="none", height=18, bg="#1e1e1e", fg="#f0f0f0", insertbackground="#f0f0f0")
        self.txt_alerts.pack(fill="both", expand=True)

        graph_frame = ttk.Frame(left)
        graph_frame.pack(fill="both", expand=True, pady=(5, 0))
        ttk.Label(graph_frame, text="Alert Counts Graph").pack(anchor="w")

        self.graph_canvas = None
        self.graph_figure = None
        if Figure and FigureCanvasTkAgg:
            self.graph_figure = Figure(figsize=(5, 2.5), dpi=100)
            self.graph_ax = self.graph_figure.add_subplot(111)
            self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=graph_frame)
            self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(graph_frame, text="(matplotlib not available; graph disabled)").pack(anchor="w")

        main.add(left, weight=3)

        # Right: notebook
        right = ttk.Notebook(main)
        main.add(right, weight=2)

        # Tab 1: status (stats + peers)
        tab_status = ttk.Frame(right)
        right.add(tab_status, text="Status")

        ttk.Label(tab_status, text="Signal Stats").pack(anchor="w")
        self.tree_stats = ttk.Treeview(tab_status, columns=("count",), show="headings", height=8)
        self.tree_stats.heading("count", text="Count")
        self.tree_stats.column("count", width=80)
        self.tree_stats.pack(fill="x", expand=False, pady=3)

        ttk.Label(tab_status, text="Mesh Peers").pack(anchor="w", pady=(5, 0))
        self.lst_peers = tk.Listbox(tab_status, height=8, bg="#1e1e1e", fg="#f0f0f0")
        self.lst_peers.pack(fill="both", expand=True, pady=(0, 5))

        # Tab 2: detectors
        tab_det = ttk.Frame(right)
        right.add(tab_det, text="Detectors")

        ttk.Label(tab_det, text="Toggle detectors (via /detectors)").pack(anchor="w")
        self.detectors_frame = ttk.Frame(tab_det)
        self.detectors_frame.pack(fill="both", expand=True, pady=5)
        self.detector_vars = {}  # name -> BooleanVar

        # Tab 3: host risk
        tab_risk = ttk.Frame(right)
        right.add(tab_risk, text="Risk")

        ttk.Label(tab_risk, text="Host Risk Scores").pack(anchor="w")
        self.tree_risk = ttk.Treeview(tab_risk, columns=("risk", "host"), show="headings", height=12)
        self.tree_risk.heading("risk", text="Risk")
        self.tree_risk.heading("host", text="Host")
        self.tree_risk.column("risk", width=60, anchor="center")
        self.tree_risk.column("host", width=200)
        self.tree_risk.pack(fill="both", expand=True, pady=5)

    # backend control
    def on_start(self):
        self.backend.start()
        self.log_tailer.start()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._status("Backend starting...")

    def on_stop(self):
        self.backend.stop()
        self.log_tailer.stop()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._status("Stopped")

    def on_reload(self):
        self.on_stop()
        time.sleep(0.5)
        self.on_start()
        self._status("Reloaded (backend restarted)")

    # probes
    def on_probe_http(self):
        url = self.entry_probe_url.get().strip()
        if not url:
            return
        if not gui_requests:
            messagebox.showerror("Error", "requests module not available in GUI")
            return
        try:
            endpoint = f"http://127.0.0.1:{self.mesh_port}/probe/http"
            gui_requests.post(endpoint, json={"url": url}, timeout=2)
            self._status(f"Probe HTTP sent: {url}")
        except Exception as e:
            self._status(f"Probe HTTP error: {e}")

    def on_probe_tcp(self):
        host = self.entry_probe_host.get().strip()
        port = self.entry_probe_port.get().strip()
        if not host or not port:
            return
        if not gui_requests:
            messagebox.showerror("Error", "requests module not available in GUI")
            return
        try:
            endpoint = f"http://127.0.0.1:{self.mesh_port}/probe/tcp"
            gui_requests.post(endpoint, json={"host": host, "port": int(port)}, timeout=2)
            self._status(f"Probe TCP sent: {host}:{port}")
        except Exception as e:
            self._status(f"Probe TCP error: {e}")

    # log handling
    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._handle_log(msg)
        except queue.Empty:
            pass
        self.after(200, self._poll_log_queue)

    def _handle_log(self, msg):
        m = msg.get("msg", {})
        if isinstance(m, dict) and m.get("event") == "alert":
            data = m.get("data", {})
            self._append_alert(data)

    def _append_alert(self, data):
        ts = time.strftime("%H:%M:%S")
        sig = data.get("signal", "unknown")
        sev = data.get("severity", "?")
        src = data.get("src", "")
        dst = data.get("dst", "")
        line = f"[{ts}] sev={sev} {sig} {src} -> {dst}\n"
        self.txt_alerts.insert("end", line)
        self.txt_alerts.see("end")
        self._update_stats(sig)

    def _update_stats(self, sig):
        self.alert_counts[sig] += 1
        # tree
        for iid in self.tree_stats.get_children():
            if self.tree_stats.item(iid, "text") == sig:
                count = int(self.tree_stats.item(iid, "values")[0]) + 1
                self.tree_stats.item(iid, values=(count,))
                break
        else:
            self.tree_stats.insert("", "end", text=sig, values=(1,))
        # graph
        self._update_graph()

    def _update_graph(self):
        if not (self.graph_canvas and self.graph_figure):
            return
        ax = self.graph_ax
        ax.clear()
        if not self.alert_counts:
            self.graph_canvas.draw()
            return
        labels = list(self.alert_counts.keys())
        values = [self.alert_counts[k] for k in labels]
        ax.bar(labels, values, color="#569cd6")
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_xlabel("Signal")
        ax.set_title("Alert counts")
        self.graph_canvas.draw()

    # status polling: peers, detectors, host risks
    def _poll_status(self):
        if gui_requests:
            # status/peers
            try:
                endpoint = f"http://127.0.0.1:{self.mesh_port}/status"
                r = gui_requests.get(endpoint, timeout=1)
                if r.status_code == 200:
                    data = r.json()
                    peers = data.get("peers", [])
                    self._update_peers(peers)
            except Exception:
                pass
            # detectors
            try:
                endpoint = f"http://127.0.0.1:{self.mesh_port}/detectors"
                r = gui_requests.get(endpoint, timeout=1)
                if r.status_code == 200:
                    data = r.json()
                    self._update_detectors(data.get("detectors", []))
            except Exception:
                pass
            # host risks
            try:
                endpoint = f"http://127.0.0.1:{self.mesh_port}/predict/hosts"
                r = gui_requests.get(endpoint, timeout=1)
                if r.status_code == 200:
                    data = r.json()
                    risks = data.get("host_risks", {})
                    self._update_risk_table(risks)
            except Exception:
                pass
        self.after(3000, self._poll_status)

    def _update_peers(self, peers):
        self.lst_peers.delete(0, "end")
        for p in peers:
            self.lst_peers.insert("end", p)

    def _update_detectors(self, detectors):
        # if detector set changed, rebuild
        if len(detectors) != len(self.detector_vars):
            for w in self.detectors_frame.winfo_children():
                w.destroy()
            self.detector_vars.clear()
            for idx, d in enumerate(detectors):
                name = d.get("name", "unknown")
                var = tk.BooleanVar(value=bool(d.get("enabled", True)))
                chk = ttk.Checkbutton(
                    self.detectors_frame,
                    text=name,
                    variable=var,
                    command=lambda n=name, v=var: self._toggle_detector(n, v),
                )
                chk.grid(row=idx, column=0, sticky="w", pady=1)
                self.detector_vars[name] = var
        else:
            for d in detectors:
                name = d.get("name", "unknown")
                enabled = bool(d.get("enabled", True))
                if name in self.detector_vars:
                    self.detector_vars[name].set(enabled)

    def _toggle_detector(self, name, var):
        if not gui_requests:
            messagebox.showerror("Error", "requests module not available in GUI")
            return
        try:
            endpoint = f"http://127.0.0.1:{self.mesh_port}/detectors/toggle"
            gui_requests.post(endpoint, json={"name": name, "enabled": bool(var.get())}, timeout=2)
            self._status(f"Detector {name} -> {bool(var.get())}")
        except Exception as e:
            self._status(f"Toggle detector error: {e}")

    def _update_risk_table(self, risks):
        # risks: {host: score}
        for iid in self.tree_risk.get_children():
            self.tree_risk.delete(iid)
        for host, score in sorted(risks.items(), key=lambda kv: -kv[1]):
            self.tree_risk.insert("", "end", values=(score, host))

    def _status(self, text):
        self.status_var.set(text)

# =============================================================================
# Unified MAIN (CLI engine mode OR GUI mode)
# =============================================================================

def run_engine_or_gui():
    parser = argparse.ArgumentParser(description="NetMesh Monitor + GUI")
    parser.add_argument("--engine", action="store_true", help="Run engine only (no GUI)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iface", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--min-sev", type=int, default=None)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--ai-contamination", type=float, default=None)
    parser.add_argument("--cidr", type=str, default=None)
    args = parser.parse_args()

    if not args.engine:
        app = NetmeshGUI()
        app.mainloop()
        return

    # ENGINE MODE
    tmp_log = logging.getLogger("autoloader")
    if not tmp_log.handlers:
        tmp_log.setLevel(logging.INFO)
        tmp_log.addHandler(logging.StreamHandler())

    libs = autoload_all(tmp_log)
    global requests
    requests = libs.get("requests")
    yaml_mod = libs.get("yaml")
    np_mod = libs.get("numpy")
    sklearn_mod = libs.get("sklearn")

    IF_cls = None
    if sklearn_mod:
        try:
            from sklearn.ensemble import IsolationForest as IF_cls
        except Exception:
            IF_cls = None

    cfg = load_config(args.config, yaml_mod)

    if args.iface: cfg["service"]["interface"] = args.iface
    if args.backend: cfg["service"]["backend"] = args.backend
    if args.min_sev: cfg["alerts"]["min_severity"] = args.min_sev
    if args.no_ai: cfg["ai"]["enabled"] = False
    if args.ai_contamination is not None: cfg["ai"]["contamination"] = args.ai_contamination
    if args.cidr: cfg["service"]["cidr"] = args.cidr

    log = get_logger(cfg["logging"], requests_mod=requests)
    log.info({"event": "service_start", "pid": os.getpid()})

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    pipeline = Pipeline(cfg["alerts"], log)
    detectors = load_detectors(cfg["plugins"], log)
    capture = Capture(cfg["service"], log)

    host_profiler = HostProfiler(log)
    host_risk_model = HostRiskModel(log, np_mod, IF_cls)

    aijudge = None
    if cfg["ai"].get("enabled", True) and IF_cls and np_mod is not None:
        aijudge = AIJudge(cfg["ai"], log, np_mod, IF_cls)
    else:
        log.info({"event": "ai_disabled"})

    probes = ProbeAPI(cfg["probes"], log, pipeline, requests)
    scanner = NetworkScanner(cfg["service"], log, pipeline)
    peers = set()

    mesh = MeshHTTPServer(
        log,
        cfg["service"]["mesh_port"],
        pipeline,
        scanner,
        probes,
        peers,
        detectors_ref=detectors,
        host_risk_model=host_risk_model,
    )
    mesh.start()

    broadcaster = MeshBroadcaster(
        log,
        cfg["service"]["broadcast_port"],
        cfg["service"]["mesh_port"],
        cfg["service"]["broadcast_interval_sec"],
    )
    threading.Thread(target=broadcaster.announce_loop, daemon=True).start()
    threading.Thread(target=broadcaster.listen_loop, args=(peers,), daemon=True).start()

    improver = SelfImprover(cfg["self_improve"], log, pipeline, probes)
    extensions = load_extensions(improver.dir, log, {}, probes)

    weights = cfg["ai"]["weights"]
    baseline_scale = cfg["ai"]["baseline_scale"]
    iface_cidr = cfg["service"]["cidr"]

    def packet_worker():
        last_train = 0
        while RUNNING:
            pkt = capture.next_packet()
            if pkt is None:
                continue

            ai_scores = None
            if aijudge:
                obs = aijudge.observe(pkt)
                if obs:
                    ai_scores = aijudge.score(obs)

            # basic pkt info for host profiling
            length = 0
            src = dst = ""
            dst_port = None
            try:
                if hasattr(pkt, "length"):
                    length = int(pkt.length)
                elif hasattr(pkt, "wirelen"):
                    length = int(pkt.wirelen)
            except Exception:
                pass
            try:
                if hasattr(pkt, "ip"):
                    src = getattr(pkt.ip, "src", "")
                    dst = getattr(pkt.ip, "dst", "")
                elif hasattr(pkt, "haslayer") and pkt.haslayer("IP"):
                    ip_layer = pkt["IP"]
                    src = ip_layer.src
                    dst = ip_layer.dst
                if hasattr(pkt, "tcp"):
                    dst_port = int(getattr(pkt.tcp, "dstport", 0))
                elif hasattr(pkt, "haslayer") and pkt.haslayer("TCP"):
                    dst_port = int(pkt["TCP"].dport)
                elif hasattr(pkt, "haslayer") and pkt.haslayer("UDP"):
                    dst_port = int(pkt["UDP"].dport)
            except Exception:
                pass

            events = []
            for d in detectors + extensions:
                if not DETECTOR_ENABLED.get(getattr(d, "name", ""), True):
                    continue
                try:
                    ev = d.process(pkt)
                    if ev:
                        events.extend(ev if isinstance(ev, list) else [ev])
                except Exception as e:
                    log.error({"event": "detector_error", "error": str(e)})

            for ev in events:
                rule = ev.get("severity", 1)
                sev, breakdown = ensemble_severity(rule, ai_scores, weights, baseline_scale)
                ev["severity"] = sev
                ev["explain"] = breakdown

                # host profiling: anomaly + DNS/beacon info
                sig = ev.get("signal", "")
                dns_entropy = None
                beacon_flag = False
                if sig == "suspicious_dns_label":
                    dets = ev.get("details", [])
                    if dets:
                        dns_entropy = max(d.get("entropy", 0) for d in dets)
                if sig == "periodic_beaconing":
                    beacon_flag = True

                host_profiler.observe_packet(
                    src=ev.get("src", src),
                    dst=ev.get("dst", dst),
                    length=length,
                    anomaly=True,
                    dns_entropy=dns_entropy,
                    dst_port=ev.get("dst_port", dst_port),
                    beacon=beacon_flag,
                )

                pipeline.ingest(ev)

            if aijudge and time.time() - last_train > 10:
                last_train = time.time()
                aijudge.maybe_train()

        log.info({"event": "packet_worker_exit"})

    def scanner_worker():
        while RUNNING:
            scanner.maybe_scan(iface_cidr)
            for _ in range(10):
                if not RUNNING:
                    break
                time.sleep(1)
        log.info({"event": "scanner_worker_exit"})

    def improv_worker():
        while RUNNING:
            improver.propose_and_write()
            new_ext = load_extensions(improver.dir, log, {}, probes)
            names = {getattr(d, "name", "") for d in extensions}
            for e in new_ext:
                if getattr(e, "name", "") not in names:
                    extensions.append(e)
            for _ in range(improver.interval):
                if not RUNNING:
                    break
                time.sleep(1)
        log.info({"event": "improv_worker_exit"})

    def probe_worker():
        probes.worker()
        log.info({"event": "probe_worker_exit"})

    threading.Thread(target=packet_worker, daemon=True).start()
    threading.Thread(target=scanner_worker, daemon=True).start()
    threading.Thread(target=improv_worker, daemon=True).start()
    threading.Thread(target=probe_worker, daemon=True).start()

    hb = cfg["service"]["heartbeat_sec"]
    last_risk_update = 0
    while RUNNING:
        now = time.time()
        if now - last_risk_update > 15:
            snaps = host_profiler.snapshot_features()
            host_risk_model.update(snaps)
            last_risk_update = now
        log.info({
            "event": "heartbeat",
            "alerts": dict(pipeline.counts),
            "peers": list(peers),
            "host_risks": host_risk_model.get_risks(),
        })
        for _ in range(hb):
            if not RUNNING:
                break
            time.sleep(1)

    capture.close()
    mesh.close()
    probes.close()
    pipeline.flush()
    log.info({"event": "service_stop"})

# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run_engine_or_gui()


