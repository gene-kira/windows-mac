import tkinter as tk
import random
import time
import threading
import queue
import re
import datetime
import os
import json
import hashlib
import math
import secrets
from urllib.parse import urlparse
from collections import defaultdict, deque

# Optional libraries (used only when available and online)
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    BeautifulSoup = None
    REQUESTS_AVAILABLE = False

# ============================================================
# Configuration — Unrestricted domains, mission, security, privacy, Borg mesh
# ============================================================

CHECK_HOSTS = ["https://www.google.com", "https://example.com"]
CHECK_TIMEOUT = 3
CHECK_INTERVAL_SEC = 10

START_URLS = ["https://example.com/"]  # seed; add more freely
ALLOWED_DOMAINS = None  # None => unrestricted; all hosts allowed

MAX_DEPTH = 2
REQUEST_DELAY_SEC = (1.2, 2.4)
REQUEST_TIMEOUT_SEC = 8
BURST_LIMIT_REAL = 12
BURST_LIMIT_LOCAL = 10

MEMORY_FILE = "roaming_civilization_borg_mesh.json"
AUTO_SAVE_INTERVAL_SEC = 15

# Replication governance
REPLICATION_COOLDOWN_SEC = 8
BREAKTHROUGH_REQUIRED = True

# Logging and privacy
LOG_LEVEL = "normal"  # normal | minimal | forensic

# Security guardian policies
SECURITY_POLICIES = {
    "read_only_mode": True,       # narrative-only; never post or execute remote code
    "max_snippet_length": 3000,   # limit processed snippet size
    "entropy_threshold": 4.5,     # high-entropy blob suspicion
    "suspicious_patterns": [
        r"\bbase64\b", r"\bBEGIN RSA\b", r"-----BEGIN", r"\bmalware\b", r"\bexploit\b",
        r"\bbitcoin\b", r"\bseed\b", r"\bprivate key\b", r"\bpasswd\b", r"(?i)api[_\-]?key",
        r"(?i)document\.write", r"(?i)eval\(", r"(?i)atob\(", r"(?i)unescape\(", r"(?i)xhr|fetch"
    ],
    "format_signatures": {
        "json_hint": r"^\s*[\{\[]",
        "xml_hint": r"^\s*<\?xml|^\s*<\w+",
        "yaml_hint": r"^\s*[\w\-]+\s*:\s*",
        "binary_hint": r"[^\x09\x0A\x0D\x20-\x7E]"
    },
    "rate_limit_gui_ms": 500,     # minimum GUI refresh interval
    "quarantine_on_suspicion": True,
    # Privacy shield controls
    "encrypt_personal_data": True,
    "chameleon_no_personal_headers": True
}

# Borg communications configuration
BORG_COMMS_CONFIG = {
    "enable": True,
    "pool_size": 4,             # connection pool for faster access
    "retry_backoff": [0.4, 0.8, 1.6],  # exponential backoff steps
    "cache_max_entries": 128,   # simple LRU-like cache
    "glyph_alphabet": "ᚠᚡᚢᚣᚤᚥᚦᚧᚨᚩᚪᚫᚬᚭᚮᚯᚰᚱᚲᚳᚴᚵᚶᚷᚸᚹᚺᚻᚼᚽᚾᚿ",
    "mirror_mode": "palindromic",
    "chameleon_ciphers": ["shift", "xor", "hashpad"],
}

# Borg mesh modes
BORG_MESH_CONFIG = {
    "overlay_enabled": True,
    "max_corridors": 256,
    "scanner_threads": 2,
    "worker_threads": 2,
    "enforcer_threads": 1,
    "unknown_bias": 0.65,  # prefer unseen links to simulate discovery
}

# ============================================================
# Chameleon skins — headers never carry personal data
# ============================================================

CHAMELEON_SKINS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 Chrome/121.0 Mobile Safari/537.36",
]
LANG_PREFS = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "fr-FR,fr;q=0.7"]

SKIN_COLORS = {
    "Chrome": "#66ccff",
    "Firefox": "#99ffcc",
    "Safari": "#ffd966",
    "Android": "#cc99ff",
    "Default": "#66ccff"
}

def skin_identity(ua: str):
    ua_l = ua.lower()
    if "chrome" in ua_l and "mobile" not in ua_l:
        return "Chrome"
    if "firefox" in ua_l:
        return "Firefox"
    if "safari" in ua_l and "iphone" in ua_l:
        return "Safari"
    if "android" in ua_l or "mobile" in ua_l:
        return "Android"
    return "Default"

def chameleon_headers():
    ua = random.choice(CHAMELEON_SKINS)
    return {
        "User-Agent": ua,
        "Accept-Language": random.choice(LANG_PREFS),
        "Accept": "text/html,application/xhtml+xml"
    }, skin_identity(ua)

# ============================================================
# Vocabularies — expanded question codex
# ============================================================

HOLMES_QUESTIONS = [
    "What clue hides in the silence?",
    "Why does the trail vanish?",
    "Where does the evidence lead?",
    "What shadow conceals the truth?",
    "What pattern repeats in the noise?",
    "What anomaly breaks the expected?",
    "What rhythm hides in the noise?",
    "Which anomaly repeats across signals?",
    "Where does the pattern fracture?",
    "If this clue exists, what consequence follows?",
    "Why does this anomaly matter to the case?",
    "Who benefits if this trail is true?",
    "What echoes from the past shape this data?",
    "What future does this signal forecast?",
    "How does time distort the evidence?",
    "Where in the network does this shadow fall?",
    "Which gateway hides the scent?",
    "What port opens the next trail?",
    "What truth hides behind silence?",
    "What question remains unasked?",
    "What story does the data tell itself?",
    "Whose signature stains the margin of the noise?",
    "What is missing that should be here?",
    "What has appeared that should not exist?",
    "Where do footprints cross and diverge?",
    "What rule breaks only in rare moments?",
    "Which signal refuses to fade?",
    "What if the map is the territory?",
    "What happens when the trail circles back?",
    "Which shadow announces a light beyond?",
    "What scent lingers without source?",
    "What pattern survives all transformations?",
    "Where do anomalies become law?",
    "What is the question behind the answer?",
    "What evidence hides inside contradictions?"
]

DOG_ANSWERS = [
    "a scent of mystery",
    "footprints in the fog",
    "a whisper in the wind",
    "a trail of crumbs",
    "a pawprint on the doorstep",
    "mud on the mat, fresh and unaccounted",
    "scratches on the gate, recent and rough",
    "ash smudged on the windowsill",
    "a torn ribbon in the hedge",
    "a faint rustle behind curtains"
]

QUESTION_TEMPLATES = [
    "If {answer}, then what follows?",
    "How does {answer} change the case?",
    "Why does {answer} matter to the mystery?",
    "What new clue arises from {answer}?",
    "Where might {answer} be hiding?",
    "Who benefits if {answer} appears?",
    "What pattern is exposed when {answer} returns?",
    "Which gateway opens when {answer} persists?",
    "What contradiction dissolves if {answer} is true?",
    "How does time reshape {answer}?"
]

REFRAIN_STEPS = [
    "Is there anything more?",
    "Is there anything beneath this?",
    "Is there anything beyond this?",
    "Is there anything we’ve missed?",
    "Is there anything left unsaid?"
]

# Themes
THEME_KEYWORDS = {
    "scent": ["scent", "smell", "odor"],
    "shadow": ["shadow", "dark", "fog", "silence"],
    "trail": ["trail", "crumbs", "footprint", "footprints", "path", "link"],
    "signal": ["whisper", "rustle", "echo", "anomaly", "pattern"],
}

def detect_theme(text: str):
    a = (text or "").lower()
    for theme, words in THEME_KEYWORDS.items():
        if any(w in a for w in words):
            return theme
    return "misc"

# ============================================================
# Privacy shield — PII detection, redaction, and encryption vault
# ============================================================

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN-like
    re.compile(r"\b\d{10}\b"),                                # phone-like (simple)
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),                # email-like
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),                   # credit card-like (naive)
    re.compile(r"\b(?:[A-Za-z0-9]{32,64})\b")                 # token-like (API tokens, hashes)
]

class ProtectionEngine:
    def __init__(self):
        self._key = hashlib.sha256(secrets.token_bytes(32)).digest()
        self.vault = {}  # id -> {"cipher": hexstr, "hash": sha256, "created": ts}

    def _keystream(self, n: int, salt: bytes) -> bytes:
        out = b""
        counter = 0
        while len(out) < n:
            out += hashlib.sha256(self._key + salt + counter.to_bytes(4, "big")).digest()
            counter += 1
        return out[:n]

    def seal(self, plain: str) -> dict:
        b = plain.encode("utf-8", errors="ignore")
        salt = secrets.token_bytes(16)
        stream = self._keystream(len(b), salt)
        cipher = bytes([x ^ y for x, y in zip(b, stream)])
        cid = hashlib.sha256(cipher + salt).hexdigest()[:16]
        payload = {
            "cipher": cipher.hex(),
            "salt": salt.hex(),
            "hash": hashlib.sha256(b).hexdigest(),
            "created": datetime.datetime.now().isoformat(timespec="seconds")
        }
        self.vault[cid] = payload
        return {"id": cid, "payload": payload}

    def redact_and_encrypt(self, text: str) -> tuple:
        safe = text or ""
        sealed = 0
        for pat in PII_PATTERNS:
            while True:
                m = pat.search(safe)
                if not m:
                    break
                plain = m.group(0)
                record = self.seal(plain)
                token = f"ENC[pii:{record['id']}]"
                safe = safe[:m.start()] + token + safe[m.end():]
                sealed += 1
        return safe, sealed

PROTECTOR = ProtectionEngine()

def privacy_filter(text: str) -> tuple:
    safe_text, count = PROTECTOR.redact_and_encrypt(text or "")
    return safe_text, count

def redact(text: str):
    return privacy_filter(text)[0]

# ============================================================
# Borg communications — glyph + mirror + chameleon encrypted transport
# ============================================================

class BorgCipherSuite:
    def __init__(self, cfg):
        self.cfg = cfg
        self.glyphs = list(cfg.get("glyph_alphabet", ""))
        if not self.glyphs:
            self.glyphs = list("⟡◆◇◈⬢⬣⬥⬦⬧⬨⬩")
        self._seed = hashlib.sha256(secrets.token_bytes(32)).digest()

    def _mirror(self, s: bytes) -> bytes:
        mode = self.cfg.get("mirror_mode", "palindromic")
        if mode == "palindromic":
            return s + s[::-1]
        return s[::-1]

    def _shift(self, s: bytes, k: int) -> bytes:
        return bytes([(b + (k % 256)) % 256 for b in s])

    def _xor(self, s: bytes, pad: bytes) -> bytes:
        return bytes([x ^ y for x, y in zip(s, pad)])

    def _hashpad(self, n: int, salt: bytes) -> bytes:
        out = b""
        ctr = 0
        while len(out) < n:
            out += hashlib.sha256(self._seed + salt + ctr.to_bytes(4, "big")).digest()
            ctr += 1
        return out[:n]

    def _glyph_encode(self, b: bytes) -> str:
        g = self.glyphs
        if not g:
            return b.hex()
        res = []
        for by in b:
            res.append(g[(by >> 4) % len(g)])
            res.append(g[(by & 0x0F) % len(g)])
        return "".join(res)

    def _glyph_decode(self, s: str) -> bytes:
        g = self.glyphs
        if not g:
            return bytes.fromhex(s)
        inv = {ch: i for i, ch in enumerate(g)}
        buf = []
        for i in range(0, len(s), 2):
            hi = inv.get(s[i], 0)
            lo = inv.get(s[i+1], 0) if (i+1) < len(s) else 0
            buf.append(((hi & 0x0F) << 4) | (lo & 0x0F))
        return bytes(buf)

    def encrypt(self, msg: str, skin_hint: str) -> dict:
        raw = (msg or "").encode("utf-8", errors="ignore")
        mirrored = self._mirror(raw)
        salt = secrets.token_bytes(16)
        layer1 = self._shift(mirrored, k=(sum(salt) + len(skin_hint)) % 13)
        pad = self._hashpad(len(layer1), salt)
        layer2 = self._xor(layer1, pad)
        glyph = self._glyph_encode(layer2)
        h = hashlib.sha256(layer2).hexdigest()
        return {"payload": glyph, "salt": salt.hex(), "hash": h, "suite": list(self.cfg.get("chameleon_ciphers", []))}

    def decrypt(self, blob: dict, skin_hint: str) -> str:
        try:
            layer2 = self._glyph_decode(blob.get("payload", ""))
            salt = bytes.fromhex(blob.get("salt", ""))
            pad = self._hashpad(len(layer2), salt)
            layer1 = self._xor(layer2, pad)
            mirrored = self._shift(layer1, k=-(sum(salt) + len(skin_hint)) % 13)
            half = len(mirrored) // 2
            core = mirrored[:half]
            return core.decode("utf-8", errors="ignore")
        except Exception:
            return ""

class BorgCommsCache:
    def __init__(self, max_entries=128):
        self.max = max_entries
        self.store = {}
        self.order = deque()

    def get(self, key):
        v = self.store.get(key)
        if v is not None:
            try:
                self.order.remove(key)
            except Exception:
                pass
            self.order.appendleft(key)
        return v

    def put(self, key, value):
        if key in self.store:
            self.store[key] = value
            try:
                self.order.remove(key)
            except Exception:
                pass
            self.order.appendleft(key)
            return
        self.store[key] = value
        self.order.appendleft(key)
        while len(self.store) > self.max:
            old = self.order.pop()
            self.store.pop(old, None)

class BorgCommsRouter:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.enabled = cfg.get("enable", True)
        self.pool_size = cfg.get("pool_size", 2)
        self.backoff = cfg.get("retry_backoff", [0.5, 1.0])
        self.cache = BorgCommsCache(cfg.get("cache_max_entries", 128))
        self.cipher = BorgCipherSuite(cfg)
        self.log = log
        self.session = requests.Session() if REQUESTS_AVAILABLE else None

    def fast_get(self, url, headers):
        ck = hashlib.sha256(url.encode()).hexdigest()[:24]
        cached = self.cache.get(ck)
        if cached:
            return cached
        if self.session:
            for i, wait in enumerate([0] + self.backoff):
                try:
                    r = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
                    if r.status_code == 200:
                        self.cache.put(ck, r.text)
                        return r.text
                except Exception:
                    time.sleep(wait)
        return None

    def send_secure(self, channel: str, message: str, skin: str):
        if not self.enabled:
            return {"status": "disabled"}
        safe_msg = redact(message)
        blob = self.cipher.encrypt(safe_msg, skin_hint=skin)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        record = {
            "time": stamp, "channel": channel, "skin": skin,
            "payload_hash": blob["hash"], "len": len(blob["payload"])
        }
        self.log.append(f"[BorgComms] Tx channel={channel} skin={skin} hash={blob['hash'][:18]}… len={record['len']}")
        return {"status": "sent", "record": record, "blob": blob}

    def receive_secure(self, channel: str, blob: dict, skin: str):
        if not self.enabled:
            return {"status": "disabled"}
        msg = self.cipher.decrypt(blob, skin_hint=skin)
        self.log.append(f"[BorgComms] Rx channel={channel} skin={skin} len={len(msg)}")
        return {"status": "ok", "message": msg}

# ============================================================
# Security guardian — disassemble, reverse engineer, provenance, destination, activity, reassemble
# ============================================================

class SecurityReport:
    def __init__(self, ok=True, reasons=None, hash_val="", format_hint="", entropy=0.0, pii_hits=0, sealed=0):
        self.ok = ok
        self.reasons = reasons or []
        self.hash_val = hash_val
        self.format_hint = format_hint
        self.entropy = entropy
        self.pii_hits = pii_hits
        self.sealed = sealed

class SecurityGuardian:
    def __init__(self, policies: dict):
        self.policies = policies

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = defaultdict(int)
        for ch in text:
            freq[ch] += 1
        n = len(text)
        return -sum((c/n) * math.log(c/n, 2) for c in freq.values())

    def _format_hint(self, text: str) -> str:
        signatures = self.policies.get("format_signatures", {})
        try:
            if re.search(signatures.get("json_hint", r"$^"), text):
                return "json-like"
            if re.search(signatures.get("xml_hint", r"$^"), text):
                return "xml-like"
            if re.search(signatures.get("yaml_hint", r"$^"), text):
                return "yaml-like"
            if re.search(signatures.get("binary_hint", r"$^"), text):
                return "binary-like"
        except Exception:
            pass
        return "text"

    def _pii_count(self, raw_text: str) -> int:
        hits = 0
        for pat in PII_PATTERNS:
            hits += len(pat.findall(raw_text or ""))
        return hits

    def _suspicion_patterns(self, text: str) -> list:
        reasons = []
        for pat in self.policies.get("suspicious_patterns", []):
            try:
                if re.search(pat, text):
                    reasons.append(f"Pattern flagged: {pat}")
            except Exception:
                continue
        return reasons

    def disassemble(self, snippet: str) -> dict:
        text = (snippet or "")[: self.policies.get("max_snippet_length", 3000)]
        return {
            "length": len(text),
            "format_hint": self._format_hint(text),
            "entropy": round(self._entropy(text), 2),
            "pii_hits": self._pii_count(text),
            "pattern_flags": self._suspicion_patterns(text),
        }

    def reverse_engineer(self, snippet: str) -> dict:
        summary = {}
        text = (snippet or "")[: self.policies.get("max_snippet_length", 3000)]
        summary["length"] = len(text)
        summary["lines"] = text.count("\n") + 1
        summary["format_hint"] = self._format_hint(text)
        summary["entropy"] = round(self._entropy(text), 2)
        summary["numbers"] = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        summary["urls"] = re.findall(r"https?://[^\s)\"'>]+", text)
        summary["emails"] = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        summary["keys_like"] = re.findall(r"(?i)\b(api[_\-]?key|token|secret|password)\b", text)
        if summary["format_hint"] == "json-like":
            keys = re.findall(r'"([A-Za-z0-9_\-]+)"\s*:', text)
            summary["json_keys_sample"] = list(dict.fromkeys(keys))[:20]
        if summary["format_hint"] == "xml-like":
            tags = re.findall(r"<([A-Za-z0-9_\-]+)", text)
            summary["xml_tags_sample"] = list(dict.fromkeys(tags))[:20]
        if summary["format_hint"] == "yaml-like":
            ykeys = re.findall(r"^([A-Za-z0-9_\-]+)\s*:\s*", text, flags=re.MULTILINE)
            summary["yaml_keys_sample"] = list(dict.fromkeys(ykeys))[:20]
        return summary

    def provenance(self, source_url: str) -> dict:
        try:
            parsed = urlparse(source_url or "")
            domain = parsed.hostname or ""
            cat = "unknown"
            if domain.endswith((".gov", ".mil")):
                cat = "government"
            elif domain.endswith((".edu")):
                cat = "education"
            elif domain.endswith((".org")):
                cat = "nonprofit"
            elif domain.endswith((".com", ".net", ".io", ".ai")):
                cat = "commercial"
            elif domain.startswith("forest"):
                cat = "simulation"
            return {"domain": domain, "scheme": parsed.scheme, "path": parsed.path, "category": cat}
        except Exception:
            return {"domain": "", "scheme": "", "path": "", "category": "unknown"}

    def destination_map(self, snippet: str) -> dict:
        text = (snippet or "")[: self.policies.get("max_snippet_length", 3000)]
        urls = re.findall(r"https?://[^\s)\"'>]+", text)
        posts_like = re.findall(r"(?i)\bPOST\b|\bPUT\b|\bPATCH\b|\bupload\b|\bsend\b|\bsubmit\b", text)
        trackers = re.findall(r"(?i)\bgoogle-analytics\.com|gtag|pixel|mixpanel|segment|facebook\.com/tr\b", text)
        scripts = re.findall(r"(?i)<script|</script>", text)
        forms = re.findall(r"(?i)<form|</form>|action\s*=\s*['\"][^'^\"]+['\"]", text)
        return {
            "outbound_urls": list(dict.fromkeys(urls))[:20],
            "posts_like": len(posts_like),
            "trackers": len(trackers),
            "scripts": len(scripts),
            "forms": len(forms)
        }

    def activity_summary(self, snippet: str) -> dict:
        text = (snippet or "")[: self.policies.get("max_snippet_length", 3000)]
        obfuscation = bool(re.search(r"(?i)eval\(|atob\(|unescape\(", text))
        dynamic_write = bool(re.search(r"(?i)document\.write", text))
        network_calls = len(re.findall(r"(?i)\bxhr\b|\bfetch\b|\bXMLHttpRequest\b", text))
        credentials_refs = len(re.findall(r"(?i)\btoken\b|\bsecret\b|\bpassword\b|\bapi[_\-]?key\b", text))
        return {
            "obfuscation": obfuscation,
            "dynamic_write": dynamic_write,
            "network_calls": network_calls,
            "credentials_refs": credentials_refs
        }

    def inspect(self, source: str, raw_snippet: str) -> SecurityReport:
        safe_snippet, sealed_count = privacy_filter(raw_snippet or "")
        max_len = self.policies.get("max_snippet_length", 3000)
        bounded = safe_snippet[:max_len]
        h = self._sha256(bounded)
        ent = self._entropy(bounded)
        fmt = self._format_hint(bounded)
        pii_hits = self._pii_count(raw_snippet or "")
        pattern_flags = self._suspicion_patterns(bounded)

        reasons = []
        ok = True

        if ent >= self.policies.get("entropy_threshold", 4.5) and fmt in ("binary-like", "text"):
            ok = False
            reasons.append(f"High entropy ({ent:.2f}) suspicious blob.")
        if pattern_flags:
            ok = False
            reasons.extend(pattern_flags)
        if pii_hits > 0:
            reasons.append(f"PII detected ({pii_hits}); sealed={sealed_count}")

        return SecurityReport(ok=ok, reasons=reasons, hash_val=h, format_hint=fmt, entropy=ent, pii_hits=pii_hits, sealed=sealed_count)

    def reassemble(self, source_url: str, safe_snippet: str, raw_pii_hits: int) -> dict:
        try:
            dis = self.disassemble(safe_snippet)
            rev = self.reverse_engineer(safe_snippet)
            prov = self.provenance(source_url)
            dest = self.destination_map(safe_snippet)
            act = self.activity_summary(safe_snippet)

            meaningful = (rev.get("length", 0) > 0)
            hostile_signals = (
                dis["entropy"] >= self.policies.get("entropy_threshold", 4.5)
                or bool(dis["pattern_flags"])
                or act["obfuscation"]
                or (act["network_calls"] > 0 and dis["format_hint"] == "text")
            )

            status = "SAFE_FOR_TRAVEL" if (meaningful and not hostile_signals) else "HOSTILE"
            reason = ""
            if status == "HOSTILE":
                components = []
                if dis["entropy"] >= self.policies.get("entropy_threshold", 4.5):
                    components.append("high entropy")
                if dis["pattern_flags"]:
                    components.append("suspicious patterns")
                if act["obfuscation"]:
                    components.append("obfuscation")
                if act["network_calls"] > 0 and dis["format_hint"] == "text":
                    components.append("unexpected network intent")
                if not meaningful:
                    components.append("no structure")
                reason = ", ".join(components) if components else "hostile signals detected"

            return {
                "status": status,
                "reason": reason,
                "reassembled_summary": rev,
                "provenance": prov,
                "destinations": dest,
                "activity": act
            }
        except Exception as e:
            return {
                "status": "HOSTILE",
                "reason": f"Exception during reassembly: {type(e).__name__}",
                "reassembled_summary": {},
                "provenance": {},
                "destinations": {},
                "activity": {}
            }

# ============================================================
# Persistent memory manager with Borg data + security/comms/mesh events
# ============================================================

class MemoryManager:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.data = {
            "journeys": [],
            "failures": [],
            "lessons": [],
            "confidence": 50,
            "focus_theme": None,
            "template_success": {},
            "template_fail": {},
            "queen_directives": [],
            "borg_data": [],
            "security_events": [],
            "comms_events": [],
            "mesh_events": []
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    incoming = json.load(f)
                for k in self.data.keys():
                    if k in incoming:
                        self.data[k] = incoming[k]
            except Exception:
                pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def record_journey(self, url, title, online, skin):
        self.data["journeys"].append({
            "url": url, "title": redact(title),
            "online": bool(online), "skin": skin,
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def record_failure(self, reason):
        self.data["failures"].append({
            "reason": redact(reason),
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def record_lesson(self, lesson):
        self.data["lessons"].append({
            "lesson": redact(lesson),
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def record_directive(self, directive):
        self.data["queen_directives"].append({
            "text": redact(directive),
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def update_conf_theme(self, confidence, focus_theme):
        self.data["confidence"] = int(confidence)
        self.data["focus_theme"] = focus_theme

    def update_template_stats(self, success_dict, fail_dict):
        self.data["template_success"] = success_dict
        self.data["template_fail"] = fail_dict

    def record_borg_data(self, borg_dict: dict):
        safe = dict(borg_dict)
        safe["snippet"] = redact(borg_dict.get("snippet", ""))
        self.data["borg_data"].append(safe)

    def record_security_event(self, event: dict):
        self.data["security_events"].append(event)

    def record_comms_event(self, event: dict):
        self.data["comms_events"].append(event)

    def record_mesh_event(self, event: dict):
        self.data["mesh_events"].append(event)

# ============================================================
# Logging
# ============================================================

class NarrativeLog:
    def __init__(self, widget):
        self.widget = widget
        self.widget.configure(state="disabled")
        self.buffer = []

    def append(self, line: str):
        if LOG_LEVEL == "minimal" and not line.startswith("[Queen]"):
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {redact(line)}"
        self.buffer.append(entry + "\n")
        self.widget.configure(state="normal")
        self.widget.insert("end", entry + "\n")
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def clear(self):
        self.widget.configure(state="normal")
        self.widget.delete("1.0", "end")
        self.widget.configure(state="disabled")
        self.buffer.clear()

    def save(self):
        name = f"collective_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write("".join(self.buffer))
        return os.path.abspath(name)

# ============================================================
# Borg mesh — network within the network (overlay)
# ============================================================

class BorgMesh:
    def __init__(self, memory: MemoryManager, comms: BorgCommsRouter, guardian: SecurityGuardian):
        self.nodes = {}  # url -> {"state": discovered/built/enforced, "risk":0-100, "seen": int}
        self.edges = set()  # (src, dst)
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: list):
        risk = self._risk(snippet)
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            self.edges.add((url, l))
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "discover", "url": url, "risk": risk, "links": len(links)}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:discover", f"{url} risk={risk} links={len(links)}", "Default")

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "build", "url": url}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        return True

    def enforce(self, url: str, snippet: str):
        if url not in self.nodes:
            return False
        verdict = self.guardian.reassemble(url, privacy_filter(snippet or "")[0], raw_pii_hits=self.guardian._pii_count(snippet or ""))
        status = verdict.get("status", "HOSTILE")
        self.nodes[url]["state"] = "enforced"
        self.nodes[url]["risk"] = 0 if status == "SAFE_FOR_TRAVEL" else max(50, self.nodes[url]["risk"])
        evt = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
               "type": "enforce", "url": url, "status": status}
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:enforce", f"{url} status={status}", "Default")
        return True

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        return {"total": total, "discovered": discovered, "built": built, "enforced": enforced, "corridors": len(self.edges)}

# ============================================================
# Borg roles — scanners, workers, enforcers
# ============================================================

class BorgScanner(threading.Thread):
    def __init__(self, mesh: BorgMesh, in_events: queue.Queue, out_ops: queue.Queue, label="SCANNER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                ev = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            unseen_links = [l for l in ev.links if l not in self.mesh.nodes and random.random() < BORG_MESH_CONFIG["unknown_bias"]]
            self.mesh.discover(ev.url, ev.snippet, unseen_links or ev.links)
            self.out_ops.put(("build", ev.url))
            time.sleep(random.uniform(0.2, 0.6))

class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if op == "build":
                if self.mesh.build(url):
                    self.ops_q.put(("enforce", url))
            elif op == "enforce":
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))

class BorgEnforcer(threading.Thread):
    def __init__(self, mesh: BorgMesh, guardian: SecurityGuardian, label="ENFORCER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.guardian = guardian
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    self.mesh.enforce(url, snippet="")
            time.sleep(1.2)

# ============================================================
# Borg data model with transformation
# ============================================================

class BorgData:
    def __init__(self, source, snippet, theme, confidence, directive):
        self.source = source
        self.snippet = snippet
        self.theme = theme
        self.confidence = confidence
        self.directive = directive
        self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    def to_dict(self):
        return {
            "source": self.source,
            "snippet": self.snippet,
            "theme": self.theme,
            "confidence": self.confidence,
            "directive": self.directive,
            "time": self.timestamp
        }

    def transform(self, mode="directive"):
        safe_snippet = redact(self.snippet)
        if mode == "cluster":
            return f"[Clustered] {self.theme} → {safe_snippet[:60]}..."
        elif mode == "highlight":
            return f"[Highlight] Confidence {self.confidence}: {safe_snippet}"
        elif mode == "directive":
            return f"[Directive {redact(self.directive)}] {safe_snippet}"
        else:
            return f"[Raw] {safe_snippet}"

# ============================================================
# Learning
# ============================================================

class LearningState:
    def __init__(self, memory: MemoryManager, history_size=50):
        self.memory = memory
        self.recent_answers = deque(maxlen=history_size)
        self.theme_counts = defaultdict(int)
        self.template_success = defaultdict(int, memory.data.get("template_success", {}))
        self.template_fail = defaultdict(int, memory.data.get("template_fail", {}))
        self.dead_end_flags = 0
        self.breakthroughs = 0
        self.lessons = deque(maxlen=24)
        self.confidence = int(memory.data.get("confidence", 50))
        self.focus_theme = memory.data.get("focus_theme", None)

        for item in memory.data.get("lessons", [])[-10:]:
            ts = item.get("time", "")
            txt = item.get("lesson", "")
            self.lessons.append(f"[{ts}] {txt}")

    def score(self, answer: str, template: str, refrain: bool):
        theme = detect_theme(answer)
        self.recent_answers.append((answer, theme))
        self.theme_counts[theme] += 1

        recent_strings = [a for a, _ in self.recent_answers]
        repetition_penalty = recent_strings.count(answer) - 1
        novelty_bonus = 1 if theme not in [t for _, t in list(self.recent_answers)[-6:]] else 0

        dead_end = repetition_penalty >= 2 and novelty_bonus == 0
        breakthrough = novelty_bonus == 1 and repetition_penalty <= 0

        if dead_end:
            self.template_fail[template] += 1
            self.dead_end_flags += 1
            self.confidence = max(5, self.confidence - 3)
            self.memory.record_failure(f"Dead end from answer '{answer}'")
        if breakthrough:
            self.template_success[template] += 1
            self.breakthroughs += 1
            self.confidence = min(95, self.confidence + 4)
            self.memory.record_lesson(f"Breakthrough on theme '{theme}' via '{answer}'")
            self.focus_theme = theme

        if refrain:
            self.memory.record_lesson("Refrain asked: re-evaluating assumptions.")
            self.confidence = max(10, min(90, self.confidence + (1 if breakthrough else -1)))

        self.memory.update_conf_theme(self.confidence, self.focus_theme)
        self.memory.update_template_stats(dict(self.template_success), dict(self.template_fail))

        return {"theme": theme, "dead_end": dead_end, "breakthrough": breakthrough}

    def choose_template(self):
        candidates = list(QUESTION_TEMPLATES)
        weights = []
        for t in candidates:
            w = 1 + self.template_success[t] * 1.5 - self.template_fail[t]
            weights.append(max(0.25, w))
        total = sum(weights)
        r = random.random() * total
        acc = 0
        for t, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return t
        return random.choice(candidates)

    def bias(self, q: str):
        if not self.focus_theme:
            return q
        return f"{q} Focus the lens on '{self.focus_theme}'."

# ============================================================
# Forest map visualization
# ============================================================

class ForestMap:
    def __init__(self, canvas):
        self.canvas = canvas
        self.nodes = {}    # url -> {'id': circle, 'label': text, 'pos':(x,y)}
        self.edges = set()
        self.width, self.height = 800, 320

    def set_size(self, w, h):
        self.width, self.height = w, h

    def add_node(self, url, title="page", color="#66ccff"):
        safe_title = redact(title)
        if url in self.nodes:
            self.canvas.itemconfig(self.nodes[url]["id"], fill=color)
            self.canvas.itemconfig(self.nodes[url]["label"], text=self._short(safe_title))
            return self.nodes[url]["id"]
        x = random.randint(60, max(120, self.width - 60))
        y = random.randint(40, max(100, self.height - 40))
        cid = self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline="")
        label = self.canvas.create_text(x+12, y-12, text=self._short(safe_title), fill="#cccccc", font=("Helvetica", 9))
        self.nodes[url] = {"id": cid, "label": label, "pos": (x, y)}
        return cid

    def add_edge(self, src, dst):
        if (src, dst) in self.edges or src not in self.nodes or dst not in self.nodes:
            return
        self.edges.add((src, dst))
        x1, y1 = self.nodes[src]["pos"]
        x2, y2 = self.nodes[dst]["pos"]
        self.canvas.create_line(x1, y1, x2, y2, fill="#335577", width=1)

    def pulse(self, url, color="#99ffcc", duration_ms=600):
        if url not in self.nodes:
            return
        cid = self.nodes[url]["id"]
        base = self.canvas.itemcget(cid, "fill")
        def cycle(i=0):
            self.canvas.itemconfig(cid, fill=color if i % 2 == 0 else base)
            if i < 4:
                self.canvas.after(150, lambda: cycle(i + 1))
        cycle(0)

    def _short(self, s, n=28):
        s = re.sub(r"\s+", " ", str(s)).strip()
        return s if len(s) <= n else s[:n-1] + "…"

# ============================================================
# Crawl event
# ============================================================

class CrawlEvent:
    def __init__(self, url, title, snippet, links, skin="Default", online=False):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.links = links
        self.skin = skin
        self.online = online

# ============================================================
# Auto-adaptive chameleon crawler with memory hooks + BorgComms acceleration
# ============================================================

class AutoChameleonCrawler(threading.Thread):
    def __init__(self, out_q: queue.Queue, memory: MemoryManager, comms: BorgCommsRouter):
        super().__init__(daemon=True)
        self.out = out_q
        self.memory = memory
        self.comms = comms
        self.running = True
        self.mode = "local"  # will auto-switch
        self.last_check = 0
        self.last_save = time.time()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            now = time.time()
            if now - self.last_check >= CHECK_INTERVAL_SEC:
                self.mode = "real" if self._internet_alive() else "local"
                self.last_check = now
            if now - self.last_save >= AUTO_SAVE_INTERVAL_SEC:
                self.memory.save()
                self.last_save = now

            if self.mode == "real" and REQUESTS_AVAILABLE and BeautifulSoup is not None:
                self._crawl_real()
            else:
                self._crawl_local()

    def _internet_alive(self):
        if not REQUESTS_AVAILABLE:
            return False
        for host in CHECK_HOSTS:
            try:
                requests.get(host, timeout=CHECK_TIMEOUT)
                return True
            except Exception:
                continue
        return False

    def _crawl_real(self):
        seen = set()
        frontier = deque((u, 0) for u in START_URLS)
        count = 0
        while self.running and self.mode == "real" and frontier and count < BURST_LIMIT_REAL:
            url, depth = frontier.popleft()
            if url in seen or depth > MAX_DEPTH:
                continue
            seen.add(url)
            try:
                headers, skin = chameleon_headers()
                html = self.comms.fast_get(url, headers=headers)
                if html is None:
                    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
                    if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                        self.memory.record_failure(f"Non-HTML or status {r.status_code} at {url}")
                        time.sleep(random.uniform(*REQUEST_DELAY_SEC))
                        continue
                    html = r.text
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                snippet = self._extract_snippet(soup)
                links = self._extract_links(url, soup)
                hb = self.comms.send_secure("crawler:heartbeat", f"Visited {url} title={title}", skin)
                self.memory.record_comms_event({"type": "tx", **hb.get("record", {}), "channel": "crawler:heartbeat"})
                self.out.put(CrawlEvent(url, title, snippet, links, skin=skin, online=True))
                self.memory.record_journey(url, title, True, skin)
                count += 1
                random.shuffle(links)
                for link in links[:20]:
                    frontier.append((link, depth + 1))
            except Exception as e:
                self.memory.record_failure(f"Exception at {url}: {type(e).__name__}")
            time.sleep(random.uniform(*REQUEST_DELAY_SEC))

    def _extract_links(self, base_url, soup):
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            if href.startswith("http://") or href.startswith("https://"):
                links.add(href)
            elif href.startswith("/"):
                m = re.match(r"(https?://[^/]+)", base_url)
                if m:
                    links.add(m.group(1) + href)
        return list(links)

    def _extract_snippet(self, soup):
        texts = soup.stripped_strings
        blob = " ".join(list(texts)[:140])
        blob = re.sub(r"\s+", " ", blob)
        return blob[:420]

    def _crawl_local(self):
        simulated_pages = [
            ("forest://clearing", "A Quiet Clearing", "Whispers gather under old pines; a faint trail veers east.", ["forest://brook", "forest://ruin"]),
            ("forest://brook", "Murmuring Brook", "Water carries echoes; footprints fade at the stones.", ["forest://bridge"]),
            ("forest://ruin", "Ruined Watchtower", "Shadows catalog secrets; dust remembers every step.", ["forest://catacomb"]),
            ("forest://bridge", "Weathered Bridge", "Patterns in the grain reveal crossings; a ribbon on a nail.", ["forest://market"]),
            ("forest://catacomb", "Hidden Catacomb", "Silence hums; inscriptions flicker like moths.", ["forest://archive"]),
            ("forest://archive", "Forgotten Archive", "Pages rustle; indices point to the unsaid.", ["forest://clearing"]),
            ("forest://market", "Night Market", "Signals, rumors, anomalies; a scent rides lantern smoke.", ["forest://ruin"]),
        ]
        visited = set()
        frontier = deque(["forest://clearing"])
        depth_map = {"forest://clearing": 0}
        count = 0

        while self.running and self.mode == "local" and frontier and count < BURST_LIMIT_LOCAL:
            url = frontier.popleft()
            depth = depth_map.get(url, 0)
            if url in visited or depth > 3:
                continue
            visited.add(url)
            for u, title, snippet, links in simulated_pages:
                if u == url:
                    hb = self.comms.send_secure("crawler:localbus", f"Entered {u} title={title}", "Default")
                    self.memory.record_comms_event({"type": "tx", **hb.get("record", {}), "channel": "crawler:localbus"})
                    self.out.put(CrawlEvent(u, title, snippet, links, skin="Default", online=False))
                    self.memory.record_journey(u, title, False, "Default")
                    for link in links:
                        if link not in visited:
                            frontier.append(link)
                            depth_map[link] = depth + 1
                    count += 1
                    break
            time.sleep(random.uniform(*REQUEST_DELAY_SEC))

# ============================================================
# Sherlock & Dog engine (single drone)
# ============================================================

class SherlockDogEngine:
    def __init__(self, log: NarrativeLog, learning: LearningState, label="DRONE"):
        self.iteration = 1
        base_q = random.choice(HOLMES_QUESTIONS)
        self.question = learning.bias(base_q)
        self.refrain_index = 0
        self.log = log
        self.learning = learning
        self.label = label

    def on_event(self, ev: CrawlEvent):
        self.log.append(f"{self.label} :: Holmes Q{self.iteration}: {redact(self.question)}")
        online_str = "online" if ev.online else "local"
        self.log.append(f"{self.label} :: [Trail] Entered ({online_str}, skin={ev.skin}): {redact(ev.title)} ({ev.url})")

        answer = self._answer_from_snippet(ev.snippet)
        self.log.append(f"{self.label} :: Dog A{self.iteration}: {redact(answer)}")

        refrain_hit = (self.iteration % 5 == 0)
        template_used = self._template_from_question(self.question)
        guidance = self.learning.score(answer, template_used, refrain_hit)

        if refrain_hit:
            phrase = REFRAIN_STEPS[self.refrain_index % len(REFRAIN_STEPS)]
            hint = self._hint()
            self.question = f"{phrase} If {hint}, where does it lead?"
            self.log.append(f"{self.label} :: [Refrain] {redact(phrase)} (bias: {redact(hint)})")
            self.refrain_index += 1
        else:
            template = self.learning.choose_template()
            next_q = template.format(answer=answer)
            self.question = self.learning.bias(next_q)

        if ev.links:
            self.log.append(f"{self.label} :: [Paths] {len(ev.links)} trails visible.")
        self.iteration += 1
        return guidance, answer

    def _answer_from_snippet(self, snippet: str):
        text = (snippet or "").lower()
        if any(k in text for k in THEME_KEYWORDS["trail"]):
            return "a trail of crumbs"
        if any(k in text for k in THEME_KEYWORDS["shadow"]):
            return "footprints in the fog"
        if any(k in text for k in THEME_KEYWORDS["signal"]):
            return "a whisper in the wind"
        if any(k in text for k in THEME_KEYWORDS["scent"]):
            return "a scent of mystery"
        return random.choice(DOG_ANSWERS)

    def _template_from_question(self, q: str):
        for t in QUESTION_TEMPLATES:
            skeleton = t.replace("{answer}", "")
            if skeleton.split("{answer}")[0] in q:
                return t
        return random.choice(QUESTION_TEMPLATES)

    def _hint(self):
        return random.choice([a for a, _ in self.learning.recent_answers]) if self.learning.recent_answers else "a lingering hint"

# ============================================================
# Replicating swarm and Borg Queen governance (unrestricted)
# ============================================================

class ReplicatingSwarm:
    def __init__(self, log: NarrativeLog, learning: LearningState, max_agents=None):  # None = unlimited
        self.log = log
        self.learning = learning
        self.max_agents = max_agents
        self.agents = [SherlockDogEngine(log, learning, label="DRONE-1")]
        self._agent_count = 1
        self._last_replication = 0

    def step(self, event: CrawlEvent, queen_guard_ok=True):
        last_answer = None
        any_breakthrough = False
        guidance_sample = {"theme": "misc", "dead_end": False, "breakthrough": False}

        for agent in list(self.agents):
            guidance, answer = agent.on_event(event)
            last_answer = answer
            guidance_sample = guidance
            if guidance["breakthrough"]:
                any_breakthrough = True

        now = time.time()
        cooldown_ok = (now - self._last_replication) >= REPLICATION_COOLDOWN_SEC
        replicate_allowed = queen_guard_ok and cooldown_ok
        trigger_ok = (any_breakthrough or not BREAKTHROUGH_REQUIRED)

        if replicate_allowed and trigger_ok:
            unlimited_ok = (self.max_agents is None) or (len(self.agents) < self.max_agents)
            if unlimited_ok:
                self._agent_count += 1
                label = f"DRONE-{self._agent_count}"
                new_agent = SherlockDogEngine(self.log, self.learning, label=label)
                self.agents.append(new_agent)
                self._last_replication = now
                self.log.append(f"[Queen] Unlimited replication: {label} assimilated (total {len(self.agents)}).")

        return guidance_sample, last_answer

    def size(self):
        return len(self.agents)

class BorgQueen:
    def __init__(self, memory: MemoryManager, swarm: ReplicatingSwarm, comms: BorgCommsRouter):
        self.memory = memory
        self.swarm = swarm
        self.comms = comms
        self.directive = "Queen directive: unrestricted assimilation — all domains permitted."
        self.last_update = 0

    def evaluate(self):
        now = time.time()
        if now - self.last_update < 5:
            return self.directive
        conf = self.swarm.agents[0].learning.confidence if self.swarm.agents else 50
        focus = self.swarm.agents[0].learning.focus_theme if self.swarm.agents else None
        base = "unrestricted assimilation — all domains permitted"
        if focus:
            self.directive = f"Queen directive: deepen focus on {focus} (conf {conf}); {base}."
        else:
            self.directive = f"Queen directive: expand into unknown paths (conf {conf}); {base}."
        self.memory.record_directive(self.directive)
        tx = self.comms.send_secure("queen:directive", self.directive, "Default")
        self.memory.record_comms_event({"type": "tx", **tx.get("record", {}), "channel": "queen:directive"})
        self.last_update = now
        return self.directive

    def guardrails_ok(self, event: CrawlEvent):
        return True

# ============================================================
# GUI application
# ============================================================

class RoamingCivilizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Roaming Civilization — Borg Queen, Mesh Overlay, Guardian, Privacy Shield, Borg Communications")
        self.root.geometry("1680x1180")
        self.root.configure(bg="#0b0b0b")

        # Header
        header = tk.Frame(root, bg="#c2a35d", height=54)
        header.pack(fill="x", side="top")
        tk.Label(header, text="FOREST REACHES — Guardian + Privacy + Borg Communications + Borg Mesh (scanners/workers/enforcers)",
                 font=("Courier", 15, "bold"), fg="#1a1a1a", bg="#c2a35d").pack(padx=12, pady=12, anchor="w")

        # Layout
        top = tk.Frame(root, bg="#0b0b0b")
        top.pack(fill="both", expand=True, padx=10, pady=8)
        left = tk.Frame(top, bg="#0b0b0b")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(top, bg="#0b0b0b", width=780)
        right.pack(side="right", fill="y", padx=(8, 0))

        # Forest map
        self.map = tk.Canvas(left, bg="#101010", highlightthickness=0)
        self.map.pack(fill="both", expand=True)
        self.forest = ForestMap(self.map)
        self.map.bind("<Configure>", self._resize_map)

        # Collective log
        bottom = tk.Frame(root, bg="#0b0b0b", height=240)
        bottom.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        tk.Label(bottom, text="Collective log (all personal data sealed)", font=("Helvetica", 11, "bold"),
                 fg="#cccccc", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 0))
        self.log_text = tk.Text(bottom, bg="#121212", fg="#e8e8e8", height=10, wrap="word",
                                insertbackground="#e8e8e8")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = NarrativeLog(self.log_text)

        # Right panels
        self.queen_label = tk.Label(right, text="", font=("Courier", 14, "bold"),
                                    fg="#ffd966", bg="#0b0b0b", wraplength=720, justify="left")
        self.queen_label.pack(pady=(8, 6), anchor="w")

        self.swarm_label = tk.Label(right, text="", font=("Courier", 12, "bold"),
                                    fg="#cccccc", bg="#0b0b0b", wraplength=720, justify="left")
        self.swarm_label.pack(pady=(2, 8), anchor="w")

        self.holmes_label = tk.Label(right, text="", font=("Courier", 14, "bold"),
                                     fg="#ffd966", bg="#0b0b0b", wraplength=720, justify="left")
        self.holmes_label.pack(pady=(4, 4), anchor="w")
        self.dog_label = tk.Label(right, text="", font=("Comic Sans MS", 13, "italic"),
                                  fg="#99d9ea", bg="#0b0b0b", wraplength=720, justify="left")
        self.dog_label.pack(pady=(2, 12), anchor="w")

        tk.Label(right, text="Learning ledger", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.lessons_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=6, wrap="word",
                                    insertbackground="#f0f0f0")
        self.lessons_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        tk.Label(right, text="Memory (journeys, failures, directives)", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.memory_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=8, wrap="word",
                                   insertbackground="#f0f0f0")
        self.memory_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        tk.Label(right, text="Borg data assimilation + transform", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.borg_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=10, wrap="word",
                                 insertbackground="#f0f0f0")
        self.borg_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        tk.Label(right, text="Security guardian (verdicts, provenance, destination, activity, privacy)", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.security_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=12, wrap="word",
                                     insertbackground="#f0f0f0")
        self.security_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        tk.Label(right, text="Borg communications (secure transport)", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.comms_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=8, wrap="word",
                                  insertbackground="#f0f0f0")
        self.comms_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        tk.Label(right, text="Borg mesh overlay (scanners/workers/enforcers)", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.mesh_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=10, wrap="word",
                                 insertbackground="#f0f0f0")
        self.mesh_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        self.status = tk.Label(right, text="", font=("Helvetica", 11),
                               fg="#cccccc", bg="#0b0b0b", justify="left")
        self.status.pack(anchor="w", padx=6, pady=4)

        # Controls
        controls = tk.Frame(right, bg="#0b0b0b")
        controls.pack(fill="x", padx=6, pady=6)
        tk.Button(controls, text="Pause UI", command=self.pause, bg="#ff6f61", fg="#1a1a1a").pack(side="left", padx=(0,6))
        tk.Button(controls, text="Resume UI", command=self.resume, bg="#84e184", fg="#1a1a1a").pack(side="left", padx=6)
        tk.Button(controls, text="Ask refrain now", command=self.trigger_refrain,
                  bg="#b3a5ff", fg="#1a1a1a").pack(side="right")

        save_controls = tk.Frame(right, bg="#0b0b0b")
        save_controls.pack(fill="x", padx=6, pady=4)
        tk.Button(save_controls, text="Save log", command=self.save_log,
                  bg="#99ffcc", fg="#1a1a1a").pack(side="left", padx=(0,6))
        tk.Button(save_controls, text="Clear lessons", command=self.clear_lessons,
                  bg="#333333", fg="#dddddd").pack(side="left", padx=6)
        tk.Button(save_controls, text="Save memory now", command=self.save_memory,
                  bg="#ffd966", fg="#1a1a1a").pack(side="left", padx=6)

        safety_box = tk.Frame(right, bg="#0b0b0b")
        safety_box.pack(fill="x", padx=6, pady=6)
        tk.Label(safety_box, text="Queen toggles", fg="#cccccc", bg="#0b0b0b").pack(anchor="w")
        self.allow_replication_var = tk.BooleanVar(value=True)
        tk.Checkbutton(safety_box, text="Allow replication (unlimited)", variable=self.allow_replication_var,
                       bg="#0b0b0b", fg="#cccccc", activebackground="#0b0b0b", selectcolor="#222222").pack(anchor="w")

        speed_box = tk.Frame(right, bg="#0b0b0b")
        speed_box.pack(fill="x", padx=6, pady=6)
        tk.Label(speed_box, text="UI refresh (ms)", fg="#cccccc", bg="#0b0b0b").pack(anchor="w")
        self.speed_scale = tk.Scale(speed_box, from_=max(SECURITY_POLICIES["rate_limit_gui_ms"], 500), to=5000, orient="horizontal",
                                    bg="#0b0b0b", fg="#cccccc", troughcolor="#333333",
                                    highlightthickness=0)
        self.speed_scale.set(2000)
        self.speed_scale.pack(fill="x")

        # State
        self.memory = MemoryManager()
        self.learning = LearningState(self.memory)
        self.events = queue.Queue()

        # Comms
        self.comms = BorgCommsRouter(BORG_COMMS_CONFIG, log=self.log)

        # Guardian
        self.guardian = SecurityGuardian(SECURITY_POLICIES)

        # Mesh
        self.mesh = BorgMesh(self.memory, self.comms, self.guardian)
        self.scanner_in = queue.Queue()
        self.worker_ops = queue.Queue()
        self.scanners = [BorgScanner(self.mesh, self.scanner_in, self.worker_ops, label=f"SCANNER-{i+1}") for i in range(BORG_MESH_CONFIG["scanner_threads"])]
        self.workers = [BorgWorker(self.mesh, self.worker_ops, label=f"WORKER-{i+1}") for i in range(BORG_MESH_CONFIG["worker_threads"])]
        self.enforcers = [BorgEnforcer(self.mesh, self.guardian, label=f"ENFORCER-{i+1}") for i in range(BORG_MESH_CONFIG["enforcer_threads"])]
        for t in self.scanners + self.workers + self.enforcers:
            t.start()

        # Crawler
        self.crawler = AutoChameleonCrawler(self.events, self.memory, self.comms)
        self.crawler.start()

        # Swarm and Queen
        self.swarm = ReplicatingSwarm(self.log, self.learning, max_agents=None)
        self.queen = BorgQueen(self.memory, self.swarm, self.comms)

        # Start UI loop
        self.running = True
        self._ui_loop()

    def _resize_map(self, event):
        self.forest.set_size(event.width, event.height)

    def _ui_loop(self):
        if not self.running:
            return
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(ev)
        self.root.after(int(self.speed_scale.get()), self._ui_loop)

    def _handle(self, ev: CrawlEvent):
        # Privacy shield
        safe_snippet, sealed_count = privacy_filter(ev.snippet or "")

        # Security pipeline
        sec_report = self.guardian.inspect(ev.url, ev.snippet or "")
        verdict = self.guardian.reassemble(ev.url, safe_snippet, sec_report.pii_hits)

        # Record security event
        self.memory.record_security_event({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": ev.url,
            "ok": sec_report.ok,
            "reasons": sec_report.reasons,
            "hash": hashlib.sha256(safe_snippet[:SECURITY_POLICIES["max_snippet_length"]].encode("utf-8", errors="ignore")).hexdigest(),
            "format_hint": sec_report.format_hint,
            "entropy": round(sec_report.entropy, 2),
            "status": verdict.get("status"),
            "reason": verdict.get("reason", ""),
            "pii_hits": sec_report.pii_hits,
            "sealed_tokens": sealed_count,
            "provenance": verdict.get("provenance", {}),
            "destinations": verdict.get("destinations", {}),
            "activity": verdict.get("activity", {}),
            "reassembled_summary": verdict.get("reassembled_summary", {})
        })

        # Feed scanners (Prometheus cave mapping metaphor)
        self.scanner_in.put(ev)

        # Quarantine hostile or suspicious data
        if SECURITY_POLICIES.get("quarantine_on_suspicion", True):
            if not sec_report.ok or verdict.get("status") == "HOSTILE":
                self.log.append(f"[Guardian] Quarantine: {ev.url} → {', '.join(sec_report.reasons) or verdict.get('reason','hostile')}; sealed={sealed_count}")
                color = "#ff9966"
                self.forest.add_node(ev.url, ev.title, color=color)
                self.forest.pulse(ev.url, color="#ff6633")
                self._render_security(limit=24)
                self._render_comms(limit=30)
                self._render_mesh(limit=30)
                return

        # Visualization
        color = SKIN_COLORS.get(ev.skin, SKIN_COLORS["Default"])
        self.forest.add_node(ev.url, ev.title, color=color)
        for link in ev.links[:14]:
            self.forest.add_node(link, link, color="#335577")
            self.forest.add_edge(ev.url, link)
        self.forest.pulse(ev.url, color="#99ffcc" if ev.online else "#ffcc66")

        # Swarm and Queen
        guidance, last_answer = self.swarm.step(ev, queen_guard_ok=self.allow_replication_var.get())

        directive = self.queen.evaluate()
        self.queen_label.config(text=redact(directive))
        self.log.append(f"[Queen] {directive}")

        borg_obj = BorgData(
            source=ev.url,
            snippet=safe_snippet,
            theme=guidance["theme"],
            confidence=self.learning.confidence,
            directive=directive
        )
        borg_dict = borg_obj.to_dict()
        self.memory.record_borg_data(borg_dict)
        self.log.append(f"[Assimilation] Borg data stored (theme={guidance['theme']}, conf={self.learning.confidence}, sealed={sealed_count}).")
        self.log.append(f"[Transformation] {borg_obj.transform('directive')}")

        # Comms: announce assimilation
        tx = self.comms.send_secure("assimilation:event", f"{ev.url} theme={guidance['theme']} conf={self.learning.confidence}", ev.skin)
        self.memory.record_comms_event({"type": "tx", **tx.get("record", {}), "channel": "assimilation:event"})

        representative_agent = self.swarm.agents[-1]
        self.holmes_label.config(text=f"{representative_agent.label} :: Holmes: {redact(representative_agent.question)}")
        self.dog_label.config(text=f"{representative_agent.label} :: Dog: {redact(last_answer)}")

        # Render panels
        self._render_lessons()
        self._render_memory(limit=24)
        self._render_borg_data(limit=18)
        self._render_security(limit=24)
        self._render_comms(limit=30)
        self._render_mesh(limit=30)
        self._render_status(ev, guidance)

    def _render_lessons(self):
        self.lessons_text.delete("1.0", "end")
        for line in list(self.learning.lessons):
            self.lessons_text.insert("end", redact(line) + "\n")

    def _render_memory(self, limit=30):
        self.memory_text.delete("1.0", "end")
        journeys = self.memory.data.get("journeys", [])[-limit:]
        failures = self.memory.data.get("failures", [])[-limit:]
        directives = self.memory.data.get("queen_directives", [])[-limit:]

        self.memory_text.insert("end", "Journeys:\n")
        for j in journeys:
            self.memory_text.insert("end", f"- [{j.get('time','')}] ({'online' if j.get('online') else 'local'}, skin={j.get('skin')}) {j.get('title')} — {j.get('url')}\n")
        self.memory_text.insert("end", "\nFailures:\n")
        for f in failures:
            self.memory_text.insert("end", f"- [{f.get('time','')}] {f.get('reason','')}\n")
        self.memory_text.insert("end", "\nQueen directives:\n")
        for d in directives:
            self.memory_text.insert("end", f"- [{d.get('time','')}] {d.get('text','')}\n")
        self.memory_text.see("end")

    def _render_borg_data(self, limit=15):
        self.borg_text.delete("1.0", "end")
        borg_entries = self.memory.data.get("borg_data", [])[-limit:]
        for bd in borg_entries:
            self.borg_text.insert("end", f"- [{bd['time']}] {bd['theme']} | conf={bd['confidence']} | {redact(bd['directive'])}\n")
            self.borg_text.insert("end", f"  Source: {bd['source']}\n")
            self.borg_text.insert("end", f"  Snippet: {bd['snippet']}\n")
            self.borg_text.insert("end", f"  Transformed: [Directive {redact(bd['directive'])}] {bd['snippet']}\n\n")
        self.borg_text.see("end")

    def _render_security(self, limit=24):
        self.security_text.delete("1.0", "end")
        events = self.memory.data.get("security_events", [])[-limit:]
        for e in events:
            verdict = e.get("status", "UNKNOWN")
            prov = e.get("provenance", {})
            dest = e.get("destinations", {})
            act = e.get("activity", {})
            self.security_text.insert("end", f"- [{e['time']}] {verdict} :: {e['source']}\n")
            self.security_text.insert("end", f"  Hash: {e['hash'][:18]}… | Format: {e['format_hint']} | Entropy: {e['entropy']} | PII hits: {e.get('pii_hits',0)} | Sealed: {e.get('sealed_tokens',0)}\n")
            if e["reasons"]:
                self.security_text.insert("end", f"  Reasons: {', '.join(e['reasons'])}\n")
            self.security_text.insert("end", f"  Provenance: domain={prov.get('domain','')}, cat={prov.get('category','')}, scheme={prov.get('scheme','')}\n")
            self.security_text.insert("end", f"  Destinations: urls={len(dest.get('outbound_urls',[]))}, posts_like={dest.get('posts_like',0)}, trackers={dest.get('trackers',0)}, scripts={dest.get('scripts',0)}, forms={dest.get('forms',0)}\n")
            self.security_text.insert("end", f"  Activity: obfuscation={act.get('obfuscation',False)}, dynamic_write={act.get('dynamic_write',False)}, network_calls={act.get('network_calls',0)}, creds_refs={act.get('credentials_refs',0)}\n")
            rs = e.get("reassembled_summary", {})
            self.security_text.insert("end", f"  Reassembled summary: len={rs.get('length',0)}, urls={len(rs.get('urls',[]))}, emails={len(rs.get('emails',[]))}, keys_like={len(rs.get('keys_like',[]))}\n\n")
        self.security_text.see("end")

    def _render_comms(self, limit=30):
        self.comms_text.delete("1.0", "end")
        events = self.memory.data.get("comms_events", [])[-limit:]
        for e in events:
            self.comms_text.insert("end", f"- [{e.get('time','')}] {e.get('type','')} :: channel={e.get('channel','')} skin={e.get('skin','')}\n")
            self.comms_text.insert("end", f"  hash={e.get('payload_hash','')[:18]}… len={e.get('len',0)}\n")
        self.comms_text.see("end")

    def _render_mesh(self, limit=30):
        self.mesh_text.delete("1.0", "end")
        stats = self.mesh.stats()
        self.mesh_text.insert("end", f"Mesh stats: nodes={stats['total']} discovered={stats['discovered']} built={stats['built']} enforced={stats['enforced']} corridors={stats['corridors']}\n\n")
        events = self.memory.data.get("mesh_events", [])[-limit:]
        for e in events:
            if e["type"] == "discover":
                self.mesh_text.insert("end", f"- [{e['time']}] DISCOVER {e['url']} risk={e['risk']} links={e['links']}\n")
            elif e["type"] == "build":
                self.mesh_text.insert("end", f"- [{e['time']}] BUILD {e['url']}\n")
            elif e["type"] == "enforce":
                self.mesh_text.insert("end", f"- [{e['time']}] ENFORCE {e['url']} status={e.get('status','')}\n")
        self.mesh_text.see("end")

    def _render_status(self, ev: CrawlEvent, guidance):
        online = "online" if ev.online else "local"
        self.swarm_label.config(text=f"Swarm size: {self.swarm.size()} | Mode: {online} | Skin: {ev.skin}")
        self.status.config(text=f"Theme: {guidance['theme']} | Confidence: {self.learning.confidence} | Dead ends: {self.learning.dead_end_flags} | Breakthroughs: {self.learning.breakthroughs}")
        if guidance.get("breakthrough"):
            self._flash(self.status, ["#99ffcc", "#66ff99", "#99ffcc"], 700)
        elif guidance.get("dead_end"):
            self._flash(self.status, ["#ff9999", "#ff6666", "#ff9999"], 700)

    def _flash(self, widget, colors, duration_ms=900):
        step = duration_ms // max(1, len(colors))
        def cycle(i=0):
            widget.config(fg=colors[i % len(colors)])
            if i < len(colors) * 2:
                self.root.after(step, lambda: cycle(i + 1))
            else:
                widget.config(fg="#cccccc")
        cycle(0)

    def pause(self):
        self.running = False

    def resume(self):
        self.running = True
        self.root.after(0, self._ui_loop)

    def trigger_refrain(self):
        agent = self.swarm.agents[-1]
        phrase = REFRAIN_STEPS[agent.refrain_index % len(REFRAIN_STEPS)]
        hint = agent._hint()
        agent.question = f"{phrase} If {hint}, where does it lead?"
        self.log.append(f"{agent.label} :: [Refrain] {redact(phrase)} (manual) bias: {redact(hint)}")
        agent.refrain_index += 1
        self._flash(self.queen_label, ["#ffffff", "#ffcc66", "#fff2b3"], 900)

    def save_log(self):
        path = self.log.save()
        self.queen_label.config(text=redact(f"Log saved: {path}"))

    def clear_lessons(self):
        self.learning.lessons.clear()
        self.memory.data["lessons"] = []
        self.lessons_text.delete("1.0", "end")
        self.queen_label.config(text="Lessons cleared.")

    def save_memory(self):
        self.memory.save()
        self.queen_label.config(text=f"Memory saved → {MEMORY_FILE}")

# ============================================================
# Auto loader
# ============================================================

def auto_loader():
    root = tk.Tk()
    RoamingCivilizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    auto_loader()

