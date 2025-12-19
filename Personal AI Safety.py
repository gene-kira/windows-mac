import sys
import subprocess
import importlib
import json
import os
import re
import threading
import time
import hashlib
import base64
import platform
import uuid
from urllib.parse import urlparse, parse_qs

# ============================================================
#  AUTO-LOADER FOR OPTIONAL LIBRARIES (EXTENSIBLE)
# ============================================================

REQUIRED_PACKAGES = [
    "cryptography",  # for encryption of local data
]


def ensure_packages_installed(packages):
    """
    Try to import each package; if import fails, install via pip, then import.
    This keeps the script self-contained and auto-resolving dependencies.
    """
    for pkg in packages:
        module_name = pkg.replace('-', '_')
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[AUTOLOADER] Installing missing package: {pkg}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                importlib.import_module(module_name)
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install {pkg}: {e}")


ensure_packages_installed(REQUIRED_PACKAGES)

# Now we can safely import cryptography and GUI + HTTP bits
from cryptography.fernet import Fernet
import tkinter as tk
from tkinter import messagebox
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

# ============================================================
#  FILE LOCATIONS / STORAGE
# ============================================================

DATA_FILE = "web_guardian_habits.dat"      # encrypted
HISTORY_FILE = "web_guardian_history.dat"  # encrypted
CACHE_FILE = "web_guardian_cache.dat"      # encrypted

API_HOST = "127.0.0.1"
API_PORT = 8765

# ============================================================
#  ENCRYPTION UTILITIES (MACHINE-TIED KEY)
# ============================================================

def get_machine_secret():
    """
    Build a machine-specific secret string.
    Not bulletproof, but ties data to this environment.
    """
    parts = [
        platform.system(),
        platform.node(),
        platform.machine(),
        str(uuid.getnode()),
    ]
    return "|".join(parts)


def get_fernet():
    """
    Derive a Fernet key from the machine secret using SHA-256.
    """
    secret = get_machine_secret().encode("utf-8")
    digest = hashlib.sha256(secret).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


FERNET = get_fernet()


def encrypt_bytes(data: bytes) -> bytes:
    return FERNET.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    return FERNET.decrypt(data)


# ============================================================
#  ENCRYPTED JSON FILE HELPERS
# ============================================================

def load_json_file(path, default):
    """
    Load JSON from an encrypted file.
    If file missing/corrupted/decryption fails, return default.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "rb") as f:
            enc = f.read()
        if not enc:
            return default
        raw = decrypt_bytes(enc)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return default


def save_json_file(path, data):
    """
    Save JSON to an encrypted file.
    """
    try:
        raw = json.dumps(data, indent=2).encode("utf-8")
        enc = encrypt_bytes(raw)
        with open(path, "wb") as f:
            f.write(enc)
    except Exception as e:
        print(f"[STORAGE] Failed to save {path}: {e}")


# ============================================================
#  HABIT STORAGE (TRUSTED / BLOCKED SITES)
# ============================================================

def load_habits():
    data = load_json_file(DATA_FILE, {"trusted": [], "blocked": []})
    data.setdefault("trusted", [])
    data.setdefault("blocked", [])
    return data


def save_habits(habits):
    save_json_file(DATA_FILE, habits)


def normalize_domain(url):
    """
    Extract a normalized domain from the URL.
    - Ensures scheme is present for parsing.
    - Strips 'www.' and port.
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        host = host.split(':')[0]  # strip port
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


# ============================================================
#  ACTION HISTORY & PREDICTION ENGINE (RITA-STYLE)
# ============================================================

def load_history():
    data = load_json_file(HISTORY_FILE, {"actions": []})
    data.setdefault("actions", [])
    return data


def save_history(history):
    save_json_file(HISTORY_FILE, history)


def log_action(action_type, metadata=None):
    """
    Log an action taken by the user:
    action_type: string, e.g. "check_url", "mark_trusted", "mark_blocked"
    metadata: dict with extra information, e.g. {"domain": "example.com"}
    """
    history = load_history()
    if metadata is None:
        metadata = {}
    entry = {
        "type": action_type,
        "meta": metadata,
        "ts": time.time()
    }
    history["actions"].append(entry)
    # Keep history from growing too large
    if len(history["actions"]) > 500:
        history["actions"] = history["actions"][-500:]
    save_history(history)


def get_recent_actions(limit=30):
    history = load_history()
    return history["actions"][-limit:]


def build_transition_counts(actions):
    """
    Build a simple Markov-like transitions dict:
    transitions[from_type][to_type] = count
    """
    transitions = {}
    for i in range(len(actions) - 1):
        a = actions[i]["type"]
        b = actions[i + 1]["type"]
        if a not in transitions:
            transitions[a] = {}
        transitions[a][b] = transitions[a].get(b, 0) + 1
    return transitions


def predict_next_actions():
    """
    Predict likely next actions based on last action and transitions.
    Returns a list of (action_type, score) sorted by score descending.
    """
    actions = get_recent_actions()
    if len(actions) < 2:
        # Not enough data, fallback to default guess
        return [("check_url", 0.5), ("mark_trusted", 0.25), ("mark_blocked", 0.25)]

    transitions = build_transition_counts(actions)
    last_action_type = actions[-1]["type"]

    if last_action_type not in transitions:
        return [("check_url", 0.5), ("mark_trusted", 0.25), ("mark_blocked", 0.25)]

    next_counts = transitions[last_action_type]
    total = sum(next_counts.values())
    if total == 0:
        return [("check_url", 0.5), ("mark_trusted", 0.25), ("mark_blocked", 0.25)]

    predicted = []
    for a_type, count in next_counts.items():
        score = count / total
        predicted.append((a_type, score))

    predicted.sort(key=lambda x: x[1], reverse=True)
    return predicted


# ============================================================
#  SIMPLE URL / PHISHING HEURISTICS
# ============================================================

def is_ip_address(domain):
    return re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain) is not None


def count_suspicious_chars(url):
    suspicious_set = set("@%$!^*{}[]|\\`~")
    return sum(1 for c in url if c in suspicious_set)


def is_punycode(domain):
    return "xn--" in domain


def subdomain_depth(domain):
    return len(domain.split("."))


def basic_url_risk_score(url, habits):
    """
    Compute a heuristic risk score for a URL.
    Returns (score, verdict, reasons)
    score: 0-100 (higher = more risky)
    verdict: 'trusted', 'blocked', 'suspicious', 'unknown'
    reasons: list of strings
    """
    reasons = []
    domain = normalize_domain(url)
    if not domain:
        return 80, "suspicious", ["Invalid or unparsable URL"]

    trusted = habits.get("trusted", [])
    blocked = habits.get("blocked", [])
    if domain in blocked:
        return 95, "blocked", [f"Domain {domain} is in your blocked list"]
    if domain in trusted:
        reasons.append(f"Domain {domain} is in your trusted list")
        base_score = 5
    else:
        base_score = 20

    if is_ip_address(domain):
        base_score += 25
        reasons.append("Domain is a raw IP address (common in phishing).")

    if is_punycode(domain):
        base_score += 25
        reasons.append("Domain uses punycode (possible homograph attack).")

    depth = subdomain_depth(domain)
    if depth >= 4:
        base_score += 15
        reasons.append(f"Domain has many subdomains ({depth}).")

    url_len = len(url)
    if url_len > 80:
        base_score += 10
        reasons.append(f"URL is very long ({url_len} characters).")

    sus_chars = count_suspicious_chars(url)
    if sus_chars > 3:
        base_score += 10
        reasons.append(f"URL contains many suspicious characters ({sus_chars}).")

    score = max(0, min(100, base_score))

    if domain in trusted and score < 30:
        verdict = "trusted"
    elif domain in blocked or score >= 70:
        verdict = "suspicious"
    elif score >= 40:
        verdict = "unknown"
    else:
        verdict = "unknown"

    return score, verdict, reasons


# ============================================================
#  CACHE / PRELOAD MANAGER (RITA-STYLE READ-AHEAD)
# ============================================================

def load_cache():
    data = load_json_file(CACHE_FILE, {
        "trusted_set": [],
        "blocked_set": [],
        "precomputed_scores": {}
    })
    data.setdefault("trusted_set", [])
    data.setdefault("blocked_set", [])
    data.setdefault("precomputed_scores", {})
    return data


def save_cache(cache):
    save_json_file(CACHE_FILE, cache)


def rebuild_cache(habits):
    """
    Rebuild cached sets and any precomputed scores.
    """
    cache = load_cache()
    cache["trusted_set"] = list(set(habits.get("trusted", [])))
    cache["blocked_set"] = list(set(habits.get("blocked", [])))
    save_cache(cache)
    print("[CACHE] Rebuilt cache from habits.")
    return cache


def preload_likely_domains(habits):
    """
    Preload scores for commonly-used domains, based on history.
    Acts like a read-ahead step so checks feel faster.
    """
    history = load_history()
    domain_counts = {}
    for action in history.get("actions", []):
        if action["type"] == "check_url":
            domain = action["meta"].get("domain")
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Pick top 10 domains
    top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    cache = load_cache()
    for domain, _count in top_domains:
        url = "http://" + domain
        score, verdict, reasons = basic_url_risk_score(url, habits)
        cache["precomputed_scores"][domain] = {
            "score": score,
            "verdict": verdict,
            "reasons": reasons
        }
    save_cache(cache)
    print("[CACHE] Preloaded top domains:", [d for d, _ in top_domains])


def get_cached_score_for_domain(domain, habits):
    """
    Use cache if available; otherwise compute and optionally store.
    """
    cache = load_cache()
    pre = cache.get("precomputed_scores", {})
    if domain in pre:
        data = pre[domain]
        return data["score"], data["verdict"], data["reasons"]
    # Not cached, compute now
    url = "http://" + domain
    score, verdict, reasons = basic_url_risk_score(url, habits)
    pre[domain] = {
        "score": score,
        "verdict": verdict,
        "reasons": reasons
    }
    cache["precomputed_scores"] = pre
    save_cache(cache)
    return score, verdict, reasons


# ============================================================
#  BACKGROUND RITA THREAD
# ============================================================

class RitaPreloader(threading.Thread):
    """
    Background thread that:
      - Watches history
      - Predicts likely next actions
      - Preloads accordingly
    """

    def __init__(self, habits_getter, interval=10.0):
        super().__init__(daemon=True)
        self.habits_getter = habits_getter
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            try:
                predicted = predict_next_actions()
                if predicted:
                    top_action, score = predicted[0]
                    if top_action == "check_url":
                        habits = self.habits_getter()
                        rebuild_cache(habits)
                        preload_likely_domains(habits)
                time.sleep(self.interval)
            except Exception as e:
                print(f"[RITA] Background error: {e}")
                time.sleep(self.interval)

    def stop(self):
        self.running = False


# ============================================================
#  HTTP API SERVER
# ============================================================

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class GuardianRequestHandler(BaseHTTPRequestHandler):
    """
    Simple HTTP API:

    - GET /health
      -> {"status": "ok"}

    - GET /check?url=...
      -> {"score": int, "verdict": str, "reasons": [...], "domain": str}
    """

    # These will be set from outside:
    app_ref = None  # WebGuardianApp instance

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/health":
                return self._send_json({"status": "ok"})

            if path == "/check":
                url_param = query.get("url", [""])[0].strip()
                if not url_param:
                    return self._send_json({"error": "missing url parameter"}, status=400)

                app = self.app_ref
                if app is None:
                    return self._send_json({"error": "guardian app not ready"}, status=503)

                domain = normalize_domain(url_param) or "unknown"
                log_action("check_url_api", {"domain": domain})

                if domain != "unknown":
                    score, verdict, reasons = get_cached_score_for_domain(domain, app.habits)
                else:
                    score, verdict, reasons = basic_url_risk_score(url_param, app.habits)

                return self._send_json({
                    "domain": domain,
                    "score": score,
                    "verdict": verdict,
                    "reasons": reasons
                })

            # Unknown path
            return self._send_json({"error": "not found"}, status=404)

        except Exception as e:
            return self._send_json({"error": str(e)}, status=500)

    def log_message(self, format, *args):
        # Silence default logging; uncomment for debug
        # sys.stderr.write("%s - - [%s] %s\n" %
        #                  (self.client_address[0],
        #                   self.log_date_time_string(),
        #                   format%args))
        pass


class APIServerThread(threading.Thread):
    def __init__(self, app_ref, host=API_HOST, port=API_PORT):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.httpd = None
        GuardianRequestHandler.app_ref = app_ref

    def run(self):
        try:
            self.httpd = ThreadedHTTPServer((self.host, self.port), GuardianRequestHandler)
            print(f"[API] HTTP API listening on http://{self.host}:{self.port}")
            self.httpd.serve_forever()
        except Exception as e:
            print(f"[API] Failed to start HTTP server: {e}")

    def stop(self):
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass


# ============================================================
#  GUI APPLICATION
# ============================================================

class WebGuardianApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Web Guardian - RITA Encrypted + API")

        self.habits = load_habits()

        # Start RITA-style preloader
        self.rita_thread = RitaPreloader(self.get_habits, interval=15.0)
        self.rita_thread.start()

        # Start API server
        self.api_thread = APIServerThread(self)
        self.api_thread.start()

        # URL entry
        self.url_label = tk.Label(root, text="Enter URL:")
        self.url_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.url_entry = tk.Entry(root, width=60)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=3, sticky="we")

        self.check_button = tk.Button(root, text="Check URL", command=self.check_url)
        self.check_button.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        self.verdict_var = tk.StringVar(value="Status: idle")
        self.verdict_label = tk.Label(root, textvariable=self.verdict_var, fg="blue")
        self.verdict_label.grid(row=2, column=0, padx=5, pady=5, columnspan=4, sticky="w")

        self.details_text = tk.Text(root, width=80, height=10, state="disabled")
        self.details_text.grid(row=3, column=0, padx=5, pady=5, columnspan=4, sticky="nsew")

        self.trust_button = tk.Button(root, text="Mark as Trusted", command=self.mark_trusted)
        self.trust_button.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        self.block_button = tk.Button(root, text="Mark as Blocked", command=self.mark_blocked)
        self.block_button.grid(row=4, column=2, padx=5, pady=5, sticky="we")

        self.status_var = tk.StringVar(value=f"Ready | API on {API_HOST}:{API_PORT}")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=5, column=0, padx=5, pady=5, columnspan=4, sticky="we")

        # Prediction / RITA status
        self.rita_var = tk.StringVar(value="RITA prediction: learning your habits...")
        self.rita_label = tk.Label(root, textvariable=self.rita_var, fg="purple")
        self.rita_label.grid(row=6, column=0, padx=5, pady=5, columnspan=4, sticky="w")

        # Resize behavior
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)

        self.last_checked_url = None

        # Periodically update RITA prediction label
        self.root.after(5000, self.update_rita_label)

    def get_habits(self):
        return self.habits

    def update_rita_label(self):
        """
        Periodically show what RITA thinks you might do next.
        This is just for transparency / feel.
        """
        try:
            preds = predict_next_actions()
            if not preds:
                self.rita_var.set("RITA prediction: not enough data yet.")
            else:
                parts = [f"{t} ({score:.2f})" for t, score in preds[:3]]
                self.rita_var.set("RITA prediction: " + ", ".join(parts))
        except Exception as e:
            self.rita_var.set(f"RITA prediction error: {e}")
        finally:
            # Schedule next update
            self.root.after(5000, self.update_rita_label)

    def get_current_url(self):
        return self.url_entry.get().strip()

    def check_url(self):
        url = self.get_current_url()
        if not url:
            messagebox.showwarning("No URL", "Please enter a URL to check.")
            return

        self.last_checked_url = url
        domain = normalize_domain(url) or "unknown"

        # Log action
        log_action("check_url", {"domain": domain})

        # Use cache-aware scoring
        if domain != "unknown":
            score, verdict, reasons = get_cached_score_for_domain(domain, self.habits)
        else:
            score, verdict, reasons = basic_url_risk_score(url, self.habits)

        verdict_text = f"Status: {verdict.upper()} (score: {score}/100) for {domain}"
        self.verdict_var.set(verdict_text)

        if verdict == "trusted":
            color = "green"
        elif verdict in ("suspicious", "blocked"):
            color = "red"
        else:
            color = "orange"
        self.verdict_label.config(fg=color)

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        if reasons:
            self.details_text.insert(tk.END, "Reasons:\n")
            for r in reasons:
                self.details_text.insert(tk.END, f" - {r}\n")
        else:
            self.details_text.insert(tk.END, "No specific reasons. Default classification.\n")

        if domain in self.habits.get("trusted", []):
            self.details_text.insert(tk.END, f"\nNote: {domain} is in your TRUSTED list.\n")
        if domain in self.habits.get("blocked", []):
            self.details_text.insert(tk.END, f"\nNote: {domain} is in your BLOCKED list.\n")

        self.details_text.config(state="disabled")
        self.status_var.set(f"Last check completed | API on {API_HOST}:{API_PORT}")

    def mark_trusted(self):
        url = self.get_current_url() or self.last_checked_url
        if not url:
            messagebox.showwarning("No URL", "Check a URL first or enter one to mark.")
            return
        domain = normalize_domain(url)
        if not domain:
            messagebox.showerror("Invalid URL", "Cannot parse this URL; cannot mark as trusted.")
            return

        if domain not in self.habits["trusted"]:
            self.habits["trusted"].append(domain)
        if domain in self.habits["blocked"]:
            self.habits["blocked"].remove(domain)

        save_habits(self.habits)
        rebuild_cache(self.habits)

        log_action("mark_trusted", {"domain": domain})

        self.status_var.set(f"{domain} added to trusted sites | API on {API_HOST}:{API_PORT}")
        messagebox.showinfo("Trusted", f"{domain} is now trusted.")
        self.check_url()

    def mark_blocked(self):
        url = self.get_current_url() or self.last_checked_url
        if not url:
            messagebox.showwarning("No URL", "Check a URL first or enter one to mark.")
            return
        domain = normalize_domain(url)
        if not domain:
            messagebox.showerror("Invalid URL", "Cannot parse this URL; cannot mark as blocked.")
            return

        if domain not in self.habits["blocked"]:
            self.habits["blocked"].append(domain)
        if domain in self.habits["trusted"]:
            self.habits["trusted"].remove(domain)

        save_habits(self.habits)
        rebuild_cache(self.habits)

        log_action("mark_blocked", {"domain": domain})

        self.status_var.set(f"{domain} added to blocked sites | API on {API_HOST}:{API_PORT}")
        messagebox.showinfo("Blocked", f"{domain} is now blocked.")
        self.check_url()

    def shutdown(self):
        try:
            if self.rita_thread is not None:
                self.rita_thread.stop()
        except Exception:
            pass

        try:
            if self.api_thread is not None:
                self.api_thread.stop()
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = WebGuardianApp(root)

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

