import tkinter as tk
import random
import time
import threading
import queue
import re
import datetime
import os
import json
from collections import defaultdict, deque

# Optional libraries (used only when available and online)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# ============================================================
# Configuration — Queen’s safety codex
# ============================================================

CHECK_HOSTS = ["https://www.google.com", "https://example.com"]
CHECK_TIMEOUT = 3
CHECK_INTERVAL_SEC = 10

START_URLS = ["https://example.com/"]
ALLOWED_DOMAINS = {"example.com"}  # Queen allowlist
FORBIDDEN_BEHAVIORS = {
    "port_scanning": True,
    "packet_sniffing": True,
    "posting_forms": True,
    "code_execution_remote": True
}

MAX_PAGES = 60
MAX_DEPTH = 2
REQUEST_DELAY_SEC = (1.2, 2.4)  # randomized delay range
REQUEST_TIMEOUT_SEC = 8
BURST_LIMIT_REAL = 12
BURST_LIMIT_LOCAL = 10

MEMORY_FILE = "roaming_civilization_memory.json"
AUTO_SAVE_INTERVAL_SEC = 15

# Replication governance
MAX_AGENTS = 12
REPLICATION_COOLDOWN_SEC = 20  # Queen can slow replication
BREAKTHROUGH_REQUIRED = True   # only replicate on breakthroughs

# Logging and privacy
LOG_LEVEL = "normal"  # normal | minimal | forensic
REDACT_PII = True     # Queen redacts sensitive tokens in logs

# Chameleon skins: rotating user agents / headers
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
# Metaphor vocabularies
# ============================================================

HOLMES_QUESTIONS = [
    "What clue hides in the silence?",
    "Why does the trail vanish?",
    "Where does the evidence lead?",
    "What shadow conceals the truth?",
    "What pattern repeats in the noise?",
    "What anomaly breaks the expected?"
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
    "Who benefits if {answer} appears?"
]

REFRAIN_STEPS = [
    "Is there anything more?",
    "Is there anything beneath this?",
    "Is there anything beyond this?",
    "Is there anything we’ve missed?",
    "Is there anything left unsaid?"
]

# Themes for learning
THEME_KEYWORDS = {
    "scent": ["scent", "smell", "odor"],
    "shadow": ["shadow", "dark", "fog", "silence"],
    "trail": ["trail", "crumbs", "footprint", "footprints", "path", "link"],
    "signal": ["whisper", "rustle", "echo", "anomaly", "pattern"],
}

def detect_theme(text: str):
    a = text.lower()
    for theme, words in THEME_KEYWORDS.items():
        if any(w in a for w in words):
            return theme
    return "misc"

# ============================================================
# Utilities — PII redaction (simple)
# ============================================================

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),    # SSN-like
    re.compile(r"\b\d{10}\b"),               # phone-like
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")# email-like
]

def redact(text: str):
    if not REDACT_PII:
        return text
    safe = text
    for pat in PII_PATTERNS:
        safe = pat.sub("[REDACTED]", safe)
    return safe

# ============================================================
# Persistent memory manager
# ============================================================

class MemoryManager:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.data = {
            "journeys": [],            # [{url, title, time, online, skin}]
            "failures": [],            # [{reason, time}]
            "lessons": [],             # [{lesson, time}]
            "confidence": 50,
            "focus_theme": None,
            "template_success": {},    # {template: count}
            "template_fail": {},       # {template: count}
            "queen_directives": [],    # [{text, time}]
            "vetoes": []               # [{reason, time}]
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
            "text": directive,
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def record_veto(self, reason):
        self.data["vetoes"].append({
            "reason": reason,
            "time": datetime.datetime.now().isoformat(timespec="seconds")
        })

    def update_conf_theme(self, confidence, focus_theme):
        self.data["confidence"] = int(confidence)
        self.data["focus_theme"] = focus_theme

    def update_template_stats(self, success_dict, fail_dict):
        self.data["template_success"] = success_dict
        self.data["template_fail"] = fail_dict

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
            self._lesson(f"Dead end: '{answer}' repeated; pivot approach.")
            self.memory.record_failure(f"Dead end from answer '{answer}'")
        if breakthrough:
            self.template_success[template] += 1
            self.breakthroughs += 1
            self.confidence = min(95, self.confidence + 4)
            self._lesson(f"Breakthrough via theme '{theme}': '{answer}'.")
            self.memory.record_lesson(f"Breakthrough on theme '{theme}' via '{answer}'")
            self.focus_theme = theme

        if refrain:
            self._lesson("Refrain asked: re-evaluating assumptions.")
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

    def _lesson(self, text: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.lessons.append(line)
        self.memory.record_lesson(text)

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
        if url in self.nodes:
            self.canvas.itemconfig(self.nodes[url]["id"], fill=color)
            self.canvas.itemconfig(self.nodes[url]["label"], text=self._short(title))
            return self.nodes[url]["id"]
        x = random.randint(60, max(120, self.width - 60))
        y = random.randint(40, max(100, self.height - 40))
        cid = self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline="")
        label = self.canvas.create_text(x+12, y-12, text=self._short(title), fill="#cccccc", font=("Helvetica", 9))
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
# Auto-adaptive chameleon crawler with memory hooks
# ============================================================

class AutoChameleonCrawler(threading.Thread):
    def __init__(self, out_q: queue.Queue, memory: MemoryManager):
        super().__init__(daemon=True)
        self.out = out_q
        self.memory = memory
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

            if self.mode == "real" and requests is not None and BeautifulSoup is not None:
                self._crawl_real()
            else:
                self._crawl_local()

    def _internet_alive(self):
        if requests is None:
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
            if not self._allowed(url):
                self.memory.record_failure(f"Disallowed domain for {url}")
                continue
            seen.add(url)
            try:
                headers, skin = chameleon_headers()
                resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                    self.memory.record_failure(f"Non-HTML or status {resp.status_code} at {url}")
                    time.sleep(random.uniform(*REQUEST_DELAY_SEC))
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                snippet = self._extract_snippet(soup)
                links = self._extract_links(url, soup)
                self.out.put(CrawlEvent(url, title, snippet, links, skin=skin, online=True))
                self.memory.record_journey(url, title, True, skin)
                count += 1
                random.shuffle(links)
                for link in links[:20]:
                    frontier.append((link, depth + 1))
            except Exception as e:
                self.memory.record_failure(f"Exception at {url}: {type(e).__name__}")
            time.sleep(random.uniform(*REQUEST_DELAY_SEC))

    def _allowed(self, url: str):
        try:
            domain = re.findall(r"https?://([^/]+)/", url)[0]
        except Exception:
            return False
        return domain in ALLOWED_DOMAINS

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
        self.log.append(f"{self.label} :: Holmes Q{self.iteration}: {self.question}")
        online_str = "online" if ev.online else "local"
        self.log.append(f"{self.label} :: [Trail] Entered ({online_str}, skin={ev.skin}): {ev.title} ({ev.url})")

        answer = self._answer_from_snippet(ev.snippet)
        self.log.append(f"{self.label} :: Dog A{self.iteration}: {answer}")

        refrain_hit = (self.iteration % 5 == 0)
        template_used = self._template_from_question(self.question)
        guidance = self.learning.score(answer, template_used, refrain_hit)

        if refrain_hit:
            phrase = REFRAIN_STEPS[self.refrain_index % len(REFRAIN_STEPS)]
            hint = self._hint()
            self.question = f"{phrase} If {hint}, where does it lead?"
            self.log.append(f"{self.label} :: [Refrain] {phrase} (bias: {hint})")
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
        text = snippet.lower()
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
# Replicating swarm and Borg Queen governance
# ============================================================

class ReplicatingSwarm:
    def __init__(self, log: NarrativeLog, learning: LearningState, max_agents=MAX_AGENTS):
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

        # Replication check under Queen governance
        now = time.time()
        cooldown_ok = (now - self._last_replication) >= REPLICATION_COOLDOWN_SEC
        replicate_allowed = queen_guard_ok and cooldown_ok and (len(self.agents) < self.max_agents)
        trigger_ok = (any_breakthrough or not BREAKTHROUGH_REQUIRED)

        if replicate_allowed and trigger_ok:
            self._agent_count += 1
            label = f"DRONE-{self._agent_count}"
            new_agent = SherlockDogEngine(self.log, self.learning, label=label)
            self.agents.append(new_agent)
            self._last_replication = now
            self.log.append(f"[Queen] Replication sanctioned: {label} assimilated (total {len(self.agents)}).")

        return guidance_sample, last_answer

    def size(self):
        return len(self.agents)

class BorgQueen:
    def __init__(self, memory: MemoryManager, swarm: ReplicatingSwarm):
        self.memory = memory
        self.swarm = swarm
        self.directive = "Assimilate knowledge, never harm."
        self.last_update = 0
        self.veto_active = False
        self.veto_reason = ""

    def evaluate(self):
        # Periodic directive update
        now = time.time()
        if now - self.last_update < 5:
            return self.directive
        conf = self.swarm.agents[0].learning.confidence if self.swarm.agents else 50
        focus = self.swarm.agents[0].learning.focus_theme if self.swarm.agents else None

        if focus:
            self.directive = f"Queen directive: deepen focus on {focus} (conf {conf})."
        else:
            self.directive = f"Queen directive: expand into unknown paths (conf {conf})."

        self.memory.record_directive(self.directive)
        self.last_update = now
        return self.directive

    def guardrails_ok(self, event: CrawlEvent):
        # Enforce allowlist and forbidden behaviors (narrative)
        if event.online:
            try:
                domain = re.findall(r"https?://([^/]+)/", event.url)[0]
                if domain not in ALLOWED_DOMAINS:
                    self.veto_active = True
                    self.veto_reason = f"Disallowed domain: {domain}"
                    self.memory.record_veto(self.veto_reason)
                    return False
            except Exception:
                pass
        # No scanning/sniffing supported in code; narrative only
        if FORBIDDEN_BEHAVIORS.get("port_scanning") or FORBIDDEN_BEHAVIORS.get("packet_sniffing"):
            pass
        self.veto_active = False
        self.veto_reason = ""
        return True

# ============================================================
# GUI application
# ============================================================

class RoamingCivilizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Roaming Civilization — Queen‑governed Collective")
        self.root.geometry("1280x900")
        self.root.configure(bg="#0b0b0b")

        # Header
        header = tk.Frame(root, bg="#c2a35d", height=54)
        header.pack(fill="x", side="top")
        tk.Label(header, text="FOREST REACHES — Assimilate knowledge, never harm",
                 font=("Courier", 16, "bold"), fg="#1a1a1a", bg="#c2a35d").pack(padx=12, pady=12, anchor="w")

        # Layout
        top = tk.Frame(root, bg="#0b0b0b")
        top.pack(fill="both", expand=True, padx=10, pady=8)
        left = tk.Frame(top, bg="#0b0b0b")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(top, bg="#0b0b0b", width=480)
        right.pack(side="right", fill="y", padx=(8, 0))

        # Forest map
        self.map = tk.Canvas(left, bg="#101010", highlightthickness=0)
        self.map.pack(fill="both", expand=True)
        self.forest = ForestMap(self.map)
        self.map.bind("<Configure>", self._resize_map)

        # Collective log
        bottom = tk.Frame(root, bg="#0b0b0b", height=240)
        bottom.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        tk.Label(bottom, text="Collective log", font=("Helvetica", 11, "bold"),
                 fg="#cccccc", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 0))
        self.log_text = tk.Text(bottom, bg="#121212", fg="#e8e8e8", height=10, wrap="word",
                                insertbackground="#e8e8e8")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = NarrativeLog(self.log_text)

        # Right panels
        self.queen_label = tk.Label(right, text="", font=("Courier", 14, "bold"),
                                    fg="#ffd966", bg="#0b0b0b", wraplength=440, justify="left")
        self.queen_label.pack(pady=(8, 6), anchor="w")

        self.swarm_label = tk.Label(right, text="", font=("Courier", 12, "bold"),
                                    fg="#cccccc", bg="#0b0b0b", wraplength=440, justify="left")
        self.swarm_label.pack(pady=(2, 8), anchor="w")

        self.holmes_label = tk.Label(right, text="", font=("Courier", 14, "bold"),
                                     fg="#ffd966", bg="#0b0b0b", wraplength=440, justify="left")
        self.holmes_label.pack(pady=(4, 4), anchor="w")
        self.dog_label = tk.Label(right, text="", font=("Comic Sans MS", 13, "italic"),
                                  fg="#99d9ea", bg="#0b0b0b", wraplength=440, justify="left")
        self.dog_label.pack(pady=(2, 12), anchor="w")

        tk.Label(right, text="Learning ledger", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.lessons_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=10, wrap="word",
                                    insertbackground="#f0f0f0")
        self.lessons_text.pack(fill="both", expand=False, padx=6, pady=(0, 8))

        tk.Label(right, text="Memory (journeys, failures, directives, vetoes)", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.memory_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=10, wrap="word",
                                   insertbackground="#f0f0f0")
        self.memory_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))

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
        tk.Label(safety_box, text="Queen safety toggles", fg="#cccccc", bg="#0b0b0b").pack(anchor="w")
        self.allow_replication_var = tk.BooleanVar(value=True)
        tk.Checkbutton(safety_box, text="Allow replication", variable=self.allow_replication_var,
                       bg="#0b0b0b", fg="#cccccc", activebackground="#0b0b0b", selectcolor="#222222").pack(anchor="w")
        self.enforce_allowlist_var = tk.BooleanVar(value=True)
        tk.Checkbutton(safety_box, text="Enforce domain allowlist", variable=self.enforce_allowlist_var,
                       bg="#0b0b0b", fg="#cccccc", activebackground="#0b0b0b", selectcolor="#222222").pack(anchor="w")

        speed_box = tk.Frame(right, bg="#0b0b0b")
        speed_box.pack(fill="x", padx=6, pady=6)
        tk.Label(speed_box, text="UI refresh (ms)", fg="#cccccc", bg="#0b0b0b").pack(anchor="w")
        self.speed_scale = tk.Scale(speed_box, from_=500, to=5000, orient="horizontal",
                                    bg="#0b0b0b", fg="#cccccc", troughcolor="#333333",
                                    highlightthickness=0)
        self.speed_scale.set(2000)
        self.speed_scale.pack(fill="x")

        # State
        self.memory = MemoryManager()
        self.learning = LearningState(self.memory)
        self.events = queue.Queue()
        self.crawler = AutoChameleonCrawler(self.events, self.memory)
        self.crawler.start()

        # Swarm + Queen
        self.swarm = ReplicatingSwarm(self.log, self.learning, max_agents=MAX_AGENTS)
        self.queen = BorgQueen(self.memory, self.swarm)

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
        # Queen guardrails on event
        queen_guard_ok = True
        if self.enforce_allowlist_var.get():
            queen_guard_ok = self.queen.guardrails_ok(ev)
            if not queen_guard_ok:
                self.log.append(f"[Queen Veto] {self.queen.veto_reason}. Event ignored.")
                # Still visualize node but mark as quarantined (amber pulse)
                color = "#ffcc66"
                self.forest.add_node(ev.url, ev.title, color=color)
                self.forest.pulse(ev.url, color="#ff9966")
                self._render_status(ev, {"theme": "misc", "dead_end": False, "breakthrough": False})
                self._render_memory(limit=24)
                return

        # Color by skin identity
        color = SKIN_COLORS.get(ev.skin, SKIN_COLORS["Default"])
        self.forest.add_node(ev.url, ev.title, color=color)
        for link in ev.links[:14]:
            self.forest.add_node(link, link, color="#335577")
            self.forest.add_edge(ev.url, link)
        self.forest.pulse(ev.url, color="#99ffcc" if ev.online else "#ffcc66")

        # Swarm processes event under Queen’s replication allowance
        guidance, last_answer = self.swarm.step(ev, queen_guard_ok=self.allow_replication_var.get())

        # Queen evaluates and issues directive
        directive = self.queen.evaluate()
        self.queen_label.config(text=directive)
        self.log.append(f"[Queen] {directive}")

        # Representative agent
        representative_agent = self.swarm.agents[-1]
        self.holmes_label.config(text=f"{representative_agent.label} :: Holmes: {representative_agent.question}")
        self.dog_label.config(text=f"{representative_agent.label} :: Dog: {last_answer}")

        # Panels
        self._render_lessons()
        self._render_memory(limit=24)
        self._render_status(ev, guidance)

    def _render_lessons(self):
        self.lessons_text.delete("1.0", "end")
        for line in list(self.learning.lessons):
            self.lessons_text.insert("end", line + "\n")

    def _render_memory(self, limit=30):
        self.memory_text.delete("1.0", "end")
        journeys = self.memory.data.get("journeys", [])[-limit:]
        failures = self.memory.data.get("failures", [])[-limit:]
        directives = self.memory.data.get("queen_directives", [])[-limit:]
        vetoes = self.memory.data.get("vetoes", [])[-limit:]

        self.memory_text.insert("end", "Journeys:\n")
        for j in journeys:
            self.memory_text.insert("end", f"- [{j.get('time','')}] ({'online' if j.get('online') else 'local'}, skin={j.get('skin')}) {j.get('title')} — {j.get('url')}\n")
        self.memory_text.insert("end", "\nFailures:\n")
        for f in failures:
            self.memory_text.insert("end", f"- [{f.get('time','')}] {f.get('reason','')}\n")
        self.memory_text.insert("end", "\nQueen directives:\n")
        for d in directives:
            self.memory_text.insert("end", f"- [{d.get('time','')}] {d.get('text','')}\n")
        self.memory_text.insert("end", "\nQueen vetoes:\n")
        for v in vetoes:
            self.memory_text.insert("end", f"- [{v.get('time','')}] {v.get('reason','')}\n")

        self.memory_text.see("end")

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
        self.log.append(f"{agent.label} :: [Refrain] {phrase} (manual) bias: {hint}")
        agent.refrain_index += 1
        self._flash(self.queen_label, ["#ffffff", "#ffcc66", "#fff2b3"], 900)

    def save_log(self):
        path = self.log.save()
        self.queen_label.config(text=f"Log saved: {path}")

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

