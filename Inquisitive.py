import tkinter as tk
import random
import time
import threading
import queue
import re
import datetime
import os
from collections import defaultdict, deque

# Optional libraries (used only when available and online)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# ============================================================
# Configuration
# ============================================================

# Start online if reachable; auto-switches at runtime
CHECK_HOSTS = ["https://www.google.com", "https://example.com"]
CHECK_TIMEOUT = 3
CHECK_INTERVAL_SEC = 10

START_URLS = ["https://example.com/"]
ALLOWED_DOMAINS = {"example.com"}
MAX_PAGES = 60
MAX_DEPTH = 2
REQUEST_DELAY_SEC = (1.2, 2.4)  # randomized delay range
REQUEST_TIMEOUT_SEC = 8

# Chameleon skins: rotating user agents / headers
CHAMELEON_SKINS = [
    # Desktop
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Gecko/20100101 Firefox/119.0",
    # Mobile
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
# Logging
# ============================================================

class NarrativeLog:
    def __init__(self, widget):
        self.widget = widget
        self.widget.configure(state="disabled")
        self.buffer = []

    def append(self, line: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
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
        name = f"case_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write("".join(self.buffer))
        return os.path.abspath(name)

# ============================================================
# Learning
# ============================================================

class LearningState:
    def __init__(self, history_size=50):
        self.recent_answers = deque(maxlen=history_size)
        self.theme_counts = defaultdict(int)
        self.template_success = defaultdict(int)
        self.template_fail = defaultdict(int)
        self.dead_end_flags = 0
        self.breakthroughs = 0
        self.lessons = deque(maxlen=16)
        self.confidence = 50
        self.focus_theme = None

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
        if breakthrough:
            self.template_success[template] += 1
            self.breakthroughs += 1
            self.confidence = min(95, self.confidence + 4)
            self._lesson(f"Breakthrough via theme '{theme}': '{answer}'.")
            self.focus_theme = theme

        if refrain:
            self._lesson("Refrain asked: re-evaluating assumptions.")
            self.confidence = max(10, min(90, self.confidence + (1 if breakthrough else -1)))

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
        self.lessons.append(f"[{ts}] {text}")

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
            # update color/label if necessary
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
# Auto-adaptive chameleon crawler
# ============================================================

class AutoChameleonCrawler(threading.Thread):
    def __init__(self, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.out = out_q
        self.running = True
        self.mode = "local"  # will auto-switch
        self.last_check = 0

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            now = time.time()
            # Heartbeat connectivity check
            if now - self.last_check >= CHECK_INTERVAL_SEC:
                self.mode = "real" if self._internet_alive() else "local"
                self.last_check = now

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
        # limited session crawl burst before next heartbeat
        burst_limit = 12
        while self.running and self.mode == "real" and frontier and count < burst_limit:
            url, depth = frontier.popleft()
            if url in seen or depth > MAX_DEPTH:
                continue
            if not self._allowed(url):
                continue
            seen.add(url)
            try:
                headers, skin = chameleon_headers()
                resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                    time.sleep(random.uniform(*REQUEST_DELAY_SEC))
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                snippet = self._extract_snippet(soup)
                links = self._extract_links(url, soup)
                self.out.put(CrawlEvent(url, title, snippet, links, skin=skin, online=True))
                count += 1
                # randomized link selection order
                random.shuffle(links)
                for link in links[:20]:
                    frontier.append((link, depth + 1))
            except Exception:
                pass
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
        graph = defaultdict(list)
        for url, title, snippet, links in simulated_pages:
            graph[url] = links

        visited = set()
        frontier = deque(["forest://clearing"])
        depth_map = {"forest://clearing": 0}
        burst_limit = 10
        count = 0

        while self.running and self.mode == "local" and frontier and count < burst_limit:
            url = frontier.popleft()
            depth = depth_map.get(url, 0)
            if url in visited or depth > 3:
                continue
            visited.add(url)
            for u, title, snippet, links in simulated_pages:
                if u == url:
                    self.out.put(CrawlEvent(u, title, snippet, links, skin="Default", online=False))
                    for link in links:
                        if link not in visited:
                            frontier.append(link)
                            depth_map[link] = depth + 1
                    count += 1
                    break
            time.sleep(random.uniform(*REQUEST_DELAY_SEC))

# ============================================================
# Sherlock & Dog engine
# ============================================================

class SherlockDogEngine:
    def __init__(self, log: NarrativeLog, learning: LearningState):
        self.iteration = 1
        self.question = random.choice(HOLMES_QUESTIONS)
        self.refrain_index = 0
        self.log = log
        self.learning = learning

    def on_event(self, ev: CrawlEvent):
        # Holmes asks / page entered
        self.log.append(f"Holmes Q{self.iteration}: {self.question}")
        online_str = "online" if ev.online else "local"
        self.log.append(f"[Trail] Entered ({online_str}, skin={ev.skin}): {ev.title} ({ev.url})")

        # Dog answers based on snippet themes
        answer = self._answer_from_snippet(ev.snippet)
        self.log.append(f"Dog A{self.iteration}: {answer}")

        # Learning score
        refrain_hit = (self.iteration % 5 == 0)
        template_used = self._template_from_question(self.question)
        guidance = self.learning.score(answer, template_used, refrain_hit)

        # Refrain or next question
        if refrain_hit:
            phrase = REFRAIN_STEPS[self.refrain_index % len(REFRAIN_STEPS)]
            hint = self._hint()
            self.question = f"{phrase} If {hint}, where does it lead?"
            self.log.append(f"[Refrain] {phrase} (bias: {hint})")
            self.refrain_index += 1
        else:
            template = self.learning.choose_template()
            next_q = template.format(answer=answer)
            self.question = self.learning.bias(next_q)

        if ev.links:
            self.log.append(f"[Paths] {len(ev.links)} trails visible.")
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
# GUI application
# ============================================================

class RoamingSpotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sherlock & Curious Dog — Roaming Spot (Auto + Chameleon)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0b0b0b")

        # Header
        header = tk.Frame(root, bg="#c2a35d", height=54)
        header.pack(fill="x", side="top")
        tk.Label(header, text="FOREST REACHES — The Question That Learns",
                 font=("Courier", 16, "bold"), fg="#1a1a1a", bg="#c2a35d").pack(padx=12, pady=12, anchor="w")

        # Layout
        top = tk.Frame(root, bg="#0b0b0b")
        top.pack(fill="both", expand=True, padx=10, pady=8)
        left = tk.Frame(top, bg="#0b0b0b")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(top, bg="#0b0b0b", width=400)
        right.pack(side="right", fill="y", padx=(8, 0))

        # Forest map
        self.map = tk.Canvas(left, bg="#101010", highlightthickness=0)
        self.map.pack(fill="both", expand=True)
        self.forest = ForestMap(self.map)
        self.map.bind("<Configure>", self._resize_map)

        # Case log
        bottom = tk.Frame(root, bg="#0b0b0b", height=240)
        bottom.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        tk.Label(bottom, text="Case log", font=("Helvetica", 11, "bold"),
                 fg="#cccccc", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 0))
        self.log_text = tk.Text(bottom, bg="#121212", fg="#e8e8e8", height=10, wrap="word",
                                insertbackground="#e8e8e8")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = NarrativeLog(self.log_text)

        # Right panels
        self.holmes_label = tk.Label(right, text="", font=("Courier", 15, "bold"),
                                     fg="#ffd966", bg="#0b0b0b", wraplength=360, justify="left")
        self.holmes_label.pack(pady=(8, 4), anchor="w")
        self.dog_label = tk.Label(right, text="", font=("Comic Sans MS", 14, "italic"),
                                  fg="#99d9ea", bg="#0b0b0b", wraplength=360, justify="left")
        self.dog_label.pack(pady=(4, 8), anchor="w")

        self.pulse = tk.Label(right, text="", font=("Helvetica", 12, "bold"),
                              fg="#ffffff", bg="#0b0b0b")
        self.pulse.pack(pady=(0, 8), anchor="w")

        tk.Label(right, text="Learning ledger", font=("Helvetica", 12, "bold"),
                 fg="#eeeeee", bg="#0b0b0b").pack(anchor="w", padx=6, pady=(6, 2))
        self.lessons_text = tk.Text(right, bg="#121212", fg="#f0f0f0", height=12, wrap="word",
                                    insertbackground="#f0f0f0")
        self.lessons_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))

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

        speed_box = tk.Frame(right, bg="#0b0b0b")
        speed_box.pack(fill="x", padx=6, pady=6)
        tk.Label(speed_box, text="UI refresh (ms)", fg="#cccccc", bg="#0b0b0b").pack(anchor="w")
        self.speed_scale = tk.Scale(speed_box, from_=500, to=5000, orient="horizontal",
                                    bg="#0b0b0b", fg="#cccccc", troughcolor="#333333",
                                    highlightthickness=0)
        self.speed_scale.set(2000)
        self.speed_scale.pack(fill="x")

        # State
        self.learning = LearningState()
        self.engine = SherlockDogEngine(self.log, self.learning)

        # Event queue and crawler
        self.events = queue.Queue()
        self.crawler = AutoChameleonCrawler(self.events)
        self.crawler.start()

        # Start UI loop
        self.running = True
        self._ui_loop()

    def _resize_map(self, event):
        self.forest.set_size(event.width, event.height)

    def _ui_loop(self):
        if not self.running:
            return
        processed = 0
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            self._handle(ev)
        self.root.after(int(self.speed_scale.get()), self._ui_loop)

    def _handle(self, ev: CrawlEvent):
        # Color by skin identity
        color = SKIN_COLORS.get(ev.skin, SKIN_COLORS["Default"])
        self.forest.add_node(ev.url, ev.title, color=color)
        for link in ev.links[:14]:
            self.forest.add_node(link, link, color="#335577")
            self.forest.add_edge(ev.url, link)
        self.forest.pulse(ev.url, color="#99ffcc" if ev.online else "#ffcc66")

        guidance, last_answer = self.engine.on_event(ev)

        # Update panels
        self.holmes_label.config(text=f"Holmes: {self.engine.question}")
        self.dog_label.config(text=f"Dog: {last_answer}")

        # Lessons
        self.lessons_text.delete("1.0", "end")
        for line in list(self.learning.lessons):
            self.lessons_text.insert("end", line + "\n")

        # Status
        online = "online" if ev.online else "local"
        self.status.config(text=f"Mode: {online} | Skin: {ev.skin} | Theme: {guidance['theme']} | "
                                f"Confidence: {self.learning.confidence} | Dead ends: {self.learning.dead_end_flags} | Breakthroughs: {self.learning.breakthroughs}")

        # Pulse
        if guidance["breakthrough"]:
            self._flash(self.status, ["#99ffcc", "#66ff99", "#99ffcc"], 700)
            self.pulse.config(text="Breakthrough")
        elif guidance["dead_end"]:
            self._flash(self.status, ["#ff9999", "#ff6666", "#ff9999"], 700)
            self.pulse.config(text="Dead end — pivot")
        else:
            self.pulse.config(text="")

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
        self.pulse.config(text="UI Paused")

    def resume(self):
        self.running = True
        self.pulse.config(text="")
        self.root.after(0, self._ui_loop)

    def trigger_refrain(self):
        phrase = REFRAIN_STEPS[self.engine.refrain_index % len(REFRAIN_STEPS)]
        hint = self.engine._hint()
        self.engine.question = f"{phrase} If {hint}, where does it lead?"
        self.log.append(f"[Refrain] {phrase} (manual) bias: {hint}")
        self.engine.refrain_index += 1
        self._flash(self.pulse, ["#ffffff", "#ffcc66", "#fff2b3"], 900)

    def save_log(self):
        path = self.log.save()
        self.pulse.config(text=f"Saved log: {path}")

    def clear_lessons(self):
        self.learning.lessons.clear()
        self.lessons_text.delete("1.0", "end")
        self.pulse.config(text="Lessons cleared")

# ============================================================
# Auto loader
# ============================================================

def auto_loader():
    root = tk.Tk()
    RoamingSpotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    auto_loader()

