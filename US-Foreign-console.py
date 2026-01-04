#!/usr/bin/env python3
"""
US Foreign Actions Console
- One unified GUI
- Left: continuous information stream
- Right: breaking news alerts
- Auto-start watcher
- Auto-translate breaking alerts to English
- Sound alerts on breaking news
- Cross-platform friendly (Windows/macOS/Linux)

Dependencies:
    pip install requests googletrans==4.0.0rc1
"""

import os
import json
import time
import threading
import queue
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import requests

# GUI / sound
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
try:
    import winsound  # Windows-only, for sound alerts
except ImportError:
    winsound = None

from googletrans import Translator

# ---------------- CONFIGURATION ----------------

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "us_foreign_actions.json")

# Poll interval for the watcher in seconds
POLL_INTERVAL_SECONDS = 60 * 15  # 15 minutes

# How "recent" must an item be to count as breaking (for hybrid logic)
BREAKING_RECENT_MINUTES = 60

# Keywords that mark an item as potentially breaking
BREAKING_KEYWORDS = [
    "attack", "attacked", "invasion", "invaded", "sanctions", "sanction",
    "military", "airstrike", "air strike", "covert", "coup", "strike",
    "missile", "bombing", "crisis", "uprising", "troops", "deployment",
    "regime change", "assassination", "blockade", "naval", "nuclear",
]

# Search queries (for APIs if added later)
SEARCH_QUERIES = [
    "United States foreign policy",
    "US military operations abroad",
    "US intervention in foreign countries",
    "US sanctions imposed on other countries",
    "US involvement in international conflicts",
    "US covert operations foreign countries",
]

# RSS feeds for world / US-related news
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "https://feeds.reuters.com/Reuters/worldNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.cnn.com/rss/edition_world.rss",
]

# Optional: placeholder for a news API key if you want to integrate one later.
NEWS_API_KEY = ""  # currently unused in this unified script

# Basic country name list for guessing (can be extended)
COUNTRY_KEYWORDS = {
    "afghanistan": "Afghanistan",
    "argentina": "Argentina",
    "australia": "Australia",
    "brazil": "Brazil",
    "canada": "Canada",
    "china": "China",
    "cuba": "Cuba",
    "france": "France",
    "germany": "Germany",
    "iran": "Iran",
    "iraq": "Iraq",
    "israel": "Israel",
    "italy": "Italy",
    "japan": "Japan",
    "mexico": "Mexico",
    "north korea": "North Korea",
    "south korea": "South Korea",
    "russia": "Russia",
    "saudi arabia": "Saudi Arabia",
    "syria": "Syria",
    "turkey": "Turkey",
    "ukraine": "Ukraine",
    "united kingdom": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "venezuela": "Venezuela",
    "yemen": "Yemen",
    "palestine": "Palestine",
    "gaza": "Palestine",
    "west bank": "Palestine",
    "haiti": "Haiti",
    "libya": "Libya",
    "pakistan": "Pakistan",
    "colombia": "Colombia",
    "nicaragua": "Nicaragua",
    "panama": "Panama",
    "chile": "Chile",
    "guatemala": "Guatemala",
    "honduras": "Honduras",
    "el salvador": "El Salvador",
    "ecuador": "Ecuador",
    "peru": "Peru",
    "bolivia": "Bolivia",
    "dominican": "Dominican Republic",
}

translator = Translator(service_urls=["translate.googleapis.com"])

# ---------------- UTILS / STORAGE ----------------

def ensure_data_file() -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"items": []}, f, indent=2, ensure_ascii=False)


def load_existing_items() -> List[Dict[str, Any]]:
    ensure_data_file()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def save_items(items: List[Dict[str, Any]]) -> None:
    ensure_data_file()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2, ensure_ascii=False)


def make_item_id(source: str, url: str, title: str) -> str:
    h = hashlib.sha256()
    key = f"{source}||{url}||{title}".encode("utf-8", errors="ignore")
    h.update(key)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return None


def guess_country(text: str) -> str:
    lower = text.lower()
    for key, country in COUNTRY_KEYWORDS.items():
        if key in lower:
            return country
    return "Unknown"


def clean_html(s: str) -> str:
    import html
    s = html.unescape(s or "")
    for tag in ["<![CDATA[", "]]>"]:
        s = s.replace(tag, "")
    return s.strip()


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()

# ---------------- FETCHING (RSS) ----------------

def fetch_rss_feed(url: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch RSS {url}: {e}")
        return results

    text = resp.text
    chunks = text.split("<item")
    for chunk in chunks[1:]:
        title = extract_between(chunk, "<title>", "</title>") or ""
        summary = extract_between(chunk, "<description>", "</description>") or ""
        link = extract_between(chunk, "<link>", "</link>") or ""

        merged_text = f"{title} {summary}".lower()
        # Filter to keep at least some US-related or foreign content heuristic
        if any(k in merged_text for k in ["united states", "u.s.", "us ", "u.s. ", "american", "washington"]):
            results.append({
                "source": f"rss:{url}",
                "query": None,
                "title": clean_html(title),
                "summary": clean_html(summary),
                "url": clean_html(link),
                "published_at": None,  # Many RSS feeds have pubDate; can be added if needed
                "fetched_at": now_iso(),
            })

    return results


def fetch_all_sources() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for feed in RSS_FEEDS:
        items.extend(fetch_rss_feed(feed))
    return items

# ---------------- BREAKING NEWS LOGIC ----------------

def is_breaking(item: Dict[str, Any]) -> bool:
    title = (item.get("title") or "").lower()
    summary = (item.get("summary") or "").lower()
    merged = f"{title} {summary}"

    # Keyword rule
    if any(kw in merged for kw in BREAKING_KEYWORDS):
        return True

    # Recency rule
    published = parse_iso(item.get("published_at")) or parse_iso(item.get("fetched_at"))
    if not published:
        return False
    age = datetime.now(timezone.utc) - published
    if age <= timedelta(minutes=BREAKING_RECENT_MINUTES):
        return True

    return False

# ---------------- WATCHER THREAD ----------------

class WatcherThread(threading.Thread):
    def __init__(self, new_items_queue: "queue.Queue[Dict[str, Any]]", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.new_items_queue = new_items_queue
        self.stop_event = stop_event

    def run(self):
        print("[INFO] Watcher thread started.")
        while not self.stop_event.is_set():
            try:
                self.one_cycle()
            except Exception as e:
                print(f"[ERROR] Watcher cycle failed: {e}")
            # Sleep between cycles
            for _ in range(POLL_INTERVAL_SECONDS):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        print("[INFO] Watcher thread stopped.")

    def one_cycle(self):
        print(f"[INFO] Watcher cycle at {now_iso()}")
        existing = load_existing_items()
        existing_ids = {it.get("id") for it in existing}

        fetched = fetch_all_sources()
        new_unique_items: List[Dict[str, Any]] = []

        for it in fetched:
            url = it.get("url") or ""
            title = it.get("title") or ""
            source = it.get("source") or "unknown"
            item_id = make_item_id(source, url, title)
            if item_id in existing_ids:
                continue
            it["id"] = item_id
            existing.append(it)
            existing_ids.add(item_id)
            new_unique_items.append(it)

        if new_unique_items:
            save_items(existing)
            for it in new_unique_items:
                self.new_items_queue.put(it)
            print(f"[INFO] New items added: {len(new_unique_items)}")
        else:
            print("[INFO] No new items this cycle.")

# ---------------- GUI APPLICATION ----------------

class USForeignConsoleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("US Foreign Actions Console")
        self.geometry("1400x800")
        self.minsize(1000, 600)

        self.new_items_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.watcher_thread: Optional[WatcherThread] = None

        self.items: List[Dict[str, Any]] = load_existing_items()

        self.create_widgets()
        self.populate_initial_items()

        # Auto-start watcher
        self.start_watcher()

        # Periodic check for new items from watcher
        self.after(500, self.process_new_items)

    # ---------- GUI BUILD ----------

    def create_widgets(self):
        # Top: control bar
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Watcher: stopped")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)

        start_btn = ttk.Button(control_frame, text="Start Watcher", command=self.start_watcher)
        start_btn.pack(side=tk.LEFT, padx=(10, 5))

        stop_btn = ttk.Button(control_frame, text="Stop Watcher", command=self.stop_watcher)
        stop_btn.pack(side=tk.LEFT, padx=5)

        reload_btn = ttk.Button(control_frame, text="Reload Data", command=self.reload_data_from_disk)
        reload_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Double-click row to open URL").pack(side=tk.RIGHT)

        # Middle: horizontal split
        main_frame = ttk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left: continuous stream (Treeview)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        ttk.Label(left_frame, text="Flowing Information (all items)").pack(side=tk.TOP, anchor="w")

        columns = ("time", "title", "country", "source")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("time", text="Fetched Time")
        self.tree.heading("title", text="Title")
        self.tree.heading("country", text="Country")
        self.tree.heading("source", text="Source")

        self.tree.column("time", width=150, anchor="w")
        self.tree.column("title", width=550, anchor="w")
        self.tree.column("country", width=150, anchor="w")
        self.tree.column("source", width=200, anchor="w")

        vsb_left = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb_left.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_left.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # Right: breaking alerts (Text with scroll)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        ttk.Label(right_frame, text="Breaking News Alerts (auto-translated to English)").pack(side=tk.TOP, anchor="w")

        self.breaking_text = tk.Text(right_frame, wrap="word")
        vsb_right = ttk.Scrollbar(right_frame, orient="vertical", command=self.breaking_text.yview)
        self.breaking_text.configure(yscrollcommand=vsb_right.set)

        self.breaking_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_right.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- WATCHER CONTROL ----------

    def start_watcher(self):
        if self.watcher_thread and self.watcher_thread.is_alive():
            self.status_var.set("Watcher: running")
            return
        self.stop_event.clear()
        self.watcher_thread = WatcherThread(self.new_items_queue, self.stop_event)
        self.watcher_thread.start()
        self.status_var.set("Watcher: running")

    def stop_watcher(self):
        self.stop_event.set()
        self.status_var.set("Watcher: stopping...")
        # Thread will stop after current sleep cycle; we don't block the GUI

    # ---------- DATA LOADING / DISPLAY ----------

    def populate_initial_items(self):
        # Clear tree
        for row in self.tree.get_children():
            self.tree.delete(row)
        # Insert existing items, newest first
        for it in sorted(self.items, key=lambda x: x.get("fetched_at", ""), reverse=True):
            self.insert_item_into_tree(it)

    def reload_data_from_disk(self):
        self.items = load_existing_items()
        self.populate_initial_items()
        messagebox.showinfo("Reload", "Data reloaded from disk.")

    def insert_item_into_tree(self, item: Dict[str, Any]):
        fetched_at = item.get("fetched_at")
        dt = parse_iso(fetched_at)
        if dt:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = fetched_at or ""

        title = item.get("title") or ""
        summary = item.get("summary") or ""
        merged_text = f"{title} {summary}"
        country = guess_country(merged_text)
        source = item.get("source") or "unknown"

        iid = item.get("id", "")

        self.tree.insert("", "end", iid=iid, values=(time_str, title, country, source))

    def append_breaking_alert(self, item: Dict[str, Any]):
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        url = item.get("url") or ""
        fetched_at = item.get("fetched_at") or ""
        dt = parse_iso(fetched_at)
        if dt:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = fetched_at

        merged_text = f"{title} {summary}"
        country = guess_country(merged_text)

        # Auto-translate summary to English
        translated_text = ""
        text_to_translate = summary.strip() or title
        if text_to_translate:
            try:
                result = translator.translate(text_to_translate, dest="en")
                translated_text = result.text
            except Exception as e:
                translated_text = f"(Translation failed: {e})"

        self.breaking_text.insert(
            tk.END,
            f"\n=== BREAKING ({time_str}) ===\n"
            f"Country: {country}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Original summary:\n{summary}\n\n"
            f"English translation:\n{translated_text}\n"
            "------------------------------\n"
        )
        self.breaking_text.see(tk.END)

        # Sound alert
        self.play_alert_sound()

    def play_alert_sound(self):
        # Windows: use winsound
        if winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return
            except Exception:
                pass
        # Others: use Tk's bell
        try:
            self.bell()
        except Exception:
            pass

    # ---------- NEW ITEMS FROM WATCHER ----------

    def process_new_items(self):
        updated = False
        while True:
            try:
                item = self.new_items_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.items.append(item)
                self.insert_item_into_tree(item)
                if is_breaking(item):
                    self.append_breaking_alert(item)
                updated = True

        if updated:
            self.status_var.set(f"Watcher: running (last update {datetime.now().strftime('%H:%M:%S')})")

        # Schedule next check
        self.after(1000, self.process_new_items)

    # ---------- EVENTS ----------

    def on_tree_double_click(self, event):
        item_id = self.tree.focus()
        if not item_id:
            return
        # Find the item
        for it in self.items:
            if it.get("id") == item_id:
                url = it.get("url") or ""
                if url:
                    webbrowser.open(url)
                else:
                    messagebox.showinfo("No URL", "No URL available for this item.")
                return

# ---------------- MAIN ENTRY ----------------

def main():
    app = USForeignConsoleApp()
    app.mainloop()

if __name__ == "__main__":
    main()

