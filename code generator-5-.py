# ============================================================
# Code Generator Dashboard — Unified Feeds + Drag-and-Drop (Any Format)
# Streaming for Huge Files + Progress Bar + Internet Feed (with defaults)
# Stop Button + Threaded Generation + Manual Drive Selection (Local + Network)
# Guaranteed Preview Updates and Robust Logging
# ============================================================

import os
import sys
import time
import hashlib
import subprocess
import importlib
import random
import threading
import socket
import json
import re
import tkinter as tk
from tkinter import ttk, filedialog

# --- Autoloader ---
def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as e:
                print(f"[Autoloader] Failed to install {pkg}: {e}")

autoload(["psutil", "requests", "tkinterdnd2"])

# Optional imports
try:
    import psutil
except Exception:
    psutil = None

try:
    import requests
except Exception:
    requests = None

try:
    from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD
except Exception:
    DND_FILES = None
    DND_TEXT = None
    TkinterDnD = None

# --- Drive utilities ---
def list_all_drives():
    drives = []
    if os.name == "nt":
        for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{d}:\\"
            if os.path.exists(root):
                drives.append(root)
    else:
        if psutil:
            for part in psutil.disk_partitions(all=True):
                drives.append(part.mountpoint)
        else:
            drives.append(os.getcwd())
    return drives

def choose_storage_dir_auto():
    base = "generated"
    if psutil is None:
        return os.path.join(os.getcwd(), base)
    if os.name == "nt":
        default = "C:\\"
        try:
            usage = psutil.disk_usage(default)
        except Exception:
            usage = None
        if usage and usage.percent <= 50.0:
            return os.path.join(default, base)
        drives = list_all_drives()
        best = default
        best_free = -1
        for d in drives:
            try:
                u = psutil.disk_usage(d)
                if u.free > best_free:
                    best_free = u.free
                    best = d
            except Exception:
                continue
        return os.path.join(best, base)
    else:
        drives = list_all_drives()
        best = os.getcwd()
        best_free = -1
        for d in drives:
            try:
                u = psutil.disk_usage(d)
                if u.free > best_free:
                    best_free = u.free
                    best = d
            except Exception:
                continue
        return os.path.join(best, base)

# --- CPU throttle ---
class Throttle:
    def __init__(self, target_percent: int):
        self.target = max(0, min(100, int(target_percent)))
        self.sleep_min = 0.0
        self.sleep_max = 0.15
        self.sleep_s = self.sleep_max * (1.0 - (self.target / 100.0))

    def adjust(self):
        if psutil is None:
            return
        try:
            cpu = psutil.cpu_percent(interval=0.12)
            if cpu > self.target:
                self.sleep_s = min(self.sleep_max, self.sleep_s * 1.25 + 0.02)
            else:
                self.sleep_s = max(self.sleep_min, self.sleep_s * 0.85 - 0.01)
        except Exception:
            pass

    def wait(self):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

# --- System feed ---
def collect_system_feed(include_cpu=True, include_mem=True, include_disk=True, include_proc=True, max_procs=12):
    data = {}
    if psutil is None:
        return {"error": "psutil_unavailable"}
    try:
        if include_cpu:
            data["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        if include_mem:
            vm = psutil.virtual_memory()
            data["memory"] = {"total": vm.total, "available": vm.available, "percent": vm.percent}
        if include_disk:
            disks = {}
            for part in psutil.disk_partitions(all=False):
                mp = part.mountpoint
                try:
                    u = psutil.disk_usage(mp)
                    disks[mp] = {"total": u.total, "free": u.free, "percent": u.percent}
                except Exception:
                    continue
            data["disks"] = disks
        if include_proc:
            procs = []
            for p in psutil.process_iter(attrs=["pid", "name"]):
                if len(procs) >= max_procs:
                    break
                try:
                    info = p.info
                    procs.append({"pid": info["pid"], "name": info.get("name", "")})
                except Exception:
                    continue
            data["processes_sample"] = procs
    except Exception as e:
        data["error"] = f"system_feed_error: {e}"
    return data

# --- Internet helpers ---
DEFAULT_TARGETS = [
    "http://example.com",
    "https://www.wikipedia.org",
    "localhost:80",
    "localhost:8080"
]

def is_url(text):
    return bool(re.match(r"^(https?://|ftp://)", text.strip(), re.IGNORECASE))

def is_host_port(text):
    m = re.match(r"^([A-Za-z0-9\.\-]+):(\d{1,5})$", text.strip())
    return m is not None

def fetch_url(url, timeout=4, preview_limit=3000):
    if not requests:
        return {"url": url, "error": "requests_unavailable"}
    try:
        r = requests.get(url, timeout=timeout)
        ct = (r.headers.get("content-type") or "").lower()
        text = r.text[:preview_limit]
        return {"url": url, "status_code": r.status_code, "content_type": ct, "content_preview": text}
    except Exception as e:
        return {"url": url, "error": f"http_error: {e}"}

def probe_host_port(target, timeout=3, preview_limit=512):
    m = re.match(r"^([A-Za-z0-9\.\-]+):(\d{1,5})$", target.strip())
    if not m:
        return {"target": target, "error": "invalid_host_port"}
    host, port = m.group(1), int(m.group(2))
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            try:
                data = s.recv(preview_limit)
                preview = data.decode("utf-8", errors="replace")
            except socket.timeout:
                preview = ""
            return {"target": target, "reachable": True, "preview": preview}
    except Exception as e:
        return {"target": target, "reachable": False, "error": f"tcp_error: {e}"}

def fetch_any_targets(targets, max_items=20):
    out = {"http": [], "tcp": []}
    count = 0
    for raw in targets:
        t = raw.strip()
        if not t:
            continue
        if count >= max_items:
            break
        if is_url(t):
            out["http"].append(fetch_url(t))
            count += 1
        elif is_host_port(t):
            out["tcp"].append(probe_host_port(t))
            count += 1
        else:
            if re.match(r"^[A-Za-z0-9\.\-]+$", t):
                out["http"].append(fetch_url("http://" + t))
                count += 1
            else:
                out.setdefault("other", []).append({"input": t, "note": "unrecognized target"})
    return out

# --- Drag-and-drop parsing and streaming ---
HUGE_FILE_THRESHOLD = 100 * 1024 * 1024
MANUAL_EMBED_MAX = 2 * 1024 * 1024
TARGET_SCAN_MAX = 512 * 1024

def parse_dnd_file_list(data: str):
    items = []
    token = ""
    in_quote = False
    for ch in data:
        if ch == "{":
            in_quote = True
            token = ""
        elif ch == "}":
            in_quote = False
            items.append(token)
            token = ""
        elif ch in [" ", "\n", "\r", "\t"] and not in_quote:
            if token:
                items.append(token)
                token = ""
        else:
            token += ch
    if token:
        items.append(token)
    return [i.strip() for i in items if i.strip()]

def read_file_preview(path, max_bytes=MANUAL_EMBED_MAX):
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
        return {"path": path, "size": size, "preview": text}
    except Exception as e:
        return {"path": path, "size": size, "error": f"read_error: {e}"}

def stream_file_chunks(path, chunk_size=1 * 1024 * 1024, max_chunks=4):
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    yielded = 0
    try:
        with open(path, "rb") as f:
            offset = 0
            while yielded < max_chunks:
                data = f.read(chunk_size)
                if not data:
                    break
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("latin-1", errors="replace")
                yield {"path": path, "offset": offset, "size": size, "chunk": text[:2000]}
                offset += len(data)
                yielded += 1
    except Exception as e:
        yield {"path": path, "offset": 0, "size": size, "error": f"stream_error: {e}"}

def extract_targets_from_text(text):
    urls = re.findall(r"(https?://[^\s]+)", text, flags=re.IGNORECASE)
    hp = re.findall(r"([A-Za-z0-9\.\-]+:\d{1,5})", text)
    return urls + hp

# --- CodeBuilder ---
class CodeBuilder:
    def __init__(self, name: str, idea: dict, throttle: Throttle = None,
                 unified_feed: dict = None, manual_feed: str = "", streamed_files: list = None, stop_flag_getter=None):
        self.name = name
        self.idea = idea
        self.lines = []
        self.throttle = throttle or Throttle(50)
        self.unified_feed = unified_feed or {}
        self.manual_feed = manual_feed or ""
        self.streamed_files = streamed_files or []
        self.stop_flag_getter = stop_flag_getter or (lambda: False)

    def _emit(self, text: str):
        if self.stop_flag_getter():
            return
        self.lines.append(text)
        if self.throttle:
            self.throttle.adjust()
            self.throttle.wait()

    def add_header(self):
        header = [
            f"# Project: {self.idea.get('title', 'untitled')}",
            f"# Name: {self.name}",
            f"# Params: {self.idea.get('params', {})}",
            "",
            "import sys, os, json, math, time, random, logging",
            "logging.basicConfig(level=logging.INFO)",
            ""
        ]
        for ln in header:
            self._emit(ln)

    def add_feed_section(self):
        self._emit("# --- Unified Feed Section ---")
        if self.manual_feed.strip():
            mf = self.manual_feed.replace("\n", "\\n")
            self._emit(f"# Manual feed (truncated): {mf[:2000]}")
        uf_json = json.dumps(self.unified_feed, ensure_ascii=False)
        self._emit(f"# Unified feed snapshot (truncated): {uf_json[:12000]}")
        self._emit("")
        if self.streamed_files:
            self._emit("# --- Streamed file samples (path + offset + chunk) ---")
            for entry in self.streamed_files:
                if "error" in entry:
                    self._emit(f"# Stream error for {entry.get('path')}: {entry['error']}")
                    continue
                path = entry.get("path")
                size = entry.get("size")
                offset = entry.get("offset")
                chunk = (entry.get("chunk") or "").replace("\n", "\\n")
                self._emit(f"# SAMPLE path={path} size={size} offset={offset} chunk={chunk[:2000]}")
            self._emit("")

    def add_core(self):
        core = [
            "class Config:",
            "    def __init__(self):",
            f"        self.name = '{self.name}'",
            f"        self.title = '{self.idea.get('title', 'untitled')}'",
            "        self.iterations = 100",
            "        self.output_dir = os.path.join(os.getcwd(), 'out')",
            "",
            "    def ensure(self):",
            "        os.makedirs(self.output_dir, exist_ok=True)",
            "",
            "class ProjectCore:",
            "    def __init__(self, cfg: Config):",
            "        self.cfg = cfg",
            "    def run(self):",
            "        logging.info(f'Running project {self.cfg.title}')",
            "        return {'status': 'ok', 'iterations': self.cfg.iterations}",
            "",
            "def main():",
            "    cfg = Config(); cfg.ensure()",
            "    core = ProjectCore(cfg)",
            "    result = core.run()",
            "    path = os.path.join(cfg.output_dir, 'result.json')",
            "    with open(path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=2)",
            "    logging.info(f'Result written to {path}')",
            "",
            "if __name__ == '__main__':",
            "    main()",
            ""
        ]
        for ln in core:
            self._emit(ln)

    def add_fillers(self, target_lines=200):
        i = 0
        while len(self.lines) < target_lines:
            if self.stop_flag_getter():
                break
            self._emit(f"def stub_func_{i}(x): return x * {i}")
            i += 1

    def build(self, min_lines=200) -> str:
        self.add_header()
        self.add_feed_section()
        self.add_core()
        self.add_fillers(min_lines)
        return "\n".join(self.lines)

# --- GUI ---
class GeneratorGUI((TkinterDnD.Tk if TkinterDnD else tk.Tk)):
    def __init__(self):
        super().__init__()
        self.title("Code Generator — Unified Feeds + DnD + Internet + Stop + Drives")
        self.geometry("1320x940")

        # Top bar
        topbar = ttk.Frame(self); topbar.pack(fill="x", padx=10, pady=10)
        self.lbl_status = ttk.Label(topbar, text="Ready", foreground="green")
        self.lbl_status.pack(side="left", padx=5)

        # Controls
        controls = ttk.Frame(self); controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Lines of code:").pack(side="left")
        self.slider_lines = tk.Scale(controls, from_=100, to=20000, orient="horizontal", length=350)
        self.slider_lines.set(800)
        self.slider_lines.pack(side="left", padx=10)
        self.entry_lines = ttk.Entry(controls, width=10)
        self.entry_lines.insert(0, "800")
        self.entry_lines.pack(side="left", padx=10)

        ttk.Label(controls, text="CPU target (%):").pack(side="left", padx=(20, 0))
        self.slider_cpu = tk.Scale(controls, from_=0, to=100, orient="horizontal", length=220)
        self.slider_cpu.set(50)
        self.slider_cpu.pack(side="left", padx=10)

        # Buttons row
        btns = ttk.Frame(self); btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(btns, text="Generate Code", command=self.generate_code_threaded).pack(side="left", padx=10)
        ttk.Button(btns, text="Auto Generate", command=self.auto_generate).pack(side="left", padx=10)
        ttk.Button(btns, text="Stop", command=self.stop_auto).pack(side="left", padx=10)
        ttk.Button(btns, text="Internet Feed", command=self.fetch_targets_async).pack(side="left", padx=10)
        ttk.Button(btns, text="Select Drive", command=self.select_drive).pack(side="left", padx=10)

        # Unified Feeds panel
        feeds = ttk.LabelFrame(self, text="Unified feeds (drag files/text/URLs here)")
        feeds.pack(fill="both", expand=False, padx=10, pady=10)

        # Manual feed
        ttk.Label(feeds, text="Manual feed (paste or drop text/files/URLs):").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.txt_manual_feed = tk.Text(feeds, height=8, wrap="word")
        self.txt_manual_feed.grid(row=1, column=0, columnspan=6, sticky="nsew", padx=8, pady=8)

        # Drag-and-drop registration
        if TkinterDnD and DND_FILES:
            self.txt_manual_feed.drop_target_register(DND_FILES)
            self.txt_manual_feed.dnd_bind('<<Drop>>', self.on_drop_manual_files)
        if TkinterDnD and DND_TEXT:
            self.txt_manual_feed.drop_target_register(DND_TEXT)
            self.txt_manual_feed.dnd_bind('<<Drop>>', self.on_drop_manual_text)

        # System feed checkboxes
        ttk.Label(feeds, text="System telemetry:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.chk_cpu = tk.BooleanVar(value=True)
        self.chk_mem = tk.BooleanVar(value=True)
        self.chk_disk = tk.BooleanVar(value=True)
        self.chk_proc = tk.BooleanVar(value=False)
        ttk.Checkbutton(feeds, text="CPU", variable=self.chk_cpu).grid(row=3, column=0, sticky="w", padx=8)
        ttk.Checkbutton(feeds, text="Memory", variable=self.chk_mem).grid(row=3, column=1, sticky="w", padx=8)
        ttk.Checkbutton(feeds, text="Disk", variable=self.chk_disk).grid(row=3, column=2, sticky="w", padx=8)
        ttk.Checkbutton(feeds, text="Processes (sample)", variable=self.chk_proc).grid(row=3, column=3, sticky="w", padx=8)

        # Internet targets
        ttk.Label(feeds, text="Internet targets (one per line — URL or host:port; drop files/text here too):").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        self.txt_targets = tk.Text(feeds, height=6, wrap="none")
        self.txt_targets.grid(row=5, column=0, columnspan=4, sticky="nsew", padx=8, pady=8)

        # Drag-and-drop for targets
        if TkinterDnD and DND_FILES:
            self.txt_targets.drop_target_register(DND_FILES)
            self.txt_targets.dnd_bind('<<Drop>>', self.on_drop_targets_files)
        if TkinterDnD and DND_TEXT:
            self.txt_targets.drop_target_register(DND_TEXT)
            self.txt_targets.dnd_bind('<<Drop>>', self.on_drop_targets_text)

        self.btn_fetch_targets = ttk.Button(feeds, text="Fetch/Probe targets", command=self.fetch_targets_async)
        self.btn_fetch_targets.grid(row=5, column=4, sticky="n", padx=8)

        # Feed preview
        self.txt_feed_preview = tk.Text(feeds, height=10, wrap="word")
        self.txt_feed_preview.grid(row=6, column=0, columnspan=6, sticky="nsew", padx=8, pady=8)

        # Feed progression bar
        self.feed_progress = ttk.Progressbar(feeds, orient="horizontal", mode="indeterminate", length=400)
        self.feed_progress.grid(row=7, column=0, columnspan=6, padx=8, pady=6, sticky="we")

        # Grid weights
        for col in range(6):
            feeds.columnconfigure(col, weight=1)
        for row in [1, 5, 6]:
            feeds.rowconfigure(row, weight=1)

        # Log box
        self.txt_log = tk.Text(self, height=8)
        self.txt_log.pack(fill="x", padx=10, pady=10)

        # Preview pane (output appears here)
        preview_frame = ttk.LabelFrame(self, text="Generated code preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.preview_scroll_y = ttk.Scrollbar(preview_frame, orient="vertical")
        self.preview_scroll_x = ttk.Scrollbar(preview_frame, orient="horizontal")
        self.txt_preview = tk.Text(preview_frame, wrap="none",
                                   yscrollcommand=self.preview_scroll_y.set,
                                   xscrollcommand=self.preview_scroll_x.set)
        self.preview_scroll_y.config(command=self.txt_preview.yview)
        self.preview_scroll_x.config(command=self.txt_preview.xview)
        self.txt_preview.pack(fill="both", expand=True)
        self.preview_scroll_y.pack(fill="y", side="right")
        self.preview_scroll_x.pack(fill="x", side="bottom")

        # State
        self.last_code = ""
        self.last_file = ""
        self.unified_feed_cache = {}
        self.large_file_paths = []
        self.selected_drive = None
        self.stop_requested = False

        # Live CPU display
        self._cpu_display_running = True
        threading.Thread(target=self._cpu_display_loop, daemon=True).start()

    # --- Drag-and-drop handlers ---
    def on_drop_manual_files(self, event):
        paths = parse_dnd_file_list(event.data)
        added_small = 0
        added_large = 0
        for p in paths:
            if os.path.isdir(p):
                self._append_file_meta_to_feed_preview(p, None, mode="directory")
                self.txt_manual_feed.insert("end", f"\n[Directory dropped: {p}]\n")
                continue
            try:
                size = os.path.getsize(p)
            except Exception:
                size = None
            ext = (os.path.splitext(p)[1] or "").lower()
            text_like_exts = {".txt", ".py", ".md", ".json", ".xml", ".csv", ".ini", ".log", ".yaml", ".yml"}
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".tiff", ".webp"}
            if size is not None and size >= HUGE_FILE_THRESHOLD:
                self.large_file_paths.append(p)
                self._append_file_meta_to_feed_preview(p, size, mode="path_only")
                self.txt_manual_feed.insert("end", f"\n[Large file registered: {p} size={size}]\n")
                self.log(f"Registered large file (path-only): {p} ({size} bytes)")
                added_large += 1
            else:
                if ext in text_like_exts or size is None:
                    preview = read_file_preview(p, max_bytes=MANUAL_EMBED_MAX)
                    if "error" in preview:
                        self.txt_manual_feed.insert("end", f"\n# --- Error reading {p}: {preview['error']} ---\n")
                    else:
                        self.txt_manual_feed.insert("end", f"\n# --- Dropped file: {p} (size={preview['size']}) ---\n{preview['preview']}\n")
                    self._append_file_meta_to_feed_preview(p, size, mode="embedded_preview")
                elif ext in image_exts:
                    self.txt_manual_feed.insert("end", f"\n[Image file dropped: {p} size={size}]\n")
                    self.unified_feed_cache.setdefault("images", []).append({"path": p, "size": size})
                    self._append_file_meta_to_feed_preview(p, size, mode="image_meta")
                else:
                    self.txt_manual_feed.insert("end", f"\n[File dropped: {p} size={size}]\n")
                    self._append_file_meta_to_feed_preview(p, size, mode="embedded_meta")
                added_small += 1
        self.log(f"Dropped {len(paths)} item(s) into Manual feed. Small: {added_small}, Large: {added_large}")

    def on_drop_manual_text(self, event):
        text = event.data
        self.txt_manual_feed.insert("end", text + "\n")
        found = extract_targets_from_text(text)
        if found:
            self.txt_targets.insert("end", "\n".join(found) + "\n")
            self.log(f"Detected {len(found)} target(s) from dropped text and added to Internet targets.")

    def on_drop_targets_files(self, event):
        paths = parse_dnd_file_list(event.data)
        added = 0
        for p in paths:
            if os.path.isdir(p):
                self._append_file_meta_to_feed_preview(p, None, mode="directory")
                continue
            try:
                size = os.path.getsize(p)
            except Exception:
                size = None
            if size is not None and size >= HUGE_FILE_THRESHOLD:
                for entry in stream_file_chunks(p, chunk_size=TARGET_SCAN_MAX, max_chunks=1):
                    if "chunk" in entry:
                        targets = extract_targets_from_text(entry["chunk"])
                        if targets:
                            self.txt_targets.insert("end", "\n".join(targets) + "\n")
                            added += len(targets)
                self.large_file_paths.append(p)
                self._append_file_meta_to_feed_preview(p, size, mode="path_only")
                self.log(f"Scanned large file for targets: {p} (size={size})")
            else:
                preview = read_file_preview(p, max_bytes=TARGET_SCAN_MAX)
                if "preview" in preview:
                    targets = extract_targets_from_text(preview["preview"])
                    if targets:
                        self.txt_targets.insert("end", "\n".join(targets) + "\n")
                        added += len(targets)
                self._append_file_meta_to_feed_preview(p, size, mode="embedded_preview")
        self.log(f"Dropped {len(paths)} item(s) into Internet targets; added {added} entries.")

    def on_drop_targets_text(self, event):
        text = event.data
        targets = extract_targets_from_text(text)
        if targets:
            self.txt_targets.insert("end", "\n".join(targets) + "\n")
            self.log(f"Added {len(targets)} target(s) from dropped text.")
        else:
            self.txt_targets.insert("end", text.strip() + "\n")
            self.log("Added raw dropped text to targets.")

    def _append_file_meta_to_feed_preview(self, path, size, mode="embedded_meta"):
        meta_entry = {"path": path, "size": size, "mode": mode}
        files = self.unified_feed_cache.get("files", [])
        files.append(meta_entry)
        self.unified_feed_cache["files"] = files
        self._update_feed_preview()

    # --- Live CPU display ---
    def _cpu_display_loop(self):
        while self._cpu_display_running:
            try:
                if psutil:
                    cpu = psutil.cpu_percent(interval=0.5)
                    self.lbl_status.config(text=f"Ready | Live CPU: {cpu:.0f}%", foreground="green")
                else:
                    self.lbl_status.config(text="Ready", foreground="green")
            except Exception:
                pass
            time.sleep(0.5)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")

    # --- Controls helpers ---
    def get_line_count(self):
        try:
            manual_val = int(self.entry_lines.get())
            return manual_val
        except ValueError:
            return int(self.slider_lines.get())

    def _build_idea(self):
        return {
            "title": "gravity-field_weaving-ring_habitat-stability",
            "params": {"scale": 10.0, "power_budget": 1000.0, "tolerance": 0.1, "mutation_rate": 0.2}
        }

    def _collect_manual_feed(self):
        return self.txt_manual_feed.get("1.0", "end").strip()

    def _collect_system_feed(self):
        sys_data = collect_system_feed(
            include_cpu=self.chk_cpu.get(),
            include_mem=self.chk_mem.get(),
            include_disk=self.chk_disk.get(),
            include_proc=self.chk_proc.get()
        )
        self.unified_feed_cache["system"] = sys_data
        return sys_data

    # --- Manual drive selection ---
    def select_drive(self):
        path = filedialog.askdirectory(title="Select drive or folder for saving projects")
        if path:
            self.selected_drive = path
            self.log(f"Drive manually set to: {path}. All projects will save here.")

    # --- Internet fetching with defaults + progress ---
    def fetch_targets_async(self):
        targets_text = self.txt_targets.get("1.0", "end")
        targets = [line.strip() for line in targets_text.splitlines() if line.strip()]
        if not targets:
            targets = DEFAULT_TARGETS[:]
            self.txt_targets.insert("end", "\n".join(targets) + "\n")
            self.log(f"No targets provided. Using defaults: {', '.join(targets)}")
        self.log(f"Fetching/probing {len(targets)} targets...")
        self.btn_fetch_targets.config(state="disabled")
        self.feed_progress.config(mode="indeterminate")
        self.feed_progress.start(10)
        t = threading.Thread(target=self._do_fetch_targets, args=(targets,), daemon=True)
        t.start()

    def _do_fetch_targets(self, targets):
        try:
            result = fetch_any_targets(targets, max_items=20)
            self.unified_feed_cache["http"] = result.get("http", [])
            self.unified_feed_cache["tcp"] = result.get("tcp", [])
            if "other" in result:
                self.unified_feed_cache["other"] = result["other"]
            self._update_feed_preview()
            self.log("Targets fetched/probed.")
        except Exception as e:
            self.log(f"Targets fetch error: {e}")
        finally:
            self.feed_progress.stop()
            self.feed_progress["value"] = 0
            self.btn_fetch_targets.config(state="normal")

    def _update_feed_preview(self):
        preview = json.dumps(self.unified_feed_cache, indent=2, ensure_ascii=False)
        self.txt_feed_preview.delete("1.0", "end")
        self.txt_feed_preview.insert("1.0", preview)

    # --- Threaded generation with stop + drive override + guaranteed preview ---
    def generate_code_threaded(self):
        lines = self.get_line_count()
        self.stop_requested = False
        threading.Thread(target=self._generate_worker, args=(lines,), daemon=True).start()

    def auto_generate(self):
        min_val = int(self.slider_lines.cget("from"))
        max_val = int(self.slider_lines.cget("to"))
        lines = random.randint(min_val, max_val)
        self.log(f"Auto-generating with {lines} lines...")
        self.stop_requested = False
        threading.Thread(target=self._generate_worker, args=(lines,), daemon=True).start()

    def stop_auto(self):
        self.stop_requested = True
        self.log("Stop requested.")

    def _generate_worker(self, lines):
        try:
            self.log("Starting code generation...")
            lines = max(50, min(50000, int(lines)))
            cpu_target = int(self.slider_cpu.get())
            throttle = Throttle(target_percent=cpu_target)

            self.lbl_status.config(text="Generating...", foreground="blue")
            idea = self._build_idea()
            manual_feed = self._collect_manual_feed()
            system_feed = self._collect_system_feed()

            unified_feed = {
                "system": system_feed,
                "http": self.unified_feed_cache.get("http", []),
                "tcp": self.unified_feed_cache.get("tcp", []),
                "files": self.unified_feed_cache.get("files", []),
                "images": self.unified_feed_cache.get("images", []),
            }
            if "other" in self.unified_feed_cache:
                unified_feed["other"] = self.unified_feed_cache["other"]

            streamed_samples = []
            total_paths = len(self.large_file_paths)
            if total_paths > 0:
                self.feed_progress.stop()
                self.feed_progress.config(mode="determinate", maximum=total_paths)
                self.feed_progress["value"] = 0

            for idx, p in enumerate(self.large_file_paths):
                if self.stop_requested:
                    break
                for entry in stream_file_chunks(p, chunk_size=1 * 1024 * 1024, max_chunks=4):
                    if self.stop_requested:
                        break
                    streamed_samples.append(entry)
                self.feed_progress["value"] = idx + 1
                self.update_idletasks()

            if total_paths > 0:
                self.feed_progress["value"] = 0
                self.feed_progress.config(mode="indeterminate")

            name = "Project-" + hashlib.sha1(str(time.time()).encode()).hexdigest()[:6]
            builder = CodeBuilder(
                name, idea, throttle=throttle,
                unified_feed=unified_feed, manual_feed=manual_feed,
                streamed_files=streamed_samples, stop_flag_getter=lambda: self.stop_requested
            )

            code = builder.build(min_lines=lines)
            self.log("Code built successfully.")

            # Always update preview safely on main thread
            self.after(0, lambda: (
                self.txt_preview.delete("1.0", "end"),
                self.txt_preview.insert("1.0", code)
            ))

            if self.stop_requested:
                self.log("Generation stopped before save.")
                self.lbl_status.config(text="Stopped", foreground="orange")
                return

            # Choose save directory
            if self.selected_drive:
                out_dir = os.path.join(self.selected_drive, "generated")
            else:
                out_dir = choose_storage_dir_auto()
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"{name}.py")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(code)

            self.last_code = code
            self.last_file = out_file

            self.log(f"Generated {out_file} with {len(code.splitlines())} lines (CPU target {cpu_target}%).")
            self.lbl_status.config(text="Done", foreground="green")

        except Exception as e:
            self.log(f"Generate error: {e}")
            self.lbl_status.config(text="Error", foreground="red")

    def destroy(self):
        self._cpu_display_running = False
        super().destroy()

# --- Entry point ---
if __name__ == "__main__":
    app = GeneratorGUI()
    app.mainloop()

