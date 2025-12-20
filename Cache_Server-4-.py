#!/usr/bin/env python
"""
Evolved predictive multi-tier caching server (RAM + Disk) with integrated GUI and pressure-aware behavior.

- Predictive cache server (FastAPI + uvicorn)
- Integrated Tkinter GUI control panel in the same process
- Auto-installs dependencies: fastapi, uvicorn, pydantic, httpx

New behavior:
- Pressure metrics:
    - RAM pressure (% of MAX_RAM_ITEMS)
    - Disk pressure (approx, via file count vs dynamic heuristic)
- Modes:
    - normal
    - burst (higher traffic, moderate pressure)
    - survival (high pressure)
- Auto-tune mode:
    - Adjusts prefetch fanout, TTL scaling, and heat decay based on pressure and hit ratio
- GUI:
    - Shows pressure, mode, adaptive RAM suggestions
    - Allows toggling auto-tune
    - Manual override of core parameters still possible

Run:
    python cache_server_gui.py
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List, Deque
from collections import defaultdict, deque
import threading
import queue

# -----------------------------
# Auto-loader for dependencies
# -----------------------------

REQUIRED_LIBS = ["fastapi", "uvicorn", "pydantic", "httpx"]


def ensure_dependencies():
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            print(f"[AutoLoader] Missing library '{lib}', installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "pip", "install", lib])


ensure_dependencies()

# Safe imports after auto-loader
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import httpx
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------
# Config
# -----------------------------

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./disk_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    MAX_RAM_ITEMS = int(os.environ.get("MAX_RAM_ITEMS", "1000"))
except ValueError:
    MAX_RAM_ITEMS = 1000

_default_ttl_env = os.environ.get("DEFAULT_TTL", "").strip()
TTL_SECONDS: Optional[int] = int(_default_ttl_env) if _default_ttl_env else None

HOST = os.environ.get("HOST", "127.0.0.1")  # local for GUI
try:
    PORT = int(os.environ.get("PORT", "8000"))
except ValueError:
    PORT = 8000

PREFETCH_ENABLED = os.environ.get("PREFETCH_ENABLED", "1") != "0"
try:
    PREFETCH_FANOUT = int(os.environ.get("PREFETCH_FANOUT", "3"))
except ValueError:
    PREFETCH_FANOUT = 3

try:
    HEAT_DECAY_HALF_LIFE = float(os.environ.get("HEAT_DECAY_HALF_LIFE", "300"))
except ValueError:
    HEAT_DECAY_HALF_LIFE = 300.0

HEAT_DECAY_FACTOR_PER_SEC = 0.5 ** (1.0 / HEAT_DECAY_HALF_LIFE) if HEAT_DECAY_HALF_LIFE > 0 else 1.0

GOSSIP_ENABLED = os.environ.get("GOSSIP_ENABLED", "0") != "0"
GOSSIP_TARGET_URL = os.environ.get("GOSSIP_TARGET_URL", "").strip() or None

# Auto-tune (AI assist) toggle
AUTO_TUNE_ENABLED = True

# -----------------------------
# Metrics and profiling
# -----------------------------

class Metrics:
    def __init__(self):
        self.total_get = 0
        self.total_set = 0
        self.hit_ram = 0
        self.hit_disk = 0
        self.miss = 0
        self._latency_samples = deque(maxlen=200)

    def record_get(self, tier: Optional[str], latency: float):
        self.total_get += 1
        if tier == "ram":
            self.hit_ram += 1
        elif tier == "disk":
            self.hit_disk += 1
        else:
            self.miss += 1
        self._latency_samples.append(latency)

    def record_set(self):
        self.total_set += 1

    def hit_ratio(self) -> float:
        if self.total_get == 0:
            return 0.0
        return (self.hit_ram + self.hit_disk) / self.total_get

    def avg_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) * 1000.0 / len(self._latency_samples)


metrics = Metrics()

# -----------------------------
# RAM Tier (LRU + heat-aware)
# -----------------------------

class LRUCache:
    """
    LRU cache with TTL and heat-aware eviction.
    Stores:
        key -> (value, last_access_time, expires_at or None)
    """

    def __init__(self, max_items: int):
        self.max_items = max_items
        self._store: Dict[str, Tuple[Any, float, Optional[float]]] = {}

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and self._now() > expires_at

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, last_access, expires_at = entry
        if self._is_expired(expires_at):
            del self._store[key]
            return None
        self._store[key] = (value, self._now(), expires_at)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = self._now() + ttl if ttl is not None else None
        self._store[key] = (value, self._now(), expires_at)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self._store) <= self.max_items:
            return

        def sort_key(item):
            key, (value, last_access, expires_at) = item
            heat = _key_heat(key)
            return (last_access, -heat)

        items = sorted(self._store.items(), key=sort_key)
        to_remove = len(self._store) - self.max_items
        for i in range(to_remove):
            key, _ = items[i]
            self._store.pop(key, None)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

    def keys_snapshot(self) -> List[str]:
        return list(self._store.keys())


ram_cache = LRUCache(MAX_RAM_ITEMS)

# -----------------------------
# Disk Tier + write-coalescing
# -----------------------------

def _key_to_filename(key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _disk_read(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _disk_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix("..tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def disk_get(key: str) -> Optional[Any]:
    path = _key_to_filename(key)
    if not path.exists():
        return None
    try:
        data = _disk_read(path)
        expires_at = data.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
        return data.get("value")
    except Exception as e:
        print(f"[DiskTier] Error reading {path}: {e}")
        return None


_write_queue: "queue.Queue[Tuple[str, Any, Optional[int]]]" = queue.Queue()


def disk_set_buffered(key: str, value: Any, ttl: Optional[int]) -> None:
    _write_queue.put((key, value, ttl))


def _disk_worker():
    while True:
        try:
            batch: List[Tuple[str, Any, Optional[int]]] = []
            item = _write_queue.get()
            if item is None:
                break
            batch.append(item)
            try:
                while len(batch) < 128:
                    batch.append(_write_queue.get_nowait())
            except queue.Empty:
                pass

            now = time.time()
            for key, value, ttl in batch:
                expires_at = now + ttl if ttl is not None else None
                path = _key_to_filename(key)
                data = {"value": value, "expires_at": expires_at}
                try:
                    _disk_write(path, data)
                except Exception as e:
                    print(f"[DiskTier] Error writing {path}: {e}")
        except Exception as e:
            print(f"[DiskTier] Worker error: {e}")
            time.sleep(1.0)


def disk_delete(key: str) -> None:
    path = _key_to_filename(key)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[DiskTier] Error deleting {path}: {e}")


def disk_clear() -> None:
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
        except Exception as e:
            print(f"[DiskTier] Error clearing {p}: {e}")


def disk_size() -> int:
    return sum(1 for _ in CACHE_DIR.glob("*.json"))


def cleanup_expired_on_disk(max_per_cycle: int = 200) -> None:
    now = time.time()
    count = 0
    for p in CACHE_DIR.glob("*.json"):
        if count >= max_per_cycle:
            break
        try:
            data = _disk_read(p)
            expires_at = data.get("expires_at")
            if expires_at is not None and now > expires_at:
                p.unlink()
            count += 1
        except Exception:
            try:
                p.unlink()
            except Exception:
                pass
            count += 1


_disk_thread = threading.Thread(target=_disk_worker, daemon=True)
_disk_thread.start()

# -----------------------------
# Prediction engine
# -----------------------------

class KeyStats:
    """
    - hits
    - last_access
    - heat (decaying)
    - per-hour access counts
    """

    def __init__(self):
        self.hits = 0
        self.last_access = 0.0
        self.heat = 0.0
        self.hour_counts = [0] * 24

    def record_access(self):
        now = time.time()
        if self.last_access > 0:
            dt = now - self.last_access
            self.heat *= (HEAT_DECAY_FACTOR_PER_SEC ** dt)
        self.heat += 1.0
        self.hits += 1
        self.last_access = now
        hour = time.localtime(now).tm_hour
        self.hour_counts[hour] += 1

    def current_heat(self) -> float:
        if self.last_access <= 0:
            return self.heat
        dt = time.time() - self.last_access
        return self.heat * (HEAT_DECAY_FACTOR_PER_SEC ** dt)


_key_stats: Dict[str, KeyStats] = defaultdict(KeyStats)
_ngram_transitions: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_access_history: Deque[str] = deque(maxlen=3)
_prefetch_queue: "queue.Queue[str]" = queue.Queue()
_family_heat: Dict[str, float] = defaultdict(float)


def key_family(key: str) -> Optional[str]:
    parts = key.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}:*"
    return None


def _update_family_heat(key: str):
    fam = key_family(key)
    if not fam:
        return
    ks = _key_stats[key]
    _family_heat[fam] = max(_family_heat[fam], ks.current_heat())


def _key_heat(key: str) -> float:
    ks = _key_stats.get(key)
    return ks.current_heat() if ks else 0.0


def record_access_and_predict(key: str) -> None:
    ks = _key_stats[key]
    ks.record_access()
    _update_family_heat(key)

    history = tuple(_access_history)
    if history:
        for order in range(1, len(history) + 1):
            ngram = history[-order:]
            _ngram_transitions[ngram][key] += 1
    _access_history.append(key)

    if PREFETCH_ENABLED:
        likely_next = predict_next_keys(history, current_key=key, limit=PREFETCH_FANOUT)
        for nxt in likely_next:
            if nxt != key:
                try:
                    _prefetch_queue.put_nowait(nxt)
                except queue.Full:
                    break

        fam = key_family(key)
        if fam:
            fam_keys = [k for k in _key_stats.keys() if key_family(k) == fam]
            fam_keys = sorted(fam_keys, key=lambda k: _key_stats[k].current_heat(), reverse=True)
            for fk in fam_keys[:PREFETCH_FANOUT]:
                if fk != key:
                    try:
                        _prefetch_queue.put_nowait(fk)
                    except queue.Full:
                        break


def predict_next_keys(history: Tuple[str, ...], current_key: str, limit: int = 3) -> List[str]:
    candidates: Dict[str, int] = defaultdict(int)
    seq = list(history) + [current_key]
    for order in range(min(3, len(seq)), 0, -1):
        ngram = tuple(seq[-order:])
        trans = _ngram_transitions.get(ngram)
        if not trans:
            continue
        for nxt, count in trans.items():
            candidates[nxt] += count * order
    if not candidates:
        return []
    items = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:limit]]


def get_adaptive_ttl_for_key(key: str, mode: str) -> Optional[int]:
    ks = _key_stats.get(key)
    base = TTL_SECONDS if TTL_SECONDS is not None else 300
    if not ks:
        ttl = base
    else:
        heat = ks.current_heat()
        fam = key_family(key)
        fam_heat = _family_heat.get(fam, 0.0) if fam else 0.0
        total_heat = heat + 0.5 * fam_heat
        factor = 1.0 + min(total_heat, 4.0)
        ttl = int(base * factor)

    if mode == "burst":
        ttl = int(ttl * 1.25)
    elif mode == "survival":
        ttl = int(ttl * 0.5)
    return max(ttl, 1)


async def prefetch_worker():
    while True:
        try:
            key = await asyncio.get_event_loop().run_in_executor(None, _prefetch_queue.get)
            if key is None:
                continue
            if ram_cache.get(key) is not None:
                continue
            value = disk_get(key)
            if value is not None:
                mode, _, _ = compute_mode_and_pressure()
                ttl = get_adaptive_ttl_for_key(key, mode)
                ram_cache.set(key, value, ttl=ttl)
        except Exception as e:
            print(f"[Prefetch] Worker error: {e}")
            await asyncio.sleep(1.0)


async def maintenance_worker():
    while True:
        try:
            cleanup_expired_on_disk(max_per_cycle=300)
        except Exception as e:
            print(f"[Maintenance] Error: {e}")
        await asyncio.sleep(30.0)


async def gossip_worker():
    if not GOSSIP_ENABLED or not GOSSIP_TARGET_URL:
        return
    client = httpx.AsyncClient(timeout=2.0)
    while True:
        try:
            hot = sorted(
                ((k, ks.current_heat()) for k, ks in _key_stats.items()),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            payload = {"hot_keys": [k for k, _ in hot]}
            await client.post(GOSSIP_TARGET_URL, json=payload)
        except Exception as e:
            print(f"[Gossip] Error: {e}")
        await asyncio.sleep(15.0)

# -----------------------------
# Pressure & mode computation
# -----------------------------

def compute_mode_and_pressure():
    ram_used = ram_cache.size()
    ram_capacity = max(MAX_RAM_ITEMS, 1)
    ram_pressure = ram_used / ram_capacity

    disk_items = disk_size()
    disk_soft_cap = max(1000, ram_capacity * 10)
    disk_pressure = min(disk_items / disk_soft_cap, 1.5)

    hit_ratio = metrics.hit_ratio()

    if ram_pressure > 0.9 or disk_pressure > 1.0:
        mode = "survival"
    elif ram_pressure > 0.6 or disk_pressure > 0.7:
        if hit_ratio > 0.5:
            mode = "burst"
        else:
            mode = "normal"
    else:
        mode = "normal"

    return mode, ram_pressure, disk_pressure


def auto_tune_if_enabled():
    global PREFETCH_FANOUT, HEAT_DECAY_HALF_LIFE, HEAT_DECAY_FACTOR_PER_SEC
    if not AUTO_TUNE_ENABLED:
        return

    mode, ram_p, disk_p = compute_mode_and_pressure()
    hit = metrics.hit_ratio()

    if mode == "normal":
        target_fanout = 3 if hit >= 0.3 else 1
        target_half_life = 300.0
    elif mode == "burst":
        target_fanout = 5 if hit >= 0.5 else 2
        target_half_life = 200.0
    else:
        target_fanout = 1
        target_half_life = 120.0

    PREFETCH_FANOUT = max(0, min(10, int((PREFETCH_FANOUT * 3 + target_fanout) / 4)))
    HEAT_DECAY_HALF_LIFE = (HEAT_DECAY_HALF_LIFE * 3 + target_half_life) / 4.0
    HEAT_DECAY_FACTOR_PER_SEC = 0.5 ** (1.0 / HEAT_DECAY_HALF_LIFE)


def compute_adaptive_ram_hint():
    base = MAX_RAM_ITEMS
    hit = metrics.hit_ratio()
    mode, ram_p, disk_p = compute_mode_and_pressure()

    if mode == "survival" and ram_p > 0.95 and hit > 0.6:
        return int(base * 1.5)
    elif mode == "burst" and hit > 0.5:
        return int(base * 1.2)
    elif mode == "normal" and hit < 0.2 and ram_p < 0.5:
        return int(base * 0.8)
    else:
        return base

# -----------------------------
# API models
# -----------------------------

class SetRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None


class GossipPayload(BaseModel):
    hot_keys: List[str]

# -----------------------------
# FastAPI app (server)
# -----------------------------

app = FastAPI(title="Evolved Predictive Cache Server", version="4.0.0")


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    loop.create_task(prefetch_worker())
    loop.create_task(maintenance_worker())
    if GOSSIP_ENABLED and GOSSIP_TARGET_URL:
        loop.create_task(gossip_worker())
        print(f"[Startup] Gossip enabled, target={GOSSIP_TARGET_URL}")
    print("[Startup] Background workers started.")


@app.get("/get")
def api_get(key: str):
    start = time.time()

    value = ram_cache.get(key)
    if value is not None:
        record_access_and_predict(key)
        latency = time.time() - start
        metrics.record_get("ram", latency)
        auto_tune_if_enabled()
        return {"hit": True, "tier": "ram", "key": key, "value": value}

    value = disk_get(key)
    if value is not None:
        record_access_and_predict(key)
        mode, _, _ = compute_mode_and_pressure()
        ttl = get_adaptive_ttl_for_key(key, mode)
        ram_cache.set(key, value, ttl=ttl)
        latency = time.time() - start
        metrics.record_get("disk", latency)
        auto_tune_if_enabled()
        return {"hit": True, "tier": "disk", "key": key, "value": value}

    record_access_and_predict(key)
    latency = time.time() - start
    metrics.record_get(None, latency)
    auto_tune_if_enabled()
    return {"hit": False, "tier": None, "key": key, "value": None}


@app.post("/set")
def api_set(req: SetRequest):
    record_access_and_predict(req.key)
    mode, _, _ = compute_mode_and_pressure()
    adaptive_ttl = get_adaptive_ttl_for_key(req.key, mode)
    if req.ttl is not None:
        effective_ttl = req.ttl
    elif adaptive_ttl is not None:
        effective_ttl = adaptive_ttl
    else:
        effective_ttl = 300

    ram_cache.set(req.key, req.value, ttl=effective_ttl)
    disk_set_buffered(req.key, req.value, ttl=effective_ttl)
    metrics.record_set()
    auto_tune_if_enabled()
    return {"ok": True, "ttl": effective_ttl}


@app.delete("/delete")
def api_delete(key: str):
    ram_cache.delete(key)
    disk_delete(key)
    return {"ok": True}


@app.post("/clear")
def api_clear():
    ram_cache.clear()
    disk_clear()
    _key_stats.clear()
    _ngram_transitions.clear()
    _family_heat.clear()
    return {"ok": True}


@app.post("/gossip")
def api_gossip(payload: GossipPayload):
    if not PREFETCH_ENABLED:
        return {"ok": True, "ignored": True}
    for key in payload.hot_keys:
        try:
            _prefetch_queue.put_nowait(key)
        except queue.Full:
            break
    return {"ok": True, "received": len(payload.hot_keys)}


@app.get("/stats")
def api_stats():
    hot_keys = sorted(
        ((k, ks.current_heat(), ks.hits, ks.last_access) for k, ks in _key_stats.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    transitions_summary = {}
    for ngram, dsts in _ngram_transitions.items():
        sorted_dsts = sorted(dsts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        transitions_summary[" -> ".join(ngram)] = sorted_dsts

    families = sorted(_family_heat.items(), key=lambda kv: kv[1], reverse=True)[:20]

    mode, ram_p, disk_p = compute_mode_and_pressure()
    adaptive_ram_hint = compute_adaptive_ram_hint()

    return {
        "ram_items": ram_cache.size(),
        "disk_items": disk_size(),
        "mode": mode,
        "pressure": {
            "ram_pressure": ram_p,
            "disk_pressure": disk_p,
        },
        "metrics": {
            "total_get": metrics.total_get,
            "total_set": metrics.total_set,
            "hit_ram": metrics.hit_ram,
            "hit_disk": metrics.hit_disk,
            "miss": metrics.miss,
            "hit_ratio": metrics.hit_ratio(),
            "avg_latency_ms": metrics.avg_latency_ms(),
        },
        "config": {
            "CACHE_DIR": str(CACHE_DIR),
            "MAX_RAM_ITEMS": MAX_RAM_ITEMS,
            "ADAPTIVE_RAM_HINT": adaptive_ram_hint,
            "DEFAULT_TTL": TTL_SECONDS,
            "HOST": HOST,
            "PORT": PORT,
            "PREFETCH_ENABLED": PREFETCH_ENABLED,
            "PREFETCH_FANOUT": PREFETCH_FANOUT,
            "HEAT_DECAY_HALF_LIFE": HEAT_DECAY_HALF_LIFE,
            "AUTO_TUNE_ENABLED": AUTO_TUNE_ENABLED,
            "GOSSIP_ENABLED": GOSSIP_ENABLED,
            "GOSSIP_TARGET_URL": GOSSIP_TARGET_URL,
        },
        "hot_keys": [
            {
                "key": k,
                "heat": heat,
                "hits": hits,
                "last_access": last_access,
            }
            for k, heat, hits, last_access in hot_keys
        ],
        "transitions": transitions_summary,
        "families": [{"family": fam, "heat": heat} for fam, heat in families],
    }

# -----------------------------
# Run server in background thread
# -----------------------------

def start_server_in_thread():
    def run():
        print(f"[CacheServer] Starting on {HOST}:{PORT}")
        print(f"[CacheServer] Cache dir: {CACHE_DIR}")
        print(f"[CacheServer] MAX_RAM_ITEMS: {MAX_RAM_ITEMS}, DEFAULT_TTL: {TTL_SECONDS}")
        print(f"[CacheServer] PREFETCH_ENABLED: {PREFETCH_ENABLED}, PREFETCH_FANOUT: {PREFETCH_FANOUT}")
        print(f"[CacheServer] AUTO_TUNE_ENABLED: {AUTO_TUNE_ENABLED}")
        if GOSSIP_ENABLED and GOSSIP_TARGET_URL:
            print(f"[CacheServer] Gossip to: {GOSSIP_TARGET_URL}")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# -----------------------------
# Integrated GUI (Tkinter)
# -----------------------------

class CacheGUI:
    def __init__(self, root, server_port: int):
        self.root = root
        self.port = server_port
        self.client = httpx.Client(timeout=2.0)

        root.title("Predictive Cache Control Panel")
        root.geometry("950x650")

        self._build_controls()
        self._build_stats()
        self._schedule_refresh()

    def _build_controls(self):
        frame = ttk.LabelFrame(self.root, text="Controls")
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.prefetch_var = tk.BooleanVar(value=PREFETCH_ENABLED)
        chk_prefetch = ttk.Checkbutton(frame, text="Prefetch Enabled", variable=self.prefetch_var,
                                       command=self.on_toggle_prefetch)
        chk_prefetch.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.autotune_var = tk.BooleanVar(value=AUTO_TUNE_ENABLED)
        chk_autotune = ttk.Checkbutton(frame, text="Auto-Tune Enabled", variable=self.autotune_var,
                                       command=self.on_toggle_autotune)
        chk_autotune.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(frame, text="Max RAM Items:").grid(row=0, column=2, sticky="e")
        self.ram_spin = ttk.Spinbox(frame, from_=100, to=1000000, increment=100,
                                    width=10, command=self.on_change_ram)
        self.ram_spin.delete(0, tk.END)
        self.ram_spin.insert(0, str(MAX_RAM_ITEMS))
        self.ram_spin.grid(row=0, column=3, padx=5)

        ttk.Label(frame, text="Default TTL (s):").grid(row=0, column=4, sticky="e")
        self.ttl_spin = ttk.Spinbox(frame, from_=0, to=86400, increment=60,
                                    width=10, command=self.on_change_ttl)
        self.ttl_spin.delete(0, tk.END)
        self.ttl_spin.insert(0, str(TTL_SECONDS if TTL_SECONDS is not None else 300))
        self.ttl_spin.grid(row=0, column=5, padx=5)

        ttk.Label(frame, text="Heat Half-life (s):").grid(row=1, column=0, sticky="e")
        self.half_spin = ttk.Spinbox(frame, from_=10, to=3600, increment=10,
                                     width=10, command=self.on_change_half_life)
        self.half_spin.delete(0, tk.END)
        self.half_spin.insert(0, str(int(HEAT_DECAY_HALF_LIFE)))
        self.half_spin.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame, text="Prefetch Fanout:").grid(row=1, column=2, sticky="e")
        self.fanout_spin = ttk.Spinbox(frame, from_=0, to=20, increment=1,
                                       width=10, command=self.on_change_fanout)
        self.fanout_spin.delete(0, tk.END)
        self.fanout_spin.insert(0, str(PREFETCH_FANOUT))
        self.fanout_spin.grid(row=1, column=3, padx=5, pady=2)

        btn_clear = ttk.Button(frame, text="Clear RAM+Disk", command=self.clear_cache)
        btn_clear.grid(row=1, column=4, padx=5, pady=2)

    def _build_stats(self):
        stats_frame = ttk.LabelFrame(self.root, text="Stats & Pressure")
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.stats_text = tk.Text(stats_frame, height=10, wrap="none")
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)

        hot_frame = ttk.LabelFrame(self.root, text="Hot Keys")
        hot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.hot_list = tk.Listbox(hot_frame)
        self.hot_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fam_frame = ttk.LabelFrame(self.root, text="Families")
        fam_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fam_list = tk.Listbox(fam_frame)
        self.fam_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ---- control handlers ----

    def on_toggle_prefetch(self):
        global PREFETCH_ENABLED
        PREFETCH_ENABLED = bool(self.prefetch_var.get())
        messagebox.showinfo("Prefetch", f"Prefetch is now {'ENABLED' if PREFETCH_ENABLED else 'DISABLED'}.")

    def on_toggle_autotune(self):
        global AUTO_TUNE_ENABLED
        AUTO_TUNE_ENABLED = bool(self.autotune_var.get())
        messagebox.showinfo("Auto-Tune", f"Auto-Tune is now {'ENABLED' if AUTO_TUNE_ENABLED else 'DISABLED'}.")

    def on_change_ram(self):
        global MAX_RAM_ITEMS
        try:
            val = int(self.ram_spin.get())
            if val > 0:
                MAX_RAM_ITEMS = val
                ram_cache.max_items = val
        except ValueError:
            pass

    def on_change_ttl(self):
        global TTL_SECONDS
        try:
            val = int(self.ttl_spin.get())
            if val <= 0:
                TTL_SECONDS = None
            else:
                TTL_SECONDS = val
        except ValueError:
            pass

    def on_change_half_life(self):
        global HEAT_DECAY_HALF_LIFE, HEAT_DECAY_FACTOR_PER_SEC
        try:
            val = float(self.half_spin.get())
            if val <= 0:
                return
            HEAT_DECAY_HALF_LIFE = val
            HEAT_DECAY_FACTOR_PER_SEC = 0.5 ** (1.0 / HEAT_DECAY_HALF_LIFE)
        except ValueError:
            pass

    def on_change_fanout(self):
        global PREFETCH_FANOUT
        try:
            val = int(self.fanout_spin.get())
            if val >= 0:
                PREFETCH_FANOUT = val
        except ValueError:
            pass

    def clear_cache(self):
        try:
            resp = self.client.post(f"http://127.0.0.1:{self.port}/clear")
            if resp.status_code == 200:
                messagebox.showinfo("Clear", "Cache cleared (RAM + Disk).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {e}")

    # ---- stats polling ----

    def _schedule_refresh(self):
        self.refresh_stats()
        self.root.after(1000, self._schedule_refresh)

    def refresh_stats(self):
        try:
            resp = self.client.get(f"http://127.0.0.1:{self.port}/stats")
            if resp.status_code != 200:
                return
            data = resp.json()
        except Exception:
            return

        self.stats_text.delete("1.0", tk.END)
        cfg = data.get("config", {})
        met = data.get("metrics", {})
        pressure = data.get("pressure", {})
        mode = data.get("mode", "unknown")

        ram_p = pressure.get("ram_pressure", 0.0)
        disk_p = pressure.get("disk_pressure", 0.0)
        ram_pct = int(ram_p * 100)
        disk_pct = int(min(disk_p, 1.0) * 100)

        adaptive_hint = cfg.get("ADAPTIVE_RAM_HINT", MAX_RAM_ITEMS)

        lines = []
        lines.append(f"Mode: {mode.upper()}")
        lines.append(f"RAM items: {data.get('ram_items')} / {cfg.get('MAX_RAM_ITEMS')}  (Pressure: {ram_pct}%)")
        lines.append(f"Disk items: {data.get('disk_items')}  (Pressure: {disk_pct}%)")
        lines.append(f"Hit ratio: {met.get('hit_ratio', 0):.3f}")
        lines.append(f"Avg latency: {met.get('avg_latency_ms', 0):.2f} ms")
        lines.append(f"Total GET: {met.get('total_get', 0)}, SET: {met.get('total_set', 0)}")
        lines.append(f"Adaptive RAM hint: {adaptive_hint}")
        lines.append("")
        lines.append("Config:")
        for k, v in cfg.items():
            lines.append(f"  {k}: {v}")

        self.stats_text.insert(tk.END, "\n".join(lines))

        self.hot_list.delete(0, tk.END)
        for hk in data.get("hot_keys", []):
            self.hot_list.insert(tk.END, f"{hk['key']} | heat={hk['heat']:.2f} | hits={hk['hits']}")

        self.fam_list.delete(0, tk.END)
        for fam in data.get("families", []):
            self.fam_list.insert(tk.END, f"{fam['family']} | heat={fam['heat']:.2f}")


# -----------------------------
# Entry point
# -----------------------------

def main():
    server_thread = start_server_in_thread()
    root = tk.Tk()
    gui = CacheGUI(root, server_port=PORT)
    root.mainloop()


if __name__ == "__main__":
    main()

