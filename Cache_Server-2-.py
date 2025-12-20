#!/usr/bin/env python
"""
Predictive multi-tier caching server (RAM + Disk) with auto-loader.

Features:
- Auto-installs required libraries (fastapi, uvicorn, pydantic) if missing
- RAM tier: LRU + TTL + heat-aware promotion/demotion
- Disk tier: JSON files per key
- Prediction engine:
    - Tracks per-key stats (hits, last access, heat score)
    - Tracks simple key sequences (A -> B transitions)
    - Prefetches likely-next keys into RAM
    - Adaptive TTL based on heat
- Background workers:
    - Prefetch worker (uses sequences)
    - Cleanup worker (expired disk entries)
    - Write-coalescing buffer for disk writes

HTTP API:
    GET    /get?key=...
    POST   /set  (JSON: {key, value, ttl?})
    DELETE /delete?key=...
    POST   /clear
    GET    /stats

Config via environment variables:
    CACHE_DIR          (default: ./disk_cache)
    MAX_RAM_ITEMS      (default: 1000)
    DEFAULT_TTL        (default: None = no expiration)
    HOST               (default: 0.0.0.0)
    PORT               (default: 8000)
    PREFETCH_ENABLED   (default: 1)
    PREFETCH_FANOUT    (default: 3)
    HEAT_DECAY_HALF_LIFE (seconds, default: 300)
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List
from collections import defaultdict, deque

import threading
import queue

# -----------------------------
# Auto-loader for dependencies
# -----------------------------

REQUIRED_LIBS = ["fastapi", "uvicorn", "pydantic"]


def ensure_dependencies():
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            print(f"[AutoLoader] Missing library '{lib}', installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


ensure_dependencies()

# Safe to import external libs after auto-loader
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio

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

HOST = os.environ.get("HOST", "0.0.0.0")
try:
    PORT = int(os.environ.get("PORT", "8000"))
except ValueError:
    PORT = 8000

# Prediction / behavior tuning
PREFETCH_ENABLED = os.environ.get("PREFETCH_ENABLED", "1") != "0"
try:
    PREFETCH_FANOUT = int(os.environ.get("PREFETCH_FANOUT", "3"))
except ValueError:
    PREFETCH_FANOUT = 3

# Heat decay: half-life in seconds
try:
    HEAT_DECAY_HALF_LIFE = float(os.environ.get("HEAT_DECAY_HALF_LIFE", "300"))
except ValueError:
    HEAT_DECAY_HALF_LIFE = 300.0

HEAT_DECAY_FACTOR_PER_SEC = 0.5 ** (1.0 / HEAT_DECAY_HALF_LIFE) if HEAT_DECAY_HALF_LIFE > 0 else 1.0

# -----------------------------
# RAM Tier (LRU + heat-aware)
# -----------------------------


class LRUCache:
    """
    LRU cache with TTL and simple heat-aware eviction hints.
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
        items = sorted(self._store.items(), key=lambda kv: kv[1][1])  # by last_access_time
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
    tmp = path.with_suffix(".tmp")
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


# write-coalescing: queue writes, flush in background
_write_queue: "queue.Queue[Tuple[str, Any, Optional[int]]]" = queue.Queue()


def disk_set_buffered(key: str, value: Any, ttl: Optional[int]) -> None:
    _write_queue.put((key, value, ttl))


def _disk_worker():
    """
    Background worker that batches disk writes.
    """
    while True:
        try:
            batch: List[Tuple[str, Any, Optional[int]]] = []
            # block for first item
            item = _write_queue.get()
            if item is None:
                # shutdown signal if ever used
                break
            batch.append(item)
            # non-blocking drain with small limit
            try:
                while len(batch) < 100:
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


def cleanup_expired_on_disk(max_per_cycle: int = 100) -> None:
    """
    Best-effort cleanup of expired entries on disk.
    """
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
            # on error, skip or delete
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
    Tracks stats per key:
    - hits
    - last_access
    - heat (decaying score)
    """

    def __init__(self):
        self.hits = 0
        self.last_access = 0.0
        self.heat = 0.0  # decays over time

    def record_access(self):
        now = time.time()
        # decay existing heat
        if self.last_access > 0:
            dt = now - self.last_access
            self.heat *= (HEAT_DECAY_FACTOR_PER_SEC ** dt)
        # increase heat for this hit
        self.heat += 1.0
        self.hits += 1
        self.last_access = now

    def current_heat(self) -> float:
        if self.last_access <= 0:
            return self.heat
        dt = time.time() - self.last_access
        return self.heat * (HEAT_DECAY_FACTOR_PER_SEC ** dt)


# key -> stats
_key_stats: Dict[str, KeyStats] = defaultdict(KeyStats)

# Transition map: A -> {B: count}
_transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# Deque to remember last accessed key per "session" (simplified)
_last_key_accessed_global: Optional[str] = None

# Prefetch queue (keys to warm into RAM)
_prefetch_queue: "queue.Queue[str]" = queue.Queue()


def record_access_and_predict(key: str) -> None:
    """
    Update stats and transitions, enqueue prefetches.
    """
    global _last_key_accessed_global
    ks = _key_stats[key]
    ks.record_access()

    if _last_key_accessed_global is not None and _last_key_accessed_global != key:
        prev = _last_key_accessed_global
        _transition_counts[prev][key] += 1

    _last_key_accessed_global = key

    if PREFETCH_ENABLED:
        likely_next = predict_next_keys(key, limit=PREFETCH_FANOUT)
        for nxt in likely_next:
            if nxt != key:
                try:
                    _prefetch_queue.put_nowait(nxt)
                except queue.Full:
                    break


def predict_next_keys(key: str, limit: int = 3) -> List[str]:
    """
    Based on simple transition counts: from 'key', what are the most likely next keys?
    """
    trans = _transition_counts.get(key)
    if not trans:
        return []
    items = sorted(trans.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:limit]]


def get_adaptive_ttl_for_key(key: str) -> Optional[int]:
    """
    Generate an adaptive TTL based on key heat.
    """
    ks = _key_stats.get(key)
    if ks is None:
        return TTL_SECONDS
    heat = ks.current_heat()
    if TTL_SECONDS is None:
        base = 300  # 5 min default if no global TTL
    else:
        base = TTL_SECONDS
    # simple scaling: hotter keys keep longer TTL, but capped
    # heat ~ 1 => base
    # higher heat => up to 4x base
    factor = 1.0 + min(heat, 3.0)
    ttl = int(base * factor)
    return ttl


async def prefetch_worker():
    """
    Async worker that pulls keys from _prefetch_queue and warms them into RAM.
    """
    while True:
        try:
            key = await asyncio.get_event_loop().run_in_executor(None, _prefetch_queue.get)
            if key is None:
                continue
            # If already in RAM, skip
            if ram_cache.get(key) is not None:
                continue
            # Try disk
            value = disk_get(key)
            if value is not None:
                ttl = get_adaptive_ttl_for_key(key)
                ram_cache.set(key, value, ttl=ttl)
        except Exception as e:
            print(f"[Prefetch] Worker error: {e}")
            await asyncio.sleep(1.0)


async def maintenance_worker():
    """
    Periodic maintenance:
    - cleanup expired disk entries
    - can evolve to do more (rebalance, logging, etc.)
    """
    while True:
        try:
            cleanup_expired_on_disk(max_per_cycle=200)
        except Exception as e:
            print(f"[Maintenance] Error: {e}")
        await asyncio.sleep(30.0)


# -----------------------------
# API models
# -----------------------------


class SetRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None  # seconds


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="Predictive Multi-tier Cache Server", version="2.0.0")


@app.on_event("startup")
async def on_startup():
    # Start background async workers
    loop = asyncio.get_event_loop()
    loop.create_task(prefetch_worker())
    loop.create_task(maintenance_worker())
    print("[Startup] Background workers started.")


@app.get("/get")
def api_get(key: str):
    """
    Get a value from cache.
    Search order: RAM -> Disk.
    If found on disk, promote to RAM using adaptive TTL.
    Also records access for prediction.
    """
    # 1. RAM
    value = ram_cache.get(key)
    if value is not None:
        record_access_and_predict(key)
        return {"hit": True, "tier": "ram", "key": key, "value": value}

    # 2. Disk
    value = disk_get(key)
    if value is not None:
        record_access_and_predict(key)
        ttl = get_adaptive_ttl_for_key(key)
        ram_cache.set(key, value, ttl=ttl)
        return {"hit": True, "tier": "disk", "key": key, "value": value}

    # 3. Miss
    # still record as seen (cold access)
    record_access_and_predict(key)
    return {"hit": False, "tier": None, "key": key, "value": None}


@app.post("/set")
def api_set(req: SetRequest):
    """
    Set a value in cache (write-through:
    - RAM immediately
    - Disk via buffered writer

    TTL priority: request.ttl > adaptive_ttl > DEFAULT_TTL > fallback.
    """
    # record access for prediction
    record_access_and_predict(req.key)

    adaptive_ttl = get_adaptive_ttl_for_key(req.key)
    if req.ttl is not None:
        effective_ttl = req.ttl
    elif adaptive_ttl is not None:
        effective_ttl = adaptive_ttl
    else:
        effective_ttl = 300

    ram_cache.set(req.key, req.value, ttl=effective_ttl)
    disk_set_buffered(req.key, req.value, ttl=effective_ttl)
    return {"ok": True, "ttl": effective_ttl}


@app.delete("/delete")
def api_delete(key: str):
    """
    Delete a key from both RAM and Disk.
    """
    ram_cache.delete(key)
    disk_delete(key)
    return {"ok": True}


@app.post("/clear")
def api_clear():
    """
    Clear all cached data (RAM + Disk).
    """
    ram_cache.clear()
    disk_clear()
    _key_stats.clear()
    _transition_counts.clear()
    return {"ok": True}


@app.get("/stats")
def api_stats():
    """
    Basic stats: RAM size, Disk size, config, prediction hints.
    """
    # top hot keys
    hot_keys = sorted(
        ((k, ks.current_heat(), ks.hits, ks.last_access) for k, ks in _key_stats.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    transitions_summary = {}
    for src, dsts in _transition_counts.items():
        sorted_dsts = sorted(dsts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        transitions_summary[src] = sorted_dsts

    return {
        "ram_items": ram_cache.size(),
        "disk_items": disk_size(),
        "config": {
            "CACHE_DIR": str(CACHE_DIR),
            "MAX_RAM_ITEMS": MAX_RAM_ITEMS,
            "DEFAULT_TTL": TTL_SECONDS,
            "HOST": HOST,
            "PORT": PORT,
            "PREFETCH_ENABLED": PREFETCH_ENABLED,
            "PREFETCH_FANOUT": PREFETCH_FANOUT,
            "HEAT_DECAY_HALF_LIFE": HEAT_DECAY_HALF_LIFE,
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
    }


# -----------------------------
# Entry point
# -----------------------------


def main():
    print(f"[CacheServer] Starting on {HOST}:{PORT}")
    print(f"[CacheServer] Cache dir: {CACHE_DIR}")
    print(f"[CacheServer] MAX_RAM_ITEMS: {MAX_RAM_ITEMS}, DEFAULT_TTL: {TTL_SECONDS}")
    print(f"[CacheServer] PREFETCH_ENABLED: {PREFETCH_ENABLED}, PREFETCH_FANOUT: {PREFETCH_FANOUT}")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

