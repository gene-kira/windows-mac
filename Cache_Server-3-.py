#!/usr/bin/env python
"""
Evolved predictive multi-tier caching server (RAM + Disk) with auto-loader.

Concept:
- Multi-tier cache: RAM (Tier 1) + Disk (Tier 2)
- Predictive engine with:
    - Variable-order sequence tracking (n-grams up to order 3)
    - Per-key heat (decaying score)
    - Family clustering by prefix (e.g. "user:123:*")
    - Prefetching next keys + family members
- Resource intelligence:
    - Hit/miss tracking
    - Heat-aware LRU eviction
    - Simple adaptive hints for RAM sizing & TTL
- Autonomous behavior:
    - Background workers (prefetch, maintenance, metrics)
    - Optional distributed gossip hooks for hot keys

HTTP API:
    GET    /get?key=...
    POST   /set  (JSON: {key, value, ttl?})
    DELETE /delete?key=...
    POST   /clear
    GET    /stats
    POST   /gossip   (optional inbound hints from other nodes)

Config via environment variables:
    CACHE_DIR              (default: ./disk_cache)
    MAX_RAM_ITEMS          (default: 1000)
    DEFAULT_TTL            (default: None = no expiration)
    HOST                   (default: 0.0.0.0)
    PORT                   (default: 8000)
    PREFETCH_ENABLED       (default: 1)
    PREFETCH_FANOUT        (default: 3)
    HEAT_DECAY_HALF_LIFE   (seconds, default: 300)
    GOSSIP_ENABLED         (default: 0)
    GOSSIP_TARGET_URL      (default: empty)
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


ensure_dependencies()

# Safe imports after auto-loader
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import httpx

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
        # heat-aware LRU:
        # primary: last access time
        # secondary: inverse of heat (colder evicted first)
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
    """
    Background worker that batches disk writes.
    """
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
    Tracks stats per key:
    - hits
    - last_access
    - heat (decaying score)
    - per-hour access counts (0-23)
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

    def hottest_hour(self) -> int:
        return max(range(24), key=lambda h: self.hour_counts[h])


_key_stats: Dict[str, KeyStats] = defaultdict(KeyStats)

# n-gram transitions: tuple(prev_keys) -> {next_key: count}
_ngram_transitions: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# Global “session” history (up to last 3 keys)
_access_history: Deque[str] = deque(maxlen=3)

# Prefetch queue for keys to warm
_prefetch_queue: "queue.Queue[str]" = queue.Queue()

# Family patterns: family_id -> heat
_family_heat: Dict[str, float] = defaultdict(float)


def key_family(key: str) -> Optional[str]:
    """
    Simple family detection by prefix before second colon:
        "user:123:settings" -> "user:123:*"
        "game:asset:chunk_1" -> "game:asset:*"
    """
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
    # update n-gram transitions for orders 1..len(history)
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
    """
    Use variable-order n-gram transitions:
    - try longest history first, then shorter
    - merge results with counts
    """
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


def get_adaptive_ttl_for_key(key: str) -> Optional[int]:
    ks = _key_stats.get(key)
    base = TTL_SECONDS if TTL_SECONDS is not None else 300
    if not ks:
        return base
    heat = ks.current_heat()
    fam = key_family(key)
    fam_heat = _family_heat.get(fam, 0.0) if fam else 0.0
    total_heat = heat + 0.5 * fam_heat
    factor = 1.0 + min(total_heat, 4.0)
    ttl = int(base * factor)
    return ttl


async def prefetch_worker():
    """
    Async worker that warms keys into RAM when requested by prediction engine.
    """
    while True:
        try:
            key = await asyncio.get_event_loop().run_in_executor(None, _prefetch_queue.get)
            if key is None:
                continue
            if ram_cache.get(key) is not None:
                continue
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
    - could later rebalance or compress
    """
    while True:
        try:
            cleanup_expired_on_disk(max_per_cycle=300)
        except Exception as e:
            print(f"[Maintenance] Error: {e}")
        await asyncio.sleep(30.0)


async def gossip_worker():
    """
    Optional worker that gossips top hot keys to another node.
    """
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
# API models
# -----------------------------

class SetRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None  # seconds


class GossipPayload(BaseModel):
    hot_keys: List[str]

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="Evolved Predictive Cache Server", version="3.0.0")


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
    """
    Get a value from cache.
    Search order: RAM -> Disk.
    If found on disk, promote to RAM using adaptive TTL.
    Also records access for prediction & metrics.
    """
    start = time.time()

    value = ram_cache.get(key)
    if value is not None:
        record_access_and_predict(key)
        latency = time.time() - start
        metrics.record_get("ram", latency)
        return {"hit": True, "tier": "ram", "key": key, "value": value}

    value = disk_get(key)
    if value is not None:
        record_access_and_predict(key)
        ttl = get_adaptive_ttl_for_key(key)
        ram_cache.set(key, value, ttl=ttl)
        latency = time.time() - start
        metrics.record_get("disk", latency)
        return {"hit": True, "tier": "disk", "key": key, "value": value}

    record_access_and_predict(key)
    latency = time.time() - start
    metrics.record_get(None, latency)
    return {"hit": False, "tier": None, "key": key, "value": None}


@app.post("/set")
def api_set(req: SetRequest):
    """
    Set a value in cache (write-through):
    - RAM immediately
    - Disk via buffered writer

    TTL priority:
        request.ttl > adaptive_ttl > DEFAULT_TTL > fallback.
    """
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
    metrics.record_set()
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
    Clear all cached data (RAM + Disk) and prediction state.
    """
    ram_cache.clear()
    disk_clear()
    _key_stats.clear()
    _ngram_transitions.clear()
    _family_heat.clear()
    return {"ok": True}


@app.post("/gossip")
def api_gossip(payload: GossipPayload):
    """
    Inbound gossip: remote node sends hot keys.
    We treat them as candidates for warming.
    """
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
    """
    Stats:
    - RAM size, Disk size
    - Config
    - Hit/miss, avg latency
    - Top hot keys
    - Example n-gram transitions
    - Family heat
    """
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

    adaptive_ram_hint = int(MAX_RAM_ITEMS * (1.0 + metrics.hit_ratio()))

    return {
        "ram_items": ram_cache.size(),
        "disk_items": disk_size(),
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
# Entry point
# -----------------------------

def main():
    print(f"[CacheServer] Starting on {HOST}:{PORT}")
    print(f"[CacheServer] Cache dir: {CACHE_DIR}")
    print(f"[CacheServer] MAX_RAM_ITEMS: {MAX_RAM_ITEMS}, DEFAULT_TTL: {TTL_SECONDS}")
    print(f"[CacheServer] PREFETCH_ENABLED: {PREFETCH_ENABLED}, PREFETCH_FANOUT: {PREFETCH_FANOUT}")
    if GOSSIP_ENABLED and GOSSIP_TARGET_URL:
        print(f"[CacheServer] Gossip to: {GOSSIP_TARGET_URL}")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

