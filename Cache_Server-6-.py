#!/usr/bin/env python
"""
Autonomous predictive multi-tier caching server (RAM + Disk) with integrated GUI.

Features:
- FastAPI + uvicorn cache server
- RAM + Disk tiers
- Improved prediction engine:
    - per-key stats
    - n-gram transitions (1..3)
    - confidence scoring (frequency, recency, order, time-of-day)
    - family clustering and family transitions
- Modes: normal / burst / survival
- Policy brain:
    - max_ram_items
    - base_ttl
    - prefetch_fanout
    - heat_half_life
    - prefetch_enabled
- Health score:
    score = f(hit_ratio, latency, RAM pressure, Disk pressure)
- Long-term memory:
    - health snapshots over time
    - summaries over last 60s/300s
- Reflex layer:
    - fast micro-adjustments on spikes (latency, hit collapse, survival)
- Strategist layer:
    - slow pattern-aware adjustments using health history
- Learning loop:
    - periodically perturbs policy
    - evaluates health
    - keeps or reverts
- Tkinter GUI:
    - shows health, mode, pressures
    - shows current policy
    - shows health summary & mode timeline
    - shows adaptive RAM hint
    - shows last experiments (accepted/rejected)
    - manual override + auto-tune toggle

Run:
    python autonomous_cache_gui.py
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
import random

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

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import httpx
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------
# Base config (initial seeds)
# -----------------------------

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./disk_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("HOST", "127.0.0.1")
try:
    PORT = int(os.environ.get("PORT", "8000"))
except ValueError:
    PORT = 8000

try:
    SEED_MAX_RAM_ITEMS = int(os.environ.get("MAX_RAM_ITEMS", "1000"))
except ValueError:
    SEED_MAX_RAM_ITEMS = 1000

_default_ttl_env = os.environ.get("DEFAULT_TTL", "").strip()
SEED_TTL_SECONDS: Optional[int] = int(_default_ttl_env) if _default_ttl_env else 300

try:
    SEED_HEAT_DECAY_HALF_LIFE = float(os.environ.get("HEAT_DECAY_HALF_LIFE", "300"))
except ValueError:
    SEED_HEAT_DECAY_HALF_LIFE = 300.0

try:
    SEED_PREFETCH_FANOUT = int(os.environ.get("PREFETCH_FANOUT", "3"))
except ValueError:
    SEED_PREFETCH_FANOUT = 3

SEED_PREFETCH_ENABLED = os.environ.get("PREFETCH_ENABLED", "1") != "0"

GOSSIP_ENABLED = os.environ.get("GOSSIP_ENABLED", "0") != "0"
GOSSIP_TARGET_URL = os.environ.get("GOSSIP_TARGET_URL", "").strip() or None

AUTO_TUNE_ENABLED = True

# -----------------------------
# Policy brain
# -----------------------------

class Policy:
    def __init__(
        self,
        max_ram_items: int,
        base_ttl: int,
        prefetch_fanout: int,
        heat_half_life: float,
        prefetch_enabled: bool,
    ):
        self.max_ram_items = max_ram_items
        self.base_ttl = base_ttl
        self.prefetch_fanout = prefetch_fanout
        self.heat_half_life = heat_half_life
        self.prefetch_enabled = prefetch_enabled

    def copy(self) -> "Policy":
        return Policy(
            self.max_ram_items,
            self.base_ttl,
            self.prefetch_fanout,
            self.heat_half_life,
            self.prefetch_enabled,
        )

    def to_dict(self) -> dict:
        return {
            "max_ram_items": self.max_ram_items,
            "base_ttl": self.base_ttl,
            "prefetch_fanout": self.prefetch_fanout,
            "heat_half_life": self.heat_half_life,
            "prefetch_enabled": self.prefetch_enabled,
        }


policy = Policy(
    max_ram_items=SEED_MAX_RAM_ITEMS,
    base_ttl=SEED_TTL_SECONDS,
    prefetch_fanout=SEED_PREFETCH_FANOUT,
    heat_half_life=SEED_HEAT_DECAY_HALF_LIFE,
    prefetch_enabled=SEED_PREFETCH_ENABLED,
)


def heat_decay_factor_per_sec() -> float:
    return 0.5 ** (1.0 / policy.heat_half_life) if policy.heat_half_life > 0 else 1.0

# -----------------------------
# Metrics and health
# -----------------------------

class Metrics:
    def __init__(self):
        self.total_get = 0
        self.total_set = 0
        self.hit_ram = 0
        self.hit_disk = 0
        self.miss = 0
        self._latency_samples = deque(maxlen=500)

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


def compute_pressures():
    ram_used = ram_cache.size()
    ram_capacity = max(policy.max_ram_items, 1)
    ram_pressure = ram_used / ram_capacity

    disk_items = disk_size()
    disk_soft_cap = max(1000, ram_capacity * 10)
    disk_pressure = min(disk_items / disk_soft_cap, 2.0)

    return ram_pressure, disk_pressure, disk_items


def compute_mode():
    ram_p, disk_p, _ = compute_pressures()
    hit = metrics.hit_ratio()

    if ram_p > 0.9 or disk_p > 1.0:
        return "survival"
    if (ram_p > 0.6 or disk_p > 0.7) and hit > 0.5:
        return "burst"
    return "normal"


def compute_health() -> float:
    hit = metrics.hit_ratio()
    lat = metrics.avg_latency_ms()
    ram_p, disk_p, _ = compute_pressures()

    score = 0.0
    score += hit * 0.5
    score -= min(lat / 1000.0, 1.0) * 0.2
    score -= min(ram_p, 1.0) * 0.15
    score -= min(disk_p / 1.5, 1.0) * 0.15
    return max(0.0, min(1.0, score))

# -----------------------------
# Long-term health memory
# -----------------------------

class HealthSnapshot:
    def __init__(self, ts: float, mode: str, health: float,
                 hit_ratio: float, avg_latency_ms: float,
                 ram_pressure: float, disk_pressure: float):
        self.ts = ts
        self.mode = mode
        self.health = health
        self.hit_ratio = hit_ratio
        self.avg_latency_ms = avg_latency_ms
        self.ram_pressure = ram_pressure
        self.disk_pressure = disk_pressure


health_history: Deque[HealthSnapshot] = deque(maxlen=300)
_last_health_sample_time = 0.0


def record_health_snapshot():
    global _last_health_sample_time
    now = time.time()
    if now - _last_health_sample_time < 1.0:
        return
    _last_health_sample_time = now

    mode = compute_mode()
    health = compute_health()
    hit = metrics.hit_ratio()
    lat = metrics.avg_latency_ms()
    ram_p, disk_p, _ = compute_pressures()

    health_history.append(
        HealthSnapshot(
            ts=now,
            mode=mode,
            health=health,
            hit_ratio=hit,
            avg_latency_ms=lat,
            ram_pressure=ram_p,
            disk_pressure=disk_p,
        )
    )

# -----------------------------
# RAM Tier (LRU + heat-aware)
# -----------------------------

class LRUCache:
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


ram_cache = LRUCache(policy.max_ram_items)

# -----------------------------
# Disk tier + write-coalescing
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
# Prediction engine (sharper)
# -----------------------------

class KeyStats:
    """
    Per-key behavioral stats:
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
            self.heat *= (heat_decay_factor_per_sec() ** dt)
        self.heat += 1.0
        self.hits += 1
        self.last_access = now
        hour = time.localtime(now).tm_hour
        self.hour_counts[hour] += 1

    def current_heat(self) -> float:
        if self.last_access <= 0:
            return self.heat
        dt = time.time() - self.last_access
        return self.heat * (heat_decay_factor_per_sec() ** dt)

    def hour_weight(self, hour: int) -> float:
        total = sum(self.hour_counts) or 1
        return self.hour_counts[hour] / total


_key_stats: Dict[str, KeyStats] = defaultdict(KeyStats)

# n-gram transitions:
#   ngram -> { next_key: {"count": float, "last_seen": float} }
_ngram_transitions: Dict[Tuple[str, ...], Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

_access_history: Deque[str] = deque(maxlen=5)

_prefetch_queue: "queue.Queue[str]" = queue.Queue()

_family_heat: Dict[str, float] = defaultdict(float)

# family_n-gram -> { next_family: {"count": float, "last_seen": float} }
_family_transitions: Dict[Tuple[str, ...], Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))


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
    """
    Record access, update key + family stats,
    update n-gram transitions, and enqueue high-confidence prefetches.
    """
    ks = _key_stats[key]
    ks.record_access()
    _update_family_heat(key)

    now = time.time()
    history = tuple(_access_history)

    # Key-level n-grams
    if history:
        for order in range(1, min(3, len(history)) + 1):
            ngram = history[-order:]
            trans = _ngram_transitions[ngram]
            entry = trans.get(key)
            if entry is None:
                trans[key] = {"count": 1.0, "last_seen": now}
            else:
                entry["count"] += 1.0
                entry["last_seen"] = now

    # Family-level n-grams
    fam = key_family(key)
    if fam:
        fam_history = tuple(f for f in (key_family(k) for k in history) if f is not None)
        if fam_history:
            for order in range(1, min(3, len(fam_history)) + 1):
                fam_ngram = fam_history[-order:]
                ftrans = _family_transitions[fam_ngram]
                fentry = ftrans.get(fam)
                if fentry is None:
                    ftrans[fam] = {"count": 1.0, "last_seen": now}
                else:
                    fentry["count"] += 1.0
                    fentry["last_seen"] = now

    _access_history.append(key)

    if policy.prefetch_enabled and policy.prefetch_fanout > 0:
        current_hour = time.localtime(now).tm_hour
        candidates = predict_next_keys(
            history=tuple(_access_history),
            current_key=key,
            current_hour=current_hour,
            limit=policy.prefetch_fanout,
        )
        for nxt in candidates:
            if nxt != key:
                try:
                    _prefetch_queue.put_nowait(nxt)
                except queue.Full:
                    break


def predict_next_keys(
    history: Tuple[str, ...],
    current_key: str,
    current_hour: int,
    limit: int = 3,
) -> List[str]:
    """
    Predict next keys using:
    - variable-order n-grams (1..3)
    - frequency (count)
    - recency (time decay)
    - order weight (higher order = more weight)
    - time-of-day weighting
    - family reinforcement
    Returns keys sorted by confidence, filtered by minimum confidence threshold.
    """
    now = time.time()
    candidates: Dict[str, float] = defaultdict(float)

    def add_ngram_scores(seq: Tuple[str, ...], weight_order: float):
        trans = _ngram_transitions.get(seq)
        if not trans:
            return
        for nxt, info in trans.items():
            count = info.get("count", 0.0)
            last_seen = info.get("last_seen", 0.0)
            age = max(now - last_seen, 0.0)
            recency_factor = 0.5 ** (age / 300.0)  # 5-min half-life
            key_stats = _key_stats.get(nxt)
            hour_weight = key_stats.hour_weight(current_hour) if key_stats else 1.0 / 24.0
            score = count * recency_factor * weight_order * (0.5 + hour_weight)
            candidates[nxt] += score

    seq = list(history[-4:] + (current_key,))
    for order in range(min(3, len(seq)), 0, -1):
        ngram = tuple(seq[-order:])
        add_ngram_scores(ngram, weight_order=order)

    cur_fam = key_family(current_key)
    if cur_fam:
        fam_seq = tuple(f for f in (key_family(k) for k in history[-4:] + (current_key,)) if f is not None)

        def add_family_scores(fseq: Tuple[str, ...], weight_order: float):
            ftrans = _family_transitions.get(fseq)
            if not ftrans:
                return
            for next_fam, info in ftrans.items():
                fcount = info.get("count", 0.0)
                flast = info.get("last_seen", 0.0)
                fage = max(now - flast, 0.0)
                frecency = 0.5 ** (fage / 600.0)
                fam_score = fcount * frecency * weight_order
                for k, ks in _key_stats.items():
                    if key_family(k) == next_fam:
                        candidates[k] += fam_score * (0.2 + ks.current_heat() * 0.1)

        for order in range(min(3, len(fam_seq)), 0, -1):
            fng = tuple(fam_seq[-order:])
            add_family_scores(fng, weight_order=order)

    if not candidates:
        return []

    max_score = max(candidates.values())
    min_threshold = max_score * 0.2

    filtered = [(k, s) for k, s in candidates.items() if s >= min_threshold]
    filtered.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in filtered[:limit]]


def get_adaptive_ttl_for_key(key: str, mode: str) -> int:
    ks = _key_stats.get(key)
    base = policy.base_ttl or 300

    if not ks:
        ttl = base
    else:
        heat = ks.current_heat()
        fam = key_family(key)
        fam_heat = _family_heat.get(fam, 0.0) if fam else 0.0
        total_heat = heat + 0.7 * fam_heat
        raw_factor = 1.0 + min(total_heat, 6.0)
        factor = 1.0 + (raw_factor - 1.0) * 0.7
        ttl = int(base * factor)

    if mode == "burst":
        ttl = int(ttl * 1.2)
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
                mode = compute_mode()
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
# Reflex layer
# -----------------------------

def reflex_adjustments():
    """
    Fast, local reactions to acute conditions.
    Small in-place nudges; learning loop does long-term mutations.
    """
    mode = compute_mode()
    health = compute_health()
    hit = metrics.hit_ratio()
    ram_p, disk_p, _ = compute_pressures()

    # Latency spike + high RAM pressure -> reduce prefetch
    if metrics.avg_latency_ms() > 50.0 and ram_p > 0.8:
        old = policy.prefetch_fanout
        policy.prefetch_fanout = max(0, int(policy.prefetch_fanout * 0.7))
        if old != policy.prefetch_fanout:
            print(f"[Reflex] Latency+RAM spike: fanout {old} -> {policy.prefetch_fanout}")

    # Hit ratio collapse with no extreme pressure -> reduce base TTL
    if hit < 0.2 and (ram_p < 0.8 and disk_p < 0.8):
        old = policy.base_ttl
        policy.base_ttl = max(60, int(policy.base_ttl * 0.8))
        if old != policy.base_ttl:
            print(f"[Reflex] Hit collapse: base_ttl {old} -> {policy.base_ttl}")

    # Survival mode -> decrease heat half-life so new patterns dominate
    if mode == "survival":
        old = policy.heat_half_life
        policy.heat_half_life = max(60.0, policy.heat_half_life * 0.8)
        if abs(old - policy.heat_half_life) > 5.0:
            print(f"[Reflex] Survival: heat_half_life {old:.1f} -> {policy.heat_half_life:.1f}")

# -----------------------------
# Strategist layer
# -----------------------------

def summarize_recent_health(window_sec: float = 60.0):
    if not health_history:
        return None
    now = time.time()
    window = [h for h in health_history if now - h.ts <= window_sec]
    if not window:
        return None

    avg_health = sum(h.health for h in window) / len(window)
    avg_hit = sum(h.hit_ratio for h in window) / len(window)
    avg_lat = sum(h.avg_latency_ms for h in window) / len(window)
    avg_ram_p = sum(h.ram_pressure for h in window) / len(window)
    avg_disk_p = sum(h.disk_pressure for h in window) / len(window)
    modes = [h.mode for h in window]
    mode_counts = defaultdict(int)
    for m in modes:
        mode_counts[m] += 1
    dominant_mode = max(mode_counts, key=mode_counts.get)

    return {
        "avg_health": avg_health,
        "avg_hit_ratio": avg_hit,
        "avg_latency_ms": avg_lat,
        "avg_ram_pressure": avg_ram_p,
        "avg_disk_pressure": avg_disk_p,
        "dominant_mode": dominant_mode,
    }


def strategist_adjustments():
    """
    Slower, pattern-based adjustments using health_history.
    """
    summary = summarize_recent_health(window_sec=120.0)
    if summary is None:
        return

    avg_health = summary["avg_health"]
    avg_hit = summary["avg_hit_ratio"]
    avg_ram_p = summary["avg_ram_pressure"]
    avg_disk_p = summary["avg_disk_pressure"]
    dominant_mode = summary["dominant_mode"]

    # Normal mode, good health -> slightly increase prefetch fanout
    if dominant_mode == "normal" and avg_health > 0.7 and avg_hit > 0.6 and avg_ram_p < 0.7:
        old = policy.prefetch_fanout
        policy.prefetch_fanout = min(policy.prefetch_fanout + 1, 10)
        if policy.prefetch_fanout != old:
            print(f"[Strategist] Normal-stable: fanout {old} -> {policy.prefetch_fanout}")

    # Burst mode, health stable, low disk pressure -> increase base TTL a bit
    if dominant_mode == "burst" and avg_health > 0.6 and avg_disk_p < 0.8:
        old = policy.base_ttl
        policy.base_ttl = min(int(policy.base_ttl * 1.1), 7200)
        if policy.base_ttl != old:
            print(f"[Strategist] Burst-stable: base_ttl {old} -> {policy.base_ttl}")

    # Survival mode, poor health -> gently shrink RAM footprint
    if dominant_mode == "survival" and avg_health < 0.4:
        old = policy.max_ram_items
        new_val = max(100, int(policy.max_ram_items * 0.9))
        if new_val != old:
            policy.max_ram_items = new_val
            ram_cache.max_items = new_val
            print(f"[Strategist] Survival-bad: max_ram_items {old} -> {policy.max_ram_items}")

# -----------------------------
# Learning loop (experiments)
# -----------------------------

class ExperimentRecord:
    def __init__(self, before: dict, after: dict, health_before: float, health_after: float, accepted: bool):
        self.before = before
        self.after = after
        self.health_before = health_before
        self.health_after = health_after
        self.accepted = accepted
        self.timestamp = time.time()


recent_experiments: Deque[ExperimentRecord] = deque(maxlen=20)
_last_experiment_time = 0.0


def perturb_policy(base: Policy) -> Policy:
    new = base.copy()
    for attr, low, high, step in [
        ("prefetch_fanout", 0, 10, 1),
        ("base_ttl", 30, 3600, 30),
        ("heat_half_life", 30, 1800, 10),
        ("max_ram_items", 100, 2000000, 100),
    ]:
        if random.random() < 0.4:
            direction = random.choice([-1, 1])
            change_factor = 1.0 + direction * random.uniform(0.1, 0.3)
            val = getattr(new, attr)
            if attr in ("max_ram_items", "base_ttl"):
                val = int(val * change_factor)
            else:
                val = val * change_factor
            val = max(low, min(high, val))
            setattr(new, attr, val)

    if random.random() < 0.2:
        new.prefetch_enabled = not new.prefetch_enabled

    return new


def apply_policy(p: Policy):
    global policy, ram_cache
    policy = p
    ram_cache.max_items = policy.max_ram_items


def learning_step():
    global _last_experiment_time
    if not AUTO_TUNE_ENABLED:
        return

    now = time.time()
    if now - _last_experiment_time < 20.0:
        return
    _last_experiment_time = now

    health_before = compute_health()
    current_policy = policy
    candidate = perturb_policy(current_policy)
    apply_policy(candidate)
    time.sleep(5.0)
    health_after = compute_health()

    improvement = health_after - health_before
    accepted = improvement >= 0.02

    if not accepted:
        apply_policy(current_policy)

    rec = ExperimentRecord(
        before=current_policy.to_dict(),
        after=candidate.to_dict(),
        health_before=health_before,
        health_after=health_after,
        accepted=accepted,
    )
    recent_experiments.append(rec)
    print(f"[Learning] Experiment accepted={accepted} Δhealth={improvement:.3f}")


async def learning_worker():
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(None, learning_step)
        except Exception as e:
            print(f"[Learning] Error: {e}")
        await asyncio.sleep(10.0)

# -----------------------------
# Brain worker (memory + reflex + strategist)
# -----------------------------

async def brain_worker():
    last_strategist_run = 0.0
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(None, record_health_snapshot)
            await loop.run_in_executor(None, reflex_adjustments)

            now = time.time()
            if now - last_strategist_run > 15.0:
                await loop.run_in_executor(None, strategist_adjustments)
                last_strategist_run = now

        except Exception as e:
            print(f"[Brain] Error: {e}")
        await asyncio.sleep(1.0)

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
# FastAPI app
# -----------------------------

app = FastAPI(title="Autonomous Predictive Cache Server", version="6.0.0")


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    loop.create_task(prefetch_worker())
    loop.create_task(maintenance_worker())
    loop.create_task(learning_worker())
    loop.create_task(brain_worker())
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
        return {"hit": True, "tier": "ram", "key": key, "value": value}

    value = disk_get(key)
    if value is not None:
        record_access_and_predict(key)
        mode = compute_mode()
        ttl = get_adaptive_ttl_for_key(key, mode)
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
    record_access_and_predict(req.key)
    mode = compute_mode()
    adaptive_ttl = get_adaptive_ttl_for_key(req.key, mode)
    if req.ttl is not None:
        effective_ttl = req.ttl
    else:
        effective_ttl = adaptive_ttl

    ram_cache.set(req.key, req.value, ttl=effective_ttl)
    disk_set_buffered(req.key, req.value, ttl=effective_ttl)
    metrics.record_set()
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
    _family_transitions.clear()
    return {"ok": True}


@app.post("/gossip")
def api_gossip(payload: GossipPayload):
    if not policy.prefetch_enabled:
        return {"ok": True, "ignored": True}
    for key in payload.hot_keys:
        try:
            _prefetch_queue.put_nowait(key)
        except queue.Full:
            break
    return {"ok": True, "received": len(payload.hot_keys)}


def compute_adaptive_ram_hint():
    base = policy.max_ram_items
    hit = metrics.hit_ratio()
    mode = compute_mode()
    ram_p, _, _ = compute_pressures()

    if mode == "survival" and ram_p > 0.95 and hit > 0.6:
        return int(base * 1.5)
    elif mode == "burst" and hit > 0.5:
        return int(base * 1.2)
    elif mode == "normal" and hit < 0.2 and ram_p < 0.5:
        return int(base * 0.8)
    else:
        return base


@app.get("/stats")
def api_stats():
    hot_keys = sorted(
        ((k, ks.current_heat(), ks.hits, ks.last_access) for k, ks in _key_stats.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    transitions_summary = {}
    for ngram, dsts in _ngram_transitions.items():
        sorted_dsts = sorted(dsts.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:5]
        transitions_summary[" -> ".join(ngram)] = [(k, v["count"]) for k, v in sorted_dsts]

    families = sorted(_family_heat.items(), key=lambda kv: kv[1], reverse=True)[:20]

    health = compute_health()
    mode = compute_mode()
    ram_p, disk_p, disk_items = compute_pressures()
    adaptive_ram_hint = compute_adaptive_ram_hint()

    summary_60 = summarize_recent_health(window_sec=60.0)
    summary_300 = summarize_recent_health(window_sec=300.0)
    mode_timeline = "".join(h.mode[0].upper() for h in list(health_history)[-20:])

    experiments = []
    for rec in list(recent_experiments)[-10:]:
        experiments.append({
            "time": rec.timestamp,
            "before": rec.before,
            "after": rec.after,
            "health_before": rec.health_before,
            "health_after": rec.health_after,
            "accepted": rec.accepted,
        })

    return {
        "ram_items": ram_cache.size(),
        "disk_items": disk_items,
        "mode": mode,
        "health": health,
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
        "policy": policy.to_dict(),
        "config": {
            "CACHE_DIR": str(CACHE_DIR),
            "HOST": HOST,
            "PORT": PORT,
            "AUTO_TUNE_ENABLED": AUTO_TUNE_ENABLED,
            "ADAPTIVE_RAM_HINT": adaptive_ram_hint,
            "GOSSIP_ENABLED": GOSSIP_ENABLED,
            "GOSSIP_TARGET_URL": GOSSIP_TARGET_URL,
        },
        "hot_keys": [
            {"key": k, "heat": heat, "hits": hits, "last_access": last_access}
            for k, heat, hits, last_access in hot_keys
        ],
        "transitions": transitions_summary,
        "families": [{"family": fam, "heat": heat} for fam, heat in families],
        "experiments": experiments,
        "health_summary": {
            "last_60s": summary_60,
            "last_300s": summary_300,
            "mode_timeline": mode_timeline,
        },
    }

# -----------------------------
# Server thread
# -----------------------------

def start_server_in_thread():
    def run():
        print(f"[CacheServer] Starting on {HOST}:{PORT}")
        print(f"[CacheServer] Cache dir: {CACHE_DIR}")
        print(f"[CacheServer] Initial policy: {policy.to_dict()}")
        print(f"[CacheServer] AUTO_TUNE_ENABLED: {AUTO_TUNE_ENABLED}")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# -----------------------------
# GUI
# -----------------------------

class CacheGUI:
    def __init__(self, root, port: int):
        self.root = root
        self.port = port
        self.client = httpx.Client(timeout=2.0)

        root.title("Autonomous Predictive Cache Control Panel")
        root.geometry("1000x700")

        self._build_controls()
        self._build_stats()
        self._schedule_refresh()

    def _build_controls(self):
        frame = ttk.LabelFrame(self.root, text="Controls")
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.prefetch_var = tk.BooleanVar(value=policy.prefetch_enabled)
        chk_prefetch = ttk.Checkbutton(
            frame, text="Prefetch Enabled", variable=self.prefetch_var,
            command=self.on_toggle_prefetch
        )
        chk_prefetch.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.autotune_var = tk.BooleanVar(value=AUTO_TUNE_ENABLED)
        chk_autotune = ttk.Checkbutton(
            frame, text="Auto-Tune Enabled", variable=self.autotune_var,
            command=self.on_toggle_autotune
        )
        chk_autotune.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(frame, text="Max RAM Items:").grid(row=0, column=2, sticky="e")
        self.ram_spin = ttk.Spinbox(frame, from_=100, to=2000000, increment=100,
                                    width=10, command=self.on_change_ram)
        self.ram_spin.delete(0, tk.END)
        self.ram_spin.insert(0, str(policy.max_ram_items))
        self.ram_spin.grid(row=0, column=3, padx=5)

        ttk.Label(frame, text="Base TTL (s):").grid(row=0, column=4, sticky="e")
        self.ttl_spin = ttk.Spinbox(frame, from_=30, to=86400, increment=30,
                                    width=10, command=self.on_change_ttl)
        self.ttl_spin.delete(0, tk.END)
        self.ttl_spin.insert(0, str(policy.base_ttl))
        self.ttl_spin.grid(row=0, column=5, padx=5)

        ttk.Label(frame, text="Heat Half-life (s):").grid(row=1, column=0, sticky="e")
        self.half_spin = ttk.Spinbox(frame, from_=30, to=3600, increment=10,
                                     width=10, command=self.on_change_half_life)
        self.half_spin.delete(0, tk.END)
        self.half_spin.insert(0, str(int(policy.heat_half_life)))
        self.half_spin.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame, text="Prefetch Fanout:").grid(row=1, column=2, sticky="e")
        self.fanout_spin = ttk.Spinbox(frame, from_=0, to=20, increment=1,
                                       width=10, command=self.on_change_fanout)
        self.fanout_spin.delete(0, tk.END)
        self.fanout_spin.insert(0, str(policy.prefetch_fanout))
        self.fanout_spin.grid(row=1, column=3, padx=5, pady=2)

        btn_clear = ttk.Button(frame, text="Clear RAM+Disk", command=self.clear_cache)
        btn_clear.grid(row=1, column=4, padx=5, pady=2)

    def _build_stats(self):
        stats_frame = ttk.LabelFrame(self.root, text="Stats & Health")
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.stats_text = tk.Text(stats_frame, height=12, wrap="none")
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        hot_frame = ttk.LabelFrame(bottom_frame, text="Hot Keys")
        hot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.hot_list = tk.Listbox(hot_frame)
        self.hot_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fam_frame = ttk.LabelFrame(bottom_frame, text="Families")
        fam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fam_list = tk.Listbox(fam_frame)
        self.fam_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        exp_frame = ttk.LabelFrame(bottom_frame, text="Recent Experiments")
        exp_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.exp_list = tk.Listbox(exp_frame)
        self.exp_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def on_toggle_prefetch(self):
        policy.prefetch_enabled = bool(self.prefetch_var.get())
        messagebox.showinfo("Prefetch", f"Prefetch is now {'ENABLED' if policy.prefetch_enabled else 'DISABLED'}.")

    def on_toggle_autotune(self):
        global AUTO_TUNE_ENABLED
        AUTO_TUNE_ENABLED = bool(self.autotune_var.get())
        messagebox.showinfo("Auto-Tune", f"Auto-Tune is now {'ENABLED' if AUTO_TUNE_ENABLED else 'DISABLED'}.")

    def on_change_ram(self):
        try:
            val = int(self.ram_spin.get())
            if val > 0:
                policy.max_ram_items = val
                ram_cache.max_items = val
        except ValueError:
            pass

    def on_change_ttl(self):
        try:
            val = int(self.ttl_spin.get())
            if val > 0:
                policy.base_ttl = val
        except ValueError:
            pass

    def on_change_half_life(self):
        try:
            val = float(self.half_spin.get())
            if val > 0:
                policy.heat_half_life = val
        except ValueError:
            pass

    def on_change_fanout(self):
        try:
            val = int(self.fanout_spin.get())
            if val >= 0:
                policy.prefetch_fanout = val
        except ValueError:
            pass

    def clear_cache(self):
        try:
            resp = self.client.post(f"http://127.0.0.1:{self.port}/clear")
            if resp.status_code == 200:
                messagebox.showinfo("Clear", "Cache cleared (RAM + Disk).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {e}")

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
        policy_info = data.get("policy", {})
        cfg = data.get("config", {})
        met = data.get("metrics", {})
        pressure = data.get("pressure", {})
        mode = data.get("mode", "unknown")
        health = data.get("health", 0.0)
        health_summary = data.get("health_summary", {})
        last_60 = health_summary.get("last_60s") or {}
        mode_timeline = health_summary.get("mode_timeline") or ""

        ram_p = pressure.get("ram_pressure", 0.0)
        disk_p = pressure.get("disk_pressure", 0.0)
        ram_pct = int(ram_p * 100)
        disk_pct = int(min(disk_p, 1.0) * 100)
        adaptive_hint = cfg.get("ADAPTIVE_RAM_HINT", policy.max_ram_items)

        lines = []
        lines.append(f"Mode: {mode.upper()}    Health: {health:.3f}")
        lines.append(f"RAM items: {data.get('ram_items')} / {policy_info.get('max_ram_items')} (Pressure: {ram_pct}%)")
        lines.append(f"Disk items: {data.get('disk_items')} (Pressure: {disk_pct}%)")
        lines.append(f"Hit ratio: {met.get('hit_ratio', 0):.3f}")
        lines.append(f"Avg latency: {met.get('avg_latency_ms', 0):.2f} ms")
        lines.append(f"Total GET: {met.get('total_get', 0)}, SET: {met.get('total_set', 0)}")
        lines.append(f"Adaptive RAM hint: {adaptive_hint}")

        lines.append("")
        lines.append("Recent (60s):")
        lines.append(f"  avg_health={last_60.get('avg_health', 0):.3f} "
                     f"hit={last_60.get('avg_hit_ratio', 0):.3f} "
                     f"ramP={last_60.get('avg_ram_pressure', 0):.2f} "
                     f"diskP={last_60.get('avg_disk_pressure', 0):.2f} "
                     f"dom_mode={last_60.get('dominant_mode', '')}")

        if mode_timeline:
            lines.append(f"Mode timeline (recent): {mode_timeline}")

        lines.append("")
        lines.append("Policy:")
        for k, v in policy_info.items():
            lines.append(f"  {k}: {v}")
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

        self.exp_list.delete(0, tk.END)
        for rec in data.get("experiments", []):
            ts = time.strftime("%H:%M:%S", time.localtime(rec["time"]))
            status = "ACCEPT" if rec["accepted"] else "REJECT"
            dh = rec["health_after"] - rec["health_before"]
            self.exp_list.insert(
                tk.END,
                f"{ts} [{status}] Δhealth={dh:.3f} "
                f"fanout {rec['before']['prefetch_fanout']}→{rec['after']['prefetch_fanout']} "
                f"ttl {rec['before']['base_ttl']}→{rec['after']['base_ttl']}"
            )

# -----------------------------
# Entry point
# -----------------------------

def main():
    start_server_in_thread()
    root = tk.Tk()
    CacheGUI(root, PORT)
    root.mainloop()


if __name__ == "__main__":
    main()

