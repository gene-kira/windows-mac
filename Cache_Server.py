#!/usr/bin/env python
"""
Simple multi-tier caching server (RAM + Disk) with auto-loader.

Features:
- Auto-installs required libraries (fastapi, uvicorn, pydantic) if missing
- RAM tier: LRU cache with optional TTL and max item limit
- Disk tier: JSON files per key in a cache directory
- HTTP API:
    GET    /get?key=...
    POST   /set  (JSON: {key, value, ttl?})
    DELETE /delete?key=...
    POST   /clear
    GET    /stats
- Config via environment variables:
    CACHE_DIR        (default: ./disk_cache)
    MAX_RAM_ITEMS    (default: 1000)
    DEFAULT_TTL      (default: None = no expiration)
    HOST             (default: 0.0.0.0)
    PORT             (default: 8000)

Run:
    python cache_server.py
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

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

# -----------------------------
# Config
# -----------------------------

# Cache directory for disk tier
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./disk_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Max items to keep in RAM cache
try:
    MAX_RAM_ITEMS = int(os.environ.get("MAX_RAM_ITEMS", "1000"))
except ValueError:
    MAX_RAM_ITEMS = 1000

# Global default TTL (seconds). If None, no expiration unless per-key TTL is set.
_default_ttl_env = os.environ.get("DEFAULT_TTL", "").strip()
TTL_SECONDS: Optional[int] = int(_default_ttl_env) if _default_ttl_env else None

# Bind address and port
HOST = os.environ.get("HOST", "0.0.0.0")
try:
    PORT = int(os.environ.get("PORT", "8000"))
except ValueError:
    PORT = 8000

# -----------------------------
# RAM Tier (LRU cache)
# -----------------------------


class LRUCache:
    """
    Simple LRU cache with optional TTL per item.
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
            # Expired: remove and treat as miss
            del self._store[key]
            return None
        # Update last access time for LRU behavior
        self._store[key] = (value, self._now(), expires_at)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = self._now() + ttl if ttl is not None else None
        self._store[key] = (value, self._now(), expires_at)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self._store) <= self.max_items:
            return
        # Evict least recently used entries until under limit
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


ram_cache = LRUCache(MAX_RAM_ITEMS)

# -----------------------------
# Disk Tier
# -----------------------------


def _key_to_filename(key: str) -> Path:
    """
    Map arbitrary key string to a filename-safe path using SHA-256.
    """
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def disk_get(key: str) -> Optional[Any]:
    path = _key_to_filename(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        expires_at = data.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            # Expired on disk, remove file
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
        return data.get("value")
    except Exception as e:
        print(f"[DiskTier] Error reading {path}: {e}")
        return None


def disk_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    expires_at = time.time() + ttl if ttl is not None else None
    path = _key_to_filename(key)
    tmp = path.with_suffix(".tmp")
    data = {"value": value, "expires_at": expires_at}
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception as e:
        print(f"[DiskTier] Error writing {path}: {e}")


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

app = FastAPI(title="Simple Multi-tier Cache Server", version="1.0.0")


@app.get("/get")
def api_get(key: str):
    """
    Get a value from cache.
    Search order: RAM -> Disk.
    If found on disk, promote to RAM.
    """
    # 1. RAM
    value = ram_cache.get(key)
    if value is not None:
        return {"hit": True, "tier": "ram", "key": key, "value": value}

    # 2. Disk
    value = disk_get(key)
    if value is not None:
        # Promote to RAM using default TTL
        ram_cache.set(key, value, ttl=TTL_SECONDS)
        return {"hit": True, "tier": "disk", "key": key, "value": value}

    # 3. Miss
    return {"hit": False, "tier": None, "key": key, "value": None}


@app.post("/set")
def api_set(req: SetRequest):
    """
    Set a value in cache (write-through: RAM + Disk).
    TTL priority: request.ttl > DEFAULT_TTL > None.
    """
    effective_ttl = req.ttl if req.ttl is not None else TTL_SECONDS
    ram_cache.set(req.key, req.value, ttl=effective_ttl)
    disk_set(req.key, req.value, ttl=effective_ttl)
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
    return {"ok": True}


@app.get("/stats")
def api_stats():
    """
    Basic stats: RAM size, Disk size, config.
    """
    return {
        "ram_items": ram_cache.size(),
        "disk_items": disk_size(),
        "config": {
            "CACHE_DIR": str(CACHE_DIR),
            "MAX_RAM_ITEMS": MAX_RAM_ITEMS,
            "DEFAULT_TTL": TTL_SECONDS,
            "HOST": HOST,
            "PORT": PORT,
        },
    }


# -----------------------------
# Entry point
# -----------------------------


def main():
    print(f"[CacheServer] Starting on {HOST}:{PORT}")
    print(f"[CacheServer] Cache dir: {CACHE_DIR}")
    print(f"[CacheServer] MAX_RAM_ITEMS: {MAX_RAM_ITEMS}, DEFAULT_TTL: {TTL_SECONDS}")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

