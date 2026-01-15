#!/usr/bin/env python3
"""
Borg Backbone Orchestrator – Full Integrated Organism (with Back4Blood game phase + dynamic coach)

Features:
- Workers + Main Brain
- Foresight Engine (RAM/CPU prediction)
- CPU / APU / dGPU lanes (no slow-in-fast-lane routing)
- System-utilities GPU hook (backbone tap-in point)
- Transport Data subsystem:
    * Tracks transfer size / rate / volume
    * Computes unified Transport Data Load %
- Organs:
    * DiskOrgan (real disk IO)
    * VRAMOrgan (real VRAM if pynvml available)
    * AICoachOrgan (dynamic stance based on game phase)
    * SwarmNodeOrgan
    * Back4BloodAnalyzer (system + organs + game telemetry + phase)
- BrainCortexPanel:
    * Health / Risk / Meta-state / Stance thresholds
    * Deep channels: RAM / Backup / Network / GPU / Thermal / Disk / VRAM / Swarm / Coach
- Sci‑Fi Cortex GUI:
    * WANT – live demand
    * PLAN – foresight + routing
    * PRELOADED – actions already in motion
    * TRANSPORT DATA – unified flow percentage
    * Improvement Meter – foresight benefit estimate
- Snapshot save/load (local + network)
"""

import sys
import os
import json
import time
import threading
from datetime import datetime
import hashlib
import platform
from collections import deque

# ---------- Autoloader ----------

REQUIRED_LIBS = ["tkinter", "psutil"]

missing = []
for lib in REQUIRED_LIBS:
    try:
        __import__(lib)
    except ImportError:
        missing.append(lib)

if missing:
    print("Missing required libraries:", ", ".join(missing))
    print("Install them, for example:")
    print("  python -m pip install " + " ".join(missing))
    sys.exit(1)

import tkinter as tk
from tkinter import messagebox, filedialog
import psutil

# GPU (NVIDIA) via NVML (one possible backbone tap)
GPU_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ---------- Config ----------

LOCAL_SAVE_DIR = os.path.expanduser(r"borg_backbone_local")
NETWORK_SAVE_DIR = os.path.expanduser(r"\\NETWORK\\borg_backbone")  # change to your real UNC/path
LATEST_INDEX_FILE = "latest_snapshot.json"
UPDATE_INTERVAL_SEC = 0.5

FORESIGHT_WINDOW = 60
FORESIGHT_HORIZON = 10
FORESIGHT_MIN_CONFIDENCE = 0.6

TRANSPORT_WINDOW_SEC = 10.0  # rolling window for transport stats


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------- Backbone state ----------

BACKBONE_STATE = {
    "meta": {},
    "system": {},
    "gpu": {},
    "caches": {},
    "processes": [],
    "history": [],
    "workloads": {},
    "actions": [],
    "workers": {
        "cpu_worker": {},
        "gpu_worker": {},
        "process_workers": [],
    },
    "policy_decisions": {},
    "ai_devices": [],
    "ai_routing": {},
    "foresight": {
        "ram_series": deque(maxlen=FORESIGHT_WINDOW),
        "cpu_series": deque(maxlen=FORESIGHT_WINDOW),
        "last_prediction": None,
    },
    "transport": {
        "events": deque(),  # (timestamp, size_bytes)
        "stats": {
            "avg_size": 0.0,
            "avg_rate": 0.0,
            "avg_volume": 0.0,
            "load_percent": 0.0,
            "min_size": None,
            "max_size": None,
            "min_rate": None,
            "max_rate": None,
            "min_volume": None,
            "max_volume": None,
        },
    },
    "organs": {
        "DiskOrgan": {},
        "VRAMOrgan": {},
        "AICoachOrgan": {},
        "SwarmNodeOrgan": {},
        "Back4BloodAnalyzer": {},
    },
    "brain_cortex": {
        "health": {},
        "meta_conf": {},
        "model_integrity": {},
        "meta_state": {},
        "stance_thresholds": {},
        "deep_channels": {
            "ram": {},
            "backup": {},
            "network": {},
            "gpu": {},
            "thermal": {},
            "disk": {},
            "vram": {},
            "swarm": {},
            "coach": {},
        },
    },
}


def compute_checksum(data_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(data_bytes)
    return h.hexdigest()


def atomic_write(path: str, data_bytes: bytes):
    directory = os.path.dirname(path)
    ensure_dir(directory)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------- Workloads + Actions + Transport ----------

def transport_log_transfer(size_bytes: int):
    now = time.time()
    ev_q = BACKBONE_STATE["transport"]["events"]
    ev_q.append((now, float(size_bytes)))
    cutoff = now - TRANSPORT_WINDOW_SEC
    while ev_q and ev_q[0][0] < cutoff:
        ev_q.popleft()


def transport_update_stats():
    ev_q = BACKBONE_STATE["transport"]["events"]
    stats = BACKBONE_STATE["transport"]["stats"]

    if not ev_q:
        stats["avg_size"] = 0.0
        stats["avg_rate"] = 0.0
        stats["avg_volume"] = 0.0
        stats["load_percent"] = 0.0
        return

    now = time.time()
    window_start = ev_q[0][0]
    window_duration = max(0.001, now - window_start)

    sizes = [s for (_, s) in ev_q]
    total_size = sum(sizes)
    count = len(sizes)

    avg_size = total_size / count
    avg_rate = total_size / window_duration  # bytes/sec
    avg_volume = total_size / TRANSPORT_WINDOW_SEC  # bytes/sec normalized to fixed window

    def update_min_max(key_min, key_max, value):
        if value <= 0:
            return
        if stats[key_min] is None or value < stats[key_min]:
            stats[key_min] = value
        if stats[key_max] is None or value > stats[key_max]:
            stats[key_max] = value

    update_min_max("min_size", "max_size", avg_size)
    update_min_max("min_rate", "max_rate", avg_rate)
    update_min_max("min_volume", "max_volume", avg_volume)

    def normalize(value, vmin, vmax):
        if value <= 0 or vmin is None or vmax is None or vmax <= vmin:
            return 0.0
        return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

    S = normalize(avg_size, stats["min_size"], stats["max_size"])
    R = normalize(avg_rate, stats["min_rate"], stats["max_rate"])
    V = normalize(avg_volume, stats["min_volume"], stats["max_volume"])

    load_percent = (S + R + V) / 3.0 * 100.0

    stats["avg_size"] = avg_size
    stats["avg_rate"] = avg_rate
    stats["avg_volume"] = avg_volume
    stats["load_percent"] = max(0.0, min(100.0, load_percent))


def queue_action(kind, payload):
    BACKBONE_STATE["actions"].append({
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        "payload": payload,
    })
    try:
        size_bytes = len(json.dumps(payload).encode("utf-8"))
    except Exception:
        size_bytes = 0
    if size_bytes > 0:
        transport_log_transfer(size_bytes)


def init_example_workloads():
    BACKBONE_STATE["workloads"] = {
        "ExampleWorkload": {
            "pid": None,
            "type": "demo",
            "hot_assets": ["asset_A", "asset_B"],
            "vram_budget_bytes": 2 * 1024**3,
            "lane": "fast",
        }
    }


# ---------- Live collectors ----------

def collect_live_system_state():
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_freq = psutil.cpu_freq()
    cpu_stats = psutil.cpu_stats()

    BACKBONE_STATE["system"] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "ram": {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        },
        "cpu": {
            "percent": cpu_percent,
            "freq_current": cpu_freq.current if cpu_freq else None,
            "freq_min": cpu_freq.min if cpu_freq else None,
            "freq_max": cpu_freq.max if cpu_freq else None,
            "ctx_switches": cpu_stats.ctx_switches,
            "interrupts": cpu_stats.interrupts,
            "soft_interrupts": cpu_stats.soft_interrupts,
            "syscalls": cpu_stats.syscalls,
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
        },
    }


# ---------- System utilities backbone GPU hook ----------

def _gpu_from_system_utilities():
    """
    Backbone tap-in point.

    Wire this to your real system utilities / vendor APIs.
    Return a list of dicts:
      [
        {
          "index": 0,
          "name": "GPU Name",
          "memory_total": <bytes>,
          "memory_used": <bytes>,
        },
        ...
      ]
    """
    return []


def collect_live_gpu_state():
    gpus = []
    reason = None

    if GPU_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8", errors="ignore")
                gpus.append({
                    "index": i,
                    "name": name,
                    "memory_total": mem_info.total,
                    "memory_used": mem_info.used,
                    "memory_free": mem_info.free,
                    "memory_used_percent": (mem_info.used / mem_info.total * 100.0) if mem_info.total else None,
                })
        except Exception as e:
            reason = f"NVML query failed: {e}"

    if not gpus:
        try:
            util_gpus = _gpu_from_system_utilities()
            for i, g in enumerate(util_gpus):
                total = g.get("memory_total") or 0
                used = g.get("memory_used") or 0
                gpus.append({
                    "index": g.get("index", i),
                    "name": g.get("name", f"GPU_{i}"),
                    "memory_total": total,
                    "memory_used": used,
                    "memory_free": max(0, total - used),
                    "memory_used_percent": (used / total * 100.0) if total else None,
                })
        except Exception as e:
            reason = f"System utilities GPU hook failed: {e}"

    if gpus:
        BACKBONE_STATE["gpu"] = {
            "available": True,
            "gpus": gpus,
        }
    else:
        BACKBONE_STATE["gpu"] = {
            "available": False,
            "reason": reason or "no GPU info from NVML or system utilities backbone",
        }


def detect_cache_topology():
    info = {
        "note": "Cache sizes are static topology; live fill level is not exposed by hardware to user space.",
        "l1": None,
        "l2": None,
        "l3": None,
        "raw": {},
    }

    try:
        if sys.platform.startswith("linux"):
            base = "/sys/devices/system/cpu/cpu0/cache"
            levels = {}
            if os.path.isdir(base):
                for idx in os.listdir(base):
                    idx_path = os.path.join(base, idx)
                    if not os.path.isdir(idx_path):
                        continue
                    try:
                        with open(os.path.join(idx_path, "level")) as f:
                            level = f.read().strip()
                        with open(os.path.join(idx_path, "size")) as f:
                            size_str = f.read().strip().lower()
                        if size_str.endswith("k"):
                            size_bytes = int(size_str[:-1]) * 1024
                        elif size_str.endswith("m"):
                            size_bytes = int(size_str[:-1]) * 1024 * 1024
                        else:
                            size_bytes = int(size_str)
                        levels[level] = size_bytes
                    except Exception:
                        continue
                info["l1"] = levels.get("1")
                info["l2"] = levels.get("2")
                info["l3"] = levels.get("3")
                info["raw"] = levels
        else:
            info["raw"] = {"platform": platform.processor()}
    except Exception as e:
        info["raw"] = {"error": str(e)}

    BACKBONE_STATE["caches"] = info


def collect_process_list():
    procs = []
    for p in psutil.process_iter(attrs=["pid", "name", "username", "memory_info", "cpu_percent"]):
        try:
            info = p.info
            mem = info.get("memory_info")
            procs.append({
                "pid": info.get("pid"),
                "name": info.get("name"),
                "user": info.get("username"),
                "rss": mem.rss if mem else None,
                "cpu_percent": info.get("cpu_percent"),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x.get("rss") or 0, reverse=True)
    BACKBONE_STATE["processes"] = procs[:50]


def update_backbone_meta():
    BACKBONE_STATE["meta"] = {
        "version": "1.1_borg_backbone_full_transport_b4b",
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "gpu_support": BACKBONE_STATE.get("gpu", {}).get("available", False),
    }


# ---------- AI devices / lanes (CPU + APU + dGPU) ----------

def discover_ai_devices():
    devices = []

    sys_state = BACKBONE_STATE.get("system", {})
    cpu = sys_state.get("cpu", {})
    devices.append({
        "type": "CPU",
        "name": platform.processor() or "CPU",
        "index": 0,
        "speed_rank": 1,
        "cores_logical": cpu.get("cores_logical"),
        "cores_physical": cpu.get("cores_physical"),
    })

    devices.append({
        "type": "APU",
        "name": "Integrated GPU / APU",
        "index": 0,
        "speed_rank": 2,
        "vram_total": None,
        "vram_used": None,
    })

    gpu_state = BACKBONE_STATE.get("gpu", {})
    if gpu_state.get("available"):
        for g in gpu_state.get("gpus", []):
            devices.append({
                "type": "dGPU",
                "name": g["name"],
                "index": g["index"],
                "speed_rank": 3,
                "vram_total": g["memory_total"],
                "vram_used": g["memory_used"],
            })

    BACKBONE_STATE["ai_devices"] = devices


# ---------- Borg workers ----------

def update_cpu_worker():
    sys_state = BACKBONE_STATE.get("system", {})
    cpu = sys_state.get("cpu", {})
    ram = sys_state.get("ram", {})
    BACKBONE_STATE["workers"]["cpu_worker"] = {
        "timestamp_utc": sys_state.get("timestamp_utc"),
        "cpu_percent": cpu.get("percent", 0),
        "freq_current": cpu.get("freq_current"),
        "freq_max": cpu.get("freq_max"),
        "ram_percent": ram.get("percent", 0),
    }


def update_gpu_worker():
    gpu_state = BACKBONE_STATE.get("gpu", {})
    if not gpu_state.get("available"):
        BACKBONE_STATE["workers"]["gpu_worker"] = {
            "available": False,
            "reason": gpu_state.get("reason", "not available"),
        }
        return

    gpus = gpu_state.get("gpus", [])
    if not gpus:
        BACKBONE_STATE["workers"]["gpu_worker"] = {
            "available": False,
            "reason": "no devices",
        }
        return

    g = gpus[0]
    BACKBONE_STATE["workers"]["gpu_worker"] = {
        "available": True,
        "name": g["name"],
        "index": g["index"],
        "vram_total": g["memory_total"],
        "vram_used": g["memory_used"],
        "vram_used_percent": g["memory_used_percent"],
    }


def update_process_workers():
    workers = []
    for p in BACKBONE_STATE.get("processes", []):
        workers.append({
            "pid": p["pid"],
            "name": p["name"],
            "rss": p.get("rss") or 0,
            "cpu_percent": p.get("cpu_percent") or 0,
        })
    BACKBONE_STATE["workers"]["process_workers"] = workers


# ---------- Foresight engine ----------

def foresight_update_series():
    sys_state = BACKBONE_STATE.get("system", {})
    ram = sys_state.get("ram", {})
    cpu = sys_state.get("cpu", {})

    ram_percent = ram.get("percent", 0.0)
    cpu_percent = cpu.get("percent", 0.0)

    fs = BACKBONE_STATE["foresight"]
    fs["ram_series"].append(ram_percent)
    fs["cpu_series"].append(cpu_percent)


def simple_trend_predict(series, horizon):
    if len(series) < 5:
        return None, 0.0

    values = list(series)
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = num / den
    intercept = mean_y - slope * mean_x

    future_x = n - 1 + horizon
    pred = intercept + slope * future_x

    diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    slope_mag = abs(slope)
    stability = max(0.0, 1.0 - min(1.0, avg_diff / 20.0))
    conf = max(0.0, min(1.0, slope_mag / 2.0)) * stability

    return pred, conf


def foresight_engine():
    fs = BACKBONE_STATE["foresight"]
    ram_pred, ram_conf = simple_trend_predict(fs["ram_series"], FORESIGHT_HORIZON)
    cpu_pred, cpu_conf = simple_trend_predict(fs["cpu_series"], FORESIGHT_HORIZON)

    prediction = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "ram_pred": ram_pred,
        "ram_conf": ram_conf,
        "cpu_pred": cpu_pred,
        "cpu_conf": cpu_conf,
    }
    fs["last_prediction"] = prediction

    workloads = BACKBONE_STATE.get("workloads", {})
    gpu_w = BACKBONE_STATE["workers"].get("gpu_worker", {})
    gpu_available = gpu_w.get("available", False)

    if gpu_available and ram_pred is not None and ram_conf >= FORESIGHT_MIN_CONFIDENCE and ram_pred > 70.0:
        for w_name, w in workloads.items():
            budget = w.get("vram_budget_bytes", 0)
            hot_assets = w.get("hot_assets", [])
            if not hot_assets or budget <= 0:
                continue
            queue_action("PRELOAD_VRAM_FORESIGHT", {
                "workload": w_name,
                "pid": w.get("pid"),
                "assets": hot_assets,
                "vram_budget_bytes": budget,
                "reason": f"Foresight: RAM predicted {ram_pred:.1f}% (conf {ram_conf:.2f}), preloading to VRAM before spike",
            })

    if cpu_pred is not None and cpu_conf >= FORESIGHT_MIN_CONFIDENCE and cpu_pred > 80.0:
        queue_action("REQUEST_CPU_BOOST", {
            "reason": f"Foresight: CPU predicted {cpu_pred:.1f}% (conf {cpu_conf:.2f}), request higher clocks",
        })
        queue_action("USE_ALL_CORES_THREADS", {
            "reason": "Foresight: high CPU predicted, spread workload across all cores/threads",
        })


# ---------- Borg main brain ----------

def _lane_rank_for_workload(workload_lane: str) -> int:
    if workload_lane == "fast":
        return 3
    if workload_lane == "medium":
        return 2
    return 1


def borg_main_brain():
    cpu_w = BACKBONE_STATE["workers"].get("cpu_worker", {})
    gpu_w = BACKBONE_STATE["workers"].get("gpu_worker", {})
    proc_ws = BACKBONE_STATE["workers"].get("process_workers", [])
    workloads = BACKBONE_STATE.get("workloads", {})
    ai_devices = BACKBONE_STATE.get("ai_devices", [])

    ram_percent = cpu_w.get("ram_percent", 0)
    cpu_percent = cpu_w.get("cpu_percent", 0)
    gpu_available = gpu_w.get("available", False)

    decisions = []
    total_rss = sum(w.get("rss", 0) for w in proc_ws) or 1
    for w in proc_ws:
        rss_share = (w["rss"] / total_rss) * 100.0
        score = rss_share + w["cpu_percent"]
        decisions.append({
            "pid": w["pid"],
            "name": w["name"],
            "score": score,
            "reason": "Borg worker: RSS share + CPU",
        })
    decisions.sort(key=lambda d: d["score"], reverse=True)
    top_decisions = decisions[:10]

    routing = {}
    for w_name, w in workloads.items():
        desired_lane = w.get("lane", "fast")
        desired_rank = _lane_rank_for_workload(desired_lane)

        candidates = [d for d in ai_devices if d.get("speed_rank", 1) >= desired_rank]
        if not candidates:
            candidates = ai_devices
        if not candidates:
            continue

        d_gpu_like = [d for d in candidates if d["type"] in ("dGPU", "APU")]
        if d_gpu_like:
            def free_vram(d):
                total = d.get("vram_total") or 0
                used = d.get("vram_used") or 0
                return total - used
            best = max(d_gpu_like, key=free_vram)
        else:
            cpu_devices = [d for d in candidates if d["type"] == "CPU"]
            best = cpu_devices[0] if cpu_devices else candidates[0]

        routing[w_name] = {
            "device_type": best["type"],
            "device_name": best["name"],
            "device_index": best["index"],
            "speed_rank": best["speed_rank"],
            "desired_lane": desired_lane,
            "reason": "Lane-aware routing: no slow-in-fast-lane; picked fastest compatible device",
        }

    BACKBONE_STATE["ai_routing"] = routing

    if gpu_available and ram_percent > 60.0:
        for w_name, w in workloads.items():
            budget = w.get("vram_budget_bytes", 0)
            hot_assets = w.get("hot_assets", [])
            if not hot_assets or budget <= 0:
                continue
            queue_action("PRELOAD_VRAM", {
                "workload": w_name,
                "pid": w.get("pid"),
                "assets": hot_assets,
                "vram_budget_bytes": budget,
                "reason": "Borg main brain: high RAM, route hot assets to VRAM",
            })

    if cpu_percent > 80.0:
        queue_action("REQUEST_CPU_BOOST", {
            "reason": f"Main brain: CPU at {cpu_percent:.1f}%, request higher clocks",
        })
        queue_action("USE_ALL_CORES_THREADS", {
            "reason": "Main brain: high CPU, use all cores/threads for smoothness",
        })

    BACKBONE_STATE["policy_decisions"] = {
        "global_state": {
            "ram_percent": ram_percent,
            "cpu_percent": cpu_percent,
            "gpu_available": gpu_available,
        },
        "top_processes": top_decisions,
        "foresight": BACKBONE_STATE["foresight"].get("last_prediction"),
    }


# ---------- Organs ----------

def update_disk_organ():
    disk = {}
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except PermissionError:
            continue
        disk[part.mountpoint] = {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent,
        }
    BACKBONE_STATE["organs"]["DiskOrgan"] = disk
    BACKBONE_STATE["brain_cortex"]["deep_channels"]["disk"] = disk


def update_vram_organ():
    gpu_state = BACKBONE_STATE.get("gpu", {})
    if not gpu_state.get("available"):
        BACKBONE_STATE["organs"]["VRAMOrgan"] = {"available": False, "reason": gpu_state.get("reason")}
        BACKBONE_STATE["brain_cortex"]["deep_channels"]["vram"] = BACKBONE_STATE["organs"]["VRAMOrgan"]
        return

    vram = {}
    for g in gpu_state.get("gpus", []):
        vram[g["index"]] = {
            "name": g["name"],
            "total": g["memory_total"],
            "used": g["memory_used"],
            "percent": g["memory_used_percent"],
        }
    BACKBONE_STATE["organs"]["VRAMOrgan"] = {"available": True, "devices": vram}
    BACKBONE_STATE["brain_cortex"]["deep_channels"]["vram"] = BACKBONE_STATE["organs"]["VRAMOrgan"]


def update_swarm_node_organ():
    swarm = {
        "node_id": BACKBONE_STATE["meta"].get("host", "unknown"),
        "status": "STANDALONE",
        "peers": [],
    }
    BACKBONE_STATE["organs"]["SwarmNodeOrgan"] = swarm
    BACKBONE_STATE["brain_cortex"]["deep_channels"]["swarm"] = swarm


# ---------- Back 4 Blood Telemetry + Analyzer + Dynamic Coach ----------

def read_back4blood_telemetry():
    """
    Stub for real Back 4 Blood telemetry.
    Replace with actual hooks (memory reader, overlay, log parser, etc.).
    """
    return {
        "player_hp": 85,
        "team_hp_avg": 72,
        "ammo": 140,
        "threat": 0.2,
        "specials": 1,
        "horde_active": False,
        "panic": False,
    }


def detect_game_phase(t):
    if t["panic"] or t["player_hp"] < 25:
        return "PANIC"
    if t["horde_active"] or t["specials"] >= 3 or t["threat"] > 0.7:
        return "HORDE"
    if t["threat"] > 0.4 or t["specials"] >= 1:
        return "TENSION"
    if t["player_hp"] > 80 and t["team_hp_avg"] > 70:
        return "CALM"
    return "RECOVERY"


def update_back4blood_analyzer():
    telemetry = read_back4blood_telemetry()
    phase = detect_game_phase(telemetry)

    analyzer = {
        "system": BACKBONE_STATE.get("system", {}),
        "transport": BACKBONE_STATE.get("transport", {}),
        "organs": BACKBONE_STATE.get("organs", {}),
        "telemetry": telemetry,
        "phase": phase,
    }

    BACKBONE_STATE["organs"]["Back4BloodAnalyzer"] = analyzer
    BACKBONE_STATE["brain_cortex"]["meta_state"] = {"game_phase": phase}


def update_ai_coach_organ():
    analyzer = BACKBONE_STATE["organs"].get("Back4BloodAnalyzer", {})
    phase = analyzer.get("phase", "CALM")

    thresholds = {
        "CALM":      {"cpu": 50, "ram": 60, "transport": 40, "stance": "CALM"},
        "TENSION":   {"cpu": 65, "ram": 70, "transport": 55, "stance": "ALERT"},
        "HORDE":     {"cpu": 80, "ram": 85, "transport": 70, "stance": "AGGRESSIVE"},
        "PANIC":     {"cpu": 90, "ram": 90, "transport": 85, "stance": "EMERGENCY"},
        "RECOVERY":  {"cpu": 55, "ram": 60, "transport": 45, "stance": "STABILIZE"},
    }

    t = thresholds.get(phase, thresholds["CALM"])

    coach = {
        "phase": phase,
        "stance": t["stance"],
        "thresholds": t,
    }

    BACKBONE_STATE["organs"]["AICoachOrgan"] = coach
    BACKBONE_STATE["brain_cortex"]["deep_channels"]["coach"] = coach
    BACKBONE_STATE["brain_cortex"]["stance_thresholds"] = t


# ---------- Collect all live state ----------

def collect_all_live_state():
    collect_live_system_state()
    collect_live_gpu_state()
    if not BACKBONE_STATE["caches"]:
        detect_cache_topology()
    collect_process_list()
    update_backbone_meta()
    discover_ai_devices()

    update_cpu_worker()
    update_gpu_worker()
    update_process_workers()

    foresight_update_series()
    foresight_engine()

    borg_main_brain()

    transport_update_stats()

    update_disk_organ()
    update_vram_organ()
    update_swarm_node_organ()
    update_back4blood_analyzer()
    update_ai_coach_organ()

    sys_state = BACKBONE_STATE.get("system", {})
    BACKBONE_STATE["history"].append({
        "t": sys_state.get("timestamp_utc"),
        "ram_percent": sys_state.get("ram", {}).get("percent"),
        "cpu_percent": sys_state.get("cpu", {}).get("percent"),
    })
    if len(BACKBONE_STATE["history"]) > 200:
        BACKBONE_STATE["history"] = BACKBONE_STATE["history"][-200:]


# ---------- Snapshot + reload ----------

def generate_live_snapshot() -> dict:
    snapshot = json.loads(json.dumps(BACKBONE_STATE))
    snapshot.setdefault("meta", {})
    snapshot["meta"]["snapshot_utc"] = datetime.utcnow().isoformat() + "Z"
    return snapshot


def save_snapshot_to(path_dir: str, snapshot: dict) -> dict:
    ensure_dir(path_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{ts}.json"
    full_path = os.path.join(path_dir, filename)

    data_bytes = json.dumps(snapshot, indent=2, sort_keys=True).encode("utf-8")
    checksum = compute_checksum(data_bytes)

    atomic_write(full_path, data_bytes)

    with open(full_path, "rb") as f:
        verify_bytes = f.read()
    verify_checksum = compute_checksum(verify_bytes)

    if verify_checksum != checksum:
        raise IOError(f"Checksum mismatch for {full_path}")

    return {
        "path": full_path,
        "checksum": checksum,
        "timestamp_utc": snapshot["meta"].get("snapshot_utc"),
    }


def update_latest_index(index_dir: str, local_meta: dict, network_meta: dict):
    ensure_dir(index_dir)
    index_path = os.path.join(index_dir, LATEST_INDEX_FILE)

    index_data = {
        "version": BACKBONE_STATE["meta"].get("version", "1.1_borg_backbone_full_transport_b4b"),
        "updated_utc": datetime.utcnow().isoformat() + "Z",
        "local": local_meta,
        "network": network_meta,
    }

    data_bytes = json.dumps(index_data, indent=2, sort_keys=True).encode("utf-8")
    atomic_write(index_path, data_bytes)


def load_latest_snapshot(base_dir: str):
    index_path = os.path.join(base_dir, LATEST_INDEX_FILE)
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        local_meta = index.get("local")
        if not local_meta:
            return None
        snap_path = local_meta.get("path")
        if not snap_path or not os.path.isfile(snap_path):
            return None
        with open(snap_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        return snapshot
    except Exception:
        return None


def restore_backbone_from_snapshot(snapshot: dict):
    global BACKBONE_STATE
    BACKBONE_STATE = snapshot


# ---------- Sci‑Fi Cortex GUI ----------

class BorgBackboneGUI:
    def __init__(self, master):
        self.master = master
        master.title("Borg Cortex – Full Backbone with Transport + Game Phase")

        self.local_dir_var = tk.StringVar(value=LOCAL_SAVE_DIR)
        self.network_dir_var = tk.StringVar(value=NETWORK_SAVE_DIR)

        top_frame = tk.Frame(master, bg="#050510")
        top_frame.grid(row=0, column=0, columnspan=3, sticky="we")
        tk.Label(top_frame, text="Local save:", fg="#A0A0FF", bg="#050510").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.local_entry = tk.Entry(top_frame, textvariable=self.local_dir_var, width=40, bg="#101020", fg="#E0E0FF")
        self.local_entry.grid(row=0, column=1, padx=5, pady=3)
        tk.Button(top_frame, text="Browse", command=self.browse_local).grid(row=0, column=2, padx=5, pady=3)

        tk.Label(top_frame, text="Network save:", fg="#A0A0FF", bg="#050510").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.network_entry = tk.Entry(top_frame, textvariable=self.network_dir_var, width=40, bg="#101020", fg="#E0E0FF")
        self.network_entry.grid(row=1, column=1, padx=5, pady=3)
        tk.Button(top_frame, text="Browse", command=self.browse_network).grid(row=1, column=2, padx=5, pady=3)

        want_frame = tk.LabelFrame(master, text="WANT – Live System Demand", fg="#FFCC66", bg="#050510")
        want_frame.grid(row=1, column=0, sticky="nwe", padx=5, pady=5)

        self.ram_var = tk.StringVar(value="RAM: --")
        self.cpu_var = tk.StringVar(value="CPU: --")
        self.gpu_var = tk.StringVar(value="GPU: --")
        self.cache_var = tk.StringVar(value="Cache: --")

        tk.Label(want_frame, textvariable=self.ram_var, fg="#FFEEAA", bg="#050510").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Label(want_frame, textvariable=self.cpu_var, fg="#FFEEAA", bg="#050510").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Label(want_frame, textvariable=self.gpu_var, fg="#FFEEAA", bg="#050510").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Label(want_frame, textvariable=self.cache_var, fg="#8888AA", bg="#050510").grid(row=3, column=0, sticky="w", padx=5, pady=2)

        plan_frame = tk.LabelFrame(master, text="PLAN – Borg Foresight & Routing", fg="#66FFCC", bg="#050510")
        plan_frame.grid(row=1, column=1, sticky="nwe", padx=5, pady=5)

        self.plan_list = tk.Listbox(plan_frame, height=10, width=45, bg="#081818", fg="#CCFFEE")
        self.plan_list.grid(row=0, column=0, padx=5, pady=5, sticky="we")

        preload_frame = tk.LabelFrame(master, text="PRELOADED – Actions Already in Motion", fg="#66CCFF", bg="#050510")
        preload_frame.grid(row=1, column=2, sticky="nwe", padx=5, pady=5)

        self.actions_list = tk.Listbox(preload_frame, height=10, width=45, bg="#081020", fg="#CCE6FF")
        self.actions_list.grid(row=0, column=0, padx=5, pady=5, sticky="we")

        tk.Label(master, text="Process field (who is asking for bread):", fg="#CCCCFF", bg="#050510").grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        self.proc_list = tk.Listbox(master, height=8, width=120, bg="#050510", fg="#E0E0FF")
        self.proc_list.grid(row=3, column=0, columnspan=3, sticky="we", padx=5, pady=2)

        meter_frame = tk.LabelFrame(master, text="Improvement Meter – How much smarter than baseline?", fg="#FF99FF", bg="#050510")
        meter_frame.grid(row=4, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        self.improvement_canvas = tk.Canvas(meter_frame, width=400, height=20, bg="#101010", highlightthickness=0)
        self.improvement_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.improvement_label = tk.Label(meter_frame, text="Improvement: -- %", fg="#FFCCFF", bg="#050510")
        self.improvement_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        transport_frame = tk.LabelFrame(master, text="TRANSPORT DATA – Unified Flow Load", fg="#99FFCC", bg="#050510")
        transport_frame.grid(row=5, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        self.transport_canvas = tk.Canvas(transport_frame, width=400, height=20, bg="#101010", highlightthickness=0)
        self.transport_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.transport_label = tk.Label(transport_frame, text="Transport Load: -- %", fg="#CCFFEE", bg="#050510")
        self.transport_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        cortex_frame = tk.LabelFrame(master, text="Brain Cortex Panel", fg="#FFDD88", bg="#050510")
        cortex_frame.grid(row=6, column=0, columnspan=3, sticky="we", padx=5, pady=5)

        self.cortex_health_var = tk.StringVar(value="Health: --")
        self.cortex_meta_var = tk.StringVar(value="Meta-state: --")
        self.cortex_stance_var = tk.StringVar(value="Stance thresholds: --")
        self.cortex_channels_var = tk.StringVar(value="Deep channels: --")

        tk.Label(cortex_frame, textvariable=self.cortex_health_var, fg="#FFEEDD", bg="#050510").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Label(cortex_frame, textvariable=self.cortex_meta_var, fg="#FFEEDD", bg="#050510").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Label(cortex_frame, textvariable=self.cortex_stance_var, fg="#FFEEDD", bg="#050510").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Label(cortex_frame, textvariable=self.cortex_channels_var, fg="#CCFFDD", bg="#050510").grid(row=3, column=0, sticky="w", padx=5, pady=2)

        self.status_var = tk.StringVar(value="Borg cortex with foresight, lanes, transport, and game phase online.")
        tk.Label(master, textvariable=self.status_var, fg="#8888FF", bg="#050510").grid(row=7, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        self.save_button = tk.Button(master, text="SAVE LIVE SNAPSHOT (Local + Network)", command=self.save_snapshot)
        self.save_button.grid(row=8, column=0, columnspan=2, padx=5, pady=10, sticky="we")

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.grid(row=8, column=2, padx=5, pady=10, sticky="we")

        master.configure(bg="#050510")
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)

        self._stop = False
        self.start_live_updates()

    def browse_local(self):
        directory = filedialog.askdirectory()
        if directory:
            self.local_dir_var.set(directory)

    def browse_network(self):
        directory = filedialog.askdirectory()
        if directory:
            self.network_dir_var.set(directory)

    def save_snapshot(self):
        self.status_var.set("Freezing live state and saving snapshot...")
        self.master.update_idletasks()
        try:
            snapshot = generate_live_snapshot()
            local_dir = self.local_dir_var.get()
            network_dir = self.network_dir_var.get()

            local_meta = save_snapshot_to(local_dir, snapshot)
            network_meta = save_snapshot_to(network_dir, snapshot)

            update_latest_index(local_dir, local_meta, network_meta)

            self.status_var.set("Snapshot saved to local + network.")
            messagebox.showinfo("Success", "Live snapshot saved to local and network.")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to save snapshot:\n{e}")

    def start_live_updates(self):
        def loop():
            while not self._stop:
                collect_all_live_state()
                self.update_labels()
                time.sleep(UPDATE_INTERVAL_SEC)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _draw_bar(self, canvas, percent, low_color, mid_color, high_color):
        canvas.delete("all")
        w = 400
        h = 20
        p = max(0.0, min(100.0, percent)) / 100.0
        fill_w = int(w * p)
        if percent >= 70.0:
            color = high_color
        elif percent >= 40.0:
            color = mid_color
        else:
            color = low_color
        canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        canvas.create_rectangle(0, 0, w, h, outline="#666666")

    def _estimate_improvement(self, global_state, foresight):
        if not global_state or not foresight:
            return 0.0

        current_risk = max(global_state.get("ram_percent", 0.0),
                           global_state.get("cpu_percent", 0.0))
        ram_pred = foresight.get("ram_pred")
        cpu_pred = foresight.get("cpu_pred")
        ram_conf = foresight.get("ram_conf") or 0.0
        cpu_conf = foresight.get("cpu_conf") or 0.0

        predicted_risk = max(ram_pred or 0.0, cpu_pred or 0.0)
        conf = max(ram_conf, cpu_conf)

        if predicted_risk <= current_risk or conf < FORESIGHT_MIN_CONFIDENCE:
            return 0.0

        avoided = predicted_risk - current_risk
        improvement = avoided * conf
        return max(0.0, min(100.0, improvement))

    def update_labels(self):
        sys_state = BACKBONE_STATE.get("system", {})
        ram = sys_state.get("ram", {})
        cpu = sys_state.get("cpu", {})

        if ram.get("total", 0):
            ram_str = f"RAM: {ram.get('used', 0) / (1024**3):.2f} / {ram.get('total', 1) / (1024**3):.2f} GB ({ram.get('percent', 0):.1f}%)"
        else:
            ram_str = "RAM: --"
        self.ram_var.set(ram_str)

        cpu_str = f"CPU: {cpu.get('percent', 0):.1f}% @ {cpu.get('freq_current', 0) or 0:.0f} MHz (max {cpu.get('freq_max', 0) or 0:.0f})"
        self.cpu_var.set(cpu_str)

        gpu_state = BACKBONE_STATE.get("gpu", {})
        if gpu_state.get("available"):
            gpus = gpu_state.get("gpus", [])
            if gpus:
                g = gpus[0]
                gpu_str = f"dGPU[0] {g['name']}: {g['memory_used'] / (1024**3):.2f} / {g['memory_total'] / (1024**3):.2f} GB ({g['memory_used_percent']:.1f}%)"
            else:
                gpu_str = "dGPU: available but no devices?"
        else:
            reason = gpu_state.get("reason", "not available")
            gpu_str = f"dGPU/APU: {reason}"
        self.gpu_var.set(gpu_str)

        caches = BACKBONE_STATE.get("caches", {})
        l1 = caches.get("l1")
        l2 = caches.get("l2")
        l3 = caches.get("l3")
        if l1 or l2 or l3:
            def fmt(x):
                return f"{x / 1024:.0f} KB" if x else "?"
            cache_str = f"Cache topology: L1={fmt(l1)}, L2={fmt(l2)}, L3={fmt(l3)} (static)"
        else:
            cache_str = "Cache topology: not detected (static only)."
        self.cache_var.set(cache_str)

        self.proc_list.delete(0, tk.END)
        for w in BACKBONE_STATE["workers"].get("process_workers", []):
            rss_gb = w["rss"] / (1024**3)
            line = f"{w['pid']:5d} {w['name'][:30]:30s}  RAM={rss_gb:5.2f} GB  CPU={w['cpu_percent']:4.1f}%"
            self.proc_list.insert(tk.END, line)

        self.plan_list.delete(0, tk.END)
        policy = BACKBONE_STATE.get("policy_decisions", {})
        global_state = policy.get("global_state", {})
        foresight = policy.get("foresight", {})

        self.plan_list.insert(tk.END, f"Global load → RAM={global_state.get('ram_percent', 0):.1f}%  CPU={global_state.get('cpu_percent', 0):.1f}%")
        if foresight:
            self.plan_list.insert(
                tk.END,
                f"Foresight → RAM_pred={foresight.get('ram_pred')} (conf={foresight.get('ram_conf')}) | "
                f"CPU_pred={foresight.get('cpu_pred')} (conf={foresight.get('cpu_conf')})"
            )
        routing = BACKBONE_STATE.get("ai_routing", {})
        if routing:
            self.plan_list.insert(tk.END, "-" * 60)
            self.plan_list.insert(tk.END, "Routing decisions (lanes):")
            for w_name, r in routing.items():
                self.plan_list.insert(
                    tk.END,
                    f"{w_name}: lane={r['desired_lane']} → {r['device_type']}[{r['device_index']}] {r['device_name']} (rank={r['speed_rank']})"
                )

        self.actions_list.delete(0, tk.END)
        for a in BACKBONE_STATE.get("actions", [])[-25:]:
            self.actions_list.insert(tk.END, f"{a['kind']}: {a['payload']}")

        improvement = self._estimate_improvement(global_state, foresight)
        self._draw_bar(self.improvement_canvas, improvement, "#33FF66", "#FFCC33", "#FF4444")
        self.improvement_label.config(text=f"Improvement: {improvement:5.1f} %")

        t_stats = BACKBONE_STATE["transport"]["stats"]
        load = t_stats.get("load_percent", 0.0)
        self._draw_bar(self.transport_canvas, load, "#33FFAA", "#FFDD55", "#FF5555")
        self.transport_label.config(text=f"Transport Load: {load:5.1f} %")

        # Brain Cortex Panel
        coach = BACKBONE_STATE["organs"].get("AICoachOrgan", {})
        risk = max(global_state.get("ram_percent", 0.0),
                   global_state.get("cpu_percent", 0.0),
                   t_stats.get("load_percent", 0.0))

        health_str = f"Health/Risk: {risk:4.1f}%  |  Transport={t_stats.get('load_percent', 0.0):4.1f}%"
        self.cortex_health_var.set(health_str)

        meta_state = BACKBONE_STATE["brain_cortex"].get("meta_state", {}).get("game_phase", "UNKNOWN")
        stance = coach.get("stance", "UNKNOWN")
        self.cortex_meta_var.set(f"Meta-state: phase={meta_state}  stance={stance}")

        thresholds = BACKBONE_STATE["brain_cortex"].get("stance_thresholds", {})
        if thresholds:
            self.cortex_stance_var.set(
                f"Stance thresholds: CPU≤{thresholds.get('cpu', 0)}  RAM≤{thresholds.get('ram', 0)}  Transport≤{thresholds.get('transport', 0)}"
            )
        else:
            self.cortex_stance_var.set("Stance thresholds: --")

        channels = BACKBONE_STATE["brain_cortex"]["deep_channels"]
        ch_flags = []
        for name in ["ram", "backup", "network", "gpu", "thermal", "disk", "vram", "swarm", "coach"]:
            ch = channels.get(name)
            active = bool(ch)
            ch_flags.append(f"{name.upper()}={'ON' if active else 'off'}")
        self.cortex_channels_var.set("Deep channels: " + "  ".join(ch_flags))

    def stop(self):
        self._stop = True


# ---------- Main ----------

def main():
    init_example_workloads()

    snapshot = load_latest_snapshot(LOCAL_SAVE_DIR)
    if snapshot:
        restore_backbone_from_snapshot(snapshot)

    root = tk.Tk()
    app = BorgBackboneGUI(root)
    try:
        root.mainloop()
    finally:
        app.stop()
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()

