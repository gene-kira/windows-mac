"""
prometheus_predictive_full_opportunities_persistent_multibackup_plus_ml_and_sync_fixed.py

Full build with:

- Predictive Routing Brain (pattern learning, risk cones, forecasting, time-of-day awareness, anomaly detection)
- Real network probing hooks (ICMP ping, traceroute, DNS timing, jitter/loss estimation)
- Network Watcher feeding the learner
- Future Planner (forks, merged futures, hybrid futures, lane-aware scoring with forecast + history influence, upgrade window prediction)
- Execution Bypass Engine (self-learning, pre-emptive, lane events, context-aware, fingerprint clustering, fallback scoring, persistence)
- Backup Manager:
    * Multiple backup paths
    * Persistent path list
    * Persistent active + last-used backup path
    * Per-path scoring (latency + reliability)
    * Per-path forecast
    * Active path selection
    * Mirror groups (multi-path redundancy)
    * Persistent mirror groups
    * Backup window prediction
- Redundant Backup Manager (integrated with multi-path model)
- Code Lane Timeline (GUI)
- Hazard Radar with per-metric bands (CPU, DNS, Routes, Env, Code, Backup)
- Bypass Heatmap
- Fingerprint Cluster panel
- Drive Scores table
- Area of Opportunity Engine:
    * Scans routing, backup, bypass, lanes, drives, forks
    * Computes opportunity scores (impact vs effort vs gap)
    * Surfaces a ranked “Area of Opportunity” table in its own tab
    * Persists opportunity history
- Persistence Layer:
    * routing history
    * bypass fingerprints
    * drive scores
    * lane events
    * opportunity history
    * backup_paths
    * mirror_groups
    * active_backup_path
    * last_used_backup_path
- Node Sync Manager (file-based distributed sync skeleton)
- Plugin Registry (predictors, fallback providers, planners, opportunity scanners)
- ML Predictor hooks (stub for LSTM/transformer/regression)
- Upgrade Engine stub (package managers / Windows APIs / rollback snapshots / dependency graphs)
- Self-driving mode aware of:
    * routing risk + forecast + anomalies
    * backup risk + predicted windows + auto-switch
    * lane stability + forecast + history
    * upgrade window prediction
- Self-driving intent log + forecast strip + upgrade window indicator
- Crash-proof persistence (atomic writes, guarded loads)
- Auto-save backup paths and active path
- Thread-safe GUI logging (Qt signal-based logger)
- Guaranteed backup path restore on startup (active path restored from persistence)
"""

import math
import sys
import time
import socket
import threading
import traceback
import os
import shutil
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable

from PyQt5 import QtWidgets, QtCore, QtGui


# ============================================================
# Thread-safe GUI Logger
# ============================================================

class ThreadSafeLogger(QtCore.QObject):
    message_emitted = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, msg: str):
        # Safe to call from any thread; Qt will queue to GUI thread
        self.message_emitted.emit(msg)


# ============================================================
# Persistence Layer
# ============================================================

class PersistenceLayer:
    """
    Crash-resistant JSON persistence for:
    - routing history
    - bypass fingerprints
    - drive scores
    - lane events
    - opportunity history
    - backup_paths
    - mirror_groups
    - active_backup_path
    - last_used_backup_path
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.path.expanduser("~"), "prometheus_state")
        os.makedirs(self.base_dir, exist_ok=True)
        self.json_path = os.path.join(self.base_dir, "state.json")
        self._state = {
            "routing_history": [],
            "bypass_fingerprints": [],
            "drive_scores": {},
            "lane_events": [],
            "opportunity_history": [],
            "backup_paths": [],
            "mirror_groups": [],
            "active_backup_path": None,
            "last_used_backup_path": None,
        }
        self._load()

    def _load(self):
        try:
            if os.path.isfile(self.json_path):
                import json
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._state.update(data)
        except Exception:
            # Corrupt or unreadable file: keep defaults, do not crash
            pass

    def _save(self):
        try:
            import json
            tmp = self.json_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.json_path)
        except Exception:
            # Never crash on persistence failure
            pass

    def append_routing_sample(self, dest: str, latency_ms: float, success: bool, ts: Optional[float] = None):
        ts = ts or time.time()
        self._state["routing_history"].append(
            {"ts": ts, "dest": dest, "latency_ms": latency_ms, "success": success}
        )
        if len(self._state["routing_history"]) > 5000:
            self._state["routing_history"] = self._state["routing_history"][-5000:]
        self._save()

    def append_bypass_fingerprint(self, func_name: str, exc_type: str, ctx_key: str, ts: Optional[float] = None):
        ts = ts or time.time()
        self._state["bypass_fingerprints"].append(
            {"ts": ts, "func": func_name, "exc": exc_type, "ctx": ctx_key}
        )
        if len(self._state["bypass_fingerprints"]) > 5000:
            self._state["bypass_fingerprints"] = self._state["bypass_fingerprints"][-5000:]
        self._save()

    def update_drive_scores(self, drive_stats: Dict[str, Dict[str, int]]):
        self._state["drive_scores"] = drive_stats
        self._save()

    def append_lane_event(self, event: "CodeLaneEvent"):
        self._state["lane_events"].append(
            {
                "ts": event.timestamp,
                "func": event.func_name,
                "type": event.event_type,
                "details": event.details,
            }
        )
        if len(self._state["lane_events"]) > 5000:
            self._state["lane_events"] = self._state["lane_events"][-5000:]
        self._save()

    def append_opportunity_snapshot(self, opportunities: List["Opportunity"]):
        snap = {
            "ts": time.time(),
            "items": [
                {
                    "area": o.area,
                    "subsystem": o.subsystem,
                    "score": o.score,
                    "impact": o.impact,
                    "effort": o.effort,
                    "notes": o.notes,
                }
                for o in opportunities
            ],
        }
        self._state["opportunity_history"].append(snap)
        if len(self._state["opportunity_history"]) > 500:
            self._state["opportunity_history"] = self._state["opportunity_history"][-500:]
        self._save()

    def get_state(self) -> Dict[str, Any]:
        return self._state

    # --- Backup paths + mirror groups + active path ---

    def get_backup_paths(self) -> List[str]:
        return list(self._state.get("backup_paths", []))

    def set_backup_paths(self, paths: List[str]):
        self._state["backup_paths"] = list(paths)
        self._save()

    def get_mirror_groups(self) -> List[Dict[str, Any]]:
        return list(self._state.get("mirror_groups", []))

    def set_mirror_groups(self, groups: List[Dict[str, Any]]):
        self._state["mirror_groups"] = list(groups)
        self._save()

    def get_active_backup_path(self) -> Optional[str]:
        return self._state.get("active_backup_path")

    def set_active_backup_path(self, path: Optional[str]):
        self._state["active_backup_path"] = path
        self._save()

    def get_last_used_backup_path(self) -> Optional[str]:
        return self._state.get("last_used_backup_path")

    def set_last_used_backup_path(self, path: Optional[str]):
        self._state["last_used_backup_path"] = path
        self._save()


# ============================================================
# Network Probe (real probing hooks)
# ============================================================

class NetworkProbe:
    """
    Real network probing:
    - ICMP ping (via ping command)
    - traceroute (via tracert on Windows)
    - DNS timing
    - jitter + packet loss estimation (from repeated pings)
    """

    def __init__(self, logger: Callable[[str], None]):
        self.logger = logger

    def ping(self, host: str, count: int = 4, timeout: int = 1000) -> Dict[str, Any]:
        import subprocess, statistics
        cmd = ["ping", "-n", str(count), "-w", str(timeout), host]
        try:
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            duration = (time.time() - start) * 1000.0
            out = proc.stdout
            rtts = []
            for line in out.splitlines():
                if "time=" in line.lower():
                    parts = line.split("time=")[-1]
                    val = ""
                    for ch in parts:
                        if ch.isdigit() or ch == ".":
                            val += ch
                        else:
                            break
                    if val:
                        rtts.append(float(val))
            if not rtts:
                return {"success": False, "latency_ms": duration, "jitter_ms": 0.0, "loss": 1.0}
            avg = statistics.mean(rtts)
            jitter = statistics.pstdev(rtts) if len(rtts) > 1 else 0.0
            loss = max(0.0, 1.0 - len(rtts) / count)
            return {"success": True, "latency_ms": avg, "jitter_ms": jitter, "loss": loss}
        except Exception as e:
            self.logger(f"[Probe] ping failed for {host}: {e}")
            return {"success": False, "latency_ms": 0.0, "jitter_ms": 0.0, "loss": 1.0}

    def traceroute(self, host: str, max_hops: int = 20) -> str:
        import subprocess
        cmd = ["tracert", "-h", str(max_hops), host]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return proc.stdout
        except Exception as e:
            self.logger(f"[Probe] traceroute failed for {host}: {e}")
            return ""

    def dns_timing(self, host: str) -> float:
        start = time.time()
        try:
            socket.gethostbyname(host)
        except Exception:
            pass
        return (time.time() - start) * 1000.0


# ============================================================
# ML Predictor Hooks (stub)
# ============================================================

class MLPredictor:
    """
    Hook for ML-based prediction.
    Real models (LSTM/transformer/regression) can be loaded here.
    For now, this is a stub that can be swapped out.
    """

    def __init__(self):
        self.enabled = False

    def predict_routing_sequence(self, history: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        if not self.enabled or not history:
            return []
        return []

    def predict_drive_failure_risk(self, drive_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        if not self.enabled:
            return {k: 0.0 for k in drive_stats.keys()}
        return {k: 0.0 for k in drive_stats.keys()}

    def predict_upgrade_risk(self, fork_features: Dict[str, Any]) -> float:
        if not self.enabled:
            return 0.0
        return 0.0


# ============================================================
# Redundant Backup Manager (mirrored backups)
# ============================================================

class RedundantBackupManager:
    """
    Manages mirrored backups across multiple paths.
    Uses BackupManagerEngine for testing/scoring each path.
    """

    def __init__(self, backup_engine: "BackupManagerEngine", logger: Callable[[str], None]):
        self.backup_engine = backup_engine
        self.logger = logger
        self.mirrors: List[str] = []

    def set_mirrors(self, paths: List[str]):
        self.mirrors = paths
        self.logger(f"[RedundantBackup] Mirrors set: {paths}")

    def add_mirror(self, path: str):
        if path not in self.mirrors:
            self.mirrors.append(path)
            self.logger(f"[RedundantBackup] Mirror added: {path}")

    def remove_mirror(self, path: str):
        if path in self.mirrors:
            self.mirrors.remove(path)
            self.logger(f"[RedundantBackup] Mirror removed: {path}")

    def test_all(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for p in self.mirrors:
            results[p] = self.backup_engine.test_backup_path(p)
        return results

    def choose_primary(self) -> Optional[str]:
        if not self.mirrors:
            return None
        best = max(self.mirrors, key=self.backup_engine.path_score)
        self.logger(f"[RedundantBackup] Primary chosen: {best}")
        self.backup_engine.set_active_path(best)
        return best


# ============================================================
# Node Sync Manager (distributed sync skeleton)
# ============================================================

class NodeSyncManager:
    """
    Very simple distributed sync via shared folder / file drop.
    Real implementation could use HTTP, gRPC, message bus, etc.
    """

    def __init__(self, persistence: PersistenceLayer, logger: Callable[[str], None]):
        self.persistence = persistence
        self.logger = logger
        self.sync_dir = os.path.join(self.persistence.base_dir, "sync")
        os.makedirs(self.sync_dir, exist_ok=True)

    def export_state(self, node_id: str):
        import json
        path = os.path.join(self.sync_dir, f"{node_id}_state.json")
        try:
            with open(path, "w") as f:
                json.dump(self.persistence.get_state(), f)
        except Exception:
            return
        self.logger(f"[Sync] Exported state to {path}")

    def import_states(self) -> List[Dict[str, Any]]:
        import json
        states = []
        for name in os.listdir(self.sync_dir):
            if not name.endswith("_state.json"):
                continue
            path = os.path.join(self.sync_dir, name)
            try:
                with open(path, "r") as f:
                    states.append(json.load(f))
            except Exception:
                continue
        return states


# ============================================================
# Plugin Registry
# ============================================================

class PluginRegistry:
    """
    Simple plugin registry:
    - predictors
    - fallback providers
    - planners
    - opportunity scanners
    """

    def __init__(self):
        self.predictors: List[Callable[[Dict[str, Any]], None]] = []
        self.fallback_providers: List[Callable[[str], Optional[Callable]]] = []
        self.planners: List[Callable[["FuturePlannerEngine"], None]] = []
        self.opportunity_scanners: List[Callable[["AreaOfOpportunityEngine"], None]] = []

    def register_predictor(self, fn: Callable[[Dict[str, Any]], None]):
        self.predictors.append(fn)

    def register_fallback_provider(self, fn: Callable[[str], Optional[Callable]]):
        self.fallback_providers.append(fn)

    def register_planner(self, fn: Callable[["FuturePlannerEngine"], None]):
        self.planners.append(fn)

    def register_opportunity_scanner(self, fn: Callable[["AreaOfOpportunityEngine"], None]):
        self.opportunity_scanners.append(fn)


# ============================================================
# Upgrade Engine Stub
# ============================================================

class UpgradeEngine:
    """
    Real upgrade engine stub:
    - package managers
    - Windows Update APIs
    - rollback snapshots
    - dependency graphs
    """

    def __init__(self, logger: Callable[[str], None]):
        self.logger = logger

    def apply_plan(self, base_snapshot: int, upgrades: Dict[str, str]):
        self.logger(f"[UpgradeEngine] Would apply from snapshot {base_snapshot}: {upgrades}")

    def create_rollback_snapshot(self, label: str) -> int:
        ts = int(time.time())
        self.logger(f"[UpgradeEngine] Created rollback snapshot {ts} ({label})")
        return ts


# ============================================================
# Execution Bypass Engine
# ============================================================

class ExecutionBypassEngine:
    """
    Execution Bypass Engine:

    - Wraps functions
    - Catches failures
    - Reroutes to fallbacks
    - Logs every reroute
    - Tracks failure/reroute counts (for heatmap)
    - Learns self-healing replacements over time
    - Records failure fingerprints and can pre-emptively bypass
    - Clusters fingerprints to understand failure families
    - Scores fallbacks by reliability
    - Emits code lane events into the Future Planner
    - Persists fingerprints (via PersistenceLayer if attached)
    """

    def __init__(self, logger: Callable[[str], None],
                 promote_threshold: int = 3):
        self.logger = logger
        self.bypass_enabled = True

        self.failure_counts = defaultdict(int)
        self.reroute_counts = defaultdict(int)
        self.fallback_success_counts = defaultdict(int)

        self.self_healing_map: Dict[str, Callable] = {}

        self.promote_threshold = promote_threshold

        self.planner = None

        self.failure_fingerprints = defaultdict(int)
        self.preemptive_threshold = 3

        self.context_provider: Optional[Callable[[], Dict[str, Any]]] = None

        self.fallback_stats = defaultdict(lambda: {"success": 0, "fail": 0})

        self.fingerprint_clusters: Dict[int, Dict[str, Any]] = {}
        self._cluster_counter = 0

        self.persistence: Optional[PersistenceLayer] = None

    def attach_planner(self, planner_engine):
        self.planner = planner_engine

    def attach_context_provider(self, provider: Callable[[], Dict[str, Any]]):
        self.context_provider = provider

    def set_bypass_enabled(self, enabled: bool):
        self.bypass_enabled = enabled
        state = "ENABLED" if enabled else "DISABLED"
        self.logger(f"[Bypass] Execution bypass {state}")

    def get_heatmap_data(self) -> Dict[str, Dict[str, int]]:
        data = {}
        for name in set(list(self.failure_counts.keys()) + list(self.reroute_counts.keys())):
            data[name] = {
                "failures": self.failure_counts.get(name, 0),
                "reroutes": self.reroute_counts.get(name, 0)
            }
        return data

    def _emit_event(self, func_name: str, event_type: str, details: str):
        if self.planner is None:
            return
        evt = CodeLaneEvent(
            timestamp=time.time(),
            func_name=func_name,
            event_type=event_type,
            details=details
        )
        self.planner.record_code_lane_event(evt)

    def _promote_if_ready(self, func_name: str, candidate: Callable):
        count = self.fallback_success_counts[func_name]
        if count >= self.promote_threshold and func_name not in self.self_healing_map:
            self.self_healing_map[func_name] = candidate
            msg = (f"[Bypass] Self-healing promotion: {func_name} "
                   f"fallback promoted after {count} successful reroutes")
            self.logger(msg)
            self._emit_event(
                func_name,
                "promotion",
                f"Fallback promoted to primary after {count} successes"
            )

    def _current_context_key(self) -> str:
        if not self.context_provider:
            return "no_context"
        ctx = self.context_provider()
        routing = ctx.get("routing_risk", "unknown")
        backup = ctx.get("backup_risk", "unknown")
        lane = ctx.get("lane_risk", "unknown")
        active_lane = ctx.get("active_lane", "none")
        return f"routing={routing}|backup={backup}|lane={lane}|lane={active_lane}"

    def _cluster_fingerprint(self, key) -> int:
        func_name, exc_type, ctx_key = key
        for cid, info in self.fingerprint_clusters.items():
            for ex in info["examples"]:
                f2, e2, ctx2 = ex
                if f2 == func_name and e2 == exc_type:
                    info["count"] += 1
                    if len(info["examples"]) < 5:
                        info["examples"].append(key)
                    return cid
        cid = self._cluster_counter
        self._cluster_counter += 1
        self.fingerprint_clusters[cid] = {"count": 1, "examples": [key]}
        return cid

    def fingerprint_cluster_summary(self) -> Dict[int, Dict[str, Any]]:
        return {
            cid: {
                "count": info["count"],
                "examples": info["examples"][:3]
            }
            for cid, info in self.fingerprint_clusters.items()
        }

    def _record_failure_fingerprint(self, func_name: str, exc: Exception):
        exc_type = type(exc).__name__
        ctx_key = self._current_context_key()
        key = (func_name, exc_type, ctx_key)
        self.failure_fingerprints[key] += 1
        cluster_id = self._cluster_fingerprint(key)
        self.logger(f"[Bypass] Fingerprint clustered into #{cluster_id}")
        if self.persistence:
            self.persistence.append_bypass_fingerprint(func_name, exc_type, ctx_key)
        return key, self.failure_fingerprints[key]

    def _should_preempt(self, func_name: str) -> bool:
        if not self.context_provider:
            return False
        ctx_key = self._current_context_key()
        for (fname, _, ckey), count in self.failure_fingerprints.items():
            if fname == func_name and ckey == ctx_key and count >= self.preemptive_threshold:
                return True
        return False

    def _record_fallback_result(self, func_name: str, success: bool):
        stats = self.fallback_stats[func_name]
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1

    def fallback_score(self, func_name: str) -> float:
        stats = self.fallback_stats[func_name]
        s, f = stats["success"], stats["fail"]
        if s + f == 0:
            return 0.0
        return s / (s + f)

    def wrap(self,
             func: Callable,
             func_name: Optional[str] = None,
             fallback: Optional[Callable] = None):
        name = func_name or func.__name__

        def wrapped(*args, **kwargs):
            if self.bypass_enabled and fallback is not None and self._should_preempt(name):
                self.reroute_counts[name] += 1
                self.logger(f"[Bypass] PRE-EMPTIVE bypass for {name} based on failure history")
                self._emit_event(
                    name,
                    "preemptive_bypass",
                    "Primary path skipped due to known bad context"
                )
                try:
                    result = fallback(*args, **kwargs)
                    self.fallback_success_counts[name] += 1
                    self._record_fallback_result(name, True)
                    self._promote_if_ready(name, fallback)
                    return result
                except Exception as e:
                    self.failure_counts[name] += 1
                    self._record_fallback_result(name, False)
                    self.logger(f"[Bypass] Pre-emptive fallback for {name} failed: {e}")
                    self.logger(traceback.format_exc())
                    raise

            primary = self.self_healing_map.get(name, func)

            try:
                return primary(*args, **kwargs)
            except Exception as e:
                self.failure_counts[name] += 1
                self.logger(f"[Bypass] Failure in {name}: {e}")
                self.logger(traceback.format_exc())
                fp_key, fp_count = self._record_failure_fingerprint(name, e)
                self.logger(f"[Bypass] Failure fingerprint {fp_key} count={fp_count}")

                if not self.bypass_enabled:
                    raise

                if primary is not func:
                    self.logger(f"[Bypass] Learned replacement for {name} failed, "
                                f"falling back to original implementation")
                    self._emit_event(
                        name,
                        "demotion",
                        "Learned replacement failed; reverting to original"
                    )
                    try:
                        result = func(*args, **kwargs)
                        self.logger(f"[Bypass] Original implementation for {name} succeeded, "
                                    f"demoting learned replacement")
                        if name in self.self_healing_map:
                            del self.self_healing_map[name]
                        return result
                    except Exception as e2:
                        self.failure_counts[name] += 1
                        self.logger(f"[Bypass] Original implementation for {name} also failed: {e2}")
                        self.logger(traceback.format_exc())
                        self._record_failure_fingerprint(name, e2)

                if fallback is not None:
                    self.reroute_counts[name] += 1
                    self.logger(f"[Bypass] Rerouting {name} to fallback")
                    self._emit_event(
                        name,
                        "fallback_used",
                        "Fallback executed due to primary failure"
                    )
                    try:
                        result = fallback(*args, **kwargs)
                        self.fallback_success_counts[name] += 1
                        self._record_fallback_result(name, True)
                        self._promote_if_ready(name, fallback)
                        return result
                    except Exception as e3:
                        self.failure_counts[name] += 1
                        self._record_fallback_result(name, False)
                        self.logger(f"[Bypass] Fallback for {name} also failed: {e3}")
                        self.logger(traceback.format_exc())
                        self._record_failure_fingerprint(name, e3)

                raise

        return wrapped


# ============================================================
# Predictive Routing Brain
# ============================================================

class PatternLearningEngine:
    def __init__(self, max_sequence_len: int = 5):
        self.dest_stats = defaultdict(lambda: {"count": 0, "last_seen": 0.0})
        self.sequence_counts = defaultdict(int)
        self.recent_sequence = deque(maxlen=max_sequence_len)
        self.time_buckets = defaultdict(lambda: defaultdict(int))

    def observe_destination(self, dest: str):
        now = time.time()
        hour = time.localtime(now).tm_hour

        self.dest_stats[dest]["count"] += 1
        self.dest_stats[dest]["last_seen"] = now
        self.time_buckets[hour][dest] += 1

        if self.recent_sequence:
            prev = self.recent_sequence[-1]
            self.sequence_counts[(prev, dest)] += 1
        self.recent_sequence.append(dest)

    def top_destinations(self, n: int = 5) -> List[str]:
        return sorted(
            self.dest_stats.keys(),
            key=lambda d: self.dest_stats[d]["count"],
            reverse=True
        )[:n]

    def predict_next_after(self, current: str, n: int = 3) -> List[str]:
        candidates = []
        for (a, b), count in self.sequence_counts.items():
            if a == current:
                candidates.append((b, count))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:n]]

    def predict_top_for_current_hour(self, n: int = 5) -> List[str]:
        now = time.time()
        hour = time.localtime(now).tm_hour
        bucket = self.time_buckets[hour]
        if not bucket:
            return self.top_destinations(n)
        return sorted(bucket.keys(), key=lambda d: bucket[d], reverse=True)[:n]


class ZeroLatencyResolver:
    def __init__(self):
        self.dns_cache: Dict[str, str] = {}
        self.route_cache: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def preload(self, dest: str):
        if self._is_ip(dest):
            ip = dest
        else:
            ip = self._resolve_dns(dest)
        if ip:
            self._cache_route(ip)

    def resolve(self, dest: str) -> Tuple[Optional[str], Optional[Any]]:
        with self.lock:
            if self._is_ip(dest):
                ip = dest
            else:
                ip = self.dns_cache.get(dest)

        if not ip:
            ip = self._resolve_dns(dest)

        route_info = self.route_cache.get(ip)
        if not route_info:
            route_info = self._compute_route(ip)

        return ip, route_info

    def _resolve_dns(self, domain: str) -> Optional[str]:
        with self.lock:
            if domain in self.dns_cache:
                return self.dns_cache[domain]
        try:
            ip = socket.gethostbyname(domain)
        except Exception:
            ip = None
        if ip:
            with self.lock:
                self.dns_cache[domain] = ip
        return ip

    def _cache_route(self, ip: str):
        route_info = self._compute_route(ip)
        with self.lock:
            self.route_cache[ip] = route_info

    def _compute_route(self, ip: str) -> Dict[str, Any]:
        return {
            "ip": ip,
            "interface": "auto",
            "metric": 10,
            "via": "default_gateway"
        }

    def _is_ip(self, s: str) -> bool:
        parts = s.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False


class DestinationPreloader:
    def __init__(self, resolver: ZeroLatencyResolver):
        self.resolver = resolver

    def preload_destinations(self, dests: List[str]):
        for d in dests:
            self.resolver.preload(d)


class GameMovementPredictor:
    def __init__(self, learner: PatternLearningEngine):
        self.learner = learner

    def observe_game_server(self, server: str):
        self.learner.observe_destination(f"game:{server}")

    def predict_next_servers(self, current_server: str, n: int = 3) -> List[str]:
        preds = self.learner.predict_next_after(f"game:{current_server}", n)
        return [p.replace("game:", "") for p in preds]


class BrowserDestinationPredictor:
    def __init__(self, learner: PatternLearningEngine):
        self.learner = learner

    def observe_domain(self, domain: str):
        self.learner.observe_destination(f"web:{domain}")

    def predict_next_domains(self, current_domain: str, n: int = 3) -> List[str]:
        preds = self.learner.predict_next_after(f"web:{current_domain}", n)
        return [p.replace("web:", "") for p in preds]


class RoutingRiskModel:
    def __init__(self):
        self.latency_ewma = {}
        self.failure_ewma = {}
        self.alpha = 0.3
        self.latency_history = defaultdict(lambda: deque(maxlen=50))

    def observe(self, dest: str, latency_ms: float, success: bool):
        self._update_ewma(self.latency_ewma, dest, latency_ms)
        self._update_ewma(self.failure_ewma, dest, 0.0 if success else 1.0)
        self.latency_history[dest].append(latency_ms)

    def _update_ewma(self, store, key, value):
        if key not in store:
            store[key] = (value, self.alpha)
        else:
            prev, a = store[key]
            store[key] = (a * value + (1 - a) * prev, a)

    def cone_for_dest(self, dest: str, steps: int = 5) -> Dict[str, List[Dict]]:
        lat = self.latency_ewma.get(dest, (50.0, self.alpha))[0]
        fail = self.failure_ewma.get(dest, (0.1, self.alpha))[0]

        cones = {"latency": [], "failure": []}
        for i in range(1, steps + 1):
            spread_lat = lat * 0.1 * i
            cones["latency"].append({
                "step": i,
                "center": lat,
                "lower": max(0.0, lat - spread_lat),
                "upper": lat + spread_lat
            })
            spread_fail = 0.1 * i
            center_fail = fail
            cones["failure"].append({
                "step": i,
                "center": center_fail,
                "lower": max(0.0, center_fail - spread_fail),
                "upper": min(1.0, center_fail + spread_fail)
            })
        return cones

    def overall_routing_risk_level(self) -> str:
        if not self.failure_ewma:
            return "low"
        avg_fail = sum(v for v, _ in self.failure_ewma.values()) / len(self.failure_ewma)
        if avg_fail < 0.2:
            return "low"
        if avg_fail < 0.5:
            return "medium"
        return "high"

    def is_latency_anomalous(self, dest: str, threshold_sigma: float = 3.0) -> bool:
        hist = self.latency_history[dest]
        if len(hist) < 10:
            return False
        avg = sum(hist) / len(hist)
        var = sum((x - avg) ** 2 for x in hist) / len(hist)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return False
        latest = hist[-1]
        return abs(latest - avg) > threshold_sigma * std

    def forecast_routing_risk(self, horizon_steps: int = 3) -> List[str]:
        if not self.failure_ewma:
            return ["low"] * horizon_steps
        avg_fail = sum(v for v, _ in self.failure_ewma.values()) / len(self.failure_ewma)
        anomaly_penalty = 0.0
        for dest in self.latency_history.keys():
            if self.is_latency_anomalous(dest):
                anomaly_penalty += 0.05
        forecast = []
        for i in range(1, horizon_steps + 1):
            projected = min(1.0, avg_fail + 0.03 * i + anomaly_penalty)
            if projected < 0.2:
                forecast.append("low")
            elif projected < 0.5:
                forecast.append("medium")
            else:
                forecast.append("high")
        return forecast


class UnifiedAccessRoadController:
    def __init__(self, probe: Optional[NetworkProbe] = None):
        self.learner = PatternLearningEngine()
        self.resolver = ZeroLatencyResolver()
        self.preloader = DestinationPreloader(self.resolver)
        self.game_predictor = GameMovementPredictor(self.learner)
        self.browser_predictor = BrowserDestinationPredictor(self.learner)
        self.risk_model = RoutingRiskModel()
        self.probe = probe

    def observe_connection(self, dest: str, latency_ms: float, success: bool):
        self.learner.observe_destination(f"net:{dest}")
        self.risk_model.observe(dest, latency_ms, success)

    def observe_game_server(self, server: str, latency_ms: float, success: bool):
        self.game_predictor.observe_game_server(server)
        self.risk_model.observe(server, latency_ms, success)

    def observe_browser_domain(self, domain: str, latency_ms: float, success: bool):
        self.browser_predictor.observe_domain(domain)
        self.risk_model.observe(domain, latency_ms, success)

    def predict_next_for_game(self, current_server: str, n: int = 3) -> List[str]:
        return self.game_predictor.predict_next_servers(current_server, n)

    def predict_next_for_browser(self, current_domain: str, n: int = 3) -> List[str]:
        return self.browser_predictor.predict_next_domains(current_domain, n)

    def predict_next_generic(self, current_dest: str, n: int = 3) -> List[str]:
        preds = self.learner.predict_next_after(f"net:{current_dest}", n)
        return [p.replace("net:", "") for p in preds]

    def predict_top_for_current_hour(self, n: int = 5) -> List[str]:
        preds = self.learner.predict_top_for_current_hour(n)
        cleaned = []
        for p in preds:
            if p.startswith("net:"):
                cleaned.append(p.replace("net:", ""))
            elif p.startswith("web:"):
                cleaned.append(p.replace("web:", ""))
            elif p.startswith("game:"):
                cleaned.append(p.replace("game:", ""))
            else:
                cleaned.append(p)
        return cleaned

    def preload_generic(self, current_dest: str, n: int = 3):
        preds = self.predict_next_generic(current_dest, n)
        self.preloader.preload_destinations(preds)

    def preload_time_of_day_top(self, n: int = 5):
        preds = self.predict_top_for_current_hour(n)
        self.preloader.preload_destinations(preds)

    def resolve_fast(self, dest: str) -> Tuple[Optional[str], Optional[Any]]:
        return self.resolver.resolve(dest)

    def routing_risk_level(self) -> str:
        return self.risk_model.overall_routing_risk_level()

    def routing_cones_for_top_dests(self, n: int = 3) -> Dict[str, Dict[str, List[Dict]]]:
        top = self.learner.top_destinations(n)
        cones = {}
        for d in top:
            if d.startswith("net:"):
                key = d.replace("net:", "")
            elif d.startswith("web:"):
                key = d.replace("web:", "")
            elif d.startswith("game:"):
                key = d.replace("game:", "")
            else:
                key = d
            cones[key] = self.risk_model.cone_for_dest(key)
        return cones

    def forecast_routing_risk(self, horizon_steps: int = 3) -> List[str]:
        return self.risk_model.forecast_routing_risk(horizon_steps)


class NetworkWatcher:
    def __init__(self, routing_brain: UnifiedAccessRoadController):
        self.brain = routing_brain

    def on_connection(self, dest: str, latency_ms: float, success: bool):
        self.brain.observe_connection(dest, latency_ms, success)


# ============================================================
# Future Planner Data Models
# ============================================================

@dataclass
class ForkResult:
    risk_level: str
    aggregate_break_risk: float
    risk_cones: Dict[str, List[Dict]]
    rollback_available: bool


@dataclass
class Fork:
    name: str
    snapshot_ts: int
    upgrades: Dict[str, str]
    result: ForkResult


@dataclass
class MergedFuture:
    base_fork: str
    base_snapshot: int
    merged_upgrades: Dict[str, str]
    merged_risk_level: str
    merged_score: float
    notes: List[str] = field(default_factory=list)


@dataclass
class CodeLaneEvent:
    timestamp: float
    func_name: str
    event_type: str
    details: str


# ============================================================
# Future Planner Engine
# ============================================================

class FuturePlannerEngine:
    def __init__(self):
        self.forks: List[Fork] = []
        self.code_lane_events: List[CodeLaneEvent] = []
        self.routing_forecast: List[str] = []
        self.backup_forecast: List[str] = []
        self.fork_history_scores: Dict[str, List[float]] = defaultdict(list)
        self.persistence: Optional[PersistenceLayer] = None
        self.ml_predictor: Optional[MLPredictor] = None

    def record_code_lane_event(self, event: CodeLaneEvent):
        self.code_lane_events.append(event)
        if self.persistence:
            self.persistence.append_lane_event(event)

    def update_forecasts(self, routing: List[str], backup: List[str]):
        self.routing_forecast = routing
        self.backup_forecast = backup

    def record_fork_outcome(self, fork_name: str, stability_score: float):
        self.fork_history_scores[fork_name].append(stability_score)

    def fork_reliability(self, fork_name: str) -> float:
        hist = self.fork_history_scores.get(fork_name, [])
        if not hist:
            return 1.0
        return sum(hist) / len(hist)

    def _risk_level_to_score(self, level: str) -> float:
        if level == "low":
            return 1.0
        if level == "medium":
            return 2.0
        return 3.0

    def _trend_penalty(self, fork: Fork) -> float:
        penalty = 0.0
        for metric, cone in fork.result.risk_cones.items():
            if not cone:
                continue
            last = cone[-1]
            spread = last["upper"] - last["lower"]
            penalty += spread
        return penalty

    def _rollback_bonus(self, fork: Fork) -> float:
        return -0.5 if fork.result.rollback_available else 0.5

    def _lane_stability_penalty(self, fork: Fork) -> float:
        recent = self.code_lane_events[-20:]
        demotions = sum(1 for e in recent if e.event_type == "demotion")
        fallbacks = sum(1 for e in recent if e.event_type in ("fallback_used", "preemptive_bypass"))
        return 0.1 * demotions + 0.05 * fallbacks

    def _forecast_penalty(self) -> float:
        penalty = 0.0
        for lvl in self.routing_forecast:
            if lvl == "medium":
                penalty += 0.1
            elif lvl == "high":
                penalty += 0.3
        for lvl in self.backup_forecast:
            if lvl == "medium":
                penalty += 0.1
            elif lvl == "high":
                penalty += 0.3
        return penalty

    def _history_penalty(self, fork: Fork) -> float:
        rel = self.fork_reliability(fork.name)
        return (1.0 - rel)

    def score_fork(self, fork: Fork) -> float:
        base = self._risk_level_to_score(fork.result.risk_level)
        break_risk = fork.result.aggregate_break_risk
        trend = self._trend_penalty(fork)
        rollback = self._rollback_bonus(fork)
        lane_penalty = self._lane_stability_penalty(fork)
        forecast_penalty = self._forecast_penalty()
        history_penalty = self._history_penalty(fork)

        ml_adj = 0.0
        if self.ml_predictor and self.ml_predictor.enabled:
            ml_adj = self.ml_predictor.predict_upgrade_risk({
                "risk_level": fork.result.risk_level,
                "aggregate_break_risk": fork.result.aggregate_break_risk,
            })

        return base + 0.1 * break_risk + 0.05 * trend + rollback + lane_penalty + forecast_penalty + history_penalty + ml_adj

    def rank_forks(self) -> List[Dict[str, Any]]:
        ranked = []
        for f in self.forks:
            score = self.score_fork(f)
            ranked.append({"fork": f, "score": score})
        ranked.sort(key=lambda x: x["score"])
        return ranked

    def merge_best_future(self) -> MergedFuture:
        ranked = self.rank_forks()
        if not ranked:
            return MergedFuture(
                base_fork="none",
                base_snapshot=0,
                merged_upgrades={},
                merged_risk_level="high",
                merged_score=math.inf,
                notes=["No forks available"]
            )
        best = ranked[0]["fork"]
        best_score = ranked[0]["score"]
        notes = [
            f"Base fork: {best.name}",
            f"Snapshot: {best.snapshot_ts}",
            f"Risk level: {best.result.risk_level}",
            f"Aggregate break risk: {best.result.aggregate_break_risk}",
            f"Rollback available: {best.result.rollback_available}",
            f"Routing forecast: {self.routing_forecast}",
            f"Backup forecast: {self.backup_forecast}",
        ]
        return MergedFuture(
            base_fork=best.name,
            base_snapshot=best.snapshot_ts,
            merged_upgrades=dict(best.upgrades),
            merged_risk_level=best.result.risk_level,
            merged_score=best_score,
            notes=notes
        )

    def merge_hybrid_future(self) -> MergedFuture:
        ranked = self.rank_forks()
        if not ranked:
            return self.merge_best_future()
        base_fork = ranked[0]["fork"]
        merged_upgrades = {}
        for entry in ranked:
            f = entry["fork"]
            for pkg, ver in f.upgrades.items():
                if pkg not in merged_upgrades:
                    merged_upgrades[pkg] = ver
        merged_risk = base_fork.result.risk_level
        merged_score = min(e["score"] for e in ranked)
        notes = [
            "Hybrid merge of multiple forks",
            f"Base fork: {base_fork.name}",
            f"Routing forecast: {self.routing_forecast}",
            f"Backup forecast: {self.backup_forecast}",
        ]
        return MergedFuture(
            base_fork=base_fork.name,
            base_snapshot=base_fork.snapshot_ts,
            merged_upgrades=merged_upgrades,
            merged_risk_level=merged_risk,
            merged_score=merged_score,
            notes=notes
        )

    def predict_upgrade_window(self) -> str:
        if not self.routing_forecast and not self.backup_forecast:
            return "unknown"
        if "high" in self.routing_forecast or "high" in self.backup_forecast:
            return "bad"
        if "medium" in self.routing_forecast or "medium" in self.backup_forecast:
            return "caution"
        return "good"

    def all_future_upgrades(self) -> Dict[str, Dict[str, str]]:
        forks_view = {}
        combined = {}
        for f in self.forks:
            forks_view[f.name] = dict(f.upgrades)
            for pkg, ver in f.upgrades.items():
                combined[pkg] = ver
        return {"forks": forks_view, "combined": combined}


# ============================================================
# Backup Manager Engine (multi-backup, persistent)
# ============================================================

class BackupRiskModel:
    def __init__(self):
        self.latency_ewma = {}
        self.failure_ewma = {}
        self.alpha = 0.3

    def observe(self, key: str, latency_ms: float, success: bool):
        self._update_ewma(self.latency_ewma, key, latency_ms)
        self._update_ewma(self.failure_ewma, key, 0.0 if success else 1.0)

    def _update_ewma(self, store, key, value):
        if key not in store:
            store[key] = (value, self.alpha)
        else:
            prev, a = store[key]
            store[key] = (a * value + (1 - a) * prev, a)

    def cone_for_path(self, key: str, steps: int = 5) -> Dict[str, List[Dict]]:
        lat = self.latency_ewma.get(key, (50.0, self.alpha))[0]
        fail = self.failure_ewma.get(key, (0.1, self.alpha))[0]

        cones = {"latency": [], "failure": []}
        for i in range(1, steps + 1):
            spread_lat = lat * 0.1 * i
            cones["latency"].append({
                "step": i,
                "center": lat,
                "lower": max(0.0, lat - spread_lat),
                "upper": lat + spread_lat
            })
            spread_fail = 0.1 * i
            center_fail = fail
            cones["failure"].append({
                "step": i,
                "center": center_fail,
                "lower": max(0.0, center_fail - spread_fail),
                "upper": min(1.0, center_fail + spread_fail)
            })
        return cones

    def overall_backup_risk_level(self) -> str:
        if not self.failure_ewma:
            return "low"
        avg_fail = sum(v for v, _ in self.failure_ewma.values()) / len(self.failure_ewma)
        if avg_fail < 0.2:
            return "low"
        if avg_fail < 0.5:
            return "medium"
        return "high"

    def forecast_backup_risk(self, key: str, horizon_steps: int = 3) -> List[str]:
        if key not in self.failure_ewma:
            return ["low"] * horizon_steps
        fail = self.failure_ewma[key][0]
        forecast = []
        for i in range(1, horizon_steps + 1):
            projected = min(1.0, fail + 0.05 * i)
            if projected < 0.2:
                forecast.append("low")
            elif projected < 0.5:
                forecast.append("medium")
            else:
                forecast.append("high")
        return forecast


class BackupManagerEngine:
    """
    Multi-backup engine:
    - Multiple backup paths
    - Persistent path list
    - Persistent active + last-used path
    - Per-path scoring
    - Active path
    - Mirror groups
    """

    def __init__(self, logger: Callable[[str], None], planner: FuturePlannerEngine, persistence: PersistenceLayer):
        self.logger = logger
        self.planner = planner
        self.persistence = persistence

        self.active_path: Optional[str] = None
        self.last_used_path: Optional[str] = None
        self.last_backup_ts: Optional[float] = None
        self.risk_model = BackupRiskModel()
        self.last_test_result: Dict[str, Any] = {}

        self.availability_log: List[Tuple[float, bool]] = []

        self.path_stats = defaultdict(lambda: {"tests": 0, "success": 0, "fail": 0})

        self.known_paths: List[str] = []
        self.mirror_groups: List[Dict[str, Any]] = []

        self._load_from_persistence()

    def _load_from_persistence(self):
        self.known_paths = self.persistence.get_backup_paths()
        self.mirror_groups = self.persistence.get_mirror_groups()
        persisted_active = self.persistence.get_active_backup_path()
        persisted_last = self.persistence.get_last_used_backup_path()

        if self.known_paths:
            self.logger(f"[Backup] Loaded known backup paths: {self.known_paths}")
        if self.mirror_groups:
            self.logger(f"[Backup] Loaded mirror groups: {self.mirror_groups}")

        # Guaranteed restore: prefer persisted active path if still present
        if persisted_active and persisted_active in self.known_paths:
            self.active_path = persisted_active
            self.logger(f"[Backup] Restored active backup path from persistence: {persisted_active}")
        elif persisted_last and persisted_last in self.known_paths:
            self.active_path = persisted_last
            self.logger(f"[Backup] Restored active backup path from last-used: {persisted_last}")
        elif self.known_paths:
            self.active_path = self.known_paths[0]
            self.logger(f"[Backup] No persisted active path; defaulting to first known: {self.active_path}")

        self.last_used_path = persisted_last

    def _save_paths(self):
        self.persistence.set_backup_paths(self.known_paths)

    def _save_mirror_groups(self):
        self.persistence.set_mirror_groups(self.mirror_groups)

    def _save_active_and_last(self):
        self.persistence.set_active_backup_path(self.active_path)
        self.persistence.set_last_used_backup_path(self.last_used_path)

    def add_backup_path(self, path: str):
        if path not in self.known_paths:
            self.known_paths.append(path)
            self.logger(f"[Backup] Added backup path: {path}")
            self._save_paths()

    def remove_backup_path(self, path: str):
        if path in self.known_paths:
            self.known_paths.remove(path)
            self.logger(f"[Backup] Removed backup path: {path}")
            self._save_paths()
        if path == self.active_path:
            self.active_path = None
            self.logger("[Backup] Active path removed; no active path now")
            self._save_active_and_last()

    def set_active_path(self, path: str):
        if path not in self.known_paths:
            self.add_backup_path(path)
        self.active_path = path
        self.last_used_path = path
        self.logger(f"[Backup] Active backup path set to: {path}")
        self._save_paths()
        self._save_active_and_last()

    def get_all_paths(self) -> List[str]:
        return list(self.known_paths)

    def auto_detect_drives(self) -> List[Dict[str, Any]]:
        drives = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            if os.path.exists(root):
                try:
                    total, used, free = shutil.disk_usage(root)
                except Exception:
                    total = used = free = 0
                drives.append({
                    "letter": letter,
                    "root": root,
                    "label": "",
                    "type": "Fixed/Unknown",
                    "free_gb": round(free / (1024 ** 3), 2),
                    "total_gb": round(total / (1024 ** 3), 2) if total else 0.0,
                })
        return drives

    def test_backup_path(self, path: str) -> Dict[str, Any]:
        start = time.time()
        success = False
        error_msg = ""
        latency_ms = 0.0
        try:
            if not os.path.isdir(path):
                raise RuntimeError("Path is not a directory or not accessible")
            fd, temp_path = tempfile.mkstemp(prefix="backup_test_", dir=path)
            os.close(fd)
            with open(temp_path, "wb") as f:
                f.write(b"PROMETHEUS_BACKUP_TEST")
            with open(temp_path, "rb") as f:
                _ = f.read()
            os.remove(temp_path)
            success = True
        except Exception as e:
            error_msg = str(e)
        finally:
            latency_ms = (time.time() - start) * 1000.0

        self.risk_model.observe(path, latency_ms, success)
        self.availability_log.append((time.time(), success))
        stats = self.path_stats[path]
        stats["tests"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1

        self.last_test_result = {
            "path": path,
            "success": success,
            "latency_ms": latency_ms,
            "error": error_msg,
            "timestamp": time.time()
        }

        if not success:
            self.logger(f"[Backup] Test FAILED for {path}: {error_msg} (latency={latency_ms:.1f} ms)")
            evt = CodeLaneEvent(
                timestamp=time.time(),
                func_name="backup_path",
                event_type="fallback_used",
                details=f"Backup path failure at {path}: {error_msg}"
            )
            self.planner.record_code_lane_event(evt)
        else:
            self.logger(f"[Backup] Test OK for {path} (latency={latency_ms:.1f} ms)")

        return self.last_test_result

    def test_all_paths(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for p in self.known_paths:
            results[p] = self.test_backup_path(p)
        return results

    def record_backup_success(self, path: Optional[str] = None):
        if path is None:
            path = self.active_path
        if path:
            self.last_backup_ts = time.time()
            self.risk_model.observe(path, 10.0, True)

    def backup_risk_level(self) -> str:
        return self.risk_model.overall_backup_risk_level()

    def backup_cone_for_active(self) -> Dict[str, List[Dict]]:
        if not self.active_path:
            return {"latency": [], "failure": []}
        return self.risk_model.cone_for_path(self.active_path)

    def forecast_backup_risk_for_active(self, horizon_steps: int = 3) -> List[str]:
        if not self.active_path:
            return ["low"] * horizon_steps
        return self.risk_model.forecast_backup_risk(self.active_path, horizon_steps)

    def forecast_backup_risk_for_path(self, path: str, horizon_steps: int = 3) -> List[str]:
        return self.risk_model.forecast_backup_risk(path, horizon_steps)

    def predict_stable_backup_window(self, horizon_seconds: int = 3600) -> bool:
        cutoff = time.time() - horizon_seconds
        recent = [s for (ts, s) in self.availability_log if ts >= cutoff]
        if not recent:
            return True
        success_rate = sum(1 for s in recent if s) / len(recent)
        return success_rate >= 0.7

    def path_score(self, path: str) -> float:
        stats = self.path_stats[path]
        t, s, f = stats["tests"], stats["success"], stats["fail"]
        if t == 0:
            return 0.0
        reliability = s / t
        latency = self.risk_model.latency_ewma.get(path, (50.0, 0.0))[0]
        latency_factor = max(0.1, min(1.0, 200.0 / (latency + 1.0)))
        return reliability * latency_factor

    def best_known_path(self) -> Optional[str]:
        if not self.path_stats:
            return None
        return max(self.path_stats.keys(), key=self.path_score)

    def auto_switch_if_needed(self, risk_threshold: str = "high"):
        current_level = self.backup_risk_level()
        order = {"low": 0, "medium": 1, "high": 2}
        if order[current_level] < order[risk_threshold]:
            return
        best = self.best_known_path()
        if best and best != self.active_path:
            self.logger(f"[Backup] Auto-switching backup path from {self.active_path} to {best} due to high risk")
            self.set_active_path(best)

    # --- Mirror groups ---

    def create_mirror_group(self, name: str, paths: List[str]):
        for p in paths:
            if p not in self.known_paths:
                self.add_backup_path(p)
        group = {"name": name, "paths": list(paths)}
        self.mirror_groups.append(group)
        self.logger(f"[Backup] Created mirror group '{name}' with paths: {paths}")
        self._save_mirror_groups()

    def add_path_to_group(self, group_name: str, path: str):
        for g in self.mirror_groups:
            if g["name"] == group_name:
                if path not in g["paths"]:
                    g["paths"].append(path)
                    self.logger(f"[Backup] Added {path} to mirror group '{group_name}'")
                    self._save_mirror_groups()
                return

    def remove_path_from_group(self, group_name: str, path: str):
        for g in self.mirror_groups:
            if g["name"] == group_name and path in g["paths"]:
                g["paths"].remove(path)
                self.logger(f"[Backup] Removed {path} from mirror group '{group_name}'")
                self._save_mirror_groups()
                return

    def get_mirror_groups(self) -> List[Dict[str, Any]]:
        return list(self.mirror_groups)


# ============================================================
# Area of Opportunity Engine
# ============================================================

@dataclass
class Opportunity:
    area: str
    subsystem: str
    score: float
    impact: float
    effort: float
    notes: str


class AreaOfOpportunityEngine:
    """
    Scans the whole system and surfaces ranked “Areas of Opportunity”.
    """

    def __init__(self):
        self.opportunities: List[Opportunity] = []

    def _level_to_gap(self, level: str) -> float:
        if level == "low":
            return 0.1
        if level == "medium":
            return 0.5
        return 1.0

    def _compute_score(self, impact: float, effort: float, gap: float) -> float:
        if effort <= 0:
            effort = 0.1
        return impact * gap / effort

    def update_from_system(
        self,
        routing_level: str,
        routing_forecast: List[str],
        backup_level: str,
        backup_forecast: List[str],
        lane_risk: str,
        heatmap: Dict[str, Dict[str, int]],
        clusters: Dict[int, Dict[str, Any]],
        drive_stats: Dict[str, Dict[str, int]],
        fork_scores: List[Dict[str, Any]],
    ):
        self.opportunities.clear()

        routing_gap = self._level_to_gap(routing_level)
        if routing_gap > 0.1 or "high" in routing_forecast:
            impact = 0.9
            effort = 0.5
            score = self._compute_score(impact, effort, routing_gap)
            notes = f"Routing risk={routing_level}, forecast={routing_forecast}"
            self.opportunities.append(
                Opportunity(
                    area="Routing Hardening",
                    subsystem="Routing",
                    score=score,
                    impact=impact,
                    effort=effort,
                    notes=notes,
                )
            )

        backup_gap = self._level_to_gap(backup_level)
        if backup_gap > 0.1 or "high" in backup_forecast:
            impact = 0.85
            effort = 0.4
            score = self._compute_score(impact, effort, backup_gap)
            notes = f"Backup risk={backup_level}, forecast={backup_forecast}"
            self.opportunities.append(
                Opportunity(
                    area="Backup Path Optimization",
                    subsystem="Backup",
                    score=score,
                    impact=impact,
                    effort=effort,
                    notes=notes,
                )
            )

        lane_gap = self._level_to_gap(lane_risk)
        if lane_gap > 0.1:
            impact = 0.8
            effort = 0.6
            score = self._compute_score(impact, effort, lane_gap)
            notes = f"Lane risk={lane_risk}"
            self.opportunities.append(
                Opportunity(
                    area="Code Lane Stabilization",
                    subsystem="Execution / Planner",
                    score=score,
                    impact=impact,
                    effort=effort,
                    notes=notes,
                )
            )

        for func_name, stats in heatmap.items():
            failures = stats["failures"]
            reroutes = stats["reroutes"]
            if failures + reroutes < 3:
                continue
            impact = min(1.0, 0.2 + 0.1 * (failures + reroutes))
            effort = 0.7
            gap = 0.7
            score = self._compute_score(impact, effort, gap)
            notes = f"{failures} failures, {reroutes} reroutes"
            self.opportunities.append(
                Opportunity(
                    area=f"Refactor / Harden {func_name}",
                    subsystem="Bypass / Code",
                    score=score,
                    impact=impact,
                    effort=effort,
                    notes=notes,
                )
            )

        for cid, info in clusters.items():
            count = info["count"]
            if count < 3:
                continue
            impact = min(1.0, 0.3 + 0.05 * count)
            effort = 0.8
            gap = 0.8
            score = self._compute_score(impact, effort, gap)
            notes = f"{count} failures in cluster #{cid}"
            self.opportunities.append(
                Opportunity(
                    area=f"Harden Cluster #{cid}",
                    subsystem="Bypass / Patterns",
                    score=score,
                    impact=impact,
                    effort=effort,
                    notes=notes,
                )
            )

        for path, st in drive_stats.items():
            tests = st["tests"]
            if tests == 0:
                continue
            success = st["success"]
            reliability = success / tests
            if reliability < 0.8:
                impact = 0.75
                effort = 0.5
                gap = 1.0 - reliability
                score = self._compute_score(impact, effort, gap)
                notes = f"Drive {path} reliability={reliability:.2f}, tests={tests}"
                self.opportunities.append(
                    Opportunity(
                        area=f"Improve / Replace Drive {path}",
                        subsystem="Backup / Storage",
                        score=score,
                        impact=impact,
                        effort=effort,
                        notes=notes,
                    )
                )

        for entry in fork_scores:
            f: Fork = entry["fork"]
            score_val = entry["score"]
            if score_val > 3.0:
                impact = 0.9
                effort = 0.9
                gap = 0.9
                score = self._compute_score(impact, effort, gap)
                notes = f"Fork {f.name} has high risk score={score_val:.2f}"
                self.opportunities.append(
                    Opportunity(
                        area=f"De-risk Fork {f.name}",
                        subsystem="Planner / Upgrades",
                        score=score,
                        impact=impact,
                        effort=effort,
                        notes=notes,
                    )
                )

        self.opportunities.sort(key=lambda o: o.score, reverse=True)


# ============================================================
# GUI Widgets
# ============================================================

class HazardRadarWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_level = "low"
        self.metrics = {
            "CPU": "low",
            "DNS": "low",
            "Routes": "low",
            "Env": "low",
            "Code": "low",
            "Backup": "low"
        }
        self.setMinimumHeight(200)

    def update_risk(self, risk_snapshot):
        self.current_level = risk_snapshot.get("level", "low")
        metrics = risk_snapshot.get("metrics", {})
        for k in self.metrics.keys():
            self.metrics[k] = metrics.get(k, "low")
        self.update()

    def _color_for_level(self, level):
        if level == "low":
            return QtGui.QColor(80, 200, 120)
        if level == "medium":
            return QtGui.QColor(255, 200, 0)
        return QtGui.QColor(220, 80, 80)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 4

        color = self._color_for_level(self.current_level)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.drawText(rect, QtCore.Qt.AlignCenter, f"Hazard: {self.current_level.upper()}")

        band_height = 18
        y = rect.bottom() - len(self.metrics) * band_height - 10
        for name, level in self.metrics.items():
            band_rect = QtCore.QRect(rect.left() + 10, y, rect.width() - 20, band_height - 4)
            painter.setBrush(QtGui.QBrush(self._color_for_level(level)))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRect(band_rect)
            painter.setPen(QtGui.QPen(QtCore.Qt.black))
            painter.drawText(band_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f" {name}: {level}")
            y += band_height


class ForecastStripWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.routing_forecast: List[str] = []
        self.backup_forecast: List[str] = []
        self.lane_forecast: List[str] = []
        self.setMinimumHeight(60)

    def update_forecast(self, routing: List[str], backup: List[str], lane: List[str]):
        self.routing_forecast = routing
        self.backup_forecast = backup
        self.lane_forecast = lane
        self.update()

    def _color_for_level(self, level):
        if level == "low":
            return QtGui.QColor(80, 200, 120)
        if level == "medium":
            return QtGui.QColor(255, 200, 0)
        return QtGui.QColor(220, 80, 80)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        rows = [
            ("Routing", self.routing_forecast),
            ("Backup", self.backup_forecast),
            ("Lane", self.lane_forecast),
        ]
        if not any(rows):
            return

        row_height = height // len(rows)
        for i, (label, forecast) in enumerate(rows):
            y = rect.top() + i * row_height
            painter.setPen(QtGui.QPen(QtCore.Qt.black))
            painter.drawText(rect.left() + 5, y + 15, label)
            if not forecast:
                continue
            step_width = max(10, (width - 80) // len(forecast))
            x = rect.left() + 70
            for lvl in forecast:
                color = self._color_for_level(lvl)
                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRect(x, y + 5, step_width - 4, row_height - 10)
                x += step_width


class BypassHeatmapWidget(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Function", "Failures", "Reroutes"])
        self.horizontalHeader().setStretchLastSection(True)

    def update_heatmap(self, data: Dict[str, Dict[str, int]]):
        self.setRowCount(len(data))
        for row, (name, stats) in enumerate(sorted(data.items(), key=lambda x: x[1]["failures"], reverse=True)):
            failures = stats["failures"]
            reroutes = stats["reroutes"]
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(str(failures)))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(reroutes)))

            bg = QtGui.QColor(255, 200 - min(200, failures * 10), 200 - min(200, reroutes * 10))
            for col in range(3):
                item = self.item(row, col)
                if item:
                    item.setBackground(bg)


class FingerprintClustersWidget(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Cluster ID", "Count", "Example Fingerprints"])
        self.horizontalHeader().setStretchLastSection(True)

    def update_clusters(self, clusters: Dict[int, Dict[str, Any]]):
        self.setRowCount(len(clusters))
        for row, (cid, info) in enumerate(sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True)):
            count = info["count"]
            examples = info["examples"]
            ex_strs = []
            for func_name, exc_type, ctx_key in examples:
                ex_strs.append(f"{func_name}/{exc_type}")
            ex_text = ", ".join(ex_strs)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cid)))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(str(count)))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(ex_text))


class CodeLaneTimelineWidget(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Time", "Function", "Event", "Details"])
        self.horizontalHeader().setStretchLastSection(True)

    def update_events(self, events: List[CodeLaneEvent]):
        self.setRowCount(len(events))
        for i, e in enumerate(events):
            t_str = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(t_str))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(e.func_name))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(e.event_type))
            self.setItem(i, 3, QtWidgets.QTableWidgetItem(e.details))


class LogPanel(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def log(self, message: str):
        self.append(message)


class FuturePlannerTab(QtWidgets.QWidget):
    active_lane_changed = QtCore.pyqtSignal(str)

    def __init__(self, planner: FuturePlannerEngine, log_panel: LogPanel, parent=None):
        super().__init__(parent)
        self.planner = planner
        self.log_panel = log_panel

        self.fork_list = QtWidgets.QTableWidget()
        self.fork_list.setColumnCount(5)
        self.setMinimumHeight(200)
        self.fork_list.setHorizontalHeaderLabels(
            ["Fork", "Snapshot", "Risk", "Break Risk", "Score"]
        )

        self.upgrades_text = QtWidgets.QTextEdit()
        self.upgrades_text.setReadOnly(True)

        self.merged_text = QtWidgets.QTextEdit()
        self.merged_text.setReadOnly(True)

        self.refresh_button = QtWidgets.QPushButton("Refresh Futures")
        self.lane_button = QtWidgets.QPushButton("Set Selected Fork as Active Lane")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Forked Futures"))
        layout.addWidget(self.fork_list)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.lane_button)
        layout.addWidget(QtWidgets.QLabel("All Future Upgrades (per fork + combined)"))
        layout.addWidget(self.upgrades_text)
        layout.addWidget(QtWidgets.QLabel("Merged Recommended Future"))
        layout.addWidget(self.merged_text)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.refresh_view)
        self.lane_button.clicked.connect(self.set_active_lane)

    def refresh_view(self):
        ranked = self.planner.rank_forks()
        self.fork_list.setRowCount(len(ranked))

        for row, entry in enumerate(ranked):
            f: Fork = entry["fork"]
            score = entry["score"]
            self.fork_list.setItem(row, 0, QtWidgets.QTableWidgetItem(f.name))
            self.fork_list.setItem(row, 1, QtWidgets.QTableWidgetItem(str(f.snapshot_ts)))
            self.fork_list.setItem(row, 2, QtWidgets.QTableWidgetItem(f.result.risk_level))
            self.fork_list.setItem(row, 3, QtWidgets.QTableWidgetItem(str(f.result.aggregate_break_risk)))
            self.fork_list.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{score:.2f}"))

        upgrades = self.planner.all_future_upgrades()
        lines = ["Per fork:"]
        for fork_name, up in upgrades["forks"].items():
            lines.append(f"  {fork_name}:")
            for pkg, ver in up.items():
                lines.append(f"    {pkg} -> {ver}")
        lines.append("")
        lines.append("Combined view:")
        for pkg, ver in upgrades["combined"].items():
            lines.append(f"  {pkg} -> {ver}")
        self.upgrades_text.setText("\n".join(lines))

        merged = self.planner.merge_best_future()
        hybrid = self.planner.merge_hybrid_future()
        m_lines = [
            f"Base fork: {merged.base_fork}",
            f"Base snapshot: {merged.base_snapshot}",
            f"Merged risk level: {merged.merged_risk_level}",
            f"Merged score: {merged.merged_score:.2f}",
            "",
            "Merged upgrades:"
        ]
        for pkg, ver in merged.merged_upgrades.items():
            m_lines.append(f"  {pkg} -> {ver}")
        m_lines.append("")
        m_lines.append("Hybrid merged future (multi-fork):")
        for pkg, ver in hybrid.merged_upgrades.items():
            m_lines.append(f"  {pkg} -> {ver}")
        m_lines.append("")
        m_lines.append("Notes:")
        m_lines.extend(f"  - {n}" for n in merged.notes)
        self.merged_text.setText("\n".join(m_lines))

    def set_active_lane(self):
        row = self.fork_list.currentRow()
        if row < 0:
            return
        fork_name = self.fork_list.item(row, 0).text()
        self.log_panel.log(f"[Lane] Active lane set to: {fork_name}")
        self.active_lane_changed.emit(fork_name)


class BackupManagerTab(QtWidgets.QWidget):
    def __init__(self, engine: BackupManagerEngine, log_panel: LogPanel, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.log_panel = log_panel

        self.local_button = QtWidgets.QPushButton("Add Local Backup Folder")
        self.smb_button = QtWidgets.QPushButton("Add SMB / Network Backup Folder")
        self.auto_button = QtWidgets.QPushButton("Auto-Detect Drives")
        self.test_active_button = QtWidgets.QPushButton("Test Active Backup Path")
        self.test_all_button = QtWidgets.QPushButton("Test All Backup Paths")

        self.active_path_label = QtWidgets.QLabel("Active Path: (none)")
        self.last_used_label = QtWidgets.QLabel("Last Used Path: (none)")
        self.last_test_label = QtWidgets.QLabel("Last Test: (none)")
        self.last_backup_label = QtWidgets.QLabel("Last Backup: (none)")
        self.risk_label = QtWidgets.QLabel("Backup Risk: low")
        self.window_label = QtWidgets.QLabel("Predicted Stable Backup Window: likely")

        self.drive_table = QtWidgets.QTableWidget()
        self.drive_table.setColumnCount(6)
        self.drive_table.setHorizontalHeaderLabels(
            ["Letter", "Root", "Type", "Free (GB)", "Total (GB)", "Health"]
        )
        self.drive_table.horizontalHeader().setStretchLastSection(True)

        self.drive_scores_table = QtWidgets.QTableWidget()
        self.drive_scores_table.setColumnCount(3)
        self.drive_scores_table.setHorizontalHeaderLabels(["Path", "Score", "Tests"])
        self.drive_scores_table.horizontalHeader().setStretchLastSection(True)

        self.paths_table = QtWidgets.QTableWidget()
        self.paths_table.setColumnCount(4)
        self.paths_table.setHorizontalHeaderLabels(["Path", "Score", "Tests", "Set Active"])
        self.paths_table.horizontalHeader().setStretchLastSection(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Backup Drive Selection"))
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.local_button)
        btn_row.addWidget(self.smb_button)
        btn_row.addWidget(self.auto_button)
        layout.addLayout(btn_row)

        layout.addWidget(QtWidgets.QLabel("Detected Drives"))
        layout.addWidget(self.drive_table)

        layout.addWidget(QtWidgets.QLabel("Known Backup Paths"))
        layout.addWidget(self.paths_table)

        layout.addWidget(QtWidgets.QLabel("Drive Scores (learned reliability + latency)"))
        layout.addWidget(self.drive_scores_table)

        layout.addWidget(QtWidgets.QLabel("Active Backup Path"))
        layout.addWidget(self.active_path_label)
        layout.addWidget(self.last_used_label)
        layout.addWidget(self.last_backup_label)
        layout.addWidget(self.last_test_label)
        layout.addWidget(self.risk_label)
        layout.addWidget(self.window_label)

        test_row = QtWidgets.QHBoxLayout()
        test_row.addWidget(self.test_active_button)
        test_row.addWidget(self.test_all_button)
        layout.addLayout(test_row)

        self.setLayout(layout)

        self.local_button.clicked.connect(self.add_local_folder)
        self.smb_button.clicked.connect(self.add_smb_folder)
        self.auto_button.clicked.connect(self.auto_detect_drives)
        self.test_active_button.clicked.connect(self.test_active_path)
        self.test_all_button.clicked.connect(self.test_all_paths)

        self.refresh_status()

    def refresh_status(self):
        ap = self.engine.active_path or "(none)"
        lp = self.engine.last_used_path or "(none)"
        self.active_path_label.setText(f"Active Path: {ap}")
        self.last_used_label.setText(f"Last Used Path: {lp}")

        if self.engine.last_backup_ts:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.engine.last_backup_ts))
            self.last_backup_label.setText(f"Last Backup: {ts_str}")
        else:
            self.last_backup_label.setText("Last Backup: (none)")

        if self.engine.last_test_result:
            r = self.engine.last_test_result
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["timestamp"]))
            status = "OK" if r["success"] else f"FAILED: {r['error']}"
            self.last_test_label.setText(
                f"Last Test: {status} at {ts_str} (latency={r['latency_ms']:.1f} ms)"
            )
        else:
            self.last_test_label.setText("Last Test: (none)")

        risk = self.engine.backup_risk_level()
        self.risk_label.setText(f"Backup Risk: {risk}")

        stable = self.engine.predict_stable_backup_window()
        self.window_label.setText(
            "Predicted Stable Backup Window: likely" if stable else "Predicted Stable Backup Window: uncertain"
        )

        stats = self.engine.path_stats
        self.drive_scores_table.setRowCount(len(stats))
        for row, (path, st) in enumerate(stats.items()):
            score = self.engine.path_score(path)
            tests = st["tests"]
            self.drive_scores_table.setItem(row, 0, QtWidgets.QTableWidgetItem(path))
            self.drive_scores_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{score:.3f}"))
            self.drive_scores_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(tests)))

        paths = self.engine.get_all_paths()
        self.paths_table.setRowCount(len(paths))
        for row, p in enumerate(paths):
            score = self.engine.path_score(p)
            tests = self.engine.path_stats[p]["tests"]
            self.paths_table.setItem(row, 0, QtWidgets.QTableWidgetItem(p))
            self.paths_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{score:.3f}"))
            self.paths_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(tests)))
            btn = QtWidgets.QPushButton("Set Active")
            btn.clicked.connect(lambda _, path=p: self.set_active_from_table(path))
            self.paths_table.setCellWidget(row, 3, btn)

    def set_active_from_table(self, path: str):
        self.engine.set_active_path(path)
        self.log_panel.log(f"[Backup] Active path set from table: {path}")
        self.refresh_status()

    def add_local_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add Local Backup Folder")
        if path:
            self.engine.add_backup_path(path)
            self.engine.set_active_path(path)
            self.refresh_status()

    def add_smb_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add SMB / Network Backup Folder")
        if path:
            self.engine.add_backup_path(path)
            self.engine.set_active_path(path)
            self.refresh_status()

    def auto_detect_drives(self):
        drives = self.engine.auto_detect_drives()
        self.drive_table.setRowCount(len(drives))
        for row, d in enumerate(drives):
            self.drive_table.setItem(row, 0, QtWidgets.QTableWidgetItem(d["letter"]))
            self.drive_table.setItem(row, 1, QtWidgets.QTableWidgetItem(d["root"]))
            self.drive_table.setItem(row, 2, QtWidgets.QTableWidgetItem(d["type"]))
            self.drive_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(d["free_gb"])))
            self.drive_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(d["total_gb"])))
            health = "OK" if d["free_gb"] > 1 else "LOW SPACE"
            self.drive_table.setItem(row, 5, QtWidgets.QTableWidgetItem(health))
        self.log_panel.log("[Backup] Auto-detected drives updated")

    def test_active_path(self):
        if not self.engine.active_path:
            self.log_panel.log("[Backup] No active path set to test")
            return
        result = self.engine.test_backup_path(self.engine.active_path)
        self.refresh_status()
        if not result["success"]:
            self.log_panel.log(f"[Backup] Active path test FAILED: {result['error']}")
        else:
            self.log_panel.log("[Backup] Active path test OK")

    def test_all_paths(self):
        if not self.engine.get_all_paths():
            self.log_panel.log("[Backup] No backup paths to test")
            return
        results = self.engine.test_all_paths()
        self.refresh_status()
        for p, r in results.items():
            if r["success"]:
                self.log_panel.log(f"[Backup] Test OK for {p}")
            else:
                self.log_panel.log(f"[Backup] Test FAILED for {p}: {r['error']}")


class OpportunitiesTab(QtWidgets.QWidget):
    def __init__(self, engine: AreaOfOpportunityEngine, parent=None):
        super().__init__(parent)
        self.engine = engine

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Area", "Subsystem", "Score", "Impact", "Effort", "Notes"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Ranked Areas of Opportunity"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh_view(self):
        opps = self.engine.opportunities
        self.table.setRowCount(len(opps))
        for row, o in enumerate(opps):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(o.area))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(o.subsystem))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{o.score:.3f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{o.impact:.2f}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{o.effort:.2f}"))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(o.notes))


# ============================================================
# PrometheusCore
# ============================================================

class PrometheusCore:
    def __init__(self,
                 bypass_engine: ExecutionBypassEngine,
                 probe: Optional[NetworkProbe] = None,
                 upgrade_engine: Optional[UpgradeEngine] = None):
        self._tick = 0
        self.routing_brain = UnifiedAccessRoadController(probe=probe)
        self.network_watcher = NetworkWatcher(self.routing_brain)
        self.bypass_engine = bypass_engine
        self.upgrade_engine = upgrade_engine or UpgradeEngine(logger=print)
        self.persistence: Optional[PersistenceLayer] = None

        self.apply_upgrade_plan = self.bypass_engine.wrap(
            self._apply_upgrade_plan_impl,
            func_name="apply_upgrade_plan",
            fallback=self._apply_upgrade_plan_fallback
        )

    def _dummy_cone(self, base=10):
        return [
            {"step": i, "center": base, "lower": base - 0.5 * i, "upper": base + 0.5 * i}
            for i in range(1, 6)
        ]

    def simulate_network_activity(self):
        targets = [
            ("1.1.1.1", 20, True),
            ("8.8.8.8", 40, True),
            ("github.com", 80, True),
            ("unstable.example.com", 150, False),
        ]
        dest, lat, ok = targets[self._tick % len(targets)]

        if self.routing_brain.probe:
            probe_result = self.routing_brain.probe.ping(dest)
            if probe_result["success"]:
                lat = probe_result["latency_ms"]
                ok = probe_result["loss"] < 0.5

        self.network_watcher.on_connection(dest, lat, ok)
        if self.persistence:
            self.persistence.append_routing_sample(dest, lat, ok)
        self.routing_brain.preload_time_of_day_top(3)

    def get_current_forks(self) -> List[Fork]:
        self._tick += 1
        self.simulate_network_activity()
        t = 1710000000 + self._tick

        routing_risk = self.routing_brain.routing_risk_level()
        routing_penalty = {"low": 0, "medium": 2, "high": 5}[routing_risk]

        f1 = Fork(
            name="Fork A",
            snapshot_ts=t,
            upgrades={"pkg1": "2.0.0", "pkg2": "1.5.0"},
            result=ForkResult(
                risk_level="medium",
                aggregate_break_risk=4 + routing_penalty,
                risk_cones={"metric1": self._dummy_cone(10)},
                rollback_available=True
            )
        )

        f2 = Fork(
            name="Fork B",
            snapshot_ts=t + 100,
            upgrades={"pkg1": "2.1.0", "pkg3": "0.9.0"},
            result=ForkResult(
                risk_level="high",
                aggregate_break_risk=9 + routing_penalty,
                risk_cones={"metric1": self._dummy_cone(12)},
                rollback_available=False
            )
        )

        f3 = Fork(
            name="Fork C",
            snapshot_ts=t + 200,
            upgrades={"pkg4": "3.0.0"},
            result=ForkResult(
                risk_level="low",
                aggregate_break_risk=1 + routing_penalty,
                risk_cones={"metric1": self._dummy_cone(8)},
                rollback_available=True
            )
        )

        return [f1, f2, f3]

    def get_current_routing_risk_level(self) -> str:
        return self.routing_brain.routing_risk_level()

    def forecast_routing_risk(self, horizon_steps: int = 3) -> List[str]:
        return self.routing_brain.forecast_routing_risk(horizon_steps)

    def _apply_upgrade_plan_impl(self, base_snapshot: int, upgrades: Dict[str, str]):
        self.upgrade_engine.apply_plan(base_snapshot, upgrades)

    def _apply_upgrade_plan_fallback(self, base_snapshot: int, upgrades: Dict[str, str]):
        print(f"[PrometheusCore] FALLBACK: Queuing upgrade plan instead of applying immediately: {upgrades}")


# ============================================================
# Main Window
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,
                 prometheus: PrometheusCore,
                 bypass_engine: ExecutionBypassEngine,
                 planner: FuturePlannerEngine,
                 backup_engine: BackupManagerEngine,
                 opportunities_engine: AreaOfOpportunityEngine,
                 persistence: PersistenceLayer,
                 redundant_backup: RedundantBackupManager,
                 sync_manager: NodeSyncManager,
                 plugins: PluginRegistry,
                 ml_predictor: MLPredictor,
                 gui_logger: ThreadSafeLogger,
                 parent=None):
        super().__init__(parent)
        self.prometheus = prometheus
        self.bypass_engine = bypass_engine
        self.future_planner_engine = planner
        self.backup_engine = backup_engine
        self.opportunities_engine = opportunities_engine
        self.persistence = persistence
        self.redundant_backup = redundant_backup
        self.sync_manager = sync_manager
        self.plugins = plugins
        self.ml_predictor = ml_predictor
        self.gui_logger = gui_logger

        self.setWindowTitle("Prometheus Predictive Platform (Planner + Routing + Bypass + Multi-Backup + Opportunities)")

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.log_panel = LogPanel()
        self.gui_logger.message_emitted.connect(self.log_panel.log)

        self.bypass_engine.attach_planner(self.future_planner_engine)
        self.bypass_engine.attach_context_provider(self._current_context_snapshot)

        self.hazard_radar = HazardRadarWidget()
        self.forecast_strip = ForecastStripWidget()
        self.future_planner_tab = FuturePlannerTab(self.future_planner_engine, self.log_panel)
        self.bypass_heatmap = BypassHeatmapWidget()
        self.fingerprint_clusters = FingerprintClustersWidget()
        self.code_lane_timeline = CodeLaneTimelineWidget()
        self.backup_tab = BackupManagerTab(self.backup_engine, self.log_panel)
        self.opportunities_tab = OpportunitiesTab(self.opportunities_engine)

        self.future_planner_tab.active_lane_changed.connect(self.on_active_lane_changed)
        self.active_lane = None

        self.bypass_indicator = QtWidgets.QLabel("Bypass: OFF")
        self.bypass_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self._update_bypass_indicator()

        self.upgrade_window_indicator = QtWidgets.QLabel("Upgrade Window: unknown")
        self.upgrade_window_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.upgrade_window_indicator.setStyleSheet("color: white; background-color: #7f8c8d; font-weight: bold;")

        self.intent_log = QtWidgets.QTextEdit()
        self.intent_log.setReadOnly(True)

        # --- Future Planner main tab ---
        planner_container = QtWidgets.QWidget()
        planner_layout = QtWidgets.QVBoxLayout()
        planner_layout.addWidget(self.hazard_radar)
        planner_layout.addWidget(QtWidgets.QLabel("Short-Horizon Forecast (Routing / Backup / Lane)"))
        planner_layout.addWidget(self.forecast_strip)
        planner_layout.addWidget(self.future_planner_tab)
        planner_layout.addWidget(self.upgrade_window_indicator)
        planner_container.setLayout(planner_layout)

        # --- Heatmap tab ---
        heatmap_container = QtWidgets.QWidget()
        heatmap_layout = QtWidgets.QVBoxLayout()
        heatmap_layout.addWidget(QtWidgets.QLabel("Execution Bypass Heatmap"))
        heatmap_layout.addWidget(self.bypass_heatmap)
        heatmap_container.setLayout(heatmap_layout)

        # --- Fingerprint Clusters tab ---
        clusters_container = QtWidgets.QWidget()
        clusters_layout = QtWidgets.QVBoxLayout()
        clusters_layout.addWidget(QtWidgets.QLabel("Fingerprint Clusters"))
        clusters_layout.addWidget(self.fingerprint_clusters)
        clusters_container.setLayout(clusters_layout)

        # --- Code Lane Timeline tab ---
        timeline_container = QtWidgets.QWidget()
        timeline_layout = QtWidgets.QVBoxLayout()
        timeline_layout.addWidget(QtWidgets.QLabel("Code Lane Timeline"))
        timeline_layout.addWidget(self.code_lane_timeline)
        timeline_container.setLayout(timeline_layout)

        # --- Self-Driving Intent tab ---
        intent_container = QtWidgets.QWidget()
        intent_layout = QtWidgets.QVBoxLayout()
        intent_layout.addWidget(QtWidgets.QLabel("Self-Driving Intent Log"))
        intent_layout.addWidget(self.intent_log)
        intent_container.setLayout(intent_layout)

        # --- Bypass Log tab ---
        bypass_log_container = QtWidgets.QWidget()
        bypass_log_layout = QtWidgets.QVBoxLayout()
        bypass_log_layout.addWidget(QtWidgets.QLabel("Self-Driving & Bypass Log"))
        bypass_log_layout.addWidget(self.bypass_indicator)
        bypass_log_layout.addWidget(self.log_panel)
        bypass_log_container.setLayout(bypass_log_layout)

        # --- Add top-level tabs ---
        self.tabs.addTab(planner_container, "Future Planner")
        self.tabs.addTab(self.backup_tab, "Backup Manager")
        self.tabs.addTab(self.opportunities_tab, "Areas of Opportunity")
        self.tabs.addTab(heatmap_container, "Heatmap")
        self.tabs.addTab(clusters_container, "Fingerprint Clusters")
        self.tabs.addTab(timeline_container, "Code Lane Timeline")
        self.tabs.addTab(intent_container, "Self-Driving Intent")
        self.tabs.addTab(bypass_log_container, "Bypass Log")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(3000)

        self.self_driving_enabled = False
        self.self_driving_action = QtWidgets.QAction("Self-Driving Mode", self)
        self.self_driving_action.setCheckable(True)
        self.self_driving_action.toggled.connect(self.set_self_driving)
        self.menuBar().addAction(self.self_driving_action)

        self.bypass_action = QtWidgets.QAction("Execution Bypass Mode", self)
        self.bypass_action.setCheckable(True)
        self.bypass_action.setChecked(True)
        self.bypass_action.toggled.connect(self.set_bypass_mode)
        self.menuBar().addAction(self.bypass_action)

    def _update_bypass_indicator(self):
        if self.bypass_engine.bypass_enabled:
            self.bypass_indicator.setText("Bypass: ON")
            self.bypass_indicator.setStyleSheet("color: white; background-color: #d35400; font-weight: bold;")
        else:
            self.bypass_indicator.setText("Bypass: OFF")
            self.bypass_indicator.setStyleSheet("color: white; background-color: #7f8c8d; font-weight: bold;")

    def _update_upgrade_window_indicator(self, window_state: str):
        if window_state == "good":
            text = "Upgrade Window: GOOD"
            style = "color: white; background-color: #27ae60; font-weight: bold;"
        elif window_state == "caution":
            text = "Upgrade Window: CAUTION"
            style = "color: black; background-color: #f1c40f; font-weight: bold;"
        elif window_state == "bad":
            text = "Upgrade Window: BAD"
            style = "color: white; background-color: #c0392b; font-weight: bold;"
        else:
            text = "Upgrade Window: UNKNOWN"
            style = "color: white; background-color: #7f8c8d; font-weight: bold;"
        self.upgrade_window_indicator.setText(text)
        self.upgrade_window_indicator.setStyleSheet(style)

    def set_bypass_mode(self, enabled: bool):
        self.bypass_engine.set_bypass_enabled(enabled)
        self._update_bypass_indicator()

    def set_self_driving(self, enabled):
        self.self_driving_enabled = enabled
        state = "ENABLED" if enabled else "DISABLED"
        self.log_panel.log(f"[Self-Driving] {state}")

    def on_active_lane_changed(self, fork_name):
        self.active_lane = fork_name
        self.log_panel.log(f"[Lane] Active lane set to: {fork_name}")

    def _compute_code_lane_risk(self) -> str:
        recent = self.future_planner_engine.code_lane_events[-5:]
        if any(e.event_type == "demotion" for e in recent):
            return "high"
        if any(e.event_type in ("fallback_used", "preemptive_bypass") for e in recent):
            return "medium"
        return "low"

    def _lane_forecast(self, horizon_steps: int = 3) -> List[str]:
        risk = self._compute_code_lane_risk()
        if risk == "low":
            return ["low"] * horizon_steps
        if risk == "medium":
            return ["medium"] * horizon_steps
        return ["high"] * horizon_steps

    def _current_context_snapshot(self) -> Dict[str, Any]:
        routing = self.prometheus.get_current_routing_risk_level()
        backup = self.backup_engine.backup_risk_level()
        lane = self._compute_code_lane_risk()
        return {
            "routing_risk": routing,
            "backup_risk": backup,
            "lane_risk": lane,
            "active_lane": self.active_lane or "none"
        }

    def tick(self):
        forks = self.prometheus.get_current_forks()
        self.future_planner_engine.forks = forks

        routing_forecast = self.prometheus.forecast_routing_risk(3)
        backup_forecast = self.backup_engine.forecast_backup_risk_for_active(3)
        lane_forecast = self._lane_forecast(3)
        self.future_planner_engine.update_forecasts(routing_forecast, backup_forecast)

        self.future_planner_tab.refresh_view()

        routing_level = self.prometheus.get_current_routing_risk_level()
        code_lane_risk = self._compute_code_lane_risk()
        backup_risk = self.backup_engine.backup_risk_level()

        metrics = {
            "CPU": "low",
            "DNS": routing_level,
            "Routes": routing_level,
            "Env": "low",
            "Code": code_lane_risk,
            "Backup": backup_risk
        }
        overall_level = max(
            [routing_level, code_lane_risk, backup_risk],
            key=lambda lvl: {"low": 0, "medium": 1, "high": 2}[lvl]
        )
        risk_snapshot = {"level": overall_level, "metrics": metrics}
        self.hazard_radar.update_risk(risk_snapshot)

        self.forecast_strip.update_forecast(routing_forecast, backup_forecast, lane_forecast)

        heatmap_data = self.bypass_engine.get_heatmap_data()
        self.bypass_heatmap.update_heatmap(heatmap_data)

        clusters = self.bypass_engine.fingerprint_cluster_summary()
        self.fingerprint_clusters.update_clusters(clusters)

        self.code_lane_timeline.update_events(self.future_planner_engine.code_lane_events)

        self.backup_tab.refresh_status()

        if self.persistence:
            self.persistence.update_drive_scores(self.backup_engine.path_stats)

        upgrade_window = self.future_planner_engine.predict_upgrade_window()
        self._update_upgrade_window_indicator(upgrade_window)

        fork_scores = self.future_planner_engine.rank_forks()
        self.opportunities_engine.update_from_system(
            routing_level=routing_level,
            routing_forecast=routing_forecast,
            backup_level=backup_risk,
            backup_forecast=backup_forecast,
            lane_risk=code_lane_risk,
            heatmap=heatmap_data,
            clusters=clusters,
            drive_stats=self.backup_engine.path_stats,
            fork_scores=fork_scores,
        )
        self.opportunities_tab.refresh_view()

        if self.persistence:
            self.persistence.append_opportunity_snapshot(self.opportunities_engine.opportunities)

        self.sync_manager.export_state("node1")

        if self.self_driving_enabled:
            self.run_self_driving_step(routing_forecast, backup_forecast, lane_forecast, upgrade_window)

    def run_self_driving_step(self,
                              routing_forecast: List[str],
                              backup_forecast: List[str],
                              lane_forecast: List[str],
                              upgrade_window: str):
        ranked = self.future_planner_engine.rank_forks()
        if not ranked:
            return

        recent_events = self.future_planner_engine.code_lane_events[-3:]
        if any(e.event_type == "demotion" for e in recent_events):
            msg = "[Self-Driving] Code lane instability detected — delaying upgrades"
            self.log_panel.log(msg)
            self.intent_log.append(msg)
            return

        if upgrade_window == "bad":
            msg = "[Self-Driving] Upgrade window predicted BAD — holding all upgrades"
            self.log_panel.log(msg)
            self.intent_log.append(msg)
            return

        if "high" in backup_forecast:
            self.log_panel.log("[Self-Driving] Backup forecast HIGH — attempting auto-switch to better backup path")
            self.backup_engine.auto_switch_if_needed("medium")
            backup_forecast = self.backup_engine.forecast_backup_risk_for_active(3)
            if "high" in backup_forecast:
                msg = "[Self-Driving] Backup risk remains HIGH after auto-switch — delaying upgrades"
                self.log_panel.log(msg)
                self.intent_log.append(msg)
                return

        best_entry = ranked[0]
        if self.active_lane:
            for entry in ranked:
                if entry["fork"].name == self.active_lane:
                    if entry["score"] <= 1.1 * best_entry["score"]:
                        best_entry = entry
                    break

        best_fork = best_entry["fork"]
        merged = self.future_planner_engine.merge_best_future()

        intent_msg = (
            f"[Self-Driving Intent] Considering fork {best_fork.name} "
            f"(score={best_entry['score']:.2f}, risk={best_fork.result.risk_level}) "
            f"with routing forecast={routing_forecast}, backup forecast={backup_forecast}, "
            f"lane forecast={lane_forecast}, upgrade_window={upgrade_window}"
        )
        self.intent_log.append(intent_msg)

        if "medium" in backup_forecast and self.backup_engine.active_path:
            self.log_panel.log("[Self-Driving] Backup forecast MEDIUM — re-testing active backup path pre-emptively")
            self.backup_engine.test_backup_path(self.backup_engine.active_path)

        if merged.merged_risk_level == "low" and best_fork.result.rollback_available and upgrade_window != "bad":
            apply_msg = (
                f"[Self-Driving] Applying merged future from fork {merged.base_fork}, "
                f"snapshot {merged.base_snapshot}"
            )
            self.log_panel.log(apply_msg)
            self.intent_log.append(apply_msg)
            self.prometheus.apply_upgrade_plan(
                merged.base_snapshot,
                merged.merged_upgrades
            )
        else:
            msg = (
                "[Self-Driving] Not safe to apply upgrades yet "
                f"(risk={merged.merged_risk_level}, rollback={best_fork.result.rollback_available}, "
                f"upgrade_window={upgrade_window})"
            )
            self.log_panel.log(msg)
            self.intent_log.append(msg)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    persistence = PersistenceLayer()
    ml_predictor = MLPredictor()
    planner_engine = FuturePlannerEngine()
    planner_engine.persistence = persistence
    planner_engine.ml_predictor = ml_predictor

    gui_logger = ThreadSafeLogger()

    bypass_engine = ExecutionBypassEngine(logger=gui_logger.log)
    bypass_engine.persistence = persistence

    backup_engine = BackupManagerEngine(logger=gui_logger.log, planner=planner_engine, persistence=persistence)
    opportunities_engine = AreaOfOpportunityEngine()

    probe = NetworkProbe(logger=gui_logger.log)
    upgrade_engine = UpgradeEngine(logger=gui_logger.log)
    prometheus = PrometheusCore(bypass_engine=bypass_engine, probe=probe, upgrade_engine=upgrade_engine)
    prometheus.persistence = persistence

    redundant_backup = RedundantBackupManager(backup_engine, logger=gui_logger.log)
    sync_manager = NodeSyncManager(persistence, logger=gui_logger.log)
    plugins = PluginRegistry()

    window = MainWindow(
        prometheus,
        bypass_engine,
        planner_engine,
        backup_engine,
        opportunities_engine,
        persistence,
        redundant_backup,
        sync_manager,
        plugins,
        ml_predictor,
        gui_logger,
    )

    window.resize(1700, 950)
    window.show()

    sys.exit(app.exec_())

