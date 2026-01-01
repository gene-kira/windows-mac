#!/usr/bin/env python3
"""
Backbone Total Unified Intelligent – Water/Data Physics Engine
--------------------------------------------------------------

Hybrid intelligence:
- Surface UI: clean status, NetHost regime, simple confidence, alerts.
- Diagnostics tab: technical predictions + narrative interpretation side-by-side.

Core concepts:
- Water → Data
- Gravity → Routing pull
- Surface tension → Stickiness
- Pressure → Environment compatibility
- Viscosity → Movement resistance
- Turbulence → Instability
- Flow rate → Effective throughput
- Phase → Liquid / vapor / ice (Earth / ISS / Moon)

Intelligence layer:
- Live internet-based NetHost (success_ratio + latency).
- Rolling NetHost history.
- Multi-horizon prediction (short/medium/long).
- Probabilistic regimes (Earth/ISS/Moon probabilities).
- Confidence scoring.
- Pattern detection (oscillation, decay, collapse, outage, turbulence).
- Anomaly clustering (episodes).
- Adaptive autopilot focused on risky nodes.
"""

import sys
import math
import threading
import time
import random
import json
import os
import socket
import platform
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# --- GUI import --------------------------------------------------------------

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Error: tkinter (Tk) is not available on this system.")
    print("Install/enable it first, then re-run this script.")
    sys.exit(1)

# ============================================================================
# Core physics – water → data
# ============================================================================


def bond_number(routing_pull: float, stickiness: float, scale: float = 1.0) -> float:
    stickiness = max(stickiness, 1e-3)
    return (routing_pull * (scale ** 2)) / stickiness


def phase_stability(pressure: float) -> float:
    s = 2.0 * pressure - 1.0
    return max(-1.0, min(1.0, s))


def pouring_index(pressure: float, routing_pull: float, stickiness: float) -> float:
    s_phase = phase_stability(pressure)
    bo = bond_number(routing_pull, stickiness)
    return s_phase * math.tanh(bo)


def viscosity(pressure: float, stickiness: float) -> float:
    inv_pressure = 1.0 - max(0.0, min(1.0, pressure))
    base = stickiness * (0.5 + inv_pressure)
    return max(0.0, min(1.5, base))


def turbulence(pressure: float, routing_pull: float, stickiness: float) -> float:
    p = max(0.0, min(1.0, pressure))
    r = max(0.0, min(1.5, routing_pull)) / 1.5
    s = max(0.05, min(1.5, stickiness)) / 1.5

    competition = 1.0 - abs(r - s)
    competition *= (r + s) / 2.0

    marginal_p = 1.0 - abs(p - 0.5) * 2.0
    marginal_p = max(0.0, marginal_p)

    turb = 0.6 * competition + 0.4 * marginal_p
    return max(0.0, min(1.0, turb))


def flow_rate(pressure: float, routing_pull: float, stickiness: float) -> float:
    v = viscosity(pressure, stickiness)
    p = max(0.0, min(1.0, pressure))
    r_norm = max(0.0, min(1.5, routing_pull)) / 1.5
    potential = 0.5 * p + 0.5 * r_norm
    flow = potential * (1.0 - min(1.0, v / 1.5))
    return max(0.0, min(1.0, flow))


def classify_regime(pi: float, pressure: float) -> Tuple[str, str]:
    if pressure < 0.15:
        return "Moon", "#aa0000"
    if pi > 0.5:
        return "Earth", "#008800"
    elif pi < -0.5:
        return "Moon", "#aa0000"
    else:
        return "ISS", "#aa8800"


# ============================================================================
# Local physics simulation (single bucket)
# ============================================================================


def simulate_local_pour(pressure: float,
                        routing_pull: float,
                        stickiness: float,
                        volume: float = 5.0,
                        prefix: str = "Local") -> List[str]:
    logs: List[str] = []
    logs.append(f"=== {prefix} Data Pour Simulation ===")
    logs.append(f"  pressure (env compatibility): {pressure:.2f}")
    logs.append(f"  routing pull (gravity):       {routing_pull:.2f}")
    logs.append(f"  stickiness (surface tension): {stickiness:.2f}")
    logs.append(f"  volume (relative units):      {volume:.2f}")
    logs.append("")

    pi = pouring_index(pressure, routing_pull, stickiness)
    bo = bond_number(routing_pull, stickiness)
    s_phase = phase_stability(pressure)
    visc = viscosity(pressure, stickiness)
    turb = turbulence(pressure, routing_pull, stickiness)
    fr = flow_rate(pressure, routing_pull, stickiness)

    logs.append("Derived quantities:")
    logs.append(f"  S_phase (phase factor): {s_phase:.3f}")
    logs.append(f"  Bo (Bond-like number):  {bo:.3f}")
    logs.append(f"  Pi_pour (index):        {pi:.3f}")
    logs.append(f"  viscosity:              {visc:.3f}")
    logs.append(f"  turbulence:             {turb:.3f}")
    logs.append(f"  flow_rate (0..1):       {fr:.3f}")
    logs.append("")

    regime, _ = classify_regime(pi, pressure)
    logs.append(f"Regime classification: {regime}")
    logs.append("")

    if s_phase <= -0.2:
        logs.append("Environment is hostile to 'liquid data'.")
        logs.append("Data tends to become errors/logs (vapor) or dead storage (ice).")
        logs.append("")
        logs.append("Result:")
        logs.append("  - No smooth stream forms.")
        logs.append("  - Scattered errors, partial writes, dead artifacts.")
        return logs

    if regime == "Earth":
        logs.append("Environment supports stable, liquid data.")
        logs.append("Global routing pull dominates over local stickiness.")
        logs.append(f"Flow: moderate viscosity ({visc:.2f}),"
                    f" flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result:")
        logs.append("  - Coherent streams and storage puddles.")
        logs.append("  - Backpressure shapes but does not prevent flow.")
    elif regime == "ISS":
        logs.append("Liquid data is stable, but global pull is weak.")
        logs.append("Local stickiness competes with or beats routing pull.")
        logs.append(f"Flow: higher viscosity ({visc:.2f}),"
                    f" patchy flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result:")
        logs.append("  - Blobs of local state and caches.")
        logs.append("  - Movement is patchy and trigger-driven.")
    else:
        logs.append("Environment is marginal and unstable for liquid data.")
        logs.append("Some data survives as liquid, much becomes ice/vapor.")
        logs.append(f"Flow: high viscosity ({visc:.2f}),"
                    f" low flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result:")
        logs.append("  - Fragile flow, partial pipelines, intermittent success.")
        logs.append("  - Significant fraction becomes errors or cold artifacts.")

    return logs


# ============================================================================
# Backbone model
# ============================================================================


@dataclass
class Node:
    name: str
    pressure: float
    routing_pull: float
    stickiness: float
    volume_capacity: float = 10.0

    def pouring_index(self) -> float:
        return pouring_index(self.pressure, self.routing_pull, self.stickiness)

    def regime(self) -> str:
        pi = self.pouring_index()
        reg, _ = classify_regime(pi, self.pressure)
        return reg

    def regime_color(self) -> str:
        pi = self.pouring_index()
        _, color = classify_regime(pi, self.pressure)
        return color

    def viscosity(self) -> float:
        return viscosity(self.pressure, self.stickiness)

    def turbulence(self) -> float:
        return turbulence(self.pressure, self.routing_pull, self.stickiness)

    def flow_rate(self) -> float:
        return flow_rate(self.pressure, self.routing_pull, self.stickiness)


@dataclass
class Edge:
    source: str
    target: str
    bandwidth: float = 1.0
    reliability: float = 0.9


from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Backbone:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)



    
    

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def neighbors(self, node_name: str) -> List[Edge]:
        return [e for e in self.edges if e.source == node_name]

    def get_node(self, name: str) -> Optional[Node]:
        return self.nodes.get(name)

    def describe(self) -> List[str]:
        lines: List[str] = []
        lines.append("Backbone topology:")
        lines.append("")
        for node in self.nodes.values():
            pi = node.pouring_index()
            reg = node.regime()
            visc = node.viscosity()
            turb = node.turbulence()
            fr = node.flow_rate()
            lines.append(
                f"  Node {node.name}: "
                f"regime={reg}, "
                f"pressure={node.pressure:.2f}, "
                f"routing={node.routing_pull:.2f}, "
                f"stickiness={node.stickiness:.2f}, "
                f"Pi={pi:.3f}, "
                f"visc={visc:.2f}, "
                f"turb={turb:.2f}, "
                f"flow={fr:.2f}"
            )
        lines.append("")
        lines.append("Edges:")
        if not self.edges:
            lines.append("  (none)")
        else:
            for e in self.edges:
                lines.append(
                    f"  {e.source} -> {e.target}  "
                    f"(bandwidth={e.bandwidth:.2f}, reliability={e.reliability:.2f})"
                )
        return lines

    def find_paths(self, source: str, target: str, max_depth: int = 6) -> List[List[str]]:
        paths: List[List[str]] = []

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path[:])
                return
            for edge in self.neighbors(current):
                if edge.target in path:
                    continue
                path.append(edge.target)
                dfs(edge.target, path, depth + 1)
                path.pop()

        if source in self.nodes and target in self.nodes:
            dfs(source, [source], 0)
        return paths


# fix typo: field_default_factory
Backbone.__dataclass_fields__['edges'].default_factory = list


# ============================================================================
# Backbone pour simulation
# ============================================================================


def simulate_backbone_pour(backbone: Backbone,
                           source: str,
                           target: str,
                           volume: float = 5.0,
                           prefix: str = "Backbone") -> List[str]:
    logs: List[str] = []
    logs.append(f"=== {prefix} Pour Simulation ===")
    logs.append(f"Source: {source}")
    logs.append(f"Target: {target}")
    logs.append(f"Initial volume: {volume:.2f}")
    logs.append("")

    if source not in backbone.nodes:
        logs.append(f"[ERROR] Source node '{source}' not found.")
        return logs
    if target not in backbone.nodes:
        logs.append(f"[ERROR] Target node '{target}' not found.")
        return logs

    paths = backbone.find_paths(source, target, max_depth=8)
    if not paths:
        logs.append("No path found from source to target.")
        logs.append("Result: data cannot form a continuous stream between these nodes.")
        return logs

    logs.append(f"Found {len(paths)} path(s) from '{source}' to '{target}'.")
    logs.append("")

    for i, path in enumerate(paths, start=1):
        logs.append(f"Path #{i}: {' -> '.join(path)}")
        logs.append("-" * 60)
        remaining_volume = volume
        cumulative_reliability = 1.0

        for idx, node_name in enumerate(path):
            node = backbone.get_node(node_name)
            if node is None:
                logs.append(f"  [ERROR] Node '{node_name}' missing from backbone.")
                break

            pi = node.pouring_index()
            reg = node.regime()
            s_phase = phase_stability(node.pressure)
            bo = bond_number(node.routing_pull, node.stickiness)
            visc = node.viscosity()
            turb = node.turbulence()
            fr = node.flow_rate()

            logs.append(f"  Node {node_name}:")
            logs.append(f"    regime={reg}, S_phase={s_phase:.3f}, Bo={bo:.3f}, Pi={pi:.3f}")
            logs.append(f"    viscosity={visc:.3f}, turbulence={turb:.3f}, flow_rate={fr:.3f}")

            if reg == "Moon":
                logs.append("    Hostile to liquid data: errors/logs (vapor) or dead storage (ice).")
                remaining_volume *= 0.3 * (0.5 + fr)
                cumulative_reliability *= 0.5 * (0.7 + (1.0 - turb) * 0.3)
            elif reg == "ISS":
                logs.append("    Liquid data but weak global pull: local caches dominate.")
                remaining_volume *= 0.7 * (0.5 + fr)
                cumulative_reliability *= 0.8 * (0.7 + (1.0 - turb) * 0.3)
            else:
                logs.append("    Stable liquid data: strong routing pull, coherent streams.")
                remaining_volume *= 0.9 * (0.7 + fr * 0.6)
                cumulative_reliability *= 0.95 * (0.8 + (1.0 - turb) * 0.2)

            logs.append(f"    Remaining volume after node: {remaining_volume:.2f}")
            logs.append(f"    Cumulative reliability so far: {cumulative_reliability:.3f}")
            logs.append("")

            if idx < len(path) - 1:
                next_name = path[idx + 1]
                edge = next(
                    (e for e in backbone.edges
                     if e.source == node_name and e.target == next_name),
                    None
                )
                if edge:
                    logs.append(
                        f"    Edge to {next_name}: "
                        f"bandwidth={edge.bandwidth:.2f}, reliability={edge.reliability:.2f}"
                    )
                    edge_factor = edge.reliability * (0.8 + fr * 0.2) * (0.8 + (1.0 - turb) * 0.2)
                    cumulative_reliability *= edge_factor
                    logs.append(f"    Reliability after edge: {cumulative_reliability:.3f}")
                    logs.append("")
                else:
                    logs.append(f"    [WARNING] No explicit edge {node_name} -> {next_name} recorded.")
                    logs.append("")

        logs.append(f"Final estimate for path #{i}:")
        logs.append(f"  Delivered volume:     {remaining_volume:.2f} / {volume:.2f}")
        logs.append(f"  Effective reliability: {cumulative_reliability:.3f}")
        logs.append("")
        logs.append("")

    return logs


# ============================================================================
# Internet sensors & NetHost history (predictive, probabilistic)
# ============================================================================


def check_host_latency(host: str, port: int, timeout: float = 1.0) -> Optional[float]:
    start = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return time.time() - start
    except OSError:
        return None


def sample_internet_status() -> Tuple[float, float]:
    targets = [
        ("1.1.1.1", 53),
        ("8.8.8.8", 53),
        ("www.google.com", 80),
        ("www.microsoft.com", 80),
    ]
    latencies = []
    successes = 0
    total = len(targets)

    for host, port in targets:
        latency = check_host_latency(host, port, timeout=1.0)
        if latency is not None:
            successes += 1
            latencies.append(latency)

    success_ratio = successes / total if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 1.5
    return success_ratio, avg_latency


@dataclass
class NetHostSample:
    ts: float
    success_ratio: float
    avg_latency: float
    pressure: float
    routing_pull: float
    stickiness: float
    pi: float
    regime: str


@dataclass
class AnomalyEpisode:
    start_ts: float
    end_ts: float
    types: List[str] = field(default_factory=list)

    def add_type(self, t: str):
        if t not in self.types:
            self.types.append(t)


class NetHostHistory:
    def __init__(self, maxlen: int = 50):
        self.samples: deque[NetHostSample] = deque(maxlen=maxlen)
        self.episodes: List[AnomalyEpisode] = []
        self.current_episode: Optional[AnomalyEpisode] = None

    def add_sample(self, s: NetHostSample, anomalies: Dict[str, bool]):
        self.samples.append(s)
        active_types = [k for k, v in anomalies.items() if v]

        if active_types:
            if self.current_episode is None:
                self.current_episode = AnomalyEpisode(start_ts=s.ts, end_ts=s.ts)
            self.current_episode.end_ts = s.ts
            for t in active_types:
                self.current_episode.add_type(t)
        else:
            if self.current_episode is not None:
                self.episodes.append(self.current_episode)
                self.current_episode = None

    def has_enough(self, n: int = 3) -> bool:
        return len(self.samples) >= n

    def latest(self) -> Optional[NetHostSample]:
        return self.samples[-1] if self.samples else None

    def rolling_mean(self) -> Optional[NetHostSample]:
        if not self.samples:
            return None
        n = len(self.samples)
        sr = sum(s.success_ratio for s in self.samples) / n
        lat = sum(s.avg_latency for s in self.samples) / n
        p = sum(s.pressure for s in self.samples) / n
        r = sum(s.routing_pull for s in self.samples) / n
        st = sum(s.stickiness for s in self.samples) / n
        pi = sum(s.pi for s in self.samples) / n
        reg, _ = classify_regime(pi, p)
        return NetHostSample(ts=self.samples[-1].ts,
                             success_ratio=sr,
                             avg_latency=lat,
                             pressure=p,
                             routing_pull=r,
                             stickiness=st,
                             pi=pi,
                             regime=reg)

    def trend(self, field_getter, min_samples: int = 3) -> float:
        if len(self.samples) < min_samples:
            return 0.0
        first = self.samples[0]
        last = self.samples[-1]
        dt = max(1e-3, last.ts - first.ts)
        return (field_getter(last) - field_getter(first)) / dt

    def trend_pi(self) -> float:
        return self.trend(lambda s: s.pi)

    def pattern_label(self) -> str:
        if len(self.samples) < 5:
            return "insufficient_history"

        pis = [s.pi for s in self.samples]
        times = [s.ts for s in self.samples]
        diffs = [pis[i+1] - pis[i] for i in range(len(pis)-1)]
        signs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

        # Oscillation: sign changes multiple times
        sign_changes = sum(1 for i in range(len(signs)-1) if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1])
        if sign_changes >= 3:
            return "oscillation"

        pi_trend = self.trend_pi()
        mean_pi = sum(pis) / len(pis)

        # Slow decay towards Moon
        if pi_trend < 0 and mean_pi < 0.3:
            return "slow_decay"

        # Sudden collapse: last delta much larger negative than typical
        last_delta = diffs[-1]
        mean_mag = sum(abs(d) for d in diffs[:-1]) / max(1, len(diffs)-1)
        if last_delta < -2 * mean_mag and last_delta < -0.2:
            return "sudden_collapse"

        # Partial outage: success ratio often between 0 and 1
        srs = [s.success_ratio for s in self.samples]
        if any(0.0 < x < 1.0 for x in srs) and sum(1 for x in srs if x == 0.0) > 0:
            return "partial_outage"

        # High turbulence: flow/turbulence combination harsh
        press = [s.pressure for s in self.samples]
        routs = [s.routing_pull for s in self.samples]
        sticks = [s.stickiness for s in self.samples]
        turbs = [turbulence(press[i], routs[i], sticks[i]) for i in range(len(press))]
        if sum(1 for t in turbs if t > 0.7) > len(turbs)/2:
            return "high_turbulence"

        if mean_pi > 0.6:
            return "stable_earth"
        if mean_pi > 0.0:
            return "recovering_iss"
        if mean_pi < -0.4:
            return "pre_moon_vacuum"

        return "neutral"

    def anomaly_flags(self) -> Dict[str, bool]:
        flags = {"latency_spike": False, "success_drop": False}
        if len(self.samples) < 5:
            return flags

        baseline = list(self.samples)[:-2]
        n = len(baseline)
        mean_sr = sum(s.success_ratio for s in baseline) / n
        mean_lat = sum(s.avg_latency for s in baseline) / n
        last = self.latest()

        if mean_sr > 0 and last.success_ratio < mean_sr * 0.5:
            flags["success_drop"] = True
        if mean_lat > 0 and last.avg_latency > mean_lat * 2.0:
            flags["latency_spike"] = True
        return flags

    def regime_probabilities(self, pi_value: float, pressure: float) -> Dict[str, float]:
        base_reg, _ = classify_regime(pi_value, pressure)
        # crude probabilities around classification
        if base_reg == "Earth":
            probs = {"Earth": 0.7, "ISS": 0.25, "Moon": 0.05}
        elif base_reg == "ISS":
            probs = {"Earth": 0.2, "ISS": 0.6, "Moon": 0.2}
        else:
            probs = {"Earth": 0.05, "ISS": 0.25, "Moon": 0.7}
        return probs

    def confidence_score(self) -> float:
        if len(self.samples) < 3:
            return 0.2
        pis = [s.pi for s in self.samples]
        mean_pi = sum(pis) / len(pis)
        var = sum((p - mean_pi) ** 2 for p in pis) / len(pis)
        stability = max(0.0, 1.0 - min(var * 5.0, 1.0))  # small variance -> high stability
        horizon_penalty = 0.2  # we don't look very far
        return max(0.0, min(1.0, stability - horizon_penalty))


def compute_nethost_node_params(history: NetHostHistory) -> Tuple[float, float, float, dict]:
    success_ratio, avg_latency = sample_internet_status()
    lat_norm = max(0.0, min(1.0, avg_latency / 1.5))

    pressure = success_ratio * (1.0 - 0.5 * lat_norm)
    pressure = max(0.0, min(1.0, pressure))

    routing_pull = 0.2 + 1.3 * success_ratio * (1.0 - 0.3 * lat_norm)
    routing_pull = max(0.0, min(1.5, routing_pull))

    stickiness = 0.2 + (1.0 - success_ratio) * 1.3 + lat_norm * 0.7
    stickiness = max(0.05, min(1.5, stickiness))

    pi_now = pouring_index(pressure, routing_pull, stickiness)
    reg_now, _ = classify_regime(pi_now, pressure)

    sample = NetHostSample(
        ts=time.time(),
        success_ratio=success_ratio,
        avg_latency=avg_latency,
        pressure=pressure,
        routing_pull=routing_pull,
        stickiness=stickiness,
        pi=pi_now,
        regime=reg_now,
    )

    anomalies = history.anomaly_flags()
    history.add_sample(sample, anomalies)

    mean_sample = history.rolling_mean() or sample

    short_pi = pi_now + history.trend_pi() * 5.0
    mid_pi = pi_now + history.trend_pi() * 30.0
    long_pi = pi_now + history.trend_pi() * 120.0

    short_reg, _ = classify_regime(short_pi, mean_sample.pressure)
    mid_reg, _ = classify_regime(mid_pi, mean_sample.pressure)
    long_reg, _ = classify_regime(long_pi, mean_sample.pressure)

    short_probs = history.regime_probabilities(short_pi, mean_sample.pressure)
    mid_probs = history.regime_probabilities(mid_pi, mean_sample.pressure)
    long_probs = history.regime_probabilities(long_pi, mean_sample.pressure)

    conf_now = history.confidence_score()
    conf_short = max(0.0, conf_now - 0.1)
    conf_mid = max(0.0, conf_now - 0.25)
    conf_long = max(0.0, conf_now - 0.4)

    pattern = history.pattern_label()

    debug = {
        "success_ratio": success_ratio,
        "avg_latency": avg_latency,
        "lat_norm": lat_norm,
        "pi_now": pi_now,
        "reg_now": reg_now,
        "pi_short": short_pi,
        "pi_mid": mid_pi,
        "pi_long": long_pi,
        "reg_short": short_reg,
        "reg_mid": mid_reg,
        "reg_long": long_reg,
        "probs_short": short_probs,
        "probs_mid": mid_probs,
        "probs_long": long_probs,
        "conf_now": conf_now,
        "conf_short": conf_short,
        "conf_mid": conf_mid,
        "conf_long": conf_long,
        "pattern": pattern,
        "anomalies": anomalies,
    }
    return pressure, routing_pull, stickiness, debug


# ============================================================================
# Example backbone (with NetHost)
# ============================================================================


def create_example_backbone(history: NetHostHistory) -> Tuple[Backbone, dict]:
    b = Backbone()

    n_pressure, n_routing, n_stickiness, debug = compute_nethost_node_params(history)
    b.add_node(Node("NetHost", pressure=n_pressure,
                    routing_pull=n_routing,
                    stickiness=n_stickiness))

    b.add_node(Node("Ingest",      pressure=0.95, routing_pull=1.0,  stickiness=0.2))
    b.add_node(Node("PipelineA",   pressure=0.90, routing_pull=0.9,  stickiness=0.3))
    b.add_node(Node("StorageLake", pressure=0.92, routing_pull=0.8,  stickiness=0.4))

    b.add_node(Node("ServiceX", pressure=0.90, routing_pull=0.1, stickiness=0.9))
    b.add_node(Node("CacheX",   pressure=0.90, routing_pull=0.05, stickiness=1.2))

    b.add_node(Node("Quarantine",  pressure=0.05, routing_pull=0.6, stickiness=0.5))
    b.add_node(Node("DeepArchive", pressure=0.10, routing_pull=0.3, stickiness=0.8))

    b.add_edge(Edge("NetHost",   "Ingest",      bandwidth=0.8, reliability=0.95))
    b.add_edge(Edge("Ingest",    "PipelineA",   bandwidth=1.0, reliability=0.98))
    b.add_edge(Edge("PipelineA", "StorageLake", bandwidth=0.9, reliability=0.97))

    b.add_edge(Edge("PipelineA", "ServiceX",    bandwidth=0.5, reliability=0.90))
    b.add_edge(Edge("ServiceX",  "CacheX",      bandwidth=0.3, reliability=0.85))

    b.add_edge(Edge("Ingest",    "Quarantine",  bandwidth=0.2, reliability=0.80))
    b.add_edge(Edge("Quarantine","DeepArchive", bandwidth=0.4, reliability=0.90))
    b.add_edge(Edge("PipelineA", "DeepArchive", bandwidth=0.3, reliability=0.70))

    return b, debug


# ============================================================================
# Persistent history logger (JSONL)
# ============================================================================


HISTORY_FILE = "backbone_history_intelligent.jsonl"
HISTORY_LOCK = threading.Lock()


def append_history_record(record: dict):
    try:
        with HISTORY_LOCK:
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ============================================================================
# GUI – Local Physics Tab
# ============================================================================


class LocalPhysicsFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self._build_widgets()

    def _build_widgets(self):
        controls_frame = ttk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        log_frame = ttk.Frame(self)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(8, 0))

        preset_frame = ttk.LabelFrame(controls_frame, text="Presets")
        preset_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        ttk.Button(preset_frame, text="Earth", command=self.set_earth).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(preset_frame, text="ISS",   command=self.set_iss).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(preset_frame, text="Moon",  command=self.set_moon).pack(fill=tk.X, padx=4, pady=2)

        sliders_frame = ttk.LabelFrame(controls_frame, text="Environment & Pour")
        sliders_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pressure_var = tk.DoubleVar(value=0.95)
        self.routing_var = tk.DoubleVar(value=0.80)
        self.stickiness_var = tk.DoubleVar(value=0.30)
        self.volume_var = tk.DoubleVar(value=5.0)

        self._add_slider(sliders_frame, "Pressure (env compatibility)",
                         self.pressure_var, 0.0, 1.0)
        self._add_slider(sliders_frame, "Routing pull (gravity)",
                         self.routing_var, 0.0, 1.5)
        self._add_slider(sliders_frame, "Stickiness (surface tension)",
                         self.stickiness_var, 0.05, 1.5)
        self._add_slider(sliders_frame, "Volume (relative data amount)",
                         self.volume_var, 1.0, 10.0)

        right_frame = ttk.LabelFrame(controls_frame, text="Regime Indicator")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.regime_label = ttk.Label(right_frame,
                                      text="Regime: (not evaluated yet)",
                                      anchor="center",
                                      font=("TkDefaultFont", 11, "bold"))
        self.regime_label.pack(fill=tk.X, pady=(8, 4), padx=4)

        self.pi_label = ttk.Label(right_frame, text="Pi_pour: N/A", anchor="center")
        self.pi_label.pack(fill=tk.X, pady=(4, 4), padx=4)

        self.detail_label = ttk.Label(right_frame,
                                      text="Adjust sliders or use presets, then click 'Simulate Pour'.",
                                      anchor="center",
                                      wraplength=260,
                                      justify="center")
        self.detail_label.pack(fill=tk.BOTH, expand=True, pady=(4, 4), padx=4)

        self.sim_button = ttk.Button(right_frame, text="Simulate Pour", command=self.on_simulate)
        self.sim_button.pack(fill=tk.X, pady=(4, 8), padx=8)

        ttk.Label(log_frame, text="Local Simulation Log").pack(anchor="w")
        self.log_text = tk.Text(log_frame,
                                wrap="word",
                                height=15,
                                state="disabled",
                                background="#101010",
                                foreground="#dddddd")
        self.log_text.configure(font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        self.update_regime_preview()

    def _add_slider(self, parent, label, var, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=label).pack(anchor="w")
        scale = ttk.Scale(frame, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X)

        value_label = ttk.Label(frame, text=f"{var.get():.2f}", width=8, anchor="e")
        value_label.pack(anchor="e")

        def update_value(_event=None, v=var, lbl=value_label):
            lbl.config(text=f"{v.get():.2f}")
            self.update_regime_preview()

        scale.bind("<B1-Motion>", update_value)
        scale.bind("<ButtonRelease-1>", update_value)

    def set_earth(self):
        self.pressure_var.set(0.95)
        self.routing_var.set(1.0)
        self.stickiness_var.set(0.2)
        self.volume_var.set(5.0)
        self.update_regime_preview()

    def set_iss(self):
        self.pressure_var.set(0.95)
        self.routing_var.set(0.05)
        self.stickiness_var.set(0.9)
        self.volume_var.set(5.0)
        self.update_regime_preview()

    def set_moon(self):
        self.pressure_var.set(0.05)
        self.routing_var.set(0.6)
        self.stickiness_var.set(0.5)
        self.volume_var.set(5.0)
        self.update_regime_preview()

    def update_regime_preview(self):
        pressure = self.pressure_var.get()
        routing = self.routing_var.get()
        stickiness = self.stickiness_var.get()

        pi = pouring_index(pressure, routing, stickiness)
        regime, color = classify_regime(pi, pressure)

        self.regime_label.config(text=f"Regime: {regime}", foreground=color)
        self.pi_label.config(text=f"Pi_pour: {pi:.3f}")

        visc = viscosity(pressure, stickiness)
        turb = turbulence(pressure, routing, stickiness)
        fr = flow_rate(pressure, routing, stickiness)

        if regime == "Earth":
            text = (
                "Earth regime:\n"
                "- Environment supports liquid data.\n"
                "- Global routing pull dominates.\n"
                f"- Flow: visc={visc:.2f}, turb={turb:.2f}, flow={fr:.2f}."
            )
        elif regime == "ISS":
            text = (
                "ISS regime:\n"
                "- Liquid data is stable, but global pull is weak.\n"
                "- Local stickiness dominates.\n"
                f"- Flow: visc={visc:.2f}, turb={turb:.2f}, flow={fr:.2f}."
            )
        else:
            text = (
                "Moon regime:\n"
                "- Hostile to liquid data.\n"
                "- Data evaporates or freezes.\n"
                f"- Flow: visc={visc:.2f}, turb={turb:.2f}, flow={fr:.2f}."
            )

        self.detail_label.config(text=text)

    def append_log(self, lines: List[str]):
        self.log_text.configure(state="normal")
        for line in lines:
            self.log_text.insert(tk.END, line + "\n")
        self.log_text.insert(tk.END, "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def on_simulate(self, prefix: str = "Local"):
        pressure = self.pressure_var.get()
        routing = self.routing_var.get()
        stickiness = self.stickiness_var.get()
        volume = self.volume_var.get()

        def worker():
            logs = simulate_local_pour(pressure, routing, stickiness, volume, prefix=prefix)
            record = {
                "ts": time.time(),
                "mode": "local" if prefix == "Local" else "autopilot_local",
                "prefix": prefix,
                "pressure": pressure,
                "routing": routing,
                "stickiness": stickiness,
                "volume": volume,
                "pi": pouring_index(pressure, routing, stickiness),
                "viscosity": viscosity(pressure, stickiness),
                "turbulence": turbulence(pressure, routing, stickiness),
                "flow_rate": flow_rate(pressure, routing, stickiness),
            }
            append_history_record(record)
            time.sleep(0.02)
            self.after(0, lambda: self.append_log(logs))

        threading.Thread(target=worker, daemon=True).start()


# ============================================================================
# GUI – Backbone Tab
# ============================================================================


class BackboneFrame(ttk.Frame):
    def __init__(self, master, backbone: Backbone):
        super().__init__(master, padding=8)
        self.backbone = backbone
        self._build_widgets()
        self.refresh_node_list()
        self.refresh_flow_map()

    def _build_widgets(self):
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        node_frame = ttk.LabelFrame(left, text="Backbone Nodes")
        node_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("name", "regime", "pressure", "routing", "stickiness", "pi", "visc", "turb", "flow")
        self.node_tree = ttk.Treeview(node_frame, columns=columns, show="headings", height=10)

        self.node_tree.heading("name",       text="Name")
        self.node_tree.heading("regime",     text="Regime")
        self.node_tree.heading("pressure",   text="Pressure")
        self.node_tree.heading("routing",    text="Routing")
        self.node_tree.heading("stickiness", text="Stickiness")
        self.node_tree.heading("pi",         text="Pi")
        self.node_tree.heading("visc",       text="Visc")
        self.node_tree.heading("turb",       text="Turb")
        self.node_tree.heading("flow",       text="Flow")

        self.node_tree.column("name",       width=100, anchor="w")
        self.node_tree.column("regime",     width=90,  anchor="center")
        self.node_tree.column("pressure",   width=80,  anchor="e")
        self.node_tree.column("routing",    width=80,  anchor="e")
        self.node_tree.column("stickiness", width=90,  anchor="e")
        self.node_tree.column("pi",         width=70,  anchor="e")
        self.node_tree.column("visc",       width=60,  anchor="e")
        self.node_tree.column("turb",       width=60,  anchor="e")
        self.node_tree.column("flow",       width=60,  anchor="e")

        self.node_tree.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(left, text="Backbone Pour Simulation")
        controls.pack(fill=tk.X)

        names = list(self.backbone.nodes.keys())

        ttk.Label(controls, text="Source:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.source_var = tk.StringVar(value=names[0] if names else "")
        self.source_combo = ttk.Combobox(controls, textvariable=self.source_var,
                                         values=names, state="readonly")
        self.source_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Target:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.target_var = tk.StringVar(value=names[-1] if len(names) > 1 else "")
        self.target_combo = ttk.Combobox(controls, textvariable=self.target_var,
                                         values=names, state="readonly")
        self.target_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Volume:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.volume_var = tk.DoubleVar(value=5.0)
        self.volume_scale = ttk.Scale(controls, from_=1.0, to=20.0,
                                      orient=tk.HORIZONTAL, variable=self.volume_var)
        self.volume_scale.grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        self.volume_label = ttk.Label(controls, text="5.00")
        self.volume_label.grid(row=2, column=2, sticky="e", padx=4, pady=4)

        controls.columnconfigure(1, weight=1)

        self.sim_button = ttk.Button(controls, text="Simulate Pour", command=self.on_simulate)
        self.sim_button.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=(4, 6))

        def update_volume_label(_event=None):
            self.volume_label.config(text=f"{self.volume_var.get():.2f}")

        self.volume_scale.bind("<B1-Motion>", update_volume_label)
        self.volume_scale.bind("<ButtonRelease-1>", update_volume_label)

        top_right = ttk.LabelFrame(right, text="Backbone Flow Map (Textual)")
        top_right.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.flow_text = tk.Text(top_right,
                                 wrap="word",
                                 height=10,
                                 state="disabled",
                                 background="#111111",
                                 foreground="#dddddd")
        self.flow_text.configure(font=("Courier New", 9))
        self.flow_text.pack(fill=tk.BOTH, expand=True)

        bottom_right = ttk.LabelFrame(right, text="Backbone Simulation Log")
        bottom_right.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(bottom_right,
                                wrap="word",
                                height=10,
                                state="disabled",
                                background="#000000",
                                foreground="#dddddd")
        self.log_text.configure(font=("Courier New", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def refresh_node_list(self):
        for item in self.node_tree.get_children():
            self.node_tree.delete(item)

        for node in self.backbone.nodes.values():
            pi = node.pouring_index()
            regime = node.regime()
            visc = node.viscosity()
            turb = node.turbulence()
            fr = node.flow_rate()
            self.node_tree.insert(
                "",
                "end",
                values=(
                    node.name,
                    regime,
                    f"{node.pressure:.2f}",
                    f"{node.routing_pull:.2f}",
                    f"{node.stickiness:.2f}",
                    f"{pi:.3f}",
                    f"{visc:.2f}",
                    f"{turb:.2f}",
                    f"{fr:.2f}",
                ),
            )

    def refresh_flow_map(self):
        lines = self.backbone.describe()
        self.flow_text.configure(state="normal")
        self.flow_text.delete("1.0", tk.END)
        for line in lines:
            self.flow_text.insert(tk.END, line + "\n")
        self.flow_text.configure(state="disabled")

    def append_log(self, lines: List[str]):
        self.log_text.configure(state="normal")
        for line in lines:
            self.log_text.insert(tk.END, line + "\n")
        self.log_text.insert(tk.END, "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def on_simulate(self, prefix: str = "Backbone"):
        source = self.source_var.get()
        target = self.target_var.get()
        volume = self.volume_var.get()

        def worker():
            logs = simulate_backbone_pour(self.backbone, source, target, volume, prefix=prefix)
            record = {
                "ts": time.time(),
                "mode": "backbone" if prefix == "Backbone" else "autopilot_backbone",
                "prefix": prefix,
                "source": source,
                "target": target,
                "volume": volume,
            }
            append_history_record(record)
            time.sleep(0.02)
            self.after(0, lambda: self.append_log(logs))

        threading.Thread(target=worker, daemon=True).start()


# ============================================================================
# GUI – Diagnostics Tab (technical + narrative)
# ============================================================================


class DiagnosticsFrame(ttk.Frame):
    def __init__(self, master, history: NetHostHistory, get_debug_callback):
        super().__init__(master, padding=8)
        self.history = history
        self.get_debug_callback = get_debug_callback
        self._build_widgets()
        self.refresh()

    def _build_widgets(self):
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        tech_frame = ttk.Labelframe(main, text="Technical View")
        narrative_frame = ttk.Labelframe(main, text="Narrative View")

        main.add(tech_frame, weight=3)
        main.add(narrative_frame, weight=2)

        self.tech_text = tk.Text(tech_frame,
                                 wrap="word",
                                 state="disabled",
                                 background="#101010",
                                 foreground="#dddddd")
        self.tech_text.configure(font=("Courier New", 9))
        self.tech_text.pack(fill=tk.BOTH, expand=True)

        self.narrative_text = tk.Text(narrative_frame,
                                      wrap="word",
                                      state="disabled",
                                      background="#000000",
                                      foreground="#dddddd")
        self.narrative_text.configure(font=("Courier New", 9))
        self.narrative_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        self.refresh_button = ttk.Button(btn_frame, text="Refresh Diagnostics", command=self.refresh)
        self.refresh_button.pack(side=tk.RIGHT)

    def _set_text(self, widget: tk.Text, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state="disabled")

    def refresh(self):
        debug = self.get_debug_callback() or {}
        latest = self.history.latest()
        mean_sample = self.history.rolling_mean()

        # Technical
        tech_lines = []
        tech_lines.append("=== NetHost Technical Diagnostics ===")
        tech_lines.append("")
        tech_lines.append(f"success_ratio: {debug.get('success_ratio', 0.0):.3f}")
        tech_lines.append(f"avg_latency:   {debug.get('avg_latency', 0.0):.3f} s")
        tech_lines.append("")
        tech_lines.append(f"Pi_now:        {debug.get('pi_now', 0.0):.3f}")
        tech_lines.append(f"Pi_short:      {debug.get('pi_short', 0.0):.3f}")
        tech_lines.append(f"Pi_mid:        {debug.get('pi_mid', 0.0):.3f}")
        tech_lines.append(f"Pi_long:       {debug.get('pi_long', 0.0):.3f}")
        tech_lines.append("")
        tech_lines.append(f"reg_now:   {debug.get('reg_now', '?')}")
        tech_lines.append(f"reg_short: {debug.get('reg_short', '?')}")
        tech_lines.append(f"reg_mid:   {debug.get('reg_mid', '?')}")
        tech_lines.append(f"reg_long:  {debug.get('reg_long', '?')}")
        tech_lines.append("")
        tech_lines.append(f"probs_short: {debug.get('probs_short', {})}")
        tech_lines.append(f"probs_mid:   {debug.get('probs_mid', {})}")
        tech_lines.append(f"probs_long:  {debug.get('probs_long', {})}")
        tech_lines.append("")
        tech_lines.append(f"conf_now:   {debug.get('conf_now', 0.0):.3f}")
        tech_lines.append(f"conf_short: {debug.get('conf_short', 0.0):.3f}")
        tech_lines.append(f"conf_mid:   {debug.get('conf_mid', 0.0):.3f}")
        tech_lines.append(f"conf_long:  {debug.get('conf_long', 0.0):.3f}")
        tech_lines.append("")
        tech_lines.append(f"pattern:   {debug.get('pattern', 'n/a')}")
        tech_lines.append(f"anomalies: {debug.get('anomalies', {})}")
        tech_lines.append("")
        tech_lines.append("Recent anomaly episodes:")
        if not self.history.episodes and self.history.current_episode is None:
            tech_lines.append("  (none)")
        else:
            for ep in self.history.episodes[-5:]:
                dur = ep.end_ts - ep.start_ts
                tech_lines.append(
                    f"  Episode {time.strftime('%H:%M:%S', time.localtime(ep.start_ts))}"
                    f" - {time.strftime('%H:%M:%S', time.localtime(ep.end_ts))},"
                    f" duration={dur:.1f}s, types={ep.types}"
                )
            if self.history.current_episode is not None:
                ep = self.history.current_episode
                dur = ep.end_ts - ep.start_ts
                tech_lines.append(
                    f"  (ongoing) {time.strftime('%H:%M:%S', time.localtime(ep.start_ts))}"
                    f" - now, duration={dur:.1f}s, types={ep.types}"
                )

        self._set_text(self.tech_text, "\n".join(tech_lines))

        # Narrative
        narrative_lines = []
        narrative_lines.append("=== NetHost Narrative ===")
        narrative_lines.append("")

        reg_now = debug.get("reg_now", "?")
        conf_now = debug.get("conf_now", 0.0)
        pattern = debug.get("pattern", "neutral")
        anomalies = debug.get("anomalies", {})

        narrative_lines.append(f"Current regime: {reg_now} (confidence {conf_now:.2f}).")

        if pattern == "stable_earth":
            narrative_lines.append("The network atmosphere is stable and Earth-like; flows should be smooth and robust.")
        elif pattern == "recovering_iss":
            narrative_lines.append("The system appears to be recovering from stress; flows are present but still reorganizing.")
        elif pattern == "oscillation":
            narrative_lines.append("The system is oscillating: conditions swing between better and worse, causing jittery behavior.")
        elif pattern == "slow_decay":
            narrative_lines.append("There is a slow decay towards harsher conditions; expect gradual degradation if this continues.")
        elif pattern == "sudden_collapse":
            narrative_lines.append("A recent sudden collapse was detected; this likely corresponds to an abrupt outage or failure event.")
        elif pattern == "partial_outage":
            narrative_lines.append("Connectivity is partially degraded; some paths work, others fail. Expect inconsistent behavior.")
        elif pattern == "high_turbulence":
            narrative_lines.append("Flow is turbulent; even when connectivity exists, jitter and instability may disrupt pipelines.")
        elif pattern == "pre_moon_vacuum":
            narrative_lines.append("Conditions are trending towards a vacuum-like state; live data may soon fail to survive.")
        elif pattern == "insufficient_history":
            narrative_lines.append("There is not yet enough history to characterize the overall pattern confidently.")
        else:
            narrative_lines.append("Conditions are neither strongly stable nor clearly degrading; the system is in a neutral state.")

        if any(anomalies.values()):
            narrative_lines.append("")
            narrative_lines.append("Active anomaly signals:")
            if anomalies.get("latency_spike"):
                narrative_lines.append("- Latency has spiked significantly above recent baseline.")
            if anomalies.get("success_drop"):
                narrative_lines.append("- Success ratio has dropped sharply compared to recent history.")
            narrative_lines.append("These signals suggest increased risk to flowing data.")
        else:
            narrative_lines.append("")
            narrative_lines.append("No strong anomaly signals are active; behavior appears within expected historical bounds.")

        narrative_lines.append("")
        narrative_lines.append("Prediction outlook:")
        reg_short = debug.get("reg_short", "?")
        reg_mid = debug.get("reg_mid", "?")
        reg_long = debug.get("reg_long", "?")
        conf_short = debug.get("conf_short", 0.0)
        conf_mid = debug.get("conf_mid", 0.0)
        conf_long = debug.get("conf_long", 0.0)

        narrative_lines.append(
            f"- Short horizon: likely {reg_short} (confidence {conf_short:.2f})."
        )
        narrative_lines.append(
            f"- Medium horizon: likely {reg_mid} (confidence {conf_mid:.2f})."
        )
        narrative_lines.append(
            f"- Long horizon: likely {reg_long} (confidence {conf_long:.2f})."
        )

        if latest and mean_sample:
            narrative_lines.append("")
            narrative_lines.append(
                "In plain terms: the system is trying to guess whether your network feels "
                "like Earth (smooth streams), ISS (floating blobs of state), or Moon "
                "(vacuum/ice where data struggles to survive) over the next few moments."
            )

        self._set_text(self.narrative_text, "\n".join(narrative_lines))


# ============================================================================
# Autopilot Engine
# ============================================================================


class Autopilot:
    def __init__(self,
                 local_frame: LocalPhysicsFrame,
                 backbone_frame: BackboneFrame,
                 backbone: Backbone,
                 interval_var: tk.DoubleVar,
                 enabled_var: tk.BooleanVar,
                 nethost_history: NetHostHistory,
                 update_nethost_callback):
        self.local_frame = local_frame
        self.backbone_frame = backbone_frame
        self.backbone = backbone
        self.interval_var = interval_var
        self.enabled_var = enabled_var
        self.history = nethost_history
        self.update_nethost_callback = update_nethost_callback

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            interval = max(1.0, self.interval_var.get())
            if self.enabled_var.get():
                self.update_nethost_callback()
                self._autopilot_local()
                self._autopilot_backbone()
            time.sleep(interval)

    def _autopilot_local(self):
        def perturb(value, min_v, max_v, magnitude=0.05):
            jitter = (random.random() - 0.5) * 2 * magnitude
            nv = value + jitter
            return max(min_v, min(max_v, nv))

        p = self.local_frame.pressure_var.get()
        r = self.local_frame.routing_var.get()
        s = self.local_frame.stickiness_var.get()
        v = self.local_frame.volume_var.get()

        p2 = perturb(p, 0.0, 1.0, magnitude=0.03)
        r2 = perturb(r, 0.0, 1.5, magnitude=0.08)
        s2 = perturb(s, 0.05, 1.5, magnitude=0.08)
        v2 = perturb(v, 1.0, 10.0, magnitude=0.5)

        def apply_and_run():
            self.local_frame.pressure_var.set(p2)
            self.local_frame.routing_var.set(r2)
            self.local_frame.stickiness_var.set(s2)
            self.local_frame.volume_var.set(v2)
            self.local_frame.update_regime_preview()
            self.local_frame.on_simulate(prefix="Autopilot Local")

        self.local_frame.after(0, apply_and_run)

    def _autopilot_backbone(self):
        names = list(self.backbone.nodes.keys())
        if len(names) < 2:
            return

        src = None
        if "NetHost" in self.backbone.nodes:
            src = "NetHost"

        if src is None:
            risk_sorted = sorted(
                self.backbone.nodes.values(),
                key=lambda n: (n.flow_rate(), -n.turbulence())
            )
            if risk_sorted:
                src = risk_sorted[0].name

        if src is None:
            src = random.choice(names)

        other = [n for n in names if n != src]
        if not other:
            return
        tgt = random.choice(other)
        volume = random.uniform(3.0, 15.0)

        def apply_and_run():
            self.backbone_frame.source_var.set(src)
            self.backbone_frame.target_var.set(tgt)
            self.backbone_frame.volume_var.set(volume)
            self.backbone_frame.on_simulate(prefix="Autopilot Backbone")

        self.backbone_frame.after(0, apply_and_run)


# ============================================================================
# Unified main window (with Diagnostics tab)
# ============================================================================


class UnifiedApp(tk.Tk):
    def __init__(self, backbone: Backbone, nethost_history: NetHostHistory, initial_debug: dict):
        super().__init__()
        self.backbone = backbone
        self.nethost_history = nethost_history
        self.net_debug = initial_debug

        self.title("Backbone Total Unified Intelligent – Water/Data Physics Engine")
        self.geometry("1400x800")
        self.minsize(1150, 650)

        self._build_style()
        self.autopilot_enabled = tk.BooleanVar(value=True)
        self.autopilot_interval = tk.DoubleVar(value=10.0)

        self._build_widgets()

        self.autopilot = Autopilot(
            local_frame=self.local_frame,
            backbone_frame=self.backbone_frame,
            backbone=self.backbone,
            interval_var=self.autopilot_interval,
            enabled_var=self.autopilot_enabled,
            nethost_history=self.nethost_history,
            update_nethost_callback=self.update_nethost_from_internet,
        )

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    def _build_widgets(self):
        top_bar = ttk.Frame(self, padding=(8, 4))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Autopilot:").pack(side=tk.LEFT, padx=(0, 4))

        autopilot_check = ttk.Checkbutton(
            top_bar,
            text="Enabled",
            variable=self.autopilot_enabled
        )
        autopilot_check.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top_bar, text="Interval (seconds):").pack(side=tk.LEFT, padx=(0, 4))

        interval_scale = ttk.Scale(
            top_bar,
            from_=2.0,
            to=60.0,
            orient=tk.HORIZONTAL,
            variable=self.autopilot_interval
        )
        interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.interval_label = ttk.Label(top_bar, text="10.0")
        self.interval_label.pack(side=tk.LEFT, padx=(0, 4))

        def update_interval_label(_event=None):
            self.interval_label.config(text=f"{self.autopilot_interval.get():.1f}")

        interval_scale.bind("<B1-Motion>", update_interval_label)
        interval_scale.bind("<ButtonRelease-1>", update_interval_label)

        host_info = f"{platform.system()} {platform.release()} | Python {platform.python_version()}"
        host_label = ttk.Label(top_bar, text=host_info, anchor="w")
        host_label.pack(side=tk.LEFT, padx=(12, 4))

        self.netinfo_label = ttk.Label(top_bar, text="", anchor="e")
        self.netinfo_label.pack(side=tk.RIGHT, padx=(4, 4))

        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.local_frame = LocalPhysicsFrame(notebook)
        self.backbone_frame = BackboneFrame(notebook, self.backbone)
        self.diagnostics_frame = DiagnosticsFrame(
            notebook,
            self.nethost_history,
            get_debug_callback=lambda: self.net_debug
        )

        notebook.add(self.local_frame, text="Local Physics")
        notebook.add(self.backbone_frame, text="Backbone Map")
        notebook.add(self.diagnostics_frame, text="Diagnostics")

        self.refresh_netinfo_label()

    def refresh_netinfo_label(self):
        debug = self.net_debug or {}
        sr = debug.get("success_ratio", 0.0)
        lat = debug.get("avg_latency", 0.0)
        pi_now = debug.get("pi_now", 0.0)
        conf_now = debug.get("conf_now", 0.0)
        reg_now = debug.get("reg_now", "?")
        reg_short = debug.get("reg_short", "?")
        anomalies = debug.get("anomalies", {})
        alert_flags = [k for k, v in anomalies.items() if v]
        alert_text = " | ALERT: " + ",".join(alert_flags) if alert_flags else ""

        text = (
            f"NetHost: success={sr:.2f}, lat={lat:.3f}s, "
            f"reg_now={reg_now}, Pi_now={pi_now:.2f}, "
            f"conf={conf_now:.2f}, next={reg_short}{alert_text}"
        )
        self.netinfo_label.config(text=text, foreground="#ff4444" if alert_flags else "#dddddd")

    def update_nethost_from_internet(self):
        history = self.nethost_history
        pressure, routing, stickiness, debug = compute_nethost_node_params(history)
        self.net_debug = debug

        def apply():
            node = self.backbone.get_node("NetHost")
            if node:
                node.pressure = pressure
                node.routing_pull = routing
                node.stickiness = stickiness
            self.backbone_frame.refresh_node_list()
            self.backbone_frame.refresh_flow_map()
            self.refresh_netinfo_label()
            self.diagnostics_frame.refresh()

            latest = history.latest()
            if latest:
                record = {
                    "ts": latest.ts,
                    "mode": "nethost_sample",
                    "success_ratio": latest.success_ratio,
                    "avg_latency": latest.avg_latency,
                    "pressure": latest.pressure,
                    "routing": latest.routing_pull,
                    "stickiness": latest.stickiness,
                    "pi": latest.pi,
                    "regime": latest.regime,
                    "pi_short": debug.get("pi_short", latest.pi),
                    "pi_mid": debug.get("pi_mid", latest.pi),
                    "pi_long": debug.get("pi_long", latest.pi),
                    "reg_now": debug.get("reg_now", ""),
                    "reg_short": debug.get("reg_short", ""),
                    "reg_mid": debug.get("reg_mid", ""),
                    "reg_long": debug.get("reg_long", ""),
                    "conf_now": debug.get("conf_now", 0.0),
                    "conf_short": debug.get("conf_short", 0.0),
                    "conf_mid": debug.get("conf_mid", 0.0),
                    "conf_long": debug.get("conf_long", 0.0),
                    "pattern": debug.get("pattern", ""),
                    "anomalies": debug.get("anomalies", {}),
                }
                append_history_record(record)

        self.after(0, apply)

    def on_close(self):
        if hasattr(self, "autopilot") and self.autopilot:
            self.autopilot.stop()
        self.destroy()


# ============================================================================
# main
# ============================================================================


def main():
    nethost_history = NetHostHistory(maxlen=50)
    backbone, debug = create_example_backbone(nethost_history)
    app = UnifiedApp(backbone, nethost_history, debug)
    app.mainloop()


if __name__ == "__main__":
    main()

