#!/usr/bin/env python3
"""
Backbone Total Unified – Fully Autonomous Water/Data Physics Engine
-------------------------------------------------------------------

Water → Data
Gravity → Routing pull (global data attraction)
Surface tension → Stickiness (local data trapping)
Pressure → Environment compatibility (protocols, routes, health)
Viscosity → Data movement resistance (serialization, protocol overhead, format friction)
Turbulence → Instability in flow (jitter, packet loss, retries, burstiness)
Flow rate → Effective throughput given environment and forces
Phase → Data usability (liquid, vapor, ice)

This file combines:

- Core physics:
    * Bond-like number Bo
    * Phase factor S_phase
    * Pouring index Pi_pour = S_phase * tanh(Bo)
    * Extended metrics: viscosity, turbulence, flow_rate
    * Regime classification: Earth / ISS / Moon

- Local physics simulator:
    * Sliders: pressure, routing_pull, stickiness, volume
    * Presets: Earth, ISS, Moon
    * Detailed logs including viscosity, turbulence, flow rate

- Backbone graph:
    * Nodes with physics parameters
    * Special "NetHost" node driven by LIVE internet data (reachability + latency)
    * Edges with bandwidth and reliability
    * Pour simulation along all paths with extended fluid metrics

- GUI:
    * Tab 1: Local Physics (controls + log)
    * Tab 2: Backbone Map (node table, topology view, simulation log)

- Autopilot:
    * Periodically perturbs local environment and simulates a local pour
    * Periodically picks random backbone source→target and simulates a pour
    * Controlled via Autopilot Enabled + Interval slider
    * Logs all autopilot cycles and user-triggered simulations to JSONL history

Standard library only: tkinter, socket, json, threading, math, time, random, etc.
Works on Windows and Linux when tkinter is available.
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
    """
    Synthetic Bond number analogue:
        Bo = (routing_pull * scale^2) / stickiness
    Larger Bo → routing pull dominates → streams/puddles.
    Smaller Bo → stickiness dominates → blobs/local state.
    """
    stickiness = max(stickiness, 1e-3)
    return (routing_pull * (scale ** 2)) / stickiness


def phase_stability(pressure: float) -> float:
    """
    Phase factor S_phase as function of pressure (0..1):
        S_phase = 2 * pressure - 1, clamped to [-1, 1].
    Lower pressure → hostile (vacuum-like).
    Higher pressure → stable atmosphere.
    """
    s = 2.0 * pressure - 1.0
    return max(-1.0, min(1.0, s))


def pouring_index(pressure: float, routing_pull: float, stickiness: float) -> float:
    """
    Unified pouring index:
        Pi_pour = S_phase * tanh(Bo)

    Interpret:
      ≈ +1 → Earth: stable liquid, strong flows (streams + puddles)
      ≈  0 → ISS: liquid, but blobs/local state (weak global pull)
      ≈ -1 → Moon: hostile, vapor/ice (errors, dead storage)
    """
    s_phase = phase_stability(pressure)
    bo = bond_number(routing_pull, stickiness)
    return s_phase * math.tanh(bo)


def viscosity(pressure: float, stickiness: float) -> float:
    """
    Synthetic viscosity:
      - Higher when stickiness is high and pressure is low.
      - Lower when environment is supportive and stickiness is low.
    Rough mapping into [0, 1.5].
    """
    inv_pressure = 1.0 - max(0.0, min(1.0, pressure))
    base = stickiness * (0.5 + inv_pressure)
    return max(0.0, min(1.5, base))


def turbulence(pressure: float, routing_pull: float, stickiness: float) -> float:
    """
    Synthetic turbulence:
      - Higher when routing_pull and stickiness are both moderate/high
        (competing forces) OR when pressure is marginal.
      - Lower when routing_pull clearly dominates or stickiness clearly dominates
        in a stable pressure regime.
    Rough mapping into [0, 1].
    """
    p = max(0.0, min(1.0, pressure))
    r = max(0.0, min(1.5, routing_pull)) / 1.5  # normalize to 0..1
    s = max(0.05, min(1.5, stickiness)) / 1.5   # normalize to 0..1

    # Competition term: largest when r and s are similar and non-trivial
    competition = 1.0 - abs(r - s)
    competition *= (r + s) / 2.0

    # Marginal pressure term: highest near p ~ 0.4..0.6
    marginal_p = 1.0 - abs(p - 0.5) * 2.0
    marginal_p = max(0.0, marginal_p)

    turb = 0.6 * competition + 0.4 * marginal_p
    return max(0.0, min(1.0, turb))


def flow_rate(pressure: float, routing_pull: float, stickiness: float) -> float:
    """
    Synthetic flow_rate:
      - Increases with routing_pull and pressure.
      - Decreases with viscosity (which is tied to stickiness and low pressure).
    Rough mapping into [0, 1].
    """
    v = viscosity(pressure, stickiness)
    p = max(0.0, min(1.0, pressure))
    r_norm = max(0.0, min(1.5, routing_pull)) / 1.5
    # Basic potential flow from pressure and routing
    potential = 0.5 * p + 0.5 * r_norm
    # Penalize with viscosity
    flow = potential * (1.0 - min(1.0, v / 1.5))
    return max(0.0, min(1.0, flow))


def classify_regime(pi: float, pressure: float) -> Tuple[str, str]:
    """
    Map Pi_pour + pressure to regime and color.
    Regimes: Earth / ISS / Moon
    """
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
    """
    Simulate a local 'bucket' pour with given environment.
    Includes extended metrics: viscosity, turbulence, flow_rate.
    """
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
        logs.append("Data cannot exist comfortably as live, usable objects here.")
        logs.append("Instead, it tends to:")
        logs.append("  - Boil off into errors, logs, retries (vapor).")
        logs.append("  - Freeze into backups, dumps, quarantined blobs (ice).")
        logs.append("")
        logs.append("Result of the pour:")
        logs.append("  - No smooth stream forms between components.")
        logs.append("  - You observe scattered errors, partial writes, and dead artifacts.")
        return logs

    if regime == "Earth":
        logs.append("Environment supports stable, liquid data.")
        logs.append("Global routing pull dominates over local stickiness.")
        logs.append(f"Flow context: moderate viscosity ({visc:.2f}),"
                    f" flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result of the pour:")
        logs.append("  - Data forms a coherent stream from source to sinks.")
        logs.append("  - Pipelines, queues, and storage 'puddles' emerge naturally.")
        logs.append("  - Backpressure shapes the flow, but does not prevent it.")
    elif regime == "ISS":
        logs.append("Environment supports liquid data, but global routing is weak.")
        logs.append("Local stickiness competes with or beats routing pull.")
        logs.append(f"Flow context: higher viscosity ({visc:.2f}),"
                    f" patchy flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result of the pour:")
        logs.append("  - Data forms isolated 'blobs' of local state and caches.")
        logs.append("  - Movement is patchy and event-driven, not a continuous stream.")
        logs.append("  - Data sticks to components until explicit triggers move it.")
    else:
        logs.append("Environment is marginal and unstable for liquid data.")
        logs.append("Some data survives as liquid, but much is forced into ice/vapor forms.")
        logs.append(f"Flow context: high viscosity ({visc:.2f}),"
                    f" low flow_rate {fr:.2f}, turbulence {turb:.2f}.")
        logs.append("")
        logs.append("Result of the pour:")
        logs.append("  - Data flow is fragile: partial pipelines, intermittent success.")
        logs.append("  - Significant fraction of data becomes errors or cold artifacts.")

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
    reliability: float = 0.9  # 0..1


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
        """
        DFS to find paths from source to target up to max_depth.
        """
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

            # Decide how node's regime affects remaining_volume and reliability
            if reg == "Moon":
                logs.append("    Environment hostile to liquid data at this node.")
                logs.append("    Data here tends to become errors/logs (vapor) or dead storage (ice).")
                remaining_volume *= 0.3 * (0.5 + fr)  # flow mitigates slightly
                cumulative_reliability *= 0.5 * (0.7 + (1.0 - turb) * 0.3)
            elif reg == "ISS":
                logs.append("    Liquid data but no strong global pull.")
                logs.append("    Data tends to get trapped as local state/caches.")
                remaining_volume *= 0.7 * (0.5 + fr)
                cumulative_reliability *= 0.8 * (0.7 + (1.0 - turb) * 0.3)
            else:  # Earth
                logs.append("    Stable liquid data and strong routing pull.")
                logs.append("    Node participates in coherent streams and storage puddles.")
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
                    # Edge reliability interacts with turbulence and flow
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
# Internet sensors (NetHost node parameters, stdlib only)
# ============================================================================


def check_host_latency(host: str, port: int, timeout: float = 1.0) -> Optional[float]:
    """
    Try to open a TCP connection to host:port once.
    Returns latency in seconds if success, None if fail.
    """
    start = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return time.time() - start
    except OSError:
        return None


def sample_internet_status() -> Tuple[float, float]:
    """
    Check a few well-known public endpoints and return:
      - success_ratio: fraction of reachable endpoints (0..1)
      - avg_latency: average latency (seconds) or default if none succeed
    """
    targets = [
        ("1.1.1.1", 53),       # Cloudflare DNS
        ("8.8.8.8", 53),       # Google DNS
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


def compute_nethost_node_params() -> Tuple[float, float, float, dict]:
    """
    Derive (pressure, routing_pull, stickiness) for the special 'NetHost' node
    based ONLY on live internet data.

    Returns:
      (pressure, routing_pull, stickiness, debug_dict)
    """
    success_ratio, avg_latency = sample_internet_status()

    # Normalize latency: assume 0..1.5 sec range
    lat_norm = max(0.0, min(1.0, avg_latency / 1.5))

    # Pressure: base on success and latency
    pressure = success_ratio * (1.0 - 0.5 * lat_norm)
    pressure = max(0.0, min(1.0, pressure))

    # Routing pull: more connectivity → more pull outward
    routing_pull = 0.2 + 1.3 * success_ratio * (1.0 - 0.3 * lat_norm)  # ~0.2..1.5
    routing_pull = max(0.0, min(1.5, routing_pull))

    # Stickiness: bad internet → high stickiness (data is forced to remain local)
    stickiness = 0.2 + (1.0 - success_ratio) * 1.3 + lat_norm * 0.7
    stickiness = max(0.05, min(1.5, stickiness))

    debug = {
        "success_ratio": success_ratio,
        "avg_latency": avg_latency,
        "lat_norm": lat_norm,
    }
    return pressure, routing_pull, stickiness, debug


# ============================================================================
# Example backbone (with NetHost node using live internet)
# ============================================================================


def create_example_backbone() -> Tuple[Backbone, dict]:
    """
    Example topology. Plus a special 'NetHost' node driven by live internet data.
    Returns:
      (backbone, nethost_debug_info)
    """
    b = Backbone()

    # Special NetHost node using internet sensors
    n_pressure, n_routing, n_stickiness, debug = compute_nethost_node_params()
    b.add_node(Node("NetHost", pressure=n_pressure,
                    routing_pull=n_routing,
                    stickiness=n_stickiness))

    # Earth-like core (synthetic)
    b.add_node(Node("Ingest",      pressure=0.95, routing_pull=1.0,  stickiness=0.2))
    b.add_node(Node("PipelineA",   pressure=0.90, routing_pull=0.9,  stickiness=0.3))
    b.add_node(Node("StorageLake", pressure=0.92, routing_pull=0.8,  stickiness=0.4))

    # ISS-like blob side
    b.add_node(Node("ServiceX", pressure=0.90, routing_pull=0.1, stickiness=0.9))
    b.add_node(Node("CacheX",   pressure=0.90, routing_pull=0.05, stickiness=1.2))

    # Moon-like hostile zone
    b.add_node(Node("Quarantine",  pressure=0.05, routing_pull=0.6, stickiness=0.5))
    b.add_node(Node("DeepArchive", pressure=0.10, routing_pull=0.3, stickiness=0.8))

    # Edges (NetHost as an upstream root)
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


HISTORY_FILE = "backbone_history.jsonl"
HISTORY_LOCK = threading.Lock()


def append_history_record(record: dict):
    """
    Append a JSON record to history file in a thread-safe way.
    """
    try:
        with HISTORY_LOCK:
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        # Observability, not critical
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

        # Presets
        preset_frame = ttk.LabelFrame(controls_frame, text="Presets")
        preset_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        ttk.Button(preset_frame, text="Earth", command=self.set_earth).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(preset_frame, text="ISS",   command=self.set_iss).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(preset_frame, text="Moon",  command=self.set_moon).pack(fill=tk.X, padx=4, pady=2)

        # Sliders
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
                "- Environment is hostile to liquid data.\n"
                "- Data evaporates into errors/logs or freezes as dead storage.\n"
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
# Autopilot Engine
# ============================================================================


class Autopilot:
    """
    Periodically:
      - Perturbs local physics parameters slightly and runs a local sim.
      - Picks random backbone source→target and runs a sim.
      - Logs to persistent history file.
    """

    def __init__(self,
                 local_frame: LocalPhysicsFrame,
                 backbone_frame: BackboneFrame,
                 backbone: Backbone,
                 interval_var: tk.DoubleVar,
                 enabled_var: tk.BooleanVar):
        self.local_frame = local_frame
        self.backbone_frame = backbone_frame
        self.backbone = backbone
        self.interval_var = interval_var
        self.enabled_var = enabled_var
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            interval = max(0.5, self.interval_var.get())
            if self.enabled_var.get():
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
        src, tgt = random.sample(names, 2)
        volume = random.uniform(3.0, 15.0)

        def apply_and_run():
            self.backbone_frame.source_var.set(src)
            self.backbone_frame.target_var.set(tgt)
            self.backbone_frame.volume_var.set(volume)
            self.backbone_frame.on_simulate(prefix="Autopilot Backbone")

        self.backbone_frame.after(0, apply_and_run)


# ============================================================================
# Unified main window (with Autopilot)
# ============================================================================


class UnifiedApp(tk.Tk):
    def __init__(self, backbone: Backbone, nethost_debug: dict):
        super().__init__()
        self.backbone = backbone
        self.nethost_debug = nethost_debug

        self.title("Backbone Total Unified – Autonomous Water/Data Physics Engine")
        self.geometry("1350x780")
        self.minsize(1100, 650)

        self._build_style()
        self.autopilot_enabled = tk.BooleanVar(value=True)
        self.autopilot_interval = tk.DoubleVar(value=6.0)

        self._build_widgets()

        self.autopilot = Autopilot(
            local_frame=self.local_frame,
            backbone_frame=self.backbone_frame,
            backbone=self.backbone,
            interval_var=self.autopilot_interval,
            enabled_var=self.autopilot_enabled,
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
            to=30.0,
            orient=tk.HORIZONTAL,
            variable=self.autopilot_interval
        )
        interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.interval_label = ttk.Label(top_bar, text="6.0")
        self.interval_label.pack(side=tk.LEFT, padx=(0, 4))

        def update_interval_label(_event=None):
            self.interval_label.config(text=f"{self.autopilot_interval.get():.1f}")

        interval_scale.bind("<B1-Motion>", update_interval_label)
        interval_scale.bind("<ButtonRelease-1>", update_interval_label)

        # Host info (OS and Python)
        host_info = f"{platform.system()} {platform.release()} | Python {platform.python_version()}"
        host_label = ttk.Label(top_bar, text=host_info, anchor="w")
        host_label.pack(side=tk.LEFT, padx=(12, 4))

        # NetHost debug info
        netinfo_text = (
            f"NetHost: success={self.nethost_debug.get('success_ratio', 0.0):.2f}, "
            f"latency={self.nethost_debug.get('avg_latency', 0.0):.3f}s"
        )
        netinfo_label = ttk.Label(top_bar, text=netinfo_text, anchor="e")
        netinfo_label.pack(side=tk.RIGHT, padx=(4, 4))

        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.local_frame = LocalPhysicsFrame(notebook)
        self.backbone_frame = BackboneFrame(notebook, self.backbone)

        notebook.add(self.local_frame, text="Local Physics")
        notebook.add(self.backbone_frame, text="Backbone Map")

    def on_close(self):
        if hasattr(self, "autopilot") and self.autopilot:
            self.autopilot.stop()
        self.destroy()


# ============================================================================
# main
# ============================================================================


def main():
    backbone, debug = create_example_backbone()
    app = UnifiedApp(backbone, debug)
    app.mainloop()


if __name__ == "__main__":
    main()

