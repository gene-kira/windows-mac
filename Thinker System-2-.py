import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ================================
# Persistent Memory Manager (Local + SMB, configurable)
# ================================

class MemoryManager:
    """
    MPRO Persistent Memory System
    - Local drive path (configurable)
    - SMB/network path (optional, configurable)
    - JSON-based state storage
    """

    def __init__(self, local_path: str = "mpro_memory.json", smb_path: Optional[str] = None):
        self.local_path = local_path
        self.smb_path = smb_path
        self.state = {
            "learned_patterns": [],
            "engine_stats": {},
            "history": [],
            "meta_state_history": [],
            "glitches": [],
            "mazes": {}  # maze_id -> fingerprint
        }

    def set_paths(self, local_path: Optional[str] = None, smb_path: Optional[str] = None):
        """
        Update paths at runtime (used by GUI 'pick location' buttons).
        """
        if local_path:
            self.local_path = local_path
        if smb_path is not None:
            self.smb_path = smb_path

    def load_memory(self) -> str:
        """
        Try load from local; if fails, try SMB; otherwise keep fresh state.
        Returns a status message.
        """
        if self.local_path and os.path.exists(self.local_path):
            try:
                with open(self.local_path, "r") as f:
                    self.state = json.load(f)
                return f"Loaded memory from local drive: {self.local_path}"
            except Exception:
                pass

        if self.smb_path and os.path.exists(self.smb_path):
            try:
                with open(self.smb_path, "r") as f:
                    self.state = json.load(f)
                return f"Loaded memory from SMB/network path: {self.smb_path}"
            except Exception:
                pass

        return "No previous memory found. Starting fresh."

    def save_memory(self):
        """
        Save to local, then SMB if configured.
        """
        if self.local_path:
            try:
                with open(self.local_path, "w") as f:
                    json.dump(self.state, f, indent=4)
            except Exception:
                pass

        if self.smb_path:
            try:
                with open(self.smb_path, "w") as f:
                    json.dump(self.state, f, indent=4)
            except Exception:
                pass

    def record_event(self, event: Dict[str, Any]):
        self.state["history"].append(event)

    def record_engine_stats(self, engine_name: str, success: bool = True):
        stats = self.state["engine_stats"].setdefault(engine_name, {"success": 0, "fail": 0})
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1

    def record_pattern(self, pattern: Dict[str, Any]):
        self.state["learned_patterns"].append(pattern)

    def record_meta_state(self, meta_state: str, context: Optional[Dict[str, Any]] = None):
        self.state["meta_state_history"].append({
            "meta_state": meta_state,
            "context": context or {}
        })

    def record_glitch(self, glitch_info: Dict[str, Any]):
        self.state["glitches"].append(glitch_info)

    def store_maze_fingerprint(self, maze_id: str, fingerprint: Dict[str, Any]):
        self.state["mazes"][maze_id] = fingerprint

    def get_all_mazes(self) -> Dict[str, Dict[str, Any]]:
        return self.state.get("mazes", {})


# ================================
# Meta-State Manager
# ================================

class MetaStateManager:
    """
    Meta-state manager with simple emotional inertia + glitch awareness.
    """

    def __init__(self, memory: Optional[MemoryManager] = None):
        self.state = "calm"
        self.memory = memory

    def get_state(self) -> str:
        return self.state

    def update_state_on_result(self, fusion_result: Dict[str, Any]):
        previous_state = self.state

        has_glitch = fusion_result.get("has_glitch", False)
        chosen = fusion_result.get("final_choice", {})
        candidate = chosen.get("candidate", {}) if chosen else {}
        category = candidate.get("category", "")

        if has_glitch:
            self.state = "glitch_alert"
        else:
            if category == "cheat_logic":
                self.state = "exploratory"
            elif category in ("standard", "identity", "survival", "defensive", "preservation", "maze_recognition"):
                if previous_state == "glitch_alert":
                    self.state = "recovery"
                else:
                    self.state = "calm"
            elif category == "meta":
                self.state = "glitch_alert"
            else:
                if previous_state == "glitch_alert":
                    self.state = "recovery"
                else:
                    self.state = "calm"

        if self.memory:
            self.memory.record_meta_state(self.state, {
                "previous_state": previous_state,
                "has_glitch": has_glitch,
                "chosen_category": category
            })
            self.memory.save_memory()


# ================================
# Base Reasoning Engine
# ================================

class ReasoningEngine(ABC):
    @abstractmethod
    def can_handle(self, problem: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass


# ================================
# Engine 1: Math Target == 1 (with difficulty)
# ================================

class MathTargetOneEngine(ReasoningEngine):
    """
    Generates multiple mathematical expressions that evaluate to a given target
    (default: 1). Annotated with difficulty (1=easy, 2=medium, 3=hard, 4=extreme).
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "math_target"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        target = problem.get("target", 1)
        candidates = []

        if target == 1:
            candidates.extend([
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Subtraction path",
                    "expression": "2 - 1",
                    "value": 2 - 1,
                    "difficulty": 1
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Division path",
                    "expression": "5 / 5",
                    "value": 5 / 5,
                    "difficulty": 1
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Exponent zero path",
                    "expression": "99 ** 0",
                    "value": 99 ** 0,
                    "difficulty": 1
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "symbolic",
                    "description": "Algebraic identity path",
                    "expression": "x / x for x != 0 (symbolic)",
                    "value": 1,
                    "difficulty": 2
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "identity",
                    "description": "Trigonometric identity path",
                    "expression": "cos^2(theta) + sin^2(theta)",
                    "value": 1,
                    "difficulty": 3
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Square root path",
                    "expression": "sqrt(1)",
                    "value": 1,
                    "difficulty": 1
                }
            ])
        else:
            candidates.extend([
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Addition path",
                    "expression": f"{target - 1} + 1",
                    "value": (target - 1) + 1,
                    "difficulty": 1
                },
                {
                    "engine": "MathTargetOneEngine",
                    "category": "standard",
                    "description": "Multiplication path",
                    "expression": f"{target} * 1",
                    "value": target * 1,
                    "difficulty": 1
                }
            ])

        return candidates


# ================================
# Engine 2: Obstacle Manhole
# ================================

class ObstacleManholeEngine(ReasoningEngine):
    """
    Generates multiple strategies to move from A to B when a manhole is in the way.
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "obstacle" and problem.get("obstacle") == "manhole"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        start = problem.get("from", "A")
        end = problem.get("to", "B")
        candidates = []

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "physical",
            "strategy": "go_around",
            "description": f"Walk around the manhole from {start} to {end}.",
            "difficulty": 1
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "physical",
            "strategy": "step_over",
            "description": "Step over the manhole cover if it is stable and safe.",
            "difficulty": 1
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "physical",
            "strategy": "temporary_remove_cover",
            "description": "Lift and temporarily move the cover with proper tools, pass safely, then replace it.",
            "difficulty": 2
        })

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "path_rewrite",
            "strategy": "reroute_path",
            "description": f"Choose an alternate route that avoids the area around the manhole between {start} and {end}.",
            "difficulty": 2
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "path_rewrite",
            "strategy": "go_above",
            "description": "Use an elevated path to bypass the manhole entirely.",
            "difficulty": 2
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "path_rewrite",
            "strategy": "go_below",
            "description": "Use an underground or indoor path to go under the street.",
            "difficulty": 2
        })

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "constraint_bending",
            "strategy": "change_medium",
            "description": "Use a drone or remote vehicle that bypasses the physical risk of the manhole.",
            "difficulty": 3
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "constraint_bending",
            "strategy": "redefine_goal",
            "description": f"Send information or a proxy from {start} to {end} instead of your physical body.",
            "difficulty": 3
        })

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "tools",
            "strategy": "safety_equipment",
            "description": "Use safety gear (harness, rope, barriers) to mitigate risk while passing near the manhole.",
            "difficulty": 2
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "tools",
            "strategy": "stability_check",
            "description": "Inspect or use sensors to determine if the cover is stable before stepping on or near it.",
            "difficulty": 2
        })

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "delegation",
            "strategy": "request_assistance",
            "description": "Ask maintenance or authorities to secure or fix the manhole, then proceed once it is safe.",
            "difficulty": 1
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "delegation",
            "strategy": "send_proxy",
            "description": "Send a proxy (person, robot, or service) from A to B instead of you.",
            "difficulty": 2
        })

        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "meta",
            "strategy": "log_and_learn",
            "description": "Treat the manhole as a learning event: mark this location as risky and update future planning.",
            "difficulty": 2
        })
        candidates.append({
            "engine": "ObstacleManholeEngine",
            "category": "meta",
            "strategy": "simulate_before_action",
            "description": "Internally simulate navigation options, evaluate risk and cost, and choose the best path.",
            "difficulty": 3
        })

        return candidates


# ================================
# Engine 3: Cheat-Hack Engine
# ================================

class CheatHackEngine(ReasoningEngine):
    """
    Cheat-Hack Engine:
    Generates unconventional, frame-breaking, perspective-shifting solutions
    that still converge to the same final answer or goal.
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return True

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = []

        if problem.get("type") == "math_target":
            target = problem.get("target", 1)

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "description": "Redefine the number system so the target is an identity element.",
                "expression": f"In a custom algebraic system, define all values to collapse to {target}.",
                "value": target,
                "difficulty": 3
            })

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "description": "Use a limit expression that converges to the target.",
                "expression": f"limit of ({target} + 1/n) as n → ∞",
                "value": target,
                "difficulty": 4
            })

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "description": "Self-referential identity: define x directly as the target.",
                "expression": f"x = {target}",
                "value": target,
                "difficulty": 2
            })

        if problem.get("type") == "obstacle":
            obstacle = problem.get("obstacle", "obstacle")
            start = problem.get("from", "A")
            end = problem.get("to", "B")

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "strategy": "redefine_space",
                "description": f"Redefine the coordinate system so the {obstacle} is no longer between {start} and {end}.",
                "difficulty": 4
            })

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "strategy": "teleport_goal",
                "description": f"Move point {end} to your current location. You have now reached {end} without moving.",
                "difficulty": 3
            })

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "strategy": "invert_problem",
                "description": f"Instead of going from {start} to {end}, send {end} (or what it represents) to {start}.",
                "difficulty": 3
            })

            candidates.append({
                "engine": "CheatHackEngine",
                "category": "cheat_logic",
                "strategy": "abstract_equivalence",
                "description": "Represent the journey symbolically instead of physically. The symbolic traversal completes instantly.",
                "difficulty": 2
            })

        return candidates


# ================================
# Engine 4: Game Tactical Engine (generic)
# ================================

class GameTacticalEngine(ReasoningEngine):
    """
    Reads game state and proposes tactical actions:
    - position / movement
    - team behavior
    - risk posture
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "game_state"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        ctx = problem.get("context", "unknown")
        hp = problem.get("player_health", 100)
        team_status = problem.get("team_status", "grouped")
        ammo = problem.get("ammo_level", "normal")
        threat = problem.get("threat_level", "medium")
        objective = problem.get("objective", problem.get("distance_to_safe_room", "unknown"))

        candidates = []

        candidates.append({
            "engine": "GameTacticalEngine",
            "category": "survival",
            "strategy": "regroup_and_heal",
            "description": "Regroup with team, prioritize healing and resupply before pushing objective.",
            "recommended_action": "regroup",
            "notes": f"context={ctx}, hp={hp}, ammo={ammo}, team={team_status}, threat={threat}, objective={objective}",
            "difficulty": 2
        })

        candidates.append({
            "engine": "GameTacticalEngine",
            "category": "aggressive",
            "strategy": "fast_push",
            "description": "Push quickly toward the objective while exploiting enemy gaps. High risk, high reward.",
            "recommended_action": "push_forward",
            "notes": f"context={ctx}, hp={hp}, ammo={ammo}, threat={threat}, objective={objective}",
            "difficulty": 3
        })

        candidates.append({
            "engine": "GameTacticalEngine",
            "category": "control",
            "strategy": "hold_and_kite",
            "description": "Hold a defensible position, kite enemies, and thin them before moving.",
            "recommended_action": "hold_position",
            "notes": f"context={ctx}, hp={hp}, team={team_status}, threat={threat}",
            "difficulty": 3
        })

        candidates.append({
            "engine": "GameTacticalEngine",
            "category": "tactical_split",
            "strategy": "split_and_flank",
            "description": "Split team: one unit draws aggro, the other flanks or secures objective.",
            "recommended_action": "split_team",
            "notes": f"context={ctx}, team={team_status}, objective={objective}",
            "difficulty": 4
        })

        return candidates


# ================================
# Engine 5: System Guardian Risk Engine
# ================================

class GuardianRiskEngine(ReasoningEngine):
    """
    System guardian decision engine:
    - handles alerts, error spikes, latency
    - proposes mitigation, throttling, routing, degradation
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "system_guardian"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = problem.get("alerts", [])
        latency = problem.get("latency_ms", 0)
        error_rate = problem.get("error_rate", 0.0)
        critical = problem.get("critical_services", [])

        candidates = []

        candidates.append({
            "engine": "GuardianRiskEngine",
            "category": "containment",
            "strategy": "rate_limit_noncritical",
            "description": "Apply rate limiting or degradation to non-critical services to protect core systems.",
            "recommended_action": "throttle_non_critical",
            "notes": f"alerts={alerts}, latency={latency}, error_rate={error_rate}, critical={critical}",
            "difficulty": 2
        })

        candidates.append({
            "engine": "GuardianRiskEngine",
            "category": "scale_out",
            "strategy": "add_capacity",
            "description": "Add capacity/instances to relieve load and reduce latency if resources allow.",
            "recommended_action": "scale_out",
            "notes": f"alerts={alerts}, latency={latency}",
            "difficulty": 3
        })

        candidates.append({
            "engine": "GuardianRiskEngine",
            "category": "failover",
            "strategy": "reroute_traffic",
            "description": "Reroute traffic away from degraded nodes or regions to healthier ones.",
            "recommended_action": "reroute",
            "notes": f"alerts={alerts}",
            "difficulty": 3
        })

        candidates.append({
            "engine": "GuardianRiskEngine",
            "category": "observe",
            "strategy": "increase_observability_and_delay_action",
            "description": "Increase logging/metrics and confirm if spike is transient before heavy interventions.",
            "recommended_action": "observe_then_decide",
            "notes": f"error_rate={error_rate}, latency={latency}",
            "difficulty": 2
        })

        return candidates


# ================================
# Engine 6: Network Adaptation Engine
# ================================

class NetworkAdaptEngine(ReasoningEngine):
    """
    Network adaptation decision engine:
    - reacts to latency, packet loss, bandwidth
    - proposes QoS, bitrate, path, or mode changes
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "network_event"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        latency = problem.get("latency_ms", 0)
        loss = problem.get("packet_loss", 0.0)
        bw = problem.get("bandwidth_mbps", 0)
        session_type = problem.get("session_type", "generic")

        candidates = []

        candidates.append({
            "engine": "NetworkAdaptEngine",
            "category": "quality_adjust",
            "strategy": "lower_bitrate",
            "description": "Lower bitrate/quality to fit current bandwidth and reduce sensitivity to loss.",
            "recommended_action": "reduce_quality",
            "notes": f"latency={latency}, loss={loss}, bandwidth={bw}, session={session_type}",
            "difficulty": 2
        })

        candidates.append({
            "engine": "NetworkAdaptEngine",
            "category": "routing",
            "strategy": "switch_route",
            "description": "Change routing path or server region to find lower latency or more stable path.",
            "recommended_action": "change_route",
            "notes": f"latency={latency}, loss={loss}",
            "difficulty": 3
        })

        candidates.append({
            "engine": "NetworkAdaptEngine",
            "category": "mode_shift",
            "strategy": "add_buffer",
            "description": "Introduce more buffering to trade latency for smoother experience.",
            "recommended_action": "increase_buffer",
            "notes": f"session={session_type}, latency={latency}",
            "difficulty": 2
        })

        candidates.append({
            "engine": "NetworkAdaptEngine",
            "category": "aggressive",
            "strategy": "maintain_quality",
            "description": "Maintain current quality and monitor. Accept temporary degradation.",
            "recommended_action": "hold_quality",
            "notes": f"latency={latency}, loss={loss}, bandwidth={bw}",
            "difficulty": 3
        })

        return candidates


# ================================
# Engine 7: Scenario Decision Engine
# ================================

class ScenarioDecisionEngine(ReasoningEngine):
    """
    Abstract strategic decision engine:
    - given options, goal, time pressure, info clarity
    - proposes which option aligns with risk/goal posture
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "decision_scenario"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        goal = problem.get("goal", "unknown")
        time_pressure = problem.get("time_pressure", "medium")
        clarity = problem.get("info_clarity", "medium")
        options = problem.get("options", [])

        candidates = []

        for opt in options:
            if "push" in opt:
                category = "aggressive"
                description = "High-risk, high-reward option, better under high time pressure and moderate clarity."
                difficulty = 3
            elif "hold" in opt:
                category = "defensive"
                description = "Stabilizing option, safer when info clarity is low."
                difficulty = 2
            elif "fall_back" in opt or "retreat" in opt:
                category = "preservation"
                description = "Preserve resources and regroup when conditions are strongly unfavorable."
                difficulty = 2
            else:
                category = "neutral"
                description = "Balanced option with moderate risk and payoff."
                difficulty = 2

            candidates.append({
                "engine": "ScenarioDecisionEngine",
                "category": category,
                "strategy": f"choose_{opt}",
                "description": description,
                "recommended_action": opt,
                "notes": f"goal={goal}, time_pressure={time_pressure}, info_clarity={clarity}",
                "difficulty": difficulty
            })

        return candidates

# ================================
# Engine 8: Back 4 Blood Outcome Prediction Engine
# ================================

class B4BOutcomePredictionEngine(ReasoningEngine):
    """
    Predicts Back 4 Blood outcomes (survive vs wipe) based on deep state.
    Uses effective health, trauma, downs, threat, team state, econ, cards.
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "game_state" and problem.get("context") == "back4blood"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        eff_hp = problem.get("player_health", 100)
        trauma = problem.get("trauma", 0)
        down_count = problem.get("down_count", 0)
        max_downs = problem.get("max_downs", 3)

        threat = problem.get("threat_level", "medium")
        team_status = problem.get("team_status", "grouped")
        distance = problem.get("distance_to_safe_room", "mid")
        econ = problem.get("econ_state", "normal")
        support = problem.get("support_state", "medium")

        active_cards = problem.get("active_cards", [])
        corruption_cards = problem.get("corruption_cards", [])

        survival_score = 0.0

        if eff_hp > 60:
            survival_score += 2.0
        elif eff_hp > 30:
            survival_score += 1.0

        if trauma < 30:
            survival_score += 1.0

        if down_count == 0:
            survival_score += 1.0
        elif down_count < max_downs - 1:
            survival_score += 0.5

        if team_status in ("grouped", "mixed"):
            survival_score += 1.5

        if econ in ("normal", "rich"):
            survival_score += 0.5
        if support in ("medium", "high"):
            survival_score += 1.0

        if "Face Your Fears" in active_cards or "Amped Up" in active_cards:
            survival_score += 0.5
        if "Combat Knife" in active_cards:
            survival_score += 0.5

        if threat == "high":
            survival_score -= 2.0
        elif threat == "low":
            survival_score += 1.0

        if distance == "near":
            survival_score += 1.0
        elif distance == "far":
            survival_score -= 0.5

        wipe_score = 0.0

        if eff_hp < 30:
            wipe_score += 2.0
        if trauma > 50:
            wipe_score += 1.5
        if down_count >= max_downs - 1:
            wipe_score += 2.0

        if team_status in ("split", "critical"):
            wipe_score += 2.0

        if threat == "high":
            wipe_score += 2.0

        if "Hordes" in corruption_cards or "Boss" in corruption_cards:
            wipe_score += 1.0

        if distance == "far" and econ == "poor":
            wipe_score += 1.0

        total = max(survival_score + wipe_score, 0.001)
        survival_conf = max(survival_score / total, 0.0)
        wipe_conf = max(wipe_score / total, 0.0)

        return [
            {
                "engine": "B4BOutcomePredictionEngine",
                "category": "prediction",
                "strategy": "predict_survival_b4b",
                "description": "B4B-specific estimate of reaching safe room alive.",
                "predicted_outcome": "survive",
                "confidence": survival_conf,
                "notes": f"survival_score={survival_score:.2f}, wipe_score={wipe_score:.2f}",
                "difficulty": 3
            },
            {
                "engine": "B4BOutcomePredictionEngine",
                "category": "prediction",
                "strategy": "predict_wipe_b4b",
                "description": "B4B-specific estimate of team wipe or failure.",
                "predicted_outcome": "wipe",
                "confidence": wipe_conf,
                "notes": f"survival_score={survival_score:.2f}, wipe_score={wipe_score:.2f}",
                "difficulty": 3
            }
        ]


# ================================
# Engine 9: Generic Survival Probability Engine
# ================================

class SurvivalProbabilityEngine(ReasoningEngine):
    """
    Generic survival probability engine:
    - Looks at player_health, ammo_level, and progress (distance/objective)
    - Estimates probability to survive vs die.
    Works for any 'game_state' problem (Back4Blood, Fortnite, CS:GO mapped in).
    """

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "game_state"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        hp = problem.get("player_health", 100)
        ammo_level = problem.get("ammo_level", "medium")
        context = problem.get("context", "unknown")

        distance = problem.get("distance_to_safe_room", None)
        objective = problem.get("objective", "")

        ammo_factor = 0.0
        if ammo_level == "low":
            ammo_factor = -1.0
        elif ammo_level == "medium":
            ammo_factor = 0.0
        elif ammo_level == "high":
            ammo_factor = 1.0

        progress_factor = 0.0
        if distance:
            if distance == "near":
                progress_factor = 1.0
            elif distance == "mid":
                progress_factor = 0.0
            elif distance == "far":
                progress_factor = -1.0
        else:
            obj = objective.lower()
            if "near" in obj or "final" in obj:
                progress_factor = 1.0
            elif "mid" in obj:
                progress_factor = 0.0
            elif "far" in obj or "early" in obj:
                progress_factor = -1.0

        survival_score = 0.0
        if hp > 70:
            survival_score += 2.0
        elif hp > 40:
            survival_score += 1.0
        elif hp > 20:
            survival_score += 0.3
        else:
            survival_score -= 1.5

        survival_score += 0.8 * ammo_factor
        survival_score += 0.7 * progress_factor

        normalized = (survival_score + 3.0) / 7.0
        survival_prob = min(max(normalized, 0.0), 1.0)
        death_prob = 1.0 - survival_prob

        return [
            {
                "engine": "SurvivalProbabilityEngine",
                "category": "prediction",
                "strategy": "generic_survival",
                "description": "Generic probability that the player survives given HP, ammo, and progress.",
                "predicted_outcome": "survive",
                "probability": survival_prob,
                "notes": f"context={context}, hp={hp}, ammo={ammo_level}, distance={distance}, objective={objective}",
                "difficulty": 2
            },
            {
                "engine": "SurvivalProbabilityEngine",
                "category": "prediction",
                "strategy": "generic_death",
                "description": "Generic probability that the player dies or fails given HP, ammo, and progress.",
                "predicted_outcome": "death",
                "probability": death_prob,
                "notes": f"context={context}, hp={hp}, ammo={ammo_level}, distance={distance}, objective={objective}",
                "difficulty": 2
            }
        ]


# ================================
# Engine 10: Maze Equivalence & Recognition Engine
# ================================

class MazeEquivalenceEngine(ReasoningEngine):
    """
    Maze recognition & structural equivalence engine:
    - Stores maze fingerprints in memory
    - Detects if a new maze is structurally equivalent/reversed to a known one
    - Estimates recognition_time_seconds
    """

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory = memory_manager

    def can_handle(self, problem: Dict[str, Any]) -> bool:
        return problem.get("type") == "maze"

    def generate_candidates(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.memory:
            return []

        maze_id = problem.get("maze_id", "unknown_maze")
        width = problem.get("width", 10)
        height = problem.get("height", 10)
        complexity = float(problem.get("complexity", 0.5))
        symmetry = float(problem.get("symmetry_score", 0.5))
        direction = problem.get("direction", "normal")

        fingerprint = self._build_fingerprint(maze_id, width, height, complexity, symmetry, direction)

        known_mazes = self.memory.get_all_mazes()
        best_match_id = None
        best_match_score = 0.0
        reversed_match = False

        for known_id, known_fp in known_mazes.items():
            if known_id == maze_id:
                continue
            score, is_reversed = self._compare_fingerprints(fingerprint, known_fp)
            if score > best_match_score:
                best_match_score = score
                best_match_id = known_id
                reversed_match = is_reversed

        self.memory.store_maze_fingerprint(maze_id, fingerprint)
        self.memory.save_memory()

        recognition_time = self._estimate_recognition_time(width, height, complexity, symmetry, best_match_score)

        candidates = []

        if best_match_id:
            candidates.append({
                "engine": "MazeEquivalenceEngine",
                "category": "maze_recognition",
                "strategy": "recognize_equivalent_maze",
                "description": f"Recognized maze as structurally equivalent to '{best_match_id}'.",
                "maze_id": maze_id,
                "matched_maze_id": best_match_id,
                "similarity_score": best_match_score,
                "is_reversed_equivalent": reversed_match,
                "recognition_time_seconds": recognition_time,
                "difficulty": 3
            })
        else:
            candidates.append({
                "engine": "MazeEquivalenceEngine",
                "category": "maze_recognition",
                "strategy": "no_match_found",
                "description": "No structurally equivalent maze found in memory.",
                "maze_id": maze_id,
                "similarity_score": 0.0,
                "is_reversed_equivalent": False,
                "recognition_time_seconds": None,
                "difficulty": 2
            })

        return candidates

    def _build_fingerprint(self, maze_id, width, height, complexity, symmetry, direction):
        aspect_ratio = width / max(height, 1)
        normalized_size = (width * height) / 10000.0

        return {
            "maze_id": maze_id,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "normalized_size": normalized_size,
            "complexity": complexity,
            "symmetry": symmetry,
            "direction": direction
        }

    def _compare_fingerprints(self, fp_new: Dict[str, Any], fp_old: Dict[str, Any]):
        size_diff = abs(fp_new["normalized_size"] - fp_old["normalized_size"])
        aspect_diff = abs(fp_new["aspect_ratio"] - fp_old["aspect_ratio"])
        complexity_diff = abs(fp_new["complexity"] - fp_old["complexity"])
        symmetry_diff = abs(fp_new["symmetry"] - fp_old["symmetry"])

        size_score = max(0.0, 1.0 - size_diff * 5.0)
        aspect_score = max(0.0, 1.0 - aspect_diff * 2.0)
        complexity_score = max(0.0, 1.0 - complexity_diff * 3.0)
        symmetry_score = max(0.0, 1.0 - symmetry_diff * 2.0)

        similarity_score = (size_score + aspect_score + complexity_score + symmetry_score) / 4.0

        dir_new = fp_new.get("direction", "normal")
        dir_old = fp_old.get("direction", "normal")
        is_reversed_equivalent = (dir_new != dir_old)

        if is_reversed_equivalent and similarity_score > 0.6:
            similarity_score = min(1.0, similarity_score + 0.1)

        return similarity_score, is_reversed_equivalent

    def _estimate_recognition_time(self, width, height, complexity, symmetry, similarity_score):
        base_time = 0.1
        node_count = width * height
        complexity_factor = (node_count / 1000.0) * (0.2 + complexity)
        symmetry_penalty = symmetry * 1.5
        similarity_bonus = similarity_score * 2.0

        recognition_time = base_time + complexity_factor + symmetry_penalty - similarity_bonus
        return max(0.05, recognition_time)


# ================================
# Game Environment Adapter Base
# ================================

class GameEnvironmentAdapter(ABC):
    def __init__(self, game_id: str):
        self.game_id = game_id

    @abstractmethod
    def game_state_to_problem(self, raw_state: dict) -> dict:
        pass

    @abstractmethod
    def decision_to_game_action(self, mpro_result: dict) -> dict:
        pass


# ================================
# Back 4 Blood Adapter
# ================================

class Back4BloodAdapter(GameEnvironmentAdapter):
    """
    Deep Back 4 Blood adapter:
    maps real B4B state into MPRO game_state + optional decision_scenario.
    """

    def __init__(self):
        super().__init__("back4blood")

    def _compute_effective_health(self, hp, temp_hp, trauma):
        max_hp = 100 - trauma
        if max_hp < 0:
            max_hp = 0
        effective_hp = min(hp + temp_hp, max_hp)
        return max(0, effective_hp), max_hp

    def _derive_threat_level(self, horde_pressure, specials_present, special_count, boss_present, corruption_cards):
        threat = "medium"

        if horde_pressure == "high":
            threat = "high"
        if specials_present and special_count >= 2:
            threat = "high"
        if boss_present:
            threat = "high"

        if "Fog" in corruption_cards or "Hordes" in corruption_cards:
            if threat == "medium":
                threat = "high"

        if horde_pressure == "low" and not specials_present and not boss_present:
            threat = "low"

        return threat

    def _derive_team_status(self, team_split, team_incapacitated, bots_present):
        if team_incapacitated > 0:
            return "critical"
        if team_split:
            return "split"
        if bots_present > 0:
            return "mixed"
        return "grouped"

    def _derive_resource_state(self, ammo_state, copper, healing_items, utility_items):
        econ = "normal"
        if copper < 300:
            econ = "poor"
        elif copper > 1000:
            econ = "rich"

        support = "low"
        if healing_items + utility_items >= 3:
            support = "high"
        elif healing_items + utility_items >= 1:
            support = "medium"

        return econ, support

    def game_state_to_problem(self, raw_state: dict) -> dict:
        hp = raw_state.get("hp", 100)
        trauma = raw_state.get("trauma", 0)
        temp_hp = raw_state.get("temp_hp", 0)
        down_count = raw_state.get("down_count", 0)
        max_downs = raw_state.get("max_downs", 3)

        ammo_state = raw_state.get("ammo_state", "medium")
        copper = raw_state.get("copper", 0)
        healing_items = raw_state.get("healing_items", 0)
        utility_items = raw_state.get("utility_items", 0)

        team_split = raw_state.get("team_split", False)
        team_incapacitated = raw_state.get("team_incapacitated", 0)
        bots_present = raw_state.get("bots_present", 0)

        horde_pressure = raw_state.get("horde_pressure", "medium")
        specials_present = raw_state.get("specials_present", False)
        special_count = raw_state.get("special_count", 0)
        boss_present = raw_state.get("boss_present", False)
        mutations_seen = raw_state.get("mutations_seen", [])

        distance_to_safe_room = raw_state.get("distance_to_safe_room", "mid")
        current_chapter = raw_state.get("current_chapter", "Unknown")
        map_type = raw_state.get("map_type", "travel")

        active_cards = raw_state.get("active_cards", [])
        corruption_cards = raw_state.get("corruption_cards", [])

        effective_hp, max_hp = self._compute_effective_health(hp, temp_hp, trauma)
        threat_level = self._derive_threat_level(horde_pressure, specials_present, special_count, boss_present, corruption_cards)
        team_status = self._derive_team_status(team_split, team_incapacitated, bots_present)
        econ_state, support_state = self._derive_resource_state(ammo_state, copper, healing_items, utility_items)

        problem = {
            "type": "game_state",
            "context": "back4blood",
            "player_health": effective_hp,
            "player_max_health": max_hp,
            "trauma": trauma,
            "down_count": down_count,
            "max_downs": max_downs,

            "ammo_level": ammo_state,
            "copper": copper,
            "econ_state": econ_state,
            "support_state": support_state,

            "team_status": team_status,
            "team_incapacitated": team_incapacitated,
            "bots_present": bots_present,

            "threat_level": threat_level,
            "horde_pressure": horde_pressure,
            "specials_present": specials_present,
            "special_count": special_count,
            "boss_present": boss_present,
            "mutations_seen": mutations_seen,

            "distance_to_safe_room": distance_to_safe_room,
            "map_type": map_type,
            "current_chapter": current_chapter,

            "active_cards": active_cards,
            "corruption_cards": corruption_cards
        }

        return problem

    def build_decision_scenario(self, raw_state: dict) -> dict:
        distance = raw_state.get("distance_to_safe_room", "mid")
        horde = raw_state.get("horde_pressure", "medium")
        specials_present = raw_state.get("specials_present", False)

        time_pressure = "medium"
        if horde == "high" or specials_present:
            time_pressure = "high"
        elif distance == "far":
            time_pressure = "medium"
        else:
            time_pressure = "low"

        info_clarity = "medium"
        if "Fog" in raw_state.get("corruption_cards", []):
            info_clarity = "low"

        options = [
            "push_forward",
            "hold_position",
            "fall_back",
            "regroup_team"
        ]

        scenario = {
            "type": "decision_scenario",
            "goal": "reach_safe_room_with_team_alive",
            "time_pressure": time_pressure,
            "info_clarity": info_clarity,
            "options": options
        }
        return scenario

    def decision_to_game_action(self, mpro_result: dict) -> dict:
        final = mpro_result.get("final_choice")
        if not final:
            return {"summary": "[B4B] No clear recommendation.", "action": None}

        cand = final["candidate"]
        action = cand.get("recommended_action") or cand.get("strategy") or "unknown"

        summary = f"[B4B] Recommended: {action} | Engine={cand.get('engine')} | Category={cand.get('category')}"

        return {
            "summary": summary,
            "action": action,
            "engine": cand.get("engine"),
            "category": cand.get("category"),
            "notes": cand.get("notes", cand.get("description", "")),
        }


# ================================
# Fusion Layer (Hybrid + Difficulty + Glitch Detection)
# ================================

class FusionLayer:
    """
    Neural-style hybrid fusion:
    - Weighted logic
    - Probabilistic exploration
    - Reinforcement-style learning (engine_stats)
    - Meta-state modulation
    - Difficulty preference
    - Glitch detection
    """

    def __init__(self, memory_manager=None, meta_state_getter=None):
        self.memory = memory_manager
        self.get_meta_state = meta_state_getter

    def fuse(self, problem: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        meta_state = self.get_meta_state() if self.get_meta_state else "calm"

        ranked = []
        for c in candidates:
            components, score = self._score_candidate(problem, c, meta_state)
            ranked.append({
                "candidate": c,
                "score": score,
                "components": components
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        final_choice = ranked[0] if ranked else None

        has_glitch, glitches = self._detect_glitches(problem, ranked)

        result = {
            "problem": problem,
            "meta_state": meta_state,
            "final_choice": final_choice,
            "ranked_candidates": ranked,
            "has_glitch": has_glitch,
            "glitches": glitches
        }

        if self.memory:
            self._log_fusion_decision(result)

        return result

    def _difficulty_preference_mod(self, difficulty: int, meta_state: str) -> float:
        mod = 1.0

        if meta_state in ("calm", "recovery"):
            if difficulty == 1:
                mod += 0.3
            elif difficulty == 2:
                mod += 0.1
            elif difficulty >= 3:
                mod -= 0.2

        elif meta_state == "exploratory":
            if difficulty == 1:
                mod -= 0.1
            elif difficulty == 2:
                mod += 0.2
            elif difficulty == 3:
                mod += 0.3
            elif difficulty == 4:
                mod += 0.1

        elif meta_state == "aggressive":
            if difficulty == 1:
                mod -= 0.2
            elif difficulty == 2:
                mod -= 0.1
            elif difficulty == 3:
                mod += 0.3
            elif difficulty == 4:
                mod += 0.4

        elif meta_state == "glitch_alert":
            if difficulty == 1:
                mod += 0.4
            elif difficulty == 2:
                mod += 0.2
            elif difficulty >= 3:
                mod -= 0.3

        return max(0.1, mod)

    def _score_candidate(self, problem: Dict[str, Any], c: Dict[str, Any], meta_state: str):
        engine_name = c.get("engine", "UnknownEngine")
        category = c.get("category", "unknown")
        strategy = c.get("strategy", c.get("description", ""))

        logic_score = 1.0
        ptype = problem.get("type")

        if ptype == "obstacle":
            if category == "physical":
                logic_score += 0.5
            if category == "meta":
                logic_score += 0.3
            if category == "cheat_logic":
                logic_score += 0.2

        elif ptype == "math_target":
            if category in ("standard", "identity"):
                logic_score += 0.5
            if category == "cheat_logic":
                logic_score += 0.3

        elif ptype == "game_state":
            if category == "survival":
                logic_score += 0.5
            if category == "control":
                logic_score += 0.3
            if category == "aggressive":
                logic_score += 0.2
            if category == "prediction":
                logic_score += 0.3

        elif ptype == "system_guardian":
            if category == "containment":
                logic_score += 0.5
            if category == "observe":
                logic_score += 0.2
            if category == "failover":
                logic_score += 0.3

        elif ptype == "network_event":
            if category == "quality_adjust":
                logic_score += 0.5
            if category == "routing":
                logic_score += 0.3
            if category == "mode_shift":
                logic_score += 0.2

        elif ptype == "decision_scenario":
            if category == "defensive":
                logic_score += 0.4
            if category == "preservation":
                logic_score += 0.4
            if category == "aggressive":
                logic_score += 0.3

        elif ptype == "maze":
            if category == "maze_recognition":
                logic_score += 0.5

        if self.memory:
            stats = self.memory.state.get("engine_stats", {}).get(engine_name, None)
            if stats:
                total = stats["success"] + stats["fail"]
                if total > 0:
                    success_rate = stats["success"] / total
                    reliability_score = 1.0 + success_rate
                else:
                    reliability_score = 1.0
            else:
                reliability_score = 1.0
        else:
            reliability_score = 1.0

        novelty_score = 0.0
        category_hash = abs(hash(category + str(strategy))) % 1000
        novelty_score += (category_hash / 1000.0) * 0.5
        novelty_score += random.uniform(0.0, 0.5)

        meta_mod = 1.0
        if meta_state == "exploratory":
            if category in ("cheat_logic", "meta", "aggressive", "maze_recognition"):
                meta_mod += 0.3
        elif meta_state == "cautious":
            if category in ("cheat_logic", "aggressive"):
                meta_mod -= 0.3
        elif meta_state == "aggressive":
            if category in ("cheat_logic", "physical", "aggressive"):
                meta_mod += 0.3
        elif meta_state == "glitch_alert":
            if category in ("meta", "observe", "control", "defensive", "maze_recognition"):
                meta_mod += 0.4
        elif meta_state == "recovery":
            if category in ("standard", "identity", "survival", "defensive", "preservation"):
                meta_mod += 0.4

        difficulty = int(c.get("difficulty", 2))
        diff_mod = self._difficulty_preference_mod(difficulty, meta_state)

        base_score = logic_score + reliability_score + novelty_score
        final_score = base_score * meta_mod * diff_mod

        components = {
            "logic_score": logic_score,
            "reliability_score": reliability_score,
            "novelty_score": novelty_score,
            "meta_mod": meta_mod,
            "difficulty": difficulty,
            "difficulty_mod": diff_mod,
            "final_score": final_score
        }
        return components, final_score

    def _detect_glitches(self, problem: Dict[str, Any], ranked: List[Dict[str, Any]]):
        glitches = []
        has_glitch = False

        if not ranked:
            has_glitch = True
            glitches.append("No candidates generated for this problem. Possible regime failure.")

        if len(ranked) >= 2:
            top = ranked[0]
            second = ranked[1]
            score_diff = abs(top["score"] - second["score"])
            top_cat = top["candidate"].get("category", "")
            second_cat = second["candidate"].get("category", "")
            if score_diff < 0.1 and top_cat != second_cat:
                has_glitch = True
                glitches.append(
                    f"High-scoring candidates from conflicting categories: {top_cat} vs {second_cat} with similar scores."
                )

        engine_counts = {}
        for entry in ranked:
            eng = entry["candidate"].get("engine", "UnknownEngine")
            engine_counts[eng] = engine_counts.get(eng, 0) + 1

        if engine_counts:
            max_engine = max(engine_counts, key=engine_counts.get)
            if engine_counts[max_engine] > len(ranked) * 0.8:
                has_glitch = True
                glitches.append(
                    f"One engine ({max_engine}) dominates {engine_counts[max_engine]} of {len(ranked)} candidates."
                )

        if self.memory and has_glitch:
            glitch_info = {
                "problem": problem,
                "glitches": glitches
            }
            self.memory.record_glitch(glitch_info)
            self.memory.save_memory()

        return has_glitch, glitches

    def _log_fusion_decision(self, result: Dict[str, Any]):
        chosen_engine = None
        chosen_score = None
        if result["final_choice"]:
            chosen_engine = result["final_choice"]["candidate"].get("engine")
            chosen_score = result["final_choice"]["score"]

        event = {
            "type": "fusion_decision",
            "problem": result["problem"],
            "meta_state": result["meta_state"],
            "chosen_engine": chosen_engine,
            "chosen_score": chosen_score,
            "has_glitch": result.get("has_glitch", False)
        }
        self.memory.record_event(event)
        self.memory.save_memory()


# ================================
# Many-Path Reasoning Organism (MPRO)
# ================================

class ManyPathReasoningOrganism:
    def __init__(self, memory_manager=None,
                 fusion_layer: Optional[FusionLayer] = None,
                 meta_state_manager: Optional[MetaStateManager] = None):
        self.engines: List[ReasoningEngine] = []
        self.memory = memory_manager
        self.meta_state_manager = meta_state_manager or MetaStateManager(memory_manager)
        self.fusion_layer = fusion_layer or FusionLayer(
            memory_manager=self.memory,
            meta_state_getter=self.meta_state_manager.get_state
        )

        if self.memory:
            msg = self.memory.load_memory()
            print(msg)

    def register_engine(self, engine: ReasoningEngine):
        self.engines.append(engine)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        all_candidates = []

        for engine in self.engines:
            if engine.can_handle(problem):
                try:
                    candidates = engine.generate_candidates(problem)
                    all_candidates.extend(candidates)
                    if self.memory:
                        self.memory.record_engine_stats(engine.__class__.__name__, success=True)
                except Exception:
                    if self.memory:
                        self.memory.record_engine_stats(engine.__class__.__name__, success=False)

        fusion_result = self.fusion_layer.fuse(problem, all_candidates)
        self.meta_state_manager.update_state_on_result(fusion_result)

        return {
            "problem": problem,
            "meta_state": fusion_result["meta_state"],
            "final_choice": fusion_result["final_choice"],
            "ranked_candidates": fusion_result["ranked_candidates"],
            "has_glitch": fusion_result["has_glitch"],
            "glitches": fusion_result["glitches"]
        }
# ================================
# Unified World Problem Builder
# ================================

class WorldProblemBuilder:
    """
    Converts raw world state into one composite 'thought' plus
    derived engine-specific problem views.
    """

    def __init__(self, game_adapters: Dict[str, GameEnvironmentAdapter]):
        self.game_adapters = game_adapters

    def build(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        active_game = world_state["world"].get("active_game")

        # 1) Game problem
        if active_game and active_game in self.game_adapters:
            adapter = self.game_adapters[active_game]
            raw_game_state = world_state["game"]["raw_state"]
            game_problem = adapter.game_state_to_problem(raw_game_state)
        else:
            game_problem = {
                "type": "game_state",
                "context": world_state["game"].get("context", "none"),
                "player_health": world_state["game"].get("player_health", 100),
                "team_status": world_state["game"].get("team_status", "unknown"),
                "ammo_level": world_state["game"].get("ammo_level", "medium"),
                "threat_level": world_state["game"].get("threat_level", "medium"),
                "objective": world_state["game"].get("objective", "none")
            }

        # 2) System guardian problem
        system = world_state["system"]
        guardian_problem = {
            "type": "system_guardian",
            "alerts": system.get("alerts", []),
            "latency_ms": system.get("net_latency_ms", 0),
            "error_rate": system.get("error_rate", 0.0),
            "critical_services": system.get("critical_services", [])
        }

        # 3) Network problem
        net = world_state["network"]
        network_problem = {
            "type": "network_event",
            "event": "latency_spike" if net.get("latency_ms", 0) > 150 else "normal",
            "latency_ms": net.get("latency_ms", 0),
            "packet_loss": net.get("packet_loss", 0.0),
            "bandwidth_mbps": net.get("bandwidth_mbps", 0),
            "session_type": net.get("session_type", "generic")
        }

        # 4) Scenario problem
        scenario_problem = {
            "type": "decision_scenario",
            "goal": "maximize_survivability_and_stability",
            "time_pressure": self._infer_time_pressure(game_problem, guardian_problem),
            "info_clarity": "medium",
            "options": ["push_forward", "hold_position", "fall_back", "regroup_team"]
        }

        # 5) Maze problem (optional)
        maze_problem = None
        if "maze" in world_state:
            maze = world_state["maze"]
            maze_problem = {
                "type": "maze",
                "maze_id": maze.get("maze_id", "maze_unknown"),
                "width": maze.get("width", 10),
                "height": maze.get("height", 10),
                "complexity": maze.get("complexity", 0.5),
                "symmetry_score": maze.get("symmetry_score", 0.5),
                "direction": maze.get("direction", "normal")
            }

        return {
            "game_problem": game_problem,
            "guardian_problem": guardian_problem,
            "network_problem": network_problem,
            "scenario_problem": scenario_problem,
            "maze_problem": maze_problem
        }

    def _infer_time_pressure(self, game_problem, guardian_problem):
        threat = game_problem.get("threat_level", "medium")
        alerts = guardian_problem.get("alerts", [])
        if threat == "high" or "high_cpu" in alerts:
            return "high"
        return "medium"


# ================================
# Autonomous Controller
# ================================

class AutonomousController:
    """
    Fully autonomous loop:
    - collects world state
    - builds composite problems
    - runs MPRO
    - logs / reacts
    """

    def __init__(self, mpro: ManyPathReasoningOrganism, problem_builder: WorldProblemBuilder):
        self.mpro = mpro
        self.problem_builder = problem_builder
        self.running = False

    def collect_world_state(self) -> Dict[str, Any]:
        """
        Placeholder: replace with real telemetry hooks.
        """
        world = {
            "active_game": "back4blood",
            "processes": [],
            "timestamp": time.time()
        }

        game = {
            "context": "back4blood",
            "player_health": 62,
            "team_status": "split",
            "ammo_level": "low",
            "threat_level": "high",
            "objective": "reach_safe_room_far",
            "raw_state": {
                "hp": 62,
                "trauma": 35,
                "temp_hp": 10,
                "down_count": 1,
                "max_downs": 3,
                "ammo_state": "low",
                "copper": 450,
                "healing_items": 1,
                "utility_items": 2,
                "team_split": True,
                "team_incapacitated": 0,
                "bots_present": 1,
                "horde_pressure": "high",
                "specials_present": True,
                "special_count": 3,
                "boss_present": False,
                "mutations_seen": ["Reeker", "Stinger"],
                "distance_to_safe_room": "far",
                "current_chapter": "Act1-3",
                "map_type": "travel",
                "active_cards": ["Combat Knife", "Broadside"],
                "corruption_cards": ["Hordes", "Fog"]
            }
        }

        system = {
            "cpu_load": 0.73,
            "mem_usage": 0.81,
            "net_latency_ms": 85,
            "packet_loss": 0.01,
            "error_rate": 0.02,
            "alerts": ["high_cpu"],
            "critical_services": ["auth", "matchmaking"]
        }

        network = {
            "session_type": "game_stream",
            "latency_ms": 85,
            "packet_loss": 0.01,
            "bandwidth_mbps": 80
        }

        maze = {
            "maze_id": "maze_001",
            "width": 20,
            "height": 20,
            "complexity": 0.6,
            "symmetry_score": 0.3,
            "direction": "normal"
        }

        return {
            "world": world,
            "game": game,
            "system": system,
            "network": network,
            "maze": maze
        }

    def run_once(self):
        world_state = self.collect_world_state()
        problems = self.problem_builder.build(world_state)

        results = {}
        for key, sub_problem in problems.items():
            if sub_problem is None:
                continue
            results[key] = self.mpro.solve(sub_problem)

        return results

    def run_loop(self, interval_seconds: float = 1.0):
        self.running = True
        while self.running:
            self.run_once()
            time.sleep(interval_seconds)

    def stop(self):
        self.running = False


# ================================
# GUI with Manual SMB/Local Path Pickers
# ================================

class MPROGUI:
    def __init__(self, root, organism: ManyPathReasoningOrganism, memory: MemoryManager, controller: AutonomousController):
        self.root = root
        self.organism = organism
        self.memory = memory
        self.controller = controller

        self.root.title("MPRO Organism Brain Panel")

        self.local_path_var = tk.StringVar(value=self.memory.local_path)
        self.smb_path_var = tk.StringVar(value=self.memory.smb_path or "")

        self.meta_state_var = tk.StringVar(value="calm")
        self.final_engine_var = tk.StringVar(value="-")
        self.final_desc_var = tk.StringVar(value="-")
        self.final_score_var = tk.StringVar(value="-")
        self.glitch_flag_var = tk.StringVar(value="No")

        self.candidates = []

        self._build_layout()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Memory path controls
        mem_frame = ttk.LabelFrame(main, text="Memory Paths", padding=10)
        mem_frame.pack(fill=tk.X)

        ttk.Label(mem_frame, text="Local Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(mem_frame, textvariable=self.local_path_var, width=40).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(mem_frame, text="Browse", command=self.pick_local_path).grid(row=0, column=2, padx=5)

        ttk.Label(mem_frame, text="SMB Path:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(mem_frame, textvariable=self.smb_path_var, width=40).grid(row=1, column=1, sticky=tk.W)
        ttk.Button(mem_frame, text="Browse", command=self.pick_smb_path).grid(row=1, column=2, padx=5)

        ttk.Button(mem_frame, text="Apply Paths", command=self.apply_paths).grid(row=2, column=1, pady=5)

        # Autonomous controls
        auto_frame = ttk.LabelFrame(main, text="Autonomous Control", padding=10)
        auto_frame.pack(fill=tk.X, pady=10)

        ttk.Button(auto_frame, text="Start Autonomous Mode", command=self.start_auto).grid(row=0, column=0, padx=5)
        ttk.Button(auto_frame, text="Stop Autonomous Mode", command=self.stop_auto).grid(row=0, column=1, padx=5)
        ttk.Button(auto_frame, text="Run One Cycle", command=self.run_once).grid(row=0, column=2, padx=5)

        # Status
        status_frame = ttk.LabelFrame(main, text="Organism State", padding=10)
        status_frame.pack(fill=tk.X)

        ttk.Label(status_frame, text="Meta-state:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.meta_state_var, foreground="blue").grid(row=0, column=1, sticky=tk.W)

        ttk.Label(status_frame, text="Glitch detected:").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.glitch_flag_var, foreground="red").grid(row=0, column=3, sticky=tk.W)

        ttk.Label(status_frame, text="Final engine:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.final_engine_var).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(status_frame, text="Final score:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.final_score_var).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(status_frame, text="Final description:").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.final_desc_var, wraplength=400).grid(row=3, column=1, columnspan=3, sticky=tk.W)

        # Candidate list
        cand_frame = ttk.LabelFrame(main, text="Candidates", padding=10)
        cand_frame.pack(fill=tk.BOTH, expand=True)

        self.candidate_list = tk.Listbox(cand_frame, height=15)
        self.candidate_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.candidate_list.bind("<<ListboxSelect>>", self.on_candidate_select)

        scrollbar = ttk.Scrollbar(cand_frame, orient=tk.VERTICAL, command=self.candidate_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.candidate_list.config(yscrollcommand=scrollbar.set)

        # Details
        details_frame = ttk.LabelFrame(main, text="Candidate Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True)

        self.details_text = tk.Text(details_frame, height=10, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    # Path pickers
    def pick_local_path(self):
        path = filedialog.asksaveasfilename(title="Select Local Memory File")
        if path:
            self.local_path_var.set(path)

    def pick_smb_path(self):
        path = filedialog.asksaveasfilename(title="Select SMB/Network Memory File")
        if path:
            self.smb_path_var.set(path)

    def apply_paths(self):
        self.memory.set_paths(
            local_path=self.local_path_var.get(),
            smb_path=self.smb_path_var.get()
        )
        self.memory.save_memory()
        messagebox.showinfo("Paths Applied", "Memory paths updated successfully.")

    # Autonomous controls
    def start_auto(self):
        self.controller.running = True
        self._auto_loop()

    def _auto_loop(self):
        if not self.controller.running:
            return
        self.run_once()
        self.root.after(1000, self._auto_loop)

    def stop_auto(self):
        self.controller.running = False

    def run_once(self):
        results = self.controller.run_once()
        game_result = results.get("game_problem")
        if game_result:
            self.update_display(game_result)

    def update_display(self, result):
        self.meta_state_var.set(result.get("meta_state", "unknown"))
        self.glitch_flag_var.set("Yes" if result.get("has_glitch") else "No")

        final = result.get("final_choice")
        if final:
            cand = final["candidate"]
            self.final_engine_var.set(cand.get("engine", "-"))
            self.final_score_var.set(f"{final.get('score', 0):.3f}")
            desc = cand.get("description", cand.get("expression", cand.get("strategy", "-")))
            self.final_desc_var.set(desc)
        else:
            self.final_engine_var.set("-")
            self.final_score_var.set("-")
            self.final_desc_var.set("No candidates.")

        self.candidates = result.get("ranked_candidates", [])
        self.candidate_list.delete(0, tk.END)
        for idx, entry in enumerate(self.candidates):
            cand = entry["candidate"]
            score = entry["score"]
            engine = cand.get("engine", "UnknownEngine")
            category = cand.get("category", "unknown")
            label = f"{idx+1}. {engine} [{category}] score={score:.3f}"
            self.candidate_list.insert(tk.END, label)

        self.details_text.delete("1.0", tk.END)

    def on_candidate_select(self, event):
        selection = self.candidate_list.curselection()
        if not selection:
            return
        idx = selection[0]
        entry = self.candidates[idx]
        cand = entry["candidate"]
        comps = entry["components"]

        lines = []
        for k, v in cand.items():
            lines.append(f"{k}: {v}")
        lines.append("\nFusion scoring:")
        for k, v in comps.items():
            lines.append(f"{k}: {v}")

        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, "\n".join(lines))


# ================================
# Main Entry
# ================================

def main():
    memory = MemoryManager(local_path="mpro_memory.json", smb_path=None)
    meta_state_manager = MetaStateManager(memory)
    fusion_layer = FusionLayer(memory_manager=memory, meta_state_getter=meta_state_manager.get_state)
    mpro = ManyPathReasoningOrganism(memory_manager=memory,
                                     fusion_layer=fusion_layer,
                                     meta_state_manager=meta_state_manager)

    # Register engines
    mpro.register_engine(MathTargetOneEngine())
    mpro.register_engine(ObstacleManholeEngine())
    mpro.register_engine(CheatHackEngine())
    mpro.register_engine(GameTacticalEngine())
    mpro.register_engine(GuardianRiskEngine())
    mpro.register_engine(NetworkAdaptEngine())
    mpro.register_engine(ScenarioDecisionEngine())
    mpro.register_engine(B4BOutcomePredictionEngine())
    mpro.register_engine(SurvivalProbabilityEngine())
    mpro.register_engine(MazeEquivalenceEngine(memory))

    adapters = {
        "back4blood": Back4BloodAdapter()
    }

    builder = WorldProblemBuilder(adapters)
    controller = AutonomousController(mpro, builder)

    root = tk.Tk()
    app = MPROGUI(root, mpro, memory, controller)
    root.mainloop()


if __name__ == "__main__":
    main()



            

