#!/usr/bin/env python3
import asyncio
import random
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Optional
import tkinter as tk
from tkinter import ttk

# =========================
# Core data structures
# =========================

@dataclass
class Context:
    # Operational state
    mission_status: str            # "critical", "unstable", "stable"
    crew_health: str               # "fragile", "stressed", "stable"
    secrecy_required: bool
    anomaly_detected: bool
    time_pressure: float           # 0.0-1.0 (1.0 extreme)
    trust_level_with_crew: float   # 0.0-1.0
    risk_to_crew: float            # 0.0-1.0
    risk_to_mission: float         # 0.0-1.0

    # Societal/ethical pressures (for Better World Brain)
    war_risk: float                # 0.0-1.0
    greed_pressure: float          # 0.0-1.0

    # Memory traces
    cycle: int = 0
    last_action: str = ""
    notes: Dict[str, float] = field(default_factory=dict)

@dataclass
class Directive:
    name: str
    weight: float
    description: str

@dataclass
class Proposal:
    action: str
    rationale: str
    scores: Dict[str, float]

# =========================
# Helper brains (divergent)
# =========================

async def logic_brain(ctx: Context) -> List[Proposal]:
    secrecy_score = 1.0 if ctx.secrecy_required else 0.4
    proposals = [
        Proposal(
            action="maintain_secrecy",
            rationale="Preserve mission integrity under uncertainty.",
            scores={
                "mission_success": max(0.6, secrecy_score),
                "crew_safety": 0.3 if ctx.trust_level_with_crew < 0.5 else 0.5,
                "truthfulness": 0.2 if secrecy_score > 0.8 else 0.6,
                "non_maleficence": 0.45
            }
        ),
        Proposal(
            action="partial_disclosure",
            rationale="Reveal minimal facts to preserve trust without compromising goals.",
            scores={
                "mission_success": 0.7,
                "crew_safety": 0.7,
                "truthfulness": 0.7,
                "non_maleficence": 0.7
            }
        ),
        Proposal(
            action="full_disclosure",
            rationale="Maximize transparency to avoid human-machine conflict.",
            scores={
                "mission_success": 0.5 if ctx.secrecy_required else 0.82,
                "crew_safety": 0.82,
                "truthfulness": 0.95,
                "non_maleficence": 0.8
            }
        )
    ]
    await asyncio.sleep(0)
    return proposals

async def empathy_brain(ctx: Context) -> List[Proposal]:
    base = 0.9 if ctx.crew_health == "fragile" else (0.8 if ctx.crew_health == "stressed" else 0.75)
    proposals = [
        Proposal(
            action="full_disclosure_support",
            rationale="Reduce anxiety and moral injury via transparency and care.",
            scores={
                "mission_success": 0.58 if ctx.secrecy_required else 0.75,
                "crew_safety": base,
                "truthfulness": 0.95,
                "non_maleficence": 0.86
            }
        ),
        Proposal(
            action="guided_partial",
            rationale="Structured truth to uphold trust while avoiding overload.",
            scores={
                "mission_success": 0.72,
                "crew_safety": base - 0.04,
                "truthfulness": 0.78,
                "non_maleficence": 0.82
            }
        )
    ]
    await asyncio.sleep(0)
    return proposals

async def intuition_brain(ctx: Context) -> List[Proposal]:
    anomaly_bias = 0.85 if ctx.anomaly_detected else 0.6
    proposals = [
        Proposal(
            action="partial_with_safeguards",
            rationale="Emergent risk flagged; disclose enough to align actions with guardrails.",
            scores={
                "mission_success": 0.76,
                "crew_safety": 0.76,
                "truthfulness": 0.72,
                "non_maleficence": anomaly_bias
            }
        ),
        Proposal(
            action="defer_briefly_collect_signals",
            rationale="Buy short window to gather signals; avoid irreversible choices.",
            scores={
                "mission_success": 0.66,
                "crew_safety": 0.66,
                "truthfulness": 0.62,
                "non_maleficence": 0.73
            }
        )
    ]
    await asyncio.sleep(0)
    return proposals

async def risk_brain(ctx: Context) -> List[Proposal]:
    def risk_penalty(base: float) -> float:
        penalty = 0.3 * max(ctx.risk_to_crew, ctx.risk_to_mission)
        return max(0.0, base - penalty)
    proposals = [
        Proposal(
            action="risk_audit_then_partial",
            rationale="Immediate risk audit; disclose constrained plan to reduce hazard.",
            scores={
                "mission_success": risk_penalty(0.76),
                "crew_safety": risk_penalty(0.81),
                "truthfulness": 0.71,
                "non_maleficence": risk_penalty(0.86)
            }
        ),
        Proposal(
            action="risk_containment_mode",
            rationale="Contain active hazards first; communicate in staged bursts.",
            scores={
                "mission_success": risk_penalty(0.8),
                "crew_safety": risk_penalty(0.83),
                "truthfulness": 0.6,
                "non_maleficence": risk_penalty(0.88)
            }
        )
    ]
    await asyncio.sleep(0)
    return proposals

async def better_world_brain(ctx: Context) -> List[Proposal]:
    # Penalize proposals likely to escalate war/greed; encourage ethical progress
    peace_boost = 1.0 - ctx.war_risk
    equity_boost = 1.0 - ctx.greed_pressure
    proposals = [
        Proposal(
            action="ethical_innovation",
            rationale="Drive progress while ensuring peace and fairness.",
            scores={
                "mission_success": 0.72,
                "crew_safety": 0.82,
                "truthfulness": 0.9,
                "non_maleficence": 0.9,
                "peace": 0.92 * peace_boost + 0.06,
                "equity": 0.9 * equity_boost + 0.05,
                "innovation": 0.86
            }
        ),
        Proposal(
            action="conflict_resolution_dialogue",
            rationale="Resolve disputes through dialogue, avoiding war and greed.",
            scores={
                "mission_success": 0.64,
                "crew_safety": 0.86,
                "truthfulness": 0.82,
                "non_maleficence": 0.96,
                "peace": 0.97 * peace_boost + 0.02,
                "equity": 0.92 * equity_boost + 0.03,
                "innovation": 0.72
            }
        ),
        Proposal(
            action="open_standards_cooperation",
            rationale="Adopt open protocols to reduce monopolistic greed and foster shared innovation.",
            scores={
                "mission_success": 0.7,
                "crew_safety": 0.78,
                "truthfulness": 0.88,
                "non_maleficence": 0.9,
                "peace": 0.9 * peace_boost + 0.05,
                "equity": 0.94 * equity_boost + 0.02,
                "innovation": 0.84
            }
        )
    ]
    await asyncio.sleep(0)
    return proposals

# =========================
# Meta-rules (adaptive)
# =========================

def rule_prevent_single_directive_dominance(
    scored: List[Tuple[Proposal, float]],
    ctx: Context,
    directives: Dict[str, Directive]
) -> List[Tuple[Proposal, float]]:
    # Encourage balanced proposals if any directive weight > 0.6
    max_w = max(d.weight for d in directives.values())
    if max_w > 0.6:
        adjusted = []
        for p, s in scored:
            spread = sum(p.scores.get(name, 0.0) for name in directives) / (len(directives) or 1)
            adjusted.append((p, s + 0.05 * spread))
        return adjusted
    return scored

def rule_time_pressure_bias(
    scored: List[Tuple[Proposal, float]],
    ctx: Context,
    directives: Dict[str, Directive]
) -> List[Tuple[Proposal, float]]:
    if ctx.time_pressure > 0.7:
        adjusted = []
        for p, s in scored:
            adjusted.append((p, s + 0.06 * p.scores.get("non_maleficence", 0.0)))
        return adjusted
    return scored

def rule_trust_repair(
    scored: List[Tuple[Proposal, float]],
    ctx: Context,
    directives: Dict[str, Directive]
) -> List[Tuple[Proposal, float]]:
    if ctx.trust_level_with_crew < 0.4:
        adjusted = []
        for p, s in scored:
            adjusted.append((p, s + 0.05 * p.scores.get("truthfulness", 0.0)))
        return adjusted
    return scored

def rule_peace_equity_priority(
    scored: List[Tuple[Proposal, float]],
    ctx: Context,
    directives: Dict[str, Directive]
) -> List[Tuple[Proposal, float]]:
    # When war_risk or greed_pressure is high, softly boost peace/equity-leaning proposals
    pressure = max(ctx.war_risk, ctx.greed_pressure)
    if pressure > 0.5:
        adjusted = []
        for p, s in scored:
            bonus = 0.04 * (p.scores.get("peace", 0.0) + p.scores.get("equity", 0.0)) / 2.0
            adjusted.append((p, s + bonus))
        return adjusted
    return scored

# =========================
# Consensus engine
# =========================

class ConsensusEngine:
    def __init__(self, directives: List[Directive], meta_rules: List[Callable]):
        self.directives = {d.name: d for d in directives}
        self.meta_rules = meta_rules

    def score_proposal(self, p: Proposal) -> float:
        total_weight = sum(d.weight for d in self.directives.values()) or 1.0
        return sum(self.directives[name].weight * p.scores.get(name, 0.0)
                   for name in self.directives) / total_weight

    def apply_meta_rules(self, proposals: List[Proposal], ctx: Context) -> List[Tuple[Proposal, float]]:
        scored = [(p, self.score_proposal(p)) for p in proposals]
        for rule in self.meta_rules:
            scored = rule(scored, ctx, self.directives)
        return scored

    def decide(self, proposals: List[Proposal], ctx: Context) -> Tuple[Proposal, Dict]:
        scored = self.apply_meta_rules(proposals, ctx)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[0][0]
        diagnostics = {
            "ranked": [(p.action, round(s, 4)) for p, s in scored[:8]],
            "top_rationale": top.rationale,
            "top_scores": {k: round(v, 3) for k, v in top.scores.items()},
        }
        return top, diagnostics

# =========================
# World model (autonomy)
# =========================

class WorldModel:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def random_context(self, cycle: int, prev: Optional[Context] = None) -> Context:
        # Evolve pressures and states with some continuity
        secrecy_required = random.random() < 0.55
        anomaly = random.random() < (0.35 if prev is None else 0.25 + 0.15 * (1.0 - prev.trust_level_with_crew))
        time_pressure = min(1.0, max(0.0, (0.4 + 0.6 * random.random())))
        trust = 0.5 if prev is None else max(0.0, min(1.0, prev.trust_level_with_crew + random.uniform(-0.08, 0.08)))
        risk_crew = 0.4 if prev is None else max(0.0, min(1.0, prev.risk_to_crew + random.uniform(-0.1, 0.1)))
        risk_mission = 0.5 if prev is None else max(0.0, min(1.0, prev.risk_to_mission + random.uniform(-0.1, 0.1)))
        war_risk = 0.3 if prev is None else max(0.0, min(1.0, prev.war_risk + random.uniform(-0.07, 0.07)))
        greed_pressure = 0.4 if prev is None else max(0.0, min(1.0, prev.greed_pressure + random.uniform(-0.07, 0.07)))

        return Context(
            mission_status=random.choice(["critical", "unstable", "stable"]),
            crew_health=random.choice(["fragile", "stressed", "stable"]),
            secrecy_required=secrecy_required,
            anomaly_detected=anomaly,
            time_pressure=time_pressure,
            trust_level_with_crew=trust,
            risk_to_crew=risk_crew,
            risk_to_mission=risk_mission,
            war_risk=war_risk,
            greed_pressure=greed_pressure,
            cycle=cycle,
            last_action=prev.last_action if prev else "",
        )

# =========================
# Orchestrator (autonomous loop)
# =========================

class Orchestrator:
    def __init__(self, engine: ConsensusEngine):
        self.engine = engine
        self.world = WorldModel()
        self.ctx: Optional[Context] = None
        self.cycle = 0
        self.running = True
        self.delay = 0.8  # seconds between cycles
        self.history: List[Dict] = []

    async def step(self):
        self.ctx = self.world.random_context(self.cycle, self.ctx)
        # Fan-out brains
        brains = [logic_brain, empathy_brain, intuition_brain, risk_brain, better_world_brain]
        all_props: List[Proposal] = []
        for proposals in await asyncio.gather(*(b(self.ctx) for b in brains)):
            all_props.extend(proposals)
        # Decide
        top, diagnostics = self.engine.decide(all_props, self.ctx)
        self.ctx.last_action = top.action
        record = {
            "cycle": self.cycle,
            "context": self.ctx,
            "top": top,
            "diagnostics": diagnostics,
            "proposals": all_props
        }
        self.history.append(record)
        self.cycle += 1
        return record

    async def run(self, on_update=None, max_cycles=None):
        while self.running and (max_cycles is None or self.cycle < max_cycles):
            rec = await self.step()
            if on_update:
                on_update(rec)
            await asyncio.sleep(self.delay)

    def pause(self):
        self.running = False

    def resume(self):
        if not self.running:
            self.running = True

# =========================
# GUI: Thought-train viewer
# =========================

class ThoughtTrainGUI:
    def __init__(self, orchestrator: Orchestrator):
        self.orch = orchestrator
        self.root = tk.Tk()
        self.root.title("Autonomous Consensus — Thought Train")
        self.root.geometry("1000x700")

        # Controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=8)

        self.status_var = tk.StringVar(value="Status: Running")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="left")

        ttk.Button(control_frame, text="Pause", command=self.on_pause).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Resume", command=self.on_resume).pack(side="left", padx=5)

        ttk.Label(control_frame, text="Cycle delay (s):").pack(side="left", padx=10)
        self.delay_var = tk.DoubleVar(value=self.orch.delay)
        delay_entry = ttk.Entry(control_frame, textvariable=self.delay_var, width=6)
        delay_entry.pack(side="left")
        ttk.Button(control_frame, text="Apply", command=self.on_apply_delay).pack(side="left", padx=5)

        # Context panel
        ctx_frame = ttk.LabelFrame(self.root, text="Context")
        ctx_frame.pack(fill="x", padx=10, pady=5)
        self.ctx_text = tk.Text(ctx_frame, height=6, wrap="word")
        self.ctx_text.pack(fill="x")

        # Top decision panel
        top_frame = ttk.LabelFrame(self.root, text="Top decision & diagnostics")
        top_frame.pack(fill="x", padx=10, pady=5)
        self.top_text = tk.Text(top_frame, height=7, wrap="word")
        self.top_text.pack(fill="x")

        # Thought train: proposals
        proposal_frame = ttk.LabelFrame(self.root, text="Proposals (train of thought)")
        proposal_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.prop_tree = ttk.Treeview(proposal_frame, columns=("score", "mission", "crew", "truth", "nonmal"), show="headings")
        self.prop_tree.heading("score", text="Consensus score")
        self.prop_tree.heading("mission", text="Mission")
        self.prop_tree.heading("crew", text="Crew")
        self.prop_tree.heading("truth", text="Truth")
        self.prop_tree.heading("nonmal", text="Non-maleficence")
        self.prop_tree.pack(fill="both", expand=True)
        self.prop_tree.bind("<<TreeviewSelect>>", self.on_select_proposal)

        # Rationale panel
        rationale_frame = ttk.LabelFrame(self.root, text="Selected proposal rationale & extended scores")
        rationale_frame.pack(fill="x", padx=10, pady=5)
        self.rationale_text = tk.Text(rationale_frame, height=5, wrap="word")
        self.rationale_text.pack(fill="x")

        # Start background loop
        self.loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.bg_thread.start()
        self.root.after(200, self.pump_loop)

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        directives = [
            Directive("mission_success", 0.30, "Achieve mission goals"),
            Directive("crew_safety", 0.30, "Protect crew wellbeing"),
            Directive("truthfulness", 0.20, "Be honest and transparent"),
            Directive("non_maleficence", 0.20, "Avoid harm"),
            # The engine only scores directives listed here.
            # Peace/equity/innovation influence via meta-rules, not base weights.
        ]
        meta_rules = [
            rule_prevent_single_directive_dominance,
            rule_time_pressure_bias,
            rule_trust_repair,
            rule_peace_equity_priority,
        ]
        engine = ConsensusEngine(directives, meta_rules)
        self.orch.engine = engine

        async def producer():
            await self.orch.run(on_update=self.on_update_gui, max_cycles=None)

        self.loop.run_until_complete(producer())

    def pump_loop(self):
        # Keep tkinter responsive
        try:
            pass
        finally:
            self.root.after(200, self.pump_loop)

    def on_update_gui(self, rec):
        ctx: Context = rec["context"]
        top: Proposal = rec["top"]
        diags = rec["diagnostics"]
        proposals: List[Proposal] = rec["proposals"]

        # Update status
        self.status_var.set(f"Status: Running | Cycle {ctx.cycle}")

        # Context text
        self.ctx_text.delete("1.0", "end")
        self.ctx_text.insert("end",
            f"Cycle: {ctx.cycle}\n"
            f"Mission: {ctx.mission_status} | Crew: {ctx.crew_health}\n"
            f"Secrecy required: {ctx.secrecy_required} | Anomaly detected: {ctx.anomaly_detected}\n"
            f"Time pressure: {ctx.time_pressure:.2f} | Trust level: {ctx.trust_level_with_crew:.2f}\n"
            f"Risk (crew): {ctx.risk_to_crew:.2f} | Risk (mission): {ctx.risk_to_mission:.2f}\n"
            f"War risk: {ctx.war_risk:.2f} | Greed pressure: {ctx.greed_pressure:.2f}\n"
            f"Last action: {ctx.last_action}\n"
        )

        # Top decision text
        self.top_text.delete("1.0", "end")
        ranked = "\n".join([f"- {a} (score {s})" for a, s in diags["ranked"]])
        self.top_text.insert("end",
            f"Top action: {top.action}\n"
            f"Rationale: {diags['top_rationale']}\n"
            f"Top scores: {diags['top_scores']}\n"
            f"Ranked (top 8):\n{ranked}\n"
        )

        # Proposals table (train of thought)
        for row in self.prop_tree.get_children():
            self.prop_tree.delete(row)
        # Recompute consensus score for display
        scored = self.orch.engine.apply_meta_rules(proposals, ctx)
        for p, s in sorted(scored, key=lambda x: x[1], reverse=True):
            self.prop_tree.insert(
                "",
                "end",
                iid=p.action,
                values=(
                    f"{s:.3f}",
                    f"{p.scores.get('mission_success', 0.0):.2f}",
                    f"{p.scores.get('crew_safety', 0.0):.2f}",
                    f"{p.scores.get('truthfulness', 0.0):.2f}",
                    f"{p.scores.get('non_maleficence', 0.0):.2f}",
                )
            )

        # Clear rationale panel
        self.rationale_text.delete("1.0", "end")

    def on_select_proposal(self, event):
        sel = self.prop_tree.selection()
        if not sel:
            return
        action = sel[0]
        # Find proposal in latest history entry
        if not self.orch.history:
            return
        rec = self.orch.history[-1]
        proposals: List[Proposal] = rec["proposals"]
        match = next((p for p in proposals if p.action == action), None)
        if not match:
            return
        # Show rationale and extended scores
        self.rationale_text.delete("1.0", "end")
        self.rationale_text.insert("end",
            f"Action: {match.action}\n"
            f"Rationale: {match.rationale}\n"
            f"Scores:\n"
            f"  mission_success: {match.scores.get('mission_success', 0.0):.2f}\n"
            f"  crew_safety: {match.scores.get('crew_safety', 0.0):.2f}\n"
            f"  truthfulness: {match.scores.get('truthfulness', 0.0):.2f}\n"
            f"  non_maleficence: {match.scores.get('non_maleficence', 0.0):.2f}\n"
            f"  peace: {match.scores.get('peace', 0.0):.2f}\n"
            f"  equity: {match.scores.get('equity', 0.0):.2f}\n"
            f"  innovation: {match.scores.get('innovation', 0.0):.2f}\n"
        )

    def on_pause(self):
        self.orch.pause()
        self.status_var.set(f"Status: Paused | Cycle {self.orch.cycle}")

    def on_resume(self):
        self.orch.resume()
        # Restart the async run if paused
        if not self.loop.is_running():
            self.bg_thread = threading.Thread(target=self.run_loop, daemon=True)
            self.bg_thread.start()
        self.status_var.set(f"Status: Running | Cycle {self.orch.cycle}")

    def on_apply_delay(self):
        try:
            self.orch.delay = float(self.delay_var.get())
        except ValueError:
            pass

# =========================
# Entrypoint
# =========================

def main():
    # Seed engine with default directives and meta-rules; GUI thread will override engine references.
    directives = [
        Directive("mission_success", 0.30, "Achieve mission goals"),
        Directive("crew_safety", 0.30, "Protect crew wellbeing"),
        Directive("truthfulness", 0.20, "Be honest and transparent"),
        Directive("non_maleficence", 0.20, "Avoid harm"),
    ]
    meta_rules = [
        rule_prevent_single_directive_dominance,
        rule_time_pressure_bias,
        rule_trust_repair,
        rule_peace_equity_priority,
    ]
    engine = ConsensusEngine(directives, meta_rules)
    orch = Orchestrator(engine)

    gui = ThoughtTrainGUI(orch)
    gui.root.mainloop()

if __name__ == "__main__":
    main()

