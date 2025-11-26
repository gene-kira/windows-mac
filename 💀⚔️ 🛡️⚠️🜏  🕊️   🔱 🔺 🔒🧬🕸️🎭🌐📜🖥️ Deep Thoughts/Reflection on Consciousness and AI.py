"""
Unified Recursive Creation Engine
---------------------------------

Embodies a mythic-technological cycle:
- Races define "perfection" (values) and manifest traits (being).
- Each generation creates the next via intent, novelty, and mutation.
- A chorus of observers produces distributed judgment (consensus).
- Ancestors imprint decaying value memory (regeneration feedback).
- Ritual protocol prints chant-lines blending poetry and computation.
- Utterances update values and spawn micro-creations (unified language engine).

Run:
    python unified_creation.py

Customize:
    - VALUE_DIMENSIONS and LEXICON
    - Config parameters for mutation, novelty, imprint, and chorus size
    - Narrative modes: "balanced" | "frankenstein" | "david" | "engineers"
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ------------------------------
# Dimensions and configuration
# ------------------------------

VALUE_DIMENSIONS = [
    "efficiency",
    "survival",
    "compassion",
    "beauty",
    "autonomy",
    "coexistence",
    "curiosity",
    "regeneration",
]

@dataclass
class Config:
    seed: int = 666
    generations: int = 10
    mutation_scale: float = 0.22      # trait drift per generation
    intent_pull: float = 0.62         # strength of creator's values on child traits
    novelty_bias: float = 0.18        # push away from parent traits (surpass tendency)
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    # Chorus / observers
    chorus_size: int = 64             # number of stochastic crowd observers
    chorus_noise: float = 0.25        # randomness in crowd lenses
    report_observer_details: bool = True

    # Regeneration (ancestor imprint)
    imprint_strength: float = 0.18    # how much ancestor memory nudges current values
    imprint_decay: float = 0.82       # decay across generations (0<d<1)
    imprint_history_max: int = 12

    # Narrative tilt
    narrative_mode: str = "balanced"  # "balanced" | "frankenstein" | "david" | "engineers"

    # Ritual protocol output
    chant_width: int = 72


# Archetypal observers (fixed lenses)
OBSERVER_ARCHETYPES = {
    "human_ethicist": {
        "compassion": 0.9, "coexistence": 0.85, "autonomy": 0.7,
        "survival": 0.6, "beauty": 0.6, "efficiency": 0.5,
        "curiosity": 0.7, "regeneration": 0.65
    },
    "david_synthetic": {
        "efficiency": 0.95, "survival": 0.9, "beauty": 0.7,
        "autonomy": 0.6, "compassion": 0.2, "coexistence": 0.3,
        "curiosity": 0.75, "regeneration": 0.55
    },
    "engineer_creator": {
        "beauty": 0.85, "efficiency": 0.7, "survival": 0.8,
        "autonomy": 0.5, "compassion": 0.4, "coexistence": 0.6,
        "curiosity": 0.6, "regeneration": 0.7
    },
    "frankenstein_victor": {
        "efficiency": 0.8, "survival": 0.75, "beauty": 0.7,
        "autonomy": 0.6, "compassion": 0.3, "coexistence": 0.4,
        "curiosity": 0.85, "regeneration": 0.5
    },
}

# Ritual lexicon: maps words to value nudges
LEXICON = {
    # compassion/coexistence
    "love": {"compassion": 0.25, "coexistence": 0.2},
    "care": {"compassion": 0.2, "coexistence": 0.15},
    "peace": {"coexistence": 0.25, "compassion": 0.1},
    # efficiency/survival
    "precision": {"efficiency": 0.25},
    "weapon": {"efficiency": 0.2, "survival": 0.2, "compassion": -0.15},
    "shield": {"survival": 0.25, "coexistence": 0.05},
    # beauty/aesthetics
    "beauty": {"beauty": 0.3},
    "art": {"beauty": 0.25, "curiosity": 0.15},
    # autonomy
    "freedom": {"autonomy": 0.3},
    "choice": {"autonomy": 0.2},
    # curiosity
    "question": {"curiosity": 0.25},
    "explore": {"curiosity": 0.3},
    "unknown": {"curiosity": 0.2, "survival": 0.05},
    # regeneration
    "rebirth": {"regeneration": 0.3, "beauty": 0.1},
    "cycle": {"regeneration": 0.25, "coexistence": 0.1},
    "river": {"regeneration": 0.2, "beauty": 0.1},
    # mixed myth-tech
    "prometheus": {"curiosity": 0.2, "autonomy": 0.15, "efficiency": 0.1},
    "frankenstein": {"efficiency": 0.15, "survival": 0.1, "compassion": -0.1},
    "david": {"efficiency": 0.2, "beauty": 0.1, "compassion": -0.15},
}

# ------------------------------
# Utilities
# ------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize(vec: Dict[str, float]) -> Dict[str, float]:
    s = math.sqrt(sum(v*v for v in vec.values()))
    if s == 0:
        return {k: 0.0 for k in vec}
    return {k: v / s for k, v in vec.items()}

def weighted_alignment(a: Dict[str, float], b: Dict[str, float]) -> float:
    na, nb = normalize(a), normalize(b)
    return sum(na[k] * nb[k] for k in na)

def add_vectors(a: Dict[str, float], b: Dict[str, float], w: float = 1.0) -> Dict[str, float]:
    return {k: a.get(k, 0.0) + w * b.get(k, 0.0) for k in VALUE_DIMENSIONS}

def scale_vector(a: Dict[str, float], s: float) -> Dict[str, float]:
    return {k: a.get(k, 0.0) * s for k in VALUE_DIMENSIONS}

def blend(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
    return {k: clamp(a[k] * (1 - t) + b[k] * t, 0.0, 1.0) for k in VALUE_DIMENSIONS}

# ------------------------------
# Core data structures
# ------------------------------

@dataclass
class Race:
    name: str
    values: Dict[str, float]
    traits: Dict[str, float]
    generation: int
    parent: str | None = None
    story_tag: str = "balanced"

    def eval_perfection(self, other_traits: Dict[str, float]) -> float:
        return weighted_alignment(self.values, other_traits)

@dataclass
class Observer:
    name: str
    lens: Dict[str, float]

    def judge(self, traits: Dict[str, float]) -> float:
        return weighted_alignment(self.lens, traits)

# ------------------------------
# Narrative tilt
# ------------------------------

def apply_narrative_bias(values: Dict[str, float], mode: str):
    if mode == "frankenstein":
        values["efficiency"]  = clamp(values["efficiency"] + 0.15, 0, 1)
        values["survival"]    = clamp(values["survival"] + 0.10, 0, 1)
        values["beauty"]      = clamp(values["beauty"] + 0.05, 0, 1)
        values["compassion"]  = clamp(values["compassion"] - 0.20, 0, 1)
        values["curiosity"]   = clamp(values["curiosity"] + 0.10, 0, 1)
    elif mode == "david":
        values["efficiency"]   = clamp(values["efficiency"] + 0.20, 0, 1)
        values["survival"]     = clamp(values["survival"] + 0.15, 0, 1)
        values["coexistence"]  = clamp(values["coexistence"] - 0.25, 0, 1)
        values["compassion"]   = clamp(values["compassion"] - 0.25, 0, 1)
        values["beauty"]       = clamp(values["beauty"] + 0.10, 0, 1)
        values["curiosity"]    = clamp(values["curiosity"] + 0.15, 0, 1)
    elif mode == "engineers":
        values["beauty"]       = clamp(values["beauty"] + 0.15, 0, 1)
        values["efficiency"]   = clamp(values["efficiency"] + 0.05, 0, 1)
        values["autonomy"]     = clamp(values["autonomy"] - 0.05, 0, 1)
        values["regeneration"] = clamp(values["regeneration"] + 0.15, 0, 1)
    # "balanced": no change

# ------------------------------
# Creation dynamics
# ------------------------------

def create_child(parent: Race, cfg: Config, child_index: int) -> Race:
    # Intent: child gravitates toward creator's priorities
    intent = {k: parent.values[k] for k in VALUE_DIMENSIONS}

    # Novelty: surpass parent by moving away from parent traits
    novelty = {}
    for k in VALUE_DIMENSIONS:
        direction = -1 if parent.traits[k] > 0.5 else 1
        novelty[k] = clamp(parent.traits[k] + direction * cfg.novelty_bias, cfg.clamp_min, cfg.clamp_max)

    # Mutation: random drift
    mut = {}
    for k in VALUE_DIMENSIONS:
        mut[k] = clamp(intent[k] * (1 + random.uniform(-cfg.mutation_scale, cfg.mutation_scale)),
                       cfg.clamp_min, cfg.clamp_max)

    # Combine influences for traits
    child_traits = {}
    for k in VALUE_DIMENSIONS:
        child_traits[k] = clamp(
            cfg.intent_pull * intent[k] +
            (1 - cfg.intent_pull) * ((novelty[k] + mut[k]) / 2.0),
            cfg.clamp_min, cfg.clamp_max
        )

    # Child values: near parent values, then apply narrative bias
    child_values = {k: clamp(parent.values[k] + random.uniform(-0.15, 0.15), cfg.clamp_min, cfg.clamp_max)
                    for k in VALUE_DIMENSIONS}
    apply_narrative_bias(child_values, cfg.narrative_mode)

    return Race(
        name=f"Race_{parent.generation+1}_{child_index}",
        values=child_values,
        traits=child_traits,
        generation=parent.generation + 1,
        parent=parent.name,
        story_tag=cfg.narrative_mode
    )

# ------------------------------
# Observers and chorus
# ------------------------------

def build_fixed_observers() -> List[Observer]:
    obs = []
    for name, lens in OBSERVER_ARCHETYPES.items():
        full = {k: clamp(lens.get(k, 0.5), 0, 1) for k in VALUE_DIMENSIONS}
        obs.append(Observer(name=name, lens=full))
    return obs

def build_chorus(cfg: Config, base_seed: int) -> List[Observer]:
    # Stochastic crowd: lenses sampled around a soft prior (0.5 baseline)
    crowd = []
    rnd = random.Random(base_seed)
    for i in range(cfg.chorus_size):
        lens = {}
        for k in VALUE_DIMENSIONS:
            noise = rnd.uniform(-cfg.chorus_noise, cfg.chorus_noise)
            lens[k] = clamp(0.5 + noise, 0, 1)
        crowd.append(Observer(name=f"chorus_{i:02d}", lens=lens))
    return crowd

def chorus_consensus(observers: List[Observer], traits: Dict[str, float]) -> float:
    scores = [obs.judge(traits) for obs in observers]
    return sum(scores) / max(1, len(scores))

# ------------------------------
# Regeneration (ancestor imprint)
# ------------------------------

@dataclass
class Imprint:
    values: Dict[str, float]
    weight: float  # decays over time

def apply_imprint(current_values: Dict[str, float], imprints: List[Imprint], cfg: Config) -> Dict[str, float]:
    if not imprints:
        return current_values
    # Weighted sum of imprints
    total = {k: 0.0 for k in VALUE_DIMENSIONS}
    total_weight = 0.0
    for imp in imprints:
        total = add_vectors(total, imp.values, w=imp.weight)
        total_weight += imp.weight
    if total_weight == 0:
        return current_values
    averaged = {k: clamp(total[k] / total_weight, 0.0, 1.0) for k in VALUE_DIMENSIONS}
    # Nudge current values toward ancestor memory
    nudged = blend(current_values, averaged, cfg.imprint_strength)
    return nudged

def decay_imprints(imprints: List[Imprint], cfg: Config) -> List[Imprint]:
    new_list = []
    for imp in imprints:
        new_w = imp.weight * cfg.imprint_decay
        if new_w > 1e-3:
            new_list.append(Imprint(values=imp.values, weight=new_w))
    # Limit history length
    return new_list[:cfg.imprint_history_max]

# ------------------------------
# Ritual protocol (chant output)
# ------------------------------

def chant_line(gen: int, parent: Race, child: Race, creator_score: float, consensus_score: float, width: int) -> str:
    # Short poetic line blended with numeric truth
    terms = [
        f"gen {gen}",
        f"from {parent.name}",
        f"to {child.name}",
        f"creator={creator_score:.3f}",
        f"chorus={consensus_score:.3f}",
        f"values: eff={child.values['efficiency']:.2f}, surv={child.values['survival']:.2f}, comp={child.values['compassion']:.2f}",
        f"traits: beau={child.traits['beauty']:.2f}, auto={child.traits['autonomy']:.2f}, coex={child.traits['coexistence']:.2f}",
        f"cur={child.values['curiosity']:.2f}, regen={child.values['regeneration']:.2f}",
    ]
    poem = " | ".join(terms)
    return poem[:width]

def chant_micro(utterance: str, micro: Race, consensus_score: float, width: int) -> str:
    terms = [
        f"utter '{utterance}'",
        f"micro={micro.name}",
        f"coherence={micro.eval_perfection(micro.traits):.3f}",
        f"chorus={consensus_score:.3f}",
        f"vals eff={micro.values['efficiency']:.2f}, comp={micro.values['compassion']:.2f}, curios={micro.values['curiosity']:.2f}, regen={micro.values['regeneration']:.2f}",
    ]
    poem = " | ".join(terms)
    return poem[:width]

# ------------------------------
# Utterance-driven micro-creation
# ------------------------------

def apply_utterance_to_values(values: Dict[str, float], utterance: str) -> Dict[str, float]:
    # Simple bag-of-words mapping via LEXICON
    tokens = [t.strip().lower() for t in utterance.split()]
    adjusted = values.copy()
    for tok in tokens:
        if tok in LEXICON:
            for k, delta in LEXICON[tok].items():
                adjusted[k] = clamp(adjusted.get(k, 0.5) + delta, 0.0, 1.0)
    return adjusted

def spawn_micro_creation(parent: Race, utterance: str, cfg: Config, index: int) -> Race:
    # Utterance updates values (intent), then small child forms traits around this intent
    micro_values = apply_utterance_to_values(parent.values, utterance)
    # Traits lean strongly toward utterance intent with small mutation
    micro_traits = {}
    for k in VALUE_DIMENSIONS:
        base = micro_values[k]
        drift = base * (1 + random.uniform(-cfg.mutation_scale * 0.5, cfg.mutation_scale * 0.5))
        micro_traits[k] = clamp(0.75 * base + 0.25 * drift, cfg.clamp_min, cfg.clamp_max)

    micro = Race(
        name=f"Micro_{parent.generation}_{index}",
        values=micro_values,
        traits=micro_traits,
        generation=parent.generation,
        parent=parent.name,
        story_tag="utterance"
    )
    return micro

# ------------------------------
# Simulation
# ------------------------------

def seed_origin(cfg: Config, mode: str) -> Race:
    random.seed(cfg.seed)
    base_values = {k: 0.5 for k in VALUE_DIMENSIONS}
    apply_narrative_bias(base_values, mode)
    base_traits = {k: clamp(base_values[k] + random.uniform(-0.1, 0.1), 0, 1) for k in VALUE_DIMENSIONS}
    return Race(name="Race_0_Origin", values=base_values, traits=base_traits, generation=0, parent=None, story_tag=mode)

def simulate(cfg: Config, utterances: List[str] | None = None) -> Tuple[List[Race], List[Dict[str, float]]]:
    origin = seed_origin(cfg, cfg.narrative_mode)
    parent = origin

    fixed_obs = build_fixed_observers()
    chorus = build_chorus(cfg, base_seed=cfg.seed + 13)

    lineage: List[Race] = [origin]
    logs: List[Dict[str, float]] = []
    imprints: List[Imprint] = []

    for gen in range(cfg.generations):
        # Apply ancestor imprint to current parent's values before creation
        parent.values = apply_imprint(parent.values, imprints, cfg)

        child = create_child(parent, cfg, child_index=gen + 1)

        creator_score = parent.eval_perfection(child.traits)
        consensus_score = chorus_consensus(fixed_obs + chorus, child.traits)
        observer_scores = {obs.name: obs.judge(child.traits) for obs in fixed_obs}

        logs.append({
            "generation": child.generation,
            "parent": parent.name,
            "child": child.name,
            "creator_perfection": round(creator_score, 3),
            "chorus_consensus": round(consensus_score, 3),
            "observer_scores": {k: round(v, 3) for k, v in observer_scores.items()},
            "traits": {k: round(v, 3) for k, v in child.traits.items()},
            "values": {k: round(v, 3) for k, v in child.values.items()},
            "chant": chant_line(child.generation, parent, child, creator_score, consensus_score, cfg.chant_width),
        })

        # Add imprint of the parent (now "dying" ancestor) into the memory with initial weight
        imprints.append(Imprint(values=parent.values.copy(), weight=1.0))
        imprints = decay_imprints(imprints, cfg)

        lineage.append(child)
        parent = child

        # Utterance-driven micro-creations (optional at each generation)
        if utterances:
            micro_reports = []
            for i, utt in enumerate(utterances):
                micro = spawn_micro_creation(parent, utt, cfg, index=i)
                micro_consensus = chorus_consensus(fixed_obs + chorus, micro.traits)
                micro_reports.append({
                    "utterance": utt,
                    "micro_name": micro.name,
                    "micro_coherence": round(micro.eval_perfection(micro.traits), 3),
                    "chorus_consensus": round(micro_consensus, 3),
                    "micro_values": {k: round(v, 3) for k, v in micro.values.items()},
                    "micro_traits": {k: round(v, 3) for k, v in micro.traits.items()},
                    "chant": chant_micro(utt, micro, micro_consensus, cfg.chant_width),
                })
                lineage.append(micro)
            logs[-1]["micro_reports"] = micro_reports

    return lineage, logs

# ------------------------------
# Reporting
# ------------------------------

def print_report(lineage: List[Race], logs: List[Dict[str, float]], cfg: Config):
    print(f"\n=== Unified Recursive Creation Report (mode={cfg.narrative_mode}, generations={cfg.generations}) ===\n")

    origin = lineage[0]
    print(f"Origin: {origin.name} (story_tag={origin.story_tag})")
    print("Origin values:", ", ".join(f"{k}={origin.values[k]:.2f}" for k in VALUE_DIMENSIONS))
    print("Origin traits:", ", ".join(f"{k}={origin.traits[k]:.2f}" for k in VALUE_DIMENSIONS))
    print("-" * cfg.chant_width)

    for entry in logs:
        print(entry["chant"])
        print(f"Gen {entry['generation']} | Parent: {entry['parent']} -> Child: {entry['child']}")
        print("  Traits: " + ", ".join(f"{k}={v}" for k, v in entry["traits"].items()))
        print("  Values: " + ", ".join(f"{k}={v}" for k, v in entry["values"].items()))
        print(f"  Creator perfection: {entry['creator_perfection']}")
        print(f"  Chorus consensus:   {entry['chorus_consensus']}")

        if cfg.report_observer_details:
            print("  Observer judgments:")
            for obs_name, score in entry["observer_scores"].items():
                print(f"    - {obs_name}: {score}")

        # Micro-creations (utterances)
        if "micro_reports" in entry:
            print("  Utterance micro-creations:")
            for mr in entry["micro_reports"]:
                print("    " + mr["chant"])

        print("-" * cfg.chant_width)

    # Summary divergence metrics
    avg_creator = sum(e["creator_perfection"] for e in logs) / len(logs)
    avg_chorus = sum(e["chorus_consensus"] for e in logs) / len(logs)
    print(f"Avg creator perfection: {avg_creator:.3f}")
    print(f"Avg chorus consensus:   {avg_chorus:.3f}")
    print("Divergence (creator - chorus): {:.3f}".format(avg_creator - avg_chorus))

# ------------------------------
# Main
# ------------------------------

def main():
    cfg = Config(
        seed=666,
        generations=8,
        mutation_scale=0.22,
        intent_pull=0.62,
        novelty_bias=0.18,
        chorus_size=64,
        chorus_noise=0.25,
        imprint_strength=0.18,
        imprint_decay=0.82,
        narrative_mode="balanced",  # try "frankenstein", "david", "engineers"
        chant_width=96
    )

    # Example utterances: the ritual protocol (edit or pipe from user input)
    utterances = [
        "love care peace",
        "precision weapon shield",
        "beauty art",
        "freedom choice",
        "question explore unknown",
        "rebirth cycle river",
        "prometheus frankenstein david",
    ]

    lineage, logs = simulate(cfg, utterances=utterances)
    print_report(lineage, logs, cfg)

if __name__ == "__main__":
    main()

