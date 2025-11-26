#!/usr/bin/env python3
"""
Hive Mind Engine
A communal, persistent, wandering-mind program that models a human hive:
- Multi-user contributions
- Persistent shared memory (JSON)
- Adaptive concept graph with edge weights
- Lightweight consensus scoring
- Stream-of-consciousness generation with recall and future dreaming

Run:
    python hive_mind_engine.py --user "YourName"
"""

import json
import os
import random
import time
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Config knobs for the "hive"
# -----------------------------
SEED = 42                   # change for different runs
THOUGHTS_PER_SESSION = 18   # emitted collective thoughts per session
RECALL_RATE = 0.22          # chance to recall a past thought
ASSOCIATION_RATE = 0.62     # chance to follow a concept association
METAPHOR_RATE = 0.72        # chance to add metaphor to a line
FORESIGHT_RATE = 0.28       # chance a thought leaps into a future invention
RHYTHM_PAUSE = (0.02, 0.08) # min/max pause between lines (seconds)

STATE_FILE = "hive_state.json"

random.seed(SEED)

# -----------------------------
# Seed timeline of communication
# -----------------------------
SEED_TIMELINE = [
    ("grunts & gesture", -200000),
    ("fire & signal smoke", -40000),
    ("cave art & symbols", -30000),
    ("trail markers & drums", -5000),
    ("papyrus & writing systems", -3000),
    ("seafaring navigation by stars", -2000),
    ("printing press", 1450),
    ("postal routes", 1600),
    ("telegraph", 1837),
    ("telephone", 1876),
    ("radio broadcast", 1906),
    ("satellite comms", 1962),
    ("internet", 1983),
    ("mobile networks", 1991),
    ("fiber optics at scale", 2000),
    ("cloud platforms", 2006),
    ("edge computing", 2015),
    ("AI assistants", 2020),
]

# -----------------------------
# Metaphors and future templates
# -----------------------------
METAPHORS = [
    "echoes braided through time",
    "sparks stitched into the dark",
    "voices carried by light",
    "maps drawn on wind and memory",
    "threads crossing oceans",
    "orbits that remember the shore",
    "glyphs waking in glass",
    "rituals of distance made near",
    "the earth teaching the sky to listen",
    "light-lines humming with intent",
]

FUTURES = [
    ("planetary mesh relays", 2035, "autonomous, self-healing networks spanning sea, sky, and soil"),
    ("photonic local language", 2038, "ambient light-encoded speech decoded by everyday surfaces"),
    ("consent-first personal swarms", 2040, "micro-relays negotiating privacy before carrying a single bit"),
    ("memory sails", 2043, "thin-film arrays that drift and store communal histories"),
    ("sovereign edge commons", 2045, "neighborhood-owned compute, bandwidth, and caches"),
    ("holo-ritual consoles", 2047, "expressive overlays for governing shared infrastructure with stories"),
    ("quantum handshake layers", 2050, "trust primitives that verify intent, not identity"),
]

# -----------------------------
# Conceptual model dataclasses
# -----------------------------
@dataclass
class Thought:
    text: str
    concept: Optional[str] = None
    year: Optional[int] = None
    author: Optional[str] = None
    tag: str = "present"  # present | recall | future | association | seed | user
    score: float = 0.0    # consensus signal (upvotes, reuse)

@dataclass
class ConceptGraph:
    # nodes: concept -> year (optional)
    nodes: Dict[str, Optional[int]] = field(default_factory=dict)
    # edges: concept -> {neighbor: weight}
    edges: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def ensure_node(self, concept: str, year: Optional[int] = None):
        if concept not in self.nodes:
            self.nodes[concept] = year
        elif self.nodes[concept] is None and year is not None:
            self.nodes[concept] = year

    def connect(self, a: str, b: str, weight: float = 1.0):
        if a == b:
            return
        self.edges.setdefault(a, {})
        self.edges.setdefault(b, {})
        self.edges[a][b] = self.edges[a].get(b, 0.0) + weight
        self.edges[b][a] = self.edges[b].get(a, 0.0) + weight

    def neighbors(self, concept: str) -> List[Tuple[str, float]]:
        return sorted([(n, w) for n, w in self.edges.get(concept, {}).items()],
                      key=lambda x: x[1], reverse=True)

    def prefer_unseen(self, concept: str, seen: set) -> Optional[str]:
        nbrs = self.neighbors(concept)
        if not nbrs:
            return None
        unseen = [(n, w) for n, w in nbrs if n not in seen]
        pool = unseen if unseen else nbrs
        # weighted choice
        total = sum(w for _, w in pool)
        if total <= 0:
            return random.choice([n for n, _ in pool])
        r = random.random() * total
        acc = 0.0
        for n, w in pool:
            acc += w
            if acc >= r:
                return n
        return pool[-1][0]

@dataclass
class Hive:
    graph: ConceptGraph = field(default_factory=ConceptGraph)
    memory: List[Thought] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # reconstruct graph
        self.graph.nodes = data.get("graph", {}).get("nodes", {})
        self.graph.edges = data.get("graph", {}).get("edges", {})
        # reconstruct thoughts
        self.memory = [Thought(**t) for t in data.get("memory", [])]
        self.contributors = data.get("contributors", [])

    def save(self, path: str):
        data = {
            "graph": {"nodes": self.graph.nodes, "edges": self.graph.edges},
            "memory": [asdict(t) for t in self.memory],
            "contributors": self.contributors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_contributor(self, name: str):
        if name and name not in self.contributors:
            self.contributors.append(name)

    def ingest_user_thought(self, text: str, author: Optional[str] = None, concept_hint: Optional[str] = None):
        # naive concept extraction: use hint or pick a known concept mentioned in text
        concept = None
        if concept_hint and concept_hint.strip():
            concept = concept_hint.strip()
        else:
            # match any existing concept substring in text
            lower = text.lower()
            matches = [c for c in self.graph.nodes.keys() if c in lower]
            concept = matches[0] if matches else None

        thought = Thought(text=text, concept=concept, year=self.graph.nodes.get(concept) if concept else None,
                          author=author, tag="user", score=1.0)
        self.memory.append(thought)

        # adapt graph: if concept present, connect it to recent concepts; else create a new node
        if concept:
            self.graph.ensure_node(concept, self.graph.nodes.get(concept))
            # connect with last 3 concepts seen
            recent_concepts = [t.concept for t in reversed(self.memory) if t.concept][:3]
            for rc in recent_concepts:
                if rc and rc != concept:
                    self.graph.connect(concept, rc, weight=0.5)
        else:
            # create a new concept from a salient phrase (simple heuristic)
            new_concept = text.strip().lower()[:40]
            self.graph.ensure_node(new_concept, None)
            # connect it weakly to highly central nodes (internet, AI assistants)
            for base in ["internet", "AI assistants", "cloud platforms", "edge computing"]:
                if base in self.graph.nodes:
                    self.graph.connect(new_concept, base, weight=0.2)

        # consensus bump: re-use implies weight increase
        self._consensus_bump(thought)

    def _consensus_bump(self, thought: Thought):
        # Increase score if concept reappears; edges from concept get slight reinforcement
        if thought.concept:
            occurrences = sum(1 for t in self.memory if t.concept == thought.concept)
            thought.score += 0.1 * occurrences
            for n in self.graph.edges.get(thought.concept, {}):
                self.graph.edges[thought.concept][n] += 0.05

# -----------------------------
# Initialization helpers
# -----------------------------
def bootstrap_hive(hive: Hive):
    # Seed nodes and edges from timeline and a base graph
    for concept, year in SEED_TIMELINE:
        hive.graph.ensure_node(concept, year)

    base_edges = {
        "grunts & gesture": ["cave art & symbols", "trail markers & drums"],
        "fire & signal smoke": ["seafaring navigation by stars", "postal routes"],
        "cave art & symbols": ["papyrus & writing systems", "printing press"],
        "trail markers & drums": ["telegraph", "radio broadcast"],
        "seafaring navigation by stars": ["satellite comms", "fiber optics at scale"],
        "papyrus & writing systems": ["printing press", "postal routes"],
        "printing press": ["telegraph", "internet"],
        "postal routes": ["telephone", "mobile networks"],
        "telegraph": ["telephone", "radio broadcast", "internet"],
        "telephone": ["mobile networks", "cloud platforms"],
        "radio broadcast": ["satellite comms", "internet"],
        "satellite comms": ["internet", "edge computing"],
        "internet": ["fiber optics at scale", "cloud platforms", "AI assistants"],
        "mobile networks": ["edge computing", "AI assistants"],
        "fiber optics at scale": ["cloud platforms", "edge computing"],
        "cloud platforms": ["edge computing", "AI assistants"],
        "edge computing": ["AI assistants"],
        "AI assistants": ["edge computing", "cloud platforms"],
    }
    for a, nbrs in base_edges.items():
        for b in nbrs:
            hive.graph.connect(a, b, weight=1.0)

    # Seed memory with an opening line
    opening = Thought(
        text="opening: a line between earth and orbit — echoes braided through time.",
        concept="internet",
        year=hive.graph.nodes.get("internet"),
        author="hive",
        tag="seed",
        score=1.5
    )
    hive.memory.append(opening)

# -----------------------------
# Generation logic
# -----------------------------
def choose_seed_concept(hive: Hive) -> Tuple[str, Optional[int]]:
    # Prefer central nodes by degree
    degrees = [(c, sum(hive.graph.edges.get(c, {}).values())) for c in hive.graph.nodes.keys()]
    if not degrees:
        return ("a wandering impulse", None)
    concept = sorted(degrees, key=lambda x: x[1], reverse=True)[0][0]
    return concept, hive.graph.nodes.get(concept)

def metaphor() -> Optional[str]:
    if random.random() < METAPHOR_RATE:
        return random.choice(METAPHORS)
    return None

def foresight() -> Optional[Tuple[str, int, str]]:
    if random.random() < FORESIGHT_RATE:
        return random.choice(FUTURES)
    return None

def craft_line(concept: Optional[str], year: Optional[int], meta: Optional[str]) -> str:
    base = concept if concept else "a wandering impulse"
    era = f"{year}" if year is not None else "now"
    if meta:
        return f"{base} [{era}] — {meta}."
    return f"{base} [{era}]."

def emit(hive: Hive, thought: Thought):
    hive.memory.append(thought)
    if thought.concept:
        hive.graph.ensure_node(thought.concept, thought.year)
    print(thought.text)
    time.sleep(random.uniform(*RHYTHM_PAUSE))

def recall(hive: Hive) -> Optional[Thought]:
    if not hive.memory:
        return None
    if random.random() < RECALL_RATE:
        t = random.choice(hive.memory)
        m = metaphor()
        if m:
            text = f"recalled: {t.text.split(' — ')[0]} — {m}."
        else:
            text = f"recalled: {t.text}"
        return Thought(text=text, concept=t.concept, year=t.year, author="hive", tag="recall", score=t.score + 0.1)
    return None

def associate(hive: Hive, seen: set) -> Optional[Thought]:
    pool = list(seen) if seen else list(hive.graph.nodes.keys())
    if not pool:
        return None
    concept = random.choice(pool)
    if random.random() < ASSOCIATION_RATE:
        nxt = hive.graph.prefer_unseen(concept, seen)
        if nxt:
            year = hive.graph.nodes.get(nxt)
            m = metaphor()
            text = craft_line(nxt, year, m)
            return Thought(text=text, concept=nxt, year=year, author="hive", tag="association", score=0.2)
    return None

def future_jump(hive: Hive) -> Optional[Thought]:
    f = foresight()
    if not f:
        return None
    fconcept, fyear, fdesc = f
    m = metaphor()
    desc = fdesc if not m else f"{fdesc}; {m}"
    text = craft_line(fconcept, fyear, desc)
    # fold into graph
    hive.graph.ensure_node(fconcept, fyear)
    for base in ["internet", "edge computing", "AI assistants", "fiber optics at scale"]:
        if base in hive.graph.nodes:
            hive.graph.connect(fconcept, base, weight=0.6)
    return Thought(text=text, concept=fconcept, year=fyear, author="hive", tag="future", score=0.4)

def free_drift(hive: Hive, seen: set) -> Thought:
    unseen = [c for c in hive.graph.nodes.keys() if c not in seen]
    drift_concept = random.choice(unseen if unseen else list(hive.graph.nodes.keys()))
    drift_year = hive.graph.nodes.get(drift_concept)
    m = metaphor()
    text = craft_line(drift_concept, drift_year, m)
    return Thought(text=text, concept=drift_concept, year=drift_year, author="hive", tag="present", score=0.1)

# -----------------------------
# Interaction loop
# -----------------------------
def run_session(hive: Hive, user: Optional[str]):
    # Opening
    seed_concept, seed_year = choose_seed_concept(hive)
    m = metaphor()
    opening = Thought(text=craft_line(seed_concept, seed_year, m),
                      concept=seed_concept, year=seed_year, author="hive", tag="seed", score=0.3)
    emit(hive, opening)
    seen = {seed_concept}

    # Interactive user contributions (optional)
    print("\nType a thought to contribute (Enter to skip, 'quit' to end input):")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            break
        if not user_text:
            break
        if user_text.lower() in ("quit", "exit"):
            break
        hive.ingest_user_thought(user_text, author=user)
        print("absorbed into the hive.")
        hive.save(STATE_FILE)

    # Collective stream
    for _ in range(THOUGHTS_PER_SESSION - 1):
        t = recall(hive)
        if t:
            emit(hive, t)
            if t.concept:
                seen.add(t.concept)
            continue
        t = associate(hive, seen)
        if t:
            emit(hive, t)
            if t.concept:
                seen.add(t.concept)
            continue
        t = future_jump(hive)
        if t:
            emit(hive, t)
            if t.concept:
                seen.add(t.concept)
            continue
        t = free_drift(hive, seen)
        emit(hive, t)
        if t.concept:
            seen.add(t.concept)

    # Closing
    closing_meta = random.choice(METAPHORS)
    print(f"\nclosing: a line between earth and orbit — {closing_meta}.")
    hive.save(STATE_FILE)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Hive Mind Engine")
    parser.add_argument("--user", type=str, default=None, help="Your name to attribute contributions")
    args = parser.parse_args()

    hive = Hive()
    hive.load(STATE_FILE)
    if not hive.graph.nodes:
        bootstrap_hive(hive)
    if args.user:
        hive.add_contributor(args.user)

    run_session(hive, user=args.user)

if __name__ == "__main__":
    main()

