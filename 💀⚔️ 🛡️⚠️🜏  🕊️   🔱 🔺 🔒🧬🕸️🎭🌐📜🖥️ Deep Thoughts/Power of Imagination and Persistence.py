#!/usr/bin/env python3
# Live Protocol Engine: iterative truth-seeking with distributed consensus
# Standard library only; integrates via TCP JSON-lines.
# Run: python live_protocol.py --host 0.0.0.0 --port 8765

import asyncio
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# ===== Core data types =====

@dataclass
class Thought:
    """A unit of meaning: both statement and function-call."""
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0                # Confidence from consensus
    truth_estimate: float = 0.0       # Engine's estimate of 'fit'
    history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Candidate:
    """Agent-produced refinement or response."""
    content: str
    rationale: str
    quality: float                    # Agent's self-rated quality [0..1]
    error_signal: float               # How 'wrong' it thinks the input was [0..1]
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===== Utilities =====

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def now_ms() -> int:
    return int(time.time() * 1000)

# ===== Agents (voices in the chorus) =====

class Agent:
    name: str = "agent"
    weight: float = 1.0  # Voting weight in consensus

    async def process(self, thought: Thought) -> Candidate:
        raise NotImplementedError

class Predictor(Agent):
    """Proposes a direct answer by pattern completion and simplification."""
    name = "predictor"
    weight = 1.0

    async def process(self, thought: Thought) -> Candidate:
        content = thought.content.strip()
        rationale = "Pattern-complete and simplify the proposition."
        # Simple heuristic: if content looks like a math equation, try to normalize.
        if any(s in content for s in ["=", "+", "-", "*", "/", "^"]):
            proposal = f"Proposed resolution: normalize and test each step for consistency.\nInput: {content}"
            quality = 0.6
            error_signal = 0.4
        else:
            proposal = f"Direct answer attempt: {content}"
            quality = 0.5
            error_signal = 0.5
        return Candidate(
            content=proposal,
            rationale=rationale,
            quality=quality,
            error_signal=error_signal,
            metadata={"type": "proposal"}
        )

class Critic(Agent):
    """Identifies hidden assumptions and failure modes."""
    name = "critic"
    weight = 1.2

    async def process(self, thought: Thought) -> Candidate:
        text = thought.content
        assumptions = []
        if "always" in text.lower(): assumptions.append("Absolute claim 'always'")
        if "never" in text.lower(): assumptions.append("Absolute claim 'never'")
        if "=" in text and "?" not in text: assumptions.append("Equation stated without conditions")
        rationale = "Surface hidden assumptions and test boundary conditions."
        critique = "Assumptions detected: " + (", ".join(assumptions) if assumptions else "none explicit")
        quality = 0.65 + 0.1 * bool(assumptions)
        error_signal = 0.55 if assumptions else 0.45
        return Candidate(
            content=critique,
            rationale=rationale,
            quality=clamp(quality),
            error_signal=clamp(error_signal),
            metadata={"type": "critique", "assumptions": assumptions}
        )

class Refiner(Agent):
    """Turns critique into an improved next step with testable checks."""
    name = "refiner"
    weight = 1.3

    async def process(self, thought: Thought) -> Candidate:
        # Propose measurable checks and a smaller next step
        rationale = "Refine to a testable step with simple checks."
        checks = [
            "Define the target claim precisely.",
            "List two boundary cases that could falsify it.",
            "Run a consistency check: does the claim contradict any prior step?"
        ]
        proposal = "Refined next step:\n- " + "\n- ".join(checks)
        return Candidate(
            content=proposal,
            rationale=rationale,
            quality=0.7,
            error_signal=0.35,
            metadata={"type": "refinement", "checks": checks}
        )

class Synthesizer(Agent):
    """Fuses proposals and critiques into a coherent, minimal answer."""
    name = "synthesizer"
    weight = 1.5

    async def process(self, thought: Thought) -> Candidate:
        rationale = "Fuse proposals and critiques into minimal coherent form."
        # Minimal synthesis: compress to a single actionable instruction
        synthesis = "Action: state the claim, list assumptions, test 2 edge cases, revise once."
        return Candidate(
            content=synthesis,
            rationale=rationale,
            quality=0.75,
            error_signal=0.3,
            metadata={"type": "synthesis"}
        )

# ===== Consensus and convergence =====

class Consensus:
    """Weighted voting and confidence aggregation."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def aggregate(self, candidates: List[Tuple[Agent, Candidate]]) -> Tuple[str, float, Dict[str, Any]]:
        # Weighted average of quality minus error; pick top content by adjusted score
        scored = []
        for agent, cand in candidates:
            adjusted = clamp(cand.quality * agent.weight - 0.5 * cand.error_signal)
            scored.append((adjusted, agent, cand))
        scored.sort(key=lambda t: t[0], reverse=True)
        best_adjusted, best_agent, best_cand = scored[0]
        confidence = clamp(sum(s[0] for s in scored) / max(1, len(scored)))
        meta = {
            "winner": best_agent.name,
            "confidence": confidence,
            "votes": [
                {"agent": a.name, "score": clamp(s), "quality": c.quality, "error": c.error_signal, "type": c.metadata.get("type")}
                for (s, a, c) in scored
            ]
        }
        return best_cand.content, confidence, meta

def has_converged(prev: Thought, current: Thought, threshold: float = 0.75) -> bool:
    return current.truth_estimate >= threshold and current.score >= threshold

# ===== Engine =====

class LiveProtocolEngine:
    def __init__(self, agents: Optional[List[Agent]] = None, max_iters: int = 4):
        self.agents = agents or [Predictor(), Critic(), Refiner(), Synthesizer()]
        self.consensus = Consensus(self.agents)
        self.max_iters = max_iters

    async def iterate(self, thought: Thought) -> Thought:
        for step in range(self.max_iters):
            candidates: List[Tuple[Agent, Candidate]] = []
            for agent in self.agents:
                cand = await agent.process(thought)
                candidates.append((agent, cand))

            best_content, conf, meta = self.consensus.aggregate(candidates)

            # Update thought
            thought.history.append({
                "ts": now_ms(),
                "step": step,
                "candidates": [
                    {"agent": a.name, "content": c.content, "quality": c.quality, "error": c.error_signal}
                    for (a, c) in [(ag, ca) for (ag, ca) in [(a, c) for (a, c) in candidates]]
                ],
                "winner": meta["winner"],
                "confidence": conf,
            })

            # Simple improvement rule: truth_estimate grows with confidence and diversity
            diversity_bonus = 0.05 * len(set(a.name for a, _ in candidates))
            thought.content = best_content
            thought.score = conf
            thought.truth_estimate = clamp(thought.truth_estimate * 0.6 + conf * 0.35 + diversity_bonus)

            if has_converged(thought, thought):
                break

        return thought

# ===== TCP JSON-lines server =====

class ProtocolServer:
    """
    TCP server. Each line is a JSON object:
    {
      "utterance": "string",
      "context": {...},        # optional
      "max_iters": 4           # optional
    }

    Response is a JSON object per request:
    {
      "content": "string",     # synthesized result
      "score": 0.0..1.0,
      "truth_estimate": 0.0..1.0,
      "history": [ ... ],
      "meta": { "engine": "live-protocol", "version": "0.1" }
    }
    """
    def __init__(self, host: str, port: int, engine: LiveProtocolEngine):
        self.host = host
        self.port = port
        self.engine = engine

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    writer.write((json.dumps({"error": "invalid_json"}) + "\n").encode())
                    await writer.drain()
                    continue

                utterance = req.get("utterance", "")
                context = req.get("context", {})
                max_iters = req.get("max_iters", self.engine.max_iters)

                engine = LiveProtocolEngine(agents=self.engine.agents, max_iters=max_iters)
                thought = Thought(content=utterance, context=context)
                result = await engine.iterate(thought)

                resp = {
                    "content": result.content,
                    "score": round(result.score, 4),
                    "truth_estimate": round(result.truth_estimate, 4),
                    "history": result.history,
                    "meta": {
                        "engine": "live-protocol",
                        "version": "0.1",
                        "host": self.host,
                        "port": self.port,
                        "received_from": addr,
                    }
                }
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        except Exception as e:
            # Keep the server live; report minimal errors
            writer.write((json.dumps({"error": "server_exception"}) + "\n").encode())
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
        print(f"[live-protocol] listening on {addrs}")
        async with server:
            await server.serve_forever()

# ===== Entry point =====

def parse_args():
    p = argparse.ArgumentParser(description="Live Protocol Engine")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--iters", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    engine = LiveProtocolEngine(max_iters=args.iters)
    server = ProtocolServer(args.host, args.port, engine)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()

