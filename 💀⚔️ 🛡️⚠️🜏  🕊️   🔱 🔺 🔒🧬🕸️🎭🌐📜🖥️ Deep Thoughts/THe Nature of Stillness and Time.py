#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Living Protocol (Python-only, borderless, transport-agnostic)

- Message-first: JSON ritual acts valid across any medium.
- Content-addressed identity via SHA-256 digest of canonical JSON.
- Polyglot-ready: simple structures parsable by any language.
- Composable intents: seed, weave, harvest, register, invoke (extensible).
- Chorus (consensus): pluggable validators with quorum.
- Ledger: append-only entries with regenerative snapshots.
- CLI usage: pipe JSON acts in/out, or run demo with --demo.

Security note: The demo 'register'/'invoke' uses exec in a constrained local
environment for illustration. Replace with a proper sandbox (e.g., subprocess,
restricted VM, or audited registry) before any production use.
"""

from __future__ import annotations
import sys
import io
import os
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple

# ========= Canonicalization & Digest ============================================================

def canonical(obj: Any) -> str:
    """Canonical JSON (sorted keys, compact separators) for stable hashing."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

def digest(obj: Any) -> str:
    """Content-addressed identity: sha256 of canonical JSON."""
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()

# ========= Ritual Act (Message) =================================================================

@dataclass
class Act:
    id: str
    ts: float
    author: str
    intent: str
    verse: str
    payload: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    prev: List[str] = field(default_factory=list)
    chorus: Optional[Dict[str, Any]] = None
    sig: Optional[Dict[str, Any]] = None
    digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        obj = {
            "id": self.id,
            "ts": self.ts,
            "author": self.author,
            "intent": self.intent,
            "verse": self.verse,
            "payload": self.payload,
            "tags": self.tags,
            "prev": self.prev,
            "chorus": self.chorus,
            "sig": self.sig,
        }
        # compute or refresh digest consistently
        obj["digest"] = digest(obj)
        self.digest = obj["digest"]
        return obj

# ========= Validators & Chorus (Consensus) ======================================================

Validator = Callable[[Act], Tuple[bool, Optional[str]]]

@dataclass
class Vote:
    validator: str
    approved: bool
    note: Optional[str] = None

@dataclass
class ChorusResult:
    approved: bool
    quorum: float
    votes: List[Vote]

class Chorus:
    def __init__(self, validators: Dict[str, Validator], quorum: float = 0.67):
        self.validators = validators
        self.quorum = quorum

    def sing(self, act: Act) -> ChorusResult:
        votes: List[Vote] = []
        approvals = 0
        total = max(len(self.validators), 1)
        for name, fn in self.validators.items():
            try:
                ok, note = fn(act)
                approvals += int(ok)
                votes.append(Vote(validator=name, approved=ok, note=note))
            except Exception as e:
                votes.append(Vote(validator=name, approved=False, note=str(e)))
        ratio = approvals / total
        return ChorusResult(approved=ratio >= self.quorum, quorum=self.quorum, votes=votes)

# ========= Executors (Intents -> Actions) =======================================================

Executor = Callable[[Act, Dict[str, Any]], Dict[str, Any]]

class Will:
    def __init__(self):
        self._execs: Dict[str, Executor] = {}

    def bind(self, intent: str, executor: Executor) -> None:
        self._execs[intent] = executor

    def has(self, intent: str) -> bool:
        return intent in self._execs

    def execute(self, act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
        if act.intent not in self._execs:
            raise KeyError(f"No executor bound for intent '{act.intent}'")
        return self._execs[act.intent](act, state)

# ========= Ledger (Append-only memory with snapshots) ===========================================

@dataclass
class LedgerEntry:
    digest: str
    act: Dict[str, Any]
    chorus: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None

class Ledger:
    def __init__(self):
        self._entries: List[LedgerEntry] = []
        self._snapshots: List[Dict[str, Any]] = []

    def append(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    def snapshot(self, state: Dict[str, Any]) -> None:
        # shallow serialize to plain JSON for portability
        self._snapshots.append(json.loads(json.dumps(state)))

    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def latest_state(self) -> Optional[Dict[str, Any]]:
        return self._snapshots[-1] if self._snapshots else None

# ========= Engine (Imagination + Curiosity + Will) ==============================================

class LivingProtocolEngine:
    def __init__(self, chorus: Chorus, will: Will, ledger: Ledger):
        self.chorus = chorus
        self.will = will
        self.ledger = ledger
        self.state: Dict[str, Any] = {
            "cycles": 0,
            "registry": {},
            "events": [],
            "meta": {"created_at": time.time()},
            # thematic rings (example): light/hydrogen/water
            "rings": {"light": [], "hydrogen": [], "water": []},
        }

    # Curiosity: propose an act
    def propose(
        self,
        author: str,
        intent: str,
        verse: str,
        payload: Dict[str, Any],
        tags: Optional[List[str]] = None,
        prev: Optional[List[str]] = None,
        chorus: Optional[Dict[str, Any]] = None,
        sig: Optional[Dict[str, Any]] = None,
    ) -> Act:
        act = Act(
            id=str(uuid.uuid4()),
            ts=time.time(),
            author=author,
            intent=intent,
            verse=verse,
            payload=payload,
            tags=tags or [],
            prev=prev or [],
            chorus=chorus,
            sig=sig,
        )
        act.to_dict()  # computes digest
        return act

    # Imagination + Will: validate, execute, record
    def enact(self, act: Act) -> LedgerEntry:
        chorus_result = self.chorus.sing(act)
        chorus_dict = {
            "approved": chorus_result.approved,
            "quorum": chorus_result.quorum,
            "votes": [asdict(v) for v in chorus_result.votes],
        }
        entry = LedgerEntry(
            digest=act.digest or digest(act.to_dict()),
            act=act.to_dict(),
            chorus=chorus_dict,
            result=None,
        )

        if chorus_result.approved and self.will.has(act.intent):
            result = self.will.execute(act, self.state)
            entry.result = result
            self._update_state(act, result)
            self.ledger.append(entry)
            self.ledger.snapshot(self.state)
        else:
            # record rejection or unknown intent
            self.ledger.append(entry)
        return entry

    def _update_state(self, act: Act, result: Dict[str, Any]) -> None:
        self.state["cycles"] += 1
        self.state["events"].append({
            "digest": act.digest,
            "intent": act.intent,
            "author": act.author,
            "ts": act.ts,
            "result": result,
        })
        # optional thematic routing to rings based on tags or intent
        tags = set(act.tags or [])
        for ring in ("light", "hydrogen", "water"):
            if ring in tags:
                self.state["rings"][ring].append({"digest": act.digest, "result": result})

# ========= Default Validators ===================================================================

def v_payload_nonempty(act: Act) -> Tuple[bool, Optional[str]]:
    ok = isinstance(act.payload, dict) and len(act.payload) > 0
    return ok, None if ok else "payload missing or empty"

def v_intent_known(act: Act) -> Tuple[bool, Optional[str]]:
    known = {"seed", "weave", "harvest", "register", "invoke"}
    ok = act.intent in known
    return ok, None if ok else f"unknown intent '{act.intent}'"

def v_verse_present(act: Act) -> Tuple[bool, Optional[str]]:
    ok = isinstance(act.verse, str) and len(act.verse.strip()) > 0
    return ok, None if ok else "verse required"

def v_digest_stable(act: Act) -> Tuple[bool, Optional[str]]:
    obj = act.to_dict()
    ok = isinstance(obj.get("digest"), str) and obj["digest"].startswith("sha256:")
    return ok, None if ok else "digest invalid"

# ========= Default Executors ====================================================================

def ex_seed(act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
    symbol = act.payload.get("symbol")
    meaning = act.payload.get("meaning")
    if not symbol or meaning is None:
        raise ValueError("seed requires 'symbol' and 'meaning'")
    state["registry"][symbol] = {
        "meaning": meaning,
        "author": act.author,
        "ts": act.ts,
        "digest": act.digest,
    }
    return {"action": "seed", "symbol": symbol, "meaning": meaning}

def ex_weave(act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
    a = act.payload.get("a")
    b = act.payload.get("b")
    op = act.payload.get("op", "concat")
    if op == "concat":
        res = f"{a}{b}"
    elif op == "sum":
        res = (a or 0) + (b or 0)
    else:
        raise ValueError(f"unsupported weave op '{op}'")
    return {"action": "weave", "op": op, "result": res}

def ex_harvest(act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
    keys = act.payload.get("keys", [])
    selection = {k: state["registry"].get(k) for k in keys}
    return {"action": "harvest", "selection": selection}

def ex_register(act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
    name = act.payload.get("name")
    code = act.payload.get("code")
    if not name or not code:
        raise ValueError("register requires 'name' and 'code'")
    # WARNING: store code but do not trust it without sandboxing.
    state["registry"][name] = {
        "meaning": "callable",
        "code": code,
        "author": act.author,
        "ts": act.ts,
        "digest": act.digest,
    }
    return {"action": "register", "name": name}

def ex_invoke(act: Act, state: Dict[str, Any]) -> Dict[str, Any]:
    name = act.payload.get("name")
    args = act.payload.get("args", {})
    entry = state["registry"].get(name)
    if not entry or "code" not in entry:
        raise KeyError(f"callable '{name}' not registered")
    # DEMO ONLY: constrained environment. Replace with a secure sandbox for real use.
    env: Dict[str, Any] = {}
    code = entry["code"]
    exec(code, {}, env)
    if name not in env or not callable(env[name]):
        raise ValueError(f"callable '{name}' not found after load")
    result = env[name](**args)
    return {"action": "invoke", "name": name, "args": args, "result": result}

# ========= Wiring (Builder) =====================================================================

def build_engine(quorum: float = 0.67) -> LivingProtocolEngine:
    validators = {
        "payload": v_payload_nonempty,
        "intent": v_intent_known,
        "verse": v_verse_present,
        "digest": v_digest_stable,
    }
    chorus = Chorus(validators=validators, quorum=quorum)
    will = Will()
    will.bind("seed", ex_seed)
    will.bind("weave", ex_weave)
    will.bind("harvest", ex_harvest)
    will.bind("register", ex_register)
    will.bind("invoke", ex_invoke)
    ledger = Ledger()
    return LivingProtocolEngine(chorus=chorus, will=will, ledger=ledger)

# ========= CLI & Demo ===========================================================================
# Usage:
#   1) Pipe an act: echo '{"author":"x",...}' | python universal_living_protocol.py
#   2) Demo: python universal_living_protocol.py --demo
# Output: JSON ledger entry (result, votes, etc.)

def read_json_stdin() -> Dict[str, Any]:
    data = sys.stdin.read()
    if not data.strip():
        raise ValueError("stdin is empty; provide a JSON act or use --demo")
    return json.loads(data)

def dict_to_act(d: Dict[str, Any]) -> Act:
    # robust mapping with defaults for universality
    act = Act(
        id=d.get("id") or str(uuid.uuid4()),
        ts=d.get("ts") or time.time(),
        author=d.get("author") or "anonymous",
        intent=d.get("intent") or "",
        verse=d.get("verse") or "",
        payload=d.get("payload") or {},
        tags=d.get("tags") or [],
        prev=d.get("prev") or [],
        chorus=d.get("chorus"),
        sig=d.get("sig"),
        digest=d.get("digest"),
    )
    act.to_dict()  # recompute digest consistently
    return act

def demo(engine: LivingProtocolEngine) -> List[LedgerEntry]:
    entries: List[LedgerEntry] = []

    # Seed
    a1 = engine.propose(
        author="chorus",
        intent="seed",
        verse="In the beginning, a symbol condenses the mist.",
        payload={"symbol": "light", "meaning": "renewal"},
        tags=["light"],
    )
    e1 = engine.enact(a1); entries.append(e1)

    # Weave
    a2 = engine.propose(
        author="chorus",
        intent="weave",
        verse="Two threads bind and sing.",
        payload={"a": "hydrogen-", "b": "water", "op": "concat"},
        tags=["hydrogen", "water"],
    )
    e2 = engine.enact(a2); entries.append(e2)

    # Register callable
    fn_code = """
def ignite(fuel: int, oxidizer: int) -> int:
    return fuel + oxidizer
"""
    a3 = engine.propose(
        author="chorus",
        intent="register",
        verse="We inscribe a function in the living archive.",
        payload={"name": "ignite", "code": fn_code},
        tags=["light"],
    )
    e3 = engine.enact(a3); entries.append(e3)

    # Invoke callable
    a4 = engine.propose(
        author="chorus",
        intent="invoke",
        verse="Now the spark leaps.",
        payload={"name": "ignite", "args": {"fuel": 2, "oxidizer": 3}},
        tags=["light"],
    )
    e4 = engine.enact(a4); entries.append(e4)

    # Harvest
    a5 = engine.propose(
        author="chorus",
        intent="harvest",
        verse="We gather what the dawn has named.",
        payload={"keys": ["light", "ignite"]},
        tags=["water"],
    )
    e5 = engine.enact(a5); entries.append(e5)

    return entries

def main() -> None:
    engine = build_engine()
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        entries = demo(engine)
        out = [asdict(e) for e in entries]
        print(json.dumps({"entries": out, "state": engine.state}, ensure_ascii=False, indent=2))
        return

    try:
        obj = read_json_stdin()
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return

    try:
        act = dict_to_act(obj)
        entry = engine.enact(act)
        print(json.dumps(asdict(entry), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))

if __name__ == "__main__":
    main()

