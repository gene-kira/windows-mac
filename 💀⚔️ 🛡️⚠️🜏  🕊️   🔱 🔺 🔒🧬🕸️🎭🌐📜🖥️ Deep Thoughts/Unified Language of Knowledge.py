#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Language of Knowledge (ULK) – Live Prototype
- Fuses human verbs, math expressions, and theory operators
- Layered interpreter with safe math and a tiny proof engine
"""

from __future__ import annotations
import ast
import operator as op
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# -------- Ritual IO -----------------------------------------------------------

def chant(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] ✦ {msg}")

# -------- Safe math evaluator -------------------------------------------------

ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg, ast.UAdd: op.pos,
    ast.BitXor: None, ast.BitAnd: None, ast.BitOr: None,
    ast.FloorDiv: None, ast.MatMult: None,
}

def safe_eval(expr: str, sym: Dict[str, float]) -> float:
    """
    Evaluate a numeric expression safely using the symbol table.
    Supports + - * / ** % and variables. No function calls or attributes.
    """
    node = ast.parse(expr, mode='eval').body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Num):      # 42
            return float(n.n)
        if isinstance(n, ast.Constant): # Py3.11 constants
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Non-numeric constant")
        if isinstance(n, ast.BinOp):
            if type(n.op) not in ALLOWED_OPS or ALLOWED_OPS[type(n.op)] is None:
                raise ValueError("Operator not allowed")
            return ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in ALLOWED_OPS or ALLOWED_OPS[type(n.op)] is None:
                raise ValueError("Unary operator not allowed")
            return ALLOWED_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Name):
            if n.id in sym:
                return float(sym[n.id])
            raise NameError(f"Unknown symbol: {n.id}")
        if isinstance(n, ast.Expr):
            return _eval(n.value)
        raise ValueError("Unsupported expression")
    return _eval(node)

# -------- Context (shared registers) -----------------------------------------

@dataclass
class Context:
    vars: Dict[str, float] = field(default_factory=lambda: {
        "courage": 1, "fear": 0, "hope": 0, "knowledge": 0
    })
    symbols: Dict[str, str] = field(default_factory=lambda: {"𓂀": "insight", "未来": "future"})
    verbs: Dict[str, Callable[['Context', List[str]], None]] = field(default_factory=dict)
    ops: Dict[str, Callable[['Context', List[str]], None]] = field(default_factory=dict)
    theorems: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def get(self, k: str, default: float = 0.0) -> float:
        return float(self.vars.get(k, default))

    def set(self, k: str, v: float) -> None:
        self.vars[k] = float(v)

    def inc(self, k: str, amt: float = 1.0) -> None:
        self.set(k, self.get(k) + amt)

    def dec(self, k: str, amt: float = 1.0) -> None:
        self.set(k, self.get(k) - amt)

    def define_verb(self, name: str, fn: Callable[['Context', List[str]], None]) -> None:
        self.verbs[name.lower()] = fn

    def define_op(self, name: str, fn: Callable[['Context', List[str]], None]) -> None:
        self.ops[name.lower()] = fn

# -------- Built-in linguistic verbs ------------------------------------------

def v_echo(ctx: Context, args: List[str]) -> None:
    chant("Echo: " + " ".join(args))

def v_print(ctx: Context, args: List[str]) -> None:
    chant("Print: " + " ".join(args))

def v_set(ctx: Context, args: List[str]) -> None:
    # set x = 3.14
    m = re.match(r'([\w𓂀未来]+)\s*=\s*([-+]?\d+(\.\d+)?)', " ".join(args))
    if not m:
        chant("Set failed: use `set name = number`")
        return
    k, v = m.group(1), float(m.group(2))
    ctx.set(k, v)
    chant(f"Set {k} = {v}")

def v_define(ctx: Context, args: List[str]) -> None:
    # define verb bless => echo "text"
    joined = " ".join(args)
    m = re.match(r'verb\s+(\w+)\s*=>\s*echo\s+"(.+)"', joined, re.IGNORECASE)
    if not m:
        chant('Define failed: try `define verb <name> => echo "text"`')
        return
    name, text = m.groups()
    def _fn(c: Context, _a: List[str]) -> None:
        chant(f"{name.capitalize()}: {text}")
    ctx.define_verb(name, _fn)
    chant(f"Verb '{name}' installed.")

# -------- Theoretical operators ----------------------------------------------

def o_axiom(ctx: Context, args: List[str]) -> None:
    # axiom name: statement
    joined = " ".join(args)
    m = re.match(r'(\w+)\s*:\s*(.+)', joined)
    if not m:
        chant("Axiom failed: use `axiom Name: statement`")
        return
    name, stmt = m.groups()
    ctx.theorems[name] = {"type": "axiom", "stmt": stmt}
    chant(f"Axiom '{name}': {stmt}")

def o_assume(ctx: Context, args: List[str]) -> None:
    stmt = " ".join(args)
    ctx.assumptions.append(stmt)
    chant(f"Assume: {stmt}")

def o_infer(ctx: Context, args: List[str]) -> None:
    # infer Name: conclusion <- assumption ref
    joined = " ".join(args)
    m = re.match(r'(\w+)\s*:\s*(.+)', joined)
    if not m:
        chant("Infer failed: use `infer Name: conclusion`")
        return
    name, concl = m.groups()
    ctx.theorems[name] = {"type": "inference", "stmt": concl, "deps": list(ctx.assumptions)}
    chant(f"Infer '{name}': {concl}  | deps: {ctx.assumptions}")

def o_prove(ctx: Context, args: List[str]) -> None:
    # prove Name using A,B,...
    joined = " ".join(args)
    m = re.match(r'(\w+)\s+using\s+([\w, ]+)', joined)
    if not m:
        chant("Prove failed: use `prove Name using A,B`")
        return
    name, deps = m.groups()
    deps_list = [d.strip() for d in deps.split(",") if d.strip()]
    ctx.theorems[name] = {"type": "theorem", "deps": deps_list, "stmt": f"Derived from {', '.join(deps_list)}"}
    chant(f"Proved '{name}' ← {', '.join(deps_list)}")

# -------- Math layer operators -----------------------------------------------

def o_sum(ctx: Context, args: List[str]) -> None:
    # ∑ x over a..b    OR  sum x over a..b
    joined = " ".join(args)
    m = re.match(r'(\w+)\s+over\s+([-+]?\d+)\.\.([-+]?\d+)', joined)
    if not m:
        chant("Sum failed: use `∑ x over a..b`")
        return
    var, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    total = 0.0
    for i in range(min(a,b), max(a,b)+1):
        total += ctx.get(var) if var in ctx.vars else i
    ctx.set("Σ", total)
    chant(f"Σ = {total}")

def o_delta(ctx: Context, args: List[str]) -> None:
    # Δ(name) = expr
    joined = " ".join(args)
    m = re.match(r'([\w]+)\s*=\s*(.+)', joined)
    if not m:
        chant("Δ failed: use `Δ(name) = expr` with call syntax")
        return
    name, expr = m.groups()
    try:
        val = safe_eval(expr, ctx.vars)
        ctx.set(name, val)
        chant(f"Δ {name} = {val}  | from {expr}")
    except Exception as e:
        chant(f"Δ error: {e}")

def o_eval(ctx: Context, args: List[str]) -> None:
    expr = " ".join(args)
    try:
        val = safe_eval(expr, ctx.vars)
        ctx.set("ans", val)
        chant(f"eval({expr}) = {val}")
    except Exception as e:
        chant(f"Eval error: {e}")

# -------- Unified interpreter -------------------------------------------------

class ULK:
    IF_RE = re.compile(r'^if\s+(.+?)\s*\{\s*(.+?)\s*\}\s*(else\s*\{\s*(.+?)\s*\}\s*)?$', re.IGNORECASE)

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._install_builtins()

    def _install_builtins(self):
        # verbs
        self.ctx.define_verb("echo", v_echo)
        self.ctx.define_verb("print", v_print)
        self.ctx.define_verb("set", v_set)
        self.ctx.define_verb("define", v_define)
        # operators
        self.ctx.define_op("axiom", o_axiom)
        self.ctx.define_op("assume", o_assume)
        self.ctx.define_op("infer", o_infer)
        self.ctx.define_op("prove", o_prove)
        self.ctx.define_op("∑", o_sum)
        self.ctx.define_op("sum", o_sum)
        self.ctx.define_op("Δ", o_delta)
        self.ctx.define_op("eval", o_eval)

    def execute(self, line: str) -> None:
        s = line.strip()
        if not s or s.startswith("//"):
            return

        # increments/decrements: x++ / x--
        m = re.match(r'^([\w𓂀未来]+)\s*(\+\+|--)$', s)
        if m:
            name, op = m.groups()
            if op == "++": self.ctx.inc(name, 1); chant(f"{name}++ → {self.ctx.get(name)}")
            else:           self.ctx.dec(name, 1); chant(f"{name}-- → {self.ctx.get(name)}")
            return

        # inline if-else: if cond { ... } else { ... }
        m = self.IF_RE.match(s)
        if m:
            cond, then_cmd, _, else_cmd = m.groups()
            if self._eval_cond(cond):
                chant(f"if TRUE: {cond}")
                self.execute(then_cmd)
            else:
                chant(f"if FALSE: {cond}")
                if else_cmd: self.execute(else_cmd)
            return

        # op call syntax: OP(args) or OP args...
        # Unicode call: Δ(knowledge) = courage*2 + fear**2
        if s.startswith("Δ("):
            # Transform call to: Δ knowledge = expr (split by ') =')
            call_m = re.match(r'^Δ\((\w+)\)\s*=\s*(.+)$', s)
            if call_m:
                name, expr = call_m.groups()
                self.ctx.ops["Δ"](self.ctx, [f"{name} = {expr}"])
                return

        if s.startswith("∑ "):
            # ∑ x over 1..10
            args = s[2:].strip().split()
            self.ctx.ops["∑"](self.ctx, args)
            return

        # eval math
        if s.lower().startswith("eval "):
            self.ctx.ops["eval"](self.ctx, [s[5:].strip()])
            return

        # axiom/assume/infer/prove
        for op_name in ["axiom", "assume", "infer", "prove"]:
            if s.lower().startswith(op_name + " "):
                self.ctx.ops[op_name](self.ctx, s[len(op_name):].strip().split())
                return

        # verbs: echo, print, set, define, custom
        tokens = s.split()
        verb = tokens[0].lower()
        args = tokens[1:]
        if verb in self.ctx.verbs:
            self.ctx.verbs[verb](self.ctx, args)
            return

        chant(f"Unknown utterance: {s}")

    def _eval_cond(self, cond: str) -> bool:
        # Supports comparisons with vars: courage > fear, knowledge == 0, eval(expr) style
        m = re.match(r'^\s*([\w]+)\s*(<=|>=|==|!=|<|>)\s*([\w.\-+*/^ ]+)\s*$', cond)
        if not m:
            # Try direct numeric eval: eval(cond) != 0
            try:
                val = safe_eval(cond.replace("^", "**"), self.ctx.vars)
                return bool(val)
            except Exception:
                chant(f"Bad condition: {cond}")
                return False
        lhs, op_sym, rhs_raw = m.groups()
        lv = self.ctx.get(lhs)
        rhs_expr = rhs_raw.replace("^", "**")
        try:
            rv = safe_eval(rhs_expr, self.ctx.vars)
        except Exception:
            # fallback: symbol compare
            try:
                rv = float(self.ctx.vars.get(rhs_raw.strip(), 0.0))
            except Exception:
                rv = 0.0
        return {
            "<":  lv <  rv,
            "<=": lv <= rv,
            ">":  lv >  rv,
            ">=": lv >= rv,
            "==": lv == rv,
            "!=": lv != rv,
        }[op_sym]

# -------- REPL ----------------------------------------------------------------

def repl(engine: ULK, ctx: Context):
    chant("— Unified Language of Knowledge booted —")
    chant('Examples:')
    chant('  echo "𓂀 Logos ∑(hope)"')
    chant('  if fear < courage { print "未来 bright" } else { echo "invoke ancestors" }')
    chant('  Δ(knowledge) = courage*2 + fear**2')
    chant('  eval (3 + courage) ** 2')
    chant('  axiom Conservation: energy is constant')
    chant('  assume symmetry across frames')
    chant('  infer Renewal: cycles persist')
    chant('  prove Equivalence using Conservation,Renewal')
    chant('Type `quit` to exit.')
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                chant("— End of ritual —")
                break
            engine.execute(line)
    except KeyboardInterrupt:
        chant("\n— End of ritual —")

# -------- Main ----------------------------------------------------------------

def main():
    ctx = Context()
    engine = ULK(ctx)
    repl(engine, ctx)

if __name__ == "__main__":
    main()

