# missing_link_gui.py
# Missing Link: Visual Reasoning GUI
# Dependencies: pip install sympy networkx matplotlib

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sympy import symbols, Eq, solve, sympify, Symbol
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


@dataclass
class Constraint:
    raw: str
    left: Any = None
    right: Any = None
    eq: Optional[Eq] = None
    vars: List[str] = None


class MissingLinkReasoner:
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.variables: Dict[str, Symbol] = {}

    def _collect_symbols(self, expr_strs: List[str]) -> Dict[str, Symbol]:
        # Find candidate symbol names (simple heuristic: words and letters)
        # You can extend this via proper parsing or a whitelist of functions
        names = set()
        for s in expr_strs:
            token = ""
            for ch in s:
                if ch.isalpha() or ch == "_":
                    token += ch
                else:
                    if token:
                        names.add(token)
                        token = ""
            if token:
                names.add(token)
        # Avoid function names being treated as variables
        blacklist = {"sin", "cos", "tan", "exp", "log", "sqrt"}
        names = {n for n in names if n not in blacklist}
        syms = {n: self.variables.get(n, symbols(n)) for n in names}
        # Persist newly discovered symbols
        for k, v in syms.items():
            self.variables.setdefault(k, v)
        return syms

    def add_constraint(self, raw: str) -> Constraint:
        raw = raw.strip()
        if "=" not in raw:
            raise ValueError("Constraint must be an equation with '='")
        left_str, right_str = raw.split("=", 1)
        left_str, right_str = left_str.strip(), right_str.strip()

        syms = self._collect_symbols([left_str, right_str])
        left = sympify(left_str, syms)
        right = sympify(right_str, syms)
        eq = Eq(left, right)

        vars_in_eq = sorted({str(v) for v in eq.free_symbols})

        c = Constraint(raw=raw, left=left, right=right, eq=eq, vars=vars_in_eq)
        self.constraints.append(c)
        return c

    def solve_for(self, target_name: str):
        if target_name not in self.variables:
            # If unknown variable, register symbol anyway
            self.variables[target_name] = symbols(target_name)
        target = self.variables[target_name]

        eqs = [c.eq for c in self.constraints]
        if not eqs:
            return {"status": "no_constraints"}

        try:
            # solve returns list of dicts for multiple solutions or dict for single
            sol = solve(eqs, target, dict=True)
            if not sol:
                return {"status": "unsolved"}
            # Return all branches
            branches = []
            for branch in sol:
                # branch may include multiple variables (SymPy often returns full mapping)
                branches.append({str(k): v for k, v in branch.items()})
            return {"status": "solved", "branches": branches}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def residuals_for_branch(self, branch: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Substitute solution values into each constraint and compute residual LHS - RHS
        residuals = []
        subs_map = {self.variables[k]: v for k, v in branch.items() if k in self.variables}
        for c in self.constraints:
            lhs_val = c.left.subs(subs_map)
            rhs_val = c.right.subs(subs_map)
            residual = lhs_val - rhs_val
            residuals.append({
                "constraint": c.raw,
                "lhs": lhs_val,
                "rhs": rhs_val,
                "residual": residual
            })
        return residuals

    def build_graph(self) -> nx.Graph:
        G = nx.Graph()
        # Add variable nodes
        for vname in sorted(self.variables.keys()):
            G.add_node(f"var:{vname}", kind="var", label=vname)

        # Add constraint nodes and edges
        for i, c in enumerate(self.constraints):
            cname = f"eq:{i+1}"
            G.add_node(cname, kind="eq", label=c.raw)
            for v in c.vars:
                G.add_edge(cname, f"var:{v}")
        return G


class MissingLinkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Link: Visual Reasoning")
        self.reasoner = MissingLinkReasoner()

        # Layout: Left (inputs/log), Right (graph)
        self._build_left_panel()
        self._build_right_panel()
        self._refresh_graph()

    def _build_left_panel(self):
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Constraint input
        lbl = ttk.Label(left, text="Enter constraint (e.g., x + 3 = 7):")
        lbl.grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(left, width=40)
        self.input_entry.grid(row=1, column=0, sticky="ew", pady=4)
        left.columnconfigure(0, weight=1)

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, sticky="ew", pady=4)
        add_btn = ttk.Button(btns, text="Add constraint", command=self._add_constraint)
        add_btn.pack(side="left", padx=4)
        clear_btn = ttk.Button(btns, text="Clear constraints", command=self._clear_constraints)
        clear_btn.pack(side="left", padx=4)

        # Constraints list
        self.constraints_box = tk.Listbox(left, height=10)
        self.constraints_box.grid(row=3, column=0, sticky="nsew", pady=6)
        left.rowconfigure(3, weight=1)

        # Variable picker + Solve
        pick_frame = ttk.Frame(left)
        pick_frame.grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Label(pick_frame, text="Solve for variable:").pack(side="left")
        self.var_combo = ttk.Combobox(pick_frame, values=[], width=10)
        self.var_combo.pack(side="left", padx=6)
        solve_btn = ttk.Button(pick_frame, text="Solve", command=self._solve)
        solve_btn.pack(side="left", padx=6)

        # Reasoning log
        ttk.Label(left, text="Reasoning log:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.log_box = tk.Text(left, height=16, wrap="word")
        self.log_box.grid(row=6, column=0, sticky="nsew")
        left.rowconfigure(6, weight=2)

    def _build_right_panel(self):
        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)

        # Matplotlib figure for graph
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

    def _add_constraint(self):
        expr = self.input_entry.get().strip()
        if not expr:
            return
        try:
            c = self.reasoner.add_constraint(expr)
            self.constraints_box.insert(tk.END, c.raw)
            self._log(f"Parsed constraint: {c.raw} → vars: {', '.join(c.vars) or '(none)'}")
            self.input_entry.delete(0, tk.END)
            self._refresh_var_picker()
            self._refresh_graph(pulse_nodes=[f"eq:{len(self.reasoner.constraints)}"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add constraint:\n{e}")

    def _clear_constraints(self):
        self.reasoner.constraints.clear()
        self.constraints_box.delete(0, tk.END)
        self._log("Cleared all constraints.")
        self._refresh_graph()

    def _refresh_var_picker(self):
        vars_sorted = sorted(self.reasoner.variables.keys())
        self.var_combo["values"] = vars_sorted
        # If empty, clear; else persist selection
        if not self.var_combo.get() and vars_sorted:
            self.var_combo.set(vars_sorted[0])

    def _solve(self):
        target = self.var_combo.get().strip()
        if not target:
            messagebox.showinfo("Info", "Select a variable to solve for (use the dropdown).")
            return
        self._log(f"Attempting to solve for: {target}")
        result = self.reasoner.solve_for(target)
        if result["status"] == "no_constraints":
            self._log("No constraints available. Add some equations first.")
            return
        if result["status"] == "unsolved":
            self._log("No solution found with current constraints.")
            return
        if result["status"] == "error":
            self._log(f"Solver error: {result['message']}")
            return

        branches = result.get("branches", [])
        if not branches:
            self._log("No solution branches returned.")
            return

        # Display all branches
        for i, branch in enumerate(branches, start=1):
            self._log(f"Branch {i}: {branch}")
            residuals = self.reasoner.residuals_for_branch(branch)
            for r in residuals:
                self._log(f"Check: {r['constraint']} | LHS={r['lhs']} RHS={r['rhs']} Residual={r['residual']}")
            # Pulse variable node(s) involved
            pulse_nodes = [f"var:{name}" for name in branch.keys()]
            self._refresh_graph(pulse_nodes=pulse_nodes)

    def _log(self, msg: str):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def _refresh_graph(self, pulse_nodes: Optional[List[str]] = None):
        self.ax.clear()
        G = self.reasoner.build_graph()
        if len(G.nodes) == 0:
            self.ax.text(0.5, 0.5, "No graph yet.\nAdd constraints to visualize reasoning.",
                         ha="center", va="center", fontsize=12)
            self.canvas.draw()
            return

        # Positioning
        pos = nx.spring_layout(G, seed=42)

        # Split nodes by kind
        var_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "var"]
        eq_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "eq"]

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=self.ax, alpha=0.3)

        # Draw variable nodes
        nx.draw_networkx_nodes(
            G, pos, nodelist=var_nodes, node_color="#39c5bb", node_shape="o",
            node_size=800, ax=self.ax, alpha=0.9
        )
        # Draw constraint nodes
        nx.draw_networkx_nodes(
            G, pos, nodelist=eq_nodes, node_color="#ff8a3d", node_shape="s",
            node_size=1000, ax=self.ax, alpha=0.9
        )

        # Labels
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=self.ax)

        # Pulse effect for highlighted nodes
        if pulse_nodes:
            for pn in pulse_nodes:
                if pn in G:
                    x, y = pos[pn]
                    pulse = plt.Circle((x, y), 0.08, color="#ffd54f", fill=False, linewidth=3)
                    self.ax.add_patch(pulse)

        self.ax.set_axis_off()
        self.ax.set_title("Reasoning graph", fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = MissingLinkGUI(root)
    root.geometry("1000x700")
    root.mainloop()


if __name__ == "__main__":
    main()

