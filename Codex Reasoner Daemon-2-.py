import importlib, subprocess, sys, psutil, tkinter as tk
from tkinter import ttk, messagebox
from sympy import symbols, Eq, solve, latex
from sympy.parsing.sympy_parser import parse_expr
from threading import Lock, Thread
import time, winreg

# === AutoLoader ===
def ensure_libs(libs):
    for lib in libs:
        try: importlib.import_module(lib)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
ensure_libs(["sympy", "tkinter", "psutil"])

# === Memory + Mutation Log ===
class MemoryDaemon:
    def __init__(self): self.cases = []
    def store_case(self, constraints, target, solution):
        self.cases.append({"constraints": constraints, "target": target, "solution": solution})

class MutationLog:
    def __init__(self): self.entries = []
    def record(self, action, detail): self.entries.append({"action": action, "detail": detail})
    def get_log(self): return self.entries

# === Codex Reasoner ===
class CodexReasoner:
    def __init__(self, memory, mutation_log):
        self.constraints, self.variables = [], {}
        self.memory, self.mutations = memory, mutation_log
        self.lock = Lock()

    def _parse_variables(self, expr: str):
        tokens = set(filter(str.isalpha, expr.replace(" ", "").replace("=", "")))
        for token in tokens:
            if token not in self.variables:
                self.variables[token] = symbols(token)
        return self.variables

    def add_constraint(self, expr: str):
        try:
            with self.lock:
                left, right = expr.split("=")
                syms = self._parse_variables(expr)
                eq = Eq(parse_expr(left, local_dict=syms), parse_expr(right, local_dict=syms))
                self.constraints.append(eq)
                self.mutations.record("add_constraint", expr)
        except Exception as e:
            self.mutations.record("error", f"Invalid constraint '{expr}': {str(e)}")

    def solve_all(self):
        with self.lock:
            for name in self.variables:
                try:
                    target_sym = self.variables[name]
                    sol = solve(self.constraints, target_sym, dict=True)
                    if sol:
                        self.mutations.record("auto_solve", {name: sol})
                        self.memory.store_case(self.constraints, name, sol)
                except Exception as e:
                    self.mutations.record("solve_error", f"{name}: {str(e)}")

    def visualize_constraints(self): return [latex(eq) for eq in self.constraints]
    def get_threats(self):
        threats = []
        for eq in self.constraints:
            if any(term.name in ["danger", "critical", "breach", "lethal"] for term in eq.free_symbols):
                threats.append(str(eq))
        return threats

    def generate_audit_log(self):
        log = []
        for entry in self.mutations.get_log():
            log.append(f"{entry['action']}: {entry['detail']}")
        for eq in self.constraints:
            log.append(f"Constraint: {eq}")
        return "\n".join(log)

# === Daemons ===
class TelemetryDaemon(Thread):
    def __init__(self, reasoner): super().__init__(daemon=True); self.reasoner = reasoner
    def run(self):
        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            self.reasoner.add_constraint(f"cpu_usage = {cpu}")
            self.reasoner.add_constraint(f"memory_usage = {mem}")
            self.reasoner.solve_all()
            time.sleep(10)

class ResurrectionDetector(Thread):
    def __init__(self, reasoner, watch_services):
        super().__init__(daemon=True)
        self.reasoner, self.watch_services, self.seen = reasoner, watch_services, set()

    def run(self):
        while True:
            active = {p.name() for p in psutil.process_iter()}
            for svc in self.watch_services:
                if svc in active and svc not in self.seen:
                    self.reasoner.add_constraint(f"{svc}_resurrected = 1")
                    self.reasoner.mutations.record("resurrection", f"{svc} reactivated")
                    self.seen.add(svc)
            time.sleep(5)

class RegistryIngestor(Thread):
    def __init__(self, reasoner, keys_to_watch):
        super().__init__(daemon=True)
        self.reasoner, self.keys_to_watch, self.last_seen = reasoner, keys_to_watch, {}

    def run(self):
        while True:
            for reg_path, value_name in self.keys_to_watch:
                try:
                    hive, subkey = self._parse_path(reg_path)
                    with winreg.OpenKey(hive, subkey) as key:
                        value, _ = winreg.QueryValueEx(key, value_name)
                        constraint = f"{value_name} = {int(value)}"
                        if self.last_seen.get((reg_path, value_name)) != value:
                            self.reasoner.add_constraint(constraint)
                            self.reasoner.mutations.record("registry_ingest", f"{reg_path}\\{value_name} = {value}")
                            self.last_seen[(reg_path, value_name)] = value
                except FileNotFoundError:
                    self.reasoner.add_constraint(f"{value_name} = -1")
                    self.reasoner.mutations.record("registry_missing", f"{reg_path}\\{value_name} not found")
            time.sleep(15)

    def _parse_path(self, full_path):
        if full_path.startswith("HKLM\\"): return winreg.HKEY_LOCAL_MACHINE, full_path[5:]
        elif full_path.startswith("HKCU\\"): return winreg.HKEY_CURRENT_USER, full_path[5:]
        else: raise ValueError("Unsupported hive")

class SwarmNode:
    def __init__(self, node_id, reasoner): self.node_id, self.reasoner = node_id, reasoner
    def sync_with(self, other_node):
        for constraint in other_node.reasoner.constraints:
            self.reasoner.add_constraint(str(constraint))
        self.reasoner.mutations.record("swarm_sync", f"{self.node_id} synced with {other_node.node_id}")
        self.sync_threats(other_node)

    def sync_threats(self, other_node):
        threats = other_node.reasoner.get_threats()
        for t in threats:
            self.reasoner.add_constraint(t)
        self.reasoner.mutations.record("swarm_threat_sync", f"{self.node_id} merged threats from {other_node.node_id}")

# === GUI ===
class CodexGUI(tk.Tk):
    def __init__(self, reasoner):
        super().__init__()
        self.title("Codex Reasoner Daemon — Autonomous Devourer Mode")
        self.geometry("1000x750")
        self.reasoner = reasoner
        self._build_tabs()
        self._build_glyph_overlay()

    def _build_tabs(self):
        notebook = ttk.Notebook(self); notebook.pack(expand=True, fill="both")
        self._build_constraint_panel(ttk.Frame(notebook)); notebook.add(notebook.winfo_children()[-1], text="🧠 Constraint Matrix")
        self._build_symbol_panel(ttk.Frame(notebook)); notebook.add(notebook.winfo_children()[-1], text="🔍 Symbol Inspector")
        self._build_mutation_panel(ttk.Frame(notebook)); notebook.add(notebook.winfo_children()[-1], text="🧬 Mutation Log")
        self._build_threat_panel(ttk.Frame(notebook)); notebook.add(notebook.winfo_children()[-1], text="🧿 Threat Matrix")
        self._build_audit_panel(ttk.Frame(notebook)); notebook.add(notebook.winfo_children()[-1], text="🧾 Audit Export")

    def _build_constraint_panel(self, parent):
        ttk.Label(parent, text="Active Constraints (LaTeX)", font=("Consolas", 14)).pack(pady=10)
        self.constraint_text = tk.Text(parent, wrap=tk.WORD, width=100, height=25, font=("Consolas", 10)); self.constraint_text.pack()
        ttk.Button(parent, text="Refresh Constraints", command=self._refresh_constraints).pack(pady=5)

    def _build_symbol_panel(self, parent):
        ttk.Label(parent, text="Symbol Bindings", font=("Consolas", 14)).pack(pady=10)
        self.symbol_text = tk.Text(parent, wrap=tk.WORD, width=100, height=25, font=("Consolas", 10)); self.symbol_text.pack()
        ttk.Button(parent, text="Refresh Symbols", command=self._refresh_symbols).pack(pady=5)

    def _build_mutation_panel(self, parent):
        ttk.Label(parent, text="Mutation History", font=("Consolas", 14)).pack(pady=10)
        self.mutation_text = tk.Text(parent, wrap=tk.WORD, width=100, height=25, font=("Consolas", 10))
        self.mutation_text.pack()
        ttk.Button(parent, text="Refresh Log", command=self._refresh_mutations).pack(pady=5)

    def _build_threat_panel(self, parent):
        ttk.Label(parent, text="Threat Glyphs", font=("Consolas", 14)).pack(pady=10)
        self.threat_text = tk.Text(parent, wrap=tk.WORD, width=100, height=25, font=("Consolas", 10), fg="red")
        self.threat_text.pack()
        ttk.Button(parent, text="Refresh Threats", command=self._refresh_threats).pack(pady=5)

    def _build_audit_panel(self, parent):
        ttk.Label(parent, text="Symbolic Audit Log", font=("Consolas", 14)).pack(pady=10)
        self.audit_text = tk.Text(parent, wrap=tk.WORD, width=100, height=25, font=("Consolas", 10))
        self.audit_text.pack()
        ttk.Button(parent, text="Generate Audit Log", command=self._generate_audit_log).pack(pady=5)

    def _build_glyph_overlay(self):
        self.glyph_label = tk.Label(self, text="", font=("Consolas", 16), fg="white", bg="black")
        self.glyph_label.place(relx=0.5, rely=0.02, anchor="n")

    def trigger_glyph(self, glyph_type):
        if glyph_type == "resurrection":
            self._show_glyph("🧿 Resurrection Detected", "red")
        elif glyph_type == "threat":
            self._show_glyph("⚠️ Threat Activated", "orange")
        elif glyph_type == "sync":
            self._show_glyph("🌐 Swarm Sync Complete", "blue")
        elif glyph_type == "devour":
            self._show_glyph("🧛 Devourer Mode Engaged", "purple")

    def _show_glyph(self, text, color):
        self.glyph_label.config(text=text, fg=color)
        self.after(3000, lambda: self.glyph_label.config(text=""))

    def _refresh_constraints(self):
        self.constraint_text.delete("1.0", tk.END)
        for latex_str in self.reasoner.visualize_constraints():
            self.constraint_text.insert(tk.END, f"{latex_str}\n\n")

    def _refresh_symbols(self):
        self.symbol_text.delete("1.0", tk.END)
        for name, sym in self.reasoner.variables.items():
            self.symbol_text.insert(tk.END, f"{name} → {sym}\n")

    def _refresh_mutations(self):
        self.mutation_text.delete("1.0", tk.END)
        for entry in self.reasoner.mutations.get_log():
            self.mutation_text.insert(tk.END, f"{entry['action']}: {entry['detail']}\n")
            if entry["action"] == "resurrection":
                self.trigger_glyph("resurrection")
            elif entry["action"] == "swarm_threat_sync":
                self.trigger_glyph("sync")
            elif entry["action"] == "add_constraint" and "danger" in str(entry["detail"]):
                self.trigger_glyph("threat")

    def _refresh_threats(self):
        self.threat_text.delete("1.0", tk.END)
        for threat in self.reasoner.get_threats():
            self.threat_text.insert(tk.END, f"⚠️ {threat}\n")
        if self.reasoner.get_threats():
            self.trigger_glyph("devour")

    def _generate_audit_log(self):
        self.audit_text.delete("1.0", tk.END)
        log = self.reasoner.generate_audit_log()
        self.audit_text.insert(tk.END, log)

# === Ritual Launch ===
if __name__ == "__main__":
    memory = MemoryDaemon()
    mutations = MutationLog()
    reasoner = CodexReasoner(memory, mutations)

    # Sample symbolic constraints
    reasoner.add_constraint("telemetry_level + user_consent = 0")
    reasoner.add_constraint("diagnostic_level <= 1")
    reasoner.add_constraint("port_open + no_auth = danger")
    reasoner.add_constraint("threat_score + firewall_breach = critical")
    reasoner.solve_all()

    # Start live telemetry ingestion
    TelemetryDaemon(reasoner).start()

    # Resurrection detection for key services
    ResurrectionDetector(reasoner, ["FeedbackHub.exe", "DiagTrack.exe"]).start()

    # Registry and Group Policy ingestion
    registry_keys = [
        ("HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\DataCollection", "AllowTelemetry"),
        ("HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Windows Error Reporting", "Disabled"),
        ("HKLM\\SOFTWARE\\Microsoft\\Windows\\Windows Error Reporting", "Disabled"),
        ("HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Feedback", "DisableFeedback"),
    ]
    RegistryIngestor(reasoner, registry_keys).start()

    # Swarm sync simulation
    node_b = SwarmNode("NodeB", CodexReasoner(memory, mutations))
    node_b.reasoner.add_constraint("external_threat + internal_breach = critical")
    SwarmNode("NodeA", reasoner).sync_with(node_b)

    # Launch GUI with devourer mode
    app = CodexGUI(reasoner)
    app.trigger_glyph("devour")
    app.mainloop()


        
        
        

