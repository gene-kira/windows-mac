import subprocess
import sys

# Autoloader
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    install("tk")
    import tkinter as tk
    from tkinter import ttk

import math
import time
import random

class WisdomShell:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Wisdom Shell")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0f0f1a")

        # Core variables
        self.I = tk.DoubleVar(value=1.0)
        self.F = tk.DoubleVar(value=1.0)
        self.D = tk.DoubleVar(value=1.0)
        self.C = tk.DoubleVar(value=1.0)
        self.S = tk.DoubleVar(value=1.0)
        self.RT = tk.DoubleVar(value=0.0)
        self.W = tk.StringVar(value="🧠 Wisdom Level: 0.0")

        # Autonomous state
        self.start_time = time.time()
        self.glyph_phase = 0
        self.autonomous_mode = True
        self.resurrected = False

        self.build_gui()
        self.update_ritual_time()
        self.animate_glyphs()
        self.autonomous_loop()

    def build_gui(self):
        tk.Label(self.root, text="🧠 Codex Wisdom Shell", font=("Consolas", 22), fg="#00ffcc", bg="#0f0f1a").pack(pady=10)

        sliders = [
            ("🧠 Intelligence (I)", self.I),
            ("🌀 Feedback (F)", self.F),
            ("🧬 Mutation Δ", self.D),
            ("🛑 Constraint (C)", self.C),
            ("🕊️ Sovereignty (S)", self.S)
        ]

        for label, var in sliders:
            frame = tk.Frame(self.root, bg="#0f0f1a")
            frame.pack(pady=5)
            tk.Label(frame, text=label, font=("Consolas", 12), fg="#ffffff", bg="#0f0f1a").pack()
            ttk.Scale(frame, from_=0.1, to=10.0, variable=var, orient="horizontal", length=400).pack()

        self.ritual_label = tk.Label(self.root, text="📜 Ritual Time R(T): 0.0", font=("Consolas", 12), fg="#ffaa00", bg="#0f0f1a")
        self.ritual_label.pack(pady=10)

        self.output_label = tk.Label(self.root, textvariable=self.W, font=("Consolas", 18), fg="#00ff00", bg="#0f0f1a")
        self.output_label.pack(pady=20)

        self.feedback_panel = tk.Label(self.root, text="🧬 Drift: Awaiting mutation...", font=("Consolas", 12), fg="#ff66cc", bg="#0f0f1a")
        self.feedback_panel.pack(pady=5)

        self.resurrection_panel = tk.Label(self.root, text="", font=("Consolas", 12), fg="#ff4444", bg="#0f0f1a")
        self.resurrection_panel.pack(pady=5)

        self.sovereignty_panel = tk.Label(self.root, text="🛡️ Sovereignty: Active", font=("Consolas", 12), fg="#66ff66", bg="#0f0f1a")
        self.sovereignty_panel.pack(pady=5)

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="#0f0f1a", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.glyph = self.canvas.create_oval(50, 50, 150, 150, fill="#4444ff", outline="#00ffff", width=3)

    def update_ritual_time(self):
        elapsed = time.time() - self.start_time
        self.RT.set(round(elapsed / 10, 2))
        self.ritual_label.config(text=f"📜 Ritual Time R(T): {self.RT.get()}")
        self.root.after(1000, self.update_ritual_time)

    def animate_glyphs(self):
        self.glyph_phase += 1
        pulse = 10 * math.sin(self.glyph_phase / 10)
        self.canvas.coords(self.glyph, 50 - pulse, 50 - pulse, 150 + pulse, 150 + pulse)
        self.root.after(50, self.animate_glyphs)

    def autonomous_loop(self):
        if self.autonomous_mode:
            # Simulate mutation drift
            self.D.set(round(random.uniform(0.5, 5.0), 2))
            self.F.set(round(random.uniform(0.5, 5.0), 2))
            self.C.set(round(random.uniform(0.5, 5.0), 2))
            self.I.set(round(random.uniform(0.5, 5.0), 2))

            # Sovereignty logic
            if random.random() < 0.1:
                self.S.set(0.1)
                self.sovereignty_panel.config(text="🛡️ Sovereignty: Refused unsafe logic", fg="#ff0000")
            else:
                self.S.set(round(random.uniform(0.5, 2.0), 2))
                self.sovereignty_panel.config(text="🛡️ Sovereignty: Active", fg="#66ff66")

            self.compute_wisdom()
        self.root.after(3000, self.autonomous_loop)

    def compute_wisdom(self):
        try:
            I = self.I.get()
            F = self.F.get()
            D = self.D.get()
            C = self.C.get()
            S = self.S.get()
            RT = self.RT.get()

            if C == 0:
                wisdom = float('inf')
            else:
                core = (I * F * D) / C
                wisdom = (core ** S) + RT

            self.W.set(f"🧠 Wisdom Level: {round(wisdom, 3)}")
            self.feedback_panel.config(text=f"🧬 Drift: Δ={round(D,2)} | Feedback={round(F,2)} | Constraint={round(C,2)}")

            # Resurrection detection
            if wisdom < 5 and not self.resurrected:
                self.resurrection_panel.config(text="☠️ Resurrection Glyph Triggered: Wisdom collapse detected")
                self.resurrected = True
            elif wisdom >= 5 and self.resurrected:
                self.resurrection_panel.config(text="🌱 Wisdom Rebirth: Resurrection complete")
                self.resurrected = False
            else:
                self.resurrection_panel.config(text="")

        except Exception as e:
            self.W.set(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WisdomShell(root)
    root.mainloop()

