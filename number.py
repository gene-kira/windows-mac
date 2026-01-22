#!/usr/bin/env python3
"""
Universal Formula Brain Chip Emulator (Windows-friendly)

- Autoloads required libraries
- Sympy-based universal formula generator
- Tkinter GUI
- Animated "brain" canvas with floating numbers
"""

import importlib
import sys
import random
import textwrap

# ---------- Autoloader for libraries ----------

def auto_import(module_name, friendly_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        label = friendly_name or module_name
        print(f"[FATAL] Missing required library: {label}")
        print(f"Install it with:  pip install {module_name}")
        sys.exit(1)

tk = auto_import("tkinter", "Tkinter (GUI)")
sympy = auto_import("sympy", "Sympy (symbolic math)")
from tkinter import ttk

symbols = sympy.symbols
sin = sympy.sin
cos = sympy.cos
exp = sympy.exp
log = sympy.log
sqrt = sympy.sqrt
Integer = sympy.Integer


# ---------- Universal Formula "Chip" Core ----------

class UniversalFormulaChip:
    """
    Conceptual emulator of the 'master formula chip':
    M(i) = Eval(Decode(U(i)))
    """

    def __init__(self):
        self.x, self.y, self.z = symbols("x y z")

    def decode(self, i: int):
        if i <= 0:
            i = 1

        op_selector = i % 7
        a = Integer(i % 11 - 5)
        b = Integer((i // 7) % 13 - 6)
        c = Integer((i // 49) % 17 - 8)

        x, y, z = self.x, self.y, self.z

        if op_selector == 0:
            expr = a * x + b * y + c
        elif op_selector == 1:
            expr = a * x**2 + b * y + c * z
        elif op_selector == 2:
            expr = sin(a * x) + cos(b * y) + c
        elif op_selector == 3:
            expr = exp(a * x + b * y) + c
        elif op_selector == 4:
            expr = log(abs(a * x + b)) + c
        elif op_selector == 5:
            expr = sqrt(abs(a * x**2 + b * y + c))
        else:
            expr = (a * x + b) / (c * y + 1)

        return sympy.simplify(expr)

    def evaluate(self, expr, substitutions=None):
        if substitutions is None:
            return sympy.simplify(expr)
        return expr.subs(substitutions)

    def master_formula(self, i: int):
        expr = self.decode(i)
        value = self.evaluate(expr)
        return expr, value


# ---------- Floating "Thought" Numbers ----------

class FloatingNumber:
    def __init__(self, canvas, text, x, y, dx, dy, color):
        self.canvas = canvas
        self.text = text
        self.dx = dx
        self.dy = dy
        self.item = canvas.create_text(x, y, text=text, fill=color, font=("Consolas", 10, "bold"))

    def move(self, width, height):
        x, y = self.canvas.coords(self.item)
        x += self.dx
        y += self.dy

        if x < 0 or x > width:
            self.dx = -self.dx
        if y < 0 or y > height:
            self.dy = -self.dy

        self.canvas.coords(self.item, x, y)


# ---------- GUI Layer ----------

class UniversalFormulaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Formula Brain Chip")

        self.chip = UniversalFormulaChip()
        self.current_index = 1

        self.floating_numbers = []
        self.animation_running = True

        self._build_layout()
        self._init_floating_numbers()
        self.animate()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Top: index + controls
        index_frame = ttk.Frame(main)
        index_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))

        ttk.Label(index_frame, text="Formula index (i):").grid(row=0, column=0, sticky="w")

        self.index_var = tk.StringVar(value=str(self.current_index))
        self.index_entry = ttk.Entry(index_frame, textvariable=self.index_var, width=10)
        self.index_entry.grid(row=0, column=1, padx=(5, 5))

        self.btn_prev = ttk.Button(index_frame, text="◀ Prev", command=self.prev_formula)
        self.btn_prev.grid(row=0, column=2, padx=(5, 5))

        self.btn_next = ttk.Button(index_frame, text="Next ▶", command=self.next_formula)
        self.btn_next.grid(row=0, column=3, padx=(5, 5))

        self.btn_go = ttk.Button(index_frame, text="Go", command=self.go_to_index)
        self.btn_go.grid(row=0, column=4, padx=(5, 5))

        # Middle: brain canvas + formula panels
        middle = ttk.Frame(main)
        middle.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        # Left: animated brain canvas
        brain_frame = ttk.LabelFrame(middle, text="Brain Activity", padding=5)
        brain_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(brain_frame, width=320, height=240, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        brain_frame.rowconfigure(0, weight=1)
        brain_frame.columnconfigure(0, weight=1)

        # Right: formula displays
        right_frame = ttk.Frame(middle)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        formula_frame = ttk.LabelFrame(right_frame, text="Decoded Formula (Sympy)", padding=5)
        formula_frame.grid(row=0, column=0, sticky="nsew")

        self.formula_text = tk.Text(formula_frame, height=5, wrap="word")
        self.formula_text.grid(row=0, column=0, sticky="nsew")
        formula_frame.rowconfigure(0, weight=1)
        formula_frame.columnconfigure(0, weight=1)

        pretty_frame = ttk.LabelFrame(right_frame, text="Pretty / Math Representation", padding=5)
        pretty_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        self.pretty_text = tk.Text(pretty_frame, height=5, wrap="word")
        self.pretty_text.grid(row=0, column=0, sticky="nsew")
        pretty_frame.rowconfigure(0, weight=1)
        pretty_frame.columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status_label.grid(row=2, column=0, sticky="we", pady=(10, 0))

        self.update_formula_display()

    def _init_floating_numbers(self):
        width = int(self.canvas["width"])
        height = int(self.canvas["height"])

        colors = ["#00FFAA", "#00CCFF", "#FFDD00", "#FF66CC", "#FF4444", "#88FF44"]
        for _ in range(40):
            x = random.randint(0, width)
            y = random.randint(0, height)
            dx = random.choice([-1, -0.5, 0.5, 1])
            dy = random.choice([-1, -0.5, 0.5, 1])
            text = str(random.randint(-99, 999))
            color = random.choice(colors)
            self.floating_numbers.append(FloatingNumber(self.canvas, text, x, y, dx, dy, color))

    def animate(self):
        if not self.animation_running:
            return

        width = int(self.canvas["width"])
        height = int(self.canvas["height"])

        for fn in self.floating_numbers:
            fn.move(width, height)

        # Occasionally change some numbers to reflect "new thoughts"
        if random.random() < 0.1:
            fn = random.choice(self.floating_numbers)
            new_val = random.randint(-999, 9999)
            self.canvas.itemconfig(fn.item, text=str(new_val))

        self.root.after(40, self.animate)  # ~25 FPS

    def _safe_get_index(self):
        try:
            i = int(self.index_var.get())
            if i <= 0:
                i = 1
            return i
        except ValueError:
            return 1

    def prev_formula(self):
        self.current_index = max(1, self._safe_get_index() - 1)
        self.index_var.set(str(self.current_index))
        self.update_formula_display()

    def next_formula(self):
        self.current_index = self._safe_get_index() + 1
        self.index_var.set(str(self.current_index))
        self.update_formula_display()

    def go_to_index(self):
        self.current_index = self._safe_get_index()
        self.update_formula_display()

    def update_formula_display(self):
        i = self.current_index
        try:
            expr, value = self.chip.master_formula(i)
            self._set_text(self.formula_text, str(expr))

            pretty = sympy.pretty(expr)
            pretty_wrapped = textwrap.dedent(pretty)
            self._set_text(self.pretty_text, pretty_wrapped)

            self.status_var.set(f"Decoded formula for i = {i}")
        except Exception as e:
            self._set_text(self.formula_text, f"Error: {e}")
            self._set_text(self.pretty_text, "")
            self.status_var.set(f"Error decoding formula for i = {i}")

    def _set_text(self, widget, content: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")


def main():
    root = tk.Tk()
    app = UniversalFormulaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

