import tkinter as tk
from tkinter import ttk
import time

# Simulated symbolic data for demonstration
input_traces = [
    {"symbol": "KEYBOARD::a1b2c3_142301", "device": "Keyboard", "timestamp": "16:23:01"},
    {"symbol": "MOUSE::d4e5f6_142305", "device": "Mouse", "timestamp": "16:23:05"},
]

telemetry_streams = [
    {"symbol": "STREAM::CAMERA_9932", "source": "camera_access", "threat": "HIGH", "status": "active"},
    {"symbol": "STREAM::INPUT_4721", "source": "keyboard_input", "threat": "MEDIUM", "status": "active"},
]

ghost_devices = [
    {"device": "Keyboard", "masked_id": "keyboard_8f3a2c1d", "status": "Authorized"},
    {"device": "Webcam", "masked_id": "webcam_9a7b4e2f", "status": "Authorized"},
]

def allow_action(symbol):
    print(f"[ALLOW] {symbol}")

def deny_action(symbol):
    print(f"[DENY] {symbol}")

def build_table(frame, data, columns, label):
    ttk.Label(frame, text=label, font=("Consolas", 12, "bold")).pack(anchor="w", pady=(10, 0))
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=5)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)
    for row in data:
        tree.insert("", "end", values=[row[col.lower()] for col in columns])
    tree.pack(fill="x", padx=5, pady=5)

    # Add Allow/Deny buttons
    def on_select(event):
        selected = tree.focus()
        if selected:
            values = tree.item(selected)["values"]
            symbol = values[0]
            action_frame = tk.Frame(frame)
            action_frame.pack()
            tk.Button(action_frame, text="Allow", command=lambda: allow_action(symbol)).pack(side="left", padx=5)
            tk.Button(action_frame, text="Deny", command=lambda: deny_action(symbol)).pack(side="left", padx=5)

    tree.bind("<<TreeviewSelect>>", on_select)

# === GUI Setup ===
root = tk.Tk()
root.title("Codex Phantom: Symbolic Overlay")
root.geometry("800x600")

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", font=("Consolas", 10), rowheight=24)
style.configure("TButton", font=("Consolas", 10))

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# === Panels ===
trace_frame = ttk.Frame(notebook)
threat_frame = ttk.Frame(notebook)
ghost_frame = ttk.Frame(notebook)

notebook.add(trace_frame, text="Input Trace Registry")
notebook.add(threat_frame, text="Threat Matrix Overlay")
notebook.add(ghost_frame, text="Ghost Mode Clearance")

build_table(trace_frame, input_traces, ["Symbol", "Device", "Timestamp"], "Live Input Traces")
build_table(threat_frame, telemetry_streams, ["Symbol", "Source", "Threat", "Status"], "Telemetry Streams")
build_table(ghost_frame, ghost_devices, ["Device", "Masked_id", "Status"], "Masked Devices")

root.mainloop()

