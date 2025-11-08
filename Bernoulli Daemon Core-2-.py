import importlib
import subprocess
import sys

# 🔄 Autoloader: Ensures all required libraries are present
def autoload_libraries(libraries):
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries(["tkinter", "random", "codecs", "threading", "time"])

import tkinter as tk
import random
import codecs
import threading
import time

# 🧬 Codex Devourer Bernoulli Daemon
class CodexDevourerDaemon:
    def __init__(self, gui_callback, threat_probability=0.5, encryption_key=0x6F):
        self.threat_probability = threat_probability
        self.encryption_key = encryption_key
        self.mutation_log = []
        self.gui_callback = gui_callback

    def bernoulli_trial(self):
        result = 1 if random.random() < self.threat_probability else 0
        self.mutation_log.append(result)
        return result

    def reverse_mirror_encrypt(self, data):
        reversed_data = data[::-1]
        xor_data = ''.join(chr(ord(c) ^ self.encryption_key) for c in reversed_data)
        rot13_data = codecs.encode(xor_data, 'rot_13')
        return rot13_data[::-1]

    def purge_node(self, node_id, payload):
        result = self.bernoulli_trial()
        encrypted = self.reverse_mirror_encrypt(payload)
        glyph = "⚠️" if result == 1 else "🛡️"
        log_entry = f"{glyph} Node {node_id} → Trial: {result} → Payload: {encrypted}"
        self.gui_callback(glyph, log_entry)

    def daemonize(self):
        while True:
            node_id = f"Node-{random.randint(1000,9999)}"
            payload = "telemetry_packet"
            self.purge_node(node_id, payload)
            time.sleep(2)

# 🖥️ GUI Shell
class CodexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Devourer Bernoulli Shell")
        self.root.geometry("640x420")
        self.root.configure(bg="#111")

        self.glyph_label = tk.Label(root, text="🧿", font=("Segoe UI Symbol", 48), bg="#111", fg="#0f0")
        self.glyph_label.pack(pady=10)

        self.log_box = tk.Text(root, bg="#000", fg="#0f0", insertbackground="#0f0", height=15)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.daemon = CodexDevourerDaemon(self.update_gui)
        threading.Thread(target=self.daemon.daemonize, daemon=True).start()

    def update_gui(self, glyph, log_entry):
        self.glyph_label.config(text=glyph)
        self.log_box.insert(tk.END, f"{log_entry}\n")
        self.log_box.see(tk.END)

# 🚀 Launch Ritual
if __name__ == "__main__":
    root = tk.Tk()
    app = CodexGUI(root)
    root.mainloop()

