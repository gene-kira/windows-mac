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

autoload_libraries(["tkinter", "random", "codecs", "threading", "time", "base64"])

import tkinter as tk
import random
import codecs
import threading
import time
import base64

# 🧬 Codex Autonomous Cipher Daemon
class CodexCipherDaemon:
    def __init__(self, gui_callback, base_threat_probability=0.5):
        self.base_threat_probability = base_threat_probability
        self.mutation_log = []
        self.gui_callback = gui_callback

    def adaptive_bernoulli(self, entropy):
        p = min(1.0, max(0.0, self.base_threat_probability + (entropy - 0.5) * 0.3))
        result = 1 if random.random() < p else 0
        self.mutation_log.append(result)
        return result

    def mutate_cipher(self, data, key=None):
        key = key or random.randint(0x10, 0xFF)
        reversed_data = data[::-1]
        xor_data = ''.join(chr(ord(c) ^ key) for c in reversed_data)
        rot13_data = codecs.encode(xor_data, 'rot_13')
        base64_data = base64.b64encode(rot13_data.encode()).decode()
        final_scramble = base64_data[::-1]
        return final_scramble, key

    def purge_node(self, node_id, payload):
        entropy = random.random()
        result = self.adaptive_bernoulli(entropy)
        encrypted, key = self.mutate_cipher(payload)
        glyph = "⚠️" if result == 1 else "🛡️"
        if entropy > 0.9:
            glyph = "🜏"
        elif entropy < 0.1:
            glyph = "💀"
        log_entry = f"{glyph} Node {node_id} → Trial: {result} → Key: {hex(key)} → Payload: {encrypted}"
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
        self.root.title("Codex Autonomous Cipher Shell")
        self.root.geometry("700x440")
        self.root.configure(bg="#111")

        self.glyph_label = tk.Label(root, text="🧿", font=("Segoe UI Symbol", 48), bg="#111", fg="#0f0")
        self.glyph_label.pack(pady=10)

        self.log_box = tk.Text(root, bg="#000", fg="#0f0", insertbackground="#0f0", height=15)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.daemon = CodexCipherDaemon(self.update_gui)
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

