import tkinter as tk
from tkinter import messagebox
import os
import subprocess

# 🧿 Ritual Functions
def enable_god_mode():
    try:
        os.makedirs("GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}", exist_ok=True)
        messagebox.showinfo("Success", "God Mode activated.")
    except Exception as e:
        messagebox.showerror("Error", f"God Mode error:\n{e}")

def enable_hidden_feature():
    try:
        subprocess.run(
            'reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer" '
            '/v EnableSnapAssistFlyout /t REG_DWORD /d 1 /f',
            shell=True
        )
        messagebox.showinfo("Success", "Hidden feature enabled.")
    except Exception as e:
        messagebox.showerror("Error", f"Registry error:\n{e}")

# 🧿 GUI Ritual Panel
root = tk.Tk()
root.title("🧿 God Mode + Hidden Feature")
root.geometry("300x150")
root.resizable(False, False)

tk.Label(root, text="Activate Features", font=("Segoe UI", 12, "bold")).pack(pady=10)

tk.Button(root, text="Enable God Mode", command=enable_god_mode, width=25).pack(pady=5)
tk.Button(root, text="Enable Hidden Feature", command=enable_hidden_feature, width=25).pack(pady=5)

root.mainloop()

