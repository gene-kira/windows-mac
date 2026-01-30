import psutil
import threading
import time
import tkinter as tk
from tkinter import ttk
import os
import json

# ---------- CONFIG ----------
CHECK_INTERVAL = 2.0
LISTS_FILE = "sentinel_lists.json"

ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()

WHITELIST_NAMES = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
}

# ---------- LIST STORAGE ----------

def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE
    if not os.path.exists(LISTS_FILE):
        return
    with open(LISTS_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    ALLOWLIST = set(d.get("allow", []))
    BLOCKLIST = set(d.get("block", []))
    RADIOACTIVE = set(d.get("radioactive", []))

def save_lists():
    with open(LISTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "allow": sorted(ALLOWLIST),
            "block": sorted(BLOCKLIST),
            "radioactive": sorted(RADIOACTIVE),
        }, f, indent=2)

# ---------- BRAIN ----------

class SentinelBrain:
    def classify(self, proc):
        try:
            name = proc.name()
            exe = (proc.exe() or "").replace("\\", "/").lower()
            name_l = name.lower()
            cpu = proc.cpu_percent(0.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        if exe in BLOCKLIST:
            status = "BLOCKED"
        elif exe in ALLOWLIST:
            status = "ALLOW"
        elif exe in RADIOACTIVE:
            status = "RADIOACTIVE"
        elif name_l in WHITELIST_NAMES:
            status = "TRUSTED"
        else:
            status = "SUSPICIOUS"

        return {
            "pid": proc.pid,
            "name": name,
            "cpu": cpu,
            "path": exe,
            "status": status,
        }

# ---------- GUI ----------

class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel – Auto Trust Manager")
        self.brain = SentinelBrain()
        self.running = True

        self._build_ui()
        self._build_context_menu()
        self.refresh_lists_ui()

        threading.Thread(target=self.monitor_loop, daemon=True).start()

    # ---------- UI ----------

    def _build_ui(self):
        pane = ttk.PanedWindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # PROCESS LIST
        left = ttk.Frame(pane)
        pane.add(left, weight=3)

        self.tree = ttk.Treeview(
            left,
            columns=("name", "pid", "cpu", "status", "path"),
            show="headings",
        )
        for c in ("name", "pid", "cpu", "status", "path"):
            self.tree.heading(c, text=c.upper())
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Button-3>", self.show_context_menu)

        # TRUST LISTS
        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        self.allow_box = self._make_box(right, "ALLOWLIST")
        self.block_box = self._make_box(right, "BLOCKLIST")
        self.radio_box = self._make_box(right, "RADIOACTIVE")

        ttk.Button(
            right, text="Remove Selected",
            command=self.remove_selected_from_lists
        ).pack(fill="x", pady=5)

    def _make_box(self, parent, title):
        f = ttk.LabelFrame(parent, text=title)
        f.pack(fill="both", expand=True, padx=5, pady=5)
        lb = tk.Listbox(f)
        lb.pack(fill="both", expand=True)
        return lb

    # ---------- CONTEXT MENU ----------

    def _build_context_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="➕ Add to Allowlist", command=self.add_selected_allow)
        self.menu.add_command(label="⛔ Add to Blocklist", command=self.add_selected_block)
        self.menu.add_command(label="☢ Add to Radioactive", command=self.add_selected_radio)

    def show_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.menu.tk_popup(event.x_root, event.y_root)

    # ---------- LIST HELPERS ----------

    def _selected_path(self):
        sel = self.tree.selection()
        if not sel:
            return None
        v = self.tree.item(sel[0], "values")
        return v[4] if v and v[4] else None

    def add_selected_allow(self):
        p = self._selected_path()
        if p:
            ALLOWLIST.add(p)
            RADIOACTIVE.discard(p)
            save_lists()
            self.refresh_lists_ui()

    def add_selected_block(self):
        p = self._selected_path()
        if p:
            BLOCKLIST.add(p)
            save_lists()
            self.refresh_lists_ui()

    def add_selected_radio(self):
        p = self._selected_path()
        if p:
            RADIOACTIVE.add(p)
            ALLOWLIST.discard(p)
            save_lists()
            self.refresh_lists_ui()

    def refresh_lists_ui(self):
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            lb.delete(0, "end")
            for item in sorted(s):
                lb.insert("end", item)

    def remove_selected_from_lists(self):
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            sel = lb.curselection()
            if sel:
                s.discard(lb.get(sel[0]))
        save_lists()
        self.refresh_lists_ui()

    # ---------- AUTO-FILL LOGIC ----------

    def auto_update_lists(self, info):
        p = info["path"]
        if not p:
            return False

        changed = False

        if info["status"] == "TRUSTED" and p not in ALLOWLIST:
            ALLOWLIST.add(p)
            RADIOACTIVE.discard(p)
            changed = True

        elif info["status"] == "SUSPICIOUS" and p not in RADIOACTIVE:
            RADIOACTIVE.add(p)
            changed = True

        if changed:
            save_lists()
        return changed

    # ---------- MONITOR LOOP ----------

    def monitor_loop(self):
        while self.running:
            updated = False
            rows = []

            for proc in psutil.process_iter():
                try:
                    info = self.brain.classify(proc)
                    if not info:
                        continue

                    if self.auto_update_lists(info):
                        updated = True

                    rows.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue  # skip inaccessible processes

            self.root.after(0, self.update_ui, rows, updated)
            time.sleep(CHECK_INTERVAL)

    def update_ui(self, rows, updated):
        self.tree.delete(*self.tree.get_children())
        for i in rows:
            self.tree.insert("", "end", values=(
                i["name"], i["pid"], f"{i['cpu']:.1f}", i["status"], i["path"]
            ))
        if updated:
            self.refresh_lists_ui()

# ---------- MAIN ----------

def main():
    load_lists()
    root = tk.Tk()
    SentinelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
