import socket, threading, time, random, platform, subprocess, sys, ctypes, os, select, requests
from tkinter import Tk, Label, Frame, StringVar, Button, OptionMenu, Text, Entry, END
from datetime import datetime

# === CONFIG ===
ROTATION_OPTIONS = {
    "30 minutes": 1800,
    "1 hour": 3600,
    "6 hours": 21600
}
ROTATION_COUNT = 5
TOTAL_PORTS = list(range(1024, 65535))
open_ports = []
sockets = []
closed_ports = set()
probed_ports = set()
resurrected_ports = set()
firewall_active = False
rotation_interval = ROTATION_OPTIONS["30 minutes"]
allowed_country = "Any"
force_rotate = False
allow_list = set()
disallow_list = set()

# === GUI VARIABLES ===
root = None
port_label = None
status_label = None
interval_var = None
country_var = None
toggle_button = None
log_viewer = None
threat_matrix = None
allow_entry = None
disallow_entry = None
allow_display = None
disallow_display = None
port_overlay = None

# === ELEVATION CHECK ===
def ensure_admin():
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False
    if not is_admin:
        print("[Codex Sentinel] Elevation required. Relaunching as administrator...")
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
        sys.exit()

# === PORT BINDING ===
def bind_port(port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        s.bind(('', port))
        s.listen(5)
        return s
    except:
        return None

# === CLOAKING (Linux only) ===
def cloak_ports():
    if platform.system() == "Linux":
        subprocess.call(["iptables", "-F"])
        subprocess.call(["iptables", "-P", "INPUT", "DROP"])
        for port in open_ports:
            subprocess.call(["iptables", "-A", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "ACCEPT"])

# === ROTATION POOL ===
def get_rotation_pool():
    if allow_list:
        return list(allow_list - disallow_list)
    return [p for p in TOTAL_PORTS if p not in disallow_list]

# === ROTATION ENGINE ===
def rotate_ports():
    global open_ports, sockets, closed_ports, resurrected_ports, force_rotate
    while True:
        if firewall_active and (force_rotate or not open_ports):
            force_rotate = False
            for s in sockets:
                try: s.close()
                except: pass
            closed_ports.update(open_ports)
            pool = get_rotation_pool()
            new_ports = random.sample(pool, ROTATION_COUNT) if len(pool) >= ROTATION_COUNT else []
            sockets = []
            open_ports.clear()
            resurrected_ports.clear()
            for p in new_ports:
                s = bind_port(p)
                if s:
                    sockets.append(s)
                    open_ports.append(p)
                    if p in closed_ports:
                        resurrected_ports.add(p)
                        log_threat(f"☠️ Resurrection: Port {p}")
            cloak_ports()
        time.sleep(rotation_interval)

# === GEOIP LOOKUP ===
def get_country(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=3)
        data = response.json()
        return data.get("country", "Unknown")
    except:
        return "Unknown"

# === THREAT DETECTION ===
def threat_daemon():
    global force_rotate
    while True:
        if firewall_active and sockets:
            try:
                readable, _, _ = select.select(sockets, [], [], 1)
                for s in readable:
                    try:
                        conn, addr = s.accept()
                        ip = addr[0]
                        port = s.getsockname()[1]
                        country = get_country(ip)
                        probed_ports.add(port)
                        threading.Timer(5, lambda: probed_ports.discard(port)).start()
                        if allowed_country != "Any" and country != allowed_country:
                            log_threat(f"🌐 Blocked {ip} from {country} on port {port}")
                            force_rotate = True
                        else:
                            log_threat(f"⚠️ Probe from {ip} ({country}) on port {port}")
                            force_rotate = True
                        conn.close()
                    except Exception as e:
                        log_event(f"Error handling connection: {e}")
            except Exception as e:
                log_event(f"Select error: {e}")
        time.sleep(1)

# === OVERLAY ENGINE ===
def overlay_daemon():
    while True:
        if firewall_active:
            overlay_text = ""
            for port in open_ports:
                if port in probed_ports:
                    glyph = "🔴"
                elif port in resurrected_ports:
                    glyph = "☠️"
                else:
                    glyph = "🟢"
                overlay_text += f"{glyph} {port}  "
            update_overlay(overlay_text)
        time.sleep(1)

def update_overlay(text):
    if port_overlay:
        port_overlay.delete(1.0, END)
        port_overlay.insert(END, text)

# === GUI OVERLAY ===
def launch_gui():
    global root, port_label, status_label, interval_var, toggle_button, log_viewer, threat_matrix
    global country_var, allow_entry, disallow_entry, allow_display, disallow_display, port_overlay

    root = Tk()
    root.title("Codex Sentinel Shell")
    root.geometry("700x800")

    Frame(root).pack()
    port_label = StringVar()
    status_label = StringVar()
    interval_var = StringVar(value="30 minutes")
    country_var = StringVar(value="Any")

    Label(root, textvariable=port_label, font=("Consolas", 12)).pack(pady=5)
    Label(root, textvariable=status_label, font=("Consolas", 10)).pack(pady=5)

    OptionMenu(root, interval_var, *ROTATION_OPTIONS.keys(), command=update_interval).pack(pady=2)
    OptionMenu(root, country_var, "Any", "US", "UK", "DE", "FR", "CN", "RU", command=update_country).pack(pady=2)

    toggle_button = Button(root, text="Start Firewall", command=toggle_firewall)
    toggle_button.pack(pady=5)

    Label(root, text="📜 Live Logs", font=("Consolas", 10)).pack()
    log_viewer = Text(root, height=6, width=80)
    log_viewer.pack(pady=2)

    Label(root, text="🧠 Threat Matrix", font=("Consolas", 10)).pack()
    threat_matrix = Text(root, height=6, width=80)
    threat_matrix.pack(pady=2)

    Label(root, text="🧬 Port Status Overlay", font=("Consolas", 10)).pack()
    port_overlay = Text(root, height=4, width=80)
    port_overlay.pack(pady=2)

    Label(root, text="✅ Allow List", font=("Consolas", 10)).pack()
    allow_entry = Entry(root)
    allow_entry.pack()
    Button(root, text="Add to Allow", command=add_allow).pack()
    Button(root, text="Clear Allow List", command=clear_allow).pack()
    allow_display = Text(root, height=4, width=80)
    allow_display.pack()

    Label(root, text="🚫 Disallow List", font=("Consolas", 10)).pack()
    disallow_entry = Entry(root)
    disallow_entry.pack()
    Button(root, text="Add to Disallow", command=add_disallow).pack()
    Button(root, text="Clear Disallow List", command=clear_disallow).pack()
    disallow_display = Text(root, height=4, width=80)
    disallow_display.pack()

    refresh_gui()
    root.mainloop()

def refresh_gui():
    if port_label and status_label:
        port_label.set(f"Open Ports: {open_ports}\nUpdated: {datetime.now().strftime('%H:%M:%S')}")
        status = "ACTIVE" if firewall_active else "INACTIVE"
        status_label.set(f"Firewall Status: {status}")
        allow_display.delete(1.0, END)
        disallow_display.delete(1.0, END)
        allow_display.insert(END, f"{sorted(allow_list)}")
        disallow_display.insert(END, f"{sorted(disallow_list)}")
    root.after(1000, refresh_gui)

def toggle_firewall():
    global firewall_active
    firewall_active = not firewall_active
    if toggle_button:
        toggle_button.config(text="Stop Firewall" if firewall_active else "Start Firewall")

def update_interval(selection):
    global rotation_interval
    rotation_interval = ROTATION_OPTIONS.get(selection, 1800)
    log_event(f"⏱️ Interval set to {selection} ({rotation_interval}s)")

def update_country(selection):
    global allowed_country
    allowed_country = selection
    log_event(f"🌐 Country set to: {allowed_country}")

def log_event(msg):
    if log_viewer:
        log_viewer.insert(END, f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")
        log_viewer.see(END)

def log_threat(msg):
    if threat_matrix:
        threat_matrix.insert(END, f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")
        threat_matrix.see(END)

def add_allow():
    try:
        port = int(allow_entry.get())
        allow_list.add(port)
        log_event(f"✅ Allowed port {port}")
    except:
        log_event("Invalid port entry for allow list")

def add_disallow():
    try:
        port = int(disallow_entry.get())
        disallow_list.add(port)
        log_event(f"🚫 Disallowed port {port}")
    except:
        log_event("Invalid port entry for disallow list")

def clear_allow():
    allow_list.clear()
    log_event("✅ Allow list cleared")

def clear_disallow():
    disallow_list.clear()
    log_event("🚫 Disallow list cleared")

# === MAIN DAEMON ===
def start_firewall():
    if platform.system() == "Windows":
        ensure_admin()
    threading.Thread(target=rotate_ports, daemon=True).start()
    threading.Thread(target=threat_daemon, daemon=True).start()
    threading.Thread(target=overlay_daemon, daemon=True).start()
    launch_gui()

if __name__ == "__main__":
    start_firewall()


    
    
        

