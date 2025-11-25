import importlib, subprocess, sys, os, platform, ctypes, hashlib, threading, requests
import http.server, socketserver
import tkinter as tk
from tkinter import scrolledtext
from collections import OrderedDict
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# === Autoloader ===
REQUIRED_LIBS = ["requests", "tkinter", "matplotlib"]
def ensure_libs():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[Autoloader] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
ensure_libs()

# === Auto-Elevation ===
def ensure_admin():
    try:
        if platform.system() == "Windows":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                print("[Codex Sentinel] Elevation required. Relaunching as administrator...")
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, f'"{script}" {params}', None, 1
                )
                sys.exit(0)
        else:
            if os.getuid() != 0:
                print("[Codex Sentinel] Elevation required. Relaunching with sudo...")
                subprocess.check_call(["sudo", sys.executable] + sys.argv)
                sys.exit(0)
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit(1)

ensure_admin()

# === Config ===
CACHE_DIR = "/mnt/nas_cache"
ADMIN_CLIENTS = ["192.168.1.100", "192.168.1.101"]
LAN_IFACE = "eth0"
PROXY_HTTP = 8080
PROXY_HTTPS = 8443

memory_cache = OrderedDict()
log_buffer = []
status_flags = {"elevated": True, "firewall": False}
stats = {"hits": 0, "misses": 0, "bandwidth_saved": 0}

def log(msg):
    print(msg)
    log_buffer.append(msg)
    if len(log_buffer) > 200:
        log_buffer.pop(0)

# === Firewall Integration ===
def setup_firewall():
    try:
        if platform.system() == "Windows":
            # Windows portproxy rules
            subprocess.run([
                "netsh", "interface", "portproxy", "add", "v4tov4",
                "listenport=80", "listenaddress=0.0.0.0",
                "connectport=8080", "connectaddress=127.0.0.1"
            ], check=True)

            subprocess.run([
                "netsh", "interface", "portproxy", "add", "v4tov4",
                "listenport=443", "listenaddress=0.0.0.0",
                "connectport=8443", "connectaddress=127.0.0.1"
            ], check=True)

            log("Windows firewall portproxy rules applied successfully.")
            status_flags["firewall"] = True

        else:
            # Linux iptables rules
            subprocess.run([
                "iptables", "-t", "nat", "-A", "PREROUTING",
                "-i", LAN_IFACE, "-p", "tcp", "--dport", "80",
                "-j", "REDIRECT", "--to-port", str(PROXY_HTTP)
            ], check=True)

            subprocess.run([
                "iptables", "-t", "nat", "-A", "PREROUTING",
                "-i", LAN_IFACE, "-p", "tcp", "--dport", "443",
                "-j", "REDIRECT", "--to-port", str(PROXY_HTTPS)
            ], check=True)

            log("iptables rules applied successfully.")
            status_flags["firewall"] = True

    except Exception as e:
        log(f"Error setting firewall rules: {e}")
        status_flags["firewall"] = False

# === Cache Functions ===
def cache_key(url): return hashlib.sha256(url.encode()).hexdigest()

def get_from_cache(url):
    key = cache_key(url)
    if key in memory_cache:
        log(f"Memory cache hit: {url}")
        stats["hits"] += 1
        return memory_cache[key]
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        log(f"Disk cache hit: {url}")
        stats["hits"] += 1
        with open(path, "rb") as f: return f.read()
    return None

def store_in_cache(url, data):
    key = cache_key(url)
    memory_cache[key] = data
    if len(memory_cache) > 1000: memory_cache.popitem(last=False)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, key), "wb") as f: f.write(data)
    stats["bandwidth_saved"] += len(data)

# === Proxy Handler ===
class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        client_ip = self.client_address[0]
        url = self.path

        if url.startswith("https://") and client_ip not in ADMIN_CLIENTS:
            self.send_response(403); self.end_headers()
            self.wfile.write(b"SSL bump not allowed for this client")
            log(f"Blocked HTTPS for {client_ip}")
            return

        data = get_from_cache(url)
        if data is None:
            log(f"Cache miss: {url}")
            stats["misses"] += 1
            try:
                resp = requests.get(url, verify=False)
                data = resp.content
                store_in_cache(url, data)
            except Exception as e:
                self.send_response(502); self.end_headers()
                self.wfile.write(f"Error fetching {url}: {e}".encode())
                log(f"Error fetching {url}: {e}")
                return

        self.send_response(200); self.end_headers()
        self.wfile.write(data)
        log(f"Served {url} to {client_ip}")

# === GUI Dashboard ===
def run_gui():
    root = tk.Tk()
    root.title("NAS Proxy Dashboard")

    status_label = tk.Label(root, text="", fg="blue", font=("Arial", 12))
    status_label.pack()

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
    text_area.pack()

    fig = Figure(figsize=(5,3), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    def update_gui():
        # Status panel
        status_text = f"Elevation: {'OK' if status_flags['elevated'] else 'FAILED'} | " \
                      f"Firewall: {'OK' if status_flags['firewall'] else 'FAILED'}"
        status_label.config(text=status_text)

        # Log panel
        text_area.delete(1.0, tk.END)
        for line in log_buffer: text_area.insert(tk.END, line + "\n")

        # Chart panel
        ax.clear()
        labels = ["Hits", "Misses"]
        values = [stats["hits"], stats["misses"]]
        ax.bar(labels, values, color=["green","red"])
        ax.set_title(f"Cache Stats | Bandwidth Saved: {stats['bandwidth_saved']/1024:.2f} KB")
        canvas.draw()

        root.after(2000, update_gui)

    update_gui()
    root.mainloop()

# === Main ===
if __name__ == "__main__":
    setup_firewall()

    def run_proxy():
        with socketserver.ThreadingTCPServer(("", PROXY_HTTP), ProxyHandler) as httpd:
            log(f"Proxy running on port {PROXY_HTTP}")
            httpd.serve_forever()

    threading.Thread(target=run_proxy, daemon=True).start()
    run_gui()

