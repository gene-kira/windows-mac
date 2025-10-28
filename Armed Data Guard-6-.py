import tkinter as tk, hashlib, json, os, time, threading, psutil, sys

# 🔐 Elevation
try:
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()
except: pass

MEMORY_FILE, PURGE_LOG, purged_ids, registry = "settings.json", [], set(), {}

# 🌍 Full ISO country list (truncated for brevity — expand as needed)
COUNTRIES = {
    "US":"United States","DE":"Germany","CN":"China","GB":"United Kingdom","FR":"France","JP":"Japan","IN":"India",
    "RU":"Russia","KR":"South Korea","BR":"Brazil","CA":"Canada","AU":"Australia","IT":"Italy","ES":"Spain",
    "MX":"Mexico","ZA":"South Africa","NG":"Nigeria","EG":"Egypt","TR":"Turkey","SA":"Saudi Arabia","IR":"Iran",
    "ID":"Indonesia","PK":"Pakistan","BD":"Bangladesh","PL":"Poland","NL":"Netherlands","SE":"Sweden","CH":"Switzerland",
    "UA":"Ukraine","AR":"Argentina","CO":"Colombia","TH":"Thailand","MY":"Malaysia","PH":"Philippines","VN":"Vietnam",
    "NO":"Norway","FI":"Finland","DK":"Denmark","BE":"Belgium","AT":"Austria","GR":"Greece","IL":"Israel","NZ":"New Zealand"
}

SYMBOLS = {"allowed":"⟁","blocked":"⛔","hijacked":"☠️","unauthorized":"🕳️","resurrected":"♻️","purged":"⚫","pending":"◌"}

class Glyph:
    def __init__(s,id,oc,eip,cip):
        s.id,s.oc,s.eip,s.cip = id,oc,eip,cip
        s.mut = hashlib.sha256(f"{id}:{oc}:{eip}".encode()).hexdigest()[:6]
        s.status = "resurrected" if id in purged_ids else "pending"
        s.symbol = SYMBOLS[s.status]; s.color = "#888"
        s.payload = None if s.status=="resurrected" else hashlib.sha256(f"{id}:{oc}".encode()).hexdigest()
        if s.status=="resurrected": PURGE_LOG.append(f"{s.id} resurrected")

    def purge(s,reason):
        s.status = reason; s.symbol = SYMBOLS.get(reason,"⚫"); s.payload = None
        purged_ids.add(s.id)
        registry[s.id] = {"mutation":s.mut,"origin":s.oc,"reason":reason,"time":time.strftime('%H:%M:%S')}
        PURGE_LOG.append(f"{s.id} purged: {reason}")

    def validate(s):
        if s.eip != s.cip: s.purge("unauthorized")
        elif s.oc not in COUNTRIES: s.purge("blocked")
        s.color = {"allowed":"#0F0","blocked":"#F00","purged":"#444","hijacked":"#F69","unauthorized":"#0FF","resurrected":"#6F6"}.get(s.status,"#888")

class Filter:
    def __init__(s): s.allowed,s.blocked = set(),set(); s.load()
    def allow(s,c): s.allowed.add(c); s.blocked.discard(c); s.save()
    def block(s,c): s.blocked.add(c); s.allowed.discard(c); s.save()
    def apply(s,glyphs):
        for g in glyphs:
            g.validate()
            if g.status not in ["unauthorized","resurrected"]:
                g.status = "allowed" if g.oc in s.allowed else "blocked" if g.oc in s.blocked else "pending"
                g.symbol = SYMBOLS[g.status]
                g.validate()
    def save(s): json.dump({"allowed":list(s.allowed),"blocked":list(s.blocked)}, open(MEMORY_FILE,"w"))
    def load(s):
        if os.path.exists(MEMORY_FILE):
            try: d=json.load(open(MEMORY_FILE)); s.allowed,s.blocked=set(d.get("allowed",[])),set(d.get("blocked",[]))
            except: pass

glyphs = []
def ingest():
    seen = set()
    while True:
        for c in psutil.net_connections(kind='inet'):
            if c.raddr and c.status=="ESTABLISHED":
                k=(c.pid,c.laddr.ip,c.raddr.ip,c.raddr.port)
                if k in seen: continue
                seen.add(k)
                glyphs.append(Glyph(f"{c.pid}:{c.raddr.port}","??",c.raddr.ip,c.laddr.ip))
        time.sleep(1)

def gui():
    root = tk.Tk(); root.title("Codex Shell Ω"); root.configure(bg="#0A0A0A")
    canvas = tk.Canvas(root, width=1000, height=700, bg="#0A0A0A"); canvas.pack(fill=tk.BOTH, expand=True)

    def draw():
        canvas.delete("all"); f.apply(glyphs)
        for i,g in enumerate(glyphs[-100:]):
            x,y=40+(i%10)*100,40+(i//10)*80
            canvas.create_text(x,y,text=g.symbol,fill=g.color,font=("Consolas",20,"bold"))
            canvas.create_text(x,y+25,text=g.mut,fill=g.color,font=("Consolas",10))
            canvas.create_rectangle(x-30,y-30,x+30,y+50,outline=g.color)
        root.after(1000,draw)

    def log():
        win=tk.Toplevel(root); win.title("Purge Log")
        txt=tk.Text(win,height=40,width=100,bg="#0A0A0A",fg="#C0C0C0",font=("Consolas",10)); txt.pack()
        def update(): txt.delete("1.0",tk.END); txt.insert(tk.END,"\n".join(PURGE_LOG[-50:])); win.after(2000,update)
        update()

    def country():
        win=tk.Toplevel(root); win.title("Country Filter")
        allbox,allowbox,blockbox=[tk.Listbox(win,height=30,width=w,bg="#111",fg="#C0C0C0") for w in(40,30,30)]
        for b in [allbox,allowbox,blockbox]: b.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        for name in sorted(COUNTRIES.values()): allbox.insert(tk.END,name)
        def refresh():
            allowbox.delete(0,tk.END); blockbox.delete(0,tk.END)
            for code,name in COUNTRIES.items():
                if code in f.allowed: allowbox.insert(tk.END,name)
                elif code in f.blocked: blockbox.insert(tk.END,name)
        def allow(): sel=allbox.curselection(); name=allbox.get(sel[0]); code=[k for k,v in COUNTRIES.items() if v==name][0]; f.allow(code); refresh()
        def block(): sel=allbox.curselection(); name=allbox.get(sel[0]); code=[k for k,v in COUNTRIES.items() if v==name][0]; f.block(code); refresh()
        tk.Button(win,text="Allow →",command=allow,bg="#222",fg="#0F0").pack()
        tk.Button(win,text="Disallow →",command=block,bg="#222",fg="#F00").pack()
        refresh()

    tk.Button(root,text="Log",command=log,bg="#222",fg="#C0C0C0").pack(side=tk.LEFT)
    tk.Button(root,text="Filter",command=country,bg="#222",fg="#C0C0C0").pack(side=tk.RIGHT)
    draw(); root.mainloop()

if __name__=="__main__":
    f=Filter()
    threading.Thread(target=ingest,daemon=True).start()
    gui()

