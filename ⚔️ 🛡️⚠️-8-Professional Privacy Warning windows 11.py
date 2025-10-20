import os, sys, ctypes, subprocess, winreg, tkinter as tk

def elevate():
    try: import win32com.shell.shell as shell
    except: subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"]); import win32com.shell.shell as shell
    if not ctypes.windll.shell32.IsUserAnAdmin():
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=' '.join([f'"{a}"' for a in sys.argv])); sys.exit()

def set_reg(r, p, n, v, t=winreg.REG_DWORD):
    try: k=winreg.CreateKeyEx(r,p,0,winreg.KEY_SET_VALUE); winreg.SetValueEx(k,n,0,t,v); k.Close(); return True
    except: return False

def del_key(r, p, s): 
    try: winreg.DeleteKey(winreg.OpenKey(r,p,0,winreg.KEY_ALL_ACCESS), s); return True
    except: return False

def purge_files(log):
    for d in [r"%LOCALAPPDATA%\Microsoft\Clipboard", r"%LOCALAPPDATA%\Packages"]:
        path = os.path.expandvars(d)
        for r, _, fs in os.walk(path):
            for f in fs:
                if "ScreenClip" in f or "Clipboard" in r:
                    try: os.remove(os.path.join(r,f)); log(f"🧹 {f}")
                    except: log(f"⚠️ {f}")

def launch():
    root=tk.Tk(); root.title("Codex Privacy Sentinel"); root.geometry("640x860")
    status=tk.StringVar(); status.set("🧠 Monitoring...")
    logbox=tk.Text(root,height=10,width=75,font=("Consolas",9)); logbox.pack()
    def log(msg): status.set(msg); logbox.insert(tk.END,msg+"\n"); logbox.see(tk.END); root.update_idletasks()
    overlay=tk.Label(root,text="🧿 Idle",font=("Consolas",12),fg="darkgreen"); overlay.pack()
    countdown=tk.Label(root,text="",font=("Consolas",10),fg="darkred"); countdown.pack()
    toggles=tk.Frame(root); toggles.pack()
    seen=set()
    def toggle(s):
        k=f"{s['path']}::{s['name']}"
        if k in seen: return
        seen.add(k)
        f=tk.Frame(toggles); f.pack()
        tk.Label(f,text=s["label"],font=("Consolas",10),width=30,anchor="w").pack(side=tk.LEFT)
        for label,val in [("Allow",s["on"]),("Deny",s["off"])]:
            tk.Button(f,text=label,command=lambda v=val: log(f"{s['label']} {label}: {'✅' if set_reg(winreg.HKEY_CURRENT_USER,s['path'],s['name'],v,s.get('type',winreg.REG_DWORD)) else '⚠️'}"),width=10).pack(side=tk.LEFT)

    for s in [
        {"label":"Clipboard History","path":r"Software\\Microsoft\\Clipboard","name":"EnableClipboardHistory","on":1,"off":0},
        {"label":"Screen Clipping","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock","name":"BlockScreenClip","on":0,"off":1},
        {"label":"Tailored Experiences","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy","name":"TailoredExperiencesWithDiagnosticDataEnabled","on":1,"off":0},
        {"label":"Advertising ID","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo","name":"Enabled","on":1,"off":0},
        {"label":"Location Services","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location","name":"Value","on":"Allow","off":"Deny","type":winreg.REG_SZ}
    ]: toggle(s)

    def scan():
        overlay.config(text="🧿 Scanning...",fg="orange")
        purge_files(log)
        base=r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"
        try: k=winreg.OpenKey(winreg.HKEY_CURRENT_USER,base); i=0; found=[]
        except: found=[]
        else:
            while True:
                try: found.append(winreg.EnumKey(k,i)); i+=1
                except: break
            k.Close()
        log(f"🔍 Found {len(found)} telemetry keys.")
        for sub in found: log(f"{'🧹 Purged' if del_key(winreg.HKEY_CURRENT_USER,base,sub) else '⚠️'}: {sub}")
        overlay.config(text="🧿 Idle",fg="darkgreen")

    def discover():
        overlay.config(text="🧿 Discovering...",fg="blue")
        for path in [
            r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy",
            r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo",
            r"Software\\Microsoft\\Clipboard",
            r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\camera"
        ]:
            try: k=winreg.OpenKey(winreg.HKEY_CURRENT_USER,path); i=0
            except: continue
            else:
                while True:
                    try:
                        name,val,typ=winreg.EnumValue(k,i)
                        toggle({"label":f"{path.split('\\\\')[-1]}: {name}","path":path,"name":name,"on":"Allow","off":"Deny","type":winreg.REG_SZ})
                        i+=1
                    except: break
                k.Close()
        overlay.config(text="🧿 Idle",fg="darkgreen"); log("🔍 Key discovery complete.")

    def manual(ms):
        overlay.config(text="⏳ Waiting...",fg="blue"); t=[ms//1000]
        def tick():
            if t[0]>0: countdown.config(text=f"⏳ Manual scan in {t[0]}s"); t[0]-=1; root.after(1000,tick)
            else: countdown.config(text="⏳ Executing..."); scan(); discover()
        tick()

    tk.Label(root,text="Manual Scan Presets:",font=("Consolas",10)).pack()
    f=tk.Frame(root); f.pack()
    for label,ms in [("1min",60000),("1hr",3600000),("1day",86400000)]:
        tk.Button(f,text=label,command=lambda m=ms: manual(m),width=6).pack(side=tk.LEFT,padx=2)

    interval=tk.IntVar(value=60000); auto_t=[interval.get()//1000]
    def auto():
        if auto_t[0]>0: countdown.config(text=f"⏳ Auto scan in {auto_t[0]}s"); auto_t[0]-=1; root.after(1000,auto)
        else: countdown.config(text="⏳ Executing auto scan..."); scan(); discover(); auto_t[0]=interval.get()//1000; auto(); root.after(interval.get(),daemon)

    def daemon(): overlay.config(text="🧿 Auto-Scan",fg="blue"); auto()
    def set_int(ms): interval.set(ms); auto_t[0]=ms//1000; log(f"🔁 Auto-scan set to {ms//1000}s")

    tk.Label(root,text="Auto Scan Interval:",font=("Consolas",10)).pack()
    f=tk.Frame(root); f.pack()
    for label,ms in [("1min",60000),("1hr",3600000),("1day",86400000)]:
        tk.Button(f,text=label,command=lambda m=ms: set_int(m),width=6).pack(side=tk.LEFT,padx=2)

    tk.Button(root,text="Start Auto-Scan Daemon",command=daemon,font=("Consolas",10)).pack(pady=10)
    tk.Button(root,text="Scan for New Privacy Keys",command=discover,font=("Consolas",10)).pack(pady=5)
    tk.Label(root,textvariable=status,wraplength=540,font=("Consolas",10),fg="blue").pack(pady=10)
    root.mainloop()

if __name__=="__main__": elevate(); launch()

