#!/usr/bin/env python3
"""
transcode_gui.py

Single-file GUI wrapper for video analysis, system probe, optional kernel build,
and short encoder benchmarking. Uses ffmpeg/ffprobe on the system.

Usage:
  python3 transcode_gui.py
"""
from __future__ import annotations
import argparse
import subprocess
import json
import shutil
import sys
import os
import tempfile
import time
import ctypes
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# -----------------------
# Configuration
# -----------------------
SYSTEM_CMDS: List[str] = ["ffmpeg", "ffprobe", "g++"]
TEST_DURATION_DEFAULT = 10
ENCODER_CANDIDATES = ["h264_nvenc", "hevc_nvenc", "h264_vaapi", "hevc_vaapi", "h264_qsv", "libx264", "libx265"]

# -----------------------
# Utilities
# -----------------------
def run(cmd, capture: bool = False, check: bool = True, shell: bool = False, **kwargs):
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              check=check, shell=shell, **kwargs)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              check=check, shell=True, **kwargs)
    return proc

def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def ffprobe_info(path: str) -> Dict:
    try:
        p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], capture=True)
        out = p.stdout.decode('utf-8', errors='ignore')
        return json.loads(out)
    except Exception:
        return {}

def analyze_streams(info: Dict) -> Optional[Dict]:
    video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        return None
    v = video_streams[0]
    def safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default
    return {
        "codec": v.get("codec_name"),
        "codec_long": v.get("codec_long_name"),
        "width": safe_int(v.get("width")),
        "height": safe_int(v.get("height")),
        "pix_fmt": v.get("pix_fmt"),
        "r_frame_rate": v.get("r_frame_rate"),
        "bit_rate": safe_int(v.get("bit_rate")),
        "duration": safe_float(v.get("duration"))
    }

def check_hw_accel() -> Dict[str, bool]:
    hw = {"nvidia": False, "nvenc": False, "vaapi": False, "intel_qsv": False, "nvdec": False}
    if which("nvidia-smi"):
        hw["nvidia"] = True
    if which("ffmpeg"):
        try:
            p = run(["ffmpeg", "-hide_banner", "-encoders"], capture=True)
            encs = p.stdout.decode('utf-8', errors='ignore')
            if "h264_nvenc" in encs or "hevc_nvenc" in encs:
                hw["nvenc"] = True
            if "h264_vaapi" in encs or "hevc_vaapi" in encs:
                hw["vaapi"] = True
            if "h264_qsv" in encs or "hevc_qsv" in encs:
                hw["intel_qsv"] = True
        except Exception:
            pass
        try:
            p2 = run(["ffmpeg", "-hide_banner", "-decoders"], capture=True)
            decs = p2.stdout.decode('utf-8', errors='ignore')
            if "h264_cuvid" in decs or "hevc_cuvid" in decs:
                hw["nvdec"] = True
        except Exception:
            pass
    return hw

def detect_cuda_toolchain() -> bool:
    return which("nvcc") and which("nvidia-smi")

# -----------------------
# Build shared lib (simple)
# -----------------------
CPP_CPU_SOURCE = r'''
#include <cstdint>
#include <cstring>
extern "C" void process_frame_cpu(const uint8_t* in, uint8_t* out, int w, int h) {
    int stride = w * 3;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * stride + x * 3;
            int r = in[idx+0], g = in[idx+1], b = in[idx+2];
            int rc = (int)(1.05f * r);
            int gc = (int)(1.05f * g);
            int bc = (int)(1.05f * b);
            out[idx+0] = (uint8_t) (rc > 255 ? 255 : rc);
            out[idx+1] = (uint8_t) (gc > 255 ? 255 : gc);
            out[idx+2] = (uint8_t) (bc > 255 ? 255 : bc);
        }
    }
}
'''

CUDA_SOURCE = r'''
extern "C" {
#include <stdint.h>
}
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void process_kernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = (y * w + x) * 3;
    unsigned char r = in[idx+0], g = in[idx+1], b = in[idx+2];
    int rc = (int)(1.05f * r);
    int gc = (int)(1.05f * g);
    int bc = (int)(1.05f * b);
    out[idx+0] = rc > 255 ? 255 : rc;
    out[idx+1] = gc > 255 ? 255 : gc;
    out[idx+2] = bc > 255 ? 255 : bc;
}

extern "C" void process_frame_cuda(const unsigned char* in, unsigned char* out, int w, int h) {
    size_t bytes = (size_t)w * h * 3;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaError_t e;
    e = cudaMalloc((void**)&d_in, bytes); if (e != cudaSuccess) { fprintf(stderr,"cudaMalloc in failed\\n"); return; }
    e = cudaMalloc((void**)&d_out, bytes); if (e != cudaSuccess) { cudaFree(d_in); fprintf(stderr,"cudaMalloc out failed\\n"); return; }
    e = cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice); if (e != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); fprintf(stderr,"cudaMemcpy H2D failed\\n"); return; }
    dim3 block(16,16); dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    process_kernel<<<grid, block>>>(d_in, d_out, w, h);
    cudaDeviceSynchronize();
    e = cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}
'''

def build_shared_lib(tmpdir: str, use_cuda: bool, log_fn) -> Tuple[str, str]:
    libname = "libvidproc"
    so_name = os.path.join(tmpdir, libname + (".so" if os.name != "nt" else ".dll"))
    if use_cuda:
        cu_file = os.path.join(tmpdir, "proc.cu")
        with open(cu_file, "w") as f:
            f.write(CUDA_SOURCE)
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc not found")
        cmd = f"{nvcc} -std=c++11 -O3 --compiler-options '-fPIC' -shared {cu_file} -o {so_name} -lcudart"
        log_fn(f"Running: {cmd}")
        run(cmd, shell=True)
        return so_name, "cuda"
    else:
        cpp_file = os.path.join(tmpdir, "proc.cpp")
        with open(cpp_file, "w") as f:
            f.write(CPP_CPU_SOURCE)
        gpp = shutil.which("g++") or shutil.which("clang++")
        if not gpp:
            raise RuntimeError("No C++ compiler found")
        cmd = [gpp, "-std=c++11", "-O3", "-fPIC", "-shared", cpp_file, "-o", so_name]
        log_fn(f"Running: {' '.join(cmd)}")
        run(cmd)
        return so_name, "cpu"

class ProcLib:
    def __init__(self, path: str, mode: str):
        self.lib = ctypes.CDLL(path)
        self.mode = mode
        if mode == "cuda":
            self.fn = self.lib.process_frame_cuda
        else:
            self.fn = self.lib.process_frame_cpu
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
        self.fn.restype = None

    def process(self, in_bytes: bytes, out_buf: bytearray, w: int, h: int):
        in_arr = (ctypes.c_ubyte * len(in_bytes)).from_buffer_copy(in_bytes)
        out_arr = (ctypes.c_ubyte * len(out_buf)).from_buffer(out_buf)
        self.fn(in_arr, out_arr, int(w), int(h))

# -----------------------
# FFmpeg helpers
# -----------------------
def run_test_transcode(input_path: str, output_path: str, encoder: str, test_seconds: int, scale: Optional[Tuple[int,int]], log_fn) -> Dict:
    vf = []
    if scale:
        vf.append(f"scale={scale[0]}:{scale[1]}")
    vf_str = ",".join(vf) if vf else None

    if "vaapi" in encoder:
        cmd = ["ffmpeg", "-y", "-ss", "0", "-t", str(test_seconds), "-i", input_path]
        if vf_str:
            cmd += ["-vf", f"{vf_str},format=nv12,hwupload"]
        else:
            cmd += ["-vf", "format=nv12,hwupload"]
        cmd += ["-vaapi_device", "/dev/dri/renderD128", "-c:v", encoder, "-b:v", "4M", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-ss", "0", "-t", str(test_seconds), "-i", input_path]
        if vf_str:
            cmd += ["-vf", vf_str]
        if "nvenc" in encoder:
            cmd += ["-c:v", encoder, "-preset", "p1", "-rc", "vbr_hq", "-cq", "19", "-b:v", "0"]
        elif "qsv" in encoder:
            cmd += ["-c:v", encoder, "-preset", "veryfast", "-b:v", "4M"]
        else:
            cmd += ["-c:v", encoder, "-preset", "fast", "-crf", "23"]
        cmd += ["-c:a", "copy", output_path]

    log_fn(f"Running ffmpeg for encoder {encoder}")
    t0 = time.time()
    p = run(cmd, capture=True)
    t1 = time.time()
    elapsed = t1 - t0
    log_fn(p.stderr.decode('utf-8', errors='ignore'))
    frames = None
    try:
        info = ffprobe_info(output_path)
        v = analyze_streams(info)
        if v:
            fps_expr = v.get("r_frame_rate", "30/1")
            try:
                fps_val = eval(fps_expr)
            except Exception:
                fps_val = 30.0
            frames = int(round(v.get("duration", test_seconds) * fps_val))
    except Exception:
        frames = None
    fps = frames / elapsed if frames and elapsed > 0 else None
    return {"elapsed": elapsed, "fps": fps, "stdout": p.stdout.decode('utf-8', errors='ignore'), "stderr": p.stderr.decode('utf-8', errors='ignore')}

def compute_quality_metrics(orig: str, recon: str, log_fn) -> Dict:
    cmd = [
        "ffmpeg", "-i", orig, "-i", recon,
        "-lavfi", "psnr=stats_file=-;[0:v][1:v]ssim=stats_file=-",
        "-f", "null", "-"
    ]
    log_fn("Computing PSNR/SSIM")
    p = run(cmd, capture=True)
    out = p.stderr.decode('utf-8', errors='ignore')
    log_fn(out)
    psnr = None
    ssim = None
    for line in out.splitlines():
        if "PSNR" in line and "average" in line:
            try:
                part = line.split("average:")[-1].strip().split()[0]
                psnr = float(part)
            except Exception:
                pass
        if "All:" in line and "SSIM" in line:
            try:
                ssim = float(line.split("All:")[1].split()[0])
            except Exception:
                pass
    return {"psnr": psnr, "ssim": ssim, "raw": out}

# -----------------------
# GUI and orchestration
# -----------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Transcode Analyzer and Tester")
        self.frame = ttk.Frame(root, padding=10)
        self.frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Input file
        ttk.Label(self.frame, text="Input file").grid(column=0, row=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(self.frame, textvariable=self.input_var, width=60)
        self.input_entry.grid(column=0, row=1, sticky="we", columnspan=3)
        ttk.Button(self.frame, text="Browse", command=self.browse_input).grid(column=3, row=1, sticky="e")

        # Output dir
        ttk.Label(self.frame, text="Output directory").grid(column=0, row=2, sticky="w")
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.frame, textvariable=self.output_var, width=60)
        self.output_entry.grid(column=0, row=3, sticky="we", columnspan=3)
        ttk.Button(self.frame, text="Browse", command=self.browse_output).grid(column=3, row=3, sticky="e")

        # Options
        self.auto_install_var = tk.BooleanVar(value=False)
        self.build_kernel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame, text="Auto-install missing deps", variable=self.auto_install_var).grid(column=0, row=4, sticky="w")
        ttk.Checkbutton(self.frame, text="Build processing kernel", variable=self.build_kernel_var).grid(column=1, row=4, sticky="w")

        ttk.Label(self.frame, text="Test seconds").grid(column=0, row=5, sticky="w")
        self.test_seconds_var = tk.IntVar(value=TEST_DURATION_DEFAULT)
        ttk.Entry(self.frame, textvariable=self.test_seconds_var, width=6).grid(column=1, row=5, sticky="w")

        ttk.Label(self.frame, text="Scale (WxH)").grid(column=2, row=5, sticky="e")
        self.scale_var = tk.StringVar(value="")
        ttk.Entry(self.frame, textvariable=self.scale_var, width=12).grid(column=3, row=5, sticky="w")

        # Buttons
        self.run_button = ttk.Button(self.frame, text="Run Analysis and Tests", command=self.start_worker)
        self.run_button.grid(column=0, row=6, columnspan=2, pady=(8,0))
        self.stop_button = ttk.Button(self.frame, text="Stop", command=self.stop_worker, state="disabled")
        self.stop_button.grid(column=2, row=6, columnspan=2, pady=(8,0))

        # Log area
        ttk.Label(self.frame, text="Log").grid(column=0, row=7, sticky="w")
        self.log_widget = scrolledtext.ScrolledText(self.frame, width=100, height=24)
        self.log_widget.grid(column=0, row=8, columnspan=4, sticky="nsew", pady=(4,0))
        self.frame.rowconfigure(8, weight=1)

        # Worker control
        self.worker_thread = None
        self.stop_event = threading.Event()

    def browse_input(self):
        path = filedialog.askopenfilename(title="Select input video", filetypes=[("Video files","*.mp4 *.mkv *.mov *.avi *.webm"), ("All files","*.*")])
        if path:
            self.input_var.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log_widget.insert(tk.END, f"[{ts}] {text}\n")
        self.log_widget.see(tk.END)

    def start_worker(self):
        inp = self.input_var.get().strip()
        outdir = self.output_var.get().strip()
        if not inp or not Path(inp).exists():
            messagebox.showerror("Input error", "Please select a valid input file.")
            return
        if not outdir:
            messagebox.showerror("Output error", "Please select an output directory.")
            return
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "Worker is already running.")
            return
        self.stop_event.clear()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.worker_thread = threading.Thread(target=self.worker_main, args=(inp, outdir, self.auto_install_var.get(), self.build_kernel_var.get(), self.test_seconds_var.get(), self.scale_var.get()))
        self.worker_thread.start()

    def stop_worker(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.log("Stop requested. Waiting for worker to finish current step...")
            self.stop_button.config(state="disabled")

    def worker_main(self, input_path, output_dir, auto_install, build_kernel, test_seconds, scale_text):
        try:
            self.log("Starting analysis")
            # 1) Check system commands
            missing = [c for c in SYSTEM_CMDS if not which(c)]
            if missing:
                self.log(f"Missing system commands: {missing}")
                if auto_install and detect_distro() == "debian":
                    self.log("Attempting apt install for missing commands (requires sudo)")
                    try:
                        run(["sudo", "apt", "update"], check=True)
                        run(["sudo", "apt", "install", "-y"] + missing, check=True)
                        self.log("Apt install completed")
                    except Exception as e:
                        self.log(f"Apt install failed: {e}")
                else:
                    self.log("Auto-install not enabled or unsupported distro; continuing but ffmpeg may be missing")
            # 2) Probe input
            self.log("Probing input with ffprobe")
            info = ffprobe_info(input_path)
            stream_info = analyze_streams(info)
            self.log(f"Stream info: {stream_info}")
            # 3) Detect hardware
            hw = check_hw_accel()
            self.log(f"Hardware detection: {hw}")
            cuda_toolchain = detect_cuda_toolchain()
            self.log(f"CUDA toolchain present: {cuda_toolchain}")
            # 4) Select candidates
            candidates = []
            if hw.get("nvenc"):
                candidates += [e for e in ["h264_nvenc", "hevc_nvenc"] if e in ENCODER_CANDIDATES]
            if hw.get("vaapi"):
                candidates += [e for e in ["h264_vaapi", "hevc_vaapi"] if e in ENCODER_CANDIDATES]
            if hw.get("intel_qsv"):
                candidates += [e for e in ["h264_qsv"] if e in ENCODER_CANDIDATES]
            for cpu in ["libx264", "libx265"]:
                if cpu in ENCODER_CANDIDATES and cpu not in candidates:
                    candidates.append(cpu)
            # filter by ffmpeg availability
            if which("ffmpeg"):
                try:
                    p = run(["ffmpeg", "-hide_banner", "-encoders"], capture=True)
                    encs = p.stdout.decode('utf-8', errors='ignore')
                    candidates = [c for c in candidates if c in encs]
                except Exception:
                    pass
            if not candidates:
                candidates = ["libx264"]
            self.log(f"Encoder candidates: {candidates}")
            # 5) Optionally build kernel
            proc_lib = None
            tmpdir = None
            if build_kernel:
                tmpdir = tempfile.mkdtemp(prefix="vidproc_")
                try:
                    use_cuda = cuda_toolchain
                    self.log(f"Building shared lib in {tmpdir} use_cuda={use_cuda}")
                    so_path, mode = build_shared_lib(tmpdir, use_cuda, self.log)
                    self.log(f"Built shared lib at {so_path} mode={mode}")
                    proc_lib = ProcLib(so_path, mode)
                    # quick test: extract one frame
                    w = stream_info.get("width", 640) if stream_info else 640
                    h = stream_info.get("height", 480) if stream_info else 480
                    tmp_frame = os.path.join(tmpdir, "frame.rgb")
                    run(["ffmpeg", "-y", "-i", input_path, "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", tmp_frame], check=True)
                    with open(tmp_frame, "rb") as f:
                        data = f.read()
                    outbuf = bytearray(len(data))
                    proc_lib.process(data, outbuf, w, h)
                    self.log("Shared lib processing test succeeded")
                except Exception as e:
                    self.log(f"Kernel build/test failed: {e}")
                    proc_lib = None
            # 6) Run short tests
            scale = None
            if scale_text:
                try:
                    w,h = scale_text.split("x")
                    scale = (int(w), int(h))
                except Exception:
                    self.log("Invalid scale; ignoring")
            results = []
            for enc in candidates:
                if self.stop_event.is_set():
                    self.log("Stop requested; aborting tests")
                    break
                out_file = os.path.join(output_dir, f"test_{enc.replace('/', '_')}.mp4")
                self.log(f"Testing encoder {enc} -> {out_file}")
                try:
                    res = run_test_transcode(input_path, out_file, enc, int(test_seconds), scale, self.log)
                    metrics = compute_quality_metrics(input_path, out_file, self.log)
                    res.update({"encoder": enc, "out_file": out_file, "metrics": metrics})
                    self.log(f"Result encoder={enc} elapsed={res['elapsed']:.2f}s fps={res.get('fps')} psnr={metrics.get('psnr')} ssim={metrics.get('ssim')}")
                except Exception as e:
                    self.log(f"Encoder {enc} failed: {e}")
                    res = {"encoder": enc, "error": str(e)}
                results.append(res)
            # 7) Summarize
            self.log("=== SUMMARY ===")
            for r in results:
                self.log(str(r))
            if tmpdir:
                self.log(f"Temporary build directory: {tmpdir}")
            self.log("All done")
        except Exception as e:
            self.log("Unhandled exception in worker:\n" + traceback.format_exc())
        finally:
            self.run_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.stop_event.clear()

def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("980x700")
    root.mainloop()

if __name__ == "__main__":
    main()

