#!/usr/bin/env python3
"""
Live image + system monitor for Jupyter / Kaggle / Colab, styled like Windows 11 (Fluent).

Notebook usage:
    %run preview.py --target_folder /root/LoRA/output/sample
or:
    from preview import PreviewMonitor
    m = PreviewMonitor("/path/to/folder")
    m.tampilkan_ui()
    m.mulai_auto_refresh()

Recommended for GPU stats without spawning nvidia-smi (falls back to nvidia-smi if missing):
    pip install nvidia-ml-py
"""
from __future__ import annotations

import argparse
import base64
import builtins
import io
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict, defaultdict, deque
from html import escape

import psutil
from IPython.display import HTML, clear_output, display, update_display
from PIL import Image, ImageOps

try:  # widgets update over their own channel, so they keep working when other cells run
    import ipywidgets as _w
except ImportError:
    _w = None

try:  # NVML gives microsecond-cheap, real-time GPU readings
    import pynvml
except ImportError:
    pynvml = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
HISTORY = 120          # samples kept for sparklines (120 x 0.5s = 60s)
THUMB_PX = 360
THUMB_CACHE = 96


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _read(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _kv(text):
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None  # e.g. "[N/A]" / "[Not Supported]"


def _fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _level(p):
    if p is None:
        return ""
    return "bad" if p >= 90 else "warn" if p >= 75 else ""


class _Rate:
    """Turns monotonically increasing counters into per-second rates."""

    def __init__(self):
        self.prev = None

    def update(self, now, *vals):
        if self.prev is None:
            self.prev = (now, vals)
            return [0.0] * len(vals)
        t0, v0 = self.prev
        dt = max(now - t0, 1e-6)
        self.prev = (now, vals)
        return [max((a - b) / dt, 0.0) for a, b in zip(vals, v0)]


# --------------------------------------------------------------------------- #
# Container awareness (Docker / Kaggle): psutil reports HOST totals, which is
# misleading inside a limited container. Read the cgroup limits instead.
# --------------------------------------------------------------------------- #
class _Cgroup:
    def __init__(self):
        self.v2 = os.path.exists("/sys/fs/cgroup/cgroup.controllers")
        host_cpus = psutil.cpu_count(logical=True) or 1
        self.cpu_limit = float(host_cpus)
        self.cpu_limited = False
        self._cpu_usage_path = None
        self._mem_limited = False
        host_mem = psutil.virtual_memory().total

        if self.v2:
            parts = (_read("/sys/fs/cgroup/cpu.max") or "").split()
            if len(parts) == 2 and parts[0] != "max":
                try:
                    lim = int(parts[0]) / int(parts[1])
                    if lim < host_cpus - 0.01:
                        self.cpu_limit, self.cpu_limited = lim, True
                except (ValueError, ZeroDivisionError):
                    pass
            lim = _read("/sys/fs/cgroup/memory.max")
            if lim and lim != "max" and lim.isdigit() and int(lim) < host_mem:
                self._mem_limited = True
        else:
            for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
                q, p = _read(f"{base}/cpu.cfs_quota_us"), _read(f"{base}/cpu.cfs_period_us")
                if q and p and int(q) > 0 and int(p) > 0:
                    lim = int(q) / int(p)
                    if lim < host_cpus - 0.01:
                        self.cpu_limit, self.cpu_limited = lim, True
                    break
            for path in ("/sys/fs/cgroup/cpuacct/cpuacct.usage",
                         "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
                         "/sys/fs/cgroup/cpu/cpuacct.usage"):
                if os.path.exists(path):
                    self._cpu_usage_path = path
                    break
            lim = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if lim and lim.isdigit() and int(lim) < host_mem:
                self._mem_limited = True

    def cpu_usage(self):
        """Cumulative CPU seconds used by this cgroup, or None."""
        if self.v2:
            v = _kv(_read("/sys/fs/cgroup/cpu.stat")).get("usage_usec")
            return v / 1e6 if v is not None else None
        raw = _read(self._cpu_usage_path) if self._cpu_usage_path else None
        return int(raw) / 1e9 if raw and raw.isdigit() else None

    def mem(self):
        """(used, limit) in bytes if the container has a memory limit, else None."""
        if not self._mem_limited:
            return None
        try:
            if self.v2:
                limit = int(_read("/sys/fs/cgroup/memory.max"))
                cur = int(_read("/sys/fs/cgroup/memory.current"))
                inactive = _kv(_read("/sys/fs/cgroup/memory.stat")).get("inactive_file", 0)
            else:
                limit = int(_read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
                cur = int(_read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                st = _kv(_read("/sys/fs/cgroup/memory/memory.stat"))
                inactive = st.get("total_inactive_file", st.get("inactive_file", 0))
            return max(cur - inactive, 0), limit  # same formula as `docker stats`
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------------------- #
# GPU: NVML when available, otherwise a background nvidia-smi poller
# (never blocks the UI thread, even when the GPU is 100% busy).
# --------------------------------------------------------------------------- #
class _GPUReader:
    SMI_QUERY = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                 "power.draw,power.limit,clocks.sm,clocks.mem,fan.speed")

    def __init__(self):
        self.backend = None
        self._handles, self._names = [], []
        self._latest = None
        self._stop = threading.Event()
        self._thread = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    self._handles.append(h)
                    self._names.append(name.decode(errors="replace") if isinstance(name, bytes) else name)
                if self._handles:
                    self.backend = "nvml"
            except Exception:
                self._handles, self._names = [], []
        if self.backend is None and shutil.which("nvidia-smi"):
            self.backend = "smi"

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self.backend == "smi" and not (self._thread and self._thread.is_alive()):
            self._stop.clear()
            self._thread = threading.Thread(target=self._smi_loop, daemon=True, name="preview-smi")
            self._thread.start()

    def stop(self):
        self._stop.set()

    # -- reading -------------------------------------------------------------
    def read(self):
        if self.backend == "nvml":
            return [self._nvml(i, h) for i, h in enumerate(self._handles)]
        if self.backend == "smi":
            if self._latest is None:
                self._smi_once()
            return list(self._latest or [])
        return []

    @staticmethod
    def _try(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def _nvml(self, i, h):
        t = self._try
        util = t(pynvml.nvmlDeviceGetUtilizationRates, h)
        mem = t(pynvml.nvmlDeviceGetMemoryInfo, h)
        power = t(pynvml.nvmlDeviceGetPowerUsage, h)
        limit = t(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
        return dict(
            index=i, name=self._names[i],
            util=util.gpu if util is not None else None,
            vram_used=mem.used if mem is not None else None,
            vram_total=mem.total if mem is not None else None,
            temp=t(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
            power=power / 1000 if power is not None else None,
            power_limit=limit / 1000 if limit is not None else None,
            clock=t(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS),
            mclock=t(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM),
            fan=t(pynvml.nvmlDeviceGetFanSpeed, h),
        )

    def _smi_once(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={self.SMI_QUERY}", "--format=csv,noheader,nounits"],
                encoding="utf-8", timeout=3, stderr=subprocess.DEVNULL)
            rows = []
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) < 11:
                    continue
                mib = lambda v: None if v is None else v * 1048576
                rows.append(dict(
                    index=int(_num(p[0]) or 0), name=p[1], util=_num(p[2]),
                    vram_used=mib(_num(p[3])), vram_total=mib(_num(p[4])),
                    temp=_num(p[5]), power=_num(p[6]), power_limit=_num(p[7]),
                    clock=_num(p[8]), mclock=_num(p[9]), fan=_num(p[10])))
            self._latest = rows
        except Exception:
            if self._latest is None:
                self._latest = []

    def _smi_loop(self):
        while not self._stop.is_set():
            self._smi_once()
            self._stop.wait(0.4)


# --------------------------------------------------------------------------- #
# Rendering pieces
# --------------------------------------------------------------------------- #
_ICONS = {
    "cpu": '<rect x="4" y="4" width="8" height="8" rx="1.5"/><path d="M6 1.5V4M10 1.5V4M6 12v2.5M10 12v2.5M1.5 6H4M1.5 10H4M12 6h2.5M12 10h2.5"/>',
    "ram": '<rect x="1.5" y="4.5" width="13" height="6" rx="1"/><path d="M4 10.5v2M7 10.5v2M10 10.5v2M13 10.5v2M4.5 7.5h1M7.5 7.5h1M10.5 7.5h1"/>',
    "gpu": '<rect x="1.5" y="4" width="13" height="8" rx="1.5"/><circle cx="8" cy="8" r="2.2"/><path d="M4 12v2M12 12v2"/>',
    "disk": '<ellipse cx="8" cy="4" rx="5" ry="2"/><path d="M3 4v8c0 1.1 2.2 2 5 2s5-.9 5-2V4M3 8c0 1.1 2.2 2 5 2s5-.9 5-2"/>',
    "net": '<path d="M5 13V3M2.5 5.5 5 3l2.5 2.5M11 3v10M8.5 10.5 11 13l2.5-2.5"/>',
    "image": '<rect x="2" y="2.5" width="12" height="11" rx="1.5"/><circle cx="6" cy="6.5" r="1.2"/><path d="m2.5 12 3.5-3.5 2.5 2.5 2-2 3 3"/>',
}


def _icon(name):
    return ('<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" '
            'stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="color:var(--c,var(--accent));flex:none">{_ICONS[name]}</svg>')


def _spark(values, uid, vmax=100.0, floor=1.0):
    vals = list(values)
    n = HISTORY
    if len(vals) < 2:
        return '<svg class="spark" viewBox="0 0 100 36" preserveAspectRatio="none"></svg>'
    top = vmax if vmax else max(max(vals), floor)
    off = n - len(vals)  # graph grows in from the right, like Task Manager
    xs, pts = [], []
    for i, v in enumerate(vals):
        x = (off + i) * 100.0 / (n - 1)
        y = 34.0 - min(max(v / top, 0.0), 1.0) * 32.0
        xs.append(x)
        pts.append(f"{x:.2f},{y:.2f}")
    line = " ".join(pts)
    area = f"{xs[0]:.2f},36 {line} {xs[-1]:.2f},36"
    return ('<svg class="spark" viewBox="0 0 100 36" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="currentColor" stop-opacity=".35"/>'
            '<stop offset="1" stop-color="currentColor" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{uid})"/>'
            f'<polyline points="{line}" fill="none" stroke="currentColor" stroke-width="1.5" '
            'vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def _bar(pct, level=""):
    return f'<div class="bar {level}"><i style="width:{max(0.0, min(100.0, pct)):.1f}%"></i></div>'


def _card(title, icon, big, unit="", sub="", spark="", pct=None, body="", level=""):
    return (f'<div class="card {level}"><div class="ch">{_icon(icon)}<span>{title}</span></div>'
            f'<div class="big num">{big}<small>{unit}</small></div>'
            + (f'<div class="sub">{sub}</div>' if sub else "")
            + spark
            + (_bar(pct) if pct is not None else "")
            + body + '</div>')


_CSS = """<style>
.mw{--accent:#60cdff;--ok:#6ccb5f;--warn:#fce100;--bad:#ff99a4;--onaccent:#003a52;
 font-family:"Segoe UI Variable Text","Segoe UI Variable","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;
 font-size:13px;line-height:1.35;color:var(--text);background:var(--bg);border:1px solid var(--stroke);
 border-radius:10px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.25);margin:6px 0;box-sizing:border-box}
.mw *{box-sizing:border-box}
.mw.dark{--bg:#202020;--bar:#2b2b2b;--card:rgba(255,255,255,.055);--well:rgba(0,0,0,.28);--stroke:rgba(255,255,255,.08);
 --text:#fff;--text2:rgba(255,255,255,.72);--text3:rgba(255,255,255,.46);--track:rgba(255,255,255,.12);--hover:rgba(255,255,255,.08)}
.mw.light{--bg:#f3f3f3;--bar:#e9e9e9;--card:rgba(255,255,255,.85);--well:rgba(0,0,0,.045);--stroke:rgba(0,0,0,.09);
 --text:#1b1b1b;--text2:rgba(0,0,0,.66);--text3:rgba(0,0,0,.46);--track:rgba(0,0,0,.1);--hover:rgba(0,0,0,.06);
 --accent:#0067c0;--warn:#9d5d00;--bad:#c42b1c;--ok:#0f7b0f;--onaccent:#fff;box-shadow:0 8px 24px rgba(0,0,0,.12)}
.mw .num{font-variant-numeric:tabular-nums}
.mw .live{display:flex;justify-content:flex-end;align-items:center;gap:6px;padding:8px 12px 0;font-size:11px;color:var(--text3)}
.mw .live i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);transition:opacity .2s}
.mw .live i.off{opacity:.15;box-shadow:none}
.mw .live.err i{background:var(--bad);box-shadow:0 0 6px var(--bad)}
.mw .live.stop i{background:var(--text3);box-shadow:none;opacity:.6}
.mw .live+.grid{padding-top:6px}
.mw .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;padding:10px}
.mw .card{--c:var(--accent);background:var(--card);border:1px solid var(--stroke);border-radius:8px;padding:12px 14px;
 display:flex;flex-direction:column;gap:7px;min-width:0}
.mw .card.warn{--c:var(--warn)}
.mw .card.bad{--c:var(--bad)}
.mw .ch{display:flex;align-items:center;gap:8px;color:var(--text2);font-size:12px;min-width:0}
.mw .ch span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.mw .big{font-size:26px;font-weight:600;letter-spacing:-.02em;line-height:1.1;display:flex;align-items:baseline;gap:5px;white-space:nowrap}
.mw .big small{font-size:12px;font-weight:400;letter-spacing:0;color:var(--text2)}
.mw .sub{font-size:11.5px;color:var(--text3)}
.mw .spark{width:100%;height:38px;display:block;color:var(--c);border-radius:4px;background-color:var(--well);
 background-image:linear-gradient(var(--stroke) 1px,transparent 1px);background-size:100% 25%}
.mw .bar{height:4px;border-radius:2px;background:var(--track);overflow:hidden}
.mw .bar i{display:block;height:100%;border-radius:2px;background:var(--c)}
.mw .bar.warn i{background:var(--warn)}
.mw .bar.bad i{background:var(--bad)}
.mw .kv{display:flex;flex-wrap:wrap;gap:3px 14px;font-size:11.5px;color:var(--text2)}
.mw .kv b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
.mw .kv.sp{justify-content:space-between}
.mw .cores{display:flex;gap:2px;height:18px;align-items:flex-end}
.mw .cores i{flex:1;min-width:2px;min-height:2px;border-radius:1px;background:var(--c);opacity:.85}
.mw .gal{display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:8px;padding:10px}
.mw .tile{position:relative;margin:0;padding:6px;background:var(--card);border:1px solid var(--stroke);
 border-radius:8px;transition:transform .12s ease,border-color .12s ease}
.mw .tile:hover{transform:translateY(-2px);border-color:var(--accent)}
.mw .ph{aspect-ratio:1/1;background:var(--well);border-radius:5px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.mw .ph img{width:100%;height:100%;object-fit:contain;display:block}
.mw .cap{padding:6px 2px 1px;display:flex;flex-direction:column;gap:1px;min-width:0}
.mw .cap .n{font-size:11.5px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mw .cap .m{font-size:10.5px;color:var(--text3)}
.mw .badge{position:absolute;top:11px;left:11px;z-index:1;background:var(--accent);color:var(--onaccent);
 font-size:10px;font-weight:600;padding:1px 8px;border-radius:999px}
.mw .empty{padding:44px 10px;text-align:center;color:var(--text3);display:flex;flex-direction:column;align-items:center;gap:10px}
.mw .empty svg{width:28px;height:28px;color:var(--text3)}
</style>"""


def _registry():
    """Survives `%run` re-executions so old monitors can always be stopped."""
    reg = getattr(builtins, "_preview_monitors", None)
    if reg is None:
        reg = []
        builtins._preview_monitors = reg
    return reg


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class PreviewMonitor:
    def __init__(self, folder_path, max_images=16, kolom=4, interval_detik=2.0,
                 stats_interval=0.5, theme="dark"):
        self.folder_path = os.path.abspath(os.path.expanduser(folder_path))
        self.max_images = max_images
        self.kolom = max(1, int(kolom))
        self.image_interval = max(0.5, float(interval_detik))   # how often the folder is checked
        self.stats_interval = max(0.2, float(stats_interval))   # how often stats refresh
        self.theme = theme if theme in ("dark", "light") else "dark"

        self.uid = str(int(time.time() * 1000) % 10**9)
        self.stats_id = f"mon_stats_{self.uid}"
        self.gallery_id = f"mon_gallery_{self.uid}"

        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.RLock()
        self._last_error = None
        self._t0 = time.monotonic()   # start of the running timer
        self._tick = 0                # heartbeat counter: the dot blinks once per stats update
        self._stopped = False

        self._cg = _Cgroup()
        self.gpu = _GPUReader()
        self._hist = defaultdict(lambda: deque(maxlen=HISTORY))
        self._disk_rate, self._net_rate = _Rate(), _Rate()
        self._cg_cpu_prev = None
        self._freq, self._freq_t = None, 0.0

        psutil.cpu_percent(None)                  # warm-up: first call is always 0.0
        psutil.cpu_percent(None, percpu=True)

        self._thumbs = OrderedDict()
        self._gal_last_html = None

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _sample(self):
        now = time.monotonic()
        cg, h = self._cg, self._hist
        s = {}

        # CPU: container quota aware, otherwise host-wide
        pct = None
        if cg.cpu_limited:
            usage = cg.cpu_usage()
            if usage is not None:
                if self._cg_cpu_prev is not None:
                    t0, u0 = self._cg_cpu_prev
                    dt = now - t0
                    if dt > 0:
                        pct = max(0.0, min(100.0, (usage - u0) / (dt * cg.cpu_limit) * 100))
                self._cg_cpu_prev = (now, usage)
        host_pct = psutil.cpu_percent(None)
        cores = psutil.cpu_percent(None, percpu=True)
        s["cpu"] = host_pct if pct is None else pct
        s["cores"] = None if cg.cpu_limited else cores
        s["cpu_count"] = cg.cpu_limit if cg.cpu_limited else len(cores)
        if now - self._freq_t > 2.0:  # cpu_freq() can be slow on many-core hosts
            try:
                f = psutil.cpu_freq()
                self._freq = f.current if f and f.current else None
            except Exception:
                self._freq = None
            self._freq_t = now
        s["freq"] = self._freq
        h["cpu"].append(s["cpu"])

        # Memory
        vm = psutil.virtual_memory()
        used, total, src = vm.total - vm.available, vm.total, "host"
        lim = cg.mem()
        if lim:
            (used, total), src = lim, "container limit"
        s["ram"] = (used, total, used / total * 100 if total else 0.0, src)
        h["ram"].append(s["ram"][2])

        # Disk usage of the filesystem holding the output folder + I/O throughput
        try:
            du = shutil.disk_usage(self.folder_path if os.path.exists(self.folder_path) else os.sep)
            s["disk"] = (du.used, du.total, du.used / du.total * 100 if du.total else 0.0)
        except OSError:
            s["disk"] = None
        try:
            io_c = psutil.disk_io_counters()
        except Exception:
            io_c = None
        if io_c:
            r, w = self._disk_rate.update(now, io_c.read_bytes, io_c.write_bytes)
            s["disk_io"] = (r, w)
            h["disk_io"].append(r + w)
        else:
            s["disk_io"] = None

        # Network throughput
        try:
            n = psutil.net_io_counters()
            down, up = self._net_rate.update(now, n.bytes_recv, n.bytes_sent)
            s["net"] = (down, up)
            h["net"].append(down + up)
        except Exception:
            s["net"] = None

        # GPU(s)
        s["gpus"] = self.gpu.read()
        for g in s["gpus"]:
            i = g["index"]
            if g["util"] is not None:
                h[f"gpu{i}"].append(g["util"])
            if g["vram_used"] is not None and g["vram_total"]:
                h[f"vram{i}"].append(g["vram_used"] / g["vram_total"] * 100)
        return s

    # ------------------------------------------------------------------ #
    # System stats UI
    # ------------------------------------------------------------------ #
    def _stats_html(self):
        with self._lock:
            return self._compose_stats(self._sample())

    def _compose_stats(self, s):
        u, h = self.uid, self._hist
        cards = []

        # CPU
        cpu = s["cpu"]
        cnt = s["cpu_count"]
        sub = f"{cnt:g} vCPU (limit)" if self._cg.cpu_limited else f"{cnt:g} logical cores"
        if s["freq"]:
            sub += f" \u00b7 {s['freq'] / 1000:.2f} GHz"
        body = ""
        if s["cores"]:
            body = '<div class="cores">' + "".join(
                f'<i style="height:{max(p, 2):.0f}%"></i>' for p in s["cores"][:64]) + "</div>"
        cards.append(_card("CPU", "cpu", f"{cpu:.1f}", "%", sub,
                           _spark(h["cpu"], f"g{u}c"), body=body, level=_level(cpu)))

        # Memory
        used, total, rpct, src = s["ram"]
        cards.append(_card("Memory", "ram", f"{used / 1024**3:.1f}", f"/ {total / 1024**3:.1f} GB \u00b7 {rpct:.0f}%",
                           f"{_fmt_bytes(max(total - used, 0))} available \u00b7 {src}",
                           _spark(h["ram"], f"g{u}m"), pct=rpct, level=_level(rpct)))

        # GPU(s)
        if not s["gpus"]:
            cards.append(_card("GPU", "gpu", "\u2014", "", "No NVIDIA GPU detected"))
        for g in s["gpus"]:
            i, util = g["index"], g["util"]
            vp = (g["vram_used"] / g["vram_total"] * 100) if g["vram_used"] is not None and g["vram_total"] else None
            body = ""
            if vp is not None:
                body += (f'<div class="kv sp"><span>VRAM <b>{_fmt_bytes(g["vram_used"])} / {_fmt_bytes(g["vram_total"])}</b></span>'
                         f'<b>{vp:.0f}%</b></div>{_bar(vp, _level(vp))}')
            chips = []
            if g["temp"] is not None:
                chips.append(f'<span>Temp <b>{g["temp"]:.0f} \u00b0C</b></span>')
            if g["power"] is not None:
                pw = f'{g["power"]:.0f}' + (f' / {g["power_limit"]:.0f}' if g.get("power_limit") else "")
                chips.append(f'<span>Power <b>{pw} W</b></span>')
            if g["clock"] is not None:
                chips.append(f'<span>Core <b>{g["clock"]:.0f} MHz</b></span>')
            if g.get("mclock") is not None:
                chips.append(f'<span>Mem <b>{g["mclock"]:.0f} MHz</b></span>')
            if g.get("fan") is not None:
                chips.append(f'<span>Fan <b>{g["fan"]:.0f}%</b></span>')
            if chips:
                body += f'<div class="kv">{"".join(chips)}</div>'
            lvl = "bad" if "bad" in (_level(util), _level(vp)) else "warn" if "warn" in (_level(util), _level(vp)) else ""
            title = f'GPU {i} \u00b7 {escape(g["name"])}' if len(s["gpus"]) > 1 else f'GPU \u00b7 {escape(g["name"])}'
            cards.append(_card(title, "gpu", f"{util:.0f}" if util is not None else "\u2014", "%", "",
                               _spark(h[f"gpu{i}"], f"g{u}g{i}"), body=body, level=lvl))

        # Disk
        if s["disk"]:
            du, dt, dp = s["disk"]
            body = ""
            if s["disk_io"]:
                body = (f'<div class="kv"><span>Read <b>{_fmt_bytes(s["disk_io"][0])}/s</b></span>'
                        f'<span>Write <b>{_fmt_bytes(s["disk_io"][1])}/s</b></span></div>')
            cards.append(_card("Disk", "disk", f"{dp:.1f}", "% used",
                               f"{_fmt_bytes(dt - du)} free of {_fmt_bytes(dt)}",
                               _spark(h["disk_io"], f"g{u}d", vmax=None, floor=1e6) if s["disk_io"] else "",
                               pct=dp, body=body, level=_level(dp)))

        # Network
        if s["net"]:
            down, up = s["net"]
            cards.append(_card("Network", "net", _fmt_bytes(down), "/s down", "",
                               _spark(h["net"], f"g{u}n", vmax=None, floor=1e6),
                               body=f'<div class="kv"><span>Down <b>{_fmt_bytes(down)}/s</b></span>'
                                    f'<span>Up <b>{_fmt_bytes(up)}/s</b></span></div>'))

        return f'<div class="mw {self.theme}">{self._live_html()}<div class="grid">{"".join(cards)}</div></div>'

    def _live_html(self):
        """Small heartbeat: the dot toggles on every stats update, so if updates stop, the blinking stops."""
        self._tick += 1
        sec = int(time.monotonic() - self._t0)
        h, r = divmod(sec, 3600)
        m, sc = divmod(r, 60)
        clock = f"{h:02d}:{m:02d}:{sc:02d}"
        if self._stopped:
            return f'<div class="live stop"><i></i><span class="num">Stopped \u00b7 {clock}</span></div>'
        if self._last_error:
            tip = escape(self._last_error, quote=True)
            return f'<div class="live err" title="{tip}"><i></i><span class="num">Error \u00b7 {clock}</span></div>'
        dot = "" if self._tick % 2 else ' class="off"'
        return f'<div class="live"><i{dot}></i><span class="num">Running \u00b7 {clock}</span></div>'

    # ------------------------------------------------------------------ #
    # Image gallery UI
    # ------------------------------------------------------------------ #
    def _scan(self):
        items, total = [], 0
        try:
            with os.scandir(self.folder_path) as it:
                for e in it:
                    if e.name.lower().endswith(IMG_EXTS) and e.is_file():
                        total += 1
                        try:
                            st = e.stat()
                            items.append((e.name, st.st_mtime_ns, st.st_size))
                        except OSError:
                            pass  # file is being rewritten
        except OSError:
            return [], 0
        items.sort(key=lambda x: x[1], reverse=True)
        return items[: self.max_images], total

    def _thumb(self, name, mtime, size):
        key = (name, mtime, size)
        hit = self._thumbs.get(key)
        if hit:
            self._thumbs.move_to_end(key)
            return hit
        try:
            with Image.open(os.path.join(self.folder_path, name)) as im:
                orig = im.size
                im.draft("RGB", (THUMB_PX * 2, THUMB_PX * 2))  # fast JPEG downscale on decode
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (40, 40, 40))
                    bg.paste(im, mask=im.getchannel("A"))
                    im = bg
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im.thumbnail((THUMB_PX, THUMB_PX))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
        except Exception:
            return None  # unreadable / half-written; retried on the next pass
        val = (base64.b64encode(buf.getvalue()).decode("ascii"), orig[0], orig[1])
        self._thumbs[key] = val
        while len(self._thumbs) > THUMB_CACHE:
            self._thumbs.popitem(last=False)
        return val

    def _gallery_html(self, force=False):
        """Returns new HTML, or None if nothing changed since the last push."""
        items, total = self._scan()
        tiles = []
        for idx, (name, mt, sz) in enumerate(items):
            th = self._thumb(name, mt, sz)
            if th is None:
                continue
            b64, w, hgt = th
            n = escape(name, quote=True)
            clock = time.strftime("%H:%M:%S", time.localtime(mt / 1e9))
            tiles.append(
                '<figure class="tile">' + ('<span class="badge">Latest</span>' if idx == 0 else "")
                + f'<div class="ph"><img alt="" src="data:image/jpeg;base64,{b64}"></div>'
                f'<figcaption class="cap"><span class="n" title="{n}">{n}</span>'
                f'<span class="m num">{w}\u00d7{hgt} \u00b7 {clock}</span></figcaption></figure>')

        if tiles:
            content = f'<div class="gal" style="--cols:{self.kolom}">{"".join(tiles)}</div>'
        else:
            content = (f'<div class="empty">{_icon("image")}<div>Waiting for images</div>'
                       '<div class="sub">New files in this folder appear here automatically.</div></div>')
        html = f'<div class="mw {self.theme}">{content}</div>'
        if not force and html == self._gal_last_html:
            return None
        self._gal_last_html = html
        return html

    # ------------------------------------------------------------------ #
    # Public API (names kept compatible with the original script)
    # ------------------------------------------------------------------ #
    def _push_stats(self, html):
        if getattr(self, "_wst", None) is not None:
            self._wst.value = _CSS + html
        else:
            update_display(HTML(html), display_id=self.stats_id)

    def _push_gallery(self, html):
        if getattr(self, "_wgal", None) is not None:
            self._wgal.value = _CSS + html
        else:
            update_display(HTML(html), display_id=self.gallery_id)

    def tampilkan_ui(self):
        self._wst = self._wgal = None
        if _w is not None:
            try:
                try:  # Colab needs this for third-party widgets
                    from google.colab import output as _co
                    _co.enable_custom_widget_manager()
                except Exception:
                    pass
                self._wst = _w.HTML(_CSS + self._stats_html())
                self._wgal = _w.HTML(_CSS + self._gallery_html(force=True))
                display(_w.VBox([self._wst, self._wgal]))
                return
            except Exception:
                self._wst = self._wgal = None  # fall back to update_display
        display(HTML(_CSS))
        display(HTML(self._stats_html()), display_id=self.stats_id)
        display(HTML(self._gallery_html(force=True)), display_id=self.gallery_id)

    def refresh_manual(self, force_img_update=False):
        self._push_stats(self._stats_html())
        g = self._gallery_html(force=force_img_update)
        if g:
            self._push_gallery(g)

    def mulai_auto_refresh(self):
        if any(t.is_alive() for t in self._threads):
            return
        self._stop.clear()
        self._stopped = False
        self._t0 = time.monotonic()
        self.gpu.start()
        self._threads = [
            threading.Thread(target=self._stats_loop, daemon=True, name="preview-stats"),
            threading.Thread(target=self._gallery_loop, daemon=True, name="preview-gallery"),
        ]
        for t in self._threads:
            t.start()
        _registry().append(self)

    def hentikan_auto_refresh(self):
        self._stop.set()
        self.gpu.stop()
        for t in self._threads:
            t.join(timeout=1.5)
        self._stopped = True
        try:  # show a final "Stopped" state
            self._push_stats(self._stats_html())
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Background loops
    # ------------------------------------------------------------------ #
    def _stats_loop(self):
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self._push_stats(self._stats_html())
                self._last_error = None
            except Exception as e:  # keep the loop alive
                self._last_error = f"{type(e).__name__}: {e}"
            next_t += self.stats_interval
            delay = next_t - time.monotonic()
            if delay < 0:                      # we fell behind; don't try to catch up
                next_t, delay = time.monotonic(), 0
            self._stop.wait(delay)

    def _gallery_loop(self):
        while not self._stop.is_set():
            try:
                g = self._gallery_html()
                if g:
                    self._push_gallery(g)
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
            self._stop.wait(self.image_interval)


MonitorGambarKaggle = PreviewMonitor  # backwards-compatible alias


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default="/root/LoRA/output/sample")
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--image_interval", type=float, default=2.0, help="seconds between folder checks")
    parser.add_argument("--stats_interval", type=float, default=0.5, help="seconds between stats refreshes")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    args, _ = parser.parse_known_args(argv)  # ignores Jupyter's own -f kernel args

    os.makedirs(args.target_folder, exist_ok=True)

    # Stop monitors from previous runs of this cell so threads never pile up
    for old in list(_registry()):
        old.hentikan_auto_refresh()
    _registry().clear()

    clear_output(wait=True)
    mon = PreviewMonitor(args.target_folder, args.max_images, args.cols,
                         args.image_interval, args.stats_interval, args.theme)
    mon.tampilkan_ui()
    mon.mulai_auto_refresh()
    return mon


if __name__ == "__main__":
    monitor_aktif = main()
