"""Hardware detection, first-run install and the managed ComfyUI process.

Two engine modes are stored in data/config.json:
  external - the user runs their own ComfyUI; we only connect to its URL
  managed  - we installed ComfyUI ourselves and start/stop it with the app
"""

import asyncio
import ctypes
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import aiohttp

import catalog

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ---------- config ----------

class Config:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, **kw):
        self.data.update(kw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    @property
    def configured(self):
        return self.data.get("engine") in ("external", "managed", "cloud") and self.data.get("setup_done")


# ---------- hardware ----------

def detect_hardware(install_dir=None):
    hw = {"nvidia": False, "gpu": None, "vram_gb": 0, "compute_cap": 0, "driver": 0, "ram_gb": 0, "disk_free_gb": None}
    try:
        def smi(fields):
            return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
                                  creationflags=CREATE_NO_WINDOW).stdout
        out = smi("name,memory.total,compute_cap,driver_version")
        has_cc = True
        if not out.strip() or "not a valid field" in out:
            # old drivers do not know compute_cap: ask without it
            has_cc = False
            out = smi("name,memory.total,driver_version")
        gpus = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        if gpus:
            best = max(gpus, key=lambda g: float(g[1]))  # pick the card with most memory
            hw.update(nvidia=True, gpu=best[0].replace("NVIDIA GeForce ", ""), vram_gb=round(float(best[1]) / 1024))
            if has_cc:
                hw["compute_cap"] = float(best[2])
            try:
                hw["driver"] = float(best[3 if has_cc else 2])
            except (IndexError, ValueError):
                pass
    except Exception:
        pass

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        ms = MemStatus(dwLength=ctypes.sizeof(MemStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        hw["ram_gb"] = round(ms.ullTotalPhys / 1024 ** 3)
        hw["ram_free_gb"] = round(ms.ullAvailPhys / 1024 ** 3, 1)
        # commit limit = RAM + paging file; what is still free of it. When it runs low, Windows swaps the models to disk and the first step crawls
        hw["commit_limit_gb"] = round(ms.ullTotalPageFile / 1024 ** 3, 1)
        hw["commit_free_gb"] = round(ms.ullAvailPageFile / 1024 ** 3, 1)
    except Exception:
        pass

    if install_dir:
        hw["disk_free_gb"] = disk_free_gb(install_dir)
    return hw


def disk_free_gb(path):
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1e9)
    except OSError:
        return None


_found_cache = [0.0, []]


def find_installs():
    """Folders on any drive that already hold the program's ComfyUI (ComfyUI_windows_portable) with models or settings:
    the drive root and up to two levels down. The user is warned before a second copy is made (it would repeat ~50-90 GB)."""
    if time.time() - _found_cache[0] < 20:
        return list(_found_cache[1])
    skip = {"windows", "program files", "program files (x86)", "programdata", "$recycle.bin", "system volume information", "users", "recovery", "msocache"}
    out = []

    def has(base):
        try:
            c = base / catalog.COMFY_DIRNAME
            return c.is_dir() and ((c / "ComfyUI" / "models").is_dir() or (base / "data" / "config.json").is_file())
        except OSError:
            return False

    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\")
        try:
            if not root.exists():
                continue
            kids = [p for p in root.iterdir() if p.is_dir() and p.name.lower() not in skip]
        except OSError:
            continue
        if has(root):
            out.append(str(root))
        for k in kids:
            if has(k):
                out.append(str(k))
                continue
            try:
                for g in list(k.iterdir())[:300]:
                    if g.is_dir() and has(g):
                        out.append(str(g))
            except OSError:
                continue
    _found_cache[0], _found_cache[1] = time.time(), out
    return list(out)


def default_install_dir():
    # 1) a folder of an earlier install on any drive is used again (no second copy of ComfyUI and the models)
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZC":
        old = Path(f"{letter}:\\") / "MinimaxH3Workspace"
        try:
            if (old / "ComfyUI_windows_portable").is_dir() or (old / "data" / "config.json").is_file():
                return str(old)
        except OSError:
            continue
    # 2) the drive the program itself is installed on, when it has room: the models land next to the program, not on another drive
    if getattr(sys, "frozen", False):
        drv = Path(sys.executable).drive
        if drv and drv.upper() != "C:":
            try:
                if shutil.disk_usage(drv + "\\").free > 120e9:
                    return str(Path(drv + "\\") / "MinimaxH3Workspace")
            except OSError:
                pass
    # 3) biggest fixed drive other than C: if there is one, else the user's local app data
    best, best_free = None, 0
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if os.path.exists(root):
            try:
                free = shutil.disk_usage(root).free
            except OSError:
                continue
            if free > best_free:
                best, best_free = root, free
    if best and best_free > 120e9:
        return str(Path(best) / "MinimaxH3Workspace")
    return str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MinimaxH3Workspace")


def free_port(start=8188):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Нет свободного порта для ComfyUI")


def find_7z(app_root):
    for p in (app_root / "bin" / "7zr.exe", Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip" / "7z.exe"):
        if p.exists():
            return p
    return shutil.which("7z") or shutil.which("7zr")


# ---------- install ----------

class Cancelled(Exception):
    pass


HF_MIRROR = ""   # host of a Hugging Face mirror when the user switched it on (for slow connections)


def mirrored(url):
    """The same file on the mirror, if the user chose one."""
    if HF_MIRROR and url.startswith("https://huggingface.co/"):
        return f"https://{HF_MIRROR}/" + url[len("https://huggingface.co/"):]
    return url


def system_proxy():
    """The proxy a VPN/proxy program set in Windows (or HTTPS_PROXY), if any: our own requests would otherwise bypass it."""
    try:
        import urllib.request
        px = urllib.request.getproxies()
        return px.get("https") or px.get("http") or None
    except Exception:
        return None


class Installer:
    """Runs the install steps and reports progress through `emit(dict)`."""

    def __init__(self, app_root, session, emit):
        self.app_root = app_root
        self.session = session
        self.emit = emit
        self.cancelled = False
        self.task = None

    def check(self):
        if self.cancelled:
            raise Cancelled()

    async def step(self, sid, state, **kw):
        await self.emit({"type": "setup", "step": sid, "state": state, **kw})

    async def download(self, url, dest, sid, label, size_hint=0, attempts=40):
        """Resumable download to dest (via dest.part)."""
        url = mirrored(url)
        dest = Path(dest)
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        if size_hint:  # сказать сразу, а не через час: хватит ли места на остаток?
            have = part.stat().st_size if part.exists() else 0
            need = max(0, size_hint - have)
            try:
                free = shutil.disk_usage(str(dest.parent)).free
            except OSError:
                free = None
            if free is not None and free < need * 1.02:
                raise RuntimeError(f"Мало места для «{label}»: нужно ещё {need / 1e9:.1f} ГБ, "
                                   f"а свободно {free / 1e9:.1f} ГБ. Освободите место и повторите.")
        for attempt in range(attempts):  # a long outage: wait and continue from the saved .part file
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                async with self.session.get(url, headers=headers, proxy=system_proxy(), timeout=aiohttp.ClientTimeout(sock_read=120)) as r:
                    if r.status == 416:  # already complete
                        break
                    if r.status not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status} при скачивании {label}")
                    if r.status == 200:
                        have = 0
                    total = have + int(r.headers.get("Content-Length", 0)) or size_hint
                    done, last = have, 0.0
                    with open(part, "ab" if have else "wb") as f:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            self.check()
                            f.write(chunk)
                            done += len(chunk)
                            now = time.monotonic()
                            if now - last > 0.5:
                                last = now
                                await self.step(sid, "progress", label=label, done=done, total=total)
                    break
            except (Cancelled, asyncio.CancelledError):
                raise
            except Exception as e:
                if attempt == attempts - 1:
                    raise RuntimeError(f"Не удалось скачать {label}: {e}")
                why = (str(e) or type(e).__name__)[:90]
                await self.step(sid, "progress", label=f"{label} — обрыв связи ({why}), пробуем снова…")
                await asyncio.sleep(min(30, 3 + attempt * 3))
        got = part.stat().st_size if part.exists() else 0
        if size_hint and got < size_hint * 0.98:
            # недокачанный файл не должен стать «готовой» моделью: ложный 416 делает именно это
            part.unlink(missing_ok=True)
            raise RuntimeError(f"«{label}»: файл пришёл не целиком ({got / 1e9:.1f} из {size_hint / 1e9:.1f} ГБ). "
                               "Повторите скачивание — оно начнётся заново.")
        part.replace(dest)

    async def install_comfy(self, install_dir):
        root = Path(install_dir) / catalog.COMFY_DIRNAME
        if (root / "python_embeded" / "python.exe").exists():
            await self.step("comfy", "done", label="ComfyUI уже установлен")
            return root
        archive = Path(install_dir) / "downloads" / "ComfyUI_windows_portable_nvidia.7z"
        await self.step("comfy", "progress", label=f"Скачиваем ComfyUI {catalog.COMFY_VERSION}")
        await self.download(catalog.COMFY_URL, archive, "comfy", f"ComfyUI {catalog.COMFY_VERSION}", catalog.COMFY_SIZE)
        seven = find_7z(self.app_root)
        if not seven:
            # official standalone 7-Zip extractor (~600 KB)
            seven = Path(install_dir) / "tools" / "7zr.exe"
            for url in ("https://github.com/ip7z/7zip/releases/latest/download/7zr.exe", "https://www.7-zip.org/a/7zr.exe"):
                try:
                    await self.download(url, seven, "comfy", "распаковщик 7-Zip")
                    break
                except RuntimeError:
                    if url.startswith("https://www."):
                        raise
        await self.step("comfy", "progress", label="Распаковываем ComfyUI… (пара минут)", done=0, total=100)
        proc = await asyncio.create_subprocess_exec(
            str(seven), "x", "-y", "-bsp1", f"-o{install_dir}", str(archive),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        buf = b""
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            buf = (buf + chunk)[-64:]
            m = re.findall(rb"(\d+)%", buf)
            if m:
                await self.step("comfy", "progress", label="Распаковываем ComfyUI…", done=int(m[-1]), total=100)
            if self.cancelled:
                proc.kill()
                raise Cancelled()
        if await proc.wait() != 0 or not (root / "python_embeded" / "python.exe").exists():
            raise RuntimeError("Не удалось распаковать ComfyUI")
        archive.unlink(missing_ok=True)
        await self.step("comfy", "done", label="ComfyUI установлен")
        return root

    async def install_nodes(self, comfy_root):
        disable_duplicate_packs(comfy_root / "ComfyUI" / "custom_nodes")
        nodes_dir = comfy_root / "ComfyUI" / "custom_nodes"
        for i, pack in enumerate(catalog.NODE_PACKS):
            target = nodes_dir / pack["name"]
            label = f'Ноды: {pack["name"]}'
            await self.step("nodes", "progress", label=label, done=i, total=len(catalog.NODE_PACKS))
            if target.exists():
                continue
            urls = (f'https://github.com/{pack["repo"]}/archive/{pack["commit"]}.zip',
                    f'https://codeload.github.com/{pack["repo"]}/zip/{pack["commit"]}')  # second address: some networks block only one
            zpath = comfy_root.parent / "downloads" / f'{pack["name"]}.zip'
            for rnd in range(12):
                try:
                    await self.download(urls[rnd % 2], zpath, "nodes", label, attempts=2)
                    break
                except RuntimeError as e:
                    if rnd == 11:
                        raise
                    await self.step("nodes", "progress", label=f"{label} — {str(e)[-120:]}; пробуем другой адрес…")
                    await asyncio.sleep(5)
            with zipfile.ZipFile(zpath) as z:
                top = z.namelist()[0].split("/")[0]
                z.extractall(nodes_dir)
            (nodes_dir / top).rename(target)
            zpath.unlink(missing_ok=True)
            req = target / "requirements.txt"
            if req.exists() and req.read_text(encoding="utf-8", errors="ignore").strip():
                await self.step("nodes", "progress", label=f'{label}: ставим зависимости')
                py = comfy_root / "python_embeded" / "python.exe"
                proc = await asyncio.create_subprocess_exec(
                    str(py), "-s", "-m", "pip", "install", "-r", str(req),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW, env=local_env(comfy_root))
                await proc.wait()
        await self.step("nodes", "done", label="Ноды установлены")

    @staticmethod
    def write_extra_paths(comfy_root, models_dir):
        """Let ComfyUI also see an existing models folder (nothing is copied)."""
        # older setups keep diffusion models in "unet" and text encoders in "clip"
        cats = {"diffusion_models": ["diffusion_models", "unet"], "text_encoders": ["text_encoders", "clip"],
                "vae": ["vae"], "vae_approx": ["vae_approx"], "loras": ["loras"],
                "latent_upscale_models": ["latent_upscale_models"]}
        # one common folder for several programs: the files of the catalog may lie anywhere inside it (any depth, any subfolder),
        # so every folder that holds a wanted file is added to its category; ComfyUI reads subfolders of a listed folder by itself
        wanted = {}
        for variants in catalog.VARIANTS.values():
            for v in variants.values():
                wanted[Path(v["name"].replace("\\", "/")).name.lower()] = v["category"]
        found = {c: [] for c in cats}
        base = Path(models_dir)
        for dirpath, dirnames, filenames in os.walk(base):
            if len(Path(dirpath).relative_to(base).parts) >= 8:
                dirnames[:] = []
            for fn in filenames:
                cat = wanted.get(fn.lower())
                if cat in found and Path(dirpath).as_posix() not in found[cat]:
                    found[cat].append(Path(dirpath).as_posix())
        lines = ["h3studio_user_models:", f"    base_path: {base.as_posix()}"]
        for cat, folders in cats.items():
            lines += [f"    {cat}: |"] + [f"        {f}" for f in folders + found[cat]]
        (comfy_root / "ComfyUI" / "extra_model_paths.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def download_models(self, models_root, missing):
        total = sum(m["size"] for m in missing)
        done_before = 0
        for i, m in enumerate(missing):
            dest = Path(models_root) / m["category"] / Path(m["name"].replace("\\", "/"))
            label = f'{m["title"]} — {m["file"]}'
            base = done_before

            original_emit = self.emit  # emit_overall must call this one: self.emit is replaced below

            async def emit_overall(ev, base=base, idx=i, original_emit=original_emit):
                if ev.get("done") is not None:
                    ev = dict(ev, overall_done=base + ev["done"], overall_total=total, index=idx + 1, count=len(missing))
                await original_emit(ev)

            self.emit = emit_overall
            try:
                await self.download(m["url"], dest, "models", label, m["size"])
            finally:
                self.emit = original_emit
            done_before += m["size"]
        await self.step("models", "done", label="Все модели на месте")


def disable_duplicate_packs(nodes_dir):
    """Two copies of one node pack (the app's folder and, say, «ComfyUI-KJNodes-main» from a manual download) register the same ComfyUI route:
    ComfyUI then does not start («Added route will never be executed»). The extra copy gets «.disabled» (ComfyUI skips such folders)."""
    nodes_dir = Path(nodes_dir)
    if not nodes_dir.is_dir():
        return
    for pack in catalog.NODE_PACKS:
        if not (nodes_dir / pack["name"]).is_dir():
            continue
        for suffix in ("-main", "-master", "-" + pack["commit"], "-" + pack["commit"][:7]):
            extra = nodes_dir / (pack["name"] + suffix)
            if extra.is_dir():
                try:
                    extra.rename(extra.with_name(extra.name + ".disabled"))
                except OSError:
                    pass


# ---------- managed ComfyUI process ----------

def local_env(root):
    """Environment that keeps caches and temp files inside the install folder instead of the Windows profile."""
    cache = Path(root) / "cache"
    dirs = {"PIP_CACHE_DIR": "pip", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
            "HF_HOME": "huggingface", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg", "TEMP": "temp", "TMP": "temp"}
    env = dict(os.environ)
    for var, sub in dirs.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        env[var] = str(cache / sub)
    # On Russian Windows (cp1251) ComfyUI's log cannot print emoji (the upscaler node prints a check mark):
    # the error handler itself then crashes, the worker thread dies and Lipsync hangs forever.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# launch flags the user may add (for experts): only speed and memory ones. Anything that changes the address, the port, the folders
# or opens the network is refused: the program talks to ComfyUI on 127.0.0.1 and reads its output folder.
FLAGS_OK = {
    "--disable-pinned-memory": 0, "--use-sage-attention": 0, "--use-pytorch-cross-attention": 0, "--use-flash-attention": 0,
    "--use-split-cross-attention": 0, "--use-quad-cross-attention": 0, "--fast": "list", "--lowvram": 0, "--novram": 0, "--highvram": 0,
    "--normalvram": 0, "--gpu-only": 0, "--force-fp16": 0, "--force-fp32": 0, "--fp16-unet": 0, "--bf16-unet": 0, "--fp32-unet": 0,
    "--fp8_e4m3fn-unet": 0, "--fp8_e5m2-unet": 0, "--fp16-vae": 0, "--bf16-vae": 0, "--fp32-vae": 0, "--cpu-vae": 0,
    "--disable-smart-memory": 0, "--reserve-vram": "number", "--cache-lru": "number", "--cache-none": 0, "--cache-classic": 0,
    "--async-offload": "number", "--disable-async-offload": 0, "--dont-upcast-attention": 0, "--force-upcast-attention": 0,
    "--disable-xformers": 0, "--force-channels-last": 0, "--cuda-malloc": 0, "--disable-cuda-malloc": 0, "--disable-mmap": 0,
    "--mmap-torch-files": 0, "--deterministic": 0, "--supports-fp8-compute": 0,
}


def parse_flags(text):
    """(list of arguments, error text). Empty text is fine."""
    toks = str(text or "").replace(",", " ").split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t not in FLAGS_OK:
            return None, f"Флаг «{t}» не разрешён: можно только настройки скорости и памяти (например --disable-pinned-memory --use-sage-attention --fast)."
        kind = FLAGS_OK[t]
        out.append(t)
        i += 1
        if kind == "number" or kind == "list":
            vals = []
            while i < len(toks) and not toks[i].startswith("--"):
                vals.append(toks[i]); i += 1
            if kind == "number" and len(vals) > 1:
                return None, f"У флага {t} должно быть одно число."
            for v in vals:
                if not re.fullmatch(r"[0-9.]+", v) if kind == "number" else not re.fullmatch(r"[A-Za-z0-9_]+", v):
                    return None, f"Недопустимое значение «{v}» у флага {t}."
            out.extend(vals)
        elif i < len(toks) and not toks[i].startswith("--"):
            return None, f"У флага {t} не бывает значения («{toks[i]}»)."
    return out, ""


class ComfyProcess:
    def __init__(self, comfy_root, log_path, extra_args=()):
        self.root = Path(comfy_root)
        self.extra_args = list(extra_args)
        self.log_path = log_path
        self.proc = None
        self.port = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        py = self.root / "python_embeded" / "python.exe"
        self.kill_orphans(py)
        self.port = free_port(8188)
        args = [str(py), "-s", "ComfyUI\\main.py", "--windows-standalone-build", "--disable-auto-launch",
                "--listen", "127.0.0.1", "--port", str(self.port), "--preview-method", "auto", *self.extra_args]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:   # keep the previous run's log: after a crash and restart it holds the reason
            if self.log_path.exists() and self.log_path.stat().st_size:
                self.log_path.replace(self.log_path.with_name("comfyui.prev.log"))
        except OSError:
            pass
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(args, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=CREATE_NO_WINDOW, env=local_env(self.root))

    @staticmethod
    def kill_orphans(py):
        """After a crash the app's own ComfyUI may stay in the background and hold comfyui.db and the port.
        Kill only python processes started from THIS install's python_embeded (never the user's own ComfyUI)."""
        exe = str(py).replace("'", "''")
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ $_.ExecutablePath -eq '{exe}' }} | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            time.sleep(1)
        except Exception:
            pass

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            try:  # whole tree: a helper process left behind would keep comfyui.db and the port locked
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass
        self.proc = None

    def log_tail(self, n=15):
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""
