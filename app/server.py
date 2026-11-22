"""Minimax H3 Workspace local server.

Serves the UI and talks to ComfyUI over its HTTP/WebSocket API: uploads inputs,
queues workflows, relays progress and live previews to the browser, and keeps a
small library of finished videos. On first run it either connects to the user's
own ComfyUI or installs one (plus node packs and models) and manages it.

    python app/server.py [--port 8765] [--open]
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import uuid
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

import agent as ag
import websearch
import catalog
import cloud
import customwf
import engine
import monitor
import mp4info
import projects as prj
import reflab
import refmodes
import report
import rhlib
import rhstudio
import speech
import updater
import voicelab
import watchdog
from version import APP_VERSION


def build_stamp():
    """Build date and time shown in the UI: the exe file's own timestamp (frozen build) — tells the user
    which build they are actually running, since the release number changes only at release time."""
    from datetime import datetime
    try:
        p = Path(sys.executable) if getattr(sys, "frozen", False) else Path(__file__)
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return ""


def split_frames(text):
    """One frame per line / numbered item: "1: ...", "Frame 2. ...", "Кадр 3 - ..."."""
    parts = re.split(r"(?m)^\s*(?:frame|кадр)?\s*\d+\s*[:.)\-–]\s*", text.strip(), flags=re.I)
    frames = [p.strip().replace("\n", " ") for p in parts if p.strip()]
    if len(frames) < 2:
        frames = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return frames


def _ssl_context():
    # Windows store + the bundled certifi list: helps PCs with an outdated root-certificate store
    import ssl
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass
    return ctx


class AuthNeeded(Exception):
    pass


class JobStore(dict):
    """prompt_id -> job; every change is written to disk so a closed program or a power cut does not lose the queue."""

    def __init__(self, on_change):
        super().__init__()
        self.on_change = on_change

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.on_change()

    def pop(self, k, *a):
        r = super().pop(k, *a)
        self.on_change()
        return r


def round_size(form, step=32, low=256, high=2048, max_px=2200000):
    """The user's own width/height from the form, rounded to the model's step (None when not given)."""
    w, h = str(form.get("width", "")), str(form.get("height", ""))
    if not (w.isdigit() and h.isdigit()):
        return None
    w, h = int(w), int(h)
    if min(w, h) < 64:
        return None
    k = (max_px / (w * h)) ** 0.5 if w * h > max_px else 1.0
    r = lambda v: min(high, max(low, round(v * k / step) * step))
    return r(w), r(h)


H3IMG_MAX_PX = 20_000_000   # MiniMax H3 Image works at 4-20 MP: its own size limit is much higher than for the other picture models
SUPPORT_URL = "https://pay.cloudtips.ru/p/0e4d4824"
# Immers.Cloud: the author's referral link goes here (a discount for the new user, a bonus for the author). Empty: the plain site is opened.
RENT_REFERRAL = "https://immers.cloud/signup/r/20240927-7463913-939/"
RENT_SITE = "https://immers.cloud/"
import workflows as wf

# frozen builds keep code inside the bundle but user data next to the exe
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
HOME = Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT
UI_DIR = ROOT / "ui"
DATA_DIR = Path(os.environ.get("H3STUDIO_DATA", HOME / "data"))
LONG_DIR = DATA_DIR / "long"   # saved scene states of the long video made scene by scene
LIBRARY_FILE = DATA_DIR / "library.json"
QUEUE_FILE = DATA_DIR / "queue.json"
WORKFLOWS_DIR = DATA_DIR / "workflows"
REFMODES_FILE = DATA_DIR / "refmodes.json"   # Reference Mode cards (see refmodes.py); their files are in REFMODES_DIR
REFMODES_DIR = DATA_DIR / "refmodes"
MAX_SECONDS = 25.0  # H3 itself is made for up to 15 s; longer clips are allowed, quality is the user's call

CLIENT_ID = "h3studio-" + uuid.uuid4().hex[:8]

# ComfyUI binary websocket event types
BIN_PREVIEW = 1
BIN_PREVIEW_META = 4


class Library:
    def __init__(self, path):
        self.path = path
        self.items = []
        if path.exists():
            self.items = json.loads(path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, item_id):
        return next((x for x in self.items if x["id"] == item_id), None)

    def add(self, item):
        self.items.insert(0, item)
        self.save()

    def remove(self, item_id):
        self.items = [x for x in self.items if x["id"] != item_id]
        self.save()


class App:
    def __init__(self, comfy_override=None):
        self.config = engine.Config(DATA_DIR / "config.json")
        self.comfy_override = comfy_override.rstrip("/") if comfy_override else None
        self.session = None
        self.browsers = set()
        self.saved_queue = self.load_queue()
        # ComfyUI шлёт события по шагам тому client_id, с которым задание отправляли.
        self.client_id = next((j['client_id'] for j in self.saved_queue if j.get('client_id')), CLIENT_ID)
        self.recovery_lock = asyncio.Lock()
        self.jobs = JobStore(self.save_queue)  # prompt_id -> job dict
        self.library = Library(LIBRARY_FILE)
        self.refmodes = refmodes.Store(REFMODES_FILE, REFMODES_DIR)
        self.voices = voicelab.Store(DATA_DIR / "voicelab")
        self.reflab = reflab.Store(DATA_DIR / "reflab")
        self.output_dir = None
        self.comfy_online = False
        self.process = None  # engine.ComfyProcess in managed mode
        self._bg = set()  # background tasks (kept so they are not garbage collected)
        self.installer = None
        self.setup_log = []  # last setup events, replayed to a reopened window
        self.model_names = {}  # role -> name as ComfyUI lists it
        self.missing = []  # catalog entries ComfyUI does not see
        self.skills = ag.Skills(ROOT / "skills", DATA_DIR / "skills", self.config)
        self.projects = prj.Projects(DATA_DIR / "projects")
        self.llm = ag.LlamaServer(self.llm_root(), DATA_DIR / "llama.log")
        self.agent_installer = None
        self.hw = None
        self.rh_models = None  # model paths found in the user's RunningHub workflow
        self.monitor = monitor.Monitor()
        self.recent_errors = []  # last problems, for the report to the author

    def llm_root(self):
        if self.config.get("llm_dir"):
            return Path(self.config.get("llm_dir"))
        if self.config.get("engine") == "managed" and self.config.get("install_dir"):
            return Path(self.config.get("install_dir")) / "llm"
        return DATA_DIR.parent / "llm"

    async def llm_variant(self):
        if not self.llm.alive():
            self.llm.root = self.llm_root()  # follows the install folder chosen in setup
            self.llm.custom = self.config.get("llm_custom")
        if self.config.get("llm_variant") in ag.LLM_VARIANTS:
            return self.config.get("llm_variant")
        if self.hw is None:
            self.hw = await asyncio.to_thread(engine.detect_hardware)
        return ag.pick_variant(self.hw)

    async def stop_llm(self):
        if self.llm.alive():
            await asyncio.to_thread(self.llm.stop)

    @property
    def comfy(self):
        if self.comfy_override:
            return self.comfy_override
        if self.process:
            return self.process.url
        return self.config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")

    def comfy_root(self):
        d = self.config.get("install_dir")
        return Path(d) / catalog.COMFY_DIRNAME if d else None

    def choice(self):
        return {**catalog.DEFAULT_CHOICE, **(self.config.get("choice") or {}), **(self.config.get("choice_override") or {})}

    def modes(self):
        modes = list(self.config.get("modes") or catalog.MODE_ROLES)
        if "actor" not in modes and ("refs" in modes or "lipsync" in modes):
            modes.append("actor")  # same models, so set-ups made before this mode existed get it too
        if "swap" not in modes and "refs" in modes:
            modes.append("swap")   # character-swap preset: References models + a 0.16 GB LoRA, fetched by "Докачать"
        return modes + [m for m in getattr(self, "auto_modes", ()) if m not in modes]

    # ---------- ComfyUI connection ----------

    def apply_mirror(self):
        engine.HF_MIRROR = "hf-mirror.com" if self.config.get("hf_mirror") else ""

    async def h_mirror(self, req):
        on = bool((await req.json()).get("on"))
        self.config.update(hf_mirror=on)
        self.apply_mirror()
        return web.json_response({"ok": True, "on": on})

    async def h_live_preview(self, req):
        on = bool((await req.json()).get("on"))
        self.config.update(live_preview=on)
        return web.json_response({"ok": True, "on": on})

    def awake_loop(self):
        """While generations are queued or running (and for ten minutes after the last one), Windows is not allowed to go to sleep:
        a long video made scene after scene overnight must not be stopped by the power plan. The screen may still switch off."""
        import ctypes
        import time
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        last_busy = 0.0
        while True:
            if self.jobs:
                last_busy = time.time()
            keep = time.time() - last_busy < 600
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if keep else 0))
            except Exception:
                return
            time.sleep(15)

    async def start(self, _app):
        if sys.platform == "win32":
            import threading
            threading.Thread(target=self.awake_loop, daemon=True, name="keep-awake").start()
        self.apply_mirror()
        trace = aiohttp.TraceConfig()

        async def _comfy_key(_s, _c, params):
            # key of a rented server: sent only to the ComfyUI address, never to other sites
            key = self.config.get("comfy_key")
            if params.url.host and params.url.host == urllib.parse.urlparse(self.comfy).hostname                     and urllib.parse.urlparse(self.comfy).hostname not in ("127.0.0.1", "localhost", "::1"):
                # proxies of rented servers (RunPod and others) may serve a check page to the default «Python/aiohttp» client
                params.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            if key and params.url.host and params.url.host == urllib.parse.urlparse(self.comfy).hostname:
                params.headers["X-API-Key"] = key
                params.headers["Authorization"] = f"Bearer {key}"
        trace.on_request_start.append(_comfy_key)
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_ssl_context()), trace_configs=[trace])
        if self.config.get("engine") == "managed" and self.config.get("setup_done") and not self.comfy_override:
            if self.missing_node_packs():
                self.packs_task = asyncio.create_task(self.install_missing_packs())
            else:
                self.start_managed()
        self.ws_task = asyncio.create_task(self.comfy_ws_loop())
        self.watchdog_task = asyncio.create_task(self.jobs_watchdog())
        self.poller_task = asyncio.create_task(self.jobs_poller())

    def missing_node_packs(self):
        """Own ComfyUI: node packs from the catalog that are not in its custom_nodes yet (a newer app version may need more)."""
        root = self.comfy_root()
        if not root:
            return []
        nodes_dir = root / "ComfyUI" / "custom_nodes"
        if not nodes_dir.is_dir():
            return []
        return [p["name"] for p in catalog.NODE_PACKS if not (nodes_dir / p["name"]).exists()]

    async def install_missing_packs(self):
        """Put the missing node packs in place, then start ComfyUI. Runs in the background so the app opens at once;
        ComfyUI is started even when the download fails (no internet), only the new sections stay unavailable."""
        async def quiet(_ev):
            pass
        try:
            await engine.Installer(ROOT, self.session, quiet).install_nodes(self.comfy_root())
        except Exception as e:
            print(f"Не удалось докачать ноды ({type(e).__name__}): {e}", file=sys.stderr)
        if not self.process:
            self.start_managed()

    async def stop(self, _app):
        self.ws_task.cancel()
        if self.installer:
            self.installer.cancelled = True
        if self.process:
            await asyncio.to_thread(self.process.stop)
        await asyncio.to_thread(self.llm.stop)
        await self.session.close()

    def start_managed(self):
        root = self.comfy_root()
        engine.disable_duplicate_packs(root / "ComfyUI" / "custom_nodes")
        flags, _err = engine.parse_flags(self.config.get("comfy_flags"))
        self.process = engine.ComfyProcess(root, DATA_DIR / "comfyui.log", flags or [])
        self.process.start()
        self.output_dir = root / "ComfyUI" / "output"

    async def wait_comfy(self, timeout=300):
        """Wait until ComfyUI answers; raise with its log if the process died."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if self.process and not self.process.alive():
                tail = self.process.log_tail()
                hint = ("\n\nПохоже, проблема в драйвере видеокарты. Обновите драйвер NVIDIA (nvidia.com/drivers), "
                        "перезагрузите компьютер и нажмите «Продолжить».") if re.search(r"cuda|nvidia|driver", tail, re.I) else ""
                raise RuntimeError("ComfyUI не запустился:\n" + tail + hint)
            try:
                await self.system_stats()
                return
            except Exception:
                await asyncio.sleep(2)
        raise RuntimeError("ComfyUI долго не отвечает:\n" + (self.process.log_tail() if self.process else ""))

    async def resolve_models(self):
        """Match the chosen model files against what ComfyUI sees (any subfolder)."""
        visible = {}
        for cat in {v["category"] for variants in catalog.VARIANTS.values() for v in variants.values()}:
            try:
                _, lst = await self.comfy_json("GET", f"/models/{cat}")
            except Exception:
                lst = []
            visible[cat] = lst or []
        self.visible = {cat: {n.replace("/", "\\").split("\\")[-1] for n in lst} for cat, lst in visible.items()}
        names, found, entries = {}, set(), {}
        for role, variant in self.choice().items():
            entry = entries[role] = catalog.model_entry(role, variant)
            picked = (self.config.get("model_pick") or {}).get(role) if not self.locked() else None  # a file the user chose by hand (never in «Я новичок»: the set is fixed)
            def seen(file):
                ok = {file, *catalog.FILE_ALIASES.get(file, ())}  # same file under another name (e.g. upscaler from Hugging Face)
                return next((n for n in visible[entry["category"]] if n.replace("/", "\\").split("\\")[-1] in ok), None)
            others = {v["file"] for r, vs in catalog.VARIANTS.items() if r != role for v in vs.values()}
            if picked and picked.replace("/", "\\").split("\\")[-1] in others:  # the file of ANOTHER model of the app: never feed it to this role
                picked = None
            hit = picked if picked in visible[entry["category"]] else seen(entry["file"])
            if not hit and role not in (self.config.get("choice_override") or {}):  # another variant the user already has (v1.3 instead of v2); never for a version picked by hand
                hit = next(filter(None, (seen(v["file"]) for v in catalog.VARIANTS[role].values())), None)
            names[role] = hit or entry["name"]
            if hit:
                found.add(role)
        # modes added after a set-up was made: with your own ComfyUI they switch on by themselves when the files are there
        self.auto_modes = {m for m in ("krea", "kreareal", "qwen", "flux", "h3img", "upscale", "long") if self.config.get("engine") == "external"
                           and all(r in found for r in catalog.MODE_ROLES[m])}
        needed = catalog.roles_for(self.modes())
        missing = [entries[r] for r in needed if r in entries and r not in found]
        self.model_names, self.missing = names, missing
        return names, missing

    async def comfy_json(self, method, path, **kw):
        async with self.session.request(method, self.comfy + path, **kw) as r:
            if r.status == 401:   # the server (or its proxy) wants a login and password
                raise AuthNeeded("сервер просит логин и пароль")
            body = await r.text()
            try:
                data = json.loads(body) if body else None
            except ValueError:
                snippet = re.sub(r"<[^>]+>|\s+", " ", body).strip()[:160]
                raise ValueError(f"сервер ответил не ComfyUI-данными (HTTP {r.status}): «{snippet}»") from None
            return r.status, data

    async def system_stats(self):
        # a rented server behind a proxy (and a VPN) can answer slower than 8 seconds
        remote = urllib.parse.urlparse(self.comfy).hostname not in ("127.0.0.1", "localhost", "::1", None)
        t_stats, t_queue = (30, 20) if remote else (8, 8)
        try:
            status, data = await self.comfy_json("GET", "/system_stats", timeout=aiohttp.ClientTimeout(total=t_stats))
            if status != 200 or not isinstance(data, dict) or "system" not in data:
                raise ValueError("no stats")
        except AuthNeeded:
            raise
        except Exception:
            # some builds hide or slow down /system_stats; a queue answer still means ComfyUI is alive
            status, _ = await self.comfy_json("GET", "/queue", timeout=aiohttp.ClientTimeout(total=t_queue))
            if status != 200:
                raise
            data = {"system": {}, "devices": []}
        argv = data["system"].get("argv", [])
        if "--output-directory" in argv:
            self.output_dir = Path(argv[argv.index("--output-directory") + 1])
        return data

    async def comfy_ws_loop(self):
        while True:
            if not self.config.get("setup_done") and not self.comfy_override:
                await asyncio.sleep(2)
                continue
            ws_url = self.comfy.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
            try:
                async with self.session.ws_connect(ws_url, max_msg_size=0, heartbeat=20) as ws:
                    self.comfy_online = True
                    await self.resolve_models()
                    await self.broadcast({"type": "engine", "online": True})
                    await self.warn_weak_gpu()
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self.on_comfy_event(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await self.on_comfy_binary(msg.data)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if self.comfy_online:
                self.comfy_online = False
                await self.broadcast({"type": "engine", "online": False})
            await asyncio.sleep(2)

    async def warn_weak_gpu(self):
        """The 'max' profile (full ~32 GB Singularity) is meant for 24 GB cards. Say so once per run on a weaker card."""
        if getattr(self, "_weak_warned", False) or self.config.get("choice", {}).get("lipsync_unet") != "full":
            return
        try:
            if self.hw is None:
                self.hw = await asyncio.to_thread(engine.detect_hardware)
            vram = self.hw.get("vram_gb") or 0
        except Exception:
            return
        if 0 < vram < 24:
            self._weak_warned = True
            text = (f"Выбрана самая тяжёлая модель (профиль «Максимум»), а у видеокарты {vram:g} ГБ памяти: "
                    "генерация может зависать на первом шаге. Откройте «Настройки», выберите для Lipsync/Референсов "
                    "облегчённую версию (pruned) и скачайте её.")
            self.note_error(text)
            await self.broadcast({"type": "notice", "text": text})

    async def on_comfy_binary(self, data):
        if len(data) < 8:
            return
        kind = struct.unpack(">I", data[:4])[0]
        if kind == BIN_PREVIEW:
            image = data[8:]
        elif kind == BIN_PREVIEW_META:
            meta_len = struct.unpack(">I", data[4:8])[0]
            meta = json.loads(data[8:8 + meta_len])
            if meta.get("prompt_id") not in self.jobs:
                return
            image = data[8 + meta_len:]
        else:
            return
        if not any(j["state"] == "running" for j in self.jobs.values()):
            return
        for b in list(self.browsers):
            try:
                await b.send_bytes(image)
            except Exception:
                self.browsers.discard(b)

    async def on_comfy_event(self, msg):
        t, d = msg.get("type"), msg.get("data") or {}
        if t == "kj_preview_override":   # animated live preview of KJNodes (no prompt id in it: it belongs to the running job)
            run = next((j for j in self.jobs.values() if j["state"] == "running"), None)
            if run and d.get("image"):
                await self.broadcast({"type": "preview_anim", "prompt_id": run["prompt_id"], "image": d["image"],
                                      "mime": d.get("mime") or "image/jpeg", "step": d.get("step"), "total": d.get("total")})
            return
        job = self.jobs.get(d.get("prompt_id"))
        if not job:
            return
        pid = job["prompt_id"]
        if t in ("execution_start", "executing", "progress"):
            job["last_ev"] = asyncio.get_event_loop().time()
        if t == "execution_start":
            job["state"] = "running"
            await self.send_stage(job, "load")
        elif t == "executing" and d.get("node"):
            stage = job["stages"].get(d["node"])
            if stage:
                await self.send_stage(job, stage, node=d["node"])
        elif t == "progress":
            stage = job["stages"].get(str(d.get("node")), job.get("stage"))
            if stage in ("sample", "sample2"):
                await self.send_stage(job, stage, d.get("value"), d.get("max"), node=str(d.get("node")))
        elif t == "execution_success":
            await self.finish_job(job)
        elif t == "execution_error":
            job["state"] = "error"
            self.note_error(f'Ошибка генерации, узел {d.get("node_type", "")}: {d.get("exception_message", "").strip()}')
            hint = self.model_hint((job.get("meta") or {}).get("mode", ""), d.get("exception_message", ""))
            await self.broadcast({"type": "error", "prompt_id": pid,
                                  "message": f'{d.get("node_type", "")}: {d.get("exception_message", "").strip()}' + hint})
            self.jobs.pop(pid, None)
        elif t == "execution_interrupted":
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
            self.jobs.pop(pid, None)

    async def send_stage(self, job, stage, step=None, total=None, node=None):
        job["stage"] = stage
        msg = {"type": "stage", "prompt_id": job["prompt_id"], "stage": stage, "step": step, "max": total}
        sc = job.get("scenes")
        if sc and node is not None and str(node) in sc.get("map", {}):
            job["scene_now"] = sc["map"][str(node)] + 1
        if sc and job.get("scene_now"):
            msg.update(scene=job["scene_now"], scenes=sc["n"])
        await self.broadcast(msg)

    async def finish_job(self, job):
        pid = job["prompt_id"]
        if self.jobs.get(pid) is not job or job.get('_finishing'):
            return
        job['_finishing'] = True
        try:
            await self._finish_tracked(job)
            self.jobs.pop(pid, None)
        finally:
            job.pop('_finishing', None)

    async def _finish_tracked(self, job):
        pid = job['prompt_id']
        # ComfyUI sends «success» a moment BEFORE it writes the result into its history (it still registers the files); on a busy or
        # slow machine the first look is empty although the video is already in the output folder: look again for up to ~40 seconds
        outputs = {}
        for attempt in range(40):
            try:
                _, hist = await self.comfy_json("GET", f"/history/{pid}")
            except Exception:
                hist = None
            entry = (hist or {}).get(pid, {})
            if (entry.get('status') or {}).get('status_str') == 'error':
                await self.broadcast({'type': 'error', 'prompt_id': pid, 'message': self.history_error(entry['status'])})
                return
            outputs = entry.get('outputs', {})
            if outputs:
                break
            await asyncio.sleep(1)
        await self._collect_outputs(job, outputs)

    async def _collect_outputs(self, job, outputs):
        pid = job['prompt_id']
        tagged = [(nid, f) for nid, out in outputs.items() for lst in out.values() if isinstance(lst, list)
                  for f in lst if isinstance(f, dict) and "filename" in f]
        files = [f for _, f in tagged]
        node_of = {id(f): nid for nid, f in tagged}
        videos = [f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".mkv", ".mov"))]
        videos = [f for f in videos if f.get("type", "output") == "output"] or videos  # not the previews of loaded clips
        lg = (job.get("meta") or {}).get("long")
        if lg and not lg.get("test"):  # a scene of the long video: keep the tail of its latent for the next scene (not for a size test)
            lat = next((f for f in files if f["filename"].lower().endswith(".latent")), None)
            if lat and videos:
                try:
                    async with self.session.get(self.comfy + "/view", params={
                            "filename": lat["filename"], "subfolder": lat.get("subfolder", ""), "type": lat.get("type", "output")}) as r:
                        body = await r.read()
                    if r.status == 200 and body:
                        d = LONG_DIR / lg["run"]
                        d.mkdir(parents=True, exist_ok=True)
                        sfx = "_d" if lg.get("draft") else ""
                        (d / f"state_{lg['part']}{sfx}.latent").write_bytes(body)
                        (d / f"state_{lg['part']}{sfx}.json").write_text(
                            json.dumps({"res": (job.get("meta") or {}).get("res", ""), "kind": lg.get("kind", "")}, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    self.note_error(f"Не удалось сохранить состояние сцены: {e}")
        images = [f for f in files if f["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                  and f.get("type", "output") == "output"]
        if not videos and not images:
            self.note_error("ComfyUI закончил работу, но не вернул ни видео, ни изображение")
            await self.broadcast({"type": "error", "prompt_id": pid, "message": "ComfyUI не вернул ни видео, ни изображение"})
            return
        if videos:
            picked = [videos[-1] if job.get("last_output") else videos[0]]
        else:  # a workflow that makes pictures: every saved picture goes to the library
            picked = images[:12]
            def order(f):  # the glued sheet first, then the frames 1, 2, 3...
                t = str((job.get("titles") or {}).get(node_of.get(id(f)), ""))
                return (0 if t.startswith("Лист") else 1, t)
            picked.sort(key=order)
        items = []
        previous = [it for it in self.library.items if it.get('prompt_id') == pid]
        for f in picked:
            saved = next((it for it in previous if it['file'] == f), None)
            if saved:
                items.append(self.public(saved))
                continue
            item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                        date=datetime.now().isoformat(timespec="seconds"), fav=False, file=f,
                        ext=Path(f["filename"]).suffix.lstrip(".").lower() or "mp4")
            if not videos:
                item["kind"] = "image"
                title = (job.get("titles") or {}).get(node_of.get(id(f)), "")  # "Результат: апскейл (после…)" -> "апскейл"
                label = re.split(r"\s*[(\[]", title.split(":")[-1].strip())[0].strip()[:24]
                if label:
                    item["label"] = label
            meta = f.get("metadata") or {}
            if meta.get("width") and meta.get("height") and not (item.get("mode") == "upscale" and item.get("res")):
                item["res"] = f'{meta["width"]}×{meta["height"]}'
            self.library.add(item)
            await self.archive_item(item)
            items.append(self.public(item))
        await self.broadcast({"type": "done", "prompt_id": pid, "item": items[0], "items": items})
        return items

    async def broadcast(self, payload):
        text = json.dumps(payload, ensure_ascii=False)
        for b in list(self.browsers):
            try:
                await b.send_str(text)
            except Exception:
                self.browsers.discard(b)

    # ---------- helpers ----------

    def public(self, item):
        p = self.file_path(item)
        # a video deleted from disk stays in the list until the user removes it, but is marked
        return {k: v for k, v in item.items() if k != "file"} | {"url": f"/api/file/{item['id']}",
                                                                 "missing": bool(p) and not p.exists()}

    def file_path(self, item):
        if item["file"].get("type") == "local":
            return Path(item["file"]["path"])
        orig = None
        if self.output_dir and item["file"].get("type") == "output":
            orig = self.output_dir / item["file"].get("subfolder", "") / item["file"]["filename"]
        if orig and orig.exists():
            return orig
        local = item.get("local")   # the program's own copy: it stays when ComfyUI's output folder is cleaned
        if local and Path(local).exists():
            return Path(local)
        return orig

    # ---------- own copies of the results (they must not vanish when ComfyUI's output folder is cleaned) ----------

    def default_results_dir(self):
        """Videos\\Minimax H3\\output in the user's folder: it does not depend on where the program lies and survives updates."""
        return Path.home() / "Videos" / "Minimax H3" / "output"

    def results_dir(self):
        d = self.config.get("results_dir")
        return Path(d) if d else self.default_results_dir()

    def keep_copies(self):
        """Own copies of the results. «Я новичок» keeps them as before; with the user's own ComfyUI there is NO copy by default (its output
        folder is the one place of the files, a second copy would only be a duplicate); the user can switch a copy on and choose its folder."""
        v = self.config.get("keep_copies")
        if v is not None:
            return bool(v)
        return self.config.get("engine") != "external"

    async def archive_item(self, item):
        """Copy the finished file of a library item into the program's own results folder."""
        if not self.keep_copies() or item.get("local") and Path(item["local"]).exists():
            return False
        f = item["file"]
        if f.get("type") == "local":
            return False
        try:
            dest_dir = self.results_dir()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{str(item.get('date', ''))[:10].replace('-', '')}_{item['id']}_{f['filename']}"
            src = self.output_dir / f.get("subfolder", "") / f["filename"] if self.output_dir and f.get("type") == "output" else None
            if src and src.exists():
                await asyncio.to_thread(shutil.copy2, src, dest)
            else:   # output folder unknown (own ComfyUI): take the file from ComfyUI itself
                params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
                async with self.session.get(self.comfy + "/view", params=params) as r:
                    if r.status != 200:
                        return False
                    with open(dest, "wb") as out:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            out.write(chunk)
            item["local"] = str(dest)
            self.library.save()
            return True
        except Exception as e:
            self.note_error(f"Не удалось сохранить копию результата: {e}")
            return False

    async def upload_state(self, path, name):
        """Put a saved scene state (.latent) into ComfyUI's input folder, where its LoadLatent node looks for it."""
        data = aiohttp.FormData()
        data.add_field("image", path.read_bytes(), filename=name, content_type="application/octet-stream")
        data.add_field("type", "input")
        data.add_field("overwrite", "true")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать состояние сцены в ComfyUI")
        return res["name"]

    async def upload_input(self, field, rh=None):
        """Put an image/audio/video file into ComfyUI's input/h3studio folder (or upload it to RunningHub)."""
        data = aiohttp.FormData()
        name = f"h3s_{uuid.uuid4().hex[:8]}{Path(field.filename or 'img.png').suffix or '.png'}"
        content_type = field.headers.get("Content-Type", "application/octet-stream")
        raw = await field.read()
        if getattr(self, "_capture", None) is not None:
            self._capture[field.name] = (Path(field.filename or "x.png").suffix.lower() or ".png", raw)
        if rh:
            try:
                return await rh.upload(name, raw, content_type)
            except cloud.CloudError as e:
                raise web.HTTPBadGateway(text=str(e))
        data.add_field("image", raw, filename=name, content_type=content_type)
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось загрузить файл в ComfyUI")
        return f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]

    # ---------- HTTP handlers ----------

    async def h_status(self, _req):
        base = {"configured": bool(self.config.configured or self.comfy_override),
                "engine": self.config.get("engine"), "full_mode": bool(self.config.get("full_mode")), "installing": bool(self.installer),
                "saved_modes": self.config.get("modes") or [],
                # the name of a rented server (never the login or the password): the UI shows plainly which engine is in use
                "server_host": (urllib.parse.urlparse(self.comfy).hostname or "") if self.config.get("engine") == "external"
                               and urllib.parse.urlparse(self.comfy).hostname not in ("127.0.0.1", "localhost", "::1", None) else ""}
        try:
            stats = await self.system_stats()
        except Exception:
            return web.json_response({**base, "online": False})
        dev = (stats.get("devices") or [{}])[0]
        await self.resolve_models()
        missing = {m["role"] for m in self.missing}
        needed = catalog.roles_for(self.modes())
        models = [{"role": role, "title": catalog.ROLE_TITLES[role], "name": self.model_names[role],
                   "category": catalog.VARIANTS[role][variant]["category"],
                   "folder": catalog.VARIANTS[role][variant]["category"], "marks": catalog.marks_text(role),
                   "file": catalog.VARIANTS[role][variant]["file"], "url": engine.mirrored(catalog.VARIANTS[role][variant]["url"]),
                   "size": catalog.VARIANTS[role][variant]["size"], "have": role not in missing,
                   "needed": role in needed, "picked": role in (self.config.get("model_pick") or {}),
                   "modes": [m for m, roles in catalog.MODE_ROLES.items() if role in roles],
                   "variants": [{"key": k, "label": catalog.VARIANT_LABELS.get(role, {}).get(k, k), "size": v["size"],
                                 "current": k == variant, "have": v["file"] in getattr(self, "visible", {}).get(v["category"], ())}
                                for k, v in catalog.VARIANTS[role].items()] if len(catalog.VARIANTS[role]) > 1 else []}
                  for role, variant in self.choice().items() if role in needed]
        choice = self.choice()
        add_modes = {m: round((catalog.plan_size(choice, self.modes() + [m]) - catalog.plan_size(choice, self.modes())) / 1e9, 1)   # gigabytes (plan_size is in bytes)
                     for m in catalog.MODE_ROLES if m not in self.modes()}
        return web.json_response({
            **base, "online": True, "modes": self.modes(), "add_modes": add_modes, "hf_mirror": bool(self.config.get("hf_mirror")), "live_preview": bool(self.config.get("live_preview", True)),
            "offer_modes": [{"mode": m, "gb": add_modes[m]} for m in catalog.OFFER_MODES
                            if m in add_modes and m not in (self.config.get("offered_modes") or [])],
            "gpu": gpu_name(dev.get("name", "")),
            "vram_total": dev.get("vram_total"), "vram_free": dev.get("vram_free"),
            "comfy_version": stats["system"].get("comfyui_version"),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "paths": {"install": self.config.get("install_dir"), "models": self.config.get("models_dir"),
                      "ext_models": self.config.get("ext_models_dir"),
                      "assistant": str(self.llm.root), "data": str(DATA_DIR)},
            "models": models,
        })

    async def nodes_status(self):
        """Which custom nodes the chosen sections need, and whether this ComfyUI knows them."""
        modes = set(self.modes())
        out = []
        for c in catalog.NODE_CHECKS:
            used = [m for m in c["modes"] if m in modes]
            if not used:
                continue
            have = True
            for cls in c["classes"]:
                try:
                    status, data = await self.comfy_json("GET", f"/object_info/{cls}", timeout=aiohttp.ClientTimeout(total=8))
                    have = have and status == 200 and isinstance(data, dict) and cls in data
                except Exception:
                    have = False
            repo = catalog.node_repo(c["pack"])
            out.append({"pack": c["pack"], "have": have, "optional": c["optional"], "why": c["why"], "modes": used,
                        "url": f"https://github.com/{repo}"})
        return out

    async def h_nodes(self, _req):
        try:
            return web.json_response({"nodes": await self.nodes_status()})
        except aiohttp.ClientError:
            return web.json_response({"error": "ComfyUI не отвечает"}, status=503)

    async def h_nodes_install(self, req):
        """Own ComfyUI: put the missing node packs into its custom_nodes (the user gives the folder once; it is remembered).
        ComfyUI has to be restarted afterwards - it only reads custom_nodes at start."""
        import zipfile
        if self.config.get("engine") != "external":
            return web.json_response({"error": "Эта кнопка только для своего ComfyUI. Во встроенном ноды ставятся сами."}, status=400)
        try:
            body = await req.json()
        except Exception:
            body = {}
        raw = str(body.get("dir") or self.config.get("own_comfy_dir") or "").strip().strip('"')
        if not raw:
            return web.json_response({"error": "Укажите папку вашего ComfyUI (где лежит папка custom_nodes)."}, status=400)
        base = Path(raw)
        nodes_dir = next((c for c in (base / "custom_nodes", base / "ComfyUI" / "custom_nodes") if c.is_dir()), None)
        if not nodes_dir:
            return web.json_response({"error": "В этой папке нет custom_nodes. Укажите папку ComfyUI, в которой лежит custom_nodes (у портативной версии это ComfyUI_windows_portable\\ComfyUI)."}, status=400)
        try:
            missing = {n["pack"] for n in await self.nodes_status() if not n["have"]}
        except aiohttp.ClientError:
            return web.json_response({"error": "ComfyUI не отвечает: запустите его, чтобы программа увидела, каких нод не хватает."}, status=503)
        packs = [p for p in catalog.NODE_PACKS if p["name"] in missing]
        py = next((c for c in (nodes_dir.parent.parent / "python_embeded" / "python.exe", nodes_dir.parent / "python_embeded" / "python.exe") if c.exists()), None)
        done, skipped, failed = [], [], []
        for pack in packs:
            target = nodes_dir / pack["name"]
            if target.exists():
                skipped.append(pack["name"])   # the folder is there, ComfyUI just has not loaded it: a restart is needed
                continue
            zpath = nodes_dir.parent / f'_h3_{pack["name"]}.zip'
            ok = False
            for url in (f'https://github.com/{pack["repo"]}/archive/{pack["commit"]}.zip',
                        f'https://codeload.github.com/{pack["repo"]}/zip/{pack["commit"]}'):
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as r:
                        if r.status != 200:
                            continue
                        zpath.write_bytes(await r.read())
                    with zipfile.ZipFile(zpath) as z:
                        top = z.namelist()[0].split("/")[0]
                        z.extractall(nodes_dir)
                    (nodes_dir / top).rename(target)
                    ok = True
                    break
                except Exception as e:
                    print(f"Нода {pack['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            zpath.unlink(missing_ok=True)
            if not ok:
                failed.append(pack["name"])
                continue
            done.append(pack["name"])
            req_txt = target / "requirements.txt"
            if py and req_txt.exists() and req_txt.read_text(encoding="utf-8", errors="ignore").strip():
                proc = await asyncio.create_subprocess_exec(str(py), "-s", "-m", "pip", "install", "-r", str(req_txt),
                                                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                            creationflags=engine.CREATE_NO_WINDOW)
                await proc.wait()
        self.config.update(own_comfy_dir=raw)
        return web.json_response({"ok": not failed, "installed": done, "need_restart": skipped, "failed": failed, "pip": bool(py)})

    async def h_models_files(self, req):
        """Files ComfyUI sees in the folder this model role lives in, for picking one by hand."""
        role = req.query.get("role", "")
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        cat = next(iter(catalog.VARIANTS[role].values()))["category"]
        try:
            _, lst = await self.comfy_json("GET", f"/models/{cat}")
        except Exception:
            raise web.HTTPServiceUnavailable(text="ComfyUI не отвечает")
        return web.json_response({"files": lst or [], "folder": cat})

    async def h_models_download(self, req):
        """Own ComfyUI: download missing model files of the app's catalog into the user's models folder (resumable)."""
        body = await req.json()
        managed = self.config.get("engine") == "managed"
        if managed:  # the app's own ComfyUI: the folder is known
            folder = str(Path(self.config.get("install_dir") or "") / catalog.COMFY_DIRNAME / "ComfyUI" / "models")
        else:
            folder = (body.get("folder") or self.config.get("ext_models_dir") or "").strip()
        p = Path(folder) if folder else None
        if not p or not p.is_dir():
            raise web.HTTPBadRequest(text="Папка не найдена. Укажите папку models вашего ComfyUI.")
        if not any((p / sub).is_dir() for sub in ("diffusion_models", "loras", "vae", "text_encoders", "checkpoints")):
            raise web.HTTPBadRequest(text="Это не похоже на папку models ComfyUI: внутри нет diffusion_models, loras, vae. Выберите папку, где они лежат.")
        if getattr(self, "dl_task", None) and not self.dl_task.done():
            raise web.HTTPConflict(text="Скачивание уже идёт. Дождитесь окончания или остановите его.")
        choice = self.choice()
        roles = [r for r in (body.get("roles") or [body.get("role")]) if r in catalog.VARIANTS and r in choice]
        if not roles:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        if not managed:
            self.config.update(ext_models_dir=str(p))
        entries = [catalog.model_entry(r, choice[r]) for r in roles]

        async def emit(ev):
            await self.broadcast({"type": "mdl", "state": ev.get("state"), "label": ev.get("label"),
                                  "done": ev.get("overall_done", ev.get("done")), "total": ev.get("overall_total", ev.get("total")),
                                  "index": ev.get("index"), "count": ev.get("count")})

        inst = engine.Installer(ROOT, self.session, emit)
        self.dl_installer = inst

        async def run():
            total = sum(e["size"] for e in entries)
            base = 0
            try:
                for i, e in enumerate(entries):
                    dest = p / e["category"] / Path(e["name"].replace("\\", "/"))
                    label = catalog.ROLE_TITLES.get(e["role"], e["role"])

                    async def emit_overall(ev, base=base, idx=i, label=label):
                        if ev.get("done") is not None:
                            ev = dict(ev, overall_done=base + ev["done"], overall_total=total, index=idx + 1, count=len(entries), label=label)
                        await emit(ev)
                    inst.emit = emit_overall
                    await inst.download(e["url"], dest, "models", label, e["size"])
                    base += e["size"]
                await self.resolve_models()
                await self.broadcast({"type": "mdl", "state": "done", "label": "Готово: файлы скачаны и лежат в вашей папке моделей"})
            except engine.Cancelled:
                await self.broadcast({"type": "mdl", "state": "error", "label": "Скачивание остановлено. Скачанное сохранено: нажмите «Скачать» снова, и оно продолжится."})
            except Exception as ex:
                await self.broadcast({"type": "mdl", "state": "error", "label": str(ex)[:300]})
        self.dl_task = asyncio.create_task(run())
        return web.json_response({"ok": True, "folder": str(p), "count": len(entries), "size": sum(e["size"] for e in entries)})

    async def h_models_download_cancel(self, _req):
        inst = getattr(self, "dl_installer", None)
        if inst:
            inst.cancelled = True
        return web.json_response({"ok": True})

    async def h_models_open(self, req):
        """Open the download page of a model in the browser (only links of the app's own catalog)."""
        body = await req.json()
        role = body.get("role")
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        url = catalog.VARIANTS[role][self.choice().get(role) or next(iter(catalog.VARIANTS[role]))]["url"]
        url = engine.mirrored(url)
        webbrowser.open(url)
        return web.json_response({"ok": True, "url": url})

    async def h_models_compat(self, req):
        """Does an own model file fit the mode it is picked in (the speed-up files are made for one family of models)."""
        role, name, mode = req.query.get("role", ""), req.query.get("name", ""), req.query.get("mode", "")
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        if self.hw is None:
            self.hw = await asyncio.to_thread(engine.detect_hardware)
        notes = catalog.compat_notes(mode, role, name or self.model_names.get(role, ""), self.hw.get("compute_cap") or 0)
        missing = {m["role"] for m in self.missing}
        picked = self.config.get("model_pick") or {}
        loras = [{"title": catalog.ROLE_TITLES.get(r, r), "name": self.model_names.get(r, ""), "have": r not in missing,
                  "must": r in catalog.MODE_ROLES.get(mode, []), "mine": r in picked}
                 for r in catalog.MODE_ROLES.get(mode, []) + catalog.MODE_OPTIONAL.get(mode, []) if r.endswith("_lora") and r in self.model_names]
        loras += [{"title": "Своя LoRA", "name": x["file"], "have": True, "must": False, "mine": True, "custom": True, "off": mode in (x.get("skip") or [])}
                  for x in self.unique_loras() if x.get("on", True)]
        return web.json_response({"notes": [{"level": lv, "text": t} for lv, t in notes], "loras": loras,
                                  "picked": picked.get(role), "name": self.model_names.get(role)})

    async def h_models_pick(self, req):
        if self.locked():
            raise web.HTTPForbidden(text="В режиме «Я новичок» модели подобраны и менять их нельзя: так всё работает без сбоев. Чтобы выбрать свою модель, откройте «Настройки», раздел «Для новичков с опытом: хочу использовать свои модели и LoRA» и нажмите «Открыть все настройки…».")
        body = await req.json()
        role, name = body.get("role"), body.get("name") or ""
        if role not in catalog.VARIANTS:
            raise web.HTTPBadRequest(text="Неизвестная модель")
        if name and not catalog.looks_right(role, name):
            foreign = catalog.foreign_marks(role, name)
            if foreign:
                raise web.HTTPBadRequest(text=(
                    f"Этот файл — «{catalog.ROLE_TITLES.get(foreign[0], foreign[0])}», а не «{catalog.ROLE_TITLES.get(role, role)}». "
                    f"Выберите файл для нужной строки, иначе генерация сломается с непонятной ошибкой."))
            if not body.get("force"):
                title = catalog.ROLE_TITLES.get(role, role)
                return web.json_response({"ok": False, "warn": (
                    f"Название файла «{name.replace('/', chr(92)).split(chr(92))[-1]}» не похоже на нужное для «{title}». "
                    f"В названии обычно есть {catalog.marks_text(role)}. Если выбрать не тот файл, генерация сломается с "
                    f"непонятной ошибкой. Всё равно выбрать этот файл?")})
        picks = dict(self.config.get("model_pick") or {})
        if name:
            picks[role] = name
        else:
            picks.pop(role, None)
        self.config.update(model_pick=picks)
        await self.resolve_models()
        return web.json_response({"ok": True})

    def model_hint(self, mode, text):
        """After a ComfyUI error that smells like a wrong model file: say which models of this mode to check."""
        if not re.search(r"shape|size|conv|dtype|mat1|mat2|cannot be|expected|mismatch|tensor|dimension|state_dict|missing key|unexpected key|invalid", text or "", re.I):
            return ""
        roles = catalog.MODE_ROLES.get(mode, [])
        if not roles or not self.model_names:
            return ""
        picked = self.config.get("model_pick") or {}
        sus = [r for r in roles if r in self.model_names and (r in picked or not catalog.looks_right(r, self.model_names[r]))]
        bad = [r for r in sus if not catalog.looks_right(r, self.model_names[r])]
        lines = ["", "Подсказка: похоже, одна из моделей этого раздела не подходит (ошибка про размеры или формат данных)."]
        for r in bad:
            lines.append(f"• «{catalog.ROLE_TITLES.get(r, r)}»: сейчас стоит {self.model_names[r].replace('/', chr(92)).split(chr(92))[-1]}, "
                         f"а в названии нужного файла есть {catalog.marks_text(r)}.")
        for r in sus:
            if r not in bad:
                lines.append(f"• «{catalog.ROLE_TITLES.get(r, r)}»: вы выбирали файл вручную, это "
                             f"{self.model_names[r].replace('/', chr(92)).split(chr(92))[-1]}. Проверьте его.")
        lines.append("Откройте «Настройки» и для нужной строки нажмите «Заменить…», выберите файл «как по умолчанию» или тот, "
                     "что указан в подсказке строки. Если непонятно, нажмите «Отчёт для разработчика» и пришлите файл.")
        return "\n".join(lines)

    async def h_models_variant(self, req):
        """Use another version of the same model (Singularity v1.3 -> v2, a lighter one...). The file is fetched by "Докачать"."""
        body = await req.json()
        role, variant = body.get("role"), body.get("variant") or ""
        if role not in catalog.VARIANTS or (variant and variant not in catalog.VARIANTS[role]):
            raise web.HTTPBadRequest(text="Неизвестная версия модели")
        over = dict(self.config.get("choice_override") or {})
        if variant:
            over[role] = variant
        else:
            over.pop(role, None)
        picks = dict(self.config.get("model_pick") or {})
        picks.pop(role, None)  # a file picked by hand belonged to the old choice
        self.config.update(choice_override=over, model_pick=picks)
        await self.resolve_models()
        entry = catalog.model_entry(role, self.choice()[role])
        have = role not in {m["role"] for m in self.missing}
        return web.json_response({"ok": True, "have": have, "size": entry["size"], "file": entry["file"]})

    async def h_update_check(self, _req):
        info = await updater.check(self.session)
        self.update_info = info if info.get("available") else None
        return web.json_response({**{k: v for k, v in info.items() if k not in ("asset", "sha256")},
                                  "can_update": updater.can_update()})

    async def h_update_progress(self, _req):
        return web.json_response(getattr(self, "update_prog", None) or {"done": 0, "total": 0})

    async def h_update_install(self, _req):
        info = getattr(self, "update_info", None)
        if not info or not updater.can_update():
            raise web.HTTPBadRequest(text="Обновление недоступно")
        if self.jobs:
            raise web.HTTPConflict(text="Сейчас идёт генерация. Дождитесь её конца и нажмите «Обновить» снова.")
        self.update_prog = {"done": 0, "total": int(info.get("size") or 0)}
        try:
            installer = await updater.download(self.session, info, self.update_prog)
        except Exception as e:
            raise web.HTTPBadGateway(text=str(e))
        updater.run(installer)
        # the installer is already running on its own: leave, so it can replace the files and open the app again
        asyncio.get_running_loop().call_later(3, lambda: os._exit(0))
        return web.json_response({"ok": True})

    async def h_system(self, _req):
        """CPU / RAM / GPU / disk readings for the sidebar."""
        where = self.output_dir or self.config.get("install_dir") or HOME
        return web.json_response(await asyncio.to_thread(self.monitor.sample, where))

    # ---------- first-run setup ----------

    async def setup_emit(self, ev):
        if ev.get("state") == "error":
            self.note_error(f'Установка, шаг {ev.get("step")}: {ev.get("label")}')
        if ev.get("state") != "progress" or ev.get("step") != getattr(self, "_last_step", None):
            self.setup_log = (self.setup_log + [ev])[-40:]
        self._last_step = ev.get("step")
        self.setup_last = ev
        await self.broadcast(ev)

    async def h_setup_info(self, req):
        install_dir = req.query.get("install_dir") or self.config.get("install_dir") or engine.default_install_dir()
        modes = [m for m in (req.query.get("modes") or ",".join(catalog.MODE_ROLES)).split(",") if m in catalog.MODE_ROLES]
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        choice, notes, verdict, profile = catalog.choose_models(hw, req.query.get("profile"))
        plan = [catalog.model_entry(r, choice[r]) for r in catalog.roles_for(modes or ["video"])]
        need_gb = round((catalog.COMFY_SIZE * 2.5 + sum(m["size"] for m in plan)) / 1e9)
        # download size of each profile for each set of modes, for the size picker
        sizes = {}
        for p in catalog.PROFILES:
            c = catalog.choose_models(hw, p)[0]
            sizes[p] = {m: round(catalog.plan_size(c, [m]) / 1e9) for m in catalog.MODE_ROLES}
            sizes[p]["selected"] = round(catalog.plan_size(c, modes or ["video"]) / 1e9)
        return web.json_response({
            "hardware": hw, "choice": choice, "notes": notes, "verdict": verdict,
            "profile": profile, "recommended": catalog.recommended_profile(hw), "modes": modes,
            "profiles": {k: {"title": v["title"], "about": v["about"], "sizes": sizes[k]} for k, v in catalog.PROFILES.items()},
            "install_dir": install_dir, "need_gb": need_gb,
            "found_installs": await asyncio.to_thread(engine.find_installs),
            "app_drive": (Path(sys.executable).drive if getattr(sys, "frozen", False) else ""),
            "plan": [{k: m[k] for k in ("role", "title", "file", "size", "variant")} for m in plan],
            "config": {k: self.config.get(k) for k in ("engine", "install_dir", "models_dir", "comfy_url", "setup_done")},
            "installing": bool(self.installer), "log": self.setup_log,
            "last": getattr(self, "setup_last", None),
        })

    async def h_setup_external(self, req):
        body = await req.json()
        url = (body.get("url") or "http://127.0.0.1:8188").strip().rstrip("/")
        if not url.startswith("http"):
            url = "http://" + url
        old, old_key = self.config.get("comfy_url"), self.config.get("comfy_key")
        key = (body.get("key") or "").strip()
        pu = urllib.parse.urlparse(url)
        tok = (urllib.parse.parse_qs(pu.query).get("token") or [""])[0]
        remote = pu.hostname not in ("127.0.0.1", "localhost", "::1")
        base = f"{pu.scheme}://{pu.netloc}"
        login, pwd = (body.get("user") or "").strip(), body.get("password") or ""
        if login and remote:   # login and password of the server: kept inside the address, the HTTP client sends them itself
            host = pu.hostname + (f":{pu.port}" if pu.port else "")
            base = f"{pu.scheme}://{urllib.parse.quote(login, safe='')}:{urllib.parse.quote(pwd, safe='')}@{host}"
        if tok and not key:   # the whole link of a rented server was pasted («http://IP/?token=...»): take the key out of it
            key = tok
            if remote and not pu.port:
                base = f"{pu.scheme}://{pu.hostname}:8000"   # such a link opens the page; the API of the Immers.Cloud image is on port 8000
        url = base + ("/api" if pu.path.rstrip("/").endswith("/api") else "")   # a path or a query in the pasted address is dropped
        self.config.data["comfy_key"] = key
        self.process = None
        stats, last_err, used = None, None, url
        # a rented server may publish the API only under «/api» (as the Immers.Cloud image does): try both
        for cand in ([url] + ([url + "/api"] if remote and not url.rstrip("/").endswith("/api") else [])):
            self.config.data["comfy_url"] = cand
            try:
                stats = await self.system_stats()
                used = cand
                break
            except Exception as e:
                if not isinstance(last_err, AuthNeeded):   # «needs a login» must not be overwritten by the 404 of the other path
                    last_err = e
                if isinstance(e, (asyncio.TimeoutError, TimeoutError)):   # no answer at all: the other path will not help
                    break
        if stats is None:
            self.config.data["comfy_url"] = old
            self.config.data["comfy_key"] = old_key
            if isinstance(last_err, AuthNeeded):
                msg = ("Сервер просит логин и пароль. Впишите их в окошке ниже: это те, что вы вводите в браузере при входе на страницу ComfyUI."
                       if not login else "Сервер не принял этот логин и пароль. Проверьте их и попробуйте ещё раз.")
                return web.json_response({"error": msg, "need_login": True}, status=400)
            hint = " Если сервер арендован: проверьте, что под или сервер запущен, что адрес скопирован целиком (в RunPod порт 8188) и открывается в браузере, и ключ или логин с паролем, если сервис их выдал." if remote else ""
            blocked = remote and bool(re.search(r"HTTP 40[13]|HTTP 1020|Access denied|Attention Required|Cloudflare|blocked", str(last_err), re.I))
            if remote:   # a VPN is the most frequent cause: the protection of the service (Cloudflare at RunPod) blocks some VPN addresses and countries
                hint += ((" ВАЖНО: похоже, сервер вас не пустил из-за VPN. Отключите VPN или выберите в нём другую страну (США обычно работает; Германия и часть стран Европы блокируются) и нажмите «Подключить» ещё раз. Проверка: если адрес не открывается в обычном браузере при включённом VPN, дело в нём.")
                         if blocked else " Если у вас включён VPN: отключите его или смените страну, и попробуйте ещё раз: сервисы аренды часто блокируют адреса VPN.")
            why = ("сервер не ответил за 30 секунд: это сеть до сервера, а не логин" if remote and isinstance(last_err, (asyncio.TimeoutError, TimeoutError))
                   else last_err if isinstance(last_err, ValueError) else type(last_err).__name__)
            return web.json_response({"error": f"ComfyUI не отвечает по адресу {url} ({why}). Откройте этот адрес "
                                      "в браузере: если страница ComfyUI не открывается, проверьте адрес и порт в окне, "
                                      "где запущен ComfyUI." + hint}, status=400)
        url = used
        ws_ok = True
        try:   # progress and «done» of every job come over the WebSocket: with a rented server it may be blocked by its proxy
            async with self.session.ws_connect(url.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}", timeout=8) as _ws:
                pass
        except Exception:
            ws_ok = False
        self.config.update(comfy_key=key)
        modes = [m for m in (body.get("modes") or []) if m in catalog.MODE_ROLES] or list(catalog.MODE_ROLES)
        self.config.update(engine="external", comfy_url=url, setup_done=True, choice=catalog.DEFAULT_CHOICE, modes=modes)
        names, missing = await self.resolve_models()
        try:
            nodes = [n["pack"] for n in await self.nodes_status() if not n["have"] and not n["optional"]]
        except Exception:
            nodes = []
        return web.json_response({"ok": True, "version": stats["system"].get("comfyui_version"), "url": url, "ws_ok": ws_ok,
                                  "missing": [m["file"] for m in missing], "nodes_missing": nodes})

    async def h_mode_offered(self, req):
        mode = (await req.json()).get("mode")
        self.config.update(offered_modes=sorted(set(self.config.get("offered_modes") or []) | {mode}))
        return web.json_response({"ok": True})

    async def h_mode_add(self, req):
        """Own ComfyUI: switch on a mode added after the set-up, so its model files show up in Models (with download buttons)."""
        mode = (await req.json()).get("mode")
        if mode not in catalog.MODE_ROLES:
            return web.json_response({"error": "Неизвестный режим"}, status=400)
        modes = sorted(set(self.modes()) | {mode}, key=list(catalog.MODE_ROLES).index)
        self.config.update(modes=modes)
        await self.resolve_models()
        return web.json_response({"ok": True})

    async def h_setup_install(self, req):
        if self.installer:
            return web.json_response({"error": "Установка уже идёт"}, status=409)
        body = await req.json()
        install_dir = str(Path(body.get("install_dir") or engine.default_install_dir()))
        models_dir = (body.get("models_dir") or "").strip() or None
        if models_dir and not Path(models_dir).is_dir():
            return web.json_response({"error": f"Папка не найдена: {models_dir}"}, status=400)
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        profile = body.get("profile") or self.config.get("profile")
        choice, _, verdict, profile = catalog.choose_models(hw, profile)
        choice = {**choice, **(self.config.get("choice_override") or {})}  # a version the user picked by hand stays
        modes = [m for m in (body.get("modes") or self.modes()) if m in catalog.MODE_ROLES] or ["video"]
        if body.get("add_mode"):
            modes = sorted(set(self.modes()) | {body["add_mode"]}, key=list(catalog.MODE_ROLES).index)
        if self.process:   # files of the old installation must not be in use
            await asyncio.to_thread(self.process.stop)
            self.process = None
        self.config.update(engine="managed", install_dir=install_dir, models_dir=models_dir,
                           choice=choice, profile=profile, modes=modes, setup_done=False)
        self.setup_log = []
        self.installer = engine.Installer(ROOT, self.session, self.setup_emit)
        self.installer.task = asyncio.create_task(self.run_install(install_dir, models_dir))
        return web.json_response({"ok": True})

    async def run_install(self, install_dir, models_dir):
        inst = self.installer
        try:
            Path(install_dir).mkdir(parents=True, exist_ok=True)
            root = await inst.install_comfy(install_dir)
            await inst.install_nodes(root)
            if models_dir:
                inst.write_extra_paths(root, models_dir)
            await inst.step("start", "progress", label="Запускаем ComfyUI…")
            if self.process:
                await asyncio.to_thread(self.process.stop)
            self.start_managed()
            await self.wait_comfy()
            await inst.step("start", "done", label="ComfyUI запущен")
            _, missing = await self.resolve_models()
            # shared files and the video mode first, so the first mode is usable sooner
            order = ["text_encoder", "video_vae", "audio_vae", "preview", "fast_unet"]
            missing.sort(key=lambda m: order.index(m["role"]) if m["role"] in order else len(order))
            if missing:
                gb = sum(m["size"] for m in missing) / 1e9
                await inst.step("models", "progress", label=f"Скачиваем модели: {len(missing)} файлов, {gb:.0f} ГБ")
                await inst.download_models(root / "ComfyUI" / "models", missing)
                await self.resolve_models()
            else:
                await inst.step("models", "done", label="Все модели уже есть")
            self.config.update(setup_done=True)
            await self.setup_emit({"type": "setup", "step": "finish", "state": "done", "label": "Готово!"})
        except engine.Cancelled:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "cancelled",
                                   "label": "Установка остановлена. Скачанное сохранено — можно продолжить позже."})
        except Exception as e:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "error", "label": str(e)})
        finally:
            self.installer = None

    def note_error(self, text):
        self.recent_errors = (self.recent_errors + [f"{datetime.now():%H:%M:%S} {text}"])[-15:]

    async def h_report(self, req):
        """Collect everything the author needs into one text file and show it in Explorer."""
        try:
            page = await req.json()
        except Exception:
            page = {}
        comfy = {"адрес": self.comfy, "режим": self.config.get("engine"), "онлайн": self.comfy_online,
                 "настройка завершена": bool(self.config.get("setup_done"))}
        try:
            st = await self.system_stats()
            sysinfo = st.get("system", {})
            comfy.update({"версия ComfyUI": sysinfo.get("comfyui_version"), "версия PyTorch": sysinfo.get("pytorch_version"),
                          "запуск": " ".join(map(str, sysinfo.get("argv", []))),
                          "устройства": "; ".join(f'{d.get("name")} (всего {round(d.get("vram_total", 0) / 1024 ** 3, 1)} ГБ, '
                                                  f'свободно {round(d.get("vram_free", 0) / 1024 ** 3, 1)} ГБ)' for d in st.get("devices", []))})
            await self.resolve_models()
            needed = catalog.roles_for(self.modes())
            missing = {m["role"] for m in self.missing}
            models = [{"title": catalog.ROLE_TITLES[r], "name": self.model_names.get(r), "have": r not in missing}
                      for r in self.choice() if r in needed]
        except Exception as e:
            comfy["ошибка связи"] = f"{type(e).__name__}: {e}"
            models = []
        hw = await asyncio.to_thread(engine.detect_hardware, self.config.get("install_dir"))
        long_states = []
        try:
            for d in sorted(LONG_DIR.iterdir()) if LONG_DIR.is_dir() else []:
                files = sorted(d.glob("state_*.latent"))
                if not files:
                    continue
                sizes = ", ".join(f"{f.name}: {round(f.stat().st_size / 1024, 1)} КБ" for f in files)
                long_states.append(f"{d.name}: {sizes}")
        except OSError as e:
            long_states.append(f"не удалось прочитать папку: {e}")
        text = await asyncio.to_thread(report.build, {
            "long_states": long_states,
            "home": HOME, "data_dir": DATA_DIR, "hw": hw, "page": page.get("page"), "comfy": comfy, "models": models,
            "config": self.config.data, "errors": self.recent_errors, "toasts": page.get("toasts"),
            "comfy_log": self.process.log_tail(60) if self.process else "",
            "llm_log": self.llm.log_tail(12)})
        folder = DATA_DIR / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"Отчёт для разработчика {datetime.now():%Y-%m-%d %H-%M-%S}.txt"
        path.write_text(text, encoding="utf-8")
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
        except Exception:
            pass
        return web.json_response({"ok": True, "path": str(path), "name": path.name})

    async def h_open_driver(self, _req):
        webbrowser.open("https://www.nvidia.com/Download/index.aspx?lang=ru")
        return web.json_response({"ok": True})

    async def h_search_web(self, req):
        """Open a web search in the user's browser (missing model file / custom node in 'Свои workflow')."""
        body = await req.json()
        q = str(body.get("q", "")).strip()[:200]
        if not q:
            raise web.HTTPBadRequest(text="Пустой запрос")
        webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
        return web.json_response({"ok": True})

    def load_queue(self):
        try:
            data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        week = datetime.now().timestamp() - 7 * 86400
        return [j for j in data if isinstance(j, dict) and j.get("graph") and j.get("saved", 0) > week]

    def queue_origin(self):
        # У встроенного движка порт может меняться после перезапуска; в адресе своего ComfyUI бывает личное.
        root = self.comfy_root() if self.config.get('engine') == 'managed' and not self.comfy_override else None
        source = f'managed:{root}' if root else self.comfy.rstrip('/')
        return hashlib.sha256(source.encode('utf-8')).hexdigest()

    def save_queue(self):
        items = list(self.saved_queue) + [
            {"graph": j["graph"], "stages": j.get("stages") or {}, "meta": j.get("meta") or {}, "last_output": j.get("last_output", False),
             "titles": j.get("titles") or {}, "scenes": j.get("scenes"), "pid": j["prompt_id"],
             "client_id": self.client_id, "origin": j.get('origin'), "saved": datetime.now().timestamp()}
            for j in dict.values(self.jobs) if j.get("graph")]
        try:
            if items:
                QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = QUEUE_FILE.with_suffix('.tmp')
                tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
                tmp.replace(QUEUE_FILE)
            elif QUEUE_FILE.exists():
                QUEUE_FILE.unlink()
        except OSError as e:
            self.note_error(f'Не удалось сохранить очередь: {e}')

    @staticmethod
    def recovery_row(j):
        return {'prompt_id': j.get('prompt_id') or j.get('pid'), 'mode': j['meta'].get('mode'),
                'prompt': (j['meta'].get('prompt') or '')[:120]}

    async def h_queue_saved(self, _req):
        return web.json_response({'count': len(self.saved_queue), 'items': [self.recovery_row(j) for j in self.saved_queue],
                                  'jobs': [dict(self.recovery_row(j), state=j['state'], stage=j.get('stage'))
                                           for j in self.jobs.values() if j.get('recovered')]})

    async def _recovery_get(self, path):
        status, data = await self.comfy_json('GET', path, timeout=aiohttp.ClientTimeout(total=10))
        if status != 200 or not isinstance(data, dict):
            raise ValueError('ComfyUI did not return a snapshot')
        return data

    @staticmethod
    def history_error(status):
        return next((m[1]['exception_message'] for m in status.get('messages', [])
                     if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == 'execution_error'
                     and isinstance(m[1], dict) and m[1].get('exception_message')), 'ComfyUI прервал задание.')

    async def h_queue_resume(self, req):
        """{} reconciles saved local jobs without submitting. {restart_id} opts one missing job into replay.
        Missing and unconfirmed records stay on disk; each partial result is reported independently.
        """
        try:
            body = await req.json() if req.can_read_body else {}
        except ValueError:
            body = None
        if not isinstance(body, dict) or ('restart_id' in body and not isinstance(body['restart_id'], str)):
            return web.json_response({'error': 'Нужен идентификатор сохранённого задания.'}, status=400)
        async with self.recovery_lock:  # два клика или два окна не должны отправить один граф дважды
            return await self._recover_queue(body.get('restart_id'))

    async def _recover_queue(self, restart_id):
        result = {key: [] for key in ('jobs', 'finished', 'missing', 'failed', 'deferred')}
        if not self.saved_queue:
            return web.json_response(result)
        if self.rh():
            return web.json_response({'error': 'Для восстановления переключитесь с RunningHub на исходный ComfyUI.'}, status=409)
        unavailable = (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, AuthNeeded)
        try:
            q = await self._recovery_get('/queue')
            alive = {}
            for key, state in (('queue_pending', 'queued'), ('queue_running', 'running')):
                if not isinstance(q.get(key), list):
                    raise ValueError('invalid queue')
                for row in q[key]:
                    if not isinstance(row, list) or len(row) < 2 or not isinstance(row[1], str):
                        raise ValueError('invalid queue item')
                    alive[row[1]] = state
        except unavailable:
            return web.json_response({'error': 'Не удалось проверить очередь ComfyUI. Сохранённые задания не изменены.'}, status=503)

        for j in list(self.saved_queue):
            row = self.recovery_row(j)
            pid = row['prompt_id']
            origin = self.queue_origin()
            if j.get('origin') and j['origin'] != origin:
                result['deferred'].append(dict(row, message='Запись относится к другому ComfyUI. Верните исходное подключение.'))
                continue
            job = dict(j, prompt_id=pid, recovered=True, origin=origin, stage=None, t0=asyncio.get_running_loop().time())
            if pid in alive:
                self.saved_queue.remove(j)
                self.jobs[pid] = dict(job, state=alive[pid])  # JobStore persists this handoff in one snapshot
                result['jobs'].append(dict(row, state=alive[pid]))
                continue
            try:
                hist = await self._recovery_get(f'/history/{pid}')
                if hist and pid not in hist:
                    raise ValueError('unexpected history response')
                if pid in hist:
                    entry = hist[pid]
                    if not isinstance(entry, dict) or not isinstance(entry.get('status', {}), dict):
                        raise ValueError('invalid history')
                    st = entry.get('status', {})
                    if st.get('status_str') == 'error':
                        result['failed'].append(dict(row, message=self.history_error(st)))
                    elif st.get('completed') or st.get('status_str') == 'success' or (not st and entry.get('outputs')):
                        items = await self._collect_outputs(job, entry.get('outputs') or {})
                        if items:
                            result['finished'].append(dict(row, items=items))
                        else:
                            result['failed'].append(dict(row, message='В результате нет видео или изображения.'))
                    else:
                        result['deferred'].append(dict(row, message='Запись есть в истории, но результат ещё не подтверждён.'))
                        continue
                else:
                    # Запись в Библиотеке может пережить историю ComfyUI: она чистится при перезапуске.
                    items = [self.public(it) for it in self.library.items if it.get('prompt_id') == pid]
                    if items:
                        result['finished'].append(dict(row, items=items))
                    elif pid and restart_id == pid:
                        # Повторяем сохранённый граф как есть. Текущий ComfyUI принимает исходный UUID,
                        # поэтому потерянное подтверждение сверяется следующей проверкой, а не слепым ретраем.
                        await self.stop_llm()
                        status, reply = await self.comfy_json('POST', '/prompt', json={
                            'prompt': j['graph'], 'prompt_id': pid, 'client_id': self.client_id,
                            'extra_data': {'preview_method': 'auto'}})
                        if status != 200 or not isinstance(reply, dict) or not reply.get('prompt_id'):
                            result['deferred'].append(dict(row, message=comfy_error_text(reply)))
                            continue
                        new_id = reply['prompt_id']
                        self.saved_queue.remove(j)
                        self.jobs[new_id] = dict(job, prompt_id=new_id, state='queued')
                        result['jobs'].append(dict(row, prompt_id=new_id, state='queued'))
                        continue
                    else:
                        result['missing'].append(row)
                        continue
                self.saved_queue.remove(j)
                self.save_queue()
            except unavailable:
                result['deferred'].append(dict(row, message='Нет подтверждения от ComfyUI. Задание сохранено.'))
        return web.json_response(result)

    async def h_queue_discard(self, req):
        try:
            body = await req.json() if req.can_read_body else {}
        except ValueError:
            body = None
        if not isinstance(body, dict) or ('ids' in body and not (
                isinstance(body['ids'], list) and all(isinstance(k, str) for k in body['ids']))):
            return web.json_response({'error': 'Нужен список идентификаторов заданий.'}, status=400)
        async with self.recovery_lock:
            ids = set(body['ids']) if 'ids' in body else None  # empty body: legacy explicit discard-all
            self.saved_queue = [j for j in self.saved_queue if ids is not None and j.get('pid') not in ids]
            self.save_queue()
        return web.json_response({'ok': True})

    async def h_support(self, _req):
        """Open the author's donation page (a fixed address, never taken from the request)."""
        webbrowser.open(SUPPORT_URL)
        return web.json_response({"ok": True})

    DMAD_URL = "https://civitai.com/models/2837571?modelVersionId=3391964"   # the exact version «dmad_larryvrhv4_merge» (a fixed address)

    def dmad_dir(self):
        """Where the DMAD LoRA goes: loras\\MinimaxH3 of the app's own ComfyUI; with an own ComfyUI only when its models folder is known."""
        if self.config.get("engine") == "managed":
            root = self.comfy_root()
            return root / "ComfyUI" / "models" / "loras" / "MinimaxH3" if root else None
        ext = (self.config.get("ext_models_dir") or "").strip()
        return Path(ext) / "loras" / "MinimaxH3" if ext else None

    async def h_open_dmad(self, _req):
        """Open the page of the DMAD LoRA on Civitai (the exact version) in the user's browser."""
        webbrowser.open(self.DMAD_URL)
        return web.json_response({"ok": True, "url": self.DMAD_URL})

    async def h_dmad_folder(self, _req):
        """Open (and create) the folder where the DMAD file belongs."""
        d = self.dmad_dir()
        if not d:
            raise web.HTTPBadRequest(text="Папка вашего ComfyUI программе не известна: положите файл в models\\loras\\MinimaxH3 вашего ComfyUI.")
        d.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["explorer", str(d)])
        except Exception:
            pass
        return web.json_response({"ok": True, "path": str(d)})

    async def h_dmad_install(self, _req):
        """«I have downloaded it»: find the DMAD file in the Downloads (or Desktop) folder and move it to the loras folder."""
        d = self.dmad_dir()
        if not d:
            raise web.HTTPBadRequest(text="Папка вашего ComfyUI программе не известна: положите файл в models\\loras\\MinimaxH3 вашего ComfyUI.")
        names = {n.lower() for n in catalog.ACTION_DMAD_FILES}
        names |= {n.lower().replace(".safetensors", "") + " (1).safetensors" for n in catalog.ACTION_DMAD_FILES}   # a browser adds « (1)» to a repeated download

        def work():
            found = []
            for base in (Path.home() / "Downloads", Path.home() / "Desktop"):
                if base.is_dir():
                    found += [f for f in base.iterdir() if f.is_file() and f.name.lower() in names]
            if not found:
                return None
            src = max(found, key=lambda f: f.stat().st_size)   # the big file (3.9 GB) rather than the small «_fro» one
            d.mkdir(parents=True, exist_ok=True)
            dst = d / catalog.ACTION_DMAD_FILES[-1]
            if src.stat().st_size < 3_000_000_000:
                dst = d / src.name.replace(" (1)", "")
            if dst.exists():
                return dst   # already there
            shutil.move(str(src), str(dst))
            return dst
        dst = await asyncio.to_thread(work)
        if not dst:
            raise web.HTTPNotFound(text="Не нашёл файл в папках «Загрузки» и «Рабочий стол». Скачайте minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors на странице Civitai и нажмите эту кнопку снова. Файл на 500 МБ (Turbo) не подходит.")
        return web.json_response({"ok": True, "path": str(dst), "name": dst.name})

    async def h_rent_open(self, _req):
        """Open the rental site (the author's referral link if there is one): a fixed address, never taken from the request."""
        webbrowser.open(RENT_REFERRAL or RENT_SITE)
        return web.json_response({"ok": True})

    async def h_rent_info(self, _req):
        return web.json_response({"ref": bool(RENT_REFERRAL)})

    async def h_clipboard(self, _req):
        """Text from the Windows clipboard (the web view may refuse to let the page read it)."""
        text = ""
        try:
            import ctypes
            u, k = ctypes.windll.user32, ctypes.windll.kernel32
            u.GetClipboardData.restype = ctypes.c_void_p
            k.GlobalLock.restype = ctypes.c_void_p
            k.GlobalLock.argtypes = [ctypes.c_void_p]
            k.GlobalUnlock.argtypes = [ctypes.c_void_p]
            if u.OpenClipboard(0):
                try:
                    h = u.GetClipboardData(13)  # CF_UNICODETEXT
                    if h:
                        p = k.GlobalLock(h)
                        if p:
                            text = ctypes.wstring_at(p)
                            k.GlobalUnlock(h)
                finally:
                    u.CloseClipboard()
        except Exception:
            pass
        return web.json_response({"text": text})

    async def h_setup_cancel(self, _req):
        if self.installer:
            self.installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_setup_reset(self, _req):
        """Back to the first-run wizard; installed files stay where they are."""
        if self.installer:
            return web.json_response({"error": "Сначала остановите установку"}, status=409)
        # Opening the wizard must not undo the set-up: before, «Изменить настройки» cleared «setup_done» and stopped the
        # engine, so closing the program with the wizard open left it «not configured» at the next start.
        # The engine is stopped only when a new installation really starts (h_setup_install).
        return web.json_response({"ok": True})

    async def h_comfy_flags(self, req):
        """The user's own ComfyUI launch flags (experts, the built-in ComfyUI only). A set that stops ComfyUI from starting is dropped again."""
        if self.config.get("engine") != "managed":
            return web.json_response({"error": "Свои флаги запуска можно задать только для встроенного ComfyUI. Свой ComfyUI вы запускаете сами."}, status=400)
        if req.method == "GET":
            return web.json_response({"flags": self.config.get("comfy_flags") or "", "allowed": sorted(engine.FLAGS_OK)})
        b = await req.json()
        text = " ".join(str(b.get("flags", "")).split())
        flags, err = engine.parse_flags(text)
        if err:
            return web.json_response({"error": err}, status=400)
        self.config.update(comfy_flags=text)
        if not b.get("apply"):
            return web.json_response({"ok": True, "saved": text, "applied": False})
        if self.jobs or self.installer or getattr(self, "comfy_updating", False):
            return web.json_response({"ok": True, "saved": text, "applied": False, "note": "Сейчас идёт задание: флаги сохранены и применятся при следующем запуске ComfyUI."})
        if self.process:
            await asyncio.to_thread(self.process.stop)
        self.start_managed()
        try:
            await self.wait_comfy()
        except Exception as e:
            self.config.update(comfy_flags="")   # these flags do not let ComfyUI start: back to the default start
            self.start_managed()
            try:
                await self.wait_comfy()
            except Exception:
                pass
            return web.json_response({"error": "ComfyUI не запустился с этими флагами, они сброшены. " + str(e)[-400:]}, status=500)
        return web.json_response({"ok": True, "saved": text, "applied": True})

    async def h_nodes_fix(self, _req):
        """Built-in ComfyUI: download the node packs it lacks (the ones the program needs), then restart ComfyUI."""
        if self.config.get("engine") != "managed":
            return web.json_response({"error": "Эта кнопка только для встроенного ComfyUI."}, status=400)
        if self.installer or getattr(self, "comfy_updating", False):
            return web.json_response({"error": "Сейчас идёт установка или обновление. Дождитесь конца."}, status=409)
        root = self.comfy_root()
        before = set(self.missing_node_packs())
        self.comfy_updating = True
        try:
            if self.process:
                await asyncio.to_thread(self.process.stop)

            async def quiet(_ev):
                pass
            err = None
            try:
                await engine.Installer(ROOT, self.session, quiet).install_nodes(root)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            self.start_managed()
            try:
                await self.wait_comfy()
            except Exception as e:
                return web.json_response({"error": "ComfyUI не запустился после установки нод. " + str(e)[-400:]}, status=500)
            if err:
                return web.json_response({"error": "Не удалось скачать ноды (проверьте интернет): " + err[-300:]}, status=502)
            return web.json_response({"ok": True, "installed": sorted(before - set(self.missing_node_packs()))})
        finally:
            self.comfy_updating = False

    async def h_comfy_update(self, _req):
        """«Обновить ComfyUI»: runs the portable build's own update_comfyui.bat (the plain one: NOT _stable and NOT
        _and_python_dependencies, which can swap PyTorch for a CPU build and leave ComfyUI unable to start)."""
        if self.config.get("engine") != "managed":
            return web.json_response({"error": "Обновление ComfyUI из программы работает только для встроенного ComfyUI."}, status=400)
        if self.installer or getattr(self, "comfy_updating", False):
            return web.json_response({"error": "Сейчас идёт установка или обновление. Дождитесь конца."}, status=409)
        root = self.comfy_root()
        bat = root / "update" / "update_comfyui.bat"
        if not bat.exists():
            return web.json_response({"error": "Не найден файл update\\update_comfyui.bat во встроенном ComfyUI."}, status=404)
        self.comfy_updating = True
        try:
            before = None
            try:
                before = (await self.system_stats())["system"].get("comfyui_version")
            except Exception:
                pass
            if self.process:
                await asyncio.to_thread(self.process.stop)

            def run():
                # an argument makes the .bat skip its final «pause» (otherwise it would wait for a key press forever)
                p = subprocess.run(["cmd", "/c", "update_comfyui.bat", "nopause"], cwd=bat.parent, capture_output=True,
                                   timeout=1200, creationflags=engine.CREATE_NO_WINDOW, env=engine.local_env(root))
                return p.returncode, (p.stdout or b"").decode("cp866", "replace") + (p.stderr or b"").decode("cp866", "replace")
            try:
                code, out = await asyncio.to_thread(run)
            except Exception as e:
                code, out = -1, f"{type(e).__name__}: {e}"
            self.start_managed()
            try:
                await self.wait_comfy()
            except Exception as e:
                return web.json_response({"error": "Обновление выполнено, но ComfyUI не запустился после него.\n" + str(e),
                                          "log": out[-1500:]}, status=500)
            after = None
            try:
                after = (await self.system_stats())["system"].get("comfyui_version")
            except Exception:
                pass
            return web.json_response({"ok": code == 0, "before": before, "after": after, "log": out[-1500:]})
        finally:
            self.comfy_updating = False

    # ---------- cloud (RunningHub) ----------

    def rh(self):
        """A RunningHub client when cloud generation is switched on, else None."""
        if self.config.get("rh_enabled") and self.config.get("rh_key"):
            return cloud.RunningHub(self.session, self.config.get("rh_key"), self.config.get("rh_host", "ai"))
        return None

    async def h_cloud_get(self, _req):
        c = self.config
        return web.json_response({"enabled": bool(c.get("rh_enabled")), "has_key": bool(c.get("rh_key")),
                                  "host": c.get("rh_host", "ai"), "workflow_id": c.get("rh_workflow_id", ""),
                                  "instance": c.get("rh_instance", "default")})

    async def h_cloud_set(self, req):
        b = await req.json()
        upd = {}
        if b.get("key"):
            upd["rh_key"] = b["key"].strip()
        if b.get("clear_key"):
            upd["rh_key"] = None
            upd["rh_enabled"] = False
        if b.get("host") in cloud.HOSTS:
            upd["rh_host"] = b["host"]
        if "workflow_id" in b:
            upd["rh_workflow_id"] = str(b["workflow_id"]).strip()
        if b.get("instance") in cloud.INSTANCE_TYPES:
            upd["rh_instance"] = b["instance"]
        if "enabled" in b:
            upd["rh_enabled"] = bool(b["enabled"]) and bool(upd.get("rh_key", self.config.get("rh_key")))
        self.config.update(**upd)
        self.rh_models = None
        return await self.h_cloud_get(req)

    # ---------- «Облако RunningHub»: ready workflows (rhlib.py) run on RunningHub, independent of the «cloud engine» switch ----------

    def rh_client(self):
        """A RunningHub client whenever a key is saved (the library does not need the global cloud switch)."""
        key = self.config.get("rh_key")
        return cloud.RunningHub(self.session, key, self.config.get("rh_host", "ai")) if key else None

    async def h_open_link(self, req):
        """Open an author's page of a library workflow in the browser (only the links that are listed in rhlib.py)."""
        url = str((await req.json()).get("url", ""))
        if url not in {c.get("author", {}).get("url") for c in rhlib.CARDS.values() if c.get("author", {}).get("url")}:
            raise web.HTTPBadRequest(text="Эта ссылка не из списка")
        await asyncio.to_thread(webbrowser.open, url)
        return web.json_response({"ok": True})

    async def h_rhlib_id(self, req):
        """The workflow ID of one card of the library (the «Референсы» page and the «Экшн» pages have their own)."""
        b = await req.json()
        cid, wid = str(b.get("card", "")), str(b.get("id", "")).strip()
        if cid not in rhlib.CARDS or (wid and not wid.isdigit()):
            return web.json_response({"error": "ID workflow: только цифры из адреса workflow"}, status=400)
        ids = dict(self.config.get("rh_cards") or {})
        ids[cid] = wid
        self.config.update(rh_cards=ids)
        return web.json_response({"ok": True, "ids": ids})

    async def h_rhlib_get(self, _req):
        c = self.config
        return web.json_response({"cards": rhlib.public(), "has_key": bool(c.get("rh_key")), "host": c.get("rh_host", "ai"),
                                  "instance": c.get("rh_instance", "default"), "ids": c.get("rh_cards") or {},
                                  "default_id": c.get("rh_workflow_id", "")})

    async def h_rhlib_run(self, req):
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        form, files = {}, {}
        async for field in await req.multipart():
            if re.fullmatch(r"(img[0-8]|vid[0-2]|aud[0-2])", field.name or "") and field.filename:
                raw = await field.read()
                suffix = Path(field.filename).suffix or {"vid": ".mp4", "aud": ".mp3"}.get(field.name[:3], ".png")
                try:
                    files[field.name] = await rh.upload(f"h3s_{uuid.uuid4().hex[:8]}{suffix}", raw,
                                                        field.headers.get("Content-Type", "application/octet-stream"))
                except cloud.CloudError as e:
                    return web.json_response({"error": str(e)}, status=400)
            else:
                form[field.name] = (await field.text())[:20000]
        cid = form.get("card", "")
        if cid not in rhlib.CARDS:
            return web.json_response({"error": "Такой карточки нет"}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (только цифры из адреса workflow)"}, status=400)
        need = rhlib.missing(cid, files)
        if need:
            return web.json_response({"error": need}, status=400)
        try:
            graph, seed = rhlib.build(cid, {k: form.get(k) for k in ("prompt", "duration", "aspect", "mp", "speed", "dmad", "seed", "ref_size")}, files)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        return await self.rh_launch(rh, cid, graph, seed, wf_id, form.get("instance"), form.get("prompt", ""), form.get("aspect", "16:9"), form.get("duration"))

    async def rh_launch(self, rh, cid, graph, seed, wf_id, inst, prompt, aspect, duration, final_mp=None, sub=None, extra=None):
        """Send a ready graph of a library card to RunningHub and watch the task."""
        inst = inst if inst in cloud.INSTANCE_TYPES else "default"
        ids = dict(self.config.get("rh_cards") or {})
        ids[cid] = wf_id
        self.config.update(rh_cards=ids, rh_instance=inst, rh_workflow_id=wf_id)
        try:
            bal0 = cloud.balance(await rh.account())   # to tell the user what this very video cost
        except Exception:
            bal0 = None
        try:
            task = await rh.create(graph, wf_id, inst)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        pid = "rh-" + task
        dur = float(duration or 0)
        mp = float(final_mp or graph[rhlib.CARDS[cid]["nodes"]["res"]]["inputs"]["megapixels"])
        meta = {"mode": "rh", "sub": sub or f"{rhlib.CARDS[cid]['title']} · RunningHub", "prompt": str(prompt)[:2000],
                "aspect": aspect, "q": f"{mp:g} МП", "len": dur, "seed": seed, "cloud": True,
                "rh": {"card": rhlib.CARDS[cid]["title"], "gpu": {"default": "24 ГБ", "plus": "48 ГБ", "ultra": "84 ГБ"}.get(inst, inst),
                       "mp": mp, "mult": graph[rhlib.CARDS[cid]["nodes"]["res"]]["inputs"].get("multiple", 32), "task": task, "bal0": bal0,
                       "t0": datetime.now().timestamp(), "wf": wf_id, "out": rhlib.CARDS[cid].get("out")}}
        if extra:
            meta.update(extra)
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": {}, "stage": None, "meta": meta, "task": task}
        asyncio.create_task(self.poll_cloud(rh, pid))
        return web.json_response({"prompt_id": pid, "seed": seed})

    async def h_generate_rhaction(self, req):
        """The «Экшн-видео» page of the RunningHub library: the same form as the local page, the graph is the ready RunningHub workflow."""
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        try:
            form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 6, rh=rh)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (вверху страницы, рядом с ключом)"}, status=400)
        files = {**{f"img{i}": n for i, n in enumerate(images)}, **{f"rvid{i}": n for i, n in enumerate(videos)}, **{f"aud{i}": n for i, n in enumerate(audios)}}
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        dmad, speed = lora_strength(form, "dmad", 0.8), lora_strength(form, "speed", 4.0)   # None: the user switched it off
        mp = round(wf.QUALITY[quality] / 1048576, 3)
        try:
            graph, seed = rhlib.build("action", {"prompt": wf.lipsync_prompt(prompt, len(images), int(form.get("_nown") or 0)), "duration": seconds, "aspect": aspect, "mp": mp,
                                                 "ref_size": ref_size, "seed": seed, "dmad": 0.0 if dmad is None else dmad,
                                                 "speed": 0.0 if speed is None else speed}, files)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        return await self.rh_launch(rh, "action", graph, seed, wf_id, form.get("instance"), prompt, aspect, seconds)

    async def h_generate_rhactionsc(self, req):
        """The «Экшн-видеостудия» page of the RunningHub library: also serves the RunningHub «Видеостудия» and «Создание клипа». The UI sends one scene per task (part=k); scene k>0 gets the last
        frame of the previous finished scene as an extra picture (cont0), because RunningHub keeps no saved state. Without part the scenes are glued into ONE multi-shot prompt."""
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        fields = ("img0", "img1", "img2", "img3", "img4", "aud0", "aud1", "aud2", "rvid0", "rvid1", "rvid2", "cont0", "uvid0", "uvid1", "aref0", "aref1") + tuple(f"simg{i}_{j}" for i in range(80) for j in range(5)) + tuple(f"svoice{i}" for i in range(80)) + tuple(f"samp{i}_{j}" for i in range(80) for j in range(2)) + ("rmaud0", "rmaud1", "rmaud2", "prevvid", "clipsong")
        try:
            form, files, _p, aspect, quality, seed = await self.read_form(req, fields, need_prompt=False, rh=rh)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (вверху страницы, рядом с ключом)"}, status=400)
        cid = form.get("card") if form.get("card") in ("studio", "clip") else "actionsc"   # the «Видеостудия» and «Создание клипа» pages have their own cards (own workflow ID)
        try:
            raw = [(str(x.get("prompt", "")).strip(), float(x.get("sec", 5)), i) for i, x in enumerate(json.loads(form.get("scenes") or "[]"))]
        except (ValueError, TypeError, AttributeError):
            raw = []
        scenes = [(p, max(1.0, min(15.0, sec)), i) for p, sec, i in raw if p][:80]
        if not scenes:
            return web.json_response({"error": "Напишите хотя бы одну сцену"}, status=400)
        part = form.get("part", "").strip()
        one = None
        if part.isdigit():   # one scene alone (the «Сделать эту сцену» button): every scene is a separate task on RunningHub, nothing is continued
            one = next((x for x in scenes if x[2] == int(part)), None)
            if not one:
                return web.json_response({"error": "Эта сцена пустая: напишите для неё текст"}, status=400)
        total = one[1] if one else sum(sec for _, sec, _ in scenes)
        if total > 25.5:
            return web.json_response({"error": f"На RunningHub весь ролик одним заданием не длиннее 25 секунд (у вас {total:.0f} с). Сократите сцены."}, status=400)
        parts, t = [], 0.0
        subj = {}   # RefMod lines "<Subject N> is ..." sit once at the top of the glued prompt, not in front of every scene
        cut = []
        for p, sec, i in ([one] if one else scenes):
            if p.startswith("<Subject "):
                head, _, p = p.partition("\n\n")
                for chunk in re.split(r"(?=<Subject \d+> is )", head):
                    mm = re.match(r"<Subject (\d+)>", chunk)
                    if mm:
                        subj.setdefault(mm.group(1), chunk.strip())
            cut.append((p, sec, i))
        defs_line = " ".join(subj[k] for k in sorted(subj))
        for k, (p, sec, _i) in enumerate(cut):
            parts.append(p if k == 0 else f"[Shot {k + 1}] At {int(t // 60):02d}:{t % 60:06.3f}, a new plan: {p}")
            t += sec
        images = [files[f"img{i}"] for i in range(5) if f"img{i}" in files]
        images += [files[f"simg{i}_{j}"] for _p, _s, i in ([one] if one else scenes) for j in range(5) if f"simg{i}_{j}" in files]
        studio_cont = bool(cid in ("studio", "clip") and one and files.get("prevvid"))   # the Video studio continues by the previous CLIP (like the local studio), not by a picture
        cont = None if studio_cont else (files.get("cont0") if one else None)   # the last frame of the previous finished scene: this scene starts from it
        images = images[:9 - (1 if cont else 0)]
        if cont:
            images.append(cont)
            parts[0] = (f"<Picture {len(images)}> is the last frame of the previous shot: this shot starts exactly from this frame, the same place, "
                        "the same characters, clothes and light, and continues the action without a cut.\n\n") + parts[0]
        rvids = [files[f"rvid{i}"] for i in range(3) if f"rvid{i}" in files]
        n_rm = len(rvids)
        uvids = [files[f"uvid{i}"] for i in range(2) if f"uvid{i}" in files][:max(0, 3 - n_rm)]   # the user's own camera videos: <Video N> after the Reference Mode ones
        rvids = rvids + uvids
        cam = list(range(n_rm + 1, n_rm + 1 + len(uvids)))
        if not images and not rvids and cid not in ("studio", "clip"):   # the Video studio and the clip may go by text alone, as the local ones do
            return web.json_response({"error": "Добавьте хотя бы одно изображение героя или локации"}, status=400)
        rfiles = {f"img{i}": n for i, n in enumerate(images)}
        rfiles.update({f"rvid{i}": n for i, n in enumerate(rvids)})
        auds = [files[f"aud{i}"] for i in range(3) if files.get(f"aud{i}")]
        n_voice = 0
        n_rmv = 0
        labs = []
        if cid == "studio":   # the voice of the scene and the sound references go in as audio references (RunningHub cannot lock a voice like the local graph does: the model takes the timbre)
            own_s = [(files[f"samp{one[2]}_{j}"], form.get(f"samplab{one[2]}_{j}", "")) for j in range(2) if one and files.get(f"samp{one[2]}_{j}")]   # voice samples of this scene (a dialogue: man + woman)
            smp = own_s or [(files[f"aref{i}"], "") for i in range(2) if files.get(f"aref{i}")]
            vo_a = [files[f"svoice{one[2]}"]] if one and files.get(f"svoice{one[2]}") and not (one and cid == "studio") else []   # the Video studio locks the voice in the graph (exact words), the other pages give it as a reference
            extra_a = vo_a + [x[0] for x in smp]
            labs = [""] * len(vo_a) + [x[1] for x in smp]
            n_rmv = len(auds)   # voices of the Reference Mode heroes are <Audio 1..>, ours come after them
            auds = (auds + extra_a)[:3]
            n_voice = max(0, len(auds) - n_rmv)
        for i, n in enumerate(auds):
            rfiles[f"aud{i}"] = n
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        dmad, speed = lora_strength(form, "dmad", 0.8), lora_strength(form, "speed", 4.0)
        width, height = wf.resolution(aspect, quality, form)
        amodel = "hybrid" if form.get("amodel") == "action" else "sing"   # the same two buttons as in the local studio
        text = (defs_line + "\n\n" if defs_line else "") + "\n\n".join(parts)
        if cid in ("studio", "clip") and one:   # the SAME scene graph as the local Video studio (see rhstudio.py), models and attention nodes of the RunningHub graph
            try:
                sfiles = dict(rfiles)
                if files.get(f"svoice{one[2]}"):
                    sfiles["voice"] = files[f"svoice{one[2]}"]
                if cid == "clip" and files.get("clipsong"):
                    sfiles["song"] = files["clipsong"]
                    sfiles["song_lead"] = float(form.get("clipsong_lead") or 0)
                if studio_cont:
                    sfiles["prev"] = files["prevvid"]
                    sfiles["prev_frames"] = int(float(form.get("prevframes") or 0))
                graph, seed = rhstudio.build(rhlib.load("studio"), {"prompt": wf.lipsync_prompt(text, len(images), 0, cam, n_voice, labs, n_rmv),
                                                                    "width": width, "height": height, "seed": seed, "seconds": one[1], "pika": form.get("apika") == "1",
                                                                    "ref_size": "match",
                                                                    **({"dmad": dmad or 0.0, "speed": speed or 0.0} if cid == "clip" else {})}, sfiles)   # the clip may use DMAD and Speed Slider (off unless chosen)   # like the local studio
            except (ValueError, KeyError) as e:
                return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
            return await self.rh_launch(rh, cid, graph, seed, wf_id, form.get("instance"), text, aspect, total, final_mp=width * height / 1048576,
                                        sub=f"сцена {int(part) + 1} из {len(raw)} · RunningHub",
                                        extra={"long": {"run": form.get("run", ""), "part": int(part), "n": len(raw), "draft": False, "test": False, "kind": "action"}})
        try:
            graph, seed = rhlib.build(cid, {"prompt": wf.lipsync_prompt(text, len(images), 0, cam, n_voice, ([""] * 0 + labs) if cid == "studio" else (), n_rmv if cid == "studio" else 0), "width": width, "height": height,
                                                   "frames": wf.frames(total), "model": amodel, "pika": form.get("apika") == "1",
                                                   "ref_size": ref_size, "seed": seed, "dmad": 0.0 if dmad is None else dmad,
                                                   "speed": 0.0 if speed is None else speed}, rfiles)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        return await self.rh_launch(rh, cid, graph, seed, wf_id, form.get("instance"), text, aspect, total, final_mp=width * height / 1048576,
                                    sub=(f"экшн · сцена {int(part) + 1} из {len(raw)} · RunningHub" if one else None),
                                    extra=({"long": {"run": form.get("run", ""), "part": int(part), "n": len(raw), "draft": False, "test": False, "kind": "action"}} if one else None))

    async def h_generate_rhlipsync(self, req):
        """The «Lipsync» page of the RunningHub library: the same form as the local page (the audio clip is already cut there)."""
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        ref_fields = tuple(f"ref{i}" for i in range(9))
        try:
            form, files, prompt, aspect, quality, seed = await self.read_form(req, ref_fields + ("audio", "rvid0", "rvid1", "rvid2"), rh=rh)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (вверху страницы, рядом с ключом)"}, status=400)
        if "audio" not in files:
            return web.json_response({"error": "Загрузите аудио"}, status=400)
        refs = [files[f] for f in ref_fields if f in files]
        rvids = [files[f"rvid{i}"] for i in range(3) if f"rvid{i}" in files]
        if not refs and not rvids:
            return web.json_response({"error": "Добавьте хотя бы один референс — лицо персонажа"}, status=400)
        try:
            seconds = float(form.get("audio_len", 0))
        except ValueError:
            seconds = 0.0
        if not 1 <= seconds <= 15:
            return web.json_response({"error": "Фрагмент аудио должен быть от 1 до 15 секунд"}, status=400)
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        scale = wf.lipsync_scale(quality)
        final_mp = round(wf.QUALITY[quality] / 1048576, 3)
        first_mp = round(final_mp / scale ** 2, 3)   # the first pass is small, the second one enlarges it by `scale`
        realism = lora_strength(form, "realism", None)   # off unless the user switched it on
        rfiles = {**{f"img{i}": n for i, n in enumerate(refs)}, **{f"rvid{i}": n for i, n in enumerate(rvids)}, "audio": files["audio"]}
        try:
            graph, seed = rhlib.build("lipsync", {"prompt": wf.lipsync_prompt(prompt, len(refs)), "duration": seconds, "aspect": aspect, "mp": first_mp,
                                                  "ref_size": ref_size, "seed": seed, "scale": scale, "realism": 0.0 if realism is None else realism,
                                                  "accel": form.get("rh_accel") == "1"}, rfiles)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        return await self.rh_launch(rh, "lipsync", graph, seed, wf_id, form.get("instance"), prompt, aspect, seconds, final_mp)

    async def h_generate_rhvideo(self, req):
        """The «Image-to-Video» page of the RunningHub library (Fast Video): text, first frame, first + last frame, a loop."""
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        try:
            form, files, prompt, aspect, quality, seed = await self.read_form(req, ("first", "last"), rh=rh)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (вверху страницы, рядом с ключом)"}, status=400)
        seconds = min(MAX_SECONDS, max(3.0, float(form.get("duration", 5))))
        first = files.get("first")
        last = files.get("last") if first else None
        loop = bool(first) and form.get("loop") == "1"
        mp = round(wf.QUALITY[quality] / 1048576, 3)
        try:
            graph, seed = rhlib.build("video", {"prompt": prompt, "duration": seconds, "aspect": aspect, "mp": mp, "seed": seed, "loop": loop},
                                      {k: v for k, v in (("first", first), ("last", last)) if v})
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        sub = "Зацикленное видео" if loop else "Кадр → кадр" if last else "Изображение → видео" if first else "Текст → видео"
        return await self.rh_launch(rh, "video", graph, seed, wf_id, form.get("instance"), prompt, aspect, seconds, sub=sub + " · RunningHub")

    async def h_generate_rhrefs(self, req):
        """The «Референсы» page of the RunningHub library: the same form as the local page, the graph is the ready RunningHub workflow."""
        rh = self.rh_client()
        if not rh:
            return web.json_response({"error": "Сначала вставьте API-ключ RunningHub и нажмите «Сохранить»"}, status=400)
        try:
            form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 9, rh=rh)
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        wf_id = form.get("wf", "").strip()
        if not wf_id.isdigit():
            return web.json_response({"error": "Впишите ID workflow RunningHub (вверху страницы, рядом с ключом)"}, status=400)
        files = {**{f"img{i}": n for i, n in enumerate(images)}, **{f"vid{i}": n for i, n in enumerate(videos)}, **{f"aud{i}": n for i, n in enumerate(audios)}}
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        steps = int(form["steps"]) if str(form.get("steps", "")).isdigit() else 6   # as the local page: 4 fast, 6 balance, 8 dynamics
        mp = round(wf.QUALITY[quality] / 1048576, 3)
        try:
            graph, seed = rhlib.build("refs", {"prompt": wf.lipsync_prompt(prompt, len(images), int(form.get("_nown") or 0)), "duration": seconds, "aspect": aspect,
                                               "mp": mp, "ref_size": ref_size, "seed": seed, "steps": steps,
                                               "accel": form.get("rh_accel") == "1"}, files)
        except (ValueError, KeyError) as e:
            return web.json_response({"error": f"Неверные значения: {e}"}, status=400)
        return await self.rh_launch(rh, "refs", graph, seed, wf_id, form.get("instance"), prompt, aspect, seconds)

    async def rh_finish(self, rh, item, video):
        """The facts about one finished RunningHub video: time, cost (Wallet before/after), card, size, task number."""
        info = item["rh"]
        info["secs"] = round(datetime.now().timestamp() - info.pop("t0", datetime.now().timestamp()))
        mp = info.get("mp") or 1.0
        m = info.get("mult") or 32
        try:
            aw, ah = (float(x) for x in str(item.get("aspect") or "16:9").split(":"))
        except ValueError:
            aw, ah = 16.0, 9.0
        item["res"] = f'{round((mp * 1048576 * aw / ah) ** 0.5 / m) * m}×{round((mp * 1048576 * ah / aw) ** 0.5 / m) * m}'
        # whatever RunningHub itself says about this task's price/time (field names are not documented: keep the short plain ones)
        info["billing"] = {k: v for k, v in video.items()
                           if isinstance(v, (int, float, str)) and len(str(v)) < 40 and "url" not in k.lower() and k.lower() not in ("filetype", "nodeid")}
        await asyncio.sleep(3)   # the payment is booked a moment after the task ends
        try:
            bal1 = cloud.balance(await rh.account())
        except Exception:
            bal1 = None
        info["bal1"] = bal1
        if info.get("bal0") is not None and bal1 is not None:
            info["cost"] = round(info["bal0"] - bal1, 4)

    async def h_setup_cloud(self, _req):
        """First-run choice "cloud only": nothing installed locally."""
        if not (self.config.get("rh_key") and self.config.get("rh_enabled")):
            return web.json_response({"error": "Сначала сохраните ключ и включите облако"}, status=400)
        self.config.update(engine="cloud", setup_done=True, modes=list(catalog.MODE_ROLES))
        return web.json_response({"ok": True})

    async def h_cloud_test(self, _req):
        key = self.config.get("rh_key")
        if not key:
            return web.json_response({"error": "Вставьте API-ключ"}, status=400)
        rh = cloud.RunningHub(self.session, key, self.config.get("rh_host", "ai"))
        try:
            acc = await rh.account()
            names = await rh.model_names(self.config.get("rh_workflow_id")) if self.config.get("rh_workflow_id") else set()
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "account": acc, "models": sorted(names)[:40]})

    async def queue_cloud(self, rh, graph, stages, meta):
        if not self.config.get("rh_workflow_id"):
            return web.json_response({"error": "Укажите ID любого вашего workflow на RunningHub (раздел «Настройки» → «Облако»)"},
                                     status=400)
        if getattr(self, "rh_models", None) is None:
            try:
                self.rh_models = await rh.model_names(self.config.get("rh_workflow_id"))
            except cloud.CloudError:
                self.rh_models = set()
        cloud.remap_models(graph, self.rh_models)
        try:
            task = await rh.create(graph, self.config.get("rh_workflow_id"), self.config.get("rh_instance", "default"))
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        pid = "rh-" + task
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": {}, "stage": None,
                          "meta": dict(meta, cloud=True), "task": task}
        asyncio.create_task(self.poll_cloud(rh, pid))
        return web.json_response({"prompt_id": pid, "seed": meta["seed"]})

    async def poll_cloud(self, rh, pid):
        job = self.jobs.get(pid)
        last = None
        try:
            while pid in self.jobs:
                status = await rh.status(job["task"])
                if status != last:
                    last = status
                    stage = {"QUEUED": "cloud_queued", "RUNNING": "cloud_run"}.get(status)
                    if stage:
                        await self.send_stage(job, stage)
                if status == "SUCCESS":
                    outs = await rh.outputs(job["task"]) or []
                    vids = [o for o in outs if str(o.get("fileType", "")).lower() in ("mp4", "webm", "mov", "mkv")
                            or str(o.get("fileUrl", "")).lower().split("?")[0].endswith((".mp4", ".webm", ".mov"))]
                    out_node = (job["meta"].get("rh") or {}).get("out")
                    video = (next((o for o in vids if str(o.get("nodeId")) == str(out_node)), None) or (vids[-1] if vids else None)) if out_node else (vids[0] if vids else None)
                    if not video:
                        raise cloud.CloudError("RunningHub не вернул видео")
                    dest = DATA_DIR / "cloud" / f'{pid}.mp4'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    async with self.session.get(video["fileUrl"]) as r:
                        dest.write_bytes(await r.read())
                    self.jobs.pop(pid, None)
                    item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                                date=datetime.now().isoformat(timespec="seconds"), fav=False,
                                file={"type": "local", "path": str(dest), "filename": dest.name})
                    if video.get("consumeCoins"):
                        item["sub"] = f'{item.get("sub", "")} · ☁ {video["consumeCoins"]} RH'.strip(" ·")
                    if isinstance(item.get("rh"), dict):
                        await self.rh_finish(rh, item, video)
                    self.library.add(item)
                    await self.broadcast({"type": "done", "prompt_id": pid, "item": self.public(item)})
                    return
                if status == "FAILED":
                    await rh.outputs(job["task"])  # raises with the failure reason
                    raise cloud.CloudError("Задание на RunningHub завершилось ошибкой")
                await asyncio.sleep(5)
        except Exception as e:
            if pid in self.jobs:
                self.jobs.pop(pid, None)
                await self.broadcast({"type": "error", "prompt_id": pid, "message": str(e)})

    async def read_form(self, req, file_fields, need_prompt=True, rh=None):
        form, files = {}, {}
        rh = rh or self.rh()
        async for field in await req.multipart():
            if field.name in file_fields and field.filename:
                files[field.name] = await self.upload_input(field, rh)
            else:
                form[field.name] = await field.text()
        prompt = form.get("prompt", "").strip()
        aspect, quality = form.get("aspect", "16:9"), form.get("quality", "720p")
        if not prompt and need_prompt:
            raise web.HTTPBadRequest(text="Напишите промпт")
        if aspect not in wf.ASPECTS or quality not in wf.QUALITY:
            raise web.HTTPBadRequest(text="Неизвестный формат или качество")
        seed = int(form["seed"]) if form.get("seed", "").isdigit() else random.randrange(2 ** 50)
        return form, files, prompt, aspect, quality, seed

    async def queue(self, graph, stages, meta, last_output=False, titles=None):
        scenes = stages.pop("__scenes__", None)   # which sampler is which scene (long video), for the progress text
        rh = self.rh()
        if rh:
            return await self.queue_cloud(rh, graph, stages, meta)
        await self.stop_llm()  # the agent's model and H3 do not fit in VRAM together
        mode_key = (meta or {}).get("mode", "")
        # the animated live preview («see what comes out, stop in time») in the video modes that have none of their own; one switch in Settings
        if mode_key in ("video", "lipsync", "refs", "actor", "swap") and self.config.get("live_preview", True) \
                and not any(n.get("class_type") == "ModelPreviewOverrideKJ" for n in graph.values()):
            try:
                wf.add_preview(graph, await self.preview_opts("long"))
            except Exception as e:
                self.note_error(f"Живое превью не включилось: {e}")
        same = mode_key == getattr(self, "last_mode", None) and mode_key in ("krea", "kreareal", "qwen", "flux")
        self.last_mode = mode_key
        # nothing is running or waiting: drop the previous job's models and cache so each job starts clean.
        # Not between pictures of the same mode: the same big models would just be loaded from disk again.
        if not self.jobs and not same:
            try:
                await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                await asyncio.sleep(1.5)
                gpu = await asyncio.to_thread(self.monitor.gpu)
                used = (gpu or {}).get("vram_used_gb")
                if used is not None and used > 2.5:  # right after /free ComfyUI itself holds well under this
                    await self.broadcast({"type": "notice", "text": (
                        f"Перед заданием занято {used:g} ГБ видеопамяти. Скорее всего, её держат другие программы "
                        "(браузер, Adobe, игры). Закройте их, иначе генерация может идти медленнее.")})
            except Exception:
                pass
        status, res = await self.comfy_json("POST", "/prompt", json={
            "prompt": graph, "client_id": self.client_id,
            "extra_data": {"preview_method": "auto"}})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": stages, "stage": None, "meta": meta, "last_output": last_output, "titles": titles or {},
                          "graph": graph, "scenes": scenes, "origin": self.queue_origin(),
                          "t0": asyncio.get_event_loop().time()}
        return web.json_response({"prompt_id": pid, "seed": meta["seed"], "queue_number": res.get("number")})

    async def comfy_old(self):
        """True when this ComfyUI is older than 0.45.0 (no MiniMax H3 Image / RelayVSR nodes)."""
        try:
            v = (await self.system_stats())["system"].get("comfyui_version") or ""
        except Exception:
            return False
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
        return bool(m) and int(m.group(1)) == 0 and int(m.group(2)) < 45

    async def need_models(self, mode):
        if mode == "h3img" and not self.rh() and await self.comfy_old():
            how = ("Откройте «Настройки», блок «Движок», и нажмите «Обновить ComfyUI»." if self.config.get("engine") == "managed"
                   else "Закройте программу и ComfyUI, откройте в папке ComfyUI папку update и запустите update_comfyui.bat (только его), затем запустите снова.")
            raise web.HTTPBadRequest(text="MiniMax H3 Image нужен ComfyUI версии 0.45.0 или новее: ваш старше, поэтому нужного узла в нём нет. " + how)
        if self.rh():  # models live on RunningHub
            if not self.model_names:
                self.model_names = {r: catalog.VARIANTS[r][v]["name"] for r, v in self.choice().items()}
            return
        if mode not in self.modes():
            raise web.HTTPBadRequest(text="Этот режим не установлен. Добавьте его в разделе «Настройки».")
        if not self.model_names:
            await self.resolve_models()
        lost = [m["file"] for m in self.missing if m["role"] in catalog.MODE_ROLES[mode]]
        if lost:  # the answer may be stale (ComfyUI was still starting, files finished downloading later): ask ComfyUI again before refusing
            await self.resolve_models()
            lost = [m["file"] for m in self.missing if m["role"] in catalog.MODE_ROLES[mode]]
        if lost:
            raise web.HTTPBadRequest(text="ComfyUI не видит модели: " + ", ".join(lost)
                                     + ". Откройте раздел «Настройки».")

    # ---------- the user's own workflows (API format), for those with their own ComfyUI ----------

    def wf_path(self, wid):
        if not wid.isalnum():
            raise web.HTTPNotFound()
        return WORKFLOWS_DIR / f"{wid}.json"

    def wf_load(self, wid):
        p = self.wf_path(wid)
        if not p.exists():
            raise web.HTTPNotFound(text="Workflow не найден")
        return json.loads(p.read_text(encoding="utf-8"))

    async def h_wf_list(self, _req):
        items = []
        for p in sorted(WORKFLOWS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if WORKFLOWS_DIR.exists() else []:
            try:
                items.append({"id": p.stem, "name": json.loads(p.read_text(encoding="utf-8")).get("name", p.stem)})
            except (OSError, ValueError):
                pass
        return web.json_response(items)

    async def h_wf_add(self, req):
        body = await req.json()
        graph = body.get("graph")
        try:
            customwf.validate_api(graph)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
        wid = uuid.uuid4().hex[:10]
        name = (body.get("name") or "Мой workflow").strip()[:80]
        self.wf_path(wid).write_text(json.dumps({"name": name, "graph": graph}, ensure_ascii=False), encoding="utf-8")
        return web.json_response({"id": wid, "name": name})

    async def h_wf_delete(self, req):
        p = self.wf_path(req.match_info["id"])
        p.unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def wf_infos(self, graph):
        """/object_info of every node class the workflow uses (a class ComfyUI does not know is left out)."""
        async def one(cls):
            try:
                status, data = await self.comfy_json("GET", f"/object_info/{cls}", timeout=aiohttp.ClientTimeout(total=15))
            except Exception:
                raise web.HTTPServiceUnavailable(text="ComfyUI не отвечает")
            return cls, (data[cls] if status == 200 and isinstance(data, dict) and cls in data else None)
        pairs = await asyncio.gather(*(one(c) for c in {n["class_type"] for n in graph.values()}))
        return {c: i for c, i in pairs if i is not None}

    async def h_wf_get(self, req):
        wf_file = self.wf_load(req.match_info["id"])
        report = customwf.analyze(wf_file["graph"], await self.wf_infos(wf_file["graph"]))
        return web.json_response({"id": req.match_info["id"], "name": wf_file["name"], **report})

    async def h_generate_custom(self, req):
        form, uploads = {}, {}
        async for field in await req.multipart():
            if field.name.startswith("f:") and field.filename:
                uploads[field.name[2:]] = await self.upload_input(field)
            else:
                form[field.name] = await field.text()
        wf_file = self.wf_load(form.get("wf", ""))
        graph = wf_file["graph"]
        report = customwf.analyze(graph, await self.wf_infos(graph))
        if report["missing_nodes"]:
            raise web.HTTPBadRequest(text="В вашем ComfyUI нет нод: " + ", ".join(sorted({m["cls"] for m in report["missing_nodes"]})))
        try:
            values = json.loads(form.get("values") or "{}")
        except ValueError:
            values = {}
        graph = customwf.apply_values(graph, values, uploads)
        await self.stop_llm()
        prompt = next((str(values.get(w["key"], w["value"])) for w in report["widgets"] if w["role"] == "prompt"), "")
        seed = next((values.get(w["key"], w["value"]) for w in report["widgets"] if w["role"] == "seed"), None)
        return await self.queue(graph, customwf.stage_map(graph), {
            "mode": "custom", "sub": wf_file["name"], "wf": form.get("wf"), "cw_values": values,
            "prompt": prompt, "aspect": "", "q": "", "len": 0.0, "res": "",
            "seed": seed if isinstance(seed, (int, float)) else None}, last_output=True,
            titles={nid: (n.get("_meta") or {}).get("title", "") for nid, n in graph.items() if n["class_type"].startswith("Save")})

    async def read_image_form(self, req):
        """Multipart form of the picture sections: text fields plus one optional file `image`."""
        if self.rh():
            raise web.HTTPBadRequest(text="Изображения работают только на своём ComfyUI (не в облаке RunningHub).")
        form, image, extra, mask, crop = {}, None, [], None, None
        async for field in await req.multipart():
            if field.name == "image" and field.filename:
                image = await self.upload_input(field)
            elif field.name == "mask" and field.filename:
                mask = await self.upload_input(field)
            elif field.name == "crop" and field.filename:
                crop = await self.upload_input(field)
            elif field.name == "orig" and field.filename:
                form["_orig"] = await self.upload_input(field)
            elif field.name in ("image2", "image3", "image4", "image5") and field.filename:
                extra.append(await self.upload_input(field))
            else:
                form[field.name] = await field.text()
        form["_extra"] = extra
        if mask:
            form["_mask"] = mask
        if crop:
            form["_crop"] = crop
        seed = int(form["seed"]) if form.get("seed", "").isdigit() else random.randrange(2 ** 50)
        return form, image, seed

    async def need_relay_node(self):
        """The fast upscale uses a node that ComfyUI has had only since 0.45.0: say so in plain words instead of a node error."""
        try:
            _, oi = await self.comfy_json("GET", "/object_info/RelayVSRConditioning", timeout=aiohttp.ClientTimeout(total=15))
        except Exception:
            return   # ComfyUI does not answer: the usual error will tell
        if oi:
            return
        how = ("Закройте программу, откройте папку установки, затем ComfyUI_windows_portable\\update и запустите "
               "update_comfyui.bat. После этого запустите программу снова." if self.config.get("engine") == "managed" else
               "Обновите ComfyUI (в его папке update запустите update_comfyui.bat) и перезапустите его.")
        raise web.HTTPBadRequest(text="Для быстрого увеличения нужен ComfyUI версии 0.45.0 или новее: в вашем нет нужной функции. " + how
                                      + " Обычное («Качественнее — SeedVR2») работает и на старой версии.")

    async def h_upscale(self, req):
        """Improve the quality of a finished video or picture from the library (SeedVR2), the sound stays."""
        peek = await req.json()
        soft = peek.get("method") == "soft2k"  # "up to 2K", soft (pictures only)
        relay = peek.get("engine") == "relay"   # the fast RelayVSR x2 (video only) instead of SeedVR2
        dlss = peek.get("engine") == "dlss"
        if dlss:
            await self.need_dlss_node((self.library.get(peek.get("id", "")) or {}).get("kind") == "image")
        else:
            await self.need_models("relayvsr" if relay else "upscale")
        if relay:
            await self.need_relay_node()
        if self.rh():
            raise web.HTTPBadRequest(text="Увеличение работает только на своём ComfyUI (не в облаке RunningHub).")
        body = peek
        item = self.library.get(body.get("id", ""))
        if not item:
            raise web.HTTPBadRequest(text="Файл не найден")
        if soft and item.get("kind") != "image":
            raise web.HTTPBadRequest(text="«До 2К» работает для изображений. Для видео выберите «Увеличить в 1,5 или 2 раза».")
        path = self.file_path(item)
        content = None
        if path and path.exists():
            content = path.read_bytes()
            suffix = path.suffix
        else:
            # output folder unknown (own ComfyUI without --output-directory): take the file from ComfyUI itself
            f = item["file"]
            params = {"filename": f.get("filename", ""), "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
            try:
                async with self.session.get(self.comfy + "/view", params=params) as r:
                    if r.status == 200:
                        content = await r.read()
            except aiohttp.ClientError:
                content = None
            suffix = Path(f.get("filename", "")).suffix
        if not content:
            raise web.HTTPBadRequest(text="Файл не найден")
        return await self._upscale_core(body, item, content, suffix, relay, soft)

    async def h_dlss_check(self, req):
        """Does the user's ComfyUI have the DLSS 5 node (video / picture)? The window shows instructions instead of the settings if not."""
        out = {}
        for key, pic in (("video", False), ("image", True)):
            try:
                out[key] = bool(await self.need_dlss_node(pic))
            except web.HTTPBadRequest:
                out[key] = False
        out["managed"] = self.config.get("engine") == "managed"
        nd = self.dlss_nodes_dir()
        out["folder"] = bool(nd)
        pack = nd / "ComfyUI-DLSS5-orex" if nd else None
        out["pack_dir"] = bool(pack and pack.is_dir())
        out["runtime_file"] = bool(pack and (pack / "runtime" / "nvngx_dlssnr.dll").is_file())
        return web.json_response(out)

    def dlss_nodes_dir(self):
        """custom_nodes of the ComfyUI in use (the app's own, or the folder the user gave once), if known."""
        if self.config.get("engine") == "managed":
            root = self.comfy_root()
            d = root / "ComfyUI" / "custom_nodes" if root else None
        else:
            raw = str(self.config.get("own_comfy_dir") or "").strip().strip('"')
            base = Path(raw) if raw else None
            d = next((c for c in (base / "custom_nodes", base / "ComfyUI" / "custom_nodes") if c.is_dir()), None) if base else None
        return d if d and d.is_dir() else None

    async def h_dlss_open(self, req):
        """Opens the node's GitHub page or the custom_nodes folder: the user installs the DLSS 5 node and the NVIDIA file by hand."""
        body = await req.json()
        if body.get("what") == "github":
            webbrowser.open("https://github.com/orex2121/ComfyUI-DLSS5-orex/releases")
            return web.json_response({"ok": True})
        if body.get("what") == "dll":   # the link the node's author gives in his runtime\README.txt; the user downloads it himself
            webbrowser.open("https://github.com/orex2121/ComfyUI-DLSS5-orex/releases/download/nvngx-v1/nvngx_dlssnr.dll")
            return web.json_response({"ok": True})
        nd = self.dlss_nodes_dir()
        if not nd:
            raise web.HTTPBadRequest(text="Папка custom_nodes вашего ComfyUI не известна программе: откройте её сами (рядом с main.py ComfyUI).")
        pack = nd / "ComfyUI-DLSS5-orex" / "runtime"
        os.startfile(str(pack if body.get("what") == "runtime" and pack.is_dir() else nd))
        return web.json_response({"ok": True})

    async def need_dlss_node(self, pic=False):
        """Checks the unofficial DLSS 5 node; for a picture returns the class of the node named «DLSS 5 (OreX)» (its class name is not fixed here)."""
        try:
            if not pic:
                _, oi = await self.comfy_json("GET", "/object_info/DLSS5NeuralRenderingVideo", timeout=aiohttp.ClientTimeout(total=15))
                found = "DLSS5NeuralRenderingVideo" if oi else ""
            else:
                _, oi = await self.comfy_json("GET", "/object_info/DLSS5NeuralRendering", timeout=aiohttp.ClientTimeout(total=15))
                if oi:
                    return "DLSS5NeuralRendering"
                _, oi = await self.comfy_json("GET", "/object_info", timeout=aiohttp.ClientTimeout(total=60))
                found = next((k for k, v in (oi or {}).items() if "DLSS" in k.upper() and "VIDEO" not in k.upper()
                              and "image" in (v.get("input", {}).get("required", {}) or {})), "")
        except Exception:
            return "" if pic else None   # ComfyUI does not answer: the usual error will tell
        if not found:
            raise web.HTTPBadRequest(text="В вашем ComfyUI нет узла " + ("«DLSS 5 (OreX)» для картинок" if pic else "«DLSS 5 Video (OreX)»")
                                          + ". Это сторонний экспериментальный узел, программа его сама не ставит. Поставьте его в ComfyUI и перезапустите ComfyUI.")
        return found

    async def h_upscale_file(self, req):
        """The same enlarge for ANY file of the user (made elsewhere): the file comes with the request, not from the library."""
        form, content, name = {}, None, ""
        async for part in await req.multipart():
            if part.name == "file":
                name = part.filename or ""
                content = await part.read()
            else:
                form[part.name] = (await part.text()).strip()
        if not content:
            raise web.HTTPBadRequest(text="Файл не получен")
        suffix = Path(name).suffix.lower()
        pic = suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        relay = form.get("engine") == "relay"
        soft = form.get("method") == "soft2k"
        if form.get("engine") == "dlss":
            await self.need_dlss_node(pic)
        else:
            await self.need_models("relayvsr" if relay else "upscale")
        if relay:
            await self.need_relay_node()
        if self.rh():
            raise web.HTTPBadRequest(text="Увеличение работает только на своём ComfyUI (не в облаке RunningHub).")
        if soft and not pic:
            raise web.HTTPBadRequest(text="«До 2К» работает для изображений. Для видео выберите «Увеличить в 1,5 или 2 раза».")
        item = {"id": "", "kind": "image" if pic else "video", "res": form.get("res", ""), "len": float(form.get("len") or 0),
                "prompt": "Увеличение: " + name[:60], "aspect": "", "q": ""}
        return await self._upscale_core(form, item, content, suffix, relay, soft)

    async def _upscale_core(self, body, item, content, suffix, relay, soft):
        pic = item.get("kind") == "image"
        dlss = body.get("engine") == "dlss"
        if relay:
            if pic:
                raise web.HTTPBadRequest(text="Быстрое увеличение работает только для видео. Для изображения выберите SeedVR2.")
            try:
                w0, h0 = (int(x) for x in str(item.get("res", "")).split("×"))
                secs = float(item.get("len") or 0)
            except ValueError:
                w0 = h0 = 0; secs = 0.0
            if w0 and secs and w0 * h0 * secs > wf.RELAY_PIXEL_SECONDS:   # decoded in memory as a whole: longer clips do not fit
                ok_secs = wf.RELAY_PIXEL_SECONDS / (w0 * h0)
                raise web.HTTPBadRequest(text=f"Видео слишком большое для быстрого увеличения: оно целиком загружается в память. "
                                              f"При размере {w0}×{h0} можно не больше {ok_secs:.0f} с (у вас {secs:.0f} с). "
                                              f"Выберите SeedVR2 или сделайте ролик короче.")
        scale = 2.0 if relay else wf.UPSCALE_SCALES.get(str(body.get("scale")), 1.5)
        data = aiohttp.FormData()
        name = f"h3s_{uuid.uuid4().hex[:8]}{suffix or ('.png' if pic else '.mp4')}"
        data.add_field("image", content, filename=name, content_type="image/png" if pic else "video/mp4")
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать файл в ComfyUI")
        video = f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]
        src_file = {"filename": res["name"], "subfolder": res.get("subfolder", ""), "type": "input"}  # the "before" of the before/after slider
        seed = random.randrange(2 ** 40)
        try:
            w, h = (int(x) for x in str(item.get("res", "")).split("×"))
            res = f"{round(w * scale / 2) * 2}×{round(h * scale / 2) * 2}"
        except ValueError:
            res = ""
        if soft:  # long side 2048, SeedVR2 mixed half with the plain enlarge
            try:
                w0, h0 = (int(x) for x in str(item.get("res", "")).split("×"))
            except ValueError:
                raise web.HTTPBadRequest(text="Не знаю размер изображения, увеличить до 2К не получится.")
            scale = max(1.0, 2048 / max(w0, h0))
            graph, stages = wf.upscale_image_graph(self.model_names, video, scale, seed, soften=0.5)
            return await self.queue(graph, stages, {
                "mode": "upscale", "sub": "До 2К, мягко", "prompt": item.get("prompt", "") or "Увеличение до 2К",
                "aspect": item.get("aspect", ""), "q": "2K", "len": 0.0, "res": f"{round(w0 * scale / 2) * 2}×{round(h0 * scale / 2) * 2}", "seed": seed, "src": item.get("id", ""), "src_file": src_file})
        def num(key, default):   # a slider value 0..1 from the window (a string when it comes with an uploaded file)
            try:
                return min(1.0, max(0.0, float(body.get(key))))
            except (TypeError, ValueError):
                return default
        if dlss:
            dp = body.get("dlss", {})
            if isinstance(dp, str):
                try:
                    dp = json.loads(dp)
                except ValueError:
                    dp = {}
            dp = dp if isinstance(dp, dict) else {}
            if pic:
                graph, stages = wf.dlss_image_graph(self.model_names, video, dp, await self.need_dlss_node(True))
            else:
                graph, stages = wf.dlss_graph(self.model_names, video, dp)
            return await self.queue(graph, stages, {
                "mode": "upscale", "sub": "DLSS 5 (" + str(dp.get("upscaling_mode", "")) + ")", "prompt": item.get("prompt", "") or "DLSS",
                "aspect": item.get("aspect", ""), "q": item.get("q", ""), "len": item.get("len", 0.0), "res": "", "seed": seed,
                "src": item.get("id", ""), "src_file": src_file})
        if relay:
            graph, stages = wf.relay_graph(self.model_names, video, seed, detail=num("detail", 0.7))
        else:
            soften = round(1.0 - num("sharp", 1.0), 2)   # 0 = full sharpness of SeedVR2, more = mixed with the plain enlarge
            graph, stages = (wf.upscale_image_graph if pic else wf.upscale_graph)(self.model_names, video, scale, seed, soften=soften)
        return await self.queue(graph, stages, {
            "mode": "upscale", "sub": "Быстрое увеличение ×2 (RelayVSR)" if relay else f"Увеличение ×{body.get('scale', '1.5')}", "prompt": item.get("prompt", "") or "Увеличение",
            "aspect": item.get("aspect", ""), "q": item.get("q", ""), "len": item.get("len", 0.0), "res": res, "seed": seed,
            "src": item.get("id", ""), "src_file": src_file})

    async def h_generate_krea(self, req):
        await self.need_models("krea")
        form, _, seed = await self.read_image_form(req)
        prompt = form.get("prompt", "").strip()
        if not prompt:
            raise web.HTTPBadRequest(text="Напишите промпт")
        lab = {"label": form["batch_label"][:20]} if form.get("batch_label") else {}  # "3 из 8" for a series from a prompt list
        aspect = form.get("aspect", "16:9")
        if aspect not in wf.KREA_SIZES:
            raise web.HTTPBadRequest(text="Неизвестный формат")
        width, height = wf.KREA_SIZES[aspect]
        custom = round_size(form)
        if custom:
            width, height = custom
        engine = form.get("engine", "krea")
        if engine == "flux":
            await self.need_models("flux")
            graph, stages = wf.flux_t2i_graph(self.model_names, prompt, width, height, seed)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Изображение (FLUX.2)", "prompt": prompt, "aspect": aspect, "q": "1K",
                "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})
        if engine == "h3img":   # MiniMax H3 Image: the size comes from the format and the area (megapixels), not from the 1 MP table
            await self.need_models("h3img")
            mp = wf.H3IMG_MP.get(form.get("mp", "4"), 4.0)
            width, height = round_size(form, high=8192, max_px=H3IMG_MAX_PX) or wf.h3img_size(aspect, mp)   # the own size is not cut to 2 MP here
            graph, stages = wf.h3img_t2i_graph(self.model_names, prompt, width, height, seed)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Изображение (MiniMax H3 Image)", "prompt": prompt, "aspect": aspect, "q": f"{mp:g} МП",
                "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})
        if engine == "qwen":
            await self.need_models("qwen")
            graph, stages = wf.qwen_t2i_graph(self.model_names, prompt, form.get("negative", "").strip()[:600], width, height, seed)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Изображение (Qwen 2.1)", "prompt": prompt, "aspect": aspect, "q": "1K",
                "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})
        if form.get("style") == "real":
            await self.need_models("kreareal")
            scale = wf.KREA_REAL_SCALES.get(form.get("scale", "2k"), 1.5)
            graph, stages = wf.krea_real_graph(self.model_names, prompt, width, height, seed, scale)
            return await self.queue(graph, stages, {
                "mode": "krea", "sub": "Изображение: реализм", "prompt": prompt, "aspect": aspect, "q": "4K" if scale > 2 else "2K",
                "len": 0.0, "res": f"{int(width * scale)}×{int(height * scale)}", "seed": seed, **lab})
        graph, stages = wf.krea_graph(self.model_names, prompt, width, height, seed)
        return await self.queue(graph, stages, {
            "mode": "krea", "sub": "Изображение по тексту", "prompt": prompt, "aspect": aspect, "q": "1K",
            "len": 0.0, "res": f"{width}×{height}", "seed": seed, **lab})

    async def h_generate_qwen(self, req):
        form, image, seed = await self.read_image_form(req)
        flux = form.get("kind") != "sheet" and form.get("engine") == "flux"
        h3 = form.get("engine") == "h3img" and form.get("kind", "edit") in ("edit", "story", "sheet")   # MiniMax H3 Image: edit / storyboard / character sheet, up to 5 reference pictures
        await self.need_models("h3img" if h3 else "flux" if flux else "qwen")
        if form.get("kind") == "bg":   # one click: the background becomes plain white, the subject stays untouched
            if not image:
                raise web.HTTPBadRequest(text="Загрузите изображение")
            mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
            graph, stages = wf.qwen_edit_graph(self.model_names, wf.BG_REMOVE_PROMPT, image, mp, seed)
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Убрать фон", "prompt": "", "aspect": "",
                "q": "", "len": 0.0, "res": "", "seed": seed})
        if form.get("kind") == "outpaint":   # extend the picture: client builds the larger canvas and the offset
            if not image or not form.get("_orig"):
                raise web.HTTPBadRequest(text="Загрузите изображение")
            try:
                w, h, x, y = (max(0, int(form.get(k, "0"))) for k in ("width", "height", "x", "y"))
            except ValueError:
                raise web.HTTPBadRequest(text="Неверный размер холста")
            if not (320 <= w <= 4096 and 320 <= h <= 4096):
                raise web.HTTPBadRequest(text="Слишком большой или маленький холст")
            user_text = form.get("prompt", "").strip()
            prompt = wf.OUTPAINT_PROMPT + (f" Scene: {user_text}" if user_text else "")
            lora = None
            try:
                _, loras = await self.comfy_json("GET", "/models/loras", timeout=aiohttp.ClientTimeout(total=15))
                lora = next((f for f in (loras or []) if f.lower().endswith("qwen-image-2.1-outpaint-v2.safetensors")), None)
            except Exception:
                pass
            try:
                ow, oh = int(form.get("ow", "0")), int(form.get("oh", "0"))
            except ValueError:
                ow = oh = 0
            graph, stages = wf.qwen_outpaint_graph(self.model_names, prompt, image, form["_orig"], x, y, (w, h), seed, lora,
                                                   (ow, oh) if ow > 0 and oh > 0 else None)
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Расширение изображения", "prompt": prompt, "aspect": "",
                "q": "", "len": 0.0, "res": f"{w}×{h}", "seed": seed})
        if form.get("kind") == "story" and form.get("frames") == "1":
            # storyboard frame by frame (FLUX.2 klein): every frame is its own picture, the sheet is glued from them
            use_qwen = form.get("engine") == "qwen"
            await self.need_models("h3img" if h3 else "qwen" if use_qwen else "flux")
            frames = split_frames(form.get("prompt", ""))
            if len(frames) not in wf.FRAME_COLS:
                raise web.HTTPBadRequest(text=f"Нужно 2, 3, 4, 6, 8 или 9 кадров, а в описании их {len(frames)}. Напишите каждый кадр с новой строки: «1: ...», «2: ...».")
            refs = [x for x in ([image] if image else []) + list(form.get("_extra") or [])]
            build = wf.h3img_frames_graph if h3 else wf.qwen_frames_graph if use_qwen else wf.flux_frames_graph
            graph, stages, titles = build(self.model_names, refs, frames, seed, size=(round_size(form, high=8192, max_px=H3IMG_MAX_PX) or wf.h3img_size("16:9", min(wf.H3IMG_MP.get(form.get("size", "8"), 8.0), 8.0))) if h3 else (round_size(form) or wf.FRAME_SIZE),   # a frame: up to 8 MP (9 frames of 16 MP would take an hour)
                                          line=form.get("line", "white"))
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Раскадровка по кадрам", "prompt": form.get("prompt", "").strip(), "aspect": "",
                "q": "", "len": 0.0, "res": "", "seed": seed}, titles=titles)
        if not image:
            raise web.HTTPBadRequest(text="Загрузите изображение")
        if form.get("kind") == "story":
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Опишите историю по кадрам")
            refs = [image] + list(form.get("_extra") or [])
            sz = round_size(form) or wf.STORY_SIZE
            if h3:
                graph, stages = wf.h3img_story_graph(self.model_names, refs, prompt, seed, round_size(form, high=8192, max_px=H3IMG_MAX_PX) or wf.h3img_size("16:9", wf.H3IMG_MP.get(form.get("size", "16"), 16.0)))
            elif flux:
                graph, stages = wf.flux_story_graph(self.model_names, refs, prompt, seed, sz)
            else:
                graph, stages = wf.qwen_story_graph(self.model_names, refs, prompt, seed, sz)
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Раскадровка по изображениям", "prompt": prompt, "aspect": "",
                "q": "", "len": 0.0, "res": "%d×%d" % wf.STORY_SIZE, "seed": seed})
        if h3 and form.get("kind", "edit") == "edit":
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Напишите, что сделать с изображениями")
            mp = wf.H3IMG_MP.get(form.get("size", "4"), 4.0)
            size = round_size(form, high=8192, max_px=H3IMG_MAX_PX) or (wf.h3img_size(form["aspect"], mp) if form.get("aspect") else None)
            refs = [image] + list(form.get("_extra") or [])
            graph, stages = wf.h3img_edit_graph(self.model_names, prompt, refs, mp, seed, size, min(mp, 4.0))
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": f"Правка (MiniMax H3 Image, картинок: {len(refs)})", "prompt": prompt, "aspect": form.get("aspect", ""),
                "q": f"{mp:g} МП", "len": 0.0, "res": "%d×%d" % size if size else "", "seed": seed})
        if flux:
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Напишите, что изменить на изображении")
            mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
            graph, stages = wf.flux_edit_graph(self.model_names, prompt, image, mp, seed, form.get("_extra") or (), round_size(form))
            return await self.queue(graph, stages, {
                "mode": "qwen", "sub": "Правка изображения (FLUX.2)", "prompt": prompt, "aspect": "",
                "q": "", "len": 0.0, "res": "", "seed": seed})
        if form.get("kind") == "sheet":
            panels = 6 if form.get("panels") == "6" else 4
            sheet_mp = wf.H3IMG_MP.get(form.get("size", "16"), 16.0)
            kw = {"mp": sheet_mp} if h3 else {}
            graph, stages, prompt = (wf.h3img_sheet_graph if h3 else wf.qwen_sheet_graph)(
                self.model_names, image, seed, headless=form.get("headless", "1") != "0",
                background=form.get("background", "grey"), marks=form.get("marks", "")[:300], panels=panels, **kw)
            sub, res = "Лист персонажа", "%d×%d" % (wf.h3img_size("11:5" if panels == 6 else "16:9", sheet_mp) if h3 else (wf.SHEET6_SIZE if panels == 6 else wf.SHEET_SIZE))
            prompt = "Лист персонажа" + (" (6 панелей)" if panels == 6 else "") + (": " + form["marks"].strip()[:80] if form.get("marks", "").strip() else "")
        else:
            prompt = form.get("prompt", "").strip()
            if not prompt:
                raise web.HTTPBadRequest(text="Напишите, что изменить на изображении")
            mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
            size = round_size(form) or wf.qwen_edit_size(form.get("aspect", ""), mp)
            graph, stages = wf.qwen_edit_graph(self.model_names, prompt, image, mp, seed, size, form.get("_extra") or ())
            sub, res = "Правка изображения", ("%d×%d" % size if size else "")
        return await self.queue(graph, stages, {
            "mode": "qwen", "sub": sub, "prompt": prompt, "aspect": form.get("aspect", "") if form.get("kind") != "sheet" else "",
            "q": "", "len": 0.0, "res": res, "seed": seed})

    async def h_generate_inpaint(self, req):
        form, image, seed = await self.read_image_form(req)
        await self.need_models("qwen")
        if not image:
            raise web.HTTPBadRequest(text="Загрузите изображение")
        if not form.get("_mask") or not form.get("_crop"):
            raise web.HTTPBadRequest(text="Закрасьте кисточкой место, которое нужно переделать")
        prompt = form.get("prompt", "").strip()
        if not prompt:
            raise web.HTTPBadRequest(text="Напишите, что должно быть в закрашенном месте")
        try:
            x, y = int(form.get("x", "0")), int(form.get("y", "0"))
        except ValueError:
            x, y = 0, 0
        mp = wf.QWEN_MP.get(form.get("size", "1"), 1.0)
        try:
            cw, ch = int(form.get("cw", "0")), int(form.get("ch", "0"))
        except ValueError:
            cw = ch = 0
        graph, stages = wf.qwen_inpaint_graph(self.model_names, prompt, image, form["_crop"], form["_mask"], x, y, mp, seed,
                                              (cw, ch) if cw > 0 and ch > 0 else None)
        return await self.queue(graph, stages, {
            "mode": "qwen", "sub": "Инпейнт", "prompt": prompt, "aspect": "",
            "q": "", "len": 0.0, "res": "", "seed": seed})

    async def h_generate_video(self, req):
        await self.need_models("video")
        form, images, prompt, aspect, quality, seed = await self.read_form(req, ("first", "last"))
        seconds = min(MAX_SECONDS, max(3.0, float(form.get("duration", 5))))
        width, height = wf.resolution(aspect, quality, form)
        first, last = images.get("first"), images.get("last") if images.get("first") else None
        loop = bool(first) and form.get("loop") == "1"
        if loop:  # seamless loop (live wallpaper): the video ends on the same picture it starts with
            last = first

        fast = None
        if form.get("accel") == "sla":
            miss = {mm["role"] for mm in self.missing}
            if "sla_lora" in miss or "sla_lora" not in self.model_names:
                raise web.HTTPBadRequest(text="Для быстрого режима нужен файл Turbo-SLA (около 2 ГБ). Нажмите «Скачать» под переключателем «Ускорение» "
                                              "или выберите «Обычное».")
            fast = self.model_names["sla_lora"]
        graph, stages = wf.video_graph(self.model_names, prompt, width, height, seconds, seed, first, last, self.custom_loras("video"), fast)
        sub = "Зацикленное видео" if loop else "Кадр → кадр" if last else "Изображение → видео" if first else "Текст → видео"
        if fast:
            sub += " · быстрый черновик"
        return await self.queue(graph, stages, {
            "mode": "video", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality,
            "len": seconds, "res": f"{width}×{height}", "seed": seed})

    async def h_generate_lipsync(self, req):
        await self.need_models("lipsync")
        ref_fields = tuple(f"ref{i}" for i in range(9))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, ref_fields + ("audio", "rvid0", "rvid1", "rvid2"))
        rvids = [files[f"rvid{i}"] for i in range(3) if f"rvid{i}" in files]   # RefMod videos of the cards
        if "audio" not in files:
            raise web.HTTPBadRequest(text="Загрузите аудио")
        refs = [files[f] for f in ref_fields if f in files]
        if not refs and not rvids:
            raise web.HTTPBadRequest(text="Добавьте хотя бы один референс — лицо персонажа")
        seconds = float(form.get("audio_len", 0))
        if not 1 <= seconds <= 15:
            raise web.HTTPBadRequest(text="Фрагмент аудио должен быть от 1 до 15 секунд")

        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        scale = wf.lipsync_scale(quality)
        out_w, out_h = wf.resolution(aspect, quality, form)
        width, height = wf.r32(out_w / scale), wf.r32(out_h / scale)
        extra = self.optional_loras(form, (("realism", "realism_lora"),)) + self.custom_loras("lipsync")
        graph, stages = wf.lipsync_graph(self.model_names, wf.lipsync_prompt(prompt, len(refs)), width, height,
                                         seconds, seed, files["audio"], refs, scale, ref_size, extra, rvids)
        speed = "максимум" if ref_size == "max" else "быстро"
        return await self.queue(graph, stages, {
            "mode": "lipsync", "sub": f"{len(refs)} реф. ({speed}) · аудио {seconds:.1f} с", "prompt": prompt,
            "aspect": aspect, "q": quality, "len": seconds, "res": f"{out_w}×{out_h}", "seed": seed,
            "ref_size": ref_size})

    async def read_ref_form(self, req, max_images, rh=None):
        fields = tuple(f"img{i}" for i in range(max_images)) + tuple(f"vid{i}" for i in range(3)) + \
            tuple(f"rvid{i}" for i in range(3)) + tuple(f"aud{i}" for i in range(3))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, fields, rh=rh)
        pick = lambda p, n: [files[f"{p}{i}"] for i in range(n) if f"{p}{i}" in files]
        images, videos, audios = pick("img", max_images), pick("vid", 3), pick("aud", 3)
        form["_nown"] = str(len(videos))   # the user's own reference videos come first: only they are described as "motion, camera and setting"
        videos = (videos + pick("rvid", 3))[:3]   # then the RefMod videos of the Reference Mode cards (the heroes)
        if not images and not videos:
            raise web.HTTPBadRequest(text="Добавьте хотя бы одно референс-изображение")
        if len(images) + len(videos) + len(audios) > 12:
            raise web.HTTPBadRequest(text="Всего не больше 12 референсов (изображения + видео + аудио)")
        seconds = min(MAX_SECONDS, max(3.0, float(form.get("duration", 6))))
        return form, prompt, aspect, quality, seed, images, videos, audios, seconds

    def custom_loras(self, mode=None):
        """(file, strength) for the LoRAs the user added by hand on the Models screen (not used with the cloud).
        A LoRA can be switched off in single modes (right in the mode, in the «Модель» row): its `skip` lists those modes."""
        if self.rh() or self.locked():   # «Я новичок»: only the LoRAs of the ready workflows
            return []
        return [(x["file"], float(x["strength"])) for x in self.unique_loras() if x.get("on", True) and mode not in (x.get("skip") or [])]

    def unique_loras(self):
        """The user's own LoRAs, one entry per file (older versions let one file be added several times: it was mixed in again and again)."""
        seen, out = set(), []
        for x in self.config.get("custom_loras") or []:
            key = str(x.get("file") or "").replace("/", "\\").lower()
            if key and key not in seen:
                seen.add(key)
                out.append(x)
        return out

    async def h_loras_custom(self, _req):
        try:
            _, files = await self.comfy_json("GET", "/models/loras")
        except Exception:
            raise web.HTTPServiceUnavailable(text="ComfyUI не отвечает")
        items = self.unique_loras()
        if len(items) != len(self.config.get("custom_loras") or []):
            self.config.update(custom_loras=items)   # the doubles are removed for good
        return web.json_response({"files": files or [], "items": items})

    async def h_loras_custom_set(self, req):
        if self.locked():
            raise web.HTTPForbidden(text="В режиме «Я новичок» свои LoRA не подключаются: всё уже настроено.")
        items = []
        for x in (await req.json()).get("items", [])[:10]:
            try:
                strength = max(-4.0, min(4.0, float(x.get("strength", 0.8))))
            except (TypeError, ValueError):
                strength = 0.8
            if isinstance(x.get("file"), str) and x["file"]:
                skip = [m for m in (x.get("skip") or []) if m in catalog.MODE_ROLES][:20]
                items.append({"file": x["file"], "strength": strength, "on": bool(x.get("on", True)), "skip": skip})
        self.config.update(custom_loras=items)
        return web.json_response({"ok": True})

    async def h_loras_custom_mode(self, req):
        """Switch an own LoRA on / off for ONE mode (from the «Модель» row of that mode)."""
        body = await req.json()
        file, mode = body.get("file"), body.get("mode")
        if mode not in catalog.MODE_ROLES:
            raise web.HTTPBadRequest(text="Неизвестный режим")
        items = self.unique_loras()
        hit = next((x for x in items if x.get("file") == file), None)
        if not hit:
            raise web.HTTPBadRequest(text="Такой LoRA нет в списке")
        skip = [m for m in (hit.get("skip") or []) if m != mode]
        if not body.get("on"):
            skip.append(mode)
        hit["skip"] = skip
        self.config.update(custom_loras=items)
        return web.json_response({"ok": True})

    def optional_loras(self, form, pairs):
        """(file name, strength) for the optional LoRAs the user switched on and this ComfyUI has."""
        missing = {m["role"] for m in self.missing}
        out = []
        for key, role in pairs:
            v = lora_strength(form, key, None)
            if v is not None and role not in missing and role in self.model_names:
                out.append((self.model_names[role], v))
        return out

    async def h_generate_refs(self, req):
        await self.need_models("refs")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 9)
        steps = int(form.get("steps", 6)) if form.get("steps", "6").isdigit() else 6
        steps = min(8, max(4, steps))
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        extra = self.optional_loras(form, (("realism", "realism_lora"), ("motion", "motion_repair_lora"))) + self.custom_loras("refs")
        width, height = wf.resolution(aspect, quality, form)
        pika = None
        if form.get("accel") == "pika":
            if "pika_lora" in {m["role"] for m in self.missing}:
                raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден в ComfyUI. "
                                              "Скачайте его в разделе «Настройки» или выберите прежнее ускорение.")
            pika, steps = self.model_names["pika_lora"], max(6, steps)
        graph, stages = wf.ref_graph(self.model_names, wf.lipsync_prompt(prompt, len(images), int(form.get("_nown") or 0)), width, height, seconds, seed, images, videos, audios,
                                     steps, ref_size, extra, pika)
        sub = f"{len(images)} карт." + (f", {len(videos)} видео" if videos else "") + (f", {len(audios)} аудио" if audios else "") \
            + f" · {steps} шагов" + (" · Pika" if pika else "")
        return await self.queue(graph, stages, {
            "mode": "refs", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "steps": steps, "ref_size": ref_size})

    async def long_source(self, item_id, uploaded, first_sec):
        """A video the user already has (from the library, or a file from the computer) to continue from: put it into ComfyUI and
        find out how many frames it has. Returns (file name in ComfyUI input, frames)."""
        if uploaded:
            name = uploaded   # read_form already uploaded it to ComfyUI
        else:
            it = self.library.get(item_id)
            if not it or it.get("kind") == "image":
                raise web.HTTPBadRequest(text="Видео для продолжения не найдено в Библиотеке")
            data = await self.item_bytes(it)
            fd = aiohttp.FormData()
            nm = f"h3s_c_{uuid.uuid4().hex[:10]}.{it.get('ext') or 'mp4'}"
            fd.add_field("image", data, filename=nm, content_type="video/mp4")
            fd.add_field("type", "input")
            status, res = await self.comfy_json("POST", "/upload/image", data=fd)
            if status != 200:
                raise web.HTTPBadGateway(text="Не удалось передать видео в ComfyUI")
            name = res["name"]
        # how long is it: a tiny job that only opens the video and reports its number of frames
        probe = {"L": {"class_type": "LoadVideo", "inputs": {"file": name}},
                 "G": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}},
                 "Z": {"class_type": "GetImageSize", "inputs": {"image": ["G", 0]}},
                 "N": {"class_type": "PreviewAny", "inputs": {"source": ["Z", 2]}},
                 "F": {"class_type": "PreviewAny", "inputs": {"source": ["G", 2]}}}
        status, res = await self.comfy_json("POST", "/prompt", json={"prompt": probe, "client_id": self.client_id})
        if status != 200:
            raise web.HTTPBadRequest(text="ComfyUI не смог открыть это видео: " + comfy_error_text(res))
        pid = res["prompt_id"]
        loop = asyncio.get_running_loop()
        end = loop.time() + 180
        meta = None
        while loop.time() < end and not meta:
            await asyncio.sleep(0.7)
            _, hist = await self.comfy_json("GET", f"/history/{pid}")
            e = (hist or {}).get(pid)
            if e:
                if e.get("status", {}).get("status_str") == "error":
                    break
                try:
                    meta = {"frame_count": int(float(e["outputs"]["N"]["text"][0])), "fps": float(e["outputs"]["F"]["text"][0])}
                except (KeyError, IndexError, ValueError):
                    break
        if not meta:
            raise web.HTTPBadRequest(text="Не удалось прочитать это видео. Подойдёт обычный mp4.")
        frames = int(meta["frame_count"])
        need = wf.long_plan([first_sec], cont_first=True)[0]["ov"]
        if frames < need:
            raise web.HTTPBadRequest(text=f"Видео слишком короткое для продолжения: в нём {frames} кадров, нужно не меньше {need} (около 2,5 секунд).")
        return (name, frames)

    async def preview_opts(self, mode=None):
        """Settings for the animated live preview, or None when ComfyUI has no KJNodes' Model Preview Override.
        Only the action scenes use it (there the motion is what has to be judged, as in the original workflow); in the
        other modes it is off: the animated preview made generation noticeably slower, the usual single-frame preview is used."""
        if mode not in ("action", "long"):
            return None
        try:
            status, oi = await self.comfy_json("GET", "/object_info/ModelPreviewOverrideKJ")
        except Exception:
            return None
        node = (oi or {}).get("ModelPreviewOverrideKJ") if status == 200 else None
        if not node:
            return None
        tiny = "none"
        try:
            spec = node["input"].get("optional", {}).get("tiny_vae") or []
            options = (spec[1] or {}).get("options") if len(spec) > 1 and isinstance(spec[1], dict) else spec[0]
            tiny = next((o for o in options or [] if o.replace("\\", "/").split("/")[-1].startswith("taeh3")), "none")
        except Exception:
            pass
        return {"tiny_vae": tiny}

    def keep_long_refs(self, run, captured):
        """The reference pictures (and the sound) of a long video project, next to its saved scene states."""
        if not run:
            return
        d = LONG_DIR / run / "refs"
        d.mkdir(parents=True, exist_ok=True)
        for key, (ext, raw) in captured.items():
            if key.startswith(("img", "aud", "simg")) and len(raw) < 80_000_000:
                for old in d.glob(key + ".*"):
                    old.unlink(missing_ok=True)
                (d / (key + ext)).write_bytes(raw)

    async def h_long_project(self, req):
        """The project of a long video (scenes, settings) is kept on disk too, so it survives anything that happens to the browser storage."""
        if req.method == "POST":
            body = await req.json()
            run = re.sub(r"[^A-Za-z0-9]", "", str(body.get("run", "")))[:32]
            if not run or not isinstance(body.get("project"), dict):
                raise web.HTTPBadRequest(text="bad project")
            d = LONG_DIR / run
            d.mkdir(parents=True, exist_ok=True)
            (d / "project.json").write_text(json.dumps(body["project"], ensure_ascii=False), encoding="utf-8")
            return web.json_response({"ok": True})
        if req.query.get("list"):   # the saved projects on the disk, newest first (to open an earlier one)
            out = []
            files = sorted(LONG_DIR.glob("*/project.json"), key=lambda f: f.stat().st_mtime, reverse=True) if LONG_DIR.is_dir() else []
            for f in files[:60]:
                try:
                    pr = json.loads(f.read_text(encoding="utf-8"))
                    sc = pr.get("scenes") or []
                except (OSError, ValueError):
                    continue
                if len(sc) < 2:
                    continue
                out.append({"run": f.parent.name, "t": int(f.stat().st_mtime * 1000), "scenes": len(sc), "title": str(pr.get("title") or "")[:60],
                            "done": sum(1 for x in sc if x.get("done") or x.get("draftDone")),
                            "first": re.sub(r"<[^>]*>", "", str((sc[0] or {}).get("prompt", "")))[:80]})
            return web.json_response(out)
        run = re.sub(r"[^A-Za-z0-9]", "", req.query.get("run", ""))[:32]
        if not run and req.query.get("last"):   # the most recently saved project
            files = sorted(LONG_DIR.glob("*/project.json"), key=lambda f: f.stat().st_mtime, reverse=True) if LONG_DIR.is_dir() else []
            run = files[0].parent.name if files else ""
        f = LONG_DIR / run / "project.json" if run else None
        if not f or not f.exists():
            return web.json_response({})
        refs = sorted(p.name for p in (LONG_DIR / run / "refs").glob("*")) if (LONG_DIR / run / "refs").is_dir() else []
        return web.json_response({"run": run, "project": json.loads(f.read_text(encoding="utf-8")), "refs": refs})

    async def h_long_stage(self, req):
        """A saved stage of a long video: the scene continuation points and the reference pictures are copied aside, so redoing a
        scene later does not spoil the stage. action: save | restore | delete."""
        body = await req.json()
        run = re.sub(r"[^A-Za-z0-9]", "", str(body.get("run", "")))[:32]
        sid = re.sub(r"[^A-Za-z0-9]", "", str(body.get("id", "")))[:32]
        act = body.get("action")
        if not run or not sid or act not in ("save", "restore", "delete"):
            raise web.HTTPBadRequest(text="bad stage")
        base = LONG_DIR / run
        st = base / "stages" / sid
        names = ("state_*.latent", "state_*.json")

        def work():
            if act == "save":
                if st.exists():
                    shutil.rmtree(st)
                st.mkdir(parents=True, exist_ok=True)
                n = 0
                for pat in names:
                    for f in base.glob(pat):
                        shutil.copy2(f, st / f.name)
                        n += 1
                if (base / "refs").is_dir():
                    shutil.copytree(base / "refs", st / "refs")
                return {"ok": True, "states": n, "refs": (st / "refs").is_dir()}
            if act == "delete":
                if st.exists() and st.parent.name == "stages":
                    shutil.rmtree(st)
                return {"ok": True}
            if not st.is_dir():   # a stage saved by an older version: only the texts and settings come back
                return {"ok": False}
            for pat in names:
                for f in base.glob(pat):
                    f.unlink(missing_ok=True)
            for f in st.glob("state_*"):
                shutil.copy2(f, base / f.name)
            if (st / "refs").is_dir():
                if (base / "refs").is_dir():
                    shutil.rmtree(base / "refs")
                shutil.copytree(st / "refs", base / "refs")
            return {"ok": True, "refs": (st / "refs").is_dir()}
        return web.json_response(await asyncio.to_thread(work))

    async def h_long_ref(self, req):
        run = re.sub(r"[^A-Za-z0-9]", "", req.match_info["run"])[:32]
        name = Path(req.match_info["name"]).name
        f = LONG_DIR / run / "refs" / name
        if not f.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(f)

    async def h_generate_long(self, req):
        if req.query.get("kind") != "action":   # the action scenes (page «Экшн» → «По сценам») need the action models instead
            await self.need_models("long")
        self._capture = {}
        try:
            return await self.generate_long(req)
        finally:
            captured, self._capture = self._capture or {}, None
            run = getattr(self, "_long_run", None)
            if run:
                self.keep_long_refs(run, captured)
                self._long_run = None

    async def generate_long(self, req):
        form, files, prompt, aspect, quality, seed = await self.read_form(req, ("img0", "img1", "img2", "img3", "img4", "aud0", "aud1", "aud2", "srcvid", "rvid0", "rvid1", "rvid2", "uvid0", "uvid1", "aref0", "aref1") + tuple(f"simg{i}_{j}" for i in range(80) for j in range(5)) + tuple(f"svoice{i}" for i in range(80)) + tuple(f"samp{i}_{j}" for i in range(80) for j in range(2)) + ("rmaud0", "rmaud1", "rmaud2"), need_prompt=False)   # the texts are in the scenes, there is no single prompt
        self._long_run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]   # the references are kept even if the request is refused later
        try:
            scenes = [(str(x.get("prompt", "")).strip(), float(x.get("sec", 10)), bool(x.get("fresh"))) for x in json.loads(form.get("scenes") or "[]")]
        except (ValueError, TypeError, AttributeError):
            scenes = []
        kept = [i for i, (p, _s, _f) in enumerate(scenes) if p][:80]
        samps = [[(files[f"samp{i}_{j}"], form.get(f"samplab{i}_{j}", "")) for j in range(2) if f"samp{i}_{j}" in files] for i in kept]   # voice samples of each scene (timbre; a dialogue has two)
        scenes = [(p, min(wf.LONG_MAX_SCENE, max(3.0, sec)), [files[f"simg{i}_{j}"] for j in range(5) if f"simg{i}_{j}" in files], fr, files.get(f"svoice{i}"))
                  for i, (p, sec, fr) in enumerate(scenes) if p][:80]   # every scene may bring its own references (instead of the common ones)
        fresh = [sc_[3] for sc_ in scenes]   # a scene marked as a NEW SHOT does not continue the previous one (no quality loss)
        scenes = [(p, sec, extra, vo) for p, sec, extra, _, vo in scenes]   # vo: the own voice recording of the scene
        test = form.get("test") == "1"   # a size test of one scene: stand-alone, does not touch the other scenes
        action = form.get("kind") == "action"   # action scenes: the action model, 8+4 steps, scene by scene (see wf.action_part_graph)
        sing = False
        if action:
            if not form.get("part", "").isdigit():
                raise web.HTTPBadRequest(text="Экшн-видеостудия считает по частям: выберите режим «По частям».")
            if not self.model_names:
                await self.resolve_models()
            # a file counts only when ComfyUI really sees it (the list of «missing» models knows only the installed sections)
            base_of = lambda n: str(n or "").replace("/", "\\").split("\\")[-1]
            seen_in = lambda role, cat: base_of(self.model_names.get(role)) in (getattr(self, "visible", None) or {}).get(cat, set())
            # the base model: Singularity of the Video studio (default) when its files are there, else the action hybrid
            sing = (form.get("amodel", "sing") != "action" and seen_in("lipsync_unet", "diffusion_models")
                    and seen_in("text_encoder", "text_encoders") and seen_in("video_vae", "vae") and seen_in("audio_vae", "vae"))
            if not sing:
                await self.need_models("action")
        draft = form.get("draft") == "1" and not test and not action   # quick draft: 4 steps, smaller frame; the finished row is made later with the same seed
        sfx = "_d" if draft else ""
        if form.get("run") and not form.get("seed"):   # one seed for the whole project: the draft and the finished version look alike
            seed = int(hashlib.md5(form["run"].encode()).hexdigest()[:12], 16)
        try:   # «redo this scene»: every redo is another variant, otherwise the same seed would give the same clip again
            seed += max(0, min(1000, int(form.get("variant") or 0))) * 1000003
        except ValueError:
            pass
        if not scenes:
            raise web.HTTPBadRequest(text="Напишите хотя бы одну сцену")
        images = [files[f"img{i}"] for i in range(5) if f"img{i}" in files]
        audio = None if action else files.get("aud0")   # the action studio takes up to 3 audio SAMPLES instead of one track
        auds = [files[f"aud{i}"] for i in range(3) if f"aud{i}" in files] if action else []
        rvids = [files[f"rvid{i}"] for i in range(3) if f"rvid{i}" in files]   # RefMod videos of the Reference Mode cards (heroes as <Video N>)
        n_rm = len(rvids)   # the user's own camera videos (<Video N> after the Reference Mode ones) and voice / sound references
        uvids = [files[f"uvid{i}"] for i in range(2) if f"uvid{i}" in files][:max(0, 3 - n_rm)]
        rvids = rvids + uvids
        cam = list(range(n_rm + 1, n_rm + 1 + len(uvids)))
        rm_a = [] if action else [files[f"rmaud{i}"] for i in range(3) if f"rmaud{i}" in files]   # voices of the Reference Mode heroes: <Audio 1..> (the text of the scene already names them)
        arefs = [] if action else [files[f"aref{i}"] for i in range(2) if f"aref{i}" in files]
        try:
            start = max(0.0, float(form.get("start") or 0))
        except ValueError:
            start = 0.0
        vocals = None
        if audio and form.get("vocals") == "1":
            if "vocal_model" in {m["role"] for m in self.missing} or "vocal_model" not in self.model_names:
                raise web.HTTPBadRequest(text="Для «Только вокал» нужна модель MelBand. Скачайте её в разделе «Настройки».")
            vocals = self.model_names["vocal_model"]
        no_pika = form.get("accel") == "old" and not self.locked()   #«Без Pika»: the quality route without a speed LoRA (12 steps), for those whom Pika gives trouble
        if not action and not no_pika and "pika_lora" in {m["role"] for m in self.missing}:
            raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден. Скачайте его в разделе «Настройки» или выберите «Без Pika».")
        fin = None if no_pika or action else self.model_names["pika_lora"]
        width, height = wf.resolution(aspect, quality, form)
        dmad = None
        if draft:
            pass   # a draft has the size the user chose (it may be enough for a clip that is only to be enlarged)
            _, loras = await self.comfy_json("GET", "/models/loras")
            dmad = next((n for n in loras or [] if re.search(r"dmad[_\-\s]?4[_\-\s]?step", n, re.I)), None)
            if not dmad:
                raise web.HTTPBadRequest(text="Для черновика нужен файл minimax_h3_DMAD_4step_full_lora в папке loras. Или сделайте сцену сразу в хорошем качестве.")
        extras = self.custom_loras("long")
        if not action:   # the clip page can add DMAD and Speed Slider for a fast scene (off unless the user switched them on)
            for key in ("dmad", "speed"):
                st = lora_strength(form, key, None)
                if not st:
                    continue
                if key == "dmad":
                    _, la = await self.comfy_json("GET", "/models/loras")
                    fn = next((n for n in la or [] if catalog.is_action_dmad(n)), None)
                else:
                    fn = self.model_names.get("speed_slider_lora")
                if not fn:
                    raise web.HTTPBadRequest(text=f"Нет файла LoRA {'DMAD' if key == 'dmad' else 'Speed Slider'}: выключите её или установите (раздел «Экшн-видео»).")
                extras = extras + [(fn, st)]
        sc_ar = [(rm_a + ([x[0] for x in sm] or arefs))[:3] for sm in samps] if not action else [[] for _ in scenes]   # a scene with its own samples uses them, the others the common ones
        sc_lb = [[x[1] for x in sm] if sm else [] for sm in samps]
        scenes = [(wf.lipsync_prompt(p, len(own or images), 0, cam, len(sc_ar[n]) - len(rm_a), sc_lb[n], len(rm_a)), sec, own, vo) for n, (p, sec, own, vo) in enumerate(scenes)]
        part = form.get("part", "")
        if part.isdigit():  # one scene at a time: scene k continues from the saved tail of scene k-1
            k = int(part)
            run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]
            self._long_run = run
            if not run or k >= len(scenes):
                raise web.HTTPBadRequest(text="Неверный номер сцены")
            state, cont = None, None
            if k == 0 and (form.get("src_item") or files.get("srcvid")):
                cont = await self.long_source(form.get("src_item"), files.get("srcvid"), scenes[0][1])
            if test:
                fresh = list(fresh); fresh[k] = True
            if k and not fresh[k]:
                src = LONG_DIR / run / f"state_{k - 1}{sfx}.latent"
                if not src.exists():
                    raise web.HTTPBadRequest(text=f"Сначала сделайте сцену {k}: следующая продолжает её конец.")
                try:  # the saved tail only fits a scene of the same frame size
                    info = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
                    was, was_kind = info.get("res"), info.get("kind", "")
                except (OSError, ValueError):
                    was, was_kind = None, ""
                if was and was != f"{width}×{height}":
                    raise web.HTTPBadRequest(text=f"Сцена {k} сделана в формате {was}, а сейчас выбран {width}×{height}. "
                                                  "Верните прежний формат и качество или переделайте сцену 1 в новом.")
                if was and was_kind != ("action" if action else ""):
                    raise web.HTTPBadRequest(text=f"Сцена {k} сделана {'обычным' if action else 'экшн'}-путём, а сейчас выбран другой тип сцен. "
                                                  "Верните прежний тип или переделайте сцену 1 в новом.")
                state = await self.upload_state(src, f"h3s_long_{run}_{k - 1}.latent")
            if action:
                _, loras_all = await self.comfy_json("GET", "/models/loras")
                dmad_a = next((n for n in loras_all or [] if catalog.is_action_dmad(n)), None)
                if not dmad_a:
                    raise web.HTTPBadRequest(text="Для экшн-видео нужен файл DMAD (minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors) в папке loras. Он описан в разделе «Экшн-видео».")
                a_loras = [(dmad_a, lora_strength(form, "dmad", 0.8))]
                if seen_in("speed_slider_lora", "loras"):
                    a_loras.append((self.model_names["speed_slider_lora"], lora_strength(form, "speed", 4.0)))
                a_loras = [x for x in a_loras if x[1] is not None] + self.custom_loras("action")
                a_steps = 12
                if sing and form.get("apika") == "1" and seen_in("pika_lora", "loras"):
                    a_loras.append((self.model_names["pika_lora"], 1.0))   # the same speed LoRA as in the Video studio
                    a_steps = 8                                             # and its 8 steps
                # like the Video studio: one pass at full size, the continuation is kept by a noise mask (SelfLift cannot take such a mask)
                graph, stages = wf.action_part_graph(self.model_names, scenes, k, width, height, seed, images, a_loras,
                                                     await self.preview_opts("action") if form.get("preview") != "0" else None, state, fresh,
                                                     False, None, True, sing, a_steps, None, start, None, auds, rvids)
                plan = wf.long_plan([sc[1] for sc in scenes], fresh=fresh)
                return await self.queue(graph, stages, {
                    "mode": "long", "sub": f"экшн · сцена {k + 1} из {len(scenes)}" + (f" · аудио-образцов: {len(auds)}" if auds else "") + (f" · проба {width}×{height}" if test else ""),
                    "prompt": scenes[k][0], "aspect": aspect, "q": quality,
                    "len": round((plan[k]["total"] - plan[k]["ov"]) / wf.FPS, 1), "res": f"{width}×{height}", "seed": seed,
                    "long": {"run": run, "part": k, "n": len(scenes), "draft": False, "test": test, "kind": "action"}})
            graph, stages = wf.long_part_graph(self.model_names, scenes, k, width, height, seed, images, audio, start,
                                               dmad if draft else fin, extras, "match", vocals, state,
                                               await self.preview_opts("long") if form.get("preview") != "0" else None, cont, form.get("src_project") == "1", fresh, draft, rvids,
                                               quiet=form.get("quiet") == "1", audios=sc_ar[k])
            plan = wf.long_plan([sc[1] for sc in scenes], cont_first=bool(cont) or form.get("src_project") == "1", fresh=fresh)
            sub = f"сцена {k + 1} из {len(scenes)}" + (" · музыка" if audio else " · звук от модели")
            return await self.queue(graph, stages, {
                "mode": "long", "sub": sub + (" · черновик" if draft else f" · проба {width}×{height}" if test else ""), "prompt": scenes[k][0], "aspect": aspect, "q": quality,
                "len": round((plan[k]["total"] - plan[k]["ov"]) / wf.FPS, 1), "res": f"{width}×{height}", "seed": seed,
                "long": {"run": run, "part": k, "n": len(scenes), "draft": draft, "test": test}})
        graph, stages = wf.long_graph(self.model_names, scenes, width, height, seed, images, audio, start,
                                      dmad if draft else fin, extras, "match", vocals, await self.preview_opts("long") if form.get("preview") != "0" else None, fresh, False, draft, rvids, (rm_a + arefs)[:3])
        run = re.sub(r"[^A-Za-z0-9]", "", form.get("run", ""))[:32]
        self._long_run = run
        total = wf.long_plan([sc[1] for sc in scenes], fresh=fresh)[-1]["acc"] / wf.FPS
        sub = f"{len(scenes)} сцен · {len(images)} карт." + (", музыка" if audio else ", звук от модели") + (" · черновик" if draft else "")
        meta = {"mode": "long", "sub": sub, "prompt": "\n\n".join(sc[0] for sc in scenes), "aspect": aspect, "q": quality,
                "len": round(total, 1), "res": f"{width}×{height}", "seed": seed}
        if run:
            meta["long"] = {"run": run, "part": len(scenes) - 1, "n": len(scenes), "all": True}
        return await self.queue(graph, stages, meta)

    async def h_long_state(self, req):
        """Which scenes of a long video project have a saved continuation point (so the project can go on after a restart)."""
        run = re.sub(r"[^A-Za-z0-9]", "", req.query.get("run", ""))[:32]
        d = LONG_DIR / run
        names = [f.name for f in d.glob("state_*.latent")] if run and d.is_dir() else []
        parts = sorted(int(m.group(1)) for n in names if (m := re.match(r"state_(\d+)\.latent$", n)))
        dparts = sorted(int(m.group(1)) for n in names if (m := re.match(r"state_(\d+)_d\.latent$", n)))
        return web.json_response({"parts": parts, "draft_parts": dparts})

    async def item_bytes(self, item):
        """The file of a library item: from disk, or from ComfyUI when its output folder is not known to the app."""
        p = self.file_path(item)
        if p and p.exists():
            return await asyncio.to_thread(p.read_bytes)
        f = item["file"]
        async with self.session.get(self.comfy + "/view", params={
                "filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}) as r:
            if r.status != 200:
                raise web.HTTPBadRequest(text="Не удалось найти один из клипов: он удалён с диска?")
            return await r.read()

    async def h_montage(self, req):
        """Join library clips in the given order with ComfyUI (no ffmpeg in the app)."""
        form, files, music_bytes, aud_files = {}, {}, None, {}
        async for field in await req.multipart():
            if field.name.startswith("file_"):   # a clip taken from the user's computer
                files[field.name[5:]] = (field.filename or "clip.mp4", bytes(await field.read()))
            elif field.name == "music":   # the song of a music clip: laid under the whole video in one piece
                music_bytes = bytes(await field.read())
            elif field.name.startswith("aud_"):   # an audio file the user put on an audio track
                aud_files[field.name[4:]] = (field.filename or "audio.wav", bytes(await field.read()))
            else:
                form[field.name] = await field.text()
        try:
            ids = [i for i in json.loads(form.get("ids") or "[]") if isinstance(i, str)][:30]
        except ValueError:
            ids = []
        try:
            durs = json.loads(form.get("durs") or "[]")
        except ValueError:
            durs = []
        items = []
        for i in ids:
            if i.startswith("up:"):
                k = i[3:]
                if k not in files:
                    raise web.HTTPBadRequest(text="Файл с компьютера не дошёл до программы, добавьте его ещё раз")
                try:
                    ln = float(form.get("len_" + k) or 0)
                except ValueError:
                    ln = 0.0
                items.append({"kind": "video", "_up": files[k], "res": form.get("res_" + k, ""), "len": ln,
                              "ext": (Path(files[k][0]).suffix.lstrip(".").lower() or "mp4")[:5]})
            else:
                items.append(self.library.get(i))
        if len(items) < 2 or any(not it or it.get("kind") == "image" for it in items):
            raise web.HTTPBadRequest(text="Добавьте хотя бы два видео")
        if self.rh():
            raise web.HTTPBadRequest(text="Монтаж работает только с локальным ComfyUI, не с облаком.")
        try:
            starts_raw = json.loads(form.get("starts") or "[]")
        except ValueError:
            starts_raw = []
        starts = []   # the second of the clip where its piece begins (0: from the start)
        for n_ in range(len(items)):
            s_ = starts_raw[n_] if n_ < len(starts_raw) else 0
            starts.append(round(float(s_), 3) if isinstance(s_, (int, float)) and 0 < s_ < 3600 else 0.0)
        cuts = []
        for n_, it in enumerate(items):   # trimmed length of a clip (seconds) or None
            d = durs[n_] if n_ < len(durs) else None
            cuts.append(round(float(d), 3) if isinstance(d, (int, float)) and 0.3 <= d < 3600 else None)
        names, sizes, timing = [], [], []
        for it in items:
            data = it["_up"][1] if "_up" in it else await self.item_bytes(it)
            tr = mp4info.track_seconds(data)   # picture / sound lengths: the montage fits the sound of every clip to its picture
            timing.append((tr["vide"], tr.get("soun")) if tr.get("vide") else None)
            name = f"h3s_m_{uuid.uuid4().hex[:10]}.{it.get('ext') or 'mp4'}"
            fd = aiohttp.FormData()
            fd.add_field("image", data, filename=name, content_type="video/mp4")
            fd.add_field("type", "input")
            status, res = await self.comfy_json("POST", "/upload/image", data=fd)
            if status != 200:
                raise web.HTTPBadGateway(text="Не удалось передать клип в ComfyUI")
            names.append(res["name"])
            mm = re.match(r"\s*(\d+)\D+(\d+)", it.get("res") or "")
            sizes.append((int(mm.group(1)), int(mm.group(2))) if mm else None)
        music = None
        if music_bytes and all(t for t in timing):
            try:
                m_start = max(0.0, float(form.get("music_start") or 0))
            except ValueError:
                m_start = 0.0
            m_name = f"h3s_mm_{uuid.uuid4().hex[:10]}.wav"
            fd = aiohttp.FormData()
            fd.add_field("image", music_bytes, filename=m_name, content_type="audio/wav")
            fd.add_field("type", "input")
            status, res = await self.comfy_json("POST", "/upload/image", data=fd)
            if status == 200:
                m_len = sum((min(max(0.1, t[0] - starts[n_]), cuts[n_]) if cuts[n_] else max(0.1, t[0] - starts[n_])) for n_, t in enumerate(timing))
                music = (res["name"], m_start, round(m_len, 3))
        try:
            mute = [bool(x) for x in json.loads(form.get("mute") or "[]")]
        except ValueError:
            mute = []
        v_total = sum((min(max(0.1, t[0] - starts[n_]), cuts[n_]) if cuts[n_] else max(0.1, t[0] - starts[n_])) for n_, t in enumerate(timing)) if timing and all(timing) else 0.0
        tracks = []
        try:
            wanted = [x for x in json.loads(form.get("atracks") or "[]") if isinstance(x, dict)][:40]
        except ValueError:
            wanted = []
        for x_ in wanted:   # sounds on the audio tracks: a piece of a Library clip or a file of the user
            try:
                src_ = str(x_.get("src") or "")
                if src_.startswith("lib:"):
                    a_item = self.library.get(src_[4:])
                    if not a_item or a_item.get("kind") == "image":
                        continue
                    a_data, a_ext = await self.item_bytes(a_item), (a_item.get("ext") or "mp4")
                elif src_ in aud_files:
                    a_data, a_ext = aud_files[src_][1], (Path(aud_files[src_][0]).suffix.lstrip(".").lower() or "wav")[:5]
                else:
                    continue
                a_name = f"h3s_ma_{uuid.uuid4().hex[:10]}.{a_ext}"
                afd = aiohttp.FormData()
                afd.add_field("image", a_data, filename=a_name, content_type="application/octet-stream")
                afd.add_field("type", "input")
                status, ares = await self.comfy_json("POST", "/upload/image", data=afd)
                if status != 200:
                    raise web.HTTPBadGateway(text="Не удалось передать звук в ComfyUI")
                tracks.append({"name": ares["name"], "at": float(x_.get("at") or 0), "a": float(x_.get("a") or 0),
                               "dur": float(x_.get("dur") or 0), "vol": float(x_.get("vol") or 0)})
            except (TypeError, ValueError):
                continue
        graph, stages = wf.montage_graph(names, sizes, cuts, timing, music, starts, mute, tracks, v_total or None)
        total = sum((cuts[n_] if cuts[n_] is not None else max(0.1, float(it.get("len") or 0) - starts[n_])) for n_, it in enumerate(items))
        return await self.queue(graph, stages, {
            "mode": "montage", "sub": f"{len(items)} клипов", "prompt": "", "aspect": items[0].get("aspect", ""), "q": "",
            "len": round(total, 1), "res": items[0].get("res", ""), "seed": 0}, last_output=True)

    async def h_grain(self, req):
        """Film grain on ONE clip (a library clip or a file from the computer) as a job of its own: the frames of a clip are decoded
        in memory, so doing the clips one after another keeps memory free between them (all at once, in one graph, it does not fit)."""
        form, content, name, ref_bytes = {}, None, "", None
        async for part in await req.multipart():
            if part.name == "file":
                name = part.filename or ""
                content = bytes(await part.read())
            elif part.name == "ref_file":   # the colour reference clip, when it comes from the user's computer
                ref_bytes = bytes(await part.read())
            else:
                form[part.name] = (await part.text()).strip()

        def num(key, default, lo, hi):
            try:
                return min(hi, max(lo, float(form.get(key) or default)))
            except ValueError:
                return default
        strength = num("strength", 0.0, 0.0, 100.0) / 100
        look = {"bright": num("bright", 100.0, 0.0, 200.0) / 100, "contrast": num("contrast", 100.0, 0.0, 200.0) / 100,
                "sat": num("sat", 100.0, 0.0, 200.0) / 100}
        want_match = form.get("match") == "1" and bool(form.get("match_id") or ref_bytes)
        look["match"] = num("ms", 100.0, 0.0, 100.0) / 100 if want_match else 0.0
        color_on = look["match"] > 0 or any(abs(look[k] - 1.0) > 0.005 for k in ("bright", "contrast", "sat"))
        if strength <= 0 and not color_on:
            raise web.HTTPBadRequest(text="Зерно и цвет выключены")
        if self.rh():
            raise web.HTTPBadRequest(text="Зерно и цвет работают только с локальным ComfyUI.")
        need = []   # (node, what it is for, which pack has it)
        if strength > 0:
            need.append(("ImageNoiseAugmentation", "зерна плёнки", "ComfyUI-KJNodes"))
        if look["match"] > 0:
            need.append(("ColorMatch", "подгонки цвета", "ComfyUI-KJNodes"))
        if abs(look["bright"] - 1.0) > 0.005:
            need.append(("AdjustBrightness", "яркости", "ComfyUI (обновите его)"))
        if abs(look["contrast"] - 1.0) > 0.005:
            need.append(("AdjustContrast", "контраста", "ComfyUI (обновите его)"))
        if abs(look["sat"] - 1.0) > 0.005:
            need.append(("JWImageSaturation", "насыщенности", "comfyui-various"))
        for node, what, pack in need:
            try:
                status, info = await self.comfy_json("GET", f"/object_info/{node}", timeout=aiohttp.ClientTimeout(total=15))
            except Exception:
                raise web.HTTPBadGateway(text="ComfyUI не отвечает. Проверьте, что он запущен.")
            if status != 200 or not isinstance(info, dict) or node not in info:
                raise web.HTTPBadRequest(text=f"Для {what} в вашем ComfyUI нужна нода {node} (пакет {pack}). "
                                              "Поставьте пакет (например, через ComfyUI Manager) или верните этот ползунок на исходное значение.")
        ref_name = None
        if look["match"] > 0:
            if ref_bytes is None:
                ref_item = self.library.get(form.get("match_id", ""))
                if not ref_item or ref_item.get("kind") == "image":
                    raise web.HTTPBadRequest(text="Клип-образец цвета не найден")
                ref_bytes = await self.item_bytes(ref_item)
            ref_name = f"h3s_cr_{uuid.uuid4().hex[:10]}.mp4"
            rfd = aiohttp.FormData()
            rfd.add_field("image", ref_bytes, filename=ref_name, content_type="video/mp4")
            rfd.add_field("type", "input")
            status, rres = await self.comfy_json("POST", "/upload/image", data=rfd)
            if status != 200:
                raise web.HTTPBadGateway(text="Не удалось передать клип-образец в ComfyUI")
            ref_name = rres["name"]
        item = {"kind": "video", "res": form.get("res", ""), "prompt": "", "aspect": ""}
        if content is None:
            item = self.library.get(form.get("id", ""))
            if not item or item.get("kind") == "image":
                raise web.HTTPBadRequest(text="Видео не найдено")
            content = await self.item_bytes(item)
            ext = item.get("ext") or "mp4"
        else:
            ext = Path(name).suffix.lstrip(".").lower()[:5] or "mp4"
        fname = f"h3s_g_{uuid.uuid4().hex[:10]}.{ext}"
        fd = aiohttp.FormData()
        fd.add_field("image", content, filename=fname, content_type="video/mp4")
        fd.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=fd)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать клип в ComfyUI")
        graph, stages = wf.grain_graph(res["name"], strength, random.randrange(2 ** 31), look, ref_name)
        try:
            ln = float(item.get("len") or form.get("len") or 0)
        except ValueError:
            ln = 0.0
        return await self.queue(graph, stages, {
            "mode": "montage", "sub": ("Цвет и зерно" if color_on and strength > 0 else "Цвет" if color_on else "Зерно плёнки"), "prompt": (item.get("prompt") or name or "")[:100], "aspect": item.get("aspect", ""), "q": "",
            "len": ln, "res": item.get("res", ""), "seed": 0}, last_output=True)

    async def h_generate_actor(self, req):
        await self.need_models("actor")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 3)
        realism = None
        v = lora_strength(form, "realism", 0.7)  # on by default, as in the original workflow
        missing = {m["role"] for m in self.missing}
        if v is not None and "realism_lora" not in missing and "realism_lora" in self.model_names:
            realism = (self.model_names["realism_lora"], v)
        width, height = wf.resolution(aspect, quality, form)
        graph, stages = wf.actor_graph(self.model_names, wf.lipsync_prompt(prompt, len(images), int(form.get("_nown") or 0)), width, height, seconds, seed,
                                       images, audios, realism, self.custom_loras("actor"), videos)
        sub = f"{len(images)} карт." + (f", голос ({len(audios)})" if audios else ", без голоса") + " · 10 шагов"
        return await self.queue(graph, stages, {
            "mode": "actor", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed})

    async def h_generate_swap(self, req):
        """Preset «Замена персонажа»: one video + up to two character pictures, the app writes the prompt."""
        await self.need_models("swap")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 2)
        if not videos:
            raise web.HTTPBadRequest(text="Добавьте видео, в котором нужно заменить человека")
        if not images:
            raise web.HTTPBadRequest(text="Добавьте изображение персонажа, на которого заменяем")
        shots = []
        if form.get("shots"):   # several plans (shots): every one is swapped on its own, then they are joined
            try:
                for x in json.loads(form["shots"])[:12]:
                    s0, e0 = float(x["s"]), float(x["e"])
                    if e0 - s0 < 0.5:
                        continue
                    shots.append({"s": s0, "e": e0, "pic": int(x.get("pic", 0)), "who": str(x.get("who") or "").strip(),
                                  "a_end": float(x["a_end"]) if x.get("a_end") not in (None, "") else None})
            except (ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest(text="Не удалось разобрать список планов")
            if any(0 <= x["pic"] and x["e"] - x["s"] < 2 for x in shots):
                raise web.HTTPBadRequest(text="Каждый план для замены должен быть не короче 2 секунд: объедините короткие планы с соседними")
        whos = [(form.get(f"who{i}") or (form.get("who") if i == 0 else "") or "").strip() for i in range(len(images))]
        try:
            cuts = [float(x) for x in (form.get("cuts") or "").replace(";", ",").split(",") if x.strip()]
        except ValueError:
            cuts = []
        text = wf.swap_prompt(whos, (form.get("extra") or "").strip(), cuts, form.get("prompt_style") != "plain")
        if form.get("raw_prompt", "").strip():   # a prompt written by hand in the MiniMax format (tests, experts): used as it is
            text = form["raw_prompt"].strip()
        pika = None
        if form.get("accel") == "pika":
            if "pika_lora" in {m["role"] for m in self.missing}:
                raise web.HTTPBadRequest(text="Файл Pika (pika_h3_flex_turbo_8step_hd_ref2va) не найден в ComfyUI. "
                                              "Скачайте его в разделе «Настройки» или выберите обычное ускорение.")
            pika = self.model_names["pika_lora"]
        names = self.model_names
        if form.get("unet"):   # another base model for this job (tests): only a file this ComfyUI really has
            try:
                _, have = await self.comfy_json("GET", "/models/diffusion_models")
            except Exception:
                have = []
            pick = next((f for f in (have or []) if f.replace("/", "\\") == form["unet"].replace("/", "\\")), None)
            if not pick:
                raise web.HTTPBadRequest(text="Такой модели нет в ComfyUI: " + form["unet"])
            names = {**self.model_names, "lipsync_unet": pick}
        width, height = wf.resolution(aspect, quality, form)
        seconds = min(15.0, max(3.0, seconds))
        loras = self.optional_loras(form, (("realism", "realism_lora"), ("motion", "motion_repair_lora"))) + self.custom_loras("swap")
        if shots:
            snd = form.get("sound") if form.get("sound") in ("original", "lock") else "original"
            graph, stages = wf.swap_shots_graph(names, shots, width, height, seed, images, videos[0],
                                                (form.get("extra") or "").strip(), pika, loras, snd,
                                                form.get("small_ref") == "1")
            seconds = max(x["e"] for x in shots) - min(x["s"] for x in shots)
        else:
            snd = form.get("sound") if form.get("sound") in ("original", "lock", "model") else "original"
            graph, stages = wf.swap_graph(names, text, width, height, seconds, seed, images, videos[0], loras, pika, snd,
                                             small_ref=form.get("small_ref") == "1")
        return await self.queue(graph, stages, {
            "mode": "swap", "sub": "Замена персонажа" + (f", планов: {len(shots)}" if shots else (f" ({len(images)})" if len(images) > 1 else "")) + (" · Pika" if pika else ""),
            "prompt": prompt or whos[0], "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed})

    async def action_setup(self):
        """What the original action workflow needs and what this ComfyUI has."""
        if not self.model_names:
            await self.resolve_models()
        _, oi = await self.comfy_json("GET", "/object_info/SelfLiftH3Sampler")
        _, oi2 = await self.comfy_json("GET", "/object_info/H3SigmaRefiner")
        _, loras = await self.comfy_json("GET", "/models/loras")
        dmad = next((n for n in loras or [] if catalog.is_action_dmad(n)), None)
        missing = {m["role"] for m in self.missing}
        have_up = "action_upscaler" not in missing
        nodes = bool(oi) and bool(oi2)
        return {
            "faithful": nodes and have_up,
            "selflift": bool(oi), "refiner": bool(oi2), "refiner_def": (oi2 or {}).get("H3SigmaRefiner"),
            "upscaler": have_up, "dmad": dmad,
            "speed": None if "speed_slider_lora" in missing else self.model_names.get("speed_slider_lora"),
        }

    async def h_action_info(self, _req):
        try:
            s = await self.action_setup()
        except aiohttp.ClientError:
            return web.json_response({"error": "ComfyUI не отвечает"}, status=503)
        s.pop("refiner_def", None)
        return web.json_response(s)

    async def h_generate_action(self, req):
        await self.need_models("action")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 6)
        if self.rh():
            # on RunningHub only built-in nodes are assumed; LoRAs are used if the user's RH workflow has them
            rh_files = {n.replace("\\", "/").split("/")[-1].lower(): n for n in (getattr(self, "rh_models", None) or [])}
            speed = rh_files.get(catalog.VARIANTS["speed_slider_lora"]["v1.1"]["file"].lower())
            dmad = next((n for b, n in rh_files.items() if catalog.is_action_dmad(b)), None)
            s = {"faithful": False, "dmad": dmad, "speed": speed}
        else:
            s = await self.action_setup()
        loras = []
        for key, name, default in (("dmad", s["dmad"], 0.8), ("speed", s["speed"], 4.0)):
            v = lora_strength(form, key, default)  # the user can switch them off or change the strength
            if name and v is not None:
                loras.append((name, v))
        loras += self.custom_loras("action")
        names = dict(self.model_names)
        width, height = wf.resolution(aspect, quality, form)
        preview = await self.preview_opts("action") if form.get("preview") != "0" and not self.rh() else None
        graph, stages = wf.action_graph(names, wf.lipsync_prompt(prompt, len(images), int(form.get("_nown") or 0)), width, height, seconds, seed, images, audios,
                                        s["faithful"], loras, preview, videos)
        if s["faithful"]:
            # the refiner's input names come from the installed node; values as in the original workflow
            req_inputs = [k for k in (s["refiner_def"]["input"].get("required") or {}) if k != "sigmas"]
            graph["13"]["inputs"] = {"sigmas": ["10", 0], **dict(zip(req_inputs, [1, 0.7, 0.0, "cosine"]))}
        sub = ("точная копия workflow" if s["faithful"] else "упрощённая схема") + f" · {len(images)} карт."
        return await self.queue(graph, stages, {
            "mode": "action", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "faithful": s["faithful"]})

    # ---------- audio: voice / music separation (MelBand RoFormer through ComfyUI) ----------

    async def separation_ready(self):
        """(ok, reason) - is the node pack there and is the model visible to ComfyUI."""
        if self.rh():
            return False, "Отделение голоса работает на этом компьютере, а сейчас включено облако."
        try:
            _, node = await self.comfy_json("GET", "/object_info/MelBandRoFormerSampler")
        except aiohttp.ClientError:
            return False, "ComfyUI не запущен."
        if not node:
            return False, "В ComfyUI нет ноды MelBandRoFormer (ComfyUI-MelBandRoFormer)."
        if not self.model_names:
            await self.resolve_models()
        if "vocal_model" in {m["role"] for m in self.missing}:
            return False, "Не найдена модель MelBandRoformer_fp16.safetensors (папка diffusion_models)."
        return True, ""

    async def h_audio_info(self, _req):
        ok, reason = await self.separation_ready()
        return web.json_response({"available": ok, "reason": reason})

    async def h_audio_separate(self, req):
        """WAV in -> the voice only (FLAC) out. Blocks until ComfyUI is done (usually well under a minute)."""
        ok, reason = await self.separation_ready()
        if not ok:
            return web.json_response({"error": reason}, status=400)
        name = None
        async for field in await req.multipart():
            if field.name == "file" and field.filename:
                name = await self.upload_input(field)
        if not name:
            return web.json_response({"error": "Нет аудио"}, status=400)
        await self.stop_llm()  # the agent's model would compete for video memory
        graph = {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": name}},
            "2": {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": self.model_names["vocal_model"]}},
            "3": {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["2", 0], "audio": ["1", 0]}},
            "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "H3Workspace/audio/vocals"}},
        }
        status, res = await self.comfy_json("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        loop = asyncio.get_running_loop()
        end = loop.time() + 600
        while loop.time() < end:
            await asyncio.sleep(1.5)
            _, hist = await self.comfy_json("GET", f"/history/{pid}")
            entry = (hist or {}).get(pid)
            if not entry:
                continue
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                return web.json_response({"error": "Не удалось отделить голос: " + (msgs[0].strip() if msgs else "ошибка ComfyUI")}, status=500)
            audio = next((f for out in entry.get("outputs", {}).values() for f in out.get("audio", [])), None)
            if audio:
                async with self.session.get(self.comfy + "/view", params={
                        "filename": audio["filename"], "subfolder": audio.get("subfolder", ""), "type": audio.get("type", "output")}) as r:
                    data = await r.read()
                return web.Response(body=data, content_type="audio/flac")
        return web.json_response({"error": "Отделение голоса заняло слишком много времени"}, status=504)

    # ---------- Voice Lab (OmniVoice through ComfyUI, see voicelab.py) ----------

    async def voice_model_options(self):
        """(node known, the model names the node offers). The node list changes when the user downloads a model."""
        try:
            _, oi = await self.comfy_json("GET", "/object_info/OmniVoiceVoiceDesignTTS", timeout=aiohttp.ClientTimeout(total=15))
        except aiohttp.ClientError:
            return None, []
        info = (oi or {}).get("OmniVoiceVoiceDesignTTS")
        if not info:
            return False, []
        try:
            return True, list(info["input"]["required"]["model"][0])
        except (KeyError, IndexError, TypeError):
            return True, []

    async def running_comfy_python(self):
        """The python.exe of the ComfyUI that answers on our port (the user's own portable build): its folder tells where custom_nodes is."""
        try:
            port = int(str(self.comfy).rsplit(":", 1)[-1].strip("/"))
        except ValueError:
            return None
        cmd = f"(Get-Process -Id (Get-NetTCPConnection -LocalPort {port} -State Listen | Select-Object -First 1).OwningProcess).Path"
        try:
            proc = await asyncio.create_subprocess_exec("powershell", "-NoProfile", "-NonInteractive", "-Command", cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.DEVNULL, creationflags=engine.CREATE_NO_WINDOW)
            out, _ = await asyncio.wait_for(proc.communicate(), 20)
        except (OSError, asyncio.TimeoutError):
            return None
        path = Path(out.decode("utf-8", "ignore").strip().splitlines()[-1].strip()) if out.strip() else None
        return path if path and path.name.lower() == "python.exe" and path.is_file() else None

    async def h_voice_progress(self, _req):
        """How far the download of the voice conversion model is: it lands in <ComfyUI>/models/tts/chatterbox, unfinished files end with .incomplete."""
        dirs = []
        if self.config.get("engine") == "managed" and self.comfy_root():
            dirs.append(self.comfy_root() / "ComfyUI" / "models" / "tts" / "chatterbox")
        py = getattr(self, "_vc_py", None) or await self.running_comfy_python()
        if py:
            self._vc_py = py
            dirs.append(py.parent.parent / "ComfyUI" / "models" / "tts" / "chatterbox")
        size, busy = 0, False
        for d in dirs:
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        size += f.stat().st_size
                    except OSError:
                        continue
                    busy = busy or f.name.endswith(".incomplete")
        return web.json_response({"downloading": busy, "bytes": size, "total": 3_200_000_000})

    async def voice_vc_info(self):
        """The object_info of the voice conversion node (Chatterbox VC) or None when it is not installed."""
        try:
            _, oi = await self.comfy_json("GET", "/object_info/ChatterboxVC", timeout=aiohttp.ClientTimeout(total=15))
        except aiohttp.ClientError:
            return None
        return (oi or {}).get("ChatterboxVC") or None

    async def h_voice_status(self, _req):
        if self.rh():
            return web.json_response({"ready": False, "state": "cloud", "reason": "Voice Lab работает на этом компьютере, а сейчас включено облако RunningHub."})
        have, models = await self.voice_model_options()
        if have is None:
            return web.json_response({"ready": False, "state": "offline", "reason": "ComfyUI не запущен."})
        if not have:
            nd = self.dlss_nodes_dir()
            if nd and (nd / voicelab.VOICE_PACK["name"]).is_dir():   # the files are there, but the node did not load: say why (the engine log tells)
                tail = self.process.log_tail(400) if self.process else ""
                why = next((ln.strip() for ln in reversed(tail.splitlines()) if "[OmniVoice]" in ln and ("fail" in ln.lower() or "missing" in ln.lower() or "broken" in ln.lower())), "")
                return web.json_response({"ready": False, "state": "nonode", "installed": True,
                                          "reason": "Модуль OmniVoice установлен, но не запускается" + (f": {why}" if why else "") + ". Нажмите кнопку «Установить модуль голоса»: я сама доустановлю всё недостающее. Потом нажмите «Перезапустить программу», и всё заработает."})
            return web.json_response({"ready": False, "state": "nonode", "reason": "Для голосов нужен модуль OmniVoice. Его можно поставить кнопкой ниже."})
        local = [m for m in models if "auto download" not in m]
        return web.json_response({"ready": True, "model": voicelab.pick_model(models), "models": models, "downloaded": bool(local), "vc": bool(await self.voice_vc_info())})

    async def h_voice_install(self, req):
        """Puts the OmniVoice node pack into custom_nodes and installs its python packages (the node's own install.py).
        ComfyUI has to be restarted afterwards: it reads custom_nodes only at start."""
        import zipfile
        try:
            body = await req.json()
        except Exception:
            body = {}
        nodes_dir = self.dlss_nodes_dir()
        auto_py = None
        if not nodes_dir and not body.get("dir"):   # the user's own ComfyUI: find it by the running process (python_embeded\\python.exe next to ComfyUI\\custom_nodes)
            auto_py = await self.running_comfy_python()
            cand = auto_py.parent.parent / "ComfyUI" / "custom_nodes" if auto_py else None
            if cand and cand.is_dir():
                nodes_dir = cand
                self.config.update(own_comfy_dir=str(auto_py.parent.parent))
        if not nodes_dir:
            raw = str(body.get("dir") or self.config.get("own_comfy_dir") or "").strip().strip('"')
            base = Path(raw) if raw else None
            nodes_dir = next((c for c in (base / "custom_nodes", base / "ComfyUI" / "custom_nodes") if c.is_dir()), None) if base else None
            if not nodes_dir:
                return web.json_response({"error": "Не нашла папку custom_nodes вашего ComfyUI. Укажите папку ComfyUI (где лежит custom_nodes)."}, status=400)
            self.config.update(own_comfy_dir=raw)
        pack = voicelab.VC_PACK if body.get("pack") == "vc" else voicelab.VOICE_PACK
        target = nodes_dir / pack["name"]
        managed = self.config.get("engine") == "managed"
        root = self.comfy_root() if managed else None
        cands = [nodes_dir.parent.parent / "python_embeded" / "python.exe", nodes_dir.parent / "python_embeded" / "python.exe"]
        if root:
            cands.insert(0, root / "python_embeded" / "python.exe")
        py = auto_py or next((c for c in cands if c.exists()), None)
        if not target.exists():
            zpath = nodes_dir.parent / f'_h3_{pack["name"]}.zip'
            ok = False
            for url in (f'https://github.com/{pack["repo"]}/archive/{pack["commit"]}.zip', f'https://codeload.github.com/{pack["repo"]}/zip/{pack["commit"]}'):
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as r:
                        if r.status != 200:
                            continue
                        zpath.write_bytes(await r.read())
                    with zipfile.ZipFile(zpath) as z:
                        top = z.namelist()[0].split("/")[0]
                        z.extractall(nodes_dir)
                    (nodes_dir / top).rename(target)
                    ok = True
                    break
                except Exception as e:
                    print(f"Voice Lab: {type(e).__name__}: {e}", file=sys.stderr)
            zpath.unlink(missing_ok=True)
            if not ok:
                return web.json_response({"error": "Не удалось скачать модуль голоса. Проверьте интернет и попробуйте ещё раз."}, status=502)
        pip_ok = None
        script = ["-s", str(target / "install.py")] if (target / "install.py").is_file() else \
            (["-s", "-m", "pip", "install", "-r", str(target / "requirements.txt")] if (target / "requirements.txt").is_file() else None)   # Chatterbox has only a requirements file
        if py and script:
            extra = {"env": engine.local_env(root)} if root else {}
            proc = await asyncio.create_subprocess_exec(str(py), *script, cwd=str(target),
                                                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                        creationflags=engine.CREATE_NO_WINDOW, **extra)
            try:
                await asyncio.wait_for(proc.wait(), 1500)
                pip_ok = proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                pip_ok = False
        if py and pack is voicelab.VOICE_PACK:
            if not await self.voice_pip_fix(py, root):
                pip_ok = False
        return web.json_response({"ok": True, "pip": pip_ok, "python_found": bool(py)})

    async def voice_pip_fix(self, py, root):
        """The node installs «omnivoice» with --no-deps, so some of its libraries are missing: Python 3.13 has no built-in «audioop» (pydub fails with
        «No module named pyaudioop»: audioop-lts brings it back), and transformers needs «accelerate» to load the model (device_map).
        Then anything else the import still lacks is installed one by one («No module named X»). True when everything is in place."""
        extra = {"env": engine.local_env(root)} if root else {}
        ok = True

        async def run_py(*args, timeout=600):
            p = await asyncio.create_subprocess_exec(str(py), "-s", *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                                                     creationflags=engine.CREATE_NO_WINDOW, **extra)
            _, err = await asyncio.wait_for(p.communicate(), timeout)
            return p.returncode, err.decode("utf-8", "ignore")
        try:
            for pkg, mod in (("audioop-lts", "audioop"), ("accelerate", "accelerate")):
                if (await run_py("-c", f"import {mod}", timeout=120))[0] != 0:
                    if (await run_py("-m", "pip", "install", pkg))[0] != 0:
                        ok = False
            tried = set()
            for _ in range(6):
                rc, err = await run_py("-c", "import omnivoice", timeout=300)
                m = re.search(r"No module named '([A-Za-z0-9_\-]+)", err)
                if rc == 0 or not m or m.group(1) in tried:
                    if rc != 0:
                        ok = False
                    break
                if m.group(1) in ("omnivoice", "torch", "torchaudio", "torchvision"):   # never pulled in here: omnivoice pins its own torch and would replace the GPU build
                    ok = False
                    break
                name = {"pyaudioop": "audioop-lts", "yaml": "pyyaml", "cv2": "opencv-python-headless"}.get(m.group(1), m.group(1))
                tried.add(m.group(1))
                if (await run_py("-m", "pip", "install", name))[0] != 0:
                    ok = False
                    break
        except (OSError, asyncio.TimeoutError) as e:
            print(f"Voice Lab packages: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
        return ok

    # ---------- Reference Lab (see reflab.py) ----------

    async def h_reflab_folder(self, _req):
        d = self.reflab.root
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))
        return web.json_response({"ok": True})

    async def h_reflab_describe(self, req):
        """«Описать»: the agent's model looks at the pictures and writes a short 'who' phrase and the features to keep."""
        try:
            body = await req.json()
        except ValueError:
            raise web.HTTPBadRequest(text="Не удалось прочитать данные")
        imgs = [u for u in (body.get("images") or [])[:4] if isinstance(u, str) and u.startswith("data:image")]
        typ = body.get("type") if body.get("type") in reflab.TYPES else "character"
        if not imgs:
            raise web.HTTPBadRequest(text="Сначала добавьте картинку: по ней программа составит описание")
        if self.jobs:
            raise web.HTTPBadRequest(text="Идёт генерация: описание можно будет сделать, когда она закончится.")
        variant = await self.llm_variant()
        if not self.llm.installed(variant):
            raise web.HTTPBadRequest(text="Модель агента ещё не скачана: её можно скачать в разделе «Агент». Пока впишите описание вручную.")
        if not self.llm.vision(variant):
            raise web.HTTPBadRequest(text="У выбранной модели агента нет файла «зрения» (mmproj): она не видит картинки. Впишите описание вручную.")
        if not self.llm.alive():
            try:
                await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
            except Exception:
                pass
            self.llm.start(variant)
            await ag.wait_ready(self.session, self.llm)
        what = {"character": "the person or creature", "object": "the object", "location": "the place"}[typ]
        ask = (f"Look at the picture(s): {what} is shown. Reply with ONLY a JSON object with two string fields. "
               '"who": one short English noun phrase starting with "the" (like "the man in a dark denim jacket"). '
               '"features": one English sentence listing what must stay the same in every view (colours, materials, clothes, marks, shapes, style). '
               "No other text.")
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": ask}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]}],
                   "stream": False, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}, "thinking_budget_tokens": 0}
        try:
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=240)) as r:
                if r.status != 200:
                    raise RuntimeError(f"ошибка {r.status}")
                text = (await r.json())["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raise web.HTTPBadGateway(text=f"Не удалось описать картинку ({e}). Впишите описание вручную.")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except ValueError:
            d = {}
        who, feats = reflab.clean(d.get("who"), 300), reflab.clean(d.get("features"), 1200)
        if not who and not feats:
            who = reflab.clean(text, 300)
        return web.json_response({"who": who, "features": feats})

    async def h_voice_cancel(self, _req):
        self.voice_stop = True
        pid = getattr(self, "voice_pid", None)
        if pid:
            await self.comfy_json("POST", "/queue", json={"delete": [pid]})   # still waiting in line
            await self.comfy_json("POST", "/interrupt", json={"prompt_id": pid})   # already running
        else:
            await self.comfy_json("POST", "/interrupt", json={})
        return web.json_response({"ok": True})

    async def voice_upload(self, raw, suffix):
        data = aiohttp.FormData()
        name = f"h3v_{uuid.uuid4().hex[:8]}{suffix or '.wav'}"
        data.add_field("image", raw, filename=name, content_type="application/octet-stream")
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось передать звук в ComfyUI")
        return f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]

    async def h_voice_generate(self, req):
        """One speech: mode design / clone / dialog. Blocks until ComfyUI is done (the first run also downloads the model: minutes)."""
        if self.rh():
            return web.json_response({"error": "Voice Lab работает на этом компьютере, а сейчас включено облако."}, status=400)
        p, files = {}, {}
        try:
            async for field in await req.multipart():
                if field.name == "params":
                    p = json.loads((await field.read()).decode("utf-8"))
                elif field.name and field.filename:
                    raw = bytes(await field.read())
                    if len(raw) > voicelab.MAX_UPLOAD:
                        return web.json_response({"error": "Файл слишком большой (до 60 МБ)"}, status=400)
                    files[field.name] = (Path(field.filename).suffix.lower(), raw)
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "Не удалось прочитать данные"}, status=400)
        mode = p.get("mode")
        text = str(p.get("text") or "").strip()
        if not text and mode != "vc":
            return web.json_response({"error": "Напишите, что должен сказать голос"}, status=400)
        if len(text) > 20000:
            return web.json_response({"error": "Текст слишком длинный (до 20 000 знаков)"}, status=400)
        if mode == "vc":
            vc_info = await self.voice_vc_info()
            if not vc_info:
                return web.json_response({"error": "В ComfyUI нет модуля преобразования голоса (Chatterbox) или ComfyUI не запущен. Установите его на вкладке «Своя интонация»."}, status=400)
            models = []
        else:
            have, models = await self.voice_model_options()
            if not have:
                return web.json_response({"error": "В ComfyUI нет модуля OmniVoice или ComfyUI не запущен."}, status=400)
        model = p.get("model") if p.get("model") in models else voicelab.pick_model(models)

        raws = {}

        async def sample(key, saved_id):
            """A sample for cloning: an uploaded file or a saved voice -> the name inside ComfyUI."""
            if saved_id and str(saved_id).startswith("take:"):
                f = self.voices.out_path(str(saved_id)[5:])
                if not f:
                    raise web.HTTPBadRequest(text="Этого варианта больше нет: он удалён")
                raws[key] = f.read_bytes()
                return await self.voice_upload(raws[key], f.suffix)
            if saved_id:
                f = self.voices.path(str(saved_id))
                if not f:
                    raise web.HTTPBadRequest(text="Этого сохранённого голоса больше нет")
                raws[key] = f.read_bytes()
                return await self.voice_upload(raws[key], f.suffix)
            if key in files:
                if files[key][0] not in voicelab.AUD_EXT:
                    raise web.HTTPBadRequest(text="Нужен звуковой файл (mp3, wav, flac, m4a, ogg)")
                raws[key] = files[key][1]
                return await self.voice_upload(raws[key], files[key][0])
            return ""

        async def words_of(key, typed):
            """What is said in the sample: the typed words, or the app's own speech recognition (the node would download its own Whisper: minutes)."""
            typed = str(typed or "").strip()[:1000]
            if typed or key not in raws:
                return typed
            await self.llm_variant()
            if not self.speech_status()["ready"]:
                raise web.HTTPBadRequest(text="Программа не может сама понять слова в образце: модель распознавания речи не скачана (карточка «Модель» в разделе «Агент»). Впишите, что сказано в образце, вручную.")
            try:
                text, _, _ = await speech.transcribe(self.llm.root, raws[key], "auto", self.config.get("whisper_dir"))
            except Exception as e:
                raise web.HTTPBadRequest(text="Не удалось распознать слова в образце (" + str(e)[:200] + "). Впишите их вручную.")
            if not text:
                raise web.HTTPBadRequest(text="В образце не слышно слов. Возьмите запись с чистой речью или впишите слова вручную.")
            return text[:1000]

        try:
            if mode == "design":
                graph = voicelab.design_graph(model, text, p)
            elif mode == "clone":
                ref = await sample("ref", p.get("ref_saved"))
                if not ref:
                    return web.json_response({"error": "Выберите образец голоса: загрузите файл или возьмите сохранённый голос"}, status=400)
                graph = voicelab.clone_graph(model, text, p, ref, await words_of("ref", p.get("ref_text")))
            elif mode == "vc":
                src = await sample("src", p.get("src_saved"))
                if not src:
                    return web.json_response({"error": "Загрузите свою запись: то, что вы наговорили с нужной интонацией"}, status=400)
                ref = await sample("ref", p.get("ref_saved"))
                if not ref:
                    return web.json_response({"error": "Выберите, чьим голосом сказать: загрузите образец или возьмите сохранённый голос"}, status=400)
                graph = voicelab.convert_graph(vc_info, src, ref, p)
            elif mode == "dialog":
                speakers = []
                for i, sp in enumerate((p.get("speakers") or [])[:voicelab.MAX_SPEAKERS], 1):
                    nm = await sample(f"sp{i}", sp.get("saved"))
                    if not nm:
                        return web.json_response({"error": f"Для голоса {i} выберите образец: файл или сохранённый голос"}, status=400)
                    speakers.append((nm, await words_of(f"sp{i}", sp.get("ref_text"))))
                if len(speakers) < 2:
                    return web.json_response({"error": "В диалоге нужно минимум два голоса"}, status=400)
                graph = voicelab.dialog_graph(model, voicelab.dialog_text(text, len(speakers)), p, speakers)
            else:
                return web.json_response({"error": "Неизвестный режим"}, status=400)
        except web.HTTPException as e:
            return web.json_response({"error": e.text or "Ошибка"}, status=e.status)
        await self.stop_llm()   # the agent's model would compete for video memory
        status, res = await self.comfy_json("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        self.voice_pid, self.voice_stop = pid, False
        loop = asyncio.get_running_loop()
        end = loop.time() + 2400
        while loop.time() < end:
            await asyncio.sleep(1.5)
            if self.voice_stop:
                self.voice_pid = None
                return web.json_response({"error": "Остановлено"}, status=499)
            _, hist = await self.comfy_json("GET", f"/history/{pid}")
            entry = (hist or {}).get(pid)
            if not entry:
                continue
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                m0 = (msgs[0].strip() if msgs else "ошибка ComfyUI")
                if "nterrupt" in m0:
                    return web.json_response({"error": "Остановлено"}, status=499)
                if re.search(r"requires .accelerate|No module named|pyaudioop", m0):   # a library of the voice module is missing: put it in, the restart makes ComfyUI see it
                    py = await self.running_comfy_python()
                    managed = self.config.get("engine") == "managed" and self.comfy_root()
                    py = py or (self.comfy_root() / "python_embeded" / "python.exe" if managed else None)
                    if py and Path(py).is_file() and await self.voice_pip_fix(py, self.comfy_root() if managed else None):
                        return web.json_response({"error": "Для голоса не хватало служебной библиотеки, я её доустановила. Теперь нажмите «Перезапустить программу» и повторите озвучку.", "fixed": True}, status=500)
                return web.json_response({"error": "Не удалось сделать голос: " + m0[:600]}, status=500)
            audio = next((f for out in entry.get("outputs", {}).values() for f in out.get("audio", [])), None)
            if audio:
                async with self.session.get(self.comfy + "/view", params={
                        "filename": audio["filename"], "subfolder": audio.get("subfolder", ""), "type": audio.get("type", "output")}) as r:
                    data = await r.read()
                inp = next((n["inputs"] for n in graph.values() if str(n.get("class_type", "")).startswith("OmniVoice")), {})
                if mode == "vc":
                    inp = {"steps": graph["1"]["inputs"].get("n_timesteps")}
                    text = "Своя интонация, другой голос"
                meta = {"mode": mode, "text": text[:400], "seed": inp.get("seed"), "steps": inp.get("steps"), "params": {k: v for k, v in p.items() if k not in ("text",)}}
                take = self.voices.add_take(data, Path(audio["filename"]).suffix or ".mp3", meta)
                return web.json_response(take)
        return web.json_response({"error": "Озвучка заняла слишком много времени"}, status=504)

    async def h_interrupt(self, req):
        pid = (await req.json()).get("prompt_id")
        job = self.jobs.get(pid)
        if job and job.get("task"):  # cloud task
            self.jobs.pop(pid, None)
            rh = self.rh() or self.rh_client()
            if rh:
                try:
                    await rh.cancel(job["task"])
                except cloud.CloudError:
                    pass
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        elif job and job["state"] == "queued":
            await self.comfy_json("POST", "/queue", json={"delete": [pid]})
            self.jobs.pop(pid, None)
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        else:
            await self.comfy_json("POST", "/interrupt", json={"prompt_id": pid})
            task = asyncio.create_task(self.interrupt_watch(pid))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        return web.json_response({"ok": True})

    async def interrupt_watch(self, pid, wait=15):
        """ComfyUI sometimes ignores «Cancel» (stuck on a heavy step). If the job is still there after `wait` seconds:
        managed engine - restart it; the user's own ComfyUI - only tell the user."""
        await asyncio.sleep(wait)
        if pid not in self.jobs:
            return
        if not self.process:
            text = ("ComfyUI не отвечает на «Отмена» уже больше 15 секунд. Это ваш собственный ComfyUI: "
                    "закройте его окно и запустите заново.")
            self.note_error(text)
            await self.broadcast({"type": "notice", "text": text})
            return
        text = "ComfyUI не отвечает на «Отмена». Перезапускаем движок, это займёт около минуты."
        self.note_error(text)
        await self.broadcast({"type": "notice", "text": text})
        try:
            await asyncio.to_thread(self.process.stop)
            await asyncio.to_thread(self.process.start)
            await self.wait_comfy()
        except Exception as e:
            self.note_error(f"Не удалось перезапустить движок после отмены: {e}")
        # the restarted engine has an empty queue: nothing of what was running or waiting is there any more
        for jid in [j for j, job in self.jobs.items() if not job.get("task")]:
            self.jobs.pop(jid, None)
            await self.broadcast({"type": "interrupted", "prompt_id": jid})

    async def jobs_watchdog(self):
        """Kill jobs that ComfyUI has lost: e.g. it was restarted while the app waited for events that will
        never come («Ждёт, пока освободится ComfyUI» forever). A job sitting in ComfyUI's queue is legitimately
        waiting, so only a job ABSENT from the queue and from the history is stale."""
        while True:
            await asyncio.sleep(60)
            try:
                if self.rh() or not self.jobs:
                    continue
                _, q = await self.comfy_json("GET", "/queue", timeout=aiohttp.ClientTimeout(total=10))
                running_ids = {str(item[1]) for item in (q or {}).get('queue_running', [])}
                queued_ids = running_ids | {str(item[1]) for item in (q or {}).get('queue_pending', [])}
            except Exception:
                continue   # ComfyUI unreachable or restarting: not the watchdog's case
            now = asyncio.get_event_loop().time()
            for pid, job in list(self.jobs.items()):
                if job.get("task"):   # cloud (RunningHub) jobs are watched by their own poller
                    continue
                age, silent = now - job.get("t0", now), now - job.get("last_ev", job.get("t0", now))
                if pid in queued_ids:
                    # В старых снимках нет client_id: когда задание начнётся, скажет сама очередь.
                    if job.get('recovered') and job['state'] == 'queued' and pid in running_ids:
                        job['state'] = 'running'
                        job['last_ev'] = now
                        await self.send_stage(job, 'running')
                        continue
                    if job["state"] == "running" and silent > watchdog.QUIET_NOTICE:
                        # ComfyUI всё ещё держит задание: длинный клип тратит ~20 минут на один шаг,
                        # поэтому тишина сама по себе — не зависание; только часы тишины при простое карты
                        gpu = await asyncio.to_thread(self.monitor.gpu)
                        busy = None if not gpu else (gpu.get("util") or 0) >= 5
                        if watchdog.job_verdict(True, silent, busy) == "notice":
                            if not job.get("quiet_notice"):
                                job["quiet_notice"] = True
                                await self.broadcast({"type": "notice", "text":
                                    "Долгий шаг: ComfyUI молчит больше 10 минут. Это не зависание — "
                                    "длинный клип считает один шаг десятки минут, ждём."})
                            continue
                        if self.process:   # managed engine: restart it, like «Остановить генерацию» does
                            text = ("Генерация не подаёт признаков жизни больше двух часов, а видеокарта простаивает. "
                                    "Перезапускаем движок, это займёт около минуты.")
                            self.note_error(text)
                            await self.broadcast({"type": "notice", "text": text})
                            try:
                                await asyncio.to_thread(self.process.stop)
                                await asyncio.to_thread(self.process.start)
                                await self.wait_comfy()
                            except Exception as e:
                                self.note_error(f"Не удалось перезапустить движок после зависания: {e}")
                            for jid, j in list(self.jobs.items()):
                                if not j.get("task"):
                                    self.jobs.pop(jid, None)
                                    await self.broadcast({"type": "error" if jid == pid else "interrupted",
                                                          "prompt_id": jid,
                                                          **({"message": "Генерация зависла в ComfyUI и была снята при перезапуске движка. "
                                                                          "Отправьте её ещё раз."} if jid == pid else {})})
                        else:
                            await self._stale_job(job, "Генерация не подаёт признаков жизни больше двух часов, "
                                                       "а видеокарта при этом простаивает. Проверьте окно ComfyUI "
                                                       "и отправьте генерацию ещё раз.")
                    continue
                if job["state"] == "queued" and age > 240:
                    _, hist = await self.comfy_json("GET", f"/history/{pid}")
                    if (hist or {}).get(pid):   # it actually finished while nobody watched: collect the result
                        try:
                            await self.finish_job(job)
                        except Exception:
                            self.jobs.pop(pid, None)
                            await self.broadcast({"type": "interrupted", "prompt_id": pid})
                    else:
                        await self._stale_job(job, "ComfyUI это задание не получил (скорее всего, он перезапускался). "
                                                   "Отправьте генерацию ещё раз.")
                elif job["state"] == "running" and silent > 900:
                    _, hist = await self.comfy_json("GET", f"/history/{pid}")
                    if (hist or {}).get(pid):
                        try:
                            await self.finish_job(job)
                        except Exception:
                            self.jobs.pop(pid, None)
                            await self.broadcast({"type": "interrupted", "prompt_id": pid})
                    else:
                        await self._stale_job(job, "Связь с ComfyUI прервалась: задание потеряно. "
                                                   "Отправьте генерацию ещё раз.")

    async def jobs_poller(self):
        """Safety net: if the events of ComfyUI never reach the app (a stale connection), a finished job would hang in
        the window on «В очереди» for minutes. Every few seconds ask the history about jobs that have been quiet."""
        while True:
            await asyncio.sleep(4)
            try:
                if self.rh() or not self.jobs:
                    continue
                now = asyncio.get_event_loop().time()
                for pid, job in list(self.jobs.items()):
                    if job.get("task") or pid not in self.jobs:
                        continue
                    if now - job.get("last_ev", job.get("t0", now)) < 12:
                        continue
                    _, hist = await self.comfy_json("GET", f"/history/{pid}", timeout=aiohttp.ClientTimeout(total=10))
                    entry = (hist or {}).get(pid)
                    if not entry or pid not in self.jobs:
                        continue
                    st = (entry.get("status") or {})
                    if st.get("status_str") == "success" or st.get("completed"):
                        self.note_error(f"Результат забран из истории ComfyUI: события о конце задания не пришли (задание {pid[:8]})")
                        await self.finish_job(job)
                    elif st.get("status_str") == "error":
                        msgs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                        await self._stale_job(job, "Ошибка генерации в ComfyUI: " + (msgs[0].strip() if msgs else "неизвестная ошибка"))
            except Exception:
                continue

    async def _stale_job(self, job, message):
        pid = job["prompt_id"]
        job["state"] = "error"
        self.note_error("Задание зависло и снято: " + message)
        await self.broadcast({"type": "error", "prompt_id": pid, "message": message})
        self.jobs.pop(pid, None)

    async def h_library(self, _req):
        return web.json_response([self.public(x) for x in self.library.items])

    async def h_library_patch(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        body = await req.json()
        if "fav" in body:
            item["fav"] = bool(body["fav"])
            self.library.save()
        return web.json_response(self.public(item))

    async def h_library_delete(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        res = await self.delete_items([item])
        if res["failed"]:
            return web.json_response({"error": "Не удалось переместить файл в корзину"}, status=500)
        return web.json_response({"ok": True, "recycled": res["recycled"]})

    async def delete_items(self, items):
        """Library items to the Recycle Bin (the original and the program's own copy), all files in one go."""
        todo, files = [], {}
        for item in items:
            paths = {p for p in (self.file_path(item), Path(item["local"]) if item.get("local") else None) if p and p.exists()}
            files[item["id"]] = paths
            todo += paths
        gone = set(await asyncio.to_thread(send_to_recycle_bin, todo)) if todo else set()
        done, failed = [], []
        for item in items:
            if files[item["id"]] <= gone:
                self.library.remove(item["id"])
                done.append(item["id"])
            else:
                failed.append(item["id"])
        return {"done": done, "failed": failed, "recycled": bool(gone)}

    async def h_library_delete_many(self, req):
        ids = [i for i in (await req.json()).get("ids", []) if isinstance(i, str)][:2000]
        items = [x for x in (self.library.get(i) for i in ids) if x]
        return web.json_response(await self.delete_items(items))

    async def h_results_get(self, _req):
        items = self.library.items
        lost = sum(1 for x in items if (p := self.file_path(x)) and not p.exists())
        nocopy = sum(1 for x in items if not (x.get("local") and Path(x["local"]).exists()) and (p := self.file_path(x)) and p.exists())
        return web.json_response({"dir": str(self.results_dir()), "keep": self.keep_copies(), "lost": lost, "nocopy": nocopy,
                                  "default": str(self.default_results_dir()),
                                  "output": str(self.output_dir) if self.output_dir else None})

    async def h_results_open(self, _req):
        """Open the folder where ComfyUI puts the results (the one place of the files; with a copy switched on, the copy folder is next to it)."""
        d = Path(self.output_dir) if self.output_dir else self.results_dir()
        if not d.is_dir():
            raise web.HTTPBadRequest(text="Папка с результатами программе не известна: откройте output вашего ComfyUI сами.")
        try:
            subprocess.Popen(["explorer", str(d)])
        except Exception:
            pass
        return web.json_response({"ok": True, "path": str(d)})

    def locked(self):
        """«Я новичок» with the simple view: the model set is fixed. «Для новичков с опытом» (full_mode) opens own models and LoRAs on the same ComfyUI."""
        return self.config.get("engine") == "managed" and not self.config.get("full_mode")

    async def h_full_mode_set(self, req):
        self.config.update(full_mode=bool((await req.json()).get("full")))
        return web.json_response({"full": bool(self.config.get("full_mode"))})

    async def h_window_gpu_get(self, _req):
        return web.json_response({"software": bool(self.config.get("software_ui"))})

    async def h_window_gpu_set(self, req):
        body = await req.json()
        self.config.update(software_ui=bool(body.get("software")))   # read by main.py at the next start
        return await self.h_window_gpu_get(req)

    async def h_results_set(self, req):
        body = await req.json()
        if "keep" in body:
            self.config.update(keep_copies=bool(body["keep"]))
        d = (body.get("dir") or "").strip().strip('"')
        if d:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except OSError:
                return web.json_response({"error": "Не получается создать папку. Проверьте путь и диск"}, status=400)
            self.config.update(results_dir=d)
        elif body.get("reset"):
            self.config.update(results_dir=None)
        return await self.h_results_get(req)

    async def h_results_backfill(self, req):
        """Make copies of the files that are still in ComfyUI's output folder but have no copy of their own yet."""
        n = 0
        for x in list(self.library.items):
            p = self.file_path(x)
            if p and p.exists() and not (x.get("local") and Path(x["local"]).exists()):
                n += 1 if await self.archive_item(x) else 0
        return web.json_response({"copied": n, **json.loads((await self.h_results_get(req)).text)})

    async def h_results_clean(self, req):
        """Remove the Library records whose files are gone from the disk (nothing on the disk is touched)."""
        gone = [x["id"] for x in self.library.items if (p := self.file_path(x)) and not p.exists()]
        for i in gone:
            self.library.remove(i)
        return web.json_response({"removed": len(gone), **json.loads((await self.h_results_get(req)).text)})

    async def h_file(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        f = item["file"]
        headers = {}
        if req.query.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{f["filename"]}"'
        path = self.file_path(item)
        if path and path.exists():
            return web.FileResponse(path, headers=headers)
        # output folder unknown: stream through ComfyUI
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        async with self.session.get(self.comfy + "/view", params=params) as r:
            if r.status != 200:
                raise web.HTTPNotFound(text="Файл не найден")
            resp = web.StreamResponse(headers={"Content-Type": r.headers.get("Content-Type", "video/mp4"), **headers})
            await resp.prepare(req)
            async for chunk in r.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            return resp

    async def h_frame_save(self, req):
        """A picture taken from a video by the browser (one frame, PNG): into the Library and the program's results folder."""
        png, form = None, {}
        async for field in await req.multipart():
            if field.name == "frame":
                png = await field.read(decode=False)
            else:
                form[field.name] = (await field.text())[:200]
        if not png or png[:8] != b"\x89PNG\r\n\x1a\n":
            return web.json_response({"error": "Кадр не получен. Остановите видео и попробуйте ещё раз"}, status=400)
        iid = uuid.uuid4().hex[:12]
        now = datetime.now()
        src = self.library.get(form.get("src", "")) or {}
        part = {"first": "первый", "last": "последний"}.get(form.get("which", ""), "кадр")
        try:
            d = self.results_dir()
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{now.strftime('%Y%m%d')}_{iid}_frame.png"
            await asyncio.to_thread(dest.write_bytes, png)
        except OSError as e:
            return web.json_response({"error": f"Не удалось сохранить кадр: {e}"}, status=500)
        sub = form.get("sub", "").strip()[:80] or f"Кадр из видео ({part})"   # Reference Lab names its own frames and sheets
        item = {"id": iid, "kind": "image", "mode": "frame", "sub": sub, "prompt": src.get("prompt", ""),
                "date": now.isoformat(timespec="seconds"), "fav": False, "ext": "png",
                "res": form.get("res", ""), "file": {"type": "local", "path": str(dest), "filename": dest.name}}
        self.library.add(item)
        return web.json_response(self.public(item))

    async def h_srcfile(self, req):
        """The file an enlarged result was made from (the 'before' of the before/after slider)."""
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        src = self.library.get(item.get("src") or "")
        if src and (p := self.file_path(src)) and p.exists():
            return web.FileResponse(p)
        f = item.get("src_file")
        if not f:
            raise web.HTTPNotFound()
        # The «before» video must be seekable (the before/after slider keeps it on the same moment as the result): a plain body from
        # ComfyUI has no Range support, so the player could only play the first piece. Keep a copy on disk and serve it as a file.
        cache = DATA_DIR / "srccache"
        key = hashlib.md5(json.dumps(f, sort_keys=True).encode()).hexdigest() + Path(str(f.get("filename", ""))).suffix
        p = cache / key
        if not p.exists():
            async with self.session.get(self.comfy + "/view", params=f) as r:
                if r.status != 200:
                    raise web.HTTPNotFound(text="Исходный файл не найден")
                body = await r.read()
            try:
                cache.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(p.suffix + ".part")
                await asyncio.to_thread(tmp.write_bytes, body)
                tmp.replace(p)
                now = datetime.now().timestamp()
                for old in cache.iterdir():   # a week is enough for looking at a comparison
                    if old != p and now - old.stat().st_mtime > 7 * 86400:
                        old.unlink(missing_ok=True)
            except OSError:
                return web.Response(body=body, content_type="application/octet-stream")
        return web.FileResponse(p)

    async def h_events(self, req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        self.browsers.add(ws)
        await ws.send_str(json.dumps({"type": "engine", "online": self.comfy_online}))
        try:
            async for _ in ws:
                pass
        finally:
            self.browsers.discard(ws)
        return ws

    # ---------- agent ----------

    def skill_public(self, s):
        return {k: s[k] for k in ("id", "name", "description", "builtin", "mode", "keywords")} | \
               {"refs": [r.name for r in s["refs"]]}

    async def h_agent_status(self, _req):
        variant = await self.llm_variant()
        v = ag.LLM_VARIANTS[variant]
        custom = self.config.get("llm_custom")
        vram = (self.hw or {}).get("vram_gb") or 0
        candidates = []
        if custom and custom.get("dir") and Path(custom["dir"]).is_dir():
            _, _, found = await asyncio.to_thread(ag.find_gguf, custom["dir"])
            candidates = [{"path": str(p), "name": p.name, "gb": round(p.stat().st_size / 1e9, 1),
                           "fits": ag.fits(p, vram), "current": str(p) == custom["model"]} for p in found]
        options = []
        for k, ov in ag.LLM_VARIANTS.items():
            mp, pp = ag.variant_files(self.llm.root, ov)
            options.append({"key": k, "title": ov["title"], "uncensored": bool(ov.get("uncensored")),
                            "size": ov["model_size"] + ov["mmproj_size"], "have": mp.exists() and pp.exists(),
                            "current": k == variant and not custom, "fits": vram >= ov.get("min_vram", 0),
                            "min_vram": ov.get("min_vram", 0)})
        return web.json_response({
            "variants": options,
            "my_model": ({"title": Path((custom or self.config.get("llm_custom_saved") or {}).get("model", "")).stem,
                          "current": bool(custom)} if (custom or (self.config.get("llm_custom_saved") or {}).get("model")) else None),
            "variant": variant, "title": Path(custom["model"]).stem if custom else v["title"],
            "size": 37e6 if custom else v["model_size"] + v["mmproj_size"], "candidates": candidates, "vram": vram,
            "custom": custom, "vision": self.llm.vision(variant), "has_model": self.llm.has_model(variant),
            "speech_ready": self.speech_status()["ready"], "speech_size": speech.MODEL_SIZE + speech.WHISPER_ZIP_SIZE,
            "speech": self.speech_status(),
            "installed": self.llm.installed(variant), "running": self.llm.alive(),
            "engine": self.llm.backend(),
            "placement": self.llm.placement() if self.llm.alive() else None,
            "installing": bool(self.agent_installer), "dir": str(self.llm.root), "managed": self.config.get("engine") == "managed",
            "dir_chosen": bool(self.config.get("llm_dir")) or self.config.get("engine") == "managed",
            "skills": [self.skill_public(s) for s in self.skills.list()],
        })

    async def h_agent_install(self, req):
        if self.agent_installer:
            return web.json_response({"error": "Уже скачивается"}, status=409)
        variant = await self.llm_variant()
        try:
            wanted = ((await req.json()).get("dir") or "").strip()
        except ValueError:
            wanted = ""
        if self.config.get("engine") != "managed" and not self.llm.installed(variant):
            # with the user's own ComfyUI nothing goes anywhere the user did not name
            if not wanted and not self.config.get("llm_dir"):
                return web.json_response({"error": "Сначала укажите папку, куда скачать помощника"}, status=400)
            if wanted:
                target = Path(wanted) if Path(wanted).name == "MinimaxH3_assistant" else Path(wanted) / "MinimaxH3_assistant"
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError:
                    return web.json_response({"error": "Не получается создать папку. Проверьте путь и диск"}, status=400)
                self.config.update(llm_dir=str(target))
                await self.llm_variant()

        async def emit(ev):
            await self.broadcast(dict(ev, type="agent_setup"))

        self.agent_installer = engine.Installer(ROOT, self.session, emit)

        async def run():
            try:
                await ag.install(self.agent_installer, self.llm.root, variant,
                                 runtime_only=bool(self.config.get("llm_custom")),
                                 whisper_dir=self.config.get("whisper_dir"), hw=self.hw)
                await emit({"step": "agent", "state": "done", "label": "Модель агента установлена"})
            except engine.Cancelled:
                await emit({"step": "agent", "state": "cancelled", "label": "Скачивание остановлено"})
            except Exception as e:
                await emit({"step": "agent", "state": "error", "label": str(e)})
            finally:
                self.agent_installer = None

        asyncio.create_task(run())
        return web.json_response({"ok": True})

    def speech_status(self):
        s = speech.locate(self.llm.root, self.config.get("whisper_dir"))
        return {"ready": s["ready"], "missing": s["missing"], "custom": s["custom"],
                "exe": str(s["exe"]) if s["exe"] else None, "model": s["model"].name if s["model"] else None}

    async def h_speech_folder(self, req):
        """Use whisper files the user already has (or go back to the app's own copy)."""
        body = await req.json()
        await self.llm_variant()
        if body.get("reset"):
            self.config.update(whisper_dir=None)
        else:
            folder = (body.get("dir") or "").strip().strip('"')
            if not Path(folder).is_dir():
                return web.json_response({"error": f"Папка не найдена: {folder}"}, status=400)
            found = await asyncio.to_thread(speech.locate, self.llm.root, folder)
            if not (speech._find_exe(folder) or speech._find_model(folder)):
                return web.json_response({"error": "В папке нет ни whisper-cli.exe, ни модели ggml-*.bin "
                                                   "(нужна многоязычная модель, не «.en»)."}, status=400)
            self.config.update(whisper_dir=folder)
        return await self.h_agent_status(req)

    async def h_transcribe(self, req):
        """Words from an audio clip (WAV from the browser) - lyrics for lipsync prompts and voice input."""
        await self.llm_variant()  # keeps llm.root in sync with the install folder
        if not self.speech_status()["ready"]:
            return web.json_response({"error": "Распознавание речи ещё не скачано (карточка «Модель» в разделе «Агент»)."},
                                     status=400)
        wav, lang = None, "auto"
        async for field in await req.multipart():
            if field.name == "file":
                wav = await field.read()
            elif field.name == "lang":
                lang = (await field.text()).strip() or "auto"
        if not wav:
            return web.json_response({"error": "Нет аудио"}, status=400)
        try:
            text, code, name = await speech.transcribe(self.llm.root, wav, lang, self.config.get("whisper_dir"))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"text": text, "lang": code, "lang_name": name})

    async def h_agent_variant(self, req):
        """Choose which of the app's own models the agent uses (Gemma / Qwen, with or without refusals)."""
        key = (await req.json()).get("variant")
        if key not in ag.LLM_VARIANTS:
            return web.json_response({"error": "Неизвестная модель"}, status=400)
        cur = self.config.get("llm_custom")
        # the user's own model is not lost: it stays as «Моя модель» and can be switched back on
        self.config.update(llm_variant=key, llm_custom=None, **({"llm_custom_saved": cur} if cur else {}))
        await self.stop_llm()
        return await self.h_agent_status(req)

    async def h_agent_model(self, req):
        """Use a GGUF model the user already has (or go back to the app's own Gemma)."""
        body = await req.json()
        if body.get("reset"):
            cur = self.config.get("llm_custom")
            self.config.update(llm_custom=None, **({"llm_custom_saved": cur} if cur else {}))
        elif body.get("restore"):
            saved = self.config.get("llm_custom_saved")
            if not saved or not Path(saved.get("model", "")).exists():
                return web.json_response({"error": "Файл вашей модели не найден"}, status=400)
            self.config.update(llm_custom=saved)
        else:
            folder = (body.get("dir") or "").strip().strip('"')
            if not Path(folder).is_dir():
                return web.json_response({"error": f"Папка не найдена: {folder}"}, status=400)
            if self.hw is None:
                self.hw = await asyncio.to_thread(engine.detect_hardware)
            model, proj, models = await asyncio.to_thread(ag.find_gguf, folder, body.get("model"), self.hw.get("vram_gb"))
            if not model:
                return web.json_response({"error": "В папке нет моделей .gguf. Нужна обычная Gemma в формате GGUF "
                                                   "(версия для LTX в text_encoders не подойдёт)."}, status=400)
            self.config.update(llm_custom={"model": str(model), "mmproj": str(proj) if proj else None, "dir": folder})
        await self.stop_llm()
        return await self.h_agent_status(req)

    async def h_agent_install_cancel(self, _req):
        if self.agent_installer:
            self.agent_installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_agent_chat(self, req):
        body = await req.json()
        messages = body.get("messages") or []
        target = body.get("target") if body.get("target") in ag.AGENT_TARGETS else "video"
        ctx = body.get("ctx") or {}
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(req)

        async def send(obj):
            await resp.write(f"data: {ag.dumps(obj)}\n\n".encode())

        try:
            if self.jobs:   # the agent's model would compete with the video for video memory: wait for the end of the generation, then start by itself
                waited = 0
                while self.jobs:
                    if req.transport is None or req.transport.is_closing():   # the user stopped the request or left
                        return resp
                    await send({"type": "status", "text": f"Идёт генерация видео — агент запустится сам, когда она закончится… (ждём {waited // 60}:{waited % 60:02d})"})
                    await asyncio.sleep(3)
                    waited += 3
            variant = await self.llm_variant()
            if not self.llm.installed(variant):
                raise RuntimeError("Модель агента ещё не скачана.")
            if not self.llm.alive():
                await send({"type": "status", "text": "Освобождаем видеопамять…"})
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                await send({"type": "status", "text": f'Загружаем {ag.LLM_VARIANTS[variant]["title"]}… (до минуты)'})
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)

            if not self.llm.vision(variant) and any(m.get("images") for m in messages):
                # without a vision projector the model cannot take pictures
                messages = [dict(m, images=None) for m in messages]
                await send({"type": "status", "text": "У этой модели нет файла «зрения» (mmproj) — изображения она не увидит"})
            # frames cut out of a video are the last pictures of a message and are not hero pictures (Picture N)
            ctx["images"] = sum(max(0, len(m.get("images") or []) - int(m.get("frames") or 0)) for m in messages if m["role"] == "user")
            active = self.skills.active(ag.chat_text(messages), target, ctx.get("model"))
            if target == "actor" and not any(s["id"] == "akterskaya-igra" for s in active):
                # this mode is acting-first: its skill is always on here
                active += [s for s in self.skills.list() if s["id"] == "akterskaya-igra" and s["mode"] != "off"]
            if target == "chat" and body.get("skills"):   # the skills picked by hand for this chat (chips), whatever their Auto/Off mode
                want = [str(x) for x in body["skills"]]
                active = [s for s in self.skills.list() if s["id"] in want]
            await send({"type": "skills", "names": [s["name"] for s in active]})
            system = ag.system_prompt("\n\n".join(self.skills.text(s) for s in active), target, ctx)
            if target == "chat" and body.get("project"):   # a project is open: its skills, summary and documents go to the model
                q = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "user"), "")
                pctx = await asyncio.to_thread(self.projects.context, str(body["project"]), q)
                if pctx:
                    system += "\n\n" + pctx
            if target == "chat" and body.get("web"):   # «search the internet» ticked: the last question goes to the search, the answers are given to the model
                q = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "user"), "")
                await send({"type": "status", "text": "Ищу в интернете…"})
                found, err = [], ""
                try:
                    found = await websearch.search(self.session, await self.web_query(q))
                except Exception as e:
                    err = f" ({e})"
                if found:
                    system += ("\n\nРезультаты поиска в интернете по последнему вопросу пользователя (найдены только что). Отвечай по ним: "
                               "бери факты оттуда, не выдумывай то, чего в них нет. Если в результатах нет ответа, они противоречат друг другу или заметно устарели "
                               "(смотри даты), так и скажи. Называй сайт, откуда взят факт. Если пользователь не спрашивал про свежие данные, поиск можно не использовать.\n"
                               + websearch.as_text(found))
                else:
                    system += ("\n\nПоиск в интернете включён, но ничего не нашёл" + err + ". Скажи об этом пользователю и "
                               "не выдумывай свежие данные: у тебя нет интернета.")
            await send({"type": "status", "text": ("Скиллы: " + ", ".join(s["name"] for s in active) + ". Думаю…") if active else "Думаю…"})   # which skills the Agent works by is visible in every status line

            thinking = False
            payload = {"messages": [{"role": "system", "content": system}] + ag.to_openai(messages),
                       "stream": True, "temperature": 0.6, "cache_prompt": True}
            if body.get("think") != "deep":
                # "fast" (default): no hidden reasoning before the answer. On the user's machine a light request took 17 s with it
                # (about 2000 characters of thoughts) and 1.5 s without; the long-video rules took minutes with it
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["thinking_budget_tokens"] = 0
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as r:
                if r.status != 200:
                    raise RuntimeError(f"Модель ответила ошибкой {r.status}: {(await r.text())[:300]}")
                async for raw in r.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        await send({"type": "delta", "text": delta["content"]})
                    elif delta.get("reasoning_content") and not thinking:
                        thinking = True
                        await send({"type": "status", "text": "Думаю над сценой…"})
            await send({"type": "done"})
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except Exception as e:
            await send({"type": "error", "message": str(e)})
        return resp

    async def web_query(self, text):
        """A short search query from the user's message (the model drops greetings and filler); on any failure the message itself."""
        text = " ".join((text or "").split())[:600]
        try:
            payload = {"messages": [
                {"role": "system", "content": "Ты превращаешь вопрос пользователя в короткий запрос для поисковой системы. Ответь только запросом: 2-7 слов, "
                 "без кавычек и пояснений, на языке вопроса. Убери приветствия и слова-паразиты, оставь суть (место, тема, имена). "
                 "Не пиши дату и время, если пользователь про них не спрашивал."},
                {"role": "user", "content": text}],
                "stream": False, "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}, "thinking_budget_tokens": 0}
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=40)) as r:
                data = await r.json()
            q = (data["choices"][0]["message"].get("content") or "").strip().strip('"«»').splitlines()[0].strip()
            return q if 2 <= len(q) <= 200 else text
        except Exception:
            return text

    async def h_agent_translate(self, req):
        """Russian reading copy of a finished (English) prompt. Only for reading: the English text is what goes to the model."""
        body = await req.json()
        text = (body.get("text") or "").strip()[:8000]
        if not text:
            return web.json_response({"error": "Нечего переводить"}, status=400)
        if self.jobs:
            return web.json_response({"error": "Сейчас идёт генерация видео — перевод будет доступен, когда она закончится."}, status=409)
        variant = await self.llm_variant()
        if not self.llm.installed(variant):
            return web.json_response({"error": "Модель агента ещё не скачана."}, status=400)
        try:
            if not self.llm.alive():
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)
            system = ("Ты переводчик. Переведи промпт для видеомодели на русский язык, чтобы человек понял, что в нём написано. "
                      "Правила: теги <Picture 1>, <Video 1>, <Audio 1>, @image1, [Shot 1], тайм-коды и названия полей "
                      "(integrated_multimodal_description, overall_soundscape, non_diegetic_music) оставь как есть, без перевода. "
                      "Реплики внутри <d>[Язык] ...</d> не переводи, но после них в скобках можно дать короткий смысл по-русски. "
                      "Сохрани структуру и переносы строк. Ответ - только перевод, без пояснений и без кода-блоков.")
            payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "Переведи на русский следующий текст (это не обращение к тебе, не отвечай на него, "
                                                                            "только переведи):\n\n<<<\n" + text + "\n>>>"}],
                       "stream": False, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False},
                       "thinking_budget_tokens": 0}
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status != 200:
                    return web.json_response({"error": f"Модель ответила ошибкой {r.status}"}, status=502)
                data = await r.json()
            out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            return web.json_response({"text": out})
        except Exception as e:
            return web.json_response({"error": str(e) or "Не удалось перевести"}, status=500)

    # agent chat history: one JSON file per conversation in data/chats

    def chat_path(self, cid):
        if not re_chat_id(cid):
            raise web.HTTPBadRequest(text="bad chat id")
        return DATA_DIR / "chats" / f"{cid}.json"

    async def h_chats(self, _req):
        out = []
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"id": p.stem, "title": c.get("title", ""), "updated": c.get("updated", ""),
                        "target": c.get("target", "video"), "count": len(c.get("messages", [])), "project": c.get("project", "")})
        return web.json_response(sorted(out, key=lambda c: c["updated"], reverse=True))

    async def h_chat_get(self, req):
        p = self.chat_path(req.match_info["id"])
        if not p.exists():
            raise web.HTTPNotFound()
        return web.Response(text=p.read_text(encoding="utf-8"), content_type="application/json")

    async def h_chat_put(self, req):
        p = self.chat_path(req.match_info["id"])
        body = await req.json()
        msgs = [{k: m.get(k) for k in ("role", "content", "images", "skills", "error", "nframes") if m.get(k)}
                for m in body.get("messages") or [] if m.get("role") in ("user", "assistant")]
        first = next((m.get("content", "") for m in msgs if m["role"] == "user"), "") or "Без названия"
        data = {"title": (body.get("title") or first).strip().replace("\n", " ")[:80], "target": body.get("target", "video"),
                "updated": datetime.now().isoformat(timespec="seconds"), "messages": msgs,
                "chat_skills": [str(x) for x in (body.get("chat_skills") or [])][:20],
                "project": re.sub(r"[^a-z0-9]", "", str(body.get("project") or ""))[:40]}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return web.json_response({"ok": True, "title": data["title"]})

    async def h_chat_delete(self, req):
        self.chat_path(req.match_info["id"]).unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_chats_clear(self, _req):
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            p.unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_skill_upload(self, req):
        files = []
        async for field in await req.multipart():
            if field.filename:
                files.append((field.filename.replace("\\", "/"), await field.read()))
        try:
            ids = await asyncio.to_thread(self.skills.save_upload, files)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "id": ids[0], "ids": ids})

    async def h_skill_patch(self, req):
        body = await req.json()
        self.skills.save_settings(req.match_info["id"], body.get("mode"), body.get("keywords"))
        return web.json_response({"ok": True})

    async def h_skill_delete(self, req):
        self.skills.delete(req.match_info["id"])
        return web.json_response({"ok": True})

    # projects of the LLM section

    def _proj(self, req):
        pid = req.match_info["id"]
        try:
            self.projects._meta(pid)
        except (KeyError, ValueError):
            raise web.HTTPNotFound(text="Проект не найден")
        return pid

    @staticmethod
    async def _files(req):
        files = []
        async for field in await req.multipart():
            if field.filename:
                files.append((field.filename.replace("\\", "/"), await field.read()))
        return files

    async def h_projects(self, _req):
        return web.json_response(self.projects.list())

    async def h_project_create(self, req):
        body = await req.json()
        return web.json_response({"ok": True, "id": self.projects.create(body.get("name"))})

    async def h_project_get(self, req):
        return web.json_response(self.projects.detail(self._proj(req)))

    async def h_project_patch(self, req):
        pid, body = self._proj(req), await req.json()
        if body.get("name") is not None:
            self.projects.rename(pid, body["name"])
        if body.get("summary") is not None:
            self.projects.set_summary(pid, body["summary"])
        return web.json_response({"ok": True})

    async def h_project_delete(self, req):
        pid = self._proj(req)
        self._project_chats(pid, delete=True)
        self.projects.delete(pid)
        return web.json_response({"ok": True})

    async def h_project_clear(self, req):
        pid = self._proj(req)
        self._project_chats(pid, delete=True)
        self.projects.clear(pid)
        return web.json_response({"ok": True})

    def _project_chats(self, pid, delete=False):
        n = 0
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("project") == pid:
                    n += 1
                    if delete:
                        p.unlink(missing_ok=True)
            except ValueError:
                continue
        return n

    async def h_project_docs(self, req):
        pid, files = self._proj(req), await self._files(req)
        try:
            n = await asyncio.to_thread(self.projects.add_docs, pid, files)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "added": n})

    async def h_project_doc_delete(self, req):
        self.projects.del_doc(self._proj(req), req.match_info["name"])
        return web.json_response({"ok": True})

    async def h_project_skills(self, req):
        pid, files = self._proj(req), await self._files(req)
        try:
            n = await asyncio.to_thread(self.projects.add_skills, pid, files)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "added": n})

    async def h_project_skill_delete(self, req):
        self.projects.del_skill(self._proj(req), req.match_info["name"])
        return web.json_response({"ok": True})

    async def h_project_open(self, req):
        d = self.projects.path(self._proj(req))
        try:
            subprocess.Popen(["explorer", str(d)])
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"ok": True, "path": str(d)})

    async def h_project_summary(self, req):
        """«Update summary»: the model rewrites the project's summary from the old one and the current chat."""
        pid, body = self._proj(req), await req.json()
        if self.jobs:
            return web.json_response({"error": "Сейчас идёт генерация видео — сводку можно обновить, когда она закончится."}, status=409)
        variant = await self.llm_variant()
        if not self.llm.installed(variant):
            return web.json_response({"error": "Модель агента ещё не скачана."}, status=400)
        chat = "\n".join(f"{'Пользователь' if m.get('role') == 'user' else 'Помощник'}: {(m.get('content') or '')[:3000]}"
                         for m in (body.get("messages") or [])[-16:] if m.get("role") in ("user", "assistant") and m.get("content"))
        old = self.projects.detail(pid)["summary"]
        if not chat.strip() and not old.strip():
            return web.json_response({"error": "Сводку не из чего собрать: напишите что-нибудь в чате проекта."}, status=400)
        try:
            if not self.llm.alive():
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)
            system = ("Ты ведёшь краткую сводку проекта (его «память»). По старой сводке и последней переписке напиши новую сводку целиком: "
                      "герои и их особенности, стиль и настроение, принятые решения, о чём договорились, что осталось сделать. "
                      "Сохрани важное из старой сводки, добавь новое, убери устаревшее. Короткие пункты списком, не больше 1500 слов. "
                      "Язык: русский. Ответ - только текст сводки, без пояснений.")
            payload = {"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": f"СТАРАЯ СВОДКА:\n{old or '(пусто)'}\n\nПОСЛЕДНЯЯ ПЕРЕПИСКА:\n{chat or '(пусто)'}"}],
                       "stream": False, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}, "thinking_budget_tokens": 0}
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status != 200:
                    return web.json_response({"error": f"Модель ответила ошибкой {r.status}"}, status=502)
                data = await r.json()
            out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            if not out:
                return web.json_response({"error": "Модель вернула пустой ответ"}, status=502)
            self.projects.set_summary(pid, out)
            return web.json_response({"ok": True, "summary": out})
        except Exception as e:
            return web.json_response({"error": str(e) or "Не удалось обновить сводку"}, status=500)

    async def h_index(self, _req):
        return web.FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


def lora_strength(form, key, default):
    """Strength sent by the UI: a number when the LoRA is on, "off" when off, nothing -> the default (None = off)."""
    v = form.get(f"lora_{key}")
    if v in (None, ""):
        return default
    if v == "off":
        return None
    try:
        return float(v)
    except ValueError:
        return default


def re_chat_id(cid):
    return cid.isalnum() and 4 <= len(cid) <= 40


def gpu_name(raw):
    # "cuda:0 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync" -> "NVIDIA GeForce RTX 4060 Ti"
    name = raw.split(" : ")[0]
    return name.split(" ", 1)[1] if name.startswith("cuda:") and " " in name else name


def comfy_error_text(res):
    if not isinstance(res, dict):
        return "ComfyUI отклонил задание"
    err = res.get("error") or {}
    parts = [err.get("message", "") + (": " + err.get("details", "") if err.get("details") else "")]
    for node_id, ne in (res.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            parts.append(f'{ne.get("class_type", node_id)}: {e.get("message")} {e.get("details", "")}'.strip())
    return "\n".join(p for p in parts if p) or "ComfyUI отклонил задание"


def send_to_recycle_bin(paths):
    """Move files to the Recycle Bin with ONE hidden PowerShell (before: one visible blue window per file). Returns the paths that are gone."""
    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    if not paths:
        return []
    ps = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
          "foreach ($p in ($env:H3_DEL -split '\\|')) { if ($p) { try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p, 'OnlyErrorDialogs', 'SendToRecycleBin') } catch {} } }")   # '|' cannot be in a Windows path
    subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps], capture_output=True,
                   env={**os.environ, "H3_DEL": "|".join(str(p) for p in paths)}, creationflags=engine.CREATE_NO_WINDOW)
    return [p for p in paths if not p.exists()]


def build_app(comfy_override=None):
    state = App(comfy_override)
    app = web.Application(client_max_size=300 * 1024 * 1024)
    app.on_startup.append(state.start)
    app.on_cleanup.append(state.stop)
    app.add_routes([
        web.get("/", state.h_index),
        web.get("/api/status", state.h_status),
        web.get("/api/system", state.h_system),
        web.get("/api/audio/info", state.h_audio_info),
        web.post("/api/audio/separate", state.h_audio_separate),
        web.get("/api/reflab/presets", state.reflab.h_presets),
        web.post("/api/reflab/prompt", state.reflab.h_prompt),
        web.post("/api/reflab/presets", state.reflab.h_preset_save),
        web.delete("/api/reflab/presets/{id}", state.reflab.h_preset_delete),
        web.get("/api/reflab/runs", state.reflab.h_list),
        web.post("/api/reflab/runs", state.reflab.h_save),
        web.post("/api/reflab/runs/clear", state.reflab.h_clear),
        web.post("/api/reflab/runs/{id}/rate", state.reflab.h_rate),
        web.delete("/api/reflab/runs/{id}", state.reflab.h_delete),
        web.get("/api/reflab/file/{id}/{name}", state.reflab.h_file),
        web.post("/api/reflab/folder", state.h_reflab_folder),
        web.post("/api/reflab/describe", state.h_reflab_describe),
        web.get("/api/voice/status", state.h_voice_status),
        web.get("/api/voice/progress", state.h_voice_progress),
        web.post("/api/voice/install", state.h_voice_install),
        web.post("/api/voice/generate", state.h_voice_generate),
        web.post("/api/voice/cancel", state.h_voice_cancel),
        web.get("/api/voice/out/{name}", state.voices.h_out),
        web.get("/api/voice/takes", state.voices.h_takes),
        web.post("/api/voice/takes/clear", state.voices.h_takes_clear),
        web.post("/api/voice/takes/{id}/star", state.voices.h_take_star),
        web.post("/api/voice/takes/{id}/rename", state.voices.h_take_rename),
        web.delete("/api/voice/takes/{id}", state.voices.h_take_delete),
        web.get("/api/voice/saved", state.voices.h_list),
        web.post("/api/voice/saved", state.voices.h_save),
        web.get("/api/voice/saved/{id}/file", state.voices.h_file),
        web.delete("/api/voice/saved/{id}", state.voices.h_delete),
        web.get("/api/setup/info", state.h_setup_info),
        web.post("/api/setup/external", state.h_setup_external),
        web.post("/api/setup/install", state.h_setup_install),
        web.post("/api/mode/add", state.h_mode_add),
        web.post("/api/settings/mirror", state.h_mirror),
        web.post("/api/settings/live-preview", state.h_live_preview),
        web.post("/api/mode/offered", state.h_mode_offered),
        web.post("/api/setup/cancel", state.h_setup_cancel),
        web.post("/api/comfy/update", state.h_comfy_update),
        web.get("/api/settings/comfy-flags", state.h_comfy_flags),
        web.post("/api/settings/comfy-flags", state.h_comfy_flags),
        web.post("/api/open-driver-page", state.h_open_driver),
        web.post("/api/dmad/open", state.h_open_dmad),
        web.post("/api/results/open", state.h_results_open),
        web.post("/api/dmad/folder", state.h_dmad_folder),
        web.post("/api/dmad/install", state.h_dmad_install),
        web.post("/api/clipboard", state.h_clipboard),
        web.post("/api/support", state.h_support),
        web.post("/api/rent/open", state.h_rent_open),
        web.get("/api/rent/info", state.h_rent_info),
        web.get("/api/queue/saved", state.h_queue_saved),
        web.post("/api/queue/resume", state.h_queue_resume),
        web.post("/api/queue/discard", state.h_queue_discard),
        web.post("/api/search-web", state.h_search_web),
        web.post("/api/report", state.h_report),
        web.post("/api/setup/reset", state.h_setup_reset),
        web.post("/api/generate/video", state.h_generate_video),
        web.post("/api/generate/krea", state.h_generate_krea),
        web.get("/api/dlss_check", state.h_dlss_check),
        web.post("/api/dlss_open", state.h_dlss_open),
        web.post("/api/upscale", state.h_upscale),
        web.post("/api/upscale_file", state.h_upscale_file),
        web.post("/api/grain", state.h_grain),
        web.get("/api/version", lambda _req: web.json_response({"version": APP_VERSION,
                                                                 "built": build_stamp()})),
        web.post("/api/nodes/install", state.h_nodes_install),
        web.post("/api/nodes/fix", state.h_nodes_fix),
        web.get("/api/update/check", state.h_update_check),
        web.get("/api/update/progress", state.h_update_progress),
        web.post("/api/update/install", state.h_update_install),
        web.post("/api/generate/qwen", state.h_generate_qwen),
        web.post("/api/generate/inpaint", state.h_generate_inpaint),
        web.post("/api/generate/lipsync", state.h_generate_lipsync),
        web.get("/api/nodes", state.h_nodes),
        web.get("/api/models/files", state.h_models_files),
        web.post("/api/models/pick", state.h_models_pick),
        web.get("/api/models/compat", state.h_models_compat),
        web.post("/api/loras/custom/mode", state.h_loras_custom_mode),
        web.post("/api/models/open", state.h_models_open),
        web.post("/api/models/download", state.h_models_download),
        web.post("/api/models/download/cancel", state.h_models_download_cancel),
        web.post("/api/models/variant", state.h_models_variant),
        web.get("/api/wf/list", state.h_wf_list),
        web.post("/api/wf/add", state.h_wf_add),
        web.get("/api/wf/{id}", state.h_wf_get),
        web.delete("/api/wf/{id}", state.h_wf_delete),
        web.post("/api/generate/custom", state.h_generate_custom),
        web.get("/api/loras/custom", state.h_loras_custom),
        web.post("/api/loras/custom", state.h_loras_custom_set),
        web.post("/api/generate/refs", state.h_generate_refs),
        web.post("/api/generate/actor", state.h_generate_actor),
        web.post("/api/generate/swap", state.h_generate_swap),
        web.post("/api/generate/long", state.h_generate_long),
        web.post("/api/montage", state.h_montage),
        web.get("/api/long/state", state.h_long_state),
        web.get("/api/long/project", state.h_long_project),
        web.post("/api/long/project", state.h_long_project),
        web.get("/api/long/ref/{run}/{name}", state.h_long_ref),
        web.post("/api/long/stage", state.h_long_stage),
        web.post("/api/generate/action", state.h_generate_action),
        web.get("/api/action/info", state.h_action_info),
        web.get("/api/cloud", state.h_cloud_get),
        web.post("/api/cloud", state.h_cloud_set),
        web.post("/api/cloud/test", state.h_cloud_test),
        web.post("/api/setup/cloud", state.h_setup_cloud),
        web.post("/api/interrupt", state.h_interrupt),
        web.get("/api/agent/status", state.h_agent_status),
        web.post("/api/agent/translate", state.h_agent_translate),
        web.post("/api/agent/install", state.h_agent_install),
        web.post("/api/agent/install/cancel", state.h_agent_install_cancel),
        web.post("/api/agent/model", state.h_agent_model),
        web.post("/api/agent/variant", state.h_agent_variant),
        web.post("/api/speech/transcribe", state.h_transcribe),
        web.post("/api/speech/folder", state.h_speech_folder),
        web.post("/api/agent/chat", state.h_agent_chat),
        web.get("/api/agent/chats", state.h_chats),
        web.delete("/api/agent/chats", state.h_chats_clear),
        web.get("/api/agent/chats/{id}", state.h_chat_get),
        web.put("/api/agent/chats/{id}", state.h_chat_put),
        web.delete("/api/agent/chats/{id}", state.h_chat_delete),
        web.get("/api/projects", state.h_projects),
        web.post("/api/projects", state.h_project_create),
        web.get("/api/projects/{id}", state.h_project_get),
        web.patch("/api/projects/{id}", state.h_project_patch),
        web.delete("/api/projects/{id}", state.h_project_delete),
        web.post("/api/projects/{id}/clear", state.h_project_clear),
        web.post("/api/projects/{id}/docs", state.h_project_docs),
        web.delete("/api/projects/{id}/docs/{name}", state.h_project_doc_delete),
        web.post("/api/projects/{id}/skills", state.h_project_skills),
        web.delete("/api/projects/{id}/skills/{name}", state.h_project_skill_delete),
        web.post("/api/projects/{id}/open", state.h_project_open),
        web.post("/api/projects/{id}/summary", state.h_project_summary),
        web.post("/api/agent/skills", state.h_skill_upload),
        web.patch("/api/agent/skills/{id}", state.h_skill_patch),
        web.delete("/api/agent/skills/{id}", state.h_skill_delete),
        web.get("/api/results", state.h_results_get),
        web.post("/api/results", state.h_results_set),
        web.get("/api/window-gpu", state.h_window_gpu_get),
        web.post("/api/full-mode", state.h_full_mode_set),
        web.post("/api/window-gpu", state.h_window_gpu_set),
        web.post("/api/results/backfill", state.h_results_backfill),
        web.post("/api/results/clean", state.h_results_clean),
        web.get("/api/library", state.h_library),
        web.patch("/api/library/{id}", state.h_library_patch),
        web.delete("/api/library/{id}", state.h_library_delete),
        web.post("/api/library/delete", state.h_library_delete_many),
        web.post("/api/frame", state.h_frame_save),
        web.get("/api/rhlib", state.h_rhlib_get),
        web.post("/api/rhlib/id", state.h_rhlib_id),
        web.post("/api/rhlib/link", state.h_open_link),
        web.post("/api/rhlib/run", state.h_rhlib_run),
        web.post("/api/generate/rh_refs", state.h_generate_rhrefs),
        web.post("/api/generate/rh_video", state.h_generate_rhvideo),
        web.post("/api/generate/rh_lipsync", state.h_generate_rhlipsync),
        web.post("/api/generate/rh_action", state.h_generate_rhaction),
        web.post("/api/generate/rh_actionsc", state.h_generate_rhactionsc),
        web.get("/api/refmodes", state.refmodes.h_list),
        web.post("/api/refmodes", state.refmodes.h_save),
        web.delete("/api/refmodes/{id}", state.refmodes.h_delete),
        web.get("/api/refmodes/{id}/file/{name}", state.refmodes.h_file),
        web.get("/api/file/{id}", state.h_file),
        web.get("/api/srcfile/{id}", state.h_srcfile),
        web.get("/api/events", state.h_events),
        web.static("/ui", UI_DIR),
    ])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", help="always use this ComfyUI URL (skips the setup wizard)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = ap.parse_args()

    app = build_app(args.comfy)
    if args.open:
        async def open_browser(_app):
            asyncio.get_running_loop().call_later(1, webbrowser.open, f"http://127.0.0.1:{args.port}")
        app.on_startup.append(open_browser)
    web.run_app(app, host="127.0.0.1", port=args.port, print=lambda *_: print(f"Minimax H3 Workspace: http://127.0.0.1:{args.port}"))


if __name__ == "__main__":
    main()
