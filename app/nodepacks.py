"""Installs the outside node packs of the web-animation page (the same versions the program was tested with) into a ComfyUI:
download of one fixed archive with a checksum, unpacking into custom_nodes, the Python libraries, the model weights.
PyTorch of that ComfyUI is never replaced: its installed versions are given to pip as fixed constraints."""
import asyncio
import hashlib
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import engine

PACKS = {
    "rmbg": {"title": "ComfyUI-RMBG", "cls": "RMBG", "dir": "comfyui-rmbg", "names": ("comfyui-rmbg", "comfyui_rmbg"),
             "url": "https://cdn.comfy.org/ailab/comfyui-rmbg/3.0.0/node.zip", "size": 3361959,
             "sha": "eb09e23f3775802483c899eb7edf225822fec08fec778b1267540bca788262d9",
             "need": ("torch", "torchvision", "transformers", "huggingface_hub", "cv2", "timm", "kornia", "einops", "safetensors"),
             "weights": ()},
    "mat": {"title": "ComfyUI-MatAnyone", "cls": "MatAnyone2", "dir": "matanyone", "names": ("matanyone", "comfyui-matanyone", "comfyui_matanyone"),
            "url": "https://cdn.comfy.org/fuoumarinas/matanyone/2.1.4/node.zip", "size": 1059243,
            "sha": "e455447e4a1f5560c66d8b91ba3194a651d514aad837b9e1e30a98771c193575",
            "need": ("torch", "torchvision", "omegaconf", "cv2", "tqdm"),
            "weights": ({"file": "matanyone2.pth", "sub": "checkpoint", "size": 141429115,
                         "url": "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"},)},
}
EXTRA_LIBS = {"mat": ("opencv-python", "tqdm")}   # imported by the pack but not named in its requirements.txt


def find_pack(nodes_dir, key):
    """The folder of the pack inside custom_nodes (any usual spelling, «-main» copies too), else None."""
    names = PACKS[key]["names"]
    try:
        for d in Path(nodes_dir).iterdir():
            n = re.sub(r"[-_](main|master)$", "", d.name.lower())
            if d.is_dir() and n in names and (d / "__init__.py").is_file():
                return d
    except OSError:
        pass
    return None


def python_near(nodes_dir):
    """The Python of the ComfyUI that owns this custom_nodes folder: the portable build's python_embeded or a venv. None when unsure."""
    if not nodes_dir:
        return None
    comfy = Path(nodes_dir).parent
    for c in (comfy.parent / "python_embeded" / "python.exe", comfy / "python_embeded" / "python.exe",
              comfy / "venv" / "Scripts" / "python.exe", comfy / ".venv" / "Scripts" / "python.exe",
              comfy.parent / "venv" / "Scripts" / "python.exe", comfy.parent / ".venv" / "Scripts" / "python.exe"):
        if c.is_file():
            return c
    return None


async def _run(py, args, env, timeout=1800):
    """(exit code, output) of «python args»."""
    proc = await asyncio.create_subprocess_exec(str(py), "-s", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                creationflags=engine.CREATE_NO_WINDOW, env=env, cwd=tempfile.gettempdir())
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "слишком долго, установка прервана"
    except asyncio.CancelledError:
        proc.kill()
        raise
    return proc.returncode, (out or b"").decode("utf-8", "replace")


def requirement_lines(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lines = (l.split("#", 1)[0].strip() for l in text.splitlines())
    return [l for l in lines if l and not l.startswith("-")]


async def install_libs(key, folder, py, env, say):
    """pip install of the pack's libraries. Returns a note about unimportant libraries that did not install; raises when the pack cannot work."""
    pack = PACKS[key]
    code, out = await _run(py, ["-c", "import importlib.metadata as m\nfor n in ('torch','torchvision','torchaudio'):\n    try: print(n+'=='+m.version(n))\n    except Exception: pass"], env, 120)
    pins = [l.strip() for l in out.splitlines() if l.strip().startswith("torch")]
    if code != 0 or not any(p.startswith("torch==") for p in pins):
        raise RuntimeError(f"В этом Python нет PyTorch: {py}. Похоже, это не Python вашего ComfyUI.")
    reqs = requirement_lines(Path(folder) / "requirements.txt")
    reqs += [x for x in EXTRA_LIBS.get(key, ()) if not (x.startswith("opencv") and any(r.startswith("opencv") for r in reqs))]
    # PyTorch is already there and must stay as it is
    reqs = [r for r in reqs if not re.match(r"(torch|torchvision|torchaudio)\b(?![-_])", r, re.I)]
    cons = Path(tempfile.gettempdir()) / f"h3s_constraints_{uuid.uuid4().hex[:8]}.txt"
    cons.write_text("\n".join(pins) + "\n", encoding="ascii")
    failed, last = [], ""
    try:
        if reqs:
            await say(f"{pack['title']}: ставлю библиотеки (может занять несколько минут)…")
            code, last = await _run(py, ["-m", "pip", "install", "--disable-pip-version-check", "-c", str(cons), *reqs], env)
            if code != 0:   # one broken library must not block the rest: install them one by one
                for i, r in enumerate(reqs):
                    await say(f"{pack['title']}: библиотека {i + 1} из {len(reqs)}: {r}")
                    c, o = await _run(py, ["-m", "pip", "install", "--disable-pip-version-check", "-c", str(cons), r], env)
                    if c != 0:
                        failed.append(r)
                        last = o
    finally:
        cons.unlink(missing_ok=True)
    await say(f"{pack['title']}: проверяю библиотеки…")
    check = "import importlib\nbad=[]\nfor n in %r:\n    try: importlib.import_module(n)\n    except Exception as e: bad.append(n+' ('+str(e)[:80]+')')\nprint('BAD:'+'; '.join(bad))" % (tuple(pack["need"]),)
    code, out = await _run(py, ["-c", check], env, 300)
    bad = next((l[4:].strip() for l in out.splitlines() if l.startswith("BAD:")), None)
    if bad is None or bad:
        why = " ".join(l for l in last.splitlines() if "ERROR" in l)[-250:]
        raise RuntimeError(f"{pack['title']}: не установились нужные библиотеки: {bad or out[-200:]}. {why} Закройте ComfyUI, проверьте интернет и нажмите «Скачать» ещё раз.")
    return f"Не установились дополнительные библиотеки (для этой программы они не нужны): {', '.join(failed)}." if failed else ""


async def install(key, nodes_dir, py, installer, env, progress):
    """The whole pack: files (only when there are none yet), weights, libraries. `progress(label, done=None, total=None)` is async.
    Returns a note for the user, may be empty."""
    pack = PACKS[key]
    nodes_dir = Path(nodes_dir)
    folder = find_pack(nodes_dir, key)

    def emitter(label):
        async def emit(ev):
            await progress(ev.get("label") or label, ev.get("done"), ev.get("total"))
        return emit

    if not folder:
        target = nodes_dir / pack["dir"]
        if target.exists():
            raise RuntimeError(f"Папка уже есть, но в ней нет файлов нода: {target}. Удалите или переименуйте её и нажмите «Скачать» ещё раз.")
        await progress(f"Скачиваю {pack['title']}…")
        work = nodes_dir.parent / f"h3s_download_{uuid.uuid4().hex[:6]}"
        try:
            zpath = work / "node.zip"
            installer.emit = emitter(pack["title"])
            await installer.download(pack["url"], zpath, "td", pack["title"], pack["size"], attempts=8)
            if hashlib.sha256(zpath.read_bytes()).hexdigest() != pack["sha"]:
                raise RuntimeError(f"{pack['title']}: скачанный архив повреждён или подменён. Попробуйте ещё раз.")

            def unpack():
                tmp = work / "files"
                tmp.mkdir()
                with zipfile.ZipFile(zpath) as z:
                    for n in z.namelist():
                        dst = (tmp / n).resolve()
                        if tmp.resolve() not in dst.parents:
                            raise RuntimeError("в архиве странный путь: " + n)
                    z.extractall(tmp)
                if not (tmp / "__init__.py").is_file():
                    raise RuntimeError(f"{pack['title']}: в архиве нет __init__.py")
                os.replace(tmp, target)
            await asyncio.to_thread(unpack)
            folder = target
        finally:
            shutil.rmtree(work, ignore_errors=True)
    for w in pack["weights"]:
        dest = folder / w["sub"] / w["file"]
        if not dest.is_file():
            installer.emit = emitter(w["file"])
            await installer.download(w["url"], dest, "td", w["file"], w["size"])
    if not py:
        raise RuntimeError(f"Файлы лежат в {folder}, но не нашла Python вашего ComfyUI, чтобы поставить библиотеки. "
                           f"Запустите ваш ComfyUI и нажмите «Скачать» ещё раз.")
    return await install_libs(key, folder, py, env, progress)
