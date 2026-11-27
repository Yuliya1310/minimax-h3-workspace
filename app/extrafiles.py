"""Additional model files that a user may already have somewhere: look for them in the usual folders before anything is downloaded,
and put a found one where ComfyUI sees it. Nothing here searches whole drives: only model folders, Downloads and the Desktop."""
import os
import re
import shutil
from pathlib import Path

DRAFT_NAME = "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"
ACTION_NAME = "minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors"
FILES = {
    "draft": {"name": DRAFT_NAME, "size": 366647000,
              "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/" + DRAFT_NAME},
    "action": {"name": ACTION_NAME, "size": 3913285640, "url": ""},   # Civitai asks for a login: the user downloads it in the browser
}


def clean_name(name):
    """The name without the « (1)» a browser adds to a repeated download."""
    return re.sub(r" \(\d+\)(?=\.safetensors$)", "", name, flags=re.I)


def good(key, name, size):
    """Is this a complete copy of the wanted file (by its name and size)."""
    n = clean_name(name).lower()
    if not n.endswith(".safetensors"):
        return False
    if n == FILES[key]["name"].lower():
        return size >= FILES[key]["size"] * 0.995
    if key == "draft":
        return bool(re.search(r"dmad[_\-\s]?4[_\-\s]?step", n)) and size > 100_000_000
    import catalog
    return catalog.is_action_dmad(n) and size > 800_000_000


def walk(roots):
    """(path, size, root, kind) of every .safetensors file under the roots: [(folder, kind, depth)]."""
    seen = set()
    for root, kind, depth in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base = len(root.parts)
        for folder, dirs, names in os.walk(root):
            if len(Path(folder).parts) - base >= depth:
                dirs[:] = []
            for fn in names:
                if not fn.lower().endswith(".safetensors"):
                    continue
                p = Path(folder) / fn
                k = os.path.normcase(str(p))
                if k in seen:
                    continue
                seen.add(k)
                try:
                    yield p, p.stat().st_size, root, kind
                except OSError:
                    pass


def count(dirs):
    """How many model files lie in each folder (to recommend the one the user really uses)."""
    return [sum(1 for _ in walk([(d, "models", 6)])) for d in dirs]


def folder_for(chosen, dirs):
    """The folder the file goes into, from what the user picked: a loras folder, the ComfyUI folder or `models`, or any folder as it is."""
    p = Path(chosen)
    if p in dirs or p.name.lower() == "loras":
        return p / "MinimaxH3"
    if any(d in p.parents for d in dirs):
        return p
    for c in (p / "loras", p / "models" / "loras", p / "ComfyUI" / "models" / "loras"):
        if c.is_dir():
            return c / "MinimaxH3"
    return p


def find(key, roots):
    """Complete copies of the file, the biggest first: [{path, size, kind}]."""
    out = [{"path": str(p), "size": s, "kind": kind} for p, s, _r, kind in walk(roots) if good(key, p.name, s)]
    return sorted(out, key=lambda f: -f["size"])


def find_same(roots, name, size):
    """A complete file with this very name and size, else None. Small files are not looked for (many share names like config.json)."""
    if size < 50_000_000:
        return None
    want = Path(name).name.lower()
    for p, s, _r, _k in walk(roots):
        if clean_name(p.name).lower() == want and abs(s - size) <= size * 0.005:
            return p
    return None


def put(src, dest):
    """Brings a found file to `dest` without downloading: moved out of Downloads/Desktop is the caller's `move`; otherwise a hard link
    (takes no disk space) or, on another drive, a copy. The original model folder keeps its file."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        return "there"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "link"
    except OSError:
        pass
    if shutil.disk_usage(dest.parent).free < src.stat().st_size + 1024 ** 3:
        raise RuntimeError(f"На диске с папкой {dest.parent} мало места для копии файла.")
    part = dest.with_name(dest.name + ".copy")
    try:
        shutil.copyfile(src, part)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return "copy"
