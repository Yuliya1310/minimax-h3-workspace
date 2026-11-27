"""Ready-made workflows for RunningHub (the «Облако RunningHub» page on the start screen).

Each card is a complete API-format graph (exported from ComfyUI on RunningHub, so every node and model name exists there).
The page only fills in a few fields (prompt, pictures, videos, audio, length, size, strengths, seed); the rest of the graph is
sent as it is. The reference slots (pictures / videos / audio) are built from the files the user gave: the exported nodes are
used as templates and as many of them are made as there are files (up to the limits of the model).
The graph goes through the same /task/openapi/create as the app's own cloud mode (see cloud.py).
"""

import copy
import json
import random
import re
import sys
from pathlib import Path

DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "rh_workflows"

# the option names of ComfyUI's ResolutionSelector node (docs.comfy.org/built-in-nodes/ResolutionSelector); 16:9 is the one in the exported workflows
ASPECTS = {"16:9": "16:9 (Widescreen)", "9:16": "9:16 (Portrait Widescreen)", "1:1": "1:1 (Square)", "4:3": "4:3 (Standard)",
           "3:4": "3:4 (Portrait Standard)", "21:9": "21:9 (Ultrawide)"}

KINDS = (("images", "img", "ref_images.ref_image_", 100), ("videos", "vid", "ref_videos.ref_video_", 130), ("audios", "aud", "ref_audios.ref_audio_", 160))


def _fix_output_ref2v(g):
    """The exported workflow saves ProRes (.mov), which the program's player cannot show: save an ordinary mp4 instead."""
    n = g.get("264")
    if n and n.get("class_type") == "VHS_VideoCombine":
        n["inputs"].pop("profile", None)
        n["inputs"].update({"format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True})
        n["inputs"]["filename_prefix"] = "H3Workspace/rh_refs"


AUTHOR = {"name": "автор workflow на RunningHub", "url": "https://www.runninghub.ai/user-center/2072160124218466306/userPost?inviteCode=rh-v1580"}

CARDS = {
    "action": {
        "title": "Экшн-видео",
        "text": "Динамичная сцена с вашим героем: погоня, бой, бег. Точная копия проверенного workflow (SelfLift + DMAD + Speed Slider), считается на видеокарте RunningHub.",
        "file": "action_ref2v.json",
        "need": "изображения героев (до 6), по желанию аудио",
        "wf_default": "2120936715551592449",   # the workflow ID shipped with the program (digits from the workflow's address on RunningHub)
        "author": AUTHOR,   # the author of the workflow: shown with a link
        "slots": {"images": {"max": 9, "required": True, "label": "Изображения героев и локации", "node": "51"},
                  "audios": {"max": 3, "required": False, "label": "Аудио (голос, ритм)"}},
        "nodes": {"prompt": "263", "duration": "259", "res": "252", "seed": "397", "speed": "343", "dmad": "53", "refsize": "265"},
        "dur": [3, 25],
        "defaults": {"duration": 15, "aspect": "16:9", "mp": 1.0, "speed": 4.0, "dmad": 0.8},
    },
    "refs": {
        "title": "Референсы",
        "text": "Видео по вашим референсам: до 9 изображений (персонаж, одежда, локация, стартовый кадр), до 3 видео (движение камеры) и до 3 аудио (голос, музыка). Звук делает сама модель. Проверенный workflow Singularity Ref2V.",
        "file": "ref2v.json",
        "need": "изображения, по желанию видео и аудио",
        "wf_default": "2115895182347493377",
        "author": AUTHOR,
        "slots": {"images": {"max": 9, "required": False, "label": "Изображения (@image1, @image2…)", "node": "51", "also": ["49"]},
                  "videos": {"max": 3, "required": False, "label": "Видео (@video1…): движение камеры", "node": "27"},
                  "audios": {"max": 3, "required": False, "label": "Аудио (@audio1…): голос или музыка"}},
        "nodes": {"prompt": "263", "duration": "259", "res": "252", "seed": "256", "refsize": "265", "steps": "332"},
        "dur": [3, 15],
        "defaults": {"duration": 5, "aspect": "16:9", "mp": 0.5},
        "fix": _fix_output_ref2v,
        "accel": {"consumers": [["332", "model"]]},   # the sampler takes the model from the LoRA: the attention speed-up goes in between
    },
}


CARDS["lipsync"] = {
    "title": "Lipsync",
    "text": "Персонаж говорит или поёт под ваше аудио: губы попадают в слова, длина видео равна длине выбранного фрагмента. Два прохода (черновой размер и увеличение). Проверенный workflow Singularity Ref2V с Audio Lock.",
    "file": "lipsync.json",
    "need": "аудио и лицо персонажа (до 9 референсов)",
    "wf_default": "2113947468442767362",
    "author": AUTHOR,
    "slots": {"images": {"max": 9, "required": True, "label": "Референсы (лицо персонажа)", "node": "51",
                         "also": ["49", "331", "43", "19", "23", "199", "200", "201"]}},
    "nodes": {"prompt": "263", "duration": "347", "res": "252", "seed": "256", "refsize": "265", "scale": "283", "audio": ["345", "346"]},
    # the exported graph has a LoRA that was BYPASSED in ComfyUI («Optional extra LoRA»): its file name is not in RunningHub's list and the graph is refused.
    # The node is cut out and its neighbours are joined, as ComfyUI does with a bypassed node.
    "bypass": ["128"],
    "accel": {"consumers": [["223", "model"]]},
    "needs_audio": True,
    "out": "214",   # the finished video is the second pass; the first pass (a small size) is saved too
    "dur": [1, 15],
    "defaults": {"duration": 5, "aspect": "9:16", "mp": 0.8},
}


HYBRID_UNET = "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"

CARDS["actionsc"] = {
    "title": "Экшн-видеостудия",
    "text": "Сцены в один ролик: Singularity или Hybrid, DMAD, Speed Slider, по желанию Pika, до 3 аудио-образцов. Ваш workflow «MiniMax H3 Action Scene», считается на видеокарте RunningHub.",
    "file": "action_scene.json",
    "need": "изображения героев и места, по желанию аудио",
    "wf_default": "2122356550684696577",
    "author": {},
    "direct": True,   # size and length are written straight into the reference node (width / height / length in frames), there is no resolution node
    "slots": {"images": {"max": 9, "required": True, "label": "Изображения героев и локации", "node": "100"},
              "audios": {"max": 3, "required": False, "label": "Аудио (голос, музыка, атмосфера)"}},
    "nodes": {"prompt": "201", "res": "201", "seed": "204", "speed": "61", "dmad": "60", "refsize": "201", "unet": "1", "pika": "62", "sched": "10"},
    "dur": [3, 25],
    "defaults": {"duration": 8, "aspect": "16:9", "mp": 0.9, "speed": 4.0, "dmad": 0.8},
}


# IDs shipped by earlier versions (without the loading nodes for the hero videos): a saved copy of them counts as «not set»
OLD_DEFAULTS = {"studio": "2123839054161072130", "clip": "2123840263789957122"}
CARDS["studio"] = {**CARDS["actionsc"], "wf_default": "2125298075140104194", "title": "Видеостудия",
                   "text": "Видеостудия на RunningHub: сцены подряд, каждая продолжает последний кадр прошлой. Свой номер workflow и общий API-ключ."}
CARDS["clip"] = {**CARDS["actionsc"], "wf_default": "2125299015235854338", "title": "Создание клипа",
                 "text": "Клип под вашу музыку на RunningHub: каждая сцена получает свой кусок песни. Свой номер workflow и общий API-ключ."}


CARDS["video"] = {
    "title": "Image-to-Video: видео по тексту и по картинке",
    "text": "Новинка. Видео по описанию, по одной картинке (первый кадр) или между первым и последним кадром, со звуком от модели. Быстрая модель FastH3, 8 шагов.",
    "file": "fastvideo.json",
    "need": "текст, по желанию первый и последний кадр",
    "wf_default": "2122206426290921474",
    "author": {},
    "slots": {},
    "frames": True,   # the first / last frame pictures are put into the graph only when the user gave them
    "optional_files": True,
    "nodes": {"prompt": "13", "duration": "11", "res": "9", "seed": "14"},
    "dur": [3, 25],
    "defaults": {"duration": 5, "aspect": "16:9", "mp": 0.6},
}


def _prompt_key(node):
    return "text" if "text" in node["inputs"] else "prompt"


def load(card_id):
    card = CARDS[card_id]
    return json.loads((DIR / card["file"]).read_text(encoding="utf-8"))


def default_prompt(card_id):
    g = load(card_id)
    node = g[CARDS[card_id]["nodes"]["prompt"]]
    return node["inputs"][_prompt_key(node)]


def public():
    out = []
    for cid, c in CARDS.items():
        out.append({"id": cid, "title": c["title"], "text": c["text"], "need": c["need"], "defaults": c["defaults"],
                    "prompt": default_prompt(cid),
                    "slots": {k: {"max": v["max"], "required": v["required"], "label": v["label"]} for k, v in c["slots"].items()},
                    "wf_default": c.get("wf_default", ""), "author": c.get("author") or {},
                    "dur": c["dur"], "speed": "speed" in c["nodes"], "dmad": "dmad" in c["nodes"], "refsize": "refsize" in c["nodes"],
                    "aspects": list(ASPECTS)})
    return out


def _count(files, key):
    return sum(1 for k, v in files.items() if re.fullmatch(key + r"\d+", k) and v)


def missing(card_id, files):
    """The text of what the user still has to add, or ''. `files`: {'img0': name, ..., 'vid0': name, 'aud0': name}."""
    c = CARDS[card_id]
    for kind, key, _field, _base in KINDS:
        s = c["slots"].get(kind)
        if s and s["required"] and not _count(files, key) and not (kind == "images" and _count(files, "rvid")):
            return f"Добавьте: {s['label']}"
    if c.get("needs_audio") and not files.get("audio"):
        return "Загрузите аудио"
    if not c.get("optional_files") and not any(files.values()):
        return "Добавьте хотя бы одно изображение, видео или аудио"
    return ""


def build(card_id, p, files):
    """The graph of the card with the user's values put in. `p`: prompt, duration, aspect, mp, speed, dmad, seed (None = random).
    `files`: {'img0': uploaded name, 'vid0': ..., 'aud0': ...}. The reference nodes are made from the files that were given."""
    c = CARDS[card_id]
    n = c["nodes"]
    g = copy.deepcopy(load(card_id))
    d = c["defaults"]
    pn = g[n["prompt"]]
    pn["inputs"][_prompt_key(pn)] = (p.get("prompt") or "").strip() or default_prompt(card_id)
    cond = next((v for v in g.values() if v.get("class_type") == "MiniMaxH3ReferenceToVideo"), None)
    if c.get("frames"):   # first / last frame: a picture node and the same fitting to the size as in the original workflow
        i2v = next(v for v in g.values() if v.get("class_type") == "MiniMaxH3ImageToVideo")
        def frame(tag, name):
            g[tag + "_load"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g[tag + "_scale"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": [n["res"], 0], "height": [n["res"], 1],
                                                                 "crop": "center", "image": [tag + "_load", 0]}}
            return [tag + "_scale", 0]
        if files.get("first"):
            i2v["inputs"]["first_frame"] = frame("first", files["first"])
            if p.get("loop"):
                i2v["inputs"]["last_frame"] = i2v["inputs"]["first_frame"]   # a seamless loop: it ends on the picture it starts with
            elif files.get("last"):
                i2v["inputs"]["last_frame"] = frame("last", files["last"])
    for kind, key, field, base in KINDS:
        s = c["slots"].get(kind)
        if not s or cond is None:
            continue
        template = copy.deepcopy(g.get(s.get("node", ""))) if s.get("node") else None
        for nid in [s.get("node")] + list(s.get("also", [])):   # the exported reference nodes and their links go: new ones are made below
            g.pop(nid, None)
        for k in [k for k in cond["inputs"] if k.startswith(field)]:
            cond["inputs"].pop(k, None)
        names = [files[k] for k in sorted((k for k in files if re.fullmatch(key + r"\d+", k) and files[k]), key=lambda k: int(k[len(key):]))][:s["max"]]
        for i, name in enumerate(names):
            nid = str(base + i)
            if kind == "audios":
                g[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            elif template:
                node = copy.deepcopy(template)
                node["inputs"]["image" if kind == "images" else "video"] = name
                g[nid] = node
            cond["inputs"][f"{field}{i}"] = [nid, 0]
    for kind, key, field, base in KINDS:
        if cond is not None and kind not in c["slots"]:   # the exported graph has slots the card does not offer: they are cut out with their links
            for k in [k for k in cond["inputs"] if k.startswith(field)]:
                link = cond["inputs"].pop(k)
                if isinstance(link, list):
                    g.pop(str(link[0]), None)
    if n.get("audio"):   # the clip of the audio is already cut by the page: the trim node just takes it whole
        load_id, trim_id = n["audio"]
        g[load_id]["inputs"]["audio"] = files["audio"]
        g[trim_id]["inputs"].update({"start_index": 0, "duration": float(p.get("duration") or d["duration"])})
    if n.get("scale") and p.get("scale"):
        g[n["scale"]]["inputs"]["value"] = float(p["scale"])
    if n.get("realism") and p.get("realism") is not None:
        g[n["realism"]]["inputs"]["strength_model"] = float(p["realism"])
    for nid in c.get("bypass", []):
        node = g.pop(nid, None)
        if node:
            src = node["inputs"].get("model")
            for other in g.values():
                for k, v in other.get("inputs", {}).items():
                    if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid:
                        other["inputs"][k] = src
    if c.get("accel") and p.get("accel"):
        # the attention speed-up of the action workflow (it runs on RunningHub): the built-in «comfy kitchen attention» backend and the
        # block-sparse attention (sol-attn), put in front of the sampler / guider. The exported graph has neither.
        first = c["accel"]["consumers"][0]
        src = g[first[0]]["inputs"][first[1]]
        g["acc_backend"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": src, "attention": "comfy kitchen attention"}}
        g["acc_sparse"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["acc_backend", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        for nid, key in c["accel"]["consumers"]:
            g[nid]["inputs"][key] = ["acc_sparse", 0]
    rv = [files[k] for k in sorted((k for k in files if re.fullmatch(r"rvid\d+", k) and files[k]), key=lambda k: int(k[4:]))][:3]
    if rv and cond is not None:   # RefMod: the Reference Mode heroes as <Video N> (after the user's own reference videos, if the card has them)
        have = len([k for k in cond["inputs"] if k.startswith("ref_videos.ref_video_")])
        for j, name in enumerate(rv[:max(0, 3 - have)]):
            nid = str(190 + j)
            g[nid] = {"class_type": "VHS_LoadVideo", "inputs": {"video": name, "force_rate": 24, "force_size": "Disabled", "custom_width": 0, "custom_height": 0,
                                                              "frame_load_cap": 360, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
            cond["inputs"][f"ref_videos.ref_video_{have + j}"] = [nid, 0]
    lo, hi = c["dur"]
    if c.get("direct"):
        g[n["res"]]["inputs"].update(width=int(p["width"]), height=int(p["height"]), length=int(p["frames"]))
        hybrid = p.get("model") == "hybrid"
        if hybrid:
            g[n["unet"]]["inputs"]["unet_name"] = HYBRID_UNET
        pika = bool(p.get("pika")) and not hybrid
        if not pika and n["pika"] in g:   # without Pika the speed LoRA is taken out of the chain (the same as «bypass») and 12 steps are used
            src = g.pop(n["pika"])["inputs"]["model"]
            for other in g.values():
                for k, v in other.get("inputs", {}).items():
                    if isinstance(v, list) and len(v) == 2 and str(v[0]) == n["pika"]:
                        other["inputs"][k] = src
        g[n["sched"]]["inputs"]["steps"] = 8 if pika else 12
    else:
        g[n["duration"]]["inputs"]["value"] = max(float(lo), min(float(hi), float(p.get("duration") or d["duration"])))
        g[n["res"]]["inputs"]["aspect_ratio"] = ASPECTS.get(p.get("aspect"), ASPECTS[d["aspect"]])
        g[n["res"]]["inputs"]["megapixels"] = max(0.3, min(2.1, float(p.get("mp") or d["mp"])))
    if n.get("speed") and p.get("speed") not in (None, ""):
        g[n["speed"]]["inputs"]["strength_model"] = float(p["speed"])
    if n.get("dmad") and p.get("dmad") not in (None, ""):
        g[n["dmad"]]["inputs"]["strength_model"] = float(p["dmad"])
    if n.get("steps") and p.get("steps"):
        g[n["steps"]]["inputs"]["steps"] = max(4, min(8, int(p["steps"])))
    if n.get("refsize") and p.get("ref_size") in ("max", "match"):
        g[n["refsize"]]["inputs"]["ref_image_size"] = p["ref_size"]
    seed = p.get("seed")
    seed = int(seed) if seed not in (None, "") else random.randint(1, 2 ** 48)
    g[n["seed"]]["inputs"]["noise_seed" if "noise_seed" in g[n["seed"]]["inputs"] else "seed"] = seed
    if c.get("fix"):
        c["fix"](g)
    return g, seed
