"""Pictures on RunningHub: the app sends the same graph as to a local ComfyUI. This module takes the user's choice of files from their
saved RunningHub workflow and tells in plain words what is missing there (which node, which file, what to do). No network here."""
import re

# the workflows shipped with the program (digits from their addresses on RunningHub); a user's own number replaces them
DEFAULT_IDS = {"krea": "2126329580089159681", "qwen": "2126348387310546946", "flux": "2126332156258598913", "h3img": "2126331351271325698"}
MODELS = {"krea": "Krea 2", "qwen": "Qwen Image 2.1", "flux": "FLUX.2 klein", "h3img": "MiniMax H3 Image"}
CLIP_TYPES = {"krea2": "krea", "qwen_image": "qwen", "flux2": "flux", "minimax": "h3img"}
FILE_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth")
NODES = {"UNETLoader": "Load Diffusion Model (основная модель)", "CLIPLoader": "Load CLIP (текстовый энкодер)",
         "VAELoader": "Load VAE", "LoraLoaderModelOnly": "LoraLoaderModelOnly (LoRA)", "LoraLoader": "Load LoRA"}
FOLDERS = {"unet_name": "diffusion_models", "clip_name": "text_encoders", "vae_name": "vae", "lora_name": "loras"}


def base(name):
    return str(name).replace("\\", "/").split("/")[-1]


def node_title(cls):
    return f"«{NODES[cls]}»" if cls in NODES else f"«{cls}»"


def model_of(graph):
    """Which picture model a graph is for (by the type of its text encoder loader), '' when unknown."""
    for node in graph.values():
        if node.get("class_type") == "CLIPLoader":
            return CLIP_TYPES.get((node.get("inputs") or {}).get("type"), "")
    return ""


def files_of(graph):
    """[(node id, class, input name, file name)] of every model file a graph loads."""
    return [(nid, node.get("class_type", ""), key, v) for nid, node in graph.items()
            for key, v in (node.get("inputs") or {}).items() if isinstance(v, str) and v.lower().endswith(FILE_EXT)]


def default_names(graph, local, defaults):
    """The names of this computer's ComfyUI (own folders, own picks) are replaced by the standard file names: `local` and `defaults` = {role: name}."""
    swap = {local[r]: defaults[r] for r in defaults if local.get(r)}
    for nid, _cls, key, v in files_of(graph):
        graph[nid]["inputs"][key] = base(swap.get(v, v))
    return graph


def adapt(graph, saved):
    """Takes the file names from the user's saved workflow. The same file in any folder: its path there. Another file chosen in a main
    loader (the only one of its kind there, same picture model): that file. Returns [(node class, ours, theirs)] of such replacements."""
    theirs = files_of(saved)
    by_base = {base(v).lower(): v for _n, _c, _k, v in theirs}
    same_model = model_of(saved) == model_of(graph)
    realism = any(c == "LoraLoaderModelOnly" and "realism" in base(v).lower() for _n, c, _k, v in files_of(graph))
    changed = []
    for nid, cls, key, v in files_of(graph):
        hit = by_base.get(base(v).lower())
        if not hit and same_model and cls in ("UNETLoader", "CLIPLoader", "VAELoader") and not (realism and cls == "UNETLoader"):
            options = {x for _n, c, k, x in theirs if c == cls and k == key}
            if len(options) == 1:
                hit = options.pop()
                changed.append((cls, base(v), base(hit)))
        if hit:
            graph[nid]["inputs"][key] = hit
    return changed


def check(graph, saved, key):
    """The report of «Проверить workflow»: (all right, lines)."""
    lines, ok = [], True
    other = model_of(saved)
    if other != key:
        what = f"для {MODELS[other]}" if other else "без узла Load CLIP для картинок"
        return False, [f"✕ По этому номеру лежит workflow {what}, а нужен {MODELS[key]}.",
                       "Что делать: откройте нужный workflow на RunningHub и впишите сюда номер из его адреса."]
    theirs = files_of(saved)
    names = {base(v).lower() for _n, _c, _k, v in theirs}
    realism = any(c == "LoraLoaderModelOnly" and "realism" in base(v).lower() for _n, c, _k, v in files_of(graph))
    for _nid, cls, k, v in files_of(graph):
        if base(v).lower() in names:
            lines.append(f"✓ {node_title(cls)}: {base(v)}")
            continue
        options = {base(x) for _n, c, kk, x in theirs if c == cls and kk == k}
        if cls in ("UNETLoader", "CLIPLoader", "VAELoader") and len(options) == 1 and not (realism and cls == "UNETLoader"):
            lines.append(f"↻ {node_title(cls)}: у вас в workflow выбран {options.pop()} — программа возьмёт его (вместо {base(v)}).")
        else:
            ok = False
            lines.append(f"? {node_title(cls)}: файла {base(v)} в вашем workflow нет. Проверится при генерации.")
    return ok, lines


def similar(name, options):
    """Up to five files from the list RunningHub offers that look like the wanted one."""
    words = lambda s: {w for w in re.split(r"[^a-z0-9]+", base(s).lower().rsplit(".", 1)[0]) if len(w) > 1}
    want = words(name)
    scored = sorted(((len(want & words(o)), str(o)) for o in options), key=lambda x: (-x[0], x[1]))
    return [o for n, o in scored if n >= 2][:5]


def _one(nid, info, graph):
    """Plain words about the errors of one node (the format of ComfyUI's node_errors)."""
    cls = info.get("class_type") or (graph.get(nid) or {}).get("class_type") or f"№{nid}"
    out = []
    for e in info.get("errors") or []:
        extra = e.get("extra_info") or {}
        field = extra.get("input_name") or str(e.get("details", "")).split(":")[0]
        got = extra.get("received_value")
        if got is None:
            got = (graph.get(nid) or {}).get("inputs", {}).get(field)
        kind = e.get("type", "")
        if kind == "value_not_in_list" and str(got).lower().endswith(FILE_EXT):
            cfg = extra.get("input_config") or []
            options = cfg[0] if cfg and isinstance(cfg[0], list) else []
            text = f"✕ Узел {node_title(cls)}: на RunningHub нет файла {base(got)} (папка {FOLDERS.get(field, 'models')})."
            near = similar(got, options)
            if near:
                text += "\n   Похожие, что там есть: " + ", ".join(near)
            text += f"\n   Что делать: откройте workflow на RunningHub, в узле {node_title(cls)} выберите этот файл или похожий (если его нет в списке, добавьте), сохраните и запустите снова."
        elif kind == "value_not_in_list":
            text = (f"✕ Узел {node_title(cls)}: на RunningHub нет значения «{got}» в настройке «{field}».\n"
                    "   Что делать: в блоке «Настройки картинки» нажмите «Вернуть рекомендуемые».")
        else:
            text = f"✕ Узел {node_title(cls)}: {e.get('message', 'ошибка')} {e.get('details', '')}".rstrip()
        out.append(text)
    return out or [f"✕ Узел {node_title(cls)}: RunningHub его не принял."]


def explain(raw, node_errors, graph, key, wf_id, saved=None):
    """The whole message for the screen. `raw` = the technical text, `node_errors` = ComfyUI's dict when RunningHub sent it."""
    raw = str(raw or "")
    head = f"RunningHub не смог сделать картинку ({MODELS.get(key, 'картинка')}, workflow {wf_id})."
    lines = []
    for nid, info in (node_errors or {}).items():
        if isinstance(info, dict):
            lines += _one(str(nid), info, graph)
    if not lines:
        missing = re.search(r"node(?: type)? ['\"]?([A-Za-z0-9_ |.+\-]+?)['\"]? (?:does not exist|not found)|missing_node_type[^A-Za-z]+([A-Za-z0-9_]+)", raw, re.I)
        files = [v for _n, _c, _k, v in files_of(graph) if base(v).lower() in raw.lower()]
        if missing:
            name = missing.group(1) or missing.group(2)
            lines.append(f"✕ На RunningHub нет узла «{name}».\n   Что делать: откройте workflow на RunningHub, проверьте, нет ли красного узла, "
                         "установите или замените его, сохраните и запустите снова.")
        elif re.search(r"out of memory|OOM|Allocation on device", raw, re.I):
            lines.append("✕ Не хватило видеопамяти.\n   Что делать: выберите видеокарту 48 ГБ или уменьшите размер картинки.")
        elif files:
            for nid, cls, k, v in files_of(graph):
                if v in files:
                    lines.append(f"✕ Узел {node_title(cls)}: проблема с файлом {base(v)} (папка {FOLDERS.get(k, 'models')}).\n"
                                 "   Что делать: откройте workflow на RunningHub, выберите в этом узле другой файл, сохраните и запустите снова.")
        elif not re.search("[А-Яа-яЁё]", raw):
            lines.append("Что делать: откройте workflow на RunningHub и посмотрите на названный ниже узел. Если непонятно, пришлите этот текст автору.")
        else:
            return raw
    if saved is not None and model_of(saved) != key:
        lines.append(f"Заметьте: по номеру {wf_id} лежит workflow не для {MODELS.get(key, '')}. Проверьте номер.")
    text = head + "\n" + "\n".join(lines)
    if "Что делать" not in text:
        text += "\nЧто делать: исправьте это в workflow на RunningHub и запустите снова."
    return text + (f"\n\nТехническая причина: {raw[:700]}" if raw else "")
