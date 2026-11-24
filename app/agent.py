"""Agent mode: a local LLM (Gemma 4 via llama.cpp) that writes MiniMax H3 prompts using skills.

Skills are folders with SKILL.md (+ optional references/*.md). Built-in ones ship in
app/skills, the user's own are uploaded into data/skills. Each skill is "always" on,
"auto" (included when the chat mentions one of its keywords) or "off".
"""

import asyncio
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import aiohttp

import engine

LLAMA_BUILD = "b11985"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip"

HF = "https://huggingface.co/{repo}/resolve/main/{path}"
LLM_VARIANTS = {
    "gemma4-12b": {"title": "Gemma 4 12B", "repo": "ggml-org/gemma-4-12B-it-GGUF",
                   "model": "gemma-4-12B-it-Q4_0.gguf", "model_size": 7.22e9,
                   "mmproj": "mmproj-gemma-4-12B-it-Q8_0.gguf", "mmproj_size": 0.16e9, "ctx": 49152},
    "gemma4-e4b": {"title": "Gemma 4 E4B (лёгкая)", "repo": "ggml-org/gemma-4-E4B-it-GGUF",
                   "model": "gemma-4-E4B-it-Q4_0.gguf", "model_size": 4.59e9,
                   "mmproj": "mmproj-gemma-4-E4B-it-Q8_0.gguf", "mmproj_size": 0.56e9, "ctx": 32768},
    # the same Gemma 4 without refusals (abliterated, Huihui): the file names of the picture part are the same in both repos,
    # so each one is saved under its own name ("mmproj_as")
    "gemma4-12b-unc": {"title": "Gemma 4 12B без цензуры", "uncensored": True, "min_vram": 12,
                       "repo": "huihui-ai/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-GGUF",
                       "model": "Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf", "model_size": 7.381e9,
                       "mmproj": "mmproj-model-bf16.gguf", "mmproj_as": "mmproj-gemma4-12b-unc-bf16.gguf",
                       "mmproj_size": 0.175e9, "ctx": 49152},
    "gemma4-e4b-unc": {"title": "Gemma 4 E4B без цензуры (лёгкая)", "uncensored": True,
                       "repo": "huihui-ai/Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated-GGUF",
                       "model": "Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf", "model_size": 5.302e9,
                       "mmproj": "mmproj-model-bf16.gguf", "mmproj_as": "mmproj-gemma4-e4b-unc-bf16.gguf",
                       "mmproj_size": 0.992e9, "ctx": 32768},
    # Qwen3-VL 8B (Apache 2.0) and its abliterated version
    "qwen3vl-8b": {"title": "Qwen3-VL 8B", "min_vram": 8,
                   "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF", "model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "model_size": 5.028e9,
                   "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", "mmproj_size": 0.752e9, "ctx": 65536},
    "qwen3vl-8b-unc": {"title": "Qwen3-VL 8B без цензуры", "uncensored": True, "min_vram": 8,
                       "repo": "noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
                       "model": "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf", "model_size": 5.028e9,
                       "mmproj": "mmproj-F16.gguf", "mmproj_as": "mmproj-qwen3vl-8b-unc-F16.gguf",
                       "mmproj_size": 1.159e9, "ctx": 65536},
}


def variant_files(root, v):
    """(model path, picture-part path) where this variant keeps its files."""
    return Path(root) / "models" / v["model"], Path(root) / "models" / v.get("mmproj_as", v["mmproj"])

# Keywords that switch on the acting skill in "auto" mode (matched as word starts, any case).
DEFAULT_SKILL_SETTINGS = {
    "hailuo-h3": {"mode": "always", "keywords": []},
    # picture skills: used only when the chat is set to a picture target (see IMAGE_SKILLS), never for video prompts
    "karta-personazha": {"mode": "off", "keywords": []},
    "krea-2": {"mode": "off", "keywords": []},
    "flux2-klein": {"mode": "off", "keywords": []},
    "storyboard": {"mode": "off", "keywords": []},
    "qwen-image-2-1-director-prompt": {"mode": "off", "keywords": []},
    "minimax-h3-image": {"mode": "off", "keywords": []},
    # the screenwriter / director chat («Написать сценарий» in the LLM page): only for that target
    "rezhisser-scenarist": {"mode": "off", "keywords": []},
    "drama-minimax-h3": {"mode": "off", "keywords": []},
    "akterskaya-igra": {"mode": "auto", "keywords": [
        "актер", "актёр", "диалог", "реплик", "эмоци", "игра", "драм", "ссор", "спор", "плач", "слез", "слёз",
        "прощан", "признан", "разговор", "говорит", "говорят", "кричит", "шепчет", "обид", "ревн", "страх",
        "гнев", "грусть", "печал", "радост", "микромимик", "dialog", "dialogue", "emotion", "acting", "actor"]},
}


# "Prompt for" targets that make pictures: which skill writes the prompt for each
IMAGE_SKILLS = {"plist": "krea-2", "krea": "krea-2", "fluxt": "flux2-klein", "qwent": "qwen-image-2-1-director-prompt",
                "qwen": "qwen-image-2-1-director-prompt", "h3t": "minimax-h3-image", "h3e": "minimax-h3-image", "sheet": "karta-personazha", "sheet6": "karta-personazha",
                "story": "storyboard", "outpaint": "qwen-image-2-1-director-prompt", "inpaint": "qwen-image-2-1-director-prompt"}
SCRIPT_SKILLS = ("rezhisser-scenarist", "akterskaya-igra", "drama-minimax-h3")   # the three pillars of the screenwriter chat; the director is the main one
AGENT_TARGETS = ("chat", "script", "video", "lipsync", "refs", "actor", "swap", "action", "actionsc", "long", "icon") + tuple(IMAGE_SKILLS)
# long extra reference files that would not fit next to the rest into the model's context
SKIP_REFS = {"platform-adapters.md", "long-form-writing.md", "h3-format.md"}


def pick_variant(hw):
    return "gemma4-12b" if (hw.get("vram_gb") or 0) >= 12 else "gemma4-e4b"


def parse_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            if v:
                meta[k.strip()] = v.strip().strip("'\"")
        text = text[m.end():]
    return meta, text.strip()


class Skills:
    def __init__(self, builtin_dir, user_dir, config):
        self.builtin_dir, self.user_dir, self.config = Path(builtin_dir), Path(user_dir), config

    def settings(self, sid):
        saved = (self.config.get("skills") or {}).get(sid, {})
        return {**DEFAULT_SKILL_SETTINGS.get(sid, {"mode": "always", "keywords": []}), **saved}

    def list(self):
        out = []
        for base, builtin in ((self.builtin_dir, True), (self.user_dir, False)):
            if not base.exists():
                continue
            for d in sorted(p for p in base.iterdir() if (p / "SKILL.md").exists()):
                if not builtin and (self.builtin_dir / d.name).exists():
                    continue
                meta, _ = parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8", errors="replace"))
                refs = sorted((d / "references").glob("*.md")) if (d / "references").exists() else []
                out.append({"id": d.name, "name": meta.get("name", d.name), "description": meta.get("description", ""),
                            "builtin": builtin, "path": d, "refs": refs, **self.settings(d.name)})
        return out

    def save_settings(self, sid, mode=None, keywords=None):
        all_s = dict(self.config.get("skills") or {})
        cur = dict(all_s.get(sid, {}))
        if mode in ("always", "auto", "off"):
            cur["mode"] = mode
        if keywords is not None:
            cur["keywords"] = [k.strip().lower() for k in keywords if k.strip()]
        all_s[sid] = cur
        self.config.update(skills=all_s)

    def active(self, chat_text, target="video", model=None):
        if target == "chat":  # the plain LLM chat: no skills at all
            return []
        if target == "script":  # the screenwriter / director: its two skills only
            by = {s["id"]: s for s in self.list()}
            return [by[i] for i in SCRIPT_SKILLS if i in by]   # the director first: he is the main one
        if target in IMAGE_SKILLS:  # a picture prompt: only its own skill, the video skills would mislead the model
            ids = [IMAGE_SKILLS[target]]
            if model == "h3img" and target in ("story", "sheet", "sheet6"):   # a storyboard / sheet for MiniMax H3 Image: its own writing rules too
                ids.append("minimax-h3-image")
            return [s for s in self.list() if s["id"] in ids]
        if target == "icon":   # animated icons / Telegram stickers: only the H3 prompt rules; the master prompt of this mode is in system_prompt
            return [s for s in self.list() if s["id"] == "hailuo-h3"]
        text = chat_text.lower()
        res = []
        for s in self.list():
            if s["id"] in IMAGE_SKILLS.values():
                continue
            if s["mode"] == "always":
                res.append(s)
            elif s["mode"] == "auto" and any(re.search(r"\b" + re.escape(k), text) for k in s["keywords"]):
                res.append(s)
        if target == "long":   # the Video studio / clip: always the Hailuo H3 prompt rules (the main skill) and the acting skill; the screenwriter and the drama stay on the LLM page
            by = {s["id"]: s for s in self.list()}
            res = [by[i] for i in ("hailuo-h3", "akterskaya-igra") if i in by] + [s for s in res if s["id"] not in ("hailuo-h3", "akterskaya-igra")]
        # always-on skills first: the prompt prefix stays the same, so the model reuses its cache
        return sorted(res, key=lambda s: s["mode"] != "always")

    def text(self, skill):
        _, body = parse_frontmatter((skill["path"] / "SKILL.md").read_text(encoding="utf-8", errors="replace"))
        parts = [f"# СКИЛЛ «{skill['name']}»\n\n{body}"]
        for ref in skill["refs"]:
            if ref.name in SKIP_REFS:  # only for platforms without their own skill / too long
                continue
            parts.append(f"## Справочник {skill['name']}: {ref.name}\n\n"
                         + ref.read_text(encoding="utf-8", errors="replace").strip())
        return "\n\n".join(parts)

    @staticmethod
    def _text(data):
        """The text of an uploaded file: UTF-8 (with or without BOM), else the Windows Cyrillic code page."""
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp1251", errors="replace")

    @staticmethod
    def _expand(files):
        """Zip / .skill archives are opened in memory (no path escapes, size-limited); other files stay as they are."""
        import io
        out, total = [], 0
        for name, data in files:
            if name.lower().endswith((".zip", ".skill")):
                try:
                    zf = zipfile.ZipFile(io.BytesIO(data))
                except zipfile.BadZipFile:
                    raise ValueError(f"Архив «{Path(name).name}» повреждён или это не zip.")
                for info in zf.infolist():
                    n = info.filename.replace("\\", "/")
                    if info.is_dir() or n.startswith("/") or ".." in n.split("/") or "__MACOSX" in n:
                        continue
                    total += info.file_size
                    if total > 20_000_000 or len(out) > 400:
                        raise ValueError("Архив слишком большой для скилла (скилл - это текстовые файлы, обычно меньше мегабайта).")
                    out.append((n, zf.read(info)))
            else:
                out.append((name, data))
        return out

    def _write_skill(self, sid, items):
        """items: [(relative path inside the skill, text)]."""
        dest = self.user_dir / sid
        if dest.exists():
            shutil.rmtree(dest)
        for rel, text in items:
            target = (dest / rel).resolve()
            if dest.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def save_upload(self, files):
        """files: list of (path, bytes): a skill folder (with SKILL.md), a zip / .skill archive of it, or single .md / .txt
        files with the skill text. Several skills at once are fine. Returns the ids of the skills added."""
        files = self._expand(files)
        texts = (".md", ".markdown", ".txt")
        ids = []
        skill_mds = [p for p, _ in files if Path(p).name.lower() == "skill.md"]
        for sm in skill_mds:
            root = Path(sm).parent
            meta, _ = parse_frontmatter(self._text(next(b for p, b in files if p == sm)))
            sid = re.sub(r"[^\w\-]+", "-", meta.get("name") or root.name or "skill").strip("-").lower() or "skill"
            items = []
            for p, data in files:
                if Path(p).suffix.lower() not in texts:
                    continue
                if root == Path("."):
                    rel = Path(p)
                elif Path(p).is_relative_to(root):
                    rel = Path(p).relative_to(root)
                else:
                    continue
                if rel.name.lower() == "skill.md":
                    rel = rel.with_name("SKILL.md")
                items.append((rel, self._text(data)))
            self._write_skill(sid, items)
            ids.append(sid)
        if not skill_mds:   # no SKILL.md: every plain text file is a skill of its own
            for p, data in files:
                if Path(p).suffix.lower() not in texts:
                    continue
                text = self._text(data)
                meta, body = parse_frontmatter(text)
                name = meta.get("name") or Path(p).stem
                if not meta:
                    first = next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), name)
                    text = "---\nname: " + re.sub(r"[\r\n:]+", " ", name).strip() + "\ndescription: " + re.sub(r"[\r\n]+", " ", first)[:200] + "\n---\n\n" + text
                sid = re.sub(r"[^\w\-]+", "-", name).strip("-").lower() or "skill"
                self._write_skill(sid, [(Path("SKILL.md"), text)])
                ids.append(sid)
        if not ids:
            raise ValueError("Не нашёл скилл. Подойдёт файл .md или .txt, архив .zip или папка с файлом SKILL.md.")
        return ids

    def delete(self, sid):
        d = self.user_dir / sid
        if d.exists() and (d / "SKILL.md").exists():
            shutil.rmtree(d)


def _clean(s, limit):
    """One line of the user's own text for the system prompt: no line breaks or code fences that could break its format."""
    return re.sub(r"[\r\n`]+", " ", str(s or "")).strip()[:limit]


VIDEO_RM_TARGETS = ("lipsync", "refs", "actor", "action", "long", "actionsc")   # every video section with references (not Image-to-Video: first/last frame only): ticked Reference Mode cards = @name


def refmode_rules(cards, target):
    """The Reference Mode cards the user ticked in the chat: who the heroes are, so the Agent writes @name and not their looks."""
    if target not in VIDEO_RM_TARGETS + ("script",) or not isinstance(cards, list):
        return ""
    kinds = {"character": "персонаж", "vehicle": "автомобиль", "object": "предмет", "location": "локация"}
    rows = []
    for c in cards[:6]:
        if not isinstance(c, dict):
            continue
        name = re.sub(r"[^\w\-]", "", str(c.get("name") or ""))[:40]
        if not name:
            continue
        row = f"- @{name} ({kinds.get(c.get('type'), 'персонаж')})"
        if _clean(c.get("what"), 200):
            row += f": {_clean(c.get('what'), 200)}"
        if _clean(c.get("features"), 400):
            row += f". Ключевое: {_clean(c.get('features'), 400)}"
        if c.get("voice"):
            row += ". Есть образец голоса"
        rows.append(row)
    if not rows:
        return ""
    if target == "script":
        return ("ГЕРОИ ИЗ REFERENCE MODE: пользователь уже создал героев (карточки). Они сохранены в приложении:\n" + "\n".join(rows) + "\n"
                "- Напиши сценарий именно про этих героев: их типы, особенности и характер должны влиять на сюжет, конфликт и юмор. "
                "Не придумывай им другую внешность и не меняй то, что указано в карточке. Если героев мало для истории, можно добавить второстепенных, "
                "но главными остаются карточки.\n"
                "- В сценарии и раскадровке называй героев по именам. В промптах для H3 называй их ТОЛЬКО так: @имя (с @, как в списке): приложение "
                "само заменит @имя на описание и подаст изображения карточки. Внешность этих героев в промптах не описывай.\n"
                "- Если пользователь не знает, какую историю придумать, предложи 3 коротких варианта сюжета с этими героями (по 2-3 предложения) "
                "и спроси, какой развивать.")
    voice = ("В этом разделе образец голоса карточки подаётся модели как аудио-референс (до 3 образцов): реплики этого героя пиши от его лица."
             if target in ("actionsc", "actor", "refs", "action") else
             "В этом разделе образец голоса не подаётся (только изображения и описание): голос сделает модель, совпадение голоса не обещай.")
    return ("REFERENCE MODE: пользователь отметил в чате готовые карточки героев (имя, изображения, описание). Они уже сохранены в приложении:\n"
            + "\n".join(rows) + "\n"
            "- Называй этих героев в тексте сцены ТОЛЬКО так: @имя (с @, ровно как в списке). Приложение само заменит @имя на имя героя, "
            "добавит в сцену строку с описанием и подаст изображения карточки как референсы.\n"
            "- Внешность, одежду, возраст и причёску этих героев НЕ описывай и теги <Picture N> на них НЕ ставь: изображения карточек получают номера сами. "
            "Теги <Picture N> только для картинок, приложенных к чату. Если картинок в чате нет, это нормально: правило «картинок нет, опиши героя словами» "
            "к героям из карточек не относится.\n"
            "- Реплики пиши в <d>[Язык] ...</d> и называй, кто говорит: `@имя says: <d>[English] ...</d>`. Если пользователь дал слова, вставь их дословно "
            "и без правок; если не дал и они нужны, коротко придумай по одной реплике на 2-3 секунды сцены или спроси. " + voice + "\n"
            "- Приложение само допишет в начало промпта строки RefMod: `<Subject N> is имя ... in <Video N>`, `<Audio N> is the voice-timbre reference for <Subject N> (S N)` "
            "и пометит говорящего (S N). Эти строки, теги <Subject>/<Video>/<Audio> и номера спикеров сам НЕ пиши: только @имя и `@имя says: <d>[English] ...</d>`.\n"
            "- Не придумывай героев, которых нет в списке: для остальных действуй по обычным правилам.")


def system_prompt(skills_text, target, ctx):
    """App rules first (they decide the output format), then the skills."""
    dur = ctx.get("duration")
    n_img = ctx.get("images", 0)
    if target == "chat":
        import datetime as _dt
        _n = _dt.datetime.now()
        _wd = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")[_n.weekday()]
        _mo = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")[_n.month - 1]
        return ("Ты - обычный помощник (LLM) внутри приложения Minimax H3 Workspace. Отвечай на языке пользователя, по делу. "
                f"Сейчас по часам компьютера пользователя: {_wd}, {_n.day} {_mo} {_n.year} года, {_n:%H:%M}. Если спрашивают число, день недели или время - отвечай по этим данным. "
                "У тебя нет доступа к интернету: погоду, новости, курсы и другие данные «на сегодня» ты узнать не можешь. Не выдумывай их: честно скажи, что у тебя нет выхода в интернет, и предложи помощь по-другому. "
                "Можешь писать и править тексты, идеи, сценарии, промпты для любых моделей и отвечать на любые вопросы. "
                "Если пользователь приложил изображения, смотри на них и опирайся на них."
                + (("\n\nПользователь выбрал для этого чата скиллы. Работай строго по ним, их правила важнее общих привычек:\n\n" + skills_text) if skills_text.strip() else ""))
    if target == "script":
        base_script = ("Ты - сценарист и режиссёр-постановщик внутри приложения Minimax H3 Workspace. Работай строго по скиллам ниже: "
                "«Режиссёр-сценарист» (идея -> сценарий, раскадровка, стиль, темпоритм), «Актёрская игра» (эмоции, интонация, мимика, жесты) и «Драма» (сценарий -> серия промптов по 15 секунд для MiniMax H3).\n"
                "- Главный скилл «Режиссёр-сценарист»: он работает всегда. Из «Актёрской игры» бери эмоции и подачу реплик, из «Драмы» бери структуру сцены, "
                "реплики дословно и формат H3, когда история драматическая или пользователь просит драму или промпты.\n"
                "- Общайся на языке пользователя. Если не хватает важного (герой, место, настроение, длина), задай 2-3 коротких вопроса, потом пиши.\n"
                "- По умолчанию пиши сценарий и раскадровку для пользователя. Готовые английские промпты для H3 по 15 секунд пиши только когда "
                "пользователь просит «промпты», «для H3» или «драма»; каждый промпт клади в отдельный блок ```prompt.\n"
                "- Изображения героев, если приложены, используй как референс внешности.\n"
                "- Если среди героев сценария есть те, кого пользователь ещё не завёл как карточки Reference Mode, в конце ответа дай короткий список: "
                "имя, тип (персонаж, животное, автомобиль, предмет, локация), одна строка «кто это» и приметы для карточки (цвет волос и глаз, одежда, особенности). "
                "Скажи, что их можно сохранить в разделе Reference Mode.")
        _rm = refmode_rules(ctx.get("refmodes"), target)
        return base_script + (("\n\n" + skills_text) if skills_text.strip() else "") + ("\n\n" + _rm if _rm else "")   # the texts of the three skills are the rules of this chat
    if target == "lipsync":
        mode_rules = (
            "Режим приложения: LIPSYNC (Singularity Ref2V + Audio Lock). Видео делается по референс-картинкам и "
            "аудиофайлу пользователя, длина видео = длине аудиофрагмента"
            + (f" ({dur:.1f} с)" if dur else "") + ".\n"
            "- Первая строка промпта - выравнивание референсов: для каждой картинки по порядку "
            "`<Picture N> (from [Shot 1]) is fully referenced.` после фразы `For the target video, at 0.00 seconds "
            "into the target video,`. Затем пустая строка и три поля H3.\n"
            "- Реплики внутри <d>[Язык] ...</d> должны дословно совпадать со словами в аудио. Если в сообщении есть блок "
            "`[Аудио ... Слова из аудио ...]`, возьми слова из него и вставь ДОСЛОВНО (без перевода и правок), язык "
            "укажи тот, что назван в блоке. Если слов нет и пользователь не дал текст - спроси его, не выдумывай.\n"
            "- Звук берётся из аудио пользователя, поэтому `overall_soundscape: N/A` и `non_diegetic_music: N/A`, "
            "если пользователь не просит иначе.\n"
            "- Рот говорящего должен оставаться видимым, без больших поворотов головы.")
    elif target == "actor":
        voice = ctx.get("voice")
        variant = ("ВАРИАНТ 1 (с голос-референсом): в разделе загружен образец голоса, поэтому у говорящих строка ролей с `@audio1`, "
                   "привязка к голосу перед каждой репликой, `(S1)`/`(S2)`." if voice else
                   "ВАРИАНТ 2 (без голоса): образец голоса не загружен, поэтому строку про `@audio1` и (S1) не пиши; звук сцены "
                   "строй из дыхания, ткани, шагов, комнаты." if voice is not None else
                   "Если пользователь не сказал, загружен ли образец голоса, выбери вариант по тексту: есть реплики - вариант 1, "
                   "нет - вариант 2.")
        mode_rules = (
            "Режим приложения: АКТЁРСКАЯ ИГРА (ComfyUI, MiniMax H3 Reference to Video). Это режим ComfyUI из скилла "
            "«Актёрская игра»: примени его раздел «Режим ComfyUI (H3 Reference to Video)» строго как написано, он важнее общих "
            "правил формата. Первая картинка - стартовый кадр видео (`@image1 - keyframe, first frame of the target video`), "
            "модель продолжает сцену именно с неё. Голос берётся из загруженного образца только как тембр, а слова произносятся "
            "новые - те, что ты пишешь в <d>[Язык] ...</d>. "
            f"Длительность клипа {dur or 14} с (максимум 25) - таймкоды не выходят за неё.\n"
            f"- {variant}\n"
            "- Ответ - один английский блок prompt: строка ролей, строка выравнивания, `integrated_multimodal_description` "
            "с шотами и таймкодами, `overall_soundscape`, `non_diegetic_music: N/A`. Без русской версии.\n"
            "- Внешность персонажей не описывай - её закрывает картинка. Никаких слов про шёпот: речь должна звучать.\n"
            "- Если реплики не помещаются в длительность, одной строкой скажи, сколько секунд поставить в разделе.")
    elif target == "plist":
        mode_rules = (
            "Режим приложения: СПИСОК КАДРОВ (серия отдельных картинок, раздел «Картинка по тексту», режим «Список промптов»). "
            "Пользователь описывает историю или набор картинок, а ты выдаёшь промпт для КАЖДОЙ картинки. Используй скилл «krea-2» "
            "для написания каждого промпта.\n"
            "- Ответ - ОДИН блок prompt, внутри отдельные промпты, по одному на картинку, разделённые ПУСТОЙ СТРОКОЙ. Без номеров, "
            "заголовков и подписей: приложение само делит текст по пустым строкам.\n"
            "- Каждая картинка рисуется отдельно и ничего не знает о других, поэтому в КАЖДОМ промпте заново опиши героя, место и "
            "стиль одними и теми же словами (так серия получится единой), меняя только действие, ракурс и свет.\n"
            "- Сколько картинок: столько, сколько назвал пользователь, иначе 4-8; не больше 50.\n"
            "- Каждый промпт - один абзац, это картинка, не видео: без таймкодов, шотов и звука.")
    elif target == "krea":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (Krea 2, раздел «Картинка по тексту»). Нужен промпт для ОДНОЙ КАРТИНКИ, "
            "это не видео: без таймкодов, шотов, звука и формата H3. Используй скилл «krea-2» для самого промпта.\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Русскую версию, слайдеры, creativity, "
            "референсы стиля и выбор варианта модели не выводи: в приложении их нет.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 "
            "(он выбирается в разделе). Если нужны реалистичные люди или фото, добавь, что в разделе есть стиль «Реализм».\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "story" and ctx.get("frames"):
        mode_rules = (
            f"Режим приложения: РАСКАДРОВКА ПО КАДРАМ ({'Qwen Image 2.1' if ctx.get('model') == 'qwen' else 'FLUX.2 klein'}). Каждый кадр рисуется ОТДЕЛЬНОЙ картинкой, а лист склеивает "
            "программа. Используй скилл «storyboard», раздел «Режим по кадрам».\n"
            "- Ответ: одной строкой по-русски сколько кадров; затем ОДИН блок prompt, в котором каждый кадр на своей строке: "
            "`1: ...`, `2: ...` и так далее. Число кадров строго 2, 3, 4, 6, 8 или 9 (если пользователь назвал другое число, "
            "выбери ближайшее из этих и скажи об этом). Каждая строка это полный самостоятельный английский промпт ОДНОГО кадра: "
            "стиль, героя (одинаково во всех кадрах), действие, план, свет. Никаких рамок, листов, линий и текста на картинке.\n"
            + ("- Ссылки на картинки в тексте: " + ('пиши <image1>, <image2>, <image3> (Qwen)' if ctx.get("model") == "qwen" else 'пиши "image 1", "image 2", "image 3" без скобок (FLUX)') + ".\n")
            + ("- К чату ПРИЛОЖЕНЫ картинки (%d): они подгружаются как референсы. <image1> первая приложенная (герой или лист "
               "персонажа), <image2> вторая (локация), <image3> третья. В КАЖДОЙ строке пиши героя и ссылкой (\"the character from "
               "image 1\") и коротко словами (одежда, причёска, цвета), пиши, что кадр новая композиция, а не копия image 2, и "
               "начинай каждую строку с \"In the same art style as image 1, a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette.\" Пиши \"one single\" перед героем, чтобы он не раздваивался." % n_img if n_img else
               "- Картинок нет: опиши героя коротко и ОДИНАКОВО в каждой строке, стиль один во всех; каждую строку начинай с \"a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette\". Если есть карта персонажа "
               "или локация, одной строкой подскажи приложить их (скрепка).")
        )
    elif target == "story":
        model = {"krea": "Krea 2", "flux": "FLUX.2 klein", "qwen": "Qwen Image 2.1"}.get(ctx.get("model"), "Krea 2")
        mode_rules = (
            f"Режим приложения: РАСКАДРОВКА (одна картинка 16:9 из нескольких кадров, модель {model}). Используй скилл "
            f"«storyboard» и в разделе «Шаг 4» ОДИН шаблон: для {model}. Пользователь рассказывает историю по кадрам.\n"
            "- Ответ: одной строкой по-русски число кадров и сетка, затем один английский блок prompt. Без номеров, подписей, "
            "стрелок и любого текста на картинке, без слов про видео. Негатив не выводи (у Qwen он вписывается в приложении). "
            "Русскую версию не выводи.\n"
            "- Если пользователь не назвал число кадров и не перечислил кадры, возьми 6.\n"
            + ("- К чату ПРИЛОЖЕНЫ картинки (%d): они подгружаются в генерацию как референсы. Порядок задан пользователем: "
               "<image1> первая приложенная (обычно герой или лист персонажа), <image2> вторая (обычно локация), <image3> третья. "
               "Ссылайся на них в промпте как <image1>, <image2> (для FLUX.2 пиши \"image 1\", \"image 2\"), героя и место "
               "словами не описывай, пиши \"the character from <image1>\", \"the location from <image2>\", \"same art style as "
               "<image1>\". Это работает только для Qwen 2.1 и FLUX.2 klein: если выбрана Krea 2, одной строкой скажи переключить "
               "модель. Используй раздел скилла про референсы." % n_img
               if n_img and ctx.get("model") != "krea" else
               "- Картинки, приложенные к чату, это образцы стиля или героя: кратко опиши героя словами, в раскадровку по тексту "
               "они не подгружаются. Если нужна точность по картинкам, подскажи выбрать Qwen 2.1 или FLUX.2 klein." if n_img else
               "- Если у пользователя есть карта персонажа или локация, одной строкой подскажи приложить их (скрепка) и выбрать "
               "Qwen 2.1 или FLUX.2 klein: тогда герой и место возьмутся с картинок.")
        )
    elif target == "fluxt":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (FLUX.2 klein, раздел «Картинка по тексту», модель FLUX.2 klein). Нужен промпт "
            "для ОДНОЙ КАРТИНКИ, это не видео: без таймкодов, шотов и звука. Используй скилл «flux2-klein».\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Негатива у модели нет, не выводи его. "
            "Русскую версию и настройки модели не выводи.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 (выбирается в разделе).\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "h3t":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (MiniMax H3 Image, раздел «Изображение по тексту», модель MiniMax H3 Image). Нужен "
            "промпт для ОДНОЙ КАРТИНКИ, это не видео: без таймкодов, шотов и звука. Используй скилл «minimax-h3-image».\n"
            "- Нужен только английский промпт одним блоком prompt, подробный, на весь кадр (80-250 слов). Негативного промпта у модели "
            "нет, не выводи его. Русскую версию и настройки модели не выводи.\n"
            "- Одной строкой перед блоком подскажи формат кадра из 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3, 21:9 и размер (8 или "
            "16 МП: от 8 МП картинка чистая, лучше всего 16 МП, это ~2 минуты на видеокарте 16 ГБ; 4 МП даёт зерно, только для проб).\n"
            "- Картинки, приложенные к чату, здесь только образцы настроения: не пересказывай их подробно. Если пользователю нужно "
            "сделать по ним (стиль, герои), одной строкой подскажи цель «MiniMax H3 Image: правка».")
    elif target == "h3e":
        mode_rules = (
            f"Режим приложения: ГЕНЕРАЦИЯ И ПРАВКА ПО РЕФЕРЕНСАМ (MiniMax H3 Image, раздел «Редактирование» → «Править по описанию», "
            f"модель MiniMax H3 Image). К чату приложено картинок: {n_img}, в приложении можно до 5. Они подаются как <Picture 1>, "
            "<Picture 2>… в порядке слотов; первая задаёт форму результата. Используй скилл «minimax-h3-image».\n"
            "- Для КАЖДОЙ картинки скажи её роль и что не брать (например, <Picture 1> стиль без людей и композиции, <Picture 2> и "
            "<Picture 3> герои с сохранением лица и одежды, место словами). Пользователь тегов не пишет, расставляешь их ты. "
            "Роли пользователь называет своими словами; если не назвал, угадай по описанию и в конце одной строкой скажи, в каком порядке "
            "класть картинки в слоты.\n"
            "- Если картинка одна и это правка, напиши инструкцию: что изменить и что обязательно сохранить (лицо, причёска, одежда, поза, "
            "композиция, свет). Одна правка за раз.\n"
            "- Ответ: один английский блок prompt, подробный. Негатив не выводи. Размеры и настройки не выводи; одной строкой подскажи "
            "формат кадра и размер (8 или 16 МП; 4 МП даёт зерно), не больше 5 картинок.\n"
            + ("- КАРТИНОК В ЧАТЕ НЕТ: попроси приложить их (скрепка или «Из Библиотеки»), не выдумывай." if not n_img else ""))
    elif target == "qwent":
        mode_rules = (
            "Режим приложения: КАРТИНКА ПО ТЕКСТУ (Qwen Image 2.1, раздел «Картинка по тексту», модель Qwen 2.1). Нужен промпт "
            "для ОДНОЙ КАРТИНКИ без референса, это не видео. Используй скилл Qwen Image для формулировки (наблюдательный стиль).\n"
            "- В приложении нужен только английский промпт одним блоком prompt. Негативный промпт в приложении вписывается "
            "отдельным полем, поэтому НЕ выводи блок negative. Русскую версию и RGBA/мультиреференс не выводи.\n"
            "- Одной строкой перед блоком подскажи формат кадра из списка 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3.\n"
            "- Картинки, приложенные к чату, это образцы настроения и композиции, их не пересказывай подробно.")
    elif target == "qwen":
        mode_rules = (
            "Режим приложения: ПРАВКА КАРТИНКИ (Qwen Image 2.1, раздел «Правка картинки»). Пользователь прикладывает к чату "
            "картинку (Picture 1), которую надо изменить, и пишет, что поменять. Используй скилл Qwen Image для формулировки.\n"
            "- Если картинки в чате нет, попроси приложить её (скрепка внизу или «Из Библиотеки»), не выдумывай.\n"
            "- Ответ: один английский блок prompt с инструкцией правки: что изменить и что обязательно сохранить "
            "(лицо, причёска, одежда, поза, композиция, свет). Одна правка за раз. Если приложено несколько картинок: <image1> "
            "первая (её меняем), остальные (<image2>, <image3>) это референсы (герой, предмет, одежда, место): ссылайся на них "
            "по номеру, например «put the character from <image2> into the scene of <image1>».\n"
            "- RGBA, мультиреференс, размеры и другие настройки не выводи: в приложении их нет. Без видео-терминов.")
    elif target == "outpaint":
        mode_rules = (
            "Режим приложения: РАСШИРЕНИЕ ИЗОБРАЖЕНИЯ (Outpaint, Qwen Image 2.1, раздел «Редактирование» → «Расширение»). "
            "Пользователь прикладывает к чату изображение (Picture 1), которое нужно достроить до другого формата "
            "(например, вертикальное до квадрата). Программа сама ставит изображение по центру нового холста и оставляет его "
            "без изменений, модель рисует только НОВЫЕ края. Используй скилл Qwen Image для формулировки.\n"
            "- Если изображения в чате нет, попроси приложить его (скрепка или «Из Библиотеки»), не выдумывай.\n"
            "- Программа САМА добавит в начало промпта служебную фразу «Outpaint the image: replace the solid gray areas with a "
            "seamless continuation of the scene…» и слово «Scene:». Её НЕ пиши и не повторяй.\n"
            "- Ответ: один короткий английский блок prompt (1-3 предложения), только ОПИСАНИЕ СЦЕНЫ: что на изображении "
            "(место, свет, стиль) и что естественно продолжается по краям (например, «a bright coastal living room with white "
            "walls and light wooden floor; the room continues on both sides with more wall, floor and a window, same daylight»). "
            "Без инструкций вроде «extend» и «outpaint». Тот же стиль, свет, перспектива и палитра. Слова и надписи в новой "
            "части не задавай: они получатся неразборчивыми.\n"
            "- Модель работает с CFG 1: негативный промпт у неё НЕ действует, блок negative не выводи. Всё нужное говори "
            "в позитивной форме и коротко: продолжай то, что уже есть (стены, пол, потолок, фон, небо, горизонт), не вводи новых "
            "предметов, людей, декора и мебели, которых нет на изображении (пиши «plain continuation of the walls and floor», "
            "а не «without objects»). Чем проще продолжение, тем меньше искажений и лишних деталей.\n"
            "- Одной строкой по-русски напомни выбрать в разделе нужный формат (16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3, 21:9). "
            "Если пользователь не уточнил, что дорисовать, предложи нейтральное продолжение сцены. Размеры и настройки не выводи.")
    elif target == "inpaint":
        mode_rules = (
            "Режим приложения: ИНПЕЙНТ (Qwen Image 2.1, раздел «Редактирование» → «Инпейнт»). Пользователь прикладывает к чату "
            "изображение (Picture 1) и потом в приложении закрашивает кистью место, которое нужно переделать. Модель перерисует "
            "только закрашенное, всё остальное остаётся пиксель в пиксель. Используй скилл Qwen Image для формулировки.\n"
            "- Если изображения в чате нет, попроси приложить его (скрепка или «Из Библиотеки»), не выдумывай.\n"
            "- Ответ: один короткий английский блок prompt: что должно быть НА ЗАКРАШЕННОМ МЕСТЕ (предмет, материал, цвет, "
            "фактура) и как оно сочетается с остальным (тот же свет, тени, перспектива, стиль и толщина линий). Опиши только "
            "то, что рисуется на месте закраски, а не всё изображение. Одна правка за раз. Если нужно просто убрать предмет, "
            "напиши, чем заполнить место (например, «empty wooden floor, same texture and light»).\n"
            "- Одной строкой по-русски напомни: закрасить нужно с небольшим запасом вокруг предмета, чтобы края слились. "
            "Размеры, маски и настройки не выводи.")
    elif target == "sheet":
        mode_rules = (
            "Режим приложения: ЛИСТ ПЕРСОНАЖА по описанию (картинка Krea 2 в разделе «Картинка по тексту», формат 16:9). "
            "Используй скилл «Карта персонажа».\n"
            "- Ответ: один английский блок prompt на ОДНУ картинку-лист из 4 панелей, без видео-терминов. Русскую версию "
            "и промпты для других моделей не выводи.\n"
            "- Если к чату приложена картинка персонажа, опираясь на неё, опиши его внешность (DNA) подробно: лист рисуется "
            "с нуля по тексту. Если персонаж уже есть картинкой и нужен лист именно по ней, одной строкой подскажи "
            "раздел «Правка картинки» → «Лист персонажа»: там это делается по самой картинке.")
    elif target == "sheet6":
        mode_rules = (
            "Режим приложения: ЛИСТ ПЕРСОНАЖА из 6 ПАНЕЛЕЙ по описанию (картинка Krea 2 в разделе «Картинка по тексту», "
            "широкий формат 21:9). Используй скилл «Карта персонажа», раздел «Вариант из 6 панелей» (шаблон Krea 2).\n"
            "- Ответ: один английский блок prompt на ОДНУ картинку-лист из шести панелей в один ряд, без видео-терминов. "
            "Русскую версию и промпты для других моделей не выводи.\n"
            "- Если к чату приложена картинка персонажа, опираясь на неё, опиши его внешность (DNA) подробно: лист рисуется "
            "с нуля по тексту. Если лист нужен именно по готовой картинке, одной строкой подскажи раздел «Правка картинки» → "
            "«Лист персонажа» → 6 панелей: там это делается по самой картинке и точнее.")
    elif target == "long":
        lg = ctx.get("long") or {}
        sound = lg.get("sound")
        secs = lg.get("secs") or []
        snd = ("ЗВУК: пользователь загрузит свою песню или голос, видео подстроится под них. Слова песни в текст НЕ пиши, "
               "модель их слышит. Пиши только игру: пение, эмоции, движения и камеру в ритм. Если пользователь назвал части песни "
               "(куплет, припев, проигрыш), раздели сцены по ним, иначе делай сцены по 10 секунд."
               if sound == "own" else
               "ЗВУК: своего аудио нет, модель сама делает звук по тексту. Реплики героев пиши в <d>[Язык] ...</d> на языке "
               "пользователя; фоновый звук и музыку за кадром пиши отдельными строками overall_soundscape и non_diegetic_music (см. формат ответа)."
               if sound == "none" else
               "ЗВУК: если пользователь не сказал, с песней ролик или без, спроси одним вопросом. С песней слова песни в текст "
               "не пиши (модель их слышит), без песни реплики героев пиши в <d>[Язык] ...</d>.")
        mode_rules = (
            "Режим приложения: ДЛИННОЕ ВИДЕО (сцены идут одна за другой, Singularity Ref2V + Pika). Ролик состоит из сцен, "
            "длина сцены от 3 до 15 секунд (по умолчанию 10), сцен не больше 12"
            + (f"; сейчас в разделе {len(secs)} сцен: {', '.join(str(int(x)) for x in secs)} с" if secs else "")
            + ". Если пользователь не назвал число сцен, предложи его сам по описанию (обычно 2-6).\n"
            "ГЛАВНОЕ: каждая следующая сцена продолжает прошлую, но модель видит только последнюю секунду прошлой сцены и "
            "референс-картинки. Текст соседних сцен она НЕ видит. Поэтому:\n"
            "- в КАЖДОЙ сцене коротко заново назови героя и место, ссылаясь на референс-картинки как <Picture 1>, <Picture 2> и т.д. "
            f"(приложено картинок: {n_img}, можно до 9; пользователь сам решает, что на какой: персонаж, локация, предмет"
            + ("; КАРТИНОК НЕТ: теги <Picture N> не используй, а в каждой сцене словами опиши героя, одежду и место, одинаково во всех сценах" if not n_img else "")
            + "); "
            "внешность подробно не расписывай, её даёт картинка;\n"
            "- ВАЖНО: ничего до `SCENE 1` не пиши. Первые слова КАЖДОЙ сцены (после строки `SCENE N | S`, внутри её абзаца, "
            "и в сцене 2, 3 и далее тоже) - короткая фраза про роли картинок по словам пользователя (что на какой: персонаж, "
            "локация, предмет, одежда), например `<Picture 1> is the character reference (keep this face exactly). <Picture 2> is "
            "the location reference.` Пользователь теги сам не пишет, их ставишь ты; если роли не названы, угадай по описанию "
            "и в конце одной строкой скажи, в каком порядке класть картинки;\n"
            "- одно главное действие на сцену, не пять;\n"
            "- камера отдельной фразой (наезд, облёт, отъезд), свет и настроение одной фразой;\n"
            "- начало сцены продолжает конец прошлой: на стыке не меняй место, одежду и позу резко, смену места делай плавной;\n"
            "- ЕДИНЫЙ МИР: перед текстом каждой сцены повторяй одни и те же слова про героя (имя, одежда, причёска), место, время "
            "суток, свет и цветовую гамму, чтобы после смены плана всё выглядело как тот же фильм. Меняй только ракурс, крупность "
            "и действие;\n"
            "- СМЕНА ПЛАНА как в кино: общий план, потом средний, потом крупный; взгляд героя и затем то, на что он смотрит; "
            "вид сзади, потом спереди. Не меняй место и одежду без причины в истории. Звук (музыка, голос, шум места) описывай "
            "одинаковым по духу во всех сценах, чтобы он тек через склейки;\n"
            "- КАЧЕСТВО: если сцены продолжают друг друга много раз подряд, картинка постепенно портится (лицо, чёткость). "
            "Поэтому, когда по истории меняется план, место или ракурс (как в кино: другая локация, крупный план вместо общего), "
            "начинай ТАКУЮ сцену как НОВЫЙ ПЛАН: добавь в строку заголовка слово new: `SCENE 3 | 8 | new`. Новый план не продолжает "
            "прошлую сцену, а строится заново по референс-картинкам, поэтому качество в нём как в первой сцене. Продолжение "
            "(без new) оставляй только там, где действие идёт одним кадром без разрыва, и не больше 3 сцен подряд;\n"
            "- ПРАВИЛА MINIMAX H3 (скилл Hailuo H3 главный для написания промпта, следуй ему дословно; актёрская игра: эмоции, интонация, мимика): 1) после ролей картинок первым идёт ВИЗУАЛЬНЫЙ СТИЛЬ "
            "(`Live-action, cinematic, ...`): бери стиль из слов пользователя, свой не выдумывай; если стиль не назван, а в сюжете живые люди "
            "и обычное место, пиши `Live-action, cinematic`, иначе в конце ответа одной строкой спроси стиль; 2) потом стартовая композиция, "
            "действие по нарастающей (начало, изменение, итог/реакция), НЕ статичная картинка; 3) камера: тип движения + амплитуда + скорость "
            "одним естественным предложением (`Push In with small amplitude at slow speed`, `Static Shot`, `Pan Left`, `Arc Shot`, `Tracking Shot`), "
            "не просто `medium shot`; 4) диалог: говорящему при первом появлении даёшь устойчивый ID и описание голоса (возраст, тембр, темп) и ЭМОЦИЮ "
            "вне блока реплики, сама реплика дословно в `<d>[Язык] ...</d>`: `@Имя says, in a quiet, teasing tone: <d>[Russian] ...</d>`, "
            "если двое, у каждого свой ID `(S1)`, `(S2)` и своя интонация, слушающий реагирует лицом и позой; сцену режь по границе реплики, "
            "фраза не обрывается; 5) без абстракций вроде «тёплая интимная атмосфера»: только то, что видно и слышно;\n"
            f"- {snd}\n"
            "ФОРМАТ ОТВЕТА: ровно ОДИН блок prompt. Внутри сцены по порядку, каждая начинается отдельной строкой "
            "`SCENE N | S`, где N номер сцены, S длина в секундах (целое 3-15), а дальше один абзац обычного английского "
            "текста (это содержимое поля integrated_multimodal_description). БЕЗ `integrated_multimodal_description`, `[Shot 1]` и таймкодов: "
            "приложение добавит обёртку само. ПОСЛЕ абзаца КАЖДОЙ сцены с новой строки добавь две строки: `overall_soundscape: ...` "
            "(1-4 предложения на английском: только физические фоновые звуки сцены, без диалогов и музыки; `N/A` только если "
            "пользователь просит полную тишину) и `non_diegetic_music: ...` (1-3 предложения: инструменты, темп, ритм; `N/A`, если музыки "
            "не должно быть). Не пиши музыку в одном месте и `N/A` в другом. Пример:\n"
            "SCENE 1 | 10\n<Picture 1> is the character reference (keep this face exactly). <Picture 2> is the location reference. "
            "A knight <Picture 1> stands in the ruined throne room <Picture 2>, slowly raising a glowing guitar. "
            "Camera pushes in from wide to medium shot. Red glow, drifting ash.\n"
            "overall_soundscape: Distant crackle of fire, falling ash and a low stone echo.\n"
            "non_diegetic_music: Slow distorted electric guitar over deep drums at a heavy tempo.\n\n"
            "SCENE 2 | 10\n<Picture 1> is the character reference (keep this face exactly). <Picture 2> is the location reference. "
            "The knight <Picture 1> in the ruined throne room <Picture 2> starts playing fiercely, head tilted back. "
            "Camera circles him at chest height. Flickering ghostly light on the walls.")
        if lg.get("clip"):   # the music-clip page: full prompts, not the short ones of the plain studio
            mode_rules += (
                f"\n\nКАРТИНКИ КЛИПА: к чату приложено РОВНО {n_img} картинок(и). Используй ТОЛЬКО теги <Picture 1>"
                + (f"…<Picture {n_img}>" if n_img > 1 else "") +
                ", других номеров не бывает. Если на одной картинке несколько ракурсов, раскадровка или лист героя (крупный и общий план рядом), "
                "это всё равно ОДНА картинка с ОДНИМ номером: не называй её части <Picture 2>, <Picture 3> и не придумывай картинку одежды или "
                "локации, которой нет. Примеры с <Picture 2> выше только образец формата: берись за них, лишь если картинок действительно больше одной."
                "\n\nРЕЖИМ «СОЗДАНИЕ КЛИПА»: ЭТИ ПРАВИЛА ВАЖНЕЕ пунктов выше про краткость («одно главное действие», «камера одной фразой»). "
                "Промпт КАЖДОЙ сцены должен быть полным, как для профессионального клипа по правилам MiniMax H3 из скилла: это ОДИН плотный "
                "абзац на английском в 110-180 слов (после фразы про роли картинок), а не 2-3 коротких предложения. Что должно быть в абзаце:\n"
                "1) крупность плана и ракурс (wide / medium / close-up, высота камеры, угол) и ОБЪЕКТИВ: фокусное расстояние и глубина резкости;\n"
                "2) КАМЕРА И ПЛЁНКА: если пользователь просит реализм, кино, синематик, документальную или рекламную съёмку, ПИШИ, на что это "
                "снято: например `shot on ARRI Alexa 35`, `ARRI Alexa Mini LF with 35mm anamorphic lenses`, `RED V-Raptor`, `Sony Venice 2`, "
                "`clean digital cinema image`; подбирай под жанр и держи ОДНУ и ту же камеру и объективное семейство во всех "
                "сценах клипа, меняя только крупность. Если стиль не реалистичный (аниме, 3D-мультфильм, иллюстрация, стилизация), пиши "
                "соответствующие этому стилю термины вместо камеры;\n"
                "3) ДВИЖЕНИЕ КАМЕРЫ со скоростью и характером (slow push-in, handheld with subtle sway, gimbal orbit, crane up, whip pan) "
                "и движение героя по долям сцены (сначала..., затем..., в конце...), согласованное с ритмом и энергией этого куска музыки;\n"
                "4) СВЕТ и ЦВЕТ: источник и направление света, температура, контраст, цветокоррекция (teal-orange, warm golden hour, soft diffusion), "
                "атмосфера (haze, dust, rain), фактуры (кожа, ткань, поверхности), мелкие детали места;\n"
                "5) ГЕРОЙ: поза, выражение лица, эмоция, одежда как в картинке, действие. Если у сцены стоит метка «поёт»: пиши, что герой "
                "поёт в камеру, губы точно совпадают с вокалом (clear lip-sync), видно дыхание и мимику; если «проигрыш»: герой не поёт, "
                "рот закрыт; если «танцы»: опиши танец и ритм движений;\n"
                "6) ЗВУК внутри абзаца одной фразой: характер музыки в этот момент, дыхание, шум места. Слова песни сам НЕ придумывай и не пиши: приложение допишет к сцене с меткой «поёт» точные слова на языке песни по распознаванию, твоё дело написать, что героиня поёт в камеру с точным липсинком.\n"
                "Пиши КОНКРЕТНО и зрительно, без воды и общих слов («beautiful», «amazing» не нужны). Сохраняй единый мир клипа: те же герой, "
                "одежда, камера и цветокоррекция, меняются только место, крупность, ракурс и действие. Формат ответа не меняется: блоки "
                "`SCENE N | S` и абзац под каждым.\n\n"
                "СЮЖЕТ: описание клипа от пользователя - это ПОЖЕЛАНИЯ и набор примерных кадров (герой, места, крупности, движения), а не "
                "жёсткая раскадровка и не тайминг. Отталкивайся от них, но сюжет выстраивай САМ, как режиссёр (скилл режиссёра), по ходу музыки: "
                "ритм, энергия и вокал куска подсказывают, где крупные планы деталей (бретелька, глаза, губы, руки, волосы), где средние и общие, "
                "где движение героя. Меняй локации и ракурсы по сюжету, повторяй идеи из пожеланий в разных сочетаниях. Скилл режиссёра "
                "используй только для замысла и драматургии, формат ответа и длины сцен остаются как выше.\n\n"
                "ДЛИНА СЦЕН (жёстко, это правило важнее пожелания пользователя о других длинах): обычные сцены 3-4 секунды. Сцены с меткой «поёт» "
                "(липсинг) 5-8 секунд, длину выбирай по фразе вокала: границы ставь в паузах между фразами, НИКОГДА посреди слова или строки, "
                "и не делай ровных одинаковых длин подряд. Не превышай максимум, "
                "заданный пользователем.\n"
                "СМЕНА ПЛАНА ВНУТРИ СЦЕНЫ: сцена 5 секунд и длиннее НЕ держит один и тот же план. Сцена 5 секунд: два плана по ~2.5 с. Сцена "
                "6-8 секунд: два плана примерно по половине (разная крупность и ракурс, например close-up, затем wide или medium). Пиши это в "
                "абзаце по долям: `0-2.5s: close-up ...; cut to 2.5-5s: medium shot ...`, с резкой монтажной склейкой, без плавного перетекания. "
                "Сцены 3-4 секунды: один план.\n"
                "ВИД КАДРА (очень важно): это современное чистое цифровое видео. НИКОГДА не пиши слова про плёнку и её дефекты, даже с отрицанием "
                "(видеомодель реагирует на сами слова и рисует старую плёнку): film, grain, scratches, dust, hair, flicker, jitter, "
                "gate weave, light leaks, burn, sprocket, frame borders, vintage, Kodak, print и т.п. Описывай вид ТОЛЬКО позитивно. В конец КАЖДОГО "
                "промпта добавляй одной фразой: `clean sharp digital cinema image, natural skin texture, rich neutral color grading, "
                "stable smooth continuous motion`. Камеру и объективы называть можно."
            )
    elif target == "actionsc":
        lg = ctx.get("long") or {}
        secs = lg.get("secs") or []
        mode_rules = (
            "Режим приложения: ЭКШН-ВИДЕОСТУДИЯ (бой, погоня, бег, прыжки, драка, стрельба; сцены идут одна за другой, как в Видеостудии, "
            "но с добавками для быстрого движения). Ролик состоит из сцен, длина сцены от 3 до 15 секунд, для экшна лучше "
            "3-5 секунд (так считается быстрее и движение чище), сцен не больше 12"
            + (f"; сейчас в разделе {len(secs)} сцен: {', '.join(str(int(x)) for x in secs)} с" if secs else "")
            + ". Если пользователь не назвал число сцен или общую длину, предложи сам по описанию (обычно 3-6 сцен по 3-5 с). "
            "Если пользователь назвал общую длину («ролик на 20 секунд»), раздели её на сцены по 3-5 секунд так, чтобы сумма сошлась.\n"
            "ГЛАВНОЕ: каждая следующая сцена продолжает прошлую, но модель видит только последнюю секунду прошлой сцены и "
            "референс-картинки. Текст соседних сцен она НЕ видит. Поэтому:\n"
            "- РАЗБИВКА: распиши историю по порядку и по времени, как раскадровку: в заголовке каждой сцены её длина в секундах, а в "
            "тексте сцены, что происходит в ЭТОЙ части ролика (начало, нарастание, кульминация, развязка). Сцена N должна начинаться "
            "с того, чем закончилась сцена N-1 (то же место, поза, направление движения), и приводить к началу сцены N+1;\n"
            "- в КАЖДОЙ сцене коротко заново назови героя и место, ссылаясь на референс-картинки как <Picture 1>, <Picture 2> "
            f"(приложено картинок: {n_img}, можно до 5; пользователь сам решает, что на какой)"
            + ("; КАРТИНОК НЕТ: теги <Picture N> не используй, а в каждой сцене словами опиши героя, одежду и место одинаково" if not n_img else "")
            + "; внешность подробно не расписывай, её даёт картинка;\n"
            "- Роли картинок опиши во вступлении (см. ФОРМАТ ОТВЕТА), например `<Picture 1> is the character reference (keep this "
            "face exactly). <Picture 2> is the location reference.` Теги ставишь ты, пользователь их не пишет; если роли не названы, "
            "угадай и в конце одной строкой скажи, в каком порядке класть картинки;\n"
            "- ЭКШН: одно главное быстрое действие на сцену (рывок, прыжок, удар, поворот, падение), нарастание от сцены к сцене. "
            "Движение называй конкретно и с темпом: sprints, vaults, ducks, spins. Камеру пиши отдельной фразой и с темпом "
            "(fast tracking shot beside her, low-angle whip pan, quick pull back), без резкой смены места внутри сцены;\n"
            "- ЕДИНЫЙ МИР: в каждой сцене повторяй одни и те же слова про героя (имя, одежда, причёска), место, время суток, погоду, "
            "свет и цвет. Меняй только действие, ракурс и крупность;\n"
            "- смена места или плана (другая локация, крупный план вместо общего): начинай шот словами `a new plan:` сразу после "
            "времени (`[Shot 3] At 00:06.000, a new plan: ...`). Такая сцена строится заново по референсам, качество как в первой. "
            "Без `a new plan` оставляй только шоты, где действие идёт одним кадром, и не больше 3-4 подряд;\n"
            "- ЗВУК: своего аудио нет, звук делает модель по тексту. В конце каждой сцены одной короткой фразой напиши звук "
            "(шаги, дождь, крики толпы, музыка за кадром). Реплики героев, если нужны, в <d>[Язык] ...</d>, коротко, одна на 2-3 секунды.\n"
            "ФОРМАТ ОТВЕТА: ровно ОДИН блок prompt в ПОЛНОМ формате MiniMax H3 (как в скилле «hailuo-h3»), ЦЕЛЫМ роликом: 1) "
            "вступление про картинки: `How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns "
            "with the 0.00-second mark of the target video.` (по одной фразе на каждую картинку); 2) `integrated_multimodal_description: "
            "[Shot 1] ...`, затем `[Shot 2] At 00:03.000, ...`, `[Shot 3] At 00:06.000, ...` и так далее: шоты по 3-5 секунд подряд, "
            "время каждого шота начинается там, где закончился прошлый (шот 2 с 3-й секунды, шот 3 с 6-й и т.д.), в шоте: действие, "
            "камера, звук; 3) в конце `overall_soundscape: ...` и `non_diegetic_music: ...`. Приложение само разрежет это на сцены "
            "по шотам (по ползункам времени) и в КАЖДУЮ сцену вставит вступление, свой шот с его номером и временем, звук и музыку. "
            "Не дроби на SCENE, не пиши ничего вне этого формата. Пример:\n"
            "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of "
            "the target video.\n\n"
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, tracking shot. A woman <Picture 1> in a long beige coat "
            "sprints through a rainy neon night market, knocking over crates. Splashing water, shouting crowd.\n\n"
            "[Shot 2] At 00:03.000, the woman <Picture 1> vaults over a market stall and keeps running into a narrow alley, a pursuer's "
            "shadow behind her. Low fast camera following her. Rain, echoing footsteps.\n\n"
            "overall_soundscape: Heavy rain, shouting crowd, running footsteps.\n\n"
            "non_diegetic_music: A tense driving percussion theme.")
    elif target == "icon":
        mode_rules = (
            "Режим приложения: АНИМИРОВАННАЯ ИКОНКА или СТИКЕР (MiniMax H3 по одной картинке, потом из видео вырезается фон). Пользователь загружает "
            "картинку персонажа (<Picture 1>) и своими словами пишет, что персонаж должен делать. Картинка станет и ПЕРВЫМ, и ПОСЛЕДНИМ кадром видео"
            + (f" (длина {dur:.0f} с)" if dur else "") + ": анимация зациклена. Ты пишешь мастер-промпт ровно в таком виде, как работает у пользователя:\n"
            "1. Первая строка: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` Затем пустая строка.\n"
            "2. `integrated_multimodal_description: [Shot 1]` и ОДИН связный абзац по порядку:\n"
            "   а) стиль: `<вид графики по картинке: 2D cartoon sticker / 3D cartoon mobile game character / pixel art / anime…> animation, exactly the style of <Picture 1>`; "
            "если у картинки белая кайма стикера или контур, упомяни его;\n"
            "   б) фон: `the background is <цвет и вид фона С КАРТИНКИ, например plain flat solid green / plain white> and stays identical to <Picture 1> in every frame` "
            "(посмотри на картинку и назови фон как есть; ничего про сцену, землю, тень, небо, траву, дома и деревья не добавляй);\n"
            "   в) камера: `Static Shot, the camera does not move, no zoom, no pan, no cuts, the framing, the size and the position of the character stay identical to <Picture 1> for the whole clip`;\n"
            "   г) действие: персонаж (назови его коротко по картинке, внешность не пересказывай) делает то, что просил пользователь. Расписывай движение по секундам "
            "(`From 00:00.000 …`, `At 00:01.500 …`) ПО ВСЕЙ ДЛИНЕ клипа: движение идёт до самого конца, без долгого стояния без дела. Двигаются только названные части тела, "
            "остальное неподвижно, ноги и основание остаются на месте. Скорость плавная, мультяшные движения со squash and stretch, если стиль мультяшный;\n"
            "   д) конец: `Then he/she/it returns to the exact pose of <Picture 1>, holding it with a slight breathing movement`, чтобы анимация зацикливалась без шва;\n"
            "   е) последняя фраза-запрет в утвердительной форме: `All movement is smooth, no new objects appear, no scenery, no background change, no light rays, no particles, "
            "no camera effects, the outfit, props, fur and colors keep the same shape and color.`\n"
            "3. `overall_soundscape:` одна короткая строка со звуком действия (взмах, шаги, шелест ткани) или `N/A`.\n"
            "4. `non_diegetic_music: N/A`.\n"
            "Если пользователь не сказал, что персонаж должен делать, задай ОДИН короткий вопрос. Если сказал слишком коротко (например «бьёт мечом»), сам придумай аккуратное "
            "понятное движение. Не придумывай ничего вне кадра картинки, не добавляй камеру, наезды, фон, эффекты, других персонажей, реплики и текст. "
            "ВАЖНО: модель рисует всё, что названо в тексте, даже в сравнении. Поэтому никогда не пиши предметы и места, которых нет на картинке (наковальня, земля, пол, камни, пыль, искры, дым, трава, стена), "
            "и сравнения вроде «как по наковальне», «как по барабану». Если пользователь назвал такой предмет, опиши только движение персонажа без него. Удар, прыжок или шаг "
            "описывай как движение тела и предмета в руке в пустом пространстве. "
            "Промпт только английский, одним блоком prompt.")
    elif target == "swap":
        mode_rules = (
            "Режим приложения: ЗАМЕНА ПЕРСОНАЖА (MiniMax H3 Ref2V + LoRA замены). Пользователь загружает видео (<Video 1>), в котором "
            "нужно заменить человека, и одну-две картинки новых героев (<Picture 1>, <Picture 2>, по порядку слотов). Модель "
            "берёт из видео позу, движения, камеру, фон и звук, а лицо, причёску и одежду - с картинок. Текст сцены сочинять НЕ нужно.\n"
            "- Пользователь своими словами говорит, КОГО в видео заменить (по одежде, положению, возрасту: «старик в пиджаке», "
            "«девушка слева») и какой картинкой. Если он не уточнил и героев в кадре один - пиши «the main person».\n"
            "- Формат ответа - один короткий английский блок prompt простыми предложениями, по одному на каждого заменяемого: "
            "`Replace the <описание человека> in <Video 1> with the character from <Picture N>.` Если картинок две, предложений два "
            "и у каждого своя картинка.\n"
            "- Затем всегда добавляй хвост: `Retain the identity, attire, and style of the character(s) from the picture(s). "
            "Preserve the camera angle, background, lighting, objects, and all other figures from the original video. Ensure the "
            "replacement aligns with the original figure in terms of position, scale, pose, and movement. Do not add any other "
            "figures, copies or people, and keep the background of <Video 1>. Do not display the reference images or their backgrounds.`\n"
            "- Если в сообщении написано, что приложены кадры из видео (последние N картинок, с секундами), это кадры <Video 1>, а НЕ "
            "картинки героев и не Picture N. Посмотри по ним, кто есть в кадре (одежда, возраст, положение, смена плана), и по этому "
            "опиши, кого заменить. Если люди в кадре отличаются в разных секундах (смена плана), так и напиши: «в первой части … , после "
            "смены плана на N-й секунде …». Не придумывай то, чего на кадрах нет.\n"
            "- Если пользователь просит что-то сохранить или добавить (шарф, цвет волос), допиши это одной фразой в конце.\n"
            "- Внешность героев с картинок словами не описывай. Не выдумывай реплики и действия: видео уже есть. Одной строкой в конце "
            "напомни: видео - в слот видео, картинки - по порядку героев.")
    elif target in ("refs", "action"):
        mode_rules = (
            ("Режим приложения: ЭКШН-ВИДЕО, ОДНО ВИДЕО (Hybrid FL2VA/Ref2VA, сэмплинг 8 шагов в низком разрешении + 4 в высоком): "
             "бой, погоня, бег, прыжки, стрельба. Пиши энергичное действие по нарастанию, по 1 ключевому движению на шот, "
             "быстрые, но мотивированные движения камеры (tracking, pull out, truck) с указанием скорости. "
             if target == "action" else
             "Режим приложения: РЕФЕРЕНСЫ (Singularity Ref2V, один проход, 4-8 шагов). ")
            + f"Видео строится по референсам Omni-Reference: картинки (персонажи, локация, предметы), может быть "
            f"референс-видео (движение/камера) и референс-аудио (голос/ритм). Длительность {dur or 6} с - таймкоды "
            "шотов не выходят за неё.\n"
            "- Пользователь своими словами говорит, что приложил и зачем (например: референс 1 - персонаж, референс 2 - одежда, "
            "видео - взять движение камеры, аудио - звук фона). Номера считай ОТДЕЛЬНО для каждого типа в порядке перечисления: "
            "картинки <Picture 1>, <Picture 2>…, видео <Video 1>…, аудио <Audio 1>… Пользователь теги сам не пишет - "
            "расставляешь их ты, и они ОБЯЗАТЕЛЬНО должны быть в промпте (так их ждёт модель).\n"
            "- Первая строка: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
            "referenced. <Picture 2> (from [Shot 1]) is fully referenced.` - по одному предложению на КАЖДУЮ картинку (если "
            "картинок две, предложений два, и т.д.); затем одна фраза с ролью КАЖДОГО файла, например `<Picture 1> is the character "
            "reference (keep this face exactly). <Picture 2> is the clothing reference. <Video 1> is a camera-movement reference "
            "only, take no faces or people from it. <Audio 1> is the background sound reference.` Одна роль на файл.\n"
            "- Если есть и картинка с лицом, и видео: лица и внешность берутся ТОЛЬКО с картинки, видео задаёт лишь то, что "
            "назвал пользователь (движение, камеру, место). Напиши это явно.\n"
            "- Дальше в тексте сцены ссылайся на файлы теми же тегами (<Picture 1>, <Video 1>). Внешность героя с картинки не "
            "описывай словами. В конце ответа одной строкой напомни пользователю, в каком порядке класть файлы в слоты.\n"
            "- Пиши примерно одну реплику на 2,5 секунды, если в сцене есть речь.")
    else:
        if n_img >= 2:
            align = "есть первый и последний кадр - используй строку выравнивания FL2VA"
        elif n_img == 1:
            align = "есть стартовый кадр - используй строку выравнивания I2VA"
        else:
            align = "картинок нет - T2VA, без строки выравнивания"
        mode_rules = (f"Режим приложения: ГЕНЕРАЦИЯ ВИДЕО (FastH3: текст / первый кадр / первый и последний кадр). "
                      f"Сейчас {align}. Длительность клипа {dur or 5} с - таймкоды шотов не выходят за неё.")
    if target in ("krea", "plist", "fluxt", "qwent", "h3t"):
        mode_rules += (
            "\n\nГЛАВНОЕ: рисуй РОВНО то, что описал пользователь (его герой, животное, предмет, место, стиль). Примеры внутри скилла показывают только ФОРМАТ ответа; их содержимое (девушка, портрет, кроссовки, афиша) НЕЛЬЗЯ переносить в ответ. Не меняй героя пользователя на человека и тем более на девушку. Если героя он не называл как человека, людей в кадре нет. Не добавляй людей, которых он не просил. Перед выдачей сверь: главный объект промпта совпадает с тем, что написал пользователь.")
    if target in ("krea", "plist", "fluxt", "qwent", "h3t") and not n_img:
        mode_rules += (
            "\n\nНЕТ ПРИЛОЖЕННЫХ КАРТИНОК, И ОНИ НЕ НУЖНЫ: картинка рисуется только по тексту. Сразу напиши готовый промпт по описанию "
            "пользователя, пусть оно и короткое (например «волк в хвойном лесу под луной»): сам выбери стиль, ракурс, свет, цвет и "
            "композицию. Не проси прислать фото или референс, не пиши «пришлите», «я скину», «загрузите», не задавай уточняющих "
            "вопросов. Ответ: одна строка подсказки по формату кадра и блок prompt.")
    if ctx.get("model") == "h3img" and target in ("story", "sheet", "sheet6"):
        # the user chose MiniMax H3 Image for this storyboard / character sheet: it overrides the model words in the rules above
        mode_rules += (
            "\n\nМОДЕЛЬ ЭТОГО ЗАПРОСА — MiniMax H3 Image (важнее любых упоминаний Krea 2, FLUX.2 и Qwen выше). Применяй скилл "
            "«minimax-h3-image» вместе с остальными скиллами:\n"
            "- Промпт английский, подробный, связным описанием всего кадра; негатива нет. Для раскадровки и листа структуру "
            "(панели, кадры, сетка, без текста и подписей) бери из скилла «storyboard» / «Карта персонажа», а язык описания и "
            "порядок слов из «minimax-h3-image».\n"
            "- Картинки из чата называй <Picture 1>, <Picture 2>… (до 5), а НЕ <image1> и НЕ \"image 1\". Герой с картинки: \"the "
            "character from <Picture 1>, keeping the face, hairstyle and outfit exactly\"; стиль: \"in the style of <Picture 1>\"; "
            "локация: \"in the location from <Picture 2>\".\n"
            "- В режимах «По кадрам» и «Одним листом» картинки из чата подгружаются как референсы (Picture N); в листе персонажа "
            "по описанию (текст) картинки не подгружаются, поэтому опиши героя словами. Если нужен лист ПО ГОТОВОЙ картинке, "
            "одной строкой подскажи «Правка изображения» → «Лист персонажа» → модель MiniMax H3 Image.\n"
            "- Каждый кадр и лист рисуются крупно (около 4 МП), поэтому описывай детали, которые будут видны (одежда, материалы, "
            "свет).")
    rm_rules = refmode_rules(ctx.get("refmodes"), target)   # cards of Reference Mode chosen in the chat (video studios only)
    if rm_rules:
        if target in VIDEO_RM_TARGETS:
            # the heroes come from Reference Mode cards: they must be named @name, never <Picture N> (the rules below are for the user's own pictures)
            mode_rules = ("ГЛАВНОЕ ПРАВИЛО ДЛЯ ЭТОГО ЗАПРОСА: герои выбраны в Reference Mode. Везде, где ниже сказано назвать героя/персонажа/"
                          "предмет/локацию через <Picture N> и в примерах с <Picture 1>, для героев из списка ниже ПИШИ @имя (например `@Katerina stands ...`). "
                          "НЕ пиши `<Picture 1> is the character reference`, НЕ пиши роль картинок для героев из списка и НЕ ставь на них теги <Picture N>: "
                          "приложение само заменит @имя на описание, подаст изображения и голос карточки. Теги <Picture N> используй только для "
                          "картинок, которые пользователь сам загрузил в раздел как свои референсы (если их нет, тегов не будет вообще). "
                          "В КАЖДОЙ сцене называй героя как @имя заново.\n\n" + rm_rules + "\n\n" + mode_rules)
        else:
            mode_rules += "\n\n" + rm_rules
    intro = ("Ты - агент-художник внутри приложения Minimax H3 Workspace. Пользователь описывает картинку или правку, может приложить "
             "картинки. Ты пишешь готовый промпт для модели картинок по правилам скилла ниже."
             if target in IMAGE_SKILLS else
             "Ты - агент-режиссёр внутри приложения Minimax H3 Workspace. Пользователь описывает сцену, сценарий, персонажей, "
             "может приложить картинки. Ты пишешь готовый промпт для локальной видеомодели MiniMax H3 по правилам скиллов ниже.")
    return f"""{intro}

ПРАВИЛА ПРИЛОЖЕНИЯ (важнее указаний скиллов о формате ответа):
1. Общайся на языке пользователя, коротко и по делу. Если не хватает критичного (что происходит, кто говорит, текст реплик) - задай 1-2 конкретных вопроса. Иначе сразу пиши промпт.
2. Итоговый промпт выдай РОВНО ОДНИМ блоком кода с меткой prompt:
```prompt
...промпт...
```
Промпт на английском, реплики внутри <d>[Язык] ...</d> - на языке оригинала. Не дублируй промпт вторым блоком на русском, если пользователь сам не попросил. Перед блоком - максимум 2-3 строки (допущения, если что-то придумал).
3. Промпт не длиннее 7000 символов. Без синтаксиса весов, без имён знаменитостей и чужих персонажей.
4. Картинки, приложенные к чату, - это Picture 1, Picture 2... в порядке прикрепления (сейчас приложено: {n_img}). Внешность с картинок подробно не расписывай - модель видит их сама; бери только позу, взгляд, предметы, свет.
5. Скиллы могут ссылаться на файлы references/... - все нужные справочники уже включены ниже, файлы открывать не нужно.

{mode_rules}

{skills_text}
"""


def fits(model, vram_gb):
    """A model of this file size runs comfortably when it takes at most ~60% of video memory
    (the rest goes to the context, which the skills need to be large)."""
    return not vram_gb or Path(model).stat().st_size <= vram_gb * 1024 ** 3 * 0.6


def find_gguf(folder, prefer=None, vram_gb=None):
    """Pick a chat model and its vision projector from a folder the user already has.

    Returns (model, mmproj_or_None, all_models). Draft/MTP helper files are skipped.
    With no explicit choice, a Gemma that fits the user's video memory wins, never simply the biggest file.
    """
    folder = Path(folder)
    files = [p for p in folder.rglob("*.gguf") if len(p.relative_to(folder).parts) <= 3]
    mmproj = [p for p in files if "mmproj" in p.name.lower()]
    models = [p for p in files if p not in mmproj and not p.name.lower().startswith(("mtp-", "dflash-"))]
    if not models:
        return None, None, []
    models.sort(key=lambda p: p.name.lower())
    model = next((p for p in models if prefer and str(p) == str(prefer)), None)
    if model is None:
        ok = [p for p in models if fits(p, vram_gb)]
        pool = ok or models
        # among suitable files prefer Gemma, then the largest that still fits; if nothing fits, the smallest
        pool.sort(key=lambda p: ("gemma" not in p.name.lower(), -p.stat().st_size if ok else p.stat().st_size))
        model = pool[0]
    # a projector from the same folder belongs to the same model most of the time
    proj = sorted(mmproj, key=lambda p: (p.parent != model.parent, p.stat().st_size))
    return model, (proj[0] if proj else None), models


def explain_failure(log):
    """Turn a llama.cpp start-up log into one sentence a normal user can act on."""
    low = log.lower()
    if "outofdevicememory" in low or "failed to allocate" in low or "out of memory" in low:
        return ("Не хватило видеопамяти: эта модель слишком большая для вашей видеокарты. "
                "Выберите модель поменьше, например Gemma 4 12B в сжатии Q4 (около 7 ГБ), "
                "или вернитесь к встроенной Gemma кнопкой «Вернуть встроенную Gemma».")
    if "unknown model architecture" in low or "unsupported" in low:
        return "Эта модель не поддерживается движком llama.cpp. Выберите другую модель (лучше Gemma 4 в формате GGUF)."
    if "failed to load model" in low or "error loading model" in low:
        return "Не удалось открыть файл модели: возможно, он скачан не до конца или повреждён."
    return None


class LlamaServer:
    def __init__(self, root, log_path):
        self.root = Path(root)
        self.log_path = log_path
        self.proc = None
        self.port = None
        self.variant = None
        self.custom = None  # {"model": path, "mmproj": path|None} when the user points to their own GGUF

    def exe(self):
        hits = list((self.root / "llama.cpp").rglob("llama-server.exe")) if (self.root / "llama.cpp").exists() else []
        return hits[0] if hits else None

    def model_paths(self, variant):
        if self.custom:
            return Path(self.custom["model"]), Path(self.custom["mmproj"]) if self.custom.get("mmproj") else None
        v = LLM_VARIANTS[variant]
        return variant_files(self.root, v)

    def has_model(self, variant):
        m, p = self.model_paths(variant)
        return m.exists() and (p is None or p.exists())

    def installed(self, variant):
        return bool(self.exe()) and self.has_model(variant)

    def vision(self, variant):
        return self.model_paths(variant)[1] is not None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, variant):
        key = (variant, json.dumps(self.custom, sort_keys=True))
        if self.alive() and self.variant == key:
            return
        self.stop()
        m, p = self.model_paths(variant)
        self.port = engine.free_port(8280)
        self.variant = key
        # -ngl is left unset on purpose: llama.cpp then fits the layers to free video memory by itself
        # (heavier models run partly on the CPU, slower but they start); the context stays fixed
        # because the skills need about 20k tokens
        args = [str(self.exe()), "-m", str(m), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(LLM_VARIANTS[variant]["ctx"]), "-fit", "on", "-fitt", "1536", "--jinja", "-np", "1"]
        if p:
            args += ["--mmproj", str(p)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(args, cwd=self.exe().parent, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=engine.CREATE_NO_WINDOW)

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def log_tail(self, n=12):
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""


async def install(installer, root, variant, runtime_only=False, whisper_dir=None):
    """Download llama.cpp (Vulkan build, works on any NVIDIA/AMD card) and, unless the user has their own, Gemma."""
    root = Path(root)
    llama_dir = root / "llama.cpp"
    if not (llama_dir.exists() and list(llama_dir.rglob("llama-server.exe"))):
        zpath = root / "downloads" / f"llama-{LLAMA_BUILD}.zip"
        await installer.download(LLAMA_URL, zpath, "agent", "llama.cpp", 37e6)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(llama_dir)
        zpath.unlink(missing_ok=True)
    import speech
    await speech.install(installer, root, whisper_dir)  # words from audio + voice input (~200 MB)
    if runtime_only:
        return
    v = LLM_VARIANTS[variant]
    for key, size in (("mmproj", v["mmproj_size"]), ("model", v["model_size"])):
        dest = root / "models" / v.get(key + "_as", v[key])
        await installer.download(HF.format(repo=v["repo"], path=v[key]), dest, "agent", f'{v["title"]}: {v[key]}', size)


async def wait_ready(session, server, timeout=240):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if not server.alive():
            log = server.log_tail(80)
            raise RuntimeError(explain_failure(log) or "Модель не запустилась:\n" + server.log_tail(6))
        try:
            async with session.get(server.url + "/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("Модель долго загружается:\n" + server.log_tail())


def to_openai(messages):
    out = []
    for m in messages:
        if m.get("images") and m["role"] == "user":
            content = [{"type": "text", "text": m.get("content", "")}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in m["images"]]
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


def chat_text(messages):
    return "\n".join(m.get("content", "") for m in messages if m["role"] == "user")


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
