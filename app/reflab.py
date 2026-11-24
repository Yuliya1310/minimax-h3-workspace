"""Reference Lab: sheets of references of a character, an object or a place from ONE short video.

The pictures go to MiniMax H3 (reference to video, the same graph as the «Референсы» page); the camera walks through the chosen
angles ("beats"); the browser cuts the frames at the beat moments and lays the sheet out. This module keeps what the server has to know:
the ready-made sheet types with their camera texts, the prompt, the user's own presets, and the history of the lab's works
(frames + sheet files in data/reflab/runs/<id>/).
"""
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from aiohttp import web

TYPES = {"character": "Персонаж", "object": "Объект", "location": "Локация"}
MAX_UPLOAD = 40 * 1024 * 1024
KEEP_RUNS = 80        # the newest not-starred works kept on disk
SECONDS = 8.0

# beat: [short name, camera text, moment in the video as a share 0..1]
P = lambda title, hint, secs, beats: {"title": title, "hint": hint, "seconds": secs,
                                      "beats": [{"name": n, "text": t, "at": a} for n, t, a in beats]}

PRESETS = {
    "location": {
        "coverage": P("Со всех сторон", "общий · с обратной стороны · слева · справа · деталь", 8, [
            ("общий", "wide establishing shot of the whole place, eye level, the camera is still", 0.0),
            ("обратно", "the camera has turned around and now looks at the place from the opposite side, reverse shot", 0.3),
            ("слева", "the camera is moving along the left side of the place, left side view", 0.55),
            ("справа", "the camera is moving along the right side of the place, right side view", 0.8),
            ("деталь", "slow push in to a close-up of one characteristic detail of the place", 1.0)]),
        "pan360": P("Осмотр на месте 360°", "камера стоит и поворачивается вокруг", 8, [
            ("0°", "the camera stands still and looks straight ahead", 0.0),
            ("90°", "the camera has turned in place a quarter of a full turn to the right", 0.25),
            ("180°", "the camera has turned half a turn and looks at the opposite side", 0.5),
            ("270°", "the camera has turned three quarters of a full turn", 0.75),
            ("360°", "the camera has finished the full turn and looks straight ahead again", 1.0)]),
        "walk": P("Пройтись по месту", "камера идёт вперёд по месту", 8, [
            ("вход", "wide shot at the entrance of the place, the camera starts to walk forward", 0.0),
            ("вперёд", "the camera keeps walking forward at walking speed, the place opens up around", 0.35),
            ("середина", "the camera is in the middle of the place and slowly looks around", 0.7),
            ("вглубь", "the camera has reached the far side of the place and looks back", 1.0)]),
        "time": P("Время суток", "то же место утром, днём, вечером, ночью", 8, [
            ("рассвет", "the same view at dawn, soft pink light, long shadows", 0.0),
            ("день", "the same view at noon, bright sun, short shadows", 0.3),
            ("закат", "the same view at golden hour, warm low light", 0.6),
            ("ночь", "the same view at night, artificial lights and moonlight", 1.0)]),
        "empty": P("Пустая сцена", "место без людей, спокойный облёт", 6, [
            ("начало", "empty place without any people, the camera is still", 0.0),
            ("сдвиг", "the camera drifts slowly to the right, still no people", 0.5),
            ("конец", "the camera ends on the opposite corner of the empty place", 1.0)]),
        "single": P("Один дубль", "один кадр места", 5, [("кадр", "one steady wide shot of the place", 0.5)]),
        "custom": P("Свой", "свои ракурсы", 8, [("ракурс 1", "wide shot of the place", 0.0)]),
    },
    "character": {
        "face": P("Поворот лица", "левый профиль · анфас · правый профиль", 6, [
            ("слева", "head and shoulders close-up, the face turned fully to the left in profile", 0.0),
            ("анфас", "head and shoulders close-up, the head slowly turns to look straight into the camera, front view", 0.5),
            ("справа", "head and shoulders close-up, the head keeps turning and ends fully to the right in profile", 1.0)]),
        "orbit360": P("Поворот 360°", "камера облетает героя в полный рост · 5 кадров: спереди, 3/4, бок, сзади, другой бок", 12, [
            ("спереди", "full body front view, the person stands completely still in a relaxed neutral pose, the camera starts one slow smooth orbit around them", 0.0),
            ("3/4", "the camera has orbited 45 degrees around the standing person, three-quarter view, full body from head to feet in the frame", 45 / 360),
            ("бок", "the camera has orbited 90 degrees around the standing person, side profile view, full body from head to feet in the frame", 90 / 360),
            ("сзади", "the camera has orbited 180 degrees around the standing person, back view, full body from head to feet in the frame", 180 / 360),
            ("другой бок", "the camera has orbited 270 degrees around the standing person, side profile view from the other side, full body from head to feet in the frame", 270 / 360)]),
        "single": P("Один дубль", "один кадр героя", 5, [("кадр", "one steady full body shot of the person facing the camera", 0.5)]),
        "custom": P("Свой", "свои ракурсы", 8, [("ракурс 1", "full body front view", 0.0)]),
    },
    "object": {
        "table": P("Поворот на столе", "предмет медленно крутится на месте", 8, [
            ("спереди", "the object stands on a plain surface seen from the front", 0.0),
            ("бок", "the object has turned a quarter turn, side view", 0.25),
            ("сзади", "the object has turned half a turn, seen from behind", 0.5),
            ("другой бок", "the object has turned three quarters, the other side view", 0.75),
            ("сверху", "the camera rises and looks at the object from above", 1.0)]),
        "detail": P("Детали", "общий вид и крупные планы", 8, [
            ("общий", "the whole object in the frame, still camera", 0.0),
            ("деталь 1", "slow push in to a close-up of the most characteristic detail of the object", 0.4),
            ("деталь 2", "the camera moves along the object to another detail, close-up", 0.75),
            ("фактура", "extreme close-up of the surface and the material of the object", 1.0)]),
        "single": P("Один дубль", "один кадр предмета", 5, [("кадр", "one steady shot of the object", 0.5)]),
        "custom": P("Свой", "свои ракурсы", 8, [("ракурс 1", "the whole object seen from the front", 0.0)]),
    },
}
RU = {   # the same beats, told in plain Russian (shown to the user; the English text is what the model reads)
    ("location", "coverage"): ["Общий план всего места, камера стоит", "Камера развернулась и смотрит на место с противоположной стороны",
                               "Камера идёт вдоль левой стороны", "Камера идёт вдоль правой стороны", "Камера медленно приближается к одной характерной детали"],
    ("location", "pan360"): ["Камера стоит и смотрит прямо", "Повернулась на месте на четверть оборота вправо", "Повернулась на пол-оборота, смотрит в обратную сторону",
                             "Повернулась на три четверти оборота", "Сделала полный оборот и снова смотрит прямо"],
    ("location", "walk"): ["Общий план у входа, камера начинает идти вперёд", "Камера идёт вперёд обычным шагом, место раскрывается", "Камера в середине места, медленно оглядывается",
                           "Камера дошла до дальней стороны и смотрит назад"],
    ("location", "time"): ["Тот же вид на рассвете, мягкий розовый свет", "Тот же вид в полдень, яркое солнце", "Тот же вид на закате, тёплый низкий свет", "Тот же вид ночью, огни и лунный свет"],
    ("location", "empty"): ["Пустое место без людей, камера стоит", "Камера плавно сдвигается вправо, людей по-прежнему нет", "Камера оказалась в противоположном углу пустого места"],
    ("location", "single"): ["Один спокойный общий кадр места"], ("location", "custom"): ["Общий план места"],
    ("character", "face"): ["Крупный план головы, лицо повёрнуто влево, профиль", "Голова плавно поворачивается и смотрит прямо в камеру", "Голова продолжает поворот и оказывается в профиль вправо"],
    ("character", "orbit360"): ["Спереди в полный рост, герой стоит спокойно, камера начинает плавный облёт",
                                "Камера обошла героя на 45°: вид в три четверти, он виден целиком, с головы до ног",
                                "Камера обошла героя на 90°: вид сбоку, он виден целиком, с головы до ног",
                                "Камера обошла героя на 180°: вид сзади, он виден целиком, с головы до ног",
                                "Камера обошла героя на 270°: вид с другого бока, он виден целиком, с головы до ног"],
    ("character", "single"): ["Один спокойный кадр героя в полный рост"], ("character", "custom"): ["В полный рост спереди"],
    ("object", "table"): ["Предмет стоит на ровной поверхности, вид спереди", "Повернулся на четверть оборота, вид сбоку", "Повернулся на пол-оборота, вид сзади",
                          "Повернулся на три четверти, вид с другого бока", "Камера поднимается и смотрит на предмет сверху"],
    ("object", "detail"): ["Весь предмет в кадре, камера стоит", "Камера медленно приближается к самой характерной детали", "Камера идёт вдоль предмета к другой детали, крупный план",
                           "Самый крупный план: поверхность и материал"],
    ("object", "single"): ["Один спокойный кадр предмета"], ("object", "custom"): ["Весь предмет спереди"],
}
for (_t, _k), _lines in RU.items():
    for _b, _r in zip(PRESETS[_t][_k]["beats"], _lines):
        _b["ru"] = _r
for _t in PRESETS.values():   # the share is a position inside the clip: the last beat sits a little before the end
    for _p in _t.values():
        for _b in _p["beats"]:
            _b["at"] = round(min(0.97, max(0.0, _b["at"])), 3)


def _err(text, status=400):
    return web.json_response({"error": text}, status=status)


def clean(v, limit):
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


def build_prompt(p):
    """The English text for H3: one continuous take; the camera goes through the beats in time."""
    typ = p.get("type") if p.get("type") in TYPES else "character"
    seconds = max(3.0, min(15.0, float(p.get("seconds") or SECONDS)))
    who = clean(p.get("who"), 300) or {"character": "the person", "object": "the object", "location": "the place"}[typ]
    beats = [b for b in (p.get("beats") or []) if isinstance(b, dict) and clean(b.get("text"), 300)][:12]
    if not beats:
        beats = PRESETS[typ]["single"]["beats"]
    head = {"character": f"A character reference clip of {who}. The person stays calm and does not talk, the face and outfit stay exactly the same.",
            "object": f"A product reference clip of {who}. The object does not change its shape or colours.",
            "location": f"A location reference clip of {who}. No people are in the frame, nothing important moves in the place."}[typ]
    parts = [head]
    if clean(p.get("features"), 1200):
        parts.append("Keep: " + clean(p.get("features"), 1200).rstrip(".") + ".")
    if clean(p.get("style"), 200):
        parts.append("Look: " + clean(p.get("style"), 200).rstrip(".") + ".")
    if clean(p.get("setdress"), 300):
        parts.append("Setting: " + clean(p.get("setdress"), 300).rstrip(".") + ".")
    if clean(p.get("focus"), 200):
        parts.append("Every view is centred on " + clean(p.get("focus"), 200).rstrip(".") + ".")
    if any("orbit" in clean(b.get("text"), 300) for b in beats):   # the full-turn sheet: the hero stands still, the camera goes round him
        parts.append("The person does not move at all and the camera makes one smooth full 360-degree orbit around them at a constant distance; the whole body from head to feet stays in the frame.")
    texts = " ".join(clean(b.get("text"), 300).lower() for b in beats)
    close = typ == "character" and ("close-up" in texts or "head and shoulders" in texts) and "full body" not in texts and "orbit" not in texts
    if typ != "location":   # a flat one-colour backdrop: no floor, no horizon, no shadows, so the figure can be cut out cleanly
        parts.append("The background is one flat seamless light grey colour everywhere, like a cut-out on a plain sheet: no floor, no floor line, no horizon, "
                     "no gradient, no vignette, no backdrop edges, no cast shadows, no ground shadow, no reflections, no props. "
                     "The subject is lit evenly from the front. ")
        if close:   # a face sheet: every view is a tight close-up, even when the reference picture shows the whole body
            parts.append("The framing is a tight head-and-shoulders close-up in every moment of the clip: the face is large and fills most of the frame, "
                         "the shoulders are cut by the lower edge of the frame, the body below the chest is never shown, even if the reference picture shows the whole body.")
        else:
            parts.append("The subject fills most of the frame height, with only a small margin above and below.")
    parts.append("One continuous take, no cuts, steady and clean camera, "
                 + ("" if typ == "location" else "flat even light, ") + "no text, no music, no dialogue.")
    moves = []
    for b in beats:
        at = max(0.0, min(1.0, float(b.get("at") if b.get("at") is not None else 0)))
        moves.append(f"At {at * seconds:.1f} seconds: {clean(b.get('text'), 300).rstrip('.')}.")
    parts.append("Camera, in time order. " + " ".join(moves))
    return " ".join(parts), seconds


def beat_times(beats, seconds):
    return [round(max(0.0, min(1.0, float(b.get("at") or 0))) * seconds, 2) for b in beats]


class Store:
    """History of the lab's works and the user's own sheet types (presets)."""

    def __init__(self, root):
        self.root = Path(root)
        self.rdir = self.root / "runs"
        self.rdir.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "runs.json"
        self.pfile = self.root / "presets.json"
        self.runs = self._read(self.index)
        self.presets = self._read(self.pfile)

    @staticmethod
    def _read(f):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return d if isinstance(d, list) else []
        except (OSError, ValueError):
            return []

    @staticmethod
    def _write(f, data):
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(f)

    def get(self, rid):
        return next((x for x in self.runs if x["id"] == rid), None)

    def public(self, r):
        base = f"/api/reflab/file/{r['id']}/"
        out = dict(r)
        out["sheet_url"] = base + r["sheet"] if r.get("sheet") else ""
        out["beats"] = [{**b, "url": base + b["file"] if b.get("file") else ""} for b in r.get("beats", [])]
        return out

    # ---- routes ----
    async def h_presets(self, _req):
        built = {t: {k: v for k, v in d.items()} for t, d in PRESETS.items()}
        return web.json_response({"types": TYPES, "presets": built, "mine": self.presets})

    async def h_prompt(self, req):
        try:
            p = await req.json()
            prompt, seconds = build_prompt(p)
        except (ValueError, TypeError):
            return _err("Не удалось собрать описание")
        return web.json_response({"prompt": prompt, "seconds": seconds})

    async def h_preset_save(self, req):
        try:
            body = await req.json()
        except ValueError:
            return _err("Не удалось прочитать данные")
        name = clean(body.get("name"), 40)
        typ = body.get("type") if body.get("type") in TYPES else "character"
        beats = [{"name": clean(b.get("name"), 30) or "ракурс", "text": clean(b.get("text"), 300), "ru": clean(b.get("ru"), 200), "at": max(0.0, min(1.0, float(b.get("at") or 0)))}
                 for b in (body.get("beats") or []) if isinstance(b, dict) and clean(b.get("text"), 300)][:12]
        if not name:
            return _err("Напишите название типа листа")
        if not beats:
            return _err("Добавьте хотя бы один ракурс с описанием камеры")
        item = {"id": uuid.uuid4().hex[:8], "name": name, "type": typ, "seconds": max(3.0, min(15.0, float(body.get("seconds") or SECONDS))), "beats": beats}
        self.presets = [x for x in self.presets if not (x["name"].lower() == name.lower() and x["type"] == typ)] + [item]
        self._write(self.pfile, self.presets)
        return web.json_response(item)

    async def h_preset_delete(self, req):
        self.presets = [x for x in self.presets if x["id"] != req.match_info["id"]]
        self._write(self.pfile, self.presets)
        return web.json_response({"ok": True})

    async def h_list(self, _req):
        return web.json_response([self.public(r) for r in self.runs])

    async def h_file(self, req):
        r = self.get(req.match_info["id"])
        name = Path(req.match_info["name"]).name
        f = self.rdir / (r["id"] if r else "-") / name
        if not r or not f.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(f, headers={"Cache-Control": "no-store"})

    async def h_save(self, req):
        """A work: `meta` (json) and files `sheet`, `f_<n>` (PNG frames of the beats). Files that are not sent stay as they were."""
        meta, up = None, {}
        try:
            async for field in await req.multipart():
                if field.name == "meta":
                    meta = json.loads((await field.read()).decode("utf-8"))
                elif field.name and field.filename:
                    data = bytes(await field.read())
                    if len(data) > MAX_UPLOAD:
                        return _err("Файл слишком большой")
                    if data[:8] != b"\x89PNG\r\n\x1a\n":
                        return _err("Нужны картинки PNG")
                    up[field.name] = data
        except (ValueError, UnicodeDecodeError):
            return _err("Не удалось прочитать данные")
        if not isinstance(meta, dict):
            return _err("Не удалось прочитать данные")
        rid = re.sub(r"\W", "", str(meta.get("id") or "")) or uuid.uuid4().hex[:10]
        old = self.get(rid)
        folder = self.rdir / rid
        folder.mkdir(parents=True, exist_ok=True)
        old_files = {b["file"]: b for b in (old or {}).get("beats", []) if b.get("file")}
        beats = []
        for n, b in enumerate((meta.get("beats") or [])[:12]):
            if not isinstance(b, dict):
                continue
            file = Path(str(b.get("file") or "")).name
            key = f"f_{n}"
            if key in up:
                file = f"{uuid.uuid4().hex[:8]}.png"
                (folder / file).write_bytes(up[key])
            elif file not in old_files:
                file = ""
            beats.append({"name": clean(b.get("name"), 30), "text": clean(b.get("text"), 300), "t": float(b.get("t") or 0), "on": bool(b.get("on", True)), "file": file})
        sheet = (old or {}).get("sheet", "")
        if "sheet" in up:
            sheet = f"sheet_{uuid.uuid4().hex[:6]}.png"
            (folder / sheet).write_bytes(up["sheet"])
        keep = {b["file"] for b in beats} | {sheet}
        for f in folder.iterdir():
            if f.name not in keep:
                f.unlink(missing_ok=True)
        run = {"id": rid, "name": clean(meta.get("name"), 60) or "Без названия", "type": meta.get("type") if meta.get("type") in TYPES else "character",
               "preset": clean(meta.get("preset"), 60), "status": "finished" if meta.get("status") == "finished" else "draft",
               "date": (old or {}).get("date") or datetime.now().isoformat(timespec="seconds"), "star": bool((old or {}).get("star")),
               "rating": int((old or {}).get("rating") or 0), "res": clean(meta.get("res"), 20), "seed": clean(meta.get("seed"), 20),
               "video": clean(meta.get("video"), 40), "draft_video": clean(meta["draft_video"], 40) if "draft_video" in meta else (old or {}).get("draft_video", ""), "seconds": float(meta.get("seconds") or SECONDS), "layout": meta.get("layout") if isinstance(meta.get("layout"), dict) else {},
               "form": meta.get("form") if isinstance(meta.get("form"), dict) else {}, "beats": beats, "sheet": sheet}
        if old:
            self.runs[self.runs.index(old)] = run
        else:
            self.runs.insert(0, run)
        self._trim()
        self._write(self.index, self.runs)
        return web.json_response(self.public(run))

    def _trim(self):
        keep, drop = [], []
        for r in self.runs:
            (keep if r.get("star") or sum(1 for k in keep if not k.get("star")) < KEEP_RUNS else drop).append(r)
        for r in drop:
            shutil.rmtree(self.rdir / r["id"], ignore_errors=True)
        self.runs = keep

    async def h_rate(self, req):
        r = self.get(req.match_info["id"])
        if not r:
            raise web.HTTPNotFound()
        try:
            body = await req.json()
        except ValueError:
            body = {}
        if "rating" in body:
            r["rating"] = max(-1, min(1, int(body["rating"] or 0)))
        if "star" in body:
            r["star"] = bool(body["star"])
        self._write(self.index, self.runs)
        return web.json_response({"rating": r["rating"], "star": r["star"]})

    async def h_delete(self, req):
        r = self.get(req.match_info["id"])
        if not r:
            raise web.HTTPNotFound()
        self.runs = [x for x in self.runs if x is not r]
        self._write(self.index, self.runs)
        folder = self.rdir / r["id"]
        if folder.is_dir() and self.rdir.resolve() in folder.resolve().parents:
            shutil.rmtree(folder, ignore_errors=True)
        return web.json_response({"ok": True})

    async def h_clear(self, _req):
        gone = [r for r in self.runs if not r.get("star")]
        for r in gone:
            shutil.rmtree(self.rdir / r["id"], ignore_errors=True)
        self.runs = [r for r in self.runs if r.get("star")]
        self._write(self.index, self.runs)
        return web.json_response({"removed": len(gone)})

    @property
    def folder(self):
        return self.root
