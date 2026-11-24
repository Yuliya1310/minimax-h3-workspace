"""Exact song plan: ComfyUI's SheetSage2 turns a song into notes (ABC text). Here the notes become a timed plan:
which bars are sung (voice "Vocal"), which are only the instruments ("Ins"), and the parts of the song (intro, verse, chorus...).
The plan goes to the Agent as a skeleton for dividing the clip into scenes."""
import re

MODEL_FILE = "sheetsage2_bf16.safetensors"
MODEL_URL = "https://huggingface.co/Comfy-Org/YuE2/blob/main/audio_encoders/sheetsage2_bf16.safetensors"
MODEL_DIR = "ComfyUI/models/audio_encoders"

RU = {"silence": "тишина", "intro": "вступление", "outro": "концовка", "verse": "куплет", "chorus": "припев", "bridge": "бридж",
      "pre-chorus": "пред-припев", "post-chorus": "после припева", "interlude": "проигрыш", "fade-out": "затухание", "rap": "речитатив",
      "instrumental": "инструментал", "solo": "соло", "pre-outro": "перед концовкой", "theme": "тема", "development": "развитие", "variation": "вариация"}


def _bars(lines):
    """The bars of one voice in one group of lines: text between the bar lines."""
    txt = re.sub(r'"[^"]*"', "", " ".join(lines))            # chord names
    txt = re.sub(r"\[[A-Za-z]:[^\]]*\]", "", txt)             # inline fields like [K:C]
    parts = re.split(r"\|+", txt)
    out = []
    for p in parts:
        p = p.strip().strip(":[]").strip()
        if not p:
            continue
        z = re.fullmatch(r"Z(\d*)", p)   # «Z4» = four whole bars of rest in a row
        if z:
            out.extend(["z"] * max(1, int(z.group(1) or 1)))
        else:
            out.append(p)
    return out


def _sounds(bar):
    """True when the bar has at least one real note (not only rests)."""
    bar = re.sub(r"![^!]*!", "", bar)
    bar = re.sub(r"\{[^}]*\}", "", bar)
    return bool(re.search(r"[A-Ga-g]", bar))


def parse(abc, total=0.0):
    """abc text -> {tempo, bars:[{t0,t1,vocal,inst,section}], sections:[...], runs:[...]} (times in seconds from the start of the piece)."""
    tempo, meter = 120.0, (4, 4)
    groups, cur, voice, label = [], None, None, ""
    for raw in abc.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        m = re.match(r"Q:\s*1/4\s*=\s*(\d+(?:\.\d+)?)", ln)
        if m and cur is None:
            tempo = max(30.0, float(m.group(1)))
            continue
        m = re.match(r"M:\s*(\d+)/(\d+)", ln)
        if m:
            meter = (int(m.group(1)), int(m.group(2)))
            if cur is not None:
                cur["meter"] = meter
            continue
        if ln.startswith("%"):
            label = ln.lstrip("% ").strip()
            cur = None
            voice = None
            groups.append({"label": label, "meter": meter, "V": {}, "new": True})
            cur = groups[-1]
            continue
        m = re.match(r"V:\s*(\w+)\s*$", ln)
        if m:
            voice = m.group(1)
            if cur is None or voice in cur["V"]:
                groups.append({"label": "", "meter": meter, "V": {}, "new": False})
                cur = groups[-1]
            cur["V"].setdefault(voice, [])
            continue
        if re.match(r"[A-Za-z]:", ln) and not ln.startswith("V:"):
            continue
        if cur is not None and voice:
            cur["V"][voice].append(ln)
    t, bars, sec = 0.0, [], ""
    for g in groups:
        if g["label"]:
            sec = g["label"]
        voc, ins = _bars(g["V"].get("Vocal", [])), _bars(g["V"].get("Ins", []))
        n = max(len(voc), len(ins))
        if not n:
            continue
        num, den = g["meter"]
        dur = num * (4.0 / den) * 60.0 / tempo
        for i in range(n):
            bars.append({"t0": t, "t1": t + dur, "vocal": i < len(voc) and _sounds(voc[i]), "inst": i < len(ins) and _sounds(ins[i]), "section": sec})
            t += dur
    if bars and total > 0 and 0.8 < t / total < 1.2:   # the tempo is an average: stretch to the real length of the piece
        k = total / t
        for b in bars:
            b["t0"] *= k
            b["t1"] *= k
        t = total
    runs = []
    for b in bars:
        kind = "sing" if b["vocal"] else "inst" if b["inst"] else "quiet"
        if runs and runs[-1]["kind"] == kind and runs[-1]["section"] == b["section"]:
            runs[-1]["t1"] = b["t1"]
        else:
            runs.append({"t0": b["t0"], "t1": b["t1"], "kind": kind, "section": b["section"]})
    sections = []
    for r in runs:
        if sections and sections[-1]["name"] == r["section"]:
            sections[-1]["t1"] = r["t1"]
        else:
            sections.append({"name": r["section"], "t0": r["t0"], "t1": r["t1"]})
    return {"tempo": tempo, "bars": len(bars), "end": t, "runs": runs, "sections": sections}


def mmss(x):
    x = max(0.0, x)
    return f"{int(x // 60)}:{x % 60:04.1f}"


def plan_text(plan, offset=0.0, a=0.0, b=None):
    """Plan lines for the Agent. offset = where the piece starts in the whole track; a..b = the window inside the piece (seconds)."""
    KIND = {"sing": "поёт голос", "inst": "только инструменты (проигрыш)", "quiet": "тихо / пауза"}
    b = plan["end"] if b is None else b
    lines = [f"Темп около {plan['tempo']:.0f} ударов в минуту."]
    for r in plan["runs"]:
        s, e = max(r["t0"], a), min(r["t1"], b)
        if e - s < 0.4:
            continue
        nm = RU.get(r["section"].lower(), r["section"]) if r["section"] else ""
        lines.append(f"{mmss(offset + s)}–{mmss(offset + e)} {nm + ': ' if nm else ''}{KIND[r['kind']]}")
    return "\n".join(lines)
