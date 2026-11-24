"""Montage with ffmpeg: join, trim, colour, grain and sound tracks in ONE pass (no ComfyUI, no frame-by-frame processing).
The ffmpeg.exe lives in the program's bin folder (it is part of the installation), so it works for everyone."""
import asyncio
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_enc = {}


def find_ffmpeg(root: Path):
    for p in (root / "bin" / "ffmpeg.exe", root / "bin" / "ffmpeg"):
        if p.exists():
            return str(p)
    return shutil.which("ffmpeg")


def encoder(ff: str):
    """h264_nvenc when the video card can do it (fast), otherwise libx264."""
    if ff in _enc:
        return _enc[ff]
    name = "libx264"
    try:
        r = subprocess.run([ff, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-frames:v", "2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=25, creationflags=NOWIN)
        if r.returncode == 0:
            name = "h264_nvenc"
    except Exception:
        pass
    _enc[ff] = name
    return name


def probe(ff: str, path: str):
    """(width, height, fps) of the first video stream, read from the text ffmpeg prints about a file."""
    try:
        r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=30, creationflags=NOWIN)
        txt = r.stderr.decode("utf-8", "replace")
        m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", txt)
        f = re.search(r"([\d.]+)\s*fps", txt)
        return (int(m.group(1)), int(m.group(2)), float(f.group(1)) if f else 24.0) if m else None
    except Exception:
        return None


def curve(look: dict, n: int = 16):
    """The tone curve (n+1 points 0..1) for one clip: levels (black / white / middle), lifted blacks, dimmed whites,
    brightness and contrast. The same maths as the preview in the montage window."""
    ub, uw, um = float(look.get("black") or 0), float(look.get("white") or 0), float(look.get("lmid") or 0)
    bright, contrast = float(look.get("bright", 1.0)), float(look.get("contrast", 1.0))
    b = max(0.0, -ub) * 1.2 / 255
    w = 1 - max(0.0, uw) * 1.2 / 255
    lift, dim = max(0.0, ub) * 0.0025, max(0.0, -uw) * 0.0025
    mid = b + (w - b) * (0.5 - 0.3 * um / 100)
    gm = math.log(0.5) / math.log((mid - b) / (w - b)) if mid > b and w > b else 1.0
    pts = []
    for k in range(n + 1):
        x = k / n
        y = min(1.0, max(0.0, (x - b) / (w - b))) ** gm
        y = (y * (1 - lift) + lift) * (1 - dim)
        y = min(1.0, max(0.0, y * bright))
        y = min(1.0, max(0.0, (y - 0.5) * contrast + 0.5))
        pts.append((x, y))
    return pts


def is_flat(pts):
    return all(abs(x - y) < 0.004 for x, y in pts)


def look_filters(look: dict, grain: float):
    """ffmpeg filters (a list of strings) for the colour of one clip and the film grain."""
    out = []
    pts = curve(look)
    if not is_flat(pts):
        out.append("curves=all='" + " ".join(f"{x:.4f}/{y:.4f}" for x, y in pts) + "'")
    t, h = float(look.get("temp") or 0) / 100, float(look.get("tint") or 0) / 100
    if abs(t) > 0.004 or abs(h) > 0.004:   # temperature: warm = more red, less blue; tint: magenta = less green
        gr, gb, gg = 1 + 0.28 * t + 0.1 * h, 1 - 0.28 * t + 0.1 * h, 1 - 0.2 * h
        out.append(f"colorchannelmixer=rr={min(2.0, max(0.0, gr)):.4f}:gg={min(2.0, max(0.0, gg)):.4f}:bb={min(2.0, max(0.0, gb)):.4f}")
    s = float(look.get("sat", 1.0))
    if abs(s - 1.0) > 0.005:
        out.append(f"eq=saturation={min(3.0, max(0.0, s)):.3f}")
    if grain > 0:
        out.append(f"noise=alls={max(1, round(grain * 0.35))}:allf=t+u")
    return out


async def run(ff: str, args: list, total: float, on_progress, proc_box: dict):
    """Run ffmpeg; on_progress(0..1) is called while it works. Returns (return code, the tail of its messages)."""
    proc = await asyncio.create_subprocess_exec(ff, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, creationflags=NOWIN)
    proc_box["proc"] = proc
    tail = []

    async def err():
        async for raw in proc.stderr:
            tail.append(raw.decode("utf-8", "replace").rstrip())
            del tail[:-30]

    task = asyncio.create_task(err())
    last = 0.0
    async for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            try:
                sec = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                continue
            if total > 0 and sec - last >= 0.25:
                last = sec
                await on_progress(min(0.99, sec / total))
    await proc.wait()
    await task
    return proc.returncode, "\n".join(tail[-8:])


def build_args(ff, jobdir, clips, total_size, fps, tracks, grain, quality, no_audio, out_path):
    """clips: dicts path, start, dur (the piece, seconds), has_audio, mute, look. tracks: dicts path, at, a, dur, vol (dB).
    Returns the ffmpeg arguments. The filter graph goes to a file, so no quoting problems."""
    W, H = total_size
    args = ["-hide_banner", "-v", "error", "-nostdin", "-y"]
    parts = []
    for i, c in enumerate(clips):
        if c["start"] > 0:
            args += ["-ss", f"{c['start']:.3f}"]
        args += ["-t", f"{c['dur']:.3f}", "-i", c["path"]]
    base = len(clips)
    for t in tracks:
        args += ["-i", t["path"]]
    for i, c in enumerate(clips):
        chain = [f"scale={W}:{H}:force_original_aspect_ratio=decrease", f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black", "setsar=1", f"fps={fps:g}"]
        chain += look_filters(c["look"], grain)
        chain += [f"trim=duration={c['dur']:.3f}", "setpts=PTS-STARTPTS", "format=yuv420p"]
        parts.append(f"[{i}:v]" + ",".join(chain) + f"[v{i}]")
        if not no_audio:
            if c["has_audio"] and not c["mute"]:
                parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad=whole_dur={c['dur']:.3f},"
                             f"atrim=0:{c['dur']:.3f},asetpts=PTS-STARTPTS[a{i}]")
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo:d={c['dur']:.3f},aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
    n = len(clips)
    if no_audio:
        parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    else:
        parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][abase]")
        if tracks:
            labels = ["[abase]"]
            for j, t in enumerate(tracks):
                ms = max(0, int(round(t["at"] * 1000)))
                parts.append(f"[{base + j}:a]atrim=start={t['a']:.3f}:duration={t['dur']:.3f},asetpts=PTS-STARTPTS,aresample=48000,"
                             f"aformat=sample_fmts=fltp:channel_layouts=stereo,volume={t['vol']:.1f}dB,adelay={ms}|{ms}[t{j}]")
                labels.append(f"[t{j}]")
            parts.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first:dropout_transition=0[a]")
        else:
            parts.append("[abase]anull[a]")
    script = Path(jobdir) / "graph.txt"
    script.write_text(";\n".join(parts), encoding="utf-8")
    args += ["-filter_complex_script", str(script), "-map", "[v]"]
    if not no_audio:
        args += ["-map", "[a]"]
    enc = encoder(ff)
    q = {"high": (17, 19), "normal": (21, 23), "small": (26, 28)}.get(quality, (21, 23))
    if enc == "h264_nvenc":
        args += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(q[1]), "-b:v", "0"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(q[0])]
    args += ["-pix_fmt", "yuv420p"]
    if not no_audio:
        args += ["-c:a", "aac", "-b:a", "192k"]
    args += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(out_path)]
    return args


async def thumb(ff: str, src: str, dest: Path):
    """A small JPEG of a video (a frame half a second in), so lists do not have to open the real video."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ss in ("0.5", "0"):
        proc = await asyncio.create_subprocess_exec(ff, "-hide_banner", "-v", "error", "-nostdin", "-y", "-ss", ss, "-i", src, "-frames:v", "1",
                                                    "-vf", "scale=360:-2", "-q:v", "5", str(dest), stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL, creationflags=NOWIN)
        await proc.wait()
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
    dest.unlink(missing_ok=True)
    return False
