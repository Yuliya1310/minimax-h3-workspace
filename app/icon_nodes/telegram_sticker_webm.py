import os
import torch
import numpy as np
import torch.nn.functional as F
import av

try:
    import folder_paths
    _OUT = folder_paths.get_output_directory()
except Exception:
    _OUT = os.path.join(os.getcwd(), "output")


def _prepare(images, fps_in, max_seconds, out_fps, side):
    n, h, w, c = images.shape
    t = n / float(fps_in)
    if t > max_seconds:
        t = max_seconds
    count = max(2, int(round(t * out_fps)))
    idx = (torch.arange(count) * n // count).clamp(max=n - 1)
    x = images[idx].float().clamp(0, 1)
    if c == 3:
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    rgb = x[..., :3]
    a = x[..., 3:4]
    k = float(side) / max(h, w)
    nh = int(round(h * k / 2.0)) * 2
    nw = int(round(w * k / 2.0)) * 2
    if h >= w:
        nh = side
    else:
        nw = side
    p = torch.cat([rgb * a, a], dim=-1).permute(0, 3, 1, 2)
    p = F.interpolate(p, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    p = p.permute(0, 2, 3, 1)
    a2 = p[..., 3:4]
    rgb2 = (p[..., :3] / a2.clamp_min(1e-4)).clamp(0, 1)
    rgb2 = torch.where(a2 < 1e-3, torch.zeros_like(rgb2), rgb2)
    out = torch.cat([rgb2, a2], dim=-1)
    return (out * 255.0 + 0.5).byte().cpu().numpy(), nh, nw


def _encode(frames, fps, crf, path):
    n, h, w, _ = frames.shape
    container = av.open(path, mode="w", format="webm")
    stream = container.add_stream("libvpx-vp9", rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuva420p"
    stream.bit_rate = 0
    stream.options = {
        "crf": str(int(crf)),
        "auto-alt-ref": "0",
        "row-mt": "1",
        "deadline": "good",
        "cpu-used": "2",
    }
    try:
        stream.metadata["alpha_mode"] = "1"
    except Exception:
        pass
    for i in range(n):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[i]), format="rgba")
        frame = frame.reformat(format="yuva420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return os.path.getsize(path)


class TelegramStickerWebM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "telegram/sticker"}),
                "side": ("INT", {"default": 512, "min": 100, "max": 512}),
                "max_kb": ("INT", {"default": 256, "min": 16, "max": 256}),
                "max_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/postprocessing"

    def save(self, images, fps, filename_prefix, side, max_kb, max_seconds):
        limit = int(max_kb) * 1000
        sub, name = os.path.split(filename_prefix)
        folder = os.path.join(_OUT, sub) if sub else _OUT
        os.makedirs(folder, exist_ok=True)
        counter = 1
        while os.path.exists(os.path.join(folder, "%s_%05d.webm" % (name, counter))):
            counter += 1
        final = os.path.join(folder, "%s_%05d.webm" % (name, counter))
        tmp = final + ".tmp.webm"

        best = None
        for out_fps in (30, 24, 20, 15):
            frames, nh, nw = _prepare(images, fps, float(max_seconds), out_fps, int(side))
            lo, hi = 8, 63
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                size = _encode(frames, out_fps, mid, tmp)
                if size <= limit:
                    found = (mid, size)
                    if os.path.exists(final):
                        os.remove(final)
                    os.replace(tmp, final)
                    hi = mid - 1
                else:
                    lo = mid + 1
            if found:
                best = (out_fps, found[0], found[1], nw, nh, len(frames))
                break
        if os.path.exists(tmp):
            os.remove(tmp)
        if best is None:
            size = _encode(frames, 15, 63, final)
            print("[TelegramSticker] WARNING: не удалось уложиться в %d КБ, размер %d байт" % (max_kb, size))
            return (final,)
        print("[TelegramSticker] %s: %dx%d, %d кадров, %d fps, crf %d, %d байт (%.1f КБ), %.2f с"
              % (final, best[3], best[4], best[5], best[0], best[1], best[2], best[2] / 1000.0, best[5] / best[0]))
        return (final,)


NODE_CLASS_MAPPINGS = {"TelegramStickerWebM": TelegramStickerWebM}
NODE_DISPLAY_NAME_MAPPINGS = {"TelegramStickerWebM": "Telegram стикер WebM (VP9, прозрачный, до 256 КБ)"}
