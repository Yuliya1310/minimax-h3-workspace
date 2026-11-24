import torch
import torch.nn.functional as F


class LoopCleanup:
    """Final cleanup of a cut-out animation (RGBA frames):
    1) removes green spots and green tint (despill),
    2) finds the best loop point and trims the frames so the loop is seamless.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "despill": ("BOOLEAN", {"default": True}),
                "find_loop": ("BOOLEAN", {"default": True}),
                "min_loop_frames": ("INT", {"default": 40, "min": 4, "max": 5000}),
                "key_color": (["green", "blue"], {"default": "green"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def run(self, images, despill, find_loop, min_loop_frames, key_color="green"):
        img = images.clone().float()
        B, H, W, C = img.shape

        # 1) key-color spots and tint (only for RGBA cut-outs)
        if despill and C == 4:
            r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
            m = a > 0
            if key_color == "blue":
                mx = torch.maximum(r, g)
                blob = m & ((b - mx) > 60.0 / 255.0) & (b > 140.0 / 255.0)
                b = torch.where(m, torch.minimum(b, mx), b)
            else:
                mx = torch.maximum(r, b)
                blob = m & ((g - mx) > 60.0 / 255.0) & (g > 140.0 / 255.0)
                g = torch.where(m, torch.minimum(g, mx), g)
            a = torch.where(blob, torch.zeros_like(a), a)
            img = torch.stack([r, g, b, a], dim=-1)

        # 2) best loop point
        if find_loop and B > min_loop_frames + 1:
            if C == 4:
                a = img[..., 3:4]
                comp = img[..., :3] * a + 0.5 * (1.0 - a)
            else:
                comp = img[..., :3]
            k = max(1, min(H, W) // 64)
            small = F.avg_pool2d(comp.permute(0, 3, 1, 2), k).flatten(1)
            d = torch.cdist(small, small, p=1) / small.shape[1]

            idx = torch.arange(B)
            length = idx.view(1, B) - idx.view(B, 1)  # e - i
            valid = length >= min_loop_frames
            cost = torch.where(valid, d, torch.full_like(d, float("inf")))
            c0 = cost.min()
            cand = valid & (cost <= c0 * 1.1 + 1e-6)
            score = torch.where(cand, length, torch.full_like(length, -1))
            flat = int(score.argmax())
            i, e = flat // B, flat % B
            out = img[i:e + 1]
            print("[LoopCleanup] frames %d -> %d (from %d to %d), wrap diff %.4f"
                  % (B, out.shape[0], i, e, float(d[i, e])))
            img = out

        return (img,)


NODE_CLASS_MAPPINGS = {"LoopCleanup": LoopCleanup}
NODE_DISPLAY_NAME_MAPPINGS = {"LoopCleanup": "Цикл: подрезка склейки + убрать зелёное"}
