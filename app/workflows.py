import copy
"""ComfyUI API graphs for Minimax H3 Workspace modes.

Each builder returns (graph, stages): the API-format prompt and a map
node_id -> UI stage name, so progress can be shown in plain words.
The graphs mirror the user's workflows in the project root.
"""

import math
import re

FPS = 24

# Model files come in as `m`: role -> name as ComfyUI lists it (see catalog.py).

ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9)}
# sizes of the original workflow's table (16:9 base, multiples of 32); the four old ids are kept for saved settings
QUALITY = {"0.2": 608 * 352, "0.3": 736 * 416, "480p": 864 * 480, "0.5": 960 * 544, "0.6": 1056 * 608, "0.7": 1152 * 640,
           "0.8": 1216 * 672, "720p": 1280 * 736, "HD+": 1376 * 768, "1.2": 1504 * 832, "1.5": 1664 * 928,
           "1.8": 1824 * 1024, "1080p": 1920 * 1088}


def r32(v):
    return max(32, round(v / 32) * 32)


def resolution(aspect, quality, form=None):
    """Size from the format + quality, or the user's own width/height (rounded to a multiple of 32)."""
    if form and str(form.get("width", "")).isdigit() and str(form.get("height", "")).isdigit():
        cw, ch = int(form["width"]), int(form["height"])
        if cw >= 64 and ch >= 64:
            return min(2048, r32(cw)), min(2048, r32(ch))
    w, h = ASPECTS[aspect]
    px = QUALITY[quality]
    return r32(math.sqrt(px * w / h)), r32(math.sqrt(px * h / w))


def frames(seconds):
    # same formula as the workflow's ComfyMathExpression: 24 fps, length = 17k + 5
    f = max(5, round(seconds * FPS))
    return f + (5 - f % 17) % 17


def video_graph(m, prompt, width, height, seconds, seed, first=None, last=None, extra_loras=(), fast=None):
    """MINIMAXH3_FASTvideo_I2V: text / first frame / first + last frame."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["fast_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": 10, "shift_audio": 3}},
        "3": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["2", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["3", 0], "selection": "vsa", "selection.keep_percent": 10.0,
            "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
            "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds)}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["4", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["12", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0], "audio": ["15", 0], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "17": {"class_type": "SaveVideo", "inputs": {
            "video": ["16", 0], "filename_prefix": "H3Workspace/video",
            "format": "auto", "format.codec": "auto"}},
    }
    for slot, (load_id, scale_id, image) in {
        "first_frame": ("20", "21", first), "last_frame": ("22", "23", last)}.items():
        if not image:
            continue
        g[load_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g[scale_id] = {"class_type": "ImageScale", "inputs": {
            "image": [load_id, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        g["8"]["inputs"][slot] = [scale_id, 0]
    loras = ([(fast, 1.0)] if fast else []) + list(extra_loras)  # Turbo-SLA first, then the user's own LoRAs
    if loras:
        g["2"]["inputs"]["model"] = _lora_chain(g, ["1", 0], loras)
    if fast:  # "fast draft": Turbo-SLA 4 steps, flow shifts 6 / 3; block-sparse attention is switched OFF (it left square
        # mosaic artifacts on busy scenes: debris, explosions), dense attention is cleaner and still about 1.3x faster
        g["2"]["inputs"].update(shift_video=6, shift_audio=3)
        g["11"]["inputs"]["model"] = ["3", 0]
        g["12"]["inputs"]["model"] = ["3", 0]
        g.pop("4", None)
        g["11"]["inputs"]["steps"] = 4

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "encode" for n in ("8", "20", "21", "22", "23")})
    stages.update({"13": "sample", "14": "decode", "15": "decode", "16": "save", "17": "save"})
    return g, stages


def _refs(g, cond_node, images, videos=(), audios=(), base=100, vsize=None):
    """LoadImage / VHS_LoadVideo (24 fps) / LoadAudio nodes wired into MiniMaxH3ReferenceToVideo."""
    inputs = g[cond_node]["inputs"]
    for i, name in enumerate(images[:9]):
        nid = str(base + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
    for i, name in enumerate(videos[:3]):
        nid = str(base + 10 + i)
        g[nid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": name, "force_rate": 24, "custom_width": vsize[0] if vsize else 0, "custom_height": vsize[1] if vsize else 0,
            "frame_load_cap": 360, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
        inputs[f"ref_videos.ref_video_{i}"] = [nid, 0]
    for i, name in enumerate(audios[:3]):
        nid = str(base + 20 + i)
        g[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        inputs[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    return [str(base + i) for i in range(len(images[:9]))] + [str(base + 10 + i) for i in range(len(videos[:3]))] \
        + [str(base + 20 + i) for i in range(len(audios[:3]))]


def tune_sampling(g, cfg=None, steps=None):
    """The user's own CFG / steps for a picture graph (None = as the graph has it). Only the main pass (denoise 1): the second pass of
    Krea Real keeps its own. CFG goes to KSampler and CFGGuider (a BasicGuider has no CFG: MiniMax H3 Image takes only the steps)."""
    for v in g.values():
        ct, inp = v.get("class_type"), v.get("inputs", {})
        if ct == "KSampler" and float(inp.get("denoise", 1) or 1) >= 0.99:
            if cfg is not None:
                inp["cfg"] = cfg
            if steps is not None:
                inp["steps"] = steps
        elif ct == "CFGGuider" and cfg is not None:
            inp["cfg"] = cfg
        elif ct in ("BasicScheduler", "Flux2Scheduler") and steps is not None and float(inp.get("denoise", 1) or 1) >= 0.99:
            inp["steps"] = steps
    return g


def add_loras(g, loras):
    """The user's own LoRAs for a picture graph: a LoraLoaderModelOnly stack right after the main model (the first UNETLoader);
    every node that took the model from the loader now takes it from the end of the stack. `loras` = [(file, strength)]."""
    if not loras:
        return g
    uid = next((k for k, v in g.items() if v.get("class_type") == "UNETLoader"), None)
    if uid is None:
        return g
    prev = [uid, 0]
    for i, (name, strength) in enumerate(loras):
        nid = f"own-lora-{i}"
        g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": prev, "lora_name": name, "strength_model": strength}}
        prev = [nid, 0]
    for k, v in g.items():
        if k.startswith("own-lora-"):
            continue
        for key, val in v.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and val[0] == uid and val[1] == 0:
                v["inputs"][key] = prev
    return g


def _lora_chain(g, start, loras, base=60):
    """Stack LoraLoaderModelOnly nodes; returns the last model output."""
    prev = start
    for i, (name, strength) in enumerate(loras):
        nid = str(base + i)
        g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": prev, "lora_name": name, "strength_model": strength}}
        prev = [nid, 0]
    return prev


def ref_graph(m, prompt, width, height, seconds, seed, images, videos, audios, steps=6,
              ref_size="max", extra_loras=(), pika=None):
    """MinimaxH3_Ref to Video (ULTRA SPEED Singularity 4-step), single pass.

    The original samples with T8's Dual Clock node (shift video 12 / audio 4, beta);
    here the same shifts go through ComfyUI's built-in MiniMaxH3SigmaShift + beta scheduler.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
    }
    if pika:  # Pika H3 Flex Turbo (8 steps, HD): its own LoRA instead of the 4-step one; no sigma shift, scheduler "simple"
        mref = _lora_chain(g, ["2", 0], list(extra_loras) + [(pika, 1.0)])
        steps, scheduler = max(6, steps), "simple"
    else:
        model = _lora_chain(g, ["2", 0], list(extra_loras) + [(m["ref2v_turbo_lora"], 0.7)])
        g["9"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}}
        mref, scheduler = ["9", 0], "beta"
    g.update({
        "10": {"class_type": "BasicScheduler", "inputs": {"model": mref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": mref, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/refs", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, videos, audios)
    stages = {n: "load" for n in ["1", "2", "4", "5", "6"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "16": "decode", "17": "save", "18": "save"})
    return g, stages


# MiniMax_H3_Direct-ai-Actor: one pass, 10 steps with the author's own sigma list
ACTOR_SIGMAS = "1, .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0"


def actor_graph(m, prompt, width, height, seconds, seed, images, audios, realism=None, extra_loras=(), videos=()):
    """Acting scenes: first frame + a voice sample, new speech written in the prompt (<d>...</d>).

    Same Singularity Ref2V model as the references mode, but with sol-attn sparse attention,
    sigma shift 12.2 / 3, er_sde and the turbo LoRA at full strength, as in the original workflow.
    `realism` is the (file, strength) of the people-realism LoRA when it is on.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": "match"}},
    }
    loras = [(m["ref2v_turbo_lora"], 1.0)] + ([realism] if realism else []) + list(extra_loras)
    model = _lora_chain(g, ["7", 0], loras)
    g.update({
        "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/actor", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, videos, audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "16": "decode", "17": "save", "18": "save"})
    return g, stages


def swap_prompt(whos, extra="", cuts=None, structured=True):
    """Prompt of the character swap preset. `whos`: who to replace with <Picture 1>, <Picture 2>... (one phrase each, in the
    user's words; a single string is accepted too). `extra`: optional addition. `cuts`: seconds where shot 2, 3... start (when
    several people are swapped in one pass: one person per shot). Tags follow the order the files are connected.
    `structured`: MiniMax's official three-field prompt (alignment line, [Shot N] timeline, soundscape, music), as References use."""
    whos = [whos] if isinstance(whos, str) else list(whos)
    tail = (" Retain the identity, attire, and style of the "
            + ("character from <Picture 1>." if len(whos) == 1 else "replacement characters from their pictures.")
            + " Simultaneously, preserve the camera angle, background, lighting, objects, and all other figures from the original video."
            " Ensure each replacement character aligns with the original figure in terms of position, scale, pose, and movement."
            + (" Only this one character is added: do not add any other figures, copies or people from the picture, and keep the background of <Video 1>."
               if len(whos) == 1 else " Do not add any other figures, copies or people, and keep the background of <Video 1>.")
            + " Do not display the reference images or their backgrounds.")
    names = [(w or "").strip() or ("the main person" if len(whos) == 1 else f"person {i + 1}") for i, w in enumerate(whos)]
    if not structured:
        t = " ".join(f"Replace {'only ' if len(whos) == 1 else ''}{w} in <Video 1> with the character from <Picture {i + 1}>."
                     for i, w in enumerate(names)) + tail
        return t + (" " + extra.strip() if extra and extra.strip() else "")
    pics = " ".join(f"<Picture {i + 1}> (from [Shot {i + 1}]) is fully referenced." for i in range(len(names)))
    head = (f"For the target video, at 0.00 seconds into the target video, {pics} <Video 1> is the video to edit: its motion, camera, "
            "timing, setting and sound are kept. The faces and identities of the replaced people must match the reference pictures exactly.")
    shots = []
    for i, w in enumerate(names):
        if i == 0:
            lead = "[Shot 1] Live-action, "
        else:
            t0 = (cuts[i - 1] if cuts and len(cuts) >= i else None)
            lead = (f"[Shot {i + 1}] At {int(t0 // 60):02d}:{t0 % 60:06.3f}, the camera cuts to the next shot of <Video 1>. " if t0 is not None
                    else f"[Shot {i + 1}] After the cut in <Video 1>, ")
        verb = "Replace" if lead.endswith(". ") else "replace"
        shots.append(lead + f"{verb} {'only ' if len(names) == 1 else ''}{w} in <Video 1> with the character from <Picture {i + 1}>.")
    body = " ".join(shots) + tail + (" " + extra.strip() if extra and extra.strip() else "")
    return f"{head}\n\nintegrated_multimodal_description: {body}\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A"


def swap_graph(m, prompt, width, height, seconds, seed, images, video, extra_loras=(), pika=None, audio="original", small_ref=False):
    """Character swap preset (fallback group of Smart Bobo's «MiniMax H3 Character Swap» workflow):
    Ref2V model + swap LoRA 1.0 + turbo LoRA, sol-attn, sigma shift 12.2 / 3, er_sde, 10 manual sigmas.
    With `pika` (file name): the Pika H3 Flex Turbo LoRA instead of the turbo one, 8 steps, scheduler simple (as in References).
    `audio`: "original" - the sound of the video is put into the result; "lock" - the same, and Audio Lock makes the mouths follow
    that exact sound (lip sync); "model" - the sound the model invents (for videos that have no sound)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": "match"}},
    }
    if pika:
        model = _lora_chain(g, ["2", 0], [(m["swap_lora"], 1.0), (pika, 1.0)] + list(extra_loras))
        g.update({
            "10": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 8, "denoise": 1.0}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        })
    else:
        g["3"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}}
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}}
        model = _lora_chain(g, ["7", 0], [(m["swap_lora"], 1.0), (m["ref2v_turbo_lora"], 1.0)] + list(extra_loras))
        g.update({
            "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        })
    latent, snd = ["8", 1], ["110", 2]
    if audio == "lock":   # exact sound of the video encoded into the latent: video is denoised while listening to it
        g["9"] = {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
            "model": ["1", 0], "av_latent": ["8", 1], "audio_vae": ["6", 0], "audio": ["110", 2]}}
        g["10"]["inputs"]["model"] = ["9", 0]
        latent, snd = ["9", 1], ["9", 2]
    g.update({
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": latent}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": snd, "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/swap", "format": "auto", "format.codec": "auto"}},
    })
    images = [images] if isinstance(images, str) else list(images)
    if audio == "model":   # a video without sound: the result gets the sound the model invents
        g["16"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}}
        g["17"]["inputs"]["audio"] = ["16", 0]
    loaders = _refs(g, "8", images[:2], [video], (), vsize=(width, height) if small_ref else None)   # node 110 = the video loader; its sound goes into the result
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] if n in g}
    stages.update({k: "load" for k in g if k.isdigit() and 60 <= int(k) < 70})
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "17": "save", "18": "save"})
    return g, stages


def swap_shots_graph(m, shots, width, height, seed, images, video, extra="", pika=None, extra_loras=(), audio="original", small_ref=False):
    """Character swap over several shots (plans) of one video. The LoRA is made for one continuous shot, so every shot
    is swapped on its own and the pieces are joined again.

    shots: [{"s": start sec, "e": end sec, "pic": picture index or -1 (keep the original shot), "who": words}].
    VHS_LoadVideo takes just that part of the video (skip_first_frames / frame_load_cap, 24 fps), so no cutting is needed."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    if pika:
        model = _lora_chain(g, ["2", 0], [(m["swap_lora"], 1.0), (pika, 1.0)] + list(extra_loras))
        g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 8, "denoise": 1.0}}
        g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    else:
        g["3"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}}
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}}
        model = _lora_chain(g, ["7", 0], [(m["swap_lora"], 1.0), (m["ref2v_turbo_lora"], 1.0)] + list(extra_loras))
        g["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}}
        g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}}
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] if n in g}
    stages.update({k: "load" for k in g if k.isdigit() and 60 <= int(k) < 70})
    pieces = {}
    for k, sh in enumerate(shots):
        n = max(5, int(round((sh["e"] - sh["s"]) * 24)))
        skip = int(round(sh["s"] * 24))
        sec = min(15.0, max(2.0, (sh["e"] - sh["s"])))
        vid = f"V{k}"
        g[vid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video, "force_rate": 24, "custom_width": width if small_ref else 0, "custom_height": height if small_ref else 0,
            "frame_load_cap": min(360, n),
            "skip_first_frames": skip, "select_every_nth": 1, "format": "None"}}
        stages[vid] = "encode"
        if sh.get("pic", -1) < 0 or sh["pic"] >= len(images):   # this shot stays as it is, with its own sound
            g[f"C{k}"] = {"class_type": "CreateVideo", "inputs": {
                "images": [vid, 0], "audio": [vid, 2], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}}
            stages[f"C{k}"] = "decode"
            pieces[k] = [f"C{k}", 0]
            continue
        prompt = swap_prompt(sh.get("who") or "", extra)
        pic = f"I{k}"
        g[pic] = {"class_type": "LoadImage", "inputs": {"image": images[sh["pic"]]}}
        ref = f"R{k}"
        g[ref] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt, "width": width, "height": height,
            "length": frames(sec), "ref_image_size": "match", "ref_images.ref_image_0": [pic, 0], "ref_videos.ref_video_0": [vid, 0]}}
        latent, sigmas, snd = [ref, 1], ["10", 0], [vid, 2]
        if audio == "lock":   # this shot's own sound is locked in: the mouths follow it, and the right person speaks the right lines
            lock_audio = [vid, 2]
            a_end = sh.get("a_end")
            if a_end is not None and sh["s"] < a_end < sh["e"] - 0.1:
                # the sound leads the picture (the next person starts to speak while this one is still on the screen):
                # only the part up to a_end drives this person's mouth, the rest is silence; the real sound stays in the video
                g[f"X{k}"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": [vid, 2], "start_index": 0.0, "duration": round(a_end - sh["s"], 3)}}
                g[f"Y{k}"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(sh["e"] - a_end, 3), "sample_rate": 44100, "channels": 2}}
                g[f"W{k}"] = {"class_type": "AudioConcat", "inputs": {"audio1": [f"X{k}", 0], "audio2": [f"Y{k}", 0], "direction": "after"}}
                lock_audio = [f"W{k}", 0]
                stages.update({f"X{k}": "encode", f"Y{k}": "encode", f"W{k}": "encode"})
            g[f"K{k}"] = {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
                "model": ["1", 0], "av_latent": [ref, 1], "audio_vae": ["6", 0], "audio": lock_audio}}
            sch = dict(g["10"]["inputs"]); sch["model"] = [f"K{k}", 0]
            g[f"H{k}"] = {"class_type": g["10"]["class_type"], "inputs": sch}
            latent, sigmas = [f"K{k}", 1], [f"H{k}", 0]   # the video keeps the real sound of the shot (snd stays the original)
            stages[f"K{k}"] = "encode"
        g[f"G{k}"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": [ref, 0]}}
        g[f"N{k}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        g[f"S{k}"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [f"N{k}", 0], "guider": [f"G{k}", 0], "sampler": ["12", 0], "sigmas": sigmas, "latent_image": latent}}
        g[f"D{k}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"S{k}", 1], "vae": ["5", 0]}}
        g[f"T{k}"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"D{k}", 0], "batch_index": 0, "length": n}}
        g[f"A{k}"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [f"S{k}", 1], "vae": ["6", 0]}}
        g[f"C{k}"] = {"class_type": "CreateVideo", "inputs": {
            "images": [f"T{k}", 0], "audio": snd, "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}}
        del g[f"A{k}"]   # the sound of the original video is kept, the one the model invents is not used
        stages.update({pic: "encode", ref: "encode", f"S{k}": "sample", f"D{k}": "decode", f"T{k}": "decode", f"C{k}": "decode"})
        pieces[k] = [f"C{k}", 0]
    if len(pieces) == 1:
        last = pieces[0]
    else:
        g["J"] = {"class_type": "ConcatenateVideo", "inputs": {**{f"videos.video{k}": v for k, v in pieces.items()}, "codec": "auto"}}
        stages["J"] = "decode"
        last = ["J", 0]
    g["Z"] = {"class_type": "SaveVideo", "inputs": {"video": last, "filename_prefix": "H3Workspace/swap",
                                                    "format": "auto", "format.codec": "auto"}}
    stages["Z"] = "save"
    return g, stages


# ---------- long video: scenes continue each other (Music-Video / Long-Shot workflow) ----------

def grain_graph(name, strength, seed, look=None, ref_name=None):
    """Colour and grain on one clip. Colour first (match to the colour of a reference clip, brightness, contrast, saturation),
    then film-like grain: KJNodes' per-pixel noise (it differs from frame to frame). strength 0..1 (0: no grain); the sound stays.
    `look`: {"match": 0..1 strength of the match to `ref_name` (a video in ComfyUI's input folder), "bright", "contrast", "sat": factors, 1.0 = as is}."""
    look = look or {}
    if not strength or strength <= 0:
        return _color_graph(name, look, ref_name)
    # Plain per-pixel colour noise looked like flicker. Film grain is a single-colour (brightness) noise with grains of 2-3 pixels:
    # noise is made at half size on mid-grey, taken from one channel, enlarged back and laid over the picture with «soft light».
    g = {"L": {"class_type": "LoadVideo", "inputs": {"file": name}},
         "N": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}},
         "H": {"class_type": "ImageScaleBy", "inputs": {"image": ["N", 0], "upscale_method": "area", "scale_by": 0.5}},
         "Z": {"class_type": "ImageBlend", "inputs": {"image1": ["H", 0], "image2": ["H", 0], "blend_factor": 1.0, "blend_mode": "difference"}},   # black, same size and frame count
         "I": {"class_type": "ImageInvert", "inputs": {"image": ["Z", 0]}},
         "G": {"class_type": "ImageBlend", "inputs": {"image1": ["Z", 0], "image2": ["I", 0], "blend_factor": 0.5, "blend_mode": "normal"}},     # mid-grey
         "R": {"class_type": "ImageNoiseAugmentation", "inputs": {"image": ["G", 0], "noise_aug_strength": 0.3, "seed": seed}},
         "M": {"class_type": "ImageToMask", "inputs": {"image": ["R", 0], "channel": "red"}},
         "K": {"class_type": "MaskToImage", "inputs": {"mask": ["M", 0]}},
         "B": {"class_type": "ImageScaleBy", "inputs": {"image": ["K", 0], "upscale_method": "bilinear", "scale_by": 2.0}},
         "F": {"class_type": "ImageBlend", "inputs": {"image1": ["N", 0], "image2": ["B", 0], "blend_factor": round(min(1.0, strength * 2), 3), "blend_mode": "soft_light"}},   # measured: 30% (factor 0.3) gave only about 5 of 255 levels, too weak
         "W": {"class_type": "CreateVideo", "inputs": {"images": ["F", 0], "fps": ["N", 2], "audio": ["N", 1]}},
         "S": {"class_type": "SaveVideo", "inputs": {"video": ["W", 0], "filename_prefix": "H3Workspace/grain", "format": "auto", "codec": "auto"}}}
    base, ids = _color_nodes(g, ["N", 0], look, ref_name)
    g["F"]["inputs"]["image1"] = base
    return g, {k: "encode" for k in "LNHZIGRMKBF"} | {i: "encode" for i in ids} | {"W": "decode", "S": "save"}


def _color_nodes(g, base, look, ref_name):
    """Colour steps over the frames `base` (a node output): match to a reference clip's frame, brightness, contrast, saturation.
    Returns (the last output, ids of the nodes added). Core nodes AdjustBrightness / AdjustContrast, KJNodes ColorMatch, comfyui-various saturation."""
    ids = []
    if ref_name and (look.get("match") or 0) > 0:
        g["RL"] = {"class_type": "LoadVideo", "inputs": {"file": ref_name}}
        g["RN"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["RL", 0]}}
        g["RF"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["RN", 0], "batch_index": 12, "length": 1}}   # a frame half a second in: not the first one, which may be dark
        g["CM"] = {"class_type": "ColorMatch", "inputs": {"image_ref": ["RF", 0], "image_target": base, "method": "mkl",
                                                          "strength": round(float(look["match"]), 3), "multithread": True}}
        ids += ["RL", "RN", "RF", "CM"]
        base = ["CM", 0]
    for key, cls, field, name in (("bright", "AdjustBrightness", "images", "CB"), ("contrast", "AdjustContrast", "images", "CC"),
                                  ("sat", "JWImageSaturation", "image", "CS")):
        v = look.get(key)
        if v is not None and abs(float(v) - 1.0) > 0.005:
            g[name] = {"class_type": cls, "inputs": {field: base, "factor": round(min(2.0, max(0.0, float(v))), 3)}}
            ids.append(name)
            base = [name, 0]
    # levels, all three sliders: 0 in the middle, left = darker, right = lighter (as in photo editors), -100..100.
    # black: left crushes the shadows (levels node), right lifts them (a white layer); white: right brightens (levels node), left dims (a black layer); mid: right = lighter
    ub, uw, um = float(look.get("black") or 0), float(look.get("white") or 0), float(look.get("lmid") or 0)
    lb, lw, lm = max(0.0, -ub) * 1.2, 255.0 - max(0.0, uw) * 1.2, um
    if lb > 0.5 or lw < 254.5 or abs(lm) > 0.5:
        g["LV"] = {"class_type": "Image Levels Adjustment", "inputs": {"image": base, "black_level": round(lb, 1), "white_level": round(lw, 1),
                                                                      "mid_level": round(lb + (lw - lb) * (0.5 - 0.3 * lm / 100.0), 1)}}
        ids.append("LV")
        base = ["LV", 0]
    for tag, v, col in (("LF", ub, 0xFFFFFF), ("LD", -uw, 0x000000)):   # lifted blacks / dimmed whites: a flat layer over the frames
        if v > 0.5:
            g[tag + "E"] = {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": col}}
            g[tag + "B"] = {"class_type": "ImageBlend", "inputs": {"image1": base, "image2": [tag + "E", 0], "blend_factor": round(v * 0.0025, 4), "blend_mode": "normal"}}
            ids += [tag + "E", tag + "B"]
            base = [tag + "B", 0]
    # white balance: temperature (cold .. warm) and tint (green .. magenta) are a soft colour layer over the frames (core nodes EmptyImage + ImageBlend)
    for key, pos, neg, tag in (("temp", 0xFFB347, 0x4C8DFF, "WT"), ("tint", 0xFF5FD2, 0x4CFF6A, "WH")):
        v = float(look.get(key) or 0)
        if abs(v) > 0.5:
            g[tag + "E"] = {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": pos if v > 0 else neg}}
            g[tag + "B"] = {"class_type": "ImageBlend", "inputs": {"image1": base, "image2": [tag + "E", 0],
                                                                    "blend_factor": round(min(1.0, abs(v) / 100.0) * 0.4, 3), "blend_mode": "overlay"}}
            ids += [tag + "E", tag + "B"]
            base = [tag + "B", 0]
    return base, ids


def _color_graph(name, look, ref_name):
    """Colour only (no grain): the clip with its sound, graded."""
    g = {"L": {"class_type": "LoadVideo", "inputs": {"file": name}},
         "N": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}}}
    base, ids = _color_nodes(g, ["N", 0], look, ref_name)
    g["W"] = {"class_type": "CreateVideo", "inputs": {"images": base, "fps": ["N", 2], "audio": ["N", 1]}}
    g["S"] = {"class_type": "SaveVideo", "inputs": {"video": ["W", 0], "filename_prefix": "H3Workspace/color", "format": "auto", "codec": "auto"}}
    return g, {k: "encode" for k in ["L", "N"] + ids} | {"W": "decode", "S": "save"}


def _mix_tracks(g, stages, base, items, total):
    """Audio tracks over the base soundtrack. Every item is made exactly `total` seconds long (silence before and after it), so the
    core AudioMerge only has to add equal-length pieces. items: dicts name (file in ComfyUI input), at (second on the timeline),
    a (from which second of the file), dur, vol (dB, whole number)."""
    if base is None:
        g["B0"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(total, 3), "sample_rate": 48000, "channels": 2}}
        base = ["B0", 0]
        stages["B0"] = "decode"
    for j, it in enumerate(items):
        at = max(0.0, float(it["at"]))
        dur = min(float(it["dur"]), total - at)
        if dur < 0.05:
            continue
        p = f"X{j}"
        g[p + "L"] = {"class_type": "LoadAudio", "inputs": {"audio": it["name"]}}
        g[p + "T"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": [p + "L", 0], "start_index": round(float(it.get("a") or 0), 3), "duration": round(dur, 3)}}
        piece, ids = [p + "T", 0], [p + "L", p + "T"]
        vol = int(round(float(it.get("vol") or 0)))
        if vol:
            g[p + "V"] = {"class_type": "AudioAdjustVolume", "inputs": {"audio": piece, "volume": max(-100, min(100, vol))}}
            piece = [p + "V", 0]
            ids.append(p + "V")
        if at > 0.005:
            g[p + "E"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(at, 3), "sample_rate": 48000, "channels": 2}}
            g[p + "F"] = {"class_type": "AudioConcat", "inputs": {"audio1": [p + "E", 0], "audio2": piece, "direction": "after"}}
            piece = [p + "F", 0]
            ids += [p + "E", p + "F"]
        tail = total - at - dur
        if tail > 0.005:
            g[p + "G"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(tail, 3), "sample_rate": 48000, "channels": 2}}
            g[p + "H"] = {"class_type": "AudioConcat", "inputs": {"audio1": piece, "audio2": [p + "G", 0], "direction": "after"}}
            piece = [p + "H", 0]
            ids += [p + "G", p + "H"]
        g[p + "M"] = {"class_type": "AudioMerge", "inputs": {"audio1": base, "audio2": piece, "merge_method": "add"}}
        ids.append(p + "M")
        for i_ in ids:
            stages[i_] = "decode"
        base = [p + "M", 0]
    return base


def _fit_soundtrack(g, stages, names, cuts, timing, starts=None, mute=None):
    """One soundtrack for the whole montage, built clip by clip: the sound of each clip is cut or padded with silence to the exact
    length of ITS picture, so the small differences (a few hundredths of a second per clip) do not add up over many clips.
    `timing[i]` = (picture seconds, sound seconds or None). Returns the audio output to give to ConcatenateVideo or None."""
    if not timing or any(t is None for t in timing) or all(t[1] is None for t in timing):
        return None
    out = None
    for i, (vdur, adur) in enumerate(timing):
        st0 = float(starts[i]) if starts and i < len(starts) and starts[i] else 0.0   # the clip is cut from this second on
        left = max(0.1, vdur - st0)
        want = round(min(left, cuts[i]) if cuts and i < len(cuts) and cuts[i] else left, 3)
        if mute and i < len(mute) and mute[i]:   # the user took the sound of this clip away
            adur = None
        if adur is None:   # a clip without sound: silence of its length
            piece = [f"Q{i}", 0]
            g[f"Q{i}"] = {"class_type": "EmptyAudio", "inputs": {"duration": want, "sample_rate": 48000, "channels": 2}}
        else:
            g[f"A{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": names[i]}}
            piece = [f"A{i}", 0]
            have = max(0.0, adur - st0)   # sound that is there from the cut point on
            if st0 > 0.001 or adur > st0 + want + 0.002:
                if have < 0.05:   # the cut point is past the sound: silence
                    g[f"Q{i}"] = {"class_type": "EmptyAudio", "inputs": {"duration": want, "sample_rate": 48000, "channels": 2}}
                    piece, have = [f"Q{i}", 0], want
                else:
                    g[f"R{i}"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": piece, "start_index": round(st0, 3), "duration": round(min(want, have), 3)}}
                    piece = [f"R{i}", 0]
            adur = have if st0 > 0.001 or adur > st0 + want + 0.002 else adur
            if want - adur > 0.002 and piece != [f"Q{i}", 0]:
                g[f"E{i}"] = {"class_type": "EmptyAudio", "inputs": {"duration": round(want - adur, 3), "sample_rate": 48000, "channels": 2}}
                g[f"P{i}"] = {"class_type": "AudioConcat", "inputs": {"audio1": piece, "audio2": [f"E{i}", 0], "direction": "after"}}
                piece = [f"P{i}", 0]
        if out is None:
            out = piece
        else:
            g[f"J{i}"] = {"class_type": "AudioConcat", "inputs": {"audio1": out, "audio2": piece, "direction": "after"}}
            out = [f"J{i}", 0]
    return out


def montage_graph(names, sizes, cuts=None, timing=None, music=None, starts=None, mute=None, tracks=None, total=None):
    """Join clips end to end with ComfyUI's own nodes (no ffmpeg). Clips of another size than the first one are scaled to it
    first: the join node refuses clips of different sizes. `sizes`: (width, height) per clip or None when unknown."""
    g, ins, stages = {}, {}, {}
    first = next((s for s in sizes[:1] if s), None)
    for i, name in enumerate(names):
        g[f"L{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        stages[f"L{i}"] = "encode"
        src = [f"L{i}", 0]
        st0 = float(starts[i]) if starts and i < len(starts) and starts[i] else 0.0
        has_cut = bool(cuts and i < len(cuts) and cuts[i])
        if has_cut or st0 > 0.001:   # the clip is trimmed: a piece of it from `st0` on, `cuts[i]` seconds (or to the end)
            vdur = timing[i][0] if timing and i < len(timing) and timing[i] else None
            dur = float(cuts[i]) if has_cut else (max(0.1, vdur - st0) if vdur else 3600.0)
            g[f"T{i}"] = {"class_type": "Video Slice", "inputs": {"video": src, "start_time": round(st0, 3), "duration": round(dur, 3), "strict_duration": False}}
            stages[f"T{i}"] = "encode"
            src = [f"T{i}", 0]
        if first and sizes[i] and tuple(sizes[i]) != tuple(first):
            g[f"G{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": src}}
            g[f"Z{i}"] = {"class_type": "ImageScale", "inputs": {"image": [f"G{i}", 0], "upscale_method": "lanczos",
                                                                 "width": first[0], "height": first[1], "crop": "center"}}
            g[f"V{i}"] = {"class_type": "CreateVideo", "inputs": {"images": [f"Z{i}", 0], "fps": [f"G{i}", 2], "audio": [f"G{i}", 1]}}
            stages.update({f"G{i}": "encode", f"Z{i}": "encode", f"V{i}": "decode"})
            src = [f"V{i}", 0]
        ins[f"videos.video{i}"] = src
    g["C"] = {"class_type": "ConcatenateVideo", "inputs": {**ins, "codec": "auto"}}
    if music:   # a music clip: ONE piece of the original song under the whole video (no clicks at the joins of separate sounds)
        m_name, m_start, m_len = music
        g["MA"] = {"class_type": "LoadAudio", "inputs": {"audio": m_name}}
        g["MT"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": ["MA", 0], "start_index": round(m_start, 3), "duration": m_len}}
        g["C"]["inputs"]["complete_audio"] = ["MT", 0]
        stages.update({"MA": "decode", "MT": "decode"})
        sound = None
    else:
        sound = _fit_soundtrack(g, stages, names, cuts, timing, starts, mute)
    if tracks and total:   # the user's own audio tracks laid over the sound of the clips
        sound = _mix_tracks(g, stages, sound if sound else (["MT", 0] if music else None), tracks, total)
        g["C"]["inputs"]["complete_audio"] = sound
        stages.update({k: "decode" for k in g if k[:1] in "AEJPQR" and k[1:].isdigit()})
        sound = None
    if sound:
        g["C"]["inputs"]["complete_audio"] = sound
        stages.update({k: "decode" for k in g if k[:1] in "AEJPQR" and k[1:].isdigit()})
    g["S"] = {"class_type": "SaveVideo", "inputs": {"video": ["C", 0], "filename_prefix": "H3Workspace/montage",
                                                    "format": "auto", "codec": "auto"}}
    stages.update({"C": "decode", "S": "save"})
    return g, stages


LONG_MAX_SCENE = 15      # seconds per scene (the model is trained for about 5-15 s)
LONG_OVERLAP = 1.0       # seconds of the previous scene that the next one continues from


def _snap(f):
    f = int(round(f))
    return f + (5 - f % 17) % 17


def _tokens(f):
    return round((f - 5) / 17 * 5 + 2)


def long_plan(scene_seconds, overlap=LONG_OVERLAP, cont_first=False, fresh=()):
    """Frame arithmetic of the original workflow: per scene (frames to generate, overlap frames, frames kept so far).
    `cont_first`: the first scene continues an existing video, so it is planned like a later scene.
    `fresh[i]`: scene i is a NEW SHOT: it does not continue the previous scene (no overlap, no saved tail), it starts from the
    reference pictures like scene 1, so the quality of a long video does not wear off from scene to scene."""
    plan, acc = [], 0
    for i, sec in enumerate(scene_seconds):
        seg = frames(min(LONG_MAX_SCENE, max(1, sec)))
        if i > 0 and i < len(fresh) and fresh[i]:
            plan.append({"total": seg, "ov": 0, "acc_before": acc, "acc": acc + seg})
            acc += seg
        elif i == 0 and not cont_first:
            plan.append({"total": seg, "ov": 0, "acc_before": 0, "acc": seg})
            acc = seg
        else:
            total = _snap(seg + _snap(max(5, round(overlap * FPS))))
            ov = _snap(total - seg)
            plan.append({"total": total, "ov": ov, "acc_before": acc, "acc": acc - ov + total})
            acc = acc - ov + total
    return plan


def add_preview(g, preview):
    """The animated live preview for a finished graph of any video mode: KJNodes' Model Preview Override is put in front of
    every guider (the model the sampler actually uses). Returns True when something was added."""
    if not preview:
        return False
    n = 0
    for nid, node in list(g.items()):
        if node.get("class_type") not in ("BasicGuider", "CFGGuider", "DualCFGGuider"):
            continue
        src = node["inputs"].get("model")
        if not isinstance(src, list) or str(src[0]).startswith("pv_"):
            continue
        pid = f"pv_{nid}"
        g[pid] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {
            "model": src, "max_resolution": 512, "jpeg_quality": 75, "suppress_default_preview": False,
            "preview_frames": 12, "preview_fps": 8, "tiny_vae": preview.get("tiny_vae", "none")}}
        node["inputs"]["model"] = [pid, 0]
        n += 1
    return n > 0


def _with_preview(g, model, preview):
    """KJNodes' Model Preview Override: an ANIMATED live preview (a few frames of the video latent per step), like in the
    ComfyUI workflow. Without it ComfyUI shows only the first frame. `preview` is None when that node is not installed."""
    if not preview:
        return model
    g["14"] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {
        "model": model, "max_resolution": 512, "jpeg_quality": 75, "suppress_default_preview": False,
        "preview_frames": 12, "preview_fps": 8, "tiny_vae": preview.get("tiny_vae", "none")}}
    return ["14", 0]


def _long_base(m, pika, extra_loras, draft=False):
    """Model chain shared by the long-video graphs: Singularity Ref2V + memory savers + Pika (or the quality route)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "MiniMaxChunkFeedForward", "inputs": {"model": ["2", 0], "chunks": 2, "seq_threshold": 4096}},
        "4": {"class_type": "MiniMaxLowVRAMAttention", "inputs": {"model": ["3", 0], "head_chunks": 4}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    steps = 8
    if pika:
        model = _lora_chain(g, ["4", 0], list(extra_loras) + [(pika, 1.0)])
        sched, sampler_name = "simple", "euler"
        if draft:   # a quick draft: `pika` is the DMAD 4-step LoRA here (lcm sampler, shifts as in the editor it comes from)
            g["8"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 2.0}}
            model, sampler_name, steps = ["8", 0], "lcm", 4
    else:  # quality route without a speed LoRA, as in the original (er_sde, beta, 12 steps)
        model = _lora_chain(g, ["4", 0], list(extra_loras)) if extra_loras else ["4", 0]
        g["8"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}}
        model, sched, sampler_name, steps = ["8", 0], "beta", "er_sde", 12
    return g, model, sched, sampler_name, steps


def _audio_tokens(frames):
    return round(frames * 5 / 3)   # the audio latent of H3 has 40 steps per second, a video frame is 1/24 s


FADE_DB = (-4, -9, -18, -40)   # 4 steps of 50 ms: the sound dies away (or swells up) around a join of two scenes, no click or jump
FADE_STEP = 0.05


def _audio_fade(add, audio, dur, head=False, tail=False):
    """The same sound with a short fade-in at the start and/or a fade-out at the end (core nodes only: volume steps of 50 ms)."""
    n = len(FADE_DB)
    edge = n * FADE_STEP
    if dur < 2 * edge + 0.1 or not (head or tail):
        return audio
    t0 = edge if head else 0.0
    t1 = dur - (edge if tail else 0.0)
    parts = []
    if head:
        for j in range(n):
            piece = add("TrimAudioDuration", audio=audio, start_index=round(j * FADE_STEP, 3), duration=FADE_STEP)
            parts.append([add("AudioAdjustVolume", audio=[piece, 0], volume=FADE_DB[n - 1 - j]), 0])
    parts.append([add("TrimAudioDuration", audio=audio, start_index=round(t0, 3), duration=round(t1 - t0, 3)), 0])
    if tail:
        for j in range(n):
            piece = add("TrimAudioDuration", audio=audio, start_index=round(t1 + j * FADE_STEP, 3), duration=FADE_STEP)
            parts.append([add("AudioAdjustVolume", audio=[piece, 0], volume=FADE_DB[j]), 0])
    out = parts[0]
    for nxt in parts[1:]:
        out = [add("AudioConcat", audio1=out, audio2=nxt, direction="after"), 0]
    return out


def _audio_continue(add, old_audio, new_audio, ov, total):
    """The audio latent of a scene starts with the end of the previous scene's audio (kept as it is) and continues with new sound:
    the same trick as for the picture. Returns the audio latent for the sampler."""
    a_ov, a_tot = _audio_tokens(ov), _audio_tokens(total)
    cut = add("LatentCut", samples=old_audio, dim="y", index=-a_ov, amount=a_ov)
    cat = add("LatentConcat", samples1=[cut, 0], samples2=new_audio, dim="y")
    cut2 = add("LatentCut", samples=[cat, 0], dim="y", index=0, amount=a_tot)
    keep = add("SolidMask", value=0.0, width=a_ov, height=2)
    ones = add("SolidMask", value=1.0, width=a_tot, height=2)
    mask = add("MaskComposite", destination=[ones, 0], source=[keep, 0], x=0, y=0, operation="multiply")
    return [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[mask, 0]), 0]


def _scene_voice(add, voice, lead, total):
    """The recording of a scene's voice as the sound of the whole scene: silence over the overlap with the previous scene (`lead`, s),
    then the voice, then silence up to the scene's full length (`total` frames)."""
    v = [add("LoadAudio", audio=voice), 0]
    if lead > 0:
        v = [add("AudioConcat", audio1=[add("EmptyAudio", duration=round(lead, 3), sample_rate=48000, channels=2), 0], audio2=v, direction="after"), 0]
    pad = add("EmptyAudio", duration=round(total / FPS, 3), sample_rate=48000, channels=2)
    v = [add("AudioConcat", audio1=v, audio2=[pad, 0], direction="after"), 0]
    return [add("TrimAudioDuration", audio=v, start_index=0.0, duration=round(total / FPS, 3)), 0]


def long_graph(m, scenes, width, height, seed, images, audio=None, start=0.0, pika=None, extra_loras=(), ref_size="match", vocals=None, preview=None, fresh=(), audio_cont=False, draft=False, videos=(), audios=()):
    """Scenes [(prompt, seconds), ...] rendered one after another; every next scene continues the last second of the
    previous one (latent cut + noise mask), with the same reference pictures, and optionally follows a music track.

    Singularity Ref2V + Pika H3 Flex Turbo (8 steps), as chosen for the app. Without `audio` the model makes its own sound
    for every scene (voices and music are written in the prompts).
    """
    g, model, sched, sampler_name, steps = _long_base(m, pika, extra_loras, draft)
    model = _with_preview(g, model, preview)
    g["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler_name}}
    g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": sched, "steps": steps, "denoise": 1.0}}
    heard = ["11", 0]  # what the model hears; the finished video always carries the full track
    if audio:
        g["11"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        if vocals:  # voice only: helps lip sync, the music in the result stays complete
            g["12"] = {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": vocals}}
            g["13"] = {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["12", 0], "audio": ["11", 0]}}
            heard = ["13", 0]
    stages = {n: "load" for n in g}
    plan = long_plan([sc[1] for sc in scenes], fresh=fresh)
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    prev_lat = prev_img = prev_aud = None
    scene_of = {}
    for i, (sc, p) in enumerate(zip(scenes, plan)):
        prompt = sc[0]
        own = list(sc[2]) if len(sc) > 2 and sc[2] else []  # this scene's own references; none: the common ones are used
        total, ov = p["total"], p["ov"]
        ref = add("MiniMaxH3ReferenceToVideo", clip=["5", 0], vae=["6", 0], audio_vae=["7", 0], prompt=prompt,
                  width=width, height=height, length=total, ref_image_size=ref_size)
        _refs(g, ref, own or images, videos=videos, audios=audios, base=1000 + 40 * i)
        new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
        voice = sc[3] if len(sc) > 3 and sc[3] else None   # this scene has the user's own voice: the model hears it and speaks with it
        if voice:
            enc = add("VAEEncodeAudio", audio=_scene_voice(add, voice, ov / FPS, total), vae=["7", 0])
            amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
            a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
        elif audio:
            ptr = start + (p["acc_before"] - ov) / FPS if i else start
            trim = add("TrimAudioDuration", audio=heard, start_index=round(ptr, 3), duration=round(total / FPS, 3))
            enc = add("VAEEncodeAudio", audio=[trim, 0], vae=["7", 0])
            amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
            a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
        else:
            a_lat = [new, 1]
        if i == 0 or ov == 0:   # the first scene, or a new shot: starts from the references only
            v_lat = [new, 0]
        else:
            old = add("LTXVSeparateAVLatent", av_latent=prev_lat)
            t_ov, t_tot = _tokens(ov), _tokens(total)
            cut = add("LatentCut", samples=[old, 0], dim="t", index=-t_ov, amount=t_ov)
            cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
            cut2 = add("LatentCut", samples=[cat, 0], dim="t", index=0, amount=t_tot)
            lw, lh = round(width / 16), round(height / 16)
            black = add("EmptyImage", width=lw, height=lh, batch_size=t_ov, color=0)
            white = add("EmptyImage", width=lw, height=lh, batch_size=max(1, t_tot - t_ov), color=16777215)
            both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
            vmask = add("ImageToMask", image=[both, 0], channel="red")
            v_lat = [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[vmask, 0]), 0]
            if audio_cont and not audio and not voice:   # the sound made by the model continues the sound of the previous scene (no jump at the join)
                a_lat = _audio_continue(add, [old, 1], [new, 1], ov, total)
        lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
        noise = add("RandomNoise", noise_seed=seed + 100 * i)
        guider = add("BasicGuider", model=model, conditioning=[ref, 0])
        smp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["9", 0], sigmas=["10", 0],
                  latent_image=[lat, 0])
        img = add("VAEDecode", samples=[smp, 1], vae=["6", 0])
        if i == 0:
            all_img = [img, 0]
        else:
            keep = add("ImageFromBatch", image=prev_img, batch_index=0, length=p["acc_before"] - ov)
            all_img = [add("ImageBatch", image1=[keep, 0], image2=[img, 0]), 0]
        if not audio:
            aud = add("VAEDecodeAudio", samples=[smp, 1], vae=["7", 0])
            if i == 0:
                all_aud = [aud, 0]
            else:
                kdur = round((p["acc_before"] - ov) / FPS, 3)
                kept = add("TrimAudioDuration", audio=prev_aud, start_index=0.0, duration=kdur)
                a_out = _audio_fade(add, [kept, 0], kdur, tail=True)           # the old scene dies away...
                a_in = _audio_fade(add, [aud, 0], round(total / FPS, 3), head=True)   # ...the new one swells up
                all_aud = [add("AudioConcat", audio1=a_out, audio2=a_in, direction="after"), 0]
            prev_aud = all_aud
            stages[aud] = "decode"
        stages.update({ref: "encode", smp: "sample", img: "decode"})
        scene_of[smp] = i
        prev_lat, prev_img = [smp, 1], all_img
    if audio:
        final_a = add("TrimAudioDuration", audio=["11", 0], start_index=float(start), duration=round(plan[-1]["acc"] / FPS, 3))
        final_a = [final_a, 0]
    else:
        final_a = prev_aud
    vid = add("CreateVideo", images=prev_img, audio=final_a, fps=float(FPS), bit_depth="auto", color_space="sRGB", codec="none")
    save = add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long", format="auto", **{"format.codec": "auto"})
    # the tail of the last scene's latent: lets the finished video be continued with one more scene later
    sep2 = add("LTXVSeparateAVLatent", av_latent=prev_lat)
    tail = add("LatentCut", samples=[sep2, 0], dim="t", index=-LONG_STATE_TOKENS, amount=LONG_STATE_TOKENS)
    keep = add("SaveLatent", samples=[tail, 0], filename_prefix="H3Workspace/long_state")
    stages.update({vid: "save", save: "save", sep2: "save", tail: "save", keep: "save"})
    stages["__scenes__"] = {"n": len(scenes), "map": scene_of}   # progress text "scene 2 of 4"
    return g, stages


ANCHOR_TOKENS = 6   # tokens at the end of a redone scene kept as they were (~0.8 s): enough to keep the pose and the motion at the join; the rest is free for the new prompt


def can_anchor(total, ov, next_ov):
    """Is the scene long enough to hold the frozen end of its old version: the tokens of the next scene's overlap at the end, those of its own
    overlap at the start (`ov` = 0: it starts a shot) and a free middle for the new action."""
    return next_ov > 0 and _tokens(total) - (_tokens(ov) if ov else 0) - min(ANCHOR_TOKENS, _tokens(next_ov)) >= 4


LONG_STATE_TOKENS = 17   # latent tokens of the last second(s) of a scene that the next scene continues from


def long_part_graph(m, scenes, k, width, height, seed, images, audio=None, start=0.0, pika=None, extra_loras=(),
                    ref_size="match", vocals=None, state=None, preview=None, src=None, cont_plan=False, fresh=(), draft=False, videos=(), quiet=False, audios=(),
                    end_state=None):
    """Scene number `k` (0-based) of the long video as its own job. `end_state`: the saved tail of the OLD version of this scene: its end is
    kept as it was (frozen in the latent), so the next scene, made from that tail, still joins this one when only this scene is redone. Scene 0 starts fresh; scene k > 0 continues from the saved
    latent tail of scene k-1 (`state`, a .latent file in ComfyUI's input folder). The clip it saves has the overlap with the
    previous scene cut off, so the clips join end to end. Also saves the tail of its own latent for scene k+1."""
    g, model, sched, sampler_name, steps = _long_base(m, pika, extra_loras, draft)
    model = _with_preview(g, model, preview)
    g["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler_name}}
    g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": sched, "steps": steps, "denoise": 1.0}}
    heard = ["11", 0]
    if audio:
        g["11"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        if vocals:
            g["12"] = {"class_type": "MelBandRoFormerModelLoader", "inputs": {"model_name": vocals}}
            g["13"] = {"class_type": "MelBandRoFormerSampler", "inputs": {"model": ["12", 0], "audio": ["11", 0]}}
            heard = ["13", 0]
    stages = {n: "load" for n in g}
    cont = bool(src) and k == 0   # the first scene continues a video the user already has (src = (file in ComfyUI input, frames))
    plan = long_plan([sc[1] for sc in scenes], cont_first=bool(src) or cont_plan, fresh=fresh)   # a project that continues an existing video
    p, sc = plan[k], scenes[k]
    new_shot = k > 0 and k < len(fresh) and bool(fresh[k])   # does not continue scene k-1
    total, ov = p["total"], p["ov"]
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    ref = add("MiniMaxH3ReferenceToVideo", clip=["5", 0], vae=["6", 0], audio_vae=["7", 0], prompt=sc[0],
              width=width, height=height, length=total, ref_image_size=ref_size)
    own = list(sc[2]) if len(sc) > 2 and sc[2] else []   # this scene's own references; none: the common ones are used
    _refs(g, ref, own or images, videos=videos, audios=audios, base=100)
    new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
    voice = sc[3] if len(sc) > 3 and sc[3] else None   # the user's own voice recording for THIS scene (file in ComfyUI input)
    vsrc = _scene_voice(add, voice, ov / FPS if (cont or (k > 0 and not new_shot)) else 0.0, total) if voice else None
    if voice:
        enc = add("VAEEncodeAudio", audio=vsrc, vae=["7", 0])
        amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
        a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
    elif audio:
        ptr = max(0.0, start - ov / FPS) if cont else start + ((p["acc_before"] - ov) / FPS if k else 0)
        if quiet:   # nobody sings in this scene (backing vocals, ad-libs, a break): the model hears silence, so it has nothing to lip-sync to
            trim = add("EmptyAudio", duration=round(total / FPS, 3), sample_rate=48000, channels=2)
        else:
            trim = add("TrimAudioDuration", audio=heard, start_index=round(ptr, 3), duration=round(total / FPS, 3))
        enc = add("VAEEncodeAudio", audio=[trim, 0], vae=["7", 0])
        amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
        a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
    else:
        a_lat = [new, 1]
    t_ov, t_tot = _tokens(ov), _tokens(total)
    next_ov = plan[k + 1]["ov"] if k + 1 < len(plan) else 0
    t_end = min(ANCHOR_TOKENS, _tokens(next_ov)) if next_ov else 0
    anchor = bool(end_state) and can_anchor(total, 0 if ((k == 0 and not cont) or new_shot) else ov, next_ov)

    def with_end(front, n_front, black_front):
        """`front` (a latent of n_front tokens, the first black_front of them kept) + the frozen end of the old version: t_tot tokens in all."""
        lw, lh = round(width / 16), round(height / 16)
        if anchor:
            fin = add("LoadLatent", latent=end_state)
            stages[fin] = "load"
            fcut = add("LatentCut", samples=[fin, 0], dim="t", index=-t_end, amount=t_end)   # the same tokens the next scene continues from
            head = add("LatentCut", samples=front, dim="t", index=0, amount=t_tot - t_end)
            lat_v = [add("LatentConcat", samples1=[head, 0], samples2=[fcut, 0], dim="t"), 0]
        else:
            lat_v = [add("LatentCut", samples=front, dim="t", index=0, amount=t_tot), 0]
        parts = []
        if black_front:
            parts.append(add("EmptyImage", width=lw, height=lh, batch_size=black_front, color=0))
        parts.append(add("EmptyImage", width=lw, height=lh, batch_size=max(1, t_tot - black_front - (t_end if anchor else 0)), color=16777215))
        if anchor:
            parts.append(add("EmptyImage", width=lw, height=lh, batch_size=t_end, color=0))
        both = parts[0]
        for q in parts[1:]:
            both = add("ImageBatch", image1=[both, 0], image2=[q, 0])
        vmask = add("ImageToMask", image=[both, 0], channel="red")
        return [add("SetLatentNoiseMask", samples=lat_v, mask=[vmask, 0]), 0]

    if (k == 0 and not cont) or new_shot:
        v_lat = with_end([new, 0], t_tot, 0) if anchor else [new, 0]
    else:
        if cont:   # the tail of the existing video, encoded back into a latent
            vid_in = add("LoadVideo", file=src[0])
            comp = add("GetVideoComponents", video=[vid_in, 0])
            last = add("ImageFromBatch", image=[comp, 0], batch_index=max(0, src[1] - ov), length=ov)
            fit = add("ImageScale", image=[last, 0], upscale_method="lanczos", width=width, height=height, crop="center")
            old = add("VAEEncode", pixels=[fit, 0], vae=["6", 0])
            stages.update({vid_in: "encode", comp: "encode", last: "encode", fit: "encode", old: "encode"})
        else:
            old = add("LoadLatent", latent=state)
        cut = add("LatentCut", samples=[old, 0], dim="t", index=-t_ov, amount=t_ov)
        cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
        v_lat = with_end([cat, 0], t_tot, t_ov)
        stages[old] = "load"
    lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
    noise = add("RandomNoise", noise_seed=seed + 100 * k)
    guider = add("BasicGuider", model=model, conditioning=[ref, 0])
    smp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["9", 0], sigmas=["10", 0],
              latent_image=[lat, 0])
    img = add("VAEDecode", samples=[smp, 1], vae=["6", 0])
    clip_len = total - ov
    frames_out = [img, 0] if ((not k and not cont) or new_shot) else [add("ImageFromBatch", image=[img, 0], batch_index=ov, length=clip_len), 0]
    if voice:
        aud = [add("TrimAudioDuration", audio=vsrc, start_index=round(ov / FPS if (cont or (k > 0 and not new_shot)) else 0.0, 3), duration=round(clip_len / FPS, 3)), 0]
    elif audio:
        a0 = start + (p["acc_before"] / FPS if k else 0)
        aud = [add("TrimAudioDuration", audio=["11", 0], start_index=round(a0, 3), duration=round(clip_len / FPS, 3)), 0]
    else:
        dec = add("VAEDecodeAudio", samples=[smp, 1], vae=["7", 0])
        stages[dec] = "decode"
        aud = [dec, 0] if ((not k and not cont) or new_shot) else [add("TrimAudioDuration", audio=[dec, 0], start_index=round(ov / FPS, 3),
                                                          duration=round(clip_len / FPS, 3)), 0]
        # soft edges: this clip is joined to the others later, a jump of sound at the join is hidden by a short fade
        aud = _audio_fade(add, aud, round(clip_len / FPS, 3), head=bool(k or cont), tail=True)
    vid = add("CreateVideo", images=frames_out, audio=aud, fps=float(FPS), bit_depth="auto", color_space="sRGB", codec="none")
    save = add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long_part", format="auto", **{"format.codec": "auto"})
    sep2 = add("LTXVSeparateAVLatent", av_latent=[smp, 1])
    tail = add("LatentCut", samples=[sep2, 0], dim="t", index=-LONG_STATE_TOKENS, amount=LONG_STATE_TOKENS)
    keep = add("SaveLatent", samples=[tail, 0], filename_prefix="H3Workspace/long_state")
    stages.update({ref: "encode", smp: "sample", img: "decode", vid: "save", save: "save", sep2: "save", tail: "save", keep: "save"})
    stages["__scenes__"] = {"n": len(scenes), "map": {smp: k}}
    return g, stages


ACTION_LOWRES = 0.6  # SelfLift lowres_scale in the original


def action_sizes(width, height):
    """(low-res w, h, full-res w, h) of the action route: pass 1 at 0.6x, the learned upscaler brings it back (aligned to 32)."""
    s = round(1 / ACTION_LOWRES, 3)
    lw, lh = r32(width * ACTION_LOWRES), r32(height * ACTION_LOWRES)
    return lw, lh, max(32, round(lw * s / 32) * 32), max(32, round(lh * s / 32) * 32)


def action_part_graph(m, scenes, k, width, height, seed, images, loras, preview=None, state=None, fresh=(), faithful=False,
                      refiner_args=None, single=False, sing=False, steps=12, audio=None, start=0.0, vocals=None, audios=(), videos=()):
    """ACTION scene number `k` (0-based) of a scene-by-scene video, the same idea as long_part_graph (scene k continues the saved
    latent tail of scene k-1, the clip is cut off at the overlap) but with the action model: hybrid, DMAD + Speed Slider,
    8 steps at 0.6x size, learned latent upscale, 4 steps at full size. The sound is made by the model, unless `audio` (the user's
    own track, as in the Video studio: the piece of it that belongs to this scene is locked in the audio latent and put into the clip;
    only in the one-pass route `single`, which is what the page uses).

    The tail of the previous scene (full size) is decoded, scaled down and encoded again to continue pass 1 at low size; before
    pass 2 the true full-size tail is put back in front of the upscaled new part, so the join stays sharp.

    faithful=True: the exact scheme of the original workflow (SelfLiftH3Sampler + H3SigmaRefiner, cleaner picture). The sampler
    takes the full-size latent with the tail of the previous scene and a noise mask, so there is no separate low-size pass."""
    plan = long_plan([sc[1] for sc in scenes], fresh=fresh)
    p, sc = plan[k], scenes[k]
    new_shot = k > 0 and k < len(fresh) and bool(fresh[k])
    total, ov = p["total"], p["ov"]
    cont = k > 0 and not new_shot
    lw, lh, fw, fh = action_sizes(width, height)
    if faithful or single:
        lw, lh, fw, fh = width, height, width, height
    g = {
        # `sing`: the model of the Video studio (Singularity Ref2V) under the same action modules (LoRAs, sparse attention, shifts)
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet" if sing else "action_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    model = _lora_chain(g, ["1", 0], loras)
    g.update({
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
    })
    mdl = ["7", 0]
    if preview:
        g["30"] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {
            "model": ["7", 0], "max_resolution": 1024, "jpeg_quality": 80, "suppress_default_preview": True,
            "preview_frames": 1024, "preview_fps": 12, "tiny_vae": preview.get("tiny_vae", "none"), "vae": ["5", 0]}}
        mdl = ["30", 0]
    g.update({
        "10": {"class_type": "BasicScheduler", "inputs": {"model": mdl, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
    })
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    ref = add("MiniMaxH3ReferenceToVideo", clip=["4", 0], vae=["5", 0], audio_vae=["6", 0], prompt=sc[0],
              width=lw, height=lh, length=total, ref_image_size="max")
    own = list(sc[2]) if len(sc) > 2 and sc[2] else []
    _refs(g, ref, own or images, videos=videos, audios=audios, base=100)   # RefMod videos of the cards (<Video N>); up to 3 audio samples (voice, music, atmosphere): the model listens, the sound is still its own
    new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
    stages = {}
    old_full = None
    if not cont:
        v_lat = [new, 0]
    else:
        old_full = add("LoadLatent", latent=state)
        if faithful or single:   # full-size tail goes in as it is
            old_low = old_full
        else:
            dec = add("VAEDecode", samples=[old_full, 0], vae=["5", 0])
            n_dec = round((LONG_STATE_TOKENS - 2) / 5 * 17 + 5)   # frames in the saved tail
            last = add("ImageFromBatch", image=[dec, 0], batch_index=max(0, n_dec - ov), length=ov)
            fit = add("ImageScale", image=[last, 0], upscale_method="lanczos", width=lw, height=lh, crop="center")
            old_low = add("VAEEncode", pixels=[fit, 0], vae=["5", 0])
            stages.update({n: "encode" for n in (dec, last, fit)})
        t_ov, t_tot = _tokens(ov), _tokens(total)
        cut = add("LatentCut", samples=[old_low, 0], dim="t", index=-t_ov, amount=t_ov)
        cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
        cut2 = add("LatentCut", samples=[cat, 0], dim="t", index=0, amount=t_tot)
        black = add("EmptyImage", width=round(lw / 16), height=round(lh / 16), batch_size=t_ov, color=0)
        white = add("EmptyImage", width=round(lw / 16), height=round(lh / 16), batch_size=max(1, t_tot - t_ov), color=16777215)
        both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
        vmask = add("ImageToMask", image=[both, 0], channel="red")
        if faithful:   # SelfLift wants a mask of shape [1, 1, T, H, W]; this node (ComfyUI-LTXVideo) builds it from one mask per latent frame
            v_lat = [add("LTXVSetVideoLatentNoiseMasks", samples=[cut2, 0], masks=[vmask, 0]), 0]
        else:
            v_lat = [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[vmask, 0]), 0]
        stages.update({n: "encode" for n in (old_full, old_low)})
    own_audio = bool(audio) and single and not faithful
    a_lat = [new, 1]
    if own_audio:   # the same way as long_part_graph: this scene's piece of the track is encoded and masked out of the noise
        src = [add("LoadAudio", audio=audio), 0]
        heard = src
        if vocals:
            ml = add("MelBandRoFormerModelLoader", model_name=vocals)
            heard = [add("MelBandRoFormerSampler", model=[ml, 0], audio=src), 0]
        ptr = start + ((p["acc_before"] - ov) / FPS if k else 0)
        trim = add("TrimAudioDuration", audio=heard, start_index=round(max(0.0, ptr), 3), duration=round(total / FPS, 3))
        enc = add("VAEEncodeAudio", audio=[trim, 0], vae=["6", 0])
        amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
        a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
    lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
    noise = add("RandomNoise", noise_seed=seed + 100 * k)
    if faithful:
        neg = add("ConditioningZeroOut", conditioning=[ref, 0])
        refiner = add("H3SigmaRefiner", sigmas=["10", 0], **(refiner_args or {"extra_steps": 1, "strength": 0.7, "offset": 0.0, "curve": "cosine"}))
        # the sampler cannot tile the high-resolution stage together with a noise mask: only the first scene may use the tiling
        smp = add("SelfLiftH3Sampler", model=mdl, positive=[ref, 0], negative=[neg, 0], vae=["5", 0], latent_image=[lat, 0],
                  sampler=["12", 0], sigmas=[refiner, 0], seed=seed + 100 * k, cfg=1.0, transition_step=8, lowres_scale=ACTION_LOWRES,
                  rho=0.0, w_min=0.5, w_max=1.0, upscaler_model=m["action_upscaler"], highres_tiling=not cont, upscaler_unload=True)
    elif single:   # the way of the Video studio: one pass at full size, the tail of the previous scene kept by the noise mask
        guider = add("BasicGuider", model=mdl, conditioning=[ref, 0])
        smp1 = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["12", 0], sigmas=["10", 0], latent_image=[lat, 0])
    else:
        guider = add("BasicGuider", model=mdl, conditioning=[ref, 0])
        smp1 = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["12", 0], sigmas=["9", 0], latent_image=[lat, 0])
        sep = add("LTXVSeparateAVLatent", av_latent=[smp1, 1])
        up = add("MinimaxH3LatentUpscaler3D", latent=[sep, 0], model_name=m["upscaler"], mode="scale by multiplier",
                 **{"mode.scale": round(1 / ACTION_LOWRES, 3)}, align=32, enable_temporal_chunking=True,
                 force_unload=True, device="cuda", precision="fp16")
    if faithful:
        final, smp_stages = [smp, 0], {smp: "sample", refiner: "encode"}
    elif single:
        final, smp_stages = [smp1, 1], {smp1: "sample"}
    elif cont:   # the sharp full-size tail of the previous scene in front of the new, upscaled part
        t_ov, t_tot = _tokens(ov), _tokens(total)
        tail_f = add("LatentCut", samples=[old_full, 0], dim="t", index=-t_ov, amount=t_ov)
        part = add("LatentCut", samples=[up, 0], dim="t", index=t_ov, amount=t_tot - t_ov)
        catf = add("LatentConcat", samples1=[tail_f, 0], samples2=[part, 0], dim="t")
        black = add("EmptyImage", width=round(fw / 16), height=round(fh / 16), batch_size=t_ov, color=0)
        white = add("EmptyImage", width=round(fw / 16), height=round(fh / 16), batch_size=max(1, t_tot - t_ov), color=16777215)
        both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
        vmask = add("ImageToMask", image=[both, 0], channel="red")
        v2 = [add("SetLatentNoiseMask", samples=[catf, 0], mask=[vmask, 0]), 0]
    else:
        v2 = [up, 0]
    if not faithful and not single:
        lat2 = add("LTXVConcatAVLatent", video_latent=v2, audio_latent=[sep, 1])
        smp2 = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["12", 0], sigmas=["9", 1], latent_image=[lat2, 0])
        final = [smp2, 1]
        smp_stages = {smp1: "sample", sep: "upscale", up: "upscale", smp2: "sample2"}
    img = add("VAEDecode", samples=final, vae=["5", 0])
    clip_len = total - ov
    frames_out = [img, 0] if not cont else [add("ImageFromBatch", image=[img, 0], batch_index=ov, length=clip_len), 0]
    dec_a = None
    if own_audio:   # the user's track itself goes into the clip (the part of it that this scene shows)
        a0 = start + (p["acc_before"] / FPS if k else 0)
        aud = [add("TrimAudioDuration", audio=src, start_index=round(a0, 3), duration=round(clip_len / FPS, 3)), 0]
    else:
        dec_a = add("VAEDecodeAudio", samples=final, vae=["6", 0])
        aud = [dec_a, 0] if not cont else [add("TrimAudioDuration", audio=[dec_a, 0], start_index=round(ov / FPS, 3),
                                                  duration=round(clip_len / FPS, 3)), 0]
        aud = _audio_fade(add, aud, round(clip_len / FPS, 3), head=cont, tail=True)
    vid = add("CreateVideo", images=frames_out, audio=aud, fps=float(FPS), bit_depth="auto", color_space="sRGB", codec="none")
    save = add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long_part", format="auto", **{"format.codec": "auto"})
    sep2 = add("LTXVSeparateAVLatent", av_latent=final)
    tail = add("LatentCut", samples=[sep2, 0], dim="t", index=-LONG_STATE_TOKENS, amount=LONG_STATE_TOKENS)
    keep = add("SaveLatent", samples=[tail, 0], filename_prefix="H3Workspace/long_state")
    stages.update({n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [x for x in g if x.isdigit() and 60 <= int(x) < 70]})
    if preview:
        stages["30"] = "load"
    stages.update({ref: "encode", img: "decode", vid: "save", save: "save", sep2: "save", tail: "save", keep: "save"})
    if dec_a:
        stages[dec_a] = "decode"
    stages.update(smp_stages)
    stages["__scenes__"] = {"n": len(scenes), "map": {n: k for n, s in smp_stages.items() if s.startswith("sample")}}   # progress text "scene 2 of 4"
    return g, stages


def action_graph(m, prompt, width, height, seconds, seed, images, audios, faithful, loras, preview=None, videos=()):
    """PM Minimax H3 Best Action Workflow (Prompt Mastery): hybrid model, sol-attn, 8 low-res + 4 high-res steps.

    faithful=True runs the original SelfLiftH3Sampler + H3SigmaRefiner; otherwise the same schedule is
    done with built-in nodes: pass 1 at 0.6x, learned latent upscale, pass 2 at full size.
    width/height are the final size.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["action_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    model = _lora_chain(g, ["1", 0], loras)  # dmad 0.8, speed slider 4 in the original
    g.update({
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
    })
    mdl = ["7", 0]
    if preview:   # the animated live preview of the original workflow: sits right after the sigma shift, everything below uses it
        g["30"] = {"class_type": "ModelPreviewOverrideKJ", "inputs": {
            "model": ["7", 0], "max_resolution": 1024, "jpeg_quality": 80, "suppress_default_preview": True,
            "preview_frames": 1024, "preview_fps": 12, "tiny_vae": preview.get("tiny_vae", "none"), "vae": ["5", 0]}}
        mdl = ["30", 0]
    g.update({
        "10": {"class_type": "BasicScheduler", "inputs": {"model": mdl, "scheduler": "simple", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    })
    if faithful:
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": width, "height": height, "length": frames(seconds), "ref_image_size": "max"}},
            "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
            "13": {"class_type": "H3SigmaRefiner", "inputs": {
                "sigmas": ["10", 0], "extra_steps": 1, "strength": 0.7, "offset": 0.0, "curve": "cosine"}},
            "14": {"class_type": "SelfLiftH3Sampler", "inputs": {
                "model": mdl, "positive": ["8", 0], "negative": ["9", 0], "vae": ["5", 0],
                "latent_image": ["8", 1], "sampler": ["12", 0], "sigmas": ["13", 0], "seed": seed, "cfg": 1.0,
                "transition_step": 8, "lowres_scale": ACTION_LOWRES, "rho": 0.0, "w_min": 0.5, "w_max": 1.0,
                "upscaler_model": m["action_upscaler"], "highres_tiling": True,
                "upscaler_unload": True}},   # as in the original workflow: the upscaler leaves the video memory after use
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample"}
    else:
        lw, lh = r32(width * ACTION_LOWRES), r32(height * ACTION_LOWRES)
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": lw, "height": lh, "length": frames(seconds), "ref_image_size": "max"}},
            "13": {"class_type": "BasicGuider", "inputs": {"model": mdl, "conditioning": ["8", 0]}},
            "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
            "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}},
            "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 1]}},
            "16": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
                "latent": ["15", 0], "model_name": m["upscaler"], "mode": "scale by multiplier",
                "mode.scale": round(1 / ACTION_LOWRES, 3), "align": 32, "enable_temporal_chunking": True,
                "force_unload": True, "device": "cuda", "precision": "fp16"}},
            "17": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["16", 0], "audio_latent": ["15", 1]}},
            "18": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 1], "latent_image": ["17", 0]}},
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 1], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["18", 1], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample", "15": "upscale", "16": "upscale", "17": "upscale", "18": "sample2"}
    g.update({
        "22": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["21", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "23": {"class_type": "SaveVideo", "inputs": {
            "video": ["22", 0], "filename_prefix": "H3Workspace/action", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, videos, audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    if preview:
        stages["30"] = "load"
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update(sample_stages)
    stages.update({"20": "decode", "21": "decode", "22": "save", "23": "save"})
    return g, stages


def lipsync_scale(quality):
    return 2.0 if QUALITY.get(quality, 0) >= 1824 * 1024 else 1.5


def lipsync_prompt(text, n_refs, n_videos=0, cam_videos=(), n_audio=0, audio_labels=(), audio_start=0):
    """Wrap free text into H3's structured prompt unless the user already wrote one.
    Reference videos are tagged <Video N> as the official workflow asks; with pictures present the face comes from the pictures.
    A text that starts with RefMod definitions ("<Subject 1> is Name ... in <Video 1>. <Audio 1> is the voice-timbre reference ...")
    keeps them as their own first paragraph: those heroes are NOT shown as pictures at 0.00 seconds (that made a face fill the frame)."""
    if "integrated_multimodal_description" in text:
        return text
    defs = ""
    if text.startswith("<Subject "):
        defs, _, text = text.partition("\n\n")
        defs += "\n\n"
    refs = " ".join(f"<Picture {i}> (from [Shot 1]) is fully referenced." for i in range(1, n_refs + 1))
    if n_videos:
        vids = ", ".join(f"<Video {i}>" for i in range(1, n_videos + 1))
        refs += f" {vids} {'is' if n_videos == 1 else 'are'} referenced for motion, camera and setting only."
        if n_refs:
            refs += " The faces and identities must match the reference pictures exactly."
    if cam_videos:   # the user's own camera / motion reference videos (numbered after the Reference Mode videos)
        cv = ", ".join(f"<Video {i}>" for i in cam_videos)
        refs += f" {cv} {'is' if len(cam_videos) == 1 else 'are'} referenced for camera movement and motion only; the characters and the place come from the reference pictures."
    if n_audio:   # voice / sound references
        av = ", ".join(f"<Audio {i}>" for i in range(audio_start + 1, audio_start + n_audio + 1))
        refs += f" {av} {'is' if n_audio == 1 else 'are'} referenced for the voice timbre and the sound only, not for the words."
        who = {"man": "the man", "woman": "the woman", "child": "the child"}
        tone = {"man": "male ", "woman": "female ", "child": "child's "}
        labs = list(audio_labels) + [""] * max(0, n_audio - len(audio_labels))
        # the voice is tied to the speaker: (S1) speaks with the timbre heard in <Audio N> (the way the H3 guide asks for a voice reference); only timbre and manner, new words
        bind = [f"{who.get(l, 'the speaker')} (S{k}) speaks with the exact {tone.get(l, '')}voice timbre, accent and manner heard in <Audio {audio_start + k}>"
                for k, l in enumerate(labs[:n_audio], 1)]
        bind = [b[0].upper() + b[1:] if k == 0 else b for k, b in enumerate(bind)]
        refs += " " + "; ".join(bind) + ". Copy only the voice timbre and manner, do not play or reuse the audio, generate new speech with the words written below, spoken aloud and clearly audible."
        if n_audio > 1:   # a dialogue
            refs += " Each character speaks only with their own voice."
    head = f"For the target video, at 0.00 seconds into the target video, {refs}\n\n" if refs.strip() else ""
    sound, music, keep = "N/A", "N/A", []
    for ln in text.split("\n"):   # the agent writes the sound of the scene on its own lines: they go to their own fields
        m = re.match(r"\s*(overall_soundscape|non_diegetic_music)\s*:\s*(.+)", ln, re.I)
        if m:
            if m.group(1).lower() == "overall_soundscape":
                sound = m.group(2).strip()
            else:
                music = m.group(2).strip()
        else:
            keep.append(ln)
    text = "\n".join(keep).strip()
    return (defs + head + f"integrated_multimodal_description: [Shot 1] {text}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")


def lipsync_graph(m, prompt, width, height, seconds, seed, audio, ref_images, scale, ref_size="max", extra_loras=(), videos=()):
    """MiniMax H3 Singularity Ref2V + Audio Lock, two passes.

    width/height are pass 1 size; pass 2 upscales the latent by `scale`.
    `audio` is the already trimmed clip; the video length follows it.
    ref_size: "max" (best identity, several times slower) or "match" (refs scaled to the video area).
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        # the original workflow uses KJNodes' SageAttention patch; the built-in INT8 attention
        # needs no extra install (sageattention is hard to set up on Windows)
        "2": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": m["ref2v_turbo_lora"], "strength_model": 0.55}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
        "9": {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
            "model": ["1", 0], "av_latent": ["8", 1], "audio_vae": ["6", 0], "audio": ["7", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {
            "model": ["9", 0], "scheduler": "beta", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["8", 0]}},
        # pass 1
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 0], "latent_image": ["9", 1]}},
        # pass 2: upscale the video latent, keep the audio latent, finish the low sigmas
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 1]}},
        "17": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": ["16", 0], "model_name": m["upscaler"],
            "mode": "scale by multiplier", "mode.scale": scale, "align": 32,
            "enable_temporal_chunking": True, "force_unload": True,
            "device": "cuda", "precision": "fp16"}},
        "18": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["17", 0], "audio_latent": ["16", 1]}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 1], "latent_image": ["18", 0]}},
        "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 1], "vae": ["5", 0]}},
        # the saved video carries the original audio, not a regenerated one
        "21": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["9", 2], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "22": {"class_type": "SaveVideo", "inputs": {
            "video": ["21", 0], "filename_prefix": "H3Workspace/lipsync",
            "format": "auto", "format.codec": "auto"}},
    }
    for i, image in enumerate(ref_images[:9]):
        node_id = str(30 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["8"]["inputs"][f"ref_images.ref_image_{i}"] = [node_id, 0]
    if videos:   # RefMod videos of the cards: the heroes as <Video N>
        _refs(g, "8", [], videos, (), base=40)
    if extra_loras:  # optional LoRAs go in front of the speed LoRA, as in the original workflow
        g["3"]["inputs"]["model"] = _lora_chain(g, ["2", 0], list(extra_loras))

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "load" for n in g if n.isdigit() and 60 <= int(n) < 70})
    stages.update({n: "encode" for n in ["8", "9"] + [str(30 + i) for i in range(9)]})
    stages.update({"15": "sample", "16": "upscale", "17": "upscale", "18": "upscale",
                   "19": "sample2", "20": "decode", "21": "save", "22": "save"})
    return g, stages


# ---------- pictures: Krea 2 (text) and Qwen Image 2.1 (edit, character sheet) ----------

# Krea 2 works at about 1 MP only
KREA_SIZES = {"16:9": (1376, 768), "9:16": (768, 1376), "1:1": (1024, 1024), "4:3": (1184, 896),
              "3:4": (896, 1184), "3:2": (1248, 832), "2:3": (832, 1248), "21:9": (1568, 672)}
QWEN_MP = {"1": 1.0, "2": 2.0}
SHEET_SIZE = (1920, 1088)
SHEET6_SIZE = (2112, 960)  # six tall panels in one row, about 2 MP like the four-panel sheet
SHEET_BG = {"grey": "light grey", "white": "white", "black": "black"}


def krea_graph(m, prompt, width, height, seed):
    """Krea 2 Turbo, text to picture: 10 steps, CFG 1, no negative prompt (the model has none)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["krea_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 10, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea", "images": ["8", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "encode", "5": "encode", "6": "encode",
              "7": "sample", "8": "decode", "9": "save"}
    return g, stages


KREA_REAL_SCALES = {"2k": 1.5, "4k": 2.5}


def krea_real_graph(m, prompt, width, height, seed, scale):
    """Krea 2 realism (RedCraft RedMix + Realism Engine LoRA 0.5), two passes: draw ~1 MP, enlarge the latent, refine with denoise 0.6."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["kreareal_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": m["kreareal_lora"], "strength_model": 0.5, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 1, "sampler_name": "euler_ancestral", "scheduler": "sgm_uniform", "denoise": 1,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "LatentUpscaleBy", "inputs": {"upscale_method": "bicubic", "scale_by": scale, "samples": ["8", 0]}},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 6, "cfg": 1, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 0.6,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea_real", "images": ["11", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "sample", "9": "upscale", "10": "sample2", "11": "decode", "12": "save"}
    return g, stages


def _qwen_base(m, image, megapixels, prompt, seed, size=None, extras=()):
    """The graph of the "One Image Edit" / "Character Sheet" workflows on plain ComfyUI nodes.
    `size` (width, height) gives an empty canvas of its own; otherwise the result keeps the picture's shape."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 32, "image": ["5", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0],
            "images.image_1": ["6", 0], "vae": ["3", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen", "images": ["10", 0]}},
    }
    for k, img in enumerate(list(extras)[:2], 2):  # more reference pictures: <image2>, <image3>
        a, b = f"12{k}", f"13{k}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 32, "image": [a, 0]}}
        g["7"]["inputs"][f"images.image_{k}"] = [b, 0]
    if size:
        g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
        g["9"]["inputs"]["latent_image"] = ["8", 0]
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "encode", "9": "sample", "10": "decode", "11": "save"}
    for k in (2, 3):
        stages[f"12{k}"] = stages[f"13{k}"] = "encode"
    return g, stages


H3IMG_MP = {"4": 4.0, "8": 8.0, "16": 16.0}   # the model is tuned for about 4 megapixels and up (2048x2048); best around 16
H3IMG_MAX_REFS = 5


def h3img_size(aspect, megapixels):
    """Width and height (multiples of 32) of the given aspect ratio and area in megapixels."""
    try:
        aw, ah = (float(x) for x in str(aspect).split(":"))
        assert aw > 0 and ah > 0
    except (ValueError, AssertionError):
        aw, ah = 1.0, 1.0
    px = megapixels * 1_000_000
    return r32(math.sqrt(px * aw / ah)), r32(math.sqrt(px * ah / aw))


def _h3img_base(m):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["h3img_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["h3img_vae"]}},
        "4": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
    }


def h3img_t2i_graph(m, prompt, width, height, seed):
    """MiniMax H3 Image, picture by text (the official t2i workflow): 6 steps, CFG 1, no negative prompt. No prompt enhancer: the
    prompt is written by the agent's skill."""
    g = _h3img_base(m)
    g.update({
        "5": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"prompt": prompt, "width": width, "height": height,
                                                                        "clip": ["2", 0], "vae": ["3", 0]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["5", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["4", 0], "scheduler": "beta", "steps": 6, "denoise": 1.0}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["6", 0], "guider": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/h3img", "images": ["11", 0]}},
    })
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode", "8": "encode",
              "9": "encode", "10": "sample", "11": "decode", "12": "save"}
    return g, stages


def h3img_edit_graph(m, prompt, images, megapixels, seed, size=None, ref_mp=None):
    """MiniMax H3 Image with reference pictures (up to H3IMG_MAX_REFS): <Picture 1> is the picture to edit (and gives the shape of the
    result unless `size` is given), the others are style / character / object references. Every picture is scaled to about
    `ref_mp` (default: `megapixels`); the size of the result follows the first picture scaled to `megapixels` (multiples of 32).
    A small `ref_mp` with a big `megapixels` gives a big clean result without feeding huge references to the model."""
    g = _h3img_base(m)
    ref_mp = ref_mp or megapixels
    stages = {"1": "load", "2": "load", "3": "load", "4": "load"}
    cond = {"prompt": prompt, "clip": ["2", 0], "vae": ["3", 0]}
    for i, name in enumerate(list(images)[:H3IMG_MAX_REFS]):
        lid, rid = f"2{i}0", f"2{i}1"
        g[lid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[rid] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "resize_type": "scale total pixels", "resize_type.megapixels": ref_mp, "scale_method": "area", "input": [lid, 0]}}
        last = rid
        if i == 0:
            src = rid
            if ref_mp != megapixels:   # the size of the result comes from the same picture at the bigger area
                g["h3img-edit-size-source"] = {"class_type": "ResizeImageMaskNode", "inputs": {
                    "resize_type": "scale total pixels", "resize_type.megapixels": megapixels, "scale_method": "area", "input": [lid, 0]}}
                src = "h3img-edit-size-source"
                stages["h3img-edit-size-source"] = "encode"
            g["h3img-edit-size-aligned"] = {"class_type": "ResizeImageMaskNode", "inputs": {
                "resize_type": "scale to multiple", "resize_type.multiple": 32, "scale_method": "area", "input": [src, 0]}}
            g["h3img-edit-size-dimensions"] = {"class_type": "GetImageSize", "inputs": {"image": ["h3img-edit-size-aligned", 0]}}
            last = "h3img-edit-size-aligned" if ref_mp == megapixels else rid   # the model gets the picture at `ref_mp`
            stages.update({"h3img-edit-size-aligned": "encode", "h3img-edit-size-dimensions": "encode"})
        cond[f"images.image_{i}"] = [last, 0]
        stages.update({lid: "encode", rid: "encode"})
    cond["width"], cond["height"] = (size[0], size[1]) if size else (["h3img-edit-size-dimensions", 0], ["h3img-edit-size-dimensions", 1])
    g.update({
        "5": {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["5", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/h3img_edit", "images": ["8", 0]}},
    })
    stages.update({"5": "encode", "6": "encode", "7": "sample", "8": "decode", "9": "save"})
    return g, stages


H3IMG_SHEET_SIZE = (3840, 2176)    # about 8 MP: at 4 MP (the lower limit of the model) the picture is grainy, 8-16 MP is clean
H3IMG_SHEET6_SIZE = (4352, 1984)
H3IMG_FRAME_SIZE = (3328, 1856)    # one storyboard frame, about 6 MP


def h3img_tags(text):
    """The prompts written for Qwen call the pictures <image1>, <image2>; MiniMax H3 Image calls them <Picture 1>, <Picture 2>."""
    return re.sub(r"<image\s*(\d)>", r"<Picture \1>", text)


def h3img_sheet_graph(m, image, seed, headless=True, background="grey", marks="", panels=4, mp=None, desc=""):
    """Character sheet from one picture on MiniMax H3 Image (the same sheet text as for Qwen, tags <Picture 1>).
    `mp`: area of the whole sheet in megapixels (the panels share it: a sheet needs 8-16 MP to look clean)."""
    prompt = h3img_tags(sheet_prompt(headless, background, marks, panels, desc))
    size = h3img_size("11:5" if panels == 6 else "16:9", mp) if mp else (H3IMG_SHEET6_SIZE if panels == 6 else H3IMG_SHEET_SIZE)
    g, stages = h3img_edit_graph(m, prompt, [image], 2.0, seed, size)
    return g, stages, prompt


def h3img_story_graph(m, images, prompt, seed, size=H3IMG_SHEET_SIZE):
    """Storyboard as ONE picture from 1-5 pictures (character, location...): <Picture 1>, <Picture 2> in the prompt."""
    return h3img_edit_graph(m, prompt, images, 2.0, seed, size)


def h3img_frames_graph(m, refs, prompts, seed, size=H3IMG_FRAME_SIZE, line="white"):
    """The frame-by-frame storyboard on MiniMax H3 Image: every frame is its own picture, the sheet is glued from them."""
    g = _h3img_base(m)
    stages = {"1": "load", "2": "load", "3": "load", "4": "load"}
    scaled = []
    for i, img in enumerate(list(refs)[:H3IMG_MAX_REFS]):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "resize_type": "scale total pixels", "resize_type.megapixels": 2.0, "scale_method": "area", "input": [a, 0]}}
        scaled.append(b)
        stages[a] = stages[b] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        en, zr, ks, dc, sv = (str(base + k) for k in range(5))
        cond = {"prompt": text, "width": size[0], "height": size[1], "clip": ["2", 0], "vae": ["3", 0]}
        for k, b in enumerate(scaled):
            cond[f"images.image_{k}"] = [b, 0]
        g[en] = {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond}
        g[zr] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [en, 0]}}
        g[ks] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": [en, 0], "negative": [zr, 0], "latent_image": [en, 1]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [ks, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({en: "encode", zr: "encode", ks: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def flux_t2i_graph(m, prompt, width, height, seed):
    """FLUX.2 klein 4B (distilled), picture by text: 4 steps, CFG 1, no negative prompt."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["6", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["7", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_t2i", "images": ["12", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "encode", "5": "encode", "6": "encode", "7": "encode",
              "8": "encode", "9": "encode", "10": "encode", "11": "sample", "12": "decode", "13": "save"}
    return g, stages


def qwen_t2i_graph(m, prompt, negative, width, height, seed):
    """Qwen Image 2.1, picture by text (no reference). With a negative prompt CFG goes above 1 so that it works."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": negative, "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 3 if negative.strip() else 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_t2i", "images": ["10", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "save"}
    return g, stages


STORY_SIZE = (1920, 1088)


def qwen_story_graph(m, images, prompt, seed, size=STORY_SIZE):
    """Storyboard from 1-3 pictures (character, location...) on Qwen Image 2.1: <image1>, <image2> in the prompt."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
    }
    enc = {"prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
    stages = {"1": "load", "2": "load", "3": "load", "4": "load"}
    for i, img in enumerate(images, 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        enc[f"images.image_{i}"] = [b, 0]
        stages[a] = stages[b] = "encode"
    g["7"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
        "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_story", "images": ["10", 0]}}
    stages.update({"7": "encode", "8": "encode", "9": "sample", "10": "decode", "11": "save"})
    return g, stages


FRAME_COLS = {2: 2, 3: 3, 4: 2, 6: 3, 8: 4, 9: 3}  # frames -> columns of the sheet (rows follow)
FRAME_SIZE = (1376, 768)
FRAME_LINE = {"white": ("white", 6), "black": ("black", 4), "none": ("white", 0)}


def _glue(g, stages, titles, frame_nodes, line):
    """Glue the decoded frames into one sheet with ImageStitch (exact thin line), saved as "Лист раскадровки"."""
    n = len(frame_nodes)
    color, gap = FRAME_LINE.get(line, FRAME_LINE["white"])
    cols = FRAME_COLS.get(n, n)
    rows = [frame_nodes[k:k + cols] for k in range(0, n, cols)]
    row_out = []
    for r, items in enumerate(rows):
        cur = items[0]
        for k, nxt in enumerate(items[1:]):
            nid = f"4{r}{k}"
            g[nid] = {"class_type": "ImageStitch", "inputs": {
                "image1": cur, "direction": "right", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
            stages[nid] = "decode"
            cur = [nid, 0]
        row_out.append(cur)
    cur = row_out[0]
    for r, nxt in enumerate(row_out[1:]):
        nid = f"49{r}"
        g[nid] = {"class_type": "ImageStitch", "inputs": {
            "image1": cur, "direction": "down", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
        stages[nid] = "decode"
        cur = [nid, 0]
    g["500"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_sheet", "images": cur}}
    stages["500"] = "save"
    titles["500"] = "Лист раскадровки"


def flux_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white"):
    """Storyboard frame by frame on FLUX.2 klein: every frame is its own picture (saved separately, full size) and the
    sheet is glued from them by ImageStitch with an exact thin line, so there are no margins or double frames."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "13": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "13": "encode", "14": "encode", "15": "encode", "8": "encode"}
    lat = []
    for i, img in enumerate(refs[:3], 1):
        a, b, v = f"10{i}", f"11{i}", f"12{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        lat.append(v)
        for k in (a, b, v):
            stages[k] = "encode"
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    neg = ["9", 0]
    for j, v in enumerate(lat):  # the negative side does not depend on the frame: build it once
        nid = f"15{j}"
        g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        neg = [nid, 0]
        stages[nid] = "encode"
    stages["9"] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        t = str(base)
        g[t] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}}
        pos = [t, 0]
        for j, v in enumerate(lat):
            nid = str(base + 1 + j)
            g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
            pos = [nid, 0]
            stages[nid] = "encode"
        gd, nz, sm, dc, sv = (str(base + k) for k in (5, 6, 7, 8, 9))
        g[gd] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}}
        g[nz] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + i}}
        g[sm] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [nz, 0], "guider": [gd, 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [sm, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({t: "encode", gd: "encode", nz: "encode", sm: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def qwen_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white"):
    """The same frame-by-frame storyboard on Qwen Image 2.1 (slower than FLUX, no commercial license)."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "8": "encode"}
    scaled = []
    for i, img in enumerate(refs[:3], 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        scaled.append(b)
        stages[a] = stages[b] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        en, ks, dc, sv = (str(base + k) for k in (0, 1, 2, 3))
        enc = {"prompt": text, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
        for k, b in enumerate(scaled, 1):
            enc[f"images.image_{k}"] = [b, 0]
        g[en] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
        g[ks] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": [en, 0], "negative": [en, 1], "latent_image": ["8", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [ks, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({en: "encode", ks: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def flux_story_graph(m, images, prompt, seed, size=STORY_SIZE):
    """Storyboard from 1-3 pictures on FLUX.2 klein: every picture is one more ReferenceLatent (image 1, image 2 ...)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "7": "encode", "8": "encode"}
    pos, neg = ["7", 0], ["8", 0]
    for i, img in enumerate(images, 1):
        a, b, v, rp, rn = f"10{i}", f"11{i}", f"12{i}", f"13{i}", f"14{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        g[rp] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
        g[rn] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        pos, neg = [rp, 0], [rn, 0]
        for k in (a, b, v, rp, rn):
            stages[k] = "encode"
    g["12"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}}
    g["13"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["14"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}}
    g["15"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["16"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["17"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["16", 0], "guider": ["12", 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}}
    g["18"] = {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}}
    g["19"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_story", "images": ["18", 0]}}
    stages.update({"12": "encode", "13": "encode", "14": "encode", "15": "encode", "16": "encode",
                   "17": "sample", "18": "decode", "19": "save"})
    return g, stages


def flux_edit_graph(m, prompt, image, megapixels, seed, extras=(), size=None):
    """FLUX.2 klein 4B (distilled), edit a picture: the official ComfyUI template on plain nodes, 4 steps, CFG 1.
    The result keeps the shape of the picture (sides in steps of 16)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16, "image": ["4", 0]}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}},
    }
    pos, neg = ["10", 0], ["11", 0]
    for k, img in enumerate(list(extras)[:2]):  # more reference pictures: image 2, image 3
        a, b, v, rp, rn = (f"{p}{k}" for p in ("30", "31", "32", "33", "34"))
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        g[rp] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
        g[rn] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        pos, neg = [rp, 0], [rn, 0]
    g.update({
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}},
        "13": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["6", 0], "height": ["6", 1], "batch_size": 1}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": ["6", 0], "height": ["6", 1]}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["16", 0], "guider": ["12", 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux", "images": ["18", 0]}},
    })
    if size:  # own size: the picture is scaled to it, the result follows
        g["5"] = {"class_type": "ImageScale", "inputs": {
            "image": ["4", 0], "upscale_method": "lanczos", "width": size[0], "height": size[1], "crop": "center"}}
    stages = {str(i): "load" for i in (1, 2, 3)}
    stages.update({str(i): "encode" for i in range(4, 17)})
    stages.update({k: "encode" for k in g if k.startswith(("30", "31", "32", "33", "34")) and len(k) == 3})
    stages.update({"17": "sample", "18": "decode", "19": "save"})
    return g, stages


def qwen_edit_size(aspect, megapixels):
    """Canvas for a chosen aspect ratio (about 1 or 2 megapixels, sides in steps of 32); None = keep the picture's own shape."""
    if aspect not in KREA_SIZES:
        return None
    w, h = KREA_SIZES[aspect]
    k = (megapixels or 1.0) ** 0.5
    return (round(w * k / 32) * 32, round(h * k / 32) * 32)


def qwen_edit_graph(m, prompt, image, megapixels, seed, size=None, extras=()):
    return _qwen_base(m, image, megapixels, prompt, seed, size, extras)


def qwen_inpaint_graph(m, prompt, image, crop, mask, x, y, megapixels, seed, size=None):
    """Local edit like the proven crop-and-stitch inpainting: Qwen Image 2.1 redraws only the masked area.
    `crop`/`mask` are the picture cut around the painted region (client-side), `x`/`y` its place in the original
    `image`, `size` = (width, height) of that crop. The crop is scaled to ~megapixels for the model, so the redrawn
    patch comes back at that size: it MUST be scaled back to the size of the crop before it is pasted at x,y
    (without it the patch lands enlarged and shifted). The mask keeps the size of the crop."""
    prompt = prompt.rstrip() + (" Keep the line art, outlines, shading and colors of the surrounding picture; "
                                "the redrawn part must continue the existing lines seamlessly.")
    g, stages = _qwen_base(m, crop, megapixels, prompt, seed)
    g["19"] = {"class_type": "InvertMask", "inputs": {"mask": ["21", 1]}}   # LoadImage's MASK is 1-alpha: invert = «painted area»
    # a hard edge cuts the lines at the border of the paint: grow the mask a little and soften its edge, so the redrawn
    # part melts into the original instead of leaving a seam (the edge zone is only a few pixels wide)
    g["24"] = {"class_type": "GrowMask", "inputs": {"mask": ["19", 0], "expand": 6, "tapered_corners": True}}
    g["25"] = {"class_type": "MaskToImage", "inputs": {"mask": ["24", 0]}}
    g["26"] = {"class_type": "ImageBlur", "inputs": {"image": ["25", 0], "blur_radius": 5, "sigma": 2.5}}
    g["27"] = {"class_type": "ImageToMask", "inputs": {"image": ["26", 0], "channel": "red"}}
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["21"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
    src = ["10", 0]
    if size and size[0] > 0 and size[1] > 0:
        g["23"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": int(size[0]), "height": int(size[1]),
                                                           "crop": "disabled", "image": ["10", 0]}}
        src = ["23", 0]
        stages["23"] = "decode"
    g["22"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["20", 0], "source": src, "mask": ["27", 0], "x": x, "y": y, "resize_source": False}}
    g["11"]["inputs"]["images"] = ["22", 0]
    stages.update({"19": "encode", "24": "encode", "25": "encode", "26": "encode", "27": "encode",
                   "20": "load", "21": "load", "22": "decode"})
    return g, stages


BG_REMOVE_PROMPT = (
    "Remove the background completely: replace it with a plain pure white background, evenly lit, no shadows. "
    "Keep the subject exactly as in <image1> — the same face, hair, clothes, colors, edges and style. "
    "Do not change, redraw or add anything in the subject itself.")

# The outpaint LoRA (qwen-image-2.1-outpaint-v2) was trained on this exact first sentence and on a flat gray (#808080)
# fill of the new areas; the user's text goes after it as «Scene: …». Without the sentence the model just copies the fill.
OUTPAINT_PROMPT = "Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, keeping the existing picture unchanged."


def qwen_outpaint_graph(m, prompt, canvas, orig, x, y, size, seed, outpaint_lora=None, osize=None):
    """Extend a picture: `canvas` is the original placed at x,y on the new larger `size`; the model paints the whole
    canvas (with the special outpaint LoRA when the user has it), then the untouched original is pasted back over
    its place, so only the new edges come from the model."""
    # the model must see the canvas at the size it draws: a smaller reference makes the redrawn middle sit a few pixels
    # off the original, and the blended seam then shows as double, glassy stripes
    g, stages = _qwen_base(m, canvas, max(0.25, size[0] * size[1] / 1e6), prompt, seed, size)
    if outpaint_lora:
        g["23"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["4", 0], "lora_name": outpaint_lora, "strength_model": 1.0}}
        g["9"]["inputs"]["model"] = ["23", 0]
        stages["23"] = "load"
    if osize and osize[0] > 0 and osize[1] > 0:
        # soft seam: a hard paste of the original leaves a visible edge (colour jump, cut lines). The original is kept
        # only inside a rectangle that is shrunk by a few pixels and blurred, so it melts into what the model drew there.
        # The mask is made on the whole canvas; the sides that touch the border of the canvas keep the original fully.
        W, H = int(size[0]), int(size[1])
        g["30"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": W, "height": H}}
        g["31"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": int(osize[0]), "height": int(osize[1])}}
        g["32"] = {"class_type": "MaskComposite", "inputs": {"destination": ["30", 0], "source": ["31", 0], "x": x, "y": y, "operation": "add"}}
        g["33"] = {"class_type": "GrowMask", "inputs": {"mask": ["32", 0], "expand": -6, "tapered_corners": False}}
        g["34"] = {"class_type": "MaskToImage", "inputs": {"mask": ["33", 0]}}
        g["35"] = {"class_type": "ImageBlur", "inputs": {"image": ["34", 0], "blur_radius": 5, "sigma": 2.5}}
        g["36"] = {"class_type": "ImageToMask", "inputs": {"image": ["35", 0], "channel": "red"}}
        g["22"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["10", 0], "source": ["5", 0], "mask": ["36", 0], "x": 0, "y": 0, "resize_source": False}}
        g["11"]["inputs"]["images"] = ["22", 0]
        stages.update({"30": "encode", "31": "encode", "32": "encode", "33": "encode", "34": "encode", "35": "encode", "36": "encode", "22": "decode"})
        return g, stages
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": orig}}
    g["22"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["10", 0], "source": ["20", 0], "x": x, "y": y, "resize_source": False}}
    g["11"]["inputs"]["images"] = ["22", 0]
    stages.update({"20": "load", "22": "decode"})
    return g, stages


def sheet_prompt(headless=True, background="grey", marks="", panels=4, desc=""):
    """Four panels: full body front (without the head by default), full body back, close-up front, close-up profile.
    Tested on Qwen Image 2.1 with a cartoon character: same style and outfit on all panels."""
    bg = SHEET_BG.get(background, "light grey")
    first = ("full-body front view, the figure has no head at all, not a mannequin and not a neck stump, the collar is "
             "simply empty and the plain background shows through the neckline, standing in a relaxed neutral pose, "
             "showing the whole outfit from the neckline to the shoes." if headless else
             "full-body front view facing the viewer, standing in a relaxed neutral pose, showing the whole outfit "
             "from head to shoes.")
    desc = (desc or "").strip()
    if desc:   # the user describes the outfit / details: they win over what the picture shows
        keep = ("Keep the same face, hairstyle, " + (marks.strip().rstrip(".,") + ", " if marks.strip() else "")
                + "proportions and art style as in <image1> in every panel. "
                + "The outfit and details, identical on every panel, follow this description and replace the clothing of <image1> where they differ: "
                + desc.rstrip(".") + ".")
    else:
        keep = "Keep the same face, hairstyle, " + (marks.strip().rstrip(".,") + ", " if marks.strip() else "") \
            + "clothing, colors, proportions and art style as in <image1> in every panel."
    if panels == 6:
        return (
            "A character reference sheet of the character from <image1>, split into six equal tall panels side by side in one "
            "row, all at the same scale, in the same art style as <image1>.\n"
            f"Panel one: {first}\n"
            "Panel two: full-body three-quarter view of the same character, turned about forty-five degrees to the left, "
            "same pose and outfit, head fully visible.\n"
            "Panel three: full-body side view in profile facing left, same pose and outfit, head fully visible.\n"
            "Panel four: full-body back view of the same character, same pose and outfit, seen from behind, head fully visible.\n"
            "Panel five: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
            "Panel six: close-up head-and-shoulders portrait in three-quarter view, with the same lighting and scale as panel five.\n"
            f"{keep}\n"
            f"Plain solid {bg} background, soft even lighting, no text, no labels.")
    return (
        "A character reference sheet of the character from <image1>, split into four equal panels side by side in one "
        "row, all at the same scale, in the same art style as <image1>.\n"
        f"Panel one: {first}\n"
        "Panel two: full-body back view of the same character in the same pose and outfit, seen from behind, head fully visible.\n"
        "Panel three: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
        "Panel four: close-up head-and-shoulders portrait in profile, with the same lighting and scale as panel three.\n"
        f"{keep}\n"
        f"Plain solid {bg} background, soft even lighting, no text, no labels.")


def qwen_sheet_graph(m, image, seed, headless=True, background="grey", marks="", panels=4, desc=""):
    prompt = sheet_prompt(headless, background, marks, panels, desc)
    size = SHEET6_SIZE if panels == 6 else SHEET_SIZE
    g, stages = _qwen_base(m, image, size[0] * size[1] / 1048576, prompt, seed, size=size)
    return g, stages, prompt


# ---------- improve quality of a finished video: SeedVR2 on plain ComfyUI nodes, one step, tiled ----------

UPSCALE_SCALES = {"1.5": 1.5, "2": 2.0}
UPSCALE_SEC_PER_MP_FRAME = 3.24  # measured on an RTX 4060 Ti 16 GB: 5 s of 1056x608 -> 1584x912 took ~10 minutes


def upscale_image_graph(m, image, scale, seed, soften=0.0):
    """The same SeedVR2 one-step enlarge for a single picture."""
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["1", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/upscale", "images": ["11", 0]}},
    }
    stages = {"1": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save"}
    if soften:  # soft result: SeedVR2 mixed with the plain enlarged picture (less "digital" sharpness)
        g["13"] = {"class_type": "ImageBlend", "inputs": {"image1": ["11", 0], "image2": ["3", 0], "blend_factor": soften, "blend_mode": "normal"}}
        g["12"]["inputs"]["images"] = ["13", 0]
        stages["13"] = "decode"
    return g, stages


RELAY_PIXEL_SECONDS = 16 * 1376 * 768  # the whole clip is decoded in memory: 15 s at 1376x768 (a "15 s" clip is really 15.1 s: 17k+5 frames) (was 10 s; raised by the author, memory on weaker cards not verified)
RELAY_SEC_PER_MP_FRAME = 0.7   # the author: 5 s of 720p in ~22 s on an RTX 5090 (0.2 s per megapixel-frame); a 4060 Ti is ~3.5x slower: an estimate, not measured here


def relay_graph(m, video, seed, detail=1.0):
    """RelayVSR x2: the whole video enlarged 2x in one sampling step (ComfyUI's built-in nodes), sound and fps kept."""
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": m["relayvsr_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["3", 0], "images": ["2", 0]}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["3", 0], "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["4", 1]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["relayvsr_vae"]}},
        "7": {"class_type": "VSRDecode", "inputs": {"vae": ["6", 0], "samples": ["5", 0], "images": ["2", 0], "detail": detail}},
        "8": {"class_type": "CreateVideo", "inputs": {"images": ["7", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "9": {"class_type": "SaveVideo", "inputs": {
            "video": ["8", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "6": "load", "4": "encode", "5": "sample", "7": "decode", "8": "save", "9": "save"}
    return g, stages


DLSS_MODES = ["1x (DLAA / native)", "1.5x (Quality)", "1.724x (Balanced)", "2x (Performance)", "3x (Ultra Performance)"]
DLSS_PRESETS = ["Default", "J", "K", "L", "M"]
DLSS_STYLES = ["natural", "cinematic", "default"]


def dlss_inputs(p):
    def f(k, d, lo, hi):
        try:
            return min(hi, max(lo, float(p.get(k, d))))
        except (TypeError, ValueError):
            return d
    def i(k, d, lo, hi):
        return int(f(k, d, lo, hi))
    def b(k, d):
        v = p.get(k, d)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
    pick = lambda k, lst, d: p.get(k) if p.get(k) in lst else d
    return {
            "upscaling_mode": pick("upscaling_mode", DLSS_MODES, DLSS_MODES[1]),
            "dlss_model_preset": pick("dlss_model_preset", DLSS_PRESETS, "M"), "style": pick("style", DLSS_STYLES, "natural"),
            "intensity": f("intensity", 1.0, 0, 2), "tone": f("tone", 2.0, 0, 2), "structure": f("structure", 2.0, 0, 2),
            "skin": f("skin", 2.0, -1, 2), "auto_mask": b("auto_mask", False),
            "temporal_mode": pick("temporal_mode", ["auto", "none"], "auto"),
            "scene_change_threshold": f("scene_change_threshold", 0.24, 0.01, 1), "warmup_frames": i("warmup_frames", 0, 0, 16),
            "gpu_mode": b("gpu_mode", True), "chunk_frames": i("chunk_frames", 8, 0, 256), "change_frame": i("change_frame", 0, 0, 100000)}


def dlss_image_graph(m, image, p, cls):
    """The same DLSS 5 for one picture: `cls` is the class of the picture node ("DLSS 5 (OreX)") found in the user's ComfyUI."""
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image}},
         "2": {"class_type": cls, "inputs": {"image": ["1", 0], **dlss_inputs(p)}},
         "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/upscale", "images": ["2", 0]}}}
    return g, {"1": "load", "2": "sample", "3": "save"}


def dlss_graph(m, video, p):
    """DLSS 5 Neural Rendering (unofficial node DLSS5NeuralRenderingVideo): the user's own settings, the sound of the source is put back."""
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "DLSS5NeuralRenderingVideo", "inputs": {"video": ["1", 0], **dlss_inputs(p)}},
        "4": {"class_type": "GetVideoComponents", "inputs": {"video": ["3", 0]}},
        "5": {"class_type": "CreateVideo", "inputs": {"images": ["4", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "6": {"class_type": "SaveVideo", "inputs": {
            "video": ["5", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "3": "sample", "4": "decode", "5": "save", "6": "save"}
    return g, stages


def upscale_graph(m, video, scale, seed, soften=0.0):
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["2", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save", "13": "save"}
    if soften:  # softer result: SeedVR2 mixed with the plain enlarged video (the share of the plain one is `soften`)
        g["14"] = {"class_type": "ImageBlend", "inputs": {"image1": ["11", 0], "image2": ["3", 0], "blend_factor": soften, "blend_mode": "normal"}}
        g["12"]["inputs"]["images"] = ["14", 0]
        stages["14"] = "decode"
    return g, stages


def icons_graph(video, o):
    """Animated icons: a video on a plain or any background -> transparent frames (PNG), an animated WEBP and a Telegram WebM.
    Built from the user's workflows «WebP Loop» (cut by green / RMBG / MatAnyone2, then LoopCleanup).
    `video`: a file in ComfyUI's input folder. `o`: method ('green' | 'rmbg' | 'matanyone'), edge (-8..4: negative = the edge goes inside, the green rim goes away),
    threshold (green only), loop (cut the seam of a loop), despill, key ('green' | 'blue'), width/height (0: as the video), prefix, png, webp, telegram,
    r_erode, r_dilate (MatAnyone's start mask)."""
    method = o.get("method", "matanyone")
    edge = int(o.get("edge", -2))
    g = {"L": {"class_type": "LoadVideo", "inputs": {"file": video}},
         "N": {"class_type": "GetVideoComponents", "inputs": {"video": ["L", 0]}}}
    stages = {"L": "load", "N": "encode"}

    def add(i, cls, **kw):
        g[i] = {"class_type": cls, "inputs": kw}
        stages[i] = "encode"

    rmbg = dict(model="RMBG-2.0", sensitivity=1.0, process_res=1024, mask_blur=0, invert_output=False, refine_foreground=True,
                background="Alpha", background_color="#222222")
    if method == "green":   # no neural network: a pixel is background when it is greener than red AND blue
        add("R", "ImageToMask", image=["N", 0], channel="red")
        add("G", "ImageToMask", image=["N", 0], channel="green")
        add("B", "ImageToMask", image=["N", 0], channel="blue")
        add("GR", "MaskComposite", destination=["G", 0], source=["R", 0], x=0, y=0, operation="subtract")
        add("GB", "MaskComposite", destination=["G", 0], source=["B", 0], x=0, y=0, operation="subtract")
        add("GM", "MaskComposite", destination=["GR", 0], source=["GB", 0], x=0, y=0, operation="multiply")
        add("T", "ThresholdMask", mask=["GM", 0], value=float(o.get("threshold", 0.02)))
        add("W", "GrowMask", mask=["T", 0], expand=max(-8, min(8, edge + 3)), tapered_corners=True)
        add("MI", "MaskToImage", mask=["W", 0])
        add("BL", "ImageBlur", image=["MI", 0], blur_radius=2, sigma=1.0)   # soft edge (the user's workflow blurs it by a shader)
        add("MK", "ImageToMask", image=["BL", 0], channel="red")
        add("A", "JoinImageWithAlpha", image=["N", 0], alpha=["MK", 0])
    elif method == "rmbg":
        add("A", "RMBG", image=["N", 0], mask_offset=edge, **rmbg)
    else:   # MatAnyone2: the mask of the hero from the first frame (RMBG), then MatAnyone leads it through all frames
        add("F1", "ImageFromBatch", image=["N", 0], batch_index=0, length=1)
        add("RM", "RMBG", image=["F1", 0], mask_offset=-2, **rmbg)
        add("MA", "MatAnyone2", src_video=["N", 0], mask_frame=0, n_warmup=10, foreground_MASK=["RM", 1],
            r_erode=int(o.get("r_erode", 0)), r_dilate=int(o.get("r_dilate", 0)), max_internal_size=960, max_mem_frames=5, use_long_term=False)
        add("MR", "ImageToMask", image=["MA", 0], channel="red")
        add("GM", "GrowMask", mask=["MR", 0], expand=edge, tapered_corners=False)
        add("IM", "InvertMask", mask=["GM", 0])   # 1 = transparent background
        add("A", "JoinImageWithAlpha", image=["N", 0], alpha=["IM", 0])
    add("C", "LoopCleanup", images=["A", 0], despill=bool(o.get("despill", True)), find_loop=bool(o.get("loop", True)),
        min_loop_frames=40, key_color=o.get("key", "green") if o.get("key") in ("green", "blue") else "green")
    X = ["C", 0]
    w, h = int(o.get("width") or 0), int(o.get("height") or 0)
    if w >= 16 and h >= 16:   # the size of the frames (PNG, WEBP, sprite sheet)
        add("S", "ImageScale", image=["C", 0], upscale_method="lanczos", width=w, height=h, crop="disabled")
        X = ["S", 0]
    prefix = o.get("prefix", "H3Workspace/icons/icon")
    if o.get("png", True):
        g["P"] = {"class_type": "SaveImage", "inputs": {"images": ["C", 0], "filename_prefix": prefix + "/frame"}}   # PNG frames: the video's own size, as in the workflow
        stages["P"] = "save"
    if o.get("webp"):
        g["Q"] = {"class_type": "SaveAnimatedWEBP", "inputs": {"images": X, "filename_prefix": prefix + "/animation", "fps": ["N", 2],
                                                               "lossless": True, "quality": 95, "method": "default"}}
        stages["Q"] = "save"
    if o.get("telegram"):
        g["TG"] = {"class_type": "TelegramStickerWebM", "inputs": {"images": ["C", 0], "fps": ["N", 2], "filename_prefix": prefix + "/telegram",
                                                                   "side": 512, "max_kb": 256, "max_seconds": 3.0}}
        stages["TG"] = "save"
    return g, stages


ICON_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
ICON_LORA = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"


def icons_video_graph(m, unet, lora, prompt, width, height, seconds, seed, image):
    """The picture -> video part of the user's Telegram workflow (the subgraph of the node «MiniMax H3»): the fl2va model,
    Turbo 4-step v1.2 LoRA at 0.8, 6 steps, res_multistep; the picture is BOTH the first and the last frame (a seamless loop, the look is kept)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 0.8}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image}},
        "21": {"class_type": "ImageScale", "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}},
        "8": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": prompt, "width": width, "height": height, "length": frames(seconds),
            "first_frame": ["21", 0], "last_frame": ["21", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["12", 0], "sampler": ["10", 0],
                                                                  "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": float(FPS),
                                                       "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": "H3Workspace/video", "format": "auto", "format.codec": "auto"}},
    }
    stages = {n: "load" for n in ("1", "2", "5", "6", "7")}
    stages.update({n: "encode" for n in ("8", "20", "21")})
    stages.update({"13": "sample", "14": "decode", "15": "decode", "16": "save", "17": "save"})
    return g, stages


def icons_api_graph(base, o):
    """The user's own, finished ComfyUI graph (her two workflows «WebP Loop» + «WebM Telegram», exported in API format and merged): nothing is rebuilt,
    only her switches and inputs are set. o: mode ('image' | 'video'), image / video (file in ComfyUI's input), prompt, seconds, seed, vsize (W, H or None =
    as the picture), bg ('' | green | pink | blue | white | gray | black), cutbg, method, edge, threshold, r_erode, r_dilate, despill, loop, key, size (W, H or None),
    folder, png, webp, telegram."""
    g = copy.deepcopy(base)

    def put(i, k, v):
        if i in g:
            g[i]["inputs"][k] = v
    folder = o["folder"]
    method, edge = o.get("method", "matanyone"), int(o.get("edge", -2))
    for i in ("223", "231"):
        put(i, "value", method == "matanyone")       # MatAnyone on / the old way (green or RMBG)
    for i in ("178", "204"):
        put(i, "value", method == "green")           # green cut on / RMBG
    for i in ("235", "236"):
        put(i, "expand", edge)                       # the edge of MatAnyone's cut
    for i in ("132", "190"):
        put(i, "mask_offset", edge)                  # the edge of RMBG
    for i in ("183", "208"):
        put(i, "expand", max(-8, min(8, edge + 3)))  # the edge of the green cut
    for i in ("174", "201"):
        put(i, "value", float(o.get("threshold", 0.02)))
    for i in ("219", "227"):
        put(i, "r_erode", int(o.get("r_erode", 0)))
        put(i, "r_dilate", int(o.get("r_dilate", 0)))
    for i in ("240", "241"):
        put(i, "despill", bool(o.get("despill", True)))
        put(i, "find_loop", bool(o.get("loop", True)))
        put(i, "key_color", o.get("key") if o.get("key") in ("green", "blue") else "green")
    sz = o.get("size")
    for i in ("188", "213"):
        put(i, "value", bool(sz))                    # WEBP: own size / as the video
    if sz:
        for i in ("184", "209"):
            put(i, "value", int(sz[0]))
        for i in ("185", "210"):
            put(i, "value", int(sz[1]))
    out = {"image": ("92", "133", "137", "242"), "video": ("191", "192", "245")}
    if o["mode"] == "image":
        put("114", "image", o["image"])
        put("105:104", "prompt", o["prompt"])
        put("105:111", "value", float(o["seconds"]))
        put("105:15", "noise_seed", int(o["seed"]))
        put("138", "value", folder)
        vs = o.get("vsize")
        put("164", "value", bool(vs))                # own resolution / the picture's own size
        if vs:
            put("162", "width", int(vs[0]))
            put("162", "height", int(vs[1]))
        for k, i in (("green", "142"), ("pink", "143"), ("blue", "144"), ("white", "145"), ("gray", "146"), ("black", "147")):
            put(i, "value", o.get("bg") == k)
        put("167", "value", bool(o.get("cutbg")))
        want = {"92": True, "133": bool(o.get("png")), "137": bool(o.get("webp")), "242": bool(o.get("telegram"))}
    else:
        put("214", "file", o["video"])
        put("191", "filename_prefix", folder + "/alpha_frames/frame")
        put("192", "filename_prefix", folder + "/webp/animation")
        put("245", "filename_prefix", folder + "/telegram/sticker")
        want = {"191": bool(o.get("png")), "192": bool(o.get("webp")), "245": bool(o.get("telegram"))}
    for i in out["image"] + out["video"]:
        if want.get(i) is not True:
            g.pop(i, None)
    # what really runs: everything upstream of the kept outputs
    need, stack = set(), [i for i in want if want[i] and i in g]
    while stack:
        i = stack.pop()
        if i in need or i not in g:
            continue
        need.add(i)
        for v in g[i]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    stage = {"105:6": "load", "105:121": "load", "105:13": "load", "105:11": "load", "105:24": "load", "105:104": "encode", "105:14": "sample",
             "105:10": "decode", "105:23": "decode", "105:91": "save", "92": "save", "131": "decode", "189": "decode", "132": "upscale", "190": "upscale",
             "218": "upscale", "226": "upscale", "219": "sample2", "227": "sample2", "133": "save", "137": "save", "191": "save", "192": "save",
             "242": "save", "245": "save"}
    stages = {i: stage[i] for i in need if i in stage}
    return g, stages, need


# ---------- 3D model from one picture (her two ComfyUI graphs: TripoSplat «Точно» and Trellis 2 «Чисто», API format in graphs3d/) ----------

TD_NODES = {
    "triposplat": ["LoadImage", "RemoveBackground", "LoadBackgroundRemovalModel", "TripoSplatPreprocessImage", "CLIPVisionLoader", "VAELoader", "UNETLoader",
                   "TripoSplatConditioning", "KSampler", "VAEDecodeTripoSplat", "SplatToMesh", "RemeshMesh", "DecimateMesh", "MeshSmoothNormals", "SMeshClay", "SSave3D"],
    "trellis": ["LoadImage", "RemoveBackground", "LoadBackgroundRemovalModel", "ImageCropToMask", "ComfySwitchNode", "CLIPVisionLoader", "VAELoader", "UNETLoader",
                "CFGOverride", "RescaleCFG", "ModelSamplingSD3", "Trellis2Conditioning", "EmptyTrellis2LatentStructure", "VaeDecodeStructureTrellis2", "Trellis2ShapeStage",
                "Trellis2UpsampleStage", "VaeDecodeShapeTrellis", "KSampler", "ComfyMathExpression", "PrimitiveInt", "RemeshMesh", "DecimateMesh", "MeshSmoothNormals", "SMeshClay", "SSave3D"],
}
# the model files of each variant: (loader class, field, file name as in the graph)
TD_MODELS = {
    "triposplat": [("UNETLoader", "unet_name", "3D/triposplat_fp16.safetensors"), ("CLIPVisionLoader", "clip_name", "3D/dino_v3_vit_h.safetensors"),
                   ("VAELoader", "vae_name", "3D/flux2-vae.safetensors"), ("VAELoader", "vae_name", "3D/triposplat_vae_decoder_fp16.safetensors"),
                   ("LoadBackgroundRemovalModel", "bg_removal_name", "3D/birefnet.safetensors")],
    "trellis": [("UNETLoader", "unet_name", "3D/trellis_2_int8_convrot.safetensors"), ("CLIPVisionLoader", "clip_name", "3D/dino_v3_L_naf_fp32.safetensors"),
                ("VAELoader", "vae_name", "3D/trellis_2_shape_vae_bf16.safetensors"), ("LoadBackgroundRemovalModel", "bg_removal_name", "3D/birefnet.safetensors")],
}


def _hf(repo, sub, name, size):
    return {"file": name, "url": f"https://huggingface.co/{repo}/resolve/main/{sub}/{name}", "page": f"https://huggingface.co/{repo}/blob/main/{sub}/{name}", "size": size}


# download links (Hugging Face, the same files as in the README of sofiakholina-3D-Generation); size in bytes
TD_FILES = {f["file"].lower(): f for f in (
    _hf("VAST-AI/TripoSplat", "diffusion_models", "triposplat_fp16.safetensors", 741106994),
    _hf("VAST-AI/TripoSplat", "clip_vision", "dino_v3_vit_h.safetensors", 1681247696),
    _hf("VAST-AI/TripoSplat", "vae", "triposplat_vae_decoder_fp16.safetensors", 576148442),
    _hf("VAST-AI/TripoSplat", "vae", "flux2-vae.safetensors", 336213556),
    _hf("Comfy-Org/BiRefNet", "background_removal", "birefnet.safetensors", 444473596),
    _hf("Comfy-Org/TRELLIS.2", "diffusion_models", "trellis_2_int8_convrot.safetensors", 5253048192),
    _hf("Comfy-Org/Pixal3D", "clip_vision", "dino_v3_L_naf_fp32.safetensors", 1215214176),
    _hf("Comfy-Org/Pixal3D", "vae", "trellis_2_shape_vae_bf16.safetensors", 1095844024),
)}


def td_api_graph(base, o):
    """o: variant ('triposplat' | 'trellis'), image (file in ComfyUI's input), seed, faces, format ('glb' | 'fbx' | 'obj'), folder (where ComfyUI writes the file),
    name (file name), nobg (Trellis: cut the background out first), models (dict: lower-case file name -> the name ComfyUI lists).
    The viewer node (Preview3DAdvanced) is dropped: it only works inside ComfyUI's own window."""
    g = copy.deepcopy(base)
    v = o["variant"]
    for i in [k for k, n in g.items() if n["class_type"] in ("Preview3DAdvanced", "MeshToFile3D")]:
        g.pop(i)

    def first(cls):
        return next(k for k, n in g.items() if n["class_type"] == cls)
    g[first("LoadImage")]["inputs"]["image"] = o["image"]
    seed = int(o["seed"])
    for k, n in g.items():
        if n["class_type"] == "PrimitiveInt" and n["inputs"].get("value") not in (1024,):
            n["inputs"]["value"] = seed
        if n["class_type"] == "DecimateMesh":
            n["inputs"]["target_face_count"] = int(o["faces"])
        if n["class_type"] == "ComfySwitchNode" and v == "trellis":
            n["inputs"]["switch"] = bool(o.get("nobg", True))
        for field, val in list(n["inputs"].items()):
            if isinstance(val, str) and n["class_type"] in ("UNETLoader", "CLIPVisionLoader", "VAELoader", "LoadBackgroundRemovalModel"):
                real = (o.get("models") or {}).get(val.replace(chr(92), "/").split("/")[-1].lower())   # the file name exactly as this ComfyUI lists it
                if real:
                    n["inputs"][field] = real
    s = g[first("SSave3D")]["inputs"]
    fmt = o["format"] if o["format"] in ("glb", "fbx", "obj") else "glb"
    s.update(autosave=True, format=fmt, filename_prefix=o["name"], folder=o["folder"], save_units="m" if fmt != "fbx" else "mm", input_units="m")
    pv = copy.deepcopy(g[first("SSave3D")])
    pv["inputs"].update(format="glb", folder=o["preview_dir"], filename_prefix=o["name"] + "_pv", save_units="m")
    g["pv_save"] = pv
    stages = {}
    for k, n in g.items():
        c = n["class_type"]
        stages[k] = ("load" if c in ("LoadImage", "CLIPVisionLoader", "VAELoader", "UNETLoader", "LoadBackgroundRemovalModel") else
                     "encode" if "Conditioning" in c or c in ("RemoveBackground", "TripoSplatPreprocessImage", "ImageCropToMask") else
                     "sample" if c == "KSampler" or "Stage" in c else
                     "save" if c == "SSave3D" else "decode")
    return g, stages
