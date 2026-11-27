"""The Video studio scene for RunningHub: the SAME scene graph as the local studio (workflows.long_part_graph), only the parts that cannot
exist on RunningHub are replaced:

- model loaders and attention nodes come from the proven RunningHub graph (their file names exist there);
- RunningHub keeps no saved state, so a scene that continues the previous one gets the previous clip (a video file): its last frames
  are encoded back into the latent (the same `cont` branch the local studio uses for «continue my video»);
- the scene's own voice recording locks the audio part of the latent exactly as in the local studio.

The local code is only read here, nothing in it is changed.
"""

import copy

import workflows as wf

BASE_NODES = ("1", "2", "3", "4", "5", "6", "10", "12", "60", "61", "62")


def build(base, p, files):
    """`base`: the API graph of the RunningHub card (loaders). `p`: prompt, width, height, seed, seconds, pika, ref_size.
    `files`: {'img0'.., 'vid0'.., 'aud0'.., 'voice': name, 'prev': name of the previous clip, 'prev_frames': int}. Returns (graph, seed)."""
    g = {k: copy.deepcopy(base[k]) for k in BASE_NODES if k in base}
    pika = bool(p.get("pika", True))
    for k, key in (("60", "dmad"), ("61", "speed")):   # DMAD and the speed slider are not part of the studio route; the clip page may switch them on
        g[k]["inputs"]["strength_model"] = float(p.get(key) or 0.0)
    g["62"]["inputs"]["strength_model"] = 1.0 if pika else 0.0   # Pika: the speed LoRA of the studio (8 steps); without it: the quality route (12 steps)
    if pika:
        model, sched, samp, steps = ["3", 0], "simple", "euler", 8
    else:
        g["8"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 4.0}}
        model, sched, samp, steps = ["8", 0], "beta", "er_sde", 12
    g["10"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": sched, "steps": steps, "denoise": 1.0}}
    g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": samp}}
    width, height, seed = int(p["width"]), int(p["height"]), int(p["seed"])
    prev, prev_frames = files.get("prev"), int(files.get("prev_frames") or 0)
    cont = bool(prev and prev_frames)
    plan = wf.long_plan([float(p["seconds"])], cont_first=cont)[0]
    total, ov = plan["total"], plan["ov"]
    nid = [200]

    def add(cls, **inputs):
        nid[0] += 1
        g[str(nid[0])] = {"class_type": cls, "inputs": inputs}
        return str(nid[0])

    ref = add("MiniMaxH3ReferenceToVideo", clip=["4", 0], vae=["5", 0], audio_vae=["6", 0], prompt=p["prompt"],
              width=width, height=height, length=total, ref_image_size=p.get("ref_size") or "max")
    pick = lambda key: [files[k] for k in sorted((k for k in files if k.startswith(key) and k[len(key):].isdigit()), key=lambda k: int(k[len(key):]))]
    wf._refs(g, ref, pick("img"), videos=pick("rvid") + pick("vid"), audios=pick("aud"), base=100)   # the server passes the Reference Mode videos (and the user's camera videos) as rvid0..: they were skipped before
    new = add("LTXVSeparateAVLatent", av_latent=[ref, 1])
    voice = files.get("voice")
    song = files.get("song")   # Clip: a piece of the user's song that starts `song_lead` seconds before this scene; the overlap with the previous scene hears the real song
    vsrc = None
    if song:
        lead = float(files.get("song_lead") or 0.0)
        beg = max(0.0, lead - ov / wf.FPS) if cont else lead
        ld = [add("LoadAudio", audio=song), 0]
        cutp = [add("TrimAudioDuration", audio=ld, start_index=round(beg, 3), duration=round(total / wf.FPS, 3)), 0]
        pad = add("EmptyAudio", duration=round(total / wf.FPS, 3), sample_rate=48000, channels=2)
        vsrc = [add("TrimAudioDuration", audio=[add("AudioConcat", audio1=cutp, audio2=[pad, 0], direction="after"), 0], start_index=0.0, duration=round(total / wf.FPS, 3)), 0]
        voice = song
    if voice:   # the user's own voice of this scene: the audio part of the latent is locked, the words are exactly these
        if not song:
            vsrc = wf._scene_voice(add, voice, ov / wf.FPS if cont else 0.0, total)
        enc = add("VAEEncodeAudio", audio=vsrc, vae=["6", 0])
        amask = add("SolidMask", value=0.0, width=round(total * 5 / 3), height=2)
        a_lat = [add("SetLatentNoiseMask", samples=[enc, 0], mask=[amask, 0]), 0]
    else:
        a_lat = [new, 1]
    if not cont:
        v_lat = [new, 0]
    else:   # the tail of the previous clip, encoded back into a latent; the new part continues it
        vid_in = add("LoadVideo", file=prev)
        comp = add("GetVideoComponents", video=[vid_in, 0])
        last = add("ImageFromBatch", image=[comp, 0], batch_index=max(0, prev_frames - ov), length=ov)
        fit = add("ImageScale", image=[last, 0], upscale_method="lanczos", width=width, height=height, crop="center")
        old = add("VAEEncode", pixels=[fit, 0], vae=["5", 0])
        t_ov, t_tot = wf._tokens(ov), wf._tokens(total)
        cut = add("LatentCut", samples=[old, 0], dim="t", index=-t_ov, amount=t_ov)
        cat = add("LatentConcat", samples1=[cut, 0], samples2=[new, 0], dim="t")
        cut2 = add("LatentCut", samples=[cat, 0], dim="t", index=0, amount=t_tot)
        lw, lh = round(width / 16), round(height / 16)
        black = add("EmptyImage", width=lw, height=lh, batch_size=t_ov, color=0)
        white = add("EmptyImage", width=lw, height=lh, batch_size=max(1, t_tot - t_ov), color=16777215)
        both = add("ImageBatch", image1=[black, 0], image2=[white, 0])
        vmask = add("ImageToMask", image=[both, 0], channel="red")
        v_lat = [add("SetLatentNoiseMask", samples=[cut2, 0], mask=[vmask, 0]), 0]
    lat = add("LTXVConcatAVLatent", video_latent=v_lat, audio_latent=a_lat)
    noise = add("RandomNoise", noise_seed=seed)
    guider = add("BasicGuider", model=model, conditioning=[ref, 0])
    smp = add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=["12", 0], sigmas=["10", 0], latent_image=[lat, 0])
    img = add("VAEDecode", samples=[smp, 1], vae=["5", 0])
    clip_len = total - ov
    frames_out = [img, 0] if not cont else [add("ImageFromBatch", image=[img, 0], batch_index=ov, length=clip_len), 0]
    if voice:
        aud = [add("TrimAudioDuration", audio=vsrc, start_index=round(ov / wf.FPS if cont else 0.0, 3), duration=round(clip_len / wf.FPS, 3)), 0]
    else:
        dec = add("VAEDecodeAudio", samples=[smp, 1], vae=["6", 0])
        aud = [dec, 0] if not cont else [add("TrimAudioDuration", audio=[dec, 0], start_index=round(ov / wf.FPS, 3), duration=round(clip_len / wf.FPS, 3)), 0]
        aud = wf._audio_fade(add, aud, round(clip_len / wf.FPS, 3), head=cont, tail=True)
    vid = add("CreateVideo", images=frames_out, audio=aud, fps=float(wf.FPS), bit_depth="auto", color_space="sRGB", chroma_subsampling="auto", codec="none")
    add("SaveVideo", video=[vid, 0], filename_prefix="H3Workspace/long_part", format="auto", **{"format.codec": "auto"}, codec="auto")
    return g, seed
