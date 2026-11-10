import asyncio
import json
import os
import shutil
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image

from . import core


def _input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        return ""


def _resolve(name, work):
    """Graydient stages media fields as local filenames in the input dir; a full path or an http(s) URL also works."""
    name = (name or "").strip()
    if not name:
        return ""
    if name.startswith(("http://", "https://")):
        dst = os.path.join(work, "download_" + (os.path.basename(name.split("?")[0]) or "media"))
        with urllib.request.urlopen(name, timeout=120) as resp, open(dst, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return dst
    for cand in (name, os.path.join(_input_dir(), name)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"media file not found: {name}")


def _image_tensor(path_or_array):
    import torch
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
    else:
        a = np.asarray(Image.open(path_or_array).convert("RGB"))
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0) if a.ndim == 3 else \
        torch.from_numpy(a.astype(np.float32) / 255.0)


def _fit_frame(img, w, h, work):
    """A supplied IMAGE tensor [1, H, W, 3] (any size) -> PNG path at exactly w x h (cover, centre crop)."""
    from PIL import ImageOps
    a = (img[0].cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    fitted = ImageOps.fit(Image.fromarray(a), (int(w), int(h)), method=Image.LANCZOS)
    path = os.path.join(work, "supplied.png")
    fitted.save(path)
    return path


def _fit_png(path, w, h, work, name="fitted.png"):
    """Any picture file -> a PNG at exactly w x h (cover, centre crop); returns the new path."""
    from PIL import ImageOps
    im = ImageOps.fit(Image.open(path).convert("RGB"), (int(w), int(h)), method=Image.LANCZOS)
    out = os.path.join(work, name)
    im.save(out)
    return out


def _stack(work, bundle, images):
    """The ordered stack of edited frames: the pictures of `bundle` (video2) first, then image1..image8 in slot order.
    A slot may hold a picture or a bundle (an mp4 with attached pictures); a bundle expands where it sits."""
    out = []

    def keep(path, slot=""):
        im = Image.open(path).convert("RGB")
        # Graydient hands an image slot a PICTURE of a render, never the video: a bundle given as /imageN:R<hash> arrives as
        # its black first frame. Refuse it instead of using it as a chunk's edited frame.
        if slot and float(np.asarray(im.resize((32, 32))).std()) < 2.0:
            raise RuntimeError(f"{slot} is a blank picture. If it is a bundle, pass it in video2 (a bundle cannot travel in an "
                               f"image slot), and bundle in the order you want with ripple-chain-bundle (video1 = the earlier bundle)")
        dst = os.path.join(work, f"stack{len(out) + 1:03d}.png")
        im.save(dst)
        out.append(dst)

    def add_path(path, slot=""):
        try:
            Image.open(path).verify()
        except Exception:
            sub = os.path.join(work, f"bundle{len(out):03d}")
            os.makedirs(sub, exist_ok=True)
            pics = core.extract_covers(path, sub)
            if not pics:
                raise RuntimeError(f"{os.path.basename(path)} is neither a picture nor a bundle of pictures")
            for pic in pics:
                keep(pic)
            return
        keep(path, slot)

    for slot, ref in [("video2", bundle)] + [(f"image{i + 1}", r) for i, r in enumerate(images)]:
        if (ref or "").strip():
            add_path(_resolve(ref, work), slot)
    return out


def _font(size, bold=True):
    from PIL import ImageFont
    for name in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _write_status_card(path, step, left, done, w=768, h=432):
    """A small progress picture saved next to the video: which step this was, how many are left, what to do next."""
    from PIL import ImageDraw
    im = Image.new("RGB", (w, h), (18, 22, 30))
    d = ImageDraw.Draw(im)
    total = step + left
    if done:
        head, sub = "DONE", f"All {total} steps rendered. This is your finished video."
        accent = (88, 200, 140)
    else:
        head = f"STEP {step} OF {total}"
        sub = f"{left} step{'s' if left != 1 else ''} left. Reply to this render to render the next one."
        accent = (110, 170, 255)

    def centred(text, y, font, fill):
        box = d.textbbox((0, 0), text, font=font)
        d.text(((w - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)

    centred(head, h * 0.30, _font(64), accent)
    centred(sub, h * 0.50, _font(22, False), (225, 228, 235))
    bar_x, bar_w, bar_y = int(w * 0.12), int(w * 0.76), int(h * 0.66)
    d.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 18], radius=9, fill=(48, 54, 68))
    d.rounded_rectangle([bar_x, bar_y, bar_x + max(18, int(bar_w * step / max(1, total))), bar_y + 18], radius=9, fill=accent)
    if not done:
        centred("Keep the same /seed. Nothing else to type.", h * 0.76, _font(18, False), (150, 158, 172))
    im.save(path)
    return path


class SwapChainLoad:
    """Start or continue a swap chain.

    Step 1: video = the guide clip, image1 = the character photo, prompt = the character description.
    Later steps: video = the previous step's result (a swapchain mp4 carries the guide, its audio, the processed reference,
    the prompt, the size and the cursor); nothing else is needed. The node tells the two apart by the state tag.

    Outputs this step's slice of the guide (frames on the model's 17k+5 grid, ending on a scene cut when one falls near the
    cap), the matching audio slice, the working size, the reference (the raw photo on step 1; on later steps the stored,
    already background-removed one, ref_ready = 1, so the graph can skip BiRefNet), the prompt and a ctx string for Finish.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"default": ""}),
            "image1": ("STRING", {"default": ""}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "max_chunk_frames": ("INT", {"default": core.MAX_FRAMES, "min": 22, "max": 163, "step": 1}),
            "target_pixels": ("INT", {"default": 262144, "min": 65536, "max": 4194304}),
            "max_seconds": ("INT", {"default": 60, "min": 1, "max": 600}),
        }, "optional": {
            "tail_anchor": ("INT", {"default": 0, "min": 0, "max": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "AUDIO", "STRING", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("guide_frames", "ref_image", "ref_ready", "frames", "width", "height", "audio", "prompt", "ctx",
                    "tail_frames", "tail_ready")
    FUNCTION = "run"
    CATEGORY = "SwapChain"

    async def run(self, video, image1, prompt, width, height, max_chunk_frames, target_pixels, max_seconds, tail_anchor=0):
        return await asyncio.to_thread(self._run, video, image1, prompt, width, height, max_chunk_frames,
                                       target_pixels, max_seconds, tail_anchor)

    def _run(self, video, image1, prompt, width, height, max_chunk_frames, target_pixels, max_seconds, tail_anchor=0):
        import torch
        work = tempfile.mkdtemp(prefix="swapchain_")
        src = _resolve(video, work)
        if not src:
            raise RuntimeError("video is empty: give the guide clip (/video1:URL) or the previous step (/video1:R<hash>)")
        given = ""
        img_path = _resolve(image1, work)
        if img_path:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((1024, 1024))
            given = os.path.join(work, "given.png")
            im.save(given)

        if core.is_chain(src):
            prev = src
            st, vsec = core.read_state(prev)
            if st.get("done"):
                raise RuntimeError("this swap chain is already complete (done=true): nothing left to render")
            w, h = int(st["w"]), int(st["h"])
            cursor, total, step = float(st.get("cursor") or vsec), float(st["total"]), int(st.get("step", 0)) + 1
            cuts = st.get("cuts") or []
            use_prompt = prompt.strip() or st.get("prompt", "")
            guide_path, guide_idx = prev, 1
            has_audio = bool(st.get("has_audio"))
            audio_path, audio_idx = (prev, 0) if has_audio else (None, None)
            if given:
                ref, ref_ready = given, 0
            else:
                ref, ref_ready = core.extract_cover(prev, os.path.join(work, "stored_ref.png")), 1
        else:
            prev = None
            if not given:
                raise RuntimeError("first step needs the character photo (image1) as well as the guide video")
            info = core.probe(src)
            vs, secs = core.main_video(info)
            total = min(secs, float(max_seconds))
            w, h = core.pick_size(int(vs["width"]), int(vs["height"]), width, height, target_pixels)
            guide_path = core.prepare_guide(src, os.path.join(work, "guide.mp4"), w, h, total)
            guide_idx = 0
            cuts = core.detect_cuts(guide_path)
            audio_path = core.extract_audio(src, os.path.join(work, "guide_audio.m4a"), total)
            audio_idx = 0 if audio_path else None
            has_audio = bool(audio_path)
            cursor, step, use_prompt = 0.0, 1, prompt.strip() or core.DEFAULT_DETAILS
            ref, ref_ready = given, 0

        # Tail anchor: on a later step the chunk starts TAIL_FRAMES early, on the exact frames the previous step ended with, and
        # those finished frames guide the start of the new chunk (MiniMaxH3AddGuide). They are dropped again when stitching.
        # No anchor when the previous chunk ended on a scene cut: there is nothing to stay consistent with.
        ov = 0
        if prev and int(tail_anchor) and cursor - core.TAIL_FRAMES / core.FPS >= 0 and not core.near_cut(cursor, cuts):
            ov = core.TAIL_FRAMES
            cursor = cursor - ov / core.FPS
        frames = core.pick_frames(cursor, total, max_chunk_frames, cuts)
        gf = core.decode_guide(guide_path, guide_idx, cursor, frames, w, h)
        tail = core.tail_frames(prev, ov, w, h) if ov else np.zeros((1, h, w, 3), np.uint8)
        seg = core.decode_audio(audio_path, audio_idx, cursor, frames) if audio_path else \
            core.decode_audio(None, None, cursor, frames)
        ctx = {"prev": prev, "work": work, "cursor": cursor, "frames": frames, "total": total, "step": step,
               "prompt": use_prompt, "w": w, "h": h, "cuts": cuts, "has_audio": has_audio,
               "guide_clip": "" if prev else guide_path, "guide_audio": "" if prev or not audio_path else audio_path,
               "ov": ov, "tail": int(tail_anchor), "chunk": int(max_chunk_frames)}
        audio_out = {"waveform": torch.from_numpy(seg).unsqueeze(0), "sample_rate": core.SR}
        return (_image_tensor(gf), _image_tensor(ref), int(ref_ready), int(frames), int(w), int(h), audio_out,
                use_prompt, json.dumps(ctx), _image_tensor(tail), 1 if ov else 0)


class SwapChainFinish:
    """Stitch the decoded clip after the previous video and save the self-contained swapchain mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "ref_image": ("IMAGE",),
            "ctx": ("STRING", {"forceInput": True}),
            "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
            "filename_prefix": ("STRING", {"default": "video/SwapChain"}),
        }, "optional": {
            "color_match_frames": ("INT", {"default": 0, "min": 0, "max": 96}),
            "status_card": ("INT", {"default": 0, "min": 0, "max": 1}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "SwapChain"

    async def run(self, images, ref_image, ctx, fps, filename_prefix, color_match_frames=0, status_card=0):
        return await asyncio.to_thread(self._run, images, ref_image, ctx, fps, filename_prefix, color_match_frames, status_card)

    def _run(self, images, ref_image, ctx, fps, filename_prefix, color_match_frames=0, status_card=0):
        import folder_paths
        c = json.loads(ctx)
        ov = int(c.get("ov") or 0)
        frames = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        if ov and int(color_match_frames) > 0 and c.get("prev"):
            # same instants as the overlap: the previous step's real last frames vs what this chunk made of them
            frames = core.color_match(frames, ov, core.tail_frames(c["prev"], ov, c["w"], c["h"]), int(color_match_frames))
        else:
            frames = frames[ov:]
        size = (c["w"], c["h"])
        clip = core.write_clip(frames, fps, size, os.path.join(c["work"], "newclip.mp4"))
        ref = (ref_image.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)[0]
        ref_png = os.path.join(c["work"], "ref_processed.png")
        Image.fromarray(ref).save(ref_png)
        new_cursor = c["cursor"] + c["frames"] / core.FPS
        done = new_cursor >= c["total"] - 0.05
        state = {"tag": core.TAG, "prompt": c["prompt"], "cursor": round(new_cursor, 4), "total": round(c["total"], 4),
                 "done": done, "step": c["step"], "w": c["w"], "h": c["h"], "cuts": c["cuts"], "has_audio": c["has_audio"]}
        out_dir = folder_paths.get_output_directory()
        sub, prefix = os.path.split(filename_prefix)
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        name = f"{prefix}_{int(time.time())}_{c['step']:03d}.mp4"
        out_path = os.path.join(out_dir, sub, name)
        core.finish(c["work"], clip, c["prev"], c.get("guide_clip") or None, c.get("guide_audio") or None, ref_png, state,
                    out_path, c["total"])
        shown = [{"filename": name, "subfolder": sub, "type": "output"}]
        if int(status_card):
            left = 0 if done else core.steps_left(new_cursor, c["total"], int(c.get("chunk") or core.MAX_FRAMES), c["cuts"],
                                                  bool(c.get("tail")))
            card = name[:-4] + "_status.png"     # sorts after the .mp4, so the video stays item 0 of the render
            status_card_png = os.path.join(out_dir, sub, card)
            _write_status_card(status_card_png, int(c["step"]), left, done)
            shown.append({"filename": card, "subfolder": sub, "type": "output"})
        return {"ui": {"images": shown, "animated": (True,)}}


class SwapChainLoadLTX:
    """Start or continue a chain for the LTX Ripple graph (first-frame edit + propagation).

    mode 0 (swap, default). Step 1: video = the guide clip; the graph edits frame 0 itself (Qwen) from the reference photo
    it loads on its own. Later steps: video = the previous step's result; its last frame becomes the next chunk's finished
    first frame (own_present = "1"), so each chunk starts exactly where the last one ended.

    mode 1 (frames only). Returns the guide's frame at the start of every chunk (chunk_frames) so you can edit them; no model
    runs. mode 2 (your frames). Step 1 takes the edited frames as a stack: the pictures of `bundle` (video2) first, then
    image1..image8 in slot order. (A bundle can only travel in video2: Graydient turns a bundle given in an image slot into a picture of it.)
    Chunk k then starts on picture k. The stack is stored in the result file, so later steps need only the previous hash.

    Outputs mirror LoadVideoFromURLMatched (frames = N-1 source frames, width, height, length N = 8k+1, fps) plus the first
    frame to use, a ctx string for Finish and (mode 1) the stills.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "max_chunk_frames": ("INT", {"default": 160, "min": 8, "max": 400, "step": 1}),
            "target_pixels": ("INT", {"default": 393216, "min": 65536, "max": 4194304}),
            "max_seconds": ("INT", {"default": 60, "min": 1, "max": 600}),
        }, "optional": {
            "bundle": ("STRING", {"default": ""}),
            "image1": ("STRING", {"default": ""}), "image2": ("STRING", {"default": ""}),
            "image3": ("STRING", {"default": ""}), "image4": ("STRING", {"default": ""}),
            "image5": ("STRING", {"default": ""}), "image6": ("STRING", {"default": ""}),
            "image7": ("STRING", {"default": ""}), "image8": ("STRING", {"default": ""}),
            "mode": ("INT", {"default": 0, "min": 0, "max": 2}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "FLOAT", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "width", "height", "length", "fps", "own_frame", "own_present", "ctx", "chunk_frames")
    FUNCTION = "run"
    CATEGORY = "SwapChain"

    async def run(self, video, width, height, max_chunk_frames, target_pixels, max_seconds, bundle="",
                  image1="", image2="", image3="", image4="", image5="", image6="", image7="", image8="", mode=0):
        return await asyncio.to_thread(self._run, video, width, height, max_chunk_frames, target_pixels, max_seconds, bundle,
                                       [image1, image2, image3, image4, image5, image6, image7, image8], mode)

    def _run(self, video, width, height, max_chunk_frames, target_pixels, max_seconds, bundle="", images=(), mode=0):
        import torch
        mode = int(mode)
        work = tempfile.mkdtemp(prefix="swapchain_")
        src = _resolve(video, work)
        if not src:
            raise RuntimeError("video is empty: give the guide clip (/video1:URL) or the previous step (/video1:R<hash>)")
        own, own_present, stills, own_frames = None, "", [], False
        if core.is_chain(src):
            prev = src
            st, vsec = core.read_state(prev)
            if st.get("done"):
                raise RuntimeError("this chain is already complete (done=true): nothing left to render")
            if st.get("engine") != "ltx":
                raise RuntimeError("this chain was made by the H3 workflow, not the ripple one")
            w, h = int(st["w"]), int(st["h"])
            cursor, total, step = float(st.get("cursor") or vsec), float(st["total"]), int(st.get("step", 0)) + 1
            max_chunk_frames = int(st.get("chunk") or max_chunk_frames)   # fixed on step 1: later steps cannot move the chunks
            guide_path, guide_idx = prev, 1
            has_audio = bool(st.get("has_audio"))
            own = core.last_frame(prev, os.path.join(work, "last.png"))
            own_present = "1"
            guide_clip = guide_audio = ""
            own_frames = bool(st.get("own_frames"))
            stills = core.extract_covers(prev, work) if own_frames else []
        else:
            prev = None
            info = core.probe(src)
            vs, secs = core.main_video(info)
            total = min(secs, float(max_seconds))
            w, h = core.pick_size(int(vs["width"]), int(vs["height"]), width, height, target_pixels)
            guide_path = core.prepare_guide(src, os.path.join(work, "guide.mp4"), w, h, total)
            guide_idx = 0
            audio_path = core.extract_audio(src, os.path.join(work, "guide_audio.m4a"), total)
            has_audio = bool(audio_path)
            cursor, step = 0.0, 1
            guide_clip, guide_audio = guide_path, audio_path or ""
            if mode == 2:
                own_frames = True
                stills = _stack(work, bundle, images)
                need = core.count_steps_ltx(total, max_chunk_frames)
                if not stills:
                    raise RuntimeError("your-frames mode needs the edited frames: image1..image8 and/or video2 (a bundle)")
                if len(stills) < need:
                    raise RuntimeError(f"this clip takes {need} steps (one edited frame each) and you gave {len(stills)}: "
                                       f"run frames-only mode (slot6=1) and edit every frame it returns")
        steps_total = core.count_steps_ltx(total, max_chunk_frames)
        if mode == 1:
            # frames only: the target frame at the start of every chunk, for the person to edit; no model runs
            starts = core.chunk_starts_ltx(total, max_chunk_frames)
            frames_arr = np.concatenate([core.decode_guide(guide_path, guide_idx, t, 1, w, h) for t in starts], axis=0)
            blank = torch.zeros((1, h, w, 3), dtype=torch.float32)
            ctx = {"engine": "ltx", "mode": 1, "work": work, "starts": [round(t, 3) for t in starts], "total": total,
                   "steps_total": steps_total, "w": w, "h": h}
            return (blank, int(w), int(h), 9, 24.0, blank, "", json.dumps(ctx), _image_tensor(frames_arr))
        frames = core.pick_frames_ltx(cursor, total, max_chunk_frames)
        gf = core.decode_guide(guide_path, guide_idx, cursor, frames, w, h)
        # your-frames mode: chunk k starts on the k-th picture of the stack (instead of the AI edit or the previous last frame)
        if own_frames and step - 1 < len(stills):
            own = _fit_png(stills[step - 1], w, h, work)
            own_present = "1"
        own_t = _image_tensor(own) if own else torch.zeros((1, h, w, 3), dtype=torch.float32)
        ctx = {"engine": "ltx", "mode": 0, "prev": prev, "work": work, "cursor": cursor, "frames": frames, "total": total,
               "step": step, "steps_total": steps_total, "w": w, "h": h, "cuts": [], "own_frames": own_frames,
               "chunk": int(max_chunk_frames),
               "has_audio": has_audio, "guide_clip": guide_clip, "guide_audio": guide_audio, "prompt": "",
               "stills": [_fit_png(p, w, h, work, f"carry{i + 1:02d}.png") for i, p in enumerate(stills)]}
        return (_image_tensor(gf), int(w), int(h), int(frames + 1), 24.0, own_t, own_present, json.dumps(ctx),
                torch.zeros((1, h, w, 3), dtype=torch.float32))


class SwapChainFinishLTX:
    """Stitch a ripple chunk (N frames: the first-frame edit, then the chunk) after the previous video and save."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "ctx": ("STRING", {"forceInput": True}),
            "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
            "drop_first": ("INT", {"default": 1, "min": 0, "max": 4}),
            "filename_prefix": ("STRING", {"default": "video/SwapRipple"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "SwapChain"

    async def run(self, images, ctx, fps, drop_first, filename_prefix):
        return await asyncio.to_thread(self._run, images, ctx, fps, drop_first, filename_prefix)

    def _run(self, images, ctx, fps, drop_first, filename_prefix):
        import folder_paths
        c = json.loads(ctx)
        if c.get("mode") == 1:
            # frames only: save the target frame of every chunk as its own picture, named by chunk and time
            arr = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            out_dir = folder_paths.get_output_directory()
            os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
            stamp, total, shown = int(time.time()), len(arr), []
            for i, (a, t) in enumerate(zip(arr, c["starts"])):
                name = f"chunk_{i + 1:02d}_of_{total:02d}_at_{t:.1f}s_{stamp}.png"
                Image.fromarray(a).save(os.path.join(out_dir, "frames", name))
                shown.append({"filename": name, "subfolder": "frames", "type": "output"})
            return {"ui": {"images": shown}}
        arr = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)[int(drop_first):][: c["frames"]]
        clip = core.write_clip(arr, fps, (c["w"], c["h"]), os.path.join(c["work"], "newclip.mp4"))
        new_cursor = c["cursor"] + c["frames"] / core.FPS
        done = new_cursor >= c["total"] - 0.05
        state = {"tag": core.TAG, "engine": "ltx", "cursor": round(new_cursor, 4), "total": round(c["total"], 4), "done": done,
                 "step": c["step"], "steps_total": c["steps_total"], "w": c["w"], "h": c["h"], "has_audio": c["has_audio"],
                 "own_frames": bool(c.get("own_frames")), "chunk": c.get("chunk")}
        out_dir = folder_paths.get_output_directory()
        sub, prefix = os.path.split(filename_prefix)
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        name = f"{prefix}_{int(time.time())}_{c['step']:03d}.mp4"
        out_path = os.path.join(out_dir, sub, name)
        core.finish(c["work"], clip, c["prev"], c.get("guide_clip") or None, c.get("guide_audio") or None,
                    c.get("stills") or None, state, out_path, c["total"])
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"SwapChainLoad": SwapChainLoad, "SwapChainFinish": SwapChainFinish,
                       "SwapChainLoadLTX": SwapChainLoadLTX, "SwapChainFinishLTX": SwapChainFinishLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"SwapChainLoad": "Swap Chain - Load / Continue",
                              "SwapChainFinish": "Swap Chain - Stitch and Save",
                              "SwapChainLoadLTX": "Swap Chain (ripple) - Load / Continue",
                              "SwapChainFinishLTX": "Swap Chain (ripple) - Stitch and Save"}
