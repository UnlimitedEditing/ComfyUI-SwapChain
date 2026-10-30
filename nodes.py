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
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("guide_frames", "ref_image", "ref_ready", "frames", "width", "height", "audio", "prompt", "ctx")
    FUNCTION = "run"
    CATEGORY = "SwapChain"

    async def run(self, video, image1, prompt, width, height, max_chunk_frames, target_pixels, max_seconds):
        return await asyncio.to_thread(self._run, video, image1, prompt, width, height, max_chunk_frames,
                                       target_pixels, max_seconds)

    def _run(self, video, image1, prompt, width, height, max_chunk_frames, target_pixels, max_seconds):
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

        frames = core.pick_frames(cursor, total, max_chunk_frames, cuts)
        gf = core.decode_guide(guide_path, guide_idx, cursor, frames, w, h)
        seg = core.decode_audio(audio_path, audio_idx, cursor, frames) if audio_path else \
            core.decode_audio(None, None, cursor, frames)
        ctx = {"prev": prev, "work": work, "cursor": cursor, "frames": frames, "total": total, "step": step,
               "prompt": use_prompt, "w": w, "h": h, "cuts": cuts, "has_audio": has_audio,
               "guide_clip": "" if prev else guide_path, "guide_audio": "" if prev or not audio_path else audio_path}
        audio_out = {"waveform": torch.from_numpy(seg).unsqueeze(0), "sample_rate": core.SR}
        return (_image_tensor(gf), _image_tensor(ref), int(ref_ready), int(frames), int(w), int(h), audio_out,
                use_prompt, json.dumps(ctx))


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
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "SwapChain"

    async def run(self, images, ref_image, ctx, fps, filename_prefix):
        return await asyncio.to_thread(self._run, images, ref_image, ctx, fps, filename_prefix)

    def _run(self, images, ref_image, ctx, fps, filename_prefix):
        import folder_paths
        c = json.loads(ctx)
        frames = (images.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
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
        return {"ui": {"images": [{"filename": name, "subfolder": sub, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"SwapChainLoad": SwapChainLoad, "SwapChainFinish": SwapChainFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"SwapChainLoad": "Swap Chain - Load / Continue",
                              "SwapChainFinish": "Swap Chain - Stitch and Save"}
