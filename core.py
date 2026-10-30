"""SwapChain core: pure python + ffmpeg (no ComfyUI imports), so it can be tested on its own.

A chain is ONE mp4 that carries its own state, so the next step needs only that file (same idea as ComfyUI-VoiceChain):
  video stream 0   the swapped clips stitched so far (h264, stream-copy concat)
  audio stream 0   the guide video's own audio, FULL length (aac, copied on later steps); absent if the guide is silent
  video stream 1   the GUIDE video, already cropped/scaled to the working size at 24 fps (h264); later steps read it at the cursor
  video stream 2   the character reference, already background-removed and scaled (attached_pic), so later steps skip BiRefNet
  metadata comment JSON: prompt, cursor (seconds consumed), total, done, step, size, scene cuts
The cursor is also the main video's duration, so a file with lost metadata can still be continued.
"""
import json
import math
import os
import re
import subprocess

import numpy as np

FPS = 24
SR = 44100
MAX_FRAMES = 124          # 17k+5 grid; 124 = 5.17 s. 90 f measured 72-105 s `Prompt executed` at 704x384 (4090)
MIN_FRAMES = 22
TAG = "swapchain/1"
DEFAULT_DETAILS = "Person, average height, average build, natural hair, casual clothing"   # step 1 with no description


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def snap_up(frames):
    f = max(MIN_FRAMES, int(frames))
    while f % 17 != 5:
        f += 1
    return f


def snap_down(frames):
    return max(MIN_FRAMES, ((int(frames) - 5) // 17) * 17 + 5)


def snap32(v, lo=256, hi=1280):
    return int(min(hi, max(lo, round(float(v) / 32.0) * 32)))


def probe(path):
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    if r.returncode:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr[-300:]!r}")
    return json.loads(r.stdout)


def main_video(info):
    """-> (stream dict, seconds) of the first real (non attached-picture) video stream."""
    for s in info["streams"]:
        if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"):
            secs = float(s.get("duration") or info["format"].get("duration") or 0)
            return s, secs
    raise RuntimeError("no video stream")


def has_audio(info):
    return any(s["codec_type"] == "audio" for s in info["streams"])


def pick_size(src_w, src_h, width=0, height=0, target_pixels=262144, lo=256, hi=1280):
    """Working size, multiples of 32, keeping the source aspect. 0 = not given (width/height fields use 0 as 'auto')."""
    width, height = int(width or 0), int(height or 0)
    if width <= 0 and height <= 0:
        s = math.sqrt(target_pixels / float(src_w * src_h))
        w, h = src_w * s, src_h * s
    elif width <= 0:
        w, h = height * src_w / float(src_h), height
    elif height <= 0:
        w, h = width, width * src_h / float(src_w)
    else:
        w, h = width, height
    return snap32(w, lo, hi), snap32(h, lo, hi)


def prepare_guide(src, out_path, w, h, max_seconds):
    """Transcode the user's guide once: 24 fps, cover-crop to w x h, capped length, no audio, h264."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-t", f"{max_seconds:.3f}", "-map", "0:v:0",
           "-vf", f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-video_track_timescale", "24000", "-movflags", "+faststart", out_path]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"guide transcode failed: {r.stderr[-400:]!r}")
    return out_path


def extract_audio(src, out_path, max_seconds):
    """Guide's own audio as aac, or None when the source has none."""
    if not has_audio(probe(src)):
        return None
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", src, "-t", f"{max_seconds:.3f}", "-map", "0:a:0", "-vn", "-ac", "2",
              "-ar", str(SR), "-c:a", "aac", "-b:a", "160k", out_path])
    if r.returncode:
        raise RuntimeError(f"guide audio extract failed: {r.stderr[-300:]!r}")
    return out_path


def detect_cuts(path, threshold=0.4):
    """Scene-cut times (seconds) in the prepared guide."""
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
    return [round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr.decode(errors="replace"))]


def pick_frames(cursor, total, max_frames=MAX_FRAMES, cuts=(), tol=0.12):
    """This step's frame count on the 17k+5 grid. Prefers the longest chunk (not shorter than 60% of the cap) that ends
    on a scene cut, so the next chunk starts on a clean cut; otherwise the cap. The last stretch is just rendered out."""
    max_frames = snap_down(max_frames)
    remaining = total - cursor
    if remaining <= max_frames / FPS + 0.05:
        return snap_up(int(math.ceil(remaining * FPS - 1e-6)))
    f = max_frames
    while f >= max(MIN_FRAMES, int(max_frames * 0.6)):
        t = cursor + f / FPS
        if any(abs(c - t) <= tol for c in cuts):
            return f
        f -= 17
    return max_frames


def decode_guide(path, video_index, cursor, frames, w, h):
    """Guide frames [frames, h, w, 3] uint8 from the `video_index`-th video stream of `path`, starting at `cursor` s.
    Past the end of the guide the last frame is repeated."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{cursor:.4f}", "-i", path, "-map", f"0:v:{video_index}",
           "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"guide decode failed: {r.stderr[-300:]!r}")
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = a.size // (w * h * 3)
    if n == 0:
        raise RuntimeError(f"no guide frames at {cursor:.2f}s (the guide ends before the cursor)")
    a = a[: n * w * h * 3].reshape(n, h, w, 3)
    if n < frames:
        a = np.concatenate([a, np.repeat(a[-1:], frames - n, axis=0)], axis=0)
    return a


def decode_audio(path, stream, cursor, frames):
    """Audio slice [2, N] float32 at SR for this chunk; zeros past the end / when there is no audio stream."""
    n = int(round(frames / FPS * SR))
    if stream is None:
        return np.zeros((2, n), np.float32)
    r = _run(["ffmpeg", "-v", "error", "-ss", f"{cursor:.4f}", "-i", path, "-t", f"{frames / FPS:.4f}", "-map", f"0:a:{stream}",
              "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"])
    if r.returncode:
        raise RuntimeError(f"audio slice failed: {r.stderr[-300:]!r}")
    a = np.frombuffer(r.stdout, dtype=np.float32)
    a = a.reshape(-1, 2).T.copy() if a.size else np.zeros((2, 0), np.float32)
    if a.shape[1] < n:
        a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
    return a[:, :n]


def read_state(path):
    """-> (state dict, main video seconds). Raises if this is not a swapchain mp4."""
    info = probe(path)
    tags = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
    raw = tags.get("comment", "")
    st = json.loads(raw) if raw.startswith("{") else {}
    if st.get("tag") != TAG:
        raise RuntimeError(f"{os.path.basename(path)} is not a swapchain video (no state in its comment tag)")
    return st, main_video(info)[1]


def is_chain(path):
    try:
        read_state(path)
        return True
    except Exception:
        return False


def extract_cover(path, out_png):
    info = probe(path)
    for s in info["streams"]:
        if s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic"):
            r = _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-map", f"0:{s['index']}", "-frames:v", "1", out_png])
            if r.returncode:
                raise RuntimeError(f"cover extract failed: {r.stderr[-300:]!r}")
            return out_png
    raise RuntimeError("the video has no attached reference image")


def write_clip(frames_rgb, fps, size, out_path):
    """frames_rgb: uint8 [F, H, W, 3] -> video-only h264 mp4 (fixed params so clips concat by stream copy)."""
    h, w = frames_rgb.shape[1:3]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if size and (w, h) != tuple(size):
        cmd += ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos"]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-video_track_timescale", "24000", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames_rgb).tobytes(), capture_output=True)
    if r.returncode:
        raise RuntimeError(f"clip encode failed: {r.stderr[-400:]!r}")
    return out_path


def finish(work, new_clip, prev_video, guide_clip, guide_audio, ref_png, state, out_path, total):
    """Stitch new_clip after prev_video's main stream and write the self-contained mp4.

    Step 1 passes guide_clip / guide_audio (files made by prepare_guide / extract_audio); later steps pass prev_video and
    both are stream-copied out of it (never re-encoded)."""
    parts = []
    if prev_video:
        prev_v = os.path.join(work, "prev_video.mp4")
        r = _run(["ffmpeg", "-y", "-v", "error", "-i", prev_video, "-map", "0:v:0", "-c", "copy", prev_v])
        if r.returncode:
            raise RuntimeError(f"prev video split failed: {r.stderr[-300:]!r}")
        parts.append(prev_v)
    parts.append(new_clip)
    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + p.replace("\\", "/").replace("'", r"'\''") + "'\n")
    joined = os.path.join(work, "joined.mp4")
    r = _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    if r.returncode:
        raise RuntimeError(f"concat failed: {r.stderr[-300:]!r}")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", joined]
    n_in = 1
    if prev_video:
        cmd += ["-i", prev_video]
        a_map = f"{n_in}:a:0" if state.get("has_audio") else None
        g_map = f"{n_in}:v:1"
        n_in += 1
    else:
        a_map = g_map = None
        if guide_audio:
            cmd += ["-i", guide_audio]
            a_map = f"{n_in}:a:0"
            n_in += 1
        cmd += ["-i", guide_clip]
        g_map = f"{n_in}:v:0"
        n_in += 1
    cmd += ["-i", ref_png]
    ref_map = f"{n_in}:v:0"
    cmd += ["-map", "0:v:0"]
    if a_map:
        cmd += ["-map", a_map]
    if state.get("done"):
        # finished: nothing will continue from this file, so deliver a clean video + audio file that any player handles
        cmd += ["-c:v:0", "copy"]
    else:
        # the carried guide is not a playable track: no default flag, so players keep to stream 0
        cmd += ["-map", g_map, "-map", ref_map, "-c:v:0", "copy", "-c:v:1", "copy", "-c:v:2", "png",
                "-disposition:v:0", "default", "-disposition:v:1", "0", "-disposition:v:2", "attached_pic"]
    if a_map:
        cmd += ["-c:a", "copy"]
    cmd += ["-metadata", "comment=" + json.dumps(state, separators=(",", ":")), "-movflags", "+faststart"]
    if state.get("done"):
        cmd += ["-t", f"{total:.4f}"]
    cmd += [out_path]
    r = _run(cmd)
    if r.returncode:
        raise RuntimeError(f"final mux failed: {r.stderr[-400:]!r}")
    return out_path
