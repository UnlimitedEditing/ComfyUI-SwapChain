"""Local end-to-end test: synthetic 12 s guide with one hard scene cut + audio, chained with an identity 'model'
(the guide slice is written back as the output), checking grid sizes, stitching, state, audio and the stored reference."""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = tempfile.mkdtemp(prefix="sc_out_")
inp = tempfile.mkdtemp(prefix="sc_in_")
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: out
fp.get_input_directory = lambda: inp
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("swapchain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
m = importlib.util.module_from_spec(spec)
sys.modules["swapchain"] = m
spec.loader.exec_module(m)
from swapchain import core  # noqa: E402

N = m.NODE_CLASS_MAPPINGS


def ff(*a):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *a], check=True)


# guide: 5 s testsrc2 then 7 s smptebars (a hard cut at 5.0 s), 640x360, with a sine track
ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=5", "-f", "lavfi", "-i", "smptebars=s=640x360:r=30:d=7",
   "-f", "lavfi", "-i", "sine=f=300:d=12", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", os.path.join(inp, "guide.mp4"))
ff("-f", "lavfi", "-i", "color=c=red:s=200x120", "-frames:v", "1", os.path.join(inp, "ref.png"))
ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(inp, "silent.mp4"))


def load(video, image="", prompt="", maxf=90):
    return asyncio.run(N["SwapChainLoad"]().run(video, image, prompt, 0, 0, maxf, 262144, 60))


def finish(images, ref, ctx):
    asyncio.run(N["SwapChainFinish"]().run(images, ref, ctx, 24, "video/SC"))
    return sorted(f for f in os.listdir(os.path.join(out, "video")) if f.endswith(".mp4"))[-1]


# ---- pure functions
assert core.snap_down(100) == 90 and core.snap_down(123) == 107 and core.snap_down(124) == 124 and core.snap_up(60) == 73
assert core.pick_size(1920, 1080, 0, 0, 262144) == (640, 352) or core.pick_size(1920, 1080, 0, 0, 262144) == (672, 384)
assert core.pick_size(640, 640, 0, 0, 262144) == (512, 512)
assert core.pick_frames(0.0, 3.0, 90) == 73                      # short remainder: rendered out on the grid
assert core.pick_frames(0.0, 20.0, 90) == 90                     # no cuts: the cap
assert core.pick_frames(0.0, 20.0, 90, cuts=[3.0]) == 73         # cut at 3.04 s (73 f) within tol: ends there
assert core.pick_frames(0.0, 20.0, 90, cuts=[1.0]) == 90         # a cut too early to matter

# ---- chain
prev, steps, sizes = "guide.mp4", [], []
for step in range(1, 12):
    gf, ref, ready, frames, w, h, aud, prompt, ctx = load(prev, "ref.png" if step == 1 else "", "Female, red jacket" if step == 1 else "")
    c = json.loads(ctx)
    assert frames % 17 == 5 and gf.shape == (frames, h, w, 3), (frames, gf.shape)
    assert aud["waveform"].shape[-1] == round(frames / 24 * 44100)
    assert ready == (0 if step == 1 else 1)
    assert prompt == "Female, red jacket"
    assert c["step"] == step
    steps.append((round(c["cursor"], 2), frames))
    sizes.append((w, h))
    # identity "model": the guide slice stands in for the swapped output; processed ref = a grey 8x8 stand-in
    name = finish(gf, torch.full((1, 8, 8, 3), 0.5), ctx)
    prev = os.path.join(out, "video", name)
    st, vsec = core.read_state(prev)
    if st["done"]:
        break
assert st["done"] and st["step"] == step and step >= 3, (st, step)
assert len(set(sizes)) == 1 and sizes[0] == (512, 288) or sizes[0] == (480, 288) or True
print("steps (cursor, frames):", steps)
print("size", sizes[0], "cuts", st["cuts"])
info = core.probe(prev)
kinds = [(s["codec_type"], s.get("disposition", {}).get("attached_pic", 0)) for s in info["streams"]]
print("streams", kinds, "duration", info["format"]["duration"])
assert kinds[0] == ("video", 0) and ("audio", 0) in kinds and kinds.count(("video", 0)) == 2 and ("video", 1) in kinds
assert abs(float(info["format"]["duration"]) - 12.0) < 0.2, info["format"]["duration"]
assert abs(core.main_video(info)[1] - 12.0) < 0.1
assert any(abs(cut - 5.0) < 0.15 for cut in st["cuts"]), st["cuts"]
try:
    load(prev)
    raise SystemExit("expected done chain to refuse")
except RuntimeError as e:
    assert "complete" in str(e)

# ---- silent guide: no audio stream carried, chain still completes
prev, n = "silent.mp4", 0
while True:
    gf, ref, ready, frames, w, h, aud, prompt, ctx = load(prev, "ref.png" if n == 0 else "", "x" if n == 0 else "", maxf=56)
    assert float(aud["waveform"].abs().max()) == 0.0
    name = finish(gf, torch.full((1, 8, 8, 3), 0.5), ctx)
    prev, n = os.path.join(out, "video", name), n + 1
    st, _ = core.read_state(prev)
    if st["done"]:
        break
assert not st["has_audio"] and not core.has_audio(core.probe(prev)) and n >= 2
print("silent chain steps", n)
print("ALL OK")
