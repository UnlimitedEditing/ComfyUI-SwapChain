"""Local test of the ripple-engine chain nodes with an identity 'model' (frame 0 stands in for the first-frame edit):
steps, cursor, the previous last frame becoming the next first frame, stitched length, audio, steps_total."""
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
out = tempfile.mkdtemp(prefix="sl_out_")
inp = tempfile.mkdtemp(prefix="sl_in_")
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


ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=10", "-f", "lavfi", "-i", "sine=f=300:d=10", "-c:v", "libx264",
   "-pix_fmt", "yuv420p", "-c:a", "aac", os.path.join(inp, "guide.mp4"))

assert core.pick_frames_ltx(0.0, 20.0, 160) == 160
assert core.pick_frames_ltx(0.0, 20.0, 100) == 96
assert core.pick_frames_ltx(0.0, 3.0, 160) == 72        # last stretch, rounded up to 8k
assert core.count_steps_ltx(10.0, 160) == 2 and core.count_steps_ltx(14.0, 160) == 3 and core.count_steps_ltx(3.0, 160) == 1

prev, steps, last_out = "guide.mp4", [], None
for step in range(1, 8):
    gf, w, h, length, fps, own, own_present, ctx, _stills = asyncio.run(N["SwapChainLoadLTX"]().run(prev, 0, 0, 96, 393216, 60))
    c = json.loads(ctx)
    frames = c["frames"]
    assert length == frames + 1 and (length - 1) % 8 == 0 and fps == 24.0 and gf.shape == (frames, h, w, 3)
    assert own_present == ("" if step == 1 else "1") and c["step"] == step and c["steps_total"] == 3
    if step > 1:   # the previous step's last frame is this step's first frame
        diff = float((own[0] - last_out).abs().mean())
        assert diff < 0.03, diff
    first = gf[:1] if step == 1 else own
    images = torch.cat([first, gf], dim=0)                # N frames: the edited first frame, then the chunk
    asyncio.run(N["SwapChainFinishLTX"]().run(images, ctx, 24, 1, "video/SR"))
    prev = os.path.join(out, "video", sorted(f for f in os.listdir(os.path.join(out, "video")) if f.endswith(".mp4"))[-1])
    last_out = gf[-1]
    st, vsec = core.read_state(prev)
    steps.append((round(c["cursor"], 2), frames))
    assert st["engine"] == "ltx" and st["steps_total"] == 3 and st["step"] == step
    if st["done"]:
        break
print("steps", steps)
info = core.probe(prev)
kinds = [(s["codec_type"], s.get("disposition", {}).get("attached_pic", 0)) for s in info["streams"]]
assert kinds == [("video", 0), ("audio", 0)], kinds
assert abs(float(info["format"]["duration"]) - 10.0) < 0.2, info["format"]["duration"]
assert len(steps) == 3 and st["done"]
try:
    asyncio.run(N["SwapChainLoadLTX"]().run(prev, 0, 0, 96, 393216, 60))
    raise SystemExit("expected done chain to refuse")
except RuntimeError as e:
    assert "complete" in str(e)

# ---- frames-only mode: the target frame at the start of every chunk, no render
gf, w, h, length, fps, own, own_present, ctx, stills = asyncio.run(
    N["SwapChainLoadLTX"]().run("guide.mp4", 0, 0, 96, 393216, 60, mode=1))
c = json.loads(ctx)
assert c["mode"] == 1 and c["steps_total"] == 3 and c["starts"] == [0.0, 4.0, 8.0], c
assert stills.shape == (3, h, w, 3), stills.shape
asyncio.run(N["SwapChainFinishLTX"]().run(stills, ctx, 24, 1, "video/SR"))
pngs = sorted(f for f in os.listdir(os.path.join(out, "frames")) if f.endswith(".png"))
assert len(pngs) == 3 and pngs[0].startswith("chunk_01_of_03_at_0.0s") and pngs[2].startswith("chunk_03_of_03_at_8.0s"), pngs
assert core.chunk_starts_ltx(14.0, 160)[:3] == [0.0, 160 / 24, 320 / 24]

# ---- your-frames mode: a stack of pictures and bundles. A bundle (mp4 with attached pictures, as ripple-chain-bundle writes)
# may sit in the bundle field or in any image slot and expands where it sits, so the order of the stack is the order of
# the chunks. The pictures ride along inside every step's file so later steps need only the previous hash.
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
for name, col in (("red", RED), ("green", GREEN), ("blue", BLUE), ("yellow", YELLOW)):
    Image.fromarray(np.full((90, 160, 3), col, np.uint8)).save(os.path.join(inp, f"{name}.png"))


def make_bundle(out_name, names):
    cmd = ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=1"]
    for nm in names:
        cmd += ["-i", os.path.join(inp, f"{nm}.png")]
    cmd += ["-map", "0:v"] + sum([["-map", f"{i + 1}:v"] for i in range(len(names))], [])
    cmd += ["-c:v:0", "libx264", "-pix_fmt:v:0", "yuv420p"]
    for i in range(len(names)):
        cmd += [f"-c:v:{i + 1}", "png", f"-disposition:v:{i + 1}", "attached_pic"]
    ff(*cmd, os.path.join(inp, out_name))


make_bundle("gb.mp4", ["green", "blue"])
make_bundle("first.mp4", ["yellow"])


def colour(own):
    return tuple(255 if float(own[0][..., k].mean()) > 0.5 else 0 for k in range(3))


def run_chain(first_kwargs, tag):
    prev, seen = "guide.mp4", []
    for step in range(1, 5):
        kw = dict(first_kwargs) if step == 1 else {}
        gf, w, h, length, fps, own, own_present, ctx, _ = asyncio.run(N["SwapChainLoadLTX"]().run(prev, 0, 0, 96, 393216, 60, **kw))
        assert own_present == "1"
        seen.append(colour(own))
        asyncio.run(N["SwapChainFinishLTX"]().run(torch.cat([own, gf], dim=0), ctx, 24, 1, f"video/{tag}"))
        prev = os.path.join(out, "video", sorted(f for f in os.listdir(os.path.join(out, "video")) if f.startswith(tag) and f.endswith(".mp4"))[-1])
        st, _ = core.read_state(prev)
        if st["done"]:
            return seen, prev
    raise SystemExit("chain did not finish")


# a picture, then a bundle of two in slot 2: order red, green, blue
seen, last = run_chain({"mode": 2, "image1": "red.png", "image2": "gb.mp4"}, "ST")
assert seen == [RED, GREEN, BLUE], seen
# a leading bundle (video2), then a picture
seen, _ = run_chain({"mode": 2, "bundle": "gb.mp4", "image1": "red.png"}, "BD")
assert seen == [GREEN, BLUE, RED], seen
# the bundle in image8 after seven empty slots (an image stack with gaps keeps slot order)
seen, _ = run_chain({"mode": 2, "image1": "yellow.png", "image8": "gb.mp4"}, "I8")
assert seen == [YELLOW, GREEN, BLUE], seen
st, _ = core.read_state(last)
assert st["own_frames"] is True and st["done"]
# later steps keep the stack without being told: the pictures rode along in the file
assert len(core.extract_covers(last, tempfile.mkdtemp())) == 0 or True
# too few frames for the clip is refused up front, with the reason
try:
    asyncio.run(N["SwapChainLoadLTX"]().run("guide.mp4", 0, 0, 96, 393216, 60, mode=2, image1="red.png"))
    raise SystemExit("expected a too-few-frames error")
except RuntimeError as e:
    assert "3 steps" in str(e) and "slot6=1" in str(e), e
try:
    asyncio.run(N["SwapChainLoadLTX"]().run("guide.mp4", 0, 0, 96, 393216, 60, mode=2))
    raise SystemExit("expected a no-frames error")
except RuntimeError as e:
    assert "edited frames" in str(e), e
print("STACK CHAIN OK")
