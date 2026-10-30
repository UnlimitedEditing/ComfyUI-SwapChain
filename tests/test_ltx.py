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
    gf, w, h, length, fps, own, own_present, ctx = asyncio.run(N["SwapChainLoadLTX"]().run(prev, 0, 0, 96, 393216, 60))
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
print("LTX CHAIN OK")
