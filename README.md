# ComfyUI-SwapChain

Chained video-to-video steps where the result mp4 carries its own state (guide video, its audio, the processed
reference picture, prompt, size, cursor), so a long clip can be processed in several short ComfyUI jobs.
Continue a chain by passing only the previous step's mp4. Pure python + ffmpeg; no model code.

Nodes: `SwapChainLoad` (start / continue; outputs the guide slice on the 17k+5 frame grid, audio slice, size, reference,
prompt, ctx) and `SwapChainFinish` (stitch by stream copy and save). Idea borrowed from ComfyUI-VoiceChain.

Test: `python tests/test_chain.py` (needs ffmpeg and torch).
