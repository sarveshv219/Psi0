"""What one cached Qwen3-VL frame actually costs -> token count, hidden width, bytes, corpus size.

Step 2.5, and it exists to replace an estimate with a measurement before ~100 GB is committed. The
cache plan was sized from config constants (2048 hidden from ``--model.view_feature_dim``, a 28-px
patch from ``max_pixels: 576*28*28``); this reads the real numbers off a real forward on a real
pilot frame.

**What Psi0 caches is the whole last hidden state, not "vision patches".** ``Psi0Model.forward``
calls the VLM once over text AND image together and takes ``hidden_states[-1]`` entire — template,
instruction and image tokens — then hands it to the action header as ``views``. So vision and
language are not separable and the cache key is (frame, command), not (frame). One command per
episode is what collapses that back to one forward per frame.

**The resize check is the point of the `--pixels` sweep.** 448x252 was chosen so Qwen patchifies the
render with no resize and no crop. That only holds if the processor's own ``min_pixels`` /
``max_pixels`` leave the frame alone — and the processor's defaults are NOT Psi0's config values, so
both are reported. A grid other than 16x9 means something rescaled and the property is lost.

Runs in the `fcrl` conda env: transformers 4.57.6 there already has Qwen3VL, and nothing here needs
torchcodec (frames are decoded with imageio, as `verify_video` does). Weights land in $HF_HOME.

Usage:
    P=/home/sarvesh/miniconda3/envs/fcrl/bin/python
    $P scripts/data/vibe/probe_qwen_tokens.py --rollouts data/vibe_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# The command vocabulary this corpus is meant to carry: one short imperative in which exactly ONE
# token varies. The invariant framing goes in the system turn, where it carries no information
# between episodes but puts an instruct-tuned backbone in the right regime; the user turn is the
# image plus the request. Token counts depend on this text, so the probe measures the real thing
# rather than a placeholder.
SYSTEM = ("You control a humanoid robot facing a coloured cube. "
          "Output the whole-body motion that satisfies the instruction.")
def command(color: str) -> str:
    return f"Flip the cube so the {color} face is up."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, help="collect_rollouts.py output dir")
    ap.add_argument("--episode", default=None, help="episode dir name (default: first)")
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame index within the episode; -1 = the middle, where the cube is "
                         "most likely in view")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corpus-frames", type=int, default=818421,
                    help="frames in the FULL corpus, for the cache extrapolation. Default is the "
                         "size of the previous colour-first corpus")
    return ap.parse_args()


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    root = Path(args.rollouts)
    summary = json.loads((root / "summary.json").read_text())
    eps = sorted((root / "episodes").iterdir())
    d = (root / "episodes" / args.episode) if args.episode else eps[0]

    frames = np.stack(imageio.mimread(str(d / "frames.mp4"), memtest=False))
    meta = json.loads((d / "meta.json").read_text())
    t = (len(frames) // 2) if args.frame < 0 else args.frame
    img, color = frames[t], meta["commanded_color"]
    print(f"[probe] {d.name} frame {t}/{len(frames)}  {img.shape[1]}x{img.shape[0]}  "
          f"commanded={color}  (corpus rendered at "
          f"{summary['cam_width']}x{summary['cam_height']}, {summary['video_pix_fmt']})")
    assert (img.shape[1], img.shape[0]) == (summary["cam_width"], summary["cam_height"]), \
        "decoded frame size differs from what the corpus says it rendered"

    proc = AutoProcessor.from_pretrained(MODEL)
    ip = proc.image_processor
    print(f"[probe] processor defaults: min_pixels {getattr(ip, 'min_pixels', '?')}  "
          f"max_pixels {getattr(ip, 'max_pixels', '?')}  patch {getattr(ip, 'patch_size', '?')}  "
          f"merge {getattr(ip, 'merge_size', '?')}")
    print(f"[probe] frame is {img.shape[0] * img.shape[1]} px")

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": command(color)}]},
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[text], images=[img], return_tensors="pt")

    grid = batch["image_grid_thw"][0].tolist()
    merge = int(getattr(ip, "merge_size", 2))
    n_img_tok = int(np.prod(grid)) // (merge ** 2)
    ids = batch["input_ids"][0]
    S = int(ids.numel())
    # Count the image placeholders directly rather than trusting the grid arithmetic -- the two
    # disagreeing is exactly the silent-resize failure this probe exists to catch.
    img_tok_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_placeholder = int((ids == img_tok_id).sum())
    print(f"[probe] image_grid_thw {grid} (patch units)  merge {merge} -> {n_img_tok} image tokens")
    print(f"[probe] input_ids {S} tokens = {n_placeholder} image + {S - n_placeholder} text/template")
    exp_w, exp_h = summary["qwen_patch_grid"]
    got = (grid[2] // merge, grid[1] // merge)
    print(f"[probe] patch grid {got[0]} x {got[1]}, corpus designed for "
          f"{int(exp_w)} x {int(exp_h)}  "
          f"{'MATCH — no resize' if got == (int(exp_w), int(exp_h)) else 'MISMATCH — the processor rescaled'}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    with torch.no_grad():
        out = model(**{k: v.to(args.device) for k, v in batch.items()}, output_hidden_states=True)
    h = out.hidden_states[-1]
    D = h.shape[-1]
    print(f"[probe] hidden_states[-1] {tuple(h.shape)}  dtype {h.dtype}  "
          f"({len(out.hidden_states)} layers available)")

    per_frame = S * D * 2                       # bf16
    print()
    print(f"[cache] {S} tokens x {D} dims x 2 B = {per_frame / 1e3:.1f} KB per frame")
    print(f"[cache] full corpus {args.corpus_frames} frames:")
    for stride in (1, 2, 5, 10):
        n = args.corpus_frames // stride
        print(f"          stride {stride:2d} ({50 / stride:4.1f} Hz)  {n:>7} frames  "
              f"{n * per_frame / 1e9:7.1f} GB")
    print(f"[cache] + attn mask {S} B/frame, negligible; a 6-colour swap slice over 2048 windows "
          f"is {2048 * 6 * per_frame / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
