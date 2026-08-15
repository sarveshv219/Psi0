"""Can Qwen3-VL see the commanded colour, and does the resize path cost anything?

The go/no-go for the Psi0 plan, and a resolution A/B in one pass. Three conditions on identical
frames:

    native      512x288, no resize          144 image tokens
    psi0-near   320x240 NEAREST             80 tokens   (Psi0's shipped SONIC preset, as-is)
    psi0-bilin  320x240 BILINEAR            80 tokens   (same size, better kernel)

`native` vs `psi0-near` is the whole preset. `psi0-near` vs `psi0-bilin` isolates the interpolation
kernel from the resolution, which is the pair that says WHICH of the two matters. `ResizeImage` uses
NEAREST, which point-samples instead of averaging — on a colour task a thin or oblique face can
alias away entirely.

**What is measured.** Per frame: one VLM forward over (system prompt, image, command sentence);
take `hidden_states[-1]` at the image-token positions only and mean-pool to 2048 dims. Fit logistic
regression on those features to predict "is the commanded colour visible in this frame", label =
the corpus's own `commanded_visible_frac > 0.01` (read off the renderer's segmentation channel, not
inferred from pixels). Split by EPISODE so no frame of a test episode is seen in training. Report
ROC AUC overall and per commanded colour.

**Reference points**, measured on the previous corpus with different encoders — same label
definition, so the numbers are comparable in kind but not a controlled comparison: ClearCLIP's
detector reached **AUC 0.913**, a standardized-similarity detector **0.934**. Near 0.5 means the
vision path carries no colour and the plan is not worth the collection.

**What this does not establish.** That a better detector AUC means better generated motion. It is a
necessary condition (the colour must be legible to the frozen backbone at all) and not a sufficient
one. It also scores a linear read-out of a mean-pooled frame; the action header cross-attends to all
144 tokens, so it can use spatial structure this probe deliberately discards.

Runs in `fcrl`. Usage:
    P=/home/sarvesh/miniconda3/envs/fcrl/bin/python
    $P scripts/data/vibe/probe_color_grounding.py --rollouts data/vibe_pilot --episodes 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SYSTEM = ("You control a humanoid robot facing a coloured cube. "
          "Output the whole-body motion that satisfies the instruction.")

# Psi0 does NOT train against stock Qwen3-VL -- `--model.model_name_or_path` points at a Qwen3-VL-2B
# that was further trained on EgoDex 200k + HE 30k. Those 230k steps of humanoid-manipulation
# training could erode the general colour grounding this probe measures, and the action expert's
# pretrained `transformer_blocks` cross-attend to THAT model's hidden states, so the two cannot be
# mixed and matched for free. `--model` re-runs the identical probe against whichever backbone will
# actually be frozen underneath training.
VISIBLE_MIN_FRAC = 0.01          # same cut collect_rollouts.py uses for `commanded_visible_rate`

# (name, target HxW or None for native, torchvision interpolation mode name)
CONDITIONS = [
    ("native",     None,       None),
    ("psi0-near",  (240, 320), "NEAREST"),
    ("psi0-bilin", (240, 320), "BILINEAR"),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model", default=MODEL,
                    help="HF id or local checkpoint dir of the VLM to probe. Default is stock "
                         "Qwen3-VL; pass Psi0's pretrained backbone to measure the one that will "
                         "actually be frozen under training.")
    ap.add_argument("--episodes", type=int, default=12, help="episodes to sample frames from")
    ap.add_argument("--frame-stride", type=int, default=6,
                    help="consecutive frames are 20 ms apart and nearly identical; striding buys "
                         "independent samples rather than more of the same")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    return ap.parse_args()


def load_frames(root: Path, n_eps: int, stride: int):
    """-> list of (rgb HxWx3 uint8, commanded_color, visible_bool, episode_index)."""
    out = []
    for d in sorted((root / "episodes").iterdir())[:n_eps]:
        meta = json.loads((d / "meta.json").read_text())
        vis = np.asarray(meta["commanded_visible_frac"], np.float32) > VISIBLE_MIN_FRAC
        frames = np.stack(imageio.mimread(str(d / "frames.mp4"), memtest=False))
        n = min(len(frames), len(vis))
        for t in range(0, n, stride):
            out.append((frames[t], meta["commanded_color"], bool(vis[t]), int(meta["episode"])))
    return out


def main():
    args = parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from torchvision.transforms import v2
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    root = Path(args.rollouts)
    summary = json.loads((root / "summary.json").read_text())
    samples = load_frames(root, args.episodes, args.frame_stride)
    y = np.array([s[2] for s in samples], np.int32)
    eps = np.array([s[3] for s in samples])
    colors = np.array([s[1] for s in samples])
    print(f"[probe] {len(samples)} frames from {len(np.unique(eps))} episodes of {root.name} "
          f"({summary['cam_width']}x{summary['cam_height']}, {summary['video_pix_fmt']})")
    print(f"[probe] commanded colour visible in {y.mean() * 100:.1f}% of them "
          f"(a base rate near 0 or 100 makes AUC unstable)")

    print(f"[probe] backbone {args.model}"
          f"{'  (STOCK Qwen -- not what training freezes)' if args.model == MODEL else ''}")
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # Held out by episode: consecutive frames of one episode are near-duplicates, so a random frame
    # split would put a frame's own neighbours in train and report memorization as accuracy.
    uniq = np.unique(eps)
    test_eps = set(uniq[::3].tolist())
    te = np.array([e in test_eps for e in eps])
    print(f"[probe] split: {(~te).sum()} train / {te.sum()} test frames, "
          f"{len(uniq) - len(test_eps)}/{len(test_eps)} episodes\n")

    for name, size, interp in CONDITIONS:
        tf = (v2.Resize(size, interpolation=getattr(v2.InterpolationMode, interp))
              if size else None)
        feats = []
        for i in range(0, len(samples), args.batch):
            chunk = samples[i:i + args.batch]
            imgs, texts = [], []
            for rgb, color, _, _ in chunk:
                if tf is not None:
                    t = torch.from_numpy(rgb).permute(2, 0, 1)
                    rgb = tf(t).permute(1, 2, 0).numpy()
                imgs.append(rgb)
                texts.append(proc.apply_chat_template([
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Flip the cube so the {color} face is up."}]},
                ], tokenize=False, add_generation_prompt=True))
            b = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
            b = {k: v.to(args.device) for k, v in b.items()}
            with torch.no_grad():
                h = model(**b, output_hidden_states=True).hidden_states[-1]
            mask = (b["input_ids"] == img_tok).unsqueeze(-1)          # image positions only
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats.append(pooled.float().cpu().numpy())
        X = np.concatenate(feats)
        n_tok = int((proc(text=[texts[0]], images=[imgs[0]], return_tensors="pt")["input_ids"][0]
                     == img_tok).sum())

        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        per = []
        for c in sorted(set(colors[te])):
            m = colors[te] == c
            per.append(f"{c} {roc_auc_score(y[te][m], p[m]):.3f}" if len(set(y[te][m])) > 1
                       else f"{c} n/a")
        print(f"  {name:11s} {n_tok:3d} tok   AUC {auc:.4f}   " + "  ".join(per))

    print("\n  references (previous corpus, different encoders, same label definition):"
          "\n    ClearCLIP detector 0.913     standardized-similarity detector 0.934"
          "\n    0.5 = chance; near 0.5 means the vision path carries no colour")


if __name__ == "__main__":
    main()
