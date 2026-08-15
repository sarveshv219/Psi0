# vibe -> Psi0 corpus

Rolls out the visually-adapted SONIC policy in the `vibe` sim on the BigCube repose task and packs
what the robot did into a corpus Psi0 can fine-tune on. Psi0 predicts the **64-d pre-FSQ SONIC
latent**, which `vibe`'s policy consumes in place of its own tokenizer encoder — so Psi0 plans and
SONIC executes.

Task: a G1 walks to a cube and flips it so a named colour ends up on top.

## Two environments, one interface

Collection needs `mjlab` / `orcs` / `vibe` / `mocke`, which live in the **`fcrl` conda env**. Psi0
trains in **`.venv-psi`**. Steps 1–2 import no `psi`; step 3 imports no sim — the npz corpus on disk
is the boundary.

```bash
P=/home/sarvesh/miniconda3/envs/fcrl/bin/python     # steps 1-2
source .venv-psi/bin/activate                       # step 3 and training
```

`.venv-psi` is built by the repo README's recipe. One trap: `uv sync --active` reads `--active`
from `$VIRTUAL_ENV`, so running it without sourcing `.venv-psi/bin/activate` first silently
installs 6.9 GB into the project default `.venv` instead. Either activate first, or skip `--active`
and pass `UV_PROJECT_ENVIRONMENT="$PWD/.venv-psi"`.

## Pipeline

```bash
# 1. roll out: executed trajectories + sys1-camera video + per-episode labels     [fcrl]
#    RAM-bound, not GPU-bound: several --steps auto passes with --ep-offset, not one run.
$P scripts/data/vibe/collect_rollouts.py --num-envs 160 --steps auto --max-buffer-gb 20 \
     --ep-offset $(ls data/vibe_g1/episodes | wc -l) \
     --out data/vibe_g1 --wandb-run-path vbp/repose/011pgzbh --verify-video 1

# 2. frozen SONIC encoder -> one 64-d pre-FSQ latent per frame                    [fcrl]
$P scripts/data/vibe/extract_latents.py --rollouts data/vibe_g1 --check
$P scripts/data/vibe/extract_latents.py --rollouts data/vibe_g1

# 3. pack to LeRobot at fps=50                                                    [.venv-psi]
python scripts/data/vibe/raw_vibe_to_psi_lerobot.py \
     --rollouts data/vibe_g1 --out data/lerobot/vibe_repose_g1
```

Two probes sit between 2 and 3. They are diagnostics, not pipeline stages:

```bash
$P scripts/data/vibe/probe_qwen_tokens.py    --rollouts data/vibe_g1
$P scripts/data/vibe/probe_color_grounding.py --rollouts data/vibe_g1 --episodes 40
```

## Corpus

**Raw** — `data/vibe_g1`, 4,032 episodes / 1,030,414 frames / 5.0 GB, `sys1_cam` at 512×288 h264
yuv444p, 50 fps. Success (commanded colour ends up on top) **94.7%**; colours 647–692 episodes each,
a 1.07× spread. Mean episode 255.6 frames, range 95–992.

**Packed** — `data/lerobot/vibe_repose_g1`, 3,819 episodes / 921,278 frames / 3.2 GB (2.7 GB video,
524 MB parquet) = **5.12 robot-hours**. 213 episodes dropped as failures. 6 tasks, 616–661 episodes
each. `action` 64-d in [−1.29, 0.95], per-dim std 0.113–0.269, all finite; no dim is degenerate
(narrowest q01..q99 width 0.50), so `ActionStateTransform`'s near-zero-range guard never fires.
Verified to load through `psi.data.lerobot.compat.LeRobotDataset` with
`SimpleRepackTransform(action_chunk_size=50).delta_timestamps(50)`, yielding
`action (50, 64)`, `states (1, 32)`, one `(3, 288, 512)` image, and the task string resolved.

## What the pilot established

**Qwen3-VL can read the commanded colour — this was the go/no-go and it passed.**

Data: frames strided 6 apart from the first 40 episodes of `data/vibe_g1`, split by episode
(`uniq[::3]` held out) so no frame's own neighbours are in training. Computed: one VLM forward per
frame over (system prompt, image, command sentence); `hidden_states[-1]` at the `<|image_pad|>`
positions only, mean-pooled to 2048 dims. Metric: ROC AUC of a logistic regression predicting "the
commanded colour is visible this frame", ground truth = the renderer's own segmentation channel at
`commanded_visible_frac > 0.01`.

| condition | image tokens | AUC | weakest colour |
|---|---|---|---|
| native 512×288, no resize | 144 | 0.9903 | yellow 0.960 |
| Psi0 preset 320×240 NEAREST | 80 | 0.9759 | yellow 0.908 |
| Psi0 preset 320×240 BILINEAR | 80 | 0.9790 | orange 0.970 |

References measured on the previous corpus with different encoders — same label definition, so
comparable in kind but not a controlled comparison: ClearCLIP detector **0.913**, standardized-
similarity detector **0.934**. 0.5 is chance.

**What this does not establish.** That any of these produces better generated motion. It scores a
linear read-out of a *mean-pooled* frame, discarding the spatial structure the action header
cross-attends over. It is a necessary condition — the colour must be legible to the frozen backbone
at all — not a sufficient one.

### Why the corpus renders at 512×288 but trains at 320×240

All three conditions are near ceiling, so the 1.4-point native advantage is not a decision; it says
the same thing for all three, which is that **colour legibility is not this task's bottleneck**.
Training therefore stays on Psi0's shipped `240 320` preset: 1.8× fewer view tokens, ~100 GB of VLM
cache instead of ~155 GB at stride 5, and the pretrained `transformer_blocks` stay on the view-token
count they were trained against.

The render stays at 512×288 anyway. It costs 5.0 GB for the whole corpus, and re-caching at a
different resolution is then a re-run of one script rather than a re-collection. That asymmetry is
the whole argument — start cheap on the reversible side.

Note the preset **squashes** rather than crops: `--resize.size 240 320` with
`--center_crop.size 240 320` makes the crop a no-op, and `v2.Resize` with a 2-tuple scales the axes
independently. A 16:9 frame becomes 4:3 anisotropically, keeping the full 69.3° H-FOV. No part of
the frame is discarded.

### The one thing the probe changed in Psi0

`ResizeImage` hardcoded `InterpolationMode.NEAREST`. Yellow scored **0.908 under NEAREST and 0.965
under BILINEAR at the same 80 tokens** — point-sampling a downscale drops thin and oblique coloured
faces outright, and on this task the face colour *is* the label. Changed to BILINEAR in
`src/psi/config/augmentation.py`. Bilinear costs a little on orange (1.000 → 0.970) and pink
(1.000 → 0.986), so per-colour it is close to a wash in aggregate; yellow is the only column that
moves further than run-to-run spread.

### Cache cost

Measured, not estimated: one real forward on a real frame. At 512×288 the batch is 189 tokens
(144 image + 45 text/template) × 2048 dims × 2 B = **774 KB/frame**. At the 320×240 preset it is
125 tokens = **512 KB/frame**, so a stride-5 cache over the 976k kept frames is ~100 GB.

Cache the whole `hidden_states[-1]`, keyed by **(frame, command)** — not "vision patches".
`Psi0Model.forward` calls the VLM once over text and image together and hands the entire last
hidden state to the action header as `views`, so the two are not separable. One command per episode
is what collapses that back to one forward per frame.

Stride belongs in the cache, never in the rows: `lerobot_patch` sets `tolerance_s=1e-4` and rejects
gaps, so the parquet stays dense at 50 Hz. Give each episode a **random stride offset** so cached
frames do not phase-align with chunk starts.

## Design notes

**Camera.** `sys1_cam` at 512×288 — 16:9 matching the head cam's mount and fovy, and exactly 16×9
patches at Qwen3-VL's **32-px effective patch** (`patch_size=16 × merge_size=2`). Not 28: that is
the Qwen2-VL convention, and it is still what Psi0's `max_pixels: 576*28*28` encodes. An earlier
448×252 was silently rescaled to 448×256 because of it. `collect_rollouts.py` prints the patch grid
and asserts nothing resized.

**Segmentation rides on sys1_cam, not the head cam.** The visibility labels are what the grounding
probe scores against, so they are measured on the frame the model is given. The head cam keeps rgb
only and is otherwise untouched — it is a live *input* to the policy being rolled out.

**Frames go to h264.** At 512×288 a raw frame is 442 KB, so an npz corpus would be ~450 GB. The
head cam is still npz (21 KB/frame) and byte-exact.

**Codec loss, measured on sys1_cam.** `--verify-video 3` on two 200-episode pilot runs identical
except for `--pix-fmt`. Data: the first 3 episodes of each run (112–119 frames each). Computed: raw
in-memory frames vs `imageio.mimread` of the written mp4, per-pixel per-channel `|decoded − source|`
in uint8 levels. Both at `--crf 10`.

| pix_fmt | mean\|err\| | p99 | max | KB/frame |
|---|---|---|---|---|
| yuv420p | 1.67 / 1.73 / 1.70 | 8 / 10 / 15 | 207 / 251 / 198 | 2.5–3.5 |
| **yuv444p** | **0.37 / 0.58 / 0.40** | **4 / 5 / 4** | **82 / 91 / 93** | **2.8–3.3** |

3.8× lower mean error and 2.4× lower max **at the same file size**, which is why this is not a
tradeoff. The two runs drew different episodes, so it is not paired; it holds because the
between-condition gap far exceeds the spread within either condition.

**`crf` does not fix 4:2:0.** Measured separately on raw head-cam frames (the only stream saved
uncompressed, so the only one diffable after the fact): yuv420p mean 2.89 at crf 10 and 2.64 at
crf 0, against yuv444p's 1.11 and 0.38. The loss is the 2×2 chroma downsample, which happens
*before* encoding; `crf` controls quantization only. The same test put **0.694%** of pixels more
than 64 levels off under yuv420p against **0.0016%** under yuv444p — 4:2:0 is accurate over flat
regions and wrong at saturated boundaries, which on this scene are the cube's face edges, and the
face colour is the label.

**What that does not establish.** That yuv420p would have hurt the grounding probe — plausibly not,
since the VLM pools whole patches and scattered bad pixels may wash out before reaching a token.
The case for 4:4:4 is that it costs nothing, so the question never has to be answered.

**num-envs is capped by RAM, not the GPU.** Frames buffer per env until the episode ends; at 442
KB/frame a mean 255-frame episode pins ~113 MB per env. `collect_rollouts.py` estimates this at
startup and refuses rather than OOM-ing hours in. Measured at 160 envs: 20.4 s startup, 0.58 s/step,
peak GPU 3,217 MiB.

**Colour-first binding is preserved.** Left alone the env draws a clip and the goal colour falls out
of that clip's final orientation, so the colour is a *consequence* of the motion and cannot predict
it. `bind_color_first` draws the colour first and masks the clip pool to clips that achieve it, and
a per-step assert fails loudly if the env's goal ever disagrees with the colour that was drawn.

**Language is written at step 3, not at collection.** The command is a pure function of `meta.json`,
so editing the phrasing never invalidates a collection or a video cache.

## Training flags this corpus implies

```
--data.root_dir=data/lerobot  --data.train_repo_ids=vibe_repose_g1
--data.transform.field.stat-path=meta/stats_psi0.json
--data.transform.model.resize.size 240 320
--data.transform.model.center_crop.size 240 320
--model.action-dim=64          # latent only; this G1 has no hands
--model.odim=32                # joint_pos(29) + projected gravity(3)
--model.action-chunk-size=50   # 1.0 s at 50 fps, NOT the preset's 30
```

`--model.action-chunk-size=50` is the one that matters. At 50 fps the preset's chunk of 30 covers
0.6 s. songen's failure was a target nearly determined by the present — `vs_zoh` stuck at 0.89 and
`drop_lang` negative through run6 — and a longer horizon is the main thing Psi0 is being brought in
to buy. The flag is free: `psi/trainers/sonic.py` already reinitializes the in/out projections
because `action_dim` differs from 78, so a changed chunk size costs no additional pretrained weight.

## Gotchas

- `traj.npz` is MuJoCo joint order; `extract_latents.py` asserts this.
- The SONIC encoder is **SiLU**, and `extract_latents.py` reads the activation off
  `SonicBaseModel.__init__`'s default rather than naming one. A hardcoded `nn.ELU` here once
  produced latents that were a different function of the same input, with correct shapes and a
  happy `load_state_dict` — three training runs were void and nothing offline flagged it, because
  both sides of every error used the same wrong encoder.
- The latent target is **pre-FSQ**. FSQ `tanh`-bounds before rounding to a 32-level grid, so a
  latent on the wrong scale is silently squashed rather than rejected. Put a magnitude check at the
  deploy seam: generated mean `|z|` against the live encoder's.
- **81.3% of frames have an unclamped tokenizer future window and step 3 keeps the other 18.7%.** The
  last `(FUTURE_STEPS−1)×FRAME_SKIP = 45` frames of each episode have a window clamped to the final
  frame. Those are real — `MultiClipMotionCommand.future_frames` clamps identically at a clip
  boundary — and dropping them would delete the end of every episode, which is exactly where the
  flip completes. Counted per episode as `frames_with_full_future` in `episodes.jsonl`.
- **torchcodec reads yuv444p, and reads it identically to imageio-ffmpeg.** This was the last open
  link: `verify_video` measured codec loss with imageio-ffmpeg, but training decodes with
  torchcodec. Measured on the first 3 episodes of `data/vibe_g1` (336 frames, 512×288×3):
  `torchcodec 0.4.0` vs `imageio.mimread`, per-pixel per-channel `|a − b|` in uint8 levels is
  **exactly 0 — max 0, mean 0.0000, no pixel differs at all**. So the 0.37–0.58 mean error in the
  codec table is what training actually sees, and no transcode is needed. Step 3 still runs the
  probe every time and falls back to a yuv420p transcode if it ever fails, recording which happened
  in `info.json`.
- `imageio-ffmpeg` defaults `macro_block_size=16` and **silently resizes** frames to a multiple of
  16. `write_video` pins it to 1; do not remove that.
- `--wandb-run-path vbp/repose/011pgzbh` is the policy whose rollouts become the corpus. A wrong one
  silently produces a corpus of different behaviour; `--wandb-run-path ""` falls back to the
  untrained base agent.
