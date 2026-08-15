# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this fork is

Upstream `Psi0` (USC PSI-lab, Apache-2.0) is a VLA: a **Qwen3-VL-2B backbone (System-2)** + a
**~500M SD3-style flow-matching action expert (System-1)** + an RL tracking controller (System-0).

This fork retargets the action expert to predict **64-d pre-FSQ SONIC motion latents** for a
Unitree G1 **cube-repose** task simulated in the sibling `vibe` repo: the robot walks to a
coloured cube and flips it so a named colour ends up on top. Psi0 plans, SONIC executes.

**The seam is `SonicBaseModel._encode_mlp`** — vibe's policy normally derives its 64-d latent from
a 640-d tokenizer window; at deploy Psi0's action head produces that latent instead. Everything
in this fork exists to make Psi0 predict exactly that vector.

**Why Psi0 and not the previous model.** The predecessor (`songen`, a 34.66M flow DiT, now
deleted) failed in a specific, diagnosable way: its `vs_zoh` — model error against a zero-order
hold on the current latent — sat at **0.89 through six runs**, and `drop_lang` (the val-MAE
penalty for blanking the command) stayed **negative**, i.e. the language input was worse than
useless. The target was nearly determined by the present, so there was no gradient pressure to
read a command. Psi0 buys a longer horizon and a much stronger language encoder. **Those two
metrics are the point of this project and are now implemented here** (see Diagnostics).

Upstream: <https://github.com/physical-superintelligence-lab/Psi0>. `origin` is this fork;
`upstream` is theirs. Modifications are enumerated in `NOTICE` (Apache §4b).

## Two environments, one interface

| env | what | used by |
|---|---|---|
| `fcrl` conda (`/home/sarvesh/miniconda3/envs/fcrl/bin/python`) | mjlab / orcs / vibe / mocke | corpus steps 1–2, both probes |
| `.venv-psi` (uv, py3.10) | psi, torch 2.7.0+cu126, torchcodec, flash_attn | step 3, training |

Nothing in steps 1–2 imports `psi`; step 3 imports no sim. **The npz corpus on disk is the
boundary.** Never try to merge the two envs.

`.venv-psi` build trap: `uv sync --active` reads `--active` from `$VIRTUAL_ENV`, so running it
without sourcing `.venv-psi/bin/activate` first silently installs 6.9 GB into the project default
`.venv`. Use `UV_PROJECT_ENVIRONMENT="$PWD/.venv-psi"` instead.

## The corpus pipeline

Lives in `scripts/data/vibe/` — **read its README before touching any of it**; it holds every
measurement behind the design choices and this file does not repeat them.

```
1. collect_rollouts.py    [fcrl]      vibe sim -> traj.npz + frames.mp4 + meta.json per episode
2. extract_latents.py     [fcrl]      frozen SONIC encoder -> latents.npz (64-d pre-FSQ)
3. raw_vibe_to_psi_lerobot.py [.venv-psi]  -> LeRobot dataset + a stratified val split
```

**Current state on this machine:**

| | |
|---|---|
| `data/vibe_g1` | 4,032 episodes / 1,030,414 frames / 6.9 GB, 94.7% success |
| `data/lerobot/vibe_repose_g1` | 3,246 episodes / 780,660 frames (train) |
| `data/lerobot/vibe_repose_g1_val` | 573 episodes / 140,618 frames (15%, held out by episode) |
| checkpoints | `$PSI_HOME/cache/checkpoints/psi0/` — VLM 4.0 GB + action header 1.9 GB |

`data/` is gitignored. Datasets belong on the HF Hub (`--push`), never in git.

## Column semantics

| column | content | flag |
|---|---|---|
| `action` | 64-d **pre-FSQ** SONIC latent | `--model.action-dim=64` |
| `states` | joint_pos(29) + projected gravity(3) | `--model.odim=32` |
| `task` | one of 6 sentences, exactly ONE token varies | — |
| video | `sys1_cam` 512×288 h264 **yuv444p**, 50 fps | — |

**`joint_vel` is deliberately excluded from `states`.** It is available and it *is* informative
— which is the problem. More present proprioception widens exactly the shortcut that killed
songen. Do not "improve" the state vector by adding it without running the `drop_lang` control.

**The command vocabulary is `"Flip the cube so the {colour} face is up."`** Verified: across the
six colours the tokenized prompts differ at **exactly one position (index 90)**, and every colour
is a single token. That is what makes the `drop_lang` ablation a clean intervention.

## Training

```bash
./scripts/train/psi0/finetune-vibe-repose-psi0.sh <exp>      # BATCH=, SCRATCH=, DRYRUN= env vars
```

`DRYRUN=1` prints the resolved flag list without launching — use it to inspect config changes.

### Four config traps (all already handled in that script; do not undo them)

1. **Never pass `--data.transform.model.img-aug`.** `ColorJitter` defaults to `hue=0.05` = ±18°.
   Measured on this corpus's palette, red and orange are **24.7° apart** and yellow's nearest
   neighbour is 35.3°, so one draw covers 51–73% of the distance between two labels. On a task
   where the commanded colour IS the label that is label noise, not augmentation. There is also
   no domain gap to close — we roll out in vibe and deploy in vibe.
2. **`repack.action_chunk_size` is NOT synced to `model.action_chunk_size`.** Both default to 30,
   which is why upstream configs agree. Set BOTH or the dataloader silently emits 30-step chunks.
3. **`val_repo_ids` defaults to `[train_repo_ids[0]]`** — i.e. it validates on the training set.
   Pass `--data.val_repo_ids=vibe_repose_g1_val` or every number below is in-sample.
4. **`--model.action-chunk-size=50`, not 30.** Our fps is 50; upstream's SONIC pipeline is 30
   (`raw_sonic_to_psi_lerobot.py:20` hardcodes `FPS = 30`), so their chunk of 30 is 1.0 s and
   ours must be 50 to match that duration. `--model.action-exec-horizon` must follow.

### What the checkpoint load actually does

`sonic.py` loads the pretrained action header by **shape compatibility** (this fork's change;
upstream filtered by a `transformer_blocks` name prefix). Measured on the real module:
**497.5M of 497.7M params load (99.93%, 174/181 tensors)**. Only 7 reinitialize —
`action_proj_in.{ac_proj,dec_pos}`, `action_proj_out.linear`, `obs_proj._obs_proc.1.weight`.

No SONIC-embodiment header was ever released: **all 30 published Psi0 checkpoints are
`action-dim=36, chunk=30, odim=36`**, all from `postpre.1by1.pad36...he30k`, and every one of them
also only loads `transformer_blocks` (their chunk 30 ≠ the checkpoint's 16). This fork's partial
load is the normal path, not a degraded one.

## Diagnostics — the reason this project can be evaluated at all

`evaluate()` reports three numbers beyond val loss, all computed on **unpadded steps only**.
`PSI_EVAL_ABLATIONS=0` disables them (they cost ~2 extra denoising passes per val batch).

| metric | meaning | read |
|---|---|---|
| `vs_zoh` | model L1 ÷ persistence L1 (repeat the current latent across the chunk) | <1 beats doing nothing. songen sat at 0.89 |
| `drop_lang` | L1 penalty when commands are permuted across the batch | **POSITIVE = the model reads the command.** songen's was negative |
| `drop_vision` | same, for the image | — |

The ablations **permute** an input across the batch rather than blanking it, so the marginal
distribution the model sees is exactly unchanged and any change in error is attributable to the
destroyed correspondence. `_deranged` uses a random rotation — a derangement by construction.

**Val MAE alone cannot distinguish a model that solved this task from one that ignores the
command.** A wrong-but-consistent target trains to a respectable MAE; that is how songen ran six
times before the failure was visible. Do not report progress on val loss alone.

### The padding decision

The last 49 rows of every episode have chunks that overrun the end; LeRobot clamps the index, so
those steps are verbatim copies of the final latent — **10.2% of all steps**, always at episode
end. They are KEPT in the training loss (measured near-truth: ‖z[t] − z[final]‖ averages 0.147
over the padded span vs 1.41 for an equal mid-episode span, because `--tail-steps 10` and the
clamped tokenizer window make the ending a genuine static hold) and **EXCLUDED from every eval
metric** via a new `action_is_pad` key. Note that key travels repack → field → model transform →
collator, and the collator copies an **explicit allowlist** — a key missing from it is dropped
silently, which for a mask degrades to "no masking" with no error.

## Resource facts

| | |
|---|---|
| training VRAM | **~13 GB** — 4.26 (VLM bf16, frozen) + 1.99 (head fp32) + 1.99 (grads) + 3.98 (AdamW) |
| inference VRAM | **6.25 GB** (measured) |
| batch scaling | ~0.04 GB/sample — dominated by static state, so push batch on a big card |
| 8 GB cards | cannot train. `scripts/deepspeed/zero3_offload.json` is the only local path |

The frozen VLM stores **no activations**: its params are `requires_grad=False` and its inputs are
non-differentiable, so `hidden_states[-1]` returns without a graph. That is why this is affordable.

**`inference()` re-runs the whole 2.1B VLM at every denoising step** even though `views` is
constant across the loop. Eval is therefore ~N× more expensive than necessary. Hoisting `views`
out would need a signature change to `Psi0Model.forward`; not done.

## Verification status

Verified locally: config parsing; dataset build; collation; forward; loss; **gradient coverage
(0 params with `.grad is None`, so DDP will not raise)**; the shape-filtered checkpoint load;
train/val split disjointness; VRAM. The 2 zero-grad tensors are `transformer_blocks.5.attn.add_q_proj`
— the last block sets `context_pre_only=True` and discards its context stream. Benign; zero-grad
is fine for DDP, only `None`-grad breaks it.

**Never verified:** a complete `optimizer.step()`; `evaluate()` against the real model; multi-GPU
collectives. All three need a card bigger than 8 GB. **Submit a 20-step job before a long one.**

## Gotchas

- The SONIC encoder is **SiLU**. `extract_latents.py` reads the activation off
  `SonicBaseModel.__init__`'s default rather than naming one. A hardcoded `nn.ELU` there once
  produced latents that were a different function of the same input — correct shapes, happy
  `load_state_dict`, three training runs void, nothing offline flagged it.
- The latent target is **pre-FSQ**. FSQ `tanh`-bounds before rounding, so a wrong-scale latent is
  silently squashed rather than rejected. Put a magnitude check at the deploy seam.
- **Qwen3-VL's effective patch is 32 px** (`patch_size=16 × merge_size=2`), not the 28 of
  Qwen2-VL that upstream's `max_pixels: 576*28*28` still encodes. At the `240 320` preset the
  grid is `[1, 16, 20]` → exactly **80 image tokens**, 100 total with the prompt.
- `--data.transform.model.resize.size 240 320` with an equal `center_crop` **squashes**, it does
  not crop: `v2.Resize` with a 2-tuple scales axes independently. Full 69.3° H-FOV is retained.
- `train.py` runs `git add . && git commit && git tag <run>` on rank zero at startup.
- `WANDB_API_KEY` in `.env` must be set — the launch script passes `--log.report_to=wandb`.
- The vibe workspace (`../vibe`, `../songen`) is **not a git repo**. `collect_rollouts.py` and
  `extract_latents.py` cannot run without it, so this repo alone is not reproducible.
