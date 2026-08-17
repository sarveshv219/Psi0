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
| `.venv-psi` (uv, py3.10) | psi, torch 2.7.0+**cu128**, torchcodec, flash_attn | step 3, training |

Nothing in steps 1–2 imports `psi`; step 3 imports no sim. **The npz corpus on disk is the
boundary.** Never try to merge the two envs.

`.venv-psi` build trap: `uv sync --active` reads `--active` from `$VIRTUAL_ENV`, so running it
without sourcing `.venv-psi/bin/activate` first silently installs 6.9 GB into the project default
`.venv`. Use `UV_PROJECT_ENVIRONMENT="$PWD/.venv-psi"` instead.

### Bringing up a new machine

Four traps, all hit for real on 2026-08-15 while standing this up from scratch. None are in either
README except the first, and each one costs a run.

**1. `uv pip install torch==2.7.0` is a NO-OP against a different CUDA variant.** uv matches on the
version string, so a `+cu126` build already present satisfies a request that resolves to `+cu128`,
and only the *other* packages get swapped. Force it:

```bash
uv pip install --reinstall-package torch --reinstall-package torchvision \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

cu128 is required on **Blackwell (sm_120: RTX 5090, RTX 6000 Pro)** — README troubleshooting #5.
The symptom is a startup warning listing `sm_50 … sm_90` followed by `CUDA error: no kernel image
is available for execution on the device` at the first `.to(device)`, which in `sonic.py` is the
`loss_w` tensor, long before any real work. Rebuild `flash_attn` afterwards.

**2. torchcodec needs FFmpeg *shared libraries* on the loader path.** It `dlopen`s
`libavutil.so.5{6,7,8,9}` by soname (FFmpeg 4–7; Ubuntu 24.04's 6.1.1 gives so.58). The failure
mode is nasty: not an import error at startup but a **DataLoader worker crash on the first batch**,
after the 4 GB VLM has already loaded — `RuntimeError: Could not load libtorchcodec`. Notes:

- `pip`/`uv` cannot supply these. `imageio-ffmpeg` ships a static *binary*, no `.so`; PyAV bundles
  hash-mangled sonames (`libavutil-b5680d75.so.60`) that `dlopen` will never match, and so.60 is
  FFmpeg 8 — outside torchcodec's supported range anyway.
- With sudo: `apt-get install ffmpeg`. Without: `conda create -y -n ffmpeg6 -c conda-forge
  'ffmpeg=6.1'` and put its `lib/` on `LD_LIBRARY_PATH`. Borrowing C shared objects from a conda
  env does **not** cross the two-environment boundary — no conda Python is imported.
- If you go the `LD_LIBRARY_PATH` route it must be exported **in the shell, before the process
  starts**. Putting it in `.env` does nothing: glibc caches the search path at startup, and
  `load_dotenv()` runs long after that.
- Do **not** work around this by switching LeRobot's `video_backend` to pyav. torchcodec was
  measured byte-identical to imageio-ffmpeg on this corpus (max diff 0); pyav was not.

**3. The HF dataset pull gets rate-limited, and lies about it.** The packed dataset is **7,649
files**, and Xet requests a token *per file*, which blows the 1000-requests-per-5-minutes quota and
429s. Worse, a rate-limited `snapshot_download` **exits 0** and silently returns the partial local
dir. Never trust the exit code — count what landed:

```bash
HF_HUB_DISABLE_XET=1 hf download sarveshv219/vibe-repose-sim --repo-type=dataset \
    --local-dir "$HF_LEROBOT_HOME" --max-workers 4
find "$HF_LEROBOT_HOME"/vibe_repose_g1     -name '*.parquet' | wc -l   # expect 3246
find "$HF_LEROBOT_HOME"/vibe_repose_g1_val -name '*.parquet' | wc -l   # expect 573
```

**4. `pypi.nvidia.com` times out** pulling torch's ~3 GB of bundled CUDA libs (cudnn alone is
693 MB). `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2` fixes it; uv banks successful wheels, so
repeating the same command makes progress rather than restarting.

**5. On old glibc, `rerun-sdk` blocks the whole sync — and it is unreachable code.** RHEL-8-class
clusters (CARC) are `manylinux_2_28`; Ubuntu 24.04 is 2.39, so this only appears off the dev box:

```
error: Distribution `rerun-sdk==0.22.1` can't be installed because it doesn't have a
source distribution or wheel for the current platform
```

There is **no version that satisfies both constraints**: `lerobot` caps it at `<0.23.0,>=0.21.0`,
and rerun-sdk published no `manylinux_2_28` wheel until **0.31.3**. Don't try to upgrade past the
cap. Skip it instead — `uv sync --no-install-package rerun-sdk`, already in `mode_setup`.

Safe because nothing on the training path loads it: importing `LeRobotDataset` leaves `rerun` out
of `sys.modules`, the only lerobot modules that import it are `visualize_dataset`, `teleoperate`,
`record`, and `visualization_utils`, and `psi` imports exactly one lerobot module
(`datasets.lerobot_dataset`). It would break `lerobot`'s own visualization CLIs, which we never run.

**6. flash-attn installs a wheel built for a NEWER glibc, and no resolver catches it.** Its
`setup.py` prefers a prebuilt wheel from GitHub releases over compiling, and those are linked
against ~glibc 2.35. Install succeeds; the failure is at import, inside model init:

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
             (required by .../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

Proof it was never compiled on the cluster: you cannot produce a `GLIBC_2.32` requirement by
building on glibc 2.28. The wheel tag `cp310-cp310-linux_x86_64` looks locally-built but isn't.

**flash-attn is optional in this fork.** `sonic.py`'s `_attn_implementation()` tries a real import
and returns `sdpa` when it fails, and `mode_setup` **uninstalls** a present-but-unimportable
flash_attn. That uninstall is load-bearing: `transformers.is_flash_attn_2_available()` tests
package *metadata*, not importability, so a broken-but-installed flash_attn makes it return `True`
and every guard built on it — including `psi0.py:1533`'s — fails open.

The cost is ~nothing. The VLM is frozen at **100 tokens** (80 image + prompt), far below where
flash-attn's tiling pays, and it is the only module that requests a backend; the action expert
uses diffusers' attention processors. `PSI0_SKIP_FLASH_ATTN=1` skips the attempt entirely.

**Never install flash-attn without `--no-deps`.** It declares a bare, unbounded `torch`, so any
install of it may re-resolve the pinned stack — adding `--reinstall` once took CARC from
`2.7.0+cu126` to **`2.13.0+cu130`**, silently, and the next build then failed against a `cuda/12.6`
module with a "detected CUDA version mismatches" error that looks like a module problem and isn't.
Recover with `uv sync` (declarative against `uv.lock`), not with more `uv pip install`.

To build it for real, do it yourself — **`FLASH_ATTENTION_FORCE_BUILD` does not survive uv's
PEP 517 subprocess**, so `uv pip install` silently downloads the prebuilt wheel no matter what:

```bash
salloc -c 8 --mem 32G -t 2:00:00 && module load gcc cuda      # cuda 12.x, gcc <= 13
curl -sL https://files.pythonhosted.org/packages/source/f/flash-attn/flash_attn-2.7.4.post1.tar.gz | tar xz
cd flash_attn-2.7.4.post1
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4 NVCC_THREADS=2 \
    $VENV/bin/python setup.py bdist_wheel
uv pip install dist/flash_attn-*.whl --no-deps --python $VENV/bin/python
```

The arch variable is **`FLASH_ATTN_CUDA_ARCHS`** (`setup.py:69`), *not* `TORCH_CUDA_ARCH_LIST`,
which this package never reads. Its only branches are `80/90/100/120` — there is no `89`, and none
is needed: cubins are binary-compatible across minor versions within a major arch, so `80` runs on
sm_86 and sm_89. Watch the first lines — `Guessing wheel URL:` means it is still downloading.

**Not a trap:** `warning: transformers==4.57.0 is yanked`. The pin is upstream's, uv installs it
anyway, and it is the version everything here was verified against. Leave it.

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

`data/` is gitignored. Datasets belong on the HF Hub, never in git.

### Fetching the corpus on a new machine

| HF repo (private) | what | need it for |
|---|---|---|
| `sarveshv219/vibe-repose-sim` | the packed LeRobot dataset, ~3.2 GB | **training** — this is all you need |
| `sarveshv219/vibe-repose-sim-raw` | the npz corpus, 6.9 GB | only to re-run step 3 with different packing |

```bash
set -a; source .env; set +a                  # HF_TOKEN, HF_LEROBOT_HOME, HF_HOME
HF_HUB_DISABLE_XET=1 hf download sarveshv219/vibe-repose-sim --repo-type=dataset \
    --local-dir "$HF_LEROBOT_HOME" --max-workers 4
```

`HF_HUB_DISABLE_XET=1` and the worker cap are load-bearing, and the exit code is not trustworthy —
see trap 3 above for why, and for the file counts to verify against.

`HF_LEROBOT_HOME` must point at the download target — LeRobot resolves `repo_id` relative to it,
so the two dataset dirs have to land as `$HF_LEROBOT_HOME/vibe_repose_g1{,_val}`. Both repos are
**private**; `HF_TOKEN` is required to read them and is in `.env` (gitignored, never committed).

Re-uploading after a repack:

```bash
hf upload-large-folder sarveshv219/vibe-repose-sim data/lerobot \
    --repo-type=dataset --private --num-workers 8
```

`upload-large-folder` is resumable — it keeps per-file hashes in `data/lerobot/.cache/upload/`, so
a killed run re-runs with the identical command and skips the hashing pass.

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
./scripts/train/psi0/finetune-vibe-repose-psi0.sh <exp> [OVR...]   # BATCH=, SCRATCH=, DRYRUN=
```

`DRYRUN=1` prints the resolved flag list without launching — use it to inspect config changes.
Trailing `OVR` args are appended *after* the resolved flags and win (tyro is last-wins), which is
how the smoke run overrides step counts without a second script. `SCRATCH=1` swaps the pretrained
action header for a random one — the control arm for whether a header post-trained on
Humanoid-Everyday *joint* actions transfers to a *latent* action space, which nobody has tested.

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

**Throughput, measured once:** 20 steps at batch 16 on an RTX 5090 took **67 s wall including one
validation pass ≈ 3.3 s/step**. Extrapolated, 40k steps is **~37 h**, which does **not** fit
`carc.sh`'s `DEF_HOURS=24`. Treat that as an order-of-magnitude figure from a single short run, not
a benchmark — re-measure on the target card, then pick `-t` (or cut steps, or wire resume) *before*
submitting, rather than discovering it at hour 24.

**`inference()` re-runs the whole 2.1B VLM at every denoising step** even though `views` is
constant across the loop. Eval is therefore ~N× more expensive than necessary. Hoisting `views`
out would need a signature change to `Psi0Model.forward`; not done.

## Verification status

Verified locally: config parsing; dataset build; collation; forward; loss; **gradient coverage
(0 params with `.grad is None`, so DDP will not raise)**; the shape-filtered checkpoint load;
train/val split disjointness; VRAM. The 2 zero-grad tensors are `transformer_blocks.5.attn.add_q_proj`
— the last block sets `context_pre_only=True` and discards its context stream. Benign; zero-grad
is fine for DDP, only `None`-grad breaks it.

**Verified 2026-08-15 by a 20-step smoke run** on an RTX 5090 (32 GB), batch 16, validation at
step 10. This closed the three items that had never executed anywhere:

| previously unverified | result |
|---|---|
| a complete `optimizer.step()` | 20/20 steps, `Happy Ending!` |
| `evaluate()` against the real model | emitted all three ablations |
| checkpointing | `ckpt_20` written — the final save fires at max steps even with `checkpointing_steps=100000` |

Reproduce it with (the launcher appends trailing args after the resolved flags, tyro last-wins):

```bash
./scripts/train/psi0/finetune-vibe-repose-psi0.sh smoke20 \
    --train.max_training_steps=20 --train.validation_steps=10 \
    --train.val_num_batches=2 --train.checkpointing_steps=100000
```

Its numbers are a **pipeline receipt, not a result** — at step 20 the lr is still 2e-6 in warmup
and 7 tensors were reinitialized minutes earlier:

```
eval/val_l1_masked 0.841   eval/vs_zoh 6.62   eval/drop_lang +0.0035   eval/drop_vision +0.0057
```

`vs_zoh` above 1 means worse than persistence — expected at init, and it is the number that decides
this project. `drop_lang` at 0.4% of val L1 is indistinguishable from zero: right sign, no evidence.
That `val_l1_masked` exists at all is the useful signal — it confirms `action_is_pad` survived the
repack → field → transform → collator allowlist.

**Still never verified: multi-GPU collectives.** That is the first thing a multi-GPU run exercises,
and DDP raises only on `None` grads, which local runs already rule out.
**Submit a 20-step job before a long one.**

## Porting to CARC (USC Discovery)

`scripts/train/psi0/carc.sh` is the SLURM launcher — **one run per GPU**, never one run sharded
across GPUs, because the static footprint is ~13 GB and activations are only ~0.04 GB/sample.

```bash
salloc -c 8 --mem 32G -t 2:00:00       # flash-attn COMPILES; do not do it on a login node
bash carc.sh setup                     # once: uv venv + deps + checkpoints
exit
bash carc.sh data                      # login node: pull the corpus, verify the counts
bash carc.sh smoke -g l40s             # 20 steps. DO THIS FIRST on any new cluster
bash carc.sh submit -g l40s -t 48 -- <exp> [OVR...]
bash carc.sh sweep  -g l40s --manifest runs.tsv    # job array, 1 GPU per element
```

The four bring-up traps above are now **implemented in the launcher**, not left to the operator:
`setup_ffmpeg()` runs in both `mode_setup` and `setup_run_env` and **fails the job immediately**
if no `libavutil.so.5{6,7,8,9}` is on the loader path; `mode_data` disables Xet, caps workers, and
verifies parquet counts because the download exits 0 when throttled; `uv sync` carries the timeout
settings; flash-attn gets `MAX_JOBS=4` and warns outside an allocation. Escape hatches:

| var | effect |
|---|---|
| `PSI0_FFMPEG_DIR` | prefix with `lib/libavutil.so.5x`; skips the `module load` hunt entirely |
| `FFMPEG_MODULES` | module names to try, in order (default covers `ffmpeg/4.4`–`6.1.1`) |
| `PSI0_HF_ONLINE=1` | undo the default `HF_HUB_OFFLINE=1` if a run must reach the hub |
| `HF_DATASET` | override the dataset repo `mode_data` pulls |
| `PSI0_CUDA_HOME` | toolkit prefix for deepspeed's import scan; skips the `CUDA_MODULES` hunt |

**A ddp-only run still needs an `nvcc` on the node, because of deepspeed.** `accelerate`'s `extract_model_from_parallel` does `from deepspeed import
DeepSpeedEngine` whenever the package is merely *installed* (`is_deepspeed_available()` is a
`find_spec` + metadata check, `imports.py:163`) — it only wants the class for an `isinstance`
tuple. Importing deepspeed 0.17.1 runs an op-compatibility scan at module scope
(`git_version_info.py:29`), and `fp_quantizer.is_compatible()` calls `installed_cuda_version()`
**without catching** the `MissingCUDAException` it raises when `torch.utils.cpp_extension.CUDA_HOME`
is `None`. So the model loads, training steps run, and the job dies at the **first `evaluate()`**:

```
deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist,
    unable to compile CUDA op(s)
```

It never fires on the dev box, which has `/usr/bin/nvcc`. `setup_cuda_home()` handles it: it needs
only `$CUDA_HOME/bin/nvcc -V` to answer, so it exports **`CUDA_HOME` and nothing else**, reading
the prefix out of a **subshell** `module load` so the toolkit's `lib64` never reaches this shell's
`LD_LIBRARY_PATH` — which is searched *before* the RUNPATH torch uses for its bundled CUDA, i.e.
loading the module for real is exactly the shadowing this file warns against below. It also sets
`DS_SKIP_CUDA_CHECK=1`, since only the CUDA *major* has to match torch's and nothing is compiled.

The preflight now imports deepspeed too, so a node without `nvcc` fails in 10 s at startup rather
than after the 4 GB model load. If no toolkit exists at all, `uv pip uninstall deepspeed` is safe
here — `data_parallel=ddp` never constructs a `DeepSpeedEngine`; it only forecloses
`zero3_offload.json`.

### Long runs self-smoke in the first two minutes

**`evaluate()` runs at `global_step == 0`** — `train.py:260` ORs that in ahead of the
`validation_steps` modulus, so the eval path is exercised on the first optimizer step no matter
what cadence you pass. Everything that only `evaluate()` touches — `unwrap_model`, the val
dataloader against `vibe_repose_g1_val`, the three ablations, the `action_is_pad` masking — either
works within ~2 min of the job starting or the job is already dead.

That is why submitting the 48 h sweep directly is reasonable when the queue wait is hours: the
first `eval/*` row in wandb **is** the smoke test, and it arrives before the queue wait would have
paid for a separate one. `carc.sh smoke` still buys one thing the full run cannot — it reaches
`save_checkpoint()` (at its step 20) in a 1 h allocation instead of at step 5000.

**Job arrays share a node, so torchrun must not pick a fixed port.** `--gres=gpu:l40s:1` leaves
the other cards on a 2–4 GPU node free, and SLURM packs further array elements onto them. Every
element is a separate job that runs `run_one 0`, so a per-GPU port offset deconflicts nothing —
they all compute 29500, and all but the first die at rendezvous:

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local
network address. port: 29500, ... EADDRINUSE
```

`run_one` therefore passes **`--standalone`** (rdzv-endpoint `localhost:0`, kernel-assigned
ephemeral port), not `--master_port`. The elastic agent still exports `MASTER_ADDR`/`MASTER_PORT`
to the worker from the c10d bootstrap store, which is the only thing accelerate reads. Do not
"fix" this by hashing `SLURM_JOB_ID` into a port — that trades a certain collision for a rare one
and still loses to a stale process holding the port from a previous job on the same node.

`setup_run_env` also runs one real CUDA kernel before training starts. An arch mismatch otherwise
surfaces as `no kernel image is available` on `sonic.py`'s `loss_w` tensor, minutes in and with the
traceback buried inside a `ChildFailedError`.

### Modules

Determined from what the venv actually links against; resolve exact names with `module spider`,
and `module purge` first so nothing is inherited from the login shell.

| when | module | why |
|---|---|---|
| **every job** | `ffmpeg` | torchcodec, see trap 2 above. Loaded *inside* the allocation by `setup_run_env()`; `module purge` runs first so a login shell's modules are not inherited |
| setup only | CUDA toolkit (`nvcc`) + `gcc` | only if you force a flash-attn source build — see trap 6. The default path needs neither |
| setup only | `git` / `git-lfs` | the clone (`GIT_LFS_SKIP_SMUDGE=1` is already in `mode_setup`) |

**Do not load:** a `python` module — uv downloads its own standalone CPython 3.10 and a system one
only confuses the resolution. **Do not load** a CUDA *runtime*, `cudnn`, or `nccl` module either:
torch bundles all of them under `site-packages/nvidia/`, and a version-mismatched module is
actively harmful. Only the GPU node's driver matters, and that is always present.

If Discovery has no `ffmpeg` module, the conda-forge fallback from trap 2 works there too (no sudo
needed on a cluster either), with `LD_LIBRARY_PATH` exported in `setup_run_env()`.

### `.env` is machine-specific

The local `.env` hardcodes `/home/sarvesh/gyms/Psi0/...` into `HF_LEROBOT_HOME` and `DATA_HOME`;
**never copy it to CARC.** `mode_setup` writes a fresh one with cluster paths when none exists —
you only fill in `HF_TOKEN` and `WANDB_API_KEY`. Note `train.py:4` asserts `load_dotenv()` is
truthy, so an absent *or empty* `.env` is a hard failure. `carc.sh` exports `PSI_HOME` and `HF_HOME`
itself before Python starts and `load_dotenv()` does not override already-set vars, so those two
survive — but the rest would silently point at paths that do not exist.

Also note `WANDB_PROJECT`: `WandbConfig.project` is a **CLI flag that defaults to the literal
`"psi"`**, and reads no env var (only `WANDB_ENTITY` is read from the environment, in
`config.py:29`). Exporting `WANDB_PROJECT` alone sends every run to a project called `psi`. Both
launchers now pass `--wandb.project` explicitly; keep it that way.

**`WANDB_ENTITY` is the mirror-image trap, and `.env` loses it by default.** `train.py:4` calls
`load_dotenv()` with the default `override=False`, whose rule is literally `if k in os.environ and
not self.override: continue`. So **any value a launcher exports silently beats `.env`** — and
"any value" includes the empty string, because the test is membership, not truthiness. `carc.sh`
used to default it to a lab org and export it, so every CARC run landed there while the 5090
launcher, which never touches the variable, correctly honoured `.env`. `carc.sh` now sets no
default and exports it only when non-empty, and prints the resolved value at startup:

```
[wandb] project=psi0-vibe-repose entity=sarveshv219 (from .env)
```

Resolution order is CLI `--wandb.entity` → `WANDB_ENTITY` in the environment → `.env` → wandb's
default org for the API key. A blank `WANDB_ENTITY=` line means that last one; `config.py:29`
folds `""` to `None` so blank and absent behave identically.

**A `.env` written by an older `mode_setup` still has the baked-in org in it** — the code change
cannot rewrite a file that already exists. Check it on any machine set up before 2026-08-16.

## Closed-loop deploy: Psi0 plans, SONIC executes

**The seam is `SonicBaseModel._encode_mlp`**, unchanged from songen's deploy path — a 64-d pre-FSQ
latent replaces the tokenizer encoder's output, leaving FSQ, the decoder, the visual adapter and
`latent_residual` running as trained.

**Two processes, on purpose.** Psi0 needs ~6.25 GB to infer and mjlab plus the SONIC policy needs
the rest of an 8 GB card; the two environments also have no compatible dependency set. So:

| side | env | file |
|---|---|---|
| planner | `.venv-psi` | `src/psi/deploy/psi0_serve_real_sonic.py`, `scripts/deploy/serve_psi0-rtc-sonic.sh` |
| sim | `fcrl` | `../vibe/scripts/play_psi0.py` + `psi0_planner.py` |

The client imports no `psi`; the server imports no mjlab. **Every transform stays on the server** —
resize, centre crop, state normalization, action denormalization all come from the checkpoint's own
`run_config.json`. The client sends a raw 512×288 uint8 frame and a raw 32-d state and receives
latents already in pre-FSQ scale. The only thing duplicated is the ~20-line numpy-over-JSON wire
codec, because importing `psi.deploy.helpers` would drag fastapi across the boundary that exists to
keep them apart; it is verified byte-exact against the real `helpers.py` in both directions.

**`pad_to_len(x, None)` raises, and the server used to hit it.** `psi0_serve_real_sonic.py` called
it unconditionally while `pad_state_dim` defaults to `None` — which is every SONIC-latent run,
whose 32-d states already match `odim`. `current_len >= None` is a `TypeError`, and it lands in
`predict_action`'s own `except`, so the server answered **200 with a status string and no action**.
Now guarded the way `ActionStateTransform.__call__` already guards it (`transform.py:59`).

**Four arms, and a number from one alone is unreadable**: `psi0` (the cadence), `psi0-hold` (index
0 held, isolating the chunk tail), `encoder` (checkpoint untouched — the same-harness ceiling), and
`mean` (constant training-mean latent — the floor). songen's harness measured this same low-level
policy landing an *intended* flip 6/55 times on a real recorded slice; most failure in any arm is
execution, not planning.

**It is not a held-out measurement.** Psi0's val split is 15% of collected *episodes*; the env's
motion library is a different axis and essentially every clip contributes to both splits. Unlike
`play_songen.py --samples val`, there is no clip restriction that produces initial conditions Psi0
never saw. The harness reports `"held_out": false` rather than implying otherwise.

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
- `train.py` runs `git add . && git commit && git tag <run>` on rank zero at startup **only when
  `--auto-tag-run` is passed** (`train.py:81`, `auto_tag_run` defaults to `False` in
  `config.py:149`). Neither vibe-repose launcher passes it, so this does not fire by default —
  but if you ever turn it on, note it is a bare `git add .`, so anything untracked in the repo
  gets swept in. `PSI_HOME` locally sits *inside* the repo and is only saved by `psi_home/cache/`
  matching an ignore rule; on CARC it is `$ROOT/psi_home`, outside the tree.
- `WANDB_API_KEY` in `.env` must be set — the launch script passes `--log.report_to=wandb`.
  `--wandb.project` must be passed explicitly too; see the CARC section for why.
- **`.runs/<train.name>/<exp>...` holds the real log**, at
  `wandb/run-*/files/output.log`, plus `wandb-summary.json` with the final metrics. When a run
  dies under `torchrun`, the console shows only `ChildFailedError` with no cause — the actual
  traceback is in that `output.log`. Look there first.
- The vibe workspace (`../vibe`, `../songen`) is **not a git repo** and exists only on this
  machine. `collect_rollouts.py` and `extract_latents.py` import it, so steps 1–2 **cannot be
  re-run anywhere else**. The `-raw` HF repo above is the only copy of their output — treat it as
  the reproducibility boundary, not as a convenience mirror.
