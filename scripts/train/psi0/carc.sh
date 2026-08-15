#!/bin/bash
# =============================================================================
# Psi0 vibe-repose launcher — USC CARC Discovery.
#
# Same shape as vibe/train.sh: ONE RUN PER GPU. Psi0's static footprint is ~13 GB
# (VLM 4.26 frozen + head 1.99 + grads 1.99 + AdamW 3.98) and activations are only
# ~0.04 GB/sample, so a 48 GB card fits one run with a large batch and there is no
# reason to shard a run across cards. -n packs n INDEPENDENT runs onto one node.
#
#   setup   bash carc.sh setup
#           Login node, once: uv venv + deps + pretrained checkpoints.
#           flash-attn compiles from source -- run this in an interactive job
#           (salloc -c 8 --mem 32G -t 2:00:00), not on a login node.
#
#   data    bash carc.sh data
#           Login node, once: pull the LeRobot corpus from the HF dataset repo
#           and VERIFY the file counts. A rate-limited download exits 0.
#
#   smoke   bash carc.sh smoke -g l40s
#           20 steps, validation at 10. RUN THIS FIRST — no complete optimizer
#           step and no evaluate() against the real model has ever executed
#           (the dev box is 8 GB and OOMs in backward).
#
#   submit  bash carc.sh submit -g l40s [-t 24] [-b 32] [-s 40000] \
#               [--dry-run] -- <EXP> [OVR...]
#           One run, one GPU. n>1 packs n runs on one node via a manifest:
#           bash carc.sh submit -g l40s -n 2 --manifest runs.tsv
#
#   sweep   bash carc.sh sweep -g l40s --manifest runs.tsv
#           Job array: 1 GPU + 1 manifest line per element (schedules fastest).
#
#   Manifest: one run per line, whitespace-split:  EXP [OVR...]
#             e.g. the pretrained-vs-scratch control that decides whether an
#             expert post-trained on Humanoid-Everyday JOINT actions transfers
#             to a LATENT action space (untested by anyone):
#                 arm-pretrained
#                 arm-scratch    --model.pretrained-action-header-path=None
#
#   run / run-array are internal (executed inside the allocation by sbatch).
# =============================================================================

set -euo pipefail

ROOT=${CARC_ROOT:-/scratch1/$USER}
PSI=$ROOT/Psi0
VENV=$PSI/.venv-psi
SELF=$PSI/scripts/train/psi0/carc.sh
export PSI_HOME=${PSI_HOME:-$ROOT/psi_home}

WANDB_PROJECT=${WANDB_PROJECT:-psi0-vibe-repose}
WANDB_ENTITY=${WANDB_ENTITY:-vbp}

# Psi0 is dataloader-hungry where vibe is not: every sample decodes one h264 frame
# through torchcodec, so CPU per GPU is ~2x vibe's and host memory holds the worker
# pool. Verify with `seff <JID>` and trim.
CPUS_PER_GPU=8
MEM_GB_PER_GPU=32

DEF_GPU=l40s DEF_N=1 DEF_HOURS=24 DEF_BATCH=32 DEF_STEPS=40000 DEF_PART=gpu

CKPT_VLM=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
CKPT_HEAD=psi0/postpre.1by1.pad36.2601131206.ckpt.he30k

HF_DATASET=${HF_DATASET:-sarveshv219/vibe-repose-sim}
N_TRAIN_EP=3246 N_VAL_EP=573

# torchcodec dlopens libavutil.so.5{6,7,8,9} by soname at FIRST DECODE, which is inside a
# DataLoader worker on batch 1 -- i.e. after the 4 GB VLM has loaded and the allocation is
# already spent. pip cannot supply these (imageio-ffmpeg ships a static binary, PyAV ships
# hash-mangled sonames). Resolve them BEFORE python starts; glibc caches the search path at
# process startup, so exporting this from inside Python or from .env is too late.
#
# Set PSI0_FFMPEG_DIR to a prefix with lib/libavutil.so.5x to skip the module hunt, e.g. a
# conda env made with: conda create -y -p $ROOT/ffmpeg6 -c conda-forge 'ffmpeg=6.1'
FFMPEG_MODULES=${FFMPEG_MODULES:-"ffmpeg ffmpeg/6.1.1 ffmpeg/6.0 ffmpeg/5.1.2 ffmpeg/4.4"}

setup_ffmpeg() {
    if [[ -n ${PSI0_FFMPEG_DIR:-} ]]; then
        export PATH="$PSI0_FFMPEG_DIR/bin:$PATH"
        export LD_LIBRARY_PATH="$PSI0_FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
        echo "[ffmpeg] PSI0_FFMPEG_DIR=$PSI0_FFMPEG_DIR"
    elif command -v module >/dev/null 2>&1; then
        local m
        for m in $FFMPEG_MODULES; do
            module load "$m" 2>/dev/null && { echo "[ffmpeg] module $m"; break; }
        done
    fi

    # Verify rather than hope: this is the exact lookup torchcodec will do, and doing it here
    # costs a second where failing later costs the job.
    local found
    found=$(ldconfig -p 2>/dev/null | grep -oE 'libavutil\.so\.5[6-9]' | head -1)
    [[ -z $found && -n ${LD_LIBRARY_PATH:-} ]] && found=$(
        ls ${LD_LIBRARY_PATH//:/ }/libavutil.so.5[6-9] 2>/dev/null | head -1)
    if [[ -n $found ]]; then
        echo "[ffmpeg] OK: $found"
    else
        echo "[ffmpeg] FATAL: no libavutil.so.5{6,7,8,9} on the loader path." >&2
        echo "         torchcodec will crash a DataLoader worker on the first batch." >&2
        echo "         Fix: module spider ffmpeg   OR   set PSI0_FFMPEG_DIR (see comment above)." >&2
        return 1
    fi
}

# =============================================================================
# setup — login node, once
# =============================================================================
mode_setup() {
    echo "=== Psi0 CARC setup: venv=$VENV repo=$PSI PSI_HOME=$PSI_HOME ==="
    [[ -d $PSI ]] || { echo "[fatal] clone the repo to $PSI first"; exit 1; }
    cd "$PSI"

    command -v uv >/dev/null || {
        echo "[setup] installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    }

    # NOT `uv sync --active`: --active reads $VIRTUAL_ENV, and with nothing activated
    # uv silently installs ~7 GB into the project default .venv instead.
    #
    # UV_HTTP_TIMEOUT/CONCURRENT_DOWNLOADS: torch drags ~3 GB of bundled CUDA libs from
    # pypi.nvidia.com (cudnn alone is 693 MB) and the default timeout expires mid-wheel.
    # uv banks completed wheels, so re-running this makes progress rather than restarting.
    # --no-install-package rerun-sdk: CARC is glibc 2.28 (manylinux_2_28) and rerun-sdk has no
    # 2_28 wheel between 0.23 and 0.31.3, while lerobot caps it at <0.23.0 -- so NO version
    # satisfies both and `uv sync` hard-fails on a package we never load. Verified: importing
    # LeRobotDataset leaves `rerun` out of sys.modules; the only lerobot files that touch it are
    # visualize_dataset / teleoperate / record / visualization_utils, and psi imports none of
    # them. If a future lerobot lifts the cap, prefer `[tool.uv] override-dependencies =
    # ["rerun-sdk>=0.31.3"]` over this flag.
    UV_PROJECT_ENVIRONMENT="$VENV" GIT_LFS_SKIP_SMUDGE=1 \
    UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 uv sync \
        --group serve --group viz --group psi --index-strategy unsafe-best-match \
        --no-install-package rerun-sdk

    # flash-attn is OPTIONAL here and the install is best-effort. Its setup.py prefers a
    # prebuilt wheel from GitHub releases over compiling, and those are linked against a newer
    # glibc than an RHEL-8 cluster has -- the result imports on the build host and dies here
    # with "GLIBC_2.32 not found", from a DataLoader-free code path that no resolver can catch.
    #
    # So: install, then VERIFY BY IMPORTING, and remove it if it is a lie. Removal matters
    # beyond tidiness -- transformers' is_flash_attn_2_available() tests package metadata, not
    # importability, so a broken-but-present flash_attn makes every such guard return True.
    # sonic.py falls back to sdpa on its own; at 100 VLM tokens the difference is noise.
    #
    # --no-deps IS LOAD-BEARING. flash-attn declares a bare, unbounded `torch`, so any install
    # of it is licensed to re-resolve the pinned stack. Adding --reinstall once dragged CARC
    # from torch 2.7.0+cu126 to 2.13.0+cu130, which then failed the flash-attn build against a
    # cuda/12.6 module. Never install this package without --no-deps.
    #
    # To insist on flash-attn, build the wheel yourself -- FLASH_ATTENTION_FORCE_BUILD does not
    # survive uv's PEP 517 subprocess, and the arch var is FLASH_ATTN_CUDA_ARCHS (setup.py:69),
    # NOT TORCH_CUDA_ARCH_LIST, with branches for only 80/90/100/120. "80" covers sm_86/sm_89
    # too: cubins are binary-compatible across minor versions within a major arch.
    #     salloc -c 8 --mem 32G -t 2:00:00 && module load gcc cuda
    #     curl -sL https://files.pythonhosted.org/packages/source/f/flash-attn/\
    #         flash_attn-2.7.4.post1.tar.gz | tar xz && cd flash_attn-2.7.4.post1
    #     FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4 NVCC_THREADS=2 \
    #         $VENV/bin/python setup.py bdist_wheel
    #     uv pip install dist/flash_attn-*.whl --no-deps --python $VENV/bin/python
    if [[ ${PSI0_SKIP_FLASH_ATTN:-0} != 1 ]]; then
        MAX_JOBS=${MAX_JOBS:-4} VIRTUAL_ENV="$VENV" \
            uv pip install flash_attn==2.7.4.post1 --no-build-isolation --no-deps || true
        if "$VENV/bin/python" -c "import flash_attn" 2>/dev/null; then
            echo "[setup] flash_attn OK"
        else
            echo "[setup] flash_attn present but not importable here — removing it so"
            echo "        is_flash_attn_2_available() reports False. Training uses sdpa."
            VIRTUAL_ENV="$VENV" uv pip uninstall flash_attn >/dev/null 2>&1 || true
        fi
    fi

    # The local .env hardcodes /home/sarvesh paths; copying it points DATA_HOME and
    # HF_LEROBOT_HOME at directories that do not exist here. Write a fresh one.
    # train.py:4 asserts load_dotenv() is truthy, so an absent OR EMPTY .env is fatal there.
    if [[ ! -f $PSI/.env ]]; then
        echo "[setup] writing a CARC .env (fill in HF_TOKEN / WANDB_API_KEY)"
        cat > "$PSI/.env" <<EOF
HF_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=$WANDB_ENTITY
PSI_HOME=$PSI_HOME
DATA_HOME=$PSI/data
HF_HOME=$ROOT/cache/hf
HF_LEROBOT_HOME=$PSI/data/lerobot
OMP_NUM_THREADS=$CPUS_PER_GPU
TOKENIZERS_PARALLELISM=false
TF_CPP_MIN_LOG_LEVEL=3
EOF
    fi
    grep -q '^HF_TOKEN=.\+'      "$PSI/.env" || echo "[warn] HF_TOKEN empty — the dataset repo is private"
    grep -q '^WANDB_API_KEY=.\+' "$PSI/.env" || echo "[warn] WANDB_API_KEY empty and the config passes --log.report_to=wandb"

    for d in "$CKPT_VLM" "$CKPT_HEAD"; do
        [[ -d $PSI_HOME/cache/checkpoints/$d ]] && { echo "[setup] have $d"; continue; }
        "$VENV/bin/python" scripts/data/download.py --repo-id=USC-PSI-Lab/psi-model \
            --repo-type=model --remote-dir="$d" --local-dir="$PSI_HOME/cache/checkpoints/$d"
    done

    # `import torchcodec` is the real check: it is what runs load_torchcodec_shared_libraries()
    # and raises "Could not load libtorchcodec". Training never hits it in the main process --
    # only a DataLoader worker does -- so import it HERE, under the same loader path a job gets.
    setup_ffmpeg || exit 1
    "$VENV/bin/python" -c "import psi, torch, torchcodec; print('[setup] imports OK', torch.__version__)"

    mkdir -p "$PSI/logs/slurm/manifests"
    if [[ -d $PSI/data/lerobot/vibe_repose_g1 ]]; then
        echo "=== setup done. Next: bash carc.sh smoke -g l40s ==="
    else
        echo "=== setup done. Next: bash carc.sh data ==="
    fi
}

# =============================================================================
# data — login node, once. Compute nodes should never reach for the network.
# =============================================================================
mode_data() {
    cd "$PSI"
    local dest=$PSI/data/lerobot
    mkdir -p "$dest"

    # HF_HUB_DISABLE_XET: the packed dataset is ~7,650 files and Xet requests a token PER FILE,
    # which blows the 1000-per-5-minutes quota and 429s. --max-workers 4 keeps it under.
    HF_HUB_DISABLE_XET=1 "$VENV/bin/hf" download "$HF_DATASET" --repo-type=dataset \
        --local-dir "$dest" --max-workers 4 || true

    # A rate-limited snapshot_download EXITS 0 and returns the partial directory, so the exit
    # code above proves nothing. Count what actually landed; re-running resumes.
    local ok=1 n
    for pair in "vibe_repose_g1:$N_TRAIN_EP" "vibe_repose_g1_val:$N_VAL_EP"; do
        n=$(find "$dest/${pair%%:*}" -name '*.parquet' 2>/dev/null | wc -l)
        if [[ $n -eq ${pair##*:} ]]; then
            echo "[data] ${pair%%:*}: $n parquet OK"
        else
            echo "[data] ${pair%%:*}: $n parquet, expected ${pair##*:}  <-- INCOMPLETE" >&2
            ok=0
        fi
    done
    (( ok )) || { echo "[fatal] partial download — re-run 'bash carc.sh data' until counts match"; exit 1; }
    echo "=== data done. Next: bash carc.sh smoke -g l40s ==="
}

# =============================================================================
# options
# =============================================================================
GPU=$DEF_GPU N=$DEF_N HOURS=$DEF_HOURS BATCH=$DEF_BATCH STEPS=$DEF_STEPS
PART=$DEF_PART MANIFEST="" DRY=0 ACCOUNT=${SBATCH_ACCOUNT:-}

parse_opts() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -A) ACCOUNT=$2; shift 2 ;;
            -g) GPU=$2; shift 2 ;;
            -n) N=$2; shift 2 ;;
            -t) HOURS=$2; shift 2 ;;
            -b) BATCH=$2; shift 2 ;;
            -s) STEPS=$2; shift 2 ;;
            -p) PART=$2; shift 2 ;;
            --manifest) MANIFEST=$2; shift 2 ;;
            --dry-run) DRY=1; shift ;;
            --) shift; RUN_ARGS=("$@"); return ;;
            *) echo "[fatal] unknown option: $1"; exit 2 ;;
        esac
    done
    RUN_ARGS=()
}

# Snapshot so edits after submit don't change a queued job.
snapshot_manifest() {
    [[ -f $MANIFEST ]] || { echo "[fatal] manifest not found: $MANIFEST"; exit 1; }
    local snap="$PSI/logs/slurm/manifests/$(date +%Y%m%d_%H%M%S).tsv"
    mkdir -p "$(dirname "$snap")"
    grep -vE '^\s*(#|$)' "$MANIFEST" > "$snap"
    echo "$snap"
}

do_sbatch() {  # do_sbatch <extra sbatch flags...> -- <payload args...>
    local flags=() payload=()
    while [[ $1 != -- ]]; do flags+=("$1"); shift; done; shift
    payload=("$@")
    local cmd=(sbatch --job-name=train --partition="$PART"
               ${ACCOUNT:+--account=$ACCOUNT}
               --nodes=1 --ntasks=1
               --cpus-per-task=$((CPUS_PER_GPU * N)) --mem=$((MEM_GB_PER_GPU * N))G
               --time="${HOURS}:00:00"
               --output="$PSI/logs/slurm/%x-%j.out" --error="$PSI/logs/slurm/%x-%j.err"
               "${flags[@]}" "$SELF" "${payload[@]}")
    if [[ $DRY = 1 ]]; then echo "[dry-run] ${cmd[*]}"; return; fi
    mkdir -p "$PSI/logs/slurm"
    "${cmd[@]}"
    squeue --me
}

# =============================================================================
# submit / smoke / sweep
# =============================================================================
mode_submit() {
    parse_opts "$@"
    if [[ -n $MANIFEST ]]; then
        local snap; snap=$(snapshot_manifest)
        local lines; lines=$(wc -l < "$snap")
        (( lines >= N )) || { echo "[fatal] manifest has $lines runs < $N gpus"; exit 1; }
        do_sbatch --gres="gpu:$GPU:$N" -- run "$BATCH" "$STEPS" manifest "$snap" "$N"
    else
        [[ ${#RUN_ARGS[@]} -ge 1 ]] || { usage; exit 2; }
        (( N == 1 )) || { echo "[fatal] n>1 needs --manifest"; exit 2; }
        do_sbatch --gres="gpu:$GPU:1" -- run "$BATCH" "$STEPS" inline "${RUN_ARGS[@]}"
    fi
}

mode_smoke() {
    parse_opts "$@"
    # 20 steps with validation at 10 exercises the three things the dev box cannot:
    # a complete optimizer.step(), evaluate() against the real model, and checkpointing.
    N=1 HOURS=1
    do_sbatch --gres="gpu:$GPU:1" -- run "$BATCH" 20 inline smoke20 \
        --train.validation_steps=10 --train.val_num_batches=2 \
        --train.checkpointing_steps=100000
}

mode_sweep() {
    parse_opts "$@"
    [[ -n $MANIFEST ]] || { echo "[fatal] sweep needs --manifest"; exit 2; }
    local snap; snap=$(snapshot_manifest)
    local lines; lines=$(wc -l < "$snap")
    N=1  # resources are per-element
    do_sbatch --gres="gpu:$GPU:1" --array="0-$((lines - 1))" \
        --output="$PSI/logs/slurm/%x-%A_%a.out" --error="$PSI/logs/slurm/%x-%A_%a.err" \
        -- run-array "$BATCH" "$STEPS" "$snap"
}

# =============================================================================
# payload (inside the allocation)
# =============================================================================
setup_run_env() {
    cd "$PSI"

    # Purge first: a login shell's modules are inherited by the batch script, and a stray
    # python/cudnn/nccl module is worse than none -- torch bundles its own under
    # site-packages/nvidia/ and a version-mismatched module silently shadows them.
    if command -v module >/dev/null 2>&1; then module purge 2>/dev/null || true; fi
    setup_ffmpeg || exit 1

    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    export WANDB_PROJECT WANDB_ENTITY WANDB_DIR="$PSI/wandb"
    export HF_HOME=${HF_HOME:-$ROOT/cache/hf}

    # Every checkpoint and every parquet is on local disk by now, so any hub call is either a
    # revision check that can hang or a rate limit that can 429. Turn both into an immediate
    # error. Set PSI0_HF_ONLINE=1 if a run genuinely needs to fetch something.
    if [[ ${PSI0_HF_ONLINE:-0} != 1 ]]; then
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    fi

    JOB_LOG_DIR=$PSI/logs/slurm/${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}
    mkdir -p "$JOB_LOG_DIR" "$WANDB_DIR" "$HF_HOME"
    echo "=== $(hostname) | job ${SLURM_JOB_ID:-?} | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | batch=$BATCH steps=$STEPS ==="

    # Run one real kernel on the allocated card. If the wheel's CUDA variant does not cover this
    # arch, the alternative is discovering it as "no kernel image is available" on the loss_w
    # tensor in sonic.py, minutes in and with the traceback buried in a ChildFailedError.
    python - <<'PY' || exit 1
import torch, sys
if not torch.cuda.is_available():
    sys.exit("[preflight] FATAL: torch.cuda.is_available() is False")
name = torch.cuda.get_device_name(0)
cap  = "sm_%d%d" % torch.cuda.get_device_capability(0)
try:
    (torch.ones(8, device="cuda") * 2).sum().item()
except RuntimeError as e:
    sys.exit(f"[preflight] FATAL: {name} ({cap}) cannot run a kernel from torch "
             f"{torch.__version__}\n  {e}\n  Reinstall torch for this arch "
             f"(see CLAUDE.md 'Bringing up a new machine' trap 1).")
print(f"[preflight] {name} {cap} OK, torch {torch.__version__}")
PY
}

run_one() {  # run_one <gpu_idx> <exp> [ovr...]
    local gpu=$1 exp=$2; shift 2
    echo "=== [gpu:$gpu] $exp ${*:+ovr=$*} ==="
    # One process per GPU. torchrun --nproc_per_node=1 (rather than plain python) so the
    # distributed env vars accelerate expects under data_parallel=ddp are always set; the
    # port is per-GPU so packed runs on one node cannot collide on rendezvous.
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29500 + gpu)) \
        scripts/train.py \
        finetune_real_psi0_config \
        --seed=292285 --exp="$exp" \
        --train.name=sonic --train.data_parallel=ddp --train.mixed_precision=bf16 \
        --train.train_batch_size="$BATCH" \
        --train.max_training_steps="$STEPS" \
        --train.learning_rate=1e-4 --train.warmup_steps=1000 --train.warmup_ratio=None \
        --train.checkpointing_steps=5000 --train.validation_steps=1000 \
        --train.val_num_batches=20 --train.max_checkpoints_to_keep=5 \
        --train.gradient_accumulation_steps=1 --train.max_grad_norm=1.0 \
        --train.lr_scheduler_type=cosine \
        --train.lr_scheduler_kwargs.weight_decay=1e-6 \
        --train.lr_scheduler_kwargs.betas 0.95 0.999 \
        --log.report_to=wandb \
        --wandb.project="$WANDB_PROJECT" \
        --data.root_dir=data/lerobot \
        --data.train_repo_ids=vibe_repose_g1 \
        --data.val_repo_ids=vibe_repose_g1_val \
        --data.transform.field.stat-path=meta/stats_psi0.json \
        --data.transform.field.stat-action-key=action \
        --data.transform.field.stat-state-key=states \
        --data.transform.field.action_norm_type=bounds \
        --data.transform.field.no-use-norm-mask \
        --data.transform.field.normalize-state \
        --data.transform.model.resize.size 240 320 \
        --data.transform.model.center_crop.size 240 320 \
        --data.transform.model.no-img-aug \
        --data.transform.repack.action_chunk_size=50 \
        --model.model_name_or_path="$PSI_HOME/cache/checkpoints/$CKPT_VLM" \
        --model.pretrained-action-header-path="$PSI_HOME/cache/checkpoints/$CKPT_HEAD" \
        --model.noise-scheduler=flow --model.train-diffusion-steps=1000 \
        --model.n_conditions=0 --model.observation-horizon=1 \
        --model.view_feature_dim=2048 \
        --model.action-dim=64 --model.odim=32 \
        --model.action-chunk-size=50 --model.action-exec-horizon=50 \
        --model.no-tune-vlm --model.no-use_film --model.no-combined_temb \
        --model.rtc --model.max-delay=8 \
        "$@" \
        > "$JOB_LOG_DIR/gpu$gpu.out" 2> "$JOB_LOG_DIR/gpu$gpu.err"
}

# Gate the next launch on this one reaching step 1. Packed runs otherwise all load a
# 6.25 GB checkpoint set at once and thrash page cache and the HF lock. Releases on
# crash/timeout so one bad run cannot wedge the node.
wait_ready() {
    local gpu=$1 pid=$2 out="$JOB_LOG_DIR/gpu$1.err" t=900 start=$SECONDS
    while true; do
        grep -qE "Running training|Training steps" "$out" 2>/dev/null && { echo "[stagger] gpu$gpu up"; return; }
        kill -0 "$pid" 2>/dev/null || { echo "[stagger] WARN gpu$gpu exited before step 1"; return; }
        (( SECONDS - start > t )) && { echo "[stagger] WARN gpu$gpu timeout ${t}s"; return; }
        sleep 5
    done
}

mode_run() {  # run <batch> <steps> inline <exp> [ovr...] | manifest <snap> <n>
    BATCH=$1 STEPS=$2; local kind=$3; shift 3
    setup_run_env
    if [[ $kind = inline ]]; then
        run_one 0 "$@"
    else
        local snap=$1 n=$2 i=0
        while IFS= read -r line && (( i < n )); do
            # shellcheck disable=SC2086
            run_one "$i" $line &
            (( i < n - 1 )) && wait_ready "$i" "$!"
            i=$((i + 1))
        done < "$snap"
        echo "[launch] all $i started"
        wait
    fi
    echo "=== done | $JOB_LOG_DIR/gpu*.{out,err} ==="
}

mode_run_array() {  # run-array <batch> <steps> <snap>
    BATCH=$1 STEPS=$2; local snap=$3
    setup_run_env
    local line; line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$snap")
    [[ -n $line ]] || { echo "[fatal] no manifest line $SLURM_ARRAY_TASK_ID"; exit 1; }
    # shellcheck disable=SC2086
    run_one 0 $line
}

usage() { sed -n '2,/^# =\{20,\}$/p' "$0" | sed 's/^# \{0,1\}//'; }

MODE=${1:-}; [[ $# -gt 0 ]] && shift
case "$MODE" in
    setup)     mode_setup ;;
    data)      mode_data ;;
    submit)    mode_submit "$@" ;;
    smoke)     mode_smoke "$@" ;;
    sweep)     mode_sweep "$@" ;;
    run)       mode_run "$@" ;;
    run-array) mode_run_array "$@" ;;
    *)         usage; exit 2 ;;
esac
