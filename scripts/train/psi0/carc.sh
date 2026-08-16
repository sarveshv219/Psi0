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
# Ordered newest-first within torchcodec's supported range (FFmpeg 4-7 -> libavutil.so.56-59).
# `ffmpeg` bare is LAST on purpose: on an Lmod hierarchy the bare name can resolve only after
# its compiler prerequisite is loaded, so an explicit version is the more reliable ask.
FFMPEG_MODULES=${FFMPEG_MODULES:-"ffmpeg/7.0 ffmpeg/6.1.1 ffmpeg/6.1 ffmpeg/6.0 ffmpeg/5.1.2 ffmpeg/4.4 ffmpeg"}
# Prerequisites to load before ffmpeg when the module tree is hierarchical. Loading a compiler
# is what makes the ffmpeg/* names visible at all.
FFMPEG_PREREQS=${FFMPEG_PREREQS:-"usc gcc"}

# Toolkit modules to hunt for an `nvcc` -- read in a subshell, never loaded into this one. Only
# the MAJOR has to match torch's (12), so any cuda/12.x works; see setup_cuda_home().
CUDA_MODULES=${CUDA_MODULES:-"cuda/12.6.3 cuda/12.6 cuda/12.4 cuda/12.2 cuda/12 cuda"}
[[ -n ${PSI0_CUDA_HOME:-} ]] && export CUDA_HOME=$PSI0_CUDA_HOME

# `module` is a shell FUNCTION from /etc/profile.d, and an sbatch payload runs in a
# non-interactive, non-login shell. Lmod exports it (BASH_FUNC_module%%) so --export=ALL usually
# carries it in -- but "usually" is how a sweep dies. Source the init explicitly when it is
# missing, so the script can load its own modules instead of depending on the caller's shell.
init_modules() {
    command -v module >/dev/null 2>&1 && return 0
    local f
    for f in "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash" \
             /usr/share/lmod/lmod/init/bash \
             /usr/share/Modules/init/bash \
             /etc/profile.d/modules.sh \
             /etc/profile.d/lmod.sh; do
        # shellcheck disable=SC1090
        [[ -r $f ]] && { source "$f" 2>/dev/null || true; }
        command -v module >/dev/null 2>&1 && { echo "[modules] initialized from $f"; return 0; }
    done
    echo "[modules] WARN: no module command available" >&2
    return 1
}

have_libavutil() {
    local d
    ldconfig -p 2>/dev/null | grep -qE 'libavutil\.so\.5[6-9]' && return 0
    for d in ${LD_LIBRARY_PATH//:/ }; do
        compgen -G "$d/libavutil.so.5[6-9]" >/dev/null 2>&1 && return 0
    done
    return 1
}

setup_ffmpeg() {
    # FIRST: is it already here? SLURM propagates the submitting shell's environment
    # (--export=ALL is the default), so a job usually INHERITS a working ffmpeg. Checking
    # before touching anything is what stops us from destroying a good environment.
    if have_libavutil; then echo "[ffmpeg] OK (inherited)"; return 0; fi

    if [[ -n ${PSI0_FFMPEG_DIR:-} ]]; then
        export PATH="$PSI0_FFMPEG_DIR/bin:$PATH"
        export LD_LIBRARY_PATH="$PSI0_FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
        echo "[ffmpeg] PSI0_FFMPEG_DIR=$PSI0_FFMPEG_DIR"
    elif init_modules; then
        local m
        for m in $FFMPEG_PREREQS; do module load "$m" 2>/dev/null || true; done
        for m in $FFMPEG_MODULES; do
            module load "$m" 2>/dev/null && have_libavutil \
                && { echo "[ffmpeg] module $m"; break; }
        done
    fi

    if have_libavutil; then
        echo "[ffmpeg] OK: $(ldconfig -p 2>/dev/null | grep -oE 'libavutil\.so\.5[6-9]' | head -1)${LD_LIBRARY_PATH:+ (or on LD_LIBRARY_PATH)}"
    else
        echo "[ffmpeg] FATAL: no libavutil.so.5{6,7,8,9} on the loader path." >&2
        echo "         torchcodec will crash a DataLoader worker on the first batch." >&2
        echo "         Loaded modules:"; module list 2>&1 | sed 's/^/           /' >&2 || true
        echo "         Fix: module spider ffmpeg, then re-submit with" >&2
        echo "              FFMPEG_MODULES='ffmpeg/<ver>'   or   PSI0_FFMPEG_DIR=<prefix>" >&2
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

# deepspeed 0.17.1 runs an op-compatibility scan at IMPORT time (git_version_info.py:29), and
# fp_quantizer's is_compatible() calls installed_cuda_version() without catching the
# MissingCUDAException it raises when torch.utils.cpp_extension.CUDA_HOME is None. accelerate's
# extract_model_from_parallel imports deepspeed whenever the package is merely INSTALLED -- it
# only wants DeepSpeedEngine for an isinstance tuple -- so a pure `data_parallel=ddp` run with no
# nvcc on the node dies at its FIRST evaluate() -- which train.py:260 fires at global_step == 0,
# NOT at validation_steps, so it costs ~2 min of a job, not hours.
#
# deepspeed needs exactly one thing: $CUDA_HOME/bin/nvcc to answer -V. So export the variable and
# NOTHING else. Do not `module load cuda` in this shell: its lib64 lands ahead of torch's bundled
# CUDA on LD_LIBRARY_PATH, which is searched before the RUNPATH torch resolves its own libs with.
# Reading the prefix out of a SUBSHELL gets the path without the side effects.
setup_cuda_home() {
    local p
    # Test the variable BEFORE using it as a prefix: unset, "${CUDA_HOME:-}/bin/nvcc" is the
    # absolute path /bin/nvcc, which on a box with a system toolkit exists -- and then p is set
    # to the empty string and the real search never runs.
    if [[ -n ${CUDA_HOME:-} && -x $CUDA_HOME/bin/nvcc ]]; then p=$CUDA_HOME
    elif [[ -n ${CUDA_PATH:-} && -x $CUDA_PATH/bin/nvcc ]]; then p=$CUDA_PATH
    elif p=$(command -v nvcc 2>/dev/null) && [[ -n $p ]]; then p=$(dirname "$(dirname "$p")")
    elif init_modules; then
        local m
        for m in $CUDA_MODULES; do
            # Subshell: the module's PATH/LD_LIBRARY_PATH edits die with it, the string survives.
            p=$(module load "$m" >/dev/null 2>&1 && command -v nvcc 2>/dev/null) || continue
            [[ -n $p ]] && { p=$(dirname "$(dirname "$p")"); echo "[cuda_home] module $m"; break; }
        done
    fi
    if [[ -z ${p:-} || ! -x $p/bin/nvcc ]]; then
        echo "[cuda_home] WARN: no nvcc found. If deepspeed is installed, the first evaluate()" >&2
        echo "  will raise MissingCUDAException. Fix with PSI0_CUDA_HOME=/path/to/cuda, or drop" >&2
        echo "  the package (\`uv pip uninstall deepspeed\`) -- ddp never constructs it." >&2
        return 1
    fi
    export CUDA_HOME=$p
    # is_compatible() only compares CUDA majors, but a builder that calls assert_no_cuda_mismatch
    # would reject 12.6.3-vs-12.6 on the minor. Nothing is compiled here, so waive it.
    export DS_SKIP_CUDA_CHECK=1
    echo "[cuda_home] $CUDA_HOME ($("$p/bin/nvcc" --version | tail -1 | tr -s ' '))"
}

# =============================================================================
# payload (inside the allocation)
# =============================================================================
setup_run_env() {
    cd "$PSI"

    # DO NOT `module purge` here. SLURM's default --export=ALL means the job inherits the
    # submitting shell's modules, so a purge throws away a working ffmpeg -- and on an Lmod
    # hierarchy it also unloads the compiler that makes `ffmpeg/*` visible, so nothing can
    # reload it. That killed a whole sweep. Set PSI0_MODULE_PURGE=1 only if a stray module is
    # actually causing trouble, and pass FFMPEG_MODULES with it.
    if [[ ${PSI0_MODULE_PURGE:-0} = 1 ]] && command -v module >/dev/null 2>&1; then
        module purge 2>/dev/null || true
    fi
    setup_ffmpeg || exit 1
    setup_cuda_home || true   # only fatal if deepspeed is installed; the preflight decides

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
import torch, sys, os
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

# accelerate imports deepspeed inside unwrap_model() whenever the package is merely installed,
# and deepspeed scans its CUDA op builders at import. Do it HERE so a missing nvcc costs 10
# seconds at startup instead of after the model load and the step-0 validation.
from accelerate.utils.imports import is_deepspeed_available
if is_deepspeed_available():
    try:
        import deepspeed  # noqa: F401
    except Exception as e:
        sys.exit(f"[preflight] FATAL: deepspeed is installed but unimportable\n"
                 f"  {type(e).__name__}: {e}\n"
                 f"  CUDA_HOME={os.environ.get('CUDA_HOME', '<unset>')}\n"
                 f"  accelerate's extract_model_from_parallel imports it for an isinstance\n"
                 f"  check, so this WILL kill the run at its first evaluate(). Either point\n"
                 f"  PSI0_CUDA_HOME at a toolkit with bin/nvcc, or `uv pip uninstall deepspeed`\n"
                 f"  -- data_parallel=ddp never constructs a DeepSpeedEngine.")
    print(f"[preflight] deepspeed {deepspeed.__version__} imports OK")
PY
}

run_one() {  # run_one <gpu_idx> <exp> [ovr...]
    local gpu=$1 exp=$2; shift 2
    echo "=== [gpu:$gpu] $exp ${*:+ovr=$*} ==="
    # One process per GPU. torchrun --nproc_per_node=1 (rather than plain python) so the
    # distributed env vars accelerate expects under data_parallel=ddp are always set.
    #
    # --standalone, NOT --master_port=$((29500 + gpu)). A per-GPU offset only deconflicts runs
    # that this shell launched, and every array element calls run_one with gpu=0 -- so as soon
    # as SLURM packs two array tasks onto one node (which it does: an L40S node has 2-4 cards
    # and --gres=gpu:1 leaves the rest free) both bind 29500 and all but one die with
    # `DistNetworkError ... EADDRINUSE`. --standalone sets rdzv-endpoint=localhost:0, so the
    # kernel assigns a free ephemeral port; the agent still exports MASTER_ADDR/MASTER_PORT to
    # the worker, which is all accelerate reads.
    CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nproc_per_node=1 \
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
