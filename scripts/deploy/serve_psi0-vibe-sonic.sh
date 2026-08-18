#!/bin/bash
# Serve a vibe-repose SONIC-latent checkpoint for vibe/scripts/play_psi0.py.
#
#   bash scripts/deploy/serve_psi0-vibe-sonic.sh .runs/sonic/<run-dir> 40000
#
# Synchronous by default: action_exec_horizon == action_chunk_size == 50, so the client consumes a
# whole 1.0 s plan before asking for the next one and there is no inference delay for RTC to hide.
# Set RTC=1 to take predict_action_with_training_rtc_flow instead (the run trained with
# --model.rtc --model.max-delay=8, so the weights support it); the client signals episode
# boundaries with history={"reset": true}, which is what makes the server drop its previous chunk.
#
# psi0_serve_real_sonic.py, NOT the `serve_psi0` entry point: this is the sonic server, and its
# wire contract is states (To, Ds) padded on dim=1 -- what psi0_planner.py sends.
#
# The run dir is self-contained. run_config.json carries action_min/max (64) and state_min/max (32)
# inline, so DataConfig.load_stats short-circuits and the corpus does NOT need to be on this
# machine -- despite stat_path being the unresolvable relative "meta/stats_psi0.json".
set -euo pipefail

cd "$(dirname "$0")/../.."
[[ $# -ge 2 ]] || { echo "usage: $0 RUN_DIR CKPT_STEP [PORT]"; exit 2; }
RUN_DIR=$1 CKPT_STEP=$2 PORT=${3:-8014}

[[ -f $RUN_DIR/run_config.json ]] || { echo "[fatal] no run_config.json in $RUN_DIR"; exit 1; }
[[ -f $RUN_DIR/argv.txt ]]        || { echo "[fatal] no argv.txt in $RUN_DIR"; exit 1; }
[[ -f $RUN_DIR/checkpoints/ckpt_$CKPT_STEP/model.safetensors ]] || {
    echo "[fatal] no checkpoints/ckpt_$CKPT_STEP/model.safetensors in $RUN_DIR"
    echo "        have: $(ls "$RUN_DIR/checkpoints" 2>/dev/null | tr '\n' ' ')"; exit 1; }

# shellcheck disable=SC1091
source .venv-psi/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

echo "[serve] $RUN_DIR @ ckpt_$CKPT_STEP -> 0.0.0.0:$PORT  (GPU $CUDA_VISIBLE_DEVICES, rtc=${RTC:-0})"
python src/psi/deploy/psi0_serve_real_sonic.py \
    --host 0.0.0.0 --port "$PORT" \
    --policy psi0 \
    --run-dir="$RUN_DIR" \
    --ckpt-step="$CKPT_STEP" \
    ${RTC:+--rtc}
