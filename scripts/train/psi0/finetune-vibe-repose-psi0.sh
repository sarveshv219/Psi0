#!/bin/bash
# Fine-tune Psi0's action expert to predict vibe's 64-d pre-FSQ SONIC latents.
#
# Corpus:  scripts/data/vibe/  (see its README) -> data/lerobot/vibe_repose_g1
#          3,819 episodes / 921,278 frames / 5.12 robot-hours at 50 fps.
# Task:    a G1 walks to a cube and flips it so a named colour ends up on top.
#
# Derived from finetune-real-sonic-psi0.sh. Every line that differs from it is a consequence of
# one of four facts, and each is commented at the flag:
#   (a) our action is the 64-d latent, not 78 = motion_token + hand -- this G1 has no hands
#   (b) our data is 50 fps, not 30, so a 1.0 s chunk is 50 steps and not 30
#   (c) our state is 32 = joint_pos(29) + projected gravity(3), not 43
#   (d) the label IS a colour, which rules out hue augmentation
#
# Usage:  ./scripts/train/psi0/finetune-vibe-repose-psi0.sh [exp-name]
#         BATCH=4 ./scripts/train/psi0/finetune-vibe-repose-psi0.sh smoke      # single small GPU
#         SCRATCH=1 ./scripts/train/psi0/finetune-vibe-repose-psi0.sh scratch  # control arm

set -euo pipefail
cd "$(dirname "$0")/../../.."

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ulimit -n 65535

exp=${1:-vibe-repose}
CKPT_ROOT=${PSI_HOME:-/mnt/ssd500/psi_home}/cache/checkpoints/psi0
VLM=$CKPT_ROOT/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
HEADER=$CKPT_ROOT/postpre.1by1.pad36.2601131206.ckpt.he30k

# Control arm. The pretrained header was post-trained on Humanoid-Everyday joint actions, and
# whether that transfers to a *latent* action space is untested by anyone. Two runs differing only
# in this flag answer it; see the README's "action space" note.
if [ "${SCRATCH:-0}" = "1" ]; then
    HEADER_FLAG="--model.pretrained-action-header-path=None"
else
    HEADER_FLAG="--model.pretrained-action-header-path=$HEADER"
fi

echo "Experiment : $exp"
echo "GPUs       : $NPROC_PER_NODE ($CUDA_VISIBLE_DEVICES)"
echo "Action head: ${SCRATCH:-0}" | sed 's/0$/pretrained/;s/1$/SCRATCH (control arm)/'

args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=sonic \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=${BATCH:-16} \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
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
--model.model_name_or_path=$VLM \
$HEADER_FLAG \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.observation-horizon=1 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8 \
"

# ---------------------------------------------------------------------------- #
# The flags that differ from the SONIC preset
# ---------------------------------------------------------------------------- #

# (a) 64, not 78. Psi0's SONIC preset predicts motion_token(64) + hand(14); this G1 has no hands,
#     and `extract_latents.py` writes exactly the 64 pre-FSQ dims.
# (c) 32, not 43. joint_pos(29) + projected gravity(3). joint_vel is deliberately excluded -- see
#     raw_vibe_to_psi_lerobot.py's docstring on why more present proprioception is a hazard here.
args="$args --model.action-dim=64 --model.odim=32"

# (b) 50, not 30. Our rows are 50 fps, so 30 steps is 0.58 s where the preset's 30 @ 30 fps is
#     1.00 s. Matching the DURATION the vendor tuned for means 50 steps.
#
#     `SimpleRepackTransform.action_chunk_size` defaults to 30 and NOTHING in the codebase syncs it
#     to `model.action_chunk_size` -- their configs agree only because both defaults are 30. Setting
#     the model side alone leaves the dataloader emitting 30-step chunks. Set both.
args="$args --model.action-chunk-size=50 --model.action-exec-horizon=50 \
            --data.transform.repack.action_chunk_size=50"

# (d) NO image augmentation. The preset passes --img-aug, whose ColorJitter defaults to hue=0.05 --
#     a +/-18 deg hue rotation. Measured on this corpus's actual palette, red and orange are 24.7 deg
#     apart and yellow's nearest neighbour is 35.3 deg, so a single draw covers 51-73% of the
#     distance between adjacent labels. On a task where the commanded colour IS the label, that is
#     label noise, not augmentation. There is also no domain gap to close: we roll out in vibe and
#     deploy in vibe.
#
#     If a later run needs robustness, re-enable with --data.transform.model.color_jitter.hue=0.0
#     rather than turning the whole thing back on.
args="$args --data.transform.model.no-img-aug"

# `LerobotDataConfig` defaults val_repo_ids to [train_repo_ids[0]] -- i.e. it validates on the
# TRAINING set unless told otherwise, which would make every number below in-sample, drop_lang and
# vs_zoh included. `raw_vibe_to_psi_lerobot.py --val-frac` writes vibe_repose_g1_val: a 15% split
# held out by EPISODE and stratified over (motion clip, colour), so all 75 clips and all 6 colours
# appear on both sides under unseen initial conditions. Passed explicitly above.

find_free_port() {
    port=${1:-29500}
    while ! python -c "import socket,sys; s=socket.socket();
try: s.bind(('0.0.0.0',$port)); s.close()
except OSError: sys.exit(1)" 2>/dev/null; do
        port=$((port + 1))
        [ $port -gt $((${1:-29500} + 1000)) ] && { echo "no free port" >&2; exit 1; }
    done
    echo $port
}

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "$args"
    exit 0
fi

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$(find_free_port 29500)" \
    scripts/train.py ${args}
