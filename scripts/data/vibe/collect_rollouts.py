"""Roll out the adapted SONIC policy in mjlab and record what the robot actually did.

Step 1 of 3 for the Psi0 corpus. This writes executed trajectories + sys1-camera video + per-sample
tags; ``extract_latents.py`` (step 2) runs the frozen SONIC encoder over each trajectory to produce
the 64-d latents Psi0 predicts, and ``raw_vibe_to_psi_lerobot.py`` (step 3) packs both into LeRobot
format. The collect/encode split exists because the encoder is a pure function of the trajectory
(``adapt_encoder=False`` in ``vibe/core/rl.py``), so re-encoding never needs a re-run.

**This script does not run in Psi0's venv.** It needs mjlab / orcs / vibe / mocke, which live in the
`fcrl` conda env; Psi0 trains in `.venv-psi`. The npz corpus written here is the interface between
the two, which is why nothing on this side imports `psi`.

    /home/sarvesh/miniconda3/envs/fcrl/bin/python scripts/data/vibe/collect_rollouts.py ...

**Two cameras, and they are not interchangeable.** `head_cam` (112x63) is what the *policy* consumes
and must stay untouched — it is an input to the network being rolled out. `sys1_cam` is the same
mount, same fovy, same 16:9 lens at a higher resolution (``vibe.core.sensors.sys1_cam_cfg``: "sys0's
tensor is a pure downscale of this frame and no FOV argument can differ between the two"), and it is
what Qwen3-VL sees. The default 512x288 is exactly 16:9 — the head cam's own aspect — and 16x9 Qwen
patches at its 32-px effective patch size, so nothing in the chain resizes or crops. Pass
`--data.transform.model.resize.size 288 512 --center_crop.size 288 512` on the Psi0 side to keep
that true; the shipped SONIC preset (240 320) resamples it to 320x256 and 80 tokens.

**Frames go to h264, not npz.** At 512x288 a raw frame is 442 KB, so songen's `frames.npz` layout
would be ~277 GB for a full corpus. One mp4 per episode is a few GB for the same content and is what
LeRobot's video column wants anyway.

It is lossy, and `--crf` alone does not fix that: the 2x2 chroma downsample in `yuv420p` happens
*before* encoding, so crf controls quantization only and crf 0 barely helps. Measured on real render
frames (README): yuv420p puts **0.69% of pixels more than 64 levels off** with p99 54, against
yuv444p's **0.0016%** and p99 9 — at the same file size. Those errors land on saturated high-contrast
boundaries, which on this scene means the cube's face edges, and the face colour is the label. Hence
`--pix-fmt yuv444p` by default. `--verify-video` measures the round trip rather than assuming it.

Task is ``Vibe-Repose-BigCubeFloor-ImgFeat-Ext`` and the env is built in **play mode**, which is
what makes the data usable rather than a coincidence. Verified by building both configs:

    play=False  start_from_zero=False  joint_position_range=(-0.05, 0.05)  13 events
    play=True   start_from_zero=True   joint_position_range=(0.0, 0.0)      3 events

- ``start_from_zero=True`` -> every episode runs a sample from frame 0 to its natural end, and the
  sample is drawn uniformly. Under train it starts mid-sample and picks samples weighted by
  length, so longer front-flips would dominate and every trajectory would be a fragment.
- RSI noise zeroed -> the trajectory starts on the sample, so executed and reference are
  comparable frame by frame.
- The two colour events SURVIVE play (``rand_face_colors``, ``rand_terrain_color``): they are the
  task channel, not the render domain, so ``_play_overrides`` does not subtract them. That is the
  whole reason play mode is usable here — everything else about the domain is gone.

Each env draws its own face->colour permutation ONCE at startup (``mode="startup"``), and each
reset draws a sample uniformly at random. Neither is keyed to the env index, so N envs give N
independent (sample, perm) draws. Do not "improve" this by deriving either from the env id:
``gcd(24 perms, 78 samples) = 6``, so index-derived draws would visit only
``lcm(78, 24) = 312`` of the 1872 combinations and nothing would report it.

Body/joint ORDER is the env's own (MJ), not the on-disk reference layout (IL). Recording the 14
tracked bodies in ``mocke.mdp.joint_maps.G1_TRACKED_BODIES`` order is exactly what
``orcs.core.data.loader`` produces *after* its ``IL2MJ`` / ``_IL_BODY_IDS`` remap, so the latent
pass consumes these files with no conversion. Writing a fake 37-body IL array to imitate the
reference files would mean converting twice and getting one of them wrong.

**num-envs is capped by RAM, not by the GPU.** Frames are buffered per env until the episode ends,
and at 442 KB/frame (plus 21 KB of head cam) a mean 255-frame episode pins ~118 MB per env in flight;
the longest sample, 984 frames, pins 456 MB alone. Against the 12 GB default guard that puts the
ceiling near 100 envs, versus songen's 4096 at 112x63. `main` estimates this at startup and refuses
rather than OOM-ing at hour three, so a full corpus is several step-budgeted passes with
`--ep-offset`, not one run.

Usage:
    P=/home/sarvesh/miniconda3/envs/fcrl/bin/python

    # pilot: enough episodes to measure Qwen token counts and run the grounding probe
    $P scripts/data/vibe/collect_rollouts.py --num-envs 64 --episodes 200 \
        --out data/vibe_pilot --wandb-run-path vbp/repose/011pgzbh --verify-video 3

    # one pass of a full corpus; repeat with --ep-offset to accumulate
    $P scripts/data/vibe/collect_rollouts.py --num-envs 128 --steps auto \
        --out data/vibe_g1 --wandb-run-path vbp/repose/011pgzbh
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

# mjlab BEFORE orcs, always: mjlab runs its entry-point scan as the last line of its own
# __init__, and reaching that scan from inside a half-imported orcs makes every Vibe-* task fail
# to register with only a warning (vibe/CLAUDE.md, registration chain).
import mjlab  # noqa: F401
import orcs  # noqa: F401
import vibe  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mocke.mdp.joint_maps import G1_TRACKED_BODIES

# Verified against `mjlab.tasks.registry.list_tasks()` on 2026-08-13, not inherited: songen's
# `Vibe-Repose-AdaptSonic-ImgFeat-Ext` is no longer registered and this is its replacement. It is
# the `-Ext` variant on purpose — that is the `CrossAttentionExtractor` + LoRA-adapted decoder whose
# `_encode_mlp` seam Psi0's latent replaces. The plain `-ImgFeat`, `-Lfd`, `-Sfd`, `-ObjKin` and
# `-ImgRgb` siblings are different actors, and `SmallCubeTable` is a different scene entirely.
TASK = "Vibe-Repose-BigCubeFloor-ImgFeat-Ext"

# sys1 camera render size. 512x288 is exactly 16:9 — the head cam's own 112x63 aspect, so the same
# lens with no geometric distortion — and both axes are whole multiples of Qwen3-VL's 32-px effective
# patch, so the processor patchifies the render with no resize and no crop. Grid is 16 x 9 = 144
# image tokens.
#
# **The effective patch is 32, not 28.** `patch_size=16` x `merge_size=2`, read off the live
# processor. 28 is the Qwen2-VL convention and is what Psi0's own config still encodes
# (`max_pixels: 576*28*28`) — which is where an earlier 448x252 default came from. That size is
# 14x8 patches with the height silently rescaled 252 -> 256, distorting the aspect to 1.750.
#
# Verified end to end through Psi0's real `ResizeImage` + `CenterCrop` into the processor: at
# `--data.transform.model.resize.size 288 512 --center_crop.size 288 512` the transform is a no-op
# and the grid is 16x9 at 512x288. **Those two flags must be passed** — the shipped SONIC preset
# (240 320) resamples this frame to 320x256 and 80 tokens. Rendering at exactly the config size also
# sidesteps `ResizeImage`'s NEAREST interpolation, which would alias a colour image on any downscale.
CAM_W, CAM_H = 512, 288
QWEN_PATCH = 32

# Bytes of RAM one buffered sys1 frame costs, and the ceiling `main` refuses to exceed. The buffer
# is the binding resource here (see the module docstring), so it is checked at startup against the
# machine rather than assumed from a config that was fine at 112x63.
MAX_BUFFER_GB = 12.0

# Cube edge length in metres. Recorded as a tag but NOT used to count flips -- see `count_flips`.
EDGE = 0.6096

# mjOBJ_GEOM, the MuJoCo object type the segmentation channel reports for a geom hit. Read from
# mujoco at import so it can never fall out of step with the enum.
MJOBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

# Minimum frames the nearest-up face must hold to count as a dwell rather than a flicker. Measured
# rather than chosen: across all 78 reference samples there are 171 face runs and the SHORTEST is 35
# frames, so no reference sample needs debouncing at all and any value from 1 to 15 gives identical
# counts. It is kept for the EXECUTED rollouts, where a cube the policy drops or spins can flicker
# in a way the clean reference motions never do. 5 frames = 0.1 s at 50 Hz.
MIN_FACE_RUN = 5

# Face normals in object frame, in `_FACE_GEOMS` / `_FACE_COLORS` order: +X, -X, +Y, -Y, +Z, -Z.
# Spelled out rather than imported so this module stays importable without vibe -- but it MUST
# match `vibe.repose.repose_cube_env_cfg._FACE_GEOMS`, and `main` asserts that it does.
FACE_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def up_face_dots(quat: np.ndarray) -> np.ndarray:
    """World-z component of each of the 6 face normals -> (T, 6).

    The bottom row of the rotation matrix and its negation: for a unit local axis e,
    dot(R @ e, z_world) is R[2, :] @ e, so three numbers cover all six faces.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    zx = 2.0 * (x * z - w * y)
    zy = 2.0 * (y * z + w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return np.stack([zx, -zx, zy, -zy, zz, -zz], axis=1)


def count_flips(quat: np.ndarray, min_run: int = MIN_FACE_RUN):
    """Number of up-face changes, and the sequence of faces -> (int, list).

    Read from the cube's ORIENTATION -- specifically from which face normal points most nearly up,
    with NO tilt threshold. Two earlier versions were wrong on real samples:

    - Distance travelled. `cube_sideflip/sample42` moves 0.235 m, under half a cube edge, and
      flips once; `cube_frontflip/sample11` moves 1.042 m and also flips once. 9 of the 78 samples
      disagree with any displacement rule.
    - Nearest-up face gated on "settled" (dot > 0.9). This deletes flips where the cube ends up
      HELD at a tilt instead of lying flat: `cube_frontflip/sample21` goes 5 -> 0 -> 5 with the
      middle face at dot 0.82-0.87, and `cube_sideflip/sample40` goes 5 -> 1 held at 0.82 on its
      far edge. Both were counted as zero flips. A flip is a change of which face is up; how flat
      the cube then lies is a different question and must not gate it.

    So: run-length encode the argmax and count the changes. `min_run` only rejects flicker, and the
    reference data contains none (see MIN_FACE_RUN).

    Returns (n_flips, face sequence). n_flips is None for an empty trajectory.
    """
    if len(quat) == 0:
        return None, []
    k = up_face_dots(quat).argmax(1)
    seq: list[int] = []
    run_face, run_len = int(k[0]), 1
    for v in k[1:]:
        v = int(v)
        if v == run_face:
            run_len += 1
            continue
        if run_len >= min_run and (not seq or seq[-1] != run_face):
            seq.append(run_face)
        run_face, run_len = v, 1
    if run_len >= min_run and (not seq or seq[-1] != run_face):
        seq.append(run_face)
    return (len(seq) - 1 if seq else None), seq


# A colour is "visible" above this fraction of the image. Below it the face is a few pixels of
# antialiasing, not something a policy or an encoder could act on.
VISIBLE_MIN_FRAC = 0.01



def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--episodes", type=int, default=None,
                    help="stop after this many episodes. Length-biased: a run that stops on an "
                         "episode COUNT ends before long samples finish even one, so the longest "
                         "sample can be missing entirely. Prefer --steps")
    ap.add_argument("--steps", default=None,
                    help="stop after this many env steps instead, saving every episode that "
                         "terminates. 'auto' = longest sample + tail, the budget at which EVERY "
                         "sample completes at least once")
    ap.add_argument("--wandb-run-path", default="",
                    help='trained checkpoint to roll out; "" = initial agent (frozen base)')
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-len", type=int, default=20,
                    help="drop episodes shorter than this many steps")
    ap.add_argument("--tail-steps", type=int, default=10,
                    help="frozen-reference steps kept after a sample's last frame "
                         "(exceeded_motion.epsilon_steps). The env default of 100 made 46%% of the "
                         "first run a static hold; 0 cuts the tail entirely")
    ap.add_argument("--only-sample", default=None,
                    help="restrict the motion library to one sample, e.g. "
                         "cube_frontflip/sample21 — for topping up a sample the main run missed")
    ap.add_argument("--ep-offset", type=int, default=0,
                    help="first episode number, so a top-up run can write into an existing --out")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip camera frames; trajectories and tags only (~1%% the size)")
    ap.add_argument("--cam-width", type=int, default=CAM_W,
                    help=f"sys1 camera width (default {CAM_W}, see CAM_W)")
    ap.add_argument("--cam-height", type=int, default=CAM_H,
                    help=f"sys1 camera height (default {CAM_H})")
    ap.add_argument("--crf", type=int, default=10,
                    help="h264 quantizer, 23 = ffmpeg default, lower = better. Deploy sees a live "
                         "camera with no codec at all, so this is a train/deploy gap, not just "
                         "disk. NOTE crf 0 is NOT lossless under yuv420p — the chroma downsample "
                         "happens before encoding and crf does not touch it (measured: mean 1.47 "
                         "levels at crf 0 vs 1.81 at crf 10; yuv444p crf 0 gets 0.32). Use "
                         "--pix-fmt yuv444p if you want near-lossless")
    ap.add_argument("--pix-fmt", default="yuv444p", choices=("yuv420p", "yuv444p"),
                    help="yuv420p halves CHROMA resolution and MEASURABLY corrupts saturated edges "
                         "— on real render content 0.69%% of pixels land >64 levels off, p99 54, "
                         "against yuv444p's 0.0016%% and p99 9 at the SAME file size. The cube's "
                         "face boundaries are exactly those edges, and face colour is the label. "
                         "yuv420p only if a decoder demands it; transcode down at conversion "
                         "instead, since chroma never encoded cannot be recovered")
    ap.add_argument("--verify-video", type=int, default=0, metavar="N",
                    help="after writing, decode the first N episodes back and report mean|error| "
                         "against the source frames. Measures what the codec cost instead of "
                         "assuming it. Costs one extra decode per checked episode")
    ap.add_argument("--no-head-cam", action="store_true",
                    help="skip the 112x63 head-cam frames. They are ~0.5 GB for a full corpus and "
                         "are what a songen-vs-Psi0 comparison on a byte-identical corpus would "
                         "need, so they are kept by default")
    ap.add_argument("--max-buffer-gb", type=float, default=MAX_BUFFER_GB,
                    help="refuse to start if the estimated in-flight frame buffer exceeds this")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# per-sample tags
# --------------------------------------------------------------------------- #


def sample_tags(motion_files: list[str]) -> list[dict]:
    """One tag dict per sample, index-aligned with the motion loader's sample index.

    Everything here is a property of the reference sample, so it is computed once from the source
    files rather than measured per rollout. ``sample_id`` needs both parts of the path — sample
    numbers restart in each family, so ``sample12`` alone names two different motions.
    """
    tags = []
    for i, mf in enumerate(motion_files):
        p = Path(mf)
        sample_dir = p.parent
        family = sample_dir.parent.name          # cube_frontflip | cube_sideflip
        m = np.load(mf)
        fps = float(m["fps"][0]) if "fps" in m else 50.0
        root_xy = m["body_pos_w"][:, 0, :2]
        root_path = float(np.linalg.norm(np.diff(root_xy, axis=0), axis=1).sum())
        root_net = float(np.linalg.norm(root_xy[-1] - root_xy[0]))
        n_frames = int(m["joint_pos"].shape[0])

        obj_net, n_flips, face_seq = None, None, []
        op = sample_dir / "object_motion.npz"
        if op.exists():
            od = np.load(op)
            xy = od["obj_pos_w"][:, :2]
            obj_net = float(np.linalg.norm(xy[-1] - xy[0]))
            n_flips, face_seq = count_flips(od["obj_quat_w"])

        tags.append({
            "sample_index": i,
            "sample_id": f"{family}/{sample_dir.name}",
            "family": family,
            "n_frames": n_frames,
            "duration_s": round(n_frames / fps, 3),
            "root_path_m": round(root_path, 4),
            "root_net_m": round(root_net, 4),
            # Threshold, not a measurement: 0.5 m separates the two modes (front-flip median
            # 0.71, side-flip 0.20) but ~12 samples sit near it. Check those against their
            # retargeted_motion.mp4 and override by hand rather than trusting the number.
            "walks": bool(root_net > 0.5),
            "obj_net_m": None if obj_net is None else round(obj_net, 4),
            # From the cube's orientation, not its displacement. `n_flips == 0` means the cube
            # moved without ever going over -- a push, not a repose (sample21, sample40).
            "n_flips": n_flips,
            "up_face_seq": face_seq,
        })
    return tags


# --------------------------------------------------------------------------- #
# per-frame colour visibility
# --------------------------------------------------------------------------- #


def face_pixel_fracs(seg: np.ndarray, face_geom_ids: np.ndarray) -> np.ndarray:
    """Fraction of each frame covered by each of the 6 cube faces -> (B, 6) float32.

    Read from the camera's SEGMENTATION channel, which stores the MuJoCo object id per pixel
    (background is -1). So this is the renderer's own answer to "which face is this pixel", not an
    inference from its colour.

    An RGB classifier was written first and thrown away, because both variants are wrong in a way
    that silently corrupts labels:

    - Nearest-colour with a distance cut misses shaded faces. Rendered cube pixels sit 26-67
      rgb-dist off their nominal rgba (docs/randomize_color.md), so the cut has to be loose, and
      the doc records that a "safe-looking" 60 matched nothing.
    - The brightness-invariant channel-bit test fixes shading and then aliases the floor:
      GROUND_RGBAS' maroon (0.40, 0.12, 0.12) has the same channel pattern as red (0.9, 0.2, 0.2)
      and passes any saturation test, so a maroon floor reads as a red cube face.

    Segmentation has neither failure and costs one render channel. Labels are the product here, so
    the exact answer is worth the channel.

    Faces, not colours: the face->colour map is the per-env perm, so colour fractions are derived
    from these by permutation. Keeping faces means a relabelling never needs a re-render.
    """
    # Channel 0 is the object id, channel 1 the MuJoCo object TYPE: a geom hit is
    # (geom_id, mjOBJ_GEOM) and a flex hit is (flex_id, mjOBJ_FLEX), so ids alone could collide
    # with a flex of the same number. This scene has no flex, but checking the type costs one
    # comparison and means the labels cannot drift if one is ever added.
    ids = seg[..., 0].reshape(seg.shape[0], -1)
    is_geom = seg[..., 1].reshape(seg.shape[0], -1) == MJOBJ_GEOM
    n_px = ids.shape[1]
    out = np.zeros((seg.shape[0], 6), np.float32)
    for f, gid in enumerate(face_geom_ids):
        out[:, f] = ((ids == gid) & is_geom).sum(1) / n_px
    return out


def face_to_color_fracs(face_fracs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """(B, 6) per-face fractions -> (B, 6) per-colour fractions, using each env's own perm.

    ``perm[e, f]`` is the colour index shown on geometric face ``f`` (``rand_face_colors``).
    """
    out = np.zeros_like(face_fracs)
    for e in range(face_fracs.shape[0]):
        np.add.at(out[e], perm[e], face_fracs[e])
    return out


def ground_color_idx(env, palette: np.ndarray) -> np.ndarray:
    """Which GROUND_RGBAS entry each env drew -> (B,) long.

    Read back off the model instead of having ``rand_terrain_color`` report it: the event does not
    stash its pick, and reading is additive where patching a pinned dependency is not.
    """
    from vibe.core.mdp.events import GROUND_RGBAS

    ter = env.scene.terrain
    if ter is None:
        return np.full(env.num_envs, -1, np.int64)
    gid = ter.indexing.geom_ids[0]
    rgba = env.sim.model.geom_rgba[:, gid].cpu().numpy()             # (B, 4)
    pal = np.asarray(GROUND_RGBAS, np.float32)
    return np.linalg.norm(rgba[:, None, :3] - pal[None, :, :3], axis=-1).argmin(1)


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #


def build(task: str, device: str, num_envs: int, run_path: str,
          no_frames: bool = False, tail_steps: int | None = None,
          only_sample: str | None = None, cam_width: int = CAM_W, cam_height: int = CAM_H):
    """Play-mode env + the task's real runner. Mirrors scripts/bench/collect_frames.py's build."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import get_wandb_checkpoint_path
    from vibe.core.sensors import HEAD_CAM_NAME, SYS1_CAM_NAME, attach_sys1_cam

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = num_envs
    if not no_frames:
        # sys1_cam is the VLM's view: same parent body, pos, quat and fovy as the head cam by
        # construction (`sys1_cam_cfg` copies all four), so there is no FOV or extrinsic to keep in
        # sync between the frame Qwen sees and the frame the policy sees.
        #
        # Segmentation rides on THIS camera, not the head cam as in songen. The visibility labels
        # (`commanded_visible_frac`, `visible_colors`) are what the grounding probe scores against,
        # so they have to be measured on the frame the model is actually given — at 112x63 a face
        # near the VISIBLE_MIN_FRAC=0.01 cut is ~70 pixels and aliasing decides it.
        #
        # rgb+segmentation and NOT depth: `sys1_cam_cfg` defaults to ("rgb", "depth") for the
        # planner's use, nothing here consumes depth, and it is a third render channel per env.
        assert any(getattr(s, "name", None) == HEAD_CAM_NAME for s in (env_cfg.scene.sensors or ())), (
            f"{task} has no {HEAD_CAM_NAME} for {SYS1_CAM_NAME} to derive its mount and fovy from; "
            "a task without the camera cannot be collected with frames (use --no-frames).")
        attach_sys1_cam(env_cfg, width=cam_width, height=cam_height,
                        data_types=("rgb", "segmentation"))
        # The head cam keeps rgb only and is otherwise untouched — it is a live INPUT to the policy
        # being rolled out, so adding a channel to it would change what the network is fed.

    # Frozen tail. Past a sample's last frame the reference holds still and the robot just
    # stabilises; `exceeded_motion` truncates after this many such steps. The env ships 100, which
    # is right for PPO (the hold is a skill) and wrong for a motion generator: measured over the
    # first run it was 46% of every episode and 49% on the shortest samples, with the cube moving
    # 6 cm against 76 cm during the motion. Shortening it here rather than trimming afterwards
    # means the episode actually ENDS sooner -- less wall clock, less RAM and less disk, instead of
    # recording frames to throw away.
    if tail_steps is not None:
        term = env_cfg.terminations.get("exceeded_motion")
        assert term is not None, "no `exceeded_motion` termination — --tail-steps has nothing to set"
        term.params = {**(term.params or {}), "epsilon_steps": int(tail_steps)}

    if only_sample:
        # dataset_dir is a list of roots and the loader finds sample folders depth-invariantly, so
        # a single sample folder IS a one-sample library. episode_length_s was computed at cfg
        # build from the FULL set, so it stays long enough for any single sample.
        root = Path(load_env_cfg(task, play=True).commands["motion"].dataset_dir[0]).parent
        one = root / only_sample
        assert (one / "motion.npz").exists(), f"no motion.npz at {one}"
        env_cfg.commands["motion"].dataset_dir = [str(one)]
        print(f"[collect] restricted to one sample: {only_sample}")

    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # The task names its own runner class; VibeOnPolicyRunner is not imported directly so a task
    # that ships a different one still works. `asdict`, not `.to_dict()` -- VibeRunnerCfg is a
    # plain dataclass.
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    if run_path:
        log_root = (Path("logs/rsl_rl") / agent_cfg.experiment_name).resolve()
        resume_path, cached = get_wandb_checkpoint_path(log_root, Path(run_path))
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True,
                    map_location=device)
        print(f"[collect] loaded {resume_path.name} from {run_path} "
              f"({'cached' if cached else 'downloaded'})")
    else:
        print("[collect] no wandb run path — initial agent (frozen base, no adapter training)")
    policy = runner.get_inference_policy(device=device)
    return env, wrapped, policy


def bind_color_first(env, cmd, tags: list[dict], perm: np.ndarray) -> None:
    """Draw the goal COLOUR first, then restrict the clip pool to clips that achieve it.

    **Why the collection has to change at all.** Left alone the env does the opposite: it draws a
    clip, and the goal face falls out of that clip's final object orientation
    (``ReposeMotionCommand._resample_command`` sets
    ``_goal_up_face_idx = up_face_idx(_object_goal_quat)``). The commanded colour is then
    ``perm[goal_face]`` — a *consequence* of the clip, not a request. With the permutation fixed per
    env, one clip therefore carries 24 different colour words over one motion, and the colour cannot
    predict the latent. Measured on the old corpus: holding the clip fixed and changing the commanded
    colour moves the target by 0.0238 mean |Δz| against 0.0219 for pure rollout noise (1.088x), while
    changing the clip moves it 0.1895. The colour explained ~nothing, which is why `drop_lang` decayed
    to -2.2%. No architecture or schedule fixes a signal the data does not contain.

    **The hook.** ``_clip_allowance(env_ids) -> (n, n_clips)`` is called at every reset and
    explicitly never cached (``orcs/core/mdp/commands.py`` module docstring), and a mask is exactly
    what ``_sample_init_frame`` already consumes. So nothing about RSI, the frame draw or the
    permutation changes — only which clips are eligible.

    ``perm`` is face->colour, so the face carrying colour ``c`` is ``argwhere(perm[e] == c)``, never
    ``perm[c]`` (``docs/sys1_planner.md`` §9 lists getting this backwards as a sharp bit).

    Feasibility is not assumed: the 78 clips end on faces with counts [10, 16, 12, 15, 21, 4], so
    every face is reachable and no request is unserviceable. Face 5 draws from only 4 clips, so ~1/6
    of episodes see less motion diversity than the rest — reported by ``main`` as the per-colour
    episode counts rather than hidden.
    """
    n_clips = len(tags)
    final = np.array([t["up_face_seq"][-1] if t["up_face_seq"] else -1 for t in tags])
    missing = [i for i, f in enumerate(final) if f < 0]
    assert not missing, (
        f"{len(missing)} clips have no up-face sequence, e.g. {missing[:3]} — their final face is "
        "unknown so they cannot serve a colour request")
    counts = np.bincount(final, minlength=6)
    assert (counts > 0).all(), (
        f"faces {np.flatnonzero(counts == 0).tolist()} are the final face of no clip; a colour "
        "mapping to one of them could never be requested")
    # (6, n_clips): row f selects the clips whose final up-face is f.
    face_mask = torch.zeros(6, n_clips, device=env.device)
    face_mask[torch.as_tensor(final, device=env.device),
              torch.arange(n_clips, device=env.device)] = 1.0
    perm_t = torch.as_tensor(perm, device=env.device)
    wanted = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _clip_allowance(env_ids: torch.Tensor) -> torch.Tensor:
        colors = torch.randint(6, (len(env_ids),), device=env.device)
        wanted[env_ids] = colors
        # face carrying `colors` under this env's permutation
        goal_face = (perm_t[env_ids] == colors[:, None]).float().argmax(1)
        return face_mask[goal_face]

    cmd._clip_allowance = _clip_allowance
    cmd._songen_wanted_color = wanted
    print(f"[collect] colour-first draw installed; clips per final face {counts.tolist()}")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) if args.out else Path("data/rollouts") / time.strftime("%Y%m%d_%H%M%S")
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    target = args.episodes if args.episodes is not None else args.num_envs

    env, wrapped, policy = build(args.task, device, args.num_envs, args.wandb_run_path,
                                 args.no_frames, args.tail_steps, args.only_sample,
                                 args.cam_width, args.cam_height)

    from vibe.repose.mdp.cube_faces import face_color_names, face_colors

    color_names = list(face_color_names())
    palette = face_colors(torch.device("cpu")).numpy()               # (6, 4) rgba
    cmd = env.command_manager.get_term("motion")
    robot, obj = env.scene["robot"], env.scene["object"]
    head_cam = env.scene.sensors["head_cam"]
    cam = None if args.no_frames else env.scene.sensors["sys1_cam"]
    save_head = not (args.no_frames or args.no_head_cam)
    fps = round(1.0 / env.step_dt, 3)

    # The patch grid Qwen3-VL will see. Reported rather than assumed: a non-integer grid means the
    # processor resizes or pads, which silently breaks the "no resample anywhere in the chain"
    # property the 512x288 default was chosen for (see CAM_W).
    if not args.no_frames:
        gw, gh = args.cam_width / QWEN_PATCH, args.cam_height / QWEN_PATCH
        exact = gw.is_integer() and gh.is_integer()
        print(f"[collect] sys1_cam {args.cam_width}x{args.cam_height} -> Qwen patch grid "
              f"{gw:g} x {gh:g} = {gw * gh:g} image tokens "
              f"({'exact' if exact else 'NOT INTEGER — the processor will resize or pad'})")

    # MuJoCo order, every body: this is an mjlab env, so the env's own layout is the format, and
    # recording all of them means the latent pass selects what it needs BY NAME rather than
    # inheriting a subset chosen here. `tracked_body_ids` records where the 14 bodies the SONIC
    # tokenizer reads (mocke's G1_TRACKED_BODIES) sit in that order, so no name lookup is needed
    # downstream either.
    body_names = list(robot.body_names)
    body_ids = list(range(len(body_names)))
    tracked = [n for n, _ in G1_TRACKED_BODIES]
    missing = [n for n in tracked if n not in body_names]
    assert not missing, f"tracked bodies absent from the env: {missing}"
    tracked_body_ids = [body_names.index(n) for n in tracked]

    # The 6 face geoms, in _FACE_GEOMS order — the same ids rand_face_colors writes colours to, so
    # face f here is face f there and the perm applies without a second convention.
    geom_names = list(obj.geom_names)
    face_gids = obj.indexing.geom_ids[
        [geom_names.index(f"cube_face_{i}") for i in range(6)]].cpu().numpy()

    # FACE_AXES must match the env's own face geometry or every up-face label is wrong.
    #
    # songen read the module constant `vibe.repose.repose_cube_env_cfg._FACE_GEOMS`, which no longer
    # exists: the shell poses are now built by `vibe.assets.repose.face_geoms(half_extent)`, the same
    # function `colored_cube_spec` calls to place the `cube_face_{i}` geoms. Same canonical order,
    # same six axes — so this stays a check against the geometry the env actually builds rather than
    # against a second copy of it. The half extent passed here is arbitrary because only the
    # DIRECTION is compared; EDGE/2 is used so the call reads like the cube it describes.
    from vibe.assets.repose import face_geoms
    for i, (pos, _sz) in enumerate(face_geoms(EDGE / 2)):
        v = np.asarray(pos, np.float64)
        v = v / np.linalg.norm(v)
        assert np.allclose(v, FACE_AXES[i], atol=1e-6), (
            f"FACE_AXES[{i}]={FACE_AXES[i]} does not match _FACE_GEOMS[{i}] axis {v.tolist()} — "
            "the cube geometry changed and the flip/up-face labels are stale")

    perm = getattr(env, "_face_color_perm", None)
    assert perm is not None, (
        "no _face_color_perm — rand_face_colors did not run. Without it every env shows the same "
        "colouring and the commanded colour carries no information.")
    perm = perm.cpu().numpy()
    tags = sample_tags(cmd.motion.motion_files)
    grounds = ground_color_idx(env, palette)

    # The frame buffer, not the GPU, is what caps --num-envs here (module docstring): frames are
    # held per env until the episode ends, and at 512x288 one is 442 KB against the head cam's 21.
    # Estimated from the mean reference length and checked, because the failure mode otherwise is
    # an OOM hours into a run rather than a message at t=0. Only a mean — the 984-frame sample pins
    # 3.9x its share while it plays, so leave headroom rather than tuning to this number.
    if not args.no_frames:
        mean_len = float(np.mean([t["n_frames"] for t in tags]))
        per_frame = args.cam_width * args.cam_height * 3 + (112 * 63 * 3 if save_head else 0)
        est_gb = args.num_envs * mean_len * per_frame / 1e9
        print(f"[collect] frame buffer estimate {est_gb:.1f} GB "
              f"({args.num_envs} envs x {mean_len:.0f} mean frames x {per_frame / 1e3:.0f} KB)")
        assert est_gb <= args.max_buffer_gb, (
            f"estimated frame buffer {est_gb:.1f} GB exceeds --max-buffer-gb {args.max_buffer_gb}. "
            f"Lower --num-envs to about {int(args.num_envs * args.max_buffer_gb / est_gb)}, and "
            "collect the corpus in several --steps passes with --ep-offset.")

    # COLOUR FIRST. Installed before any episode is recorded, then all envs are reset so no
    # recorded episode predates it.
    bind_color_first(env, cmd, tags, perm)
    obs0 = wrapped.reset()
    del obs0

    hist = np.bincount(cmd.goal_color_idx.cpu().numpy(), minlength=6)
    print(f"[collect] task={args.task} envs={env.num_envs} fps={fps} "
          f"cam={'none' if cam is None else f'{cam.cfg.width}x{cam.cfg.height}'} "
          f"head_cam={'saved' if save_head else 'not saved'} samples={len(tags)}")
    print("[collect] commanded colour per env: "
          + ", ".join(f"{color_names[c]}={n}" for c, n in enumerate(hist)))
    print(f"[collect] perms in play: {len(np.unique(perm, axis=0))}/24  "
          f"grounds: {len(np.unique(grounds))}/16")
    with open(out / "sample_tags.json", "w") as fh:
        json.dump(tags, fh, indent=2)

    B = env.num_envs
    bufs: list[dict | None] = [None] * B
    saved = 0
    seen_samples: dict[int, int] = {}
    seen_colors = np.zeros(6, np.int64)
    stale = 0                      # buffers dropped because a reset was missed

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    origins = env.scene.env_origins.cpu().numpy()   # fixed for the run

    # Stopping rule. Episode-count is length-biased: every env runs episodes serially, so a long
    # sample occupies its env for its whole duration and completes fewer times. The run then ends
    # after ~episodes/envs * mean_episode steps, and any sample longer than THAT never finishes
    # once -- cube_frontflip/sample21 (984 frames) produced 0 of 4096 episodes for exactly this
    # reason. A step budget of (longest sample + tail) removes the bias by construction: every env
    # holding any sample has time to finish it.
    step_budget = None
    if args.steps is not None:
        if str(args.steps) == "auto":
            step_budget = max(t["n_frames"] for t in tags) + int(args.tail_steps) + 1
        else:
            step_budget = int(args.steps)
        longest = max(t["n_frames"] for t in tags) + int(args.tail_steps)
        verdict = ("covers every sample" if step_budget >= longest
                   else "TOO SHORT — the longest samples will never complete")
        print(f"[collect] step budget {step_budget}; longest sample needs {longest} ({verdict})")

    t0 = time.time()
    steps_run = 0
    while (steps_run < step_budget) if step_budget else (saved < target):
        # Read state BEFORE stepping, and open a buffer only for envs that have none. The sample
        # id has to be read here rather than at episode end: a reset overwrites _clip_ids, so by
        # the time `dones` reports the episode the id belongs to the NEXT one.
        # The colour-first draw is only real if the goal the env derived from the chosen clip is the
        # colour that was requested. The mask targets `up_face_seq[-1]` (from sample_tags, off
        # object_motion.npz) while the env sets the goal from `up_face_idx(_object_goal_quat)`; if
        # those two disagree by even one face, every commanded_color label is silently wrong. A (B,)
        # compare per step is free, and this must fail loudly rather than produce a plausible corpus.
        bad = torch.nonzero(cmd.goal_color_idx != cmd._songen_wanted_color).flatten()
        assert len(bad) == 0, (
            f"{len(bad)} envs got a goal colour other than the one drawn, e.g. env {int(bad[0])}: "
            f"wanted {int(cmd._songen_wanted_color[bad[0]])}, "
            f"got {int(cmd.goal_color_idx[bad[0]])}. `up_face_seq[-1]` and "
            "`up_face_idx(_object_goal_quat)` disagree — fix before collecting.")

        sample_ids = cmd._clip_ids.cpu().numpy()
        # env.episode_length_buf counts steps taken in the CURRENT episode and is zeroed on reset
        # (manager_based_rl_env.py:446 increments, :617 zeroes). Before appending, an open buffer
        # must hold exactly that many frames. If it does not, the env reset without the loop
        # seeing a `done` and the buffer now spans two episodes -- the commanded colour, the
        # sample id and the trajectory would all be a mix of both. Checked rather than assumed:
        # `dones = terminated | truncated` covers every reset today, but a silent mismatch here
        # produces a plausible-looking episode that nothing downstream could detect.
        ep_len = env.episode_length_buf.cpu().numpy()
        jp = robot.data.joint_pos.cpu().numpy()
        jv = robot.data.joint_vel.cpu().numpy()
        bp = robot.data.body_link_pos_w[:, body_ids].cpu().numpy()
        bq = robot.data.body_link_quat_w[:, body_ids].cpu().numpy()
        blv = robot.data.body_link_lin_vel_w[:, body_ids].cpu().numpy()
        bav = robot.data.body_link_ang_vel_w[:, body_ids].cpu().numpy()
        op = obj.data.root_link_pos_w.cpu().numpy()
        oq = obj.data.root_link_quat_w.cpu().numpy()
        olv = obj.data.root_link_lin_vel_w.cpu().numpy()
        oav = obj.data.root_link_ang_vel_w.cpu().numpy()
        cur_color = cmd.current_color_idx.cpu().numpy()

        frames, head_frames, face_frac = None, None, None
        if not args.no_frames:
            rgb = cam.data.rgb.cpu().numpy()
            frames = rgb if rgb.dtype == np.uint8 else (
                rgb * 255.0).clip(0, 255).astype(np.uint8)
            # Off sys1_cam, so the visibility labels are measured on the frame the VLM is given
            # rather than on a 16x smaller one (see `build`).
            face_frac = face_pixel_fracs(cam.data.segmentation.cpu().numpy(), face_gids)
        if save_head:
            hrgb = head_cam.data.rgb.cpu().numpy()
            head_frames = hrgb if hrgb.dtype == np.uint8 else (
                hrgb * 255.0).clip(0, 255).astype(np.uint8)

        for e in range(B):
            if bufs[e] is not None and len(bufs[e]["steps"]) != int(ep_len[e]):
                # Unseen reset. Drop the buffer rather than write a two-episode trajectory.
                stale += 1
                bufs[e] = None
            if bufs[e] is None:
                bufs[e] = {
                    "sample_index": int(sample_ids[e]),
                    "goal_color": int(cmd.goal_color_idx[e]),
                    "steps": [],
                }
            b = bufs[e]
            # .copy() on EVERY slice, not for tidiness. `arr[e]` is a numpy VIEW into the full
            # (B, ...) per-step array, so one buffered slice keeps all B envs' data for that step
            # alive. At 2048 envs the frame array is 41 MB per step, and a single env playing the
            # 984-frame sample would pin 984 x 41 MB = 39.7 GB while every other env finished and
            # restarted. Copying makes the buffer hold only its own env's bytes, which is the
            # ~7.6 GB the memory estimate assumes.
            b["steps"].append((
                jp[e].copy(), jv[e].copy(), bp[e].copy(), bq[e].copy(),
                blv[e].copy(), bav[e].copy(),
                (op[e] - origins[e]).astype(np.float32), oq[e].copy(),
                olv[e].copy(), oav[e].copy(),
                int(cur_color[e]),
                None if frames is None else frames[e].copy(),
                None if face_frac is None else face_frac[e].copy(),
                None if head_frames is None else head_frames[e].copy(),
            ))

        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        steps_run += 1

        for e in torch.nonzero(dones, as_tuple=False).flatten().tolist():
            b, bufs[e] = bufs[e], None
            if b is None:
                continue
            # Drop the first frame. Both colour events are mode="startup", so they fire once at
            # env build and do NOT restage per reset -- the per-episode colour staleness
            # docs/randomize_color.md warns about applies to reset-mode DR, not to these. What the
            # drop does buy: the very first render of the RUN can predate the startup write, and
            # one frame out of ~255 is a cheap way to not care. Recorded as
            # `dropped_first_frame` so a consumer can see it rather than infer it.
            steps = b["steps"][1:]
            if len(steps) < args.min_len:
                continue
            n = write_episode(out, args.ep_offset + saved, b, steps, tags, perm[e], int(grounds[e]),
                              color_names, fps, args)
            seen_samples[b["sample_index"]] = seen_samples.get(b["sample_index"], 0) + 1
            seen_colors[b["goal_color"]] += 1
            saved += 1
            if saved % 200 == 0:
                rate = saved / max(time.time() - t0, 1e-9)
                prog = (f"step {steps_run}/{step_budget}" if step_budget else f"{saved}/{target}")
                print(f"[{prog}] saved {saved}: {tags[b['sample_index']]['sample_id']} "
                      f"T={n} cmd={color_names[b['goal_color']]} ({rate:.1f} ep/s)")
            if step_budget is None and saved >= target:
                break

    # Coverage is reported, not assumed. Random draws over 78 samples x 24 perms will leave holes
    # at these counts; the numbers say how big, and a degenerate draw (one sample, one colour) is
    # visible here rather than three steps downstream in a training curve.
    summary = {
        "task": args.task,
        "num_envs": B,
        "episodes": saved,
        "fps": fps,
        "wandb_run_path": args.wandb_run_path,
        "frames_saved": not args.no_frames,
        # Everything the VLM cache and the LeRobot conversion need to know about the video, carried
        # by the corpus rather than re-derived from a flag someone remembers passing. `cam_source`
        # names WHICH camera: head_cam and sys1_cam share a mount and a fovy but not a resolution,
        # and a corpus that does not say which one it recorded is unusable.
        "cam_source": None if args.no_frames else "sys1_cam",
        "cam_width": None if args.no_frames else args.cam_width,
        "cam_height": None if args.no_frames else args.cam_height,
        "head_cam_saved": save_head,
        "video_codec": None if args.no_frames else "h264",
        "video_crf": None if args.no_frames else args.crf,
        "video_pix_fmt": None if args.no_frames else args.pix_fmt,
        # Qwen3-VL patchifies at a 32-px effective patch (patch_size 16 x merge_size 2). Recorded so the cache builder can assert
        # the grid it computes matches the grid this corpus was rendered for.
        "qwen_patch": QWEN_PATCH,
        "qwen_patch_grid": (None if args.no_frames else
                            [args.cam_width / QWEN_PATCH, args.cam_height / QWEN_PATCH]),
        "samples_seen": len(seen_samples),
        "samples_total": len(tags),
        "episodes_per_sample": {tags[i]["sample_id"]: n for i, n in sorted(seen_samples.items())},
        "perms_seen": int(len(np.unique(perm, axis=0))),
        # The cube's palette AT COLLECTION TIME, in face order, so the corpus carries it instead of
        # a constant in songen having to agree with one in vibe. It has already gone wrong once:
        # vibe recoloured faces 1/3/5 from cyan/magenta/yellow to orange/yellow/pink, and songen
        # went on prompting "put the magenta face on top" at a cube with no magenta on it -- silent,
        # because both sides were internally consistent. Read by SongenWindowedDataset and asserted
        # against the live env in songen/deploy/planner.py.
        "face_color_names": list(color_names),
        "face_colors": [[round(float(x), 4) for x in c[:3]]
                        for c in face_colors(torch.device("cpu")).tolist()],
        "commanded_color_counts": {color_names[c]: int(n) for c, n in enumerate(seen_colors)},
        # Must be 0. Anything else means resets were seen late and episodes were being mixed.
        "buffers_dropped_unseen_reset": stale,
        "body_order": body_names,
        "tracked_body_ids": tracked_body_ids,
        "joint_order": "mj",
        "frame_layout": "mujoco (env-native): all bodies in model index order, joints in MJ order",
        "min_face_run": MIN_FACE_RUN,
        "tail_steps": args.tail_steps,
        "only_sample": args.only_sample,
        "ep_offset": args.ep_offset,
        "steps_run": steps_run,
        "stop_rule": ("steps" if step_budget else "episodes"),
        "step_budget": step_budget,
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[collect] {saved} episodes -> {out}")
    if stale:
        print(f"[collect] WARNING: {stale} buffers dropped on an unseen reset — `dones` is not "
              "reporting every reset, so episode boundaries cannot be trusted. Investigate "
              "before using this data.")
    print(f"[collect] samples seen {len(seen_samples)}/{len(tags)}  "
          f"colours " + ", ".join(f"{color_names[c]}={n}" for c, n in enumerate(seen_colors)))
    env.close()


def write_video(path: Path, frames: np.ndarray, fps: float, crf: int, pix_fmt: str) -> None:
    """(T, H, W, 3) uint8 -> h264 mp4 at `path`.

    ``macro_block_size=1`` is not a tuning knob. imageio-ffmpeg defaults it to 16 and **silently
    resizes** any frame whose dimensions are not a multiple of 16 — 448 is 28x16, but 252 is 15.75,
    so the default would write 448x256 and every downstream "no resize anywhere in the chain" claim
    would be false with nothing reporting it. h264 still needs even dimensions under yuv420p; both
    defaults are even, and an odd request fails here rather than in ffmpeg's stderr.

    ``quality=None`` because imageio's `quality` emits `-qscale`, which ffmpeg ignores when `-crf`
    is also present — passing both silently picks one.
    """
    h, w = frames.shape[1:3]
    assert h % 2 == 0 and w % 2 == 0, f"h264 needs even dimensions, got {w}x{h}"
    with imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=float(fps), codec="libx264",
        pixelformat=pix_fmt, ffmpeg_params=["-crf", str(int(crf))],
        quality=None, macro_block_size=1, ffmpeg_log_level="error",
    ) as w_:
        for f in frames:
            w_.append_data(f)


def verify_video(path: Path, frames: np.ndarray) -> dict:
    """Decode `path` back and compare against the frames that were written -> stats dict.

    The codec is the only lossy step between the renderer and what the VLM is cached from, and at
    deploy there is no codec at all — a live camera feeds the model directly. So this is a
    train/deploy gap, and it is worth a number rather than an assumption. Reported in uint8 levels
    (0-255), per channel and at the worst pixel.
    """
    back = np.stack(imageio.mimread(str(path), memtest=False))
    assert back.shape == frames.shape, f"decoded {back.shape}, wrote {frames.shape}"
    err = np.abs(back.astype(np.int16) - frames.astype(np.int16))
    # `frac_gt64` is the statistic the pix_fmt choice actually turns on, and the mean hides it: 4:2:0
    # is accurate over flat regions and wrong at saturated boundaries, so it lands a small fraction
    # of pixels enormously off while the mean stays near 2 levels. On this scene those boundaries
    # ARE the cube's face edges and the face colour is the label, so a tail statistic is the honest
    # one. Measured on real sys1 frames: 4:2:0 ~0.7% of pixels >64 levels off, 4:4:4 ~0.002%.
    return {
        "frames": int(len(frames)),
        "mean_abs_err": round(float(err.mean()), 4),
        "p99_abs_err": round(float(np.percentile(err, 99)), 2),
        "p9999_abs_err": round(float(np.percentile(err, 99.99)), 2),
        "max_abs_err": int(err.max()),
        "frac_gt64": round(float((err > 64).mean()), 8),
        "bytes": int(path.stat().st_size),
        "bytes_per_frame": int(path.stat().st_size / max(len(frames), 1)),
    }


def write_episode(out: Path, idx: int, b: dict, steps: list, tags: list[dict],
                  perm: np.ndarray, ground: int, color_names: list[str],
                  fps: float, args) -> int:
    """One episode -> traj.npz (+ frames.npz) + meta.json under episodes/ep<idx>/."""
    d = out / "episodes" / f"ep{idx:06d}"
    d.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*steps))

    def st(i):
        return np.stack(cols[i])

    np.savez_compressed(
        d / "traj.npz",
        fps=np.array([fps]),
        joint_pos=st(0), joint_vel=st(1),
        body_pos_w=st(2), body_quat_w=st(3),
        body_lin_vel_w=st(4), body_ang_vel_w=st(5),
        obj_pos_w=st(6), obj_quat_w=st(7),
        obj_lin_vel_w=st(8), obj_ang_vel_w=st(9),
    )
    T = len(steps)
    cur_color = np.array(cols[10], np.int64)
    obj_quat = st(7)

    # The reference tags say what the sample DOES; these say what the robot actually achieved.
    # Both are kept because they disagree whenever the policy fails, and a dataset that carries
    # only the reference's flip count silently labels failed rollouts as successes.
    exec_flips, exec_seq = count_flips(obj_quat)
    up_dot = up_face_dots(obj_quat)
    ref = tags[b["sample_index"]]

    meta = {
        "episode": idx,
        "n_frames": T,
        "fps": fps,
        # reference-sample tags, prefixed so they can never be mistaken for measurements
        **{f"ref_{k}" if k not in ("sample_index", "sample_id", "family") else k: v
           for k, v in ref.items()},
        "face_perm": perm.tolist(),
        "commanded_color": color_names[b["goal_color"]],
        "commanded_color_idx": b["goal_color"],
        "ground_color_idx": ground,
        # what the executed rollout did
        "exec_n_flips": exec_flips,
        "exec_up_face_seq": exec_seq,
        "exec_matches_ref_flips": exec_flips == ref.get("n_flips"),
        "up_face_idx": up_dot.argmax(1).tolist(),
        # How flat the cube lies, reported and NOT used to define a flip: a face can be up
        # while the cube is held on an edge at dot 0.82 (sample40).
        "up_face_flat": up_dot.max(1).round(4).tolist(),
        "cur_color_idx": cur_color.tolist(),
        # Did the commanded colour end up on top? Read off the FINAL settled face, not the last
        # frame: the cube can be mid-tumble when the episode ends.
        "final_up_color_idx": (int(perm[exec_seq[-1]]) if exec_seq else None),
        "success": bool(exec_seq and int(perm[exec_seq[-1]]) == b["goal_color"]),
        "dropped_first_frame": True,
    }

    if not args.no_frames:
        frames = np.stack(cols[11])
        face_frac = np.stack(cols[12])                       # (T, 6) per geometric face
        color_frac = face_to_color_fracs(face_frac, np.tile(perm, (len(face_frac), 1)))
        write_video(d / "frames.mp4", frames, fps, args.crf, args.pix_fmt)
        if args.verify_video and idx - args.ep_offset < args.verify_video:
            v = verify_video(d / "frames.mp4", frames)
            print(f"[verify] ep{idx:06d} {v['frames']}f  mean|err| {v['mean_abs_err']:.3f} "
                  f"p99 {v['p99_abs_err']:.1f} p99.99 {v['p9999_abs_err']:.1f} "
                  f"max {v['max_abs_err']} levels  >64: {v['frac_gt64'] * 100:.4f}%  "
                  f"{v['bytes_per_frame'] / 1e3:.1f} KB/frame "
                  f"({args.cam_width * args.cam_height * 3 / v['bytes_per_frame']:.0f}x vs raw)")
            (d / "video_check.json").write_text(json.dumps(v, indent=2))
        if cols[13][0] is not None:
            # The head cam stays npz: 112x63 is 21 KB/frame raw, ~0.5 GB for a full corpus, and
            # keeping it BYTE-EXACT is the point — it is the policy's own input and the only way a
            # songen-vs-Psi0 comparison runs on one corpus rather than two collections.
            np.savez_compressed(d / "head_frames.npz", frames=np.stack(cols[13]))
        cmd_vis = color_frac[:, b["goal_color"]]
        n_vis = (face_frac > VISIBLE_MIN_FRAC).sum(1)
        meta |= {
            "visible_faces": face_frac.round(5).tolist(),
            "visible_colors": color_frac.round(5).tolist(),
            "n_visible_faces": n_vis.tolist(),
            "commanded_visible_frac": cmd_vis.round(5).tolist(),
            # Fraction of frames where the commanded colour is on screen at all. Low means the
            # commanded face spent the episode out of view, which makes those frames ambiguous
            # supervision rather than hard examples -- the model cannot see what it is asked about.
            "commanded_visible_rate": float((cmd_vis > VISIBLE_MIN_FRAC).mean()),
            # Two or more faces are needed to read the cube's orientation from one image. With one,
            # several orientations look identical and the correct motion is not determined.
            "orientation_readable_rate": float((n_vis >= 2).mean()),
        }

    with open(d / "meta.json", "w") as fh:
        json.dump(meta, fh)
    return T


if __name__ == "__main__":
    main()
