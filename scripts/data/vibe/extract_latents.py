"""Run the frozen SONIC tokenizer encoder over collected rollouts -> one latent per frame.

Step 2 of 3. ``collect_rollouts.py`` records what the robot executed; this replays each
trajectory through the encoder as if it were a reference motion and keeps the pre-quantizer
latent ``z``. ``raw_vibe_to_psi_lerobot.py`` then packs `z` as the LeRobot `action` column.

**These 64 dims are Psi0's entire action space.** Psi0's own SONIC preset predicts 78 =
motion_token(64) + hand(14); the G1 here has no hands, so `--model.action-dim=64`. That mismatch
against the released action header is handled — `psi/trainers/sonic.py` reloads only
`transformer_blocks` when `action_dim` or `action_chunk_size` differ, reinitializing the in/out
projections, which is what you want for a new action space anyway.

Runs in the `fcrl` conda env, not `.venv-psi`: it needs mocke and rsl_rl for the encoder weights.

**Pre-FSQ, not the tokens.** ``SonicBaseModel.encode_tokens`` computes
``z = encoder(tokenizer_obs)`` and then ``fsq_quantize(z)``; this takes ``z``. The encoder is not
adapted (``adapt_encoder: False`` in ``vibe/core/rl.py``), so ``z`` depends only on the tokenizer
window -- nothing from the vision path or the adapter leaks into it, which is what makes it a
usable prediction target.

**The anchor is the trap.** ``mocke.sonic.mdp.observations.sonic_g1_tokenizer`` builds each frame
from 10 future reference frames at ``FRAME_SKIP`` spacing, plus a 6D orientation difference
between the LIVE robot pelvis and the reference pelvis at those future frames. Encoding an
executed trajectory means live and reference are the SAME trajectory, so the difference is
``quat_inv(pelvis[t]) * pelvis[t+f]``. Getting this backwards computes the reference against
itself, which is exactly the bug the executed-rollout collection exists to avoid.

The last ``FUTURE_STEPS * FRAME_SKIP`` frames of a trajectory have no real future. Their window is
clamped to the last frame -- the same clamp ``MultiClipMotionCommand.future_frames`` applies at a
clip boundary, so those latents are what the policy actually saw. ``has_future`` marks them so a
consumer can drop them instead of guessing.

Usage:
    P=/home/sarvesh/miniconda3/envs/fcrl/bin/python
    $P scripts/data/vibe/extract_latents.py --rollouts data/vibe_pilot --check   # verify only
    $P scripts/data/vibe/extract_latents.py --rollouts data/vibe_pilot
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

# mjlab before orcs (vibe/CLAUDE.md registration chain); mocke is what we actually need.
import mjlab  # noqa: F401
import mocke
from mocke.mdp.joint_maps import G1_TRACKED_BODY_NAMES
from mocke.sonic.mdp.observations import FRAME_SKIP, FUTURE_STEPS

ANCHOR_BODY = "pelvis"
CKPT = "sonic/last_ported.pt"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", required=True, help="collect_rollouts.py output dir")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64, help="episodes per encoder call")
    ap.add_argument("--overwrite", action="store_true", help="re-encode episodes already done")
    ap.add_argument("--check", action="store_true",
                    help="encode 8 episodes and report, without writing")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #


def load_encoder(device: str) -> torch.nn.Sequential:
    """The frozen tokenizer encoder from the ported SONIC checkpoint.

    Built with **rsl_rl's own** ``_mlp`` and ``SonicBaseModel``'s own default activation, rather
    than by hand. Constructing a ``SonicBaseModel`` outright is still avoided -- that needs a live
    env to infer its dims from an obs TensorDict, and the weights already state every shape -- but
    the layer topology and the nonlinearity now come from the code that will consume these latents.

    **This function hardcoded ``nn.ELU`` until 2026-08-11 and the real encoder is ``nn.SiLU``.**
    The Linear weights were correct, the shapes were correct, ``load_state_dict`` was happy, and
    every latent was silently the wrong function of its input. Measured on ep000000 of
    ``data/rollouts_cf`` (455 frames, same Linear weights loaded into both, same tokenizer windows):
    ``max |z_SiLU - z_ELU| = 3.08`` and mean ``|z|`` 0.174 SiLU against 1.018 ELU, a 5.9x scale
    error. Nothing downstream could have caught it -- a wrong-but-consistent target trains to a
    perfectly respectable MAE. Hence `strict=True` below and no literal activation here: the
    activation must be a property of the model, never of this script.
    """
    from rsl_rl.models.sonic_base_model import SonicBaseModel, _mlp

    act = inspect.signature(SonicBaseModel.__init__).parameters["activation"].default
    sd = torch.load(str(mocke.PRETRAINED_DIR / CKPT), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    idx = sorted({int(k.split(".")[0]) for k in enc_sd})
    shapes = [enc_sd[f"{i}.weight"].shape for i in idx]
    # in_dim from the first layer, out_dim from the last, hidden from everything between. `_mlp`
    # then decides where the activations go, so this cannot disagree with the policy's encoder.
    enc = _mlp(shapes[0][1], tuple(s[0] for s in shapes[:-1]), shapes[-1][0], act)
    # strict: a topology mismatch must fail here rather than load into the wrong slots.
    enc.load_state_dict(enc_sd, strict=True)
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"[latents] encoder {shapes[0][1]} -> {shapes[-1][0]}, {len(idx)} linear layers, {act}")
    return enc


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def tokenizer_obs(joint_pos: torch.Tensor, joint_vel: torch.Tensor,
                  pelvis_quat: torch.Tensor) -> torch.Tensor:
    """(T, 640) SONIC tokenizer input, one row per frame of one trajectory.

    Mirrors mocke.sonic.mdp.observations.sonic_g1_tokenizer:
        cat([jp_future.flat, jv_future.flat]) -> (F, 2*29), then per-frame 6D pelvis-relative
        reference orientation -> (F, 6), concatenated and flattened.
    """
    T = joint_pos.shape[0]
    dev = joint_pos.device
    idx = torch.arange(T, device=dev)[:, None] + torch.arange(FUTURE_STEPS, device=dev) * FRAME_SKIP
    idx = idx.clamp(max=T - 1)                                        # (T, F)

    jp = joint_pos[idx].reshape(T, -1)                                # (T, F*29)
    jv = joint_vel[idx].reshape(T, -1)
    chop = torch.cat([jp, jv], dim=-1).reshape(T, FUTURE_STEPS, -1)   # (T, F, 58)

    # Live pelvis at t against the reference pelvis at the future frames. Same trajectory here --
    # see the module docstring on why this direction matters.
    live = quat_inv(pelvis_quat)[:, None, :].expand(T, FUTURE_STEPS, 4)
    ori = matrix_from_quat(quat_mul(live.reshape(-1, 4), pelvis_quat[idx].reshape(-1, 4)))
    ori = ori[..., :2].reshape(T, FUTURE_STEPS, 6)

    return torch.cat([chop, ori], dim=-1).reshape(T, -1)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    root = Path(args.rollouts)
    summary = json.loads((root / "summary.json").read_text())

    # The trajectories are in the env's MJ joint order and full MJ body order; the tokenizer wants
    # MJ joints (which is what the motion loader produces after its IL2MJ remap) and the pelvis
    # quaternion. Assert the layout rather than trust it -- a silent reorder here would produce
    # latents that look completely reasonable and mean nothing.
    assert summary["joint_order"] == "mj", summary["joint_order"]
    body_order = summary["body_order"]
    anchor = body_order.index(ANCHOR_BODY)
    assert anchor == summary["tracked_body_ids"][G1_TRACKED_BODY_NAMES.index(ANCHOR_BODY)]

    enc = load_encoder(args.device)
    in_dim = enc[0].in_features
    expected = FUTURE_STEPS * (2 * 29 + 6)
    assert in_dim == expected, f"encoder wants {in_dim}, tokenizer builds {expected}"
    print(f"[latents] encoder {in_dim} -> {enc[-1].out_features}, "
          f"future {FUTURE_STEPS} x skip {FRAME_SKIP}, anchor {ANCHOR_BODY} (body {anchor})")

    eps = sorted((root / "episodes").iterdir())
    if args.check:
        eps = eps[:8]
    todo = [d for d in eps if args.overwrite or args.check or not (d / "latents.npz").exists()]
    print(f"[latents] {len(todo)} of {len(eps)} episodes to encode")

    t0, done, n_frames = time.time(), 0, 0
    stats = []
    for d in todo:
        t = np.load(d / "traj.npz")
        jp = torch.from_numpy(t["joint_pos"]).to(args.device)
        jv = torch.from_numpy(t["joint_vel"]).to(args.device)
        pq = torch.from_numpy(t["body_quat_w"][:, anchor]).to(args.device)
        with torch.no_grad():
            z = enc(tokenizer_obs(jp, jv, pq))
        z = z.cpu().numpy().astype(np.float32)

        T = len(z)
        # Frames whose 10-frame future window fits inside the trajectory. Past this the window is
        # clamped to the last frame, exactly as the command clamps at a clip boundary -- real, but
        # increasingly repeated, so a consumer may want to drop them.
        has_future = np.arange(T) + (FUTURE_STEPS - 1) * FRAME_SKIP < T

        if not args.check:
            np.savez_compressed(d / "latents.npz", z=z, has_future=has_future,
                                future_steps=FUTURE_STEPS, frame_skip=FRAME_SKIP)
        done += 1
        n_frames += T
        stats.append((z, has_future))
        if done % 200 == 0:
            print(f"[{done}/{len(todo)}] {n_frames} frames "
                  f"({n_frames / max(time.time() - t0, 1e-9):.0f} frames/s)")

    Z = np.concatenate([z for z, _ in stats])
    hf = np.concatenate([h for _, h in stats])
    d1 = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    print(f"[latents] {done} episodes, {n_frames} frames, {Z.shape[1]}-d")
    print(f"  per-dim std {Z.std(0).min():.3f}-{Z.std(0).max():.3f}   range "
          f"[{Z.min():.2f}, {Z.max():.2f}]   finite {bool(np.isfinite(Z).all())}")
    print(f"  step-to-step |dz| mean {d1.mean():.3f}   frames with a full future "
          f"{hf.mean() * 100:.1f}%")
    if args.check:
        print("[latents] --check: nothing written")
    else:
        meta = {"encoder_ckpt": CKPT, "latent_dim": int(Z.shape[1]),
                "future_steps": FUTURE_STEPS, "frame_skip": FRAME_SKIP,
                "anchor_body": ANCHOR_BODY, "pre_fsq": True, "episodes": done,
                "frames": n_frames}
        (root / "latents_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[latents] -> {root}/episodes/*/latents.npz + latents_meta.json")


if __name__ == "__main__":
    main()
