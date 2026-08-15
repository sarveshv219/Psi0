"""Pack collected vibe rollouts into the LeRobot dataset Psi0 trains on.

Step 3 of 3. ``collect_rollouts.py`` wrote what the robot executed, ``extract_latents.py`` wrote the
64-d pre-FSQ SONIC latent per frame; this turns those episode directories into the same on-disk
schema ``raw_sonic_to_psi_lerobot.py`` produces, so ``scripts/train.py`` consumes it unmodified.

Runs in **`.venv-psi`**, not `fcrl`: it needs `datasets`/`pandas` and it probes `torchcodec`. It
imports no sim -- the npz corpus on disk is the boundary between the two environments.

Usage:
    source .venv-psi/bin/activate
    python scripts/data/vibe/raw_vibe_to_psi_lerobot.py \
        --rollouts data/vibe_g1 --out data/lerobot/vibe_repose_g1

--------------------------------------------------------------------------------------------------
WHAT GOES IN EACH COLUMN, AND WHY
--------------------------------------------------------------------------------------------------

``action`` = the 64-d pre-FSQ latent, nothing else. Psi0's shipped SONIC preset predicts
78 = motion_token(64) + hand(14); this G1 has no hands. Train with ``--model.action-dim=64``.

``states`` = joint_pos(29) + projected gravity(3) = 32. Train with ``--model.odim=32``.

    **joint_vel is deliberately excluded.** It is available in `traj.npz` and it is *informative* --
    the latent is a function of the future joint trajectory, so present velocity predicts a chunk of
    it. That is exactly the problem. songen's diagnosis was that its target was nearly determined by
    the present: `vs_zoh` (the model against a zero-order hold on the previous latent) stuck at 0.89
    through run6, and `drop_lang` -- the val-MAE penalty for blanking the language input -- stayed
    NEGATIVE, i.e. the command was worse than useless. Every extra channel of present proprioception
    widens that shortcut. joint_pos and gravity are the minimum needed to make the target
    well-posed at all (the latent is pose-relative), so that is what goes in.

    This is a design choice reasoned from a measurement on a different model, not itself a measured
    result. The controlled experiment is two fine-tunes identical except for the state vector,
    compared on `drop_lang`.

``task`` -> a 6-sentence vocabulary in which exactly ONE token varies:

    "Flip the cube so the {red|orange|green|yellow|blue|pink} face is up."

    Generated here from ``meta.json``, never written at collection time, so rephrasing the command
    is a re-run of this script and not a re-collection. `SimpleRepackTransform` lowercases it.
    One varying token means the language channel carries exactly log2(6) bits and nothing else --
    so if the fine-tune ignores the command, that shows up cleanly instead of hiding behind
    correlated phrasing.

--------------------------------------------------------------------------------------------------
THREE FILTERING / ALIGNMENT DECISIONS
--------------------------------------------------------------------------------------------------

1. **Failed episodes are dropped** (``meta["success"]``, 94.7% of 4032 on `data/vibe_g1`). In a
   failed episode the commanded colour did NOT end up on top, so the pair (command, motion) is
   mislabeled: keeping it trains the model that the command does not constrain the motion. Costs
   ~5% of the corpus. ``--keep-failures`` exists only to measure that claim, not to use.

2. **Frames with a clamped future window are KEPT.** `extract_latents.py` marks `has_future=False`
   for the last ``(FUTURE_STEPS-1)*FRAME_SKIP = 45`` frames, whose tokenizer window is clamped to
   the final frame. Those latents are not fabricated -- `MultiClipMotionCommand.future_frames`
   applies the identical clamp at a clip boundary, so they are what the policy actually consumed.
   Dropping them would remove the last 45 frames of *every* episode, which on this task is
   precisely where the flip completes and the commanded colour arrives on top. The corpus would
   contain approaches and no outcomes. The count is recorded per episode in `episodes.jsonl` so a
   consumer can still exclude them.

3. **fps is 50, and `action_chunk_size=30` therefore covers 0.6 s, not the 1.0 s of Psi0's 30 fps
   preset.** Rows stay dense at 50 Hz because `lerobot_patch` sets `tolerance_s=1e-4` and rejects
   gaps. Fix the horizon at the model, not in the data: **train with
   ``--model.action-chunk-size=50``** for a 1.0 s chunk. That flag is free here -- `sonic.py`
   already reinitializes the in/out projections because `action_dim` differs from 78, so a changed
   chunk size costs no additional pretrained weight.

--------------------------------------------------------------------------------------------------
VIDEO
--------------------------------------------------------------------------------------------------

Copied at the rendered 512x288, not downscaled: the model transform resizes to 320x240 at load
time, and keeping the render as the source of truth means re-caching at a different resolution
never needs a re-collection.

The source is **yuv444p**, chosen because 4:2:0 chroma subsampling put 0.694% of pixels more than
64 uint8 levels off at saturated colour boundaries -- which on this scene are the cube's face
edges, and the face colour is the label. `verify_video` decoded that with imageio-ffmpeg; training
decodes with **torchcodec**, which is a different decoder on the same codec and was never tested
against 4:4:4. So this script decodes one episode with torchcodec before it converts anything, and
transcodes the whole corpus to yuv420p only if that probe fails. The result is recorded in
`info.json`. It is a measurement, not a flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from datasets.utils.logging import set_verbosity_error
from tqdm import tqdm

set_verbosity_error()
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

CODE_VERSION = "v2.1"
VIDEO_KEY = "observation.images.egocentric"
ANCHOR_BODY = "pelvis"

# Ordered to match `summary["face_color_names"]`; asserted against it at load, never trusted.
COLORS = ["red", "orange", "green", "yellow", "blue", "pink"]


def command(color: str) -> str:
    return f"Flip the cube so the {color} face is up."


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


# --------------------------------------------------------------------------- #
# state vector
# --------------------------------------------------------------------------- #


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T, 3) world gravity expressed in the pelvis frame -- the IsaacLab/mjlab convention.

    ``quat_rotate_inverse(root_quat_w, (0, 0, -1))``. With R body->world, that is -R[2, :], the
    third ROW of the rotation matrix negated. Writing the row out directly avoids materializing
    3x3 matrices for a million frames.
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return -np.stack([2 * (x * z - w * y),
                      2 * (y * z + w * x),
                      1 - 2 * (x * x + y * y)], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #


def ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def transcode_420(src: Path, dst: Path, crf: int) -> None:
    subprocess.run(
        [ffmpeg_bin(), "-loglevel", "error", "-y", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(dst)],
        check=True,
    )


def probe_torchcodec(video: Path) -> Tuple[bool, str]:
    """Can the decoder training actually uses read this file? Returns (ok, message)."""
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise SystemExit(
            "torchcodec is not importable -- run this in .venv-psi, not fcrl. It is what "
            f"training decodes video with, and whether it reads yuv444p is the open question "
            f"this script settles before writing anything. ({e})"
        )
    try:
        dec = VideoDecoder(str(video))
        frame = dec[0]
        n = dec.metadata.num_frames
        return True, f"decoded {video.name}: {tuple(frame.shape)} {frame.dtype}, {n} frames"
    except Exception as e:  # noqa: BLE001 -- any decoder failure means the same thing here
        return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# per-episode worker
# --------------------------------------------------------------------------- #


def make_one_episode(src: str, task_index: int, episode_index: int, index_offset: int,
                     anchor: int, fps: float, out_base: str, chunks_size: int,
                     features: Features, transcode: bool, crf: int):
    src_d, out = Path(src), Path(out_base)
    chunk = f"chunk-{episode_index // chunks_size:03d}"

    traj = np.load(src_d / "traj.npz")
    lat = np.load(src_d / "latents.npz")
    meta = json.loads((src_d / "meta.json").read_text())

    jp = traj["joint_pos"].astype(np.float32)                    # (T, 29)
    pg = projected_gravity(traj["body_quat_w"][:, anchor])       # (T, 3)
    z = lat["z"].astype(np.float32)                              # (T, 64)
    T = len(z)
    assert len(jp) == T == meta["n_frames"], (len(jp), T, meta["n_frames"])

    states = np.concatenate([jp, pg], axis=1)                    # (T, 32)
    rows = [
        {
            "states": states[i].tolist(),
            "action": z[i].tolist(),
            "timestamp": i / fps,
            "frame_index": i,
            "episode_index": episode_index,
            "index": index_offset + i,
            "task_index": task_index,
            "next.done": (i == T - 1),
        }
        for i in range(T)
    ]

    pq_dir = out / "data" / chunk
    pq_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = out / "videos" / chunk / VIDEO_KEY
    vid_dir.mkdir(parents=True, exist_ok=True)

    tmp = out / "data" / f"_tmp_{episode_index:06d}"
    tmp.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=features).to_parquet(str(tmp / "e.parquet"))
    os.replace(tmp / "e.parquet", pq_dir / f"episode_{episode_index:06d}.parquet")
    shutil.rmtree(tmp, ignore_errors=True)

    dst_vid = vid_dir / f"episode_{episode_index:06d}.mp4"
    if transcode:
        transcode_420(src_d / "frames.mp4", dst_vid, crf)
    else:
        shutil.copyfile(src_d / "frames.mp4", dst_vid)

    ep_stats = {
        "episode_index": episode_index,
        "stats": {
            "action": {"min": z.min(0).tolist(), "max": z.max(0).tolist(),
                       "mean": z.mean(0).tolist(), "std": z.std(0).tolist(), "count": [T]},
            "states": {"min": states.min(0).tolist(), "max": states.max(0).tolist(),
                       "mean": states.mean(0).tolist(), "std": states.std(0).tolist(),
                       "count": [T]},
            "timestamp": {"min": [0.0], "max": [(T - 1) / fps], "mean": [((T - 1) / 2) / fps],
                          "std": [T / (2 * fps * math.sqrt(3))], "count": [T]},
        },
    }
    # Returned rather than appended from the worker: 4k processes appending to one jsonl needs a
    # lock to stay well-formed, and the parent has to sort by episode_index anyway.
    return episode_index, T, ep_stats, z, states, int(lat["has_future"].sum())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, help="collect_rollouts.py output dir")
    ap.add_argument("--out", required=True, help="LeRobot dataset dir to write")
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--crf", type=int, default=10, help="only used if a transcode is needed")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes, for a smoke test")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of episodes held out into <out>_val, stratified by "
                         "(sample_id, commanded_color). 0 disables. Psi0's `val_repo_ids` defaults "
                         "to the TRAINING repo, so without a real split every validation number -- "
                         "including drop_lang and vs_zoh -- is measured in-sample.")
    ap.add_argument("--keep-failures", action="store_true",
                    help="keep episodes where the commanded colour did NOT end up on top. These "
                         "are mislabeled for a command-following task; see the module docstring. "
                         "Exists to measure the cost of the filter, not to train on.")
    ap.add_argument("--overwrite", action="store_true", help="delete --out first")
    ap.add_argument("--push", metavar="REPO_ID", default=None,
                    help="after packing, upload the PARENT of --out to the Hugging Face dataset "
                         "repo REPO_ID, so both splits land under the directory names training "
                         "resolves against (--data.root_dir + --data.{train,val}_repo_ids) and a "
                         "download round-trips with no renaming. e.g. "
                         "--push sarveshv219/vibe-repose-sim. ~3.2 GB -- too large for GitHub, "
                         "whose hard cap is 100 MB per file. Needs HF_TOKEN in .env.")
    ap.add_argument("--private", action="store_true", help="with --push, create private repos")
    return ap.parse_args()


def push_to_hub(local: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import create_repo, upload_large_folder
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"[push] {local} -> https://huggingface.co/datasets/{repo_id}")
    upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local))


def main():
    args = parse_args()
    root, out = Path(args.rollouts), Path(args.out)
    summary = json.loads((root / "summary.json").read_text())
    lat_meta = json.loads((root / "latents_meta.json").read_text())

    assert lat_meta["pre_fsq"], "latents are post-quantizer; re-run extract_latents.py"
    assert summary["joint_order"] == "mj", summary["joint_order"]
    assert list(summary["face_color_names"]) == COLORS, summary["face_color_names"]
    anchor = summary["body_order"].index(ANCHOR_BODY)
    fps = float(summary["fps"])
    assert fps == int(fps), fps
    action_dim, W, H = int(lat_meta["latent_dim"]), summary["cam_width"], summary["cam_height"]

    out_val = out.with_name(out.name + "_val")
    for d in (out, out_val):
        if d.exists():
            if not args.overwrite:
                raise SystemExit(f"{d} exists; pass --overwrite to replace it")
            shutil.rmtree(d)

    # --- select episodes ---------------------------------------------------- #
    all_eps = sorted((root / "episodes").iterdir())
    if args.limit:
        all_eps = all_eps[:args.limit]
    kept, n_fail = [], 0
    for d in all_eps:
        m = json.loads((d / "meta.json").read_text())
        if not (d / "latents.npz").exists():
            raise SystemExit(f"{d.name} has no latents.npz -- run extract_latents.py first")
        ok = m["success"] and m["final_up_color_idx"] == m["commanded_color_idx"]
        if ok or args.keep_failures:
            kept.append((d, m))
        else:
            n_fail += 1
    print(f"[pack] {len(all_eps)} episodes, {len(kept)} kept, {n_fail} dropped as failures "
          f"({n_fail / max(len(all_eps), 1) * 100:.1f}%)")

    # --- video decoder probe, before anything is written -------------------- #
    ok, msg = probe_torchcodec(kept[0][0] / "frames.mp4")
    transcode = not ok
    pix_fmt = summary["video_pix_fmt"]
    if ok:
        print(f"[pack] torchcodec reads {pix_fmt} -- copying video as-is. {msg}")
    else:
        pix_fmt = "yuv420p"
        print(f"[pack] torchcodec REFUSED {summary['video_pix_fmt']}: {msg}")
        print(f"[pack] transcoding the corpus to yuv420p at crf {args.crf}. This is lossy in "
              f"chroma at saturated edges -- see the module docstring.")

    # --- train / val split --------------------------------------------------- #
    # Stratified by (sample_id, commanded_color) and deterministic: within each cell the episodes
    # are ordered and every k-th goes to val. That keeps all 78 motion clips and all 6 colours
    # present on BOTH sides, which is what this task needs -- the question is not whether the model
    # generalizes to unseen motions (at deploy it faces exactly these 78) but whether it picks the
    # right one from the command, so val must cover the same clips under unseen initial conditions.
    #
    # Splitting by episode is safe: episodes are independent rollouts, so no frame of a val episode
    # appears in train. A random FRAME split would not be -- rows 20 ms apart are near-duplicates.
    tasks_meta = {i: command(c) for i, c in enumerate(COLORS)}
    train_eps, val_eps = kept, []
    if args.val_frac > 0:
        k = max(2, int(round(1.0 / args.val_frac)))
        cells: Dict[Tuple[str, int], List[int]] = {}
        for i, (_d, m) in enumerate(kept):
            cells.setdefault((m["sample_id"], int(m["commanded_color_idx"])), []).append(i)
        val_idx = {i for idxs in cells.values() for i in idxs[::k]}
        train_eps = [e for i, e in enumerate(kept) if i not in val_idx]
        val_eps = [e for i, e in enumerate(kept) if i in val_idx]
        print(f"[pack] split {len(cells)} (sample_id, colour) cells -> "
              f"{len(train_eps)} train / {len(val_eps)} val episodes "
              f"({len(val_eps) / max(len(kept), 1) * 100:.1f}%), every {k}th per cell")

    print(f"[pack] {len(tasks_meta)} tasks, states 29+3={29 + 3}, action {action_dim}, "
          f"fps {int(fps)}  ->  chunk 30 = {30 / fps:.2f}s "
          f"(train with --model.action-chunk-size={int(fps)} for 1.0s)")

    features = Features({
        "states": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "index": Value("int64"),
        "task_index": Value("int64"),
        "next.done": Value("bool"),
    })


    def col_stats(X: np.ndarray) -> Dict[str, List[float]]:
        return {"min": X.min(0).tolist(), "max": X.max(0).tolist(),
                "mean": X.mean(0).tolist(), "std": X.std(0).tolist(),
                "q01": np.quantile(X, 0.01, axis=0).tolist(),
                "q99": np.quantile(X, 0.99, axis=0).tolist(),
                "count": [int(len(X))]}

    def pack_split(eps, out: Path, stats_from=None):
        """Convert one list of (episode_dir, meta) into a LeRobot dataset at `out`.

        `stats_from` makes the val split reuse the TRAIN split's normalization stats. Fitting
        separate min/max on val would put the two datasets on different scales, so a val error
        would not be comparable to a train error -- and at deploy only the train stats exist.
        """
        plan, cursor = [], 0
        for ep_index, (d, m) in enumerate(eps):
            plan.append((str(d), int(m["commanded_color_idx"]), ep_index, cursor))
            cursor += int(m["n_frames"])

        lengths, ep_stats, has_future, A, S = {}, {}, {}, [], []
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = [ex.submit(make_one_episode, src, ti, ei, off, anchor, fps, str(out),
                              args.chunks_size, features, transcode, args.crf)
                    for (src, ti, ei, off) in plan]
            for f in tqdm(as_completed(futs), total=len(futs), desc=out.name, unit="ep"):
                ei, n, st, z, s, hf = f.result()
                lengths[ei], ep_stats[ei], has_future[ei] = n, st, hf
                A.append(z)
                S.append(s)

        A, S = np.concatenate(A), np.concatenate(S)
        total_frames = int(sum(lengths.values()))
        assert total_frames == len(A) == cursor, (total_frames, len(A), cursor)

        meta_dir = out / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for ei in sorted(ep_stats):
                f.write(json.dumps(ep_stats[ei], separators=(",", ":")) + "\n")

        ep_rows, c = [], 0
        for (src, ti, ei, _off) in plan:
            n = lengths[ei]
            ep_rows.append({"episode_index": ei, "tasks": [ti], "length": n,
                            "dataset_from_index": c, "dataset_to_index": c + (n - 1),
                            "robot_type": "g1", "instruction": tasks_meta[ti],
                            "source_episode": Path(src).name,
                            "frames_with_full_future": has_future[ei]})
            c += n
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for r in ep_rows:
                f.write(json.dumps(r) + "\n")

        with open(meta_dir / "tasks.jsonl", "w") as f:
            for ti, desc in sorted(tasks_meta.items()):
                f.write(json.dumps({"task_index": ti, "task": desc,
                                    "category": "default", "description": desc}) + "\n")

        # `ActionStateTransform.populate_stats` reads min/max under action_norm_type=bounds and
        # q01/q99 under bounds_q99; both are written so the flag can change without re-packing.
        stats = stats_from if stats_from is not None else {
            "action": col_stats(A), "states": col_stats(S)}
        (meta_dir / "stats_psi0.json").write_text(json.dumps(stats, indent=2))

        info = InfoDict(
            codebase_version=CODE_VERSION, robot_type="g1",
            total_episodes=len(lengths), total_frames=total_frames, total_tasks=len(tasks_meta),
            total_videos=len(lengths),
            total_chunks=math.ceil(len(lengths) / args.chunks_size), chunks_size=args.chunks_size,
            fps=int(fps),
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            features={
                VIDEO_KEY: {"dtype": "video", "shape": [H, W, 3],
                            "names": ["height", "width", "channel"],
                            "video_info": {"video.fps": fps, "video.codec": "h264",
                                           "video.pix_fmt": pix_fmt,
                                           "video.is_depth_map": False, "has_audio": False}},
                "states": {"dtype": "float32", "shape": [S.shape[1]]},
                "action": {"dtype": "float32", "shape": [action_dim]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        )
        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))

        hf_frac = sum(has_future.values()) / total_frames
        print(f"\n[pack] -> {out}")
        print(f"  {len(lengths)} episodes, {total_frames} frames, {int(fps)} fps "
              f"({total_frames / fps / 3600:.2f} robot-hours)")
        print(f"  action  {A.shape[1]}-d  range [{A.min():.2f}, {A.max():.2f}]  "
              f"per-dim std {A.std(0).min():.3f}-{A.std(0).max():.3f}  "
              f"finite {np.isfinite(A).all()}")
        print(f"  states  {S.shape[1]}-d = joint_pos(29) + projected_gravity(3);  "
              f"|g| mean {np.linalg.norm(S[:, 29:], axis=1).mean():.4f} (should be 1.0)")
        print(f"  {hf_frac * 100:.1f}% of frames have an unclamped tokenizer future window")
        return stats, S.shape[1]

    train_stats, odim = pack_split(train_eps, out)
    if val_eps:
        pack_split(val_eps, out_val, stats_from=train_stats)

    print(f"\n[pack] video {W}x{H} h264 {pix_fmt}"
          f"{' (TRANSCODED from ' + summary['video_pix_fmt'] + ')' if transcode else ''}")
    print(f"\n  train with:  --data.root_dir={out.parent}")
    print(f"               --data.train_repo_ids={out.name}")
    if val_eps:
        print(f"               --data.val_repo_ids={out_val.name}   "
              f"<-- REQUIRED; it defaults to the TRAINING repo")
    print(f"               --model.action-dim={action_dim} --model.odim={odim} "
          f"--model.action-chunk-size={int(fps)}")
    print(f"               --data.transform.field.stat-path=meta/stats_psi0.json")

    if args.push:
        # The PARENT, not each split: `upload-large-folder` writes to the repo root, so
        # uploading `data/lerobot` reproduces `vibe_repose_g1/` and `vibe_repose_g1_val/`
        # verbatim on the Hub. Two separate repos would force a rename on download, and the
        # split names are load-bearing -- `val_repo_ids` defaults to the TRAINING repo, so a
        # mismatched name silently validates in-sample rather than erroring.
        push_to_hub(out.parent, args.push, args.private)


if __name__ == "__main__":
    main()
