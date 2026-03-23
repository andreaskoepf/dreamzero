#!/usr/bin/env python3
"""Open-loop evaluation for DreamZero LoRA checkpoint on DK-1 merged dataset.

Picks random episodes, runs multi-chunk autoregressive inference from a random
start position, generates predicted video + actions, and compares with ground-truth.

Outputs per-episode:
  - Side-by-side video (GT left | predicted right) as MP4
  - Action prediction vs ground-truth plots (full multi-chunk rollout)
  - Per-chunk and overall MSE

Usage (single GPU, gloo backend):
    python scripts/eval_dk1_open_loop.py \
        --model_path /workspace/checkpoints/dreamzero_dk1_merged_lora/checkpoint-20000 \
        --dataset_path /workspace/data/dk1-merge-2026-03 \
        --num_episodes 64 --num_chunks 4 \
        --output_dir results_dk1_eval
"""

import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import json
import os
import random
import time

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


# ---------------------------------------------------------------------------
# DK-1 layout (from dk1-merge-2026-03/meta/modality.json)
# ---------------------------------------------------------------------------

VIDEO_CAMERAS = {
    "video.top":         "observation.images.top",
    "video.left_wrist":  "observation.images.left_wrist",
    "video.right_wrist": "observation.images.right_wrist",
}

STATE_SLICES = {
    "state.left_joint_pos":    (0, 6),
    "state.left_gripper_pos":  (6, 7),
    "state.right_joint_pos":   (7, 13),
    "state.right_gripper_pos": (13, 14),
}

ACTION_SLICES = {
    "action.left_joint_pos":    (0, 6),
    "action.left_gripper_pos":  (6, 7),
    "action.right_joint_pos":   (7, 13),
    "action.right_gripper_pos": (13, 14),
}

ACTION_KEY_ORDER = [
    "action.left_joint_pos",
    "action.left_gripper_pos",
    "action.right_joint_pos",
    "action.right_gripper_pos",
]

ACTION_HORIZON = 24  # model predicts 24 future steps per chunk


# ---------------------------------------------------------------------------
# Dataset reader (LeRobot v3 chunked format)
# ---------------------------------------------------------------------------

class DK1Dataset:
    """Reads DK-1 merged dataset in LeRobot v3 chunked parquet + MP4 format."""

    def __init__(self, dataset_path: str):
        self.root = dataset_path
        meta_dir = os.path.join(dataset_path, "meta")

        episodes_path = os.path.join(meta_dir, "episodes.jsonl")
        self.episodes_meta = []
        with open(episodes_path) as f:
            for line in f:
                self.episodes_meta.append(json.loads(line))
        self.num_episodes = len(self.episodes_meta)

        tasks_path = os.path.join(meta_dir, "tasks.jsonl")
        self.tasks = {}
        with open(tasks_path) as f:
            for line in f:
                entry = json.loads(line)
                self.tasks[entry["task_index"]] = entry["task"]

        with open(os.path.join(meta_dir, "info.json")) as f:
            self.info = json.load(f)
        self.chunks_size = self.info["chunks_size"]

        print(f"DK1Dataset: {self.num_episodes} episodes, "
              f"{self.info['total_frames']} total frames, "
              f"{self.info['total_tasks']} tasks")

    def _parquet_path(self, episode_id: int) -> str:
        chunk_idx = episode_id // self.chunks_size
        file_idx = episode_id % self.chunks_size
        return os.path.join(
            self.root, "data",
            f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet"
        )

    def _video_path(self, episode_id: int, video_key: str) -> str:
        chunk_idx = episode_id // self.chunks_size
        file_idx = episode_id % self.chunks_size
        return os.path.join(
            self.root, "videos", video_key,
            f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4"
        )

    def get_episode_length(self, episode_id: int) -> int:
        return self.episodes_meta[episode_id]["length"]

    def get_episode_task(self, episode_id: int) -> str:
        tasks = self.episodes_meta[episode_id]["tasks"]
        return tasks[0] if tasks else ""

    def load_episode_parquet(self, episode_id: int):
        return pq.read_table(self._parquet_path(episode_id))

    def get_state(self, table, row: int) -> np.ndarray:
        return np.array(table.column("observation.state")[row].as_py(), dtype=np.float64)

    def get_action(self, table, row: int) -> np.ndarray:
        return np.array(table.column("action")[row].as_py(), dtype=np.float64)

    def get_frame(self, episode_id: int, video_key: str, row: int) -> np.ndarray:
        """Read one video frame -> (H, W, 3) uint8 RGB."""
        mp4 = self._video_path(episode_id, video_key)
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, row)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {row} from {mp4}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frames_range(self, episode_id: int, video_key: str,
                         start: int, count: int) -> list[np.ndarray]:
        """Read a range of consecutive frames -> list of (H, W, 3) uint8 RGB."""
        mp4 = self._video_path(episode_id, video_key)
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_obs(dataset: DK1Dataset, episode_id: int, table, row: int, prompt: str) -> dict:
    """Build an obs dict matching what GrootSimPolicy expects."""
    obs = {}

    for server_key, folder_name in VIDEO_CAMERAS.items():
        frame = dataset.get_frame(episode_id, folder_name, row)
        obs[server_key] = frame.astype(np.uint8)  # (H, W, C) — unsqueeze adds batch+time dims

    state = dataset.get_state(table, row)
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = state[start:end].reshape(1, -1).astype(np.float64)  # (1, D) = (T=1, D)

    obs["annotation.task"] = prompt
    return obs


def get_gt_actions(table, row: int, horizon: int) -> dict:
    """Get ground-truth actions for `horizon` steps starting at `row`."""
    max_row = table.num_rows
    gt = {k: [] for k in ACTION_KEY_ORDER}
    for t in range(horizon):
        r = min(row + t, max_row - 1)
        action_flat = np.array(table.column("action")[r].as_py(), dtype=np.float64)
        for key in ACTION_KEY_ORDER:
            s, e = ACTION_SLICES[key]
            gt[key].append(action_flat[s:e])
    return {k: np.stack(v) for k, v in gt.items()}


def extract_pred_actions(result_batch) -> dict:
    """Extract predicted actions from model output, ensuring 2D (horizon, D)."""
    pred_actions = {}
    for key in ACTION_KEY_ORDER:
        if key in result_batch.act:
            val = result_batch.act[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            val = np.atleast_1d(val)
            if val.ndim == 1:
                val = val[:, np.newaxis]  # (horizon,) -> (horizon, 1)
            elif val.ndim == 3:
                val = val[0]  # remove batch dim -> (horizon, D)
            pred_actions[key] = val
    return pred_actions


# ---------------------------------------------------------------------------
# Video decoding & saving
# ---------------------------------------------------------------------------

def decode_latent_video(policy, latent_video: torch.Tensor) -> np.ndarray:
    """Decode VAE latent video -> (T, H, W, 3) uint8 numpy array.

    Returns the full tiled 2x2 grid (2*h x 2*w).
    """
    ah = policy.trained_model.action_head
    with torch.inference_mode():
        frames = ah.vae.decode(
            latent_video,
            tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
        )
    # frames: (B, C, T, H, W)
    frames = rearrange(frames, "B C T H W -> B T H W C")
    frames = frames[0]  # drop batch dim
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return frames


def split_tiled_views(tiled_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 2x2 tiled frame into individual views.

    DK1 training layout (from _prepare_video):
        [view 0 = top       | view 2 = right_wrist]
        [view 1 = left_wrist| BLACK               ]

    Args:
        tiled_frame: (H, W, C) where H=2*h, W=2*w
    Returns:
        dict with keys 'top', 'left_wrist', 'right_wrist'
    """
    h = tiled_frame.shape[0] // 2
    w = tiled_frame.shape[1] // 2
    return {
        "top":         tiled_frame[:h, :w],
        "left_wrist":  tiled_frame[h:, :w],
        "right_wrist": tiled_frame[:h, w:],
    }


def save_per_view_comparison_video(dataset, episode_id, start_row,
                                   pred_frames: np.ndarray,
                                   output_path: str, fps: int = 5):
    """Save 3-row comparison video: GT left | Pred right, for each camera view.

    Each row is one camera view (top, left_wrist, right_wrist).
    """
    n_frames = pred_frames.shape[0]
    view_names = ["top", "left_wrist", "right_wrist"]
    cam_folders = {
        "top": "observation.images.top",
        "left_wrist": "observation.images.left_wrist",
        "right_wrist": "observation.images.right_wrist",
    }

    # Load GT frames for each view
    gt_views = {}
    for name in view_names:
        gt_views[name] = dataset.get_frames_range(
            episode_id, cam_folders[name], start_row, n_frames
        )

    combined_frames = []
    for i in range(n_frames):
        pred_views = split_tiled_views(pred_frames[i])
        rows = []
        for name in view_names:
            gt = gt_views[name][i] if i < len(gt_views[name]) else np.zeros_like(pred_views[name])
            pr = pred_views[name]
            # Resize pred to match GT if needed
            if gt.shape[0] != pr.shape[0] or gt.shape[1] != pr.shape[1]:
                pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]))
            gt_labeled = gt.copy()
            pr_labeled = pr.copy()
            label_scale = max(0.4, gt.shape[0] / 360)
            cv2.putText(gt_labeled, f"GT {name}", (5, int(20 * label_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1)
            cv2.putText(pr_labeled, f"Pred {name}", (5, int(20 * label_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1)
            row = np.concatenate([gt_labeled, pr_labeled], axis=1)
            rows.append(row)
        # Stack 3 views vertically
        frame = np.concatenate(rows, axis=0)
        combined_frames.append(frame)

    if combined_frames:
        imageio.mimsave(output_path, combined_frames, fps=fps, codec="libx264")


def save_action_plots(gt_actions: dict, pred_actions: dict,
                      output_path: str, task: str, episode_id: int, start_row: int,
                      num_chunks: int = 1):
    """Plot predicted vs ground-truth actions for all keys over full rollout."""
    nkeys = len(ACTION_KEY_ORDER)
    total_horizon = gt_actions[ACTION_KEY_ORDER[0]].shape[0]
    fig, axes = plt.subplots(nkeys, 1, figsize=(max(12, total_horizon * 0.15), 3 * nkeys), squeeze=False)
    fig.suptitle(f"Episode {episode_id}, frame {start_row}, {num_chunks} chunks\nTask: {task}", fontsize=11)

    for i, key in enumerate(ACTION_KEY_ORDER):
        ax = axes[i][0]
        gt = gt_actions[key]
        pr = pred_actions[key]
        D = gt.shape[1]
        for d in range(D):
            ax.plot(gt[:, d], "--", alpha=0.6, lw=1.0, label=f"gt_d{d}" if i == 0 else None)
            ax.plot(pr[:, d], alpha=0.8, lw=1.0, label=f"pred_d{d}" if i == 0 else None)
        # Draw chunk boundaries
        for c in range(1, num_chunks):
            ax.axvline(x=c * ACTION_HORIZON, color="gray", ls=":", alpha=0.5)
        mse = float(np.mean((gt - pr) ** 2))
        ax.set_title(f"{key}  (MSE={mse:.6f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    if nkeys > 0:
        axes[0][0].legend(fontsize=6, ncol=6, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-chunk autoregressive rollout
# ---------------------------------------------------------------------------

def run_episode_rollout(policy, dataset, episode_id, table, start_row, task,
                        num_chunks, rank, ar_rollout=False):
    """Run multi-chunk AR rollout for one episode.

    When ar_rollout=False (default): each chunk is conditioned on a fresh GT
    frame at its time position (open-loop w.r.t. actions, GT-conditioned video).

    When ar_rollout=True: only the first chunk gets a GT frame.  Subsequent
    chunks feed the previous chunk's latent video prediction back into the
    model, producing a longer video from a single start frame.

    Returns:
        all_pred_actions: dict of concatenated actions across chunks
        all_gt_actions: dict of concatenated GT actions
        all_latent_videos: list of latent video tensors for VAE decoding
        total_time: total inference time
        per_chunk_mse: list of per-chunk MSE values
    """
    ep_len = table.num_rows
    total_horizon = num_chunks * ACTION_HORIZON

    # Collect across chunks
    all_pred_actions = {k: [] for k in ACTION_KEY_ORDER}
    all_gt_actions = {k: [] for k in ACTION_KEY_ORDER}
    all_latent_videos = []
    per_chunk_mse = []
    total_time = 0.0

    # Reset model's internal AR state by setting language to None
    policy.trained_model.action_head.language = None
    policy.trained_model.action_head.current_start_frame = 0

    prev_video_pred = None

    for chunk_idx in range(num_chunks):
        # Current frame position for this chunk (for GT actions comparison)
        current_row = min(start_row + chunk_idx * ACTION_HORIZON, ep_len - 1)

        if ar_rollout and chunk_idx > 0:
            # AR continuation: reuse the first chunk's GT frame observation
            # but pass the previous chunk's latent video as conditioning.
            obs = build_obs(dataset, episode_id, table, start_row, task)
        else:
            # GT-conditioned: build observation from GT data at this position
            obs = build_obs(dataset, episode_id, table, current_row, task)

        t0 = time.perf_counter()
        with torch.inference_mode():
            latent_video_arg = prev_video_pred if (ar_rollout and chunk_idx > 0) else None
            result_batch, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=obs), latent_video=latent_video_arg,
            )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        prev_video_pred = video_pred

        # Extract predictions
        chunk_pred = extract_pred_actions(result_batch)

        # GT actions for this chunk
        gt_horizon = min(ACTION_HORIZON, ep_len - current_row)
        chunk_gt = get_gt_actions(table, current_row, gt_horizon)

        # Compute chunk MSE
        pred_flat = np.concatenate([chunk_pred[k][:gt_horizon] for k in ACTION_KEY_ORDER], axis=-1)
        gt_flat = np.concatenate([chunk_gt[k] for k in ACTION_KEY_ORDER], axis=-1)
        min_h = min(pred_flat.shape[0], gt_flat.shape[0])
        chunk_mse = float(np.mean((pred_flat[:min_h] - gt_flat[:min_h]) ** 2))
        per_chunk_mse.append(chunk_mse)

        # Accumulate
        for key in ACTION_KEY_ORDER:
            all_pred_actions[key].append(chunk_pred[key][:gt_horizon])
            all_gt_actions[key].append(chunk_gt[key])

        if video_pred is not None:
            all_latent_videos.append(video_pred)

        mode = "AR" if (ar_rollout and chunk_idx > 0) else "GT"
        if rank == 0:
            print(f"    Chunk {chunk_idx+1}/{num_chunks} [{mode}]: row={current_row}, "
                  f"MSE={chunk_mse:.6f}, time={elapsed:.2f}s")

    # Concatenate across chunks
    all_pred_actions = {k: np.concatenate(v, axis=0) for k, v in all_pred_actions.items()}
    all_gt_actions = {k: np.concatenate(v, axis=0) for k, v in all_gt_actions.items()}

    return all_pred_actions, all_gt_actions, all_latent_videos, total_time, per_chunk_mse


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    rank = dist.get_rank()

    print(f"[rank {rank}] Loading model from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.DK1,
        model_path=args.model_path,
        device=args.device,
    )
    print(f"[rank {rank}] Model loaded.")

    dataset = DK1Dataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pick random episodes
    rng = random.Random(args.seed)
    episode_ids = rng.sample(range(dataset.num_episodes), min(args.num_episodes, dataset.num_episodes))

    results_summary = []

    for ep_idx, episode_id in enumerate(episode_ids):
        ep_len = dataset.get_episode_length(episode_id)
        task = dataset.get_episode_task(episode_id)

        # Need room for num_chunks * ACTION_HORIZON frames
        min_remaining = args.num_chunks * ACTION_HORIZON + 8
        if ep_len <= min_remaining:
            print(f"  Skipping episode {episode_id} (too short: {ep_len} < {min_remaining})")
            continue
        start_row = rng.randint(0, ep_len - min_remaining - 1)

        if rank == 0:
            print(f"\n[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id}, "
                  f"start={start_row}, length={ep_len}, task={task!r}")

        table = dataset.load_episode_parquet(episode_id)

        # Multi-chunk AR rollout
        all_pred, all_gt, all_latents, total_time, per_chunk_mse = run_episode_rollout(
            policy, dataset, episode_id, table, start_row, task,
            args.num_chunks, rank, ar_rollout=args.ar_rollout,
        )

        # Overall MSE
        pred_flat = np.concatenate([all_pred[k] for k in ACTION_KEY_ORDER], axis=-1)
        gt_flat = np.concatenate([all_gt[k] for k in ACTION_KEY_ORDER], axis=-1)
        min_h = min(pred_flat.shape[0], gt_flat.shape[0])
        mse = float(np.mean((pred_flat[:min_h] - gt_flat[:min_h]) ** 2))

        per_key_mse = {}
        for key in ACTION_KEY_ORDER:
            p, g = all_pred[key], all_gt[key]
            h = min(p.shape[0], g.shape[0])
            per_key_mse[key] = float(np.mean((p[:h] - g[:h]) ** 2))

        results_summary.append({
            "episode_id": episode_id,
            "start_row": start_row,
            "task": task,
            "mse": mse,
            "per_key_mse": per_key_mse,
            "per_chunk_mse": per_chunk_mse,
            "num_chunks": args.num_chunks,
            "total_inference_time": total_time,
        })

        if rank == 0:
            print(f"  Overall MSE: {mse:.6f} ({args.num_chunks} chunks, {total_time:.1f}s)")

        # Save outputs
        if rank == 0:
            ep_dir = os.path.join(args.output_dir, f"ep{episode_id:04d}_f{start_row:05d}")
            os.makedirs(ep_dir, exist_ok=True)

            # Action plots (full rollout)
            save_action_plots(
                all_gt, all_pred,
                os.path.join(ep_dir, "actions.png"),
                task, episode_id, start_row, args.num_chunks,
            )

            # Decode and save videos
            if all_latents:
                try:
                    # Decode each chunk independently to avoid VAE temporal
                    # interpolation artifacts at chunk boundaries.
                    chunk_frames = []
                    for lat in all_latents:
                        chunk_frames.append(decode_latent_video(policy, lat))
                    pred_frames = np.concatenate(chunk_frames, axis=0)
                    n_pred_frames = pred_frames.shape[0]

                    # Save full tiled prediction
                    imageio.mimsave(
                        os.path.join(ep_dir, "video_pred_tiled.mp4"),
                        list(pred_frames), fps=5, codec="libx264",
                    )

                    # Save per-view GT vs Pred comparison (3 rows)
                    save_per_view_comparison_video(
                        dataset, episode_id, start_row, pred_frames,
                        os.path.join(ep_dir, "video_comparison.mp4"),
                        fps=5,
                    )

                    print(f"  Saved {n_pred_frames} predicted frames + per-view comparison")
                except Exception as e:
                    print(f"  Warning: Failed to decode/save video: {e}")
                    import traceback; traceback.print_exc()

    # Save summary
    if rank == 0 and results_summary:
        overall_mse = float(np.mean([r["mse"] for r in results_summary]))

        # Per-task breakdown
        task_mses = {}
        for r in results_summary:
            task_mses.setdefault(r["task"], []).append(r["mse"])
        task_summary = {t: {"mean_mse": float(np.mean(v)), "count": len(v)}
                        for t, v in task_mses.items()}

        print(f"\n{'='*60}")
        print(f"Overall MSE across {len(results_summary)} episodes: {overall_mse:.6f}")
        print(f"Per-task breakdown:")
        for t, s in sorted(task_summary.items(), key=lambda x: x[1]["mean_mse"]):
            print(f"  {t}: MSE={s['mean_mse']:.6f} (n={s['count']})")
        print(f"{'='*60}")

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "overall_mse": overall_mse,
                "num_episodes": len(results_summary),
                "num_chunks": args.num_chunks,
                "task_summary": task_summary,
                "episodes": results_summary,
            }, f, indent=2)
        print(f"Summary saved to {summary_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="Path to checkpoint dir (config.json, model.safetensors, experiment_cfg/)")
    p.add_argument("--dataset_path", default="/workspace/data/dk1-merge-2026-03",
                   help="Root of DK-1 merged dataset")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_episodes", type=int, default=64,
                   help="Number of random episodes to evaluate")
    p.add_argument("--num_chunks", type=int, default=4,
                   help="Number of AR chunks per episode (each = 24 action steps)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for episode/position selection")
    p.add_argument("--ar_rollout", action="store_true",
                   help="Generate longer video from a single start frame by chaining "
                        "each chunk's latent output as conditioning for the next chunk, "
                        "instead of re-conditioning on GT at each chunk boundary")
    p.add_argument("--output_dir", default="results_dk1_eval")
    main_args = p.parse_args()
    evaluate(main_args)


if __name__ == "__main__":
    main()
