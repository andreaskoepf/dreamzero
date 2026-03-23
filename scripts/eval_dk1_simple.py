#!/usr/bin/env python3
"""Minimal single-episode eval to debug video generation quality."""

import torch._dynamo
torch._dynamo.config.disable = True

import json
import os
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch
from PIL import Image

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


def main():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    model_path = "/workspace/checkpoints/dreamzero_dk1_merged_lora/checkpoint-20000"
    dataset_path = "/workspace/data/dk1-merge-2026-03"
    out_dir = "debug_eval"
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.DK1,
        model_path=model_path,
        device="cuda:0",
    )
    print("Model loaded.")

    # Pick episode 0, frame 100
    episode_id = 0
    start_row = 100

    # Load data
    chunks_size = 1000
    chunk_idx = episode_id // chunks_size
    file_idx = episode_id % chunks_size
    pq_path = os.path.join(dataset_path, "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")
    table = pq.read_table(pq_path)

    with open(os.path.join(dataset_path, "meta", "episodes.jsonl")) as f:
        for line in f:
            meta = json.loads(line)
            if meta["episode_index"] == episode_id:
                task = meta["tasks"][0]
                break

    print(f"Episode {episode_id}, frame {start_row}, task: {task!r}")
    print(f"Episode length: {table.num_rows}")

    # Read and save GT frames
    for cam_name, cam_folder in [("top", "observation.images.top"),
                                  ("left_wrist", "observation.images.left_wrist"),
                                  ("right_wrist", "observation.images.right_wrist")]:
        vid_path = os.path.join(dataset_path, "videos", cam_folder,
                                f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4")
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_row)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(os.path.join(out_dir, f"gt_{cam_name}.png"))
            print(f"  GT {cam_name}: shape={frame_rgb.shape}")

    # Build observation
    obs = {}
    for server_key, folder_name in [("video.top", "observation.images.top"),
                                     ("video.left_wrist", "observation.images.left_wrist"),
                                     ("video.right_wrist", "observation.images.right_wrist")]:
        vid_path = os.path.join(dataset_path, "videos", folder_name,
                                f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4")
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_row)
        ret, frame = cap.read()
        cap.release()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        obs[server_key] = frame_rgb.astype(np.uint8)  # (H, W, C)
        print(f"  obs[{server_key}] shape: {obs[server_key].shape}, dtype: {obs[server_key].dtype}")

    state = np.array(table.column("observation.state")[start_row].as_py(), dtype=np.float64)
    for key, (s, e) in [("state.left_joint_pos", (0, 6)),
                         ("state.left_gripper_pos", (6, 7)),
                         ("state.right_joint_pos", (7, 13)),
                         ("state.right_gripper_pos", (13, 14))]:
        obs[key] = state[s:e].reshape(1, -1).astype(np.float64)
        print(f"  obs[{key}] shape: {obs[key].shape}, values: {obs[key]}")

    obs["annotation.task"] = task
    print(f"  obs['annotation.task'] = {task!r}")

    # Run inference
    print("\nRunning inference...")
    policy.trained_model.action_head.language = None
    policy.trained_model.action_head.current_start_frame = 0

    with torch.inference_mode():
        result_batch, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))

    # Print video_pred info
    if video_pred is not None:
        print(f"\nvideo_pred type: {type(video_pred)}")
        print(f"video_pred shape: {video_pred.shape}")
        print(f"video_pred dtype: {video_pred.dtype}")
        print(f"video_pred min/max: {video_pred.min():.3f} / {video_pred.max():.3f}")

        # Decode with VAE
        print("\nDecoding latent video with VAE...")
        ah = policy.trained_model.action_head
        with torch.inference_mode():
            decoded = ah.vae.decode(
                video_pred,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
        print(f"VAE decoded shape: {decoded.shape}")  # (B, C, T, H, W)
        print(f"VAE decoded min/max: {decoded.min():.3f} / {decoded.max():.3f}")

        # Convert to frames
        frames = rearrange(decoded, "B C T H W -> B T H W C")
        frames = frames[0]  # (T, H, W, C)
        frames_np = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        print(f"Final frames shape: {frames_np.shape}")

        # Save each frame as PNG
        for i in range(frames_np.shape[0]):
            frame = frames_np[i]
            Image.fromarray(frame).save(os.path.join(out_dir, f"pred_frame_{i:03d}.png"))
            print(f"  Saved pred_frame_{i:03d}.png: shape={frame.shape}")

    # Print action predictions
    print("\nAction predictions:")
    for key in ["action.left_joint_pos", "action.left_gripper_pos",
                "action.right_joint_pos", "action.right_gripper_pos"]:
        if key in result_batch.act:
            val = result_batch.act[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            print(f"  {key}: shape={np.array(val).shape}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
