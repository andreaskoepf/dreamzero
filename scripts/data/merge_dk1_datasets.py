#!/usr/bin/env python3
"""Merge all DK-1 datasets into a single homogeneous LeRobot v3 dataset.

Reads individual DK-1 datasets from a data root, applies homogenization
(action dim slicing, camera name mapping, video re-encoding to 640x480 h264),
and writes a single merged dataset suitable for HuggingFace Hub upload and
DreamZero training.

Usage:
    python scripts/data/merge_dk1_datasets.py \
        --data-root /workspace/data/dk1 \
        --output /workspace/data/dk1-merge-march2026 \
        --ffmpeg-workers 8 --resume

    # Verify only (no writing):
    python scripts/data/merge_dk1_datasets.py \
        --data-root /workspace/data/dk1 \
        --output /workspace/data/dk1-merge-march2026 \
        --verify-only
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DK-1 constants
# ---------------------------------------------------------------------------

STATE_KEYS = {
    "left_joint_pos": [0, 6],
    "left_gripper_pos": [6, 7],
    "right_joint_pos": [7, 13],
    "right_gripper_pos": [13, 14],
}
ACTION_KEYS = {
    "left_joint_pos": [0, 6],
    "left_gripper_pos": [6, 7],
    "right_joint_pos": [7, 13],
    "right_gripper_pos": [13, 14],
}
RELATIVE_ACTION_KEYS = [
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
]

# Datasets to always exclude
EXCLUDE_NAMES = {
    "sfeduniak_dhl_0310",
    "sfeduniak_towels_0310",
    "FabianKerj_dualarm-pretrain-417ep",
    "Zasha01_eval_pi05-cube-transfer-final-full15",
    "Zasha01_eval_pi05-cube-transfer_v1",
}

# Overview camera names → normalised name "top"
OVERVIEW_CAM_NAMES = {"top", "context", "base_0"}
WRIST_CAMS = {"left_wrist", "right_wrist"}
TARGET_CAM_NAMES = ["top", "left_wrist", "right_wrist"]  # output camera order

FPS = 30
CHUNKS_SIZE = 1000
TARGET_ACTION_DIM = 14
TARGET_STATE_DIM = 14


# ===================================================================
# Phase 1: Discovery & Filtering
# ===================================================================

def load_source_info(dataset_path: Path) -> dict | None:
    """Load info.json for a source dataset. Returns None on failure."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        return json.load(f)


def get_camera_info(info: dict) -> dict:
    """Extract camera names and classify them."""
    features = info.get("features", {})
    video_keys = {
        k.replace("observation.images.", ""): v
        for k, v in features.items()
        if v.get("dtype") == "video"
    }
    overview = [c for c in video_keys if c in OVERVIEW_CAM_NAMES]
    wrists = [c for c in video_keys if c in WRIST_CAMS]
    return {
        "all": video_keys,
        "overview": overview[0] if overview else None,
        "wrists": sorted(wrists),
        "count": len(video_keys),
    }


def get_action_dim(info: dict) -> int:
    """Get action dimensionality from info.json."""
    features = info.get("features", {})
    action_feat = features.get("action", {})
    shape = action_feat.get("shape", [0])
    return shape[0] if isinstance(shape, list) else shape


def discover_datasets(data_root: Path) -> tuple[list[dict], list[dict]]:
    """Discover and filter datasets. Returns (included, excluded) lists."""
    included = []
    excluded = []

    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        name = ds_dir.name

        # Hard exclusions
        if name in EXCLUDE_NAMES:
            excluded.append({"name": name, "reason": "hard-excluded"})
            continue

        info = load_source_info(ds_dir)
        if info is None:
            excluded.append({"name": name, "reason": "no info.json"})
            continue

        cam_info = get_camera_info(info)

        # Camera count check
        if cam_info["count"] < 3:
            excluded.append({"name": name, "reason": f"only {cam_info['count']} cameras"})
            continue

        if cam_info["overview"] is None:
            excluded.append({"name": name, "reason": "no overview camera (top/context/base_0)"})
            continue

        if len(cam_info["wrists"]) < 2:
            excluded.append({"name": name, "reason": f"missing wrist cameras: {cam_info['wrists']}"})
            continue

        # Frame count check
        total_frames = info.get("total_frames", 0)
        if total_frames < 300:
            excluded.append({"name": name, "reason": f"too few frames ({total_frames})"})
            continue

        action_dim = get_action_dim(info)
        if action_dim not in (14, 28):
            excluded.append({"name": name, "reason": f"unexpected action dim {action_dim}"})
            continue

        # Discover parquet and video files
        parquet_dir = ds_dir / "data"
        parquet_files = sorted(parquet_dir.rglob("*.parquet")) if parquet_dir.exists() else []
        if not parquet_files:
            excluded.append({"name": name, "reason": "no parquet files"})
            continue

        included.append({
            "name": name,
            "path": ds_dir,
            "info": info,
            "cam_info": cam_info,
            "action_dim": action_dim,
            "parquet_files": parquet_files,
            "total_frames": total_frames,
            "fps": info.get("fps", FPS),
        })

    return included, excluded


# ===================================================================
# Phase 2: Validation
# ===================================================================

def validate_dataset(ds: dict) -> list[str]:
    """Validate a single dataset. Returns list of error strings."""
    errors = []
    ds_path = ds["path"]
    cam_info = ds["cam_info"]
    overview_cam = cam_info["overview"]

    # Check parquet files have expected columns
    first_pq = ds["parquet_files"][0]
    try:
        df = pd.read_parquet(first_pq, columns=["action", "observation.state"])
        row = df.iloc[0]
        action = np.array(row["action"])
        state = np.array(row["observation.state"])
        if state.shape[0] != TARGET_STATE_DIM:
            errors.append(f"state dim {state.shape[0]} != {TARGET_STATE_DIM}")
        if action.shape[0] != ds["action_dim"]:
            errors.append(f"action dim mismatch: parquet={action.shape[0]} info={ds['action_dim']}")
    except Exception as e:
        errors.append(f"failed to read parquet: {e}")

    # Check video files exist for each parquet file
    for pf in ds["parquet_files"]:
        # Derive chunk/file from parquet path
        chunk_name = pf.parent.name  # e.g., "chunk-000"
        file_stem = pf.stem  # e.g., "file-000"

        for cam_name in [overview_cam] + sorted(WRIST_CAMS):
            full_cam_key = f"observation.images.{cam_name}"
            video_path = ds_path / "videos" / full_cam_key / chunk_name / f"{file_stem}.mp4"
            if not video_path.exists():
                errors.append(f"missing video: {video_path.relative_to(ds_path)}")

    return errors


# ===================================================================
# Phase 3: Task Merging
# ===================================================================

def load_source_tasks(ds_path: Path) -> dict[int, str]:
    """Load task_index -> task_string from source dataset."""
    # Try tasks.parquet first (v3)
    tasks_pq = ds_path / "meta" / "tasks.parquet"
    if tasks_pq.exists():
        df = pd.read_parquet(tasks_pq).reset_index()
        task_col = "task" if "task" in df.columns else df.columns[0]
        idx_col = "task_index" if "task_index" in df.columns else None
        if idx_col:
            return {int(row[idx_col]): str(row[task_col]) for _, row in df.iterrows()}
        return {i: str(row[task_col]) for i, row in df.iterrows()}

    # Fall back to tasks.jsonl
    tasks_jsonl = ds_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        lookup = {}
        with open(tasks_jsonl) as f:
            for line in f:
                entry = json.loads(line.strip())
                lookup[entry["task_index"]] = entry["task"]
        return lookup

    return {0: ""}


def merge_tasks(datasets: list[dict]) -> tuple[list[dict], dict[str, dict[str, dict[int, int]]]]:
    """Merge tasks across all datasets.

    Returns:
        tasks: list of {"task_index": int, "task": str}
        remap: dict[dataset_name -> dict of old_task_index -> new_task_index]
    """
    global_tasks: dict[str, int] = {}  # task_string -> global_index
    remap: dict[str, dict[int, int]] = {}

    for ds in datasets:
        source_tasks = load_source_tasks(ds["path"])
        ds_remap: dict[int, int] = {}
        for old_idx, task_str in source_tasks.items():
            if task_str not in global_tasks:
                global_tasks[task_str] = len(global_tasks)
            ds_remap[old_idx] = global_tasks[task_str]
        remap[ds["name"]] = ds_remap

    tasks = [{"task_index": idx, "task": text} for text, idx in sorted(global_tasks.items(), key=lambda x: x[1])]
    return tasks, remap


# ===================================================================
# Phase 4: Parquet Merge
# ===================================================================

def split_parquet_by_episode(
    source_parquet: Path,
    action_dim: int,
    fps: float,
    task_remap: dict[int, int],
) -> list[tuple[pd.DataFrame, int, int]]:
    """Split a source parquet file into per-episode DataFrames.

    Returns list of (episode_df, original_episode_index, global_row_start_in_file).
    Each episode_df has action sliced, task_index remapped, but episode_index/index/
    frame_index/timestamp are NOT yet set (caller assigns global values).
    """
    df = pd.read_parquet(source_parquet)

    # Slice action to 14D if needed
    if action_dim > TARGET_ACTION_DIM:
        df["action"] = df["action"].apply(lambda a: np.array(a)[:TARGET_ACTION_DIM])

    # Remap task_index
    if "task_index" in df.columns:
        df["task_index"] = df["task_index"].map(lambda x: task_remap.get(int(x), 0))

    # Split by original episode_index
    episodes = []
    cumulative_row = 0
    for orig_ep_idx in sorted(df["episode_index"].unique()):
        ep_df = df[df["episode_index"] == orig_ep_idx].copy()
        episodes.append((ep_df, int(orig_ep_idx), cumulative_row))
        cumulative_row += len(ep_df)

    return episodes


def write_episode_parquet(
    ep_df: pd.DataFrame,
    output_parquet: Path,
    global_episode_index: int,
    global_index_start: int,
    fps: float,
) -> int:
    """Write a single episode DataFrame to a parquet file.

    Returns the number of frames written.
    """
    n_frames = len(ep_df)

    ep_df["episode_index"] = global_episode_index
    ep_df["frame_index"] = np.arange(n_frames, dtype=np.int64)
    ep_df["index"] = np.arange(global_index_start, global_index_start + n_frames, dtype=np.int64)
    ep_df["timestamp"] = np.arange(n_frames, dtype=np.float32) / fps

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(output_parquet, index=False)
    return n_frames


# ===================================================================
# Phase 5: Video Re-encoding
# ===================================================================

def get_video_info(video_path: Path) -> dict | None:
    """Get video metadata via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,width,height,codec_name",
                "-of", "json", str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return {
            "nb_frames": int(stream.get("nb_frames", 0)),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "codec": stream.get("codec_name", "unknown"),
        }
    except Exception:
        return None


def reencode_video(
    source_video: Path,
    output_video: Path,
    target_width: int,
    target_height: int,
) -> bool:
    """Re-encode a video to target resolution and h264 codec.

    Uses ffmpeg with scale + center-crop filter.
    Returns True on success.
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Scale shortest side up, then center-crop to exact target
    vf = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        str(output_video),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error("ffmpeg failed for %s: %s", source_video, result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out for %s", source_video)
        return False
    except Exception as e:
        log.error("ffmpeg error for %s: %s", source_video, e)
        return False


def _reencode_worker(args: tuple) -> tuple[str, bool]:
    """Worker function for parallel video re-encoding."""
    src, dst, tw, th = args
    ok = reencode_video(Path(src), Path(dst), tw, th)
    return str(dst), ok


def build_source_video_frame_map(
    ds_path: Path,
    cam_key: str,
) -> list[tuple[Path, int, int]]:
    """Build a cumulative frame map for a camera's video files in a source dataset.

    Returns list of (video_path, cumulative_start, num_frames).
    """
    import decord

    cam_dir = ds_path / "videos" / cam_key
    if not cam_dir.exists():
        return []

    entries = []
    cum_start = 0
    for chunk_dir in sorted(cam_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        for vf in sorted(chunk_dir.glob("file-*.mp4")):
            vr = decord.VideoReader(str(vf))
            n_frames = len(vr)
            entries.append((vf, cum_start, n_frames))
            cum_start += n_frames
    return entries


def extract_episode_video(
    frame_map: list[tuple[Path, int, int]],
    global_start: int,
    num_frames: int,
    output_path: Path,
    target_width: int,
    target_height: int,
    fps: int = 30,
) -> bool:
    """Extract frames for one episode from source video(s) and write to output.

    Uses decord to read exact frames, pipes raw frames to ffmpeg for encoding.
    Handles episodes that span multiple source video files.

    Args:
        frame_map: List of (video_path, cumulative_start, num_frames) for source.
        global_start: First frame index (global) for this episode.
        num_frames: Number of frames in this episode.
        output_path: Output video file path.
        target_width: Target width for output.
        target_height: Target height for output.
        fps: Output FPS.

    Returns True on success.
    """
    import decord

    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_end = global_start + num_frames

    # Collect frames from source video(s)
    all_frames = []
    for video_path, cum_start, file_nframes in frame_map:
        cum_end = cum_start + file_nframes

        # Skip files that don't overlap with our range
        if cum_end <= global_start or cum_start >= global_end:
            continue

        # Compute local frame range within this file
        local_start = max(0, global_start - cum_start)
        local_end = min(file_nframes, global_end - cum_start)
        indices = list(range(local_start, local_end))

        if not indices:
            continue

        vr = decord.VideoReader(str(video_path))
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
        all_frames.append(frames)

    if not all_frames:
        log.error("No frames found for episode at global_start=%d, num_frames=%d", global_start, num_frames)
        return False

    frames = np.concatenate(all_frames, axis=0) if len(all_frames) > 1 else all_frames[0]

    if len(frames) != num_frames:
        log.warning(
            "Frame count mismatch: expected %d, got %d (global_start=%d)",
            num_frames, len(frames), global_start,
        )

    # Pipe raw frames to ffmpeg for encoding with resize
    src_h, src_w = frames.shape[1], frames.shape[2]

    # Build ffmpeg filter: scale + center-crop to target resolution
    vf = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{src_w}x{src_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=frames.tobytes(),
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            log.error("ffmpeg failed for %s: %s", output_path, proc.stderr[-500:].decode())
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out for %s", output_path)
        return False
    except Exception as e:
        log.error("ffmpeg error for %s: %s", output_path, e)
        return False


def _extract_episode_worker(args: tuple) -> tuple[str, bool]:
    """Worker for parallel per-episode video extraction."""
    frame_map_serialized, global_start, num_frames, output_path, tw, th, fps = args
    # Deserialize frame_map (can't pickle Path objects in some setups)
    frame_map = [(Path(p), cs, nf) for p, cs, nf in frame_map_serialized]
    ok = extract_episode_video(frame_map, global_start, num_frames, Path(output_path), tw, th, fps)
    return str(output_path), ok


def _write_frames_to_video(
    frames: np.ndarray,
    output_path: Path,
    target_width: int,
    target_height: int,
    fps: int = 30,
    timeout: int = 600,
) -> bool:
    """Write a numpy frame array to a video file via ffmpeg pipe.

    Args:
        frames: (N, H, W, 3) uint8 array.
        output_path: Output .mp4 path.
        target_width, target_height: Target resolution.
        fps: Output FPS.
        timeout: ffmpeg timeout in seconds.

    Returns True on success.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src_h, src_w = frames.shape[1], frames.shape[2]

    vf = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{src_w}x{src_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd, input=frames.tobytes(),
            capture_output=True, timeout=timeout,
        )
        if proc.returncode != 0:
            log.error("ffmpeg failed for %s: %s", output_path, proc.stderr[-500:].decode())
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out for %s", output_path)
        return False
    except Exception as e:
        log.error("ffmpeg error for %s: %s", output_path, e)
        return False


def batch_split_source_video(
    source_video_path: Path,
    episode_splits: list[tuple[int, int, Path]],
    target_width: int,
    target_height: int,
    fps: int = 30,
    resume: bool = False,
) -> tuple[int, int]:
    """Read one source video file once, split into per-episode output videos.

    Args:
        source_video_path: Path to the source .mp4 file.
        episode_splits: List of (local_frame_start, num_frames, output_path) for
            each episode within this video file.
        target_width, target_height: Target output resolution.
        fps: Output FPS.
        resume: Skip already-existing output files.

    Returns (num_success, num_failures).
    """
    import decord

    # Filter out already-done episodes if resuming
    if resume:
        episode_splits = [
            (start, nf, out) for start, nf, out in episode_splits
            if not out.exists()
        ]

    if not episode_splits:
        return 0, 0

    # Read entire source video once (sequential read — network friendly)
    try:
        vr = decord.VideoReader(str(source_video_path))
    except Exception as e:
        log.error("Failed to open video %s: %s", source_video_path, e)
        return 0, len(episode_splits)

    total_video_frames = len(vr)
    successes = 0
    failures = 0

    for local_start, num_frames, output_path in episode_splits:
        local_end = min(local_start + num_frames, total_video_frames)
        actual_frames = local_end - local_start

        if actual_frames <= 0:
            log.warning("No frames for %s (start=%d, total=%d)", output_path.name, local_start, total_video_frames)
            failures += 1
            continue

        try:
            indices = list(range(local_start, local_end))
            frames = vr.get_batch(indices).asnumpy()
        except Exception as e:
            log.error("Failed to read frames %d-%d from %s: %s", local_start, local_end, source_video_path, e)
            failures += 1
            continue

        ok = _write_frames_to_video(frames, output_path, target_width, target_height, fps)
        if ok:
            successes += 1
        else:
            failures += 1

        # Free memory
        del frames

    return successes, failures


def _batch_split_worker(args: tuple) -> tuple[int, int]:
    """Worker for parallel batch video splitting."""
    src_path, episode_splits_serialized, tw, th, fps, resume = args
    episode_splits = [
        (start, nf, Path(out)) for start, nf, out in episode_splits_serialized
    ]
    return batch_split_source_video(
        Path(src_path), episode_splits, tw, th, fps, resume,
    )


# ===================================================================
# Phase 6: Metadata Generation
# ===================================================================

def build_info_json(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    target_width: int,
    target_height: int,
) -> dict:
    """Build merged info.json."""
    return {
        "codebase_version": "v3.0",
        "robot_type": "bi_dk1_follower",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": 0,  # updated after writing
        "video_files_size_in_mb": 0,  # updated after writing
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": [
                    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
                    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
                    "left_gripper.pos",
                    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
                    "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
                    "right_gripper.pos",
                ],
                "shape": [TARGET_ACTION_DIM],
            },
            "observation.state": {
                "dtype": "float32",
                "names": [
                    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
                    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
                    "left_gripper.pos",
                    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
                    "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
                    "right_gripper.pos",
                ],
                "shape": [TARGET_STATE_DIM],
            },
            **{
                f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": [target_height, target_width, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": target_height,
                        "video.width": target_width,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": fps,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
                for cam in TARGET_CAM_NAMES
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def build_modality_json() -> dict:
    """Build DreamZero-compatible modality.json for DK-1."""
    modality: dict = {"state": {}, "action": {}, "video": {}, "annotation": {}}

    for name, (start, end) in STATE_KEYS.items():
        modality["state"][name] = {
            "original_key": "observation.state",
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    for name, (start, end) in ACTION_KEYS.items():
        modality["action"][name] = {
            "original_key": "action",
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    for cam in TARGET_CAM_NAMES:
        modality["video"][cam] = {"original_key": f"observation.images.{cam}"}

    modality["annotation"]["task"] = {"original_key": "task_index"}

    return modality


def compute_stats(parquet_paths: list[Path], columns: list[str]) -> dict:
    """Compute mean/std/min/max/q01/q99 for numeric columns across all episodes."""
    all_data: dict[str, list] = {col: [] for col in columns}
    for pp in tqdm(parquet_paths, desc="Computing stats"):
        df = pd.read_parquet(pp)
        for col in columns:
            if col not in df.columns:
                continue
            arr = np.stack(df[col].values)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            all_data[col].append(arr)

    stats = {}
    for col in columns:
        if not all_data[col]:
            continue
        data = np.concatenate(all_data[col], axis=0).astype(np.float64)
        stats[col] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def compute_relative_stats(
    parquet_paths: list[Path],
    modality: dict,
    relative_action_keys: list[str],
    action_horizon: int = 24,
) -> dict:
    """Compute relative-action statistics: (action[t+d] - state[t]) for each key."""
    stats: dict = {}
    for rel_key in relative_action_keys:
        if rel_key not in modality["action"] or rel_key not in modality["state"]:
            log.warning("Relative action key '%s' missing from action or state modality, skipping", rel_key)
            continue

        action_meta = modality["action"][rel_key]
        state_meta = modality["state"][rel_key]

        all_relative = []
        for pp in tqdm(parquet_paths, desc=f"Relative stats [{rel_key}]"):
            df = pd.read_parquet(pp)
            action_col = action_meta["original_key"]
            state_col = state_meta["original_key"]
            if action_col not in df.columns or state_col not in df.columns:
                continue

            action_data = np.stack(df[action_col].values).astype(np.float64)
            state_data = np.stack(df[state_col].values).astype(np.float64)
            if action_data.ndim == 1:
                action_data = action_data.reshape(-1, 1)
            if state_data.ndim == 1:
                state_data = state_data.reshape(-1, 1)

            a_start, a_end = action_meta["start"], action_meta["end"]
            s_start, s_end = state_meta["start"], state_meta["end"]
            action_slice = action_data[:, a_start:a_end]
            state_slice = state_data[:, s_start:s_end]

            traj_len = len(df)
            usable = traj_len - action_horizon
            for i in range(max(usable, 0)):
                ref_state = state_slice[i]
                chunk_end = min(i + action_horizon, traj_len)
                actions = action_slice[i:chunk_end]
                relative = actions - ref_state
                all_relative.extend(relative)

        if not all_relative:
            log.warning("No relative actions computed for '%s'", rel_key)
            continue

        data = np.array(all_relative)
        stats[rel_key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    return stats


# ===================================================================
# Phase 7: Verification
# ===================================================================

def verify_dataset(output_path: Path, total_episodes: int, target_width: int, target_height: int) -> list[str]:
    """Verify the merged dataset. Returns list of error strings."""
    errors = []
    frame_count_mismatches = 0

    for ep_idx in tqdm(range(total_episodes), desc="Verifying"):
        chunk_idx = ep_idx // CHUNKS_SIZE
        file_idx = ep_idx % CHUNKS_SIZE

        # Check parquet exists
        pq_path = output_path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        if not pq_path.exists():
            errors.append(f"missing parquet: ep={ep_idx}")
            continue

        df = pd.read_parquet(pq_path)
        n_rows = len(df)

        # Check videos
        for cam in TARGET_CAM_NAMES:
            vid_path = (
                output_path / f"videos/observation.images.{cam}"
                / f"chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            )
            if not vid_path.exists():
                errors.append(f"missing video: ep={ep_idx} cam={cam}")
                continue

            vinfo = get_video_info(vid_path)
            if vinfo is None:
                errors.append(f"unreadable video: ep={ep_idx} cam={cam}")
                continue

            if vinfo["width"] != target_width or vinfo["height"] != target_height:
                errors.append(
                    f"wrong resolution: ep={ep_idx} cam={cam} "
                    f"got {vinfo['width']}x{vinfo['height']}"
                )

            # Frame count check: exact match expected, tolerance +-1 for
            # ffprobe reporting quirks with h264 (last frame timing).
            diff = abs(vinfo["nb_frames"] - n_rows)
            if diff > 1:
                frame_count_mismatches += 1
                if frame_count_mismatches <= 10:
                    errors.append(
                        f"frame count mismatch: ep={ep_idx} cam={cam} "
                        f"video={vinfo['nb_frames']} parquet={n_rows}"
                    )
            elif diff == 1:
                log.debug(
                    "frame count off-by-1: ep=%d cam=%s video=%d parquet=%d",
                    ep_idx, cam, vinfo["nb_frames"], n_rows,
                )

    if frame_count_mismatches > 10:
        errors.append(f"... and {frame_count_mismatches - 10} more frame count mismatches")

    return errors


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge DK-1 datasets into a single LeRobot v3 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default="/workspace/data/dk1",
                        help="Root directory containing individual DK-1 datasets")
    parser.add_argument("--output", type=str, default="/workspace/data/dk1-merge-march2026",
                        help="Output directory for merged dataset")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--ffmpeg-workers", type=int, default=8,
                        help="Number of parallel ffmpeg workers for video re-encoding")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-written files (for resuming interrupted runs)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification on existing output, no writing")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip video re-encoding (for faster parquet-only testing)")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip stats computation")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    if not data_root.exists():
        log.error("Data root does not exist: %s", data_root)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1: Discovery & Filtering
    # ------------------------------------------------------------------
    log.info("Phase 1: Discovering datasets in %s", data_root)
    included, excluded = discover_datasets(data_root)

    log.info("Included: %d datasets", len(included))
    for ds in included:
        log.info("  + %-45s  %d files  %d frames  action=%dD  overview=%s",
                 ds["name"], len(ds["parquet_files"]), ds["total_frames"],
                 ds["action_dim"], ds["cam_info"]["overview"])

    log.info("Excluded: %d datasets", len(excluded))
    for ex in excluded:
        log.info("  - %-45s  reason: %s", ex["name"], ex["reason"])

    if not included:
        log.error("No datasets to merge!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Validation
    # ------------------------------------------------------------------
    log.info("\nPhase 2: Validating datasets")
    critical_errors = False
    for ds in included:
        errors = validate_dataset(ds)
        if errors:
            log.warning("  %s: %d errors", ds["name"], len(errors))
            for e in errors[:5]:
                log.warning("    %s", e)
            if len(errors) > 5:
                log.warning("    ... and %d more", len(errors) - 5)
            critical_errors = True

    if critical_errors:
        log.warning("Validation found errors — proceeding anyway (videos may be missing for some episodes)")

    # ------------------------------------------------------------------
    # Verify-only mode
    # ------------------------------------------------------------------
    if args.verify_only:
        # Load info.json from existing output
        info_path = output_path / "meta" / "info.json"
        if not info_path.exists():
            log.error("No info.json at %s — nothing to verify", info_path)
            sys.exit(1)
        info = json.load(open(info_path))
        total_episodes = info["total_episodes"]
        log.info("\nPhase 7: Verifying %d episodes", total_episodes)
        errors = verify_dataset(output_path, total_episodes, args.target_width, args.target_height)
        if errors:
            log.error("Verification found %d errors:", len(errors))
            for e in errors:
                log.error("  %s", e)
            sys.exit(1)
        else:
            log.info("Verification passed!")
        return

    # ------------------------------------------------------------------
    # Phase 3: Task Merging
    # ------------------------------------------------------------------
    log.info("\nPhase 3: Merging tasks")
    tasks, task_remap = merge_tasks(included)
    log.info("  %d unique tasks across all datasets", len(tasks))

    # ------------------------------------------------------------------
    # Phase 4 + 5: Per-episode parquet split + video extraction
    # ------------------------------------------------------------------
    log.info("\nPhase 4+5: Splitting per-episode (parquet + video)")
    output_path.mkdir(parents=True, exist_ok=True)

    global_episode_index = 0
    global_frame_index = 0
    episodes_meta: list[dict] = []
    video_jobs: list[tuple] = []
    skipped_videos = 0
    skipped_parquets = 0

    for ds in tqdm(included, desc="Datasets"):
        ds_task_remap = task_remap[ds["name"]]
        overview_cam = ds["cam_info"]["overview"]

        # Build video frame maps for this dataset (one per camera)
        cam_mapping = {
            f"observation.images.{overview_cam}": "observation.images.top",
            "observation.images.left_wrist": "observation.images.left_wrist",
            "observation.images.right_wrist": "observation.images.right_wrist",
        }
        cam_frame_maps: dict[str, list[tuple[Path, int, int]]] = {}
        if not args.skip_videos:
            for src_cam_key in cam_mapping:
                cam_frame_maps[src_cam_key] = build_source_video_frame_map(
                    ds["path"], src_cam_key,
                )

        # Track cumulative row offset across parquet files within this dataset.
        # This is the correct video frame offset — it counts rows sequentially
        # rather than relying on the parquet "index" column which may not start at 0.
        ds_row_offset = 0

        for pf in ds["parquet_files"]:
            # Split this parquet into individual episodes
            episodes = split_parquet_by_episode(
                source_parquet=pf,
                action_dim=ds["action_dim"],
                fps=ds["fps"],
                task_remap=ds_task_remap,
            )

            # Count total rows in this parquet file for advancing ds_row_offset
            file_total_rows = sum(len(ep_df) for ep_df, _, _ in episodes)

            for ep_df, orig_ep_idx, row_offset_in_file in episodes:
                n_frames = len(ep_df)
                if n_frames < 10:  # Skip tiny episodes
                    continue

                chunk_idx = global_episode_index // CHUNKS_SIZE
                file_idx = global_episode_index % CHUNKS_SIZE
                out_pq = output_path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"

                # Write parquet
                if args.resume and out_pq.exists():
                    skipped_parquets += 1
                    n_frames = len(pd.read_parquet(out_pq, columns=["frame_index"]))
                else:
                    n_frames = write_episode_parquet(
                        ep_df, out_pq, global_episode_index, global_frame_index, ds["fps"],
                    )

                # Collect task strings for this episode
                ep_task_indices = set()
                if "task_index" in ep_df.columns:
                    for tidx in ep_df["task_index"].unique():
                        ep_task_indices.add(int(tidx))
                if not ep_task_indices:
                    ep_task_indices.add(0)
                ep_tasks = [t["task"] for t in tasks if t["task_index"] in ep_task_indices]
                if not ep_tasks:
                    ep_tasks = [tasks[0]["task"] if tasks else ""]

                episodes_meta.append({
                    "episode_index": global_episode_index,
                    "tasks": ep_tasks,
                    "length": n_frames,
                })

                # Queue video extraction jobs
                if not args.skip_videos:
                    # Video frame offset = cumulative rows from prior parquet files
                    # + row offset of this episode within the current file
                    video_global_start = ds_row_offset + row_offset_in_file

                    for src_cam_key, dst_cam_key in cam_mapping.items():
                        dst_video = (
                            output_path / "videos" / dst_cam_key
                            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                        )

                        if args.resume and dst_video.exists():
                            skipped_videos += 1
                            continue

                        frame_map = cam_frame_maps.get(src_cam_key, [])
                        if not frame_map:
                            continue

                        # Serialize frame_map for multiprocessing
                        fm_serialized = [(str(p), cs, nf) for p, cs, nf in frame_map]
                        video_jobs.append((
                            fm_serialized, video_global_start, n_frames,
                            str(dst_video), args.target_width, args.target_height, FPS,
                        ))

                global_frame_index += n_frames
                global_episode_index += 1

            ds_row_offset += file_total_rows

    total_episodes = global_episode_index
    total_frames = global_frame_index
    log.info("  Wrote %d episode parquets, %d total frames", total_episodes, total_frames)
    if skipped_parquets:
        log.info("  Skipped %d existing parquets (--resume)", skipped_parquets)

    # Execute video jobs using batch splitting (one read per source video file)
    if not args.skip_videos:
        # Group video_jobs by source video file for batch processing
        # video_jobs contains: (fm_serialized, video_global_start, n_frames, dst_path, tw, th, fps)
        # We need to reorganize: for each source video file → list of (local_start, n_frames, output_path)
        from collections import defaultdict
        batch_jobs: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

        crossfile_jobs = []  # fallback jobs for episodes spanning multiple files

        for fm_serialized, video_global_start, n_frames, dst_path, tw, th, fps_val in video_jobs:
            # Find which source video file(s) contain these frames
            global_end = video_global_start + n_frames
            matched = False
            for src_path_str, cum_start, file_nframes in fm_serialized:
                cum_end = cum_start + file_nframes
                if video_global_start >= cum_start and video_global_start < cum_end:
                    local_start = video_global_start - cum_start
                    if local_start + n_frames <= file_nframes:
                        # Episode fits entirely in this file — use batch path
                        batch_jobs[src_path_str].append((local_start, n_frames, dst_path))
                    else:
                        # Episode spans files — queue for sequential extraction
                        log.warning("Episode spans video files at %s frame %d — using fallback", src_path_str, video_global_start)
                        crossfile_jobs.append((fm_serialized, video_global_start, n_frames, dst_path, tw, th, fps_val))
                    matched = True
                    break

            if not matched:
                log.error("No source video found for frame %d in %s", video_global_start, dst_path)

        # Process cross-file episodes sequentially (rare case)
        for fm_serialized, video_global_start, n_frames, dst_path, tw, th, fps_val in crossfile_jobs:
            frame_map = [(Path(p), cs, nf) for p, cs, nf in fm_serialized]
            extract_episode_video(frame_map, video_global_start, n_frames, Path(dst_path), tw, th, fps_val)

        total_batch_episodes = sum(len(eps) for eps in batch_jobs.values())
        log.info("  Batch splitting: %d source videos → %d episode videos (%d workers)",
                 len(batch_jobs), total_batch_episodes, args.ffmpeg_workers)

        # Submit batch jobs (one per source video file)
        batch_job_list = []
        for src_path, episode_splits in batch_jobs.items():
            serialized_splits = [(s, n, str(o)) for s, n, o in episode_splits]
            batch_job_list.append((
                src_path, serialized_splits,
                args.target_width, args.target_height, FPS, args.resume,
            ))

        total_successes = 0
        total_failures = 0
        with ProcessPoolExecutor(max_workers=args.ffmpeg_workers) as executor:
            futures = {executor.submit(_batch_split_worker, job): job for job in batch_job_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Source videos"):
                successes, failures = future.result()
                total_successes += successes
                total_failures += failures

        log.info("  Video extraction: %d successes, %d failures", total_successes, total_failures)
    else:
        log.info("\nPhase 5: Skipped (--skip-videos)")

    # ------------------------------------------------------------------
    # Phase 6: Metadata Generation
    # ------------------------------------------------------------------
    log.info("\nPhase 6: Generating metadata")
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json
    info = build_info_json(total_episodes, total_frames, len(tasks), FPS, args.target_width, args.target_height)
    # Calculate actual file sizes
    data_size = sum(f.stat().st_size for f in output_path.rglob("data/**/*.parquet")) / (1024 * 1024)
    video_size = sum(f.stat().st_size for f in output_path.rglob("videos/**/*.mp4")) / (1024 * 1024)
    info["data_files_size_in_mb"] = round(data_size)
    info["video_files_size_in_mb"] = round(video_size)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    log.info("  Wrote info.json")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    log.info("  Wrote episodes.jsonl (%d episodes)", len(episodes_meta))

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    log.info("  Wrote tasks.jsonl (%d tasks)", len(tasks))

    # tasks.parquet
    tasks_df = pd.DataFrame({"task": [t["task"] for t in tasks], "task_index": [t["task_index"] for t in tasks]})
    tasks_df = tasks_df.set_index("task")
    tasks_df.to_parquet(meta_dir / "tasks.parquet")
    log.info("  Wrote tasks.parquet")

    # modality.json
    modality = build_modality_json()
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    log.info("  Wrote modality.json")

    # embodiment.json
    embodiment = {"robot_type": "dk1", "embodiment_tag": "dk1"}
    with open(meta_dir / "embodiment.json", "w") as f:
        json.dump(embodiment, f, indent=4)
    log.info("  Wrote embodiment.json")

    # stats.json
    if not args.skip_stats:
        log.info("  Computing stats...")
        merged_parquets = sorted(output_path.rglob("data/**/*.parquet"))
        stats = compute_stats(merged_parquets, ["observation.state", "action"])
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        log.info("  Wrote stats.json")

        # relative_stats_dreamzero.json
        log.info("  Computing relative stats...")
        rel_stats = compute_relative_stats(merged_parquets, modality, RELATIVE_ACTION_KEYS)
        if rel_stats:
            with open(meta_dir / "relative_stats_dreamzero.json", "w") as f:
                json.dump(rel_stats, f, indent=4)
            log.info("  Wrote relative_stats_dreamzero.json (%d keys)", len(rel_stats))
    else:
        log.info("  Skipping stats (--skip-stats)")

    # README.md - provenance
    readme_lines = [
        "# DK-1 Merged Dataset",
        "",
        f"Generated: 2026-03-18",
        f"Total episodes: {total_episodes}",
        f"Total frames: {total_frames}",
        f"Total tasks: {len(tasks)}",
        f"Resolution: {args.target_width}x{args.target_height}",
        f"Codec: h264",
        f"FPS: {FPS}",
        "",
        "## Source datasets",
        "",
    ]
    for ds in included:
        readme_lines.append(f"- {ds['name']} ({len(ds['parquet_files'])} episodes, {ds['total_frames']} frames)")
    readme_lines.append("")
    readme_lines.append("## Excluded datasets")
    readme_lines.append("")
    for ex in excluded:
        readme_lines.append(f"- {ex['name']}: {ex['reason']}")

    with open(output_path / "README.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    log.info("  Wrote README.md")

    # ------------------------------------------------------------------
    # Phase 7: Verification
    # ------------------------------------------------------------------
    log.info("\nPhase 7: Verification")
    errors = verify_dataset(output_path, total_episodes, args.target_width, args.target_height)
    if errors:
        log.warning("Verification found %d issues:", len(errors))
        for e in errors[:20]:
            log.warning("  %s", e)
    else:
        log.info("Verification passed!")

    # Summary
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  Output:    {output_path}")
    print(f"  Episodes:  {total_episodes}")
    print(f"  Frames:    {total_frames}")
    print(f"  Tasks:     {len(tasks)}")
    print(f"  Data size: {data_size:.0f} MB")
    print(f"  Video size: {video_size:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
