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
    # Duplicates: daily t-shirt datasets are a subset of trlc_tshirt_folding
    "Gongsta_dk1_2026-02-28",
    "Gongsta_dk1_2026-03-01",
    "Gongsta_dk1_2026-03-02",
    "Gongsta_dk1_2026-03-03",
    # Duplicates: 100% contained in pingpongmegamerge
    "qualiaadmin_pingpongred1",
    # Duplicates: 100% contained in FabianKerj_dualarm-pretrain-127ep
    "qualiaadmin_mandminbox",
    "qualiaadmin_plasticinbox50episodesimpedance",
    # Bad data: mixed camera streams across episodes (contradictory visual/action)
    "qualiaadmin_pingpongmegamerge",
    # Bad data: wrong camera streams for context & left_wrist (show operator)
    "qualiaadmin_spoon1",
    # No video files at all (incomplete upload)
    "Gongsta_dagger_dk1_tshirt_corrections",
    # Inconsistent camera mapping across episodes
    "apaszynska_merged_diffrenet_grips",
    "apaszynska_merged-slow-regrasp-200ep",
    # Different embodiment (top-down mount, inverted wrists)
    "apaszynska_merged-dataset-regrasp-200",
    "apaszynska_portafilter-to-grinder-regrasp-v6-with-sticker",
}

# Overview camera names → normalised name "top"
OVERVIEW_CAM_NAMES = {"top", "context", "base_0", "head", "camera_0"}
WRIST_CAMS = {"left_wrist", "right_wrist", "camera_1", "camera_2"}
TARGET_CAM_NAMES = ["head", "left_wrist", "right_wrist"]  # output camera order

# Source camera name → target camera name
CAM_NAME_REMAP = {
    "top": "head",
    "context": "head",
    "base_0": "head",
    "head": "head",
    "camera_0": "head",
    "left_wrist": "left_wrist",
    "camera_1": "right_wrist",
    "right_wrist": "right_wrist",
    "camera_2": "left_wrist",
}

DEFAULT_FPS = 30
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


def get_state_dim(info: dict) -> int:
    """Get state dimensionality from info.json."""
    features = info.get("features", {})
    state_feat = features.get("observation.state", {})
    shape = state_feat.get("shape", [0])
    return shape[0] if isinstance(shape, list) else shape


# Indices to extract 14D joint positions from 40D state (pos/vel/torque interleaved)
STATE_40D_POS_INDICES = [0, 3, 6, 9, 12, 15, 18, 20, 23, 26, 29, 32, 35, 38]


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

        state_dim = get_state_dim(info)

        included.append({
            "name": name,
            "path": ds_dir,
            "info": info,
            "cam_info": cam_info,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "parquet_files": parquet_files,
            "total_frames": total_frames,
            "fps": info.get("fps", DEFAULT_FPS),
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

        for cam_name in [overview_cam] + cam_info["wrists"]:
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
    state_dim: int,
    fps: float,
    task_remap: dict[int, int],
) -> list[tuple[pd.DataFrame, int, int]]:
    """Split a source parquet file into per-episode DataFrames.

    Returns list of (episode_df, original_episode_index, global_row_start_in_file).
    Each episode_df has action/state sliced, task_index remapped, but episode_index/index/
    frame_index/timestamp are NOT yet set (caller assigns global values).
    """
    df = pd.read_parquet(source_parquet)

    # Slice action to 14D if needed
    if action_dim > TARGET_ACTION_DIM:
        df["action"] = df["action"].apply(lambda a: np.array(a)[:TARGET_ACTION_DIM])

    # Slice state to 14D if needed (extract joint positions from pos/vel/torque layout)
    if state_dim > TARGET_STATE_DIM:
        if state_dim == 40:
            df["observation.state"] = df["observation.state"].apply(
                lambda s: np.array(s)[STATE_40D_POS_INDICES]
            )
        else:
            df["observation.state"] = df["observation.state"].apply(
                lambda s: np.array(s)[:TARGET_STATE_DIM]
            )

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


def _get_frame_count(video_path: Path) -> int:
    """Get frame count from a video file using ffprobe (no RAM usage).

    Uses the fast nb_frames metadata first (reliable for h264 mp4).
    Falls back to slow -count_frames only if metadata is unavailable.
    """
    # Fast path: read nb_frames from container metadata
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "csv=p=0", str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        count = int(result.stdout.strip())
        if count > 0:
            return count
    except (ValueError, AttributeError):
        pass
    except Exception as e:
        log.warning("ffprobe fast path failed for %s: %s", video_path, e)

    # Slow fallback: decode all frames to count them
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(video_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        return int(result.stdout.strip())
    except Exception as e:
        log.error("Failed to get frame count for %s: %s", video_path, e)
        return 0


def build_source_video_frame_map(
    ds_path: Path,
    cam_key: str,
) -> list[tuple[Path, int, int]]:
    """Build a cumulative frame map for a camera's video files in a source dataset.

    Uses ffprobe to count frames — no video data loaded into RAM.

    Returns list of (video_path, cumulative_start, num_frames).
    """
    cam_dir = ds_path / "videos" / cam_key
    if not cam_dir.exists():
        return []

    entries = []
    cum_start = 0
    for chunk_dir in sorted(cam_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        for vf in sorted(chunk_dir.glob("file-*.mp4")):
            n_frames = _get_frame_count(vf)
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
    """Split a source video into per-episode output videos using ffmpeg only.

    Uses ffmpeg seeking + frame count extraction — no decord, no loading frames
    into RAM. This keeps memory usage minimal even for very large source videos.

    Args:
        source_video_path: Path to the source .mp4 file.
        episode_splits: List of (local_frame_start, num_frames, output_path) for
            each episode within this video file.
        target_width, target_height: Target output resolution.
        fps: Output FPS.
        resume: Skip already-existing output files.

    Returns (num_success, num_failures).
    """
    # Filter out already-done episodes if resuming
    if resume:
        episode_splits = [
            (start, nf, out) for start, nf, out in episode_splits
            if not out.exists()
        ]

    if not episode_splits:
        return 0, 0

    successes = 0
    failures = 0

    vf = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}"
    )

    for local_start, num_frames, output_path in episode_splits:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ffmpeg to seek by timestamp and extract exact frame count.
        # Seeking by time is faster than frame-accurate seeking for large files.
        start_time = local_start / fps

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.6f}",
            "-i", str(source_video_path),
            "-frames:v", str(num_frames),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                log.error("ffmpeg failed for %s: %s", output_path, result.stderr[-500:])
                failures += 1
            else:
                successes += 1
        except subprocess.TimeoutExpired:
            log.error("ffmpeg timed out for %s", output_path)
            failures += 1
        except Exception as e:
            log.error("ffmpeg error for %s: %s", output_path, e)
            failures += 1

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
    # Phase 4: Collect all episodes (two-pass: collect, shuffle, write)
    # ------------------------------------------------------------------
    log.info("\nPhase 4: Collecting episodes from all datasets")
    output_path.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect all episode records without assigning global indices.
    # Uses episodes parquet metadata to locate each episode's video segment
    # (file_index + from_timestamp), which works for all dataset layouts.
    all_episodes: list[dict] = []

    for ds in tqdm(included, desc="Collecting"):
        ds_task_remap = task_remap[ds["name"]]
        overview_cam = ds["cam_info"]["overview"]

        # Build camera mapping (source cam name → target cam name)
        wrist_cams = ds["cam_info"]["wrists"]
        cam_mapping = {
            f"observation.images.{overview_cam}": "observation.images.head",
        }
        for wc in wrist_cams:
            target = CAM_NAME_REMAP.get(wc, wc)
            cam_mapping[f"observation.images.{wc}"] = f"observation.images.{target}"

        # Load episodes metadata (video file_index + from_timestamp per episode per camera)
        ep_video_meta: dict[int, dict] = {}  # episode_index -> {cam_key: {file_index, from_ts}}
        for ep_pq in sorted((ds["path"] / "meta" / "episodes").rglob("*.parquet")):
            ep_meta_df = pd.read_parquet(ep_pq)
            for _, row in ep_meta_df.iterrows():
                ep_i = int(row["episode_index"])
                cam_meta = {}
                for src_cam_key in cam_mapping:
                    fi_col = f"videos/{src_cam_key}/file_index"
                    ci_col = f"videos/{src_cam_key}/chunk_index"
                    ft_col = f"videos/{src_cam_key}/from_timestamp"
                    if fi_col in ep_meta_df.columns and ft_col in ep_meta_df.columns:
                        cam_meta[src_cam_key] = {
                            "file_index": int(row[fi_col]),
                            "chunk_index": int(row[ci_col]) if ci_col in ep_meta_df.columns else 0,
                            "from_timestamp": float(row[ft_col]),
                        }
                ep_video_meta[ep_i] = cam_meta

        has_video_meta = len(ep_video_meta) > 0
        if not has_video_meta:
            log.warning("  %s: no episodes parquet — video mapping may be unreliable", ds["name"])

        for pf in ds["parquet_files"]:
            episodes = split_parquet_by_episode(
                source_parquet=pf,
                action_dim=ds["action_dim"],
                state_dim=ds["state_dim"],
                fps=ds["fps"],
                task_remap=ds_task_remap,
            )

            for ep_df, orig_ep_idx, row_offset_in_file in episodes:
                n_frames = len(ep_df)
                if n_frames < 10:
                    continue

                # Collect task strings
                ep_task_indices = set()
                if "task_index" in ep_df.columns:
                    for tidx in ep_df["task_index"].unique():
                        ep_task_indices.add(int(tidx))
                if not ep_task_indices:
                    ep_task_indices.add(0)
                ep_tasks = [t["task"] for t in tasks if t["task_index"] in ep_task_indices]
                if not ep_tasks:
                    ep_tasks = [tasks[0]["task"] if tasks else ""]

                # Build video source info using episodes metadata
                video_sources = []
                if not args.skip_videos:
                    cam_meta = ep_video_meta.get(orig_ep_idx, {})
                    for src_cam_key, dst_cam_key in cam_mapping.items():
                        meta = cam_meta.get(src_cam_key)
                        if meta is not None:
                            # Use metadata: exact video file + timestamp
                            chunk_i = meta["chunk_index"]
                            file_i = meta["file_index"]
                            from_ts = meta["from_timestamp"]
                            src_video = (
                                ds["path"] / "videos" / src_cam_key
                                / f"chunk-{chunk_i:03d}" / f"file-{file_i:03d}.mp4"
                            )
                        else:
                            # Fallback: assume parquet file maps to same-named video file
                            src_video = (
                                ds["path"] / "videos" / src_cam_key
                                / pf.parent.name / f"{pf.stem}.mp4"
                            )
                            from_ts = 0.0

                        if not src_video.exists():
                            log.warning("Missing source video: %s (ep %d)", src_video, orig_ep_idx)
                            continue

                        video_sources.append({
                            "src_video": str(src_video),
                            "dst_cam_key": dst_cam_key,
                            "from_timestamp": from_ts,
                            "n_frames": n_frames,
                            "fps": ds["fps"],
                        })

                all_episodes.append({
                    "ep_df": ep_df,
                    "tasks": ep_tasks,
                    "fps": ds["fps"],
                    "n_frames": n_frames,
                    "video_sources": video_sources,
                    "source_dataset": ds["name"],
                })

    log.info("  Collected %d episodes from %d datasets", len(all_episodes), len(included))

    # Shuffle episodes for better training diversity
    import random
    random.seed(42)
    random.shuffle(all_episodes)
    log.info("  Shuffled episode order (seed=42)")

    # ------------------------------------------------------------------
    # Phase 5: Write shuffled episodes (parquet + video)
    # ------------------------------------------------------------------
    log.info("\nPhase 5: Writing shuffled episodes (parquet + video)")

    global_episode_index = 0
    global_frame_index = 0
    episodes_meta: list[dict] = []
    video_jobs: list[tuple] = []
    skipped_videos = 0
    skipped_parquets = 0

    for ep_record in all_episodes:
        ep_df = ep_record["ep_df"]
        n_frames = ep_record["n_frames"]
        ep_fps = ep_record["fps"]

        chunk_idx = global_episode_index // CHUNKS_SIZE
        file_idx = global_episode_index % CHUNKS_SIZE
        out_pq = output_path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"

        # Write parquet
        if args.resume and out_pq.exists():
            skipped_parquets += 1
            n_frames = len(pd.read_parquet(out_pq, columns=["frame_index"]))
        else:
            n_frames = write_episode_parquet(
                ep_df, out_pq, global_episode_index, global_frame_index, ep_fps,
            )

        episodes_meta.append({
            "episode_index": global_episode_index,
            "tasks": ep_record["tasks"],
            "length": n_frames,
            "fps": ep_fps,
        })

        # Queue video extraction jobs — each source has src_video + from_timestamp
        for vs in ep_record["video_sources"]:
            dst_video = (
                output_path / "videos" / vs["dst_cam_key"]
                / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            )

            if args.resume and dst_video.exists():
                skipped_videos += 1
                continue

            # All video sources use the same format: seek to from_timestamp, extract n_frames
            frame_offset = round(vs["from_timestamp"] * vs["fps"])
            fm_serialized = [(vs["src_video"], 0, frame_offset + n_frames + 100)]
            video_jobs.append((
                fm_serialized, frame_offset, n_frames,
                str(dst_video), args.target_width, args.target_height, vs["fps"],
            ))

        global_frame_index += n_frames
        global_episode_index += 1

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
        # Group jobs by source video file for batch processing.
        # All jobs are now single-file with a local frame offset (from episodes metadata).
        batch_jobs: dict[str, tuple[list[tuple[int, int, str]], int]] = {}

        for fm_serialized, frame_offset, n_frames, dst_path, tw, th, fps_val in video_jobs:
            src_path_str = fm_serialized[0][0]  # single source file
            if src_path_str not in batch_jobs:
                batch_jobs[src_path_str] = ([], fps_val)
            batch_jobs[src_path_str][0].append((frame_offset, n_frames, dst_path))

        total_batch_episodes = sum(len(eps) for eps, _ in batch_jobs.values())
        log.info("  Batch splitting: %d source videos → %d episode videos (%d workers)",
                 len(batch_jobs), total_batch_episodes, args.ffmpeg_workers)

        # Submit batch jobs (one per source video file)
        batch_job_list = []
        for src_path, (episode_splits, fps_val) in batch_jobs.items():
            serialized_splits = [(s, n, str(o)) for s, n, o in episode_splits]
            batch_job_list.append((
                src_path, serialized_splits,
                args.target_width, args.target_height, fps_val, args.resume,
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
    # Use the most common fps across episodes for the top-level fps field
    from collections import Counter
    fps_counts = Counter(ep["fps"] for ep in episodes_meta)
    dominant_fps = fps_counts.most_common(1)[0][0]
    all_fps = sorted(fps_counts.keys())
    if len(all_fps) > 1:
        log.info("  Mixed FPS detected: %s (using %d for info.json)", dict(fps_counts), dominant_fps)
    info = build_info_json(total_episodes, total_frames, len(tasks), dominant_fps, args.target_width, args.target_height)
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

    # Collect all merged parquet paths (used by stats + HF metadata)
    merged_parquets = sorted(output_path.rglob("data/**/*.parquet"))

    # stats.json
    if not args.skip_stats:
        log.info("  Computing stats...")
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

    # README.md - HuggingFace dataset card
    from datetime import date
    today = date.today().isoformat()

    # Build source dataset table with HF repo links
    source_rows = []
    for ds in included:
        # Convert local name back to HF repo format (first _ becomes /)
        name = ds["name"]
        parts = name.split("_", 1)
        hf_repo = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else name
        source_rows.append(f"| [{hf_repo}](https://huggingface.co/datasets/{hf_repo}) |")

    readme = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# DK-1 Merged Dataset

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=andreaskoepf/dk1-merge-2026-03">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>

## Dataset Description

Merged and deduplicated DK-1 bimanual robot dataset. All source videos are re-encoded to {args.target_width}x{args.target_height} h264 with {TARGET_ACTION_DIM}D joint-space actions. Per-episode FPS is preserved from the source datasets.

- **Generated:** {today}
- **Total episodes:** {total_episodes:,}
- **Total frames:** {total_frames:,}
- **Total tasks:** {len(tasks)}
- **Resolution:** {args.target_width}x{args.target_height}
- **Codec:** h264
- **FPS:** {", ".join(str(f) for f in all_fps)} (per-episode, stored in episodes.jsonl)
- **Cameras:** {", ".join(TARGET_CAM_NAMES)}
- **Action dim:** {TARGET_ACTION_DIM} (6 joint + 1 gripper per arm)

## Source Datasets

| Dataset |
|---|
{chr(10).join(source_rows)}
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme)
    log.info("  Wrote README.md")

    # Add HuggingFace metadata to data parquets (needed for HF dataset viewer)
    import pyarrow.parquet as pq
    hf_parquet_meta = json.dumps({
        "info": {
            "features": {
                "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": TARGET_ACTION_DIM, "_type": "List"},
                "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": TARGET_STATE_DIM, "_type": "List"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        }
    }).encode("utf-8")

    log.info("  Adding HuggingFace metadata to data parquets...")
    for pf in tqdm(merged_parquets, desc="HF metadata"):
        table = pq.read_table(pf)
        existing = table.schema.metadata or {}
        existing[b"huggingface"] = hf_parquet_meta
        table = table.replace_schema_metadata(existing)
        pq.write_table(table, pf)

    # Episodes parquet (for HF dataset viewer — has per-episode stats and video mapping)
    log.info("  Generating episodes parquet...")
    ep_rows = []
    cumulative_index = 0
    cam_keys = [f"observation.images.{c}" for c in TARGET_CAM_NAMES]

    for ep in tqdm(episodes_meta, desc="Episodes parquet"):
        ep_idx = ep["episode_index"]
        length = ep["length"]
        ep_fps = ep.get("fps", dominant_fps)
        chunk_idx = ep_idx // CHUNKS_SIZE
        file_idx = ep_idx % CHUNKS_SIZE

        row = {
            "episode_index": ep_idx,
            "tasks": ep["tasks"],
            "length": length,
            "fps": ep_fps,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": cumulative_index,
            "dataset_to_index": cumulative_index + length,
        }
        for cam in cam_keys:
            row[f"videos/{cam}/chunk_index"] = chunk_idx
            row[f"videos/{cam}/file_index"] = file_idx
            row[f"videos/{cam}/from_timestamp"] = 0.0
            row[f"videos/{cam}/to_timestamp"] = (length - 1) / ep_fps

        # Per-episode stats
        pq_path = output_path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        df = pd.read_parquet(pq_path)
        for col in ["action", "observation.state"]:
            arr = np.stack(df[col].values).astype(np.float64)
            row[f"stats/{col}/min"] = np.min(arr, axis=0).tolist()
            row[f"stats/{col}/max"] = np.max(arr, axis=0).tolist()
            row[f"stats/{col}/mean"] = np.mean(arr, axis=0).tolist()
            row[f"stats/{col}/std"] = np.std(arr, axis=0).tolist()
            row[f"stats/{col}/count"] = [len(arr)]
            for q in [0.01, 0.10, 0.50, 0.90, 0.99]:
                qname = f"q{int(q*100):02d}"
                row[f"stats/{col}/{qname}"] = np.quantile(arr, q, axis=0).tolist()
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            arr = df[col].values.astype(np.float64)
            row[f"stats/{col}/min"] = [float(np.min(arr))]
            row[f"stats/{col}/max"] = [float(np.max(arr))]
            row[f"stats/{col}/mean"] = [float(np.mean(arr))]
            row[f"stats/{col}/std"] = [float(np.std(arr))]
            row[f"stats/{col}/count"] = [len(arr)]
            for q in [0.01, 0.10, 0.50, 0.90, 0.99]:
                qname = f"q{int(q*100):02d}"
                row[f"stats/{col}/{qname}"] = [float(np.quantile(arr, q))]
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        ep_rows.append(row)
        cumulative_index += length

    ep_df = pd.DataFrame(ep_rows)
    ep_out_dir = meta_dir / "episodes" / "chunk-000"
    ep_out_dir.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(ep_out_dir / "file-000.parquet", index=False)
    log.info("  Wrote episodes parquet (%d episodes, %d columns)", len(ep_df), len(ep_df.columns))

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
