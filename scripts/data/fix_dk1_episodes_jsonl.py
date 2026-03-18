#!/usr/bin/env python3
"""Fix DK-1 dataset episodes.jsonl files.

In the native LeRobot DROID format, episodes.jsonl has one entry per robot episode,
but parquet/video files pack multiple episodes per file. The GEAR/DreamZero code
expects one trajectory per file.

This script rewrites episodes.jsonl so each entry corresponds to one parquet FILE
(not one episode). Each "trajectory" becomes the entire contents of one parquet file.

Usage:
    python scripts/data/fix_dk1_episodes_jsonl.py /workspace/data/dk1
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd


def fix_dataset(dataset_path: Path) -> bool:
    """Fix episodes.jsonl for a single dataset. Returns True if modified."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    info_path = dataset_path / "meta" / "info.json"
    parquet_dir = dataset_path / "data" / "chunk-000"

    if not episodes_path.exists() or not parquet_dir.exists():
        return False

    # Read current episodes.jsonl
    with open(episodes_path) as f:
        old_episodes = [json.loads(line) for line in f]

    # Find all parquet files
    parquet_files = sorted(
        [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    )
    n_parquet = len(parquet_files)
    n_episodes = len(old_episodes)

    if n_parquet == n_episodes:
        print(f"  {dataset_path.name}: OK ({n_episodes} eps = {n_parquet} files)")
        return False

    print(f"  {dataset_path.name}: {n_episodes} eps → {n_parquet} files, rewriting...")

    # Build new episodes.jsonl: one entry per parquet file
    new_episodes = []
    total_frames = 0
    for file_idx, parquet_file in enumerate(parquet_files):
        parquet_path = parquet_dir / parquet_file
        df = pd.read_parquet(parquet_path, columns=["episode_index"])
        n_rows = len(df)
        # Collect all task strings from old episodes that fall in this file
        episode_indices = sorted(df["episode_index"].unique())
        tasks = set()
        for old_ep in old_episodes:
            if old_ep["episode_index"] in episode_indices:
                for t in old_ep.get("tasks", []):
                    tasks.add(t)
        if not tasks:
            tasks = {"Perform the default behavior."}

        new_episodes.append({
            "episode_index": file_idx,
            "tasks": sorted(tasks),
            "length": n_rows,
        })
        total_frames += n_rows

    # Backup old file
    backup_path = episodes_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        episodes_path.rename(backup_path)
    else:
        # Already backed up from a previous run
        pass

    # Write new episodes.jsonl
    with open(episodes_path, "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # Update info.json
    if info_path.exists():
        info = json.load(open(info_path))
        info["total_episodes"] = n_parquet
        info["total_frames"] = total_frames
        # Set chunks_size to be larger than n_parquet so all files are in chunk-000
        info["chunks_size"] = max(n_parquet, info.get("chunks_size", 1000))
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    print(f"    → {n_parquet} trajectories, {total_frames} total frames")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_dk1_episodes_jsonl.py <dk1_data_root>")
        sys.exit(1)

    dk1_root = Path(sys.argv[1])
    modified = 0
    for ds in sorted(os.listdir(dk1_root)):
        ds_path = dk1_root / ds
        if ds_path.is_dir() and (ds_path / "meta" / "episodes.jsonl").exists():
            if fix_dataset(ds_path):
                modified += 1

    print(f"\nDone. Modified {modified} datasets.")


if __name__ == "__main__":
    main()
