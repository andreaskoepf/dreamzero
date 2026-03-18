#!/usr/bin/env python3
"""Compute global relative-action stats across all DK-1 datasets.

Produces a single ``relative_stats_dreamzero.json`` that can be shared by
every DK-1 sub-dataset via the ``relative_stats_path`` kwarg, ensuring
consistent normalization across the mixture.

Usage:
    python scripts/data/compute_dk1_global_relative_stats.py \
        --data_root /workspace/data/dk1 \
        --output /workspace/data/dk1/global_relative_stats_dreamzero.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DK-1 action / state layout  (must match modality.json for DK-1 datasets)
# ---------------------------------------------------------------------------
# These are the action keys we compute relative stats for.
# Each entry maps:  key_name -> (state_col, state_slice, action_col, action_slice)
#
# The slicing info is read from the first dataset's modality.json so that we
# don't hard-code column indices.
# ---------------------------------------------------------------------------

RELATIVE_ACTION_KEYS = [
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
]


def load_modality_meta(dataset_path: Path) -> dict:
    meta_path = dataset_path / "meta" / "modality.json"
    with open(meta_path) as f:
        return json.load(f)


def get_slice_info(modality_meta: dict, key: str, modality: str) -> tuple[str, int, int]:
    """Return (original_column, start, end) for a key within a modality."""
    mod = modality_meta[modality]
    entry = mod[key]
    return entry["original_key"], entry["start"], entry["end"]


def collect_relative_actions(
    dataset_path: Path,
    key: str,
    state_col: str,
    state_start: int,
    state_end: int,
    action_col: str,
    action_start: int,
    action_end: int,
    action_delta_indices: list[int] | None = None,
    max_episodes: int = 10000,
) -> np.ndarray:
    """Compute relative actions (action - ref_state) for one dataset & key.

    For each timestep t and each action horizon delta d in
    ``action_delta_indices``, the relative action is::

        action[t + d] - state[t]

    Correctly respects episode boundaries — relative actions are never
    computed across episodes within multi-episode parquet files.
    """
    if action_delta_indices is None:
        action_delta_indices = list(range(24))  # default DK-1 action horizon

    # Discover all parquet files
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        return np.empty((0,))
    parquet_paths = sorted(data_dir.rglob("*.parquet"))
    if not parquet_paths:
        return np.empty((0,))

    max_delta = max(action_delta_indices)
    all_relative = []
    seen_episodes = set()

    rng = np.random.default_rng(seed=42)

    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue

        if state_col not in df.columns or action_col not in df.columns:
            continue

        # Group by episode to avoid computing across episode boundaries
        if "episode_index" in df.columns:
            groups = df.groupby("episode_index")
        else:
            groups = [(0, df)]

        for ep_idx, ep_df in groups:
            ep_idx = int(ep_idx)
            if ep_idx in seen_episodes:
                continue
            seen_episodes.add(ep_idx)

            if len(seen_episodes) > max_episodes:
                break

            full_state = np.stack(ep_df[state_col].values)
            full_action = np.stack(ep_df[action_col].values)

            state_data = full_state[:, state_start:state_end]
            action_data = full_action[:, action_start:action_end]

            n = len(ep_df)
            usable_length = n - max_delta
            for i in range(max(usable_length, 0)):
                ref_state = state_data[i]  # reference = state at current step
                for d in action_delta_indices:
                    if i + d < n:
                        all_relative.append(action_data[i + d] - ref_state)

        if len(seen_episodes) > max_episodes:
            break

    if not all_relative:
        return np.empty((0,))
    return np.array(all_relative)


def main():
    parser = argparse.ArgumentParser(description="Compute global DK-1 relative action stats")
    parser.add_argument("--data_root", type=str, default="/workspace/data/dk1")
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/data/dk1/global_relative_stats_dreamzero.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific dataset names to include (default: all in data_root that are in the training config)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Data root not found: {data_root}"

    # Use the same datasets listed in dk1_relative.yaml
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = [
            d.name
            for d in sorted(data_root.iterdir())
            if d.is_dir() and (d / "meta" / "modality.json").exists()
        ]

    print(f"Found {len(dataset_names)} datasets under {data_root}")

    # Load modality metadata from first dataset to get column layout
    first_ds = data_root / dataset_names[0]
    modality_meta = load_modality_meta(first_ds)

    global_stats: dict[str, dict] = {}

    for key in RELATIVE_ACTION_KEYS:
        print(f"\n{'='*60}")
        print(f"Computing relative stats for: {key}")
        print(f"{'='*60}")

        state_col, state_start, state_end = get_slice_info(modality_meta, key, "state")
        action_col, action_start, action_end = get_slice_info(modality_meta, key, "action")
        print(f"  State: col={state_col}[{state_start}:{state_end}]")
        print(f"  Action: col={action_col}[{action_start}:{action_end}]")

        all_relative = []
        for ds_name in tqdm(dataset_names, desc=f"Datasets for {key}"):
            ds_path = data_root / ds_name
            # Verify this dataset has compatible modality layout
            try:
                ds_meta = load_modality_meta(ds_path)
                ds_state_col, ds_s0, ds_s1 = get_slice_info(ds_meta, key, "state")
                ds_action_col, ds_a0, ds_a1 = get_slice_info(ds_meta, key, "action")
                if (ds_state_col != state_col or ds_s0 != state_start or ds_s1 != state_end
                        or ds_action_col != action_col or ds_a0 != action_start or ds_a1 != action_end):
                    print(f"  WARNING: {ds_name} has different column layout for {key}, skipping")
                    continue
            except (KeyError, FileNotFoundError):
                print(f"  WARNING: {ds_name} missing modality info for {key}, skipping")
                continue

            rel = collect_relative_actions(
                ds_path, key,
                state_col, state_start, state_end,
                action_col, action_start, action_end,
            )
            if len(rel) > 0:
                all_relative.append(rel)
                print(f"  {ds_name}: {len(rel)} samples")

        if not all_relative:
            print(f"  WARNING: No data collected for {key}")
            continue

        combined = np.concatenate(all_relative, axis=0)
        print(f"  Total: {len(combined)} samples across {len(all_relative)} datasets")

        global_stats[key] = {
            "max": np.max(combined, axis=0).tolist(),
            "min": np.min(combined, axis=0).tolist(),
            "mean": np.mean(combined, axis=0).tolist(),
            "std": np.std(combined, axis=0).tolist(),
            "q01": np.quantile(combined, 0.01, axis=0).tolist(),
            "q99": np.quantile(combined, 0.99, axis=0).tolist(),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(global_stats, f, indent=4)
    print(f"\nGlobal relative stats saved to {output_path}")


if __name__ == "__main__":
    main()
