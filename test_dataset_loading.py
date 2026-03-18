#!/usr/bin/env python3
"""Quick test: instantiate the DK-1 dataset mixture without loading the model.
Usage: python test_dataset_loading.py
"""
import os
import sys
import torch

# Minimal distributed init (single process)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
torch.distributed.init_process_group(backend="gloo")

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "groot", "vla", "configs")

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(
        config_name="experiment",
        overrides=[
            "data=dreamzero/dk1_relative",
            "model=dreamzero/vla",
            "model/dreamzero/action_head=wan_flow_matching_action_tf",
            "model/dreamzero/transform=dreamzero_cotrain",
            "num_frames=33",
            "action_horizon=24",
            "num_views=3",
            "num_frame_per_block=2",
            "num_action_per_block=24",
            "num_state_per_block=1",
            "image_resolution_width=320",
            "image_resolution_height=176",
            "max_chunk_size=4",
            "frame_seqlen=880",
            "dk1_data_root=/workspace/data/dk1",
            "train_architecture=lora",
            "per_device_train_batch_size=1",
            "max_steps=10",
        ],
    )

print("Instantiating train_dataset...")
try:
    train_dataset = instantiate(cfg.train_dataset)
    print(f"SUCCESS: Dataset loaded with {len(train_dataset)} samples")
    print(f"  Datasets in mixture: {len(train_dataset.datasets)}")
    for i, ds in enumerate(train_dataset.datasets):
        print(f"  [{i}] {ds.dataset_name}: {len(ds.trajectory_ids)} episodes")

    # Test fetching one sample
    print("\nFetching sample 0...")
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")
    print("\nDATASET TEST PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
