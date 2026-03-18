# DreamZero DK-1 Fine-tuning — Training Notes

## Current Run (2026-03-18)

**Config:** LoRA rank 4, 10,000 steps, batch 1×4 GPUs, lr=1e-5, DeepSpeed ZeRO-2
**Status:** Running (~step 1870 at time of writing)

### Performance Observations

| Metric | Value |
|---|---|
| Training step time | ~10–13 s/step (varies by rank) |
| Model forward time | ~3 s (GPU work: forward + backward + optimizer) |
| Data loading overhead | ~7 s/step (CPU-bound, the GPU idle gap) |
| Total CPU utilization | <9% during training |
| RAM usage | ~480 GB (steady state during training) |
| RAM limit | 704 GB (hard limit, machine has 1.5 TB but we're restricted) |
| Checkpoint size | ~87 GB (208 MB LoRA weights + ~87 GB DeepSpeed optimizer states) |
| Checkpoint interval | Every 1,000 steps |
| save_total_limit | 2 (keeps latest + one backup) |
| Disk usage | ~340 GB total (/workspace), ~160 GB free |
| Loss | Started at 0.097, currently ~0.07–0.09 (noisy) |
| ETA | ~45 hours total from start |

### Data Loading Bottleneck

With `dataloader_num_workers=0`, all data processing (video frame decoding from shard cache,
transforms, tokenization) happens in the main process. The GPU idles for ~7s between steps
while waiting for the next batch.

**Recommendation for future runs:** Set `dataloader_num_workers=2` and
`dataloader_persistent_workers=true`. With 2 workers × 4 GPUs = 8 worker processes,
each prefetching the next batch, GPU utilization should improve significantly.
Estimated RAM overhead: 8 workers × ~20 GB/shard = ~160 GB extra → ~640 GB total, still
within the 704 GB limit.

### Paper vs Our Config

The DreamZero paper post-training recipe uses:
- **Full fine-tuning** (not LoRA) — paper footnote: "We experimented with LoRA but found it
  led to suboptimal results"
- **50K steps per task** with all DiT blocks + state/action encoder/decoder trainable
- Text encoder, image encoder, VAE frozen
- Co-training with pre-training data to prevent catastrophic forgetting

Our current run uses LoRA rank 4 as a pipeline validation. If results are underwhelming,
next steps would be full fine-tuning with `train_architecture=full` (may need ZeRO-3).

---

## Code Changes for DK-1 Dataset Support

### Problem: LeRobot DROID Format Incompatibility

DK-1 datasets come in the LeRobot v3 DROID format which differs from what the GEAR/DreamZero
codebase expects:

1. **Multi-episode parquet files**: A single `file-000.parquet` contains data for many robot
   episodes (e.g., 290 episodes × ~450 rows each = 131K rows total). The codebase assumed
   1 episode = 1 file.

2. **Multi-episode video files**: Video files are also packed — e.g., 8 `.mp4` files containing
   all 290 episodes concatenated. The number of video files ≠ number of parquet files.

3. **Per-episode timestamps**: Each episode's `timestamp` column resets to 0.0, but the video
   file has cumulative timestamps. `get_frames_by_timestamps` would always return episode 0's
   frames for any episode.

4. **Mixed camera resolutions**: Different DK-1 datasets have different native resolutions
   per camera (800×600, 960×540, 640×480, 640×360, 1280×720). Different cameras within the
   same dataset can also differ.

### Solution: Episodes.jsonl Rewrite + Index-Based Video Loading

#### 1. Rewrite `episodes.jsonl` (one entry per parquet FILE)

Script: `scripts/data/fix_dk1_episodes_jsonl.py`

Instead of one episode per entry (which doesn't match the file structure), each entry now
represents one parquet FILE. `trajectory_id = file_index`, and `trajectory_length = total
rows in that file`.

#### 2. Index-based video loading (`lerobot_sharded.py`)

The parquet `index` column provides a global frame counter across all episodes/files
(0, 1, 2, ..., N). This is used instead of timestamps to load the correct video frames:

- **`_build_video_frame_map()`**: At init, scans all video files per camera with decord to
  build a cumulative frame count table: `[(file_path, cumulative_start, num_frames), ...]`.
  Results are cached to `meta/video_frame_map.json` to avoid re-scanning on every startup
  (was taking minutes across 4 GPU processes × 29 datasets).

- **`load_frames_by_global_indices()`**: Given global frame indices from the parquet `index`
  column, maps each to the correct video file + local frame offset, then extracts frames
  with `get_frames_by_indices` (decord). Handles frames spanning multiple video files.

- **`get_shard()`**: Modified to use index-based loading when `video_frame_map` is available.
  Falls back to timestamp-based loading for non-DROID datasets.

#### 3. Windowed video loading (RAM management)

The shard caching in `get_shard()` previously loaded ALL video frames for a trajectory into
RAM. For large single-file datasets (e.g., `qualiaadmin_pingpongmegamerge`: 131K frames =
271 GB per shard), this caused OOM.

Fix: Added `max_steps_per_trajectory` parameter. When a trajectory exceeds this limit,
a random contiguous window of `num_steps_per_shard` frames is loaded instead of the full
trajectory. The `__iter__` method filters `allowed_indices` to the window bounds.

With `num_steps_per_shard=10000`:
- Per shard: ~10K frames × 3 cameras × 640×360 × 3 bytes ≈ 20 GB
- 8 concurrent shards (4 GPUs × 2 shards each) = 160 GB video cache
- Total with model: ~310 GB — well under 704 GB limit

#### 4. Positional row slicing in `get_trajectory_data()`

The DROID parquet `episode_index` column doesn't correspond to `trajectory_id` (which is the
file index). Changed `get_trajectory_data()` to use precomputed positional row slices
(`shard_df_start_rows`, `shard_df_sizes`) instead of `episode_index` filtering.

#### 5. Mixed resolution handling (video transforms)

- **`VideoToTensor`**: Overrode `apply()` to process each camera view independently instead
  of concatenating all views (which fails when views have different H×W).

- **`VideoResolutionNormalize`**: Changed `apply()` to compute resize parameters dynamically
  from actual input resolution rather than relying on metadata (which may not match actual
  video dimensions). All views are normalized to 640×360 before the shared crop/resize pipeline.

- **`VideoCrop`**: Set explicit `height: 360, width: 640` in the DK-1 config to match the
  post-normalization resolution.

#### 6. Annotation key fix

DK-1 parquets use `task_index` (integer column) for task annotations, not `annotation.task`
(string column). Fixed `modality.json` for all 29 datasets to set
`original_key: "task_index"` so the language loader resolves task strings via `tasks.parquet`.

#### 7. DK-1 embodiment registration

- Added `DK1 = "dk1"` to `EmbodimentTag` enum
- Added `dk1: 33` to the projector index mapping (`base.yaml`)
- Added DK-1 text template to `DreamTransform` collate function (3-view bimanual robot
  description)

### Files Modified

| File | Changes |
|---|---|
| `groot/vla/data/dataset/lerobot_sharded.py` | Windowed loading, index-based video, video frame map, positional slicing |
| `groot/vla/data/dataset/lerobot.py` | (previous session: parquet/video path fixes) |
| `groot/vla/data/transform/video.py` | Per-view VideoToTensor, dynamic VideoResolutionNormalize, relaxed VideoCrop |
| `groot/vla/data/schema/embodiment_tags.py` | Added `DK1` tag |
| `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | DK-1 modality config, transforms, VideoCrop explicit dims |
| `groot/vla/configs/data/dreamzero/dk1_relative.yaml` | Dataset config with all DK-1 dataset paths |
| `groot/vla/configs/model/dreamzero/transform/base.yaml` | Added `dk1: 33` projector index |
| `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` | DK-1 text template in collate |
| `groot/vla/experiment/base.py` | Relaxed `save_total_limit >= 5` assertion |
| `scripts/train/dk1_training.sh` | Training launcher |
| `scripts/data/fix_dk1_episodes_jsonl.py` | Episodes.jsonl rewriter |
| `scripts/data/convert_lerobot_to_gear.py` | (previous session: LeRobot v3 support) |

### DK-1 Datasets (29 total, in `/workspace/data/dk1/`)

Mixed sources: qualiaadmin (ping pong, spoons, plastic cups), Zasha01 (cube transfer,
lego, packaging), Gongsta (various tasks, tshirt folding), dopaul (PCB placement),
FabianKerj (pretrain data), sfeduniak (DHL, towels).

Total frames: ~3.1M across all datasets. Resolutions vary per dataset and per camera.
All use 30 FPS, 3 cameras (top/base_0, left_wrist, right_wrist), bimanual 14-DoF
(6 joint + 1 gripper per arm).
