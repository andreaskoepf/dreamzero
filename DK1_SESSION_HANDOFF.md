# DreamZero DK-1 Fine-tuning — Session Handoff

> Read this file at the start of a new session to pick up exactly where we left off.

---

## Current Status (as of 2026-03-17 ~23:17)

**Smoke test (10 steps) is running in background.** PID group started at 23:17.

```bash
# Check if still running
ps aux | grep experiment.py | grep -v grep | wc -l  # should be 4
tail -80 /tmp/dk1_smoke.log
free -h  # watch RAM — should stay well under 704 GB
```

If the smoke test passed (loss printed, no crash), launch **full training**:
```bash
pkill -f "experiment.py" 2>/dev/null; sleep 2
source /workspace/.venv/bin/activate && cd /workspace/code/dreamzero
MAX_STEPS=5000 DK1_DATA_ROOT=/workspace/data/dk1 bash scripts/train/dk1_training.sh \
    > /workspace/checkpoints/dreamzero_dk1_lora/train.log 2>&1 &
```

If the smoke test failed or was killed, restart it:
```bash
pkill -f "experiment.py" 2>/dev/null; sleep 2
source /workspace/.venv/bin/activate && cd /workspace/code/dreamzero
MAX_STEPS=10 DK1_DATA_ROOT=/workspace/data/dk1 bash scripts/train/dk1_training.sh \
    > /tmp/dk1_smoke.log 2>&1 &
tail -f /tmp/dk1_smoke.log
```

### RAM OOM History & Fixes Applied (2026-03-17)

**Root cause of OOM**: The shard caching in `get_shard()` loaded ALL video frames for a
trajectory regardless of size. `qualiaadmin_pingpongmegamerge` has 131k steps → 271 GB/shard.
With 8 concurrent shards (4 GPU processes × 2 shards each), this = 2168 GB → instant OOM.

**Additional fix (after first smoke test attempt):**
- `persistent_workers` was `true` in `groot/vla/configs/conf.yaml` — this conflicts with `num_workers=0`
- Added `dataloader_persistent_workers=false` to `dk1_training.sh`

**Fixes applied:**
1. **Windowed loading** in `ShardedLeRobotSubLangSingleActionChunkDatasetDROID.get_shard()`:
   - Randomly selects a contiguous window of `num_steps_per_shard` frames per trajectory
   - Parquet (state/action) still loaded in full (tiny, ~100 MB)
   - Returns `window_starts` and `window_sizes` dicts
   - `get_video()` translates absolute step indices → window-relative before shard lookup
   - `__iter__` filters `allowed_indices` to window bounds
2. **`num_steps_per_shard: 10000`** in `dk1_relative.yaml` (was 2000 which didn't help single-file datasets)
3. **`dataloader_num_workers=0`** in `dk1_training.sh` (was 2 → reduces concurrent shards from 16 to 8)

**Expected RAM with these fixes:**
- 10k steps × 3 cams × 640×360 × 3 bytes ≈ 20 GB/shard
- 8 concurrent shards × 20 GB = 160 GB video + ~150 GB overhead = ~310 GB total
- Well under 704 GB limit

**If OOM persists**, try:
- Reduce `num_steps_per_shard` to 5000 (→ ~80 GB total video)
- Check actual RAM with `free -h` during shard caching phase
- The shard caching prints: `"Caching shard"` and `"Windowed trajectory N: rows=..."` to the log

---

## Environment

- **Python venv**: `/workspace/.venv` (use `uv`, not `python -m venv`)
- **Working dir**: `/workspace/code/dreamzero`
- **GPUs**: 4× H100/A100 80GB
- **Data**: `/workspace/data/dk1/` — 29 datasets in GEAR format
- **Checkpoints**:
  - `/workspace/checkpoints/Wan2.1-I2V-14B-480P/` — base DiT
  - `/workspace/checkpoints/DreamZero-AgiBot/` — fine-tuned checkpoint (LoRA init)
  - `/workspace/checkpoints/umt5-xxl/` — tokenizer only (not full weights)
  - `/workspace/checkpoints/dreamzero_dk1_lora/` — output dir

---

## What Was Done

### 1. Data Conversion
All 29 DK-1 datasets converted from LeRobot v3 → GEAR format using:
```bash
bash scripts/data/convert_all_dk1.sh
```
- Datasets live at `/workspace/data/dk1/<name>/`
- Each has `meta/modality.json`, `data/`, `meta/stats.json`
- Camera name normalization: `context→top`, `base_0→top` per dataset
- Skipped: sfeduniak (Cartesian only), 2-camera-only datasets

### 2. Code Changes Made

#### `groot/vla/data/schema/embodiment_tags.py`
Added `DK1 = "dk1"` to the enum.

#### `groot/vla/data/transform/video.py`
- Added `VideoResolutionNormalize` class (lines 16–99) — normalizes each camera view to a common resolution using smallest-side-scale + center-crop
- Relaxed assertion in `VideoCrop.get_transform()` to allow mixed resolutions (changed `assert` → `pass`)

#### `groot/vla/data/transform/__init__.py`
Added `VideoResolutionNormalize` to the exports.

#### `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
Added DK-1 config blocks:
- `modality_config_dk1`: 3 video views (top, left_wrist, right_wrist), state/action keys
- `transform_dk1`: includes `VideoResolutionNormalize(640×360)` before crop/resize
- Registered in `modality_configs`, `transforms`, `metadata_versions` (0221), `fps` (30) maps

#### `groot/vla/configs/data/dreamzero/dk1_relative.yaml` *(new file)*
Dataset config listing all 29 individual dataset paths explicitly.

#### `scripts/data/convert_lerobot_to_gear.py`
- Added "dk1" to VALID_EMBODIMENT_TAGS
- Fixed LeRobot v3 parquet path discovery (file-based naming)
- Added `load_task_lookup()` for v3 task strings
- Added `--video-key-remap` CLI flag for camera name normalization

#### `scripts/data/convert_all_dk1.sh` *(new file)*
Batch conversion script with per-dataset camera remapping.

#### `scripts/train/dk1_training.sh` *(new file)*
LoRA training launcher (4 GPUs, batch=1, max_steps=5000).

---

## Architecture / Key Numbers

- **Model**: DreamZero = Wan2.1-I2V-14B-480P + action head (flow matching)
- **Robot**: DK-1 bimanual, 14-DoF each arm → 28-dim actions, only first 14 used (joint positions)
- **Action keys**: `left_joint_pos` (6), `left_gripper_pos` (1), `right_joint_pos` (6), `right_gripper_pos` (1)
- **Views**: `top` (overview), `left_wrist`, `right_wrist`
- **Training resolution**: **320×176** (set by `image_resolution_width/height` in training script)
- **`VideoResolutionNormalize` target**: 640×360 — *intermediate only*, not the training resolution
- **Full transform pipeline per camera**:
  `VideoToTensor → VideoResolutionNormalize(640×360) → VideoCrop(scale=0.95→~608×342) → VideoResize(320×176) → VideoColorJitter → VideoToNumpy`

---

## Known Issues / Watch Out For

1. **Mixed camera resolutions across DK-1 datasets** — solved by `VideoResolutionNormalize`
   - Native resolutions: 800×600, 960×540, 640×480, 640×360, 1280×720
   - Each camera is independently resized to 640×360 before the shared crop

2. **umt5-xxl**: Only tokenizer files needed (not the 36GB model weights). Text embeddings loaded from `models_t5_umt5-xxl-enc-bf16.pth` in the Wan2.1 checkpoint dir.

3. **LeRobot v3 format**: Task strings are in `meta/tasks.parquet`, not inline in episode data.

4. **`skip_component_loading=true` and `defer_lora_injection=true`**: Required flags to load the AgiBot checkpoint as LoRA init without loading mismatched components.

---

## If Training Fails

Common errors and fixes:

- **`VideoResolutionNormalize not found`**: Already fixed in `__init__.py` — verify with:
  ```bash
  python -c "from groot.vla.data.transform import VideoResolutionNormalize; print('OK')"
  ```

- **`All video keys must have the same resolution`**: The `pass` in `VideoCrop.get_transform()` should suppress this — check `video.py:~248`

- **`meta/modality.json not found`**: The YAML must list individual dataset paths, not the parent `/workspace/data/dk1/` — check `dk1_relative.yaml`

- **OOM (out of memory)**: See the "RAM OOM History & Fixes Applied" section above. If RAM is still too high, lower `num_steps_per_shard` in `dk1_relative.yaml` (try 5000)

---

## Next Steps After Smoke Test

1. If smoke test passes → launch full training (5000 steps, ~2–3 days)
2. Monitor with: `watch -n30 nvidia-smi` and `tail -f /workspace/checkpoints/dreamzero_dk1_lora/train.log`
3. Checkpoints saved every 500 steps to `/workspace/checkpoints/dreamzero_dk1_lora/`
