# DreamZero DK-1 Fine-Tuning — Claude Code Briefing

## Mission

Fine-tune **DreamZero** (14B World-Action Model) on DK-1 bimanual robot data.
DreamZero **jointly predicts future video frames AND robot actions** from observation history + language instruction.

This is the right model for our use case — unlike DreamDojo (which only does actions→video),
DreamZero is a true World-Action Model: given last N frames + task description → predicts both
what will happen (video) and what actions to take.

**Base architecture**: Wan2.1-I2V-14B-480P (14B video diffusion transformer) + action head.
**Training strategy**: LoRA fine-tuning from DreamZero-AgiBot checkpoint.
**Reference**: arXiv 2602.15922. Code: github.com/dreamzero0/dreamzero.

---

## Paths on This Machine

| Path | Contents |
|------|----------|
| `/workspace/code/dreamzero/` | DreamZero repo (your working directory) |
| `/workspace/data/dk1/` | DK-1 datasets (downloading, ~85GB) |
| `/workspace/checkpoints/Wan2.1-I2V-14B-480P/` | Wan backbone (~28GB, downloading) |
| `/workspace/checkpoints/DreamZero-AgiBot/` | LoRA base checkpoint (~45GB, downloading) |
| `/workspace/checkpoints/umt5-xxl/` | Text tokenizer (~5GB, downloading) |

---

## DK-1 Robot Specs

- **Robot**: DK-1 bimanual desktop arm (The Robot Learning Company)
- **DoF**: 14 total
  - Left arm: 6 joints (indices 0-5)
  - Left gripper: 1 DoF (index 6)
  - Right arm: 6 joints (indices 7-12)
  - Right gripper: 1 DoF (index 13)
- **Action representation**: Absolute joint positions (NOT Cartesian, NOT deltas)
- **Dataset format**: LeRobot **v3.0** (important: NOT v2 — see compatibility section)
- **FPS**: 30fps (most datasets)

---

## DK-1 Dataset Overview

Datasets are at `/workspace/data/dk1/` with folder names `{Author}_{dataset_name}`.
After download, you should have ~34 dataset directories.

### Camera availability by author:

| Author prefix | Cameras | Resolution | Notes |
|---------------|---------|------------|-------|
| `Gongsta_` | top + wrist_left + wrist_right | 800×600 / 960×540 | **Best: has 3 cameras natively** |
| `Zasha01_` | context only | 640×360 | 1 camera |
| `qualiaadmin_` | context only | 640×360 | 1 camera |
| `dopaul_` | top only | 640×480 | 1 camera |
| `FabianKerj_` | unknown | unknown | Check |
| `sfeduniak_` | top | 640×480 | **SKIP: Cartesian 16-DoF, incompatible** |

**DreamZero requires exactly 3 camera views** (`num_views=3`).
Camera view order convention (important for transfer from DreamZero-AgiBot):
→ [overview/top, secondary_exterior, wrist]
→ For DK-1: [top/context, wrist_left, wrist_right]

**For single-camera datasets**: either replicate the single camera for all 3 slots,
or pad missing views with black frames. Check how the converter handles this.

---

## LeRobot v3 Compatibility — CRITICAL

The converter `scripts/data/convert_lerobot_to_gear.py` was written for LeRobot **v2**.
DK-1 datasets are **v3.0**.

**Key v3 difference you WILL hit**:
`meta/tasks.parquet` in v3 stores the task description string in the **DataFrame index**
(row label), not in a `task` column.

Fix when it errors:
```python
# Instead of: df["task"]
# Do:
df = pd.read_parquet("meta/tasks.parquet")
df = df.reset_index().rename(columns={"index": "task"})
task_strings = df["task"].tolist()
```

Apply this fix in `convert_lerobot_to_gear.py` wherever it reads tasks.parquet.

---

## Step-by-Step Plan

Follow the official guide at `docs/DATASET_TO_GEAR_AND_TRAIN.md` — read it first!
The closest existing embodiment to reference is **YAM** (also bimanual, same 14-DoF structure).

### Step 0: Verify downloads

```bash
ls /workspace/data/dk1/ | wc -l              # expect ~34
ls /workspace/checkpoints/
ls /workspace/checkpoints/Wan2.1-I2V-14B-480P/ | head
ls /workspace/checkpoints/DreamZero-AgiBot/ | head
```

### Step 1: Inspect a DK-1 dataset

```bash
ls /workspace/data/dk1/Gongsta_dk1_2026-02-28/
ls /workspace/data/dk1/Gongsta_dk1_2026-02-28/meta/
cat /workspace/data/dk1/Gongsta_dk1_2026-02-28/meta/info.json
```

Check the actual column names in the parquet files:
```python
import pandas as pd
df = pd.read_parquet('/workspace/data/dk1/Gongsta_dk1_2026-02-28/data/chunk-000/episode_000000.parquet')
print(df.columns.tolist())
print(df.dtypes)
print(df.head(2))
```

The action/state column is likely `observation.state` and `action`.
Check video dirs:
```bash
ls /workspace/data/dk1/Gongsta_dk1_2026-02-28/videos/chunk-000/
```

### Step 2: Register dk1 embodiment tag

Edit `groot/vla/data/schema/embodiment_tags.py`:
```python
class EmbodimentTag(str, Enum):
    ...
    DK1 = "dk1"
```

Also add to `VALID_EMBODIMENT_TAGS` in `scripts/data/convert_lerobot_to_gear.py`.

### Step 3: Run the GEAR converter

The converter adds metadata files to `meta/` in-place (does NOT modify videos or parquet).

For a Gongsta dataset (3 cameras, adjust video key names after inspecting the videos/ dir):
```bash
cd /workspace/code/dreamzero
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path /workspace/data/dk1/Gongsta_dk1_2026-02-28 \
    --embodiment-tag dk1 \
    --state-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
    --action-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
    --relative-action-keys left_joint_pos right_joint_pos \
    --task-key annotation.task
```

NOTE: Adjust `--state-keys` / `--action-keys` based on actual column structure.
If the state is a flat 14-dim vector in `observation.state`, the above is correct.

Fix v3 tasks.parquet errors as they appear. Run on all non-sfeduniak datasets.

After conversion, each dataset should have:
- `meta/modality.json`
- `meta/embodiment.json`
- `meta/stats.json`
- `meta/relative_stats_dreamzero.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`

### Step 4: Add DK-1 modality config and transforms

Edit `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`.

First look at `modality_config_yam` in the same file — that's our template.

Add `modality_config_dk1` (after the yam config):
```yaml
modality_config_dk1:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    eval_delta_indices: [0]
    modality_keys:
      - video.top          # overview camera (adjust to actual key from modality.json)
      - video.wrist_left   # left wrist camera
      - video.wrist_right  # right wrist camera
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - state.left_joint_pos
      - state.left_gripper_pos
      - state.right_joint_pos
      - state.right_gripper_pos
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    modality_keys:
      - action.left_joint_pos
      - action.left_gripper_pos
      - action.right_joint_pos
      - action.right_gripper_pos
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.task
```

Add `transform_dk1` (based on `transform_yam`):
```yaml
transform_dk1:
  _target_: groot.vla.data.transform.ComposedModalityTransform
  transforms:
    - <<: *totensor_cfg
      apply_to: ${modality_config_dk1.video.modality_keys}
    - <<: *crop_cfg
      apply_to: ${modality_config_dk1.video.modality_keys}
    - <<: *resize_cfg
      apply_to: ${modality_config_dk1.video.modality_keys}
    - <<: *color_jitter_cfg
      apply_to: ${modality_config_dk1.video.modality_keys}
    - <<: *to_numpy_cfg
      apply_to: ${modality_config_dk1.video.modality_keys}
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_dk1.state.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_dk1.state.modality_keys}
      normalization_modes:
        state.left_joint_pos: q99
        state.left_gripper_pos: q99
        state.right_joint_pos: q99
        state.right_gripper_pos: q99
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_dk1.action.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_dk1.action.modality_keys}
      normalization_modes:
        action.left_joint_pos: q99
        action.left_gripper_pos: q99
        action.right_joint_pos: q99
        action.right_gripper_pos: q99
    - _target_: groot.vla.data.transform.ConcatTransform
      video_concat_order: ${modality_config_dk1.video.modality_keys}
      state_concat_order: ${modality_config_dk1.state.modality_keys}
      action_concat_order: ${modality_config_dk1.action.modality_keys}
    - ${model_specific_transform}
```

Register in global maps at bottom of the YAML:
```yaml
modality_configs:
  dk1: ${modality_config_dk1}

transforms:
  dk1: ${transform_dk1}

metadata_versions:
  dk1: '0221'

fps:
  dk1: 30
```

### Step 5: Create dataset YAML

Copy `groot/vla/configs/data/dreamzero/yam_relative.yaml` to `dk1_relative.yaml`
and replace all occurrences of `yam` with `dk1`.

Key fields to verify/update:
- `max_state_dim: 64` (14 DoF fits in 64)
- `relative_action_keys: [left_joint_pos, right_joint_pos]` (gripper absolute is fine)
- `dk1_data_root: ???` (set via CLI)
- `mixture_spec.dataset_path.dk1: [${dk1_data_root}]`

### Step 6: Create training script

Create `scripts/train/dk1_training.sh` based on `yam_training.sh`.
Replace `yam` with `dk1` throughout. Key parameters:

```bash
#!/bin/bash
export HYDRA_FULL_ERROR=1
export WORKDIR=/workspace

DK1_DATA_ROOT=${DK1_DATA_ROOT:-"/workspace/data/dk1"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/checkpoints/dreamzero_dk1_lora"}
NUM_GPUS=${NUM_GPUS:-4}
WAN_CKPT_DIR="/workspace/checkpoints/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="/workspace/checkpoints/umt5-xxl"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/dk1_relative \
    wandb_project=dreamzero-dk1 \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    dk1_data_root=$DK1_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=/workspace/checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
```

### Step 7: Smoke test (10 steps)

```bash
source /workspace/.venv/bin/activate
cd /workspace/code/dreamzero
MAX_STEPS=10 bash scripts/train/dk1_training.sh 2>&1 | tee /workspace/smoke_test.log
```

Watch for:
- DataLoader loading samples (first batch may take a few minutes)
- Loss values appearing (not NaN)
- No CUDA OOM
- Both rank0 and rank1 stepping

### Step 8: Full training

If smoke test passes:
```bash
tmux new -s training
source /workspace/.venv/bin/activate
cd /workspace/code/dreamzero
bash scripts/train/dk1_training.sh 2>&1 | tee /workspace/training.log
```

Monitor at: https://wandb.ai/andreaskoepf/dreamzero-dk1

---


## IMPORTANT: Full LeRobot v3 Compatibility Patch for DreamZero Converter

DK-1 episode parquets have **NO** `annotation.task` column — only `task_index` (int).
The task string must be looked up from `meta/tasks.parquet` where in v3 the string is in the DataFrame index.

Two-part fix needed in `scripts/data/convert_lerobot_to_gear.py`:

### Part 1: Load task lookup from meta/tasks.parquet
Add a helper function:
```python
def load_task_lookup(dataset_path: Path) -> dict[int, str]:
    tasks_path = dataset_path / "meta/tasks.parquet"
    if not tasks_path.exists():
        return {}
    df = pd.read_parquet(tasks_path)
    # v3: task string is in the DataFrame index (not a column)
    df = df.reset_index()
    task_col = "task" if "task" in df.columns else df.columns[0]
    return {i: str(row[task_col]) for i, row in df.iterrows()}
```

### Part 2: Inject task string into episode parquets during processing
In `build_tasks()` and `build_episodes()`, when reading episode parquets:
```python
task_lookup = load_task_lookup(dataset_path)
# Then when iterating parquet rows:
if "task_index" in df.columns and task_lookup:
    df["annotation.task"] = df["task_index"].map(task_lookup)
```

Use `--task-key annotation.task` as normal — the injected column will be present.
## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `tasks.parquet` has no `task` column (v3) | `df.reset_index().rename(columns={'index': 'task'})` |
| Single-camera datasets need 3 views | Replicate top cam or add black frame padding in modality config |
| sfeduniak datasets are Cartesian (16-DoF) | Skip entirely, don't convert |
| Video key names differ by dataset | Check `videos/chunk-000/` dir structure, update modality_keys accordingly |
| `observation.state` vs `state` column name | Check parquet, adjust `--state-keys` original_key if needed |

---

## W&B

- Entity: `andreaskoepf`
- Project: `dreamzero-dk1`
- Run `wandb login` before training

---

## Success Criteria

- [ ] Converter runs on all ~34 non-sfeduniak datasets without errors
- [ ] 10-step smoke test completes, loss is not NaN
- [ ] W&B shows training loss curve after step 1
- [ ] LoRA checkpoint saved at step 500 (`output_dir/checkpoint-500/`)
- [ ] No CUDA OOM with 4x H100 + ZeRO-2 + batch_size=1
