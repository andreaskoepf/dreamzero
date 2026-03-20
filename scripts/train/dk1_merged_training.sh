#!/bin/bash
# DreamZero DK-1 Training Script (merged & deduplicated dataset)
#
# Usage:
#   bash scripts/train/dk1_merged_training.sh
#
# Prerequisites:
#   - Merged DK-1 dataset at DK1_MERGED_DATA_PATH
#   - Wan2.1-I2V-14B-480P weights at /workspace/checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer at /workspace/checkpoints/umt5-xxl
#   - DreamZero-AgiBot checkpoint at /home/claude/DreamZero-AgiBot

export HYDRA_FULL_ERROR=1

# ============ CHANGE THESE VARIABLES ============
DK1_MERGED_DATA_PATH=${DK1_MERGED_DATA_PATH:-"/workspace/data/dk1-merge-2026-03"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/checkpoints/dreamzero_dk1_merged_lora"}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/workspace/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/workspace/checkpoints/umt5-xxl"}

if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-4}
# =============================================

# Validate required paths
for dir in "$DK1_MERGED_DATA_PATH" "$WAN_CKPT_DIR" "$TOKENIZER_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory not found: $dir"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/dk1_merged_relative \
    wandb_project=dreamzero-dk1-merged \
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
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=${MAX_STEPS:-20000} \
    weight_decay=1e-5 \
    save_total_limit=2 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=${DATALOADER_NUM_WORKERS:-1} \
    dataloader_persistent_workers=true \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    dk1_merged_data_path=$DK1_MERGED_DATA_PATH \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=/home/claude/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
