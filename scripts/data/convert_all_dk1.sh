#!/bin/bash
# Convert all DK-1 datasets to GEAR format
# Skips sfeduniak datasets (Cartesian 16-DoF, incompatible)
# Skips 2-camera datasets (DreamZero requires 3 views)
#
# Camera name normalization:
#   Overview: top / context / base_0  →  all mapped to "top" in modality.json
#   Wrists:   left_wrist / right_wrist  →  kept as-is
#
# Usage:
#   source /workspace/.venv/bin/activate
#   bash scripts/data/convert_all_dk1.sh

set -euo pipefail

DATA_ROOT="${DK1_DATA_ROOT:-/workspace/data/dk1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER="$SCRIPT_DIR/convert_lerobot_to_gear.py"

# Common args for all DK-1 datasets:
# - 14-DoF state: left arm [0:6] + left gripper [6:7] + right arm [7:13] + right gripper [13:14]
# - 28-dim action but we only use first 14 (joint positions, same layout as state)
# - Relative actions for joint positions only (grippers are absolute)
STATE_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'
ACTION_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'

SKIP_COUNT=0
CONVERT_COUNT=0
FAIL_COUNT=0

for dataset_dir in "$DATA_ROOT"/*/; do
    name=$(basename "$dataset_dir")

    # Skip sfeduniak (Cartesian 16-DoF actions)
    if [[ "$name" == sfeduniak_* ]]; then
        echo "SKIP (Cartesian): $name"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Check camera count — skip datasets with fewer than 3 cameras
    cam_count=$(python3 -c "
import json
info = json.load(open('$dataset_dir/meta/info.json'))
print(sum(1 for v in info['features'].values() if v.get('dtype') == 'video'))
" 2>/dev/null || echo "0")

    if [ "$cam_count" -lt 3 ]; then
        echo "SKIP (only $cam_count cameras): $name"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Determine video key remap based on which overview camera this dataset has
    # All datasets have left_wrist + right_wrist; the overview varies:
    #   Gongsta_*:  observation.images.top         → "top" (no remap needed)
    #   Zasha01_*:  observation.images.context      → needs remap context→top
    #   dopaul_*:   observation.images.top          → "top" (no remap needed)
    #   FabianKerj: observation.images.base_0       → needs remap base_0→top
    #   qualiaadmin: observation.images.base_0 or context → check dynamically
    VIDEO_REMAP=""
    overview_cam=$(python3 -c "
import json
info = json.load(open('$dataset_dir/meta/info.json'))
cams = [k.replace('observation.images.', '') for k, v in info['features'].items() if v.get('dtype') == 'video']
# Find the overview camera (not left_wrist or right_wrist)
overview = [c for c in cams if c not in ('left_wrist', 'right_wrist')]
print(overview[0] if overview else '')
" 2>/dev/null || echo "")

    if [ "$overview_cam" = "context" ]; then
        VIDEO_REMAP='{"context": "top"}'
    elif [ "$overview_cam" = "base_0" ]; then
        VIDEO_REMAP='{"base_0": "top"}'
    fi
    # "top" needs no remap

    echo "========================================"
    echo "Converting: $name ($cam_count cameras, overview=$overview_cam)"
    echo "========================================"

    REMAP_ARGS=()
    if [ -n "$VIDEO_REMAP" ]; then
        REMAP_ARGS=(--video-key-remap "$VIDEO_REMAP")
    fi

    if python3 "$CONVERTER" \
        --dataset-path "$dataset_dir" \
        --embodiment-tag dk1 \
        --state-keys "$STATE_KEYS" \
        --action-keys "$ACTION_KEYS" \
        --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
        --task-key annotation.task \
        "${REMAP_ARGS[@]}" \
        --force; then
        CONVERT_COUNT=$((CONVERT_COUNT + 1))
    else
        echo "FAILED: $name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "========================================"
echo "Done! Converted: $CONVERT_COUNT | Skipped: $SKIP_COUNT | Failed: $FAIL_COUNT"
echo "========================================"
