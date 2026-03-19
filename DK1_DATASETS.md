# DK-1 Datasets Reference

All datasets are in LeRobot v3 DROID format, hosted on HuggingFace.
Downloaded to `/workspace/data/dk1/` on the training machine.

## HuggingFace Sources

| Local Name | HuggingFace Repo | Notes |
|---|---|---|
| FabianKerj_dualarm-pretrain-127ep | `FabianKerj/dualarm-pretrain-127ep` | |
| FabianKerj_dualarm-pretrain-417ep | `FabianKerj/dualarm-pretrain-417ep` | Excluded from training (broken episodes.jsonl) |
| Gongsta_dk1_2026-02-28 | `Gongsta/dk1_2026-02-28` | |
| Gongsta_dk1_2026-03-01 | `Gongsta/dk1_2026-03-01` | |
| Gongsta_dk1_2026-03-02 | `Gongsta/dk1_2026-03-02` | |
| Gongsta_dk1_2026-03-03 | `Gongsta/dk1_2026-03-03` | |
| Gongsta_trlc_tshirt_folding | `Gongsta/trlc_tshirt_folding` | Largest dataset (1.1M frames) |
| Zasha01_cube_transfer_final | `Zasha01/cube_transfer_final` | |
| Zasha01_eval_pi05-cube-transfer-final-full15 | `Zasha01/eval_pi05-cube-transfer-final-full15` | Excluded (eval data, 1 frame) |
| Zasha01_eval_pi05-cube-transfer_v1 | `Zasha01/eval_pi05-cube-transfer_v1` | Excluded (eval data) |
| Zasha01_lego_cube | `Zasha01/lego_cube` | |
| Zasha01_lego_cube_final | `Zasha01/lego_cube_final` | |
| Zasha01_packaging | `Zasha01/packaging` | |
| Zasha01_test_tube_insertion6 | `Zasha01/test_tube_insertion6` | |
| Zasha01_test_tube_insertion7 | `Zasha01/test_tube_insertion7` | |
| Zasha01_towel_folding | `Zasha01/towel_folding` | |
| Zasha01_video4 | `Zasha01/video4` | |
| dopaul_pcb_dummy_v1 | `dopaul/pcb_dummy_v1` | Only 2 cameras — excluded from merge |
| dopaul_pcb_placement_100x_1st_item | `dopaul/pcb_placement_100x_1st_item` | |
| dopaul_pcb_placement_v1 | `dopaul/pcb_placement_v1` | |
| qualiaadmin_43 | `qualiaadmin/43` | Only 2 cameras — excluded from merge |
| qualiaadmin_52 | `qualiaadmin/52` | Only 2 cameras — excluded from merge |
| qualiaadmin_53 | `qualiaadmin/53` | Only 2 cameras — excluded from merge |
| qualiaadmin_Fabian | `qualiaadmin/Fabian` | Only 2 cameras — excluded from merge |
| qualiaadmin_eventgrab11 | `qualiaadmin/eventgrab11` | |
| qualiaadmin_mandminbox | `qualiaadmin/mandminbox` | |
| qualiaadmin_pingpongmegamerge | `qualiaadmin/pingpongmegamerge` | 290 episodes in 1 parquet file |
| qualiaadmin_pingpongred1 | `qualiaadmin/pingpongred1` | |
| qualiaadmin_pinkthing_reward1_10 | `qualiaadmin/pinkthing_reward1_10` | |
| qualiaadmin_pinkthing_reward1_20 | `qualiaadmin/pinkthing_reward1_20` | |
| qualiaadmin_plasticcuprepeat1 | `qualiaadmin/plasticcuprepeat1` | |
| qualiaadmin_plasticinbox50episodesimpedance | `qualiaadmin/plasticinbox50episodesimpedance` | |
| qualiaadmin_spoon1 | `qualiaadmin/spoon1` | |
| qualiaadmin_spoon10 | `qualiaadmin/spoon10` | Excluded (too few frames: 272) |
| sfeduniak_dhl_0310 | `sfeduniak/dhl_0310` | Excluded (different cam names, 16D action) |
| sfeduniak_towels_0310 | `sfeduniak/towels_0310` | Excluded (different cam names, 16D action) |

## Download Command

```bash
# Example for one dataset:
huggingface-cli download qualiaadmin/pingpongmegamerge \
    --repo-type dataset \
    --local-dir /workspace/data/dk1/qualiaadmin_pingpongmegamerge

# All datasets (replace _ with / for HF repo name, except special cases):
for ds in FabianKerj/dualarm-pretrain-127ep Gongsta/dk1_2026-02-28 \
          Gongsta/dk1_2026-03-01 Gongsta/dk1_2026-03-02 Gongsta/dk1_2026-03-03 \
          Gongsta/trlc_tshirt_folding \
          Zasha01/cube_transfer_final Zasha01/lego_cube Zasha01/lego_cube_final \
          Zasha01/packaging Zasha01/test_tube_insertion6 Zasha01/test_tube_insertion7 \
          Zasha01/towel_folding Zasha01/video4 \
          dopaul/pcb_placement_100x_1st_item dopaul/pcb_placement_v1 \
          qualiaadmin/eventgrab11 qualiaadmin/mandminbox \
          qualiaadmin/pingpongmegamerge qualiaadmin/pingpongred1 \
          qualiaadmin/pinkthing_reward1_10 qualiaadmin/pinkthing_reward1_20 \
          qualiaadmin/plasticcuprepeat1 qualiaadmin/plasticinbox50episodesimpedance \
          qualiaadmin/spoon1; do
    local_name=$(echo "$ds" | tr '/' '_')
    echo "Downloading $ds -> $local_name"
    huggingface-cli download "$ds" --repo-type dataset --local-dir "/workspace/data/dk1/$local_name"
done
```

## Dataset Details

| Dataset | Episodes | Frames | FPS | Resolution | Action Dim | Cameras | Tasks |
|---|---|---|---|---|---|---|---|
| FabianKerj_dualarm-pretrain-127ep | 127 | 69,860 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | put pink thing in box, remove pink thing, put green bag, put plastic tube |
| Gongsta_dk1_2026-02-28 | 28 | 75,895 | 30 | 800x600, 960x540 | 28 | top, left_wrist, right_wrist | Fold the t-shirt |
| Gongsta_dk1_2026-03-01 | 141 | 306,425 | 30 | 800x600, 960x540 | 28 | top, left_wrist, right_wrist | Fold the t-shirt |
| Gongsta_dk1_2026-03-02 | 61 | 120,265 | 30 | 800x600, 960x540 | 28 | top, left_wrist, right_wrist | Fold the t-shirt |
| Gongsta_dk1_2026-03-03 | 132 | 273,043 | 30 | 800x600, 960x540 | 28 | top, left_wrist, right_wrist | Fold the t-shirt |
| Gongsta_trlc_tshirt_folding | 532 | 1,106,053 | 30 | 800x600, 960x540 | 28 | top, left_wrist, right_wrist | Fold the t-shirt |
| Zasha01_cube_transfer_final | 102 | 59,726 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Transfer the lego cube to the other arm |
| Zasha01_lego_cube | 25 | 17,385 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Transfer the Lego Cube to the other arm |
| Zasha01_lego_cube_final | 251 | 100,660 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Transfer the Lego Cube to the other arm |
| Zasha01_packaging | 1 | 987 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Pick up the green cubes from the box |
| Zasha01_test_tube_insertion6 | 1 | 2,752 | 30 | 1280x720 | 14 | context, left_wrist, right_wrist | Pick up test tube and cork and insert it |
| Zasha01_test_tube_insertion7 | 1 | 867 | 30 | 1280x720 | 14 | context, left_wrist, right_wrist | Pick up test tube and cork and insert it |
| Zasha01_towel_folding | 1 | 2,550 | 30 | 1280x720 | 14 | context, left_wrist, right_wrist | Fold the towel |
| Zasha01_video4 | 15 | 8,867 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Stack the Lego Cubes |
| dopaul_pcb_placement_100x_1st_item | 100 | 65,461 | 30 | 640x480 | 14 | top, left_wrist, right_wrist | Take PCB from box and place in testbed |
| dopaul_pcb_placement_v1 | 50 | 30,616 | 30 | 640x480 | 14 | top, left_wrist, right_wrist | Take PCB from box and place in testbed |
| qualiaadmin_eventgrab11 | 57 | 42,657 | 30 | 640x360 | 14 | base_0, left_wrist, right_wrist | Grab the cap |
| qualiaadmin_mandminbox | 30 | 12,211 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | Put the green bag in the box |
| qualiaadmin_pingpongmegamerge | 290 | 131,112 | 30 | 640x360 | 14 | base_0, left_wrist, right_wrist | Put the ball in the cup |
| qualiaadmin_pingpongred1 | 80 | 35,331 | 30 | 640x360 | 14 | base_0, left_wrist, right_wrist | Put the ball in the cup |
| qualiaadmin_pinkthing_reward1_10 | 47 | 22,200 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | Put pink thing in box, remove pink thing |
| qualiaadmin_pinkthing_reward1_20 | 5 | 3,932 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | Put pink thing in box, remove pink thing |
| qualiaadmin_plasticcuprepeat1 | 50 | 70,066 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | Put the plastic in the cup |
| qualiaadmin_plasticinbox50episodesimpedance | 50 | 35,449 | 30 | 640x360, 640x480 | 14 | base_0, left_wrist, right_wrist | Put the plastic tube the box |
| qualiaadmin_spoon1 | 20 | 2,737 | 30 | 640x360 | 14 | context, left_wrist, right_wrist | Pick up the spoon |

**Totals (included in merge):** 25 datasets, ~2,196 episodes, ~2.6M frames, 16 unique tasks

## Data Format Notes

- **LeRobot v3 DROID format**: Multiple episodes packed per parquet/video file
- **Parquet columns**: `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
- **Camera name variations**: `base_0`/`context`/`top` for overview camera → normalized to `top` in merge
- **Action dim**: 14D (6 joint + 1 gripper per arm) or 28D (sliced to first 14 in merge)
- **Mixed resolutions**: 640x360, 640x480, 800x600, 960x540, 1280x720 → normalized to 640x360 in merge
- **Episodes per file**: Source parquets contain 1-290 episodes each; merge script splits to 1 episode per file

## Merge Script

```bash
python scripts/data/merge_dk1_datasets.py \
    --data-root /path/to/dk1 \
    --output /path/to/dk1-merged \
    --target-width 640 --target-height 360 \
    --ffmpeg-workers 16 \
    --resume
```

After videos complete, compute stats:
```bash
python scripts/data/merge_dk1_datasets.py \
    --data-root /path/to/dk1 \
    --output /path/to/dk1-merged \
    --target-width 640 --target-height 360 \
    --resume --skip-videos
```
