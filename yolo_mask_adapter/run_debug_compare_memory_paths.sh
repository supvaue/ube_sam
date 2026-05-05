#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/debug_compare_memory_paths.py \
  --video ../../Datasets/标注片段.mp4 \
  --cache-item yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_mask_cache/autolabel_forceps_exp1_iou05_track_clips__track_000001__frame_022416.npz \
  --category-id 12 \
  --start-frame 22416 \
  --end-frame 22417 \
  --adapter-checkpoint yolo_mask_adapter/results/forceps_no_memory_decoder_autolabel_film_mlp_nomemattn_reinit_e8/best.pt \
  --model-checkpoint yolo_mask_adapter/results/forceps_shared_decoder_autolabel_film_mlp_memonly_e4/best.pt \
  --config-file configs/sam2.1/sam2.1_hiera_s_rvos.yaml \
  --training-config-file configs/rvos_training/17/sam2.1_s_ref17_resurgsam \
  --ckpt-path checkpoints/sam2.1_hiera_s_ref17.pth \
  --apply-long-term-memory \
  --disable-obj-score-gating \
  --use-no-mem-attention \
  --use-mask-as-output \
  --direct-memory-mode raw_logits \
  --force-memory-object-score 10.0 \
  --device cuda:0 \
  --output-json yolo_mask_adapter/results/debug_compare_memory_paths_22416.json \
  2>&1 | tee yolo_mask_adapter/results/logs/debug_compare_memory_paths_22416.log
