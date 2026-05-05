#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/eval_refined_init_tracking.py \
  --video ../../Datasets/标注片段.mp4 \
  --annotations ../../Datasets/exp1_cu_full/annotations/instances_default.json \
  --category-id 12 \
  --start-frame 0 \
  --max-video-frames 400 \
  --max-gt-evals 10 \
  --cache-dir yolo_mask_adapter/results/forceps_mask_cache_full \
  --cache-dataset exp1_cu_full \
  --adapter-checkpoint yolo_mask_adapter/results/forceps_no_memory_decoder_autolabel_film_mlp_nomemattn_reinit_e8/best.pt \
  --model-checkpoint yolo_mask_adapter/results/forceps_shared_decoder_autolabel_film_mlp_memonly_e4/best.pt \
  --config-file configs/sam2.1/sam2.1_hiera_s_rvos.yaml \
  --training-config-file configs/rvos_training/17/sam2.1_s_ref17_resurgsam \
  --ckpt-path checkpoints/sam2.1_hiera_s_ref17.pth \
  --apply-long-term-memory \
  --disable-obj-score-gating \
  --use-no-mem-attention \
  --use-mask-as-output \
  --device cuda:0 \
  --output-json yolo_mask_adapter/results/forceps_refined_init_tracking_memonly_10gt_smoke.json \
  2>&1 | tee yolo_mask_adapter/results/logs/forceps_refined_init_tracking_memonly_10gt_smoke.log
