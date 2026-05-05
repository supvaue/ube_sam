#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs

CUDA_VISIBLE_DEVICES=3 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/eval_dense_tracking_protocol.py \
  --video ../../Datasets/标注片段.mp4 \
  --annotations ../../Datasets/exp1_cu_full/annotations/instances_default.json \
  --category-id 1 \
  --start-frame 0 \
  --max-video-frames 121 \
  --max-gt-evals 5 \
  --init-source yolo_token \
  --cache-dir yolo_mask_adapter/results/ligamentum_flavum_mask_cache_300 \
  --cache-dataset exp1_cu_full \
  --mask-token-checkpoint yolo_mask_adapter/results/ligamentum_flavum_dense_condition_reinit_all_bal600_resume_fidelity1_more_e5_cuda0/best.pt \
  --config-file configs/sam2.1/sam2.1_hiera_s_rvos.yaml \
  --training-config-file configs/rvos_training/17/sam2.1_s_ref17_resurgsam \
  --ckpt-path checkpoints/sam2.1_hiera_s_ref17.pth \
  --apply-long-term-memory \
  --disable-obj-score-gating \
  --use-mask-as-output \
  --num-cifs-candidate-frame 1 \
  --num-cand-to-cond-frame 1 \
  --credible-obj-score-threshold 0.0 \
  --credible-iou-threshold 0.0 \
  --hydra-override ++scratch.forward_text_emb=false \
  --hydra-override ++model.use_obj_ptrs_in_encoder=false \
  --device cuda:0 \
  --debug \
  --output-json yolo_mask_adapter/results/debug_yolotoken_forced_cond_no_objptr_ligament_5gt.json \
  2>&1 | tee yolo_mask_adapter/results/logs/debug_yolotoken_forced_cond_no_objptr_ligament_5gt.log
