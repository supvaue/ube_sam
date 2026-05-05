#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_mask_cache
CACHE_TMP=/tmp/resurgsam2_runtime_cache/forceps_autolabel_exp1_track_clips_mask_cache
NO_MEM_CKPT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_no_memory_decoder_autolabel_film_mlp_nomemattn_reinit_e8/best.pt
SHARED_CKPT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_shared_decoder_autolabel_film_mlp_memonly_e4/best.pt

if [[ ! -s "${BASE_CKPT_TMP}" ]]; then
  echo "[stage] copy base checkpoint to ${BASE_CKPT_TMP}"
  cp "${BASE_CKPT_SRC}" "${BASE_CKPT_TMP}"
fi

CACHE_COUNT=$(find "${CACHE_TMP}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
SRC_COUNT=$(find "${CACHE_SRC}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [[ "${CACHE_COUNT}" -lt "${SRC_COUNT}" ]]; then
  echo "[stage] copy forceps autolabel cache to ${CACHE_TMP}"
  mkdir -p "${CACHE_TMP}"
  cp -n "${CACHE_SRC}"/*.npz "${CACHE_TMP}/"
fi

echo "[stage] eval pair-level refined-memory gap=1"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/eval_pair_refined_memory.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets autolabel_forceps_exp1_iou05_track_clips \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --resume-checkpoint "${SHARED_CKPT}" \
  --mask-token-checkpoint "${NO_MEM_CKPT}" \
  --condition-fusion-mode film_mlp \
  --target-class-id 12 \
  --max-gap 1 \
  --max-items 100 \
  --batch 1 \
  --device cuda:0 \
  --refined-memory-mode raw_logits \
  --force-memory-object-score 10.0 \
  --disable-non-cond-memory \
  --disable-obj-score-gating \
  --apply-long-term-memory \
  --use-real-gap \
  --output-json yolo_mask_adapter/results/pair_refined_memory_gap1_eval.json \
  2>&1 | tee yolo_mask_adapter/results/logs/pair_refined_memory_gap1_eval.log
