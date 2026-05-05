#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_mask_cache
CACHE_TMP=/tmp/resurgsam2_runtime_cache/forceps_autolabel_exp1_track_clips_mask_cache

if [[ ! -s "${BASE_CKPT_TMP}" ]]; then
  echo "[stage] copy base checkpoint to ${BASE_CKPT_TMP}"
  cp "${BASE_CKPT_SRC}" "${BASE_CKPT_TMP}"
fi

CACHE_COUNT=$(find "${CACHE_TMP}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
SRC_COUNT=$(find "${CACHE_SRC}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [[ "${SRC_COUNT}" -lt 2000 ]]; then
  echo "[error] cache is not ready: ${CACHE_SRC} has only ${SRC_COUNT} npz files" >&2
  echo "[hint] run: bash yolo_mask_adapter/run_build_forceps_autolabel_cache.sh" >&2
  exit 1
fi
if [[ "${CACHE_COUNT}" -lt "${SRC_COUNT}" ]]; then
  echo "[stage] copy forceps autolabel cache to ${CACHE_TMP}"
  mkdir -p "${CACHE_TMP}"
  cp -n "${CACHE_SRC}"/*.npz "${CACHE_TMP}/"
fi

echo "[stage] train forceps no-memory-decoder on SAMURAI autolabel track clips"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_mask_token_prompt_adapter.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets autolabel_forceps_exp1_iou05_track_clips \
  --target-class-id 12 \
  --prompt-mode dense_condition \
  --train-parts adapter cstmamba prompt_encoder mask_decoder \
  --reinit-train-parts \
  --disable-obj-score-gating \
  --use-no-mem-attention \
  --yolo-fidelity-weight 0.5 \
  --epochs 8 \
  --batch 2 \
  --lr 1.0e-5 \
  --adapter-lr 2.0e-5 \
  --model-lr 1.0e-5 \
  --prior-mixer-depth 1 \
  --prior-mixer-heads 4 \
  --device cuda:0 \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --output-dir yolo_mask_adapter/results/forceps_no_memory_decoder_autolabel_nomemattn_reinit_e8 \
  2>&1 | tee yolo_mask_adapter/results/logs/train_forceps_no_memory_decoder_autolabel_nomemattn_reinit_e8.log
