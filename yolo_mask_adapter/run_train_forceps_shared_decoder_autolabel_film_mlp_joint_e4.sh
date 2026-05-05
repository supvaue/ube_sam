#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_mask_cache
CACHE_TMP=/tmp/resurgsam2_runtime_cache/forceps_autolabel_exp1_track_clips_mask_cache
NO_MEM_CKPT=${NO_MEM_CKPT:-/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_no_memory_decoder_autolabel_film_mlp_nomemattn_reinit_e8/best.pt}
OUT_DIR=yolo_mask_adapter/results/forceps_shared_decoder_autolabel_film_mlp_joint_e4
LOG_PATH=yolo_mask_adapter/results/logs/train_forceps_shared_decoder_autolabel_film_mlp_joint_e4.log

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
if [[ ! -s "${NO_MEM_CKPT}" ]]; then
  echo "[error] no-memory checkpoint not found: ${NO_MEM_CKPT}" >&2
  echo "[hint] first run: bash yolo_mask_adapter/run_train_forceps_no_memory_decoder_autolabel_film_mlp_e8.sh" >&2
  exit 1
fi

echo "[stage] train forceps shared decoder with memory/no-memory joint loss"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_shared_decoder_tracking.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets autolabel_forceps_exp1_iou05_track_clips \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --resume-checkpoint "${NO_MEM_CKPT}" \
  --mask-token-checkpoint "${NO_MEM_CKPT}" \
  --output-dir "${OUT_DIR}" \
  --condition-fusion-mode film_mlp \
  --train-parts cstmamba prompt_encoder mask_decoder memory_attention memory_encoder \
  --target-class-id 12 \
  --start-mask-key start_gt \
  --start-memory-source gt_logits \
  --use-prior-conditioned-key \
  --tracking-prompt-source target_prior_cls \
  --memory-loss-weight 1.0 \
  --key-loss-weight 1.0 \
  --epochs 4 \
  --batch 1 \
  --lr 1.0e-6 \
  --split-mode tail \
  --max-gap 1 \
  --device cuda:0 \
  --disable-obj-score-gating \
  --apply-long-term-memory \
  --use-real-gap \
  2>&1 | tee "${LOG_PATH}"

echo "[stage] wrote ${OUT_DIR}"
