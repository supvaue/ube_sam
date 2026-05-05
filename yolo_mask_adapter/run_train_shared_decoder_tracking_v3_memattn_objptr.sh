#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
RESUME_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_mask_prior_mixer_d1_from_best_e5/best.pt
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_mask_cache_balanced_600
CACHE_TMP=/tmp/resurgsam2_runtime_cache/ligamentum_flavum_mask_cache_balanced_600
IMAGE_CACHE_ROOT=/tmp/resurgsam2_runtime_cache/image_cache
OUT_DIR=yolo_mask_adapter/results/shared_decoder_tracking_ligament_v3_memattn_objptr_e3

if [[ ! -s "${BASE_CKPT_TMP}" ]]; then
  echo "[stage] copy base checkpoint to ${BASE_CKPT_TMP}"
  cp "${BASE_CKPT_SRC}" "${BASE_CKPT_TMP}"
fi
CACHE_COUNT=$(find "${CACHE_TMP}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [[ "${CACHE_COUNT}" -lt 600 ]]; then
  echo "[stage] copy balanced mask cache to ${CACHE_TMP}"
  mkdir -p "${CACHE_TMP}"
  cp -n "${CACHE_SRC}"/*.npz "${CACHE_TMP}/"
fi
IMAGE_COUNT=$(find "${IMAGE_CACHE_ROOT}/Datasets" -type f -name 'frame_*.PNG' 2>/dev/null | wc -l || true)
if [[ "${IMAGE_COUNT}" -lt 600 ]]; then
  echo "[stage] copy referenced images to ${IMAGE_CACHE_ROOT}"
  for npz_path in "${CACHE_TMP}"/*.npz; do
    npz_name=$(basename "${npz_path}")
    dataset="${npz_name%%__frame_*}"
    frame_part="${npz_name##*__frame_}"
    frame_part="${frame_part%.npz}"
    src="/root/share/Zhuzhelin/Datasets/${dataset}/images/frame_${frame_part}.PNG"
    dst="${IMAGE_CACHE_ROOT}/Datasets/${dataset}/images/frame_${frame_part}.PNG"
    if [[ -f "${src}" && ! -f "${dst}" ]]; then
      mkdir -p "$(dirname "${dst}")"
      cp -n "${src}" "${dst}"
    fi
  done
fi

echo "[stage] start shared-decoder tracking v3 memory-attention + objptr training"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_shared_decoder_tracking.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets exp1_cu_full exp2_cu_full \
  --image-cache-root "${IMAGE_CACHE_ROOT}" \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --resume-checkpoint "${RESUME_CKPT_SRC}" \
  --mask-token-checkpoint "${RESUME_CKPT_SRC}" \
  --target-class-id 1 \
  --output-dir "${OUT_DIR}" \
  --train-parts prompt_encoder mask_decoder memory_attention obj_ptr \
  --start-mask-key start_gt \
  --start-memory-source gt_logits \
  --use-prior-conditioned-key \
  --tracking-prompt-source target_prior_cls \
  --key-loss-weight 0.25 \
  --epochs 3 \
  --batch 1 \
  --lr 5.0e-7 \
  --max-train-items 120 \
  --max-val-items 40 \
  --device cuda:0 \
  --disable-obj-score-gating \
  2>&1 | tee yolo_mask_adapter/results/logs/train_shared_decoder_tracking_ligament_v3_memattn_objptr_e3.log

echo "[stage] wrote ${OUT_DIR}"
