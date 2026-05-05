#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
RESUME_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_dense_condition_reinit_all_bal600_resume_fidelity1_more_e5_cuda0/best.pt
RESUME_CKPT_TMP=/tmp/resurgsam2_runtime_cache/ligamentum_flavum_best_prior_source.pt
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_mask_cache_balanced_600
CACHE_TMP=/tmp/resurgsam2_runtime_cache/ligamentum_flavum_mask_cache_balanced_600
IMAGE_CACHE_ROOT=/tmp/resurgsam2_runtime_cache/image_cache
OUT_DIR=yolo_mask_adapter/results/ligamentum_flavum_mask_prior_aux_d1_adapter_only_e5

if [[ ! -s "${BASE_CKPT_TMP}" ]]; then
  echo "[stage] copy base checkpoint to ${BASE_CKPT_TMP}"
  cp "${BASE_CKPT_SRC}" "${BASE_CKPT_TMP}"
fi
if [[ ! -s "${RESUME_CKPT_TMP}" ]]; then
  echo "[stage] copy resume checkpoint to ${RESUME_CKPT_TMP}"
  cp "${RESUME_CKPT_SRC}" "${RESUME_CKPT_TMP}"
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
echo "[stage] start prior-feature auxiliary training"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_mask_token_prompt_adapter.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets exp1_cu_full exp2_cu_full \
  --target-class-id 1 \
  --prompt-mode dense_condition \
  --train-parts adapter \
  --disable-obj-score-gating \
  --yolo-fidelity-weight 1.0 \
  --aux-feature-loss-weight 0.05 \
  --aux-feature-source sentence_flat \
  --epochs 5 \
  --batch 2 \
  --lr 1.0e-5 \
  --adapter-lr 1.0e-5 \
  --prior-mixer-depth 1 \
  --prior-mixer-heads 4 \
  --device cuda:0 \
  --resume-checkpoint "${RESUME_CKPT_TMP}" \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --image-cache-root "${IMAGE_CACHE_ROOT}" \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee yolo_mask_adapter/results/logs/train_ligament_prior_aux_d1_adapter_only_e5.log

echo "[stage] wrote ${OUT_DIR}"
