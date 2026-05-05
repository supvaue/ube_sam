#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs

RUNTIME_ROOT=/tmp/resurgsam2_runtime_cache
BASE_CKPT=${RUNTIME_ROOT}/sam2.1_hiera_s_ref17.pth
CACHE_DIR=${RUNTIME_ROOT}/ligamentum_flavum_mask_cache_balanced_600
IMAGE_CACHE_ROOT=${RUNTIME_ROOT}/image_cache

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[stage] copy base checkpoint to ${BASE_CKPT}"
  mkdir -p "${RUNTIME_ROOT}"
  cp checkpoints/sam2.1_hiera_s_ref17.pth "${BASE_CKPT}"
fi

if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "[error] ${CACHE_DIR} does not exist. Run run_train_ligament_prior_mixer.sh once to populate /tmp cache." >&2
  exit 1
fi

OLD_CKPT=yolo_mask_adapter/results/ligamentum_flavum_dense_condition_reinit_all_bal600_resume_fidelity1_more_e5_cuda0/best.pt
NEW_CKPT=yolo_mask_adapter/results/ligamentum_flavum_mask_prior_mixer_d1_from_best_e5/best.pt
AUX_CKPT=yolo_mask_adapter/results/ligamentum_flavum_mask_prior_aux_d1_adapter_only_e5/best.pt
OUT_JSON=yolo_mask_adapter/results/ligamentum_flavum_mask_prior_feature_probe_old_vs_d1_vs_aux.json
LOG_FILE=yolo_mask_adapter/results/logs/eval_mask_prior_features_ligament_$(date +%Y%m%d_%H%M%S).log

CHECKPOINTS=("${OLD_CKPT}" "${NEW_CKPT}")
if [[ -f "${AUX_CKPT}" ]]; then
  CHECKPOINTS+=("${AUX_CKPT}")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/eval_mask_prior_features.py \
  --cache-dir "${CACHE_DIR}" \
  --datasets exp1_cu_full,exp2_cu_full \
  --max-items 600 \
  --val-fraction 0.2 \
  --split-mode interleaved \
  --split-seed 0 \
  --class-id 1 \
  --base-ckpt-path "${BASE_CKPT}" \
  --image-cache-root "${IMAGE_CACHE_ROOT}" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --batch-size 4 \
  --device cuda:0 \
  --probe-epochs 200 \
  --output-json "${OUT_JSON}" \
  2>&1 | tee "${LOG_FILE}"

echo "[stage] wrote ${OUT_JSON}"
