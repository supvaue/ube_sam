#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin
mkdir -p Code/ReSurgSAM2/yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

YOLO_SRC=/root/share/Zhuzhelin/Code/ultralytics-main/runs/segment/runs/joint_train_v2/weights/best.pt
YOLO_TMP=/tmp/resurgsam2_runtime_cache/yolo_joint_train_v2_best.pt
MANIFEST_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_manifest.json
MANIFEST_TMP=/tmp/resurgsam2_runtime_cache/forceps_autolabel_exp1_track_clips_manifest.json
CACHE_FINAL=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_autolabel_exp1_track_clips_mask_cache
CACHE_TMP=/tmp/resurgsam2_runtime_cache/forceps_autolabel_exp1_track_clips_mask_cache
LOG_PATH=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/logs/build_forceps_autolabel_exp1_track_clips_mask_cache.log

if [[ ! -s "${YOLO_TMP}" ]]; then
  echo "[stage] copy YOLO checkpoint to ${YOLO_TMP}"
  cp "${YOLO_SRC}" "${YOLO_TMP}"
fi

echo "[stage] copy manifest to ${MANIFEST_TMP}"
cp "${MANIFEST_SRC}" "${MANIFEST_TMP}"

mkdir -p "${CACHE_FINAL}" "${CACHE_TMP}"
FINAL_COUNT=$(find "${CACHE_FINAL}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
TMP_COUNT=$(find "${CACHE_TMP}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [[ "${FINAL_COUNT}" -gt "${TMP_COUNT}" ]]; then
  echo "[stage] warm tmp cache from final cache (${FINAL_COUNT} files)"
  cp -n "${CACHE_FINAL}"/*.npz "${CACHE_TMP}/" 2>/dev/null || true
fi

echo "[stage] build forceps autolabel mask cache in ${CACHE_TMP}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONDONTWRITEBYTECODE=1 \
/root/anaconda3/bin/python -u Code/ReSurgSAM2/yolo_mask_adapter/build_forceps_mask_cache.py \
  --manifest "${MANIFEST_TMP}" \
  --yolo "${YOLO_TMP}" \
  --output-dir "${CACHE_TMP}" \
  --datasets autolabel_forceps_exp1_iou05_track_clips \
  --target-yolo-cls 11 \
  --imgsz 640 \
  --conf 0.20 \
  --iou 0.70 \
  --device 0 \
  --sample-mode ordered \
  --progress-every 50 \
  --skip-existing \
  2>&1 | tee "${LOG_PATH}"

echo "[stage] sync tmp cache back to ${CACHE_FINAL}"
cp -n "${CACHE_TMP}"/*.npz "${CACHE_FINAL}/" 2>/dev/null || true
cp "${CACHE_TMP}/summary.json" "${CACHE_FINAL}/summary.json"
echo "[stage] final cache count: $(find "${CACHE_FINAL}" -maxdepth 1 -name '*.npz' | wc -l)"
