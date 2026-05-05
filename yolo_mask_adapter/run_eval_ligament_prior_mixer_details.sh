#!/usr/bin/env bash
set -euo pipefail

cd /root/share/Zhuzhelin/Code/ReSurgSAM2
mkdir -p yolo_mask_adapter/results/logs
mkdir -p /tmp/resurgsam2_runtime_cache

BASE_CKPT_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/checkpoints/resurgsam2_downloaded/sam2.1_hiera_s_ref17.pth
BASE_CKPT_TMP=/tmp/resurgsam2_runtime_cache/sam2.1_hiera_s_ref17.pth
CACHE_SRC=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_mask_cache_balanced_600
CACHE_TMP=/tmp/resurgsam2_runtime_cache/ligamentum_flavum_mask_cache_balanced_600
IMAGE_CACHE_ROOT=/tmp/resurgsam2_runtime_cache/image_cache

OLD_CKPT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_dense_condition_reinit_all_bal600_resume_fidelity1_more_e5_cuda0/best.pt
NEW_CKPT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_mask_prior_mixer_d1_from_best_e5/best.pt
OLD_OUT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_eval_details_old_best
NEW_OUT=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_eval_details_prior_mixer_d1_best
COMPARE_JSON=/root/share/Zhuzhelin/Code/ReSurgSAM2/yolo_mask_adapter/results/ligamentum_flavum_prior_mixer_d1_vs_old_best_details_compare.json

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

echo "[stage] eval old best"
CUDA_VISIBLE_DEVICES=0 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_mask_token_prompt_adapter.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets exp1_cu_full exp2_cu_full \
  --target-class-id 1 \
  --prompt-mode dense_condition \
  --train-parts adapter cstmamba prompt_encoder mask_decoder \
  --disable-obj-score-gating \
  --yolo-fidelity-weight 1.0 \
  --batch 2 \
  --prior-mixer-depth 0 \
  --prior-mixer-heads 4 \
  --device cuda:0 \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --resume-checkpoint "${OLD_CKPT}" \
  --image-cache-root "${IMAGE_CACHE_ROOT}" \
  --eval-only \
  --eval-details-json "${OLD_OUT}/details.json" \
  --output-dir "${OLD_OUT}" \
  2>&1 | tee yolo_mask_adapter/results/logs/eval_ligament_old_best_details.log

echo "[stage] eval prior mixer d1 best"
CUDA_VISIBLE_DEVICES=0 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/share/Zhuzhelin/Code/ReSurgSAM2:/root/share/Zhuzhelin/Code/ReSurgSAM2/mamba:${PYTHONPATH:-} \
/root/anaconda3/bin/python -u yolo_mask_adapter/train_mask_token_prompt_adapter.py \
  --cache-dir "${CACHE_TMP}" \
  --train-datasets exp1_cu_full exp2_cu_full \
  --target-class-id 1 \
  --prompt-mode dense_condition \
  --train-parts adapter cstmamba prompt_encoder mask_decoder \
  --disable-obj-score-gating \
  --yolo-fidelity-weight 1.0 \
  --batch 2 \
  --prior-mixer-depth 1 \
  --prior-mixer-heads 4 \
  --device cuda:0 \
  --base-ckpt-path "${BASE_CKPT_TMP}" \
  --resume-checkpoint "${NEW_CKPT}" \
  --image-cache-root "${IMAGE_CACHE_ROOT}" \
  --eval-only \
  --eval-details-json "${NEW_OUT}/details.json" \
  --output-dir "${NEW_OUT}" \
  2>&1 | tee yolo_mask_adapter/results/logs/eval_ligament_prior_mixer_d1_best_details.log

echo "[stage] compare details"
/root/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

old_path = Path("yolo_mask_adapter/results/ligamentum_flavum_eval_details_old_best/details.json")
new_path = Path("yolo_mask_adapter/results/ligamentum_flavum_eval_details_prior_mixer_d1_best/details.json")
out_path = Path("yolo_mask_adapter/results/ligamentum_flavum_prior_mixer_d1_vs_old_best_details_compare.json")

old = json.loads(old_path.read_text())
new = json.loads(new_path.read_text())

old_rows = {(r["dataset"], r["file_name"]): r for r in old["rows"]} if "rows" in old else {}
new_rows = {(r["dataset"], r["file_name"]): r for r in new["rows"]} if "rows" in new else {}

rows = []
for key in sorted(set(old_rows) & set(new_rows)):
    o = old_rows[key]
    n = new_rows[key]
    rows.append({
        "dataset": key[0],
        "file_name": key[1],
        "old_iou": o["model_iou"],
        "new_iou": n["model_iou"],
        "yolo_iou": n["yolo_iou"],
        "delta_new_old": n["model_iou"] - o["model_iou"],
        "old_delta_yolo": o["delta_iou"],
        "new_delta_yolo": n["delta_iou"],
    })

def mean(values):
    return sum(values) / len(values) if values else 0.0

by_dataset = {}
for dataset in sorted({r["dataset"] for r in rows}):
    ds = [r for r in rows if r["dataset"] == dataset]
    by_dataset[dataset] = {
        "count": len(ds),
        "mean_delta_new_old": mean([r["delta_new_old"] for r in ds]),
        "improved_count": sum(r["delta_new_old"] > 0 for r in ds),
        "worse_count": sum(r["delta_new_old"] < 0 for r in ds),
    }

compare = {
    "old_overall": old["overall"],
    "new_overall": new["overall"],
    "count": len(rows),
    "mean_delta_new_old": mean([r["delta_new_old"] for r in rows]),
    "improved_count": sum(r["delta_new_old"] > 0 for r in rows),
    "worse_count": sum(r["delta_new_old"] < 0 for r in rows),
    "by_dataset": by_dataset,
    "best_new_old": sorted(rows, key=lambda r: r["delta_new_old"], reverse=True)[:20],
    "worst_new_old": sorted(rows, key=lambda r: r["delta_new_old"])[:20],
}
out_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2))
print(json.dumps({
    "compare_json": str(out_path),
    "mean_delta_new_old": compare["mean_delta_new_old"],
    "improved_count": compare["improved_count"],
    "worse_count": compare["worse_count"],
}, ensure_ascii=False))
PY
