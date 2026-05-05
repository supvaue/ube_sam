"""Analyze GT-vs-YOLO quality for a mask cache directory."""

import argparse
import json
from pathlib import Path

import numpy as np


def mask_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred_area + gt_area) if pred_area + gt_area else 1.0
    precision = inter / pred_area if pred_area else float(gt_area == 0)
    recall = inter / gt_area if gt_area else float(pred_area == 0)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "gt_area": float(gt.mean()),
        "yolo_area": float(pred.mean()),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    keys = ["iou", "dice", "precision", "recall", "gt_area", "yolo_area", "yolo_conf"]
    out = {"count": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(values.mean())
        out[f"p10_{key}"] = float(np.quantile(values, 0.10))
        out[f"p50_{key}"] = float(np.quantile(values, 0.50))
        out[f"p90_{key}"] = float(np.quantile(values, 0.90))
    out["with_yolo_mask"] = int(sum(row["yolo_area"] > 0 for row in rows))
    out["low_iou_lt_050"] = int(sum(row["iou"] < 0.50 for row in rows))
    out["low_iou_lt_070"] = int(sum(row["iou"] < 0.70 for row in rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--worst-k", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.cache_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        metrics = mask_metrics(data["gt_mask"], data["yolo_mask"])
        metrics.update(
            {
                "cache_path": str(path),
                "dataset": str(data["dataset"]),
                "file_name": str(data["file_name"]),
                "frame_number": int(data["frame_number"]),
                "yolo_conf": float(data["yolo_conf"]),
            }
        )
        rows.append(metrics)

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_dataset[dataset] = summarize([row for row in rows if row["dataset"] == dataset])

    low_quality = sorted(rows, key=lambda row: row["iou"])[: args.worst_k]
    out = {
        "cache_dir": args.cache_dir,
        "overall": summarize(rows),
        "by_dataset": by_dataset,
        "worst": low_quality,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
