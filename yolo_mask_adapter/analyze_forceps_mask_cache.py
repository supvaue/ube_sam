"""Analyze cached GT/YOLO forceps mask quality."""

import argparse
import json
from pathlib import Path

import numpy as np


def mask_stats(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    gt_area = gt.sum()
    pred_area = pred.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (gt_area + pred_area) if (gt_area + pred_area) else 1.0
    precision = inter / pred_area if pred_area else 0.0
    recall = inter / gt_area if gt_area else 0.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "gt_area": float(gt.mean()),
        "pred_area": float(pred.mean()),
        "has_pred": bool(pred_area > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    rows = []
    for path in sorted(cache_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        stats = mask_stats(data["gt_mask"], data["yolo_mask"])
        rows.append(
            {
                "cache_path": str(path),
                "dataset": str(data["dataset"]),
                "file_name": str(data["file_name"]),
                "frame_number": int(data["frame_number"]),
                "yolo_conf": float(data["yolo_conf"]),
                **stats,
            }
        )

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {}
        return {
            "count": len(items),
            "with_pred": sum(1 for r in items if r["has_pred"]),
            "mean_iou": float(np.mean([r["iou"] for r in items])),
            "mean_dice": float(np.mean([r["dice"] for r in items])),
            "mean_precision": float(np.mean([r["precision"] for r in items])),
            "mean_recall": float(np.mean([r["recall"] for r in items])),
            "mean_yolo_conf": float(np.mean([r["yolo_conf"] for r in items])),
        }

    summary = {
        "overall": summarize(rows),
        "by_dataset": {name: summarize(items) for name, items in sorted(by_dataset.items())},
        "worst_iou": sorted(rows, key=lambda r: r["iou"])[:10],
        "rows": rows,
    }
    output = Path(args.output) if args.output else cache_dir / "quality_summary.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"summary={output}")


if __name__ == "__main__":
    main()

