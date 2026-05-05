"""Evaluate a trained forceps refinement adapter against YOLO mask cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_forceps_refine_adapter import ForcepsCacheDataset, ForcepsRefineAdapter, batch_metrics


def per_sample_metrics(logits: torch.Tensor, yolo: torch.Tensor, gt: torch.Tensor) -> list[dict]:
    refined = logits.sigmoid() > 0.5
    yolo_b = yolo > 0.5
    gt_b = gt > 0.5
    rows = []
    for pred in [yolo_b, refined]:
        inter = (pred & gt_b).float().sum(dim=(1, 2, 3))
        union = (pred | gt_b).float().sum(dim=(1, 2, 3))
        area_p = pred.float().sum(dim=(1, 2, 3))
        area_g = gt_b.float().sum(dim=(1, 2, 3))
        iou = torch.where(union > 0, inter / union, torch.ones_like(union))
        dice = torch.where(area_p + area_g > 0, 2 * inter / (area_p + area_g), torch.ones_like(union))
        rows.append((iou.detach().cpu().numpy(), dice.detach().cpu().numpy()))
    return [
        {
            "yolo_iou": float(rows[0][0][i]),
            "yolo_dice": float(rows[0][1][i]),
            "refined_iou": float(rows[1][0][i]),
            "refined_dice": float(rows[1][1][i]),
            "delta_iou": float(rows[1][0][i] - rows[0][0][i]),
        }
        for i in range(gt.shape[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/resurg_feature_cache_full")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--datasets", nargs="+", default=["exp2_cu_full"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ds = ForcepsCacheDataset(Path(args.cache_dir), set(args.datasets), has_features=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    adapter = ForcepsRefineAdapter(feature_dim=256).to(args.device)
    ckpt = torch.load(args.weights, map_location=args.device, weights_only=True)
    adapter.load_state_dict(ckpt["model"])
    adapter.eval()

    rows = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["feature"].to(args.device)
            gt = batch["gt_mask"].to(args.device)
            yolo = batch["yolo_mask"].to(args.device)
            logits = adapter(features, yolo)
            metrics = per_sample_metrics(logits, yolo, gt)
            for i, metric in enumerate(metrics):
                metric.update(
                    {
                        "dataset": batch["dataset"][i],
                        "file_name": batch["file_name"][i],
                    }
                )
                rows.append(metric)

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {}
        return {
            "count": len(items),
            "mean_yolo_iou": float(np.mean([r["yolo_iou"] for r in items])),
            "mean_refined_iou": float(np.mean([r["refined_iou"] for r in items])),
            "mean_delta_iou": float(np.mean([r["delta_iou"] for r in items])),
            "improved_count": sum(1 for r in items if r["delta_iou"] > 1e-6),
        }

    low = [r for r in rows if r["yolo_iou"] < 0.6]
    mid = [r for r in rows if 0.6 <= r["yolo_iou"] < 0.85]
    high = [r for r in rows if r["yolo_iou"] >= 0.85]
    summary = {
        "overall": summarize(rows),
        "low_yolo_iou_lt_0.6": summarize(low),
        "mid_yolo_iou_0.6_0.85": summarize(mid),
        "high_yolo_iou_ge_0.85": summarize(high),
        "worst_delta": sorted(rows, key=lambda r: r["delta_iou"])[:10],
        "best_delta": sorted(rows, key=lambda r: r["delta_iou"], reverse=True)[:10],
        "rows": rows,
    }
    output = Path(args.output) if args.output else Path(args.weights).parent / "eval_refine.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"summary={output}")


if __name__ == "__main__":
    main()

