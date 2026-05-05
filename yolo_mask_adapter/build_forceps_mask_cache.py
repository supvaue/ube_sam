"""Cache GT and YOLO masks for forceps adapter training/evaluation."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT.parent / "ultralytics-main"


def register_joint_yolo_class() -> None:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    import torch.nn as nn
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.tasks import SegmentationModel

    class JointSegClsModel(SegmentationModel):
        def __init__(self, cfg="yolo11m-seg.yaml", nc=28, nc_cls=5, verbose=True):
            self._p5 = [None]
            super().__init__(cfg=cfg, nc=nc, verbose=verbose)
            self._p5[0] = None
            self.nc_cls = nc_cls
            self.cls_head = nn.Sequential(
                Conv(512, 256, 1, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(256, nc_cls),
            )

        def _predict_once(self, x, profile=False, visualize=False, embed=None):
            y, dt, embeddings = [], [], []
            embed = frozenset(embed) if embed is not None else {-1}
            for i, m in enumerate(self.model):
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
                y.append(x if m.i in self.save else None)
                if i == 22:
                    self._p5[0] = x
            return x

    globals()["JointSegClsModel"] = JointSegClsModel


def decode_compressed_rle_counts(encoded: str) -> list[int]:
    counts = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            char_value = ord(encoded[position]) - 48
            position += 1
            value |= (char_value & 0x1F) << shift
            shift += 5
            if not (char_value & 0x20):
                if char_value & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_rle(segmentation: dict, height: int, width: int) -> np.ndarray:
    counts = segmentation.get("counts", [])
    if isinstance(counts, str):
        counts = decode_compressed_rle_counts(counts)
    elif isinstance(counts, bytes):
        counts = decode_compressed_rle_counts(counts.decode("ascii"))
    size = segmentation.get("size", [height, width])
    rle_height, rle_width = int(size[0]), int(size[1])
    flat = np.zeros(rle_height * rle_width, dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        next_offset = min(offset + int(count), flat.size)
        if value:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value
    mask = flat.reshape((rle_height, rle_width), order="F")
    if (rle_height, rle_width) != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def rasterize_polygons(annotations: list[dict], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, dict):
            mask |= decode_rle(segmentation, height=height, width=width)
            continue
        if not isinstance(segmentation, list):
            raise TypeError(f"Unsupported segmentation type: {type(segmentation).__name__}")
        for poly in segmentation:
            if isinstance(poly, dict):
                mask |= decode_rle(poly, height=height, width=width)
                continue
            if len(poly) < 6:
                continue
            points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


def yolo_forceps_mask(result, target_cls: int, height: int, width: int) -> tuple[np.ndarray, float]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    cls_ids = result.boxes.cls.detach().cpu().long()
    confs = result.boxes.conf.detach().cpu().float()
    candidates = (cls_ids == target_cls).nonzero(as_tuple=False).flatten()
    if len(candidates) == 0:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    best = candidates[confs[candidates].argmax()].item()
    mask = result.masks.data[best].detach().cpu().float()
    if tuple(mask.shape) != (height, width):
        mask = torch.nn.functional.interpolate(
            mask[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0]
    return (mask.numpy() > 0.5).astype(np.uint8), float(confs[best].item())


def iter_entries(manifest: dict, dataset_names: set[str] | None):
    for dataset in manifest["datasets"]:
        if dataset_names and dataset["dataset"] not in dataset_names:
            continue
        for entry in dataset["entries"]:
            yield entry


def grouped_entries(manifest: dict, dataset_names: set[str] | None) -> dict[str, list[dict]]:
    groups = {}
    for dataset in manifest["datasets"]:
        if dataset_names and dataset["dataset"] not in dataset_names:
            continue
        entries = sorted(dataset["entries"], key=lambda e: (e["frame_number"], e["file_name"]))
        groups[dataset["dataset"]] = entries
    return groups


def uniform_temporal_sample(entries: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if count >= len(entries):
        return list(entries)
    indices = np.linspace(0, len(entries) - 1, count, dtype=int).tolist()
    return [entries[index] for index in indices]


def random_sample(entries: list[dict], count: int, rng: np.random.Generator) -> list[dict]:
    if count <= 0:
        return []
    if count >= len(entries):
        return list(entries)
    indices = sorted(rng.choice(len(entries), size=count, replace=False).tolist())
    return [entries[index] for index in indices]


def select_entries(
    manifest: dict,
    dataset_names: set[str] | None,
    max_items: int,
    sample_mode: str,
    sample_seed: int,
) -> tuple[list[dict], dict]:
    if sample_mode == "ordered":
        entries = list(iter_entries(manifest, dataset_names))
        if max_items:
            entries = entries[:max_items]
        return entries, {"sample_mode": sample_mode, "max_items": max_items}

    groups = grouped_entries(manifest, dataset_names)
    if not groups:
        return [], {"sample_mode": sample_mode, "max_items": max_items, "groups": {}}

    rng = np.random.default_rng(sample_seed)
    if max_items:
        per_dataset = max_items // len(groups)
        remainder = max_items % len(groups)
        quotas = {
            dataset: min(len(entries), per_dataset + (idx < remainder))
            for idx, (dataset, entries) in enumerate(sorted(groups.items()))
        }
        leftover = max_items - sum(quotas.values())
        if leftover > 0:
            for dataset, entries in sorted(groups.items(), key=lambda item: len(item[1]) - quotas[item[0]], reverse=True):
                add = min(leftover, len(entries) - quotas[dataset])
                quotas[dataset] += add
                leftover -= add
                if leftover == 0:
                    break
    else:
        quotas = {dataset: len(entries) for dataset, entries in groups.items()}

    selected = []
    sampling_summary = {
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "max_items": max_items,
        "groups": {},
    }
    for dataset, entries in sorted(groups.items()):
        quota = quotas[dataset]
        if sample_mode == "dataset_uniform":
            dataset_entries = uniform_temporal_sample(entries, quota)
        elif sample_mode == "dataset_random":
            dataset_entries = random_sample(entries, quota, rng)
        else:
            raise ValueError(f"Unknown sample mode: {sample_mode}")
        selected.extend(dataset_entries)
        sampling_summary["groups"][dataset] = {
            "available": len(entries),
            "selected": len(dataset_entries),
            "frame_range": [
                dataset_entries[0]["frame_number"],
                dataset_entries[-1]["frame_number"],
            ]
            if dataset_entries
            else None,
        }
    selected.sort(key=lambda e: (e["dataset"], e["frame_number"], e["file_name"]))
    return selected, sampling_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_manifest.json",
    )
    parser.add_argument(
        "--yolo",
        default="Code/ultralytics-main/runs/segment/runs/joint_train_v2/weights/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache",
    )
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--target-yolo-cls", type=int, default=11)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        choices=["ordered", "dataset_uniform", "dataset_random"],
        default="ordered",
        help="ordered keeps old behavior; dataset_uniform balances datasets and samples each dataset across time.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    register_joint_yolo_class()
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = set(args.datasets) if args.datasets else None
    selected_entries, sampling_summary = select_entries(
        manifest,
        dataset_names=dataset_names,
        max_items=args.max_items,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )

    rows = []
    for index, entry in enumerate(selected_entries, start=1):
        image_path = Path(entry["image_path"])
        height, width = int(entry["height"]), int(entry["width"])
        stem = f"{entry['dataset']}__{Path(entry['file_name']).stem}"
        npz_path = out_dir / f"{stem}.npz"
        if args.skip_existing and npz_path.exists():
            cached = np.load(npz_path, allow_pickle=False)
            gt_mask = cached["gt_mask"]
            pred_mask = cached["yolo_mask"]
            pred_conf = float(cached["yolo_conf"])
        else:
            gt_mask = rasterize_polygons(entry["annotations"], height=height, width=width)
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            pred_mask, pred_conf = yolo_forceps_mask(result, args.target_yolo_cls, height, width)
            np.savez_compressed(
                npz_path,
                gt_mask=gt_mask,
                yolo_mask=pred_mask,
                image_path=str(image_path),
                dataset=entry["dataset"],
                file_name=entry["file_name"],
                frame_number=entry["frame_number"],
                yolo_conf=pred_conf,
            )
        rows.append(
            {
                "dataset": entry["dataset"],
                "file_name": entry["file_name"],
                "frame_number": entry["frame_number"],
                "cache_path": str(npz_path),
                "gt_area": float(gt_mask.mean()),
                "yolo_area": float(pred_mask.mean()),
                "yolo_conf": pred_conf,
            }
        )
        if args.progress_every and (index % args.progress_every == 0 or index == len(selected_entries)):
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(selected_entries),
                        "dataset": entry["dataset"],
                        "frame_number": entry["frame_number"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "count": len(rows),
        "with_yolo_mask": sum(1 for row in rows if row["yolo_area"] > 0),
        "mean_gt_area": float(np.mean([row["gt_area"] for row in rows])) if rows else 0.0,
        "mean_yolo_area": float(np.mean([row["yolo_area"] for row in rows])) if rows else 0.0,
        "sampling": sampling_summary,
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
