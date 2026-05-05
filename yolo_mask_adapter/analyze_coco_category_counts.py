"""Analyze COCO category frequencies for the surgical datasets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_dataset(dataset_root: Path) -> dict:
    ann_path = dataset_root / "annotations" / "instances_default.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
    stats = {
        cat_id: {
            "category_id": cat_id,
            "category_name": name,
            "annotations": 0,
            "images": set(),
            "area_sum": 0.0,
        }
        for cat_id, name in categories.items()
    }
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id not in stats:
            stats[cat_id] = {
                "category_id": cat_id,
                "category_name": str(cat_id),
                "annotations": 0,
                "images": set(),
                "area_sum": 0.0,
            }
        row = stats[cat_id]
        row["annotations"] += 1
        row["images"].add(ann.get("image_id"))
        row["area_sum"] += float(ann.get("area", 0.0) or 0.0)

    rows = []
    for row in stats.values():
        annotations = row["annotations"]
        rows.append(
            {
                "category_id": row["category_id"],
                "category_name": row["category_name"],
                "annotations": annotations,
                "images": len(row["images"]),
                "mean_area": row["area_sum"] / annotations if annotations else 0.0,
            }
        )
    rows.sort(key=lambda item: (-item["annotations"], item["category_id"]))
    return {
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "num_images_total": len(data.get("images", [])),
        "num_annotations_total": len(data.get("annotations", [])),
        "categories": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "Datasets/exp1_cu_full",
            "Datasets/exp2_cu_full",
            "Datasets/exp1_cu",
            "Datasets/exp2_cu",
            "Datasets/seg_class_v3",
        ],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="Code/ReSurgSAM2/yolo_mask_adapter/results/category_counts.json")
    args = parser.parse_args()

    datasets = [analyze_dataset(Path(path)) for path in args.datasets]
    combined = defaultdict(lambda: {"category_id": None, "category_name": "", "annotations": 0, "images": 0})
    for dataset in datasets:
        for row in dataset["categories"]:
            item = combined[row["category_id"]]
            item["category_id"] = row["category_id"]
            item["category_name"] = row["category_name"]
            item["annotations"] += row["annotations"]
            item["images"] += row["images"]
    combined_rows = sorted(combined.values(), key=lambda item: (-item["annotations"], item["category_id"]))

    output = {"datasets": datasets, "combined": combined_rows}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    for dataset in datasets:
        print(f"\n{dataset['dataset']} images={dataset['num_images_total']} anns={dataset['num_annotations_total']}")
        for row in dataset["categories"][: args.top_k]:
            print(
                f"  id={row['category_id']:>2} anns={row['annotations']:>5} "
                f"images={row['images']:>5} mean_area={row['mean_area']:.1f} {row['category_name']}"
            )
    print("\ncombined")
    for row in combined_rows[: args.top_k]:
        print(
            f"  id={row['category_id']:>2} anns={row['annotations']:>5} "
            f"images={row['images']:>5} {row['category_name']}"
        )
    print(f"\nsaved={out_path}")


if __name__ == "__main__":
    main()
