"""Build ordered single-target manifests for forceps experiments."""

import argparse
import json
import re
from pathlib import Path


FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def frame_number(file_name: str) -> int:
    match = FRAME_RE.search(Path(file_name).stem)
    return int(match.group(1)) if match else -1


def build_dataset_entries(dataset_root: Path, category_id: int) -> dict:
    ann_path = dataset_root / "annotations" / "instances_default.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        if ann.get("category_id") != category_id:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    entries = []
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        file_name = img["file_name"]
        image_path = dataset_root / file_name
        if not image_path.exists():
            image_path = dataset_root / "images" / file_name
        entries.append(
            {
                "dataset": dataset_root.name,
                "dataset_root": str(dataset_root),
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(image_path),
                "exists": image_path.exists(),
                "frame_number": frame_number(file_name),
                "height": img.get("height"),
                "width": img.get("width"),
                "category_id": category_id,
                "category_name": categories.get(category_id, str(category_id)),
                "annotations": [
                    {
                        "id": ann.get("id"),
                        "bbox": ann.get("bbox"),
                        "area": ann.get("area"),
                        "iscrowd": ann.get("iscrowd", 0),
                        "segmentation": ann.get("segmentation"),
                    }
                    for ann in anns
                ],
            }
        )

    entries.sort(key=lambda x: (x["frame_number"], x["file_name"]))
    return {
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "annotation_path": str(ann_path),
        "category_id": category_id,
        "category_name": categories.get(category_id, str(category_id)),
        "num_images_total": len(images),
        "num_forceps_images": len(entries),
        "num_forceps_annotations": sum(len(e["annotations"]) for e in entries),
        "missing_images": [e["image_path"] for e in entries if not e["exists"]],
        "entries": entries,
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
        ],
    )
    parser.add_argument("--category-id", type=int, default=12)
    parser.add_argument(
        "--output",
        default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_manifest.json",
    )
    args = parser.parse_args()

    manifests = [
        build_dataset_entries(Path(dataset), category_id=args.category_id)
        for dataset in args.datasets
    ]
    out = {
        "category_id": args.category_id,
        "category_name": manifests[0]["category_name"] if manifests else str(args.category_id),
        "datasets": manifests,
        "summary": {
            item["dataset"]: {
                "num_images_total": item["num_images_total"],
                "num_forceps_images": item["num_forceps_images"],
                "num_forceps_annotations": item["num_forceps_annotations"],
                "missing_images": len(item["missing_images"]),
            }
            for item in manifests
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
