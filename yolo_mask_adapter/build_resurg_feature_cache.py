"""Cache frozen ReSurgSAM2 low-level features for adapter training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_frozen_resurgsam2(device: str):
    model = build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_s_rvos.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_s_ref17.pth",
        device=device,
        strict_loading=False,
        apply_long_term_memory=True,
        hydra_overrides_extra=[
            "++scratch.use_sp_bimamba=true",
            "++scratch.use_dwconv=true",
            "++model.use_mask_input_as_output_without_sam=false",
        ],
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_image(path: Path, image_size: int) -> torch.Tensor:
    if not path.exists() and not path.is_absolute():
        path = WORKSPACE_ROOT / path
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    return (image_t - IMG_MEAN) / IMG_STD


@torch.inference_mode()
def extract_low_feature(model, image: torch.Tensor) -> torch.Tensor:
    backbone_out = model.forward_image(image[None])
    _, current_vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    h, w = feat_sizes[-1]
    return current_vision_feats[-1].permute(1, 2, 0).reshape(model.hidden_dim, h, w)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-cache-dir", default="yolo_mask_adapter/results/forceps_mask_cache_full")
    parser.add_argument("--output-dir", default="yolo_mask_adapter/results/resurg_feature_cache_full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    mask_cache_dir = Path(args.mask_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_frozen_resurgsam2(args.device)

    rows = []
    paths = sorted(mask_cache_dir.glob("*.npz"))
    if args.max_items:
        paths = paths[: args.max_items]
    for idx, path in enumerate(paths, 1):
        data = np.load(path, allow_pickle=False)
        stem = path.stem
        out_path = out_dir / f"{stem}.npz"
        image = load_image(Path(str(data["image_path"])), args.image_size).to(args.device)
        feature = extract_low_feature(model, image).detach().cpu().to(torch.float16).numpy()
        np.savez_compressed(
            out_path,
            feature=feature,
            gt_mask=data["gt_mask"],
            yolo_mask=data["yolo_mask"],
            dataset=str(data["dataset"]),
            file_name=str(data["file_name"]),
            frame_number=int(data["frame_number"]),
            yolo_conf=float(data["yolo_conf"]),
        )
        rows.append({"source": str(path), "cache_path": str(out_path)})
        if args.progress_every and idx % args.progress_every == 0:
            print(f"cached {idx}/{len(paths)}", flush=True)

    summary = {"count": len(rows), "rows": rows}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"count": len(rows), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

