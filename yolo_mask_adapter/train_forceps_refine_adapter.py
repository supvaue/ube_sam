"""Train a lightweight forceps mask refinement adapter on frozen ReSurgSAM2 features."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MASK_CACHE_NAME_RE = re.compile(r"^(?P<dataset>.+)__frame_(?P<frame>\d+)\.npz$")


class ForcepsCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        datasets: set[str],
        image_size: int = 512,
        has_features: bool = False,
        paths: list[Path] | None = None,
    ):
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.has_features = has_features
        self.paths = []
        candidate_paths = sorted(paths) if paths is not None else sorted(cache_dir.glob("*.npz"))
        for path in candidate_paths:
            data = np.load(path, allow_pickle=False)
            if str(data["dataset"]) in datasets:
                self.paths.append(path)
        if not self.paths:
            raise FileNotFoundError(f"No cache files for datasets={sorted(datasets)} in {cache_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        data = np.load(self.paths[index], allow_pickle=False)
        if self.has_features:
            image_t = torch.empty(0)
            feature = torch.from_numpy(data["feature"].astype(np.float32))
        else:
            image_path = Path(str(data["image_path"]))
            if not image_path.exists() and not image_path.is_absolute():
                image_path = WORKSPACE_ROOT / image_path
            image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
            image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
            image_t = (image_t - IMG_MEAN) / IMG_STD
            feature = torch.empty(0)

        gt = torch.from_numpy(data["gt_mask"].astype(np.float32))[None, None]
        yolo = torch.from_numpy(data["yolo_mask"].astype(np.float32))[None, None]
        gt = F.interpolate(gt, size=(self.image_size, self.image_size), mode="nearest")[0]
        yolo = F.interpolate(yolo, size=(self.image_size, self.image_size), mode="nearest")[0]
        return {
            "image": image_t,
            "feature": feature,
            "gt_mask": gt,
            "yolo_mask": yolo,
            "dataset": str(data["dataset"]),
            "file_name": str(data["file_name"]),
        }


def load_cache_index(cache_dir: Path, datasets: set[str]) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.npz")):
        match = MASK_CACHE_NAME_RE.match(path.name)
        if match:
            dataset = match.group("dataset")
            frame_number = int(match.group("frame"))
            file_name = path.name
            if dataset not in datasets:
                data = np.load(path, allow_pickle=False)
                dataset = str(data["dataset"])
                frame_number = int(data["frame_number"])
                file_name = str(data["file_name"])
        else:
            data = np.load(path, allow_pickle=False)
            dataset = str(data["dataset"])
            frame_number = int(data["frame_number"])
            file_name = str(data["file_name"])
        if dataset not in datasets:
            continue
        rows.append(
            {
                "path": path,
                "dataset": dataset,
                "file_name": file_name,
                "frame_number": frame_number,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No cache files for datasets={sorted(datasets)} in {cache_dir}")
    return rows


def split_train_val(
    rows: list[dict],
    val_fraction: float,
    split_mode: str,
    split_seed: int,
) -> tuple[list[Path], list[Path], dict]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    train_paths, val_paths = [], []
    split_summary = {"mode": split_mode, "val_fraction": val_fraction, "seed": split_seed, "datasets": {}}
    rng = np.random.default_rng(split_seed)
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: (row["frame_number"], row["file_name"]),
        )
        n = len(dataset_rows)
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n - 1, n_val)
        if split_mode == "interleaved":
            # Spread validation frames through the video while preserving chronological ordering.
            val_indices = set(np.linspace(0, n - 1, n_val, dtype=int).tolist())
        elif split_mode == "random":
            val_indices = set(rng.choice(n, size=n_val, replace=False).tolist())
        elif split_mode == "tail":
            val_indices = set(range(n - n_val, n))
        else:
            raise ValueError(f"Unknown split mode: {split_mode}")

        dataset_train, dataset_val = [], []
        for idx, row in enumerate(dataset_rows):
            if idx in val_indices:
                val_paths.append(row["path"])
                dataset_val.append(row)
            else:
                train_paths.append(row["path"])
                dataset_train.append(row)
        split_summary["datasets"][dataset] = {
            "total": n,
            "train": len(dataset_train),
            "val": len(dataset_val),
            "train_frame_range": [
                dataset_train[0]["frame_number"],
                dataset_train[-1]["frame_number"],
            ],
            "val_frame_numbers": [row["frame_number"] for row in dataset_val],
        }

    return train_paths, val_paths, split_summary


class ForcepsRefineAdapter(nn.Module):
    """Tiny decoder that refines YOLO mask using frozen ReSurgSAM2 image features."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + 1, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, image_features: torch.Tensor, yolo_mask_512: torch.Tensor) -> torch.Tensor:
        low_mask = F.interpolate(
            yolo_mask_512,
            size=image_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual_low = self.net(torch.cat([image_features, low_mask], dim=1))
        residual = F.interpolate(
            residual_low,
            size=yolo_mask_512.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        yolo_logit = torch.logit(yolo_mask_512.clamp(1e-4, 1 - 1e-4))
        return yolo_logit + self.residual_scale.tanh() * residual


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


@torch.inference_mode()
def extract_low_features(model, images: torch.Tensor) -> torch.Tensor:
    backbone_out = model.forward_image(images)
    _, current_vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    h, w = feat_sizes[-1]
    return current_vision_feats[-1].permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    pred = logits.sigmoid() > 0.5
    gt = targets > 0.5
    inter = (pred & gt).float().sum(dim=(1, 2, 3))
    union = (pred | gt).float().sum(dim=(1, 2, 3))
    pred_area = pred.float().sum(dim=(1, 2, 3))
    gt_area = gt.float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union, torch.ones_like(union))
    dice = torch.where(pred_area + gt_area > 0, 2 * inter / (pred_area + gt_area), torch.ones_like(union))
    return {"iou": float(iou.mean().item()), "dice": float(dice.mean().item())}


def run_epoch(model, adapter, loader, optimizer, device: str, train: bool, has_features: bool) -> dict:
    adapter.train(train)
    total, count = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "yolo_iou": 0.0}, 0
    for batch in loader:
        images = batch["image"].to(device) if not has_features else None
        cached_features = batch["feature"].to(device) if has_features else None
        gt = batch["gt_mask"].to(device)
        yolo = batch["yolo_mask"].to(device)
        if has_features:
            features = cached_features
        else:
            with torch.inference_mode():
                features = extract_low_features(model, images)
        logits = adapter(features, yolo)
        loss = F.binary_cross_entropy_with_logits(logits, gt) + dice_loss_from_logits(logits, gt)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        refined = batch_metrics(logits.detach(), gt)
        yolo_logits = torch.logit(yolo.clamp(1e-4, 1 - 1e-4))
        yolo_metrics = batch_metrics(yolo_logits, gt)
        bs = gt.shape[0]
        total["loss"] += float(loss.item()) * bs
        total["iou"] += refined["iou"] * bs
        total["dice"] += refined["dice"] * bs
        total["yolo_iou"] += yolo_metrics["iou"] * bs
        count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_mask_cache_full")
    parser.add_argument("--feature-cache", action="store_true")
    parser.add_argument("--train-datasets", nargs="+", default=["exp1_cu_full"])
    parser.add_argument("--val-datasets", nargs="+", default=["exp2_cu_full"])
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="If >0, split each train dataset into train/val and ignore --val-datasets.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["interleaved", "random", "tail"],
        default="interleaved",
        help="How to hold out validation samples when --val-fraction is used.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="Code/ReSurgSAM2/yolo_mask_adapter/results/forceps_refine_adapter_stage_a")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    cache_dir = Path(args.cache_dir)
    split_summary = None
    if args.val_fraction > 0:
        datasets = set(args.train_datasets)
        rows = load_cache_index(cache_dir, datasets)
        train_paths, val_paths, split_summary = split_train_val(
            rows,
            val_fraction=args.val_fraction,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
        )
        train_ds = ForcepsCacheDataset(cache_dir, datasets, has_features=args.feature_cache, paths=train_paths)
        val_ds = ForcepsCacheDataset(cache_dir, datasets, has_features=args.feature_cache, paths=val_paths)
    else:
        train_ds = ForcepsCacheDataset(cache_dir, set(args.train_datasets), has_features=args.feature_cache)
        val_ds = ForcepsCacheDataset(cache_dir, set(args.val_datasets), has_features=args.feature_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = None if args.feature_cache else build_frozen_resurgsam2(device)
    feature_dim = 256 if args.feature_cache else model.hidden_dim
    adapter = ForcepsRefineAdapter(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_summary is not None:
        split_summary["train_count"] = len(train_ds)
        split_summary["val_count"] = len(val_ds)
        with (out_dir / "split.json").open("w", encoding="utf-8") as f:
            json.dump(split_summary, f, ensure_ascii=False, indent=2)
    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, adapter, train_loader, optimizer, device, train=True, has_features=args.feature_cache)
        val_metrics = run_epoch(model, adapter, val_loader, optimizer, device, train=False, has_features=args.feature_cache)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": adapter.state_dict(), "epoch": epoch, "val": val_metrics}, out_dir / "best.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    torch.save({"model": adapter.state_dict(), "epoch": args.epochs, "val": history[-1]["val"]}, out_dir / "last.pt")


if __name__ == "__main__":
    main()
