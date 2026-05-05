"""Probe whether MaskPriorEncoder tokens carry useful reliability features.

This script does not train the segmentation decoder. It freezes ReSurgSAM2 and
MaskTokenEncoder checkpoints, exports mask-prior tokens, then trains tiny probes
on top of those frozen features. The goal is to measure whether the encoder
learned a representation beyond raw YOLO mask geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from yolo_mask_adapter.mask_token_encoder import MaskTokenEncoder
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val
from yolo_mask_adapter.train_mask_token_prompt_adapter import (
    ForcepsPromptDataset,
    build_resurgsam2,
    detach_feature_lists,
    limit_paths_balanced,
)


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    a = mask_a > 0.5
    b = mask_b > 0.5
    inter = (a & b).float().flatten(1).sum(dim=1)
    union = (a | b).float().flatten(1).sum(dim=1)
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))


def boundary_ring(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    pos = labels > 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


def standardize(train_x: torch.Tensor, val_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def train_binary_probe(train_x, train_y, val_x, val_y, epochs: int, lr: float, weight_decay: float) -> dict:
    train_x, val_x = standardize(train_x, val_x)
    head = nn.Linear(train_x.shape[1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(train_x).squeeze(1), train_y.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = head(val_x).squeeze(1)
        probs = logits.sigmoid()
        pred = probs >= 0.5
        acc = (pred == (val_y > 0.5)).float().mean()
        return {
            "acc": float(acc.item()),
            "auc": binary_auc(probs, val_y),
            "positive_rate": float((val_y > 0.5).float().mean().item()),
        }


def train_multiclass_probe(train_x, train_y, val_x, val_y, classes: int, epochs: int, lr: float, weight_decay: float) -> dict:
    train_x, val_x = standardize(train_x, val_x)
    head = nn.Linear(train_x.shape[1], classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_x), train_y.long())
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = head(val_x)
        pred = logits.argmax(dim=1)
        acc = (pred == val_y).float().mean()
        per_class = {}
        for cls in range(classes):
            mask = val_y == cls
            per_class[str(cls)] = float((pred[mask] == cls).float().mean().item()) if mask.any() else float("nan")
        return {"acc": float(acc.item()), "per_class_acc": per_class}


def train_regression_probe(train_x, train_y, val_x, val_y, epochs: int, lr: float, weight_decay: float) -> dict:
    train_x, val_x = standardize(train_x, val_x)
    head = nn.Linear(train_x.shape[1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = head(train_x).squeeze(1)
        loss = F.smooth_l1_loss(pred, train_y.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(val_x).squeeze(1)
        mae = (pred - val_y).abs().mean()
        ss_res = ((pred - val_y) ** 2).sum()
        ss_tot = ((val_y - val_y.mean()) ** 2).sum().clamp_min(1e-8)
        r2 = 1.0 - ss_res / ss_tot
        corr = torch.corrcoef(torch.stack([pred, val_y.float()]))[0, 1]
        return {"mae": float(mae.item()), "r2": float(r2.item()), "pearson": float(corr.item())}


def run_probe_suite(features: dict[str, dict[str, torch.Tensor]], labels: dict[str, dict[str, torch.Tensor]], args) -> dict:
    tasks = {
        "bad_yolo_iou_lt_050": ("binary", 2),
        "good_yolo_iou_gte_070": ("binary", 2),
        "yolo_iou_bucket_3": ("multiclass", 3),
        "yolo_iou_regression": ("regression", 1),
        "boundary_error_regression": ("regression", 1),
        "area_error_regression": ("regression", 1),
    }
    out = {}
    for feature_name, split_feats in features.items():
        out[feature_name] = {}
        # Feature export runs under inference_mode; clone tensors before using
        # them in probe training so autograd can save activations normally.
        train_x = split_feats["train"].float().clone()
        val_x = split_feats["val"].float().clone()
        for task_name, (kind, classes) in tasks.items():
            train_y = labels["train"][task_name].clone()
            val_y = labels["val"][task_name].clone()
            if kind == "binary":
                out[feature_name][task_name] = train_binary_probe(
                    train_x, train_y, val_x, val_y, args.probe_epochs, args.probe_lr, args.probe_weight_decay
                )
            elif kind == "multiclass":
                out[feature_name][task_name] = train_multiclass_probe(
                    train_x, train_y, val_x, val_y, classes, args.probe_epochs, args.probe_lr, args.probe_weight_decay
                )
            else:
                out[feature_name][task_name] = train_regression_probe(
                    train_x, train_y, val_x, val_y, args.probe_epochs, args.probe_lr, args.probe_weight_decay
                )
    return out


def quality_bucket(iou: torch.Tensor) -> torch.Tensor:
    bucket = torch.zeros_like(iou, dtype=torch.long)
    bucket = torch.where(iou >= 0.50, torch.ones_like(bucket), bucket)
    bucket = torch.where(iou >= 0.70, torch.full_like(bucket, 2), bucket)
    return bucket


def adapter_from_checkpoint(path: Path, args) -> tuple[str, MaskTokenEncoder]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("adapter", checkpoint)
    depth = int(checkpoint.get("prior_mixer_depth", args.prior_mixer_depth))
    heads = int(checkpoint.get("prior_mixer_heads", args.prior_mixer_heads))
    sentence_tokens = int(checkpoint.get("sentence_tokens", args.sentence_tokens))
    adapter = MaskTokenEncoder(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        class_id=args.class_id,
        sentence_tokens=sentence_tokens,
        token_mixer_depth=depth,
        token_mixer_heads=heads,
    )
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    name = path.parent.name
    print(
        json.dumps(
            {
                "checkpoint": str(path),
                "name": name,
                "prior_mixer_depth": depth,
                "prior_mixer_heads": heads,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            ensure_ascii=False,
        )
    )
    adapter.eval()
    return name, adapter


@torch.inference_mode()
def export_features_for_split(model, adapters, scorer, loader, device: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    feature_chunks = {"geometry": [], "geometry_reliability": []}
    for name, _adapter in adapters:
        feature_chunks[f"{name}:sentence_mean"] = []
        feature_chunks[f"{name}:cls"] = []
        feature_chunks[f"{name}:sentence_flat"] = []
        feature_chunks[f"{name}:sentence_mean_cls_geometry"] = []
    label_chunks = {
        "bad_yolo_iou_lt_050": [],
        "good_yolo_iou_gte_070": [],
        "yolo_iou_bucket_3": [],
        "yolo_iou_regression": [],
        "boundary_error_regression": [],
        "area_error_regression": [],
    }

    for batch in loader:
        images = batch["image"].to(device)
        gt = batch["gt_mask"].to(device)
        yolo = batch["yolo_mask"].to(device)
        yolo_conf = batch["yolo_conf"].to(device)
        class_ids = batch["class_id"].to(device)

        backbone_out = model.forward_image(images)
        _, current_vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        current_vision_feats = detach_feature_lists(current_vision_feats)
        low_feat = current_vision_feats[-1]
        h, w = feat_sizes[-1]
        image_features = low_feat.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
        reliability, geometry = scorer.score(yolo, yolo_conf, class_ids)

        yolo_iou = mask_iou(yolo, gt)
        yolo_boundary_iou = mask_iou(boundary_ring(yolo), boundary_ring(gt))
        yolo_area = (yolo > 0.5).float().flatten(1).mean(dim=1)
        gt_area = (gt > 0.5).float().flatten(1).mean(dim=1)
        area_error = (yolo_area - gt_area).abs() / torch.maximum(gt_area, torch.full_like(gt_area, 1e-6))

        feature_chunks["geometry"].append(geometry.cpu())
        feature_chunks["geometry_reliability"].append(torch.cat([geometry, reliability[:, None]], dim=1).cpu())
        label_chunks["bad_yolo_iou_lt_050"].append((yolo_iou < 0.50).long().cpu())
        label_chunks["good_yolo_iou_gte_070"].append((yolo_iou >= 0.70).long().cpu())
        label_chunks["yolo_iou_bucket_3"].append(quality_bucket(yolo_iou).cpu())
        label_chunks["yolo_iou_regression"].append(yolo_iou.cpu())
        label_chunks["boundary_error_regression"].append((1.0 - yolo_boundary_iou).cpu())
        label_chunks["area_error_regression"].append(area_error.clamp(0.0, 10.0).cpu())

        for name, adapter in adapters:
            adapter = adapter.to(device)
            out = adapter(image_features=image_features, yolo_mask=yolo, geometry=geometry, class_ids=class_ids)
            sentence = out.mask_emb_sentence.detach().cpu()
            cls = out.mask_emb_cls.squeeze(1).detach().cpu()
            sentence_mean = sentence.mean(dim=1)
            feature_chunks[f"{name}:sentence_mean"].append(sentence_mean)
            feature_chunks[f"{name}:cls"].append(cls)
            feature_chunks[f"{name}:sentence_flat"].append(sentence.flatten(1))
            feature_chunks[f"{name}:sentence_mean_cls_geometry"].append(
                torch.cat([sentence_mean, cls, geometry.detach().cpu()], dim=1)
            )

    features = {name: torch.cat(chunks, dim=0) for name, chunks in feature_chunks.items()}
    labels = {name: torch.cat(chunks, dim=0) for name, chunks in label_chunks.items()}
    return features, labels


def summarize_labels(labels: dict[str, dict[str, torch.Tensor]]) -> dict:
    out = {}
    for split, split_labels in labels.items():
        iou = split_labels["yolo_iou_regression"].float()
        buckets = split_labels["yolo_iou_bucket_3"].long()
        out[split] = {
            "count": int(iou.numel()),
            "mean_yolo_iou": float(iou.mean().item()),
            "bad_lt_050": int((buckets == 0).sum().item()),
            "mid_050_070": int((buckets == 1).sum().item()),
            "good_gte_070": int((buckets == 2).sum().item()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/ligamentum_flavum_mask_cache_balanced_600")
    parser.add_argument("--datasets", default="exp1_cu_full,exp2_cu_full")
    parser.add_argument("--max-items", type=int, default=600)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["interleaved", "random", "tail"], default="interleaved")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-cache-root", default="")
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--prior-mixer-depth", type=int, default=0)
    parser.add_argument("--prior-mixer-heads", type=int, default=4)
    parser.add_argument("--base-ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    cache_dir = Path(args.cache_dir)
    rows = load_cache_index(cache_dir, datasets)
    paths = [row["path"] for row in rows]
    paths = limit_paths_balanced(paths, args.max_items)
    rows = load_cache_index(cache_dir, datasets)
    rows = [row for row in rows if row["path"] in set(paths)]
    train_paths, val_paths, split_summary = split_train_val(rows, args.val_fraction, args.split_mode, args.split_seed)

    image_cache_root = Path(args.image_cache_root) if args.image_cache_root else None
    train_ds = ForcepsPromptDataset(train_paths, image_size=args.image_size, class_id=args.class_id, image_cache_root=image_cache_root)
    val_ds = ForcepsPromptDataset(val_paths, image_size=args.image_size, class_id=args.class_id, image_cache_root=image_cache_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(json.dumps({"stage": "build_resurgsam2", "ckpt": args.base_ckpt_path, "device": device}, ensure_ascii=False))
    model = build_resurgsam2(device, ckpt_path=args.base_ckpt_path)
    for param in model.parameters():
        param.requires_grad_(False)
    scorer = MaskReliabilityScorer(target_class_id=args.class_id)
    adapters = [adapter_from_checkpoint(Path(path), args) for path in args.checkpoints]

    print(json.dumps({"stage": "export_train_features", "count": len(train_ds)}, ensure_ascii=False))
    train_features, train_labels = export_features_for_split(model, adapters, scorer, train_loader, device)
    print(json.dumps({"stage": "export_val_features", "count": len(val_ds)}, ensure_ascii=False))
    val_features, val_labels = export_features_for_split(model, adapters, scorer, val_loader, device)

    features = {
        name: {"train": train_features[name], "val": val_features[name]}
        for name in sorted(train_features)
    }
    labels = {"train": train_labels, "val": val_labels}
    probes = run_probe_suite(features, labels, args)
    result = {
        "config": vars(args),
        "split": split_summary,
        "label_summary": summarize_labels(labels),
        "feature_dims": {
            name: {"train": list(value["train"].shape), "val": list(value["val"].shape)}
            for name, value in features.items()
        },
        "probes": probes,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", "output_json": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
