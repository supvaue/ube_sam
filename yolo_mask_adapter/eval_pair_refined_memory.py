"""Evaluate pair-level tracking when start memory comes from refined no-memory logits.

This is a C-style diagnostic: it uses the same pair-level memory path as
train_shared_decoder_tracking.py and avoids the video predictor state loop. The
goal is to separate refined-key memory quality from add_new_mask/preflight/
propagate_in_video behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val
from yolo_mask_adapter.train_mask_token_prompt_adapter import batch_metrics, limit_paths_balanced
from yolo_mask_adapter.train_shared_decoder_tracking import (
    SparseTrackingPairDataset,
    build_adapter_from_checkpoint,
    build_model,
    extract_features,
    forward_sam_heads_with_prior,
    high_res_features_from,
    low_pix_feat,
    prior_conditioned_outputs,
)


def detach_output_dict(out: dict) -> dict:
    detached = {}
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            detached[key] = value.detach()
        elif isinstance(value, list):
            detached[key] = [item.detach() if isinstance(item, torch.Tensor) else item for item in value]
        else:
            detached[key] = value
    return detached


def make_cond_memory_from_logits(
    model,
    start_feats,
    start_sizes,
    pred_masks_high_res: torch.Tensor,
    low_res_masks: torch.Tensor,
    obj_ptr: torch.Tensor,
    object_score_logits: torch.Tensor,
    ious: torch.Tensor,
    force_object_score: float | None,
) -> dict:
    if force_object_score is not None:
        object_score_logits = torch.full(
            (pred_masks_high_res.shape[0], 1),
            float(force_object_score),
            device=pred_masks_high_res.device,
            dtype=pred_masks_high_res.dtype,
        )
    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
        current_vision_feats=start_feats,
        feat_sizes=start_sizes,
        pred_masks_high_res=pred_masks_high_res,
        object_score_logits=object_score_logits,
        is_mask_from_pts=True,
    )
    return detach_output_dict(
        {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": low_res_masks,
            "pred_masks_high_res": pred_masks_high_res,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "ious": ious,
        }
    )


@torch.no_grad()
def forward_pair_refined_memory(model, adapter, scorer, batch: dict, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_images = batch["start_image"].to(args.device)
    target_images = batch["target_image"].to(args.device)
    start_yolo = batch["start_yolo"].to(args.device)
    target_yolo = batch["target_yolo"].to(args.device)
    class_ids = torch.full((start_images.shape[0],), args.target_class_id, device=args.device, dtype=torch.long)
    start_conf = torch.ones(start_images.shape[0], device=args.device)

    start_feats, start_pos, start_sizes = extract_features(model, start_images)
    target_feats, target_pos, target_sizes = extract_features(model, target_images)
    start_high_res = high_res_features_from(start_feats, start_sizes)
    target_high_res = high_res_features_from(target_feats, target_sizes)

    key_outputs, _prior_sparse_tokens = prior_conditioned_outputs(
        model=model,
        adapter=adapter,
        scorer=scorer,
        images=start_images,
        yolo=start_yolo,
        yolo_conf=start_conf,
        class_ids=class_ids,
        feats=start_feats,
        pos=start_pos,
        sizes=start_sizes,
        high_res=start_high_res,
    )
    key_low_multi, key_high_res, key_ious, key_low, _key_best_high, key_obj, key_score = key_outputs
    if args.refined_memory_mode == "binary_logits":
        key_high_for_mem = (torch.sigmoid(key_high_res) >= args.refine_threshold).float() * 20.0 - 10.0
    else:
        key_high_for_mem = key_high_res
    cond_out = make_cond_memory_from_logits(
        model=model,
        start_feats=start_feats,
        start_sizes=start_sizes,
        pred_masks_high_res=key_high_for_mem,
        low_res_masks=key_low,
        obj_ptr=key_obj,
        object_score_logits=key_score,
        ious=key_ious,
        force_object_score=args.force_memory_object_score,
    )

    frame_idx = int(batch["gap"].max().item()) if args.use_real_gap else 1
    output_dict = {
        "cond_frame_outputs": {0: cond_out},
        "non_cond_frame_outputs": {},
        "long_mem_candidate": [],
        "long_mem": [],
    }
    old_disable_non_cond = getattr(model, "disable_non_cond_memory", False)
    model.disable_non_cond_memory = args.disable_non_cond_memory
    try:
        pix_feat_with_mem = model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=target_feats[-1:],
            current_vision_pos_embeds=target_pos[-1:],
            feat_sizes=target_sizes[-1:],
            output_dict=output_dict,
            num_frames=frame_idx + 1,
            track_in_reverse=False,
        )
    finally:
        model.disable_non_cond_memory = old_disable_non_cond

    old_pred_obj_scores = model.pred_obj_scores
    if args.disable_obj_score_gating:
        model.pred_obj_scores = False
    try:
        _track_low_multi, track_high_res, _track_ious, _track_low, _track_high, _track_obj, _track_score = (
            forward_sam_heads_with_prior(
                model,
                backbone_features=pix_feat_with_mem,
                mask_inputs=None,
                high_res_features=target_high_res,
                fusion_cls_tokens=None,
                multimask_output=False,
            )
        )
    finally:
        model.pred_obj_scores = old_pred_obj_scores
    return track_high_res, key_high_res, target_yolo


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["track_iou", "key_iou", "yolo_iou", "track_yolo_iou"]
    out = {"count": len(rows)}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        if vals:
            out[f"mean_{key}"] = float(np.mean(vals))
            out[f"min_{key}"] = float(np.min(vals))
            out[f"p50_{key}"] = float(np.median(vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--train-datasets", nargs="+", required=True)
    parser.add_argument("--image-cache-root", default="")
    parser.add_argument("--config-file", default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml")
    parser.add_argument("--base-ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--mask-token-checkpoint", required=True)
    parser.add_argument("--condition-fusion-mode", choices=["twoway", "film_mlp", "hybrid_film_mlp"], default="film_mlp")
    parser.add_argument("--target-class-id", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["random", "tail"], default="tail")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--eval-split", choices=["val", "all"], default="val")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--refined-memory-mode", choices=["raw_logits", "binary_logits"], default="raw_logits")
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument("--force-memory-object-score", type=float, default=10.0)
    parser.add_argument("--disable-non-cond-memory", action="store_true")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--apply-long-term-memory", action="store_true")
    parser.add_argument("--use-real-gap", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = load_cache_index(Path(args.cache_dir), set(args.train_datasets))
    _train_paths, val_paths, split_summary = split_train_val(rows, args.val_fraction, args.split_mode, args.split_seed)
    if args.eval_split == "all":
        val_paths = [Path(row["path"]) for row in rows]
        split_summary = {**split_summary, "eval_split": "all"}
    else:
        split_summary = {**split_summary, "eval_split": "val"}
    if args.max_items and args.eval_split != "all":
        val_paths = limit_paths_balanced(val_paths, args.max_items)
    image_cache_root = Path(args.image_cache_root) if args.image_cache_root else None
    try:
        ds = SparseTrackingPairDataset(val_paths, args.image_size, image_cache_root=image_cache_root, max_gap=args.max_gap)
    except FileNotFoundError:
        if args.eval_split != "val":
            raise
        val_paths = [Path(row["path"]) for row in rows]
        split_summary = {**split_summary, "eval_split": "all_fallback_after_empty_val"}
        if args.max_items:
            val_paths = limit_paths_balanced(val_paths, args.max_items)
        ds = SparseTrackingPairDataset(val_paths, args.image_size, image_cache_root=image_cache_root, max_gap=args.max_gap)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = build_model(args).eval()
    adapter = build_adapter_from_checkpoint(args)
    scorer = MaskReliabilityScorer(target_class_id=args.target_class_id)
    if args.disable_obj_score_gating:
        model.pred_obj_scores = False

    eval_rows = []
    for index, batch in enumerate(loader):
        if index % 10 == 0:
            print(json.dumps({"stage": "eval_pair", "index": index, "total": len(ds)}, ensure_ascii=False), flush=True)
        target_gt = batch["target_gt"].to(args.device)
        track_logits, key_logits, target_yolo = forward_pair_refined_memory(model, adapter, scorer, batch, args)
        if track_logits.shape[-2:] != target_gt.shape[-2:]:
            track_logits = F.interpolate(track_logits, size=target_gt.shape[-2:], mode="bilinear", align_corners=False)
        if key_logits.shape[-2:] != target_gt.shape[-2:]:
            key_logits = F.interpolate(key_logits, size=target_gt.shape[-2:], mode="bilinear", align_corners=False)
        track_metrics = batch_metrics(track_logits, target_gt)
        key_metrics = batch_metrics(key_logits, target_gt)
        yolo_metrics = batch_metrics(torch.logit(target_yolo.to(args.device).clamp(1e-4, 1 - 1e-4)), target_gt)
        track_yolo_metrics = batch_metrics(track_logits, target_yolo.to(args.device))
        eval_rows.append(
            {
                "index": index,
                "dataset": batch["dataset"][0],
                "start_file": batch["start_file"][0],
                "target_file": batch["target_file"][0],
                "gap": int(batch["gap"][0]),
                "track_iou": track_metrics["iou"],
                "key_iou": key_metrics["iou"],
                "yolo_iou": yolo_metrics["iou"],
                "track_yolo_iou": track_yolo_metrics["iou"],
            }
        )

    result = {
        "mode": "pair_refined_memory",
        "checkpoint": args.resume_checkpoint,
        "mask_token_checkpoint": args.mask_token_checkpoint,
        "refined_memory_mode": args.refined_memory_mode,
        "force_memory_object_score": args.force_memory_object_score,
        "disable_non_cond_memory": bool(args.disable_non_cond_memory),
        "disable_obj_score_gating": bool(args.disable_obj_score_gating),
        "split": split_summary,
        "summary": summarize_rows(eval_rows),
        "rows": eval_rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
