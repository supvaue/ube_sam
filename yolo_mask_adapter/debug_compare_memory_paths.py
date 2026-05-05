"""Compare direct pair-level memory construction with add_new_mask/preflight memory.

This diagnostic is not meant to make the video path identical to the direct path.
It measures where the two memory distributions differ so training can target the
actual video predictor loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from yolo_mask_adapter.eval_dense_tracking_protocol import extract_video_segment, mask_metrics
from yolo_mask_adapter.eval_refined_init_tracking import (
    build_predictor,
    build_refine_batch,
    load_adapter,
    refined_init_mask,
    resize_np_mask,
    validate_full_architecture,
)
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_shared_decoder_tracking import (
    extract_features,
    forward_sam_heads_with_prior,
    high_res_features_from,
    low_pix_feat,
)


def tensor_stats(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    x = tensor.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(tensor.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "norm": float(x.norm()),
    }


def cosine_flat(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None or tuple(a.shape) != tuple(b.shape):
        return None
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    return float(F.cosine_similarity(af[None], bf[None]).item())


def diff_stats(a: torch.Tensor | None, b: torch.Tensor | None) -> dict | None:
    if a is None or b is None or tuple(a.shape) != tuple(b.shape):
        return None
    d = (a.detach().float() - b.detach().float()).cpu()
    return {
        "l1_mean": float(d.abs().mean()),
        "l2": float(d.norm()),
        "max_abs": float(d.abs().max()),
        "cosine": cosine_flat(a, b),
    }


def direct_memory_from_refined_mask(model, image: torch.Tensor, mask_512: np.ndarray, args) -> dict:
    images = image.to(args.device)
    start_feats, _start_pos, start_sizes = extract_features(model, images)
    start_high_res = high_res_features_from(start_feats, start_sizes)
    start_pix = low_pix_feat(model, start_feats, start_sizes)
    mask_tensor = torch.from_numpy(mask_512.astype(np.float32))[None, None].to(args.device)
    _low_multi, high_res_masks, ious, low_res_masks, _best_high, obj_ptr, object_score_logits = forward_sam_heads_with_prior(
        model,
        backbone_features=start_pix,
        mask_inputs=mask_tensor,
        high_res_features=start_high_res,
        fusion_cls_tokens=None,
        multimask_output=False,
    )
    if args.direct_memory_mode == "binary_logits":
        high_for_mem = (torch.sigmoid(high_res_masks) >= args.refine_threshold).float() * 20.0 - 10.0
    elif args.direct_memory_mode == "input_binary_logits":
        high_for_mem = mask_tensor.float() * 20.0 - 10.0
    else:
        high_for_mem = high_res_masks
    if args.force_memory_object_score is not None:
        object_score_logits = torch.full(
            (high_for_mem.shape[0], 1),
            float(args.force_memory_object_score),
            device=args.device,
            dtype=high_for_mem.dtype,
        )
    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
        current_vision_feats=start_feats,
        feat_sizes=start_sizes,
        pred_masks_high_res=high_for_mem,
        object_score_logits=object_score_logits,
        is_mask_from_pts=True,
    )
    return {
        "pred_masks": low_res_masks.detach(),
        "pred_masks_high_res": high_res_masks.detach(),
        "memory_input_high_res": high_for_mem.detach(),
        "maskmem_features": maskmem_features.detach(),
        "maskmem_pos_enc": [x.detach() for x in maskmem_pos_enc],
        "obj_ptr": obj_ptr.detach(),
        "object_score_logits": object_score_logits.detach(),
        "ious": ious.detach(),
    }


def video_memory_from_add_new_mask(predictor, video_path: Path, start_frame: int, end_frame: int, mask_512: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory(prefix="memory_path_compare_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(video_path, frame_dir, start_frame, end_frame)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
        predictor.forward_text_emb = False
        predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id="target",
            mask=torch.from_numpy(mask_512.astype(np.float32)),
        )
        predictor.propagate_in_video_preflight(state)
        predictor.forward_text_emb = old_forward_text_emb
        out = state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
        return {
            key: value.detach() if isinstance(value, torch.Tensor) else [x.detach() for x in value]
            for key, value in out.items()
        }


def summarize_path(prefix: str, out: dict) -> dict:
    return {
        f"{prefix}_pred_masks": tensor_stats(out.get("pred_masks")),
        f"{prefix}_pred_masks_high_res": tensor_stats(out.get("pred_masks_high_res")),
        f"{prefix}_memory_input_high_res": tensor_stats(out.get("memory_input_high_res")),
        f"{prefix}_maskmem_features": tensor_stats(out.get("maskmem_features")),
        f"{prefix}_obj_ptr": tensor_stats(out.get("obj_ptr")),
        f"{prefix}_object_score_logits": tensor_stats(out.get("object_score_logits")),
        f"{prefix}_ious": tensor_stats(out.get("ious")),
        f"{prefix}_iou": tensor_stats(out.get("iou")),
        f"{prefix}_maskmem_pos_enc": [tensor_stats(x) for x in out.get("maskmem_pos_enc", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--cache-item", required=True)
    parser.add_argument("--gt-mask-item", default="")
    parser.add_argument("--category-id", type=int, default=12)
    parser.add_argument("--start-frame", type=int, default=22416)
    parser.add_argument("--end-frame", type=int, default=22417)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--config-file", default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml")
    parser.add_argument("--training-config-file", default="")
    parser.add_argument("--ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument("--direct-memory-mode", choices=["raw_logits", "binary_logits", "input_binary_logits"], default="raw_logits")
    parser.add_argument("--force-memory-object-score", type=float, default=10.0)
    parser.add_argument("--use-no-mem-attention", action="store_true")
    parser.add_argument("--use-mask-as-output", action="store_true")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--apply-long-term-memory", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    predictor = build_predictor(args).eval()
    if args.disable_obj_score_gating:
        predictor.pred_obj_scores = False
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    validate_full_architecture(adapter_checkpoint, model_checkpoint, require=True)
    from yolo_mask_adapter.train_mask_token_prompt_adapter import set_condition_fusion_mode

    set_condition_fusion_mode(predictor, adapter_checkpoint.get("condition_fusion_mode"))
    set_condition_fusion_mode(predictor, model_checkpoint.get("condition_fusion_mode"))
    adapter, adapter_info = load_adapter(adapter_checkpoint, predictor.hidden_dim, args.category_id, args.sentence_tokens, args.device)
    if "model" in adapter_checkpoint:
        predictor.load_state_dict(adapter_checkpoint["model"], strict=False)
    predictor.load_state_dict(model_checkpoint["model"], strict=False)
    scorer = MaskReliabilityScorer(target_class_id=args.category_id)

    cache_item = {"path": args.cache_item}
    data = np.load(args.cache_item, allow_pickle=False)
    gt_mask_np = data["gt_mask"].astype(np.float32) if "gt_mask" in data.files else data["yolo_mask"].astype(np.float32)
    batch = build_refine_batch(cache_item, gt_mask_np, args.category_id, args.image_size)
    mask_512, refine_debug = refined_init_mask(predictor, adapter, scorer, batch, args)

    direct = direct_memory_from_refined_mask(predictor, batch["image"], mask_512, args)
    video = video_memory_from_add_new_mask(predictor, Path(args.video), args.start_frame, args.end_frame, mask_512)

    gt_shape = tuple(gt_mask_np.shape)
    refined_video = resize_np_mask(mask_512, gt_shape)
    result = {
        "cache_item": args.cache_item,
        "start_frame": args.start_frame,
        "direct_memory_mode": args.direct_memory_mode,
        "force_memory_object_score": args.force_memory_object_score,
        "adapter": adapter_info,
        "refine_debug": refine_debug,
        "refined_vs_gt": mask_metrics(refined_video, gt_mask_np.astype(np.uint8)),
        **summarize_path("direct", direct),
        **summarize_path("video", video),
        "diff_pred_masks": diff_stats(direct.get("pred_masks"), video.get("pred_masks")),
        "diff_maskmem_features": diff_stats(direct.get("maskmem_features"), video.get("maskmem_features")),
        "diff_obj_ptr": diff_stats(direct.get("obj_ptr"), video.get("obj_ptr")),
        "diff_object_score_logits": diff_stats(direct.get("object_score_logits"), video.get("object_score_logits")),
        "diff_pos_enc": [
            diff_stats(a, b)
            for a, b in zip(direct.get("maskmem_pos_enc", []), video.get("maskmem_pos_enc", []))
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
