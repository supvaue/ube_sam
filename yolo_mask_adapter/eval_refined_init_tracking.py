"""Evaluate video tracking after a trained no-memory refined key prompt.

This intentionally bypasses the old text/referring `yolo_token` video entry.
The first frame is decoded with the same path used by
train_mask_token_prompt_adapter.py:

  MaskPriorEncoder/CSTMamba + dense YOLO mask prompt + PromptEncoder
  -> shared decoder -> refined key mask

The refined mask is then inserted as a normal SAM2 conditioning mask and the
video is propagated with memory attention.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from yolo_mask_adapter.eval_dense_tracking_protocol import (
    cifs_mask_iou,
    clear_outputs_after_frame,
    cuda_sync,
    extract_video_segment,
    load_gt_by_frame,
    load_mask_cache_by_frame,
    mask_metrics,
    summarize,
)
from yolo_mask_adapter.mask_token_encoder import MaskTokenEncoder
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_mask_token_prompt_adapter import (
    IMG_MEAN,
    IMG_STD,
    forward_prompt_model,
    set_condition_fusion_mode,
)


def build_predictor(args):
    overrides = [
        "++scratch.use_sp_bimamba=true",
        "++scratch.use_dwconv=true",
        f"++model.use_mask_input_as_output_without_sam={'true' if args.use_mask_as_output else 'false'}",
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device=args.device,
        strict_loading=False,
        apply_long_term_memory=args.apply_long_term_memory,
        training_config_file=args.training_config_file,
        hydra_overrides_extra=overrides,
    )
    return predictor


def load_adapter(adapter_checkpoint: dict, hidden_dim: int, class_id: int, sentence_tokens: int, device: str):
    prior_mixer_depth = int(adapter_checkpoint.get("prior_mixer_depth", 0))
    prior_mixer_heads = int(adapter_checkpoint.get("prior_mixer_heads", 4))
    adapter = MaskTokenEncoder(
        embed_dim=hidden_dim,
        class_id=class_id,
        sentence_tokens=sentence_tokens,
        token_mixer_depth=prior_mixer_depth,
        token_mixer_heads=prior_mixer_heads,
    ).to(device)
    missing, unexpected = adapter.load_state_dict(adapter_checkpoint["adapter"], strict=False)
    adapter.eval()
    return adapter, {
        "prior_mixer_depth": prior_mixer_depth,
        "prior_mixer_heads": prior_mixer_heads,
        "adapter_missing_keys": len(missing),
        "adapter_unexpected_keys": len(unexpected),
    }


def validate_full_architecture(adapter_checkpoint: dict, model_checkpoint: dict, require: bool) -> dict:
    """Preflight guard for the current full refined-key + memory-tracking design."""
    model_state = model_checkpoint.get("model", {})
    train_parts = set(model_checkpoint.get("train_parts", []))
    required_train_parts = {"cstmamba", "prompt_encoder", "mask_decoder", "memory_attention", "memory_encoder"}
    probes = {
        "adapter": "adapter" in adapter_checkpoint,
        "model": "model" in model_checkpoint,
        "film_mlp_weights": any(".image_to_token_film." in key for key in model_state),
        "mask_decoder_weights": any(key.startswith("sam_mask_decoder.") for key in model_state),
        "prompt_encoder_weights": any(key.startswith("sam_prompt_encoder.") for key in model_state),
        "memory_attention_weights": any(key.startswith("memory_attention.") for key in model_state),
        "memory_encoder_weights": any(key.startswith("memory_encoder.") for key in model_state),
        "cstmamba_weights": any(key.startswith("cross_modal_fusion.") for key in model_state),
        "condition_fusion_match": adapter_checkpoint.get("condition_fusion_mode")
        == model_checkpoint.get("condition_fusion_mode"),
        "tracking_prompt_source_none": model_checkpoint.get("tracking_prompt_source") == "none",
        "required_train_parts_present": required_train_parts.issubset(train_parts),
    }
    failures = [name for name, ok in probes.items() if not ok]
    info = {
        "required": bool(require),
        "ok": not failures,
        "failures": failures,
        "adapter_condition_fusion_mode": adapter_checkpoint.get("condition_fusion_mode"),
        "model_condition_fusion_mode": model_checkpoint.get("condition_fusion_mode"),
        "tracking_prompt_source": model_checkpoint.get("tracking_prompt_source"),
        "train_parts": sorted(train_parts),
    }
    if require and failures:
        raise RuntimeError(f"Full-architecture preflight failed: {json.dumps(info, ensure_ascii=False)}")
    print(json.dumps({"full_architecture_preflight": info}, ensure_ascii=False), flush=True)
    return info


def load_image_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    if not image_path.exists() and not image_path.is_absolute():
        image_path = WORKSPACE_ROOT / image_path
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    return (image_t - IMG_MEAN) / IMG_STD


def load_flexible_mask_cache_by_frame(cache_dir: str, dataset_name: str) -> dict[int, dict]:
    """Load both sparse GT-style caches and autolabel track-clip caches by frame number."""
    if not cache_dir:
        return {}
    out = {}
    root = Path(cache_dir)
    for path in root.glob("*.npz"):
        data = np.load(path, allow_pickle=False)
        if dataset_name and "dataset" in data.files and str(data["dataset"]) != dataset_name:
            continue
        if "frame_number" not in data.files:
            continue
        frame_idx = int(data["frame_number"])
        if "yolo_mask" not in data.files:
            continue
        out[frame_idx] = {
            "path": str(path),
            "yolo_mask": data["yolo_mask"].astype(np.uint8),
            "yolo_conf": float(data["yolo_conf"]) if "yolo_conf" in data.files else 1.0,
        }
    return out


def build_refine_batch(cache_item: dict, gt_mask_np: np.ndarray | None, class_id: int, image_size: int) -> dict:
    data = np.load(cache_item["path"], allow_pickle=False)
    image = load_image_tensor(Path(str(data["image_path"])), image_size)
    yolo = torch.from_numpy(data["yolo_mask"].astype(np.float32))[None, None]
    if gt_mask_np is None:
        if "gt_mask" in data.files:
            gt_mask_np = data["gt_mask"].astype(np.float32)
        else:
            gt_mask_np = data["yolo_mask"].astype(np.float32)
    gt = torch.from_numpy(gt_mask_np.astype(np.float32))[None, None]
    yolo = F.interpolate(yolo, size=(image_size, image_size), mode="nearest")[0]
    gt = F.interpolate(gt, size=(image_size, image_size), mode="nearest")[0]
    return {
        "image": image.unsqueeze(0),
        "gt_mask": gt.unsqueeze(0),
        "yolo_mask": yolo.unsqueeze(0),
        "yolo_conf": torch.tensor([float(data["yolo_conf"])], dtype=torch.float32),
        "class_id": torch.tensor([class_id], dtype=torch.long),
    }


@torch.inference_mode()
def refined_init_mask(model, adapter, scorer, batch: dict, args) -> tuple[np.ndarray, dict]:
    result = forward_prompt_model(
        model=model,
        adapter=adapter,
        scorer=scorer,
        batch=batch,
        device=args.device,
        prompt_mode="dense_condition",
        disable_obj_score_gating=args.disable_obj_score_gating,
        use_no_mem_attention=args.use_no_mem_attention,
        return_details=True,
    )
    logits = result.logits if hasattr(result, "logits") else result
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    mask = (prob >= args.refine_threshold).astype(np.uint8)
    pred_iou = None
    obj_score = None
    if hasattr(result, "ious") and result.ious is not None:
        pred_iou = float(result.ious.detach().float().flatten()[0].cpu())
    if hasattr(result, "object_score_logits") and result.object_score_logits is not None:
        obj_score = float(result.object_score_logits.detach().float().sigmoid().flatten()[0].cpu())
    return mask, {
        "threshold": args.refine_threshold,
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
        "prob_mean": float(prob.mean()),
        "mask_area": float(mask.mean()),
        "pred_iou": pred_iou,
        "object_score": obj_score,
    }


def resize_np_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=shape, mode="nearest")[0, 0].numpy()
    return (resized > 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--category-id", type=int, default=12)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-video-frames", type=int, default=400)
    parser.add_argument("--max-gt-evals", type=int, default=10)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-dataset", default="exp1_cu_full")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--config-file", default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml")
    parser.add_argument("--training-config-file", default="")
    parser.add_argument("--ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument("--enable-cifs-correction", action="store_true")
    parser.add_argument("--cifs-trigger-mode", choices=["disagreement", "official_window", "two_stage_window"], default="disagreement")
    parser.add_argument("--cifs-cache-dir", default="")
    parser.add_argument("--cifs-cache-dataset", default="")
    parser.add_argument("--cifs-window-size", type=int, default=5)
    parser.add_argument("--cifs-iou-threshold", type=float, default=0.7)
    parser.add_argument("--cifs-obj-score-threshold", type=float, default=-1.0)
    parser.add_argument("--cifs-min-track-yolo-iou", type=float, default=0.55)
    parser.add_argument("--cifs-correction-interval", type=int, default=0)
    parser.add_argument("--require-full-architecture", action="store_true")
    parser.add_argument(
        "--score-corrected-frame",
        action="store_true",
        help="When CIFS corrects at an eval frame, score the corrected refined mask for that frame.",
    )
    parser.add_argument("--use-no-mem-attention", action="store_true")
    parser.add_argument("--use-mask-as-output", action="store_true")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--apply-long-term-memory", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    gt_by_frame = load_gt_by_frame(Path(args.annotations), args.category_id)
    gt_frames = sorted(frame for frame in gt_by_frame if frame >= args.start_frame)
    if len(gt_frames) < 2:
        raise RuntimeError("Need at least two GT frames")
    start_frame = gt_frames[0]
    eval_gt_frames = gt_frames[: args.max_gt_evals]
    requested_end = start_frame + args.max_video_frames - 1
    end_frame = min(requested_end, eval_gt_frames[-1])
    eval_gt_frames = [frame for frame in eval_gt_frames if frame <= end_frame]

    cache_by_frame = load_mask_cache_by_frame(args.cache_dir, args.cache_dataset)
    cifs_cache_by_frame = (
        load_flexible_mask_cache_by_frame(
            args.cifs_cache_dir,
            args.cifs_cache_dataset or args.cache_dataset,
        )
        if args.cifs_cache_dir
        else cache_by_frame
    )
    if start_frame not in cache_by_frame:
        raise RuntimeError(f"No cache item for start frame {start_frame}")
    if args.enable_cifs_correction and args.cifs_trigger_mode == "official_window" and not cifs_cache_by_frame:
        raise RuntimeError("Official-window CIFS requires per-frame cache; provide --cifs-cache-dir")

    predictor = build_predictor(args)
    predictor.eval()
    if args.disable_obj_score_gating:
        predictor.pred_obj_scores = False

    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    full_architecture_info = validate_full_architecture(
        adapter_checkpoint,
        model_checkpoint,
        require=args.require_full_architecture,
    )
    set_condition_fusion_mode(predictor, adapter_checkpoint.get("condition_fusion_mode"))
    set_condition_fusion_mode(predictor, model_checkpoint.get("condition_fusion_mode"))
    adapter, adapter_info = load_adapter(adapter_checkpoint, predictor.hidden_dim, args.category_id, args.sentence_tokens, args.device)
    if "model" in adapter_checkpoint:
        predictor.load_state_dict(adapter_checkpoint["model"], strict=False)
    missing, unexpected = predictor.load_state_dict(model_checkpoint["model"], strict=False)
    model_info = {
        "checkpoint": args.model_checkpoint,
        "epoch": model_checkpoint.get("epoch"),
        "val": model_checkpoint.get("val"),
        "condition_fusion_mode": model_checkpoint.get("condition_fusion_mode"),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    scorer = MaskReliabilityScorer(target_class_id=args.category_id)

    batch = build_refine_batch(cache_by_frame[start_frame], gt_by_frame[start_frame]["mask"], args.category_id, args.image_size)
    init_mask_512, refine_debug = refined_init_mask(predictor, adapter, scorer, batch, args)
    init_mask_video = resize_np_mask(init_mask_512, gt_by_frame[start_frame]["mask"].shape)
    init_refined_metrics = mask_metrics(init_mask_video, gt_by_frame[start_frame]["mask"])
    yolo_init_metrics = mask_metrics(cache_by_frame[start_frame]["yolo_mask"], gt_by_frame[start_frame]["mask"])

    with tempfile.TemporaryDirectory(prefix="resurg_refined_track_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(Path(args.video), frame_dir, start_frame, end_frame)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
        predictor.forward_text_emb = False
        init_ret = predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id="target",
            mask=torch.from_numpy(init_mask_512.astype(np.float32)),
        )
        init_pred_video = (init_ret[2][0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
        init_added_metrics = mask_metrics(init_pred_video, gt_by_frame[start_frame]["mask"])
        predictor.propagate_in_video_preflight(state)
        iterator = predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=end_frame - start_frame + 1,
        )
        eval_lookup = {frame - start_frame: frame for frame in eval_gt_frames}
        tracking_times = []
        rows = []
        corrections = [
            {
                "frame_idx": start_frame,
                "local_idx": 0,
                "reason": "initial_refined_key",
                "init_refined_iou": init_refined_metrics["iou"],
                "init_yolo_iou": yolo_init_metrics["iou"],
            }
        ]
        eval_index = 0
        official_window = []
        last_official_cond_local = 0
        cifs_fallback_active = False
        cifs_fallback_reason = None
        cifs_fallback_start_frame = None
        cifs_diagnostics = []
        cifs_trigger_events = []
        while True:
            cuda_sync(args.device)
            step_start = time.perf_counter()
            try:
                local_idx, _obj_ids, masks = next(iterator)
            except StopIteration:
                break
            cuda_sync(args.device)
            tracking_times.append(time.perf_counter() - step_start)
            local_idx = int(local_idx)
            actual_step_frame = start_frame + local_idx
            pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
            official_trigger = False
            official_reason = None
            official_selected = None
            cifs_item = cifs_cache_by_frame.get(actual_step_frame)
            step_track_yolo_iou = cifs_mask_iou(pred, cifs_item["yolo_mask"]) if cifs_item is not None else None
            step_failure_trigger = False
            step_failure_reason = "stable"
            if (
                args.enable_cifs_correction
                and args.cifs_trigger_mode == "two_stage_window"
                and local_idx != 0
            ):
                if step_track_yolo_iou is not None and step_track_yolo_iou < args.cifs_min_track_yolo_iou:
                    step_failure_trigger = True
                    step_failure_reason = "stage2_track_yolo_disagreement"
                if step_failure_trigger and not cifs_fallback_active:
                    cifs_fallback_active = True
                    cifs_fallback_reason = step_failure_reason
                    cifs_fallback_start_frame = actual_step_frame
                    official_window = []
                    cifs_trigger_events.append(
                        {
                            "frame_idx": actual_step_frame,
                            "local_idx": local_idx,
                            "reason": step_failure_reason,
                            "track_yolo_iou": step_track_yolo_iou,
                        }
                    )

            if (
                args.enable_cifs_correction
                and args.cifs_trigger_mode in {"official_window", "two_stage_window"}
                and local_idx != 0
            ):
                should_search_stage1 = args.cifs_trigger_mode == "official_window" or cifs_fallback_active
                if cifs_item is None:
                    official_window = []
                elif should_search_stage1:
                    if official_window and local_idx - int(official_window[-1]["local_idx"]) != 1:
                        official_window = []
                    candidate_batch = build_refine_batch(cifs_item, None, args.category_id, args.image_size)
                    candidate_mask_512, candidate_debug = refined_init_mask(
                        predictor,
                        adapter,
                        scorer,
                        candidate_batch,
                        args,
                    )
                    pred_iou = candidate_debug.get("pred_iou")
                    obj_score = candidate_debug.get("object_score")
                    credible = pred_iou is not None and pred_iou > args.cifs_iou_threshold
                    if args.cifs_obj_score_threshold >= 0:
                        credible = credible and obj_score is not None and obj_score > args.cifs_obj_score_threshold
                    cifs_diagnostics.append(
                        {
                            "frame_idx": actual_step_frame,
                            "local_idx": local_idx,
                            "mode": args.cifs_trigger_mode,
                            "fallback_active": bool(cifs_fallback_active),
                            "fallback_start_frame": cifs_fallback_start_frame,
                            "fallback_reason": cifs_fallback_reason,
                            "track_yolo_iou": step_track_yolo_iou,
                            "pred_iou": pred_iou,
                            "object_score": obj_score,
                            "credible": bool(credible),
                            "refined_mask_area": candidate_debug.get("mask_area"),
                        }
                    )
                    if credible:
                        official_window.append(
                            {
                                "frame_idx": actual_step_frame,
                                "local_idx": local_idx,
                                "pred_iou": float(pred_iou),
                                "object_score": obj_score,
                                "mask_512": candidate_mask_512,
                                "debug": candidate_debug,
                            }
                        )
                        official_window = official_window[-args.cifs_window_size :]
                    else:
                        official_window = []

                    if len(official_window) == args.cifs_window_size:
                        official_selected = max(official_window, key=lambda item: item["pred_iou"])
                        if int(official_selected["local_idx"]) > last_official_cond_local:
                            official_trigger = True
                            official_reason = "official_cifs_window"
                            selected_local_idx = int(official_selected["local_idx"])
                            old_add_all = getattr(predictor, "add_all_frames_to_correct_as_cond", False)
                            try:
                                predictor.add_all_frames_to_correct_as_cond = True
                                predictor.add_new_mask(
                                    state,
                                    frame_idx=selected_local_idx,
                                    obj_id="target",
                                    mask=torch.from_numpy(official_selected["mask_512"].astype(np.float32)),
                                )
                            finally:
                                predictor.add_all_frames_to_correct_as_cond = old_add_all
                            clear_outputs_after_frame(state, selected_local_idx)
                            last_official_cond_local = selected_local_idx
                            corrections.append(
                                {
                                    "frame_idx": int(official_selected["frame_idx"]),
                                    "local_idx": selected_local_idx,
                                    "reason": official_reason,
                                    "fallback_start_frame": cifs_fallback_start_frame,
                                    "fallback_reason": cifs_fallback_reason,
                                    "selected_pred_iou": float(official_selected["pred_iou"]),
                                    "selected_object_score": official_selected["object_score"],
                                    "window": [
                                        {
                                            "frame_idx": int(item["frame_idx"]),
                                            "local_idx": int(item["local_idx"]),
                                            "pred_iou": float(item["pred_iou"]),
                                            "object_score": item["object_score"],
                                        }
                                        for item in official_window
                                    ],
                                    "refine_debug": official_selected["debug"],
                                }
                            )
                            cifs_fallback_active = False
                            cifs_fallback_reason = None
                            cifs_fallback_start_frame = None
                        official_window = []

            if local_idx not in eval_lookup:
                continue
            actual_frame = eval_lookup[local_idx]
            gt_mask = gt_by_frame[actual_frame]["mask"]
            metrics = mask_metrics(pred, gt_mask)
            yolo_item = cache_by_frame.get(actual_frame)
            yolo_iou = cifs_mask_iou(yolo_item["yolo_mask"], gt_mask) if yolo_item is not None else None
            track_yolo_iou = cifs_mask_iou(pred, yolo_item["yolo_mask"]) if yolo_item is not None else None

            trigger = official_trigger
            reason = official_reason or ("stable" if args.enable_cifs_correction else "disabled")
            corrected_metrics = None
            if args.enable_cifs_correction and args.cifs_trigger_mode == "disagreement" and local_idx != 0:
                reason = "stable"
                if track_yolo_iou is not None and track_yolo_iou < args.cifs_min_track_yolo_iou:
                    trigger = True
                    reason = "track_yolo_disagreement"
                if args.cifs_correction_interval and eval_index > 0 and eval_index % args.cifs_correction_interval == 0:
                    trigger = True
                    reason = "scheduled_interval"

            if trigger:
                if args.cifs_trigger_mode in {"official_window", "two_stage_window"}:
                    if official_selected is not None and int(official_selected["frame_idx"]) == actual_frame:
                        correction_mask_video = resize_np_mask(official_selected["mask_512"], gt_mask.shape)
                        corrected_metrics = mask_metrics(correction_mask_video, gt_mask)
                elif yolo_item is None:
                    raise RuntimeError(f"CIFS correction requested but no YOLO cache item for frame {actual_frame}")
                else:
                    correction_batch = build_refine_batch(yolo_item, gt_mask, args.category_id, args.image_size)
                    correction_mask_512, correction_debug = refined_init_mask(predictor, adapter, scorer, correction_batch, args)
                    correction_mask_video = resize_np_mask(correction_mask_512, gt_mask.shape)
                    corrected_metrics = mask_metrics(correction_mask_video, gt_mask)
                    old_add_all = getattr(predictor, "add_all_frames_to_correct_as_cond", False)
                    try:
                        predictor.add_all_frames_to_correct_as_cond = True
                        predictor.add_new_mask(
                            state,
                            frame_idx=local_idx,
                            obj_id="target",
                            mask=torch.from_numpy(correction_mask_512.astype(np.float32)),
                        )
                    finally:
                        predictor.add_all_frames_to_correct_as_cond = old_add_all
                    clear_outputs_after_frame(state, local_idx)
                    corrections.append(
                        {
                            "frame_idx": actual_frame,
                            "local_idx": local_idx,
                            "reason": reason,
                            "pre_iou": metrics["iou"],
                            "corrected_iou": corrected_metrics["iou"],
                            "track_yolo_iou": track_yolo_iou,
                            "yolo_iou": yolo_iou,
                            "refine_debug": correction_debug,
                        }
                    )

            row_metrics = corrected_metrics if (corrected_metrics is not None and args.score_corrected_frame) else metrics
            rows.append(
                {
                    "eval_index": eval_index,
                    "frame_idx": actual_frame,
                    "local_idx": local_idx,
                    **row_metrics,
                    "pre_correction_iou": metrics["iou"],
                    "yolo_iou": yolo_iou,
                    "track_yolo_iou": track_yolo_iou,
                    "triggered": bool(trigger),
                    "trigger_reason": reason,
                    "stage2_failure_trigger": bool(step_failure_trigger),
                    "stage2_failure_reason": step_failure_reason,
                    "selected_frame": int(official_selected["frame_idx"]) if official_selected is not None else None,
                    "selected_pred_iou": float(official_selected["pred_iou"]) if official_selected is not None else None,
                    "corrected_iou": corrected_metrics["iou"] if corrected_metrics is not None else None,
                }
            )
            eval_index += 1
        predictor.forward_text_emb = old_forward_text_emb

    arr = np.asarray(tracking_times[1:], dtype=np.float64)
    result = {
        "video": args.video,
        "annotations": args.annotations,
        "category_id": args.category_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "gt_eval_frames": eval_gt_frames,
        "adapter_checkpoint": {"checkpoint": args.adapter_checkpoint, **adapter_info},
        "model_checkpoint": model_info,
        "full_architecture_preflight": full_architecture_info,
        "use_no_mem_attention": bool(args.use_no_mem_attention),
        "use_mask_as_output": bool(args.use_mask_as_output),
        "cifs": {
            "enabled": bool(args.enable_cifs_correction),
            "trigger_mode": args.cifs_trigger_mode,
            "window_size": int(args.cifs_window_size),
            "iou_threshold": float(args.cifs_iou_threshold),
            "obj_score_threshold": float(args.cifs_obj_score_threshold),
            "stage2_min_track_yolo_iou": float(args.cifs_min_track_yolo_iou),
            "cache_dir": args.cifs_cache_dir,
            "cache_dataset": args.cifs_cache_dataset or args.cache_dataset,
        },
        "refine_debug": refine_debug,
        "init_refined_metrics_before_add": init_refined_metrics,
        "init_yolo_metrics": yolo_init_metrics,
        "init_added_mask_metrics": init_added_metrics,
        "eval_rows": rows,
        "summary": summarize(rows),
        "corrections": corrections,
        "cifs_trigger_events": cifs_trigger_events,
        "cifs_diagnostics": cifs_diagnostics,
        "num_corrections": len(corrections),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
