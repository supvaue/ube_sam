"""Evaluate dense video tracking and score only annotated GT frames."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from yolo_mask_adapter.build_forceps_mask_cache import rasterize_polygons
from yolo_mask_adapter.cifs import CIFSConfig, candidate_score, mask_iou as cifs_mask_iou, select_credible_frame, should_trigger_correction


FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def frame_number(file_name: str) -> int:
    match = FRAME_RE.search(Path(file_name).stem)
    return int(match.group(1)) if match else -1


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def build_predictor(args):
    overrides = [
        "++scratch.use_sp_bimamba=true",
        "++scratch.use_dwconv=true",
        f"++model.use_mask_input_as_output_without_sam={'true' if args.use_mask_as_output else 'false'}",
    ]
    if args.init_source == "yolo_token" or args.mask_token_checkpoint:
        overrides.append("++model._target_=yolo_mask_adapter.yolo_mask_video_predictor.YOLOMaskSAM2VideoPredictor")
    overrides.extend(args.hydra_override)
    print(
        "[stage] build predictor start "
        f"device={args.device} ckpt={args.ckpt_path} training_config={args.training_config_file} "
        f"overrides={overrides}",
        flush=True,
    )
    predictor = build_sam2_video_predictor(
        config_file=args.config_file,
        ckpt_path=args.ckpt_path,
        device=args.device,
        strict_loading=False,
        apply_long_term_memory=args.apply_long_term_memory,
        apply_credible_initial_frame=not args.disable_credible_initial_frame,
        num_cand_to_cond_frame=args.num_cand_to_cond_frame,
        num_cifs_candidate_frame=args.num_cifs_candidate_frame,
        num_long_mem_frame=args.num_long_mem_frame,
        training_config_file=args.training_config_file,
        hydra_overrides_extra=overrides,
    )
    print("[stage] build predictor done", flush=True)
    return predictor


def set_condition_fusion_mode(predictor, mode: str | None) -> None:
    if not mode or mode == "twoway":
        return
    if not hasattr(predictor, "cross_modal_fusion") or not hasattr(predictor.cross_modal_fusion, "set_condition_fusion_mode"):
        raise AttributeError("cross_modal_fusion does not support condition fusion modes")
    predictor.cross_modal_fusion.set_condition_fusion_mode(mode)
    print(json.dumps({"condition_fusion_mode": mode}, ensure_ascii=False), flush=True)


def load_mask_token_checkpoint(predictor, checkpoint_path: str, target_class_id: int, sentence_tokens: int) -> dict:
    """Load trained MaskTokenEncoder and selected ReSurgSAM2 heads into predictor."""
    if not checkpoint_path:
        return {"loaded": False}
    if not hasattr(predictor, "init_yolo_mask_adapter"):
        raise RuntimeError(
            "mask-token checkpoint requires YOLOMaskSAM2VideoPredictor; "
            "use --init-source yolo_token or provide the predictor target override"
        )
    print(f"[stage] loading mask-token checkpoint on CPU: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prior_mixer_depth = int(checkpoint.get("prior_mixer_depth", checkpoint.get("adapter_config", {}).get("prior_mixer_depth", 0)))
    prior_mixer_heads = int(checkpoint.get("prior_mixer_heads", checkpoint.get("adapter_config", {}).get("prior_mixer_heads", 4)))
    predictor.init_yolo_mask_adapter(
        target_class_id=target_class_id,
        sentence_tokens=sentence_tokens,
        prior_mixer_depth=prior_mixer_depth,
        prior_mixer_heads=prior_mixer_heads,
    )
    info = {
        "loaded": True,
        "checkpoint": checkpoint_path,
        "epoch": checkpoint.get("epoch"),
        "val": checkpoint.get("val"),
        "prior_mixer_depth": prior_mixer_depth,
        "prior_mixer_heads": prior_mixer_heads,
        "adapter_tensors": len(checkpoint.get("adapter", {})),
        "model_tensors": len(checkpoint.get("model", {})),
        "condition_fusion_mode": checkpoint.get("condition_fusion_mode", "twoway"),
    }
    set_condition_fusion_mode(predictor, checkpoint.get("condition_fusion_mode"))
    if "adapter" in checkpoint:
        print("[stage] loading adapter state", flush=True)
        missing, unexpected = predictor.yolo_mask_adapter.load_state_dict(checkpoint["adapter"], strict=False)
        info["adapter_missing_keys"] = len(missing)
        info["adapter_unexpected_keys"] = len(unexpected)
    if "model" in checkpoint:
        print("[stage] loading model state", flush=True)
        missing, unexpected = predictor.load_state_dict(checkpoint["model"], strict=False)
        info["missing_keys"] = len(missing)
        info["unexpected_keys"] = len(unexpected)
    predictor.yolo_mask_adapter.eval()
    print("[stage] checkpoint loaded", flush=True)
    return info


def load_model_checkpoint(predictor, checkpoint_path: str) -> dict:
    if not checkpoint_path:
        return {"loaded": False}
    print(f"[stage] loading model checkpoint on CPU: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    info = {
        "loaded": True,
        "checkpoint": checkpoint_path,
        "epoch": checkpoint.get("epoch"),
        "val": checkpoint.get("val"),
        "model_tensors": len(checkpoint.get("model", {})),
        "condition_fusion_mode": checkpoint.get("condition_fusion_mode", "twoway"),
    }
    set_condition_fusion_mode(predictor, checkpoint.get("condition_fusion_mode"))
    if "model" in checkpoint:
        missing, unexpected = predictor.load_state_dict(checkpoint["model"], strict=False)
        info["missing_keys"] = len(missing)
        info["unexpected_keys"] = len(unexpected)
    print("[stage] model checkpoint loaded", flush=True)
    return info


def load_gt_by_frame(annotation_path: Path, category_id: int) -> dict[int, dict]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        if ann.get("category_id") == category_id:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out = {}
    for image_id, anns in anns_by_image.items():
        image = images[image_id]
        frame_idx = frame_number(image["file_name"])
        if frame_idx < 0:
            continue
        out[frame_idx] = {
            "frame_idx": frame_idx,
            "file_name": image["file_name"],
            "height": int(image["height"]),
            "width": int(image["width"]),
            "mask": rasterize_polygons(anns, height=int(image["height"]), width=int(image["width"])),
        }
    return out


def extract_video_segment(video_path: Path, output_dir: Path, start_frame: int, end_frame: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for actual_idx in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        out_path = output_dir / f"{actual_idx - start_frame:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cap.release()


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred_area + gt_area) if pred_area + gt_area else 1.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "pred_area": float(pred.mean()),
        "gt_area": float(gt.mean()),
    }


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def load_mask_cache_by_frame(cache_dir: str, dataset_name: str) -> dict[int, dict]:
    if not cache_dir:
        return {}
    out = {}
    for path in Path(cache_dir).glob(f"{dataset_name}__frame_*.npz"):
        data = np.load(path, allow_pickle=False)
        frame_idx = int(data["frame_number"])
        out[frame_idx] = {
            "path": str(path),
            "yolo_mask": data["yolo_mask"].astype(np.uint8),
            "yolo_conf": float(data["yolo_conf"]),
        }
    return out


def prompt_mask_for_frame(
    source: str,
    frame_idx: int,
    gt_by_frame: dict[int, dict],
    cache_by_frame: dict[int, dict],
) -> tuple[np.ndarray, float | None]:
    if source in {"gt", "bbox"}:
        return gt_by_frame[frame_idx]["mask"].astype(np.float32), None
    if source in {"yolo", "yolo_bbox", "yolo_token"}:
        if frame_idx not in cache_by_frame:
            raise RuntimeError(f"No YOLO cache item for frame {frame_idx}; provide --cache-dir/--cache-dataset")
        item = cache_by_frame[frame_idx]
        return item["yolo_mask"].astype(np.float32), item["yolo_conf"]
    raise ValueError(f"Unknown init source: {source}")


def select_cifs_candidate(candidates: list[dict], config: CIFSConfig, policy: str) -> dict:
    if policy == "best":
        return select_credible_frame(candidates, config)
    if policy == "latest":
        usable = [
            item
            for item in candidates[-config.window_size :]
            if float(item.get("confidence", 0.0)) >= config.min_confidence
        ]
        selected = (usable or candidates[-config.window_size :])[-1]
        previous_mask = candidates[-2]["mask"] if len(candidates) >= 2 else None
        return {
            **selected,
            "cifs": candidate_score(
                selected["mask"],
                float(selected.get("confidence", 0.0)),
                previous_mask,
                config,
            ),
        }
    raise ValueError(f"Unknown CIFS selection policy: {policy}")


def tensor_stats(value) -> dict:
    if value is None:
        return {"exists": False}
    if isinstance(value, list):
        return {
            "exists": True,
            "type": "list",
            "len": len(value),
            "items": [tensor_stats(v) for v in value[:2]],
        }
    if not torch.is_tensor(value):
        return {"exists": True, "type": type(value).__name__}
    t = value.detach().float()
    return {
        "exists": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "mean": float(t.mean().cpu()) if t.numel() else 0.0,
        "std": float(t.std(unbiased=False).cpu()) if t.numel() else 0.0,
        "norm": float(t.norm().cpu()) if t.numel() else 0.0,
    }


def output_debug(out: dict | None) -> dict:
    if out is None:
        return {"exists": False}
    score = out.get("object_score_logits")
    iou = out.get("iou")
    return {
        "exists": True,
        "keys": sorted(out.keys()),
        "pred_masks": tensor_stats(out.get("pred_masks")),
        "maskmem_features": tensor_stats(out.get("maskmem_features")),
        "maskmem_pos_enc": tensor_stats(out.get("maskmem_pos_enc")),
        "object_score_logits": tensor_stats(score),
        "object_score_sigmoid": float(score.detach().sigmoid().flatten()[0].cpu()) if torch.is_tensor(score) and score.numel() else None,
        "iou_head": float(iou.detach().flatten()[0].cpu()) if torch.is_tensor(iou) and iou.numel() else None,
    }


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return tensor_stats(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def state_debug(predictor, state: dict, obj_idx: int = 0) -> dict:
    obj_out = state["output_dict_per_obj"].get(obj_idx, {})
    obj_temp = state["temp_output_dict_per_obj"].get(obj_idx, {})
    cond = obj_out.get("cond_frame_outputs", {})
    non_cond = obj_out.get("non_cond_frame_outputs", {})
    ref = obj_out.get("ref_frame_outputs", {})
    temp_cond = obj_temp.get("cond_frame_outputs", {})
    temp_non_cond = obj_temp.get("non_cond_frame_outputs", {})
    temp_ref = obj_temp.get("ref_frame_outputs", {})
    first_cond = sorted(cond)[0] if cond else None
    first_ref = sorted(ref)[0] if ref else None
    first_temp_ref = sorted(temp_ref)[0] if temp_ref else None
    first_temp_cond = sorted(temp_cond)[0] if temp_cond else None
    return {
        "predictor": {
            "class": type(predictor).__name__,
            "image_size": int(getattr(predictor, "image_size", -1)),
            "num_maskmem": int(getattr(predictor, "num_maskmem", -1)),
            "pred_obj_scores": bool(getattr(predictor, "pred_obj_scores", False)),
            "use_mask_input_as_output_without_sam": bool(getattr(predictor, "use_mask_input_as_output_without_sam", False)),
            "binarize_mask_from_pts_for_mem_enc": bool(getattr(predictor, "binarize_mask_from_pts_for_mem_enc", False)),
            "sigmoid_scale_for_mem_enc": float(getattr(predictor, "sigmoid_scale_for_mem_enc", 1.0)),
            "sigmoid_bias_for_mem_enc": float(getattr(predictor, "sigmoid_bias_for_mem_enc", 0.0)),
            "disable_non_cond_memory": bool(getattr(predictor, "disable_non_cond_memory", False)),
            "use_long_term_memory": bool(getattr(predictor, "use_long_term_memory", False)),
            "use_credible_initial_frame": bool(getattr(predictor, "use_credible_initial_frame", False)),
        },
        "state": {
            "num_frames": int(state.get("num_frames", -1)),
            "video_height": int(state.get("video_height", -1)),
            "video_width": int(state.get("video_width", -1)),
            "obj_ids": list(state.get("obj_ids", [])),
            "cond_frames": sorted(cond.keys()),
            "ref_frames": sorted(ref.keys()),
            "non_cond_count": len(non_cond),
            "temp_cond_frames": sorted(temp_cond.keys()),
            "temp_ref_frames": sorted(temp_ref.keys()),
            "temp_non_cond_frames": sorted(temp_non_cond.keys()),
            "first_cond_output": output_debug(cond.get(first_cond)) if first_cond is not None else None,
            "first_ref_output": output_debug(ref.get(first_ref)) if first_ref is not None else None,
            "first_temp_cond_output": output_debug(temp_cond.get(first_temp_cond)) if first_temp_cond is not None else None,
            "first_temp_ref_output": output_debug(temp_ref.get(first_temp_ref)) if first_temp_ref is not None else None,
        },
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "min_iou": float(np.min([row["iou"] for row in rows])),
        "p50_iou": float(np.quantile([row["iou"] for row in rows], 0.5)),
    }


def run_relay_segments(
    args,
    predictor,
    gt_by_frame: dict[int, dict],
    gt_frames: list[int],
    cache_by_frame: dict[int, dict],
) -> dict:
    segment_rows = []
    tracking_times = []
    pairs = list(zip(gt_frames[:-1], gt_frames[1:]))
    if args.relay_max_interval:
        pairs = [(start, end) for start, end in pairs if end - start <= args.relay_max_interval]
    if args.max_segments:
        pairs = pairs[: args.max_segments]
    if not pairs:
        raise RuntimeError("Need at least one GT frame pair for relay tracking")

    for segment_index, (seg_start, seg_end) in enumerate(pairs):
        with tempfile.TemporaryDirectory(prefix="resurg_relay_track_") as tmp:
            frame_dir = Path(tmp) / "frames"
            extract_video_segment(Path(args.video), frame_dir, seg_start, seg_end)
            state = predictor.init_state(str(frame_dir), frame_interval=1)
            init_mask_np, prompt_conf = prompt_mask_for_frame(args.init_source, seg_start, gt_by_frame, cache_by_frame)

            if args.init_source not in {"gt", "yolo", "bbox", "yolo_bbox"}:
                raise ValueError(f"Unknown init source: {args.init_source}")
            init_ret = add_prompt_to_state(
                predictor,
                state,
                local_frame_idx=0,
                source=args.init_source,
                prompt_mask_np=init_mask_np,
                prompt_confidence=prompt_conf,
            )

            init_pred = (init_ret[2][0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
            init_metrics = mask_metrics(init_pred, init_mask_np)

            old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
            if is_memory_only_prompt_source(args.init_source):
                predictor.forward_text_emb = False
            iterator = predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=seg_end - seg_start + 1,
            )
            end_pred = None
            while True:
                cuda_sync(args.device)
                step_start = time.perf_counter()
                try:
                    local_idx, _obj_ids, masks = next(iterator)
                except StopIteration:
                    break
                cuda_sync(args.device)
                tracking_times.append(time.perf_counter() - step_start)
                if int(local_idx) == seg_end - seg_start:
                    end_pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
            predictor.forward_text_emb = old_forward_text_emb

            if end_pred is None:
                raise RuntimeError(f"Relay segment {seg_start}->{seg_end} did not produce the end frame")
            end_metrics = mask_metrics(end_pred, gt_by_frame[seg_end]["mask"])
            segment_rows.append(
                {
                    "segment_index": segment_index,
                    "start_frame": seg_start,
                    "end_frame": seg_end,
                    "interval": seg_end - seg_start,
                    "prompt_confidence": prompt_conf,
                    "init_iou": init_metrics["iou"],
                    "init_dice": init_metrics["dice"],
                    "iou": end_metrics["iou"],
                    "dice": end_metrics["dice"],
                    "pred_area": end_metrics["pred_area"],
                    "gt_area": end_metrics["gt_area"],
                }
            )

    arr = np.asarray(tracking_times, dtype=np.float64)
    return {
        "mode": "relay_reset_at_gt",
        "segments": segment_rows,
        "summary": summarize(segment_rows),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }


def track_span(
    args,
    predictor,
    gt_by_frame: dict[int, dict],
    cache_by_frame: dict[int, dict],
    start_frame: int,
    end_frame: int,
    init_source: str,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="resurg_span_track_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(Path(args.video), frame_dir, start_frame, end_frame)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        init_mask_np, prompt_conf = prompt_mask_for_frame(init_source, start_frame, gt_by_frame, cache_by_frame)
        init_ret = add_prompt_to_state(
            predictor,
            state,
            local_frame_idx=0,
            source=init_source,
            prompt_mask_np=init_mask_np,
            prompt_confidence=prompt_conf,
        )
        init_iou = init_iou_against_prompt(init_ret, init_mask_np)
        old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
        if is_memory_only_prompt_source(init_source):
            predictor.forward_text_emb = False
        iterator = predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=end_frame - start_frame + 1,
        )
        end_pred = None
        tracking_times = []
        while True:
            cuda_sync(args.device)
            step_start = time.perf_counter()
            try:
                local_idx, _obj_ids, masks = next(iterator)
            except StopIteration:
                break
            cuda_sync(args.device)
            tracking_times.append(time.perf_counter() - step_start)
            if int(local_idx) == end_frame - start_frame:
                end_pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
        predictor.forward_text_emb = old_forward_text_emb
        if end_pred is None:
            raise RuntimeError(f"Tracking span {start_frame}->{end_frame} did not produce the end frame")
        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "prompt_confidence": prompt_conf,
            "init_iou_against_prompt": init_iou,
            "pred": end_pred,
            "tracking_times": tracking_times,
        }


def add_prompt_to_state(
    predictor,
    state: dict,
    local_frame_idx: int,
    source: str,
    prompt_mask_np: np.ndarray,
    prompt_confidence: float | None = None,
    force_condition: bool = False,
):
    old_add_all = getattr(predictor, "add_all_frames_to_correct_as_cond", False)
    old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
    if force_condition:
        predictor.add_all_frames_to_correct_as_cond = True
    try:
        if source in {"gt", "yolo"}:
            predictor.forward_text_emb = False
            return predictor.add_new_mask(
                state,
                frame_idx=local_frame_idx,
                obj_id="target",
                mask=torch.from_numpy(prompt_mask_np.astype(np.float32)),
            )
        if source == "yolo_token":
            if not hasattr(predictor, "add_new_yolo_mask"):
                raise RuntimeError("init_source=yolo_token requires YOLOMaskSAM2VideoPredictor")
            confidence = 0.95 if prompt_confidence is None else float(prompt_confidence)
            return predictor.add_new_yolo_mask(
                state,
                frame_idx=local_frame_idx,
                obj_id="target",
                mask=torch.from_numpy(prompt_mask_np.astype(np.float32)),
                confidence=confidence,
                class_id=getattr(predictor, "yolo_token_class_id", 12),
                require_usable=False,
            )
        if source in {"bbox", "yolo_bbox"}:
            bbox = mask_to_bbox(prompt_mask_np)
            if bbox is None:
                raise RuntimeError("Cannot initialize with bbox from an empty prompt mask")
            predictor.forward_text_emb = False
            return predictor.add_new_points_or_box(state, frame_idx=local_frame_idx, obj_id="target", box=bbox)
        raise ValueError(f"Unknown init source: {source}")
    finally:
        predictor.add_all_frames_to_correct_as_cond = old_add_all
        predictor.forward_text_emb = old_forward_text_emb


def correction_source_for_init(init_source: str) -> str:
    if init_source == "yolo_token":
        return "yolo_token"
    if init_source in {"yolo", "gt"}:
        return "yolo"
    return "yolo_bbox"


def correction_source_for_args(args) -> str:
    if args.cifs_correction_source != "auto":
        return args.cifs_correction_source
    return correction_source_for_init(args.init_source)


def is_memory_only_prompt_source(source: str) -> bool:
    """Prompt sources that should propagate without requiring text embeddings."""
    return source in {"gt", "yolo", "bbox", "yolo_bbox"}


def state_has_text_prompt(state: dict, obj_idx: int = 0) -> bool:
    return "text_emb" in state.get("text_inputs_per_obj", {}).get(obj_idx, {})


def init_iou_against_prompt(init_ret, prompt_mask_np: np.ndarray) -> float | None:
    if isinstance(init_ret, (tuple, list)) and len(init_ret) >= 3:
        init_pred = (init_ret[2][0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
        return mask_metrics(init_pred, prompt_mask_np)["iou"]
    return None


def clear_text_prompt_state(state: dict, obj_idx: int = 0) -> None:
    if obj_idx in state.get("text_inputs_per_obj", {}):
        state["text_inputs_per_obj"][obj_idx].clear()
    obj_output = state["output_dict_per_obj"].get(obj_idx, {})
    obj_temp = state["temp_output_dict_per_obj"].get(obj_idx, {})
    for storage in ("ref_frame_outputs",):
        obj_output.get(storage, {}).clear()
        obj_temp.get(storage, {}).clear()


def clear_outputs_after_frame(state: dict, local_frame_idx: int, obj_idx: int = 0) -> None:
    """Drop stale propagated outputs after inserting a correction frame."""
    obj_output = state["output_dict_per_obj"][obj_idx]
    obj_temp = state["temp_output_dict_per_obj"][obj_idx]
    for storage in ("non_cond_frame_outputs", "ref_frame_outputs"):
        for key in list(obj_output[storage].keys()):
            if key > local_frame_idx:
                obj_output[storage].pop(key, None)
    for storage in ("cond_frame_outputs", "non_cond_frame_outputs", "ref_frame_outputs"):
        for key in list(obj_temp[storage].keys()):
            if key > local_frame_idx:
                obj_temp[storage].pop(key, None)
    tracked = state["frames_tracked_per_obj"][obj_idx]
    for key in list(tracked.keys()):
        if key > local_frame_idx:
            tracked.pop(key, None)
    state["cached_features"].clear()


def propagate_state_span(args, predictor, state: dict, local_start: int, local_end: int) -> tuple[np.ndarray, list[float]]:
    old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
    if old_forward_text_emb and not state_has_text_prompt(state):
        predictor.forward_text_emb = False
    iterator = predictor.propagate_in_video(
        state,
        start_frame_idx=local_start,
        max_frame_num_to_track=local_end - local_start + 1,
    )
    end_pred = None
    tracking_times = []
    while True:
        cuda_sync(args.device)
        step_start = time.perf_counter()
        try:
            local_idx, _obj_ids, masks = next(iterator)
        except StopIteration:
            break
        cuda_sync(args.device)
        tracking_times.append(time.perf_counter() - step_start)
        if int(local_idx) == local_end:
            end_pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
    predictor.forward_text_emb = old_forward_text_emb
    if end_pred is None:
        raise RuntimeError(f"State propagation span {local_start}->{local_end} did not produce the end frame")
    return end_pred, tracking_times


def run_cifs_correction_single_state(
    args,
    predictor,
    gt_by_frame: dict[int, dict],
    gt_frames: list[int],
    cache_by_frame: dict[int, dict],
) -> dict:
    eval_frames = gt_frames[: args.max_gt_evals]
    if args.relay_max_interval:
        filtered = [eval_frames[0]]
        for frame in eval_frames[1:]:
            if frame - filtered[-1] <= args.relay_max_interval:
                filtered.append(frame)
            elif len(filtered) <= 1:
                filtered = [frame]
        eval_frames = filtered
    if args.max_segments:
        eval_frames = eval_frames[: args.max_segments + 1]
    if len(eval_frames) < 2:
        raise RuntimeError("Need at least two frames for CIFS single-state relay")

    cifs_config = CIFSConfig(
        window_size=args.cifs_window_size,
        min_confidence=args.cifs_min_confidence,
        area_jump_trigger=args.cifs_area_jump_trigger,
        low_conf_trigger=args.cifs_low_conf_trigger,
    )
    segment_start = eval_frames[0]
    segment_end = eval_frames[-1]
    setup_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="resurg_cifs_single_state_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(Path(args.video), frame_dir, segment_start, segment_end)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        init_mask_np, prompt_conf = prompt_mask_for_frame(args.init_source, segment_start, gt_by_frame, cache_by_frame)
        init_ret = add_prompt_to_state(predictor, state, 0, args.init_source, init_mask_np, prompt_confidence=prompt_conf)
        init_iou = init_iou_against_prompt(init_ret, init_mask_np)
        setup_time = time.perf_counter() - setup_start

        active_start = segment_start
        active_local_start = 0
        recent_candidates = []
        recent_states = []
        rows = []
        corrections = [
            {
                "frame_idx": active_start,
                "local_frame_idx": active_local_start,
                "reason": "initial",
                "source": args.init_source,
                "prompt_confidence": prompt_conf,
                "init_iou_against_prompt": init_iou,
            }
        ]
        all_tracking_times = []
        recomputed_frames = 0

        for eval_index, frame in enumerate(eval_frames[1:], start=1):
            local_frame = frame - segment_start
            pred, tracking_times = propagate_state_span(args, predictor, state, active_local_start, local_frame)
            all_tracking_times.extend(tracking_times)
            recomputed_frames += max(0, local_frame - active_local_start + 1)

            gt = gt_by_frame[frame]["mask"]
            metrics = mask_metrics(pred, gt)
            yolo_item = cache_by_frame.get(frame)
            yolo_iou = cifs_mask_iou(yolo_item["yolo_mask"], gt) if yolo_item is not None else None
            track_yolo_iou = cifs_mask_iou(pred, yolo_item["yolo_mask"]) if yolo_item is not None else None
            yolo_conf = yolo_item["yolo_conf"] if yolo_item is not None else 0.0
            state_score = track_yolo_iou if track_yolo_iou is not None else metrics["iou"]
            recent_states.append(
                {
                    "frame_idx": frame,
                    "area_ratio": metrics["pred_area"],
                    "confidence": yolo_conf,
                    "score": state_score,
                }
            )
            if yolo_item is not None:
                recent_candidates.append(
                    {
                        "index": frame,
                        "mask": yolo_item["yolo_mask"],
                        "confidence": yolo_conf,
                        "frame_idx": frame,
                    }
                )
                recent_candidates = recent_candidates[-args.cifs_window_size :]

            trigger, reason = should_trigger_correction(recent_states, cifs_config)
            if track_yolo_iou is not None and track_yolo_iou < args.cifs_min_track_yolo_iou:
                trigger, reason = True, "track_yolo_disagreement"
            if args.cifs_correction_interval and eval_index % args.cifs_correction_interval == 0:
                trigger, reason = True, "scheduled_interval"

            selected = None
            if trigger and recent_candidates:
                selected = select_cifs_candidate(recent_candidates, cifs_config, args.cifs_selection_policy)
                selected_frame = int(selected["frame_idx"])
                selected_local = selected_frame - segment_start
                correction_source = correction_source_for_args(args)
                selected_mask, selected_conf = prompt_mask_for_frame(correction_source, selected_frame, gt_by_frame, cache_by_frame)
                add_prompt_to_state(
                    predictor,
                    state,
                    selected_local,
                    correction_source,
                    selected_mask,
                    prompt_confidence=selected_conf,
                    force_condition=True,
                )
                if args.clear_text_after_mask_correction and correction_source != "yolo_token":
                    clear_text_prompt_state(state)
                    predictor.forward_text_emb = False
                clear_outputs_after_frame(state, selected_local)
                active_start = selected_frame
                active_local_start = selected_local
                corrections.append(
                    {
                        "frame_idx": active_start,
                        "local_frame_idx": active_local_start,
                        "reason": reason,
                        "source": correction_source,
                        "prompt_confidence": selected_conf,
                        "cifs": selected["cifs"],
                    }
                )
                recent_states = []
            else:
                active_local_start = local_frame

            rows.append(
                {
                    "eval_index": eval_index,
                    "frame_idx": frame,
                    "active_start_frame": active_start,
                    "active_interval": frame - active_start,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "pred_area": metrics["pred_area"],
                    "gt_area": metrics["gt_area"],
                    "yolo_iou": yolo_iou,
                    "track_yolo_iou": track_yolo_iou,
                    "triggered": bool(trigger),
                    "trigger_reason": reason,
                    "selected_frame": int(selected["frame_idx"]) if selected is not None else None,
                }
            )

    arr = np.asarray(all_tracking_times, dtype=np.float64)
    result = {
        "mode": "cifs_correction_single_state",
        "eval_rows": rows,
        "summary": summarize(rows),
        "corrections": corrections,
        "num_corrections": len(corrections),
        "setup_time_sec": float(setup_time),
        "tracked_steps": int(arr.size),
        "recomputed_frames": int(recomputed_frames),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }
    if rows and rows[0].get("yolo_iou") is not None:
        result["mean_yolo_iou_at_eval_frames"] = float(np.mean([r["yolo_iou"] for r in rows if r["yolo_iou"] is not None]))
        result["mean_track_yolo_iou"] = float(np.mean([r["track_yolo_iou"] for r in rows if r["track_yolo_iou"] is not None]))
    return result


def run_cifs_correction_stream_state(
    args,
    predictor,
    gt_by_frame: dict[int, dict],
    gt_frames: list[int],
    cache_by_frame: dict[int, dict],
) -> dict:
    eval_frames = gt_frames[: args.max_gt_evals]
    if args.relay_max_interval:
        filtered = [eval_frames[0]]
        for frame in eval_frames[1:]:
            if frame - filtered[-1] <= args.relay_max_interval:
                filtered.append(frame)
            elif len(filtered) <= 1:
                filtered = [frame]
        eval_frames = filtered
    if args.max_segments:
        eval_frames = eval_frames[: args.max_segments + 1]
    if len(eval_frames) < 2:
        raise RuntimeError("Need at least two frames for CIFS stream-state relay")

    cifs_config = CIFSConfig(
        window_size=args.cifs_window_size,
        min_confidence=args.cifs_min_confidence,
        area_jump_trigger=args.cifs_area_jump_trigger,
        low_conf_trigger=args.cifs_low_conf_trigger,
    )
    segment_start = eval_frames[0]
    segment_end = eval_frames[-1]
    setup_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="resurg_cifs_stream_state_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(Path(args.video), frame_dir, segment_start, segment_end)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        init_mask_np, prompt_conf = prompt_mask_for_frame(args.init_source, segment_start, gt_by_frame, cache_by_frame)
        init_ret = add_prompt_to_state(predictor, state, 0, args.init_source, init_mask_np, prompt_confidence=prompt_conf)
        init_iou = init_iou_against_prompt(init_ret, init_mask_np)
        setup_time = time.perf_counter() - setup_start

        active_start = segment_start
        active_local_start = 0
        next_eval_pos = 1
        recent_candidates = []
        recent_states = []
        rows = []
        corrections = [
            {
                "frame_idx": active_start,
                "local_frame_idx": active_local_start,
                "reason": "initial",
                "source": args.init_source,
                "prompt_confidence": prompt_conf,
                "init_iou_against_prompt": init_iou,
            }
        ]
        all_tracking_times = []
        restarts = 0

        while next_eval_pos < len(eval_frames):
            restarts += 1
            iterator = predictor.propagate_in_video(
                state,
                start_frame_idx=active_local_start,
                max_frame_num_to_track=segment_end - segment_start - active_local_start + 1,
            )
            restart_requested = False
            while True:
                cuda_sync(args.device)
                step_start = time.perf_counter()
                try:
                    _local_idx, _obj_ids, masks = next(iterator)
                except StopIteration:
                    break
                cuda_sync(args.device)
                all_tracking_times.append(time.perf_counter() - step_start)
                local_idx = int(_local_idx)
                target_frame = eval_frames[next_eval_pos]
                target_local = target_frame - segment_start
                if local_idx != target_local:
                    continue

                pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
                gt = gt_by_frame[target_frame]["mask"]
                metrics = mask_metrics(pred, gt)
                yolo_item = cache_by_frame.get(target_frame)
                yolo_iou = cifs_mask_iou(yolo_item["yolo_mask"], gt) if yolo_item is not None else None
                track_yolo_iou = cifs_mask_iou(pred, yolo_item["yolo_mask"]) if yolo_item is not None else None
                yolo_conf = yolo_item["yolo_conf"] if yolo_item is not None else 0.0
                state_score = track_yolo_iou if track_yolo_iou is not None else metrics["iou"]
                recent_states.append(
                    {
                        "frame_idx": target_frame,
                        "area_ratio": metrics["pred_area"],
                        "confidence": yolo_conf,
                        "score": state_score,
                    }
                )
                if yolo_item is not None:
                    recent_candidates.append(
                        {
                            "index": target_frame,
                            "mask": yolo_item["yolo_mask"],
                            "confidence": yolo_conf,
                            "frame_idx": target_frame,
                        }
                    )
                    recent_candidates = recent_candidates[-args.cifs_window_size :]

                trigger, reason = should_trigger_correction(recent_states, cifs_config)
                if track_yolo_iou is not None and track_yolo_iou < args.cifs_min_track_yolo_iou:
                    trigger, reason = True, "track_yolo_disagreement"
                if args.cifs_correction_interval and next_eval_pos % args.cifs_correction_interval == 0:
                    trigger, reason = True, "scheduled_interval"

                selected = None
                if trigger and recent_candidates:
                    selected = select_cifs_candidate(recent_candidates, cifs_config, args.cifs_selection_policy)

                rows.append(
                    {
                        "eval_index": next_eval_pos,
                        "frame_idx": target_frame,
                        "active_start_frame": active_start,
                        "active_interval": target_frame - active_start,
                        "iou": metrics["iou"],
                        "dice": metrics["dice"],
                        "pred_area": metrics["pred_area"],
                        "gt_area": metrics["gt_area"],
                        "yolo_iou": yolo_iou,
                        "track_yolo_iou": track_yolo_iou,
                        "triggered": bool(trigger),
                        "trigger_reason": reason,
                        "selected_frame": int(selected["frame_idx"]) if selected is not None else None,
                    }
                )
                next_eval_pos += 1

                if selected is not None:
                    selected_frame = int(selected["frame_idx"])
                    selected_local = selected_frame - segment_start
                    correction_source = correction_source_for_args(args)
                    selected_mask, selected_conf = prompt_mask_for_frame(correction_source, selected_frame, gt_by_frame, cache_by_frame)
                    add_prompt_to_state(
                        predictor,
                        state,
                        selected_local,
                        correction_source,
                        selected_mask,
                        prompt_confidence=selected_conf,
                        force_condition=True,
                    )
                    clear_outputs_after_frame(state, selected_local)
                    active_start = selected_frame
                    active_local_start = selected_local
                    corrections.append(
                        {
                            "frame_idx": active_start,
                            "local_frame_idx": active_local_start,
                            "reason": reason,
                            "source": correction_source,
                            "prompt_confidence": selected_conf,
                            "cifs": selected["cifs"],
                        }
                    )
                    recent_states = []
                    restart_requested = True
                    break

                if next_eval_pos >= len(eval_frames):
                    break
            if not restart_requested:
                break

    arr = np.asarray(all_tracking_times, dtype=np.float64)
    result = {
        "mode": "cifs_correction_stream_state",
        "eval_rows": rows,
        "summary": summarize(rows),
        "corrections": corrections,
        "num_corrections": len(corrections),
        "setup_time_sec": float(setup_time),
        "tracked_steps": int(arr.size),
        "stream_restarts": int(restarts),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }
    if rows and rows[0].get("yolo_iou") is not None:
        result["mean_yolo_iou_at_eval_frames"] = float(np.mean([r["yolo_iou"] for r in rows if r["yolo_iou"] is not None]))
        result["mean_track_yolo_iou"] = float(np.mean([r["track_yolo_iou"] for r in rows if r["track_yolo_iou"] is not None]))
    return result


def run_cifs_correction_relay(
    args,
    predictor,
    gt_by_frame: dict[int, dict],
    gt_frames: list[int],
    cache_by_frame: dict[int, dict],
) -> dict:
    eval_frames = gt_frames[: args.max_gt_evals]
    if args.relay_max_interval:
        # Preserve frame order while skipping only the intervals that are too large.
        filtered = [eval_frames[0]]
        for frame in eval_frames[1:]:
            if frame - filtered[-1] <= args.relay_max_interval:
                filtered.append(frame)
            elif len(filtered) <= 1:
                filtered = [frame]
        eval_frames = filtered
    if args.max_segments:
        eval_frames = eval_frames[: args.max_segments + 1]
    if len(eval_frames) < 2:
        raise RuntimeError("Need at least two frames for CIFS correction relay")

    cifs_config = CIFSConfig(
        window_size=args.cifs_window_size,
        min_confidence=args.cifs_min_confidence,
        area_jump_trigger=args.cifs_area_jump_trigger,
        low_conf_trigger=args.cifs_low_conf_trigger,
    )
    correction_start = eval_frames[0]
    correction_source = args.init_source
    recent_candidates = []
    recent_states = []
    rows = []
    corrections = [
        {
            "frame_idx": correction_start,
            "reason": "initial",
            "source": correction_source,
        }
    ]
    all_tracking_times = []

    for eval_index, frame in enumerate(eval_frames[1:], start=1):
        span = track_span(args, predictor, gt_by_frame, cache_by_frame, correction_start, frame, correction_source)
        all_tracking_times.extend(span["tracking_times"])
        pred = span["pred"]
        gt = gt_by_frame[frame]["mask"]
        metrics = mask_metrics(pred, gt)
        yolo_item = cache_by_frame.get(frame)
        yolo_iou = cifs_mask_iou(yolo_item["yolo_mask"], gt) if yolo_item is not None else None
        track_yolo_iou = cifs_mask_iou(pred, yolo_item["yolo_mask"]) if yolo_item is not None else None
        yolo_conf = yolo_item["yolo_conf"] if yolo_item is not None else 0.0
        state_score = track_yolo_iou if track_yolo_iou is not None else metrics["iou"]
        recent_states.append(
            {
                "frame_idx": frame,
                "area_ratio": metrics["pred_area"],
                "confidence": yolo_conf,
                "score": state_score,
            }
        )
        if yolo_item is not None:
            recent_candidates.append(
                {
                    "index": frame,
                    "mask": yolo_item["yolo_mask"],
                    "confidence": yolo_conf,
                    "frame_idx": frame,
                }
            )
            recent_candidates = recent_candidates[-args.cifs_window_size :]

        trigger, reason = should_trigger_correction(recent_states, cifs_config)
        if track_yolo_iou is not None and track_yolo_iou < args.cifs_min_track_yolo_iou:
            trigger, reason = True, "track_yolo_disagreement"
        if args.cifs_correction_interval and eval_index % args.cifs_correction_interval == 0:
            trigger, reason = True, "scheduled_interval"

        selected = None
        if trigger and recent_candidates:
            selected = select_cifs_candidate(recent_candidates, cifs_config, args.cifs_selection_policy)
            correction_start = int(selected["frame_idx"])
            correction_source = correction_source_for_args(args)
            corrections.append(
                {
                    "frame_idx": correction_start,
                    "reason": reason,
                    "source": correction_source,
                    "cifs": selected["cifs"],
                }
            )
            recent_states = []

        rows.append(
            {
                "eval_index": eval_index,
                "frame_idx": frame,
                "active_start_frame": span["start_frame"],
                "active_interval": frame - span["start_frame"],
                "prompt_confidence": span["prompt_confidence"],
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "pred_area": metrics["pred_area"],
                "gt_area": metrics["gt_area"],
                "yolo_iou": yolo_iou,
                "track_yolo_iou": track_yolo_iou,
                "triggered": bool(trigger),
                "trigger_reason": reason,
                "selected_frame": int(selected["frame_idx"]) if selected is not None else None,
            }
        )

    arr = np.asarray(all_tracking_times, dtype=np.float64)
    result = {
        "mode": "cifs_correction_relay",
        "eval_rows": rows,
        "summary": summarize(rows),
        "corrections": corrections,
        "num_corrections": len(corrections),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }
    if rows and rows[0].get("yolo_iou") is not None:
        result["mean_yolo_iou_at_eval_frames"] = float(np.mean([r["yolo_iou"] for r in rows if r["yolo_iou"] is not None]))
        result["mean_track_yolo_iou"] = float(np.mean([r["track_yolo_iou"] for r in rows if r["track_yolo_iou"] is not None]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-video-frames", type=int, default=241)
    parser.add_argument("--max-gt-evals", type=int, default=10)
    parser.add_argument("--relay-reset-at-gt", action="store_true")
    parser.add_argument("--cifs-correction-relay", action="store_true")
    parser.add_argument("--cifs-single-state", action="store_true")
    parser.add_argument("--cifs-stream-state", action="store_true")
    parser.add_argument("--relay-max-interval", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--cifs-window-size", type=int, default=5)
    parser.add_argument("--cifs-min-confidence", type=float, default=0.45)
    parser.add_argument("--cifs-area-jump-trigger", type=float, default=0.45)
    parser.add_argument("--cifs-low-conf-trigger", type=float, default=0.35)
    parser.add_argument("--cifs-min-track-yolo-iou", type=float, default=0.55)
    parser.add_argument("--cifs-correction-interval", type=int, default=0)
    parser.add_argument("--cifs-selection-policy", choices=["best", "latest"], default="best")
    parser.add_argument(
        "--cifs-correction-source",
        choices=["auto", "yolo", "yolo_token", "yolo_bbox"],
        default="auto",
        help="Prompt type used when CIFS inserts a correction frame. auto preserves the init-source family.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config-file", default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--training-config-file", default="")
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument("--apply-long-term-memory", action="store_true")
    parser.add_argument("--num-long-mem-frame", type=int, default=3)
    parser.add_argument("--disable-credible-initial-frame", action="store_true")
    parser.add_argument("--num-cand-to-cond-frame", type=int, default=1)
    parser.add_argument("--num-cifs-candidate-frame", type=int, default=5)
    parser.add_argument(
        "--disable-non-cond-memory",
        action="store_true",
        help="Use only selected conditioning memory and skip propagated non-cond/long-term memories.",
    )
    parser.add_argument(
        "--credible-obj-score-threshold",
        type=float,
        default=0.9,
        help="Object-score threshold used by ReSurgSAM2 credible initial frame selection.",
    )
    parser.add_argument(
        "--credible-iou-threshold",
        type=float,
        default=0.7,
        help="IoU-head threshold used by ReSurgSAM2 credible initial frame selection.",
    )
    parser.add_argument(
        "--binarize-mask-for-mem-enc",
        action="store_true",
        help="Force selected prompt/referring masks to be binarized before memory encoding.",
    )
    parser.add_argument("--init-source", choices=["gt", "bbox", "yolo", "yolo_bbox", "yolo_token", "text"], default="gt")
    parser.add_argument(
        "--text-prompt",
        default="ligamentum flavum",
        help="Text expression used when --init-source text follows the official RVOS/referring entry.",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cache-dataset", default="exp1_cu_full")
    parser.add_argument("--mask-token-checkpoint", default="")
    parser.add_argument(
        "--model-checkpoint",
        default="",
        help="Optional shared decoder/model checkpoint loaded after --mask-token-checkpoint; useful when adapter and model are saved separately.",
    )
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--use-mask-as-output", action="store_true")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument(
        "--clear-text-after-mask-correction",
        action="store_true",
        help="When CIFS correction writes a mask/bbox prompt, clear stale referring text tokens and continue as memory-only tracking.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    gt_by_frame = load_gt_by_frame(Path(args.annotations), args.category_id)
    gt_frames = sorted(frame for frame in gt_by_frame if frame >= args.start_frame)
    if not gt_frames:
        raise RuntimeError("No GT frames found after start frame")
    start_frame = gt_frames[0]
    eval_gt_frames = gt_frames[: args.max_gt_evals]
    requested_end = start_frame + args.max_video_frames - 1
    end_frame = min(requested_end, eval_gt_frames[-1])
    eval_gt_frames = [frame for frame in eval_gt_frames if frame <= end_frame]
    if len(eval_gt_frames) < 2:
        raise RuntimeError("Need at least two GT frames inside the dense segment")

    predictor = build_predictor(args)
    predictor.eval()
    predictor.credible_obj_score_threshold = args.credible_obj_score_threshold
    predictor.credible_iou_threshold = args.credible_iou_threshold
    predictor.disable_non_cond_memory = args.disable_non_cond_memory
    predictor.yolo_token_class_id = args.category_id
    mask_token_info = load_mask_token_checkpoint(
        predictor,
        args.mask_token_checkpoint,
        target_class_id=args.category_id,
        sentence_tokens=args.sentence_tokens,
    )
    model_checkpoint_info = load_model_checkpoint(predictor, args.model_checkpoint)
    if args.binarize_mask_for_mem_enc:
        predictor.binarize_mask_from_pts_for_mem_enc = True
    if args.disable_obj_score_gating:
        predictor.pred_obj_scores = False
    cache_by_frame = load_mask_cache_by_frame(args.cache_dir, args.cache_dataset)

    if args.relay_reset_at_gt:
        result = {
            "video": args.video,
            "annotations": args.annotations,
            "category_id": args.category_id,
            "config_file": args.config_file,
            "ckpt_path": args.ckpt_path,
            "training_config_file": args.training_config_file,
            "init_source": args.init_source,
            "cifs_correction_source": args.cifs_correction_source,
            "use_mask_as_output": args.use_mask_as_output,
            "disable_obj_score_gating": args.disable_obj_score_gating,
            "clear_text_after_mask_correction": args.clear_text_after_mask_correction,
            "mask_token_checkpoint": mask_token_info,
            "model_checkpoint": model_checkpoint_info,
            "apply_long_term_memory": args.apply_long_term_memory,
            "disable_credible_initial_frame": args.disable_credible_initial_frame,
            "gt_eval_frames": eval_gt_frames,
            **run_relay_segments(args, predictor, gt_by_frame, eval_gt_frames, cache_by_frame),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return

    if args.cifs_correction_relay:
        if args.cifs_stream_state:
            cifs_runner = run_cifs_correction_stream_state
        elif args.cifs_single_state:
            cifs_runner = run_cifs_correction_single_state
        else:
            cifs_runner = run_cifs_correction_relay
        result = {
            "video": args.video,
            "annotations": args.annotations,
            "category_id": args.category_id,
            "config_file": args.config_file,
            "ckpt_path": args.ckpt_path,
            "training_config_file": args.training_config_file,
            "init_source": args.init_source,
            "cifs_correction_source": args.cifs_correction_source,
            "use_mask_as_output": args.use_mask_as_output,
            "disable_obj_score_gating": args.disable_obj_score_gating,
            "clear_text_after_mask_correction": args.clear_text_after_mask_correction,
            "mask_token_checkpoint": mask_token_info,
            "model_checkpoint": model_checkpoint_info,
            "apply_long_term_memory": args.apply_long_term_memory,
            "disable_credible_initial_frame": args.disable_credible_initial_frame,
            "gt_eval_frames": eval_gt_frames,
            **cifs_runner(args, predictor, gt_by_frame, eval_gt_frames, cache_by_frame),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return

    with tempfile.TemporaryDirectory(prefix="resurg_dense_track_") as tmp:
        frame_dir = Path(tmp) / "frames"
        extract_video_segment(Path(args.video), frame_dir, start_frame, end_frame)
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        debug_snapshots = {}
        if args.debug:
            debug_snapshots["after_init_state"] = state_debug(predictor, state)
        init_mask_np = None
        _prompt_conf = None
        if args.init_source != "text":
            init_mask_np, _prompt_conf = prompt_mask_for_frame(args.init_source, start_frame, gt_by_frame, cache_by_frame)
        if args.init_source in {"gt", "yolo"}:
            init_ret = add_prompt_to_state(
                predictor,
                state,
                local_frame_idx=0,
                source=args.init_source,
                prompt_mask_np=init_mask_np,
                prompt_confidence=_prompt_conf,
            )
            init_masks_for_debug = init_ret[2]
        elif args.init_source == "yolo_token":
            init_ret = add_prompt_to_state(
                predictor,
                state,
                local_frame_idx=0,
                source="yolo_token",
                prompt_mask_np=init_mask_np,
                prompt_confidence=0.95 if _prompt_conf is None else float(_prompt_conf),
            )
            init_masks_for_debug = None
        elif args.init_source in {"bbox", "yolo_bbox"}:
            init_ret = add_prompt_to_state(
                predictor,
                state,
                local_frame_idx=0,
                source=args.init_source,
                prompt_mask_np=init_mask_np,
                prompt_confidence=_prompt_conf,
            )
            init_masks_for_debug = init_ret[2]
        elif args.init_source == "text":
            init_ret = predictor.add_new_text(
                state,
                frame_idx=0,
                obj_id="target",
                text=[args.text_prompt],
            )
            init_masks_for_debug = None
        else:
            raise ValueError(f"Unknown init source: {args.init_source}")
        if args.debug:
            debug_snapshots["after_add_prompt"] = state_debug(predictor, state)
            debug_snapshots["add_prompt_result"] = json_safe(init_ret)
            if init_masks_for_debug is not None and init_mask_np is not None:
                init_pred = (init_masks_for_debug[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
                debug_snapshots["init_prompt_metrics"] = mask_metrics(init_pred, init_mask_np)

        predictor.propagate_in_video_preflight(state)
        if args.debug:
            debug_snapshots["after_preflight"] = state_debug(predictor, state)
        old_forward_text_emb = getattr(predictor, "forward_text_emb", False)
        if is_memory_only_prompt_source(args.init_source):
            predictor.forward_text_emb = False
        iterator = predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=end_frame - start_frame + 1,
        )
        eval_lookup = {frame - start_frame: frame for frame in eval_gt_frames}
        tracking_times = []
        rows = []
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
            if local_idx not in eval_lookup:
                continue
            actual_frame = eval_lookup[local_idx]
            pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
            gt = gt_by_frame[actual_frame]["mask"]
            row = {
                "frame_idx": actual_frame,
                "local_idx": local_idx,
                **mask_metrics(pred, gt),
            }
            if args.debug:
                obj_out = state["output_dict_per_obj"][0]
                current_out = obj_out["cond_frame_outputs"].get(local_idx) or obj_out["non_cond_frame_outputs"].get(local_idx)
                row["output_debug"] = output_debug(current_out)
            rows.append(row)
        predictor.forward_text_emb = old_forward_text_emb
        if args.debug:
            debug_snapshots["after_propagation"] = state_debug(predictor, state)

    arr = np.asarray(tracking_times[1:], dtype=np.float64)
    result = {
        "video": args.video,
        "annotations": args.annotations,
        "category_id": args.category_id,
        "config_file": args.config_file,
        "ckpt_path": args.ckpt_path,
        "training_config_file": args.training_config_file,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "video_frames": end_frame - start_frame + 1,
        "init_source": args.init_source,
        "text_prompt": args.text_prompt if args.init_source == "text" else None,
        "use_mask_as_output": args.use_mask_as_output,
        "disable_obj_score_gating": args.disable_obj_score_gating,
        "mask_token_checkpoint": mask_token_info,
        "model_checkpoint": model_checkpoint_info,
        "apply_long_term_memory": args.apply_long_term_memory,
        "disable_credible_initial_frame": args.disable_credible_initial_frame,
        "gt_eval_frames": eval_gt_frames,
        "eval_rows": rows,
        "summary": summarize(rows),
        "tracking_fps": float(1.0 / arr.mean()) if arr.size else 0.0,
        "tracking_mean_ms": float(arr.mean() * 1000.0) if arr.size else 0.0,
    }
    if args.debug:
        result["debug"] = debug_snapshots
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
