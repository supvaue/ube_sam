"""Evaluate CIFS-selected refinement mask followed by ReSurgSAM2 tracking."""

import argparse
import json
import os
import shutil
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
from yolo_mask_adapter.cifs import CIFSConfig, select_credible_frame
from yolo_mask_adapter.mask_token_encoder import MaskTokenEncoder
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val
from yolo_mask_adapter.train_mask_token_prompt_adapter import forward_prompt_model


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def load_cache_item(path: Path, image_size: int, class_id: int) -> dict:
    data = np.load(path, allow_pickle=False)
    image_path = Path(str(data["image_path"]))
    if not image_path.exists() and not image_path.is_absolute():
        image_path = WORKSPACE_ROOT / image_path
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    image_t = (image_t - IMG_MEAN) / IMG_STD
    gt = torch.from_numpy(data["gt_mask"].astype(np.float32))[None, None]
    yolo = torch.from_numpy(data["yolo_mask"].astype(np.float32))[None, None]
    gt_512 = F.interpolate(gt, size=(image_size, image_size), mode="nearest")[0]
    yolo_512 = F.interpolate(yolo, size=(image_size, image_size), mode="nearest")[0]
    return {
        "image": image_t,
        "gt_mask": gt_512,
        "gt_mask_orig": data["gt_mask"].astype(np.uint8),
        "yolo_mask": yolo_512,
        "yolo_mask_orig": data["yolo_mask"].astype(np.uint8),
        "yolo_conf": float(data["yolo_conf"]),
        "class_id": class_id,
        "image_path": image_path,
        "dataset": str(data["dataset"]),
        "file_name": str(data["file_name"]),
        "frame_number": int(data["frame_number"]),
    }


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred_area + gt_area) if pred_area + gt_area else 1.0
    return {"iou": float(iou), "dice": float(dice)}


def make_frame_dir(items: list[dict], work_dir: Path) -> Path:
    frame_dir = work_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(items):
        source = item["image_path"]
        suffix = source.suffix.lower() if source.suffix else ".png"
        target = frame_dir / f"{idx:05d}{suffix}"
        try:
            os.symlink(source, target)
        except OSError:
            shutil.copy2(source, target)
    return frame_dir


def build_predictor(device: str, use_mask_as_output: bool):
    return build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_s_rvos.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_s_ref17.pth",
        device=device,
        strict_loading=False,
        apply_long_term_memory=True,
        hydra_overrides_extra=[
            "++scratch.use_sp_bimamba=true",
            "++scratch.use_dwconv=true",
            f"++model.use_mask_input_as_output_without_sam={'true' if use_mask_as_output else 'false'}",
        ],
    )


@torch.inference_mode()
def refine_mask(model, adapter, scorer, item: dict, device: str, prompt_mode: str, disable_obj_score_gating: bool) -> np.ndarray:
    batch = {
        "image": item["image"][None],
        "gt_mask": item["gt_mask"][None],
        "yolo_mask": item["yolo_mask"][None],
        "yolo_conf": torch.tensor([item["yolo_conf"]], dtype=torch.float32),
        "class_id": torch.tensor([item["class_id"]], dtype=torch.long),
    }
    logits = forward_prompt_model(
        model,
        adapter,
        scorer,
        batch,
        device,
        prompt_mode,
        disable_obj_score_gating=disable_obj_score_gating,
    )
    pred = (logits.sigmoid() > 0.5).float()
    pred = F.interpolate(pred, size=item["gt_mask_orig"].shape, mode="nearest")
    return pred[0, 0].detach().cpu().numpy().astype(np.uint8)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["exp1_cu_full", "exp2_cu_full"])
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--target-class-id", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-mode", choices=["dense", "condition", "dense_condition"], default="dense_condition")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--use-mask-as-output", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = load_cache_index(Path(args.cache_dir), set(args.datasets))
    train_paths, val_paths, _ = split_train_val(rows, val_fraction=0.2, split_mode="interleaved", split_seed=0)
    paths = {"train": train_paths, "val": val_paths, "all": train_paths + val_paths}[args.split]
    paths = sorted(paths)[: args.max_frames]
    items = [load_cache_item(path, args.image_size, args.target_class_id) for path in paths]

    ref_model = build_predictor(args.device, use_mask_as_output=False)
    ref_model.eval()
    adapter = MaskTokenEncoder(embed_dim=ref_model.hidden_dim, class_id=args.target_class_id).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    adapter.load_state_dict(checkpoint["adapter"])
    ref_model.load_state_dict(checkpoint["model"], strict=False)
    adapter.eval()
    scorer = MaskReliabilityScorer(target_class_id=args.target_class_id)

    cifs_candidates = [
        {"index": idx, "mask": item["yolo_mask_orig"], "confidence": item["yolo_conf"], "item": item}
        for idx, item in enumerate(items[: args.window_size])
    ]
    selected = select_credible_frame(cifs_candidates, CIFSConfig(window_size=args.window_size))
    key_idx = int(selected["index"])
    key_item = selected["item"]

    cuda_sync(args.device)
    refine_start = time.perf_counter()
    refined_init = refine_mask(
        ref_model,
        adapter,
        scorer,
        key_item,
        args.device,
        args.prompt_mode,
        disable_obj_score_gating=args.disable_obj_score_gating,
    )
    cuda_sync(args.device)
    refine_time = time.perf_counter() - refine_start
    del ref_model
    del adapter
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    predictor = build_predictor(args.device, use_mask_as_output=args.use_mask_as_output)
    predictor.eval()

    with tempfile.TemporaryDirectory(prefix="resurg_cifs_track_") as tmp:
        frame_dir = make_frame_dir(items, Path(tmp))
        state = predictor.init_state(str(frame_dir), frame_interval=1)
        predictor.add_new_mask(state, frame_idx=key_idx, obj_id="target", mask=torch.from_numpy(refined_init))
        iterator = predictor.propagate_in_video(state, start_frame_idx=key_idx, max_frame_num_to_track=len(items) - key_idx)
        track_rows = []
        yolo_rows = []
        tracking_times = []
        while True:
            cuda_sync(args.device)
            start = time.perf_counter()
            try:
                frame_idx, _obj_ids, masks = next(iterator)
            except StopIteration:
                break
            cuda_sync(args.device)
            tracking_times.append(time.perf_counter() - start)
            pred = (masks[0].detach().cpu().numpy().squeeze() > 0).astype(np.uint8)
            item = items[int(frame_idx)]
            track_rows.append(mask_metrics(pred, item["gt_mask_orig"]))
            yolo_rows.append(mask_metrics(item["yolo_mask_orig"], item["gt_mask_orig"]))

    result = {
        "cache_dir": args.cache_dir,
        "split": args.split,
        "max_frames": len(items),
        "key_frame": {
            "index": key_idx,
            "dataset": key_item["dataset"],
            "frame_number": key_item["frame_number"],
            "cifs": selected["cifs"],
            "refined_init": mask_metrics(refined_init, key_item["gt_mask_orig"]),
            "yolo_init": mask_metrics(key_item["yolo_mask_orig"], key_item["gt_mask_orig"]),
        },
        "refine_time_ms": float(refine_time * 1000.0),
        "tracked_frames": len(track_rows),
        "tracking_eval": summarize(track_rows),
        "yolo_eval_same_frames": summarize(yolo_rows),
    }
    if tracking_times:
        arr = np.asarray(tracking_times, dtype=np.float64)
        result["tracking_output_overhead_ms"] = float(arr.mean() * 1000.0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
