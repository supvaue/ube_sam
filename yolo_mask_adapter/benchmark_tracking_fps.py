"""Benchmark ReSurgSAM2 tracking-only FPS after one mask initialization."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


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


def selected_rows(cache_dir: Path, datasets: set[str], split: str, max_frames: int) -> list[dict]:
    rows = load_cache_index(cache_dir, datasets)
    train_paths, val_paths, _ = split_train_val(rows, val_fraction=0.2, split_mode="interleaved", split_seed=0)
    if split == "train":
        paths = train_paths
    elif split == "val":
        paths = val_paths
    else:
        paths = train_paths + val_paths

    out = []
    for path in sorted(paths):
        data = np.load(path, allow_pickle=False)
        image_path = Path(str(data["image_path"]))
        if not image_path.exists() and not image_path.is_absolute():
            image_path = WORKSPACE_ROOT / image_path
        out.append(
            {
                "cache_path": path,
                "image_path": image_path,
                "dataset": str(data["dataset"]),
                "frame_number": int(data["frame_number"]),
            }
        )
        if max_frames and len(out) >= max_frames:
            break
    return out


def make_frame_dir(rows: list[dict], work_dir: Path) -> Path:
    frame_dir = work_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        source = row["image_path"]
        suffix = source.suffix.lower() if source.suffix else ".png"
        target = frame_dir / f"{idx:05d}{suffix}"
        try:
            os.symlink(source, target)
        except OSError:
            shutil.copy2(source, target)
    return frame_dir


def load_init_mask(path: Path, source: str) -> torch.Tensor:
    data = np.load(path, allow_pickle=False)
    key = "yolo_mask" if source == "yolo" else "gt_mask"
    return torch.from_numpy(data[key].astype(np.float32))


def summarize(times: list[float]) -> dict:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "fps": float(1.0 / arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/forceps_mask_cache_full")
    parser.add_argument("--datasets", nargs="+", default=["exp1_cu_full", "exp2_cu_full"])
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mask-source", choices=["yolo", "gt"], default="yolo")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--use-mask-as-output", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = selected_rows(Path(args.cache_dir), set(args.datasets), args.split, args.max_frames)
    if len(rows) < 2:
        raise RuntimeError("Need at least 2 frames to benchmark tracking")

    predictor = build_predictor(args.device, use_mask_as_output=args.use_mask_as_output)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model" in checkpoint:
            missing, unexpected = predictor.load_state_dict(checkpoint["model"], strict=False)
            print(
                json.dumps(
                    {
                        "checkpoint": args.checkpoint,
                        "loaded_model_tensors": len(checkpoint["model"]),
                        "missing_keys": len(missing),
                        "unexpected_keys": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    predictor.eval()

    with tempfile.TemporaryDirectory(prefix="resurg_tracking_fps_") as tmp:
        frame_dir = make_frame_dir(rows, Path(tmp))
        state = predictor.init_state(
            str(frame_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            frame_interval=1,
        )
        init_mask = load_init_mask(rows[0]["cache_path"], args.mask_source)
        predictor.add_new_mask(state, frame_idx=0, obj_id="forceps", mask=init_mask)

        times = []
        iterator = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=len(rows))
        measured_frames = []
        step = 0
        while True:
            cuda_sync(args.device)
            start = time.perf_counter()
            try:
                frame_idx, _obj_ids, _masks = next(iterator)
            except StopIteration:
                break
            cuda_sync(args.device)
            elapsed = time.perf_counter() - start
            if frame_idx != 0:
                step += 1
                if step > args.warmup:
                    times.append(elapsed)
                    measured_frames.append(int(frame_idx))

    result = {
        "device": args.device,
        "max_frames": len(rows),
        "warmup": args.warmup,
        "mask_source": args.mask_source,
        "checkpoint": args.checkpoint,
        "use_mask_as_output": args.use_mask_as_output,
        "offload_video_to_cpu": args.offload_video_to_cpu,
        "offload_state_to_cpu": args.offload_state_to_cpu,
        "measured_items": len(times),
        "summary": summarize(times),
        "first_rows": [
            {"dataset": row["dataset"], "frame_number": row["frame_number"]}
            for row in rows[:5]
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
