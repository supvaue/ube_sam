"""Smoke test for injecting a YOLO mask into ReSurgSAM2 video predictor."""

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor


def build_predictor(device: str, use_mask_as_output: bool):
    overrides = [
        "++model._target_=yolo_mask_adapter.yolo_mask_video_predictor.YOLOMaskSAM2VideoPredictor",
        "++scratch.use_sp_bimamba=true",
        "++scratch.use_dwconv=true",
    ]
    overrides.append(
        f"++model.use_mask_input_as_output_without_sam={'true' if use_mask_as_output else 'false'}"
    )
    return build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_s_rvos.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_s_ref17.pth",
        device=device,
        strict_loading=False,
        apply_long_term_memory=True,
        hydra_overrides_extra=overrides,
    )


def make_center_mask(height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(height, width, dtype=torch.float32)
    y0, y1 = int(height * 0.36), int(height * 0.64)
    x0, x1 = int(width * 0.36), int(width * 0.64)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def register_joint_yolo_class(ultralytics_root: Path) -> None:
    """Register the custom class name stored in joint_train_v2 checkpoints."""
    sys.path.insert(0, str(ultralytics_root))
    import torch.nn as nn
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.tasks import SegmentationModel

    class JointSegClsModel(SegmentationModel):
        def __init__(self, cfg="yolo11m-seg.yaml", nc=28, nc_cls=5, verbose=True):
            self._p5 = [None]
            super().__init__(cfg=cfg, nc=nc, verbose=verbose)
            self._p5[0] = None
            self.nc_cls = nc_cls
            self.cls_head = nn.Sequential(
                Conv(512, 256, 1, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(256, nc_cls),
            )

        def _predict_once(self, x, profile=False, visualize=False, embed=None):
            y, dt, embeddings = [], [], []
            embed = frozenset(embed) if embed is not None else {-1}
            for i, m in enumerate(self.model):
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
                y.append(x if m.i in self.save else None)
                if i == 22:
                    self._p5[0] = x
            return x

    globals()["JointSegClsModel"] = JointSegClsModel


def image_paths(video: str) -> list[str]:
    path = Path(video)
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            raise FileNotFoundError(f"No images found in {path}")
        return [str(p) for p in images]
    return [video]


def make_yolo_mask(
    yolo_path: str,
    source: str,
    device: str,
    target_yolo_cls: int,
    conf: float,
    iou: float,
    imgsz: int,
    search_frames: int,
) -> tuple[torch.Tensor, float, int, dict]:
    ultralytics_root = REPO_ROOT.parent / "ultralytics-main"
    register_joint_yolo_class(ultralytics_root)
    from ultralytics import YOLO

    model = YOLO(yolo_path)
    checked = []
    found = None
    for source_index, source_item in enumerate(image_paths(source)[: max(search_frames, 1)]):
        result = model.predict(
            source=source_item,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device.replace("cuda:", ""),
            verbose=False,
        )[0]
        names = result.names
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            checked.append({"source": source_item, "classes": []})
            continue
        cls_ids = result.boxes.cls.detach().cpu().long()
        confs = result.boxes.conf.detach().cpu().float()
        checked.append({"source": source_item, "classes": cls_ids.tolist()})
        candidates = (cls_ids == target_yolo_cls).nonzero(as_tuple=False).flatten()
        if len(candidates) > 0:
            found = (source_index, source_item, result, names, cls_ids, confs, candidates)
            break
    if found is None:
        raise RuntimeError(
            f"YOLO produced no masks for class {target_yolo_cls}; checked={checked[:8]}"
        )

    source_index, source_item, result, names, cls_ids, confs, candidates = found
    best = candidates[confs[candidates].argmax()].item()
    mask = result.masks.data[best].detach().cpu().float()
    h, w = result.orig_shape
    if tuple(mask.shape) != (h, w):
        mask = torch.nn.functional.interpolate(
            mask[None, None],
            size=(h, w),
            mode="nearest",
        )[0, 0]
    meta = {
        "target_yolo_cls": int(target_yolo_cls),
        "cls_name": names.get(target_yolo_cls, str(target_yolo_cls)) if isinstance(names, dict) else names[target_yolo_cls],
        "confidence": float(confs[best].item()),
        "mask_area": float((mask > 0.5).float().mean().item()),
        "source": source_item,
        "source_index": int(source_index),
        "checked_frames": len(checked),
    }
    return (mask > 0.5).float(), float(confs[best].item()), int(target_yolo_cls), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../../Datasets/标注片段.mp4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--frame-interval", type=int, default=30)
    parser.add_argument("--yolo", default="")
    parser.add_argument("--target-yolo-cls", type=int, default=11)
    parser.add_argument("--yolo-conf", type=float, default=0.20)
    parser.add_argument("--yolo-iou", type=float, default=0.70)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-search-frames", type=int, default=20)
    parser.add_argument(
        "--prompt-mode",
        choices=("condition", "dense", "dense_condition"),
        default="condition",
    )
    parser.add_argument(
        "--use-mask-as-output",
        action="store_true",
        help="Bypass SAM prompt encoder/decoder for mask inputs. Keep disabled for dense prompt tests.",
    )
    args = parser.parse_args()

    predictor = build_predictor(args.device, use_mask_as_output=args.use_mask_as_output)
    predictor.init_yolo_mask_adapter(target_class_id=12)

    state = predictor.init_state(
        args.video,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        frame_interval=args.frame_interval,
    )
    yolo_meta = None
    if args.yolo:
        mask, confidence, _, yolo_meta = make_yolo_mask(
            yolo_path=args.yolo,
            source=args.video,
            device=args.device,
            target_yolo_cls=args.target_yolo_cls,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            search_frames=args.yolo_search_frames,
        )
        prompt_frame_idx = int(yolo_meta["source_index"])
    else:
        mask = make_center_mask(state["video_height"], state["video_width"])
        confidence = 0.95
        prompt_frame_idx = 0
    add_result = None
    dense_result = None
    if args.prompt_mode in {"condition", "dense_condition"}:
        add_result = predictor.add_new_yolo_mask(
            state,
            frame_idx=prompt_frame_idx,
            obj_id="forceps",
            mask=mask,
            confidence=confidence,
            class_id=12,
            require_usable=False,
        )
    if args.prompt_mode in {"dense", "dense_condition"}:
        dense_frame_idx, dense_obj_ids, dense_masks = predictor.add_new_mask(
            state,
            frame_idx=prompt_frame_idx,
            obj_id="forceps",
            mask=mask,
        )
        dense_result = {
            "frame_idx": int(dense_frame_idx),
            "obj_ids": list(dense_obj_ids),
            "mask_shape": tuple(dense_masks.shape),
            "mask_mean": float(dense_masks.float().sigmoid().mean().item()),
        }

    outputs = []
    for frame_idx, obj_ids, masks in islice(
        predictor.propagate_in_video(
            state,
            start_frame_idx=prompt_frame_idx,
            max_frame_num_to_track=args.max_frames,
        ),
        args.max_frames,
    ):
        outputs.append(
            {
                "frame_idx": int(frame_idx),
                "obj_ids": list(obj_ids),
                "mask_shape": tuple(masks.shape),
                "mask_mean": float(masks.float().sigmoid().mean().item()),
            }
        )

    print(
        {
            "add_result": add_result,
            "dense_result": dense_result,
            "prompt_mode": args.prompt_mode,
            "use_mask_as_output": args.use_mask_as_output,
            "yolo_meta": yolo_meta,
            "num_frames_loaded": state["num_frames"],
            "outputs": outputs,
        }
    )


if __name__ == "__main__":
    main()
