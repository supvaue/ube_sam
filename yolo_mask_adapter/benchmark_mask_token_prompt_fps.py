"""Benchmark YOLO-mask token + ReSurgSAM2 prompt/CSTMamba inference FPS."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from yolo_mask_adapter.mask_token_encoder import MaskTokenEncoder
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val
from yolo_mask_adapter.train_mask_token_prompt_adapter import build_resurgsam2, set_condition_fusion_mode


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def load_item(path: Path, image_size: int) -> dict:
    data = np.load(path, allow_pickle=False)
    image_path = Path(str(data["image_path"]))
    if not image_path.exists() and not image_path.is_absolute():
        image_path = WORKSPACE_ROOT / image_path
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    image_t = (image_t - IMG_MEAN) / IMG_STD
    yolo = torch.from_numpy(data["yolo_mask"].astype(np.float32))[None, None]
    yolo = F.interpolate(yolo, size=(image_size, image_size), mode="nearest")[0]
    return {
        "image": image_t[None],
        "yolo_mask": yolo[None],
        "yolo_conf": torch.tensor([float(data["yolo_conf"])], dtype=torch.float32),
    }


@torch.inference_mode()
def timed_forward(
    model,
    adapter,
    scorer,
    batch: dict,
    device: str,
    prompt_mode: str,
    target_class_id: int,
    disable_obj_score_gating: bool,
    use_no_mem_attention: bool,
) -> tuple[torch.Tensor, dict]:
    timings = {}
    images = batch["image"].to(device, non_blocking=True)
    yolo = batch["yolo_mask"].to(device, non_blocking=True)
    yolo_conf = batch["yolo_conf"].to(device, non_blocking=True)
    class_ids = torch.full((images.shape[0],), target_class_id, dtype=torch.long, device=device)

    cuda_sync(device)
    start = time.perf_counter()
    backbone_out = model.forward_image(images)
    _, current_vision_feats, current_vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    cuda_sync(device)
    timings["image_encoder"] = time.perf_counter() - start

    if len(current_vision_feats) > 1:
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]
    else:
        high_res_features = None

    low_feat = current_vision_feats[-1]
    h, w = feat_sizes[-1]
    image_features = low_feat.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)

    start = time.perf_counter()
    _, geometry = scorer.score(yolo, yolo_conf, class_ids)
    adapter_out = adapter(image_features=image_features, yolo_mask=yolo, geometry=geometry, class_ids=class_ids)
    cuda_sync(device)
    timings["mask_token_encoder"] = time.perf_counter() - start

    use_condition = prompt_mode in {"condition", "dense_condition"}
    use_dense = prompt_mode in {"dense", "dense_condition"}
    if use_condition:
        start = time.perf_counter()
        fusion_image_embeddings, fusion_cls_tokens = model.cross_modal_fusion(
            image_embeddings=current_vision_feats,
            image_pe=current_vision_pos_embeds,
            text_embeddings=adapter_out.mask_emb_sentence,
            feat_sizes=feat_sizes,
            previous_ref_feats_list=[],
            previous_ref_pos_embeds_list=[],
        )
        if use_no_mem_attention:
            no_mem_start = time.perf_counter()
            conditioned_feats = list(current_vision_feats)
            conditioned_feats[-1] = fusion_image_embeddings
            pix_feat = model._prepare_memory_conditioned_features(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=conditioned_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                track_in_reverse=False,
            )
            cuda_sync(device)
            timings["no_mem_attention"] = time.perf_counter() - no_mem_start
        else:
            pix_feat = fusion_image_embeddings.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
        sparse_tokens = torch.cat([fusion_cls_tokens, adapter_out.mask_emb_cls], dim=1)
        cuda_sync(device)
        timings["cstmamba_fusion"] = time.perf_counter() - start
    else:
        pix_feat = image_features
        sparse_tokens = None
        timings["cstmamba_fusion"] = 0.0
        timings["no_mem_attention"] = 0.0

    start = time.perf_counter()
    old_pred_obj_scores = model.pred_obj_scores
    if disable_obj_score_gating:
        model.pred_obj_scores = False
    try:
        _, high_res_masks, _, _, _, _, _ = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=adapter_out.dense_prompt_mask if use_dense else None,
            high_res_features=high_res_features,
            multimask_output=False,
            fusion_cls_tokens=sparse_tokens,
            text_emb_cls=None,
        )
    finally:
        model.pred_obj_scores = old_pred_obj_scores
    cuda_sync(device)
    timings["prompt_encoder_decoder"] = time.perf_counter() - start
    timings["model_total"] = sum(timings.values())
    return high_res_masks, timings


def summarize(times: dict[str, list[float]]) -> dict:
    out = {}
    for key, values in times.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean_ms": float(arr.mean() * 1000.0),
            "p50_ms": float(np.percentile(arr, 50) * 1000.0),
            "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        }
    model_total = np.asarray(times["model_total"], dtype=np.float64)
    e2e_total = np.asarray(times["e2e_total"], dtype=np.float64)
    out["model_fps"] = float(1.0 / model_total.mean())
    out["e2e_fps"] = float(1.0 / e2e_total.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/forceps_mask_cache_full")
    parser.add_argument("--datasets", nargs="+", default=["exp1_cu_full", "exp2_cu_full"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["interleaved", "random", "tail"], default="interleaved")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--prompt-mode", choices=["dense", "condition", "dense_condition"], default="dense_condition")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--target-class-id", type=int, default=12)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--condition-fusion-mode", choices=["twoway", "film_mlp", "hybrid_film_mlp"], default="twoway")
    parser.add_argument("--use-no-mem-attention", action="store_true")
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = load_cache_index(Path(args.cache_dir), set(args.datasets))
    train_paths, val_paths, _ = split_train_val(
        rows,
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
    )
    if args.split == "train":
        paths = train_paths
    elif args.split == "val":
        paths = val_paths
    else:
        paths = train_paths + val_paths
    paths = sorted(paths)[: args.max_items]

    model = build_resurgsam2(args.device)
    set_condition_fusion_mode(model, args.condition_fusion_mode)
    model.eval()
    checkpoint = None
    prior_mixer_depth = 0
    prior_mixer_heads = 4
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        prior_mixer_depth = int(checkpoint.get("prior_mixer_depth", 0))
        prior_mixer_heads = int(checkpoint.get("prior_mixer_heads", 4))
    adapter = MaskTokenEncoder(
        embed_dim=model.hidden_dim,
        class_id=args.target_class_id,
        sentence_tokens=args.sentence_tokens,
        token_mixer_depth=prior_mixer_depth,
        token_mixer_heads=prior_mixer_heads,
    ).to(args.device)
    if checkpoint is not None:
        if "adapter" in checkpoint:
            adapter.load_state_dict(checkpoint["adapter"], strict=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
    adapter.eval()
    scorer = MaskReliabilityScorer(target_class_id=args.target_class_id)

    times = defaultdict(list)
    with torch.inference_mode():
        for idx, path in enumerate(paths):
            e2e_start = time.perf_counter()
            batch = load_item(path, args.image_size)
            _, step_times = timed_forward(
                model,
                adapter,
                scorer,
                batch,
                args.device,
                args.prompt_mode,
                target_class_id=args.target_class_id,
                disable_obj_score_gating=args.disable_obj_score_gating,
                use_no_mem_attention=args.use_no_mem_attention,
            )
            cuda_sync(args.device)
            e2e_total = time.perf_counter() - e2e_start
            if idx >= args.warmup:
                for key, value in step_times.items():
                    times[key].append(value)
                times["e2e_total"].append(e2e_total)

    result = {
        "device": args.device,
        "prompt_mode": args.prompt_mode,
        "image_size": args.image_size,
        "target_class_id": args.target_class_id,
        "checkpoint": args.checkpoint,
        "condition_fusion_mode": args.condition_fusion_mode,
        "use_no_mem_attention": bool(args.use_no_mem_attention),
        "disable_obj_score_gating": bool(args.disable_obj_score_gating),
        "warmup": args.warmup,
        "measured_items": max(0, len(paths) - args.warmup),
        "summary": summarize(times),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
