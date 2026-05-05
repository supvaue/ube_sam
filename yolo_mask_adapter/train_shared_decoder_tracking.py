"""Train the shared SAM mask decoder on sparse GT tracking pairs.

The goal is not to create a second decoder. This script uses the existing
`self.sam_mask_decoder` after memory attention and teaches it to decode
memory-conditioned features while preserving no-memory mask-prompt decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path = [p for p in sys.path if p != str(REPO_ROOT)]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from yolo_mask_adapter.mask_token_encoder import MaskTokenEncoder
from yolo_mask_adapter.reliability import MaskReliabilityScorer
from yolo_mask_adapter.train_forceps_refine_adapter import load_cache_index, split_train_val
from yolo_mask_adapter.train_mask_token_prompt_adapter import (
    IMG_MEAN,
    IMG_STD,
    batch_metrics,
    dice_loss_from_logits,
    limit_paths_balanced,
    mask_loss_from_logits,
)


def resolve_image_path(path_value: str, image_cache_root: Path | None) -> Path:
    image_path = Path(path_value)
    if image_cache_root is not None and not image_path.is_absolute():
        cached = image_cache_root / image_path
        if cached.exists():
            return cached
    if image_path.exists():
        return image_path
    if not image_path.is_absolute():
        candidate = WORKSPACE_ROOT / image_path
        if candidate.exists():
            return candidate
    return image_path


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    return (image_t - IMG_MEAN) / IMG_STD


def load_mask(data, key: str, image_size: int) -> torch.Tensor:
    mask = torch.from_numpy(data[key].astype(np.float32))[None, None]
    return F.interpolate(mask, size=(image_size, image_size), mode="nearest")[0]


def sequence_key_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if "__frame_" in stem:
        return stem.split("__frame_", 1)[0]
    return "__single_sequence__"


class SparseTrackingPairDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        image_cache_root: Path | None = None,
        max_gap: int = 0,
    ) -> None:
        self.image_size = image_size
        self.image_cache_root = image_cache_root
        rows = []
        for path in paths:
            data = np.load(path, allow_pickle=False)
            rows.append(
                {
                    "path": path,
                    "dataset": str(data["dataset"]),
                    "frame_number": int(data["frame_number"]),
                    "file_name": str(data["file_name"]),
                    "sequence_key": sequence_key_from_file_name(str(data["file_name"])),
                }
            )
        self.pairs = []
        for dataset, sequence_key in sorted({(row["dataset"], row["sequence_key"]) for row in rows}):
            ds_rows = sorted(
                [row for row in rows if row["dataset"] == dataset and row["sequence_key"] == sequence_key],
                key=lambda row: (row["frame_number"], row["file_name"]),
            )
            for start, target in zip(ds_rows[:-1], ds_rows[1:]):
                gap = int(target["frame_number"]) - int(start["frame_number"])
                if gap <= 0:
                    continue
                if max_gap and gap > max_gap:
                    continue
                self.pairs.append((start, target))
        if not self.pairs:
            raise FileNotFoundError("No sparse tracking pairs could be built from cache paths")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        start_row, target_row = self.pairs[index]
        start_data = np.load(start_row["path"], allow_pickle=False)
        target_data = np.load(target_row["path"], allow_pickle=False)
        start_image_path = resolve_image_path(str(start_data["image_path"]), self.image_cache_root)
        target_image_path = resolve_image_path(str(target_data["image_path"]), self.image_cache_root)
        return {
            "start_image": load_image_tensor(start_image_path, self.image_size),
            "target_image": load_image_tensor(target_image_path, self.image_size),
            "start_gt": load_mask(start_data, "gt_mask", self.image_size),
            "target_gt": load_mask(target_data, "gt_mask", self.image_size),
            "start_yolo": load_mask(start_data, "yolo_mask", self.image_size),
            "target_yolo": load_mask(target_data, "yolo_mask", self.image_size),
            "gap": torch.tensor(int(target_row["frame_number"]) - int(start_row["frame_number"]), dtype=torch.long),
            "dataset": start_row["dataset"],
            "start_file": start_row["file_name"],
            "target_file": target_row["file_name"],
        }


def build_model(args):
    model = build_sam2_video_predictor(
        config_file=args.config_file,
        ckpt_path=args.base_ckpt_path,
        device=args.device,
        strict_loading=False,
        apply_long_term_memory=args.apply_long_term_memory,
        hydra_overrides_extra=[
            "++scratch.use_sp_bimamba=true",
            "++scratch.use_dwconv=true",
            "++model.use_mask_input_as_output_without_sam=false",
        ],
    )
    if args.condition_fusion_mode != "twoway":
        if not hasattr(model, "cross_modal_fusion") or not hasattr(model.cross_modal_fusion, "set_condition_fusion_mode"):
            raise AttributeError("cross_modal_fusion does not support condition fusion modes")
        model.cross_modal_fusion.set_condition_fusion_mode(args.condition_fusion_mode)
        print(json.dumps({"condition_fusion_mode": args.condition_fusion_mode}, ensure_ascii=False), flush=True)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        if "model" in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            print(
                json.dumps(
                    {
                        "resume_checkpoint": args.resume_checkpoint,
                        "loaded_model_tensors": len(checkpoint["model"]),
                        "missing_keys": len(missing),
                        "unexpected_keys": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return model


def build_adapter_from_checkpoint(args) -> MaskTokenEncoder | None:
    if not args.mask_token_checkpoint:
        return None
    checkpoint = torch.load(args.mask_token_checkpoint, map_location="cpu")
    prior_mixer_depth = int(checkpoint.get("prior_mixer_depth", 0))
    prior_mixer_heads = int(checkpoint.get("prior_mixer_heads", 4))
    adapter = MaskTokenEncoder(
        embed_dim=args.hidden_dim,
        num_classes=args.num_classes,
        class_id=args.target_class_id,
        sentence_tokens=args.sentence_tokens,
        token_mixer_depth=prior_mixer_depth,
        token_mixer_heads=prior_mixer_heads,
    )
    missing, unexpected = adapter.load_state_dict(checkpoint["adapter"], strict=False)
    print(
        json.dumps(
            {
                "mask_token_checkpoint": args.mask_token_checkpoint,
                "prior_mixer_depth": prior_mixer_depth,
                "prior_mixer_heads": prior_mixer_heads,
                "adapter_missing_keys": len(missing),
                "adapter_unexpected_keys": len(unexpected),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter.to(args.device)


def set_trainable(model, train_parts: set[str]) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if "prompt_encoder" in train_parts:
        for param in model.sam_prompt_encoder.parameters():
            param.requires_grad_(True)
    if "cstmamba" in train_parts:
        for param in model.cross_modal_fusion.parameters():
            param.requires_grad_(True)
    if "mask_decoder" in train_parts:
        for param in model.sam_mask_decoder.parameters():
            param.requires_grad_(True)
    if "memory_attention" in train_parts:
        for param in model.memory_attention.parameters():
            param.requires_grad_(True)
    if "memory_encoder" in train_parts:
        for param in model.memory_encoder.parameters():
            param.requires_grad_(True)
    if "obj_ptr" in train_parts:
        for module_name in ["obj_ptr_proj", "obj_ptr_tpos_proj"]:
            module = getattr(model, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(True)


def trainable_model_state(model) -> dict:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


@torch.no_grad()
def extract_features(model, images: torch.Tensor):
    backbone_out = model.forward_image(images)
    _, current_vision_feats, current_vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    return current_vision_feats, current_vision_pos_embeds, feat_sizes


def high_res_features_from(current_vision_feats, feat_sizes):
    if len(current_vision_feats) <= 1:
        return None
    return [
        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s).detach()
        for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
    ]


def low_pix_feat(model, current_vision_feats, feat_sizes):
    low = current_vision_feats[-1]
    h, w = feat_sizes[-1]
    return low.permute(1, 2, 0).reshape(low.shape[1], model.hidden_dim, h, w)


def mask_to_logits(mask: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    return mask.float() * scale - scale / 2.0


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


def make_cond_memory(model, start_feats, start_pos, feat_sizes, start_mask, start_high_res, args) -> dict:
    start_pix = low_pix_feat(model, start_feats, feat_sizes)
    if args.start_memory_source == "decoder":
        with torch.no_grad():
            _, high_res_masks, ious, low_res_masks, _best_high, obj_ptr, object_score_logits = model._forward_sam_heads(
                backbone_features=start_pix,
                point_inputs=None,
                mask_inputs=start_mask,
                high_res_features=start_high_res,
                multimask_output=False,
            )
    else:
        with torch.no_grad():
            _, _high_res_masks, ious, low_res_masks, _best_high, obj_ptr, _object_score_logits = model._forward_sam_heads(
                backbone_features=start_pix,
                point_inputs=None,
                mask_inputs=start_mask,
                high_res_features=start_high_res,
                multimask_output=False,
            )
            high_res_masks = mask_to_logits(start_mask)
            object_score_logits = start_mask.new_ones((start_mask.shape[0], 1)) * 10.0

    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
        current_vision_feats=start_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=high_res_masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=True,
    )
    return detach_output_dict(
        {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "ious": ious,
        }
    )


def forward_sam_heads_with_prior(
    model,
    backbone_features: torch.Tensor,
    mask_inputs: torch.Tensor | None,
    high_res_features,
    fusion_cls_tokens: torch.Tensor | None,
    multimask_output: bool = False,
):
    """SAM heads path with fusion_cls_tokens, even for predictor classes whose wrapper omits it."""
    bsz = backbone_features.size(0)
    device = backbone_features.device
    sam_point_coords = torch.zeros(bsz, 1, 2, device=device)
    sam_point_labels = -torch.ones(bsz, 1, dtype=torch.int32, device=device)
    if mask_inputs is not None:
        if mask_inputs.shape[-2:] != model.sam_prompt_encoder.mask_input_size:
            sam_mask_prompt = F.interpolate(
                mask_inputs.float(),
                size=model.sam_prompt_encoder.mask_input_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        else:
            sam_mask_prompt = mask_inputs
    else:
        sam_mask_prompt = None
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(sam_point_coords, sam_point_labels),
        boxes=None,
        masks=sam_mask_prompt,
        fusion_cls_tokens=fusion_cls_tokens,
        text_emb_cls=None,
    )
    low_res_multimasks, ious, sam_output_tokens, object_score_logits = model.sam_mask_decoder(
        image_embeddings=backbone_features,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    if model.pred_obj_scores:
        low_res_multimasks = torch.where(
            (object_score_logits > 0)[:, None, None],
            low_res_multimasks,
            torch.full_like(low_res_multimasks, -1024.0),
        )
    low_res_multimasks = low_res_multimasks.float()
    high_res_multimasks = F.interpolate(
        low_res_multimasks,
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )
    if multimask_output:
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(bsz, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        sam_output_token = sam_output_tokens[batch_inds, best_iou_inds] if sam_output_tokens.size(1) > 1 else sam_output_tokens[:, 0]
    else:
        low_res_masks = low_res_multimasks
        high_res_masks = high_res_multimasks
        sam_output_token = sam_output_tokens[:, 0]
    obj_ptr = model.obj_ptr_proj(sam_output_token)
    if model.pred_obj_scores:
        lambda_is_obj_appearing = (object_score_logits > 0).float()
        if model.soft_no_obj_ptr:
            lambda_is_obj_appearing = object_score_logits.sigmoid()
        if model.fixed_no_obj_ptr:
            obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * model.no_obj_ptr
    return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits


def prior_conditioned_outputs(model, adapter, scorer, images, yolo, yolo_conf, class_ids, feats, pos, sizes, high_res):
    low_feat = feats[-1]
    h, w = sizes[-1]
    image_features = low_feat.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
    _reliability, geometry = scorer.score(yolo, yolo_conf, class_ids)
    adapter_out = adapter(image_features=image_features, yolo_mask=yolo, geometry=geometry, class_ids=class_ids)
    fusion_image_embeddings, fusion_cls_tokens = model.cross_modal_fusion(
        image_embeddings=feats,
        image_pe=pos,
        text_embeddings=adapter_out.mask_emb_sentence,
        feat_sizes=sizes,
        previous_ref_feats_list=[],
        previous_ref_pos_embeds_list=[],
    )
    pix_feat = fusion_image_embeddings.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
    sparse_tokens = torch.cat([fusion_cls_tokens, adapter_out.mask_emb_cls], dim=1)
    outputs = forward_sam_heads_with_prior(
        model,
        backbone_features=pix_feat,
        mask_inputs=adapter_out.dense_prompt_mask,
        high_res_features=high_res,
        fusion_cls_tokens=sparse_tokens,
        multimask_output=False,
    )
    return outputs, sparse_tokens


def forward_tracking_pair(model, adapter, scorer, batch: dict, args) -> tuple[torch.Tensor, torch.Tensor]:
    device = args.device
    start_images = batch["start_image"].to(device)
    target_images = batch["target_image"].to(device)
    start_mask = batch[args.start_mask_key].to(device)
    target_gt = batch["target_gt"].to(device)
    target_yolo = batch["target_yolo"].to(device)
    target_conf = torch.ones(target_yolo.shape[0], device=device)
    class_ids = torch.full((target_yolo.shape[0],), args.target_class_id, device=device, dtype=torch.long)

    start_feats, start_pos, start_sizes = extract_features(model, start_images)
    target_feats, target_pos, target_sizes = extract_features(model, target_images)
    start_high_res = high_res_features_from(start_feats, start_sizes)
    target_high_res = high_res_features_from(target_feats, target_sizes)
    cond_out = make_cond_memory(model, start_feats, start_pos, start_sizes, start_mask, start_high_res, args)

    # Use a local timeline for temporal encodings. For batch training, all pairs
    # share the same target frame index to keep output_dict shape simple.
    frame_idx = int(batch["gap"].max().item()) if args.use_real_gap else 1
    output_dict = {
        "cond_frame_outputs": {0: cond_out},
        "non_cond_frame_outputs": {},
        "long_mem_candidate": [],
        "long_mem": [],
    }
    old_disable_non_cond = getattr(model, "disable_non_cond_memory", False)
    model.disable_non_cond_memory = True
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
        if adapter is not None and args.use_prior_conditioned_key:
            key_outputs, prior_sparse_tokens = prior_conditioned_outputs(
                model=model,
                adapter=adapter,
                scorer=scorer,
                images=target_images,
                yolo=target_yolo,
                yolo_conf=target_conf,
                class_ids=class_ids,
                feats=target_feats,
                pos=target_pos,
                sizes=target_sizes,
                high_res=target_high_res,
            )
            _key_low_multi, key_high_res, _key_ious, _key_low, _key_high, _key_obj, _key_score = key_outputs
            tracking_cls = prior_sparse_tokens if args.tracking_prompt_source == "target_prior_cls" else None
        else:
            _key_low_multi, key_high_res, _key_ious, _key_low, _key_high, _key_obj, _key_score = forward_sam_heads_with_prior(
                model,
                backbone_features=low_pix_feat(model, target_feats, target_sizes),
                mask_inputs=target_yolo,
                high_res_features=target_high_res,
                fusion_cls_tokens=None,
                multimask_output=False,
            )
            tracking_cls = None
        _track_low_multi, track_high_res, _track_ious, _track_low, _track_high, _track_obj, _track_score = forward_sam_heads_with_prior(
            model,
            backbone_features=pix_feat_with_mem,
            mask_inputs=None,
            high_res_features=target_high_res,
            fusion_cls_tokens=tracking_cls,
            multimask_output=False,
        )
    finally:
        model.pred_obj_scores = old_pred_obj_scores
    if track_high_res.shape[-2:] != target_gt.shape[-2:]:
        track_high_res = F.interpolate(track_high_res, size=target_gt.shape[-2:], mode="bilinear", align_corners=False)
    if key_high_res.shape[-2:] != target_gt.shape[-2:]:
        key_high_res = F.interpolate(key_high_res, size=target_gt.shape[-2:], mode="bilinear", align_corners=False)
    return track_high_res, key_high_res


def run_epoch(model, adapter, scorer, loader, optimizer, args, train: bool) -> dict:
    model.train(train)
    total = {"loss": 0.0, "track_loss": 0.0, "key_loss": 0.0, "track_iou": 0.0, "key_iou": 0.0, "yolo_iou": 0.0}
    count = 0
    for batch in loader:
        target_gt = batch["target_gt"].to(args.device)
        target_yolo = batch["target_yolo"].to(args.device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        track_logits, key_logits = forward_tracking_pair(model, adapter, scorer, batch, args)
        track_loss = mask_loss_from_logits(track_logits, target_gt)
        key_loss = mask_loss_from_logits(key_logits, target_gt)
        loss = args.memory_loss_weight * track_loss + args.key_loss_weight * key_loss
        if train:
            loss.backward()
            optimizer.step()
        bs = target_gt.shape[0]
        track_metrics = batch_metrics(track_logits.detach(), target_gt)
        key_metrics = batch_metrics(key_logits.detach(), target_gt)
        yolo_metrics = batch_metrics(torch.logit(target_yolo.clamp(1e-4, 1 - 1e-4)), target_gt)
        total["loss"] += float(loss.item()) * bs
        total["track_loss"] += float(track_loss.item()) * bs
        total["key_loss"] += float(key_loss.item()) * bs
        total["track_iou"] += track_metrics["iou"] * bs
        total["key_iou"] += key_metrics["iou"] * bs
        total["yolo_iou"] += yolo_metrics["iou"] * bs
        count += bs
    return {key: value / max(count, 1) for key, value in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/ligamentum_flavum_mask_cache_balanced_600")
    parser.add_argument("--train-datasets", nargs="+", default=["exp1_cu_full", "exp2_cu_full"])
    parser.add_argument("--image-cache-root", default="")
    parser.add_argument("--output-dir", default="yolo_mask_adapter/results/shared_decoder_tracking_pair_smoke")
    parser.add_argument("--config-file", default="configs/sam2.1/sam2.1_hiera_s_rvos.yaml")
    parser.add_argument("--base-ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--mask-token-checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["interleaved", "random", "tail"], default="interleaved")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--train-parts", nargs="+", default=["prompt_encoder", "mask_decoder"], choices=["prompt_encoder", "mask_decoder", "memory_attention", "memory_encoder", "obj_ptr", "cstmamba"])
    parser.add_argument("--target-class-id", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument("--start-mask-key", choices=["start_gt", "start_yolo"], default="start_gt")
    parser.add_argument("--start-memory-source", choices=["gt_logits", "decoder"], default="gt_logits")
    parser.add_argument("--use-prior-conditioned-key", action="store_true")
    parser.add_argument("--tracking-prompt-source", choices=["none", "target_prior_cls"], default="none")
    parser.add_argument("--key-loss-weight", type=float, default=0.25)
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--condition-fusion-mode",
        choices=["twoway", "film_mlp", "hybrid_film_mlp"],
        default="twoway",
        help=(
            "How cross_modal_fusion writes mask/prior token information back into image tokens. "
            "twoway keeps the original attention path; film_mlp replaces token->image attention "
            "with a condition-conditioned MLP; hybrid_film_mlp uses both."
        ),
    )
    parser.add_argument("--disable-obj-score-gating", action="store_true")
    parser.add_argument("--apply-long-term-memory", action="store_true")
    parser.add_argument("--use-real-gap", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    rows = load_cache_index(cache_dir, set(args.train_datasets))
    train_paths, val_paths, split_summary = split_train_val(rows, args.val_fraction, args.split_mode, args.split_seed)
    if args.max_train_items:
        train_paths = limit_paths_balanced(train_paths, args.max_train_items)
    if args.max_val_items:
        val_paths = limit_paths_balanced(val_paths, args.max_val_items)
    image_cache_root = Path(args.image_cache_root) if args.image_cache_root else None
    train_ds = SparseTrackingPairDataset(train_paths, args.image_size, image_cache_root=image_cache_root, max_gap=args.max_gap)
    val_ds = SparseTrackingPairDataset(val_paths, args.image_size, image_cache_root=image_cache_root, max_gap=args.max_gap)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    print(f"[stage] build model from {args.base_ckpt_path}", flush=True)
    model = build_model(args)
    adapter = build_adapter_from_checkpoint(args)
    scorer = MaskReliabilityScorer(target_class_id=args.target_class_id)
    set_trainable(model, set(args.train_parts))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summary.update(
        {
            "train_pairs": len(train_ds),
            "val_pairs": len(val_ds),
            "train_parts": args.train_parts,
            "start_mask_key": args.start_mask_key,
            "start_memory_source": args.start_memory_source,
            "use_prior_conditioned_key": bool(args.use_prior_conditioned_key),
            "tracking_prompt_source": args.tracking_prompt_source,
            "mask_token_checkpoint": args.mask_token_checkpoint,
            "key_loss_weight": args.key_loss_weight,
            "memory_loss_weight": args.memory_loss_weight,
            "condition_fusion_mode": args.condition_fusion_mode,
            "resume_checkpoint": args.resume_checkpoint,
        }
    )
    (out_dir / "split.json").write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, adapter, scorer, train_loader, optimizer, args, train=True)
        val_metrics = run_epoch(model, adapter, scorer, val_loader, optimizer, args, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["track_iou"] > best_iou:
            best_iou = val_metrics["track_iou"]
            torch.save(
                {
                    "model": trainable_model_state(model),
                    "epoch": epoch,
                    "val": val_metrics,
                    "train_parts": args.train_parts,
                    "start_mask_key": args.start_mask_key,
                    "start_memory_source": args.start_memory_source,
                    "use_prior_conditioned_key": bool(args.use_prior_conditioned_key),
                    "tracking_prompt_source": args.tracking_prompt_source,
                    "mask_token_checkpoint": args.mask_token_checkpoint,
                    "key_loss_weight": args.key_loss_weight,
                    "memory_loss_weight": args.memory_loss_weight,
                    "condition_fusion_mode": args.condition_fusion_mode,
                },
                out_dir / "best.pt",
            )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "model": trainable_model_state(model),
            "epoch": args.epochs,
            "val": history[-1]["val"],
            "train_parts": args.train_parts,
            "memory_loss_weight": args.memory_loss_weight,
            "key_loss_weight": args.key_loss_weight,
            "condition_fusion_mode": args.condition_fusion_mode,
        },
        out_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
