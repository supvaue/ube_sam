"""Train YOLO-mask condition tokens through ReSurgSAM2 prompt/CSTMamba heads."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class PromptForwardResult:
    logits: torch.Tensor
    adapter_out: object
    geometry: torch.Tensor
    ious: torch.Tensor | None = None
    object_score_logits: torch.Tensor | None = None


class ForcepsPromptDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        image_size: int = 512,
        class_id: int = 12,
        image_cache_root: Path | None = None,
    ):
        self.paths = sorted(paths)
        self.image_size = image_size
        self.class_id = class_id
        self.image_cache_root = image_cache_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        data = np.load(self.paths[index], allow_pickle=False)
        image_path = Path(str(data["image_path"])) if "image_path" in data.files else None
        if image_path is None:
            raise KeyError(f"{self.paths[index]} does not contain image_path; use mask cache, not feature cache")
        if self.image_cache_root is not None and not image_path.is_absolute():
            cached_image_path = self.image_cache_root / image_path
            if cached_image_path.exists():
                image_path = cached_image_path
        if not image_path.exists() and not image_path.is_absolute():
            image_path = WORKSPACE_ROOT / image_path

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image_t = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        image_t = (image_t - IMG_MEAN) / IMG_STD

        gt = torch.from_numpy(data["gt_mask"].astype(np.float32))[None, None]
        yolo = torch.from_numpy(data["yolo_mask"].astype(np.float32))[None, None]
        gt = F.interpolate(gt, size=(self.image_size, self.image_size), mode="nearest")[0]
        yolo = F.interpolate(yolo, size=(self.image_size, self.image_size), mode="nearest")[0]
        return {
            "image": image_t,
            "gt_mask": gt,
            "yolo_mask": yolo,
            "yolo_conf": torch.tensor(float(data["yolo_conf"]), dtype=torch.float32),
            "class_id": torch.tensor(self.class_id, dtype=torch.long),
            "dataset": str(data["dataset"]),
            "file_name": str(data["file_name"]),
        }


def limit_paths_balanced(paths: list[Path], max_items: int) -> list[Path]:
    if not max_items or len(paths) <= max_items:
        return paths
    rows = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        rows.append(
            {
                "path": path,
                "dataset": str(data["dataset"]),
                "frame_number": int(data["frame_number"]),
                "file_name": str(data["file_name"]),
            }
        )

    groups = {
        dataset: sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: (row["frame_number"], row["file_name"]),
        )
        for dataset in sorted({row["dataset"] for row in rows})
    }
    per_dataset = max_items // len(groups)
    remainder = max_items % len(groups)
    selected = []
    for idx, (dataset, dataset_rows) in enumerate(groups.items()):
        quota = min(len(dataset_rows), per_dataset + (idx < remainder))
        if quota >= len(dataset_rows):
            selected.extend(dataset_rows)
            continue
        indices = np.linspace(0, len(dataset_rows) - 1, quota, dtype=int).tolist()
        selected.extend(dataset_rows[index] for index in indices)
    selected.sort(key=lambda row: (row["dataset"], row["frame_number"], row["file_name"]))
    return [row["path"] for row in selected]


def build_resurgsam2(device: str, ckpt_path: str = "checkpoints/sam2.1_hiera_s_ref17.pth"):
    model = build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_s_rvos.yaml",
        ckpt_path=ckpt_path,
        device=device,
        strict_loading=False,
        apply_long_term_memory=True,
        hydra_overrides_extra=[
            "++scratch.use_sp_bimamba=true",
            "++scratch.use_dwconv=true",
            "++model.use_mask_input_as_output_without_sam=false",
        ],
    )
    model.eval()
    return model


def set_condition_fusion_mode(model, mode: str) -> None:
    if mode == "twoway":
        return
    if not hasattr(model, "cross_modal_fusion"):
        raise AttributeError("Model has no cross_modal_fusion module")
    if hasattr(model.cross_modal_fusion, "set_condition_fusion_mode"):
        model.cross_modal_fusion.set_condition_fusion_mode(mode)
    else:
        raise AttributeError("cross_modal_fusion does not support condition fusion modes")
    print(json.dumps({"condition_fusion_mode": mode}, ensure_ascii=False), flush=True)


def set_trainable(model, train_parts: set[str]) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if "cstmamba" in train_parts:
        for param in model.cross_modal_fusion.parameters():
            param.requires_grad_(True)
    if "prompt_encoder" in train_parts:
        for param in model.sam_prompt_encoder.parameters():
            param.requires_grad_(True)
    if "mask_decoder" in train_parts:
        for param in model.sam_mask_decoder.parameters():
            param.requires_grad_(True)


def reset_module_parameters(module: torch.nn.Module) -> None:
    """Best-effort recursive parameter reset for modules selected for scratch training."""
    for child in module.children():
        reset_module_parameters(child)
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
    if isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, std=0.02)


def reinitialize_selected_modules(model, train_parts: set[str]) -> list[str]:
    """Reinitialize trainable heads so experiments can train from scratch."""
    reset_names = []
    if "cstmamba" in train_parts:
        reset_module_parameters(model.cross_modal_fusion)
        if hasattr(model.cross_modal_fusion, "_init_weights"):
            model.cross_modal_fusion.apply(model.cross_modal_fusion._init_weights)
        if hasattr(model.cross_modal_fusion, "_init_condition_fusion_identity"):
            model.cross_modal_fusion._init_condition_fusion_identity()
        reset_names.append("cross_modal_fusion")
    if "prompt_encoder" in train_parts:
        reset_module_parameters(model.sam_prompt_encoder)
        reset_names.append("sam_prompt_encoder")
    if "mask_decoder" in train_parts:
        reset_module_parameters(model.sam_mask_decoder)
        reset_names.append("sam_mask_decoder")
    return reset_names


def trainable_model_state(model) -> dict:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def mask_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_from_logits(logits, targets)


def per_sample_mask_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").flatten(1).mean(dim=1)
    probs = logits.sigmoid()
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    pred = logits.sigmoid() > 0.5
    gt = targets > 0.5
    inter = (pred & gt).float().sum(dim=(1, 2, 3))
    union = (pred | gt).float().sum(dim=(1, 2, 3))
    pred_area = pred.float().sum(dim=(1, 2, 3))
    gt_area = gt.float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    dice = torch.where(
        pred_area + gt_area > 0,
        2 * inter / (pred_area + gt_area).clamp_min(1.0),
        torch.ones_like(union),
    )
    return {"iou": float(iou.mean().item()), "dice": float(dice.mean().item())}


def per_sample_metrics(logits: torch.Tensor, targets: torch.Tensor) -> list[dict]:
    pred = logits.sigmoid() > 0.5
    gt = targets > 0.5
    inter = (pred & gt).float().sum(dim=(1, 2, 3))
    union = (pred | gt).float().sum(dim=(1, 2, 3))
    pred_area = pred.float().sum(dim=(1, 2, 3))
    gt_area = gt.float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    dice = torch.where(
        pred_area + gt_area > 0,
        2 * inter / (pred_area + gt_area).clamp_min(1.0),
        torch.ones_like(union),
    )
    return [
        {"iou": float(iou[index].item()), "dice": float(dice[index].item())}
        for index in range(logits.shape[0])
    ]


def mask_iou_tensor(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
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


def quality_bucket(iou: torch.Tensor) -> torch.Tensor:
    bucket = torch.zeros_like(iou, dtype=torch.long)
    bucket = torch.where(iou >= 0.50, torch.ones_like(bucket), bucket)
    bucket = torch.where(iou >= 0.70, torch.full_like(bucket, 2), bucket)
    return bucket


class MaskPriorAuxHeads(nn.Module):
    """Training-only probes that force mask-prior tokens to retain quality cues."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.bad_mask = nn.Linear(feature_dim, 1)
        self.good_mask = nn.Linear(feature_dim, 1)
        self.quality_bucket = nn.Linear(feature_dim, 3)
        self.yolo_iou = nn.Linear(feature_dim, 1)
        self.boundary_error = nn.Linear(feature_dim, 1)
        self.area_error = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "bad_mask": self.bad_mask(features).squeeze(1),
            "good_mask": self.good_mask(features).squeeze(1),
            "quality_bucket": self.quality_bucket(features),
            "yolo_iou": self.yolo_iou(features).squeeze(1),
            "boundary_error": self.boundary_error(features).squeeze(1),
            "area_error": self.area_error(features).squeeze(1),
        }


def mask_prior_feature(adapter_out, geometry: torch.Tensor, source: str) -> torch.Tensor:
    sentence = adapter_out.mask_emb_sentence
    cls = adapter_out.mask_emb_cls.squeeze(1)
    if source == "sentence_flat":
        return sentence.flatten(1)
    if source == "sentence_mean":
        return sentence.mean(dim=1)
    if source == "cls":
        return cls
    if source == "sentence_mean_cls_geometry":
        return torch.cat([sentence.mean(dim=1), cls, geometry.to(sentence.dtype)], dim=1)
    raise ValueError(f"Unknown aux feature source: {source}")


def mask_prior_feature_dim(hidden_dim: int, sentence_tokens: int, source: str) -> int:
    if source == "sentence_flat":
        return hidden_dim * sentence_tokens
    if source == "sentence_mean":
        return hidden_dim
    if source == "cls":
        return hidden_dim
    if source == "sentence_mean_cls_geometry":
        return hidden_dim * 2 + 10
    raise ValueError(f"Unknown aux feature source: {source}")


def mask_prior_aux_loss(
    aux_heads: MaskPriorAuxHeads,
    adapter_out,
    geometry: torch.Tensor,
    yolo: torch.Tensor,
    gt: torch.Tensor,
    feature_source: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = mask_prior_feature(adapter_out, geometry, feature_source)
    pred = aux_heads(features)
    with torch.no_grad():
        yolo_iou = mask_iou_tensor(yolo, gt)
        boundary_error = 1.0 - mask_iou_tensor(boundary_ring(yolo), boundary_ring(gt))
        yolo_area = (yolo > 0.5).float().flatten(1).mean(dim=1)
        gt_area = (gt > 0.5).float().flatten(1).mean(dim=1)
        area_error = (yolo_area - gt_area).abs() / torch.maximum(gt_area, torch.full_like(gt_area, 1e-6))
        area_error = area_error.clamp(0.0, 10.0)
        bad = (yolo_iou < 0.50).float()
        good = (yolo_iou >= 0.70).float()
        bucket = quality_bucket(yolo_iou)

    bad_loss = F.binary_cross_entropy_with_logits(pred["bad_mask"], bad)
    good_loss = F.binary_cross_entropy_with_logits(pred["good_mask"], good)
    bucket_loss = F.cross_entropy(pred["quality_bucket"], bucket)
    iou_loss = F.smooth_l1_loss(pred["yolo_iou"].sigmoid(), yolo_iou)
    boundary_loss = F.smooth_l1_loss(pred["boundary_error"].sigmoid(), boundary_error.clamp(0.0, 1.0))
    area_loss = F.smooth_l1_loss(pred["area_error"].sigmoid(), (area_error / 10.0).clamp(0.0, 1.0))
    loss = bad_loss + good_loss + bucket_loss + iou_loss + boundary_loss + area_loss
    metrics = {
        "aux_bad_loss": float(bad_loss.detach().item()),
        "aux_good_loss": float(good_loss.detach().item()),
        "aux_bucket_loss": float(bucket_loss.detach().item()),
        "aux_iou_loss": float(iou_loss.detach().item()),
        "aux_boundary_loss": float(boundary_loss.detach().item()),
        "aux_area_loss": float(area_loss.detach().item()),
    }
    return loss, metrics


def summarize_eval_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    out = {"count": len(rows)}
    for key in ["model_iou", "model_dice", "yolo_iou", "yolo_dice", "delta_iou"]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(values.mean())
        out[f"p50_{key}"] = float(np.quantile(values, 0.50))
    out["improved_count"] = int(sum(row["delta_iou"] > 0 for row in rows))
    return out


@torch.no_grad()
def run_eval_details(
    model,
    adapter,
    scorer,
    loader,
    device: str,
    prompt_mode: str,
    disable_obj_score_gating: bool,
    use_no_mem_attention: bool,
) -> dict:
    adapter.eval()
    model.eval()
    rows = []
    for batch in loader:
        gt = batch["gt_mask"].to(device)
        yolo = batch["yolo_mask"].to(device)
        logits = forward_prompt_model(
            model,
            adapter,
            scorer,
            batch,
            device,
            prompt_mode,
            disable_obj_score_gating=disable_obj_score_gating,
            use_no_mem_attention=use_no_mem_attention,
        )
        yolo_logits = torch.logit(yolo.clamp(1e-4, 1 - 1e-4))
        model_metrics = per_sample_metrics(logits.detach(), gt)
        yolo_metrics = per_sample_metrics(yolo_logits, gt)
        for index, (model_row, yolo_row) in enumerate(zip(model_metrics, yolo_metrics)):
            rows.append(
                {
                    "dataset": batch["dataset"][index],
                    "file_name": batch["file_name"][index],
                    "model_iou": model_row["iou"],
                    "model_dice": model_row["dice"],
                    "yolo_iou": yolo_row["iou"],
                    "yolo_dice": yolo_row["dice"],
                    "delta_iou": model_row["iou"] - yolo_row["iou"],
                }
            )

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_dataset[dataset] = summarize_eval_rows([row for row in rows if row["dataset"] == dataset])
    bins = {
        "yolo_iou_lt_050": [row for row in rows if row["yolo_iou"] < 0.50],
        "yolo_iou_050_070": [row for row in rows if 0.50 <= row["yolo_iou"] < 0.70],
        "yolo_iou_gte_070": [row for row in rows if row["yolo_iou"] >= 0.70],
    }
    return {
        "overall": summarize_eval_rows(rows),
        "by_dataset": by_dataset,
        "by_yolo_quality": {name: summarize_eval_rows(bin_rows) for name, bin_rows in bins.items()},
        "rows": rows,
        "worst_delta": sorted(rows, key=lambda row: row["delta_iou"])[:20],
        "best_delta": sorted(rows, key=lambda row: row["delta_iou"], reverse=True)[:20],
    }


def detach_feature_lists(features: list[torch.Tensor]) -> list[torch.Tensor]:
    return [feat.detach() for feat in features]


def forward_prompt_model(
    model,
    adapter: MaskTokenEncoder,
    scorer: MaskReliabilityScorer,
    batch: dict,
    device: str,
    prompt_mode: str,
    disable_obj_score_gating: bool = False,
    use_no_mem_attention: bool = False,
    return_details: bool = False,
) -> torch.Tensor:
    images = batch["image"].to(device)
    gt = batch["gt_mask"].to(device)
    yolo = batch["yolo_mask"].to(device)
    yolo_conf = batch["yolo_conf"].to(device)
    class_ids = batch["class_id"].to(device)

    with torch.no_grad():
        backbone_out = model.forward_image(images)
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    current_vision_feats = detach_feature_lists(current_vision_feats)
    current_vision_pos_embeds = detach_feature_lists(current_vision_pos_embeds)

    if len(current_vision_feats) > 1:
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s).detach()
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]
    else:
        high_res_features = None

    low_feat = current_vision_feats[-1]
    h, w = feat_sizes[-1]
    image_features = low_feat.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
    _, geometry = scorer.score(yolo, yolo_conf, class_ids)
    adapter_out = adapter(
        image_features=image_features,
        yolo_mask=yolo,
        geometry=geometry,
        class_ids=class_ids,
    )

    use_condition = prompt_mode in {"condition", "dense_condition"}
    use_dense = prompt_mode in {"dense", "dense_condition"}
    if use_condition:
        fusion_image_embeddings, fusion_cls_tokens = model.cross_modal_fusion(
            image_embeddings=current_vision_feats,
            image_pe=current_vision_pos_embeds,
            text_embeddings=adapter_out.mask_emb_sentence,
            feat_sizes=feat_sizes,
            previous_ref_feats_list=[],
            previous_ref_pos_embeds_list=[],
        )
        if use_no_mem_attention:
            no_mem_vision_feats = list(current_vision_feats)
            no_mem_vision_feats[-1] = fusion_image_embeddings
            pix_feat = model._prepare_memory_conditioned_features(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=no_mem_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                track_in_reverse=False,
            )
        else:
            pix_feat = fusion_image_embeddings.permute(1, 2, 0).reshape(images.shape[0], model.hidden_dim, h, w)
        sparse_tokens = torch.cat([fusion_cls_tokens, adapter_out.mask_emb_cls], dim=1)
    else:
        if use_no_mem_attention:
            pix_feat = model._prepare_memory_conditioned_features(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                track_in_reverse=False,
            )
        else:
            pix_feat = image_features
        sparse_tokens = None

    mask_inputs = adapter_out.dense_prompt_mask if use_dense else None
    old_pred_obj_scores = model.pred_obj_scores
    if disable_obj_score_gating:
        model.pred_obj_scores = False
    try:
        _, high_res_masks, ious, _, _, _, object_score_logits = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=False,
            fusion_cls_tokens=sparse_tokens,
            text_emb_cls=None,
        )
    finally:
        model.pred_obj_scores = old_pred_obj_scores
    if high_res_masks.shape[-2:] != gt.shape[-2:]:
        high_res_masks = F.interpolate(high_res_masks, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    if return_details:
        return PromptForwardResult(
            logits=high_res_masks,
            adapter_out=adapter_out,
            geometry=geometry,
            ious=ious,
            object_score_logits=object_score_logits,
        )
    return high_res_masks


def run_epoch(
    model,
    adapter,
    scorer,
    loader,
    optimizer,
    aux_heads,
    device: str,
    prompt_mode: str,
    train: bool,
    disable_obj_score_gating: bool,
    use_no_mem_attention: bool,
    yolo_fidelity_weight: float,
    adaptive_fidelity: bool,
    fidelity_min_weight: float,
    fidelity_max_weight: float,
    aux_feature_loss_weight: float,
    aux_feature_source: str,
) -> dict:
    adapter.train(train)
    if aux_heads is not None:
        aux_heads.train(train)
    if any(param.requires_grad for param in model.parameters()):
        model.train(train)
    else:
        model.eval()
    total = {
        "loss": 0.0,
        "gt_loss": 0.0,
        "fidelity_loss": 0.0,
        "aux_feature_loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "yolo_iou": 0.0,
        "fidelity_weight": 0.0,
        "aux_bad_loss": 0.0,
        "aux_good_loss": 0.0,
        "aux_bucket_loss": 0.0,
        "aux_iou_loss": 0.0,
        "aux_boundary_loss": 0.0,
        "aux_area_loss": 0.0,
    }
    count = 0
    for batch in loader:
        gt = batch["gt_mask"].to(device)
        yolo = batch["yolo_mask"].to(device)
        result = forward_prompt_model(
            model,
            adapter,
            scorer,
            batch,
            device,
            prompt_mode,
            disable_obj_score_gating=disable_obj_score_gating,
            use_no_mem_attention=use_no_mem_attention,
            return_details=aux_heads is not None and aux_feature_loss_weight > 0,
        )
        if isinstance(result, PromptForwardResult):
            logits = result.logits
        else:
            logits = result
        gt_loss = mask_loss_from_logits(logits, gt)
        if yolo_fidelity_weight > 0:
            if adaptive_fidelity:
                yolo_conf = batch["yolo_conf"].to(device)
                class_ids = batch["class_id"].to(device)
                reliability, _ = scorer.score(yolo, yolo_conf, class_ids)
                fidelity_weights = fidelity_min_weight + (fidelity_max_weight - fidelity_min_weight) * reliability
                fidelity_weights = yolo_fidelity_weight * fidelity_weights.clamp_min(0.0)
                per_sample_fidelity = per_sample_mask_loss_from_logits(logits, yolo)
                fidelity_loss = (per_sample_fidelity * fidelity_weights).mean()
                mean_fidelity_weight = float(fidelity_weights.detach().mean().item())
            else:
                fidelity_loss = mask_loss_from_logits(logits, yolo)
                mean_fidelity_weight = yolo_fidelity_weight
        else:
            fidelity_loss = logits.new_tensor(0.0)
            mean_fidelity_weight = 0.0
        loss = gt_loss + yolo_fidelity_weight * fidelity_loss
        if adaptive_fidelity:
            loss = gt_loss + fidelity_loss
        if aux_heads is not None and aux_feature_loss_weight > 0:
            aux_feature_loss, aux_metrics = mask_prior_aux_loss(
                aux_heads=aux_heads,
                adapter_out=result.adapter_out,
                geometry=result.geometry,
                yolo=yolo,
                gt=gt,
                feature_source=aux_feature_source,
            )
            loss = loss + aux_feature_loss_weight * aux_feature_loss
        else:
            aux_feature_loss = logits.new_tensor(0.0)
            aux_metrics = {
                "aux_bad_loss": 0.0,
                "aux_good_loss": 0.0,
                "aux_bucket_loss": 0.0,
                "aux_iou_loss": 0.0,
                "aux_boundary_loss": 0.0,
                "aux_area_loss": 0.0,
            }
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        refined = batch_metrics(logits.detach(), gt)
        yolo_logits = torch.logit(yolo.clamp(1e-4, 1 - 1e-4))
        yolo_metrics = batch_metrics(yolo_logits, gt)
        bs = gt.shape[0]
        total["loss"] += float(loss.item()) * bs
        total["gt_loss"] += float(gt_loss.item()) * bs
        total["fidelity_loss"] += float(fidelity_loss.item()) * bs
        total["aux_feature_loss"] += float(aux_feature_loss.item()) * bs
        total["iou"] += refined["iou"] * bs
        total["dice"] += refined["dice"] * bs
        total["yolo_iou"] += yolo_metrics["iou"] * bs
        total["fidelity_weight"] += mean_fidelity_weight * bs
        for key, value in aux_metrics.items():
            total[key] += value * bs
        count += bs
    return {key: value / max(count, 1) for key, value in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="yolo_mask_adapter/results/forceps_mask_cache_full")
    parser.add_argument("--train-datasets", nargs="+", default=["exp1_cu_full", "exp2_cu_full"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["interleaved", "random", "tail"], default="interleaved")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=["dense", "condition", "dense_condition"], default="dense_condition")
    parser.add_argument("--train-parts", nargs="+", default=["adapter"], choices=["adapter", "cstmamba", "prompt_encoder", "mask_decoder"])
    parser.add_argument(
        "--reinit-train-parts",
        action="store_true",
        help="Reinitialize selected CSTMamba/prompt/decoder modules before training instead of fine-tuning checkpoint weights.",
    )
    parser.add_argument("--output-dir", default="yolo_mask_adapter/results/mask_token_prompt_adapter_b1")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--base-ckpt-path", default="checkpoints/sam2.1_hiera_s_ref17.pth")
    parser.add_argument(
        "--image-cache-root",
        default="",
        help="Optional root containing copied images with the same relative paths as image_path entries.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-details-json", default="")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=0.0)
    parser.add_argument("--model-lr", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sentence-tokens", type=int, default=8)
    parser.add_argument(
        "--prior-mixer-depth",
        type=int,
        default=1,
        help="Number of lightweight self-attention mixer blocks inside MaskPriorEncoder. 0 keeps only per-token MLPs.",
    )
    parser.add_argument("--prior-mixer-heads", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--target-class-id", type=int, default=12)
    parser.add_argument(
        "--disable-obj-score-gating",
        action="store_true",
        help="Disable SAM object-score no-object mask suppression during training/eval forward.",
    )
    parser.add_argument(
        "--use-no-mem-attention",
        action="store_true",
        help=(
            "For no-memory/key-frame training, route fused visual features through SAM2's "
            "learned no_mem_embed/no_mem_pos_enc memory_attention path before the mask decoder."
        ),
    )
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
    parser.add_argument(
        "--yolo-fidelity-weight",
        type=float,
        default=0.0,
        help="Add BCE+Dice regularization toward the YOLO prior mask. 0 disables it.",
    )
    parser.add_argument(
        "--adaptive-fidelity",
        action="store_true",
        help="Scale fidelity regularization by YOLO mask reliability instead of using a fixed weight.",
    )
    parser.add_argument(
        "--fidelity-min-weight",
        type=float,
        default=0.25,
        help="Minimum multiplier for adaptive fidelity on low-reliability YOLO masks.",
    )
    parser.add_argument(
        "--fidelity-max-weight",
        type=float,
        default=1.50,
        help="Maximum multiplier for adaptive fidelity on high-reliability YOLO masks.",
    )
    parser.add_argument(
        "--aux-feature-loss-weight",
        type=float,
        default=0.0,
        help="Training-only auxiliary loss weight for forcing MaskPriorEncoder tokens to predict YOLO quality/error cues.",
    )
    parser.add_argument(
        "--aux-feature-source",
        choices=["sentence_flat", "sentence_mean", "cls", "sentence_mean_cls_geometry"],
        default="sentence_flat",
        help="Which frozen-style token feature is supervised by auxiliary quality/error heads.",
    )
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    datasets = set(args.train_datasets)
    rows = load_cache_index(cache_dir, datasets)
    train_paths, val_paths, split_summary = split_train_val(
        rows,
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
    )
    if args.max_train_items:
        train_paths = limit_paths_balanced(train_paths, args.max_train_items)
    if args.max_val_items:
        val_paths = limit_paths_balanced(val_paths, args.max_val_items)
    image_cache_root = Path(args.image_cache_root) if args.image_cache_root else None
    train_ds = ForcepsPromptDataset(
        train_paths,
        image_size=args.image_size,
        class_id=args.target_class_id,
        image_cache_root=image_cache_root,
    )
    val_ds = ForcepsPromptDataset(
        val_paths,
        image_size=args.image_size,
        class_id=args.target_class_id,
        image_cache_root=image_cache_root,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    print(f"[stage] build ReSurgSAM2 from {args.base_ckpt_path}", flush=True)
    model = build_resurgsam2(args.device, ckpt_path=args.base_ckpt_path)
    set_condition_fusion_mode(model, args.condition_fusion_mode)
    print("[stage] build ReSurgSAM2 done", flush=True)
    train_parts = set(args.train_parts)
    reinitialized_modules = reinitialize_selected_modules(model, train_parts) if args.reinit_train_parts else []
    set_trainable(model, train_parts)
    adapter = MaskTokenEncoder(
        embed_dim=model.hidden_dim,
        class_id=args.target_class_id,
        sentence_tokens=args.sentence_tokens,
        token_mixer_depth=args.prior_mixer_depth,
        token_mixer_heads=args.prior_mixer_heads,
    ).to(args.device)
    scorer = MaskReliabilityScorer(target_class_id=args.target_class_id)
    aux_heads = None
    if args.aux_feature_loss_weight > 0:
        aux_heads = MaskPriorAuxHeads(
            mask_prior_feature_dim(model.hidden_dim, args.sentence_tokens, args.aux_feature_source)
        ).to(args.device)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        if "adapter" in checkpoint:
            missing, unexpected = adapter.load_state_dict(checkpoint["adapter"], strict=False)
            print(
                json.dumps(
                    {
                        "resume_checkpoint": args.resume_checkpoint,
                        "loaded_adapter_tensors": len(checkpoint["adapter"]),
                        "adapter_missing_keys": len(missing),
                        "adapter_unexpected_keys": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
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
        if aux_heads is not None and "aux_heads" in checkpoint:
            missing, unexpected = aux_heads.load_state_dict(checkpoint["aux_heads"], strict=False)
            print(
                json.dumps(
                    {
                        "resume_checkpoint": args.resume_checkpoint,
                        "loaded_aux_head_tensors": len(checkpoint["aux_heads"]),
                        "aux_head_missing_keys": len(missing),
                        "aux_head_unexpected_keys": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    adapter_lr = args.adapter_lr if args.adapter_lr > 0 else args.lr
    model_lr = args.model_lr if args.model_lr > 0 else args.lr
    param_groups = [{"params": list(adapter.parameters()), "lr": adapter_lr}]
    if aux_heads is not None:
        param_groups.append({"params": list(aux_heads.parameters()), "lr": adapter_lr})
    model_params = [param for param in model.parameters() if param.requires_grad]
    if model_params:
        param_groups.append({"params": model_params, "lr": model_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summary["train_count"] = len(train_ds)
    split_summary["val_count"] = len(val_ds)
    split_summary["prompt_mode"] = args.prompt_mode
    split_summary["train_parts"] = sorted(train_parts)
    split_summary["prior_mixer_depth"] = args.prior_mixer_depth
    split_summary["prior_mixer_heads"] = args.prior_mixer_heads
    split_summary["image_cache_root"] = str(image_cache_root) if image_cache_root is not None else ""
    split_summary["reinit_train_parts"] = bool(args.reinit_train_parts)
    split_summary["reinitialized_modules"] = reinitialized_modules
    split_summary["adapter_lr"] = adapter_lr
    split_summary["model_lr"] = model_lr
    split_summary["target_class_id"] = args.target_class_id
    split_summary["disable_obj_score_gating"] = bool(args.disable_obj_score_gating)
    split_summary["use_no_mem_attention"] = bool(args.use_no_mem_attention)
    split_summary["condition_fusion_mode"] = args.condition_fusion_mode
    split_summary["yolo_fidelity_weight"] = args.yolo_fidelity_weight
    split_summary["adaptive_fidelity"] = bool(args.adaptive_fidelity)
    split_summary["fidelity_min_weight"] = args.fidelity_min_weight
    split_summary["fidelity_max_weight"] = args.fidelity_max_weight
    split_summary["aux_feature_loss_weight"] = args.aux_feature_loss_weight
    split_summary["aux_feature_source"] = args.aux_feature_source
    split_summary["resume_checkpoint"] = args.resume_checkpoint
    with (out_dir / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)

    if args.eval_only:
        val_metrics = run_epoch(
            model,
            adapter,
            scorer,
            val_loader,
            optimizer=None,
            aux_heads=aux_heads,
            device=args.device,
            prompt_mode=args.prompt_mode,
            train=False,
            disable_obj_score_gating=args.disable_obj_score_gating,
            use_no_mem_attention=args.use_no_mem_attention,
            yolo_fidelity_weight=args.yolo_fidelity_weight,
            adaptive_fidelity=args.adaptive_fidelity,
            fidelity_min_weight=args.fidelity_min_weight,
            fidelity_max_weight=args.fidelity_max_weight,
            aux_feature_loss_weight=args.aux_feature_loss_weight,
            aux_feature_source=args.aux_feature_source,
        )
        row = {"eval_only": True, "val": val_metrics}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (out_dir / "eval.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        if args.eval_details_json:
            details = run_eval_details(
                model,
                adapter,
                scorer,
                val_loader,
                args.device,
                args.prompt_mode,
                disable_obj_score_gating=args.disable_obj_score_gating,
                use_no_mem_attention=args.use_no_mem_attention,
            )
            output = Path(args.eval_details_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                json.dump(details, f, ensure_ascii=False, indent=2)
            print(json.dumps({"eval_details_json": str(output), "overall": details["overall"]}, ensure_ascii=False), flush=True)
        return

    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            adapter,
            scorer,
            train_loader,
            optimizer,
            aux_heads,
            args.device,
            args.prompt_mode,
            train=True,
            disable_obj_score_gating=args.disable_obj_score_gating,
            use_no_mem_attention=args.use_no_mem_attention,
            yolo_fidelity_weight=args.yolo_fidelity_weight,
            adaptive_fidelity=args.adaptive_fidelity,
            fidelity_min_weight=args.fidelity_min_weight,
            fidelity_max_weight=args.fidelity_max_weight,
            aux_feature_loss_weight=args.aux_feature_loss_weight,
            aux_feature_source=args.aux_feature_source,
        )
        val_metrics = run_epoch(
            model,
            adapter,
            scorer,
            val_loader,
            optimizer,
            aux_heads,
            args.device,
            args.prompt_mode,
            train=False,
            disable_obj_score_gating=args.disable_obj_score_gating,
            use_no_mem_attention=args.use_no_mem_attention,
            yolo_fidelity_weight=args.yolo_fidelity_weight,
            adaptive_fidelity=args.adaptive_fidelity,
            fidelity_min_weight=args.fidelity_min_weight,
            fidelity_max_weight=args.fidelity_max_weight,
            aux_feature_loss_weight=args.aux_feature_loss_weight,
            aux_feature_source=args.aux_feature_source,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "aux_heads": aux_heads.state_dict() if aux_heads is not None else {},
                    "model": trainable_model_state(model),
                    "epoch": epoch,
                    "val": val_metrics,
                    "prompt_mode": args.prompt_mode,
                    "train_parts": sorted(train_parts),
                    "disable_obj_score_gating": bool(args.disable_obj_score_gating),
                    "use_no_mem_attention": bool(args.use_no_mem_attention),
                    "condition_fusion_mode": args.condition_fusion_mode,
                    "prior_mixer_depth": args.prior_mixer_depth,
                    "prior_mixer_heads": args.prior_mixer_heads,
                    "yolo_fidelity_weight": args.yolo_fidelity_weight,
                    "adaptive_fidelity": bool(args.adaptive_fidelity),
                    "fidelity_min_weight": args.fidelity_min_weight,
                    "fidelity_max_weight": args.fidelity_max_weight,
                    "aux_feature_loss_weight": args.aux_feature_loss_weight,
                    "aux_feature_source": args.aux_feature_source,
                },
                out_dir / "best.pt",
            )
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "aux_heads": aux_heads.state_dict() if aux_heads is not None else {},
            "epoch": args.epochs,
            "val": history[-1]["val"],
            "prior_mixer_depth": args.prior_mixer_depth,
            "prior_mixer_heads": args.prior_mixer_heads,
            "use_no_mem_attention": bool(args.use_no_mem_attention),
            "condition_fusion_mode": args.condition_fusion_mode,
            "aux_feature_loss_weight": args.aux_feature_loss_weight,
            "aux_feature_source": args.aux_feature_source,
        },
        out_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
