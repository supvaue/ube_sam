"""Mixin for injecting YOLO mask condition tokens into ReSurgSAM2 predictors."""

from typing import Optional

import torch
import torch.nn.functional as F

from .mask_token_encoder import MaskTokenEncoder
from .reliability import MaskReliabilityScorer, ReliabilityState


class YOLOMaskPromptMixin:
    """Add YOLO-mask prompt support to a ReSurgSAM2 video predictor.

    This mixin expects the host class to provide ReSurgSAM2 predictor methods:
    `_obj_id_to_idx`, `_get_image_feature`, and `_consolidate_temp_output_across_obj`.
    It stores YOLO mask condition tokens under the existing `text_emb` key so the
    current referring path can consume them without changing CSTMamba first.
    """

    def init_yolo_mask_adapter(
        self,
        target_class_id: int = 12,
        sentence_tokens: int = 8,
        prior_mixer_depth: int = 1,
        prior_mixer_heads: int = 4,
        reliable_threshold: float = 0.78,
        usable_threshold: float = 0.55,
    ) -> None:
        self.yolo_mask_adapter = MaskTokenEncoder(
            embed_dim=self.hidden_dim,
            class_id=target_class_id,
            sentence_tokens=sentence_tokens,
            token_mixer_depth=prior_mixer_depth,
            token_mixer_heads=prior_mixer_heads,
        ).to(self.device)
        self.yolo_reliability_scorer = MaskReliabilityScorer(
            target_class_id=target_class_id,
            reliable_threshold=reliable_threshold,
            usable_threshold=usable_threshold,
        )
        self.yolo_reliability_state = {}

    @torch.inference_mode()
    def add_new_mask_tokens(
        self,
        inference_state,
        frame_idx: int,
        obj_id,
        mask_emb_sentence: torch.Tensor,
        mask_emb_cls: torch.Tensor,
        source: str = "mask_tokens",
        reliability: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        dense_prompt_mask: Optional[torch.Tensor] = None,
    ):
        """Inject precomputed mask-conditioned tokens into the referring path.

        This mirrors `add_new_text`: the tokens are stored under the existing
        `text_emb` key because ReSurgSAM2's referring code consumes that
        contract. Semantically, however, these are mask/Yolo-conditioned tokens:

        - mask_emb_sentence: [B, N_mask, hidden_dim], consumed by CSTMamba /
          CrossModalFusion together with image features.
        - mask_emb_cls: [B, 1, hidden_dim], consumed as sparse prompt evidence
          by PromptEncoder/MaskDecoder when `forward_text_emb` is enabled.
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        device = inference_state["device"]
        mask_emb_sentence = mask_emb_sentence.to(device=device, dtype=torch.float32)
        mask_emb_cls = mask_emb_cls.to(device=device, dtype=torch.float32)

        if mask_emb_sentence.ndim != 3:
            raise ValueError("mask_emb_sentence must be [B, N_mask, C]")
        if mask_emb_cls.ndim != 3:
            raise ValueError("mask_emb_cls must be [B, 1, C]")
        if mask_emb_sentence.shape[0] != mask_emb_cls.shape[0]:
            raise ValueError("mask_emb_sentence and mask_emb_cls batch size must match")
        if mask_emb_cls.shape[1] != 1:
            raise ValueError("mask_emb_cls must have exactly one CLS token")
        if mask_emb_sentence.shape[-1] != self.hidden_dim or mask_emb_cls.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"mask token dim must equal predictor hidden_dim={self.hidden_dim}; "
                f"got sentence={mask_emb_sentence.shape[-1]}, cls={mask_emb_cls.shape[-1]}"
            )

        text_inputs_per_object = inference_state["text_inputs_per_obj"][obj_idx]
        text_inputs_per_object["text_emb"] = {
            "text_emb_sentence": mask_emb_sentence,
            "text_emb_cls": mask_emb_cls,
            "source": source,
        }
        if reliability is not None:
            text_inputs_per_object["text_emb"]["reliability"] = reliability.detach()
        if geometry is not None:
            text_inputs_per_object["text_emb"]["geometry"] = geometry.detach()
        if dense_prompt_mask is not None:
            dense_prompt_mask = dense_prompt_mask.to(device=device, dtype=torch.float32)
            if dense_prompt_mask.ndim != 4 or dense_prompt_mask.shape[:2] != (mask_emb_sentence.shape[0], 1):
                raise ValueError("dense_prompt_mask must be [B, 1, H, W]")
            text_inputs_per_object["text_emb"]["dense_prompt_mask"] = dense_prompt_mask.detach()

        # Match add_new_text behavior: create a referring placeholder so
        # propagate_in_video can start from the referring/CIFS branch without
        # requiring a point/mask conditioning frame first.
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        obj_temp_output_dict["ref_frame_outputs"][frame_idx] = None
        return {
            "accepted": True,
            "source": source,
            "sentence_shape": tuple(mask_emb_sentence.shape),
            "cls_shape": tuple(mask_emb_cls.shape),
        }

    @torch.inference_mode()
    def add_new_yolo_mask(
        self,
        inference_state,
        frame_idx: int,
        obj_id,
        mask,
        confidence: float,
        class_id: int = 12,
        previous_state: Optional[ReliabilityState] = None,
        require_usable: bool = True,
    ):
        """Encode a YOLO mask as ReSurgSAM2 referring condition.

        The method does not directly make the mask a conditioning memory. It first
        injects condition tokens, allowing CIFS to decide whether a frame is credible
        enough to enter memory.
        """
        if not hasattr(self, "yolo_mask_adapter"):
            self.init_yolo_mask_adapter(target_class_id=class_id)

        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array/tensor")

        mask = mask.float().to(inference_state["device"])[None, None]
        if mask.shape[-2:] != (self.image_size, self.image_size):
            mask = F.interpolate(
                mask,
                size=(self.image_size, self.image_size),
                mode="nearest",
            )
        confidence_t = torch.tensor([confidence], device=inference_state["device"])
        class_t = torch.tensor([class_id], device=inference_state["device"])
        if previous_state is None:
            previous_state = self.yolo_reliability_state.get(obj_idx)

        reliability, geometry = self.yolo_reliability_scorer.score(
            mask,
            confidence_t,
            class_t,
            previous=previous_state,
        )
        if require_usable and reliability.item() < self.yolo_reliability_scorer.usable_threshold:
            return {
                "accepted": False,
                "reason": "low_reliability",
                "reliability": float(reliability.item()),
            }

        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state,
            frame_idx,
            batch_size=1,
        )
        low_feat = current_vision_feats[-1]
        h, w = feat_sizes[-1]
        image_features = low_feat.permute(1, 2, 0).view(1, self.hidden_dim, h, w)

        adapter_out = self.yolo_mask_adapter(
            image_features=image_features,
            yolo_mask=mask,
            geometry=geometry,
            class_ids=class_t,
        )

        token_result = self.add_new_mask_tokens(
            inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask_emb_sentence=adapter_out.mask_emb_sentence,
            mask_emb_cls=adapter_out.mask_emb_cls,
            source="yolo_mask_adapter_v0.2",
            reliability=reliability,
            geometry=geometry,
            dense_prompt_mask=adapter_out.dense_prompt_mask,
        )

        if reliability.item() >= self.yolo_reliability_scorer.reliable_threshold:
            self.yolo_reliability_state[obj_idx] = ReliabilityState(
                mask=mask.detach().clone(),
                class_id=class_id,
            )

        return {
            "accepted": True,
            "reliability": float(reliability.item()),
            "geometry": geometry.detach().cpu(),
            "sentence_shape": token_result["sentence_shape"],
            "cls_shape": token_result["cls_shape"],
        }


def resize_mask_to_video(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Utility for callers that need to normalize mask shape before injection."""
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    return F.interpolate(mask.float(), size=(height, width), mode="nearest")
