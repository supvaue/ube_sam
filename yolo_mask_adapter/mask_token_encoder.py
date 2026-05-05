"""Encode YOLO masks into condition tokens for ReSurgSAM2 CSTMamba.

This module is intentionally independent from the ReSurgSAM2 predictor so it can
be unit-tested before the mamba/checkpoint environment is ready.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MaskTokenEncoderOutput:
    """Condition tokens matching ReSurgSAM2 text embedding interfaces."""

    mask_emb_sentence: torch.Tensor
    mask_emb_cls: torch.Tensor
    dense_prompt_mask: torch.Tensor


class ResidualTokenMixerBlock(nn.Module):
    """Lightweight token mixer for mask/prior tokens."""

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attn(tokens)
        mixed, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + mixed
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens


class MaskTokenEncoder(nn.Module):
    """Turn a YOLO mask plus image features into CSTMamba condition tokens.

    The output mirrors ReSurgSAM2's text embedding contract:
    - mask_emb_sentence: [B, N, C] for cross-modal fusion.
    - mask_emb_cls: [B, 1, C] for sparse prompt/class token injection.

    Unlike a pure binary-mask encoder, this module binds the mask to current
    image features through masked and boundary pooling, so the token carries
    target appearance, edge evidence, and reliability/geometry metadata.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 32,
        class_id: int = 12,
        geometry_dim: int = 10,
        sentence_tokens: int = 8,
        token_mixer_depth: int = 1,
        token_mixer_heads: int = 4,
    ) -> None:
        super().__init__()
        if sentence_tokens < 4:
            raise ValueError("sentence_tokens must be at least 4")
        self.embed_dim = embed_dim
        self.class_id = class_id
        self.sentence_tokens = sentence_tokens
        self.token_mixer_depth = token_mixer_depth
        self.token_mixer_heads = token_mixer_heads

        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.mask_visual_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.boundary_visual_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if token_mixer_depth > 0:
            self.token_mixer = nn.Sequential(
                *[
                    ResidualTokenMixerBlock(
                        embed_dim=embed_dim,
                        num_heads=token_mixer_heads,
                    )
                    for _ in range(token_mixer_depth)
                ]
            )
        else:
            self.token_mixer = nn.Identity()
        self.token_out_norm = nn.LayerNorm(embed_dim)
        self.cls_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        extra = sentence_tokens - 4
        self.extra_tokens = nn.Parameter(torch.zeros(1, extra, embed_dim)) if extra else None
        if self.extra_tokens is not None:
            nn.init.trunc_normal_(self.extra_tokens, std=0.02)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Backward compatibility: old checkpoints used `token_mixer` for the
        # per-token MLP. The new name `token_proj` leaves `token_mixer` for
        # cross-token prior mixing.
        if "token_proj.0.weight" not in state_dict and "token_mixer.0.weight" in state_dict:
            state_dict = dict(state_dict)
            for old_key in list(state_dict):
                if old_key.startswith("token_mixer."):
                    new_key = "token_proj." + old_key[len("token_mixer.") :]
                    state_dict[new_key] = state_dict.pop(old_key)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        image_features: torch.Tensor,
        yolo_mask: torch.Tensor,
        geometry: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        previous_reliable_token: Optional[torch.Tensor] = None,
    ) -> MaskTokenEncoderOutput:
        """Encode mask into condition tokens.

        Args:
            image_features: Current ReSurgSAM2 feature map, [B, C, H, W].
            yolo_mask: YOLO binary/probability mask at any resolution, [B, 1, h, w].
            geometry: Reliability/geometry vector, [B, 10].
            class_ids: Category ids. Defaults to class_id=12, the nucleus forceps.
            previous_reliable_token: Optional history prototype, [B, C] or [B, 1, C].
        """
        if image_features.ndim != 4:
            raise ValueError("image_features must be [B, C, H, W]")
        if yolo_mask.ndim != 4 or yolo_mask.shape[1] != 1:
            raise ValueError("yolo_mask must be [B, 1, H, W]")
        if geometry.ndim != 2 or geometry.shape[1] != 10:
            raise ValueError("geometry must be [B, 10]")

        bsz, channels, height, width = image_features.shape
        if channels != self.embed_dim:
            raise ValueError(f"image feature dim {channels} != embed_dim {self.embed_dim}")
        if geometry.shape[0] != bsz:
            raise ValueError("geometry batch size must match image_features")

        mask_small = self._resize_mask(yolo_mask, (height, width))
        boundary_small = self._boundary_ring(mask_small)

        masked_visual = self._masked_average(image_features, mask_small)
        boundary_visual = self._masked_average(image_features, boundary_small)

        if class_ids is None:
            class_ids = torch.full(
                (bsz,),
                self.class_id,
                dtype=torch.long,
                device=image_features.device,
            )
        else:
            class_ids = class_ids.to(device=image_features.device, dtype=torch.long).view(-1)
        class_token = self.class_embed(class_ids)

        if previous_reliable_token is None:
            history_token = torch.zeros_like(masked_visual)
        else:
            history_token = previous_reliable_token
            if history_token.ndim == 3:
                history_token = history_token.squeeze(1)
            history_token = history_token.to(device=image_features.device, dtype=image_features.dtype)

        tokens = [
            self.mask_visual_proj(masked_visual),
            self.boundary_visual_proj(boundary_visual),
            self.geometry_proj(geometry.to(device=image_features.device, dtype=image_features.dtype)),
            self.temporal_proj(history_token),
        ]
        if self.extra_tokens is not None:
            extra = self.extra_tokens.expand(bsz, -1, -1)
            tokens.append(extra)
        sentence = torch.cat([t.unsqueeze(1) if t.ndim == 2 else t for t in tokens], dim=1)
        sentence = sentence[:, : self.sentence_tokens]
        sentence = self.token_proj(sentence)
        sentence = self.token_out_norm(self.token_mixer(sentence))

        cls_seed = sentence.mean(dim=1) + class_token
        cls_token = self.cls_mixer(cls_seed).unsqueeze(1)

        return MaskTokenEncoderOutput(
            mask_emb_sentence=sentence,
            mask_emb_cls=cls_token,
            dense_prompt_mask=yolo_mask.float(),
        )

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        mask = F.interpolate(mask.float(), size=size, mode="bilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _boundary_ring(mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        ring = (dilated - eroded).clamp(0.0, 1.0)
        return ring

    @staticmethod
    def _masked_average(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(dtype=features.dtype)
        denom = weights.flatten(2).sum(dim=-1).clamp_min(1.0)
        pooled = (features * weights).flatten(2).sum(dim=-1) / denom
        empty = (weights.flatten(2).sum(dim=-1) <= 1e-6).expand_as(pooled)
        global_pooled = features.flatten(2).mean(dim=-1)
        return torch.where(empty, global_pooled, pooled)
