"""Reliability scoring for YOLO masks used as ReSurgSAM2 prompts."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ReliabilityState:
    """Previous reliable mask state used for temporal consistency scoring."""

    mask: Optional[torch.Tensor] = None
    class_id: Optional[int] = None


class MaskReliabilityScorer:
    """Compute a bounded reliability score for YOLO mask prompts."""

    def __init__(
        self,
        target_class_id: int = 12,
        reliable_threshold: float = 0.78,
        usable_threshold: float = 0.55,
    ) -> None:
        self.target_class_id = target_class_id
        self.reliable_threshold = reliable_threshold
        self.usable_threshold = usable_threshold

    def score(
        self,
        mask: torch.Tensor,
        confidence: torch.Tensor,
        class_id: torch.Tensor,
        previous: Optional[ReliabilityState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reliability score and geometry vector.

        Geometry vector layout:
        [conf, area_ratio, bbox_w, bbox_h, aspect, edge_smooth, hole_free,
         temporal_iou, area_stability, class_stability]
        """
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must be [B, 1, H, W]")
        bsz = mask.shape[0]
        mask_bin = (mask.float() >= 0.5).float()
        conf = confidence.to(mask.device, dtype=mask.dtype).view(bsz).clamp(0.0, 1.0)
        class_id = class_id.to(mask.device).view(bsz)

        area_ratio = mask_bin.flatten(1).mean(dim=1)
        bbox_w, bbox_h, aspect = self._bbox_geometry(mask_bin)
        edge_smooth = self._edge_smooth_score(mask_bin)
        hole_free = self._hole_free_score(mask_bin)

        temporal_iou = torch.ones_like(conf) * 0.5
        area_stability = torch.ones_like(conf) * 0.5
        if previous is not None and previous.mask is not None:
            prev_mask = previous.mask.to(mask.device)
            if prev_mask.shape[-2:] != mask_bin.shape[-2:]:
                prev_mask = F.interpolate(
                    prev_mask.float(),
                    size=mask_bin.shape[-2:],
                    mode="nearest",
                )
            prev_bin = (prev_mask >= 0.5).float()
            temporal_iou = self._mask_iou(mask_bin, prev_bin)
            prev_area = prev_bin.flatten(1).mean(dim=1)
            area_stability = 1.0 - (area_ratio - prev_area).abs() / torch.maximum(
                area_ratio.maximum(prev_area),
                torch.full_like(area_ratio, 1e-6),
            )
            area_stability = area_stability.clamp(0.0, 1.0)

        class_stability = (class_id == self.target_class_id).float()
        if previous is not None and previous.class_id is not None:
            class_stability = class_stability * float(previous.class_id == self.target_class_id)

        geometry = torch.stack(
            [
                conf,
                area_ratio,
                bbox_w,
                bbox_h,
                aspect,
                edge_smooth,
                hole_free,
                temporal_iou,
                area_stability,
                class_stability,
            ],
            dim=1,
        )
        reliability = (
            0.25 * conf
            + 0.20 * edge_smooth
            + 0.15 * hole_free
            + 0.20 * temporal_iou
            + 0.10 * area_stability
            + 0.10 * class_stability
        ).clamp(0.0, 1.0)
        return reliability, geometry

    @staticmethod
    def _bbox_geometry(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, height, width = mask.shape
        device = mask.device
        ys = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
        xs = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
        present = mask > 0.5
        large = torch.full((bsz,), 1e6, device=device)
        small = torch.full((bsz,), -1e6, device=device)
        x_vals = xs.expand_as(mask)
        y_vals = ys.expand_as(mask)
        x_min = torch.where(present, x_vals, large.view(bsz, 1, 1, 1)).flatten(1).min(dim=1).values
        x_max = torch.where(present, x_vals, small.view(bsz, 1, 1, 1)).flatten(1).max(dim=1).values
        y_min = torch.where(present, y_vals, large.view(bsz, 1, 1, 1)).flatten(1).min(dim=1).values
        y_max = torch.where(present, y_vals, small.view(bsz, 1, 1, 1)).flatten(1).max(dim=1).values
        has_mask = present.flatten(1).any(dim=1)
        bbox_w = torch.where(has_mask, (x_max - x_min).clamp_min(0.0), torch.zeros_like(x_min))
        bbox_h = torch.where(has_mask, (y_max - y_min).clamp_min(0.0), torch.zeros_like(y_min))
        aspect = (bbox_w / bbox_h.clamp_min(1e-6)).clamp(0.0, 10.0) / 10.0
        return bbox_w, bbox_h, aspect

    @staticmethod
    def _edge_smooth_score(mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        edge = (dilated - eroded).clamp(0.0, 1.0)
        edge_area = edge.flatten(1).mean(dim=1)
        mask_area = mask.flatten(1).mean(dim=1).clamp_min(1e-6)
        roughness = (edge_area / mask_area.sqrt()).clamp(0.0, 2.0) / 2.0
        return (1.0 - roughness).clamp(0.0, 1.0)

    @staticmethod
    def _hole_free_score(mask: torch.Tensor) -> torch.Tensor:
        closed = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=5, stride=1, padding=2)
        opened_holes = (mask - closed).abs().flatten(1).mean(dim=1)
        area = mask.flatten(1).mean(dim=1).clamp_min(1e-6)
        hole_ratio = (opened_holes / area).clamp(0.0, 1.0)
        return 1.0 - hole_ratio

    @staticmethod
    def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        inter = (mask_a * mask_b).flatten(1).sum(dim=1)
        union = ((mask_a + mask_b) > 0).float().flatten(1).sum(dim=1)
        return torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))

