"""Credible initial/correction frame selection utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIFSConfig:
    window_size: int = 5
    min_confidence: float = 0.45
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.65
    confidence_weight: float = 0.45
    area_weight: float = 0.20
    temporal_weight: float = 0.25
    smoothness_weight: float = 0.10
    area_jump_trigger: float = 0.45
    low_conf_trigger: float = 0.35
    low_score_patience: int = 3


def mask_area_ratio(mask: np.ndarray) -> float:
    return float(mask.astype(bool).mean())


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def edge_density(mask: np.ndarray) -> float:
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return 1.0
    dy = np.abs(np.diff(mask, axis=0)).sum()
    dx = np.abs(np.diff(mask, axis=1)).sum()
    return float((dx + dy) / mask.sum())


def candidate_score(
    mask: np.ndarray,
    confidence: float,
    prev_mask: np.ndarray | None,
    config: CIFSConfig,
) -> dict:
    area = mask_area_ratio(mask)
    area_valid = config.min_area_ratio <= area <= config.max_area_ratio
    area_score = 1.0 if area_valid else 0.0
    temporal_score = mask_iou(mask, prev_mask) if prev_mask is not None else 0.5
    smoothness_score = 1.0 / (1.0 + edge_density(mask))
    confidence_score = float(np.clip(confidence, 0.0, 1.0))
    total = (
        config.confidence_weight * confidence_score
        + config.area_weight * area_score
        + config.temporal_weight * temporal_score
        + config.smoothness_weight * smoothness_score
    )
    return {
        "score": float(total),
        "confidence": confidence_score,
        "area_ratio": area,
        "area_valid": bool(area_valid),
        "temporal_iou": float(temporal_score),
        "smoothness": float(smoothness_score),
    }


def select_credible_frame(candidates: list[dict], config: CIFSConfig | None = None) -> dict:
    config = config or CIFSConfig()
    if not candidates:
        raise ValueError("CIFS requires at least one candidate")
    window = candidates[: config.window_size]
    prev_mask = None
    scored = []
    for candidate in window:
        metrics = candidate_score(
            candidate["mask"],
            float(candidate.get("confidence", 0.0)),
            prev_mask,
            config,
        )
        scored.append({**candidate, "cifs": metrics})
        prev_mask = candidate["mask"]
    usable = [
        item
        for item in scored
        if item["cifs"]["confidence"] >= config.min_confidence and item["cifs"]["area_valid"]
    ]
    pool = usable if usable else scored
    return max(pool, key=lambda item: item["cifs"]["score"])


def should_trigger_correction(recent_states: list[dict], config: CIFSConfig | None = None) -> tuple[bool, str]:
    config = config or CIFSConfig()
    if len(recent_states) < 2:
        return False, "insufficient_history"
    last = recent_states[-1]
    prev = recent_states[-2]
    last_area = float(last.get("area_ratio", 0.0))
    prev_area = float(prev.get("area_ratio", 0.0))
    if prev_area > 0:
        jump = abs(last_area - prev_area) / max(prev_area, 1e-6)
        if jump >= config.area_jump_trigger:
            return True, "area_jump"
    if float(last.get("confidence", 1.0)) < config.low_conf_trigger:
        return True, "low_confidence"
    tail = recent_states[-config.low_score_patience :]
    if len(tail) == config.low_score_patience and all(float(s.get("score", 1.0)) < config.min_confidence for s in tail):
        return True, "persistent_low_score"
    return False, "stable"
