"""YOLO mask prompt adapter for ReSurgSAM2 experiments."""

from .mask_token_encoder import MaskTokenEncoder, MaskTokenEncoderOutput
from .predictor_mixin import YOLOMaskPromptMixin, resize_mask_to_video
from .reliability import MaskReliabilityScorer, ReliabilityState


def __getattr__(name):
    if name == "YOLOMaskSAM2VideoPredictor":
        from .yolo_mask_video_predictor import YOLOMaskSAM2VideoPredictor

        return YOLOMaskSAM2VideoPredictor
    raise AttributeError(name)

__all__ = [
    "MaskTokenEncoder",
    "MaskTokenEncoderOutput",
    "YOLOMaskPromptMixin",
    "resize_mask_to_video",
    "MaskReliabilityScorer",
    "ReliabilityState",
    "YOLOMaskSAM2VideoPredictor",
]
