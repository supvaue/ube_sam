"""Video predictor variant that accepts YOLO mask condition tokens."""

from sam2.sam2_video_predictor import SAM2VideoPredictor

from .predictor_mixin import YOLOMaskPromptMixin


class YOLOMaskSAM2VideoPredictor(YOLOMaskPromptMixin, SAM2VideoPredictor):
    """ReSurgSAM2 video predictor with YOLO-mask prompt injection.

    The class intentionally keeps SAM2VideoPredictor behavior unchanged until
    `add_new_yolo_mask` is called. This makes adapter ablations simple:
    use the stock predictor for text/mask baselines, or this subclass for the
    YOLO mask condition-token path.
    """

    pass

