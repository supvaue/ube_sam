"""Smoke test for the YOLO mask adapter.

Run from repository root:
    python Code/ReSurgSAM2/yolo_mask_adapter/smoke_test.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_mask_adapter import MaskReliabilityScorer, MaskTokenEncoder, ReliabilityState


def main() -> None:
    torch.manual_seed(0)
    batch, channels, height, width = 2, 256, 32, 32
    features = torch.randn(batch, channels, height, width, requires_grad=True)
    masks = torch.zeros(batch, 1, 128, 128)
    masks[0, :, 30:80, 40:100] = 1
    masks[1, :, 50:110, 20:70] = 1
    confidence = torch.tensor([0.91, 0.62])
    class_ids = torch.tensor([12, 9])

    scorer = MaskReliabilityScorer(target_class_id=12)
    reliability, geometry = scorer.score(
        masks,
        confidence,
        class_ids,
        previous=ReliabilityState(mask=masks.clone(), class_id=12),
    )

    encoder = MaskTokenEncoder(embed_dim=channels, sentence_tokens=8, class_id=12)
    output = encoder(features, masks, geometry, class_ids=class_ids)
    loss = output.mask_emb_sentence.mean() + output.mask_emb_cls.mean()
    loss.backward()

    print(f"reliability={reliability.tolist()}")
    print(f"geometry_shape={tuple(geometry.shape)}")
    print(f"sentence_shape={tuple(output.mask_emb_sentence.shape)}")
    print(f"cls_shape={tuple(output.mask_emb_cls.shape)}")
    print(f"dense_prompt_shape={tuple(output.dense_prompt_mask.shape)}")
    print(f"grad_ok={features.grad is not None}")
    print(f"feature_grad_mean={float(features.grad.abs().mean())}")

    assert geometry.shape == (batch, 10)
    assert output.mask_emb_sentence.shape == (batch, 8, channels)
    assert output.mask_emb_cls.shape == (batch, 1, channels)
    assert output.dense_prompt_mask.shape == masks.shape
    assert features.grad is not None


if __name__ == "__main__":
    main()

