# Copilot instructions for ReSurgSAM2

## Big picture
- ReSurgSAM2 is a two-stage referring video segmentation system built on SAM2: text expression -> CLIP text encoding -> cross-modal fusion -> SAM mask decoding -> video tracking with memory.
- Core architecture lives in sam2/modeling/: sam2_base.py wires the backbone, prompt/mask heads, memory attention, and CLIP; cross_modal_fusion.py fuses text with the top image feature level; sam2_video_predictor.py runs interactive tracking.

## Main code paths
- Training entrypoint: training/train.py uses Hydra + Submitit and writes resolved configs to sam2_logs/<config>/.
- Training wrapper: training/model/sam2.py builds BatchedVideoDatapoint prompts and switches between point, box, mask, and text prompts.
- Inference/eval: tools/rvos_inference.py loads meta_expressions.json, calls predictor.add_new_text(), then propagate_in_video(), and saves both per-object PNGs and merged palette PNGs.

## Config conventions
- Training configs are in sam2/configs/rvos_training/17/ and sam2/configs/rvos_training/18/; shared knobs live under scratch and are wired through Hydra _target_ blocks.
- Ref-EndoVis17 uses used_object_ids [1..10]; Ref-EndoVis18 uses [1..12, 17]. Standard settings use resolution 512, num_ref_frames 3, num_tracking_frames 7; pretrained-stage configs set num_tracking_frames 0.
- build_sam.py maps predictor options to Hydra overrides, especially apply_long_term_memory, apply_credible_initial_frame, num_cifs_candidate_frame, and postprocessing flags.

## Data layout
- PNGRawDataset expects data/Ref-Endovis17|18/{train,valid}/{JPEGImages,Annotations,Meta} with one Meta/<video>.json per video and palettized PNG masks.
- The dataset README documents two corrected label entries for the released Ref-EndoVis17/18 training sets; keep those fixes intact when touching preprocessing.

## Tracking and memory behavior
- SAM2VideoPredictor maintains per-object state in point_inputs_per_obj, mask_inputs_per_obj, text_inputs_per_obj, output_dict_per_obj, temp_output_dict_per_obj, and frames_tracked_per_obj.
- Credible Initial Frame Selection (CIFS) picks candidate frames when object score and IoU thresholds are met; long-term memory keeps long_mem_candidate and long_mem deques and selects diverse frames by feature similarity.
- Preserve object/frame indexing, mask resolution handling, and non-overlap logic; the predictor writes outputs at original video resolution before evaluation.

## Text prompting details
- SAM2Base builds a CLIP text encoder by default (language_encoder_name='clip') and freezes it; the projected sentence embeddings and CLS token flow into cross_modal_fusion and the SAM prompt encoder.
- training/model/sam2.py and sam2_video_predictor.py both cache text_emb_sentence and text_emb_cls; keep this contract stable when modifying text prompting.

## Workflow expectations
- Install the repo with pip install -e ".[dev]" and install the local mamba/ package separately with pip install -e . from mamba/.
- Typical commands are python training/train.py -c <config> --num-gpus <N> and python tools/rvos_inference.py ... --apply_long_term_memory --num_cifs_candidate_frame 5.
- Hydra errors are already verbose because HYDRA_FULL_ERROR=1 is set in training/train.py; use the generated config.yaml and config_resolved.yaml in the log directory to debug runs.

## Editing guidance
- Prefer config/Hydra changes over hardcoding behavior in Python.
- Keep tensor shapes stable across sam2_base.py, cross_modal_fusion.py, and sam2_video_predictor.py; most bugs here are shape, frame-order, or state-dict mismatches.
- If you change inference behavior, verify both per-object outputs and the merged palette output in tools/rvos_inference.py.
- If you touch Mamba-specific code, also check mamba/README.md and mamba/tests/ for the expected local-package workflow.
