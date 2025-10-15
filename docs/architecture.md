# Detector Architecture Plan

## Objectives

- Compare lightweight CNN detectors (YOLOv8-n, YOLOv5-n, YOLOv10-n, PP-PicoDet-S) and transformer-based RT-DETR-tiny.
- Maintain TensorRT-friendly graph constraints (fused Conv+BN+Act, channels % 8, NCHW).
- Support both PTQ and QAT pipelines for INT8 deployment.

## Module Responsibilities

- `src/uaelp_det/data/`: preprocessing, augmentation, calibration subset management.
- `src/uaelp_det/models/`: backbone builders and registry.
- `src/uaelp_det/training/`: unified loss, trainer loop, hooks (EMA, grad clip, AMP, QAT).
- `src/uaelp_det/evaluation/`: metric logging (mAP, precision/recall/F1, throughput).
- `src/uaelp_det/quantization/`: QAT preparation and PTQ calibration utilities.
- `src/uaelp_det/export/`: ONNX and TensorRT export workflows.

## Training Lifecyle

1. Load configs (`configs/training/default.yaml`), compose dataset/transforms.
2. Build models via registry and allocate optimizer/scheduler.
3. Train with cosine LR, EMA, AMP; enable QAT for final 30% epochs.
4. Log metrics to CSV/JSONL; persist checkpoints atomically under `runs/YYYYMMDD-HHMMSS_<model>_<run_id>/`.
5. Export best checkpoint to ONNX → TensorRT INT8 plan using calibration cache.
