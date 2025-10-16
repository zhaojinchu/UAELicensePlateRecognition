# UAE License Plate Detector (Lean Scaffold)

This repository provides a minimal-yet-production-minded scaffold for the UAE license plate detector research plan. It keeps the code surface area small while wiring in the core ingredients needed for INT8 deployment: consistent preprocessing, unified loss, AMP + EMA, late-stage QAT, rich metric logging, and checkpoint rotation.

## Repository Layout

```
data/                 # Place raw/processed images, YOLO labels, calibration sets, splits
modules/              # Reusable code: data pipeline, models, losses, metrics, utils, quantization
runs/                 # Training outputs (metrics, checkpoints, exports)
config.yaml           # Single YAML config describing experiment/data/model/training settings
requirements.txt      # Minimal dependency list
train.py              # End-to-end training + validation loop with logging & checkpointing
```

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# or
source .venv/bin/activate     # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

1. Place your COCO and/or YOLO formatted datasets under `data/`.
2. (Optional) Merge COCO JSON and YOLO TXT labels into a single YOLO-style directory:
   ```python
   from pathlib import Path
   from modules.data import MergeConfig, merge_coco_yolo_datasets

   merge_coco_yolo_datasets(
       MergeConfig(
           output_root=Path("data/processed/unified"),
           coco_annotations=Path("data/annotations/train.json"),
           coco_images_dir=Path("data/raw/images"),
           yolo_labels_dir=Path("data/yolo/labels"),
           yolo_images_dir=Path("data/yolo/images"),
           overwrite=False,
       )
   )
   ```
3. Define reproducible splits (`data/splits/train.txt`, `data/splits/val.txt`) with one image path per line.
4. Reserve 200–500 in-distribution images (no augmentations) under `data/calibration/` for PTQ calibration.

## Configuration (`config.yaml`)

- `experiment`: run name, output directory, random seed.
- `dataset`: unified dataset root, label/image dirs, split files, image size (224×224 letterbox), augmentation knobs.
- `model`: detector identifier (`toy_cnn` placeholder—register YOLOv8/YOLOv10/RT-DETR/PP-PicoDet as you integrate them).
- `training`: epochs, batch size, AdamW hyperparameters, AMP/EMA switches, cosine + warmup schedule, QAT trigger.
- `loss`: unified detection loss (Sigmoid Focal + CIoU + L1) weights.
- `evaluation`: MAP thresholds, score threshold, small-object definition.
- `quantization`: backend, QAT schedule, PTQ calibration options.

Update the paths to match your dataset layout before launching training.

## Running Training

```bash
python train.py --config config.yaml
```

During training:

- Data pipeline performs 224×224 letterbox resize (nearest neighbor), geometric & photometric jitter, motion blur, noise, JPEG compression, cutouts, synthetic rain/fog/shadow stressors.
- Unified loss combines Sigmoid Focal (γ=2, α=0.25) with CIoU + L1 (λbox=2, λL1=1, λcls=1).
- AMP, EMA, gradient clipping, and cosine LR are enabled; QAT automatically activates for the final 30% of epochs (configurable).
- Metrics (`metrics.csv`, `metrics.jsonl`) capture the required fields: losses, MAP@0.50/.50:.95, precision/recall/F1, AP/AR small, FPS, grad norms, LR, GPU memory, EMA/AMP/QAT flags, etc.
- Checkpoints are stored under `runs/<timestamp>_<model>_<run>/checkpoints/` with rotation:
  - `last.pth`
  - `best_map5095.pth`
  - `topk/ckpt_epoch{E:03d}_map5095{AP:.3f}.pth` (top 3 by mAP@0.50:.95)
  - Each checkpoint includes model/EMA states, optimizer, scaler, config snapshot, dataset hash, and git commit.

## INT8 Calibration & Export Stubs

- QAT preparation (`modules.quantization.prepare_qat`) fuses Conv+BN+Act and inserts fake-quant observers once the configured epoch ratio is reached.
- PTQ calibration helper (`modules.quantization.calibrate_int8`) duplicates the model, runs calibration batches, and returns an INT8-quantized copy ready for export.
- Extend `modules/modeling.py` to register real detector backbones and add ONNX/TensorRT export steps once models are integrated.

## Next Steps

1. Register production detector builders (YOLOv8/YOLOv10/YOLOv5, RT-DETR tiny, PP-PicoDet S) inside `modules/modeling.py`.
2. Replace the toy head with TensorRT-friendly implementations and keep channel counts multiple of 8.
3. Enhance `compute_detection_metrics` with full COCO-style evaluation if multi-plate scenarios arise.
4. Integrate export scripts for ONNX + TensorRT plan files (`best.onnx`, `best_int8.plan`) under each run directory.
5. Add CI smoke tests (dataset loader, loss forward, one training/validation step) as the project matures.

This scaffold is intentionally compact—every component can be swapped out as the research pipeline solidifies, while keeping the data/quantization/perf expectations from the plan front and center.
