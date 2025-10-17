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
visualize_predictions.py  # Utility to compare ground-truth and predicted boxes on a sample image
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

`matplotlib` is included in the requirements list for the visualization helper; install it via the command above if you plan to generate overlay figures.

## Data Preparation

1. Download and extract the provided `coco_dataset/` and `yolov8_dataset/` folders at the repo root (as supplied).
2. Run the dataset prep script (no arguments needed; it knows the folder layout):
   ```bash
   python process_dataset.py
   ```
   This moves the raw datasets to `data/raw/{coco_dataset,yolov8_dataset}/` and builds the unified YOLO-format dataset under `data/processed/unified/`.
3. Define reproducible splits (`data/splits/train.txt`, `data/splits/val.txt`) with one image path per line.
4. Reserve 200-500 in-distribution images (no augmentations) under `data/calibration/` for PTQ calibration.

## Configuration (`config.yaml`)

- `experiment`: run name, output directory, random seed.
- `dataset`: unified dataset root, label/image dirs, split files, image size (224x224 letterbox), augmentation knobs.
- `dataset.max_targets_per_image`: limit the number of ground-truth plates used per image (default `1` to focus on the most prominent plate; set to `0`/omit for full multi-plate detection).
- `model`: detector identifier (available keys: `toy_cnn`, `yolov8n`, `yolov5n`, `yolov10n`, `rtdetr_tiny`, `pp_picodet_s`).
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

- Data pipeline performs 224x224 letterbox resize (nearest neighbor), geometric & photometric jitter, motion blur, noise, JPEG compression, cutouts, synthetic rain/fog/shadow stressors.
- Dense detection head shared by all backbones predicts one objectness logit and a normalized box per spatial cell; Unified loss combines Sigmoid Focal (gamma=2, alpha=0.25) with CIoU + L1 (lambda_box=2, lambda_L1=1, lambda_cls=1) on the cells that host the selected plates.
- AMP, EMA, gradient clipping, and cosine LR are enabled; QAT automatically activates for the final 30% of epochs (configurable).
- Metrics (`metrics.csv`, `metrics.jsonl`) capture the required fields: losses, MAP@0.50 / 0.50:0.95, precision/recall/F1, AP/AR small, FPS, grad norms, LR, GPU memory, EMA/AMP/QAT flags, etc.
- Checkpoints are stored under `runs/<timestamp>_<model>_<run>/checkpoints/` with rotation:
  - `last.pth`
  - `best_map5095.pth`
  - `topk/ckpt_epoch{E:03d}_map5095{AP:.3f}.pth` (top 3 by mAP@0.50:0.95)
  - Each checkpoint includes model/EMA states, optimizer, scaler, config snapshot, dataset hash, and git commit.

## INT8 Calibration & Export Stubs

- QAT preparation (`modules.quantization.prepare_qat`) fuses Conv+BN+Act and inserts fake-quant observers once the configured epoch ratio is reached.
  - `nn.SiLU` activations currently fall back to Conv+BN fusion because PyTorch's fuser lacks a Conv+BN+SiLU recipe; this still permits QAT on the stem blocks.
  - Ensure your PyTorch build was compiled with the requested quantized backend (`fbgemm` on x86, `qnnpack` on ARM); otherwise setting `quantization.backend` will raise an error.
- PTQ calibration helper (`modules.quantization.calibrate_int8`) duplicates the model, runs calibration batches, and returns an INT8-quantized copy ready for export.
- Extend `modules/modeling.py` to register real detector backbones and add ONNX/TensorRT export steps once models are integrated.

## Prediction Visualization

Save a checkpoint from training (for example, `runs/<timestamp>_<model>_<run>/checkpoints/best_map5095.pth`) and run:

```bash
python visualize_predictions.py \
  --config config.yaml \
  --checkpoint runs/<timestamp>_<model>_<run>/checkpoints/best_map5095.pth \
  --split val \
  --index 42 \
  --output visuals/sample_42.png
```

Key options:

- `--use-ema` loads EMA weights if the checkpoint contains them.
- `--score-threshold` controls which predictions are displayed (default 0.25).
- `--device auto|cpu|cuda` chooses the inference device (defaults to auto-detect).
- `--show` opens the figure in a window instead of only saving it.

The script denormalizes the 224×224 letterboxed image, overlays green ground-truth boxes and red predicted boxes (with confidence scores), and writes the result to the path supplied with `--output`.

## Next Steps

1. Extend the new backbones with full detector heads (anchor grids, decoder, post-processing) to move beyond the single-box scaffold.
2. Replace the simple classification/regression heads with TensorRT-friendly modules and keep channel counts multiples of 8.
3. Enhance `compute_detection_metrics` with full COCO-style evaluation if multi-plate scenarios arise.
4. Integrate export scripts for ONNX + TensorRT plan files (`best.onnx`, `best_int8.plan`) under each run directory.
5. Add CI smoke tests (dataset loader, loss forward, one training/validation step) as the project matures.

This scaffold is intentionally compact—every component can be swapped out as the research pipeline solidifies, while keeping the data/quantization/performance expectations from the plan front and center.
