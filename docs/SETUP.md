# Setup

## Data

Put the datasets at the repo root:

```
coco_dataset/{train,valid,test}/          images and _annotations.coco.json
yolov8_dataset/{train,valid,test}/images/
yolov8_dataset/{train,valid,test}/labels/
```

`python process_dataset.py` moves both folders into `data/raw/` and rebuilds `data/processed/unified/`, with the images in `images/` and a YOLO label file for each in `labels/`.

The split files are `data/splits/train.txt` and `data/splits/val.txt`, one image per line (a file name, with or without its extension, or an absolute path). No script writes them. Without them, the training and validation loaders both use every image in `data/processed/unified/images/`.

## Configuration

`config.yaml` has these sections:

- `experiment`: run name, output folder, seed.
- `dataset`: paths, split files, `image_size` (512x512 here), normalisation and augmentation. `max_targets_per_image` keeps only the largest N plates in each image (1 here; 0 keeps them all).
- `model.name`: `toy_cnn`, `yolov8n`, `yolov5n`, `yolov10n`, `rtdetr_tiny` or `pp_picodet_s`.
- `training`: 120 epochs, batch size 8, AdamW, gradient clipping at 1.0, 3 warm-up epochs then cosine down to 10% of the base rate, AMP, EMA decay 0.9999.
- `loss`, `evaluation`: loss weights, score and NMS thresholds, the IoU thresholds for the metrics.
- `quantization`: backend (`fbgemm` for x86, `qnnpack` for ARM) and when QAT starts (`start_ratio: 0.7`).

The YAML files in `configs/` are not read by any script.

## Outputs

`train.py --device auto|cpu|cuda` picks the device. Each run writes to `runs/<timestamp>_<model>_<run_name>/`:

- `train.log` and `config.json`.
- `metrics.csv` and `metrics.jsonl`, one row per epoch: losses, `map50`, `map5095`, precision, recall, F1, small-box recall, validation images per second, GPU memory, gradient norm, learning rate and the AMP, EMA and QAT flags.
- `checkpoints/last.pth`, `checkpoints/best_map5095.pth`, and the best 3 by `map5095` in `checkpoints/topk/`. Each holds the model and EMA weights, optimiser and scaler state, the config, a hash of the dataset file list and the git commit.

`map50` is the share of plates matched by a prediction at IoU 0.5, after the 0.25 score threshold, and `map5095` is that share averaged over IoU 0.5 to 0.95 in steps of 0.05 (`compute_detection_metrics` in `modules/metrics.py`). Both are recall at fixed IoU thresholds.

`modules/quantization.py` also has `calibrate_int8`, which calibrates a copy of a model on batches from a data loader and returns it converted to INT8.

## Visualising predictions

```bash
python visualize_predictions.py \
  --config config.yaml \
  --checkpoint runs/<timestamp>_<model>_<run>/checkpoints/best_map5095.pth \
  --split val \
  --index 42 \
  --output visuals/sample_42.png
```

- `--use-ema` loads the EMA weights if the checkpoint has them.
- `--score-threshold` sets the lowest score drawn (default 0.25), and `--nms-iou-threshold` overrides the config's NMS IoU.
- `--device auto|cpu|cuda` picks the device (default auto).
- `--index` picks the image; without it one is picked at random with `--seed` (default 42).
- `--show` opens the figure in a window. It is saved only when `--output` is given.

True boxes are drawn in green and predictions in red with their scores. `--split test` reads every image in the folder, since `config.yaml` has no test split file.
