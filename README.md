# UAE License Plate Detector Scaffold

Project scaffolding for a research-oriented detector that targets INT8 deployment on TensorRT-compatible hardware. The focus is on maintainable structure, consistent data processing, and a unified training pipeline that can compare multiple lightweight backbones.

## Repository Layout

```
configs/                 # YAML configs for data, models, training, quantization, exports
data/                    # Placeholder directories for raw, processed, annotations, calibration assets
docs/                    # Design notes for architecture, data pipeline, quantization strategy
notebooks/               # Reserved for exploratory analysis
runs/                    # Training outputs (metrics, checkpoints, exports)
scripts/                 # CLI entry points (train/evaluate/export/calibrate)
src/uaelp_det/           # Package with data, model, training, evaluation, quantization, export modules
tests/                   # Unit/integration test scaffolding
```

## Detector Roadmap

1. Implement dataset loading and augmentation pipeline aligning with `configs/data/dataset.yaml`.
2. Wire model builders into `src/uaelp_det/models/registry.py` for YOLOv8-n, YOLOv5-n, YOLOv10-n, RT-DETR-tiny, PP-PicoDet-S.
3. Flesh out the unified loss and training loop with EMA, AMP, cosine LR schedule, and QAT enablement.
4. Record metrics every epoch (`metrics.csv` / `metrics.jsonl`) and rotate checkpoints under `runs/YYYYMMDD-HHMMSS_<model>_<run_id>/`.
5. Finalize export pipeline: ONNX with Q/DQ nodes, TensorRT INT8 engine build using PTQ/QAT artifacts.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

Populate `data/` with raw images and annotations before implementing loaders. Update `pyproject.toml` as new dependencies are introduced.

## Testing

Use `pytest` for unit tests once functionality is implemented:

```bash
pytest
```

Additional integration tests can be added under `tests/integration/` as training workflows mature.
