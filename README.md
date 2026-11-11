# UAELicensePlateRecognition

PyTorch training code for a one-class licence plate detector on UAE images. It has 6 small backbones that share one dense detection head, and it trains with mixed precision, EMA weights and quantisation-aware training (QAT) over the last 30% of epochs.

## How it works

- The backbones in `modules/modeling.py` are written after the architectures they are named for (`yolov8n`, `yolov5n`, `yolov10n`, `rtdetr_tiny`, `pp_picodet_s`), plus a 4-convolution `toy_cnn`. They train from random initialisation.
- Images are letterboxed to 512x512 (`image_size` in `config.yaml`).
- Each backbone gives one feature map at stride 8, and the shared head predicts, for every cell, an objectness logit and a box (centre, width and height, scaled 0 to 1).
- The loss is sigmoid focal on objectness, plus CIoU and L1 on the boxes at the cells near each plate's centre.
- From epoch 84 of 120 (70% of training in `config.yaml`), `prepare_qat` fuses Conv+BN layers and inserts fake-quant observers.

## Run it

The datasets are gitignored, so put `coco_dataset/` and `yolov8_dataset/` at the repo root first. The layout is in [docs/SETUP.md](docs/SETUP.md).

```bash
pip install -r requirements.txt
python process_dataset.py
python train.py --config config.yaml
python visualize_predictions.py --checkpoint runs/<timestamp>_<model>_<run_name>/checkpoints/best_map5095.pth --output visuals/sample.png
```

`process_dataset.py` merges both datasets into `data/processed/unified/` in YOLO format. Without split files in `data/splits/`, training and validation both use every image there. Each run writes its logs and checkpoints to `runs/<timestamp>_<model>_<run_name>/`.

The config settings, the outputs and the visualiser options are in [docs/SETUP.md](docs/SETUP.md).
