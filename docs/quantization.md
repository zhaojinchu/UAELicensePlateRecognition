# Quantization Strategy

## Goals

- Deliver INT8 detectors optimized for Jetson-class and TensorRT backends.
- Maintain accuracy parity with FP16 baselines via late-stage QAT.

## Workflow

1. **FP32/FP16 Training:** Run baseline training with AMP for faster convergence.
2. **QAT Phase:** Enable fake-quant modules for the final 30% epochs to adapt weights.
3. **Checkpointing:** Save both FP checkpoint (`last.pth`) and QAT-aware best weights.
4. **Export:** Generate ONNX with Q/DQ nodes or clean FP graph for PTQ calibration.
5. **Calibration:** For PTQ, run calibration loader (200-500 samples) to produce TensorRT cache (`*.cache`).
6. **Engine Build:** Fuse layers, ensure channels multiple of 8, export INT8 engine (`*.plan`).

## Implementation Notes

- Fused Conv+BN+Act layers required prior to export.
- Align preprocess pipeline between training and calibration (letterbox, normalization).
- Maintain calibration manifests in `data/calibration/`.
