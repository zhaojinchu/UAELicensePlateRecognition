# Data Pipeline Notes

## Dataset Overview

- ~12.9k auto-oriented vehicle images, pre-letterboxed to 1440x1280.
- Dominant scenario: single UAE plate, variable lighting/weather.
- Augmentation aims to synthesize night/rain/glare, motion blur, occlusions.

## Preprocessing Targets

- Letterbox resize to 224x224 without stretching; pad to maintain aspect.
- Normalize using ImageNet statistics; maintain NCHW layout.
- Keep transforms consistent between training, validation (no augmentation), and calibration.

## Augmentation Blocks

- **Geometric:** random scale (0.5-1.5×), aspect (0.9-1.1), translate (±10%), rotate (±7°), shear (±5°), perspective (±0.02).
- **Photometric:** brightness/contrast ±0.2, HSV jitters (H ±5°, S/V ±0.2), color temperature shift.
- **Degradations:** motion blur (p=0.2, k=3-7), Gaussian blur σ≤1.2, Gaussian noise σ≤0.01, JPEG 40-90, sharpening.
- **Occlusion:** small cutouts, synthetic shadows/reflections, edge occlusions; mosaic (≤2 images) only with min-plate safeguards.

## Splits & Calibration

- Maintain `data/splits/{train,val,test}.txt` for reproducible splits.
- Reserve 200-500 samples (balanced conditions) for PTQ calibration.
- Persist processed artifacts in `data/processed/` for reuse (cache, edge-case augment sets).
