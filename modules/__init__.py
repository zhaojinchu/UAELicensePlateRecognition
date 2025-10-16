"""Minimal module collection for the UAE license plate detector scaffold."""

from .data import (
    MergeConfig,
    create_dataloader,
    merge_coco_yolo_datasets,
    prepare_transforms,
    resolve_dataset_paths,
)
from .losses import UnifiedDetectionLoss
from .metrics import DetectionMetricsLogger, compute_detection_metrics
from .modeling import build_model
from .quantization import calibrate_int8, prepare_qat
from .utils import (
    AverageMeter,
    ModelEMA,
    atomic_save,
    compute_dataset_id,
    compute_grad_norm,
    create_logger,
    ensure_dir,
    get_git_commit,
    load_yaml,
    save_config,
    set_seed,
)

__all__ = [
    "MergeConfig",
    "merge_coco_yolo_datasets",
    "prepare_transforms",
    "resolve_dataset_paths",
    "create_dataloader",
    "build_model",
    "prepare_qat",
    "calibrate_int8",
    "UnifiedDetectionLoss",
    "DetectionMetricsLogger",
    "compute_detection_metrics",
    "AverageMeter",
    "ModelEMA",
    "compute_dataset_id",
    "compute_grad_norm",
    "get_git_commit",
    "save_config",
    "atomic_save",
    "load_yaml",
    "create_logger",
    "ensure_dir",
    "set_seed",
]
