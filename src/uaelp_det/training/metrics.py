"""Metrics logging scaffolding for training and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


CSV_HEADER = (
    "run_id,time_epoch_start,time_epoch_end,epoch,steps,lr,train_loss,train_box,"
    "train_cls,train_dfl,val_loss,val_box,val_cls,val_dfl,map50,map5095,ap_small,"
    "ar_small,fps,gpu_mem_mb,precision,recall,f1,num_params,model_size_mb,grad_norm,"
    "ema_on,amp_on,qat_on"
)


@dataclass
class MetricLoggerConfig:
    """Defines output paths for metric logging."""

    output_dir: Path
    run_id: str


class MetricLogger:
    """Write metrics to CSV and JSONL files for later visualization."""

    def __init__(self, config: MetricLoggerConfig) -> None:
        self.config = config
        self.csv_path = config.output_dir / "metrics.csv"
        self.jsonl_path = config.output_dir / "metrics.jsonl"

    def initialize(self) -> None:
        """Create header row for CSV and ensure directories exist."""
        raise NotImplementedError("Implement metrics file initialization.")

    def log_epoch(self, metrics: Dict[str, float]) -> None:
        """Append epoch metrics to CSV and JSONL outputs."""
        _ = metrics
        raise NotImplementedError("Implement metrics logging.")
