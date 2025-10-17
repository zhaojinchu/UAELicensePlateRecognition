"""Training entrypoint following the UAE detector research plan."""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from modules import (
    AverageMeter,
    DetectionMetricsLogger,
    ModelEMA,
    UnifiedDetectionLoss,
    atomic_save,
    build_model,
    compute_dataset_id,
    compute_grad_norm,
    create_dataloader,
    create_logger,
    ensure_dir,
    get_git_commit,
    load_yaml,
    save_config,
    set_seed,
)
from modules.metrics import compute_detection_metrics
from modules.quantization import prepare_qat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UAE license plate detector scaffold.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--device", type=str, default="auto", help="Force device (auto|cpu|cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    set_seed(int(config.get("experiment", {}).get("seed", 42)))

    model_cfg = config["model"]
    training_cfg = config["training"]
    loss_cfg = config.get("loss", {})
    eval_cfg = config.get("evaluation", {})
    quant_cfg = config.get("quantization", {})
    dataset_cfg = config["dataset"]
    max_targets_per_image = dataset_cfg.get("max_targets_per_image", 1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = config.get("experiment", {}).get("run_name", "run")
    run_dir = ensure_dir(Path(config.get("experiment", {}).get("output_dir", "runs")) / f"{timestamp}_{model_cfg.get('name', 'model')}_{run_name}")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    ensure_dir(checkpoints_dir / "topk")

    logger = create_logger("train", run_dir / "train.log")
    logger.info("Starting run '%s'", run_dir.name)
    save_config(config, run_dir / "config.json")

    device = select_device(args.device)
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        device_desc = f"CUDA:{device_index} ({torch.cuda.get_device_name(device_index)})"
    else:
        device_desc = device.type.upper()
    logger.info("Using device: %s", device_desc)

    pin_memory = device.type == "cuda"

    train_loader = create_dataloader(
        dataset_cfg,
        train=True,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        dataset_cfg,
        train=False,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )

    dataset_id = compute_dataset_id(
        list(getattr(train_loader.dataset, "items", []))
        + list(getattr(val_loader.dataset, "items", []))
    )
    class_names = dataset_cfg.get("class_names", ["license_plate"])

    model = build_model(model_cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    loss_fn = UnifiedDetectionLoss(loss_cfg)

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
    )

    total_epochs = training_cfg["epochs"]
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    final_lr_ratio = training_cfg.get("cosine_final_ratio", 0.1)

    amp_enabled = bool(training_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    ema = ModelEMA(model, decay=training_cfg.get("ema_decay", 0.9999)) if training_cfg.get("ema_decay") else None
    ema_on = ema is not None

    qat_enabled = bool(quant_cfg.get("qat", {}).get("enable", False))
    qat_start_ratio = quant_cfg.get("qat", {}).get("start_ratio")
    if qat_start_ratio is None:
        qat_epoch = training_cfg.get("qat_start_epoch", int(0.7 * total_epochs))
        qat_start_ratio = qat_epoch / max(1, total_epochs)
    qat_backend = quant_cfg.get("backend", "fbgemm")
    qat_prepared = False

    metrics_logger = DetectionMetricsLogger(
        csv_path=run_dir / "metrics.csv",
        jsonl_path=run_dir / "metrics.jsonl",
    )

    git_commit = get_git_commit(Path.cwd())
    run_id = run_dir.name

    topk: List[Tuple[float, Path]] = []
    topk_limit = 3
    best_map5095 = -1.0
    global_step = 0

    logger.info(
        "Model params: %.2fM | Size: %.2f MB | Train batches: %d | Val batches: %d",
        num_params / 1e6,
        model_size_mb,
        len(train_loader),
        len(val_loader),
    )

    for epoch in range(total_epochs):
        epoch_start = datetime.now(timezone.utc).isoformat()

        lr = adjust_learning_rate(
            optimizer,
            base_lr=training_cfg["lr"],
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            final_ratio=final_lr_ratio,
        )

        if qat_enabled and not qat_prepared and (epoch + 1) / total_epochs >= qat_start_ratio:
            logger.info("Enabling QAT (backend=%s) at epoch %d", qat_backend, epoch + 1)
            model = prepare_qat(model, {"backend": qat_backend})
            if ema_on:
                ema = ModelEMA(model, decay=training_cfg.get("ema_decay", 0.9999))
            qat_prepared = True

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_stats = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=training_cfg.get("grad_clip_norm"),
            ema=ema,
            max_targets=max_targets_per_image,
        )
        global_step += train_stats["steps"]

        val_stats, pred_targets = validate_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            ema=ema,
            max_targets=max_targets_per_image,
        )

        metrics = compute_detection_metrics(
            predictions=pred_targets["predictions"],
            targets=pred_targets["targets"],
            thresholds=eval_cfg.get(
                "map_thresholds",
                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            ),
            score_threshold=eval_cfg.get("score_threshold", 0.25),
            small_area_threshold=eval_cfg.get("small_box_area", 0.05),
        )

        gpu_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        epoch_end = datetime.now(timezone.utc).isoformat()

        record = {
            "run_id": run_id,
            "time_epoch_start": epoch_start,
            "time_epoch_end": epoch_end,
            "epoch": epoch + 1,
            "steps": train_stats["steps"],
            "lr": round(lr, 8),
            "train_loss": round(train_stats["losses"]["total"], 6),
            "train_box": round(train_stats["losses"]["box"], 6),
            "train_cls": round(train_stats["losses"]["cls"], 6),
            "train_dfl": round(train_stats["losses"]["dfl"], 6),
            "val_loss": round(val_stats["losses"]["total"], 6),
            "val_box": round(val_stats["losses"]["box"], 6),
            "val_cls": round(val_stats["losses"]["cls"], 6),
            "val_dfl": round(val_stats["losses"]["dfl"], 6),
            "map50": round(metrics["map50"], 6),
            "map5095": round(metrics["map5095"], 6),
            "ap_small": round(metrics["ap_small"], 6),
            "ar_small": round(metrics["ar_small"], 6),
            "fps": round(val_stats["fps"], 2),
            "gpu_mem_mb": round(gpu_mem_mb, 2),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "num_params": num_params,
            "model_size_mb": round(model_size_mb, 2),
            "grad_norm": round(train_stats["grad_norm"], 6),
            "ema_on": int(ema_on),
            "amp_on": int(amp_enabled),
            "qat_on": int(qat_prepared),
        }
        metrics_logger.log(record)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f map50=%.4f map5095=%.4f",
            epoch + 1,
            total_epochs,
            train_stats["losses"]["total"],
            val_stats["losses"]["total"],
            metrics["map50"],
            metrics["map5095"],
        )

        checkpoint_state = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict() if ema_on else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled and scaler is not None else None,
            "best_map5095": best_map5095,
            "model_name": model_cfg.get("name"),
            "git_commit": git_commit,
            "dataset_id": dataset_id,
            "class_names": class_names,
            "config": config,
            "qat_active": qat_prepared,
        }
        atomic_save(checkpoint_state, checkpoints_dir / "last.pth")

        if metrics["map5095"] > best_map5095:
            best_map5095 = metrics["map5095"]
            checkpoint_state["best_map5095"] = best_map5095
            atomic_save(checkpoint_state, checkpoints_dir / "best_map5095.pth")

        manage_topk_checkpoints(
            topk=topk,
            topk_limit=topk_limit,
            new_score=metrics["map5095"],
            state=checkpoint_state,
            checkpoints_dir=checkpoints_dir,
            epoch=epoch + 1,
        )

    logger.info("Training finished. Best mAP@0.50:0.95 = %.4f", max(best_map5095, 0.0))


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    final_ratio: float,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = (base_lr - base_lr * final_ratio) * cosine + base_lr * final_ratio

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: UnifiedDetectionLoss,
    scaler: GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float | None,
    ema: ModelEMA | None,
    *,
    max_targets: Optional[int],
) -> Dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_l1 = 0.0
    total_dfl = 0.0
    grad_norm_meter = AverageMeter()
    data_meter = AverageMeter()
    step_meter = AverageMeter()

    start_time = time.perf_counter()
    last_time = start_time
    steps = 0

    for images, targets in tqdm(data_loader, desc="Train", leave=False):
        data_meter.update(time.perf_counter() - last_time)
        batch_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            cls_logits = outputs["cls_logits"]
            box_preds = outputs["boxes"]
            cls_targets, box_targets, positive_mask = prepare_targets(
                targets,
                feature_shape=cls_logits.shape,
                device=device,
                max_targets=max_targets,
            )
            losses = loss_fn(cls_logits, box_preds, cls_targets, box_targets, positive_mask)
            loss = losses["total"]

        if amp_enabled:
            if scaler is None:
                raise RuntimeError("AMP is enabled but GradScaler is not initialized.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model.parameters())
            if grad_clip is not None:
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip).item()
            else:
                grad_norm = compute_grad_norm(model.parameters())
            optimizer.step()

        if ema is not None:
            ema.update(model)

        grad_norm_meter.update(float(grad_norm))
        total_loss += float(losses["total"].item())
        total_cls += float(losses["cls"].item())
        total_box += float(losses["box"].item())
        total_l1 += float(losses["l1"].item())
        total_dfl += float(losses["dfl"].item())
        steps += 1

        step_meter.update(time.perf_counter() - batch_start)
        last_time = time.perf_counter()

    epoch_time = time.perf_counter() - start_time
    throughput = len(data_loader.dataset) / epoch_time if len(data_loader.dataset) else 0.0

    return {
        "losses": {
            "total": total_loss / max(1, steps),
            "cls": total_cls / max(1, steps),
            "box": total_box / max(1, steps),
            "l1": total_l1 / max(1, steps),
            "dfl": total_dfl / max(1, steps),
        },
        "grad_norm": grad_norm_meter.avg,
        "steps": steps,
        "epoch_time": epoch_time,
        "throughput": throughput,
        "data_time": data_meter.avg,
        "step_time": step_meter.avg,
    }


def validate_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    loss_fn: UnifiedDetectionLoss,
    device: torch.device,
    amp_enabled: bool,
    ema: ModelEMA | None,
    *,
    max_targets: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, torch.Tensor]]]]:
    context = ema.apply_shadow(model) if ema is not None else contextlib.nullcontext()
    predictions: List[Dict[str, torch.Tensor]] = []
    targets_list: List[Dict[str, torch.Tensor]] = []

    with context, torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_cls = 0.0
        total_box = 0.0
        total_l1 = 0.0
        total_dfl = 0.0
        steps = 0

        start_time = time.perf_counter()

        for images, targets in tqdm(data_loader, desc="Validate", leave=False):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                cls_logits = outputs["cls_logits"]
                box_preds = outputs["boxes"]
                cls_targets, box_targets, positive_mask = prepare_targets(
                    targets,
                    feature_shape=cls_logits.shape,
                    device=device,
                    max_targets=max_targets,
                )
                losses = loss_fn(cls_logits, box_preds, cls_targets, box_targets, positive_mask)

            total_loss += float(losses["total"].item())
            total_cls += float(losses["cls"].item())
            total_box += float(losses["box"].item())
            total_l1 += float(losses["l1"].item())
            total_dfl += float(losses["dfl"].item())
            steps += 1

            batch_scores, batch_boxes = flatten_predictions(torch.sigmoid(cls_logits), box_preds)

            for idx in range(images.size(0)):
                predictions.append(
                    {
                        "scores": batch_scores[idx].detach().cpu(),
                        "boxes": batch_boxes[idx].detach().cpu(),
                    }
                )
                filtered_boxes, filtered_labels = select_prominent_targets(
                    targets[idx],
                    max_targets=max_targets,
                )
                targets_list.append(
                    {
                        "boxes": filtered_boxes.detach().cpu(),
                        "labels": filtered_labels.detach().cpu(),
                    }
                )

        epoch_time = time.perf_counter() - start_time
        throughput = len(data_loader.dataset) / epoch_time if len(data_loader.dataset) else 0.0

    losses = {
        "total": total_loss / max(1, steps),
        "cls": total_cls / max(1, steps),
        "box": total_box / max(1, steps),
        "l1": total_l1 / max(1, steps),
        "dfl": total_dfl / max(1, steps),
    }
    return (
        {"losses": losses, "epoch_time": epoch_time, "fps": throughput},
        {"predictions": predictions, "targets": targets_list},
    )


def prepare_targets(
    targets: List[Dict[str, torch.Tensor]],
    *,
    feature_shape: torch.Size,
    device: torch.device,
    max_targets: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = feature_shape
    cls_targets = torch.zeros((batch_size, height, width), device=device)
    box_targets = torch.zeros((batch_size, 4, height, width), device=device)
    positive_mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=device)

    for batch_index, target in enumerate(targets):
        boxes_cpu, _ = select_prominent_targets(target, max_targets=max_targets)
        boxes = boxes_cpu.to(device)

        if boxes.numel() == 0:
            continue

        for box in boxes:
            cx, cy, w, h = box
            grid_x = int(torch.clamp(cx * width, 0, width - 1).item())
            grid_y = int(torch.clamp(cy * height, 0, height - 1).item())
            cls_targets[batch_index, grid_y, grid_x] = 1.0
            box_targets[batch_index, :, grid_y, grid_x] = box
            positive_mask[batch_index, grid_y, grid_x] = True

    return cls_targets, box_targets, positive_mask


def flatten_predictions(
    scores: torch.Tensor,
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = scores.size(0)
    if scores.ndim == 4:
        scores = scores.squeeze(1)
    flat_scores = scores.view(batch_size, -1)
    flat_boxes = boxes.view(batch_size, 4, -1).permute(0, 2, 1)
    return flat_scores, flat_boxes


def select_prominent_targets(
    target: Dict[str, torch.Tensor],
    *,
    max_targets: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes = target["boxes"]
    labels = target["labels"]
    if boxes.numel() == 0 or not max_targets or max_targets <= 0:
        return boxes, labels

    areas = boxes[:, 2] * boxes[:, 3]
    k = min(max_targets, boxes.size(0))
    topk = torch.topk(areas, k=k, largest=True).indices
    return boxes[topk], labels[topk]


def manage_topk_checkpoints(
    topk: List[Tuple[float, Path]],
    topk_limit: int,
    new_score: float,
    state: Dict[str, Any],
    checkpoints_dir: Path,
    epoch: int,
) -> None:
    topk_dir = ensure_dir(checkpoints_dir / "topk")
    checkpoint_path = topk_dir / f"ckpt_epoch{epoch:03d}_map5095{new_score:.3f}.pth"
    atomic_save(state, checkpoint_path)
    topk.append((new_score, checkpoint_path))
    topk.sort(key=lambda item: item[0], reverse=True)

    while len(topk) > topk_limit:
        _, path = topk.pop()
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
