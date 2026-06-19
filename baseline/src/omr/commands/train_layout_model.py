"""Train the layout/keypoint detector on generated layout_v0 data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from omr.layout_training import (
    MASK_CHANNELS,
    LayoutDataset,
    build_layout_model,
    finalize_metric_sums,
    layout_loss,
    mean_channel_metric,
    merge_metric_sums,
    parse_float_list,
    pixel_metrics,
    resolve_path,
)


BASELINE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BASELINE_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net layout detector.")
    parser.add_argument("--dataset-dir", default="data_train/layout_v0")
    parser.add_argument("--output-dir", default="baseline/reports/layout_v0_runs/unet_416")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-width", type=int, default=416)
    parser.add_argument("--image-height", type=int, default=596)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--pos-weights", default="3,30,10,6")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def make_loader(
    dataset: LayoutDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weights: tuple[float, ...],
    dice_weight: float,
    mse_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    progress_every: int = 0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    metric_sums: dict[str, dict[str, float]] = {}
    batch_count = len(loader)

    for batch_index, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = layout_loss(
                logits,
                targets,
                pos_weights=pos_weights,
                dice_weight=dice_weight,
                mse_weight=mse_weight,
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        batch_metrics = pixel_metrics(logits.detach(), targets.detach(), MASK_CHANNELS)
        merge_metric_sums(metric_sums, batch_metrics, weight=batch_size)

        if training and progress_every > 0 and (
            batch_index == 1 or batch_index % progress_every == 0 or batch_index == batch_count
        ):
            running_metrics = finalize_metric_sums(metric_sums, total_weight=total_items)
            print(
                f"epoch={epoch:03d} batch={batch_index}/{batch_count} "
                f"loss={total_loss / max(total_items, 1):.4f} "
                f"mean_dice={mean_channel_metric(running_metrics, 'dice'):.4f}",
                flush=True,
            )

    metrics = finalize_metric_sums(metric_sums, total_weight=total_items)
    return {
        "loss": total_loss / max(total_items, 1),
        "pixel": metrics,
        "mean_dice": mean_channel_metric(metrics, "dice"),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(data: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def checkpoint_data(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "args": vars(args),
        "image_size": [args.image_width, args.image_height],
        "mask_channels": list(MASK_CHANNELS),
        "model_config": {
            "base_channels": args.base_channels,
            "out_channels": len(MASK_CHANNELS),
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    dataset_dir = resolve_path(args.dataset_dir, REPO_ROOT)
    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    prepare_output_dir(output_dir, args.overwrite)

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("--image-width and --image-height must be positive")

    pos_weights = parse_float_list(args.pos_weights, len(MASK_CHANNELS))
    image_size = (args.image_width, args.image_height)
    device = choose_device(args.device)

    train_dataset = LayoutDataset(
        dataset_dir,
        split="train",
        image_size=image_size,
        limit=args.limit_train,
    )
    val_dataset = LayoutDataset(
        dataset_dir,
        split="val",
        image_size=image_size,
        limit=args.limit_val,
    )
    if not train_dataset or not val_dataset:
        raise SystemExit("dataset must contain non-empty train and val splits")

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        device=device,
    )

    model = build_layout_model(
        out_channels=len(MASK_CHANNELS),
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "mask_channels": list(MASK_CHANNELS),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "pos_weights": pos_weights,
    }
    save_json(run_config, output_dir / "args.json")

    print(f"dataset_dir={dataset_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"cuda_name={torch.cuda.get_device_name(0)}", flush=True)
    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}", flush=True)
    print(f"image_size={image_size} batch_size={args.batch_size}", flush=True)

    best_score = -1.0
    best_epoch = 0
    history_path = output_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        print(f"starting_epoch={epoch:03d}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            pos_weights=pos_weights,
            dice_weight=args.dice_weight,
            mse_weight=args.mse_weight,
            optimizer=optimizer,
            epoch=epoch,
            progress_every=args.progress_every,
        )
        print(f"validating_epoch={epoch:03d}", flush=True)
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                pos_weights=pos_weights,
                dice_weight=args.dice_weight,
                mse_weight=args.mse_weight,
                optimizer=None,
            )

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        append_jsonl(record, history_path)
        save_json(record, output_dir / "last_metrics.json")

        checkpoint = checkpoint_data(
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            metrics=record,
        )
        torch.save(checkpoint, output_dir / "last_model.pt")

        score = float(val_metrics["mean_dice"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "best_model.pt")
            save_json(record, output_dir / "best_metrics.json")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['mean_dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['mean_dice']:.4f} "
            f"best_epoch={best_epoch:03d} best_val_dice={best_score:.4f}",
            flush=True,
        )

    print(f"best_model={output_dir / 'best_model.pt'}", flush=True)
    return 0
