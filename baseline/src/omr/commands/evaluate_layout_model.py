"""Evaluate a trained layout/keypoint detector."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from omr.layout_training import (
    MASK_CHANNELS,
    LayoutDataset,
    build_layout_model,
    denormalize_image,
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
    parser = argparse.ArgumentParser(description="Evaluate a trained layout detector checkpoint.")
    parser.add_argument("--dataset-dir", default="data_train/layout_v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="baseline/reports/layout_v0_eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pos-weights", default="3,30,10,6")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--marker-threshold", type=float, default=0.35)
    parser.add_argument("--bubble-threshold", type=float, default=0.25)
    parser.add_argument("--marker-radius", type=float, default=6.0)
    parser.add_argument("--bubble-radius", type=float, default=5.0)
    parser.add_argument("--marker-nms-radius", type=int, default=4)
    parser.add_argument("--bubble-nms-radius", type=int, default=3)
    parser.add_argument("--max-marker-peaks", type=int, default=30)
    parser.add_argument("--max-bubble-peaks", type=int, default=900)
    parser.add_argument("--visual-limit", type=int, default=12)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_peaks(
    probability: np.ndarray,
    *,
    threshold: float,
    nms_radius: int,
    max_peaks: int,
) -> list[tuple[float, float, float]]:
    kernel_size = max(1, nms_radius * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(probability.astype(np.float32), kernel)
    mask = (probability >= threshold) & (probability >= local_max - 1e-6)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    scores = probability[ys, xs]
    order = np.argsort(-scores)
    peaks: list[tuple[float, float, float]] = []
    for index in order[:max_peaks]:
        peaks.append((float(xs[index]), float(ys[index]), float(scores[index])))
    return peaks


def scaled_points(label: dict[str, Any], *, kind: str, output_size: tuple[int, int]) -> np.ndarray:
    source_width = float(label["image"]["width"])
    source_height = float(label["image"]["height"])
    output_width, output_height = output_size
    scale_x = output_width / source_width
    scale_y = output_height / source_height

    if kind == "markers":
        records = [item for item in label["markers"] if item.get("visible")]
    elif kind == "bubbles":
        records = [item for item in label["bubbles"] if item.get("visible")]
    else:
        raise ValueError(f"unknown point kind: {kind}")

    return np.array(
        [[float(item["x"]) * scale_x, float(item["y"]) * scale_y] for item in records],
        dtype=np.float32,
    )


def match_peaks(
    peaks: list[tuple[float, float, float]],
    gt_points: np.ndarray,
    *,
    radius: float,
) -> dict[str, float]:
    if len(gt_points) == 0:
        return {
            "tp": 0.0,
            "fp": float(len(peaks)),
            "fn": 0.0,
            "distance_sum": 0.0,
        }

    used = np.zeros((len(gt_points),), dtype=bool)
    true_positive = 0
    false_positive = 0
    distance_sum = 0.0

    for x, y, _ in peaks:
        distances = np.linalg.norm(gt_points - np.array([[x, y]], dtype=np.float32), axis=1)
        distances[used] = np.inf
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance <= radius:
            used[best_index] = True
            true_positive += 1
            distance_sum += best_distance
        else:
            false_positive += 1

    false_negative = int((~used).sum())
    return {
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "distance_sum": float(distance_sum),
    }


def finalize_peak_metrics(raw: dict[str, float], *, threshold: float, radius: float) -> dict[str, float]:
    tp = raw.get("tp", 0.0)
    fp = raw.get("fp", 0.0)
    fn = raw.get("fn", 0.0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": threshold,
        "match_radius_px": radius,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance_px": raw.get("distance_sum", 0.0) / tp if tp > 0 else None,
    }


def add_peak_sums(total: dict[str, float], batch: dict[str, float]) -> None:
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + float(value)


def draw_prediction_overlay(
    *,
    image_tensor: torch.Tensor,
    probabilities: np.ndarray,
    sample_id: str,
    output_dir: Path,
    marker_threshold: float,
    bubble_threshold: float,
    marker_nms_radius: int,
    bubble_nms_radius: int,
) -> None:
    image_rgb = denormalize_image(image_tensor)
    overlay = image_rgb.copy()
    page = probabilities[0] >= 0.5
    grid = probabilities[3] >= 0.5
    overlay[page] = (overlay[page] * 0.82 + np.array([40, 90, 255]) * 0.18).astype(np.uint8)
    overlay[grid] = (255, 40, 40)

    marker_peaks = extract_peaks(
        probabilities[1],
        threshold=marker_threshold,
        nms_radius=marker_nms_radius,
        max_peaks=30,
    )
    bubble_peaks = extract_peaks(
        probabilities[2],
        threshold=bubble_threshold,
        nms_radius=bubble_nms_radius,
        max_peaks=900,
    )
    for x, y, _ in bubble_peaks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (30, 220, 80), 1, lineType=cv2.LINE_AA)
    for x, y, _ in marker_peaks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 5, (255, 220, 0), 2, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(image_rgb, 0.68, overlay, 0.32, 0)
    output_path = output_dir / "visuals" / f"{sample_id}_pred_overlay.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main() -> int:
    args = parse_args()
    dataset_dir = resolve_path(args.dataset_dir, REPO_ROOT)
    checkpoint_path = resolve_path(args.checkpoint, REPO_ROOT)
    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    prepare_output_dir(output_dir, args.overwrite)

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    checkpoint_size = checkpoint.get("image_size", [416, 596])
    image_width = args.image_width or int(checkpoint_size[0])
    image_height = args.image_height or int(checkpoint_size[1])
    base_channels = args.base_channels or int(config.get("base_channels", 24))
    image_size = (image_width, image_height)

    model = build_layout_model(
        out_channels=int(config.get("out_channels", len(MASK_CHANNELS))),
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = LayoutDataset(
        dataset_dir,
        split=args.split,
        image_size=image_size,
        limit=args.limit,
    )
    if not dataset:
        raise SystemExit(f"dataset split is empty: {args.split}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    pos_weights = parse_float_list(args.pos_weights, len(MASK_CHANNELS))
    total_loss = 0.0
    total_items = 0
    metric_sums: dict[str, dict[str, float]] = {}
    marker_sums: dict[str, float] = {}
    bubble_sums: dict[str, float] = {}
    visual_count = 0

    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"dataset_dir={dataset_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"split={args.split} samples={len(dataset)} image_size={image_size}", flush=True)
    print(f"device={device}", flush=True)

    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = layout_loss(
                logits,
                targets,
                pos_weights=pos_weights,
                dice_weight=args.dice_weight,
                mse_weight=args.mse_weight,
            )
            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            batch_metrics = pixel_metrics(logits, targets, MASK_CHANNELS)
            merge_metric_sums(metric_sums, batch_metrics, weight=batch_size)

            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            label_paths = meta["label_path"]
            sample_ids = meta["sample_id"]
            for item_index, label_path in enumerate(label_paths):
                label = load_json(Path(label_path))
                marker_gt = scaled_points(label, kind="markers", output_size=image_size)
                bubble_gt = scaled_points(label, kind="bubbles", output_size=image_size)

                marker_peaks = extract_peaks(
                    probabilities[item_index, 1],
                    threshold=args.marker_threshold,
                    nms_radius=args.marker_nms_radius,
                    max_peaks=args.max_marker_peaks,
                )
                bubble_peaks = extract_peaks(
                    probabilities[item_index, 2],
                    threshold=args.bubble_threshold,
                    nms_radius=args.bubble_nms_radius,
                    max_peaks=args.max_bubble_peaks,
                )
                add_peak_sums(
                    marker_sums,
                    match_peaks(marker_peaks, marker_gt, radius=args.marker_radius),
                )
                add_peak_sums(
                    bubble_sums,
                    match_peaks(bubble_peaks, bubble_gt, radius=args.bubble_radius),
                )

                if visual_count < args.visual_limit:
                    draw_prediction_overlay(
                        image_tensor=images[item_index].detach().cpu(),
                        probabilities=probabilities[item_index],
                        sample_id=sample_ids[item_index],
                        output_dir=output_dir,
                        marker_threshold=args.marker_threshold,
                        bubble_threshold=args.bubble_threshold,
                        marker_nms_radius=args.marker_nms_radius,
                        bubble_nms_radius=args.bubble_nms_radius,
                    )
                    visual_count += 1

    pixel = finalize_metric_sums(metric_sums, total_weight=total_items)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "samples": total_items,
        "image_size": [image_width, image_height],
        "loss": total_loss / max(total_items, 1),
        "pixel": pixel,
        "mean_dice": mean_channel_metric(pixel, "dice"),
        "peaks": {
            "markers": finalize_peak_metrics(
                marker_sums,
                threshold=args.marker_threshold,
                radius=args.marker_radius,
            ),
            "bubbles": finalize_peak_metrics(
                bubble_sums,
                threshold=args.bubble_threshold,
                radius=args.bubble_radius,
            ),
        },
        "visuals_dir": str(output_dir / "visuals") if args.visual_limit > 0 else None,
    }
    save_json(metrics, output_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0
