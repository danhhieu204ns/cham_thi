"""Run layout_v0 detector inference on answer-sheet images."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from omr.layout_training import (
    MASK_CHANNELS,
    build_layout_model,
    extract_heatmap_peaks,
    resolve_path,
)


BASELINE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BASELINE_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained layout_v0 detector on scans.")
    parser.add_argument(
        "--checkpoint",
        default="baseline/reports/layout_v0_runs/unet_v2/best_model.pt",
    )
    parser.add_argument("--metadata", default="baseline/data/labels/sheets.jsonl")
    parser.add_argument(
        "--image-list",
        default="",
        help="Optional text file of image ids or tab-separated records to process.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--input-glob",
        default="",
        help="Optional glob relative to repo root, for example baseline/phieu_thi/*.jpg.",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline/reports/layout_v0_infer/unet_v2_best",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--page-threshold", type=float, default=0.5)
    parser.add_argument("--grid-threshold", type=float, default=0.5)
    parser.add_argument("--marker-threshold", type=float, default=0.25)
    parser.add_argument("--bubble-threshold", type=float, default=0.25)
    parser.add_argument("--marker-nms-radius", type=int, default=4)
    parser.add_argument("--bubble-nms-radius", type=int, default=3)
    parser.add_argument("--max-marker-peaks", type=int, default=40)
    parser.add_argument("--max-bubble-peaks", type=int, default=900)
    parser.add_argument("--visual-limit", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_for_json(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_selected_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            selected.add(text.split()[0])
    return selected


def image_id_from_path(path: Path) -> str:
    return path.with_suffix("").name


def collect_images(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.image:
        images = []
        for value in args.image:
            path = resolve_path(value, REPO_ROOT)
            images.append((image_id_from_path(path), path))
        return images[: args.limit or None]

    if args.input_glob:
        paths = sorted(REPO_ROOT.glob(args.input_glob))
        images = [(image_id_from_path(path), path) for path in paths if path.is_file()]
        return images[: args.limit or None]

    metadata_path = resolve_path(args.metadata, REPO_ROOT)
    image_list_path = resolve_path(args.image_list, REPO_ROOT) if args.image_list else None
    selected_ids = load_selected_ids(image_list_path)
    images = []
    for record in read_jsonl(metadata_path):
        if record.get("status") != "ok":
            continue
        image_id = str(record["image_id"])
        if selected_ids is not None and image_id not in selected_ids:
            continue
        images.append((image_id, BASELINE_ROOT / record["relative_path"]))
    return images[: args.limit or None]


def preprocess_image(image_bgr: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, torch.Tensor]:
    output_width, output_height = image_size
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return resized, tensor.unsqueeze(0)


def scale_peak_records(
    peaks: list[tuple[float, float, float]],
    *,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, float]]:
    return [
        {
            "x": round(x * scale_x, 3),
            "y": round(y * scale_y, 3),
            "model_x": round(x, 3),
            "model_y": round(y, 3),
            "score": round(score, 6),
        }
        for x, y, score in peaks
    ]


def mask_bbox(mask: np.ndarray, *, scale_x: float, scale_y: float) -> dict[str, list[float] | None]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"model_bbox": None, "source_bbox": None}
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return {
        "model_bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "source_bbox": [
            round(x1 * scale_x, 3),
            round(y1 * scale_y, 3),
            round(x2 * scale_x, 3),
            round(y2 * scale_y, 3),
        ],
    }


def draw_overlay(
    *,
    image_rgb: np.ndarray,
    probabilities: np.ndarray,
    marker_peaks: list[tuple[float, float, float]],
    bubble_peaks: list[tuple[float, float, float]],
    output_path: Path,
    page_threshold: float,
    grid_threshold: float,
) -> None:
    overlay = image_rgb.copy()
    page = probabilities[0] >= page_threshold
    grid = probabilities[3] >= grid_threshold
    overlay[page] = (overlay[page] * 0.82 + np.array([40, 90, 255]) * 0.18).astype(np.uint8)
    overlay[grid] = (255, 40, 40)

    for x, y, _ in bubble_peaks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (30, 220, 80), 1, lineType=cv2.LINE_AA)
    for x, y, _ in marker_peaks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 5, (255, 220, 0), 2, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(image_rgb, 0.68, overlay, 0.32, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def predict_one(
    *,
    image_id: str,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    args: argparse.Namespace,
    visual_path: Path | None,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    source_height, source_width = image_bgr.shape[:2]
    image_rgb, image_tensor = preprocess_image(image_bgr, image_size)
    image_tensor = image_tensor.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()

    marker_peaks = extract_heatmap_peaks(
        probabilities[1],
        threshold=args.marker_threshold,
        nms_radius=args.marker_nms_radius,
        max_peaks=args.max_marker_peaks,
    )
    bubble_peaks = extract_heatmap_peaks(
        probabilities[2],
        threshold=args.bubble_threshold,
        nms_radius=args.bubble_nms_radius,
        max_peaks=args.max_bubble_peaks,
    )

    if visual_path is not None:
        draw_overlay(
            image_rgb=image_rgb,
            probabilities=probabilities,
            marker_peaks=marker_peaks,
            bubble_peaks=bubble_peaks,
            output_path=visual_path,
            page_threshold=args.page_threshold,
            grid_threshold=args.grid_threshold,
        )

    model_width, model_height = image_size
    scale_x = source_width / float(model_width)
    scale_y = source_height / float(model_height)
    page = mask_bbox(
        probabilities[0] >= args.page_threshold,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return {
        "image_id": image_id,
        "source_path": path_for_json(image_path),
        "source_size": [source_width, source_height],
        "model_input_size": [model_width, model_height],
        "visual_path": path_for_json(visual_path) if visual_path is not None else None,
        "counts": {
            "markers": len(marker_peaks),
            "bubbles": len(bubble_peaks),
        },
        "page": page,
        "peaks": {
            "markers": scale_peak_records(marker_peaks, scale_x=scale_x, scale_y=scale_y),
            "bubbles": scale_peak_records(bubble_peaks, scale_x=scale_x, scale_y=scale_y),
        },
    }


def main() -> int:
    args = parse_args()
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

    images = collect_images(args)
    if not images:
        raise SystemExit("no input images matched")

    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"images={len(images)} image_size={image_size}", flush=True)
    print(f"device={device}", flush=True)

    predictions_path = output_dir / "predictions.jsonl"
    records = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, (image_id, image_path) in enumerate(images):
            visual_path = (
                output_dir / "visuals" / f"{image_id}_layout_overlay.jpg"
                if index < args.visual_limit
                else None
            )
            record = predict_one(
                image_id=image_id,
                image_path=image_path,
                model=model,
                device=device,
                image_size=image_size,
                args=args,
                visual_path=visual_path,
            )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"{index + 1}/{len(images)} {image_id} "
                f"markers={record['counts']['markers']} bubbles={record['counts']['bubbles']}",
                flush=True,
            )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "samples": len(records),
        "device": str(device),
        "image_size": [image_width, image_height],
        "thresholds": {
            "page": args.page_threshold,
            "grid": args.grid_threshold,
            "markers": args.marker_threshold,
            "bubbles": args.bubble_threshold,
        },
        "mean_counts": {
            "markers": sum(record["counts"]["markers"] for record in records) / len(records),
            "bubbles": sum(record["counts"]["bubbles"] for record in records) / len(records),
        },
        "visuals_dir": str(output_dir / "visuals") if args.visual_limit > 0 else None,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0
