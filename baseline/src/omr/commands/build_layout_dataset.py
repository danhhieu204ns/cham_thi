"""Build synthetic layout/keypoint training data for OMR alignment."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from omr.cli_utils import filter_ok_records, write_json_file
from omr.jsonl_io import read_jsonl_records, write_jsonl_records
from omr.markers import detect_registration_markers
from omr.template import canonical_size, load_template
from omr.warp import warp_from_markers


BASELINE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BASELINE_ROOT.parent


@dataclass(frozen=True)
class PointLabel:
    id: str
    group: str
    x: float
    y: float
    meta: dict[str, Any]


@dataclass(frozen=True)
class AugmentedGeometry:
    homography: np.ndarray
    dx_map: np.ndarray
    dy_map: np.ndarray
    description: dict[str, Any]
    background_color: tuple[int, int, int]
    occluded_markers: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/labels/sheets.jsonl")
    parser.add_argument("--template", default="data/labels/template_tnthpt.json")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-dir", default="../data_train/layout_v0")
    parser.add_argument("--image-width", type=int, default=832)
    parser.add_argument("--image-height", type=int, default=1192)
    parser.add_argument("--augmentations-per-scan", type=int, default=1)
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--val-count", type=int, default=25)
    parser.add_argument("--test-count", type=int, default=22)
    parser.add_argument(
        "--limit-base",
        type=int,
        default=0,
        help="Debug limit for base scans after split assignment. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir before writing new data.",
    )
    parser.add_argument(
        "--save-canonical",
        action="store_true",
        help="Also save marker-warped canonical source sheets for audit.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base / path


def bbox_to_polygon(bbox: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def iter_part1_bubbles(template: dict) -> Iterable[PointLabel]:
    choices = template["grids"]["part1"]["choices"]
    for column in template["grids"]["part1"]["columns"]:
        question_start = int(column["question_start"])
        question_count = int(column["question_count"])
        row_y_start = float(column["row_y_start"])
        row_y_step = float(column["row_y_step"])
        choice_x = [float(value) for value in column["choice_x"]]

        for row_index in range(question_count):
            question_number = question_start + row_index
            y = row_y_start + row_index * row_y_step
            for choice, x in zip(choices, choice_x, strict=True):
                yield PointLabel(
                    id=f"part1_q{question_number:03d}_{choice}",
                    group="part1",
                    x=x,
                    y=y,
                    meta={"question_number": question_number, "choice": choice},
                )


def iter_identity_bubbles(template: dict, group: str) -> Iterable[PointLabel]:
    grid = template["grids"][group]
    digits = grid["digits"]
    row_y_start = float(grid["row_y_start"])
    row_y_step = float(grid["row_y_step"])
    for column_index, x_raw in enumerate(grid["column_x"], start=1):
        x = float(x_raw)
        for digit_index, digit in enumerate(digits):
            y = row_y_start + digit_index * row_y_step
            yield PointLabel(
                id=f"{group}_c{column_index:02d}_d{digit}",
                group=group,
                x=x,
                y=y,
                meta={"column": column_index, "digit": digit},
            )


def iter_part2_bubbles(template: dict) -> Iterable[PointLabel]:
    grid = template["grids"]["part2"]
    choices = grid["choices"]
    statements = grid["statements"]
    row_y_start = float(grid["row_y_start"])
    row_y_step = float(grid["row_y_step"])
    for question in grid["questions"]:
        question_number = int(question["question_number"])
        choice_x = [float(value) for value in question["choice_x"]]
        for statement_index, statement in enumerate(statements):
            y = row_y_start + statement_index * row_y_step
            for choice, x in zip(choices, choice_x, strict=True):
                yield PointLabel(
                    id=f"part2_q{question_number:02d}_{statement}_{choice}",
                    group="part2",
                    x=x,
                    y=y,
                    meta={
                        "question_number": question_number,
                        "statement": statement,
                        "choice": choice,
                    },
                )


def iter_part3_bubbles(template: dict) -> Iterable[PointLabel]:
    grid = template["grids"]["part3"]
    digits = grid["digit_choices"]
    row_minus_y = float(grid["row_minus_y"])
    row_comma_y = float(grid["row_comma_y"])
    digit_row_y_start = float(grid["digit_row_y_start"])
    digit_row_y_step = float(grid["digit_row_y_step"])
    for question in grid["questions"]:
        question_number = int(question["question_number"])
        minus_x = float(question["minus_x"])
        yield PointLabel(
            id=f"part3_q{question_number:02d}_minus",
            group="part3",
            x=minus_x,
            y=row_minus_y,
            meta={"question_number": question_number, "kind": "minus"},
        )

        for comma_index, x_raw in enumerate(question["comma_x"], start=1):
            yield PointLabel(
                id=f"part3_q{question_number:02d}_comma{comma_index}",
                group="part3",
                x=float(x_raw),
                y=row_comma_y,
                meta={
                    "question_number": question_number,
                    "kind": "comma",
                    "comma_index": comma_index,
                },
            )

        for slot_index, x_raw in enumerate(question["column_x"], start=1):
            x = float(x_raw)
            for digit_index, digit in enumerate(digits):
                y = digit_row_y_start + digit_index * digit_row_y_step
                yield PointLabel(
                    id=f"part3_q{question_number:02d}_s{slot_index}_d{digit}",
                    group="part3",
                    x=x,
                    y=y,
                    meta={
                        "question_number": question_number,
                        "slot": slot_index,
                        "digit": digit,
                    },
                )


def collect_bubble_labels(template: dict) -> list[PointLabel]:
    return [
        *iter_identity_bubbles(template, "sbd"),
        *iter_identity_bubbles(template, "exam_code"),
        *iter_part1_bubbles(template),
        *iter_part2_bubbles(template),
        *iter_part3_bubbles(template),
    ]


def collect_region_polygons(template: dict) -> dict[str, list[tuple[float, float]]]:
    return {
        name: bbox_to_polygon(region["bbox"])
        for name, region in template["regions"].items()
    }


def collect_grid_polylines(template: dict, width: int, height: int) -> list[list[tuple[float, float]]]:
    polylines: list[list[tuple[float, float]]] = [
        [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
    ]
    for region in template["regions"].values():
        polylines.append(bbox_to_polygon(region["bbox"]))
    for column in template["grids"]["part1"]["columns"]:
        polylines.append(bbox_to_polygon(column["bbox"]))
    for group in ("sbd", "exam_code"):
        polylines.append(bbox_to_polygon(template["grids"][group]["bbox"]))
    for question in template["grids"]["part2"]["questions"]:
        polylines.append(bbox_to_polygon(question["bbox"]))
    for question in template["grids"]["part3"]["questions"]:
        polylines.append(bbox_to_polygon(question["bbox"]))
    return polylines


def split_records(records: list[dict], args: argparse.Namespace) -> dict[str, list[dict]]:
    rng = np.random.default_rng(args.seed)
    ordered = sorted(records, key=lambda item: item["image_id"])
    indices = np.arange(len(ordered))
    rng.shuffle(indices)
    shuffled = [ordered[int(index)] for index in indices]

    requested = args.train_count + args.val_count + args.test_count
    if requested > len(shuffled):
        raise ValueError(
            f"requested split size {requested} exceeds available ok records {len(shuffled)}"
        )

    train = shuffled[: args.train_count]
    val = shuffled[args.train_count : args.train_count + args.val_count]
    test = shuffled[
        args.train_count + args.val_count : args.train_count + args.val_count + args.test_count
    ]

    splits = {"train": train, "val": val, "test": test}
    if args.limit_base > 0:
        limited: dict[str, list[dict]] = {}
        remaining = args.limit_base
        for split_name in ("train", "val", "test"):
            selected = splits[split_name][:remaining]
            limited[split_name] = selected
            remaining -= len(selected)
            if remaining <= 0:
                remaining = 0
        splits = limited
    return splits


def canonicalize_image(
    image_path: Path,
    target_markers: dict[str, list[int]],
    output_size: tuple[int, int],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, {"status": "error", "reason": "image_read_failed"}

    matched = detect_registration_markers(image, target_markers)
    if len(matched) < 3:
        return None, {
            "status": "error",
            "reason": "not_enough_markers",
            "marker_count": len(matched),
            "markers": sorted(matched),
        }

    try:
        warped, matrix = warp_from_markers(image, matched, target_markers, output_size)
    except Exception as exc:  # pragma: no cover - defensive for corrupt images.
        return None, {
            "status": "error",
            "reason": "warp_failed",
            "message": str(exc),
            "marker_count": len(matched),
        }

    return warped, {
        "status": "ok",
        "marker_count": len(matched),
        "markers": sorted(matched),
        "canonical_homography": matrix.round(8).tolist(),
    }


def clean_geometry(
    canonical_width: int,
    canonical_height: int,
    output_width: int,
    output_height: int,
) -> AugmentedGeometry:
    src = np.array(
        [
            [0.0, 0.0],
            [canonical_width - 1.0, 0.0],
            [canonical_width - 1.0, canonical_height - 1.0],
            [0.0, canonical_height - 1.0],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    zeros = np.zeros((output_height, output_width), dtype=np.float32)
    return AugmentedGeometry(
        homography=homography,
        dx_map=zeros,
        dy_map=zeros,
        description={
            "kind": "clean_resize",
            "homography": homography.round(8).tolist(),
            "elastic_max_abs_px": 0.0,
        },
        background_color=(255, 255, 255),
        occluded_markers=set(),
    )


def random_page_corners(
    rng: np.random.Generator,
    canonical_width: int,
    canonical_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    aspect = canonical_width / canonical_height
    base_height = output_height * float(rng.uniform(0.86, 1.08))
    base_width = base_height * aspect
    if base_width > output_width * 1.08:
        base_width = output_width * float(rng.uniform(0.86, 1.08))
        base_height = base_width / aspect

    rect = np.array(
        [
            [-base_width / 2.0, -base_height / 2.0],
            [base_width / 2.0, -base_height / 2.0],
            [base_width / 2.0, base_height / 2.0],
            [-base_width / 2.0, base_height / 2.0],
        ],
        dtype=np.float32,
    )
    angle = math.radians(float(rng.uniform(-24.0, 24.0)))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    center = np.array(
        [
            output_width / 2.0 + output_width * float(rng.uniform(-0.08, 0.08)),
            output_height / 2.0 + output_height * float(rng.uniform(-0.08, 0.08)),
        ],
        dtype=np.float32,
    )
    corners = rect @ rotation.T + center
    jitter = np.column_stack(
        [
            rng.uniform(-0.055, 0.055, size=4) * output_width,
            rng.uniform(-0.055, 0.055, size=4) * output_height,
        ]
    ).astype(np.float32)
    return corners + jitter


def random_displacement_maps(
    rng: np.random.Generator,
    output_width: int,
    output_height: int,
    max_abs_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if max_abs_px <= 0:
        zeros = np.zeros((output_height, output_width), dtype=np.float32)
        return zeros, zeros

    grid_w = 5
    grid_h = 7
    dx_small = rng.normal(0.0, max_abs_px / 2.5, size=(grid_h, grid_w)).astype(np.float32)
    dy_small = rng.normal(0.0, max_abs_px / 2.5, size=(grid_h, grid_w)).astype(np.float32)
    dx_map = cv2.resize(dx_small, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    dy_map = cv2.resize(dy_small, (output_width, output_height), interpolation=cv2.INTER_CUBIC)

    x_line = np.linspace(0, 2 * math.pi, output_width, dtype=np.float32)
    y_line = np.linspace(0, 2 * math.pi, output_height, dtype=np.float32)
    bend_x = math.sin(float(rng.uniform(0, 2 * math.pi))) * np.sin(y_line)[:, None]
    bend_y = math.cos(float(rng.uniform(0, 2 * math.pi))) * np.sin(x_line)[None, :]
    dx_map += (max_abs_px * 0.35 * bend_x).astype(np.float32)
    dy_map += (max_abs_px * 0.35 * bend_y).astype(np.float32)

    return (
        np.clip(dx_map, -max_abs_px, max_abs_px).astype(np.float32),
        np.clip(dy_map, -max_abs_px, max_abs_px).astype(np.float32),
    )


def random_geometry(
    rng: np.random.Generator,
    canonical_width: int,
    canonical_height: int,
    output_width: int,
    output_height: int,
    marker_names: list[str],
) -> AugmentedGeometry:
    src = np.array(
        [
            [0.0, 0.0],
            [canonical_width - 1.0, 0.0],
            [canonical_width - 1.0, canonical_height - 1.0],
            [0.0, canonical_height - 1.0],
        ],
        dtype=np.float32,
    )
    dst = random_page_corners(rng, canonical_width, canonical_height, output_width, output_height)
    homography = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    elastic_max_abs_px = float(rng.uniform(2.0, 12.0))
    dx_map, dy_map = random_displacement_maps(rng, output_width, output_height, elastic_max_abs_px)

    occluded_markers: set[str] = set()
    if marker_names and rng.random() < 0.35:
        marker_count = int(rng.integers(1, min(3, len(marker_names)) + 1))
        occluded_markers = set(rng.choice(marker_names, size=marker_count, replace=False).tolist())

    background_value = int(rng.integers(180, 242))
    background_color = (
        int(np.clip(background_value + rng.integers(-8, 9), 0, 255)),
        int(np.clip(background_value + rng.integers(-8, 9), 0, 255)),
        int(np.clip(background_value + rng.integers(-8, 9), 0, 255)),
    )
    return AugmentedGeometry(
        homography=homography,
        dx_map=dx_map,
        dy_map=dy_map,
        description={
            "kind": "synthetic_photo",
            "page_corners": dst.round(3).tolist(),
            "homography": homography.round(8).tolist(),
            "elastic_max_abs_px": round(elastic_max_abs_px, 3),
            "occluded_markers": sorted(occluded_markers),
        },
        background_color=background_color,
        occluded_markers=occluded_markers,
    )


def transform_points(points: np.ndarray, geometry: AugmentedGeometry) -> np.ndarray:
    if len(points) == 0:
        return points.copy()

    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), geometry.homography
    ).reshape(-1, 2)
    return apply_displacement_to_points(transformed, geometry.dx_map, geometry.dy_map)


def apply_displacement_to_points(
    points: np.ndarray,
    dx_map: np.ndarray,
    dy_map: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return points.copy()

    height, width = dx_map.shape[:2]
    adjusted = points.astype(np.float32).copy()
    x = np.clip(np.rint(adjusted[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(adjusted[:, 1]).astype(np.int32), 0, height - 1)
    adjusted[:, 0] += dx_map[y, x]
    adjusted[:, 1] += dy_map[y, x]
    return adjusted


def warp_image(canonical: np.ndarray, geometry: AugmentedGeometry, output_size: tuple[int, int]) -> np.ndarray:
    output_width, output_height = output_size
    warped = cv2.warpPerspective(
        canonical,
        geometry.homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=geometry.background_color,
    )
    if float(np.max(np.abs(geometry.dx_map))) <= 0 and float(np.max(np.abs(geometry.dy_map))) <= 0:
        return warped

    xs, ys = np.meshgrid(
        np.arange(output_width, dtype=np.float32),
        np.arange(output_height, dtype=np.float32),
    )
    map_x = (xs - geometry.dx_map).astype(np.float32)
    map_y = (ys - geometry.dy_map).astype(np.float32)
    return cv2.remap(
        warped,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=geometry.background_color,
    )


def apply_photo_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    clean: bool,
) -> np.ndarray:
    if clean:
        return image

    result = image.astype(np.float32)
    alpha = float(rng.uniform(0.78, 1.22))
    beta = float(rng.uniform(-24.0, 24.0))
    result = result * alpha + beta

    height, width = result.shape[:2]
    if rng.random() < 0.75:
        x_gradient = np.linspace(0.75, 1.12, width, dtype=np.float32)
        y_gradient = np.linspace(0.85, 1.10, height, dtype=np.float32)
        if rng.random() < 0.5:
            x_gradient = x_gradient[::-1]
        if rng.random() < 0.5:
            y_gradient = y_gradient[::-1]
        gradient = np.sqrt(y_gradient[:, None] * x_gradient[None, :])
        result *= gradient[:, :, None]

    if rng.random() < 0.45:
        center_x = float(rng.uniform(0, width))
        center_y = float(rng.uniform(0, height))
        radius = float(rng.uniform(width * 0.35, width * 0.85))
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        distance = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
        shadow = 1.0 - 0.28 * np.clip(1.0 - distance / radius, 0.0, 1.0)
        result *= shadow[:, :, None].astype(np.float32)

    if rng.random() < 0.55:
        sigma = float(rng.uniform(2.0, 9.0))
        noise = rng.normal(0.0, sigma, size=result.shape).astype(np.float32)
        result += noise

    result = np.clip(result, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        kernel = int(rng.choice([3, 5]))
        result = cv2.GaussianBlur(result, (kernel, kernel), float(rng.uniform(0.4, 1.2)))

    if rng.random() < 0.55:
        quality = int(rng.integers(55, 92))
        ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                result = decoded
    return result


def sample_polyline(points: list[tuple[float, float]], samples_per_edge: int = 32) -> np.ndarray:
    sampled: list[tuple[float, float]] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        for step in range(samples_per_edge):
            t = step / float(samples_per_edge)
            sampled.append((start[0] * (1.0 - t) + end[0] * t, start[1] * (1.0 - t) + end[1] * t))
    return np.array(sampled, dtype=np.float32)


def page_boundary_points(width: int, height: int, samples_per_edge: int = 96) -> np.ndarray:
    return sample_polyline(
        [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)],
        samples_per_edge=samples_per_edge,
    )


def in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0.0 <= x < width and 0.0 <= y < height


def draw_gaussian_heatmap(
    heatmap: np.ndarray,
    x: float,
    y: float,
    *,
    sigma: float,
) -> None:
    height, width = heatmap.shape[:2]
    if not in_frame(x, y, width, height):
        return

    radius = int(math.ceil(sigma * 3.0))
    center_x = int(round(x))
    center_y = int(round(y))
    x1 = max(0, center_x - radius)
    x2 = min(width - 1, center_x + radius)
    y1 = max(0, center_y - radius)
    y2 = min(height - 1, center_y + radius)
    if x1 > x2 or y1 > y2:
        return

    xs = np.arange(x1, x2 + 1, dtype=np.float32)
    ys = np.arange(y1, y2 + 1, dtype=np.float32)
    patch = np.exp(-((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2) / (2.0 * sigma * sigma))
    patch_u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    current = heatmap[y1 : y2 + 1, x1 : x2 + 1]
    np.maximum(current, patch_u8, out=current)


def draw_marker_occlusions(
    image: np.ndarray,
    marker_records: list[dict[str, Any]],
    occluded_markers: set[str],
    rng: np.random.Generator,
) -> None:
    for marker in marker_records:
        if marker["name"] not in occluded_markers or not marker["in_frame"]:
            continue
        x = int(round(float(marker["x"])))
        y = int(round(float(marker["y"])))
        radius = int(rng.integers(12, 24))
        color_value = int(rng.integers(205, 252))
        color = (color_value, color_value, color_value)
        cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), color, -1)
        marker["visible"] = False
        marker["occluded"] = True


def build_masks(
    marker_records: list[dict[str, Any]],
    bubble_records: list[dict[str, Any]],
    page_polygon: np.ndarray,
    grid_polygons: list[np.ndarray],
    output_width: int,
    output_height: int,
) -> dict[str, np.ndarray]:
    page_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    page_points = np.rint(page_polygon).astype(np.int32)
    cv2.fillPoly(page_mask, [page_points], 255)

    marker_heatmap = np.zeros((output_height, output_width), dtype=np.uint8)
    for marker in marker_records:
        if marker["visible"]:
            draw_gaussian_heatmap(marker_heatmap, float(marker["x"]), float(marker["y"]), sigma=4.0)

    bubble_heatmap = np.zeros((output_height, output_width), dtype=np.uint8)
    for bubble in bubble_records:
        if bubble["visible"]:
            draw_gaussian_heatmap(bubble_heatmap, float(bubble["x"]), float(bubble["y"]), sigma=2.6)

    red_grid_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    for polygon in grid_polygons:
        points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(red_grid_mask, [points], isClosed=True, color=255, thickness=2, lineType=cv2.LINE_AA)

    return {
        "page_mask": page_mask,
        "marker_heatmap": marker_heatmap,
        "bubble_heatmap": bubble_heatmap,
        "red_grid_mask": red_grid_mask,
    }


def relative_to_output(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def write_split_files(splits: dict[str, list[dict]], output_dir: Path) -> None:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        ids = [record["image_id"] for record in records]
        (splits_dir / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "images",
        "labels",
        "masks/page_mask",
        "masks/marker_heatmap",
        "masks/bubble_heatmap",
        "masks/red_grid_mask",
        "splits",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def write_image(path: Path, image: np.ndarray, *, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(path), image)


def make_sample(
    *,
    canonical_image: np.ndarray,
    template: dict,
    split_name: str,
    record: dict,
    sample_index: int,
    clean: bool,
    geometry: AugmentedGeometry,
    bubble_labels: list[PointLabel],
    region_polygons: dict[str, list[tuple[float, float]]],
    grid_polylines: list[list[tuple[float, float]]],
    output_dir: Path,
    output_width: int,
    output_height: int,
    canonical_width: int,
    canonical_height: int,
    rng: np.random.Generator,
    canonical_info: dict[str, Any],
) -> dict[str, Any]:
    sample_id = f"{record['image_id']}_{'clean' if clean else f'aug{sample_index:03d}'}"
    image = warp_image(canonical_image, geometry, (output_width, output_height))

    marker_items = sorted(template["registration_marks"]["centers"].items())
    marker_points = np.array([value for _, value in marker_items], dtype=np.float32)
    transformed_markers = transform_points(marker_points, geometry)
    marker_records: list[dict[str, Any]] = []
    for (name, _), (x, y) in zip(marker_items, transformed_markers, strict=True):
        marker_records.append(
            {
                "name": name,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "in_frame": in_frame(float(x), float(y), output_width, output_height),
                "visible": in_frame(float(x), float(y), output_width, output_height),
                "occluded": False,
            }
        )

    draw_marker_occlusions(image, marker_records, geometry.occluded_markers, rng)
    image = apply_photo_noise(image, rng, clean=clean)

    bubble_points = np.array([[label.x, label.y] for label in bubble_labels], dtype=np.float32)
    transformed_bubbles = transform_points(bubble_points, geometry)
    bubble_records: list[dict[str, Any]] = []
    for label, (x, y) in zip(bubble_labels, transformed_bubbles, strict=True):
        visible = in_frame(float(x), float(y), output_width, output_height)
        bubble_records.append(
            {
                "id": label.id,
                "group": label.group,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "visible": visible,
                **label.meta,
            }
        )

    region_records: dict[str, list[list[float]]] = {}
    for name, polygon in region_polygons.items():
        transformed = transform_points(np.array(polygon, dtype=np.float32), geometry)
        region_records[name] = [[round(float(x), 3), round(float(y), 3)] for x, y in transformed]

    page_polygon = transform_points(
        page_boundary_points(canonical_width, canonical_height), geometry
    )
    grid_polygons = [
        transform_points(sample_polyline(polyline), geometry) for polyline in grid_polylines
    ]
    masks = build_masks(
        marker_records,
        bubble_records,
        page_polygon,
        grid_polygons,
        output_width,
        output_height,
    )

    image_path = output_dir / "images" / split_name / f"{sample_id}.jpg"
    label_path = output_dir / "labels" / split_name / f"{sample_id}.json"
    mask_paths = {
        name: output_dir / "masks" / name / split_name / f"{sample_id}.png"
        for name in masks
    }

    write_image(image_path, image, quality=90)
    for name, mask in masks.items():
        write_image(mask_paths[name], mask)

    label_record = {
        "dataset_version": "layout_v0",
        "sample_id": sample_id,
        "base_image_id": record["image_id"],
        "split": split_name,
        "image": {
            "path": relative_to_output(image_path, output_dir),
            "width": output_width,
            "height": output_height,
        },
        "source": {
            "relative_path": record.get("relative_path"),
            "width": record.get("width"),
            "height": record.get("height"),
            "canonicalization": canonical_info,
        },
        "canonical_size": [canonical_width, canonical_height],
        "augmentation": geometry.description,
        "markers": marker_records,
        "regions": region_records,
        "bubbles": bubble_records,
        "masks": {name: relative_to_output(path, output_dir) for name, path in mask_paths.items()},
        "counts": {
            "markers": len(marker_records),
            "visible_markers": sum(int(item["visible"]) for item in marker_records),
            "bubbles": len(bubble_records),
            "visible_bubbles": sum(int(item["visible"]) for item in bubble_records),
        },
    }
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "sample_id": sample_id,
        "base_image_id": record["image_id"],
        "split": split_name,
        "image_path": relative_to_output(image_path, output_dir),
        "label_path": relative_to_output(label_path, output_dir),
        "mask_paths": {name: relative_to_output(path, output_dir) for name, path in mask_paths.items()},
        "clean": clean,
        "visible_markers": label_record["counts"]["visible_markers"],
        "visible_bubbles": label_record["counts"]["visible_bubbles"],
    }


def main() -> int:
    args = parse_args()
    metadata_path = resolve_path(args.metadata, BASELINE_ROOT)
    template_path = resolve_path(args.template, BASELINE_ROOT)
    source_root = resolve_path(args.source_root, BASELINE_ROOT)
    output_dir = resolve_path(args.output_dir, BASELINE_ROOT)

    if args.augmentations_per_scan < 0:
        raise ValueError("--augmentations-per-scan must be >= 0")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("--image-width and --image-height must be positive")

    prepare_output_dir(output_dir, args.overwrite)
    template = load_template(template_path)
    canonical_width, canonical_height = canonical_size(template)
    target_markers = template["registration_marks"]["centers"]
    bubble_labels = collect_bubble_labels(template)
    region_polygons = collect_region_polygons(template)
    grid_polylines = collect_grid_polylines(template, canonical_width, canonical_height)

    records = filter_ok_records(read_jsonl_records(metadata_path), sort_field="image_id")
    splits = split_records(records, args)
    write_split_files(splits, output_dir)

    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    marker_names = sorted(target_markers)

    for split_name, split_records_for_name in splits.items():
        for record in split_records_for_name:
            image_path = source_root / record["relative_path"]
            canonical_image, canonical_info = canonicalize_image(
                image_path, target_markers, (canonical_width, canonical_height)
            )
            if canonical_image is None:
                skipped.append(
                    {
                        "image_id": record["image_id"],
                        "split": split_name,
                        "relative_path": record.get("relative_path"),
                        **canonical_info,
                    }
                )
                continue

            if args.save_canonical:
                canonical_path = output_dir / "canonical" / split_name / f"{record['image_id']}.jpg"
                write_image(canonical_path, canonical_image, quality=92)

            sample_counter = 0
            if args.include_clean:
                geometry = clean_geometry(
                    canonical_width,
                    canonical_height,
                    args.image_width,
                    args.image_height,
                )
                manifest.append(
                    make_sample(
                        canonical_image=canonical_image,
                        template=template,
                        split_name=split_name,
                        record=record,
                        sample_index=sample_counter,
                        clean=True,
                        geometry=geometry,
                        bubble_labels=bubble_labels,
                        region_polygons=region_polygons,
                        grid_polylines=grid_polylines,
                        output_dir=output_dir,
                        output_width=args.image_width,
                        output_height=args.image_height,
                        canonical_width=canonical_width,
                        canonical_height=canonical_height,
                        rng=rng,
                        canonical_info=canonical_info,
                    )
                )
                sample_counter += 1

            for augment_index in range(args.augmentations_per_scan):
                geometry = random_geometry(
                    rng,
                    canonical_width,
                    canonical_height,
                    args.image_width,
                    args.image_height,
                    marker_names,
                )
                manifest.append(
                    make_sample(
                        canonical_image=canonical_image,
                        template=template,
                        split_name=split_name,
                        record=record,
                        sample_index=augment_index,
                        clean=False,
                        geometry=geometry,
                        bubble_labels=bubble_labels,
                        region_polygons=region_polygons,
                        grid_polylines=grid_polylines,
                        output_dir=output_dir,
                        output_width=args.image_width,
                        output_height=args.image_height,
                        canonical_width=canonical_width,
                        canonical_height=canonical_height,
                        rng=rng,
                        canonical_info=canonical_info,
                    )
                )

    write_jsonl_records(manifest, output_dir / "manifest.jsonl")
    write_jsonl_records(skipped, output_dir / "skipped.jsonl")

    summary = {
        "dataset_version": "layout_v0",
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "template": str(template_path),
        "seed": args.seed,
        "image_size": [args.image_width, args.image_height],
        "augmentations_per_scan": args.augmentations_per_scan,
        "include_clean": args.include_clean,
        "base_split_counts": {name: len(value) for name, value in splits.items()},
        "samples": len(manifest),
        "skipped": len(skipped),
        "bubble_labels_per_sample": len(bubble_labels),
        "marker_labels_per_sample": len(target_markers),
        "mask_channels": ["page_mask", "marker_heatmap", "bubble_heatmap", "red_grid_mask"],
    }
    write_json_file(summary, output_dir / "summary.json", sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if skipped:
        print(f"Skipped {len(skipped)} base scans; see {output_dir / 'skipped.jsonl'}")
    return 0
