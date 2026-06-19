"""Shared helpers for bubble crop extraction and rule-based pre-labeling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class BubbleCropResult:
    crop: Image.Image
    darkness_score: float
    fill_score: float
    ink_ratio_inside: float
    background_noise: float
    darkness_contrast: float
    connected_component_score: float
    prelabel: str


def darkness_score(crop: Image.Image) -> float:
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    return float((gray < 100).mean())


def _circle_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius = min(width, height) / 2.0
    core = distance <= radius * 0.58
    inner = distance <= radius * 0.72
    background = distance >= radius * 0.88
    return core, inner, background


def bubble_features(crop: Image.Image) -> dict[str, float]:
    """Return robust rule features for a normalized bubble crop."""
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    core_mask, inner_mask, background_mask = _circle_masks(gray.shape)

    inner = gray[inner_mask]
    core = gray[core_mask]
    background = gray[background_mask]
    if inner.size == 0 or core.size == 0 or background.size == 0:
        score = darkness_score(crop)
        return {
            "darkness_score": score,
            "fill_score": score,
            "ink_ratio_inside": score,
            "background_noise": 0.0,
            "darkness_contrast": score,
            "connected_component_score": 0.0,
        }

    background_darkness = float(np.mean((255.0 - background.astype(np.float32)) / 255.0))
    inner_darkness = float(np.mean((255.0 - inner.astype(np.float32)) / 255.0))
    darkness_contrast = max(0.0, inner_darkness - background_darkness)

    adaptive_threshold = min(190.0, float(np.percentile(background, 20)) - 20.0)
    adaptive_threshold = max(80.0, adaptive_threshold)
    dark_core = (gray < adaptive_threshold) & core_mask
    ink_ratio_inside = float(dark_core.sum() / max(1, int(core_mask.sum())))

    component_score = 0.0
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_core.astype(np.uint8),
        connectivity=8,
    )
    if labels_count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(areas.max()) if areas.size else 0
        component_score = float(largest_area / max(1, int(core_mask.sum())))

    raw_score = (
        0.50 * ink_ratio_inside
        + 0.35 * darkness_contrast
        + 0.20 * component_score
        - 0.20 * background_darkness
    )
    fill_score = max(0.0, min(1.0, raw_score))

    return {
        "darkness_score": darkness_score(crop),
        "fill_score": fill_score,
        "ink_ratio_inside": ink_ratio_inside,
        "background_noise": background_darkness,
        "darkness_contrast": darkness_contrast,
        "connected_component_score": component_score,
    }


def prelabel_from_score(score: float, blank_threshold: float, filled_threshold: float) -> str:
    if score <= blank_threshold:
        return "blank"
    if score >= filled_threshold:
        return "filled"
    return "ambiguous"


def safe_crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    crop = Image.new("RGB", (x2 - x1, y2 - y1), "white")
    source_box = (
        max(0, x1),
        max(0, y1),
        min(image.width, x2),
        min(image.height, y2),
    )
    paste_x = max(0, -x1)
    paste_y = max(0, -y1)
    if source_box[0] < source_box[2] and source_box[1] < source_box[3]:
        crop.paste(image.crop(source_box), (paste_x, paste_y))
    return crop


def crop_and_prelabel(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    blank_threshold: float,
    filled_threshold: float,
) -> BubbleCropResult:
    crop = safe_crop(image, bbox)
    features = bubble_features(crop)
    prelabel = prelabel_from_score(features["fill_score"], blank_threshold, filled_threshold)
    return BubbleCropResult(crop=crop, prelabel=prelabel, **features)
