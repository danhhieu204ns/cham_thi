"""Shared helpers for bubble crop extraction and rule-based pre-labeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

@dataclass(frozen=True)
class BubbleCropResult:
    crop: Image.Image
    darkness_score: float
    prelabel: str


def darkness_score(crop: Image.Image) -> float:
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    return float((gray < 100).mean())


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
    score = darkness_score(crop)
    prelabel = prelabel_from_score(score, blank_threshold, filled_threshold)
    return BubbleCropResult(crop=crop, darkness_score=score, prelabel=prelabel)
