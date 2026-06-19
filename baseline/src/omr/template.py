"""Template utilities for grid-based OMR cropping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

from .geometry import bubble_bbox


@dataclass(frozen=True)
class BubbleSpec:
    question_id: str
    question_number: int
    choice: str
    center_x: int
    center_y: int
    crop_bbox: tuple[int, int, int, int]


def load_template(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_size(template: dict) -> tuple[int, int]:
    size = template["coordinate_system"]["canonical_size"]
    return int(size["width"]), int(size["height"])


def part1_bubbles(template: dict) -> Iterator[BubbleSpec]:
    crop_width, crop_height = template["bubble_crop"]["size"]
    choices = template["grids"]["part1"]["choices"]

    for column in template["grids"]["part1"]["columns"]:
        question_start = int(column["question_start"])
        question_count = int(column["question_count"])
        row_y_start = float(column["row_y_start"])
        row_y_step = float(column["row_y_step"])
        choice_x = [float(value) for value in column["choice_x"]]

        for row_index in range(question_count):
            question_number = question_start + row_index
            center_y = int(round(row_y_start + row_index * row_y_step))
            for choice, center_x_raw in zip(choices, choice_x, strict=True):
                center_x = int(round(center_x_raw))
                yield BubbleSpec(
                    question_id=f"I_{question_number:03d}",
                    question_number=question_number,
                    choice=choice,
                    center_x=center_x,
                    center_y=center_y,
                    crop_bbox=bubble_bbox(center_x, center_y, (crop_width, crop_height)),
                )

