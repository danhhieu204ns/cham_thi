"""Part I multiple-choice extraction."""

from __future__ import annotations

from .decode_answers import decode_part1 as decode_part1_records
from .decode_answers import summarize_decoded_sheets
from .template import part1_bubbles


def build_part1_specs(template: dict) -> list[dict]:
    specs = []
    for bubble in part1_bubbles(template):
        specs.append(
            {
                "section": "part1",
                "part": "I",
                "spec_id": f"{bubble.question_id}_{bubble.choice}",
                "item_id": bubble.question_id,
                "question_id": bubble.question_id,
                "question_number": bubble.question_number,
                "choice": bubble.choice,
                "label": bubble.choice,
                "role": "multiple_choice",
                "slot": None,
                "center": [bubble.center_x, bubble.center_y],
                "bbox": list(bubble.crop_bbox),
            }
        )
    return specs
