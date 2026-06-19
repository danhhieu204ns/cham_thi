"""Part I multiple-choice extraction."""

from __future__ import annotations

from .decode_answers import decode_part1 as decode_part1_records
from .decode_answers import summarize_decoded_sheets
from .template import part1_bubbles


def _grid_block_for_question(template: dict, question_number: int) -> str:
    for column in template["grids"]["part1"]["columns"]:
        question_start = int(column["question_start"])
        question_end = question_start + int(column["question_count"]) - 1
        if question_start <= question_number <= question_end:
            return f"part1:q{question_start:03d}-{question_end:03d}"
    return "part1"


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
                "alignment_block": "part1",
                "grid_block": _grid_block_for_question(template, bubble.question_number),
                "center": [bubble.center_x, bubble.center_y],
                "bbox": list(bubble.crop_bbox),
            }
        )
    return specs
