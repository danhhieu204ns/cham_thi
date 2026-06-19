"""Part II true/false extraction."""

from __future__ import annotations

from collections import Counter

from .geometry import bubble_bbox
from .section_decode import decode_group


def build_specs(template: dict) -> list[dict]:
    grid = template["grids"]["part2"]
    size = tuple(template["bubble_crop"]["size"])
    row_y_start = float(grid["row_y_start"])
    row_y_step = float(grid["row_y_step"])
    specs = []
    for question in grid["questions"]:
        question_number = int(question["question_number"])
        for row_index, statement in enumerate(grid["statements"]):
            center_y = int(round(row_y_start + row_index * row_y_step))
            item_id = f"II_{question_number:03d}_{statement}"
            for choice, center_x_raw in zip(grid["choices"], question["choice_x"], strict=True):
                center_x = int(round(center_x_raw))
                specs.append(
                    {
                        "section": "part2",
                        "spec_id": f"{item_id}_{choice}",
                        "item_id": item_id,
                        "question_id": item_id,
                        "question_number": question_number,
                        "statement": statement,
                        "choice": choice,
                        "label": grid.get("choice_labels", {}).get(choice, choice),
                        "role": "true_false",
                        "slot": row_index + 1,
                        "alignment_block": "part2",
                        "grid_block": f"part2:q{question_number:03d}",
                        "center": [center_x, center_y],
                        "bbox": list(bubble_bbox(center_x, center_y, size)),
                    }
                )
    return specs


def decode(groups: dict[str, list[dict]]) -> dict:
    answers = {}
    status_counts: Counter[str] = Counter()
    for question_number in range(1, 9):
        for statement in ("a", "b", "c", "d"):
            item_id = f"II_{question_number:03d}_{statement}"
            decoded = decode_group(groups.get(item_id, []))
            decoded.update(
                {
                    "question_id": item_id,
                    "question_number": question_number,
                    "statement": statement,
                }
            )
            answers[item_id] = decoded
            status_counts[decoded["status"]] += 1
    return {
        "answers": answers,
        "counts": dict(sorted(status_counts.items())),
    }
