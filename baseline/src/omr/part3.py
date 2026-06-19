"""Part III short-answer bubble-grid extraction."""

from __future__ import annotations

from collections import Counter

from .geometry import bubble_bbox
from .section_decode import decode_group


def build_specs(template: dict) -> list[dict]:
    grid = template["grids"]["part3"]
    size = tuple(template["bubble_crop"]["size"])
    specs = []
    row_minus_y = int(round(float(grid["row_minus_y"])))
    row_comma_y = int(round(float(grid["row_comma_y"])))
    digit_y_start = float(grid["digit_row_y_start"])
    digit_y_step = float(grid["digit_row_y_step"])

    for question in grid["questions"]:
        question_number = int(question["question_number"])
        qid = f"III_{question_number:03d}"
        minus_x = int(round(question["minus_x"]))
        specs.append(
            {
                "section": "part3",
                "spec_id": f"{qid}_sign_minus",
                "item_id": f"{qid}_sign",
                "question_id": qid,
                "question_number": question_number,
                "choice": "-",
                "label": "-",
                "role": "sign",
                "slot": 0,
                "alignment_block": "part3",
                "grid_block": f"part3:q{question_number:03d}",
                "center": [minus_x, row_minus_y],
                "bbox": list(bubble_bbox(minus_x, row_minus_y, size)),
            }
        )

        for comma_index, center_x_raw in enumerate(question["comma_x"], start=1):
            center_x = int(round(center_x_raw))
            choice = f"after_{comma_index}"
            specs.append(
                {
                    "section": "part3",
                    "spec_id": f"{qid}_comma_{comma_index}",
                    "item_id": f"{qid}_comma",
                    "question_id": qid,
                    "question_number": question_number,
                    "choice": choice,
                    "label": ",",
                    "role": "comma",
                    "slot": comma_index,
                    "alignment_block": "part3",
                    "grid_block": f"part3:q{question_number:03d}",
                    "center": [center_x, row_comma_y],
                    "bbox": list(bubble_bbox(center_x, row_comma_y, size)),
                }
            )

        for slot, center_x_raw in enumerate(question["column_x"], start=1):
            center_x = int(round(center_x_raw))
            for digit_index, digit in enumerate(grid["digit_choices"]):
                center_y = int(round(digit_y_start + digit_index * digit_y_step))
                specs.append(
                    {
                        "section": "part3",
                        "spec_id": f"{qid}_digit_{slot}_{digit}",
                        "item_id": f"{qid}_digit_{slot}",
                        "question_id": qid,
                        "question_number": question_number,
                        "choice": digit,
                        "label": digit,
                        "role": "digit",
                        "slot": slot,
                        "alignment_block": "part3",
                        "grid_block": f"part3:q{question_number:03d}",
                        "center": [center_x, center_y],
                        "bbox": list(bubble_bbox(center_x, center_y, size)),
                    }
                )
    return specs


def compact_value(
    sign: str | None,
    comma: str | None,
    digits: list[str | None],
) -> tuple[str, str | None]:
    raw = [digit if digit is not None else "_" for digit in digits]
    if comma and comma.startswith("after_"):
        after_slot = int(comma.split("_", 1)[1])
        insert_at = max(1, min(len(raw), after_slot))
        raw.insert(insert_at, ",")
    raw_value = ("-" if sign else "") + "".join(raw)
    compact = raw_value.replace("_", "")
    if compact in {"", "-", ",", "-,"}:
        compact = None
    return raw_value, compact


def decode(groups: dict[str, list[dict]]) -> dict:
    answers = {}
    status_counts: Counter[str] = Counter()
    for question_number in range(1, 7):
        qid = f"III_{question_number:03d}"
        sign = decode_group(groups.get(f"{qid}_sign", []))
        comma = decode_group(groups.get(f"{qid}_comma", []))
        digit_groups = []
        for slot in range(1, 5):
            digit = decode_group(groups.get(f"{qid}_digit_{slot}", []))
            digit["slot"] = slot
            digit_groups.append(digit)

        selected_digits = [
            str(digit["selected"]) if digit["status"] == "accepted" else None
            for digit in digit_groups
        ]
        sign_selected = sign["selected"] if sign["status"] == "accepted" else None
        comma_selected = comma["selected"] if comma["status"] == "accepted" else None
        raw_value, decoded_value = compact_value(
            sign_selected,
            comma_selected,
            selected_digits,
        )

        component_statuses = [sign["status"], comma["status"], *(digit["status"] for digit in digit_groups)]
        if any(status in {"need_review", "multi_mark"} for status in component_statuses):
            status = "need_review"
        elif decoded_value is None:
            status = "blank"
        else:
            status = "accepted"
        confidence_components = [sign, comma, *digit_groups]
        if status == "blank":
            confidence_components = [
                component for component in confidence_components if component.get("confidence") is not None
            ]
        else:
            confidence_components = [
                component
                for component in confidence_components
                if component.get("status") != "blank" and component.get("confidence") is not None
            ]
        component_confidences = [
            float(component["confidence"])
            for component in confidence_components
        ]

        answer = {
            "question_id": qid,
            "question_number": question_number,
            "value": decoded_value,
            "raw_value": raw_value,
            "status": status,
            "confidence": round(min(component_confidences), 6) if component_confidences else None,
            "sign": sign,
            "comma": comma,
            "digits": digit_groups,
        }
        answers[qid] = answer
        status_counts[status] += 1

    return {
        "answers": answers,
        "counts": dict(sorted(status_counts.items())),
    }
