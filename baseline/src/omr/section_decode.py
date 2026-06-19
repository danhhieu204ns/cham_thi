"""Shared decoding helpers for bubble groups."""

from __future__ import annotations

from typing import Iterable


DIGITS = tuple(str(value) for value in range(10))
PART2_CHOICES = ("T", "F")
MODEL_STATE_KEYS = (
    "prelabel_source",
    "model_label",
    "model_confidence",
    "filled_probability",
    "blank_probability",
    "model_argmax_label",
    "model_argmax_confidence",
    "model_filled_threshold",
)


def choice_order(choice: str) -> tuple[int, str]:
    if choice in DIGITS:
        return int(choice), choice
    if choice in PART2_CHOICES:
        return PART2_CHOICES.index(choice), choice
    if choice == "-":
        return 0, choice
    if choice.startswith("after_"):
        return int(choice.split("_", 1)[1]), choice
    return 99, choice


def decode_group(records: Iterable[dict]) -> dict:
    records = sorted(records, key=lambda record: choice_order(str(record["choice"])))
    states = {
        str(record["choice"]): _state_from_record(record)
        for record in records
    }
    filled = [record for record in records if record["prelabel"] == "filled"]
    invalid = [record for record in records if record["prelabel"] == "invalid"]
    ambiguous = [record for record in records if record["prelabel"] == "ambiguous"]

    review_reasons = []
    if invalid:
        review_reasons.append("invalid_bubble")
    if ambiguous and not filled:
        review_reasons.append("ambiguous_bubble")
    if len(filled) > 1:
        review_reasons.append("multi_mark")

    filled = sorted(filled, key=lambda record: float(record["darkness_score"]), reverse=True)
    selected = str(filled[0]["choice"]) if filled else None
    if review_reasons:
        status = "multi_mark" if "multi_mark" in review_reasons else "need_review"
    elif selected is not None:
        status = "accepted"
    else:
        status = "blank"

    selected_score = float(filled[0]["darkness_score"]) if selected is not None else None
    other_scores = [
        float(record["darkness_score"])
        for record in records
        if selected is None or str(record["choice"]) != selected
    ]
    margin = None
    if selected_score is not None and other_scores:
        margin = round(selected_score - max(other_scores), 6)

    return {
        "selected": selected,
        "status": status,
        "need_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "darkness_score": round(selected_score, 6) if selected_score is not None else None,
        "darkness_margin": margin,
        "states": states,
    }


def _state_from_record(record: dict) -> dict:
    state = {
        "label": record.get("label", record["choice"]),
        "prelabel": record["prelabel"],
        "darkness_score": record["darkness_score"],
        "crop_path": record.get("crop_path"),
    }
    for key in MODEL_STATE_KEYS:
        if key in record:
            state[key] = record[key]
    return state
