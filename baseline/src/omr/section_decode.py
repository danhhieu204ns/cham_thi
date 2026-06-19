"""Shared decoding helpers for bubble groups."""

from __future__ import annotations

from typing import Iterable


DIGITS = tuple(str(value) for value in range(10))
PART2_CHOICES = ("T", "F")
MODEL_STATE_KEYS = (
    "prelabel_source",
    "fill_score",
    "ink_ratio_inside",
    "background_noise",
    "darkness_contrast",
    "connected_component_score",
    "model_label",
    "model_confidence",
    "filled_probability",
    "blank_probability",
    "model_argmax_label",
    "model_argmax_confidence",
    "model_filled_threshold",
    "alignment_block",
    "alignment_status",
    "alignment_confidence",
    "alignment_method",
    "alignment_marker_count",
    "marker_pre_residual_px",
    "marker_residual_px",
    "marker_max_residual_px",
    "grid_refinement_status",
    "grid_refinement_confidence",
    "grid_refinement_method",
    "grid_refinement_matched_count",
    "grid_refinement_inlier_count",
    "grid_residual_px",
    "grid_max_residual_px",
    "grid_refinement_decode_allowed",
    "group_score_key",
    "group_top_score",
    "group_second_score",
    "group_score_margin",
    "group_filled_threshold",
    "group_margin_threshold",
    "group_relabel_source",
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
        str(record["choice"]): state_from_record(record)
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
    alignment_needs_review, hard_alignment_failure = alignment_gate(records)
    if alignment_needs_review:
        review_reasons.append("alignment_failed")

    filled = sorted(filled, key=record_score, reverse=True)
    selected_record = None if hard_alignment_failure or not filled else filled[0]
    selected = str(selected_record["choice"]) if selected_record is not None else None

    selected_score = record_score(selected_record) if selected_record is not None else None
    other_scores = [
        record_score(record)
        for record in records
        if selected is None or str(record["choice"]) != selected
    ]
    score_margin = None
    if selected_score is not None and other_scores:
        score_margin = round(selected_score - max(other_scores), 6)
    confidence = decode_confidence(
        selected_record,
        selected_score=selected_score,
        score_margin=score_margin,
        records=records,
    )
    if selected is not None and confidence is not None and confidence < 0.45:
        review_reasons.append("low_confidence")

    if review_reasons:
        status = "multi_mark" if "multi_mark" in review_reasons else "need_review"
    elif selected is not None:
        status = "accepted"
    else:
        status = "blank"

    selected_darkness_score = float(selected_record["darkness_score"]) if selected_record is not None else None
    other_darkness_scores = [
        float(record["darkness_score"])
        for record in records
        if selected is None or str(record["choice"]) != selected
    ]
    darkness_margin = None
    if selected_darkness_score is not None and other_darkness_scores:
        darkness_margin = round(selected_darkness_score - max(other_darkness_scores), 6)

    return {
        "selected": selected,
        "status": status,
        "need_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "confidence": confidence,
        "score": round(selected_score, 6) if selected_score is not None else None,
        "score_margin": score_margin,
        "darkness_score": round(selected_darkness_score, 6) if selected_darkness_score is not None else None,
        "darkness_margin": darkness_margin,
        "states": states,
    }


def state_from_record(record: dict) -> dict:
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


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def record_score(record: dict | None) -> float:
    if record is None:
        return 0.0
    score_key = record.get("group_score_key")
    if score_key and record.get(str(score_key)) is not None:
        return float(record[str(score_key)])
    if record.get("filled_probability") is not None:
        return float(record["filled_probability"])
    if record.get("fill_score") is not None:
        return float(record["fill_score"])
    return float(record.get("darkness_score") or 0.0)


def _score_confidence(
    selected_record: dict,
    *,
    selected_score: float,
    score_margin: float | None,
) -> float:
    filled_threshold = _as_float(selected_record.get("group_filled_threshold"), 0.08) or 0.08
    margin_threshold = _as_float(selected_record.get("group_margin_threshold"), 0.025) or 0.025
    if score_margin is not None:
        return _clamp(score_margin / max(margin_threshold * 2.0, 1e-6))
    return _clamp(selected_score / max(filled_threshold * 2.0, 1e-6))


def _metadata_confidence(record: dict) -> float:
    values = []
    for key in (
        "alignment_confidence",
        "grid_refinement_confidence",
        "model_confidence",
        "model_argmax_confidence",
    ):
        value = _as_float(record.get(key))
        if value is not None:
            values.append(_clamp(value))
    return min(values) if values else 1.0


def _blank_confidence(records: list[dict]) -> float | None:
    if not records:
        return None
    top_score = max(record_score(record) for record in records)
    filled_threshold = _as_float(records[0].get("group_filled_threshold"), 0.08) or 0.08
    return round(_clamp(1.0 - (top_score / max(filled_threshold, 1e-6))), 6)


def decode_confidence(
    selected_record: dict | None,
    *,
    selected_score: float | None,
    score_margin: float | None,
    records: list[dict],
) -> float | None:
    if selected_record is None or selected_score is None:
        return _blank_confidence(records)
    score_confidence = _score_confidence(
        selected_record,
        selected_score=selected_score,
        score_margin=score_margin,
    )
    confidence = min(score_confidence, _metadata_confidence(selected_record))
    return round(confidence, 6)


def alignment_gate(records: Iterable[dict]) -> tuple[bool, bool]:
    needs_review = False
    hard_failure = False
    for record in records:
        alignment_status = record.get("alignment_status")
        grid_status = record.get("grid_refinement_status")
        if grid_status is not None:
            if grid_status != "ok":
                needs_review = True
            if grid_status == "alignment_failed" or record.get("grid_refinement_decode_allowed") is False:
                hard_failure = True
            continue

        if alignment_status not in {None, "ok"}:
            needs_review = True
        if alignment_status == "alignment_failed":
            hard_failure = True
    return needs_review, hard_failure
